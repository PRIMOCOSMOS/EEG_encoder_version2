"""Training loop for SEED-VII dual-head model — OOM-safe 流式版.

核心改造：
- 不再一次性 np.concatenate 全部 X 到 RAM (13.6 GB → OOM Killed)
- Pass 1: scan_npz_metadata 只加载 y/s/meta (< 50 MB)
- Pass 2: MmapXStore 将每个 npz 的 X 解压为 .npy → memmap (内存 ≈ 0)
- EEGMmapDataset.__getitem__ 从 memmap 按需读一条，OS 页面缓存管理

两种训练入口：
- run_training(cfg)             : 全被试混合训练，一个模型（原始行为）
- run_training_per_subject(cfg) : 每个被试独立训练，输出 N 个模型（被试内泛化）

样本均衡：
- 训练集使用 WeightedRandomSampler，按各类样本数倒数加权
- 验证集、测试集保持原始分布（不均衡），反映真实场景
"""
from __future__ import annotations

import gc, json, logging, math, os, random, time
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import CONFORMER_CONFIG, EEGNET_CONFIG, TRAIN_DEFAULTS
from .dataset import (
    EEGMmapDataset,
    MmapXStore,
    filter_by_subjects,
    scan_npz_metadata,
    split_trials_from_meta,
)
from .losses import LossConfig, WeightedDualLoss
from .model import build_model, count_parameters, freeze_intensity_head


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    seed: int                   = int(TRAIN_DEFAULTS["seed"])
    batch_size: int             = int(TRAIN_DEFAULTS["batch_size"])
    num_workers: int            = int(TRAIN_DEFAULTS["num_workers"])
    lr: float                   = float(TRAIN_DEFAULTS["lr"])
    min_lr: float               = float(TRAIN_DEFAULTS["min_lr"])
    beta1: float                = float(TRAIN_DEFAULTS["beta1"])
    beta2: float                = float(TRAIN_DEFAULTS["beta2"])
    weight_decay: float         = float(TRAIN_DEFAULTS["weight_decay"])
    grad_clip: float            = float(TRAIN_DEFAULTS["grad_clip"])
    pretrain_epochs: int        = int(TRAIN_DEFAULTS["pretrain_epochs"])
    max_epochs: int             = int(TRAIN_DEFAULTS["max_epochs"])
    patience: int               = int(TRAIN_DEFAULTS["patience"])
    alpha_cls: float            = float(TRAIN_DEFAULTS["alpha_cls_start"])
    beta_reg: float             = float(TRAIN_DEFAULTS["beta_reg_start"])
    gamma_rank_start: float     = float(TRAIN_DEFAULTS["gamma_rank_start"])
    gamma_rank_end: float       = float(TRAIN_DEFAULTS["gamma_rank_end"])
    rank_warmup_epochs: int     = int(TRAIN_DEFAULTS["rank_warmup_epochs"])
    enable_rank: bool           = bool(TRAIN_DEFAULTS["enable_rank"])
    rank_margin: float          = float(TRAIN_DEFAULTS["rank_margin"])
    label_smoothing: float      = float(TRAIN_DEFAULTS["label_smoothing"])
    sample_weight_mode: str     = str(TRAIN_DEFAULTS["sample_weight_mode"])
    intensity_threshold: float  = float(TRAIN_DEFAULTS["intensity_threshold"])
    weak_sample_weight: float   = float(TRAIN_DEFAULTS["weak_sample_weight"])
    device: str                 = str(TRAIN_DEFAULTS["device"])
    amp: bool                   = bool(TRAIN_DEFAULTS["amp"])
    save_last: bool             = bool(TRAIN_DEFAULTS["save_last"])
    save_features: bool         = bool(TRAIN_DEFAULTS["save_features"])
    feature_type: str           = str(TRAIN_DEFAULTS["feature_type"])
    resume: bool                = False
    resume_path: str            = ""
    save_interval: int          = int(TRAIN_DEFAULTS["save_interval"])
    max_runtime_hours: float    = float(TRAIN_DEFAULTS["max_runtime_hours"])
    pin_memory: bool            = False
    persistent_workers: bool    = False
    freeze_intensity_head: bool = False
    train_subjects: str         = ""
    val_subjects: str           = ""
    test_subjects: str          = ""
    model_type: str             = str(TRAIN_DEFAULTS["model_type"])
    val_ratio: float            = float(TRAIN_DEFAULTS["val_ratio"])
    test_ratio: float           = float(TRAIN_DEFAULTS["test_ratio"])
    mmap_cache_dir: str         = ""
    # ★ 样本均衡开关：True = 训练集用 WeightedRandomSampler 均衡各类
    balance_train: bool         = bool(TRAIN_DEFAULTS.get("balance_train", True))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_device(d: str):
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "cuda" and not torch.cuda.is_available():
        print("[WARN] No CUDA, fallback CPU.")
        return torch.device("cpu")
    return torch.device(d)

def optimizer_to(opt, device):
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)

def setup_logger(od: Path, name: str = "seed_vii_trainer"):
    od.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(od / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh)
    return lg

def cosine_lr(ep, tot, base, mn):
    if tot <= 1:
        return base
    return mn + (base - mn) * 0.5 * (1 + math.cos(math.pi * ep / max(1, tot - 1)))

def gamma_schedule(ep, cfg, started):
    if not cfg.enable_rank:
        return 0.0
    if cfg.rank_warmup_epochs <= 0:
        return cfg.gamma_rank_end
    p = min(1.0, max(0.0, max(0, ep - started) / float(cfg.rank_warmup_epochs)))
    return cfg.gamma_rank_start + (cfg.gamma_rank_end - cfg.gamma_rank_start) * p

def _parse_subjects(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


# --------------------------------------------------------------------------
# ★ 训练集均衡采样器
# --------------------------------------------------------------------------

def make_balanced_sampler(y: np.ndarray, indices: np.ndarray) -> WeightedRandomSampler:
    """对训练集按类别频率倒数加权，使每个 epoch 中各类期望样本数相等。

    Args:
        y:       全局标签数组 (N_total,)
        indices: 训练集在 y 中的索引

    Returns:
        WeightedRandomSampler，replacement=True，num_samples = len(indices)
    """
    labels = y[indices]                          # 训练集标签
    classes, counts = np.unique(labels, return_counts=True)
    # 每个类的权重 = 1 / count，归一化使总权重 = n_classes
    class_weight = {c: 1.0 / cnt for c, cnt in zip(classes.tolist(), counts.tolist())}
    sample_weights = np.array([class_weight[int(lbl)] for lbl in labels], dtype=np.float64)
    sample_weights /= sample_weights.sum()       # 归一化为概率分布

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(indices),                # 每 epoch 采样数与原训练集相同
        replacement=True,                        # 有放回采样，少数类可被重复采到
    )


def make_balanced_loader(ds, y_all: np.ndarray, train_idx: np.ndarray,
                         bs: int, nw: int, pin: bool = False) -> DataLoader:
    """构造均衡训练 DataLoader（用 sampler 替代 shuffle）。"""
    sampler = make_balanced_sampler(y_all, train_idx)
    return DataLoader(
        ds, batch_size=bs, sampler=sampler,
        num_workers=nw, pin_memory=pin,
        persistent_workers=False, drop_last=False,
    )


# --------------------------------------------------------------------------
# Loss / eval / train
# --------------------------------------------------------------------------

def build_loss(cfg, gamma, enable_rank):
    return WeightedDualLoss(LossConfig(
        alpha=cfg.alpha_cls, beta=cfg.beta_reg, gamma=gamma,
        label_smoothing=cfg.label_smoothing, rank_margin=cfg.rank_margin,
        enable_rank=enable_rank, sample_weight_mode=cfg.sample_weight_mode,
        intensity_threshold=cfg.intensity_threshold,
        weak_sample_weight=cfg.weak_sample_weight,
    ))

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    losses, cls_l, reg_l = [], [], []
    correct = total = 0
    abs_err = 0.0
    for xb, yb, sb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        sb = sb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, ps, _ = model(xb)
            _, parts = criterion(logits, ps, yb, sb)
        losses.append(parts["loss"]); cls_l.append(parts["cls"]); reg_l.append(parts["reg"])
        correct += (logits.argmax(1) == yb).sum().item()
        total   += yb.numel()
        abs_err += (ps - sb).abs().sum().item()
    n = max(1, total)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "cls":  float(np.mean(cls_l))  if cls_l  else float("nan"),
        "reg":  float(np.mean(reg_l))  if reg_l  else float("nan"),
        "acc":  correct / n,
        "intensity_mae": abs_err / n,
    }

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    grad_clip, deadline_ts=None):
    model.train()
    losses, cls_l, reg_l, rank_l = [], [], [], []
    correct = total = 0
    hit_deadline = False
    for xb, yb, sb in loader:
        if deadline_ts and time.time() >= deadline_ts:
            hit_deadline = True
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        sb = sb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, ps, _ = model(xb)
            loss, parts = criterion(logits, ps, yb, sb)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(parts["loss"]); cls_l.append(parts["cls"])
        reg_l.append(parts["reg"]); rank_l.append(parts["rank"])
        correct += (logits.argmax(1) == yb).sum().item()
        total   += yb.numel()
    n = max(1, total)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "cls":  float(np.mean(cls_l))  if cls_l  else float("nan"),
        "reg":  float(np.mean(reg_l))  if reg_l  else float("nan"),
        "rank": float(np.mean(rank_l)) if rank_l else 0.0,
        "acc":  correct / n,
    }, hit_deadline

# --------------------------------------------------------------------------
# Checkpoint I/O
# --------------------------------------------------------------------------

def save_state(path, epoch, model, optimizer, scaler, bv, be, bad, rse, ti, vi, tsi, cfg):
    torch.save({
        "epoch": int(epoch), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "best_val_acc": float(bv), "best_epoch": int(be), "bad_epochs": int(bad),
        "rank_started_at_epoch": int(rse),
        "train_idx": ti, "val_idx": vi, "test_idx": tsi,
        "config": asdict(cfg) | {
            "data_dir": str(cfg.data_dir), "output_dir": str(cfg.output_dir)
        },
    }, path)

def save_encoder(path, model, config_dict):
    torch.save({"model": model.state_dict(), "config": config_dict}, path)

def make_loader(ds, bs, shuffle, nw, pin=False, pw=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      pin_memory=pin, persistent_workers=pw if nw > 0 else False,
                      drop_last=False)

# --------------------------------------------------------------------------
# 内部核心：跑完整训练循环
# --------------------------------------------------------------------------

def _run_single_training(
    cfg: TrainConfig,
    x_store: MmapXStore,
    file_map: np.ndarray,
    y_all: np.ndarray,
    s_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
    logger,
    active_config: dict,
    model_label: str = "",
) -> Dict:
    device = resolve_device(cfg.device)
    use_amp = bool(cfg.amp and device.type == "cuda")

    model = build_model(cfg.model_type,
                        EEGNET_CONFIG if cfg.model_type == "eegnet" else CONFORMER_CONFIG)
    model = model.to(device)
    if cfg.freeze_intensity_head:
        freeze_intensity_head(model)
    n_params = count_parameters(model)
    logger.info(f"{model_label}[MODEL] {cfg.model_type}, params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 betas=(cfg.beta1, cfg.beta2),
                                 weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Dataset
    train_ds = EEGMmapDataset(x_store, file_map, y_all, s_all, indices=train_idx)
    val_ds   = EEGMmapDataset(x_store, file_map, y_all, s_all, indices=val_idx)
    test_ds  = EEGMmapDataset(x_store, file_map, y_all, s_all, indices=test_idx) \
               if len(test_idx) > 0 else None

    nw = min(cfg.num_workers, 2)

    # ★ 训练集：均衡采样 vs 普通 shuffle
    if cfg.balance_train:
        train_loader = make_balanced_loader(
            train_ds, y_all, train_idx, cfg.batch_size, nw, pin=cfg.pin_memory)
        # 统计均衡前后分布，写入日志
        train_labels = y_all[train_idx]
        classes, counts = np.unique(train_labels, return_counts=True)
        dist_str = ", ".join(f"cls{c}:{cnt}" for c, cnt in zip(classes, counts))
        logger.info(f"{model_label}[BALANCE] Train dist (原始): {dist_str} "
                    f"→ WeightedRandomSampler 均衡各类至等频")
    else:
        train_loader = make_loader(train_ds, cfg.batch_size, True, nw, pin=cfg.pin_memory)

    # 验证集、测试集：保持原始分布（不均衡），反映真实场景
    val_loader  = make_loader(val_ds,  cfg.batch_size, False, nw, pin=cfg.pin_memory)
    test_loader = make_loader(test_ds, cfg.batch_size, False, nw, pin=cfg.pin_memory) \
                  if test_ds else None

    logger.info(f"{model_label}[LOADER] train={len(train_ds)}, val={len(val_ds)}, "
                f"test={len(test_ds) if test_ds else 0}, "
                f"batch={cfg.batch_size}, balance={cfg.balance_train}")

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path        = output_dir / "train_state.pt"
    best_model_path   = output_dir / "best_model.pt"
    best_encoder_path = output_dir / "best_encoder.pt"

    best_val_acc, best_epoch, bad_epochs = -1.0, -1, 0
    start_epoch = 1
    rse = cfg.pretrain_epochs + 1

    if cfg.resume and state_path.exists():
        rs = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(rs["model"])
        optimizer.load_state_dict(rs["optimizer"])
        optimizer_to(optimizer, device)
        if "scaler" in rs and use_amp:
            try:
                scaler.load_state_dict(rs["scaler"])
            except Exception:
                pass
        best_val_acc = float(rs.get("best_val_acc", -1.0))
        best_epoch   = int(rs.get("best_epoch", -1))
        bad_epochs   = int(rs.get("bad_epochs", 0))
        rse          = int(rs.get("rank_started_at_epoch", rse))
        start_epoch  = int(rs.get("epoch", 0)) + 1
        del rs; gc.collect()
        logger.info(f"{model_label}[RESUME] ep={start_epoch}, best={best_val_acc:.4f}@{best_epoch}")

    t0 = time.time()
    deadline_ts = t0 + cfg.max_runtime_hours * 3600 if cfg.max_runtime_hours > 0 else None

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        last_epoch = epoch
        is_pre = epoch <= cfg.pretrain_epochs

        if is_pre:
            criterion = WeightedDualLoss(LossConfig(
                alpha=1.0, beta=0.0, gamma=0.0,
                label_smoothing=cfg.label_smoothing, enable_rank=False,
                sample_weight_mode=cfg.sample_weight_mode,
                intensity_threshold=cfg.intensity_threshold,
                weak_sample_weight=cfg.weak_sample_weight,
            ))
        else:
            gamma = gamma_schedule(epoch, cfg, rse)
            if cfg.freeze_intensity_head:
                criterion = WeightedDualLoss(LossConfig(
                    alpha=cfg.alpha_cls, beta=0.0, gamma=gamma,
                    label_smoothing=cfg.label_smoothing,
                    enable_rank=cfg.enable_rank and gamma > 0,
                    rank_margin=cfg.rank_margin,
                    sample_weight_mode=cfg.sample_weight_mode,
                    intensity_threshold=cfg.intensity_threshold,
                    weak_sample_weight=cfg.weak_sample_weight,
                ))
            else:
                criterion = build_loss(cfg, gamma, cfg.enable_rank and gamma > 0)

        lr_now = cosine_lr(epoch - 1, cfg.max_epochs, cfg.lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        tr, dh = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                 device, use_amp, cfg.grad_clip, deadline_ts)
        va = evaluate(model, val_loader, criterion, device, use_amp)
        phase = "PRE" if is_pre else ("CLS" if cfg.freeze_intensity_head else "JOINT")
        logger.info(f"{model_label}[E{epoch:03d}|{phase}] lr={lr_now:.2e} "
                    f"tr loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
                    f"va loss={va['loss']:.4f} acc={va['acc']:.4f} mae={va['intensity_mae']:.4f}")

        if va["acc"] > best_val_acc:
            best_val_acc, best_epoch, bad_epochs = va["acc"], epoch, 0
            torch.save({"model": model.state_dict(), "config": active_config, "val": va},
                       best_model_path)
            save_encoder(best_encoder_path, model, active_config)
        else:
            bad_epochs += 1

        if epoch % cfg.save_interval == 0 or va["acc"] >= best_val_acc or dh:
            save_state(state_path, epoch, model, optimizer, scaler, best_val_acc,
                       best_epoch, bad_epochs, rse, train_idx, val_idx, test_idx, cfg)
        if dh:
            logger.warning(f"{model_label}[TIMEUP] @{epoch}")
            break
        if bad_epochs >= cfg.patience:
            logger.info(f"{model_label}[EARLY-STOP] @{epoch}, best={best_val_acc:.4f}@{best_epoch}")
            break

    if cfg.save_last:
        torch.save({"model": model.state_dict(), "config": active_config, "epoch": last_epoch},
                   output_dir / "last_model.pt")

    test_metrics = {}
    if best_model_path.exists() and test_loader:
        ck = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); del ck; gc.collect()
        test_metrics = evaluate(model, test_loader,
                                build_loss(cfg, 0.0, False), device, use_amp)
        logger.info(f"{model_label}[TEST] acc={test_metrics['acc']:.4f} "
                    f"mae={test_metrics['intensity_mae']:.4f}")

    del model, optimizer, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_val_acc":    best_val_acc,
        "best_epoch":      best_epoch,
        "test":            test_metrics,
        "n_params":        int(n_params),
        "epochs_run":      int(last_epoch),
        "elapsed_seconds": float(time.time() - t0),
        "model_type":      cfg.model_type,
        "balance_train":   cfg.balance_train,
    }


# --------------------------------------------------------------------------
# 入口 1：全被试混合训练
# --------------------------------------------------------------------------

def run_training(cfg: TrainConfig) -> Dict:
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    logger = setup_logger(cfg.output_dir)

    if cfg.model_type == "eegnet":
        active_config = EEGNET_CONFIG
    elif cfg.model_type == "conformer":
        active_config = CONFORMER_CONFIG
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    train_subj = _parse_subjects(cfg.train_subjects)
    val_subj   = _parse_subjects(cfg.val_subjects)
    test_subj  = _parse_subjects(cfg.test_subjects)
    all_subjects = list(set(train_subj + val_subj + test_subj)) \
                   if (train_subj or val_subj or test_subj) else None

    npz_paths, y_all, s_all, meta, file_map = scan_npz_metadata(
        str(cfg.data_dir), subjects=all_subjects)
    n_total = len(y_all)
    logger.info(f"Data: N={n_total} windows across {len(npz_paths)} files")

    mmap_dir = cfg.mmap_cache_dir or str(cfg.output_dir / "_mmap_cache")
    x_store = MmapXStore(npz_paths, cache_dir=mmap_dir)

    use_subj_filter = bool(train_subj or val_subj or test_subj)
    if use_subj_filter:
        all_subj_in_data = sorted(set(str(m.get("subject", "")) for m in meta))
        if train_subj:
            train_idx = filter_by_subjects(meta, train_subj)
        else:
            excl = set(val_subj) | set(test_subj)
            train_idx = filter_by_subjects(
                meta, [s for s in all_subj_in_data if s not in excl])
        if val_subj:
            val_idx = filter_by_subjects(meta, val_subj)
        else:
            rng = np.random.default_rng(cfg.seed)
            pool = np.setdiff1d(np.arange(n_total), train_idx)
            n_val = max(1, int(round(len(pool) * cfg.val_ratio)))
            val_idx = np.sort(rng.choice(pool, size=min(n_val, len(pool)), replace=False))
        if test_subj:
            test_idx = filter_by_subjects(meta, test_subj)
        else:
            used = set(train_idx.tolist()) | set(val_idx.tolist())
            pool = np.array([i for i in range(n_total) if i not in used])
            if len(pool) > 0:
                rng = np.random.default_rng(cfg.seed + 1)
                n_test = max(1, int(round(len(pool) * 0.5)))
                test_idx = np.sort(rng.choice(pool, size=min(n_test, len(pool)), replace=False))
            else:
                test_idx = np.array([], dtype=np.int64)
        logger.info(f"[SUBJECT-FILTER] train={len(train_idx)}, val={len(val_idx)}, "
                    f"test={len(test_idx)}")
    else:
        splits = split_trials_from_meta(
            meta, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, seed=cfg.seed)
        train_idx = splits["train"]
        val_idx   = splits["val"]
        test_idx  = splits["test"]
        logger.info(f"[TRIAL-SPLIT] train={len(train_idx)}, val={len(val_idx)}, "
                    f"test={len(test_idx)}")

    if len(val_idx) == 0:
        logger.error("Empty val set."); x_store.close(); return {"error": "empty val"}
    if len(train_idx) == 0:
        logger.error("Empty train set."); x_store.close(); return {"error": "empty train"}

    with open(cfg.output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg) | {"data_dir": str(cfg.data_dir),
                                 "output_dir": str(cfg.output_dir)},
                  f, indent=2, ensure_ascii=False)

    summary = _run_single_training(
        cfg, x_store, file_map, y_all, s_all,
        train_idx, val_idx, test_idx,
        output_dir=cfg.output_dir,
        logger=logger,
        active_config=active_config,
        model_label="",
    )
    summary["mode"] = "all_subjects"

    with open(cfg.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary -> {cfg.output_dir / 'summary.json'}")

    x_store.close()
    return summary


# --------------------------------------------------------------------------
# 入口 2：单被试独立训练
# --------------------------------------------------------------------------

def run_training_per_subject(cfg: TrainConfig) -> Dict:
    """为每个被试独立训练一个模型（被试内 trial-level 分割）。"""
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    global_logger = setup_logger(cfg.output_dir, name="per_subject_global")

    if cfg.model_type == "eegnet":
        active_config = EEGNET_CONFIG
    elif cfg.model_type == "conformer":
        active_config = CONFORMER_CONFIG
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    npz_dir = Path(cfg.data_dir)
    all_npz = sorted(npz_dir.glob("*.npz"))
    if not all_npz:
        raise FileNotFoundError(f"No .npz files found in {cfg.data_dir}")

    subject_filter = _parse_subjects(cfg.train_subjects)
    if subject_filter:
        all_npz = [p for p in all_npz if p.stem in subject_filter]
        global_logger.info(f"[PER-SUBJ] Subject filter: {subject_filter}")

    global_logger.info(f"[PER-SUBJ] Will train {len(all_npz)} subject model(s): "
                       f"{[p.stem for p in all_npz]}")

    all_summaries: Dict[str, Dict] = {}
    t_total = time.time()
    deadline_ts = (t_total + cfg.max_runtime_hours * 3600
                   if cfg.max_runtime_hours > 0 else None)

    for subj_npz in all_npz:
        subj = subj_npz.stem
        label = f"[Subject {subj}] "
        subj_out = cfg.output_dir / f"subject_{subj}"
        subj_out.mkdir(parents=True, exist_ok=True)

        if deadline_ts and time.time() >= deadline_ts:
            global_logger.warning(f"[PER-SUBJ] Time limit reached before subject {subj}.")
            break

        global_logger.info(f"\n{'='*60}")
        global_logger.info(f"[PER-SUBJ] === Subject {subj} → {subj_out} ===")
        global_logger.info(f"{'='*60}")

        subj_logger = setup_logger(subj_out, name=f"trainer_subj_{subj}")

        try:
            npz_paths, y_all, s_all, meta, file_map = scan_npz_metadata(
                str(npz_dir), subjects=[subj])
        except FileNotFoundError as e:
            global_logger.error(f"{label}npz not found: {e}")
            continue

        splits = split_trials_from_meta(
            meta, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, seed=cfg.seed)
        train_idx = splits["train"]
        val_idx   = splits["val"]
        test_idx  = splits["test"]
        subj_logger.info(f"{label}Split: train={len(train_idx)}, "
                         f"val={len(val_idx)}, test={len(test_idx)} windows")

        if len(train_idx) == 0 or len(val_idx) == 0:
            global_logger.warning(f"{label}Skipped: train or val empty.")
            continue

        mmap_dir = cfg.mmap_cache_dir or str(cfg.output_dir / "_mmap_cache")
        x_store = MmapXStore(npz_paths, cache_dir=mmap_dir)

        subj_cfg_dict = asdict(cfg) | {
            "data_dir": str(cfg.data_dir),
            "output_dir": str(subj_out),
            "subject": subj,
        }
        with open(subj_out / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(subj_cfg_dict, f, indent=2, ensure_ascii=False)

        remaining_hours = (
            (deadline_ts - time.time()) / 3600.0
            if deadline_ts else cfg.max_runtime_hours
        )
        subj_cfg = copy.copy(cfg)
        subj_cfg.output_dir        = subj_out
        subj_cfg.max_runtime_hours = max(0.0, remaining_hours)
        subj_cfg.resume            = False

        summary = _run_single_training(
            subj_cfg, x_store, file_map, y_all, s_all,
            train_idx, val_idx, test_idx,
            output_dir=subj_out,
            logger=subj_logger,
            active_config=active_config,
            model_label=label,
        )
        summary["subject"] = subj

        with open(subj_out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        all_summaries[subj] = summary
        global_logger.info(
            f"{label}Done. best_val_acc={summary['best_val_acc']:.4f}, "
            f"test_acc={summary.get('test', {}).get('acc', float('nan')):.4f}, "
            f"test_mae={summary.get('test', {}).get('intensity_mae', float('nan')):.4f}"
        )

        x_store.close()
        gc.collect()

    if all_summaries:
        test_accs = [v["test"]["acc"]
                     for v in all_summaries.values()
                     if v.get("test") and "acc" in v["test"]]
        test_maes = [v["test"]["intensity_mae"]
                     for v in all_summaries.values()
                     if v.get("test") and "intensity_mae" in v["test"]]
        val_accs  = [v["best_val_acc"] for v in all_summaries.values()]

        aggregate = {
            "mode":            "per_subject",
            "n_subjects":      len(all_summaries),
            "mean_val_acc":    float(np.mean(val_accs)),
            "std_val_acc":     float(np.std(val_accs)),
            "mean_test_acc":   float(np.mean(test_accs))  if test_accs else None,
            "std_test_acc":    float(np.std(test_accs))   if test_accs else None,
            "mean_test_mae":   float(np.mean(test_maes))  if test_maes else None,
            "std_test_mae":    float(np.std(test_maes))   if test_maes else None,
            "total_elapsed_s": float(time.time() - t_total),
            "per_subject":     all_summaries,
        }

        out_path = cfg.output_dir / "all_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)

        global_logger.info(f"\n{'='*60}")
        global_logger.info(f"[PER-SUBJ] ALL DONE — {len(all_summaries)} subjects")
        if test_accs:
            global_logger.info(
                f"[PER-SUBJ] Test acc: mean={np.mean(test_accs):.4f} "
                f"± {np.std(test_accs):.4f} "
                f"(min={min(test_accs):.4f}, max={max(test_accs):.4f})"
            )
        if test_maes:
            global_logger.info(
                f"[PER-SUBJ] Test MAE: mean={np.mean(test_maes):.4f} "
                f"± {np.std(test_maes):.4f}"
            )
        global_logger.info(f"[PER-SUBJ] → {out_path}")
        global_logger.info(f"{'='*60}")

        return aggregate

    return {"error": "no subjects trained"}

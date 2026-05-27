"""Training loop for SEED-VII dual-head model (EEGNet or EEGConformer).

重构版：
- 从 per-subject .npz 目录加载数据
- Trial-level 划分避免数据泄漏
- 支持被试筛选（跨被试/被试内训练）
- 断点续训 / 软超时 / 周期保存
"""
from __future__ import annotations

import gc, json, logging, math, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CONFORMER_CONFIG, EEGNET_CONFIG, TRAIN_DEFAULTS
from .dataset import (
    EEGWindowArrayDataset,
    filter_by_subjects,
    load_multi_subject_npz,
    split_trials_from_meta,
)
from .losses import LossConfig, WeightedDualLoss
from .model import build_model, count_parameters, freeze_intensity_head


@dataclass
class TrainConfig:
    data_dir: Path              # 存放 per-subject .npz 文件的目录
    output_dir: Path
    seed: int = int(TRAIN_DEFAULTS["seed"])
    batch_size: int = int(TRAIN_DEFAULTS["batch_size"])
    num_workers: int = int(TRAIN_DEFAULTS["num_workers"])
    lr: float = float(TRAIN_DEFAULTS["lr"])
    min_lr: float = float(TRAIN_DEFAULTS["min_lr"])
    beta1: float = float(TRAIN_DEFAULTS["beta1"])
    beta2: float = float(TRAIN_DEFAULTS["beta2"])
    weight_decay: float = float(TRAIN_DEFAULTS["weight_decay"])
    grad_clip: float = float(TRAIN_DEFAULTS["grad_clip"])
    pretrain_epochs: int = int(TRAIN_DEFAULTS["pretrain_epochs"])
    max_epochs: int = int(TRAIN_DEFAULTS["max_epochs"])
    patience: int = int(TRAIN_DEFAULTS["patience"])
    alpha_cls: float = float(TRAIN_DEFAULTS["alpha_cls_start"])
    beta_reg: float = float(TRAIN_DEFAULTS["beta_reg_start"])
    gamma_rank_start: float = float(TRAIN_DEFAULTS["gamma_rank_start"])
    gamma_rank_end: float = float(TRAIN_DEFAULTS["gamma_rank_end"])
    rank_warmup_epochs: int = int(TRAIN_DEFAULTS["rank_warmup_epochs"])
    enable_rank: bool = bool(TRAIN_DEFAULTS["enable_rank"])
    rank_margin: float = float(TRAIN_DEFAULTS["rank_margin"])
    label_smoothing: float = float(TRAIN_DEFAULTS["label_smoothing"])
    sample_weight_mode: str = str(TRAIN_DEFAULTS["sample_weight_mode"])
    intensity_threshold: float = float(TRAIN_DEFAULTS["intensity_threshold"])
    weak_sample_weight: float = float(TRAIN_DEFAULTS["weak_sample_weight"])
    device: str = str(TRAIN_DEFAULTS["device"])
    amp: bool = bool(TRAIN_DEFAULTS["amp"])
    save_last: bool = bool(TRAIN_DEFAULTS["save_last"])
    save_features: bool = bool(TRAIN_DEFAULTS["save_features"])
    feature_type: str = str(TRAIN_DEFAULTS["feature_type"])
    resume: bool = False
    resume_path: str = ""
    save_interval: int = int(TRAIN_DEFAULTS["save_interval"])
    max_runtime_hours: float = float(TRAIN_DEFAULTS["max_runtime_hours"])
    pin_memory: bool = False
    persistent_workers: bool = False
    freeze_intensity_head: bool = False
    train_subjects: str = ""
    val_subjects: str = ""
    test_subjects: str = ""
    model_type: str = str(TRAIN_DEFAULTS["model_type"])
    val_ratio: float = float(TRAIN_DEFAULTS["val_ratio"])
    test_ratio: float = float(TRAIN_DEFAULTS["test_ratio"])


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

def setup_logger(od: Path):
    od.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("seed_vii_trainer")
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
        total += yb.numel()
        abs_err += (ps - sb).abs().sum().item()
    n = max(1, total)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "cls": float(np.mean(cls_l)) if cls_l else float("nan"),
        "reg": float(np.mean(reg_l)) if reg_l else float("nan"),
        "acc": correct / n,
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
        total += yb.numel()
    n = max(1, total)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "cls": float(np.mean(cls_l)) if cls_l else float("nan"),
        "reg": float(np.mean(reg_l)) if reg_l else float("nan"),
        "rank": float(np.mean(rank_l)) if rank_l else 0.0,
        "acc": correct / n,
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
# MAIN TRAINING ORCHESTRATOR
# --------------------------------------------------------------------------

def run_training(cfg: TrainConfig) -> Dict[str, object]:
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    logger = setup_logger(cfg.output_dir)
    device = resolve_device(cfg.device)
    use_amp = bool(cfg.amp and device.type == "cuda")

    # ================================================================
    # STEP 0: 构建模型
    # ================================================================
    if cfg.model_type == "eegnet":
        model = build_model("eegnet", EEGNET_CONFIG)
        active_config = EEGNET_CONFIG
        logger.info(f"[MODEL] EEGNet (F1={EEGNET_CONFIG['F1']}, D={EEGNET_CONFIG['D']}, "
                     f"kernLength={EEGNET_CONFIG['kernLength']})")
    elif cfg.model_type == "conformer":
        model = build_model("conformer", CONFORMER_CONFIG)
        active_config = CONFORMER_CONFIG
        logger.info(f"[MODEL] EEGConformer (embed={CONFORMER_CONFIG['embed_dim']})")
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    model = model.to(device)

    if cfg.freeze_intensity_head:
        nf = freeze_intensity_head(model)
        n_params = count_parameters(model)
        logger.info(f"[FREEZE] {nf:,} frozen, {n_params:,} trainable")
    else:
        n_params = count_parameters(model)
    logger.info(f"[MODEL] Params={n_params:,} ({n_params/1e6:.4f}M)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 betas=(cfg.beta1, cfg.beta2),
                                 weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ================================================================
    # STEP 1: 加载数据（per-subject npz 目录）
    # ================================================================
    train_subj = _parse_subjects(cfg.train_subjects)
    val_subj = _parse_subjects(cfg.val_subjects)
    test_subj = _parse_subjects(cfg.test_subjects)

    # 确定加载哪些被试
    all_subjects = None
    if train_subj or val_subj or test_subj:
        all_subjects = list(set(train_subj + val_subj + test_subj))
    X_full, y_full, s_full, meta = load_multi_subject_npz(
        str(cfg.data_dir), subjects=all_subjects)
    n_total = len(y_full)
    logger.info(f"Data: X shape={X_full.shape}, N={n_total}")

    # ================================================================
    # STEP 2: 确定 indices
    # ================================================================
    use_subj_filter = bool(train_subj or val_subj or test_subj)

    if use_subj_filter:
        # 按被试划分
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
        logger.info(f"[SUBJECT-FILTER] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        # Trial-level 划分
        splits = split_trials_from_meta(meta, val_ratio=cfg.val_ratio,
                                        test_ratio=cfg.test_ratio, seed=cfg.seed)
        train_idx = splits["train"]
        val_idx = splits["val"]
        test_idx = splits["test"]
        logger.info(f"[TRIAL-SPLIT] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    if len(val_idx) == 0:
        logger.error("Empty val set.")
        return {"error": "empty val"}
    if len(train_idx) == 0:
        logger.error("Empty train set.")
        return {"error": "empty train"}

    # ================================================================
    # STEP 3: 创建 Dataset 和 DataLoader
    # ================================================================
    train_ds = EEGWindowArrayDataset(X_full, y_full, s_full, indices=train_idx)
    val_ds = EEGWindowArrayDataset(X_full, y_full, s_full, indices=val_idx)
    test_ds = EEGWindowArrayDataset(X_full, y_full, s_full, indices=test_idx) \
        if len(test_idx) > 0 else None

    nw = min(cfg.num_workers, 2)
    train_loader = make_loader(train_ds, cfg.batch_size, True, nw)
    val_loader = make_loader(val_ds, cfg.batch_size, False, nw)
    test_loader = make_loader(test_ds, cfg.batch_size, False, nw) if test_ds else None

    state_path = Path(cfg.resume_path) if cfg.resume_path else (cfg.output_dir / "train_state.pt")
    best_model_path = cfg.output_dir / "best_model.pt"
    best_encoder_path = cfg.output_dir / "best_encoder.pt"
    summary_path = cfg.output_dir / "summary.json"

    with open(cfg.output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg) | {
            "data_dir": str(cfg.data_dir), "output_dir": str(cfg.output_dir)
        }, f, indent=2, ensure_ascii=False)

    # Resume
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
        best_epoch = int(rs.get("best_epoch", -1))
        bad_epochs = int(rs.get("bad_epochs", 0))
        rse = int(rs.get("rank_started_at_epoch", rse))
        start_epoch = int(rs.get("epoch", 0)) + 1
        del rs; gc.collect()
        logger.info(f"[RESUME] ep {start_epoch} (best={best_val_acc:.4f} @{best_epoch})")

    t0 = time.time()
    deadline_ts = t0 + cfg.max_runtime_hours * 3600 if cfg.max_runtime_hours > 0 else None

    # ================================================================
    # STEP 4: 训练循环
    # ================================================================
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
        logger.info(f"[E{epoch:03d}|{phase}] lr={lr_now:.2e} "
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
            logger.warning(f"[TIMEUP] @{epoch}")
            break
        if bad_epochs >= cfg.patience:
            logger.info(f"[EARLY-STOP] @{epoch}, best={best_val_acc:.4f} @{best_epoch}")
            break

    if cfg.save_last:
        torch.save({"model": model.state_dict(), "config": active_config, "epoch": last_epoch},
                   cfg.output_dir / "last_model.pt")

    test_metrics = {}
    if best_model_path.exists() and test_loader:
        ck = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); del ck; gc.collect()
        test_metrics = evaluate(model, test_loader, build_loss(cfg, 0.0, False), device, use_amp)
        logger.info(f"[TEST] acc={test_metrics['acc']:.4f} mae={test_metrics['intensity_mae']:.4f}")

    summary = {
        "best_val_acc": best_val_acc, "best_epoch": best_epoch, "test": test_metrics,
        "n_params": int(n_params), "epochs_run": int(last_epoch),
        "elapsed_seconds": float(time.time() - t0),
        "model_type": cfg.model_type,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary -> {summary_path}")
    return summary

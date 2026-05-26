"""Training loop for SEED-VII dual-head EEG-Conformer.

特性（覆盖 Design.md 全部代码原则）：
- 两阶段训练：(1) 仅 L_cls 预训练 N epochs；(2) 联合训练 L_cls + L_reg (+L_rank)。
- 冻结 Intensity Head 选项（freeze_intensity_head），只训练分类分支。
- 单被试训练 + 跨被试验证（train_subjects / val_subjects / test_subjects）。
- 余弦退火 (lr_max -> 1e-5)。
- 周期断点 (`train_state.pt`) + `--resume`。
- 软超时 (`--max-runtime-hours`)。
- AMP 混合精度（CUDA）。
- 早停（基于验证集分类准确率）。

OOM 修复（v3）：
- 索引式 Dataset：不复制子数组，所有 split 共享一份底层 x/y/s（~2X → ~1X 内存）
- 关闭 npz 文件释放解压缓冲区
- DataLoader：persistent_workers=False / pin_memory=False（避免 COW + 锁页开销）
- 积极释放：创建 dataset 后 del 原始 numpy 数组 + gc.collect()
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CONFORMER_CONFIG, TRAIN_DEFAULTS
from .dataset import EEGWindowArrayDataset, filter_by_subjects
from .losses import LossConfig, WeightedDualLoss
from .model import EEGConformerDualHead, count_parameters, freeze_intensity_head

# ---------------------------------------------------------------------------
# Config & helpers
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_path: Path
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

    # loss
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

    # ---- DataLoader OOM 修复 ----
    pin_memory: bool = bool(TRAIN_DEFAULTS.get("pin_memory", False))
    persistent_workers: bool = bool(TRAIN_DEFAULTS.get("persistent_workers", False))

    # ---- 过拟合缓解 ----
    freeze_intensity_head: bool = bool(TRAIN_DEFAULTS.get("freeze_intensity_head", False))
    train_subjects: str = str(TRAIN_DEFAULTS.get("train_subjects", ""))
    val_subjects: str = str(TRAIN_DEFAULTS.get("val_subjects", ""))
    test_subjects: str = str(TRAIN_DEFAULTS.get("test_subjects", ""))

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable, fallback to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)

def optimizer_to(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("seed_vii_trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def cosine_lr(epoch: int, total_epochs: int, base_lr: float, min_lr: float) -> float:
    if total_epochs <= 1:
        return base_lr
    cos = 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, total_epochs - 1)))
    return min_lr + (base_lr - min_lr) * cos

def gamma_schedule(epoch: int, cfg: TrainConfig, started_at_epoch: int) -> float:
    if not cfg.enable_rank:
        return 0.0
    if cfg.rank_warmup_epochs <= 0:
        return cfg.gamma_rank_end
    progress = max(0, epoch - started_at_epoch) / float(cfg.rank_warmup_epochs)
    progress = min(1.0, max(0.0, progress))
    return cfg.gamma_rank_start + (cfg.gamma_rank_end - cfg.gamma_rank_start) * progress

# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def build_loss(cfg: TrainConfig, gamma: float, enable_rank: bool) -> WeightedDualLoss:
    return WeightedDualLoss(LossConfig(
        alpha=cfg.alpha_cls,
        beta=cfg.beta_reg,
        gamma=gamma,
        label_smoothing=cfg.label_smoothing,
        rank_margin=cfg.rank_margin,
        enable_rank=enable_rank,
        sample_weight_mode=cfg.sample_weight_mode,
        intensity_threshold=cfg.intensity_threshold,
        weak_sample_weight=cfg.weak_sample_weight,
    ))

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: WeightedDualLoss,
             device: torch.device, use_amp: bool) -> Dict[str, float]:
    model.eval()
    losses, cls_losses, reg_losses = [], [], []
    correct = total = 0
    abs_err = 0.0
    for xb, yb, sb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        sb = sb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, pred_s, _ = model(xb)
            total_loss, parts = criterion(logits, pred_s, yb, sb)
        losses.append(parts["loss"])
        cls_losses.append(parts["cls"])
        reg_losses.append(parts["reg"])
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        abs_err += (pred_s - sb).abs().sum().item()
    n = max(1, total)
    return {
        "loss": float(np.mean(losses)),
        "cls": float(np.mean(cls_losses)),
        "reg": float(np.mean(reg_losses)),
        "acc": correct / n,
        "intensity_mae": abs_err / n,
    }

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: WeightedDualLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    deadline_ts: Optional[float] = None,
) -> Tuple[Dict[str, float], bool]:
    model.train()
    losses, cls_losses, reg_losses, rank_losses = [], [], [], []
    correct = total = 0
    deadline_hit = False
    for xb, yb, sb in loader:
        if deadline_ts is not None and time.time() >= deadline_ts:
            deadline_hit = True
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        sb = sb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, pred_s, _ = model(xb)
            loss, parts = criterion(logits, pred_s, yb, sb)
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(parts["loss"])
        cls_losses.append(parts["cls"])
        reg_losses.append(parts["reg"])
        rank_losses.append(parts["rank"])
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    n = max(1, total)
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "cls": float(np.mean(cls_losses)) if cls_losses else float("nan"),
        "reg": float(np.mean(reg_losses)) if reg_losses else float("nan"),
        "rank": float(np.mean(rank_losses)) if rank_losses else 0.0,
        "acc": correct / n,
    }
    return metrics, deadline_hit

# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_train_state(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    best_val_acc: float,
    best_epoch: int,
    bad_epochs: int,
    rank_started_at_epoch: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: TrainConfig,
) -> None:
    state = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "rank_started_at_epoch": int(rank_started_at_epoch),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "config": asdict(cfg) | {"data_path": str(cfg.data_path), "output_dir": str(cfg.output_dir)},
    }
    torch.save(state, path)

def save_encoder_only(path: Path, model: EEGConformerDualHead) -> None:
    torch.save({"model": model.state_dict(), "config": CONFORMER_CONFIG}, path)

# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def make_loader(
    ds: EEGWindowArrayDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    """OOM 修复版 DataLoader：默认关闭 pin_memory 和 persistent_workers。"""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False,
    )

def _parse_subjects(s: str) -> List[str]:
    return [p.strip() for p in s.split(",") if p.strip()]

def _log_memory(logger: logging.Logger, label: str = "") -> None:
    """Log current process RSS (if psutil available)."""
    try:
        import psutil
        rss_gb = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"[MEM{' '+label if label else ''}] RSS = {rss_gb:.2f} GB")
    except ImportError:
        pass

def run_training(cfg: TrainConfig) -> Dict[str, object]:
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    logger = setup_logger(cfg.output_dir)
    device = resolve_device(cfg.device)
    use_amp = bool(cfg.amp and device.type == "cuda")

    # ---- data ----
    from .dataset import load_dataset_npz
    x, y, s, meta, splits = load_dataset_npz(cfg.data_path)
    n_total = len(y)
    logger.info(f"Loaded data: X={x.shape} ({x.nbytes/1024**3:.2f} GB), y={y.shape}, s={s.shape}")
    _log_memory(logger, "after-load")

    # ---- subject-level filtering ----
    train_subj = _parse_subjects(cfg.train_subjects)
    val_subj   = _parse_subjects(cfg.val_subjects)
    test_subj  = _parse_subjects(cfg.test_subjects)
    use_subject_filter = bool(train_subj or val_subj or test_subj)

    if use_subject_filter:
        logger.info(f"[SUBJECT-FILTER] train={train_subj or 'ALL'}, "
                     f"val={val_subj or 'REST'}, test={test_subj or 'REST'}")
        all_subjects_in_data = sorted(set(str(m.get("subject", "")) for m in meta))
        logger.info(f"[SUBJECT-FILTER] subjects in data: {all_subjects_in_data}")

        if train_subj:
            train_idx = filter_by_subjects(meta, train_subj)
        else:
            excluded = set(val_subj) | set(test_subj)
            remaining = [sv for sv in all_subjects_in_data if sv not in excluded]
            train_idx = filter_by_subjects(meta, remaining) if remaining else np.arange(n_total, dtype=np.int64)

        if val_subj:
            val_idx = filter_by_subjects(meta, val_subj)
        else:
            rng = np.random.default_rng(cfg.seed)
            pool = np.setdiff1d(np.arange(n_total), np.concatenate([train_idx, filter_by_subjects(meta, test_subj)] if test_subj else [train_idx]))
            if len(pool) == 0:
                pool = np.setdiff1d(np.arange(n_total), train_idx)
            n_val = max(1, int(round(len(pool) * 0.1)))
            val_idx = rng.choice(pool, size=min(n_val, len(pool)), replace=False)
            val_idx.sort()

        if test_subj:
            test_idx = filter_by_subjects(meta, test_subj)
        else:
            used = set(train_idx.tolist()) | set(val_idx.tolist())
            pool = np.array([i for i in range(n_total) if i not in used])
            if len(pool) > 0:
                rng = np.random.default_rng(cfg.seed + 1)
                n_test = max(1, int(round(len(pool) * 0.5)))
                test_idx = rng.choice(pool, size=min(n_test, len(pool)), replace=False)
                test_idx.sort()
            else:
                test_idx = np.array([], dtype=np.int64)

        logger.info(f"[SUBJECT-FILTER] sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        if {"train", "val", "test"} <= set(splits.keys()):
            train_idx = splits["train"]
            val_idx = splits["val"]
            test_idx = splits["test"]
        else:
            rng = np.random.default_rng(cfg.seed)
            idx = np.arange(n_total)
            rng.shuffle(idx)
            n_test = int(round(n_total * 0.1))
            n_val = int(round(n_total * 0.1))
            test_idx = idx[:n_test]
            val_idx = idx[n_test:n_test + n_val]
            train_idx = idx[n_test + n_val:]
            logger.warning("No baked-in splits; random window-level split fallback.")

    logger.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    if len(val_idx) == 0:
        logger.error("Validation set is EMPTY. Aborting.")
        return {"error": "empty val set"}
    if len(train_idx) == 0:
        logger.error("Training set is EMPTY. Aborting.")
        return {"error": "empty train set"}

    # =====================================================
    # OOM 修复：用 indices 模式创建 Dataset（零拷贝）
    # 所有 dataset 共享同一份 x/y/s 底层数组，内存 ~1X
    # 旧代码 x[train_idx] 会复制 ~80% 数据，导致 ~2X 内存
    # =====================================================
    train_ds = EEGWindowArrayDataset(x, y, s, indices=train_idx)
    val_ds = EEGWindowArrayDataset(x, y, s, indices=val_idx)
    test_ds = EEGWindowArrayDataset(x, y, s, indices=test_idx) if len(test_idx) > 0 else None

    # OOM 修复：创建 DataLoader（关闭 pin_memory / persistent_workers）
    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers,
                               pin_memory=cfg.pin_memory,
                               persistent_workers=cfg.persistent_workers)
    val_loader = make_loader(val_ds, cfg.batch_size, False, cfg.num_workers,
                             pin_memory=cfg.pin_memory,
                             persistent_workers=cfg.persistent_workers)
    test_loader = make_loader(test_ds, cfg.batch_size, False, cfg.num_workers,
                              pin_memory=cfg.pin_memory,
                              persistent_workers=cfg.persistent_workers
                              ) if test_ds is not None else None

    # OOM 修复：积极释放 numpy 数组 + meta 列表
    # torch.from_numpy 创建的是零拷贝视图，torch tensor 持有底层内存
    # 删除 numpy 引用后，内存由 torch tensor 管理
    del meta, splits
    gc.collect()
    _log_memory(logger, "after-dataset-create")

    # ---- model ----
    model = EEGConformerDualHead().to(device)

    if cfg.freeze_intensity_head:
        n_frozen = freeze_intensity_head(model)
        n_params = count_parameters(model)
        logger.info(f"[FREEZE-INTENSITY] Frozen {n_frozen:,} params. "
                     f"Trainable: {n_params:,} ({n_params/1e6:.3f}M)")
    else:
        n_params = count_parameters(model)
        logger.info(f"EEGConformerDualHead params={n_params/1e6:.3f}M")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- checkpoint paths ----
    state_path = Path(cfg.resume_path) if cfg.resume_path else (cfg.output_dir / "train_state.pt")
    best_model_path = cfg.output_dir / "best_model.pt"
    best_encoder_path = cfg.output_dir / "best_encoder.pt"
    last_model_path = cfg.output_dir / "last_model.pt"
    summary_path = cfg.output_dir / "summary.json"
    config_dump_path = cfg.output_dir / "train_config.json"

    with open(config_dump_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg) | {"data_path": str(cfg.data_path), "output_dir": str(cfg.output_dir)},
                  fh, indent=2, ensure_ascii=False)

    # ---- resume ----
    best_val_acc, best_epoch, bad_epochs = -1.0, -1, 0
    start_epoch = 1
    rank_started_at_epoch = cfg.pretrain_epochs + 1
    if cfg.resume and state_path.exists():
        rs = torch.load(state_path, map_location="cpu")
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
        rank_started_at_epoch = int(rs.get("rank_started_at_epoch", rank_started_at_epoch))
        start_epoch = int(rs.get("epoch", 0)) + 1
        del rs
        gc.collect()
        logger.info(f"[RESUME] from epoch {start_epoch} (best={best_val_acc:.4f} @ ep {best_epoch})")

    # ---- runtime budget ----
    t0 = time.time()
    deadline_ts: Optional[float] = None
    if cfg.max_runtime_hours and cfg.max_runtime_hours > 0:
        deadline_ts = t0 + cfg.max_runtime_hours * 3600.0
        logger.info(f"Max runtime: {cfg.max_runtime_hours:.2f}h")

    # ---- training loop ----
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        last_epoch = epoch
        is_pretrain = epoch <= cfg.pretrain_epochs

        if is_pretrain:
            criterion = WeightedDualLoss(LossConfig(
                alpha=1.0, beta=0.0, gamma=0.0,
                label_smoothing=cfg.label_smoothing,
                enable_rank=False,
                sample_weight_mode=cfg.sample_weight_mode,
                intensity_threshold=cfg.intensity_threshold,
                weak_sample_weight=cfg.weak_sample_weight,
            ))
        else:
            gamma = gamma_schedule(epoch, cfg, rank_started_at_epoch)
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
                criterion = build_loss(cfg, gamma=gamma, enable_rank=cfg.enable_rank and gamma > 0)

        lr_now = cosine_lr(epoch - 1, cfg.max_epochs, cfg.lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        tr, deadline_hit = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, use_amp, cfg.grad_clip, deadline_ts,
        )
        va = evaluate(model, val_loader, criterion, device, use_amp)
        phase = "PRE" if is_pretrain else "JOINT"
        if cfg.freeze_intensity_head and not is_pretrain:
            phase = "CLS-ONLY"
        logger.info(
            f"[E{epoch:03d}|{phase}] lr={lr_now:.2e} "
            f"train: loss={tr['loss']:.4f} acc={tr['acc']:.4f} cls={tr['cls']:.4f} reg={tr['reg']:.4f} rank={tr['rank']:.4f} | "
            f"val: loss={va['loss']:.4f} acc={va['acc']:.4f} mae={va['intensity_mae']:.4f}"
        )

        improved = va["acc"] > best_val_acc
        if improved:
            best_val_acc, best_epoch, bad_epochs = va["acc"], epoch, 0
            torch.save({"model": model.state_dict(), "config": CONFORMER_CONFIG, "val": va}, best_model_path)
            save_encoder_only(best_encoder_path, model)
        else:
            bad_epochs += 1

        if epoch % cfg.save_interval == 0 or improved or deadline_hit:
            save_train_state(
                state_path, epoch, model, optimizer, scaler,
                best_val_acc, best_epoch, bad_epochs,
                rank_started_at_epoch,
                train_idx, val_idx, test_idx, cfg,
            )

        if deadline_hit:
            logger.warning(f"[TIMEUP] reached {cfg.max_runtime_hours:.2f}h at epoch {epoch}; saved.")
            break

        if bad_epochs >= cfg.patience:
            logger.info(f"[EARLY-STOP] no improvement for {cfg.patience} epochs. best={best_val_acc:.4f} @ ep {best_epoch}")
            break

    # ---- final save ----
    if cfg.save_last:
        torch.save({"model": model.state_dict(), "config": CONFORMER_CONFIG, "epoch": last_epoch}, last_model_path)

    # ---- test eval ----
    test_metrics = {}
    if best_model_path.exists() and test_loader is not None:
        ck = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ck["model"])
        del ck
        gc.collect()
        final_criterion = build_loss(cfg, gamma=0.0, enable_rank=False)
        test_metrics = evaluate(model, test_loader, final_criterion, device, use_amp)
        logger.info(f"[TEST] acc={test_metrics['acc']:.4f} mae={test_metrics['intensity_mae']:.4f}")
    elif best_model_path.exists() and test_loader is None:
        logger.info("[TEST] No test set; skipping.")
    else:
        logger.warning("No best model; skipping test.")

    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "n_params": int(n_params),
        "epochs_run": int(last_epoch),
        "elapsed_seconds": float(time.time() - t0),
        "freeze_intensity_head": cfg.freeze_intensity_head,
        "train_subjects": cfg.train_subjects,
        "val_subjects": cfg.val_subjects,
        "test_subjects": cfg.test_subjects,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info(f"Summary -> {summary_path}")
    return summary

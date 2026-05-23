"""Training loop for SEED-VII dual-head EEG-Conformer.

特性（覆盖 Design.md 全部代码原则）：
- 两阶段训练：(1) 仅 L_cls 预训练 N epochs；(2) 联合训练 L_cls + L_reg (+L_rank)。
- 余弦退火 (lr_max -> 1e-5)。
- 周期断点 (`train_state.pt`) + `--resume`。
- 软超时 (`--max-runtime-hours`, 默认 10 小时)：自动优雅保存退出，防进程被杀丢数据。
- AMP 混合精度（CUDA）。
- 早停（基于验证集分类准确率）。
- 训练日志逐 epoch 写入 `train.log`。
"""
from __future__ import annotations

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
from .dataset import EEGWindowArrayDataset
from .losses import LossConfig, WeightedDualLoss
from .model import EEGConformerDualHead, count_parameters


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
    """Linear warmup of ranking weight γ from start to end across `rank_warmup_epochs`."""
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
    """Returns (metrics, deadline_hit)."""
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
    """Save just the encoder weights (patch + pos_embed + transformer + feature_proj),
    plus the dual heads (so encode_seed_vii_features.py can also load classifier if needed)."""
    torch.save({"model": model.state_dict(), "config": CONFORMER_CONFIG}, path)


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def make_loader(ds: EEGWindowArrayDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
    )


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
    logger.info(f"Loaded data: X={x.shape}, y={y.shape}, s={s.shape}, splits_in_npz={list(splits.keys())}")

    # If split indices were baked into the npz at preprocess time, use them; otherwise random fallback
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
        logger.warning("No baked-in splits found in npz; falling back to random window-level split. "
                       "For zero-leak: do trial-level split during preprocessing instead.")

    logger.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_ds = EEGWindowArrayDataset(x[train_idx], y[train_idx], s[train_idx])
    val_ds = EEGWindowArrayDataset(x[val_idx], y[val_idx], s[val_idx])
    test_ds = EEGWindowArrayDataset(x[test_idx], y[test_idx], s[test_idx])
    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers)
    val_loader = make_loader(val_ds, cfg.batch_size, False, cfg.num_workers)
    test_loader = make_loader(test_ds, cfg.batch_size, False, cfg.num_workers)

    # ---- model ----
    model = EEGConformerDualHead().to(device)
    n_params = count_parameters(model)
    logger.info(f"EEGConformerDualHead params={n_params/1e6:.3f}M (target 0.7-0.8M)")

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
        logger.info(f"[RESUME] from epoch {start_epoch} (best={best_val_acc:.4f} @ ep {best_epoch})")

    # ---- runtime budget ----
    t0 = time.time()
    deadline_ts: Optional[float] = None
    if cfg.max_runtime_hours and cfg.max_runtime_hours > 0:
        deadline_ts = t0 + cfg.max_runtime_hours * 3600.0
        logger.info(f"Max runtime: {cfg.max_runtime_hours:.2f}h, deadline_ts={deadline_ts}")

    # ---- training ----
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        last_epoch = epoch
        # phase: pretrain (cls only) vs joint
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
            criterion = build_loss(cfg, gamma=gamma, enable_rank=cfg.enable_rank and gamma > 0)

        # cosine LR
        lr_now = cosine_lr(epoch - 1, cfg.max_epochs, cfg.lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        tr, deadline_hit = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, use_amp, cfg.grad_clip, deadline_ts,
        )
        va = evaluate(model, val_loader, criterion, device, use_amp)
        phase = "PRE" if is_pretrain else "JOINT"
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
            logger.warning(f"[TIMEUP] reached {cfg.max_runtime_hours:.2f}h budget at epoch {epoch}; saved and exiting gracefully.")
            break

        if bad_epochs >= cfg.patience:
            logger.info(f"[EARLY-STOP] no val improvement for {cfg.patience} epochs. best={best_val_acc:.4f} @ ep {best_epoch}")
            break

    # ---- final save ----
    if cfg.save_last:
        torch.save({"model": model.state_dict(), "config": CONFORMER_CONFIG, "epoch": last_epoch}, last_model_path)

    # ---- test eval (on best model) ----
    if best_model_path.exists():
        ck = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ck["model"])
    final_criterion = build_loss(cfg, gamma=0.0, enable_rank=False)
    test_metrics = evaluate(model, test_loader, final_criterion, device, use_amp)
    logger.info(f"[TEST] acc={test_metrics['acc']:.4f}  mae={test_metrics['intensity_mae']:.4f}")

    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "n_params": int(n_params),
        "epochs_run": int(last_epoch),
        "elapsed_seconds": float(time.time() - t0),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info(f"Summary -> {summary_path}")
    return summary

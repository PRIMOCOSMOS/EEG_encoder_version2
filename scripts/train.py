#!/usr/bin/env python3
"""Train (or resume) the SEED-VII dual-head model — OOM-safe 流式版.

核心：不一次性加载全部 X 到 RAM，用 memmap 流式读取。

用法:
  python scripts/train.py \
    --data-dir /workspace/preprocessed \
    --output-dir /workspace/runs \
    --model-type eegnet \
    --device auto --amp \
    --max-runtime-hours 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DEFAULTS
from src.trainer import TrainConfig, run_training


def parse_args():
    p = argparse.ArgumentParser(description="Train SEED-VII dual-head model (OOM-safe)")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory containing per-subject .npz files")
    p.add_argument("--output-dir", type=str, default=str(TRAIN_DEFAULTS["output_dir"]))
    p.add_argument("--seed", type=int, default=int(TRAIN_DEFAULTS["seed"]))
    p.add_argument("--batch-size", type=int, default=int(TRAIN_DEFAULTS["batch_size"]))
    p.add_argument("--num-workers", type=int, default=int(TRAIN_DEFAULTS["num_workers"]))
    p.add_argument("--lr", type=float, default=float(TRAIN_DEFAULTS["lr"]))
    p.add_argument("--min-lr", type=float, default=float(TRAIN_DEFAULTS["min_lr"]))
    p.add_argument("--weight-decay", type=float, default=float(TRAIN_DEFAULTS["weight_decay"]))
    p.add_argument("--grad-clip", type=float, default=float(TRAIN_DEFAULTS["grad_clip"]))
    p.add_argument("--pretrain-epochs", type=int, default=int(TRAIN_DEFAULTS["pretrain_epochs"]))
    p.add_argument("--max-epochs", type=int, default=int(TRAIN_DEFAULTS["max_epochs"]))
    p.add_argument("--patience", type=int, default=int(TRAIN_DEFAULTS["patience"]))
    p.add_argument("--alpha-cls", type=float, default=float(TRAIN_DEFAULTS["alpha_cls_start"]))
    p.add_argument("--beta-reg", type=float, default=float(TRAIN_DEFAULTS["beta_reg_start"]))
    p.add_argument("--gamma-rank-start", type=float,
                   default=float(TRAIN_DEFAULTS["gamma_rank_start"]))
    p.add_argument("--gamma-rank-end", type=float,
                   default=float(TRAIN_DEFAULTS["gamma_rank_end"]))
    p.add_argument("--rank-warmup-epochs", type=int,
                   default=int(TRAIN_DEFAULTS["rank_warmup_epochs"]))
    p.add_argument("--enable-rank", action="store_true")
    p.add_argument("--rank-margin", type=float, default=float(TRAIN_DEFAULTS["rank_margin"]))
    p.add_argument("--label-smoothing", type=float,
                   default=float(TRAIN_DEFAULTS["label_smoothing"]))
    p.add_argument("--sample-weight-mode", choices=["continuous", "threshold", "none"],
                   default=str(TRAIN_DEFAULTS["sample_weight_mode"]))
    p.add_argument("--intensity-threshold", type=float,
                   default=float(TRAIN_DEFAULTS["intensity_threshold"]))
    p.add_argument("--weak-sample-weight", type=float,
                   default=float(TRAIN_DEFAULTS["weak_sample_weight"]))
    p.add_argument("--device", choices=["auto", "cuda", "cpu"],
                   default=str(TRAIN_DEFAULTS["device"]))
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-path", type=str, default="")
    p.add_argument("--save-interval", type=int, default=int(TRAIN_DEFAULTS["save_interval"]))
    p.add_argument("--max-runtime-hours", type=float,
                   default=float(TRAIN_DEFAULTS["max_runtime_hours"]))
    p.add_argument("--save-features", action="store_true")
    p.add_argument("--feature-type", choices=["projected", "flatten"],
                   default=str(TRAIN_DEFAULTS["feature_type"]))
    p.add_argument("--train-subjects", type=str, default="",
                   help="Comma-separated subject IDs for training (e.g. '1,2,3')")
    p.add_argument("--val-subjects", type=str, default="",
                   help="Comma-separated subject IDs for validation")
    p.add_argument("--test-subjects", type=str, default="",
                   help="Comma-separated subject IDs for testing")
    p.add_argument("--freeze-intensity-head", action="store_true")
    p.add_argument("--model-type", choices=["eegnet", "conformer"],
                   default=str(TRAIN_DEFAULTS.get("model_type", "eegnet")),
                   help="Model architecture: 'eegnet' (default) or 'conformer'")
    p.add_argument("--val-ratio", type=float, default=float(TRAIN_DEFAULTS["val_ratio"]))
    p.add_argument("--test-ratio", type=float, default=float(TRAIN_DEFAULTS["test_ratio"]))
    p.add_argument("--mmap-cache-dir", type=str, default="",
                   help="Directory for memmap .npy cache (default: output_dir/_mmap_cache). "
                        "Set to a fast disk for best I/O performance.")
    args = p.parse_args()

    amp = bool(TRAIN_DEFAULTS["amp"])
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    return TrainConfig(
        data_dir=Path(args.data_dir), output_dir=Path(args.output_dir),
        seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers,
        lr=args.lr, min_lr=args.min_lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        pretrain_epochs=args.pretrain_epochs, max_epochs=args.max_epochs,
        patience=args.patience,
        alpha_cls=args.alpha_cls, beta_reg=args.beta_reg,
        gamma_rank_start=args.gamma_rank_start, gamma_rank_end=args.gamma_rank_end,
        rank_warmup_epochs=args.rank_warmup_epochs,
        enable_rank=bool(args.enable_rank),
        rank_margin=args.rank_margin, label_smoothing=args.label_smoothing,
        sample_weight_mode=args.sample_weight_mode,
        intensity_threshold=args.intensity_threshold,
        weak_sample_weight=args.weak_sample_weight,
        device=args.device, amp=amp, resume=args.resume, resume_path=args.resume_path,
        save_interval=args.save_interval, max_runtime_hours=args.max_runtime_hours,
        save_features=args.save_features, feature_type=args.feature_type,
        train_subjects=args.train_subjects, val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
        freeze_intensity_head=bool(args.freeze_intensity_head),
        model_type=args.model_type,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        mmap_cache_dir=args.mmap_cache_dir,
    )


if __name__ == "__main__":
    run_training(parse_args())

#!/usr/bin/env python3
"""每个被试独立训练一个模型（被试内 trial-level 分割）。

用法：
    python scripts/train_per_subject.py \
      --data-dir /workspace/preprocessed \
      --output-dir /workspace/runs_per_subject \
      --model-type eegnet \
      --device auto --amp \
      --max-runtime-hours 20

只跑部分被试（调试用）：
    python scripts/train_per_subject.py \
      --data-dir /workspace/preprocessed \
      --output-dir /workspace/runs_per_subject \
      --subjects 1,2,3

关闭训练集均衡：
    python scripts/train_per_subject.py ... --no-balance-train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DEFAULTS
from src.trainer import TrainConfig, run_training_per_subject


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Per-subject training: one model per subject, intra-subject trial split."
    )
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="runs_per_subject")
    p.add_argument("--subjects", type=str, default="",
                   help="Comma-separated subject IDs (default: all .npz files)")
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
    p.add_argument("--label-smoothing", type=float,
                   default=float(TRAIN_DEFAULTS["label_smoothing"]))
    p.add_argument("--sample-weight-mode", choices=["continuous", "threshold", "none"],
                   default=str(TRAIN_DEFAULTS["sample_weight_mode"]))
    p.add_argument("--intensity-threshold", type=float,
                   default=float(TRAIN_DEFAULTS["intensity_threshold"]))
    p.add_argument("--weak-sample-weight", type=float,
                   default=float(TRAIN_DEFAULTS["weak_sample_weight"]))
    p.add_argument("--val-ratio", type=float, default=float(TRAIN_DEFAULTS["val_ratio"]))
    p.add_argument("--test-ratio", type=float, default=float(TRAIN_DEFAULTS["test_ratio"]))
    p.add_argument("--device", choices=["auto", "cuda", "cpu"],
                   default=str(TRAIN_DEFAULTS["device"]))
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--model-type", choices=["eegnet", "conformer"],
                   default=str(TRAIN_DEFAULTS.get("model_type", "eegnet")))
    p.add_argument("--max-runtime-hours", type=float,
                   default=float(TRAIN_DEFAULTS["max_runtime_hours"]))
    p.add_argument("--save-interval", type=int, default=int(TRAIN_DEFAULTS["save_interval"]))
    p.add_argument("--freeze-intensity-head", action="store_true")
    p.add_argument("--mmap-cache-dir", type=str, default="")
    # ★ 样本均衡开关
    p.add_argument("--no-balance-train", action="store_true",
                   help="关闭训练集类别均衡（默认开启 WeightedRandomSampler）")

    args = p.parse_args()

    amp = bool(TRAIN_DEFAULTS["amp"])
    if args.amp:    amp = True
    if args.no_amp: amp = False

    balance_train = not args.no_balance_train

    return TrainConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        pretrain_epochs=args.pretrain_epochs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        alpha_cls=args.alpha_cls,
        beta_reg=args.beta_reg,
        label_smoothing=args.label_smoothing,
        sample_weight_mode=args.sample_weight_mode,
        intensity_threshold=args.intensity_threshold,
        weak_sample_weight=args.weak_sample_weight,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        device=args.device,
        amp=amp,
        model_type=args.model_type,
        max_runtime_hours=args.max_runtime_hours,
        save_interval=args.save_interval,
        freeze_intensity_head=bool(args.freeze_intensity_head),
        mmap_cache_dir=args.mmap_cache_dir,
        train_subjects=args.subjects,
        balance_train=balance_train,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result = run_training_per_subject(cfg)
    print("\n===== ALL SUBJECTS SUMMARY =====")
    print(json.dumps(
        {k: v for k, v in result.items() if k != "per_subject"},
        indent=2, ensure_ascii=False,
    ))

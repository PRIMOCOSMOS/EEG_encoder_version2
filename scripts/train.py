#!/usr/bin/env python3
"""全被试混合训练入口（一个模型）。

训练模式（--training-mode）：
  cls_only           只训练分类，强度头冻结，损失只有 L_cls（默认）
  joint              全程联合训练 L_cls + L_reg
  pretrain_then_joint 前 N epoch 只开 L_cls，之后联合训练

用法：
    # 只训练分类（推荐先跑）
    python scripts/train.py \\
      --data-dir /workspace/preprocessed \\
      --output-dir /workspace/runs \\
      --training-mode cls_only \\
      --device auto --amp

    # 联合训练
    python scripts/train.py \\
      --data-dir /workspace/preprocessed \\
      --output-dir /workspace/runs \\
      --training-mode joint \\
      --device auto --amp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DEFAULTS
from src.trainer import TrainConfig, run_training, VALID_TRAINING_MODES


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train SEED-VII dual-head model (OOM-safe)")
    p.add_argument("--data-dir",   type=str, required=True)
    p.add_argument("--output-dir", type=str, default=str(TRAIN_DEFAULTS["output_dir"]))
    p.add_argument("--seed",       type=int, default=int(TRAIN_DEFAULTS["seed"]))
    p.add_argument("--batch-size", type=int, default=int(TRAIN_DEFAULTS["batch_size"]))
    p.add_argument("--num-workers",type=int, default=int(TRAIN_DEFAULTS["num_workers"]))
    p.add_argument("--lr",         type=float, default=float(TRAIN_DEFAULTS["lr"]))
    p.add_argument("--min-lr",     type=float, default=float(TRAIN_DEFAULTS["min_lr"]))
    p.add_argument("--weight-decay",type=float, default=float(TRAIN_DEFAULTS["weight_decay"]))
    p.add_argument("--grad-clip",  type=float, default=float(TRAIN_DEFAULTS["grad_clip"]))
    p.add_argument("--pretrain-epochs", type=int,
                   default=int(TRAIN_DEFAULTS["pretrain_epochs"]))
    p.add_argument("--max-epochs", type=int, default=int(TRAIN_DEFAULTS["max_epochs"]))
    p.add_argument("--patience",   type=int, default=int(TRAIN_DEFAULTS["patience"]))
    p.add_argument("--alpha-cls",  type=float, default=float(TRAIN_DEFAULTS["alpha_cls_start"]))
    p.add_argument("--beta-reg",   type=float, default=float(TRAIN_DEFAULTS["beta_reg_start"]))
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
    p.add_argument("--sample-weight-mode",
                   choices=["continuous", "threshold", "none"],
                   default=str(TRAIN_DEFAULTS["sample_weight_mode"]))
    p.add_argument("--intensity-threshold", type=float,
                   default=float(TRAIN_DEFAULTS["intensity_threshold"]))
    p.add_argument("--weak-sample-weight", type=float,
                   default=float(TRAIN_DEFAULTS["weak_sample_weight"]))
    p.add_argument("--device", choices=["auto", "cuda", "cpu"],
                   default=str(TRAIN_DEFAULTS["device"]))
    p.add_argument("--amp",    action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-path", type=str, default="")
    p.add_argument("--save-interval", type=int, default=int(TRAIN_DEFAULTS["save_interval"]))
    p.add_argument("--max-runtime-hours", type=float,
                   default=float(TRAIN_DEFAULTS["max_runtime_hours"]))
    p.add_argument("--save-features", action="store_true")
    p.add_argument("--feature-type", choices=["projected", "flatten"],
                   default=str(TRAIN_DEFAULTS["feature_type"]))
    p.add_argument("--train-subjects", type=str, default="")
    p.add_argument("--val-subjects",   type=str, default="")
    p.add_argument("--test-subjects",  type=str, default="")
    p.add_argument("--model-type", choices=["eegnet", "conformer"],
                   default=str(TRAIN_DEFAULTS.get("model_type", "eegnet")))
    p.add_argument("--val-ratio",  type=float, default=float(TRAIN_DEFAULTS["val_ratio"]))
    p.add_argument("--test-ratio", type=float, default=float(TRAIN_DEFAULTS["test_ratio"]))
    p.add_argument("--mmap-cache-dir", type=str, default="")
    # ★ 训练模式
    p.add_argument(
        "--training-mode",
        choices=list(VALID_TRAINING_MODES),
        default=str(TRAIN_DEFAULTS.get("training_mode", "cls_only")),
        help=(
            "训练模式：\n"
            "  cls_only           — 强度头冻结，只训练分类（L_cls）\n"
            "  joint              — 全程联合训练（L_cls + L_reg）\n"
            "  pretrain_then_joint— 前 N epoch 只开 L_cls，之后联合训练"
        ),
    )
    # 样本均衡
    p.add_argument("--no-balance-train", action="store_true",
                   help="关闭训练集类别均衡（默认开启）")

    args = p.parse_args()

    amp = bool(TRAIN_DEFAULTS["amp"])
    if args.amp:    amp = True
    if args.no_amp: amp = False

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
        model_type=args.model_type,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        mmap_cache_dir=args.mmap_cache_dir,
        training_mode=args.training_mode,
        balance_train=not args.no_balance_train,
    )


if __name__ == "__main__":
    run_training(parse_args())

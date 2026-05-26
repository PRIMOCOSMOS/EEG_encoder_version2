#!/usr/bin/env python3
"""Train (or resume) the SEED-VII EEG-Conformer dual-head model."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_DEFAULTS  # noqa: E402
from src.trainer import TrainConfig, run_training  # noqa: E402

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train SEED-VII EEG-Conformer (dual-head)")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=str(TRAIN_DEFAULTS["output_dir"]))
    p.add_argument("--seed", type=int, default=int(TRAIN_DEFAULTS["seed"]))
    p.add_argument("--batch-size", type=int, default=int(TRAIN_DEFAULTS["batch_size"]))
    p.add_argument("--num-workers", type=int, default=int(TRAIN_DEFAULTS["num_workers"]))
    p.add_argument("--lr", type=float, default=float(TRAIN_DEFAULTS["lr"]))
    p.add_argument("--min-lr", type=float, default=float(TRAIN_DEFAULTS["min_lr"]))
    p.add_argument("--grad-clip", type=float, default=float(TRAIN_DEFAULTS["grad_clip"]))
    p.add_argument("--pretrain-epochs", type=int, default=int(TRAIN_DEFAULTS["pretrain_epochs"]))
    p.add_argument("--max-epochs", type=int, default=int(TRAIN_DEFAULTS["max_epochs"]))
    p.add_argument("--patience", type=int, default=int(TRAIN_DEFAULTS["patience"]))

    # loss
    p.add_argument("--alpha-cls", type=float, default=float(TRAIN_DEFAULTS["alpha_cls_start"]))
    p.add_argument("--beta-reg", type=float, default=float(TRAIN_DEFAULTS["beta_reg_start"]))
    p.add_argument("--gamma-rank-start", type=float, default=float(TRAIN_DEFAULTS["gamma_rank_start"]))
    p.add_argument("--gamma-rank-end", type=float, default=float(TRAIN_DEFAULTS["gamma_rank_end"]))
    p.add_argument("--rank-warmup-epochs", type=int, default=int(TRAIN_DEFAULTS["rank_warmup_epochs"]))
    p.add_argument(
        "--enable-rank",
        action="store_true",
        help="Turn on the margin ranking loss. Default off (退化方案).",
    )
    p.add_argument("--rank-margin", type=float, default=float(TRAIN_DEFAULTS["rank_margin"]))
    p.add_argument(
        "--label-smoothing", type=float, default=float(TRAIN_DEFAULTS["label_smoothing"])
    )
    p.add_argument(
        "--sample-weight-mode",
        choices=["continuous", "threshold", "none"],
        default=str(TRAIN_DEFAULTS["sample_weight_mode"]),
    )
    p.add_argument(
        "--intensity-threshold", type=float, default=float(TRAIN_DEFAULTS["intensity_threshold"])
    )
    p.add_argument(
        "--weak-sample-weight", type=float, default=float(TRAIN_DEFAULTS["weak_sample_weight"])
    )

    p.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default=str(TRAIN_DEFAULTS["device"])
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-path", type=str, default="")
    p.add_argument("--save-interval", type=int, default=int(TRAIN_DEFAULTS["save_interval"]))
    p.add_argument(
        "--max-runtime-hours", type=float, default=float(TRAIN_DEFAULTS["max_runtime_hours"])
    )
    p.add_argument("--save-features", action="store_true")
    p.add_argument(
        "--feature-type",
        choices=["projected", "flatten"],
        default=str(TRAIN_DEFAULTS["feature_type"]),
    )

    # ---- 过拟合缓解：新增参数 ----
    p.add_argument(
        "--freeze-intensity-head",
        action="store_true",
        help="Freeze the intensity regression head; only train the classification head. "
             "Recommended when val loss does not decrease (overfitting on regression).",
    )
    p.add_argument(
        "--train-subjects",
        type=str,
        default="",
        help="Comma-separated subject IDs for training (e.g. '1,2,3'). "
             "If empty, uses all except val/test subjects.",
    )
    p.add_argument(
        "--val-subjects",
        type=str,
        default="",
        help="Comma-separated subject IDs for validation (e.g. '10,11'). "
             "These subjects are held out entirely from training.",
    )
    p.add_argument(
        "--test-subjects",
        type=str,
        default="",
        help="Comma-separated subject IDs for testing (e.g. '19,20'). "
             "These subjects are held out entirely from training and validation.",
    )

    # ---- ModelScope auto-download fallback -----
    p.add_argument(
        "--ms-data",
        type=str,
        default="",
        help="ModelScope dataset id (e.g. DEREKVERSE/SEED-VII) to auto-download --data if missing locally",
    )
    p.add_argument(
        "--ms-data-path",
        type=str,
        default="",
        help="Path inside the ModelScope dataset for the npz file (e.g. artifacts/preprocessed/seed_vii.npz)",
    )
    p.add_argument("--ms-revision", type=str, default="master")
    p.add_argument(
        "--ms-token",
        type=str,
        default="",
        help="ModelScope token (or use env MODELSCOPE_API_TOKEN)",
    )

    args = p.parse_args()

    # ---- Auto-download training data from ModelScope if local file missing -----
    data_path = Path(args.data)
    if not data_path.exists():
        if args.ms_data and args.ms_data_path:
            print(
                f"[INFO] Local data not found at {data_path}. "
                f"Downloading from {args.ms_data}:{args.ms_data_path} ..."
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            from src.ms_download import download_one_file, login_if_token

            login_if_token(args.ms_token or os.environ.get("MODELSCOPE_API_TOKEN"))
            downloaded = download_one_file(
                dataset_id=args.ms_data,
                file_path=args.ms_data_path,
                local_dir=str(data_path.parent),
                revision=args.ms_revision,
                token=args.ms_token or os.environ.get("MODELSCOPE_API_TOKEN"),
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.resolve() != data_path.resolve():
                if downloaded_path.exists() and not data_path.exists():
                    downloaded_path.rename(data_path)
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Auto-download failed: expected {data_path} after downloading {downloaded_path}"
                )
            print(f"[OK] Downloaded training data to {data_path}")
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_path}. "
                "Provide --ms-data and --ms-data-path to auto-download from ModelScope."
            )

    amp = bool(TRAIN_DEFAULTS["amp"])
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    return TrainConfig(
        data_path=data_path,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        min_lr=args.min_lr,
        grad_clip=args.grad_clip,
        pretrain_epochs=args.pretrain_epochs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        alpha_cls=args.alpha_cls,
        beta_reg=args.beta_reg,
        gamma_rank_start=args.gamma_rank_start,
        gamma_rank_end=args.gamma_rank_end,
        rank_warmup_epochs=args.rank_warmup_epochs,
        enable_rank=bool(args.enable_rank),
        rank_margin=args.rank_margin,
        label_smoothing=args.label_smoothing,
        sample_weight_mode=args.sample_weight_mode,
        intensity_threshold=args.intensity_threshold,
        weak_sample_weight=args.weak_sample_weight,
        device=args.device,
        amp=amp,
        resume=args.resume,
        resume_path=args.resume_path,
        save_interval=args.save_interval,
        max_runtime_hours=args.max_runtime_hours,
        save_features=args.save_features,
        feature_type=args.feature_type,
        # ---- new ----
        freeze_intensity_head=bool(args.freeze_intensity_head),
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
    )

def main() -> None:
    cfg = parse_args()
    run_training(cfg)

if __name__ == "__main__":
    main()

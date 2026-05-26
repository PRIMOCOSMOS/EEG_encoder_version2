#!/usr/bin/env python3
"""Encode SEED-VII windows into projected / flatten features using a trained checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import encode_npz  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Encode SEED-VII windows with trained EEGNet")
    p.add_argument("--data", required=True, help="preprocessed npz")
    p.add_argument("--checkpoint", required=True, help="best_encoder.pt or best_model.pt")
    p.add_argument("--output", required=True, help="output .npz")
    p.add_argument("--feature-type", choices=["projected", "flatten"], default="projected")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--subset", choices=["train", "val", "test"], default=None)
    args = p.parse_args()

    encode_npz(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        device_arg=args.device,
        use_amp=args.amp,
        subset=args.subset,
    )


if __name__ == "__main__":
    main()
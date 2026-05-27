#!/usr/bin/env python3
"""Encode windows into features using a trained checkpoint."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import encode_npz


def main():
    p = argparse.ArgumentParser(description="Encode SEED-VII windows with trained model")
    p.add_argument("--data", required=True, help="Preprocessed .npz path")
    p.add_argument("--checkpoint", required=True, help="best_encoder.pt or best_model.pt")
    p.add_argument("--output", required=True, help="Output .npz path")
    p.add_argument("--model-type", choices=["eegnet", "conformer"], default="eegnet")
    p.add_argument("--feature-type", choices=["projected", "flatten"], default="projected")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--subset", choices=["train", "val", "test"], default=None)
    args = p.parse_args()

    encode_npz(
        data_path=args.data, checkpoint_path=args.checkpoint, output_path=args.output,
        model_type=args.model_type, feature_type=args.feature_type,
        batch_size=args.batch_size, device_arg=args.device,
        use_amp=args.amp, subset=args.subset,
    )


if __name__ == "__main__":
    main()
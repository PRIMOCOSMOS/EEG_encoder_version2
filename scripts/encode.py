#!/usr/bin/env python3
"""Encode windows into features using a trained checkpoint — OOM-safe."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import encode_from_npz_dir


def main():
    p = argparse.ArgumentParser(description="Encode SEED-VII windows (OOM-safe)")
    p.add_argument("--data-dir", required=True,
                   help="Directory containing per-subject .npz files")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-type", choices=["eegnet", "conformer"], default="eegnet")
    p.add_argument("--feature-type", choices=["projected", "flatten"], default="projected")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--subjects", type=str, default=None,
                   help="Comma-separated subject IDs to encode (default: all)")
    p.add_argument("--mmap-cache-dir", type=str, default="",
                   help="Directory for memmap .npy cache")
    args = p.parse_args()

    encode_from_npz_dir(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_type=args.model_type,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        device_arg=args.device,
        use_amp=args.amp,
        subjects=args.subjects,
        mmap_cache_dir=args.mmap_cache_dir,
    )


if __name__ == "__main__":
    main()

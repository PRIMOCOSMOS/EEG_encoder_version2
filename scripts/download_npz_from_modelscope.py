#!/usr/bin/env python3
"""Download preprocessed .npz files from ModelScope dataset for training.

在预处理已完成并上传后，训练时用此脚本将 .npz 文件拉回本地。

用法:
  export MODELSCOPE_API_TOKEN=...
  python scripts/download_npz_from_modelscope.py \
    --dataset DEREKVERSE/SEED-VII \
    --path-prefix preprocessed_npz \
    --local-dir /workspace/preprocessed \
    --subjects 1,2,3,4,5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description="Download .npz from ModelScope")
    p.add_argument("--dataset", type=str, default="DEREKVERSE/SEED-VII")
    p.add_argument("--path-prefix", type=str, default="preprocessed_npz")
    p.add_argument("--local-dir", type=str, required=True)
    p.add_argument("--subjects", type=str, default="",
                   help="Comma-separated subject IDs (default: 1-20)")
    p.add_argument("--token", type=str, default="")
    args = p.parse_args()

    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        print("[ERROR] No token. Set MODELSCOPE_API_TOKEN or use --token.")
        sys.exit(1)

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] \
        if args.subjects else [str(i) for i in range(1, 21)]

    from src.ms_io import download_dataset_file

    for subj in subjects:
        file_path = f"{args.path_prefix}/{subj}.npz"
        local_path = local_dir / f"{subj}.npz"
        if local_path.exists() and local_path.stat().st_size > 0:
            print(f"[SKIP] {local_path} already exists")
            continue
        try:
            download_dataset_file(
                dataset_id=args.dataset,
                file_path=file_path,
                local_dir=str(local_dir),
                token=token,
            )
        except Exception as e:
            print(f"[ERROR] Failed to download {file_path}: {e}")


if __name__ == "__main__":
    main()

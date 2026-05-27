#!/usr/bin/env python3
"""Upload preprocessed per-subject .npz files back to ModelScope dataset.

将预处理后的 20 个 .npz 文件上传到 ModelScope 数据集的 preprocessed_npz/ 目录。
已有的旧 npz 文件作废，以新的为准。

用法:
  export MODELSCOPE_API_TOKEN=...
  python scripts/upload_npz_to_modelscope.py \
    --npz-dir /workspace/preprocessed \
    --dataset DEREKVERSE/SEED-VII \
    --path-prefix preprocessed_npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description="Upload .npz files to ModelScope dataset")
    p.add_argument("--npz-dir", type=str, required=True,
                   help="Local directory containing per-subject .npz files")
    p.add_argument("--dataset", type=str, default="DEREKVERSE/SEED-VII",
                   help="ModelScope dataset id")
    p.add_argument("--path-prefix", type=str, default="preprocessed_npz",
                   help="Target path prefix in the dataset repo")
    p.add_argument("--token", type=str, default="",
                   help="ModelScope token (or set MODELSCOPE_API_TOKEN env)")
    p.add_argument("--use-folder-upload", action="store_true",
                   help="Use folder upload instead of per-file upload")
    args = p.parse_args()

    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        print("[ERROR] No token. Set MODELSCOPE_API_TOKEN or use --token.")
        sys.exit(1)

    npz_dir = Path(args.npz_dir)
    if not npz_dir.is_dir():
        print(f"[ERROR] Directory not found: {npz_dir}")
        sys.exit(1)

    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in {npz_dir}")
        sys.exit(1)

    from src.ms_io import login, upload_file_to_dataset, upload_folder_to_dataset

    login(token)

    if args.use_folder_upload:
        # Upload the whole folder at once
        upload_folder_to_dataset(
            local_dir=str(npz_dir),
            dataset_id=args.dataset,
            path_in_repo=args.path_prefix,
            token=token,
            commit_message=f"Upload {len(npz_files)} preprocessed .npz files (no ICA)",
            allow_patterns=["*.npz"],
        )
    else:
        # Upload one by one (more reliable for large files)
        for i, f in enumerate(npz_files):
            path_in_repo = f"{args.path_prefix}/{f.name}"
            print(f"\n[{i+1}/{len(npz_files)}] Uploading {f.name} -> {path_in_repo}")
            try:
                upload_file_to_dataset(
                    local_file=str(f),
                    dataset_id=args.dataset,
                    path_in_repo=path_in_repo,
                    token=token,
                    commit_message=f"Upload preprocessed {f.name} (no ICA)",
                )
            except Exception as e:
                print(f"  [ERROR] Failed to upload {f.name}: {e}")
                continue

    print(f"\n[DONE] Uploaded {len(npz_files)} .npz files to {args.dataset}/{args.path_prefix}/")


if __name__ == "__main__":
    main()

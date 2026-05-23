#!/usr/bin/env python3
"""Stream-merge the SEED-VII split volumes hosted on ModelScope into one zip
and upload it back to the same dataset, **without ever materializing the full
zip on disk**.

设计约束：
    - ModelScope 实例只有 100 GB 持久化盘
    - 32 个分卷合计 ≈ 160 GB
    - 完整 zip ≈ 160 GB
    → 不能让完整 zip 落盘；任意时刻最多保留 1 个分卷 (~5.4 GB)。

实现策略：见 src/stream_merge.py

资源占用上界：
    磁盘:  ≤ 1 volume (~5.4 GB) + 状态/日志 (<10 MB)
    内存:  O(buffer_size_mb) (~16 MB) + SHA-256 state (常数)
    网络:  下载 ≈ 2 × 160 GB（hash pass + upload pass，除非 LFS 全局已有 blob）
           上传 ≈ 1 × 160 GB

可恢复性：
    全过程状态写入 --state-file（默认 scratch/_merge_upload_state.json）。
    任何阶段中断后再跑都会从断点继续；尤其是 ModelScope LFS 的去重特性
    会让"已上传 blob"被 Stage 2 识别，自动跳过 Stage 3 整个上传。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stream_merge import stream_merge_and_upload  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description="Stream-merge split volumes on ModelScope and upload the combined zip.",
    )
    p.add_argument("--dataset", default="DEREKVERSE/SEED-VII",
                   help="ModelScope dataset id 'NAMESPACE/NAME'")
    p.add_argument("--pattern", default="*.zip.*",
                   help="Glob to find the split volumes in the dataset")
    p.add_argument("--path-in-repo", default="SEED-VII.zip",
                   help="Target file path in the dataset for the merged zip")
    p.add_argument("--scratch-dir", required=True,
                   help="Local directory for transient volume cache "
                        "(needs ≥ 1× volume size of free space, e.g. ~8 GB)")
    p.add_argument("--state-file", default="",
                   help="Where to persist resumable state (default: <scratch>/_merge_upload_state.json)")
    p.add_argument("--revision", default="master")
    p.add_argument("--token", default="", help="ModelScope token (overrides env MODELSCOPE_API_TOKEN)")
    p.add_argument("--commit-message", default="")
    p.add_argument("--commit-description", default="")
    p.add_argument("--chunk-size-mb", type=int, default=16,
                   help="Read chunk size for SHA-256 pass (default 16 MB)")
    p.add_argument("--buffer-size-mb", type=int, default=16,
                   help="Buffer size for upload pass (default 16 MB)")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the Stage-5 re-list verification")
    args = p.parse_args()

    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN") or None
    if not token:
        print("[WARN] No ModelScope token in --token or MODELSCOPE_API_TOKEN env. "
              "Upload will fail; only listing/hashing will work.", file=sys.stderr)

    st = stream_merge_and_upload(
        dataset_id=args.dataset,
        pattern=args.pattern,
        path_in_repo=args.path_in_repo,
        scratch_dir=args.scratch_dir,
        revision=args.revision,
        token=token,
        state_path=args.state_file or None,
        commit_message=args.commit_message,
        commit_description=args.commit_description,
        skip_verify=args.skip_verify,
        chunk_size=args.chunk_size_mb * 1024 * 1024,
        buffer_size_mb=args.buffer_size_mb,
    )
    print("\n[SUMMARY]")
    print(f"  sha256          = {st.sha256_hex}")
    print(f"  total_size      = {st.total_size}  ({st.total_size/1024/1024/1024:.2f} GB)")
    print(f"  blob_reused     = {st.blob_reused}")
    print(f"  commit_id       = {st.commit_id}")
    print(f"  state_file      = {Path(args.state_file or (Path(args.scratch_dir)/'_merge_upload_state.json'))}")


if __name__ == "__main__":
    main()

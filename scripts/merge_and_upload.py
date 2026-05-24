#!/usr/bin/env python3
"""Stream-merge SEED-VII split volumes on ModelScope and upload as one zip.

**v2: OSS Multipart Upload-based pipeline with real per-volume resumability.**

For each split volume in sequence:
 1) download to scratch
 2) upload as ONE OSS part
 3) delete local file

At any point ≤ 1 volume (~5 GiB) lives on disk. If interrupted, rerun the same
command: it queries OSS `list_parts()` to identify already-uploaded parts and
continues from there. No need to re-download/re-hash 160 GB.

Falls back to the legacy `stream_merge` (LFS-based, double-pass) only if
`--use-lfs` is passed.

**v2.1: 新增 `--workers` 参数，支持全面并发。**
当 workers > 1 时，使用 `src/oss_merge_concurrent.py` 让多个分卷同时
下载+上传，上下行带宽同时打满，速度提升数倍。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(
        description="Stream-merge split volumes on ModelScope (OSS multipart, resumable)",
    )
    p.add_argument(
        "--dataset",
        default="DEREKVERSE/SEED-VII",
        help="ModelScope dataset id 'NAMESPACE/NAME'",
    )
    p.add_argument(
        "--pattern",
        default="part_*",
        help="Glob to find the split volumes in the dataset",
    )
    p.add_argument(
        "--path-in-repo",
        default="SEED-VII.zip",
        help="Target file path in the dataset for the merged zip",
    )
    p.add_argument(
        "--scratch-dir",
        required=True,
        help="Local dir for transient volume cache (needs ≥ 1× volume size, e.g. ~6 GB)",
    )
    p.add_argument(
        "--state-file",
        default="",
        help="Where to persist resumable state (default: /_oss_merge_state.json)",
    )
    p.add_argument("--revision", default="master")
    p.add_argument(
        "--token", default="", help="ModelScope token (overrides env)"
    )
    p.add_argument(
        "--max-part-retries",
        type=int,
        default=5,
        help="Retry budget per OSS upload_part call (default 5)",
    )

    # ----- 并发参数 -----
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发线程数（默认 1 为串行，推荐 3~5）。峰值磁盘 ≈ workers × 5.4 GB。",
    )

    # ----- legacy / fallback path -----
    p.add_argument(
        "--use-lfs",
        action="store_true",
        help="Use the older LFS-based pipeline (requires full SHA-256 pre-pass; NOT resumable mid-stage)",
    )
    p.add_argument("--commit-message", default="")
    p.add_argument("--commit-description", default="")
    p.add_argument(
        "--chunk-size-mb",
        type=int,
        default=16,
        help="(LFS mode) read chunk for SHA-256",
    )
    p.add_argument(
        "--buffer-size-mb",
        type=int,
        default=16,
        help="(LFS mode) upload buffer",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="(LFS mode) skip re-list verification",
    )

    # ----- cleanup helper -----
    p.add_argument(
        "--abort-in-progress",
        action="store_true",
        help="Just cancel any in-progress multipart uploads for this path and exit",
    )

    args = p.parse_args()

    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN") or None
    if not token:
        print(
            "[ERROR] No ModelScope token in --token or MODELSCOPE_API_TOKEN env. Cannot upload.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Abort path
    if args.abort_in_progress:
        from src.oss_merge import abort_in_progress_upload

        n = abort_in_progress_upload(
            dataset_id=args.dataset,
            path_in_repo=args.path_in_repo,
            revision=args.revision,
            token=token,
        )
        print(f"Aborted {n} in-progress upload(s).")
        return

    # Legacy LFS path (kept for reference)
    if args.use_lfs:
        from src.stream_merge import stream_merge_and_upload

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
        print("\n[SUMMARY] (LFS pipeline)")
        print(f" sha256 = {st.sha256_hex}")
        print(f" total_size = {st.total_size}")
        print(f" blob_reused = {st.blob_reused}")
        print(f" commit_id = {st.commit_id}")
        return

    # Default: OSS multipart path (recommended, fully resumable)
    if args.workers > 1:
        print(f"[INFO] Using concurrent mode with {args.workers} workers.")
        from src.oss_merge_concurrent import stream_merge_and_upload_via_oss_concurrent

        st = stream_merge_and_upload_via_oss_concurrent(
            dataset_id=args.dataset,
            pattern=args.pattern,
            path_in_repo=args.path_in_repo,
            scratch_dir=args.scratch_dir,
            revision=args.revision,
            token=token,
            state_path=args.state_file or None,
            max_part_retries=args.max_part_retries,
            max_workers=args.workers,
        )
    else:
        from src.oss_merge import stream_merge_and_upload_via_oss

        st = stream_merge_and_upload_via_oss(
            dataset_id=args.dataset,
            pattern=args.pattern,
            path_in_repo=args.path_in_repo,
            scratch_dir=args.scratch_dir,
            revision=args.revision,
            token=token,
            state_path=args.state_file or None,
            max_part_retries=args.max_part_retries,
        )

    print("\n[SUMMARY] (OSS multipart pipeline)")
    print(f" total_size = {st.total_size} ({st.total_size/1024/1024/1024:.2f} GB)")
    print(f" oss_object_key = {st.oss_object_key}")
    print(f" upload_id = {st.upload_id}")
    print(f" parts_uploaded = {len(st.done_parts)}")
    print(f" final_etag = {st.final_etag}")
    print(
        f" state_file = {args.state_file or (Path(args.scratch_dir) / '_oss_merge_state.json')}"
    )


if __name__ == "__main__":
    main()

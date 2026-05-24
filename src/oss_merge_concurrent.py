"""Concurrent OSS multipart upload for SEED-VII split volumes.

核心改进：把原来的“下载 1 卷 → 上传 1 Part → 删除”串行流程改成多线程并发。
每个线程独立完成一个分卷的 download+upload+delete，多个卷同时在网络中流转，
上下行带宽可以同时打满。OSS Multipart Upload 协议天然允许 Part 乱序上传，
最后通过 list_parts + complete_multipart_upload 组装即可。

磁盘峰值：max_workers × 单卷大小（默认 4 × 5.4 GB ≈ 22 GB），100 GB 持久盘安全。
断点续传：仍通过 _oss_merge_state.json 保存已完成的 Part 列表，重跑自动跳过。
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .ms_download import download_one_file, login_if_token
from .oss_merge import (
    OssMergeState,
    _PartInfo,
    _make_oss_bucket,
    discover_volumes,
    init_or_resume_multipart,
    upload_one_volume_as_part,
)


def stream_merge_and_upload_via_oss_concurrent(
    dataset_id: str,
    pattern: str,
    path_in_repo: str,
    scratch_dir: str,
    revision: str = "master",
    token: Optional[str] = None,
    state_path: Optional[str] = None,
    max_part_retries: int = 5,
    max_workers: int = 4,
    show_progress: bool = True,
) -> OssMergeState:
    """End-to-end concurrent: discover → init multipart → concurrent per-volume download+upload → complete."""
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    sp = Path(state_path) if state_path else (scratch / "_oss_merge_state.json")

    login_if_token(token)

    # ---- Load or init state ----
    st = OssMergeState.load(sp)
    if st is None or st.dataset_id != dataset_id or st.path_in_repo != path_in_repo:
        st = OssMergeState(
            dataset_id=dataset_id,
            pattern=pattern,
            path_in_repo=path_in_repo,
            revision=revision,
        )

    # ---- Stage 0: discover volumes ----
    if not st.volumes:
        print(f"[Stage 0] Listing remote volumes in {dataset_id} matching {pattern!r} ...")
        st.volumes = discover_volumes(dataset_id, pattern, revision, token)
        st.total_size = sum(s for _, s in st.volumes)
        print(
            f"[Stage 0] Found {len(st.volumes)} volumes, total "
            f"{st.total_size / 1024 ** 3:.2f} GB"
        )
        st.save(sp)
    else:
        print(
            f"[Stage 0] (resume) {len(st.volumes)} volumes, "
            f"{st.total_size / 1024 ** 3:.2f} GB"
        )

    # ---- Stage 1: init/resume multipart upload ----
    print("[Stage 1] Connecting to OSS via ModelScope STS ...")
    bucket, bucket_name, oss_dir = _make_oss_bucket(dataset_id, revision, token)
    object_key = oss_dir.rstrip("/") + "/" + path_in_repo.lstrip("/")
    st.bucket_name = bucket_name
    st.oss_object_key = object_key

    if not st.upload_id:
        upload_id, done_part_numbers = init_or_resume_multipart(
            bucket, object_key, total_volumes=len(st.volumes)
        )
        st.upload_id = upload_id
        if done_part_numbers:
            print(
                f"[Stage 1] (resume) Found in-progress upload_id={upload_id[:12]}..., "
                f"{len(done_part_numbers)} parts already uploaded"
            )
            import oss2

            etag_map = {}
            for p in oss2.PartIterator(bucket, object_key, upload_id):
                etag_map[p.part_number] = (p.etag, int(p.size))

            st.done_parts = []
            for pn in done_part_numbers:
                etag, sz = etag_map.get(pn, (None, 0))
                if pn - 1 < len(st.volumes):
                    vol_name, vol_size = st.volumes[pn - 1]
                    if sz != vol_size:
                        print(
                            f" [WARN] part {pn} server-size={sz} != expected={vol_size}; "
                            "will re-upload"
                        )
                        continue
                    st.done_parts.append(
                        asdict(
                            _PartInfo(
                                part_number=pn,
                                etag=etag,
                                volume_name=vol_name,
                                volume_size=vol_size,
                            )
                        )
                    )
        else:
            print(f"[Stage 1] New multipart upload: {upload_id[:12]}...")
        st.stage_init_done = True
        st.save(sp)
    else:
        print(
            f"[Stage 1] (resume) upload_id={st.upload_id[:12]}..., "
            f"{len(st.done_parts)} parts already uploaded"
        )

    # ---- Determine remaining work ----
    done_pn_set = {p["part_number"] for p in st.done_parts}
    remaining = [
        (i + 1, name, size)
        for i, (name, size) in enumerate(st.volumes)
        if (i + 1) not in done_pn_set
    ]

    # Cleanup stale partial-downloads from previous interrupted runs
    for f in scratch.iterdir():
        if f.is_file() and any(f.name == n for n, _ in st.volumes):
            try:
                f.unlink()
                print(f" [cleanup] removed stale {f.name}")
            except Exception:
                pass

    # ---- Stage 2: concurrent download + upload ----
    if remaining:
        print(
            f"[Stage 2] {len(remaining)}/{len(st.volumes)} parts remaining to upload "
            f"(skipping {len(done_pn_set)} already done).\n"
            f"          Launching {max_workers} concurrent workers "
            f"(peak disk ≈ {max_workers * 5.4:.1f} GB)."
        )

        lock = threading.Lock()
        pbar = tqdm(
            total=len(st.volumes),
            initial=len(done_pn_set),
            desc="Concurrent Parts",
            unit="part",
            disable=not show_progress,
        )
        errors: list = []

        def _process_one(part_no: int, vol_name: str, vol_size: int):
            """Download one volume, upload it as an OSS part, delete local file."""
            # Each thread gets its own OSS bucket handle for thread safety
            local_bucket, _, _ = _make_oss_bucket(dataset_id, revision, token)
            local_path = scratch / vol_name
            try:
                # 1) Download (with one retry on size mismatch)
                if not local_path.exists() or local_path.stat().st_size != vol_size:
                    if local_path.exists():
                        local_path.unlink()
                    downloaded = download_one_file(
                        dataset_id,
                        vol_name,
                        str(scratch),
                        revision=revision,
                        token=token,
                    )
                    local_path = Path(downloaded)
                    if local_path.stat().st_size != vol_size:
                        # Retry once
                        local_path.unlink()
                        downloaded = download_one_file(
                            dataset_id,
                            vol_name,
                            str(scratch),
                            revision=revision,
                            token=token,
                        )
                        local_path = Path(downloaded)
                        if local_path.stat().st_size != vol_size:
                            raise RuntimeError(
                                f"Download size mismatch after retry: {vol_name}"
                            )

                # 2) Upload as one OSS part
                etag = upload_one_volume_as_part(
                    local_bucket,
                    object_key,
                    st.upload_id,
                    part_no,
                    local_path,
                    max_retries=max_part_retries,
                    progress_callback=None,
                )

                # 3) Delete immediately to free disk
                try:
                    local_path.unlink()
                except FileNotFoundError:
                    pass

                # 4) Record state (thread-safe)
                with lock:
                    st.done_parts.append(
                        asdict(
                            _PartInfo(
                                part_number=part_no,
                                etag=etag,
                                volume_name=vol_name,
                                volume_size=vol_size,
                            )
                        )
                    )
                    # Sort to keep deterministic order for final assembly
                    st.done_parts.sort(key=lambda x: x["part_number"])
                    st.save(sp)
                    pbar.update(1)
                return (part_no, "ok")
            except Exception as e:
                # Don't delete on error so user can inspect/retry
                with lock:
                    errors.append(
                        (part_no, vol_name, str(e), traceback.format_exc())
                    )
                    pbar.update(1)
                return (part_no, "error", e)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_one, pn, vn, vs): pn
                for pn, vn, vs in remaining
            }
            for fut in as_completed(futures):
                if fut.exception():
                    pn = futures[fut]
                    with lock:
                        errors.append(
                            (
                                pn,
                                None,
                                str(fut.exception()),
                                traceback.format_exc(),
                            )
                        )
                        pbar.update(1)

        pbar.close()

        if errors:
            print(
                f"\n[WARNING] {len(errors)} parts encountered errors. "
                "Rerun the same command to resume."
            )
            for pn, vn, msg, _ in errors[:5]:
                print(f"  part {pn} ({vn}): {msg}")
        else:
            st.stage_upload_done = True
            st.save(sp)
            print("\n[Stage 2] All parts uploaded successfully.")
    else:
        print("[Stage 2] All parts already uploaded.")

    # ---- Stage 3: complete multipart upload ----
    if not st.stage_complete_done:
        import oss2

        parts = [
            oss2.models.PartInfo(p["part_number"], p["etag"])
            for p in sorted(st.done_parts, key=lambda x: x["part_number"])
        ]
        if len(parts) == len(st.volumes):
            print(
                f"[Stage 3] Completing multipart upload with {len(parts)} parts ..."
            )
            result = bucket.complete_multipart_upload(
                object_key, st.upload_id, parts
            )
            st.final_etag = result.etag
            st.stage_complete_done = True
            st.save(sp)
            print(f"[Stage 3] Complete. Final ETag = {result.etag}")
        else:
            print(
                f"[Stage 3] Incomplete ({len(parts)}/{len(st.volumes)} parts); "
                "skipping complete. Please rerun after all parts succeed."
            )
    else:
        print(f"[Stage 3] (resume) already completed. ETag = {st.final_etag}")

    # ---- Verify ----
    print(f"\n[VERIFY] Checking object {object_key} exists on OSS ...")
    try:
        head = bucket.head_object(object_key)
        size_ok = int(head.content_length) == st.total_size
        print(
            f" ✓ exists; OSS size = {head.content_length} "
            f"({'matches' if size_ok else 'MISMATCH'} expected {st.total_size})"
        )
    except Exception as e:
        print(f" ✗ head_object failed: {e}")

    print(f"\n[SUCCESS] {path_in_repo} should now be visible at:")
    print(
        f" https://www.modelscope.cn/datasets/{dataset_id}/file/view/master?fileName={path_in_repo}"
    )
    return st

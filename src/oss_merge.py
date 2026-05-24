"""Stream-merge multi-volume split-zip on ModelScope using **OSS Multipart Upload**.

为什么用 OSS 直传而不是 ModelScope 的 LFS upload_file？
    1) ModelScope LFS upload 需要先知道完整文件的 SHA-256，而 hashlib 不能序列化
       中间状态 → 单遍上传一旦中断必须从头再算 160GB SHA-256，纯 Python 实现又
       慢到不可用（0.3 MB/s）。
    2) OSS multipart upload 是阿里云 OSS 原生协议，**支持任意时刻断点续传**：
       崩了重跑 → `list_parts()` 取已上传的 part 列表 → 跳过 → 继续。
    3) OSS upload_part 上限 5 GiB，恰好匹配用户分卷大小 (≈5 GiB)。
       直接 1 分卷 = 1 OSS part，不需要切片重组。
    4) ModelScope 自己的 `MsDataset.upload` 内部也是用 OSS，但要求本地完整文件，
       我们用更底层的 oss2 API 跳过这个限制。

资源占用上界：
    磁盘:  ≤ 1 分卷 (~5 GiB)
    内存:  oss2 SDK 默认 buffer (~10 MB)
    网络:  下载 ≈ 1 × 160 GB + 上传 ≈ 1 × 160 GB
    (相比 LFS 方案省了一遍 160 GB 的 SHA-256 下载！)

断点续传：
    状态写入 JSON 文件 (上次的 upload_id, 已完成 parts)，
    任意阶段崩了重跑同一条命令即可继续。
    即使状态文件丢了，也能通过 `list_multipart_uploads()` 找回 upload_id。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .ms_download import (
    download_one_file,
    list_dataset_files,
    login_if_token,
)
from .zip_stream import volume_sort_key


# =============================================================================
# State
# =============================================================================

@dataclass
class _PartInfo:
    part_number: int           # 1-based, OSS requires consecutive integers
    etag: str                  # returned by upload_part
    volume_name: str           # source split-volume name (informational)
    volume_size: int           # source split-volume size in bytes


@dataclass
class OssMergeState:
    dataset_id: str
    pattern: str
    path_in_repo: str
    revision: str = "master"

    # discovery
    volumes: List[Tuple[str, int]] = field(default_factory=list)
    total_size: int = 0

    # OSS multipart
    bucket_name: str = ""
    oss_object_key: str = ""        # full OSS object key (incl. dataset dir prefix)
    upload_id: str = ""             # OSS multipart upload id
    done_parts: List[Dict] = field(default_factory=list)   # serialized _PartInfo

    # progress
    stage_init_done: bool = False
    stage_upload_done: bool = False
    stage_complete_done: bool = False
    final_etag: str = ""

    started_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    def save(self, path: Path) -> None:
        self.last_updated_at = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> Optional["OssMergeState"]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["volumes"] = [tuple(v) for v in d.get("volumes", [])]
        return cls(**d)


# =============================================================================
# Helpers
# =============================================================================

def discover_volumes(dataset_id: str, pattern: str, revision: str, token: Optional[str]
                     ) -> List[Tuple[str, int]]:
    """List & sort remote split volumes; each entry is (path, size_in_bytes)."""
    import fnmatch as _fn
    login_if_token(token)
    files = list_dataset_files(dataset_id, revision=revision, token=token)
    files = [f for f in files if _fn.fnmatch(f.get("Path", ""), pattern)]
    if not files:
        raise RuntimeError(f"No remote volumes matched {pattern!r} in {dataset_id}")
    files.sort(key=lambda f: volume_sort_key(f.get("Path", "")))
    sized: List[Tuple[str, int]] = []
    for f in files:
        name = f.get("Path")
        size = int(f.get("Size", 0) or 0)
        if size <= 0:
            raise RuntimeError(f"Volume {name!r} has no Size in listing")
        sized.append((name, size))
    return sized


def _make_oss_bucket(dataset_id: str, revision: str, token: Optional[str]):
    """Build an oss2.Bucket using ModelScope's STS credentials."""
    import oss2  # lazy
    from modelscope.hub.api import HubApi

    login_if_token(token)
    namespace, name = dataset_id.split("/", 1)
    api = HubApi()

    oss_config = api.get_dataset_access_config_session(
        dataset_name=name, namespace=namespace, check_cookie=False, revision=revision,
    )

    region = oss_config["Region"]
    endpoint = f"https://{region}.aliyuncs.com"

    # ModelScope rotates STS creds; use a provider so oss2 refreshes them automatically
    from oss2 import CredentialsProvider
    from oss2.credentials import Credentials

    class _Provider(CredentialsProvider):
        def __init__(self, api, name, namespace, revision):
            self._api = api; self._name = name
            self._namespace = namespace; self._revision = revision
        def get_credentials(self):
            cfg = self._api.get_dataset_access_config_session(
                dataset_name=self._name, namespace=self._namespace,
                check_cookie=False, revision=self._revision,
            )
            return Credentials(
                cfg["AccessId"], cfg["AccessSecret"], cfg["SecurityToken"],
            )
    auth = oss2.ProviderAuthV4(_Provider(api, name, namespace, revision))

    bucket_name = oss_config["Bucket"]
    bucket = oss2.Bucket(
        auth=auth, endpoint=endpoint,
        bucket_name=bucket_name,
        region=region.lstrip("oss-"),
    )
    return bucket, bucket_name, oss_config["Dir"]


# =============================================================================
# Stage: init / resume the multipart upload
# =============================================================================

def init_or_resume_multipart(
    bucket, object_key: str, total_volumes: int,
):
    """Init a new multipart upload OR resume an in-progress one for the same key.

    Returns (upload_id, list_of_already_done_part_numbers).
    """
    import oss2

    existing_id = None
    try:
        for u in oss2.MultipartUploadIterator(bucket, prefix=object_key):
            if u.key == object_key:
                existing_id = u.upload_id
    except oss2.exceptions.AccessDenied as e:
        print(
            "[Stage 1] [WARN] OSS STS policy does not allow ListMultipartUploads; "
            "cannot auto-discover existing upload_id from bucket. "
            "Falling back to a fresh multipart upload. "
            "Keep the local state file for resume.\n"
            f"original error: {e}"
        )
        existing_id = None

    if existing_id is None:
        result = bucket.init_multipart_upload(object_key)
        return result.upload_id, []

    done = []
    for p in oss2.PartIterator(bucket, object_key, existing_id):
        done.append(p.part_number)
    return existing_id, sorted(done)


# =============================================================================
# Stage: upload one volume as one part (with retries)
# =============================================================================

def upload_one_volume_as_part(
    bucket, object_key: str, upload_id: str,
    part_number: int, local_path: Path,
    max_retries: int = 5,
    progress_callback=None,
) -> str:
    """Upload `local_path` as part #part_number; return ETag string."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            with open(local_path, "rb") as fh:
                result = bucket.upload_part(
                    object_key, upload_id, part_number, fh,
                    progress_callback=progress_callback,
                )
            return result.etag
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 60)
            print(f"  [retry {attempt}/{max_retries}] part {part_number} failed: {e}; "
                  f"sleeping {wait}s ...")
            time.sleep(wait)
    raise RuntimeError(
        f"Failed to upload part {part_number} after {max_retries} attempts: {last_err}"
    )


def verify_part_size(bucket, object_key: str, upload_id: str,
                     part_number: int, expected_size: int) -> bool:
    """Confirm that the already-uploaded part has the expected byte size."""
    import oss2
    for p in oss2.PartIterator(bucket, object_key, upload_id):
        if p.part_number == part_number:
            return int(p.size) == int(expected_size)
    return False


# =============================================================================
# Main orchestrator
# =============================================================================

def stream_merge_and_upload_via_oss(
    dataset_id: str,
    pattern: str,
    path_in_repo: str,
    scratch_dir: str,
    revision: str = "master",
    token: Optional[str] = None,
    state_path: Optional[str] = None,
    max_part_retries: int = 5,
    show_progress: bool = True,
) -> OssMergeState:
    """End-to-end: discover → init multipart → per-volume (download → upload → delete) → complete.

    Fully resumable: rerun the same command to continue. ``done_parts`` is
    persisted to ``state_path`` AND derived from OSS ``list_parts()`` so even
    a lost state file recovers gracefully.
    """
    scratch = Path(scratch_dir); scratch.mkdir(parents=True, exist_ok=True)
    sp = Path(state_path) if state_path else (scratch / "_oss_merge_state.json")

    # Load / init state
    st = OssMergeState.load(sp)
    if st is None or st.dataset_id != dataset_id or st.path_in_repo != path_in_repo:
        st = OssMergeState(
            dataset_id=dataset_id, pattern=pattern,
            path_in_repo=path_in_repo, revision=revision,
        )

    # ---- Stage 0: discover volumes ----
    if not st.volumes:
        print(f"[Stage 0] Listing remote volumes in {dataset_id} matching {pattern!r} ...")
        st.volumes = discover_volumes(dataset_id, pattern, revision, token)
        st.total_size = sum(s for _, s in st.volumes)
        print(f"[Stage 0] Found {len(st.volumes)} volumes, total "
              f"{st.total_size/1024/1024/1024:.2f} GB")
        st.save(sp)
    else:
        print(f"[Stage 0] (resume) {len(st.volumes)} volumes, "
              f"{st.total_size/1024/1024/1024:.2f} GB")

    # ---- Stage 1: open OSS bucket + init/resume multipart ----
    print("[Stage 1] Connecting to OSS via ModelScope STS ...")
    bucket, bucket_name, oss_dir = _make_oss_bucket(dataset_id, revision, token)
    object_key = oss_dir.rstrip("/") + "/" + path_in_repo.lstrip("/")
    st.bucket_name = bucket_name
    st.oss_object_key = object_key

    if not st.upload_id:
        upload_id, done_part_numbers = init_or_resume_multipart(
            bucket, object_key, total_volumes=len(st.volumes),
        )
        st.upload_id = upload_id
        if done_part_numbers:
            # Resync: rebuild done_parts from OSS (server-side truth)
            print(f"[Stage 1] (resume) Found in-progress upload_id={upload_id[:12]}..., "
                  f"{len(done_part_numbers)} parts already uploaded")
            import oss2
            etag_map = {}
            for p in oss2.PartIterator(bucket, object_key, upload_id):
                etag_map[p.part_number] = (p.etag, int(p.size))
            st.done_parts = []
            for pn in done_part_numbers:
                etag, sz = etag_map[pn]
                vol_name, vol_size = st.volumes[pn - 1]
                if sz != vol_size:
                    print(f"  [WARN] part {pn} server-size={sz} != expected={vol_size}; "
                          f"will re-upload")
                    continue
                st.done_parts.append(asdict(_PartInfo(
                    part_number=pn, etag=etag,
                    volume_name=vol_name, volume_size=vol_size,
                )))
        else:
            print(f"[Stage 1] New multipart upload: {upload_id[:12]}...")
        st.stage_init_done = True
        st.save(sp)
    else:
        print(f"[Stage 1] (resume) upload_id={st.upload_id[:12]}..., "
              f"{len(st.done_parts)} parts already uploaded")

    # ---- Stage 2: per-volume download → upload → delete ----
    if not st.stage_upload_done:
        done_pn_set = {p["part_number"] for p in st.done_parts}
        remaining = [(i + 1, name, size) for i, (name, size) in enumerate(st.volumes)
                     if (i + 1) not in done_pn_set]
        # Clean stale partial-downloads from previous interrupted runs to free disk
        for f in scratch.iterdir():
            if f.is_file() and any(f.name == n for n, _ in st.volumes):
                try:
                    f.unlink()
                    print(f"  [cleanup] removed stale {f.name}")
                except Exception:
                    pass
        print(f"[Stage 2] {len(remaining)}/{len(st.volumes)} parts remaining to upload "
              f"(skipping {len(done_pn_set)} already done)")

        for part_no, vol_name, vol_size in remaining:
            print(f"\n[Stage 2] === Part {part_no}/{len(st.volumes)}: {vol_name} "
                  f"({vol_size/1024/1024/1024:.2f} GB) ===")
            t0 = time.time()

            # 1) download this volume
            print(f"  [download] {vol_name} -> {scratch} ...")
            local = download_one_file(
                dataset_id, vol_name, str(scratch),
                revision=revision, token=token,
            )
            actual_size = local.stat().st_size
            if actual_size != vol_size:
                # Could be a partial download; retry once
                print(f"  [WARN] downloaded size {actual_size} != expected {vol_size}; redownloading")
                try: local.unlink()
                except Exception: pass
                local = download_one_file(
                    dataset_id, vol_name, str(scratch),
                    revision=revision, token=token,
                )
                if local.stat().st_size != vol_size:
                    raise RuntimeError(f"Volume {vol_name} download size mismatch")
            dl_secs = time.time() - t0
            dl_mbps = (vol_size / 1024 / 1024) / max(1.0, dl_secs)
            print(f"  [download] done in {dl_secs:.1f}s ({dl_mbps:.1f} MB/s)")

            # 2) upload as a part
            t0 = time.time()
            pbar = None
            def _cb(consumed, total):
                nonlocal pbar
                if pbar is None and show_progress:
                    pbar = tqdm(total=total, unit="B", unit_scale=True,
                                desc=f"  [upload part {part_no}]")
                if pbar is not None:
                    pbar.update(consumed - pbar.n)

            try:
                etag = upload_one_volume_as_part(
                    bucket, object_key, st.upload_id, part_no, local,
                    max_retries=max_part_retries,
                    progress_callback=_cb,
                )
            finally:
                if pbar is not None:
                    pbar.close()

            up_secs = time.time() - t0
            up_mbps = (vol_size / 1024 / 1024) / max(1.0, up_secs)
            print(f"  [upload] etag={etag} in {up_secs:.1f}s ({up_mbps:.1f} MB/s)")

            # 3) delete local volume IMMEDIATELY to free disk
            try:
                local.unlink()
            except FileNotFoundError:
                pass

            # 4) persist state
            st.done_parts.append(asdict(_PartInfo(
                part_number=part_no, etag=etag,
                volume_name=vol_name, volume_size=vol_size,
            )))
            st.save(sp)

        st.stage_upload_done = True
        st.save(sp)
        print(f"\n[Stage 2] All {len(st.volumes)} parts uploaded.")
    else:
        print(f"[Stage 2] (resume) all parts already uploaded")

    # ---- Stage 3: complete multipart upload ----
    if not st.stage_complete_done:
        import oss2
        parts = [oss2.models.PartInfo(p["part_number"], p["etag"])
                 for p in sorted(st.done_parts, key=lambda x: x["part_number"])]
        print(f"[Stage 3] Completing multipart upload with {len(parts)} parts ...")
        result = bucket.complete_multipart_upload(object_key, st.upload_id, parts)
        st.final_etag = result.etag
        st.stage_complete_done = True
        st.save(sp)
        print(f"[Stage 3] ✓ Complete. Final ETag = {result.etag}")
    else:
        print(f"[Stage 3] (resume) already completed. ETag = {st.final_etag}")

    # ---- Verify ----
    print(f"\n[VERIFY] Checking object {object_key} exists on OSS ...")
    try:
        head = bucket.head_object(object_key)
        size_ok = int(head.content_length) == st.total_size
        print(f"  ✓ exists; OSS size = {head.content_length} "
              f"({'matches' if size_ok else 'MISMATCH'} expected {st.total_size})")
    except Exception as e:
        print(f"  ✗ head_object failed: {e}")

    print(f"\n[SUMMARY] {path_in_repo} should now be visible at:")
    print(f"  https://www.modelscope.cn/datasets/{dataset_id}/file/view/master?fileName={path_in_repo}")
    return st


def abort_in_progress_upload(
    dataset_id: str, path_in_repo: str,
    revision: str = "master", token: Optional[str] = None,
) -> int:
    """Cancel any in-progress multipart upload for the given path. Useful for cleanup.

    Returns the number of uploads aborted.
    """
    import oss2
    bucket, _, oss_dir = _make_oss_bucket(dataset_id, revision, token)
    object_key = oss_dir.rstrip("/") + "/" + path_in_repo.lstrip("/")
    n = 0
    for u in oss2.MultipartUploadIterator(bucket, prefix=object_key):
        if u.key == object_key:
            try:
                bucket.abort_multipart_upload(object_key, u.upload_id)
                print(f"aborted upload_id={u.upload_id[:12]}...")
                n += 1
            except Exception as e:
                print(f"abort failed for {u.upload_id[:12]}: {e}")
    return n

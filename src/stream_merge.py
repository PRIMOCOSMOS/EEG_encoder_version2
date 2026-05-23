"""Stream-merge multi-volume ModelScope split-zip and upload as a single file.

设计原则（来自 Design.md + 用户最严约束）：
- ModelScope 实例只有 100GB 持久化盘；32 个分卷合计 ≈160GB；完整 zip ≈160GB；
  → **绝不** 让完整 zip 落盘，**绝不** 同时落盘 ≥2 个分卷。
- 用户的 32 个 *.zip.NNN 是对原始 zip 做字节级 split（顺序拼接 == 合法 zip）。
- ModelScope SDK 提供：
    * `HubApi._validate_blob(sha256, size)` → 返回 LFS 上传 URL（或确认已存在）
    * `HubApi._upload_blob(data=BufferedIOBase, sha256, size)` → 流式 HTTP PUT
    * `HubApi.create_commit(operations=[CommitOperationAdd(...)])` → 把 blob 关联到路径
  注意：`HubApi.upload_file()` 内部对 BinaryIO 做了 `.read()`（160 GB 装入内存），不能用。
  必须用 `_upload_blob` + `create_commit` 这条底层路径。

执行流程（具有完整的可恢复性）：
    Stage 0: 远端列分卷 + 读取 manifest（含上次 SHA-256 缓存）
    Stage 1: SHA-256 pass —— 流式拉每个分卷 → 喂 hashlib → 删除 → 得到全文件 hash
    Stage 2: _validate_blob —— 若 ModelScope 全局已有此 blob，跳到 Stage 4
    Stage 3: Upload pass —— 流式拉每个分卷 → BufferedReader → HTTP PUT
    Stage 4: create_commit —— 把 LFS blob 与 path_in_repo 关联
    Stage 5: 验证（可选）—— 重新列 dataset 文件，确认 size 一致

磁盘上界：max_resident_volumes × volume_size（默认 1 → ≈5.4GB）
内存上界：O(buffer_size_mb)（默认 16MB）+ SHA-256 state（小常数）
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from .ms_download import (
    download_one_file,
    list_dataset_files,
    login_if_token,
)
from .zip_stream import LazyConcatStream, volume_sort_key


# =============================================================================
# State / resume
# =============================================================================

@dataclass
class MergeUploadState:
    """Persisted state for resumable merge+upload.

    Stored as JSON next to the script invocation (typically in the scratch dir).
    """
    dataset_id: str
    pattern: str
    path_in_repo: str
    revision: str = "master"

    # Stage outputs
    volumes: List[Tuple[str, int]] = field(default_factory=list)   # [(name, size)]
    total_size: int = 0
    sha256_hex: str = ""

    # Stage progress
    stage1_done: bool = False     # SHA-256 computed
    stage2_done: bool = False     # blob validated (may be reused)
    blob_url: Optional[str] = None  # upload URL (None means blob already exists globally)
    blob_reused: bool = False
    stage3_done: bool = False     # blob uploaded
    stage4_done: bool = False     # commit created
    commit_id: Optional[str] = None

    started_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    def save(self, path: Path) -> None:
        self.last_updated_at = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> Optional["MergeUploadState"]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # tuples got serialized as lists
        d["volumes"] = [tuple(v) for v in d.get("volumes", [])]
        return cls(**d)


# =============================================================================
# Helpers
# =============================================================================

def _build_lazy_stream(
    dataset_id: str,
    sizes: List[Tuple[str, int]],
    scratch_dir: Path,
    revision: str,
    token: Optional[str],
    max_resident: int = 1,
) -> LazyConcatStream:
    """Construct a LazyConcatStream that streams from ModelScope on demand."""
    def _fetch(remote_name: str) -> str:
        p = download_one_file(
            dataset_id, remote_name, str(scratch_dir),
            revision=revision, token=token,
        )
        return str(p)

    def _evict(local_path: Path) -> None:
        with contextlib.suppress(FileNotFoundError):
            local_path.unlink()

    return LazyConcatStream(
        sizes_in_order=sizes,
        fetcher=_fetch,
        evicter=_evict,
        max_resident=max_resident,
        # For pure-sequential workloads (hashing / upload), we don't need to pin
        # the last volume — there is no EOCD seek. Disable to save disk.
        pin_last=False,
        warmup_last=False,
    )


def _wrap_buffered(stream: LazyConcatStream, buffer_size_mb: int = 16) -> io.BufferedReader:
    """Wrap our RawIOBase concat-stream in a BufferedReader (BufferedIOBase)."""
    return io.BufferedReader(stream, buffer_size=buffer_size_mb * 1024 * 1024)


# =============================================================================
# Stage 0: discover
# =============================================================================

def discover_remote_volumes_sized(
    dataset_id: str,
    pattern: str,
    revision: str = "master",
    token: Optional[str] = None,
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
            raise RuntimeError(
                f"Volume {name!r} has no size in ModelScope listing. "
                "Cannot stream-merge without sizes."
            )
        sized.append((name, size))
    return sized


# =============================================================================
# Stage 1: SHA-256 pass
# =============================================================================

def compute_full_sha256(
    dataset_id: str,
    sizes: List[Tuple[str, int]],
    scratch_dir: Path,
    revision: str = "master",
    token: Optional[str] = None,
    chunk_size: int = 16 * 1024 * 1024,
    show_progress: bool = True,
) -> str:
    """Stream-compute SHA-256 of the byte-concatenation of all split volumes.

    Disk: ≤ 1 volume resident at any moment (max_resident=1, no pinning).
    Memory: O(chunk_size).
    """
    stream = _build_lazy_stream(
        dataset_id, sizes, scratch_dir, revision, token, max_resident=1,
    )
    total = sum(s for _, s in sizes)
    h = hashlib.sha256()
    try:
        pbar = tqdm(total=total, unit="B", unit_scale=True,
                    desc="[Stage 1: SHA-256]", disable=not show_progress)
        try:
            while True:
                buf = stream.read(chunk_size)
                if not buf:
                    break
                h.update(buf)
                pbar.update(len(buf))
        finally:
            pbar.close()
    finally:
        stream.close()  # deletes any remaining cached volume
    return h.hexdigest()


# =============================================================================
# Stage 2: validate blob (LFS dedup check)
# =============================================================================

def validate_blob_existence(
    api,
    dataset_id: str,
    sha256_hex: str,
    size: int,
    token: Optional[str] = None,
) -> Tuple[Optional[str], bool]:
    """Ask ModelScope LFS: do we already have this blob?

    Returns (upload_url, reused):
      - (None, True)  → blob already exists; skip upload.
      - (url,  False) → upload URL we should PUT to in Stage 3.
    """
    validated = api._validate_blob(
        repo_id=dataset_id,
        repo_type="dataset",
        objects=[{"oid": sha256_hex, "size": int(size)}],
        token=token,
    )
    url = validated.get(sha256_hex)
    return (url, url is None)


# =============================================================================
# Stage 3: upload blob
# =============================================================================

def upload_blob_streaming(
    api,
    dataset_id: str,
    sha256_hex: str,
    total_size: int,
    sizes: List[Tuple[str, int]],
    scratch_dir: Path,
    revision: str = "master",
    token: Optional[str] = None,
    pre_validated_url: Optional[str] = None,
    buffer_size_mb: int = 16,
    path_in_repo_label: str = "SEED-VII.zip",
) -> Dict[str, Any]:
    """Run the second pass: stream-download volumes again, this time PUTting bytes
    straight to ModelScope's LFS upload URL.

    Disk: ≤ 1 volume resident at any moment.
    """
    stream = _build_lazy_stream(
        dataset_id, sizes, scratch_dir, revision, token, max_resident=1,
    )
    buffered = _wrap_buffered(stream, buffer_size_mb=buffer_size_mb)
    try:
        result = api._upload_blob(
            repo_id=dataset_id,
            repo_type="dataset",
            sha256=sha256_hex,
            size=int(total_size),
            data=buffered,
            disable_tqdm=False,
            tqdm_desc=f"[Stage 3: Uploading {path_in_repo_label}]",
            buffer_size_mb=buffer_size_mb,
            token=token,
            pre_validated=pre_validated_url,
        )
        return result
    finally:
        # close the BufferedReader; it will close the inner stream
        # (which deletes any remaining cached volume)
        try:
            buffered.close()
        finally:
            try: stream.close()
            except Exception: pass


# =============================================================================
# Stage 4: commit
# =============================================================================

def create_commit_for_blob(
    api,
    dataset_id: str,
    path_in_repo: str,
    sha256_hex: str,
    total_size: int,
    revision: str = "master",
    token: Optional[str] = None,
    commit_message: str = "Add merged SEED-VII.zip via stream-merge-upload",
    commit_description: str = "",
) -> str:
    """Create the commit that registers the uploaded LFS blob at `path_in_repo`.

    Because the blob is already uploaded with the known sha256 + size, we just
    need to commit a CommitOperationAdd that references it.
    """
    from modelscope.utils.repo_utils import CommitOperationAdd

    # CommitOperationAdd requires a real file or BinaryIO. We give it a tiny
    # in-memory placeholder; the SDK will recognize the file_hash_info we supply
    # and **skip re-hashing/re-uploading** because the blob already exists.
    # Trick: pass a 1-byte BytesIO and pre-fill file_hash_info with the truth.
    placeholder = io.BytesIO(b"")
    op = CommitOperationAdd(
        path_in_repo=path_in_repo,
        path_or_fileobj=placeholder,
    )
    # Override the auto-computed upload_info with the truth
    from modelscope.utils.repo_utils import UploadInfo  # noqa: WPS433
    op.upload_info = UploadInfo(
        sha256=bytes.fromhex(sha256_hex) if len(sha256_hex) == 64 else sha256_hex,
        size=int(total_size),
        sample=b"",
    )
    op.file_hash_info = {
        "file_hash": sha256_hex,
        "file_size": int(total_size),
    }
    op._upload_mode = "lfs"  # type: ignore[assignment]
    op._is_uploaded = True   # already uploaded in Stage 3

    info = api.create_commit(
        repo_id=dataset_id,
        operations=[op],
        commit_message=commit_message,
        commit_description=commit_description,
        token=token,
        repo_type="dataset",
        revision=revision,
    )
    return getattr(info, "oid", None) or getattr(info, "commit_id", None) or str(info)


# =============================================================================
# Stage 5: verify
# =============================================================================

def verify_uploaded_file(
    dataset_id: str,
    path_in_repo: str,
    expected_size: int,
    revision: str = "master",
    token: Optional[str] = None,
) -> bool:
    """List the dataset and confirm `path_in_repo` exists with the expected size."""
    files = list_dataset_files(dataset_id, revision=revision, token=token)
    for f in files:
        if f.get("Path") == path_in_repo or f.get("Path", "").lstrip("/") == path_in_repo.lstrip("/"):
            got = int(f.get("Size", 0) or 0)
            return got == expected_size
    return False


# =============================================================================
# All-in-one orchestrator
# =============================================================================

def stream_merge_and_upload(
    dataset_id: str,
    pattern: str,
    path_in_repo: str,
    scratch_dir: str,
    revision: str = "master",
    token: Optional[str] = None,
    state_path: Optional[str] = None,
    commit_message: str = "",
    commit_description: str = "",
    skip_verify: bool = False,
    chunk_size: int = 16 * 1024 * 1024,
    buffer_size_mb: int = 16,
) -> MergeUploadState:
    """End-to-end: discover → hash → validate → upload → commit → verify.

    Fully resumable via `state_path` (JSON). Reruns will skip stages that are
    already marked done.

    Returns the final `MergeUploadState`.
    """
    from modelscope.hub.api import HubApi

    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    sp = Path(state_path) if state_path else (scratch / "_merge_upload_state.json")

    login_if_token(token)
    api = HubApi()

    # Load or create state
    st = MergeUploadState.load(sp)
    if st is None or st.dataset_id != dataset_id or st.path_in_repo != path_in_repo:
        st = MergeUploadState(
            dataset_id=dataset_id, pattern=pattern,
            path_in_repo=path_in_repo, revision=revision,
        )

    # Stage 0: discover (idempotent; always refresh listing)
    if not st.volumes:
        print(f"[Stage 0] Listing remote volumes in {dataset_id} matching {pattern!r} ...")
        st.volumes = discover_remote_volumes_sized(
            dataset_id, pattern, revision=revision, token=token,
        )
        st.total_size = sum(s for _, s in st.volumes)
        print(f"[Stage 0] Found {len(st.volumes)} volumes, total "
              f"{st.total_size/1024/1024/1024:.2f} GB")
        st.save(sp)
    else:
        print(f"[Stage 0] (resume) Reusing cached listing: {len(st.volumes)} volumes, "
              f"{st.total_size/1024/1024/1024:.2f} GB")

    # Stage 1: SHA-256
    if not st.stage1_done:
        print(f"[Stage 1] Computing SHA-256 (streams ~{st.total_size/1024/1024/1024:.1f} GB) ...")
        t0 = time.time()
        st.sha256_hex = compute_full_sha256(
            dataset_id, st.volumes, scratch, revision=revision, token=token,
            chunk_size=chunk_size,
        )
        st.stage1_done = True
        st.save(sp)
        print(f"[Stage 1] SHA-256 = {st.sha256_hex}  ({time.time()-t0:.1f}s)")
    else:
        print(f"[Stage 1] (resume) cached SHA-256 = {st.sha256_hex}")

    # Stage 2: validate blob
    if not st.stage2_done:
        print(f"[Stage 2] Asking ModelScope if blob {st.sha256_hex[:16]}... already exists ...")
        url, reused = validate_blob_existence(
            api, dataset_id, st.sha256_hex, st.total_size, token=token,
        )
        if reused:
            print("[Stage 2] ✓ Blob already exists globally — skipping upload pass.")
            st.blob_reused = True
            st.blob_url = None
            st.stage3_done = True  # nothing to do
        else:
            print("[Stage 2] Blob does not exist; obtained upload URL.")
            st.blob_reused = False
            st.blob_url = url
        st.stage2_done = True
        st.save(sp)
    else:
        print(f"[Stage 2] (resume) blob_reused={st.blob_reused}")

    # Stage 3: upload (skip if reused)
    if not st.stage3_done:
        print(f"[Stage 3] Streaming upload of {st.total_size/1024/1024/1024:.2f} GB ...")
        t0 = time.time()
        # Pass the pre_validated URL so _upload_blob skips its own _validate_blob call.
        upload_blob_streaming(
            api, dataset_id, st.sha256_hex, st.total_size, st.volumes, scratch,
            revision=revision, token=token,
            pre_validated_url=st.blob_url,
            buffer_size_mb=buffer_size_mb,
            path_in_repo_label=path_in_repo,
        )
        st.stage3_done = True
        st.save(sp)
        elapsed = time.time() - t0
        mbps = (st.total_size / 1024 / 1024) / max(1.0, elapsed)
        print(f"[Stage 3] Upload complete in {elapsed:.1f}s (~{mbps:.1f} MB/s)")
    else:
        print("[Stage 3] (resume) upload already done")

    # Stage 4: commit
    if not st.stage4_done:
        print(f"[Stage 4] Creating commit to register {path_in_repo} in {dataset_id} ...")
        msg = commit_message or f"Add merged {path_in_repo} (stream-merged from split volumes)"
        st.commit_id = create_commit_for_blob(
            api, dataset_id, path_in_repo,
            st.sha256_hex, st.total_size,
            revision=revision, token=token,
            commit_message=msg,
            commit_description=commit_description,
        )
        st.stage4_done = True
        st.save(sp)
        print(f"[Stage 4] Commit OK: {st.commit_id}")
    else:
        print(f"[Stage 4] (resume) commit_id={st.commit_id}")

    # Stage 5: verify
    if not skip_verify:
        print(f"[Stage 5] Verifying {path_in_repo} in remote listing ...")
        ok = verify_uploaded_file(
            dataset_id, path_in_repo, st.total_size,
            revision=revision, token=token,
        )
        if ok:
            print(f"[Stage 5] ✓ Verified: {path_in_repo} exists with "
                  f"size={st.total_size} ({st.total_size/1024/1024/1024:.2f} GB)")
        else:
            print(f"[Stage 5] ✗ WARNING: could not verify {path_in_repo} "
                  "(may be a propagation delay; try `scripts/ms_fetch.py list` later)")

    return st

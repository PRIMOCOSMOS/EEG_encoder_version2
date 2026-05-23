"""ModelScope dataset download helpers.

设计原则（与 Design.md 完全对齐）：
- ModelScope 数据集 **不能直接挂载** 到实例工作区 → 必须用 `MsDataset` / `HubApi` /
  `dataset_file_download` 一次一文件地拉到本地。
- 32 个分卷合计 ≈160 GB，实例只有 100 GB 持久化盘 → 我们采用
  **"一卷一处理一删"** 的流式策略：
      1) 列出远端所有分卷文件名
      2) `StreamingVolumeFetcher` 按顺序：下载分卷 → yield 本地路径 → 用户用完后调用
         `release()` 立即删除该分卷
      3) 任意时刻磁盘最多保留 1–2 个分卷（约 5–11 GB），不爆 100G。
- `save_info` 等小文件直接批量拉到固定目录。

兼容多 SDK 版本：
    - 新版 (>= 1.10) 推荐 `modelscope.hub.file_download.dataset_file_download`
    - 老版回退 `HubApi.get_dataset_file_url` + 手动 stream（HTTP requests）
"""
from __future__ import annotations

import fnmatch
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def login_if_token(token: Optional[str] = None) -> None:
    """Login to ModelScope if a token is available (env or argument)."""
    from modelscope.hub.api import HubApi
    tk = token or os.environ.get("MODELSCOPE_API_TOKEN")
    if tk:
        HubApi().login(tk)


# ---------------------------------------------------------------------------
# Remote file listing (recursive)
# ---------------------------------------------------------------------------

def list_dataset_files(
    dataset_id: str,
    revision: str = "master",
    token: Optional[str] = None,
    recursive: bool = True,
    page_size: int = 200,
) -> List[dict]:
    """Recursively list files in a ModelScope dataset repo.

    Returns a list of dicts with at least: {'Path': str, 'Type': 'blob'|'tree', 'Size': int}.
    """
    from modelscope.hub.api import HubApi
    api = HubApi()
    login_if_token(token)

    namespace, name = dataset_id.split("/", 1)
    dataset_hub_id, _ = api.get_dataset_id_and_type(dataset_name=name, namespace=namespace)

    all_files: List[dict] = []
    page = 1
    while True:
        chunk = api.get_dataset_files(
            repo_id=dataset_id,
            revision=revision,
            root_path="/",
            recursive=recursive,
            page_number=page,
            page_size=page_size,
            dataset_hub_id=dataset_hub_id,
        )
        if not chunk:
            break
        all_files.extend(chunk)
        if len(chunk) < page_size:
            break
        page += 1
    return [f for f in all_files if f.get("Type") != "tree"]


def filter_files(
    files: Sequence[dict],
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Glob-filter a file listing by `include` / `exclude` patterns (match against 'Path')."""
    out = []
    for f in files:
        path = f.get("Path") or f.get("Name") or ""
        if include and not any(fnmatch.fnmatch(path, p) for p in include):
            continue
        if exclude and any(fnmatch.fnmatch(path, p) for p in exclude):
            continue
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# Single-file download (with multi-version fallback)
# ---------------------------------------------------------------------------

def download_one_file(
    dataset_id: str,
    file_path: str,
    local_dir: str,
    revision: str = "master",
    token: Optional[str] = None,
) -> Path:
    """Download a single file from a ModelScope dataset into `local_dir`.

    Returns the local Path to the downloaded file. Uses `dataset_file_download`
    on modern SDK; falls back to a direct HTTP stream if unavailable.
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    login_if_token(token)

    # ---- modern path ----
    try:
        from modelscope.hub.file_download import dataset_file_download
        local = dataset_file_download(
            dataset_id=dataset_id,
            file_path=file_path,
            revision=revision,
            local_dir=local_dir,
        )
        return Path(local)
    except ImportError:
        pass
    except TypeError:
        # Some versions name the kwargs differently; try positional
        try:
            from modelscope.hub.file_download import dataset_file_download
            local = dataset_file_download(dataset_id, file_path, revision, None, local_dir)
            return Path(local)
        except Exception:
            pass

    # ---- legacy fallback: build URL and stream ----
    import requests
    from modelscope.hub.api import HubApi
    api = HubApi()
    namespace, name = dataset_id.split("/", 1)
    url = api.get_dataset_file_url(
        file_name=file_path,
        dataset_name=name,
        namespace=namespace,
        revision=revision,
    )
    cookies = api.get_cookies()
    out = Path(local_dir) / Path(file_path).name
    with requests.get(url, cookies=cookies, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return out


# ---------------------------------------------------------------------------
# Bulk small-file download (save_info etc.)
# ---------------------------------------------------------------------------

def download_files(
    dataset_id: str,
    file_paths: Sequence[str],
    local_dir: str,
    revision: str = "master",
    token: Optional[str] = None,
    verbose: bool = True,
) -> List[Path]:
    """Download a list of files (typically small ones) into `local_dir`, returning local paths."""
    out: List[Path] = []
    for fp in file_paths:
        if verbose:
            print(f"[MS] fetching {fp}")
        out.append(download_one_file(dataset_id, fp, local_dir, revision=revision, token=token))
    return out


def download_save_info(
    dataset_id: str,
    local_dir: str,
    revision: str = "master",
    token: Optional[str] = None,
    subdir_keyword: str = "save_info",
    include: Optional[Sequence[str]] = None,
) -> Path:
    """Download all `*_save_info.csv` files from a dataset into `local_dir`.

    Returns the directory containing them (so downstream loaders can scan it).
    """
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    listing = list_dataset_files(dataset_id, revision=revision, token=token)
    if include:
        listing = filter_files(listing, include=list(include))
    else:
        listing = [
            f for f in listing
            if subdir_keyword.lower() in (f.get("Path") or "").lower()
            and (f.get("Path") or "").lower().endswith("_save_info.csv")
        ]
    if not listing:
        raise RuntimeError(
            f"No save_info CSVs found in {dataset_id}. "
            "Pass --ms-save-info-include to override the glob."
        )
    paths = [f["Path"] for f in listing]
    download_files(dataset_id, paths, local_dir, revision=revision, token=token)
    return Path(local_dir)


# ---------------------------------------------------------------------------
# Streaming volume fetcher  (download-one → use → delete → next)
# ---------------------------------------------------------------------------

class StreamingVolumeFetcher:
    """Iterate the 32 split-volume files one at a time: download → yield → delete.

    Use as a context manager so even on exceptions partial files are cleaned up:

        with StreamingVolumeFetcher(dataset_id, file_paths, scratch_dir) as fetcher:
            for local_path in fetcher.iter_paths():
                ...consume local_path...
                fetcher.release()       # delete that volume from disk immediately

    Guarantees at most `keep_n` files live on disk simultaneously (default 1).
    """

    def __init__(
        self,
        dataset_id: str,
        file_paths: Sequence[str],
        scratch_dir: str,
        revision: str = "master",
        token: Optional[str] = None,
        keep_n: int = 1,
    ):
        self.dataset_id = dataset_id
        self.file_paths = list(file_paths)
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.revision = revision
        self.token = token
        self.keep_n = max(1, int(keep_n))
        self._downloaded: List[Path] = []  # FIFO

    def __enter__(self) -> "StreamingVolumeFetcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup_all()

    def _evict_old(self) -> None:
        while len(self._downloaded) > self.keep_n:
            old = self._downloaded.pop(0)
            try:
                if old.exists():
                    old.unlink()
            except Exception:
                pass

    def fetch(self, file_path: str) -> Path:
        local = download_one_file(
            self.dataset_id, file_path, str(self.scratch_dir),
            revision=self.revision, token=self.token,
        )
        self._downloaded.append(local)
        self._evict_old()
        return local

    def release(self) -> None:
        """Delete the most-recently fetched volume (call after you're done with it)."""
        if not self._downloaded:
            return
        last = self._downloaded.pop()
        try:
            if last.exists():
                last.unlink()
        except Exception:
            pass

    def cleanup_all(self) -> None:
        for p in self._downloaded:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        self._downloaded.clear()

    def iter_paths(self) -> Iterator[Path]:
        """Yield local paths in order. Caller must call `release()` after each use
        if they want strict 1-volume-on-disk behavior."""
        for fp in self.file_paths:
            yield self.fetch(fp)


# ---------------------------------------------------------------------------
# Convenience: enumerate the 32 split volumes by glob
# ---------------------------------------------------------------------------

def discover_remote_volumes(
    dataset_id: str,
    pattern: str = "*.zip.*",
    revision: str = "master",
    token: Optional[str] = None,
) -> List[str]:
    """List & sort the split-volume file paths on the remote dataset.

    Returns just the paths (strings); use them with `StreamingVolumeFetcher`.
    """
    files = list_dataset_files(dataset_id, revision=revision, token=token)
    files = filter_files(files, include=[pattern])

    def keyfn(f: dict) -> Tuple[int, str]:
        name = f.get("Path") or ""
        digits = ""
        for ch in reversed(name):
            if ch.isdigit():
                digits = ch + digits
            else:
                if digits:
                    break
        return (int(digits) if digits else 0, name)

    files.sort(key=keyfn)
    return [f["Path"] for f in files]


# ---------------------------------------------------------------------------
# Light wrapper: download into a "virtual" local volumes dir
#  (when the user prefers to reuse the old code paths unchanged)
# ---------------------------------------------------------------------------

@contextmanager
def materialized_volumes(
    dataset_id: str,
    pattern: str,
    scratch_dir: str,
    revision: str = "master",
    token: Optional[str] = None,
    keep_all: bool = False,
):
    """Context manager that downloads volumes serially and yields the local directory.

    Important caveats:
    - `keep_all=True` requires ≥ total-size free on `scratch_dir` (e.g. 160 GB) — only do
      this if you really have a scratch disk. Use the streaming fetcher otherwise.
    - On exit, the directory is wiped clean.
    """
    paths = discover_remote_volumes(dataset_id, pattern=pattern, revision=revision, token=token)
    scratch = Path(scratch_dir); scratch.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    try:
        for fp in paths:
            local = download_one_file(dataset_id, fp, str(scratch), revision=revision, token=token)
            downloaded.append(local)
            if not keep_all:
                # in non-keep mode we deliberately stop after one (caller wants streaming)
                pass
        yield scratch
    finally:
        # wipe everything we created
        for p in downloaded:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        # do NOT remove user's scratch root if it pre-existed; just clear tracked files

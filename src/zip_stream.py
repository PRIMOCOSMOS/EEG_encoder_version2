"""Streaming readers over multi-volume split ZIPs.

设计原则（来自 Design.md）：
- 32 个分卷（每个 5.37GB，合计 ≈160GB），ModelScope 实例只有 100GB 持久化盘。
- **绝不** 一次性把完整 zip 解压到磁盘。
- 两种 reader：
    1) `ConcatStream`        —— 本地已经有所有分卷文件时使用（虚拟拼接 + zipfile）
    2) `LazyConcatStream`    —— 分卷在远端 ModelScope 上，**按需下载切片** + **LRU 释放**
                                 → 磁盘任意时刻只保留 1–2 个分卷（~5–11 GB）。
- 两者都封装成 `io.RawIOBase` 提供给 `zipfile.ZipFile`；zipfile 通过 central directory 做随机寻址，
  只会读取被请求 `.mat` 涉及的字节段。
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple


@dataclass
class _Part:
    path: Path
    size: int
    offset: int   # global offset where this part starts


# =============================================================================
# 1) Local concatenation stream (all volumes already on disk)
# =============================================================================

class ConcatStream(io.RawIOBase):
    """Read-only seekable file-like view over a list of files concatenated in order."""

    def __init__(self, paths: Sequence[Path]):
        self._parts: List[_Part] = []
        offset = 0
        for p in paths:
            p = Path(p)
            if not p.is_file():
                raise FileNotFoundError(p)
            size = p.stat().st_size
            self._parts.append(_Part(path=p, size=size, offset=offset))
            offset += size
        self._total = offset
        self._pos = 0
        self._cur_idx: int = -1
        self._cur_fh: Optional[io.BufferedReader] = None

    # ---- io.RawIOBase API ----
    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def writable(self) -> bool: return False
    def tell(self) -> int: return self._pos

    @property
    def total_size(self) -> int:
        return self._total

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self._total + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if new < 0:
            raise ValueError(f"Negative seek: {new}")
        self._pos = new
        return self._pos

    def _locate(self, pos: int) -> int:
        for i, part in enumerate(self._parts):
            if part.offset <= pos < part.offset + part.size:
                return i
        return len(self._parts)

    def _ensure_fh(self, idx: int) -> Optional[io.BufferedReader]:
        if idx >= len(self._parts):
            return None
        if self._cur_idx != idx:
            if self._cur_fh is not None:
                self._cur_fh.close()
                self._cur_fh = None
            self._cur_fh = open(self._parts[idx].path, "rb")
            self._cur_idx = idx
        return self._cur_fh

    def readinto(self, b) -> int:  # type: ignore[override]
        if self._pos >= self._total:
            return 0
        n_wanted = len(b)
        n_got = 0
        mv = memoryview(b)
        while n_got < n_wanted and self._pos < self._total:
            idx = self._locate(self._pos)
            fh = self._ensure_fh(idx)
            if fh is None:
                break
            part = self._parts[idx]
            local_off = self._pos - part.offset
            fh.seek(local_off)
            remain = part.size - local_off
            to_read = min(n_wanted - n_got, remain)
            buf = fh.read(to_read)
            if not buf:
                break
            mv[n_got:n_got + len(buf)] = buf
            n_got += len(buf)
            self._pos += len(buf)
        return n_got

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        if size is None or size < 0:
            size = self._total - self._pos
        buf = bytearray(size)
        n = self.readinto(buf)
        return bytes(buf[:n])

    def close(self) -> None:
        if self._cur_fh is not None:
            try: self._cur_fh.close()
            except Exception: pass
            self._cur_fh = None
        super().close()


# =============================================================================
# 2) Lazy/streaming concatenation over REMOTE volumes (ModelScope etc.)
# =============================================================================

@dataclass
class _RemotePart:
    name: str                 # remote path (e.g. 'volumes/SEED-VII.zip.012')
    size: int                 # known size in bytes (must be supplied upfront)
    offset: int               # global offset

    local_path: Optional[Path] = None     # set after fetch
    last_used: float = 0.0                # for LRU


class LazyConcatStream(io.RawIOBase):
    """`ConcatStream` cousin that fetches each split volume on demand and evicts old ones.

    Constructor takes:
        sizes_in_order : list[tuple[remote_name, size_bytes]]
        fetcher        : Callable[remote_name -> local_path_str]   (downloader)
        evicter        : Callable[local_path -> None]              (e.g. os.unlink)
        max_resident   : int                                       (how many volumes to keep)

    Reading bytes that fall inside an evicted volume re-triggers `fetcher`. So pure
    sequential workloads (like zipfile reading whole .mat members in order) keep only
    `max_resident` volumes on disk at any time.
    """

    def __init__(
        self,
        sizes_in_order: Sequence[Tuple[str, int]],
        fetcher: Callable[[str], str],
        evicter: Optional[Callable[[Path], None]] = None,
        max_resident: int = 2,
    ):
        self._parts: List[_RemotePart] = []
        off = 0
        for name, sz in sizes_in_order:
            self._parts.append(_RemotePart(name=name, size=int(sz), offset=off))
            off += int(sz)
        self._total = off
        self._pos = 0

        self._fetcher = fetcher
        self._evicter = evicter or (lambda p: p.unlink(missing_ok=True))
        self._max_resident = max(1, int(max_resident))

        self._cur_idx: int = -1
        self._cur_fh: Optional[io.BufferedReader] = None

        # LRU clock
        import itertools
        self._clock = itertools.count()

    # ---- io.RawIOBase ----
    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def writable(self) -> bool: return False
    def tell(self) -> int: return self._pos

    @property
    def total_size(self) -> int: return self._total

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET: new = offset
        elif whence == io.SEEK_CUR: new = self._pos + offset
        elif whence == io.SEEK_END: new = self._total + offset
        else: raise ValueError(f"Invalid whence: {whence}")
        if new < 0: raise ValueError("negative seek")
        self._pos = new
        return self._pos

    def _locate(self, pos: int) -> int:
        # short list (<=64), linear scan is fine
        for i, part in enumerate(self._parts):
            if part.offset <= pos < part.offset + part.size:
                return i
        return len(self._parts)

    def _maybe_evict(self) -> None:
        resident = [p for p in self._parts if p.local_path is not None]
        if len(resident) <= self._max_resident:
            return
        # don't evict the one currently open
        cur_part = self._parts[self._cur_idx] if 0 <= self._cur_idx < len(self._parts) else None
        candidates = [p for p in resident if p is not cur_part]
        candidates.sort(key=lambda p: p.last_used)
        while len(resident) > self._max_resident and candidates:
            victim = candidates.pop(0)
            try:
                if victim.local_path is not None and victim.local_path.exists():
                    self._evicter(victim.local_path)
            except Exception:
                pass
            victim.local_path = None
            resident = [p for p in self._parts if p.local_path is not None]

    def _ensure_resident(self, idx: int) -> Path:
        part = self._parts[idx]
        if part.local_path is None or not part.local_path.exists():
            local_str = self._fetcher(part.name)
            part.local_path = Path(local_str)
        part.last_used = float(next(self._clock))
        self._maybe_evict()
        return part.local_path

    def _ensure_fh(self, idx: int) -> Optional[io.BufferedReader]:
        if idx >= len(self._parts):
            return None
        if self._cur_idx != idx or self._cur_fh is None:
            if self._cur_fh is not None:
                try: self._cur_fh.close()
                except Exception: pass
                self._cur_fh = None
            local = self._ensure_resident(idx)
            self._cur_fh = open(local, "rb")
            self._cur_idx = idx
        return self._cur_fh

    def readinto(self, b) -> int:  # type: ignore[override]
        if self._pos >= self._total:
            return 0
        n_wanted = len(b)
        n_got = 0
        mv = memoryview(b)
        while n_got < n_wanted and self._pos < self._total:
            idx = self._locate(self._pos)
            fh = self._ensure_fh(idx)
            if fh is None:
                break
            part = self._parts[idx]
            local_off = self._pos - part.offset
            fh.seek(local_off)
            remain = part.size - local_off
            to_read = min(n_wanted - n_got, remain)
            buf = fh.read(to_read)
            if not buf:
                break
            mv[n_got:n_got + len(buf)] = buf
            n_got += len(buf)
            self._pos += len(buf)
        return n_got

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        if size is None or size < 0:
            size = self._total - self._pos
        buf = bytearray(size)
        n = self.readinto(buf)
        return bytes(buf[:n])

    def close(self) -> None:
        if self._cur_fh is not None:
            try: self._cur_fh.close()
            except Exception: pass
            self._cur_fh = None
        # evict ALL on close
        for p in self._parts:
            try:
                if p.local_path is not None and p.local_path.exists():
                    self._evicter(p.local_path)
            except Exception:
                pass
            p.local_path = None
        super().close()


# =============================================================================
# Discovery & high-level helpers
# =============================================================================

def discover_volumes(volumes_dir: os.PathLike, pattern: str = "*.zip.*") -> List[Path]:
    """Find and sort local split volume files by trailing digits."""
    d = Path(volumes_dir)
    parts = sorted(d.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"No volumes matched {pattern} under {d}")
    def keyfn(p: Path) -> Tuple[int, str]:
        name = p.name
        digits = ""
        for ch in reversed(name):
            if ch.isdigit():
                digits = ch + digits
            else:
                if digits:
                    break
        return (int(digits) if digits else 0, name)
    parts.sort(key=keyfn)
    return parts


def open_concat_zip(volumes_dir: os.PathLike, pattern: str = "*.zip.*") -> zipfile.ZipFile:
    """Open the local-volumes concatenated zip as a single ZipFile (read-only)."""
    parts = discover_volumes(volumes_dir, pattern)
    stream = ConcatStream(parts)
    return zipfile.ZipFile(stream, mode="r")


def open_remote_concat_zip(
    sizes_in_order: Sequence[Tuple[str, int]],
    fetcher: Callable[[str], str],
    evicter: Optional[Callable[[Path], None]] = None,
    max_resident: int = 2,
) -> zipfile.ZipFile:
    """Open a ModelScope multi-volume zip as a single ZipFile, with lazy fetching."""
    stream = LazyConcatStream(
        sizes_in_order=sizes_in_order,
        fetcher=fetcher,
        evicter=evicter,
        max_resident=max_resident,
    )
    return zipfile.ZipFile(stream, mode="r")


def iter_mat_members(
    zf: zipfile.ZipFile,
    subdir_keyword: str = "EEG_preprocessed",
) -> Iterator[zipfile.ZipInfo]:
    """Iterate ZipInfo entries that look like EEG_preprocessed/*.mat files."""
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        low = name.lower()
        if not low.endswith(".mat"):
            continue
        if subdir_keyword and subdir_keyword.lower() not in low:
            continue
        yield info


def extract_mat_bytes(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Read the raw bytes of a single .mat member (held in memory transiently)."""
    with zf.open(info, "r") as fh:
        return fh.read()

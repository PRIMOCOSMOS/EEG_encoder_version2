"""Streaming readers over byte-spliced multi-volume ZIPs.

重要前提（来自用户实际制作分卷的方式）：
    用户的 32 个 "*.zip.NNN" 分卷是对**原始 zip 文件**做 **字节级顺序切片**
    (固定大小 split，与 `split -b` / `dd` 等价)；它们**不是** zip 自身的
    多卷 spanning 格式 (.z01/.z02/.../.zip)。所以：
        - 按序字节拼接 == 一个合法 zip
        - 单独一个分卷 == 不是合法 zip，不能 zipfile.ZipFile(...) 单开
        - 文件 > 4 GB 的合并体必然是 ZIP64

设计原则（来自 Design.md）：
- 32 个分卷（每个 5.37GB，合计 ≈160GB），ModelScope 实例只有 100GB 持久化盘。
- **绝不** 把完整 zip 解压到磁盘。
- 两种 reader：
    1) `ConcatStream`        —— 所有分卷已在本地，直接虚拟拼接 + zipfile
    2) `LazyConcatStream`    —— 分卷在远端 ModelScope，按需下载 + LRU 释放 + 持久 pin
                                + 末卷预热（避免 zipfile.ZipFile(...) 一打开就触发末卷下载）
- 两者都封装成 `io.RawIOBase`；`zipfile` 通过 EOCD/central-directory 做随机寻址，
  我们只在被请求的字节区段所在的分卷上做物理下载。
"""
from __future__ import annotations

import io
import itertools
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Set, Tuple


# =============================================================================
# Volume sort key (rugged)
# =============================================================================

_DIGIT_RUN_RE = re.compile(r"(\d+)")


def volume_sort_key(name: str) -> Tuple[int, str]:
    """Return a sort key that uses the **last** run of digits in the basename.

    Robust to:
        SEED-VII.zip.001 / .zip.032
        SEED-VII.zip.1   / SEED-VII.zip.32
        SEED-VII.part01.zip / .part32.zip
        SEED-VII_001.bin / SEED-VII_032.bin
        SEED-VII.z01 / SEED-VII.zip  (NOT this user's case, but tolerated)
    """
    base = os.path.basename(name)
    runs = _DIGIT_RUN_RE.findall(base)
    if not runs:
        return (10 ** 12, base)
    # use the LAST numeric run — that's the volume index in all sane splitters
    return (int(runs[-1]), base)


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
    name: str                            # remote path (e.g. 'volumes/SEED-VII.zip.012')
    size: int                            # known size in bytes (must be supplied upfront)
    offset: int                          # global offset
    local_path: Optional[Path] = None    # set after fetch
    last_used: float = 0.0               # for LRU
    pinned: bool = False                 # if True, never evicted by LRU


class LazyConcatStream(io.RawIOBase):
    """`ConcatStream` cousin that fetches each split volume on demand and evicts old ones.

    Constructor takes:
        sizes_in_order : list[tuple[remote_name, size_bytes]]
        fetcher        : Callable[remote_name -> local_path_str]
        evicter        : Callable[local_path -> None]
        max_resident   : int   (how many non-pinned volumes to keep)

    Pinning:
        Use `pin(idx)` to mark a volume as never-evictable. We always pin the **last
        volume** during `__init__` so that opening with `zipfile.ZipFile(...)` (which
        seeks to end-of-file to find the EOCD/central-directory) doesn't keep
        re-downloading the tail.

    Disk footprint:
        len(pinned) + max_resident volumes (default: 1 pinned + 2 LRU = 3 volumes ≈ 16 GB).
    """

    def __init__(
        self,
        sizes_in_order: Sequence[Tuple[str, int]],
        fetcher: Callable[[str], str],
        evicter: Optional[Callable[[Path], None]] = None,
        max_resident: int = 2,
        pin_last: bool = True,
        warmup_last: bool = True,
    ):
        if not sizes_in_order:
            raise ValueError("sizes_in_order is empty")

        self._parts: List[_RemotePart] = []
        off = 0
        for name, sz in sizes_in_order:
            sz = int(sz)
            if sz <= 0:
                raise ValueError(
                    f"Volume {name!r} has invalid size {sz}. "
                    "Sizes must be known upfront (LazyConcatStream cannot do HEAD probing)."
                )
            self._parts.append(_RemotePart(name=name, size=sz, offset=off))
            off += sz
        self._total = off
        self._pos = 0

        self._fetcher = fetcher
        self._evicter = evicter or (lambda p: p.unlink(missing_ok=True))
        self._max_resident = max(1, int(max_resident))

        self._cur_idx: int = -1
        self._cur_fh: Optional[io.BufferedReader] = None
        self._clock = itertools.count()

        # ---- KEY FIX: pre-fetch the LAST volume so that zipfile's EOCD seek
        # doesn't trigger a download/evict cycle on every open() ----
        if pin_last and len(self._parts) > 0:
            self.pin(len(self._parts) - 1, fetch_now=bool(warmup_last))

    # ---- io.RawIOBase ----
    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def writable(self) -> bool: return False
    def tell(self) -> int: return self._pos

    @property
    def total_size(self) -> int:
        return self._total

    @property
    def num_parts(self) -> int:
        return len(self._parts)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET: new = offset
        elif whence == io.SEEK_CUR: new = self._pos + offset
        elif whence == io.SEEK_END: new = self._total + offset
        else: raise ValueError(f"Invalid whence: {whence}")
        if new < 0:
            raise ValueError("negative seek")
        self._pos = new
        return self._pos

    def _locate(self, pos: int) -> int:
        # short list (<=64), linear scan is fine
        for i, part in enumerate(self._parts):
            if part.offset <= pos < part.offset + part.size:
                return i
        return len(self._parts)

    # ---- pinning ----
    def pin(self, idx: int, fetch_now: bool = True) -> Optional[Path]:
        """Mark a volume as never-evictable. Optionally fetch it immediately."""
        if not (0 <= idx < len(self._parts)):
            raise IndexError(idx)
        part = self._parts[idx]
        part.pinned = True
        if fetch_now:
            return self._ensure_resident(idx)
        return part.local_path

    def unpin(self, idx: int, evict_now: bool = False) -> None:
        if not (0 <= idx < len(self._parts)):
            raise IndexError(idx)
        part = self._parts[idx]
        part.pinned = False
        if evict_now and part.local_path is not None:
            try:
                if part.local_path.exists():
                    self._evicter(part.local_path)
            except Exception:
                pass
            part.local_path = None

    def pinned_indices(self) -> List[int]:
        return [i for i, p in enumerate(self._parts) if p.pinned]

    def resident_indices(self) -> List[int]:
        return [i for i, p in enumerate(self._parts) if p.local_path is not None]

    # ---- eviction ----
    def _maybe_evict(self, protect_idx: Optional[int] = None) -> None:
        """Evict oldest non-pinned residents until we are at most
        `max_resident` non-pinned residents.

        Always protects:
          - pinned parts
          - the currently-opened part (self._cur_idx)
          - optionally, `protect_idx` (used by callers that just fetched but
            haven't yet updated self._cur_idx)
        """
        protected: set = set()
        if 0 <= self._cur_idx < len(self._parts):
            protected.add(self._cur_idx)
        if protect_idx is not None and 0 <= protect_idx < len(self._parts):
            protected.add(protect_idx)

        non_pinned_resident = [
            (i, p) for i, p in enumerate(self._parts)
            if p.local_path is not None and not p.pinned and i not in protected
        ]
        # cur_counts: how many protected parts count toward the budget (non-pinned)
        cur_counts = sum(
            1 for i in protected
            if 0 <= i < len(self._parts)
            and self._parts[i].local_path is not None
            and not self._parts[i].pinned
        )
        allowed = self._max_resident - cur_counts
        if len(non_pinned_resident) <= max(0, allowed):
            return
        non_pinned_resident.sort(key=lambda t: t[1].last_used)
        while len(non_pinned_resident) > max(0, allowed):
            _, victim = non_pinned_resident.pop(0)
            try:
                if victim.local_path is not None and victim.local_path.exists():
                    self._evicter(victim.local_path)
            except Exception:
                pass
            victim.local_path = None

    def _ensure_resident(self, idx: int) -> Path:
        part = self._parts[idx]
        if part.local_path is None or not part.local_path.exists():
            local_str = self._fetcher(part.name)
            part.local_path = Path(local_str)
            if part.local_path.stat().st_size != part.size:
                # don't crash hard; warn — some endpoints report compressed size
                import warnings
                warnings.warn(
                    f"[LazyConcatStream] volume {part.name!r} size mismatch: "
                    f"expected {part.size}, got {part.local_path.stat().st_size}",
                    RuntimeWarning,
                )
        part.last_used = float(next(self._clock))
        # Capture path BEFORE eviction (in case `idx` itself becomes a victim because
        # `self._cur_idx` is not yet updated by the caller).
        local = part.local_path
        # Evict everything else, but protect the just-touched part too.
        self._maybe_evict(protect_idx=idx)
        # If eviction nulled it out (e.g. max_resident=0 misconfiguration), restore.
        if part.local_path is None:
            part.local_path = local
        return local

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
        # evict ALL (including pinned) on close
        for p in self._parts:
            try:
                if p.local_path is not None and p.local_path.exists():
                    self._evicter(p.local_path)
            except Exception:
                pass
            p.local_path = None
            p.pinned = False
        super().close()


# =============================================================================
# Discovery & high-level helpers
# =============================================================================

def discover_volumes(volumes_dir: os.PathLike, pattern: str = "*.zip.*") -> List[Path]:
    """Find and sort local split volume files by trailing-digit run."""
    d = Path(volumes_dir)
    parts = sorted(d.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"No volumes matched {pattern} under {d}")
    parts.sort(key=lambda p: volume_sort_key(p.name))
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
    pin_last: bool = True,
    warmup_last: bool = True,
) -> Tuple[zipfile.ZipFile, LazyConcatStream]:
    """Open a multi-volume zip as a single ZipFile, with lazy fetching.

    Returns (ZipFile, underlying_stream) so callers can pin/unpin extra volumes.
    """
    stream = LazyConcatStream(
        sizes_in_order=sizes_in_order,
        fetcher=fetcher,
        evicter=evicter,
        max_resident=max_resident,
        pin_last=pin_last,
        warmup_last=warmup_last,
    )
    zf = zipfile.ZipFile(stream, mode="r")
    return zf, stream


# =============================================================================
# Member iteration / extraction helpers
# =============================================================================

def _norm_path(name: str) -> str:
    return name.replace("\\", "/").lower()


def iter_mat_members(
    zf: zipfile.ZipFile,
    subdir_keyword: str = "EEG_preprocessed",
) -> Iterator[zipfile.ZipInfo]:
    """Iterate ZipInfo entries that look like EEG_preprocessed/*.mat files.

    Case-insensitive, slash-normalized, handles `subdir_keyword=""` to mean "any .mat".
    """
    kw = subdir_keyword.lower() if subdir_keyword else ""
    for info in zf.infolist():
        if info.is_dir():
            continue
        norm = _norm_path(info.filename)
        if not norm.endswith(".mat"):
            continue
        if kw and kw not in norm:
            continue
        yield info


def extract_mat_bytes(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Read the raw bytes of a single .mat member (held in memory transiently)."""
    with zf.open(info, "r") as fh:
        return fh.read()


# =============================================================================
# Volume-aware member grouping (minimizes cross-volume churn)
# =============================================================================

@dataclass
class _MemberLocale:
    info: zipfile.ZipInfo
    start_offset: int        # local file header offset (global, in the concat stream)
    end_offset: int          # exclusive
    start_part: int          # part index containing header
    end_part: int            # part index containing last byte


def locate_members_in_stream(
    zf: zipfile.ZipFile,
    members: Sequence[zipfile.ZipInfo],
    stream: LazyConcatStream,
) -> List[_MemberLocale]:
    """Compute which split volume(s) each member spans.

    Useful for scheduling reads in part-order so the LRU only needs to fault each
    volume in once (instead of bouncing around).
    """
    out: List[_MemberLocale] = []
    for info in members:
        # local-header offset within the (virtual) zip file
        hdr = int(info.header_offset)
        # data length estimate: local header (>=30 B) + filename + extra + compressed size
        approx_local_header = 30 + len(info.filename.encode("utf-8")) + len(info.extra or b"")
        data_len = int(getattr(info, "compress_size", info.file_size))
        end_excl = hdr + approx_local_header + data_len
        # locate parts
        s = stream._locate(hdr)
        e = stream._locate(max(hdr, end_excl - 1))
        out.append(_MemberLocale(
            info=info,
            start_offset=hdr, end_offset=end_excl,
            start_part=s, end_part=e,
        ))
    return out


def schedule_members_by_part(
    locales: Sequence[_MemberLocale],
) -> List[_MemberLocale]:
    """Sort members so we walk volumes monotonically (header part ASC, then offset ASC)."""
    return sorted(locales, key=lambda m: (m.start_part, m.start_offset))

"""Streaming reader over a multi-volume split ZIP.

设计原则（来自 Design.md）：
- 32 个分卷（每个 5.37GB，合计 ≈160GB），ModelScope 实例只有 100GB 持久化盘。
- **绝不** 一次性把完整 zip 解压到磁盘。
- 采用「虚拟拼接 + zipfile 顺序流式读取」：
    * `ConcatStream` 把 N 个 split 文件伪装成一个连续的可 seek 文件对象。
    * `zipfile.ZipFile` 直接打开它（zipfile 通过 central directory 找文件，O(1) 跳读）。
    * 每次只读取所需 `.mat` 的字节，处理完即释放。

注意：用户的 split 是「裸字节顺序拼接」（不是 zip 自带的多卷 .z01/.z02 spanning 格式），
所以拼接后等同于一个普通 zip。
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple


@dataclass
class _Part:
    path: Path
    size: int
    offset: int   # global offset where this part starts


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
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

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
        # binary search would be fine; linear ok since N<=64
        for i, part in enumerate(self._parts):
            if part.offset <= pos < part.offset + part.size:
                return i
        # EOF
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
            remain_in_part = part.size - local_off
            to_read = min(n_wanted - n_got, remain_in_part)
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
            try:
                self._cur_fh.close()
            except Exception:
                pass
            self._cur_fh = None
        super().close()


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def discover_volumes(volumes_dir: os.PathLike, pattern: str = "*.zip.*") -> List[Path]:
    """Find and sort split volume files by their numeric suffix.

    Supports common naming schemes, e.g.:
        SEED-VII.zip.001 ... SEED-VII.zip.032
        SEED-VII.z01 ... SEED-VII.z31, SEED-VII.zip
        SEED-VII.part01.rar (not supported here)
    """
    d = Path(volumes_dir)
    parts = sorted(d.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"No volumes matched {pattern} under {d}")
    # robust numeric sort by trailing digits
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
    """Open the concatenated multi-volume zip as a single ZipFile (read-only)."""
    parts = discover_volumes(volumes_dir, pattern)
    stream = ConcatStream(parts)
    # zipfile uses central directory; works fine on our seekable stream.
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

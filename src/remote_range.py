"""HTTP Range-based seekable stream for ModelScope-hosted files.

Use case: the merged ~160GB SEED-VII.zip is now a single file in the dataset.
We want to read it on the instance **without downloading it** to disk
(only ~100GB persistent disk available).

ModelScope routes file downloads via Aliyun OSS pre-signed URLs, which support
HTTP `Range:` requests. We wrap that in an `io.RawIOBase` so it can be passed
to `zipfile.ZipFile` exactly like our `ConcatStream` / `LazyConcatStream`.

Highlights:
- **Lazy URL resolution** (and re-resolution on expiry / 403)
- **Connection pooling** via a single requests.Session with retries
- **Auto retry** on transient 5xx / connection errors
- **In-memory LRU cache** of recently-read byte ranges (default ≤ 256 MB), so
  zipfile's repeated scans of central-directory + per-mat local-header reads
  don't trigger one HTTP round-trip per call
- **Disk usage: 0 bytes**. Memory usage: O(cache_size).
"""
from __future__ import annotations

import io
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry


_DEFAULT_TIMEOUT = (30, 600)  # (connect, read) seconds


def _make_session(max_retries: int = 5) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=2,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["HEAD", "GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


# =============================================================================
# Tiny LRU range cache (immutable byte ranges)
# =============================================================================

class _ByteRangeCache:
    """In-memory LRU cache mapping (start, end) -> bytes.

    We cache full GET-Range responses (chunk-aligned). Subsequent reads that
    fall entirely inside a cached chunk are served from RAM.
    """

    def __init__(self, max_bytes: int = 256 * 1024 * 1024):
        self.max_bytes = int(max_bytes)
        self._od: "OrderedDict[Tuple[int, int], bytes]" = OrderedDict()
        self._total = 0

    def _evict(self) -> None:
        while self._total > self.max_bytes and self._od:
            (_, v) = self._od.popitem(last=False)
            self._total -= len(v)

    def get(self, start: int, end_exclusive: int) -> Optional[bytes]:
        # find any cached chunk that covers [start, end)
        for (cs, ce), data in self._od.items():
            if cs <= start and end_exclusive <= ce:
                self._od.move_to_end((cs, ce))
                return data[start - cs : end_exclusive - cs]
        return None

    def put(self, start: int, end_exclusive: int, data: bytes) -> None:
        key = (start, end_exclusive)
        if key in self._od:
            self._od.move_to_end(key)
            return
        self._od[key] = data
        self._total += len(data)
        self._evict()

    def clear(self) -> None:
        self._od.clear()
        self._total = 0


# =============================================================================
# RemoteRangeStream
# =============================================================================

@dataclass
class _UrlState:
    url: str
    expires_at: float = 0.0   # 0 means unknown — never auto-expire on time


class RemoteRangeStream(io.RawIOBase):
    """Seekable, read-only view of a remote file served by HTTP Range.

    Args:
        url_provider:  callable() -> str   (refreshable URL)
        total_size:    file size in bytes (must be known upfront)
        session:       requests.Session (a default one with retries is created if None)
        chunk_size:    size of each Range GET (also the cache chunk size)
        cache_bytes:   max bytes held in the in-memory LRU range cache
        cookies:       optional cookies for auth (e.g. m_session_id)
        headers:       optional extra headers
        url_ttl_seconds: if >0, refresh the URL after this many seconds even
                       without a 403 (pre-signed URLs typically last 1 hour)
    """

    def __init__(
        self,
        url_provider: Callable[[], str],
        total_size: int,
        session: Optional[requests.Session] = None,
        chunk_size: int = 8 * 1024 * 1024,        # 8 MB per HTTP fetch
        cache_bytes: int = 256 * 1024 * 1024,     # 256 MB LRU cache
        cookies=None,
        headers: Optional[dict] = None,
        url_ttl_seconds: float = 1800.0,          # refresh URL every 30 min
        timeout=_DEFAULT_TIMEOUT,
    ):
        if total_size <= 0:
            raise ValueError(f"total_size must be > 0, got {total_size}")
        self._url_provider = url_provider
        self._total = int(total_size)
        self._pos = 0
        self._sess = session or _make_session()
        self._chunk = int(chunk_size)
        self._cache = _ByteRangeCache(max_bytes=cache_bytes)
        self._cookies = cookies
        self._headers = dict(headers or {})
        self._url_ttl = float(url_ttl_seconds)
        self._timeout = timeout
        self._urlstate: Optional[_UrlState] = None
        # diagnostics
        self.n_http_requests = 0
        self.n_cache_hits = 0
        self.bytes_fetched = 0

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

    # ---- URL handling ----
    def _get_url(self, force_refresh: bool = False) -> str:
        now = time.time()
        if (
            force_refresh
            or self._urlstate is None
            or (self._url_ttl > 0 and now >= self._urlstate.expires_at)
        ):
            url = self._url_provider()
            self._urlstate = _UrlState(
                url=url,
                expires_at=(now + self._url_ttl) if self._url_ttl > 0 else 0.0,
            )
        return self._urlstate.url

    def _http_range_get(self, start: int, end_exclusive: int) -> bytes:
        """One Range GET. Retries on 403 (URL expiry) once with refreshed URL."""
        if start >= self._total:
            return b""
        end_inclusive = min(end_exclusive, self._total) - 1
        headers = dict(self._headers)
        headers["Range"] = f"bytes={start}-{end_inclusive}"

        for attempt_refresh in (False, True):
            url = self._get_url(force_refresh=attempt_refresh)
            try:
                self.n_http_requests += 1
                r = self._sess.get(
                    url, headers=headers, cookies=self._cookies,
                    timeout=self._timeout, allow_redirects=True, stream=False,
                )
            except requests.exceptions.RequestException:
                if attempt_refresh:
                    raise
                continue
            if r.status_code in (200, 206):
                buf = r.content
                self.bytes_fetched += len(buf)
                return buf
            if r.status_code in (403, 401) and not attempt_refresh:
                # URL likely expired
                continue
            r.raise_for_status()
        raise RuntimeError("unreachable")

    # ---- chunk-aligned fetching with cache ----
    def _fetch_chunk_containing(self, pos: int) -> Tuple[int, int, bytes]:
        """Fetch a chunk-aligned window starting at floor(pos/chunk)*chunk."""
        chunk = self._chunk
        cs = (pos // chunk) * chunk
        ce = min(cs + chunk, self._total)
        # cache lookup
        hit = self._cache.get(cs, ce)
        if hit is not None and len(hit) == ce - cs:
            self.n_cache_hits += 1
            return cs, ce, hit
        data = self._http_range_get(cs, ce)
        self._cache.put(cs, ce, data)
        return cs, ce, data

    def readinto(self, b) -> int:  # type: ignore[override]
        if self._pos >= self._total:
            return 0
        n_wanted = len(b)
        n_got = 0
        mv = memoryview(b)
        while n_got < n_wanted and self._pos < self._total:
            cs, ce, data = self._fetch_chunk_containing(self._pos)
            local_off = self._pos - cs
            take = min(n_wanted - n_got, len(data) - local_off)
            if take <= 0:
                break
            mv[n_got:n_got + take] = data[local_off:local_off + take]
            n_got += take
            self._pos += take
        return n_got

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        if size is None or size < 0:
            size = self._total - self._pos
        buf = bytearray(size)
        n = self.readinto(buf)
        return bytes(buf[:n])

    def close(self) -> None:
        try:
            self._cache.clear()
        finally:
            super().close()
        # We deliberately do NOT close the session; it may be shared.


# =============================================================================
# ModelScope-specific helpers
# =============================================================================

def open_dataset_file_as_range_stream(
    dataset_id: str,
    path_in_repo: str,
    revision: str = "master",
    token: Optional[str] = None,
    total_size: Optional[int] = None,
    chunk_size: int = 8 * 1024 * 1024,
    cache_bytes: int = 256 * 1024 * 1024,
) -> RemoteRangeStream:
    """Open a single file in a ModelScope dataset as a seekable byte stream.

    - `total_size`: pass it if you already know the size (avoids a HEAD request).
      Otherwise we get it from the dataset listing.
    """
    from modelscope.hub.api import HubApi

    from .ms_download import (
        list_dataset_files,
        login_if_token,
    )

    login_if_token(token)
    api = HubApi()

    # 1) discover size if unknown
    if total_size is None:
        files = list_dataset_files(dataset_id, revision=revision, token=token)
        match = None
        for f in files:
            if (f.get("Path") == path_in_repo
                    or f.get("Path", "").lstrip("/") == path_in_repo.lstrip("/")):
                match = f; break
        if match is None:
            raise FileNotFoundError(f"{path_in_repo!r} not found in {dataset_id}")
        total_size = int(match.get("Size", 0) or 0)
        if total_size <= 0:
            raise RuntimeError(f"{path_in_repo!r} has no size in listing")

    namespace, name = dataset_id.split("/", 1)
    cookies = api.get_cookies(access_token=token)

    def _url_provider() -> str:
        return api.get_dataset_file_url(
            file_name=path_in_repo, dataset_name=name, namespace=namespace,
            revision=revision,
        )

    return RemoteRangeStream(
        url_provider=_url_provider,
        total_size=total_size,
        cookies=cookies,
        chunk_size=chunk_size,
        cache_bytes=cache_bytes,
    )

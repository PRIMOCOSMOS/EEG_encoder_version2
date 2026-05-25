"""SEED-VII dataset utilities.

提供：
    1) `load_save_info_intensity` —— 从官方 save_info CSV 解析每个 (subject, session, trial) 的连续强度 ∈ [0,1]
    2) `iter_trials_from_zip` —— 流式从多分卷 zip 抽出 (subject, session, trial, raw_eeg) 三元组
    3) `build_trial_index` / `split_trials` —— **trial-level** 训练/验证/测试切分（先切分，后处理）
    4) `EEGWindowArrayDataset` —— 内存中的 (X, y, s) 窗口 Dataset，供训练用
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .labels import (
    EMOTION_TO_IDX,
    field_id_to_label,
    trial_field_to_session_trial,
)
from .zip_stream import (
    extract_mat_bytes,
    iter_mat_members,
    locate_members_in_stream,
    open_concat_zip,
    open_remote_concat_zip,
    schedule_members_by_part,
    volume_sort_key,
    LazyConcatStream,
)


# ---------------------------------------------------------------------------
# 1) save_info → intensity
# ---------------------------------------------------------------------------

# 支持多种命名：
#  1) 1_20221001_1_save_info.csv  (subject_date_session)
#  2) 1_1_save_info.csv            (subject_session)
_FILENAME_RE = re.compile(
    r"^(?P<subject>\d+)(?:_(?P<date>\d{8}))?_(?P<session>\d+)_save_info\.csv$",
    re.IGNORECASE,
)
# 兼容旧的非数字subject
_FILENAME_RE_LEGACY = re.compile(
    r"^(?P<subject>[^_/\\]+)_(?P<date>[^_/\\]+)_(?P<session>\d+)_save_info\.csv$",
    re.IGNORECASE,
)


def _parse_save_info_csv(path: Path) -> Dict[int, float]:
    """Parse one save_info csv -> {trial_id_1based: intensity 0..1}.

    SEED-VII official说明：每个 movie clip 都有一行/列 score ∈ [0,1]，
    指示 targeted emotion 的诱发成功度。这里做兼容性解析：
    - 只有在能明确识别表头时，才优先寻找名为 'score' / 'intensity' / 'rating' 的列
    - 若是逐行记录的明细表，则取每行最后一个数值作为强度
    - 若是单行宽表（1 行 × N 列）则按列序取全部数值
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read()
    if not sample.strip():
        return {}

    reader = csv.reader(io.StringIO(sample))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        return {}

    header = rows[0]
    lowered = [c.strip().lower() for c in header]
    header_score_cols = {"score", "intensity", "rating", "feedback"}
    has_header = any(c in header_score_cols for c in lowered)
    if not has_header:
        # A row that contains path-like text or numeric data is usually a data row,
        # not a header. The SEED-VII save_info sample is formatted this way.
        has_numeric = any(_is_floatish(c) for c in header)
        has_path_like = any(_looks_like_pathish(c) for c in header)
        has_header = (not has_numeric) and (not has_path_like) and len(rows) > 1

    scores: List[float] = []
    if has_header:
        target_col = -1
        for cand in ("score", "intensity", "rating", "feedback"):
            if cand in lowered:
                target_col = lowered.index(cand)
                break
        if target_col >= 0:
            for r in rows[1:]:
                if target_col < len(r) and _is_floatish(r[target_col]):
                    scores.append(_to_unit(float(r[target_col])))
        else:
            # fall back: take the last numeric column in each data row
            for r in rows[1:]:
                numeric_vals = [_to_unit(float(c)) for c in r if _is_floatish(c)]
                if numeric_vals:
                    scores.append(numeric_vals[-1])
    else:
        if len(rows) == 1 and len(rows[0]) >= 5:
            # wide row
            scores.extend(_to_unit(float(c)) for c in rows[0] if _is_floatish(c))
        else:
            for r in rows:
                numeric_vals = [_to_unit(float(c)) for c in r if _is_floatish(c)]
                if numeric_vals:
                    scores.append(numeric_vals[-1])

    out: Dict[int, float] = {}
    for i, s in enumerate(scores, start=1):
        out[i] = float(s)
    return out


def _is_floatish(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _looks_like_pathish(s: str) -> bool:
    s = s.strip()
    return ("/" in s) or ("\\" in s) or ("." in Path(s).name and not _is_floatish(s))


def _to_unit(v: float) -> float:
    """Clamp to [0,1]; if value looks like 0-5 scale, divide by 5; if 0-100, /100."""
    if v < 0:
        v = 0.0
    if v <= 1.0:
        return v
    if v <= 5.0:
        return v / 5.0
    if v <= 100.0:
        return v / 100.0
    return 1.0


def load_save_info_intensity(save_info_dir: os.PathLike) -> Dict[Tuple[str, int, int], float]:
    """Load all save_info CSVs under `save_info_dir`.

    Returns dict: {(subject, session_id, trial_id_1based_in_session): intensity}
    where trial_id_1based_in_session ∈ [1,20].
    """
    d = Path(save_info_dir)
    if not d.exists():
        return {}
    out: Dict[Tuple[str, int, int], float] = {}
    for p in d.glob("*_save_info.csv"):
        m = _FILENAME_RE.match(p.name) or _FILENAME_RE_LEGACY.match(p.name)
        if not m:
            # 兜底：提取前两个数字作为 subject/session
            nums = re.findall(r"\d+", p.stem)
            if len(nums) >= 2:
                subject = nums[0]
                session = int(nums[1])
                print(f"[WARN] Unrecognized save_info filename '{p.name}', fallback parsed as subject={subject}, session={session}")
            else:
                print(f"[WARN] Skipping unrecognized save_info file: {p.name}")
                continue
        else:
            subject = m.group("subject")
            session = int(m.group("session"))
        scores = _parse_save_info_csv(p)
        for tid, val in scores.items():
            out[(subject, session, tid)] = float(val)
    print(f"[INFO] load_save_info_intensity: parsed {len(out)} entries from {len(list(d.glob('*_save_info.csv')))} files")
    return out


# ---------------------------------------------------------------------------
# 2) zip streaming → trial iterator
# ---------------------------------------------------------------------------

@dataclass
class RawTrial:
    subject: str            # filename stem, e.g. "1", "2", ..., "20"
    session_id: int         # 1..4
    trial_id: int           # 1..20 inside the session
    field_id: int           # 1..80 (original field name in the .mat)
    eeg: np.ndarray         # (62, T)


def _iter_trials_from_zipfile(
    zf,
    subdir_keyword: str = "EEG_preprocessed",
    only_subjects: Optional[Sequence[str]] = None,
    stream: Optional[LazyConcatStream] = None,
) -> Iterator[RawTrial]:
    """Core: given an opened ZipFile, iterate (subject, session, trial, eeg) records.

    If `stream` is a LazyConcatStream, we schedule the read order by **volume index**
    so the LRU pulls each split volume in at most twice in the worst case (once for the
    main bulk + once if a neighboring mat straddles the boundary), instead of bouncing
    around. This makes the wall-clock cost ≈ "download each volume once, sequentially".
    """
    members = list(iter_mat_members(zf, subdir_keyword=subdir_keyword))
    if only_subjects:
        wanted = set(map(str, only_subjects))
        members = [m for m in members if Path(m.filename).stem in wanted]

    if stream is not None:
        # Schedule by physical layout in the concat stream (monotonic volume traversal)
        locales = locate_members_in_stream(zf, members, stream)
        locales = schedule_members_by_part(locales)
        ordered = [loc.info for loc in locales]
    else:
        # Logical sort by subject filename (e.g. "1".."20")
        members.sort(key=lambda m: _natural_key(Path(m.filename).stem))
        ordered = members
        locales = None

    for k, info in enumerate(ordered):
        subject = Path(info.filename).stem  # "1".."20"

        # If this member straddles a volume boundary, temporarily pin the END
        # volume so the LRU doesn't evict it before we finish reading.
        pinned_extra: Optional[int] = None
        if stream is not None and locales is not None:
            loc = locales[k]
            if loc.end_part != loc.start_part:
                # pin the end volume (start volume is current and is auto-protected
                # by `_maybe_evict()`'s "don't evict current" rule)
                try:
                    stream.pin(loc.end_part, fetch_now=False)
                    pinned_extra = loc.end_part
                except Exception:
                    pinned_extra = None

        try:
            raw_bytes = extract_mat_bytes(zf, info)
        finally:
            if pinned_extra is not None:
                # release the temporary pin (but keep file resident; it becomes
                # LRU-evictable from here on).
                try:
                    stream.unpin(pinned_extra, evict_now=False)
                except Exception:
                    pass

        data = loadmat(
            io.BytesIO(raw_bytes),
            verify_compressed_data_integrity=False,
        )
        fields = []
        for key in data.keys():
            if key.startswith("__"):
                continue
            if key.isdigit():
                fields.append((int(key), key))
        fields.sort(key=lambda t: t[0])
        del raw_bytes  # free as early as possible
        for fid, name in fields:
            arr = np.asarray(data[name])
            if arr.ndim != 2 or arr.shape[0] != 62:
                continue
            session_id, trial_in_session = trial_field_to_session_trial(fid)
            yield RawTrial(
                subject=subject,
                session_id=session_id,
                trial_id=trial_in_session,
                field_id=fid,
                eeg=arr,
            )
        del data


def iter_trials_from_zip(
    volumes_dir: os.PathLike,
    pattern: str = "*.zip.*",
    subdir_keyword: str = "EEG_preprocessed",
    only_subjects: Optional[Sequence[str]] = None,
) -> Iterator[RawTrial]:
    """Stream every (subject × 80 trials) from a LOCAL multi-volume zip without disk extraction.

    Each `.mat` is read into memory transiently (≈ tens of MB), parsed, then discarded.
    """
    zf = open_concat_zip(volumes_dir, pattern=pattern)
    try:
        yield from _iter_trials_from_zipfile(
            zf, subdir_keyword=subdir_keyword, only_subjects=only_subjects,
        )
    finally:
        # close the zipfile and the underlying ConcatStream
        try:
            inner = zf.fp
        except Exception:
            inner = None
        try:
            zf.close()
        finally:
            if inner is not None:
                try: inner.close()
                except Exception: pass


def iter_trials_from_mat_dir(
    mat_dir: os.PathLike,
    pattern: str = "*.mat",
    only_subjects: Optional[Sequence[str]] = None,
    recursive: bool = True,
) -> Iterator[RawTrial]:
    """Stream trials from a LOCAL directory of `*.mat` files (e.g. Kaggle dataset mount).

    No zip involved. Each .mat is loaded sequentially, parsed, then released.

    The .mat file STEM is used as the `subject` identifier (e.g. ``1.mat`` → "1").
    Inside each .mat we look for numeric field names "1".."80" (per Design.md),
    each holding a `(62, T)` EEG array; these are mapped to (session, trial)
    via :func:`trial_field_to_session_trial`.

    Args:
        mat_dir:    folder containing the .mat files
        pattern:    glob pattern (default ``*.mat``)
        only_subjects: optional whitelist of subject stems (e.g. ['1', '5', '20'])
        recursive:  if True, search nested directories
    """
    d = Path(mat_dir)
    if not d.exists():
        raise FileNotFoundError(f"mat_dir does not exist: {d}")
    if recursive:
        all_mats = sorted(d.rglob(pattern))
    else:
        all_mats = sorted(d.glob(pattern))
    if not all_mats:
        raise FileNotFoundError(
            f"No files matched {pattern!r} under {d} (recursive={recursive})"
        )
    if only_subjects:
        wanted = set(map(str, only_subjects))
        all_mats = [p for p in all_mats if p.stem in wanted]
    # natural ordering by subject stem
    all_mats.sort(key=lambda p: _natural_key(p.stem))

    for mat_path in all_mats:
        subject = mat_path.stem  # "1".."20"
        data = loadmat(str(mat_path), verify_compressed_data_integrity=False)
        fields = []
        for k in data.keys():
            if k.startswith("__"):
                continue
            if k.isdigit():
                fields.append((int(k), k))
        fields.sort(key=lambda t: t[0])
        for fid, name in fields:
            arr = np.asarray(data[name])
            if arr.ndim != 2 or arr.shape[0] != 62:
                continue
            session_id, trial_in_session = trial_field_to_session_trial(fid)
            yield RawTrial(
                subject=subject,
                session_id=session_id,
                trial_id=trial_in_session,
                field_id=fid,
                eeg=arr,
            )
        del data


def iter_trials_from_modelscope(
    dataset_id: str,
    pattern: str = "*.zip.*",
    scratch_dir: str = "./_ms_volumes_cache",
    revision: str = "master",
    token: Optional[str] = None,
    subdir_keyword: str = "EEG_preprocessed",
    only_subjects: Optional[Sequence[str]] = None,
    max_resident_volumes: int = 2,
) -> Iterator[RawTrial]:
    """Stream trials directly from a ModelScope dataset (no manual download step).

    User volumes are **byte-level splits** of the original zip (`split -b ...` style),
    not zip-native spanning archives. Their byte-concatenation == one valid ZIP64.

    Internally:
      1) List & sort the remote split volumes by the volume's trailing-digit index.
      2) Build a `LazyConcatStream` that downloads each volume only when its bytes
         are needed, evicts non-pinned ones with LRU, and **pre-pins the LAST volume**
         so opening `zipfile.ZipFile(...)` (which seeks to end to find EOCD/central
         directory) does not trigger a "download-then-evict" cycle on every open.
      3) Schedule mat reads in volume-order so the LRU pulls each volume in once.

    Disk footprint at any instant:
        1 pinned last-volume + max_resident_volumes LRU-live + 1 currently-open
        ≈ (2 + max_resident_volumes) × 5.37GB. Default → ≤ ~21GB.
        After the central-directory scan you can call `stream.unpin(N-1)` to drop
        it back to ≈ (1 + max_resident_volumes) volumes.
    """
    from .ms_download import (
        download_one_file,
        list_dataset_files,
        login_if_token,
    )

    login_if_token(token)

    # 1) discover & get sizes
    listing = list_dataset_files(dataset_id, revision=revision, token=token)
    import fnmatch as _fn
    listing = [f for f in listing if _fn.fnmatch(f.get("Path", ""), pattern)]
    if not listing:
        raise RuntimeError(f"No remote volumes matched {pattern} in {dataset_id}")
    listing.sort(key=lambda f: volume_sort_key(f.get("Path", "")))

    sizes_in_order = [(f["Path"], int(f.get("Size", 0) or 0)) for f in listing]
    missing_sizes = [n for n, s in sizes_in_order if s <= 0]
    if missing_sizes:
        raise RuntimeError(
            f"Remote volume size missing from listing for: {missing_sizes[:3]}... "
            "ModelScope API did not return Size. Cannot lazy-stream without size."
        )

    Path(scratch_dir).mkdir(parents=True, exist_ok=True)

    def _fetch(remote_name: str) -> str:
        p = download_one_file(
            dataset_id, remote_name, scratch_dir,
            revision=revision, token=token,
        )
        return str(p)

    def _evict(local_path: Path) -> None:
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass

    zf, stream = open_remote_concat_zip(
        sizes_in_order=sizes_in_order,
        fetcher=_fetch,
        evicter=_evict,
        max_resident=max_resident_volumes,
        pin_last=True,
        warmup_last=True,
    )
    try:
        yield from _iter_trials_from_zipfile(
            zf,
            subdir_keyword=subdir_keyword,
            only_subjects=only_subjects,
            stream=stream,
        )
    finally:
        try:
            zf.close()
        finally:
            # zipfile.ZipFile does not close a file-like passed in; we must do it
            # ourselves to trigger our scratch cleanup of pinned/LRU volumes.
            stream.close()


def iter_trials_from_modelscope_single_file(
    dataset_id: str,
    path_in_repo: str = "SEED-VII.zip",
    revision: str = "master",
    token: Optional[str] = None,
    subdir_keyword: str = "EEG_preprocessed",
    only_subjects: Optional[Sequence[str]] = None,
    cache_mb: int = 256,
    chunk_mb: int = 8,
) -> Iterator[RawTrial]:
    """Stream trials directly from a SINGLE merged zip living on ModelScope.

    Uses HTTP Range requests against the dataset's pre-signed OSS URL.
    **Disk usage = 0 bytes** (memory cache only, default ≤256 MB).

    This is the preferred path AFTER `scripts/merge_and_upload.py` has produced
    the unified `path_in_repo` (e.g. ``SEED-VII.zip``).

    Args:
        path_in_repo: target file inside the dataset, e.g. ``SEED-VII.zip``
        cache_mb:     in-memory LRU range-cache size (default 256 MB)
        chunk_mb:     size of each Range GET request (default 8 MB)
    """
    import zipfile
    from .remote_range import open_dataset_file_as_range_stream

    stream = open_dataset_file_as_range_stream(
        dataset_id=dataset_id,
        path_in_repo=path_in_repo,
        revision=revision,
        token=token,
        chunk_size=chunk_mb * 1024 * 1024,
        cache_bytes=cache_mb * 1024 * 1024,
    )
    zf = zipfile.ZipFile(stream, mode="r")
    try:
        # No need for the volume-ordered scheduler — single zip, single source.
        yield from _iter_trials_from_zipfile(
            zf, subdir_keyword=subdir_keyword, only_subjects=only_subjects,
            stream=None,
        )
    finally:
        try:
            zf.close()
        finally:
            try: stream.close()
            except Exception: pass



def _natural_key(name: str) -> Tuple[int, str]:
    try:
        return (int(name), name)
    except Exception:
        return (10**9, name)


# ---------------------------------------------------------------------------
# 3) Trial-level splitting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialKey:
    subject: str
    session_id: int
    trial_id: int

    def as_tuple(self) -> Tuple[str, int, int]:
        return (self.subject, self.session_id, self.trial_id)


def split_trials(
    trial_keys: Sequence[TrialKey],
    labels: Sequence[int],
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    unit: str = "trial",
) -> Tuple[List[TrialKey], List[TrialKey], List[TrialKey]]:
    """Split BEFORE windowing to avoid leakage.

    unit:
        - "trial":   split independent (subject,session,trial) keys (default; recommended).
        - "subject": split whole subjects into train/val/test (cross-subject).
        - "session": for each subject, split sessions.
    """
    rng = np.random.default_rng(seed)
    if not (0.0 < val_ratio < 1.0 and 0.0 < test_ratio < 1.0 and val_ratio + test_ratio < 1.0):
        raise ValueError("val_ratio, test_ratio must be in (0,1) and sum < 1")

    if unit == "trial":
        return _stratified_split(list(trial_keys), list(labels), val_ratio, test_ratio, rng)
    if unit == "subject":
        return _split_by_group(trial_keys, key=lambda k: k.subject,
                               val_ratio=val_ratio, test_ratio=test_ratio, rng=rng)
    if unit == "session":
        return _split_by_group(trial_keys, key=lambda k: (k.subject, k.session_id),
                               val_ratio=val_ratio, test_ratio=test_ratio, rng=rng)
    raise ValueError(f"Unknown split unit: {unit}")


def _stratified_split(
    keys: List[TrialKey],
    labels: List[int],
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[TrialKey], List[TrialKey], List[TrialKey]]:
    by_label: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_label.setdefault(int(y), []).append(i)
    train, val, test = [], [], []
    for y, idxs in by_label.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_ratio))) if n - n_test >= 2 else 0
        test_ids = idxs[:n_test]
        val_ids = idxs[n_test:n_test + n_val]
        train_ids = idxs[n_test + n_val:]
        train.extend(keys[i] for i in train_ids)
        val.extend(keys[i] for i in val_ids)
        test.extend(keys[i] for i in test_ids)
    return train, val, test


def _split_by_group(trial_keys, key, val_ratio, test_ratio, rng):
    groups: Dict = {}
    for k in trial_keys:
        groups.setdefault(key(k), []).append(k)
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    n = len(group_ids)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    test_groups = set(group_ids[:n_test])
    val_groups = set(group_ids[n_test:n_test + n_val])
    train, val, test = [], [], []
    for g, items in groups.items():
        if g in test_groups:
            test.extend(items)
        elif g in val_groups:
            val.extend(items)
        else:
            train.extend(items)
    return train, val, test


# ---------------------------------------------------------------------------
# 4) Tensor dataset
# ---------------------------------------------------------------------------

class EEGWindowArrayDataset(Dataset):
    """In-memory windowed dataset for training.

    `X`: (N, 62, T) float32
    `y`: (N,) int64       class index
    `s`: (N,) float32     continuous intensity ∈ [0,1]
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, s: np.ndarray):
        assert x.ndim == 3 and x.shape[0] == y.shape[0] == s.shape[0]
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()
        self.s = torch.from_numpy(s).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        # add channel-dim -> (1, 62, T)  to match Conv2d expectations
        return self.x[idx].unsqueeze(0), self.y[idx], self.s[idx]


# ---------------------------------------------------------------------------
# 5) NPZ I/O
# ---------------------------------------------------------------------------

def save_dataset_npz(
    path: os.PathLike,
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    meta_list: List[dict],
    split_assignment: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    payload = {
        "X": x.astype(np.float32, copy=False),
        "y": y.astype(np.int64, copy=False),
        "s": s.astype(np.float32, copy=False),
        "meta": np.asarray([json.dumps(m, ensure_ascii=True) for m in meta_list], dtype=object),
    }
    if split_assignment:
        for k, v in split_assignment.items():
            payload[f"split_{k}"] = np.asarray(v, dtype=np.int64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_dataset_npz(path: os.PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict], Dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=True)
    x = np.asarray(z["X"])
    y = np.asarray(z["y"])
    s = np.asarray(z["s"]) if "s" in z.files else np.ones(len(y), dtype=np.float32)
    meta = [json.loads(str(m)) for m in z["meta"]]
    splits: Dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        k = f"split_{name}"
        if k in z.files:
            splits[name] = np.asarray(z[k], dtype=np.int64)
    return x, y, s, meta, splits

"""SEED-VII dataset utilities.

提供：
 1) `load_save_info_intensity` —— 从 save_info CSV 解析连续强度
 2) `iter_trials_from_zip` 等 —— 流式试次迭代器
 3) `split_trials` —— trial-level 划分
 4) `EEGWindowArrayDataset` —— 窗口 Dataset
 5) `filter_by_subjects` —— 按被试筛选索引
 6) `ensure_mmap_format` / `load_dataset_mmap` —— OOM 终极修复：
    将压缩 npz 一次性转换为 X.npy + meta.npz，之后用 mmap 加载，
    X 不占任何 RAM，只把实际需要的子集拷贝进内存。
"""
from __future__ import annotations

import csv
import gc
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

_FILENAME_RE = re.compile(
    r"^(?P<subject>\d+)(?:_(?P<date>\d{8}))?_(?P<session>\d+)_save_info\.csv$",
    re.IGNORECASE,
)
_FILENAME_RE_LEGACY = re.compile(
    r"^(?P<subject>[^_\\/]+)_(?P<date>[^_\\/]+)_(?P<session>\d+)_save_info\.csv$",
    re.IGNORECASE,
)

def _parse_save_info_csv(path: Path) -> Dict[int, float]:
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
            for r in rows[1:]:
                numeric_vals = [_to_unit(float(c)) for c in r if _is_floatish(c)]
                if numeric_vals:
                    scores.append(numeric_vals[-1])
    else:
        if len(rows) == 1 and len(rows[0]) >= 5:
            scores.extend(_to_unit(float(c)) for c in rows[0] if _is_floatish(c))
        else:
            for r in rows:
                numeric_vals = [_to_unit(float(c)) for c in r if _is_floatish(c)]
                if numeric_vals:
                    scores.append(numeric_vals[-1])
    out: Dict[int, float] = {}
    for i, sv in enumerate(scores, start=1):
        out[i] = float(sv)
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
    d = Path(save_info_dir)
    if not d.exists():
        return {}
    out: Dict[Tuple[str, int, int], float] = {}
    for p in d.glob("*_save_info.csv"):
        m = _FILENAME_RE.match(p.name) or _FILENAME_RE_LEGACY.match(p.name)
        if not m:
            nums = re.findall(r"\d+", p.stem)
            if len(nums) >= 2:
                subject, session = nums[0], int(nums[1])
                print(f"[WARN] Fallback parsed '{p.name}' as subject={subject}, session={session}")
            else:
                continue
        else:
            subject = m.group("subject")
            session = int(m.group("session"))
        scores = _parse_save_info_csv(p)
        for tid, val in scores.items():
            out[(subject, session, tid)] = float(val)
    print(f"[INFO] load_save_info_intensity: parsed {len(out)} entries")
    return out

# ---------------------------------------------------------------------------
# 2) zip streaming → trial iterator
# ---------------------------------------------------------------------------

@dataclass
class RawTrial:
    subject: str
    session_id: int
    trial_id: int
    field_id: int
    eeg: np.ndarray

def _iter_trials_from_zipfile(zf, subdir_keyword="EEG_preprocessed",
                               only_subjects=None, stream=None):
    members = list(iter_mat_members(zf, subdir_keyword=subdir_keyword))
    if only_subjects:
        wanted = set(map(str, only_subjects))
        members = [m for m in members if Path(m.filename).stem in wanted]
    if stream is not None:
        locales = locate_members_in_stream(zf, members, stream)
        locales = schedule_members_by_part(locales)
        ordered = [loc.info for loc in locales]
    else:
        members.sort(key=lambda m: _natural_key(Path(m.filename).stem))
        ordered = members
        locales = None
    for k, info in enumerate(ordered):
        subject = Path(info.filename).stem
        pinned_extra = None
        if stream is not None and locales is not None:
            loc = locales[k]
            if loc.end_part != loc.start_part:
                try:
                    stream.pin(loc.end_part, fetch_now=False)
                    pinned_extra = loc.end_part
                except Exception:
                    pass
        try:
            raw_bytes = extract_mat_bytes(zf, info)
        finally:
            if pinned_extra is not None:
                try: stream.unpin(pinned_extra, evict_now=False)
                except Exception: pass
        data = loadmat(io.BytesIO(raw_bytes), verify_compressed_data_integrity=False)
        fields = []
        for key in data.keys():
            if not key.startswith("__") and key.isdigit():
                fields.append((int(key), key))
        fields.sort(key=lambda t: t[0])
        del raw_bytes
        for fid, name in fields:
            arr = np.asarray(data[name])
            if arr.ndim != 2 or arr.shape[0] != 62:
                continue
            session_id, trial_in_session = trial_field_to_session_trial(fid)
            yield RawTrial(subject=subject, session_id=session_id,
                           trial_id=trial_in_session, field_id=fid, eeg=arr)
        del data

def _natural_key(name):
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
    def as_tuple(self): return (self.subject, self.session_id, self.trial_id)

def split_trials(trial_keys, labels, val_ratio, test_ratio, seed=42, unit="trial"):
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

def _stratified_split(keys, labels, val_ratio, test_ratio, rng):
    by_label = {}
    for i, y in enumerate(labels):
        by_label.setdefault(int(y), []).append(i)
    train, val, test = [], [], []
    for y, idxs in by_label.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_ratio))) if n - n_test >= 2 else 0
        test.extend(keys[i] for i in idxs[:n_test])
        val.extend(keys[i] for i in idxs[n_test:n_test + n_val])
        train.extend(keys[i] for i in idxs[n_test + n_val:])
    return train, val, test

def _split_by_group(trial_keys, key, val_ratio, test_ratio, rng):
    groups = {}
    for k in trial_keys:
        groups.setdefault(key(k), []).append(k)
    gids = list(groups.keys())
    rng.shuffle(gids)
    n = len(gids)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    tg = set(gids[:n_test])
    vg = set(gids[n_test:n_test + n_val])
    train, val, test = [], [], []
    for g, items in groups.items():
        if g in tg: test.extend(items)
        elif g in vg: val.extend(items)
        else: train.extend(items)
    return train, val, test

# ---------------------------------------------------------------------------
# 4) Tensor dataset
# ---------------------------------------------------------------------------

class EEGWindowArrayDataset(Dataset):
    """In-memory windowed dataset. Accepts indices for zero-copy subsetting."""

    def __init__(self, x, y, s, indices=None):
        if indices is not None:
            self._use_indices = True
            self._x = torch.from_numpy(np.ascontiguousarray(x))
            self._y = torch.from_numpy(np.ascontiguousarray(y))
            self._s = torch.from_numpy(np.ascontiguousarray(s))
            self._indices = torch.from_numpy(np.asarray(indices, dtype=np.int64))
            self._len = len(indices)
        else:
            self._use_indices = False
            self._x = torch.from_numpy(np.ascontiguousarray(x))
            self._y = torch.from_numpy(np.ascontiguousarray(y))
            self._s = torch.from_numpy(np.ascontiguousarray(s))
            self._len = x.shape[0]

    def __len__(self): return self._len

    def __getitem__(self, idx):
        ri = self._indices[idx].item() if self._use_indices else idx
        return self._x[ri].unsqueeze(0), self._y[ri], self._s[ri]

# ---------------------------------------------------------------------------
# 5) NPZ I/O  ——  OOM 终极修复：mmap 两阶段加载
# ---------------------------------------------------------------------------

def ensure_mmap_format(npz_path: os.PathLike) -> Tuple[str, str]:
    """One-time conversion: extract X.npy + meta.npz from compressed npz.

    Returns (x_npy_path, meta_npz_path).
    After conversion, X can be loaded with np.load(..., mmap_mode='r')
    so it consumes **zero RAM** until indexed.

    This conversion needs enough RAM to hold X once (~10-12 GB for full
    SEED-VII). If even this OOMs, you must re-preprocess with
    ``--only-subjects`` to produce a smaller npz.
    """
    npz_path = str(npz_path)
    x_npy = npz_path + ".X.npy"
    meta_npz = npz_path + ".meta.npz"

    if os.path.exists(x_npy) and os.path.exists(meta_npz):
        # Already converted
        return x_npy, meta_npz

    print(f"[MMAP-CONVERT] Converting {npz_path} → mmap format (one-time)...")
    print(f"[MMAP-CONVERT] This requires enough RAM to hold X once. If OOM,")
    print(f"[MMAP-CONVERT] re-run preprocessing with --only-subjects to make a smaller npz.")

    z = np.load(npz_path, allow_pickle=True)
    try:
        # Step 1: Save X as uncompressed .npy (mmap-friendly)
        x = z["X"]
        print(f"[MMAP-CONVERT] X shape={x.shape}, dtype={x.dtype}, "
              f"size={x.nbytes / 1024**3:.2f} GB")
        np.save(x_npy, x)
        del x
        gc.collect()

        # Step 2: Save everything else as compressed npz (small)
        payload = {}
        for k in z.files:
            if k != "X":
                payload[k] = z[k]
        np.savez_compressed(meta_npz, **payload)
    finally:
        z.close()

    x_size = os.path.getsize(x_npy) / 1024**3
    m_size = os.path.getsize(meta_npz) / 1024**3
    print(f"[MMAP-CONVERT] Done! X.npy={x_size:.2f}GB, meta.npz={m_size:.2f}GB")
    print(f"[MMAP-CONVERT] Future runs will use mmap (0 RAM for X).")
    return x_npy, meta_npz


def load_dataset_mmap(
    npz_path: os.PathLike,
    subset_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict], Dict[str, np.ndarray]]:
    """Load preprocessed data with mmap for X (zero RAM for the big array).

    **Key OOM fix**: X is memory-mapped, never fully loaded into RAM.
    If `subset_indices` is provided, only those rows are copied into RAM,
    and the returned arrays are re-indexed from 0.

    Returns (x, y, s, meta, splits).
    If subset_indices given, arrays are subset-only and splits are omitted.
    """
    npz_path = str(npz_path)
    x_npy = npz_path + ".X.npy"
    meta_npz = npz_path + ".meta.npz"

    # --- Ensure mmap format exists ---
    if not (os.path.exists(x_npy) and os.path.exists(meta_npz)):
        ensure_mmap_format(npz_path)

    # --- Load small arrays from meta.npz ---
    z = np.load(meta_npz, allow_pickle=True)
    try:
        y_full = np.asarray(z["y"])
        s_full = np.asarray(z["s"]) if "s" in z.files else np.ones(len(y_full), dtype=np.float32)
        meta_full = [json.loads(str(m)) for m in z["meta"]]
        splits_full: Dict[str, np.ndarray] = {}
        for name in ("train", "val", "test"):
            k = f"split_{name}"
            if k in z.files:
                splits_full[name] = np.asarray(z[k], dtype=np.int64)
    finally:
        z.close()

    # --- Load X via mmap (ZERO RAM) ---
    x_mmap = np.load(x_npy, mmap_mode="r")

    if subset_indices is not None:
        # Only copy the needed rows into RAM
        subset_indices = np.sort(np.asarray(subset_indices, dtype=np.int64))
        print(f"[MMAP-LOAD] Copying {len(subset_indices)}/{len(y_full)} rows into RAM "
              f"(~{len(subset_indices) * x_mmap.shape[1] * x_mmap.shape[2] * 4 / 1024**3:.2f} GB)...")
        x_sub = np.array(x_mmap[subset_indices], dtype=np.float32)
        y_sub = y_full[subset_indices]
        s_sub = s_full[subset_indices]
        meta_sub = [meta_full[i] for i in subset_indices]

        # Build index mapping: old global idx → new local idx
        idx_map = {int(old): new for new, old in enumerate(subset_indices)}

        del x_mmap, y_full, s_full, meta_full
        gc.collect()

        return x_sub, y_sub, s_sub, meta_sub, {"idx_map": idx_map}
    else:
        # Return mmap reference for X; caller must not hold it too long
        # or should copy subset manually
        return x_mmap, y_full, s_full, meta_full, splits_full


def save_dataset_npz(path, x, y, s, meta_list, split_assignment=None):
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


def load_dataset_npz(path):
    """Legacy loader — loads everything into RAM. Do NOT use for large datasets."""
    z = np.load(path, allow_pickle=True)
    try:
        x = np.asarray(z["X"])
        y = np.asarray(z["y"])
        s_val = np.asarray(z["s"]) if "s" in z.files else np.ones(x.shape[0], dtype=np.float32)
        meta = [json.loads(str(m)) for m in z["meta"]]
        splits = {}
        for name in ("train", "val", "test"):
            k = f"split_{name}"
            if k in z.files:
                splits[name] = np.asarray(z[k], dtype=np.int64)
    finally:
        try: z.close()
        except Exception: pass
    return x, y, s_val, meta, splits

# ---------------------------------------------------------------------------
# 6) Subject-level filtering
# ---------------------------------------------------------------------------

def filter_by_subjects(meta_list, subjects):
    wanted = set(str(sv) for sv in subjects)
    idxs = [i for i, m in enumerate(meta_list) if str(m.get("subject", "")) in wanted]
    return np.asarray(idxs, dtype=np.int64)

# ---------------------------------------------------------------------------
# Trial iterators (unchanged)
# ---------------------------------------------------------------------------

def iter_trials_from_zip(volumes_dir, pattern="*.zip.*", subdir_keyword="EEG_preprocessed",
                          only_subjects=None):
    zf = open_concat_zip(volumes_dir, pattern=pattern)
    try:
        yield from _iter_trials_from_zipfile(zf, subdir_keyword=subdir_keyword,
                                              only_subjects=only_subjects)
    finally:
        try: inner = zf.fp
        except Exception: inner = None
        try: zf.close()
        finally:
            if inner:
                try: inner.close()
                except Exception: pass

def iter_trials_from_mat_dir(mat_dir, pattern="*.mat", only_subjects=None, recursive=True):
    d = Path(mat_dir)
    if not d.exists():
        raise FileNotFoundError(f"mat_dir does not exist: {d}")
    all_mats = sorted(d.rglob(pattern) if recursive else d.glob(pattern))
    if not all_mats:
        raise FileNotFoundError(f"No files matched {pattern!r} under {d}")
    if only_subjects:
        wanted = set(map(str, only_subjects))
        all_mats = [p for p in all_mats if p.stem in wanted]
    all_mats.sort(key=lambda p: _natural_key(p.stem))
    for mp in all_mats:
        subject = mp.stem
        data = loadmat(str(mp), verify_compressed_data_integrity=False)
        fields = sorted([(int(k), k) for k in data if not k.startswith("__") and k.isdigit()])
        for fid, name in fields:
            arr = np.asarray(data[name])
            if arr.ndim != 2 or arr.shape[0] != 62: continue
            sid, tin = trial_field_to_session_trial(fid)
            yield RawTrial(subject=subject, session_id=sid, trial_id=tin, field_id=fid, eeg=arr)
        del data

def iter_trials_from_modelscope(dataset_id, pattern="*.zip.*", scratch_dir="./_ms_cache",
                                 revision="master", token=None, subdir_keyword="EEG_preprocessed",
                                 only_subjects=None, max_resident_volumes=2):
    from .ms_download import download_one_file, list_dataset_files, login_if_token
    login_if_token(token)
    listing = list_dataset_files(dataset_id, revision=revision, token=token)
    import fnmatch as _fn
    listing = [f for f in listing if _fn.fnmatch(f.get("Path", ""), pattern)]
    if not listing:
        raise RuntimeError(f"No remote volumes matched {pattern}")
    listing.sort(key=lambda f: volume_sort_key(f.get("Path", "")))
    sizes = [(f["Path"], int(f.get("Size", 0) or 0)) for f in listing]
    Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    def _fetch(n): return str(download_one_file(dataset_id, n, scratch_dir, revision=revision, token=token))
    def _evict(p):
        try: p.unlink(missing_ok=True)
        except Exception: pass
    zf, stream = open_remote_concat_zip(sizes_in_order=sizes, fetcher=_fetch, evicter=_evict,
                                         max_resident=max_resident_volumes, pin_last=True, warmup_last=True)
    try:
        yield from _iter_trials_from_zipfile(zf, subdir_keyword=subdir_keyword,
                                              only_subjects=only_subjects, stream=stream)
    finally:
        try: zf.close()
        finally: stream.close()

def iter_trials_from_modelscope_single_file(dataset_id, path_in_repo="SEED-VII.zip",
                                              revision="master", token=None,
                                              subdir_keyword="EEG_preprocessed",
                                              only_subjects=None, cache_mb=256, chunk_mb=8):
    import zipfile
    from .remote_range import open_dataset_file_as_range_stream
    stream = open_dataset_file_as_range_stream(dataset_id=dataset_id, path_in_repo=path_in_repo,
                                                revision=revision, token=token,
                                                chunk_size=chunk_mb*1024**2, cache_bytes=cache_mb*1024**2)
    zf = zipfile.ZipFile(stream, mode="r")
    try:
        yield from _iter_trials_from_zipfile(zf, subdir_keyword=subdir_keyword,
                                              only_subjects=only_subjects, stream=None)
    finally:
        try: zf.close()
        finally:
            try: stream.close()
            except Exception: pass

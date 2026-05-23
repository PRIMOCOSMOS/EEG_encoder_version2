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
    open_concat_zip,
)


# ---------------------------------------------------------------------------
# 1) save_info → intensity
# ---------------------------------------------------------------------------

_FILENAME_RE = re.compile(
    r"^(?P<subject>[^_/\\]+)_(?P<date>[^_/\\]+)_(?P<session>\d+)_save_info\.csv$",
    re.IGNORECASE,
)


def _parse_save_info_csv(path: Path) -> Dict[int, float]:
    """Parse one save_info csv -> {trial_id_1based: intensity 0..1}.

    SEED-VII official说明：每个 movie clip 都有一行/列 score ∈ [0,1]，
    指示 targeted emotion 的诱发成功度。这里做兼容性解析：
    - 优先寻找名为 'score' / 'intensity' / 'rating' 的列
    - 若是单列纯数字（20 行）则直接按行序号取
    - 若是宽表（1 行 × N 列）则按列序取
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
    has_header = any(not _is_floatish(c) for c in header)

    scores: List[float] = []
    if has_header:
        target_col = -1
        lowered = [c.strip().lower() for c in header]
        for cand in ("score", "intensity", "rating", "feedback"):
            if cand in lowered:
                target_col = lowered.index(cand)
                break
        if target_col >= 0:
            for r in rows[1:]:
                if target_col < len(r) and _is_floatish(r[target_col]):
                    scores.append(_to_unit(float(r[target_col])))
        else:
            # fall back: take the last numeric column
            for r in rows[1:]:
                for c in reversed(r):
                    if _is_floatish(c):
                        scores.append(_to_unit(float(c)))
                        break
    else:
        # no header
        if len(rows) == 1 and len(rows[0]) >= 5:
            # wide row
            for c in rows[0]:
                if _is_floatish(c):
                    scores.append(_to_unit(float(c)))
        else:
            for r in rows:
                for c in r:
                    if _is_floatish(c):
                        scores.append(_to_unit(float(c)))
                        break

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
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        subject = m.group("subject")
        session = int(m.group("session"))
        scores = _parse_save_info_csv(p)
        for tid, val in scores.items():
            out[(subject, session, tid)] = float(val)
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


def iter_trials_from_zip(
    volumes_dir: os.PathLike,
    pattern: str = "*.zip.*",
    subdir_keyword: str = "EEG_preprocessed",
    only_subjects: Optional[Sequence[str]] = None,
) -> Iterator[RawTrial]:
    """Stream every (subject × 80 trials) from the multi-volume zip without disk extraction.

    Each `.mat` is read into memory transiently (≈ tens of MB), parsed, then discarded.
    """
    zf = open_concat_zip(volumes_dir, pattern=pattern)
    try:
        members = list(iter_mat_members(zf, subdir_keyword=subdir_keyword))
        if only_subjects:
            wanted = set(map(str, only_subjects))
            members = [m for m in members if Path(m.filename).stem in wanted]
        members.sort(key=lambda m: _natural_key(Path(m.filename).stem))
        for info in members:
            subject = Path(info.filename).stem  # "1".."20"
            raw_bytes = extract_mat_bytes(zf, info)
            data = loadmat(
                io.BytesIO(raw_bytes),
                verify_compressed_data_integrity=False,
            )
            # collect numbered fields ("1".."80"), preserve numeric order
            fields = []
            for k in data.keys():
                if k.startswith("__"):
                    continue
                if k.isdigit():
                    fields.append((int(k), k))
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
    finally:
        zf.close()


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

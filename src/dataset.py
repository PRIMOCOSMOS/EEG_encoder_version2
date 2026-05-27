"""SEED-VII dataset utilities — 重构版.

核心变化：
- 预处理输出为 per-subject .npz 文件 (共 20 个)
- 训练时可按需加载单/多个被试的 npz
- 支持 trial-level 划分以避免数据泄漏
- 支持从本地 .mat 目录或 ModelScope 远端读取原始数据
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import EMOTION_TO_IDX, trial_field_to_session_trial, trial_id_to_emotion


# --------------------------------------------------------------------------
# 数据加载 (per-subject npz)
# --------------------------------------------------------------------------

def load_subject_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load a single subject's preprocessed npz.

    Returns (X, y, s, meta) where:
      X: (N, 62, 800) float32
      y: (N,) int64
      s: (N,) float32
      meta: list of dicts
    """
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    s = data["s"]
    meta = [json.loads(m) for m in data["meta"]]
    return X, y, s, meta


def load_multi_subject_npz(
    npz_dir: str,
    subjects: Optional[List[str]] = None,
    pattern: str = "*.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load and concatenate multiple subject npz files from a directory.

    If subjects is None, loads all .npz files found.
    Returns (X_all, y_all, s_all, meta_all).
    """
    d = Path(npz_dir)
    if subjects:
        paths = []
        for s in subjects:
            p = d / f"{s}.npz"
            if p.exists():
                paths.append(p)
            else:
                print(f"[WARN] Subject npz not found: {p}")
    else:
        paths = sorted(d.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No npz files found in {npz_dir}")

    X_list, y_list, s_list, meta_all = [], [], [], []
    for p in paths:
        X, y, s, meta = load_subject_npz(str(p))
        X_list.append(X)
        y_list.append(y)
        s_list.append(s)
        meta_all.extend(meta)
        print(f"[LOAD] {p.name}: {X.shape[0]} windows")

    return (np.concatenate(X_list, axis=0),
            np.concatenate(y_list, axis=0),
            np.concatenate(s_list, axis=0),
            meta_all)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class EEGWindowArrayDataset(Dataset):
    """In-memory dataset over a subset of preprocessed windows."""

    def __init__(self, X: np.ndarray, y: np.ndarray, s: np.ndarray,
                 indices: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.s = torch.from_numpy(s.astype(np.float32))
        self.indices = indices if indices is not None else np.arange(len(y))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        x = self.X[i].unsqueeze(0)  # (1, 62, 800)
        return x, self.y[i], self.s[i]


# --------------------------------------------------------------------------
# Trial-level split
# --------------------------------------------------------------------------

@dataclass
class TrialKey:
    subject: str
    session_id: int
    trial_id: int

    def as_tuple(self) -> Tuple[str, int, int]:
        return (self.subject, self.session_id, self.trial_id)


def split_trials_from_meta(
    meta: list,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Split window indices by trial membership to avoid data leakage.

    Design.md 原则：先切分 trial 列表，再在各自集合内独立做窗口化和归一化。
    由于预处理已在 per-subject 级别完成（z-score 独立），这里只需按 trial 分割索引。

    Returns dict with keys 'train', 'val', 'test' -> np.ndarray of window indices.
    """
    # Extract unique trial keys and their window indices
    trial_to_windows: Dict[Tuple[str, int, int], List[int]] = {}
    for i, m in enumerate(meta):
        key = (str(m["subject"]), int(m["session_id"]), int(m["trial_id"]))
        if key not in trial_to_windows:
            trial_to_windows[key] = []
        trial_to_windows[key].append(i)

    trial_keys = list(trial_to_windows.keys())
    n = len(trial_keys)

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))

    test_trial_idx = indices[:n_test]
    val_trial_idx = indices[n_test:n_test + n_val]
    train_trial_idx = indices[n_test + n_val:]

    result: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for split_name, trial_idx_arr in [("test", test_trial_idx),
                                       ("val", val_trial_idx),
                                       ("train", train_trial_idx)]:
        for ti in trial_idx_arr:
            tk = trial_keys[ti]
            result[split_name].extend(trial_to_windows[tk])

    return {k: np.array(sorted(v), dtype=np.int64) for k, v in result.items()}


def filter_by_subjects(meta: list, subjects: List[str]) -> np.ndarray:
    """Return indices of windows whose subject is in the given list."""
    subjects_set = set(str(s) for s in subjects)
    return np.array([i for i, m in enumerate(meta)
                     if str(m.get("subject", "")) in subjects_set], dtype=np.int64)


# --------------------------------------------------------------------------
# .mat 文件读取工具
# --------------------------------------------------------------------------

@dataclass
class TrialData:
    subject: str
    session_id: int
    trial_id: int
    field_id: int
    eeg: np.ndarray             # (62, T)
    emotion_code: str
    label_idx: int


def iter_trials_from_mat(mat_path: str) -> Iterator[TrialData]:
    """Iterate over all trials in a single .mat file.

    Supports both MAT v7.3 (HDF5) and older formats.
    """
    subject = Path(mat_path).stem
    field_ids = []

    # Try HDF5 first
    try:
        import h5py
        with h5py.File(mat_path, "r") as f:
            for k in f.keys():
                if k.startswith("#"):
                    continue
                if k.isdigit():
                    field_ids.append(int(k))
        field_ids.sort()
        for fid in field_ids:
            with h5py.File(mat_path, "r") as f:
                eeg = np.array(f[str(fid)], dtype=np.float64)
                if eeg.ndim == 2 and eeg.shape[0] != 62 and eeg.shape[1] == 62:
                    eeg = eeg.T
            sid, tid = trial_field_to_session_trial(fid)
            code = trial_id_to_emotion(sid, tid)
            yield TrialData(
                subject=subject, session_id=sid, trial_id=tid,
                field_id=fid, eeg=eeg,
                emotion_code=code, label_idx=EMOTION_TO_IDX[code])
        return
    except Exception:
        pass

    # Fallback: scipy
    import scipy.io as sio
    mat = sio.loadmat(mat_path)
    for k in mat.keys():
        if k.startswith("_"):
            continue
        if k.isdigit():
            field_ids.append(int(k))
    field_ids.sort()
    for fid in field_ids:
        eeg = np.array(mat[str(fid)], dtype=np.float64)
        if eeg.ndim == 2 and eeg.shape[0] != 62 and eeg.shape[1] == 62:
            eeg = eeg.T
        sid, tid = trial_field_to_session_trial(fid)
        code = trial_id_to_emotion(sid, tid)
        yield TrialData(
            subject=subject, session_id=sid, trial_id=tid,
            field_id=fid, eeg=eeg,
            emotion_code=code, label_idx=EMOTION_TO_IDX[code])


def load_save_info_intensity(save_info_dir: str) -> Dict[Tuple[str, int, int], float]:
    """Load continuous intensity labels from save_info CSV files.

    Returns {(subject, session_id, trial_id): mean_intensity}.
    """
    import csv
    d = Path(save_info_dir)
    result: Dict[Tuple[str, int, int], float] = {}

    for csv_path in sorted(d.rglob("*_save_info.csv")):
        subject = csv_path.stem.replace("_save_info", "")
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fid = int(row.get("trial_id", row.get("field_id", 0)))
                    if fid <= 0:
                        continue
                    intensity = float(row.get("intensity", row.get("mean_intensity", 0.5)))
                    sid, tid = trial_field_to_session_trial(fid)
                    result[(subject, sid, tid)] = np.clip(intensity, 0.0, 1.0)
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")

    return result

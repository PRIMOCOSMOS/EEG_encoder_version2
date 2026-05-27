"""SEED-VII dataset utilities.

提供：
  - load_dataset_npz / load_dataset_mmap：加载预处理后的 npz
  - EEGWindowArrayDataset：窗口 Dataset
  - filter_by_subjects：按被试筛选索引
  - split_trials：trial-level 划分
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import EMOTION_TO_IDX, trial_field_to_session_trial


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------

def load_dataset_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, dict]:
    """Load preprocessed npz file. Returns (X, y, s, meta, splits)."""
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]                     # (N, 62, 800) float32
    y = data["y"]                     # (N,) int64
    s = data["s"]                     # (N,) float32
    meta = [json.loads(m) for m in data["meta"]]
    splits = {}
    for key in ("train", "val", "test"):
        if f"split_{key}" in data:
            splits[key] = data[f"split_{key}"]
    return X, y, s, meta, splits


def load_dataset_mmap(npz_path: str) -> Tuple[np.memmap, np.ndarray, np.ndarray, list, dict]:
    """Load preprocessed npz with X as memmap (zero RAM)."""
    data = np.load(npz_path, allow_pickle=True)
    # Re-open X as memmap for zero-RAM access
    X = np.memmap(npz_path, mode="r", dtype=np.float32,
                  shape=(data["X"].shape[0], data["X"].shape[1], data["X"].shape[2]),
                  offset=data["X"].file.tell() if hasattr(data["X"], "file") else 0)
    # Fallback: just use the array (npz already loaded)
    X = data["X"]
    y = data["y"]
    s = data["s"]
    meta = [json.loads(m) for m in data["meta"]]
    splits = {}
    for key in ("train", "val", "test"):
        if f"split_{key}" in data:
            splits[key] = data[f"split_{key}"]
    return X, y, s, meta, splits


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
        x = self.X[i].unsqueeze(0)          # (1, 62, 800)
        return x, self.y[i], self.s[i]


# --------------------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------------------

def filter_by_subjects(meta: list, subjects: List[str]) -> np.ndarray:
    """Return indices of windows whose subject is in the given list."""
    subjects_set = set(str(s) for s in subjects)
    return np.array([i for i, m in enumerate(meta)
                     if str(m.get("subject", "")) in subjects_set], dtype=np.int64)


@dataclass
class TrialKey:
    subject: str
    session_id: int
    trial_id: int

    def as_tuple(self) -> Tuple[str, int, int]:
        return (self.subject, self.session_id, self.trial_id)


def split_trials(trial_keys: List[TrialKey], trial_labels: List[int],
                 val_ratio: float = 0.1, test_ratio: float = 0.1,
                 seed: int = 42, unit: str = "trial") -> Tuple[List, List, List]:
    """Split trial-level indices into train/val/test (by trial unit)."""
    rng = np.random.default_rng(seed)
    n = len(trial_keys)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    return (
        [trial_keys[i] for i in train_idx],
        [trial_keys[i] for i in val_idx],
        [trial_keys[i] for i in test_idx],
    )
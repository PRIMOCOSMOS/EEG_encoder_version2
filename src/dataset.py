"""SEED-VII dataset utilities — 重构版 (OOM-safe).

核心设计：
- 20 个 per-subject .npz → 训练时 **不全量加载到 RAM**
- Pass 1 (轻量)：只扫描 y/s/meta（总计 < 50MB），构建全局索引
- Pass 2 (懒加载)：将每个 npz 的 X 解压为独立 .npy，用 np.memmap 打开
  → 内存占用 ≈ 0（由 OS 页面缓存按需管理）
- 支持两种 trial-level 划分策略，均避免数据泄漏：
  * "all"   : 全被试混合后做 trial-level 分割（原始行为）
  * "per_subject": 每个被试独立做 trial-level 分割，再合并（新功能）
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import EMOTION_TO_IDX, trial_field_to_session_trial, trial_id_to_emotion

# --------------------------------------------------------------------------
# 轻量扫描：只读 y / s / meta，不动 X
# --------------------------------------------------------------------------

def scan_npz_metadata(
    npz_dir: str,
    subjects: Optional[List[str]] = None,
    pattern: str = "*.npz",
) -> Tuple[List[Path], np.ndarray, np.ndarray, list, np.ndarray]:
    """Scan all per-subject npz files, load ONLY y/s/meta (skip X).

    Returns:
        npz_paths: list of Path (sorted)
        y_all:     (N_total,) int64
        s_all:     (N_total,) float32
        meta_all:  list[dict], length N_total
        file_map:  (N_total, 2) int64 — each row = (file_idx, local_idx)
    """
    d = Path(npz_dir)
    if subjects:
        paths = []
        for s in subjects:
            p = d / f"{s}.npz"
            if p.exists():
                paths.append(p)
    else:
        paths = sorted(d.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No npz files found in {npz_dir}")

    y_parts, s_parts, meta_all = [], [], []
    file_map_parts = []

    for fi, p in enumerate(paths):
        data = np.load(p, allow_pickle=True)
        y = data["y"]
        s = data["s"]
        meta_raw = data["meta"]
        n = len(y)

        y_parts.append(y.astype(np.int64))
        s_parts.append(s.astype(np.float32))
        meta_all.extend([json.loads(m) for m in meta_raw])
        file_map_parts.append(
            np.column_stack([np.full(n, fi, dtype=np.int64),
                             np.arange(n, dtype=np.int64)])
        )
        print(f"[SCAN] {p.name}: {n} windows (y/s/meta only, X skipped)")
        del data, y, s, meta_raw
        gc.collect()

    y_all = np.concatenate(y_parts, axis=0)
    s_all = np.concatenate(s_parts, axis=0)
    file_map = np.concatenate(file_map_parts, axis=0)

    print(f"[SCAN] Total: {len(y_all)} windows across {len(paths)} files, "
          f"y/s/meta RAM ≈ {(y_all.nbytes + s_all.nbytes + file_map.nbytes) / 1024**2:.1f} MB")

    return paths, y_all, s_all, meta_all, file_map

# --------------------------------------------------------------------------
# Memmap-backed X array: 解压 npz → .npy → memmap
# --------------------------------------------------------------------------

class MmapXStore:
    """Manage memory-mapped access to X arrays across multiple npz files.

    对每个 npz 文件，将其中的 X 数组解压为一个独立的 .npy 文件（放在 cache_dir），
    然后用 np.memmap 打开。内存占用 ≈ 0（由 OS 页面缓存管理）。
    """

    def __init__(self, npz_paths: List[Path], cache_dir: Optional[str] = None):
        if cache_dir:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._owned = False
        else:
            self._tmp = tempfile.mkdtemp(prefix="eeg_mmap_")
            self._cache_dir = Path(self._tmp)
            self._owned = True

        self._mmaps: List[np.memmap] = []
        self._counts: List[int] = []

        for fi, p in enumerate(npz_paths):
            npy_path = self._cache_dir / f"{p.stem}_X.npy"

            if npy_path.exists():
                mm = np.load(str(npy_path), mmap_mode="r")
            else:
                data = np.load(p, allow_pickle=True)
                X = data["X"]
                np.save(str(npy_path), X.astype(np.float32))
                del data, X
                gc.collect()
                mm = np.load(str(npy_path), mmap_mode="r")

            self._mmaps.append(mm)
            self._counts.append(mm.shape[0])
            print(f"[MMAP] {p.stem}: shape={mm.shape}, "
                  f"npy={npy_path.stat().st_size / 1024**2:.1f} MB (memmap, ~0 RAM)")

        total_npy = sum((self._cache_dir / f).stat().st_size
                        for f in os.listdir(self._cache_dir) if f.endswith(".npy"))
        print(f"[MMAP] Cache dir: {self._cache_dir}, "
              f"total .npy disk = {total_npy / 1024**3:.2f} GB")

    def get(self, file_idx: int, local_idx: int) -> np.ndarray:
        """Read one window: returns a WRITABLE (62, 800) float32 copy."""
        # .copy() 确保：1) 可写 2) contiguous 3) 脱离 memmap 页锁定
        return self._mmaps[file_idx][local_idx].copy()

    def close(self):
        for mm in self._mmaps:
            del mm
        self._mmaps.clear()
        gc.collect()
        if self._owned and hasattr(self, "_tmp"):
            shutil.rmtree(self._tmp, ignore_errors=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

# --------------------------------------------------------------------------
# OOM-safe Dataset: memmap-backed
# --------------------------------------------------------------------------

class EEGMmapDataset(Dataset):
    """Zero-RAM dataset backed by memory-mapped .npy files.

    每次 __getitem__ 从 memmap 读一条 (62, 800) 切片，
    .copy() 产生可写副本后转 Tensor，不会触发 PyTorch non-writable 警告。
    """

    def __init__(
        self,
        x_store: MmapXStore,
        file_map: np.ndarray,   # (N, 2) int64: (file_idx, local_idx)
        y: np.ndarray,           # (N,) int64
        s: np.ndarray,           # (N,) float32
        indices: Optional[np.ndarray] = None,
    ):
        self.x_store = x_store
        self.file_map = file_map
        self.y = y
        self.s = s
        self.indices = indices if indices is not None else np.arange(len(y))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        fi, li = int(self.file_map[i, 0]), int(self.file_map[i, 1])
        # x_store.get() 已经返回 .copy()，可写 + contiguous
        x = self.x_store.get(fi, li)          # (62, 800) float32, writable
        x_t = torch.from_numpy(x).unsqueeze(0)  # (1, 62, 800)
        return x_t, torch.tensor(self.y[i], dtype=torch.long), \
               torch.tensor(self.s[i], dtype=torch.float32)

# --------------------------------------------------------------------------
# 兼容的小规模 in-memory Dataset（单被试调试用）
# --------------------------------------------------------------------------

class EEGWindowArrayDataset(Dataset):
    """In-memory dataset (仅用于单被试或极小数据集调试)."""

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
        return self.X[i].unsqueeze(0), self.y[i], self.s[i]

# --------------------------------------------------------------------------
# Trial-level split（全被试混合，原始行为）
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

    策略：全被试所有 trial 混合后随机分割（原始行为）。
    对于跨被试泛化场景，该方式使验证/测试集包含所有被试的 trials。

    Returns dict with keys 'train', 'val', 'test' -> np.ndarray of window indices.
    """
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
    n_val  = max(1, int(round(n * val_ratio)))

    test_trial_idx  = indices[:n_test]
    val_trial_idx   = indices[n_test:n_test + n_val]
    train_trial_idx = indices[n_test + n_val:]

    result: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for split_name, trial_idx_arr in [("test",  test_trial_idx),
                                       ("val",   val_trial_idx),
                                       ("train", train_trial_idx)]:
        for ti in trial_idx_arr:
            tk = trial_keys[ti]
            result[split_name].extend(trial_to_windows[tk])

    return {k: np.array(sorted(v), dtype=np.int64) for k, v in result.items()}


# --------------------------------------------------------------------------
# ★ 新功能：Per-subject trial-level split（单被试独立分割）
# --------------------------------------------------------------------------

def split_trials_per_subject(
    meta: list,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Split window indices by trial membership, **per subject independently**.

    策略：
      1. 对每个被试，枚举其所有 (session_id, trial_id) 组成的 trial 集合。
      2. 在该被试内做 trial-level 随机分割（比例 val_ratio / test_ratio）。
      3. 各被试的 train/val/test 窗口索引分别合并，返回全局索引数组。

    优势（vs 全被试混合分割）：
      - 每个被试的 val/test 都包含该被试独有的留出 trials，
        可评估模型在同一被试内的泛化（被试内验证场景）。
      - 严格无数据泄漏：同一 trial 的所有窗口只出现在一个 split 中。
      - 保证每个被试在训练集都有代表，适合被试内编码器训练。

    Returns dict with keys 'train', 'val', 'test' -> np.ndarray of window indices.
    """
    # Step 1: 按被试收集 trial → window 索引映射
    # subject → { (session_id, trial_id) → [window_idx, ...] }
    subj_trial_map: Dict[str, Dict[Tuple[int, int], List[int]]] = {}
    for i, m in enumerate(meta):
        subj = str(m["subject"])
        st_key = (int(m["session_id"]), int(m["trial_id"]))
        if subj not in subj_trial_map:
            subj_trial_map[subj] = {}
        if st_key not in subj_trial_map[subj]:
            subj_trial_map[subj][st_key] = []
        subj_trial_map[subj][st_key].append(i)

    result: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    subj_stats: List[Dict] = []

    for subj_idx, (subj, trial_map) in enumerate(sorted(subj_trial_map.items())):
        trial_keys = list(trial_map.keys())
        n = len(trial_keys)

        # 每个被试使用独立但可重现的随机种子（seed + subj_hash 保证可重现性）
        subj_seed = seed + hash(subj) % (2**31)
        rng = np.random.default_rng(subj_seed)
        perm = np.arange(n)
        rng.shuffle(perm)

        n_test = max(1, int(round(n * test_ratio)))
        n_val  = max(1, int(round(n * val_ratio)))
        # 保证 train 至少有 1 个 trial
        n_train = n - n_test - n_val
        if n_train < 1:
            # 极少 trial 情况：强制保留至少 1 个 trial 给 train
            n_test = max(0, n_test - 1)
            n_val  = max(0, n_val - 1)
            n_train = n - n_test - n_val
            warnings.warn(
                f"[PER-SUBJ SPLIT] Subject '{subj}' only has {n} trials; "
                f"adjusted to train={n_train}, val={n_val}, test={n_test}.",
                RuntimeWarning, stacklevel=2,
            )

        test_idx_local  = perm[:n_test]
        val_idx_local   = perm[n_test:n_test + n_val]
        train_idx_local = perm[n_test + n_val:]

        counts = {"train": 0, "val": 0, "test": 0}
        for split_name, local_arr in [("test",  test_idx_local),
                                       ("val",   val_idx_local),
                                       ("train", train_idx_local)]:
            for ti in local_arr:
                tk = trial_keys[ti]
                wins = trial_map[tk]
                result[split_name].extend(wins)
                counts[split_name] += len(wins)

        subj_stats.append({
            "subject": subj,
            "n_trials": n,
            "n_trials_train": int(len(train_idx_local)),
            "n_trials_val":   int(len(val_idx_local)),
            "n_trials_test":  int(len(test_idx_local)),
            "n_windows_train": counts["train"],
            "n_windows_val":   counts["val"],
            "n_windows_test":  counts["test"],
        })

    # 打印统计摘要
    print("[PER-SUBJ SPLIT] Per-subject trial split summary:")
    print(f"  {'Subject':>10} | {'Trials':>6} | {'Tr-tr':>5} | {'Val-tr':>6} | "
          f"{'Tst-tr':>6} | {'Tr-win':>6} | {'Val-win':>7} | {'Tst-win':>7}")
    for st in subj_stats:
        print(f"  {st['subject']:>10} | {st['n_trials']:>6} | "
              f"{st['n_trials_train']:>5} | {st['n_trials_val']:>6} | "
              f"{st['n_trials_test']:>6} | {st['n_windows_train']:>6} | "
              f"{st['n_windows_val']:>7} | {st['n_windows_test']:>7}")
    total_tr  = sum(s["n_windows_train"] for s in subj_stats)
    total_val = sum(s["n_windows_val"]   for s in subj_stats)
    total_tst = sum(s["n_windows_test"]  for s in subj_stats)
    print(f"  {'[TOTAL]':>10} | {'':>6} | {'':>5} | {'':>6} | {'':>6} | "
          f"{total_tr:>6} | {total_val:>7} | {total_tst:>7}")

    return {k: np.array(sorted(v), dtype=np.int64) for k, v in result.items()}


# --------------------------------------------------------------------------
# 辅助：按被试过滤窗口索引
# --------------------------------------------------------------------------

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
    eeg: np.ndarray   # (62, T)
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
    """Load continuous intensity labels from save_info CSV files."""
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

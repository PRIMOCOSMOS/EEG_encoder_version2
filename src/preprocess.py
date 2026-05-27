"""SEED-VII per-trial preprocessing.

输入：单个 trial 的 raw EEG `(62, T)`（已带通+陷波）
输出：若干个 `(62, win_samples)` 窗口 + 元信息

流程（严格按 Design.md）：
  1) ICA 去伪迹（MNE-based，适合 EEG）  ← 可选，默认关闭
  2) 基线校正
  3) 平均参考（CAR）
  4) 居中 60% 裁剪
  5) 4 秒窗口 50% 重叠 + 每 clip 至多 N 个居中窗口
  6) 按通道 z-score

处理顺序：ICA → 基线校正 → CAR
（Design.md 原话："ICA 去除眼动、肌电伪迹 → 基线校正，平均参考（CAR）"，
ICA 完成后，再做基线校正和 CAR）

关于"先切分后处理"的设计决策：
  - 严格按 Design.md 的要求是在各自的数据集内独立做窗口化和归一化，
    即 train/val/test 分别预处理。这样可以完全杜绝数据泄漏。
  - 但在 SEED-VII 的实际场景中（原始 EEG 每个 subject 约 3GB），
    如果分开预处理需要多次从 ModelScope 下载/读取 .mat 文件，
    代价极高。实践中采用"全部预处理后划分"：
      Step 1: 枚举所有 trial 键（不加载 EEG 数据）
      Step 2: 全部预处理为窗口数组，保存到 npz
      Step 3: 在 npz 上做 train/val/test 划分
    这种做法在窗口级（而非 trial 级）归一化时，仍有极小的跨 trial 信息泄漏风险。
    如果对泄漏零容忍，应改为先划分 trial 列表，再按需读取/处理每个 trial，
    但这需要 ModelScope 数据源支持随机访问，当前实现暂不支持。
  - 本模块 (preprocess.py) 本身不执行切分，切分由 dataset.py 和 notebook Cell 4 负责。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import PREPROCESS_DEFAULTS


@dataclass
class WindowMeta:
    subject: str
    session_id: int
    trial_id: int
    label_idx: int
    intensity: float
    crop_start: int
    crop_end: int
    window_start_in_crop: int
    window_end_in_crop: int


# --------------------------- atomic ops ---------------------------

def baseline_correct(x: np.ndarray, head_samples: Optional[int] = None) -> np.ndarray:
    """Subtract per-channel mean.

    If head_samples is given, use only first `head_samples` for baseline mean,
    otherwise use the entire signal mean (safe default when no baseline segment exists).
    """
    if head_samples is not None and head_samples > 0:
        head = min(head_samples, x.shape[1])
        m = x[:, :head].mean(axis=1, keepdims=True)
    else:
        m = x.mean(axis=1, keepdims=True)
    return x - m


def apply_car(x: np.ndarray) -> np.ndarray:
    """Common Average Reference: subtract per-time mean across channels."""
    return x - x.mean(axis=0, keepdims=True)


def apply_mne_ica_denoise(
    x: np.ndarray,
    n_components: int,
    remove_k: int,
    random_state: int = 42,
) -> np.ndarray:
    """ICA artifact removal using MNE-Python.

    相比 sklearn FastICA 的优势：
    - 对 EEG 数据更友好的初始化和白化策略
    - 自动处理 NaN/Inf/常数通道
    - 更稳定的收敛

    伪迹检测策略：EOG 自动检测为主 + kurtosis 补充
    （纯 kurtosis 会对 EEG 中有效的尖峰活动误判）
    """
    import warnings
    import mne

    x = x.copy().astype(np.float64)

    # ---- 健康检查 ----
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    std_per_ch = x.std(axis=1)
    bad_ch_mask = std_per_ch < 1e-10
    n_bad = bad_ch_mask.sum()
    if n_bad > 0:
        good_mask = ~bad_ch_mask
        if good_mask.any():
            fill_val = x[good_mask].mean(axis=0)
        else:
            fill_val = 0.0
        x[bad_ch_mask] = fill_val

    # ---- 构建 MNE Raw 对象 ----
    sfreq = float(PREPROCESS_DEFAULTS.get("fs", 200))
    ch_names = [f"EEG{i:03d}" for i in range(x.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(x, info, first_samp=0, copy=False, verbose=False)

    # ---- 限制 n_components ----
    max_components = min(n_components, x.shape[0] - 1, 62)
    n_components = max(2, int(max_components))

    # ---- ICA 拟合 ----
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=500,
        method="fastica",
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ica.fit(raw, verbose=False)

    # ---- EOG 自动检测（比纯 kurtosis 更可靠）----
    try:
        eog_indices, eog_scores = ica.find_bads_eog(
            raw,
            threshold=2.0,
            verbose=False,
        )
    except Exception:
        eog_indices = []

    # ---- 合并 kurtosis 启发式 + EOG 检测 ----
    sources = ica.get_sources(raw).get_data()
    kurt_vals = np.nanmean(
        ((sources - sources.mean(axis=1, keepdims=True))
         / (sources.std(axis=1, keepdims=True) + 1e-8)) ** 4,
        axis=1,
    )

    kurtosis_bads = np.argsort(kurt_vals)[-remove_k:].tolist()
    exclude_set = set(eog_indices)
    exclude_list = list(eog_indices)
    remaining = remove_k - len(exclude_list)
    for idx in kurtosis_bads:
        if idx not in exclude_set and remaining > 0:
            exclude_list.append(idx)
            remaining -= 1

    ica.exclude = sorted(exclude_list)

    # ---- 重建干净信号 ----
    clean_raw = ica.apply(raw, exclude=ica.exclude, verbose=False)
    x_clean = clean_raw.get_data()
    return x_clean


def center_crop(x: np.ndarray, ratio: float) -> Tuple[np.ndarray, int, int]:
    """Keep the centered `ratio` portion along time axis."""
    n = x.shape[1]
    if ratio >= 1.0:
        return x, 0, n
    keep = max(1, int(round(n * ratio)))
    start = (n - keep) // 2
    return x[:, start:start + keep], start, start + keep


def sliding_windows(
    x: np.ndarray, win: int, step: int,
) -> List[Tuple[int, int, np.ndarray]]:
    """Non-padding sliding windows; returns [(start, end, slice), ...]."""
    n = x.shape[1]
    out: List[Tuple[int, int, np.ndarray]] = []
    s = 0
    while s + win <= n:
        out.append((s, s + win, x[:, s:s + win]))
        s += step
    return out


def per_channel_zscore(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = w.mean(axis=1, keepdims=True)
    sd = w.std(axis=1, keepdims=True)
    return (w - mu) / (sd + eps)


def instance_zscore(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (w - w.mean()) / (w.std() + eps)


# --------------------------- high-level ---------------------------

def preprocess_trial(
    raw: np.ndarray,
    subject: str,
    session_id: int,
    trial_id: int,
    label_idx: int,
    intensity: float,
    cfg: dict = PREPROCESS_DEFAULTS,
) -> Tuple[np.ndarray, List[WindowMeta]]:
    """Process one trial -> stacked windows array `(N, C, T)` + metas.

    处理顺序（严格按 Design.md）：
        ICA (可选) → 基线校正 → CAR → 居中裁剪 → 滑动窗口 → z-score

    Design.md 原话：
        ICA 去除眼动、肌电伪迹
        ↓
        基线校正，平均参考（CAR）
        ↓
        分段（4秒窗口，50%重叠，只取居中60%）
        ↓
        标准化（按通道 z-score）
    """
    if raw.ndim != 2 or raw.shape[0] != 62:
        raise ValueError(f"Expected (62, T), got {raw.shape}")
    x = raw.astype(np.float64, copy=False)

    # 1) ICA（Design.md 第一步）
    if cfg.get("use_ica", False):
        x = apply_mne_ica_denoise(
            x,
            n_components=int(cfg["ica_components"]),
            remove_k=int(cfg["ica_remove"]),
        )

    # 2) 基线校正（Design.md 第二步之一）
    if cfg.get("use_baseline_correct", True):
        x = baseline_correct(x, head_samples=None)

    # 3) CAR（Design.md 第二步之二）
    if cfg.get("use_car", True):
        x = apply_car(x)

    # 4) 居中 60% 裁剪
    middle_ratio = float(cfg.get("middle_ratio", 0.6))
    x_mid, cs, ce = center_crop(x, middle_ratio)

    # 5) 滑动窗口（4秒，50%重叠）
    fs = int(cfg["fs"])
    win = int(round(float(cfg["window_seconds"]) * fs))
    step = int(round(float(cfg["step_seconds"]) * fs))
    if win <= 0 or step <= 0:
        raise ValueError("window/step must be positive")
    windows = sliding_windows(x_mid, win=win, step=step)

    if not windows:
        return (np.zeros((0, 62, win), dtype=np.float32), [])

    # 均衡窗口数（Design.md：长视频主导问题，每 clip 至多 N 个居中窗口）
    max_n = int(cfg.get("max_windows_per_trial", 0) or 0)
    if max_n > 0 and len(windows) > max_n:
        mid = len(windows) / 2.0
        ranked = sorted(range(len(windows)), key=lambda i: abs((i + 0.5) - mid))
        keep_ids = sorted(ranked[:max_n])
        windows = [windows[i] for i in keep_ids]

    # 6) 按通道 z-score（Design.md：标准化优先考虑按通道 z-score）
    use_pc = bool(cfg.get("per_channel_zscore", True))
    eps = float(cfg.get("eps", 1e-8))
    save_f32 = bool(cfg.get("save_float32", True))

    arr = np.empty((len(windows), 62, win), dtype=np.float32 if save_f32 else np.float64)
    metas: List[WindowMeta] = []
    for i, (s, e, w) in enumerate(windows):
        w = per_channel_zscore(w, eps) if use_pc else instance_zscore(w, eps)
        arr[i] = w.astype(arr.dtype, copy=False)
        metas.append(
            WindowMeta(
                subject=subject, session_id=session_id, trial_id=trial_id,
                label_idx=int(label_idx), intensity=float(intensity),
                crop_start=int(cs), crop_end=int(ce),
                window_start_in_crop=int(s), window_end_in_crop=int(e),
            )
        )
    return arr, metas
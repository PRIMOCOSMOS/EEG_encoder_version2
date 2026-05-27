"""SEED-VII per-trial preprocessing.

输入：单个 trial 的 raw EEG `(62, T)`（已带通+陷波）
输出：若干个 `(62, win_samples)` 窗口 + 元信息

流程：
1) 基线校正（默认整段去均值；如有 baseline 段可改 head-N 秒）
2) ICA 去伪迹（MNE-based，适合 EEG 数据）
3) CAR 平均参考
4) 居中 60% 裁剪
5) 4 秒窗口 50% 重叠 + 每 clip 至多 N 个居中窗口（防长视频主导）
6) 按通道 z-score

注意：本模块不做训练/验证集划分，所有标准化是 **窗口级别 instance-wise**，
保证 trial 之间互不污染（划分由 dataset.py 完成；先切分、后处理由调用方保证）。
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
    """ICA artifact removal using MNE-Python (更适合 EEG).

    相比 sklearn FastICA 的优势：
    - 对 EEG 数据更友好的初始化和白化策略
    - 自动处理 NaN/Inf/常数通道
    - 更稳定的收敛

    流程：MNE RawArray → ICA.fit() → EOG-based 自动检测伪迹成分
          → 标记 remove_k 个最高 EOG 相关成分 → 重建信号
    """
    import warnings

    import mne

    # ---- 健康检查 ----
    x = x.copy().astype(np.float64)

    # 检查/处理 NaN / Inf
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # 检查/处理常数或几乎为零方差的通道
    std_per_ch = x.std(axis=1)
    bad_ch_mask = std_per_ch < 1e-10
    n_bad = bad_ch_mask.sum()
    if n_bad > 0:
        # 用有有限方差的通道的均值填充坏通道（保持空间结构）
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
        max_iter=500,        # MNE 默认 500，通常足够
        method="fastica",    # 与 sklearn FastICA 等价，但对 EEG 初始化更好
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ica.fit(raw, verbose=False)

    # ---- EOG 自动检测（比纯 kurtosis 更可靠）----
    # 尝试自动检测眼动相关成分
    try:
        eog_indices, eog_scores = ica.find_bads_eog(
            raw,
            threshold=2.0,   # Z-score 阈值，高于 2σ 的成分被标记
            verbose=False,
        )
    except Exception:
        eog_indices = []

    # ---- 合并 kurtosis 启发式 + EOG 检测 ----
    # 计算 kurtosis 并选择最高的前 remove_k 个（排除已由 EOG 标记的）
    sources = ica.get_sources(raw).get_data()          # (n_components, T)
    kurt_vals = np.nanmean(
        ((sources - sources.mean(axis=1, keepdims=True))
         / (sources.std(axis=1, keepdims=True) + 1e-8)) ** 4,
        axis=1,
    )

    # 按 kurtosis 排序，选择最高的 remove_k 个
    kurtosis_bads = np.argsort(kurt_vals)[-remove_k:].tolist()

    # 合并：EOG 检测的优先保留，kurtosis 补充剩余名额
    exclude_set = set(eog_indices)
    exclude_list = list(eog_indices)   # EOG 成分优先排除

    # 补足 remove_k - len(eog_indices) 个最高 kurtosis 成分
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

    处理顺序（调整后）：
        基线校正 → ICA (MNE) → CAR → 居中裁剪 → 滑动窗口 → z-score

    相比原版（基线校正 → CAR → ICA）的优势：
        ICA 在 CAR 之前：信号方差更均匀，通道间相关性更小，
        分解更稳定，不容易出现不收敛问题。
    """
    if raw.ndim != 2 or raw.shape[0] != 62:
        raise ValueError(f"Expected (62, T), got {raw.shape}")
    x = raw.astype(np.float64, copy=False)

    # 1) 基线校正
    if cfg.get("use_baseline_correct", True):
        x = baseline_correct(x, head_samples=None)

    # 2) ICA（调整到 CAR 之前）
    if cfg.get("use_ica", False):
        x = apply_mne_ica_denoise(
            x,
            n_components=int(cfg["ica_components"]),
            remove_k=int(cfg["ica_remove"]),
        )

    # 3) CAR（移到 ICA 之后）
    if cfg.get("use_car", True):
        x = apply_car(x)

    # 4) 居中 60% 裁剪
    middle_ratio = float(cfg.get("middle_ratio", 0.6))
    x_mid, cs, ce = center_crop(x, middle_ratio)

    # 5) 滑动窗口
    fs = int(cfg["fs"])
    win = int(round(float(cfg["window_seconds"]) * fs))
    step = int(round(float(cfg["step_seconds"]) * fs))
    if win <= 0 or step <= 0:
        raise ValueError("window/step must be positive")
    windows = sliding_windows(x_mid, win=win, step=step)

    if not windows:
        return (
            np.zeros((0, 62, win), dtype=np.float32),
            [],
        )

    # ---- balance: at most `max_windows_per_trial`, take the most centered ones ----
    max_n = int(cfg.get("max_windows_per_trial", 0) or 0)
    if max_n > 0 and len(windows) > max_n:
        mid = len(windows) / 2.0
        ranked = sorted(
            range(len(windows)),
            key=lambda i: abs((i + 0.5) - mid),  # closer to center first
        )
        keep_ids = sorted(ranked[:max_n])
        windows = [windows[i] for i in keep_ids]

    # 6) z-score
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
                subject=subject,
                session_id=session_id,
                trial_id=trial_id,
                label_idx=int(label_idx),
                intensity=float(intensity),
                crop_start=int(cs),
                crop_end=int(ce),
                window_start_in_crop=int(s),
                window_end_in_crop=int(e),
            )
        )
    return arr, metas
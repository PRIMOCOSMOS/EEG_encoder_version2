"""SEED-VII per-trial preprocessing.

输入：单个 trial 的 raw EEG `(62, T)`（已带通+陷波）
输出：若干个 `(62, win_samples)` 窗口 + 元信息

流程（严格按 Design.md）：
1) 基线校正（默认整段去均值；如有 baseline 段可改 head-N 秒）
2) CAR 平均参考
3) （可选）ICA 去伪迹
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


def apply_fastica_denoise(
    x: np.ndarray,
    n_components: int,
    remove_k: int,
    random_state: int = 42,
) -> np.ndarray:
    """Heuristic ICA artifact removal: drop top-k high-kurtosis components."""
    from sklearn.decomposition import FastICA  # lazy import

    xt = x.T  # (T, C)
    n_components = int(max(2, min(n_components, xt.shape[1], xt.shape[0] - 1)))
    ica = FastICA(
        n_components=n_components,
        random_state=random_state,
        whiten="unit-variance",
        max_iter=1000,
    )
    s = ica.fit_transform(xt)
    a = ica.mixing_
    kurt = np.mean(((s - s.mean(0)) / (s.std(0) + 1e-8)) ** 4, axis=0)
    remove_k = min(remove_k, s.shape[1])
    bad = np.argsort(kurt)[-remove_k:]
    s[:, bad] = 0.0
    return ((s @ a.T) + ica.mean_).T


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
    """Process one trial -> stacked windows array `(N, C, T)` + metas."""
    if raw.ndim != 2 or raw.shape[0] != 62:
        raise ValueError(f"Expected (62, T), got {raw.shape}")
    x = raw.astype(np.float64, copy=False)

    if cfg.get("use_baseline_correct", True):
        x = baseline_correct(x, head_samples=None)
    if cfg.get("use_car", True):
        x = apply_car(x)
    if cfg.get("use_ica", False):
        x = apply_fastica_denoise(
            x,
            n_components=int(cfg["ica_components"]),
            remove_k=int(cfg["ica_remove"]),
        )

    middle_ratio = float(cfg.get("middle_ratio", 0.6))
    x_mid, cs, ce = center_crop(x, middle_ratio)

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

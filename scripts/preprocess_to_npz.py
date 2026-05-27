#!/usr/bin/env python3
"""Preprocess SEED-VII .mat files into per-subject .npz files (无 ICA 版本).

核心流程：
  对 20 个 .mat 文件逐一进行：
    1) 读取 80 个 trial (62, T)
    2) 基线校正 → CAR → 居中60%裁剪 → 4秒窗口50%重叠 → z-score
    3) 保存为 {subject}.npz
  最终在 --output-dir 中生成 20 个 .npz 文件，每个对应一个被试。

输出 npz 字段：
  X: (N, 62, 800) float32
  y: (N,) int64
  s: (N,) float32
  meta: (N,) JSON-string

数据源支持：
  A. --mat-dir: 本地 .mat 文件目录
  B. --ms-dataset: 从 ModelScope 下载（流式，用完即删）

Design.md 关键原则：
  - 不做 ICA
  - 先切分后处理（z-score 在每个 trial 的每个窗口内独立完成，不跨 trial）
  - 每 clip 最多 max_windows_per_trial 个居中窗口，避免长视频主导
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PREPROCESS_DEFAULTS
from src.dataset import (
    TrialData,
    iter_trials_from_mat,
    load_save_info_intensity,
)
from src.labels import EMOTION_TO_IDX, trial_id_to_emotion
from src.preprocess import preprocess_trial


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess SEED-VII → per-subject .npz")

    # Data source (mutually exclusive)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mat-dir", type=str,
                   help="Local directory containing {1..20}.mat files")
    g.add_argument("--ms-dataset", type=str,
                   help="ModelScope dataset id, e.g. 'DEREKVERSE/SEED-VII'")

    # ModelScope options
    p.add_argument("--ms-zip-path", type=str, default="SEED-VII.zip",
                   help="Path of the zip inside the ModelScope dataset")
    p.add_argument("--ms-mat-prefix", type=str, default="EEG_preprocessed",
                   help="Prefix inside the zip where .mat files live")
    p.add_argument("--ms-scratch-dir", type=str, default="/tmp/ms_scratch",
                   help="Temp dir for ModelScope downloads")
    p.add_argument("--ms-token", type=str, default="")

    # Output
    p.add_argument("--output-dir", type=str, required=True,
                   help="Output directory for per-subject .npz files")

    # Save info (continuous labels)
    p.add_argument("--save-info-dir", type=str, default="",
                   help="Directory containing *_save_info.csv files")
    p.add_argument("--default-intensity", type=float, default=0.5,
                   help="Default intensity if save_info is unavailable")

    # Preprocessing parameters
    p.add_argument("--window-seconds", type=float,
                   default=float(PREPROCESS_DEFAULTS["window_seconds"]))
    p.add_argument("--step-seconds", type=float,
                   default=float(PREPROCESS_DEFAULTS["step_seconds"]))
    p.add_argument("--middle-ratio", type=float,
                   default=float(PREPROCESS_DEFAULTS["middle_ratio"]))
    p.add_argument("--max-windows-per-trial", type=int,
                   default=int(PREPROCESS_DEFAULTS["max_windows_per_trial"]))
    p.add_argument("--no-car", action="store_true")

    # Filter
    p.add_argument("--only-subjects", type=str, default="",
                   help="Comma-separated list of subject IDs to process (default: all)")

    # Misc
    p.add_argument("--compress", action="store_true",
                   help="Use np.savez_compressed (slower but smaller)")

    return p.parse_args()


def _get_mat_paths(mat_dir: str, only_subjects: Optional[List[str]] = None) -> List[Path]:
    """Find .mat files in a directory."""
    d = Path(mat_dir)
    all_mats = sorted(d.glob("*.mat"))
    if only_subjects:
        wanted = set(only_subjects)
        all_mats = [p for p in all_mats if p.stem in wanted]
    if not all_mats:
        # Try recursive
        all_mats = sorted(d.rglob("*.mat"))
        if only_subjects:
            wanted = set(only_subjects)
            all_mats = [p for p in all_mats if p.stem in wanted]
    return all_mats


def process_one_subject(
    mat_path: str,
    cfg: dict,
    intensities: Dict[Tuple[str, int, int], float],
    default_intensity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """Process one .mat file -> (X, y, s, meta_list)."""
    subject = Path(mat_path).stem

    X_list, y_list, s_list, meta_list = [], [], [], []

    for trial in iter_trials_from_mat(mat_path):
        key = (trial.subject, trial.session_id, trial.trial_id)
        s_val = float(intensities.get(key, default_intensity))

        arr, metas = preprocess_trial(
            raw=trial.eeg,
            subject=trial.subject,
            session_id=trial.session_id,
            trial_id=trial.trial_id,
            field_id=trial.field_id,
            label_idx=trial.label_idx,
            emotion_code=trial.emotion_code,
            intensity=s_val,
            cfg=cfg,
        )

        if arr.shape[0] == 0:
            continue

        X_list.append(arr)
        y_list.extend([trial.label_idx] * arr.shape[0])
        s_list.extend([s_val] * arr.shape[0])
        for m in metas:
            meta_list.append(m.__dict__.copy())

        # Release memory
        del arr, metas, trial
        gc.collect()

    if not X_list:
        return (np.zeros((0, 62, 800), dtype=np.float32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                [])

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    s = np.array(s_list, dtype=np.float32)
    return X, y, s, meta_list


def main():
    args = parse_args()
    cfg = dict(PREPROCESS_DEFAULTS)
    cfg["window_seconds"] = args.window_seconds
    cfg["step_seconds"] = args.step_seconds
    cfg["middle_ratio"] = args.middle_ratio
    cfg["max_windows_per_trial"] = args.max_windows_per_trial
    cfg["use_ica"] = False          # 明确关闭 ICA
    cfg["use_car"] = not args.no_car

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load intensities
    intensities: Dict[Tuple[str, int, int], float] = {}
    if args.save_info_dir:
        intensities = load_save_info_intensity(args.save_info_dir)
        print(f"[INFO] Loaded {len(intensities)} continuous labels from {args.save_info_dir}")
    else:
        print(f"[WARN] No save_info_dir; using default intensity={args.default_intensity}")

    # Get mat file paths
    only = [s.strip() for s in args.only_subjects.split(",") if s.strip()] \
        if args.only_subjects else None

    if args.mat_dir:
        mat_paths = _get_mat_paths(args.mat_dir, only)
    elif args.ms_dataset:
        # Download from ModelScope
        mat_paths = _download_mats_from_ms(args, only)
    else:
        print("[ERROR] Must specify --mat-dir or --ms-dataset")
        sys.exit(1)

    if not mat_paths:
        print("[ERROR] No .mat files found!")
        sys.exit(1)

    print(f"[INFO] Processing {len(mat_paths)} .mat files -> {out_dir}")
    print(f"[INFO] Config: window={cfg['window_seconds']}s, step={cfg['step_seconds']}s, "
          f"middle={cfg['middle_ratio']}, max_wpt={cfg['max_windows_per_trial']}, "
          f"CAR={'on' if cfg['use_car'] else 'off'}, ICA=off")

    total_windows = 0
    for mat_path in tqdm(mat_paths, desc="Subjects"):
        subject = Path(mat_path).stem
        out_path = out_dir / f"{subject}.npz"

        print(f"\n[PROCESSING] Subject {subject}: {mat_path}")
        X, y, s, meta_list = process_one_subject(
            str(mat_path), cfg, intensities, args.default_intensity)

        if X.shape[0] == 0:
            print(f"  [WARN] No windows produced for {subject}, skipping")
            continue

        meta_arr = np.asarray(
            [json.dumps(m, ensure_ascii=True) for m in meta_list], dtype=object)

        if args.compress:
            np.savez_compressed(out_path, X=X, y=y, s=s, meta=meta_arr)
        else:
            np.savez(out_path, X=X, y=y, s=s, meta=meta_arr)

        n = X.shape[0]
        total_windows += n
        print(f"  [OK] {subject}: {n} windows, X={X.shape} -> {out_path}")
        print(f"       Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Release memory
        del X, y, s, meta_list, meta_arr
        gc.collect()

    print(f"\n[DONE] Total: {total_windows} windows across {len(mat_paths)} subjects -> {out_dir}")


def _download_mats_from_ms(args, only_subjects) -> List[Path]:
    """Download .mat files from ModelScope dataset, extract from zip if needed."""
    scratch = Path(args.ms_scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    from src.ms_io import download_dataset_file

    token = args.ms_token or os.environ.get("MODELSCOPE_API_TOKEN", "")

    # Strategy: download the whole zip is impossible (160GB).
    # Instead, we need the .mat files to be available individually.
    # If the dataset has per-subject .mat files directly, download them.
    # Otherwise, we need to use the zip_stream approach.

    # For now, try to download individual .mat files if they exist in the dataset
    mat_dir = scratch / "mat_files"
    mat_dir.mkdir(parents=True, exist_ok=True)

    subjects = only_subjects or [str(i) for i in range(1, 21)]
    paths = []
    for subj in subjects:
        mat_name = f"{subj}.mat"
        local_path = mat_dir / mat_name
        if local_path.exists() and local_path.stat().st_size > 0:
            paths.append(local_path)
            continue
        try:
            p = download_dataset_file(
                dataset_id=args.ms_dataset,
                file_path=f"EEG_preprocessed/{mat_name}",
                local_dir=str(mat_dir),
                token=token,
            )
            paths.append(Path(p))
        except Exception as e:
            print(f"[WARN] Could not download {mat_name}: {e}")

    return paths


if __name__ == "__main__":
    main()

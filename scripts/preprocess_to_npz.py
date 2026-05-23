#!/usr/bin/env python3
"""Stream-preprocess SEED-VII into a single npz.

输入源（四选一，按优先顺序）：
    D) --mat-dir         **本地 .mat 目录**（Kaggle 数据集挂载场景；最简单）
    C) --ms-single-zip   ModelScope 上**已合并的单一 zip**（零落盘，HTTP Range 流式）
                         需先用 `scripts/merge_and_upload.py` 完成合并+上传
    B) --ms-dataset      ModelScope 上的**多分卷**（按需"一卷下载-一处理-一删"）
    A) --volumes-dir     本地分卷目录

严格执行 Design.md：
    1) **流式** 从分卷 zip 中按需读取每个 subject 的 .mat（处理完立即释放）。
       ModelScope 模式磁盘占用 ≤ 2×5.37GB ≈ 11GB。
    2) **先切分、后处理**：依据 (subject, session, trial) 做 trial-level 划分。
    3) 预处理：基线 → CAR →（可选 ICA）→ 居中 60% → 4s 窗口 50% 重叠 → 按通道 z-score。
    4) 标签：7 类整数 + 来自 save_info CSV 的连续强度 ∈ [0,1]（缺失时填默认值）。
       save_info 可以本地 --save-info-dir 也可以从同一个 MS dataset 拉取（--ms-save-info-include）。

输出 npz 字段：
    X (N,62,800) float32, y (N,) int64, s (N,) float32,
    meta (N,) JSON-string, split_train/val/test (M_*,) int64 索引
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PREPROCESS_DEFAULTS, TRAIN_DEFAULTS  # noqa: E402
from src.dataset import (  # noqa: E402
    TrialKey,
    iter_trials_from_mat_dir,
    iter_trials_from_modelscope,
    iter_trials_from_modelscope_single_file,
    iter_trials_from_zip,
    load_save_info_intensity,
    save_dataset_npz,
    split_trials,
)
from src.labels import EMOTION_TO_IDX, trial_id_to_emotion  # noqa: E402
from src.preprocess import preprocess_trial  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Stream-preprocess SEED-VII (local volumes OR ModelScope) to npz")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--volumes-dir", help="Local directory of split volumes")
    src.add_argument("--ms-dataset", help="ModelScope dataset id (multi-volume mode)")
    src.add_argument("--ms-single-zip", help="ModelScope dataset id (single merged zip mode; see --ms-single-zip-path)")
    src.add_argument("--mat-dir", help="Local directory of .mat files (e.g. Kaggle dataset mount)")
    ap.add_argument("--pattern", default="*.zip.*",
                    help="Glob for split volumes inside the source (default: *.zip.*)")
    ap.add_argument("--ms-revision", default="master")
    ap.add_argument("--ms-token", default="", help="ModelScope token (overrides env)")
    ap.add_argument("--ms-scratch-dir", default="./_ms_volumes_cache",
                    help="Where to cache downloaded volumes during streaming")
    ap.add_argument("--ms-max-resident-volumes", type=int, default=2,
                    help="How many split volumes are allowed on disk simultaneously")

    # Single-merged-zip mode (C)
    ap.add_argument("--ms-single-zip-path", default="SEED-VII.zip",
                    help="Path of the merged zip inside the dataset (default: SEED-VII.zip)")
    ap.add_argument("--ms-range-cache-mb", type=int, default=256,
                    help="In-memory LRU range cache size when reading the merged zip (MB)")
    ap.add_argument("--ms-range-chunk-mb", type=int, default=8,
                    help="HTTP Range GET chunk size when reading the merged zip (MB)")

    # Local mat-dir mode (D, e.g. Kaggle)
    ap.add_argument("--mat-pattern", default="*.mat",
                    help="Glob for .mat files inside --mat-dir (default: *.mat)")
    ap.add_argument("--mat-non-recursive", action="store_true",
                    help="If set, do not recurse into subdirectories of --mat-dir")

    # save_info
    ap.add_argument("--save-info-dir", default="",
                    help="Local folder of save_info CSVs (continuous labels)")
    ap.add_argument("--ms-save-info-include", default="",
                    help="If set with --ms-dataset, download save_info CSVs matching this glob "
                         "(e.g. 'save_info/*_save_info.csv') into --ms-scratch-dir/save_info")

    ap.add_argument("--output", required=True, help="Output .npz path")
    ap.add_argument("--subdir-keyword", default="EEG_preprocessed")

    # preprocess overrides
    ap.add_argument("--window-seconds", type=float, default=float(PREPROCESS_DEFAULTS["window_seconds"]))
    ap.add_argument("--step-seconds", type=float, default=float(PREPROCESS_DEFAULTS["step_seconds"]))
    ap.add_argument("--middle-ratio", type=float, default=float(PREPROCESS_DEFAULTS["middle_ratio"]))
    ap.add_argument("--max-windows-per-trial", type=int, default=int(PREPROCESS_DEFAULTS["max_windows_per_trial"]))
    ap.add_argument("--use-ica", action="store_true")
    ap.add_argument("--no-car", action="store_true")
    ap.add_argument("--default-intensity", type=float, default=1.0)

    # split
    ap.add_argument("--val-ratio", type=float, default=float(TRAIN_DEFAULTS["val_ratio"]))
    ap.add_argument("--test-ratio", type=float, default=float(TRAIN_DEFAULTS["test_ratio"]))
    ap.add_argument("--split-unit", choices=["trial", "subject", "session"],
                    default=str(TRAIN_DEFAULTS["split_unit"]))
    ap.add_argument("--seed", type=int, default=int(TRAIN_DEFAULTS["seed"]))

    # subset
    ap.add_argument("--only-subjects", default="", help="Comma-list of subject filenames, e.g. '1,2,3'")
    return ap.parse_args()


def _make_iter(args):
    """Factory returning a function `() -> iterator of RawTrial` for two passes."""
    only = list(args.only_subjects.split(",")) if args.only_subjects else None
    if args.mat_dir:
        def _it():
            return iter_trials_from_mat_dir(
                mat_dir=args.mat_dir,
                pattern=args.mat_pattern,
                only_subjects=only,
                recursive=not bool(args.mat_non_recursive),
            )
        return _it
    elif args.ms_single_zip:
        def _it():
            return iter_trials_from_modelscope_single_file(
                dataset_id=args.ms_single_zip,
                path_in_repo=args.ms_single_zip_path,
                revision=args.ms_revision,
                token=(args.ms_token or None),
                subdir_keyword=args.subdir_keyword,
                only_subjects=only,
                cache_mb=args.ms_range_cache_mb,
                chunk_mb=args.ms_range_chunk_mb,
            )
        return _it
    elif args.volumes_dir:
        def _it():
            return iter_trials_from_zip(
                args.volumes_dir, pattern=args.pattern,
                subdir_keyword=args.subdir_keyword,
                only_subjects=only,
            )
        return _it
    else:
        def _it():
            return iter_trials_from_modelscope(
                dataset_id=args.ms_dataset,
                pattern=args.pattern,
                scratch_dir=args.ms_scratch_dir,
                revision=args.ms_revision,
                token=(args.ms_token or None),
                subdir_keyword=args.subdir_keyword,
                only_subjects=only,
                max_resident_volumes=args.ms_max_resident_volumes,
            )
        return _it


def _resolve_save_info_dir(args) -> str:
    if args.save_info_dir:
        return args.save_info_dir
    ms_ds = args.ms_single_zip or args.ms_dataset
    if ms_ds and args.ms_save_info_include:
        from src.ms_download import download_save_info
        local = Path(args.ms_scratch_dir) / "save_info"
        download_save_info(
            dataset_id=ms_ds,
            local_dir=str(local),
            revision=args.ms_revision,
            token=(args.ms_token or None),
            include=[p.strip() for p in args.ms_save_info_include.split(",") if p.strip()],
        )
        return str(local)
    return ""


def main():
    args = parse_args()
    cfg = dict(PREPROCESS_DEFAULTS)
    cfg["window_seconds"] = args.window_seconds
    cfg["step_seconds"] = args.step_seconds
    cfg["middle_ratio"] = args.middle_ratio
    cfg["max_windows_per_trial"] = args.max_windows_per_trial
    cfg["use_ica"] = bool(args.use_ica)
    cfg["use_car"] = not bool(args.no_car)

    # ---- intensities ----
    intensities: Dict[Tuple[str, int, int], float] = {}
    save_info_dir = _resolve_save_info_dir(args)
    if save_info_dir:
        intensities = load_save_info_intensity(save_info_dir)
        print(f"[INFO] loaded {len(intensities)} continuous labels from {save_info_dir}")
    else:
        print("[WARN] No save_info provided; defaulting intensities to "
              f"{args.default_intensity}")

    iter_factory = _make_iter(args)

    # ---- pass 1: enumerate trial keys & labels (needed for stratified split) ----
    print("[INFO] Pass 1: enumerating trial keys & labels ...")
    trial_keys: List[TrialKey] = []
    trial_labels: List[int] = []
    for trial in iter_factory():
        code = trial_id_to_emotion(trial.session_id, trial.trial_id)
        y = EMOTION_TO_IDX[code]
        trial_keys.append(TrialKey(trial.subject, trial.session_id, trial.trial_id))
        trial_labels.append(y)
    print(f"[INFO] total trials enumerated: {len(trial_keys)}")

    train_keys, val_keys, test_keys = split_trials(
        trial_keys, trial_labels,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        seed=args.seed, unit=args.split_unit,
    )
    key_to_split: Dict[Tuple[str, int, int], str] = {}
    for k in train_keys: key_to_split[k.as_tuple()] = "train"
    for k in val_keys:   key_to_split[k.as_tuple()] = "val"
    for k in test_keys:  key_to_split[k.as_tuple()] = "test"
    print(f"[INFO] split trials: train={len(train_keys)} val={len(val_keys)} test={len(test_keys)}")

    # ---- pass 2: stream + preprocess ----
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    s_list: List[float] = []
    meta_list: List[dict] = []
    split_indices: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    print("[INFO] Pass 2: streaming + preprocessing ...")
    pbar = tqdm(iter_factory(), total=len(trial_keys))
    for trial in pbar:
        key_t = (trial.subject, trial.session_id, trial.trial_id)
        split_name = key_to_split.get(key_t)
        if split_name is None:
            continue
        code = trial_id_to_emotion(trial.session_id, trial.trial_id)
        y_idx = EMOTION_TO_IDX[code]
        s_val = float(intensities.get(key_t, args.default_intensity))

        arr, metas = preprocess_trial(
            raw=trial.eeg,
            subject=trial.subject,
            session_id=trial.session_id,
            trial_id=trial.trial_id,
            label_idx=y_idx,
            intensity=s_val,
            cfg=cfg,
        )
        if arr.shape[0] == 0:
            continue

        start_idx = len(y_list)
        for i in range(arr.shape[0]):
            X_list.append(arr[i:i+1])
            y_list.append(y_idx)
            s_list.append(s_val)
            m = metas[i].__dict__.copy()
            m["split"] = split_name
            m["emotion_code"] = code
            meta_list.append(m)
        end_idx = len(y_list)
        split_indices[split_name].extend(range(start_idx, end_idx))

        pbar.set_postfix({"subj": trial.subject, "sess": trial.session_id,
                          "trial": trial.trial_id, "n": end_idx - start_idx,
                          "split": split_name})
        del arr, metas

    if not X_list:
        raise RuntimeError("No windows produced. Check source / pattern / save_info.")

    X = np.concatenate(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    s = np.asarray(s_list, dtype=np.float32)
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in split_indices.items()}

    print(f"[INFO] final: X={X.shape}, y={y.shape}, s={s.shape}, "
          f"train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    save_dataset_npz(args.output, X, y, s, meta_list, split_assignment=splits)
    print(f"[OK] saved -> {args.output}")


if __name__ == "__main__":
    main()

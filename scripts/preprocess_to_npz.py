#!/usr/bin/env python3
"""Stream-preprocess SEED-VII into a single npz.

*** 内存友好版（OOM 修复） ***

主要改动相对原版：
1) Pass 1 只读 .mat 的字段名，不再 loadmat 整个 EEG（V7 用 h5py，旧版回退到 whosmat / 仅取键）。
   原版每个 subject 会把 ~3 GB EEG 读入内存只为了拿 trial key，极度浪费。
2) Pass 2 改为「磁盘 memmap 流式写入」：
   - 预先按 enumerated trial 数 × max_windows_per_trial 申请上限 memmap，
   - 处理一个 trial 立刻写入 memmap 对应区段，并 del 释放原数组，
   - 完全不再走 X_list.append + np.concatenate 的 2× 内存峰值路径。
3) Pass 2 末尾不再 np.savez_compressed（它会再吃一份内存做压缩缓冲）。
   改为：先把 memmap 截断为真实长度，再用 np.savez(uncompressed) 写出，
   或可选 --compress 走 streaming 拷贝（一次一个数组）。
4) preprocess_trial 内部我们额外用 in-place CAR / 基线 / zscore 的小路径
   通过 cfg["inplace"]=True 触发（见 src/preprocess.py 同步补丁）。
5) 顺手把 --tmp-dir 暴露出来；默认放在 --output 同级，磁盘吃紧时可指向更大的卷。

输出 npz 字段（与原版兼容）：
    X (N,62,800) float32, y (N,) int64, s (N,) float32,
    meta (N,) JSON-string, split_train/val/test (M_*,) int64 索引
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

from src.config import PREPROCESS_DEFAULTS, TRAIN_DEFAULTS  # noqa: E402
from src.dataset import (  # noqa: E402
    TrialKey,
    iter_trials_from_mat_dir,
    iter_trials_from_modelscope,
    iter_trials_from_modelscope_single_file,
    iter_trials_from_zip,
    load_save_info_intensity,
    split_trials,
)
from src.labels import EMOTION_TO_IDX, trial_id_to_emotion  # noqa: E402
from src.preprocess import preprocess_trial  # noqa: E402


# ------------------------------------------------------------------
# Lightweight Pass-1: enumerate (subject, session, trial) WITHOUT
# loading the EEG arrays. Saves ~3 GB of RSS per subject on SEED-VII.
# ------------------------------------------------------------------
def _light_enumerate_trial_keys_from_mat_dir(
    mat_dir: str,
    pattern: str,
    only_subjects: Optional[List[str]],
    recursive: bool,
) -> List[Tuple[str, int, int]]:
    from src.labels import trial_field_to_session_trial

    d = Path(mat_dir)
    if recursive:
        all_mats = sorted(d.rglob(pattern))
    else:
        all_mats = sorted(d.glob(pattern))
    if only_subjects:
        wanted = set(map(str, only_subjects))
        all_mats = [p for p in all_mats if p.stem in wanted]

    keys: List[Tuple[str, int, int]] = []
    for mp in all_mats:
        subject = mp.stem
        field_ids: List[int] = []

        # Try MAT v7.3 (HDF5) first -- this is what SEED-VII ships.
        used_h5 = False
        try:
            import h5py  # type: ignore

            with h5py.File(str(mp), "r") as f:
                for k in f.keys():
                    if k.startswith("#"):
                        continue
                    if k.isdigit():
                        field_ids.append(int(k))
            used_h5 = True
        except Exception:
            used_h5 = False

        if not used_h5:
            # Older MAT (<v7.3). Use whosmat: it only reads the directory,
            # not the matrix data.
            try:
                from scipy.io import whosmat
                info = whosmat(str(mp))
                for name, _shape, _dtype in info:
                    if name.isdigit():
                        field_ids.append(int(name))
            except Exception as e:
                print(f"[WARN] light-enum failed on {mp.name}: {e}; "
                      "falling back to full loadmat for this file.")
                from scipy.io import loadmat
                data = loadmat(str(mp), verify_compressed_data_integrity=False)
                for k in data.keys():
                    if k.startswith("__"):
                        continue
                    if k.isdigit():
                        field_ids.append(int(k))
                del data
                gc.collect()

        field_ids.sort()
        for fid in field_ids:
            session_id, trial_in_session = trial_field_to_session_trial(fid)
            keys.append((subject, session_id, trial_in_session))
    return keys


def parse_args():
    ap = argparse.ArgumentParser(
        description="Stream-preprocess SEED-VII (local volumes OR ModelScope) to npz "
                    "(memory-friendly, OOM-safe)"
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--volumes-dir", help="Local directory of split volumes")
    src.add_argument("--ms-dataset", help="ModelScope dataset id (multi-volume mode)")
    src.add_argument("--ms-single-zip", help="ModelScope dataset id (single merged zip mode)")
    src.add_argument("--mat-dir", help="Local directory of .mat files (e.g. Kaggle dataset mount)")
    ap.add_argument("--pattern", default="*.zip.*")
    ap.add_argument("--ms-revision", default="master")
    ap.add_argument("--ms-token", default="")
    ap.add_argument("--ms-scratch-dir", default="./_ms_volumes_cache")
    ap.add_argument("--ms-max-resident-volumes", type=int, default=2)

    ap.add_argument("--ms-single-zip-path", default="SEED-VII.zip")
    ap.add_argument("--ms-range-cache-mb", type=int, default=256)
    ap.add_argument("--ms-range-chunk-mb", type=int, default=8)

    ap.add_argument("--mat-pattern", default="*.mat")
    ap.add_argument("--mat-non-recursive", action="store_true")

    ap.add_argument("--save-info-dir", default="")
    ap.add_argument("--ms-save-info-include", default="")

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

    ap.add_argument("--only-subjects", default="")

    # NEW: memory-safety knobs
    ap.add_argument("--tmp-dir", default="",
                    help="Where to place the on-disk memmap shard (default: alongside --output)")
    ap.add_argument("--compress", action="store_true",
                    help="If set, write the final npz with np.savez_compressed (slower, extra RAM). "
                         "Default: np.savez (uncompressed, almost zero extra RAM).")
    return ap.parse_args()


def _make_iter(args):
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
    # Hint preprocess to avoid float64 copies where safe.
    cfg.setdefault("inplace", True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else out_path.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---- intensities ----
    intensities: Dict[Tuple[str, int, int], float] = {}
    save_info_dir = _resolve_save_info_dir(args)
    if save_info_dir:
        intensities = load_save_info_intensity(save_info_dir)
        print(f"[INFO] loaded {len(intensities)} continuous labels from {save_info_dir}")
    else:
        print(f"[WARN] No save_info; defaulting intensities to {args.default_intensity}")

    # ---- Pass 1 (LIGHT): enumerate keys without loading EEG ----
    print("[INFO] Pass 1 (light): enumerating trial keys without loading EEG ...")
    only = list(args.only_subjects.split(",")) if args.only_subjects else None
    if args.mat_dir:
        triples = _light_enumerate_trial_keys_from_mat_dir(
            args.mat_dir, args.mat_pattern, only,
            recursive=not bool(args.mat_non_recursive),
        )
    else:
        # For non-mat sources we still need the original (heavy) enumeration.
        print("[INFO] non --mat-dir source: falling back to full Pass-1 enumeration "
              "(still memory-friendly because each trial is released immediately).")
        triples = []
        iter_factory_pre = _make_iter(args)
        for trial in iter_factory_pre():
            triples.append((trial.subject, trial.session_id, trial.trial_id))
            # drop the EEG payload ASAP
            try:
                del trial
            except Exception:
                pass
        gc.collect()

    trial_keys: List[TrialKey] = [TrialKey(s, ss, tt) for (s, ss, tt) in triples]
    trial_labels: List[int] = [
        EMOTION_TO_IDX[trial_id_to_emotion(ss, tt)] for (_, ss, tt) in triples
    ]
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

    # ---- Geometry for the memmap shard ----
    fs = int(cfg["fs"])
    win_samples = int(round(float(cfg["window_seconds"]) * fs))
    max_wpt = int(cfg["max_windows_per_trial"])
    cap = len(trial_keys) * max_wpt   # absolute upper bound on N
    shard_path = tmp_dir / (out_path.stem + ".X.f32.memmap")
    print(f"[INFO] reserving memmap shard: shape=({cap},62,{win_samples}) "
          f"≈ {cap * 62 * win_samples * 4 / 1024**3:.2f} GB at {shard_path}")
    X_mm = np.memmap(shard_path, dtype=np.float32, mode="w+",
                     shape=(cap, 62, win_samples))

    y_arr = np.empty((cap,), dtype=np.int64)
    s_arr = np.empty((cap,), dtype=np.float32)
    meta_list: List[dict] = []
    split_indices: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    n_written = 0
    flush_every_trials = 25  # periodically flush memmap to keep dirty pages bounded

    # ---- Pass 2: stream + preprocess + write straight to memmap ----
    print("[INFO] Pass 2: streaming + preprocessing -> memmap ...")
    iter_factory = _make_iter(args)
    pbar = tqdm(iter_factory(), total=len(trial_keys))
    seen_trials = 0
    for trial in pbar:
        key_t = (trial.subject, trial.session_id, trial.trial_id)
        split_name = key_to_split.get(key_t)
        if split_name is None:
            # release EEG and continue
            try:
                del trial
            except Exception:
                pass
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
        # Drop the raw EEG payload immediately
        try:
            trial.eeg = None  # type: ignore[attr-defined]
        except Exception:
            pass
        del trial

        n = int(arr.shape[0])
        if n == 0:
            del arr, metas
            continue

        # Bounds check (should not trigger because cap = trials * max_wpt)
        if n_written + n > cap:
            raise RuntimeError(
                f"memmap overflow: trying to write {n} more rows past cap={cap}. "
                "Did you change max_windows_per_trial mid-run?"
            )

        # *** Copy into memmap and immediately release `arr` (no view-pinning) ***
        X_mm[n_written:n_written + n] = arr  # numpy will downcast to f32 as needed
        y_arr[n_written:n_written + n] = y_idx
        s_arr[n_written:n_written + n] = s_val

        start_idx = n_written
        for i in range(n):
            m = metas[i].__dict__.copy()
            m["split"] = split_name
            m["emotion_code"] = code
            meta_list.append(m)
        end_idx = start_idx + n
        split_indices[split_name].extend(range(start_idx, end_idx))
        n_written = end_idx

        pbar.set_postfix({"n_total": n_written, "split": split_name})
        del arr, metas

        seen_trials += 1
        if seen_trials % flush_every_trials == 0:
            X_mm.flush()
            gc.collect()

    if n_written == 0:
        del X_mm
        try:
            shard_path.unlink()
        except Exception:
            pass
        raise RuntimeError("No windows produced. Check source / pattern / save_info.")

    X_mm.flush()

    # ---- Finalize: write npz WITHOUT a giant in-RAM concatenate ----
    y_final = y_arr[:n_written]
    s_final = s_arr[:n_written]
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in split_indices.items()}
    meta_arr = np.asarray(
        [json.dumps(m, ensure_ascii=True) for m in meta_list], dtype=object
    )

    # Re-open the shard as a *read-only* memmap of the actually-used prefix so
    # np.savez can pull rows directly off disk without doubling memory.
    del X_mm
    gc.collect()
    X_view = np.memmap(shard_path, dtype=np.float32, mode="r",
                       shape=(cap, 62, win_samples))[:n_written]

    print(f"[INFO] final: X={X_view.shape}, y={y_final.shape}, s={s_final.shape}, "
          f"train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    payload = {
        "X": X_view,        # memmap-backed; np.savez streams it out
        "y": y_final,
        "s": s_final,
        "meta": meta_arr,
    }
    for k, v in splits.items():
        payload[f"split_{k}"] = v

    print(f"[INFO] writing npz -> {out_path}  (compress={args.compress})")
    if args.compress:
        # Compression DOES allocate per-array buffers internally; keep this off
        # on tight-RAM machines. We still avoid the 2x duplicate of the original
        # bug because X is already on disk (memmap).
        np.savez_compressed(out_path, **payload)
    else:
        np.savez(out_path, **payload)

    # Cleanup the temporary memmap shard.
    del X_view
    gc.collect()
    try:
        shard_path.unlink()
        print(f"[OK] removed temp shard {shard_path}")
    except Exception as e:
        print(f"[WARN] could not delete shard {shard_path}: {e}")

    print(f"[OK] saved -> {out_path}")


if __name__ == "__main__":
    main()

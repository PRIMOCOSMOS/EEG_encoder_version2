"""End-to-end test of the Kaggle path: --mat-dir + --save-info-dir → npz.

Simulates a Kaggle dataset mount with:
  - 3 subjects, each with a .mat containing field "1" (=session 1, trial 1)
    and field "2" (=session 1, trial 2)
  - matching save_info CSVs with continuous intensity scores
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import scipy.io

from src.dataset import load_save_info_intensity


def main():
    tmp = Path(tempfile.mkdtemp(prefix="kaggle_mat_dir_test_"))
    print(f"tmp = {tmp}")

    # ---- 1. Build the fake Kaggle mount ----
    mat_dir = tmp / "EEG_preprocessed"; mat_dir.mkdir()
    save_dir = tmp / "save_info"; save_dir.mkdir()

    # subjects 1..3, each .mat has 2 trials (fields "1" and "2")
    for sid in range(1, 4):
        out = {}
        for fid in (1, 2):
            # field "1" → session 1, trial 1; field "2" → session 1, trial 2
            # Make timepoints long enough that center 60% + 4s windows produces several
            # samples/sec = 200Hz; 60s of data => 12000 timepoints
            out[str(fid)] = np.random.randn(62, 12000).astype(np.float64)
        bio = io.BytesIO()
        scipy.io.savemat(bio, out)
        bio.seek(0)
        (mat_dir / f"{sid}.mat").write_bytes(bio.read())

    # ---- 2. Build save_info CSVs ----
    # Filename: subjectID_date_sessionID_save_info.csv
    # Content matches the real SEED-VII export style: each row is a movie entry and
    # the final column is the continuous intensity value.
    for sid in range(1, 4):
        rows = []
        for i in range(20):
            score = 0.4 + 0.02 * i + 0.01 * sid
            rows.append(
                f"emotion tasks,movie\\七类\\{sid}\\happy\\clip_{i + 1}.mp4,{score:.4f}"
            )
        (save_dir / f"{sid}_20240101_1_save_info.csv").write_text("\n".join(rows), encoding="utf-8")
        # Add a trigger_info.csv (not used by our pipeline but should not break things)
        trig = ["trigger,time", "1,0", "2,60"] * 10
        (save_dir / f"{sid}_20240101_1_trigger_info.csv").write_text("\n".join(trig), encoding="utf-8")

    parsed = load_save_info_intensity(save_dir)
    assert len(parsed) == 60, f"unexpected parsed labels: {len(parsed)}"
    assert abs(parsed[("1", 1, 1)] - 0.41) < 1e-9
    assert abs(parsed[("3", 1, 20)] - 0.81) < 1e-9

    # ---- 3. Run preprocess_to_npz.py via subprocess (real CLI test) ----
    out_npz = tmp / "out.npz"
    cmd = [sys.executable, "scripts/preprocess_to_npz.py",
           "--mat-dir", str(mat_dir),
           "--mat-pattern", "*.mat",
           "--save-info-dir", str(save_dir),
           "--output", str(out_npz),
           "--val-ratio", "0.2",
           "--test-ratio", "0.2",
           "--split-unit", "trial",
           "--max-windows-per-trial", "10"]
    print(f"$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    print("--- STDOUT (last 25 lines) ---")
    print("\n".join(res.stdout.splitlines()[-25:]))
    if res.returncode != 0:
        print("--- STDERR ---")
        print(res.stderr)
        raise SystemExit(f"preprocess failed (code {res.returncode})")

    # ---- 4. Validate the npz ----
    assert out_npz.is_file(), "no npz produced"
    z = np.load(out_npz, allow_pickle=True)
    print("\n--- npz validation ---")
    print("keys:", list(z.files))
    X = z["X"]; y = z["y"]; s = z["s"]
    print(f"X = {X.shape} {X.dtype}")
    print(f"y = {y.shape} unique = {sorted(set(y.tolist()))}")
    print(f"s = {s.shape} range = {float(s.min()):.3f}..{float(s.max()):.3f}")

    assert X.ndim == 3 and X.shape[1] == 62 and X.shape[2] == 800
    assert y.shape[0] == X.shape[0] == s.shape[0]
    assert X.dtype == np.float32
    # all 3 subjects × 2 trials = 6 trials; each up to 10 windows
    assert 1 <= X.shape[0] <= 6 * 10
    # intensity should be in [0,1] and varied (not all default 1.0)
    assert float(s.min()) < 1.0, "intensity should reflect save_info values, not default"
    assert float(s.max()) <= 1.0

    # Splits exist
    for split in ("train", "val", "test"):
        key = f"split_{split}"
        assert key in z.files, f"missing split {key}"
        idx = z[key]
        print(f"  split_{split}: {len(idx)} windows")
    total_split = sum(len(z[f"split_{s}"]) for s in ("train", "val", "test"))
    assert total_split == X.shape[0], f"split coverage {total_split} != X[0] {X.shape[0]}"

    # Per-channel zscore: each window's per-channel mean ≈ 0, std ≈ 1
    sample = X[0]   # (62, 800)
    ch_mean = sample.mean(axis=1)
    ch_std = sample.std(axis=1)
    assert float(np.abs(ch_mean).max()) < 1e-3, f"per-channel mean not zeroed: {ch_mean[:3]}"
    assert float(np.abs(ch_std - 1.0).max()) < 1e-3, f"per-channel std not 1.0: {ch_std[:3]}"
    print("  ✓ per-channel z-score OK")

    # Trial-level no leakage: gather unique (subject,session,trial) per split via meta
    import json as _json
    metas = [_json.loads(str(m)) for m in z["meta"]]
    by_split = {"train": set(), "val": set(), "test": set()}
    for i, m in enumerate(metas):
        key = (m["subject"], m["session_id"], m["trial_id"])
        for sp in ("train", "val", "test"):
            if i in set(z[f"split_{sp}"].tolist()):
                by_split[sp].add(key)
                break
    overlap_train_val = by_split["train"] & by_split["val"]
    overlap_train_test = by_split["train"] & by_split["test"]
    overlap_val_test = by_split["val"] & by_split["test"]
    assert not (overlap_train_val | overlap_train_test | overlap_val_test), \
        f"trial-level leakage detected!\n  tr∩va={overlap_train_val}\n  tr∩te={overlap_train_test}\n  va∩te={overlap_val_test}"
    print(f"  ✓ no trial-level leakage  train={len(by_split['train'])} val={len(by_split['val'])} test={len(by_split['test'])}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n✅ ALL TESTS PASS — Kaggle mat-dir path works end-to-end")


if __name__ == "__main__":
    main()

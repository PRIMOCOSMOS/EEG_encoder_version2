"""Encoder inference: dump embeddings + class/intensity predictions — OOM-safe."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CONFORMER_CONFIG, EEGNET_CONFIG
from .dataset import EEGMmapDataset, MmapXStore, scan_npz_metadata
from .model import build_model
from .trainer import resolve_device


@torch.no_grad()
def encode_from_npz_dir(
    data_dir: str,
    checkpoint_path: str,
    output_path: str,
    model_type: str = "eegnet",
    feature_type: str = "projected",
    batch_size: int = 256,
    device_arg: str = "auto",
    use_amp: bool = False,
    subjects: Optional[str] = None,
    mmap_cache_dir: str = "",
) -> None:
    device = resolve_device(device_arg)
    use_amp = bool(use_amp and device.type == "cuda")

    subj_list = [s.strip() for s in subjects.split(",")] if subjects else None

    # Lightweight scan — only y/s/meta
    npz_paths, y, s, meta, file_map = scan_npz_metadata(data_dir, subjects=subj_list)

    # Memmap X store
    cache_dir = mmap_cache_dir if mmap_cache_dir else None
    x_store = MmapXStore(npz_paths, cache_dir=cache_dir)

    cfg = EEGNET_CONFIG if model_type == "eegnet" else CONFORMER_CONFIG
    model = build_model(model_type, cfg).to(device)

    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

    ds = EEGMmapDataset(x_store, file_map, y, s)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    feats, cls_preds, int_preds = [], [], []
    for xb, yb, sb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            f = model.encode(xb, feature_type=feature_type)
            logits, pred_s, _ = model(xb)
        feats.append(f.detach().cpu().numpy())
        cls_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        int_preds.append(pred_s.detach().cpu().numpy())

    F_arr = np.concatenate(feats, axis=0)
    Yp = np.concatenate(cls_preds, axis=0)
    Sp = np.concatenate(int_preds, axis=0)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=F_arr.astype(np.float32),
        cls_pred=Yp.astype(np.int64),
        intensity_pred=Sp.astype(np.float32),
        labels=np.asarray(y, dtype=np.int64),
        intensity_true=np.asarray(s, dtype=np.float32),
        meta=np.asarray([json.dumps(m, ensure_ascii=True) for m in meta], dtype=object),
        feature_type=np.asarray(feature_type),
        model_type=np.asarray(model_type),
    )
    print(f"[DONE] features={F_arr.shape} -> {out}")

    x_store.close()

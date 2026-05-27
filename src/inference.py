"""Encoder inference: dump embeddings + class/intensity predictions."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import CONFORMER_CONFIG, EEGNET_CONFIG
from .dataset import EEGWindowArrayDataset, load_dataset_npz
from .model import build_model
from .trainer import resolve_device


class _EEGOnly(Dataset):
    """Minimal dataset for encoding (returns EEG tensor only, no labels)."""
    def __init__(self, x: np.ndarray):
        self.x = torch.from_numpy(x).float()
    def __len__(self) -> int:
        return self.x.shape[0]
    def __getitem__(self, idx: int):
        return self.x[idx].unsqueeze(0)


@torch.no_grad()
def encode_npz(
    data_path: str,
    checkpoint_path: str,
    output_path: str,
    model_type: str = "eegnet",
    feature_type: str = "projected",
    batch_size: int = 256,
    device_arg: str = "auto",
    use_amp: bool = False,
    subset: Optional[str] = None,
) -> None:
    device = resolve_device(device_arg)
    use_amp = bool(use_amp and device.type == "cuda")

    x, y, s, meta, splits = load_dataset_npz(data_path)
    if subset and subset in splits:
        sel = splits[subset]
        x = x[sel]; y = y[sel]; s = s[sel]
        meta = [meta[i] for i in sel.tolist()]

    # Select config based on model type
    cfg = EEGNET_CONFIG if model_type == "eegnet" else CONFORMER_CONFIG

    model = build_model(model_type, cfg).to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in ck:
        model.load_state_dict(ck["model"])
    else:
        model.load_state_dict(ck)
    model.eval()

    ds = _EEGOnly(x)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)

    feats, cls_preds, int_preds = [], [], []
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            f = model.encode(xb, feature_type=feature_type)
            logits, pred_s, _ = model(xb)
        feats.append(f.detach().cpu().numpy())
        cls_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        int_preds.append(pred_s.detach().cpu().numpy())

    F = np.concatenate(feats, axis=0)
    Yp = np.concatenate(cls_preds, axis=0)
    Sp = np.concatenate(int_preds, axis=0)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=F.astype(np.float32),
        cls_pred=Yp.astype(np.int64),
        intensity_pred=Sp.astype(np.float32),
        labels=np.asarray(y, dtype=np.int64),
        intensity_true=np.asarray(s, dtype=np.float32),
        meta=np.asarray([json.dumps(m, ensure_ascii=True) for m in meta], dtype=object),
        feature_type=np.asarray(feature_type),
        model_type=np.asarray(model_type),
    )
    print(f"[DONE] features={F.shape} -> {out}")
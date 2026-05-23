"""EEG-Conformer with dual heads (emotion classification + intensity regression).

参考 PRIMOCOSMOS/EEG_encoder（SEED-IV 实现）；保持参数量 ≈0.7–0.8M。

架构：
    Input (B, 1, 62, T=800)
        -> PatchEmbedding [time conv + spatial depthwise + BN + ELU + avgpool + dropout]
        -> + positional embedding
        -> TransformerEncoder × 6 (heads=10, ffn=160)
        -> Flatten -> Linear(N_tokens * D -> head_hidden)  == projected embedding
           |\
           | --> classifier  : Linear(head_hidden -> 7), softmax in loss
           \\ --> intensity   : Linear(head_hidden -> H) -> ELU -> Linear(H -> 1) -> Sigmoid
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import CONFORMER_CONFIG


class PatchEmbedding(nn.Module):
    def __init__(self, cfg: dict = CONFORMER_CONFIG):
        super().__init__()
        d = int(cfg["embed_dim"])
        ch = int(cfg["n_channels"])
        t_kernel = int(cfg["time_kernel"])
        t_pad = int(cfg["time_padding"])
        p_kernel = int(cfg["pool_kernel"])
        p_stride = int(cfg["pool_stride"])
        drop = float(cfg["dropout"])

        self.time_conv = nn.Conv2d(1, d, kernel_size=(1, t_kernel), padding=(0, t_pad), bias=False)
        self.spatial_conv = nn.Conv2d(d, d, kernel_size=(ch, 1), groups=d, bias=False)
        self.bn = nn.BatchNorm2d(d)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, p_kernel), stride=(1, p_stride))
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.time_conv(x)        # (B, D, C, T)
        x = self.spatial_conv(x)     # (B, D, 1, T)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)             # (B, D, 1, T')
        x = self.drop(x)
        return x.squeeze(2).transpose(1, 2)  # (B, T', D)


class IntensityHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))  # ∈ (0,1)


class EEGConformerDualHead(nn.Module):
    """EEG-Conformer with dual outputs."""

    def __init__(self, cfg: dict = CONFORMER_CONFIG):
        super().__init__()
        d = int(cfg["embed_dim"])
        n_tokens = int(cfg["n_tokens"])
        n_classes = int(cfg["n_classes"])
        n_heads = int(cfg["transformer_heads"])
        n_layers = int(cfg["transformer_layers"])
        ffn_dim = int(cfg["ffn_dim"])
        head_hidden = int(cfg["head_hidden"])
        drop = float(cfg["dropout"])
        int_hidden = int(cfg.get("intensity_head_hidden", 64))

        self.patch = PatchEmbedding(cfg)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=drop,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)

        self.flatten = nn.Flatten()
        self.feature_proj = nn.Linear(n_tokens * d, head_hidden)
        self.feature_act = nn.ELU()
        self.cls_drop = nn.Dropout(drop)

        self.classifier = nn.Linear(head_hidden, n_classes)
        self.intensity_head = IntensityHead(head_hidden, int_hidden, drop)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    # ---------- encoders ----------
    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch(x)
        if z.shape[1] != self.pos_embed.shape[1]:
            raise RuntimeError(
                f"Token count mismatch: got {z.shape[1]}, expected {self.pos_embed.shape[1]}"
            )
        z = z + self.pos_embed
        return self.encoder(z)            # (B, T', D)

    def encode(self, x: torch.Tensor, feature_type: str = "projected") -> torch.Tensor:
        feat = self.flatten(self.encode_tokens(x))
        if feature_type == "flatten":
            return feat
        if feature_type == "projected":
            return self.feature_act(self.feature_proj(feat))
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    # ---------- heads ----------
    def forward(self, x: torch.Tensor):
        feat = self.encode(x, feature_type="projected")
        feat_d = self.cls_drop(feat)
        logits = self.classifier(feat_d)
        intensity = self.intensity_head(feat_d).squeeze(-1)   # (B,)
        return logits, intensity, feat


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

"""EEGNet and EEGConformer dual-head models for SEED-VII.

模型选择（通过 --model-type 参数）：
  eegnet    → EEGNetDualHead   （轻量，约 5K 参数，适合小样本）
  conformer → EEGConformerDualHead（≈0.75M 参数，全 Transformer 编码器）

两个模型共享相同的双头接口（dual-head）：
  - Classifier: 7-class emotion softmax
  - Intensity: Sigmoid [0,1] continuous regression
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# EEGNet (Lawhern et al., 2018 — adapted for SEED-VII dual-head)
# =============================================================================

class EEGNetBlock1(nn.Module):
    def __init__(self, F1: int = 8, D: int = 2, Chans: int = 62,
                 kernLength: int = 100, dropout: float = 0.5):
        super().__init__()
        self.temporal_conv = nn.Conv2d(
            in_channels=1, out_channels=F1,
            kernel_size=(1, kernLength), padding=(0, kernLength // 2), bias=False
        )
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1, out_channels=F1 * D,
            kernel_size=(Chans, 1), groups=F1, bias=False
        )
        self.bn = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        return self.dropout(x)


class EEGNetBlock2(nn.Module):
    def __init__(self, F1: int = 8, D: int = 2, F2: int = 16, dropout: float = 0.5):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=F1 * D, out_channels=F1 * D,
            kernel_size=(1, 16), groups=F1 * D, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels=F1 * D, out_channels=F2, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        return self.dropout(x)


class IntensityHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class EEGNetDualHead(nn.Module):
    """EEGNet with dual outputs for SEED-VII emotion classification + intensity regression."""

    def __init__(
        self,
        Chans: int = 62,
        Samples: int = 800,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernLength: int = 100,
        dropout: float = 0.5,
        nb_classes: int = 7,
        int_hidden: int = 64,
    ):
        super().__init__()
        self.block1 = EEGNetBlock1(F1, D, Chans, kernLength, dropout)
        self.block2 = EEGNetBlock2(F1, D, F2, dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            x = self.block1(dummy)
            x = self.block2(x)
            self._flat_size = x.view(1, -1).size(1)

        self.classifier = nn.Linear(self._flat_size, nb_classes)
        self.intensity_head = IntensityHead(self._flat_size, int_hidden, dropout)

        print(f"[EEGNetDualHead] Feature dim: {self._flat_size}")

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x.view(x.size(0), -1)

    def encode(self, x: torch.Tensor, feature_type: str = "projected") -> torch.Tensor:
        return self._extract_features(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._extract_features(x)
        logits = self.classifier(features)
        intensity = self.intensity_head(features).squeeze(-1)
        return logits, intensity, features


# =============================================================================
# EEGConformer (Song et al., 2022 — adapted for SEED-VII dual-head)
# =============================================================================

class ConvBlock(nn.Module):
    """Temporal + Spatial conv block used before Transformer encoder."""

    def __init__(
        self,
        in_ch: int,
        time_kernel: int = 20,
        time_padding: int = 10,
        pool_kernel: int = 60,
        pool_stride: int = 12,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=(1, time_kernel),
                               padding=(0, time_padding), bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)

        # Depthwise spatial conv: (1, Ch) → learns spatial filter per temporal position
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=(in_ch, 1), groups=in_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch)

        self.pool = nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)

        x = self.pool(x)
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T]


class EEGConformerDualHead(nn.Module):
    """EEGConformer: Conv frontend + Transformer Encoder + dual heads.

    Architecture:
      Input (B, 1, 62, 800)
        → ConvBlock (temporal+spatial+pool)  → (B, embed_dim, 62, T')
        → Flatten channels into sequence      → (B, T', embed_dim)
        → Positional Encoding
        → Transformer Encoder (6 layers, 10 heads)
        → Mean pooling over time
        → [Classifier head → 7 classes] + [Intensity head → sigmoid]
    """

    def __init__(
        self,
        Chans: int = 62,
        Samples: int = 800,
        embed_dim: int = 40,
        time_kernel: int = 20,
        time_padding: int = 10,
        pool_kernel: int = 60,
        pool_stride: int = 12,
        transformer_layers: int = 6,
        transformer_heads: int = 10,
        ffn_dim: int = 160,
        head_hidden: int = 256,
        dropout: float = 0.5,
        nb_classes: int = 7,
        int_hidden: int = 64,
    ):
        super().__init__()
        # ---- Conv frontend ----
        self.conv_block = ConvBlock(
            in_ch=1,
            time_kernel=time_kernel,
            time_padding=time_padding,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            dropout=dropout,
        )

        # Compute number of temporal tokens after pooling
        after_conv_t = (Samples - pool_kernel) // pool_stride + 1
        assert after_conv_t > 0, f"Pool causes negative T: {after_conv_t}"

        self.embed_dim = embed_dim
        self.n_tokens = after_conv_t

        self.pos_enc = PositionalEncoding(embed_dim, max_len=after_conv_t + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # ---- Dual heads ----
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(head_hidden, nb_classes)
        self.intensity_head = IntensityHead(head_hidden, int_hidden, dropout)

        self._flat_size = embed_dim  # for encode() API

        print(f"[EEGConformerDualHead] embed={embed_dim}, tokens={after_conv_t}, "
              f"layers={transformer_layers}, heads={transformer_heads}")

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Conv frontend: (B, 1, 62, 800) → (B, 1, 62, T')
        x = self.conv_block(x)

        # Transpose: (B, 1, 62, T') → (B, T', 62) then project to embed_dim
        B = x.size(0)
        x = x.permute(0, 3, 2, 1).contiguous()          # (B, T', 62, 1)
        x = x.squeeze(-1)                                # (B, T', 62)

        # Project (62 → embed_dim)
        if x.size(-1) != self.embed_dim:
            x = F.linear(x, self._proj_weight, self._proj_bias)  # type: ignore

        x = self.pos_enc(x)
        x = self.transformer(x)                          # (B, T', embed_dim)

        # Mean pooling over time tokens
        x = x.mean(dim=1)                                 # (B, embed_dim)
        return x

    def encode(self, x: torch.Tensor, feature_type: str = "projected") -> torch.Tensor:
        return self._extract_features(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._extract_features(x)             # (B, embed_dim)
        h = self.fc(features)                            # (B, head_hidden)
        logits = self.classifier(h)
        intensity = self.intensity_head(h).squeeze(-1)
        return logits, intensity, features


# Make ConvBlock have _proj_weight available
def _make_conformer_projection(Chans, embed_dim):
    return nn.Linear(Chans, embed_dim)


# Patch EEGConformerDualHead __init__ to include projection
_original_conformer_init = None

def _patched_conformer_init(self, Chans, Samples, embed_dim, time_kernel, time_padding,
                             pool_kernel, pool_stride, transformer_layers, transformer_heads,
                             ffn_dim, head_hidden, dropout, nb_classes, int_hidden):
    super(EEGConformerDualHead, self).__init__()
    self.conv_block = ConvBlock(
        in_ch=1, time_kernel=time_kernel, time_padding=time_padding,
        pool_kernel=pool_kernel, pool_stride=pool_stride, dropout=dropout,
    )
    after_conv_t = (Samples - pool_kernel) // pool_stride + 1
    self.embed_dim = embed_dim
    self.n_tokens = after_conv_t

    self.proj = nn.Linear(Chans, embed_dim)
    self.pos_enc = PositionalEncoding(embed_dim, max_len=after_conv_t + 10)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim, nhead=transformer_heads, dim_feedforward=ffn_dim,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    self.fc = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, head_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
    )
    self.classifier = nn.Linear(head_hidden, nb_classes)
    self.intensity_head = IntensityHead(head_hidden, int_hidden, dropout)
    self._flat_size = embed_dim

    print(f"[EEGConformerDualHead] embed={embed_dim}, tokens={after_conv_t}, "
          f"layers={transformer_layers}, heads={transformer_heads}")


# Replace __init__ with the patched version
EEGConformerDualHead.__init__ = _patched_conformer_init


def build_model(model_type: str, cfg: dict) -> nn.Module:
    """Factory: build EEGNet or EEGConformer based on config dict."""
    if model_type == "eegnet":
        return EEGNetDualHead(
            Chans=cfg["n_channels"],
            Samples=cfg["n_timepoints"],
            F1=cfg["F1"],
            D=cfg["D"],
            F2=cfg["F2"],
            kernLength=cfg["kernLength"],
            dropout=cfg["dropout"],
            nb_classes=cfg["n_classes"],
            int_hidden=cfg.get("intensity_head_hidden", 64),
        )
    elif model_type == "conformer":
        return EEGConformerDualHead(
            Chans=cfg["n_channels"],
            Samples=cfg["n_timepoints"],
            embed_dim=cfg["embed_dim"],
            time_kernel=cfg["time_kernel"],
            time_padding=cfg["time_padding"],
            pool_kernel=cfg["pool_kernel"],
            pool_stride=cfg["pool_stride"],
            transformer_layers=cfg["transformer_layers"],
            transformer_heads=cfg["transformer_heads"],
            ffn_dim=cfg["ffn_dim"],
            head_hidden=cfg["head_hidden"],
            dropout=cfg["dropout"],
            nb_classes=cfg["n_classes"],
            int_hidden=cfg.get("intensity_head_hidden", 64),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'eegnet' or 'conformer'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_intensity_head(model: nn.Module) -> int:
    """Freeze all parameters in the intensity head. Returns count."""
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith("intensity_head"):
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def print_model_summary(model: nn.Module, Chans: int = 62, Samples: int = 800) -> None:
    print("=" * 70)
    print(type(model).__name__)
    print("=" * 70)
    print(model)
    print("-" * 70)
    total = count_parameters(model)
    print(f"Total trainable parameters: {total:,} ({total/1e6:.4f}M)")
    print(f"Input shape: (B, 1, {Chans}, {Samples})")
    print("Per-module breakdown:")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:25s}: {n:>10,}")
    print("=" * 70)


if __name__ == "__main__":
    # Test both models
    print("=" * 60)
    print("Testing EEGNetDualHead")
    print("=" * 60)
    net = EEGNetDualHead(Chans=62, Samples=800, F1=8, D=2, F2=16,
                         kernLength=100, dropout=0.5, nb_classes=7)
    print_model_summary(net)
    dummy = torch.randn(2, 1, 62, 800)
    logits, intensity, features = net(dummy)
    print(f"Input: {dummy.shape} → logits={logits.shape}, intensity={intensity.shape}, features={features.shape}")

    print()
    print("=" * 60)
    print("Testing EEGConformerDualHead")
    print("=" * 60)
    conformer = EEGConformerDualHead(
        Chans=62, Samples=800, embed_dim=40, time_kernel=20, time_padding=10,
        pool_kernel=60, pool_stride=12, transformer_layers=6,
        transformer_heads=10, ffn_dim=160, head_hidden=256,
        dropout=0.5, nb_classes=7,
    )
    print_model_summary(conformer)
    dummy = torch.randn(2, 1, 62, 800)
    logits, intensity, features = conformer(dummy)
    print(f"Input: {dummy.shape} → logits={logits.shape}, intensity={intensity.shape}, features={features.shape}")

    print()
    print("✅ Both models forward pass successful!")
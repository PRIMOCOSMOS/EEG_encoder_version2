"""EEGNet-based dual-head model for SEED-VII emotion classification + intensity regression.

Replaces the original EEG-Conformer with a more parameter-efficient EEGNet architecture
to mitigate overfitting on small EEG datasets.

Architecture (Lawhern et al., 2018 - adapted for SEED-VII):
  Input (B, 1, Chans, Samples)  [Chans=62, Samples=800]
  -> Block1: Temporal Conv(1, F1=8, kernLength=100)
             + Depthwise Conv(F1, F1*D, Chans, 1) [D=2]
             + BatchNorm + ELU + AvgPool(1, 4) + Dropout(0.5)
  -> Block2: Separable Conv [Depthwise(1, 16) + Pointwise(F1*D->F2=16)]
             + BatchNorm + ELU + AvgPool(1, 8) + Dropout(0.5)
  -> Flatten
  |-> Classifier Head: Linear -> 7 classes (softmax in loss)
  \-> Intensity Head: Linear -> ELU -> Dropout -> Linear -> Sigmoid [0,1]

Key anti-overfitting design:
  1. Depthwise separable convolutions drastically reduce parameters vs Transformer
  2. Early spatial pooling reduces dimensionality before complex interactions
  3. Strong dropout (0.5) + batchnorm + weight decay regularization
  4. No self-attention → less prone to memorizing small datasets
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetBlock1(nn.Module):
    """Block 1: Temporal filtering → Spatial depthwise convolution."""
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
    """Block 2: Separable convolution (spatial → temporal refinement)."""
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
    """Intensity regression head with sigmoid output [0, 1]."""
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class EEGNetDualHead(nn.Module):
    """EEGNet with dual outputs for SEED-VII: emotion classification + intensity regression."""

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
        print(f"[EEGNetDualHead] Classifier: {self._flat_size} -> {nb_classes}")
        print(f"[EEGNetDualHead] Intensity:   {self._flat_size} -> {int_hidden} -> 1")

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv blocks and return flattened features."""
        x = self.block1(x)
        x = self.block2(x)
        return x.view(x.size(0), -1)

    def encode(self, x: torch.Tensor, feature_type: str = "projected") -> torch.Tensor:
        """Feature extraction for inference (API-compatible with EEGConformerDualHead)."""
        return self._extract_features(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits: (B, nb_classes)
            intensity: (B,) in [0,1]
            features: (B, flat_size)
        """
        features = self._extract_features(x)
        logits = self.classifier(features)
        intensity = self.intensity_head(features).squeeze(-1)
        return logits, intensity, features


class EEGConformerDualHead(nn.Module):
    """Legacy alias — delegates to EEGNetDualHead for backward compatibility."""
    def __new__(cls, *args, **kwargs):
        import warnings
        warnings.warn(
            "EEGConformerDualHead is deprecated. Using EEGNetDualHead instead.",
            DeprecationWarning, stacklevel=2
        )
        return EEGNetDualHead(*args, **kwargs)


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
    """Print architecture + parameter breakdown."""
    print("=" * 70)
    print("EEGNet Dual-Head for SEED-VII")
    print("=" * 70)
    print(model)
    print("-" * 70)
    total = count_parameters(model)
    print(f"Total trainable parameters: {total:,}")
    print(f"Input shape: (B, 1, {Chans}, {Samples})")
    print("\nPer-module breakdown:")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:20s}: {n:>8,}")
    print("=" * 70)


if __name__ == "__main__":
    model = EEGNetDualHead(
        Chans=62, Samples=800, F1=8, D=2, F2=16,
        kernLength=100, dropout=0.5, nb_classes=7
    )
    print_model_summary(model)

    print("\nTesting forward pass...")
    dummy = torch.randn(2, 1, 62, 800)
    logits, intensity, features = model(dummy)
    print(f"  Input:     {dummy.shape}")
    print(f"  Logits:    {logits.shape}")
    print(f"  Intensity: {intensity.shape}")
    print(f"  Features:  {features.shape}")
    print("\n✓ Forward pass successful!")
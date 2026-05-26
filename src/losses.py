"""Loss functions for dual-head SEED-VII training.

L = s_i · (α · L_cls + β · L_reg) + γ · L_rank

- L_cls : CrossEntropy with label smoothing
- L_reg : MSE on continuous intensity (predicted ∈ [0,1])
- L_rank: margin ranking loss between intensity predictions of randomly paired
  samples sharing the same class label (gold order from `s`).
- 样本权重 s_i：直接来自 SEED-VII 连续标签（也可阈值化 / 关闭）。

退化方案：构造 `WeightedDualLoss(enable_rank=False, gamma=0)` 即可只用 cls+reg。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _sample_weights(
    s: torch.Tensor,
    mode: str = "continuous",
    threshold: float = 0.5,
    weak_weight: float = 0.1,
) -> torch.Tensor:
    """Compute per-sample loss weight from continuous intensity s_i ∈ [0,1]."""
    s = s.clamp(0.0, 1.0)
    if mode == "none":
        return torch.ones_like(s)
    if mode == "continuous":
        return s
    if mode == "threshold":
        w = torch.where(s >= threshold, torch.ones_like(s), torch.full_like(s, weak_weight))
        return w
    raise ValueError(f"Unknown sample_weight_mode: {mode}")

def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Per-sample weighted CE; reduces to mean over batch with weight normalization."""
    ce = F.cross_entropy(
        logits, targets,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    w = weights.detach()
    return (ce * w).sum() / (w.sum().clamp_min(1e-8))

def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    mse = (pred - target) ** 2
    w = weights.detach()
    return (mse * w).sum() / (w.sum().clamp_min(1e-8))

def margin_ranking_loss_intra_class(
    pred_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
    class_idx: torch.Tensor,
    margin: float = 0.05,
    max_pairs: int = 1024,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pair up samples sharing the same class label; supervise the order with ground-truth intensity.

    损失对来源：每个类内随机抽对 (i,j)，期望 `sign(s_i - s_j) == sign(p_i - p_j)`。
    """
    device = pred_intensity.device
    losses = []
    for c in torch.unique(class_idx):
        idx = (class_idx == c).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        # random pairs
        n_pairs = min(max_pairs, idx.numel())
        perm_a = idx[torch.randperm(idx.numel(), generator=generator, device=device)[:n_pairs]]
        perm_b = idx[torch.randperm(idx.numel(), generator=generator, device=device)[:n_pairs]]
        # filter equal & equal-intensity pairs
        sa, sb = target_intensity[perm_a], target_intensity[perm_b]
        pa, pb = pred_intensity[perm_a], pred_intensity[perm_b]
        gold_order = torch.sign(sa - sb)
        mask = gold_order != 0
        if mask.sum() == 0:
            continue
        gold_order = gold_order[mask]
        diff = (pa - pb)[mask]
        # margin ranking loss: max(0, margin - gold_order * diff)
        l = torch.clamp(margin - gold_order * diff, min=0.0)
        losses.append(l.mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()

@dataclass
class LossConfig:
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.0
    label_smoothing: float = 0.05
    rank_margin: float = 0.05
    enable_rank: bool = False
    sample_weight_mode: str = "continuous"  # ["continuous", "threshold", "none"]
    intensity_threshold: float = 0.5
    weak_sample_weight: float = 0.1

class WeightedDualLoss(nn.Module):
    """Combined classification + regression (+optional ranking) loss with per-sample weights."""

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        logits: torch.Tensor,            # (B, C)
        intensity_pred: torch.Tensor,     # (B,) in [0,1]
        target_cls: torch.Tensor,         # (B,)
        target_intensity: torch.Tensor,   # (B,) in [0,1]
    ) -> Tuple[torch.Tensor, dict]:
        w = _sample_weights(
            target_intensity,
            mode=self.cfg.sample_weight_mode,
            threshold=self.cfg.intensity_threshold,
            weak_weight=self.cfg.weak_sample_weight,
        )

        l_cls = weighted_cross_entropy(
            logits, target_cls, w, label_smoothing=self.cfg.label_smoothing
        )
        l_reg = weighted_mse(intensity_pred, target_intensity, w)

        total = self.cfg.alpha * l_cls + self.cfg.beta * l_reg
        l_rank_val = torch.zeros((), device=logits.device)
        if self.cfg.enable_rank and self.cfg.gamma > 0:
            l_rank_val = margin_ranking_loss_intra_class(
                intensity_pred, target_intensity, target_cls,
                margin=self.cfg.rank_margin,
            )
            total = total + self.cfg.gamma * l_rank_val

        return total, {
            "loss": float(total.detach().cpu().item()),
            "cls": float(l_cls.detach().item()),
            "reg": float(l_reg.detach().item()),
            "rank": float(l_rank_val.detach().item()),
            "weight_mean": float(w.mean().detach().item()),
        }

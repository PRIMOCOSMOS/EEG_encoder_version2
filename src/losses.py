"""Loss functions for dual-head SEED-VII training.

L = s_i · (α · L_cls + β · L_reg) + γ · L_rank

退化方案：构造 WeightedDualLoss(enable_rank=False, gamma=0) 即可只用 cls+reg。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_weights(s: torch.Tensor, mode: str = "continuous",
                    threshold: float = 0.5, weak_weight: float = 0.1) -> torch.Tensor:
    s = s.clamp(0.0, 1.0)
    if mode == "none":   return torch.ones_like(s)
    if mode == "continuous": return s
    if mode == "threshold":
        return torch.where(s >= threshold, torch.ones_like(s), torch.full_like(s, weak_weight))
    raise ValueError(f"Unknown sample_weight_mode: {mode}")


def weighted_cross_entropy(logits, targets, weights, label_smoothing=0.0):
    ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=label_smoothing)
    w = weights.detach()
    return (ce * w).sum() / w.sum().clamp_min(1e-8)


def weighted_mse(pred, target, weights):
    mse = (pred - target) ** 2
    w = weights.detach()
    return (mse * w).sum() / w.sum().clamp_min(1e-8)


def margin_ranking_loss_intra_class(pred_int, target_int, class_idx,
                                     margin=0.05, max_pairs=1024,
                                     generator=None) -> torch.Tensor:
    device = pred_int.device
    losses = []
    for c in torch.unique(class_idx):
        idx = (class_idx == c).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        n_pairs = min(max_pairs, idx.numel())
        perm_a = idx[torch.randperm(idx.numel(), generator=generator, device=device)[:n_pairs]]
        perm_b = idx[torch.randperm(idx.numel(), generator=generator, device=device)[:n_pairs]]
        sa, sb = target_int[perm_a], target_int[perm_b]
        pa, pb = pred_int[perm_a], pred_int[perm_b]
        gold_order = torch.sign(sa - sb)
        mask = gold_order != 0
        if mask.sum() == 0:
            continue
        gold_order = gold_order[mask]
        diff = (pa - pb)[mask]
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
    sample_weight_mode: str = "continuous"
    intensity_threshold: float = 0.5
    weak_sample_weight: float = 0.1


class WeightedDualLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, logits, intensity_pred, target_cls, target_intensity) -> Tuple[torch.Tensor, dict]:
        w = _sample_weights(target_intensity, mode=self.cfg.sample_weight_mode,
                            threshold=self.cfg.intensity_threshold, weak_weight=self.cfg.weak_sample_weight)
        l_cls = weighted_cross_entropy(logits, target_cls, w, label_smoothing=self.cfg.label_smoothing)
        l_reg = weighted_mse(intensity_pred, target_intensity, w)
        total = self.cfg.alpha * l_cls + self.cfg.beta * l_reg
        l_rank_val = torch.zeros((), device=logits.device)
        if self.cfg.enable_rank and self.cfg.gamma > 0:
            l_rank_val = margin_ranking_loss_intra_class(
                intensity_pred, target_intensity, target_cls, margin=self.cfg.rank_margin)
            total = total + self.cfg.gamma * l_rank_val
        return total, {
            "loss": float(total.detach().cpu().item()),
            "cls": float(l_cls.detach().item()),
            "reg": float(l_reg.detach().item()),
            "rank": float(l_rank_val.detach().item()),
            "weight_mean": float(w.mean().detach().item()),
        }
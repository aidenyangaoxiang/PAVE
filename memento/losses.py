"""Supervision for sparse Memento token masks."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def mask_loss(
    logits: Tensor, target: Tensor, valid: Tensor, budget_weight: float = 0.10
) -> tuple[Tensor, dict[str, float]]:
    """Class-balanced BCE plus a light target-budget regularizer."""
    flat_logits, flat_target = logits[valid], target[valid]
    positives = flat_target.sum().clamp_min(1.0)
    negatives = (1.0 - flat_target).sum().clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(max=100.0)
    bce = F.binary_cross_entropy_with_logits(flat_logits, flat_target, pos_weight=pos_weight)
    predicted_rate = torch.sigmoid(flat_logits).mean()
    target_rate = flat_target.mean()
    budget = (predicted_rate - target_rate).abs()
    return bce + budget_weight * budget, {
        "bce": float(bce.detach()),
        "budget": float(budget.detach()),
        "target_keep_rate": float(target_rate.detach()),
        "predicted_keep_rate": float(predicted_rate.detach()),
    }

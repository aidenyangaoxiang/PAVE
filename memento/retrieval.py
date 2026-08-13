"""Frozen retrieval and position-aligned history priors for Memento."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class RetrievalWeights:
    screen: float = 0.5
    task: float = 0.3
    ocr: float = 0.2


def weighted_similarity(
    query_screen: Tensor,
    query_task: Tensor,
    query_ocr: Tensor,
    bank_screen: Tensor,
    bank_task: Tensor,
    bank_ocr: Tensor,
    weights: RetrievalWeights = RetrievalWeights(),
) -> Tensor:
    """Compute the fixed CLIP retrieval score used by the Memento prototype.

    All input embeddings must be L2-normalized. The caller is responsible for
    applying split/episode leakage rules before passing a bank.
    """
    return (
        weights.screen * (bank_screen @ query_screen)
        + weights.task * (bank_task @ query_task)
        + weights.ocr * (bank_ocr @ query_ocr)
    )


def topk_retrieve(scores: Tensor, k: int = 3) -> tuple[Tensor, Tensor]:
    """Return descending scores and rows for one query without retraining."""
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 0 < k <= scores.numel():
        raise ValueError("k must be between 1 and the history-bank size")
    return torch.topk(scores, k=k, largest=True, sorted=True)


def project_history_mask(history_mask: Tensor, current_grid: tuple[int, int]) -> Tensor:
    """Project a sparse historical lattice to normalized current coordinates.

    ``history_mask`` has shape ``[H_history, W_history]``. The returned map
    has shape ``[H_current, W_current]`` and does not alter GUI-Owl positions.
    """
    if history_mask.ndim != 2:
        raise ValueError("history_mask must be [height, width]")
    height, width = current_grid
    if height <= 0 or width <= 0:
        raise ValueError("current grid dimensions must be positive")
    return F.interpolate(
        history_mask[None, None].float(), size=current_grid,
        mode="bilinear", align_corners=False,
    )[0, 0]


def build_spatial_prior(
    history_masks: list[Tensor],
    retrieval_scores: Tensor,
    current_grid: tuple[int, int],
    temperature: float = 0.05,
) -> Tensor:
    """Build [N, 3]: weighted support, disagreement, support fraction.

    Full/unverified experiences should be removed by the caller: an all-one
    historical mask would not be meaningful evidence for sparse selection.
    If no verified sparse experience is available, the returned prior is zero.
    """
    count = current_grid[0] * current_grid[1]
    if not history_masks:
        return torch.zeros(count, 3, dtype=torch.float32)
    if len(history_masks) != retrieval_scores.numel() or temperature <= 0:
        raise ValueError("history masks, scores, and positive temperature are required")
    maps = torch.stack([project_history_mask(mask, current_grid) for mask in history_masks])
    weights = torch.softmax(retrieval_scores.float() / temperature, dim=0)
    mean = (weights[:, None, None] * maps).sum(dim=0)
    variance = (weights[:, None, None] * maps.square()).sum(dim=0) - mean.square()
    support_fraction = torch.full_like(mean, len(history_masks) / retrieval_scores.numel())
    return torch.stack((mean, variance.clamp_min(0.0), support_fraction), dim=-1).reshape(count, 3)

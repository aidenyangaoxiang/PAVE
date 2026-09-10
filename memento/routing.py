"""Conservative routing and position-safe visual-token selection."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


ROUTE_FULL = "full"
ROUTE_COMPACT = "compact"


@dataclass(frozen=True)
class RouteDecision:
    route: str
    similarity: float | None
    reason: str


def similarity_route(
    similarity: float | None,
    threshold: float,
    *,
    candidate_count: int,
    protected_surface: bool = False,
) -> RouteDecision:
    """Use Compact only when scoped history exists and clears the gate."""
    if protected_surface:
        return RouteDecision(ROUTE_FULL, similarity, "protected_surface")
    if candidate_count <= 0 or similarity is None:
        return RouteDecision(ROUTE_FULL, similarity, "no_scoped_history")
    if similarity < threshold:
        return RouteDecision(ROUTE_FULL, similarity, "below_similarity_threshold")
    return RouteDecision(ROUTE_COMPACT, similarity, "history_gate_passed")


def exact_budget_indices(logits: Tensor, retention_ratio: float) -> Tensor:
    """Select exact top-k rows, returned in their original spatial order."""
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not 0 < retention_ratio <= 1:
        raise ValueError("retention_ratio must be in (0, 1]")
    keep = max(1, int(round(logits.numel() * retention_ratio)))
    # topk(sorted=False) ranks the set efficiently; the final sort is required
    # before injecting rows into an action model so row/position pairs remain
    # in the original visual sequence order.
    return torch.topk(logits, k=keep, largest=True, sorted=False).indices.sort().values


def threshold_indices(logits: Tensor, probability_threshold: float) -> Tensor:
    """Select thresholded rows and retain at least one original position."""
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if not 0 <= probability_threshold <= 1:
        raise ValueError("probability_threshold must be in [0, 1]")
    probabilities = torch.sigmoid(logits)
    kept = torch.nonzero(
        probabilities >= probability_threshold, as_tuple=False
    ).flatten()
    if kept.numel() == 0:
        kept = torch.argmax(probabilities).reshape(1)
    return kept.sort().values

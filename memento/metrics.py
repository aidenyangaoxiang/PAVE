"""Evaluation helpers for post-encoder visual-token accounting."""
from __future__ import annotations

from typing import Iterable


def token_reduction(used_tokens: Iterable[int], full_tokens: Iterable[int]) -> float:
    """Compute reduction in visual tokens passed to the action model."""
    used, full = sum(used_tokens), sum(full_tokens)
    if full <= 0:
        raise ValueError("full_tokens must sum to a positive value")
    if used < 0 or used > full:
        raise ValueError("used token count must be in [0, full]")
    return 1.0 - used / full

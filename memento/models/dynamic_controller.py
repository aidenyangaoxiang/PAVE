"""Memento-Dynamic: history-conditioned coarse-to-fine token selection."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .blocks import CrossAttentionBlock, spatial_anchors


@dataclass(frozen=True)
class DynamicControllerConfig:
    visual_dim: int
    task_dim: int
    prior_dim: int = 3
    hidden_dim: int = 256
    heads: int = 4
    layers: int = 2
    anchor_bins: int = 4
    dropout: float = 0.1


class MementoDynamicController(nn.Module):
    """Predict one keep logit for every *current* post-encoder visual token.

    The retrieved history enters as a position-aligned, three-channel spatial
    prior.  It never changes GUI-Owl coordinates and never injects old visual
    tokens into the frozen action model.
    """

    def __init__(self, config: DynamicControllerConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.visual_projection = nn.Linear(config.visual_dim, hidden)
        self.position_projection = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.prior_projection = nn.Sequential(
            nn.Linear(config.prior_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.task_projection = nn.Sequential(
            nn.Linear(config.task_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.coarse_blocks = nn.ModuleList(
            CrossAttentionBlock(hidden, config.heads, config.dropout)
            for _ in range(config.layers)
        )
        self.coarse_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.fine_projection = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.fine_blocks = nn.ModuleList(
            CrossAttentionBlock(hidden, config.heads, config.dropout)
            for _ in range(config.layers)
        )
        self.fine_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def _context(self, tokens: Tensor, xy: Tensor, valid: Tensor, task: Tensor) -> tuple[Tensor, Tensor]:
        anchors, anchor_valid = spatial_anchors(tokens, xy, valid, self.config.anchor_bins)
        context = torch.cat((task, anchors), dim=1)
        context_valid = torch.cat(
            (torch.ones(valid.shape[0], 1, dtype=torch.bool, device=valid.device), anchor_valid), dim=1
        )
        return context, context_valid

    def forward(
        self,
        current_tokens: Tensor,
        xy: Tensor,
        task_embedding: Tensor,
        spatial_prior: Tensor,
        current_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, count, _ = current_tokens.shape
        if spatial_prior.shape[:2] != (batch, count):
            raise ValueError("spatial_prior must have a row for each current token")
        if spatial_prior.shape[-1] != self.config.prior_dim:
            raise ValueError("spatial_prior channel count disagrees with configuration")
        if current_valid is None:
            current_valid = torch.ones(batch, count, dtype=torch.bool, device=current_tokens.device)
        current = (
            self.visual_projection(current_tokens)
            + self.position_projection(xy)
            + self.prior_projection(spatial_prior)
        ) * current_valid.unsqueeze(-1)
        task = self.task_projection(task_embedding).unsqueeze(1)
        for block in self.coarse_blocks:
            context, context_valid = self._context(current, xy, current_valid, task)
            current = block(current, context, current_valid, context_valid)
        coarse_logits = self.coarse_head(current).squeeze(-1).masked_fill(~current_valid, float("-inf"))
        coarse_probability = torch.sigmoid(coarse_logits).masked_fill(~current_valid, 0.0)
        fine = self.fine_projection(torch.cat((current, coarse_probability.unsqueeze(-1)), dim=-1))
        fine = fine * current_valid.unsqueeze(-1)
        for block in self.fine_blocks:
            context, context_valid = self._context(fine, xy, current_valid, task)
            fine = block(fine, context, current_valid, context_valid)
        fine_logits = self.fine_head(fine).squeeze(-1).masked_fill(~current_valid, float("-inf"))
        return coarse_logits, fine_logits

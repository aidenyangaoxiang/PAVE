"""Small attention blocks shared by Memento controllers."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossAttentionBlock(nn.Module):
    """Current tokens attend to a small task/layout context in O(NK) time."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, query: Tensor, context: Tensor, query_valid: Tensor, context_valid: Tensor
    ) -> Tensor:
        attended, _ = self.cross_attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~context_valid,
            need_weights=False,
        )
        output = query + attended
        output = output + self.feed_forward(self.feed_forward_norm(output))
        return output * query_valid.unsqueeze(-1)


def spatial_anchors(
    tokens: Tensor, xy: Tensor, valid: Tensor, bins: int
) -> tuple[Tensor, Tensor]:
    """Mean-pool tokens into a normalized ``bins x bins`` layout lattice."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    batch, count, hidden = tokens.shape
    if xy.shape[:2] != (batch, count) or valid.shape != (batch, count):
        raise ValueError("tokens, xy, and valid must share the first two dimensions")
    column = torch.clamp((xy[..., 0] * bins).long(), 0, bins - 1)
    row = torch.clamp((xy[..., 1] * bins).long(), 0, bins - 1)
    cell = row * bins + column
    cell_count = bins * bins
    sums = torch.zeros(batch, cell_count, hidden, dtype=tokens.dtype, device=tokens.device)
    counts = torch.zeros(batch, cell_count, 1, dtype=tokens.dtype, device=tokens.device)
    expanded = cell.unsqueeze(-1).expand(-1, -1, hidden)
    sums.scatter_add_(1, expanded, tokens * valid.unsqueeze(-1))
    counts.scatter_add_(1, cell.unsqueeze(-1), valid.unsqueeze(-1).to(tokens.dtype))
    return sums / counts.clamp_min(1.0), counts.squeeze(-1).bool()

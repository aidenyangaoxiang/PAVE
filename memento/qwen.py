"""Position-preserving post-merger token injection for Qwen2.5-VL.

The functions are deliberately controller-agnostic: a selector supplies rows
from Qwen's merged visual feature lattice. The rows are sorted before feature
and M-RoPE selection so the retained token-to-position mapping cannot drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass(frozen=True)
class QwenVisualCache:
    features: torch.Tensor
    placeholders: torch.Tensor


def ordered_unique_indices(
    indices: torch.Tensor, token_count: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """Validate selected original rows and return them in sequence order."""
    kept = indices.to(device=device, dtype=torch.long).flatten().sort().values
    if kept.numel() == 0:
        raise ValueError("At least one visual token must be retained")
    if kept.unique().numel() != kept.numel():
        raise ValueError("Visual-token indices must be unique")
    if kept[0] < 0 or kept[-1] >= token_count:
        raise ValueError("Visual-token index is outside the merged lattice")
    return kept


def encode_visual_features(model, inputs: dict[str, torch.Tensor]) -> QwenVisualCache:
    """Cache one image's merged Qwen features and aligned placeholders."""
    with torch.inference_mode():
        output = model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
    pooled = output.pooler_output
    if not isinstance(pooled, (tuple, list)) or len(pooled) != 1:
        raise ValueError("The current adapter expects exactly one image")
    features = pooled[0]
    placeholders = (
        inputs["input_ids"][0] == int(model.config.image_token_id)
    ).nonzero().flatten()
    if placeholders.numel() != features.shape[0]:
        raise RuntimeError("Qwen image placeholders and merged features do not align")
    return QwenVisualCache(features=features, placeholders=placeholders)


def compact_inputs(
    model,
    inputs: dict[str, torch.Tensor],
    cache: QwenVisualCache,
    kept_indices: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Remove unselected image placeholders while preserving full M-RoPE IDs."""
    kept = ordered_unique_indices(
        kept_indices, int(cache.features.shape[0]), device=cache.placeholders.device
    )
    selected_placeholders = cache.placeholders[kept]
    sequence_keep = torch.ones(
        inputs["input_ids"].shape[1], dtype=torch.bool, device=inputs["input_ids"].device
    )
    sequence_keep[cache.placeholders] = False
    sequence_keep[selected_placeholders] = True
    position_ids, _ = model.model.get_rope_index(
        inputs["input_ids"],
        inputs.get("mm_token_type_ids"),
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        second_per_grid_ts=inputs.get("second_per_grid_ts"),
        attention_mask=inputs.get("attention_mask"),
    )
    excluded = {"input_ids", "attention_mask", "mm_token_type_ids"}
    compact = {key: value for key, value in inputs.items() if key not in excluded}
    compact["input_ids"] = inputs["input_ids"][:, sequence_keep]
    compact["attention_mask"] = torch.ones_like(compact["input_ids"])
    if "mm_token_type_ids" in inputs:
        compact["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, sequence_keep]
    compact["position_ids"] = position_ids[:, :, sequence_keep]
    return compact, kept


def generate_with_kept_tokens(
    model,
    inputs: dict[str, torch.Tensor],
    cache: QwenVisualCache,
    kept_indices: torch.Tensor,
    *,
    max_new_tokens: int = 96,
) -> torch.Tensor:
    """Generate using selected Qwen post-merger tokens at original positions."""
    compact, kept = compact_inputs(model, inputs, cache, kept_indices)
    selected = cache.features[kept]
    original = model.model.get_image_features

    def selected_features(*_args, **_kwargs):
        return SimpleNamespace(pooler_output=(selected,))

    model.model.get_image_features = selected_features
    try:
        with torch.inference_mode():
            return model.generate(
                **compact, max_new_tokens=max_new_tokens, do_sample=False
            )
    finally:
        model.model.get_image_features = original

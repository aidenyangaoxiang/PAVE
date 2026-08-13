"""Optional GUI-Owl adapter for position-aligned post-encoder token injection.

This module is intentionally model-specific.  The Memento controller itself
is backbone-agnostic; this adapter demonstrates the required alignment checks
for GUI-Owl's 2x2 visual token merge.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch


def merged_grid(image_grid_thw: torch.Tensor, token_count: int) -> tuple[int, int]:
    """Return the GUI-Owl post-encoder lattice after its 2x2 merge."""
    temporal, raw_height, raw_width = (int(v) for v in image_grid_thw[0].detach().cpu().tolist())
    if temporal != 1 or raw_height % 2 or raw_width % 2:
        raise ValueError("Expected one image and an even raw GUI-Owl visual grid")
    height, width = raw_height // 2, raw_width // 2
    if height * width != token_count:
        raise ValueError("Merged grid does not match post-encoder feature count")
    return height, width


def encode_visual_features(model, inputs: dict) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Extract mutually aligned GUI-Owl visual/deepstack/placeholder tensors."""
    with torch.inference_mode():
        encoded = model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
        )
    features = torch.cat(encoded.pooler_output, dim=0)
    deepstack = list(encoded.deepstack_features)
    placeholder = (inputs["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
    if len(placeholder) != len(features):
        raise RuntimeError("GUI-Owl placeholder/features mismatch")
    if any(layer.shape[0] != len(features) for layer in deepstack):
        raise RuntimeError("Deepstack features do not align with visual features")
    return features, deepstack, placeholder


def generate_with_kept_tokens(
    model,
    inputs: dict,
    features: torch.Tensor,
    deepstack: list[torch.Tensor],
    placeholder: torch.Tensor,
    kept_indices: torch.Tensor,
    max_new_tokens: int = 128,
) -> torch.Tensor:
    """Generate after retaining the same original indices in all visual streams.

    The action model sees only ``kept_indices``.  M-RoPE positions and every
    deepstack tensor are selected with the same indices, preventing coordinate
    drift from mismatched feature-position pairs.
    """
    kept = kept_indices.to(model.device, dtype=torch.long).sort().values
    if kept.ndim != 1 or kept.numel() == 0:
        raise ValueError("At least one visual token must be retained")
    if kept.unique().numel() != kept.numel() or kept.min() < 0 or kept.max() >= len(features):
        raise ValueError("Invalid visual-token indices")
    prompt_length = inputs["input_ids"].shape[1]
    selected_placeholders = placeholder[kept]
    sequence_keep = torch.ones(prompt_length, dtype=torch.bool, device=model.device)
    sequence_keep[placeholder] = False
    sequence_keep[selected_placeholders] = True
    compact = {key: value for key, value in inputs.items() if key not in {"input_ids", "attention_mask", "mm_token_type_ids"}}
    compact["input_ids"] = inputs["input_ids"][:, sequence_keep]
    compact["attention_mask"] = torch.ones_like(compact["input_ids"])
    if "mm_token_type_ids" in inputs:
        compact["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, sequence_keep]
    position_ids, _ = model.model.get_rope_index(
        inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"), attention_mask=inputs.get("attention_mask"),
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    compact["position_ids"] = position_ids[:, :, sequence_keep]
    original = model.model.get_image_features

    def selected_features(*_args, **_kwargs):
        return SimpleNamespace(
            pooler_output=(features[kept],),
            deepstack_features=[layer[kept] for layer in deepstack],
        )

    model.model.get_image_features = selected_features
    try:
        with torch.inference_mode():
            return model.generate(**compact, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        model.model.get_image_features = original

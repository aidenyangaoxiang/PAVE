"""Demonstrate PAVE's controller and token-selection API on synthetic CPU data.

Run after installing the package: python examples/controller_demo.py
The controller uses random weights. No benchmark results are produced.
"""
from __future__ import annotations

import json

import torch

from memento import DynamicControllerConfig, MementoDynamicController, exact_budget_indices
from memento.retrieval import build_spatial_prior


def main() -> None:
    torch.manual_seed(7)
    height, width = 4, 8
    count = height * width

    # Original row-major token centers, normalized to the current screen.
    rows, columns = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    xy = torch.stack(
        ((columns + 0.5) / width, (rows + 0.5) / height), dim=-1
    ).reshape(count, 2)

    historical_mask = torch.zeros(2, 4)
    historical_mask[1, 3] = 1.0
    prior = build_spatial_prior(
        [historical_mask], torch.tensor([0.9]), (height, width)
    )
    config = DynamicControllerConfig(
        visual_dim=8, task_dim=4, hidden_dim=16, heads=4, layers=1
    )
    controller = MementoDynamicController(config).eval()

    with torch.inference_mode():
        coarse, fine = controller(
            current_tokens=torch.randn(1, count, config.visual_dim),
            xy=xy.unsqueeze(0),
            task_embedding=torch.randn(1, config.task_dim),
            spatial_prior=prior.unsqueeze(0),
        )
        selected = exact_budget_indices(fine[0], retention_ratio=0.25)

    print(json.dumps({
        "demo": "synthetic controller forward pass; random weights",
        "device": "cpu",
        "current_tokens": count,
        "retained_tokens": selected.numel(),
        "retained_fraction": selected.numel() / count,
        "selected_original_indices": selected.tolist(),
        "coarse_logits_shape": list(coarse.shape),
        "fine_logits_shape": list(fine.shape),
    }, indent=2))


if __name__ == "__main__":
    main()

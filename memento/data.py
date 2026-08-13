"""Feature-cache dataset used to train Memento controllers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SpatialMaskDataset(Dataset):
    """Load cached frozen GUI-Owl features and a matching spatial prior."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.items: list[dict[str, Any]] = json.loads(self.index_path.read_text())
        if not self.items:
            raise ValueError(f"Empty dataset index: {self.index_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = self.items[index]
        feature = torch.load(Path(item["feature_path"]), map_location="cpu", weights_only=False)
        prior_record = torch.load(Path(item["spatial_prior_path"]), map_location="cpu", weights_only=False)
        required = {"current_tokens", "xy", "task_embedding", "target_mask"}
        missing = required - feature.keys()
        if missing:
            raise KeyError(f"Feature lacks {sorted(missing)}")
        prior = prior_record["spatial_prior"]
        count = feature["current_tokens"].shape[0]
        if prior.ndim != 2 or prior.shape[0] != count:
            raise ValueError(f"Prior/token mismatch for {item['experience_id']}")
        return {
            "current_tokens": feature["current_tokens"].float(),
            "xy": feature["xy"].float(),
            "task_embedding": feature["task_embedding"].reshape(-1).float(),
            "spatial_prior": prior.float(),
            "target_mask": feature["target_mask"].float(),
        }


def collate_spatial_masks(records: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable-length token sequences within a mini-batch."""
    n_max, batch = max(x["current_tokens"].shape[0] for x in records), len(records)
    visual_dim = records[0]["current_tokens"].shape[-1]
    task_dim = records[0]["task_embedding"].numel()
    prior_dim = records[0]["spatial_prior"].shape[-1]
    result = {
        "current_tokens": torch.zeros(batch, n_max, visual_dim),
        "xy": torch.zeros(batch, n_max, 2),
        "task_embedding": torch.zeros(batch, task_dim),
        "spatial_prior": torch.zeros(batch, n_max, prior_dim),
        "target_mask": torch.zeros(batch, n_max),
        "current_valid": torch.zeros(batch, n_max, dtype=torch.bool),
    }
    for row, record in enumerate(records):
        count = record["current_tokens"].shape[0]
        result["current_tokens"][row, :count] = record["current_tokens"]
        result["xy"][row, :count] = record["xy"]
        result["task_embedding"][row] = record["task_embedding"]
        result["spatial_prior"][row, :count] = record["spatial_prior"]
        result["target_mask"][row, :count] = record["target_mask"]
        result["current_valid"][row, :count] = True
    return result

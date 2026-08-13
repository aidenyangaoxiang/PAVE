#!/usr/bin/env python3
"""Train history-aware Memento-Dynamic or a matched current-only ablation."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from memento.data import SpatialMaskDataset, collate_spatial_masks
from memento.losses import mask_loss
from memento.models import DynamicControllerConfig, MementoDynamicController


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def coarse_targets(target: torch.Tensor, valid: torch.Tensor, xy: torch.Tensor, bins: int) -> torch.Tensor:
    """Mark all tokens in any coarse cell containing a final-mask token."""
    batch, _ = target.shape
    row = torch.clamp((xy[..., 1] * bins).long(), 0, bins - 1)
    col = torch.clamp((xy[..., 0] * bins).long(), 0, bins - 1)
    cell = row * bins + col
    occupied = torch.zeros(batch, bins * bins, dtype=target.dtype, device=target.device)
    occupied.scatter_reduce_(1, cell, target * valid.to(target.dtype), reduce="amax", include_self=False)
    return (occupied.gather(1, cell) > 0).to(target.dtype) * valid.to(target.dtype)


def move(batch: dict[str, torch.Tensor], device: torch.device, current_only: bool) -> dict[str, torch.Tensor]:
    result = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    if current_only:
        result["spatial_prior"] = torch.zeros_like(result["spatial_prior"])
    return result


@torch.no_grad()
def evaluate(model, loader, device, current_only: bool, bins: int) -> dict[str, float]:
    model.eval()
    total = {"loss": 0.0, "intersection": 0.0, "predicted": 0.0, "target": 0.0, "batches": 0}
    for batch in loader:
        batch = move(batch, device, current_only)
        coarse, fine = model(**{key: batch[key] for key in ("current_tokens", "xy", "task_embedding", "spatial_prior", "current_valid")})
        fine_loss, _ = mask_loss(fine, batch["target_mask"], batch["current_valid"])
        coarse_target = coarse_targets(batch["target_mask"], batch["current_valid"], batch["xy"], bins)
        coarse_loss, _ = mask_loss(coarse, coarse_target, batch["current_valid"], budget_weight=0.03)
        predicted = (torch.sigmoid(fine) >= 0.5) & batch["current_valid"]
        target = batch["target_mask"].bool() & batch["current_valid"]
        total["loss"] += float(fine_loss + 0.35 * coarse_loss)
        total["intersection"] += float((predicted & target).sum())
        total["predicted"] += float(predicted.sum())
        total["target"] += float(target.sum())
        total["batches"] += 1
    precision = total["intersection"] / max(total["predicted"], 1.0)
    recall = total["intersection"] / max(total["target"], 1.0)
    return {
        "loss": total["loss"] / max(total["batches"], 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
    }


def checkpoint(path: Path, model, metrics: dict, args) -> None:
    torch.save({"config": model.config.__dict__, "state_dict": model.state_dict(), "metrics": metrics, "train_args": vars(args)}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--anchor-bins", type=int, default=4)
    parser.add_argument("--current-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.out}")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation = SpatialMaskDataset(args.train_index), SpatialMaskDataset(args.val_index)
    sample = train[0]
    config = DynamicControllerConfig(
        visual_dim=sample["current_tokens"].shape[-1],
        task_dim=sample["task_embedding"].numel(),
        prior_dim=sample["spatial_prior"].shape[-1],
        hidden_dim=args.hidden_dim, layers=args.layers, anchor_bins=args.anchor_bins,
    )
    model = MementoDynamicController(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_spatial_masks)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spatial_masks)
    args.out.mkdir(parents=True)
    best_loss, best_f1, history = float("inf"), -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = move(batch, device, args.current_only)
            coarse, fine = model(**{key: batch[key] for key in ("current_tokens", "xy", "task_embedding", "spatial_prior", "current_valid")})
            fine_loss, _ = mask_loss(fine, batch["target_mask"], batch["current_valid"])
            coarse_target = coarse_targets(batch["target_mask"], batch["current_valid"], batch["xy"], args.anchor_bins)
            coarse_loss, _ = mask_loss(coarse, coarse_target, batch["current_valid"], budget_weight=0.03)
            loss = fine_loss + 0.35 * coarse_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach())
        metrics = evaluate(model, validation_loader, device, args.current_only, args.anchor_bins)
        metrics.update({"epoch": epoch, "train_loss": train_loss / max(len(train_loader), 1)})
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            checkpoint(args.out / "best_controller_loss.pt", model, metrics, args)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            checkpoint(args.out / "best_controller_f1.pt", model, metrics, args)
    (args.out / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps({"best_val_loss": best_loss, "best_val_f1": best_f1, "current_only": args.current_only, "config": config.__dict__}, indent=2) + "\n")


if __name__ == "__main__":
    main()

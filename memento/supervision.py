"""Context-preserving candidates for PAVE sufficient-mask label search.

These masks exist **only** to create offline supervision.  The current GT
click box is used as a seed so every candidate contains the task target, but
the controller will never receive a GT box at runtime.  Unlike the previous
candidate family, this one explicitly tests multiple local-context scales and
retrieved historical spatial support.
"""
from __future__ import annotations

import math
import re
from typing import Iterable

import torch
import torch.nn.functional as F


_STOP = {
    "the", "and", "with", "from", "that", "this", "then", "into", "your",
    "for", "using", "use", "open", "click", "tap", "select", "please", "app",
    "after", "before", "next", "first", "finally", "to", "of", "on", "in",
    "a", "an", "is", "it", "as", "at", "by", "or", "be", "my", "me",
}


def _dedupe(raw: Iterable[tuple[str, set[int]]]) -> list[tuple[str, list[int]]]:
    result: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for name, indices in raw:
        key = tuple(sorted(indices))
        if key and key not in seen:
            seen.add(key)
            result.append((name, list(key)))
    return result


def bbox_indices(bbox: list[float], height: int, width: int, halo: int = 0) -> set[int]:
    """Map a normalized [0,1000] box to a row-major GUI-Owl token grid."""
    x1, y1, x2, y2 = (float(value) for value in bbox)
    x1, x2 = sorted((max(0.0, x1), min(1000.0, x2)))
    y1, y2 = sorted((max(0.0, y1), min(1000.0, y2)))
    row0, row1 = int(y1 * height / 1000), int((y2 * height - 1e-8) / 1000)
    col0, col1 = int(x1 * width / 1000), int((x2 * width - 1e-8) / 1000)
    row0, row1 = max(0, min(height - 1, row0)), max(0, min(height - 1, row1))
    col0, col1 = max(0, min(width - 1, col0)), max(0, min(width - 1, col1))
    return {
        row * width + col
        for row in range(max(0, row0 - halo), min(height, row1 + halo + 1))
        for col in range(max(0, col0 - halo), min(width, col1 + halo + 1))
    }


def _terms(text: str) -> set[str]:
    return {
        word.lower() for word in re.findall(r"[A-Za-z0-9]+", text)
        if len(word) >= 3 and word.lower() not in _STOP
    }


def task_ocr_indices(
    record: dict, height: int, width: int, image_width: int, image_height: int
) -> set[int]:
    """Current-screen OCR regions whose text matches task or previous action."""
    query = _terms(f"{record.get('task', '')} {record.get('previous_action', '')}")
    scored: list[tuple[int, set[int]]] = []
    for item in record.get("ocr", {}).get("items", []):
        text = str(item.get("text", ""))
        words = _terms(text)
        score = len(words & query)
        lowered = text.lower()
        if any(len(term) >= 5 and term in lowered for term in query):
            score += 2
        if score <= 0:
            continue
        points = item.get("box", [])
        if len(points) < 2:
            continue
        xs = [float(point[0]) * 1000.0 / image_width for point in points]
        ys = [float(point[1]) * 1000.0 / image_height for point in points]
        scored.append((score, bbox_indices([min(xs), min(ys), max(xs), max(ys)], height, width, halo=1)))
    result: set[int] = set()
    for _, indices in sorted(scored, key=lambda item: (-item[0], len(item[1])))[:4]:
        result.update(indices)
    return result


def layout_bars(height: int, width: int) -> set[int]:
    """Generic system/navigation bands; no app-specific manual widget labels."""
    band = max(1, math.ceil(height * 0.08))
    return {
        row * width + col
        for row in list(range(band)) + list(range(max(band, height - band), height))
        for col in range(width)
    }


def _project_mask(mask: torch.Tensor, current_grid: tuple[int, int]) -> torch.Tensor:
    """Project a historical normalized token lattice to current lattice."""
    return F.interpolate(mask, size=current_grid, mode="bilinear", align_corners=False)[0, 0]


def history_support_indices(
    sparse_sources: list[tuple[list[int], tuple[int, int], float]],
    current_grid: tuple[int, int],
    ratio: float,
) -> set[int]:
    """Return top-ratio current cells supported by retrieved sparse history masks.

    Source masks are projected only through normalized grid coordinates.  They
    are a *soft context proposal*, never old GUI coordinates forwarded to the
    action model.
    """
    height, width = current_grid
    count = height * width
    if not sparse_sources:
        return set()
    maps, scores = [], []
    for indices, (old_h, old_w), score in sparse_sources:
        if not indices or old_h <= 0 or old_w <= 0:
            continue
        old_count = old_h * old_w
        if min(indices) < 0 or max(indices) >= old_count:
            continue
        source = torch.zeros(old_count, dtype=torch.float32)
        source[torch.tensor(sorted(set(indices)), dtype=torch.long)] = 1.0
        maps.append(_project_mask(source.reshape(1, 1, old_h, old_w), current_grid).reshape(-1))
        scores.append(float(score))
    if not maps:
        return set()
    values = torch.stack(maps)
    weights = torch.softmax(torch.tensor(scores, dtype=torch.float32) / 0.05, dim=0)
    support = (weights[:, None] * values).sum(dim=0)
    keep = min(count, max(1, math.ceil(count * ratio)))
    return set(torch.topk(support, k=keep, largest=True).indices.tolist())


def candidates(
    record: dict,
    height: int,
    width: int,
    image_width: int,
    image_height: int,
    sparse_history_sources: list[tuple[list[int], tuple[int, int], float]],
) -> list[tuple[str, list[int]]]:
    """Return ordered action-preservation proposals plus an explicit Full mask."""
    target = bbox_indices(record["gt_bbox_1000"], height, width, halo=0)
    halo1 = bbox_indices(record["gt_bbox_1000"], height, width, halo=1)
    halo2 = bbox_indices(record["gt_bbox_1000"], height, width, halo=2)
    halo4 = bbox_indices(record["gt_bbox_1000"], height, width, halo=4)
    ocr = task_ocr_indices(record, height, width, image_width, image_height)
    bars = layout_bars(height, width)
    history10 = history_support_indices(sparse_history_sources, (height, width), ratio=0.10)
    history20 = history_support_indices(sparse_history_sources, (height, width), ratio=0.20)
    raw = [
        ("target", target),
        ("target_halo1", halo1),
        ("target_halo2", halo2),
        ("target_halo4", halo4),
        ("target_task_ocr", target | ocr),
        ("target_halo1_task_ocr", halo1 | ocr),
        ("target_layout_bars", target | bars),
        ("target_halo1_layout_bars", halo1 | bars),
        ("target_history10", target | history10),
        ("target_history20", target | history20),
        ("target_task_ocr_history20", target | ocr | history20),
        ("full", set(range(height * width))),
    ]
    result = _dedupe(raw)
    if not any(name == "full" for name, _ in result):
        result.append(("full", list(range(height * width))))
    return result

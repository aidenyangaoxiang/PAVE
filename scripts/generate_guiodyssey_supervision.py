#!/usr/bin/env python3
"""Generate offline PAVE sufficient-mask supervision on GUIOdyssey.

For each verified-Full GUI-Owl state, this script tests a richer candidate
family through the *frozen* post-encoder Compact GUI-Owl intervention.  It is
strictly offline: GT boxes only seed/score candidate labels and never become a
runtime controller input.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from memento.guiowl import (
    MODEL_NAME, decode, encode_visual_features, generate_with_kept_tokens,
    merged_grid, prepare_state_inputs,
)
from memento.supervision import candidates


COORD = re.compile(r'"coordinate"\s*:\s*\[\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\]')
ACTION = re.compile(r'"action"\s*:\s*"([^\"]+)"')


def parse_click(answer: str) -> tuple[float, float] | None:
    action, coordinate = ACTION.search(answer), COORD.search(answer)
    if not action or action.group(1).lower() != "click" or not coordinate:
        return None
    return float(coordinate.group(1)), float(coordinate.group(2))


def in_bbox(point: tuple[float, float] | None, bbox: list[float]) -> bool:
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return x1 <= x <= x2 and y1 <= y <= y2


def read_mstar(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    return {str(row["experience_id"]): row for row in payload["records"]}


def choose_queries(
    history: list[dict], max_states: int, seed: int, query_split: str, all_states: bool,
) -> list[int]:
    """Select queries for a pilot (one per episode) or the formal all-state pass."""
    eligible = [
        index for index, row in enumerate(history)
        if row.get("split") == query_split
    ]
    if all_states:
        ordered = sorted(eligible, key=lambda idx: str(history[idx]["experience_id"]))
        return ordered if max_states <= 0 else ordered[:max_states]

    by_episode: dict[str, list[int]] = defaultdict(list)
    for index in eligible:
        by_episode[str(history[index]["episode"])].append(index)
    rng = random.Random(seed)
    episodes = list(by_episode)
    rng.shuffle(episodes)
    selected: list[int] = []
    for episode in episodes:
        options = sorted(by_episode[episode], key=lambda idx: (int(history[idx].get("step", 0)), str(history[idx]["experience_id"])))
        selected.append(options[rng.randrange(len(options))])
        if max_states > 0 and len(selected) == max_states:
            break
    if max_states > 0 and len(selected) < max_states:
        raise RuntimeError(f"Only found {len(selected)} train episodes, need {max_states}")
    return sorted(selected, key=lambda idx: str(history[idx]["experience_id"]))


def resolve_image(record: dict, screenshots_root: Path) -> Path:
    source = Path(str(record.get("image_source", "")))
    if source.is_file():
        return source
    candidate = screenshots_root / f"{record['episode']}_{record['step']}.png"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"No image for {record['experience_id']}: tried {source} and {candidate}")


def sparse_sources(
    query_index: int,
    history: list[dict],
    mstar: dict[str, dict],
    image: np.ndarray,
    task: np.ndarray,
    ocr: np.ndarray,
    top_k: int,
) -> tuple[list[tuple[list[int], tuple[int, int], float]], list[dict]]:
    """One causal-style retrieval: train support excluding the same episode."""
    query = history[query_index]
    allowed = [
        index for index, row in enumerate(history)
        if row.get("split") == "train"
        and row.get("episode") != query.get("episode")
        and str(row["experience_id"]) in mstar
    ]
    allowed_np = np.asarray(allowed, dtype=np.int64)
    scores = (
        0.5 * (image[allowed_np] @ image[query_index])
        + 0.3 * (task[allowed_np] @ task[query_index])
        + 0.2 * (ocr[allowed_np] @ ocr[query_index])
    )
    order = np.argsort(-scores)[:top_k]
    sources: list[tuple[list[int], tuple[int, int], float]] = []
    metadata: list[dict] = []
    for rank in order.tolist():
        row_index, score = int(allowed_np[rank]), float(scores[rank])
        historical = history[row_index]
        label = mstar[str(historical["experience_id"])]
        indices, grid = label.get("m_star_indices"), label.get("grid")
        metadata.append({
            "experience_id": str(historical["experience_id"]),
            "score": round(score, 6),
            "m_star_recipe": label.get("m_star_recipe"),
        })
        if label.get("m_star_recipe") not in {None, "full"} and indices and grid:
            sources.append((list(indices), (int(grid[0]), int(grid[1])), score))
    return sources, metadata


def checkpoint(path: Path, records: list[dict], model_name: str) -> None:
    path.with_suffix(path.suffix + ".partial").write_text(
        json.dumps({"model": model_name, "records": records}, ensure_ascii=False)
    )


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-bank", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--mstar", type=Path, required=True,
                        help="Existing labels used only as sparse *history* support.")
    parser.add_argument("--screenshots-root", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--system-prompt-file", type=Path, required=True,
                        help="Plain-text action-model system prompt.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=1000,
                        help="0 means all states in --query-split")
    parser.add_argument("--query-split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--all-states", action="store_true",
                        help="Use every state in --query-split, rather than one per episode.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {args.out}")

    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise ImportError(
            "Install the project dependencies before generating supervision: "
            "pip install -e ."
        ) from error

    history = json.loads(args.history_bank.read_text())["records"]
    arrays = np.load(args.embeddings)
    ids = [str(item) for item in arrays["experience_ids"].tolist()]
    if ids != [str(row["experience_id"]) for row in history]:
        raise RuntimeError("History and CLIP embedding row order disagree")
    mstar = read_mstar(args.mstar)
    image_embeddings = arrays["image_embeddings"].astype(np.float32)
    task_embeddings = arrays["task_embeddings"].astype(np.float32)
    ocr_embeddings = arrays["ocr_embeddings"].astype(np.float32)
    selected = choose_queries(
        history, args.max_states, args.seed, args.query_split, args.all_states,
    )
    states = selected[args.shard_id::args.num_shards]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    outputs = json.loads(partial.read_text())["records"] if partial.exists() else []
    completed = {str(row["experience_id"]) for row in outputs}

    system = args.system_prompt_file.read_text(encoding="utf-8").strip()
    if not system:
        raise ValueError("--system-prompt-file is empty")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=args.local_files_only,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    for number, query_index in enumerate(states, start=1):
        state = dict(history[query_index])
        if str(state["experience_id"]) in completed:
            continue
        state["image_source"] = str(resolve_image(state, args.screenshots_root))
        with Image.open(state["image_source"]) as opened:
            image_width, image_height = opened.size
        retrieved_sparse, retrieved_metadata = sparse_sources(
            query_index, history, mstar, image_embeddings, task_embeddings, ocr_embeddings,
            args.top_k,
        )
        inputs, prompt_length = prepare_state_inputs(model, processor, system, state)
        features, deepstack, placeholder = encode_visual_features(model, inputs)
        grid_h, grid_w = merged_grid(inputs["image_grid_thw"], len(features))
        candidate_indices = dict(candidates(
            state, grid_h, grid_w, image_width, image_height, retrieved_sparse
        ))
        trials = []
        for recipe, indices in candidate_indices.items():
            synchronize(); started = time.perf_counter()
            kept = torch.tensor(
                indices, dtype=torch.long, device=model.device
            )
            generated = generate_with_kept_tokens(
                model, inputs, features, deepstack, placeholder,
                kept,
            )
            compact_prompt_length = prompt_length - len(features) + len(indices)
            answer = decode(processor, generated, compact_prompt_length)
            synchronize()
            point = parse_click(answer)
            trials.append({
                "recipe": recipe,
                "retained_tokens": len(indices),
                "prediction_1000": list(point) if point else None,
                "hit_gt_bbox": in_bbox(point, state["gt_bbox_1000"]),
                "seconds": round(time.perf_counter() - started, 3),
            })
        sparse_success = [trial for trial in trials if trial["recipe"] != "full" and trial["hit_gt_bbox"]]
        chosen = min(sparse_success, key=lambda trial: trial["retained_tokens"]) if sparse_success else None
        full_trial = next(trial for trial in trials if trial["recipe"] == "full")
        row = {
            "experience_id": str(state["experience_id"]),
            "episode": str(state["episode"]),
            "step": int(state["step"]),
            "split": str(state["split"]),
            "query_row": int(query_index),
            "image_source": state["image_source"],
            "grid": [grid_h, grid_w],
            "full_visual_tokens": len(features),
            "retrieved_topk": retrieved_metadata,
            "valid_sparse_history_sources": len(retrieved_sparse),
            "candidate_trials": trials,
            "successful_sparse_recipes": [trial["recipe"] for trial in sparse_success],
            "m_star_recipe": chosen["recipe"] if chosen else "full",
            "m_star_tokens": chosen["retained_tokens"] if chosen else len(features),
            "m_star_indices": candidate_indices[chosen["recipe"]] if chosen else None,
            "full_candidate_hit": full_trial["hit_gt_bbox"],
            "unresolved_by_candidate_search": chosen is None,
        }
        outputs.append(row)
        print(json.dumps({
            "completed": number, "total": len(states),
            "experience_id": row["experience_id"], "grid": row["grid"],
            "m_star_recipe": row["m_star_recipe"], "m_star_tokens": row["m_star_tokens"],
            "valid_sparse_history_sources": row["valid_sparse_history_sources"],
            "full_candidate_hit": row["full_candidate_hit"],
        }, ensure_ascii=False), flush=True)
        if len(outputs) % args.checkpoint_every == 0:
            checkpoint(args.out, outputs, args.model)

    outputs.sort(key=lambda row: row["experience_id"])
    args.out.write_text(json.dumps({
        "schema_version": "pave_guiodyssey_supervision_v1",
        "model": args.model,
        "selection": {
            "split": args.query_split,
            "one_state_per_episode": not args.all_states,
            "max_states": args.max_states, "seed": args.seed,
        },
        "records": outputs,
    }, ensure_ascii=False))
    # Preserve the incremental checkpoint for provenance.  The completed JSON
    # is authoritative, while ``.partial`` can help audit interrupted runs.
    print(json.dumps({"DONE": str(args.out), "records": len(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit a frozen PAVE online bank without loading a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    paths = {
        "bank": args.bank_dir / "bank.json",
        "labels": args.bank_dir / "labels.json",
        "embeddings": args.bank_dir / "embeddings.npz",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete bank; missing: {missing}")

    records = json.loads(paths["bank"].read_text(encoding="utf-8"))["records"]
    labels = json.loads(paths["labels"].read_text(encoding="utf-8"))["records"]
    label_by_id = {str(row["experience_id"]): row for row in labels}
    vectors = np.load(paths["embeddings"], allow_pickle=False)
    ids = [str(value) for value in vectors["experience_ids"].tolist()]
    record_ids = [str(row["experience_id"]) for row in records]
    if ids != record_ids:
        raise RuntimeError("Embedding rows do not match bank record order")
    if len(label_by_id) != len(labels) or set(record_ids) != set(label_by_id):
        raise RuntimeError("Labels are missing, duplicated, or refer to unknown records")
    expected_rows = len(records)
    vector_shapes = {
        name: list(vectors[name].shape)
        for name in ("image_embeddings", "task_embeddings", "ocr_embeddings")
    }
    if any(shape[0] != expected_rows for shape in vector_shapes.values()):
        raise RuntimeError("Embedding arrays disagree with bank length")

    kinds = Counter(row.get("record_kind", "unknown") for row in records)
    package_counts: dict[str, Counter] = defaultdict(Counter)
    for row in records:
        package_counts[row.get("foreground_package") or "<unknown>"][
            row.get("record_kind", "unknown")
        ] += 1
    report = {
        "status": "pass",
        "bank_dir": str(args.bank_dir.resolve()),
        "records": expected_rows,
        "episodes": len({row.get("episode_index") for row in records}),
        "record_kinds": dict(sorted(kinds.items())),
        "per_package": {
            package: dict(sorted(counts.items()))
            for package, counts in sorted(package_counts.items())
        },
        "embedding_shapes": vector_shapes,
        "sha256": {name: sha256(path) for name, path in paths.items()},
    }
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()

"""Episode-atomic persistent experience for online GUI rollouts.

Records are staged while an episode is running and become retrievable only
after an external evaluator reports task success.  A Full decision contributes
a page template.  A Compact decision contributes sparse spatial evidence only
when its retained mask passes the configured sparsity admission rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


TEMPLATE_ONLY = "template_only"
OUTCOME_VALIDATED_SPARSE = "outcome_validated_sparse"


@dataclass(frozen=True)
class BankPaths:
    records: Path
    labels: Path
    embeddings: Path
    events: Path

    @classmethod
    def under(cls, directory: str | Path) -> "BankPaths":
        root = Path(directory)
        return cls(
            records=root / "bank.json",
            labels=root / "labels.json",
            embeddings=root / "embeddings.npz",
            events=root / "bank_events.jsonl",
        )


class EpisodeAtomicBank:
    """Persistent online bank with success-conditioned, episode-level writes.

    The class is independent of AndroidWorld and any vision backbone. Callers
    supply normalized screen/task/OCR embeddings and optional kept-token
    indices. Package scoping is retained in the stored records so retrieval can
    be limited to the foreground application.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        embedding_dim: int = 512,
        sparse_admission_ratio: float = 0.8,
        read_only: bool = False,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0 < sparse_admission_ratio <= 1:
            raise ValueError("sparse_admission_ratio must be in (0, 1]")
        self.paths = BankPaths.under(directory)
        self.embedding_dim = embedding_dim
        self.sparse_admission_ratio = sparse_admission_ratio
        self.read_only = read_only
        self.records: list[dict[str, Any]] = []
        self.labels: dict[str, dict[str, Any]] = {}
        self.screen_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
        self.task_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
        self.ocr_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
        self._pending: list[dict[str, Any]] = []
        self._episode_index = -1
        self._episode_meta: dict[str, Any] = {}
        self.committed_episodes = 0
        self.discarded_episodes = 0
        self.demoted_near_keep_all = 0
        self._load()

    def _load(self) -> None:
        present = [
            self.paths.records.exists(),
            self.paths.labels.exists(),
            self.paths.embeddings.exists(),
        ]
        if not any(present):
            return
        if not all(present):
            raise RuntimeError("Online bank is incomplete: JSON and NPZ files must coexist")
        self.records = json.loads(self.paths.records.read_text(encoding="utf-8"))["records"]
        label_rows = json.loads(self.paths.labels.read_text(encoding="utf-8"))["records"]
        self.labels = {str(row["experience_id"]): row for row in label_rows}
        vectors = np.load(self.paths.embeddings, allow_pickle=False)
        ids = [str(value) for value in vectors["experience_ids"].tolist()]
        record_ids = [str(row["experience_id"]) for row in self.records]
        if ids != record_ids:
            raise RuntimeError("Bank records and embedding rows are not aligned")
        if set(ids) != set(self.labels):
            raise RuntimeError("Every bank record must have exactly one label")
        self.screen_embeddings = self._validated_vectors(vectors["image_embeddings"])
        self.task_embeddings = self._validated_vectors(vectors["task_embeddings"])
        self.ocr_embeddings = self._validated_vectors(vectors["ocr_embeddings"])
        self._episode_index = max(
            (int(row.get("episode_index", -1)) for row in self.records), default=-1
        )
        self.committed_episodes = len(
            {row.get("episode_index") for row in self.records if "episode_index" in row}
        )

    def _validated_vectors(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (len(self.records), self.embedding_dim):
            raise RuntimeError(
                f"Expected embedding shape {(len(self.records), self.embedding_dim)}, "
                f"got {array.shape}"
            )
        return array

    def begin_episode(self, **metadata: Any) -> int:
        """Begin a new episode and discard any uncommitted prior staging."""
        if self._pending:
            self.discarded_episodes += 1
        self._pending = []
        self._episode_index += 1
        self._episode_meta = dict(metadata)
        return self._episode_index

    def stage_step(
        self,
        *,
        step: int,
        route: str,
        grid: tuple[int, int] | None,
        embeddings: tuple[np.ndarray, np.ndarray, np.ndarray],
        foreground_package: str,
        task_instruction: str,
        kept_indices: list[int] | None = None,
        mask_threshold: float | None = None,
        **metadata: Any,
    ) -> None:
        """Stage one step without exposing it to retrieval in this episode."""
        if self.read_only:
            raise RuntimeError("Cannot stage experience in a read-only bank")
        if not foreground_package:
            raise ValueError("foreground_package is required for scoped retrieval")
        if not task_instruction.strip():
            raise ValueError("task_instruction cannot be empty")
        vectors = tuple(self._validate_vector(value) for value in embeddings)
        grid_list = [int(grid[0]), int(grid[1])] if grid else None
        retained_ratio = None
        if route == "compact" and kept_indices and grid_list:
            total = grid_list[0] * grid_list[1]
            if total <= 0:
                raise ValueError("grid dimensions must be positive")
            retained_ratio = len(kept_indices) / total
        sparse = bool(
            route == "compact"
            and kept_indices
            and retained_ratio is not None
            and retained_ratio <= self.sparse_admission_ratio
        )
        if route == "compact" and kept_indices and not sparse:
            self.demoted_near_keep_all += 1
        identifier = f"online:{self._episode_index:04d}:{int(step):03d}"
        record = {
            **self._episode_meta,
            **metadata,
            "experience_id": identifier,
            "source": "online",
            "split": "online",
            "record_kind": OUTCOME_VALIDATED_SPARSE if sparse else TEMPLATE_ONLY,
            "episode_index": self._episode_index,
            "step": int(step),
            "route": route,
            "grid": grid_list,
            "foreground_package": foreground_package,
            "task_instruction": task_instruction,
            "retained_ratio": retained_ratio,
            "demoted_near_keep_all": bool(route == "compact" and kept_indices and not sparse),
        }
        label = {
            "experience_id": identifier,
            "split": "online",
            "grid": grid_list,
            "m_star_recipe": "dynamic" if sparse else "full",
            "m_star_indices": [int(index) for index in kept_indices] if sparse else [],
            "m_star_threshold": mask_threshold,
            "outcome_validated": sparse,
        }
        self._pending.append({"record": record, "label": label, "embeddings": vectors})

    def _validate_vector(self, value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (self.embedding_dim,):
            raise ValueError(
                f"Expected one [{self.embedding_dim}] embedding, got {vector.shape}"
            )
        return vector

    def finish_episode(self, success: bool) -> dict[str, Any]:
        """Commit all staged steps iff an external evaluator reports success."""
        pending, self._pending = self._pending, []
        summary: dict[str, Any] = {
            "episode_index": self._episode_index,
            "staged_steps": len(pending),
            "committed": False,
            "committed_steps": 0,
        }
        if not pending:
            return summary
        if not success:
            self.discarded_episodes += 1
            self._append_event({"event": "discard", **summary, **self._episode_meta})
            return summary
        if self.read_only:
            raise RuntimeError("Cannot commit experience to a read-only bank")
        for item in pending:
            record = item["record"]
            self.records.append(record)
            self.labels[str(record["experience_id"])] = item["label"]
            screen, task, ocr = item["embeddings"]
            self.screen_embeddings = np.concatenate((self.screen_embeddings, screen[None]))
            self.task_embeddings = np.concatenate((self.task_embeddings, task[None]))
            self.ocr_embeddings = np.concatenate((self.ocr_embeddings, ocr[None]))
        self.committed_episodes += 1
        self._persist()
        summary.update(committed=True, committed_steps=len(pending), **self.stats())
        self._append_event({"event": "commit", **summary, **self._episode_meta})
        return summary

    def abandon_episode(self) -> None:
        if self._pending:
            self.discarded_episodes += 1
        self._pending = []

    def candidate_indices(self, foreground_package: str) -> np.ndarray:
        """Return rows from the same foreground package only."""
        return np.asarray(
            [
                index
                for index, row in enumerate(self.records)
                if row.get("foreground_package") == foreground_package
            ],
            dtype=np.int64,
        )

    def _persist(self) -> None:
        self.paths.records.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_text(
            self.paths.records,
            json.dumps({"records": self.records}, ensure_ascii=False),
        )
        label_rows = [self.labels[str(row["experience_id"])] for row in self.records]
        self._atomic_text(
            self.paths.labels,
            json.dumps({"records": label_rows}, ensure_ascii=False),
        )
        temporary = self.paths.embeddings.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                experience_ids=np.asarray(
                    [str(row["experience_id"]) for row in self.records], dtype="<U64"
                ),
                image_embeddings=self.screen_embeddings,
                task_embeddings=self.task_embeddings,
                ocr_embeddings=self.ocr_embeddings,
            )
        os.replace(temporary, self.paths.embeddings)

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)

    def _append_event(self, event: dict[str, Any]) -> None:
        if self.read_only:
            return
        self.paths.events.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def stats(self) -> dict[str, int]:
        return {
            "bank_total": len(self.records),
            "bank_template_only": sum(
                row.get("record_kind") == TEMPLATE_ONLY for row in self.records
            ),
            "bank_outcome_validated_sparse": sum(
                row.get("record_kind") == OUTCOME_VALIDATED_SPARSE
                for row in self.records
            ),
            "bank_committed_episodes": self.committed_episodes,
            "bank_discarded_episodes": self.discarded_episodes,
            "bank_demoted_near_keep_all": self.demoted_near_keep_all,
        }

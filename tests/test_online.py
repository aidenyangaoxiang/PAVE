import numpy as np
import pytest

from memento.online import EpisodeAtomicBank


def vector(value: float) -> np.ndarray:
    return np.asarray([value, 0.0, 0.0, 0.0], dtype=np.float32)


def stage(bank: EpisodeAtomicBank, *, route="full", kept=None):
    bank.stage_step(
        step=0,
        route=route,
        grid=(2, 2),
        embeddings=(vector(1), vector(2), vector(3)),
        foreground_package="com.example.app",
        task_instruction="Open the example app",
        kept_indices=kept,
        mask_threshold=0.2,
    )


def test_failed_episode_never_enters_bank(tmp_path):
    bank = EpisodeAtomicBank(tmp_path, embedding_dim=4)
    bank.begin_episode(task_id="failed")
    stage(bank)
    summary = bank.finish_episode(False)
    assert not summary["committed"]
    assert bank.stats()["bank_total"] == 0


def test_staging_is_invisible_until_success_and_survives_reload(tmp_path):
    bank = EpisodeAtomicBank(tmp_path, embedding_dim=4)
    bank.begin_episode(task_id="success")
    stage(bank, route="compact", kept=[0])
    assert bank.candidate_indices("com.example.app").size == 0
    summary = bank.finish_episode(True)
    assert summary["committed_steps"] == 1
    assert bank.stats()["bank_outcome_validated_sparse"] == 1

    loaded = EpisodeAtomicBank(tmp_path, embedding_dim=4, read_only=True)
    assert loaded.candidate_indices("com.example.app").tolist() == [0]
    assert loaded.labels["online:0000:000"]["m_star_indices"] == [0]
    with pytest.raises(RuntimeError, match="read-only"):
        stage(loaded)


def test_near_keep_all_is_template_only(tmp_path):
    bank = EpisodeAtomicBank(tmp_path, embedding_dim=4, sparse_admission_ratio=0.8)
    bank.begin_episode()
    stage(bank, route="compact", kept=[0, 1, 2, 3])
    bank.finish_episode(True)
    assert bank.stats()["bank_template_only"] == 1
    assert bank.stats()["bank_demoted_near_keep_all"] == 1
    assert bank.labels["online:0000:000"]["m_star_indices"] == []

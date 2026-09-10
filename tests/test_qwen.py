from types import SimpleNamespace

import pytest
import torch

from memento.qwen import QwenVisualCache, compact_inputs, ordered_unique_indices


class DummyQwenCore:
    def get_rope_index(self, input_ids, _token_types, **_kwargs):
        length = input_ids.shape[1]
        positions = torch.arange(length).view(1, 1, length).expand(3, 1, length)
        return positions, None


def test_selected_indices_are_restored_to_spatial_sequence_order():
    kept = ordered_unique_indices(torch.tensor([5, 1, 3]), token_count=6)
    assert kept.tolist() == [1, 3, 5]


def test_selected_indices_reject_duplicates_and_out_of_range():
    with pytest.raises(ValueError, match="unique"):
        ordered_unique_indices(torch.tensor([1, 1]), token_count=4)
    with pytest.raises(ValueError, match="outside"):
        ordered_unique_indices(torch.tensor([4]), token_count=4)


def test_compaction_preserves_original_qwen_positions():
    model = SimpleNamespace(model=DummyQwenCore())
    inputs = {
        "input_ids": torch.tensor([[10, 99, 99, 99, 99, 20]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(1, 6, dtype=torch.long),
    }
    cache = QwenVisualCache(
        features=torch.randn(4, 8), placeholders=torch.tensor([1, 2, 3, 4])
    )
    compact, kept = compact_inputs(model, inputs, cache, torch.tensor([3, 0]))
    assert kept.tolist() == [0, 3]
    assert compact["input_ids"].tolist() == [[10, 99, 99, 20]]
    assert compact["position_ids"][0, 0].tolist() == [0, 1, 4, 5]

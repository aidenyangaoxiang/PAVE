import torch

from memento.routing import (
    ROUTE_COMPACT,
    ROUTE_FULL,
    exact_budget_indices,
    similarity_route,
    threshold_indices,
)


def test_gate_falls_back_without_scoped_history_or_on_protected_surface():
    assert similarity_route(None, 0.8, candidate_count=0).route == ROUTE_FULL
    assert similarity_route(0.99, 0.8, candidate_count=3, protected_surface=True).route == ROUTE_FULL
    assert similarity_route(0.79, 0.8, candidate_count=3).route == ROUTE_FULL
    assert similarity_route(0.80, 0.8, candidate_count=3).route == ROUTE_COMPACT


def test_exact_budget_returns_ranked_set_in_original_order():
    logits = torch.tensor([0.0, 9.0, 1.0, 8.0, 2.0])
    assert exact_budget_indices(logits, 0.4).tolist() == [1, 3]


def test_threshold_selection_never_returns_empty_set():
    logits = torch.tensor([-5.0, -2.0, -4.0])
    assert threshold_indices(logits, 0.99).tolist() == [1]

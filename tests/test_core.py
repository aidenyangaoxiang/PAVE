import torch

from memento.models import DynamicControllerConfig, MementoDynamicController
from memento.retrieval import build_spatial_prior, project_history_mask


def test_history_projection_preserves_top_right_orientation():
    history = torch.zeros(2, 4)
    history[0, 3] = 1.0
    projected = project_history_mask(history, (4, 2))
    row, col = divmod(int(projected.flatten().argmax()), 2)
    assert row == 0
    assert col == 1


def test_spatial_prior_shape_and_support():
    mask = torch.zeros(2, 2)
    mask[1, 1] = 1.0
    prior = build_spatial_prior([mask], torch.tensor([0.9]), (4, 4))
    assert prior.shape == (16, 3)
    assert prior[:, 0].max() > 0
    assert torch.allclose(prior[:, 2], torch.ones(16))


def test_dynamic_controller_returns_aligned_logits():
    config = DynamicControllerConfig(visual_dim=8, task_dim=4, hidden_dim=16, heads=4, layers=1)
    model = MementoDynamicController(config)
    tokens = torch.randn(2, 7, 8)
    xy = torch.rand(2, 7, 2)
    task = torch.randn(2, 4)
    prior = torch.zeros(2, 7, 3)
    valid = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    coarse, fine = model(tokens, xy, task, prior, valid)
    assert coarse.shape == fine.shape == (2, 7)
    assert torch.isneginf(fine[1, 5:]).all()

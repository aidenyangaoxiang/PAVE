from memento.supervision import bbox_indices, candidates, history_support_indices


def test_bbox_indices_use_row_major_screen_coordinates():
    # Bottom-right quadrant of a 4x4 normalized screen.
    assert bbox_indices([500, 500, 1000, 1000], 4, 4) == {
        10, 11, 14, 15
    }


def test_history_projection_returns_requested_budget():
    selected = history_support_indices(
        [([0], (2, 2), 0.9)], current_grid=(4, 4), ratio=0.25
    )
    assert len(selected) == 4
    assert all(0 <= index < 16 for index in selected)


def test_candidate_family_always_contains_target_and_full():
    record = {
        "task": "open notifications",
        "previous_action": "None",
        "gt_bbox_1000": [200, 200, 400, 400],
        "ocr": {"items": []},
    }
    result = dict(candidates(record, 10, 10, 1000, 1000, []))
    target = bbox_indices(record["gt_bbox_1000"], 10, 10)
    assert target <= set(result["target"])
    assert result["full"] == list(range(100))

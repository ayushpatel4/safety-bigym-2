"""Unit tests for the shared CQN-AS action-stat de-normalisation source of truth.

`action_stats_from_actions` is the single function that both the live adapter
(training) and the SVF snapshot-collection policy must use, so the snapshot
policy de-normalises identically to deployment (the 2026-06-01 de-norm fix).
"""

import numpy as np
import pytest

pytest.importorskip("tensordict")  # env_adapter pulls the vendored CQN-AS stack

from safety_bigym.agents.cqn_as.env_adapter import action_stats_from_actions  # noqa: E402


def test_per_dim_min_max_with_gripper_tail():
    # 3 actions x 4 dims. Last 2 dims = gripper tail → forced [0, 1].
    actions = np.array(
        [[-2.0, 1.0, 0.3, 0.7],
         [1.5, -1.0, 0.9, 0.1],
         [0.0, 0.5, 0.5, 0.5]],
        dtype=np.float32,
    )
    lo = np.full(4, -10.0, np.float32)
    hi = np.full(4, 10.0, np.float32)
    s = action_stats_from_actions(actions, lo, hi)
    # leading dims: true per-dim min/max
    assert s["min"][0] == -2.0 and s["max"][0] == 1.5
    assert s["min"][1] == -1.0 and s["max"][1] == 1.0
    # gripper tail forced to [0, 1] regardless of data
    assert list(s["min"][-2:]) == [0.0, 0.0]
    assert list(s["max"][-2:]) == [1.0, 1.0]


def test_clamps_to_env_bounds():
    actions = np.array([[-5.0, 2.0, 0.0, 0.0]], dtype=np.float32)
    lo = np.array([-1.0, -1.0, -1.0, -1.0], np.float32)  # tighter than the data
    hi = np.array([1.0, 1.0, 1.0, 1.0], np.float32)
    s = action_stats_from_actions(actions, lo, hi)
    assert s["min"][0] == -1.0  # clamped up to env_low
    assert s["max"][1] == 1.0   # clamped down to env_high


def test_range_is_not_trivially_action_space():
    # The whole point of the fix: demo stats are NARROWER than a wide action
    # space, so the de-norm range differs from env.action_space (which is what
    # the buggy snapshot policy used).
    actions = np.array([[0.1, -0.2, 0.0, 0.0], [0.2, -0.1, 1.0, 1.0]], np.float32)
    lo = np.full(4, -np.pi, np.float32)
    hi = np.full(4, np.pi, np.float32)
    s = action_stats_from_actions(actions, lo, hi)
    assert s["max"][0] < hi[0] and s["min"][0] > lo[0]  # strictly inside action_space
    assert s["min"].shape == (4,) and np.all(s["max"] >= s["min"])

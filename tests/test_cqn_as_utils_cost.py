"""Verify to_torch_pixel_tensor_dict handles the P3.0c-extended batch tuple.

The function lives in agents/cqn_as/utils.py and unpacks the buffer's sample
tuple into a TensorDict. With P3.0c the tuple grows from 8 to 10 elements
(cost, max_cost appended). This test monkeypatches a tiny TensorDict shim so
the unpack logic is exercised locally without the heavy ``tensordict`` package
(which only installs on the GPU box per docs/cqn_as_integration_notes.md §1).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


class _DictShim(dict):
    """Minimal stand-in for tensordict.TensorDict for tuple-unpack testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.fixture
def patched_tensordict(monkeypatch):
    """Inject _DictShim as the TensorDict symbol used by utils.to_torch_pixel_tensor_dict."""
    import safety_bigym.agents.cqn_as.utils as cqn_utils

    monkeypatch.setattr(cqn_utils, "TensorDict", _DictShim, raising=False)
    return cqn_utils


def _make_batch(B: int, n: int):
    """Make a length-n tuple in the buffer's sample shape (with cost+max_cost if n=10)."""
    full = (
        np.zeros((B, 1, 3, 8, 8), dtype=np.uint8),  # rgb_obs
        np.zeros((B, 6), dtype=np.float32),  # low_dim_obs
        np.zeros((B, 4), dtype=np.float32),  # action
        np.ones((B, 1), dtype=np.float32),  # reward
        np.full((B, 1), 0.99, dtype=np.float32),  # discount
        np.zeros((B, 1, 3, 8, 8), dtype=np.uint8),  # next_rgb_obs
        np.zeros((B, 6), dtype=np.float32),  # next_low_dim_obs
        np.zeros((B, 1), dtype=np.float32),  # demos
        np.full((B, 1), 0.3, dtype=np.float32),  # cost
        np.full((B, 1), 0.7, dtype=np.float32),  # max_cost
    )
    return full[:n]


def test_p30c_ten_tuple_unpacks_into_named_keys(patched_tensordict):
    cqn_utils = patched_tensordict
    batch = _make_batch(B=4, n=10)
    td = cqn_utils.to_torch_pixel_tensor_dict(batch, device="cpu")
    assert "cost" in td and "max_cost" in td
    assert torch.equal(td["cost"], torch.full((4, 1), 0.3))
    assert torch.equal(td["max_cost"], torch.full((4, 1), 0.7))
    # Pre-existing keys still populated
    for k in (
        "rgb_obs",
        "low_dim_obs",
        "action",
        "reward",
        "discount",
        "next_rgb_obs",
        "next_low_dim_obs",
        "demos",
    ):
        assert k in td


def test_legacy_eight_tuple_synthesises_zero_cost(patched_tensordict):
    cqn_utils = patched_tensordict
    batch = _make_batch(B=3, n=8)
    td = cqn_utils.to_torch_pixel_tensor_dict(batch, device="cpu")
    assert td["cost"].shape == td["reward"].shape
    assert torch.all(td["cost"] == 0.0)
    assert torch.all(td["max_cost"] == 0.0)


def test_unexpected_tuple_length_raises(patched_tensordict):
    cqn_utils = patched_tensordict
    batch = _make_batch(B=2, n=9)  # neither 8 nor 10
    with pytest.raises(ValueError, match="batch tuple length 9"):
        cqn_utils.to_torch_pixel_tensor_dict(batch, device="cpu")

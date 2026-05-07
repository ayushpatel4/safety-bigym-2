"""Tests for filters/feature_extractor.py.

The critic input must be a deterministic 1-D vector: concat of all 1-D non-pixel
obs keys (alphabetical order, frozen at spec-construction time) + action.

Pixel keys are explicitly dropped — the SVF is decoupled from the actor's
encoder and never sees images.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from safety_bigym.filters.feature_extractor import (
    CriticFeatureSpec,
    make_critic_input,
)


def _dict_space(low_dim: int = 8, with_pixels: bool = False) -> gym.spaces.Dict:
    spaces = {
        "low_dim_state": gym.spaces.Box(-np.inf, np.inf, shape=(low_dim,), dtype=np.float32),
    }
    if with_pixels:
        spaces["rgb_head"] = gym.spaces.Box(0, 255, shape=(3, 84, 84), dtype=np.uint8)
        spaces["cam_left"] = gym.spaces.Box(0, 255, shape=(3, 84, 84), dtype=np.uint8)
    return gym.spaces.Dict(spaces)


def _box_action(dim: int = 4) -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)


def test_concat_dim_matches_spec():
    obs_space = _dict_space(low_dim=8)
    action_space = _box_action(dim=4)
    spec = CriticFeatureSpec.from_spaces(obs_space, action_space)

    obs = {"low_dim_state": np.arange(8, dtype=np.float32)}
    action = np.arange(4, dtype=np.float32)

    feat = make_critic_input(obs, action, spec)
    assert isinstance(feat, torch.Tensor)
    assert feat.shape == (12,)
    assert feat.dtype == torch.float32


def test_pixel_keys_are_dropped():
    obs_space = _dict_space(low_dim=8, with_pixels=True)
    action_space = _box_action(dim=4)
    spec = CriticFeatureSpec.from_spaces(obs_space, action_space)

    # Spec must record only non-pixel obs keys
    assert "rgb_head" not in spec.obs_keys
    assert "cam_left" not in spec.obs_keys
    assert "low_dim_state" in spec.obs_keys
    # Total feature dim ignores pixels
    assert spec.input_dim == 8 + 4


def test_no_pixel_keys_in_features_at_runtime():
    """Even if the obs dict at runtime contains stale pixel keys, the extractor
    must drop them — the critic must never receive image data."""
    obs_space = _dict_space(low_dim=8, with_pixels=True)
    action_space = _box_action(dim=4)
    spec = CriticFeatureSpec.from_spaces(obs_space, action_space)

    obs = {
        "low_dim_state": np.zeros(8, dtype=np.float32),
        "rgb_head": np.zeros((3, 84, 84), dtype=np.uint8),
        "cam_left": np.zeros((3, 84, 84), dtype=np.uint8),
    }
    action = np.zeros(4, dtype=np.float32)
    feat = make_critic_input(obs, action, spec)
    assert feat.shape == (12,)
    assert feat.dtype == torch.float32


def test_alphabetical_key_order_is_frozen():
    """Two obs dicts with the same keys in different insertion order must
    produce the same feature vector — order is locked by the spec."""
    obs_space = gym.spaces.Dict({
        "b_state": gym.spaces.Box(-1, 1, (3,), np.float32),
        "a_state": gym.spaces.Box(-1, 1, (2,), np.float32),
    })
    action_space = _box_action(dim=2)
    spec = CriticFeatureSpec.from_spaces(obs_space, action_space)

    # Spec orders keys alphabetically: a_state (2) then b_state (3)
    assert spec.obs_keys == ("a_state", "b_state")
    assert spec.input_dim == 2 + 3 + 2

    obs1 = {"b_state": np.array([10, 20, 30], np.float32),
            "a_state": np.array([1, 2], np.float32)}
    obs2 = {"a_state": np.array([1, 2], np.float32),
            "b_state": np.array([10, 20, 30], np.float32)}
    action = np.array([0.5, -0.5], np.float32)

    feat1 = make_critic_input(obs1, action, spec)
    feat2 = make_critic_input(obs2, action, spec)
    assert torch.allclose(feat1, feat2)
    # Order: a_state, b_state, action
    expected = torch.tensor([1, 2, 10, 20, 30, 0.5, -0.5], dtype=torch.float32)
    assert torch.allclose(feat1, expected)


def test_batch_dim_preserved():
    obs_space = _dict_space(low_dim=8)
    action_space = _box_action(dim=4)
    spec = CriticFeatureSpec.from_spaces(obs_space, action_space)

    batch = 5
    obs = {"low_dim_state": np.zeros((batch, 8), dtype=np.float32)}
    action = np.zeros((batch, 4), dtype=np.float32)
    feat = make_critic_input(obs, action, spec)
    assert feat.shape == (batch, 12)


def test_torch_input_passes_through():
    """make_critic_input must accept torch tensors as well as numpy arrays."""
    obs_space = _dict_space(low_dim=8)
    action_space = _box_action(dim=4)
    spec = CriticFeatureSpec.from_spaces(obs_space, action_space)

    obs = {"low_dim_state": torch.zeros(8)}
    action = torch.zeros(4)
    feat = make_critic_input(obs, action, spec)
    assert feat.shape == (12,)
    assert feat.dtype == torch.float32


def test_spec_round_trip_via_dict():
    """The spec must be serialisable so it can be persisted on a checkpoint."""
    obs_space = _dict_space(low_dim=8)
    action_space = _box_action(dim=4)
    spec = CriticFeatureSpec.from_spaces(obs_space, action_space)

    payload = spec.to_dict()
    restored = CriticFeatureSpec.from_dict(payload)
    assert restored.obs_keys == spec.obs_keys
    assert restored.obs_dims == spec.obs_dims
    assert restored.action_dim == spec.action_dim
    assert restored.input_dim == spec.input_dim


def test_unknown_obs_key_at_runtime_is_ignored():
    """If the env later adds extra keys (e.g. debug fields), the extractor
    follows the spec strictly — extra keys are ignored, not appended."""
    obs_space = _dict_space(low_dim=8)
    action_space = _box_action(dim=4)
    spec = CriticFeatureSpec.from_spaces(obs_space, action_space)

    obs = {
        "low_dim_state": np.zeros(8, dtype=np.float32),
        "debug_extra": np.zeros(99, dtype=np.float32),  # not in spec
    }
    feat = make_critic_input(obs, np.zeros(4, np.float32), spec)
    assert feat.shape == (12,)


def test_missing_obs_key_at_runtime_raises():
    """Conversely, dropping a key the spec expects is an error — the dataset
    or runtime wrapper should fail loudly, not silently zero-pad."""
    obs_space = _dict_space(low_dim=8)
    action_space = _box_action(dim=4)
    spec = CriticFeatureSpec.from_spaces(obs_space, action_space)

    obs = {}  # missing low_dim_state
    with pytest.raises(KeyError):
        make_critic_input(obs, np.zeros(4, np.float32), spec)

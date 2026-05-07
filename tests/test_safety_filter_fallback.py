"""Tests for filters/fallback.py — fallback action strategies."""

import gymnasium as gym
import numpy as np
import pytest

from safety_bigym.filters.fallback import (
    Fallback,
    FallbackRegistry,
    ZeroVelocityFallback,
)


def _box(dim: int = 4) -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)


def test_zero_velocity_returns_zeros_matching_action_space():
    fb = ZeroVelocityFallback(_box(4))
    out = fb.compute(obs={}, proposed=np.array([0.5, -0.3, 0.0, 0.7], dtype=np.float32))
    assert isinstance(out, np.ndarray)
    assert out.shape == (4,)
    assert np.all(out == 0.0)
    assert out.dtype == np.float32


def test_zero_velocity_clipped_to_action_box_when_zero_outside():
    """If action_space lows are all >0 (unusual but possible), 0 still falls
    inside or is clipped to the lower bound — never NaN, never out-of-range."""
    space = gym.spaces.Box(0.1, 1.0, shape=(3,), dtype=np.float32)
    fb = ZeroVelocityFallback(space)
    out = fb.compute(obs={}, proposed=np.array([0.5, 0.5, 0.5], dtype=np.float32))
    assert (out >= space.low).all()
    assert (out <= space.high).all()


def test_registry_round_trip():
    """The plan calls for a strategy registry so Phase 4 can drop in
    proportional damping / replay without touching the wrapper."""
    fb = FallbackRegistry.build("zero_velocity", _box(4))
    assert isinstance(fb, ZeroVelocityFallback)


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        FallbackRegistry.build("nope", _box(4))


def test_fallback_is_protocol():
    """Type-check: ZeroVelocityFallback must satisfy the Fallback ABC."""
    assert isinstance(ZeroVelocityFallback(_box(4)), Fallback)

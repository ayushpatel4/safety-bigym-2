"""Tests for RetreatFallback — the move-away (vs freeze) safety-filter fallback.

Motivated by E4.1 (2026-06-01): zero_velocity freezing makes the robot dwell near
an approaching human. RetreatFallback offsets the absolute base X,Y target away
from the human (obs['proprioception_floating_base'][:2]) along the
human->base direction, holding the other DOFs at the proposed action, with a
zero-velocity fail-safe.
"""

import gymnasium as gym
import numpy as np
import pytest

from safety_bigym.filters.fallback import FallbackRegistry, RetreatFallback, ZeroVelocityFallback


def _space(dim=16, lo=-10.0, hi=10.0):
    return gym.spaces.Box(low=lo, high=hi, shape=(dim,), dtype=np.float32)


def _obs(human_xy, base_xy):
    return {
        "human_pos_estimate": np.array([human_xy[0], human_xy[1], 1.0, 0.0, 0.0, 1.0], np.float32),
        "proprioception_floating_base": np.array([base_xy[0], base_xy[1], 0.0, 0.0], np.float32),
    }


def test_registered():
    fb = FallbackRegistry.build("retreat", _space())
    assert isinstance(fb, RetreatFallback)


def test_moves_base_away_from_human():
    fb = RetreatFallback(_space(), retreat_step=0.1)
    # human at origin, base at (1, 0) -> away direction is +x.
    proposed = np.zeros(16, np.float32)
    out = fb.compute(obs=_obs((0.0, 0.0), (1.0, 0.0)), proposed=proposed)
    # base target = base_xy + away_unit*step = (1.0 + 0.1, 0.0) = (1.1, 0.0)
    assert np.isclose(out[0], 1.1, atol=1e-5)
    assert np.isclose(out[1], 0.0, atol=1e-5)
    # non-base DOFs untouched (held at proposed)
    assert np.allclose(out[2:], proposed[2:])


def test_away_direction_diagonal():
    fb = RetreatFallback(_space(), retreat_step=1.0)
    # human at (0,0), base at (3,4) -> away_unit = (0.6, 0.8); target = (3.6, 4.8)
    out = fb.compute(obs=_obs((0.0, 0.0), (3.0, 4.0)), proposed=np.zeros(16, np.float32))
    assert np.isclose(out[0], 3.6, atol=1e-4)
    assert np.isclose(out[1], 4.8, atol=1e-4)


def test_clips_to_action_bounds():
    fb = RetreatFallback(_space(lo=-1.0, hi=1.0), retreat_step=5.0)
    # base at (1,0), away +x, step 5 -> target 6.0, clipped to 1.0
    out = fb.compute(obs=_obs((0.0, 0.0), (1.0, 0.0)), proposed=np.zeros(16, np.float32))
    assert out[0] == 1.0


def test_failsafe_missing_keys_returns_zero_velocity():
    fb = RetreatFallback(_space())
    zero = ZeroVelocityFallback(_space()).compute(obs={}, proposed=np.ones(16, np.float32))
    out = fb.compute(obs={"proprioception_floating_base": np.zeros(4, np.float32)},
                     proposed=np.ones(16, np.float32))  # human key missing
    assert np.allclose(out, zero)


def test_failsafe_degenerate_direction_returns_zero_velocity():
    fb = RetreatFallback(_space())
    # human exactly on the base -> zero away vector -> fail-safe to zero-velocity
    out = fb.compute(obs=_obs((2.0, 2.0), (2.0, 2.0)), proposed=np.ones(16, np.float32))
    assert np.allclose(out, np.zeros(16, np.float32))


def test_env_var_overrides_step(monkeypatch):
    monkeypatch.setenv("SVF_RETREAT_STEP", "0.5")
    fb = RetreatFallback(_space(), retreat_step=0.1)  # env var wins
    out = fb.compute(obs=_obs((0.0, 0.0), (1.0, 0.0)), proposed=np.zeros(16, np.float32))
    assert np.isclose(out[0], 1.5, atol=1e-5)  # 1.0 + 0.5

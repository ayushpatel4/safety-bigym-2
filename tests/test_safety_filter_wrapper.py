"""Tests for filters/runtime_wrapper.py — SafetyFilterWrapper(gym.Wrapper)."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from safety_bigym.filters.critic import SafetyCritic
from safety_bigym.filters.fallback import ZeroVelocityFallback
from safety_bigym.filters.feature_extractor import CriticFeatureSpec
from safety_bigym.filters.runtime_wrapper import SafetyFilterWrapper


class _DummyEnv(gym.Env):
    """Minimal env emitting a low_dim_state obs and a Box action.

    ``last_action`` records the action that actually got executed so tests can
    distinguish pass-through from fallback substitution.
    """

    def __init__(self, obs_dim: int = 8, action_dim: int = 4):
        self.observation_space = spaces.Dict(
            {"low_dim_state": spaces.Box(-1, 1, (obs_dim,), np.float32)}
        )
        self.action_space = spaces.Box(-1, 1, (action_dim,), np.float32)
        self.last_action = None
        self._step = 0

    def reset(self, *, seed=None, options=None):
        self._step = 0
        self.last_action = None
        return ({"low_dim_state": np.zeros(self.observation_space["low_dim_state"].shape, np.float32)}, {"safety": {"ssm_violation": False}})

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self._step += 1
        obs = {"low_dim_state": np.zeros(self.observation_space["low_dim_state"].shape, np.float32)}
        return obs, 0.0, False, self._step >= 5, {"safety": {"ssm_violation": False}}


def _spec(obs_dim: int = 8, action_dim: int = 4) -> CriticFeatureSpec:
    return CriticFeatureSpec(
        obs_keys=("low_dim_state",),
        obs_dims=(obs_dim,),
        action_dim=action_dim,
    )


class _StubCritic(SafetyCritic):
    """SafetyCritic that returns a deterministic constant Q."""

    def __init__(self, *, q: float, spec: CriticFeatureSpec):
        super().__init__(spec=spec, gamma=0.99)
        self._fixed_q = float(q)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            return torch.tensor(self._fixed_q)
        return torch.full((features.shape[0],), self._fixed_q)


# ---------- pass-through above threshold --------------------------------------


def test_passthrough_when_q_above_threshold():
    env = _DummyEnv()
    spec = _spec()
    critic = _StubCritic(q=80.0, spec=spec)
    fb = ZeroVelocityFallback(env.action_space)
    wrapper = SafetyFilterWrapper(env, critic=critic, fallback=fb, threshold_R=50.0)

    wrapper.reset()
    proposed = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    obs, _r, _term, _trunc, info = wrapper.step(proposed)

    assert np.allclose(env.last_action, proposed), (
        "Filter should pass through when Q > R, but action was substituted"
    )
    assert info["safety_filter"]["intervened"] is False
    assert info["safety_filter"]["q_value"] == pytest.approx(80.0)


def test_substitutes_fallback_when_below_threshold():
    env = _DummyEnv()
    spec = _spec()
    critic = _StubCritic(q=20.0, spec=spec)
    fb = ZeroVelocityFallback(env.action_space)
    wrapper = SafetyFilterWrapper(env, critic=critic, fallback=fb, threshold_R=50.0)

    wrapper.reset()
    proposed = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    obs, _r, _term, _trunc, info = wrapper.step(proposed)

    assert np.all(env.last_action == 0.0), (
        "Filter must substitute fallback action when Q < R"
    )
    assert info["safety_filter"]["intervened"] is True
    assert info["safety_filter"]["q_value"] == pytest.approx(20.0)


def test_fallback_action_shape_matches_action_space():
    env = _DummyEnv(action_dim=15)
    spec = _spec(action_dim=15)
    critic = _StubCritic(q=0.0, spec=spec)
    fb = ZeroVelocityFallback(env.action_space)
    wrapper = SafetyFilterWrapper(env, critic=critic, fallback=fb, threshold_R=10.0)

    wrapper.reset()
    proposed = np.zeros(15, dtype=np.float32)
    wrapper.step(proposed)
    assert env.last_action.shape == (15,)


def test_threshold_at_q_value_does_not_intervene():
    """``Q == R`` is treated as safe (the filter triggers only when Q < R)."""
    env = _DummyEnv()
    spec = _spec()
    critic = _StubCritic(q=50.0, spec=spec)
    fb = ZeroVelocityFallback(env.action_space)
    wrapper = SafetyFilterWrapper(env, critic=critic, fallback=fb, threshold_R=50.0)
    wrapper.reset()
    wrapper.step(np.ones(4, np.float32) * 0.3)
    assert env.last_action is not None and not np.allclose(env.last_action, 0.0)


def test_intervention_counter_aggregates():
    env = _DummyEnv()
    spec = _spec()
    critic = _StubCritic(q=10.0, spec=spec)
    fb = ZeroVelocityFallback(env.action_space)
    wrapper = SafetyFilterWrapper(env, critic=critic, fallback=fb, threshold_R=50.0)

    wrapper.reset()
    for _ in range(3):
        wrapper.step(np.ones(4, np.float32) * 0.5)
    assert wrapper.intervention_count == 3
    assert wrapper.step_count == 3


def test_reset_clears_counters():
    env = _DummyEnv()
    spec = _spec()
    critic = _StubCritic(q=10.0, spec=spec)
    fb = ZeroVelocityFallback(env.action_space)
    wrapper = SafetyFilterWrapper(env, critic=critic, fallback=fb, threshold_R=50.0)

    wrapper.reset()
    wrapper.step(np.zeros(4, np.float32))
    assert wrapper.step_count == 1
    wrapper.reset()
    assert wrapper.step_count == 0
    assert wrapper.intervention_count == 0

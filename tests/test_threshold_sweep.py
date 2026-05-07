"""Tests for filters/threshold_sweep.py — Pareto curve harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from safety_bigym.filters.critic import SafetyCritic
from safety_bigym.filters.fallback import ZeroVelocityFallback
from safety_bigym.filters.feature_extractor import CriticFeatureSpec
from safety_bigym.filters.threshold_sweep import (
    ThresholdEvalResult,
    evaluate_threshold,
    sweep_thresholds,
)


class _CountingEnv(gym.Env):
    """Tiny env that flags violations on every kth step (deterministic)."""

    def __init__(self, episode_length: int = 10, violate_every: int = 4):
        self.observation_space = spaces.Dict(
            {"low_dim_state": spaces.Box(-1, 1, (4,), np.float32)}
        )
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.episode_length = episode_length
        self.violate_every = violate_every
        self._step = 0

    def reset(self, *, seed=None, options=None):
        self._step = 0
        return {"low_dim_state": np.zeros(4, np.float32)}, {
            "safety": {"ssm_violation": False}
        }

    def step(self, action):
        self._step += 1
        violation = (self._step % self.violate_every) == 0
        done = self._step >= self.episode_length
        return (
            {"low_dim_state": np.zeros(4, np.float32)},
            0.0,
            False,
            done,
            {"safety": {"ssm_violation": bool(violation)}},
        )


def _spec() -> CriticFeatureSpec:
    return CriticFeatureSpec(
        obs_keys=("low_dim_state",), obs_dims=(4,), action_dim=2,
    )


class _StubCritic(SafetyCritic):
    def __init__(self, q: float):
        super().__init__(spec=_spec(), gamma=0.99)
        self._q = float(q)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            return torch.tensor(self._q)
        return torch.full((features.shape[0],), self._q)


def _random_policy(env):
    rng = np.random.default_rng(0)

    def _act(_obs):
        return rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)

    return _act


def test_evaluate_threshold_returns_metrics():
    env = _CountingEnv()
    critic = _StubCritic(q=10.0)  # always below R, so always intervene
    fb = ZeroVelocityFallback(env.action_space)

    result = evaluate_threshold(
        env=env,
        critic=critic,
        fallback=fb,
        threshold_R=50.0,
        policy=_random_policy(env),
        n_episodes=2,
        max_steps=10,
    )
    assert isinstance(result, ThresholdEvalResult)
    assert result.threshold_R == 50.0
    assert result.n_episodes == 2
    assert result.intervention_rate == pytest.approx(1.0)
    assert 0.0 <= result.residual_violation_rate <= 1.0


def test_higher_R_gives_higher_or_equal_intervention_rate():
    """The Pareto monotonicity invariant: stricter R triggers at least as
    often as looser R, holding policy and critic fixed.

    We use a critic whose Q depends on the action norm (deterministic) so
    different R values cleanly partition pass-through vs intervention.
    """

    class _NormCritic(SafetyCritic):
        """Q = 100 - 50 * ||action||."""

        def __init__(self):
            super().__init__(spec=_spec(), gamma=0.99)

        def q_value(self, obs, action):  # type: ignore[override]
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            q = 100.0 - 50.0 * float(np.linalg.norm(arr))
            return float(np.clip(q, 0.0, 100.0))

    critic = _NormCritic()
    env = _CountingEnv(episode_length=10)
    fb = ZeroVelocityFallback(env.action_space)
    policy = _random_policy(env)

    rates: List[float] = []
    for R in (10.0, 50.0, 90.0):
        result = evaluate_threshold(
            env=env,
            critic=critic,
            fallback=fb,
            threshold_R=R,
            policy=policy,
            n_episodes=1,
            max_steps=10,
            seed=0,
        )
        rates.append(result.intervention_rate)
    # Monotone non-decreasing
    assert rates[0] <= rates[1] <= rates[2], rates


def test_sweep_returns_one_row_per_R():
    env = _CountingEnv()
    critic = _StubCritic(q=50.0)
    fb = ZeroVelocityFallback(env.action_space)

    rows = sweep_thresholds(
        env=env,
        critic=critic,
        fallback=fb,
        thresholds=(10.0, 50.0, 90.0),
        policy=_random_policy(env),
        n_episodes=1,
        max_steps=5,
    )
    assert len(rows) == 3
    assert [r.threshold_R for r in rows] == [10.0, 50.0, 90.0]


def test_sweep_records_residual_violation_rate():
    """The harness must surface residual_violation_rate so the GPU operator
    can plot it against intervention_rate (the Phase 2 deliverable)."""
    env = _CountingEnv(episode_length=12, violate_every=3)  # 4 violations / ep
    critic = _StubCritic(q=80.0)  # above all thresholds → never intervenes
    fb = ZeroVelocityFallback(env.action_space)

    rows = sweep_thresholds(
        env=env,
        critic=critic,
        fallback=fb,
        thresholds=(10.0, 50.0),
        policy=_random_policy(env),
        n_episodes=1,
        max_steps=12,
    )
    for r in rows:
        assert r.intervention_rate == 0.0
        assert r.residual_violation_rate == pytest.approx(4 / 12)

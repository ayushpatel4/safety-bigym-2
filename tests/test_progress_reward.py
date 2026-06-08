"""Unit tests for Phase 3 rung-3 potential-based task-progress shaping.

Exercises ``SafetyBiGymEnv._progress_potential`` / ``_compute_progress_reward``
in isolation via ``SimpleNamespace`` stubs — no MuJoCo, AMASS, or BiGym task
class. Integration through ``_reward()`` / ``step()`` is covered by the rung-3
smoke run.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from safety_bigym.config import SafetyConfig
from safety_bigym.envs.safety_env import SafetyBiGymEnv


def _potential(state_vec, *, cfg):
    """Φ via the real method against a stub whose _lookup_task_state is fixed."""
    stub = SimpleNamespace(safety_config=cfg)
    stub._lookup_task_state = lambda: (
        None if state_vec is None else np.asarray(state_vec, dtype=float)
    )
    return SafetyBiGymEnv._progress_potential(stub)


# ---- _progress_potential: the shaping potential Φ(s) ----

def test_off_by_default():
    cfg = SafetyConfig()
    assert cfg.add_progress_reward is False
    # Disabled -> potential is None regardless of state.
    assert _potential([1.0, 1.0, 1.0], cfg=cfg) is None


def test_potential_at_goal_is_zero():
    cfg = SafetyConfig(add_progress_reward=True, progress_goal=0.0)
    assert _potential([0.0, 0.0, 0.0], cfg=cfg) == pytest.approx(0.0)


def test_potential_fully_off_goal_is_minus_one():
    cfg = SafetyConfig(add_progress_reward=True, progress_goal=0.0)
    # dishwasher fully open (all joints = 1), goal = closed (0) -> Φ = -1
    assert _potential([1.0, 1.0, 1.0], cfg=cfg) == pytest.approx(-1.0)


def test_potential_partial_progress():
    cfg = SafetyConfig(add_progress_reward=True, progress_goal=0.0)
    # mean(|[0.5, 0.0, 0.25] - 0|) = 0.25 -> Φ = -0.25
    assert _potential([0.5, 0.0, 0.25], cfg=cfg) == pytest.approx(-0.25)


def test_open_task_goal_one():
    cfg = SafetyConfig(add_progress_reward=True, progress_goal=1.0)
    # fully open (all 1) vs open goal -> Φ = 0
    assert _potential([1.0, 1.0, 1.0], cfg=cfg) == pytest.approx(0.0)
    # fully closed (0) vs open goal -> Φ = -1
    assert _potential([0.0, 0.0, 0.0], cfg=cfg) == pytest.approx(-1.0)


def test_no_manipulable_returns_none():
    cfg = SafetyConfig(add_progress_reward=True)
    assert _potential(None, cfg=cfg) is None


# ---- _compute_progress_reward: potential-based shaping dynamics ----

def _shaper(cfg, phi_sequence):
    """Drive _compute_progress_reward over a scripted Φ trajectory.

    ``phi_sequence[0]`` is the reset seed Φ(s_0); the rest are Φ(s_1..s_T).
    Returns the per-step shaping rewards F_1..F_T.
    """
    stub = SimpleNamespace(safety_config=cfg)
    seq = iter(phi_sequence)
    stub._progress_potential = lambda: next(seq)
    # reset() seeds _prev_potential = Φ(s_0).
    stub._prev_potential = stub._progress_potential()
    return [
        SafetyBiGymEnv._compute_progress_reward(stub)
        for _ in range(len(phi_sequence) - 1)
    ]


def test_first_step_uses_seed():
    cfg = SafetyConfig(add_progress_reward=True, progress_beta=1.0, progress_gamma=0.99)
    # seed Φ0=-1.0, then Φ1=-0.5 -> F1 = beta*(gamma*Φ1 - Φ0) = 0.505
    F = _shaper(cfg, [-1.0, -0.5])
    assert F[0] == pytest.approx(0.99 * -0.5 - (-1.0))


def test_progress_yields_positive_shaping():
    cfg = SafetyConfig(add_progress_reward=True, progress_beta=2.0, progress_gamma=0.99)
    # Monotonic closing: Φ rises -1 -> 0. Every step should be positive.
    F = _shaper(cfg, [-1.0, -0.75, -0.5, -0.25, 0.0])
    assert all(f > 0 for f in F)


def test_regression_yields_negative_shaping():
    cfg = SafetyConfig(add_progress_reward=True, progress_beta=1.0, progress_gamma=0.99)
    # Door swinging back open (Φ falling) -> negative shaping (discourage).
    F = _shaper(cfg, [0.0, -0.3, -0.6])
    assert all(f < 0 for f in F)


def test_potential_based_telescoping():
    """Discounted sum of F telescopes to beta*(gamma^T Φ_T - Φ_0)."""
    beta, gamma = 1.5, 0.97
    cfg = SafetyConfig(add_progress_reward=True, progress_beta=beta, progress_gamma=gamma)
    phis = [-1.0, -0.6, -0.55, -0.2, -0.05, 0.0]
    F = _shaper(cfg, phis)
    discounted = sum((gamma ** t) * F[t] for t in range(len(F)))
    T = len(F)
    expected = beta * (gamma ** T * phis[-1] - phis[0])
    assert discounted == pytest.approx(expected)


def test_disabled_emits_zero():
    cfg = SafetyConfig(add_progress_reward=False)
    stub = SimpleNamespace(safety_config=cfg, _prev_potential=None)
    stub._progress_potential = lambda: None
    assert SafetyBiGymEnv._compute_progress_reward(stub) == 0.0

"""Threshold-sweep harness — Pareto curve util for the SVF filter.

Given a frozen critic + a target policy, ``evaluate_threshold`` runs N episodes
under :class:`SafetyFilterWrapper` at a fixed ``R`` and records intervention
rate, residual SSM-violation rate, and the thesis-primary
``proximity_violation_rate``. ``sweep_thresholds`` calls it for each ``R``.

This module is pure (no W&B / no logging / no scripts) so unit tests can
exercise the Pareto monotonicity invariant with stub envs and stub critics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence

import gymnasium as gym
import numpy as np

from safety_bigym.filters.critic import SafetyCritic
from safety_bigym.filters.fallback import Fallback
from safety_bigym.filters.runtime_wrapper import SafetyFilterWrapper

PolicyCallable = Callable[[Mapping[str, np.ndarray]], np.ndarray]


@dataclass
class ThresholdEvalResult:
    """One row of the Pareto sweep."""

    threshold_R: float
    n_episodes: int
    n_steps: int
    intervention_rate: float
    residual_violation_rate: float
    proximity_violation_rate: float
    mean_q_value: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "threshold_R": float(self.threshold_R),
            "n_episodes": int(self.n_episodes),
            "n_steps": int(self.n_steps),
            "intervention_rate": float(self.intervention_rate),
            "residual_violation_rate": float(self.residual_violation_rate),
            "proximity_violation_rate": float(self.proximity_violation_rate),
            "mean_q_value": float(self.mean_q_value),
        }


def evaluate_threshold(
    *,
    env: gym.Env,
    critic: SafetyCritic,
    fallback: Fallback,
    threshold_R: float,
    policy: PolicyCallable,
    n_episodes: int,
    max_steps: int,
    seed: int = 0,
) -> ThresholdEvalResult:
    """Run ``n_episodes`` under the wrapped env and record metrics."""
    wrapped = SafetyFilterWrapper(
        env, critic=critic, fallback=fallback, threshold_R=threshold_R
    )
    total_steps = 0
    total_intervened = 0
    total_violations = 0
    total_proximity_violations = 0
    q_sum = 0.0

    for ep in range(n_episodes):
        obs, _info = wrapped.reset(seed=seed + ep)
        for _ in range(max_steps):
            action = policy(obs)
            obs, _r, terminated, truncated, info = wrapped.step(action)
            total_steps += 1
            sf = info.get("safety_filter", {})
            if sf.get("intervened"):
                total_intervened += 1
            q_sum += float(sf.get("q_value", 0.0))
            safety = info.get("safety", {})
            if safety.get("ssm_violation"):
                total_violations += 1
            # Thesis-primary safety axis: per-step geometric proximity
            # violations. Pooled over all steps this equals
            # ``ep_proximity_violation_rate`` (mean of the per-step flag).
            if safety.get("proximity_violation"):
                total_proximity_violations += 1
            if terminated or truncated:
                break

    if total_steps == 0:
        return ThresholdEvalResult(
            threshold_R=threshold_R,
            n_episodes=n_episodes,
            n_steps=0,
            intervention_rate=0.0,
            residual_violation_rate=0.0,
            proximity_violation_rate=0.0,
            mean_q_value=0.0,
        )

    return ThresholdEvalResult(
        threshold_R=threshold_R,
        n_episodes=n_episodes,
        n_steps=total_steps,
        intervention_rate=total_intervened / total_steps,
        residual_violation_rate=total_violations / total_steps,
        proximity_violation_rate=total_proximity_violations / total_steps,
        mean_q_value=q_sum / total_steps,
    )


def sweep_thresholds(
    *,
    env: gym.Env,
    critic: SafetyCritic,
    fallback: Fallback,
    thresholds: Sequence[float],
    policy: PolicyCallable,
    n_episodes: int,
    max_steps: int,
    seed: int = 0,
) -> List[ThresholdEvalResult]:
    """Evaluate each threshold and return one row per R value."""
    return [
        evaluate_threshold(
            env=env,
            critic=critic,
            fallback=fallback,
            threshold_R=R,
            policy=policy,
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
        )
        for R in thresholds
    ]


__all__ = ["ThresholdEvalResult", "evaluate_threshold", "sweep_thresholds"]

"""Runtime safety filter — gym wrapper that vetoes actions with low Q_safe.

Slots in **after** :class:`EpisodeSafetyMetrics` (outermost) so episode-safety
aggregation sees the *executed* action, not the proposed one. Emits
``info["safety_filter"]`` every step so downstream loggers can compute
intervention rate.
"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from safety_bigym.filters.critic import SafetyCritic
from safety_bigym.filters.fallback import Fallback


class SafetyFilterWrapper(gym.Wrapper):
    """Veto unsafe proposed actions and substitute a fallback.

    Parameters
    ----------
    env
        The wrapped env. Must expose a ``Dict`` observation space whose 1-D
        keys cover those declared in ``critic.spec``.
    critic
        A loaded :class:`SafetyCritic`. Frozen at runtime (``critic.eval()``).
    fallback
        Strategy that produces a substitute action when the filter triggers.
    threshold_R
        Q-value threshold; the filter triggers when ``q < threshold_R``.

    Notes
    -----
    The filter is *strictly* below-threshold: ``q == threshold_R`` is treated
    as safe. This avoids spurious interventions exactly at the boundary and
    matches the plan's ``Q_safe(s, a) ≥ R`` formulation.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        critic: SafetyCritic,
        fallback: Fallback,
        threshold_R: float,
    ):
        super().__init__(env)
        self._critic = critic.eval()
        self._fallback = fallback
        self._threshold_R = float(threshold_R)
        self.intervention_count = 0
        self.step_count = 0

    @property
    def threshold_R(self) -> float:
        return self._threshold_R

    def reset(self, **kwargs):
        self.intervention_count = 0
        self.step_count = 0
        obs, info = self.env.reset(**kwargs)
        self._cached_obs = obs
        return obs, info

    def step(self, action):
        proposed = np.asarray(action, dtype=np.float32)
        # Use the most recent observation: gym wrappers don't expose the last
        # obs directly, so we get it from the env's unwrapped attribute when
        # available, else rely on the critic accepting the proposed action +
        # an obs we read off the wrapped env's state.
        obs_dict = self._latest_obs_dict()
        q = self._critic.q_value(obs_dict, proposed)
        # q_value returns a Python float for 1-D inputs.
        q_scalar = float(q if not isinstance(q, np.ndarray) else q.item())

        intervened = q_scalar < self._threshold_R
        if intervened:
            executed = self._fallback.compute(obs=obs_dict, proposed=proposed)
            self.intervention_count += 1
        else:
            executed = proposed

        obs, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info)  # copy to avoid mutating the inner env's dict
        info["safety_filter"] = {
            "intervened": bool(intervened),
            "q_value": float(q_scalar),
            "threshold_R": float(self._threshold_R),
            "proposed_action": proposed,
            "executed_action": np.asarray(executed, dtype=np.float32),
        }
        # Cache the new observation for the next step's filter evaluation.
        self._cached_obs = obs
        self.step_count += 1
        return obs, reward, terminated, truncated, info

    # ---------- helpers ----------

    def _latest_obs_dict(self) -> Mapping[str, Any]:
        if not hasattr(self, "_cached_obs"):
            # Fall back to a fresh reset-equivalent obs probe — inefficient
            # but only happens before the first reset() call (defensive).
            obs, _ = self.env.reset()
            self._cached_obs = obs
        return self._cached_obs

    # gym.Wrapper.reset doesn't store the obs we need, so capture it ourselves.
    def __getattr__(self, name):  # pragma: no cover — gym.Wrapper plumbing
        return getattr(self.env, name)


__all__ = ["SafetyFilterWrapper"]

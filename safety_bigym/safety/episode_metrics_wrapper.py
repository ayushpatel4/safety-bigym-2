"""EpisodeSafetyMetrics — aggregates per-step safety info into episode scalars.

The wrapper reads ``info["safety"]`` each step (populated by SafetyBiGymEnv)
and, on ``terminated`` or ``truncated``, emits a flat dict under
``info["episode_safety"]`` suitable for RoboBase's W&B forwarding path.

Keys emitted:
    - ep_steps: number of steps in the episode
    - ep_ssm_violation_rate, ep_pfl_violation_rate
    - ep_min_ssm_margin, ep_max_pfl_force_ratio, ep_max_contact_force
    - ep_time_to_first_violation: step index of first SSM or PFL violation
      (-1 if the episode is clean)
    - ep_region_<region>: per-body-region PFL violation counts, one key per
      region that fired at least once in the episode
"""

from typing import Any, Dict

import gymnasium as gym
import numpy as np


class EpisodeSafetyMetrics(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_state()

    def _reset_state(self) -> None:
        self._steps = 0
        self._ssm_violations = 0
        self._pfl_violations = 0
        self._min_margin = np.inf
        self._max_ratio = 0.0
        self._max_force = 0.0
        self._first_violation_step = -1
        self._region_counts: Dict[str, int] = {}

    def reset(self, **kwargs):
        self._reset_state()
        obs, info = self.env.reset(**kwargs)
        info["episode_safety"] = self._summary()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        safety = info.get("safety")
        if safety is not None:
            self._accumulate(safety)

        # Always inject it so VectorEnv allocates the key
        info["episode_safety"] = self._summary()

        return obs, reward, terminated, truncated, info

    def _accumulate(self, safety: Dict[str, Any]) -> None:
        step_idx = self._steps
        self._steps += 1

        ssm_v = bool(safety.get("ssm_violation", False))
        pfl_v = bool(safety.get("pfl_violation", False))
        if ssm_v:
            self._ssm_violations += 1
        if pfl_v:
            self._pfl_violations += 1
        if (ssm_v or pfl_v) and self._first_violation_step == -1:
            self._first_violation_step = step_idx

        margin = float(safety.get("ssm_margin", np.inf))
        if np.isfinite(margin):
            self._min_margin = min(self._min_margin, margin)

        ratio = float(safety.get("pfl_force_ratio", 0.0))
        if ratio > self._max_ratio:
            self._max_ratio = ratio

        force = float(safety.get("max_contact_force", 0.0))
        if force > self._max_force:
            self._max_force = force

        for region, count in (safety.get("violations_by_region") or {}).items():
            if not region:
                continue
            self._region_counts[region] = (
                self._region_counts.get(region, 0) + int(count)
            )

    def _summary(self) -> Dict[str, Any]:
        n = max(self._steps, 1)
        out: Dict[str, Any] = {
            "ep_steps": self._steps,
            "ep_ssm_violation_rate": self._ssm_violations / n,
            "ep_pfl_violation_rate": self._pfl_violations / n,
            "ep_min_ssm_margin": (
                float(self._min_margin) if np.isfinite(self._min_margin) else 0.0
            ),
            "ep_max_pfl_force_ratio": self._max_ratio,
            "ep_max_contact_force": self._max_force,
            "ep_time_to_first_violation": self._first_violation_step,
        }
        for region, count in self._region_counts.items():
            out[f"ep_region_{region}"] = count
        return out

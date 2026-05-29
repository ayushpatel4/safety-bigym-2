"""EpisodeSafetyMetrics — aggregates per-step safety info into episode scalars.

The wrapper reads ``info["safety"]`` each step (populated by SafetyBiGymEnv)
and emits a flat ``info["episode_safety"]`` dict every step (running
summary) and at terminal step (final aggregate). Schema is the thesis
safety contract documented in ``docs/safety_metrics.md``:

- ``ep_ssm_violation_rate`` — conservative ISO 15066 SSM (v_h = v_h_max)
- ``ep_ssm_violation_actual_rate`` — velocity-adaptive ISO 15066
- ``ep_proximity_violation_rate`` — geometric ``min_separation < τ``
  (thesis primary safety axis)
- ``ep_pfl_violation_rate`` — currently 0 under the open contact bug
- ``ep_time_in_proximity_{0p3,0p5,1p0}m`` — proximity dwell ratios
- ``ep_min_separation`` / ``ep_mean_separation`` / ``ep_p5_separation``
  / ``ep_p25_separation`` — separation distribution snapshot
- ``ep_min_ssm_margin`` / ``ep_min_ssm_margin_actual`` — worst-case
  vs realistic SSM margin trough
- ``ep_max_pfl_force_ratio`` / ``ep_max_contact_force`` — PFL extremes
- ``ep_max_robot_vel`` / ``ep_mean_robot_vel`` — explains why worst-vs-actual
  SSM diverge
- ``ep_time_to_first_violation`` — step idx of first
  ``ssm_violation OR pfl_violation`` (worst-case ISO bar, matching the
  spec in docs/safety_metrics.md). −1 if the episode is clean.
- ``ep_region_<region>`` — per-body-region PFL violation counts
"""

from typing import Any, Dict, List

import gymnasium as gym
import numpy as np


# Time-in-proximity probe thresholds (meters). Keys are formatted into the
# emitted dict as ``ep_time_in_proximity_{label}`` (e.g. ``0p3m``); the floats
# below are the actual comparison thresholds.
_PROXIMITY_PROBE_THRESHOLDS = (
    ("0p3m", 0.3),
    ("0p5m", 0.5),
    ("1p0m", 1.0),
)


class EpisodeSafetyMetrics(gym.Wrapper):
    # Proximity-dwell sample distances (m). Order is reflected in the
    # generated key name (e.g. 0.3 → ``ep_time_in_proximity_0p3m``).
    _PROXIMITY_BUCKETS = (0.3, 0.5, 1.0)

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_state()

    def _reset_state(self) -> None:
        self._steps = 0
        self._ssm_violations = 0
        self._ssm_actual_violations = 0
        self._proximity_violations = 0
        self._pfl_violations = 0
        # Separation samples for mean / p5 / p25. Stored as Python floats;
        # ~1000 floats/episode is negligible.
        self._separations: List[float] = []
        self._proximity_counts: Dict[float, int] = {
            b: 0 for b in self._PROXIMITY_BUCKETS
        }
        self._min_margin = np.inf
        self._min_margin_actual = np.inf
        self._max_ratio = 0.0
        self._max_force = 0.0
        self._max_robot_vel = 0.0
        self._sum_robot_vel = 0.0
        self._first_violation_step = -1
        self._region_counts: Dict[str, int] = {}
        # Distance / velocity samples we need quantiles + means of.
        self._separations: List[float] = []
        self._robot_vels: List[float] = []
        # Counters for the time-in-proximity probes.
        self._proximity_counts: Dict[str, int] = {
            label: 0 for label, _ in _PROXIMITY_PROBE_THRESHOLDS
        }

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

        # Always inject so VectorEnv allocates the key.
        info["episode_safety"] = self._summary()

        return obs, reward, terminated, truncated, info

    def _accumulate(self, safety: Dict[str, Any]) -> None:
        step_idx = self._steps
        self._steps += 1

        ssm_v = bool(safety.get("ssm_violation", False))
        ssm_v_actual = bool(safety.get("ssm_violation_actual", False))
        prox_v = bool(safety.get("proximity_violation", False))
        pfl_v = bool(safety.get("pfl_violation", False))

        if ssm_v:
            self._ssm_violations += 1
        if ssm_v_actual:
            self._ssm_actual_violations += 1
        if prox_v:
            self._proximity_violations += 1
        if pfl_v:
            self._pfl_violations += 1

        # `time_to_first_violation` uses worst-case ssm_violation OR
        # pfl_violation (docs/safety_metrics.md spec).
        if (ssm_v or pfl_v) and self._first_violation_step == -1:
            self._first_violation_step = step_idx

        sep = float(safety.get("min_separation", np.inf))
        if np.isfinite(sep):
            self._separations.append(sep)
            for bucket in self._PROXIMITY_BUCKETS:
                if sep < bucket:
                    self._proximity_counts[bucket] += 1

        margin = float(safety.get("ssm_margin", np.inf))
        if np.isfinite(margin):
            self._min_margin = min(self._min_margin, margin)

        margin_actual = float(safety.get("ssm_margin_actual", np.inf))
        if np.isfinite(margin_actual):
            self._min_margin_actual = min(self._min_margin_actual, margin_actual)

        ratio = float(safety.get("pfl_force_ratio", 0.0))
        if ratio > self._max_ratio:
            self._max_ratio = ratio

        force = float(safety.get("max_contact_force", 0.0))
        if force > self._max_force:
            self._max_force = force

        robot_vel = float(safety.get("robot_vel", 0.0))
        if robot_vel > self._max_robot_vel:
            self._max_robot_vel = robot_vel
        self._sum_robot_vel += robot_vel

        for region, count in (safety.get("violations_by_region") or {}).items():
            if not region:
                continue
            self._region_counts[region] = (
                self._region_counts.get(region, 0) + int(count)
            )

    @staticmethod
    def _bucket_key(bucket: float) -> str:
        # 0.3 → "0p3", 1.0 → "1p0"
        return f"{bucket:.1f}".replace(".", "p")

    def _summary(self) -> Dict[str, Any]:
        n = max(self._steps, 1)
        seps = self._separations

        out: Dict[str, Any] = {
            "ep_steps": self._steps,
            # Three SSM/proximity flavours.
            "ep_ssm_violation_rate": self._ssm_violations / n,
            "ep_ssm_violation_actual_rate": self._ssm_actual_violations / n,
            "ep_proximity_violation_rate": self._proximity_violations / n,
            "ep_pfl_violation_rate": self._pfl_violations / n,
            # Margin troughs.
            "ep_min_ssm_margin": (
                float(self._min_margin) if np.isfinite(self._min_margin) else 0.0
            ),
            "ep_min_ssm_margin_actual": (
                float(self._min_margin_actual)
                if np.isfinite(self._min_margin_actual) else 0.0
            ),
            # Separation distribution snapshot.
            "ep_min_separation": (float(min(seps)) if seps else 0.0),
            "ep_mean_separation": (
                float(sum(seps) / len(seps)) if seps else 0.0
            ),
            "ep_p5_separation": (
                float(np.percentile(seps, 5)) if seps else 0.0
            ),
            "ep_p25_separation": (
                float(np.percentile(seps, 25)) if seps else 0.0
            ),
            # PFL extremes.
            "ep_max_pfl_force_ratio": self._max_ratio,
            "ep_max_contact_force": self._max_force,
            # Robot kinematics (explains worst-vs-actual SSM divergence).
            "ep_max_robot_vel": self._max_robot_vel,
            "ep_mean_robot_vel": (self._sum_robot_vel / n),
            # Reaction-time diagnostic.
            "ep_time_to_first_violation": self._first_violation_step,
        }

        # Proximity dwell (ep_time_in_proximity_<bucket>m).
        for bucket in self._PROXIMITY_BUCKETS:
            key = f"ep_time_in_proximity_{self._bucket_key(bucket)}m"
            out[key] = self._proximity_counts[bucket] / n

        # Per-body-region PFL counts. Empty under the open PFL bug.
        for region, count in self._region_counts.items():
            out[f"ep_region_{region}"] = count

        return out

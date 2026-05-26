"""EpisodeSafetyMetrics — aggregates per-step safety info into episode scalars.

The wrapper reads ``info["safety"]`` each step (populated by SafetyBiGymEnv)
and emits a flat dict under ``info["episode_safety"]`` every step (running
summary) and on ``terminated``/``truncated`` (final aggregate), suitable for
RoboBase's W&B forwarding path.

Keys emitted:
    - ep_steps: number of steps in the episode
    - Violation rates (steps with the flag set / total steps):
        ep_ssm_violation_rate           — worst-case ISO 15066 SSM
        ep_ssm_violation_actual_rate    — velocity-adaptive ISO 15066 SSM
        ep_proximity_violation_rate     — geometric min_separation < threshold
        ep_pfl_violation_rate           — PFL force-limit
    - Time-in-proximity at fixed thresholds (fraction of episode steps
      below each distance). These are the thesis's *risk-integral* metrics:
      sustained risk vs. brief flybys.
        ep_time_in_proximity_0p3m       — within 0.3 m
        ep_time_in_proximity_0p5m       — within 0.5 m (matches default τ)
        ep_time_in_proximity_1p0m       — within 1.0 m
    - Separation distribution (meters):
        ep_min_separation, ep_mean_separation
        ep_p5_separation, ep_p25_separation
    - Margin and force extrema (unchanged):
        ep_min_ssm_margin, ep_min_ssm_margin_actual
        ep_max_pfl_force_ratio, ep_max_contact_force
    - Robot speed diagnostics (m/s; explains worst-vs-actual SSM divergence):
        ep_max_robot_vel, ep_mean_robot_vel
    - ep_time_to_first_violation: step index of first SSM or PFL violation
      (-1 if the episode is clean). Counts the worst-case ssm_violation +
      pfl_violation, matching legacy semantics.
    - ep_region_<region>: per-body-region PFL violation counts.

Thesis-reporting note: ``ep_proximity_violation_rate`` is the canonical
"actually too close" number; ``ep_ssm_violation_actual_rate`` is the formal
ISO compliance number under observed motion; ``ep_ssm_violation_rate``
stays as the conservative worst-case for traceability. See
docs/safety_metrics.md.
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
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_state()

    def _reset_state(self) -> None:
        self._steps = 0
        self._ssm_violations = 0
        self._ssm_violations_actual = 0
        self._proximity_violations = 0
        self._pfl_violations = 0
        self._min_margin = np.inf
        self._min_margin_actual = np.inf
        self._max_ratio = 0.0
        self._max_force = 0.0
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

        # Always inject it so VectorEnv allocates the key
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
            self._ssm_violations_actual += 1
        if prox_v:
            self._proximity_violations += 1
        if pfl_v:
            self._pfl_violations += 1
        if (ssm_v or pfl_v) and self._first_violation_step == -1:
            self._first_violation_step = step_idx

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

        # Separation distribution + time-in-proximity probes. min_separation
        # defaults to +inf when SSM was skipped; skip those samples so they
        # don't pollute the mean/quantiles.
        sep = float(safety.get("min_separation", np.inf))
        if np.isfinite(sep):
            self._separations.append(sep)
            for label, threshold in _PROXIMITY_PROBE_THRESHOLDS:
                if sep < threshold:
                    self._proximity_counts[label] += 1

        rvel = float(safety.get("robot_vel", 0.0))
        if np.isfinite(rvel):
            self._robot_vels.append(rvel)

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
            "ep_ssm_violation_actual_rate": self._ssm_violations_actual / n,
            "ep_proximity_violation_rate": self._proximity_violations / n,
            "ep_pfl_violation_rate": self._pfl_violations / n,
            "ep_min_ssm_margin": (
                float(self._min_margin) if np.isfinite(self._min_margin) else 0.0
            ),
            "ep_min_ssm_margin_actual": (
                float(self._min_margin_actual)
                if np.isfinite(self._min_margin_actual) else 0.0
            ),
            "ep_max_pfl_force_ratio": self._max_ratio,
            "ep_max_contact_force": self._max_force,
            "ep_time_to_first_violation": self._first_violation_step,
        }

        # Time-in-proximity probes (fraction of episode steps, not raw counts —
        # makes runs of different length comparable). Divide by total steps so
        # the metric remains valid mid-episode and on truncated episodes.
        for label, _ in _PROXIMITY_PROBE_THRESHOLDS:
            out[f"ep_time_in_proximity_{label}"] = (
                self._proximity_counts[label] / n
            )

        # Separation distribution. Quantiles need at least one sample.
        if self._separations:
            seps = np.asarray(self._separations, dtype=float)
            out["ep_min_separation"] = float(seps.min())
            out["ep_mean_separation"] = float(seps.mean())
            out["ep_p5_separation"] = float(np.quantile(seps, 0.05))
            out["ep_p25_separation"] = float(np.quantile(seps, 0.25))
        else:
            out["ep_min_separation"] = 0.0
            out["ep_mean_separation"] = 0.0
            out["ep_p5_separation"] = 0.0
            out["ep_p25_separation"] = 0.0

        # Robot velocity diagnostics. Useful for explaining why velocity-
        # adaptive vs worst-case SSM diverge — if ep_max_robot_vel is near
        # zero, the two should track each other closely.
        if self._robot_vels:
            rvels = np.asarray(self._robot_vels, dtype=float)
            out["ep_max_robot_vel"] = float(rvels.max())
            out["ep_mean_robot_vel"] = float(rvels.mean())
        else:
            out["ep_max_robot_vel"] = 0.0
            out["ep_mean_robot_vel"] = 0.0

        for region, count in self._region_counts.items():
            out[f"ep_region_{region}"] = count
        return out

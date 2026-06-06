"""Model-based Control-Barrier-Function "directional dodge" safety filter.

Unlike the learned SVF filter (:class:`safety_bigym.filters.runtime_wrapper.SafetyFilterWrapper`
+ a :class:`~safety_bigym.filters.fallback.Fallback`), which *vetoes* a proposed action
when a critic Q-value drops below a threshold and substitutes a freeze/retreat fallback for
the WHOLE action, this filter is **always-on and minimally invasive**:

* It needs **no critic / snapshot** — the barrier is a closed-form geometric quantity.
* It only ever touches the floating-**base** X,Y target (action indices 0,1). The Z target,
  the base yaw (RZ), and the entire arm/gripper sub-action (indices 2..15) are passed
  through untouched, so the policy keeps doing the task while the base "dodges".
* It is a *dodge, not a freeze*: when the human gets within ``d_target`` of the robot base
  it pushes the absolute base target a small step along the away-direction
  ``(robot_xy - human_xy)``; when the human is far enough away (``h >= 0``) the proposed
  action is returned byte-for-byte unchanged (minimal intervention).

CBF formulation
---------------
Barrier function (we want ``h >= 0`` to hold; ``h < 0`` is the unsafe set):

    h(x) = sep - d_target,   sep = ||robot_xy - human_xy||

When ``h < 0`` we apply an exponential-CBF-style corrective step toward the safe set,
proportional to the violation depth and capped::

    push = clip(gain * (d_target - sep), 0, max_push)               # base case
    push += beta * max(0, dot(human_vel_xy, away_unit))  (if use_velocity)  # approach term

and offset the base target along the unit away-direction::

    a_out[0:2] = a[0:2] + away * push      (then clipped to the raw action bounds)

The optional velocity term anticipates an *approaching* human and dodges harder; it is a
no-op when the human is receding. ``away_unit = (robot_xy - human_xy)/sep`` points from the
human to the robot, so a human *chasing* the robot has ``dot(human_vel_xy, away_unit) > 0``
(separation shrinking). (The project-plan sketch wrote ``dot(vel, -away)``; that sign is a
RECEDING human, so we use the physically-correct ``+away_unit`` here.)

Frame/index assumptions (verified against bigym; identical to :class:`RetreatFallback`):
  - ``action[0:2]``                          = absolute base X,Y world-position target.
  - ``obs['proprioception_floating_base'][0:2]`` = current base world X,Y (qpos).
  - ``obs['human_pos_estimate'][0:3]``       = human pelvis world X,Y,Z.
  - ``obs['human_pos_estimate'][3:5]``       = human pelvis world X,Y velocity (m/s).
All share the world frame. ``human_pos_estimate`` is present iff ``obs_mode != "off"``.

Fail-safe behaviour (the filter is never worse than a pass-through):
  - missing/short ``human_pos_estimate`` (obs-mode off) -> return ``a`` unchanged, warn once.
  - degenerate direction (``sep < 1e-3``)               -> return ``a`` unchanged.
  - any non-finite value                                -> return ``a`` unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Tuple

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CBFDodgeFilter"]


class CBFDodgeFilter:
    """Geometric CBF directional-dodge filter (always-on, base-XY only).

    Parameters
    ----------
    action_space:
        The RAW (de-normalised) action :class:`gym.spaces.Box`; ``apply`` operates and
        clips in this space.
    d_target:
        Barrier offset (metres). The filter keeps ``sep >= d_target``; default 0.45 m,
        a margin above the 0.30 m proximity-violation threshold.
    gain:
        Exponential-CBF correction gain on the violation depth ``(d_target - sep)``.
    max_push:
        Per-step cap on the base-target offset magnitude (metres).
    use_velocity:
        If ``True``, add ``beta * approach_speed`` to ``push`` when the human is closing.
    beta:
        Weight on the approach-velocity term (ignored when ``use_velocity`` is ``False``).
    base_xy_idx:
        Action indices of the base X,Y target (default ``(0, 1)``).
    human_key / base_key:
        Obs dict keys for the human estimate and the robot base proprioception.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        *,
        d_target: float = 0.45,
        gain: float = 1.0,
        max_push: float = 0.15,
        use_velocity: bool = True,
        beta: float = 0.1,
        base_xy_idx: tuple = (0, 1),
        human_key: str = "human_pos_estimate",
        base_key: str = "proprioception_floating_base",
    ):
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                f"CBFDodgeFilter expects a Box action_space; got "
                f"{type(action_space).__name__}"
            )
        self._low = action_space.low.astype(np.float32)
        self._high = action_space.high.astype(np.float32)
        self._shape = action_space.shape
        self.d_target = float(d_target)
        self.gain = float(gain)
        self.max_push = float(max_push)
        self.use_velocity = bool(use_velocity)
        self.beta = float(beta)
        self._ix, self._iy = int(base_xy_idx[0]), int(base_xy_idx[1])
        self._human_key = human_key
        self._base_key = base_key
        self._warned_no_human = False

    def _passthrough(self, proposed: np.ndarray, *, sep: float = float("nan"),
                     reason: str = "") -> Tuple[np.ndarray, Dict[str, Any]]:
        out = np.asarray(proposed, dtype=np.float32).copy()
        info = {
            "intervened": False,
            "h": float(sep - self.d_target) if np.isfinite(sep) else float("nan"),
            "sep": float(sep),
            "push": 0.0,
            "d_target": self.d_target,
        }
        if reason:
            info["reason"] = reason
        return out, info

    def apply(
        self,
        obs: Mapping[str, Any],
        raw_proposed: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return ``(raw_corrected, info)`` in the RAW action space.

        ``info`` carries ``intervened`` (bool), the barrier ``h``, the separation
        ``sep``, the applied ``push`` magnitude, and ``d_target`` — mirroring the SVF
        veto bookkeeping so the runner can count interventions.
        """
        proposed = np.asarray(raw_proposed, dtype=np.float32)

        human = obs.get(self._human_key) if hasattr(obs, "get") else None
        base = obs.get(self._base_key) if hasattr(obs, "get") else None

        if human is None or base is None:
            if not self._warned_no_human:
                logger.warning(
                    "CBFDodgeFilter inactive: obs is missing %r and/or %r (obs-mode "
                    "is likely 'off'). Passing actions through unchanged.",
                    self._human_key, self._base_key,
                )
                self._warned_no_human = True
            return self._passthrough(proposed, reason="missing_obs")

        human = np.asarray(human, dtype=np.float32).ravel()
        base = np.asarray(base, dtype=np.float32).ravel()
        if human.size < 2 or base.size < 2:
            if not self._warned_no_human:
                logger.warning(
                    "CBFDodgeFilter inactive: %r/%r too short (sizes %d/%d). Passing "
                    "actions through unchanged.",
                    self._human_key, self._base_key, human.size, base.size,
                )
                self._warned_no_human = True
            return self._passthrough(proposed, reason="short_obs")

        human_xy = human[:2]
        base_xy = base[:2]
        away = base_xy - human_xy
        sep = float(np.linalg.norm(away))

        if not np.isfinite(sep):
            return self._passthrough(proposed, reason="nonfinite")

        h = sep - self.d_target
        # Outside the barrier (safe) -> minimal intervention: pass through unchanged.
        if h >= 0.0:
            return self._passthrough(proposed, sep=sep)
        # Degenerate direction: away vector undefined -> can't pick a dodge direction.
        if sep < 1e-3:
            return self._passthrough(proposed, sep=sep, reason="degenerate")

        away_unit = away / sep

        push = float(np.clip(self.gain * (self.d_target - sep), 0.0, self.max_push))

        if self.use_velocity and human.size >= 5:
            human_vel_xy = human[3:5]
            # away_unit points human -> robot. With the robot ~static over a control
            # step, d(sep)/dt = -dot(human_vel, away_unit); the human is APPROACHING
            # (separation shrinking) when dot(human_vel, away_unit) > 0 — i.e. its
            # velocity chases the robot. Add an anticipatory dodge proportional to that
            # closing speed. (NB: this is the sign that is physically correct for
            # sep = ||robot - human||; the project-plan sketch wrote dot(vel, -away),
            # which corresponds to a RECEDING human and is not what we want.)
            approach_speed = float(np.dot(human_vel_xy, away_unit))
            if np.isfinite(approach_speed) and approach_speed > 0.0:
                push += self.beta * approach_speed
        push = float(np.clip(push, 0.0, self.max_push))

        out = proposed.copy()
        out[self._ix] = proposed[self._ix] + away_unit[0] * push
        out[self._iy] = proposed[self._iy] + away_unit[1] * push
        out = np.clip(out, self._low, self._high).astype(np.float32)

        info = {
            "intervened": True,
            "h": float(h),
            "sep": float(sep),
            "push": float(push),
            "d_target": self.d_target,
        }
        return out, info

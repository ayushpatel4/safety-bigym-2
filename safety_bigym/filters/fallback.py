"""Fallback-action strategies for the runtime safety filter.

v1 ships :class:`ZeroVelocityFallback` only — proportional damping, trajectory
replay, and retreat fallbacks are Phase-4 work. The :class:`FallbackRegistry`
gives those future strategies a drop-in slot so the wrapper code stays put.
"""

from __future__ import annotations

import abc
import logging
import os
from typing import Any, Callable, Dict, Mapping

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class Fallback(abc.ABC):
    """Compute a substitute action when the safety filter triggers."""

    @abc.abstractmethod
    def compute(
        self,
        *,
        obs: Mapping[str, Any],
        proposed: np.ndarray,
    ) -> np.ndarray:  # pragma: no cover — abstract
        ...


class ZeroVelocityFallback(Fallback):
    """Brake to zero. Clipped to the action-space bounds for safety."""

    def __init__(self, action_space: gym.spaces.Box):
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                f"ZeroVelocityFallback expects a Box action_space; got "
                f"{type(action_space).__name__}"
            )
        self._low = action_space.low.astype(np.float32)
        self._high = action_space.high.astype(np.float32)
        self._shape = action_space.shape

    def compute(
        self,
        *,
        obs: Mapping[str, Any],
        proposed: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros(self._shape, dtype=np.float32)
        return np.clip(out, self._low, self._high).astype(np.float32)


class RetreatFallback(Fallback):
    """Move the floating base AWAY from the estimated human, instead of freezing.

    Motivation (E4.1, 2026-06-01): the ``zero_velocity`` fallback *stops* the
    robot, so it DWELLS near an approaching human (episodes 449→493, proximity
    unchanged/up) — freezing doesn't increase separation when the hazard walks
    toward you. This fallback sets the absolute base X,Y target to
    ``current_base_xy + away_unit * retreat_step`` (``away_unit`` points from the
    human to the robot base), so the robot actively backs off. Non-base DOFs are
    held at the proposed action.

    Frame/index assumptions (verified against bigym):
      - ``action[0:2]`` = absolute base X,Y target (floating_dofs=[X,Y,Z,RZ] are
        the first ``dof_amount`` action entries; JointPositionActionMode absolute).
      - ``obs['proprioception_floating_base'][0:2]`` = current base world X,Y (qpos).
      - ``obs['human_pos_estimate'][0:2]`` = human pelvis world X,Y.
    All three share the world frame.

    Fail-safe: if the obs keys are missing, sizes are wrong, the values are
    non-finite, or the human is ~on top of the base (degenerate direction), it
    returns the zero-velocity action — so it is never worse than freezing.

    ``retreat_step`` (metres of base-target offset per vetoed step) defaults to
    0.10 and is overridable via the ``SVF_RETREAT_STEP`` env var (so it can be
    swept without code changes).
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        *,
        retreat_step: float = 0.10,
        base_xy_idx: tuple = (0, 1),
        human_key: str = "human_pos_estimate",
        base_key: str = "proprioception_floating_base",
    ):
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                f"RetreatFallback expects a Box action_space; got "
                f"{type(action_space).__name__}"
            )
        self._low = action_space.low.astype(np.float32)
        self._high = action_space.high.astype(np.float32)
        self._shape = action_space.shape
        self._ix, self._iy = int(base_xy_idx[0]), int(base_xy_idx[1])
        self._step = float(os.environ.get("SVF_RETREAT_STEP", retreat_step))
        self._human_key = human_key
        self._base_key = base_key
        self._logged = False

    def _zero(self) -> np.ndarray:
        return np.clip(np.zeros(self._shape, np.float32), self._low, self._high).astype(np.float32)

    def compute(self, *, obs: Mapping[str, Any], proposed: np.ndarray) -> np.ndarray:
        proposed = np.asarray(proposed, dtype=np.float32)
        human = obs.get(self._human_key) if hasattr(obs, "get") else None
        base = obs.get(self._base_key) if hasattr(obs, "get") else None
        if human is None or base is None:
            return self._zero()
        human = np.asarray(human, dtype=np.float32).ravel()
        base = np.asarray(base, dtype=np.float32).ravel()
        if human.size < 2 or base.size < 2:
            return self._zero()
        human_xy, base_xy = human[:2], base[:2]
        away = base_xy - human_xy
        d = float(np.linalg.norm(away))
        if not np.isfinite(d) or d < 1e-3:
            return self._zero()  # degenerate direction -> safest is to stop
        away_unit = away / d
        out = proposed.copy()
        out[self._ix] = base_xy[0] + away_unit[0] * self._step
        out[self._iy] = base_xy[1] + away_unit[1] * self._step
        out = np.clip(out, self._low, self._high).astype(np.float32)
        if not self._logged:
            logger.info(
                "RetreatFallback engaged: human_xy=%s base_xy=%s away_unit=%s "
                "step=%.3f -> base target=[%.3f, %.3f]. (VERIFY base_xy looks like "
                "a world position, not a joint angle — else fix base_xy_idx.)",
                human_xy, base_xy, away_unit, self._step, out[self._ix], out[self._iy],
            )
            self._logged = True
        return out


class FallbackRegistry:
    """Name → factory mapping. Phase 4 registers additional strategies here."""

    _registry: Dict[str, Callable[[gym.spaces.Box], Fallback]] = {
        "zero_velocity": ZeroVelocityFallback,
        "retreat": RetreatFallback,
    }

    @classmethod
    def build(cls, name: str, action_space: gym.spaces.Box) -> Fallback:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown fallback strategy {name!r}; known: {sorted(cls._registry)}"
            )
        return cls._registry[name](action_space)

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[[gym.spaces.Box], Fallback],
    ) -> None:
        cls._registry[name] = factory


__all__ = ["Fallback", "FallbackRegistry", "ZeroVelocityFallback", "RetreatFallback"]

"""Fallback-action strategies for the runtime safety filter.

v1 ships :class:`ZeroVelocityFallback` only — proportional damping, trajectory
replay, and retreat fallbacks are Phase-4 work. The :class:`FallbackRegistry`
gives those future strategies a drop-in slot so the wrapper code stays put.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Mapping

import gymnasium as gym
import numpy as np


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


class FallbackRegistry:
    """Name → factory mapping. Phase 4 registers additional strategies here."""

    _registry: Dict[str, Callable[[gym.spaces.Box], Fallback]] = {
        "zero_velocity": ZeroVelocityFallback,
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


__all__ = ["Fallback", "FallbackRegistry", "ZeroVelocityFallback"]

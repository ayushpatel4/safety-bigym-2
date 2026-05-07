"""Critic input contract: deterministic 1-D vector, no pixels.

The SVF is decoupled from the actor's pixel encoder. ``CriticFeatureSpec``
freezes which obs keys feed the critic (alphabetical, 1-D only) at
construction time so checkpoints stay reproducible across reloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

from gymnasium import spaces

ArrayLike = Union[np.ndarray, torch.Tensor]

# Pixel key heuristics — any key matching these substrings is dropped even if
# its space is 1-D (defensive for future additions).
_PIXEL_NAME_HINTS = ("rgb", "pixel", "cam", "image", "depth")


def _is_pixel_space(name: str, space: spaces.Space) -> bool:
    if any(hint in name.lower() for hint in _PIXEL_NAME_HINTS):
        return True
    if isinstance(space, spaces.Box):
        return len(space.shape) > 1
    return True  # anything non-Box (Discrete, Tuple, ...) is unsupported here


def _flatten_last(arr: ArrayLike) -> torch.Tensor:
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(arr))
    else:
        t = arr
    t = t.to(dtype=torch.float32)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    return t


@dataclass(frozen=True)
class CriticFeatureSpec:
    """Frozen description of the critic's input layout."""

    obs_keys: Tuple[str, ...]
    obs_dims: Tuple[int, ...]
    action_dim: int

    @property
    def input_dim(self) -> int:
        return sum(self.obs_dims) + self.action_dim

    @classmethod
    def from_spaces(
        cls,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
    ) -> "CriticFeatureSpec":
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                "CriticFeatureSpec requires a Dict observation_space; got "
                f"{type(observation_space).__name__}"
            )
        kept: list[Tuple[str, int]] = []
        for name in sorted(observation_space.spaces.keys()):
            sp = observation_space.spaces[name]
            if _is_pixel_space(name, sp):
                continue
            assert isinstance(sp, spaces.Box)
            kept.append((name, int(np.prod(sp.shape))))
        if not kept:
            raise ValueError(
                "No 1-D non-pixel obs keys found — critic has no input. "
                f"Saw keys: {list(observation_space.spaces)}"
            )
        if not isinstance(action_space, spaces.Box):
            raise TypeError(
                "CriticFeatureSpec expects a Box action_space; got "
                f"{type(action_space).__name__}"
            )
        action_dim = int(np.prod(action_space.shape))
        keys, dims = zip(*kept)
        return cls(obs_keys=tuple(keys), obs_dims=tuple(dims), action_dim=action_dim)

    @classmethod
    def from_env(cls, env: gym.Env) -> "CriticFeatureSpec":
        return cls.from_spaces(env.observation_space, env.action_space)

    def to_dict(self) -> dict:
        return {
            "obs_keys": list(self.obs_keys),
            "obs_dims": list(self.obs_dims),
            "action_dim": self.action_dim,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "CriticFeatureSpec":
        return cls(
            obs_keys=tuple(payload["obs_keys"]),
            obs_dims=tuple(int(d) for d in payload["obs_dims"]),
            action_dim=int(payload["action_dim"]),
        )


def make_critic_input(
    obs: Mapping[str, ArrayLike],
    action: ArrayLike,
    spec: CriticFeatureSpec,
) -> torch.Tensor:
    """Build the critic input tensor by concatenating ``spec.obs_keys`` then action.

    Supports both single-sample (1-D arrays) and batched (2-D, batch first)
    inputs, mixed numpy / torch.
    """
    pieces: list[torch.Tensor] = []
    has_batch = False
    for key in spec.obs_keys:
        if key not in obs:
            raise KeyError(
                f"Obs key {key!r} expected by CriticFeatureSpec is missing at runtime"
            )
        t = _flatten_last(obs[key])
        if t.ndim == 1:
            pieces.append(t)
        else:
            has_batch = True
            pieces.append(t.reshape(t.shape[0], -1))

    a = _flatten_last(action)
    if a.ndim == 1:
        pieces.append(a)
    else:
        has_batch = True
        pieces.append(a.reshape(a.shape[0], -1))

    # Promote any 1-D piece to 2-D if any piece is batched.
    if has_batch:
        batch_size = next(p.shape[0] for p in pieces if p.ndim == 2)
        pieces = [
            p if p.ndim == 2 else p.unsqueeze(0).expand(batch_size, -1)
            for p in pieces
        ]
    return torch.cat(pieces, dim=-1).contiguous()


__all__ = ["CriticFeatureSpec", "make_critic_input"]

"""Attach the Phase-2 SVF safety filter to either runner path.

Gym path (random / ACT): wrap with the real :class:`SafetyFilterWrapper` (outermost, so
``EpisodeSafetyMetrics`` aggregates the *executed* action — see its docstring).

CQN-AS path: the agent proposes a normalised action on the adapter's processed obs and the
adapter owns the raw gym env, so a gym-level wrapper can't see the obs the critic needs.
:class:`ObsCacheWrapper` caches the raw post-BodySLAM obs dict by a pure attribute swap of
``adapter._env`` (no adapter edit); the runner then applies the veto in-loop via
:func:`safety_bigym.benchmark.runners.apply_veto`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym

__all__ = [
    "load_critic",
    "assert_critic_covers_obs",
    "attach_filter_gym",
    "ObsCacheWrapper",
]


def load_critic(path: Path):
    """Load a frozen :class:`SafetyCritic` from a checkpoint payload."""
    import torch

    from safety_bigym.filters.critic import SafetyCritic

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    return SafetyCritic.from_checkpoint_payload(payload)


def assert_critic_covers_obs(critic, obs_keys: Iterable[str]) -> None:
    """Fail loud if the critic needs obs keys the env doesn't emit.

    The SVF critic's spec includes ``human_pos_estimate`` — so filtering with obs-mode
    ``off`` is a configuration error caught here instead of a cryptic KeyError deep in
    ``make_critic_input``.
    """
    have = set(obs_keys)
    missing = [k for k in critic.spec.obs_keys if k not in have]
    if missing:
        raise KeyError(
            f"Filter critic requires obs keys absent from the env: {missing}. "
            f"Run with --obs-mode oracle|noisy so 'human_pos_estimate' is present "
            f"(env emits: {sorted(have)})."
        )


def attach_filter_gym(env: gym.Env, *, critic, threshold_R: float, fallback_name: str = "zero_velocity"):
    """Wrap a gym env with :class:`SafetyFilterWrapper` (outermost)."""
    from safety_bigym.filters.fallback import FallbackRegistry
    from safety_bigym.filters.runtime_wrapper import SafetyFilterWrapper

    obs_space = env.observation_space
    keys = obs_space.spaces.keys() if isinstance(obs_space, gym.spaces.Dict) else ()
    assert_critic_covers_obs(critic, keys)
    fallback = FallbackRegistry.build(fallback_name, env.action_space)
    return SafetyFilterWrapper(env, critic=critic, fallback=fallback, threshold_R=threshold_R)


class ObsCacheWrapper(gym.Wrapper):
    """Cache the last emitted obs dict on ``.last_obs`` (for the CQN-AS in-loop veto)."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, terminated, truncated, info

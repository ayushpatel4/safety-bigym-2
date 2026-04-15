"""Tests for EpisodeSafetyMetrics wrapper.

The wrapper aggregates per-step info["safety"] into per-episode scalars
and emits them under info["episode_safety"] at terminated/truncated=True.
"""

import numpy as np
import pytest
from gymnasium import spaces
import gymnasium as gym


class _StubSafetyEnv(gym.Env):
    """Minimal env that emits scripted info['safety'] payloads per step."""

    metadata: dict = {}

    def __init__(self, scripted_infos):
        self._infos = list(scripted_infos)
        self._t = 0
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        safety = self._infos[self._t]
        self._t += 1
        terminated = self._t >= len(self._infos)
        return (
            np.zeros((1,), dtype=np.float32),
            0.0,
            terminated,
            False,
            {"safety": safety},
        )


def _safety(
    ssm_v=False, pfl_v=False, margin=1.0, ratio=0.0, force=0.0, region=None
):
    return {
        "ssm_violation": ssm_v,
        "pfl_violation": pfl_v,
        "ssm_margin": margin,
        "pfl_force_ratio": ratio,
        "min_separation": 1.0,
        "max_contact_force": force,
        "contact_region": "" if region is None else region,
        "contact_type": "",
        "violations_by_region": ({region: 1} if region else {}),
        "robot_pos": [0.0, 0.0, 0.0],
        "human_pos": [1.0, 0.0, 0.0],
        "closest_human_joint": "",
        "closest_robot_link": "",
    }


def _make_wrapped(scripted):
    from safety_bigym.safety.episode_metrics_wrapper import EpisodeSafetyMetrics

    return EpisodeSafetyMetrics(_StubSafetyEnv(scripted))


def test_no_episode_safety_until_done():
    scripted = [_safety(), _safety(), _safety()]
    env = _make_wrapped(scripted)
    env.reset()
    _, _, done, trunc, info = env.step(env.action_space.sample())
    assert not done and not trunc
    assert "episode_safety" not in info


def test_episode_safety_emitted_on_terminated():
    scripted = [_safety(margin=0.8), _safety(margin=0.3, ssm_v=True)]
    env = _make_wrapped(scripted)
    env.reset()
    for i in range(len(scripted)):
        _, _, done, trunc, info = env.step(env.action_space.sample())
    assert done
    assert "episode_safety" in info
    ep = info["episode_safety"]
    assert ep["ep_ssm_violation_rate"] == pytest.approx(0.5)
    assert ep["ep_min_ssm_margin"] == pytest.approx(0.3)


def test_max_ratio_and_force():
    scripted = [
        _safety(ratio=0.4, force=10.0),
        _safety(ratio=1.2, force=40.0, pfl_v=True),
        _safety(ratio=0.9, force=20.0),
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    assert done
    ep = info["episode_safety"]
    assert ep["ep_max_pfl_force_ratio"] == pytest.approx(1.2)
    assert ep["ep_max_contact_force"] == pytest.approx(40.0)
    assert ep["ep_pfl_violation_rate"] == pytest.approx(1 / 3)


def test_time_to_first_violation_reports_step_index():
    scripted = [
        _safety(),
        _safety(),
        _safety(ssm_v=True, margin=0.1),
        _safety(),
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    assert done
    ep = info["episode_safety"]
    assert ep["ep_time_to_first_violation"] == 2


def test_time_to_first_violation_none_when_clean():
    scripted = [_safety(), _safety()]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    ep = info["episode_safety"]
    assert ep["ep_time_to_first_violation"] == -1


def test_region_counts_aggregated():
    scripted = [
        _safety(pfl_v=True, region="hand"),
        _safety(pfl_v=True, region="hand"),
        _safety(pfl_v=True, region="chest"),
    ]
    env = _make_wrapped(scripted)
    env.reset()
    for _ in scripted:
        _, _, done, trunc, info = env.step(env.action_space.sample())
    ep = info["episode_safety"]
    assert ep["ep_region_hand"] == 2
    assert ep["ep_region_chest"] == 1


def test_reset_clears_state_between_episodes():
    from safety_bigym.safety.episode_metrics_wrapper import EpisodeSafetyMetrics

    scripted_a = [_safety(ssm_v=True, margin=0.1)]
    env = EpisodeSafetyMetrics(_StubSafetyEnv(scripted_a))
    env.reset()
    _, _, done, _, info_a = env.step(env.action_space.sample())
    assert done
    assert info_a["episode_safety"]["ep_ssm_violation_rate"] == pytest.approx(1.0)

    scripted_b = [_safety(margin=0.9)]
    env.env._infos = scripted_b
    env.reset()
    _, _, done, _, info_b = env.step(env.action_space.sample())
    assert done
    assert info_b["episode_safety"]["ep_ssm_violation_rate"] == pytest.approx(0.0)
    assert info_b["episode_safety"]["ep_min_ssm_margin"] == pytest.approx(0.9)

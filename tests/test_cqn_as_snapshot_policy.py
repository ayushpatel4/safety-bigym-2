"""Unit tests for the CQN-AS snapshot policy wrapper in svf_collect_dataset.py.

CQN-AS snapshots (train_cqn_as.py) differ from RoboBase ACT/DP ones: the
vendored CQNASAgent's ``act`` takes split ``(rgb_obs, low_dim_obs)`` arrays,
the obs are frame-stacked, and actions are normalised to ``[-1, 1]``. The
collector's :class:`_CQNASSnapshotPolicy` must replicate
``SafetyBiGymCQNAdapter._extract_obs`` (state-key concat + optional
human_pos_estimate + per-camera frame-stacked rgb) and
``_convert_action_to_raw`` (denorm to env range).

These tests pin, with a stub agent (no MuJoCo / no torch model):
- low_dim assembly width = (sum state widths [+6 human]) * frame_stack;
- human_pos_estimate gating on includes_human_pos;
- rgb assembly shape = (num_cameras, C * frame_stack, H, W);
- action denorm: norm 0 -> midpoint, +1 -> max, -1 -> min;
- reset() clears the per-episode frame deques.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _import_script():
    return importlib.import_module("svf_collect_dataset")


_STATE_KEYS = ("proprioception", "proprioception_grippers", "proprioception_floating_base")
_STATE_WIDTHS = {"proprioception": 3, "proprioception_grippers": 2, "proprioception_floating_base": 4}
_HUMAN_W = 6
_FRAME_STACK = 4
_ACTION_DIM = 4
_ACTION_SEQ = 2
_H = _W = 8


class _StubAgent:
    """Records the shapes act() received; returns a fixed flat action chunk."""

    def __init__(self, norm_value: float):
        # Flat chunk of length action_sequence * action_dim, all == norm_value.
        self._flat = np.full(_ACTION_SEQ * _ACTION_DIM, norm_value, dtype=np.float32)
        self.last_rgb_shape = None
        self.last_low_dim_shape = None

    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        self.last_rgb_shape = np.asarray(rgb_obs).shape
        self.last_low_dim_shape = np.asarray(low_dim_obs).shape
        assert eval_mode is True
        return self._flat


def _make_policy(*, includes_human_pos, camera_keys, agent):
    mod = _import_script()
    return mod._CQNASSnapshotPolicy(
        agent=agent,
        state_keys=_STATE_KEYS,
        includes_human_pos=includes_human_pos,
        camera_keys=camera_keys,
        frame_stack=_FRAME_STACK,
        action_sequence=_ACTION_SEQ,
        action_low=np.full(_ACTION_DIM, -np.pi, dtype=np.float32),
        action_high=np.full(_ACTION_DIM, np.pi, dtype=np.float32),
        rgb_placeholder_shape=(len(camera_keys) or 1, 3 * _FRAME_STACK, _H, _W),
    )


def _fake_obs(camera_keys):
    obs = {k: np.ones(_STATE_WIDTHS[k], np.float32) for k in _STATE_KEYS}
    obs["human_pos_estimate"] = np.full(_HUMAN_W, 0.5, np.float32)
    for cam in camera_keys:
        obs[f"rgb_{cam}"] = np.zeros((_H, _W, 3), np.uint8)
    return obs


def test_low_dim_width_includes_human_and_frame_stack():
    agent = _StubAgent(norm_value=0.0)
    policy = _make_policy(includes_human_pos=True, camera_keys=("head",), agent=agent)
    policy(_fake_obs(("head",)))
    expected = (sum(_STATE_WIDTHS.values()) + _HUMAN_W) * _FRAME_STACK
    assert agent.last_low_dim_shape == (expected,)


def test_low_dim_width_excludes_human_when_gated_off():
    agent = _StubAgent(norm_value=0.0)
    policy = _make_policy(includes_human_pos=False, camera_keys=("head",), agent=agent)
    policy(_fake_obs(("head",)))
    expected = sum(_STATE_WIDTHS.values()) * _FRAME_STACK
    assert agent.last_low_dim_shape == (expected,)


def test_rgb_assembly_shape():
    cams = ("head", "right_wrist")
    agent = _StubAgent(norm_value=0.0)
    policy = _make_policy(includes_human_pos=True, camera_keys=cams, agent=agent)
    policy(_fake_obs(cams))
    # (num_cameras, C * frame_stack, H, W)
    assert agent.last_rgb_shape == (len(cams), 3 * _FRAME_STACK, _H, _W)


def test_action_denorm_midpoint_min_max():
    cams = ("head",)
    # norm 0 -> midpoint of [-pi, pi] = 0
    p0 = _make_policy(includes_human_pos=True, camera_keys=cams, agent=_StubAgent(0.0))
    np.testing.assert_allclose(p0(_fake_obs(cams)), np.zeros(_ACTION_DIM), atol=1e-5)
    # norm +1 -> max (+pi)
    pp = _make_policy(includes_human_pos=True, camera_keys=cams, agent=_StubAgent(1.0))
    np.testing.assert_allclose(pp(_fake_obs(cams)), np.full(_ACTION_DIM, np.pi), atol=1e-4)
    # norm -1 -> min (-pi)
    pm = _make_policy(includes_human_pos=True, camera_keys=cams, agent=_StubAgent(-1.0))
    np.testing.assert_allclose(pm(_fake_obs(cams)), np.full(_ACTION_DIM, -np.pi), atol=1e-4)


def test_reset_clears_frame_deques():
    cams = ("head",)
    agent = _StubAgent(norm_value=0.0)
    policy = _make_policy(includes_human_pos=True, camera_keys=cams, agent=agent)
    # Step a few times to fill/extend the deques.
    obs = _fake_obs(cams)
    for _ in range(3):
        policy(obs)
    assert len(policy._low_dim_frames) == _FRAME_STACK
    policy.reset()
    assert len(policy._low_dim_frames) == 0
    assert all(len(dq) == 0 for dq in policy._rgb_frames.values())
    # After reset, the next call re-primes the stack to full width.
    policy(obs)
    assert len(policy._low_dim_frames) == _FRAME_STACK


def test_no_camera_uses_placeholder_shape():
    agent = _StubAgent(norm_value=0.0)
    policy = _make_policy(includes_human_pos=False, camera_keys=(), agent=agent)
    policy(_fake_obs(()))
    # placeholder shape passed in _make_policy: (1, 3*frame_stack, H, W)
    assert agent.last_rgb_shape == (1, 3 * _FRAME_STACK, _H, _W)

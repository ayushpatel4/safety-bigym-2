"""Tests for the camera-correct snapshot policy adapter.

The adapter (``_SnapshotPolicy.adapt_obs``) must turn bare-env HWC uint8
images into the ``(B, T, C, H, W)`` torch tensors ACT's ``act()`` expects.
Pixel keys must be read in the order declared on the policy so the actor's
``extract_many_from_batch(obs, r"rgb.*")`` sees a stable layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _import_script():
    import importlib

    return importlib.import_module("svf_collect_dataset")


def _hwc_uint8(seed: int, h: int = 84, w: int = 84) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _make_policy(cameras=("head",), resolution=(84, 84), includes_human_pos=False):
    """Build a _SnapshotPolicy with a dummy agent (we only test adapt_obs)."""
    mod = _import_script()
    return mod._SnapshotPolicy(
        agent=None,
        cameras=tuple(cameras),
        camera_resolution=tuple(resolution),
        includes_human_pos=includes_human_pos,
    )


def test_adapter_output_shape_single_camera():
    pol = _make_policy(cameras=("head",))
    obs = {
        "proprioception": np.zeros(8, dtype=np.float32),
        "rgb_head": _hwc_uint8(0),
    }
    out = pol.adapt_obs(obs)
    assert "low_dim_state" in out
    assert out["low_dim_state"].shape == (1, 1, 8)
    assert "rgb_head" in out
    assert out["rgb_head"].shape == (1, 1, 3, 84, 84)


def test_adapter_output_shape_three_cameras():
    """ACT's encoder was trained on 3 views — adapter must forward all three."""
    pol = _make_policy(cameras=("head", "right_wrist", "left_wrist"))
    obs = {
        "proprioception": np.zeros(8, dtype=np.float32),
        "rgb_head":        _hwc_uint8(0),
        "rgb_right_wrist": _hwc_uint8(1),
        "rgb_left_wrist":  _hwc_uint8(2),
    }
    out = pol.adapt_obs(obs)
    for cam in ("head", "right_wrist", "left_wrist"):
        key = f"rgb_{cam}"
        assert key in out, f"adapter dropped {key}"
        assert out[key].shape == (1, 1, 3, 84, 84)
        assert out[key].dtype == torch.uint8, (
            f"{key} should pass through uint8 (encoder handles /255 + normalize)"
        )


def test_adapter_preserves_pixel_values_post_transpose():
    """Adapter must permute HWC→CHW without corrupting pixel values."""
    pol = _make_policy(cameras=("head",))
    hwc = _hwc_uint8(42)
    obs = {"proprioception": np.zeros(4, np.float32), "rgb_head": hwc}
    out = pol.adapt_obs(obs)
    chw_tensor = out["rgb_head"][0, 0]  # drop batch + time
    # Should equal the HWC array's manual transpose
    expected = torch.from_numpy(np.transpose(hwc, (2, 0, 1)))
    assert torch.equal(chw_tensor, expected)


def test_adapter_handles_extra_cameras_in_obs():
    """Env may emit more cameras than the policy expects — forward only the
    declared ones, don't error."""
    pol = _make_policy(cameras=("head",))
    obs = {
        "proprioception": np.zeros(4, np.float32),
        "rgb_head": _hwc_uint8(0),
        "rgb_extra_unused": _hwc_uint8(1),
    }
    out = pol.adapt_obs(obs)
    assert "rgb_head" in out
    assert "rgb_extra_unused" not in out


def test_adapter_missing_camera_raises():
    pol = _make_policy(cameras=("head", "right_wrist"))
    obs = {
        "proprioception": np.zeros(4, np.float32),
        "rgb_head": _hwc_uint8(0),
        # rgb_right_wrist missing
    }
    with pytest.raises(KeyError, match="rgb_right_wrist"):
        pol.adapt_obs(obs)


def test_adapter_accepts_chw_uint8_too():
    """If something upstream already permuted to CHW, the adapter should
    pass through rather than re-transpose."""
    pol = _make_policy(cameras=("head",))
    chw = np.transpose(_hwc_uint8(7), (2, 0, 1))  # (3, 84, 84)
    obs = {"proprioception": np.zeros(4, np.float32), "rgb_head": chw}
    out = pol.adapt_obs(obs)
    assert out["rgb_head"].shape == (1, 1, 3, 84, 84)
    assert torch.equal(out["rgb_head"][0, 0], torch.from_numpy(chw))


def test_adapter_rejects_unknown_pixel_shape():
    pol = _make_policy(cameras=("head",))
    obs = {
        "proprioception": np.zeros(4, np.float32),
        "rgb_head": np.zeros((84, 84), dtype=np.uint8),  # missing channel axis
    }
    with pytest.raises(ValueError, match="Unexpected pixel shape"):
        pol.adapt_obs(obs)


def test_adapter_no_pixels_when_cameras_empty():
    """No-pixel snapshot (cameras=()) yields a low_dim-only obs dict —
    no rgb_* keys forwarded."""
    pol = _make_policy(cameras=())
    obs = {
        "proprioception": np.zeros(4, np.float32),
        "rgb_head": _hwc_uint8(0),
    }
    out = pol.adapt_obs(obs)
    assert "low_dim_state" in out
    assert not any(k.startswith("rgb") for k in out)


def test_expects_pixels_property():
    assert _make_policy(cameras=("head",)).expects_pixels is True
    assert _make_policy(cameras=()).expects_pixels is False


# ---- low_dim_state composition: ConcatDim parity --------------------------


def test_low_dim_excludes_proprioception_floating_base_actions():
    """ConcatDim training-time excludes `proprioception_floating_base_actions`
    (robobase/envs/bigym.py:95). The adapter must do the same."""
    pol = _make_policy(cameras=())
    obs = {
        "proprioception": np.arange(8, dtype=np.float32),
        "proprioception_grippers": np.arange(2, dtype=np.float32) + 100,
        "proprioception_floating_base": np.arange(3, dtype=np.float32) + 200,
        "proprioception_floating_base_actions": np.arange(3, dtype=np.float32) + 999,
    }
    out = pol.adapt_obs(obs)
    flat = out["low_dim_state"].numpy().reshape(-1)
    # length = 8 + 2 + 3 = 13 (no actions)
    assert flat.shape == (13,)
    # The 999-valued floating_base_actions must NOT appear anywhere
    assert (flat >= 990).sum() == 0, f"floating_base_actions leaked: {flat}"


def test_low_dim_preserves_obs_insertion_order():
    """ConcatDim iterates obs in insertion order. The adapter must do the same."""
    pol = _make_policy(cameras=())
    # Construct an obs dict where insertion order is non-alphabetical
    obs = {}
    obs["proprioception_grippers"] = np.array([10.0, 20.0], np.float32)
    obs["proprioception"] = np.array([1.0, 2.0, 3.0], np.float32)
    obs["proprioception_floating_base"] = np.array([100.0, 200.0, 300.0], np.float32)
    out = pol.adapt_obs(obs)
    flat = out["low_dim_state"].numpy().reshape(-1)
    # Expect grippers (10, 20), then proprioception (1, 2, 3), then floating_base (100, 200, 300)
    expected = np.array([10, 20, 1, 2, 3, 100, 200, 300], dtype=np.float32)
    assert np.allclose(flat, expected), f"order broken: got {flat}, expected {expected}"


def test_low_dim_includes_human_pos_for_phase1_snapshots():
    """Phase 1 ACT trained with BodySLAMWrapper has human_pos_estimate in
    low_dim_state. The adapter must include it when includes_human_pos=True."""
    pol = _make_policy(cameras=(), includes_human_pos=True)
    obs = {}
    obs["proprioception"] = np.zeros(4, np.float32)
    obs["proprioception_grippers"] = np.zeros(2, np.float32)
    obs["human_pos_estimate"] = np.arange(6, dtype=np.float32) + 50
    out = pol.adapt_obs(obs)
    flat = out["low_dim_state"].numpy().reshape(-1)
    assert flat.shape == (4 + 2 + 6,), f"expected dim 12, got {flat.shape}"
    # human_pos_estimate values (50..55) appear at the end
    assert np.allclose(flat[-6:], np.arange(6) + 50)


def test_low_dim_omits_human_pos_for_phase0_snapshots():
    """Phase 0 ACT trained without BodySLAMWrapper. If our env still emits
    human_pos_estimate (bodyslam mode mismatch upstream), the adapter must
    still skip it so the actor's first layer gets a Phase-0-shape vector."""
    pol = _make_policy(cameras=(), includes_human_pos=False)
    obs = {}
    obs["proprioception"] = np.zeros(4, np.float32)
    obs["proprioception_grippers"] = np.zeros(2, np.float32)
    obs["human_pos_estimate"] = np.arange(6, dtype=np.float32) + 50
    out = pol.adapt_obs(obs)
    flat = out["low_dim_state"].numpy().reshape(-1)
    assert flat.shape == (4 + 2,), f"expected dim 6, got {flat.shape}"


def test_low_dim_pixel_keys_dropped_from_low_dim():
    """Pixel keys (multi-D) must never leak into low_dim_state."""
    pol = _make_policy(cameras=("head",))
    obs = {
        "proprioception": np.zeros(4, np.float32),
        "rgb_head": _hwc_uint8(0),
    }
    out = pol.adapt_obs(obs)
    assert out["low_dim_state"].numpy().reshape(-1).shape == (4,)


def test_includes_human_pos_default_is_false():
    """Backwards-compat: the default policy treats snapshots as Phase 0."""
    mod = _import_script()
    pol = mod._SnapshotPolicy(agent=None)
    assert pol.includes_human_pos is False

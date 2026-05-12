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


def _make_policy(cameras=("head",), resolution=(84, 84)):
    """Build a _SnapshotPolicy with a dummy agent (we only test adapt_obs)."""
    mod = _import_script()
    return mod._SnapshotPolicy(
        agent=None,
        cameras=tuple(cameras),
        camera_resolution=tuple(resolution),
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

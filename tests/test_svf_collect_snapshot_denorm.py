"""B5.5 regression — _SnapshotPolicy tanh-denormalization.

The v1 SVF collection emitted raw tanh-space actions from snapshot policies
(B4.2 caveat in docs/phase2_results.md): gripper dims sat at -1.1 (env
silently clipped to [0, 1]), body-joint dims stayed in [-1, 1] rather than
spanning the env's ±π range. RoboBase deploys the policy under
``RescaleFromTanhWithMinMax``; B5.5 replicates that wrap inside
``_SnapshotPolicy.__call__`` when the snapshot payload carries
``action_stats``.

These tests pin:
- raw tanh-space output when ``action_stats`` is None (legacy path, with a
  one-shot warning logged);
- the actions land inside ``[min, max]`` componentwise when ``action_stats``
  is set;
- a saturated tanh input (+1, -1) hits exactly ``max``/``min``;
- ``min_max_margin`` widens the output range proportionally.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _import_script():
    return importlib.import_module("svf_collect_dataset")


class _StubAgent:
    """Returns a fixed tanh-space action whenever ``act`` is called.

    Shape mirrors ACT's chunked output ``(chunk, action_dim)``; the policy
    flattens this to the first row before any rescaling.
    """

    def __init__(self, action: np.ndarray, chunk: int = 1):
        self._action = np.asarray(action, dtype=np.float32)
        self._chunk = chunk

    def act(self, obs, step: int, eval_mode: bool):
        # tile to (chunk, action_dim) like ACT
        chunked = np.tile(self._action[None, :], (self._chunk, 1))
        return torch.from_numpy(chunked)


def _make_policy(action: np.ndarray, *, action_stats=None, min_max_margin: float = 0.0):
    mod = _import_script()
    return mod._SnapshotPolicy(
        agent=_StubAgent(action, chunk=4),
        cameras=(),
        camera_resolution=(84, 84),
        includes_human_pos=False,
        action_stats=action_stats,
        min_max_margin=min_max_margin,
    )


def _trivial_obs(dim: int = 4) -> dict:
    return {"proprioception": np.zeros(dim, dtype=np.float32)}


# -------------------------------------------------------------------- legacy --


def test_no_action_stats_returns_raw_tanh(caplog):
    """v1 behaviour: action_stats=None ⇒ raw tanh-space passthrough."""
    mod = _import_script()
    # Force-reset the warn-once latch so this test always sees the warning.
    mod._RAW_TANH_WARNED = False
    pol = _make_policy(np.array([-1.1, 0.0, 0.5], dtype=np.float32))
    with caplog.at_level(logging.WARNING, logger="svf_collect_dataset"):
        action = pol(_trivial_obs())
    np.testing.assert_allclose(action, [-1.1, 0.0, 0.5], atol=1e-6)
    assert action.dtype == np.float32
    assert any("action_stats" in rec.message for rec in caplog.records), (
        "expected one-shot warning about missing action_stats"
    )


def test_no_action_stats_warns_only_once(caplog):
    mod = _import_script()
    mod._RAW_TANH_WARNED = False
    pol = _make_policy(np.zeros(3, dtype=np.float32))
    with caplog.at_level(logging.WARNING, logger="svf_collect_dataset"):
        pol(_trivial_obs())
        pol(_trivial_obs())
        pol(_trivial_obs())
    warnings = [r for r in caplog.records if "action_stats" in r.message]
    assert len(warnings) == 1, f"expected 1 warning, got {len(warnings)}"


# ---------------------------------------------------------------- denorm path --


def _stats(lo, hi):
    return {"min": np.asarray(lo, dtype=np.float32),
            "max": np.asarray(hi, dtype=np.float32)}


def test_action_stats_maps_to_env_range():
    """Mid-tanh-range (0.0) maps to the midpoint of [min, max]."""
    stats = _stats([-3.14, -3.14, 0.0], [3.14, 3.14, 1.0])
    pol = _make_policy(np.zeros(3, dtype=np.float32), action_stats=stats)
    action = pol(_trivial_obs())
    np.testing.assert_allclose(action, [0.0, 0.0, 0.5], atol=1e-6)


def test_action_stats_saturated_tanh_hits_extremes():
    """+1 → max, -1 → min, exactly."""
    stats = _stats([-3.14, -3.14, 0.0], [3.14, 3.14, 1.0])

    pol_hi = _make_policy(np.array([1.0, 1.0, 1.0], dtype=np.float32), action_stats=stats)
    np.testing.assert_allclose(pol_hi(_trivial_obs()), stats["max"], atol=1e-6)

    pol_lo = _make_policy(np.array([-1.0, -1.0, -1.0], dtype=np.float32), action_stats=stats)
    np.testing.assert_allclose(pol_lo(_trivial_obs()), stats["min"], atol=1e-6)


def test_action_stats_clips_overshoot():
    """RescaleFromTanhWithMinMax.transform_from_tanh clips inputs to [-1, 1]
    before mapping. A -1.1 tanh output (the B4.2 gripper bug) must land at
    the env minimum, not below it."""
    stats = _stats([-3.14, -3.14, 0.0], [3.14, 3.14, 1.0])
    pol = _make_policy(np.array([-1.1, 1.5, -0.2], dtype=np.float32), action_stats=stats)
    action = pol(_trivial_obs())
    # -1.1 clipped to -1 ⇒ min; 1.5 clipped to 1 ⇒ max; -0.2 ⇒ interior
    np.testing.assert_allclose(action[:2], [stats["min"][0], stats["max"][1]], atol=1e-6)
    assert stats["min"][2] <= action[2] <= stats["max"][2]


def test_min_max_margin_widens_output_range():
    """A non-zero margin pushes saturated-tanh outputs past the raw min/max."""
    stats = _stats([-2.0, -2.0], [2.0, 2.0])
    pol = _make_policy(
        np.array([1.0, -1.0], dtype=np.float32),
        action_stats=stats,
        min_max_margin=0.1,
    )
    action = pol(_trivial_obs())
    # margin=0.1 ⇒ effective max = 2.0 + |2.0|*0.1 = 2.2; min = -2.2
    np.testing.assert_allclose(action, [2.2, -2.2], atol=1e-6)


def test_action_stats_preserves_float32_dtype():
    stats = _stats([-1.0], [1.0])
    pol = _make_policy(np.array([0.5], dtype=np.float32), action_stats=stats)
    action = pol(_trivial_obs())
    assert action.dtype == np.float32


def test_action_stats_supports_phase0_h1_action_dim_16():
    """End-to-end shape check: H1 + 4-dof base ⇒ action_dim=16 (the env the
    snapshot policies were trained against)."""
    body_min = np.full(14, -3.14, dtype=np.float32)
    body_max = np.full(14, 3.14, dtype=np.float32)
    grip = np.array([0.0, 0.0], dtype=np.float32)
    stats = {
        "min": np.concatenate([body_min, grip]),
        "max": np.concatenate([body_max, grip + 1]),
    }
    pol = _make_policy(np.zeros(16, dtype=np.float32), action_stats=stats)
    action = pol(_trivial_obs())
    assert action.shape == (16,)
    # Body-joint slice straddles zero (midpoint of ±π).
    np.testing.assert_allclose(action[:14], np.zeros(14), atol=1e-6)
    # Gripper slice at mid of [0, 1].
    np.testing.assert_allclose(action[14:], [0.5, 0.5], atol=1e-6)


# ---------------------------------------------------- precondition guard ------


def test_b55_precondition_guard_passes_on_current_source():
    """The shell script (run_phase2_b55.sh) greps the policy source for both
    sentinels before launching. If either disappears the abort message will
    fire — and so will this test, immediately."""
    import inspect

    mod = _import_script()
    src = inspect.getsource(mod._SnapshotPolicy)
    assert "action_stats" in src, "B5.5 patch lost the action_stats field"
    assert "transform_from_tanh" in src, (
        "B5.5 patch lost the RescaleFromTanhWithMinMax call"
    )

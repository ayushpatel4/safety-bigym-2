"""Regression test for ``train_cqn_as.Workspace._log``.

The bug (flagged 2026-05-23 in IMPLEMENTATION_STATUS.md "Outstanding gap"):
``agent.update()`` returns a ``TensorDict`` built from 0-d ``.detach()`` tensors
so its ``batch_size == torch.Size([])`` and ``len(td) == 0`` even when 4+ keys
are present. The previous ``if len(metrics) == 0: return`` check silently
swallowed every per-update train log line (``q_critic_loss``, ``bc_fosd_loss``,
``batch_reward``...).

This test does NOT import the full ``train_cqn_as`` module (it triggers a
``MUJOCO_GL=egl`` runtime error on Macs without EGL). Instead it imports just
the ``Workspace._log`` unbound method.
"""

from __future__ import annotations

import io
import logging
import os

import pytest


@pytest.fixture
def _log_capture():
    """Capture logger output for the duration of one test."""
    # Mac dev box: train_cqn_as.py sets MUJOCO_GL=egl at import time, which
    # mujoco rejects without an EGL-capable GPU. Force glfw so the import
    # succeeds; the test never actually instantiates a renderer.
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from train_cqn_as import Workspace, logger

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    prior_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield Workspace, stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)


class _FakeEmptyBatchTD:
    """Mimics tensordict 0.6.0 TensorDict with empty batch_size.

    Reproduces the exact two behaviours that triggered the regression:
      1. ``len(td)`` returns the batch-size first dim, which is 0 for
         empty-batch TDs (the case for ``agent.update()`` metrics).
      2. ``items()`` returns a single-use generator (exhausted by the first
         consumer).
    """

    def __init__(self, **kw):
        self._d = dict(kw)

    def __len__(self):
        return 0

    def __bool__(self):
        raise RuntimeError("TensorDict refuses bool conversion")

    def items(self):
        return iter(self._d.items())


class _StubWorkspace:
    _wandb_run = None

    # _log now also appends to a per-run metrics.jsonl via this helper. The
    # stub doesn't have a work_dir, so a no-op keeps the unbound-method call
    # contract intact.
    def _append_metrics_jsonl(self, *_args, **_kwargs):
        return None


def test_empty_batch_tensordict_metrics_are_logged(_log_capture):
    """Regression: per-update train metrics with empty TD batch_size must log."""
    Workspace, stream = _log_capture
    Workspace._log(
        _StubWorkspace(),
        _FakeEmptyBatchTD(
            q_critic_loss=0.5,
            bc_fosd_loss=1.2,
            bc_margin_loss=0.1,
            batch_reward=0.01,
        ),
        step=100,
        ty="train",
    )
    out = stream.getvalue()
    assert "[train] step=100" in out
    assert "q_critic_loss=0.5000" in out
    assert "bc_fosd_loss=1.2000" in out
    assert "bc_margin_loss=0.1000" in out
    assert "batch_reward=0.0100" in out


def test_plain_dict_still_works(_log_capture):
    """Episode-end and safety payloads (plain dicts) must continue to log."""
    Workspace, stream = _log_capture
    Workspace._log(
        _StubWorkspace(),
        {"episode_reward": -7.5, "episode_length": 278},
        step=200,
        ty="train",
    )
    out = stream.getvalue()
    assert "[train] step=200" in out
    assert "episode_reward=-7.5000" in out
    assert "episode_length=278" in out


def test_none_metrics_are_silent(_log_capture):
    """None must not produce a log line."""
    Workspace, stream = _log_capture
    Workspace._log(_StubWorkspace(), None, step=300)
    assert "step=300" not in stream.getvalue()


def test_empty_dict_is_silent(_log_capture):
    """An empty {} must not produce a log line."""
    Workspace, stream = _log_capture
    Workspace._log(_StubWorkspace(), {}, step=400, ty="train")
    assert "step=400" not in stream.getvalue()


# ---------------------------------------------------------------------------
# step_marks_task_success — clean task-quality signal under workspace shaping
# ---------------------------------------------------------------------------


def _ts(info):
    """Build a minimal time_step-like object with .info."""
    from types import SimpleNamespace
    return SimpleNamespace(info=info)


def test_step_marks_task_success_fires_on_success_info():
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from train_cqn_as import step_marks_task_success
    assert step_marks_task_success(_ts({"task_success": 1.0})) is True


def test_step_marks_task_success_false_on_zero():
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from train_cqn_as import step_marks_task_success
    assert step_marks_task_success(_ts({"task_success": 0.0})) is False


def test_step_marks_task_success_false_on_missing_key():
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from train_cqn_as import step_marks_task_success
    # info dict present but no task_success entry
    assert step_marks_task_success(_ts({"safety": {}})) is False


def test_step_marks_task_success_false_on_no_info_attr():
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from train_cqn_as import step_marks_task_success
    from types import SimpleNamespace
    # No `info` attribute at all (e.g. demo TimeSteps in some pre-D paths)
    assert step_marks_task_success(SimpleNamespace()) is False


def test_step_marks_task_success_false_on_non_dict_info():
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from train_cqn_as import step_marks_task_success
    # info is something weird (None, list, etc) — should not raise, just False
    assert step_marks_task_success(_ts(None)) is False
    assert step_marks_task_success(_ts([])) is False


def test_step_marks_task_success_handles_bad_value_types():
    """Defensive: a corrupt info["task_success"] (e.g. string) must not crash
    the train/eval loop with a TypeError."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from train_cqn_as import step_marks_task_success
    assert step_marks_task_success(_ts({"task_success": "yes"})) is False
    assert step_marks_task_success(_ts({"task_success": None})) is False


def test_step_marks_task_success_truthy_on_partial_value():
    """A non-1.0 positive value still counts as success (some envs may emit
    success as a partial score; we treat >0 as success)."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from train_cqn_as import step_marks_task_success
    assert step_marks_task_success(_ts({"task_success": 0.5})) is True

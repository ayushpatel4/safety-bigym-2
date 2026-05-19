"""Tests for the eval video recording helpers used by train_cqn_as.Workspace."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from safety_bigym.agents.cqn_as.eval_video import render_frame, write_eval_video


# ---------- render_frame ----------


def _env_with_render(fn):
    return SimpleNamespace(render=fn)


def test_render_frame_passes_through_hwc_uint8():
    arr = np.full((84, 84, 3), 17, dtype=np.uint8)
    frame = render_frame(_env_with_render(lambda: arr))
    assert frame.shape == (84, 84, 3)
    assert frame.dtype == np.uint8
    assert int(frame[0, 0, 0]) == 17


def test_render_frame_converts_float_to_uint8():
    arr = np.full((48, 48, 3), 0.5, dtype=np.float32)
    frame = render_frame(_env_with_render(lambda: arr))
    assert frame.dtype == np.uint8
    # 0.5 clips to 0 since float values are taken as raw pre-scaled samples
    assert int(frame[0, 0, 0]) == 0


def test_render_frame_passes_through_high_float():
    arr = np.full((24, 24, 3), 200.0, dtype=np.float32)
    frame = render_frame(_env_with_render(lambda: arr))
    assert int(frame[0, 0, 0]) == 200


def test_render_frame_strips_alpha_channel():
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[..., :3] = 200
    arr[..., 3] = 99
    frame = render_frame(_env_with_render(lambda: arr))
    assert frame.shape == (32, 32, 3)
    assert int(frame[0, 0, 0]) == 200


def test_render_frame_returns_none_on_none():
    assert render_frame(_env_with_render(lambda: None)) is None


def test_render_frame_returns_none_on_render_exception(caplog):
    def boom():
        raise RuntimeError("render backend not available")

    with caplog.at_level(logging.WARNING):
        frame = render_frame(_env_with_render(boom), global_step=1234)
    assert frame is None
    assert any("eval render failed" in rec.message for rec in caplog.records)
    assert any("1234" in rec.message for rec in caplog.records)


def test_render_frame_rejects_2d_shape():
    arr = np.zeros((84, 84), dtype=np.uint8)
    assert render_frame(_env_with_render(lambda: arr)) is None


def test_render_frame_rejects_wrong_channel_count():
    arr = np.zeros((84, 84, 5), dtype=np.uint8)
    assert render_frame(_env_with_render(lambda: arr)) is None


# ---------- write_eval_video ----------


def test_write_eval_video_produces_mp4_on_disk(tmp_path):
    frames = [np.full((48, 48, 3), i % 256, dtype=np.uint8) for i in range(30)]
    out = write_eval_video(tmp_path, frames, global_step=12345)
    expected = tmp_path / "step_12345_ep0.mp4"
    assert out == expected
    assert expected.is_file()
    assert expected.stat().st_size > 0


def test_write_eval_video_creates_dir_if_missing(tmp_path):
    frames = [np.zeros((48, 48, 3), dtype=np.uint8) for _ in range(5)]
    out_dir = tmp_path / "nested" / "eval_videos"
    assert not out_dir.exists()
    out = write_eval_video(out_dir, frames, global_step=100)
    assert out_dir.is_dir()
    assert out is not None
    assert out.is_file()


def test_write_eval_video_no_frames_is_noop(tmp_path):
    assert write_eval_video(tmp_path, [], global_step=0) is None
    assert list(tmp_path.iterdir()) == []


def test_write_eval_video_swallows_imageio_import_error(
    tmp_path, monkeypatch, caplog
):
    # Force the imageio.v2 import inside the helper to fail.
    monkeypatch.setitem(sys.modules, "imageio", None)
    monkeypatch.setitem(sys.modules, "imageio.v2", None)
    with caplog.at_level(logging.WARNING):
        out = write_eval_video(
            tmp_path, [np.zeros((4, 4, 3), dtype=np.uint8)], global_step=0
        )
    assert out is None
    assert any("imageio" in rec.message.lower() for rec in caplog.records)


def test_write_eval_video_uploads_to_wandb_when_run_active(tmp_path):
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(10)]
    fake_wandb_run = MagicMock()
    import types as _types

    fake_wandb_module = _types.ModuleType("wandb")
    fake_wandb_module.Video = MagicMock()
    sys.modules.setdefault("wandb", fake_wandb_module)

    out = write_eval_video(
        tmp_path, frames, global_step=50000, wandb_run=fake_wandb_run
    )
    assert out is not None
    fake_wandb_run.log.assert_called_once()
    args, kwargs = fake_wandb_run.log.call_args
    logged = args[0] if args else {}
    assert "eval/video" in logged
    assert kwargs.get("step") == 50000


def test_write_eval_video_handles_wandb_upload_failure_gracefully(tmp_path, caplog):
    """A wandb upload exception must not block the local mp4 write."""
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]
    fake_wandb_run = MagicMock()
    fake_wandb_run.log.side_effect = RuntimeError("wandb offline")
    import types as _types

    fake_wandb_module = _types.ModuleType("wandb")
    fake_wandb_module.Video = MagicMock()
    sys.modules.setdefault("wandb", fake_wandb_module)

    with caplog.at_level(logging.WARNING):
        out = write_eval_video(
            tmp_path, frames, global_step=7777, wandb_run=fake_wandb_run
        )
    # mp4 still written despite the wandb failure
    assert out is not None and out.is_file()
    assert any("wandb video upload failed" in rec.message for rec in caplog.records)

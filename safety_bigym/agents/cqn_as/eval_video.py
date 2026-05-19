"""Eval-time video recording helpers for the CQN-AS training loop.

Extracted from train_cqn_as.py.Workspace so unit tests can exercise the
frame-shape normalisation and mp4-writing logic without dragging the
MuJoCo / hydra / wandb import chain through pytest collection.

The Workspace calls :func:`render_frame` once per env-step during the
first eval episode of each eval cycle, accumulating frames in a list, and
hands the list to :func:`write_eval_video` at episode end. Both helpers
are best-effort: render failures are logged and swallowed so a video
glitch can't bring down a multi-hour training run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def render_frame(env: Any, *, global_step: int = 0) -> Optional[np.ndarray]:
    """Best-effort RGB capture from ``env.render()``.

    Normalises common variations:
    - returns ``None`` if the underlying renderer returned ``None`` or raised
    - converts float arrays to uint8 (clipped to [0, 255])
    - strips an alpha channel to RGB
    - rejects shapes that aren't (H, W, 3 or 4)
    """
    try:
        frame = env.render()
    except Exception as exc:  # pragma: no cover - render backends vary
        logger.warning("eval render failed at step %d: %s", global_step, exc)
        return None
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def write_eval_video(
    video_dir: Path,
    frames: List[np.ndarray],
    *,
    global_step: int,
    fps: int = 30,
    wandb_run: Any = None,
) -> Optional[Path]:
    """Dump captured frames as an mp4 under ``video_dir``.

    Returns the written path on success, ``None`` on any failure. Errors are
    logged and swallowed — a video-write failure must never bring down a
    multi-hour training run.

    If ``wandb_run`` is provided (a live W&B run), the produced mp4 is also
    uploaded under the ``eval/video`` key at the same global step.
    """
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
    except Exception:
        logger.warning(
            "save_video=true but imageio not available; skipping eval video."
        )
        return None
    try:
        video_dir.mkdir(parents=True, exist_ok=True)
        out = video_dir / f"step_{global_step}_ep0.mp4"
        imageio.mimsave(str(out), frames, fps=fps, macro_block_size=1)
        logger.info("saved eval video: %s (%d frames)", out, len(frames))
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "eval video write failed at step %d: %s", global_step, exc
        )
        return None
    if wandb_run is not None:
        try:
            import wandb

            wandb_run.log(
                {"eval/video": wandb.Video(str(out), fps=fps, format="mp4")},
                step=global_step,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("wandb video upload failed: %s", exc)
    return out


__all__ = ["render_frame", "write_eval_video"]

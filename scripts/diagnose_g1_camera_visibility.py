#!/usr/bin/env python
"""Diagnose what the CNN actually sees during a G1 ``coworker_idle`` rollout.

Step 0b of the G1 base-curriculum recovery plan
(``/Users/ayushpatel/.claude/plans/read-safety-bigym-docs-implementation-st-kind-aurora.md``):
the leading hypothesis is CNN visual OOD on G1's dark/metallic appearance,
but it depends on G1 *actually being in the camera frame*. ``coworker_idle``
keeps the human at 3.0-3.6 m — far enough that it may sit outside the
head/wrist FOVs entirely (in which case visual OOD is NOT the bottleneck and
we should look elsewhere).

The existing ``eval_videos/step_*_ep0.mp4`` recordings use ``env.render()``
which returns the third-person free camera — NOT the head/wrist cameras the
CNN consumes. This script dumps **per-policy-camera** MP4s + a contact-sheet
PNG so you can verify by eye.

Usage (GPU box, mirrors the attempt-4 stage-0 config):
    export AMASS_DATA_DIR=/path/to/CMU/CMU
    export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
    python scripts/diagnose_g1_camera_visibility.py \\
        env=safety_bigym/saucepan_to_hob \\
        disruption=coworker_idle \\
        bodyslam=oracle \\
        +steps=400 \\
        +outdir=/tmp/g1_cam_diag

Output (under ``outdir``):
    rgb_head.mp4              # what the head cam sees
    rgb_right_wrist.mp4       # what the right wrist cam sees
    rgb_left_wrist.mp4        # what the left wrist cam sees
    external.mp4              # third-person debug view (env.render())
    contact_sheet.png         # one row of 3 cams, sampled at 5 timesteps
    summary.txt               # per-cam stats (mean pixel value, std, etc)

What to look for:
- Is the G1 visible in any of head/right_wrist/left_wrist at all?
- If visible, is it dark blob / silhouette / textured? Does its appearance
  vary or stay static (G1 is mostly fixed-pose under ``coworker_idle``)?
- Does it move when you'd expect (slow walk between 3.0-3.6 m)?

If G1 is NOT visible in the head/wrist cams during ``coworker_idle``, the
visual-OOD hypothesis loses its main support and you should skip to
the ``pixels=false`` ablation (Step 1) to look for non-CNN root causes.

You can override the disruption to ``coworker_train`` to see what the full
disruption looks like:
    python scripts/diagnose_g1_camera_visibility.py \\
        env=safety_bigym/saucepan_to_hob disruption=coworker_train \\
        bodyslam=oracle +steps=400 +outdir=/tmp/g1_cam_diag_train
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# train_cqn_as.py sets MUJOCO_GL=egl at import time, but for headless GPU
# boxes this script needs the same default. Honour any user override.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import hydra
import numpy as np
from omegaconf import DictConfig

if not os.environ.get("AMASS_DATA_DIR"):
    raise RuntimeError(
        "AMASS_DATA_DIR is not set. Export it (see safety_bigym/docs/CLAUDE.md)."
    )

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory  # noqa: E402

logger = logging.getLogger(__name__)


def _frame_chw_to_hwc(frame: np.ndarray) -> np.ndarray:
    """BiGym returns ``rgb_<cam>`` shaped (3, H, W) — convert to (H, W, 3) uint8."""
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _summarise_cam(name: str, frames: list[np.ndarray]) -> str:
    """Per-cam stats: mean / std / fraction of dark pixels (proxy for G1)."""
    if not frames:
        return f"{name}: no frames captured"
    stack = np.stack(frames).astype(np.float32)  # (T, H, W, 3)
    mean_rgb = stack.mean(axis=(0, 1, 2))
    std_rgb = stack.std(axis=(0, 1, 2))
    # Fraction of pixels with luminance < 0.3 (the G1 `g1_black` rgba=0.2
    # would project to ~50/255 — a useful dark-blob detector).
    lum = 0.299 * stack[..., 0] + 0.587 * stack[..., 1] + 0.114 * stack[..., 2]
    dark_frac = (lum < 76.5).mean()
    return (
        f"{name}: shape={tuple(stack.shape)} "
        f"mean_rgb=({mean_rgb[0]:.1f},{mean_rgb[1]:.1f},{mean_rgb[2]:.1f}) "
        f"std_rgb=({std_rgb[0]:.1f},{std_rgb[1]:.1f},{std_rgb[2]:.1f}) "
        f"dark_frac<0.3={dark_frac:.3f}"
    )


def _write_mp4(out_path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    if not frames:
        logger.warning("no frames for %s; skipping", out_path)
        return
    try:
        import imageio.v2 as imageio
    except Exception:
        logger.warning("imageio unavailable; skipping %s", out_path)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps, macro_block_size=1)
    logger.info("wrote %s (%d frames)", out_path, len(frames))


def _write_contact_sheet(
    out_path: Path, cams_by_name: dict[str, list[np.ndarray]]
) -> None:
    """One row per cam, 5 samples per row across time. Quick visual gestalt."""
    try:
        import imageio.v2 as imageio
    except Exception:
        return
    if not cams_by_name:
        return
    sample_count = 5
    rows = []
    cam_names = list(cams_by_name.keys())
    for name in cam_names:
        frames = cams_by_name[name]
        if not frames:
            continue
        idxs = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
        cells = [frames[i] for i in idxs]
        row = np.concatenate(cells, axis=1)  # along width
        rows.append(row)
    if not rows:
        return
    # Pad each row to same width (safety in case cams have different sizes).
    max_w = max(r.shape[1] for r in rows)
    rows = [
        np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)), mode="constant")
        for r in rows
    ]
    sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(out_path), sheet)
    logger.info("wrote contact sheet %s", out_path)


@hydra.main(version_base=None, config_path="../cfgs", config_name="cqn_as_config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    steps = int(cfg.get("steps", 400))
    outdir = Path(str(cfg.get("outdir", "/tmp/g1_cam_diag"))).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger.info("writing diagnostic output under %s", outdir)
    logger.info(
        "env=%s disruption=%s bodyslam=%s steps=%d",
        cfg.env.task_name, cfg.env.get("disruption_type", "?"),
        cfg.env.get("bodyslam", {}).get("mode", "off"), steps,
    )

    # Build the env via the same factory the trainer uses. This honours all
    # the workspace shaping / coworker / camera config from the composed yamls.
    factory = SafetyBiGymEnvFactory()
    env = factory._create_env(cfg)
    cam_names = list(cfg.env.get("cameras", ["head", "right_wrist", "left_wrist"]))
    logger.info("cameras: %s", cam_names)

    frames_by_cam: dict[str, list[np.ndarray]] = {name: [] for name in cam_names}
    external_frames: list[np.ndarray] = []

    obs, _info = env.reset()
    for name in cam_names:
        key = f"rgb_{name}"
        if key in obs:
            frames_by_cam[name].append(_frame_chw_to_hwc(obs[key]))
        else:
            logger.warning("obs missing %s on reset", key)
    try:
        ext = env.render()
        if ext is not None:
            external_frames.append(_frame_chw_to_hwc(ext))
    except Exception as exc:
        logger.warning("external render() failed on reset: %s", exc)

    # Random policy is fine — we want to see the human, not the robot's task
    # progress. coworker_idle mostly has the human standing still anyway.
    action_space = env.action_space
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    for t in range(steps):
        if hasattr(action_space, "sample"):
            action = action_space.sample()
        else:
            action = rng.uniform(-1, 1, size=action_space.shape).astype(np.float32)
        obs, _r, term, trunc, _info = env.step(action)
        for name in cam_names:
            key = f"rgb_{name}"
            if key in obs:
                frames_by_cam[name].append(_frame_chw_to_hwc(obs[key]))
        if t % max(1, steps // 60) == 0:
            try:
                ext = env.render()
                if ext is not None:
                    external_frames.append(_frame_chw_to_hwc(ext))
            except Exception:
                pass
        if term or trunc:
            obs, _info = env.reset()

    summary_lines = [f"# G1 camera-visibility diagnostic"]
    summary_lines.append(f"# env={cfg.env.task_name}")
    summary_lines.append(f"# steps={steps}")
    summary_lines.append("")
    for name in cam_names:
        line = _summarise_cam(name, frames_by_cam[name])
        summary_lines.append(line)
        logger.info(line)
        _write_mp4(outdir / f"rgb_{name}.mp4", frames_by_cam[name], fps=30)
    if external_frames:
        _write_mp4(outdir / "external.mp4", external_frames, fps=10)
        summary_lines.append(_summarise_cam("external (env.render)", external_frames))
    _write_contact_sheet(outdir / "contact_sheet.png", frames_by_cam)

    (outdir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    logger.info("done. summary.txt + %d mp4s + contact_sheet.png in %s",
                len(cam_names) + (1 if external_frames else 0), outdir)


if __name__ == "__main__":
    main()

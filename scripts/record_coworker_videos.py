#!/usr/bin/env python
"""
Record MP4s of the COWORKER disruption across tasks and spawn modes.

For each ``(task, spawn_mode)`` pair this script builds a SafetyBiGymEnv
with the disruption forced to COWORKER, resets it, then runs a fixed
number of zero-action env steps and writes ``env.render()`` frames to an
MP4. Episodes are long enough (default 35 s of sim time) that even the
longest COWORKER_PATROL trajectory completes one full
near -> away -> back-to-near excursion within the recording window.

Usage::

    export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU
    cd safety_bigym
    python scripts/record_coworker_videos.py --out-dir vids/coworker
    # or a subset:
    python scripts/record_coworker_videos.py --tasks reach saucepan \\
        --spawns patrol --out-dir vids/coworker

On macOS the default ``glfw`` backend handles offscreen rendering
without any env-var setup. On a headless Linux box prepend
``MUJOCO_GL=egl PYOPENGL_PLATFORM=egl``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import imageio
import mujoco
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("record_coworker_videos")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bigym.action_modes import JointPositionActionMode
from bigym.utils.observation_config import CameraConfig, ObservationConfig
from safety_bigym import HumanConfig, SafetyConfig, make_safety_env
from safety_bigym.scenarios import (
    DisruptionType,
    ParameterSpace,
    ScenarioSampler,
)


# Tasks the user asked for, mapped to BiGym classes.
TASK_MAP = {
    "reach": "bigym.envs.reach_target:ReachTargetSingle",
    "saucepan": "bigym.envs.pick_and_place:SaucepanToHob",
    "dishwasher_close": "bigym.envs.dishwasher:DishwasherClose",
    "drawers_open_all": "bigym.envs.cupboards:DrawersAllOpen",
}

# Spawn modes -> trajectory_type strings forced on every reset.
SPAWN_TRAJ = {
    "walk_in": "APPROACH_LOITER_DEPART",
    "in_place": "STATIONARY",
    "patrol": "COWORKER_PATROL",
}


def _load_task_cls(task_key: str) -> type:
    spec = TASK_MAP[task_key]
    module_path, cls_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), cls_name)


def _build_env(task_cls: type, motion_dir: str):
    """Build a SafetyBiGymEnv tuned for rendering + COWORKER playback."""
    sampler = ScenarioSampler(
        parameter_space=ParameterSpace(
            clip_paths=["74/74_01_poses.npz"],
            disruption_weights={DisruptionType.COWORKER: 1.0},
        ),
        motion_dir=motion_dir,
    )
    env = make_safety_env(
        task_cls=task_cls,
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        safety_config=SafetyConfig(
            log_violations=False, terminate_on_violation=False,
        ),
        human_config=HumanConfig(
            motion_clip_dir=motion_dir,
            motion_clip_paths=["74/74_01_poses.npz"],
        ),
        scenario_sampler=sampler,
        inject_human=True,
        render_mode="rgb_array",
        observation_config=ObservationConfig(
            cameras=[CameraConfig("head", resolution=(256, 256))],
            proprioception=True,
            privileged_information=True,
        ),
    )
    return env, sampler


def _force_spawn_mode(sampler: ScenarioSampler, spawn_mode: str) -> None:
    """Patch the sampler so every sampled scenario uses the chosen mode."""
    base = sampler.sample_scenario
    target_traj = SPAWN_TRAJ[spawn_mode]

    def _patched(seed):
        s = base(seed)
        s.trajectory_type = target_traj
        # Reach target alternates EE / task object so the viewer can see
        # both. Force-active arm depends on entry side, leave to sampler.
        s.disruption_config.coworker_target_mix = (0.5, 0.5)
        # Patrol-specific: bump excursions so we always see a depart+return
        # within the recording window, regardless of how the sampler picked.
        if target_traj == "COWORKER_PATROL":
            s.patrol_excursions = max(int(getattr(s, "patrol_excursions", 1)), 2)
            # Trim near-loiter slightly so 2 excursions fit in 35 s.
            s.patrol_near_loiter = min(
                float(getattr(s, "patrol_near_loiter", 8.0)), 7.0
            )
        return s

    sampler.sample_scenario = _patched  # type: ignore[assignment]


def _set_render_resolution(env, height: int, width: int) -> None:
    """Best-effort: bump the gymnasium renderer's frame size before the
    first ``env.render()`` call. Older mujoco-gym wraps may read the size
    once at construction; falling through silently is OK."""
    try:
        env.unwrapped.mujoco_renderer.width = int(width)
        env.unwrapped.mujoco_renderer.height = int(height)
    except Exception:
        pass


def _record_one(
    task_key: str,
    spawn_mode: str,
    out_path: Path,
    motion_dir: str,
    sim_seconds: float,
    fps: int,
    seed: int,
    resolution: tuple[int, int],
) -> None:
    task_cls = _load_task_cls(task_key)
    log.info(
        "rendering task=%s spawn=%s -> %s  (sim=%.1fs, fps=%d, res=%dx%d)",
        task_key, spawn_mode, out_path, sim_seconds, fps,
        resolution[1], resolution[0],
    )

    env, sampler = _build_env(task_cls, motion_dir)
    _force_spawn_mode(sampler, spawn_mode)
    _set_render_resolution(env, resolution[0], resolution[1])

    obs, info = env.reset(seed=seed)
    scenario = env._current_scenario
    cb = env._coworker_controller
    log.info(
        "  scenario: traj=%s arm=%s mix=%s closest_approach=%.2fm",
        scenario.trajectory_type,
        scenario.disruption_config.coworker_active_arm,
        scenario.disruption_config.coworker_target_mix,
        scenario.closest_approach,
    )

    # Step count is driven by env.human_controller.t (the controller's
    # accumulated sim time), since env.step substeps vary by task.
    frames: list[np.ndarray] = []
    next_capture_t = 0.0
    frame_dt = 1.0 / fps

    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    last_log = started

    while env.human_controller.t < sim_seconds:
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        t = env.human_controller.t
        if t >= next_capture_t:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            next_capture_t += frame_dt
        # Heartbeat for long renders.
        if time.time() - last_log > 5.0:
            log.info(
                "  ...t=%.1fs / %.1fs  (%d frames captured)",
                t, sim_seconds, len(frames),
            )
            last_log = time.time()

    env.close()

    if not frames:
        log.error("  no frames captured for %s/%s — skipping write", task_key, spawn_mode)
        return

    log.info(
        "  captured %d frames in %.1fs wall, writing MP4",
        len(frames), time.time() - started,
    )
    imageio.mimsave(str(out_path), frames, fps=fps, macro_block_size=1)
    log.info("  -> %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks", nargs="+", default=list(TASK_MAP.keys()),
        choices=list(TASK_MAP.keys()),
        help="Subset of tasks to render (default: all four).",
    )
    p.add_argument(
        "--spawns", nargs="+", default=list(SPAWN_TRAJ.keys()),
        choices=list(SPAWN_TRAJ.keys()),
        help="Subset of spawn modes to render (default: all three).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT.parent / "vids" / "coworker",
        help="Where to put MP4s. Files are named '<task>_<spawn>.mp4'.",
    )
    p.add_argument(
        "--sim-seconds", type=float, default=35.0,
        help="Sim time per video. The default fits one full patrol "
        "depart+return cycle (typically ~25-30 s) with margin.",
    )
    p.add_argument(
        "--fps", type=int, default=25,
        help="Output video frame rate. Capture cadence matches.",
    )
    p.add_argument(
        "--resolution", default="480x640",
        help="Output frame size as HxW (default 480x640).",
    )
    p.add_argument(
        "--seed", type=int, default=7,
        help="Episode seed; same seed across spawn modes makes the "
        "task scene identical so the viewer can isolate human behaviour.",
    )
    args = p.parse_args()

    amass_dir = os.environ.get("AMASS_DATA_DIR")
    if not amass_dir:
        raise RuntimeError(
            "AMASS_DATA_DIR is not set. Export it to the CMU AMASS root, e.g.\n"
            "  export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU"
        )

    try:
        h, w = (int(x) for x in args.resolution.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--resolution must look like HxW, got {args.resolution!r}")

    for task in args.tasks:
        for spawn in args.spawns:
            out = args.out_dir / f"{task}_{spawn}.mp4"
            try:
                _record_one(
                    task_key=task,
                    spawn_mode=spawn,
                    out_path=out,
                    motion_dir=amass_dir,
                    sim_seconds=args.sim_seconds,
                    fps=args.fps,
                    seed=args.seed,
                    resolution=(h, w),
                )
            except Exception as e:
                log.exception("FAILED %s/%s: %s", task, spawn, e)


if __name__ == "__main__":
    main()

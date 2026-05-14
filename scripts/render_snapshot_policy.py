#!/usr/bin/env python
"""Render a snapshot policy under a forced disruption type as an MP4.

Loads an ACT/DP snapshot (workspace.py drift format), builds a
SafetyBiGymEnv with the disruption pinned to a single type, rolls out until
a successful episode is captured, and writes the frames as a video.

Designed to run headless on the GPU box; prepend ``MUJOCO_GL=egl`` (and
``PYOPENGL_PLATFORM=egl``) on nodes without an X display. Reuses
``svf_collect_dataset`` helpers so the env / policy plumbing matches the
SVF eval path exactly (same camera resolution, same bodyslam mode, same
adapter).

Example::

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    AMASS_DATA_DIR=/path/to/CMU/CMU \\
    python scripts/render_snapshot_policy.py \\
        --task saucepan_to_hob \\
        --disruption SHARED_GOAL \\
        --snapshot exp_local/act_safety/saucepan_to_hob_20260428205105/snapshots/50000_snapshot.pt \\
        --output saucepan_shared_goal.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.svf_collect_dataset import (  # noqa: E402
    TASK_REGISTRY,
    DEFAULT_CLIPS,
    _import_task,
    load_snapshot_policy,
    peek_snapshot_bodyslam_mode,
    peek_snapshot_cameras,
)

logger = logging.getLogger("render_snapshot_policy")


def _build_render_env(
    task_key: str,
    disruption: str,
    bodyslam_mode: str,
    motion_clips: Sequence[str],
    *,
    cameras: Sequence[str],
    camera_resolution: Tuple[int, int],
    render_resolution: Tuple[int, int],
):
    """Mirror ``_build_live_env`` but also wires ``render_mode="rgb_array"``.

    Kept local instead of patching the collector helper because the
    collector explicitly never renders (perf-sensitive dataset path).
    """
    amass = os.environ.get("AMASS_DATA_DIR")
    if not amass:
        raise RuntimeError(
            "AMASS_DATA_DIR is not set. Export it before running:\n"
            "  export AMASS_DATA_DIR=/path/to/CMU/CMU"
        )

    from safety_bigym import SafetyConfig, HumanConfig, make_safety_env
    from safety_bigym.perception.bodyslam_wrapper import BodySLAMWrapper
    from safety_bigym.scenarios.disruption_types import DisruptionType
    from safety_bigym.scenarios.scenario_sampler import ParameterSpace, ScenarioSampler
    from bigym.action_modes import JointPositionActionMode
    from bigym.utils.observation_config import CameraConfig, ObservationConfig

    task_cls = _import_task(task_key)
    human_config = HumanConfig(
        motion_clip_dir=amass,
        motion_clip_paths=list(motion_clips),
    )
    sampler = ScenarioSampler(
        parameter_space=ParameterSpace(
            clip_paths=human_config.motion_clip_paths,
            disruption_weights={DisruptionType[disruption]: 1.0},
        ),
        motion_dir=amass,
    )
    make_env_kwargs = {"render_mode": "rgb_array"}
    if cameras:
        make_env_kwargs["observation_config"] = ObservationConfig(
            cameras=[
                CameraConfig(
                    name=name, rgb=True, depth=False,
                    resolution=tuple(camera_resolution),
                )
                for name in cameras
            ],
            proprioception=True,
            privileged_information=False,
        )

    env = make_safety_env(
        task_cls=task_cls,
        action_mode=JointPositionActionMode(absolute=True, floating_base=True),
        safety_config=SafetyConfig(terminate_on_violation=False),
        human_config=human_config,
        scenario_sampler=sampler,
        inject_human=True,
        **make_env_kwargs,
    )

    # MujocoRenderer respects whatever default width/height it was built
    # with; nudge the underlying gymnasium renderer to the requested size.
    try:
        env.unwrapped.mujoco_renderer.width = int(render_resolution[1])
        env.unwrapped.mujoco_renderer.height = int(render_resolution[0])
    except AttributeError:
        pass

    if bodyslam_mode == "off":
        return env
    if bodyslam_mode not in ("oracle", "noisy"):
        raise ValueError(f"unexpected bodyslam_mode={bodyslam_mode!r}")
    return BodySLAMWrapper(env, mode=bodyslam_mode)


def _parse_resolution(value: str) -> Tuple[int, int]:
    try:
        h, w = (int(x) for x in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"resolution must look like HxW, got {value!r}"
        ) from exc
    return h, w


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task", required=True, choices=sorted(TASK_REGISTRY),
        help="Task key (matches svf_collect_dataset.TASK_REGISTRY).",
    )
    p.add_argument(
        "--disruption", default="SHARED_GOAL",
        help="DisruptionType name forced for every episode.",
    )
    p.add_argument(
        "--snapshot", required=True, type=Path,
        help="Path to a *_snapshot.pt file (workspace.py drift format).",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output mp4 path.",
    )
    p.add_argument(
        "--motion-clips", nargs="+", default=list(DEFAULT_CLIPS),
        help="AMASS clip paths (relative to AMASS_DATA_DIR).",
    )
    p.add_argument(
        "--num-episodes", type=int, default=20,
        help="Max episodes to roll out searching for a success.",
    )
    p.add_argument(
        "--max-steps", type=int, default=400,
        help="Per-episode step budget.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fps", type=int, default=25,
        help="Output video fps.",
    )
    p.add_argument(
        "--render-resolution", default="480x640", type=_parse_resolution,
        help="HxW for env.render(). Independent of the policy's obs camera.",
    )
    p.add_argument(
        "--save-best-on-failure", action="store_true",
        help=(
            "If no success in --num-episodes, save the longest non-failing "
            "episode instead of erroring out."
        ),
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (repeat for DEBUG).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else (
            logging.INFO if args.verbose else logging.WARNING
        ),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    snapshot_path = args.snapshot.expanduser().resolve()
    if not snapshot_path.is_file():
        raise SystemExit(f"snapshot not found: {snapshot_path}")

    bodyslam_mode = peek_snapshot_bodyslam_mode(snapshot_path)
    cameras, camera_resolution = peek_snapshot_cameras(snapshot_path)
    logger.info(
        "snapshot: bodyslam=%s cameras=%s @ %dx%d",
        bodyslam_mode, list(cameras) or "none",
        camera_resolution[0], camera_resolution[1],
    )

    env = _build_render_env(
        args.task, args.disruption, bodyslam_mode, args.motion_clips,
        cameras=cameras,
        camera_resolution=camera_resolution,
        render_resolution=args.render_resolution,
    )

    policy = load_snapshot_policy(snapshot_path, env)

    import imageio.v2 as imageio  # type: ignore

    rng = np.random.default_rng(args.seed)
    best_frames: list[np.ndarray] = []
    best_episode_idx = -1

    successful_frames: list[np.ndarray] | None = None
    success_episode_idx = -1

    for ep in range(args.num_episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        obs, _info = env.reset(seed=seed)
        frames: list[np.ndarray] = []
        last_info: dict = {}
        terminated = truncated = False
        steps = 0

        for steps in range(1, args.max_steps + 1):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            action = policy(obs).astype(np.float32, copy=False)
            obs, _reward, terminated, truncated, last_info = env.step(action)
            if terminated or truncated:
                break

        # Capture the final frame so success poses show up in the video.
        final_frame = env.render()
        if final_frame is not None:
            frames.append(final_frame)

        succeeded = bool(last_info.get("task_success", 0.0) > 0.5)
        logger.info(
            "episode %d/%d seed=%d steps=%d terminated=%s truncated=%s success=%s",
            ep + 1, args.num_episodes, seed, steps,
            terminated, truncated, succeeded,
        )

        if succeeded:
            successful_frames = frames
            success_episode_idx = ep
            break

        if len(frames) > len(best_frames):
            best_frames = frames
            best_episode_idx = ep

    env.close()

    if successful_frames is not None:
        chosen = successful_frames
        label = f"success on episode {success_episode_idx + 1}"
    elif args.save_best_on_failure and best_frames:
        chosen = best_frames
        label = (
            f"no success in {args.num_episodes} eps; saving longest "
            f"(episode {best_episode_idx + 1}, {len(best_frames)} frames)"
        )
    else:
        raise SystemExit(
            f"No successful episode in {args.num_episodes} attempts. "
            "Re-run with --save-best-on-failure to write the longest run, "
            "or raise --num-episodes / --max-steps."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.warning("writing %d frames to %s (%s)", len(chosen), args.output, label)
    imageio.mimsave(str(args.output), chosen, fps=args.fps)
    print(f"wrote {args.output} — {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

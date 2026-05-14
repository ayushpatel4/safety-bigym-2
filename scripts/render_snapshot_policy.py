#!/usr/bin/env python
"""Render a snapshot policy under a forced disruption type as an MP4.

Loads an ACT/DP snapshot (workspace.py drift format) and rolls it out
against a ``SafetyBiGymEnvFactory.make_eval_env``-built env with the
disruption pinned to a single type. Writes the first successful episode
to MP4.

Uses the full RoboBase eval-time wrapper stack (RescaleFromTanh,
ConcatDim, FrameStack, RecedingHorizonControl) so the obs/action spaces
match what the policy was trained with — re-creating only a subset of
those wrappers silently mis-shapes the encoder input and the ACT
encoder's shape assertion fires.

Headless-friendly: prepend ``MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`` on
nodes without an X display.

Example::

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
    AMASS_DATA_DIR=/path/to/CMU/CMU \\
    python scripts/render_snapshot_policy.py \\
        --task dishwasher_close \\
        --disruption SHARED_GOAL \\
        --snapshot exp_local/act_safety/dishwasher_close_20260428235941/snapshots/40000_snapshot.pt \\
        --output vids/dishwasher_close_shared_goal.mp4
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

logger = logging.getLogger("render_snapshot_policy")


# task_key → cfg.env.task_name expected by SafetyBiGymEnvFactory.
# Keep in lock-step with scripts/svf_collect_dataset.TASK_REGISTRY.
_TASK_NAME_OVERRIDES = {
    "reach_target_single": "reach_target_single",
    "dishwasher_close": "dishwasher_close",
    "dishwasher_load_plates": "dishwasher_load_plates",
    "saucepan_to_hob": "saucepan_to_hob",
}


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
        "--task", required=True, choices=sorted(_TASK_NAME_OVERRIDES),
        help="Task key.",
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
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument(
        "--max-steps", type=int, default=600,
        help="Per-episode outer-step budget (each outer step = "
             "cfg.execution_length inner env steps for ACT).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument(
        "--render-resolution", default="480x640", type=_parse_resolution,
        help="HxW for env.render(). Independent of the policy's obs camera.",
    )
    p.add_argument(
        "--save-best-on-failure", action="store_true",
        help="Save longest non-failing episode if no success in --num-episodes.",
    )
    p.add_argument(
        "--device", default="cpu",
        help="torch device for the policy (cpu/cuda).",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args(argv)


def _load_snapshot(snapshot_path: Path):
    import torch
    from omegaconf import DictConfig, OmegaConf

    payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    cfg = payload.get("cfg")
    if cfg is None:
        raise KeyError(
            f"snapshot at {snapshot_path} has no 'cfg' field — was it produced "
            "after the workspace.py drift fix?"
        )
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    return payload, cfg


def _override_cfg(cfg, *, task_key: str, disruption: str) -> None:
    from omegaconf import OmegaConf, open_dict

    # The snapshot's cfg is frozen (struct=True); open_dict relaxes it so
    # we can poke fields like disruption_type / demos onto it.
    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        # Force the requested disruption every episode (factory honours this
        # field at L204 in safety_bigym_factory.py).
        cfg.env.disruption_type = disruption
        cfg.env.render_mode = "rgb_array"
        # Make sure motion_clip_dir resolves on the current box.
        amass = os.environ.get("AMASS_DATA_DIR")
        if not amass:
            raise RuntimeError(
                "AMASS_DATA_DIR is not set. Export it before running:\n"
                "  export AMASS_DATA_DIR=/path/to/CMU/CMU"
            )
        cfg.env.motion_clip_dir = amass
        # Override task name if the snapshot was trained on a different one
        # (the same env_factory expects cfg.env.task_name to drive class
        # selection in _task_name_to_env_class).
        cfg.env.task_name = _TASK_NAME_OVERRIDES[task_key]
        # Skip demo loading at eval — snapshot carries action/obs stats.
        cfg.demos = 0
        cfg.num_train_envs = 0
        cfg.num_train_frames = 0
        cfg.num_pretrain_steps = 0
        # `replay` block is referenced by hydra.utils.instantiate for the
        # agent; leave as-is (snapshot cfg already has it).


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

    payload, cfg = _load_snapshot(snapshot_path)
    _override_cfg(cfg, task_key=args.task, disruption=args.disruption)
    logger.info(
        "snapshot cfg: task=%s frame_stack=%s pixels=%s cameras=%s "
        "action_seq=%s execution_len=%s",
        cfg.env.get("task_name"), cfg.get("frame_stack"),
        cfg.get("pixels"), list(cfg.env.get("cameras", [])),
        cfg.get("action_sequence"), cfg.get("execution_length"),
    )

    # Build env via the real factory — applies the full wrapper stack.
    from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory

    factory = SafetyBiGymEnvFactory()
    # workspace.py drift: factory captures these by reference at env-build,
    # so seed them before make_eval_env.
    if (stats := payload.get("action_stats")) is not None:
        factory._action_stats = stats
    if (stats := payload.get("obs_stats")) is not None:
        factory._obs_stats = stats

    eval_env = factory.make_eval_env(cfg)

    # Best-effort: bump the gymnasium renderer's frame size before any
    # render() call. Falls through silently on builds that read the size at
    # construction-only.
    try:
        eval_env.unwrapped.mujoco_renderer.width = int(args.render_resolution[1])
        eval_env.unwrapped.mujoco_renderer.height = int(args.render_resolution[0])
    except AttributeError:
        pass

    # Instantiate the agent the same way Workspace does.
    import hydra
    import torch

    method_cfg = cfg.method
    agent = hydra.utils.instantiate(
        method_cfg,
        device=args.device,
        observation_space=eval_env.observation_space,
        action_space=eval_env.action_space,
        num_train_envs=0,
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
        intrinsic_reward_module=None,
    )
    agent.load_state_dict(payload["agent"], strict=False)
    # Restore EMA shadow params explicitly (workspace.py drift bullet 4).
    if "actor_ema" in payload and hasattr(agent, "actor"):
        actor_ema = payload["actor_ema"]
        if hasattr(agent.actor, "ema") and hasattr(agent.actor.ema, "shadow_params"):
            for p, sp in zip(agent.actor.ema.shadow_params, actor_ema):
                p.data.copy_(sp)
    agent.train(False)
    device = torch.device(args.device)

    def _obs_to_batch(obs: dict) -> dict:
        """Lift each numpy obs to (B=1, T, ...) torch tensor on device.

        The eval env (FrameStack) already gives each value with a leading
        T axis; we add a leading B=1.
        """
        out = {}
        for k, v in obs.items():
            arr = np.asarray(v)
            t = torch.from_numpy(arr).unsqueeze(0).to(device)
            out[k] = t
        return out

    import imageio.v2 as imageio  # type: ignore

    rng = np.random.default_rng(args.seed)
    best_frames: list[np.ndarray] = []
    best_episode_idx = -1
    successful_frames: list[np.ndarray] | None = None
    success_episode_idx = -1

    for ep in range(args.num_episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        obs, _info = eval_env.reset(seed=seed)
        frames: list[np.ndarray] = []
        last_info: dict = {}
        terminated = truncated = False
        outer_steps = 0

        for outer_steps in range(1, args.max_steps + 1):
            frame = eval_env.render()
            if frame is not None:
                frames.append(frame)

            with torch.no_grad():
                action_t = agent.act(_obs_to_batch(obs), step=0, eval_mode=True)
            action = action_t.detach().cpu().numpy()
            # ACT returns either a single chunk (L, A) or (B=1, L, A). RHC
            # consumes (L, A). Strip any leading batch axis.
            if action.ndim == 3 and action.shape[0] == 1:
                action = action[0]
            obs, _reward, terminated, truncated, last_info = eval_env.step(action)
            if terminated or truncated:
                break

        final_frame = eval_env.render()
        if final_frame is not None:
            frames.append(final_frame)

        # task_success either lives directly in info (from BiGym base env)
        # or under the episode_safety block when EpisodeSafetyMetrics emits.
        ep_info = last_info.get("episode_safety", {}) or {}
        success_flag = float(
            last_info.get("task_success", ep_info.get("ep_task_success", 0.0))
            or 0.0
        )
        succeeded = success_flag > 0.5
        logger.info(
            "episode %d/%d seed=%d outer_steps=%d terminated=%s truncated=%s success=%s",
            ep + 1, args.num_episodes, seed, outer_steps,
            terminated, truncated, succeeded,
        )

        if succeeded:
            successful_frames = frames
            success_episode_idx = ep
            break

        if len(frames) > len(best_frames):
            best_frames = frames
            best_episode_idx = ep

    eval_env.close()

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

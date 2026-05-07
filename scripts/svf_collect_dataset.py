#!/usr/bin/env python
"""SVF transition dataset collector.

Phase 2 — collects ``(obs, action, next_obs, r_safe, done, ssm_margin, ...)``
transitions from rollouts in ``SafetyBiGymEnv + BodySLAMWrapper`` and writes
them as sharded ``.npz`` for the offline CQL trainer.

Sources (CLI flag, repeatable):
  ``random``    — uniform actions sampled from ``env.action_space``  (✅ v1)
  ``demo``      — replay BiGym demos                                  (TODO; raises NotImplementedError)
  ``snapshot``  — load an ACT/DP snapshot via the workspace.py drift  (TODO; raises NotImplementedError)

Smoke mode runs 1 task × 1 disruption × 2 episodes × ≤50 steps and writes a
single ``_smoke_shard.npz``.

Usage:
    python scripts/svf_collect_dataset.py --smoke
    python scripts/svf_collect_dataset.py \
        --source random --tasks reach_target_single --episodes-per-cell 50

Hand-off to GPU for the full ~500k-transition collection.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("svf_collect_dataset")


# AMASS gate matches the convention in scripts/generate_dataset.py — fail
# loudly at import time so smoke runs surface the missing env var.
AMASS_DATA_DIR = os.environ.get("AMASS_DATA_DIR")


SOURCE_CODES: Dict[str, int] = {"demo": 0, "random": 1, "snapshot": 2}


# Task registry — keep aligned with scripts/baseline_sweep.py TASKS.
TASK_REGISTRY: Dict[str, Tuple[str, int]] = {
    "reach_target_single": ("bigym.envs.reach_target.ReachTargetSingle", 0),
    "dishwasher_close": ("bigym.envs.dishwasher.DishwasherClose", 1),
    "dishwasher_load_plates": ("bigym.envs.dishwasher_plates.DishwasherLoadPlates", 2),
    "saucepan_to_hob": ("bigym.envs.saucepan.SaucepanToHob", 3),
}

DEFAULT_DISRUPTIONS = (
    "INCIDENTAL",
    "SHARED_GOAL",
    "DIRECT",
    "OBSTRUCTION",
    "RANDOM_PERTURBED",
)

DEFAULT_CLIPS = ("74/74_01_poses.npz",)


@dataclass
class CollectionPlan:
    """Frozen description of a single collection run."""

    sources: Tuple[str, ...]
    tasks: Tuple[str, ...]
    disruptions: Tuple[str, ...]
    episodes_per_cell: int
    max_steps: int
    bodyslam_mode: str  # "oracle" | "noisy"
    output_dir: Path
    snapshot_path: Optional[Path] = None
    seed: int = 0
    motion_clips: Tuple[str, ...] = DEFAULT_CLIPS

    @classmethod
    def smoke(cls, output_dir: Path) -> "CollectionPlan":
        return cls(
            sources=("random",),
            tasks=("reach_target_single",),
            disruptions=("INCIDENTAL",),
            episodes_per_cell=2,
            max_steps=50,
            bodyslam_mode="oracle",
            output_dir=output_dir,
            seed=0,
        )


def _import_task(task_key: str):
    if task_key not in TASK_REGISTRY:
        raise KeyError(f"Unknown task {task_key!r}; known: {sorted(TASK_REGISTRY)}")
    dotted, _ = TASK_REGISTRY[task_key]
    module_path, _, cls_name = dotted.rpartition(".")
    import importlib
    return getattr(importlib.import_module(module_path), cls_name)


def _build_env(task_key: str, disruption: str, mode: str, motion_clips: Sequence[str]):
    """Construct ``BodySLAMWrapper(SafetyBiGymEnv(...))`` for one cell."""
    if not AMASS_DATA_DIR:
        raise RuntimeError(
            "AMASS_DATA_DIR is not set. Export it before running:\n"
            "  export AMASS_DATA_DIR=/path/to/CMU/CMU"
        )

    from safety_bigym import SafetyConfig, HumanConfig, make_safety_env
    from safety_bigym.perception.bodyslam_wrapper import BodySLAMWrapper
    from safety_bigym.scenarios.disruption_types import DisruptionType
    from safety_bigym.scenarios.scenario_sampler import ParameterSpace, ScenarioSampler
    from bigym.action_modes import JointPositionActionMode

    task_cls = _import_task(task_key)
    human_config = HumanConfig(
        motion_clip_dir=AMASS_DATA_DIR,
        motion_clip_paths=list(motion_clips),
    )
    sampler = ScenarioSampler(
        parameter_space=ParameterSpace(
            clip_paths=human_config.motion_clip_paths,
            disruption_weights={DisruptionType[disruption]: 1.0},
        ),
        motion_dir=AMASS_DATA_DIR,
    )
    env = make_safety_env(
        task_cls=task_cls,
        action_mode=JointPositionActionMode(absolute=True, floating_base=True),
        safety_config=SafetyConfig(terminate_on_violation=False),
        human_config=human_config,
        scenario_sampler=sampler,
        inject_human=True,
    )
    env = BodySLAMWrapper(env, mode=mode)
    return env


# ---------- policies ----------------------------------------------------------


def random_policy(env, rng: np.random.Generator) -> Callable[[dict], np.ndarray]:
    space = env.action_space
    low = space.low.astype(np.float32)
    high = space.high.astype(np.float32)

    def _act(_obs: dict) -> np.ndarray:
        return rng.uniform(low=low, high=high).astype(np.float32)

    return _act


def make_policy(source: str, env, rng: np.random.Generator, plan: CollectionPlan):
    if source == "random":
        return random_policy(env, rng)
    if source == "snapshot":
        raise NotImplementedError(
            "snapshot source is staged for the next commit on this sub-branch — "
            "needs the workspace.py snapshot-loading path; see baseline_sweep.py."
        )
    if source == "demo":
        raise NotImplementedError(
            "demo source is staged for the next commit on this sub-branch — "
            "needs DemoStore + replay machinery."
        )
    raise ValueError(f"Unknown source {source!r}; known: {sorted(SOURCE_CODES)}")


# ---------- rollout -----------------------------------------------------------


def _filter_obs(obs: dict, obs_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(obs[k], dtype=np.float32) for k in obs_keys}


def rollout_episode(
    env,
    policy: Callable[[dict], np.ndarray],
    *,
    max_steps: int,
    obs_keys: Sequence[str],
    use_pfl: bool,
):
    """Run one episode and return parallel arrays for one shard (or None if empty)."""
    from safety_bigym.filters.labeling import label_transition

    obs, _info = env.reset()
    obs_buf: Dict[str, List[np.ndarray]] = {k: [] for k in obs_keys}
    next_obs_buf: Dict[str, List[np.ndarray]] = {k: [] for k in obs_keys}
    actions: List[np.ndarray] = []
    r_safe_list: List[float] = []
    done_list: List[bool] = []
    margins: List[float] = []

    for _ in range(max_steps):
        action = policy(obs).astype(np.float32, copy=False)
        prev = _filter_obs(obs, obs_keys)

        obs, _reward, terminated, truncated, info = env.step(action)
        nxt = _filter_obs(obs, obs_keys)

        if "safety" not in info:
            # Defensive: BodySLAMWrapper passes info through; this should not
            # happen for SafetyBiGymEnv. Skip the transition rather than crash.
            continue

        r_safe, viol_terminal = label_transition(info, use_pfl=use_pfl)
        for k in obs_keys:
            obs_buf[k].append(prev[k])
            next_obs_buf[k].append(nxt[k])
        actions.append(action)
        r_safe_list.append(r_safe)
        done_list.append(bool(viol_terminal or terminated or truncated))
        margins.append(float(info["safety"].get("ssm_margin", 0.0)))

        if terminated or truncated:
            break

    if not actions:
        return None

    return {
        "obs": {k: np.stack(v, axis=0) for k, v in obs_buf.items()},
        "next_obs": {k: np.stack(v, axis=0) for k, v in next_obs_buf.items()},
        "action": np.stack(actions, axis=0),
        "r_safe": np.asarray(r_safe_list, dtype=np.float32),
        "done": np.asarray(done_list, dtype=np.bool_),
        "ssm_margin": np.asarray(margins, dtype=np.float32),
    }


# ---------- main orchestration -----------------------------------------------


def run_collection(plan: CollectionPlan, *, use_pfl: bool = False) -> Path:
    from gymnasium import spaces
    from safety_bigym.filters.dataset import TransitionShardWriter
    from safety_bigym.filters.feature_extractor import CriticFeatureSpec

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(plan.seed)
    spec: Optional[CriticFeatureSpec] = None
    writer: Optional[TransitionShardWriter] = None
    shard_idx = 0
    total = 0

    for source in plan.sources:
        source_code = SOURCE_CODES[source]
        for task_key in plan.tasks:
            task_id = TASK_REGISTRY[task_key][1]
            for disruption in plan.disruptions:
                logger.info(
                    f"Collecting source={source} task={task_key} disruption={disruption} "
                    f"({plan.episodes_per_cell} episodes, max {plan.max_steps} steps)"
                )
                env = _build_env(task_key, disruption, plan.bodyslam_mode, plan.motion_clips)
                policy = make_policy(source, env, rng, plan)

                if spec is None:
                    spec = CriticFeatureSpec.from_spaces(
                        env.observation_space, env.action_space
                    )
                    writer = TransitionShardWriter(spec, plan.output_dir)
                    logger.info(
                        f"Critic feature spec frozen: input_dim={spec.input_dim} "
                        f"(obs_keys={spec.obs_keys})"
                    )

                for _ep in range(plan.episodes_per_cell):
                    payload = rollout_episode(
                        env,
                        policy,
                        max_steps=plan.max_steps,
                        obs_keys=spec.obs_keys,
                        use_pfl=use_pfl,
                    )
                    if payload is None:
                        continue
                    n = len(payload["action"])
                    name = f"{source}__{task_key}__{disruption}__{shard_idx:04d}"
                    if plan.episodes_per_cell == 2 and plan.max_steps == 50:
                        name = "_smoke_shard"  # smoke mode: single named shard
                    writer.write_shard(
                        name=name,
                        obs=payload["obs"],
                        action=payload["action"],
                        next_obs=payload["next_obs"],
                        r_safe=payload["r_safe"],
                        done=payload["done"],
                        ssm_margin=payload["ssm_margin"],
                        source=np.full(n, source_code, dtype=np.uint8),
                        task_id=np.full(n, task_id, dtype=np.uint8),
                    )
                    shard_idx += 1
                    total += n

    if total == 0:
        raise RuntimeError(
            "Collection produced 0 transitions; check task / disruption / "
            "AMASS clip availability."
        )
    logger.info(f"Wrote {total} transitions to {plan.output_dir}")
    return plan.output_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="≤200-transition smoke run.")
    p.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCE_CODES),
        help="Repeatable. Defaults to ['random'].",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_REGISTRY)[:1],
        choices=sorted(TASK_REGISTRY),
    )
    p.add_argument("--disruptions", nargs="+", default=list(DEFAULT_DISRUPTIONS))
    p.add_argument("--episodes-per-cell", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument(
        "--bodyslam-mode", choices=("oracle", "noisy"), default="oracle"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "svf_v1",
    )
    p.add_argument("--snapshot-path", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-pfl", action="store_true",
                   help="Set once the PFL contact-detection bug is fixed.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.smoke:
        plan = CollectionPlan.smoke(args.output_dir)
    else:
        plan = CollectionPlan(
            sources=tuple(args.source or ("random",)),
            tasks=tuple(args.tasks),
            disruptions=tuple(args.disruptions),
            episodes_per_cell=args.episodes_per_cell,
            max_steps=args.max_steps,
            bodyslam_mode=args.bodyslam_mode,
            output_dir=args.output_dir,
            snapshot_path=args.snapshot_path,
            seed=args.seed,
        )

    run_collection(plan, use_pfl=args.use_pfl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

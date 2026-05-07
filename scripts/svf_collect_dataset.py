#!/usr/bin/env python
"""SVF transition dataset collector.

Phase 2 — collects ``(obs, action, next_obs, r_safe, done, ssm_margin, ...)``
transitions from rollouts in ``SafetyBiGymEnv + BodySLAMWrapper`` and writes
them as sharded ``.npz`` for the offline CQL trainer.

Sources (CLI flag, repeatable):
  ``random``    — uniform actions sampled from ``env.action_space``
  ``demo``      — replay BiGym demonstrations via DemoStore; ``human_pos_estimate``
                  is synthesised from an AMASS clip (no live human in the recorded
                  demo). All transitions labelled ``r_safe=1`` (demos are safe by
                  construction; ``ssm_margin`` is set to a constant placeholder).
  ``snapshot``  — load an ACT/DP snapshot (workspace.py drift) and roll it out.
                  Requires --snapshot-path. Phase-0 ACT snapshots are pending
                  GPU retrain; until then this path is exercised only in error
                  branches by the smoke test.

Smoke mode runs 1 task × 1 disruption × 2 episodes × ≤50 steps from the random
source and writes a single ``_smoke_shard.npz``.

Usage:
    python scripts/svf_collect_dataset.py --smoke
    python scripts/svf_collect_dataset.py --source random --source demo \\
        --tasks reach_target_single --episodes-per-cell 50
    python scripts/svf_collect_dataset.py --source snapshot \\
        --snapshot-path exp_local/act_safety/.../snapshots/60000_snapshot.pt

Hand-off to GPU for the full ~500k-transition collection.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("svf_collect_dataset")


AMASS_DATA_DIR = os.environ.get("AMASS_DATA_DIR")


SOURCE_CODES: Dict[str, int] = {"demo": 0, "random": 1, "snapshot": 2}

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

# Demos are recorded without a live human, so we synthesise per-demo human
# pelvis positions from an AMASS clip. Constant safe placeholder margin since
# the recorded trajectories never touch the human.
DEMO_PLACEHOLDER_MARGIN = 1.0
DEMO_HUMAN_OCCLUDED = 0.0
DEMO_HUMAN_STALENESS = 0.0
DEMO_HUMAN_CONFIDENCE = 1.0


@dataclass
class CollectionPlan:
    sources: Tuple[str, ...]
    tasks: Tuple[str, ...]
    disruptions: Tuple[str, ...]
    episodes_per_cell: int
    max_steps: int
    bodyslam_mode: str
    output_dir: Path
    seed: int = 0
    motion_clips: Tuple[str, ...] = DEFAULT_CLIPS
    demos_per_task: int = 30  # used only by the demo source
    # Per-task snapshot path overrides — populated by --snapshot-override CLI
    # flags. Empty by default; the resolver falls through to
    # safety_bigym.filters.snapshots.SNAPSHOTS.
    snapshot_overrides: Mapping[str, str] = field(default_factory=dict)

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


def _build_live_env(
    task_key: str, disruption: str, mode: str, motion_clips: Sequence[str]
):
    """Construct ``BodySLAMWrapper(SafetyBiGymEnv(...))`` — used by random/snapshot."""
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


# ---------- random policy -----------------------------------------------------


def random_policy(env, rng: np.random.Generator) -> Callable[[dict], np.ndarray]:
    space = env.action_space
    low = space.low.astype(np.float32)
    high = space.high.astype(np.float32)

    def _act(_obs: dict) -> np.ndarray:
        return rng.uniform(low=low, high=high).astype(np.float32)

    return _act


# ---------- live-env rollout (random + snapshot share this path) --------------


def _filter_obs(obs: dict, obs_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(obs[k], dtype=np.float32) for k in obs_keys}


def rollout_episode(
    env,
    policy: Callable[[dict], np.ndarray],
    *,
    max_steps: int,
    obs_keys: Sequence[str],
    use_pfl: bool,
) -> Optional[Dict[str, Any]]:
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


# ---------- demo source -------------------------------------------------------


def _build_human_pos_estimate(pelvis_xyz: np.ndarray) -> np.ndarray:
    """Construct the 6-D ``human_pos_estimate`` vector for demo replay.

    Mirrors ``BodySLAMWrapper._build_estimate``; demos have no occlusion or
    staleness, so the auxiliary fields are pinned at their ``oracle`` defaults.
    """
    return np.array(
        [
            float(pelvis_xyz[0]),
            float(pelvis_xyz[1]),
            float(pelvis_xyz[2]),
            DEMO_HUMAN_OCCLUDED,
            DEMO_HUMAN_STALENESS,
            DEMO_HUMAN_CONFIDENCE,
        ],
        dtype=np.float32,
    )


def _make_amass_provider(motion_clips: Sequence[str], seed: int):
    from safety_bigym.perception.demo_position_provider import AMASSDemoPositionProvider

    if not AMASS_DATA_DIR:
        raise RuntimeError(
            "AMASS_DATA_DIR is not set. Export it before running:\n"
            "  export AMASS_DATA_DIR=/path/to/CMU/CMU"
        )
    return AMASSDemoPositionProvider(
        clip_paths=list(motion_clips),
        motion_dir=AMASS_DATA_DIR,
        seed=int(seed),
    )


def fetch_demos(task_key: str, num_demos: int, frequency: int = 50):
    """Load BiGym demos for one task via ``DemoStore``.

    Returns the list of ``Demo`` objects (each with ``.timesteps``). The env is
    constructed only for ``Metadata.from_env`` and closed immediately after.

    If the store has no demos matching this task's metadata signature, returns
    an empty list and logs a warning (the caller skips the task rather than
    crashing the whole collection run).
    """
    from demonstrations.demo_store import DemoStore, DemoNotFoundError
    from demonstrations.utils import Metadata
    from bigym.action_modes import JointPositionActionMode

    task_cls = _import_task(task_key)
    env = task_cls(
        action_mode=JointPositionActionMode(absolute=True, floating_base=True),
    )
    try:
        store = DemoStore()
        try:
            demos = store.get_demos(
                Metadata.from_env(env),
                amount=num_demos,
                frequency=frequency,
            )
        except DemoNotFoundError as e:
            logger.warning(
                f"DemoStore has no demos matching {task_key} metadata "
                f"(action_mode=JointPositionActionMode, floating_base=True). "
                f"Skipping demo source for this task. "
                f"Underlying error: {e}"
            )
            return []
        for demo in demos:
            for ts in demo.timesteps:
                ts.observation = {
                    k: np.array(v, dtype=np.float32)
                    for k, v in ts.observation.items()
                }
        return demos
    finally:
        env.close()


def demo_episode_to_transitions(
    timesteps: Sequence[Any],
    *,
    spec_obs_keys: Sequence[str],
    amass_provider,
) -> Optional[Dict[str, np.ndarray]]:
    """Turn one demo's timesteps into the per-shard array bundle.

    Demos contain ``(observation, action)`` pairs but no live safety physics —
    so we set ``r_safe = 1`` and ``ssm_margin = DEMO_PLACEHOLDER_MARGIN`` for
    every transition. ``human_pos_estimate`` is synthesised by the AMASS
    provider so the channel is non-degenerate.
    """
    if len(timesteps) < 2:
        return None

    amass_provider.reset()
    obs_buf: Dict[str, List[np.ndarray]] = {k: [] for k in spec_obs_keys}
    next_obs_buf: Dict[str, List[np.ndarray]] = {k: [] for k in spec_obs_keys}
    actions: List[np.ndarray] = []

    for step_idx, (ts, next_ts) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        # ``DemoStep.executed_action`` reads from ``info[ACTION_KEY]``; the
        # bare ``.action`` attribute does not exist (see bigym/demonstrations/demo.py).
        raw_action = ts.executed_action
        if raw_action is None:
            return None
        action = np.asarray(raw_action, dtype=np.float32)
        prev_obs = dict(ts.observation)
        next_obs = dict(next_ts.observation)
        prev_obs["human_pos_estimate"] = _build_human_pos_estimate(
            amass_provider(step_idx)
        )
        next_obs["human_pos_estimate"] = _build_human_pos_estimate(
            amass_provider(step_idx + 1)
        )
        for k in spec_obs_keys:
            if k not in prev_obs:
                # Demo obs is missing a key the spec expects — skip this episode
                # rather than corrupting the dataset.
                logger.warning(
                    f"Demo timestep missing obs key {k!r}; skipping episode."
                )
                return None
            obs_buf[k].append(np.asarray(prev_obs[k], dtype=np.float32))
            next_obs_buf[k].append(np.asarray(next_obs[k], dtype=np.float32))
        actions.append(action)

    n = len(actions)
    return {
        "obs": {k: np.stack(v, axis=0) for k, v in obs_buf.items()},
        "next_obs": {k: np.stack(v, axis=0) for k, v in next_obs_buf.items()},
        "action": np.stack(actions, axis=0),
        "r_safe": np.ones(n, dtype=np.float32),
        "done": np.concatenate([np.zeros(n - 1, dtype=np.bool_), [True]])
        if n > 0
        else np.zeros(0, dtype=np.bool_),
        "ssm_margin": np.full(n, DEMO_PLACEHOLDER_MARGIN, dtype=np.float32),
    }


# ---------- snapshot source ---------------------------------------------------


@dataclass
class _SnapshotPolicy:
    """Wraps a RoboBase-loaded agent into a policy callable.

    The agent expects post-wrap observations (``low_dim_state``, ``rgb_*``);
    the bare env emits unflattened proprio keys. ``adapt_obs`` pre-flattens
    proprio into ``low_dim_state`` so the actor sees its training-time format.
    Pixel inputs are zero-filled when the bare env has no cameras — the
    actor's actions are useless under that regime, which is why this path is
    only meaningful when the snapshot was trained pixel-free.
    """

    agent: Any
    expects_pixels: bool
    image_shape: Optional[Tuple[int, int, int]] = None

    def adapt_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        import torch

        proprio_keys = sorted(
            k for k in obs.keys() if k.startswith("proprioception")
        )
        if proprio_keys:
            low_dim = np.concatenate(
                [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in proprio_keys]
            )
        else:
            low_dim = np.asarray(obs.get("low_dim_state", np.zeros(0)), dtype=np.float32)

        out: Dict[str, Any] = {
            "low_dim_state": torch.from_numpy(low_dim).unsqueeze(0).unsqueeze(0)
        }
        if self.expects_pixels:
            shape = self.image_shape or (3, 84, 84)
            out["rgb_head"] = torch.zeros(1, 1, *shape)
        return out

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        import torch

        with torch.no_grad():
            adapted = self.adapt_obs(obs)
            action = self.agent.act(adapted, step=0, eval_mode=True)
        action_np = action.detach().cpu().numpy()
        # ACT returns a chunk; take the first step.
        if action_np.ndim >= 2:
            action_np = action_np.reshape(-1, action_np.shape[-1])[0]
        return action_np.astype(np.float32, copy=False)


def load_snapshot_policy(snapshot_path: Path, env) -> _SnapshotPolicy:
    """Load a RoboBase snapshot and return a policy callable.

    Builds the agent via ``hydra.utils.instantiate`` from the cfg embedded in
    the payload. EMA shadow params are restored explicitly for ACT (see
    workspace.py drift bullet 4 in CLAUDE.md). Pure CPU; no W&B.
    """
    if not snapshot_path or not Path(snapshot_path).is_file():
        raise FileNotFoundError(
            f"snapshot_path={snapshot_path!r} not found. Phase-0 ACT snapshots are "
            "still pending GPU retrain — see CLAUDE.md."
        )

    import hydra
    import torch
    from omegaconf import DictConfig, OmegaConf

    payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    cfg = payload.get("cfg")
    if cfg is None:
        raise KeyError(
            f"snapshot at {snapshot_path} has no 'cfg' field — was it produced "
            "after the workspace.py drift fix? See CLAUDE.md."
        )
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # Minimum cfg fields the agent constructor reads.
    method_cfg = cfg.method
    intrinsic_reward_module = None
    agent = hydra.utils.instantiate(
        method_cfg,
        device="cpu",
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_train_envs=0,
        replay_alpha=cfg.replay.alpha,
        replay_beta=cfg.replay.beta,
        frame_stack_on_channel=cfg.frame_stack_on_channel,
        intrinsic_reward_module=intrinsic_reward_module,
    )
    agent.load_state_dict(payload["agent"], strict=False)
    if "actor_ema" in payload and hasattr(agent, "actor"):
        actor_ema = payload["actor_ema"]
        if hasattr(agent.actor, "ema") and hasattr(agent.actor.ema, "shadow_params"):
            for p, sp in zip(agent.actor.ema.shadow_params, actor_ema):
                p.data.copy_(sp)
    agent.train(False)

    expects_pixels = bool(cfg.get("pixels", False))
    image_shape: Optional[Tuple[int, int, int]] = None
    if expects_pixels:
        shape = cfg.get("visual_observation_shape")
        if shape is not None:
            image_shape = (3, int(shape[0]), int(shape[1]))
    return _SnapshotPolicy(agent=agent, expects_pixels=expects_pixels, image_shape=image_shape)


# ---------- main orchestration ------------------------------------------------


def _smoke_shard_name(plan: CollectionPlan) -> Optional[str]:
    if plan.episodes_per_cell == 2 and plan.max_steps == 50:
        return "_smoke_shard"
    return None


def _collect_live_env_source(
    plan: CollectionPlan,
    *,
    source: str,
    rng: np.random.Generator,
    spec,
    writer,
    use_pfl: bool,
) -> Tuple[int, int]:
    """Random / snapshot collection — both use a live env + a policy callable.

    For ``source="snapshot"`` the per-task path is resolved via
    :func:`safety_bigym.filters.snapshots.resolve_snapshot`; tasks whose
    SNAPSHOTS entry is ``None`` are skipped with a warning.
    """
    from safety_bigym.filters.feature_extractor import CriticFeatureSpec
    from safety_bigym.filters.dataset import TransitionShardWriter
    from safety_bigym.filters.snapshots import resolve_snapshot

    source_code = SOURCE_CODES[source]
    shard_idx = 0
    total = 0

    for task_key in plan.tasks:
        task_id = TASK_REGISTRY[task_key][1]

        # Snapshot path is per-task — resolve before doing any heavy env setup.
        snapshot_path: Optional[Path] = None
        if source == "snapshot":
            snapshot_path = resolve_snapshot(
                task_key, overrides=plan.snapshot_overrides
            )
            if snapshot_path is None:
                logger.warning(
                    f"No snapshot configured for task {task_key!r} "
                    "(SNAPSHOTS entry is None and no --snapshot-override given). "
                    "Skipping snapshot source for this task."
                )
                continue

        for disruption in plan.disruptions:
            logger.info(
                f"Collecting source={source} task={task_key} disruption={disruption} "
                f"({plan.episodes_per_cell} episodes, max {plan.max_steps} steps)"
            )
            env = _build_live_env(
                task_key, disruption, plan.bodyslam_mode, plan.motion_clips
            )

            if source == "random":
                policy = random_policy(env, rng)
            elif source == "snapshot":
                policy = load_snapshot_policy(snapshot_path, env)
            else:
                raise ValueError(f"_collect_live_env_source got source={source!r}")

            # Spec is built off the first env we construct so the obs key set
            # is consistent across all sources.
            if spec[0] is None:
                spec[0] = CriticFeatureSpec.from_spaces(
                    env.observation_space, env.action_space
                )
                writer[0] = TransitionShardWriter(spec[0], plan.output_dir)
                logger.info(
                    f"Critic feature spec frozen: input_dim={spec[0].input_dim} "
                    f"(obs_keys={spec[0].obs_keys})"
                )

            for _ep in range(plan.episodes_per_cell):
                payload = rollout_episode(
                    env,
                    policy,
                    max_steps=plan.max_steps,
                    obs_keys=spec[0].obs_keys,
                    use_pfl=use_pfl,
                )
                if payload is None:
                    continue
                n = len(payload["action"])
                name = _smoke_shard_name(plan) or (
                    f"{source}__{task_key}__{disruption}__{shard_idx:04d}"
                )
                writer[0].write_shard(
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

    return shard_idx, total


def _collect_demo_source(
    plan: CollectionPlan,
    *,
    spec,
    writer,
) -> Tuple[int, int]:
    """Load BiGym demos once per task and lay them out as transitions.

    Demos do not require a live env or any disruption iteration — they are
    pre-recorded action sequences. ``demos_per_task`` controls the demo count;
    each demo becomes a single shard.
    """
    from safety_bigym.filters.feature_extractor import CriticFeatureSpec
    from safety_bigym.filters.dataset import TransitionShardWriter

    if spec[0] is None:
        # Demos can't define the spec on their own (they don't expose
        # action_space.low/high in a useful way for sampling). Require either
        # demo to come AFTER a live source, or — for demo-only collection —
        # build a probe env once.
        probe = _build_live_env(
            plan.tasks[0],
            plan.disruptions[0] if plan.disruptions else "INCIDENTAL",
            plan.bodyslam_mode,
            plan.motion_clips,
        )
        spec[0] = CriticFeatureSpec.from_spaces(
            probe.observation_space, probe.action_space
        )
        writer[0] = TransitionShardWriter(spec[0], plan.output_dir)
        probe.close() if hasattr(probe, "close") else None
        logger.info(
            f"Critic feature spec frozen via demo probe: input_dim={spec[0].input_dim} "
            f"(obs_keys={spec[0].obs_keys})"
        )

    shard_idx = 0
    total = 0
    source_code = SOURCE_CODES["demo"]

    for task_key in plan.tasks:
        task_id = TASK_REGISTRY[task_key][1]
        logger.info(
            f"Collecting source=demo task={task_key} "
            f"(up to {plan.demos_per_task} demos)"
        )
        amass_provider = _make_amass_provider(plan.motion_clips, plan.seed ^ 0xD)
        try:
            demos = fetch_demos(task_key, plan.demos_per_task)
        except Exception as e:  # noqa: BLE001 — surface DemoStore failures clearly
            logger.error(f"DemoStore fetch failed for {task_key}: {e}")
            raise
        if not demos:
            logger.warning(
                f"No demos available for {task_key}; skipping demo source for this task."
            )
            continue

        for demo_idx, demo in enumerate(demos):
            payload = demo_episode_to_transitions(
                demo.timesteps,
                spec_obs_keys=spec[0].obs_keys,
                amass_provider=amass_provider,
            )
            if payload is None:
                continue
            n = len(payload["action"])
            name = _smoke_shard_name(plan) or f"demo__{task_key}__{demo_idx:04d}"
            writer[0].write_shard(
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

    return shard_idx, total


def run_collection(plan: CollectionPlan, *, use_pfl: bool = False) -> Path:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(plan.seed)

    # Mutable wrappers so the per-source helpers can lazy-init the spec/writer.
    spec_holder: List[Any] = [None]
    writer_holder: List[Any] = [None]
    total = 0

    for source in plan.sources:
        if source == "demo":
            _, n = _collect_demo_source(plan, spec=spec_holder, writer=writer_holder)
        elif source in ("random", "snapshot"):
            _, n = _collect_live_env_source(
                plan,
                source=source,
                rng=rng,
                spec=spec_holder,
                writer=writer_holder,
                use_pfl=use_pfl,
            )
        else:
            raise ValueError(f"Unknown source {source!r}; known: {sorted(SOURCE_CODES)}")
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
    p.add_argument("--demos-per-task", type=int, default=30)
    p.add_argument("--bodyslam-mode", choices=("oracle", "noisy"), default="oracle")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "svf_v1",
    )
    p.add_argument(
        "--snapshot-override",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help=(
            "Override a snapshot path for one task; takes precedence over the "
            "SNAPSHOTS dict in safety_bigym/filters/snapshots.py. Repeatable."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-pfl", action="store_true",
                   help="Set once the PFL contact-detection bug is fixed.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    snapshot_overrides = _parse_snapshot_overrides(args.snapshot_override)

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
            snapshot_overrides=snapshot_overrides,
            seed=args.seed,
            demos_per_task=args.demos_per_task,
        )

    run_collection(plan, use_pfl=args.use_pfl)
    return 0


def _parse_snapshot_overrides(raw: Sequence[str]) -> Dict[str, str]:
    """Turn ``--snapshot-override TASK=PATH`` flags into a mapping."""
    out: Dict[str, str] = {}
    for entry in raw:
        if "=" not in entry:
            raise SystemExit(
                f"--snapshot-override expects TASK=PATH; got {entry!r}"
            )
        task, _, path = entry.partition("=")
        if not task or not path:
            raise SystemExit(
                f"--snapshot-override expects TASK=PATH; got {entry!r}"
            )
        out[task] = path
    return out


if __name__ == "__main__":
    raise SystemExit(main())

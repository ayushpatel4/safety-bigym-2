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

Disruption space (Phase 0.5 onward):
  Default ``--disruption-space coworker_train`` collects a single cell using
  ``make_coworker_train_space()`` — the strict-superset COWORKER ParameterSpace
  with five continuous parameter axes (closest approach, reach period, target
  mix, near loiter, walk speed) in moderate bands. ``coworker_eval`` is the
  wider eval ParameterSpace; ``legacy_multi`` reinstates the pre-0.5 mixture
  of {INCIDENTAL, SHARED_GOAL, DIRECT, OBSTRUCTION, RANDOM_PERTURBED} and
  iterates over the --disruptions string list.

Smoke mode runs 1 task × coworker_train × 2 episodes × ≤50 steps from the
random source and writes a single ``_smoke_shard.npz``.

Usage:
    python scripts/svf_collect_dataset.py --smoke
    python scripts/svf_collect_dataset.py --source random --source demo \\
        --tasks reach_target_single --episodes-per-cell 50
    # legacy escape hatch:
    python scripts/svf_collect_dataset.py --disruption-space legacy_multi \\
        --disruptions INCIDENTAL DIRECT
    python scripts/svf_collect_dataset.py --source snapshot \\
        --snapshot-path exp_local/act_safety/.../snapshots/60000_snapshot.pt

Hand-off to GPU for the full ~310k-transition collection.
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
    "drawers_open_all": ("bigym.envs.cupboards.DrawersAllOpen", 4),
}

DEFAULT_DISRUPTIONS = (
    "INCIDENTAL",
    "SHARED_GOAL",
    "DIRECT",
    "OBSTRUCTION",
    "RANDOM_PERTURBED",
)

# Sentinel cell labels for the coworker ParameterSpace factories. When a
# disruption string is one of these, _build_live_env dispatches to
# make_coworker_{train,eval}_space() instead of looking up DisruptionType[name].
COWORKER_CELL_LABELS = ("coworker_train", "coworker_eval")

DISRUPTION_SPACE_CHOICES = ("coworker_train", "coworker_eval", "legacy_multi")

DEFAULT_CLIPS = ("74/74_01_poses.npz",)

# Keys ConcatDim excludes from low_dim_state at training time
# ([robobase/envs/bigym.py:95]). The snapshot adapter must replicate this
# exclusion exactly or the actor's first layer receives a wrong-dim vector.
LOW_DIM_KEYS_TO_IGNORE: Tuple[str, ...] = ("proprioception_floating_base_actions",)

HUMAN_POS_ESTIMATE_KEY = "human_pos_estimate"

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
    # Geometric near-contact threshold (metres) used by label_transition. The
    # SVF's binary r_safe = (min_separation >= proximity_threshold). Default
    # matches labeling.label_transition's own default; surface as a CLI knob
    # because it's the most likely thing to sweep.
    proximity_threshold: float = 0.10

    @classmethod
    def smoke(cls, output_dir: Path) -> "CollectionPlan":
        return cls(
            sources=("random",),
            tasks=("dishwasher_close",),
            disruptions=("coworker_train",),
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
    task_key: str,
    disruption: str,
    mode: str,
    motion_clips: Sequence[str],
    *,
    cameras: Sequence[str] = (),
    camera_resolution: Tuple[int, int] = (84, 84),
):
    """Construct ``[BodySLAMWrapper(]SafetyBiGymEnv[)]`` — used by random/snapshot.

    ``mode``: "oracle" | "noisy" wraps with BodySLAMWrapper; "off" skips the
    wrapper entirely so the env emits no ``human_pos_estimate`` (required for
    Phase 0 ACT snapshots, which were trained without BodySLAMWrapper —
    instantiating the agent against a space that contains the extra key
    sizes the input layer wrong and silent state_dict mismatches result).

    ``cameras`` defaults to empty (bare env, no rendering) so random/demo
    callers don't pay the MuJoCo render cost. Pixel-trained snapshot policies
    must pass the cameras list from their snapshot's ``cfg.env.cameras`` so
    the obs dict carries the ``rgb_<name>`` keys the actor's encoder expects.
    """
    if not AMASS_DATA_DIR:
        raise RuntimeError(
            "AMASS_DATA_DIR is not set. Export it before running:\n"
            "  export AMASS_DATA_DIR=/path/to/CMU/CMU"
        )

    from safety_bigym import SafetyConfig, HumanConfig, make_safety_env
    from safety_bigym.perception.bodyslam_wrapper import BodySLAMWrapper
    from safety_bigym.scenarios.disruption_types import DisruptionType
    from safety_bigym.scenarios.scenario_sampler import (
        ParameterSpace,
        ScenarioSampler,
        make_coworker_train_space,
        make_coworker_eval_space,
    )
    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from bigym.utils.observation_config import CameraConfig, ObservationConfig

    task_cls = _import_task(task_key)
    human_config = HumanConfig(
        motion_clip_dir=AMASS_DATA_DIR,
        motion_clip_paths=list(motion_clips),
    )
    # Cell label dispatch: "coworker_train" / "coworker_eval" use the strict-
    # superset COWORKER factories; any other value is treated as a legacy
    # DisruptionType name and pinned to a 1.0-weight single-type space.
    if disruption == "coworker_train":
        parameter_space = make_coworker_train_space(
            clip_paths=human_config.motion_clip_paths,
        )
    elif disruption == "coworker_eval":
        parameter_space = make_coworker_eval_space(
            clip_paths=human_config.motion_clip_paths,
        )
    else:
        parameter_space = ParameterSpace(
            clip_paths=human_config.motion_clip_paths,
            disruption_weights={DisruptionType[disruption]: 1.0},
        )
    sampler = ScenarioSampler(
        parameter_space=parameter_space,
        motion_dir=AMASS_DATA_DIR,
    )
    make_env_kwargs: Dict[str, Any] = {}
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
    # 4-dof floating base (X, Y, Z, RZ) mirrors RoboBase's BiGym factory under
    # `cfg.env.enable_all_floating_dof=True` — the regime the Phase-0 ACT
    # snapshots were trained under (action_dim=16, qpos=66). The bare-BiGym
    # default of 3 dofs (X, Y, RZ) gives action_dim=15 and silent
    # state_dict shape mismatches at snapshot load. See B1.4 / B2.3 debug.
    env = make_safety_env(
        task_cls=task_cls,
        action_mode=JointPositionActionMode(
            absolute=True,
            floating_base=True,
            floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
        ),
        safety_config=SafetyConfig(terminate_on_violation=False),
        human_config=human_config,
        scenario_sampler=sampler,
        inject_human=True,
        **make_env_kwargs,
    )
    if mode == "off":
        return env
    if mode not in ("oracle", "noisy"):
        raise ValueError(
            f"_build_live_env: bodyslam mode must be one of off/oracle/noisy; "
            f"got {mode!r}"
        )
    return BodySLAMWrapper(env, mode=mode)


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
    proximity_threshold: float,
) -> Optional[Dict[str, Any]]:
    from safety_bigym.filters.labeling import label_transition

    obs, _info = env.reset()
    obs_buf: Dict[str, List[np.ndarray]] = {k: [] for k in obs_keys}
    next_obs_buf: Dict[str, List[np.ndarray]] = {k: [] for k in obs_keys}
    actions: List[np.ndarray] = []
    r_safe_list: List[float] = []
    done_list: List[bool] = []
    margins: List[float] = []
    min_seps: List[float] = []
    pfl_ratios: List[float] = []

    for _ in range(max_steps):
        action = policy(obs).astype(np.float32, copy=False)
        prev = _filter_obs(obs, obs_keys)

        obs, _reward, terminated, truncated, info = env.step(action)
        nxt = _filter_obs(obs, obs_keys)

        if "safety" not in info:
            continue

        r_safe, viol_terminal = label_transition(
            info, use_pfl=use_pfl, proximity_threshold=proximity_threshold,
        )
        for k in obs_keys:
            obs_buf[k].append(prev[k])
            next_obs_buf[k].append(nxt[k])
        actions.append(action)
        r_safe_list.append(r_safe)
        done_list.append(bool(viol_terminal or terminated or truncated))
        safety_info = info["safety"]
        margins.append(float(safety_info.get("ssm_margin", 0.0)))
        # Store the raw signals so r_safe can be recomputed later (proximity
        # threshold sweep, PFL retrofit) without re-collecting transitions.
        # ``pfl_force_ratio`` is currently identically zero; see CLAUDE.md.
        min_seps.append(float(safety_info.get("min_separation", float("inf"))))
        pfl_ratios.append(float(safety_info.get("pfl_force_ratio", 0.0)))

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
        "min_separation": np.asarray(min_seps, dtype=np.float32),
        "pfl_force_ratio": np.asarray(pfl_ratios, dtype=np.float32),
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
        # Demos have no live human-robot physics; use a safe-side placeholder
        # large enough to never trip any plausible proximity threshold and a
        # zero PFL ratio (no contact recorded). Demo source is currently
        # disabled in B3 (see CLAUDE.md), but keep the schema consistent.
        "min_separation": np.full(n, 10.0, dtype=np.float32),
        "pfl_force_ratio": np.zeros(n, dtype=np.float32),
    }


# ---------- snapshot source ---------------------------------------------------


@dataclass
class _SnapshotPolicy:
    """Wraps a RoboBase-loaded agent into a policy callable.

    The agent expects post-wrap observations: ``low_dim_state`` (concat of
    1-D proprio keys via ConcatDim, optionally including
    ``human_pos_estimate`` when the policy was trained with BodySLAMWrapper)
    plus one ``rgb_<camera>`` torch tensor per camera in
    ``(B=1, T=1, C=3, H, W)`` shape. ``adapt_obs`` replicates ConcatDim's
    concatenation rule exactly (iterate obs in insertion order, skip keys
    in ``LOW_DIM_KEYS_TO_IGNORE``, gate ``human_pos_estimate`` on
    ``includes_human_pos``). The encoder applies ``/255`` + ImageNet
    normalize on pixels internally.

    ``includes_human_pos`` is set by ``load_snapshot_policy`` from the
    snapshot's ``cfg.env.bodyslam.mode``: oracle/noisy ⇒ True (Phase 1+),
    off/missing ⇒ False (Phase 0). Phase 0 snapshots get a shorter
    low_dim_state that omits the 6-D human pose estimate; Phase 1
    snapshots include it.
    """

    agent: Any
    cameras: Tuple[str, ...] = ()  # empty ⇒ no-pixel policy
    camera_resolution: Tuple[int, int] = (84, 84)
    includes_human_pos: bool = False

    @property
    def expects_pixels(self) -> bool:
        return len(self.cameras) > 0

    def adapt_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        import torch

        # Replicate ConcatDim: iterate obs in insertion order (matches the
        # env's emit order, which is what ConcatDim sees at training), skip
        # keys in LOW_DIM_KEYS_TO_IGNORE, gate human_pos_estimate on the
        # policy's bodyslam-trained flag, and concat all remaining 1-D
        # values in order.
        low_dim_pieces: list[np.ndarray] = []
        for key, value in obs.items():
            if key in LOW_DIM_KEYS_TO_IGNORE:
                continue
            if key == HUMAN_POS_ESTIMATE_KEY and not self.includes_human_pos:
                continue
            arr = np.asarray(value)
            if arr.ndim != 1:
                continue  # pixels and other multi-D keys go through their own paths
            low_dim_pieces.append(arr.astype(np.float32, copy=False).reshape(-1))

        if not low_dim_pieces:
            # Fall back to a pre-concat'd key if the env was already wrapped
            # (e.g. shipped post-ConcatDim form). Rare; mostly defensive.
            low_dim_pieces.append(
                np.asarray(obs.get("low_dim_state", np.zeros(0)), dtype=np.float32).reshape(-1)
            )
        low_dim = np.concatenate(low_dim_pieces)

        out: Dict[str, Any] = {
            "low_dim_state": torch.from_numpy(low_dim).unsqueeze(0).unsqueeze(0)
        }
        for cam in self.cameras:
            key = f"rgb_{cam}"
            if key not in obs:
                raise KeyError(
                    f"Snapshot policy expects obs key {key!r} but it's missing. "
                    f"Did you build the env with cameras={list(self.cameras)!r}?"
                )
            arr = np.asarray(obs[key])
            # Bare bigym emits (H, W, 3) uint8. Permute HWC → CHW, add B + T.
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.ndim == 3 and arr.shape[0] == 3:
                pass  # already CHW
            else:
                raise ValueError(
                    f"Unexpected pixel shape for {key!r}: got {arr.shape}, "
                    "expected (H, W, 3) or (3, H, W)."
                )
            # ACT's encoder reads uint8 and applies /255 + ImageNet normalize.
            out[key] = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0)
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


def _peek_snapshot_cfg(snapshot_path: Path):
    """Load just the cfg field out of a snapshot payload."""
    if not snapshot_path or not Path(snapshot_path).is_file():
        raise FileNotFoundError(f"snapshot_path={snapshot_path!r} not found.")
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
    return cfg


def peek_snapshot_bodyslam_mode(snapshot_path: Path) -> str:
    """Return the BodySLAM mode the snapshot was trained with.

    ``"off"`` ⇒ Phase 0 ACT (no human-state observation). ``"oracle"`` or
    ``"noisy"`` ⇒ Phase 1+ ACT (trained with BodySLAMWrapper). Determines
    whether ``_SnapshotPolicy.adapt_obs`` should append ``human_pos_estimate``
    to ``low_dim_state`` AND what BodySLAM mode the eval env should use so
    the input noise distribution matches training.
    """
    cfg = _peek_snapshot_cfg(snapshot_path)
    bs = cfg.env.get("bodyslam") if "env" in cfg else None
    if bs is None:
        return "off"
    mode = str(bs.get("mode", "off"))
    if mode not in ("off", "oracle", "noisy"):
        raise ValueError(
            f"Unexpected bodyslam.mode={mode!r} in snapshot {snapshot_path}; "
            "expected one of off/oracle/noisy."
        )
    return mode


def peek_snapshot_cameras(
    snapshot_path: Path,
) -> Tuple[Tuple[str, ...], Tuple[int, int]]:
    """Read just the camera config out of a snapshot's embedded cfg.

    Returns ``(camera_names, (H, W))``. ``camera_names`` is empty for
    non-pixel snapshots. Used by ``_collect_live_env_source`` to build a
    camera-equipped env before fully loading the agent.
    """
    cfg = _peek_snapshot_cfg(snapshot_path)
    if not bool(cfg.get("pixels", False)):
        return (), (0, 0)
    cameras = tuple(str(c) for c in cfg.env.get("cameras", []))
    shape = cfg.get("visual_observation_shape")
    if shape is None:
        resolution = (84, 84)
    else:
        resolution = (int(shape[0]), int(shape[1]))
    return cameras, resolution


def _synthesize_snapshot_obs_space(
    env,
    cameras: Sequence[str],
    resolution: Tuple[int, int],
    includes_human_pos: bool,
):
    """Build the post-wrap observation_space the snapshot agent was trained against.

    The env we hand to ``hydra.utils.instantiate`` must look like what RoboBase's
    BiGym factory produced at training: ``ConcatDim(shape_length=1)`` collapses
    all 1-D obs keys into a single ``low_dim_state`` channel (skipping
    ``LOW_DIM_KEYS_TO_IGNORE``), and ``FrameStack(frame_stack=1)`` prepends a
    ``T=1`` axis to every remaining key (so rgb becomes ``(T, C, H, W)`` —
    [robobase/method/bc.py:155] multiplies the first two dims, asserting 4-D).

    For Phase-0 snapshots (``includes_human_pos=False``) we omit
    ``human_pos_estimate`` from the low_dim_state sum so the synthesized dim
    matches the actor's training-time input size, even though the wrapped env
    *does* emit the key (the runtime adapter strips it via the same flag).
    """
    from gymnasium import spaces

    low_dim_total = 0
    for key, space in env.observation_space.spaces.items():
        if not isinstance(space, spaces.Box):
            continue
        if len(space.shape) != 1:
            continue
        if key in LOW_DIM_KEYS_TO_IGNORE:
            continue
        if key == HUMAN_POS_ESTIMATE_KEY and not includes_human_pos:
            continue
        low_dim_total += int(space.shape[0])

    out: Dict[str, Any] = {}
    if low_dim_total > 0:
        out["low_dim_state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, low_dim_total),
            dtype=np.float32,
        )
    for cam in cameras:
        out[f"rgb_{cam}"] = spaces.Box(
            low=0,
            high=255,
            shape=(1, 3, int(resolution[0]), int(resolution[1])),
            dtype=np.uint8,
        )
    return spaces.Dict(out)


def load_snapshot_policy(snapshot_path: Path, env) -> _SnapshotPolicy:
    """Load a RoboBase snapshot and return a policy callable.

    Builds the agent via ``hydra.utils.instantiate`` from the cfg embedded in
    the payload. EMA shadow params are restored explicitly for ACT (see
    workspace.py drift bullet 4 in CLAUDE.md). Pure CPU; no W&B.

    The caller should build ``env`` with cameras matching what
    :func:`peek_snapshot_cameras` returned so the actor's encoder gets the
    pixel keys it was trained with.
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

    if int(cfg.get("frame_stack", 1)) != 1:
        raise NotImplementedError(
            "Snapshot policy adapter only supports frame_stack=1. "
            f"Snapshot's cfg.frame_stack={cfg.get('frame_stack')!r}. "
            "Extend _SnapshotPolicy.adapt_obs with a frame-stack deque first."
        )

    cameras: Tuple[str, ...] = ()
    resolution: Tuple[int, int] = (84, 84)
    if bool(cfg.get("pixels", False)):
        cameras = tuple(str(c) for c in cfg.env.get("cameras", []))
        shape = cfg.get("visual_observation_shape")
        if shape is not None:
            resolution = (int(shape[0]), int(shape[1]))

    # Phase 0 vs Phase 1 detection: was BodySLAMWrapper applied at training?
    bs = cfg.env.get("bodyslam") if "env" in cfg else None
    bs_mode = str(bs.get("mode", "off")) if bs is not None else "off"
    includes_human_pos = bs_mode in ("oracle", "noisy")

    # Synthesize the observation_space the agent was instantiated against at
    # training. The raw env (post-BodySLAMWrapper) has per-key 1-D obs and
    # rgb as (3, H, W); the agent expects ConcatDim+FrameStack output.
    synthesized_obs_space = _synthesize_snapshot_obs_space(
        env, cameras, resolution, includes_human_pos
    )

    method_cfg = cfg.method
    intrinsic_reward_module = None
    agent = hydra.utils.instantiate(
        method_cfg,
        device="cpu",
        observation_space=synthesized_obs_space,
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

    return _SnapshotPolicy(
        agent=agent,
        cameras=cameras,
        camera_resolution=resolution,
        includes_human_pos=includes_human_pos,
    )


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
        snapshot_cameras: Tuple[str, ...] = ()
        snapshot_resolution: Tuple[int, int] = (84, 84)
        bodyslam_mode = plan.bodyslam_mode
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
            # Peek at the snapshot's cfg so the env we build matches the
            # actor's *pixel* training format (same cameras + resolution).
            # The actor's low_dim_state schema (Phase 0 off vs Phase 1
            # oracle/noisy) is handled separately by _SnapshotPolicy.adapt_obs
            # and the synthesized observation_space inside load_snapshot_policy.
            snapshot_cameras, snapshot_resolution = peek_snapshot_cameras(snapshot_path)
            snap_bs = peek_snapshot_bodyslam_mode(snapshot_path)
            # Env wrapping is governed by plan.bodyslam_mode regardless of
            # what the snapshot was trained with — the SVF dataset must
            # always carry human_pos_estimate so the critic can learn the
            # SSM signal. Phase 0 snapshots (bodyslam=off) still work as
            # action samplers because adapt_obs strips the human channel
            # before feeding the actor.
            logger.info(
                f"Snapshot at {snapshot_path}: trained bodyslam={snap_bs}, "
                f"cameras={list(snapshot_cameras) or 'none'} "
                f"@ {snapshot_resolution[0]}x{snapshot_resolution[1]}; "
                f"env wrapping uses plan.bodyslam_mode={bodyslam_mode!r}."
            )

        for disruption in plan.disruptions:
            logger.info(
                f"Collecting source={source} task={task_key} disruption={disruption} "
                f"({plan.episodes_per_cell} episodes, max {plan.max_steps} steps)"
            )
            env = _build_live_env(
                task_key, disruption, bodyslam_mode, plan.motion_clips,
                cameras=snapshot_cameras,
                camera_resolution=snapshot_resolution,
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
                    proximity_threshold=plan.proximity_threshold,
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
                    min_separation=payload["min_separation"],
                    pfl_force_ratio=payload["pfl_force_ratio"],
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
            plan.disruptions[0] if plan.disruptions else "coworker_train",
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
                min_separation=payload["min_separation"],
                pfl_force_ratio=payload["pfl_force_ratio"],
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
    p.add_argument(
        "--disruption-space",
        choices=DISRUPTION_SPACE_CHOICES,
        default="coworker_train",
        help=(
            "coworker_train (default): single cell using make_coworker_train_space() "
            "— 5 continuous parameter axes in moderate bands. "
            "coworker_eval: single cell using make_coworker_eval_space() — wider "
            "bands that strictly contain the train ranges. "
            "legacy_multi: iterate over --disruptions string list (pre-Phase-0.5 "
            "5-disruption mixture). Retained as an escape hatch."
        ),
    )
    p.add_argument(
        "--disruptions",
        nargs="+",
        default=list(DEFAULT_DISRUPTIONS),
        help="Only used when --disruption-space=legacy_multi.",
    )
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
    p.add_argument(
        "--proximity-threshold",
        type=float,
        default=0.10,
        help=(
            "Geometric near-contact bar (metres) used by label_transition. "
            "Any human-joint / robot-link pair closer than this counts as a "
            "safety violation. Default 0.10 m. Surfaced for sweep; see "
            "safety_bigym/filters/labeling.py for the rationale."
        ),
    )
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
        if args.disruption_space == "legacy_multi":
            disruptions = tuple(args.disruptions)
        else:
            # coworker_train / coworker_eval: single cell labelled by the space
            # name; _build_live_env dispatches to the factory on this label.
            disruptions = (args.disruption_space,)
        plan = CollectionPlan(
            sources=tuple(args.source or ("random",)),
            tasks=tuple(args.tasks),
            disruptions=disruptions,
            episodes_per_cell=args.episodes_per_cell,
            max_steps=args.max_steps,
            bodyslam_mode=args.bodyslam_mode,
            output_dir=args.output_dir,
            snapshot_overrides=snapshot_overrides,
            seed=args.seed,
            demos_per_task=args.demos_per_task,
            proximity_threshold=args.proximity_threshold,
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

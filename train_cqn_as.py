"""CQN-AS training entrypoint for SafetyBiGym.

Modelled on CQN-AS/train_cqn_as_bigym.py but routes env construction
through safety_bigym.agents.cqn_as.env_adapter so all safety wrappers
compose. Handles num_demos=0 cleanly (skips the demo replay buffer
entirely) so the A6 smoke gate can exercise composition without
the demo pipeline (deferred to a follow-up).

Usage:
    export AMASS_DATA_DIR=/path/to/CMU/CMU
    export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0  # headless GPU box
    python train_cqn_as.py \\
        env=safety_bigym/dishwasher_close \\
        disruption=coworker_train \\
        bodyslam=oracle \\
        num_train_frames=20000 \\
        wandb.use=true \\
        wandb.name=cqn_as_dishwasher_close_smoke
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Default to EGL on headless boxes; user can override.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

import hydra
import numpy as np
import torch
from dm_env import specs
from omegaconf import DictConfig, OmegaConf

from safety_bigym.agents.cqn_as import env_adapter, utils
from safety_bigym.agents.cqn_as.eval_video import render_frame, write_eval_video
from safety_bigym.agents.cqn_as.replay_buffer import (
    ReplayBufferStorage,
    make_replay_loader,
)

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


def make_agent(rgb_obs_spec, low_dim_obs_spec, action_spec, action_sequence, cfg):
    cfg.rgb_obs_shape = list(rgb_obs_spec.shape)
    cfg.low_dim_obs_shape = list(low_dim_obs_spec.shape)
    cfg.action_shape = [action_sequence, *action_spec.shape]
    return hydra.utils.instantiate(cfg)


def step_marks_task_success(time_step) -> bool:
    """Did this time_step's env-info dict signal a task-success?

    Reads ``info["task_success"]`` (populated by BiGym's base env at
    ``bigym_env.py:329`` as ``float(self.success)`` — 1.0 on the success
    step, 0.0 otherwise; the env terminates immediately after a success
    fires, so any True observation during an episode means the episode
    completed the task). Used by both train() and eval() to log a clean,
    shaping-independent task-quality metric (workspace shaping makes
    episode_reward strongly negative even for successful eps, so reward is
    no longer a reliable success indicator).
    """
    info = getattr(time_step, "info", None)
    if not isinstance(info, dict):
        return False
    try:
        return float(info.get("task_success", 0.0)) > 0
    except (TypeError, ValueError):
        return False


class Workspace:
    def __init__(self, cfg: DictConfig):
        self.work_dir = Path.cwd()
        logger.info(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self._setup_env()
        self._setup_replay()
        self._setup_wandb()

        self.agent = make_agent(
            self.train_env.rgb_observation_spec(),
            self.train_env.low_dim_observation_spec(),
            self.train_env.action_spec(),
            self.cfg.action_sequence,
            self.cfg.agent,
        )

        self.timer = utils.Timer()
        self._update_step = 0
        self._global_step = 0
        self._global_episode = 0

        # Local JSON dumps (resilient to W&B downtime), per docs/safety_metrics.md.
        # `metrics.jsonl` streams one row per _log() call; `final_metrics.json`
        # captures the headline numbers at end of train().
        self._metrics_jsonl = self.work_dir / "metrics.jsonl"
        self._final_metrics_path = self.work_dir / "final_metrics.json"
        # Best-eval tracker — max-prefer for reward/success, min-prefer for safety.
        self._best_eval: dict = {
            "success_rate": -math.inf,
            "episode_reward": -math.inf,
            "ep_proximity_violation_rate": math.inf,
            "ep_ssm_violation_actual_rate": math.inf,
            "ep_min_separation_lowest": math.inf,
        }
        # Best eval-aligned snapshot for curriculum resume (success_rate max,
        # episode_reward tie-break). Written to snapshot_best.pt at train end.
        self._best_success_rate: float = -math.inf
        self._best_success_reward: float = -math.inf
        self._best_success_step: int | None = None
        self._best_snapshot_path: Path | None = None
        # Track last train/episode/eval rows for the final summary.
        self._last_train_episode_row: dict = {}
        self._last_episode_safety_row: dict = {}
        self._last_eval_row: dict = {}
        # Per-episode cost integral (Σ c_t). Read by _lagrangian_payload at
        # episode end and reset on env reset.
        self._episode_cost_integral = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_env(self) -> None:
        self.train_env = env_adapter.make(
            self.cfg,
            frame_stack=self.cfg.frame_stack,
            normalize_low_dim_obs=False,
        )
        if self.cfg.temporal_ensemble:
            self.train_temporal_ensemble = utils.TemporalEnsembleControl(
                self.cfg.env.episode_length,
                self.train_env.action_spec(),
                self.cfg.action_sequence,
            )
            self.eval_temporal_ensemble = utils.TemporalEnsembleControl(
                self.cfg.env.episode_length,
                self.train_env.action_spec(),
                self.cfg.action_sequence,
            )

    def _setup_replay(self) -> None:
        data_specs = (
            self.train_env.rgb_raw_observation_spec(),
            self.train_env.low_dim_raw_observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
            specs.Array((1,), np.float32, "demo"),
            # Phase 3 P3.0c: per-step cost c_t carried into the episode shards.
            # ReplayBufferStorage.add() reads `time_step["cost"]` via the
            # NamedTuple __getitem__ shim; env_adapter populates it from
            # info["safety"] each env-step.
            specs.Array((1,), np.float32, "cost"),
        )
        self.replay_storage = ReplayBufferStorage(
            data_specs, self.work_dir / "buffer", self.cfg.use_relabeling
        )
        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.action_sequence,
            self.cfg.frame_stack,
            fill_action="last_action",
        )
        # Demo plumbing only when num_demos>0; smoke gate runs with 0.
        self._demos_enabled = int(self.cfg.num_demos) > 0
        if self._demos_enabled:
            self.demo_replay_storage = ReplayBufferStorage(
                data_specs,
                self.work_dir / "demo_buffer",
                self.cfg.use_relabeling,
                is_demo_buffer=True,
            )
            self.demo_replay_loader = make_replay_loader(
                self.work_dir / "demo_buffer",
                self.cfg.replay_buffer_size,
                self.cfg.demo_batch_size,
                self.cfg.replay_buffer_num_workers,
                self.cfg.save_snapshot,
                self.cfg.nstep,
                self.cfg.discount,
                self.cfg.action_sequence,
                self.cfg.frame_stack,
                fill_action="last_action",
            )
        else:
            self.demo_replay_storage = None
            self.demo_replay_loader = None
        self._replay_iter = None

    def _setup_wandb(self) -> None:
        self._wandb_run = None
        wb_cfg = self.cfg.get("wandb", None)
        if wb_cfg is None or not bool(wb_cfg.get("use", False)):
            return
        try:
            import wandb
        except ImportError:
            logger.warning("wandb requested but not installed; skipping.")
            return
        # tags pass-through (docs/safety_metrics.md run-tagging scheme):
        # `+wandb.tags=[stage0,method=unconstrained,task=saucepan_to_hob]`.
        raw_tags = wb_cfg.get("tags") if hasattr(wb_cfg, "get") else None
        tags = [str(t) for t in raw_tags] if raw_tags else None
        self._wandb_run = wandb.init(
            project=str(wb_cfg.get("project", "safety-critic")),
            entity=wb_cfg.get("entity"),
            name=str(wb_cfg.get("name", "cqn_as_run")),
            tags=tags,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            dir=str(self.work_dir),
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, metrics, step: int, ty: str = "train") -> None:
        # `metrics` may be a TensorDict (returned by agent.update()) or a
        # plain dict. Three things need handling:
        #  1. TensorDict refuses bool conversion — can't use ``if not metrics``.
        #  2. ``len(TensorDict)`` returns the first batch-size dim, not the
        #     number of keys. ``agent.update()`` returns a TD built from 0-d
        #     ``.detach()`` tensors, so its ``batch_size == torch.Size([])``
        #     and ``len(td) == 0`` even when 4+ keys are present. Using
        #     ``len(metrics) == 0`` as an emptiness check silently swallows
        #     every per-update train log — that's the regression flagged in
        #     ``docs/IMPLEMENTATION_STATUS.md`` 2026-05-23 notes
        #     (``grep q_critic_loss`` returning nothing). Check ``items()``.
        #  3. ``TensorDict.items()`` can be a single-use generator (the dict
        #     comprehension below would exhaust it, leaving the format-
        #     string join silent). Materialise items into a list once.
        if metrics is None:
            return
        try:
            items = list(metrics.items())
        except Exception:
            return
        if not items:
            return
        materialised = []
        for k, v in items:
            if hasattr(v, "item"):  # 0-d tensor → python scalar
                try:
                    v = v.item()
                except (ValueError, RuntimeError):
                    pass  # non-scalar tensor; format() will str() it
            materialised.append((k, v))
        prefixed = {f"{ty}/{k}": v for k, v in materialised}
        if self._wandb_run is not None:
            self._wandb_run.log(prefixed, step=step)

        # Streaming local mirror — load with pandas.read_json("metrics.jsonl",
        # lines=True). Filter on `ty` to isolate train / episode / safety / eval.
        self._append_jsonl(step, ty, prefixed)
        # Snapshot the last row of each stream for final_metrics.json.
        if ty == "train" and "train/episode_reward" in prefixed:
            self._last_train_episode_row = dict(prefixed)
        elif ty == "episode":
            self._last_episode_safety_row = dict(prefixed)
        elif ty == "eval":
            self._last_eval_row = dict(prefixed)

        logger.info(
            f"[{ty}] step={step} "
            + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in items)
        )

    def _append_jsonl(self, step: int, ty: str, prefixed: dict) -> None:
        """One JSON object per _log() call. Coerces values to JSON-safe types."""
        row: dict = {"step": int(step), "ty": ty}
        for k, v in prefixed.items():
            if isinstance(v, bool):
                row[k] = bool(v)
            elif isinstance(v, (int, float)):
                # JSON refuses NaN/Inf; coerce to null.
                fv = float(v)
                row[k] = fv if math.isfinite(fv) else None
            else:
                # Fall back to str — keeps the line writable even if a stray
                # tensor / object slips through.
                row[k] = str(v)
        try:
            with self._metrics_jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            # Don't take down a training run because the disk is full.
            logger.warning(f"metrics.jsonl append failed: {e}")

    def _safety_payload(self, info: dict) -> dict:
        """Extract per-step + episode-end safety metrics from env info.

        Per-step keys forwarded (docs/safety_metrics.md): the three
        violation flavours + their margins + observed velocities + PFL
        ratio. Episode-end ``info["episode_safety"]`` is forwarded
        wholesale so any new ``ep_*`` field added to
        :class:`EpisodeSafetyMetrics` lands in W&B without a payload
        change here.
        """
        out: dict = {}
        step_safety = info.get("safety") if info else None
        if step_safety is not None:
            for key in (
                "ssm_violation",
                "ssm_violation_actual",
                "proximity_violation",
                "pfl_violation",
                "ssm_margin",
                "ssm_margin_actual",
                "min_separation",
                "pfl_force_ratio",
                "robot_vel",
                "human_vel",
            ):
                if key in step_safety:
                    val = step_safety[key]
                    if isinstance(val, bool):
                        val = float(val)
                    out[f"safety/{key}"] = val
        ep_safety = info.get("episode_safety") if info else None
        if ep_safety is not None:
            for key, val in ep_safety.items():
                if isinstance(val, (int, float, bool)):
                    out[f"episode_safety/{key}"] = float(val)
        return out

    def _lagrangian_payload(self) -> dict:
        """Two extra W&B keys for Lagrangian (P3.1) runs, no-op otherwise.

        Per docs/safety_metrics.md::Lagrangian-specific episode logging:
        - ``episode_lambda`` (only on agents exposing a ``_lambda`` field)
        - ``episode_cost_integral`` (emitted on the unconstrained baseline
          too — useful for "what would λ have been pushing on")
        """
        out: dict = {"episode_cost_integral": float(self._episode_cost_integral)}
        lam = getattr(self.agent, "_lambda", None)
        if lam is not None:
            try:
                out["episode_lambda"] = float(lam)
            except (TypeError, ValueError):
                pass
        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def global_episode(self) -> int:
        return self._global_episode

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            base = iter(self.replay_loader)
            if self._demos_enabled:
                demo = iter(self.demo_replay_loader)
                self._replay_iter = utils.DemoMergedIterator(base, demo)
            else:
                self._replay_iter = base
        return self._replay_iter

    # ------------------------------------------------------------------
    # Demos
    # ------------------------------------------------------------------

    def load_demos(self) -> None:
        if not self._demos_enabled:
            logger.warning(
                "num_demos=0 — running without demonstrations. CQN-AS is "
                "designed for demo-driven RL; this mode is intended for the "
                "A6 smoke gate only."
            )
            return
        demos = self.train_env.get_demos(self.cfg.num_demos)
        for demo in demos:
            for time_step in demo:
                self.replay_storage.add(time_step)
                self.demo_replay_storage.add(time_step)
        logger.info(
            f"Loaded {len(demos)} demos; replay size now {len(self.replay_storage)}"
        )

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(
            self.cfg.num_seed_frames, self.cfg.action_repeat
        )
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )
        do_eval = False

        time_step = self.train_env.reset()
        if self.cfg.temporal_ensemble:
            self.train_temporal_ensemble.reset()
        self.replay_storage.add(time_step)
        if self._demos_enabled:
            self.demo_replay_storage.add(time_step)

        episode_step = 0
        episode_reward = 0.0
        # Cumulative per-step cost c_t over the episode. Computed by the env
        # adapter (filters/cost_signal.compute_cost), attached to TimeStep,
        # and surfaced as `episode/episode_cost_integral` for the Pareto plot.
        episode_cost_integral = 0.0
        # Per-episode task-success tracker. info["task_success"] = 1.0 only on
        # the success step; the env terminates immediately after, so any True
        # observation during the episode means it completed the task. Logged
        # at episode-end as train/episode_success — read this rather than
        # train/episode_reward when judging task progress under workspace
        # shaping (which makes episode_reward negative even for successful eps).
        episode_success = False
        action = None
        metrics: dict = {}

        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                ep_safety_metrics = self._safety_payload(time_step.info or {})
                # Lagrangian payload (and unconstrained cost integral) ride
                # in the same `episode/*` namespace. See docs/safety_metrics.md.
                ep_safety_metrics.update(self._lagrangian_payload())
                if ep_safety_metrics:
                    self._log(ep_safety_metrics, self.global_step, ty="episode")
                # Lagrangian-specific episode-end logging (no-op when the
                # active agent isn't the Lagrangian subclass).
                lagrangian_metrics = self._lagrangian_payload(
                    episode_cost_integral
                )
                if lagrangian_metrics:
                    self._log(
                        lagrangian_metrics, self.global_step, ty="episode"
                    )
                self._log(
                    {
                        "episode_reward": episode_reward,
                        "episode_length": episode_step,
                        "episode_success": float(episode_success),
                        "episode": self._global_episode,
                        "buffer_size": len(self.replay_storage),
                    },
                    self.global_step,
                    ty="train",
                )
                # Per-episode cost integral resets at episode boundary.
                self._episode_cost_integral = 0.0
                if do_eval:
                    self.eval()
                    do_eval = False

                time_step = self.train_env.reset()
                if self.cfg.temporal_ensemble:
                    self.train_temporal_ensemble.reset()
                self.replay_storage.add(time_step)
                if self._demos_enabled:
                    self.demo_replay_storage.add(time_step)
                episode_step = 0
                episode_reward = 0.0
                episode_cost_integral = 0.0
                episode_success = False

            if (
                self.global_step >= self.cfg.eval_every_frames
                and eval_every_step(self.global_step)
            ):
                do_eval = True

            if (
                self.cfg.temporal_ensemble
                or episode_step % self.cfg.action_sequence == 0
            ):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    raw_action = self.agent.act(
                        time_step.rgb_obs,
                        time_step.low_dim_obs,
                        self.global_step,
                        eval_mode=True,
                    )
                action = raw_action.reshape([self.cfg.action_sequence, -1])
                if self.cfg.temporal_ensemble:
                    self.train_temporal_ensemble.register_action_sequence(action)

            # Worker-aware update gate. The CQN-AS replay loader stripes
            # episode files by ``eps_idx % num_workers`` (replay_buffer.py
            # _try_fetch). The DataLoader rotates across workers, so a
            # worker that hasn't seen its first eligible episode yet raises
            # IndexError on sample. Upstream CQN-AS never hits this because
            # demos pre-fill every worker; in the A6 smoke gate we run with
            # num_demos=0 and have to wait until each worker is guaranteed
            # at least one episode (i.e. global_episode >= num_workers).
            num_replay_workers = max(1, int(self.cfg.replay_buffer_num_workers))
            if (
                not seed_until_step(self.global_step)
                and self.global_step % self.cfg.agent.update_every_steps == 0
                and self._global_episode >= num_replay_workers
            ):
                for _ in range(self.cfg.num_update_steps):
                    batch = next(self.replay_iter)
                    batch = utils.to_torch_pixel_tensor_dict(batch, self.device)
                    # Guard: a non-finite tensor anywhere in the batch propagates
                    # NaN into the network -> NaN action / projection target ->
                    # floor(NaN) garbage index -> index_add_/gather dies with an
                    # opaque async CUDA device-side assert (dstIndex < dstAddDimSize)
                    # that mis-points at a later op (e.g. cumsum). Fail here instead,
                    # naming the offending field. Covers observations (low_dim_obs
                    # carries the demo-injected human_pos_estimate) + reward/action.
                    for _k in (
                        "reward", "discount", "action", "cost",
                        "low_dim_obs", "next_low_dim_obs",
                    ):
                        if _k in batch and not torch.isfinite(batch[_k]).all():
                            bad = batch[_k]
                            n_bad = int((~torch.isfinite(bad)).sum().item())
                            raise ValueError(
                                f"non-finite values in batch['{_k}'] before "
                                f"agent.update ({n_bad} of {bad.numel()} bad; "
                                f"finite range [{bad[torch.isfinite(bad)].min().item() if torch.isfinite(bad).any() else float('nan')}, "
                                f"{bad[torch.isfinite(bad)].max().item() if torch.isfinite(bad).any() else float('nan')}]); "
                                f"this would crash the C51 critic. Likely a "
                                f"malformed demo (e.g. human_pos_estimate) or env "
                                f"signal — check the demo set / obs pipeline."
                            )
                    metrics = self.agent.update(batch)
                    self._update_step += 1
                    self.agent.update_target_critic(self._update_step)
                # `metrics` is a TensorDict; truthiness raises. Defer the
                # filled-vs-empty check to _log (which handles both shapes).
                self._log(metrics, self.global_step, ty="train")

            if self.cfg.temporal_ensemble:
                sub_action = self.train_temporal_ensemble.get_action()
            else:
                sub_action = action[episode_step % self.cfg.action_sequence]
            sub_action = self.agent.add_noise_to_action(sub_action, self.global_step)
            time_step = self.train_env.step(sub_action)
            episode_reward += time_step.reward
            # Σ c_t for episode_cost_integral. env_adapter populates
            # time_step.cost from info["safety"] every env-step.
            self._episode_cost_integral += float(getattr(time_step, "cost", 0.0) or 0.0)
            self.replay_storage.add(time_step)
            if self._demos_enabled:
                self.demo_replay_storage.add(time_step)

            # Per-step safety payload (gate A6.3 — confirms info["safety"]
            # arrives per env step, not per K-step chunk).
            step_safety = self._safety_payload(time_step.info or {})
            if step_safety and self.global_step % 50 == 0:
                self._log(step_safety, self.global_step, ty="safety")

            episode_step += 1
            self._global_step += 1

        # If no eval completed (e.g. SMOKE=1 with num_train_frames <
        # eval_every_frames), fall back to a final-state snapshot so the
        # stage still has a resume checkpoint.
        if self.cfg.save_snapshot and self._best_snapshot_path is None:
            self.save_snapshot()
            self._best_snapshot_path = (
                self.work_dir / f"snapshot_{self._global_step}.pt"
            )

        if self.cfg.save_snapshot:
            self._finalize_best_snapshot()

        # Headline summary for the thesis writeup (docs/safety_metrics.md).
        self._write_final_metrics()

        # Final summary JSON — aggregates the streaming metrics.jsonl into
        # a single headline-numbers file for easy post-hoc comparison
        # across runs (best success rate, mean/last safety axes, etc.).
        self._write_final_summary()

    def _write_final_summary(self) -> None:
        """Aggregate metrics.jsonl into <work_dir>/final_metrics.json with
        the thesis headline numbers: last + best eval reward / success rate
        / safety axes, and last training-side state (lambda, cost integral)."""
        jsonl = self.work_dir / "metrics.jsonl"
        out_path = self.work_dir / "final_metrics.json"
        if not jsonl.exists():
            return
        try:
            rows: list = []
            with jsonl.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except (TypeError, ValueError):
                        continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"failed to read metrics.jsonl for summary: {exc}")
            return

        def _last_by_ty(ty: str) -> dict:
            for row in reversed(rows):
                if row.get("ty") == ty:
                    return row
            return {}

        def _best_eval(key: str, prefer: str = "max") -> float | None:
            vals: list = []
            for row in rows:
                if row.get("ty") != "eval":
                    continue
                v = row.get(f"eval/{key}")
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if not vals:
                return None
            return max(vals) if prefer == "max" else min(vals)

        summary: dict = {
            "global_step": self._global_step,
            "global_episode": self._global_episode,
            "config": {
                "task": str(self.cfg.env.get("task_name", "")),
                "disruption": str(self.cfg.get("disruption", "")),
                "num_train_frames": int(self.cfg.num_train_frames),
                "num_demos": int(self.cfg.get("num_demos", 0)),
                "agent_v_min": float(self.cfg.agent.get("v_min", 0.0)),
                "agent_v_max": float(self.cfg.agent.get("v_max", 0.0)),
                "wandb_name": str(self.cfg.wandb.get("name", ""))
                if "wandb" in self.cfg else "",
                "wandb_tags": list(self.cfg.wandb.get("tags", []) or [])
                if "wandb" in self.cfg else [],
            },
            "last_train_episode": {
                k: v for k, v in _last_by_ty("train").items()
                if k not in ("step", "ty")
            },
            "last_episode_safety": {
                k: v for k, v in _last_by_ty("episode").items()
                if k not in ("step", "ty")
            },
            "last_eval": {
                k: v for k, v in _last_by_ty("eval").items()
                if k not in ("step", "ty")
            },
            "best_eval": {
                "success_rate":  _best_eval("success_rate", prefer="max"),
                "episode_reward": _best_eval("episode_reward", prefer="max"),
                # Lower is better for the safety axes.
                "ep_proximity_violation_rate":
                    _best_eval("ep_proximity_violation_rate", prefer="min"),
                "ep_ssm_violation_actual_rate":
                    _best_eval("ep_ssm_violation_actual_rate", prefer="min"),
                "ep_min_separation_lowest":
                    _best_eval("ep_min_separation", prefer="min"),
            },
        }
        try:
            with out_path.open("w") as f:
                json.dump(summary, f, indent=2, default=repr)
            logger.info(f"wrote final summary: {out_path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"failed to write final_metrics.json: {exc}")

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def eval(self) -> None:
        step, episode, total_reward = 0, 0, 0.0
        # Task-success rate is the ground-truth quality metric — eval/episode_reward
        # is contaminated by the workspace-shaping dense penalty (a fully-successful
        # episode under shaping can still log a strongly negative episode_reward
        # because of the accumulated -beta·excess per-step term). BiGym already
        # emits ``info["task_success"] = float(self.success)`` at each step
        # (bigym_env.py:329) and the env_adapter forwards it onto time_step.info.
        # An episode counts as a success if any step had task_success > 0.
        total_success = 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        # Per-eval safety aggregation. EpisodeSafetyMetrics emits
        # info["episode_safety"] every step (running) and at episode-end
        # (final). We capture the terminal payload per eval episode and
        # mean across episodes so eval/* in W&B + metrics.jsonl carries
        # the safety axis alongside reward/success — the headline pair for
        # the thesis Pareto plot.
        ep_safety_sum: dict = {}
        ep_safety_count = 0

        # cfg.save_video gates per-eval-cycle video recording. Only the first
        # eval episode is captured per cycle (disk is cheap but not infinite,
        # and the second/third episode look very similar early in training).
        # Rendering goes through self.train_env.render() which delegates down
        # through the wrapper stack to SafetyBiGymEnv's MuJoCo renderer.
        record_video = bool(self.cfg.get("save_video", False))
        frames: list = []
        video_dir = self.work_dir / "eval_videos"

        # Per-episode aggregates (docs/safety_metrics.md): the eval() loop
        # collects each rollout's terminal info["episode_safety"] and rolls
        # them up so `eval/ep_*` lands in W&B paired with reward/success.
        ep_safety_sums: dict = defaultdict(float)
        ep_safety_mins: dict = {}
        ep_safety_maxes: dict = {}
        success_count = 0
        terminal_info_seen = 0

        while eval_until_episode(episode):
            episode_step = 0
            episode_success = False
            time_step = self.train_env.reset()
            if self.cfg.temporal_ensemble:
                self.eval_temporal_ensemble.reset()
            action = None
            if record_video and episode == 0:
                frame = render_frame(self.train_env, global_step=self.global_step)
                if frame is not None:
                    frames.append(frame)
            while not time_step.last():
                if (
                    self.cfg.temporal_ensemble
                    or episode_step % self.cfg.action_sequence == 0
                ):
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        raw_action = self.agent.act(
                            time_step.rgb_obs,
                            time_step.low_dim_obs,
                            self.global_step,
                            eval_mode=True,
                        )
                    action = raw_action.reshape([self.cfg.action_sequence, -1])
                    if self.cfg.temporal_ensemble:
                        self.eval_temporal_ensemble.register_action_sequence(action)
                if self.cfg.temporal_ensemble:
                    sub_action = self.eval_temporal_ensemble.get_action()
                else:
                    sub_action = action[episode_step % self.cfg.action_sequence]
                time_step = self.train_env.step(sub_action)
                total_reward += time_step.reward
                step += 1
                episode_step += 1
                if step_marks_task_success(time_step):
                    episode_success = True
                if record_video and episode == 0:
                    frame = render_frame(self.train_env, global_step=self.global_step)
                    if frame is not None:
                        frames.append(frame)
            # Terminal info captured here — EpisodeSafetyMetrics has filled
            # info["episode_safety"] with the full per-episode aggregate.
            term_info = getattr(time_step, "info", None) or {}
            ep_safety = term_info.get("episode_safety") if isinstance(
                term_info, dict
            ) else None
            if isinstance(ep_safety, dict):
                terminal_info_seen += 1
                for k, v in ep_safety.items():
                    if not isinstance(v, (int, float, bool)):
                        continue
                    fv = float(v)
                    if k.startswith("ep_min_"):
                        ep_safety_mins[k] = (
                            fv if k not in ep_safety_mins
                            else min(ep_safety_mins[k], fv)
                        )
                    elif k.startswith("ep_max_"):
                        ep_safety_maxes[k] = (
                            fv if k not in ep_safety_maxes
                            else max(ep_safety_maxes[k], fv)
                        )
                    else:
                        ep_safety_sums[k] += fv
            if isinstance(term_info, dict) and bool(term_info.get(
                "task_success", False
            )):
                success_count += 1
            episode += 1
            # Capture terminal-step episode_safety for this eval episode.
            ep_safety = (time_step.info or {}).get("episode_safety") or {}
            if ep_safety:
                ep_safety_count += 1
                for key, val in ep_safety.items():
                    if isinstance(val, (int, float, bool)):
                        ep_safety_sum[key] = (
                            ep_safety_sum.get(key, 0.0) + float(val)
                        )

        if record_video and frames:
            write_eval_video(
                video_dir,
                frames,
                global_step=self.global_step,
                wandb_run=self._wandb_run,
            )

        eval_row: dict = {
            "episode_reward": total_reward / max(episode, 1),
            "episode_length": step / max(episode, 1),
            "episode": self._global_episode,
        }
        # Average rates / dwell / mean fields across eval rollouts. min/max
        # fields use min/max instead of mean so the worst-case shows.
        if terminal_info_seen > 0:
            for k, v in ep_safety_sums.items():
                eval_row[k] = v / terminal_info_seen
            eval_row.update(ep_safety_mins)
            eval_row.update(ep_safety_maxes)
            eval_row["success_rate"] = success_count / terminal_info_seen
        self._log(eval_row, self.global_step, ty="eval")
        # Update best_eval (max-prefer reward/success, min-prefer safety).
        self._update_best_eval(eval_row)
        # Snapshots align with eval cycles so curriculum resume can pick the
        # peak-by-success checkpoint (see scripts/pick_best_snapshot.py).
        if self.cfg.save_snapshot:
            self.save_snapshot()
            self._mark_best_snapshot(eval_row)

    # ------------------------------------------------------------------
    # Best-eval tracking + final-metrics dump (docs/safety_metrics.md)
    # ------------------------------------------------------------------

    def _mark_best_snapshot(self, eval_row: dict) -> None:
        """Track the snapshot with the highest eval success_rate."""
        sr = eval_row.get("success_rate")
        if not isinstance(sr, (int, float)) or not math.isfinite(float(sr)):
            return
        er_raw = eval_row.get("episode_reward")
        er = (
            float(er_raw)
            if isinstance(er_raw, (int, float)) and math.isfinite(float(er_raw))
            else -math.inf
        )
        sr = float(sr)
        if sr > self._best_success_rate or (
            sr == self._best_success_rate and er > self._best_success_reward
        ):
            self._best_success_rate = sr
            self._best_success_reward = er
            self._best_success_step = int(self._global_step)
            self._best_snapshot_path = (
                self.work_dir / f"snapshot_{self._global_step}.pt"
            )

    def _finalize_best_snapshot(self) -> None:
        """Copy the best eval-aligned snapshot to snapshot_best.pt."""
        if self._best_snapshot_path is None or not self._best_snapshot_path.is_file():
            logger.warning("no best snapshot to finalize")
            return
        dest = self.work_dir / "snapshot_best.pt"
        shutil.copy2(self._best_snapshot_path, dest)
        logger.info(
            f"best snapshot: {dest} "
            f"(step={self._best_success_step}, "
            f"success_rate={self._best_success_rate:.4f}, "
            f"episode_reward={self._best_success_reward:.4f})"
        )

    def _update_best_eval(self, eval_row: dict) -> None:
        """Track best (max-prefer reward/success, min-prefer safety) across eval cycles."""
        max_prefer = ("success_rate", "episode_reward")
        for k in max_prefer:
            v = eval_row.get(k)
            if isinstance(v, (int, float)) and float(v) > self._best_eval[k]:
                self._best_eval[k] = float(v)
        # Safety: lowest violation rate is best; track ep_min_separation
        # under a distinct key so we don't conflict with the rate accessor.
        for k in ("ep_proximity_violation_rate", "ep_ssm_violation_actual_rate"):
            v = eval_row.get(k)
            if isinstance(v, (int, float)) and float(v) < self._best_eval[k]:
                self._best_eval[k] = float(v)
        # Lowest per-eval ep_min_separation is the dangerous-tail anchor.
        v = eval_row.get("ep_min_separation")
        if isinstance(v, (int, float)) and float(v) < self._best_eval[
            "ep_min_separation_lowest"
        ]:
            self._best_eval["ep_min_separation_lowest"] = float(v)

    def _write_final_metrics(self) -> None:
        """Emit final_metrics.json with headline numbers (docs/safety_metrics.md)."""
        wb_cfg = self.cfg.get("wandb", {}) or {}
        out: dict = {
            "config": {
                "task": str(self.cfg.env.get("env_name", "")),
                "disruption": str(self.cfg.get("disruption", "")),
                "num_train_frames": int(self.cfg.num_train_frames),
                "num_demos": int(self.cfg.num_demos),
                "agent_v_min": float(self.cfg.agent.get("v_min", float("nan"))),
                "agent_v_max": float(self.cfg.agent.get("v_max", float("nan"))),
                "wandb_name": str(wb_cfg.get("name", "")) if wb_cfg else "",
                "wandb_tags": list(wb_cfg.get("tags", []) or []) if wb_cfg else [],
            },
            "last_train_episode": self._last_train_episode_row,
            "last_episode_safety": self._last_episode_safety_row,
            "last_eval": self._last_eval_row,
            "best_eval": {
                k: (None if v in (math.inf, -math.inf) else v)
                for k, v in self._best_eval.items()
            },
            "best_snapshot": (
                None
                if self._best_snapshot_path is None
                else {
                    "path": "snapshot_best.pt",
                    "source": self._best_snapshot_path.name,
                    "step": self._best_success_step,
                    "success_rate": (
                        None
                        if self._best_success_rate == -math.inf
                        else self._best_success_rate
                    ),
                    "episode_reward": (
                        None
                        if self._best_success_reward == -math.inf
                        else self._best_success_reward
                    ),
                }
            ),
        }
        try:
            self._final_metrics_path.write_text(json.dumps(out, indent=2))
            logger.info(f"final metrics written: {self._final_metrics_path}")
        except OSError as e:
            logger.warning(f"final_metrics.json write failed: {e}")

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def save_snapshot(self) -> None:
        snapshot = self.work_dir / f"snapshot_{self.global_step}.pt"
        payload = {
            "agent_state": self.agent.state_dict()
            if hasattr(self.agent, "state_dict") else None,
            "step": self._global_step,
            "episode": self._global_episode,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        with snapshot.open("wb") as f:
            torch.save(payload, f)
        logger.info(f"saved snapshot: {snapshot}")

    def load_snapshot(self, path) -> None:
        """Eager-load a saved agent state into ``self.agent``.

        Called from ``main`` before ``train()`` when the user passes
        ``+snapshot_path=...`` on the Hydra CLI. The snapshot must have
        been produced by a run with a compatible model architecture
        (same bodyslam mode, same task, same agent config). Mismatches
        surface as a strict-load KeyError or shape RuntimeError, which
        is the right failure mode — we'd rather fail loudly than
        silently run with mostly-random weights.

        ``weights_only=False`` is required because snapshot payloads
        also carry the run config (an OmegaConf DictConfig); torch
        ≥2.6's default safe-globals list doesn't include it.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"snapshot_path {path} not found")
        logger.info(f"loading snapshot: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        agent_state = payload.get("agent_state")
        if agent_state is None:
            raise ValueError(
                f"snapshot {path} has no 'agent_state' field "
                f"(keys: {sorted(payload.keys())})"
            )
        # The vendored CQNASAgent isn't an nn.Module — it owns sub-modules
        # (encoder, critic, target_critic, optimizers). Its state_dict()
        # returns a dict-of-dicts; load_state_dict needs the same shape.
        self.agent.load_state_dict(agent_state)
        loaded_step = int(payload.get("step", 0))
        loaded_ep = int(payload.get("episode", 0))
        logger.info(
            f"snapshot loaded: trained for {loaded_step} steps / "
            f"{loaded_ep} episodes"
        )


@hydra.main(version_base=None, config_path="cfgs", config_name="cqn_as_config")
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    # Eval-only mode: load a saved policy, run eval() once, exit.
    # phase1_reward_pilot_cqn_as.py --eval uses this path with
    # num_train_frames=0 and +snapshot_path=<peak snapshot>.
    snapshot_path = cfg.get("snapshot_path") if hasattr(cfg, "get") else None
    if snapshot_path is not None:
        workspace.load_snapshot(snapshot_path)
    if int(cfg.num_train_frames) <= 0:
        if snapshot_path is None:
            logger.warning(
                "num_train_frames<=0 with no snapshot_path — agent is "
                "freshly initialised; eval results will be ~random."
            )
        workspace.eval()
        return
    workspace.load_demos()
    workspace.train()


if __name__ == "__main__":
    main()

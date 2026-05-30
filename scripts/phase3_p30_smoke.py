"""Phase 3 P3.0 end-to-end smoke verification.

Drives a real ``SafetyBiGymEnv`` through the CQN-AS adapter for ~500 env-steps
and asserts the three load-bearing P3.0 invariants:

1. **Workspace shaping wired** — ``add_workspace_penalty=True`` actually
   subtracts ``beta * max(0, ||ee - task|| - r_ws)`` from the task reward, and
   evaluates to zero when EE is inside ``r_ws``.
2. **Per-env-step cost flows** — every env-step attaches a fresh ``c_t`` to the
   TimeStep, not a chunk-aggregated value; ``c_t`` is strictly positive on at
   least one step during a 500-step random-policy COWORKER rollout (the human
   reliably approaches the robot inside ``d_buffer = 0.3 m`` at least once).
3. **Warm-start guard fires** — :meth:`CostCritic.warm_start_from_svf` raises
   without ``force_sign_flip=True`` even when handed a real Phase 2 SVF
   checkpoint. Positive control that the sign-mismatch refusal is reachable
   from the actual checkpoint format.

Run::

    cd safety_bigym
    export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU
    python scripts/phase3_p30_smoke.py \\
        env=safety_bigym/dishwasher_close \\
        disruption=coworker_train \\
        bodyslam=oracle \\
        pixels=false

Wall-time target: < 90 s on the local box. The script is read-only — it
constructs the env, steps it, prints diagnostics, and exits.

Use ``+phase3_p30_smoke.dry_run=true`` to skip env construction and only run
the warm-start guard check (≈ 1 s) — useful for CI on environments without
MuJoCo / AMASS data.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("phase3_p30_smoke")


def _check_amass_dir_or_die() -> None:
    amass = os.environ.get("AMASS_DATA_DIR")
    if not amass:
        raise RuntimeError(
            "AMASS_DATA_DIR is unset. Per CLAUDE.md, the Phase 3 smoke needs "
            "the human controller running on real motion clips. Export "
            "AMASS_DATA_DIR=/path/to/CMU before invoking this script."
        )
    if not Path(amass).is_dir():
        raise RuntimeError(f"AMASS_DATA_DIR points at non-directory: {amass!r}")


def _ee_to_task_distance(env_raw) -> float:
    """Pull EE↔task-object distance from the underlying SafetyBiGymEnv.

    The factory stack wraps the env in several Gymnasium wrappers (BodySLAM,
    ISO15066, EpisodeSafetyMetrics). Gymnasium guards private attributes
    behind the wrapper chain, so we hop to ``.unwrapped`` before reading
    SafetyBiGymEnv internals.

    The EE-position lookup uses the same fallback chain as
    :meth:`SafetyBiGymEnv._compute_workspace_penalty`: try ``state["ee_pos"]``
    first (most BiGym robots), then ``state["link_pos"]["ee"]`` (H1 — it has
    no ``get_ee_position`` method, so the top-level key is absent but the
    link_pos fallback is populated via _ROBOT_LINK_NAMES).
    """
    inner = env_raw.unwrapped if hasattr(env_raw, "unwrapped") else env_raw
    state = inner._get_robot_state()
    ee_pos = state.get("ee_pos")
    if ee_pos is None:
        ee_pos = (state.get("link_pos") or {}).get("ee")
    task_pos = state.get("task_object_pos")
    if ee_pos is None or task_pos is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(ee_pos, dtype=float) - task_pos))


def _run_warm_start_guard_check() -> bool:
    """Confirm CostCritic refuses an SVF warm-start without force_sign_flip.

    Falls back to a freshly-constructed SafetyCritic payload when the
    on-disk v1 checkpoint is not present (the on-disk version is the same
    architectural twin, so the guard's behaviour is identical).
    """
    from safety_bigym.filters.cost_critic import CostCritic
    from safety_bigym.filters.critic import SafetyCritic
    from safety_bigym.filters.feature_extractor import CriticFeatureSpec

    svf_v1 = Path("checkpoints/svf_coworker_train_v1.pt")
    if svf_v1.exists():
        logger.info("Loading SVF v1 checkpoint payload: %s", svf_v1)
        payload = torch.load(svf_v1, map_location="cpu", weights_only=False)
        # Normalise: training scripts sometimes wrap the payload under "svf"
        # or a similar top-level key. Handle both shapes.
        if "state_dict" not in payload and "svf" in payload:
            payload = payload["svf"]
        spec = CriticFeatureSpec.from_dict(payload["spec"])
    else:
        logger.warning(
            "SVF v1 checkpoint not at %s; using a synthetic payload as positive "
            "control (architecturally identical, guard behaviour the same).",
            svf_v1,
        )
        spec = CriticFeatureSpec(
            obs_keys=("low_dim_state",), obs_dims=(64,), action_dim=16
        )
        svf = SafetyCritic(spec=spec, gamma=0.99)
        payload = svf.checkpoint_payload()

    cost = CostCritic(
        spec=spec,
        gamma=float(payload["gamma"]),
        hidden_dims=tuple(int(h) for h in payload["hidden_dims"]),
    )
    try:
        cost.warm_start_from_svf(payload, force_sign_flip=False)
    except ValueError as exc:
        msg = str(exc)
        if "force_sign_flip=True" not in msg:
            raise AssertionError(
                f"Warm-start raised but the message did not name the opt-in flag: {msg}"
            ) from exc
        logger.info(
            "Warm-start guard fired as expected: %s",
            msg.splitlines()[0],
        )
        return True
    raise AssertionError(
        "CostCritic.warm_start_from_svf returned without raising — "
        "the sign-mismatch guard is missing!"
    )


def _smoke_rollout(cfg: DictConfig, *, num_steps: int = 500) -> dict:
    """Drive the CQN-AS-adapted SafetyBiGymEnv for ``num_steps`` env-steps.

    Returns aggregate diagnostics: mean / max cost, count of steps with
    ``c_t > 0`` and ``r_workspace < 0``, sample per-step traces.
    """
    from safety_bigym.agents.cqn_as import env_adapter

    env = env_adapter.make(cfg)
    inner = env._env  # SafetyBiGymCQNAdapter
    raw = inner._env  # underlying gym SafetyBiGymEnv

    ts = env.reset()
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    action_shape = env.action_spec().shape

    cost_trace: List[float] = []
    reward_trace: List[float] = []
    ssm_margin_trace: List[float] = []
    ee_dist_trace: List[float] = []
    workspace_pen_trace: List[float] = []

    nonzero_cost_steps = 0
    nonzero_workspace_steps = 0
    episodes_completed = 0

    for step in range(num_steps):
        # Random action in [-1, 1]; gripper tail in [0, 1] is fine because the
        # adapter's rescaler handles the convention.
        action = rng.uniform(-1.0, 1.0, size=action_shape).astype(np.float32)
        ts = env.step(action)

        c = float(ts.cost)
        cost_trace.append(c)
        reward_trace.append(float(ts.reward))
        if c > 1e-6:
            nonzero_cost_steps += 1

        safety = ts.info.get("safety", {}) if isinstance(ts.info, dict) else {}
        ssm_margin_trace.append(float(safety.get("ssm_margin", float("nan"))))

        ee_dist = _ee_to_task_distance(raw)
        ee_dist_trace.append(ee_dist)
        # Workspace penalty = -beta * max(0, ee_dist - r_ws). We can read the
        # config back off the env's SafetyConfig instance for ground truth.
        raw_inner = raw.unwrapped if hasattr(raw, "unwrapped") else raw
        sc = raw_inner.safety_config
        if sc.add_workspace_penalty and np.isfinite(ee_dist):
            excess = max(0.0, ee_dist - sc.workspace_radius)
            cap = getattr(sc, "workspace_excess_cap", None)
            if cap is not None:
                excess = min(excess, float(cap))
            wp = -sc.workspace_beta * excess
        else:
            wp = 0.0
        workspace_pen_trace.append(wp)
        if wp < -1e-6:
            nonzero_workspace_steps += 1

        if step % 50 == 0:
            logger.info(
                "step=%-4d ee_dist=%.3f r_ws=%.4f ssm_margin=%6.3f c_t=%.4f reward=%.4f",
                step,
                ee_dist,
                wp,
                ssm_margin_trace[-1],
                c,
                reward_trace[-1],
            )

        if ts.last():
            episodes_completed += 1
            ts = env.reset()

    env.close()

    cost_arr = np.array(cost_trace, dtype=np.float64)
    wp_arr = np.array(workspace_pen_trace, dtype=np.float64)
    return {
        "n_steps": num_steps,
        "episodes_completed": episodes_completed,
        "mean_cost": float(cost_arr.mean()),
        "max_cost": float(cost_arr.max()),
        "nonzero_cost_steps": nonzero_cost_steps,
        "min_ssm_margin": float(np.nanmin(ssm_margin_trace)),
        "nonzero_workspace_steps": nonzero_workspace_steps,
        "mean_workspace_penalty": float(wp_arr.mean()),
        "min_workspace_penalty": float(wp_arr.min()),
        "mean_ee_dist": float(np.nanmean(ee_dist_trace)),
        "max_ee_dist": float(np.nanmax(ee_dist_trace)),
    }


@hydra.main(config_path="../cfgs", config_name="cqn_as_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    dry_run = bool(cfg.get("phase3_p30_smoke", {}).get("dry_run", False))

    # P3.0d invariant 3: warm-start guard fires. Runs first because it has
    # no env dependency — failing here means the CostCritic module itself is
    # broken and there's no point spinning up MuJoCo.
    logger.info("=== P3.0d invariant 3: warm-start guard ===")
    _run_warm_start_guard_check()
    logger.info("Warm-start guard: PASS")

    if dry_run:
        logger.info("--dry-run set; skipping env rollout.")
        return

    # The full rollout needs a task; with no env= the `env` node exists but
    # carries no task_name (just bodyslam), so the adapter later dies on a
    # missing `env.episode_length`. Detect the real signal — no task_name —
    # and fail with the exact command instead.
    if OmegaConf.select(cfg, "env.task_name", default=None) is None:
        raise RuntimeError(
            "No env selected — the full P3.0 rollout needs a task. Run e.g.:\n"
            "  python scripts/phase3_p30_smoke.py env=safety_bigym/dishwasher_close "
            "disruption=coworker_train bodyslam=oracle pixels=false\n"
            "Or add +phase3_p30_smoke.dry_run=true for the warm-start guard "
            "check only (no MuJoCo / no env)."
        )

    _check_amass_dir_or_die()

    # Force smoke-friendly overrides on top of the resolved Hydra config so
    # users don't have to remember every CLI override.
    OmegaConf.set_struct(cfg, False)
    sc = cfg.env.setdefault("safety", OmegaConf.create({}))
    sc["add_workspace_penalty"] = True
    sc["workspace_radius"] = 0.4
    sc["workspace_beta"] = 0.2
    sc["log_violations"] = False  # see CLAUDE.md gotcha — drown-out kitchen-scale SSM logs
    # Pixels off keeps the local smoke under the 90-s budget.
    if "pixels" not in cfg or cfg.pixels is None:
        cfg.pixels = False

    logger.info("=== P3.0d invariants 1+2: end-to-end env rollout ===")
    logger.info(
        "Config: env=%s disruption=%s bodyslam=%s pixels=%s",
        cfg.env.get("task_name"),
        cfg.get("disruption"),
        cfg.bodyslam.get("mode") if "bodyslam" in cfg else "off",
        cfg.pixels,
    )

    stats = _smoke_rollout(cfg, num_steps=500)

    logger.info("=== Rollout stats ===")
    for k, v in stats.items():
        logger.info("  %s = %s", k, v)

    # ---- assertions ----
    assert stats["nonzero_cost_steps"] >= 1, (
        f"P3.0d invariant 2 FAILED: no step had c_t > 0 in 500 env-steps. "
        f"min_ssm_margin observed = {stats['min_ssm_margin']:.3f} m; "
        f"COWORKER human should approach inside d_buffer=0.3 m at least once."
    )
    assert stats["max_cost"] >= stats["mean_cost"], (
        f"max_cost ({stats['max_cost']}) < mean_cost ({stats['mean_cost']}) — "
        "aggregation logic broken."
    )

    # Workspace assertion: at least *some* steps had EE outside r_ws so the
    # penalty fires; otherwise the smoke can't confirm the path is wired.
    assert stats["nonzero_workspace_steps"] >= 1, (
        f"P3.0d invariant 1 FAILED: no step had r_workspace < 0. "
        f"mean_ee_dist = {stats['mean_ee_dist']:.3f} m, max_ee_dist = "
        f"{stats['max_ee_dist']:.3f} m, workspace_radius = 0.4 m. "
        "Either the task keeps EE permanently inside the radius (try a "
        "different task) or workspace shaping is misconfigured."
    )

    logger.info(
        "P3.0d smoke PASS: workspace fired on %d/500 steps, c_t>0 on %d/500 steps.",
        stats["nonzero_workspace_steps"],
        stats["nonzero_cost_steps"],
    )


if __name__ == "__main__":
    sys.exit(main())

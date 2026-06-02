"""Empirical probe: does the COWORKER disruption get close enough to violate?

Drives a real ``SafetyBiGymEnv`` (CQN-AS adapter path, same as training) for a
fixed number of env-steps under a chosen disruption and logs the per-step
geometric safety signals:

    - min_separation        (closest human-body <-> robot-geom surface distance)
    - closest_human_joint   (which of the 18 SMPL/G1 bodies is nearest)
    - proximity_violation   (min_separation < proximity_threshold)
    - ssm_violation         (worst-case ISO, v_h = v_h_max)
    - ssm_violation_actual  (observed-velocity ISO)

Aggregates violation *rates* and the separation distribution so you can see
whether the body-center ``closest_approach`` range actually produces violations
once the arm reaches in.

Run::

    cd safety_bigym
    export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU
    python scripts/probe_coworker_violations.py \\
        env=safety_bigym/dishwasher_close \\
        disruption=coworker_train \\
        bodyslam=oracle pixels=false \\
        +probe.num_steps=500

Read-only: constructs the env, steps it with a random policy, prints, exits.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("probe_coworker_violations")


def _check_amass_dir_or_die() -> None:
    amass = os.environ.get("AMASS_DATA_DIR")
    if not amass:
        raise RuntimeError(
            "AMASS_DATA_DIR is unset. Export "
            "AMASS_DATA_DIR=/path/to/CMU before invoking this script."
        )
    if not Path(amass).is_dir():
        raise RuntimeError(f"AMASS_DATA_DIR points at non-directory: {amass!r}")


def _probe(cfg: DictConfig, *, num_steps: int) -> dict:
    from safety_bigym.agents.cqn_as import env_adapter

    env = env_adapter.make(cfg)
    ts = env.reset()
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    action_shape = env.action_spec().shape

    sep_trace: List[float] = []
    prox_viol = 0
    ssm_viol = 0
    ssm_viol_actual = 0
    joint_counter: Counter = Counter()
    proximity_threshold = float("nan")
    episodes = 0

    for step in range(num_steps):
        action = rng.uniform(-1.0, 1.0, size=action_shape).astype(np.float32)
        ts = env.step(action)
        safety = ts.info.get("safety", {}) if isinstance(ts.info, dict) else {}

        sep = float(safety.get("min_separation", float("nan")))
        sep_trace.append(sep)
        proximity_threshold = float(safety.get("proximity_threshold", proximity_threshold))

        if bool(safety.get("proximity_violation", False)):
            prox_viol += 1
        if bool(safety.get("ssm_violation", False)):
            ssm_viol += 1
        if bool(safety.get("ssm_violation_actual", False)):
            ssm_viol_actual += 1
        joint = safety.get("closest_human_joint", "")
        if joint:
            joint_counter[joint] += 1

        if step % 50 == 0:
            logger.info(
                "step=%-4d min_sep=%6.3f m  joint=%-10s prox_viol=%s ssm_actual=%s",
                step,
                sep,
                joint or "?",
                bool(safety.get("proximity_violation", False)),
                bool(safety.get("ssm_violation_actual", False)),
            )

        if ts.last():
            episodes += 1
            ts = env.reset()

    env.close()

    sep_arr = np.array(sep_trace, dtype=np.float64)
    sep_finite = sep_arr[np.isfinite(sep_arr)]
    return {
        "n_steps": num_steps,
        "episodes_completed": episodes,
        "proximity_threshold": proximity_threshold,
        "min_separation_min": float(np.min(sep_finite)) if sep_finite.size else float("nan"),
        "min_separation_p5": float(np.percentile(sep_finite, 5)) if sep_finite.size else float("nan"),
        "min_separation_mean": float(np.mean(sep_finite)) if sep_finite.size else float("nan"),
        "proximity_violation_rate": prox_viol / num_steps,
        "ssm_violation_rate": ssm_viol / num_steps,
        "ssm_violation_actual_rate": ssm_viol_actual / num_steps,
        "closest_joint_histogram": dict(joint_counter.most_common()),
    }


@hydra.main(config_path="../cfgs", config_name="cqn_as_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    if OmegaConf.select(cfg, "env.task_name", default=None) is None:
        raise RuntimeError(
            "No env selected. Run e.g.:\n"
            "  python scripts/probe_coworker_violations.py "
            "env=safety_bigym/dishwasher_close disruption=coworker_train "
            "bodyslam=oracle pixels=false"
        )

    _check_amass_dir_or_die()

    OmegaConf.set_struct(cfg, False)
    if "pixels" not in cfg or cfg.pixels is None:
        cfg.pixels = False
    sc = cfg.env.setdefault("safety", OmegaConf.create({}))
    sc["log_violations"] = False

    num_steps = int(OmegaConf.select(cfg, "probe.num_steps", default=500))

    logger.info(
        "Probing: env=%s disruption=%s bodyslam=%s pixels=%s num_steps=%d",
        cfg.env.get("task_name"),
        cfg.get("disruption"),
        cfg.bodyslam.get("mode") if "bodyslam" in cfg else "off",
        cfg.pixels,
        num_steps,
    )

    stats = _probe(cfg, num_steps=num_steps)

    logger.info("=== Probe results ===")
    for k, v in stats.items():
        logger.info("  %s = %s", k, v)

    tau = stats["proximity_threshold"]
    logger.info(
        "Verdict: min_separation reached %.3f m (threshold %.2f m). "
        "proximity_violation fired on %.1f%% of steps; ssm_violation_actual on %.1f%%.",
        stats["min_separation_min"],
        tau,
        100.0 * stats["proximity_violation_rate"],
        100.0 * stats["ssm_violation_actual_rate"],
    )


if __name__ == "__main__":
    sys.exit(main())

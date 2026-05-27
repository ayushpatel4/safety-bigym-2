"""End-to-end smoke for the coworker swap.

Builds a SafetyBiGymEnv with the chosen ``human_model`` (g1 by default)
and ``disruption=COWORKER``, steps for ~30 s, and prints pelvis trajectory,
arm-phase transitions, and safety summary stats. Verification step (i) of
the plan.

Run from the repo root:
    cd safety_bigym && python scripts/g1_coworker_smoke.py
    # human:
    python scripts/g1_coworker_smoke.py --human g1
    python scripts/g1_coworker_smoke.py --human smplh
    # tighten coworker knobs to match cfgs/disruption/coworker_train.yaml
    python scripts/g1_coworker_smoke.py --stage train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# Per-stage continuous-knob bands — mirror cfgs/disruption/coworker_*.yaml so
# the smoke can drop in the same regime the curriculum exercises.
_STAGE_BANDS = {
    "default": {},  # use ParameterSpace defaults
    "idle": {
        "coworker_closest_approach_range": (3.0, 3.6),
        "coworker_reach_period_range": (30.0, 40.0),
        "coworker_target_mix_p_ee_range": (0.0, 0.0),
        "coworker_near_loiter_range": (1.0, 2.0),
        "coworker_walk_speed_range": (0.5, 0.8),
    },
    "easy": {
        "coworker_closest_approach_range": (1.8, 2.5),
        "coworker_reach_period_range": (8.0, 11.0),
        "coworker_target_mix_p_ee_range": (0.1, 0.25),
        "coworker_near_loiter_range": (3.0, 5.0),
        "coworker_walk_speed_range": (0.7, 1.1),
    },
    "train": {
        "coworker_closest_approach_range": (0.55, 0.85),
        "coworker_reach_period_range": (3.0, 5.0),
        "coworker_target_mix_p_ee_range": (0.55, 0.85),
        "coworker_near_loiter_range": (12.0, 18.0),
        "coworker_walk_speed_range": (1.0, 1.6),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", choices=["g1", "smplh"], default="g1")
    parser.add_argument(
        "--stage", choices=list(_STAGE_BANDS), default="default",
        help="Curriculum band to apply (mirrors cfgs/disruption/coworker_*.yaml).",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    from bigym.action_modes import JointPositionActionMode, PelvisDof

    from safety_bigym import HumanConfig, SafetyBiGymEnv, SafetyConfig
    from safety_bigym.scenarios.disruption_types import DisruptionType
    from safety_bigym.scenarios.scenario_sampler import ParameterSpace, ScenarioSampler

    band = _STAGE_BANDS[args.stage]
    # Force COWORKER only: weights = {COWORKER: 1.0}.
    param_space = ParameterSpace(
        clip_paths=[],
        disruption_weights={DisruptionType.COWORKER: 1.0},
        **band,
    )
    sampler = ScenarioSampler(parameter_space=param_space, motion_dir=None)

    action_mode = JointPositionActionMode(
        floating_base=True,
        absolute=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    )
    env = SafetyBiGymEnv(
        action_mode=action_mode,
        safety_config=SafetyConfig(log_violations=False),
        human_config=HumanConfig(human_model=args.human, motion_clip_paths=[]),
        scenario_sampler=sampler,
        inject_human=True,
        # Match the saucepan_to_hob task control frequency so per env-step
        # the human controller advances physics by 0.05 s (20 Hz). Without
        # this the raw env defaults to 500 Hz (1 substep) and ``hc.t``
        # crawls — 600 env-steps barely clears the initial approach
        # phase, so the loiter / reach cycle never fires in the smoke.
        control_frequency=20,
    )

    obs, info = env.reset(seed=0)

    pelvis_id = env._human_pelvis_id
    print(
        f"{args.human} ready. nbody={env._mojo.model.nbody}, "
        f"nu={env._mojo.model.nu}, ngeom={env._mojo.model.ngeom}"
    )
    print(f"Scenario: {info.get('scenario')} (stage={args.stage})")
    print(f"Initial Pelvis xpos: {env._mojo.data.xpos[pelvis_id]}")

    # Env runs at 20 Hz (control_frequency=20) → 0.05 s per env-step.
    control_dt = 1.0 / float(getattr(env, "_control_frequency", 20))
    n_steps = max(1, int(args.seconds / control_dt))

    phases_seen: list[str] = []
    phase_counts: dict[str, int] = {}
    traj_phase_counts: dict[str, int] = {}
    pelvis_z = []
    min_seps = []
    pfl_ratios = []
    proximity_hits = 0
    ssm_hits = 0
    ssm_actual_hits = 0
    n_with_safety = 0

    for i in range(n_steps):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        pelvis_z.append(float(env._mojo.data.xpos[pelvis_id, 2]))

        safety = info.get("safety", {}) or {}
        if safety:
            n_with_safety += 1
            if "min_separation" in safety:
                min_seps.append(float(safety["min_separation"]))
            if "pfl_force_ratio" in safety:
                pfl_ratios.append(float(safety["pfl_force_ratio"]))
            if bool(safety.get("proximity_violation")):
                proximity_hits += 1
            if bool(safety.get("ssm_violation")):
                ssm_hits += 1
            if bool(safety.get("ssm_violation_actual")):
                ssm_actual_hits += 1

        if env._coworker_controller is not None:
            phase = env._coworker_controller.last_phase
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            if not phases_seen or phases_seen[-1] != phase:
                phases_seen.append(phase)

        hc = env.human_controller
        if hc is not None and getattr(hc, "_trajectory_planner", None) is not None:
            _, _, _, traj_phase = hc._trajectory_planner.get_pose(hc.t)
            traj_phase_counts[traj_phase] = traj_phase_counts.get(traj_phase, 0) + 1

        if terminated or truncated:
            obs, info = env.reset(seed=i)

    print()
    print("=== SMOKE SUMMARY ===")
    print(f"steps run: {n_steps}  ({args.human}, stage={args.stage})")
    print(f"pelvis z  min/max: {min(pelvis_z):.3f} / {max(pelvis_z):.3f}")
    print(f"arm-phase transitions seen: "
          f"{' -> '.join(phases_seen[:20])}"
          + (" ..." if len(phases_seen) > 20 else ""))
    print(f"arm-phase step counts: {phase_counts}")
    print(f"trajectory phase counts: {traj_phase_counts}")
    if min_seps:
        print(f"SSM min_separation  min / median / max: "
              f"{min(min_seps):.3f} / {float(np.median(min_seps)):.3f} / "
              f"{max(min_seps):.3f}")
    if pfl_ratios:
        print(f"PFL max force_ratio: {max(pfl_ratios):.3f}")
    if n_with_safety:
        print(f"violation rates ({n_with_safety} safety-bearing steps):  "
              f"proximity={proximity_hits / n_with_safety:.2%}  "
              f"ssm_worst={ssm_hits / n_with_safety:.2%}  "
              f"ssm_actual={ssm_actual_hits / n_with_safety:.2%}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

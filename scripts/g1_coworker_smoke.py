"""End-to-end smoke for the G1 coworker swap.

Builds a SafetyBiGymEnv with ``human_model=g1`` and ``disruption=COWORKER``,
steps it for ~30 s, and prints pelvis trajectory, arm-phase transitions,
and safety summary stats. Verification step (i) of the plan.

Run from the repo root:
    cd safety_bigym && python scripts/g1_coworker_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from bigym.action_modes import JointPositionActionMode, PelvisDof

    from safety_bigym import HumanConfig, SafetyBiGymEnv, SafetyConfig
    from safety_bigym.scenarios.disruption_types import DisruptionType
    from safety_bigym.scenarios.scenario_sampler import ParameterSpace, ScenarioSampler

    # Force COWORKER only: weights = {COWORKER: 1.0}.
    param_space = ParameterSpace(
        clip_paths=[],
        disruption_weights={DisruptionType.COWORKER: 1.0},
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
        human_config=HumanConfig(human_model="g1", motion_clip_paths=[]),
        scenario_sampler=sampler,
        inject_human=True,
    )

    obs, info = env.reset(seed=0)

    pelvis_id = env._human_pelvis_id
    print(
        f"G1 ready. nbody={env._mojo.model.nbody}, "
        f"nu={env._mojo.model.nu}, ngeom={env._mojo.model.ngeom}"
    )
    print(f"Scenario: {info.get('scenario')}")
    print(f"Initial Pelvis xpos: {env._mojo.data.xpos[pelvis_id]}")

    # 30 s at the env's control_dt
    control_dt = env.control_dt if hasattr(env, "control_dt") else 0.05
    n_steps = max(1, int(30.0 / control_dt))

    phases_seen: list[str] = []
    pelvis_z = []
    min_seps = []
    pfl_ratios = []

    for i in range(n_steps):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        pelvis_z.append(float(env._mojo.data.xpos[pelvis_id, 2]))

        safety = info.get("safety", {}) or {}
        if "min_separation" in safety:
            min_seps.append(float(safety["min_separation"]))
        if "pfl_force_ratio" in safety:
            pfl_ratios.append(float(safety["pfl_force_ratio"]))

        if env._coworker_controller is not None:
            phase = env._coworker_controller.last_phase
            if not phases_seen or phases_seen[-1] != phase:
                phases_seen.append(phase)

        if terminated or truncated:
            obs, info = env.reset(seed=i)

    print()
    print("=== SMOKE SUMMARY ===")
    print(f"steps run: {n_steps}")
    print(f"pelvis z  min/max: {min(pelvis_z):.3f} / {max(pelvis_z):.3f}")
    print(f"arm-phase transitions: {' -> '.join(phases_seen[:20])}"
          + (" ..." if len(phases_seen) > 20 else ""))
    if min_seps:
        print(f"SSM min_separation  min/median: {min(min_seps):.3f} / "
              f"{float(np.median(min_seps)):.3f}")
    if pfl_ratios:
        print(f"PFL max force_ratio: {max(pfl_ratios):.3f}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

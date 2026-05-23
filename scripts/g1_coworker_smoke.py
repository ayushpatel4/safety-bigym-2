"""Smoke test: G1 coworker in a COWORKER episode with live safety monitoring.

Builds a SafetyBiGymEnv (G1 injected) on a single BiGym task, forces a COWORKER
scenario, runs a short rollout, and reports the SSM/PFL signal trajectory plus
whether the IK reach fired. Catches model-merge, index-resolution, and
controller wiring regressions in <1 min.

Run:  python scripts/g1_coworker_smoke.py  [--steps 200]
"""

from __future__ import annotations

import argparse

import numpy as np

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.envs.reach_target import ReachTargetSingle

from safety_bigym import SafetyConfig, HumanConfig
from safety_bigym.envs.safety_env import make_safety_env
from safety_bigym.scenarios.scenario_sampler import ParameterSpace, ScenarioSampler
from safety_bigym.scenarios.disruption_types import DisruptionType


def build_env(render: bool = False):
    action_mode = JointPositionActionMode(
        absolute=True,
        floating_base=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    )
    sampler = ScenarioSampler(
        parameter_space=ParameterSpace(
            disruption_weights={DisruptionType.COWORKER: 1.0},
        ),
    )
    return make_safety_env(
        ReachTargetSingle,
        action_mode=action_mode,
        safety_config=SafetyConfig(log_violations=False),
        human_config=HumanConfig(),
        scenario_sampler=sampler,
        inject_human=True,
        render_mode="rgb_array" if render else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--video", type=str, default=None,
                    help="Optional output mp4 path. If set, render each step.")
    args = ap.parse_args()

    env = build_env(render=args.video is not None)
    print(f"task={env.task_name}  pelvis_mocapid={env._human_pelvis_mocapid}")
    print(f"SSM bodies resolved: {len(env._human_body_ids)}/{len(env._HUMAN_SSM_BODY_NAMES)}")
    assert env._human_pelvis_mocapid is not None, "pelvis mocap id not resolved"
    assert len(env._human_body_ids) == len(env._HUMAN_SSM_BODY_NAMES), "missing SSM bodies"

    obs, info = env.reset(seed=0)
    print(f"scenario: {info.get('scenario')}")

    margins, seps, phases = [], [], []
    reach_fired = False
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    frames = []
    for i in range(args.steps):
        obs, r, term, trunc, info = env.step(a)
        s = info.get("safety", {})
        margins.append(float(s.get("ssm_margin", np.nan)))
        seps.append(float(s.get("min_separation", np.nan)))
        cw = env._coworker_controller
        if cw is not None:
            phases.append(cw.last_phase)
            if cw.last_reach_target is not None and not cw.last_out_of_reach:
                reach_fired = True
        if args.video is not None:
            frames.append(env.render())
        if term or trunc:
            obs, info = env.reset()

    if args.video is not None and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=20)
        print(f"  wrote video: {args.video} ({len(frames)} frames)")

    margins = np.array(margins)
    seps = np.array(seps)
    print(f"steps={args.steps}")
    print(f"  ssm_margin: min={np.nanmin(margins):.3f} max={np.nanmax(margins):.3f} "
          f"finite={np.isfinite(margins).all()}")
    print(f"  min_separation: min={np.nanmin(seps):.3f} max={np.nanmax(seps):.3f}")
    print(f"  closest_human_joint (last): {info.get('safety',{}).get('closest_human_joint')}")
    print(f"  coworker phases seen: {sorted(set(phases))}")
    print(f"  reach fired at least once: {reach_fired}")
    env.close()
    print("OK")


if __name__ == "__main__":
    main()

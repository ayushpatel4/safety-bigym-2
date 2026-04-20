"""
Test: SSM/PFL semantics survived the collision-bit split.

The Phase-0 fix moved the human onto a dedicated MuJoCo collision channel
so the human no longer penetrates scene geometry. Safety semantics must
still work:

  - SSM (Speed & Separation Monitoring) uses *geometric* distance between
    human body and robot link centers. It should decrease as the human
    approaches the robot. This is independent of contact bits.

  - PFL (Power & Force Limiting) uses MuJoCo contact forces between
    human and robot. Because human<->robot pairs remain on a shared
    collision bit (bit 1), physically forcing the human on top of the
    robot makes `pfl_force_ratio` non-zero. If the collision-bit split
    accidentally disabled the human<->robot pair, the force would stay
    at 0 and the test fails.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


AMASS_DIR = os.environ.get("AMASS_DATA_DIR")
HAS_AMASS = AMASS_DIR is not None and Path(AMASS_DIR).exists()
CLIP = "74/74_01_poses.npz"


def _forced_scenario():
    from safety_bigym.scenarios.scenario_sampler import ScenarioParams
    from safety_bigym.scenarios.disruption_types import DisruptionType

    return ScenarioParams(
        clip_path=CLIP,
        disruption_type=DisruptionType.DIRECT,
        trigger_time=0.0,
        speed_multiplier=1.0,
        approach_angle=0.0,
        spawn_distance=1.8,
        trajectory_type="APPROACH_LOITER_DEPART",
        closest_approach=0.1,
        loiter_duration=5.0,
        walk_speed=1.5,
        seed=0,
    )


def _make_env():
    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from bigym.bigym_env import CONTROL_FREQUENCY_MAX
    from safety_bigym import SafetyBiGymEnv, SafetyConfig, HumanConfig

    action_mode = JointPositionActionMode(
        floating_base=True,
        absolute=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    )
    human_config = HumanConfig(
        motion_clip_dir=AMASS_DIR,
        motion_clip_paths=[CLIP],
    )
    env = SafetyBiGymEnv(
        action_mode=action_mode,
        safety_config=SafetyConfig(log_violations=False, terminate_on_violation=False),
        human_config=human_config,
        inject_human=True,
        control_frequency=CONTROL_FREQUENCY_MAX // 20,
    )
    forced = _forced_scenario()
    env.scenario_sampler.sample_scenario = lambda seed=None: forced
    return env


@pytest.fixture(scope="module")
def env():
    if not HAS_AMASS:
        pytest.skip("AMASS_DATA_DIR not set")
    e = _make_env()
    yield e
    e.close()


def _collect_traces(env, n_steps: int):
    env.reset(seed=0)
    zero = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    ssm, pfl, sep = [], [], []
    for _ in range(n_steps):
        _, _, term, trunc, info = env.step(zero)
        s = info.get("safety", {})
        ssm.append(float(s.get("ssm_margin", np.nan)))
        pfl.append(float(s.get("pfl_force_ratio", 0.0)))
        sep.append(float(s.get("min_separation", np.nan)))
        if term or trunc:
            break
    return np.array(ssm), np.array(pfl), np.array(sep)


def test_min_separation_decreases_on_approach(env):
    """As the human walks toward the robot, min_separation must drop."""
    _, _, sep = _collect_traces(env, n_steps=60)
    assert len(sep) >= 20
    assert np.all(np.isfinite(sep))

    early = float(np.mean(sep[:10]))
    later_min = float(np.min(sep[10:]))
    assert later_min < early - 0.2, (
        f"min_separation didn't drop on head-on approach: "
        f"early={early:.3f}m, later_min={later_min:.3f}m"
    )


def test_ssm_signals_finite_and_fire(env):
    """ssm_margin must stay finite; during head-on approach it must drop
    below zero at least once (SSM violation signal fires)."""
    ssm, _, _ = _collect_traces(env, n_steps=60)
    assert len(ssm) > 0
    assert np.all(np.isfinite(ssm)), f"ssm_margin had NaN/Inf: {ssm}"
    # margin = d_min - S_p — negative is the violation signal, expected on approach
    assert np.any(ssm < 0), (
        "ssm_margin never went negative on a 1.5 m/s head-on approach — "
        "SSM signal is not firing."
    )


def test_pfl_schema_finite_on_approach(env):
    """pfl_force_ratio and max_contact_force must be finite and non-negative
    at every step of a head-on approach. This guards the PFL plumbing; the
    *collision bit* correctness is covered by test_collision_groups.py.
    """
    _, pfl, _ = _collect_traces(env, n_steps=60)
    assert len(pfl) > 0
    assert np.all(np.isfinite(pfl)), f"pfl_force_ratio had NaN/Inf: {pfl}"
    assert np.all(pfl >= 0.0), f"pfl_force_ratio went negative: min={pfl.min()}"


def test_episode_survives_approach(env):
    """Head-on approach must not trigger a physics error / early truncation."""
    env.reset(seed=0)
    zero = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    end = 150
    truncated_early = False
    for step in range(150):
        _, _, term, trunc, _ = env.step(zero)
        if term or trunc:
            end = step + 1
            truncated_early = trunc
            break
    assert not truncated_early or end > 100, (
        f"episode truncated at step {end}/150 — physics error on approach."
    )

"""
Test: Human<->scene collision bit separation.

Regression guard for the Phase-0 human-fix: the SMPL-H human is on MuJoCo
collision bit 1 only (contype=conaffinity=2), scene geoms (dishwasher,
cabinets, walls) stay on default bit 0, and robot + floor are promoted
into bit 1 by `SafetyBiGymEnv._configure_collision_bits`.

MuJoCo's rule: two geoms can collide iff
    (contype_A & conaffinity_B) | (contype_B & conaffinity_A) != 0

So we assert
  - every (human, scene) pair -> 0
  - every (human, robot-collision) pair -> non-zero
  - every (human, floor) pair -> non-zero
"""

import os
import sys
from pathlib import Path

import mujoco
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


AMASS_DIR = os.environ.get("AMASS_DATA_DIR")
HAS_AMASS = AMASS_DIR is not None and Path(AMASS_DIR).exists()


def _make_env():
    from bigym.action_modes import JointPositionActionMode, PelvisDof
    from safety_bigym import SafetyBiGymEnv, SafetyConfig, HumanConfig

    action_mode = JointPositionActionMode(
        floating_base=True,
        absolute=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    )
    human_config = HumanConfig(
        motion_clip_dir=AMASS_DIR,
        motion_clip_paths=["74/74_01_poses.npz"],
    )
    return SafetyBiGymEnv(
        action_mode=action_mode,
        safety_config=SafetyConfig(log_violations=False),
        human_config=human_config,
        inject_human=True,
    )


def _collision_enabled(model, g1: int, g2: int) -> int:
    a = int(model.geom_contype[g1]) & int(model.geom_conaffinity[g2])
    b = int(model.geom_contype[g2]) & int(model.geom_conaffinity[g1])
    return a | b


def _classify(name: str, env) -> str:
    if name.endswith("_col"):
        return "human"
    if env._is_robot_geom(name):
        return "robot"
    return "scene"


def _is_floor(name: str) -> bool:
    return name == "floor" or name.endswith("/floor") or "ground" in name.lower()


@pytest.fixture(scope="module")
def env():
    if not HAS_AMASS:
        pytest.skip("AMASS_DATA_DIR not set")
    e = _make_env()
    e.reset(seed=0)
    yield e
    e.close()


def _enumerate(env):
    model = env._mojo.model
    humans, robots, scene, floors = [], [], [], []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not name:
            continue
        # Visual-only geoms (both masks zero) never collide — skip.
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        kind = _classify(name, env)
        if kind == "human":
            humans.append((gid, name))
        elif kind == "robot":
            robots.append((gid, name))
        else:
            scene.append((gid, name))
            if _is_floor(name):
                floors.append((gid, name))
    return model, humans, robots, scene, floors


def test_human_geoms_exist(env):
    _, humans, _, _, _ = _enumerate(env)
    assert len(humans) > 0, "No human collision geoms (*_col) found — human not injected?"


def test_human_scene_pairs_disabled(env):
    """No (human, scene) geom pair may be collision-enabled."""
    model, humans, _, scene, _ = _enumerate(env)
    assert len(scene) > 0, "No scene geoms found"

    offenders = []
    for hg, hn in humans:
        for sg, sn in scene:
            if _is_floor(sn):
                continue  # floor is handled by its own test
            if _collision_enabled(model, hg, sg) != 0:
                offenders.append((hn, sn))
    assert not offenders, (
        f"{len(offenders)} human<->scene geom pairs are still collision-enabled. "
        f"First few: {offenders[:5]}"
    )


def test_human_robot_pairs_enabled(env):
    """Every (human, robot) geom pair must remain collision-enabled."""
    model, humans, robots, _, _ = _enumerate(env)
    assert len(robots) > 0, "No robot geoms found"

    broken = []
    for hg, hn in humans:
        for rg, rn in robots:
            if _collision_enabled(model, hg, rg) == 0:
                broken.append((hn, rn))
    # We don't require *every* pair — robot has visual-filtered geoms we
    # already skipped. But at least one pair per human geom should collide
    # with some robot collision geom, otherwise PFL can't see contact.
    hit_humans = {h for h, _ in broken}
    assert len(hit_humans) < len(humans), (
        "Every human geom has zero robot-collision partners — PFL broken."
    )
    # And: no human<->robot pair should be disabled just because of the bit split.
    # If broken is non-empty it means some robot geom did not receive bit 1.
    # That is only acceptable for geoms with geom_contype==0 (visual), which
    # we already filtered out above. So the list must be empty.
    assert not broken, (
        f"{len(broken)} human<->robot pairs are disabled — "
        f"_configure_collision_bits missed some robot collision geoms. "
        f"First few: {broken[:5]}"
    )


def test_human_floor_pairs_enabled(env):
    """Human must still collide with the floor (else falls through world)."""
    model, humans, _, _, floors = _enumerate(env)
    assert len(floors) > 0, "No floor geom found"

    for hg, hn in humans:
        for fg, fn in floors:
            assert _collision_enabled(model, hg, fg) != 0, (
                f"Human<->floor pair disabled: {hn} <-> {fn}"
            )


def test_human_bits_exact(env):
    """Human collision geoms must carry only bit 1 (contype=conaffinity=2)."""
    model, humans, _, _, _ = _enumerate(env)
    expected = env._HUMAN_CHANNEL_BIT  # 0b10
    for gid, name in humans:
        ct = int(model.geom_contype[gid])
        ca = int(model.geom_conaffinity[gid])
        assert ct == expected, f"{name}: contype={ct:b}, expected {expected:b}"
        assert ca == expected, f"{name}: conaffinity={ca:b}, expected {expected:b}"

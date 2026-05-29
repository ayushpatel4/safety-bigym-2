"""SSM + PFL integration: G1 bodies are tracked, all _col geoms map to regions."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.human import g1_human_spec  # noqa: E402
from safety_bigym.safety import pfl_limits  # noqa: E402


ASSET = Path(__file__).parent.parent / "safety_bigym" / "assets" / "g1_human_body.xml"


@pytest.fixture(scope="module")
def model():
    if not ASSET.exists():
        pytest.skip("g1_human_body.xml not generated yet")
    return mujoco.MjModel.from_xml_path(str(ASSET))


def test_ssm_body_list_matches_model(model):
    """Every name in g1_human_spec.SSM_BODY_NAMES must resolve in the model.

    Mirrors what SafetyBiGymEnv._setup_human_ssm_bodies does: each name is
    looked up via mj_name2id; misses silently drop the body from SSM. We
    want zero misses for G1.
    """
    missing = []
    for name in g1_human_spec.SSM_BODY_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            missing.append(name)
    assert not missing, f"SSM bodies missing in g1 model: {missing}"


def test_pfl_covers_every_col_geom(model):
    """Every collision geom in the G1 model has an ISO body region mapping."""
    unmapped = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not (name and name.endswith("_col")):
            continue
        if pfl_limits.get_region_for_geom(name) is None:
            unmapped.append(name)
    assert not unmapped, (
        f"PFL has no region for {len(unmapped)} G1 geoms: {unmapped}. "
        "Add entries to safety_bigym/safety/pfl_limits.py::GEOM_TO_REGION."
    )


def test_pfl_g1_new_regions_present():
    """Sanity: the G1-specific entries actually live in the map."""
    g1_only = [
        "L_Thigh_col", "R_Thigh_col",
        "L_Shin_col", "R_Shin_col",
        "L_Foot_col", "R_Foot_col",
        "L_Hand_col", "R_Hand_col",
    ]
    for geom in g1_only:
        assert pfl_limits.get_region_for_geom(geom) is not None, (
            f"GEOM_TO_REGION missing G1 entry for {geom}"
        )

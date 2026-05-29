"""Unit tests for Phase 3 P3.0a workspace reward shaping.

Exercises ``SafetyBiGymEnv._compute_workspace_penalty`` in isolation via a
stub object so the tests do not require MuJoCo, AMASS data, or any BiGym task
class. Full integration through ``_reward()`` and ``step()`` is verified by
the P3.0d smoke script.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from safety_bigym.config import SafetyConfig
from safety_bigym.envs.safety_env import SafetyBiGymEnv


def _stub(ee_pos, task_pos, *, cfg, ee_missing: bool = False, via_link_pos: bool = False):
    """Build a SimpleNamespace whose _get_robot_state returns the controlled state dict.

    ``_compute_workspace_penalty`` now reads through ``_get_robot_state`` (see
    H1's missing ``get_ee_position`` and the ``link_pos['ee']`` fallback in
    the env). This stub mirrors that surface.

    - ``ee_missing=True``: simulate a state dict with no ee_pos AND no link_pos['ee']
      (e.g. mid-rebind robot)
    - ``via_link_pos=True``: simulate H1's case — top-level ``ee_pos`` absent
      but ``link_pos['ee']`` populated by the _ROBOT_LINK_NAMES fallback
    """
    state: dict = {}
    if not ee_missing:
        if via_link_pos:
            state["link_pos"] = {"ee": ee_pos}
        else:
            state["ee_pos"] = ee_pos
    if task_pos is not None:
        state["task_object_pos"] = task_pos
    stub = SimpleNamespace(safety_config=cfg)
    stub._get_robot_state = lambda: state
    return stub


def test_inside_radius_zero_penalty():
    cfg = SafetyConfig(
        add_workspace_penalty=True, workspace_radius=0.4, workspace_beta=0.2
    )
    stub = _stub(np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.0]), cfg=cfg)
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == 0.0


def test_at_radius_boundary_zero_penalty():
    cfg = SafetyConfig(
        add_workspace_penalty=True, workspace_radius=0.4, workspace_beta=0.2
    )
    stub = _stub(np.array([0.4, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg)
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == pytest.approx(0.0)


def test_outside_radius_linear_negative():
    cfg = SafetyConfig(
        add_workspace_penalty=True, workspace_radius=0.4, workspace_beta=0.2
    )
    # dist=1.0, excess=0.6, expected penalty = -beta * 0.6 = -0.12
    stub = _stub(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg)
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == pytest.approx(-0.12)


def test_outside_radius_scales_with_beta():
    cfg = SafetyConfig(
        add_workspace_penalty=True, workspace_radius=0.4, workspace_beta=1.0
    )
    stub = _stub(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg)
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == pytest.approx(-0.6)


def test_outside_radius_scales_linearly_with_excess():
    cfg = SafetyConfig(
        add_workspace_penalty=True, workspace_radius=0.2, workspace_beta=0.5
    )
    near = _stub(np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg)
    far = _stub(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg)
    p_near = SafetyBiGymEnv._compute_workspace_penalty(near)
    p_far = SafetyBiGymEnv._compute_workspace_penalty(far)
    # excess_far - excess_near = (1.0 - 0.2) - (0.5 - 0.2) = 0.5 → diff = -beta * 0.5 = -0.25
    assert (p_far - p_near) == pytest.approx(-0.25)


def test_missing_task_object_returns_zero():
    """Tasks that don't expose a manipulable (e.g. some BiGym tasks) get no penalty."""
    cfg = SafetyConfig(
        add_workspace_penalty=True, workspace_radius=0.4, workspace_beta=0.2
    )
    stub = _stub(np.array([1.0, 0.0, 0.0]), None, cfg=cfg)
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == 0.0


def test_ee_lookup_exception_returns_zero():
    """If the robot is mid-rebind and EE is briefly unavailable, no penalty."""
    cfg = SafetyConfig(
        add_workspace_penalty=True, workspace_radius=0.4, workspace_beta=0.2
    )
    stub = _stub(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        cfg=cfg,
        ee_missing=True,
    )
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == 0.0


def test_ee_via_link_pos_fallback_used():
    """H1 path: top-level ee_pos absent but link_pos['ee'] populated. Penalty must fire."""
    cfg = SafetyConfig(
        add_workspace_penalty=True, workspace_radius=0.4, workspace_beta=0.2
    )
    stub = _stub(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        cfg=cfg,
        via_link_pos=True,
    )
    # dist=1.0, excess=0.6, expected penalty = -0.12 (same math as outside-radius case)
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == pytest.approx(-0.12)


def test_3d_distance_metric_is_euclidean():
    """Confirms penalty uses 3D Euclidean distance, not e.g. xy-only."""
    # cap disabled so the test measures the raw distance, not the bound.
    cfg = SafetyConfig(
        add_workspace_penalty=True,
        workspace_radius=0.0,
        workspace_beta=1.0,
        workspace_excess_cap=None,
    )
    # EE at (1,1,1), task at origin → ||·|| = sqrt(3) ≈ 1.732
    stub = _stub(
        np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg
    )
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == pytest.approx(
        -float(np.sqrt(3))
    )


# ---------- workspace_excess_cap: bounded penalty (2026-05-20 fix) ----------


def test_excess_cap_saturates_penalty_beyond_cap():
    """Beyond the cap the penalty is flat at -beta*cap (de-saturates the critic)."""
    cfg = SafetyConfig(
        add_workspace_penalty=True,
        workspace_radius=0.4,
        workspace_beta=0.05,
        workspace_excess_cap=1.0,
    )
    # dist=5.0 → raw excess=4.6, but capped to 1.0 → penalty = -0.05 * 1.0
    far = _stub(np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg)
    farther = _stub(np.array([9.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg)
    assert SafetyBiGymEnv._compute_workspace_penalty(far) == pytest.approx(-0.05)
    # Two distances both past the cap give the SAME penalty (the saturation we want).
    assert SafetyBiGymEnv._compute_workspace_penalty(
        farther
    ) == SafetyBiGymEnv._compute_workspace_penalty(far)


def test_excess_cap_inactive_below_cap():
    """Below the cap the penalty equals the uncapped linear value."""
    cfg = SafetyConfig(
        add_workspace_penalty=True,
        workspace_radius=0.4,
        workspace_beta=0.05,
        workspace_excess_cap=1.0,
    )
    # dist=1.0 → excess=0.6 < cap → penalty = -0.05 * 0.6 = -0.03
    stub = _stub(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg)
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == pytest.approx(-0.03)


def test_excess_cap_none_reproduces_unbounded_behaviour():
    """cap=None must reproduce the original linear (pre-fix) penalty."""
    cfg = SafetyConfig(
        add_workspace_penalty=True,
        workspace_radius=0.4,
        workspace_beta=0.2,
        workspace_excess_cap=None,
    )
    # dist=5.0 → excess=4.6, unbounded → -0.2 * 4.6 = -0.92
    stub = _stub(np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg)
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == pytest.approx(-0.92)


def test_support_invariant_holds_for_defaults():
    """beta * cap / (1 - gamma) must stay within the widened critic support |v_min|.

    Guards the design invariant from docs/phase3_base_validation_findings.md so a
    future beta/cap bump can't silently re-saturate the C51 target clamp. gamma and
    v_min are the curriculum run's values (agent.v_min=-6, gamma=0.99).
    """
    cfg = SafetyConfig()
    gamma = 0.99
    v_min_abs = 6.0
    discounted_floor = cfg.workspace_beta * cfg.workspace_excess_cap / (1.0 - gamma)
    assert discounted_floor <= v_min_abs


def _scene_attrs_stub(task_name=None):
    """Stub mirroring SafetyBiGymEnv's class-attr defaults so the unbound
    method can read ``_TASK_OBJECT_ATTRS_SCENE`` via ``self``."""
    stub = SimpleNamespace(
        _TASK_OBJECT_ATTRS_SCENE=SafetyBiGymEnv._TASK_OBJECT_ATTRS_SCENE,
    )
    if task_name is not None:
        stub.task_name = task_name
    return stub


def test_scene_attrs_default_order_for_unknown_task():
    """Non-WallCupboard tasks use the static ``_TASK_OBJECT_ATTRS_SCENE``."""
    stub = _scene_attrs_stub(task_name="SafetySaucepanToHob")
    order = SafetyBiGymEnv._scene_attrs_for_task(stub)
    assert order == SafetyBiGymEnv._TASK_OBJECT_ATTRS_SCENE


def test_scene_attrs_prefer_cabinet_wall_for_wallcupboard_close():
    """WallCupboardClose has BOTH cabinet_drawers and cabinet_wall set; the
    workspace-shaping lookup must pick cabinet_wall first (the task target),
    not cabinet_drawers (the base counter)."""
    stub = _scene_attrs_stub(task_name="SafetyWallCupboardClose")
    order = SafetyBiGymEnv._scene_attrs_for_task(stub)
    assert order[0] == "cabinet_wall"
    # cabinet_drawers must still be in the fallback list (other tasks need it).
    assert "cabinet_drawers" in order


def test_scene_attrs_prefer_cabinet_wall_for_wallcupboard_open():
    """Same override applies to WallCupboardOpen (mirror-image of close)."""
    stub = _scene_attrs_stub(task_name="SafetyWallCupboardOpen")
    order = SafetyBiGymEnv._scene_attrs_for_task(stub)
    assert order[0] == "cabinet_wall"


def test_scene_attrs_missing_task_name_falls_back():
    """No task_name attribute -> default order (defensive)."""
    stub = _scene_attrs_stub(task_name=None)
    order = SafetyBiGymEnv._scene_attrs_for_task(stub)
    assert order == SafetyBiGymEnv._TASK_OBJECT_ATTRS_SCENE


def test_config_defaults_are_base_validation_fix_values():
    """Defaults updated 2026-05-20: beta lowered + bounded penalty.

    See docs/phase3_base_validation_findings.md. The unbounded beta=0.2 dense
    penalty saturated the C51 critic; defaults now ship the bounded form.
    """
    cfg = SafetyConfig()
    assert cfg.add_workspace_penalty is False
    assert cfg.workspace_radius == 0.4
    assert cfg.workspace_beta == 0.05
    assert cfg.workspace_excess_cap == 1.0

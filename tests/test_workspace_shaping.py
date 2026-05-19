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
    cfg = SafetyConfig(
        add_workspace_penalty=True, workspace_radius=0.0, workspace_beta=1.0
    )
    # EE at (1,1,1), task at origin → ||·|| = sqrt(3) ≈ 1.732
    stub = _stub(
        np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]), cfg=cfg
    )
    assert SafetyBiGymEnv._compute_workspace_penalty(stub) == pytest.approx(
        -float(np.sqrt(3))
    )


def test_config_defaults_are_phase3_plan_values():
    """Defaults must match UPDATED_PROJECT_PLAN.md:337 (r_ws=0.4, beta=0.2)."""
    cfg = SafetyConfig()
    assert cfg.add_workspace_penalty is False
    assert cfg.workspace_radius == 0.4
    assert cfg.workspace_beta == 0.2

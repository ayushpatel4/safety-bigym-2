"""Tests for the Lagrange-multiplier PID controller (Phase 3 P3.1).

Pure-Python / torch-only -- imports the dependency-light
``safety_bigym.agents.cqn_as.lagrangian`` module so they run without
``tensordict`` installed.
"""

from safety_bigym.agents.cqn_as.lagrangian import LagrangianPID


def test_increases_monotonically_when_cost_above_budget():
    pid = LagrangianPID(k_i=1e-2, k_p=1e-1, k_d=0.0, lambda_max=1e6, cost_budget=0.01)
    lams = [pid.update(0.5) for _ in range(20)]
    assert lams[0] > 0.0
    assert all(b >= a for a, b in zip(lams, lams[1:]))  # non-decreasing


def test_decreases_when_cost_below_budget():
    pid = LagrangianPID(
        k_i=1e-2, k_p=1e-1, k_d=0.0, lambda_max=1e6, cost_budget=0.5, lambda_init=10.0
    )
    lams = [pid.update(0.0) for _ in range(20)]  # cost 0 << budget 0.5
    assert all(b <= a for a, b in zip(lams, lams[1:]))  # non-increasing


def test_clamps_to_lambda_max():
    pid = LagrangianPID(k_i=1.0, k_p=1.0, k_d=0.0, lambda_max=0.05, cost_budget=0.0)
    for _ in range(50):
        lam = pid.update(1.0)
    assert lam == 0.05


def test_clamps_to_zero_floor():
    pid = LagrangianPID(
        k_i=1.0, k_p=1.0, k_d=0.0, lambda_max=100.0, cost_budget=1.0, lambda_init=0.2
    )
    for _ in range(50):
        lam = pid.update(0.0)  # always below budget -> driven negative, clamped at 0
    assert lam == 0.0


def test_at_budget_holds_lambda_steady():
    pid = LagrangianPID(cost_budget=0.1, lambda_init=3.0, k_d=0.0)
    lam = pid.update(0.1)  # exactly at budget -> cv == 0 -> no change
    assert lam == 3.0


def test_state_dict_roundtrip():
    pid = LagrangianPID(cost_budget=0.01)
    for _ in range(5):
        pid.update(0.3)
    snap = pid.state_dict()
    restored = LagrangianPID(cost_budget=0.01)
    restored.load_state_dict(snap)
    assert restored.lam == pid.lam
    assert restored._prev_violation == pid._prev_violation

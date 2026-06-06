"""Pure-python unit tests for CBFDodgeFilter — the geometric Control-Barrier-Function
directional-dodge safety filter.

The CBF filter is ALWAYS-ON and minimally invasive: when the human is within ``d_target``
of the robot base it offsets the absolute base X,Y target along the away-direction
``(robot_xy - human_xy)``, leaving Z/RZ/arm untouched; otherwise it passes the proposed
action through unchanged. These tests construct the filter against a small Box action space
with synthetic obs dicts (no MuJoCo / env) and assert the math, the index-locality, the
clipping, and the fail-safes.
"""

import gymnasium as gym
import numpy as np
import pytest

from safety_bigym.filters.cbf_filter import CBFDodgeFilter


def _space(dim=16, lo=-10.0, hi=10.0):
    return gym.spaces.Box(low=lo, high=hi, shape=(dim,), dtype=np.float32)


def _obs(human_xy, base_xy, human_vel_xy=(0.0, 0.0)):
    """Synthetic raw obs dict matching the env schema.

    human_pos_estimate: [x, y, z, vx, vy, vz]; proprioception_floating_base: [x, y, z, rz].
    """
    return {
        "human_pos_estimate": np.array(
            [human_xy[0], human_xy[1], 1.0, human_vel_xy[0], human_vel_xy[1], 0.0],
            np.float32,
        ),
        "proprioception_floating_base": np.array(
            [base_xy[0], base_xy[1], 0.0, 0.0], np.float32
        ),
    }


def _proposed(dim=16):
    """A non-trivial proposed action so 'unchanged' tests are meaningful."""
    return np.linspace(-0.5, 0.5, dim).astype(np.float32)


# --- 1. sep >= d_target -> unchanged, intervened=False --------------------------------

def test_outside_barrier_returns_unchanged():
    fb = CBFDodgeFilter(_space(), d_target=0.45)
    proposed = _proposed()
    # human at origin, base 2 m away (>> d_target) -> safe -> pass-through.
    out, info = fb.apply(_obs((0.0, 0.0), (2.0, 0.0)), proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)
    # barrier is positive when safe.
    assert info["h"] > 0.0


def test_just_outside_barrier_passes_through():
    # sep slightly > d_target -> h > 0 -> pass-through (avoid float32 knife-edge at ==).
    fb = CBFDodgeFilter(_space(), d_target=0.45)
    proposed = _proposed()
    out, info = fb.apply(_obs((0.0, 0.0), (0.50, 0.0)), proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)


def test_just_inside_barrier_intervenes():
    # sep slightly < d_target -> h < 0 -> dodge.
    fb = CBFDodgeFilter(_space(), d_target=0.45, use_velocity=False)
    proposed = _proposed()
    out, info = fb.apply(_obs((0.0, 0.0), (0.44, 0.0)), proposed)
    assert info["intervened"] is True
    assert info["push"] > 0.0


# --- 2. sep < d_target -> base XY moved along away-dir by the capped CBF magnitude -----

def test_inside_barrier_pushes_along_away_direction():
    fb = CBFDodgeFilter(_space(), d_target=0.45, gain=1.0, max_push=0.15,
                        use_velocity=False)
    proposed = np.zeros(16, np.float32)
    sep = 0.40  # inside barrier: human at origin, base at (0.40, 0) -> away = +x.
    out, info = fb.apply(_obs((0.0, 0.0), (sep, 0.0)), proposed)
    assert info["intervened"] is True
    expected_push = np.clip(1.0 * (0.45 - sep), 0.0, 0.15)  # = 0.05
    # base target = proposed(0) + away_unit(+x)*push.
    assert np.isclose(out[0], 0.0 + expected_push, atol=1e-5)
    assert np.isclose(out[1], 0.0, atol=1e-5)
    assert np.isclose(info["push"], expected_push, atol=1e-6)


def test_push_is_capped_at_max_push():
    # Deep violation -> gain*(d_target-sep) would exceed max_push -> capped.
    fb = CBFDodgeFilter(_space(), d_target=0.45, gain=5.0, max_push=0.15,
                        use_velocity=False)
    proposed = np.zeros(16, np.float32)
    sep = 0.05  # gain*(0.45-0.05)=2.0 -> capped to 0.15.
    out, info = fb.apply(_obs((0.0, 0.0), (sep, 0.0)), proposed)
    assert info["intervened"] is True
    assert np.isclose(info["push"], 0.15, atol=1e-6)
    assert np.isclose(out[0], 0.15, atol=1e-5)


def test_diagonal_away_direction_unit_normalised():
    fb = CBFDodgeFilter(_space(), d_target=1.0, gain=1.0, max_push=10.0,
                        use_velocity=False)
    proposed = np.zeros(16, np.float32)
    # human at origin, base at (0.3, 0.4) -> sep=0.5, away_unit=(0.6, 0.8).
    out, info = fb.apply(_obs((0.0, 0.0), (0.3, 0.4)), proposed)
    push = 1.0 * (1.0 - 0.5)  # = 0.5
    assert np.isclose(info["push"], push, atol=1e-6)
    assert np.isclose(out[0], 0.6 * push, atol=1e-5)
    assert np.isclose(out[1], 0.8 * push, atol=1e-5)


# --- 3. only indices 0,1 change; 2,3,4..15 identical to proposed ----------------------

def test_only_base_xy_changes():
    fb = CBFDodgeFilter(_space(), d_target=0.45, use_velocity=False)
    proposed = _proposed()
    out, info = fb.apply(_obs((0.0, 0.0), (0.30, 0.10)), proposed)
    assert info["intervened"] is True
    # indices 2..15 (Z, RZ, arm/gripper) must be byte-identical to the proposed action.
    assert np.array_equal(out[2:], proposed[2:])
    # and at least one of the base-XY entries actually moved.
    assert not np.allclose(out[:2], proposed[:2])


# --- 4. output clipped within action bounds ------------------------------------------

def test_output_clipped_to_action_bounds():
    fb = CBFDodgeFilter(_space(lo=-1.0, hi=1.0), d_target=0.45, gain=10.0,
                        max_push=5.0, use_velocity=False)
    # base near the +x bound, away = +x, big push -> target would exceed +1, clipped.
    proposed = np.zeros(16, np.float32)
    proposed[0] = 0.99
    out, info = fb.apply(_obs((0.0, 0.0), (0.30, 0.0)), proposed)
    assert info["intervened"] is True
    assert out[0] <= 1.0 + 1e-6
    assert np.all(out <= 1.0 + 1e-6) and np.all(out >= -1.0 - 1e-6)


# --- 5. degenerate sep ~ 0 -> unchanged ----------------------------------------------

def test_degenerate_direction_returns_unchanged():
    fb = CBFDodgeFilter(_space(), d_target=0.45)
    proposed = _proposed()
    # human exactly on the base -> away vector ~0 -> direction undefined -> pass-through.
    out, info = fb.apply(_obs((2.0, 2.0), (2.0, 2.0)), proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)
    assert info.get("reason") == "degenerate"


# --- 6. missing 'human_pos_estimate' -> unchanged (inactive), no crash ----------------

def test_missing_human_key_returns_unchanged_no_crash():
    fb = CBFDodgeFilter(_space(), d_target=0.45)
    proposed = _proposed()
    obs = {"proprioception_floating_base": np.zeros(4, np.float32)}  # human key absent
    out, info = fb.apply(obs, proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)
    assert info.get("reason") == "missing_obs"


def test_missing_human_key_warns_only_once(caplog):
    import logging

    fb = CBFDodgeFilter(_space(), d_target=0.45)
    proposed = _proposed()
    obs = {"proprioception_floating_base": np.zeros(4, np.float32)}
    with caplog.at_level(logging.WARNING, logger="safety_bigym.filters.cbf_filter"):
        for _ in range(5):
            fb.apply(obs, proposed)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1  # one-time warning, not per-step spam.


def test_empty_obs_returns_unchanged():
    fb = CBFDodgeFilter(_space(), d_target=0.45)
    proposed = _proposed()
    out, info = fb.apply({}, proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)


# --- 7. approach-velocity term increases push toward, not away ------------------------

def test_approach_velocity_increases_push():
    fb = CBFDodgeFilter(_space(), d_target=0.45, gain=1.0, max_push=10.0,
                        use_velocity=True, beta=0.5)
    proposed = np.zeros(16, np.float32)
    sep = 0.40
    base_push = 1.0 * (0.45 - sep)  # = 0.05 (geometric-only)

    # human at origin, base at (0.40, 0): away_unit = +x (human -> robot). A human
    # CHASING the robot moves +x, so approach_speed = dot((1,0), away_unit=(1,0)) = +1.0
    # -> push += beta*1.0.
    out_toward, info_toward = fb.apply(
        _obs((0.0, 0.0), (sep, 0.0), human_vel_xy=(1.0, 0.0)), proposed
    )
    assert info_toward["push"] > base_push
    assert np.isclose(info_toward["push"], base_push + 0.5 * 1.0, atol=1e-6)

    # Human RECEDING (vel = -x, away from robot) -> closing speed negative -> no extra push.
    out_away, info_away = fb.apply(
        _obs((0.0, 0.0), (sep, 0.0), human_vel_xy=(-1.0, 0.0)), proposed
    )
    assert np.isclose(info_away["push"], base_push, atol=1e-6)

    # And the approaching case dodges strictly harder than the receding case.
    assert info_toward["push"] > info_away["push"]
    assert abs(out_toward[0]) > abs(out_away[0])


def test_velocity_term_disabled_ignores_human_velocity():
    fb = CBFDodgeFilter(_space(), d_target=0.45, gain=1.0, max_push=10.0,
                        use_velocity=False, beta=0.5)
    proposed = np.zeros(16, np.float32)
    sep = 0.40
    out, info = fb.apply(
        _obs((0.0, 0.0), (sep, 0.0), human_vel_xy=(5.0, 0.0)), proposed
    )
    assert np.isclose(info["push"], 1.0 * (0.45 - sep), atol=1e-6)  # velocity ignored


# --- misc: type guard + non-finite fail-safe -----------------------------------------

def test_non_box_action_space_raises():
    with pytest.raises(TypeError):
        CBFDodgeFilter(gym.spaces.Discrete(4))  # type: ignore[arg-type]


def test_nonfinite_obs_returns_unchanged():
    fb = CBFDodgeFilter(_space(), d_target=0.45)
    proposed = _proposed()
    obs = {
        "human_pos_estimate": np.array([np.nan, 0.0, 1.0, 0.0, 0.0, 0.0], np.float32),
        "proprioception_floating_base": np.array([0.0, 0.0, 0.0, 0.0], np.float32),
    }
    out, info = fb.apply(obs, proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)


# --- in-loop transform (mirrors CQNASRunner.step's CBF branch, torch-free) ------------
# CQNASRunner does, per step:
#   raw_proposed = adapter._convert_action_to_raw(sub_action)   # [-1,1] -> raw
#   raw_corrected, info = cbf.apply(obs, raw_proposed)
#   sub_action = adapter._convert_action_from_raw(raw_corrected)  # raw -> [-1,1]
#   if info["intervened"]: intervention_count += 1
# This reproduces that exact sequence with the real adapter conversion math so the
# locally-untestable CQN-AS path is de-risked without MuJoCo/torch.

class _StubAdapter:
    """Minimal stand-in with the real de/normalisation math from SafetyBiGymCQNAdapter."""

    def __init__(self, action_min, action_max):
        self._action_stats = {
            "min": np.asarray(action_min, np.float32),
            "max": np.asarray(action_max, np.float32),
        }

    def _convert_action_to_raw(self, action):
        action = np.asarray(action, np.float32)
        amin, amax = self._action_stats["min"], self._action_stats["max"]
        scaled = (action + 1.0) / 2.0
        return (scaled * (amax - amin + 1e-8) + amin).astype(np.float32)

    def _convert_action_from_raw(self, action):
        action = np.asarray(action, np.float32)
        amin, amax = self._action_stats["min"], self._action_stats["max"]
        scaled = (action - amin) / (amax - amin + 1e-8)
        return (scaled * 2.0 - 1.0).astype(np.float32)


def _runner_cbf_step(adapter, cbf, obs, sub_action_norm):
    raw_proposed = adapter._convert_action_to_raw(sub_action_norm)
    raw_corrected, info = cbf.apply(obs, raw_proposed)
    sub_action = np.asarray(adapter._convert_action_from_raw(raw_corrected), np.float32)
    return sub_action, info


def test_inloop_roundtrip_passthrough_when_safe():
    # Raw action space [-10,10]^16; normalised sub-action round-trips through raw.
    adapter = _StubAdapter([-10.0] * 16, [10.0] * 16)
    cbf = CBFDodgeFilter(_space(lo=-10.0, hi=10.0), d_target=0.45, use_velocity=False)
    sub = np.linspace(-0.8, 0.8, 16).astype(np.float32)
    out, info = _runner_cbf_step(adapter, cbf, _obs((0.0, 0.0), (2.0, 0.0)), sub)
    assert info["intervened"] is False
    assert np.allclose(out, sub, atol=1e-5)  # safe -> identity round-trip


def test_inloop_roundtrip_dodges_base_xy_when_close():
    adapter = _StubAdapter([-10.0] * 16, [10.0] * 16)
    cbf = CBFDodgeFilter(_space(lo=-10.0, hi=10.0), d_target=0.45, gain=1.0,
                         max_push=0.15, use_velocity=False)
    # sub-action whose RAW base X = 0.40 (so it matches the base position; away=+x).
    # raw = (norm+1)/2 * 20 - 10  ->  norm = raw/10. raw 0.40 -> norm 0.04.
    sub = np.zeros(16, np.float32)
    sub[0] = 0.04  # raw 0.40
    out, info = _runner_cbf_step(adapter, cbf, _obs((0.0, 0.0), (0.40, 0.0)), sub)
    assert info["intervened"] is True
    # raw correction = 0.40 + push(0.05) = 0.45 -> norm 0.045.
    assert np.isclose(out[0], 0.045, atol=1e-4)
    # arm DOFs unchanged through the round-trip.
    assert np.allclose(out[2:], sub[2:], atol=1e-5)


# =====================================================================================
# CBFRetractFilter — EE-retract ("flinch") math, with a synthetic Jacobian (no MuJoCo).
#
# The retract filter keeps the base planted and maps a capped Cartesian EE retract
# u*push to ARM joint targets via dq = damped_pinv(J_arm) @ (u*push), writing
# arm_qpos + dq into the arm action indices (absolute joint targets). These tests feed
# a known J / ee_pos / human_pos / arm_qpos / arm_action_idx and assert the linear
# algebra, index-locality, clipping, and every fail-safe pass-through.
# =====================================================================================

from safety_bigym.filters.cbf_filter import CBFRetractFilter, _damped_pinv

# Action layout under test: [0..3]=base, [4..13]=10 arm joints, [14,15]=grippers.
_ARM_IDX = list(range(4, 14))
_N_ARM = len(_ARM_IDX)


def _J_identity3(n=_N_ARM):
    """A 3xN Jacobian whose first three columns are the identity (so dq's first 3
    entries equal damped_pinv-scaled v) and the rest zero. Deterministic + invertible
    J J^T = I_3."""
    J = np.zeros((3, n), dtype=np.float64)
    J[0, 0] = J[1, 1] = J[2, 2] = 1.0
    return J


def _J_random(n=_N_ARM, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((3, n))


def _state(ee_pos, human_pos, J, *, qarm=None, idx=None):
    if qarm is None:
        qarm = np.zeros(J.shape[1], np.float64)
    if idx is None:
        idx = list(range(4, 4 + J.shape[1]))
    return {
        "ee_pos": np.asarray(ee_pos, np.float64),
        "human_pos": np.asarray(human_pos, np.float64),
        "J_arm": np.asarray(J, np.float64),
        "arm_qpos": np.asarray(qarm, np.float64),
        "arm_action_idx": np.asarray(idx, int),
    }


def _ret_space(dim=16, lo=-10.0, hi=10.0):
    return gym.spaces.Box(low=lo, high=hi, shape=(dim,), dtype=np.float32)


# --- damped pseudo-inverse identity ---------------------------------------------------

def test_damped_pinv_matches_right_pinv_formula():
    J = _J_random()
    lam = 0.05
    got = _damped_pinv(J, lam)
    expect = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(3))
    assert np.allclose(got, expect)
    # zero damping on a full-row-rank J -> exact right Moore-Penrose pinv.
    assert np.allclose(_damped_pinv(J, 0.0), np.linalg.pinv(J), atol=1e-8)


# --- 1. core dq = damped_pinv(J) @ (away_unit * push) ---------------------------------

def test_dq_equals_damped_pinv_times_cartesian_step():
    d_target, gain, max_push, damping = 0.45, 1.0, 1.0, 0.05
    fb = CBFRetractFilter(_ret_space(), d_target=d_target, gain=gain,
                          max_push=max_push, damping=damping)
    # EE at (0.40,0,0), human at origin -> sep=0.40 (<d_target), away_unit=+x.
    J = _J_random(seed=3)
    qarm = np.linspace(0.1, 0.5, _N_ARM)
    proposed = np.zeros(16, np.float32)
    out, info = fb.apply(_state((0.40, 0.0, 0.0), (0.0, 0.0, 0.0), J, qarm=qarm), proposed)

    assert info["intervened"] is True
    sep = 0.40
    push = np.clip(gain * (d_target - sep), 0.0, max_push)  # 0.05
    assert np.isclose(info["push"], push)
    u = np.array([1.0, 0.0, 0.0])  # (ee - human)/sep
    dq_expect = _damped_pinv(J, damping) @ (u * push)
    target_expect = (qarm + dq_expect).astype(np.float32)
    assert np.allclose(out[_ARM_IDX], target_expect, atol=1e-5)
    assert np.isclose(info["dq_norm"], np.linalg.norm(dq_expect), atol=1e-6)


def test_identity_jacobian_gives_dq_along_away_direction():
    # With J's first 3 cols = I and damping=0: dq[:3] = u*push, dq[3:] = 0.
    fb = CBFRetractFilter(_ret_space(), d_target=0.45, gain=1.0, max_push=1.0,
                          damping=0.0)
    J = _J_identity3()
    proposed = np.zeros(16, np.float32)
    # diagonal away: ee at (0.3,0.4,0)*scale s.t. sep<0.45. Use sep=0.40 along (0.6,0.8).
    ee = np.array([0.6, 0.8, 0.0]) * 0.40
    out, info = fb.apply(_state(ee, (0.0, 0.0, 0.0), J), proposed)
    assert info["intervened"] is True
    push = 0.45 - 0.40
    # arm joints 0,1,2 (action idx 4,5,6) move by u*push; the rest stay at qarm=0.
    assert np.isclose(out[4], 0.6 * push, atol=1e-5)
    assert np.isclose(out[5], 0.8 * push, atol=1e-5)
    assert np.isclose(out[6], 0.0, atol=1e-5)
    assert np.allclose(out[7:14], 0.0, atol=1e-5)


# --- 2. only ARM indices change; base + grippers identical ----------------------------

def test_only_arm_indices_change():
    fb = CBFRetractFilter(_ret_space(), d_target=0.45, gain=1.0, max_push=1.0)
    J = _J_random(seed=5)
    proposed = _proposed(16)
    out, info = fb.apply(_state((0.40, 0.0, 0.0), (0.0, 0.0, 0.0), J), proposed)
    assert info["intervened"] is True
    # base (0..3) and grippers (14,15) byte-identical to the proposed action.
    assert np.array_equal(out[:4], proposed[:4])
    assert np.array_equal(out[14:], proposed[14:])
    # at least one arm DOF actually moved.
    assert not np.allclose(out[_ARM_IDX], proposed[_ARM_IDX])


# --- 3. pass-through when sep_ee >= d_target ------------------------------------------

def test_passthrough_when_outside_barrier():
    fb = CBFRetractFilter(_ret_space(), d_target=0.45)
    proposed = _proposed(16)
    # EE 2 m from human -> safe.
    out, info = fb.apply(_state((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), _J_random()), proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)
    assert info["h"] > 0.0


def test_passthrough_just_outside_barrier():
    fb = CBFRetractFilter(_ret_space(), d_target=0.45)
    proposed = _proposed(16)
    out, info = fb.apply(_state((0.50, 0.0, 0.0), (0.0, 0.0, 0.0), _J_random()), proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)


# --- 4. push capped at max_push -------------------------------------------------------

def test_push_capped_at_max_push():
    fb = CBFRetractFilter(_ret_space(), d_target=0.45, gain=5.0, max_push=0.10,
                          damping=0.0)
    J = _J_identity3()
    proposed = np.zeros(16, np.float32)
    # sep=0.05 -> gain*(0.45-0.05)=2.0, capped to 0.10.
    out, info = fb.apply(_state((0.05, 0.0, 0.0), (0.0, 0.0, 0.0), J), proposed)
    assert info["intervened"] is True
    assert np.isclose(info["push"], 0.10, atol=1e-6)
    # along +x with identity J & zero damping -> arm joint 0 moves exactly max_push.
    assert np.isclose(out[4], 0.10, atol=1e-5)


# --- 5. output clipped within action bounds ------------------------------------------

def test_output_clipped_to_action_bounds():
    fb = CBFRetractFilter(_ret_space(lo=-0.05, hi=0.05), d_target=0.45, gain=10.0,
                          max_push=5.0, damping=0.0)
    J = _J_identity3()
    qarm = np.zeros(_N_ARM)
    proposed = np.zeros(16, np.float32)
    # big push along +x; arm target would exceed +0.05 -> clipped.
    out, info = fb.apply(_state((0.05, 0.0, 0.0), (0.0, 0.0, 0.0), J, qarm=qarm), proposed)
    assert info["intervened"] is True
    assert np.all(out <= 0.05 + 1e-6) and np.all(out >= -0.05 - 1e-6)


# --- 6. fail-safes: degenerate / non-finite / missing-state / bad-shape ---------------

def test_degenerate_direction_passthrough():
    fb = CBFRetractFilter(_ret_space(), d_target=0.45)
    proposed = _proposed(16)
    # EE coincident with human -> away vector ~0 -> direction undefined.
    out, info = fb.apply(_state((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), _J_random()), proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)
    assert info.get("reason") == "degenerate"


def test_nonfinite_state_passthrough():
    fb = CBFRetractFilter(_ret_space(), d_target=0.45)
    proposed = _proposed(16)
    st = _state((np.nan, 0.0, 0.0), (0.0, 0.0, 0.0), _J_random())
    out, info = fb.apply(st, proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)
    assert info.get("reason") == "nonfinite"


def test_nonfinite_jacobian_passthrough():
    fb = CBFRetractFilter(_ret_space(), d_target=0.45)
    proposed = _proposed(16)
    J = _J_random()
    J[0, 0] = np.inf
    out, info = fb.apply(_state((0.40, 0.0, 0.0), (0.0, 0.0, 0.0), J), proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)


def test_missing_state_passthrough_and_warns_once(caplog):
    import logging

    fb = CBFRetractFilter(_ret_space(), d_target=0.45)
    proposed = _proposed(16)
    with caplog.at_level(logging.WARNING, logger="safety_bigym.filters.cbf_filter"):
        for _ in range(5):
            out, info = fb.apply(None, proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)
    assert info.get("reason") == "missing_state"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1  # one-time warning, not per-step spam


def test_partial_state_missing_key_passthrough():
    fb = CBFRetractFilter(_ret_space(), d_target=0.45)
    proposed = _proposed(16)
    st = _state((0.40, 0.0, 0.0), (0.0, 0.0, 0.0), _J_random())
    del st["J_arm"]  # one required field absent
    out, info = fb.apply(st, proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)
    assert info.get("reason") == "missing_state"


def test_shape_mismatch_passthrough():
    fb = CBFRetractFilter(_ret_space(), d_target=0.45)
    proposed = _proposed(16)
    # J has N=10 cols but arm_qpos/idx say N=5 -> inconsistent -> pass-through.
    st = _state((0.40, 0.0, 0.0), (0.0, 0.0, 0.0), _J_random(n=10),
                qarm=np.zeros(5), idx=list(range(4, 9)))
    out, info = fb.apply(st, proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)
    assert info.get("reason") == "bad_shape"


def test_out_of_range_action_index_passthrough():
    fb = CBFRetractFilter(_ret_space(), d_target=0.45)
    proposed = _proposed(16)
    # action index 99 is past the 16-dim action -> bad_idx pass-through (no crash).
    st = _state((0.40, 0.0, 0.0), (0.0, 0.0, 0.0), _J_random(n=2),
                qarm=np.zeros(2), idx=[4, 99])
    out, info = fb.apply(st, proposed)
    assert info["intervened"] is False
    assert np.array_equal(out, proposed)
    assert info.get("reason") == "bad_idx"


def test_non_box_action_space_raises_retract():
    with pytest.raises(TypeError):
        CBFRetractFilter(gym.spaces.Discrete(4))  # type: ignore[arg-type]


# --- 7. closest-human selection is the one the retract uses (state contract) ----------
# compute_ee_retract_state picks the closest human body to the EE; the math then uses
# whatever single human_pos it is handed. This guards the apply() contract: given a
# human_pos that is the *closest* body, the away-direction points EE<-that body.

def test_away_direction_uses_supplied_closest_human():
    fb = CBFRetractFilter(_ret_space(), d_target=1.0, gain=1.0, max_push=10.0,
                          damping=0.0)
    J = _J_identity3()
    # EE at +x of the human -> retract should push further +x (away from human).
    ee = np.array([0.5, 0.0, 0.0])
    out, info = fb.apply(_state(ee, (0.0, 0.0, 0.0), J), np.zeros(16, np.float32))
    assert info["intervened"] is True
    assert out[4] > 0.0  # moved along +x, away from the human


# --- in-loop transform (mirrors CQNASRunner.step's ee_retract branch) -----------------
# CQNASRunner does, per step:
#   state = compute_ee_retract_state(adapter._env)            # live MuJoCo (skipped here)
#   raw_proposed = adapter._convert_action_to_raw(sub_action)
#   raw_corrected, info = ee_retract_filter.apply(state, raw_proposed)
#   sub_action = adapter._convert_action_from_raw(raw_corrected)
# This reproduces the de/normalisation round-trip with a synthetic state.

def _runner_retract_step(adapter, fb, state, sub_action_norm):
    raw_proposed = adapter._convert_action_to_raw(sub_action_norm)
    raw_corrected, info = fb.apply(state, raw_proposed)
    sub_action = np.asarray(adapter._convert_action_from_raw(raw_corrected), np.float32)
    return sub_action, info


def test_inloop_retract_roundtrip_passthrough_when_safe():
    adapter = _StubAdapter([-10.0] * 16, [10.0] * 16)
    fb = CBFRetractFilter(_ret_space(lo=-10.0, hi=10.0), d_target=0.45)
    sub = np.linspace(-0.8, 0.8, 16).astype(np.float32)
    st = _state((2.0, 0.0, 0.0), (0.0, 0.0, 0.0), _J_random())  # safe
    out, info = _runner_retract_step(adapter, fb, st, sub)
    assert info["intervened"] is False
    assert np.allclose(out, sub, atol=1e-5)


def test_inloop_retract_roundtrip_moves_only_arm_when_close():
    adapter = _StubAdapter([-10.0] * 16, [10.0] * 16)
    fb = CBFRetractFilter(_ret_space(lo=-10.0, hi=10.0), d_target=0.45, gain=1.0,
                          max_push=0.15, damping=0.05)
    sub = np.linspace(-0.5, 0.5, 16).astype(np.float32)
    st = _state((0.40, 0.0, 0.0), (0.0, 0.0, 0.0), _J_random(seed=7),
                qarm=np.zeros(_N_ARM))
    out, info = _runner_retract_step(adapter, fb, st, sub)
    assert info["intervened"] is True
    # base + grippers survive the round-trip unchanged; some arm DOF moved.
    assert np.allclose(out[:4], sub[:4], atol=1e-5)
    assert np.allclose(out[14:], sub[14:], atol=1e-5)
    assert not np.allclose(out[_ARM_IDX], sub[_ARM_IDX], atol=1e-5)

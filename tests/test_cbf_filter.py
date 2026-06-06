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

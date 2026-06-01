"""Regression test: the SVF snapshot-collection policy must execute actions the
SAME way the deployment runner does (open-loop chunks + optional temporal
ensemble) — NOT receding-horizon ``chunk[0]``.

Why this matters (2026-06-01, the *second* SVF bug): CQN-AS deploys with
``action_sequence=16`` and ``temporal_ensemble=true``. If collection rolls out
the policy receding-horizon (re-plan every step, execute ``chunk[0]``), the SVF
critic trains on raw ``chunk[0]`` actions while deployment executes ensemble-
blended / open-loop sub-actions → every deployed action is OOD for the critic →
~89% spurious veto (success → 0). ``_CQNASSnapshotPolicy`` must therefore mirror
``benchmark.runners.CQNASRunner.step`` exactly. These tests pin that.
"""

import importlib.util as _u
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("dm_env")
pytest.importorskip("tensordict")  # svf_collect imports the CQN-AS adapter stack


def _load_snapshot_policy_cls():
    """Import ``_CQNASSnapshotPolicy`` from the script module (path-loaded).

    Registers the module in ``sys.modules`` before exec so the module's
    ``@dataclass`` definitions resolve their ``__module__``. Skips if the
    heavy env stack isn't importable (e.g. no MuJoCo in a bare CI).
    """
    try:
        spec = _u.spec_from_file_location(
            "svf_collect_for_test",
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "scripts" / "svf_collect_dataset.py",
        )
        m = _u.module_from_spec(spec)
        sys.modules["svf_collect_for_test"] = m
        spec.loader.exec_module(m)
        return m._CQNASSnapshotPolicy
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"svf_collect_dataset not importable here: {type(e).__name__}: {e}")


_SEQ = 16
_DIM = 16


class _StubAgent:
    """``act`` returns a deterministic chunk: sub-action i is filled with
    ``call_id*100 + i`` so executed sub-actions are identifiable by value."""

    def __init__(self):
        self.calls = 0

    def act(self, rgb, low_dim, step, eval_mode=True):
        self.calls += 1
        base = self.calls * 100.0
        return np.array([[base + i] * _DIM for i in range(_SEQ)], np.float32).reshape(-1)


def _make_policy(temporal_ensemble=None):
    SP = _load_snapshot_policy_cls()
    agent = _StubAgent()
    # Identity-ish de-norm: low=0, high=1 -> raw = (norm+1)/2, invert with norm = 2*raw-1.
    pol = SP(
        agent=agent,
        state_keys=("s",),
        includes_human_pos=False,
        camera_keys=(),
        frame_stack=1,
        action_sequence=_SEQ,
        action_low=np.zeros(_DIM, np.float32),
        action_high=np.ones(_DIM, np.float32),
        rgb_placeholder_shape=(1, 3, 8, 8),
        temporal_ensemble=temporal_ensemble,
    )
    return pol, agent


def _executed_id(raw_action):
    """Invert the (norm+1)/2 de-norm to recover the sub-action id."""
    return float(raw_action[0] * 2.0 - 1.0)


def test_open_loop_chunk_consumption():
    """No ensemble: re-plan every action_sequence steps; consume chunk in order."""
    pol, agent = _make_policy(temporal_ensemble=None)
    obs = {"s": np.zeros(4, np.float32)}
    ids = [round(_executed_id(pol(obs)), 1) for _ in range(_SEQ + 2)]
    # act() called once per chunk: 18 steps over seq=16 -> 2 calls.
    assert agent.calls == 2
    # First chunk consumed open-loop: chunk[0..3] = 100,101,102,103 (NOT 100,200,300).
    assert ids[:4] == [100.0, 101.0, 102.0, 103.0]
    # Step 16 crosses the boundary -> re-plan -> chunk[0] of the 2nd plan = 200.
    assert ids[_SEQ] == 200.0


def test_temporal_ensemble_matches_runner_math():
    """With a TemporalEnsembleControl: act EVERY step; executed == ensemble blend."""
    from dm_env import specs

    from safety_bigym.agents.cqn_as.utils import TemporalEnsembleControl

    spec = specs.BoundedArray(shape=(_DIM,), dtype=np.float32, minimum=-1.0, maximum=1.0)
    pol, agent = _make_policy(temporal_ensemble=TemporalEnsembleControl(50, spec, _SEQ))
    obs = {"s": np.zeros(4, np.float32)}
    got = [round(_executed_id(pol(obs)), 4) for _ in range(5)]

    # act called every step under the ensemble.
    assert agent.calls == 5
    # Step 0 has a single prediction -> equals raw chunk[0] = 100.
    assert abs(got[0] - 100.0) < 1e-3
    # Steps >=1 are exp-weighted blends, NOT any raw sub-action value.
    assert got[1] not in (100.0, 101.0, 200.0)

    # Parity: feeding the SAME agent outputs straight into a standalone ensemble
    # must reproduce the policy's executed actions exactly.
    spec2 = specs.BoundedArray(shape=(_DIM,), dtype=np.float32, minimum=-1.0, maximum=1.0)
    te = TemporalEnsembleControl(50, spec2, _SEQ)
    agent2 = _StubAgent()
    expected = []
    for _ in range(5):
        chunk = np.asarray(agent2.act(None, None, 0), np.float32).reshape(_SEQ, -1)
        te.register_action_sequence(chunk)
        expected.append(round(float(te.get_action()[0]), 4))
    assert got == expected


def test_reset_clears_episode_state():
    """reset() must restart chunk indexing so episode 2 begins with a fresh plan."""
    pol, agent = _make_policy(temporal_ensemble=None)
    obs = {"s": np.zeros(4, np.float32)}
    for _ in range(3):
        pol(obs)
    assert agent.calls == 1  # still inside the first chunk
    pol.reset()
    first_after_reset = round(_executed_id(pol(obs)), 1)
    # Fresh chunk[0] of a new plan (2nd act call) -> 200, not the 4th sub-action (103).
    assert agent.calls == 2
    assert first_after_reset == 200.0

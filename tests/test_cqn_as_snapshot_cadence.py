"""Regression test for the train_cqn_as snapshot cadence fix.

Before the fix, ``Workspace.train()`` only invoked ``save_snapshot()`` from
inside the ``if time_step.last():`` block. The ``utils.Every`` gate uses
exact-equality semantics (``step % every == 0``), so a snapshot landed only
when an episode boundary coincidentally fell on a step that was a multiple
of ``snapshot_every_frames``. For COWORKER train rollouts on
``saucepan_to_hob`` (episodes ~150 control-steps, stochastic termination),
200k-step C2 runs landed zero snapshots.

The fix hoists the snapshot decision outside the episode-end block so it
fires on every step at the configured cadence, plus a final save after the
``train_until_step`` loop exits.

This test pins the decision logic — not the full Workspace, which is too
heavy to construct without MuJoCo. It simulates a 200-step training loop
with stochastic episode boundaries and verifies the post-fix sequence saves
at every multiple of ``snapshot_every_frames`` regardless of where episodes
land.
"""

from __future__ import annotations

import random
from typing import List

from safety_bigym.agents.cqn_as.utils import Every


def _simulate_loop(
    *,
    total_steps: int,
    snapshot_every: int,
    action_repeat: int,
    save_snapshot: bool,
    episode_boundary_rng_seed: int,
    placement: str,
) -> List[int]:
    """Mimic the relevant slice of Workspace.train() and return saved-step list.

    ``placement`` selects the snapshot decision's call site:
    - ``"episode_end"``: pre-fix — only fires at episode boundaries.
    - ``"per_step"``: post-fix — fires on every step at the cadence, plus a
      final save after the loop exits.
    """
    every = Every(snapshot_every, action_repeat)
    rng = random.Random(episode_boundary_rng_seed)
    saved: List[int] = []

    global_step = 0
    while global_step < total_steps:
        # Decide whether the env returns time_step.last() this iteration.
        # Episode lengths in [80, 160] mimic COWORKER train on saucepan_to_hob.
        is_episode_end = rng.random() < (1.0 / rng.randint(80, 160))

        if placement == "episode_end" and is_episode_end:
            if save_snapshot and every(global_step):
                saved.append(global_step)

        # ... critic update, env.step, etc — irrelevant to the snapshot decision ...

        global_step += 1
        if placement == "per_step":
            if save_snapshot and every(global_step):
                saved.append(global_step)

    if placement == "per_step" and save_snapshot:
        # Final-state save mirrors the post-loop call.
        if not saved or saved[-1] != global_step:
            saved.append(global_step)

    return saved


def test_pre_fix_loop_can_drop_every_snapshot():
    """Sanity-check the harness: pre-fix placement reproduces the C2 zero-snapshot bug.

    With 200 steps, snapshot_every=20, action_repeat=1: expected exactly-aligned
    fires at {20, 40, ..., 200}. Episode boundaries land at random steps so
    the intersection is typically empty.
    """
    saved = _simulate_loop(
        total_steps=200,
        snapshot_every=20,
        action_repeat=1,
        save_snapshot=True,
        episode_boundary_rng_seed=0,
        placement="episode_end",
    )
    # Under the pre-fix placement the intersection is sparse-to-empty.
    # We only require that it doesn't match the full expected schedule —
    # the bug it reproduces.
    expected_full_schedule = list(range(20, 201, 20))
    assert saved != expected_full_schedule, (
        "Pre-fix placement somehow saved at every multiple — harness is broken"
    )


def test_post_fix_loop_saves_at_every_cadence_multiple():
    """Post-fix placement: every multiple of snapshot_every gets a snapshot."""
    saved = _simulate_loop(
        total_steps=200,
        snapshot_every=20,
        action_repeat=1,
        save_snapshot=True,
        episode_boundary_rng_seed=0,
        placement="per_step",
    )
    # The check is placed AFTER global_step += 1, so the first saved value is
    # 20 (not 0), and the loop exits when global_step == 200 — so step 200
    # is reached and saved (and we also add the final-save call after the
    # loop, but it's de-duplicated since 200 is a cadence multiple).
    assert saved == list(range(20, 201, 20))


def test_post_fix_final_save_when_total_not_a_cadence_multiple():
    """If total_steps isn't a clean multiple, the final save still captures the tail."""
    saved = _simulate_loop(
        total_steps=205,  # 5 steps past the last cadence multiple
        snapshot_every=20,
        action_repeat=1,
        save_snapshot=True,
        episode_boundary_rng_seed=0,
        placement="per_step",
    )
    # Cadence fires at 20..200, plus the final save at 205.
    assert saved == list(range(20, 201, 20)) + [205]


def test_post_fix_save_disabled_keeps_disk_clean():
    saved = _simulate_loop(
        total_steps=200,
        snapshot_every=20,
        action_repeat=1,
        save_snapshot=False,
        episode_boundary_rng_seed=0,
        placement="per_step",
    )
    assert saved == []


def test_action_repeat_divides_cadence():
    """utils.Every divides the cadence by action_repeat — verify the post-fix
    placement honours that and lands on the divided multiples.
    """
    saved = _simulate_loop(
        total_steps=400,
        snapshot_every=100,
        action_repeat=4,  # effective cadence = 25
        save_snapshot=True,
        episode_boundary_rng_seed=0,
        placement="per_step",
    )
    # Effective cadence 25 → fires at 25, 50, ..., 400.
    assert saved == list(range(25, 401, 25))


def test_post_fix_independent_of_episode_boundary_rng():
    """The post-fix save schedule must NOT depend on the episode-boundary RNG."""
    saved_a = _simulate_loop(
        total_steps=200,
        snapshot_every=20,
        action_repeat=1,
        save_snapshot=True,
        episode_boundary_rng_seed=0,
        placement="per_step",
    )
    saved_b = _simulate_loop(
        total_steps=200,
        snapshot_every=20,
        action_repeat=1,
        save_snapshot=True,
        episode_boundary_rng_seed=42,
        placement="per_step",
    )
    assert saved_a == saved_b

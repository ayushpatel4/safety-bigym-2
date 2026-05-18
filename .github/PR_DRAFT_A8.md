# Vendor CQN-AS + Phase 0.5 COWORKER + Phase 2 SVF dataset/critic

Replaces DrQ-V2+ with vendored CQN-AS as the Phase 3 actor, rewires Phase 2
dataset collection through the new COWORKER disruption ParameterSpace, and
trains the proximity-based SVF v1 critic end-to-end.

23 commits on top of `main`, ~6.3k lines added (≈2.2k from vendoring upstream
CQN-AS verbatim).

## Summary

- **Phase 0.5 — COWORKER disruption infrastructure.** New
  `coworker_behavior.py` with `STATIONARY` / `APPROACH_LOITER_DEPART` /
  `COWORKER_PATROL` trajectory modes; `scenario_sampler.py` extended with
  `make_coworker_train_space()` / `make_coworker_eval_space()` factories
  (strict-superset eval band on all 5 continuous knobs); Hydra group
  `cfgs/disruption/coworker_{train,eval}.yaml`. Docs at
  `docs/coworker_disruption.md`. 11-test suite at
  `tests/test_coworker_disruption.py`.

- **CQN-AS vendor.** Six modules under
  `safety_bigym/agents/cqn_as/` — `agent.py`, `cqn_utils.py`, `utils.py`,
  `replay_buffer.py` vendored verbatim from upstream
  (`/Users/ayushpatel/Documents/FYP3/CQN-AS/` at commit `8cf806e`); local
  `env_adapter.py` (477 LOC, not vendored) translates `SafetyBiGymEnv` to
  CQN-AS's TimeStep API and injects `human_pos_estimate` into low_dim_obs
  when `bodyslam.mode != "off"`. Adapter pytest (`test_cqn_as_adapter.py`,
  24 tests) covers obs shape ± bodyslam, action [-1,1] roundtrip
  (incl. gripper-tail), TimeStep first/mid/last, info["safety"] forwarding,
  frame-stacking, pixels=False, missing-key errors, ExtendedTimeStepWrapper.

- **Training entrypoint.** `train_cqn_as.py` + `cfgs/cqn_as_config.yaml`.
  Handles `num_demos=0` for smoke gates by skipping the demo replay buffer.
  W&B wired to `project=safety-critic`. Snapshot save/load round-trips
  encoder + critic + critic_target via new `state_dict`/`load_state_dict`
  on `CQNASAgent`. `+snapshot_path=...` triggers eager-load and
  `num_train_frames=0` triggers eval-only mode.

- **Phase 2 SVF dataset rewired for COWORKER.** `svf_collect_dataset.py`
  takes `--disruption-space {coworker_train,coworker_eval,legacy_multi}`.
  Shard schema extended with `min_separation` + `pfl_force_ratio` per
  transition for retroactive relabelling. **Geometric proximity replaces
  ISO 15066 SSM as the binary safety label** — `r_safe = 0` when
  `min_separation < proximity_threshold` (production 0.50m); ISO's
  required-separation formula demands ~5m clearance at kitchen-scale robot
  velocities, which produced a degenerate 93%-violation dataset.
  `ssm_margin` is still logged for ISO traceability and as the
  continuous cost signal for the Phase 3 Lagrangian.

- **Phase 0 ACT snapshots re-rolled** on COWORKER train space for
  `dishwasher_close`, `drawers_open_all`, `saucepan_to_hob` —
  `filters/snapshots.py` SNAPSHOTS dict updated.

- **E1.4 CQN-AS sweep script.** `scripts/phase1_reward_pilot_cqn_as.py`
  — 3 cells (`bodyslam=off|oracle|noisy`) × `saucepan_to_hob` × COWORKER
  train, `env.safety.add_violation_penalty=true`,
  `violation_penalty=0.05`, `num_train_frames=200000`. Mirrors the existing
  DrQ-V2+ pilot's CLI; `--train`/`--eval`/`--smoke` modes.

## Verification

- **Adapter pytest:** `pytest tests/test_cqn_as_adapter.py` → 24/24 green
  (pure-Python; monkeypatches `SafetyBiGymEnvFactory._create_env` so no
  MuJoCo/AMASS needed).
- **Existing test suites:** `tests/test_coworker_disruption.py` (11),
  `tests/test_svf_dataset.py` (8), `tests/test_safety_labeling.py` (9),
  `tests/test_bodyslam_wrapper.py` (19+1 skipped) — all green.
- **A6 smoke gate (GPU box, 2026-05-18):** `cqn_as_smoke_dishwasher_oracle_v4`
  ran 2000 frames / 38 episodes cleanly. Encoder shape ✓ (288→64 with
  oracle vs 264→64 with off, diff = 6D × frame_stack=4), COWORKER episode
  boundary ✓, per-step safety logging ✓, action-space latency ~150ms/step
  ≪ 640ms K-step budget ✓.
- **Phase 2 v1 dataset:** `datasets/svf_coworker_train_v1/` — 315k
  transitions, 3 tasks × 2 sources (random + Phase 0 ACT snapshots),
  ~11% aggregate violation rate, `min_separation` and `pfl_force_ratio`
  in shard schema. 4 sanity checks pass (B4 in IMPLEMENTATION_STATUS.md).
- **SVF v1 critic:** `checkpoints/svf_coworker_train_v1.pt` (627k).
  200k steps, final Bellman MSE 6.20, `q_mean=3.19`. Threshold sweep
  identifies an operating point at **R≈4.0** (intervention 97–99%,
  residual_violation 3–8% vs random's 95% baseline on `coworker_eval`).

## What's NOT in this PR

- **PFL retrofit.** `pfl_force_ratio` is in the v1 shard schema but
  identically zero because of an unresolved BiGym contact-detection bug
  (`info["safety"]["pfl_*"]` is always zero across all cells). Retrofit
  recipe is in `IMPLEMENTATION_STATUS.md` 2026-05-16 notes: re-collect
  through a PFL-fixed env, then `label_transition(..., use_pfl=True)` ORs
  in `pfl_violation` at train time. The v1 critic is proximity-only;
  PFL is Phase 4/5 polish if the BiGym fix lands in scope.

- **4-dof demo source.** B3 dataset has no demo source — BiGym DemoStore
  demos were recorded with 3-dof floating base (`pelvis_x/y/rz`), but B2.4
  moved Phase 2 env construction to 4 dofs (X, Y, Z, RZ) to match RoboBase
  ACT training. `random + snapshot` covers the safe-side distribution for
  v1; re-recording demos through the 4-dof env is a tracked follow-up
  (not blocking).

- **Snapshot action denormalization.** `_SnapshotPolicy.__call__` returns
  tanh-space output without `RescaleFromTanhWithMinMax` — snapshot rollouts
  explore a narrower action subspace than a deployed policy. Defensive fix
  (~10 LOC in `_SnapshotPolicy.__call__`) deferred; only fire if the v1
  critic underperforms in actor-coupled use.

- **`ScenarioParams` missing 2 axes.** `reach_period` and `target_mix_p_ee`
  are sampled by `make_coworker_train_space()` but not stored on the
  dataclass; B4.4 coverage audit covers 3/5 axes. Dataclass extension is
  a small follow-up.

## Risk / things to look at in review

- **CQN-AS vendor delta from upstream.** Verbatim copy of 4 modules with
  imports rebased to relative + provenance comments at file headers; no
  semantic changes. `agent.py` extended with `state_dict`/`load_state_dict`
  (not in upstream — upstream relies on demo-driven training that doesn't
  need standalone snapshot save).

- **Geometric proximity vs ISO 15066 SSM.** Decision is documented in
  `IMPLEMENTATION_STATUS.md` decision log (2026-05-16) and in the
  `label_transition` docstring. ISO traceability preserved via continuous
  `ssm_margin` for Phase 3 Lagrangian.

- **Dataset re-collection is a hard cost** if reviewers want to flip
  parameters that aren't already in the per-transition shard schema. The
  schema covers proximity threshold (free) and PFL (needs re-collection).

- **6.3k LOC of added/modified code** — biggest blocks are the vendored
  CQN-AS modules (≈2.2k, 4 files verbatim from upstream `8cf806e`) and
  the COWORKER disruption infrastructure (≈700 LOC for scenarios + tests).

## Test plan

- [ ] `cd safety_bigym && pytest tests/test_cqn_as_adapter.py tests/test_coworker_disruption.py tests/test_svf_dataset.py tests/test_safety_labeling.py tests/test_bodyslam_wrapper.py` → all green
- [ ] On GPU box: `python train_cqn_as.py env=safety_bigym/dishwasher_close disruption=coworker_train bodyslam=oracle num_train_frames=2000 num_demos=0 wandb.use=true wandb.name=a8_pr_smoke` → 2000 frames complete, W&B run shows `[train] step=N q_critic_loss=...`, `safety/ssm_margin`, `episode_safety/*` lines
- [ ] On GPU box: `python -u scripts/svf_train_critic.py --smoke --dataset-dir datasets/svf_coworker_train_v1 --output /tmp/svf_smoke.pt` → "Smoke OK: bellman MSE reduced ..."
- [ ] C2 training cells (`phase1_reward_pilot_cqn_as.py --train`) run to completion on GPU box without DataLoader collate errors (the `_copying_collate` fix is the load-bearing piece)

## Follow-ups (deferred to subsequent PRs)

- B5.5 — Snapshot action denormalization fix + v2 dataset re-collection (only if v1 critic underperforms in C2 eval)
- PFL retrofit (only when BiGym contact-detection bug lands in scope)
- 4-dof demo re-recording (only if SVF eval reveals safe-side undercoverage)
- `ScenarioParams` dataclass extension for `reach_period` + `target_mix_p_ee`
- Phase 3 — constrained-RL Lagrangian formulation using the v1 SVF as the
  hard safety filter; continuous `ssm_margin` as the cost signal

🤖 Generated with [Claude Code](https://claude.com/claude-code)

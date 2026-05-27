# Implementation Status — Hybrid Safety Critic

Last updated: 2026-05-27
Active branch: `retryg1` (forked off `main`; carries G1 coworker swap + 3-flavour safety metrics + stage-2 tighten — not yet committed/pushed)
Plan: [.claude/UPDATED_PROJECT_PLAN.md](UPDATED_PROJECT_PLAN.md)
Initial-phase plan: [/Users/ayushpatel/.claude/plans/claude-updated-project-plan-md-is-the-n-precious-bunny.md](../../.claude/plans/claude-updated-project-plan-md-is-the-n-precious-bunny.md)
Changes log: [.claude/CHANGES_AND_NEXT_STEPS.md](CHANGES_AND_NEXT_STEPS.md)
Phase 3 orientation: [PHASE3_OVERVIEW.md](PHASE3_OVERVIEW.md) (goal, what's done/left, contingencies, scope)
P3.1 handoff: [PHASE3_1_HANDOFF.md](PHASE3_1_HANDOFF.md) (paste-ready prompt for the Lagrangian-glue coding session)
Phase 2 writeup: [phase2_results.md](phase2_results.md) (implementation + B5 results + B5.5 plan)
Safety-metrics schema: [safety_metrics.md](safety_metrics.md) (per-step + per-episode + JSON dump contract — now fully implemented)
G1 swap (this branch): [g1_coworker_swap.md](g1_coworker_swap.md) (design + verification + curriculum hand-off)

---

## Next session — start here

**The SMPL-H base curriculum on saucepan_to_hob finished cleanly on 2026-05-27 (run dir `exp_local/cqn_as_base_curriculum/base_curriculum_20260527_015253`).** Stage 1 hit `success=1.0` by step ~11k; stage 2 stayed at 0.8-1.0 throughout. The branch has since landed three structural changes (G1 swap, three-flavour safety metrics, tighter stage 2) and is ready to run with `HUMAN_MODEL=g1`.

### 🟢 Closed today (2026-05-27)

1. **G1 humanoid swap landed** on `retryg1` as a fresh implementation (the previous `safety-critic/g1-coworker` attempt is NOT a reference — user instructed clean retry). G1 lives behind `env.human_model=g1` (default stays `smplh`). Parallel classes `G1HumanController` / `G1HumanIK` mirror the SMPL-H interface; SMPL-H code path is byte-untouched. Initial design used skin-toned collision capsules (strategy α); per supervisor feedback the asset now renders with the upstream Unitree STL meshes (real-robot appearance). Asset-merge gap closed (`_create_merged_world` now copies `<asset>` blocks and absolutises mesh paths). Full design + verification: [g1_coworker_swap.md](g1_coworker_swap.md). 23 G1-specific tests + `test_collision_groups` parametrized over both human models.
2. **Three-flavour safety metrics implemented** (the doc [safety_metrics.md](safety_metrics.md) was prescriptive until today; now in code). `info["safety"]` emits `ssm_violation` / `ssm_violation_actual` / `proximity_violation` plus margins, observed velocities, and `proximity_threshold` echo. `EpisodeSafetyMetrics` now emits the full thesis-grade `ep_*` schema (proximity dwell at 0.3/0.5/1.0 m, separation min/mean/p5/p25, robot-vel max/mean, both SSM-margin troughs, etc.). `train_cqn_as` now writes `metrics.jsonl` (streaming) + `final_metrics.json` (headline + `best_eval`), forwards `wandb.tags` to W&B init, and emits `episode_cost_integral` every episode-end (+ `episode_lambda` when the agent exposes `_lambda` for P3.1). Eval cycles aggregate `info["episode_safety"]` across rollouts into `eval/ep_*`. 8 new tests in `test_safety_metrics_three_flavours.py`.
3. **Stage 2 disruption tightened** (`cfgs/disruption/coworker_train.yaml`): `closest_approach 0.9-1.4 → 0.55-0.85`, `reach_period 4.5-6.5 → 3.0-5.0`, `target_mix_p_ee 0.4-0.6 → 0.55-0.85`, `near_loiter 7-11 → 12-18`. 60-second smoke at 20 Hz now shows the arm reliably cycles extend/hold/retract, proximity_violation_rate ≈ 87 % (G1) / 88 % (SMPL-H), min separation ~0.02 m. Identical params apply to both human models (body-agnostic).
4. **SMPL-H controller fallback fix.** `HumanController._get_amass_targets` ignored the trajectory planner when `clip is None`, parking the pelvis at spawn. Now the no-clip path uses planner XY/yaw — matches `G1HumanController`. Doesn't affect production (AMASS is loaded there) but fixes smokes / future no-AMASS contingencies.

### 🔴 Do this first — next curriculum run on the GPU box

The previous run finished and the user wants to (a) test the G1 visuals + tighter stage 2, (b) reallocate stage budget per the prev run's signal. Sequence:

```bash
cd ~/Documents/safety_bigym
git pull       # or sync the retryg1 branch files manually if not pushed yet
python scripts/build_g1_human_body.py     # regenerates the merged-XML (paths in checked-in XML are relative; portable)
venv/bin/python -m pytest tests/test_g1_asset.py tests/test_g1_human_controller.py \
  tests/test_g1_safety_tracking.py tests/test_safety_metrics_three_flavours.py -q   # sanity (≤5 s)

export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export AMASS_DATA_DIR=/path/to/CMU/CMU      # required if HUMAN_MODEL=smplh
HUMAN_MODEL=g1 \
STAGE0_FRAMES=20000 STAGE1_FRAMES=15000 STAGE2_FRAMES=60000 \
CUDA_VISIBLE_DEVICES=2 scripts/run_base_curriculum.sh
```

**Stage-length rationale** (from the 2026-05-27 run analysis, full reasoning under "Notes / 2026-05-27"):
- **Stage 0 → 20k** (was 30k). Eval reward peaked at ~step 13-15k then degraded to half by step 28k. 20k captures the peak with margin. If new G1 visuals hit the encoder again, kill early and fall back to strategy α (skin-tone capsules — recoverable from the prior commit on this branch).
- **Stage 1 → 15k** (was 30k). Hit `success=1.0` by step 10909 in the prior run and stayed there. Saves wall-time.
- **Stage 2 → 60k** (was 40k). New distribution is structurally harder (closest_approach 0.55-0.85 m, proximity-violation rate ≈ 87 % in smoke vs 13-24 % in the prior run). Give it ~50 % more frames to adapt.

Expectations / tripwires:
- Stage 0 reward dip vs the prior anchor (`-7.2`) is OK as long as it recovers — the G1 visuals are the new variable. Kill if it's still under -10 by step 15k.
- Stage 2 early reward will drop below the prior `-1.87` (much harder distribution). Don't compare apples-to-apples on the reward axis; track `ep_proximity_violation_rate` and `success_rate` separately.

### After this run — P3.1 still next

P3.1 code (Lagrangian glue) was code-complete + unit-tested on `safety-critic/phase-3-constrained-rl` as of 2026-05-21. Once the G1 base curriculum lands a usable snapshot, the Lagrangian smoke (`agent=cqn_as_lagrangian`) is the next milestone. The `episode_lambda` / `episode_cost_integral` W&B keys are now wired in `train_cqn_as._lagrangian_payload` (cost integral fires on the unconstrained baseline too — useful for "what would λ have been pushing on").

**Why D existed:** the C2 (E1.4) re-run came back **degenerate** (31-step episodes, robot fleeing the human, identical off/oracle/noisy curves). Diagnosed as `num_demos=0` (CQN-AS is demo-driven) + no workspace shaping. The demo pipeline is now wired.

### 🔴 Do this first — GPU-box smokes for Workstream D, then Phase 3

`SafetyBiGymCQNAdapter.get_demos()` is implemented; D0/D1/D2/D3a/D3b-smoke are done. The 2000-frame smoke on the GPU box passed cleanly on 2026-05-20 (10 demos loaded, replay filled, train loop clean, ep1 ran the full 1000-step budget — the smoking-gun signal that demos unblock non-degenerate learning). **One run left:**
- **~30–50k-frame validation** with demos + workspace shaping (single bodyslam mode, oracle):
  ```bash
  export AMASS_DATA_DIR=/path/to/CMU/CMU MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
  python train_cqn_as.py env=safety_bigym/saucepan_to_hob disruption=coworker_train \
    bodyslam=oracle num_demos=10 num_train_frames=50000 \
    env.safety.add_workspace_penalty=true \
    wandb.use=true wandb.name=cqn_as_demos_validation_$(date +%s)
  ```
  Expect: non-trivial `episode_reward > 0` on some episodes, episode lengths staying long (not collapsing back to 31-step evacuation), and `safety/ssm_violation` rate trending down or stable.

Then **proceed straight to P3.1** (the Lagrangian glue). **We are NOT re-running E1.4 (C2)** as a standalone gate — the off/oracle/noisy obs-channel ablation folds into Phase 3 eval (E3.6) (user decision 2026-05-20; see decision log). The snapshot/video cadence is fixed, so any new CQN-AS run saves `snapshot_<step>.pt` every 10k + a final-state save, and `eval_videos/step_*.mp4` per eval cycle.

### What to do when the re-run C2 cells finish

1. **Harvest snapshot paths** from the 3 hydra run dirs:
   ```bash
   ls -la exp_local/cqn_as_safety/saucepan_to_hob_*/snapshot_*.pt
   ```
   Snapshots now land at every `snapshot_every_frames` (default 10000) **plus a final-state save at 200000** (the fix). Pick the peak-by-eval-curve from W&B per `bodyslam` mode and paste into `SNAPSHOTS` at the top of [`scripts/phase1_reward_pilot_cqn_as.py`](../scripts/phase1_reward_pilot_cqn_as.py:73-77). Eval-rollout mp4s are in each run's `eval_videos/` (and W&B under `eval/video`) for visual sanity-checking the channel-vs-no-channel behavior.

2. **C2.4 — fire the eval grid** (~1h, 9 cells = 3 modes × 3 seeds × 20 episodes on `disruption=coworker_eval`):
   ```bash
   python scripts/phase1_reward_pilot_cqn_as.py --eval > /tmp/c2_eval_commands.sh
   # Then fire each printed command. They use +snapshot_path=... which
   # train_cqn_as.py honours (Workspace.load_snapshot, eval-only mode
   # when num_train_frames=0).
   ```

3. **C3 — decision logic** (record in this file's Decision Log + Workstream C section):
   - If `bodyslam=noisy` cell beats `off` by ≥20% SSM reduction → `bodyslam=noisy` for Phase 3 actor.
   - If neither beats `off` → `bodyslam=off` for actor (the filter consumes the channel only).
   - If `oracle` helps but `noisy` doesn't → `bodyslam=oracle` in training, and investigate the noise model.

### Then Phase 3.1 unblocks (the Lagrangian glue)

P3.0 shipped all the scaffolding. **P3.1 is the next coding milestone** and is gated only on the C3 obs-config decision:
- λ PID updater on rolling-mean cost (start `K_I=1e-3, K_P=1e-2, K_D=0, λ_max=100, d=0.01` per the plan)
- dual-Q action selection `argmax_a [Q_r − λ·Q_c]` over CQN-AS coarse-to-fine bins
- `Q_c` training-loop integration in `train_cqn_as.py` — regress `CostCritic` on the per-step `cost` already in the batch dict (P3.0c wired it; `agent.update()` just ignores `batch["cost"]`/`batch["max_cost"]` today)
- decide warm-start: A/B `CostCritic.warm_start_from_svf(..., force_sign_flip=True)` vs fresh init (deferred from P3.0b)

Continuous `ssm_margin` is still computed by `ISO15066Wrapper` and present in `info["safety"]`; the bounded per-step cost `c_t = max(c_ssm, c_pfl)` lives in [`filters/cost_signal.py`](../safety_bigym/filters/cost_signal.py). See [.claude/UPDATED_PROJECT_PLAN.md](UPDATED_PROJECT_PLAN.md) Phase 3 (Option B-value-mean).

### Closed in 2026-05-18 → 2026-05-20 work

- **A6.1-A6.4** smoke gates green; **B5** SVF v1 critic trained/eval'd/swept → `checkpoints/svf_coworker_train_v1.pt`, operating point R≈4.0 (see 2026-05-18 notes)
- **A8** PR merged to main (CQN-AS vendor)
- **P3.0 (Phase 3 scaffolding)** merged via PR #9 — workspace reward shaping, `CostCritic` module, per-step cost pipeline, smoke verification. See Workstream P3.0 below.
- **Snapshot-cadence bug** in `train_cqn_as.py` found + fixed (`6e7fdc1`) — C2's first run wasted (zero snapshots saved). Regression test `tests/test_cqn_as_snapshot_cadence.py`.
- **Eval video recording** added (`09faaa4`) — `save_video=true` dumps `eval_videos/step_*.mp4` + W&B `eval/video`. Module `safety_bigym/agents/cqn_as/eval_video.py`, tests `tests/test_cqn_as_eval_video.py`.

Earlier ready-to-pick-up tracks (now closed): Workstream A6 (smoke + A8 PR), Workstream B5 (SVF training).

---

## Workstream A — CQN-AS Vendor + Smoke Gate

- [x] A0. Branch `safety-critic/phase-3-cqn-as-vendor` created
- [x] A0. Status file created (this file)
- [x] A1. Standalone CQN-AS smoke on stock `dishwasher_close` — `num_train_frames=20k` — passed on GPU box 2026-05-15. Headless-rendering gotcha: requires `MUJOCO_GL=egl` + `MUJOCO_EGL_DEVICE_ID=0` (no X server on the GPU box). Unblocks A3/A4/A5.
- [x] A2. Vendored modules under `safety_bigym/safety_bigym/agents/cqn_as/`
  - [x] A2.1 `__init__.py` (+ `agents/__init__.py`)
  - [x] A2.2 `cqn_utils.py` (verbatim from CQN-AS `cqn_utils.py`) + `utils.py` (verbatim from CQN-AS `utils.py`)
  - [x] A2.3 `replay_buffer.py` (verbatim from `bigym_src/replay_buffer_action_sequence.py`)
  - [x] A2.4 `agent.py` (from `bigym_src/cqn_as.py`, 842 LOC — kept as one file; imports patched to relative). Split into `networks.py` deferred unless adapter work warrants it.
  - [x] A2.5 `cfgs/agent/cqn_as.yaml` (restructured) + `cfgs/cqn_as_config.yaml` root (A5)
  - [x] All four modules AST-parse; `cqn_utils` and `replay_buffer` import cleanly without third-party deps
- [x] A3. Env adapter wraps `SafetyBiGymEnv` cleanly (`safety_bigym/agents/cqn_as/env_adapter.py`, ~470 LOC, parses)
- [x] A4. Encoder accepts `human_pos_estimate` (gated on `bodyslam.mode`) — adapter injects 6D vector into `low_dim_obs`; `C2FCriticNetwork` already takes `low_dim: int` so no agent.py changes needed; training entrypoint sizes the critic from `train_env.low_dim_raw_observation_spec().shape[0]`
- [x] A5. Training entrypoint `train_cqn_as.py` + Hydra config — handles `num_demos=0` (smoke gate) by skipping the demo replay buffer; safety info logged per-step + at episode end
- [x] A6.1 Smoke gate: encoder accepts `human_pos_estimate` — green on `cqn_as_smoke_dishwasher_oracle_v4` (2026-05-18). `adv_low_dim_encoder` is 288→64 with `bodyslam=oracle` vs 264→64 with `bodyslam=off`; the 24-unit diff = 6D `human_pos_estimate` × frame_stack=4. Encoder sizes correctly on both ends; no shape errors over 2000 frames.
- [x] A6.2 Smoke gate: COWORKER scenario doesn't crash on episode boundary — green. v4 ran 38 episodes (first one full 350-step, subsequent ~36-step truncations on random policy). Per-episode trajectory mode varies across rollouts.
- [x] A6.3 Smoke gate: per-step cost logging inside K-step chunk — green. `safety/ssm_margin`, `safety/pfl_force_ratio`, `safety/ssm_violation`, `safety/pfl_violation` log at global_step % 50 == 0; episode-end `episode_safety/*` aggregate also fires. Confirms info["safety"] surfaces per env-step, not per K-step chunk.
- [x] A6.4 Smoke gate: action space tractable — green. Pre-update env stepping ~117 steps/s; post-update (episode 3+) ~6.4 steps/s = 150ms/step including agent update. Per env-control cycle (K=16) the budget was 640ms; we run well under.
- [x] A7. Adapter test file green — [`tests/test_cqn_as_adapter.py`](../tests/test_cqn_as_adapter.py), 24/24 green locally; sibling tests (`test_coworker_disruption`, `test_svf_dataset`, `test_safety_labeling`, `test_bodyslam_wrapper`) still 46 passed / 1 skipped. Tests are pure-Python — `SafetyBiGymEnvFactory._create_env` is monkeypatched to a stub gym env so no MuJoCo/AMASS required; covers low_dim shape (with/without bodyslam), action [-1,1] roundtrip incl. gripper-tail handling, TimeStep first/mid/last typing, episode_length truncation, info["safety"] + info["episode_safety"] forwarding, frame-stack widening, pixels=False zero placeholder, ExtendedTimeStepWrapper action injection.
- [ ] A8. PR off `safety-critic/phase-3-cqn-as-vendor` to `main`

## Workstream B — Phase 2 Dataset Regen

- [x] B1.1 ACT re-roll on COWORKER train: `dishwasher_close` (GPU box; user-launched, complete)
- [x] B1.2 ACT re-roll on COWORKER train: `drawers_open_all` (GPU box; user-launched, complete)
- [x] B1.3 ACT re-roll on COWORKER train: `saucepan_to_hob` (GPU box; user-launched, complete)
- [x] B1.4 `filters/snapshots.py` SNAPSHOTS dict updated with new W&B peak-by-eval-success paths
  - `dishwasher_close` → `dishwasher_close_20260515184635/snapshots/50000_snapshot.pt`
  - `drawers_open_all` → `drawers_open_all_20260515184721/snapshots/40000_snapshot.pt`
  - `saucepan_to_hob` → `saucepan_to_hob_20260516123308/snapshots/70000_snapshot.pt`

Task selection (2026-05-15, user): `reach_target_single` excluded from training/eval tasks — horizon too short. The three working long-horizon tasks (`dishwasher_close`, `drawers_open_all`, `saucepan_to_hob`) replace it everywhere downstream.
- [x] B2. `svf_collect_dataset.py` rewired
  - [x] B2.1 `_build_live_env` dispatches on disruption cell label — `coworker_train` / `coworker_eval` → COWORKER factories, anything else → legacy `DisruptionType[...]` path (back-compat preserved)
  - [x] B2.2 `--disruption-space {coworker_train,coworker_eval,legacy_multi}` flag added; default `coworker_train`; `CollectionPlan.smoke()` updated to use it
  - [x] B2.2a Docstring + `--help` updated; existing svf collect/eval/sweep smoke tests unaffected (they construct CollectionPlan directly with legacy disruption strings, which dispatch to the legacy branch)
  - [x] B2.3 10k-transition smoke confirms violation rate ≥5% (GPU box) — passed 2026-05-16 at `--proximity-threshold 0.50`: random 11.0%, snapshot 15.5% (datasets/svf_b23_smoke_v5)
  - [x] B2.4 4-dof floating base aligned with RoboBase training: `_build_live_env` now passes `floating_dofs=[X, Y, Z, RZ]` to `JointPositionActionMode`. Bare-BiGym default (3 dofs: X, Y, RZ) gave `action_dim=15` and silent state_dict shape mismatch on Phase 0 ACT snapshots which were trained under RoboBase's `cfg.env.enable_all_floating_dof=True` path (`action_dim=16`, `qpos=66`).
  - [x] B2.5 Snapshot agent now instantiated against a synthesized observation_space that mirrors RoboBase's `ConcatDim(shape_length=1)` + `FrameStack(frame_stack=1)` output (`low_dim_state: (1, D_concat)`, `rgb_<cam>: (1, 3, H, W)`). Bare-BiGym rgb shape `(3, 84, 84)` failed the encoder's 4-D assertion at `bc.py:155`.
  - [x] B2.6 Env wrapping decoupled from snapshot's training-time `bodyslam.mode`: env is now wrapped with `BodySLAMWrapper(plan.bodyslam_mode)` *always*, so the SVF dataset records `human_pos_estimate` regardless of which actor generated the action. `_SnapshotPolicy.adapt_obs` continues to strip the channel before feeding Phase 0 actors via `includes_human_pos=False`. Previous logic (`bodyslam_mode = snap_bs`) would have produced a dataset without the critic's most important input feature.
  - [x] B2.7 `label_transition` switched from ISO 15066 SSM (`ssm_violation`) to geometric proximity (`min_separation < proximity_threshold`, default 0.10m). ISO 15066's industrial-cell calibration demanded ~5m clearance at kitchen-scale robot velocities → every transition labelled unsafe (93% on random-only smoke). `--proximity-threshold` CLI flag surfaced; `ssm_margin` still logged for ISO traceability & Phase 3 Lagrangian. `tests/test_safety_labeling.py` rewritten for the new bar (9 tests green).
  - [x] B2.8 Shard schema extended with `min_separation` and `pfl_force_ratio` per-transition. Enables retroactive relabelling without re-collection: proximity-threshold sweeps work today; PFL retrofit needs re-collection through a PFL-fixed env (current values are all zero) but the schema is forward-compatible. `TransitionShardWriter.write_shard` + `SafetyTransitionDataset.__getitem__` + `rollout_episode` + both demo/snapshot write paths updated; `tests/test_svf_dataset.py` round-trips both fields (8 tests green).
- [x] B3. Dataset collected: 315k transitions × 3 tasks × 2 sources → `datasets/svf_coworker_train_v1/` (2026-05-17)
  - 3 tasks: `dishwasher_close`, `drawers_open_all`, `saucepan_to_hob`
  - 2 sources: `random` + `snapshot` (demo source skipped — see B2 notes / blockers)
  - Settings: `--proximity-threshold 0.50`, `--bodyslam-mode noisy`, `--episodes-per-cell 210`, `--max-steps 250`
  - Per-cell violation rates: random/{dishwasher_close=4.6%, drawers_open_all=17.5%, saucepan_to_hob=14.4%}, snapshot/{dishwasher_close=10.2%, drawers_open_all=11.0%, saucepan_to_hob=9.6%}. Aggregate ~11.2% (~35k violations / 315k transitions).
- [x] B4. Sanity checks pass — 3/4 clean, 1 caveat noted, 1 logging gap tracked separately
  - [x] B4.1 Violation rate ≥5% per source — 5/6 cells clean; `random/dishwasher_close` lands at 4.6%, technically below the 5% soft floor but 2,415 absolute violations on 52,500 transitions is plenty for CQL class-weighted training. Accepted as "passing with margin". Source-diversity signal (random > snapshot per task) held at full scale.
  - [x] B4.2 Action-magnitude distribution sane — passes with caveat. Snapshot's per-dim std is markedly tighter than random's (0.15-0.33 vs 0.9-1.7 on body joints), confirming snapshot is task-driven not chaotic. Caveat: snapshot's action range looks like *raw tanh-space outputs* without `RescaleFromTanhWithMinMax` denormalization (gripper dims emit values down to -1.1 which the env then silently clips; body-joint dims sit in `[-1, 1]` rather than the env's full ±π range). Snapshot is exploring a narrower action subspace than a deployed policy would; v1 still informative but the gap is real. Tracked as a follow-up — see notes.
  - [x] B4.3 Trajectory-mode coverage — pass. 200-reset sample per task draws APPROACH_LOITER_DEPART/COWORKER_PATROL/STATIONARY at 30-36% each (within ±4pt of uniform 1/3).
  - [x] B4.4 Per-axis coverage — 3/5 axes verified pass (closest_approach, near_loiter, walk_speed all span their full target ranges with sensible distributions); `reach_period` and `target_mix_p_ee` are sampled by `make_coworker_train_space` but **not stored on `ScenarioParams`** so they can't be audited post-reset. Logging gap, not a sampling gap. Tracked as a follow-up — see notes.
- [x] B5. SVF training on v1 dataset — proximity-trained safety critic landed; checkpoint at `checkpoints/svf_coworker_train_v1.pt` (627k). Headline Phase 2 deliverable.
  - [x] B5.1 **Smoke** — green 2026-05-18. Bellman MSE 630.07 → 445.14 (29.4% reduction) over 100 grad steps. Smoke output was initially silent because `logging.basicConfig` was a no-op against a pre-configured root handler; `force=True` fix landed.
  - [x] B5.2 **Full train** — green 2026-05-18, ~1h35m wall-clock. 200k steps, batch_size=512, cql_alpha=5.0, target_violation_rate=0.30. Final step: loss=1.77, bellman=6.20, cql=-0.89, q_mean=3.19. Bellman MSE plateaued ~6.2 (not the ideal <1 fit; suggests the critic is underfit relative to the OOD eval distribution, see B5.3 notes).
  - [x] B5.3 **Eval** — green on both eval distributions:
    - **OOD (`coworker_eval`)** at R=0.5: intervention=0, residual≈0.955 — critic's Q is uniformly ≥0.5 even on the 95%-violating random rollouts. At R=4.0 (from B5.4 sweep): intervention≈99%, residual 3–8%. Hard-safety-filter operating point. Results: `results/svf_eval_v1.csv`.
    - **In-dist (`coworker_train`)** at R=3.5: intervention 28–34%, residual 74–87% (`results/svf_eval_v1_indist.csv`, 2026-05-18). This is the calibrated sensible operating point — the gate fires partially when it sees risk. The reason residual stays high is *structural to the proximity label*: `min_separation < 0.50m` is a state property; once the COWORKER human walks within 0.9-1.4m of the robot (APPROACH_LOITER_DEPART), every subsequent timestep registers as a violation regardless of the robot's action. The fallback policy stops the **robot** from contributing to proximity but can't stop the **human** approach. ~15pp residual drop from random baseline at 30% intervention rate confirms the critic reduces robot-driven proximity events.
  - [x] B5.4 **Threshold sweep** — green 2026-05-18 (v2 results in `results/svf_sweep_{task}_v2.csv`). Q distribution centred at ~3.4 with a narrow safe/unsafe gap; cliff between R=3 (10–22% intervention, residual ≈ 0.93) and R=4 (97–99% intervention, residual 3–8%). **Operating point R≈4.0** trades aggressive intervention for a 30× drop in residual violations vs random. Functional as a hard safety gate; tight for an actor-coupled filter.
  - [x] B5.5 **DONE — negative (2026-05-20).** Ran the full v2 pipeline ([`scripts/run_phase2_b55.sh`](../scripts/run_phase2_b55.sh)): snapshot action-denormalization patch landed (in `bed92f7`; tests `tests/test_svf_collect_snapshot_denorm.py`), v2 collected + trained (`checkpoints/svf_coworker_train_v2.pt`, train `q_mean≈2.96`), eval + sweep complete (`results/svf_{eval,sweep}_v2_*.csv`). **Result:** the patch is a correct fix but is NOT the residual lever. At a partial operating point (~30% intervention) v2 residual is ~87% — unchanged from v1's 74–87%. The intervention/residual tradeoff is ~linear up to the ~90% cliff → residual is dominated by the **structural proximity floor** (human-approach-driven; the robot can't prevent it), not snapshot action-subspace narrowness. Hard-gate residual *did* improve (<1% @ R≈3.5 vs v1's 3–8% @ R=4.0). **Next: change the label, not the data** — tighter-τ relabel and/or robot-controllability-aware label, both offline from the v2 shards. Full analysis: [phase2_results.md §7 + §8](phase2_results.md#results-2026-05-20--patch-fired-hypothesis-not-confirmed).

## Workstream C — E1.4 CQN-AS Observation Ablation

**Anchor task: `saucepan_to_hob`** (user decision 2026-05-15). Chosen because the legacy E1.1 side-finding showed oracle improved task success 0.22 → 0.58 while *worsening* SSM violations — the most informative cell to probe whether an RL reward signal redirects the policy to use the human-state channel for safety rather than progress.

- [x] C1. Sweep script [`scripts/phase1_reward_pilot_cqn_as.py`](../scripts/phase1_reward_pilot_cqn_as.py) — modelled on `phase1_reward_pilot.py` but invokes `train_cqn_as.py`. 3 train cells (`bodyslam=off|oracle|noisy`) × `saucepan_to_hob` × `disruption=coworker_train`, `env.safety.add_violation_penalty=true env.safety.violation_penalty=0.05`, `num_train_frames=200000`, `num_demos=0`. `--smoke` runs a 2000-frame validation on a single cell. `--eval` prints eval commands against `disruption=coworker_eval` (20 episodes × 3 seeds × 3 modes) once SNAPSHOTS at the top of the script are filled in post-train. Blocks on A6 green (smoke gate validates the same composition path).
- C2 unblocked 2026-05-18 by replay-buffer collate fix + CQNASAgent state_dict + train_cqn_as `+snapshot_path` eager-load. Eval half also unblocked (snapshot loader now functional).
- ⚠️ **C2's first run (2026-05-18) is DEAD — must be re-run.** All 3 cells trained to ~190k+ but saved **zero snapshots** due to the snapshot-cadence bug (`6e7fdc1`, see notes 2026-05-19). The run dirs have `buffer/` + `train_cqn_as.log` only, no `snapshot_*.pt`. Kill + restart with current `main` (see "Next session — start here"). The restart also picks up `save_video=true` (eval mp4s).
- [ ] C2.1 Cell `bodyslam=off` × saucepan_to_hob launched + complete *(re-run pending)*
- [ ] C2.2 Cell `bodyslam=oracle` × saucepan_to_hob launched + complete *(re-run pending)*
- [ ] C2.3 Cell `bodyslam=noisy` × saucepan_to_hob launched + complete *(re-run pending)*
- [ ] C2.4 Eval on COWORKER eval space (20 episodes × 3 seeds × 3 cells)
- [ ] C3. Decision recorded → Phase 3 obs config locked
  - [ ] If channel helps: `bodyslam=noisy` for Phase 3 actor
  - [ ] If channel doesn't help: `bodyslam=off` for actor (filter consumes channel only)
  - [ ] If oracle helps, noisy doesn't: `bodyslam=oracle` + noise-model investigation

## Workstream P3.0 — Phase 3 Scaffolding (DONE, merged PR #9)

The minimum scaffolding for the Phase 3 Lagrangian (Option B-value-mean) to launch. C3-gated only on the actor's obs config; the scaffolding itself is bodyslam-agnostic. **Does not** include the λ updater, dual-Q action selection, or any training run — those are P3.1+. Smoke passes in ~10s locally (`scripts/phase3_p30_smoke.py`).

- [x] P3.0a — **Workspace reward shaping.** `SafetyConfig.add_workspace_penalty` + `workspace_radius` (0.4) + `workspace_beta` (0.2); `SafetyBiGymEnv._compute_workspace_penalty()` subtracts `β·max(0, ‖p_ee − p_task‖ − r_ws)` from task reward in `_reward()`. Threaded through `safety_bigym_factory._create_env`. Tests: `tests/test_workspace_shaping.py` (10).
- [x] P3.0b — **`CostCritic` module** ([`filters/cost_critic.py`](../safety_bigym/filters/cost_critic.py)). Architectural twin of the Phase 2 `SafetyCritic` (same MLP, input spec, checkpoint format) but regresses on continuous cost `c_t` (high Q = dangerous). `warm_start_from_svf()` **refuses without `force_sign_flip=True`** — SVF and Q_c heads point opposite directions. Tests: `tests/test_cost_critic.py` (12). Warm-start vs fresh-init A/B deferred to P3.1.
- [x] P3.0c — **Per-step cost pipeline.** [`filters/cost_signal.py`](../safety_bigym/filters/cost_signal.py) `compute_cost()` → `c_t = min(1, max(c_ssm, c_pfl))`. env_adapter attaches `c_t` per env-step to TimeStep; replay_buffer accumulates n-step discounted `cost` + per-step `max_cost`; `to_torch_pixel_tensor_dict` exposes both in the batch dict (8/10-tuple back-compat); `train_cqn_as` data_specs carries `cost`. **`agent.update()` ignores `batch["cost"]`/`["max_cost"]` today — P3.1 wires Q_c to consume them.** Tests: `test_cost_signal.py` (16), `test_replay_cost_field.py` (4), `test_cqn_as_adapter.py` (+6), `test_cqn_as_utils_cost.py` (3).
- [x] P3.0d — **Smoke verification** [`scripts/phase3_p30_smoke.py`](../scripts/phase3_p30_smoke.py). 500 env-steps on dishwasher_close confirm workspace penalty fires (500/500 steps) + per-step `c_t > 0` (≈450/500) + warm-start guard fires. `--dry-run` skips MuJoCo (~1s). Surfaced two real bugs (see notes 2026-05-19): H1 EE lookup, factory not propagating workspace cfg.

## Workstream P3.1 — Lagrangian glue (CODE COMPLETE, unit-tested 2026-05-20; GPU smoke pending)

Branch: `safety-critic/phase-3-constrained-rl`. The coding milestone that consumes `batch["cost"]` (P3.0c wired it; vendored `agent.update()` ignored it). Option B-value-mean to start. **Not** the scientific result — that's the E3.* GPU sweeps, gated on this + D3b-validation.

- [x] P3.1a — **`LagrangianCQNASAgent`** in a sibling module ([`agents/cqn_as/lagrangian_agent.py`](../safety_bigym/agents/cqn_as/lagrangian_agent.py)), subclasses the vendored `CQNASAgent` (no edit to `agent.py`). Adds a second **C51 C2F cost critic `Q_c`** (verbatim `C2FCritic` clone, cost-range support `[0, 10]`) + its own target net + **its own `MultiViewCNNEncoder`** + optimizers. Cost backup is **per-env-step** Bellman regression on `batch["cost"]` (not `max_cost`, not per-chunk). `Q_c` target's next-action is the **dual policy** `a' = argmax_a[Q_r − λ·Q_c]` from the target nets; the reward critic keeps its vendored greedy backup.
- [x] P3.1b — **λ PID + dual-Q selection** in a dependency-light module ([`agents/cqn_as/lagrangian.py`](../safety_bigym/agents/cqn_as/lagrangian.py), torch-only so it unit-tests without `tensordict`). `LagrangianPID` drives λ on rolling-mean cost (`K_I=1e-3, K_P=1e-2, K_D=0, λ_max=100, d=0.01`); `dual_select` is the cost-aware argmax injected at every coarse-to-fine level. **λ enters ONLY at action selection, never in a critic's regression target.** `act()` + the `Q_c` backup both select via the dual rule. `update()` returns a `TensorDict` with `q_c_loss, lambda, rolling_cost, cost_violation, batch_cost`.
- [x] P3.1c — **Hydra-selectable** via `agent=cqn_as_lagrangian` ([`cfgs/agent/cqn_as_lagrangian.yaml`](../cfgs/agent/cqn_as_lagrangian.yaml), inherits `cqn_as.yaml` + adds cost/λ knobs). Plain `agent=cqn_as` (default) untouched. Config composition verified.
- [x] P3.1d — **Tests** (`test_lagrangian_pid.py` 6, `test_dual_q_selection.py` 4 — both run locally; `test_lagrangian_agent.py` 8 — `importorskip("tensordict")`, runs on GPU box). λ monotonicity + clamps; λ=0 ⇒ reward argmax / large-λ ⇒ low-cost bin; `Q_c` loss finite + decreases on fixed batch; cost-critic params change (cost consumed); base CQN-AS still constructs sans `Q_c`; target soft-update; state_dict roundtrip. 10 local pass.
- [ ] P3.1e — **2000-frame GPU-box smoke** (needs `tensordict`): `agent=cqn_as_lagrangian env=safety_bigym/saucepan_to_hob disruption=coworker_train bodyslam=oracle num_demos=10 num_train_frames=2000 env.safety.add_workspace_penalty=true wandb.use=false`. Expect λ moves off init, `q_c_loss` logged, no shape errors. **Gated on D3b-validation** (don't trust constrained-RL training until the demo+workspace base is non-degenerate).

**Deferred (not P3.1):** SVF warm-start of `Q_c` (the MLP `filters/cost_critic.py` path / option B) — chosen the clean C51-clone instead; warm-start lands with a B-value-CVaR variant. All E3.* sweeps (cost form, budget Pareto, arch A/B-mean/B-CVaR, β, WCSAC baseline).

## Workstream D — CQN-AS demo pipeline (NEW, 2026-05-20)

**Trigger:** C2 (E1.4) ran `num_demos=0` and degenerated — robot fled the human, 31-step episodes, off/oracle/noisy curves identical. CQN-AS is demo-driven; demo-less on long-horizon manipulation never discovers the task. Demos are the bigger of the two confounds (the other was no workspace shaping). **Decision (user, 2026-05-20): build the demo pipeline.**

Full scope + acceptance criteria + first investigation step in [PHASE3_DEMO_PIPELINE_HANDOFF.md](PHASE3_DEMO_PIPELINE_HANDOFF.md). One-session task.

- [x] D0. **Dof question RESOLVED — it was a red herring (2026-05-20).** Cache dir `~/.bigym/demonstrations/0.9.0/SaucepanToHob/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute` is **4-dof**. `SafetyBiGymEnvFactory()._get_demo_fn(cfg, num_demos=5)` returned 5 `Demo` objects (~427 steps each), `info["demo_action"]` shape **(16,)** (4-dof: 14 body + 2 grippers), obs keys proprioception(60)/grippers(2)/floating_base(4) + rgb cameras, per-step reward (demo0 sum=35.0, max=1.0). **Port implemented; no re-recording.**
- [x] D1. `SafetyBiGymCQNAdapter.get_demos()` implemented (2026-05-20) — reuses `_get_demo_fn` raw-env + DemoStore path, ports `convert_demo_to_timesteps`/`extract_action_stats`/`rescale_demo_actions` from `CQN-AS/bigym_src/bigym_env.py`, injects `human_pos_estimate` via `BodySLAMWrapper(demo_replay=True)` + `AMASSDemoPositionProvider` over an in-memory `_DemoReplayEnv`, carries the P3.0c `cost=0.0` field (demos have no live human → safe-side placeholder).
- [x] D2. Demo-derived `action_stats` override `self._action_stats` (shared with the live path; also addresses the B4.2 snapshot-denormalization caveat).
- [x] D3a. pytest `tests/test_cqn_as_demos.py` (stubbed DemoStore + stubbed AMASSDemoPositionProvider, no MuJoCo) — green, 11 tests; full demo+adapter suite 40 passed.
- [x] D3b-smoke. **2000-frame GPU-box smoke PASSED (2026-05-20).** `num_demos=10 num_train_frames=2000 bodyslam=oracle`: loaded 10 demos via raw-env+DemoStore (4-dof), converted 6/10 successful, demo replay filled (3663 transitions after demo fill), train loop clean — no worker-striping IndexError, no collate crash, `snapshot_2000.pt` saved at exit. **Smoking-gun signal: episode 1 ran the full 1000-step budget** (vs C2's 31-step evacuation collapse) — demos unblock non-degenerate learning. Per-step `safety/ssm_margin` swings between approach (negative) and retreat (positive) as expected; ep1 violation rate decayed to ~5% by end. Run dir: `exp_local/cqn_as_safety/saucepan_to_hob_20260520124507`.
- [x] D3b-validation **RAN 2026-05-20 → FAILED (degenerate, but a different mode).** 50k frames, `num_demos=10`, `add_workspace_penalty=true` (β=0.2). Episode length stayed ~1000 (the 31-step evacuation is gone — demos+shaping fixed that), but episode_reward fell monotonically −78 → −775 and the robot learned to park away from the task / retreat from the human, getting worse over training. **Root cause: the dense workspace penalty's discounted return blows past the CQN-AS C51 critic support [−2,+2], so the Bellman-target clamp saturates value learning** (the gradient that should pull the EE back is clipped). Secondary: demos carry no human (cost=0) vs a live coworker → distribution mismatch. **Full writeup: [phase3_base_validation_findings.md](phase3_base_validation_findings.md).** Superseded by Workstream BASE-FIX below.

## Workstream BASE-FIX — reward/critic-support fix + re-validation (2026-05-20)

Pre-P3.1 gate: D3b-validation failed (above). The fix de-saturates the critic so the shaping gradient survives, plus a human curriculum to soften the demo/live mismatch. Code landed on `safety-critic/phase-3-constrained-rl`; the staged re-validation is the next GPU-box run.

- [x] **Bounded workspace penalty.** New `SafetyConfig.workspace_excess_cap` (caps excess distance → per-step penalty bounded at `−β·cap`); threaded through `config.py` / `cfgs/env/safety_bigym.yaml` / `safety_bigym_factory.py` / `safety_env.py::_compute_workspace_penalty`. Defaults changed: `workspace_beta 0.2 → 0.05`, `workspace_excess_cap = 1.0` (set `None` to recover unbounded). Tests: `test_workspace_shaping.py` now 14 (cap saturation / inactive-below-cap / `cap=None` / support invariant). Full suite 321 pass.
- [x] **Gentle curriculum disruption** `cfgs/disruption/coworker_easy.yaml` (farther approach, less frequent reach) for stage 1.
- [x] **Staged launcher** `scripts/run_base_curriculum.sh` — stage 0 `coworker_idle` (human present but ~3 m off / non-interfering) → stage 1 `coworker_easy` → stage 2 `coworker_train`, each resuming the prior snapshot via `+snapshot_path`. Carries the widened support (`agent.v_min=-6 agent.v_max=2 agent.atoms=101`), `num_demos=36`, bounded penalty. `SMOKE=1` runs a ≤2000-frame stage-0 smoke. **Stage 0 keeps the human present (not `inject_human=false`) so obs width / architecture match across stages and snapshots resume cleanly** — Hydra also rejects `disruption=null` (override-group-to-null), and omitting the flag would spawn the interfering `weights` mix.
- [x] **Re-validation — GATE PASSED (2026-05-21).** Stages 0 (`coworker_idle`) + 1 (`coworker_easy`) completed: the robot **completes the task**, `episode_reward ≈ 1.8` (positive, no critic saturation) — base policy is non-degenerate. Stage 2 (`coworker_train`) was promising but died on a machine crash (not code); resumable via `RESUME_STAGE2=1 RESUME_DIR=<run> scripts/run_base_curriculum.sh`. Stage 2's output = the unconstrained baseline (on the real distribution) + the P3.1 warm-start, so finish it.
- [ ] **Finish stage 2, then un-park P3.1.** After stage 2: run the `agent=cqn_as_lagrangian` 2000-frame smoke on the stage-2 snapshot, then the constrained-RL runs (E3.*). NOTE: `train_cqn_as` `+snapshot_path` restores weights only (step counter + replay restart), so resumes train a fresh budget from the resumed weights.
- Design invariant to preserve on any β/cap change: `β·workspace_excess_cap/(1−γ) ≤ |agent.v_min|` (here `0.05·1.0/0.01 = 5 ≤ 6`).
- [x] **C51 projection offset bugfix (2026-05-21)** — bring-up of the stage-0 smoke surfaced an opaque CUDA device-side assert in the vendored `compute_target_q_dist` `index_add_`. **Root cause (after two wrong hypotheses — `b`-overshoot and NaN, both disproven):** the per-row scatter `offset` used `torch.linspace(..., dtype=int64)`; on CUDA integer `linspace` is computed in float32, which can't represent integers past `2**24`, and with `batch_size·atoms = B·L·D·atoms = 512·3·256·101 ≈ 39.7M` the offsets round and the boundary row addresses past `m.numel()`. Data-dependent (boundary `lower`/`upper`), so it tripped with 29/36 successful demos but not 6/10; exact on CPU so it never reproduced locally. Fixed with exact int64 `arange(batch_size)*atoms` (sanctioned edit to `agent.py`, logged in its header) — also fixes the P3.1 cost critic. Kept defensive clamps (`b`, `lower`/`upper`) + a `train_cqn_as` finite-batch guard (ruled out NaN). Confirmed via `CUDA_LAUNCH_BLOCKING=1` + an in-projection diagnostic. Tests: `tests/test_c51_projection_bounds.py` (offset arange-exactness runs locally; boundary tests tensordict-gated). **2000-frame stage-0 smoke PASSED on the GPU box (2026-05-21)** — trained past the old step-871 crash, ep3 ran the full 1000 steps, `snapshot_2000.pt` saved, no assert. The full staged curriculum is the next GPU run.

**Once the BASE-FIX re-validation confirms non-degenerate learning:** proceed straight to Phase 3 (P3.1) with `bodyslam=oracle` (or noisy) for the actor and fold the off/oracle/noisy channel ablation into Phase 3 eval (E3.6). **We are NOT re-running E1.4 (C2) as a standalone gate** (user decision 2026-05-20 — see decision log).

## Workstream G — G1 coworker swap (2026-05-27, on `retryg1`)

Replace the SMPL-H humanoid with a Unitree G1 acting as the COWORKER. Fresh implementation; the previous `safety-critic/g1-coworker` attempt is explicitly NOT a reference (user decision: a clean retry — the prior version required `MASK_PIXELS=1` to train, capping task-success ceiling). Full design + smoke + verification protocol in [g1_coworker_swap.md](g1_coworker_swap.md).

- [x] **G1 asset** — pulled upstream Unitree G1 (`safety_bigym/assets/g1/g1.xml` + 51 STLs) via path-scoped checkout from the prior branch (asset content is upstream menagerie; not "previous attempt code"). Generated wrapper `safety_bigym/assets/g1_human_body.xml` via new `scripts/build_g1_human_body.py` (idempotent; rerun on upstream refresh). Mesh paths in the checked-in XML are **relative** (`g1/assets/<file>.STL`); `_create_merged_world` absolutises them at load time so the asset is portable across machines.
- [x] **Visual strategy decision.** Initial cut was strategy α (skin-toned collision-proxy capsules, all visual meshes stripped) — minimises the visual delta the CNN encoder sees vs SMPL-H. Per supervisor feedback ("SMPLH looks poor"), switched to **rendering the upstream G1 STL meshes** — looks like a real G1 robot. Risk: re-introduces the visual delta that produced the prior attempt's encoder regression. Tripwire if it bites the next curriculum: revert to strategy α (one commit back on this branch) or fall back to MASK_PIXELS=1.
- [x] **Spec module** `safety_bigym/safety_bigym/human/g1_human_spec.py` — single source of truth for joint names (29 hinge joints), standing pose, SSM body list (14), arm chains. Imported by all G1-aware modules so a model change is a one-file edit.
- [x] **Parallel controllers, SMPL-H code untouched.** `G1HumanController` + `G1HumanIK` mirror the public surface of the SMPL-H classes; env dispatches on `HumanConfig.human_model`. Selected via Hydra (`env.human_model=g1`, default `smplh`) or env-var in `scripts/run_base_curriculum.sh` (`HUMAN_MODEL=g1`).
- [x] **Env-wrapper plumbing** — `safety_env.py` `_human_body_path` / `_ssm_body_names` / `_init_human_controller` / COWORKER IK-solver dispatch all branch on `human_model`. `_is_robot_geom` unchanged (G1 `_col` geoms are excluded by suffix). `_create_merged_world` now copies the human XML's `<asset>` block (was a silent gap — only mattered once G1 meshes were re-introduced).
- [x] **PFL region map** — `safety/pfl_limits.py::GEOM_TO_REGION` extended with 8 G1-specific entries (`L/R_Thigh_col`, `L/R_Shin_col`, `L/R_Foot_col`, `L/R_Hand_col`); overlapping anatomical names (`Pelvis_col`, `Head_col`, `Chest_col`, `L/R_Shoulder_col`, `L/R_Elbow_col`, `L/R_Wrist_col`, `Spine_col`) reuse the SMPL-H entries unchanged.
- [x] **HumanController no-clip fallback bug** uncovered along the way and fixed: when `clip is None` the SMPL-H controller ignored the trajectory planner and parked the pelvis at spawn. Now respects the planner XY/yaw same as `G1HumanController`. No production impact (AMASS always loaded in training) but fixes smokes / contingencies.
- [x] **CoworkerArmController generalised.** Hard-coded `"R_Shoulder"` / `"L_Shoulder"` literals replaced with `ik_solver.chains[arm]["shoulder_body"]`. Added `shoulder_body` to `HumanIK.chains` for SMPL-H parity. `CoworkerArmController(..., ik_solver=None)` accepts an injected solver; env passes `G1HumanIK(...)` for G1, `None` (defaults to SMPL-H `HumanIK`) otherwise.
- [x] **Tests** — `tests/test_g1_asset.py` (10), `tests/test_g1_human_controller.py` (5), `tests/test_g1_safety_tracking.py` (3), and `tests/test_collision_groups.py` parametrized over `[smplh, g1]`. Full suite **351 passed, 37 skipped (AMASS-dependent)**, no SMPL-H regressions.
- [x] **End-to-end smoke** `scripts/g1_coworker_smoke.py` (extended with `--human {g1,smplh}` and `--stage {idle,easy,train,default}` for direct curriculum-band verification). 60-second smoke at 20 Hz on stage `train` confirms both human models cycle the arm reach (`extend → hold → retract → idle`), achieve sub-2 cm min separation, fire `proximity_violation` 86-88 % of steps.
- [ ] **G1 base curriculum run** (next GPU-box action — see "Next session — start here"). Acceptance: stage-0 best `ep_reward` within ~10 % of SMPL-H anchor (`-7.2`); stages 1/2 reach `success_rate ≥ 0.8`.

## Workstream M — Three-flavour safety metrics (2026-05-27, on `retryg1`)

[docs/safety_metrics.md](safety_metrics.md) was written 2026-05-26 as the thesis-grade schema spec; the live code only emitted single-flavour SSM until today. This workstream made the spec real.

- [x] **`SSMConfig.proximity_threshold`** (default 0.5 m, configurable via Hydra) — matches Phase 2 SVF production label bar.
- [x] **`SafetyInfo` extended** — `ssm_violation_actual`, `ssm_margin_actual`, `proximity_violation`, `proximity_threshold`, `robot_vel`, `human_vel` added; `to_dict` carries them.
- [x] **`ISO15066Wrapper._ssm_into`** computes all three flavours per step. Worst-case SSM uses `v_h = v_h_max`; actual SSM uses the env-capped observed velocity; proximity is pure geometric.
- [x] **`EpisodeSafetyMetrics`** rewritten — emits the full thesis-grade `ep_*` schema:
  - rates: `ep_ssm_violation_rate`, `ep_ssm_violation_actual_rate`, `ep_proximity_violation_rate`, `ep_pfl_violation_rate`
  - dwell: `ep_time_in_proximity_{0p3,0p5,1p0}m`
  - separation distribution: `ep_min_separation`, `ep_mean_separation`, `ep_p5_separation`, `ep_p25_separation`
  - margin troughs: `ep_min_ssm_margin`, `ep_min_ssm_margin_actual`
  - robot kinematics: `ep_max_robot_vel`, `ep_mean_robot_vel`
  - existing: `ep_max_pfl_force_ratio`, `ep_max_contact_force`, `ep_time_to_first_violation`, `ep_region_<region>`
  - **Compatibility:** legacy steps that omit the new keys still work (defaults are safe).
- [x] **`train_cqn_as` logging additions.** `_safety_payload` forwards the full new schema. `_lagrangian_payload` returns `episode_cost_integral` (always; accumulator resets at episode boundary) + `episode_lambda` (only when the active agent has `_lambda`, gates P3.1 cleanly). `_setup_wandb` forwards `cfg.wandb.tags` to `wandb.init(tags=...)`. Streaming `metrics.jsonl` (one row per `_log` call) + `final_metrics.json` at end of `train()` (config + `last_train_episode` / `last_episode_safety` / `last_eval` rows + `best_eval` with max-prefer reward/success, min-prefer safety). Eval cycles aggregate `info["episode_safety"]` across `num_eval_episodes` and emit `eval/ep_*` paired with `success_rate`.
- [x] **Curriculum tag emission** — `scripts/run_base_curriculum.sh` now emits `+wandb.tags=[stage<n>,method:<m>,task:<t>,human:<h>]` per stage. Uses `:` (not `=`) as key/value separator inside tag strings because Hydra's override grammar reserves `=` and `,`.
- [x] **Tests** — `tests/test_safety_metrics_three_flavours.py` (8): all three flavours in `to_dict`; doc's three failure-mode regimes (robot-fast/human-distant, robot-still/human-inside, both-still-clear); `proximity_threshold` configurable; every new `ep_*` key emitted; backward-compat with steps missing the new fields.

## Workstream S2 — Stage 2 disruption tighten (2026-05-27, on `retryg1`)

[cfgs/disruption/coworker_train.yaml](../cfgs/disruption/coworker_train.yaml) updated so stage 2 brings the human close enough to put the arm in the robot's workspace and force violations (user request after the previous run's stage-2 metrics showed proximity-violation rate stuck at 0.13-0.24).

| Knob | Was | Now | Why |
|---|---|---|---|
| `closest_approach_range` | 0.9-1.4 | **0.55-0.85** | Shoulder→EE must clear the 0.75 m reach gate; 0.55 m floor keeps mocap-pelvis capsule clear of robot pelvis (~0.25 m clearance — penetration would generate spurious huge contact forces since the mocap pelvis can't be pushed back). |
| `reach_period_range` | 4.5-6.5 | **3.0-5.0** | More reach cycles per episode. |
| `target_mix_p_ee_range` | 0.4-0.6 | **0.55-0.85** | Mostly target the robot EE (vs task object). |
| `near_loiter_range` | 7-11 | **12-18** | Longer dwell at NEAR. |
| `walk_speed_range` | 1.0-1.6 | unchanged | |

Smoke verification at 20 Hz × 60 s (both human models):

| Metric | G1 | SMPL-H |
|---|---|---|
| arm cycle (extend/hold/retract/idle) | ✓ | ✓ |
| trajectory: loiter / approach / depart | 1085 / 65 / 50 | 1085 / 65 / 50 (identical, sampler is body-agnostic) |
| min separation | 0.019 m | 0.016 m |
| `proximity_violation_rate` | 86.67 % | 87.75 % |
| `ssm_violation_actual_rate` | 14.50 % | 54.42 % (higher b/c SMPL-H zero-pose has wider envelope) |

Body knobs flow through the same `ScenarioSampler` → `TrajectoryPlanner` → `CoworkerArmController` for both human models, so the tighter band applies identically.

---

## Carried-forward bugs (not in scope for this plan)

- [ ] PFL contact-detection bug (HIGH; see [CLAUDE.md](../CLAUDE.md))
- [ ] SSM margin outliers (LOW)

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-15 | Vendor CQN-AS into safety_bigym (vs in-place patch) | User decision; cleaner long-term, accepted bigger upfront refactor |
| 2026-05-15 | Scope: P0 + P1 (CQN-AS smoke + dataset regen + E1.4) | User decision; covers 1–2 week horizon |
| 2026-05-15 | Status file at `.claude/IMPLEMENTATION_STATUS.md`, per-task checkboxes | User decision; single living doc next to plan files |
| 2026-05-15 | Task pool: dishwasher_close, drawers_open_all, saucepan_to_hob (reach_target_single excluded) | User decision; reach_target_single horizon too short |
| 2026-05-15 | E1.4 anchor: saucepan_to_hob (single task, not 3) | User decision; aligns with E1.1 side-finding cell where oracle improved success but worsened safety — most diagnostic for "does RL redirect channel usage toward safety?" |
| 2026-05-16 | Phase 2 SVF labelling: geometric proximity, not ISO 15066 SSM | ISO 15066's stopping-distance formula demands ~5m clearance at kitchen-scale robot velocities; first B2.3 smoke at 93% violation rate confirmed the dataset would be degenerate. Switched to `min_separation < proximity_threshold`. `ssm_margin` retained as continuous cost signal for Phase 3 Lagrangian. |
| 2026-05-16 | Phase 2 `proximity_threshold` = 0.50 m for B3 production | Diagnostic d_min quantiles + smoke calibration sweep landed 0.50m at random 11% / snapshot 15.5% (sources distinguishable, dataset workable). 0.10m gave <2% (dead dataset); 0.30m gave 9% identical across sources (kitchen task is human-trajectory-dominated below the policy-effect threshold). 50cm matches the system's effective reaction window. |
| 2026-05-16 | B3 skips `demo` source entirely | BiGym DemoStore cache only has demos recorded with `floating_dofs=['pelvis_x', 'pelvis_y', 'pelvis_rz']` (3 dofs); B2.4 fix moved env construction to 4 dofs (X, Y, Z, RZ) matching RoboBase training. Metadata mismatch → `DemoNotFoundError`. User decision: ship B3 without demos rather than re-record. Loses the safe-side mass demos provide; random + snapshot must cover the safe distribution. Re-recording demos through 4-dof env is a future follow-up if SVF training reveals safe-class undercoverage. |
| 2026-05-16 | Don't block B3 on PFL contact-detection fix | PFL bug is BiGym-internal and unbounded (could be days/weeks). Proximity is a defensible *preventive* safety signal independently — fires before contact rather than at contact. B2.8 schema change stores `min_separation` and `pfl_force_ratio` per-transition so proximity threshold sweeps are free post-collection. Full PFL retrofit (training on `use_pfl=True` labels) still needs a re-collection through a PFL-fixed env because current pfl_force_ratio is identically zero. v1 SVF is a proximity-trained gate; PFL retrofit is Phase 4/5 polish if the fix lands in scope. |
| 2026-05-18 | P3.0 scope = scaffolding only (workspace shaping + CostCritic + per-step cost + smoke), C3-gated | User decision in plan mode. Unblocks the first Phase 3 training run without committing to the λ/dual-Q machinery (P3.1) before C3 locks the obs config. Built bodyslam-agnostic so it stays valid whichever way C3 lands. |
| 2026-05-18 | P3.0b: ship CostCritic with fresh init, not SVF warm-start | SVF trained on `r_safe` (high Q = safe); Q_c regresses on `c_t` (high Q = dangerous). Naive state_dict copy initialises Q_c upside-down. `warm_start_from_svf` exists but requires `force_sign_flip=True` (loads body, reinits head). Whether warm-start beats fresh init is an empirical question deferred to a P3.1 A/B. |
| 2026-05-19 | Commit snapshot-cadence + eval-video fixes directly to `main` (not a phase branch) | User decision; urgent — C2 was blocked on the snapshot bug and a re-run had to start ASAP. Breaks the CLAUDE.md "never commit to main directly" rule by explicit override. |
| 2026-05-19 | Add eval video recording to `train_cqn_as.py` (option b), not post-hoc | User asked for it during the C2-restart cycle so videos land in the same re-run rather than a third pass. Episode-0-only per eval cycle keeps disk + wall-time negligible. |
| 2026-05-20 | Build the CQN-AS demo pipeline (Workstream D) | C2 re-run came back degenerate (robot flees human, 31-step episodes, identical off/oracle/noisy curves). Diagnosed as `num_demos=0` (CQN-AS is demo-driven; never discovers the sparse task reward) + no workspace shaping. User decision: demos are needed. Note: the 2026-05-16 "demos are 3-dof / dead" claim is suspect — `saucepan_to_hob` is 4-dof, ACT loaded 36 demos at 4-dof, cache exists; Workstream D D0 re-verifies before any re-recording. |
| 2026-05-20 | **D0: "demos are 3-dof / dead" is a confirmed red herring** — demos load at 4-dof, port (not re-record) | Verified: cache dir is `..._pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute` (4-dof); `_get_demo_fn(cfg, 5)` returns 5 Demos with `info["demo_action"]` shape (16,). The old `DemoNotFoundError` (B3) came from a different env-construction path, not from demos being 3-dof. Supersedes the 2026-05-16 "B3 skips demo source" rationale's dof claim. |
| 2026-05-20 | **Do NOT re-run E1.4 (C2) as a standalone gate; fold the obs-channel ablation into Phase 3 (E3.6)** | User decision. E1.1 already signalled the observation channel isn't the dominant safety lever (no cell cleared the ≥20% SSM bar; oracle improved task success but not safety); the cost-signal Lagrangian is the thesis lever; the Phase 3 filter/`Q_c` consume `human_pos_estimate` regardless of the actor's obs config. After demos land + a single validation run confirms non-degenerate learning, go straight to P3.1 with `bodyslam=oracle` (or noisy) and ablate off/oracle/noisy inside Phase 3 eval. Saves ~3×200k frames. |
| 2026-05-20 | Commit Workstream D (`get_demos` + tests + docs) directly to `main` | User decision; matches the 2026-05-19 precedent for the snapshot-cadence + eval-video fixes. Breaks the CLAUDE.md "never commit to main directly" rule by explicit override. Unit tests green (40 passed); GPU-box smokes pending. |
| 2026-05-20 | **P3.1 `Q_c` = verbatim C51 `C2FCritic` clone (option A), not the MLP `CostCritic` (option B)** | Dual-Q selection happens inside `C2FCritic.get_action`'s per-level argmax; a second C2F critic scores the same coarse-to-fine bins cleanly (compute `q_r`/`q_c` the same way, `argmax(q_r − λ·q_c)`, share one zoom-in path) with a stationary target. Option B (MLP) would require decoding each candidate bin to a continuous action and running the MLP per-bin-per-level — awkward + slow. Cost is the SVF warm-start: deferred to a future B-value-CVaR variant. User-confirmed before building. |
| 2026-05-20 | **P3.1 `Q_c` Bellman backup evaluates the dual policy `a'=argmax[Q_r−λ·Q_c]` (target nets); reward critic untouched** | `Q_c` should estimate cost of the action the deployed constrained policy actually takes. The vendored reward critic keeps its greedy `argmax Q_r` backup (honors the no-edit-`agent.py` rule). λ stays out of every regression target — it only picks the next action — so both Q-nets keep stationary targets. User-confirmed. |
| 2026-05-20 | **P3.1 `Q_c` gets its own CNN encoder** (not shared with the reward critic) | Cost gradients never corrupt reward features and vice versa; clean decoupled critic. Costs extra memory/compute, accepted. User-confirmed. |
| 2026-05-20 | **D3b-validation FAILED; fix the reward/critic-support incompatibility + re-validate with a human curriculum before P3.1** | The 50k base run was degenerate (reward −78→−775, robot parks away from task). Diagnosed: the dense workspace penalty's discounted return (`−β(d−r_ws)/(1−γ)` ≈ −20·(d−0.4)) saturates the C51 critic support [−2,+2] — the Bellman-target clamp kills the pull-back gradient. User decision: apply 4 levers — bound the penalty (β 0.2→0.05, `workspace_excess_cap=1.0`), widen support (v_min −6 / v_max +2 / atoms 101), demos 10→36, staged human curriculum (`run_base_curriculum.sh`). Writeup: [phase3_base_validation_findings.md](phase3_base_validation_findings.md). Durable lesson now in CLAUDE.md. |
| 2026-05-20 | Open B5.5 (v2 SVF dataset with snapshot tanh-denormalization) | B5.3 in-dist eval confirmed v1 critic narrowness even on `coworker_train` (residual 74–87% at 28–34% intervention). B4.2's snapshot-action denormalization caveat (raw tanh output, env silently clips, body-joint actions stay in [-1, 1]) is the most plausible cause. Pipeline scripted at `scripts/run_phase2_b55.sh`; full plan in [phase2_results.md §B5.5](phase2_results.md#7--b55--v2-dataset-with-snapshot-action-denormalization-active). |
| 2026-05-20 | Phase 2 implementation + experiment writeup lives at `docs/phase2_results.md` | Single canonical Phase 2 doc to land alongside Phase 3 work — captures B1–B5 design decisions, v1 numbers, B5.5 plan, and a B5.3 in-dist table slot the user fills post-collection. Supersedes the partial `docs/phase2_status.md` (kept for the sub-branch/commit table). |
| 2026-05-20 | B5.5 closed negative — stop chasing the SVF residual with data, change the label | v2 (denormalized snapshot actions) reproduced v1's residual at a partial operating point (~87% @ ~30% intervention). The intervention/residual curve is ~linear to the ~90% cliff → the critic is a clamp-fraction dial, not a discriminative classifier; residual is set by the structural proximity floor (human-approach-driven). Denormalization was still worth landing (hard-gate residual <1% @ R≈3.5). Next lever is offline label work (tighter-τ relabel / robot-controllability-aware label), not a v3 collection. |
| 2026-05-27 | **Fresh G1 coworker swap on `retryg1`, NOT a port of `safety-critic/g1-coworker`** | Previous attempt (commit `8beb0ec`) trained stably only with `MASK_PIXELS=1`, capping task-success vs RGB-enabled SMPL-H. User instructed clean retry — implementation error in the prior version, don't carry it forward. SMPL-H code path is byte-untouched; G1 lives in parallel classes selected by `env.human_model`. Default stays `smplh` so existing runs are unaffected. |
| 2026-05-27 | G1 visual strategy: **real Unitree STL meshes**, not skin-tone capsules | Strategy α (capsules) shipped first; supervisor flagged it as visually unconvincing. Switched to upstream meshes — closes the `_create_merged_world` `<asset>`-merge gap to make this work. Accepts the risk of re-introducing the prior attempt's CNN-encoder visual regression; tripwire = stage-0 reward <-10 by step 15k, fallback = revert to α (one commit back). |
| 2026-05-27 | Stage-2 `coworker_train` tightened to bring the human into the workspace | Previous run's stage 2 had proximity-violation rate 0.13-0.24 — too easy; the arm rarely reached into the robot. Tightened `closest_approach 0.9-1.4 → 0.55-0.85` (and four other knobs). The 0.55 m floor is the smallest body distance that keeps the mocap-pelvis capsule clear of robot pelvis collision (overlap would produce spurious huge contact forces since mocap can't be pushed back). Body-agnostic — applies identically to SMPL-H and G1. |
| 2026-05-27 | Stage budget reallocation for next run (20k / 15k / 60k) | Previous run analysis: stage 0 peaked ~step 15k then degraded by 28k (curriculum picks up last snapshot, wasting the back half); stage 1 hit `success=1.0` by step 11k and saturated; stage 2 needs more frames for the new harder distribution. Net frames similar; allocation matches signal density. |
| 2026-05-27 | `safety_metrics.md` schema fully implemented in code | The doc (2026-05-26) was prescriptive; only single-flavour SSM lived in code. Implementing it now is necessary for the curriculum scripts to log the thesis-headline metrics (`ep_proximity_violation_rate`, `eval/ep_*` aggregates, `final_metrics.json`'s `best_eval`). |

---

## Notes / blockers

_Append as work proceeds. Each note dated. Most-recent first._

### 2026-05-27 — G1 swap + safety-metrics schema + stage-2 tighten

**Previous SMPL-H curriculum finished cleanly** at `exp_local/cqn_as_base_curriculum/base_curriculum_20260527_015253`. Eval-curve highlights per stage:

| Stage | Frames | Best `success_rate` (step) | Best `ep_reward` (step) | Final-snapshot `success_rate` |
|---|---|---|---|---|
| 0 idle | 30000 | 0.5 (13185) | -10.31 (5185) | 0.1 (28253) — degraded |
| 1 easy | 30000 | 1.0 (10909) | -1.76 (10909) | 1.0 (28496) |
| 2 full | 40000 | 1.0 (20002) | -1.85 (20002) | 0.9 (37626) |

Stage-2 `ep_proximity_violation_rate` hovered at 0.07-0.24 throughout — the agent solves the task but doesn't avoid violations (no Lagrangian; that's P3.1's job; and the OLD `coworker_train` was too easy — the human rarely reached into the workspace).

**Three structural changes landed today** on `retryg1` (workstreams G, M, S2 above). Smokes all green; full test suite **351 passed / 37 skipped**. No SMPL-H regressions.

**Key bugfix uncovered along the way (Workstream G):** `HumanController._get_amass_targets` ignored the trajectory planner when `clip is None`. Caused the SMPL-H smoke-without-AMASS to park the pelvis at spawn — explains why an early stage-2 smoke showed SMPL-H separation stuck at ~0.95 m even with tightened knobs. Fixed; SMPL-H now respects the planner the same way G1 does. No production impact (AMASS is always loaded in training).

**Asset-merge gap closed (Workstream G):** `_create_merged_world` didn't copy `<asset>` blocks. Silent under SMPL-H (empty `<asset>`) and under strategy α (no meshes referenced). The minute G1's STL meshes came back, it was load-bearing — `<mesh>` decls had to flow into the merged-into-world XML, with mesh `file=` paths absolutised so MuJoCo finds them after the merge writes to a temp dir.

**Hydra grammar gotcha (Workstream M):** the curriculum tag string `+wandb.tags=[stage0,method=unconstrained,task=...]` from the safety_metrics.md spec **does not parse** — Hydra's override grammar reserves `,` and `=`. Switched to `:` as the key/value separator inside tag strings. W&B accepts any string as a tag.

**Mesh-path portability (Workstream G):** the checked-in `g1_human_body.xml` stores mesh `file=` attributes as paths **relative to the XML's own directory** (`g1/assets/<file>.STL`). `_create_merged_world` absolutises them at load time using `self._human_body_path().resolve().parent`. So the asset is portable across machines (was originally written with absolute Mac paths; reworked when the user ran it on the GPU box at `/home/ap2322/...`).

**G1 base-curriculum run is the next GPU-box action** (full command + tripwires under "Next session — start here"). Re-running the SMPL-H curriculum **with the tighter stage 2** is also worth considering as a control if the user wants apples-to-apples comparison; the tighter band applies to both models.

### 2026-05-20 (later, post-smoke) — test-rot cleanup + new baseline
- **Documented 11 failures fixed** (commit `f550a57`): `test_cql_trainer.py` (7) + `test_svf_train_critic_smoke.py` (3) caught up to the B2.8 `write_shard` schema (added per-step `min_separation`/`pfl_force_ratio` kwargs, consistent with `r_safe`); `test_episode_safety_metrics.py` (1) stale "not until done" assertion replaced (the wrapper intentionally emits `episode_safety` every step). Those 3 files: 17 passed.
- **Repo hygiene** (`d9548c9`): 27 committed `__pycache__/*.pyc` untracked (they predated the gitignore rule), `*.mp4` ignored.
- **New baseline:** full `pytest tests/` **with `AMASS_DATA_DIR` set** = 335 passed, **5 failed in `test_svf_collect_smoke.py`**. These are AMASS-gated (they *skip* without AMASS, which is why the old "everything green" baseline missed them). Breakdown: 2 are stale 4-dof shape rot (assert action `(n,15)`, env is now `(n,16)` since B2.4); 3 are in the demo-source path (`test_demo_source_*` + the manifest test) failing on a `write_shard` shape `ValueError` — same area as the in-flight B5.5 `svf_collect_dataset.py` snapshot-denorm changes (bundled in `bed92f7`). **Deferred to the B5.5 session by user decision (2026-05-20); left untouched.**

### 2026-05-20 (later, post-smoke) — env.safety schema fix + commit-bundling note
- `cfgs/env/safety_bigym.yaml` now declares `add_workspace_penalty`/`workspace_radius`/`workspace_beta` (commit `bed92f7`). Strict-mode Hydra was rejecting `env.safety.add_workspace_penalty=true` overrides because the YAML schema only listed the violation-penalty fields; the workspace fields were already wired in `SafetyConfig` + the factory via `.get(..., default)`, but couldn't be set from CLI. Unblocks the D3b-validation run.
- Bookkeeping: commit `bed92f7` *also* carries the in-flight Workstream B5.5 work (`scripts/svf_collect_dataset.py` changes + new `tests/test_svf_collect_snapshot_denorm.py`) that was already staged when I committed the cfg fix. Files are real B5.5 progress, just landed in a commit message that doesn't mention them.

### 2026-05-20 (later) — D3b-smoke PASSED on GPU box
2000-frame smoke ran clean on swirl (`exp_local/cqn_as_safety/saucepan_to_hob_20260520124507`). Acceptance criterion 1 met end-to-end:
- `Loaded 10 demos` → `Converted 10 demos (6 successful) for CQN-AS` → `Loaded 10 demos; replay size now 3663`.
- Train loop ran 2000 frames with `replay_buffer_num_workers` default (2) and **no IndexError** — demos pre-filled both workers, the cold-start race is gone.
- **Episode 1 ran the full 1000-step budget** (the previous demo-less C2 baseline collapsed to 31-step evacuation). Ep1 `episode_reward=0` (expected at this scale; agent untrained) but `episode_length=1000` is the load-bearing signal.
- Per-step `safety/ssm_margin` swings cleanly between approach (negative, e.g. -0.36/-0.94) and retreat (positive, up to 7.95) as the COWORKER human moves. Ep1 violation rate decayed to 5.3% by step 1000.
- `snapshot_2000.pt` saved at exit (cadence fix firing).
- EGL teardown traceback at process exit is the harmless `EGL_NOT_INITIALIZED` already documented in the 2026-05-17 note — fires after the snapshot saves, no data impact.

Next run is D3b-validation at ~50k frames with `env.safety.add_workspace_penalty=true` (command in "Next session — start here"). That's the run that tells us whether the policy *attempts* the task vs runs a well-behaved random walk.

### 2026-05-20 — Workstream D landed (demo pipeline)
- **D0 resolved (red herring).** 4-dof demos load fine through `SafetyBiGymEnvFactory._get_demo_fn`; see decision log. No re-recording.
- **`get_demos` implemented** in `safety_bigym/agents/cqn_as/env_adapter.py` (replaces the `NotImplementedError` stub). Flow: `_get_demo_fn` raw-env+DemoStore load → truncate each demo at first reward>0 → (bodyslam≠off) inject `human_pos_estimate` via `BodySLAMWrapper(demo_replay=True)`+`AMASSDemoPositionProvider` driven over a new in-memory `_DemoReplayEnv` → `_convert_demo_to_timesteps` (obs via existing `_extract_obs`, action from `info["demo_action"]`, FIRST/MID/LAST typing, `cost=0.0`) → `_extract_action_stats` overrides `self._action_stats` → `_rescale_demo_actions` to [-1,1]. Returns `list[list[ExtendedTimeStep]]` consumed by `train_cqn_as.load_demos`.
- **Tests:** `tests/test_cqn_as_demos.py` (11 tests, stubbed DemoStore + AMASS provider, no MuJoCo) green; full `test_cqn_as_demos.py + test_cqn_as_adapter.py` = 40 passed.
- **Smoke is a GPU-box handoff.** This Mac venv lacks `tensordict==0.6.0` (declared in setup.py but only installed on the GPU box; user chose not to install locally). Verified locally: the live env builds and `_get_demo_fn` loads demos; `CQNASAgent` instantiation needs tensordict. Run on the GPU box:
  ```bash
  export AMASS_DATA_DIR=/path/to/CMU/CMU MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
  python train_cqn_as.py env=safety_bigym/saucepan_to_hob disruption=coworker_train \
    bodyslam=oracle num_demos=10 num_train_frames=2000 wandb.use=false
  ```
  Expect: `Loaded N demos; replay size now ...`, 2000 frames train with no crash and no worker-striping IndexError. Then the ~30–50k-frame validation with `env.safety.add_workspace_penalty=true` for task-attempt evidence.
- **Next:** straight to Phase 3 (P3.1); fold the obs-channel ablation into E3.6 (no standalone E1.4 re-run).

### 2026-05-20 — handoff snapshot
- **State of `main`:** P3.0 merged (PR #9). Two follow-up commits pushed: `6e7fdc1` (snapshot-cadence fix) and `09faaa4` (eval video). `origin/main` is current; the GPU box only needs `git pull`.
- **The one live blocker:** C2's first run saved no snapshots. Kill + restart is the first action of the next session (see "Next session — start here"). Nothing else is blocked.
- **Test suite baseline:** 11 pre-existing failures on `main` unrelated to recent work — 7 in `test_cql_trainer.py` + 3 in `test_svf_train_critic_smoke.py` (both from a `TransitionShardWriter.write_shard` signature drift that added required `min_separation`/`pfl_force_ratio` kwargs the tests don't pass), 1 in `test_episode_safety_metrics.py` (`episode_safety` now emitted every step, stale "not until done" assertion). These are real test-rot, not regressions — worth a cleanup pass but not blocking. Everything else green (~277 passing).

### 2026-05-19 — snapshot-cadence bug + eval video
- **Snapshot-cadence bug (root cause).** `Workspace.train()` called `save_snapshot()` only from inside the `if time_step.last():` block, gated by `utils.Every` (`step % every == 0`). A snapshot landed only when an episode boundary *coincidentally* fell on a multiple of `snapshot_every_frames`. For saucepan_to_hob COWORKER-train (episodes ~150 steps, stochastic termination), that alignment is ~never → 200k-step runs saved zero snapshots despite `save_snapshot=true`. Fix `6e7fdc1`: hoist the check out of the episode-end block to fire every step, **plus a final-state save** after the train loop exits (eval-curve peak is usually near the end). Regression test `tests/test_cqn_as_snapshot_cadence.py` reproduces the pre-fix bug and pins the post-fix schedule.
- **Eval video (`09faaa4`).** `cfg.save_video` existed in `cqn_as_config.yaml` but was unwired (no-op). Now `Workspace.eval()` captures episode-0 frames per eval cycle → `eval_videos/step_<step>_ep0.mp4` @30fps + W&B `eval/video`. Helpers in `safety_bigym/agents/cqn_as/eval_video.py` (separate module so tests skip the MuJoCo import chain); all failures swallowed + logged so a video glitch can't kill a training run. `phase1_reward_pilot_cqn_as.py --train` emits `save_video=true` automatically.
- **Two real bugs the P3.0d smoke surfaced** (both fixed in P3.0d commit `17c6dc0`, now on main via PR #9):
  1. **`SafetyBiGymEnv._compute_workspace_penalty` used `self._robot.get_ee_position()` — H1 has no such method** (it's `get_hand_pos(HandSide)`, which returned `[0,0,0]` at rest anyway). The original P3.0a silently no-op'd on every task. Fixed to read EE from `_get_robot_state()` with the `ee_pos → link_pos["ee"]` fallback (`link_pos["ee"]` is populated via `_ROBOT_LINK_NAMES` mj_name2id and is the load-bearing path for H1). See CLAUDE.md gotcha.
  2. **`safety_bigym_factory._create_env` wasn't threading the workspace-shaping fields** (`add_workspace_penalty`/`radius`/`beta`) from `cfg.env.safety` into `SafetyConfig`, so YAML/CLI overrides had no effect at training time. Now mirrors the `add_violation_penalty` plumbing.

### 2026-05-18 (later — B5 complete, C2 unblocked)
- **B5 done end-to-end.** SVF v1 critic trained, eval'd, threshold-swept. Headline: at R≈4.0 the runtime gate catches 95% of violations on `coworker_eval × random policy` (residual_violation 3–8% vs random's 95% baseline). The Q gap between safe and unsafe states is tight (cliff between R=3 and R=4) — diagnostic of an underfit/distribution-shifted critic, not a broken one. In-distribution eval at `coworker_train` pending to settle the question.
- **Two more train-cqn-as bugs fixed today** unblocking C2 training and C2 eval:
  1. **DataLoader collate "Trying to resize storage that is not resizable"** (`safety_bigym/agents/cqn_as/replay_buffer.py:_copying_collate`). Default torch.utils.data collate uses `torch.as_tensor(numpy_array)` (zero-copy), then in worker processes takes a shared-memory path `elem.new(storage).resize_(...)`. With numpy-backed elements, `elem.new` inherits the non-resizable attribute and `.resize_` raises. Deterministic on torch≥2.5 + num_workers>0; bit A6 v5 and C2 cell 1 (bsoff) at the second batch. Fix: custom collate that copies via `torch.from_numpy(np.ascontiguousarray(x)).clone()` before `torch.stack`.
  2. **Snapshot save/load no-op** (`safety_bigym/agents/cqn_as/agent.py` + `train_cqn_as.py:Workspace.load_snapshot`). The vendored `CQNASAgent` had no `state_dict` / `load_state_dict`; the existing `save_snapshot` path silently wrote `agent_state=None`. Added both methods (round-trip encoder + critic + critic_target; optimizers loaded best-effort). `train_cqn_as.py` main now branches on `+snapshot_path=...`: load before train/eval; if `num_train_frames<=0` run eval() once and exit. This unblocks the C2 **eval** half — `phase1_reward_pilot_cqn_as.py --eval` was already emitting `+snapshot_path=...` in command lists.
- **C2 status:** unblocked. C2 cell 1 (saucepan_to_hob / bsoff) had crashed at step 2000 with the collate error before today's fix.
- **B5.4 threshold sweep methodology note for B5.5/C2 eval:** the v1 sweep at default thresholds [10, 25, 50, 75, 90] was uniformly intervention=1.0 because the trained Q distribution lives in roughly [0.5, 5.0]. Future sweeps should start from `q_mean` (read off training log: ~3 for v1) and bracket ±1.5×. Encoded the recommendation in B5.4 entry above.

### 2026-05-18 — A6 smoke gates green
- **A6 cleared in run `cqn_as_smoke_dishwasher_oracle_v4`.** 2000 frames, 38 episodes, no crash. All four gates pass — see the A6.1-A6.4 entries above. Run accessible at wandb `safety-critic/runs/pt18q5ir`.
- **Four bugs fixed en route to green, all on the vendored CQN-AS / train entrypoint:**
  1. **Missing `tensordict` dependency** — `safety_bigym/agents/cqn_as/{agent,utils}.py` import it; not in `setup.py`. Pinned `tensordict==0.6.0` (matches `CQN-AS/conda_env.yml`) plus `dm_env`. GPU box needed `pip install tensordict==0.6.0` once.
  2. **Python 3.12 `random.seed(numpy.uint32)` rejection** — `replay_buffer.py:_worker_init_fn` passed a numpy uint32 to `random.seed`, which 3.12 rejects via TypeError (older versions tolerated via `__index__`). CQN-AS upstream's conda env pins python=3.10 so it never hit this. Fix: `random.seed(int(seed))`.
  3. **TensorDict bool conversion in train loop** — `agent.update()` returns a TensorDict; my training loop's `if metrics:` and `_log`'s `if not metrics:` both raise on TensorDict. Replaced with explicit length checks.
  4. **Worker-aware update gate** — `len(replay_storage) > 0` is necessary but not sufficient: the replay loader stripes episode files by `eps_idx % num_workers`, so worker N is empty until episode N-1 is stored. Upstream never hits this because `num_demos > 0` pre-fills every worker. With `num_demos=0` (smoke), updates must wait for `global_episode >= num_replay_workers`. Without this guard, DataLoader worker 1 raises IndexError on its first sample.
- **One follow-up still in flight (not blocking A6 green):** the training-update `[train] step=N q_critic_loss=...` lines didn't appear in the v4 log because `_log` iterated `TensorDict.items()` twice (dict comprehension + format-string join) and the second iteration was empty under the single-use-generator semantics of tensordict 0.6.0. Patched to materialise items into a list once (also calls `.item()` on 0-d tensors for clean float formatting). The training itself was happening — the step rate dropped from 117/s to 6.4/s after episode 2, which is the agent update overhead. Verifiable on the next smoke run.
- **B5.1 silent return resolved.** The earlier "py exit=0 + 0 bytes" was the script succeeding silently — `checkpoints/svf_smoke.pt` (626k) already on disk from the supposedly-silent run. Root cause: `logging.basicConfig` was a no-op because an earlier import had already configured the root handler; with no INFO handler in scope, every `logger.info` was silently dropped. Fixed with `force=True` on the basicConfig call. Next B5.1 run should print "Loading dataset…", per-step loss progression, and either "Smoke OK" or a SystemExit with the failed Bellman-MSE assertion.

### 2026-05-17 (later — A7 + C1 landed, Claude-side parallel work)
- **A7 adapter pytest landed** at [`tests/test_cqn_as_adapter.py`](../tests/test_cqn_as_adapter.py). 24 tests, all green; 46/47 still green on the sibling suites (`test_coworker_disruption`, `test_svf_dataset`, `test_safety_labeling`, `test_bodyslam_wrapper`). Tests use a monkeypatched `SafetyBiGymEnvFactory._create_env` returning a stub gym env so the suite runs in <3s with no MuJoCo / AMASS / W&B dependencies. Coverage: low_dim shape ± bodyslam, action [-1,1] roundtrip including gripper-tail [0,1] handling, TimeStep first/mid/last typing on reset/mid/terminal, episode_length-divided-by-demo_down_sample_rate truncation, info["safety"] per-step + info["episode_safety"] at episode-end forwarding, frame-stack widening + first-frame repeat-fill on reset, pixels=False zero-rgb placeholder, missing-state-key + missing-bodyslam-key fail-loud, ExtendedTimeStepWrapper action attachment + spec forwarding.
- **C1 sweep script landed** at [`scripts/phase1_reward_pilot_cqn_as.py`](../scripts/phase1_reward_pilot_cqn_as.py). Modelled on `phase1_reward_pilot.py` but invokes `train_cqn_as.py`. CLI mirrors the existing pilot: `--train` prints the 3 ready-to-run training commands (`bodyslam=off|oracle|noisy` × `saucepan_to_hob` × `disruption=coworker_train`, `num_train_frames=200000`, `num_demos=0`, `env.safety.add_violation_penalty=true env.safety.violation_penalty=0.05`, W&B on with `e1.4` tag); `--eval` prints commands against `disruption=coworker_eval` (20 episodes × 3 seeds × 3 modes) once the SNAPSHOTS dict at the top of the script is populated post-train; `--smoke --cell <mode>` runs a 2000-frame validation on a single cell for local sanity checks. Headless env vars (`MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0`) are embedded in every command.
- **C1 caveat (snapshot loading).** The eval commands pass `+snapshot_path=<path>` and `num_train_frames=0`, mirroring `train_safety.py`'s convention. `train_cqn_as.py` does NOT yet honour `+snapshot_path` (Workspace.__init__ has no eager load — there's only `save_snapshot`, no `load_snapshot`). Wiring this in is a small follow-up (~15-line addition to `train_cqn_as.py` mirroring the Phase-0 robobase drift fix at the eager-load path). Blocking only the C2 eval phase, not C2 training. Tracked here so it doesn't get lost between now and C3.
- **Both deliverables are GPU-free.** A6 smoke gates + B5 SVF training remain the next GPU dependencies; A8 PR is the only Claude-side item still gated (waits on A6 green).

### 2026-05-17
- **Mid-scale B3 validation passed.** 6 cells × 20 episodes × 250 max-steps × 0.50m proximity threshold + noisy bodyslam → 30k transitions clean. Per-cell violation rates: `random/dishwasher_close`=14.7%, `random/drawers_open_all`=32.4%, `random/saucepan_to_hob`=15.8%, `snapshot/dishwasher_close`=6.0%, `snapshot/drawers_open_all`=4.0%, `snapshot/saucepan_to_hob`=7.7%. Aggregate ~12% violation rate; all cells well above 5% floor, sources distinguishable per task (10-25 point gaps).
- **Source-diversity pattern reversed from v5** (the single-task oracle smoke). v5 had snapshot > random on `dishwasher_close` (15.5% vs 11.0% at 0.50m oracle). Mid-scale (multi-task, noisy) has **random > snapshot on every task** (factor of 2-8×). The new pattern is more intuitive — random's chaotic limb extension produces more accidental near-human transitions than a task-trained policy that mostly stays at the workstation. The v5 reversal was a single-task artifact; multi-task data clarifies the underlying signal. Doesn't change B3 design; both patterns produce informative datasets.
- **Two small fixes landed this session:**
  - `TASK_REGISTRY` import path for `saucepan_to_hob` corrected from `bigym.envs.saucepan` (doesn't exist) to `bigym.envs.pick_and_place.SaucepanToHob`. Was crashing the mid-scale run between the 4th and 5th cells; one-line fix in [svf_collect_dataset.py:72](../scripts/svf_collect_dataset.py#L72).
  - `_build_live_env` now passes `SafetyConfig(log_violations=False)`. Under proximity-based labelling, the env's per-step `SSM Violation!` WARNING fires almost every step (kitchen-scale `Required: 5-20m` is meaningless under proximity bar), drowning out useful collection logs. `ssm_margin` / `min_separation` remain populated in `info["safety"]` and stored in shards; only the logger is muted. Was flagged as a follow-up in the 05-16 notes; now landed.
- **EGL teardown traceback at process exit is harmless.** MuJoCo's renderer destructor calls `eglDestroyContext` after EGL has already shut down → `EGL_NOT_INITIALIZED`. Happens *after* the final "Wrote N transitions" line, so all data is already on disk. Affects every collection run; ignore or, if it ever blocks CI, install an `atexit` cleanup before EGL teardown. Tracked here only so future readers don't waste time chasing it.
- **v1 dataset follow-ups (none blocking SVF training, all worth knowing):**
  1. **Snapshot action denormalization.** `_SnapshotPolicy.__call__` returns the agent's tanh-space output as-is. RoboBase training wraps the env with `RescaleFromTanhWithMinMax`, which our SVF collection path doesn't replicate — the env silently clips gripper dims to [0, 1] and the body-joint actions occupy only the inner `[-1, 1]` band of the env's true ±π range. Snapshot rollouts are still task-relevant (10% per-task violation rate is non-degenerate) but explore a narrower action subspace than a deployed properly-rescaled policy would. Decision: ship v1, evaluate runtime SVF; if it behaves oddly on actions near joint limits, fix the denormalization for a v2 collection. Estimated fix: ~10 lines in `_SnapshotPolicy.__call__` to unrescale using the `_action_stats` already in the snapshot payload.
  2. **`ScenarioParams` missing two axes.** `make_coworker_train_space` samples 5 parameter axes but the `ScenarioParams` dataclass at [scenarios/scenario_sampler.py:25-64](../safety_bigym/scenarios/scenario_sampler.py#L25-L64) only exposes 3 (trajectory_type, closest_approach, walk_speed, patrol_near_loiter); `reach_period` and `target_mix_p_ee` are consumed internally but never persisted on the dataclass. B4.4 coverage check can't audit these two axes. Small dataclass extension; independent of any other workstream.
  3. **Aggregate 11% violation rate is workable but skewed safe.** With class-weighted CQL training it's fine. If the v1 SVF underperforms on the unsafe class (under-predicts risk), the right knob is probably either (a) tighten the proximity threshold for retraining-from-shards (free — `min_separation` is per-step), or (b) re-collect with `closest_approach` skewed toward the lower end of the COWORKER train range. (a) is much cheaper.

### 2026-05-16
- B2.3 → B2.7 landed in one session. The Phase 2 dataset architecture is materially different from the original B2 sub-plan; see decision log for the three substantive decisions (labelling, threshold, demo skip).
- **Dead demo source (open follow-up).** BiGym DemoStore cache has demos recorded with `JointPositionActionMode(absolute=True, floating_base=True, floating_dofs=['pelvis_x', 'pelvis_y', 'pelvis_rz'])` — 3-dof base. Phase 2 production env uses 4-dof base (X, Y, Z, RZ) to match RoboBase ACT training (B2.4). Metadata signature mismatch → `DemoNotFoundError`. Recording fresh 4-dof demos through `mojo.demonstrations` (or scripting them via the SNAPSHOT actors run as expert) is the canonical fix; tracked as a future task, not blocking B3.
- **Random vs snapshot policy-effect signal observed at threshold ≥ 0.50m.** At 0.10m and 0.30m the two sources produce statistically indistinguishable violation rates — kitchen-fixed-workspace tasks (e.g., dishwasher_close) have the human's AMASS trajectory dominating relative-proximity statistics. Above 0.50m, snapshot's task-driven reach toward the workspace overlaps more with human approach trajectories than random's whole-workspace flailing. The 4.5-point gap at 0.50m (random 11.0%, snapshot 15.5%) is the source-diversity signal we relied on for the dataset design.
- **Snapshot policy adapter loads cleanly** after B2.4-B2.6: encoder builds, `load_state_dict` succeeds, rollouts produce sensible `data.cvel`-derived robot velocities. Caveat for any future snapshot-eval path: `_SnapshotPolicy._synthesize_snapshot_obs_space` mirrors RoboBase's exact wrap order (ConcatDim → FrameStack), and `adapt_obs` mirrors ConcatDim's `keys_to_ignore` and bodyslam-aware gating. If you change either upstream, both helpers need to change in lockstep — silent state_dict mismatches are easy to introduce.
- **PFL retrofit recipe** (when the contact-detection bug eventually lands):
  1. Re-collect through a PFL-fixed env using the same `svf_collect_dataset.py` command (the writer schema is already in place; only `pfl_force_ratio` values change from zero to meaningful). Output to a new directory — don't mix.
  2. Train SVF with `label_transition(..., use_pfl=True)` so the binary label ORs in `pfl_violation`. The proximity threshold can remain at 0.50 m or be tightened; PFL catches the contact tail PFL was designed for.
  3. Compare v1 (proximity-only) and v2 (proximity + PFL) critics on identical eval episodes to confirm PFL adds discriminative signal rather than just inflating the violation rate.
  Stored `min_separation` from v1 still relabels for new proximity thresholds; stored `pfl_force_ratio` from v1 is all zeros and useless for retrofit — that's why step 1 needs fresh data, not just relabelling.

### 2026-05-15 (later)
- A1 passed on GPU box. CQN-AS reference impl learns on stock `dishwasher_close` end-to-end. Vendoring track unblocked.
- B1 (3 ACT re-rolls on COWORKER train space) ran on GPU box. Snapshot paths pending — user to paste W&B URLs or `exp_local/...` paths.
- Hydra group regression fixed: `cfgs/disruptions/` → `cfgs/disruption/`, registered in `safety_config.yaml` defaults, yamls now set `env.disruption_type=COWORKER` (forces weights override via factory line 220) so legacy weights dict merging doesn't pollute the COWORKER-only distribution.
- **GPU-box prerequisite (record once, forget never):** `export MUJOCO_GL=egl; export MUJOCO_EGL_DEVICE_ID=0` (headless box has no DISPLAY; default GLFW backend fails on `mjr_makeContext`).

### 2026-05-15
- Branch created off `main` carrying forward pre-existing uncommitted edits to: `safety_env.py`, `safety_bigym_factory.py`, `human_controller.py`, `trajectory_planner.py`, `disruption_types.py`, `scenario_sampler.py`. Committed as `c75ee12` (Phase 0.5 baseline).
- CQN-AS cloned at upstream commit `8cf806e` (HEAD of `/Users/ayushpatel/Documents/FYP3/CQN-AS/`).
- BiGym pin delta noted: safety_bigym's BiGym at `79420b0` (2 commits ahead of CQN-AS pin `72d3054`). Both delta commits are typo fixes; compatibility expected (and now confirmed by A1).

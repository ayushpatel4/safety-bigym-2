# IMPLEMENTATION_STATUS.md
### Coding-agent brief — `safety_bigym` MEng project
### Updated: 2026-06-05 (round 4 — RESULT LOCKED: fixed-λ=0.1 is the robust graceful ROW3, −22.8% proximity @ succ 0.76 across 3 seeds)

This document is the **single source of truth for what needs running**.
It pairs with `REPORT_STRATEGY.md` (why) and `report.tex` (the deliverable).
Tasks are priority-ordered; smoke gates are mandatory before any
multi-hour launch.

> **2026-06-09 status delta — P9 (WCSAC external baseline) IMPLEMENTED.**
> - `agent=wcsac` on the CQN-AS stack (`train_cqn_as.py`): faithful Worst-Case
>   SAC — stochastic squashed-Gaussian actor, twin reward critics, a Gaussian
>   safety critic (mean+variance via the WCSAC 2nd-moment recursion), closed-form
>   Gaussian CVaR_α constraint, projected-dual λ, auto entropy temperature.
>   Reuses the existing per-step cost + cost-carrying replay + eval harness;
>   **zero RoboBase drift**. Code `safety_bigym/agents/wcsac/`, config
>   `cfgs/agent/wcsac.yaml`, tests `tests/test_wcsac_agent.py` (15 pass), scripts
>   `scripts/wcsac_{smoke,train}.sh`.
> - **Verified**: 15/15 unit tests + a ≤100-frame env smoke (80 update steps,
>   0 non-finite; λ rises while CVaR>budget — the constraint mechanism works).
> - **Two honest departures from the Lagrangian baseline** (documented in
>   `wcsac_train.sh`): WCSAC trains **from scratch** (no CQN-AS warm-start —
>   architecture mismatch) and `cost_budget` is a CVaR-of-cost-**return** ceiling,
>   not the per-step mean the Lagrangian sweeps. From-scratch pixel SAC may
>   underperform the warm-started value method → the documented §disc:wcsac-honest path.
> - **Status**: GPU sweep (dish + drawers, budget grid) launched; eval pending.

> **2026-05-30 status delta.**
> - **P1 DONE** — G1 base-policy curriculum ran; stage-2 snapshot in hand
>   (the unconstrained baseline / row-1 reference).
> - **P2 CLOSED** — SVF recollected on G1+`noisy`, retrained at τ=0.3 m as
>   `svf_coworker_train_g1_0p3.pt`, and the **dense 0.3 m sweep** (R=0 baseline +
>   fine grid) is done. Operating point **R = 2.25** pinned in
>   `safety_bigym/filters/snapshots.py::SVF_FILTER_THRESHOLD_R`. Full write-up:
>   **`phase2_results.md` §0** (authoritative; supersedes the coarse sweep).
> - **Proximity label is now τ = 0.3 m** (`SSMConfig.proximity_threshold=0.3`,
>   was 0.5 m); the G1 SVF trained at this τ.
> - **Key P2 finding (bank for §results:filter-pareto):** filterless baseline
>   proximity-violation rate is **0.0435**. The P2 acceptance bar **IS met** at
>   **R=2.25: 31.7% reduction at 21.6% intervention** (≤25%) — but it's
>   **marginal and seed-fragile** (per-seed 38.4 / 41.2 / **20.6**%). The big
>   proximity win (82%) only arrives at the R=3.0 hard gate (~79% intervention,
>   robot ~frozen); the filter's robust low-cost win is the robot-velocity
>   ISO-SSM axis, because a veto→freeze filter can't stop the G1 coworker
>   approaching a stationary robot. **Core hybrid argument**: filter = edge-case
>   backstop, Lagrangian policy = proactive avoidance. R=2.25 is provisional —
>   re-confirm against the Phase-3 row-3 snapshot in P5.
> - **P3 cost-form selector DONE (2026-05-30)** — all three E3.1 cells are now
>   wired and Hydra-composable: **continuous** (`agent=cqn_as_lagrangian`),
>   **binary** (`env.safety.cost_form=binary` → `c_t=1[ssm_violation]`, via the
>   new `select_cost` in `filters/cost_signal.py` + `env_adapter.py`), and
>   **fixed** (`agent=cqn_as` + the pre-existing, already-threaded
>   `env.safety.add_violation_penalty`/`violation_penalty=0.05`). The "fixed"
>   cell needed **no new code**. `scripts/run_e3_1_cost_signal.sh` launches the
>   full 3×3 matrix. 31 cost-path tests pass; composition verified. **E3.1 is
>   launch-ready** — just set `WARMSTART` to the P1 stage-1 snapshot.

> **2026-05-31 update.**
> - **E4.1 evaluates the WHOLE table on `noisy`** (`run_e4_1_headline.sh` default
>   `OBS_MODE=noisy`). The G1 SVF is noisy-native: on `oracle` its Q collapses
>   (`mean_q≈0.016 ≪ R=2.25` → 100% intervention, success 0.78→0.0; the broken
>   oracle run `results/e4_1/..._190001` is kept as §disc evidence). Noisy keeps
>   the filter in-distribution and the comparison apples-to-apples. `OBS_MODE=oracle`
>   = policy-only reference (rows 1–3 only; filter rows are meaningless there).
> - **E4.3 is post-hoc on noisy** (`run_e4_3_internalisation.sh`), NOT the free
>   in-training `FILTER_PASSIVE` hook (same oracle-collapse → flat curve).
> - **Lagrangian warm-start bug fixed (`2577355`)** — `LagrangianCQNASAgent.load_state_dict`
>   now guards on the cost-net keys, so P3/P4 can warm-start from the plain
>   (unconstrained) stage-1 snapshot (was `KeyError: 'cost_encoder'`). Re-launch
>   e3_1/e3_2 after `git pull`; **do not set `FILTER_PASSIVE`**.
> - **Toolchain complete** — one command per stage: `svf_sweep_g1_v1_baseline.sh`
>   (P2), `run_e3_1/2_*.sh` + `analyze_e3.py` (P3/P4 + d_knee), `run_e4_1_headline.sh`
>   + `aggregate_e4_1.py` (P5), `run_e4_3_internalisation.sh` (E4.3),
>   `aggregate_e5_1.py` (E5.1). All on `origin/phase3`.
> - **Open**: P9 (WCSAC) not built; row-2 decision (4-row table vs no-shaping retrain).

> **2026-06-01 update — ⚠ P2 RE-DO AGAIN (v3): two coupled train/deploy mismatches.**
> The SVF filter had **two** bugs where the collection policy didn't match the
> deployed policy, so the critic trained on actions deployment never executes →
> over-veto. Both now fixed in `_CQNASSnapshotPolicy`; a **v3 re-collect is
> pending** (`run_p2_recollect_g1.sh`, VER=v3).
> - **Bug 1 — action de-norm** (fixed, commit `41fd93b`): de-normalised via
>   `env.action_space` not the agent's **demo-derived** stats. Symptom: benchmark
>   `mean_q≈0.02`, ~100% intervention. The v2 re-collect fixed this and the v2
>   *sweep* looked great (R=2.50, 31.9% reduction @ 7.9% intervention). But the
>   **sweep reuses the collection path**, so it was over-optimistic / in-distribution.
> - **Bug 2 — action-execution mode** (fixed, commit on `phase3`): CQN-AS deploys
>   with `action_sequence=16` + `temporal_ensemble=true` → the runner executes an
>   exp-weighted **ensemble blend** of overlapping chunks, but `_CQNASSnapshotPolicy`
>   collected data **receding-horizon (raw `chunk[0]`)**. So the v2 critic trained
>   on `chunk[0]` but deployment runs blended actions → **E4.1 rows 1/4 (noisy):
>   row-4 = 89.5% intervention, `mean_q` 0.97, success 0** (vs sweep's 7.9%/3.3).
>   15/16 open-loop fraction ≈ the 89.5% observed. **Fix**: the snapshot policy now
>   reuses the SAME `TemporalEnsembleControl` and mirrors `CQNASRunner.step`
>   (`tests/test_snapshot_policy_execution.py`).
> - **Bug 3 — env control_frequency (THE root cause, pinpointed via the diagnostic).**
>   `_build_live_env` built the collection/sweep env via `make_safety_env` WITHOUT
>   `control_frequency`, so it ran at the full `CONTROL_FREQUENCY_MAX=500 Hz` (1
>   physics substep/step, `action_scale=1`). The RoboBase factory `_create_env`
>   (which the deployment adapter uses) sets `control_frequency = 500 //
>   demo_down_sample_rate = 20 Hz` for saucepan (25 substeps/step, `action_scale=25`).
>   The 25× mismatch: each action moved the robot ~25× less/step → the policy
>   **never completed the task** (`diagnose_collection_policy`: 0% success vs
>   deployment 85%); action_scale differed 25× (so even demo-stats de-norm was
>   mis-scaled); and 1000 steps covered ~2 s vs deployment's ~50 s → coworker
>   barely approached (proximity 0.05 vs 0.30). **Fix**: `_build_live_env` now sets
>   `control_frequency = CONTROL_FREQUENCY_MAX // demo_down_sample_rate` (read from
>   the task env yaml); propagates to collection AND the sweep. The earlier
>   `max_steps 250→1000` change was necessary (deployment uses 1000 control steps
>   ≈ 50 s) but inert without this.
> - **Bug 3 fix CONFIRMED** via the diagnostic: with `control_frequency` set, the
>   collection policy now **completes the task — 100% success, ~310-step episodes,
>   reward 1.0, sane actions** (was 0% / full-length).
> - **Bug 4 — coworker scenario param drift** (surfaced once Bug 3 was fixed):
>   the policy completed the task but proximity was **0.000** (coworker stayed
>   0.53–1.07 m away) vs deployment's 0.296. `_build_live_env` built the scenario
>   via `make_coworker_train_space` → `_COWORKER_TRAIN_RANGES` (Python preset),
>   which **drifted** from `coworker_train.yaml`: the yaml was tightened
>   (closest 0.60–0.95, fast reach 1.3–2.2, near_loiter 9–14) but the preset stayed
>   loose (closest 0.9–1.4, slow reach 4.5–6.5). The deployment factory reads the
>   yaml (`cfg.env.disruptions`) and never touches the preset. **Fix**:
>   `_resolve_coworker_overrides` reads the disruption yaml + builds the
>   ParameterSpace via `_coworker_space`, matching deployment (collection + sweep).
> - **All four bugs traced to one cause**: `_build_live_env` is a hand-built
>   replica of the factory `_create_env`, and each replicated piece (de-norm,
>   execution, control_frequency, scenario) drifted. All four fixed in svf_collect.
> - **✅ P2 v3 DONE — critic VALID, but the filter is a velocity backstop, not a
>   proximity reducer.** Collection at 20 Hz: random source is too slow/unstable
>   (contact thrash) so collection is **snapshot-only** (`SOURCES=snapshot`, default;
>   fast + stable + 16.8% unsafe = good class balance). v3 dense sweep
>   (`results/svf_sweep_g1_0p3_v3/`): **R=0 proximity 0.286 == benchmark row-1 0.296**
>   — the sweep finally PREDICTS the benchmark (validates all four fixes). BUT **no
>   knee meets the P2 bar** (≥30%@≤25%): best is 8%@15% (R=2.25); 33%@67% (R=3.0);
>   and even 100% intervention only cuts proximity 58% — the other **42% is the
>   coworker walking up to the stationary robot** (exogenous). A reactive
>   veto→zero-velocity filter **can't cheaply prevent human-initiated proximity**.
>   → The filter is reframed as an **ISO-SSM velocity backstop**; proactive proximity
>   avoidance is the Lagrangian policy's job (P3). The v1/v2 "31.7%@21.6% knee" was an
>   artifact of the broken collection (policy never got close). Pinned **R=2.25**
>   (light backstop) in `snapshots.py`; `_v3` checkpoint now resolves.
> - **✅ E4.1 rows 1/4 ran; fallback sub-study COMPLETE — reactive filtering hits a
>   freeze-vs-flee dilemma on the baseline policy.** Three operating points fully
>   bracket it (all on the valid v3 critic, `mean_q`≈3, sweep==benchmark confirmed):
>   - **zero_velocity** (R=2.25, 15% interv): proximity ~unchanged (robot *dwells*
>     near the approaching human), mean robot vel ↓ ~8%, success ~baseline. ISO-SSM
>     **velocity backstop** only — no geometric-proximity reduction.
>   - **retreat aggressive** (R=2.5, step 0.10, 14% interv): proximity 0.296→0.095
>     (−68%) BUT success **0.85→0.18** and mean robot vel **0.29→1.70 (6×)** →
>     `ep_ssm_violation_actual_rate` 0.147→0.180 (WORSE). The robot *flees*: buys
>     separation by abandoning the task and speeding up (itself an ISO hazard).
>   - **retreat gentle** (R=2.25, step 0.04, 2% interv): proximity 0.296→0.290
>     (~0%), success 0.85→0.75, vel still up. Does nothing useful.
>   **Conclusion**: no reactive setting cuts geometric proximity at acceptable
>   success+velocity. Against an *approaching* human, reducing proximity needs
>   sustained robot movement → task abandonment + velocity spike. The filter is a
>   reactive backstop with an inherent freeze(dwell)-vs-flee(abandon) tradeoff;
>   **proactive proximity avoidance must come from the Lagrangian policy.** This is
>   a clean, well-evidenced thesis result (the `RetreatFallback` + `FALLBACK`/
>   `SVF_RETREAT_STEP` knobs + `tests/test_retreat_fallback.py` are landed).
>   **Full write-up + the coworker-aggressiveness framing decision (keep the
>   adversarial finding as the headline; use realism only to build a spectrum, not
>   to flatter the filter; the policy is the decisive test):
>   [docs/filter_fallback_findings.md](filter_fallback_findings.md).**
> - **Next = the hybrid (rows 3/5), the actual architecture**: finish E3.1/E3.2 →
>   `analyze_e3` → cost form + `d_knee` → set `ROW3` → full `run_e4_1_headline.sh`.
>   The policy avoids proactively (reducing proximity *without* the flee-cost); the
>   filter (zero_velocity, the velocity backstop) catches the residual. Re-confirm
>   the filter R on the row-3 policy there. `snapshots.py` stays at `_v3` + R=2.25 +
>   default zero_velocity (the honest baseline backstop).
> - **Unaffected throughout**: E4.1 rows 1–3 (policy-only) and the e3_1/e3_2 training runs.

> **2026-06-03 update — E3.1/E3.2 + 3-seed CONFIRM done; the policy question is
> ANSWERED with a twist.** The proactive Lagrangian is the decisive test, and:
> - **It DOES reduce proximity ~21% (deployment-confirmed, constraint-driven) —
>   but it was hidden by checkpoint selection.** At d=0.3 (the only viable budget:
>   d≤0.2 collapse to 0% success, d=0.5 leaves λ=0 = unconstrained), `snapshot_best`
>   (peak success) AND the final checkpoint deploy ≈ baseline (0.30); the avoidance
>   lives in a **mid-training basin** (~20k–35k steps: 6 consecutive checkpoints at
>   0.21–0.25, succ 0.72–0.80, ssm-actual *improved*). **`pick_best_snapshot.py`
>   picks peak success → WRONG for a Lagrangian** (selects against the constraint);
>   it caused two false-null E4.1 waves. Use **`--by safety`** or pick from the
>   deployment basin sweep.
> - **3-seed CONFIRM (d=0.3) → the PID-Lagrangian is SEED-UNSTABLE.** λ landed at
>   **0.000 / 0.267 / 3.855** across seeds → **unconstrained / graceful-basin /
>   windup-collapse** (succ→0). d=0.3 sits at the task's inherent cost (~0.25–0.30),
>   so the dual variable is set by each seed's stochastic cost trajectory; the
>   graceful basin reproduces in **only 1/3 seeds**.
> - **The basin IS constraint-driven, NOT checkpoint-mining** — verified against the
>   natural control: seed-1 (λ=0) stays flat ≈0.30 with only isolated noise dips (no
>   coherent basin); seed-0 (λ=0.27) has 6 consecutive sub-0.26 checkpoints. Figure:
>   `results/figs/d0p3_3seed_lambda_regimes.png`.
> - **FIX IN FLIGHT: fixed-λ≈0.27** (zero PID gains → λ pinned; Q_c still trains;
>   `dual_select` uses the constant) via `scripts/run_fixed_lambda.sh` +
>   `scripts/dispatch_fixed_lambda.sh` — retrain 3 seeds (~6–8h/cell) to test robust
>   reproduction and decouple λ from seed (definitive causality). Outcomes: all-3
>   basin → robust ROW3 (proactive policy resolves the filter's limitation); mixed →
>   residual variance; none → realism-spectrum pivot (E5).
> - **Filter side unchanged/banked:** reactive SVF is an ISO-SSM velocity backstop
>   with a freeze-vs-flee dilemma; ~42% of proximity is exogenous. Report-feeding
>   write-up (now incl. the policy result): [docs/filter_fallback_findings.md](filter_fallback_findings.md)
>   §7. Post-fix pipeline (sweep→pick→pool→hybrid):
>   [docs/POST_CONFIRM_ROW3_RUNBOOK.md](POST_CONFIRM_ROW3_RUNBOOK.md).
> - **New tooling (all on `phase3`):** `pick_best_snapshot.py --by safety` (e41f1dc);
>   `analyze_row3.py pick|aggregate` + `run_basin_sweep.sh` + `plot_proximity_basin.py`
>   (7e9d118); `run_fixed_lambda.sh` (f39e910/9dac154); `dispatch_fixed_lambda.sh`
>   (1dd132d); `plot_basin_multiseed.py` (ece0821). Tests: `test_pick_safe_snapshot.py`,
>   `test_analyze_row3.py`. Figures: `results/figs/d0p3_basin_seed0.png`,
>   `d0p3_3seed_lambda_regimes.png`.

> **2026-06-05 — RESULT LOCKED: fixed-λ=0.1 is the robust graceful ROW3.** Fixed-λ
> worked as a *stabilizer* (all 3 seeds consistent — the 0/0.27/3.86 chaos gone), and
> λ is a clean proximity–success **Pareto knob**: λ=0.27 over-constrains (succ ~0.6),
> **λ=0.1 hits the graceful point — proximity 0.228 [0.194, 0.264] (−22.8%) at success
> 0.76, robust across 3 seeds (180 ep)**, ssm-actual −23%, velocity +8%. Each seed's
> operating point is below baseline; magnitude is bounded by the ~42% exogenous floor
> (captures ~40% of the reducible proximity) — modest, but the reactive filter cut ~0%
> at acceptable cost. **Thesis headline: reactive ISO-SSM filtering is fundamentally
> limited; the proactive constrained-RL policy resolves it, reproducibly.** Figure
> `results/figs/fixlam0p1_3seed.png`. ROW3 snapshots `exp_local/fixed_lambda/fixlam_0p1/
> lam0p1_seed{0,1,2}` @ steps 30546/8225/32696. Launchers `run_fixed_lambda.sh` /
> `dispatch_fixed_lambda.sh`. (Saucepan training is done; E5.3 below adds cross-task training.)

> **E5 — generalization & robustness (2026-06-05).**
> - **E5.1 tail risk — DONE.** Worst-case separation is exogenous-floor-bound: ROW3
>   improves *mean* separation (baseline 0.132 → 0.140 m) but the **tail is unchanged**
>   across baseline / no-shaping / ROW3 (CVaR₅ ≈ 0.005, worst ≈ 0.002 m) — the human's
>   closest lunge is human-driven. Reinforces the thesis: the policy controls the
>   *frequency / dwell / mean* of proximity, not the single worst-case approach. Computed
>   from the benchmark `ep_min_separation` episode data; `aggregate_e5_1.py` formalises it.
> - **E5.2 OOD (coworker generalization) — running.** Whole E4.1 table re-evaluated on the
>   held-out `coworker_eval` disruption (broader/gentler: closest 0.6–1.8, reach 3–9 s,
>   walk 0.6–2.2 vs train's 0.6–0.95 / 1.3–2.2 / 1.0–1.4). Tests whether ROW3's proximity
>   reduction survives an unseen coworker. `run_e4_1_headline.sh DISRUPTION=coworker_eval`.
> - **E5.3 cross-task generalization — running.** Replicate the full policy pipeline (base
>   curriculum → fixed-λ=0.1 Lagrangian → eval) on **`drawers_open_all`** and
>   **`dishwasher_close`**, producing baseline-vs-Lagrangian proximity rows per task. Tests
>   whether the proactive-avoidance result holds **across manipulation tasks**, not just
>   `saucepan_to_hob`. Launched via `dispatch_task_pipeline.sh` (one GPU lane per task,
>   `TASK=drawers_open_all GPUS="0 1 2"` / `TASK=dishwasher_close GPUS="3 4 5"`); auto-runs
>   `run_task_eval.sh` → `results/figs/<task>_lam0p1_3seed.png` + pooled ROW3. **Caveat:**
>   λ=0.1 is the saucepan operating point; if a task's eval shows it over/under-constrains
>   (success <0.75 or no proximity cut), re-run Phase 2 with `LAMBDA=0.05`/`0.15`. ~1.5–2
>   days/task. Tooling task-aware as of `e95ea8b`.

> **2026-06-06 — ALL eval experiments DONE (E5.3 tasks still training). Key new
> finding: the HYBRID IS COUNTERPRODUCTIVE.** Draft write-up:
> `docs/results_discussion_draft.md`.
> - **Saucepan E4.1 table complete** (`results/e4_1/e4_1_saucepan_to_hob_noisy_20260605_213348`):
>   row1 0.85/0.296; **row3 Lagrangian λ=0.1 (pooled) 0.76/0.228 (−23%)** [seed0 0.198,
>   −33%]; row4 baseline+filter 0.78/0.303 (+2%, 15% interv — velocity backstop only);
>   **row5 hybrid 0.62/0.265, 48% interv — WORSE than the policy alone.**
> - **Hybrid over-vetoes (48% vs 15% on baseline) → freeze → proximity ↑, success ↓.**
>   The SVF critic was trained on *baseline* rollouts → OOD on the avoiding policy.
>   **E4.3** corroborates (`results/e4_3/...`: intervention stays ~40–50%, no
>   internalisation fall). **Conclusion: the proactive policy makes the
>   baseline-calibrated reactive filter redundant/counterproductive — proactive ≫
>   reactive.** **R-sweep DONE** (`results/e4_1/hybrid_Rsweep`): the filter is harmful
>   at EVERY threshold — R∈[1.0,2.25] gives 39–48% interv and prox 0.27–0.28 (vs policy
>   0.198); lowering R barely changes interv → the baseline critic is OOD-miscalibrated
>   on the avoiding policy, not mis-thresholded. **On-policy SVF re-collect DONE
>   (`VER=v3_onpolicy`) — it does NOT salvage the filter either:** the deployment-faithful
>   sweep still has NO graceful knee (≤25% interv every R raises proximity — R=1.0
>   7.8%→0.264 vs policy 0.189; reduction only at 65–100% interv, robot frozen). So
>   *both* baseline-trained AND on-policy critics fail at every threshold → the limit is
>   the **reactive paradigm** (stop=freeze/dwell, retreat=flee/abandon), not the critic.
>   **Strongest result: no reactive learned filter, however calibrated, gracefully cuts
>   proximity vs an approaching coworker; proactive constrained-RL is necessary.** Do NOT
>   pin v3_onpolicy in snapshots.py (no operating point) — report as a definitive negative.
>   Sweep `results/svf_sweep_g1_0p3_v3_onpolicy/`. Draft §3.1.
> - **Model-based CBF filter (`--safety-filter cbf`, commit `78b36c3`, 18 unit tests pass)
>   DONE — it WORKS but FLEES.** A geometric directional-dodge filter (no critic; pushes
>   the floating-base target away from the human when sep<d_target, base-XY only) reduces
>   proximity 0.198→**0.150** (d_target=0.35) at just **1.3% intervention** — where the
>   learned SVF veto failed (48%, harmful). BUT it cuts proximity by *retreating from the
>   task*: success 0.75→0.57/0.43/0.35 as d_target=0.35/0.45/0.55; proximity floors ~0.14
>   (exogenous). Same frontier as a tighter-λ policy (λ=0.27≈0.6/0.17) → doesn't beat the
>   policy. **TAXONOMY: learned veto = FREEZE horn (no reduction, dwell); model-based dodge
>   = FLEE horn (reduction at success cost); proactive policy = the only graceful point.
>   Reactive filtering, learned OR model-based, is bounded by freeze-vs-flee.** Results
>   `results/e4_1/hybrid_cbf_d0p{35,45,55}.csv`; `filters/cbf_filter.py`; draft §3.2.
> - **(2026-06-07) Base-CBF full d_target sweep + EE-retract variant.** Full base-CBF sweep
>   (d=0.30→0.55) confirms a *single flee Pareto entirely below the policy*; even d=0.30
>   (1.1% interv, dodging only when already in violation) costs 0.75→0.58 (prox 0.170) — **no
>   harmless-backstop setting; the base dodge is brittle even at minimal intervention.**
>   **EE-retract "flinch" filter committed** (`e436a9c`, merged `606b05e`): `--cbf-mode ee`,
>   `CBFRetractFilter`, real `mj_jacBody` arm-Jacobian retract (base stays planted), 36 unit
>   tests pass. *Gotcha:* the arm action = the 10 `robot.limb_actuators` joints (action index
>   `floating_base.dof_amount + i`), which EXCLUDES the 2 grippers — a naive "indices 4..15"
>   would wrongly grab grippers. EE-mode needs live-env MuJoCo access (wrapper-walk to
>   `SafetyBiGymEnv._mojo`); guarded to pass-through if unavailable, so `filter_intervention_rate>0`
>   confirms the Jacobian path fires. **EE-retract sweep DONE — it does NOT stop the flee** (`hybrid_cbf_ee_d0p{35,45,55}`,
>   interv 5.8/9.0/13.2% so the Jacobian fires): in the high-success regime it's *worse* than
>   the base dodge (d=0.35: 0.50 vs 0.57 succ at ~same prox — the EE is the closest pair so it
>   fires more and retracting the arm pauses the task); it only reaches lower proximity
>   (0.108 @ d=0.55, below the exogenous estimate) by abandoning the task (0.38 succ). Neither
>   reactive variant (base-dodge OR arm-flinch) nears the policy (0.75/0.198) → **freeze-vs-flee
>   is intrinsic to reactive control regardless of which robot part dodges; only the proactive
>   policy is graceful.** Draft §3.3 filled; figure `results/figs/filter_taxonomy.png` regenerated.
> - **E5.2 OOD (`coworker_eval`) DONE** (`results/e4_1/e5_2_ood`): the reduction
>   generalises — baseline 0.084 → Lagrangian 0.058 (−31%); higher success cost
>   (0.92→0.70 — λ=0.1 over-constrains the gentler held-out coworker). Hybrid again worse.
> - **E3.6 perception gap DONE:** ROW3 oracle 0.236 ≈ noisy 0.198 — avoidance robust to
>   perception noise/lag, NOT bottlenecked. `--obs-mode off` is architecturally N/A (the
>   CQN-AS policy is trained with the human channel; low-dim **288→264** mismatch → that
>   condition is the E1.1 BC ablation, a separate no-human-obs policy).
> - **E5.1 tail risk DONE:** worst-case separation exogenous-floor-bound (CVaR₅≈0.005,
>   worst≈0.002 m, unchanged across configs).
> - **(2026-06-08) Speed-scaling filter DONE — the filter taxonomy is complete.**
>   `SpeedScaleFilter` (`40b203e`, 54 tests): `--safety-filter speedscale`, graded
>   `scale = clip((sep−d_stop)/(d_slow−d_stop), 0, 1)` on every joint (base dims scale the
>   *delta* — BiGym's floating base is always delta-driven). Baseline sweep
>   `d_slow ∈ {0.25..0.60}` @ noisy, 3 seeds×20 ep (`results/e4_1/speedscale_base_ds*.csv`):
>   **it is the ONLY filter that meaningfully cuts any ISO axis on its own terms** —
>   `ssm_violation_actual` 0.146 → **0.048 (−67%)** at the optimum `d_slow=0.40` (U-shaped in
>   d_slow; the SVF *veto* left it at 0.148 ≈ baseline → graded beats binary). vel 0.289→0.229;
>   proximity unchanged (≈0.27, the policy's axis). **BUT the success floor (~0.50–0.57) is
>   d_slow-INVARIANT** — even d_slow=0.25 intervenes 37% (persistent co-location) and costs
>   −33% succ ⇒ speed-scaling is **"freeze-lite"**: genuinely useful on the velocity axis but
>   still pays the reactive success cost; only the policy (anticipation) escapes. Confirms the
>   **division of labour** (policy→proximity, filter→velocity). Draft §3.4 + §5.1 future-work
>   filled. **Filter program complete — all reactive modalities (veto / base-dodge /
>   arm-flinch / speed-scale) exhausted; none is graceful; the proactive policy is the only
>   graceful frontier.**
> - **(2026-06-08) Policy+speed-scaling HYBRID benchmarked — joint-coverage capstone.**
>   `results/e4_1/hybrid_speedscale_d0p{30,40}_seed{0,1,2}` (pooled 180 ep via
>   `analyze_row3.py aggregate`). The hybrid is the ONLY config that cuts BOTH ISO axes at
>   once: proximity **0.250** (retained from policy 0.228, CIs overlap) AND ssm-actual
>   **0.065** (−41% vs policy 0.112 — the filter's add). Each component leaves the OTHER axis
>   ≈baseline (policy ssmA 0.112; filter-alone prox 0.273 ≈ baseline 0.296). Strictly beats the
>   SVF-veto hybrid (which degraded prox to 0.265, no velocity gain) → a complementary-axis
>   MODEL-BASED filter composes additively where a redundant-axis OOD LEARNED filter destroyed.
>   **Cost: success 0.85→0.44 — proactive (×0.89) and reactive (×0.58) costs STACK
>   multiplicatively; the hybrid beats NEITHER specialist on its own axis (policy wins prox
>   0.228<0.250; filter wins velocity 0.048<0.065).** Honest verdict: NOT a free lunch, but the
>   only route to both-axis (proximity AND velocity) ISO compliance, with a λ/d_slow cost dial.
>   Figures `results/figs/joint_coverage.png` (both-axes scatter — capstone) + `velocity_axis.png`;
>   draft §3.4 capstone filled. **This is the Hybrid Safety Critic's actual contribution —
>   saucepan experiments fully complete.**

---

## Confirmed decision points (round 3)

1. **Coworker embodiment: Unitree G1 only.** SMPL-H is cut from the
   main pipeline and moved to future-work
   (Section~\ref{sec:fw:embodiment}). All references to "SMPL-H
   curriculum" in earlier docs are obsolete.
2. **Lagrangian architecture: B-value-mean.** Single-critic Option-A
   is a prototype only; B-value-CVaR is future work
   (Section~\ref{sec:fw:cvar}).
3. **Primary task: `saucepan_to_hob`.** Long-horizon, workspace
   overlap with G1 coworker, highest unconstrained-baseline
   `success_rate` of the four Phase 0 tasks. Optional secondary:
   `drawers_open_all` (only if compute permits).
4. **External baseline: WCSAC.** With Safety-Gym Lagrangian as
   fallback if reimplementation is fragile (decision rule in
   §disc:wcsac-honest of the report).
5. **Page limit: 60 A4 pages of content** (between contents and
   reference list, exclusive). Limit, not target.

---

## What's done (✅)

| Phase | Done | Notes |
|---|---|---|
| Phase 0 | ✅ ACT baseline on 4 tasks | `saucepan_to_hob`, `drawers_open_all`, `dishwasher_close`, `reach_target_single` |
| Phase 1 | ✅ E1.1 obs-ablation (BC, no penalty) | Negative result — channel useless under BC. Reported as load-bearing motivation |
| Phase 2 (SMPL-H) | ✅ SVF dataset + CQL training + filter wrapper + sweeps | $\alpha_{\rm CQL}=5.0$, $R=4.0$. Did **not** transfer to G1 |
| Phase 2 (G1) | ⚠ **RE-DO v3 (2026-06-01)** — 2 train/deploy bugs fixed | Bug1 action de-norm (`41fd93b`) + Bug2 execution-mode (snapshot policy ran receding-horizon `chunk[0]`, deploy runs ensemble blend → E4.1 row-4 89.5% intervention). Both fixed in `_CQNASSnapshotPolicy`; **v2/R=2.50 superseded**, v3 re-collect pending (`run_p2_recollect_g1.sh`). See status-delta |
| Adapter | ✅ CQN-AS vendor integration | 8 bugs fixed and documented in `cqn_as_integration_notes.md` |
| Phase 3 (cost forms) | ✅ P3.0/P3.1 smoke + **all 3 E3.1 cost forms wired** | continuous / binary (`cost_form`) / fixed (`add_violation_penalty`); 31 cost tests pass |
| G1 swap + **P1 curriculum** | ✅ Implemented, smoked, **curriculum run** | Stage-2 G1 baseline snapshot in hand (row-1 reference) |
| P6 harness | ✅ `benchmark_policy.py` built + tested + validated on a real CQN-AS snapshot | See **P6 below** (was pending; now DONE). Docs: `docs/benchmark_harness.md` |

---

## Priority 1 — mandatory headline (must run for top-mark thesis)

These are the experiments needed to populate `report.tex` headline
tables and figures. Approximate GPU budget: ~70 A100-hours total.

### P1. G1 base-policy curriculum (stages 0/1/2) on `saucepan_to_hob` — ✅ DONE (2026-05-30)
- **Status**: ran via `scripts/run_base_curriculum.sh` (`HUMAN_MODEL=g1`,
  3 stages idle→easy→`coworker_train`, warm-start chained). Snapshots
  (run `base_g1_30k_30k_40k_20260529_124749`, recorded in
  `filters/snapshots.py::G1_CURRICULUM`):
  - **stage 1** (warm-start for P3/P4):
    `exp_local/cqn_as_base_curriculum/base_g1_30k_30k_40k_20260529_124749/stage1_easy/snapshot_2588.pt`
  - **stage 2** (unconstrained baseline; P5 row-1/row-4 eval + P2 baseline sweep):
    `exp_local/cqn_as_base_curriculum/base_g1_30k_30k_40k_20260529_124749/stage2_full/snapshot_28203.pt`
- **Goal**: produce the unconstrained baseline snapshot used as the
  starting point for everything downstream + as row 1 of the
  feature-incremental Table~\ref{tab:e4.1-feature-incremental}.
- **Acceptance**: end-of-stage-2 `ep_reward` ≥ +1.5; `success_rate`
  ≥ 0.5 averaged over last 20 eval episodes; no value-support
  saturation (`ep_reward` monotone non-decreasing across stages).
- **Cmd** (illustrative sketch — the real run used `scripts/run_base_curriculum.sh`;
  real keys are `env=safety_bigym/<task>` and `num_train_frames=`, not `task=`/`frames=`):
  ```bash
  python train_cqn_as.py task=saucepan_to_hob \
    disruption=coworker_idle  bodyslam=oracle frames=20000 +snapshot_path=null
  python train_cqn_as.py task=saucepan_to_hob \
    disruption=coworker_easy  bodyslam=oracle frames=15000 +snapshot_path=stage0_final.pt
  python train_cqn_as.py task=saucepan_to_hob \
    disruption=coworker_train bodyslam=oracle frames=60000 +snapshot_path=stage1_final.pt
  ```
- **Populates**: §results:baseline (Table~\ref{tab:results:baseline}),
  warm-start for P3 and P5.
- **GPU**: ~20 h
- **Perception mode**: train + eval `oracle` (the baseline is the methodological reference; see Perception Mode Policy in PROJECT_PLAN.md)

### P2. Phase 2 SVF re-eval + retrain under G1 — ⚠ RE-DO v3 (2026-06-01: two train/deploy bugs)
> **2026-06-01:** the SVF critic had **two** bugs where collection didn't match
> deployment (see the dated status-delta at the top for the full narrative):
> **(1) action de-norm** — `env.action_space` instead of demo stats (fixed
> `41fd93b`, v2 re-collect); **(2) action-execution mode** — `_CQNASSnapshotPolicy`
> collected receding-horizon `chunk[0]` while deployment runs an `action_sequence=16`
> + `temporal_ensemble=true` blend, so the v2 critic over-vetoed at deploy (E4.1
> row-4: 89.5% intervention, `mean_q` 0.97, success 0, vs the v2 sweep's 7.9%/3.3
> — the sweep was over-optimistic because it reuses the collection path). Bug 2 is
> now fixed (snapshot policy mirrors `CQNASRunner.step`, reuses the same
> `TemporalEnsembleControl`; `tests/test_snapshot_policy_execution.py`). **v2/R=2.50
> and the §0 v2 numbers are SUPERSEDED**; a v3 re-collect is pending. Everything
> below is kept for method/provenance only.
- **Done (on the buggy critic — provenance only)**: (1) confirmed the old `svf_coworker_train_v1.pt` over-fires on G1;
  (2) recollected on `coworker_train`, `bodyslam=noisy` (random + snapshot,
  105k transitions); (3) retrained at **τ=0.3 m** → `svf_coworker_train_g1_0p3.pt`
  (3-MLP [256,256,256], α_CQL=5.0, τ_polyak=5e-3, 200k steps); (4) ran the
  **dense 0.3 m sweep** (R=0 baseline + fine grid, 3 seeds × 20 ep,
  `sweep_dense_seed{0,1,2}.csv`). **Authoritative write-up: `phase2_results.md` §0.**
- **Seed-averaged dense sweep** (filterless baseline at R=0):

  | R | intervention | proximity (τ=0.3) | reduction vs R=0 |
  |---|---|---|---|
  | 0.0 | 0% | 0.0435 | baseline |
  | **2.25** | **21.6%** | **0.0297** | **31.7%** ✅ |
  | 2.5 | 34.3% | 0.0265 | 39.1% (interv >25%) |
  | 3.0 | 78.5% | 0.0076 | 82.5% (hard gate, ~frozen) |

- **Operating point R = 2.25** (pinned in `snapshots.py`): the only threshold
  meeting the bar (**≥30% reduction at ≤25% intervention**). **Marginal &
  seed-fragile** (per-seed 38.4 / 41.2 / 20.6%); low-R interventions (≤2.0) are
  wasted (~0% gain); the 82% win costs ~79% intervention. The filter's robust
  win is the ISO-SSM (robot-velocity) axis → core hybrid argument. Provisional —
  re-confirm against the Phase-3 row-3 snapshot in P5.
- **NB**: the coarse `sweep_seed{0,1,2}.csv` (R={1,2,3,4,5,6,8}) was the OLD
  0.5-label critic and is **not comparable** — use the dense CSVs.
- **Populates**: §results:filter-pareto, row 4 of E4.1 table.
- **Perception mode**: train + eval **`noisy`** (the filter's deployment
  distribution; see Perception Mode Policy in PROJECT_PLAN.md, Rationale).

### P3. E3.1: cost-signal form ablation (continuous vs binary vs fixed)
- **Goal**: validate the load-bearing claim that continuous cost
  dominates binary. Three cells, 3 seeds each.
- **✅ Cost-form selector LANDED (2026-05-30)** — all three cells wired:
  1. **`filters/cost_signal.py`** — new `select_cost(safety_info, cost_form=...)`
     dispatches `continuous` → `compute_cost` (graded [0,1]) / `binary` →
     `1[ssm_violation]`; exported `COST_FORMS=("continuous","binary")`.
  2. **`agents/cqn_as/env_adapter.py`** — reads `env.safety.cost_form`
     (default `continuous`, validated) and calls `select_cost` at the per-step
     cost site.
  3. **`cfgs/env/safety_bigym.yaml`** — declares `cost_form: continuous` so
     `env.safety.cost_form=binary` overrides without `+`.
  4. **`fixed` needed NO new code** — it reuses the pre-existing, factory-threaded
     `env.safety.add_violation_penalty`/`violation_penalty=0.05` reward penalty
     under plain `agent=cqn_as` (Lagrangian off).
  5. **`tests/test_cost_signal.py`** — +8 `select_cost` tests (31 cost-path tests
     pass). Hydra composition verified for all three cells.
  `scripts/run_e3_1_cost_signal.sh` launches the full 3×3 matrix.
- **Corrected launch surface** (the plan's `task=`/`frames=`/`cost_signal=` are
  sketches — real keys verified against `train_cqn_as.py` + `cqn_as_config.yaml`):
  ```bash
  # Full 3×3 matrix — via scripts/run_e3_1_cost_signal.sh (all forms wired):
  WARMSTART=exp_local/.../stage1_easy/snapshot_XXXXX.pt \
    scripts/run_e3_1_cost_signal.sh           # {fixed,binary,continuous} x seeds {0,1,2}
  # which runs, per cell (continuous shown; binary adds env.safety.cost_form=binary;
  # fixed uses agent=cqn_as + env.safety.add_violation_penalty=true violation_penalty=0.05):
  python train_cqn_as.py env=safety_bigym/saucepan_to_hob \
    disruption=coworker_train bodyslam=oracle num_train_frames=60000 \
    agent=cqn_as_lagrangian agent.cost_budget=0.01 \
    num_demos=36 env.safety.add_workspace_penalty=true \
    agent.v_min=-6.0 agent.v_max=2.0 agent.atoms=101 \
    seed=0 +snapshot_path=$WARMSTART
  ```
  **Warm-start from the P1 stage-1 snapshot** (matches row-1's protocol: 60k on
  `coworker_train` from the same start, differing only by the Lagrangian).
- **Acceptance**: continuous row beats binary row on
  `ep_proximity_violation_rate` with non-overlapping 95% bootstrap
  CIs. If overlap, this is itself a finding to report honestly.
- **Populates**: Table~\ref{tab:e3.1-cost-signal}.
- **GPU**: 9 cells × ~2 h = ~18 h (all 3 forms now runnable)
- **Perception mode**: train + eval both `oracle` (isolates the cost-signal variable; see Perception Mode Policy in PROJECT_PLAN.md)

### P4. E3.2: cost-budget Pareto sweep
- **Goal**: identify the headline $d$ operating point as the knee.
- **Cells**: $d \in \{0.001, 0.01, 0.05, 0.1\}$, 3 seeds.
- **Launcher**: `scripts/run_e3_2_cost_budget.sh` (built 2026-05-30) — sweeps
  `agent.cost_budget` over continuous-cost Lagrangian runs, warm-started from the
  P1 stage-1 snapshot (`WARMSTART`). `SMOKE=1` for a 2000-frame composition check.
- **Independent of the P2 sweep** — warm-starts from stage-1, not the SVF filter.
- **Analysis**: `scripts/analyze_e3.py --in-dir exp_local/e3_2_cost_budget/<run>`
  (built 2026-05-31) — seed-averaged success/proximity per `d` + a suggested
  `d_knee`. Works mid-run (reads `metrics.jsonl` if `final_metrics.json` absent);
  also summarises E3.1 (`exp_local/e3_1_cost_signal/<run>`, no knee line).
- **Populates**: Figure~\ref{fig:e3.2-pareto}.
- **GPU**: 12 cells × ~2 h = ~24 h

### P5. E4.1: feature-incremental headline (THE HEADLINE TABLE)
- **Goal**: populate Table~\ref{tab:e4.1-feature-incremental}, the
  central thesis result.
- **5 rows × 3 seeds × 20 eval episodes = 15 training runs + 1 deploy eval**:
  - **Row 1** — Unconstrained baseline. **Already produced by P1.**
  - **Row 2** — + workspace shaping. Re-train P1 with workspace
    shaping enabled (already in the codebase). New runs.
  - **Row 3** — + Lagrangian (continuous cost). Re-use the P3
    `cost_signal=continuous` cells at the P4-winning $d$.
  - **Row 4** — Baseline (row 1) + runtime filter. **Pure deploy
    eval**, no training. Use the `benchmark_policy.py` harness on
    the P1 snapshot wrapped with the P2 SVF.
  - **Row 5** — Full hybrid (row 3 + filter). Pure deploy eval on
    the P3 snapshot wrapped with the P2 SVF.
- **Perception mode**: **whole table on `noisy`** (`OBS_MODE=noisy`, the driver
  default) — see the finding below; `OBS_MODE=oracle` is a policy-only reference.
- **Driver**: `scripts/run_e4_1_headline.sh` (built 2026-05-30) — runs
  `benchmark_policy.py` for every row on one obs mode, one CSV per row.
  **Incremental**: rows 1 & 4 run now from `STAGE2` (P1 stage-2) + the SVF
  filter; rows 3/5 skip until `ROW3` (the P3 d_knee snapshot) is set; row 2 skips
  until `ROW2` is set. Defaults: `STAGE2`→recorded G1 stage-2,
  `SVF_FILTER`→`svf_coworker_train_g1_0p3.pt`, `FILTER_R`→read from
  `snapshots.py` (R=2.25). `RENDER=1` (`RENDER_EPISODES=N`) writes per-row
  rollout mp4(s) to `<OUTDIR>/<label>_videos/`.
- **⚠ Finding (2026-05-30 → re-diagnosed 2026-06-01)**: the filtered rows showed
  `mean_q ≈ 0.02 ≪ R=2.25` → **~100% intervention, robot frozen, success → 0.0**.
  First read as "oracle-collapse" (filter OOD on oracle) and the table moved to
  **noisy** — but the **noisy** re-run (results/e4_1/..._221340) showed the SAME
  100% collapse, **disproving** that diagnosis. Real cause: the SVF
  snapshot-collection policy de-normalised actions via `env.action_space`, so the
  critic was trained against a mis-scaled policy and over-vetoes at deploy
  regardless of perception mode (de-norm bug, fixed `41fd93b`; see status-delta).
  **Filtered rows (4, 5) are blocked on the P2 re-do.** E4.1 still evals on
  **noisy** — but for the original correct reason (the filter is noisy-trained, so
  noisy = its deployment distribution), not "oracle-collapse." Rows 1–3
  (policy-only) are unaffected by the bug.
- **Acceptance**: row 5 dominates each of rows 1–4 on
  `ep_proximity_violation_rate` with non-overlapping CIs. If row 5
  ties row 3, the filter is redundant on a well-trained policy —
  still publishable but flagged in §disc:rqs-revisited.
- **GPU**: Row 2 = ~6 h (3 seeds). Rows 4 + 5 are pure eval (~0.5 h
  total). Total marginal: ~6.5 h.

### P6. **Snapshot-evaluation benchmark harness** (`benchmark_policy.py`) — ✅ DONE (2026-05-30)
- **Status**: built, unit-tested (8 tests, `tests/test_benchmark_harness.py`),
  and validated end-to-end on the **real CQN-AS snapshot** `snapshot_17826.pt`
  (saucepan_to_hob/G1), filter off and on. Usage doc:
  [`docs/benchmark_harness.md`](benchmark_harness.md).
  - **Code**: CLI `scripts/benchmark_policy.py` + package
    `safety_bigym/benchmark/` (`stats`, `records`, `schema`, `aggregate`,
    `env_build`, `filter_attach`, `runners`, `loader`) +
    `scripts/benchmark_visualize.py` (Pareto / bars / separation) +
    `scripts/benchmark_demo.sh`.
  - **Validated paths**: random+G1, random+SVF-filter+G1 (local
    `svf_coworker_train_v1.pt`), **CQN-AS real snapshot**, **CQN-AS + in-loop
    SVF veto**. ACT path: loader dispatch unit-tested; full run reuses the
    already-tested `svf_collect_dataset.load_snapshot_policy` (no ACT snapshot
    available locally).
  - **Schema deviations from the round-4 spec (all documented in
    `benchmark_harness.md`)**: raw rolls persisted as **parquet** (`pandas`+
    `pyarrow` added to `setup.py`) plus a JSONL sidecar — not the originally
    sketched parquet-only; `success` uses `info["task_success"]` (matches
    `train_cqn_as`) with cumulative-reward>0 fallback; one appended **CSV row
    per invocation** aggregating all seeds×episodes (the `seeds` column +
    bootstrap CIs over the pooled rolls), not one row per seed.
  - **Two portability fixes landed during validation**: (1) `build_cqn_cfg`
    rebases the snapshot's baked GPU `motion_clip_dir` onto the local
    `AMASS_DATA_DIR` (snapshots portable across machines); (2)
    `--num-demos-for-stats` caps the demo count for the CQN-AS action-stat
    step (the full 36-demo pixel load OOM-kills a laptop; use the full count
    on the GPU box).
  - **Finding for P2**: at $R=4.0$ the SVF filter over-fires on the G1 policy
    (`mean_q_value ≈ 0.31 ≪ 4.0` → ~100% intervention). The harness surfaces
    `mean_q_value`, making this diagnosable. Reinforces the P2 "re-calibrate
    $R$ under G1" task — sweep $R$ before locking E4.1 row-4/row-5 numbers.
- **Goal**: a single CLI that, given any policy checkpoint, produces
  a CSV row per (task, disruption, obs-mode, seed) cell with the
  full safety-metric schema. **This is the load-bearing piece of
  benchmark deliverable C1** (§bench:harness).
- **Acceptance**: smoke test (`--smoke` flag, CPU, < 5 min) runs
  end-to-end and produces a non-empty CSV with the documented schema.
  ✅ Met — `--smoke` finishes in ~9 s. (Implemented to use a **random
  policy** when no `--snapshot` is given, since no Phase-0 ACT snapshot
  exists on this machine; it uses whatever snapshot is passed otherwise.)
  Used as the canonical data source for every results table in the report.
- **CSV columns must include** (round-3 + round-4 schema):
  - **Safety**: `ep_proximity_violation_rate`, `ep_ssm_violation_rate`,
    `ep_ssm_violation_actual_rate`, `ep_min_separation`,
    $\cvar_{0.95}$ `ep_cost_integral`, $\cvar_{0.95}$ `ep_min_separation`,
    p99 `ep_min_separation`, per-region PFL counts (currently inert).
  - **Task**: `success_rate`, `episode_reward`,
    **`steps_to_completion`** (mean env-steps to `terminal=True`,
    among successful episodes; populates the `steps` column of
    Table~\ref{tab:e4.1-feature-incremental}). Trivial to add — env
    already returns `terminal` and the harness already counts
    steps.
  - **Filter mechanics** (only if a filter is wrapped around the
    snapshot): `filter_intervention_rate`,
    **`filter_passthrough_rate`** ($= 1 -$ intervention; included
    explicitly for the headline table's Passth.\ column),
    `mean_per_episode_interventions`. The `--filter-snapshot
    path/to/svf.pt` CLI flag toggles filter wrapping.
- **Cmd**:
  ```bash
  python scripts/benchmark_policy.py \
    --snapshot path/to/policy.pt \
    --filter-snapshot path/to/svf.pt \   # optional
    --task saucepan_to_hob \
    --disruption coworker_eval \
    --obs-mode noisy \                   # noisy for E4.1 headline (filter is noisy-native; oracle collapses it); oracle = policy-only reference — see Perception Mode Policy in PROJECT_PLAN.md
    --seeds 0,1,2 \
    --episodes 20 \
    --out results/cell.csv
  ```
- **Effort**: ~2 days engineering. **Prerequisite for P5.**
- **GPU**: minimal (eval-only, CPU OK)

---

## Priority 2 — strengthens thesis (run if time permits)

### P7. E3.6: obs-channel ablation under constrained policy
- **Goal**: closes RQ1 (off / oracle / noisy on top of B-value-mean
  with continuous cost). Resolves the E1.1 ambiguity.
- **Populates**: Table~\ref{tab:e3.6-obs-rl}.
- **GPU**: ~7 h — 3 trained policies on `bodyslam=oracle` (~6 h) + 9 eval cells across 3 modes via the harness (<1 h)
- **Perception mode**: train on `oracle`; eval sweeps `off / oracle / noisy` (this experiment's whole purpose is to *measure* the perception gap; see Perception Mode Policy in PROJECT_PLAN.md)

### P8. E4.3: filter internalisation curve — ✅ POST-HOC SCRIPT (2026-05-31)
- **Goal**: produce the internalisation curve
  (Figure~\ref{fig:e4.3-internalisation}) — direct evidence that
  policy and filter are complementary (filter intervention rate falls as the
  Lagrangian policy is trained).
- **⚠ The "free during training" hook does NOT work** — P3/P4 train on `oracle`,
  and the SVF filter's Q collapses on oracle obs (100% would-be intervention; the
  same finding that moved E4.1 to noisy). So the in-training `FILTER_PASSIVE` hook
  (still in `train_cqn_as.py`, harmless, off by default) would log a flat ~100%
  curve — **don't use it.**
- **Built (the real path): `scripts/run_e4_3_internalisation.sh`** — POST-HOC,
  on **noisy**. Loops a P3/P4 training cell's saved `snapshot_<N>.pt` through
  `benchmark_policy.py --filter-snapshot ... --obs-mode noisy` and writes
  `internalisation_curve.csv` (frame, filter_intervention_rate, success_rate,
  ep_proximity_violation_rate). On noisy the filter is in-distribution and the
  rate is meaningful.
- **Run**: `RUN_DIR=exp_local/e3_2_cost_budget/<run>/d0pXX_seed0 bash
  scripts/run_e4_3_internalisation.sh` (`SMOKE=1` = newest snapshot only).
- **GPU**: **NOT free anymore** — ~few min per snapshot (≈6–7 per run → ~30 min),
  post-hoc eval. (The old "0 GPU, piggybacks on P3" estimate assumed the
  in-training hook, which the oracle-collapse killed.)

### P9. E3.7: WCSAC external baseline
- **STATUS (2026-06-09): IMPLEMENTED — `agent=wcsac` on `train_cqn_as.py`.**
  Faithful Worst-Case SAC (`safety_bigym/agents/wcsac/`, `cfgs/agent/wcsac.yaml`,
  `tests/test_wcsac_agent.py`, `scripts/wcsac_{smoke,train}.sh`). Unit-tested +
  env-smoke-verified; GPU sweep running, eval pending. Trains from scratch — the
  SAC architecture can't load the CQN-AS C2F warm-start snapshots.
- **Goal**: place our hybrid against the standard distributional
  safe-RL method. Honest-failure path documented in §disc:wcsac-honest.
- **Acceptance gate**: reimplementation matches Safety-Gym numbers
  within ±5% within 2 days of effort. If not, report as
  best-effort.
- **GPU**: ~10 h

### P10. E5.1 + E5.2: tail-risk + OOD generalisation
- **Goal**: populate Tables \ref{tab:e5.1-tail} and
  Figure~\ref{fig:e5.2-ood}. Mostly pure eval if P5 snapshots are
  already produced.
- **E5.1 (tail-risk)**: **zero new runs** — `benchmark_policy.py` already emits
  `cvar95_ep_cost_integral`, `cvar95_ep_min_separation`, `p99_ep_min_separation`
  in every P5 row CSV. Aggregate with `scripts/aggregate_e5_1.py --in-dir
  results/e4_1/<noisy_run_tag>` (built 2026-05-31).
- **E5.2 (OOD)**: re-run the P5 driver on the wider band — just flip the
  disruption:
  ```bash
  SVF_FILTER=$SVF STAGE2=$STAGE2 ROW3=$ROW3 DISRUPTION=coworker_eval \
    RUN_TAG=e5_2_ood bash scripts/run_e4_1_headline.sh
  ```
  Compare the train→eval degradation of row 1 vs row 5 (hybrid should degrade
  less). `coworker_eval.yaml` exists.
- **GPU**: ~2 h (eval-only, both bands)

---

## Priority 3 — explicitly CUT or moved to future work

Per round-3 feedback, the priority-3 backlog from earlier rounds is
**cut** except where retained below. Each cut decision is recorded
so an examiner can verify the scope was deliberately bounded.

| Was | Now |
|---|---|
| P14 — G1 coworker as headline embodiment | ✅ **Promoted to P1**; this *is* the headline. |
| P11 — Recovery RL comparison           | **Cut.** Future work §fw:recovery. |
| P12 — Filter during training ablation  | **Cut.** Future work §fw:filter-during-training. |
| P13 — Multi-task evaluation suite      | **Cut.** Future work, possible journal extension. |
| P15 — Real-robot sim-to-real           | **Cut.** Future work §fw:realbody. |
| P16 — Adversarial human disruption     | **Cut.** Future work, motivated by §disc:arch-caveat. |
| P17 — Original ``primary task = reach_target_single'' | **Replaced.** `saucepan_to_hob` is primary (per decision 3); `drawers_open_all` optional secondary. |
| P18 — SMPL-H curriculum                | **Cut.** Future work §fw:embodiment. |

---

## Carried-forward technical risks

These are surfaced in §disc:threats of the report and are not
blockers for the headline runs, but they qualify the claims:

1. **PFL contact-detection bug.** `data.ncon = 0` for human–robot
   pairs in BiGym's runtime robot attachment, so
   $c^{\rm PFL}_t \equiv 0$. All safety claims qualified as
   "geometric / SSM-only". Fix is 1–2 weeks of BiGym-internals work
   (§fw:pfl). **Schema is forward-compatible**: the labeller, cost
   signal, and runtime filter all read $c^{\rm PFL}$ unconditionally.

2. **C51 support saturation under shaped rewards.** Resolved
   (Theorem 5.1, §method:lagrangian:support): bounded shaping with
   $\beta = 0.05$, $c_{\rm ws} = 1.0\,$m, support widened to
   $[-6, +2]$. Check $\beta \cdot c_{\rm ws} / (1-\gamma) \le |v_{\min}|$
   before any new shaping change.

3. **Curriculum dependence.** Direct training on full `coworker_train`
   collapses. Documented as a limit (§disc:curriculum) and as
   standard practice (§method:curriculum).

4. **WCSAC reimplementation fragility.** Honest-failure path
   documented (§disc:wcsac-honest).

5. **Mock-BodySLAM++ ≠ real BodySLAM++.** Calibrated against the
   published characteristics of \cite{henning2023bodyslampp} but
   not identical. Rendered-image closure is future work (§fw:realbody).

---

## Smoke-gate checklist (run before every multi-hour launch)

```bash
# Phase 2 dataset
python -m safety_bigym.filters.dataset --smoke

# Phase 2 SVF training
python -m safety_bigym.filters.train --smoke

# Phase 2 sweep
python -m safety_bigym.filters.sweep --smoke

# Phase 3 train_cqn_as.py composition — `--smoke` does NOT exist (hydra rejects
# it); the real gate is a short num_train_frames run with no demos/W&B.
python train_cqn_as.py env=safety_bigym/saucepan_to_hob \
  num_train_frames=100 num_demos=0 wandb.use=false
# Lagrangian agent composition (the P3 path)
python train_cqn_as.py env=safety_bigym/saucepan_to_hob \
  agent=cqn_as_lagrangian num_train_frames=100 num_demos=0 wandb.use=false

# Snapshot eval harness (P6) — this one IS an argparse CLI with a real --smoke
python scripts/benchmark_policy.py --smoke
```

All gates < 5 min CPU. If any fail, do not launch the long run.

---

## Suggested execution order (assuming ~70 GPU-hours)

| Day | Tasks | Notes |
|---|---|---|
| 0 (today) | Smoke gates × all; P6 implementation begin | ~2 days dev |
| 1–2 | P6 finish + smoke; P1 stages 0+1 launch | P1 stage-0/1 = 35k frames ≈ 8 h |
| 3 | P1 stage-2 launch + monitor | 60k frames ≈ 12 h |
| 4 | P2 (SVF re-eval) + P3 launch | P3 = 9 cells, par |
| 5 | P3 finishes + P4 launch | P4 = 12 cells |
| 6 | P4 finishes + P5 row 2; P8 = post-hoc `run_e4_3_internalisation.sh` on a P3/P4 cell | |
| 7 | P5 rows 4+5 (eval only); P7, P10 (eval only) | |
| 8 | P9 (WCSAC) if compute remaining | |
| 9 | Buffer for re-runs, final eval pass | |
| 10–14 | Report writing: fill `\result{X}` markers; finalise figures | |

If P9 (WCSAC) cannot fit, report the comparison as future work; the
hybrid's positioning against Recovery RL and SHIELD via
Table~\ref{tab:related-work-comparison} is still defensible without
WCSAC numbers.

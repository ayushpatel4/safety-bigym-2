# IMPLEMENTATION_STATUS.md
### Coding-agent brief — `safety_bigym` MEng project
### Updated: 2026-05-28 (round 3)

This document is the **single source of truth for what needs running**.
It pairs with `REPORT_STRATEGY.md` (why) and `report.tex` (the deliverable).
Tasks are priority-ordered; smoke gates are mandatory before any
multi-hour launch.

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
| Phase 2 | ✅ SVF dataset + CQL training + filter wrapper + sweeps | $\alpha_{\rm CQL}=5.0$, $R=4.0$ operating point. Results discussed in §results:filter-pareto |
| Adapter | ✅ CQN-AS vendor integration | 8 bugs fixed and documented in `cqn_as_integration_notes.md` |
| Phase 3 | ✅ P3.0/P3.1 smoke (B-value-mean) — code only | Full curriculum/headline runs still pending |
| G1 swap | ✅ Implemented + unit-tested + smoked | Full curriculum reproduction is **P1 below** |
| P6 harness | ✅ `benchmark_policy.py` built + tested + validated on a real CQN-AS snapshot | See **P6 below** (was pending; now DONE). Docs: `docs/benchmark_harness.md` |

---

## Priority 1 — mandatory headline (must run for top-mark thesis)

These are the experiments needed to populate `report.tex` headline
tables and figures. Approximate GPU budget: ~70 A100-hours total.

### P1. G1 base-policy curriculum (stages 0/1/2) on `saucepan_to_hob`
- **Goal**: produce the unconstrained baseline snapshot used as the
  starting point for everything downstream + as row 1 of the
  feature-incremental Table~\ref{tab:e4.1-feature-incremental}.
- **Acceptance**: end-of-stage-2 `ep_reward` ≥ +1.5; `success_rate`
  ≥ 0.5 averaged over last 20 eval episodes; no value-support
  saturation (`ep_reward` monotone non-decreasing across stages).
- **Cmd**:
  ```bash
  python train_cqn_as.py task=saucepan_to_hob \
    disruption=coworker_idle  frames=20000 +snapshot_path=null
  python train_cqn_as.py task=saucepan_to_hob \
    disruption=coworker_easy  frames=15000 +snapshot_path=stage0_final.pt
  python train_cqn_as.py task=saucepan_to_hob \
    disruption=coworker_train frames=60000 +snapshot_path=stage1_final.pt
  ```
- **Populates**: §results:baseline (Table~\ref{tab:results:baseline}),
  warm-start for P3 and P5.
- **GPU**: ~20 h

### P2. Phase 2 SVF re-evaluation under G1 coworker
- **Goal**: confirm $R=4.0$, $\alpha_{\rm CQL}=5.0$ remain the
  operating points after the G1 swap, and produce filter-on-baseline
  numbers for Table~\ref{tab:e4.1-feature-incremental} row 4.
- **Acceptance**: filter alone reduces
  `ep_proximity_violation_rate` on the unconstrained baseline by
  ≥ 30% at intervention rate ≤ 20%. If the operating point shifts,
  re-run sweep E2.3.
- **Populates**: §results:filter-pareto, row 4 of E4.1 table.
- **GPU**: ~3 h

### P3. E3.1: cost-signal form ablation (continuous vs binary vs fixed)
- **Goal**: validate the load-bearing claim that continuous cost
  dominates binary. Three cells, 3 seeds each.
- **Acceptance**: continuous row beats binary row on
  `ep_proximity_violation_rate` with non-overlapping 95% bootstrap
  CIs. If overlap, this is itself a finding to report honestly.
- **Cmd template**:
  ```bash
  python train_cqn_as.py task=saucepan_to_hob \
    disruption=coworker_train frames=60000 \
    cost_signal={fixed,binary,continuous} seed={0,1,2} \
    +snapshot_path=stage2_final.pt
  ```
- **Populates**: Table~\ref{tab:e3.1-cost-signal}.
- **GPU**: 9 cells × ~2 h = ~18 h

### P4. E3.2: cost-budget Pareto sweep
- **Goal**: identify the headline $d$ operating point as the knee.
- **Cells**: $d \in \{0.001, 0.01, 0.05, 0.1\}$, 3 seeds.
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
    --obs-mode noisy \
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
- **GPU**: ~6 h

### P8. E4.3: filter intervention rate during training
- **Goal**: produce the internalisation curve
  (Figure~\ref{fig:e4.3-internalisation}) — direct evidence that
  policy and filter are complementary.
- **Implementation**: log `filter_intervention_rate` from the P3
  training runs at each eval cycle. **Costs nothing extra** if
  added to the P3 logging setup. Mark as "free" alongside P3.
- **GPU**: 0 (piggybacks on P3)

### P9. E3.7: WCSAC external baseline
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

# Phase 3 train_cqn_as.py (with B-value-mean)
python train_cqn_as.py --smoke

# Snapshot eval harness (P6)
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
| 6 | P4 finishes + P5 row 2; P8 piggyback baked in | |
| 7 | P5 rows 4+5 (eval only); P7, P10 (eval only) | |
| 8 | P9 (WCSAC) if compute remaining | |
| 9 | Buffer for re-runs, final eval pass | |
| 10–14 | Report writing: fill `\result{X}` markers; finalise figures | |

If P9 (WCSAC) cannot fit, report the comparison as future work; the
hybrid's positioning against Recovery RL and SHIELD via
Table~\ref{tab:related-work-comparison} is still defensible without
WCSAC numbers.

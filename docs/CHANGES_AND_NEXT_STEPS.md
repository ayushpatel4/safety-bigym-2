# Changes and Next Steps

Generated alongside the updates to `UPDATED_PROJECT_PLAN.md`, `HYBRID_SAFETY_CRITIC_PLAN.md`, and `report.tex`. This document is the change inventory + forward action list — a historical record of the 2026-05-15 plan rewrite.

> **For current status and the next action, read [.claude/IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) first** — it is the living doc. The TL;DR below is kept up to date; the file-by-file inventory further down is a frozen record of the plan rewrite.

---

## TL;DR — six structural changes (status as of 2026-05-27)

| Change | Status of corresponding code | Status of corresponding writing |
|---|---|---|
| **(1) Single COWORKER disruption** | DONE (3 trajectory modes, 5 param axes, train/eval factories, 19 tests) | Documented |
| **(2) DrQ-V2+ → CQN-AS** | DONE — vendored + smoke-green (A6) + merged to main (A8). E1.4 ablation folded into Phase 3 eval (E3.6). | Documented as the target architecture; value-based equations |
| **(3) Workspace reward shaping** | DONE — `SafetyConfig.add_workspace_penalty` wired through factory + `_reward()` (P3.0a). β-sweep (E3.X.workspace) still pending. | Documented as a method subsection + experiment E3.X.workspace |
| **(4) G1 humanoid as COWORKER stand-in** | DONE 2026-05-27 (on `retryg1`) — `env.human_model=g1` dispatch, parallel `G1HumanController` / `G1HumanIK` / `g1_human_spec.py`, generated `g1_human_body.xml`, real Unitree STL visuals. Verification curriculum run pending. | New writeup at [g1_coworker_swap.md](g1_coworker_swap.md) |
| **(5) Three-flavour safety metrics** | DONE 2026-05-27 — `info["safety"]` emits `ssm_violation` / `ssm_violation_actual` / `proximity_violation` plus margins, observed velocities, threshold echo. `EpisodeSafetyMetrics` emits the full thesis `ep_*` schema. `train_cqn_as` writes `metrics.jsonl` + `final_metrics.json`, forwards W&B tags, aggregates eval `info["episode_safety"]`. | [safety_metrics.md](safety_metrics.md) is the schema spec; now load-bearing in code. |
| **(6) Stage 2 disruption tighten** | DONE 2026-05-27 — `coworker_train.yaml` knobs reset so the arm reliably reaches into the robot's workspace (~87 % proximity-violation rate in smoke vs 13-24 % previously). | Documented in [g1_coworker_swap.md](g1_coworker_swap.md) and the Workstream S2 entry in IMPLEMENTATION_STATUS. |

The remaining Phase-3 work is the Lagrangian glue (P3.1: λ PID + dual-Q `argmax_a [Q_r − λ·Q_c]` + Q_c training-loop integration); the immediate gate is the next G1 base curriculum run (see IMPLEMENTATION_STATUS "Next session — start here").

---

## File-by-file change inventory

### `UPDATED_PROJECT_PLAN.md`

**Added:**
- *Plan Change Log* section near the top — summarises all three changes, lists the rejected alternatives (manual curriculum, three-policy switching) with reasons
- *Phase 0.5: COWORKER Disruption Infrastructure — COMPLETE* — documents the new disruption with the parameter-axis table, train-vs-eval superset relationship, and downstream implications
- *E3.X.workspace* — new sub-experiment for the workspace-β sweep
- *Pre-port smoke gates* (4 items) — must pass before any full CQN-AS training run
- *Filter-under-CQN-AS* paragraph in Phase 2 explaining first-action-only filtering

**Rewritten:**
- *Project Status Summary* table — flagged Phase 1.4 as REWRITE REQUIRED, Phase 2 as DATASET REGEN REQUIRED, etc.
- *Phase 1 E1.1* — marked as legacy multi-disruption; finding still reportable as context
- *Phase 1.4* — DrQ-V2+ replaced by CQN-AS; new pre-port checklist added
- *Phase 3 Integration strategy* — three options renamed A-value / B-value-mean / B-value-CVaR; framed in value-based form; the "shared-encoder discussion becomes" subsection explains what the methodological justification now rests on
- *Phase 5 stress tests* — OOD reframed around the five COWORKER parameter axes (exploiting the strict-superset eval space, no hand-crafted OOD probes needed)
- *Risk register* — RoboBase shared-encoder risk moved to "resolved by construction"; six new risks added (CQN-AS integration, action-sequence aggregation, workspace-β both directions, distributional cost critic data sparsity, COWORKER violation rate, filter-sequence granularity mismatch)
- *Compute budget* — broken down by phase, with the regen step explicit
- *Closing summary* — new critical path: CQN-AS smoke → Phase 2 dataset regen + E1.4 in parallel → Phase 3

**Unchanged:**
- Phase 0 results (the simulator fixes are independent of all three changes)
- BodySLAMWrapper code and tests (Phase 1 wrapper build)
- The continuous cost-signal formulation
- The PID λ-update formulas
- The Phase 4 four-way comparison logic

### `HYBRID_SAFETY_CRITIC_PLAN.md`

**Added:** a clear "HISTORICAL — superseded" banner at the top, with a 3-bullet summary of what's changed since.

**Unchanged otherwise:** the rest of the document is now a frozen historical artefact. Don't reference it from the report; cite the updated plan if you cite anything.

### `report.tex`

**Preamble additions:**
- `\newcommand{\cqnas}{\textsc{cqn-as}}` macro
- `\DeclareMathOperator*{\argmax}{arg\,max}` for the new selection rule
- Three new bibtex stubs in the commented-out bibliography section: `seo2024cqnas`, `seo2024cqn`, `yarats2022drqv2`

**Replaced subsections / paragraphs:**
1. Abstract headline-claim placeholder — "five disruption types" → "sustained-coworker disruption with strict-superset train/evaluation parameter splits"
2. `sec:bg:rl` (RL Fundamentals) — DrQ-V2+ replaced by CQN-AS as the focus; DrQ-V2+ kept as comparison
3. `sec:method:wrappers:workspace` (NEW SUBSECTION) — workspace shaping equation + calibration paragraph
4. `sec:method:lagrangian:A` — Option A renamed to **Option A-value**, equation rewritten with `argmax_a Q(s,a)` and `r_workspace` included
5. `sec:method:lagrangian:Bmean` — Option B-mean → **Option B-value-mean**; the entire "Implementation in RoboBase. RoboBase's ActorCritic shares an encoder..." paragraph is *replaced* with a paragraph titled "Why the cost critic is architecturally decoupled" that justifies the choice on stationarity + weight-transfer grounds
6. `sec:method:lagrangian:Bcvar` — Option B-CVaR → **Option B-value-CVaR**; actor-objective equation replaced by argmax selection rule
7. `sec:method:lagrangian:diagram` (Figure 1) — the three "Actor π" tikz nodes relabelled as `argmax_a`, panel titles renamed to A-value / B-value-mean / B-value-CVaR, caption rewritten
8. `sec:method:lagrangian:algo` (Algorithm 1) — `\Require` adds the bin set; the `Sample a ~ π_θ` line replaced by an explicit `argmax_a` line; the actor-update line removed; the cost-Q update line annotated `not chunk-aggregated` to flag the per-step backup requirement; optional warm-start from Phase 2 SVF added
9. `sec:method:hybrid` — placeholder rewritten to use the CQN-AS argmax-bias language instead of "Lagrangian-trained actor"
10. `sec:method:asymmetry` (Asymmetric Observation Handling) — three options rewritten; the "RoboBase's shared encoder makes architectural asymmetry difficult" framing is removed; new framing is methodological (where ground-truth enters, where it does not)
11. `sec:setup:disruptions` (Disruption Scenarios) — full rewrite from "five disruption types" to "single COWORKER with three trajectory modes" + new parameter-axis table (`tab:coworker-params`)
12. `sec:setup:metrics:taskperf` & `:failures` — wording adjusted for trajectory modes / parameter status
13. `sec:setup:hparams` — actor learning rate replaced by task-Q-network learning rate; CQN-AS hyperparams (B bins, L levels, K sequence length) added as placeholders
14. `sec:setup:compute` — added Phase 0.5; bumped Phase 3 range to 55–65h to accommodate the workspace-β sweep; added sentence on CQN-AS sample efficiency offset
15. `sec:results:filter` — table caption now says "averaged across the COWORKER trajectory-mode mixture"
16. `sec:results:human-state:rl` — DrQ-V2+ replaced by CQN-AS in the E1.4 result paragraph
17. `sec:results:constrained:cvar` — "all five disruptions" → "all three modes"; cross/approach mentions replaced by COWORKER_PATROL
18. `sec:results:hybrid` (Table E4.1 caption) — "five disruption types" → "COWORKER disruption (three trajectory modes, five parameter axes)"
19. `sec:results:failures` (Table failure-modes) — rows replaced: trajectory mode rows + parameter-status rows
20. `sec:results:ood` — additional stress conditions reframed around the five COWORKER parameter axes

**Renamed throughout (sed pass):**
- `Option~A` → `Option~A-value` (27 sites total across results sections and discussion)
- `Option~B-mean` → `Option~B-value-mean`
- `Option~B-CVaR` → `Option~B-value-CVaR`

**Unchanged in report.tex (intentionally):**
- Chapter 1 (Introduction) — motivation and problem statement are abstraction-level above any of these changes
- Chapter 2 (Background) ISO 15066 sections, Constrained MDPs section, Distributional safe RL section, Recovery RL background — all about the *literature* and don't depend on which backbone we use
- Phase 0 wrapper / collision-channel / mocap content — fixes were upstream of all three changes
- Phase 2 offline filter sections — the filter is policy-agnostic and survives unchanged
- Continuous cost signal definition (`c_ssm`, `c_pfl`, `c_t`) — same formulas
- PID λ-update equation — same formulas
- WCSAC external-baseline table E3.7 — kept as headline external baseline (it's distributional safe-RL, and WCSAC vs CQN-AS framing actually strengthens the contribution claim)
- Failure-mode walkthrough placeholders — narrative is still "canonical success / canonical filter-freeze / canonical residual violation"

---

## Things you may need to redo

| Item | Why | Priority |
|---|---|---|
| **Phase 2 safety dataset (~310k transitions)** | Existing dataset was collected across 5 disruption types; downstream Phase 2/3/4/5 need transitions sampled from the COWORKER train ParameterSpace so the filter and the cost Q-network match the eval distribution | **HIGH** — blocks Phase 2 GPU work and any honest E2.2 / E3.4 result |
| **Phase 1.4 — RL pilot** | Was set up for DrQ-V2+; rewrite for CQN-AS with new wrappers | **HIGH** — gates the Phase 3 observation-config decision |
| **Phase 0 ACT snapshots (used as a Phase 2 data source)** | Currently rolled out across 5 disruption types; new rollouts on COWORKER train space needed before re-collecting the safety dataset | **MEDIUM** — can be done as part of the dataset regen step rather than separately |

| Item | Why this is **not** a redo | |
|---|---|---|
| **Phase 1 E1.1 (BC obs-ablation)** | The qualitative finding ("BC marginalises the channel without a reward gradient") is robust to the disruption change. Numbers stay as historical context. Only redo if a reviewer specifically asks for a COWORKER-only BC table. | — |
| **Phase 2 code modules** | Filter pipeline, CQL trainer, threshold sweep, runtime wrapper — all policy- and disruption-agnostic. No code changes needed; only the dataset they consume changes. | — |
| **Phase 0 simulator fixes** | Collision channels, mocap pelvis, SSM velocity cap — all upstream of disruption and RL backbone. Untouched. | — |
| **BodySLAMWrapper code** | Already disruption-agnostic. The encoder side of CQN-AS may need a small modification to consume the `human_pos_estimate` observation key, but that's a CQN-AS-side change, not a wrapper-side change. | — |

---

## Things to do next (ordered)

### P0 — Gate everything else

1. **Pull the CQN-AS reference implementation.** Project page: `younggyo.me/cqn-as`. Paper: arXiv 2411.12155 (Seo et al., 2024). Get it running on stock `reach_target_single` with no project wrappers. ~2–4 GPU-h.
2. **Compose-with-wrappers smoke.** Run CQN-AS on `reach_target_single` with the existing `BodySLAMWrapper` and the new COWORKER scenario sampler active. A few thousand env steps is enough — you're checking that:
   - The encoder accepts the `human_pos_estimate` observation key (may require small modification to fuse the 6D vector with pixels/proprio)
   - The COWORKER scenario doesn't crash the env at episode boundaries
   - The cost signal `c_t` is logged per-step inside K-step chunks (not aggregated at chunk boundary)
3. **Decide go / no-go on the CQN-AS port.** If steps 1–2 reveal a blocker, fall back to DrQ-V2+ for E1.4 (the existing config still works) and re-evaluate. Document the blocker in the Phase 3 risk register.

### P1 — Parallel workstreams once P0 clears

4. **Phase 2 dataset regen on COWORKER train space.** Wire `make_coworker_train_space()` into the existing collection scripts. ~2–3 hours wall-clock. After regen, run the existing CQL training + threshold sweep without further code changes; the SVF input/output shapes are unchanged.
5. **E1.4 CQN-AS observation ablation** on `reach_target_single`. 3 cells (off / oracle / noisy), COWORKER train space, ~10–15 GPU-h total. Eval against COWORKER eval space.

### P2 — Phase 3 build-out

6. **Workspace shaping wiring + E3.X.workspace sweep.** Add `r_workspace = -β · max(0, ‖p_ee − p_task‖ − r_ws)` to the env reward; sweep `β ∈ {0, 0.05, 0.2, 0.5, 1.0}` under Option B-value-mean on `reach_target_single`. Pick β at the knee of the evacuation-vs-violation Pareto.
7. **Implement Option A-value as a one-task prototype.** Validates the cost signal, PID gains, and λ_max clamp before B-value-mean / B-value-CVaR commitment. Single Q-network on `r_task + r_workspace − λ · c_t`.
8. **Option B-value-mean.** Two Q-networks. Warm-start `Q_c` from the Phase 2 SVF. Action selection at each coarse-to-fine level is `argmax_a [Q_r − λ · Q_c]`.
9. **Option B-value-CVaR.** Distributional `Z_c` head (start with Gaussian / WCSAC-style; switch to quantile if data permits). The Lagrangian budget is now on rolling CVaR.

### P3 — Phase 4 and writeup

10. Hybrid deployment (CQN-AS B-value-CVaR + Phase 2 filter), fallback ablation
11. Phase 5 — tail-risk + OOD on the five parameter axes
12. Writeup: fill `\result{}` placeholders left to right through the report

---

## Stale-numbers warning

The Phase 1 E1.1 result table in the report (Table 1.1 — `tab:e1.1-bc-ablation`) currently shows numbers from the multi-disruption setup:

| Task | off SSM | oracle SSM |
|---|---|---|
| reach_target_single | 0.534 | 0.548 |
| dishwasher_close | 0.533 | 0.537 |
| drawers_open_all | 0.277 | 0.239 |
| saucepan_to_hob | 0.135 | 0.203 |

These numbers are **still valid** as published — they describe an experiment we actually ran. But:

1. **Don't compare them directly to any new COWORKER-only baseline.** The disruption mixture is different.
2. **If we ever publish a COWORKER-only BC re-run, present both tables side by side** with explicit labels, and explain in the caption that the legacy table averages across `static`, `cross`, `reach`, `approach`, `occlude`, `random`.
3. **The Phase 0 / baseline `tab:baseline-iso` numbers** (SSM violation rate per task) were also from the legacy disruption mixture. The captions have been updated to say "averaged across the COWORKER trajectory-mode mixture" — but the actual `\result{0.53}` etc. placeholder numbers in the table are now mis-labelled. **Don't fill those cells with old numbers**; they need to be re-measured on COWORKER before reporting.

---

## Quick reference — the canonical option names going forward

In code, comments, plots, and writing, use these names:

- **Option A-value** — single Q on shaped reward; prototype only
- **Option B-value-mean** — dual Q (task + mean cost); the safe starting headline
- **Option B-value-CVaR** — dual Q (task + distributional cost); the final headline

The old `A` / `B-mean` / `B-CVaR` names refer to the actor-critic variants that no longer exist in the project. Any reference to those without the `-value` suffix should be assumed stale.

---

## Quick reference — the COWORKER parameter axes

| Axis | Train | Eval | Hydra key |
|---|---|---|---|
| Closest-approach distance | 0.9–1.4 m | 0.6–1.8 m | `coworker_closest_approach_range` |
| Reach period | 4.5–6.5 s | 3.0–9.0 s | `coworker_reach_period_range` |
| P(reach EE) | 0.4–0.6 | 0.1–0.9 | `coworker_target_mix_p_ee_range` |
| NEAR dwell | 7–11 s | 4–16 s | `coworker_near_loiter_range` |
| Walk speed | 1.0–1.6 m/s | 0.6–2.2 m/s | `coworker_walk_speed_range` |

Factories: `make_coworker_train_space()` / `make_coworker_eval_space()`. Hydra presets: `cfgs/disruptions/coworker_{train,eval}.yaml`.

Trajectory modes (sampled uniformly per episode): `STATIONARY`, `APPROACH_LOITER_DEPART`, `COWORKER_PATROL`.

---

## Open questions that didn't have a clear answer

These need a decision before you commit GPU time to them:

1. **Should `Q_c` see action sequences or single actions?** CQN-AS's task Q already takes a K-step sequence. If `Q_c` does too, you get sequence-level cost reasoning (good for predictive avoidance) but lose per-step interpretability. Recommendation: start with single-action `Q_c` (simpler, matches Phase 2 SVF for warm-start), upgrade to sequence-aware `Q_c` only if the policy persistently spikes violations within chunks.
2. **What's the right CQN-AS `B` (bins per level)?** The published BiGym hyperparams should work, but H1's 76 DOFs may want a different setting. Confirm in the smoke run that the published default doesn't blow up wall-clock per env step.
3. **Should the workspace radius `r_ws` be task-dependent?** For `reach_target_single` 0.4 m is generous; for `saucepan_to_hob` (vertical clearance constraints) it may be too restrictive. Plan: start with a single global `r_ws = 0.4 m`, treat per-task tuning as a fast-follow if E3.X.workspace shows differential pathology.

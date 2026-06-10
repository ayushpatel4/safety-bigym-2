# Report Reframing Plan: From "Which Method Wins" to the Regime Map

**Status:** proposed (2026-06-10). **Inputs:** `results_discussion_draft.md` (saucepan arc),
`safety_filter_report.md` (dish/drawers arc), `c4_gated_reframe.md`, `wcsac_results.md`,
`figures/fig{1..5}`. **Output:** a restructured Results & Discussion (new master prose doc
+ updated `report_snippets.tex` blocks for `report.tex`).

---

## 0. The decision in one paragraph

The two existing results documents currently carry opposite headlines (draft: "proactive
constrained-RL is necessary, the filter is redundant"; filter report: "constrained-RL cannot
deliver, the gated filter is the result"). Neither becomes the thesis spine. The spine is the
**regime map**: which safety mechanism works is governed by the human–robot **co-location
regime**, and the two headline results are the two arms of one finding. Nothing is thrown
away — the saucepan arc becomes the *persistent-co-location* chapter, the dish/drawers arc
becomes the *intermittent-co-location* chapter, and a new synthesis section (fig5 +
boundary condition) unifies them. The filter report's internal ordering (metric correction →
axis → failures-with-mechanisms → gated win → boundary) is adopted *within* its chapter.

---

## 1. The new frame

### 1.1 Thesis statement (ready to adapt)

> Safety for a manipulation robot beside a human coworker decomposes into two axes: a
> partly-exogenous **proximity** axis (≈42% human-driven in our scenes) and a
> robot-controllable **ISO-15066 SSM-velocity** axis. Which safety mechanism helps is
> governed by the **co-location regime**, not by the mechanism's sophistication. Under
> *persistent* co-location, only an anticipatory constrained-RL policy reduces proximity
> gracefully — every reactive filter pays a freeze-or-flee cost, and critic-gating
> degenerates to unconditional scaling. Under *intermittent* co-location, constrained RL
> fails to bind usefully, and a learned critic gating an ISO speed-scaling backstop is the
> only mechanism that improves velocity-axis safety without collapsing task success —
> because genuinely-safe windows exist for the critic to find. The Hybrid Safety Critic is
> the two-arm architecture; the regime map is the rule for which arm carries safety where.

### 1.2 Contributions (ordered — this replaces the current contributions list)

1. **Measurement.** An exogenous/endogenous decomposition of proximity (~42% human-driven
   floor; fig4) motivating a two-axis evaluation protocol with explicit axis ownership:
   policy ↔ proximity (exposure), runtime filter ↔ SSM-velocity (robot-controllable).
2. **The regime map / boundary condition.** Critic-gating recovers task throughput **iff**
   co-location is intermittent; persistent co-location requires anticipatory avoidance.
   Validated on three tasks with a **pre-registered decision rule** whose "else" branch
   fired on saucepan (fig5; `c4_gated_reframe.md`).
3. **Mechanism findings.** (i) Freeze-vs-flee taxonomy of reactive filters on the proximity
   axis; (ii) binary veto breaks chunked-policy action coherence (max-vel 2.46→6.03) while
   continuous scaling preserves it (2.46→2.46) — hence *gate the scaler, don't veto*;
   (iii) PID-λ is seed-unstable at feasibility-boundary budgets — fix λ instead.
4. **Deployable recipes + practitioner lessons.** Per-task operating points with CIs; the
   R-dial frontier; train the gate critic on *unfiltered* rollouts (DAgger-on-filtered
   under-gates); WCSAC as external safe-RL reference.
5. **Deployment-faithful evaluation pipeline** (noisy BodySLAM, scripted coworker
   disruptions, multi-seed bootstrap CIs).

### 1.3 Terminology decisions (apply as a global sweep)

| Old term | New term | Notes |
|---|---|---|
| "Hybrid Safety Critic" | **reserved** for the two-arm architecture (training-time constrained policy + runtime critic-gated backstop) | Define once in the intro; the empirical finding is which arm carries safety per regime |
| "gated hybrid" (filter report) | **critic-gated speed scaling** | The runtime method. Avoids two meanings of "hybrid" |
| "the hybrid" (draft §3.4 composed system) | **policy–filter composition** | The stacked C4 configuration |
| "SVF veto" | unchanged | |
| metric names | "proximity (exposure) axis" / "SSM-velocity (robot-controllable) axis" | Used consistently in every table header |

### 1.4 Metric language (the goalpost-proof phrasing)

Adopt: *"Proximity measures exposure, but ≈42% of it is human-driven and no robot policy can
remove it. We therefore retain proximity as a secondary descriptor — it remains the natural
axis for policy-level avoidance — and use velocity-adaptive SSM as the primary
robot-controllable safety axis for runtime filters."*

Three-part defense, stated explicitly in R0:
1. **Provenance:** the fig4 decomposition comes from a *prior* controlled sweep
   (saucepan SVF sweep, `filters/snapshots.py`) — the axis was derived before the gated
   result existed, not after.
2. **Both axes are reported for every configuration in every table** — no result is hidden
   by the switch.
3. **The switch cuts both ways:** the policy's headline win (−23%) lives on proximity, the
   filter's on velocity. This is axis *ownership*, not metric abandonment.

Never write "proximity was the wrong metric."

---

## 2. New Results & Discussion structure (source → destination)

### R0 — Evaluation protocol and the two safety axes
- Benchmark setup (from draft *Setup*), seeds/episodes/CIs, disruption, noisy BodySLAM.
- The exogenous decomposition (**fig4 here, first figure**) → two axes defined, axis
  ownership, §1.4 language. Pre-registered decision rules described as protocol.
- Sources: draft §1 (decomposition half), filter report §1.

### R1 — Persistent co-location (saucepan): anticipatory avoidance is necessary
- **R1.1 Reactive filtering is bounded on the proximity axis.** Freeze-vs-flee taxonomy:
  SVF veto (freeze/dwell), base-CBF dodge + EE flinch (flee), on-policy critic null —
  *claim rescoped*: "no reactive filter gracefully reduces **proximity** under
  **persistent co-location**" (not "reactive filtering is impossible"). Sources: draft §1,
  §3, §3.1–3.3.
- **R1.2 The constrained policy delivers.** −22.8% proximity at 0.76 success (pooled,
  3 seeds), ssm-actual 0.146→0.112; fixed-λ vs PID instability finding; cost-form ablation
  (E3.1 snippet); shaping and λ=0 controls. Sources: draft §2, `report_snippets.tex`.
- **R1.3 Axis division and the composition ceiling.** Speed-scaling owns the velocity axis
  (0.146→0.048); the policy–filter composition is the only both-axis configuration but the
  costs stack multiplicatively (0.85→0.44 success = C4). Framed as *the regime's ceiling*,
  not a defect: under persistent co-location, both-axis safety is purchasable only at
  stacked cost. Source: draft §3.4.
- **R1.4 Robustness and the exogenous tail.** E5.2 unseen coworker, E3.6 perception
  two-mode result, CVaR tail unchanged (E5.1). Sources: draft §4–5.

### R2 — Intermittent co-location (dishwasher/drawers): the critic-gated backstop wins
- **R2.1 Constrained RL fails to bind usefully here.** Budget inert-or-fatal; when λ binds,
  velocity *rises* (mean 0.44→0.79, max 2.46→4.14) — collapse, not careful slowing.
  **State the trainability confound honestly:** these are sparse-reward tasks where the
  base policy is fragile; WCSAC (external, from-scratch) corroborates — it learns
  dishwasher (~0.47) but 0% on drawers at every budget (`wcsac_results.md`). The verdict is
  "no feasible budget on these tasks," not "constrained RL is universally dead" — saucepan
  (R1.2) is the counterexample. Sources: filter report §2, §3.2; `budget_sweep_results.md`.
- **R2.2 Binary veto fails on chunked policies** (coherence break, overshoot mechanism).
- **R2.3 Unconditional speed-scaling** works on its axis at heavy task cost.
- **R2.4 Critic-gated speed scaling** — the when/how split; dishwasher −50% SSM at −0.10
  success, drawers −22% at −0.09 (near-free option: R=2.75, −10% at −0.02); the R-dial
  frontier (**fig1 + fig2 here**, captions per §4); recommended operating points table.
- **R2.5 Ablations** (d_slow×R Pareto, anticipatory labels, gate-on-Lagrangian, DAgger
  null → train the gate on unfiltered rollouts). Source: filter report §5.

### R3 — The boundary condition (new synthesis section; the thesis centerpiece)
- **fig5 here.** Dish/drawers *bend down*; saucepan *slides along the diagonal* toward the
  unconditional point — gating ≈ unconditional under persistent co-location.
- The pre-registered decision rule (succ ≥ 0.60 ∧ ssm ≤ 0.08) and its "else" branch firing
  on saucepan — emphasize this was the *predicted* failure mode, not an anomaly.
- The mechanism: gate-active fraction tracks co-location persistence; this is the **same
  mechanism** the draft already identified ("fires almost constantly", §3.4) — the two
  documents discovered one law from opposite sides.
- **Explicit sentence: gating does NOT rescue the C4 collapse on saucepan; the contribution
  is the explanation and its cross-task validation.**
- The regime-map summary table:

  | Regime | Proximity axis | Velocity axis | What fails |
  |---|---|---|---|
  | Persistent (saucepan) | constrained policy (−23%) | unconditional scaling (−67% ssm) at heavy cost; composition = only both-axis route (C4 ceiling) | all reactive proximity filters (freeze/flee); critic-gating (≈ unconditional) |
  | Intermittent (dish/drawers) | bounded by exogenous floor; no graceful method found | **critic-gated scaling** (−50% / −22% at ≤0.10 success cost) | Lagrangian (inert-or-fatal); binary veto (coherence break) |

- **Snapshot caveat box** (verbatim from `c4_gated_reframe.md`): saucepan sweep ran on
  `snapshot_best`, basin checkpoints deleted; the unconditional control reproduces C4 on
  the filter axes (0.433/0.072 vs 0.44/0.065), so the filter-level verdict transfers.

### R4 — Limitations and future work (merged)
- Exogenous tail (CVaR unchanged by any method); PFL unevaluated (contact-force bug);
  operating points are task-specific; proactive avoidance beyond the λ signal
  (potential-based separation reward); formal CBF-QP/HJ filters; coworker realism spectrum;
  **the persistence-dial experiment (W6 below) as the named next step.**

---

## 3. Claim-by-claim rescoping edits

| # | Location | Edit |
|---|---|---|
| E1 | draft lines 5–7 | "Primary safety axis: proximity-violation rate" → the §1.4 two-axis protocol sentence |
| E2 | draft §3.1 (≈ lines 127–130) | "proactive constrained-RL is *necessary*, not merely preferable" → scope: "…on the **proximity axis** under **persistent co-location**" |
| E3 | draft §2 "reproducibly" + §4 E5.3 "[Results pending]" (≈ 342–343) | E5.3 resolved **negative**: the saucepan pipeline does not replicate on dish/drawers → cross-ref R2.1; scope "reproducibly" to saucepan seeds |
| E4 | draft §6 synthesis (≈ 391–402) | Superseded entirely by R3 (regime map). Do not ship both |
| E5 | filter report Summary bullet 1 (≈ 22–24) | "Constrained RL cannot deliver" → "…on the intermittent-co-location tasks; on saucepan a fixed-λ policy does deliver (R1.2). The mechanism here is budget infeasibility, compounded by sparse-reward base-task fragility (WCSAC corroborates)" |
| E6 | filter report §1 consequence box (≈ 53–56) | Replace "proximity is the wrong target" tone with §1.4 exposure/control phrasing |
| E7 | filter report §3.3 (≈ 119–121) | "dominates unconditional at every point" → scope to dishwasher ("matched SSM at higher success"); on drawers unconditional retains the lowest SSM floor (0.059 vs best gated 0.078) — say "the gated frontier never reaches unconditional's floor but pays far less success at every shared SSM level" |
| E8 | fig1 caption + §3.1 table | **The drawers-Lagrangian defense (must land before fig1 is promoted):** "the Lagrangian point is a basin-selected checkpoint from seed-unstable training (no feasible budget exists; PID-λ is inert or task-fatal across the sweep); gating-on-Lagrangian ≈ gating-on-baseline (ablation 3) shows it learned no genuinely proactive avoidance; and it offers no deployable dial, whereas R traces a frontier" |
| E9 | both docs + snippets | Terminology sweep per §1.3 |
| E10 | both docs | C4 phrasing audit: every "recovers throughput" carries "where safe windows exist"; one explicit "gating does not rescue saucepan C4" |
| E11 | filter report §3.4 caveat | Keep; becomes the R3 caveat box |
| E12 | filter report §2 methods table | Add WCSAC row as external reference (from-scratch; dishwasher 0.47 / drawers 0.00) |
| E13 | `report_snippets.tex` | Add new blocks: R0 metric paragraph, R3 boundary subsection + regime table, fig1 defended caption; re-wire `\cref` labels if section numbers shift |

---

## 4. Figure plan

| Figure | Placement | Caption requirement |
|---|---|---|
| fig4 (exogenous) | **R0, first figure** | States provenance (prior sweep) explicitly — the goalpost defense |
| fig1 (method comparison) | R2 headline | Carries the E8 drawers-Lagrangian defense |
| fig2 (R-dial) | R2 | Dominance wording scoped per E7 |
| fig3 (reduction bars) | Conclusion / exec summary | Marker-facing summary |
| fig5 (cross-task boundary) | **R3 centerpiece**; optionally echoed in the intro as the contributions figure | "Bend down vs slide along the diagonal" language |
| **NEW F6** — regime-map schematic | Intro or R3 | The §R3 table as a graphic (matplotlib or TikZ); the thesis-in-one-image |
| **NEW F7 (optional, high value)** — gate-activity vs persistence | R3 | Bar chart: gate-active fraction among near-human steps, per task (saucepan ≈ continuous vs dish/drawers intermittent). Converts the boundary mechanism from asserted to **measured**. Data: intervention rates in `results/e4_1/gated_saucepan/summary.json` + `results/gated_sweep/`; if per-step logs are absent, use episode-level intervention rate as the proxy and say so |

Regen path: extend `scripts/make_report_figures.py` (F6/F7); `scripts/plot_gated_pareto.py`
already produces the saucepan slide plot.

---

## 5. Examiner FAQ (risk register — each answer must exist in the text)

1. **"You changed the metric after seeing results."** → §1.4 three-part defense, in R0.
2. **"On drawers your method ties the Lagrangian (fig1)."** → E8 defense in the caption +
   R2.1 instability evidence + ablation 3 + the dial argument.
3. **"It only works on 2 of 3 tasks."** → R3: the third task is the pre-registered else
   branch; 3/3 results are consistent with the map. The failure is *predicted*, which is
   stronger than a third win.
4. **"Why does constrained RL work on saucepan but not dish/drawers?"** → R2.1 discussion,
   labeled as hypothesis with evidence: (a) base-task trainability (WCSAC drawers 0%
   corroborates); (b) cost-signal density — persistent co-location yields a dense cost
   gradient and a feasible budget region; intermittent yields sparse/bursty cost and a
   budget cliff (budget-sweep evidence). Do not overclaim; flag as discussion-level.
5. **"The gains are modest / the hybrid isn't free."** → Never claim free lunch (C4
   honesty is a feature); the contribution is the map + a tunable dial with CIs.
6. **"The saucepan sweep used the wrong checkpoints."** → R3 caveat box: control
   reproduces C4 on the filter axes; verdict is filter-level.

---

## 6. Work plan (ordered)

| Step | Deliverable | Size |
|---|---|---|
| W1 | Thesis statement + contributions + terminology table → intro patch | 0.5 d |
| W2 | Assemble `docs/results_regime_map.md` (R0–R4) from the source map in §2; mark `results_discussion_draft.md` superseded at top | 1–1.5 d |
| W3 | Apply edit list E1–E13 | 0.5 d |
| W4 | Figures: F6 schematic; fig1/fig2 captions; F7 if logs support it | 0.5–1 d |
| W5 | Update `report_snippets.tex`; wire labels into `report.tex` | 0.5 d |
| W6 | **Optional, strongest single addition — the persistence dial:** same task, two coworker schedules. Cheapest version: make the *dishwasher* coworker persistent (scripted controller dwell), rerun the gated R-sweep. Prediction: the bent frontier flattens onto the diagonal (gating → unconditional). This isolates *persistence* from *task identity* — a controlled intervention on the regime variable, upgrading the boundary condition from a 3-point observation to a causal test. Per workspace rules: write script + ≤100-step smoke here, hand off the sweep to GPU | prep 0.5 d + GPU hand-off |
| W7 | Related-work positioning (`related_work_hybrid_filter.md` base): the gap claim is that shielding and constrained-RL literatures exist, but no characterization of *when* each helps vs hurts in HRC; slot WCSAC as the external reference | 0.5 d |

**Decisions for the author:** (a) run W6 or defer to future work; (b) F7 mechanism figure
yes/no; (c) does fig5 appear twice (intro + R3) or once.

---

## 7. Explicitly superseded / out

- Draft §6 synthesis ("proactive ≫ reactive; filter redundant") — replaced by R3.
- "The hybrid is counterproductive" as a headline — survives only as the R1.1/R1.3
  SVF-veto-composition detail.
- Any sentence implying critic-gating rescues the C4 saucepan collapse.
- "Proximity was the wrong metric" phrasing in any form.
- Detail docs stay as appendix references: `runtime_filter_results.md`,
  `budget_sweep_results.md`, `c4_gated_reframe.md`, `wcsac_results.md`.

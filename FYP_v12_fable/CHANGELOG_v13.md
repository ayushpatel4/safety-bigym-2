# Report rewrite log: v12 (saucepan arc) → v13 (regime map)

Reframing per `docs/report_reframing_plan.md` (2026-06-10), targeting the
markscheme in `report guide/`. Backups: `main_v12_backup.tex`,
`references_v12_backup.bib`. Final state: **83 PDF pages, 59 content pages
(chapters 1–8, printed pp. 11–69) — within the 60-page limit**; clean
pdflatex+bibtex build, zero undefined references/citations, zero bibtex
warnings, every figure/table referenced in text, no draft markers.

## The one-paragraph delta

v12 carried the saucepan-arc headline ("proactive constrained-RL is necessary;
reactive filtering is fundamentally limited; the SVF-veto hybrid is
counterproductive; WCSAC cut for scope; cross-task transfer in progress").
v13's spine is the **regime map**: which safety mechanism works is governed by
the human–robot **co-location regime**. The saucepan arc became the
*persistent-co-location* chapter (R1); a new *intermittent-co-location*
chapter (R2: dishwasher_close + drawers_open_all) was added; a new synthesis
section (R3) unifies them with a measured mechanism; and proximity was
rescoped from "the thesis's primary safety metric" to the *exposure* axis of a
two-axis protocol (R0), with the ~42% exogenous decomposition promoted to a
headline measurement contribution. Nothing was thrown away: every v12 result
survives, rescoped to its regime.

## Chapter-by-chapter: kept / modified / added / lost

### Abstract — REWRITTEN
Regime-map framing; both arms' headline numbers; pre-registered-rule sentence;
WCSAC mentioned as implemented corroboration; "first to characterise *when*"
closing claim.

### Ch. 1 Introduction — MODIFIED
- KEPT: motivation narrative, problem-statement CMDP, E1.1-motivating logic.
- MODIFIED: gap #2 is now "no characterisation of *when* each mechanism
  helps" (with shielding refs); problem statement gains the two-axis +
  three-task + regime-variable paragraphs; RQ2–RQ4 rewritten (RQ4 is now the
  regime map); report outline updated.
- ADDED: Terminology paragraph fixing "Hybrid Safety Critic" (two-arm
  architecture) / "critic-gated speed-scaling" (runtime arm) /
  "policy–filter composition" (stacked C4) — plan §1.3.
- REPLACED: contributions list → 6 items ordered per plan §1.2
  (benchmark+pipeline, measurement/two-axis, regime map, mechanism findings,
  recipes+lessons, value-based Lagrangian + support proposition).

### Ch. 2 Background — MODIFIED
- KEPT: ISO 15066, RL fundamentals, safe RL, CQL, sim-to-real, both
  positioning tables.
- ADDED: shielding (Alshiekh 2018) as the runtime-override anchor + the
  "override form" design axis; freezing-robot problem (Trautman & Krause
  2010); HJ overview citation (Bansal 2017); Marvel & Norcross SSM
  implementation citation; explicit "neither literature names the variable
  separating its successes from failures" gap paragraph.
- MODIFIED: positioning text — WCSAC now "reimplemented faithfully…
  external reference"; closing paragraph claims the regime map as the
  distinguishing output; CBF-RL row corrected to "training-time" guarantee
  (per the corrected citation).
- FIXED (citation audit fallout): "Chinchali et al." → Bharadhwaj (real
  author of arXiv:2110.05702); SHIELD institutional attribution softened to
  match verified author list.

### Ch. 3 Benchmark — MODIFIED
- KEPT: all infrastructure content (G1 stand-in rationale, mocap pelvis,
  collision channels, workspace shaping, harness, coworker design,
  perception model, demo pipeline, threshold calibration).
- MODIFIED: metric-flavours section rewritten to two-axis/axis-ownership
  language (plan E1/E6 — "exposure" vs "robot-controllable", both axes in
  every table); "The Primary Task" → "The Task Suite: Three Tasks, Two
  Co-location Regimes" with the regimes defined operationally (measured by
  gate activity, not asserted).
- LOST (page budget): the prox-calib CDF *figure* (its data lives on in
  Table 3.1, which is retained); various prose compressions (coworker item 2,
  arm state machine, demo section, G1 paragraphs, adapter section,
  why-new-benchmark list) — no facts removed, only wording.

### Ch. 4 Method — MODIFIED
- KEPT: SVF dataset/labelling, speed-scaling + dodge filters, Lagrangian
  derivation (Options A/B-mean/B-CVaR), support-bounding proposition + proof,
  per-step cost backup, PID + fixed-λ finding, warm-start logic, curriculum,
  algorithm float, deployment-faithfulness section, CUDA bug (compressed).
- ADDED: §4.3.5 Critic-Gated Speed-Scaling (the runtime arm; gate equation;
  when/how split; "whether the frontier bends or slides is the regime
  question"); shared dish/drawers SVF critic paragraph (52,794 transitions,
  0.10 m near-contact label, task-agnostic); §4.5 WCSAC external baseline
  (faithful reimplementation, from-scratch honesty, identity-stats validity
  fix); architecture-overview and fig caption updated to two-arm/regime
  framing.
- LOST (page budget): the 3-panel Lagrangian-formulations TikZ figure
  (unreferenced; the three options are fully specified in prose); the
  curriculum stage table (now 4 lines of prose, same facts); PID-instability
  details deduplicated with E3.2 (evidence now lives once, in results).

### Ch. 5 Setup & Results — RESTRUCTURED (the core of the rewrite)
- Setup: three tasks; baselines per regime; **R0 "The Two Safety Axes and
  the Evaluation Protocol"** replaces "Primary Safety Axis" — fig4 promoted
  to first results figure with provenance caption; three-part goalpost
  defence (provenance / both-axes-everywhere / cuts-both-ways) verbatim in
  spirit from plan §1.4; the three **pre-registered decision rules** stated
  as protocol; statistical methodology gains the single-seed flags
  (budget sweep, WCSAC); compute updated (saucepan 220 GPU-h log-reconstructed
  + intermittent arc & WCSAC ≈230–280 h estimated, total ≈450–500 h);
  experiment index extended (E2.4–E2.6, E3.3, E3.7 implemented, E4.4, E5.3
  resolved). Perception-mode matrix table → 6-line prose (same content).
- **R1 Persistent co-location (saucepan)** — all v12 results KEPT with
  rescoping edits: E3.1 (form not operative), E3.2 (budget/PID instability),
  E2.1–E2.3 filter Pareto + fallback dilemma, E4.1 headline table (C1–C4),
  basin paragraphs, shaping confound, freeze-vs-flee taxonomy (claim scoped:
  "no reactive filter gracefully reduces *proximity* under *persistent
  co-location*" + forward-pointer to R2), velocity-axis + joint-coverage
  (now one two-panel figure), **composition ceiling reframed as "the
  regime's ceiling, not a defect"** (plan R1.3), E4.3 internalisation,
  E5.1 tail (decomposition derivation moved to R0; "no configuration moves
  CVaR" strengthened), E5.2 OOD + **E5.3 resolved negative** ("reproducibly"
  scoped to saucepan seeds — plan E3).
- **R2 Intermittent co-location (dishwasher/drawers)** — ADDED:
  - R2.1 (E3.3) budget-retarget sweep table; binding collapses the task;
    "minimum achievable cost ≈ natural cost"; deployment of best cells
    (drawers −13% within noise); **velocity rises when λ binds (0.44→0.79
    mean, 2.46→4.14 max)**; trainability confound stated honestly + WCSAC
    corroboration table (E3.7: dish 0.47 / drawers 0.00; single-seed,
    oracle-eval, b30-variance caveats); "no feasible budget ≠ constrained RL
    universally dead — saucepan is the counterexample" (plan E5).
  - R2.2 (E2.4) veto breaks chunked policies: critic sound in-sample,
    override wrong; overshoot 2.46→6.03; ssm not improved.
  - R2.3 (E2.5) unconditional scaling: −50%/−41% at heavy cost; maxVel
    unchanged (2.46→2.46).
  - R2.4 (E2.6) critic-gated speed-scaling: when/how split; method table
    with both axes + WCSAC external rows (plan E12); fig1 with the
    **drawers-Lagrangian defence in the caption** (plan E8); fig2 with
    **task-scoped dominance wording** (plan E7); near-free drawers point;
    "recovers throughput *where safe windows exist*" scoping (plan E10).
  - R2.5 ablations: d_slow×R (R is the dial; overshoot returns at
    R=3.0/d_slow=0.4), anticipatory labels (selective 0.10 m label wins),
    gate-on-Lagrangian ≈ gate-on-baseline (E8 evidence), DAgger
    re-collection under-gates → **train the gate on unfiltered rollouts**.
- **R3 The Boundary Condition** — ADDED:
  - E4.4 full gated-saucepan sweep table (incl. on-policy v3op rows +
    unconditional control); pre-registered rule (succ ≥0.60 ∧ ssm ≤0.08);
    **no row passes; else-branch fired as predicted**.
  - fig5 with "bend down vs slide along the diagonal" caption.
  - **F7 (new figure): the mechanism measured** — exact per-step
    gate-active fractions (not a proxy): saucepan 61.5% [57.2,65.4] vs
    dishwasher 26.5% [16.3,36.7] vs drawers 19.2% [12.7,25.3] at matched
    R=2.75; saucepan exceeds even the unconditional trigger rate (44.2%).
  - The boundary-condition statement as a displayed quote; "two arcs
    discovered one law from opposite sides".
  - **F6 (new figure): regime-map TikZ schematic** — the thesis in one
    image, with per-cell evidence refs in the caption (absorbed the planned
    regime table; see LOST).
  - **Explicit sentence: critic-gating does NOT rescue the saucepan
    composition (0.85→0.44 stands)** (plan E10/§7).
  - Snapshot caveat paragraph (basin checkpoints deleted; unconditional
    control reproduces C4 on filter axes 0.433/0.072 ≈ 0.44/0.065; residual
    bias direction argued; verification item, not a threat) (plan E11).
- Summary of Results — REWRITTEN around RQ1–RQ4 with regime answers; fig3
  placed here as the marker-facing summary (plan fig3 placement).

### Ch. 6 Discussion — MODIFIED
- RQs revisited rewritten: RQ2 includes the **cost-signal-density
  hypothesis flagged as discussion-level** (examiner FAQ #4); RQ3 = override
  form/gate/regime; RQ4 = the regime variable + tail.
- Threats: PFL, curriculum, sim-to-real KEPT (compressed); filter-role
  subsection REWRITTEN as "axis- and regime-conditional" (narrower + broader
  halves); arch-gen subsection now records **E5.3 resolved negative**;
  WCSAC subsection REPLACED ("cut for scope" → "implemented, with stated
  limits": single seed, from-scratch conflation, oracle eval, no Safety-Gym
  revalidation); **NEW threat subsection: "The Regime Classification Is
  Observational, on Three Tasks"** (pre-registration + step-level mechanism
  + independent prediction as mitigations; persistence dial as the
  falsification test) (examiner FAQ #3).
- Broader impact: ADDED "measure the co-location pattern first" deployment
  rule; internalisation and trade-transparency lessons kept.

### Ch. 7 Future Work — RESTRUCTURED (9 → 6 sections)
- ADDED: **§7.1 The Persistence Dial** (named next step; W6; falsifiable
  prediction; infrastructure exists); **§7.3 Proactive avoidance beyond the
  λ signal** (potential-based separation reward + honest geometry-ceiling
  risk); the literature-surfaced open question (anticipatory filter with
  graded slow-down fallback) inside §7.5.
- KEPT: PFL fix (now includes the collision-imminent brake completing the
  division of labour), CVaR-Lagrangian, formal CBF-QP/HJ filters,
  real-robot transfer + coworker realism spectrum + SMPL-H replication.
- LOST (folded or cut): standalone Recovery-RL-comparison, risk-measures,
  cumulative-budget, prop-damping-fallback and filter-during-training
  sections (the latter two survive as sentences in §7.5; all their \labels
  preserved so no dangling refs).

### Ch. 8 Conclusion — REWRITTEN
Three movements: contributions (benchmark, measurement, regime map);
headline results per regime incl. the pre-registered validation; broader
significance as "a decision rule where the field had a default" + the
portable practitioner lessons.

### Appendices — MODIFIED
- Hyperparameters: Phase-4 table → per-regime deployment configurations;
  ADDED shared dish/drawers SVF paragraph and WCSAC hyperparameter
  paragraph.
- Reproducibility: ADDED the R2/R3 recipe item (scripts + results dirs +
  figure regeneration incl. `make_f7_gate_activity.py`).
- Sustainability + Declarations: compute/CO2 updated (≈450–500 GPU-h, ≈40 kg
  CO2e — partly estimated, see "Author to verify").
- Data availability: ADDED dish/drawers critic + dataset, WCSAC checkpoints,
  per-cell CSVs; honest note that the R1 basin checkpoints are lost but
  their benchmark CSVs survive.

## Explicitly LOST from v12 (deliberate, per plan §7)

- "The hybrid is counterproductive; the policy makes the filter redundant"
  as a headline — survives only as the SVF-veto-composition detail in R1.
- "Proactive constrained-RL is *necessary*" unscoped — now scoped to the
  proximity axis under persistent co-location.
- WCSAC "cut for scope / not implemented" (two sections) — superseded.
- Cross-task transfer "in progress at the time of writing" — superseded by
  the resolved negative.
- v12 §6-style synthesis ("proactive ≫ reactive; filter redundant") —
  replaced by the regime map.
- "Proximity was the wrong metric" phrasing — never present, actively
  guarded against; the residual "primary safety metric" phrases were
  removed in the final audit.

## Page-budget measures (71 → 59 content pages)

The merged content initially hit 71 pages (limit 60; exceeding caps the
Communication mark). Reclaimed via: compact heading spacing (titlesec) and
caption/float-spacing setup; float-page discipline
(`\floatpagefraction=0.85` etc.); figure-width reductions; dropping three
redundant floats (prox-calib CDF figure — table retained; filter-taxonomy
scatter — table retained; Lagrangian-formulations TikZ — prose retained) and
two tables converted to prose (curriculum stages, perception-mode matrix);
merging velocity-axis + joint-coverage into one two-panel figure; the
planned regime *table* absorbed into the F6 schematic caption; and ~25
prose compressions (no numbers or claims removed). All dropped-float labels
were cleaned; a final audit confirmed every remaining figure/table is
referenced in text.

## Figures (final inventory)

| Figure | Status |
|---|---|
| fig4_exogenous_proximity | ADDED — R0 first figure, provenance caption (goalpost defence) |
| fig1_method_comparison | ADDED — R2 headline, drawers-Lagrangian defence in caption |
| fig2_tradeoff_curve | ADDED — R2, dominance wording task-scoped per E7 |
| fig3_reduction_bars | ADDED — Summary of Results (marker-facing) |
| fig5_cross_task_boundary | ADDED — R3 centrepiece ("bend down vs slide along the diagonal") |
| fig7_gate_activity (NEW) | BUILT from exact per-step intervention data (not a proxy); numbers in FIGURE_DATA_NOTES.md |
| F6 regime-map schematic (NEW) | TikZ in R3; evidence refs in caption |
| velocity_axis + joint_coverage | MERGED into one two-panel figure (R1) |
| separation_render_grid, fixlam, lambda_regimes, hybrid-overview TikZ, E4.3 pgfplots | KEPT (resized) |
| filter_taxonomy.png, prox_calib_row5.png, Lagrangian-options TikZ | DROPPED (data preserved in tables/prose) |
| fig1–fig5 | vector PDFs generated (used in preference to PNGs by pdflatex) |

## References (see REFERENCES_AUDIT.md for per-entry detail)

- 46 existing entries audited: 29 verified, 11 corrected, 6 replaced
  (fabricated/placeholder metadata): shield2025 (real authors Yang/Werner/
  Cosner/Fridovich-Keil/Culbertson/Ames, IROS 2025), cbfrl2024 (real paper
  arXiv:2510.14959), auditing2021 (sole author Bharadhwaj, CoRL 2021),
  permissivefilter2025 (Oh/Nguyen/Hu/Fisac), latentcbf2025 (Nakamura et
  al.), dontfreeze2026 (Zhang/Xu/Aggarwal). 0 unverifiable.
- 5 ADDED (all verified): alshiekh2018shielding, trautman2010freezing,
  bansal2017hamilton, seo2025actionsequence (the CQN-AS paper — now cited
  alongside the CQN paper at every backbone mention), marvel2017implementing.
- Prose updated where corrected author lists contradicted it.

## Build environment notes

- Local TeX Live lacked `algorithm.sty`/`algorithmicx`/`siunitx`/
  `IEEEtran.bst`: `algorithmicx.sty`, `algpseudocode.sty`, `IEEEtran.bst`
  fetched from CTAN into this directory; `algorithm` replaced with a
  float.sty shim in the preamble; unused `siunitx` dropped. On Overleaf,
  these vendored files are harmless (local files shadow distribution ones).
- Compile: `pdflatex && bibtex main && pdflatex && pdflatex` (clean: zero
  errors, zero undefined refs, zero bibtex warnings).

## Items for the author to verify / decide

1. **Compute estimates**: the intermittent-arc + WCSAC GPU-hours
   (≈230–280 h) and the resulting ≈40 kg CO2e are clearly-marked estimates —
   replace with run-log numbers if available (Sections 5.10, Appendix E,
   Declarations).
2. **W6 persistence dial**: currently the named next step in Future Work
   (per plan "decisions for the author": run vs defer — deferred).
3. **fig5 placement**: appears once (R3), not echoed in the intro (page
   budget; plan left this open).
4. **Title** unchanged from v12 — consider whether the regime-map framing
   should reach the title/subtitle.
5. The F7 matched-R comparison uses different critics per task (v3 saucepan
   vs v1 dish/drawers, mean-q scales differ) — defensible (R swept per task,
   knees all at 2.75–3.0) and disclosed in FIGURE_DATA_NOTES.md; the
   operating-point comparison shows the same contrast (51–63% vs ~27%).
6. GenAI declaration (Appendix C) kept from v12 — review that it still
   reflects your actual usage for this revision.

# Report rewrite log: v12 (saucepan arc) → v13 (regime map)

Reframing per `docs/report_reframing_plan.md` (2026-06-10), targeting the
markscheme in `report guide/`. Backups: `main_v12_backup.tex`,
`references_v12_backup.bib`. Final state after the 2026-06-11 reference
modernisation (see last section): **87 PDF pages, 60 content pages
(chapters 1–8, printed pp. 11–70) — at the 60-page limit, not over**; clean
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
6. GenAI declaration was strengthened in the late merge pass below; do a final
   personal accuracy check before submission, because only you can certify the
   exact tool usage history.

## 2026-06-10 late merge pass: best-of GPT into Fable master

Purpose: keep `FYP_v12_fable` as the master report while integrating the strongest
mark-scheme-facing elements from `FYP_v12_gpt`.

### Kept

- Kept the Fable regime-map structure, figures, page-budget work, WCSAC integration,
  verified references, and compiled-report orientation.
- Kept Fable's Chapter 4 architecture, CQN-AS integration, critic-gated speed-scaling
  method, and ablation-heavy R2/R3 results as the master source.

### Added / Modified

- Added a more explicit CMDP-to-value-based-Lagrangian derivation in Chapter 4:
  the constrained objective, Lagrangian relaxation, separate $Q_r/Q_c$ targets, and
  the reason $\lambda$ should enter action selection rather than the reward critic's
  Bellman target. This strengthens the reproducible mathematical explanation.
- Added a prominent `Ablation Map: What Each Component Proves` section before the
  detailed results, summarising E1.1, E3.1--E3.3, E3.2, E2.4--E2.6, and the gate
  critic/label/DAgger ablations.
- Expanded `Broader Impact` to discuss misuse risk, domestic/unconstrained
  deployment, vulnerable users, and why a safety wrapper is not an ISO certificate.
- Strengthened the Declarations GenAI paragraph with specific uses and verification
  steps.
- Rewrote Appendix C (`Use of Generative AI`) into a structured disclosure covering
  literature triage, code scaffolding, experiment-log summarisation, LaTeX/prose
  drafting, and observed tool limitations.

### Lost

- No report results, figures, tables, or citations were removed in this merge pass.

### Validation Notes

- Needs a fresh PDF compile/page count after these additions, because Fable was
  previously at 59 content pages and this pass adds text.

## 2026-06-11 abstract / intro / reference freshness pass

### Modified

- Rewrote the abstract for readability around the classical-safety vs learned-humanoid
  safety contrast: lane keeping, adaptive cruise control, caged arms, and
  speed-and-separation interlocks are now used to set up the greenfield problem.
- Removed the family/friends acknowledgement and added the SWIRL research lab plus
  Imperial College Department of Computing for advice, feedback, and GPU compute.
- Tightened the introduction's motivation prose and removed the stale `Figure~01`
  reference.
- Rewrote the "robot learning safety record" paragraph so Brunke 2022 is not
  misattributed as making a realistic-perception/humanoid claim.
- Rephrased the SHIELD discussion to match its actual scope: runtime safety for
  humanoid locomotion/navigation, not manipulation under sustained co-working.
- Added Wachi et al. 2024 as a modern safe-RL survey so current-state claims do not
  rely only on pre-2022 sources.
- Rewrote RQ3 and RQ4 in plainer language.
- Rewrote the contributions list for readability and explained critic-gated
  speed-scaling in the terminology paragraph.
- Replaced the vague "regime map says which part carries the safety burden where"
  wording with a clearer explanation: the regime map links the human--robot
  co-location pattern to the safety mechanism that works.

### Reference audit

- Added and verified `wachi2024constraint`; updated `REFERENCES_AUDIT.md` counts
  from 51 to 52 entries.

### Still to check

- Recompile and re-check page count after these prose additions.

## 2026-06-11 final readability / reference freshness pass

### Modified

- Rewrote the abstract again for a colder reader:
  - Opens with the contrast between classical safety engineering (lane keeping,
    adaptive cruise control, caged arms, speed-and-separation interlocks) and
    learned humanoid manipulation.
  - Explains why the project is greenfield: benchmark, cost signals, perception,
    training method, runtime filter, and evaluation protocol all have to be built
    together.
  - Keeps the regime-map result, but in shorter sentences.
- Rewrote the first motivation pages to make the classical-vs-learned safety
  contrast easier to follow.
- Removed the stale `Figure~01` reference.
- Removed family/friends from acknowledgements and added SWIRL + Department of
  Computing for feedback/advice/GPU compute.
- Fixed the use of references 13/14:
  - Brunke 2022 now supports broad safe-learning context, not a specific
    humanoid/perception claim.
  - SHIELD 2025 is scoped to humanoid locomotion/navigation and used to motivate
    the analogous manipulation gap.
- Added Wachi et al. 2024 (`wachi2024constraint`) to support current-state safe-RL
  constraint-formulation claims; updated `REFERENCES_AUDIT.md` to 52 entries.
- Rewrote RQ3 and RQ4 in plain language.
- Rewrote the contributions list to remove dense jargon and explain the key terms.
- Replaced vague/internal wording such as "which part carries the safety burden
  where", "load-bearing", "byte-for-byte", "else-branch fired", and the conclusion
  scaffold headings.

### Validation

- `ReadLints` reports no issues.
- Active `main.tex` has no matches for stale terms such as `Figure~01`,
  `family and friends`, `to our knowledge`, `working draft`, `VERIFY`, `Hardik`,
  `load-bearing`, `byte-for-byte`, or `else-branch`.
- Generated `.aux/.lof/.lot` files may still contain old wording until the next
  PDF compile.

## Reference modernisation (2026-06-11)

Per `REFERENCE_MODERNISATION_PLAN.md`: every pre-2022 reference was checked
against the 2022–2026 literature (14 claim-cluster web sweeps + adversarial
refutation of the negative-existence claims + double verification of all new
sources against primary pages). 40 entries added to `references.bib`
(52 → 92, all cited in text). Net content growth: +1 page (59 → 60, at the
limit).

### Claims that changed (overclaims corrected)

- **"No work tests both safe-RL families on the same human-co-located
  manipulation benchmark"** — refuted as written by Thumm, Pelat & Althoff
  (IROS 2023): PID-Lagrangian + provably safe shield on a 6-DoF reaching task
  beside a replayed human. Intro bullet 2 and §2.4.1 now cite it as the
  nearest cross-family evidence, scope the gap to "across co-location regimes
  on a high-DOF manipulator under ISO 15066", and use their PID-Lagrangian
  failure as independent corroboration of the intermittent-regime result
  (also added to the conclusion).
- **Benchmark gap (Table 2.1 + prose)** — five rows added (Safety-CHORES,
  Assistive Gym, RCareWorld, Habitat 3.0, Human-Robot Gym). Human-Robot Gym
  (ICRA 2024) breaks the blanket "separate from manipulation with a live
  human coworker" and "do not model perception at all" phrasings — both
  rewritten with it named as the nearest exception (mocap human, shield-mode
  SSM/PFL, optional uncalibrated noise, 6–7-DoF arms). The five-axis
  intersection claim survives and is now stated against the strongest
  near-miss. §2.7 prose reorganised into three axes (safety / manipulation /
  human).
- **"Scaling a hand-engineered CBF-QP to 76 DOF is an open research
  problem"** — too strong. §2.4 now concedes solver cost at moderate scale
  (Khazoom 2022: 15 pairs, ~0.2 ms, 24-DOF sim; Morton & Pavone 2025:
  hundreds of constraints at kHz on a 7-DOF arm) and re-anchors the argument
  on what remains undemonstrated: every human-link × robot-link pair on a
  humanoid at control rate (Bena 2025 reduced-order; SPARK upper-body
  safe-set; Cai 2026 sim-only 72 constraints at ~33 Hz). Mirrored in §7.5.
- **"Not ISO-certified as a collaborative robot"** — category abolished by
  ISO 10218:2025 (collaboration is a property of the validated application;
  Hartmann et al. 2026). Intro bullet 1, conclusion, and the `unitree_g1` bib
  note rephrased to "validated for collaborative operation", with the
  Kóczi & Sárosi scoping review and the IEEE Humanoid Study Group pathway
  report as citable backing and an AiMOGA CE-certification footnote
  (company claim) to pre-empt the obvious objection.
- **"MuJoCo forces are not safety-validated against real collision
  measurements"** — narrowed to "we found no validation … against measured
  human–robot collision forces", with the actual evidence cited: Acosta 2022
  (trajectory-level only, MuJoCo stiffness-insensitive), Joseph & Dutta 2026
  (force-level but leg–terrain, outside HRC), Schlotzhauer 2022 (industrial
  PFL validation uses physical biofidelic measurement, distrusts simulated
  constrained contacts).
- **"Either hand-engineered CBFs or learned safety value functions"** —
  incomplete dichotomy. §2.4 now lists predictive filters as the third family
  (Wabersich 2023) and cites Hsu/Hu/Fisac 2024 as the unifying review.

### Claims that held (modern citations added, wording minimally touched)

- Binary-indicator cost convention (+AutoCost AAAI 2023, Safety-Gymnasium).
- Shielding anchor (+Könighofer CACM 2025), HJ foundation (+Ganai 2024).
- Sim-to-real dynamics-gap framing (+Muratore 2022, Radosavovic 2024).
- Curriculum standardness (+Rudin PMLR 164, Humanoid Parkour CoRL 2024 — the
  latter on the same Unitree H1 platform). "The standard solution" → "a
  standard solution".
- CQL "the standard" → "a standard" (+Prudencio TNNLS 2024) and the
  cost-critic pessimism design grounded as established practice (+CPQ AAAI
  2022).
- SMPL standardness (+Tian TPAMI 2023); BodySLAM++ design choice defended
  against the newer world-grounded wave (WHAM, TRAM: offline GPU pipelines).
- WCSAC canonical-baseline status (+journal extension, Yang et al. MLJ 2023,
  also cited at the reimplementation and in Appendix A).
- CVaR future work reframed from literature gap to pipeline-specific gap with
  named instruments (OffTRC, SDAC, SRCPO).
- Safety-evaluation-attention claim (Bharadhwaj 2021) now backed by 2025
  audits (RoboPAIR ICRA 2025, Hundt et al. IJSR 2025).
- SSM/PFL measurement lineage extended (Svarny RCIM 2022 2,250-collision
  campaign; HARMONIOUS T-RO 2025 humanoid successor; ISO/TS 15066 → ISO
  10218-2:2025 Annex M absorption noted at the PFL table).

### Style

- Zero new semicolons introduced; two removed (§2.4.1 gap paragraph).
- Two "to our knowledge" hedges avoided in favour of "we found no …"
  (the phrase is on the stale-terms list from the v13 validation pass).
- UK English throughout; all new sources double-verified against primary
  pages before entering `references.bib` (see plan, Part D).

### Validation (2026-06-11)

- pdflatex + bibtex + pdflatex ×2: zero errors, zero undefined
  references/citations, zero bibtex warnings.
- Stale-terms grep (`to our knowledge|load-bearing|byte-for-byte|
  else-branch|working draft|VERIFY`): zero matches in `main.tex`.
- 92/92 bibliography entries cited in text. Content pp. 11–70 (60/60).

## Reference dedup + em-dash removal (2026-06-11, second pass)

### Redundant references removed (92 → 87, all remaining entries cited)

Removal rule: an entry is redundant only if every claim it supports is fully
carried by another reference at the same citation site.

- `zhao2020sim2real` (SSCI 2020 survey) — superseded by Muratore et al. 2022
  in both of its clusters (§2.6.1 dynamics gap, §6.2.3 domain randomisation).
- `ye2022rcareworld` — identical benchmark-table profile to Assistive Gym
  (✗ ✓ ✗ ✗ ✓); the human-axis line is still carried by Assistive Gym,
  Habitat 3.0, and Human-Robot Gym.
- `wang2024tram` — second exemplar of the world-grounded estimator wave;
  WHAM alone carries the point ("estimators such as WHAM").
- `kim2024srcpo` — third instrument in the CVaR future-work sentence;
  OffTRC (off-policy fit) + SDAC (distributional) suffice.
- `rozlivek2025harmonious` — nice-to-have lineage aside in §2.1.2, removed
  with its sentence.

Considered and kept (each carries a distinct claim): hundt2025llmrobots
(pairs with robey2025 for the plural "audits"), koczi2025 + ieee2025pathway
(peer-reviewed cover + authoritative quote), bena/sun/cai (three distinct
prongs of the CBF-scalability evidence), acosta/joseph/schlotzhauer (three
prongs of the contact-fidelity claim), marvel2017 (canonical SSM
implementation), erickson2020assistivegym (kept over RCareWorld as the
recognised canonical).

### Em-dashes removed (rendered PDF now contains zero)

- Prose pair §2.4.1 ("filter---and its cost---") → parentheses.
- Section titles R1/R2/R3 and E2.6 ("R1 --- Persistent ...") → colon + parens
  form ("R1: Persistent Co-location (saucepan_to_hob)").
- Four TikZ comment markers → plain comments.
- IEEEtran.bst's long dash for repeated bibliography authors (ISO ×3,
  Unitree ×2) disabled via an `@IEEEtranBSTCTL{BSTcontrol,
  CTLdash_repeated_names="no"}` entry + `\bstctlcite{BSTcontrol}` after
  `\begin{document}` — repeated authors now print in full.
- Kóczi & Sárosi title em-dash downgraded to an en-dash (the published title
  uses an em-dash; en-dash keeps the title findable without the banned glyph).
- En-dashes (ranges, "6--7-DoF", "human--robot") are untouched: they are not
  em-dashes.

### Validation (second pass)

- pdflatex + bibtex + pdflatex ×2: zero errors, zero undefined
  references/citations, zero bibtex warnings; BSTcontrol detected by
  IEEEtran.bst.
- `grep -c -- '---' main.tex main.bbl` → 0, 0; rendered-PDF em-dash count 0.
- 87/87 bibliography entries cited. Content pp. 11–70 (60/60 limit).

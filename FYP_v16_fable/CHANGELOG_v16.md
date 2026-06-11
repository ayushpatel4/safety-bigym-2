# Report rewrite log: v15 → v16 (clarity reframe)

Goal (user brief, 2026-06-11): make the main claims and contributions
immediately apparent; restore readability lost in the v13/v15 growth;
respect the page limit with slack. **No numbers changed, no new
experiments; every result in v15 survives in v16** — re-weighted,
re-ordered, and re-titled around one thesis. Backup:
`main_v15_backup.tex`. State after this pass: 88 PDF pages, 59
content pages (superseded by the figure pass below; final state **89
PDF pages, 60/60 content pages** after the task-suite figure); clean
pdflatex+bibtex build; zero errors, zero undefined references, zero
bibtex warnings; 87/87 references cited; zero em-dashes; stale-terms
grep clean; every figure/table referenced in text.

## The one-paragraph delta

v15 stated the regime-map thesis but presented it through the
author's working taxonomy: 6 contributions, 4 RQs, experiment codes
(E1.1…E5.3), movement codes (R0–R3), option/phase codes, with the
thesis figure on ~p.55 and legacy v8-era sections interleaving the
results arc. v16 has **one thesis (stated in a box on p.14 with a
schematic regime-map figure on p.15), three research questions mapped
1:1 onto three results movements, and five contributions** ordered
regime-map-first. Experiment codes left the prose and headings
(stated once parenthetically per experiment; full mapping in a new
appendix index). The duplicated results-summary/discussion material
was merged. Title now states the finding.

## Headline changes

- **Title**: "Safety-Aware Reinforcement Learning for ISO
  15066-Compliant Humanoid Manipulation Under Sustained Co-Working" →
  **"When to Train Safety In and When to Filter at Runtime: A
  Co-location Regime Map for ISO 15066-Compliant Humanoid
  Manipulation"**.
- **Ch1**: problem statement renamed "The Problem and the Thesis";
  thesis display box + simplified regime-map TikZ schematic
  (fig:regime-map-intro) added; RQs 4 → 3 (persistent / intermittent /
  boundary, each naming its results section); contributions 6 → 5
  (regime map first; C-codes dropped); outline rewritten for the new
  chapter structure; motivation lightly compressed; goal list item 2
  reworded ("bounds" → "reports … so that whatever the system cannot
  control is measured rather than hidden" — matches the exogenous-tail
  finding).
- **Chapter split**: old Ch5 "Experimental Setup and Results" (22.8pp)
  → Ch5 "Measuring Safety Beside a Person: Tasks, Axes, and Protocol"
  (setup + two-axis protocol + statistics + compute + baselines) and
  Ch6 "Results: Two Regimes, One Law".
- **Results arc un-broken**: the legacy "RQ1: Obs Channel" and "RQ4:
  Tail & OOD" sections that interrupted the regime story were
  dissolved: E1.1 now opens the persistent movement ("Observation
  Alone Is Not a Safety Mechanism"); E3.6/E5.1/E5.2 now close it
  (perception robustness, exogenous tail, held-out coworker — the
  "how far the result stretches" block, ending on the cross-task
  negative that bridges into the intermittent movement).
- **Claim-first sections**: each movement opens with its claim in
  italics; the chapter opens with "The Shape of the Evidence"
  (roadmap + announced pre-registered prediction + condensed ablation
  map).
- **Future Work merged into Ch8** "Conclusions and Future Work"; 6 →
  4 directions (persistence dial first, with the
  proactive-avoidance question folded in; PFL force loop; "Stronger
  Guarantees" merging CVaR-Lagrangian + formal filters; real-robot
  transfer). Conclusion opening now echoes all five contributions in
  introduction order.
- **Pre-existing v15 bug fixed**: the `\section` heading for the
  Value-Based Lagrangian was missing (its `\label` sat orphaned at the
  end of the WCSAC section, so the entire Lagrangian derivation
  nested under "External Baseline: Worst-Case SAC" and
  `\ref{sec:method:lagrangian}` resolved to the wrong section).
  Heading restored; WCSAC section moved after the Lagrangian section
  (its v13-intended position).

## Compressed (numbers preserved inline; ~6pp gross)

- Results-summary ↔ Discussion "RQs Revisited" duplication: summary
  rewritten as three claim paragraphs (one per RQ) + fig3; discussion
  section rewritten as "What the Answers Mean" (interpretation only:
  IL-channel lesson, cost-signal-density hypothesis, honest-claim
  framing, exogenous-tail implication).
- E1.1 BC obs-ablation → ~0.5pp prose, table cut (key cells inline).
- E3.1 cost-form + E3.2 budget sweep → one subsection ("The Cost
  Budget, Not the Cost Form, Is the Operative Variable"), E3.1 table
  cut (numbers inline), figure kept.
- Reactive-fallback sub-study table cut; verdict + all numbers folded
  into one paragraph (discussion ref updated to section pointer).
- E5.2 OOD table → prose (both numbers + over-constraint caveat kept).
- Experiment-index table → new **Appendix C "Experiment Code Index"**.
- Ch2 RL fundamentals: duplicated Diffusion-Policy justification
  removed; CQN-AS provenance sentence tightened.
- Ch3: coworker-intro tightened (incl. a "scenarious" typo fix);
  calibration closing paragraph tightened.
- Ch4: software-architecture + metric-persistence merged into one
  section; compute budget → one paragraph (totals + per-arc split
  kept; derivation in sustainability appendix).
- Threats: marker-addressing opener replaced; computational-
  feasibility trimmed by a third.
- Ch5 protocol: perception-mode rationale and baselines enumeration
  tightened.
- Figure widths reduced (separation grid 0.8→0.7, fig1 0.85→0.75,
  fig2/fig3 0.78→0.7, fig5 0.8→0.72, fig4 0.68→0.62, fig7 0.62→0.58,
  lambda/fixlam 0.6→0.55).
- Two overfull appendix hyperparameter tables wrapped in resizebox;
  long-token overfull in the implementation paragraph fixed.

## Code/terminology sweep

- E-codes/R-codes/C-codes/Phase-codes removed from headings and
  running prose (62 → 37 body E-mentions, all now first-mention
  parentheticals, table-caption traceability tags, or comments;
  R0–R3 and Phase~n: zero in body prose).
- RQ renumbering propagated everywhere (old RQ2→RQ1, RQ3→RQ2,
  RQ4→RQ3; old RQ1 demoted into the persistent movement's opening) —
  verified by grep: every remaining RQ mention uses the new scheme.
- "byte-identical", "marker-facing", "Phase~2 SVF" and similar
  internal register removed from body prose.
- House style preserved: no new semicolons in edited passages, zero
  em-dashes, UK English.

## What did NOT change

Every quantitative result, table value, figure, citation (87
verified entries), the pre-registered rules and their outcomes, the
WCSAC baseline, the statistical methodology, all nine threats, the
support-bound proposition + proof, the deployment-faithfulness
section, the GenAI/ethics/sustainability declarations, and the
reproducibility appendix.

## Validation (2026-06-11)

- pdflatex + bibtex + pdflatex ×2: zero errors, zero undefined
  references/citations, zero bibtex warnings.
- 87/87 bibliography entries cited; every figure/table referenced.
- `grep -c -- '---' main.tex main.bbl` → 0, 0.
- Stale-terms grep (to our knowledge|load-bearing|byte-for-byte|
  else-branch|working draft|VERIFY|marker-facing) → 0.
- Headline numbers consistent across abstract / intro / results /
  conclusion (22.8%, 0.296→0.228, 0.76, −50%/−22%, 61.5% vs 19–27%,
  ≈42%, 2.46→6.03, 0.85→0.44).
- Content pages: 59/60 (printed pp. 12–70); 1 remaining overfull is
  0.85pt in a bibliography entry (negligible).

## Items for the author

1. **Confirm the new title with your supervisor** (the interim report
   carried the old title).
2. Re-certify the GenAI declaration personally before submission
   (unchanged from v15, but only you can certify tool-usage history).
3. The page count is 59, not the plan's stretch 54–55: the remaining
   distance is held by mark-bearing material the plan marked
   do-not-cut (floats, protocol, statistics, threats). Under the
   limit with slack; padding (duplication, legacy scaffolding) is
   gone. Deeper cuts are possible but would start trading Execution/
   Evaluation evidence for pages.
4. `NUMBERS_CHECKLIST.md` in this directory is the canonical
   headline-numbers audit sheet; re-run its greps after any edit.

## 2026-06-11 figure pass (Dr James's advice: figure-first readability)

Brief: "more figures the better; each figure/table self-contained with
long captions; a reader scans the figures first."

### Added

- **Task-suite scene figure** (`task_suite.png`, Figure in Ch3 task-suite
  section): 1x3 mid-episode stills from live baseline rollouts of all
  three tasks, H1 + G1 coworker both in frame, label bars carrying task,
  regime, and the ground-truth closest-pair separation at the pictured
  instant. Fills the report's biggest figure gap: previously a
  figure-skimming reader never saw the system or the coworker outside the
  frozen calibration grid. Provenance: `task_suite_manifest.json` +
  FIGURE_DATA_NOTES.md (new section); generation script
  `scripts/make_regime_figures.py` (new, repo).

### Considered and rejected

- A per-step separation trace/barcode figure of the two regimes: in a
  6-episode sample the raw-separation pattern does not cleanly separate
  the regimes (sampling variance + the regime variable being critic-risk,
  not raw bands); including it would have muddied the thesis. Artifacts
  preserved at `results/figs/regime_rollouts/` with the reasoning in
  FIGURE_DATA_NOTES.md.

### Caption self-containment pass

- Algorithm float caption expanded (was 7 words) to state the value-based
  selection rule, per-step backup, and fixed-lambda rationale.
- Two-panel axis-division wrapper caption expanded (was 13 words) to carry
  conditions and the composition-ceiling takeaway.
- Deployment-config appendix caption now states the per-regime
  recommendation; "Marker-facing summary" wording removed earlier remains
  out.
- Audit: all remaining captions >=30 words and condition-bearing.

### Page budget

The figure cost ~1pp; reclaimed via separation-grid width 0.7->0.62 and
~20 lines of Ch8 tightening (conclusion recap, future-work prose; no
numbers or citations lost; thumm2023interventions still cited 2x, the
five positioning citations remain cited 5-13x elsewhere). Final: 89 PDF
pages, content printed pp. 12-71 = 60/60 content pages (at the limit, not
over; the last content page carries one subsection). Clean build: zero
errors / undefined refs / bibtex warnings; 87/87 citations; zero
em-dashes; stale-terms grep clean; every figure/table referenced.

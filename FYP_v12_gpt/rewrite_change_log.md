# FYP v12 GPT Rewrite Change Log

Date: 2026-06-10

## Kept

- Kept the eight-chapter report skeleton: Introduction, Background, Benchmark, Method, Results, Discussion, Future Work, Conclusion.
- Kept the benchmark and method chapters largely intact, including `safety_bigym`, BodySLAM-style noisy perception, CQN-AS integration, SVF, speed scaling, C51 support-bound proposition, and deployment-faithful harness discussion.
- Kept the mandatory Declarations section before references, including originality, GenAI, ethics, sustainability, and code/data availability.
- Kept the reproducibility appendix, seed/CIs framing, PFL qualification, and code/data availability appendix.
- Kept saucepan fixed-lambda constrained-RL evidence: 22.8% proximity reduction at 0.76 success, PID-lambda instability, cost-budget feasibility framing, and exogenous proximity floor.
- Kept the reactive freeze-vs-flee taxonomy as a persistent-co-location result, not as the whole thesis.

## Added

- Added `\graphicspath{{./}{../docs/figures/}}` so the new report figures can be referenced from `docs/figures`.
- Added a new regime-map abstract: safety decomposes into proximity exposure and robot-controllable SSM velocity; persistent vs intermittent co-location decides which arm works.
- Added new RQs matching the regime-map framing.
- Added new contribution list:
  - `safety_bigym` benchmark and deployment-faithful evaluation.
  - Two-axis safety protocol.
  - Hybrid Safety Critic architecture.
  - Co-location regime map.
  - Mechanism-level lessons.
- Added a two-axis metric protocol in the metrics section.
- Added new compiled Results sections:
  - R0: Evaluation Protocol and Two Safety Axes.
  - R1: Persistent Co-location Requires Anticipatory Avoidance.
  - R2: Intermittent Co-location Favours Critic-Gated Speed Scaling.
  - R3: The Co-location Regime Map.
- Added existing new figures into the compiled report:
  - `fig4_exogenous_proximity.png`
  - `fig1_method_comparison.png`
  - `fig2_tradeoff_curve.png`
  - `fig3_reduction_bars.png`
  - `fig5_cross_task_boundary.png`
- Added new tables:
  - Axis ownership table.
  - Persistent saucepan feature table.
  - Fixed-lambda seed breakdown.
  - Saucepan gated-speed-scale sweep.
  - Intermittent task method comparison.
  - Recommended gated operating points.
  - Regime-map summary table.
- Added a new Discussion aligned with the regime map:
  - Metric-reframing defence.
  - PFL limitation.
  - Regime-map scope.
  - WCSAC/external-baseline limitation.
  - Curriculum and sim-to-real limits.
  - Broader impact and reproducibility.
- Added Future Work items:
  - Persistence-dial experiment.
  - Proactive avoidance beyond expected-cost Lagrangians.
- Added a new conclusion centred on the regime map.

## Modified

- Changed title from a general ISO-compliant humanoid manipulation title to `A Hybrid Safety Critic for ISO 15066-Referenced Humanoid Manipulation Under Human Co-Working`.
- Changed the thesis spine from "proactive policy plus costly composition" to "co-location regime map".
- Reframed Hybrid Safety Critic as the two-arm architecture:
  - Anticipatory policy learning for persistent co-location and proximity exposure.
  - Critic-gated speed scaling for intermittent co-location and SSM velocity.
- Reframed proximity:
  - Not "wrong metric".
  - Now "exposure metric, partly exogenous".
  - SSM-velocity is primary for runtime filters because it is robot-controllable.
- Reframed saucepan C4:
  - Not hidden or ignored.
  - Now the persistent-regime ceiling: both-axis coverage is possible, but only at stacked task cost.
- Reframed dishwasher/drawers:
  - Now the positive runtime-filter result.
  - Critic-gated speed scaling improves SSM without the severe collapse of unconditional speed scaling.
- Reframed constrained-RL failures on dishwasher/drawers:
  - Scoped to the current expected-cost Lagrangian recipe and intermittent tasks.
  - Not a universal claim that constrained RL cannot work.
- Updated bibliography metadata:
  - Fixed `cbfrl2024` metadata to the verified 2025 arXiv CBF-RL paper.
  - Replaced placeholder `VERIFY` authors/titles for `permissivefilter2025`, `latentcbf2025`, and `dontfreeze2026`.

## Removed From Compiled Report

- The old Results body is retained in source inside `\iffalse ... \fi` for traceability, but no longer compiles.
- The old Discussion body is retained in source inside `\iffalse ... \fi`, but no longer compiles.
- The old Conclusion body is retained in source inside `\iffalse ... \fi`, but no longer compiles.
- Removed the compiled headline that the SVF/filter is globally redundant.
- Removed the compiled headline that proactive constrained RL is globally the winner.
- Removed stale "WCSAC not implemented / cut for scope" phrasing from compiled prose.
- Removed placeholder bibliography `VERIFY` entries.

## Not Done / Still Needs Human Compile Check

- Could not compile locally because `pdflatex`, `latexmk`, and `tectonic` are not installed in this environment.
- The hidden old blocks should be deleted entirely before final submission if you want the `.tex` source itself to be clean, but they do not compile into the PDF.
- The new figure paths assume the `docs/figures/*.png` files are present relative to `FYP_v12_gpt`.
- Page count still needs checking after PDF compilation.

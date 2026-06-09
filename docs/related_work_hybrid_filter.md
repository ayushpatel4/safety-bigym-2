# Related work: can a reactive runtime filter Pareto-improve over a proactive policy?

Deep-literature survey (2026-06-09), framed at our exact question: against a *co-located,
actively-approaching, partly-unpredictable* human, can a hybrid (proactive learned policy +
reactive runtime filter) beat the policy alone — better separation / ISO-15066 SSM at
**comparable** task success — rather than just trading success for safety (freeze-vs-flee)?

## Verdict (one line)

**No published work demonstrates such a Pareto improvement for our setting, and the theory
predicts a reactive filter cannot achieve it against an unpredictable human.** Our negative
hybrid result is therefore *consistent with the entire verified literature*, and the
ISO-15066 manipulator setting is an explicit open gap the project is positioned to fill.

## The decisive facts (verified by 3-vote adversarial checking)

| # | Finding | Citation | How it maps to our results |
|---|---|---|---|
| 1 | **Every empirical "filter Pareto-improves over policy" claim failed verification** — 7 such claims refuted (ARMS success+separation; PSS-Social shaping; LatentCBF "2× completion"; latent-HJ block-manip; confidence-aware BRT −20% interv +27.75% reward). | (see Refuted list) | We are not failing where others succeeded — **no one has a verified success.** Our "hybrid counterproductive / joint-coverage-at-a-cost" is the field norm. |
| 2 | **Decisive counter-evidence.** Worst-case HJ filter is over-conservative ("inhibit the robot's ability to make progress by intervening excessively"). Trusting a human model to cut conservatism is *unsafe* — the model-trusting ablation hit a **25% collision rate** vs an unmodeled (distracted) human; the safe variant stays collision-free **only by reverting to the conservative worst-case filter** when the human deviates. → low task cost **only while the human follows the model**. | **arXiv:2109.14700** (Tian, Bajcsy, Tomizuka, Dragan; ICRA 2022) | This *predicts our result*: our coworker is adversarial/unpredictable (~42% exogenous proximity), so any reactive filter must either over-veto (freeze — our SVF) or revert to conservatism. No free lunch. **The single most important citation for our Discussion.** |
| 3 | **Separation/optimality theorem:** a *least-restrictive* (maximally permissive) filter need not degrade asymptotic RL return — but **only if** the filter is least-restrictive; an over-conservative filter breaks the equivalence. Recipe: "deploy with the most permissive filter available." Navigation-only, asymptotic, vs best *safe* policy; "does not scale" to high-dim. | **arXiv:2510.18082** | Explains *why our SVF hybrid failed specifically*: it was a learned veto OOD on the avoiding policy → **over-conservative (48% over-veto)** → broke the no-degradation guarantee. The theorem's own condition is exactly what our SVF violated. |
| 4 | **The field's own motivation states the freeze/flee problem:** existing latent safety filters use "'least-restrictive' filtering that discretely switch between nominal and safety policies, potentially undermining the task performance that makes modern visuomotor policies valuable" (task completion degraded to 14–60%). | **arXiv:2511.18606**, **arXiv:2502.00935** | Our freeze-vs-flee taxonomy is the filter designers' *own* stated central failure mode — not an artifact of our implementation. |
| 5 | **Proactive RL structurally avoids freezing** where reactive analytical solvers don't: an RL crowd-nav policy froze <1% / >99% goal-reaching, while ORCA (even with ground-truth) froze and fell to ~75% success as density rose. | **arXiv:2603.06729** | Direct analogue of our "proactive policy is the graceful frontier; reactive is freeze/flee-bounded" (2D nav → manipulation by analogy). |
| 6 | **CBF-QP failure modes are field-wide consensus:** myopic, conservative, nonpersistent feasibility, infeasibility/deadlock. A real Franka FR3 CBF filter resolves infeasibility by *deviating around* the human rather than stopping — but reports only qualitative evidence (no paired safety-vs-throughput numbers). | **arXiv:2603.00338**, **arXiv:2505.16055**, **arXiv:2310.05865** | Backs our CBF base-dodge / EE-retract "flee" results — the conservatism/deadlock we saw is the documented norm. |
| 7 | **ISO-15066 SSM/PFL velocity-scaling-on-a-policy against an approaching human: no verified published Pareto result exists.** "The open gap the project itself appears positioned to fill." | (absence across 18 confirmed claims) | Our division-of-labour result (policy→proximity, speed-scale→velocity, both-axis hybrid at a tunable cost) is **novel**, not a failure. |

## Refuted Pareto-improvement claims (all 0-3 or 1-2 on verification)

- ARMS task success 82.5% vs 79.4% RL-only — **0-3** (arXiv:2601.16686)
- ARMS separation 0.690 m vs 0.660 m — **0-3** (arXiv:2601.16686)
- PSS-Social shaping 79.6%→86.4% safe-success — **1-2** (arXiv:2603.06729)
- LatentCBF "doubles task completion" over switching — **0-3** (arXiv:2511.18606)
- Latent-HJ "safest while completing task" (block manip) — **1-2** (kensukenk.github.io/latent-safety)
- Confidence-aware BRT −20% interventions + 27.75% reward — **0-3** (arXiv:2109.14700)
- "Safety-performance trade-off is not inherent" (general empirical) — **0-3** (arXiv:2510.18082)

## The one genuinely novel future direction the survey surfaces

Open question (verbatim): *"Can the confidence-aware fallback principle (trust the human model
only with online misspecification detection, revert to conservative worst-case otherwise) be
combined with ISO-15066 graded velocity-scaling so that the conservative fallback is
'slow-down' rather than 'freeze' — yielding low task cost even when the human is unpredictable
— and has anyone measured this?"* → i.e. an **anticipatory filter whose fallback is graded
slow-down, not a freeze**. Unmeasured in the literature; a sharp, citable future-work target.

## Honest caveats (state these in the report)

- Strongest sources are **off-domain** (2D nav, autonomous driving, legged/ground robots).
  Only the Franka CBF-QP and latent-HJ family are real high-DoF manipulators with an
  approaching human — and those report *mechanism*, not paired Pareto numbers. The exact
  manipulator-HRC/ISO-15066 setting is the gap.
- Most primary sources are **2025–2026 preprints** (not all peer-reviewed).
- The separation theorem is **asymptotic, navigation-only, conditional** on a perfect filter
  the authors concede "does not scale" to high-dim systems.

## Key bibliography (arXiv IDs)

- 2109.14700 — Confidence-aware game-theoretic HJ safety (Bajcsy et al., ICRA 2022) — **counter-evidence**
- 2510.18082 — Least-restrictive filter / performance-safety separation theorem
- 2511.18606 — LatentCBF (latent safety filters)
- 2502.00935 — Latent-space HJ reachability (Nakamura/Peters/Bajcsy, RSS 2025)
- 2505.00779 — UNISafe (CoRL 2025)
- 2603.06729 — "Don't Freeze, Don't Crash" (proactive RL vs ORCA freezing)
- 2603.00338 — Layered (multistage) safety filter
- 2505.16055 — Hierarchical CBF on Franka FR3 (Maithani et al.)
- 2601.16686 — ARMS (soft-blend RL + MPC-QP)
- 2310.05865 — Multiple-backup CBF (Janwani et al.)

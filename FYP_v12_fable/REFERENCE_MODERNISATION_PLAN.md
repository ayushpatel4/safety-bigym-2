# Reference Modernisation Plan — FYP_v12_fable → v13

> **STATUS: EXECUTED 2026-06-11.** All Part A and Part B edits applied to
> `main.tex` (E1–E23; E11 optional SPARK mention folded into E10's rewrite;
> kirschner2024edged left as optional, not applied). Tier 1 + Tier 2 + four
> Tier 3 sources (wang2024tram, schlotzhauer2022pfl, joseph2026mujoco,
> kim2024srcpo) added to `references.bib` (52 → 92 entries, all cited).
> Clean build, zero undefined references, content pp. 11–70 (60/60 limit).
> Two "to our knowledge" hedges replaced with "we found no …" (stale-terms
> list). See CHANGELOG_v13.md, "Reference modernisation (2026-06-11)".

**Date:** 2026-06-11
**Scope:** every reference in `references.bib` older than 2022 (28 of 52 entries), plus every time-sensitive claim those references support in `main.tex`.
**Method:** 14 claim-cluster web sweeps over the 2022–2026 literature + independent adversarial refutation of the four negative-existence claims + a second-pass metadata verification of all 42 candidate sources against primary pages (arXiv/DOI/publisher/proceedings). Every new source below carries a ✅ (double-verified: found, metadata exact or corrected, and confirmed to support its intended use). No source goes into `references.bib` from memory.
**Companion doc:** `REFERENCES_AUDIT.md` (2026-06-10) fixed *metadata* of existing entries; this plan fixes *staleness*.

Line numbers refer to `main.tex` as of v12 (they will shift as edits are applied — anchor by the quoted text, not the number).

---

## 0. Executive summary

| Verdict | Count | What it means |
|---|---|---|
| **Claim must change** (overclaims or describes an abolished category) | 6 clusters (E1–E12) | Text edits required; new refs cited |
| **Claim holds, needs modern citation** | 10 clusters (E13–E22) | Add-only: insert 2022–2026 cites, minimal or no text change |
| **Keep as-is** (canonical origin/method/asset citations) | 22 pre-2022 keys | No action |

The four negative-existence claims fared as follows under adversarial checking:

1. **"No benchmark covers the five-axis intersection" — survives strictly, but the supporting prose overclaims.** Human-Robot Gym (Thumm, Trost & Althoff, ICRA 2024) [V1] is a safe-RL *manipulation* benchmark with a *moving mocap-replayed human*, ISO-grounded *SSM and PFL shield modes*, and *optional noise/delays on human-state measurements*. It misses only the high-DOF axis (6–7-DoF arms) and enforces safety via an always-on provable shield rather than exposing per-step cost signals to the learner. The blanket sentences "separate from manipulation with a live human coworker" and "safe-RL benchmarks … do not model [perception] at all" are false as written.
2. **"No work tests both families on the same human-co-located manipulation benchmark" — refuted as written.** Thumm, Pelat & Althoff (IROS 2023) [V2] train a PID-Lagrangian CMDP agent *and* a provably-safe shield on the same 6-DoF reaching task beside a replayed human, with explicit per-mechanism conclusions. The claim survives only in qualified form (high-DOF manipulator, ISO-derived continuous costs, regime characterisation). Silver lining: their finding that "the PID-Lagrangian agent is unable to learn a suitable policy" in the human-robot env *corroborates* this report's intermittent-regime result — cite it as supporting evidence, not just a defensive caveat.
3. **"Scaling a hand-engineered CBF-QP to 76 DOF is an open research problem" — too strong.** Real-time whole-body CBF-QPs exist (Khazoom et al. 2022 [V5]: 15 pairs, ~0.2 ms, 24-DOF sim; Morton & Pavone IROS 2025 [V6]: >400 constraints at 1 kHz in timing benchmarks, 7-DOF hardware at 1 kHz). What remains undemonstrated is exactly the report's case: every human-link × robot-link pair on a ~76-DOF humanoid at control rate — the newest humanoid filters deliberately retreat to reduced-order models [V17] or upper-body subspaces at sub-control rates [V19]. Soften from "open problem" to "undemonstrated at this scale", which the 2025–2026 evidence actively supports.
4. **"No humanoid is ISO-certified as a collaborative robot" — true but describes an abolished category.** ISO 10218:2025 deprecates "collaborative robot" entirely: collaboration is now a property of the validated *application*, not the robot [V3]. Rephrase against the 2025 edition; one humanoid CE (machinery-directive) conformity claim exists (AiMOGA "Mornine", Sept 2025 — company claim, footnote-grade).

---

## Part A — Edits where the claim must change

### E1. Benchmark-landscape table (Table 2.1, lines ~953–978)

**Problem:** the six-row table omits every 2020–2025 benchmark with a human in the scene; an examiner who knows Habitat 3.0 or Human-Robot Gym will read the omission as either ignorance or cherry-picking.

**Change:** add five rows (suggested values below) and re-anchor the caption on the *combination* that remains uncovered.

| Benchmark | High-DOF | Live human | ISO signals | Perception noise | Sustained interact. |
|---|---|---|---|---|---|
| Assistive Gym [V11] | ✗ | ✓ (seated, co-optimised) | ✗ | ✗ | ✓ (contact-rich ADL) |
| RCareWorld [V12] | ✗ | ✓ (clinically modelled, limited mobility) | ✗ | ✗ | ✓ (ADL care) |
| Habitat 3.0 [V7] | ✗ | ✓ (kinematic avatars) | ✗ | ✗ | partial (social nav/rearrange) |
| Human-Robot Gym [V1] | ✗ (6–7-DoF arms) | ✓ (mocap replay) | shield-enforced (SSM/PFL modes) | ✓ (optional noise + delays) | ✓ (handover, lifting) |
| Safety-CHORES / SafeVLA [V8] | ✗ | ✗ | ✗ | ✓ (egocentric RGB) | ✗ |

**Caption / claim rewording** (current: "…the intersection that no prior benchmark covers"):

> No prior benchmark exposes ISO-derived *continuous per-step cost signals* for training on a *high-DOF humanoid* beside a moving human. Human-Robot Gym [V1] comes closest: it pairs 6–7-DoF arms with a mocap-replayed human and enforces SSM/PFL via an always-active provable shield, but the safety mechanism is a shield, not a graded cost signal a constrained learner can optimise against, and the embodiment is an order of magnitude lower-DOF.

Notes from verification: Habitat 3.0's avatars are *kinematic* (collisions penalised, no contact dynamics) — say "kinematic avatars", not "low-DOF robots". Human-Robot Gym's shield cites ISO 10218-1:2021 (its SSM/PFL modes correspond to the two ISO/TS 15066 collaboration modes). Safety-CHORES is human-free — it covers the perception axis only.

### E2. Benchmark-landscape prose (lines ~980–989)

**Current:** "The landscape splits along two axes: safety-axis benchmarks … expose generic cost budgets on low-dimensional embodiments without humans; manipulation-axis benchmarks … expose realistic tasks without safety constraints or co-located humans. No existing benchmark covers the intersection."

**Proposed:**

> The landscape splits along two axes: \textbf{safety-axis} benchmarks (Safety-Gym~\cite{ray2019safetygym}, Safety-Gymnasium~\cite{ji2023safetygymnasium}, safe-control-gym~\cite{yuan2022safecontrolgym}, and most recently Safety-CHORES~\cite{zhang2025safevla}, which adds fine-grained costs and egocentric perception but no humans) expose cost budgets without human coworkers; \textbf{manipulation-axis} benchmarks (RLBench~\cite{james2020rlbench}, \bigym{}~\cite{chernyadev2024bigym}, HumanoidBench~\cite{sferrazza2024humanoidbench}) expose realistic tasks without safety constraints. A third, human-in-scene line — Assistive Gym~\cite{erickson2020assistivegym}, RCareWorld~\cite{ye2022rcareworld}, Habitat~3.0~\cite{puig2024habitat3}, and Human-Robot Gym~\cite{thumm2024humanrobotgym} — places simulated humans beside robots, and Human-Robot Gym additionally enforces ISO-style SSM/PFL through a provable shield with optional human-state measurement noise. None of these exposes ISO-derived continuous cost signals on a high-DOF humanoid; the intersection this work targets remains uncovered.

### E3. Intro gap bullet 1 (lines ~342–350)

**Current:** "No benchmark exposes the full ISO-specific co-working setting on a humanoid manipulator." (supporting sentences mention only Safety-Gymnasium / safe-control-gym / BiGym / HumanoidBench)

**Change:** keep the bold claim (it survives), add one sentence before "We extend BiGym…":

> Human-Robot Gym~\cite{thumm2024humanrobotgym} comes closest, pairing 6--7-DoF arms with a mocap-replayed human under a provably safe shield, but neither exposes \iso{}-derived cost signals to the learner nor scales to a humanoid embodiment.

### E4. Intro gap bullet 2 (lines ~351–365)

**Current:** "No published work characterises *when* each safe-RL mechanism helps for manipulation beside a person. The literature is split into two families that are rarely studied together…"

**Problem:** Thumm et al. 2023 [V2] evaluate both families on one human-co-located benchmark; safe-control-gym (already in the bib) was *built* to compare safe RL against safety certification on shared (low-DOF, humanless) tasks; Krasowski et al. [V10] give explicit selection guidance *within* the runtime-intervention family.

**Proposed replacement for the middle of the bullet:**

> The literature is split into two families that are rarely studied together: training-time constrained RL (Lagrangian CMDP~\cite{ray2019safetygym,stooke2020pid}) and runtime safety filtering (shielding~\cite{alshiekh2018shielding,konighofer2025shields}, CBFs~\cite{ames2019cbf,shield2025}, learned safety filters~\cite{srinivasan2020learning,thananjeyan2021recovery}). Cross-family comparisons exist but stop short of this setting: safe-control-gym~\cite{yuan2022safecontrolgym} benchmarks both families on low-DOF stabilisation tasks without humans, and Thumm et al.~\cite{thumm2023interventions} compare a PID-Lagrangian agent with a provably safe shield on a 6-DoF reaching task beside a replayed human — a single task and co-location pattern, with the constrained agent evaluated only under an always-active shield. Within the filter family, Krasowski et al.~\cite{krasowski2023provablysafe} offer selection guidance among intervention types. What no prior work provides is a rule for *which family* to reach for on a high-DOF manipulator under \iso{}-derived costs, or a variable that predicts when a runtime filter will preserve task throughput and when it will destroy it.

### E5. Background "what the literature does not provide" paragraph (lines ~889–902)

**Current:** "…neither names the variable that separates its successes from its failures, and no work tests both families on the same human-co-located manipulation benchmark."

**Proposed:**

> What the literature does not provide, on either side, is a characterisation of \emph{when} each mechanism helps. The constrained-RL line reports tasks where a cost budget is feasible; the shielding and filtering line reports settings where interventions are rare enough to be cheap. The nearest cross-family evidence is consistent with, but far narrower than, the question asked here: on a 6-DoF reaching task beside a replayed human, Thumm et al.~\cite{thumm2023interventions} found a PID-Lagrangian agent ``unable to learn a suitable policy that is both high-performing and reduces failsafe interventions'' while their shield kept violations at zero — a single-task observation this report's intermittent-regime results independently corroborate at 76 DOF (Section~\ref{sec:results:wcsac}). No prior work tests both families across co-location regimes on the same human-co-located manipulation benchmark under \iso{}-derived continuous costs, names the variable that separates their successes, or answers this for a high-DOF manipulator. The regime map of Section~\ref{sec:results:boundary} is this report's answer.

(Also add the corroboration cite at the conclusion's "WCSAC reference corroborates the trainability difficulty" sentence, line ~5002: `…difficulty~\cite{yang2021wcsac}, as does the PID-Lagrangian failure Thumm et al.\ report beside a simulated human~\cite{thumm2023interventions}.`)

### E6. Perception-gap sentences (lines ~317–322 and ~366–372)

**Current:** "…is the deployment challenge most safe-RL benchmarks~\cite{ray2019safetygym,ji2023safetygymnasium,yuan2022safecontrolgym} do not model at all." and "Safe-RL manipulation benchmarks rarely model deployment perception."

**Proposed (first):**

> …is the deployment challenge that safe-RL benchmarks largely do not model: Safety-Gym lineage and safe-control-gym~\cite{ray2019safetygym,ji2023safetygymnasium,yuan2022safecontrolgym} expose ground-truth state, and the one benchmark that does perturb human-state measurements (Human-Robot Gym~\cite{thumm2024humanrobotgym}, optional noise and delays) does not calibrate them to a deployed estimator's characteristics.

**Proposed (second):** keep, but append "— the exception being Human-Robot Gym's optional measurement noise~\cite{thumm2024humanrobotgym}, which is uncalibrated; the mock-\bodyslampp{} wrapper here is instead matched to a published estimator's error, latency, and dropout profile~\cite{henning2023bodyslampp}."

### E7. Humanoid certification claims (lines ~280–285, conclusion ~5016–5018, `unitree_g1` bib note)

**Problem:** "not ISO-certified as a collaborative robot" references a category ISO 10218:2025 abolished (collaboration is a property of the *application*, validated holistically [V3]); the claim also rests on a product page rather than citable literature; and one humanoid CE-conformity claim now exists.

**Proposed (intro item 1):**

> \emph{Commercial humanoid platforms are not certified for collaborative operation.} Under ISO 10218:2025, collaboration is a property of the validated \emph{application}, not of the robot~\cite{hartmann2026iso}; no commercial humanoid platform has been validated for collaborative operation as of mid-2026, and the standards needed for dynamically stable mobile manipulators are still in committee draft~\cite{iso25785}. The Unitree G1, the most widely deployed humanoid research platform in 2026, ships with no collaborative-operation validation~\cite{unitree_g1}; the H1 likewise requires a safety cage during ``lively testing''. Reviews of humanoid safety engineering find ``a lag between technological advances and the adaptation of key safety standards''~\cite{koczi2025humanoidsafety}, and the IEEE Humanoid Study Group concludes the current standards framework ``is not designed for'' humanoids~\cite{ieee2025pathway}.\footnote{The nearest certification milestone to date is machinery-directive conformity, not collaborative-operation validation: AiMOGA (Chery) announced CE-MD/CE-RED/EN~18031 certification of its Mornine humanoid via T\"UV Rheinland in September 2025 (company claim; PR Newswire, 26 Sep.\ 2025).}

**Conclusion (~5018):** replace "deployment without ISO certification~\cite{unitree_g1}" with "deployment without collaborative-operation validation~\cite{unitree_g1,hartmann2026iso}".

**`unitree_g1` bib note:** change "not certified as collaborative under ISO standards" → "no collaborative-operation validation under ISO 10218:2025 / ISO/TS 15066".

### E8. ISO/TS 15066 status sentences (lines ~264–265, ~571–573, PFL annex at ~609–612)

**Claim verified:** ISO/TS 15066:2016 remains a published TS; its content was absorbed into ISO 10218:2025 — PFL limit tables became *informative* Annex M of Part 2 with the normative anchor in §5.14.6 [V3].

**Change (add-only):** append `\cite{hartmann2026iso}` to both "integrated into ISO 10218:2025" sentences. Optionally, at the PFL annex sentence (~610): "Annex~A of the standard tabulates quasi-static and transient force limits per body region (carried into ISO 10218-2:2025 as Annex~M, with the PFL requirement made normative in §5.14.6~\cite{hartmann2026iso})."

### E9. Safety-filter family dichotomy (lines ~817–821)

**Current:** "Practical implementations are either hand-engineered Control Barrier Functions (CBFs)~\cite{ames2019cbf} or learned safety value functions~\cite{srinivasan2020learning,thananjeyan2021recovery}."

**Problem:** the 2023–2024 reviews treat predictive (MPC-style) filters as a coequal third family.

**Proposed:**

> Hamilton-Jacobi reachability~\cite{bansal2017hamilton,ganai2024hjsurvey} gives the theoretical foundation for safety value functions. Practical implementations span hand-engineered Control Barrier Functions (CBFs)~\cite{ames2019cbf}, predictive (MPC-style) safety filters~\cite{wabersich2023datadriven}, and learned safety value functions~\cite{srinivasan2020learning,thananjeyan2021recovery}; Hsu et al.~\cite{hsu2024safetyfilter} unify all three under a single safety-filter view, with learned value functions as the data-driven member this work instantiates.

### E10. CBF-QP scalability claim (lines ~835–841)

**Current:** "Constructing and solving a CBF quadratic program over every human-link / robot-link pair at the control rate is substantially more expensive on a 76-DOF humanoid than a single forward pass of a learned value function, and scaling a hand-engineered CBF-QP to an embodiment of this dimensionality is itself an open research problem."

**Problem:** QP solve cost at moderate constraint counts is demonstrably no longer the bottleneck (Khazoom: 15 pairs at ~0.2 ms, 24-DOF humanoid, simulation [V5]; OSCBF: >400 constraints at 1 kHz in timing benchmarks, 7-DOF arm at 1 kHz on hardware [V6]). The defensible claim is that *every-pair filtering at humanoid scale* remains undemonstrated — which the newest systems support: hardware humanoid filters use reduced-order models with a single constraint at 100 Hz [V17], SHIELD filters velocity commands on a reduced model, SPARK's real-G1 deployment is upper-body safe-set control [V18], and the most direct attempt at human-link × robot-link CBF-QP filtering on a humanoid is simulation-only, restricted to an 8-DoF upper-body subspace, and solves 72 capsule constraints at ~33 Hz on GPU (4 Hz CPU) [V19].

**Proposed:**

> A CBF-QP filter for this setting must enforce barrier constraints over every human-link / robot-link pair at the control rate. QP solve cost at moderate constraint counts is no longer the obstacle it once was: a whole-body CBF-QP with 15 collision pairs solves in $\sim$0.2\,ms on a 24-DOF humanoid in simulation~\cite{khazoom2022selfcollision}, and operational-space CBF formulations sustain hundreds of constraints at kilohertz rates on a 7-DOF arm~\cite{morton2025oscbf}. What remains undemonstrated is the full combination this benchmark poses: hardware humanoid safety filters to date either restrict to self-collision subsets in simulation~\cite{khazoom2022selfcollision}, filter reduced-order models with a single constraint~\cite{bena2025poisson,shield2025}, or run safe-set control on an upper-body subspace~\cite{sun2025spark}; the most direct attempt at human-link $\times$ robot-link CBF filtering on a humanoid is simulation-only, restricted to an 8-DoF upper-body command space, and solves its 72 capsule constraints at $\sim$33\,Hz on a GPU~\cite{cai2026humanoidcbf} — below this benchmark's control rate, before certified barrier synthesis under exogenous human motion and floating-base dynamics is even addressed~\cite{shield2025}. A learned value function reduces the runtime decision to a single forward pass at any embodiment dimension; this practical consideration, alongside the data-efficiency argument above, is why we adopt the learned-value-function path.

**Mirror edit** at future work (~4926–4933): the "curse of dimensionality at 76 DOF" sentence stands; append `\cite{hsu2024safetyfilter}` to the HJ filter mention and optionally cite \cite{bena2025poisson} as the reduced-order route worth porting.

### E11. SHIELD positioning sentence (intro ~296–303; positioning §2.7 ~1029–1034) — *minor*

Still accurate. Optional one-clause addition at §2.7: SPARK [V18] as a second humanoid-near-humans runtime system (safe-set, real G1, upper-body) — strengthens "emerging literature" framing. No text change required.

### E12. MuJoCo contact-validation claim (lines ~4614–4617)

**Current:** "Contact fidelity: MuJoCo~\cite{todorov2012mujoco} forces are not safety-validated against real collision measurements~\cite{svarny2020collision}, even once the PFL detection limitation is resolved."

**Verification outcome:** no MuJoCo-vs-human-robot-collision force validation exists (claim survives), but blanket phrasing is refutable: MuJoCo contact forces *have* been force-validated in a leg-terrain setting [V40], and simulator-vs-reality impact studies exist at trajectory level [V31]. **Important correction:** the search sweep initially attributed an LS-DYNA/crash-dummy PFL simulation to Schlotzhauer et al. 2022 — verification showed that paper actually builds a *physical* biofidelic measurement database and explicitly argues dynamic simulation is limited for constrained contacts [V39]. Described correctly, it *supports* the report's claim.

**Proposed:**

> \textbf{Contact fidelity:} to our knowledge, no rigid-body engine's contact forces have been validated against measured human--robot collision forces. Simulator validation on real impacts exists only at the trajectory level — and finds MuJoCo's impacts nearly insensitive to its contact-stiffness parameter~\cite{acosta2022simvalidation}; force-level MuJoCo validation exists only outside HRC (robot-foot/terrain contact~\cite{joseph2026mujoco}); and industrial \pfl{} validation practice relies on physical biofidelic measurement precisely because dynamic simulation of constrained contacts is considered unreliable~\cite{schlotzhauer2022pfl,svarny2020collision}. \sbigym{}'s contact forces are therefore safety-\emph{shaped}, not safety-\emph{validated}, even once the \pfl{} detection limitation is resolved.

---

## Part B — Claims that hold; add modern citations (add-only)

### E13. "Cost usually treated as a binary indicator" (lines ~630–633)
Holds. AAAI 2023 states it verbatim ("Recent approaches usually adopt indicator cost functions where a positive signal deems a state as unsafe and zero deems a state safe" — quote confirmed in the published PDF), and every Safety-Gymnasium cost is a 0/1 indicator by construction.
**Edit:** `…usually treated as a binary indicator~\cite{ray2019safetygym,achiam2017cpo,stooke2020pid,he2023autocost,ji2023safetygymnasium}.`

### E14. Shielding anchor (lines ~796–799)
Holds; the original shield-synthesis group published a CACM retrospective.
**Edit:** `…\emph{shielding}~\cite{alshiekh2018shielding,konighofer2025shields}, which synthesises…`

### E15. "CQL is the standard pessimistic offline-RL method" (line ~907)
Holds-with-caveat: CQL is *a* canonical baseline (alongside IQL/TD3+BC) per the TNNLS survey; and CQL-style pessimism on the *cost* critic is established practice (CPQ, AAAI 2022 — "make OOD actions 'unsafe'"), which strengthens the report's design rationale from "our idea" to "recognised pattern".
**Edits:** "the standard pessimistic offline-RL method" → "a standard pessimistic offline-RL method~\cite{kumar2020cql,prudencio2024offlinesurvey}". Append to the section: `Applying the same pessimism to the \emph{cost} critic is itself established in safe offline RL: CPQ penalises out-of-distribution actions in the constraint value precisely so unseen actions are not falsely certified safe~\cite{xu2022cpq}, the role the CQL-trained safety critic plays here.`

### E16. Sim-to-real "dynamics gap" framing (lines ~922–923, ~4610–4613)
Holds.
**Edit (both sites):** `…the \emph{dynamics gap}~\cite{tobin2017domainrand,zhao2020sim2real,muratore2022randomized}…` and at the limitations site: `…domain randomisation~\cite{tobin2017domainrand,zhao2020sim2real,muratore2022randomized} is the standard follow-up — it remains the deployed recipe for real-world humanoid RL~\cite{radosavovic2024humanoid} — and the gap must be acknowledged…`

### E17. Curriculum standardness (lines ~725–731, ~2371, ~2394, ~4599)
Holds.
**Edits:** at ~728: `The technique is standard in high-dimensional robotics: OpenAI's dexterous-manipulation work~\cite{openai2019solving} uses an automatic difficulty curriculum, massively-parallel legged-robot training rests on a game-inspired terrain curriculum~\cite{rudin2022walkminutes}, and current Unitree H1 pipelines auto-adjust terrain difficulty on success~\cite{zhuang2024parkour}.` At ~4599 append `\cite{rudin2022walkminutes}` to the citation list. Cite Rudin as **CoRL 2021, PMLR 164, 2022** (not "CoRL 2022"). Optional softening: "the standard solution" → "a standard solution" (demonstration-driven training is co-standard).

### E18. "SMPL-H and similar parametric models are the standard" (lines ~1146–1148)
Holds — TPAMI 2023 survey: SMPL is "currently the most widely used human body model in the research community" (verbatim, §2.2.1).
**Edit:** `SMPL-H~\cite{romero2017smplh} and similar parametric models remain the standard for human-pose research~\cite{tian2023meshsurvey}, but…`

### E19. BodySLAM++ as deployment exemplar (lines ~309–316, ~934–948)
Design choice stands (CPU-real-time, visual-inertial, on-robot). Post-2023 world-grounded estimators should be acknowledged once.
**Edit (background §2.6.2, after the BodySLAM++ description):** `A newer line of world-grounded monocular estimators (WHAM~\cite{shin2024wham}, TRAM~\cite{wang2024tram}) recovers global SMPL motion at high accuracy, but targets offline GPU pipelines rather than real-time on-robot CPU inference with IMU fusion, which is why the mock-perception model is calibrated to \bodyslampp{}.`

### E20. WCSAC citations (lines ~783, ~2069, ~5224)
WCSAC remains the canonical distributional-cost baseline (2023–2025 successors all benchmark against it). The same authors' journal extension adds the quantile-regression safety critic — cite alongside the AAAI version where the method is introduced and where reimplemented.
**Edit:** at ~783 and ~2069: `WCSAC~\cite{yang2021wcsac,yang2023wcsacjournal}`. The reimplementation appendix (~5224) should state which variant is reimplemented (Gaussian = AAAI-21; the journal version adds SafetyCritic-QR).

### E21. CVaR-optimised training future work (lines ~4905–4919)
"Natural future work" is fine for *this pipeline*, but must not read as a literature gap — explicit CVaR/spectral-risk-constrained training exists.
**Edit (append):** `Mature instruments exist: off-policy trust-region CVaR constraints~\cite{kim2022offtrc} (the closest fit to this critic-only replay-based pipeline), distributional multi-constraint safe RL~\cite{kim2023sdac}, and spectral-risk methods with convergence guarantees~\cite{kim2024srcpo}; the open question is their composition with a C51 cost head under the support invariant of Proposition~\ref{lem:support-bound}.`

### E22. "Safety has not received proportional attention" (lines ~286–291)
The Bharadhwaj claim holds but is 2021-vintage; 2024–2026 audits provide current, *empirical* backing — and show the gap is now actively studied, so phrase as persisting-but-narrowing.
**Edit:**

> \emph{Robot learning's safety record remains thin at deployment.} Bharadhwaj~\cite{auditing2021} argued in 2021 that safety and compliance receive disproportionately little attention in robot-learning evaluation; the audits that have since emerged bear this out — jailbreaks eliciting harmful physical actions from a deployed commercial robot~\cite{robey2025jailbreaking}, and every evaluated LLM-driven robot model failing basic safety and discrimination criteria~\cite{hundt2025llmrobots} — while constrained-learning responses are only now appearing~\cite{zhang2025safevla}. Brunke et al.~\cite{brunke2022safelearning} and the newer safe-RL survey…(rest unchanged)

Caveat from verification: Hundt et al. do not cite Bharadhwaj — keep them as parallel evidence, not as a citation lineage. RoboPAIR's "100\% attack success" is per-setting ("often achieving") — avoid a flat universal figure.

---

## Part C — Pre-2022 references to keep as-is

Origin/method/asset citations where age is appropriate. No edits beyond those listed above.

| Key | Year | Role | Verdict |
|---|---|---|---|
| `altman1999constrained` | 1999 | CMDP origin | Keep |
| `garcia2015comprehensive` | 2015 | Framed as "foundational survey", already paired with 2022/2024 surveys | Keep |
| `iso15066` | 2016 | The standard itself; E8 adds status cite | Keep |
| `achiam2017cpo` | 2017 | CPO origin | Keep (E13 touches its cite list) |
| `bellemare2017c51` | 2017 | C51 — backbone component | Keep |
| `marvel2017implementing` | 2017 | Canonical SSM implementation | Keep (E23 adds companions) |
| `romero2017smplh` | 2017 | SMPL-H asset | Keep (E18 adds survey) |
| `tobin2017domainrand` | 2017 | Domain-randomisation origin | Keep (E16) |
| `bansal2017hamilton` | 2017 | HJ overview | Keep (E9 adds 2024 survey) |
| `alshiekh2018shielding` | 2018 | Shielding origin | Keep (E14) |
| `dabney2018qrdqn` | 2018 | QR-DQN origin | Keep |
| `svarny2019unified` | 2019 | SSM+PFL unified treatment (equation source) | Keep (E23) |
| `mahmood2019amass` | 2019 | AMASS asset | Keep |
| `ray2019safetygym` | 2019 | Safety-Gym + Lagrangian baseline | Keep (rows/cites updated in E1–E4, E13) |
| `openai2019solving` | 2019 | ADR exemplar | Keep (E17) |
| `james2020rlbench` | 2020 | Benchmark row (facts still true) | Keep |
| `kumar2020cql` | 2020 | CQL — method used | Keep (E15 rewords "the standard") |
| `stooke2020pid` | 2020 | PID Lagrangian — method used | Keep |
| `srinivasan2020learning` | 2020 | Learned-SVF line | Keep |
| `zhao2020sim2real` | 2020 | Sim-to-real survey | Keep (E16 adds 2022+ survey) |
| `narvekar2020curriculum` | 2020 | Curriculum survey | Keep (E17) |
| `svarny2020collision` | 2021 | Collision-force measurements (UR10e/iiwa, >100% variation — confirmed accurate) | Keep (E23 adds journal successor) |
| `thananjeyan2021recovery` | 2021 | Recovery RL — positioned method | Keep |
| `yang2021wcsac` | 2021 | WCSAC — reimplemented baseline | Keep (E20 adds journal version) |
| `auditing2021` | 2021 | Position paper | Keep (E22 adds 2025 evidence) |
| `trautman2010freezing` | 2010 | Names the freezing-robot problem | Keep |
| `todorov2012mujoco` | 2012 | MuJoCo tool cite | Keep (E12) |
| `yarats2021drqv2` | 2022 (ICLR) | Already updated by metadata audit | Keep |

### E23. SSM/PFL modern companions (lines ~591, ~613) — *recommended, add-only*
The SSM attribution and the Svarny measurement claim verified correct; the same two labs extended (not overturned) them.
**Edits:** at ~613 append: `; the same group's journal-scale campaign (2{,}250 collisions, with and without protective skins) confirms the magnitude and adds contact-duration and impulse characterisation~\cite{svarny2022softskins}`. Optionally cite \cite{kirschner2024edged} where PFL limit tables are discussed (extends limits beyond blunt contact), and \cite{rozlivek2025harmonious} in related work as the same lab's humanoid-scale successor (pre-collision visual/proximity + post-collision tactile constraints on iCub — note this is *our* lineage framing; the paper does not use SSM/PFL terminology).

### Not re-verified (low risk, flagged for honesty)
The narrower future-work negative claim "No published work measures that combination [anticipatory filter + confidence-aware fallback + graded slow-down]" (~4939) was not exhaustively re-searched. It is tightly scoped and consistent with everything found; consider softening to "to our knowledge".
The intro's "greenfield problem" (~259–261) is rhetoric rather than a citation claim; with SHIELD/SPARK/Human-Robot Gym now cited nearby, consider "largely greenfield".

---

## Part D — New references (Vancouver style)

All double-verified ✅ (metadata against primary page + supports intended use). Diacritics per publisher records. Suggested BibTeX keys in brackets.

**Tier 1 — required by Part A edits:**

- **[V1]** [`thumm2024humanrobotgym`] Thumm J, Trost F, Althoff M. Human-Robot Gym: benchmarking reinforcement learning in human-robot collaboration. In: 2024 IEEE International Conference on Robotics and Automation (ICRA); 2024. p. 7405–7411.
- **[V2]** [`thumm2023interventions`] Thumm J, Pelat G, Althoff M. Reducing safety interventions in provably safe reinforcement learning. In: 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS); 2023. p. 7515–7522.
- **[V3]** [`hartmann2026iso`] Hartmann D, Hamříková K, Vysocký A, Laciok V, Bernatík A. Evolution of safety requirements in industrial robotics: comparative analysis of ISO 10218-1/2 (2011 vs. 2025) and integration of ISO/TS 15066. Results in Engineering. 2026;30:110486.
- **[V4]** [`hsu2024safetyfilter`] Hsu KC, Hu H, Fisac JF. The safety filter: a unified view of safety-critical control in autonomous systems. Annual Review of Control, Robotics, and Autonomous Systems. 2024;7:47–72.
- **[V5]** [`khazoom2022selfcollision`] Khazoom C, Gonzalez-Diaz D, Ding Y, Kim S. Humanoid self-collision avoidance using whole-body control with control barrier functions. In: 2022 IEEE-RAS 21st International Conference on Humanoid Robots (Humanoids); 2022. p. 558–565.
- **[V6]** [`morton2025oscbf`] Morton D, Pavone M. Safe, task-consistent manipulation with operational space control barrier functions. In: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS); 2025. p. 187–194.
- **[V7]** [`puig2024habitat3`] Puig X, Undersander E, Szot A, Dallaire Cote M, Yang TY, Partsey R, et al. Habitat 3.0: a co-habitat for humans, avatars, and robots. In: International Conference on Learning Representations (ICLR); 2024.
- **[V8]** [`zhang2025safevla`] Zhang B, Zhang Y, Ji J, Lei Y, Dai J, Chen Y, Yang Y. SafeVLA: towards safety alignment of vision-language-action model via constrained learning. In: Advances in Neural Information Processing Systems 38 (NeurIPS); 2025. *(7-author proceedings list — do not use the 8-author arXiv v4 list.)*
- **[V9]** [`he2023autocost`] He T, Zhao W, Liu C. AutoCost: evolving intrinsic cost for zero-violation reinforcement learning. Proceedings of the AAAI Conference on Artificial Intelligence. 2023;37(12):14847–14855.

**Tier 2 — recommended (Part B edits and table rows):**

- **[V10]** [`krasowski2023provablysafe`] Krasowski H, Thumm J, Müller M, Schäfer L, Wang X, Althoff M. Provably safe reinforcement learning: conceptual analysis, survey, and benchmarking. Transactions on Machine Learning Research. 2023.
- **[V11]** [`erickson2020assistivegym`] Erickson Z, Gangaram V, Kapusta A, Liu CK, Kemp CC. Assistive Gym: a physics simulation framework for assistive robotics. In: 2020 IEEE International Conference on Robotics and Automation (ICRA); 2020. p. 10169–10176.
- **[V12]** [`ye2022rcareworld`] Ye R, Xu W, Fu H, Jenamani RK, Nguyen V, Lu C, et al. RCareWorld: a human-centric simulation world for caregiving robots. In: 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS); 2022. p. 33–40. *(Award, if mentioned: Best RoboCup Paper Award at IROS 2022 — not overall Best Paper.)*
- **[V13]** [`koczi2025humanoidsafety`] Kóczi D, Sárosi J. Safety engineering for humanoid robots in everyday life—scoping review. Electronics. 2025;14(23):4734.
- **[V14]** [`ieee2025pathway`] IEEE Humanoid Study Group. A pathway study for future humanoid standards. Technical report; 2025 Sep. doi:10.13140/RG.2.2.27892.21122. *(Grey literature: ResearchGate-registered DOI, not an IEEE-published document — cite as techreport.)*
- **[V15]** [`svarny2022softskins`] Švarný P, Rozlivek J, Rustler L, Šrámek M, Deli Ö, Zillich M, et al. Effect of active and passive protective soft skins on collision forces in human-robot collaboration. Robotics and Computer-Integrated Manufacturing. 2022;78:102363.
- **[V16]** [`rozlivek2025harmonious`] Rozlivek J, Roncone A, Pattacini U, Hoffmann M. HARMONIOUS—human-like reactive motion control and multimodal perception for humanoid robots. IEEE Transactions on Robotics. 2025;41:378–393.
- **[V17]** [`bena2025poisson`] Bena RM, Bahati G, Werner B, Cosner RK, Yang L, Ames AD. Geometry-aware predictive safety filters on humanoids: from Poisson safety functions to CBF constrained MPC. In: 2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids); 2025. p. 1–8.
- **[V18]** [`sun2025spark`] Sun Y, Chen R, Yun KS, Fang Y, Jung S, Li F, et al. SPARK: Safe Protective and Assistive Robot Kit. arXiv:2502.03132 [preprint]; 2025. *(arXiv-only; short version presented at IFAC Symposium on Robotics 2025. Real-G1 deployment uses safe-set control on a 17-DoF upper-body+waist config — do not describe as whole-body CBF or 23-DOF.)*
- **[V19]** [`cai2026humanoidcbf`] Cai W, Abanes J, Evangeliou N, Tzes A. Safe human-to-humanoid motion imitation using control barrier functions. arXiv:2604.11447 [preprint]; 2026. *(Simulation-only.)*
- **[V20]** [`konighofer2025shields`] Könighofer B, Bloem R, Jansen N, Junges S, Pranger S. Shields for safe reinforcement learning. Communications of the ACM. 2025;68(11):80–90.
- **[V21]** [`ganai2024hjsurvey`] Ganai M, Gao S, Herbert S. Hamilton-Jacobi reachability in reinforcement learning: a survey. IEEE Open Journal of Control Systems. 2024;3:310–324.
- **[V22]** [`wabersich2023datadriven`] Wabersich KP, Taylor AJ, Choi JJ, Sreenath K, Tomlin CJ, Ames AD, et al. Data-driven safety filters: Hamilton-Jacobi reachability, control barrier functions, and predictive methods for uncertain systems. IEEE Control Systems Magazine. 2023;43(5):137–177.
- **[V23]** [`muratore2022randomized`] Muratore F, Ramos F, Turk G, Yu W, Gienger M, Peters J. Robot learning from randomized simulations: a review. Frontiers in Robotics and AI. 2022;9:799893.
- **[V24]** [`radosavovic2024humanoid`] Radosavovic I, Xiao T, Zhang B, Darrell T, Malik J, Sreenath K. Real-world humanoid locomotion with reinforcement learning. Science Robotics. 2024;9(89):eadi9579.
- **[V25]** [`prudencio2024offlinesurvey`] Prudencio RF, Maximo MROA, Colombini EL. A survey on offline reinforcement learning: taxonomy, review, and open problems. IEEE Transactions on Neural Networks and Learning Systems. 2024;35(8):10237–10257.
- **[V26]** [`xu2022cpq`] Xu H, Zhan X, Zhu X. Constraints penalized Q-learning for safe offline reinforcement learning. Proceedings of the AAAI Conference on Artificial Intelligence. 2022;36(8):8753–8760.
- **[V27]** [`tian2023meshsurvey`] Tian Y, Zhang H, Liu Y, Wang L. Recovering 3D human mesh from monocular images: a survey. IEEE Transactions on Pattern Analysis and Machine Intelligence. 2023;45(12):15406–15425.
- **[V28]** [`shin2024wham`] Shin S, Kim J, Halilaj E, Black MJ. WHAM: reconstructing world-grounded humans with accurate 3D motion. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); 2024. p. 2070–2080.
- **[V29]** [`rudin2022walkminutes`] Rudin N, Hoeller D, Reist P, Hutter M. Learning to walk in minutes using massively parallel deep reinforcement learning. In: Proceedings of the 5th Conference on Robot Learning (CoRL 2021). PMLR 164; 2022. p. 91–100.
- **[V30]** [`zhuang2024parkour`] Zhuang Z, Yao S, Zhao H. Humanoid parkour learning. In: Proceedings of the 8th Conference on Robot Learning (CoRL); 2024.
- **[V31]** [`acosta2022simvalidation`] Acosta B, Yang W, Posa M. Validating robotics simulators on real-world impacts. IEEE Robotics and Automation Letters. 2022;7(3):6471–6478.
- **[V32]** [`yang2023wcsacjournal`] Yang Q, Simão TD, Tindemans SH, Spaan MTJ. Safety-constrained reinforcement learning with a distributional safety critic. Machine Learning. 2023;112(3):859–887.
- **[V33]** [`kim2022offtrc`] Kim D, Oh S. Efficient off-policy safe reinforcement learning using trust region conditional value at risk. IEEE Robotics and Automation Letters. 2022;7(3):7644–7651.
- **[V34]** [`kim2023sdac`] Kim D, Lee K, Oh S. Trust region-based safe distributional reinforcement learning for multiple constraints. In: Advances in Neural Information Processing Systems 36 (NeurIPS); 2023.
- **[V35]** [`robey2025jailbreaking`] Robey A, Ravichandran Z, Kumar V, Hassani H, Pappas GJ. Jailbreaking LLM-controlled robots. In: 2025 IEEE International Conference on Robotics and Automation (ICRA); 2025. p. 11948–11956.
- **[V36]** [`hundt2025llmrobots`] Hundt A, Azeem R, Mansouri M, Brandão M. LLM-driven robots risk enacting discrimination, violence, and unlawful actions. International Journal of Social Robotics. 2025;17(11):2663–2711.

**Tier 3 — optional:**

- **[V37]** [`wang2024tram`] Wang Y, Wang Z, Liu L, Daniilidis K. TRAM: global trajectory and motion of 3D humans from in-the-wild videos. In: Computer Vision — ECCV 2024. LNCS 15069; 2024. p. 467–487.
- **[V38]** [`kirschner2024edged`] Kirschner RJ, Micheler CM, Zhou Y, Siegner S, Hamad M, Glowalla C, et al. Towards safe robot use with edged or pointed objects: a surrogate study assembling a human hand injury protection database. In: 2024 IEEE International Conference on Robotics and Automation (ICRA); 2024. p. 12680–12687.
- **[V39]** [`schlotzhauer2022pfl`] Schlotzhauer A, Stotz T, Awad R, Kraus W. Virtual validation of power and force limiting setups in human-robot-collaboration. Procedia CIRP. 2022;107:845–850. *(Describe as: physical biofidelic collision-measurement database + expert system; explicitly notes dynamic simulation is limited for constrained contacts. Do NOT describe as LS-DYNA/crash-dummy FEM — that is Oberer & Schraft, ICRA 2007.)*
- **[V40]** [`joseph2026mujoco`] Joseph R, Dutta A. Contact force estimation for a single leg test setup with compliance in MuJoCo. Proceedings of the Institution of Mechanical Engineers, Part C: Journal of Mechanical Engineering Science. 2026;240(5):1569–1588.
- **[V41]** [`kim2024srcpo`] Kim D, Cho T, Han S, Chung H, Lee K, Oh S. Spectral-risk safe reinforcement learning with convergence guarantees. In: Advances in Neural Information Processing Systems 37 (NeurIPS); 2024.
- **AiMOGA footnote (no bib entry):** AiMOGA Robotics (Chery Group). "World's First: AiMOGA Robot Achieves Full CE Certification for Humanoids…" PR Newswire, 26 Sep 2025. Company claim, PR-derived coverage only — cite in a footnote, attributed to the company.

---

## Part E — Ready-to-paste BibTeX (Tier 1 + Tier 2)

```bibtex
@inproceedings{thumm2024humanrobotgym,
  author    = {Thumm, Jakob and Trost, Felix and Althoff, Matthias},
  title     = {Human-Robot Gym: Benchmarking Reinforcement Learning in Human-Robot Collaboration},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {7405--7411},
  year      = {2024},
  note      = {arXiv:2310.06208}
}

@inproceedings{thumm2023interventions,
  author    = {Thumm, Jakob and Pelat, Guillaume and Althoff, Matthias},
  title     = {Reducing Safety Interventions in Provably Safe Reinforcement Learning},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {7515--7522},
  year      = {2023},
  note      = {arXiv:2303.03339}
}

@article{hartmann2026iso,
  author  = {Hartmann, Daniel and Ham{\v r}{\'\i}kov{\'a}, Krist{\'y}na and Vysock{\'y}, Ale{\v s} and Laciok, Vendula and Bernat{\'\i}k, Ale{\v s}},
  title   = {Evolution of safety requirements in industrial robotics: Comparative analysis of {ISO} 10218-1/2 (2011 vs. 2025) and integration of {ISO/TS} 15066},
  journal = {Results in Engineering},
  volume  = {30},
  pages   = {110486},
  year    = {2026},
  doi     = {10.1016/j.rineng.2026.110486}
}

@article{hsu2024safetyfilter,
  author  = {Hsu, Kai-Chieh and Hu, Haimin and Fisac, Jaime F.},
  title   = {The safety filter: A unified view of safety-critical control in autonomous systems},
  journal = {Annual Review of Control, Robotics, and Autonomous Systems},
  volume  = {7},
  pages   = {47--72},
  year    = {2024},
  doi     = {10.1146/annurev-control-071723-102940}
}

@inproceedings{khazoom2022selfcollision,
  author    = {Khazoom, Charles and Gonzalez-Diaz, Daniel and Ding, Yanran and Kim, Sangbae},
  title     = {Humanoid self-collision avoidance using whole-body control with control barrier functions},
  booktitle = {IEEE-RAS 21st International Conference on Humanoid Robots (Humanoids)},
  pages     = {558--565},
  year      = {2022},
  note      = {arXiv:2207.00692. Simulation; 15 collision-body pairs, $\sim$0.2 ms average QP solve}
}

@inproceedings{morton2025oscbf,
  author    = {Morton, Daniel and Pavone, Marco},
  title     = {Safe, task-consistent manipulation with operational space control barrier functions},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {187--194},
  year      = {2025},
  note      = {arXiv:2503.06736}
}

@inproceedings{puig2024habitat3,
  author    = {Puig, Xavier and Undersander, Eric and Szot, Andrew and Dallaire Cote, Mikael and Yang, Tsung-Yen and Partsey, Ruslan and others},
  title     = {Habitat 3.0: A co-habitat for humans, avatars, and robots},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  note      = {arXiv:2310.13724}
}

@inproceedings{zhang2025safevla,
  author    = {Zhang, Borong and Zhang, Yuhao and Ji, Jiaming and Lei, Yingshan and Dai, Juntao and Chen, Yuanpei and Yang, Yaodong},
  title     = {{SafeVLA}: Towards safety alignment of vision-language-action model via constrained learning},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {arXiv:2503.03480. Spotlight. Introduces the Safety-CHORES benchmark}
}

@inproceedings{he2023autocost,
  author    = {He, Tairan and Zhao, Weiye and Liu, Changliu},
  title     = {{AutoCost}: Evolving intrinsic cost for zero-violation reinforcement learning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {37},
  number    = {12},
  pages     = {14847--14855},
  year      = {2023},
  doi       = {10.1609/aaai.v37i12.26734}
}

@article{krasowski2023provablysafe,
  author  = {Krasowski, Hanna and Thumm, Jakob and M{\"u}ller, Marlon and Sch{\"a}fer, Lukas and Wang, Xiao and Althoff, Matthias},
  title   = {Provably safe reinforcement learning: Conceptual analysis, survey, and benchmarking},
  journal = {Transactions on Machine Learning Research},
  year    = {2023},
  note    = {arXiv:2205.06750}
}

@inproceedings{erickson2020assistivegym,
  author    = {Erickson, Zackory and Gangaram, Vamsee and Kapusta, Ariel and Liu, C. Karen and Kemp, Charles C.},
  title     = {Assistive {Gym}: A physics simulation framework for assistive robotics},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {10169--10176},
  year      = {2020},
  note      = {arXiv:1910.04700}
}

@inproceedings{ye2022rcareworld,
  author    = {Ye, Ruolin and Xu, Wenqiang and Fu, Haoyuan and Jenamani, Rajat Kumar and Nguyen, Vy and Lu, Cewu and Dimitropoulou, Katherine and Bhattacharjee, Tapomayukh},
  title     = {{RCareWorld}: A human-centric simulation world for caregiving robots},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {33--40},
  year      = {2022},
  note      = {arXiv:2210.10821. Best RoboCup Paper Award at IROS 2022}
}

@article{koczi2025humanoidsafety,
  author  = {K{\'o}czi, D{\'a}vid and S{\'a}rosi, J{\'o}zsef},
  title   = {Safety engineering for humanoid robots in everyday life---scoping review},
  journal = {Electronics},
  volume  = {14},
  number  = {23},
  pages   = {4734},
  year    = {2025},
  doi     = {10.3390/electronics14234734}
}

@techreport{ieee2025pathway,
  author      = {{IEEE Humanoid Study Group}},
  title       = {A pathway study for future humanoid standards},
  institution = {IEEE Humanoid Study Group},
  month       = sep,
  year        = {2025},
  note        = {Grey literature; DOI 10.13140/RG.2.2.27892.21122 (ResearchGate)}
}

@article{svarny2022softskins,
  author  = {{\v S}varn{\'y}, Petr and Rozlivek, Jakub and Rustler, Lukas and {\v S}r{\'a}mek, Martin and Deli, {\"O}zg{\"u}r and Zillich, Michael and Hoffmann, Matej},
  title   = {Effect of active and passive protective soft skins on collision forces in human-robot collaboration},
  journal = {Robotics and Computer-Integrated Manufacturing},
  volume  = {78},
  pages   = {102363},
  year    = {2022},
  doi     = {10.1016/j.rcim.2022.102363}
}

@article{rozlivek2025harmonious,
  author  = {Rozlivek, Jakub and Roncone, Alessandro and Pattacini, Ugo and Hoffmann, Matej},
  title   = {{HARMONIOUS}---Human-like reactive motion control and multimodal perception for humanoid robots},
  journal = {IEEE Transactions on Robotics},
  volume  = {41},
  pages   = {378--393},
  year    = {2025},
  doi     = {10.1109/TRO.2024.3502216}
}

@inproceedings{bena2025poisson,
  author    = {Bena, Ryan M. and Bahati, Gilbert and Werner, Blake and Cosner, Ryan K. and Yang, Lizhi and Ames, Aaron D.},
  title     = {Geometry-aware predictive safety filters on humanoids: From {Poisson} safety functions to {CBF} constrained {MPC}},
  booktitle = {IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids)},
  pages     = {1--8},
  year      = {2025},
  note      = {arXiv:2508.11129}
}

@misc{sun2025spark,
  author       = {Sun, Yifan and Chen, Rui and Yun, Kai S. and Fang, Yikuan and Jung, Sebin and Li, Feihan and Li, Bowei and Zhao, Weiye and Liu, Changliu},
  title        = {{SPARK}: Safe Protective and Assistive Robot Kit},
  year         = {2025},
  eprint       = {2502.03132},
  archivePrefix= {arXiv},
  note         = {Short version presented at the IFAC Symposium on Robotics, 2025}
}

@misc{cai2026humanoidcbf,
  author       = {Cai, Wenqi and Abanes, John and Evangeliou, Nikolaos and Tzes, Anthony},
  title        = {Safe human-to-humanoid motion imitation using control barrier functions},
  year         = {2026},
  eprint       = {2604.11447},
  archivePrefix= {arXiv},
  note         = {Simulation only}
}

@article{konighofer2025shields,
  author  = {K{\"o}nighofer, Bettina and Bloem, Roderick and Jansen, Nils and Junges, Sebastian and Pranger, Stefan},
  title   = {Shields for safe reinforcement learning},
  journal = {Communications of the ACM},
  volume  = {68},
  number  = {11},
  pages   = {80--90},
  year    = {2025},
  doi     = {10.1145/3715958}
}

@article{ganai2024hjsurvey,
  author  = {Ganai, Milan and Gao, Sicun and Herbert, Sylvia},
  title   = {{Hamilton-Jacobi} reachability in reinforcement learning: A survey},
  journal = {IEEE Open Journal of Control Systems},
  volume  = {3},
  pages   = {310--324},
  year    = {2024},
  doi     = {10.1109/OJCSYS.2024.3449138}
}

@article{wabersich2023datadriven,
  author  = {Wabersich, Kim P. and Taylor, Andrew J. and Choi, Jason J. and Sreenath, Koushil and Tomlin, Claire J. and Ames, Aaron D. and Zeilinger, Melanie N.},
  title   = {Data-driven safety filters: {Hamilton-Jacobi} reachability, control barrier functions, and predictive methods for uncertain systems},
  journal = {IEEE Control Systems Magazine},
  volume  = {43},
  number  = {5},
  pages   = {137--177},
  year    = {2023},
  doi     = {10.1109/MCS.2023.3291885}
}

@article{muratore2022randomized,
  author  = {Muratore, Fabio and Ramos, Fabio and Turk, Greg and Yu, Wenhao and Gienger, Michael and Peters, Jan},
  title   = {Robot learning from randomized simulations: A review},
  journal = {Frontiers in Robotics and AI},
  volume  = {9},
  pages   = {799893},
  year    = {2022},
  doi     = {10.3389/frobt.2022.799893}
}

@article{radosavovic2024humanoid,
  author  = {Radosavovic, Ilija and Xiao, Tete and Zhang, Bike and Darrell, Trevor and Malik, Jitendra and Sreenath, Koushil},
  title   = {Real-world humanoid locomotion with reinforcement learning},
  journal = {Science Robotics},
  volume  = {9},
  number  = {89},
  pages   = {eadi9579},
  year    = {2024},
  doi     = {10.1126/scirobotics.adi9579}
}

@article{prudencio2024offlinesurvey,
  author  = {Prudencio, Rafael Figueiredo and Maximo, Marcos R. O. A. and Colombini, Esther Luna},
  title   = {A survey on offline reinforcement learning: Taxonomy, review, and open problems},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  volume  = {35},
  number  = {8},
  pages   = {10237--10257},
  year    = {2024},
  doi     = {10.1109/TNNLS.2023.3250269}
}

@inproceedings{xu2022cpq,
  author    = {Xu, Haoran and Zhan, Xianyuan and Zhu, Xiangyu},
  title     = {Constraints penalized {Q}-learning for safe offline reinforcement learning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {36},
  number    = {8},
  pages     = {8753--8760},
  year      = {2022},
  doi       = {10.1609/aaai.v36i8.20855}
}

@article{tian2023meshsurvey,
  author  = {Tian, Yating and Zhang, Hongwen and Liu, Yebin and Wang, Limin},
  title   = {Recovering {3D} human mesh from monocular images: A survey},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {45},
  number  = {12},
  pages   = {15406--15425},
  year    = {2023},
  doi     = {10.1109/TPAMI.2023.3298850}
}

@inproceedings{shin2024wham,
  author    = {Shin, Soyong and Kim, Juyong and Halilaj, Eni and Black, Michael J.},
  title     = {{WHAM}: Reconstructing world-grounded humans with accurate {3D} motion},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2070--2080},
  year      = {2024},
  note      = {arXiv:2312.07531}
}

@inproceedings{rudin2022walkminutes,
  author    = {Rudin, Nikita and Hoeller, David and Reist, Philipp and Hutter, Marco},
  title     = {Learning to walk in minutes using massively parallel deep reinforcement learning},
  booktitle = {Proceedings of the 5th Conference on Robot Learning (CoRL 2021)},
  series    = {PMLR},
  volume    = {164},
  pages     = {91--100},
  year      = {2022},
  note      = {arXiv:2109.11978}
}

@inproceedings{zhuang2024parkour,
  author    = {Zhuang, Ziwen and Yao, Shenzhe and Zhao, Hang},
  title     = {Humanoid parkour learning},
  booktitle = {Proceedings of the 8th Conference on Robot Learning (CoRL)},
  year      = {2024},
  note      = {arXiv:2406.10759. Unitree H1; automatic terrain-difficulty curriculum}
}

@article{acosta2022simvalidation,
  author  = {Acosta, Brian and Yang, William and Posa, Michael},
  title   = {Validating robotics simulators on real-world impacts},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {7},
  number  = {3},
  pages   = {6471--6478},
  year    = {2022},
  doi     = {10.1109/LRA.2022.3174367}
}

@article{yang2023wcsacjournal,
  author  = {Yang, Qisong and Sim{\~a}o, Thiago D. and Tindemans, Simon H. and Spaan, Matthijs T. J.},
  title   = {Safety-constrained reinforcement learning with a distributional safety critic},
  journal = {Machine Learning},
  volume  = {112},
  number  = {3},
  pages   = {859--887},
  year    = {2023},
  doi     = {10.1007/s10994-022-06187-8}
}

@article{kim2022offtrc,
  author  = {Kim, Dohyeong and Oh, Songhwai},
  title   = {Efficient off-policy safe reinforcement learning using trust region conditional value at risk},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {7},
  number  = {3},
  pages   = {7644--7651},
  year    = {2022},
  doi     = {10.1109/LRA.2022.3184793}
}

@inproceedings{kim2023sdac,
  author    = {Kim, Dohyeong and Lee, Kyungjae and Oh, Songhwai},
  title     = {Trust region-based safe distributional reinforcement learning for multiple constraints},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  note      = {arXiv:2301.10923}
}

@inproceedings{robey2025jailbreaking,
  author    = {Robey, Alexander and Ravichandran, Zachary and Kumar, Vijay and Hassani, Hamed and Pappas, George J.},
  title     = {Jailbreaking {LLM}-controlled robots},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {11948--11956},
  year      = {2025},
  doi       = {10.1109/ICRA55743.2025.11128119}
}

@article{hundt2025llmrobots,
  author  = {Hundt, Andrew and Azeem, Rumaisa and Mansouri, Masoumeh and Brand{\~a}o, Martim},
  title   = {{LLM}-driven robots risk enacting discrimination, violence, and unlawful actions},
  journal = {International Journal of Social Robotics},
  volume  = {17},
  number  = {11},
  pages   = {2663--2711},
  year    = {2025},
  doi     = {10.1007/s12369-025-01301-x}
}
```

*(Tier 3 BibTeX on request — keys and metadata are in Part D.)*

---

## Part F — Execution order

1. Add Tier 1 + Tier 2 BibTeX to `references.bib` (Part E block).
2. Apply Part A edits E1–E12 (these change claims — re-read each section after editing for flow).
3. Apply Part B edits E13–E23 (mechanical cite insertions + the E15/E19/E21/E22 sentence additions).
4. Recompile; check no `\cite` is undefined and Table 2.1 still fits the page (it gains 5 rows — consider `\small` or splitting safety-axis/manipulation-axis/human-axis groups with `\midrule`s).
5. Record in `CHANGELOG_v13.md`.
6. Optional Tier 3 adoptions (E12 footnote-grade refs, TRAM, Kirschner, SRCPO).

**Items deliberately NOT changed:** the regime-map novelty claims (verified to survive: the strict five-axis benchmark gap, the high-DOF-under-ISO characterisation gap, and the cross-regime comparison are all still unclaimed territory in mid-2026); all canonical origin citations in Part C; the BodySLAM++ design choice; WCSAC as the reimplemented baseline.

# Speaker script — `safety_bigym` FYP presentation

**Total content: ~16:20** (hard cap 18:00) · **Q&A: up to 10:00**
Deck: `safety_bigym_presentation.pptx` — **20 slides: 14 main + 6 Q&A appendix**.
The same notes are embedded per-slide (Keynote → View → Presenter Notes).

Delivery split (department guidance ≈ half slides / half demo): **~9 min slides ·
~2.5 min live demo · ~3.5 min embedded result videos · ~1 min buffer.** The four
embedded result videos (disruption, baseline-vs-constrained, freeze-vs-flee,
speed-scaling) count as demonstration, so the demo share is comfortably ~half.

> **Honesty note (say once, on slide 9):** the trained-policy clips are *scripted
> reconstructions* — real environment, real G1 coworker, real ISO-15066 monitor,
> scripted robot — because the trained snapshots live on the GPU box. Every HUD
> number is measured from the simulation. This mirrors the report's E.2–E.4 figures.

---

## Timing budget

| # | Slide | Clock | Len |
|---|-------|-------|-----|
| 1 | Title | 0:00 | 0:15 |
| 2 | Motivation | 0:15 | 0:55 |
| 3 | Gap + 3 RQs | 1:10 | 0:55 |
| 4 | Benchmark | 2:05 | 1:00 |
| 5 | **LIVE DEMO** | 3:05 | 2:30 |
| 6 | Tasks & curriculum | 5:35 | 1:00 |
| 7 | Architecture | 6:35 | 1:10 |
| 8 | Two-axis protocol | 7:45 | 0:55 |
| 9 | RQ1 — policy works (video) | 8:40 | 1:15 |
| 10 | RQ1 — freeze vs flee (video) | 9:55 | 1:00 |
| 11 | RQ2 — gate (video) | 10:55 | 1:20 |
| 12 | RQ3 — regime map | 12:15 | 1:20 |
| 13 | Conclusion | 13:35 | 1:10 |
| 14 | Thanks → Q&A | 14:45 | 0:10 |
| — | Buffer | 14:55 | 1:25 |

**Pace markers:** live demo by **3:05**, RQ1 by **~8:40**, conclusion by **~13:35**.
If behind at RQ1: trim demo to 2:00 and slide 8 to 40 s.

---

## Slide 1 — Title  *(0:15)*
> Good morning. I'm Ayush. My project asks a safety question that's becoming urgent: when a **learned** humanoid robot works in the same space a person reaches into, what actually keeps that person safe? I'll show you the benchmark I built, the safety architecture, and the one finding I want you to remember — which I'll demo live.

## Slide 2 — Robot learning has left the cage  *(0:55)*
> Two facts. **First**, robot learning has left the cage — commercial humanoids are aimed at tasks done right next to people. **Second**, their policies optimise *task reward only*: no human distance, no ISO safety margin, no contact force, often no human in the observation at all. There's a mature standard for this — **ISO 15066**: keep a velocity-dependent separation, cap contact force — but learned policies target none of it, and no commercial humanoid is validated for collaborative operation. That's the gap.

## Slide 3 — The gap, and three questions  *(0:55)*
> The gap is an intersection: manipulation benchmarks ignore safety; safe-RL benchmarks use abstract costs, low-DoF systems, perfect state; and almost none model the imperfect human perception a real robot has. Three questions. **RQ1** — can constrained RL keep the robot away while still doing the task? **RQ2** — can a runtime filter cut unsafe speed near the human without breaking the task? **RQ3** — can a measurable property of the task tell you which one to use? I answer them in order.

## Slide 4 — Contribution 1: the benchmark  *(1:00)*
> First contribution: the benchmark. `safety_bigym` extends BiGym with a **moving G1 coworker** (a physically-grounded human stand-in; the benchmark is body-agnostic), **ISO-15066 metrics** computed every step on the closest joint-pair, **calibrated perception noise** so the policy sees a *noisy* human estimate, and a **three-task suite spanning two regimes** — persistent vs intermittent co-location. That regime split is the spine of the talk. Let me show it running.

## Slide 5 — LIVE DEMONSTRATION  *(2:30)*
**Stage cue:** windows pre-launched & arranged (see `demo_runbook.md`). MuJoCo viewer forward, printing terminal beside it.
> This is the benchmark running live. The tan figure is the coworker stand-in; the dark robot is the H1 manipulator. *(viewer)* The coworker is parked in the robot's working volume, **reaching at it** on a cycle — separation drops to ~0.15 m. *(terminal)* It prints each reach phase — extend, hold, retract — the target (the robot's end-effector), and the reach gate, live. **Key line:** "this is the *persistent* regime — the human is in the way most of the time; hold onto that."
>
> **Bridge:** "Training these policies took ~450 GPU-hours and the checkpoints live on the lab box, so the trained behaviours I show next are recorded — and, to be transparent, they're scripted reconstructions in the real environment, the way my report's figures are made."

*If the viewer misbehaves:* click the **backup recording on slide 5** (a patrol episode: coworker walks in, reaches at the robot, departs, and returns) and narrate. Don't debug live.

## Slide 6 — Why these tasks, and why a curriculum  *(1:00)*
> Two quick foundations. **Why these tasks?** The task choice *is* the experiment: I picked tasks that span the co-location axis — `saucepan_to_hob`, where human and robot share the hob and counter almost continuously (persistent), versus `dishwasher` and `drawers`, where the robot works at an appliance and the human only passes through in bursts (intermittent). All are BiGym kitchen tasks with expert demos, which the demo-driven CQN-AS backbone needs. **Why a curriculum?** Train directly on the full coworker and avoidance dominates — the robot never discovers the task. So I stage the coworker's behaviour idle → easy → full, bootstrapping competence first. Standard for long-horizon RL, and I report the dependence as a limitation.

## Slide 7 — Contribution 2: the Hybrid Safety Critic  *(1:10)*
> Two arms with different jobs. The **policy** proposes an action maximising task value *minus* λ·cost value — the constrained, **proactive** part, trained online; it learns to keep distance. A **runtime filter** then checks that action against a learned safety value function and can pass, slow, dodge, or veto — the **reactive** part, trained offline and frozen. One note for the specialists: CQN-AS is critic-only, so I re-derived the Lagrangian in **value-based form** (dual-Q argmax) and proved a **C51 support bound** so dense safety shaping doesn't saturate the critic. The headline question: **which arm carries safety?**

## Slide 8 — Measuring safety honestly  *(0:55)*
> Two axes: **proximity** (how often they're too close — exposure) and **velocity-adaptive SSM** (is the robot too fast for the current gap — robot-controllable). Key fact from a frozen-robot sweep: stop the robot completely and proximity only drops ~58%. **~42% of closeness is the human walking into a stationary robot** — exogenous, unremovable. So I judge the robot mainly on the velocity axis and report the uncontrollable tail honestly. Now the results, in three movements.

## Slide 9 — RQ1 · Persistent: train safety in  *(1:15)*
**Stage cue:** click the side-by-side video.
> *(play)* Left, red: the baseline works straight through the coworker's reach — repeated violations, ~0.2 m. Right, green: the constrained policy yields away and returns once the coworker leaves. **0.296 → 0.228, a 23% cut, across three seeds, at 0.76 success** vs 0.85 baseline. Both axes low at once needs policy + speed-scaling — the regime's ceiling, at a real cost, 0.85 → 0.44, which I report. Honesty gem: picking the checkpoint by *peak success* picks **against** the constraint; the avoidance lives in a mid-training basin — two false nulls before I found it. *(Mention here that these are scripted reconstructions; HUD measured.)*

## Slide 10 — RQ1 · Why not just bolt on a filter? Freeze vs flee  *(1:00)*
**Stage cue:** click the freeze-vs-flee video.
> The obvious objection: why not bolt a filter onto the baseline? This is the answer — and half the thesis. A reactive filter acts only once the human is *already* close, and then it can only **freeze or flee**. *(left)* FREEZE — zero-velocity veto — stops the robot, which dwells in the danger zone: proximity unchanged, 0.296 → 0.303. *(right)* FLEE — retreat veto — buys distance, 0.296 → 0.095, but abandons the task: success 0.85 → 0.18, velocity ×6. Every reactive modality hits this wall — a limit the control-barrier literature reaches too. The missing ingredient is **anticipation**, which must be trained into the policy.

## Slide 11 — RQ2 · Intermittent: gate the backstop  *(1:20)*
**Stage cue:** click the speed-scaling video (right).
> Here the story flips. Constrained RL finds **no feasible budget** — inert above the natural cost, task-fatal below; and when λ binds the robot gets *faster*, not safer (WCSAC corroborates). A binary veto shatters the chunked policy — max vel 2.5 → 6 m/s. What works: a learned critic **gating** a graded ISO speed-scaler — critic decides *when*, scaler decides *how*. **dishwasher −50% SSM at −0.10 success; drawers −22% at −0.09**, gate threshold R a deployable dial. *(video)* NO FILTER violates SSM at speed; SPEED-SCALING scales to 0.1 and stays SSM-OK.

## Slide 12 — RQ3 · The regime map  *(1:20)*
> RQ3 unifies and validates. The deciding variable is **measurable** — the gate-active fraction. *(bars)* Persistent saucepan: 61.5% of steps; intermittent: 19–27%. On saucepan the gate fires *more* than the unconditional scaler's own trigger, so "gating" becomes "always slow" — no safe windows. The validation, fixed **in advance**: gating "recovers throughput" only if some R hits success ≥ 0.60 **and** SSM ≤ 0.08; I ran the intermittent winner on the persistent task — **no row passed, exactly as predicted.** *(plot)* Intermittent bends into the safe corner; persistent slides down the diagonal. One picture, the whole thesis.

## Slide 13 — Conclusion  *(1:10)*
> The headline isn't "my method wins" — it's a **decision rule** where the field had defaults. Intermittent: gate a speed-scaler. Persistent: train safety into the policy; reactive filters freeze or flee. The deciding variable is cheap to measure on logged traffic. Five contributions: benchmark, architecture, two-axis protocol, validated regime map, method lessons. Limits, stated: PFL force is wired but blocked by a simulator contact bug, so the claim is SSM-only; and the map rests on three tasks — the named next step is the persistence dial.

## Slide 14 — Thank you  *(0:10)*
> Thank you — happy to take questions. *(Leave takeaway on screen; appendix 15–20 ready.)*

---

## Q&A bank  *(answer in 2–3 sentences, then offer the appendix slide)*

| Likely question | Crisp answer | Slide |
|---|---|---|
| Why a G1 as the "human"? | Physically-grounded stand-in; safety properties are biomechanical; benchmark is body-agnostic (SMPL-H path exists). About co-location dynamics, not human realism. | 19 |
| Isn't −23% modest? | ~42% of proximity is exogenous; −23% captures ~40% of the *reducible* part — reactive filters captured ~none gracefully. Frequency is controllable; the worst single approach isn't, and I report that. | 8, 20 |
| Are the videos real rollouts? | Scripted reconstructions — real env/coworker/ISO monitor, scripted robot (snapshots are on the GPU box); every HUD number is measured. Same method as the report's figures. The *numbers* are from the full benchmark (180/60 episodes). | 9 |
| Three tasks — is the map real? | Observational on three tasks; I'm explicit. The causal test is the **persistence dial**: fix the task, vary only the coworker's dwell, predict the frontier flattens as gate-activity rises. | — |
| PFL reads zero — is safety real? | SSM/geometric side is fully live and is the headline. PFL is wired end-to-end but blocked by a BiGym/MuJoCo contact bug; I scope the claim to SSM and flag PFL as the top follow-up. | — |
| Why CQN-AS / value-based Lagrangian? | Need demos *and* a cost channel; CQN-AS gives both but is critic-only, so dual-Q argmax + a C51 support bound. | 7 |
| PID instability — weakness? | At the feasibility boundary the dual variable has too little signal (λ = 0 / 0.27 / 3.86 across seeds). Fixing λ makes it reproducible — a transferable lesson. | 15, 16 |
| Sim-to-real? | Three gaps: perception, dynamics, contact fidelity. The reduction already survives noisy perception; next is real BodySLAM++, domain randomisation, a real H1 under ethics review. | 19 |

---

## Delivery tips
- **Rehearse to 16:20 three times** with a timer; the demo + four video clicks are the variable parts — practise the window arrangement and the click cues.
- Use **Presenter mode**; one idea per slide; narrate, don't read bullets.
- **GenAI mention** (verbal, slide 7 or the demo bridge): *"Stack is Python / MuJoCo / PyTorch on RoboBase; I used an LLM assistant for boilerplate, debugging, and report editing — in the report's GenAI appendix."*
- Cuttable if long: slide 8 → 40 s; the optional `demo_safety_visual.py` beat.

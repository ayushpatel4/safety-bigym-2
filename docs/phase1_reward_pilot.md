# Phase 1.4 — Reward-on Pilot (DrQ-V2+)

> **Historical reference.** The standalone DrQ-V2+ reward-on pilot is no longer
> the active path. The CQN-AS E1.4 replacement degenerated when run without
> demos, and the user moved the off/oracle/noisy observation-channel question
> into Phase 3 as E3.6. See [PHASE3_OVERVIEW.md](PHASE3_OVERVIEW.md) and
> [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

Branch: `safety-critic/phase-1-bodyslam-wrapper` (same branch as the
BodySLAMWrapper build — this is a follow-on experiment, not a new phase).

E1.1 ran the BodySLAMWrapper against pure-BC ACT on four tasks and came
back negative — `oracle` failed the **20% SSM-violation-rate reduction**
criterion in all four cells, and `oracle ≈ noisy` in three of four
([phase1_obs_ablation_results.json](../phase1_obs_ablation_results.json)).
Before declaring the contingency on the master plan, we need to rule out
one alternative: **maybe the channel doesn't help under BC simply because
BC has no gradient signal that would teach the policy to attend to it.**
That's what Phase 1.4 tests.

This document covers the wiring, the experiment design, and the
comparison against E1.1.

---

## What's implemented

### `env.safety` Hydra block

[cfgs/env/safety_bigym.yaml](../cfgs/env/safety_bigym.yaml) gained a
`safety:` block that surfaces the per-step violation penalty as a Hydra
override:

```yaml
env:
  safety:
    add_violation_penalty: false  # default off; opt in per launch
    violation_penalty: 0.05
```

[safety_bigym_factory.py:155-167](../safety_bigym/envs/safety_bigym_factory.py#L155)
reads this block and feeds it into the `SafetyConfig` passed to
`SafetyBiGymEnv`. The reward path itself
([safety_env.py:757-765](../safety_bigym/envs/safety_env.py#L757)) was
already in place — gated by `add_violation_penalty`, untouched by Phase
1. So the penalty is a config-only opt-in: every existing run defaults
to the same behaviour it had before.

Regression guard:
[tests/test_safety_env.py::test_violation_penalty_applied_when_enabled](../tests/test_safety_env.py)
constructs two envs (penalty on / off), forces `_step_safety_info` with
SSM and PFL flags set, and asserts the reward differs by exactly
`violation_penalty` per violating step. Catches future changes to the
reward path that might double-count, miss the PFL flag, or apply the
penalty when no violation flag is set.

### DrQ-V2+ launch

[cfgs/launch/drqv2plus_pixel_safety_bigym.yaml](../cfgs/launch/drqv2plus_pixel_safety_bigym.yaml)
is adapted from
[robobase's drqv2plus_pixel_rlbench_demo_driven launch](../../robobase/robobase/cfgs/launch/drqv2plus_pixel_rlbench_demo_driven.yaml).
Key differences from the dp/act launches:

- `is_imitation_learning: false` — engages robobase's online RL training
  loop, not BC pretraining.
- `num_train_frames: 200000` — DrQ-V2+ trains via env interaction; demos
  warm-start the replay but the policy actually rolls out.
- `replay_size_before_train: 2000` — bumped from rlbench's 500 to clear
  robobase's `replay_size_before_train * action_repeat * action_sequence
  >= env.episode_length` assertion (reach_target_single = 2000).
- `method=drqv2` with `bc_lambda: 1.0` — actor loss combines policy
  gradient and BC over demo transitions, so the agent uses both the
  reward signal *and* the demonstrator's actions.
- `use_self_imitation: true` — successful rollouts get re-added to the
  demo buffer; pairs well with the reward shaping.
- `env.safety.add_violation_penalty: true` is set at the launch level so
  this launch is **reward-on by default**.

### Driver script

[scripts/phase1_reward_pilot.py](../scripts/phase1_reward_pilot.py) is
the smallest specialisation of the
[phase1_obs_ablation.py](../scripts/phase1_obs_ablation.py) shape:

- Two tasks (`reach_target_single`, `drawers_close_all`) × three obs modes
  = 6 train cells, 30 eval cells.
- `--train` / `--eval` / `--run` / `--smoke` (same surface as
  obs_ablation).
- `--run` writes `phase1_reward_pilot_results.json` and prints a
  decision table with the **PASS** flag when off → mode SSM reduction
  ≥ 20%.
- W&B tags: `phase-1-reward`, `drqv2plus`, `<bodyslam_mode>` — distinct
  from the E1.1 `obs-ablation` tag space so the two studies don't get
  confused.

---

## What the experiment checks

**Hypothesis.** The `human_pos_estimate` channel is **only useful when
the training algorithm has access to a safety reward signal**. Under BC
(E1.1), the policy fits action distributions without using rewards, so
the channel is structurally invisible to the learning objective.

**Setup.** Two tasks (`reach_target_single`, `drawers_close_all`) × three
obs modes (off / oracle / noisy) = 6 cells. Online RL (DrQ-V2+) with
the per-step penalty turned on. 200 k env frames per cell. The variable
that moves is the obs channel; reward shaping is held constant (on for
all six cells). Evaluation: 5 disruption types × 20 episodes each,
post-training.

**Task choice rationale.** `reach_target_single` is the cleanest
baseline: E1.1 ACT achieved `episode_success = 0.88` on `off`, so the
task is definitively learnable. `drawers_close_all` is structurally
different from anything in the E1.1 sweep — multi-step contact-rich
manipulation, longer episodes (median ~132 control steps vs reach's
~53), 4 floating DOFs, 60 demonstrator demos. It provides an
**independent test** of the channel-plus-reward hypothesis on a task
where the policy has more degrees of freedom to either route safety
information through, or ignore it. Pairing a short clean task (reach)
with a longer complex task (drawers_close_all) covers both ends of the
difficulty spectrum without doubling the GPU bill of a full 4-task
sweep.

**Reward calibration.** From E1.1 reach_target_single off-cell:
`episode_reward ≈ 0.88` over `episode_length ≈ 53` ⇒ ≈ **0.017 task
reward / step**. `violation_penalty = 0.05` ≈ 3× per-step task reward,
applied only on violating steps. With baseline `ep_ssm_violation_rate ≈
0.17`, the expected per-episode safety cost is
`0.05 × 0.17 × 53 ≈ 0.45` — large enough to create real optimisation
pressure against violations, small enough that the agent still has
incentive to complete the task.

**Primary metric.** `ep_ssm_violation_rate` (proportion of episode steps
where the SSM separation envelope was breached), averaged across the 5
disruption types per cell.

### Decision rule

| Outcome (off → mode SSM reduction)               | Conclusion                                                                                                 | Next step                                                                                          |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| oracle and/or noisy ≥ **+20%**                    | Channel + reward gradient together work. BC was the bottleneck, not perception.                            | **Greenlight Phase 3** (constrained RL with the obs channel) at full scope.                        |
| oracle and noisy both within ±5% of off           | Channel doesn't help even with a reward gradient.                                                          | **Pivot to Phase 2** (Safety Value Function as a runtime filter that uses the channel directly, bypassing the policy). Phase 3 still happens but plans for `bodyslam=off`. |
| off wins by more than 5%                          | Reward shaping alone suffices; the channel is overhead the policy is paying to ignore.                     | **Retire the BodySLAMWrapper** for downstream phases. Phase 2 still uses `human_pos` (from `info["safety"]`) directly; Phase 3 trains with `bodyslam=off + penalty`. |
| oracle ≥ +20% but noisy ≈ off                     | Channel works under perfect perception but not under realistic noise.                                       | Phase 2 perception/filter work is the limiting factor — escalate it before Phase 3.                |

The asymmetry between cases 3 and 4 is intentional: case 3 means the
channel is genuinely useless given the training paradigm; case 4 means
the channel is informative but the noise we model is too aggressive.

---

## How this differs from E1.1

| Axis                       | E1.1 (BC obs-ablation)                                              | E1.4 (RL reward-on pilot)                                              |
|----------------------------|---------------------------------------------------------------------|------------------------------------------------------------------------|
| Question                   | Does an obs channel help BC learn safer behaviour?                  | Does an obs channel help **when** the reward signal can teach the policy to use it? |
| Training algorithm         | ACT / DiffusionPolicy (pure BC)                                     | DrQ-V2+ (demo-driven online RL with `bc_lambda=1.0`)                   |
| Uses env reward in loss?   | **No** — BC clones action distributions; reward is not a gradient.  | **Yes** — actor and critic both use `r_t`; safety penalty is part of `r_t`. |
| Demos                      | 30 (action-cloning target)                                          | 30 (warm-start the replay buffer + BC regularizer)                     |
| Tasks                      | 4 (reach, dishwasher_close, drawers_open_all, saucepan_to_hob)      | 2 (reach_target_single, drawers_close_all)                               |
| Cells                      | 12 (4 tasks × 3 modes, ACT only)                                    | 6 (2 tasks × 3 modes)                                                  |
| Train budget per cell      | 100 k BC pretrain steps                                             | 200 k env interaction frames                                           |
| Wall clock per cell        | ~1–2 GPU-hours                                                      | ~3–4 GPU-hours                                                         |
| Total wall clock           | ~12–24 GPU-hours                                                    | ~18–24 GPU-hours                                                       |
| Eval metric                | `ep_ssm_violation_rate`, off → mode reduction                       | Same — directly comparable                                             |

The two experiments are **complementary**, not redundant:

- **E1.1 answers: under realistic BC, does adding a perception channel
  matter?** This is the question a robotics practitioner cares about
  most — BC is the dominant paradigm for manipulation right now, and
  the answer informs whether a safety-aware perception module is worth
  building for any team using BC. Our answer: **no** (oracle did not
  meet the criterion on any of the 4 ACT cells).

- **E1.4 answers: is the *channel* useless, or is the *training
  algorithm* the bottleneck?** This isolates which subsystem is at
  fault. If E1.4 passes, the channel is fine; we just need a training
  paradigm that uses rewards (Phase 3). If E1.4 fails, the channel is
  fundamentally not useful as a policy input regardless of how we train
  — at which point Phase 2 (the channel feeds a separate filter, not
  the policy) becomes the right place to invest.

### Which is "more useful"?

Neither is strictly more useful — they answer different questions.

- For the **master plan's branch decision** (greenlight Phase 2 vs Phase 3
  vs both vs neither), **E1.4 is decisive**. E1.1's negative result is
  consistent with multiple causes (channel useless, BC ignores rewards,
  or both); only E1.4 disambiguates.

- For a **paper / writeup justifying the work to the field**, E1.1 is
  the headline result. "Adding a perception channel to imitation
  learning doesn't reduce ISO 15066 violations" is the surprising
  finding people will quote. E1.4 is the diagnostic that explains why.

The cheapest reading order: **E1.1 first, then E1.4**. E1.1 establishes
the negative under BC; E1.4 either rescues the channel under RL or
condemns it more broadly. We already have E1.1 — E1.4 is the next step.

---

## Mechanistic detail: why E1.1 was the wrong test for the channel-is-useful hypothesis

DP and ACT in robobase ([method/diffusion.py](../../robobase/robobase/method/diffusion.py),
[method/act.py](../../robobase/robobase/method/act.py)) compute the
actor loss as a negative-log-likelihood (or DDPM noise-prediction loss)
between the policy's predicted action distribution and the demonstrator's
action. There is **no term that involves the env reward**.

The demos used were collected on vanilla BiGym (no human in the scene),
so every demo timestep has:

- A `human_pos_estimate` value (synthesised by `AMASSDemoPositionProvider`
  during demo replay — see
  [perception/demo_position_provider.py](../safety_bigym/perception/demo_position_provider.py)).
- A demonstrator action **uncorrelated with that value** (the
  demonstrator was working solo, didn't see or react to a human).

When BC fits `p(action|obs)`, the only way `human_pos_estimate` matters
is if `action` and `human_pos_estimate` are statistically dependent in
the training data. In our case they're not. The encoder learns to route
information from informative features to action prediction; the
`human_pos_estimate` channel just gets routed past the network with
zero or near-zero weight. That's the "marginalisation" pattern we
observed: oracle and noisy giving near-identical SSM rates, because
both arms route the channel out.

**The fix is to make the loss depend on safety violations.** That's
what reward shaping + RL provides: when the agent steps into a
violation, `r_t` decreases, the Q-function's target decreases for that
state, and the policy gradient pushes away from actions that led to
that state. If the policy can predict the safety penalty from
`human_pos_estimate`, the obs channel becomes useful and the gradient
will push the encoder to use it. That's the chain Phase 1.4 tests.

If even this fails (E1.4 case 2: oracle ≈ noisy ≈ off), the conclusion
is stronger: the channel is not informative enough about the safety
signal for any policy to extract benefit, regardless of how we train.
At that point, the runtime filter (Phase 2) is the only path that uses
the channel productively — it consumes `human_pos_estimate` directly to
veto unsafe actions, with no policy gradient required.

---

## Files

### Created

- [cfgs/launch/drqv2plus_pixel_safety_bigym.yaml](../cfgs/launch/drqv2plus_pixel_safety_bigym.yaml)
- [scripts/phase1_reward_pilot.py](../scripts/phase1_reward_pilot.py)
- [docs/phase1_reward_pilot.md](phase1_reward_pilot.md) (this file)

### Modified

- [cfgs/env/safety_bigym.yaml](../cfgs/env/safety_bigym.yaml) — adds
  `env.safety.{add_violation_penalty, violation_penalty}` block,
  default off.
- [safety_bigym/envs/safety_bigym_factory.py:155-167](../safety_bigym/envs/safety_bigym_factory.py#L155-L167) —
  reads `cfg.env.safety` and threads it into `SafetyConfig`.
- [tests/test_safety_env.py](../tests/test_safety_env.py) — adds
  `test_violation_penalty_applied_when_enabled`.

### Not modified

- The BodySLAMWrapper itself. Phase-1 obs plumbing stands; this is purely
  a training-loop swap + reward-knob exposure.
- RoboBase. DrQ-V2+ + demo-driven replay + `bc_lambda` already exists
  upstream; no patch needed.
- Phase-1 doc [phase1_bodyslam_wrapper.md](phase1_bodyslam_wrapper.md) —
  E1.4 supersedes its references to E1.2 / E1.3 (those experiments
  remain "deferred until obs channel is shown useful").

---

## Usage

### Local smoke (before GPU hand-off)

```bash
cd safety_bigym
pytest tests/test_safety_env.py -v   # confirms penalty regression test
AMASS_DATA_DIR=$AMASS python train_safety.py \
  launch=drqv2plus_pixel_safety_bigym \
  env=safety_bigym/reach_target_single \
  bodyslam=oracle \
  num_train_frames=100 num_pretrain_steps=0 demos=2 \
  num_eval_episodes=0 replay_size_before_train=2000 \
  wandb.use=false hydra.run.dir=/tmp/p14_smoke
```

Mac note: Don't set `MUJOCO_GL=egl` locally — EGL is GPU/Linux only. The
script's printed commands include it for the GPU box.

### GPU

```bash
# 1. Train all 6 cells (~18-24 GPU-hours total)
python scripts/phase1_reward_pilot.py --train > /tmp/p14_train.sh
bash /tmp/p14_train.sh

# 2. Pick peak-by-W&B-curve snapshots; edit SNAPSHOTS in the script
$EDITOR scripts/phase1_reward_pilot.py

# 3. Eval (writes phase1_reward_pilot_results.json + decision table)
python scripts/phase1_reward_pilot.py --run
```

The `--run` table is grouped by task — within each task block, prints
**PASS** in the rightmost column for any mode meeting the 20% reduction
criterion. The final lines summarise per-task pass/fail and the overall
branch decision (`PASS` on either task is sufficient to greenlight
Phase 3).

---

## Hand-off checklist

Before kicking off the GPU runs:

- [ ] `pytest tests/` green locally (1 pre-existing failure on
      `test_no_episode_safety_until_done` is unrelated, predates Phase 1).
- [ ] Local 100-frame smoke for `bodyslam ∈ {off, oracle, noisy}` shows
      `Inserting BodySLAMWrapper(...)` for the relevant modes and exits
      rc=0.
- [ ] Branch `safety-critic/phase-1-bodyslam-wrapper` pushed with the
      Phase-1.4 commits on top.
- [ ] Robobase drift patch already applied on the GPU clone (Phase 0
      requirement, unchanged here).
- [ ] `AMASS_DATA_DIR` exported on the GPU box.

After the runs:

- [ ] Author [phase1_observation_results.md](phase1_observation_results.md)
      covering **both** E1.1 (BC negative) and E1.4 (RL — whichever
      branch). Reference both result JSONs and the decision-rule table
      from this doc.
- [ ] Update [HYBRID_SAFETY_CRITIC_PLAN.md](../../.claude/HYBRID_SAFETY_CRITIC_PLAN.md)'s
      Phase 1 section to reflect the actual outcome (oracle channel
      result, E1.2 / E1.3 status: still deferred or now scheduled, and
      the Phase 2 vs Phase 3 priority).
- [ ] Open a PR off `safety-critic/phase-1-bodyslam-wrapper` once the
      writeup lands.

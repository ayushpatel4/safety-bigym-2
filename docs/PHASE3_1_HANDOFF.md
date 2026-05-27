# Handoff — P3.1: the Lagrangian glue (constrained RL)

> **Historical handoff.** P3.1 is now code-complete and unit-tested; the live
> remaining gate is the GPU smoke / experiment handoff described in
> [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and
> [PHASE3_OVERVIEW.md](PHASE3_OVERVIEW.md). This prompt is preserved for
> implementation context and should not be used as the current task list.

Created 2026-05-20. Single coding milestone. Read this top-to-bottom before touching code.

Companion docs (all in `safety_bigym/docs/`): [PHASE3_OVERVIEW.md](PHASE3_OVERVIEW.md) (scope + status),
[UPDATED_PROJECT_PLAN.md](UPDATED_PROJECT_PLAN.md) (Phase 3 design), [cqn_as_integration_notes.md](cqn_as_integration_notes.md)
(vendor gotchas), [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) (live status), [CLAUDE.md](CLAUDE.md) (workspace rules).

---

## Paste-ready session prompt

> You're implementing **P3.1 — the Lagrangian "glue"** that turns the staged CQN-AS cost pipeline into
> actual constrained RL. Working dir: `/Users/ayushpatel/Documents/FYP3` (active git repo is
> `safety_bigym/`). Venv: `safety_bigym/venv/`. Export
> `AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU` (or the GPU box's CMU path).
>
> **READ FIRST, top-to-bottom:**
> - `safety_bigym/docs/PHASE3_OVERVIEW.md` (goal, architecture, what's done/left, contingencies)
> - `safety_bigym/docs/UPDATED_PROJECT_PLAN.md` Phase 3 section (Option B-value-mean/CVaR, the
>   cost/λ/workspace formulas, the E3.* experiments, the possible-outcomes table)
> - `safety_bigym/docs/cqn_as_integration_notes.md` (vendor gotchas — tensordict, collate,
>   TensorDict metrics, worker striping)
> - `CLAUDE.md` (PFL bug, "don't launch multi-hour training", branch strategy)
>
> **GATE — DO NOT START THE GPU SWEEP UNTIL THIS IS GREEN:**
> The Workstream D 50k validation run must show demos+workspace shaping produce a non-degenerate
> policy (`episode_reward>0` on some episodes, `episode_length` staying long, not 31-step evacuation).
> P3.1 builds ON TOP of that working base. You CAN write + unit-test all the P3.1 code in parallel
> (it's GPU-free); you just can't trust/launch the constrained-RL training until D-validation passes.
>
> **WHAT P3.1 IS (Option B-value-mean to start — dual Q-networks):**
> The per-step continuous cost `c_t` already flows end-to-end into the training batch as
> `batch["cost"]` and `batch["max_cost"]` (P3.0c). But `CQNASAgent.update()` IGNORES them today.
> P3.1 adds:
>   1. A cost Q-network `Q_c` trained (Bellman regression, **PER-ENV-STEP backup**) on `batch["cost"]`.
>   2. A PID-controlled Lagrangian multiplier `λ` on rolling-mean cost.
>   3. Dual-Q action selection: `argmax_a [Q_r(s,a) − λ·Q_c(s,a)]` at EACH coarse-to-fine level.
>   4. Logging of `λ`, rolling cost, `Q_c` loss to W&B.
>
> **HARD CONSTRAINT — DO NOT EDIT THE VENDORED AGENT:**
> `agent.py` header says cost-Q must land in "a sibling module that subclasses or composes this agent
> rather than editing it." So create e.g. `safety_bigym/agents/cqn_as/lagrangian_agent.py` with a
> `LagrangianCQNASAgent` that subclasses/composes `CQNASAgent`. `train_cqn_as.py`'s `make_agent` /
> `cfgs/agent` should select it via a Hydra flag (e.g. `agent=cqn_as_lagrangian`) so plain CQN-AS
> still works for the D smoke and E1.4-style runs.
>
> **THE CRUX — RESOLVE THIS DESIGN DECISION FIRST (it shapes everything):**
> `Q_c` must score the SAME coarse-to-fine action bins that `Q_r` does, because dual-Q selection
> happens inside `C2FCritic.get_action()` (`agent.py:415`) where each level does
> `argmax_q = qs.max(-1)[1]` (line ~428) over per-bin Q values. Two options:
>   - **(A)** Make `Q_c` a SECOND C2F-structured critic (clone `C2FCritic`, regress on cost instead of
>     reward; scalar/expected-cost head for B-mean, not C51). Then dual-Q is clean: at each level
>     compute `q_r` and `q_c` the same way and `argmax(q_r − λ·q_c)`. Downside: warm-start from the
>     Phase 2 MLP SVF (`filters/cost_critic.py CostCritic`) doesn't map cleanly — defer warm-start,
>     fresh-init `Q_c`.
>   - **(B)** Keep `Q_c` as the MLP `CostCritic(s,a)` from `filters/cost_critic.py` (enables SVF
>     warm-start via `warm_start_from_svf(force_sign_flip=True)`). Downside: you must decode each
>     candidate bin to a continuous action and evaluate the MLP per bin per level — awkward and slow
>     inside `get_action`.
> RECOMMENDATION: start with **(A)** a second C2F cost critic for B-value-mean (clean dual-Q,
> stationary target), note in docs that SVF warm-start is deferred to the B-value-CVaR variant. But
> CONFIRM this choice with the user before building — it's the load-bearing architectural call.
>
> **KEY HOOK POINTS (read these before designing):**
> - `safety_bigym/agents/cqn_as/agent.py` — `CQNASAgent` (617), `act()` (744, calls
>   `critic_target.get_action`), `update()` (841, unpacks batch, ignores cost/max_cost),
>   `update_critic()` (778, the C51 reward loss), `update_target_critic()` (877, soft update).
>   `C2FCritic.get_action()` (415) and `forward_each_level` (244) are the coarse-to-fine argmax you
>   must make cost-aware.
> - `safety_bigym/agents/cqn_as/utils.py:71 to_torch_pixel_tensor_dict` — already puts cost/max_cost
>   in the batch dict (lines 87-120).
> - `safety_bigym/agents/cqn_as/replay_buffer.py:_sample` — already accumulates n-step discounted cost
>   + max_cost (per-env-step granularity preserved — DO NOT regress this; the plan calls it out).
> - `safety_bigym/filters/cost_signal.py` — `compute_cost` (`c_t = min(1, max(c_ssm, c_pfl))`,
>   `d_buffer=0.3`). NOTE `c_pfl ≡ 0` under the open PFL bug, so `c_t == c_ssm` in practice today.
>   That's fine for P3.1.
> - `safety_bigym/filters/cost_critic.py` — the MLP `CostCritic` + `warm_start_from_svf(force_sign_flip=True)`.
> - `train_cqn_as.py` — the train loop (`agent.update(batch)`, `agent.update_target_critic(step)`); the
>   λ updater + rolling-cost tracker live here or in the agent. `Workspace` already logs per-step `c_t`.
>
> **HYPERPARAMETERS (from the plan, start here):**
> - cost: `d_buffer=0.3`
> - workspace: current bounded default is `r_ws=0.4`, `β=0.05`, `workspace_excess_cap=1.0` (`env.safety.add_workspace_penalty=true` is CLI-overridable)
> - λ PID: `K_I=1e-3, K_P=1e-2, K_D=0, λ_max=100`, cost budget `d=0.01`
> - λ update: `cost_violation = rolling_mean_cost − d`; `λ = clip(λ + K_I·cv + K_P·cv + K_D·Δcv, 0, λ_max)`
>
> **CRITICAL CORRECTNESS GOTCHAS:**
> - **PER-ENV-STEP cost backup**, not per-K-step-chunk. CQN-AS runs K-step action sequences; if `Q_c`'s
>   Bellman target averages cost over the chunk, the policy satisfies the mean budget while spiking
>   violations inside chunks. The batch already carries per-step cost — keep the backup per-step.
> - **λ enters ONLY at action-selection time** (the argmax), NEVER in any critic's regression target.
>   That's what keeps both Q-networks' targets stationary. Don't fold λ into the cost the critic regresses.
> - `Q_c` needs its **own target network + soft update** (mirror `update_target_critic` for the cost critic).
> - `agent.update()` returns a **TensorDict** (not a dict) — never bool-test it; iterate `.items()` once
>   (integration notes §3).
>
> **BRANCH:** do this on `safety-critic/phase-3-constrained-rl` (NOT main — keep main stable so the GPU
> box can re-run D). Per CLAUDE.md, ask the user before committing; never commit to main directly.
>
> **TESTS (pure-Python, no MuJoCo — mirror `tests/test_cqn_as_adapter.py` / `test_cost_critic.py` style):**
> - λ PID updater: monotonic increase when rolling cost > d, decrease when < d, clamps to [0, λ_max].
> - Dual-Q selection: with λ=0 the argmax equals plain `Q_r`'s argmax; with large λ it shifts toward
>   low-`Q_c` bins. Use small stub tensors.
> - `Q_c` Bellman update: loss is finite and decreases on toy data; per-step (not chunk) target shape.
> - Agent selection: `agent=cqn_as` still works (no cost-Q); `agent=cqn_as_lagrangian` instantiates the
>   subclass and consumes `batch["cost"]`.
>
> **SMOKE (after unit tests green; needs `tensordict` — GPU box only):**
> ```
> python train_cqn_as.py env=safety_bigym/saucepan_to_hob disruption=coworker_train bodyslam=oracle \
>   num_demos=10 num_train_frames=2000 env.safety.add_workspace_penalty=true \
>   agent=cqn_as_lagrangian wandb.use=false
> ```
> Expect: trains clean, λ moves off its init, `q_c_loss` logged, no per-step/chunk shape errors.
> Then hand the full E3.* GPU sweeps to the human (don't launch multi-hour training yourself).
>
> **DEFINITION OF DONE (P3.1, the coding milestone — NOT the full Phase 3 result):**
> - `LagrangianCQNASAgent` (sibling module) with `Q_c` + target, λ PID, dual-Q selection, logging.
> - `batch["cost"]` consumed; per-step backup verified.
> - Hydra-selectable; plain CQN-AS untouched and still working.
> - Unit tests green; 2000-frame `agent=cqn_as_lagrangian` smoke clean.
> - Update IMPLEMENTATION_STATUS Workstream P3.1 + the `Q_c`-architecture decision in the decision log.
> The E3.* sweeps (cost form, budget Pareto, architecture A/B-mean/B-CVaR, β, WCSAC baseline) are
> separate downstream tasks, gated on this landing + D-validation.

---

## Why this is one milestone, not the whole of Phase 3

P3.1 is the *coding* deliverable — the machinery. The *scientific* deliverable (does constrained RL
reduce SSM violations >50% with <10% task-reward loss?) comes from the E3.* GPU sweeps that run
**after** P3.1 lands and the D-validation confirms a non-degenerate base policy. Keep the two separate:
P3.1's "done" is "the Lagrangian agent trains cleanly and the knobs work," not "the safety result is in."

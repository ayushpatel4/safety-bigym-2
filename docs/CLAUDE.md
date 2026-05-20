# FYP3 Workspace Orientation

## What this is

A multi-project workspace for a safety-aware manipulation robot. The active project is [safety_bigym/](../) — a BiGym extension that injects a live SMPL-H human (driven by AMASS CMU motion clips) into manipulation scenes and adds ISO 15066 safety monitoring (Speed and Separation Monitoring + Power and Force Limiting). Policies are trained through RoboBase ([robobase/](robobase/)), which is a local clone rather than a pip-installed dep.

The goal of this workspace: implement the Hybrid Safety Critic described in [.claude/UPDATED_PROJECT_PLAN.md](../../.claude/UPDATED_PROJECT_PLAN.md) — a constrained-RL policy plus a decoupled Safety Value Function filter at runtime.

## Directory layout

- [safety_bigym/](../) — main package, the only git repo (remote `github.com/ayushpatel4/safety-bigym-2.git`)
- [bigym/](bigym/) — upstream BiGym environment, local clone
- [robobase/](robobase/) — RoboBase training framework, local clone with its own git history
- [SMPLSim/](SMPLSim/) — physics-aware SMPL simulator (separate project)
- [CMU/](CMU/) — AMASS CMU motion capture dataset
- [smplx/](smplx/) — SMPL-X body models
- [vids/](vids/) — rendered evaluation videos

**Important:** FYP3 root is **not** a git repo. The git boundary is [safety_bigym/](../). Always `cd safety_bigym` before running git.

## Key files

- Env wrapper: [safety_bigym/safety_bigym/envs/safety_env.py](../safety_bigym/envs/safety_env.py)
- RoboBase factory: [safety_bigym/safety_bigym/envs/safety_bigym_factory.py](../safety_bigym/envs/safety_bigym_factory.py)
- ISO 15066 monitor: [safety_bigym/safety_bigym/safety/iso15066_wrapper.py](../safety_bigym/safety/iso15066_wrapper.py)
- PFL limits table: [safety_bigym/safety_bigym/safety/pfl_limits.py](../safety_bigym/safety/pfl_limits.py)
- Config (`SSMConfig`, `HumanConfig`, `SafetyConfig`): [safety_bigym/safety_bigym/config.py](../safety_bigym/config.py)
- Human controller: [safety_bigym/safety_bigym/human/human_controller.py](../safety_bigym/human/human_controller.py)
- PD controller: [safety_bigym/safety_bigym/human/pd_controller.py](../safety_bigym/human/pd_controller.py)
- AMASS loader: [safety_bigym/safety_bigym/motion/amass_loader.py](../safety_bigym/motion/amass_loader.py)
- Scenario sampler: [safety_bigym/safety_bigym/scenarios/scenario_sampler.py](../safety_bigym/scenarios/scenario_sampler.py)
- Training entrypoint (ACT/DP via RoboBase): [safety_bigym/train_safety.py](../train_safety.py)
- Training entrypoint (CQN-AS, Phase 3): [safety_bigym/train_cqn_as.py](../train_cqn_as.py)
- CQN-AS vendor: [safety_bigym/safety_bigym/agents/cqn_as/](../safety_bigym/agents/cqn_as/) (agent, env_adapter, replay_buffer, eval_video; notes in [docs/cqn_as_integration_notes.md](../docs/cqn_as_integration_notes.md))
- Phase 2 SVF critic + filter: [safety_bigym/safety_bigym/filters/](../safety_bigym/filters/) (`critic.py`, `cost_critic.py`, `cost_signal.py`, `cql_trainer.py`, `runtime_wrapper.py`)
- Top-level Hydra config: [safety_bigym/cfgs/safety_config.yaml](../cfgs/safety_config.yaml)
- CQN-AS Hydra root: [safety_bigym/cfgs/cqn_as_config.yaml](../cfgs/cqn_as_config.yaml)
- Base env config: [safety_bigym/cfgs/env/safety_bigym.yaml](../cfgs/env/safety_bigym.yaml)
- Tests: [safety_bigym/tests/](../tests/) (no pytest.ini; run `pytest tests/` from `safety_bigym/`)

## Critical gotchas

- **AMASS data path is env-var-driven.** Export `AMASS_DATA_DIR` before running anything:
  - `export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU`
  - The base env config ([cfgs/env/safety_bigym.yaml](../cfgs/env/safety_bigym.yaml)) resolves `motion_clip_dir` via `${oc.env:AMASS_DATA_DIR,null}`; the factory reads the env var at `cfg.env.get("motion_clip_dir", os.environ.get("AMASS_DATA_DIR"))` at [safety_bigym_factory.py:129-131](../safety_bigym/envs/safety_bigym_factory.py#L129-L131).
  - Scripts under [safety_bigym/scripts/](../scripts/) raise a clear `RuntimeError` at import time when the var is unset — no silent fallback.
  - Pytest AMASS-dependent tests (`test_load_clip`, scenario phases) skip cleanly when `AMASS_DATA_DIR` is unset.
  - NOTE: a previous version of this file referenced `safety_bigym.paths.get_amass_data_dir()`. That helper does **not** exist; the factory/scripts read `os.environ` directly.

- **`_MODEL_PATH` monkey-patch** at [safety_bigym/safety_bigym/envs/safety_env.py:111-116](../safety_bigym/envs/safety_env.py#L111-L116). Temporarily reassigns the class attribute `BiGymEnv._MODEL_PATH` to a merged world+human MJCF, calls `super().__init__`, then restores. Don't touch without understanding why — it's the only hook BiGym exposes for injecting a human before physics binding.

- **PD gain mismatches** across three sources:
  - `HumanConfig.kp/kd = 200/20` at [safety_bigym/safety_bigym/config.py:95-96](../safety_bigym/config.py#L95-L96)
  - `PDGains` dataclass defaults to `100/10` at [safety_bigym/safety_bigym/human/pd_controller.py:18-19](../safety_bigym/human/pd_controller.py#L18-L19)
  - MJCF actuator defaults: `200/20` in `smplh_human.xml`, `200/20` in `smplh_human_include.xml`, **`2000/50`** in `smplh_human_body.xml`
  - Verify which asset is actually loaded before tuning. The 10× discrepancy in `smplh_human_body.xml` is the most likely source of weird behaviour.

- **`info["safety"]` cost signals are computed but unused for training.** The wrapper emits `ssm_margin` and `pfl_force_ratio` every step (schema at [iso15066_wrapper.py:76-90](../safety_bigym/safety/iso15066_wrapper.py#L76-L90)), but they're only consumed by demo scripts, not any reward path. Wiring them into rewards is **Phase 3**'s job, not Phase 0 (Phase 0 only guarantees the signals are correct and logged).
  - **Caveat: `pfl_force_ratio`, `pfl_violation`, and `max_contact_force` are currently identically zero across every cell**, even when `ssm_margin` goes strongly negative (human pelvis geometrically inside the robot). Diagnosed as an unresolved contact-detection issue specific to BiGym/mojo's runtime robot attachment — `data.ncon=0` for every human↔robot pair despite passing all bit-eligibility, parent-filter, and weldid checks; same env produces robot-internal contacts and SMPL self-pairs fine. Minimal-scene reproduction (no BiGym) works correctly. Open issue tracked at [.claude/plans/pfl_contact_detection_open_bug.md](../../.claude/plans/pfl_contact_detection_open_bug.md). Diagnostic at [safety_bigym/scripts/diagnose_contact_forces.py](../scripts/diagnose_contact_forces.py) reproduces in <30 s.

- **Phase 2 SVF labels by geometric proximity, NOT ISO 15066 SSM** (as of 2026-05-16). [`label_transition`](../safety_bigym/filters/labeling.py) returns `r_safe = 0` when `info["safety"]["min_separation"] < proximity_threshold` (default 0.10 m; B3 production uses 0.50 m). ISO SSM's required-separation formula demands ~5m clearance at kitchen-scale robot velocities — initial B2.3 smoke at `ssm_violation`-based labels produced a 93% violation rate. `ssm_margin` is still computed by `ISO15066Wrapper` and logged in `info["safety"]` for ISO traceability and as a continuous cost signal for the Phase 3 Lagrangian; only the binary label is decoupled.

- **SVF shard schema records raw safety signals per-step** (as of 2026-05-16). [`TransitionShardWriter.write_shard`](../safety_bigym/filters/dataset.py) stores `min_separation` and `pfl_force_ratio` alongside the binary `r_safe`, so future relabelling does not require re-collection:
  - **Proximity-threshold sweeps are free** — recompute `r_safe = (min_separation >= τ)` in a notebook from the existing dataset.
  - **PFL retrofit is partially free.** The schema is ready, but `pfl_force_ratio` is identically zero under the current bug. Once PFL contact detection is fixed, a *new* collection through the fixed env will produce meaningful values; the existing v1 dataset cannot be retrofitted by relabelling alone. Flip `label_transition(..., use_pfl=True)` at training time to OR in `pfl_violation` from the per-step ratios (the ratio→bool threshold lives in [iso15066_wrapper.py](../safety_bigym/safety/iso15066_wrapper.py); `use_pfl=True` consumes `info["safety"]["pfl_violation"]`, not the raw ratio).
  - **Demo source synthesizes safe-side placeholders** (`min_separation=10.0`, `pfl_force_ratio=0.0`) — irrelevant in B3 because demo is dead, but the schema is consistent so a future 4-dof demo recording slots in without writer changes.

- **`svf_collect_dataset.py` env construction must match RoboBase training** (as of 2026-05-16). Three coupled details that broke snapshot rollouts during B2.3 debugging:
  1. `_build_live_env` constructs `JointPositionActionMode` with `floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]` — RoboBase's BiGym factory uses this 4-dof set under `cfg.env.enable_all_floating_dof=True` and Phase 0 ACT snapshots were trained under that regime (`action_dim=16`, `qpos=66`). Bare-BiGym defaults give 3 dofs (`action_dim=15`, `qpos=63`) and produce silent `state_dict` shape mismatches.
  2. Snapshot agents are instantiated against a **synthesized** observation_space (`_synthesize_snapshot_obs_space`) that mirrors RoboBase's `ConcatDim(shape_length=1, keys_to_ignore=['proprioception_floating_base_actions']) → FrameStack(frame_stack=1)` output. The raw `SafetyBiGymEnv` observation_space is *not* what the trained agent was sized against.
  3. Env wrapping is always `BodySLAMWrapper(plan.bodyslam_mode)` regardless of the snapshot's training-time `bodyslam.mode`. The SVF dataset must always carry `human_pos_estimate` (it's the critic's most load-bearing input feature); `_SnapshotPolicy.adapt_obs` strips the channel before feeding Phase 0 actors via `includes_human_pos=False`.

- **BiGym DemoStore demos load fine at 4-dof — the "3-dof / dead" claim was a red herring** (corrected 2026-05-20). The SaucepanToHob cache dir is `JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute` (4-dof), and `SafetyBiGymEnvFactory._get_demo_fn(cfg, num_demos=5)` returns 5 `Demo` objects with `info["demo_action"]` shape (16,). The Phase-2 B3 `DemoNotFoundError` came from a *different* env-construction path (hand-built action mode / safety-wrapped class name reaching `Metadata.from_env`), **not** from demos being 3-dof. Workstream D's `SafetyBiGymCQNAdapter.get_demos()` loads them through the raw-env + DemoStore path (`_get_demo_fn`). The earlier 2026-05-16 note (and the B3 "skips demo source" decision) overstated the problem.

- **Human root is a mocap body, not a freejoint** (as of 2026-05-07). Pelvis declared `mocap="true"` in [smplh_human_body.xml](../safety_bigym/assets/smplh_human_body.xml); the root pose is written to `data.mocap_pos` / `data.mocap_quat` each step rather than `data.qpos[0:7]`. Body joints (L_Hip, R_Hip, ...) remain physics-simulated under the kinematic mocap parent — PD on body joints is unchanged. Track Pelvis via `_human_pelvis_mocapid` (the freejoint-derived `_human_root_qpos_start` no longer exists). The previous qpos teleport anti-pattern flagged here is gone. Side effect: Pelvis `weldid=0` (welded to world), so under default `mjOPT_FILTERPARENT` Pelvis_col vs floor is filtered — irrelevant in practice because H1 with `floating_base=True` doesn't physically stand on the floor.

- **Cross-paired human↔robot collision channel** (as of 2026-05-07). Human `_col` geoms emit on bit 1 / accept bit 2 (`contype=2 conaffinity=4`); [`_configure_collision_bits`](../safety_bigym/envs/safety_env.py#L280) promotes robot/floor with `contype |= bit2 / conaffinity |= bit1`. Cross is non-zero in both directions for human↔robot, zero for human↔human (kills the previous SMPL self-collision bug that produced 220 kN spurious Torso/Chest forces). Constants: `_HUMAN_EMIT_BIT = 0b010`, `_ROBOT_EMIT_BIT = 0b100`. Test: `tests/test_collision_groups.py::test_human_bits_exact`.

- **Phase-0 closed these wiring bugs** (safety-critic/phase-0 branch):
  - `pfl_force_ratio` used to be declared on `SafetyInfo` but never populated; `_aggregate_safety_info` now delegates to `ISO15066Wrapper.build_safety_info()` which tracks `max contact.force_ratio` across the step.
  - SSM upgraded from pelvis-to-pelvis to **closest joint pair**: `compute_ssm` takes arrays of human body / robot geom positions and returns `d_min + (closest_human_joint, closest_robot_link)`. Human bodies enumerated in `SafetyBiGymEnv._HUMAN_SSM_BODY_NAMES` (18 SMPL joints). Env smoke observed `ssm_margin ≈ 0.4 m` with `closest_human_joint=L_Wrist` — previously stuck at ~1.0 m trunk-to-trunk.
  - Per-episode safety aggregation lives in [`EpisodeSafetyMetrics`](../safety_bigym/safety/episode_metrics_wrapper.py); it wraps every env produced by `SafetyBiGymEnvFactory` and emits `info["episode_safety"]` (fields prefixed `ep_*`) at `terminated/truncated`. RoboBase's `Workspace` forwards these to W&B automatically.
  - `cfgs/safety_config.yaml` now defaults `wandb.project=safety-critic` with a templated `wandb.name`; launch-specific overrides add phase tags.

- **`SSMConfig` is already deduplicated.** The plan text mentions a duplicate between `config.py` and `iso15066_wrapper.py`, but that was resolved in commit `fd213d8 fix tech debt`. The wrapper imports from `config.py`. The only stale copy is in `implementation_plan.md.resolved` (a doc, not code). Safe to ignore that Phase 0 bullet.

- **Phase 1 is closed; contingency triggered** (as of 2026-05-07). E1.1 ACT ablation produced no cell that clears the master plan's ≥20% SSM-rate-reduction bar (off→oracle: reach −2.7%, dishwasher_close −0.8%, drawers_open_all +13.7%, saucepan_to_hob −49.9%). Per [HYBRID_SAFETY_CRITIC_PLAN.md:51, 248](../../.claude/HYBRID_SAFETY_CRITIC_PLAN.md), the cost signal is the bottleneck — Phase 2/3 priority is bumped. E1.2 (noise sweep) and E1.3 (temporal ablation) are parked (no strong cell to sweep). DP coverage explicitly skipped; rationale documented. Result writeup at [safety_bigym/docs/phase1_observation_results.md](../docs/phase1_observation_results.md). Side-finding worth carrying into Phase 3: on saucepan_to_hob, oracle improves task success 0.22 → 0.58 without reducing SSM — the policy uses human state for task progress, not safety, when the reward landscape is hard. Argument for the continuous-cost Lagrangian formulation.

  - `0b57b7d` — per-task demo/episode overrides: `dishwasher_load_plates` bumps `episode_length` to 30000 (~2.1× demo length at downsample 25) and `demos: 34`; `dishwasher_close` bumps `demos: 50`. `cfgs/method` symlink made relative (`../../robobase/robobase/cfgs/method`) so it resolves on both local and GPU layouts. `train_safety.py` drops its post-init `load_snapshot` branch when `cfg.snapshot_path` is set because `Workspace.__init__` now does that work eagerly (see robobase drift bullet).
  - `393ded3` — `scripts/baseline_sweep.py` now emits `save_snapshot=true` on every `--train-missing` command (load-bearing: per-eval snapshots only land when this flag is true) and `demos=0` on every `--eval` command (load-bearing: snapshots carry `action_stats`/`obs_stats` in-payload, so demos don't need re-fetching).
  - Robobase workspace.py drift is **not in a git repo the user owns**. Patch at `/Users/ayushpatel/Documents/FYP3/phase0_workspace_drift.patch` must be `git apply`-ed on the GPU box's robobase clone before any retrain. Local robobase (`/Users/ayushpatel/Documents/FYP3/robobase/`) already has the edits applied but uncommitted.

- **Phase 3 scaffolding (P3.0) is landed** (merged PR #9, 2026-05-20). The Lagrangian glue itself (P3.1: λ PID, dual-Q `argmax_a [Q_r − λ·Q_c]`, Q_c training-loop integration) is NOT done. What exists:
  - **Workspace reward shaping** — `SafetyConfig.add_workspace_penalty` / `workspace_radius` (0.4) / `workspace_beta` (**0.05** as of 2026-05-20) / `workspace_excess_cap` (**1.0**, bounds the penalty), applied in `SafetyBiGymEnv._reward()` via `_compute_workspace_penalty()`; threaded through `safety_bigym_factory._create_env`. Off by default.

- **Shaped/dense rewards must fit the C51 critic's value support** (as of 2026-05-20). The CQN-AS critic clamps its Bellman target to `[v_min, v_max]` (`agent.py`, `cqn_as.yaml` defaults `[-2, +2]`). A dense reward whose *discounted* return `|r|/(1−γ)` exceeds the support half-range silently **saturates value learning** — the agent trains with no error but produces a degenerate policy (this caused the 2026-05-20 base-validation failure: the unbounded `β=0.2` workspace penalty gave returns ~−20→−100, all clamped to −2, killing the pull-back gradient; the robot parked away from the task, reward −78→−775). Fixes: bound the shaping term (`workspace_excess_cap`) and/or widen support, keeping the invariant **`β·workspace_excess_cap/(1−γ) ≤ |v_min|`**. The bounded curriculum re-validation runs via `scripts/run_base_curriculum.sh`; full writeup [docs/phase3_base_validation_findings.md](phase3_base_validation_findings.md).
  - **`CostCritic`** ([filters/cost_critic.py](../safety_bigym/filters/cost_critic.py)) — Phase 3 Q_c, architectural twin of the Phase 2 `SafetyCritic`. `warm_start_from_svf()` **refuses without `force_sign_flip=True`**: SVF regresses `r_safe` (high Q = safe), Q_c regresses `c_t` (high Q = dangerous), so the heads point opposite ways. Warm-start-vs-fresh A/B deferred to P3.1.
  - **Per-step cost `c_t`** ([filters/cost_signal.py](../safety_bigym/filters/cost_signal.py), `c_t = min(1, max(c_ssm, c_pfl))`) flows end-to-end through the CQN-AS pipeline: env_adapter → TimeStep → episode shard (`cost` data_spec) → batch dict (`cost` + `max_cost`). **`agent.update()` does not consume it yet** — P3.1's job. `c_pfl` is identically zero under the open PFL bug, so `c_t == c_ssm` in practice today.
  - Smoke: `python scripts/phase3_p30_smoke.py` (~10s; `+phase3_p30_smoke.dry_run=true` skips MuJoCo).

- **H1 has no `get_ee_position()`** (it's `get_hand_pos(HandSide)`, which returns `[0,0,0]` at rest). To read the robot end-effector, use `_get_robot_state()` and take `state["ee_pos"]` if present else `state["link_pos"]["ee"]` — the `link_pos["ee"]` entry is populated via `_ROBOT_LINK_NAMES` mj_name2id lookups and is the load-bearing path for H1. Calling `self._robot.get_ee_position()` directly silently raises `AttributeError` (it bit the first cut of `_compute_workspace_penalty`, making the penalty a no-op on every task).

- **`train_cqn_as.py` snapshot/video cadence** (fixed `6e7fdc1` / `09faaa4`). Snapshots save every `snapshot_every_frames` (default 10000) **plus a final-state save** when the train loop exits — the decision lives in the main step loop, NOT inside the episode-end block (the old placement only fired when an episode boundary coincidentally aligned with a cadence multiple, so 200k-step runs saved zero snapshots). Eval videos: set `save_video=true` → `eval_videos/step_<step>_ep0.mp4` per eval cycle + W&B `eval/video` (first eval episode only). Helpers in [agents/cqn_as/eval_video.py](../safety_bigym/agents/cqn_as/eval_video.py). `phase1_reward_pilot_cqn_as.py --train` emits both flags automatically.

## How to run things

- **Tests:** `cd safety_bigym && pytest tests/` (discovers from `tests/` directly; no config)
- **Training:** `cd safety_bigym && python train_safety.py launch=dp_pixel_safety_bigym env=safety_bigym/reach_target_single` (Hydra). Configs live under [safety_bigym/cfgs/](../cfgs/).
- **W&B:** wired via Hydra configs ([safety_bigym/cfgs/safety_config.yaml](../cfgs/safety_config.yaml), [safety_bigym/cfgs/launch/dp_pixel_safety_bigym.yaml](../cfgs/launch/dp_pixel_safety_bigym.yaml)); logging is handled by RoboBase's `Workspace`. Enable with `wandb.use=true wandb.name=<run>`.
- **Virtual Environment:** Use safety_bigym/venv/

## Conventions for new code

- Safety filters → `safety_bigym/safety_bigym/filters/` (new subpackage per the main plan; doesn't exist yet)
- New wrappers → `safety_bigym/safety_bigym/safety/` alongside `iso15066_wrapper.py`, or a dedicated `wrappers/` subpackage if they're task-agnostic
- Training/experiment scripts → `safety_bigym/scripts/` (existing convention; 24 scripts already there)
- Tests → `safety_bigym/tests/` mirroring the package layout
- Generability - ensure the changes you make will still allow it to work when running on this device or the GPU device

## Branch strategy

- Base branch: `main`, not `new-human-motion` (which is an experimental line).
- One branch per phase: `safety-critic/phase-0-prep`, `safety-critic/phase-1-bodyslam-wrapper`, `safety-critic/phase-2-svf-filter`, `safety-critic/phase-3-constrained-rl`, `safety-critic/phase-4-hybrid`, `safety-critic/phase-5-eval`.
- Sub-task branches fork off the phase branch when a phase has distinct deliverables (e.g., `safety-critic/phase-2-dataset` → `...-critic-training` → `...-runtime-wrapper`).
- Never commit to `main` directly. Use PRs from phase branches for review checkpointing.

## Session-handling rules (for working on this plan with Claude Code)

- One session per phase; ideally one per sub-task. `/clear` between unrelated tasks.
- On any task that touches >1 file, ask for a file-level plan before making changes.
- Write tests alongside or before new modules (`BodySLAMWrapper`, safety critic, Lagrangian wrapper).
- Commit after each completed task, not each session. Git commits are the precise rollback points.
- **Do not launch multi-hour training jobs from within Claude Code.** Write the script, do a ≤100-step smoke run, then hand off to the human to run the real experiment.
- **RoboBase has in-place drift (Phase-0 human-fix branch).** Two files diverge from upstream `411b7c7`, all additive changes tagged `FYP3/safety_bigym drift`. Source of truth for the GPU box is `/Users/ayushpatel/Documents/FYP3/phase0_workspace_drift.patch`.
  - [robobase/robobase/workspace.py](robobase/robobase/workspace.py):
    1. `_pretrain_on_demos` now calls `self.save_snapshot()` at every pretrain-eval interval (not just at end of training), so the best-by-curve checkpoint is recoverable off disk.
    2. Snapshots are self-contained: `save_snapshot` writes `env_factory._action_stats` / `_obs_stats` into the payload; `Workspace.__init__` reads `cfg.snapshot_path` eagerly and seeds those dicts onto the factory *before* env construction (wrappers capture them by reference at build time — post-init restore can't work). Eval-only runs can now set `cfg.demos=0` without losing action denormalisation.
    3. Both `torch.load` call sites pass `weights_only=False` because snapshots contain the Hydra `DictConfig`, which isn't in PyTorch 2.6's default safe globals. (Local <2.6 works either way; GPU box is ≥2.6.)
    4. Diffusion-policy EMA shadow is persisted/restored explicitly (`payload["actor_ema"]`). The `diffusers.EMAModel` at `Actor.ema` is not an `nn.Module`, so its `shadow_params` — the actual eval-time weights (`Actor.infer` does `self.ema.copy_to(self.ema_actor.parameters())` on every forward) — are invisible to `agent.state_dict()`. Without this fix, reloaded snapshots run eval with a freshly-initialized (untrained) EMA and produce random actions, even though `load_state_dict` is `strict=True` and doesn't throw. Legacy snapshots (saved before this fix) are salvageable: `load_snapshot` falls back to seeding `ema.shadow_params` from the loaded `ema_actor.parameters()`, which are in the `agent` state_dict and contain the correct EMA weights because FYP3 drift (point 1) saves snapshots right after `_eval()` calls `infer()` → `copy_to(ema_actor)`.
  - [robobase/robobase/envs/bigym.py](robobase/robobase/envs/bigym.py) `_wrap_env`: the upstream `assert cfg.demos != 0` is relaxed to `assert cfg.demos != 0 or self._action_stats is not None` — stats-in-snapshot makes `demos=0` a valid eval config.
  Beyond these two files, don't edit RoboBase in-place. If Phase 3 Option B becomes necessary, fork RoboBase into a sibling directory.
- Start every phase-related session by reading [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md), [CHANGES_AND_NEXT_STEPS.md](CHANGES_AND_NEXT_STEPS.md), [UPDATED_PROJECT_PLAN.md](UPDATED_PROJECT_PLAN.md) and this file — all in `safety_bigym/docs/`.
- Hand off to the human when visual verification is required.

## Where the project docs live (as of 2026-05-20)

**Canonical home is `safety_bigym/docs/` — version-controlled in the safety_bigym repo. Read and edit these:**
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) — living status + the "Next session — start here" block (read this first)
- [CHANGES_AND_NEXT_STEPS.md](CHANGES_AND_NEXT_STEPS.md) — change inventory
- [UPDATED_PROJECT_PLAN.md](UPDATED_PROJECT_PLAN.md) — the Hybrid Safety Critic plan
- `docs/CLAUDE.md` — this file

**CLAUDE.md special case:** Claude Code auto-loads the FYP3-root `/CLAUDE.md` (the working dir), so that copy must exist and stay current. Treat `safety_bigym/docs/CLAUDE.md` as canonical; after editing it, mirror the prose to the root `/CLAUDE.md` (the root copy uses root-relative links like `safety_bigym/...`; the docs copy uses `../...` — only the link hrefs differ, prose is identical).

**Superseded:** the old `.claude/IMPLEMENTATION_STATUS.md`, `.claude/CHANGES_AND_NEXT_STEPS.md`, and `.claude/UPDATED_PROJECT_PLAN.md` are now stale mirrors (the FYP3 root is outside any git repo). The `docs/` copies are source of truth; the IDE may still open the `.claude/` ones, so don't edit those — edit `docs/`.

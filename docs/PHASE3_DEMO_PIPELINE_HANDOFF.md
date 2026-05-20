# Handoff — CQN-AS demo pipeline (Workstream D)

Created 2026-05-20. Single-session task. Read this top-to-bottom before touching code.

Companion docs (all in `safety_bigym/docs/`): [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) (status + decision log), [cqn_as_integration_notes.md](cqn_as_integration_notes.md) (CQN-AS vendor gotchas), [CLAUDE.md](CLAUDE.md) (workspace orientation), [UPDATED_PROJECT_PLAN.md](UPDATED_PROJECT_PLAN.md) (Phase 3 design).

---

## Why this task exists

The E1.4 CQN-AS observation ablation (C2) ran with `num_demos=0` and produced a **degenerate policy**: 31-step episodes, the robot fleeing the human, and off/oracle/noisy reward curves collapsing onto each other. Root cause is two stacked confounds:

1. **No demos.** CQN-AS is a demo-driven, sample-efficient RL algorithm — benchmarked on BiGym *with* demos. Run demo-less from scratch on a long-horizon manipulation task (`saucepan_to_hob`), the sparse task-success reward is never discovered, so the only gradient is the violation penalty → the policy learns the one easy thing (evacuate the workspace) and never attempts the task.
2. **No workspace shaping** in that run (`add_workspace_penalty=false`), so evacuation was the cheap optimum.

This task fixes confound #1: **make `num_demos>0` work for CQN-AS through `SafetyBiGymEnvFactory`.** Demos materially help both the eventual Phase 3 actor and any re-run of E1.4.

## Goal (acceptance criteria)

1. `python train_cqn_as.py env=safety_bigym/saucepan_to_hob disruption=coworker_train bodyslam=oracle num_demos=10 num_train_frames=2000 wandb.use=false` loads ≥1 demo, fills the demo replay buffer, and trains 2000 frames **without crashing** and **without the worker-striping IndexError** (the `num_demos=0` cold-start race goes away once demos pre-fill every worker — see integration notes §4).
2. The demo `low_dim_obs` carries `human_pos_estimate` (6D) when `bodyslam!=off`, synthesised from an AMASS clip (demos were recorded without a human). Confirm the demo low_dim width matches the live-env low_dim width for the same `bodyslam` mode (off vs oracle/noisy differ by 6×frame_stack).
3. A short trained run (~30–50k frames) with demos + `bodyslam=oracle` + `env.safety.add_workspace_penalty=true` shows the policy **attempting the task** (non-trivial episode lengths, task reward > 0 on some episodes) rather than the C2 evacuation collapse. This is the real "demos unblock learning" signal.
4. A pytest covering the demo conversion (pure-Python, monkeypatched DemoStore — no GPU/MuJoCo), mirroring the existing `tests/test_cqn_as_adapter.py` stub style.

## Current state — what exists vs what's missing

**The stub that must be implemented:** `SafetyBiGymCQNAdapter.get_demos()` in [../safety_bigym/agents/cqn_as/env_adapter.py](../safety_bigym/agents/cqn_as/env_adapter.py) currently raises `NotImplementedError`. `train_cqn_as.py` calls `self.train_env.get_demos(self.cfg.num_demos)` only when `num_demos>0` (see [../train_cqn_as.py](../train_cqn_as.py) `_setup_replay` + the demo-fill loop ~line 260-271), so wiring `get_demos` is the whole job on the training side.

**Building blocks that ALREADY exist (reuse, don't rebuild):**
- `SafetyBiGymEnvFactory._create_raw_bigym_env(cfg)` and `._get_demo_fn(cfg, num_demos)` in [../safety_bigym/envs/safety_bigym_factory.py](../safety_bigym/envs/safety_bigym_factory.py) — load demos via the **raw** (non-safety-wrapped) BiGym env through `DemoStore` + `Metadata.from_env`. The raw env is required because `DemoStore` matches demos by env *class name*, and the safety-wrapped class is `SafetyReachTargetSingle` not `ReachTargetSingle`.
- The factory's `_wrap_env` override inserts `BodySLAMWrapper` in **`demo_replay` mode** (synthesising `human_pos_estimate` from an `AMASSDemoPositionProvider` clip, since demos have no live human). Study how this is wired — the CQN-AS demo path needs the same human-pos injection.
- `BodySLAMWrapper` + `AMASSDemoPositionProvider` in `safety_bigym/perception/`.

**The CQN-AS reference implementation to port from** (upstream, pristine — do not modify): [../../CQN-AS/bigym_src/bigym_env.py](../../CQN-AS/bigym_src/bigym_env.py):
- `get_demos(num_demos)` (line ~229) — DemoStore fetch → filter post-reward states → `convert_demo_to_timesteps` → `extract_action_stats` → `rescale_demo_actions`.
- `convert_demo_to_timesteps(demo)` (line ~321) — turns a raw BiGym demo into a list of `ExtendedTimeStep`, reading the action from `demostep.info["demo_action"]`, building obs via `self._extract_obs(...)`, assigning FIRST/MID/LAST step types and discount.
- `extract_action_stats(demos)` (line ~279) — per-dim min/max with the gripper tail hard-coded to [0,1]. **This overrides `self._action_stats`** — important: the adapter's `_convert_action_to_raw` rescaling must use the demo-derived stats so demo actions and live actions share one normalisation.

## THE crux — resolve this FIRST (it's cheap and decides everything)

The decision log and CLAUDE.md gotchas claim *"BiGym DemoStore demos are 3-dof and unusable; 4-dof env → DemoNotFoundError."* **This claim is very likely a red herring** and you must verify it before assuming any re-recording is needed. Evidence it's wrong:

- `cfgs/env/safety_bigym/saucepan_to_hob.yaml` sets `enable_all_floating_dof: true` (**4-dof**: X, Y, Z, RZ) and `demos: 36`.
- Phase 0 ACT was trained on this exact config (4-dof) and **successfully loaded 36 demos** — the ACT snapshots exist and were used as SVF dataset sources (B1).
- RoboBase's `BiGymEnvFactory._get_demo_fn` ([../../robobase/robobase/envs/bigym.py](../../robobase/robobase/envs/bigym.py)) loads demos at 4-dof via `Metadata.from_env(env)` where the env was built with `floating_dofs=[X,Y,Z,RZ]`.
- Cached demos are present on this machine: `~/.bigym/demonstrations/0.9.0/SaucepanToHob`.

The `DemoNotFoundError` in `svf_collect_dataset.py` (B3) most plausibly came from that path constructing the env differently (hand-built `JointPositionActionMode`, or a safety-wrapped class name reaching `Metadata.from_env`) — **not** from the demos being 3-dof. So:

**Step 0 (do this first):** confirm demos load at 4-dof through the raw-env path. Quickest check:
```python
# from safety_bigym/, venv python
from omegaconf import OmegaConf
import hydra
with hydra.initialize(config_path="cfgs", version_base=None):
    cfg = hydra.compose(config_name="cqn_as_config",
                        overrides=["env=safety_bigym/saucepan_to_hob",
                                   "disruption=coworker_train", "bodyslam=oracle"])
from safety_bigym.envs.safety_bigym_factory import SafetyBiGymEnvFactory
f = SafetyBiGymEnvFactory()
demos = f._get_demo_fn(cfg, num_demos=5)   # raw-env DemoStore path
print(len(demos), type(demos[0]))
```
- **If this returns demos** → no re-recording needed. Build the CQN-AS port that reuses this path. (Expected outcome.)
- **If it raises `DemoNotFoundError`** → then the cache genuinely lacks 4-dof demos for this task, and re-recording is required (see "Fallback" below). Confirm the failure is dof/metadata, not class-name, before concluding.

## The implementation (assuming Step 0 returns demos)

Implement `SafetyBiGymCQNAdapter.get_demos(num_demos)` to:
1. Load raw demos via the factory's raw-env + `DemoStore` path (reuse `_get_demo_fn` logic, or call `DemoStore` directly with `Metadata.from_env(raw_env)`). Demos arrive in RoboBase/BiGym demo format (a list of demos, each a list of demosteps with `.observation`, `.reward`, `.info["demo_action"]`, `.termination`, `.truncation`).
2. Inject `human_pos_estimate` into each demostep's observation when `bodyslam!=off`, using the same `AMASSDemoPositionProvider`/`BodySLAMWrapper(demo_replay)` mechanism the factory's `_wrap_env` override already uses for the RoboBase demo path. The demo obs must end up with the *same keys and widths* the live `_extract_obs` produces, so `convert_demo_to_timesteps` → `_extract_obs` yields a `low_dim_obs` that matches `low_dim_raw_observation_spec()`.
3. Port `convert_demo_to_timesteps`, `extract_action_stats`, `rescale_demo_actions` from upstream (adapt imports to the vendored modules; the adapter already has `_extract_obs`, frame-stack deques, `_action_stats`).
4. Override `self._action_stats` with the demo-derived stats (so live + demo actions share normalisation — this is also the fix for the B4.2 "snapshot action denormalization" caveat noted in IMPLEMENTATION_STATUS).
5. Return the list-of-(list-of-`ExtendedTimeStep`). `train_cqn_as.py` adds them to `demo_replay_storage` via `ReplayBufferStorage.add(time_step)` per step.

**Watch the cost field (P3.0c).** The replay storage `data_specs` now includes a `cost` entry ([../train_cqn_as.py](../train_cqn_as.py) `_setup_replay`). `ReplayBufferStorage.add()` reads `time_step["cost"]`. So the demo `ExtendedTimeStep` must carry a `cost` field (the NamedTuple already has it, default 0.0). Synthesise demo cost from the demo's `info["safety"]` if present, else 0.0 — demos have no live human so safe-side `cost=0.0` is the sensible placeholder (mirrors how the SVF demo source synthesised `min_separation=10.0`). Confirm the demo `ExtendedTimeStep` shape matches what `ReplayBufferStorage.add` expects against the cost-extended `data_specs`.

## Fallback (only if Step 0 genuinely fails)

Re-record 4-dof demos through `mojo.demonstrations` or script the SNAPSHOT actors as experts. This is a much bigger lift; do NOT start here. Tracked as the canonical fix in the decision log (2026-05-16) but only if the cache truly lacks 4-dof demos. Re-confirm the failure mode first — a class-name mismatch (safety-wrapped env reaching `Metadata.from_env`) is fixable without re-recording.

## Gotchas to carry (from cqn_as_integration_notes.md)

- **Worker striping:** the demo fill is what makes `num_demos>0` avoid the `num_demos=0` cold-start IndexError (§4). With demos, every replay worker is pre-filled. Good.
- **Custom collate (§5):** `_copying_collate` already handles the demo loader; if you add fields to the demo `ExtendedTimeStep` tuple, keep them numpy-backed and contiguous.
- **`tensordict==0.6.0`, python-3.12 `random.seed(int(...))`, `logging.basicConfig(force=True)`** — already handled, don't regress.
- **4-dof everywhere:** `action_dim=16`, `qpos=66`. A 3-dof env silently shape-mismatches.
- **Snapshot/eval-video cadence** is already fixed (`6e7fdc1`/`09faaa4`); demos don't interact with it.

## Definition of done

- [ ] Step 0 resolved (demos load at 4-dof, or re-record decision made + justified)
- [ ] `get_demos` implemented; `num_demos=10 num_train_frames=2000` smoke runs clean
- [ ] demo `low_dim_obs` width matches live width for off/oracle/noisy
- [ ] action_stats demo-derived and shared with live path
- [ ] demo `ExtendedTimeStep` carries the `cost` field; `ReplayBufferStorage.add` accepts it
- [ ] pytest for the conversion (stubbed DemoStore, no MuJoCo)
- [ ] ~30–50k-frame trained smoke with demos + workspace shaping shows task attempts, not evacuation
- [ ] Update IMPLEMENTATION_STATUS Workstream D + decision log with the Step-0 finding

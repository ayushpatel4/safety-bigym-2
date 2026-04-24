# How Diffusion Policies Are Trained in safety_bigym

## 1. Entry point & config composition

[train_safety.py](../train_safety.py) is a thin Hydra wrapper: it instantiates RoboBase's `Workspace` and calls `workspace.train()`. The config stack composes bottom-up:

- **Base**: [cfgs/safety_config.yaml](../cfgs/safety_config.yaml) — global defaults (`batch_size: 256`, `num_train_frames: 1.1M`, `save_snapshot: false`, `demos: 0`).
- **Launch**: [cfgs/launch/dp_pixel_safety_bigym.yaml](../cfgs/launch/dp_pixel_safety_bigym.yaml) — `@package _global_` overlay that selects **`method: diffusion`** and sets `demos: 30`, `num_pretrain_steps: 100000`, **`num_train_frames: 0`** (pure imitation, no online RL), `action_sequence: 16`, `temporal_ensemble: true`, `pixels: true`, `use_min_max_normalization: true`.
- **Method**: `robobase/cfgs/method/diffusion.yaml` — `_target_: robobase.method.diffusion.Diffusion`, `is_rl: false`, `ConditionalUnet1D` (dims 256/512/1024, 50 DDIM steps, `squaredcos_cap_v2` beta schedule), ResNet18 pixel encoder, multi-cam flatten fusion.
- **Env**: [cfgs/env/safety_bigym/](../cfgs/env/safety_bigym) per-task overrides (e.g., `saucepan_to_hob: episode_length: 15000`; `dishwasher_close: demos: 50`).

## 2. Workspace init (`robobase/robobase/workspace.py`)

Order matters (FYP3 drift lives here):

1. **Snapshot pre-seed** (L157–174): if `cfg.snapshot_path` is set, load payload's `action_stats`/`obs_stats` onto `env_factory` *before* env creation. Wrappers capture those dicts by reference, so post-init restore can't work.
2. `env_factory.collect_or_fetch_demos(cfg, num_demos)` → computes stats from demos.
3. `make_train_env` / `make_eval_env`.
4. `post_collect_or_fetch_demos` → rescale demo actions into Tanh space, slice into replay steps.
5. Hydra `instantiate(cfg.method, ...)` builds the `Diffusion` agent.
6. Replay buffer(s) built; demos loaded via `_load_demos`.

## 3. Env factory & wrappers ([safety_bigym_factory.py](../safety_bigym/envs/safety_bigym_factory.py))

- Demos fetched through a **raw** BiGym env (L66–94) because `DemoStore` keys on class name; `SafetyBiGymEnv` would miss.
- Downsample rate: `100Hz / demo_down_sample_rate` (typically 5Hz control).
- Cameras: head + both wrists @ 84×84 RGB.
- Action mode: `JointPositionActionMode`, absolute, with floating base (x, y, z, rz).
- Human/AMASS config read from env yaml; `terminate_on_violation=False` (safety monitored, not enforced).
- Final env wrapped with [`EpisodeSafetyMetrics`](../safety_bigym/safety/episode_metrics_wrapper.py) (L197).
- Standard RoboBase wrappers applied on top: `RescaleFromTanhWithMinMax`, `ConcatDim` (low-dim normalization), `FrameStack`, `TimeLimit` (= `episode_length // demo_down_sample_rate`), `ActionSequence` (chunks of 16).

## 4. Training loop

Because `num_train_frames=0`, online RL is skipped. The loop is:

- `_load_demos()` → fills replay with (demo-flagged) transitions.
- `_pretrain_on_demos()` for 100k steps, each step: sample batch, call `agent.update()`.

**Diffusion loss** (`robobase/method/diffusion.py`): sample noise ε ~ N(0,I) and timestep t, produce noisy action chunk via DDIM forward, condition UNet on (obs features, t), MSE against true noise. EMA of actor (power 0.75) used at inference (50 DDIM reverse steps). Action chunk shape `(16, action_dim)`, applied with temporal ensembling (gain 0.01).

## 5. Eval & snapshots

- Every `eval_every_steps` (default 10k): run eval episodes → `pretrain_eval/episode_success` logged to W&B.
- **FYP3 drift #1**: when `save_snapshot=true`, snapshot is written *at every pretrain-eval interval* (not just at end), so you can pick the peak-by-curve weights off disk.
- **FYP3 drift #2**: snapshot payload carries `action_stats` + `obs_stats`, so `--eval` with `demos=0` and `+snapshot_path=...` reconstructs the full action/obs scaling without refetching demos.

[scripts/baseline_sweep.py](../scripts/baseline_sweep.py) generates the commands: `--train-missing` emits per-task training runs (with `save_snapshot=true`), `--eval` emits 3×5 (task × ISO 15066 disruption) evaluation runs.

## 6. Safety signals — not in the loss

`SafetyBiGymEnv.step` fills `info["safety"]` (`ssm_margin`, `pfl_force_ratio`, violations by region), aggregated per episode by `EpisodeSafetyMetrics` into `info["episode_safety"]` (`ep_*` fields). RoboBase's workspace forwards those to W&B automatically. **Nothing in the DP loss consumes them** — pure imitation today. Wiring into rewards is Phase 3's job per `.claude/HYBRID_SAFETY_CRITIC_PLAN.md`.

## TL;DR

`train_safety.py` → Hydra composes DP-pixel launch + diffusion method + task env → RoboBase `Workspace` pre-seeds stats from snapshot (if any), fetches BiGym demos via a raw env, normalizes actions/obs, builds a ConditionalUnet1D + ResNet18 diffusion policy → 100k-step pretrain on demos (no online RL), periodic eval writes snapshots with embedded stats → `baseline_sweep.py` orchestrates train/eval across tasks and ISO 15066 disruption types. Safety metrics are logged, never back-propagated.

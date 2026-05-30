# CQN-AS vendor — integration notes

Key findings from getting upstream CQN-AS (commit `8cf806e`) running inside
`safety_bigym` on python 3.12 + torch 2.11. Read this if you're touching
[`safety_bigym/agents/cqn_as/`](../safety_bigym/agents/cqn_as/),
[`train_cqn_as.py`](../train_cqn_as.py), or adding a new agent that uses
the CQN-AS replay buffer.

Upstream was developed against the env in
[`CQN-AS/conda_env.yml`](../../CQN-AS/conda_env.yml) — python 3.10, torch
unpinned, tensordict 0.6.0. The safety_bigym venv runs python 3.12 + torch
2.11; most of the issues below come from that version gap or from running
upstream code with `num_demos=0` (which upstream never did).

## TL;DR

1. **`tensordict==0.6.0` must be in `setup.py`** — vendored agent imports it.
2. **Python 3.12 broke `random.seed(numpy.uint32)`** — cast to `int(seed)`.
3. **`agent.update()` returns a TensorDict, not a dict** — never bool-test it; iterate `.items()` exactly once.
4. **Replay loader stripes episodes by `eps_idx % num_workers`** — with `num_demos=0`, gate updates on `global_episode >= num_workers`.
5. **`torch.as_tensor(numpy)` + DataLoader workers → "non-resizable storage" crash** — use a custom collate that copies.
6. **Hydra 1.1+ doesn't `chdir` by default** — set `hydra.job.chdir=true` or `buffer/` pollutes across runs.
7. **`logging.basicConfig` is a no-op if root already has a handler** — always pass `force=True`.
8. **`CQNASAgent` is not an `nn.Module`** — it had no `state_dict`/`load_state_dict`; we added them.
9. **Running a CQN-AS snapshot as a *policy* inside the SVF collector** (Phase 2 G1 re-eval) needs a dedicated loader (`agent_state`/`config` payload, split `act(rgb,low_dim)` args) + a vendored inference-path `.view`→`.reshape` fix for torch≥2 CPU.

---

## 1. Hidden dependencies (`tensordict`)

`safety_bigym/agents/cqn_as/{agent,utils}.py` `from tensordict import TensorDict`. Upstream's conda env pins `tensordict==0.6.0`. **It must be in our `setup.py` `install_requires`**; otherwise the Hydra `_target_: safety_bigym.agents.cqn_as.agent.CQNASAgent` instantiation fails with the misleading "Error locating target" message (Hydra wraps any import error in that wrapper).

If you vendor more upstream modules, grep their imports against the conda env file and add to `setup.py`. Already pulled in: `tensordict`, `dm_env`.

## 2. Python 3.12 broke `random.seed(numpy_int)`

`safety_bigym/agents/cqn_as/replay_buffer.py:_worker_init_fn` did:
```python
seed = np.random.get_state()[1][0] + worker_id   # numpy.uint32
np.random.seed(seed)
random.seed(seed)  # ❌ Python 3.12 rejects numpy scalars
```

Python 3.12 tightened `random.seed`: it now strictly requires `None | int | float | str | bytes | bytearray` and no longer falls back to `__index__()` on numpy ints. Cast explicitly:
```python
random.seed(int(seed))
```

Generic lesson: any vendored code that worked on 3.10 may silently bite on 3.12 around type coercion in stdlib.

## 3. `agent.update()` returns a `TensorDict`

CQN-AS's `update_critic` returns a `tensordict.TensorDict`, not a dict. Two practical consequences:

```python
metrics = self.agent.update(batch)

if metrics:                          # ❌ RuntimeError: Converting a tensordict to boolean value is not permitted
    ...
if metrics is None or len(metrics) == 0:   # ✅ length check
    ...

# Iterating .items() — TensorDict 0.6.0 returns a single-use generator
prefixed = {f"{ty}/{k}": v for k, v in metrics.items()}
log_str = " ".join(f"{k}={v}" for k, v in metrics.items())  # ❌ second iter yields nothing
# → materialise items into a list once
items = [(k, v) for k, v in metrics.items()]
```

Also: format strings like `f"{v:.4f}"` fail on tensor values. Either branch on `isinstance(v, float)` or call `.item()` on 0-d tensors first (the current `_log` does this).

This bug was particularly silent — the run completed without crashing but the `[train] step=N q_critic_loss=...` lines never appeared because the format-string join saw an empty iterator. The dict comprehension also produced an empty dict, so W&B logging dropped too.

## 4. Worker-aware update gate (with `num_demos=0`)

The vendored `ReplayBuffer._try_fetch` filters by:
```python
if eps_idx % self._num_workers != worker_id:
    continue
```
So worker N only loads episodes whose `eps_idx % num_workers == N`. Upstream never hits the cold-start race because `num_demos > 0` pre-fills demo episodes that satisfy every worker's filter at construction time.

When we run the A6 smoke gate with `num_demos=0`, the first `next(replay_iter)` call after episode 1 ends will route to *some* worker (DataLoader rotates); if it picks worker `eps_idx=1 mod 2 = 1` but only `eps_idx=0` is on disk, it raises `IndexError: Cannot choose from an empty sequence` inside `_sample_episode`.

Fix in `train_cqn_as.py` train loop: gate updates on the **episode** count, not just transition count:
```python
num_replay_workers = max(1, int(self.cfg.replay_buffer_num_workers))
if (
    not seed_until_step(self.global_step)
    and self.global_step % self.cfg.agent.update_every_steps == 0
    and self._global_episode >= num_replay_workers
):
    ...
```

Equivalent alternatives if you ever need them:
- Set `replay_buffer_num_workers=0` (no worker striping). Slower but simpler.
- Pre-fill the buffer with random episodes before the first update.

## 5. DataLoader collate vs numpy-backed tensors

`torch.utils.data._utils.collate.collate_tensor_fn` takes a worker-process fast path:
```python
storage = elem._typed_storage()._new_shared(numel)
out = elem.new(storage).resize_(len(batch), *list(elem.size()))   # ❌
```
When `elem` was created via `torch.as_tensor(numpy_array)` (the default for numpy-typed batch items), `elem.new(storage)` inherits non-resizable attributes from the numpy buffer and `.resize_` raises:
```
RuntimeError: Trying to resize storage that is not resizable
```
This is deterministic on torch ≥2.5 with `num_workers > 0`. Upstream may not hit it on the older torch that came with their conda env.

Fix in `safety_bigym/agents/cqn_as/replay_buffer.py:make_replay_loader`:
```python
def _copying_collate(batch):
    n_fields = len(batch[0])
    return tuple(
        torch.stack([
            torch.from_numpy(np.ascontiguousarray(sample[i])).clone()
            for sample in batch
        ])
        for i in range(n_fields)
    )

loader = DataLoader(..., collate_fn=_copying_collate)
```

`.clone()` materialises data in torch-native (resizable) storage; `np.ascontiguousarray` deals with non-contiguous strides from the npz mmap. Cost is one extra copy per batch — for our batch sizes (256–512) on a GPU, it's not in the bottleneck.

## 6. Hydra `job.chdir` is off by default in 1.1+

[`cfgs/cqn_as_config.yaml`](../cfgs/cqn_as_config.yaml) sets:
```yaml
hydra:
  job:
    chdir: true
```
Without this, `Path.cwd()` returns the launch dir (the repo root), so `Workspace.work_dir / "buffer"` resolves to `<repo>/buffer/` — **shared across every train_cqn_as.py invocation**. `ReplayBufferStorage._preload` scans that dir at construction time and picks up episode files from prior runs (including ones from crashed processes with different shapes).

This isn't a CQN-AS bug per se, but it interacts badly with the collate bug above (more pollution = more chances of hitting the wrong file).

If you ever clone the script for a new agent, either set `chdir: true` or pass an explicit per-run `replay_dir` to the storage constructor.

## 7. `logging.basicConfig` silent no-op

`scripts/svf_train_critic.py` originally did:
```python
logging.basicConfig(level=logging.INFO, format=...)
```
If anything in the import chain has already configured the root logger (e.g. some torch or wandb internal), basicConfig is a no-op — and if the existing handler's level is `WARNING`, every `logger.info` is silently dropped. Net effect: the smoke script completed successfully but produced **zero stdout/stderr**, looking exactly like a process that never ran.

Fix: pass `force=True`:
```python
logging.basicConfig(level=logging.INFO, format=..., force=True)
```
This rebinds root handlers/level unconditionally. Cheap insurance for any standalone script.

## 8. `CQNASAgent` is not an `nn.Module`

Upstream's `CQNASAgent` owns several `nn.Module` children (`encoder`, `critic`, `critic_target`) and optimizers, but is itself a plain Python class. It has no built-in `state_dict()`/`load_state_dict()`.

Our previous `Workspace.save_snapshot` did:
```python
"agent_state": self.agent.state_dict() if hasattr(self.agent, "state_dict") else None
```
…which silently saved `agent_state=None`. Reloading would crash on `self.agent.load_state_dict(None)`.

Fix in [`safety_bigym/agents/cqn_as/agent.py`](../safety_bigym/agents/cqn_as/agent.py): add explicit save/load methods that round-trip every owned sub-module:
```python
def state_dict(self):
    return {
        "encoder": self.encoder.state_dict(),
        "critic": self.critic.state_dict(),
        "critic_target": self.critic_target.state_dict(),
        "encoder_opt": self.encoder_opt.state_dict(),
        "critic_opt": self.critic_opt.state_dict(),
    }

def load_state_dict(self, state_dict):
    self.encoder.load_state_dict(state_dict["encoder"])
    self.critic.load_state_dict(state_dict["critic"])
    self.critic_target.load_state_dict(state_dict["critic_target"])
    # Optimizers loaded best-effort — eval-only runs don't need them
    for k in ("encoder_opt", "critic_opt"):
        if k in state_dict:
            try:
                getattr(self, k).load_state_dict(state_dict[k])
            except (ValueError, KeyError):
                pass
```

Plus an eager-load path in `train_cqn_as.py:main`:
```python
snapshot_path = cfg.get("snapshot_path") if hasattr(cfg, "get") else None
if snapshot_path is not None:
    workspace.load_snapshot(snapshot_path)
if int(cfg.num_train_frames) <= 0:
    workspace.eval()       # eval-only mode
    return
```
This is what `phase1_reward_pilot_cqn_as.py --eval` requires (it emits `+snapshot_path=...` + `num_train_frames=0`).

**Architecture must match at load time.** A snapshot trained with `bodyslam=oracle` cannot be loaded into a workspace constructed with `bodyslam=off` — the low_dim encoder input dim differs by 24 (6D × frame_stack=4) and `load_state_dict` will raise on a shape mismatch. That's correct; the eval flow in `phase1_reward_pilot_cqn_as.py` passes `bodyslam={mode}` consistently.

## 9. Loading a CQN-AS snapshot as a *policy* in the SVF collector (Phase 2 G1)

The Phase-2 SVF re-eval under G1 needs to roll out a **trained CQN-AS policy**
as the `snapshot` source inside [`scripts/svf_collect_dataset.py`](../scripts/svf_collect_dataset.py)
and [`scripts/svf_threshold_sweep.py`](../scripts/svf_threshold_sweep.py).
That collector was written for RoboBase ACT/DP snapshots; CQN-AS snapshots are
a different shape and need their own path.

**Payload dispatch.** RoboBase snapshots carry `payload["cfg"]` (a Hydra
`DictConfig`) + `payload["agent"]`; CQN-AS (`train_cqn_as.py:save_snapshot`)
carries `payload["config"]` (resolved container) + `payload["agent_state"]`.
`load_snapshot_policy` branches on `"agent_state" in payload` and, for CQN-AS,
delegates to `_load_cqn_as_snapshot_policy`. `_peek_snapshot_cfg` accepts
either `cfg` or `config`, so the camera/bodyslam peeks work for both.

**Agent rebuild.** `make_agent` bakes `rgb_obs_shape` / `low_dim_obs_shape` /
`action_shape` into `config.agent` at train time (and `save_snapshot` persists
the resolved config), so the agent rebuilds with a plain
`hydra.utils.instantiate(config.agent)` + `agent.load_state_dict(agent_state)`
— no env needed to recover the specs.

**`_CQNASSnapshotPolicy` mirrors the adapter, not the agent.** The vendored
`CQNASAgent.act(rgb_obs, low_dim_obs, step, eval_mode)` takes **split** rgb /
low_dim arrays (not a gym dict), frame-stacked, with `[-1,1]`-normalised
actions. The wrapper replicates [`env_adapter.py`](../safety_bigym/agents/cqn_as/env_adapter.py)`._extract_obs`
(state-key concat → optional `human_pos_estimate` → per-camera frames stacked
to `(V, C·frame_stack, H, W)`) and `_convert_action_to_raw` (denorm to the env
range). It queries every step and executes `chunk[0]` (receding horizon —
matching the ACT `_SnapshotPolicy` convention), and **resets its frame-stack
deques per episode** via `reset()`, which `rollout_episode` calls after
`env.reset()`.

**`includes_human_pos` is decoupled from the env's bodyslam mode.** It's read
from the *snapshot's trained* mode (`config.env.bodyslam.mode`): a
`bodyslam=oracle`-trained policy is fed low_dim **without** noise-channel
surprises, while the collection env runs `--bodyslam-mode noisy` so the SVF
dataset still records `human_pos_estimate` (the critic's load-bearing
feature). Same separation the ACT path uses. In the sweep, pass
`--bodyslam-mode noisy` explicitly to match the **critic's training-collection
mode** when it differs from the snapshot's (the sweep otherwise peeks the
snapshot's mode).

**Device portability.** Snapshots trained on the GPU box carry
`device="cuda"`; `_load_cqn_as_snapshot_policy` overrides `config.agent.device`
to CPU when CUDA is absent, so the collector/sweep load locally too.

### Vendored `.view` → `.reshape` (inference path)

On torch ≥2 / CPU, `MultiViewCNNEncoder.forward` runs conv over the
**non-contiguous slice** `obs[:, v]`, and `C2FCriticNetwork.forward_each_level`
produces non-contiguous post-GRU/cat tensors — so `.view(...)` raises
`"view size is not compatible with input tensor's size and stride"`. Switched
the **four inference-path** calls (encoder line + `forward_each_level`) to
`.reshape`, which is value-identical for contiguous tensors and copies only
when needed. The training-path `forward` / C51 projection views are untouched
(not on the `act()` path). Logged in the `agent.py` header next to the
sanctioned `linspace`→`arange` fix. Works on the GPU box (same training code
path); the change just makes snapshot inference portable to CPU/newer torch.

### Local-venv note

To load a CQN-AS snapshot locally you need `tensordict==0.6.0` **and** its
transitive `orjson` (install `--no-deps` so it doesn't touch the local torch).
Both are already present on the GPU box.

### Tests

[`tests/test_cqn_as_snapshot_policy.py`](../tests/test_cqn_as_snapshot_policy.py)
covers the obs assembly (low_dim width with/without human_pos, rgb shape,
frame-stack reset, no-camera placeholder) and action denorm (mid/min/max) with
a stub agent — no MuJoCo. Validated end-to-end on a real G1 saucepan_to_hob
snapshot (`snapshot_17826.pt`): env build (G1, AMASS-free, 3 cameras) →
load → `act` → varied in-bounds 16-d actions.

---

## What stayed sane (worth knowing for future agents)

- **The env adapter ([`env_adapter.py`](../safety_bigym/agents/cqn_as/env_adapter.py)) is the right abstraction.** It bridges the SafetyBiGymEnv (gym dict obs + tuple step) and CQN-AS's TimeStep API, including frame stacking and action [-1,1] roundtrip. Pure-Python; tested via [`tests/test_cqn_as_adapter.py`](../tests/test_cqn_as_adapter.py) (24 tests, no MuJoCo/AMASS needed — monkeypatches `SafetyBiGymEnvFactory._create_env` to a stub env).
- **`bodyslam.mode != "off"` gating works cleanly** in the adapter — the low_dim encoder sizes correctly with the 6D `human_pos_estimate` channel injected. No agent-side changes were needed.
- **Per-step `info["safety"]` flows through to W&B unchanged.** The smoke gate (A6.3) explicitly confirmed `safety/ssm_margin`, `safety/pfl_force_ratio` etc. log at single-step granularity, not at K-step chunk boundaries.

## Pointers

- Upstream pristine clone: `/Users/ayushpatel/Documents/FYP3/CQN-AS/` (commit `8cf806e`). Don't modify.
- Vendored modules: [`safety_bigym/agents/cqn_as/`](../safety_bigym/agents/cqn_as/). `cqn_utils.py`, `utils.py`, `replay_buffer.py`, `agent.py` are verbatim-with-relative-imports + `state_dict`/`load_state_dict`. `env_adapter.py` is local.
- Hydra root: [`cfgs/cqn_as_config.yaml`](../cfgs/cqn_as_config.yaml). Agent sub-config: [`cfgs/agent/cqn_as.yaml`](../cfgs/agent/cqn_as.yaml).
- Smoke gate command (A6, ~5 min on GPU):
  ```bash
  export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
  python train_cqn_as.py \
    env=safety_bigym/dishwasher_close disruption=coworker_train \
    bodyslam=oracle num_train_frames=2000 num_demos=0 \
    wandb.use=true wandb.name=cqn_as_smoke_$(date +%s)
  ```

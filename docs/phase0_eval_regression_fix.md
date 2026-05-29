# Phase 0 — Why eval task_success was 0, and what fixed it

> **Historical/reference doc.** This records the DP EMA snapshot regression and
> RoboBase drift. It remains useful for debugging old snapshots, but current
> CQN-AS status lives in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## TL;DR

Training was healthy. Evaluation was effectively running an **untrained** policy because the diffusion-policy EMA weights — the ones actually used at inference — were silently dropped from snapshots. Fixing the serialization of `Actor.ema` turned 0% eval into the same numbers W&B showed during pretrain-eval (e.g. `reach_target_single` 50k snapshot → `R: 0.4000`).

## Symptom

`baseline_sweep.py --eval` reported `task_success = 0`, `episode_reward = 0`, `pfl_force_ratio = 0` for every (task × disruption) cell, even with `inject_human=false`. Snapshots loaded without errors; `env_steps=100000` on the restored scalar confirmed weights were reaching the process.

## False leads we ruled out

- **Snapshot-loading miss** — `env_steps=100000` is a restored scalar; would be 0 on a fresh agent.
- **Action-normalisation bug** — reach qpos came back physically sensible after denormalisation; dishwasher "drift" was just scene-origin geometry, not base drift.
- **OOD-from-human** — zero-human eval at the 100k snapshot was also 0%.
- **DP over-trained past peak (original hypothesis in `.claude/plans/cheeky-nibbling-floyd.md`)** — plausible from the W&B curves but wrong. The curves drop to 0 at the end because the *saved* snapshot's eval path was broken, not because the model had forgotten the task.

## Root cause

In [robobase/robobase/method/diffusion.py](../../robobase/robobase/method/diffusion.py):

```python
self.ema       = EMAModel(parameters=self.actor.parameters(), power=0.75)
self.ema_actor = copy.deepcopy(self.actor)

def infer(self, ...):
    actor = self.ema_actor
    self.ema.copy_to(actor.parameters())   # <-- eval weights come from ema.shadow_params
    ...
```

`diffusers.EMAModel` is a plain Python object, **not** an `nn.Module`. Its `shadow_params` — the weights DP actually uses at eval — are therefore **invisible** to `agent.state_dict()` / `nn.Module.state_dict()` walks, so they were never written into snapshots.

On reload, `agent.load_state_dict(..., strict=True)` succeeded silently:
- `self.actor.*` → restored (training weights, never used at eval).
- `self.ema_actor.*` → restored (carried correct EMA weights, but immediately overwritten — see below).
- `self.ema` → re-initialised by `__init__` as a fresh `EMAModel` tracking the current `self.actor.parameters()`.

Then at eval time `Actor.infer()` ran `self.ema.copy_to(self.ema_actor.parameters())` on every forward, overwriting the correctly-loaded `ema_actor` with the untrained fresh EMA shadow. Random actions, every rollout, regardless of task.

## Fix

All edits in [robobase/robobase/workspace.py](../../robobase/robobase/workspace.py), additive, tagged `FYP3/safety_bigym drift`.

**1. Persist the EMA shadow in `save_snapshot`:**

```python
actor = getattr(self.agent, "actor", None)
if actor is not None and hasattr(actor, "ema") and hasattr(actor.ema, "state_dict"):
    payload["actor_ema"] = actor.ema.state_dict()
```

**2. Restore it in `load_snapshot`, with a legacy-snapshot fallback:**

```python
ema_state = payload.pop("actor_ema", None)
actor = getattr(self.agent, "actor", None)
if actor is not None and hasattr(actor, "ema"):
    if ema_state is not None and hasattr(actor.ema, "load_state_dict"):
        actor.ema.load_state_dict(ema_state)
    elif hasattr(actor, "ema_actor"):
        # Legacy snapshot: seed fresh EMA's shadow from the ema_actor params
        # that ARE in agent.state_dict(). Relies on FYP3 drift saving right
        # after _eval() → infer() → copy_to(ema_actor), so ema_actor carries
        # the correct EMA weights.
        for shadow_p, loaded_p in zip(
            actor.ema.shadow_params, actor.ema_actor.parameters()
        ):
            shadow_p.data.copy_(loaded_p.data)
```

The legacy fallback is what let us recover Phase-0's existing snapshots without retraining. It works specifically because FYP3 drift saves snapshots at every pretrain-eval interval — `_eval()` calls `infer()`, which runs `self.ema.copy_to(self.ema_actor.parameters())` before any rollout, so by the time the snapshot is written `ema_actor.*` contains the trained shadow weights.

## Collateral fixes found along the way

These came up while chasing the regression but are independent bugs worth recording:

- **PyTorch 2.6 `weights_only` default flipped** — `torch.load(...)` raised `UnpicklingError` on snapshots because the payload contains an `omegaconf.DictConfig`. Fixed by passing `weights_only=False` at both call sites in `workspace.py`.
- **`cfg.demos=0` rejected by upstream assert** — [robobase/robobase/envs/bigym.py:65](../../robobase/robobase/envs/bigym.py#L65) assumed stats only came from demos. With Track 4 stats-in-snapshot this is a valid eval config. Relaxed to `assert cfg.demos != 0 or self._action_stats is not None`.
- **Task swap `dishwasher_load_plates` → `saucepan_to_hob`** — the former never learned (flat 0% on W&B pretrain-eval); the latter was picked as its replacement for Phase 0 and downstream phases. New env config at [cfgs/env/safety_bigym/saucepan_to_hob.yaml](../cfgs/env/safety_bigym/saucepan_to_hob.yaml) (`demos: 39`, `episode_length: 20000`, `demo_down_sample_rate: 25` → ~2.1× mean demo length of headroom).

## Verification

- GPU box eval after patch: `reach_target_single` 50k snapshot → `R: 0.4000` (non-zero task_success; matches the W&B pretrain_eval curve at that step).
- Legacy-snapshot fallback path exercised by the same run (pre-fix snapshots lacked `actor_ema` in payload, recovered via `ema_actor` params).

## Files touched

- [robobase/robobase/workspace.py](../../robobase/robobase/workspace.py) — EMA save/load + `weights_only=False`.
- [robobase/robobase/envs/bigym.py](../../robobase/robobase/envs/bigym.py) — relaxed `demos=0` assert.
- [cfgs/env/safety_bigym/saucepan_to_hob.yaml](../cfgs/env/safety_bigym/saucepan_to_hob.yaml) — new task config.
- [cfgs/env/safety_bigym/dishwasher_load_plates.yaml](../cfgs/env/safety_bigym/dishwasher_load_plates.yaml) — deleted.
- [scripts/baseline_sweep.py](../scripts/baseline_sweep.py) — `TASKS` / `SNAPSHOTS` updated.
- [phase0_workspace_drift.patch](../../phase0_workspace_drift.patch) — regenerated (144 lines) for GPU-box `git apply`.

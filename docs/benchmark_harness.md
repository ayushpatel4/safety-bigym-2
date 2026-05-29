# Benchmark Harness (`benchmark_policy.py`) — P6

The snapshot-evaluation benchmark harness. Given any policy checkpoint, it rolls the policy
out in the SafetyBiGym env per `(task, disruption, obs-mode)` cell over one or more seeds —
optionally wrapped with the Phase-2 SVF runtime filter — and appends **one CSV row per
cell** with the full safety-metric schema (bootstrap 95% CIs, CVaR/percentile tail-risk,
task metrics, filter mechanics). This is the canonical data source for every results table
in the report (8.1, 8.3, 8.5, 8.7, headline 8.9, 8.11) and the engine behind P5 rows 4/5.

- CLI: [`scripts/benchmark_policy.py`](../scripts/benchmark_policy.py)
- Package: [`safety_bigym/benchmark/`](../safety_bigym/benchmark/) (`stats`, `records`, `schema`, `aggregate`, `env_build`, `filter_attach`, `runners`, `loader`)
- Visualizer: [`scripts/benchmark_visualize.py`](../scripts/benchmark_visualize.py)
- One-shot demo: [`scripts/benchmark_demo.sh`](../scripts/benchmark_demo.sh)
- Tests: [`tests/test_benchmark_harness.py`](../tests/test_benchmark_harness.py)

## Install

The raw per-episode rolls are persisted as parquet, so `pandas` + `pyarrow` are required
(now in `setup.py`). On a fresh clone / the GPU box:

```bash
cd safety_bigym && venv/bin/python -m pip install "pandas>=2.0" "pyarrow>=12.0"
```

## Quick start

```bash
# Smoke (random policy, G1, AMASS-free, < 1 min CPU) — the documented P6 gate
python scripts/benchmark_policy.py --smoke --out results/smoke.csv

# A headline-style cell on a trained policy + the SVF filter
python scripts/benchmark_policy.py \
  --snapshot runs/saucepan_g1/final.pt \
  --filter-snapshot svf_coworker_train_v1.pt --filter-threshold 4.0 \
  --task saucepan_to_hob --disruption coworker_train --obs-mode noisy \
  --human-model g1 --seeds 0,1,2 --episodes 20 --out results/row5.csv

# Local end-to-end proof (random ± local SVF critic under G1) + figures
bash scripts/benchmark_demo.sh
```

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--snapshot` | *(none)* | Policy checkpoint. Omit → random policy. ACT vs CQN-AS auto-detected from payload keys. |
| `--filter-snapshot` | *(none)* | SVF critic checkpoint to wrap the policy with `SafetyFilterWrapper`. |
| `--filter-threshold` | `4.0` | SVF Q-value threshold `R`; the filter vetoes when `q < R`. |
| `--fallback` | `zero_velocity` | Fallback action when the filter triggers (`FallbackRegistry`). |
| `--task` | `saucepan_to_hob` | Task key (see `env_build.TASK_REGISTRY`). For CQN-AS, defaults to the snapshot's trained task; overriding warns + merges the env config. |
| `--disruption` | `coworker_train` | `coworker_train` / `coworker_eval` (ParameterSpace presets) or a `DisruptionType` name. |
| `--obs-mode` | `noisy` | `off` / `oracle` / `noisy` (BodySLAM). **Must be `oracle`/`noisy` when filtering.** |
| `--human-model` | `g1` | `g1` (headline, AMASS-free) or `smplh` (needs `AMASS_DATA_DIR`). |
| `--seeds` | `0` | Comma-separated, e.g. `0,1,2`. All seeds × episodes aggregate into **one** row. |
| `--episodes` | `20` | Episodes per seed. |
| `--max-steps` | `300` | Max env steps per episode. |
| `--out` | *(required)* | Per-cell CSV (appended; header written once). |
| `--stats-seed` | `12345` | Bootstrap RNG seed — CIs reproduce across re-runs. |
| `--num-resamples` | `10000` | Bootstrap resamples. |
| `--render` | off | Best-effort: write a rollout mp4 next to `--out` (needs a GL backend). |
| `--smoke` | off | 1 seed × 2 episodes × 50 steps, single cell. |

### Outputs

- `--out` (CSV): one appended row per invocation, schema = `schema.COLUMNS` (61 columns).
- `<out>.raw_episodes.parquet`: every per-episode record — re-aggregate to the CSV row
  with `records.read_parquet` + `aggregate.aggregate_cell` (no re-rollout needed).
- `<out>.episodes.jsonl`: a live, crash-resilient sidecar (one JSON line per episode).

## Conventions & requirements

- **Success** = `info["task_success"]` (matches `train_cqn_as`'s `success_rate`), with
  cumulative-reward > 0 as a documented fallback when the key is absent. `steps_to_completion`
  is the env-step index of the first success, averaged over **successful episodes only**.
- **Filter ⇒ obs-mode.** The SVF critic consumes `human_pos_estimate`, so `--filter-snapshot`
  requires `--obs-mode oracle|noisy`; the harness hard-errors on `off`. Attach time also
  asserts `critic.spec.obs_keys ⊆ env obs keys`.
- **CQN-AS demo stats.** A CQN-AS snapshot does not carry `action_stats` (they are
  demo-derived). The harness replays `get_demos` for its action-stat side effect, exactly
  as training did. With `--obs-mode != off` that demo step injects `human_pos_estimate` via
  AMASS, so **`AMASS_DATA_DIR` must be exported for CQN-AS eval** (the *live* G1 rollout is
  still AMASS-free). The snapshot's baked `env.motion_clip_dir` (a machine-specific absolute
  path from the training box) is automatically rebased onto the local `AMASS_DATA_DIR` —
  the `motion_clip_paths` are relative, so clips resolve on any machine. The action-stat
  step holds all demos (with pixels) in memory at once; the snapshot's full count (e.g. 36)
  can OOM a laptop, so pass e.g. `--num-demos-for-stats 5` for an approximate local run and
  the full count on the GPU box for faithful normalisation.
- **PFL columns are inert** (`pfl_*` ≈ 0, `pfl_violations_per_region_json = "{}"`) under the
  open BiGym contact-detection bug; the schema is forward-compatible.

## CSV column dictionary

Definitions of the `ep_*` fields follow [`docs/safety_metrics.md`](safety_metrics.md).

**Identification:** `task, disruption, obs_mode, human_model, policy_kind, snapshot,
filter_snapshot, filter_threshold, seeds, episodes_per_seed, n_episodes, n_steps, git_sha,
timestamp_utc`.

**Task:** `success_rate(+_ci_lo/_ci_hi)`, `episode_reward_mean(+ci)`, `mean_episode_length`,
`steps_to_completion(+ci)` (successful episodes only).

**Safety (episode means; CI on the three headline axes):**
`ep_proximity_violation_rate(+ci)` *(thesis-primary)*, `ep_ssm_violation_rate`,
`ep_ssm_violation_actual_rate(+ci)`, `ep_pfl_violation_rate`,
`ep_time_in_proximity_{0p3,0p5,1p0}m`, `ep_min_separation(+ci)`, `ep_min_separation_lowest`
(worst episode), `ep_mean_separation`, `ep_p5_separation`, `ep_p25_separation`,
`ep_min_ssm_margin`, `ep_min_ssm_margin_actual`, `ep_max_robot_vel`, `ep_mean_robot_vel`,
`ep_time_to_first_violation`, `pfl_violations_per_region_json`.

**Tail-risk (over per-episode arrays):** `cvar95_ep_cost_integral` (mean of worst 5%
highest cost integrals), `mean_ep_cost_integral`, `cvar95_ep_min_separation` (mean of worst
5% lowest separations), `p99_ep_min_separation` (1st percentile — dangerous tail),
`p5_ep_min_separation`.

**Filter mechanics (empty unless `--filter-snapshot`):** `filter_intervention_rate(+ci)`,
`filter_passthrough_rate` (= 1 − intervention), `mean_per_episode_interventions`,
`mean_q_value`, `n_interventions`, `filter_fallback`.

## Where each path runs

| Path | Built by | Locally testable on this Mac? |
|---|---|---|
| Random policy, G1 | `env_build.build_g1_gym_env` | ✅ fully (AMASS-free) |
| Random + SVF filter, G1 | `+ filter_attach.attach_filter_gym` | ✅ fully (uses local `svf_coworker_train_v1.pt`) |
| ACT snapshot | `loader` + `svf_collect.load_snapshot_policy` | dispatch unit-tested; full run needs an ACT snapshot |
| CQN-AS snapshot | `env_build.build_cqn_*` + `runners.CQNASRunner` | veto math unit-tested (`apply_veto`); full run needs a CQN-AS snapshot + `AMASS_DATA_DIR` |

The CQN-AS in-loop veto (`runners.apply_veto`) is the one piece without a real-snapshot
local test; its kernel is unit-tested with a stub critic + identity transforms, and a
`critic.spec ⊆ obs` assertion fails loud on misconfiguration.

## Visualize

The visualizer takes any number of **existing** per-cell CSVs (each is produced by a
`benchmark_policy.py` run). The quickest way to get two real CSVs is the demo, then point
the visualizer at its outputs:

```bash
bash scripts/benchmark_demo.sh                       # writes results/benchmark_demo/*.csv (+ parquet sidecars)
venv/bin/python scripts/benchmark_visualize.py \
  --csv results/benchmark_demo/random_nofilter.csv results/benchmark_demo/random_filter.csv \
  --out-dir results/figs
```

> Use `venv/bin/python` (or activate the venv): the separation plot reads the parquet
> sidecars via `pandas`, which lives in the venv — not the system Python.

For the actual report, pass the headline cell CSVs you generated, e.g.
`--csv results/e4.1_row1.csv results/e4.1_row5.csv` (these names are illustrative — they
exist only after you run `benchmark_policy.py --out results/e4.1_row1.csv …`).

Writes `pareto.png` (intervention rate vs `ep_proximity_violation_rate`, CI error bars),
`cells_bars.png` (per-cell proximity / ssm-actual / success), and `separation.png`
(per-episode `ep_min_separation` distributions with the p99 lower-tail marker, from the
parquet sidecars).

## Tests

```bash
cd safety_bigym && pytest tests/test_benchmark_harness.py
```

Eight CPU tests (no model load, no env construction beyond a pure stub): bootstrap vs
manual numpy, CVaR/percentile constants, parquet roundtrip re-aggregation, schema
completeness, real-`SafetyFilterWrapper` attach, `apply_veto` kernel, loader dispatch.

# safety_bigym

`safety_bigym` extends [BiGym](https://github.com/chernyadev/bigym) with a
live human coworker and ISO 15066-inspired safety monitoring for manipulation
tasks. It is the experimental codebase behind a Hybrid Safety Critic: a
constrained-RL policy trained to avoid close human-robot interactions, backed
by an offline Safety Value Function (SVF) filter at deployment time.

The current headline setup uses a Unitree G1 coworker in COWORKER scenarios.
The older SMPL-H path is still supported for comparison and requires AMASS
motion clips.

## What is in this repo

- `safety_bigym/`: environment wrappers, human controllers, safety monitors,
  filters, CQN-AS integration, and WCSAC baseline code.
- `cfgs/`: Hydra configs for environments, agents, training runs, and
  benchmark cells.
- `scripts/`: training, evaluation, diagnostics, and report-figure utilities.
- `tests/`: unit and smoke tests for the safety wrappers, agents, filters, and
  benchmark harness.
- `docs/`: maintained project notes, experiment summaries, and metric
  definitions.
- `FYP_v16_fable/`: current thesis draft and report-local assets.

## Setup

This repository is designed to run inside the wider `FYP3` workspace, with
local sibling checkouts of `bigym` and `robobase`.

```bash
cd safety_bigym
./venv/bin/python -m pip install -e ".[dev]"
```

On macOS, prefer `./venv/bin/python` over activating the venv. Some shells on
this machine resolve `python` to the system interpreter after activation.

SMPL-H runs need AMASS clips:

```bash
export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU
```

G1 coworker runs do not need AMASS.

## Common commands

Run the test suite:

```bash
./venv/bin/python -m pytest tests/
```

Run a lightweight Phase 3 smoke:

```bash
./venv/bin/python scripts/phase3_p30_smoke.py \
  env=safety_bigym/dishwasher_close \
  disruption=coworker_train \
  bodyslam=oracle \
  pixels=false
```

Run the benchmark smoke:

```bash
./venv/bin/python scripts/benchmark_policy.py --smoke --out results/smoke.csv
```

Train CQN-AS / Lagrangian agents through Hydra:

```bash
./venv/bin/python train_cqn_as.py agent=cqn_as_lagrangian env=safety_bigym/saucepan_to_hob
```

Use the launch scripts in `scripts/` for full experiment matrices. Avoid
starting multi-hour training runs from an interactive coding session; run a
short smoke first, then dispatch the real run deliberately.

## Current documentation

- `docs/IMPLEMENTATION_STATUS.md`: current experiment status and run gates.
- `docs/PROJECT_PLAN.md`: technical plan for the Hybrid Safety Critic.
- `docs/safety_metrics.md`: definitions of proximity, SSM, and per-episode
  safety metrics.
- `docs/benchmark_harness.md`: snapshot evaluation and aggregation harness.
- `docs/g1_coworker_swap.md`: G1 coworker integration notes.
- `docs/cqn_as_integration_notes.md`: local CQN-AS integration notes and
  known pitfalls.

Older handoff files and superseded phase plans have been removed from the
working tree. Historical results that are still useful for the thesis remain
under `docs/`.

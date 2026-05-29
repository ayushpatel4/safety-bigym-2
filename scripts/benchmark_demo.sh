#!/usr/bin/env bash
# One-shot local proof of the P6 benchmark harness: a random policy under the G1 coworker
# (AMASS-free) evaluated twice — without the SVF filter and with the local SVF critic —
# then visualized. This is the dependency-light end-to-end smoke that runs on this Mac.
#
# Usage:
#   bash scripts/benchmark_demo.sh [OUT_DIR]
#
# On a headless box, export MUJOCO_GL=egl (Linux) before running. Add --render to either
# benchmark_policy.py call below to also write a rollout mp4 (needs a working GL backend).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=venv/bin/python
OUT="${1:-results/benchmark_demo}"
mkdir -p "$OUT"

echo "[1/3] random policy, G1, coworker_train, noisy — NO filter"
"$PY" scripts/benchmark_policy.py \
  --task saucepan_to_hob --disruption coworker_train --obs-mode noisy --human-model g1 \
  --seeds 0 --episodes 3 --max-steps 80 \
  --out "$OUT/random_nofilter.csv"

echo "[2/3] random policy, G1, coworker_train, noisy — WITH local SVF filter (R=4.0)"
"$PY" scripts/benchmark_policy.py \
  --task saucepan_to_hob --disruption coworker_train --obs-mode noisy --human-model g1 \
  --seeds 0 --episodes 3 --max-steps 80 \
  --filter-snapshot svf_coworker_train_v1.pt --filter-threshold 4.0 \
  --out "$OUT/random_filter.csv"

echo "[3/3] visualize"
"$PY" scripts/benchmark_visualize.py \
  --csv "$OUT/random_nofilter.csv" "$OUT/random_filter.csv" \
  --out-dir "$OUT/figs"

echo "Demo complete. Outputs in: $OUT"

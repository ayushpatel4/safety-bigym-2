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
# Production G1 filter + operating point (override if a different checkpoint is
# the one present locally, e.g. SVF=svf_coworker_train_v1.pt FILTER_R=4.0).
SVF="${SVF:-checkpoints/svf_coworker_train_g1_0p3.pt}"
FILTER_R="${FILTER_R:-2.25}"

echo "[1/3] random policy, G1, coworker_train, noisy — NO filter"
"$PY" scripts/benchmark_policy.py \
  --task saucepan_to_hob --disruption coworker_train --obs-mode noisy --human-model g1 \
  --seeds 0 --episodes 3 --max-steps 80 \
  --out "$OUT/random_nofilter.csv"

echo "[2/3] random policy, G1, coworker_train, noisy — WITH SVF filter (R=${FILTER_R})"
"$PY" scripts/benchmark_policy.py \
  --task saucepan_to_hob --disruption coworker_train --obs-mode noisy --human-model g1 \
  --seeds 0 --episodes 3 --max-steps 80 \
  --filter-snapshot "$SVF" --filter-threshold "$FILTER_R" \
  --out "$OUT/random_filter.csv"

echo "[3/3] visualize"
"$PY" scripts/benchmark_visualize.py \
  --csv "$OUT/random_nofilter.csv" "$OUT/random_filter.csv" \
  --out-dir "$OUT/figs"

echo "Demo complete. Outputs in: $OUT"

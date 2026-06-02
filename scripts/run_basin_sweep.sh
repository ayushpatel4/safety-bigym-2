#!/usr/bin/env bash
# scripts/run_basin_sweep.sh
# Benchmark a list of checkpoints from ONE Lagrangian stage dir on ONE obs mode,
# to map deployment proximity vs training step (the d0.3 "avoidance basin") and
# locate that seed's ROW3 operating point. PURE EVAL (benchmark_policy, no train).
#
# Why this exists: peak-success / final-checkpoint selection misses the basin
# (the constraint induces a mid-training proximity dip that late reward-chasing
# erodes). So per CONFIRM seed we sweep the basin and pick the best-DEPLOYING
# checkpoint at acceptable success (analyze_row3.py --pick), not the train-eval
# nominee.
#
# Round-robins the STEPS across GPUS. One CSV + log per checkpoint under OUT.
#
# Usage:
#   STAGE_DIR=exp_local/e3_2_cost_budget/<run>/d0p3_seed1 OBS=noisy bash scripts/run_basin_sweep.sh
#   STAGE_DIR=... STEPS="20279 22533 25303 27554 30122 33386" GPUS="0 1 2 3 4 5" bash scripts/run_basin_sweep.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"

STAGE_DIR="${STAGE_DIR:?set STAGE_DIR=exp_local/.../d0pX_seedN}"
OBS="${OBS:-noisy}"
TASK="${TASK:-saucepan_to_hob}"; DISRUPTION="${DISRUPTION:-coworker_train}"; HUMAN_MODEL="${HUMAN_MODEL:-g1}"
SEEDS="${SEEDS:-0,1,2}"; EPISODES="${EPISODES:-20}"
GPUS="${GPUS:-0 1 2 3 4 5}"
# Default basin window from d0.3/seed0 (steps where it avoided). Each CONFIRM seed
# saves at its own eval cadence, so if a step is missing it's skipped (logged) and
# you can re-run with that seed's actual STEPS (ls $STAGE_DIR/snapshot_[0-9]*.pt).
STEPS="${STEPS:-18030 20279 22533 25303 27554 30122 33386 35005}"
OUT="${OUT:-$REPO_ROOT/results/e4_1/basin_$(basename "$STAGE_DIR")_${OBS}}"; mkdir -p "$OUT"

echo "== basin sweep | stage=$STAGE_DIR obs=$OBS | GPUS='$GPUS' -> $OUT =="
read -r -a gpus <<<"$GPUS"; n=${#gpus[@]}; i=0
for st in $STEPS; do
  snap="$STAGE_DIR/snapshot_${st}.pt"
  if [[ ! -f "$snap" ]]; then echo "-- skip step $st (no $snap)"; continue; fi
  g="${gpus[$((i % n))]}"; i=$((i+1))
  echo "   gpu$g <- snapshot_${st}.pt"
  CUDA_VISIBLE_DEVICES="$g" python scripts/benchmark_policy.py --snapshot "$snap" \
    --task "$TASK" --disruption "$DISRUPTION" --human-model "$HUMAN_MODEL" --obs-mode "$OBS" \
    --num-demos-for-stats 0 --seeds "$SEEDS" --episodes "$EPISODES" \
    --out "$OUT/s${st}.csv" > "$OUT/s${st}.log" 2>&1 &
  (( i % n == 0 )) && wait   # drain a full GPU batch before launching the next
done
wait
echo "== basin sweep done. CSVs in $OUT =="
echo "   pick the operating point:  python scripts/analyze_row3.py pick --sweep-dir $OUT --success-floor 0.75"

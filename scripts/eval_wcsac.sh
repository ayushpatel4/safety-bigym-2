#!/usr/bin/env bash
# Evaluate the WCSAC (E3.7/P9) sweep via benchmark_policy.py (cvar95 tail-risk
# + bootstrap CIs, directly comparable to the Lagrangian P5 rows). Evals each
# cell's FINAL (converged) snapshot under the training disruption/obs-mode.
#
# NOTE: WCSAC trained num_demos=0, so benchmark/env_build.build_cqn_adapter now
# keeps the adapter's default IDENTITY action stats (no get_demos) -- matching
# training. Validated against train_cqn_as eval (prox_rate / success agree).
set -euo pipefail
cd "$(dirname "$0")/.."
export AMASS_DATA_DIR="${AMASS_DATA_DIR:-/home/ap2322/Documents/CMU/CMU}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

GPU="${GPU:-0}"
EPISODES="${EPISODES:-30}"
DISRUPTION="${DISRUPTION:-coworker_train}"
OBS_MODE="${OBS_MODE:-oracle}"     # WCSAC trained with bodyslam=oracle
OUT="${OUT:-results/wcsac_eval}"
TASKS="${TASKS:-dishwasher_close drawers_open_all saucepan_to_hob}"
SEEDS="${SEEDS:-0 1 2}"            # training seeds (one cell each); eval-seed fixed at 0
mkdir -p "$OUT" logs

for task in $TASKS; do
  for b in 5 15 30; do
    for s in $SEEDS; do
      cell="wcsac_${task}_b${b}_s${s}"
      [ -f "$OUT/${cell}.csv" ] && { echo "skip $cell (CSV exists)"; continue; }
      dir="exp_local/wcsac/${cell}"
      snap=$(ls "$dir"/snapshot_*.pt 2>/dev/null | grep -oE 'snapshot_[0-9]+\.pt' \
             | sort -t_ -k2 -n | tail -1)
      if [ -z "$snap" ]; then echo "!! no snapshot for $cell, skipping"; continue; fi
      echo "=================================================================="
      echo "  eval $cell  snapshot=$snap  episodes=$EPISODES  ($(date))"
      echo "=================================================================="
      # --seeds 0: fixed eval episodes across all cells so seed-to-seed
      # differences reflect the trained policy, not the eval rollouts.
      env CUDA_VISIBLE_DEVICES="$GPU" ./venv/bin/python scripts/benchmark_policy.py \
        --snapshot "$dir/$snap" --task "$task" --disruption "$DISRUPTION" \
        --obs-mode "$OBS_MODE" --human-model g1 --seeds 0 --episodes "$EPISODES" \
        --out "$OUT/${cell}.csv" 2>&1 | tee "logs/eval_${cell}.log" \
        | grep -E "seed=0 ep=|Wrote|kind=" | tail -3
    done
  done
done
echo "ALL WCSAC EVALS DONE -> $OUT"

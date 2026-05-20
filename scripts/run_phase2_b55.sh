#!/usr/bin/env bash
# Phase 2 — B5.5: re-collect SVF dataset with tanh-denormalized snapshot
# actions, retrain the safety critic, and re-eval on coworker_train +
# coworker_eval at the v1 operating point (R≈4.0).
#
# Fires only if B5.3's in-distribution eval confirms v1 critic narrowness.
# See safety_bigym/docs/phase2_results.md §B5.5 for context.
#
# ============================== PRECONDITION =================================
# The snapshot-action denormalization patch must be in `main` before running.
# Sketch (apply to `scripts/svf_collect_dataset.py`, ~10 lines):
#
#   @dataclass
#   class _SnapshotPolicy:
#       ...
#       action_stats: Optional[Dict[str, np.ndarray]] = None
#       min_max_margin: float = 0.0
#
#       def __call__(self, obs):
#           ...
#           action_np = action.detach().cpu().numpy()
#           if action_np.ndim >= 2:
#               action_np = action_np.reshape(-1, action_np.shape[-1])[0]
#           if self.action_stats is not None:
#               # Mirror RoboBase RescaleFromTanhWithMinMax: tanh-space → env
#               from robobase.envs.wrappers.rescale_from_tanh import (
#                   RescaleFromTanhWithMinMax,
#               )
#               action_np = RescaleFromTanhWithMinMax.transform_from_tanh(
#                   action_np, self.action_stats, self.min_max_margin,
#               )
#           return action_np.astype(np.float32, copy=False)
#
# And in `load_snapshot_policy`, after `torch.load(...)`:
#
#     action_stats = payload.get("action_stats")  # written by FYP3/robobase drift
#     min_max_margin = float(cfg.get("min_max_margin", 0.0))
#     ...
#     return _SnapshotPolicy(
#         agent=agent, cameras=cameras, camera_resolution=resolution,
#         includes_human_pos=includes_human_pos,
#         action_stats=action_stats, min_max_margin=min_max_margin,
#     )
#
# Plus a regression test (`tests/test_svf_collect_snapshot_denorm.py`) that
# asserts: (a) without action_stats the policy returns ≈tanh-space; (b) with
# action_stats the gripper dims land in [0, 1] and body-joint dims span the
# env's ±π range.
# ============================================================================
#
# Usage (GPU box, from ~/Documents/safety_bigym):
#     bash scripts/run_phase2_b55.sh                       # full run
#     bash scripts/run_phase2_b55.sh --smoke               # 200-transition smoke
#     STAGES=collect bash scripts/run_phase2_b55.sh        # stop after collect

set -euo pipefail

# --- env ---------------------------------------------------------------------
cd "$(dirname "$0")/.."                       # repo root = safety_bigym/
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
: "${AMASS_DATA_DIR:?AMASS_DATA_DIR must be set (e.g. ~/Documents/CMU/CMU)}"

PY="venv/bin/python"
SMOKE="${1:-}"
STAGES="${STAGES:-collect,train,eval,sweep}"     # comma-separated stage gate
DATASET_DIR="datasets/svf_coworker_train_v2"
CKPT="checkpoints/svf_coworker_train_v2.pt"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR" "$(dirname "$CKPT")"

stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# --- sanity: confirm the denormalization patch is in ------------------------
echo "[B5.5] Verifying snapshot-denormalization patch is present…"
$PY - <<'PY'
import inspect
from scripts.svf_collect_dataset import _SnapshotPolicy
src = inspect.getsource(_SnapshotPolicy)
assert "action_stats" in src and "transform_from_tanh" in src, (
    "B5.5 precondition not met: _SnapshotPolicy.__call__ still returns raw "
    "tanh-space actions. Land the denormalization patch first (see the header "
    "of scripts/run_phase2_b55.sh)."
)
print("  patch present ✓")
PY

# --- stage 1: collect v2 -----------------------------------------------------
if stage collect; then
  echo "[B5.5] Collecting v2 dataset → $DATASET_DIR"
  EPISODES=${EPISODES:-210}
  MAX_STEPS=${MAX_STEPS:-250}
  if [[ "$SMOKE" == "--smoke" ]]; then
    EXTRA="--smoke"
  else
    EXTRA="--episodes-per-cell $EPISODES --max-steps $MAX_STEPS"
  fi
  $PY scripts/svf_collect_dataset.py \
      --source random --source snapshot \
      --tasks dishwasher_close drawers_open_all saucepan_to_hob \
      --disruption-space coworker_train \
      --bodyslam-mode noisy \
      --proximity-threshold 0.50 \
      --output-dir "$DATASET_DIR" \
      --seed 0 \
      $EXTRA \
      2>&1 | tee "/tmp/svf_collect_v2_$(date +%Y%m%d_%H%M%S).log"
fi

# --- stage 2: train v2 critic -----------------------------------------------
if stage train; then
  echo "[B5.5] Training v2 SVF → $CKPT"
  TRAIN_ARGS="--num-steps 200000 --batch-size 512 --cql-alpha 5.0 --target-violation-rate 0.30"
  [[ "$SMOKE" == "--smoke" ]] && TRAIN_ARGS="--smoke"
  $PY scripts/svf_train_critic.py \
      --dataset-dir "$DATASET_DIR" \
      --output "$CKPT" \
      $TRAIN_ARGS \
      2>&1 | tee "/tmp/svf_train_v2_$(date +%Y%m%d_%H%M%S).log"
fi

# --- stage 3: eval at v1 operating point R=4.0 ------------------------------
if stage eval; then
  for cell in coworker_train coworker_eval; do
    out="$RESULTS_DIR/svf_eval_v2_${cell}.csv"
    echo "[B5.5] Eval on $cell → $out"
    EVAL_ARGS="--episodes-per-cell 20 --max-steps 250"
    [[ "$SMOKE" == "--smoke" ]] && EVAL_ARGS="--smoke"
    $PY scripts/svf_eval_filter.py \
        --critic-path "$CKPT" \
        --threshold-R 4.0 \
        --policy random \
        --tasks dishwasher_close drawers_open_all saucepan_to_hob \
        --disruptions "$cell" \
        --bodyslam-mode noisy \
        --output-csv "$out" \
        $EVAL_ARGS \
        --seed 0 \
        2>&1 | tee "/tmp/svf_eval_v2_${cell}.log"
  done
fi

# --- stage 4: threshold sweep around v2 q_mean ------------------------------
# Read v2 q_mean off the train log (last "q_mean=" line); bracket ±1.5× per
# B5.4 methodology note. Default falls back to v1's range if grep fails.
if stage sweep; then
  Q_MEAN=$(grep -oE "q_mean=[0-9.]+" "/tmp/svf_train_v2_"*.log 2>/dev/null \
           | tail -1 | awk -F= '{print $2}' || true)
  if [[ -n "$Q_MEAN" ]]; then
    LO=$(python3 -c "print(max(0.5, $Q_MEAN - 1.5))")
    HI=$(python3 -c "print($Q_MEAN + 1.5)")
    THRS=$(python3 -c "import numpy as np; print(' '.join(f'{x:.2f}' for x in np.linspace($LO, $HI, 7)))")
  else
    THRS="2.0 2.5 3.0 3.5 4.0 4.5 5.0"
  fi
  echo "[B5.5] Threshold sweep on coworker_eval at thresholds: $THRS"
  for task in dishwasher_close drawers_open_all saucepan_to_hob; do
    out="$RESULTS_DIR/svf_sweep_v2_${task}.csv"
    SW_ARGS="--episodes-per-R 10 --max-steps 250"
    [[ "$SMOKE" == "--smoke" ]] && SW_ARGS="--smoke"
    $PY scripts/svf_threshold_sweep.py \
        --critic-path "$CKPT" \
        --thresholds $THRS \
        --policy random \
        --task "$task" \
        --disruption coworker_eval \
        --bodyslam-mode noisy \
        --output-csv "$out" \
        $SW_ARGS \
        --seed 0 \
        2>&1 | tee "/tmp/svf_sweep_v2_${task}.log"
  done
fi

echo "[B5.5] Done. Results under $RESULTS_DIR/. Next: paste numbers into docs/phase2_results.md §B5.5."

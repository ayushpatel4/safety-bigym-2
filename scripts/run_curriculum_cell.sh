#!/usr/bin/env bash
# run_curriculum_cell.sh — one human-curriculum cell: stage1 (coworker_easy) then
# stage2 (coworker_train), resuming from a stage-0 snapshot, with the working
# isolation recipe (no workspace penalty, all demos, widened critic support,
# peak-snapshot resume). rung3 adds the potential-based progress reward.
#
# Driven by scripts/dispatch_curriculum.py (sets CUDA_VISIBLE_DEVICES + env).
# Env in: NAME TASK RUNG(rung1|rung3) SEED GOAL DEMOS START_SNAP
#         STAGE1_FRAMES(30000) STAGE2_FRAMES(40000)
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"

: "${NAME:?}"; : "${TASK:?}"; : "${SEED:?}"; : "${DEMOS:?}"; : "${START_SNAP:?}"
RUNG="${RUNG:-rung1}"; GOAL="${GOAL:-0.0}"
STAGE1_FRAMES="${STAGE1_FRAMES:-30000}"; STAGE2_FRAMES="${STAGE2_FRAMES:-40000}"
OUTDIR="$REPO_ROOT/exp_local/curriculum/$NAME"; mkdir -p "$OUTDIR"
export AMASS_DATA_DIR="${AMASS_DATA_DIR:-/home/ap2322/Documents/CMU/CMU}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

COMMON=(
  "env=safety_bigym/${TASK}" env.human_model=g1 env.smplh_motion=amass
  bodyslam=oracle "num_demos=${DEMOS}" env.safety.add_workspace_penalty=false
  agent.v_min=-6.0 agent.v_max=2.0 agent.atoms=101
  "seed=${SEED}" save_snapshot=true save_video=true
  wandb.use=true wandb.project=safety-critic
)
if [[ "$RUNG" == "rung3" ]]; then
  COMMON+=(env.safety.add_progress_reward=true env.safety.progress_beta=1.0
           "env.safety.progress_goal=${GOAL}" env.safety.progress_gamma=1.0)
fi

run_stage() {  # name disruption frames resume_from
  local sname="$1" disr="$2" frames="$3" resume="$4"
  local sdir="$OUTDIR/$sname"
  [[ -f "$sdir/final_metrics.json" ]] && { echo "[$NAME] $sname already done"; return 0; }
  echo "[$NAME] $sname: disruption=$disr frames=$frames resume=$resume"
  venv/bin/python train_cqn_as.py "${COMMON[@]}" \
    "disruption=$disr" "num_train_frames=$frames" \
    "+snapshot_path=$resume" \
    "hydra.run.dir=$sdir" "wandb.name=curriculum_${NAME}_${sname}" \
    "+wandb.tags=[curriculum,$RUNG,task:$TASK,$sname,seed:$SEED]"
}

best() { venv/bin/python scripts/pick_best_snapshot.py "$1" 2>/dev/null; }

run_stage stage1_easy coworker_easy "$STAGE1_FRAMES" "$START_SNAP"
SNAP1="$(best "$OUTDIR/stage1_easy")"
[[ -n "$SNAP1" ]] || { echo "[$NAME] FATAL: no stage1 snapshot" >&2; exit 1; }
run_stage stage2_full coworker_train "$STAGE2_FRAMES" "$SNAP1"
echo "[$NAME] curriculum complete -> $OUTDIR/stage2_full"

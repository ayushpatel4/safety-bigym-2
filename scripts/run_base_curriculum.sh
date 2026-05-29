#!/usr/bin/env bash
# Staged human-curriculum re-validation of the CQN-AS base policy on
# saucepan_to_hob, with the 2026-05-20 reward/critic-support fix applied.
#
# Why this script exists: the 50k single-stage validation produced a degenerate
# "retreat from human" policy. Root cause = the dense workspace penalty's
# discounted return blew past the C51 critic support [-2,2], saturating value
# learning (full writeup: docs/phase3_base_validation_findings.md). The fix is
# four levers, all applied here:
#   (1) bounded penalty:  workspace_beta=0.05, workspace_excess_cap=1.0
#   (2) widened support:  agent.v_min=-6 agent.v_max=2 agent.atoms=101
#   (3) more demos:       num_demos=36
#   (4) human curriculum: 3 stages via snapshot-resume (the env is stateless
#       w.r.t. training step, so no within-run ramp — we stage it instead):
#         stage 0  disruption=null          (no human; "can it learn the task?")
#         stage 1  disruption=coworker_easy  (gentle coworker)
#         stage 2  disruption=coworker_train (full coworker)
#
# Each stage resumes the previous stage's final snapshot via +snapshot_path=...
# (train_cqn_as.py load_snapshot). The env is rebuilt per stage, so the new
# disruption config takes effect with no code change.
#
# NOT launched from inside the agent — a human runs this on the GPU box.
#
# Prereqs:
#   export AMASS_DATA_DIR=/path/to/CMU/CMU
#   export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0   # headless GPU
#   cd safety_bigym && source venv/bin/activate    # needs tensordict
#
# Usage:
#   scripts/run_base_curriculum.sh                 # full run (stages 0->1->2)
#   SMOKE=1 scripts/run_base_curriculum.sh         # ≤2000-frame stage-0 smoke only
#   STAGE0_FRAMES=60000 STAGE1_FRAMES=30000 STAGE2_FRAMES=40000 \
#       scripts/run_base_curriculum.sh             # override per-stage budgets
#
# Auto RUN_TAG (when unset): base_<human>_<frames>_<YYYYMMDD_HHMMSS>
#   e.g. base_smplh_proc_30k_30k_40k_20260528_153045
# Manual RUN_TAG=... also gets _<YYYYMMDD_HHMMSS> appended unless already present.
#
#   RUN_TAG=my_label scripts/run_base_curriculum.sh
#
#   # Resume ONLY stage 2 (e.g. after a machine crash) — skips stages 0/1 and
#   # restarts stage 2 from the newest snapshot it can find. Point RESUME_DIR at
#   # the prior run dir; the resumed run writes to <RESUME_DIR>/stage2_full_resume.
#   RESUME_STAGE2=1 RESUME_DIR=exp_local/cqn_as_base_curriculum/<run_tag> \
#       scripts/run_base_curriculum.sh
#   # ...or pass an explicit checkpoint:
#   RESUME_STAGE2=1 RESUME_DIR=<dir> RESUME_SNAPSHOT=<path.pt> \
#       scripts/run_base_curriculum.sh

set -euo pipefail

# HUMAN_MODEL selects which humanoid plays the coworker role.
#   smplh (default): AMASS-driven SMPL-H human (requires AMASS_DATA_DIR unless
#                    SMPLH_MOTION=procedural).
#   g1            : Unitree G1 standing-pose mannequin (no AMASS). Asset is
#                   strategy α (skin-toned _col capsules only, commit 2683b67).
HUMAN_MODEL="${HUMAN_MODEL:-smplh}"
SMPLH_MOTION="${SMPLH_MOTION:-amass}"
case "${HUMAN_MODEL}" in
  smplh|g1) ;;
  *) echo "ERROR: HUMAN_MODEL must be 'smplh' or 'g1', got '${HUMAN_MODEL}'" >&2; exit 1 ;;
esac
case "${SMPLH_MOTION}" in
  amass|procedural) ;;
  *) echo "ERROR: SMPLH_MOTION must be 'amass' or 'procedural', got '${SMPLH_MOTION}'" >&2; exit 1 ;;
esac

if [[ "${HUMAN_MODEL}" == "smplh" && "${SMPLH_MOTION}" == "amass" && -z "${AMASS_DATA_DIR:-}" ]]; then
  echo "ERROR: export AMASS_DATA_DIR before running smplh+amass curriculum (see CLAUDE.md)." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  STAGE0_FRAMES="${STAGE0_FRAMES:-2000}"
  STAGE1_FRAMES=0
  STAGE2_FRAMES=0
  WANDB=(wandb.use=false)
  # Synchronous CUDA so any device-side assert points at the real op/line
  # (smoke is short; the slowdown is irrelevant here).
  export CUDA_LAUNCH_BLOCKING=1
else
  STAGE0_FRAMES="${STAGE0_FRAMES:-30000}"
  STAGE1_FRAMES="${STAGE1_FRAMES:-30000}"
  STAGE2_FRAMES="${STAGE2_FRAMES:-40000}"
  WANDB=(wandb.use=true)
fi

# Auto RUN_TAG encodes human variant + stage budgets + launch stamp so exp_local/
# and W&B names are grep-friendly and unique per invocation.
_RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${RUN_TAG:-}" ]]; then
  if [[ "${HUMAN_MODEL}" == "g1" ]]; then
    _human_tag="g1"
  elif [[ "${SMPLH_MOTION}" == "procedural" ]]; then
    _human_tag="smplh_proc"
  else
    _human_tag="smplh_amass"
  fi
  if [[ "${SMOKE:-0}" == "1" ]]; then
    _frames_tag="smoke${STAGE0_FRAMES}"
  else
    _frames_tag="$(( STAGE0_FRAMES / 1000 ))k_$(( STAGE1_FRAMES / 1000 ))k_$(( STAGE2_FRAMES / 1000 ))k"
  fi
  RUN_TAG="base_${_human_tag}_${_frames_tag}_${_RUN_STAMP}"
elif [[ ! "${RUN_TAG}" =~ _[0-9]{8}_[0-9]{6}$ ]]; then
  RUN_TAG="${RUN_TAG}_${_RUN_STAMP}"
fi
OUTDIR="${OUTDIR:-${REPO_ROOT}/exp_local/cqn_as_base_curriculum/${RUN_TAG}}"
mkdir -p "${OUTDIR}"
echo "== RUN_TAG=${RUN_TAG} OUTDIR=${OUTDIR} =="

# Task selector (2026-05-25). The curriculum has been validated on
# `saucepan_to_hob` (the anchor task). Use `TASK=<name>` to point at any other
# yaml under cfgs/env/safety_bigym/ — e.g. `TASK=wall_cupboard_close` to
# confirm generality of the G1 operating config across tasks.
TASK="${TASK:-saucepan_to_hob}"

# Shared overrides — the reward/support fix (levers 1-3) + cadence/logging.
COMMON=(
  env=safety_bigym/saucepan_to_hob
  "env.human_model=${HUMAN_MODEL}"
  "env.smplh_motion=${SMPLH_MOTION}"
  bodyslam=oracle
  num_demos=36
  agent.v_min=-6.0
  agent.v_max=2.0
  agent.atoms=101
  save_snapshot=true
  save_video=true
)

# Curriculum resume checkpoint. Default matches commit 2683b67 (newest
# snapshot_*.pt by mtime — typically the final save at num_train_frames).
# Set CURRICULUM_SNAPSHOT=best to resume from peak eval success_rate
# (snapshot_best.pt / pick_best_snapshot.py) instead.
latest_snapshot() {
  ls -t "$1"/snapshot_*.pt 2>/dev/null | grep -v snapshot_best.pt | head -1
}

best_snapshot() {
  local stage_dir="$1"
  python scripts/pick_best_snapshot.py "${stage_dir}" 2>/dev/null || true
}

pick_stage_snapshot() {
  local stage_dir="$1"
  if [[ "${CURRICULUM_SNAPSHOT:-latest}" == "best" ]]; then
    best_snapshot "${stage_dir}"
  else
    latest_snapshot "${stage_dir}"
  fi
}

run_stage() {
  local name="$1" disruption="$2" frames="$3" resume_from="${4:-}"
  local stage_dir="${OUTDIR}/${name}"
  if [[ "${frames}" -le 0 ]]; then
    echo "== skip ${name} (frames=${frames}) =="
    return
  fi
  echo "== ${name}: disruption=${disruption} frames=${frames} -> ${stage_dir} =="
  local extra=()
  [[ -n "${resume_from}" ]] && extra+=("+snapshot_path=${resume_from}")
  # Thesis run-tagging scheme (docs/safety_metrics.md). Hydra's override
  # grammar reserves `=` and `,` as syntax, so we use `:` as the key/value
  # separator inside tag strings (W&B accepts any string as a tag).
  #   tags=[stage<n>, method:<unconstrained|lagrangian|hybrid>,
  #         task:<name>, human:<smplh|g1>]
  # `name` is one of `stage0_idle`, `stage1_easy`, `stage2_full` so the
  # leading prefix becomes the stage tag. METHOD defaults to
  # `unconstrained`; the Lagrangian launcher overrides this.
  local stage_tag="${name%%_*}"  # stage0_idle -> stage0
  local method_tag="${METHOD:-unconstrained}"
  local task_tag="${TASK_TAG:-saucepan_to_hob}"
  local wb_tags="+wandb.tags=[${stage_tag},method:${method_tag},task:${task_tag},human:${HUMAN_MODEL}]"
  python train_cqn_as.py \
    "${COMMON[@]}" "${WANDB[@]}" \
    "disruption=${disruption}" \
    num_train_frames="${frames}" \
    "hydra.run.dir=${stage_dir}" \
    "wandb.name=${RUN_TAG}_${name}" \
    "${wb_tags}" \
    "${extra[@]}"
}

# --- Resume-only path: rerun stage 2 from the newest available snapshot ---
if [[ "${RESUME_STAGE2:-0}" == "1" ]]; then
  if [[ -z "${RESUME_DIR:-}" ]]; then
    echo "ERROR: RESUME_STAGE2=1 requires RESUME_DIR=<prior run dir>." >&2
    exit 1
  fi
  OUTDIR="${RESUME_DIR}"
  RUN_TAG="${RUN_TAG:-$(basename "${OUTDIR}")}"
  # Explicit snapshot wins; else best from a prior stage-2 dir (crash resume),
  # else best from stage 1.
  SNAP="${RESUME_SNAPSHOT:-}"
  if [[ -z "${SNAP}" ]]; then
    for d in "${OUTDIR}"/stage2_full*; do
      [[ -d "${d}" ]] || continue
      SNAP="$(pick_stage_snapshot "${d}")"
      [[ -n "${SNAP}" ]] && break
    done
  fi
  [[ -z "${SNAP}" ]] && SNAP="$(pick_stage_snapshot "${OUTDIR}/stage1_easy")"
  if [[ -z "${SNAP}" ]]; then
    echo "ERROR: no snapshot found under ${OUTDIR} (stage2_full*/ or stage1_easy/)." >&2
    exit 1
  fi
  echo "== resuming stage 2 from: ${SNAP} =="
  run_stage stage2_full_resume coworker_train "${STAGE2_FRAMES}" "${SNAP}"
  echo "== stage 2 resume complete. snapshots under ${OUTDIR}/stage2_full_resume =="
  exit 0
fi

# Stage 0: human present but idle/distant (no interference; keeps obs width and
# model architecture identical to stages 1-2 so snapshots resume cleanly). This
# is the GATE — episode_reward must climb >0 on some episodes with returns inside
# [-6, 2]. If it fails, the problem is demos/CQN-AS, not safety: stop and reassess.
run_stage stage0_idle coworker_idle "${STAGE0_FRAMES}"
SNAP0="$(pick_stage_snapshot "${OUTDIR}/stage0_idle")"
if [[ -z "${SNAP0}" ]]; then
  echo "ERROR: no snapshot found under ${OUTDIR}/stage0_idle" >&2
  exit 1
fi
echo "== stage 1 resumes from stage-0 snapshot (${CURRICULUM_SNAPSHOT:-latest}): ${SNAP0} =="

run_stage stage1_easy coworker_easy "${STAGE1_FRAMES}" "${SNAP0}"
SNAP1="$(pick_stage_snapshot "${OUTDIR}/stage1_easy")"
if [[ -z "${SNAP1}" ]]; then
  echo "ERROR: no snapshot found under ${OUTDIR}/stage1_easy" >&2
  exit 1
fi
echo "== stage 2 resumes from stage-1 snapshot (${CURRICULUM_SNAPSHOT:-latest}): ${SNAP1} =="

run_stage stage2_full coworker_train "${STAGE2_FRAMES}" "${SNAP1}"

echo "== curriculum complete. snapshots under ${OUTDIR} =="

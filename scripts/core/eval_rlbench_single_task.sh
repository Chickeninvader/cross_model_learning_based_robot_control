#!/usr/bin/env bash
set -euo pipefail

# Legacy-compatible single-task RLBench batch evaluation.
# Evaluates one task across policy/state combinations and writes summary at the end.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/init_script_logging.sh
source "${SCRIPT_DIR}/../lib/init_script_logging.sh"

TASK="put_rubbish_in_bin"
VARIATION=0
VARIATION_LIST=""
RUNS=10
SEED=0
MAX_STEPS=200
RENDERER="opengl3"
TASK_DESCRIPTION=""

CHECKPOINT_ROOT="${WORKSPACE_ROOT}/output/lerobot"
DATASET_PARENT="${WORKSPACE_ROOT}/datasets/lerobot"
RLBENCH_ROOT="${WORKSPACE_ROOT}/datasets/rlbench"
SAVE_ROOT=""
SAVE_ROOT_SET=0
ROBOT_SETUP="panda"
WITH_CONTEXT_PROMPT=0

POLICIES="smolvla,groot"
STATES="eef,joint"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE=""
DRY_RUN=0
SUMMARIZE_ONLY=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --task NAME                Task name (default: ${TASK})
  --variation N              Base RLBench variation (default: ${VARIATION})
  --variation_list CSV       Explicit variation list, e.g. 0,1,2,3
  --runs N                   Number of (variation,seed) pairs (default: ${RUNS})
  --seed N                   Base seed; pair i uses seed=(seed+i)
  --max_steps N              Max steps per instruction (default: ${MAX_STEPS})
  --renderer NAME            opengl|opengl3 (default: ${RENDERER})
  --task_description TEXT    Optional task description override
  --checkpoint_root PATH     Root of training outputs (default: ${CHECKPOINT_ROOT})
  --dataset_parent PATH      Parent of dataset roots (default: ${DATASET_PARENT})
  --rlbench_root PATH        RLBench scene-graph templates (default: ${RLBENCH_ROOT})
  --save_root PATH           Output root (default: output/rlbench_eval/<task>[_with_context_prompt])
  --with_context_prompt      Use context prompt instructions and save root suffix
  --robot_setup NAME         RLBench robot: panda,jaco,mico,sawyer,ur5 (default: ${ROBOT_SETUP})
  --policies CSV             Policies (default: ${POLICIES})
  --states CSV               States (default: ${STATES})
  --python BIN               Python executable (default: ${PYTHON_BIN})
  --device DEVICE            Optional torch device (e.g. cuda:0)
  --summarize_only           Aggregate existing outputs only; skip evaluation runs
  --dry_run                  Print resolved commands only
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --variation) VARIATION="$2"; shift 2 ;;
    --variation_list) VARIATION_LIST="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --renderer) RENDERER="$2"; shift 2 ;;
    --task_description) TASK_DESCRIPTION="$2"; shift 2 ;;
    --checkpoint_root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --dataset_parent) DATASET_PARENT="$2"; shift 2 ;;
    --rlbench_root) RLBENCH_ROOT="$2"; shift 2 ;;
    --save_root) SAVE_ROOT="$2"; SAVE_ROOT_SET=1; shift 2 ;;
    --with_context_prompt) WITH_CONTEXT_PROMPT=1; shift ;;
    --robot_setup) ROBOT_SETUP="$2"; shift 2 ;;
    --policies) POLICIES="$2"; shift 2 ;;
    --states) STATES="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --summarize_only) SUMMARIZE_ONLY=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${SAVE_ROOT_SET}" -eq 0 ]]; then
  if [[ "${WITH_CONTEXT_PROMPT}" -eq 1 ]]; then
    SAVE_ROOT="${WORKSPACE_ROOT}/output/rlbench_eval/${TASK}_with_context_prompt"
  else
    SAVE_ROOT="${WORKSPACE_ROOT}/output/rlbench_eval/${TASK}"
  fi
fi
mkdir -p "${SAVE_ROOT}"

ROBOT_SETUP="${ROBOT_SETUP,,}"
case "${ROBOT_SETUP}" in
  franka|franka_panda) ROBOT_SETUP="panda" ;;
esac
case "${ROBOT_SETUP}" in
  panda|jaco|mico|sawyer|ur5) ;;
  *)
    echo "[ERROR] Unsupported --robot_setup='${ROBOT_SETUP}'. Use: panda, jaco, mico, sawyer, ur5." >&2
    exit 1
    ;;
esac

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [[ "${RUNS}" -le 0 ]]; then
  echo "[ERROR] --runs must be a positive integer, got: ${RUNS}" >&2
  exit 1
fi

action_mode_for_state() {
  local state="$1"
  if [[ "${state}" == "eef" ]]; then
    echo "ee_planning"
  elif [[ "${state}" == "joint" ]]; then
    echo "joint_velocity"
  else
    echo "[ERROR] Unknown state: ${state}" >&2
    return 1
  fi
}

find_latest_training_dir() {
  local policy="$1"
  local state="$2"
  local best_path=""
  local best_ts=""

  for path in "${CHECKPOINT_ROOT}"/*; do
    [[ -d "${path}" ]] || continue
    local base base_lc ts
    base="$(basename "${path}")"
    base_lc="${base,,}"
    [[ "${base_lc}" == *"${policy}"* ]] || continue
    [[ "${base_lc}" == *"${TASK}"* ]] || continue
    [[ "${base_lc}" == *"${state}"* ]] || continue
    ts=""
    if [[ "${base}" =~ ([0-9]{8}_[0-9]{6}) ]]; then
      ts="${BASH_REMATCH[1]}"
    fi
    if [[ -z "${best_path}" ]]; then
      best_path="${path}"
      best_ts="${ts}"
      continue
    fi
    if [[ -n "${ts}" && ( -z "${best_ts}" || "${ts}" > "${best_ts}" ) ]]; then
      best_path="${path}"
      best_ts="${ts}"
    elif [[ -z "${ts}" && -z "${best_ts}" && "${path}" > "${best_path}" ]]; then
      best_path="${path}"
    fi
  done

  [[ -n "${best_path}" ]] || return 1
  echo "${best_path}"
}

infer_trial_rlbench_root_from_dataset_parent() {
  local base suffix candidate
  base="$(basename "${DATASET_PARENT}")"
  suffix=""
  if [[ "${base}" =~ ^lerobot_trial_([0-9]+)$ ]]; then
    suffix="${BASH_REMATCH[1]}"
  elif [[ "${base}" =~ ^lerobot_trial([0-9]+)$ ]]; then
    suffix="${BASH_REMATCH[1]}"
  fi
  [[ -n "${suffix}" ]] || return 1
  candidate="${WORKSPACE_ROOT}/datasets/rlbench_trial_${suffix}"
  [[ -d "${candidate}" ]] || return 1
  echo "${candidate}"
}

# CoppeliaSim uses Qt (xcb). RLBench headless=True still launches the sim GUI stack.
run_with_display_if_needed() {
  if [[ -n "${DISPLAY:-}" ]]; then
    "$@"
    return $?
  fi
  if command -v xvfb-run >/dev/null 2>&1; then
    echo "[eval_rlbench_single_task] DISPLAY unset; using xvfb-run for CoppeliaSim/Qt" >&2
    xvfb-run -a "$@"
    return $?
  fi
  echo "[ERROR] DISPLAY is not set and xvfb-run was not found. CoppeliaSim/Qt needs an X server." >&2
  echo "  Install xvfb (e.g. apt install xvfb) or run: Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99" >&2
  echo "  See TROUBLESHOOTING.md (RLBench headless rendering)." >&2
  return 1
}

template_path="${RLBENCH_ROOT}/${TASK}/${TASK}_relationship_template.json"
if [[ ! -f "${template_path}" ]]; then
  if inferred_rlbench_root="$(infer_trial_rlbench_root_from_dataset_parent)"; then
    inferred_template="${inferred_rlbench_root}/${TASK}/${TASK}_relationship_template.json"
    if [[ -f "${inferred_template}" ]]; then
      echo "[eval_rlbench_single_task] relationship template missing at ${template_path}" >&2
      echo "[eval_rlbench_single_task] auto-switching rlbench_root to ${inferred_rlbench_root}" >&2
      RLBENCH_ROOT="${inferred_rlbench_root}"
    fi
  fi
fi

declare -a VARIATIONS=()
if [[ -n "${VARIATION_LIST}" ]]; then
  IFS=',' read -r -a VARIATIONS <<< "${VARIATION_LIST}"
  for idx in "${!VARIATIONS[@]}"; do
    VARIATIONS[idx]="$(echo "${VARIATIONS[idx]}" | xargs)"
    if ! [[ "${VARIATIONS[idx]}" =~ ^-?[0-9]+$ ]]; then
      echo "[ERROR] Invalid variation in --variation_list: ${VARIATIONS[idx]}" >&2
      exit 1
    fi
  done
  RUNS="${#VARIATIONS[@]}"
else
  if ! [[ "${VARIATION}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] --variation must be an integer, got: ${VARIATION}" >&2
    exit 1
  fi
  for ((i=0; i<RUNS; i++)); do
    VARIATIONS+=("$((VARIATION + i))")
  done
fi

IFS=',' read -r -a POLICIES_ARR <<< "${POLICIES}"
IFS=',' read -r -a STATES_ARR <<< "${STATES}"

echo "[eval_rlbench_single_task] task=${TASK} robot_setup=${ROBOT_SETUP} save_root=${SAVE_ROOT} runs=${RUNS} with_context_prompt=${WITH_CONTEXT_PROMPT}"

if [[ "${SUMMARIZE_ONLY}" -eq 0 ]]; then
  for policy in "${POLICIES_ARR[@]}"; do
    policy="$(echo "${policy}" | xargs)"
    [[ -n "${policy}" ]] || continue
    for state in "${STATES_ARR[@]}"; do
      state="$(echo "${state}" | xargs)"
      [[ -n "${state}" ]] || continue
      action_mode="$(action_mode_for_state "${state}")" || continue
      dataset_root="${DATASET_PARENT}/${TASK}_${state}"

      if ! checkpoint_dir="$(find_latest_training_dir "${policy}" "${state}")"; then
        echo "[WARN] No checkpoint for policy=${policy} state=${state} task=${TASK}; skipping." >&2
        continue
      fi
      if [[ ! -d "${dataset_root}" ]]; then
        echo "[WARN] Dataset root not found: ${dataset_root}; skipping." >&2
        continue
      fi

      if [[ "${ROBOT_SETUP}" == "panda" ]]; then
        save_path="${SAVE_ROOT}/${policy}_${state}"
      else
        save_path="${SAVE_ROOT}/${policy}_${state}_${ROBOT_SETUP}"
      fi
      cmd=(
        "${PYTHON_BIN}" "${WORKSPACE_ROOT}/src/inference/rlbench/eval.py"
        --task "${TASK}"
        --variation "${VARIATIONS[0]}"
        --runs "${RUNS}"
        --seed "${SEED}"
        --max_steps "${MAX_STEPS}"
        --action_mode "${action_mode}"
        --robot_setup "${ROBOT_SETUP}"
        --renderer "${RENDERER}"
        --checkpoint "${checkpoint_dir}"
        --dataset_root "${dataset_root}"
        --save_path "${save_path}"
        --rlbench_root "${RLBENCH_ROOT}"
      )
      if [[ -n "${TASK_DESCRIPTION}" ]]; then
        cmd+=(--task_description "${TASK_DESCRIPTION}")
      fi
      if [[ -n "${DEVICE}" ]]; then
        cmd+=(--device "${DEVICE}")
      fi
      if [[ "${WITH_CONTEXT_PROMPT}" -eq 1 ]]; then
        cmd+=(--with_context_prompt)
      fi
      echo "Command: ${cmd[*]}"
      if [[ "${DRY_RUN}" -eq 0 ]]; then
        run_with_display_if_needed "${cmd[@]}"
      fi
    done
  done
fi

# Always aggregate at the end so metrics are available immediately.
summary_cmd=(
  "${PYTHON_BIN}" "${WORKSPACE_ROOT}/src/inference/rlbench/eval.py"
  --task "${TASK}"
  --aggregate_only
  --aggregate_root "${SAVE_ROOT}"
)
echo "Summary command: ${summary_cmd[*]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  "${summary_cmd[@]}"
fi

echo
echo "Single-task evaluation complete."
echo "Outputs and summaries: ${SAVE_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

# All-task model evaluation across multiple RLBench tasks.
# Uses one all-task checkpoint per (policy,state), iterates tasks, and writes summaries immediately.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

TASKS=""
TASKS_FILE=""
VARIATION=0
VARIATION_LIST=""
RUNS=10
SEED=0
MAX_STEPS=200
RENDERER="opengl3"

POLICIES="smolvla,groot"
STATES="eef,joint"

CHECKPOINT_ROOT="${WORKSPACE_ROOT}/output/lerobot"
CHECKPOINT_TAGS="all_task,all_tasks,alltasks,multi_task,multitask"

DATASET_PARENT="${WORKSPACE_ROOT}/datasets/lerobot"
DATASET_MODE="all"   # all|task
ALL_DATASET_PREFIX="all_task"

RLBENCH_ROOT="${WORKSPACE_ROOT}/datasets/rlbench"
SAVE_ROOT="${WORKSPACE_ROOT}/output/rlbench_eval/all_tasks"
DEFAULT_TASK_DESCRIPTION=""

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE=""
DRY_RUN=0
SUMMARIZE_ONLY=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --tasks CSV               Tasks to evaluate, e.g. put_rubbish_in_bin,close_jar
  --tasks_file PATH         File with one task per line (supports # comments)
  --variation N             Base RLBench variation (default: ${VARIATION})
  --variation_list CSV      Explicit variation list, e.g. 0,1,2,3
  --runs N                  Number of (variation,seed) pairs (default: ${RUNS})
  --seed N                  Base seed; pair i uses seed=(seed+i)
  --max_steps N             Max steps per instruction (default: ${MAX_STEPS})
  --renderer NAME           opengl|opengl3 (default: ${RENDERER})
  --policies CSV            Policies (default: ${POLICIES})
  --states CSV              States (default: ${STATES})
  --checkpoint_root PATH    Root of training outputs (default: ${CHECKPOINT_ROOT})
  --checkpoint_tags CSV     Tags to identify all-task checkpoints
  --dataset_parent PATH     Parent of dataset roots (default: ${DATASET_PARENT})
  --dataset_mode MODE       all|task (default: ${DATASET_MODE})
  --all_dataset_prefix STR  Prefix for all-task dataset roots (default: ${ALL_DATASET_PREFIX})
  --rlbench_root PATH       RLBench templates root (default: ${RLBENCH_ROOT})
  --save_root PATH          Output root (default: ${SAVE_ROOT})
  --default_task_description TEXT
                            Optional explicit override passed to eval.py
  --python BIN              Python executable (default: ${PYTHON_BIN})
  --device DEVICE           Optional torch device (e.g. cuda:0)
  --summarize_only          Skip evaluation and only generate per-task + global summaries
  --dry_run                 Print commands only
  -h, --help                Show this help

Examples:
  $(basename "$0") --tasks put_rubbish_in_bin,close_jar --runs 5
  $(basename "$0") --tasks_file scripts/tasks_eval.txt --dataset_mode task
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks) TASKS="$2"; shift 2 ;;
    --tasks_file) TASKS_FILE="$2"; shift 2 ;;
    --variation) VARIATION="$2"; shift 2 ;;
    --variation_list) VARIATION_LIST="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --renderer) RENDERER="$2"; shift 2 ;;
    --policies) POLICIES="$2"; shift 2 ;;
    --states) STATES="$2"; shift 2 ;;
    --checkpoint_root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --checkpoint_tags) CHECKPOINT_TAGS="$2"; shift 2 ;;
    --dataset_parent) DATASET_PARENT="$2"; shift 2 ;;
    --dataset_mode) DATASET_MODE="$2"; shift 2 ;;
    --all_dataset_prefix) ALL_DATASET_PREFIX="$2"; shift 2 ;;
    --rlbench_root) RLBENCH_ROOT="$2"; shift 2 ;;
    --save_root) SAVE_ROOT="$2"; shift 2 ;;
    --default_task_description) DEFAULT_TASK_DESCRIPTION="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --summarize_only) SUMMARIZE_ONLY=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [[ "${RUNS}" -le 0 ]]; then
  echo "[ERROR] --runs must be a positive integer, got: ${RUNS}" >&2
  exit 1
fi
if [[ "${DATASET_MODE}" != "all" && "${DATASET_MODE}" != "task" ]]; then
  echo "[ERROR] --dataset_mode must be one of: all, task" >&2
  exit 1
fi

declare -a TASKS_ARR=()
if [[ -n "${TASKS}" ]]; then
  IFS=',' read -r -a TASKS_ARR <<< "${TASKS}"
elif [[ -n "${TASKS_FILE}" ]]; then
  while IFS= read -r line; do
    line="$(echo "${line}" | sed 's/#.*$//' | xargs)"
    [[ -n "${line}" ]] && TASKS_ARR+=("${line}")
  done < "${TASKS_FILE}"
else
  for d in "${RLBENCH_ROOT}"/*; do
    [[ -d "${d}" ]] || continue
    TASKS_ARR+=("$(basename "${d}")")
  done
fi

if [[ "${#TASKS_ARR[@]}" -eq 0 ]]; then
  echo "[ERROR] No tasks resolved. Provide --tasks or --tasks_file." >&2
  exit 1
fi

for idx in "${!TASKS_ARR[@]}"; do
  TASKS_ARR[idx]="$(echo "${TASKS_ARR[idx]}" | xargs)"
done

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

find_latest_all_task_training_dir() {
  local policy="$1"
  local state="$2"
  local best_path=""
  local best_ts=""

  IFS=',' read -r -a tags <<< "${CHECKPOINT_TAGS}"
  for path in "${CHECKPOINT_ROOT}"/*; do
    [[ -d "${path}" ]] || continue
    local base base_lc ts
    base="$(basename "${path}")"
    base_lc="${base,,}"
    [[ "${base_lc}" == *"${policy}"* ]] || continue
    [[ "${base_lc}" == *"${state}"* ]] || continue
    local has_tag=0
    for tag in "${tags[@]}"; do
      tag="$(echo "${tag}" | xargs | tr '[:upper:]' '[:lower:]')"
      [[ -n "${tag}" ]] || continue
      if [[ "${base_lc}" == *"${tag}"* ]]; then
        has_tag=1
        break
      fi
    done
    [[ "${has_tag}" -eq 1 ]] || continue

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

dataset_root_for() {
  local task="$1"
  local state="$2"
  if [[ "${DATASET_MODE}" == "all" ]]; then
    echo "${DATASET_PARENT}/${ALL_DATASET_PREFIX}_${state}"
  else
    echo "${DATASET_PARENT}/${task}_${state}"
  fi
}

mkdir -p "${SAVE_ROOT}"

IFS=',' read -r -a POLICIES_ARR <<< "${POLICIES}"
IFS=',' read -r -a STATES_ARR <<< "${STATES}"

TASKS_CSV="$(IFS=','; echo "${TASKS_ARR[*]}")"

if [[ "${SUMMARIZE_ONLY}" -eq 0 ]]; then
  for policy in "${POLICIES_ARR[@]}"; do
    policy="$(echo "${policy}" | xargs)"
    [[ -n "${policy}" ]] || continue
    for state in "${STATES_ARR[@]}"; do
      state="$(echo "${state}" | xargs)"
      [[ -n "${state}" ]] || continue
      action_mode="$(action_mode_for_state "${state}")" || continue

      # For multi-task checkpoint we use the first task's dataset root as
      # reference (the policy was trained on the joint dataset).
      first_task="${TASKS_ARR[0]}"
      dataset_root="$(dataset_root_for "${first_task}" "${state}")"

      if ! checkpoint_dir="$(find_latest_all_task_training_dir "${policy}" "${state}")"; then
        echo "[WARN] No all-task checkpoint for policy=${policy} state=${state}; skipping." >&2
        continue
      fi
      if [[ ! -d "${dataset_root}" ]]; then
        echo "[WARN] Dataset root not found: ${dataset_root}; skipping." >&2
        continue
      fi

      save_path="${SAVE_ROOT}/${policy}_${state}"

      cmd=(
        "${PYTHON_BIN}" "${WORKSPACE_ROOT}/src/inference/rlbench/eval.py"
        --tasks "${TASKS_CSV}"
        --variation "${VARIATIONS[0]}"
        --runs "${RUNS}"
        --seed "${SEED}"
        --max_steps "${MAX_STEPS}"
        --action_mode "${action_mode}"
        --renderer "${RENDERER}"
        --checkpoint "${checkpoint_dir}"
        --dataset_root "${dataset_root}"
        --save_path "${save_path}"
        --rlbench_root "${RLBENCH_ROOT}"
      )
      if [[ -n "${DEFAULT_TASK_DESCRIPTION}" ]]; then
        cmd+=(--task_description "${DEFAULT_TASK_DESCRIPTION}")
      fi
      if [[ -n "${DEVICE}" ]]; then
        cmd+=(--device "${DEVICE}")
      fi
      echo "Command: ${cmd[*]}"
      if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${cmd[@]}"
      fi
    done
  done
fi

# Global summary across all tasks (requires eval.py multi-task aggregate support).
global_summary_cmd=(
  "${PYTHON_BIN}" "${WORKSPACE_ROOT}/src/inference/rlbench/eval.py"
  --task "all_tasks"
  --aggregate_root "${SAVE_ROOT}"
)
echo "Global summary command: ${global_summary_cmd[*]}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  "${global_summary_cmd[@]}"
fi

echo
echo "All-task evaluation complete."
echo "Outputs and summaries: ${SAVE_ROOT}"

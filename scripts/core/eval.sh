#!/usr/bin/env bash
set -euo pipefail

# RLBench evaluation: single-task (default) or all-task (--all_tasks).
# See README.md "RLBench Evaluation".

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/init_script_logging.sh
source "${SCRIPT_DIR}/../lib/init_script_logging.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--all_tasks] [options]

  Default: single-task evaluation (one RLBench task, per policy/state checkpoints).
  With --all_tasks: multi-task evaluation (one multi-task checkpoint per policy/state).

Global options (both modes):
  -h, --help                 This help
  --variation N              Base RLBench variation (default: 0)
  --variation_list CSV       Explicit variation list (sets run count)
  --runs N                   Runs / variation count (default: 10)
  --seed N                   Base seed (default: 0)
  --max_steps N              Max steps per instruction (default: 200)
  --renderer NAME            opengl|opengl3 (default: opengl3)
  --policies CSV             Policies (default: smolvla,groot)
  --states CSV               States eef,joint (default: eef,joint)
  --checkpoint_root PATH     Training outputs root (default: output/lerobot)
  --dataset_parent PATH      Parent of dataset roots (default: datasets/lerobot)
  --rlbench_root PATH        RLBench templates (default: datasets/rlbench)
  --save_root PATH           Output root (defaults differ by mode)
  --with_context_prompt      Context-prompt instructions + default save root suffix
  --robot_setup NAME         panda,jaco,mico,sawyer,ur5 (default: panda)
  --python BIN               Python executable (default: python or PYTHON_BIN)
  --device DEVICE            Optional torch device (e.g. cuda:0)
  --summarize_only           Skip rollouts; only aggregation / summaries
  --dry_run                  Print commands only
  --single_task              Force single-task mode (default)

Single-task mode options:
  --task NAME                Task name (default: put_rubbish_in_bin)
  --task_description TEXT   Optional override passed to eval.py

All-task mode (--all_tasks):
  --tasks CSV               Tasks (comma-separated)
  --tasks_file PATH         One task per line (# comments ok)
  --checkpoint_tags CSV     Substrings for multi-task checkpoints
  --dataset_mode MODE       all|task (default: all)
  --all_dataset_prefix STR  Prefix for all-task dataset dirs (default: all_task)
  --default_task_description TEXT  Override passed to eval.py
  --skip_completed          Reuse task dirs when summary.json matches settings

If --all_tasks and neither --tasks nor --tasks_file is given, tasks are all
subdirectories of --rlbench_root.

Examples:
  $(basename "$0") --task put_rubbish_in_bin --dataset_parent datasets/lerobot_trial_3 \\
    --rlbench_root datasets/rlbench_trial_3
  $(basename "$0") --all_tasks --tasks t1,t2 --dataset_mode all \\
    --dataset_parent datasets/lerobot_trial_3 --checkpoint_root output/lerobot/run1 \\
    --skip_completed
EOF
}

for _arg in "$@"; do
  if [[ "${_arg}" == "-h" || "${_arg}" == "--help" ]]; then
    usage
    exit 0
  fi
done

MODE="single"
for _arg in "$@"; do
  if [[ "${_arg}" == "--all_tasks" ]]; then
    MODE="all"
    break
  fi
  if [[ "${_arg}" == "--single_task" ]]; then
    MODE="single"
    break
  fi
done

# --- shared defaults ---
VARIATION=0
VARIATION_LIST=""
RUNS=10
SEED=0
MAX_STEPS=200
RENDERER="opengl3"
POLICIES="smolvla,groot"
STATES="eef,joint"
CHECKPOINT_ROOT="${WORKSPACE_ROOT}/output/lerobot"
DATASET_PARENT="${WORKSPACE_ROOT}/datasets/lerobot"
RLBENCH_ROOT="${WORKSPACE_ROOT}/datasets/rlbench"
SAVE_ROOT=""
SAVE_ROOT_SET=0
ROBOT_SETUP="panda"
WITH_CONTEXT_PROMPT=0
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE=""
DRY_RUN=0
SUMMARIZE_ONLY=0

# single-task
TASK="put_rubbish_in_bin"
TASK_DESCRIPTION=""

# all-task
TASKS=""
TASKS_FILE=""
CHECKPOINT_TAGS="all_task,all_tasks,alltasks,multi_task,multitask"
DATASET_MODE="all"
ALL_DATASET_PREFIX="all_task"
DEFAULT_TASK_DESCRIPTION=""
SKIP_COMPLETED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all_tasks|--single_task) shift ;;
    --task) TASK="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --tasks_file) TASKS_FILE="$2"; shift 2 ;;
    --variation) VARIATION="$2"; shift 2 ;;
    --variation_list) VARIATION_LIST="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --renderer) RENDERER="$2"; shift 2 ;;
    --task_description) TASK_DESCRIPTION="$2"; shift 2 ;;
    --checkpoint_root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --checkpoint_tags) CHECKPOINT_TAGS="$2"; shift 2 ;;
    --dataset_parent) DATASET_PARENT="$2"; shift 2 ;;
    --dataset_mode) DATASET_MODE="$2"; shift 2 ;;
    --all_dataset_prefix) ALL_DATASET_PREFIX="$2"; shift 2 ;;
    --rlbench_root) RLBENCH_ROOT="$2"; shift 2 ;;
    --save_root) SAVE_ROOT="$2"; SAVE_ROOT_SET=1; shift 2 ;;
    --with_context_prompt) WITH_CONTEXT_PROMPT=1; shift ;;
    --robot_setup) ROBOT_SETUP="$2"; shift 2 ;;
    --default_task_description) DEFAULT_TASK_DESCRIPTION="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --summarize_only) SUMMARIZE_ONLY=1; shift ;;
    --skip_completed) SKIP_COMPLETED=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    --policies) POLICIES="$2"; shift 2 ;;
    --states) STATES="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [[ "${RUNS}" -le 0 ]]; then
  echo "[ERROR] --runs must be a positive integer, got: ${RUNS}" >&2
  exit 1
fi

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

run_with_display_if_needed() {
  if [[ -n "${DISPLAY:-}" ]]; then
    "$@"
    return $?
  fi
  if command -v xvfb-run >/dev/null 2>&1; then
    echo "[eval] DISPLAY unset; using xvfb-run for CoppeliaSim/Qt" >&2
    xvfb-run -a "$@"
    return $?
  fi
  echo "[ERROR] DISPLAY is not set and xvfb-run was not found. CoppeliaSim/Qt needs an X server." >&2
  echo "  Install xvfb (e.g. apt install xvfb) or run: Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99" >&2
  echo "  See TROUBLESHOOTING.md (RLBench headless rendering)." >&2
  return 1
}

# --- single-task: find checkpoint matching task name ---
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

# --- all-task: find checkpoint with multi-task tags ---
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

dataset_root_for_all_task() {
  local task="$1"
  local state="$2"
  if [[ "${DATASET_MODE}" == "all" ]]; then
    echo "${DATASET_PARENT}/${ALL_DATASET_PREFIX}_${state}"
  else
    echo "${DATASET_PARENT}/${task}_${state}"
  fi
}

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

if [[ "${MODE}" == "all" ]]; then
  if [[ -n "${TASK_DESCRIPTION}" && -z "${DEFAULT_TASK_DESCRIPTION}" ]]; then
    DEFAULT_TASK_DESCRIPTION="${TASK_DESCRIPTION}"
  fi
  if [[ "${DATASET_MODE}" != "all" && "${DATASET_MODE}" != "task" ]]; then
    echo "[ERROR] --dataset_mode must be one of: all, task" >&2
    exit 1
  fi

  if [[ "${SAVE_ROOT_SET}" -eq 0 ]]; then
    if [[ "${WITH_CONTEXT_PROMPT}" -eq 1 ]]; then
      SAVE_ROOT="${WORKSPACE_ROOT}/output/rlbench_eval/all_tasks_with_context_prompt"
    else
      SAVE_ROOT="${WORKSPACE_ROOT}/output/rlbench_eval/all_tasks"
    fi
  fi

  if [[ "${CHECKPOINT_ROOT}" != /* ]]; then
    CHECKPOINT_ROOT="${WORKSPACE_ROOT}/${CHECKPOINT_ROOT}"
  fi
  if [[ "${DATASET_PARENT}" != /* ]]; then
    DATASET_PARENT="${WORKSPACE_ROOT}/${DATASET_PARENT}"
  fi
  if [[ "${SAVE_ROOT}" != /* ]]; then
    SAVE_ROOT="${WORKSPACE_ROOT}/${SAVE_ROOT}"
  fi

  if [[ "${RLBENCH_ROOT}" != /* ]]; then
    if [[ -d "${WORKSPACE_ROOT}/${RLBENCH_ROOT}" ]]; then
      RLBENCH_ROOT="${WORKSPACE_ROOT}/${RLBENCH_ROOT}"
    elif [[ -d "${WORKSPACE_ROOT}/datasets/${RLBENCH_ROOT}" ]]; then
      RLBENCH_ROOT="${WORKSPACE_ROOT}/datasets/${RLBENCH_ROOT}"
    else
      RLBENCH_ROOT="${WORKSPACE_ROOT}/${RLBENCH_ROOT}"
    fi
  fi
  if [[ ! -d "${RLBENCH_ROOT}" ]]; then
    echo "[ERROR] --rlbench_root does not exist: ${RLBENCH_ROOT}" >&2
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
    echo "[ERROR] No tasks resolved. Provide --tasks or --tasks_file, or use an rlbench_root with subdirs." >&2
    exit 1
  fi

  for idx in "${!TASKS_ARR[@]}"; do
    TASKS_ARR[idx]="$(echo "${TASKS_ARR[idx]}" | xargs)"
  done

  TASKS_CSV="$(IFS=','; echo "${TASKS_ARR[*]}")"
  first_task="${TASKS_ARR[0]}"
  template_path="${RLBENCH_ROOT}/${first_task}/${first_task}_relationship_template.json"
  if [[ ! -f "${template_path}" ]]; then
    if inferred_rlbench_root="$(infer_trial_rlbench_root_from_dataset_parent)"; then
      inferred_template="${inferred_rlbench_root}/${first_task}/${first_task}_relationship_template.json"
      if [[ -f "${inferred_template}" ]]; then
        echo "[eval] relationship template missing at ${template_path}; auto-switching rlbench_root to ${inferred_rlbench_root}" >&2
        RLBENCH_ROOT="${inferred_rlbench_root}"
      fi
    fi
  fi

  mkdir -p "${SAVE_ROOT}"
  echo "[eval] mode=all_tasks robot_setup=${ROBOT_SETUP} save_root=${SAVE_ROOT} runs=${RUNS} tasks=${#TASKS_ARR[@]} with_context_prompt=${WITH_CONTEXT_PROMPT}"

  if [[ "${SUMMARIZE_ONLY}" -eq 0 ]]; then
    for policy in "${POLICIES_ARR[@]}"; do
      policy="$(echo "${policy}" | xargs)"
      [[ -n "${policy}" ]] || continue
      for state in "${STATES_ARR[@]}"; do
        state="$(echo "${state}" | xargs)"
        [[ -n "${state}" ]] || continue
        action_mode="$(action_mode_for_state "${state}")" || continue
        dataset_root="$(dataset_root_for_all_task "${first_task}" "${state}")"

        if ! checkpoint_dir="$(find_latest_all_task_training_dir "${policy}" "${state}")"; then
          echo "[WARN] No all-task checkpoint for policy=${policy} state=${state}; skipping." >&2
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
          --tasks "${TASKS_CSV}"
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
        if [[ -n "${DEFAULT_TASK_DESCRIPTION}" ]]; then
          cmd+=(--task_description "${DEFAULT_TASK_DESCRIPTION}")
        fi
        if [[ -n "${DEVICE}" ]]; then
          cmd+=(--device "${DEVICE}")
        fi
        if [[ "${WITH_CONTEXT_PROMPT}" -eq 1 ]]; then
          cmd+=(--with_context_prompt)
        fi
        if [[ "${SKIP_COMPLETED}" -eq 1 ]]; then
          cmd+=(--skip_completed)
        fi
        echo "Command: ${cmd[*]}"
        if [[ "${DRY_RUN}" -eq 0 ]]; then
          run_with_display_if_needed "${cmd[@]}"
        fi
      done
    done
  fi

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
  exit 0
fi

# ========== single-task mode ==========
if [[ -n "${TASKS}" || -n "${TASKS_FILE}" ]]; then
  echo "[ERROR] --tasks / --tasks_file are only valid with --all_tasks." >&2
  exit 2
fi

if [[ "${SAVE_ROOT_SET}" -eq 0 ]]; then
  if [[ "${WITH_CONTEXT_PROMPT}" -eq 1 ]]; then
    SAVE_ROOT="${WORKSPACE_ROOT}/output/rlbench_eval/${TASK}_with_context_prompt"
  else
    SAVE_ROOT="${WORKSPACE_ROOT}/output/rlbench_eval/${TASK}"
  fi
fi
mkdir -p "${SAVE_ROOT}"

if [[ "${CHECKPOINT_ROOT}" != /* ]]; then
  CHECKPOINT_ROOT="${WORKSPACE_ROOT}/${CHECKPOINT_ROOT}"
fi
if [[ "${DATASET_PARENT}" != /* ]]; then
  DATASET_PARENT="${WORKSPACE_ROOT}/${DATASET_PARENT}"
fi
if [[ "${SAVE_ROOT}" != /* ]]; then
  SAVE_ROOT="${WORKSPACE_ROOT}/${SAVE_ROOT}"
fi

if [[ "${RLBENCH_ROOT}" != /* ]]; then
  if [[ -d "${WORKSPACE_ROOT}/${RLBENCH_ROOT}" ]]; then
    RLBENCH_ROOT="${WORKSPACE_ROOT}/${RLBENCH_ROOT}"
  elif [[ -d "${WORKSPACE_ROOT}/datasets/${RLBENCH_ROOT}" ]]; then
    RLBENCH_ROOT="${WORKSPACE_ROOT}/datasets/${RLBENCH_ROOT}"
  else
    RLBENCH_ROOT="${WORKSPACE_ROOT}/${RLBENCH_ROOT}"
  fi
fi

template_path="${RLBENCH_ROOT}/${TASK}/${TASK}_relationship_template.json"
if [[ ! -f "${template_path}" ]]; then
  if inferred_rlbench_root="$(infer_trial_rlbench_root_from_dataset_parent)"; then
    inferred_template="${inferred_rlbench_root}/${TASK}/${TASK}_relationship_template.json"
    if [[ -f "${inferred_template}" ]]; then
      echo "[eval] relationship template missing at ${template_path}; auto-switching rlbench_root to ${inferred_rlbench_root}" >&2
      RLBENCH_ROOT="${inferred_rlbench_root}"
    fi
  fi
fi

echo "[eval] mode=single_task task=${TASK} robot_setup=${ROBOT_SETUP} save_root=${SAVE_ROOT} runs=${RUNS} with_context_prompt=${WITH_CONTEXT_PROMPT}"

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
      if [[ "${SKIP_COMPLETED}" -eq 1 ]]; then
        cmd+=(--skip_completed)
      fi
      echo "Command: ${cmd[*]}"
      if [[ "${DRY_RUN}" -eq 0 ]]; then
        run_with_display_if_needed "${cmd[@]}"
      fi
    done
  done
fi

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

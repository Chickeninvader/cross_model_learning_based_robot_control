#!/usr/bin/env bash
set -euo pipefail

# Batch evaluate put_rubbish_in_bin across:
#   - policy: groot, smolvla
#   - state : eef, joint
#
# Defaults:
#   runs=10  (10 variation/seed pairs)
#   eef   -> action_mode=ee_planning
#   joint -> action_mode=joint_velocity

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

TASK="put_rubbish_in_bin"
VARIATION=0
VARIATION_LIST=""
RUNS=10
SEED=0
MAX_STEPS=200
RENDERER="opengl3"
TASK_DESCRIPTION="Pick up paper. Release paper, then place paper in trash bin."

CHECKPOINT_ROOT="${WORKSPACE_ROOT}/output/lerobot"
DATASET_PARENT="${WORKSPACE_ROOT}/datasets/lerobot_without_prompt"
LEGACY_SAVE_ROOT="${WORKSPACE_ROOT}/output/rlbench_eval/batch"
SAVE_ROOT=""

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE=""
DRY_RUN=0
SUMMARIZE_ONLY=0
MIGRATE_LEGACY=1

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
  --task_description TEXT    Task description string
  --checkpoint_root PATH     Root of training outputs (default: ${CHECKPOINT_ROOT})
  --dataset_parent PATH      Parent of dataset roots (default: ${DATASET_PARENT})
  --save_root PATH           Parent for eval outputs (default: output/rlbench_eval/<task>)
  --python BIN               Python executable (default: ${PYTHON_BIN})
  --device DEVICE            Optional torch device (e.g. cuda:0)
  --summarize_only           Aggregate existing outputs only; skip evaluation runs
  --no_migrate_legacy        Do not auto-migrate legacy 'batch' output folder
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
    --save_root) SAVE_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --summarize_only) SUMMARIZE_ONLY=1; shift ;;
    --no_migrate_legacy) MIGRATE_LEGACY=0; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${SAVE_ROOT}" ]]; then
  SAVE_ROOT="${WORKSPACE_ROOT}/output/rlbench_eval/${TASK}"
fi

if [[ ! -d "${CHECKPOINT_ROOT}" && "${SUMMARIZE_ONLY}" -eq 0 ]]; then
  echo "[ERROR] checkpoint_root not found: ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

if [[ "${MIGRATE_LEGACY}" -eq 1 && "${SAVE_ROOT}" != "${LEGACY_SAVE_ROOT}" ]]; then
  if [[ -d "${LEGACY_SAVE_ROOT}" && ! -d "${SAVE_ROOT}" ]]; then
    echo "[INFO] Migrating legacy output folder:"
    echo "       ${LEGACY_SAVE_ROOT}"
    echo "    -> ${SAVE_ROOT}"
    mv "${LEGACY_SAVE_ROOT}" "${SAVE_ROOT}"
  elif [[ -d "${LEGACY_SAVE_ROOT}" && -d "${SAVE_ROOT}" ]]; then
    echo "[WARN] Legacy folder exists but target already exists; skipping migration."
    echo "       Legacy: ${LEGACY_SAVE_ROOT}"
    echo "       Target: ${SAVE_ROOT}"
  fi
fi

mkdir -p "${SAVE_ROOT}"

if [[ "${SUMMARIZE_ONLY}" -eq 1 ]]; then
  cmd=(
    "${PYTHON_BIN}" "${WORKSPACE_ROOT}/src/inference/rlbench/eval_put_rubbish_in_bin.py"
    --task "${TASK}"
    --aggregate_only
    --aggregate_root "${SAVE_ROOT}"
  )
  echo "Summary command: ${cmd[*]}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${cmd[@]}"
  fi
  echo
  echo "Detailed summary generated under: ${SAVE_ROOT}"
  exit 0
fi

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [[ "${RUNS}" -le 0 ]]; then
  echo "[ERROR] --runs must be a positive integer, got: ${RUNS}" >&2
  exit 1
fi

declare -a VARIATIONS=()
if [[ -n "${VARIATION_LIST}" ]]; then
  IFS=',' read -r -a VARIATIONS <<< "${VARIATION_LIST}"
  # Trim whitespace from each variation token.
  for idx in "${!VARIATIONS[@]}"; do
    VARIATIONS[idx]="$(echo "${VARIATIONS[idx]}" | xargs)"
    if ! [[ "${VARIATIONS[idx]}" =~ ^-?[0-9]+$ ]]; then
      echo "[ERROR] Invalid variation in --variation_list: ${VARIATIONS[idx]}" >&2
      exit 1
    fi
  done

  if [[ "${#VARIATIONS[@]}" -eq 0 ]]; then
    echo "[ERROR] --variation_list is empty" >&2
    exit 1
  fi

  # When explicit variation list is provided, it defines number of evaluation pairs.
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

find_latest_training_dir() {
  local policy="$1"
  local state="$2"

  local best_path=""
  local best_ts=""

  while IFS= read -r path; do
    local base ts
    base="$(basename "${path}")"
    ts="$(echo "${base}" | grep -oE '[0-9]{8}_[0-9]{6}' | tail -n1 || true)"

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
  done < <(
    find "${CHECKPOINT_ROOT}" -maxdepth 1 -mindepth 1 -type d \
      -iname "*${policy}*" \
      -iname "*${TASK}*" \
      -iname "*${state}*" \
      | sort
  )

  if [[ -z "${best_path}" ]]; then
    echo "[ERROR] No matching training dir for policy=${policy}, state=${state}, task=${TASK}" >&2
    return 1
  fi

  echo "${best_path}"
}

for policy in smolvla groot; do
  echo
  echo "############################################################"
  echo "Policy block: ${policy} (all ${RUNS} variation/seed pairs first)"

  for state in eef joint; do
    action_mode="$(action_mode_for_state "${state}")"
    checkpoint_dir="$(find_latest_training_dir "${policy}" "${state}")"
    dataset_root="${DATASET_PARENT}/${TASK}_${state}"

    if [[ ! -d "${dataset_root}" ]]; then
      echo "[ERROR] Dataset root not found: ${dataset_root}" >&2
      exit 1
    fi

    echo
    echo "------------------------------"
    echo "state       : ${state}"
    echo "action_mode : ${action_mode}"
    echo "checkpoint  : ${checkpoint_dir}"
    echo "dataset_root: ${dataset_root}"

    for ((pair_idx=0; pair_idx<RUNS; pair_idx++)); do
      pair_variation="${VARIATIONS[pair_idx]}"
      pair_seed="$((SEED + pair_idx))"
      save_path="${SAVE_ROOT}/${policy}_${state}/var${pair_variation}_seed${pair_seed}"

      echo
      echo "============================================================"
      echo "policy      : ${policy}"
      echo "state       : ${state}"
      echo "action_mode : ${action_mode}"
      echo "variation   : ${pair_variation}"
      echo "seed        : ${pair_seed}"
      echo "checkpoint  : ${checkpoint_dir}"
      echo "dataset_root: ${dataset_root}"
      echo "save_path   : ${save_path}"

      cmd=(
        "${PYTHON_BIN}" "${WORKSPACE_ROOT}/src/inference/rlbench/eval_put_rubbish_in_bin.py"
        --task "${TASK}"
        --variation "${pair_variation}"
        --runs "1"
        --seed "${pair_seed}"
        --max_steps "${MAX_STEPS}"
        --action_mode "${action_mode}"
        --renderer "${RENDERER}"
        --checkpoint "${checkpoint_dir}"
        --dataset_root "${dataset_root}"
        --task_description "${TASK_DESCRIPTION}"
        --save_path "${save_path}"
      )

      if [[ -n "${DEVICE}" ]]; then
        cmd+=(--device "${DEVICE}")
      fi

      echo "Command: ${cmd[*]}"
      if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${cmd[@]}"
      fi
    done
  done
done

echo
echo "Batch evaluation done. Outputs are under: ${SAVE_ROOT}"

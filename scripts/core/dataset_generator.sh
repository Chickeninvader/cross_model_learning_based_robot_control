#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/init_script_logging.sh
source "${SCRIPT_DIR}/../lib/init_script_logging.sh"

RLBENCH_ROOT="${RLBENCH_ROOT:-$REPO_ROOT/external/RLBench}"
OUT_ROOT="${OUT_ROOT:-}"
PY="${PY:-python}"
PROCESSES="${PROCESSES:-4}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
START_VARIATION="${START_VARIATION:-0}"
IMAGE_WIDTH="${IMAGE_WIDTH:-256}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"
RENDERER="${RENDERER:-opengl3}"

usage() {
  cat <<'EOF'
Usage:
  scripts/dataset_generator.sh <num_variations> <task_or_task_file> [more_tasks...]

Examples:
  scripts/dataset_generator.sh 5 \
    external/RLBench/rlbench/tasks/put_banana_in_bin.py \
    external/RLBench/rlbench/tasks/put_rubbish_in_bin.py

  scripts/dataset_generator.sh 20 put_banana_in_bin put_rubbish_in_bin

Notes:
  - <num_variations> must be a positive integer.
  - Episodes per variation defaults to 1. Override with:
      EPISODES_PER_TASK=<N> scripts/dataset_generator.sh ...
  - Start variation defaults to 0. Override with:
      START_VARIATION=<N> scripts/dataset_generator.sh ...
  - OUT_ROOT (env var) is required: the RLBench dataset root to write into.
    Example:
      OUT_ROOT=datasets/rlbench_<run_name> scripts/dataset_generator.sh ...
  - Each task can be:
      1) a task python file path (e.g. .../put_banana_in_bin.py), or
      2) a task name (e.g. put_banana_in_bin).
EOF
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

NUM_VARIATIONS="$1"
shift

if ! [[ "$NUM_VARIATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: <num_variations> must be a positive integer, got: $NUM_VARIATIONS" >&2
  exit 1
fi

if ! [[ "$EPISODES_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: EPISODES_PER_TASK must be a positive integer, got: $EPISODES_PER_TASK" >&2
  exit 1
fi

task_names=()
for task_input in "$@"; do
  task_base="$(basename "$task_input")"
  task_name="${task_base%.py}"

  # Handle accidental directory input like ".../tasks/"
  if [ -z "$task_name" ] || [ "$task_name" = "tasks" ]; then
    echo "Warning: skipping invalid task input: $task_input" >&2
    continue
  fi

  task_names+=("$task_name")
done

if [ "${#task_names[@]}" -eq 0 ]; then
  echo "Error: no valid tasks were provided." >&2
  exit 1
fi

if [[ -z "$OUT_ROOT" ]]; then
  echo "Error: OUT_ROOT env var is required (dataset output root)." >&2
  echo "Example: OUT_ROOT=datasets/rlbench_<run_name> $0 $NUM_VARIATIONS <task1> [task2 ...]" >&2
  usage
  exit 1
fi

if [[ "$OUT_ROOT" != /* ]]; then
  OUT_ROOT="$REPO_ROOT/$OUT_ROOT"
fi

export PYTHONPATH="$REPO_ROOT/external/RLBench:$REPO_ROOT/external/lerobot/src:${PYTHONPATH:-}"

cd "$RLBENCH_ROOT"
mkdir -p "$OUT_ROOT"

supports_start_variation=0
if [ -f "$RLBENCH_ROOT/rlbench/dataset_generator.py" ] && \
   rg -q -- '--start_variation' "$RLBENCH_ROOT/rlbench/dataset_generator.py"; then
  supports_start_variation=1
fi

echo "--------------------------------------"
echo "Variations per task: $NUM_VARIATIONS"
echo "Episodes per variation: $EPISODES_PER_TASK"
echo "Start variation: $START_VARIATION"
echo "Tasks to generate: ${#task_names[@]}"
printf '%s\n' "${task_names[@]}"
echo "Output path: $OUT_ROOT"
echo "Processes per task: $PROCESSES"
echo "--------------------------------------"

success_tasks=()
failed_tasks=()

for task_name in "${task_names[@]}"; do
  echo
  echo "======================================"
  echo "Running task: $task_name"
  echo "======================================"

  cmd=(
    "$PY" -m rlbench.dataset_generator
    --tasks "$task_name"
    --episodes_per_task "$EPISODES_PER_TASK"
    --variations "$NUM_VARIATIONS"
    --processes "$PROCESSES"
    --image_size "$IMAGE_WIDTH" "$IMAGE_HEIGHT"
    --renderer "$RENDERER"
    --save_path "$OUT_ROOT"
  )

  if [ "$supports_start_variation" -eq 1 ]; then
    cmd+=(--start_variation "$START_VARIATION")
  elif [ "$START_VARIATION" != "0" ]; then
    echo "Warning: this RLBench checkout does not support --start_variation; ignoring START_VARIATION=$START_VARIATION" >&2
  fi

  if "${cmd[@]}"; then
    success_tasks+=("$task_name")
  else
    echo "Task failed: $task_name"
    failed_tasks+=("$task_name")
  fi
done

echo
echo "--------------------------------------"
echo "Dataset generation summary"
echo "Succeeded: ${#success_tasks[@]}"
if [ "${#success_tasks[@]}" -gt 0 ]; then
  printf '  - %s\n' "${success_tasks[@]}"
fi
echo "Failed: ${#failed_tasks[@]}"
if [ "${#failed_tasks[@]}" -gt 0 ]; then
  printf '  - %s\n' "${failed_tasks[@]}"
fi
echo "--------------------------------------"

if [ "${#failed_tasks[@]}" -gt 0 ]; then
  exit 1
fi

echo "Dataset generation complete."

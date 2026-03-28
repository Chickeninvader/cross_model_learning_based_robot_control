#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/init_script_logging.sh
source "${SCRIPT_DIR}/../lib/init_script_logging.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONVERTER="src/data_collection/convert_rlbench_to_lerobot.py"
DATASET_PATH=""
OUTPUT_ROOT=""
EPISODE="0"
FPS="20"
# LeRobot RGB video size (ffmpeg scale). Set both to 0 for native PNG resolution (e.g. 256x256).
IMAGE_WIDTH="${IMAGE_WIDTH:-128}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-128}"
ACTION_SPACE="both"
USE_CONTEXT_PROMPT="false"
TASKS_CSV=""
STRICT_MODE="false"
MERGE_ALL_TASKS="true"

usage() {
  cat <<'EOF'
Batch convert all RLBench tasks in a dataset folder to LeRobot format.

Usage:
  scripts/core/convert_rlbench_to_lerobot_batch.sh --dataset-path <path> [options]

Required:
  --dataset-path <path>       RLBench dataset root (e.g. datasets/rlbench_<run_name>)
  --output-root <path>       LeRobot output root (e.g. datasets/lerobot_<run_name>)

Optional:
  --tasks <csv>               Comma-separated task names (default: all task folders)
  --episode <id>              Episode index (default: 0)
  --fps <num>                 Output FPS (default: 20)
  --image-width <n>           LeRobot video width (default: env IMAGE_WIDTH or 128; 0 = native)
  --image-height <n>         LeRobot video height (default: env IMAGE_HEIGHT or 128; 0 = native)
  --action-space <mode>       eef|joint|both (default: both)
  --use-context-prompt        Enable context prompt mode
  --no-merge-all-tasks        Skip final merge to all_task_eef/joint
  --strict                    Exit non-zero if any task fails
  -h, --help                  Show this help

Examples:
  scripts/core/convert_rlbench_to_lerobot_batch.sh \
    --dataset-path datasets/rlbench_<run_name> \
    --output-root datasets/lerobot_<run_name>

  scripts/core/convert_rlbench_to_lerobot_batch.sh \
    --dataset-path datasets/rlbench_<run_name> \
    --output-root datasets/lerobot_<run_name> \
    --use-context-prompt

  scripts/core/convert_rlbench_to_lerobot_batch.sh \
    --dataset-path datasets/rlbench_<run_name> \
    --output-root datasets/lerobot_<run_name> \
    --tasks "lamp_on,put_rubbish_in_bin"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --tasks)
      TASKS_CSV="${2:-}"
      shift 2
      ;;
    --episode)
      EPISODE="${2:-}"
      shift 2
      ;;
    --fps)
      FPS="${2:-}"
      shift 2
      ;;
    --image-width)
      IMAGE_WIDTH="${2:-}"
      shift 2
      ;;
    --image-height)
      IMAGE_HEIGHT="${2:-}"
      shift 2
      ;;
    --action-space)
      ACTION_SPACE="${2:-}"
      shift 2
      ;;
    --use-context-prompt)
      USE_CONTEXT_PROMPT="true"
      shift
      ;;
    --no-merge-all-tasks)
      MERGE_ALL_TASKS="false"
      shift
      ;;
    --strict)
      STRICT_MODE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$DATASET_PATH" ]]; then
  echo "Error: --dataset-path is required." >&2
  usage
  exit 1
fi

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Error: dataset path not found: $DATASET_PATH" >&2
  exit 1
fi

if [[ ! -f "$CONVERTER" ]]; then
  echo "Error: converter script not found: $CONVERTER" >&2
  exit 1
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  echo "Error: --output-root is required." >&2
  usage
  exit 1
fi

case "$ACTION_SPACE" in
  eef|joint|both) ;;
  *)
    echo "Error: --action-space must be one of: eef, joint, both" >&2
    exit 1
    ;;
esac

if ! [[ "$EPISODE" =~ ^[0-9]+$ ]]; then
  echo "Error: --episode must be a non-negative integer, got: $EPISODE" >&2
  exit 1
fi

if ! [[ "$FPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --fps must be a positive integer, got: $FPS" >&2
  exit 1
fi

if ! [[ "$IMAGE_WIDTH" =~ ^[0-9]+$ && "$IMAGE_HEIGHT" =~ ^[0-9]+$ ]]; then
  echo "Error: --image-width and --image-height must be non-negative integers" >&2
  exit 1
fi
if [[ "$IMAGE_WIDTH" -eq 0 && "$IMAGE_HEIGHT" -ne 0 ]] || [[ "$IMAGE_HEIGHT" -eq 0 && "$IMAGE_WIDTH" -ne 0 ]]; then
  echo "Error: for native resolution, both --image-width and --image-height must be 0" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

declare -a TASKS=()
if [[ -n "$TASKS_CSV" ]]; then
  IFS=',' read -r -a TASKS <<< "$TASKS_CSV"
  # Trim whitespace and validate each selected task exists.
  declare -a FILTERED_TASKS=()
  for i in "${!TASKS[@]}"; do
    task_trimmed="$(echo "${TASKS[$i]}" | xargs)"
    if [[ -z "$task_trimmed" ]]; then
      continue
    fi
    if [[ ! -d "$DATASET_PATH/$task_trimmed" ]]; then
      echo "Error: task not found in dataset path: $task_trimmed" >&2
      exit 1
    fi
    FILTERED_TASKS+=("$task_trimmed")
  done
  TASKS=("${FILTERED_TASKS[@]}")
else
  while IFS= read -r d; do
    TASKS+=("$(basename "$d")")
  done < <(ls -1d "$DATASET_PATH"/*/ 2>/dev/null || true)
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "Error: no task directories found under $DATASET_PATH" >&2
  exit 1
fi

echo "RLBench dataset path: $DATASET_PATH"
echo "LeRobot output root:  $OUTPUT_ROOT"
echo "Episode:              $EPISODE"
echo "FPS:                  $FPS"
echo "Image width x height:   ${IMAGE_WIDTH}x${IMAGE_HEIGHT} (0x0 = native PNG size)"
echo "Action space:         $ACTION_SPACE"
echo "Use context prompt:   $USE_CONTEXT_PROMPT"
echo "Merge all tasks:      $MERGE_ALL_TASKS"
echo "Strict mode:          $STRICT_MODE"
echo "Tasks found:          ${#TASKS[@]}"
printf '  - %s\n' "${TASKS[@]}"
echo

success=0
failed=0

for task in "${TASKS[@]}"; do
  echo "============================================================"
  echo "Converting task: $task"
  echo "============================================================"

  cmd=(
    "$PYTHON_BIN" "$CONVERTER"
    --task_name "$task"
    --rlbench_root "$DATASET_PATH"
    --output_root "$OUTPUT_ROOT"
    --fps "$FPS"
    --image_width "$IMAGE_WIDTH"
    --image_height "$IMAGE_HEIGHT"
    --episode "$EPISODE"
    --action_space "$ACTION_SPACE"
  )

  if [[ "$USE_CONTEXT_PROMPT" == "true" ]]; then
    cmd+=(--use_context_prompt)
  fi
  if [[ "$STRICT_MODE" == "true" ]]; then
    cmd+=(--strict_variations)
  fi

  if "${cmd[@]}"; then
    ((success+=1))
  else
    echo "[FAIL] Task conversion failed: $task" >&2
    ((failed+=1))
  fi
  echo
done

merge_failed=0
if [[ "$MERGE_ALL_TASKS" == "true" ]]; then
  if [[ $success -gt 0 ]]; then
    echo
    echo "============================================================"
    echo "Merging converted task datasets -> all_task_*"
    echo "============================================================"
    merge_cmd=(
      "$PYTHON_BIN" "$CONVERTER"
      --merge_all_tasks_root "$OUTPUT_ROOT"
      --output_root "$OUTPUT_ROOT"
      --action_space "$ACTION_SPACE"
    )
    if "${merge_cmd[@]}"; then
      echo "[OK] Final merged dataset(s) created under: $OUTPUT_ROOT"
    else
      echo "[FAIL] Final merge failed." >&2
      merge_failed=1
    fi
  else
    echo "[WARN] Skip final merge because no task conversion succeeded."
  fi
fi

echo "==================== FINAL SUMMARY ===================="
echo "Successful tasks: $success"
echo "Failed tasks:     $failed"
echo "Merge failed:     $merge_failed"
echo "Output root:      $OUTPUT_ROOT"

if [[ "$STRICT_MODE" == "true" ]] && (( failed > 0 || merge_failed > 0 )); then
  exit 1
fi

exit 0

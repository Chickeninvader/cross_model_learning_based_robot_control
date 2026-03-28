#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/init_script_logging.sh
source "${SCRIPT_DIR}/../lib/init_script_logging.sh"

DATASET_PATH=""
APPLY_SCRIPT="$REPO_ROOT/src/data_collection/apply_template_batch.py"
EPISODE="0"
CAMERA="front"
SKIP_VIDEOS="false"
TASKS_CSV=""

usage() {
  cat <<'EOF'
Apply relationship templates to RLBench dataset tasks (batch).

Usage:
  scripts/core/apply_rlbench_scene_graph_templates.sh --dataset-path <path> [options]

Required:
  --dataset-path <path>   RLBench dataset root containing per-task folders with
                           <task>_relationship_template.json (and task variations).

Options:
  --episode <id>          Episode id (default: 0)
  --camera <name>         Camera: front|wrist|left_shoulder|right_shoulder|overhead (default: front)
  --skip-videos           Only generate scene graph JSONs
  --tasks <csv>           Comma-separated task names (default: all task folders)
  -h, --help              Show this help

Examples:
  scripts/core/apply_rlbench_scene_graph_templates.sh \
    --dataset-path datasets/rlbench_<run_name> \
    --skip-videos

  scripts/core/apply_rlbench_scene_graph_templates.sh \
    --dataset-path datasets/rlbench_<run_name> \
    --tasks "lamp_on,put_rubbish_in_bin" \
    --episode 0 \
    --camera front
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="${2:-}"
      shift 2
      ;;
    --episode)
      EPISODE="${2:-}"
      shift 2
      ;;
    --camera)
      CAMERA="${2:-}"
      shift 2
      ;;
    --skip-videos)
      SKIP_VIDEOS="true"
      shift
      ;;
    --tasks)
      TASKS_CSV="${2:-}"
      shift 2
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

if [[ "$DATASET_PATH" != /* ]]; then
  DATASET_PATH="$REPO_ROOT/$DATASET_PATH"
fi

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Dataset path not found: $DATASET_PATH" >&2
  exit 1
fi

if [[ ! -f "$APPLY_SCRIPT" ]]; then
  echo "apply_template_batch.py not found: $APPLY_SCRIPT" >&2
  exit 1
fi

# Ensure python can import `utils.*` from `src/`.
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

declare -a TASKS=()
if [[ -n "$TASKS_CSV" ]]; then
  IFS=',' read -r -a TASKS <<< "$TASKS_CSV"
else
  while IFS= read -r task_dir; do
    TASKS+=("$(basename "$task_dir")")
  done < <(ls -1d "$DATASET_PATH"/* 2>/dev/null || true)
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "No task directories found in $DATASET_PATH" >&2
  exit 1
fi

success=0
failed=0
skipped=0

echo "Dataset path: $DATASET_PATH"
echo "Episode: $EPISODE"
echo "Camera: $CAMERA"
echo "Skip videos: $SKIP_VIDEOS"
echo

for task in "${TASKS[@]}"; do
  task="$(echo "$task" | xargs)"
  [[ -z "$task" ]] && continue

  task_dir="$DATASET_PATH/$task"
  template_path="$task_dir/${task}_relationship_template.json"

  if [[ ! -d "$task_dir" ]]; then
    echo "[SKIP] Task folder missing: $task_dir"
    ((skipped+=1))
    continue
  fi

  if [[ ! -f "$template_path" ]]; then
    echo "[SKIP] Template missing: $template_path"
    ((skipped+=1))
    continue
  fi

  echo "============================================================"
  echo "Applying template for task: $task"
  echo "============================================================"

  cmd=(
    python "$APPLY_SCRIPT"
    --task "$task"
    --episode "$EPISODE"
    --camera "$CAMERA"
    --dataset_path "$DATASET_PATH"
  )

  if [[ "$SKIP_VIDEOS" == "true" ]]; then
    cmd+=(--skip-videos)
  fi

  if "${cmd[@]}"; then
    ((success+=1))
  else
    echo "[FAIL] Task failed: $task" >&2
    ((failed+=1))
  fi
  echo
done

echo "==================== FINAL SUMMARY ===================="
echo "Successful tasks: $success"
echo "Failed tasks:     $failed"
echo "Skipped tasks:    $skipped"

if [[ $failed -gt 0 ]]; then
  exit 1
fi

exit 0


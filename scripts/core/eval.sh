#!/usr/bin/env bash
set -euo pipefail

# Canonical evaluation entrypoint under scripts/core.
# Defaults to single-task flow. Use --all_tasks to route to all-task flow.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_TASK_SCRIPT="${SCRIPT_DIR}/eval_rlbench_single_task.sh"
ALL_TASK_SCRIPT="${SCRIPT_DIR}/eval_rlbench_all_tasks.sh"

MODE="single_task"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all_tasks)
      MODE="all_tasks"
      shift
      ;;
    --single_task)
      MODE="single_task"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${MODE}" == "all_tasks" ]]; then
  exec "${ALL_TASK_SCRIPT}" "${ARGS[@]}"
else
  exec "${SINGLE_TASK_SCRIPT}" "${ARGS[@]}"
fi

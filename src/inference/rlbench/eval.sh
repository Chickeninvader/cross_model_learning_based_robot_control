#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper.
# Canonical scripts now live under scripts/core.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TARGET="${WORKSPACE_ROOT}/scripts/core/eval_rlbench_single_task.sh"

if [[ ! -f "${TARGET}" ]]; then
  echo "[ERROR] Missing target script: ${TARGET}" >&2
  exit 1
fi

echo "[INFO] Redirecting to scripts/core/eval_rlbench_single_task.sh"
exec "${TARGET}" "$@"

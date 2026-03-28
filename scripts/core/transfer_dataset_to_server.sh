#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../lib/init_script_logging.sh
source "${SCRIPT_DIR}/../lib/init_script_logging.sh"

# Sync a local dataset (or one task inside it) to the remote server via rsync.
#
# Defaults can be overridden with env vars:
#   REMOTE_HOST, REMOTE_DATASETS_DIR, SSH_CONTROL_PATH

REMOTE_HOST="${REMOTE_HOST:-ngocbach@login.sol.rc.asu.edu}"
REMOTE_DATASETS_DIR="${REMOTE_DATASETS_DIR:-/scratch/kpham34/cross_model_learning_based_robot_control/datasets}"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-$HOME/.ssh/cm-%r@%h:%p}"
RSYNC_COMPRESS="${RSYNC_COMPRESS:-0}"
RSYNC_MAX_RETRIES="${RSYNC_MAX_RETRIES:-3}"

DATASET_PATH=""
TASK_NAME=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/transfer_dataset_to_server.sh --dataset-path <path> [--task <task_name>] [--dry-run]

Examples:
  # Transfer full dataset folder to remote datasets root
  scripts/transfer_dataset_to_server.sh \
    --dataset-path datasets/rlbench_<run_name>

  # Transfer one task subfolder only
  scripts/transfer_dataset_to_server.sh \
    --dataset-path datasets/rlbench_<run_name> \
    --task put_rubbish_in_bin

  # Transfer final merged datasets (both all_task_eef and all_task_joint)
  ./scripts/core/transfer_dataset_to_server.sh \
    --dataset-path datasets/lerobot_<run_name> \
    --task all_task

  ./scripts/core/transfer_dataset_to_server.sh \
    --dataset-path datasets/lerobot_trial_3 \
    --task all_task

Options:
  --dataset-path <path>  Required local dataset directory.
  --task <task_name>     Optional task subdirectory to transfer from dataset path.
                         Special value: all_task -> transfers all_task_eef and all_task_joint.
  --dry-run              Preview rsync actions without copying.
  -h, --help             Show this help.

Environment overrides:
  REMOTE_HOST            Remote SSH target (default: ngocbach@login.sol.rc.asu.edu)
  REMOTE_DATASETS_DIR    Remote datasets root directory
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      [[ $# -ge 2 ]] || { echo "Error: --dataset-path requires a value." >&2; usage; exit 1; }
      DATASET_PATH="$2"
      shift 2
      ;;
    --task)
      [[ $# -ge 2 ]] || { echo "Error: --task requires a value." >&2; usage; exit 1; }
      TASK_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET_PATH" ]]; then
  echo "Error: --dataset-path is required." >&2
  usage
  exit 1
fi

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Error: dataset path does not exist or is not a directory: $DATASET_PATH" >&2
  exit 1
fi

SSH_COMMON_OPTS=(
  -o ControlMaster=auto
  -o ControlPersist=600
  -o "ControlPath=$SSH_CONTROL_PATH"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)

mkdir -p "$HOME/.ssh"
open_ssh_master() {
  log "Opening SSH master connection to $REMOTE_HOST (one password prompt)"
  ssh "${SSH_COMMON_OPTS[@]}" -MNf "$REMOTE_HOST"
}

close_ssh_master() {
  ssh "${SSH_COMMON_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}

open_ssh_master
trap 'close_ssh_master' EXIT

dataset_name="$(basename "$DATASET_PATH")"
remote_dataset_dir="${REMOTE_DATASETS_DIR%/}/${dataset_name}"
rsync_common_opts=(
  -a
  --info=progress2
  --partial
  --append-verify
  --human-readable
  -e "ssh ${SSH_COMMON_OPTS[*]}"
)

if [[ "$RSYNC_COMPRESS" == "1" ]]; then
  rsync_common_opts+=(-z)
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  rsync_common_opts+=(--dry-run)
fi

run_rsync_with_retry() {
  local src_path="$1"
  local dst_path="$2"
  local attempt=1
  local rc=0

  while (( attempt <= RSYNC_MAX_RETRIES )); do
    log "rsync attempt ${attempt}/${RSYNC_MAX_RETRIES}: ${src_path} -> ${dst_path}"
    set +e
    rsync "${rsync_common_opts[@]}" "$src_path" "$dst_path"
    rc=$?
    set -e

    if [[ "$rc" -eq 0 ]]; then
      return 0
    fi

    # Common transient transport failures: socket/stream/SSH disconnect.
    if [[ "$rc" -eq 10 || "$rc" -eq 12 || "$rc" -eq 255 ]]; then
      if (( attempt < RSYNC_MAX_RETRIES )); then
        local wait_secs=$((attempt * 5))
        log "rsync failed with code ${rc}; retrying in ${wait_secs}s..."
        close_ssh_master
        open_ssh_master
        sleep "$wait_secs"
        attempt=$((attempt + 1))
        continue
      fi
    fi

    return "$rc"
  done

  return "$rc"
}

if [[ -n "$TASK_NAME" ]]; then
  remote_task_parent="$remote_dataset_dir/"
  log "Ensuring remote task parent exists: ${REMOTE_HOST}:${remote_task_parent}"
  ssh "${SSH_COMMON_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$remote_task_parent'"

  task_names=()
  if [[ "$TASK_NAME" == "all_task" ]]; then
    task_names=("all_task_eef" "all_task_joint")
  else
    IFS=',' read -r -a task_names <<< "$TASK_NAME"
  fi

  transferred_any=0
  for task in "${task_names[@]}"; do
    task="$(echo "$task" | xargs)"
    [[ -n "$task" ]] || continue
    local_task_path="${DATASET_PATH%/}/${task}"
    if [[ ! -d "$local_task_path" ]]; then
      log "Skipping missing task folder: $local_task_path"
      continue
    fi

    log "Syncing task '${task}' from '$DATASET_PATH' to ${REMOTE_HOST}:${remote_task_parent}"
    run_rsync_with_retry \
      "${local_task_path%/}/" \
      "${REMOTE_HOST}:${remote_task_parent}${task}/"
    transferred_any=1
  done

  if [[ "$transferred_any" -ne 1 ]]; then
    echo "Error: no task folders were found for --task '$TASK_NAME' under: $DATASET_PATH" >&2
    exit 1
  fi
else
  remote_parent="${REMOTE_DATASETS_DIR%/}/"
  log "Ensuring remote datasets root exists: ${REMOTE_HOST}:${remote_parent}"
  ssh "${SSH_COMMON_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$remote_parent'"

  log "Syncing dataset '${dataset_name}' to ${REMOTE_HOST}:${remote_parent}"
  run_rsync_with_retry \
    "${DATASET_PATH%/}/" \
    "${REMOTE_HOST}:${remote_parent}${dataset_name}/"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry run complete. No files were transferred."
else
  log "Transfer complete."
fi

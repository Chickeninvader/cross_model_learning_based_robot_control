#!/usr/bin/env bash
set -euo pipefail

# Transfer only the latest LeRobot checkpoint per run from a remote host.
# The script tracks completed transfers in a local state file so reruns skip
# items that have already been copied.
#
# Defaults are tailored to this project and can be overridden via env vars:
#   REMOTE_HOST, REMOTE_OUTPUT_DIR, LOCAL_OUTPUT_DIR, STATE_FILE

REMOTE_HOST="${REMOTE_HOST:-ngocbach@login.sol.rc.asu.edu}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-/scratch/kpham34/cross_model_learning_based_robot_control/output/lerobot}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$HOME/Desktop/cross_model_learning_based_robot_control/output/lerobot}"
STATE_FILE="${STATE_FILE:-$LOCAL_OUTPUT_DIR/.transferred_last_checkpoints.tsv}"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-$HOME/.ssh/cm-%r@%h:%p}"
MAX_PARALLEL_TRANSFERS="${MAX_PARALLEL_TRANSFERS:-$(nproc)}"
SCP_ENABLE_COMPRESSION="${SCP_ENABLE_COMPRESSION:-1}"

mkdir -p "$LOCAL_OUTPUT_DIR"
touch "$STATE_FILE"

SSH_COMMON_OPTS=(
    -o ControlMaster=auto
    -o ControlPersist=600
    -o "ControlPath=$SSH_CONTROL_PATH"
)
RSYNC_SSH_CMD=(ssh "${SSH_COMMON_OPTS[@]}")
RSYNC_OPTS=(
    -a
    --partial
    --append-verify
)
if [[ "$SCP_ENABLE_COMPRESSION" == "1" ]]; then
    RSYNC_OPTS+=(--compress)
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

start_master_connection() {
    mkdir -p "$HOME/.ssh"
    log "Opening SSH master connection to $REMOTE_HOST (one password prompt)"
    ssh "${SSH_COMMON_OPTS[@]}" -MNf "$REMOTE_HOST"
}

stop_master_connection() {
    ssh "${SSH_COMMON_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}

state_has() {
    local run_name="$1"
    local ckpt_path="$2"
    if awk -F '\t' -v run="$run_name" -v path="$ckpt_path" \
        '$2 == run && $3 == path { found=1; exit } END { exit !found }' \
        "$STATE_FILE"; then
        return 0
    fi
    return 1
}

state_add() {
    local run_name="$1"
    local ckpt_path="$2"
    local source_tag="${3:-scp}"
    printf '%s\t%s\t%s\t%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" \
        "$run_name" \
        "$ckpt_path" \
        "$source_tag" >> "$STATE_FILE"
}

bootstrap_existing_local() {
    local run_dir run_name last_link resolved_target local_ckpt_root
    shopt -s nullglob
    for run_dir in "$LOCAL_OUTPUT_DIR"/*; do
        [[ -d "$run_dir" ]] || continue
        run_name="$(basename "$run_dir")"
        [[ "$run_name" == .* ]] && continue

        last_link="$run_dir/checkpoints/last"
        if [[ -L "$last_link" ]]; then
            if resolved_target="$(readlink -f "$last_link" 2>/dev/null)" && [[ -d "$resolved_target" ]]; then
                if ! state_has "$run_name" "$resolved_target"; then
                    state_add "$run_name" "$resolved_target" "bootstrap-local"
                fi
            fi
            continue
        fi

        # Fallback for already-copied runs that do not have a local `last` symlink.
        # Only trust an explicit local `checkpoints/last` directory marker.
        # Do not infer completion from numeric checkpoint directories because
        # interrupted transfers can create partial trees that look valid.
        local_ckpt_root="$run_dir/checkpoints"
        [[ -d "$local_ckpt_root" ]] || continue
        if [[ -d "$local_ckpt_root/last" ]]; then
            resolved_target="${REMOTE_OUTPUT_DIR}/${run_name}/checkpoints/last"
            if ! state_has "$run_name" "$resolved_target"; then
                state_add "$run_name" "$resolved_target" "bootstrap-local"
            fi
        fi
    done
}

list_remote_last_targets() {
    ssh "${SSH_COMMON_OPTS[@]}" "$REMOTE_HOST" "bash -lc '
set -euo pipefail
shopt -s nullglob
for run_dir in \"${REMOTE_OUTPUT_DIR}\"/*; do
    [[ -d \"\$run_dir\" ]] || continue
    run_name=\"\$(basename \"\$run_dir\")\"
    [[ \"\$run_name\" == debug ]] && continue
    last_link=\"\$run_dir/checkpoints/last\"
    [[ -e \"\$last_link\" ]] || continue
    if [[ -L \"\$last_link\" ]]; then
        target=\"\$(readlink -f \"\$last_link\" || true)\"
        [[ -n \"\$target\" && -d \"\$target\" ]] || continue
    elif [[ -d \"\$last_link\" ]]; then
        target=\"\$last_link\"
    else
        continue
    fi
    printf \"%s\t%s\n\" \"\$run_name\" \"\$target\"
done
'"
}

transfer_one() {
    local run_name="$1"
    local remote_ckpt_path="$2"
    local local_run_dir local_ckpt_root local_ckpt_dirname

    local_run_dir="$LOCAL_OUTPUT_DIR/$run_name"
    local_ckpt_root="$local_run_dir/checkpoints"
    local_ckpt_dirname="$(basename "$remote_ckpt_path")"

    mkdir -p "$local_ckpt_root/$local_ckpt_dirname"
    log "Transferring $run_name ($local_ckpt_dirname) via rsync"
    rsync "${RSYNC_OPTS[@]}" -e "${RSYNC_SSH_CMD[*]}" \
        "${REMOTE_HOST}:${remote_ckpt_path}/" \
        "$local_ckpt_root/$local_ckpt_dirname/"
    if [[ "$local_ckpt_dirname" != "last" ]]; then
        ln -sfn "$local_ckpt_dirname" "$local_ckpt_root/last"
    fi
    state_add "$run_name" "$remote_ckpt_path" "rsync"
}

main() {
    local run_name remote_ckpt_path transferred_count skipped_count failed_count pid
    local -a pids
    transferred_count=0
    skipped_count=0
    failed_count=0
    pids=()

    if ! [[ "$MAX_PARALLEL_TRANSFERS" =~ ^[1-9][0-9]*$ ]]; then
        log "Invalid MAX_PARALLEL_TRANSFERS=$MAX_PARALLEL_TRANSFERS (must be positive int), using 1"
        MAX_PARALLEL_TRANSFERS=1
    fi
    if ! command -v rsync >/dev/null 2>&1; then
        log "rsync not found on local machine. Please install rsync and rerun."
        return 1
    fi

    start_master_connection
    trap stop_master_connection EXIT

    bootstrap_existing_local
    log "Starting transfer scan with MAX_PARALLEL_TRANSFERS=$MAX_PARALLEL_TRANSFERS SCP_ENABLE_COMPRESSION=$SCP_ENABLE_COMPRESSION"

    while IFS=$'\t' read -r run_name remote_ckpt_path; do
        [[ -n "${run_name:-}" && -n "${remote_ckpt_path:-}" ]] || continue

        if state_has "$run_name" "$remote_ckpt_path"; then
            skipped_count=$((skipped_count + 1))
            log "Skipping already transferred: $run_name ($remote_ckpt_path)"
            continue
        fi

        transfer_one "$run_name" "$remote_ckpt_path" &
        pids+=("$!")

        # Throttle background jobs to configured concurrency.
        while (( $(jobs -pr | wc -l) >= MAX_PARALLEL_TRANSFERS )); do
            sleep 0.2
        done
    done < <(list_remote_last_targets)

    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            transferred_count=$((transferred_count + 1))
        else
            failed_count=$((failed_count + 1))
            log "A transfer job failed (pid=$pid)"
        fi
    done

    log "Done. transferred=$transferred_count skipped=$skipped_count failed=$failed_count state_file=$STATE_FILE"
    if (( failed_count > 0 )); then
        return 1
    fi
}

main "$@"

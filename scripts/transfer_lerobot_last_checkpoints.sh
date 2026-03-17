#!/usr/bin/env bash
set -euo pipefail

# Transfer only the latest LeRobot checkpoint per run from a remote host.
# The script tracks completed transfers in a local state file so reruns skip
# items that have already been copied.
#
# Defaults are tailored to this project and can be overridden via env vars:
#   REMOTE_HOST, REMOTE_OUTPUT_DIR, LOCAL_OUTPUT_DIR, STATE_FILE

REMOTE_HOST="${REMOTE_HOST:-ngocbach@en4217548l}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-/scratch/kpham34/cross_model_learning_based_robot_control/output/lerobot}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$HOME/Desktop/cross_model_learning_based_robot_control/output/lerobot}"
STATE_FILE="${STATE_FILE:-$LOCAL_OUTPUT_DIR/.transferred_last_checkpoints.tsv}"

mkdir -p "$LOCAL_OUTPUT_DIR"
touch "$STATE_FILE"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
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
    local run_dir run_name last_link resolved_target local_ckpt_root ckpt_dir ckpt_name
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
        local_ckpt_root="$run_dir/checkpoints"
        [[ -d "$local_ckpt_root" ]] || continue
        for ckpt_dir in "$local_ckpt_root"/*; do
            [[ -d "$ckpt_dir" ]] || continue
            ckpt_name="$(basename "$ckpt_dir")"
            [[ "$ckpt_name" =~ ^[0-9]+$ ]] || continue
            if [[ -d "$ckpt_dir/pretrained_model" || -d "$ckpt_dir/training_state" ]]; then
                resolved_target="${REMOTE_OUTPUT_DIR}/${run_name}/checkpoints/${ckpt_name}"
                if ! state_has "$run_name" "$resolved_target"; then
                    state_add "$run_name" "$resolved_target" "bootstrap-local"
                fi
            fi
        done
    done
}

list_remote_last_targets() {
    ssh "$REMOTE_HOST" "bash -lc '
set -euo pipefail
shopt -s nullglob
for run_dir in \"${REMOTE_OUTPUT_DIR}\"/*; do
    [[ -d \"\$run_dir\" ]] || continue
    run_name=\"\$(basename \"\$run_dir\")\"
    [[ \"\$run_name\" == debug ]] && continue
    last_link=\"\$run_dir/checkpoints/last\"
    [[ -L \"\$last_link\" ]] || continue
    target=\"\$(readlink -f \"\$last_link\" || true)\"
    [[ -n \"\$target\" && -d \"\$target\" ]] || continue
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

    mkdir -p "$local_ckpt_root"
    log "Transferring $run_name ($local_ckpt_dirname)"
    scp -r "${REMOTE_HOST}:${remote_ckpt_path}" "$local_ckpt_root/"
    ln -sfn "$local_ckpt_dirname" "$local_ckpt_root/last"
    state_add "$run_name" "$remote_ckpt_path" "scp"
}

main() {
    local line run_name remote_ckpt_path transferred_count skipped_count
    transferred_count=0
    skipped_count=0

    bootstrap_existing_local

    while IFS=$'\t' read -r run_name remote_ckpt_path; do
        [[ -n "${run_name:-}" && -n "${remote_ckpt_path:-}" ]] || continue

        if state_has "$run_name" "$remote_ckpt_path"; then
            skipped_count=$((skipped_count + 1))
            log "Skipping already transferred: $run_name ($remote_ckpt_path)"
            continue
        fi

        transfer_one "$run_name" "$remote_ckpt_path"
        transferred_count=$((transferred_count + 1))
    done < <(list_remote_last_targets)

    log "Done. transferred=$transferred_count skipped=$skipped_count state_file=$STATE_FILE"
}

main "$@"

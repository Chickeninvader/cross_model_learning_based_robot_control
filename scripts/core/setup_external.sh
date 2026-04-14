#!/usr/bin/env bash
# ============================================================
# Clone external dependencies into external/
# Run from the project root:
#   bash scripts/core/setup_external.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT
# shellcheck source=../lib/init_script_logging.sh
source "${SCRIPT_DIR}/../lib/init_script_logging.sh"
EXT_DIR="$REPO_ROOT/external"

RLBENCH_REPO_URL="https://github.com/chickeninvader/RLBench.git"
LEROBOT_REPO_URL="https://github.com/huggingface/lerobot.git"
LEROBOT_COMMIT="8fff0fde7c79f23a93d845d1a50e985de01f8b8a"

mkdir -p "$EXT_DIR"

clone_if_missing() {
    local name="$1"
    local url="$2"
    local branch="${3:-}"
    local commit="${4:-}"
    local dest="$EXT_DIR/$name"

    if [ -d "$dest" ]; then
        echo "[skip] $name already exists at $dest"
    else
        echo "[clone] $name → $dest"
        if [ -n "$branch" ]; then
            git clone --branch "$branch" --single-branch "$url" "$dest"
        else
            git clone "$url" "$dest"
        fi

        if [ -n "$commit" ]; then
            echo "[checkout] $name → $commit"
            git -C "$dest" checkout "$commit"
        fi
    fi
}

# ---- Repositories ----

clone_if_missing "RLBench" \
    "$RLBENCH_REPO_URL" \
    "main"

clone_if_missing "lerobot" \
    "$LEROBOT_REPO_URL" \
    "" \
    "$LEROBOT_COMMIT"

echo ""
echo "All external dependencies are ready in $EXT_DIR"

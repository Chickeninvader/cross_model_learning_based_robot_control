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

mkdir -p "$EXT_DIR"

clone_if_missing() {
    local name="$1"
    local url="$2"
    local dest="$EXT_DIR/$name"

    if [ -d "$dest" ]; then
        echo "[skip] $name already exists at $dest"
    else
        echo "[clone] $name → $dest"
        git clone "$url" "$dest"
    fi
}

# ---- Repositories ----

clone_if_missing "RLBench" \
    "$RLBENCH_REPO_URL"

clone_if_missing "lerobot" \
    "https://github.com/huggingface/lerobot.git"

echo ""
echo "All external dependencies are ready in $EXT_DIR"

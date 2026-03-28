#!/usr/bin/env bash
# ============================================================
# Clone external dependencies into external/
# Run from the project root:
#   bash scripts/setup_external.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export REPO_ROOT
# shellcheck source=../lib/init_script_logging.sh
source "${SCRIPT_DIR}/../lib/init_script_logging.sh"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXT_DIR="$ROOT_DIR/external"

mkdir -p "$EXT_DIR"

clone_if_missing() {
    local name="$1"
    local url="$2"
    local commit="${3:-}"
    local dest="$EXT_DIR/$name"

    if [ -d "$dest" ]; then
        echo "[skip] $name already exists at $dest"
    else
        echo "[clone] $name → $dest"
        git clone "$url" "$dest"
        if [ -n "$commit" ]; then
            echo "[checkout] $name → $commit"
            git -C "$dest" checkout "$commit"
        fi
    fi
}

# ---- Repositories ----
clone_if_missing "OvSGTR" \
    "https://github.com/gpt4vision/OvSGTR.git" \
    "0a6984a4d2d8012fae0420aaa9c71bff8292eac0"

clone_if_missing "LASER" \
    "https://github.com/video-fm/LASER.git" \
    "1c4671ba07d26a5d8077b9ea27d6b697857916d3"

clone_if_missing "lang-segment-anything" \
    "https://github.com/luca-medeiros/lang-segment-anything.git" \
    ""  # latest main

clone_if_missing "lerobot" \
    "https://github.com/huggingface/lerobot.git" \
    "8fff0fde7c79f23a93d845d1a50e985de01f8b8a"

echo ""
echo "All external dependencies are ready in $EXT_DIR"

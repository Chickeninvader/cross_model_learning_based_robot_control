#!/usr/bin/env bash
# Tee stdout/stderr to:
#   ${REPO_ROOT}/output/script_logs/<path-to-script-under-repo>/<stem>/<timestamp>.log
#
# Usage (from a repo shell script, after set -euo pipefail):
#   SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
#   REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"   # or WORKSPACE_ROOT — see below
#   # shellcheck source=scripts/lib/init_script_logging.sh
#   source "${SCRIPT_DIR}/../lib/init_script_logging.sh"
#
# Repo root resolution: REPO_ROOT, then WORKSPACE_ROOT, then ROOT_DIR, else infer from this file
# (scripts/lib -> repo root).
#
# Opt out: SCRIPT_LOG_DISABLE=1
# Custom log root: SCRIPT_LOG_ROOT=/path/to/logs (still uses relative subdirs + filename pattern)

[[ -n "${SCRIPT_LOG_INIT:-}" ]] && return 0

_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_INFERRED_REPO="$(cd -- "${_LIB_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${REPO_ROOT:-${WORKSPACE_ROOT:-${ROOT_DIR:-$_INFERRED_REPO}}}" && pwd)"

_CALLER="${BASH_SOURCE[1]:-}"
if [[ -z "${_CALLER}" ]]; then
  echo "init_script_logging.sh: must be sourced from another bash script" >&2
  return 1
fi

if [[ "${SCRIPT_LOG_DISABLE:-0}" == "1" ]]; then
  SCRIPT_LOG_INIT=1
  return 0
fi

_CALLER_ABS="$(cd -- "$(dirname -- "${_CALLER}")" && pwd)/$(basename -- "${_CALLER}")"
if [[ "${_CALLER_ABS}" == "${REPO_ROOT}"/* ]]; then
  _REL="${_CALLER_ABS#"${REPO_ROOT}"/}"
else
  _REL="_outside_repo/$(basename -- "${_CALLER_ABS}")"
fi

_REL_DIR="$(dirname -- "${_REL}")"
if [[ "${_REL_DIR}" == "." ]]; then
  _REL_DIR=""
fi
_STEM="$(basename -- "${_CALLER_ABS}")"
_STEM="${_STEM%.sh}"
_TS="$(date +%Y%m%d_%H%M%S)"
_LOG_ROOT="${SCRIPT_LOG_ROOT:-${REPO_ROOT}/output/script_logs}"
if [[ -n "${_REL_DIR}" ]]; then
  _LOG_DIR="${_LOG_ROOT}/${_REL_DIR}/${_STEM}"
else
  _LOG_DIR="${_LOG_ROOT}/${_STEM}"
fi
mkdir -p "${_LOG_DIR}"
SCRIPT_LOG_FILE="${_LOG_DIR}/${_TS}.log"
export SCRIPT_LOG_FILE

echo "Script log file: ${SCRIPT_LOG_FILE}" >&2

exec > >(tee -a "${SCRIPT_LOG_FILE}") 2>&1
SCRIPT_LOG_INIT=1

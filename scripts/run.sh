#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper used by prior automation/instructions.
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_NAME="${CONTAINER_NAME:-quick-tts-endpoint}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_PORT="$HOST_PORT" CONTAINER_NAME="$CONTAINER_NAME" "${SCRIPT_DIR}/docker-run.sh"

#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper used by prior automation/instructions.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/docker-build.sh"

#!/bin/bash
set -euo pipefail

python3 -m venv ~/qwen-tts-venv
source ~/qwen-tts-venv/bin/activate

# Keep single worker by default to avoid loading multiple GPU model copies.
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"

uvicorn qwen_speak_server:app --host 0.0.0.0 --port 8765 --workers "${UVICORN_WORKERS}"

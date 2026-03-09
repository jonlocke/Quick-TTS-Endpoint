#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-quick-tts-endpoint}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-quick-tts-endpoint}"
HOST_PORT="${HOST_PORT:-8765}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_WAV="${QWEN_TRAIN_WAV:-${REPO_ROOT}/train.wav}"
TRAIN_TXT="${QWEN_TRAIN_TXT:-${REPO_ROOT}/train.txt}"

MOUNTS=()
if [[ -f "$TRAIN_WAV" ]]; then
  MOUNTS+=("-v" "${TRAIN_WAV}:/app/train.wav:ro" "-e" "QWEN_TRAIN_WAV=/app/train.wav")
fi
if [[ -f "$TRAIN_TXT" ]]; then
  MOUNTS+=("-v" "${TRAIN_TXT}:/app/train.txt:ro" "-e" "QWEN_TRAIN_TXT=/app/train.txt")
fi

docker run --rm \
  --gpus all \
  --name "$CONTAINER_NAME" \
  -p "${HOST_PORT}:8765" \
  "${MOUNTS[@]}" \
  -e QWEN_GPU_SYNTH_CONCURRENCY="${QWEN_GPU_SYNTH_CONCURRENCY:-1}" \
  -e QWEN_FORCE_FP32="${QWEN_FORCE_FP32:-1}" \
  -e QWEN_CUDA_CACHE_CLEAR_POLICY="${QWEN_CUDA_CACHE_CLEAR_POLICY:-pressure}" \
  -e QWEN_CUDA_CACHE_PRESSURE_THRESHOLD="${QWEN_CUDA_CACHE_PRESSURE_THRESHOLD:-0.98}" \
  "${IMAGE_NAME}:${IMAGE_TAG}"

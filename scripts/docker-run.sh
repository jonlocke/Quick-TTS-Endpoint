#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-quick-tts-endpoint}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-quick-tts-endpoint}"
HOST_PORT="${HOST_PORT:-8765}"
HEALTH_PATH="${HEALTH_PATH:-/health}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-120}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH." >&2
  exit 1
fi

TRAIN_WAV="${QWEN_TRAIN_WAV:-${REPO_ROOT}/train.wav}"
TRAIN_TXT="${QWEN_TRAIN_TXT:-${REPO_ROOT}/train.txt}"

MOUNTS=()
if [[ -f "$TRAIN_WAV" ]]; then
  MOUNTS+=("-v" "${TRAIN_WAV}:/app/train.wav:ro" "-e" "QWEN_TRAIN_WAV=/app/train.wav")
fi
if [[ -f "$TRAIN_TXT" ]]; then
  MOUNTS+=("-v" "${TRAIN_TXT}:/app/train.txt:ro" "-e" "QWEN_TRAIN_TXT=/app/train.txt")
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "Removing existing container ${CONTAINER_NAME}"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Starting ${CONTAINER_NAME} on port ${HOST_PORT}"
CONTAINER_ID="$(docker run -d \
  --gpus all \
  --name "$CONTAINER_NAME" \
  -p "${HOST_PORT}:8765" \
  "${MOUNTS[@]}" \
  -e QWEN_GPU_SYNTH_CONCURRENCY="${QWEN_GPU_SYNTH_CONCURRENCY:-1}" \
  -e QWEN_FORCE_FP32="${QWEN_FORCE_FP32:-1}" \
  -e QWEN_CUDA_CACHE_CLEAR_POLICY="${QWEN_CUDA_CACHE_CLEAR_POLICY:-pressure}" \
  -e QWEN_CUDA_CACHE_PRESSURE_THRESHOLD="${QWEN_CUDA_CACHE_PRESSURE_THRESHOLD:-0.98}" \
  "${IMAGE_NAME}:${IMAGE_TAG}")"

echo "Container started successfully: ${CONTAINER_NAME} (${CONTAINER_ID})"
HEALTH_URL="http://localhost:${HOST_PORT}${HEALTH_PATH}"
echo "Health: ${HEALTH_URL}"

for ((i=1; i<=HEALTH_TIMEOUT_SEC; i++)); do
  if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
    echo "Health check passed"
    exit 0
  fi

  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || echo false)" != "true" ]]; then
    echo "Container exited before becoming healthy. Recent logs:"
    docker logs "$CONTAINER_NAME" --tail 200 || true
    exit 1
  fi
  sleep 1
done

echo "Health check timed out after ${HEALTH_TIMEOUT_SEC}s. Recent logs:"
docker logs "$CONTAINER_NAME" --tail 200 || true
exit 1

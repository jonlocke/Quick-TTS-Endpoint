#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-quick-tts-endpoint}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-quick-tts-endpoint}"
HOST_PORT="${HOST_PORT:-8765}"
HEALTH_PATH="${HEALTH_PATH:-/health}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-120}"

PERSIST_MODEL_CACHE="${PERSIST_MODEL_CACHE:-1}"
MODEL_CACHE_VOLUME="${MODEL_CACHE_VOLUME:-quick-tts-hf-cache}"
HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-/root/.cache/huggingface}"

# GPU flag control:
# - auto (default): require NVIDIA runtime and pass --gpus all.
# - on: always force --gpus all.
# - off: never pass --gpus (only for custom CPU-capable images).
DOCKER_GPU_MODE="${DOCKER_GPU_MODE:-auto}"

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

if [[ "${PERSIST_MODEL_CACHE}" != "0" && "${PERSIST_MODEL_CACHE}" != "1" ]]; then
  echo "Error: invalid PERSIST_MODEL_CACHE='${PERSIST_MODEL_CACHE}'. Use: 0 or 1." >&2
  exit 1
fi

if [[ "${PERSIST_MODEL_CACHE}" == "1" ]]; then
  MOUNTS+=("-v" "${MODEL_CACHE_VOLUME}:${HUGGINGFACE_CACHE_DIR}")
fi

GPU_ARGS=()
case "$DOCKER_GPU_MODE" in
  on)
    GPU_ARGS=(--gpus all)
    ;;
  off)
    GPU_ARGS=()
    echo "Warning: DOCKER_GPU_MODE=off disables GPU flags; default image may fail because server enforces CUDA." >&2
    ;;
  auto)
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi 'nvidia'; then
      GPU_ARGS=(--gpus all)
    else
      echo "Error: NVIDIA Docker runtime not detected." >&2
      echo "This image requires CUDA and will fail to start without GPU support." >&2
      echo "Install/configure NVIDIA Container Toolkit, then retry." >&2
      echo "If you are using a custom CPU-capable image, set DOCKER_GPU_MODE=off explicitly." >&2
      exit 1
    fi
    ;;
  *)
    echo "Error: invalid DOCKER_GPU_MODE='${DOCKER_GPU_MODE}'. Use: auto, on, or off." >&2
    exit 1
    ;;
esac

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "Removing existing container ${CONTAINER_NAME}"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Starting ${CONTAINER_NAME} on port ${HOST_PORT}"
if [[ "${PERSIST_MODEL_CACHE}" == "1" ]]; then
  echo "Persistent model cache volume: ${MODEL_CACHE_VOLUME} -> ${HUGGINGFACE_CACHE_DIR}"
fi
CONTAINER_ID="$(docker run -d \
  "${GPU_ARGS[@]}" \
  --name "$CONTAINER_NAME" \
  -p "${HOST_PORT}:8765" \
  "${MOUNTS[@]}" \
  -e QWEN_GPU_SYNTH_CONCURRENCY="${QWEN_GPU_SYNTH_CONCURRENCY:-1}" \
  -e QWEN_FORCE_FP32="${QWEN_FORCE_FP32:-1}" \
  -e QWEN_CUDA_CACHE_CLEAR_POLICY="${QWEN_CUDA_CACHE_CLEAR_POLICY:-pressure}" \
  -e QWEN_CUDA_CACHE_PRESSURE_THRESHOLD="${QWEN_CUDA_CACHE_PRESSURE_THRESHOLD:-0.98}" \
  -e HF_HOME="${HUGGINGFACE_CACHE_DIR}" \
  -e HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_CACHE_DIR}" \
  -e TRANSFORMERS_CACHE="${HUGGINGFACE_CACHE_DIR}" \
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

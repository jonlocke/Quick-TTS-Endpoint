#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-quick-tts-endpoint}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.6.0-runtime-ubuntu24.04}"
TORCH_VERSION="${TORCH_VERSION:-2.2.2+cu118}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.17.2+cu118}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.2.2+cu118}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH." >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "Building ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Base image: ${BASE_IMAGE}"
echo "PyTorch: ${TORCH_VERSION} / torchvision: ${TORCHVISION_VERSION} / torchaudio: ${TORCHAUDIO_VERSION}"
docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION}" \
  --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}" \
  --build-arg "TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  .

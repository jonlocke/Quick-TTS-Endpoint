#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-$HOME/qwen-tts-venv}"
APPDIR="${APPDIR:-$HOME/Quick-TTS-Endpoint}"
INSTALL_TOOLKIT="${INSTALL_TOOLKIT:-1}"

TORCH_VERSION="${TORCH_VERSION:-2.2.2+cu118}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.17.2+cu118}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.2.2+cu118}"

if [[ "${INSTALL_TOOLKIT}" == "1" ]]; then
  echo "[0/6] NVIDIA Container Toolkit (Docker GPU runtime) ..."
  if command -v docker >/dev/null 2>&1; then
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi 'nvidia'; then
      echo "NVIDIA Docker runtime already detected; skipping toolkit install."
    else
      echo "Installing NVIDIA Container Toolkit for Docker..."
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
      sudo apt-get update
      sudo apt-get install -y nvidia-container-toolkit
      sudo nvidia-ctk runtime configure --runtime=docker
      sudo systemctl restart docker
      echo "Toolkit install/configuration complete."
    fi
  else
    echo "Docker not found; skipping NVIDIA Container Toolkit install." >&2
    echo "Install Docker first, then rerun with INSTALL_TOOLKIT=1." >&2
  fi
fi

echo "[1/6] System dependencies (ffmpeg, sox)..."
sudo apt-get update
sudo apt-get install -y ffmpeg sox libsox-fmt-all

echo "[2/6] Create venv (if missing) + upgrade pip tooling..."
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools

echo "[3/6] Core Python deps + NumPy<2..."
pip install -U fastapi uvicorn soundfile "numpy<2"

echo "[4/6] PyTorch CUDA wheels from cu118 index..."
pip install -U \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url https://download.pytorch.org/whl/cu118

echo "[5/6] Qwen TTS stack (known working set via PyPI)..."
pip install -U "qwen-tts==0.1.0"

echo
 echo "=== Sanity checks ==="
python - <<'PY'
import numpy as np
import torch
print("numpy", np.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cuda", torch.version.cuda)
    x = torch.randn(3, device="cuda")
    y = x.cpu().numpy()
    print("numpy bridge OK", y.shape, y.dtype)
else:
    print("CUDA unavailable; check NVIDIA drivers + toolkit runtime setup.")
PY

echo
echo "[6/6] Key pins"
pip freeze | egrep '^(qwen-tts|transformers|accelerate|huggingface[_-]hub|tokenizers|safetensors|torch|torchvision|torchaudio|numpy)=' || true

echo
echo "Manual NVIDIA Container Toolkit commands (Ubuntu/Debian):"
echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
echo "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null"
echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
echo "  sudo nvidia-ctk runtime configure --runtime=docker"
echo "  sudo systemctl restart docker"
echo "  docker info --format '{{json .Runtimes}}' | grep -i nvidia"

echo
echo "NOTE:"
echo "- Ensure your server imports the repo patch FIRST, before qwen_tts/transformers:"
echo "    import torch_pytree_patch"
echo "- If running under uvicorn from $APPDIR, ensure PYTHONPATH includes the repo:"
echo "    cd $APPDIR && PYTHONPATH=. uvicorn qwen_speak_server:app --host 0.0.0.0 --port 8765"

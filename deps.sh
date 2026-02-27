#!/usr/bin/env bash
set -euo pipefail

VENV="$HOME/qwen-tts-venv"
APPDIR="$HOME/Quick-TTS-Endpoint"

echo "[1/5] System dependencies (ffmpeg, sox)..."
sudo apt-get update
sudo apt-get install -y ffmpeg sox libsox-fmt-all

echo "[2/5] Create venv (if missing) + upgrade pip tooling..."
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools

echo "[3/5] Core Python deps + NumPy<2..."
pip install -U fastapi uvicorn soundfile
pip install -U "numpy<2"

echo "[4/5] PyTorch CUDA build for GTX 1080 (sm_61) ..."
pip install -U \
  torch==2.1.2+cu118 \
  torchvision==0.16.2+cu118 \
  torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

echo "[5/5] Qwen TTS stack (known working set via PyPI)..."
pip install -U "qwen-tts==0.1.0"

echo
echo "=== Sanity checks ==="
python - <<'EOF'
import numpy as np
import torch
print("numpy", np.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.cuda)

x = torch.randn(3, device="cuda")
y = x.cpu().numpy()
print("numpy bridge OK", y.shape, y.dtype)
EOF

echo
echo "=== Key pins ==="
pip freeze | egrep '^(qwen-tts|transformers|accelerate|huggingface[_-]hub|tokenizers|safetensors|torch|torchvision|torchaudio|numpy)=' || true

echo
echo "NOTE:"
echo "- Ensure your server imports the repo patch FIRST, before qwen_tts/transformers:"
echo "    import torch_pytree_patch"
echo "- If running under uvicorn from $APPDIR, ensure PYTHONPATH includes the repo:"
echo "    cd $APPDIR && PYTHONPATH=. uvicorn qwen_speak_server:app --host 0.0.0.0 --port 8765"
echo "- Don’t forget to port forward / expose the endpoint as needed (see fwd.PS1 on WSL)."

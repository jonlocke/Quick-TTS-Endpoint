#!/bin/bash
sudo apt-get update && sudo apt-get install -y ffmpeg
python3 -m venv ~/qwen-tts-venv
source ~/qwen-tts-venv/bin/activate
pip install -U pip wheel
pip install fastapi uvicorn soundfile 
pip install pip install -U "numpy<2" numpy
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118  # For Pascal GTX1080
#pip install torch --index-url https://download.pytorch.org/whl/cu118 # for Newer cards
pip install -U "git+https://github.com/QwenLM/Qwen3-TTS.git"
sudo apt-get update
sudo apt-get install -y sox libsox-fmt-all
python - <<'EOF'
import numpy as np
import torch
print("numpy", np.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.cuda)
# test numpy bridge
x = torch.randn(3, device="cuda")
y = x.cpu().numpy()
print("numpy bridge OK", y.shape, y.dtype)
EOF
pip freeze | egrep '^(torch|torchvision|torchaudio|numpy)='
echo "Dont forget to port forward or make the 8765 endpoint available check fwd.PS1 on WSL"

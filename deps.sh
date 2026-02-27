#!/bin/bash
sudo apt-get update && sudo apt-get install -y ffmpeg
python3 -m venv ~/qwen-tts-venv
source ~/qwen-tts-venv/bin/activate
pip install -U pip wheel
pip install fastapi uvicorn soundfile numpy
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -U "git+https://github.com/QwenLM/Qwen3-TTS.git"
sudo apt-get update
sudo apt-get install -y sox libsox-fmt-all


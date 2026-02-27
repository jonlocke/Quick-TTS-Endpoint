#!/bin/bash
python3 -m venv ~/qwen-tts-venv
source ~/qwen-tts-venv/bin/activate
uvicorn qwen_speak_server:app --host 0.0.0.0 --port 8765


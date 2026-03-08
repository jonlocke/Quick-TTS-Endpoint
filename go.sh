#!/bin/bash
set -euo pipefail

python3 -m venv ~/qwen-tts-venv
source ~/qwen-tts-venv/bin/activate
export QWEN_TRAIN_TXT="HelBonCart.txt"
export QWEN_TRAIN_WAV="HelBonCart.wav"
export QWEN_GPU_SYNTH_CONCURRENCY=1
export QWEN_FORCE_FP32=1 # work around gtx1080 fp16 prb, on ++ models try fp16 for perf. imprv.
export QWEN_WAV_ENCODER_THREADS=4
export QWEN_CUDA_EMPTY_CACHE_EACH_CHUNK=1 # Prevent exhaustion of GPU memory
export QWEN_CUDA_CACHE_CLEAR_POLICY=pressure
export QWEN_CUDA_CACHE_PRESSURE_THRESHOLD=0.98 # Point at which GPU compact occurs in pressure policy
#export QWEN_SYNTH_DISPATCH_THREADS=1 # Deprecated
# Keep single worker by default to avoid loading multiple GPU model copies.
UVICORN_WORKERS="${UVICORN_WORKERS:-1}"

uvicorn qwen_speak_server:app --host 0.0.0.0 --port 8765 --workers "${UVICORN_WORKERS}"

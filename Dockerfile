# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=nvidia/cuda:12.6.0-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    ffmpeg \
    sox \
    libsox-fmt-all \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" && \
    python -m pip install --upgrade pip wheel setuptools

ARG TORCH_VERSION=2.2.2+cu118
ARG TORCHVISION_VERSION=0.17.2+cu118
ARG TORCHAUDIO_VERSION=2.2.2+cu118

RUN python -m pip install --upgrade fastapi uvicorn soundfile "numpy<2" && \
    python -m pip install --upgrade \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
      --index-url https://download.pytorch.org/whl/cu118 && \
    python -m pip install --upgrade qwen-tts==0.1.0
COPY . /app

EXPOSE 8765

ENV QWEN_GPU_SYNTH_CONCURRENCY=1 \
    QWEN_FORCE_FP32=1 \
    QWEN_WAV_ENCODER_THREADS=4 \
    QWEN_CUDA_CACHE_CLEAR_POLICY=pressure \
    QWEN_CUDA_CACHE_PRESSURE_THRESHOLD=0.98 \
    UVICORN_WORKERS=1

CMD ["uvicorn", "qwen_speak_server:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1"]

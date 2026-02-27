#!/usr/bin/env python3
"""
Qwen3-TTS local /speak server for AImaster (qwen-tts)

- POST /speak {"text":"..."}  -> by default: queues playback (FIFO) and returns JSON
  Query params:
    play=1|0         (default 1)  queue server-side playback via ffplay
    return_audio=1|0 (default 0)  return audio/wav bytes (first chunk) instead of JSON
    chunk=1|0        (default 1)  split into sentence-ish chunks for earlier playback
    max_chars=N      (default 240) chunk size cap

- GET /speakers
- GET /languages
- GET /health

Designed for GTX 1080 stability:
- Uses FP32 by default (prevents NaN/probability asserts seen with FP16 on Pascal)
- Forces greedy decoding on internal HF components (best-effort)

Env overrides:
  QWEN_TTS_MODEL      default: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
  QWEN_SPEAKER        default: ryan
  QWEN_LANG           default: english
  QWEN_INSTRUCT       default: "Read the text clearly, naturally, and conversationally."
  QWEN_AUTOMIX_FP32   default: 0  (kept for compatibility; model dtype defaults to fp32 anyway)
  QWEN_FORCE_FP32     default: 1  (set 0 to try fp16 again)
  QWEN_PLAY_Q_MAX     default: 100

Deps (WSL):
  sudo apt-get install -y ffmpeg sox libsox-fmt-all
  pip install fastapi uvicorn soundfile numpy torch qwen-tts
"""

import io
import os
import re
import tempfile
import subprocess
import threading
import queue
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

import numpy as np
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


# ----------------------------
# Configuration
# ----------------------------
MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# GTX 1080 stability: default to fp32 on CUDA; allow override if you upgrade GPU.
FORCE_FP32 = os.environ.get("QWEN_FORCE_FP32", "1").strip() in ("1", "true", "TRUE", "yes", "YES", "on", "ON")
if DEVICE.startswith("cuda"):
    DTYPE = torch.float32 if FORCE_FP32 else torch.float16
else:
    DTYPE = torch.float32

# Optional: compute in fp32 while keeping fp16 weights (extra stability) - mostly irrelevant if DTYPE=fp32
AUTOMIX_FP32 = os.environ.get("QWEN_AUTOMIX_FP32", "0").strip() in ("1", "true", "TRUE", "yes", "YES", "on", "ON")

# Allow TF32 (helps newer GPUs; harmless on most)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEFAULT_SPEAKER = os.environ.get("QWEN_SPEAKER", "ryan").strip().lower()
DEFAULT_LANG = os.environ.get("QWEN_LANG", "english").strip().lower()
DEFAULT_INSTRUCT = os.environ.get(
    "QWEN_INSTRUCT",
    "Read the text clearly, naturally, and conversationally.",
).strip()

# Generation kwargs (best effort; still force greedy via generation_config)
GEN_KWARGS = {
    "do_sample": False,
    "num_beams": 1,
    "temperature": 1.0,
    "top_p": 1.0,
}

PRINT_PREFIX = "[qwen-speak]"

app = FastAPI()


class SpeakReq(BaseModel):
    text: str
    speaker: Optional[str] = None
    language: Optional[str] = None
    instruct: Optional[str] = None


def force_greedy(obj, label: str):
    """
    Best-effort: disable sampling on any HF model with a generation_config.
    Prevents multinomial sampling path that can trigger CUDA asserts.
    """
    try:
        if hasattr(obj, "generation_config") and obj.generation_config is not None:
            gc = obj.generation_config
            gc.do_sample = False
            gc.num_beams = 1
            gc.temperature = 1.0
            gc.top_p = 1.0
            if hasattr(gc, "top_k"):
                gc.top_k = 0
            print(f"{PRINT_PREFIX} forced greedy on {label}")
    except Exception as e:
        print(f"{PRINT_PREFIX} could not force greedy on {label}: {e}")


print(f"{PRINT_PREFIX} loading {MODEL_ID} on {DEVICE} dtype={DTYPE}")
model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=DTYPE)

# Force greedy decoding on internal components (critical for stability)
try:
    force_greedy(model.model, "model.model")
    for sub_name in ("talker", "code_predictor"):
        if hasattr(model.model, sub_name):
            force_greedy(getattr(model.model, sub_name), f"model.model.{sub_name}")
except Exception as e:
    print(f"{PRINT_PREFIX} warning: could not apply greedy forcing: {e}")

# Supported lists (best effort)
try:
    SUPPORTED_LANGS = [x.strip().lower() for x in model.get_supported_languages()]
except Exception:
    SUPPORTED_LANGS = []

try:
    SUPPORTED_SPEAKERS = [x.strip().lower() for x in model.get_supported_speakers()]
except Exception:
    SUPPORTED_SPEAKERS = []


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "queue_depth": play_q.qsize(),
        "supported_speakers": len(SUPPORTED_SPEAKERS),
        "supported_languages": len(SUPPORTED_LANGS),
    }


@app.get("/languages")
def languages():
    return {"languages": SUPPORTED_LANGS}


@app.get("/speakers")
def speakers():
    return {"speakers": SUPPORTED_SPEAKERS}


def _concat_audio(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate a list of 1-D numpy arrays into one float32 array."""
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    fixed = []
    for c in chunks:
        a = np.asarray(c)
        if a.ndim > 1:
            a = a.squeeze()
        fixed.append(a.astype(np.float32, copy=False))
    return np.concatenate(fixed, axis=0)


def _chunk_text(text: str, max_chars: int = 240) -> list[str]:
    """Cheap sentence-ish splitter + length cap."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[\.\!\?])\s+", text)

    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        s = buf.strip()
        if s:
            chunks.append(s)
        buf = ""

    for p in parts:
        p = p.strip()
        if not p:
            continue

        # Hard wrap a single overlong sentence
        if len(p) > max_chars:
            if buf:
                flush()
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i + max_chars].strip())
            continue

        candidate = (buf + " " + p).strip() if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            flush()
            buf = p

    if buf:
        flush()

    # Merge tiny trailing bits
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < 30:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)
    return merged


def _synthesize_to_wav_bytes(text: str, speaker: str, language: str, instruct: str) -> Tuple[bytes, int]:
    with torch.inference_mode():
        if DEVICE.startswith("cuda") and AUTOMIX_FP32 and DTYPE == torch.float16:
            with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
                audio_list, sr = model.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                    non_streaming_mode=True,
                    **GEN_KWARGS,
                )
        else:
            audio_list, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                non_streaming_mode=True,
                **GEN_KWARGS,
            )

    wav = _concat_audio(audio_list)
    if wav.size == 0:
        raise RuntimeError("Empty audio output from model")

    buf = io.BytesIO()
    sf.write(buf, wav, int(sr), format="WAV")
    return buf.getvalue(), int(sr)


# ----------------------------
# Playback queue (FIFO; prevents overlap)
# ----------------------------
@dataclass
class SpeakJob:
    wav_bytes: bytes
    job_id: str


PLAY_Q_MAX = int(os.environ.get("QWEN_PLAY_Q_MAX", "100"))
play_q: "queue.Queue[SpeakJob]" = queue.Queue(maxsize=PLAY_Q_MAX)
_worker_stop = threading.Event()


def _play_wav_bytes(wav_bytes: bytes) -> None:
    # Write to temp .wav and play via ffplay
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(wav_bytes)
        path = f.name
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            check=False,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _worker_loop():
    while not _worker_stop.is_set():
        try:
            job = play_q.get(timeout=0.25)
        except queue.Empty:
            continue
        try:
            _play_wav_bytes(job.wav_bytes)
        finally:
            play_q.task_done()


_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()


@app.on_event("shutdown")
def _shutdown():
    _worker_stop.set()


# ----------------------------
# Main endpoint
# ----------------------------
@app.post("/speak")
def speak(
    req: SpeakReq,
    play: bool = Query(True, description="If true (default), queue server-side playback via ffplay"),
    return_audio: bool = Query(False, description="If true, return audio/wav bytes (first chunk) instead of JSON"),
    chunk: bool = Query(True, description="If true (default), split into sentence-ish chunks"),
    max_chars: int = Query(240, ge=80, le=800, description="Chunk size cap"),
):
    text = (req.text or "").strip()
    if not text:
        return Response(status_code=204)

    speaker = (req.speaker or DEFAULT_SPEAKER).strip().lower()
    language = (req.language or DEFAULT_LANG).strip().lower()
    instruct = (req.instruct or DEFAULT_INSTRUCT).strip()

    if SUPPORTED_SPEAKERS and speaker not in SUPPORTED_SPEAKERS:
        raise HTTPException(status_code=400, detail=f"Unsupported speaker '{speaker}'. Use GET /speakers")
    if SUPPORTED_LANGS and language not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{language}'. Use GET /languages")

    parts = _chunk_text(text, max_chars=max_chars) if chunk else [text]
    if not parts:
        return Response(status_code=204)

    job_ids = []
    first_wav = None
    first_sr = None

    try:
        for idx, part in enumerate(parts):
            wav_bytes, sr = _synthesize_to_wav_bytes(part, speaker, language, instruct)
            jid = f"{os.getpid()}-{threading.get_ident()}-{idx}"
            job_ids.append(jid)

            if play:
                try:
                    play_q.put_nowait(SpeakJob(wav_bytes=wav_bytes, job_id=jid))
                except queue.Full:
                    raise HTTPException(status_code=429, detail="Playback queue is full")

            if return_audio and first_wav is None:
                first_wav, first_sr = wav_bytes, sr

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    if return_audio and first_wav is not None:
        return Response(
            content=first_wav,
            media_type="audio/wav",
            headers={
                "X-Qwen-Speaker": speaker,
                "X-Qwen-Language": language,
                "X-Qwen-SampleRate": str(first_sr or ""),
                "X-Qwen-Chunks": str(len(parts)),
                "X-Qwen-Chunked": "1" if chunk else "0",
            },
        )

    return JSONResponse({
        "ok": True,
        "queued": bool(play),
        "return_audio": bool(return_audio),
        "chunked": bool(chunk),
        "chunks": len(parts),
        "max_chars": max_chars,
        "speaker": speaker,
        "language": language,
        "queue_depth": play_q.qsize(),
        "job_ids": job_ids[:10],  # avoid huge responses
    })


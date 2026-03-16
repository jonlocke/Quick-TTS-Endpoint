#!/usr/bin/env python3
"""
Qwen3-TTS local /speak server for AImaster (qwen-tts)

- POST /speak {"text":"..."}  -> by default: queues playback (FIFO) and returns JSON
  Query params:
    play=1|0         (default 1)  queue server-side playback via ffplay
    return_audio=1|0 (default 0)  return audio/wav bytes (all chunks concatenated) instead of JSON
    chunk=1|0        (default 1)  split into sentence-ish chunks for earlier playback
    max_chars=N      (default 240) chunk size cap

- GET /speakers
- GET /languages
- GET /health

Designed for GTX 1080 stability:
- Uses FP32 by default (prevents NaN/probability asserts seen with FP16 on Pascal)
- Forces greedy decoding on internal HF components (best-effort)

Env overrides:
  QWEN_TTS_MODEL      default: Qwen/Qwen3-TTS-12Hz-0.6B-Base
  QWEN_SPEAKER        default: ryan
  QWEN_LANG           default: english
  QWEN_INSTRUCT       default: "Read the text clearly, naturally, and conversationally."
  QWEN_AUTOMIX_FP32   default: 0  (kept for compatibility; model dtype defaults to fp32 anyway)
  QWEN_FORCE_FP32     default: 1  (set 0 to try fp16 again)
  QWEN_PLAY_Q_MAX     default: 100
  QWEN_WAV_ENCODER_THREADS default: 2 (CPU pool for WAV encoding)
  QWEN_GPU_SYNTH_CONCURRENCY default: 1 (max concurrent synth calls; can increase activation memory)
  QWEN_FP16_RETRY_FP32 default: 1 (retry failed fp16 synth in fp32 autocast-off mode)
  QWEN_FORCE_PIPER default: 0 (route all /speak synthesis through piper HTTP backend)
  PIPER_MODEL default: en_US-lessac-medium
  PIPER_HTTP_URL default: http://wyoming-piper:10200/api/tts
  PIPER_HTTP_TIMEOUT_SECONDS default: 120
  PIPER_SPEAKER optional speaker id forwarded to piper HTTP backend
import urllib.error
import urllib.parse
import urllib.request
  QWEN_BUSY_FALLBACK_PIPER default: 1 (if busy, synthesize via piper instead of returning busy)
  QWEN_FORCE_PIPER default: 0 (route all /speak synthesis through piper)
  PIPER_BIN default: piper
  PIPER_MODEL default: en_US-lessac-medium
  PIPER_CONFIG optional path to piper model config
  PIPER_SPEAKER optional speaker id for multi-speaker piper models
  PIPER_DOCKER_CONTAINER default: "" (optional; when set, run piper via docker exec into existing container)
  PIPER_DOCKER_EXEC_REQUIRED default: 1 (if 1, fail when docker-exec is unavailable; if 0, fallback to local piper bin)
  (when piper backend is unavailable, busy fallback is skipped and busy response is preserved)

Deps (WSL):
  sudo apt-get install -y ffmpeg sox libsox-fmt-all
  pip install fastapi uvicorn soundfile numpy torch qwen-tts
"""
import io
import base64
import json
import concurrent.futures
import inspect
import gc
import os
import re
import subprocess
import shutil
import threading
import time
import queue
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel

import numpy as np
import torch
import soundfile as sf
import torch_pytree_patch

from qwen_tts import Qwen3TTSModel


# ----------------------------
# Configuration
# ----------------------------
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_ID_ALIASES = {
    # Common shorthand people use in docs/shell profiles.
    "qwen3-tts": DEFAULT_MODEL_ID,
    "qwen-tts": DEFAULT_MODEL_ID,
    "qwen3_tts": DEFAULT_MODEL_ID,
}


def _resolve_model_id(raw_model_id: str) -> str:
    candidate = raw_model_id.strip()
    if not candidate:
        return DEFAULT_MODEL_ID

    alias_hit = MODEL_ID_ALIASES.get(candidate.lower())
    if alias_hit:
        print(f"[qwen-speak] startup: normalized QWEN_TTS_MODEL='{candidate}' -> '{alias_hit}'", flush=True)
        return alias_hit
    return candidate


MODEL_ID = _resolve_model_id(os.environ.get("QWEN_TTS_MODEL", DEFAULT_MODEL_ID))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available, refusing CPU fallback")
DEVICE = "cuda:0"

# GTX 1080 stability: default to fp32 on CUDA; allow override if you upgrade GPU.
FORCE_FP32 = os.environ.get("QWEN_FORCE_FP32", "1").strip() in ("1", "true", "TRUE", "yes", "YES", "on", "ON")
if DEVICE.startswith("cuda"):
    DTYPE = torch.float32 if FORCE_FP32 else torch.float16
else:
    DTYPE = torch.float32

# Optional: compute in fp32 while keeping fp16 weights (extra stability) - mostly irrelevant if DTYPE=fp32
AUTOMIX_FP32 = os.environ.get("QWEN_AUTOMIX_FP32", "0").strip() in ("1", "true", "TRUE", "yes", "YES", "on", "ON")
FP16_RETRY_FP32 = os.environ.get("QWEN_FP16_RETRY_FP32", "1").strip() in ("1", "true", "TRUE", "yes", "YES", "on", "ON")
EMPTY_CACHE_EACH_CHUNK = os.environ.get("QWEN_CUDA_EMPTY_CACHE_EACH_CHUNK", "0").strip() in (
    "1", "true", "TRUE", "yes", "YES", "on", "ON"
)
CUDA_CACHE_CLEAR_POLICY = os.environ.get("QWEN_CUDA_CACHE_CLEAR_POLICY", "").strip().lower()
if not CUDA_CACHE_CLEAR_POLICY:
    CUDA_CACHE_CLEAR_POLICY = "chunk" if EMPTY_CACHE_EACH_CHUNK else "off"
CUDA_CACHE_PRESSURE_THRESHOLD = float(os.environ.get("QWEN_CUDA_CACHE_PRESSURE_THRESHOLD", "0.92"))

# Allow TF32 (helps newer GPUs; harmless on most)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

QWEN_FORCE_PIPER = os.environ.get("QWEN_FORCE_PIPER", "0").strip().lower() in ("1", "true", "yes", "on")
PIPER_MODEL = os.environ.get("PIPER_MODEL", "en_US-lessac-medium").strip() or "en_US-lessac-medium"
PIPER_HTTP_URL = os.environ.get("PIPER_HTTP_URL", "http://wyoming-piper:10200/api/tts").strip() or "http://wyoming-piper:10200/api/tts"
_piper_http_timeout_raw = (os.environ.get("PIPER_HTTP_TIMEOUT_SECONDS", "120") or "120").strip()
try:
    PIPER_HTTP_TIMEOUT_SECONDS = max(1.0, float(_piper_http_timeout_raw))
except Exception:
    PIPER_HTTP_TIMEOUT_SECONDS = 120.0
PIPER_SPEAKER = os.environ.get("PIPER_SPEAKER", "").strip() or None

def _synthesize_with_piper_http(text: str) -> Tuple[np.ndarray, int]:
    payload = {"text": text, "voice": PIPER_MODEL}
    if PIPER_SPEAKER:
        payload["speaker"] = PIPER_SPEAKER
    configured = (PIPER_HTTP_URL or "").strip()
    base_url = configured.rstrip("/")
    candidate_urls = [configured]
    if configured and (configured.endswith(":10200") or configured.endswith(":10200/")):
        candidate_urls = [base_url + "/api/tts", base_url]

    last_error = ""
    wav_bytes = b""
    for piper_url in candidate_urls:
        req = urllib.request.Request(
            piper_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=max(1.0, PIPER_HTTP_TIMEOUT_SECONDS)) as resp:
                wav_bytes = resp.read()
                if wav_bytes:
                    break
                last_error = f"Piper HTTP returned empty audio from {piper_url}"
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")[-300:]
def _piper_candidate_urls(configured_url: str) -> list[str]:
    raw = (configured_url or "").strip() or "http://wyoming-piper:10200/api/tts"
    if "://" not in raw:
        raw = "http://" + raw

    parsed = urllib.parse.urlparse(raw)
    path = (parsed.path or "").rstrip("/")
    base = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else raw.rstrip("/")

    if not path:
        return [f"{base}/api/tts", base]
    if path == "/api/tts":
        return [f"{base}/api/tts", f"{base}/api/tts/"]
    return [raw]


def _decode_piper_response_bytes(raw_bytes: bytes) -> bytes:
    if not raw_bytes:
        return b""
    try:
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        return raw_bytes

    for key in ("audio", "audio_base64", "wav_base64"):
        value = decoded.get(key)
        if isinstance(value, str) and value:
            try:
                return base64.b64decode(value)
            except Exception:
                continue
    return raw_bytes


    timeout = max(1.0, PIPER_HTTP_TIMEOUT_SECONDS)
    payloads = [{"text": text, "voice": PIPER_MODEL}, {"text": text, "model": PIPER_MODEL}, {"text": text}]
        payloads[0]["speaker"] = PIPER_SPEAKER
    if PIPER_SPEAKER is not None:
    for piper_url in _piper_candidate_urls(PIPER_HTTP_URL):
        for payload in payloads:
            req = urllib.request.Request(
                piper_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "audio/wav, application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    wav_bytes = _decode_piper_response_bytes(resp.read())
                    if wav_bytes:
                        break
                    last_error = f"Piper HTTP returned empty audio from {piper_url}"
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")[-300:]
                last_error = f"Piper HTTP request failed (url={piper_url} status={exc.code}): {body}"
                continue
            except urllib.error.URLError as exc:
                last_error = f"Piper HTTP backend unavailable (url={piper_url}): {exc.reason}"
                continue
        if wav_bytes:
            break
TRAIN_WAV_PATH = os.environ.get("QWEN_TRAIN_WAV", "train.wav").strip()
TRAIN_TXT_PATH = os.environ.get("QWEN_TRAIN_TXT", "train.txt").strip()
ALLOW_SPEAKER_WITH_TRAIN = os.environ.get("QWEN_ALLOW_SPEAKER_WITH_TRAIN", "0").strip().lower() in (
    "1", "true", "yes", "on"
)
FORCE_CUSTOM_SPEAKER = os.environ.get("QWEN_FORCE_CUSTOM_SPEAKER", "custom").strip()
PROMPT_AUDIO_ARG_OVERRIDE = os.environ.get("QWEN_PROMPT_AUDIO_ARG", "").strip()
PROMPT_TEXT_ARG_OVERRIDE = os.environ.get("QWEN_PROMPT_TEXT_ARG", "").strip()
REQUIRE_TRAINING_FILES = os.environ.get("QWEN_REQUIRE_TRAINING_FILES", "0").strip().lower() in ("1", "true", "yes", "on")

# Generation kwargs (best effort; still force greedy via generation_config)
GEN_KWARGS = {
    "do_sample": False,
    "num_beams": 1,
    "temperature": 1.0,
    "top_p": 1.0,
}

PRINT_PREFIX = "[qwen-speak]"

GPU_SYNTH_CONCURRENCY = max(1, int(os.environ.get("QWEN_GPU_SYNTH_CONCURRENCY", "1")))
CHUNK_GEN_TIMEOUT_SECONDS = float(os.environ.get("QWEN_CHUNK_GEN_TIMEOUT_SECONDS", "150"))
# Admission control: how long a request waits for a synth slot before fast-failing busy.
SYNTH_ACQUIRE_TIMEOUT_SECONDS = float(os.environ.get("QWEN_SYNTH_ACQUIRE_TIMEOUT_SECONDS", "0.25"))
SYNTH_BUSY_STATUS_CODE = int(os.environ.get("QWEN_SYNTH_BUSY_STATUS_CODE", "429"))
# Caps concurrent model inference calls per worker (primarily affects concurrent HTTP requests).
GPU_SYNTH_SEMAPHORE = threading.BoundedSemaphore(value=GPU_SYNTH_CONCURRENCY)

PIPER_BUSY_FALLBACK = os.environ.get("QWEN_BUSY_FALLBACK_PIPER", "1").strip().lower() in (
    "1", "true", "yes", "on"
)
PIPER_FORCE_REQUESTS = os.environ.get("QWEN_FORCE_PIPER", "0").strip().lower() in (
    "1", "true", "yes", "on"
)
PIPER_BIN = os.environ.get("PIPER_BIN", "piper").strip() or "piper"
PIPER_MODEL = os.environ.get("PIPER_MODEL", "en_US-lessac-medium").strip() or "en_US-lessac-medium"
PIPER_CONFIG = os.environ.get("PIPER_CONFIG", "").strip()
PIPER_SPEAKER = os.environ.get("PIPER_SPEAKER", "").strip()
PIPER_DOCKER_CONTAINER = os.environ.get("PIPER_DOCKER_CONTAINER", "").strip()
PIPER_DOCKER_EXEC_REQUIRED = os.environ.get("PIPER_DOCKER_EXEC_REQUIRED", "1").strip().lower() in (
    "1", "true", "yes", "on"
)

app = FastAPI()


class SpeakReq(BaseModel):
    text: str
    speaker: Optional[str] = None
    language: Optional[str] = None
    instruct: Optional[str] = None


def status(step: str):
    print(f"{PRINT_PREFIX} {step}", flush=True)


def _acquire_synth_slot_or_busy() -> None:
    timeout = SYNTH_ACQUIRE_TIMEOUT_SECONDS
    if timeout <= 0:
        acquired = GPU_SYNTH_SEMAPHORE.acquire(blocking=False)
    else:
        acquired = GPU_SYNTH_SEMAPHORE.acquire(timeout=timeout)
    if not acquired:
        raise HTTPException(
            status_code=SYNTH_BUSY_STATUS_CODE,
            detail=f"TTS synth is busy; retry shortly (waited {max(timeout, 0):.2f}s)",
        )


def _is_busy_http_exception(exc: Exception) -> bool:
    return isinstance(exc, HTTPException) and int(getattr(exc, "status_code", 0)) == SYNTH_BUSY_STATUS_CODE


def _build_piper_cmd(use_docker_exec: bool) -> list[str]:
    cmd = [PIPER_BIN, "--model", PIPER_MODEL, "--output_raw"]
    if PIPER_CONFIG:
        cmd.extend(["--config", PIPER_CONFIG])
    if PIPER_SPEAKER:
        cmd.extend(["--speaker", PIPER_SPEAKER])

    if use_docker_exec and PIPER_DOCKER_CONTAINER:
        return ["docker", "exec", "-i", PIPER_DOCKER_CONTAINER, *cmd]
    return cmd


def _run_piper_cmd(cmd: list[str], text: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        cmd,
        input=(text + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _piper_backend_configured() -> bool:
    """Best-effort gate to avoid replacing busy errors with missing-piper errors."""
    if PIPER_DOCKER_CONTAINER:
        # Docker-exec mode explicitly requested.
        return True
    # Local binary mode requires an installed piper executable.
    return bool(shutil.which(PIPER_BIN))


def _synthesize_with_piper(text: str) -> Tuple[np.ndarray, int]:
    attempted_docker_exec = bool(PIPER_DOCKER_CONTAINER)
    cmd = _build_piper_cmd(use_docker_exec=attempted_docker_exec)

    try:
        proc = _run_piper_cmd(cmd, text)
    except FileNotFoundError:
        if attempted_docker_exec and not PIPER_DOCKER_EXEC_REQUIRED:
            status("speak: docker CLI unavailable for piper docker-exec; retrying with local piper binary")
            try:
                proc = _run_piper_cmd(_build_piper_cmd(use_docker_exec=False), text)
            except FileNotFoundError:
                raise HTTPException(
                    status_code=_http_status_for_piper(),
                    detail=f"Piper command not found (bin={PIPER_BIN}); docker CLI also unavailable",
                )
        else:
            raise HTTPException(
                status_code=_http_status_for_piper(),
                detail=(f"Piper command not found (bin={PIPER_BIN})" if not attempted_docker_exec else "Docker CLI not found for Piper docker-exec mode (set PIPER_DOCKER_EXEC_REQUIRED=0 to allow local piper fallback)"),
            )

    if proc.returncode != 0 and attempted_docker_exec and not PIPER_DOCKER_EXEC_REQUIRED:
        status("speak: piper docker-exec failed; retrying with local piper binary")
        try:
            fallback_proc = _run_piper_cmd(_build_piper_cmd(use_docker_exec=False), text)
            if fallback_proc.returncode == 0:
                proc = fallback_proc
        except FileNotFoundError:
            pass

    if proc.returncode != 0:
        stderr_text = proc.stderr.decode("utf-8", errors="ignore")[-300:]
        raise HTTPException(
            status_code=_http_status_for_piper(),
            detail=f"Piper synthesis failed (exit={proc.returncode}): {stderr_text}",
        )

    audio_i16 = np.frombuffer(proc.stdout, dtype=np.int16)
    if audio_i16.size == 0:
        raise HTTPException(
            status_code=_http_status_for_piper(),
            detail="Piper synthesis returned empty audio",
        )

    # Piper commonly emits 22050Hz mono PCM when using --output_raw.
    sample_rate = 22050
    audio_f32 = (audio_i16.astype(np.float32) / 32768.0).squeeze()
    return audio_f32, sample_rate


def _http_status_for_piper() -> int:
    return 503 if PIPER_FORCE_REQUESTS else SYNTH_BUSY_STATUS_CODE


def _synthesize_with_busy_fallback(text: str, speaker: str, language: str, instruct: str) -> Tuple[np.ndarray, int, bool]:
    if PIPER_FORCE_REQUESTS:
        if not _piper_backend_configured():
            raise HTTPException(status_code=503, detail=f"Forced Piper mode enabled but backend unavailable (bin={PIPER_BIN}, docker_container={PIPER_DOCKER_CONTAINER or '-'})")
        status("speak: using piper (forced by QWEN_FORCE_PIPER)")
        wav, sr = _synthesize_with_piper(text)
        return wav, sr, True

    try:
        _acquire_synth_slot_or_busy()
        try:
            wav, sr = _synthesize_to_audio_with_timeout(text, speaker, language, instruct)
        finally:
            GPU_SYNTH_SEMAPHORE.release()
            _maybe_compact_cuda_heap(stage="chunk")
        return wav, sr, False
    except HTTPException as exc:
        if not (PIPER_BUSY_FALLBACK and _is_busy_http_exception(exc)):
            raise
        if not _piper_backend_configured():
            status("speak: qwen synth busy; piper backend unavailable, keeping busy response")
            raise
        status("speak: qwen synth busy, using piper fallback for chunk")
        wav, sr = _synthesize_with_piper(text)
        return wav, sr, True


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


status(f"startup: loading model={MODEL_ID} device={DEVICE} dtype={DTYPE}")
model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=DTYPE)
status("startup: model loaded")


def _device_from_module(maybe_module) -> Optional[str]:
    if maybe_module is None:
        return None
    dev = getattr(maybe_module, "device", None)
    if dev is not None:
        return str(dev)
    try:
        param = next(maybe_module.parameters())
        return str(param.device)
    except Exception:
        return None


def _ensure_model_cuda() -> str:
    """Fail fast if model is not actually placed on CUDA."""
    candidates = [
        getattr(model, "model", None),
        getattr(model, "talker", None),
        getattr(model, "code_predictor", None),
        model,
    ]
    for candidate in candidates:
        dev = _device_from_module(candidate)
        if dev:
            if not dev.startswith("cuda"):
                raise RuntimeError(f"Model loaded on {dev}, refusing non-CUDA inference")
            return dev
    raise RuntimeError("Unable to determine model device, refusing to continue")


MODEL_DEVICE = _ensure_model_cuda()
status(f"startup: model runtime device confirmed={MODEL_DEVICE}")

_GEN_CUSTOM_VOICE_PARAMS = set(inspect.signature(model.generate_custom_voice).parameters)
_GEN_CUSTOM_VOICE_ACCEPTS_VAR_KW = any(
    p.kind == inspect.Parameter.VAR_KEYWORD
    for p in inspect.signature(model.generate_custom_voice).parameters.values()
)
_GEN_CUSTOM_VOICE_REQUIRED_PARAMS = {
    name
    for name, p in inspect.signature(model.generate_custom_voice).parameters.items()
    if p.default is inspect.Parameter.empty and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
}

_HAS_GEN_VOICE_CLONE = hasattr(model, "generate_voice_clone")
if _HAS_GEN_VOICE_CLONE:
    _GEN_VOICE_CLONE_PARAMS = set(inspect.signature(model.generate_voice_clone).parameters)
    _GEN_VOICE_CLONE_ACCEPTS_VAR_KW = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(model.generate_voice_clone).parameters.values()
    )
else:
    _GEN_VOICE_CLONE_PARAMS = set()
    _GEN_VOICE_CLONE_ACCEPTS_VAR_KW = False

_HAS_CREATE_VOICE_CLONE_PROMPT = hasattr(model, "create_voice_clone_prompt")
if _HAS_CREATE_VOICE_CLONE_PROMPT:
    _CREATE_VOICE_CLONE_PROMPT_PARAMS = set(inspect.signature(model.create_voice_clone_prompt).parameters)
    _CREATE_VOICE_CLONE_PROMPT_ACCEPTS_VAR_KW = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(model.create_voice_clone_prompt).parameters.values()
    )
else:
    _CREATE_VOICE_CLONE_PROMPT_PARAMS = set()
    _CREATE_VOICE_CLONE_PROMPT_ACCEPTS_VAR_KW = False


def _resolve_training_paths() -> tuple[str, str]:
    """Resolve train.wav / train.txt using cwd + script directory fallback."""
    txt_candidates = [TRAIN_TXT_PATH]
    # Backward compatibility for earlier typo-based default.
    if TRAIN_TXT_PATH == "train.txt":
        txt_candidates.append("tran.txt")

    script_dir = os.path.dirname(__file__)
    candidates = []
    for txt_name in txt_candidates:
        candidates.append((TRAIN_WAV_PATH, txt_name))
        candidates.append((os.path.join(script_dir, TRAIN_WAV_PATH), os.path.join(script_dir, txt_name)))
    for wav_path, txt_path in candidates:
        if os.path.isfile(wav_path) and os.path.isfile(txt_path):
            return wav_path, txt_path

    raise FileNotFoundError(
        f"Could not find startup training files. Looked for wav='{TRAIN_WAV_PATH}' and txt in {txt_candidates} "
        f"in cwd and script directory."
    )


def _build_training_voice_kwargs() -> dict:
    """Load startup voice snippet and map it into supported qwen-tts kwargs."""
    status("startup: loading voice clone prompt files")
    wav_path, txt_path = _resolve_training_paths()

    with open(txt_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()
    if not transcript:
        raise RuntimeError(f"Training text file is empty: {txt_path}")

    kwargs: dict = {}

    audio_arg_candidates = [
        "prompt_audio",
        "prompt_audio_path",
        "audio_prompt_path",
        "voice_prompt_path",
        "voice",
        "prompt_wav",
        "prompt_wav_path",
        "reference_audio",
        "reference_audio_path",
    f"startup: ready speakers={len(SUPPORTED_SPEAKERS)} languages={len(SUPPORTED_LANGS)} voice_prompt={'on' if bool(TRAINING_VOICE_KWARGS) else 'off'} voice_clone_api={'on' if _HAS_GEN_VOICE_CLONE else 'off'} cached_voice_clone_prompt={'on' if ('voice_clone_prompt' in VOICE_CLONE_KWARGS) else 'off'} gpu_synth_concurrency={GPU_SYNTH_CONCURRENCY} chunk_gen_timeout_s={CHUNK_GEN_TIMEOUT_SECONDS:g} fp16_retry_fp32={'on' if FP16_RETRY_FP32 else 'off'} cuda_cache_clear_policy={CUDA_CACHE_CLEAR_POLICY} cuda_pressure_threshold={CUDA_CACHE_PRESSURE_THRESHOLD:.3f} synth_acquire_timeout_s={SYNTH_ACQUIRE_TIMEOUT_SECONDS:g} busy_status={SYNTH_BUSY_STATUS_CODE} force_piper={'on' if QWEN_FORCE_PIPER else 'off'} piper_http_url={PIPER_HTTP_URL}"
        "ref_audio_path",
        "clone_audio",
        "clone_audio_path",
        "enroll_audio",
        "enroll_audio_path",
    ]
    text_arg_candidates = [
        "prompt_text",
        "audio_prompt_text",
        "voice_prompt_text",
        "prompt_transcript",
        "transcript",
        "reference_text",
        "ref_text",
        "clone_text",
        "enroll_text",
    ]

    if _GEN_CUSTOM_VOICE_ACCEPTS_VAR_KW:
        # For wrapper APIs that accept **kwargs, keep prompt keys minimal to avoid ambiguous collisions.
        chosen_audio_key = PROMPT_AUDIO_ARG_OVERRIDE or audio_arg_candidates[0]
        chosen_text_key = PROMPT_TEXT_ARG_OVERRIDE or text_arg_candidates[0]
        kwargs[chosen_audio_key] = wav_path
        kwargs[chosen_text_key] = transcript
        kwargs["voice_prompt"] = {"audio": wav_path, "text": transcript}
    else:
        if PROMPT_AUDIO_ARG_OVERRIDE:
            if PROMPT_AUDIO_ARG_OVERRIDE in _GEN_CUSTOM_VOICE_PARAMS:
                kwargs[PROMPT_AUDIO_ARG_OVERRIDE] = wav_path
            else:
                status(
                    f"startup: ignored QWEN_PROMPT_AUDIO_ARG={PROMPT_AUDIO_ARG_OVERRIDE} (not in supported params)"
                )
        for key in audio_arg_candidates:
            if kwargs:
                break
            if key in _GEN_CUSTOM_VOICE_PARAMS:
                kwargs[key] = wav_path
                break

        if PROMPT_TEXT_ARG_OVERRIDE:
            if PROMPT_TEXT_ARG_OVERRIDE in _GEN_CUSTOM_VOICE_PARAMS:
                kwargs[PROMPT_TEXT_ARG_OVERRIDE] = transcript
            else:
                status(
                    f"startup: ignored QWEN_PROMPT_TEXT_ARG={PROMPT_TEXT_ARG_OVERRIDE} (not in supported params)"
                )
        for key in text_arg_candidates:
            if any(k in kwargs for k in text_arg_candidates + [PROMPT_TEXT_ARG_OVERRIDE]):
                break
            if key in _GEN_CUSTOM_VOICE_PARAMS:
                kwargs[key] = transcript
                break

    # Some APIs use a structured prompt object instead of separate fields.
    structured_prompt_candidates = ["voice_prompt", "prompt", "reference"]
    if not kwargs and not _GEN_CUSTOM_VOICE_ACCEPTS_VAR_KW:
        for key in structured_prompt_candidates:
            if key in _GEN_CUSTOM_VOICE_PARAMS:
                kwargs[key] = {"audio": wav_path, "text": transcript}
                break
    elif _GEN_CUSTOM_VOICE_ACCEPTS_VAR_KW:
        kwargs["voice_prompt"] = {"audio": wav_path, "text": transcript}

    if not kwargs:
        status(
            "startup: no compatible voice prompt args detected; "
            f"available={sorted(_GEN_CUSTOM_VOICE_PARAMS)}"
        )
    else:
        status(
            f"startup: voice cloning prompt ready wav={wav_path} txt={txt_path} keys={sorted(kwargs.keys())}"
        )
        if _GEN_CUSTOM_VOICE_ACCEPTS_VAR_KW:
            status("startup: generate_custom_voice accepts **kwargs; sending minimal prompt key set")

    return kwargs


def _build_voice_clone_kwargs() -> dict:
    """Load startup voice snippet and map into generate_voice_clone kwargs."""
    if not _HAS_GEN_VOICE_CLONE:
        return {}

    wav_path, txt_path = _resolve_training_paths()
    with open(txt_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()
    if not transcript:
        raise RuntimeError(f"Training text file is empty: {txt_path}")

    kwargs = {}
    if _GEN_VOICE_CLONE_ACCEPTS_VAR_KW or "ref_audio" in _GEN_VOICE_CLONE_PARAMS:
        kwargs["ref_audio"] = wav_path
    elif "reference_audio" in _GEN_VOICE_CLONE_PARAMS:
        kwargs["reference_audio"] = wav_path

    if _GEN_VOICE_CLONE_ACCEPTS_VAR_KW or "ref_text" in _GEN_VOICE_CLONE_PARAMS:
        kwargs["ref_text"] = transcript
    elif "reference_text" in _GEN_VOICE_CLONE_PARAMS:
        kwargs["reference_text"] = transcript

    can_pass_cached_prompt = _GEN_VOICE_CLONE_ACCEPTS_VAR_KW or "voice_clone_prompt" in _GEN_VOICE_CLONE_PARAMS
    if kwargs and _HAS_CREATE_VOICE_CLONE_PROMPT and can_pass_cached_prompt:
        prompt_kwargs = dict(kwargs)
        if not _CREATE_VOICE_CLONE_PROMPT_ACCEPTS_VAR_KW:
            prompt_kwargs = {k: v for k, v in prompt_kwargs.items() if k in _CREATE_VOICE_CLONE_PROMPT_PARAMS}

        if prompt_kwargs:
            try:
                voice_clone_prompt = model.create_voice_clone_prompt(**prompt_kwargs)
                kwargs = {"voice_clone_prompt": voice_clone_prompt}
                status(
                    "startup: prebuilt reusable voice_clone_prompt for generate_voice_clone"
                )
            except Exception as e:
                status(f"startup: create_voice_clone_prompt failed; falling back to ref args ({e})")

    if kwargs:
        status(
            f"startup: voice cloning reference ready wav={wav_path} txt={txt_path} keys={sorted(kwargs.keys())}"
        )
    else:
        status(
            "startup: generate_voice_clone present but no compatible ref kwargs detected; "
            f"available={sorted(_GEN_VOICE_CLONE_PARAMS)}"
        )
    return kwargs


try:
    TRAINING_VOICE_KWARGS = _build_training_voice_kwargs()
except FileNotFoundError as e:
    if REQUIRE_TRAINING_FILES:
        raise
    status(f"startup: training voice prompt files not found; continuing without clone prompt ({e})")
    TRAINING_VOICE_KWARGS = {}

try:
    VOICE_CLONE_KWARGS = _build_voice_clone_kwargs()
except FileNotFoundError as e:
    if REQUIRE_TRAINING_FILES:
        raise
    status(f"startup: voice clone reference files not found; continuing without clone refs ({e})")
    VOICE_CLONE_KWARGS = {}

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

status(
    f"startup: ready speakers={len(SUPPORTED_SPEAKERS)} languages={len(SUPPORTED_LANGS)} voice_prompt={'on' if bool(TRAINING_VOICE_KWARGS) else 'off'} voice_clone_api={'on' if _HAS_GEN_VOICE_CLONE else 'off'} cached_voice_clone_prompt={'on' if ('voice_clone_prompt' in VOICE_CLONE_KWARGS) else 'off'} gpu_synth_concurrency={GPU_SYNTH_CONCURRENCY} chunk_gen_timeout_s={CHUNK_GEN_TIMEOUT_SECONDS:g} fp16_retry_fp32={'on' if FP16_RETRY_FP32 else 'off'} cuda_cache_clear_policy={CUDA_CACHE_CLEAR_POLICY} cuda_pressure_threshold={CUDA_CACHE_PRESSURE_THRESHOLD:.3f} synth_acquire_timeout_s={SYNTH_ACQUIRE_TIMEOUT_SECONDS:g} busy_status={SYNTH_BUSY_STATUS_CODE} piper_busy_fallback={'on' if PIPER_BUSY_FALLBACK else 'off'} piper_model_configured={'yes' if bool(PIPER_MODEL) else 'no'} piper_forced={'on' if PIPER_FORCE_REQUESTS else 'off'} piper_docker_container={PIPER_DOCKER_CONTAINER or '-'} piper_docker_exec_required={'on' if PIPER_DOCKER_EXEC_REQUIRED else 'off'} piper_backend_ready={'yes' if _piper_backend_configured() else 'no'}"
)


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


def _chunk_text_sentence(text: str, max_chars: int = 240, split_oversized_sentences: bool = True) -> list[str]:
    """Sentence-ish splitter + length cap for a single text block."""
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

        if len(p) > max_chars and split_oversized_sentences:
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

    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < 30:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)
    return merged




def _append_fullstop_to_list_lines(text: str) -> str:
    """Ensure bullet/numbered lines end with punctuation for natural TTS pauses."""
    lines = (text or "").splitlines()
    out: list[str] = []
    list_re = re.compile(r"^\s*(?:\d+[\.)]|[-*•])\s+")

    for line in lines:
        stripped = line.rstrip()
        if list_re.match(stripped):
            tail = stripped.rstrip()
            if tail and tail[-1] not in ".!?;:":
                stripped = f"{tail}."
        out.append(stripped)

    return "\n".join(out)

def _split_numbered_bullets(text: str) -> list[str]:
    """Split numbered-bullet text into items, preserving wrapped continuation lines."""
    lines = [line.rstrip() for line in (text or "").splitlines()]
    bullet_re = re.compile(r"^\s*\d+[\.)]\s+")

    items: list[str] = []
    current: list[str] = []
    saw_bullet = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if bullet_re.match(line):
            saw_bullet = True
            if current:
                items.append(" ".join(current).strip())
            current = [line]
            continue

        if current:
            current.append(line)
        else:
            # Leading non-bullet text: not a numbered list block.
            return []

    if current:
        items.append(" ".join(current).strip())

    return items if saw_bullet and len(items) > 1 else []


def _chunk_text(text: str, max_chars: int = 240, paragraph_aware: bool = True) -> list[str]:
    """Paragraph-first chunking with sentence fallback when paragraph exceeds max_chars."""
    if not paragraph_aware:
        return _chunk_text_sentence(text, max_chars=max_chars)

    raw = _append_fullstop_to_list_lines((text or "")).strip()
    if not raw:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    if not paragraphs:
        return _chunk_text_sentence(raw, max_chars=max_chars)

    # lock=True means this unit should remain its own chunk (used for numbered bullets).
    units: list[tuple[str, bool]] = []
    for para in paragraphs:
        bullet_items = _split_numbered_bullets(para)
        if bullet_items:
            for item in bullet_items:
                item_inline = re.sub(r"\s+", " ", item).strip()
                if not item_inline:
                    continue
                if len(item_inline) <= max_chars:
                    units.append((item_inline, True))
                else:
                    units.extend((c, True) for c in _chunk_text_sentence(item_inline, max_chars=max_chars, split_oversized_sentences=False))
            continue

        para_inline = re.sub(r"\s+", " ", para).strip()
        if not para_inline:
            continue
        if len(para_inline) <= max_chars:
            units.append((para_inline, False))
        else:
            units.extend((c, False) for c in _chunk_text_sentence(para_inline, max_chars=max_chars, split_oversized_sentences=False))

    out: list[str] = []
    buf = ""

    def flush_buf() -> None:
        nonlocal buf
        if buf:
            out.append(buf)
            buf = ""

    for chunk, locked in units:
        if locked:
            flush_buf()
            out.append(chunk)
            continue

        candidate = (buf + " " + chunk).strip() if buf else chunk
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            flush_buf()
            buf = chunk

    flush_buf()
    return out


_NUM_0_TO_19 = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int_to_words(num: int) -> str:
    if num < 20:
        return _NUM_0_TO_19[num]
    if num < 100:
        tens = _TENS[num // 10]
        rem = num % 10
        return tens if rem == 0 else f"{tens} {_NUM_0_TO_19[rem]}"
    if num < 1000:
        hundreds = _NUM_0_TO_19[num // 100]
        rem = num % 100
        return f"{hundreds} hundred" if rem == 0 else f"{hundreds} hundred {_int_to_words(rem)}"
    if num < 1_000_000:
        thousands = _int_to_words(num // 1000)
        rem = num % 1000
        return f"{thousands} thousand" if rem == 0 else f"{thousands} thousand {_int_to_words(rem)}"
    millions = _int_to_words(num // 1_000_000)
    rem = num % 1_000_000
    return f"{millions} million" if rem == 0 else f"{millions} million {_int_to_words(rem)}"


def _normalize_text_for_tts(text: str) -> str:
    """Convert integer digits to words and remove symbols before synthesis."""
    def _replace_number(match: re.Match) -> str:
        raw = match.group(0)
        try:
            return _int_to_words(int(raw))
        except Exception:
            return raw

    normalized = re.sub(r"\d+", _replace_number, text)
    # Keep letters/numbers/whitespace and sentence punctuation cues used by TTS/chunking.
    normalized = re.sub(r"[^A-Za-z0-9\s,\.!\?;:]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _is_fp16_sampling_error(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return ("probability tensor contains" in msg) or ("nan" in msg) or ("inf" in msg)


def _run_generation_with_fp16_retry(gen_fn, kwargs: dict, label: str):
    with torch.inference_mode():
        try:
            if DEVICE.startswith("cuda") and AUTOMIX_FP32 and DTYPE == torch.float16:
                with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
                    return gen_fn(**kwargs)
            return gen_fn(**kwargs)
        except RuntimeError as err:
            should_retry = DEVICE.startswith("cuda") and DTYPE == torch.float16 and FP16_RETRY_FP32 and _is_fp16_sampling_error(err)
            if not should_retry:
                raise

            status(f"speak: {label} fp16 instability detected; retrying once with fp32-safe path")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.autocast(device_type="cuda", enabled=False):
                return gen_fn(**kwargs)




def _synthesize_to_audio(text: str, speaker: str, language: str, instruct: str) -> Tuple[np.ndarray, int]:
    current_dev = _ensure_model_cuda()
    use_voice_clone_api = bool(VOICE_CLONE_KWARGS) and _HAS_GEN_VOICE_CLONE
    is_cloned_voice = use_voice_clone_api or bool(TRAINING_VOICE_KWARGS)
    speaker_label = "cloned" if is_cloned_voice else speaker
    status(f"speak: cloning voice (len={len(text)} chars, speaker={speaker_label}, language={language})")
    status(f"speak: runtime model device={current_dev}")

    if use_voice_clone_api:
        clone_kwargs = {
            "text": text,
            "language": language,
            **VOICE_CLONE_KWARGS,
            **GEN_KWARGS,
        }
        if not _GEN_VOICE_CLONE_ACCEPTS_VAR_KW:
            clone_kwargs = {k: v for k, v in clone_kwargs.items() if k in _GEN_VOICE_CLONE_PARAMS}

        sent_ref_keys = [k for k in clone_kwargs if k in VOICE_CLONE_KWARGS]
        status(f"speak: using generate_voice_clone ref args={sorted(sent_ref_keys)}")

        audio_list, sr = _run_generation_with_fp16_retry(
            model.generate_voice_clone,
            clone_kwargs,
            label="generate_voice_clone",
        )

        wav = _concat_audio(audio_list)
        if wav.size == 0:
            raise RuntimeError("Empty audio output from model")

        status(f"speak: cloning voice finished sr={int(sr)} samples={wav.size}")
        return wav, int(sr)

    call_kwargs = {
        "text": text,
        "language": language,
        "instruct": instruct,
        "non_streaming_mode": True,
        **TRAINING_VOICE_KWARGS,
        **GEN_KWARGS,
    }

    # When using a startup voice prompt, most qwen-tts variants should avoid a fixed built-in speaker.
    if TRAINING_VOICE_KWARGS and not ALLOW_SPEAKER_WITH_TRAIN:
        if "speaker" in _GEN_CUSTOM_VOICE_PARAMS and FORCE_CUSTOM_SPEAKER:
            if SUPPORTED_SPEAKERS and FORCE_CUSTOM_SPEAKER.lower() not in SUPPORTED_SPEAKERS:
                fallback_speaker = speaker or DEFAULT_SPEAKER
                call_kwargs["speaker"] = fallback_speaker
                status(
                    "speak: custom speaker override skipped because it is not in model supported speakers; "
                    f"using fallback speaker={fallback_speaker} with voice prompt"
                )
            else:
                call_kwargs["speaker"] = FORCE_CUSTOM_SPEAKER
                status(f"speak: using cloned voice speaker override speaker={FORCE_CUSTOM_SPEAKER}")
    else:
        call_kwargs["speaker"] = speaker

    # Some qwen-tts builds require speaker even for prompt-based cloning.
    if "speaker" in _GEN_CUSTOM_VOICE_REQUIRED_PARAMS and not call_kwargs.get("speaker"):
        fallback_speaker = speaker or DEFAULT_SPEAKER
        call_kwargs["speaker"] = fallback_speaker
        status(f"speak: required speaker arg injected speaker={fallback_speaker}")

    # Only pass arguments supported by the installed API signature, unless it accepts **kwargs.
    if not _GEN_CUSTOM_VOICE_ACCEPTS_VAR_KW:
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in _GEN_CUSTOM_VOICE_PARAMS}

    prompt_keys_sent = [k for k in call_kwargs if k in TRAINING_VOICE_KWARGS]
    status(f"speak: voice prompt args in call={sorted(prompt_keys_sent)}")

    audio_list, sr = _run_generation_with_fp16_retry(
        model.generate_custom_voice,
        call_kwargs,
        label="generate_custom_voice",
    )

    wav = _concat_audio(audio_list)
    if wav.size == 0:
        raise RuntimeError("Empty audio output from model")

    status(f"speak: cloning voice finished sr={int(sr)} samples={wav.size}")
    return wav, int(sr)


    if (not QWEN_FORCE_PIPER) and SUPPORTED_SPEAKERS and speaker not in SUPPORTED_SPEAKERS:
    if (not QWEN_FORCE_PIPER) and SUPPORTED_LANGS and language not in SUPPORTED_LANGS:
                    if idx == 0 and first_word_latency_seconds is None:
                        first_word_latency_seconds = time.perf_counter() - request_start
                    if QWEN_FORCE_PIPER:
                        wav, sr = _synthesize_with_piper_http(part)
                    else:
                        _acquire_synth_slot_or_busy()
                        try:
                            wav, sr = _synthesize_to_audio_with_timeout(part, speaker, language, instruct)
                        finally:
                            GPU_SYNTH_SEMAPHORE.release()
                            _maybe_compact_cuda_heap(stage="chunk")
        raise TimeoutError(
            f"Chunk synthesis exceeded timeout of {CHUNK_GEN_TIMEOUT_SECONDS:.1f}s"
        )



def _maybe_compact_cuda_heap(stage: str) -> None:
    """
    Optional mitigation for perceived VRAM growth from the CUDA caching allocator.

    Policies:
      - off: disable explicit compaction
      - chunk: compact after each chunk (legacy behavior)
      - request: compact once when request completes
      - pressure: compact only when GPU memory pressure crosses threshold
    """
    if not torch.cuda.is_available():
        return

    policy = CUDA_CACHE_CLEAR_POLICY
    should_compact = False

    if policy == "off":
        return
    if policy == "chunk":
        should_compact = stage == "chunk"
    elif policy == "request":
        should_compact = stage == "request_end"
    elif policy == "pressure":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_ratio = 1.0 - (float(free_bytes) / float(total_bytes))
        should_compact = used_ratio >= CUDA_CACHE_PRESSURE_THRESHOLD
        if should_compact:
            status(
                f"speak: cuda pressure detected used_ratio={used_ratio:.3f} threshold={CUDA_CACHE_PRESSURE_THRESHOLD:.3f}; compacting cache"
            )
    else:
        status(f"startup: unknown QWEN_CUDA_CACHE_CLEAR_POLICY='{policy}', disabling cache compaction")
        return

    if not should_compact:
        return

    gc.collect()
    torch.cuda.empty_cache()


def _encode_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, int(sr), format="WAV")
    return buf.getvalue()


def _concat_wav_bytes(wav_chunks: list[bytes]) -> Tuple[bytes, int]:
    """Concatenate several WAV payloads into one WAV payload."""
    if not wav_chunks:
        raise RuntimeError("No WAV chunks to concatenate")

    merged_audio: list[np.ndarray] = []
    sample_rate: Optional[int] = None

    for chunk_bytes in wav_chunks:
        with io.BytesIO(chunk_bytes) as chunk_buf:
            chunk_audio, chunk_sr = sf.read(chunk_buf, dtype="float32")

        if sample_rate is None:
            sample_rate = int(chunk_sr)
        elif int(chunk_sr) != sample_rate:
            raise RuntimeError(
                f"Sample-rate mismatch across chunks ({sample_rate} vs {int(chunk_sr)})"
            )

        merged_audio.append(np.asarray(chunk_audio).squeeze())

    merged = _concat_audio(merged_audio)
    out = io.BytesIO()
    sf.write(out, merged, sample_rate, format="WAV")
    return out.getvalue(), sample_rate


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


            if idx == 0 and first_word_latency_seconds is None:
                # First-word proxy: when first chunk is about to enter model inference.
                first_word_latency_seconds = time.perf_counter() - request_start
            if QWEN_FORCE_PIPER:
                wav, sr = _synthesize_with_piper_http(part)
            else:
                _acquire_synth_slot_or_busy()
                try:
                    wav, sr = _synthesize_to_audio_with_timeout(part, speaker, language, instruct)
                finally:
                    GPU_SYNTH_SEMAPHORE.release()
                    _maybe_compact_cuda_heap(stage="chunk")
def _worker_loop():
    while not _worker_stop.is_set():
        try:
            job = play_q.get(timeout=0.25)
        except queue.Empty:
            continue
        try:
            status(f"playback: starting job={job.job_id}")
            _play_wav_bytes(job.wav_bytes)
            status(f"playback: finished job={job.job_id}")
        finally:
            play_q.task_done()


_worker_thread = threading.Thread(target=_worker_loop, daemon=True)
_worker_thread.start()


ENCODER_POOL_SIZE = int(os.environ.get("QWEN_WAV_ENCODER_THREADS", "2"))
encode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, ENCODER_POOL_SIZE))
synth_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, GPU_SYNTH_CONCURRENCY))
@app.on_event("shutdown")
def _shutdown():
    _worker_stop.set()
    encode_pool.shutdown(wait=False, cancel_futures=True)
    synth_pool.shutdown(wait=False, cancel_futures=True)


# ----------------------------
# Main endpoint
# ----------------------------
@app.post("/speak")
def speak(
    req: SpeakReq,
    play: bool = Query(True, description="If true (default), queue server-side playback via ffplay"),
    return_audio: bool = Query(False, description="If true, return audio/wav bytes (all chunks) instead of JSON"),
    stream_audio_chunks: bool = Query(False, description="If true, stream NDJSON records with base64 WAV per chunk for low-latency client playback"),
    chunk: bool = Query(True, description="If true (default), split text into chunks"),
    paragraph_chunking: bool = Query(True, description="If true (default), keep paragraphs as chunks when under max_chars"),
    max_chars: int = Query(240, ge=80, le=800, description="Chunk size cap"),
):
    request_start = time.perf_counter()
    status("speak: request received")

    raw_text = (req.text or "").strip()
    if not raw_text:
        return Response(status_code=204)

    speaker = (req.speaker or DEFAULT_SPEAKER).strip().lower()
    language = (req.language or DEFAULT_LANG).strip().lower()
    instruct = (req.instruct or DEFAULT_INSTRUCT).strip()

    if (not PIPER_FORCE_REQUESTS) and SUPPORTED_SPEAKERS and speaker not in SUPPORTED_SPEAKERS:
        raise HTTPException(status_code=400, detail=f"Unsupported speaker '{speaker}'. Use GET /speakers")
    if (not PIPER_FORCE_REQUESTS) and SUPPORTED_LANGS and language not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail=f"Unsupported language '{language}'. Use GET /languages")

    raw_parts = _chunk_text(raw_text, max_chars=max_chars, paragraph_aware=paragraph_chunking) if chunk else [raw_text]
    parts = [_normalize_text_for_tts(p) for p in raw_parts]
    parts = [p for p in parts if p]
    text = " ".join(parts).strip()
    status(
        f"speak: chunking complete chunks={len(parts)} chunking={'on' if chunk else 'off'} paragraph_chunking={'on' if paragraph_chunking else 'off'}"
    )
    if not parts:
        return Response(status_code=204)

    if return_audio and stream_audio_chunks:
        raise HTTPException(status_code=400, detail="Choose only one: return_audio=1 or stream_audio_chunks=1")

    job_ids = []
    audio_chunks: list[bytes] = []
    first_chunk_latency_seconds: Optional[float] = None
    first_word_latency_seconds: Optional[float] = None
    used_piper_fallback = False

    def _finalize_chunk(chunk_idx: int, wav_bytes: bytes) -> None:
        jid = f"{os.getpid()}-{threading.get_ident()}-{chunk_idx}"
        job_ids.append(jid)

        if play:
            try:
                play_q.put_nowait(SpeakJob(wav_bytes=wav_bytes, job_id=jid))
                status(f"speak: queued playback job={jid} depth={play_q.qsize()}")
            except queue.Full:
                raise HTTPException(status_code=429, detail="Playback queue is full")

        if return_audio:
            audio_chunks.append(wav_bytes)


    if stream_audio_chunks:
        def _stream_ndjson():
            nonlocal first_word_latency_seconds, first_chunk_latency_seconds, used_piper_fallback
            try:
                for idx, part in enumerate(parts):
                    status(f"speak: processing chunk {idx + 1}/{len(parts)} (stream)")

                    if idx == 0 and first_word_latency_seconds is None:
                        first_word_latency_seconds = time.perf_counter() - request_start
                    wav, sr, used_fallback = _synthesize_with_busy_fallback(part, speaker, language, instruct)
                    used_piper_fallback = used_piper_fallback or used_fallback

                    if idx == 0 and first_chunk_latency_seconds is None:
                        first_chunk_latency_seconds = time.perf_counter() - request_start

                    wav_bytes = _encode_wav_bytes(wav, sr)
                    _finalize_chunk(idx, wav_bytes)

                    yield (json.dumps({
                        "type": "audio_chunk",
                        "chunk_index": idx,
                        "chunks_total": len(parts),
                        "sample_rate": int(sr),
                        "text": part,
                        "audio_b64_wav": base64.b64encode(wav_bytes).decode("ascii"),
                    }) + "\n").encode("utf-8")

                elapsed_seconds = time.perf_counter() - request_start
                total_latency_seconds = elapsed_seconds
                fw = first_word_latency_seconds if first_word_latency_seconds is not None else total_latency_seconds
                fc = first_chunk_latency_seconds if first_chunk_latency_seconds is not None else total_latency_seconds
                completion_message = f"Request complete in {elapsed_seconds:.3f}s: {text}"
                status(
                    f"speak: latency first_word={fw:.3f}s first_chunk={fc:.3f}s total={total_latency_seconds:.3f}s"
                )
                status(completion_message)
                _maybe_compact_cuda_heap(stage="request_end")

                yield (json.dumps({
                    "type": "complete",
                    "ok": True,
                    "message": completion_message,
                    "chunks": len(parts),
                    "speaker": speaker,
                    "language": language,
                    "processing_seconds": round(total_latency_seconds, 3),
                    "latency_to_first_word_seconds": round(fw, 3),
                    "latency_to_first_chunk_seconds": round(fc, 3),
                    "queue_depth": play_q.qsize(),
                    "job_ids": job_ids[:10],
                    "used_piper_fallback": bool(used_piper_fallback),
                }) + "\n").encode("utf-8")
            except Exception as e:
                traceback.print_exc()
                err_payload = {"type": "error", "ok": False, "detail": str(e)}
                yield (json.dumps(err_payload) + "\n").encode("utf-8")

        return StreamingResponse(_stream_ndjson(), media_type="application/x-ndjson")

    try:
        previous_encode: Optional[Tuple[int, concurrent.futures.Future[bytes]]] = None

        for idx, part in enumerate(parts):
            status(f"speak: processing chunk {idx + 1}/{len(parts)}")

            if idx == 0 and first_word_latency_seconds is None:
                # First-word proxy: when first chunk is about to enter model inference.
                first_word_latency_seconds = time.perf_counter() - request_start
            wav, sr, used_fallback = _synthesize_with_busy_fallback(part, speaker, language, instruct)
            used_piper_fallback = used_piper_fallback or used_fallback

            if idx == 0 and first_chunk_latency_seconds is None:
                # First-chunk proxy: when first synthesized chunk exits model inference.
                first_chunk_latency_seconds = time.perf_counter() - request_start

            if previous_encode is not None:
                prev_idx, prev_future = previous_encode
                _finalize_chunk(prev_idx, prev_future.result())

            previous_encode = (idx, encode_pool.submit(_encode_wav_bytes, wav, sr))

        if previous_encode is not None:
            last_idx, last_future = previous_encode
            _finalize_chunk(last_idx, last_future.result())

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_seconds = time.perf_counter() - request_start
    total_latency_seconds = elapsed_seconds
    if first_word_latency_seconds is None:
        first_word_latency_seconds = total_latency_seconds
    if first_chunk_latency_seconds is None:
        first_chunk_latency_seconds = total_latency_seconds
    completion_message = f"Request complete in {elapsed_seconds:.3f}s: {text}"
    status(
        f"speak: latency first_word={first_word_latency_seconds:.3f}s "
        f"first_chunk={first_chunk_latency_seconds:.3f}s "
        f"total={total_latency_seconds:.3f}s"
    )
    status(completion_message)
    _maybe_compact_cuda_heap(stage="request_end")

    if return_audio and audio_chunks:
        merged_wav, merged_sr = _concat_wav_bytes(audio_chunks)
        return Response(
            content=merged_wav,
            media_type="audio/wav",
            headers={
                "X-Qwen-Request-Status": "complete",
                "X-Qwen-Speaker": speaker,
                "X-Qwen-Language": language,
                "X-Qwen-SampleRate": str(merged_sr),
                "X-Qwen-Chunks": str(len(parts)),
                "X-Qwen-Chunked": "1" if chunk else "0",
                "X-Qwen-Latency-First-Word-Seconds": f"{first_word_latency_seconds:.3f}",
                "X-Qwen-Latency-First-Chunk-Seconds": f"{first_chunk_latency_seconds:.3f}",
                "X-Qwen-Processing-Seconds": f"{total_latency_seconds:.3f}",
                "X-Qwen-Used-Piper-Fallback": "1" if used_piper_fallback else "0",
            },
        )

    return JSONResponse({
        "ok": True,
        "message": completion_message,
        "text": text,
        "processing_seconds": round(total_latency_seconds, 3),
        "latency_to_first_word_seconds": round(first_word_latency_seconds, 3),
        "latency_to_first_chunk_seconds": round(first_chunk_latency_seconds, 3),
        "total_latency_seconds": round(total_latency_seconds, 3),
        "queued": bool(play),
        "return_audio": bool(return_audio),
        "chunked": bool(chunk),
        "paragraph_chunking": bool(paragraph_chunking),
        "stream_audio_chunks": bool(stream_audio_chunks),
        "chunks": len(parts),
        "max_chars": max_chars,
        "speaker": speaker,
        "language": language,
        "queue_depth": play_q.qsize(),
        "job_ids": job_ids[:10],  # avoid huge responses
        "used_piper_fallback": bool(used_piper_fallback),
    })

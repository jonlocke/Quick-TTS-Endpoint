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

Deps (WSL):
  sudo apt-get install -y ffmpeg sox libsox-fmt-all
  pip install fastapi uvicorn soundfile numpy torch qwen-tts
"""
import io
import concurrent.futures
import inspect
import os
import re
import subprocess
import threading
import time
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
import torch_pytree_patch

from qwen_tts import Qwen3TTSModel


# ----------------------------
# Configuration
# ----------------------------
MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
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

TRAIN_WAV_PATH = os.environ.get("QWEN_TRAIN_WAV", "train.wav").strip()
TRAIN_TXT_PATH = os.environ.get("QWEN_TRAIN_TXT", "train.txt").strip()
ALLOW_SPEAKER_WITH_TRAIN = os.environ.get("QWEN_ALLOW_SPEAKER_WITH_TRAIN", "0").strip().lower() in (
    "1", "true", "yes", "on"
)
FORCE_CUSTOM_SPEAKER = os.environ.get("QWEN_FORCE_CUSTOM_SPEAKER", "custom").strip()
PROMPT_AUDIO_ARG_OVERRIDE = os.environ.get("QWEN_PROMPT_AUDIO_ARG", "").strip()
PROMPT_TEXT_ARG_OVERRIDE = os.environ.get("QWEN_PROMPT_TEXT_ARG", "").strip()

# Generation kwargs (best effort; still force greedy via generation_config)
GEN_KWARGS = {
    "do_sample": False,
    "num_beams": 1,
    "temperature": 1.0,
    "top_p": 1.0,
}

PRINT_PREFIX = "[qwen-speak]"

GPU_SYNTH_CONCURRENCY = max(1, int(os.environ.get("QWEN_GPU_SYNTH_CONCURRENCY", "1")))
# Caps concurrent model inference calls per worker (primarily affects concurrent HTTP requests).
GPU_SYNTH_SEMAPHORE = threading.BoundedSemaphore(value=GPU_SYNTH_CONCURRENCY)

app = FastAPI()


class SpeakReq(BaseModel):
    text: str
    speaker: Optional[str] = None
    language: Optional[str] = None
    instruct: Optional[str] = None


def status(step: str):
    print(f"{PRINT_PREFIX} {step}", flush=True)


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
        "ref_audio",
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


TRAINING_VOICE_KWARGS = _build_training_voice_kwargs()
VOICE_CLONE_KWARGS = _build_voice_clone_kwargs()

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
    f"startup: ready speakers={len(SUPPORTED_SPEAKERS)} languages={len(SUPPORTED_LANGS)} voice_prompt={'on' if bool(TRAINING_VOICE_KWARGS) else 'off'} voice_clone_api={'on' if _HAS_GEN_VOICE_CLONE else 'off'} gpu_synth_concurrency={GPU_SYNTH_CONCURRENCY}"
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
    # Keep letters/numbers/whitespace and sentence punctuation used by chunking.
    normalized = re.sub(r"[^A-Za-z0-9\s\.!\?]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


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

        with torch.inference_mode():
            if DEVICE.startswith("cuda") and AUTOMIX_FP32 and DTYPE == torch.float16:
                with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
                    audio_list, sr = model.generate_voice_clone(**clone_kwargs)
            else:
                audio_list, sr = model.generate_voice_clone(**clone_kwargs)

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

    with torch.inference_mode():
        if DEVICE.startswith("cuda") and AUTOMIX_FP32 and DTYPE == torch.float16:
            with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
                audio_list, sr = model.generate_custom_voice(**call_kwargs)
        else:
            audio_list, sr = model.generate_custom_voice(**call_kwargs)

    wav = _concat_audio(audio_list)
    if wav.size == 0:
        raise RuntimeError("Empty audio output from model")

    status(f"speak: cloning voice finished sr={int(sr)} samples={wav.size}")
    return wav, int(sr)


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


def _play_wav_bytes(wav_bytes: bytes) -> None:
    # Stream wav bytes directly to ffplay stdin (avoids temp-file churn).
    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
        input=wav_bytes,
        check=False,
    )


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
@app.on_event("shutdown")
def _shutdown():
    _worker_stop.set()
    encode_pool.shutdown(wait=False, cancel_futures=True)


# ----------------------------
# Main endpoint
# ----------------------------
@app.post("/speak")
def speak(
    req: SpeakReq,
    play: bool = Query(True, description="If true (default), queue server-side playback via ffplay"),
    return_audio: bool = Query(False, description="If true, return audio/wav bytes (all chunks) instead of JSON"),
    chunk: bool = Query(True, description="If true (default), split into sentence-ish chunks"),
    max_chars: int = Query(240, ge=80, le=800, description="Chunk size cap"),
):
    request_start = time.perf_counter()
    status("speak: request received")

    text = _normalize_text_for_tts((req.text or "").strip())
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
    status(f"speak: chunking complete chunks={len(parts)} chunking={'on' if chunk else 'off'}")
    if not parts:
        return Response(status_code=204)

    job_ids = []
    audio_chunks: list[bytes] = []
    first_chunk_latency_seconds: Optional[float] = None
    first_word_latency_seconds: Optional[float] = None

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

    try:
        previous_encode: Optional[Tuple[int, concurrent.futures.Future[bytes]]] = None

        for idx, part in enumerate(parts):
            status(f"speak: processing chunk {idx + 1}/{len(parts)}")

            GPU_SYNTH_SEMAPHORE.acquire()
            try:
                if idx == 0 and first_word_latency_seconds is None:
                    # First-word proxy: when first chunk is about to enter model inference.
                    first_word_latency_seconds = time.perf_counter() - request_start
                wav, sr = _synthesize_to_audio(part, speaker, language, instruct)
            finally:
                GPU_SYNTH_SEMAPHORE.release()

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
        "chunks": len(parts),
        "max_chars": max_chars,
        "speaker": speaker,
        "language": language,
        "queue_depth": play_q.qsize(),
        "job_ids": job_ids[:10],  # avoid huge responses
    })

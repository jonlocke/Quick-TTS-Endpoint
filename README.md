# Quick-TTS-Endpoint
Just a quick TTS endpoint

Default model is now set to `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (override with `QWEN_TTS_MODEL`).

For convenience, common aliases (`qwen3-tts`, `qwen-tts`, `qwen3_tts`) are normalized to
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` at startup.

Note: startup now enforces CUDA-only execution and will raise `RuntimeError("CUDA not available, refusing CPU fallback")` if no GPU is available.

## Startup voice snippet training

On startup, the server now attempts to load a training voice snippet and transcript,
then uses them as cloning reference inputs during synthesis:

- `train.wav`
- `train.txt` (also accepts legacy `tran.txt` automatically)

By default these are resolved from either:

1. The current working directory
2. The repository/script directory

You can override paths with:

- `QWEN_TRAIN_WAV`
- `QWEN_TRAIN_TXT`

When training prompt files are present, the server defaults to clone-first behavior
instead of a fixed built-in speaker. Optional tuning:

- `QWEN_ALLOW_SPEAKER_WITH_TRAIN=1` to keep passing request/default speaker
- `QWEN_FORCE_CUSTOM_SPEAKER` (default: `custom`) for APIs that need an explicit custom speaker label
- `QWEN_PROMPT_AUDIO_ARG` / `QWEN_PROMPT_TEXT_ARG` to force exact prompt arg names if your installed qwen-tts wrapper expects specific keys


For `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, the server now prefers the model's
`generate_voice_clone(...)` API when available, and passes the startup snippet as
reference inputs (`ref_audio`/`ref_text` when supported). If unavailable, it
falls back to `generate_custom_voice(...)` compatibility logic.


If you are benchmarking GPU inference, call `/speak` with `play=false` to avoid ffplay server-side playback CPU overhead.

If `nvidia-smi` appears to show VRAM climbing chunk-by-chunk, tune cache compaction with `QWEN_CUDA_CACHE_CLEAR_POLICY`:

- `off` (default): fastest, keep allocator cache for reuse
- `chunk`: compact after every chunk (legacy behavior; can add latency)
- `request`: compact once at request end
- `pressure`: compact only when used VRAM crosses `QWEN_CUDA_CACHE_PRESSURE_THRESHOLD` (default `0.92`)

`QWEN_CUDA_EMPTY_CACHE_EACH_CHUNK=1` is still supported and maps to `chunk` when policy is unset.

When `generate_voice_clone(...)` and `create_voice_clone_prompt(...)` are both available in your installed `qwen-tts`, the server now builds the reference prompt once at startup and reuses it via `voice_clone_prompt` for subsequent generations.

Input normalization: `/speak` now converts integer digits to words (e.g. `3` -> `three`) and strips non-speech symbols before synthesis.


## Low-latency chunk delivery to clients

`POST /speak` now supports incremental chunk delivery for clients that want to start
playback immediately:

- `stream_audio_chunks=1`: returns `application/x-ndjson` where each line is JSON.
  - `{"type":"audio_chunk", ... "audio_b64_wav":"..."}` for each chunk
  - final `{"type":"complete", ...}` summary record
- `paragraph_chunking=1` (default): keeps paragraph boundaries as chunks when the
  paragraph is within `max_chars`; overlong paragraphs fall back to sentence chunking.

`stream_audio_chunks=1` cannot be combined with `return_audio=1`.

## Docker workflow (NVIDIA CUDA Ubuntu 24.04 base)

This repo now includes a GPU-ready container flow built on `nvidia/cuda:12.6.0-runtime-ubuntu24.04`:

1. Build image:

```bash
./scripts/docker-build.sh
```

Optional build naming:

```bash
IMAGE_NAME=my-qwen-tts IMAGE_TAG=dev ./scripts/docker-build.sh
BASE_IMAGE=nvidia/cuda:12.6.0-runtime-ubuntu24.04 IMAGE_TAG=dev ./scripts/docker-build.sh
TORCH_VERSION=2.3.1+cu118 TORCHVISION_VERSION=0.18.1+cu118 TORCHAUDIO_VERSION=2.3.1+cu118 ./scripts/docker-build.sh
```

2. Run container with GPU access:

```bash
./scripts/docker-run.sh
```

Backwards-compatible shortcuts:

```bash
./scripts/build.sh
./scripts/run.sh
```

`./scripts/run.sh` maps host `8765` to container `8765` by default (override with `HOST_PORT`).

Optional runtime overrides:

```bash
HOST_PORT=9000 IMAGE_TAG=dev ./scripts/docker-run.sh
DOCKER_GPU_MODE=off ./scripts/docker-run.sh
```

Notes:
- `scripts/docker-run.sh` starts the container in detached mode and waits for `/health` to report ready.
- `scripts/docker-run.sh` auto-mounts `train.wav` and `train.txt` when present in the repo root.
- `DOCKER_GPU_MODE` controls GPU flags (`auto` default, `on`, `off`).
- Use `DOCKER_GPU_MODE=off` on hosts without NVIDIA Docker runtime support to avoid `failed to discover GPU vendor` startup errors.
- Service listens on container port `8765`.

# Quick-TTS-Endpoint
Just a quick TTS endpoint

Default model is now set to `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (override with `QWEN_TTS_MODEL`).

For convenience, common aliases (`qwen3-tts`, `qwen-tts`, `qwen3_tts`) are normalized to
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` at startup.

Note: startup now enforces CUDA-only execution and will raise `RuntimeError("CUDA not available, refusing CPU fallback")` if no GPU is available.

## Local voice discovery and selection

The server now exposes:

- `GET /voices`: list local clone voices discovered from `*.wav` + matching `*.txt`
- `GET /speakers`: list built-in model speakers

Example: if the repo contains `liz.wav` and `liz.txt`, `/voices` will include `liz`.

`POST /speak` now accepts either:

- `"voice": "liz"` to use a discovered local clone voice
- `"speaker": "ryan"` to use a built-in model speaker

For compatibility, if `voice` is omitted, the server still accepts `speaker`, and it will
first try to match a local clone voice by that name before falling back to a built-in speaker.

Voice files are discovered from:

1. `QWEN_VOICE_DIRS` if set, split using the platform path separator
2. the current working directory
3. the repository/script directory

Each local clone voice requires:

- `<name>.wav`
- `<name>.txt`

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

If training files are missing, startup now continues without voice-clone prompt inputs.
Set `QWEN_REQUIRE_TRAINING_FILES=1` to make missing files a hard startup error.

When startup training prompt files are present, the server defaults to clone-first behavior
instead of a fixed built-in speaker unless the request selects another local voice or built-in speaker.
Optional tuning:

- `QWEN_ALLOW_SPEAKER_WITH_TRAIN=1` to keep passing request/default speaker
- `QWEN_FORCE_CUSTOM_SPEAKER` (default: `custom`) for APIs that need an explicit custom speaker label
- `QWEN_PROMPT_AUDIO_ARG` / `QWEN_PROMPT_TEXT_ARG` to force exact prompt arg names if your installed qwen-tts wrapper expects specific keys


For `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, the server now prefers the model's
`generate_voice_clone(...)` API when available, and passes the startup snippet as
reference inputs (`ref_audio`/`ref_text` when supported). If unavailable, it
falls back to `generate_custom_voice(...)` compatibility logic.


If you are benchmarking GPU inference, call `/speak` with `play=false` to avoid ffplay server-side playback CPU overhead.

If a single synthesis chunk takes too long, the server now bails out by default after 150s (`QWEN_CHUNK_GEN_TIMEOUT_SECONDS`, i.e. 2:30). Set to `0` to disable.

If `nvidia-smi` appears to show VRAM climbing chunk-by-chunk, tune cache compaction with `QWEN_CUDA_CACHE_CLEAR_POLICY`:

- `off` (default): fastest, keep allocator cache for reuse
- `chunk`: compact after every chunk (legacy behavior; can add latency)
- `request`: compact once at request end
- `pressure`: compact only when used VRAM crosses `QWEN_CUDA_CACHE_PRESSURE_THRESHOLD` (default `0.92`)

`QWEN_CUDA_EMPTY_CACHE_EACH_CHUNK=1` is still supported and maps to `chunk` when policy is unset.

When `generate_voice_clone(...)` and `create_voice_clone_prompt(...)` are both available in your installed `qwen-tts`, the server now builds the reference prompt once at startup and reuses it via `voice_clone_prompt` for subsequent generations.

Input normalization: `/speak` now converts integer digits to words (e.g. `3` -> `three`) and strips non-speech symbols before synthesis, while preserving punctuation cues like `, . ! ? : ;`.

List formatting cue: if an input line starts as a bullet/numbered point (`-`, `*`, `•`, `1.`, `1)`), the server appends a trailing full stop when missing so speech cadence sounds natural.


## Low-latency chunk delivery to clients

`POST /speak` now supports incremental chunk delivery for clients that want to start
playback immediately:

- `stream_audio_chunks=1`: returns `application/x-ndjson` where each line is JSON.
  - `{"type":"audio_chunk", ... "audio_b64_wav":"..."}` for each chunk
  - final `{"type":"complete", ...}` summary record
- `paragraph_chunking=1` (default): keeps paragraph boundaries as chunks when the
  paragraph is within `max_chars`; overlong paragraphs fall back to sentence chunking.

`stream_audio_chunks=1` cannot be combined with `return_audio=1`.


## Prepare training audio (normalize to `train.wav`)

Use the helper script to normalize a source wav into training format:

```bash
./scripts/normalize-training-wav.sh <input-wav-or-basepath> [output-wav]
```

Example (your requested ffmpeg chain, outputting `train.wav`):

```bash
./scripts/normalize-training-wav.sh my_voice.wav train.wav
```

## NVIDIA Container Toolkit install (manual)

If Docker is installed but `docker info --format '{{json .Runtimes}}'` does not show `nvidia`, install/configure toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info --format '{{json .Runtimes}}' | grep -i nvidia
```

You can also run the helper script:

```bash
./scripts/deps.sh
```

Set `INSTALL_TOOLKIT=0` to skip toolkit changes and only install Python/system deps.

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
DOCKER_GPU_MODE=on ./scripts/docker-run.sh
DOCKER_GPU_MODE=off ./scripts/docker-run.sh  # only for custom CPU-capable images
PERSIST_MODEL_CACHE=1 MODEL_CACHE_HOST_DIR=$HOME/.cache/quick-tts-hf ./scripts/docker-run.sh
RESTART_POLICY=unless-stopped ./scripts/docker-run.sh
HEALTH_TIMEOUT_SEC=900 ./scripts/docker-run.sh
```

Notes:
- `scripts/docker-run.sh` starts the container in detached mode and waits for `/health` to report ready (default timeout `HEALTH_TIMEOUT_SEC=600`).
- Container restart policy defaults to `unless-stopped` (override with `RESTART_POLICY`).
- `scripts/docker-run.sh` auto-mounts training files only when **both** `train.wav` and `train.txt` are present in the repo root.
- The Dockerfile `CMD` only starts `uvicorn`; mounts must be provided at container run time (for example via `scripts/docker-run.sh` / `scripts/run.sh` or explicit `docker run -v ...`).
- Model cache is persisted by default using a host bind mount (`./.hf-cache` in repo root) mapped to `/root/.cache/huggingface`.
- Configure cache persistence with `PERSIST_MODEL_CACHE` (`1` default) and `MODEL_CACHE_HOST_DIR`.
- `DOCKER_GPU_MODE` controls GPU flags (`auto` default, `on`, `off`).
- Default `auto` now fails fast if NVIDIA Docker runtime is missing (instead of starting and then crashing).
- `DOCKER_GPU_MODE=off` is only for custom CPU-capable images; this repo's default server enforces CUDA at startup.
- Service listens on container port `8765`.

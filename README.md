# Quick-TTS-Endpoint
Just a quick TTS endpoint

Default model is now set to `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (override with `QWEN_TTS_MODEL`).

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

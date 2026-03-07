# Quick-TTS-Endpoint
Just a quick TTS endpoint

Default model is now set to `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (override with `QWEN_TTS_MODEL`).

## Startup voice snippet training

On startup, the server now attempts to load a training voice snippet and transcript,
then applies those prompt values to each `generate_custom_voice` call:

- `train.wav`
- `tran.txt`

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

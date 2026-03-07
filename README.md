# Quick-TTS-Endpoint
Just a quick TTS endpoint

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

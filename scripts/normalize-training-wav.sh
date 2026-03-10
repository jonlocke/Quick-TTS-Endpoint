#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input-wav-or-basepath> [output-wav]" >&2
  echo "Examples:" >&2
  echo "  $0 sample.wav" >&2
  echo "  $0 sample train.wav" >&2
  exit 1
fi

INPUT_RAW="$1"
OUTPUT_WAV="${2:-train.wav}"

if [[ "$INPUT_RAW" == *.wav ]]; then
  INPUT_WAV="$INPUT_RAW"
else
  INPUT_WAV="${INPUT_RAW}.wav"
fi

if [[ ! -f "$INPUT_WAV" ]]; then
  echo "Error: input file not found: $INPUT_WAV" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found in PATH" >&2
  exit 1
fi

ffmpeg -y -i "$INPUT_WAV" \
  -ac 1 \
  -ar 24000 \
  -af "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11,volume=3dB" \
  -c:a pcm_s16le \
  "$OUTPUT_WAV"

echo "Wrote normalized training audio: $OUTPUT_WAV"

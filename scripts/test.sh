#!/bin/bash

TEXT="$*"

curl -s -X POST "http://127.0.0.1:8765/speak" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"${TEXT//\"/\\\"}\"}" \
  --output out22.wav

ffplay -nodisp -autoexit out22.wav


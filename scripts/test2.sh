#!/bin/bash
curl -X POST "http://127.0.0.1:8765/speak?return_audio=true&play=false" \
  -H "Content-Type: application/json" \
  -d '{"text":"Return wav only."}' --output out.wav
ffplay -nodisp -autoexit out.wav


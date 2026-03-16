#!/bin/bash
docker run -d --name wyoming-piper \
  -p 10200:10200 \
  -v /path/to/local/data:/data \
  rhasspy/wyoming-piper \
  --voice en_US-lessac-medium

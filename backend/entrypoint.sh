#!/usr/bin/env bash
set -e

mkdir -p /weights

# Optional: download weights at container start (recommended vs baking into image)
# Provide these env vars in RunPod:
#   YOLO_URL, REASONER_URL
if [ ! -f "${YOLO_PATH:-/weights/yolo.pt}" ] && [ -n "${YOLO_URL}" ]; then
  echo "Downloading YOLO weights..."
  curl -L "${YOLO_URL}" -o "${YOLO_PATH:-/weights/yolo.pt}"
fi

if [ ! -f "${REASONER_PATH:-/weights/reasoner.pt}" ] && [ -n "${REASONER_URL}" ]; then
  echo "Downloading Reasoner weights..."
  curl -L "${REASONER_URL}" -o "${REASONER_PATH:-/weights/reasoner.pt}"
fi

# Run API (bind to 0.0.0.0 so RunPod can reach it) :contentReference[oaicite:2]{index=2}
exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}

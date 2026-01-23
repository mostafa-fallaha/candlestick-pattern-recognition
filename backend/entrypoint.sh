#!/usr/bin/env bash
set -e

mkdir -p /weights

# Download weights at container start
if [ ! -f "${YOLO_PATH:-/weights/yolo.pt}" ] && [ -n "${YOLO_URL}" ]; then
  echo "Downloading YOLO weights..."
  curl -L "${YOLO_URL}" -o "${YOLO_PATH:-/weights/yolo.pt}"
fi

if [ ! -f "${REASONER_PATH:-/weights/reasoner.pt}" ] && [ -n "${REASONER_URL}" ]; then
  echo "Downloading Reasoner weights..."
  curl -L "${REASONER_URL}" -o "${REASONER_PATH:-/weights/reasoner.pt}"
fi

exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}

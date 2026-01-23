#!/usr/bin/env bash
set -e

echo "=============================================="
echo "Candlestick YOLO + BLIP Backend Starting"
echo "=============================================="

mkdir -p /weights

# Download YOLO weights if not present and URL is provided
YOLO_FILE="${YOLO_PATH:-/weights/yolo.pt}"
if [ ! -f "$YOLO_FILE" ] && [ -n "${YOLO_URL}" ]; then
    echo "Downloading YOLO weights from Azure Blob Storage..."
    curl -L "${YOLO_URL}" -o "$YOLO_FILE"
    echo "YOLO weights downloaded to $YOLO_FILE"
fi

# Download BLIP model folder if not present and URL is provided
# The BLIP model is a HuggingFace checkpoint folder (config.json, model.safetensors, etc.)
BLIP_DIR="${BLIP_PATH:-/weights/blip_model}"
if [ ! -d "$BLIP_DIR" ] || [ -z "$(ls -A $BLIP_DIR 2>/dev/null)" ]; then
    if [ -n "${BLIP_URL}" ]; then
        echo "Downloading BLIP model from Azure Blob Storage..."
        mkdir -p "$BLIP_DIR"

        curl -L "${BLIP_URL}" -o /tmp/blip_model.zip
        unzip -o /tmp/blip_model.zip -d "$BLIP_DIR"
        rm /tmp/blip_model.zip
        
        SUBDIRS=$(find "$BLIP_DIR" -mindepth 1 -maxdepth 1 -type d)
        SUBDIR_COUNT=$(echo "$SUBDIRS" | wc -l)
        
        if [ "$SUBDIR_COUNT" -eq 1 ] && [ -n "$SUBDIRS" ]; then
            echo "Found model files in subdirectory, moving to $BLIP_DIR..."
            mv "$SUBDIRS"/* "$BLIP_DIR"/
            rmdir "$SUBDIRS"
        fi
        
        echo "BLIP model extracted to $BLIP_DIR"
        echo "Contents:"
        ls -la "$BLIP_DIR"
        echo "-- Tree view: --"
        ls "$BLIP_DIR"
    fi
fi

echo "=============================================="
echo "Weights loaded. Starting server on port ${PORT:-8000}"
echo "=============================================="

exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}

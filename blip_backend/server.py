"""
FastAPI server for YOLO + BLIP Candlestick Pattern Recognition

Endpoints:
- GET /health: Health check
- POST /predict: Run YOLO detection + BLIP captioning on an image
"""

import os
import io
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from model import CandlestickPipeline

# Environment configuration
YOLO_PATH = os.environ.get("YOLO_PATH", "/weights/yolo.pt")
BLIP_PATH = os.environ.get("BLIP_PATH", "/weights/blip_model")
API_KEY = os.environ.get("API_KEY", "")

app = FastAPI(
    title="Candlestick YOLO + BLIP API",
    description="Detect candlestick patterns with YOLO and generate captions with BLIP",
    docs_url="/docs",
    redoc_url=None
)

# CORS configuration - allow Streamlit and other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None


@app.on_event("startup")
def startup_event():
    """Initialize the ML pipeline on server startup."""
    global pipeline
    print("=" * 60)
    print("Starting Candlestick YOLO + BLIP Server")
    print("=" * 60)
    pipeline = CandlestickPipeline(YOLO_PATH, BLIP_PATH)
    print("=" * 60)
    print("Server ready to accept requests!")
    print("=" * 60)


def check_api_key(x_api_key: str | None):
    """Validate API key if one is configured."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "model": "yolo+blip"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.25,
    x_api_key: str | None = Header(default=None),
):
    """
    Run YOLO detection + BLIP captioning on an uploaded image.
    
    Args:
        file: Image file to analyze
        conf: YOLO confidence threshold (0.0 to 1.0)
        x_api_key: Optional API key for authentication
    
    Returns:
        PNG image with drawn bounding boxes.
        Response headers contain:
        - X-Caption: Generated BLIP caption
        - X-Detections: JSON array of detection info
    """
    check_api_key(x_api_key)
    
    # Read and decode image
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Run the pipeline
    result = pipeline.predict(img, conf_threshold=conf)
    
    # Encode annotated image as PNG
    _, img_encoded = cv2.imencode(".png", result["annotated_image"])
    
    # Prepare detection info for headers
    # Convert numpy types to native Python types for JSON serialization
    detection_info = []
    for det in result["detections"]:
        bbox = [int(x) for x in det["bbox_xyxy"]]  # Convert int64 to int
        detection_info.append({
            "pattern": det["class_name"],
            "confidence": float(round(det["confidence"], 3)),
            "bbox": bbox
        })
    
    # Build response headers
    # Base64 encode strings that may contain Unicode characters
    caption_b64 = base64.b64encode(result["caption"].encode("utf-8")).decode("ascii")
    detections_b64 = base64.b64encode(json.dumps(detection_info).encode("utf-8")).decode("ascii")
    
    headers = {
        "X-Caption": caption_b64,
        "X-Detections": detections_b64,
        "X-Detection-Count": str(len(result["detections"])),
    }
    
    # Add top detection info if available
    if result["detections"]:
        top_det = result["detections"][0]
        headers["X-Detection-Pattern"] = top_det["class_name"]
        headers["X-Detection-Conf"] = str(round(top_det["confidence"], 3))
    else:
        headers["X-Detection-Pattern"] = "None"
    
    return Response(
        content=img_encoded.tobytes(),
        media_type="image/png",
        headers=headers
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

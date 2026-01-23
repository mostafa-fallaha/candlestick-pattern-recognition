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

YOLO_PATH = os.environ.get("YOLO_PATH", "/weights/yolo.pt")
BLIP_PATH = os.environ.get("BLIP_PATH", "/weights/blip_model")
API_KEY = os.environ.get("API_KEY", "")

app = FastAPI(
    title="Candlestick YOLO + BLIP API",
    description="Detect candlestick patterns with YOLO and generate captions with BLIP",
    docs_url="/docs",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None


@app.on_event("startup")
def startup_event():
    global pipeline
    print("=" * 60)
    print("Starting Candlestick YOLO + BLIP Server")
    print("=" * 60)
    pipeline = CandlestickPipeline(YOLO_PATH, BLIP_PATH)
    print("=" * 60)
    print("Server ready to accept requests!")
    print("=" * 60)


def check_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"ok": True, "model": "yolo+blip"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.25,
    x_api_key: str | None = Header(default=None),
):
    check_api_key(x_api_key)
    
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    result = pipeline.predict(img, conf_threshold=conf)
    
    _, img_encoded = cv2.imencode(".png", result["annotated_image"])
    
    detection_info = []
    for det in result["detections"]:
        bbox = [int(x) for x in det["bbox_xyxy"]]
        detection_info.append({
            "pattern": det["class_name"],
            "confidence": float(round(det["confidence"], 3)),
            "bbox": bbox
        })
    
    caption_b64 = base64.b64encode(result["caption"].encode("utf-8")).decode("ascii")
    detections_b64 = base64.b64encode(json.dumps(detection_info).encode("utf-8")).decode("ascii")
    
    headers = {
        "X-Caption": caption_b64,
        "X-Detections": detections_b64,
        "X-Detection-Count": str(len(result["detections"])),
    }
    
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

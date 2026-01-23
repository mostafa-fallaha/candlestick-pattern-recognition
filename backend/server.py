import os
import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from model import InferenceEngine

YOLO_PATH = os.environ.get("YOLO_PATH", "/weights/yolo.pt")
REASONER_PATH = os.environ.get("REASONER_PATH", "/weights/reasoner.pt")
API_KEY = os.environ.get("API_KEY", "")

app = FastAPI(title="Candlestick Reasoner API", docs_url="/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None

@app.on_event("startup")
def _startup():
    global engine
    engine = InferenceEngine(YOLO_PATH, REASONER_PATH)

def check_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health():
    return {"ok": True}


def draw_prediction_on_image(img_bgr: np.ndarray, det: dict) -> np.ndarray:
    img = img_bgr.copy()
    x1, y1, x2, y2 = det["bbox_xyxy"]
    
    color = (255, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    pattern = det["pattern"]
    conf = det["yolo_conf"]
    label = f"{pattern} {conf:.2f}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_x = x1
    text_y = max(30, y1 - 8)
    
    cv2.putText(img, label, (text_x, text_y), font, font_scale, color, thickness)
    
    return img


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = 0.25,
    x_api_key: str | None = Header(default=None),
):
    check_key(x_api_key)

    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    dets = engine.predict(img, yolo_conf=conf)
    
    # If detection exists, draw on image and return as PNG
    if dets:
        annotated_img = draw_prediction_on_image(img, dets[0])
        _, img_encoded = cv2.imencode(".png", annotated_img)
        
        return Response(
            content=img_encoded.tobytes(),
            media_type="image/png",
            headers={
                "X-Detection-Pattern": dets[0]["pattern"],
                "X-Detection-Action": dets[0]["action"],
                "X-Detection-Conf": str(dets[0]["yolo_conf"]),
                "X-Detection-Explanation": dets[0]["explanation"],
            }
        )
    else:
        # No detection - return original image
        _, img_encoded = cv2.imencode(".png", img)
        return Response(
            content=img_encoded.tobytes(),
            media_type="image/png",
            headers={"X-Detection-Pattern": "None"}
        )

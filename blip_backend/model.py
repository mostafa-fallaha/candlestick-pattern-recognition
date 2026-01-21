"""
YOLO + BLIP Pipeline for Candlestick Pattern Recognition

This module implements:
1. YOLODetector: Detects candlestick patterns and draws bounding boxes
2. BLIPCaptioner: Generates captions from annotated images
3. CandlestickPipeline: Orchestrates the full detection + captioning pipeline
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor


# Class names for YOLO detection (must match training order)
CLASSES = ["Spinning Top", "Hikkake Pattern", "Three Outside Up-Down", "Marubozu",
    "Gravestone Doji", "Dragonfly Doji", "Hammer", "Hanging Man",
    "Three Inside Up-Down", "Advance Block", "Upside-Downside Gap Three Methods",
    "Evening Star"
]

class YOLODetector:
    """YOLO-based candlestick pattern detector."""
    
    def __init__(self, model_path: str):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to the YOLO weights file (.pt)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[YOLODetector] Loading model from {model_path} on {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print("[YOLODetector] Model loaded successfully")
    
    def detect(self, img_bgr: np.ndarray, conf_threshold: float = 0.25) -> list[dict]:
        """
        Run detection on an image and return only the highest confidence detection.
        
        Args:
            img_bgr: OpenCV image in BGR format
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List with single detection dictionary (highest confidence) or empty list.
            Detection dict has keys:
            - bbox_xyxy: [x1, y1, x2, y2] bounding box
            - class_id: Integer class ID
            - class_name: String class name
            - confidence: Detection confidence
        """
        results = self.model(img_bgr, conf=conf_threshold, verbose=False)[0]
        
        if results.boxes is None or len(results.boxes) == 0:
            return []
        
        # Get all detections
        boxes_xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy().astype(float)
        
        # Find the detection with highest confidence
        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes_xyxy[best_idx]
        class_id = int(cls_ids[best_idx])
        confidence = float(confs[best_idx])
        
        # Get class name safely
        class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"Class_{class_id}"
        
        return [{
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence
        }]
    
    def draw_detections(self, img_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            img_bgr: OpenCV image in BGR format
            detections: List of detection dictionaries
            
        Returns:
            Annotated image with bounding boxes and labels
        """
        img = img_bgr.copy()
        
        # Magenta color (BGR format)
        color = (255, 0, 255)
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name} {confidence:.2f}"
            
            # Draw text above bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            # Get text size for background rectangle
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Position text at top of bounding box
            text_x = x1
            text_y = max(text_h + 5, y1 - 8)
            
            # Draw text (green color for visibility like in the reference image)
            cv2.putText(img, label, (text_x, text_y), font, font_scale, color, thickness)
        
        return img


class BLIPCaptioner:
    """BLIP-based image captioning model (fine-tuned for candlestick patterns)."""
    
    def __init__(self, model_path: str):
        """
        Initialize the BLIP captioner.
        
        Args:
            model_path: Path to the BLIP model directory (HuggingFace format)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BLIPCaptioner] Loading model from {model_path} on {self.device}")
        
        # Load model, tokenizer, and processor using Auto classes (same as training)
        self.model = AutoModelForVision2Seq.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print("[BLIPCaptioner] Model loaded successfully")
    
    def generate_caption(self, img_bgr: np.ndarray, max_new_tokens: int = 50) -> str:
        """
        Generate a caption for the given image.
        
        Args:
            img_bgr: OpenCV image in BGR format
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated caption string
        """
        # Convert BGR to RGB PIL Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # Ensure RGB mode
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Debug: log image stats
        print(f"[BLIPCaptioner] Image size: {pil_image.size}, mode: {pil_image.mode}")
        
        # Process image (processor handles resizing and normalization)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Debug: log input tensor stats
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            print(f"[BLIPCaptioner] Input tensor shape: {pixel_values.shape}, "
                  f"min: {pixel_values.min().item():.3f}, max: {pixel_values.max().item():.3f}")
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
        
        # Decode using tokenizer (same as in Colab)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"[BLIPCaptioner] Generated: {generated_text}")
        
        return generated_text


class CandlestickPipeline:
    """
    Full pipeline: YOLO detection → Draw annotations → BLIP captioning
    """
    
    def __init__(self, yolo_path: str, blip_path: str):
        """
        Initialize the full pipeline.
        
        Args:
            yolo_path: Path to YOLO weights file
            blip_path: Path to BLIP model directory
        """
        print("[CandlestickPipeline] Initializing pipeline...")
        self.detector = YOLODetector(yolo_path)
        self.captioner = BLIPCaptioner(blip_path)
        print("[CandlestickPipeline] Pipeline ready!")
    
    def predict(
        self, 
        img_bgr: np.ndarray, 
        conf_threshold: float = 0.25
    ) -> dict:
        """
        Run the full prediction pipeline.
        
        Args:
            img_bgr: OpenCV image in BGR format
            conf_threshold: YOLO confidence threshold
            
        Returns:
            Dictionary with:
            - annotated_image: Image with drawn bounding boxes (np.ndarray)
            - detections: List of detection dictionaries
            - caption: Generated caption text
        """
        # Step 1: Run YOLO detection
        detections = self.detector.detect(img_bgr, conf_threshold)
        
        # Step 2: Draw bounding boxes on image
        if detections:
            annotated_image = self.detector.draw_detections(img_bgr, detections)
        else:
            # No detections - use original image
            annotated_image = img_bgr.copy()
        
        # Step 3: Generate caption using BLIP on the annotated image
        caption = self.captioner.generate_caption(annotated_image)
        
        return {
            "annotated_image": annotated_image,
            "detections": detections,
            "caption": caption
        }

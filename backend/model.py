import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from ultralytics import YOLO

CLASSES = ["Spinning Top", "Hikkake Pattern", "Three Outside Up-Down", "Marubozu",
    "Gravestone Doji", "Dragonfly Doji", "Hammer", "Hanging Man",
    "Three Inside Up-Down", "Advance Block", "Upside-Downside Gap Three Methods",
    "Evening Star"
]

ACTIONS = ["HOLD", "BUY", "SELL"]
id2action = {i:a for i,a in enumerate(ACTIONS)}

# Pattern classification sets
BULLISH_ALWAYS = {"Hammer", "Dragonfly Doji"}
BEARISH_ALWAYS = {"Hanging Man", "Gravestone Doji", "Evening Star", "Advance Block"}
NEUTRAL_ALWAYS = {"Spinning Top"}
BOTH_CHECK_TREND = {"Three Outside Up-Down", "Three Inside Up-Down", "Upside-Downside Gap Three Methods", "Hikkake Pattern"}
BOTH_CHECK_COLOR = {"Marubozu"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def bgr_to_tensor_norm(img_bgr: np.ndarray) -> torch.Tensor:
    img = img_bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
    t = torch.from_numpy(img).permute(2,0,1)
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(3,1,1)
    return (t - mean) / std

def expand_xyxy(xyxy, W, H, scale=2.5):
    x1,y1,x2,y2 = xyxy
    cx = (x1+x2)/2
    cy = (y1+y2)/2
    bw = (x2-x1)
    bh = (y2-y1)
    nw = bw * scale
    nh = bh * scale
    nx1 = int(max(0, cx - nw/2))
    ny1 = int(max(0, cy - nh/2))
    nx2 = int(min(W-1, cx + nw/2))
    ny2 = int(min(H-1, cy + nh/2))
    return [nx1, ny1, nx2, ny2]

def safe_crop(img_bgr, xyxy):
    x1,y1,x2,y2 = xyxy
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2].copy()

def estimate_trend_direction(ctx_bgr):
    """
    Pure CV heuristic:
    - build a mask of non-white pixels
    - per x column: median y of mask pixels (a crude price trace)
    - slope sign -> up/down/flat
    """
    if ctx_bgr is None or ctx_bgr.size == 0:
        return "flat"

    H, W, _ = ctx_bgr.shape
    left_W = max(10, int(W * 0.45))
    region = ctx_bgr[:, :left_W, :]

    s = region[:,:,0].astype(np.int32) + region[:,:,1].astype(np.int32) + region[:,:,2].astype(np.int32)
    mask = s < 740

    ys = []
    xs = []
    for x in range(mask.shape[1]):
        col = mask[:, x]
        if col.any():
            y_idx = np.where(col)[0]
            ys.append(np.median(y_idx))
            xs.append(x)

    if len(xs) < 8:
        return "flat"

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    if len(ys) >= 9:
        k = 9
        pad = k//2
        ypad = np.pad(ys, (pad,pad), mode="edge")
        ys = np.convolve(ypad, np.ones(k)/k, mode="valid")

    slope = np.polyfit(xs, ys, 1)[0]

    eps = 0.03
    if slope > eps:
        return "down"
    if slope < -eps:
        return "up"
    return "flat"

def dominant_candle_color(roi_bgr):
    """
    Decide bullish vs bearish for Marubozu by dominant green vs red pixels.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # crude red/green thresholds
    red1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    green = cv2.inRange(hsv, (35, 60, 50), (85, 255, 255))

    r = int((red > 0).sum())
    g = int((green > 0).sum())
    if r == 0 and g == 0:
        return "unknown"
    return "green" if g >= r else "red"

def resolve_action(pattern_name, trend_dir, roi_color):
    """Resolve action based on pattern type and context."""
    if pattern_name in BULLISH_ALWAYS:
        return "BUY"
    if pattern_name in BEARISH_ALWAYS:
        return "SELL"
    if pattern_name in NEUTRAL_ALWAYS:
        return "HOLD"

    if pattern_name in BOTH_CHECK_TREND:
        if trend_dir == "down": return "BUY"
        if trend_dir == "up":   return "SELL"
        return "HOLD"

    if pattern_name in BOTH_CHECK_COLOR:
        if roi_color == "green": return "BUY"
        if roi_color == "red":   return "SELL"
        return "HOLD"

    return "HOLD"

def make_explanation(pattern_name, action, trend_dir, roi_color):
    """Generate rich explanation based on pattern type and reasoning."""
    if pattern_name in BOTH_CHECK_TREND:
        if action == "BUY":
            return f"{pattern_name}: can be bullish/bearish. Prior trend looks DOWN -> bullish reversal -> BUY."
        if action == "SELL":
            return f"{pattern_name}: can be bullish/bearish. Prior trend looks UP -> bearish reversal -> SELL."
        return f"{pattern_name}: ambiguous trend -> HOLD."
    
    if pattern_name in BOTH_CHECK_COLOR:
        if action == "BUY":
            return f"{pattern_name}: direction depends on candle color. Detected GREEN (bullish) body -> BUY."
        if action == "SELL":
            return f"{pattern_name}: direction depends on candle color. Detected RED (bearish) body -> SELL."
        return f"{pattern_name}: candle color unclear -> HOLD."
    
    if pattern_name in BULLISH_ALWAYS:
        return f"{pattern_name}: typically bullish reversal/continuation signal -> BUY."
    
    if pattern_name in BEARISH_ALWAYS:
        return f"{pattern_name}: typically bearish reversal signal -> SELL."
    
    if pattern_name in NEUTRAL_ALWAYS:
        return f"{pattern_name}: neutral/indecision pattern -> HOLD."
    
    return f"{pattern_name}: insufficient context -> {action}."

def strip_orig_mod_prefix(state_dict):
    """Remove '_orig_mod.' prefix added by torch.compile() from state_dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class TwoBranchReasoner(nn.Module):
    def __init__(self, backbone_name="vit_small_patch16_224", num_patterns=12, num_actions=3):
        super().__init__()
        # Two separate backbones: one for ROI, one for context
        self.backbone_roi = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.backbone_ctx = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        D = self.backbone_roi.num_features
        self.pattern_emb = nn.Embedding(num_patterns, 64)
        self.head = nn.Sequential(
            nn.LayerNorm(D*2 + 64),
            nn.Linear(D*2 + 64, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_actions),
        )

    def forward(self, roi, ctx, pattern_id):
        f_roi = self.backbone_roi(roi)
        f_ctx = self.backbone_ctx(ctx)
        p = self.pattern_emb(pattern_id)
        return self.head(torch.cat([f_roi, f_ctx, p], dim=1))

class InferenceEngine:
    def __init__(self, yolo_path: str, reasoner_ckpt: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Speed toggles
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        self.amp_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Load YOLO
        self.yolo = YOLO(yolo_path)

        # Load reasoner
        ckpt = torch.load(reasoner_ckpt, map_location="cpu")
        backbone = ckpt.get("backbone", "vit_small_patch16_224")

        self.reasoner = TwoBranchReasoner(backbone_name=backbone, num_patterns=len(CLASSES), num_actions=len(ACTIONS))
        state_dict = strip_orig_mod_prefix(ckpt["model_state"])
        self.reasoner.load_state_dict(state_dict, strict=True)
        self.reasoner.eval().to(self.device)

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray, yolo_conf=0.25, ctx_scale=2.5, size=224):
        H, W = img_bgr.shape[:2]
        results = self.yolo.predict(source=img_bgr, conf=yolo_conf, verbose=False)
        r0 = results[0]

        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        boxes_xyxy = r0.boxes.xyxy.detach().cpu().numpy().astype(int)
        cls_ids = r0.boxes.cls.detach().cpu().numpy().astype(int)
        confs = r0.boxes.conf.detach().cpu().numpy().astype(float)

        # Find the detection with highest confidence
        best_idx = int(np.argmax(confs))
        xyxy = boxes_xyxy[best_idx]
        cls_id = cls_ids[best_idx]
        confv = confs[best_idx]

        if cls_id < 0 or cls_id >= len(CLASSES):
            return []

        xyxy = [int(x) for x in xyxy.tolist()]
        xyxy_ctx = expand_xyxy(xyxy, W, H, scale=ctx_scale)

        roi = safe_crop(img_bgr, xyxy)
        ctx = safe_crop(img_bgr, xyxy_ctx)
        if roi is None or ctx is None:
            return []

        # Get pattern name and analyze context
        pattern_name = CLASSES[cls_id]
        trend = estimate_trend_direction(ctx)
        roi_color = dominant_candle_color(roi)

        # Resize for reasoner
        roi_res = cv2.resize(roi, (size,size), interpolation=cv2.INTER_AREA)
        ctx_res = cv2.resize(ctx, (size,size), interpolation=cv2.INTER_AREA)

        roi_t = bgr_to_tensor_norm(roi_res).unsqueeze(0).to(self.device)
        ctx_t = bgr_to_tensor_norm(ctx_res).unsqueeze(0).to(self.device)
        pid_t = torch.tensor([cls_id], dtype=torch.long, device=self.device)

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=(self.device=="cuda")):
            logits = self.reasoner(roi_t, ctx_t, pid_t)
        action_id = int(logits.argmax(dim=1).item())
        reasoner_action = id2action[action_id]

        # Use rule-based action resolution for final decision
        rule_action = resolve_action(pattern_name, trend, roi_color)

        # Generate rich explanation
        explanation = make_explanation(pattern_name, rule_action, trend, roi_color)

        return [{
            "pattern": pattern_name,
            "pattern_id": int(cls_id),
            "yolo_conf": float(confv),
            "bbox_xyxy": xyxy,
            "action": rule_action,
            "reasoner_action": reasoner_action,
            "trend": trend,
            "candle_color": roi_color,
            "explanation": explanation,
        }]

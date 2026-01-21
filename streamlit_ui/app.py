import streamlit as st
import requests
import json
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Candlestick Pattern Analyzer", layout="wide")

st.title("🕯️ Candlestick Pattern Analyzer")

# Backend configuration
st.sidebar.header("⚙️ Settings")

# Backend selection
backend_type = st.sidebar.radio(
    "Select Backend",
    ["Reasoner (YOLO + Context)", "BLIP (YOLO + Caption)"],
    help="Choose which model backend to use for analysis"
)

# Backend URLs from secrets or defaults
if backend_type == "Reasoner (YOLO + Context)":
    BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")
    BACKEND_API_KEY = st.secrets.get("BACKEND_API_KEY", "")
    st.sidebar.info("📊 Using Reasoner backend: YOLO detection + context-based action reasoning")
else:
    BACKEND_URL = st.secrets.get("BLIP_BACKEND_URL", "http://localhost:8001")
    BACKEND_API_KEY = st.secrets.get("BLIP_BACKEND_API_KEY", "")
    st.sidebar.info("🤖 Using BLIP backend: YOLO detection + AI-generated captions")

# Confidence slider
conf = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

# File uploader
st.markdown("---")
uploaded = st.file_uploader("📁 Upload chart image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Input Image")
        st.image(uploaded, use_column_width=True)

    with col2:
        st.subheader("📤 Prediction Result")
        
        with st.spinner("Running inference..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            headers = {"x-api-key": BACKEND_API_KEY} if BACKEND_API_KEY else {}

            try:
                r = requests.post(
                    f"{BACKEND_URL}/predict",
                    params={"conf": conf},
                    headers=headers,
                    files=files,
                    timeout=120
                )
                
                if r.status_code != 200:
                    st.error(f"Backend error {r.status_code}: {r.text}")
                else:
                    # Display annotated image
                    result_img = Image.open(BytesIO(r.content))
                    st.image(result_img, use_column_width=True)
                    
                    # Show detection details from headers
                    pattern = r.headers.get("X-Detection-Pattern", "None")
                    
                    st.markdown("---")
                    
                    if pattern != "None":
                        if backend_type == "BLIP (YOLO + Caption)":
                            # BLIP Backend response
                            # Decode base64-encoded headers
                            caption_b64 = r.headers.get("X-Caption", "")
                            caption = base64.b64decode(caption_b64).decode("utf-8") if caption_b64 else "No caption generated"
                            
                            detection_count = r.headers.get("X-Detection-Count", "0")
                            detections_b64 = r.headers.get("X-Detections", "")
                            detections_json = base64.b64decode(detections_b64).decode("utf-8") if detections_b64 else "[]"
                            
                            st.markdown(f"**🎯 Detected Patterns:** {detection_count}")
                            
                            # Parse and display detections
                            try:
                                detections = json.loads(detections_json)
                                for i, det in enumerate(detections, 1):
                                    conf_val = det.get("confidence", 0)
                                    st.markdown(f"  {i}. **{det['pattern']}** ({conf_val:.1%})")
                            except (json.JSONDecodeError, Exception):
                                pass
                            
                            st.markdown("---")
                            st.markdown("### 🤖 AI-Generated Caption")
                            st.info(caption)
                            
                        else:
                            # Reasoner Backend response
                            action = r.headers.get("X-Detection-Action", "N/A")
                            conf_val = r.headers.get("X-Detection-Conf", "N/A")
                            explanation = r.headers.get("X-Detection-Explanation", "N/A")
                            
                            # Color the action
                            if action == "BUY":
                                action_display = f"🟢 **{action}**"
                            elif action == "SELL":
                                action_display = f"🔴 **{action}**"
                            else:
                                action_display = f"🟠 **{action}**"
                            
                            st.markdown(f"**Pattern:** {pattern}")
                            st.markdown(f"**Confidence:** {float(conf_val):.2%}")
                            st.markdown(f"**Action:** {action_display}")
                            st.markdown(f"**Explanation:** {explanation}")
                    else:
                        st.warning("No candlestick pattern detected in this image.")
                        
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to backend at {BACKEND_URL}. Is it running?")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The backend might be overloaded.")

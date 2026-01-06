import streamlit as st
import requests
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Candlestick Pattern Reasoner", layout="wide")

st.title("🕯️ Candlestick Pattern Reasoner")

# Backend configuration
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")
BACKEND_API_KEY = st.secrets.get("BACKEND_API_KEY", "")

conf = st.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
uploaded = st.file_uploader("Upload chart image", type=["png", "jpg", "jpeg", "webp"])

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
                    
                    if pattern != "None":
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
                        
                        st.markdown("---")
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

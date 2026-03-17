import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ----------------------------
# Load Model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # lightweight & fast
    return model

model = load_model()

# ----------------------------
# UI Layout
# ----------------------------
st.set_page_config(page_title="Fridge Object Detection", layout="wide")

st.title("🧊 Fridge Object Detection App")
st.markdown("Lade ein Bild deines Kühlschranks hoch und erkenne die enthaltenen Objekte.")

# Sidebar
st.sidebar.header("⚙️ Einstellungen")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# File uploader
uploaded_file = st.file_uploader("📸 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Originalbild")
        st.image(image, use_column_width=True)

    # ----------------------------
    # Prediction
    # ----------------------------
    results = model(image_np, conf=confidence_threshold)

    annotated_frame = results[0].plot()

    with col2:
        st.subheader("Erkennung")
        st.image(annotated_frame, use_column_width=True)

    # ----------------------------
    # Detected Objects List
    # ----------------------------
    st.subheader("🔍 Erkannte Objekte")

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            st.write(f"**{label}** — Confidence: {conf:.2f}")
    else:
        st.warning("Keine Objekte erkannt.")

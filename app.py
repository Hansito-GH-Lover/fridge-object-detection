import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

st.set_page_config(page_title="Kühlschrank-Objekterkennung", layout="wide")
st.title("🧊 Kühlschrank-Objekterkennung (YOLOv8)")

# Sidebar
st.sidebar.header("Einstellungen")
uploaded_file = st.sidebar.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    results = model(img_array)[0]

    # Copy image for drawing
    draw = ImageDraw.Draw(image)
    colors = ["red", "green", "blue", "orange", "purple"]

    detections = []

    for i, box in enumerate(results.boxes):
        conf = float(box.conf)
        cls = int(box.cls)
        label = model.names[cls]
        if conf >= confidence_threshold:
            detections.append((label, conf))
            x1, y1, x2, y2 = box.xyxy[0]
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 12), f"{label} {int(conf*100)}%", fill=color)

    # Sort detections
    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    # Layout mit Spalten für erkannte Objekte
    st.subheader("Erkannte Objekte")
    if detections:
        cols = st.columns(len(detections))
        for col, (label, conf) in zip(cols, detections):
            col.metric(label, f"{int(conf*100)}%")
    else:
        st.write("Keine Objekte erkannt.")

    st.subheader("Visualisierung")
    st.image(image, use_column_width=True)

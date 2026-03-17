import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

st.title("Kühlschrank-Objekterkennung (YOLOv8)")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Originalbild", use_column_width=True)

    img_array = np.array(image)

    results = model(img_array)[0]

    detections = []

    # Bounding Boxes manuell mit PIL zeichnen
    draw = ImageDraw.Draw(image)

    for box in results.boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        label = model.names[cls]
        if conf > 0.5:
            detections.append((label, conf))
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding Box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1-10), f"{label} {int(conf*100)}%", fill="red")

    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    st.subheader("Erkannte Objekte")
    if detections:
        for label, conf in detections:
            st.write(f"{label} – {int(conf*100)}%")
    else:
        st.write("Keine Objekte erkannt (Confidence > 0.5)")

    st.subheader("Visualisierung")
    st.image(image, caption="Erkannte Objekte mit Bounding Boxes", use_column_width=True)

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Reality Scanner AI",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Reality Scanner AI")
st.caption("Deep Learning Computer Vision System")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "📷 Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 SCAN REALITY", use_container_width=True):

        with st.spinner("🧠 AI is analyzing your reality..."):

            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as temp:

                image.save(temp.name)
                image_path = temp.name

            # Run YOLO
            results = model(image_path)

            result = results[0]

            # Generate annotated image
            annotated = result.plot()

            # Remove temporary file
            os.remove(image_path)

        st.success("✅ Scan complete!")

        st.subheader("🧠 AI Vision")

        st.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )

        st.subheader("🔎 Detected Objects")

        if len(result.boxes) == 0:

            st.warning("No objects detected.")

        else:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                label = model.names[class_id]

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**{label.upper()}**")

                with col2:
                    st.write(f"{confidence:.1%}")

                st.progress(confidence)
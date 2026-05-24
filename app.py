import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Page config
st.set_page_config(
    page_title="Weapon Detector",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 Weapon Detection System")
st.write("Upload an image to detect pistols, knives, cards, or phones.")

# Load model
model = YOLO("best.pt")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temporarily
    temp_file = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".jpg")

    image.save(temp_file.name)

    # Prediction
    with st.spinner("Detecting objects..."):
        results = model.predict(
            source=temp_file.name,
            conf=0.25
        )

    # Plot results
    plotted = results[0].plot()

    # Show prediction
    st.image(
        plotted,
        caption="Detection Result",
        use_container_width=True
    )

    # Show detected classes
    st.subheader("Detected Objects")

    names = model.names

    detected = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detected.append(
            f"{names[cls]} ({conf:.2f})"
        )

    if detected:
        for item in detected:
            st.write("✅", item)
    else:
        st.write("No objects detected.")
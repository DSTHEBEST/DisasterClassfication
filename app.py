import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model_2.h5")

model = load_model()
class_labels = ["Cyclone", "Earthquake", "Flood", "Wildfire"]

# Define Evacuation Mechanism
evacuation_guidelines = {
    "Cyclone": "🌪️ **Cyclone Safety**:\n- Move to higher ground.\n- Avoid flood-prone areas.\n- Stay indoors & away from windows.\n- Follow emergency alerts.",
    "Earthquake": "🌍 **Earthquake Safety**:\n- Drop, cover, and hold on!\n- Stay indoors if inside, move to an open area if outside.\n- Avoid elevators and weak structures.",
    "Flood": "🌊 **Flood Safety**:\n- Move to higher ground immediately.\n- Avoid walking or driving through floodwaters.\n- Turn off electricity & gas.",
    "Wildfire": "🔥 **Wildfire Safety**:\n- Evacuate early if advised.\n- Wear protective clothing.\n- Keep emergency supplies ready.\n- Close doors & windows to prevent smoke entry."
}

# Streamlit UI
st.title("🌍 Disaster Image Classification & Evacuation System")
st.write("Upload an image, and the model will predict the disaster type and provide evacuation steps.")

# Upload Image
uploaded_file = st.file_uploader("Choose a disaster image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (No OpenCV)
    img_resized = image.resize((128, 128))  # Resize using PIL
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_input = np.expand_dims(img_array, axis=0)  # Expand dims for model input

    # Make prediction
    prediction = model.predict(img_input)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  # Convert to percentage

    # Show prediction
    st.subheader(f"🚨 Predicted Disaster: **{predicted_class}**")
    st.write(f"🔍 Confidence Level: **{confidence:.2f}%**")

    # Show evacuation steps
    st.info(evacuation_guidelines[predicted_class])

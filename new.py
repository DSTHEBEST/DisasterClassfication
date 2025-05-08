import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import googlemaps
import os


model = tf.keras.models.load_model("cnn_model_finetuned.h5")


CLASS_NAMES = ["Cyclone", "Earthquake", "Flood", "Wildfire"]


evacuation_guidelines = {
    "Cyclone": "\U0001F32A **Cyclone Safety**:\n- Move to higher ground.\n- Avoid flood-prone areas.\n- Stay indoors & away from windows.\n- Follow emergency alerts.",
    "Earthquake": "\U0001F30D **Earthquake Safety**:\n- Drop, cover, and hold on!\n- Stay indoors if inside, move to an open area if outside.\n- Avoid elevators and weak structures.",
    "Flood": "\U0001F30A **Flood Safety**:\n- Move to higher ground immediately.\n- Avoid walking or driving through floodwaters.\n- Turn off electricity & gas.",
    "Wildfire": "\U0001F525 **Wildfire Safety**:\n- Evacuate early if advised.\n- Wear protective clothing.\n- Keep emergency supplies ready.\n- Close doors & windows to prevent smoke entry."
}
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
def get_safe_route(current_location, safe_location, api_key):
    gmaps = googlemaps.Client(key=api_key)
    directions = gmaps.directions(current_location, safe_location, mode="driving")
    if directions:
        return f"[View Safe Route](https://www.google.com/maps/dir/{current_location}/{safe_location})"
    else:
        return "No route found. Check location inputs."

st.set_page_config(page_title="Disaster Classification & Evacuation Guide", layout="wide")
st.title("🌍 Disaster Classification & Evacuation Guide")


st.sidebar.header("🗺️ Evacuation Route Planner")
current_location = st.sidebar.text_input("Enter your current location:")
safe_location = st.sidebar.text_input("Enter a safe location:")
api_key = st.sidebar.text_input("Enter Google Maps API Key:", type="password")

if st.sidebar.button("Find Safe Route"):
    if current_location and safe_location and api_key:
        route_link = get_safe_route(current_location, safe_location, api_key)
        st.sidebar.markdown(route_link, unsafe_allow_html=True)
    else:
        st.sidebar.error("Please enter valid locations and API key.")


uploaded_file = st.file_uploader("Upload an image of a disaster", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)


    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Predicted Disaster: **{predicted_class}** ({confidence:.2f}% confidence)")


    st.info(evacuation_guidelines[predicted_class])

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("model.h5")

st.title("Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image")

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    st.write("Prediction:", prediction)

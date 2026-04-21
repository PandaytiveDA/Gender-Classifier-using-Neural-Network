import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model safely
@st.cache_resource
def load_my_model():
    return load_model("gender_classifier_model.h5")

model = load_my_model()

st.title("Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ⚠️ CHANGE THIS SIZE based on your model
    IMG_SIZE = 224  

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    st.write("Raw prediction:", prediction)

    # If classification model
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

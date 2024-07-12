import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('digit_letter_classifier.h5')

st.title("Digit and Letter Classifier")
st.write("Upload an image of a handwritten digit or letter.")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(28, 28), color_mode="grayscale")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    if class_idx < 10:
        st.write(f"Prediction: {class_idx} (Digit)")
    else:
        st.write(f"Prediction: {chr(class_idx + 87)} (Letter)")

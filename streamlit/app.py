import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load the ONNX model
model_path = "handwriting_recognition_model.onnx"
session = ort.InferenceSession(model_path)

# Define the model input shape
input_shape = (128, 32)

# Function to preprocess the image
def preprocess_image(image):
    img = Image.fromarray(image.astype('uint8'), 'RGBA').convert('L')
    img = img.resize((input_shape[1], input_shape[0]))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    return img

# Function to predict the text from the image
def predict(image):
    preprocessed_img = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: preprocessed_img})
    prediction = result[0]
    predicted_text = decode_prediction(prediction)
    return predicted_text

# Function to decode the prediction
def decode_prediction(prediction):
    # You need to implement your own decoding logic based on your model's output
    # Here we assume a dummy implementation for illustration purposes
    predicted_text = "decoded_text"  # Replace this with actual decoding logic
    return predicted_text

# Streamlit interface
st.title("Handwriting Recognition")
st.write("Draw a word and click 'Predict' to see the prediction.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=128,
    width=32 * 16,  # Adjust width according to max text length
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    if st.button("Predict"):
        prediction = predict(img)
        st.write(f"Prediction: {prediction}")

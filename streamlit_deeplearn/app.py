import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests
import gzip
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# URLs of the dataset files on Google Drive
images_url = 'https://drive.google.com/uc?export=download&id=1SKjgdI4-t72-njFZFp1zXuKUDP2NwbYD'
labels_url = 'https://drive.google.com/uc?export=download&id=1R9j8F0LTQYsyK3YthqThLk5Fg5IcqkmR' # example link, replace with actual link

images_path = 'emnist-letters-train-images-idx3-ubyte.gz'
labels_path = 'emnist-letters-train-labels-idx1-ubyte.gz'

# Function to download file from Google Drive
def download_file_from_google_drive(url, destination):
    session = requests.Session()
    response = session.get(url, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        url = url + '&confirm=' + token
        response = session.get(url, stream=True)

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(32768):
            f.write(chunk)

# Download the images file if it does not exist
if not os.path.exists(images_path):
    with st.spinner("Downloading images file..."):
        download_file_from_google_drive(images_url, images_path)

# Download the labels file if it does not exist
if not os.path.exists(labels_path):
    with st.spinner("Downloading labels file..."):
        download_file_from_google_drive(labels_url, labels_path)

# Helper function to load EMNIST data
def load_emnist(images_path, labels_path):
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
        
    return images, labels

# Load the dataset
x_train_emnist, y_train_emnist = load_emnist(images_path, labels_path)

# Load and preprocess other data as before
# Load MNIST data
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

def preprocess_data(x, y):
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, -1)
    y = y.astype('int')
    return x, y

x_train_mnist, y_train_mnist = preprocess_data(x_train_mnist, y_train_mnist)
x_test_mnist, y_test_mnist = preprocess_data(x_test_mnist, y_test_mnist)

x_train_emnist, y_train_emnist = preprocess_data(x_train_emnist, y_train_emnist)
# Shift EMNIST labels to avoid collision with MNIST labels (0-9)
y_train_emnist += 9

# Combine datasets
x_train = np.concatenate([x_train_mnist, x_train_emnist], axis=0)
y_train = np.concatenate([y_train_mnist, y_train_emnist], axis=0)

# Load the model
model_path = 'digit_letter_classifier.h5'
model = load_model(model_path)

st.title("Digit and Letter Classifier")
st.write("Draw a digit or a letter and click 'Predict' to see the prediction.")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000000",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Preprocess the drawn image
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Predict the class
    if st.button("Predict"):
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        if class_idx < 10:
            st.write(f"Prediction: {class_idx} (Digit)")
        else:
            st.write(f"Prediction: {chr(class_idx + 87)} (Letter)")

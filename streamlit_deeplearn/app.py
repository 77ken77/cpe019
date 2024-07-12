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

# URL of the dataset file on Google Drive
dataset_url = 'https://drive.google.com/uc?export=download&id=1SKjgdI4-t72-njFZFp1zXuKUDP2NwbYD'
dataset_path = 'emnist-letters-train-images-idx3-ubyte.gz'

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

# Download the dataset if it does not exist
if not os.path.exists(dataset_path):
    with st.spinner("Downloading dataset..."):
        download_file_from_google_drive(dataset_url, dataset_path)

# Helper function to load EMNIST data
def load_emnist(images_path, labels_path):
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
        
    return images, labels

# Ensure that the labels file is available too
labels_path = 'emnist-letters-train-labels-idx1-ubyte.gz'
# You can add similar code to download the labels file if necessary

# Load the dataset
x_train_emnist, y_train_emnist = load_emnist(dataset_path, labels_path)

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

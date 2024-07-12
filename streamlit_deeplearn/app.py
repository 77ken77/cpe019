import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests
import gzip

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
        
  

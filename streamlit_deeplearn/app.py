import os
import stow
import tarfile
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Model
import numpy as np
import cv2
from mltu.utils.text_utils import ctc_decoder, get_cer
from mltu.inferenceModel import OnnxInferenceModel
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Download and unzip the dataset
def download_and_unzip(url, extract_to='Datasets', chunk_size=1024*1024):
    http_response = urlopen(url)
    data = b''
    iterations = http_response.length // chunk_size + 1
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)
    zipfile = ZipFile(BytesIO(data))
    zipfile.extractall(path=extract_to)

dataset_path = stow.join('Datasets', 'IAM_Words')
if not stow.exists(dataset_path):
    download_and_unzip('https://git.io/J0fjL', extract_to='Datasets')
    file = tarfile.open(stow.join(dataset_path, "words.tgz"))
    file.extractall(stow.join(dataset_path, "words"))

# Preprocess the dataset
dataset, vocab, max_len = [], set(), 0
words = open(stow.join(dataset_path, "words.txt"), "r").readlines()
for line in tqdm(words):
    if line.startswith("#"):
        continue
    line_split = line.split(" ")
    if line_split[1] == "err":
        continue
    folder1 = line_split[0][:3]
    folder2 = line_split[0][:8]
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip('\n')
    rel_path = stow.join(dataset_path, "words", folder1, folder2, file_name)
    if not stow.exists(rel_path):
        continue
    dataset.append([rel_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

# Define the model architecture
def train_model(input_dim, output_dim, activation='leaky_relu', dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")
    input_layer = layers.Lambda(lambda x: x / 255)(inputs)
    x1 = residual_block(input_layer, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x2 = residual_block(x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x3 = residual_block(x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x4 = residual_block(x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x5 = residual_block(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x6 = residual_block(x5, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x7 = residual_block(x6, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x8 = residual_block(x7, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x9 = residual_block(x8, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)
    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)
    output = layers.Dense(output_dim + 1, activation='softmax', name="output")(blstm)
    model = Model(inputs=inputs, outputs=output)
    return model

# Define residual block function
def residual_block(x, filters, activation='leaky_relu', skip_conv=False, strides=1, dropout=0.2):
    x_shortcut = x
    if skip_conv:
        x_shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(x_shortcut)
        x_shortcut = layers.BatchNormalization()(x_shortcut)
    x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, x_shortcut])
    x = layers.Activation(activation)(x)
    return x

# Build and compile the model
configs = {
    'height': 128,
    'width': 32,
    'vocab': ''.join(vocab),
    'max_text_length': max_len,
    'learning_rate': 0.001,
    'train_epochs': 50,
    'train_workers': 4
}
model = train_model(input_dim=(configs['height'], configs['width'], 3), output_dim=len(configs['vocab']))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=configs['learning_rate']), 
              loss=tf.keras.losses.CategoricalCrossentropy(), 
              metrics=['accuracy'])
model.summary()

# Define the ImageToWordModel for inference
class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image):
        image = cv2.resize(image, (configs['width'], configs['height']))
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

# Streamlit interface
st.title("Handwriting Recognition")
st.write("Draw a word and click 'Predict' to see the prediction.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=128,
    width=32 * configs['max_text_length'],
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
    img = img.resize((configs['width'] * configs['max_text_length'], configs['height']))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    if st.button("Predict"):
        model_inference = ImageToWordModel(model_path="handwriting_recognition_model.onnx", char_list=configs['vocab'])
        prediction = model_inference.predict(img)
        st.write(f"Prediction: {prediction}")

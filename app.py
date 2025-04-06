import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

# Constants
IMG_LEN = 224
IMG_SIZE = (IMG_LEN, IMG_LEN)
MODEL_PATH = "plant_disease_model.h5"
CLASS_INDEX_PATH = "class_indices.json"
UPLOAD_FOLDER = "static/uploads"

# Load Model
model = load_model(MODEL_PATH)

# Load Class Labels
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)
class_labels = list(class_indices.keys())

# Create Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Grad-CAM for Explainability
def grad_cam(model, img_array, layer_name='block_16_project'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = np.sum(guided_grads * conv_outputs, axis=-1)[0]
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    cam = cv2.resize(cam, IMG_SIZE)
    heatmap = np.uint8(255 * cam)
    return heatmap

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process Image
            img = Image.open(filepath).resize(IMG_SIZE).convert('RGB')
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = float(np.max(prediction))

            return render_template('index.html', filename=filename, prediction=predicted_class, confidence=confidence)
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
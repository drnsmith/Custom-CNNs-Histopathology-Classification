from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# Paths
MODEL_PATH = "models/histopathology_model.h5"

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Pre-process image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        image = Image.open(file)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        result = {'Benign': float(predictions[0][0]), 'Malignant': float(predictions[0][1])}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)

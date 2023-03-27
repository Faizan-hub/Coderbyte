import tensorflow as tf
import numpy as np
from flask import Flask, jsonify, request

# Load the saved model
model = tf.keras.models.load_model('/models/saved_model.pb')

# Initialize Flask application
app = Flask(__name__)

# Define a predict function
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.json['image']
    
    # Preprocess the image data
    processed_data = preprocess_image(image_data)
    
    # Use the model to make a prediction
    prediction = model.predict(processed_data)
    
    # Convert the prediction to a JSON object
    response = {'prediction': int(np.argmax(prediction))}
    
    return jsonify(response)

# Define a function to preprocess the image data
def preprocess_image(image_data):
    # Reshape the image to 28x28 pixels and normalize the pixel values
    processed_data = (np.array(image_data).reshape((1, 28, 28, 1)) / 255.0).astype(np.float32)
    
    return processed_data

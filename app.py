import tensorflow as tf
import numpy as np
import os
import torch
import base64
from io import BytesIO
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    model = StableDiffusionPipeline.from_pretrained("/models/saved_model.pb").to("cuda")
def inference(model_inputs:dict) -> dict:
    global model
    # Get the image array from the request
    image_path = model_inputs.get('image_path', None)
    image = Image.open(image_path)
    image = np.array(image)
    # Preprocess the image array
    image_array = image_array.astype('float32') / 255
    image_array = np.expand_dims(image_array, axis=0)

    # Make the prediction using the MNIST model
    prediction = model.predict(image_array).tolist()[0]

    # Return the prediction as a JSON object
    return {'prediction': prediction}

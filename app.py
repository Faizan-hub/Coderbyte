import os

from PIL import Image
import cv2
import numpy as np



# # Init is ran on server startup
# # Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = tf.saved_model.load("saved_model.pb")
def preprocess_image(image):
    # Convert to RGB format if needed
    if image.shape[-1] == 3 and image.dtype == np.uint8:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to 224x224 with bilinear interpolation
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Divide by 255
    image = image / 255.0

    # Normalize using mean and standard deviation values per channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std

    return image

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    
    global model

    # Parse out your arguments
    prompt = model_inputs.get('image_name', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    image = Image.open(prompt)
    image_array = preprocess_image(image)

    # Make a prediction using the model
    prediction = model.predict(image_array)[0]
    digit = np.argmax(prediction)


    # Return the results as a dictionary
  
    return {'image_base64': str(digit)}

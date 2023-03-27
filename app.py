import os
import torch
import tensorflow as tf



# # Init is ran on server startup
# # Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    loaded_model = tf.saved_model.load("saved_model.pb)
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('image_name', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    

    # Return the results as a dictionary
    return {'image_base64': prompt}

import json
from flask import Flask, request


app = Flask(__name__)
# model = tf.keras.models.load_model('saved_model.pb')
@app.route('/predict', methods=['POST'])
def inference():
    # Get the image file from the request
#     file = request.files['image_name']

#     # Preprocess the image
#     image = Image.open(file)
#     image_array = preprocess_image(image)

#     # Make a prediction using the model
#     prediction = model.predict(image_array)[0]
#     digit = np.argmax(prediction)

    # Return the prediction as a JSON object
    return {'prediction': "yess"}

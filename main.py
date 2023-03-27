import tensorflow as tf

model_path = "saved_model.pb"

# Load the SavedModel
model = tf.keras.models.load_model(model_path)

# Define a function that takes an image file path as input and returns the predicted class
def predict(image_path):
    # Load the image
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    # Convert the image to a numpy array and normalize its values
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    # Reshape the image to match the input shape of the model
    image = image.reshape(1, 28, 28, 1)
    # Use the model to make a prediction
    prediction = model.predict(image)
    # Get the index of the predicted class
    predicted_class = tf.argmax(prediction[0]).numpy()
    return predicted_class

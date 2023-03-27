FROM tensorflow/serving:latest

# Copy the SavedModel to the model directory
COPY saved_model.pb /models/saved_model

# Expose ports for the gRPC and REST endpoints
EXPOSE 8500
EXPOSE 8501

# Start TensorFlow Serving
CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_name=saved_model.pb", "--model_base_path=/models/saved_model.pb"]
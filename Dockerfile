FROM tensorflow/serving:latest
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install torch
# Copy the SavedModel to the model directory
COPY saved_model.pb saved_model.pb

# Expose ports for the gRPC and REST endpoints
EXPOSE 8500
EXPOSE 8501

# Start TensorFlow Serving
CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8501", "--model_name=saved_model.pb", "--model_base_path=saved_model.pb"]

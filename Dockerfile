FROM tensorflow/serving:latest
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install torch
RUN pip3 install tensorflow

WORKDIR /
# Copy the SavedModel and img
COPY saved_model.pb saved_model.pb
COPY img1.jpg img1.jpg
ADD app.py .
ADD server.py .

# Expose ports for the gRPC and REST endpoints
EXPOSE 8000

CMD python3 -u server.py

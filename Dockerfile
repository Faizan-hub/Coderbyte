FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
RUN apt-get update && apt-get install -y python3-pip
RUN apt-get update && apt-get install -y git
RUN pip3 install torch
RUN pip install tensorflow==2.11.0 --ignore-installed
WORKDIR /
ADD requirements.txt .
RUN pip install -r requirements.txt
ADD convert_to_onnx.py .
ADD test_onnx.py .
ADD n01440764_tench.jpeg .
ADD n01667114_mud_turtle.jpeg .
ADD app.py .
ADD server.py .

# Expose ports for the gRPC and REST endpoints
EXPOSE 8000

CMD python3 -u server.py

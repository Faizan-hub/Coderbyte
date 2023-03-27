import onnxruntime
import numpy as np
from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn

from torch import Tensor
import numpy as np

from torchvision import transforms

from PIL import 

# Load the ONNX model
onnx_model = onnxruntime.InferenceSession('model.onnx')
def preprocess_numpy(self, img):
    resize = transforms.Resize((224, 224))   #must same as here
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    return img
# Define input shape and data type
input_name = onnx_model.get_inputs()[0].name
input_shape = onnx_model.get_inputs()[0].shape
input_dtype = onnx_model.get_inputs()[0].type


img = Image.open("n01667114_mud_turtle.JPEG")
inp = mtailor.preprocess_numpy(img).unsqueeze(0) 

# Run inference
output = onnx_model.run(None, {input_name: input_data})

# Generate random input data
img2 = Image.open("n01440764_tench.jpeg")
inp2 = mtailor.preprocess_numpy(im2).unsqueeze(0) 

# Run inference
output2 = onnx_model.run(None, {input_name: input_data2})
# Print output
print(output)
print(output2)

import torch
import cv2
import numpy as np
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def run_yolo(file_storage):
    image_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    results = model(image)
    results.render()
    output = results.imgs[0]  # NumPy array (BGR)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output_rgb)

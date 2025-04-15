from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load YOLOv8 model (n = nano, s = small, m = medium, l = large, x = extra large)
model = YOLO('yolov8n.pt')  # You can change to 'yolov8s.pt' or others

def run_yolo(file_storage):
    # Convert uploaded file to numpy array (OpenCV image)
    image_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Run inference
    results = model(image)

    # Plot bounding boxes on the image (returns a numpy array in RGB format)
    rendered_img = results[0].plot()  # Already in RGB format

    # Convert numpy array to PIL Image and return
    return Image.fromarray(rendered_img)

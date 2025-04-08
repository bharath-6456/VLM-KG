from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8s.pt / yolov8m.pt for bigger ones

# Path to your image
image_path = "images/image.png"

# Run inference
results = model(image_path)

# Visualize the result
results[0].show()  # Opens a window with detections

# Get detected labels
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls_id]
        print(f"Detected: {label} ({conf:.2f})")

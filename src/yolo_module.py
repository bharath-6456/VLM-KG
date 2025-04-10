from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # You can use yolov8s or better if needed

def detect_objects(image_path):
    results = model(image_path)
    names = results[0].names
    detected_classes = results[0].boxes.cls.cpu().numpy()
    return list(set([names[int(cls)] for cls in detected_classes]))


from ultralytics import YOLO

model = YOLO("yolov8l.pt")

results = model.train(
    data="data.yaml",
    epochs=200,
    imgsz=300,
    batch=16,
    name="recycling_detection_v1",
    device=0,
    verbose=True
)

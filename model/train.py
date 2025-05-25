import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("yolov8l.pt")

results = model.train(
    data="model/data.yaml",
    epochs=100,
    imgsz=300,
    batch=16,
    name="model/recycling_detection_v1",
    device=0,
    verbose=True
)



df = pd.read_csv("runs/detect/waste_model_v1/results.csv")

plt.plot(df['      metrics/mAP50(B)'])
plt.title("mAP@0.5 vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5")
plt.show()
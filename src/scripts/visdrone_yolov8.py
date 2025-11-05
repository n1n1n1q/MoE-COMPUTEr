"""
Training script for baseline YOLOv8 on VisDrone dataset.

This script trains a standard YOLOv8n model with randomly initialized weights
on the VisDrone dataset for comparison with the MoE version.
"""

from ultralytics import YOLO
from src.models.moe_yolo import init_weights_random
from src.utils.random import set_seed

RANDOM_SEED = 148
set_seed(RANDOM_SEED, False)

model = YOLO("yolov8n.pt")
model.model.apply(init_weights_random)

data_path = "datasets/VisDrone.yaml"
model.train(
    data=data_path,
    epochs=100,
    imgsz=640,
    batch=16,
    workers=32,
    project="runs/train",
    name="moe_yolo_voc",
)
model.save("yolov8_random.pt")
results = model.val(data=data_path, imgsz=640, batch=16, workers=8)
print(results)

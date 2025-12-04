"""
Training script for YOLOv8 with Mixture of Experts (MoE).

This script trains a YOLOv8 model enhanced with MoE layers on the VisDrone dataset.
The model uses random weight initialization and trains for 100 epochs.
"""

from src.models.moe_yolo import init_weights_random
from src.models.moe_yolo import yolov8_moe
from src.utils.random import set_seed

RANDOM_STATE = 148

set_seed(RANDOM_STATE, False)
model = yolov8_moe()
model.model.apply(init_weights_random)

data_path = "datasets/VisDrone.yaml"
model.train(
    data=data_path,
    epochs=20,
    imgsz=640,
    batch=4,
    workers=2,
    project="runs/train",
    name="moe_yolo_voc",
    verbose=True,
)
model.save("yolov8_moe_voc_rand.pt")
results = model.val(data=data_path, imgsz=640, batch=16, workers=8)
print(results)

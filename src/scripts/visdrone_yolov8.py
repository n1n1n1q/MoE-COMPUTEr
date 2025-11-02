import torch
from ultralytics import YOLO
from src.models.moe_yolo import init_weights_random

random_state = 148
torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
    name="moe_yolo_voc"
)
model.save("yolov8_random.pt")
results = model.val(
    data=data_path,
    imgsz=640,
    batch=16,
    workers=8
)
print(results)
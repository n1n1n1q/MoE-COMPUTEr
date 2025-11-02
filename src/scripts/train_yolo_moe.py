from src.models.moe_yolo import yolov8_moe

model = yolov8_moe(random_state=148)

data_path = "datasets/VisDrone.yaml"
model.train(
    data=data_path,
    epochs=100,
    imgsz=640,
    batch=16,
    workers=16,
    project="runs/train",
    name="moe_yolo_voc"
)
model.save("yolov8_moe_voc_rand.pt")
results = model.val(
    data=data_path,
    imgsz=640,
    batch=16,
    workers=8
)
print(results)
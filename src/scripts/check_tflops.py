from ultralytics import YOLO
from calflops import calculate_flops

model = YOLO("yolov8n.pt")
batch_size = 16
input_shape = (batch_size, 3, 640, 640)
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4,
                                      print_detailed=True)
# print("YOLOv8 FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

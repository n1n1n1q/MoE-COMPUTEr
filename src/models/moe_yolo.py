"""

"""
from src.nn.moe_c2f import C2fSparseMoE
from ultralytics import YOLO

def yolov8_moe():
    model = YOLO("yolov8n.pt")
    
    old_c2f = model.model.model[6]
    in_channels = old_c2f.cv1.conv.in_channels
    out_channels = old_c2f.cv2.conv.out_channels
    model.model.model[6] = C2fSparseMoE(in_channels, out_channels, num_experts=4)
    
    old_c2f_neck = model.model.model[12]
    in_channels = old_c2f_neck.cv1.conv.in_channels
    out_channels = old_c2f_neck.cv2.conv.out_channels
    model.model.model[12] = C2fSparseMoE(in_channels, out_channels, num_experts=4)

    return model

if __name__ == "__main__":
    yolov8_moe()
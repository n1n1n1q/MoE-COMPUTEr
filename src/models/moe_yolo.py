"""
Create a YOLOv8 with MoE blocks and random-initialized weights (overrides pretrained file).
"""
import torch
import torch.nn as nn
from src.nn.moe_c2f import C2fSparseMoE
from ultralytics import YOLO


def init_weights_random(m):
    # convs
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # linears
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # batch/group norms
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

def yolov8_moe(n_experts_l1=4, k_l1=2, n_experts_l2=4, k_l2=2, random_state=None):
    # instantiate YOLO architecture (file will be loaded but we will reinit weights below)
    model = YOLO("yolov8n.pt")

    old_c2f = model.model.model[6]
    in_channels = old_c2f.cv1.conv.in_channels
    out_channels = old_c2f.cv2.conv.out_channels
    new_c2f = C2fSparseMoE(in_channels, out_channels, num_experts=n_experts_l1, k=k_l1)

    new_c2f.i = old_c2f.i
    new_c2f.f = old_c2f.f
    new_c2f.type = old_c2f.type
    model.model.model[6] = new_c2f

    old_c2f_neck = model.model.model[12]
    in_channels = old_c2f_neck.cv1.conv.in_channels
    out_channels = old_c2f_neck.cv2.conv.out_channels
    new_c2f_neck = C2fSparseMoE(in_channels, out_channels, num_experts=n_experts_l2, k=k_l2)

    new_c2f_neck.i = old_c2f_neck.i
    new_c2f_neck.f = old_c2f_neck.f
    new_c2f_neck.type = old_c2f_neck.type
    model.model.model[12] = new_c2f_neck


    if random_state is not None:
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model.model.apply(init_weights_random)

    return model

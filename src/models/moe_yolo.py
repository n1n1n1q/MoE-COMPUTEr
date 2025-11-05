"""
Create a YOLOv8 with MoE blocks and random-initialized weights (overrides pretrained file).
"""

import torch
import torch.nn as nn
from src.nn.moe_c2f import C2fSparseMoE
from ultralytics import YOLO


def init_weights_random(m):
    """
    Initialize neural network module weights randomly.

    Applies appropriate initialization strategies based on the module type:
    - Conv2d/ConvTranspose2d: Kaiming normal initialization
    - Linear: Xavier uniform initialization
    - BatchNorm/GroupNorm: Ones for weights, zeros for biases

    Args:
        m (nn.Module): PyTorch module to initialize.
    """
    # convs
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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


def yolov8_moe(n_experts_l1=4, k_l1=2, n_experts_l2=4, k_l2=2):
    """
    Create a YOLOv8 model with Mixture of Experts (MoE) layers.

    Replaces two C2f layers in the YOLOv8 architecture with C2fSparseMoE layers
    and reinitializes all weights randomly. The MoE layers are inserted at layers
    6 (backbone) and 12 (neck) of the YOLOv8n architecture.

    Args:
        n_experts_l1 (int, optional): Number of experts for first MoE layer (layer 6). Defaults to 4.
        k_l1 (int, optional): Number of active experts per input for first MoE layer. Defaults to 2.
        n_experts_l2 (int, optional): Number of experts for second MoE layer (layer 12). Defaults to 4.
        k_l2 (int, optional): Number of active experts per input for second MoE layer. Defaults to 2.

    Returns:
        YOLO: YOLOv8 model with MoE layers and randomly initialized weights.
    """
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
    new_c2f_neck = C2fSparseMoE(
        in_channels, out_channels, num_experts=n_experts_l2, k=k_l2
    )

    new_c2f_neck.i = old_c2f_neck.i
    new_c2f_neck.f = old_c2f_neck.f
    new_c2f_neck.type = old_c2f_neck.type
    model.model.model[12] = new_c2f_neck

    model.model.apply(init_weights_random)

    return model

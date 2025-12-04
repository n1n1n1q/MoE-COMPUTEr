"""
Create a YOLOv8 with MoE blocks and random-initialized weights (overrides pretrained file).
"""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ultralytics.models import yolo
from src.utils.logger import MoELogger
from src.nn.moe_c2f import C2fSparseMoE

from torch.profiler import profile, ProfilerActivity

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
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


def on_train_epoch_start(trainer):
    model = trainer.model
    imgsz = trainer.args.imgsz
    device = next(model.parameters()).device
    x = torch.randn(1, 3, imgsz, imgsz).to(device)

    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as p:
        model(x)

    ka = p.key_averages()
    
    total_flops = sum([op.flops for op in ka])

    writer = SummaryWriter(trainer.save_dir)
    writer.add_scalar("GFLOPS", total_flops / 1e9)


class MoEDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg="yolo11n.yaml",
        ch=3,
        nc=None,
        verbose=True,
        n_experts_l1=4,
        k_l1=2,
        n_experts_l2=4,
        k_l2=2,
    ):
        super().__init__(cfg, ch, nc, verbose)

        old_c2f = self.model[6]
        in_channels = old_c2f.cv1.conv.in_channels
        out_channels = old_c2f.cv2.conv.out_channels
        self.new_c2f = C2fSparseMoE(
            in_channels,
            out_channels,
            num_experts=n_experts_l1,
            top_k=k_l1,
            name="MoE C2F",
        )

        self.new_c2f.i = old_c2f.i
        self.new_c2f.f = old_c2f.f
        self.new_c2f.type = old_c2f.type

        self.model[6] = self.new_c2f

        old_c2f_neck = self.model[12]
        in_channels = old_c2f_neck.cv1.conv.in_channels
        out_channels = old_c2f_neck.cv2.conv.out_channels
        self.new_c2f_neck = C2fSparseMoE(
            in_channels,
            out_channels,
            num_experts=n_experts_l2,
            top_k=k_l2,
            name="moe_c2f_neck"
        )

        self.new_c2f_neck.i = old_c2f_neck.i
        self.new_c2f_neck.f = old_c2f_neck.f
        self.new_c2f_neck.type = old_c2f_neck.type
        self.model[12] = self.new_c2f_neck
        

class MoEDetectionTrainer(DetectionTrainer):
    def get_model(
        self, cfg: str | None = None, weights: str | None = None, verbose: bool = True
    ):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = MoEDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose,
            n_experts_l1=6,
        )

        if weights:
            model.load(weights)

        self.add_callback('on_train_batch_end', MoELogger(modules_to_monitor={
             "C2F Bottlenech [0]":  model.new_c2f.m[0],
             "Neck C2F Bottlenech [0]":  model.new_c2f_neck.m[0],
        }))

        self.add_callback('on_train_epoch_start', on_train_epoch_start)

        return model


class MoEYOLO(YOLO):
    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": MoEDetectionModel,
                "trainer": MoEDetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }


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
    model = MoEYOLO("yolov8n.pt")

    return model

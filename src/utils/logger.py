import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ultralytics.models.yolo.detect.train import DetectionTrainer

class MoELogger():

    def __init__(self, modules_to_monitor: list[nn.Module]):
        self.modules_to_monitor = modules_to_monitor
        self.step = 0

    def __call__(self, trainer: DetectionTrainer):

        writer = SummaryWriter(trainer.save_dir)

        for name, module in self.modules_to_monitor.items():
            writer.add_scalars(f"{name} / Batches per expers",
                                { str(i): batches for i, batches in enumerate(module.get_batcher_per_expert()) },
                                global_step=self.step)

            writer.add_scalar(
                f"{name} / Dispersion",
                torch.std(module.get_batcher_per_expert().float()),
                global_step=self.step
            )

        self.step += 1



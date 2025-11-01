import torch
import torch.nn as nn
import torch.nn.functional as F
from src.nn.conv import Conv
from src.nn.bottleneck import MoEBottleneck

class C2fSparseMoE(nn.Module):
    
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, num_experts: int = 4, k: int = 2):
        """
        Initialize a Sparse MoE version of the C2f layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            num_experts (int): Number of experts in the MoE layer.
            k (int): Number of experts to use per input.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        
        self.m = nn.ModuleList(
            MoEBottleneck(self.c, self.c, num_experts=num_experts, k=k, shortcut=shortcut, g=g, e=1.0) 
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C2f MoE layer.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
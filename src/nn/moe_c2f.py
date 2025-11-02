"""
C2f layer with Mixture of Experts.

This module implements the C2fSparseMoE layer, which is a modified C2f
architecture that incorporates Mixture of Experts (MoE) bottleneck blocks
for enhanced feature extraction and model capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.nn.conv import Conv
from src.nn.bottleneck import MoEBottleneck


class C2fSparseMoE(nn.Module):
    """
    C2f layer with Sparse Mixture of Experts (MoE).

    This module implements a modified C2f layer that uses MoE bottleneck blocks
    instead of standard bottlenecks. The C2f architecture splits features into
    multiple paths and progressively combines them with bottleneck transformations.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        num_experts: int = 4,
        k: int = 2,
    ):
        """
        Initialize a Sparse MoE version of the C2f layer.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int, optional): Number of MoE bottleneck blocks to stack. Defaults to 1.
            shortcut (bool, optional): Whether to use shortcut connections in bottlenecks. Defaults to False.
            g (int, optional): Number of groups for grouped convolution. Defaults to 1.
            e (float, optional): Channel expansion ratio. Defaults to 0.5.
            num_experts (int, optional): Number of experts in each MoE layer. Defaults to 4.
            k (int, optional): Number of top experts to activate per input. Defaults to 2.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            MoEBottleneck(
                self.c,
                self.c,
                num_experts=num_experts,
                k=k,
                shortcut=shortcut,
                g=g,
                e=1.0,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C2f MoE layer.

        The input is split into two parts via cv1, then progressively processed
        through MoE bottleneck blocks. All intermediate features are concatenated
        and passed through cv2 for the final output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, c1, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, c2, height, width).
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

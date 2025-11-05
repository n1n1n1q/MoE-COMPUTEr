"""
Bottleneck layer implementations.

This module provides standard bottleneck blocks with residual connections
and Mixture of Experts (MoE) bottleneck blocks for efficient and dynamic
neural network architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.conv import Conv


class Bottleneck(nn.Module):
    """
    Standard bottleneck block with residual connection.

    A bottleneck block consists of two convolution layers with an optional shortcut
    connection. The first convolution reduces channels, and the second restores them.
    """

    def __init__(
        self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5
    ):
        """
        Initialize the Bottleneck block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            shortcut (bool, optional): Whether to use residual shortcut connection. Defaults to True.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
            expansion (float, optional): Channel expansion factor for hidden layer. Defaults to 0.5.
        """
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv2 = Conv(
            hidden_channels, out_channels, kernel_size=3, stride=1, groups=groups
        )
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        """
        Forward pass through the bottleneck block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with optional residual connection applied.
        """
        y = self.conv2(self.conv1(x))
        return x + y if self.use_shortcut else y


class Gate(nn.Module):
    """
    Gating network for Mixture of Experts (MoE).
    This module computes the gating scores for selecting top-k experts
    based on the input features.
    """

    def __init__(self, dim, num_experts, top_k, bias=True):
        """
        Initialize the Gate.
        Args:
            dim (int): Dimension of the input features.
            num_experts (int): Number of experts available.
            top_k (int): Number of top experts to select.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        super(Gate, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter(torch.Tensor(dim, num_experts))
        self.bias = nn.Parameter(torch.Tensor(num_experts)) if bias else None

    def forward(self, x):
        """
        Forward pass through the gating network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Weights and indices of selected experts.
        """
        scores = F.linear(x, self.weight, self.bias)
        scores = scores.softmax(dim=-1)
        indices = scores.topk(self.top_k, dim=-1)[1]
        weights = scores.gather(-1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)
        return weights, indices


class MoEBottleneck(nn.Module):
    """
    Mixture of Experts (MoE) Bottleneck block.

    This module implements a sparse MoE layer where multiple expert bottleneck blocks
    are available, and a gating network dynamically selects the top-k experts for each
    input. The outputs from selected experts are weighted and combined.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_experts: int = 4,
        k: int = 2,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        """
        Initialize MoE Bottleneck.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            num_experts (int, optional): Number of expert bottleneck modules. Defaults to 4.
            k (int, optional): Number of top experts to select per input. Defaults to 2.
            shortcut (bool, optional): Whether to use residual shortcut connection. Defaults to True.
            g (int, optional): Number of groups for grouped convolution. Defaults to 1.
            e (float, optional): Channel expansion factor for expert bottlenecks. Defaults to 0.5.
        """
        super().__init__()
        self.dim = c1
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.shortcut = shortcut and c1 == c2

        self.experts = nn.ModuleList(
            [
                Bottleneck(c1, c2, shortcut=False, groups=g, expansion=e)
                for _ in range(num_experts)
            ]
        )

        self.gate = Gate(dim=c1, num_experts=num_experts, top_k=self.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE bottleneck.

        The gating network selects the top-k experts for each input, and their
        outputs are weighted by the gate probabilities and combined.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, c1, height, width).

        Returns:
            torch.Tensor: Output tensor with expert outputs combined and optional residual connection.
        """
        x_flattened = x.view(-1, self.dim)
        weights, indices = self.gate(x_flattened)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.num_experts).tolist()
        for i in range(self.num_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        return y + x if self.shortcut else y
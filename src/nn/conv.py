"""
Convolution layer implementations.

This module provides the Conv class, which combines 2D convolution, batch
normalization, and activation into a single layer, along with utility
functions for automatic padding calculation.
"""

import torch.nn as nn


def autopad(k, p=None, d=1):
    """
    Calculate padding automatically for convolution operations.

    Args:
        k (int or list): Kernel size, can be an integer or a list of integers.
        p (int or list, optional): Padding value. If None, padding is calculated automatically. Defaults to None.
        d (int, optional): Dilation rate. Defaults to 1.

    Returns:
        int or list: Computed padding value(s).
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """
    Standard convolution layer with batch normalization and activation.

    This module combines a 2D convolution, batch normalization, and an optional
    activation function (SiLU) into a single layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        act=True,
    ):
        """
        Initialize the Conv layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int or None, optional): Padding added to input. If None, computed automatically. Defaults to None.
            groups (int, optional): Number of blocked connections from input to output channels. Defaults to 1.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            act (bool, optional): Whether to apply activation function. Defaults to True.
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the convolution layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after convolution, batch norm, and activation.
        """
        return self.act(self.bn(self.conv(x)))

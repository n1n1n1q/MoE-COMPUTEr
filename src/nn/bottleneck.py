import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.conv import Conv

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1, groups=groups)
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_shortcut else y
    
class MoEBottleneck(nn.Module):    
    def __init__(self, c1: int, c2: int, num_experts: int = 4, k: int = 2, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize MoE Bottleneck.
        
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            num_experts (int): Number of expert bottlenecks.
            k (int): Top-k experts to use per input.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.shortcut = shortcut and c1 == c2

        self.experts = nn.ModuleList([
            Bottleneck(c1, c2, shortcut=False, groups=g, expansion=e) 
            for _ in range(num_experts)
        ])
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, num_experts),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        gate_logits = self.gate(x)

        top_k_logits, top_k_indices = torch.topk(gate_logits, self.k, dim=1)
        top_k_gates = F.softmax(top_k_logits, dim=1)
        output = torch.zeros_like(x)

        for i in range(batch_size):
            for j in range(self.k):
                expert_idx = top_k_indices[i, j]
                expert_weight = top_k_gates[i, j]
                expert_output = self.experts[expert_idx](x[i:i+1])
                output[i:i+1] += expert_weight * expert_output

        if self.shortcut:
            output = output + x
            
        return output
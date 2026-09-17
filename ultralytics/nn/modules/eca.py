"""Efficient channel attention used only at the selected UW neck fusion sites."""
import torch
from torch import nn


class ECA(nn.Module):
    """Preserve BCHW shape and apply a local channel attention kernel."""

    def __init__(self, c1: int, k_size: int = 3):
        super().__init__()
        if c1 <= 0 or k_size <= 0 or k_size % 2 == 0:
            raise ValueError('ECA requires positive channels and a positive odd kernel size')
        self.c1 = c1
        self.k_size = k_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

import torch
from torch import nn


class ChannelGate(nn.Module):
    def __init__(self, in_planes: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(1, in_planes // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x)) * x


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, no_spatial: bool = False):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction_ratio)
        self.no_spatial = no_spatial
        self.spatial_gate = None if no_spatial else SpatialGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_gate(x)
        if not self.no_spatial:
            x = self.spatial_gate(x)
        return x

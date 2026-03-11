import math
from typing import List, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .attention import CBAM


class InjectivePad(nn.Module):
    def __init__(self, pad_size: int):
        super().__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : x.size(1) - self.pad_size, :, :]


class Squeeze(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        return x.contiguous().view(b, c * 4, h // 2, w // 2)

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        return x.contiguous().view(b, c // 4, h * 2, w * 2)


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    n, c = feat.size()[:2]
    feat_var = feat.view(n, c, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


class AdaIN(nn.Module):
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        assert content.size()[:2] == style.size()[:2]
        size = content.size()
        style_mean, style_std = calc_mean_std(style)
        content_mean, content_std = calc_mean_std(content)
        normalized = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized * style_std.expand(size) + style_mean.expand(size)


class PACCoupling(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size, padding=padding),
        )
        self.att = CBAM(channels)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        att_out = self.att(x)
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        return x + alpha * conv_out + (1.0 - alpha) * att_out


class InfoMinRevBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.channels = channels
        self.coupling = PACCoupling(channels // 2, hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        y1 = x1
        y2 = self.coupling(x1) + x2 - x1
        return torch.cat([y1, y2], dim=1)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=1)
        x1 = y1
        x2 = y2 - (self.coupling(y1) - y1)
        return torch.cat([x1, x2], dim=1)


class InfoMinRevNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        pad_channels: int = 5,
        num_blocks: int = 20,
        num_squeeze: int = 2,
        hidden_channels: int = 512,
        blocks_per_stage: Optional[List[int]] = None,
    ):
        super().__init__()
        self.pad = InjectivePad(pad_channels)
        self.adain = AdaIN()

        modules: List[nn.Module] = []
        levels = num_squeeze + 1
        if blocks_per_stage is None:
            blocks_per_level = [num_blocks // levels for _ in range(levels)]
            for i in range(num_blocks % levels):
                blocks_per_level[i] += 1
        else:
            if len(blocks_per_stage) != levels or sum(blocks_per_stage) != num_blocks:
                raise ValueError(\"blocks_per_stage must match num_squeeze+1 and sum to num_blocks\")\n+            blocks_per_level = blocks_per_stage

        channels = in_channels + pad_channels
        for level_idx, n_blocks in enumerate(blocks_per_level):
            for _ in range(n_blocks):
                modules.append(InfoMinRevBlock(channels, hidden_channels))
            if level_idx < num_squeeze:
                modules.append(Squeeze())
                channels *= 4

        self.modules_fwd = nn.ModuleList(modules)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        z = self.pad(x)
        mi_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for module in self.modules_fwd:
            if isinstance(module, InfoMinRevBlock):
                z_in = z
                z = module(z)
                mi_pairs.append((z_in, z))
            else:
                z = module(z)
        return z, mi_pairs

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        for module in reversed(self.modules_fwd):
            if isinstance(module, InfoMinRevBlock):
                z = module.reverse(z)
            else:
                z = module.reverse(z)
        z = self.pad.inverse(z)
        return z

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        z_c, mi_pairs_c = self.encode(content)
        z_s, _ = self.encode(style)
        z_cs = self.adain(z_c, z_s)
        out = self.decode(z_cs)
        return out, mi_pairs_c

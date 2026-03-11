from typing import Iterable, List, Tuple

import math
import torch
from torch import nn
from torch.nn import functional as F


def _prepare_features(x: torch.Tensor, pool: int, max_dim: int) -> torch.Tensor:
    if pool > 1:
        x = F.avg_pool2d(x, kernel_size=pool, stride=pool)
    x = x.flatten(1)
    if x.shape[1] > max_dim:
        x = x[:, :max_dim]
    return x


def _kde_log_density(x: torch.Tensor, bandwidth: float) -> torch.Tensor:
    # x: [N, D]
    n, d = x.shape
    if n == 1:
        return torch.zeros(1, device=x.device)
    dist2 = torch.cdist(x, x, p=2).pow(2)
    log_kernel = -0.5 * dist2 / (bandwidth ** 2)
    log_sum = torch.logsumexp(log_kernel, dim=1)
    log_norm = -math.log(n) - d * math.log(bandwidth) - 0.5 * d * math.log(2 * math.pi)
    return log_norm + log_sum


def nmi_kde(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    log_px = _kde_log_density(x, bandwidth)
    log_py = _kde_log_density(y, bandwidth)
    xy = torch.cat([x, y], dim=1)
    log_pxy = _kde_log_density(xy, bandwidth)
    h_x = -log_px.mean()
    h_y = -log_py.mean()
    mi = (log_pxy - log_px - log_py).mean()
    nmi = 2.0 * mi / (h_x + h_y + 1e-8)
    return torch.clamp(nmi, min=0.0, max=1.0)


class NMIKDELoss(nn.Module):
    def __init__(self, bandwidth: float = 1.0, pool: int = 4, max_dim: int = 1024):
        super().__init__()
        self.bandwidth = bandwidth
        self.pool = pool
        self.max_dim = max_dim
        self.register_buffer("zero", torch.tensor(0.0))

    def forward(self, pairs: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for x, y in pairs:
            x = _prepare_features(x, self.pool, self.max_dim)
            y = _prepare_features(y, self.pool, self.max_dim)
            losses.append(nmi_kde(x, y, self.bandwidth))
        if not losses:
            return self.zero
        return torch.stack(losses).mean()

from typing import Iterable, List, Optional

import torch
from torch import nn


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class MLPProjector(nn.Module):
    def __init__(self, dims: Iterable[int]):
        super().__init__()
        dims = list(dims)
        layers: List[nn.Module] = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BarlowTwinsLoss(nn.Module):
    def __init__(
        self,
        lambda_offdiag: float = 0.0051,
        eps: float = 1e-6,
        projector: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.lambda_offdiag = lambda_offdiag
        self.eps = eps
        self.projector = projector

    def forward(self, zc: torch.Tensor, zcs: torch.Tensor) -> torch.Tensor:
        # zc, zcs: [B, D]
        if self.projector is not None:
            zc = self.projector(zc)
            zcs = self.projector(zcs)

        # Eq. (8) in paper: normalized cross-correlation along batch dimension
        zc_norm = torch.sqrt((zc ** 2).sum(dim=0) + self.eps)
        zcs_norm = torch.sqrt((zcs ** 2).sum(dim=0) + self.eps)
        c = (zc.T @ zcs) / (zc_norm[:, None] * zcs_norm[None, :])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambda_offdiag * off

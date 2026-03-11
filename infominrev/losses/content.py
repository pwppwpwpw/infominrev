import torch
from torch import nn


def mean_variance_norm(feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    n, c = feat.size()[:2]
    feat_var = feat.view(n, c, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return (feat - feat_mean) / feat_std


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, stylized: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        return self.mse(mean_variance_norm(stylized), mean_variance_norm(content))

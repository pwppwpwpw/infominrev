from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    n, c = feat.size()[:2]
    feat_var = feat.view(n, c, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(n, c)
    feat_mean = feat.view(n, c, -1).mean(dim=2)
    return feat_mean, feat_std


def _normalize_distribution(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = torch.clamp(x, min=0.0)
    return x / (x.sum(dim=1, keepdim=True) + eps)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p, q: [B, C] non-negative distributions
    p = _normalize_distribution(p, eps)
    q = _normalize_distribution(q, eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.add(eps).log() - m.add(eps).log())).sum(dim=1)
    kl_qm = (q * (q.add(eps).log() - m.add(eps).log())).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


class JSDStyleLoss(nn.Module):
    def __init__(self, warmup_steps: int = 2000, eps: float = 1e-8):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(
        self,
        style_feats: Dict[str, torch.Tensor],
        stylized_feats: Dict[str, torch.Tensor],
        step: int,
    ) -> torch.Tensor:
        loss = 0.0
        layers = style_feats.keys()
        for k in layers:
            s = style_feats[k]
            t = stylized_feats[k]
            s_mean, s_std = calc_mean_std(s, eps=self.eps)
            t_mean, t_std = calc_mean_std(t, eps=self.eps)
            if step < self.warmup_steps:
                loss = loss + self.mse(s_mean, t_mean) + self.mse(s_std, t_std)
            else:
                loss = loss + js_divergence(s_mean, t_mean, eps=self.eps).mean()
                loss = loss + js_divergence(s_std, t_std, eps=self.eps).mean()
        return loss / max(1, len(style_feats))

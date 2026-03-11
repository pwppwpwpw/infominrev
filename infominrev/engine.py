from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

from .config import get, save_config
from .data import build_loaders
from .losses.barlow import BarlowTwinsLoss, MLPProjector
from .losses.content import ContentLoss
from .losses.jsd import JSDStyleLoss
from .losses.mi import NMIKDELoss
from .models.infominrev import InfoMinRevNet
from .models.vgg import VGGEncoder
from .utils import ensure_dir, load_checkpoint, save_checkpoint, set_seed


@dataclass
class TrainState:
    step: int
    max_iter: int


class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(get(config, "device", default="cuda") if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        set_seed(get(config, "seed", default=42))

        self.output_dir = ensure_dir(get(config, "train", "output_dir", default="./runs/infominrev"))
        save_config(config, str(self.output_dir / "config.json"))

        self.content_loader, self.style_loader = build_loaders(
            get(config, "data", "content_dir"),
            get(config, "data", "style_dir"),
            get(config, "data", "image_size", default=512),
            get(config, "data", "crop_size", default=256),
            get(config, "data", "batch_size", default=4),
            get(config, "data", "num_workers", default=8),
            get(config, "data", "pin_memory", default=True),
        )
        self.content_iter = iter(self.content_loader)
        self.style_iter = iter(self.style_loader)

        self.model = InfoMinRevNet(
            in_channels=get(config, "model", "in_channels", default=3),
            pad_channels=get(config, "model", "pad_channels", default=5),
            num_blocks=get(config, "model", "num_blocks", default=20),
            num_squeeze=get(config, "model", "num_squeeze", default=2),
            hidden_channels=get(config, "model", "hidden_channels", default=512),
            blocks_per_stage=get(config, "model", "blocks_per_stage", default=None),
        ).to(self.device)

        vgg_layers = get(config, "vgg", "layers", default=["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu4_2"])
        barlow_layer = get(config, "loss", "barlow", "layer", default="relu4_1")
        if barlow_layer not in vgg_layers:
            vgg_layers = list(dict.fromkeys(vgg_layers + [barlow_layer]))
        self.vgg = VGGEncoder(vgg_layers).to(self.device)
        vgg_weights = get(config, "vgg", "weights", default=None)
        if vgg_weights:
            self.vgg.load_state_dict(torch.load(vgg_weights, map_location="cpu"))
        self.vgg.eval()

        projector_dims = get(config, "loss", "barlow", "projector_dims", default=None)
        projector = MLPProjector(projector_dims).to(self.device) if projector_dims else None

        self.loss_content = ContentLoss().to(self.device)
        self.loss_style = JSDStyleLoss(
            warmup_steps=get(config, "loss", "jsd", "warmup_steps", default=2000),
            eps=get(config, "loss", "jsd", "eps", default=1e-8),
        ).to(self.device)
        self.loss_barlow = BarlowTwinsLoss(
            lambda_offdiag=get(config, "loss", "barlow", "lambda_offdiag", default=0.0051),
            eps=get(config, "loss", "barlow", "eps", default=1e-6),
            projector=projector,
        ).to(self.device)
        self.loss_mi = NMIKDELoss(
            bandwidth=get(config, "loss", "mi", "bandwidth", default=1.0),
            pool=get(config, "loss", "mi", "pool", default=4),
            max_dim=get(config, "loss", "mi", "max_dim", default=1024),
        ).to(self.device)

        self.w_content = get(config, "loss", "content_weight", default=1.0)
        self.w_style = get(config, "loss", "style_weight", default=10.0)
        self.w_barlow = get(config, "loss", "barlow_weight", default=0.02)
        self.w_mi = get(config, "loss", "mi_weight", default=0.1)

        params = list(self.model.parameters()) + list(self.loss_barlow.parameters())
        self.optim = torch.optim.Adam(
            params,
            lr=get(config, "optim", "lr", default=1e-4),
            betas=tuple(get(config, "optim", "betas", default=[0.9, 0.999])),
            weight_decay=get(config, "optim", "weight_decay", default=0.0),
        )

        resume = get(config, "train", "resume", default=None)
        self.state = TrainState(step=0, max_iter=get(config, "train", "max_iter", default=160000))
        if resume:
            ckpt = load_checkpoint(resume)
            self.model.load_state_dict(ckpt["model"])
            self.optim.load_state_dict(ckpt["optim"])
            self.state.step = ckpt.get("iter", 0)

        self.log_every = get(config, "train", "log_every", default=50)
        self.save_every = get(config, "train", "save_every", default=10000)
        self.sample_every = get(config, "train", "sample_every", default=500)

    def _next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        content = next(self.content_iter).to(self.device)
        style = next(self.style_iter).to(self.device)
        return content, style

    def _compute_losses(self, content: torch.Tensor, style: torch.Tensor, step: int):
        stylized, mi_pairs = self.model(content, style)

        feats_content = self.vgg(content)
        feats_style = self.vgg(style)
        feats_stylized = self.vgg(stylized)

        loss_c = self.loss_content(feats_stylized["relu4_2"], feats_content["relu4_2"])
        loss_s = self.loss_style(feats_style, feats_stylized, step)

        barlow_layer = get(self.config, "loss", "barlow", "layer", default="relu4_1")
        zc = torch.mean(feats_content[barlow_layer], dim=(2, 3))
        zcs = torch.mean(feats_stylized[barlow_layer], dim=(2, 3))
        loss_b = self.loss_barlow(zc, zcs)

        loss_mi = self.loss_mi(mi_pairs)

        total = self.w_content * loss_c + self.w_style * loss_s + self.w_barlow * loss_b + self.w_mi * loss_mi
        return stylized, total, loss_c, loss_s, loss_b, loss_mi

    def _save_sample(self, content: torch.Tensor, style: torch.Tensor, stylized: torch.Tensor, step: int) -> None:
        sample_dir = ensure_dir(str(self.output_dir / "samples"))
        output = torch.cat([style, content, stylized], dim=3)
        save_image(output, str(sample_dir / f"sample_{step}.jpg"))

    def _save_checkpoint(self, step: int) -> None:
        ckpt = {"iter": step, "model": self.model.state_dict(), "optim": self.optim.state_dict()}
        save_checkpoint(str(self.output_dir / f"checkpoint_{step}.pth"), ckpt)

    def train(self) -> None:
        self.model.train()
        for step in range(self.state.step, self.state.max_iter):
            content, style = self._next_batch()
            stylized, total, loss_c, loss_s, loss_b, loss_mi = self._compute_losses(content, style, step)

            self.optim.zero_grad()
            total.backward()
            self.optim.step()

            if (step + 1) % self.log_every == 0:
                print(
                    f"[{step+1}/{self.state.max_iter}] total={total.item():.4f} "
                    f"Lc={loss_c.item():.4f} Ls={loss_s.item():.4f} "
                    f"LB={loss_b.item():.4f} Lmi={loss_mi.item():.4f}"
                )

            if (step + 1) % self.sample_every == 0:
                self._save_sample(content, style, stylized, step + 1)

            if (step + 1) % self.save_every == 0 or (step + 1) == self.state.max_iter:
                self._save_checkpoint(step + 1)

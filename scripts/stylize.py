import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from infominrev.config import load_config, get
from infominrev.models.infominrev import InfoMinRevNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def build_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


def load_images(path: str):
    p = Path(path)
    if p.is_dir():
        return [x for x in p.iterdir() if x.is_file()]
    return [p]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(get(config, "device", default="cuda") if torch.cuda.is_available() else "cpu")

    model = InfoMinRevNet(
        in_channels=get(config, "model", "in_channels", default=3),
        pad_channels=get(config, "model", "pad_channels", default=5),
        num_blocks=get(config, "model", "num_blocks", default=20),
        num_squeeze=get(config, "model", "num_squeeze", default=2),
        hidden_channels=get(config, "model", "hidden_channels", default=512),
        blocks_per_stage=get(config, "model", "blocks_per_stage", default=None),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    size = get(config, "data", "image_size", default=512)
    tf = build_transform(size)

    content_paths = load_images(args.content)
    style_paths = load_images(args.style)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for style_path in style_paths:
            style_img = tf(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)
            for content_path in content_paths:
                content_img = tf(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
                stylized, _ = model(content_img, style_img)
                name = f"{content_path.stem}__{style_path.stem}.jpg"
                save_image(stylized, str(out_dir / name))


if __name__ == "__main__":
    main()

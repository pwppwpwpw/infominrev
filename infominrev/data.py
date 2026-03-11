import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils import data
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FlatFolderDataset(data.Dataset):
    def __init__(self, root: str, transform: transforms.Compose):
        self.root = Path(root)
        self.paths = [p for p in self.root.iterdir() if p.is_file()]
        self.transform = transform

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.paths)


class InfiniteSampler(data.Sampler):
    def __init__(self, data_source: data.Dataset):
        self.num_samples = len(data_source)

    def __iter__(self) -> Iterable[int]:
        i = self.num_samples - 1
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

    def __len__(self) -> int:
        return 2 ** 31


def build_transforms(image_size: int, crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
        ]
    )


def build_loaders(
    content_dir: str,
    style_dir: str,
    image_size: int,
    crop_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[data.DataLoader, data.DataLoader]:
    transform = build_transforms(image_size, crop_size)
    content_dataset = FlatFolderDataset(content_dir, transform)
    style_dataset = FlatFolderDataset(style_dir, transform)

    content_loader = data.DataLoader(
        content_dataset,
        batch_size=batch_size,
        sampler=InfiniteSampler(content_dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    style_loader = data.DataLoader(
        style_dataset,
        batch_size=batch_size,
        sampler=InfiniteSampler(style_dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return content_loader, style_loader

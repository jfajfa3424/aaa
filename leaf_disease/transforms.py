"""Train / eval transforms. ImageNet stats keep pretrained backbones honest."""

from __future__ import annotations

from torchvision import transforms as T

from .config import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(img_size: int = 224, train: bool = True) -> T.Compose:
    if train:
        return T.Compose(
            [
                T.Resize((img_size + 32, img_size + 32)),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

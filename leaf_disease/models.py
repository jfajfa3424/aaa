"""Backbones named in the listing: VGG, ResNet, MobileNet, EfficientNet.

The proposed model is ResNet-50 + CBAM. TinyCNN is only for `--demo`.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    efficientnet_b0,
    mobilenet_v2,
    resnet50,
    vgg16,
)

from .cbam import CBAM


def _tv_weights(enum_cls, pretrained: bool):
    return enum_cls.DEFAULT if pretrained else None


class TinyCNN(nn.Module):
    """Lightweight CNN so the training pipeline can be smoke-tested on CPU."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.extract_features(x))


class ResNet50CBAM(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        backbone = resnet50(weights=_tv_weights(ResNet50_Weights, pretrained))
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)
        self.pool = backbone.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        return self.pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.extract_features(x))


class _HeadSwap(nn.Module):
    """Torchvision backbone with a replaced linear head and a feature hook."""

    def __init__(self, net: nn.Module, feature_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor], num_classes: int) -> None:
        super().__init__()
        self.net = net
        self._feature_fn = feature_fn
        _replace_head(self.net, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._feature_fn(self.net, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _replace_head(net: nn.Module, num_classes: int) -> int:
    if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
        return in_features
    if hasattr(net, "classifier"):
        classifier = net.classifier
        if isinstance(classifier, nn.Linear):
            in_features = classifier.in_features
            net.classifier = nn.Linear(in_features, num_classes)
            return in_features
        if isinstance(classifier, nn.Sequential):
            last = classifier[-1]
            if isinstance(last, nn.Linear):
                in_features = last.in_features
                classifier[-1] = nn.Linear(in_features, num_classes)
                return in_features
    raise TypeError(f"Cannot replace classification head of {type(net).__name__}")


def _resnet_features(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)
    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)
    return net.avgpool(x).flatten(1)


def _vgg_features(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = net.features(x)
    x = net.avgpool(x)
    return torch.flatten(x, 1)


def _mobilenet_features(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = net.features(x)
    return net.avgpool(x).flatten(1)


def _efficientnet_features(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = net.features(x)
    x = net.avgpool(x)
    return torch.flatten(x, 1)


def build_vgg16(num_classes: int, pretrained: bool = False) -> nn.Module:
    net = vgg16(weights=_tv_weights(VGG16_Weights, pretrained))
    return _HeadSwap(net, _vgg_features, num_classes)


def build_resnet50(num_classes: int, pretrained: bool = False) -> nn.Module:
    net = resnet50(weights=_tv_weights(ResNet50_Weights, pretrained))
    return _HeadSwap(net, _resnet_features, num_classes)


def build_mobilenet_v2(num_classes: int, pretrained: bool = False) -> nn.Module:
    net = mobilenet_v2(weights=_tv_weights(MobileNet_V2_Weights, pretrained))
    return _HeadSwap(net, _mobilenet_features, num_classes)


def build_efficientnet_b0(num_classes: int, pretrained: bool = False) -> nn.Module:
    net = efficientnet_b0(weights=_tv_weights(EfficientNet_B0_Weights, pretrained))
    return _HeadSwap(net, _efficientnet_features, num_classes)


def build_resnet50_cbam(num_classes: int, pretrained: bool = False) -> nn.Module:
    return ResNet50CBAM(num_classes=num_classes, pretrained=pretrained)


def build_tiny(num_classes: int, pretrained: bool = False) -> nn.Module:
    del pretrained
    return TinyCNN(num_classes=num_classes)


BUILDERS: dict[str, Callable[[int, bool], nn.Module]] = {
    "tiny": build_tiny,
    "vgg16": build_vgg16,
    "resnet50": build_resnet50,
    "mobilenet_v2": build_mobilenet_v2,
    "efficientnet_b0": build_efficientnet_b0,
    "resnet50_cbam": build_resnet50_cbam,
}

MODEL_NAMES = tuple(BUILDERS.keys())


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key not in BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")
    return BUILDERS[key](num_classes, pretrained)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

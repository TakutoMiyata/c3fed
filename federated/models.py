"""Model architectures for federated CIFAR-10 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision import models


class CifarCnn(nn.Module):
    """A lightweight CNN tuned for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        return self.classifier(x)


AvailableArch = Literal["simple_cnn", "resnet18"]


@dataclass(frozen=True)
class ModelConfig:
    arch: AvailableArch = "simple_cnn"
    num_classes: int = 10
    pretrained: bool = False


def create_model(config: ModelConfig) -> nn.Module:
    """Factory for model creation based on the provided config."""
    if config.arch == "simple_cnn":
        return CifarCnn(num_classes=config.num_classes)
    if config.arch == "resnet18":
        model = models.resnet18(weights=None if not config.pretrained else models.ResNet18_Weights.DEFAULT)
        # Replace the final layer to match CIFAR-10 output classes.
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
        return model
    raise ValueError(f"Unsupported architecture '{config.arch}'")


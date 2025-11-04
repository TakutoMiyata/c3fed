"""Utilities for preparing CIFAR-10 data for federated learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _default_transform(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ]
    )


def load_cifar10(data_dir: str, download: bool = True) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR-10 train/test datasets with standard transforms."""
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=_default_transform(train=True),
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=_default_transform(train=False),
    )
    return train_dataset, test_dataset


def _iid_partition(
    dataset: torch.utils.data.Dataset,
    num_clients: int,
    seed: int,
) -> List[Subset]:
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, split.tolist()) for split in splits]


def partition_dataset(
    dataset: torch.utils.data.Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[Subset]:
    """Split a dataset into IID client subsets."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if num_clients > len(dataset):
        raise ValueError("num_clients cannot exceed dataset size")
    return _iid_partition(dataset, num_clients, seed)


@dataclass(frozen=True)
class FederatedDataLoaders:
    train_loaders: Sequence[DataLoader]
    test_loader: DataLoader


def create_federated_loaders(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    num_clients: int,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 2,
) -> FederatedDataLoaders:
    """Prepare federated train loaders along with the shared test loader."""
    client_subsets = partition_dataset(train_dataset, num_clients=num_clients, seed=seed)
    train_loaders: List[DataLoader] = []
    for subset in client_subsets:
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        train_loaders.append(loader)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return FederatedDataLoaders(train_loaders=train_loaders, test_loader=test_loader)


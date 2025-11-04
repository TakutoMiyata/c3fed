import pytest
import torch
from torch.utils.data import TensorDataset

from federated.data import create_federated_loaders, partition_dataset


def _make_dataset(size: int = 100) -> TensorDataset:
    data = torch.randn(size, 3, 32, 32)
    labels = torch.randint(0, 10, (size,), dtype=torch.long)
    return TensorDataset(data, labels)


def test_partition_dataset_balanced_sizes():
    dataset = _make_dataset(100)
    subsets = partition_dataset(dataset, num_clients=5, seed=0)
    lengths = [len(subset) for subset in subsets]

    assert sum(lengths) == len(dataset)
    assert max(lengths) - min(lengths) <= 1
    # Ensure sample indices do not overlap.
    concatenated = torch.cat([torch.tensor(subset.indices) for subset in subsets])
    assert concatenated.unique().numel() == len(dataset)


def test_partition_dataset_invalid_clients():
    dataset = _make_dataset(5)
    with pytest.raises(ValueError):
        partition_dataset(dataset, num_clients=0)
    with pytest.raises(ValueError):
        partition_dataset(dataset, num_clients=10)


def test_create_federated_loaders_matches_client_count():
    dataset = _make_dataset(20)
    test_dataset = _make_dataset(10)
    loaders = create_federated_loaders(
        dataset,
        test_dataset,
        num_clients=4,
        batch_size=8,
        seed=0,
        num_workers=0,
    )

    assert len(loaders.train_loaders) == 4
    batch = next(iter(loaders.train_loaders[0]))
    assert batch[0].shape[1:] == (3, 32, 32)
    test_batch = next(iter(loaders.test_loader))
    assert test_batch[0].shape[1:] == (3, 32, 32)


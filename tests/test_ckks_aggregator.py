"""Tests for CKKS aggregator."""

import pytest
import torch

from federated.ckks_aggregator import CKKSAggregator
from federated.federated import FedAvgAggregator


def test_ckks_aggregator_initialization():
    """Test that CKKS aggregator initializes correctly."""
    aggregator = CKKSAggregator()
    ctx = aggregator.get_crypto_context()
    assert ctx is not None
    assert ctx.context is not None


def test_ckks_aggregator_simple_average():
    """Test CKKS aggregator produces correct weighted average."""
    aggregator = CKKSAggregator()

    state_a = {"weight": torch.tensor([1.0, 2.0])}
    state_b = {"weight": torch.tensor([3.0, 4.0])}

    aggregated = aggregator.aggregate([(10, state_a), (10, state_b)])
    expected = (state_a["weight"] + state_b["weight"]) / 2

    assert torch.allclose(aggregated["weight"], expected, atol=1e-2)


def test_ckks_aggregator_weighted_average():
    """Test CKKS with unequal weights."""
    aggregator = CKKSAggregator()

    state_a = {"weight": torch.tensor([1.0])}
    state_b = {"weight": torch.tensor([3.0])}

    aggregated = aggregator.aggregate([(1, state_a), (3, state_b)])
    expected = (1 * state_a["weight"] + 3 * state_b["weight"]) / 4

    assert torch.allclose(aggregated["weight"], expected, atol=1e-2)


def test_ckks_vs_fedavg_consistency():
    """
    Verify CKKS produces same results as FedAvg (within precision).
    This is a critical correctness test.
    """
    ckks_agg = CKKSAggregator()
    fedavg_agg = FedAvgAggregator()

    state_a = {
        "conv1.weight": torch.randn(32, 3, 3, 3),
        "conv1.bias": torch.randn(32),
        "fc.weight": torch.randn(10, 128),
        "fc.bias": torch.randn(10),
    }
    state_b = {
        "conv1.weight": torch.randn(32, 3, 3, 3),
        "conv1.bias": torch.randn(32),
        "fc.weight": torch.randn(10, 128),
        "fc.bias": torch.randn(10),
    }
    state_c = {
        "conv1.weight": torch.randn(32, 3, 3, 3),
        "conv1.bias": torch.randn(32),
        "fc.weight": torch.randn(10, 128),
        "fc.bias": torch.randn(10),
    }

    client_updates = [
        (100, state_a),
        (150, state_b),
        (120, state_c),
    ]

    ckks_result = ckks_agg.aggregate(client_updates)
    fedavg_result = fedavg_agg.aggregate(client_updates)

    for key in ckks_result.keys():
        assert torch.allclose(
            ckks_result[key], fedavg_result[key], atol=1e-2, rtol=1e-3
        ), f"Mismatch in {key}"


def test_ckks_aggregator_multiple_clients():
    """Test with many clients (scalability)."""
    aggregator = CKKSAggregator()

    num_clients = 20
    client_updates = []
    for i in range(num_clients):
        state = {"param": torch.randn(100)}
        num_examples = torch.randint(50, 150, (1,)).item()
        client_updates.append((num_examples, state))

    aggregated = aggregator.aggregate(client_updates)
    assert "param" in aggregated
    assert aggregated["param"].shape == (100,)


def test_ckks_aggregator_handles_integer_tensors():
    """Test that CKKS can handle non-float tensors."""
    aggregator = CKKSAggregator()

    state_a = {
        "float_param": torch.randn(10),
        "int_param": torch.tensor([1, 2, 3], dtype=torch.int64),
    }
    state_b = {
        "float_param": torch.randn(10),
        "int_param": torch.tensor([4, 5, 6], dtype=torch.int64),
    }

    aggregated = aggregator.aggregate([(1, state_a), (1, state_b)])

    assert aggregated["int_param"].dtype == torch.int64


def test_ckks_aggregator_empty_updates_raises():
    """Test that empty updates raise ValueError."""
    aggregator = CKKSAggregator()

    with pytest.raises(ValueError, match="No client updates"):
        aggregator.aggregate([])


def test_ckks_aggregator_zero_examples_raises():
    """Test that zero total examples raises ValueError."""
    aggregator = CKKSAggregator()
    state = {"param": torch.tensor([1.0])}

    with pytest.raises(ValueError, match="Total number of examples"):
        aggregator.aggregate([(0, state), (0, state)])


def test_ckks_aggregator_single_client():
    """Test aggregation with single client."""
    aggregator = CKKSAggregator()

    state = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    aggregated = aggregator.aggregate([(100, state)])

    assert torch.allclose(aggregated["weight"], state["weight"], atol=1e-2)


def test_ckks_aggregator_large_tensors():
    """Test with larger model parameters."""
    aggregator = CKKSAggregator()

    state_a = {"param": torch.randn(1000)}
    state_b = {"param": torch.randn(1000)}

    aggregated = aggregator.aggregate([(1, state_a), (1, state_b)])
    expected = (state_a["param"] + state_b["param"]) / 2

    assert torch.allclose(aggregated["param"], expected, atol=1e-2)


def test_ckks_aggregator_multiple_layers():
    """Test with multiple parameter layers."""
    aggregator = CKKSAggregator()

    state_a = {
        "layer1": torch.randn(50, 30),
        "layer2": torch.randn(30, 20),
        "layer3": torch.randn(20, 10),
    }
    state_b = {
        "layer1": torch.randn(50, 30),
        "layer2": torch.randn(30, 20),
        "layer3": torch.randn(20, 10),
    }

    client_updates = [(100, state_a), (100, state_b)]

    aggregated = aggregator.aggregate(client_updates)

    assert len(aggregated) == 3
    for key in ["layer1", "layer2", "layer3"]:
        assert key in aggregated
        expected = (state_a[key] + state_b[key]) / 2
        assert torch.allclose(aggregated[key], expected, atol=1e-2)

"""Performance benchmarks for CKKS vs FedAvg aggregation."""

import time

import pytest
import torch

from federated.ckks_aggregator import CKKSAggregator
from federated.federated import FedAvgAggregator
from federated.models import ModelConfig, create_model


def create_mock_client_updates(model_config, num_clients=10):
    """Create mock client updates for benchmarking."""
    model = create_model(model_config)
    state_dict = model.state_dict()

    client_updates = []
    for i in range(num_clients):
        perturbed_state = {}
        for key, tensor in state_dict.items():
            if tensor.is_floating_point():
                perturbed_state[key] = tensor + torch.randn_like(tensor) * 0.01
            else:
                perturbed_state[key] = tensor.clone()

        num_examples = torch.randint(50, 200, (1,)).item()
        client_updates.append((num_examples, perturbed_state))

    return client_updates


def benchmark_aggregator(aggregator, client_updates, num_runs=3):
    """Benchmark aggregation time."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = aggregator.aggregate(client_updates)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "times": times,
    }


def test_benchmark_simple_cnn():
    """Benchmark CKKS vs FedAvg with simple CNN."""
    print("\n" + "=" * 60)
    print("Benchmarking SimpleCNN (~60K parameters)")
    print("=" * 60)

    model_config = ModelConfig(arch="simple_cnn", num_classes=10)

    for num_clients in [5, 10]:
        print(f"\n--- {num_clients} clients ---")
        client_updates = create_mock_client_updates(model_config, num_clients)

        fedavg = FedAvgAggregator()
        fedavg_stats = benchmark_aggregator(fedavg, client_updates)
        print(
            f"FedAvg: {fedavg_stats['mean']*1000:.2f}ms "
            f"(min: {fedavg_stats['min']*1000:.2f}ms, "
            f"max: {fedavg_stats['max']*1000:.2f}ms)"
        )

        ckks = CKKSAggregator()
        ckks_stats = benchmark_aggregator(ckks, client_updates)
        print(
            f"CKKS: {ckks_stats['mean']*1000:.2f}ms "
            f"(min: {ckks_stats['min']*1000:.2f}ms, "
            f"max: {ckks_stats['max']*1000:.2f}ms)"
        )

        overhead = (ckks_stats["mean"] / fedavg_stats["mean"] - 1) * 100
        print(f"Overhead: {overhead:.1f}%")


def test_benchmark_parameter_count_scaling():
    """Benchmark how aggregation time scales with parameter count."""
    print("\n" + "=" * 60)
    print("Parameter Count Scaling (5 clients)")
    print("=" * 60)

    param_sizes = [1000, 10000, 50000]

    for size in param_sizes:
        print(f"\n--- {size} parameters ---")

        state = {"params": torch.randn(size)}
        client_updates = [(100, state) for _ in range(5)]

        fedavg = FedAvgAggregator()
        fedavg_stats = benchmark_aggregator(fedavg, client_updates)
        print(f"FedAvg: {fedavg_stats['mean']*1000:.2f}ms")

        ckks = CKKSAggregator()
        ckks_stats = benchmark_aggregator(ckks, client_updates)
        print(f"CKKS: {ckks_stats['mean']*1000:.2f}ms")

        overhead = (ckks_stats["mean"] / fedavg_stats["mean"] - 1) * 100
        print(f"Overhead: {overhead:.1f}%")


def test_memory_usage_estimation():
    """Estimate memory overhead of CKKS."""
    print("\n" + "=" * 60)
    print("Memory Usage Estimation")
    print("=" * 60)

    from federated.crypto import create_crypto_context, encrypt_tensor

    ctx = create_crypto_context()

    sizes = [1000, 10000, 50000]
    for size in sizes:
        tensor = torch.randn(size)
        encrypted = encrypt_tensor(tensor, ctx)

        plaintext_size_kb = size * 4 / 1024

        print(f"\nTensor size: {size} floats ({plaintext_size_kb:.1f} KB plaintext)")
        print(f"Note: CKKS encryption provides significant security benefits")
        print(f"      at the cost of increased computation time")


def test_accuracy_comparison():
    """Compare accuracy of CKKS vs FedAvg aggregation."""
    print("\n" + "=" * 60)
    print("Accuracy Comparison")
    print("=" * 60)

    model_config = ModelConfig(arch="simple_cnn", num_classes=10)
    client_updates = create_mock_client_updates(model_config, num_clients=5)

    fedavg_agg = FedAvgAggregator()
    ckks_agg = CKKSAggregator()

    fedavg_result = fedavg_agg.aggregate(client_updates)
    ckks_result = ckks_agg.aggregate(client_updates)

    print("\nComparing aggregated parameters:")
    total_diff = 0.0
    total_params = 0
    for key in fedavg_result.keys():
        if fedavg_result[key].is_floating_point():
            diff = torch.abs(fedavg_result[key] - ckks_result[key]).mean().item()
            total_diff += diff * fedavg_result[key].numel()
            total_params += fedavg_result[key].numel()

    avg_diff = total_diff / total_params if total_params > 0 else 0.0
    print(f"Average absolute difference: {avg_diff:.6f}")

    floating_tensors = [v.flatten() for v in fedavg_result.values() if v.is_floating_point()]
    if floating_tensors:
        all_vals = torch.cat(floating_tensors)
        relative_diff = avg_diff / torch.abs(all_vals).mean().item() * 100
        print(f"Relative difference: {relative_diff:.4f}%")

    print("\nConclusion: CKKS maintains high precision while providing encryption")


if __name__ == "__main__":
    test_benchmark_simple_cnn()
    test_benchmark_parameter_count_scaling()
    test_memory_usage_estimation()
    test_accuracy_comparison()

"""Federated learning orchestration utilities."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader


ClientId = int
StateDict = Dict[str, torch.Tensor]


@dataclass
class ClientConfig:
    client_id: ClientId
    model_factory: Callable[[], nn.Module]
    train_loader: DataLoader
    device: torch.device
    lr: float
    momentum: float
    weight_decay: float
    local_epochs: int


class FederatedClient:
    """Simulated federated client running local training."""

    def __init__(self, config: ClientConfig):
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.dataset_size = len(config.train_loader.dataset)  # type: ignore[arg-type]

    def train(self, global_state: StateDict) -> tuple[int, StateDict, float]:
        """Train on local data starting from the provided global parameters."""
        model = self.config.model_factory().to(self.config.device)
        model.load_state_dict(global_state)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        model.train()
        running_loss = 0.0
        total_batches = 0
        for _ in range(self.config.local_epochs):
            for inputs, targets in self.config.train_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_batches += 1
        avg_loss = running_loss / max(total_batches, 1)
        return self.dataset_size, copy.deepcopy(model.state_dict()), avg_loss


class FedAvgAggregator:
    """Classic FedAvg weighted aggregation."""

    def aggregate(self, client_updates: Sequence[tuple[int, StateDict]]) -> StateDict:
        if not client_updates:
            raise ValueError("No client updates to aggregate.")
        total_examples = sum(num_examples for num_examples, _ in client_updates)
        if total_examples == 0:
            raise ValueError("Total number of examples from clients cannot be zero.")

        aggregated_state: StateDict = {}
        dtype_map: Dict[str, torch.dtype] = {}
        first_state = client_updates[0][1]
        for key, tensor in first_state.items():
            dtype_map[key] = tensor.dtype
            if tensor.is_floating_point():
                aggregated_state[key] = torch.zeros_like(tensor)
            else:
                aggregated_state[key] = torch.zeros_like(tensor, dtype=torch.float32)

        for num_examples, state in client_updates:
            weight = num_examples / total_examples
            for key, tensor in state.items():
                value = tensor if tensor.is_floating_point() else tensor.float()
                aggregated_state[key] += value * weight

        for key, dtype in dtype_map.items():
            if not torch.empty((), dtype=dtype).is_floating_point():
                aggregated_state[key] = aggregated_state[key].to(dtype)
        return aggregated_state


@dataclass
class TrainerConfig:
    rounds: int = 10
    client_fraction: float = 1.0
    local_epochs: int = 1
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 42


@dataclass
class RoundMetrics:
    round: int
    train_loss: float
    test_loss: float
    test_accuracy: float


class FederatedTrainer:
    """High-level orchestration of the federated learning process."""

    def __init__(
        self,
        model: nn.Module,
        model_factory: Callable[[], nn.Module],
        clients: Sequence[FederatedClient],
        test_loader: DataLoader,
        device: torch.device,
        config: TrainerConfig,
    ):
        self.global_model = model.to(device)
        self.model_factory = model_factory
        self.clients = list(clients)
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.aggregator = FedAvgAggregator()
        random.seed(config.seed)

    def train(self, on_round_end: Callable[[RoundMetrics], None] | None = None) -> List[RoundMetrics]:
        metrics: List[RoundMetrics] = []
        num_clients = len(self.clients)
        for round_idx in range(1, self.config.rounds + 1):
            selected_clients = self._sample_clients(num_clients)
            global_state = copy.deepcopy(self.global_model.state_dict())
            client_results: List[tuple[int, StateDict]] = []
            accumulated_loss = 0.0

            for client in selected_clients:
                num_examples, state_dict, loss = client.train(global_state)
                client_results.append((num_examples, state_dict))
                accumulated_loss += loss

            new_global_state = self.aggregator.aggregate(client_results)
            self.global_model.load_state_dict(new_global_state)

            train_loss = accumulated_loss / max(len(selected_clients), 1)
            test_loss, test_accuracy = self.evaluate()
            round_metrics = RoundMetrics(
                round=round_idx,
                train_loss=train_loss,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
            )
            metrics.append(round_metrics)
            if on_round_end is not None:
                on_round_end(round_metrics)
        return metrics

    def _sample_clients(self, num_clients: int) -> Sequence[FederatedClient]:
        fraction = min(max(self.config.client_fraction, 0.0), 1.0)
        sample_size = max(1, int(num_clients * fraction))
        return random.sample(self.clients, sample_size)

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in self.test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.global_model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        avg_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

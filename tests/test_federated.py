import torch
from torch.utils.data import DataLoader, TensorDataset

from federated.data import create_federated_loaders
from federated.federated import (
    ClientConfig,
    FedAvgAggregator,
    FederatedClient,
    FederatedTrainer,
    TrainerConfig,
)
from federated.models import ModelConfig, create_model


def test_fedavg_aggregator_weighted_average():
    aggregator = FedAvgAggregator()
    state_a = {"weight": torch.tensor([1.0])}
    state_b = {"weight": torch.tensor([3.0])}
    aggregated = aggregator.aggregate([(1, state_a), (3, state_b)])
    expected = (1 * state_a["weight"] + 3 * state_b["weight"]) / 4
    assert torch.allclose(aggregated["weight"], expected)


def test_federated_client_train_updates_model():
    dataset = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model_config = ModelConfig(arch="simple_cnn", num_classes=10)

    def model_factory():
        return create_model(model_config)

    initial_model = model_factory()
    config = ClientConfig(
        client_id=0,
        model_factory=model_factory,
        train_loader=loader,
        device=torch.device("cpu"),
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
        local_epochs=1,
    )
    client = FederatedClient(config)
    dataset_size, updated_state, avg_loss = client.train(initial_model.state_dict())

    assert dataset_size == len(dataset)
    assert isinstance(updated_state, dict)
    assert avg_loss >= 0.0


def test_federated_trainer_runs_single_round():
    train_dataset = TensorDataset(torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,)))
    test_dataset = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    loaders = create_federated_loaders(
        train_dataset,
        test_dataset,
        num_clients=2,
        batch_size=8,
        seed=123,
        num_workers=0,
    )
    model_config = ModelConfig(arch="simple_cnn", num_classes=10)

    def model_factory():
        return create_model(model_config)

    global_model = model_factory()
    clients = []
    for client_id, train_loader in enumerate(loaders.train_loaders):
        client_config = ClientConfig(
            client_id=client_id,
            model_factory=model_factory,
            train_loader=train_loader,
            device=torch.device("cpu"),
            lr=0.01,
            momentum=0.0,
            weight_decay=0.0,
            local_epochs=1,
        )
        clients.append(FederatedClient(client_config))

    trainer = FederatedTrainer(
        model=global_model,
        model_factory=model_factory,
        clients=clients,
        test_loader=loaders.test_loader,
        device=torch.device("cpu"),
        config=TrainerConfig(
            rounds=1,
            client_fraction=1.0,
            local_epochs=1,
            lr=0.01,
            momentum=0.0,
            weight_decay=0.0,
            seed=123,
        ),
    )

    metrics = trainer.train()
    assert len(metrics) == 1
    first_round = metrics[0]
    assert 0.0 <= first_round.test_accuracy <= 1.0
    assert first_round.train_loss >= 0.0


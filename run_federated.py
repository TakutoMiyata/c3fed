"""Entry point for running federated learning with CIFAR-10."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import torch

from federated.data import create_federated_loaders, load_cifar10
from federated.federated import (
    ClientConfig,
    FederatedClient,
    FederatedTrainer,
    RoundMetrics,
    TrainerConfig,
)
from federated.models import ModelConfig, create_model


def _build_parser(parents: list[argparse.ArgumentParser] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Federated learning on CIFAR-10 (FedAvg).",
        parents=parents or [],
    )
    parser.add_argument("--data-dir", type=str, default=str(Path.home() / ".cifar10"), help="Path to store CIFAR-10.")
    parser.add_argument("--num-clients", type=int, default=10, help="Total simulated clients.")
    parser.add_argument("--client-fraction", type=float, default=1.0, help="Fraction of clients sampled each round.")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds.")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client per round.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for client dataloaders.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for local training.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay coefficient.")
    parser.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "resnet18"], help="Model architecture.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights when available.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on.")
    parser.add_argument("--save-metrics", type=str, default="", help="Optional path to save metrics as JSON.")
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="", help="Path to JSON config file overriding defaults.")

    parser = _build_parser(parents=[config_parser])

    config_args, remaining = config_parser.parse_known_args()
    config_values: dict[str, object] = {}
    if config_args.config:
        config_path = Path(config_args.config).expanduser()
        try:
            with config_path.open("r", encoding="utf-8") as fp:
                loaded = json.load(fp)
        except FileNotFoundError as exc:
            parser.error(f"Config file not found: {config_path}")  # exits
        except json.JSONDecodeError as exc:
            parser.error(f"Failed to parse JSON config {config_path}: {exc.msg}")  # exits
        if not isinstance(loaded, dict):
            parser.error("Config file must contain a JSON object with option names as keys.")
        config_values = loaded

    if config_values:
        valid_dests = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}
        unknown_keys = sorted({key for key in config_values if key not in valid_dests})
        if unknown_keys:
            parser.error(f"Unknown config options: {', '.join(unknown_keys)}")
        parser.set_defaults(**config_values)

    args = parser.parse_args(remaining)
    args.config = config_args.config
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    args = parse_args()
    if args.config:
        print(f"Loaded configuration from {args.config}")
    set_seed(args.seed)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    print(f"Using device: {device}")
    print("Loading CIFAR-10...")
    train_dataset, test_dataset = load_cifar10(args.data_dir, download=True)

    loaders = create_federated_loaders(
        train_dataset,
        test_dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model_config = ModelConfig(arch=args.model, num_classes=10, pretrained=args.pretrained)

    def model_factory() -> torch.nn.Module:
        return create_model(model_config)

    global_model = model_factory()

    clients = []
    for client_id, train_loader in enumerate(loaders.train_loaders):
        client_config = ClientConfig(
            client_id=client_id,
            model_factory=model_factory,
            train_loader=train_loader,
            device=device,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            local_epochs=args.local_epochs,
        )
        clients.append(FederatedClient(client_config))

    trainer_config = TrainerConfig(
        rounds=args.rounds,
        client_fraction=args.client_fraction,
        local_epochs=args.local_epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    trainer = FederatedTrainer(
        model=global_model,
        model_factory=model_factory,
        clients=clients,
        test_loader=loaders.test_loader,
        device=device,
        config=trainer_config,
    )

    print("Starting federated training...")

    def log_round_progress(round_metrics: RoundMetrics) -> None:
        print(
            f"[Round {round_metrics.round:03d}] "
            f"train_loss={round_metrics.train_loss:.4f} "
            f"test_loss={round_metrics.test_loss:.4f} "
            f"test_acc={round_metrics.test_accuracy:.4f}",
            flush=True,
        )

    metrics = trainer.train(on_round_end=log_round_progress)

    if args.save_metrics:
        save_path = Path(args.save_metrics)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as fp:
            json.dump([dataclasses.asdict(m) for m in metrics], fp, indent=2)
        print(f"Metrics saved to {save_path}")


if __name__ == "__main__":
    main()

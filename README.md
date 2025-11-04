# Federated CIFAR-10 Training

This repository provides a simple yet configurable **FedAvg** implementation for the CIFAR-10 dataset using PyTorch. It simulates multiple federated clients locally, performs weighted model aggregation, and reports per-round metrics.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch will download CIFAR-10 automatically on the first run (network access required).

## Usage

```bash
python run_federated.py \
  --data-dir ./data \
  --num-clients 10 \
  --rounds 5 \
  --local-epochs 1 \
  --batch-size 64 \
  --client-fraction 0.5 \
  --model simple_cnn \
  --device cpu
```

Important flags:

- `--num-clients`: total simulated clients.
- `--client-fraction`: fraction of clients sampled each round.
- `--rounds`: number of federated communication rounds.
- `--local-epochs`: number of SGD epochs per round per client.
- `--save-metrics`: optional path to write the per-round metrics JSON.

Switch `--model resnet18 --pretrained` to reuse the torchvision ResNet18 backbone.

## Outputs

The script logs per-round training and evaluation metrics to stdout and can optionally persist them to JSON via `--save-metrics`.


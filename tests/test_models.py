import torch

from federated.models import ModelConfig, create_model


def test_simple_cnn_forward_shape():
    model = create_model(ModelConfig(arch="simple_cnn", num_classes=10))
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (4, 10)


def test_resnet18_forward_shape():
    model = create_model(ModelConfig(arch="resnet18", num_classes=10, pretrained=False))
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    assert output.shape == (2, 10)


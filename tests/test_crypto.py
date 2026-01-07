"""Tests for CKKS crypto operations."""

import numpy as np
import pytest
import torch

from federated.crypto import (
    CKKSParameters,
    add_encrypted_tensors,
    create_crypto_context,
    decrypt_state_dict,
    decrypt_tensor,
    encrypt_state_dict,
    encrypt_tensor,
    multiply_encrypted_by_scalar,
)


def test_create_crypto_context():
    """Test crypto context creation with default parameters."""
    ctx = create_crypto_context()
    assert ctx.context is not None
    assert ctx.params.poly_modulus_degree == 16384


def test_encrypt_decrypt_small_tensor():
    """Test encryption and decryption roundtrip for small tensor."""
    ctx = create_crypto_context()
    original = torch.tensor([1.0, 2.0, 3.0, 4.0])

    encrypted = encrypt_tensor(original, ctx)
    decrypted = decrypt_tensor(encrypted, ctx)

    assert torch.allclose(original, decrypted, atol=1e-3)


def test_encrypt_decrypt_large_tensor():
    """Test encryption/decryption for large tensor."""
    ctx = create_crypto_context()
    original = torch.randn(10000)

    encrypted = encrypt_tensor(original, ctx)
    decrypted = decrypt_tensor(encrypted, ctx)
    assert torch.allclose(original, decrypted, atol=1e-3)


def test_encrypt_decrypt_2d_tensor():
    """Test with 2D tensor (like weight matrices)."""
    ctx = create_crypto_context()
    original = torch.randn(64, 128)

    encrypted = encrypt_tensor(original, ctx)
    decrypted = decrypt_tensor(encrypted, ctx)

    assert decrypted.shape == original.shape
    assert torch.allclose(original, decrypted, atol=1e-3)


def test_encrypt_decrypt_integer_tensor():
    """Test handling of non-floating point tensors."""
    ctx = create_crypto_context()
    original = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    encrypted = encrypt_tensor(original, ctx)
    decrypted = decrypt_tensor(encrypted, ctx)

    assert decrypted.dtype == torch.int64
    assert torch.allclose(original.float(), decrypted.float(), atol=0.5)


def test_homomorphic_addition():
    """Test encrypted addition."""
    ctx = create_crypto_context()
    tensor_a = torch.tensor([1.0, 2.0, 3.0])
    tensor_b = torch.tensor([4.0, 5.0, 6.0])

    enc_a = encrypt_tensor(tensor_a, ctx)
    enc_b = encrypt_tensor(tensor_b, ctx)
    enc_sum = add_encrypted_tensors(enc_a, enc_b, ctx)

    decrypted_sum = decrypt_tensor(enc_sum, ctx)
    expected = tensor_a + tensor_b

    assert torch.allclose(decrypted_sum, expected, atol=1e-3)


def test_homomorphic_scalar_multiplication():
    """Test encrypted multiplication by scalar."""
    ctx = create_crypto_context()
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    scalar = 2.5

    encrypted = encrypt_tensor(tensor, ctx)
    enc_multiplied = multiply_encrypted_by_scalar(encrypted, scalar, ctx)

    decrypted = decrypt_tensor(enc_multiplied, ctx)
    expected = tensor * scalar

    assert torch.allclose(decrypted, expected, atol=1e-3)


def test_weighted_average_encrypted():
    """Test computing weighted average in encrypted space."""
    ctx = create_crypto_context()

    tensor_a = torch.tensor([10.0, 20.0, 30.0])
    tensor_b = torch.tensor([40.0, 50.0, 60.0])
    weight_a = 0.3
    weight_b = 0.7

    enc_a = encrypt_tensor(tensor_a, ctx)
    enc_b = encrypt_tensor(tensor_b, ctx)

    weighted_a = multiply_encrypted_by_scalar(enc_a, weight_a, ctx)
    weighted_b = multiply_encrypted_by_scalar(enc_b, weight_b, ctx)
    enc_avg = add_encrypted_tensors(weighted_a, weighted_b, ctx)

    decrypted_avg = decrypt_tensor(enc_avg, ctx)
    expected = tensor_a * weight_a + tensor_b * weight_b

    assert torch.allclose(decrypted_avg, expected, atol=1e-3)


def test_encrypt_decrypt_state_dict():
    """Test encrypting and decrypting a full state dict."""
    ctx = create_crypto_context()
    state_dict = {
        "layer1.weight": torch.randn(64, 32),
        "layer1.bias": torch.randn(64),
        "layer2.weight": torch.randn(10, 64),
        "layer2.bias": torch.randn(10),
    }

    encrypted_state = encrypt_state_dict(state_dict, ctx)
    decrypted_state = decrypt_state_dict(encrypted_state, ctx)

    for key in state_dict.keys():
        assert torch.allclose(state_dict[key], decrypted_state[key], atol=1e-3)


def test_add_encrypted_tensors_mismatch_raises():
    """Test that adding tensors with different shapes raises error."""
    ctx = create_crypto_context()
    small_tensor = torch.randn(10)
    large_tensor = torch.randn(20)

    enc_small = encrypt_tensor(small_tensor, ctx)
    enc_large = encrypt_tensor(large_tensor, ctx)

    with pytest.raises(ValueError, match="different shapes"):
        add_encrypted_tensors(enc_small, enc_large, ctx)


def test_encrypt_tensor_preserves_shape():
    """Test that encryption preserves tensor shape information."""
    ctx = create_crypto_context()
    shapes = [(10,), (5, 10), (3, 4, 5), (2, 3, 4, 5)]

    for shape in shapes:
        tensor = torch.randn(shape)
        encrypted = encrypt_tensor(tensor, ctx)
        assert encrypted.original_shape == shape

        decrypted = decrypt_tensor(encrypted, ctx)
        assert decrypted.shape == shape


def test_zero_tensor():
    """Test encryption/decryption of zero tensor."""
    ctx = create_crypto_context()
    original = torch.zeros(100)

    encrypted = encrypt_tensor(original, ctx)
    decrypted = decrypt_tensor(encrypted, ctx)

    assert torch.allclose(original, decrypted, atol=1e-3)


def test_negative_values():
    """Test encryption/decryption of negative values."""
    ctx = create_crypto_context()
    original = torch.tensor([-1.0, -2.5, -10.0, -100.0])

    encrypted = encrypt_tensor(original, ctx)
    decrypted = decrypt_tensor(encrypted, ctx)

    assert torch.allclose(original, decrypted, atol=1e-3)


def test_very_small_values():
    """Test encryption/decryption of very small values."""
    ctx = create_crypto_context()
    original = torch.tensor([1e-5, 1e-6, 1e-7, 1e-8])

    encrypted = encrypt_tensor(original, ctx)
    decrypted = decrypt_tensor(encrypted, ctx)

    assert torch.allclose(original, decrypted, atol=1e-6, rtol=1e-3)

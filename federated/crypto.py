"""CKKS homomorphic encryption utilities for federated learning."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tenseal as ts
import torch

StateDict = Dict[str, torch.Tensor]


@contextmanager
def suppress_output():
    """Temporarily suppress both stdout and stderr at file descriptor level."""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    old_stdout = os.dup(stdout_fd)
    old_stderr = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        sys.stdout.flush()
        sys.stderr.flush()
        yield
    finally:
        os.dup2(old_stdout, stdout_fd)
        os.dup2(old_stderr, stderr_fd)
        os.close(devnull_fd)
        os.close(old_stdout)
        os.close(old_stderr)


@dataclass(frozen=True)
class CKKSParameters:
    """Fixed CKKS parameters for the federated learning system."""

    poly_modulus_degree: int = 16384
    coeff_mod_bit_sizes: List[int] = None
    global_scale: int = 2**40

    def __post_init__(self):
        if self.coeff_mod_bit_sizes is None:
            object.__setattr__(
                self, "coeff_mod_bit_sizes", [60, 40, 40, 60]
            )


@dataclass
class CryptoContext:
    """Encapsulates the CKKS crypto context and keys."""

    context: ts.Context
    params: CKKSParameters


def create_crypto_context(
    params: CKKSParameters | None = None, suppress_warnings: bool = False
) -> CryptoContext:
    """
    Create a CKKS crypto context with fixed parameters.

    This should be called on the server side at the start of federated training.

    Args:
        params: CKKS parameters (uses defaults if None)
        suppress_warnings: If True, suppress TenSEAL warnings

    Returns:
        CryptoContext with generated keys
    """
    if params is None:
        params = CKKSParameters()

    if suppress_warnings:
        with suppress_output():
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=params.poly_modulus_degree,
                coeff_mod_bit_sizes=params.coeff_mod_bit_sizes,
            )
            context.global_scale = params.global_scale
            context.generate_galois_keys()
    else:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=params.poly_modulus_degree,
            coeff_mod_bit_sizes=params.coeff_mod_bit_sizes,
        )
        context.global_scale = params.global_scale
        context.generate_galois_keys()

    return CryptoContext(context=context, params=params)


@dataclass
class EncryptedTensor:
    """Represents an encrypted tensor with metadata."""

    ciphertext: ts.CKKSVector
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype


def encrypt_tensor(
    tensor: torch.Tensor,
    crypto_ctx: CryptoContext,
    suppress_warnings: bool = False,
) -> EncryptedTensor:
    """
    Encrypt a PyTorch tensor using CKKS.

    Args:
        tensor: PyTorch tensor to encrypt
        crypto_ctx: Crypto context
        suppress_warnings: If True, suppress TenSEAL warnings

    Returns:
        EncryptedTensor containing ciphertext and metadata
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype

    if not tensor.is_floating_point():
        tensor = tensor.float()

    flat_values = tensor.flatten().cpu().numpy().astype(np.float64).tolist()

    if suppress_warnings:
        with suppress_output():
            ciphertext = ts.ckks_vector(crypto_ctx.context, flat_values)
    else:
        ciphertext = ts.ckks_vector(crypto_ctx.context, flat_values)

    return EncryptedTensor(
        ciphertext=ciphertext,
        original_shape=original_shape,
        original_dtype=original_dtype,
    )


def decrypt_tensor(
    encrypted: EncryptedTensor,
    crypto_ctx: CryptoContext,
) -> torch.Tensor:
    """
    Decrypt an EncryptedTensor back to a PyTorch tensor.

    Args:
        encrypted: EncryptedTensor to decrypt
        crypto_ctx: Crypto context

    Returns:
        Decrypted PyTorch tensor with original shape and dtype
    """
    decrypted_values = encrypted.ciphertext.decrypt()

    total_elements = int(np.prod(encrypted.original_shape))
    decrypted_array = np.array(decrypted_values[:total_elements], dtype=np.float32)

    tensor = torch.from_numpy(decrypted_array).reshape(encrypted.original_shape)

    if not torch.empty((), dtype=encrypted.original_dtype).is_floating_point():
        tensor = tensor.round().to(encrypted.original_dtype)

    return tensor


def encrypt_state_dict(
    state_dict: StateDict,
    crypto_ctx: CryptoContext,
    suppress_warnings: bool = False,
) -> Dict[str, EncryptedTensor]:
    """Encrypt all tensors in a state dict."""
    encrypted_state = {}
    for key, tensor in state_dict.items():
        encrypted_state[key] = encrypt_tensor(tensor, crypto_ctx, suppress_warnings)
    return encrypted_state


def decrypt_state_dict(
    encrypted_state: Dict[str, EncryptedTensor],
    crypto_ctx: CryptoContext,
) -> StateDict:
    """Decrypt all tensors in an encrypted state dict."""
    state_dict = {}
    for key, encrypted_tensor in encrypted_state.items():
        state_dict[key] = decrypt_tensor(encrypted_tensor, crypto_ctx)
    return state_dict


def add_encrypted_tensors(
    enc1: EncryptedTensor,
    enc2: EncryptedTensor,
    crypto_ctx: CryptoContext,
) -> EncryptedTensor:
    """Add two encrypted tensors homomorphically."""
    if enc1.original_shape != enc2.original_shape:
        raise ValueError("Cannot add encrypted tensors with different shapes")

    result_ciphertext = enc1.ciphertext + enc2.ciphertext

    return EncryptedTensor(
        ciphertext=result_ciphertext,
        original_shape=enc1.original_shape,
        original_dtype=enc1.original_dtype,
    )


def multiply_encrypted_by_scalar(
    encrypted: EncryptedTensor,
    scalar: float,
    crypto_ctx: CryptoContext,
) -> EncryptedTensor:
    """Multiply encrypted tensor by plaintext scalar homomorphically."""
    result_ciphertext = encrypted.ciphertext * scalar

    return EncryptedTensor(
        ciphertext=result_ciphertext,
        original_shape=encrypted.original_shape,
        original_dtype=encrypted.original_dtype,
    )

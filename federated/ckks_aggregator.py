"""CKKS-based secure aggregation for federated learning."""

from __future__ import annotations

from typing import Dict, Sequence

from federated.crypto import (
    CKKSParameters,
    CryptoContext,
    EncryptedTensor,
    StateDict,
    add_encrypted_tensors,
    create_crypto_context,
    decrypt_state_dict,
    encrypt_state_dict,
    multiply_encrypted_by_scalar,
)


class CKKSAggregator:
    """
    Secure aggregation using CKKS homomorphic encryption.

    This aggregator:
    1. Generates crypto context and keys on initialization (server-side)
    2. Provides public key for clients to encrypt their updates
    3. Performs weighted averaging on encrypted model parameters
    4. Decrypts the final aggregated result

    The aggregation happens entirely in encrypted space, preserving
    privacy of individual client updates.
    """

    def __init__(
        self, params: CKKSParameters | None = None, suppress_warnings: bool = False
    ):
        """
        Initialize CKKS aggregator with crypto context.

        Args:
            params: CKKS parameters (uses defaults if None)
            suppress_warnings: If True, suppress TenSEAL warnings
        """
        self.crypto_ctx = create_crypto_context(params, suppress_warnings)
        self.suppress_warnings = suppress_warnings
        self._encrypted_cache: Dict[str, list[EncryptedTensor]] = {}

    def get_crypto_context(self) -> CryptoContext:
        """Get the crypto context (for clients to access public key)."""
        return self.crypto_ctx

    def aggregate(self, client_updates: Sequence[tuple[int, StateDict]]) -> StateDict:
        """
        Aggregate client updates using CKKS homomorphic encryption.

        This method:
        1. Encrypts each client's state dict with the public key
        2. Performs weighted averaging in encrypted space
        3. Decrypts the final result

        Args:
            client_updates: List of (num_examples, state_dict) tuples

        Returns:
            Aggregated state dict (decrypted)
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate.")

        total_examples = sum(num_examples for num_examples, _ in client_updates)
        if total_examples == 0:
            raise ValueError("Total number of examples from clients cannot be zero.")

        first_state = client_updates[0][1]
        param_keys = list(first_state.keys())

        aggregated_encrypted: Dict[str, EncryptedTensor | None] = {
            key: None for key in param_keys
        }

        for num_examples, state_dict in client_updates:
            weight = num_examples / total_examples

            encrypted_state = encrypt_state_dict(
                state_dict, self.crypto_ctx, self.suppress_warnings
            )

            for key in param_keys:
                weighted_encrypted = multiply_encrypted_by_scalar(
                    encrypted_state[key], weight, self.crypto_ctx
                )

                if aggregated_encrypted[key] is None:
                    aggregated_encrypted[key] = weighted_encrypted
                else:
                    aggregated_encrypted[key] = add_encrypted_tensors(
                        aggregated_encrypted[key], weighted_encrypted, self.crypto_ctx
                    )

        aggregated_state = decrypt_state_dict(aggregated_encrypted, self.crypto_ctx)

        return aggregated_state

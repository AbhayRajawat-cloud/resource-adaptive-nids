"""
Stage 5: Phase 5.3 FedAvg Server Aggregator.

Implements sample-weighted parameter aggregation across federated clients:
  theta_shared^{(r+1)} = sum_{k=0}^{K-1} (n_k / N_total) * theta_{shared, k}^{(r, E)}

Filtering & Protocols:
- FedBN: If is_mlp=True, excludes non-trainable BatchNorm buffers
  ('running_mean', 'running_var', 'num_batches_tracked') from aggregation.
  These buffers are retained locally on each client.
- FedPer: If is_fedper=True, excludes client-private classification head parameters
  ('classifier.3.*' for HeavyGRU, 'head.*' for SupervisedDenseMLP; 33 parameters)
  from server aggregation.
"""

from typing import Dict, List, Optional, Set
import torch


BN_BUFFER_SUBSTRINGS = ("running_mean", "running_var", "num_batches_tracked")

FEDPER_HEAD_PREFIXES = ("classifier.3.", "head.", "fc2.", "fc3.")
FEDPER_HEAD_EXACT_KEYS = {
    "classifier.3.weight",
    "classifier.3.bias",
    "head.weight",
    "head.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
}


def is_bn_buffer(key: str) -> bool:
    """Return True if key is a non-trainable BatchNorm buffer."""
    return any(sub in key for sub in BN_BUFFER_SUBSTRINGS)


def is_private_head(key: str) -> bool:
    """Return True if key belongs to the private classification head (33 params)."""
    if key in FEDPER_HEAD_EXACT_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in FEDPER_HEAD_PREFIXES)


def aggregate_weighted_fedavg(
    client_states: List[Dict[str, torch.Tensor]],
    client_sample_counts: List[int],
    is_mlp: bool = False,
    is_fedper: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Perform sample-weighted Federated Averaging (FedAvg) over shared parameters.

    Args:
        client_states: List of client state dicts (length K).
        client_sample_counts: List of client Gate-passed training sample counts n_k.
        is_mlp: If True, exclude non-trainable BatchNorm buffers (FedBN protocol).
        is_fedper: If True, exclude client-private classification head parameters (FedPer protocol).

    Returns:
        Dict[str, torch.Tensor]: Aggregated shared parameter state dict.
    """
    num_clients = len(client_states)
    if num_clients == 0:
        raise ValueError("Cannot aggregate empty list of client states.")
    if len(client_sample_counts) != num_clients:
        raise ValueError(
            f"Mismatch: {num_clients} client states but {len(client_sample_counts)} sample counts."
        )

    total_samples = sum(client_sample_counts)
    if total_samples <= 0:
        raise ValueError(f"Total training sample count must be positive, got {total_samples}")

    # Compute normalized sample weights
    weights = [float(n) / float(total_samples) for n in client_sample_counts]
    weight_sum = sum(weights)
    assert abs(weight_sum - 1.0) < 1e-6, f"Sample weights must sum to 1.0 +- 1e-6, got {weight_sum}"

    first_state = client_states[0]
    aggregated_state: Dict[str, torch.Tensor] = {}

    for key, sample_tensor in first_state.items():
        # 1. Filter BatchNorm non-trainable buffers if is_mlp=True (FedBN)
        if is_mlp and is_bn_buffer(key):
            continue

        # 2. Filter private classification head if is_fedper=True (FedPer)
        if is_fedper and is_private_head(key):
            continue

        # 3. Shape validation across all client states
        for k in range(num_clients):
            c_tensor = client_states[k].get(key)
            if c_tensor is None:
                raise KeyError(f"Key '{key}' missing from client {k} state dict.")
            if c_tensor.shape != sample_tensor.shape:
                raise ValueError(
                    f"Tensor shape mismatch for key '{key}': client 0 has {sample_tensor.shape}, "
                    f"client {k} has {c_tensor.shape}"
                )

        # 4. Weighted accumulation
        orig_dtype = sample_tensor.dtype
        # Float accumulation
        acc = torch.zeros_like(sample_tensor, dtype=torch.float32)
        for k in range(num_clients):
            w_k = weights[k]
            acc.add_(client_states[k][key].detach().to(torch.float32), alpha=w_k)

        # Convert back to original parameter dtype
        aggregated_state[key] = acc.to(orig_dtype)

    return aggregated_state


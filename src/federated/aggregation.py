"""Federated weight aggregation schemes (FedAvg)."""

from typing import List, Dict
import torch


def federated_averaging(
    client_weights: List[Dict[str, torch.Tensor]],
    sample_weights: List[int]
) -> Dict[str, torch.Tensor]:
    """Aggregate client GRU model weights using standard sample-weighted FedAvg."""
    if not client_weights:
        return {}
    
    total_samples = sum(sample_weights)
    aggregated = {}

    for key in client_weights[0].keys():
        aggregated[key] = sum(
            client_weights[i][key] * (sample_weights[i] / total_samples)
            for i in range(len(client_weights))
        )
    return aggregated


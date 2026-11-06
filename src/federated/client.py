"""Federated edge client abstraction."""

from typing import Dict, Any
import torch


class FederatedClient:
    """Simulated IoT Edge client holding private local partition.
    
    Maintains:
    - Local Autoencoder (private, not shared)
    - Local Gating Regressor (private, not shared)
    - Local GRU Classifier (weights uploaded for global FedAvg aggregation)
    """

    def __init__(self, client_id: int, config: Dict[str, Any] = None):
        self.client_id = client_id
        self.config = config or {}

    def train_local_gru(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform local training on GRU model and return updated weights."""
        return global_weights

    def get_num_samples(self) -> int:
        """Return the number of samples available locally for weighting in FedAvg."""
        return 0


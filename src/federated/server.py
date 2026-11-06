"""Federated orchestrator / central coordinator."""

from typing import List, Dict, Any
from src.federated.client import FederatedClient
from src.federated.aggregation import federated_averaging


class FederatedServer:
    """Central server orchestrating communication rounds and GRU parameter aggregation."""

    def __init__(self, clients: List[FederatedClient], config: Dict[str, Any] = None):
        self.clients = clients
        self.config = config or {}
        self.global_gru_weights = {}

    def run_round(self, round_idx: int) -> Dict[str, Any]:
        """Execute a single communication round of local training and FedAvg aggregation."""
        return {
            "round": round_idx,
            "clients_participated": len(self.clients)
        }


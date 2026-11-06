"""
Local Flow Autoencoder for lightweight, unsupervised anomaly detection.

Architecture:
    5-layer symmetric MLP: 38 -> 20 -> 10 -> 20 -> 38
    Parameters:
        Layer 1 (encoder.0): Linear(38, 20) -> 38 * 20 + 20 = 780
        Layer 2 (encoder.2): Linear(20, 10) -> 20 * 10 + 10 = 210
        Layer 3 (decoder.0): Linear(10, 20) -> 10 * 20 + 20 = 220
        Layer 4 (decoder.2): Linear(20, 38) -> 20 * 38 + 38 = 798
        Total trainable parameters = 780 + 210 + 220 + 798 = 2,008 (8.03 KB Float32)

Activations:
    Hidden layers: ELU(alpha=1.0)
    Output layer: Linear (Identity)

Weight Initialization:
    Kaiming Uniform with nonlinearity='leaky_relu' for ELU layers.
"""

from typing import Tuple
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """5-layer symmetric MLP Autoencoder for network flow anomaly detection."""

    def __init__(self, input_dim: int = 38):
        super().__init__()
        self.input_dim = input_dim

        # Encoder: 38 -> 20 -> 10
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ELU(alpha=1.0),
            nn.Linear(20, 10),
            nn.ELU(alpha=1.0),
        )

        # Decoder: 10 -> 20 -> 38
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ELU(alpha=1.0),
            nn.Linear(20, 38),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform initialization with leaky_relu gain for ELU layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns reconstructed input vector x_hat."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    @staticmethod
    def compute_mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample Mean Squared Error across all features.
        
        Args:
            x: Original input tensor of shape [B, D]
            x_hat: Reconstructed tensor of shape [B, D]
            
        Returns:
            1D tensor of shape [B] representing MSE per flow sample.
        """
        return torch.mean((x - x_hat) ** 2, dim=-1)


# Alias for backward compatibility if referenced elsewhere
TrafficAutoencoder = Autoencoder

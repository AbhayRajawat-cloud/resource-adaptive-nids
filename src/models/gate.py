"""
Self-Supervised Gating Regressor predicting Autoencoder log-reconstruction error.

Architecture:
    2-layer shallow MLP: 38 -> 16 -> 1
    Parameters:
        Layer 1 (net.0): Linear(38, 16) -> 38 * 16 + 16 = 624
        Layer 2 (net.2): Linear(16, 1)  -> 16 * 1 + 1   = 17
        Total trainable parameters = 624 + 17 = 641 (2.56 KB Float32)

Activations:
    Hidden layer: ELU(alpha=1.0)
    Output layer: Linear (Identity) returning scalar log-reconstruction error prediction e_hat.

Weight Initialization:
    Kaiming Uniform with nonlinearity='leaky_relu' for ELU layer.
"""

import torch
import torch.nn as nn


class GatingRegressor(nn.Module):
    """Ultra-lightweight student regressor predicting expected log-reconstruction error."""

    def __init__(self, input_dim: int = 38):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ELU(alpha=1.0),
            nn.Linear(16, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform initialization for ELU hidden layer and Xavier for output."""
        for idx, m in enumerate(self.net):
            if isinstance(m, nn.Linear):
                if idx == 0:
                    nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity="leaky_relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting scalar log-reconstruction error.
        
        Args:
            x: Input feature tensor of shape [B, 38]
            
        Returns:
            Tensor of shape [B, 1] containing predicted log-MSE e_hat.
        """
        return self.net(x)

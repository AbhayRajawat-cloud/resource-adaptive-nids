"""
Stage 5: Heavy Local Baseline Model Architectures.

Implements:
1. HeavyGRU: Two stacked GRUCell layers (input_dim=38, hidden_dim=64) with classification head
   (Linear(64, 32) -> LeakyReLU -> Dropout(0.2) -> Linear(32, 1)).
   Expected parameter count: 47,041.
2. SupervisedDenseMLP: Three-layer dense MLP with BatchNorm1d, LeakyReLU(0.1), Dropout(0.2)
   (Linear(38, 64) -> BN(64) -> LeakyReLU(0.1) -> Dropout(0.2) ->
    Linear(64, 32) -> BN(32) -> LeakyReLU(0.1) -> Dropout(0.2) ->
    Linear(32, 1)).
   Expected parameter count: 4,801.

Both models output raw logits.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeBatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm1d with single-sample training safety.
    When training mode is active and batch size is 1, evaluates using running statistics
    rather than raising ValueError, preventing crash on uneven batch boundaries.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        if self.training and input.size(0) == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,  # training=False: use running stats
                self.momentum,
                self.eps,
            )
        return super().forward(input)


class HeavyGRU(nn.Module):
    """
    Two-layer stacked GRUCell network for 38-dimensional flow classification.
    
    Architecture:
      - GRUCell 1: input_dim=38 -> hidden_dim=64 (19,968 params)
      - GRUCell 2: hidden_dim=64 -> hidden_dim=64 (24,960 params)
      - Classifier:
          Linear(64, 32) -> LeakyReLU() -> Dropout(0.2) -> Linear(32, 1) (2,113 params)
    Total parameters: 47,041.
    """

    def __init__(self, input_dim: int = 38, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.cell1 = nn.GRUCell(input_dim, hidden_dim)
        self.cell2 = nn.GRUCell(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim).
        Returns:
            Raw logits of shape (batch_size,).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Default W=1: (batch_size, 1, input_dim)

        batch_size, seq_len, _ = x.shape
        h1 = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        h2 = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            h1 = self.cell1(x[:, t, :], h1)
            h2 = self.cell2(h1, h2)

        logits = self.classifier(h2).squeeze(-1)
        return logits


class SupervisedDenseMLP(nn.Module):
    """
    Three-layer Dense Multi-Layer Perceptron baseline with Batch Normalization.
    
    Architecture:
      - Linear(38, 64) -> SafeBatchNorm1d(64) -> LeakyReLU(0.1) -> Dropout(0.2) (2,624 params)
      - Linear(64, 32) -> SafeBatchNorm1d(32) -> LeakyReLU(0.1) -> Dropout(0.2) (2,144 params)
      - Linear(32, 1) (33 params)
    Total parameters: 4,801.
    """

    def __init__(self, input_dim: int = 38, dropout_rate: float = 0.2, negative_slope: float = 0.1):
        super().__init__()
        self.input_dim = input_dim

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            SafeBatchNorm1d(64),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout_rate),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),
            SafeBatchNorm1d(32),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout_rate),
        )

        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, input_dim) or (batch_size, 1, input_dim).
        Returns:
            Raw logits of shape (batch_size,).
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        out = self.layer1(x)
        out = self.layer2(out)
        logits = self.head(out).squeeze(-1)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable/registered parameters in a module."""
    return sum(p.numel() for p in model.parameters())


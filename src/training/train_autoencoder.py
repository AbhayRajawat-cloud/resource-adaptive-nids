"""
Stage 3: Training routines for Local Flow Autoencoders.

Constraints & Specifications:
- Trains exclusively on local benign training data (x_train[y_train == 0]).
- Adam optimizer (lr=1e-3, weight_decay=1e-5, batch_size=64, max_epochs=40).
- ReduceLROnPlateau scheduler (factor=0.5, patience=3, mode='min').
- Early stopping (min_epochs=10, patience=5, monitoring benign validation MSE on x_val[y_val == 0]).
- Calibrate local anomaly threshold tau_AE_k = Quantile_0.99(MSE_val_benign).
- Compute 1,000-sample bootstrap 95% confidence intervals for tau_AE_k.
- Saves model weights to outputs/checkpoints/stage3/client_{k}_autoencoder.pt.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.autoencoder import Autoencoder


def compute_bootstrap_ci(
    data: np.ndarray,
    quantile: float = 0.99,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute empirical quantile and its 95% bootstrap confidence interval.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    point_est = float(np.percentile(data, quantile * 100))

    if n < 10:
        return point_est, point_est, point_est

    # Bootstrap resamples
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_samples = data[boot_indices]
    boot_quantiles = np.percentile(boot_samples, quantile * 100, axis=1)

    ci_lower = float(np.percentile(boot_quantiles, 2.5))
    ci_upper = float(np.percentile(boot_quantiles, 97.5))
    return point_est, ci_lower, ci_upper


def train_client_autoencoder(
    client_id: int,
    x_train_benign: np.ndarray,
    x_val_benign: np.ndarray,
    device: torch.device,
    config: Dict[str, Any] = None,
    save_dir: Path = None,
) -> Dict[str, Any]:
    """
    Train local Autoencoder exclusively on benign training data for a single client.
    """
    cfg = config or {}
    batch_size = cfg.get("batch_size", 64)
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 1e-5)
    max_epochs = cfg.get("max_epochs", 40)
    min_epochs = cfg.get("min_epochs", 10)
    patience = cfg.get("patience", 5)

    # Subsample if extremely large to prevent excessive wall-clock time while maintaining statistical power
    # 50,000 benign training samples is more than sufficient for a 2,008-parameter model.
    # We will use min(len(x_train_benign), 50000) for fast convergence.
    max_train_samples = cfg.get("max_train_samples", 50000)
    if len(x_train_benign) > max_train_samples:
        rng = np.random.default_rng(42 + client_id)
        sub_idx = rng.choice(len(x_train_benign), size=max_train_samples, replace=False)
        train_features = x_train_benign[sub_idx]
    else:
        train_features = x_train_benign

    # Prepare TensorDatasets and Loaders
    t_train = torch.tensor(train_features, dtype=torch.float32)
    t_val = torch.tensor(x_val_benign, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(t_train), batch_size=batch_size, shuffle=True, drop_last=False
    )
    # Val loader for evaluation
    val_loader = DataLoader(
        TensorDataset(t_val), batch_size=batch_size * 2, shuffle=False
    )

    # Instantiate model
    model = Autoencoder(input_dim=train_features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, max_epochs + 1):
        # Training phase
        model.train()
        running_train_loss = 0.0
        n_train_batches = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(batch_x)
            n_train_batches += len(batch_x)

        epoch_train_loss = running_train_loss / n_train_batches

        # Validation phase on benign validation flows
        model.eval()
        running_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                running_val_loss += loss.item() * len(batch_x)
                n_val_batches += len(batch_x)

        epoch_val_loss = running_val_loss / n_val_batches
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["lr"].append(current_lr)

        scheduler.step(epoch_val_loss)

        # Early stopping logic (monitors benign val MSE)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch >= min_epochs and patience_counter >= patience:
            break

    # Restore best weights
    model.load_state_dict(best_weights)
    model.eval()

    # Calibrate local anomaly threshold tau_AE_k = Quantile_0.99(MSE_val_benign)
    with torch.no_grad():
        val_mses = []
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            batch_mse = Autoencoder.compute_mse(batch_x, recon)
            val_mses.append(batch_mse.cpu().numpy())
        val_mse_arr = np.concatenate(val_mses)

    tau_ae, tau_ae_ci_low, tau_ae_ci_high = compute_bootstrap_ci(
        val_mse_arr, quantile=0.99, n_bootstrap=1000, seed=42 + client_id
    )

    # Save checkpoint
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"client_{client_id:02d}_autoencoder.pt"
        torch.save(best_weights, ckpt_path)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "tau_ae": tau_ae,
        "tau_ae_ci_low": tau_ae_ci_low,
        "tau_ae_ci_high": tau_ae_ci_high,
        "history": history,
        "n_val_benign": len(val_mse_arr),
    }

"""
Stage 5: Phase 5.2 Independent Local Standalone Training Engine.

Trains independent local HeavyGRU (W=1) and SupervisedDenseMLP models across all 10 clients
for both Q0.99 (baseline) and Q0.95 (candidate) threshold partitions.

Hyperparameters (from specification):
- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Batch size: 32 (SafeBatchSampler)
- Max epochs: 50
- Scheduler: ReduceLROnPlateau (mode='min', factor=0.5, patience=5, min_lr=1e-5)
- Early stopping patience: 10 epochs
- Seed: 42
- Loss: BCEWithLogitsLoss with client-specific positive class weight:
    w_pos = min(10.0, N_benign_train / N_attack_train)

Checkpoints:
- outputs/checkpoints/stage5/phase5.2/Q99/client_{k:02d}_local_gru.pt
- outputs/checkpoints/stage5/phase5.2/Q99/client_{k:02d}_local_mlp.pt
- outputs/checkpoints/stage5/phase5.2/Q95/client_{k:02d}_local_gru.pt
- outputs/checkpoints/stage5/phase5.2/Q95/client_{k:02d}_local_mlp.pt

Training Metadata:
- outputs/metrics/stage5/phase5.2_training_metadata.json
"""

import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.heavy_models import HeavyGRU, SupervisedDenseMLP
from src.data.phase52_dataset import (
    load_client_gate_passed_data,
    create_safe_dataloader,
    compute_positive_class_weight,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_local_model(
    client_id: int,
    quantile: str,
    model_type: str,  # 'gru' or 'mlp'
    data_dir: Path = Path("data/processed/gate_passed"),
    ckpt_dir: Path = Path("outputs/checkpoints/stage5/phase5.2"),
    batch_size: int = 32,
    max_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    patience_scheduler: int = 5,
    patience_early_stopping: int = 10,
    min_lr: float = 1e-5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a single independent local model on a client's Gate-passed partition.
    Saves the best validation-loss checkpoint.
    """
    set_seed(seed)
    as_seq = (model_type.lower() == "gru")

    train_ds, val_ds, test_ds, meta = load_client_gate_passed_data(
        client_id=client_id,
        quantile=quantile,
        base_dir=data_dir,
        as_sequence_w1=as_seq,
    )

    train_loader = create_safe_dataloader(train_ds, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = create_safe_dataloader(val_ds, batch_size=batch_size, shuffle=False)

    # Instantiate model
    if model_type.lower() == "gru":
        model = HeavyGRU(input_dim=38, hidden_dim=64, dropout_rate=0.2)
    elif model_type.lower() == "mlp":
        model = SupervisedDenseMLP(input_dim=38, dropout_rate=0.2, negative_slope=0.1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    w_pos = meta["w_pos"]
    pos_weight_tensor = torch.tensor([w_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=patience_scheduler,
        min_lr=min_lr,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        # 1. Training Phase
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            x_b = batch["x"]
            y_b = batch["y"]
            optimizer.zero_grad()

            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            bs = len(y_b)
            running_train_loss += loss.item() * bs
            train_samples += bs

        epoch_train_loss = running_train_loss / train_samples if train_samples > 0 else 0.0

        # 2. Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                x_b = batch["x"]
                y_b = batch["y"]
                logits = model(x_b)
                loss = criterion(logits, y_b)

                bs = len(y_b)
                running_val_loss += loss.item() * bs
                val_samples += bs

        epoch_val_loss = running_val_loss / val_samples if val_samples > 0 else 0.0
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(float(epoch_train_loss))
        history["val_loss"].append(float(epoch_val_loss))
        history["lr"].append(current_lr)

        scheduler.step(epoch_val_loss)

        # 3. Model Checkpoint & Early Stopping Check
        if epoch_val_loss < best_val_loss - 1e-5:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_early_stopping:
            break

    total_training_time_s = time.perf_counter() - t0

    # Ensure best model state is saved
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save checkpoint to disk
    out_dir = ckpt_dir / quantile
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_filename = f"client_{client_id:02d}_local_{model_type.lower()}.pt"
    ckpt_path = out_dir / ckpt_filename
    torch.save(model.state_dict(), ckpt_path)

    run_record = {
        "client_id": client_id,
        "client_str": meta["client_str"],
        "quantile": quantile,
        "model_type": model_type.upper(),
        "checkpoint_path": str(ckpt_path),
        "hyperparameters": {
            "optimizer": "Adam",
            "initial_lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "scheduler_patience": patience_scheduler,
            "early_stopping_patience": patience_early_stopping,
            "min_lr": min_lr,
            "seed": seed,
        },
        "dataset_statistics": {
            "n_train_passed": meta["n_train_passed"],
            "n_val_passed": meta["n_val_passed"],
            "n_test_passed": meta["n_test_passed"],
            "benign_count_train_passed": meta["benign_count_train_passed"],
            "attack_count_train_passed": meta["attack_count_train_passed"],
            "w_pos": meta["w_pos"],
            "w_pos_fallback_used": meta["w_pos_fallback_used"],
        },
        "training_outcome": {
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "final_epoch": epoch,
            "stopped_early": bool(patience_counter >= patience_early_stopping),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "final_lr": float(history["lr"][-1]),
            "training_time_seconds": float(total_training_time_s),
        },
        "loss_history": history,
    }

    return run_record


def run_all_local_training() -> Dict[str, Any]:
    """Execute all 40 local standalone training runs across 10 clients, Q99/Q95, GRU/MLP."""
    print("=" * 70)
    print("Starting Stage 5 Phase 5.2: Independent Local Standalone Training")
    print("=" * 70)

    quantiles = ["Q99", "Q95"]
    model_types = ["gru", "mlp"]
    all_runs: List[Dict[str, Any]] = []

    total_runs = 10 * len(quantiles) * len(model_types)
    run_idx = 0

    for q in quantiles:
        for m_type in model_types:
            print(f"\n--- Training Phase: Quantile={q}, Model={m_type.upper()} ---")
            for cid in range(10):
                run_idx += 1
                rec = train_single_local_model(
                    client_id=cid,
                    quantile=q,
                    model_type=m_type,
                    seed=42,
                )
                all_runs.append(rec)
                out = rec["training_outcome"]
                ds = rec["dataset_statistics"]
                print(
                    f"  [{run_idx:02d}/{total_runs}] Client {cid:02d} ({q}, {m_type.upper()}): "
                    f"Train N={ds['n_train_passed']:,} (Benign={ds['benign_count_train_passed']:,}, Atk={ds['attack_count_train_passed']:,}, w_pos={ds['w_pos']:.3f}) | "
                    f"Best Epoch {out['best_epoch']}/{out['final_epoch']} | Best Val Loss: {out['best_val_loss']:.4f} | "
                    f"Time: {out['training_time_seconds']:.2f}s"
                )

    metrics_dir = Path("outputs/metrics/stage5")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metrics_dir / "phase5.2_training_metadata.json"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[OK] Successfully completed all {total_runs} local standalone training runs.")
    print(f"[OK] Training metadata saved to: {meta_path}")
    print("=" * 70)

    return {"total_runs": total_runs, "runs": all_runs}


if __name__ == "__main__":
    run_all_local_training()


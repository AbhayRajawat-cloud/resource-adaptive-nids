"""
Stage 5: Phase 5.3 Federated Training Engine (FedAvg, FedBN, FedPer).

Orchestrates R=50 communication rounds across K=10 clients for:
1. global_fedavg_gru_Q99: Federated HeavyGRU on Q0.99 (Primary Baseline)
2. global_fedavg_mlp_Q99: Federated DenseMLP on Q0.99 (Primary Baseline, FedBN)
3. global_fedavg_gru_Q95: Federated HeavyGRU on Q0.95 (Candidate Operating Point)
4. global_fedavg_mlp_Q95: Federated DenseMLP on Q0.95 (Candidate Operating Point, FedBN)
5. fedper_gru_Q99: 10-client Strategy B (FedPer) GRU run on Q0.99
6. fedper_mlp_Q99: 10-client Strategy B (FedPer) MLP run on Q0.99

Key Architectural & Protocol Features:
- Strict artifact immutability (Stages 1-4, Phase 5.1, Phase 5.2).
- Targeted Heavy aggregation only: AE and Gate models remain strictly local.
- Cosine Annealing Learning Rate across R=50 rounds:
    eta^{(r)} = eta_min + 0.5 * (eta_0 - eta_min) * (1 + cos(pi * r / R))
- Local Adam optimizer reinitialized at each round with fresh momentum states.
- Local BCEWithLogitsLoss with client-specific positive class weighting w_pos_k.
- Batch size: 64 using SafeBatchSampler.
- FedBN: MLP clients preserve BatchNorm running statistics locally across rounds.
- FedPer: Private classification head (33 parameters) initialized at round 0,
  updated locally, and strictly excluded from server aggregation.
- Full communication traffic accounting in binary units (KiB, MiB).
"""

import copy
import json
import math
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.heavy_models import HeavyGRU, SupervisedDenseMLP, SafeBatchNorm1d
from src.data.phase52_dataset import (
    load_client_gate_passed_data,
    create_safe_dataloader,
    compute_positive_class_weight,
)
from src.federated.fedavg_aggregator import (
    aggregate_weighted_fedavg,
    is_bn_buffer,
    is_private_head,
)


def compute_cosine_lr(
    r: int,
    total_rounds: int = 50,
    lr_0: float = 1e-3,
    lr_min: float = 1e-5,
) -> float:
    """
    Evaluate global Cosine Annealing learning rate schedule at round r:
      eta^{(r)} = eta_min + 0.5 * (eta_0 - eta_min) * (1 + cos(pi * r / R))
    """
    return lr_min + 0.5 * (lr_0 - lr_min) * (1.0 + math.cos(math.pi * float(r) / float(total_rounds)))


def set_seed(seed: int = 42):
    """Set random seeds for PyTorch and NumPy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_traffic_stats(
    num_shared_params: int,
    num_clients: int = 10,
    total_rounds: int = 50,
) -> Dict[str, Any]:
    """
    Calculate exact communication traffic in binary units (KiB, MiB).
    Assumes 32-bit float encoding (4 bytes / parameter).
    """
    bytes_per_param = 4
    bytes_per_model = num_shared_params * bytes_per_param
    kib_per_model = bytes_per_model / 1024.0
    mib_per_model = bytes_per_model / (1024.0 * 1024.0)

    # Per round
    upload_per_client_kib = kib_per_model
    broadcast_per_client_kib = kib_per_model
    round_upload_total_kib = kib_per_model * num_clients
    round_broadcast_total_kib = kib_per_model * num_clients
    round_bidirectional_total_kib = round_upload_total_kib + round_broadcast_total_kib
    round_bidirectional_total_mib = round_bidirectional_total_kib / 1024.0

    # 50-round cumulative
    cumulative_upload_kib = round_upload_total_kib * total_rounds
    cumulative_broadcast_kib = round_broadcast_total_kib * total_rounds
    cumulative_bidirectional_kib = round_bidirectional_total_kib * total_rounds
    cumulative_bidirectional_mib = cumulative_bidirectional_kib / 1024.0

    return {
        "num_shared_params": num_shared_params,
        "bytes_per_param": bytes_per_param,
        "model_bytes": bytes_per_model,
        "model_kib": kib_per_model,
        "model_mib": mib_per_model,
        "per_client_upload_kib": upload_per_client_kib,
        "per_client_broadcast_kib": broadcast_per_client_kib,
        "round_upload_total_kib": round_upload_total_kib,
        "round_broadcast_total_kib": round_broadcast_total_kib,
        "round_bidirectional_total_kib": round_bidirectional_total_kib,
        "round_bidirectional_total_mib": round_bidirectional_total_mib,
        "cumulative_upload_kib": cumulative_upload_kib,
        "cumulative_broadcast_kib": cumulative_broadcast_kib,
        "cumulative_bidirectional_kib": cumulative_bidirectional_kib,
        "cumulative_bidirectional_mib": cumulative_bidirectional_mib,
    }


def train_federated_run(
    run_name: str,
    model_type: str,  # 'gru' or 'mlp'
    quantile: str,    # 'Q99' or 'Q95'
    is_fedper: bool = False,
    num_rounds: int = 50,
    local_epochs: int = 5,
    batch_size: int = 64,
    lr_0: float = 1e-3,
    lr_min: float = 1e-5,
    weight_decay: float = 1e-4,
    seed: int = 42,
    data_dir: Path = Path("data/processed/gate_passed"),
    ckpt_dir: Path = Path("outputs/checkpoints/stage5/phase5.3"),
) -> Dict[str, Any]:
    """
    Execute a full R=50 communication round federated training run.
    """
    print(f"\n{'='*75}")
    print(f"Starting Federated Run: {run_name}")
    print(f"Model: {model_type.upper()} | Quantile: {quantile} | Strategy: {'FedPer (Strategy B)' if is_fedper else 'Standard FedAvg'}")
    print(f"Rounds: {num_rounds} | Local Epochs: {local_epochs} | Batch Size: {batch_size}")
    print(f"{'='*75}")

    set_seed(seed)
    is_mlp = (model_type.lower() == "mlp")
    as_seq = (model_type.lower() == "gru")
    num_clients = 10

    # 1. Global Model Initialization (seed 42)
    if is_mlp:
        global_model = SupervisedDenseMLP(input_dim=38, dropout_rate=0.2, negative_slope=0.1)
    else:
        global_model = HeavyGRU(input_dim=38, hidden_dim=64, dropout_rate=0.2)

    initial_global_state = copy.deepcopy(global_model.state_dict())

    # Count shared parameters
    if is_fedper:
        shared_param_keys = [k for k in initial_global_state.keys() if not is_private_head(k) and not (is_mlp and is_bn_buffer(k))]
    elif is_mlp:
        shared_param_keys = [k for k in initial_global_state.keys() if not is_bn_buffer(k)]
    else:
        shared_param_keys = list(initial_global_state.keys())

    num_shared_params = sum(initial_global_state[k].numel() for k in shared_param_keys)
    traffic_stats = calculate_traffic_stats(num_shared_params, num_clients=num_clients, total_rounds=num_rounds)

    print(f"Shared parameters per communication: {num_shared_params:,} ({traffic_stats['model_kib']:.2f} KiB)")
    print(f"Total 50-round bidirectional traffic: {traffic_stats['cumulative_bidirectional_mib']:.2f} MiB")

    # 2. Client Persistent Local States
    # Preload client datasets and calculate weights
    client_loaders = []
    client_val_loaders = []
    client_sample_counts = []
    client_w_pos = []
    client_local_bn_buffers = []  # For MLP (FedBN)
    client_private_heads = []     # For FedPer (Strategy B)

    for k in range(num_clients):
        tr_ds, val_ds, te_ds, meta = load_client_gate_passed_data(
            client_id=k,
            quantile=quantile,
            base_dir=data_dir,
            as_sequence_w1=as_seq,
        )
        tr_loader = create_safe_dataloader(tr_ds, batch_size=batch_size, shuffle=True, seed=seed + k)
        val_loader = create_safe_dataloader(val_ds, batch_size=batch_size, shuffle=False)

        client_loaders.append(tr_loader)
        client_val_loaders.append(val_loader)
        client_sample_counts.append(len(tr_ds))
        client_w_pos.append(meta["w_pos"])

        # Local BatchNorm buffer persistence for MLP
        if is_mlp:
            local_bn = {k_bn: v.clone() for k_bn, v in initial_global_state.items() if is_bn_buffer(k_bn)}
            client_local_bn_buffers.append(local_bn)
        else:
            client_local_bn_buffers.append(None)

        # Private classification head persistence for FedPer
        if is_fedper:
            # Client k initializes its own classification head once at round 0
            if is_mlp:
                c_head = {k_h: initial_global_state[k_h].clone() for k_h in initial_global_state if is_private_head(k_h)}
            else:
                c_head = {k_h: initial_global_state[k_h].clone() for k_h in initial_global_state if is_private_head(k_h)}
            client_private_heads.append(c_head)
        else:
            client_private_heads.append(None)

    total_training_samples = sum(client_sample_counts)
    client_weights = [n / total_training_samples for n in client_sample_counts]

    # Shared parameters maintained on server
    current_shared_state = {k: v.clone() for k, v in initial_global_state.items() if k in shared_param_keys}

    round_history = []
    t_start = time.perf_counter()

    # 3. Communication Rounds Loop (r = 0 .. num_rounds - 1)
    for r in range(num_rounds):
        eta_r = compute_cosine_lr(r, total_rounds=num_rounds, lr_0=lr_0, lr_min=lr_min)
        client_updated_states = []
        round_train_losses = []
        round_val_losses = []

        # Local Client Training Phase
        for k in range(num_clients):
            # Instantiate local client model
            if is_mlp:
                local_model = SupervisedDenseMLP(input_dim=38, dropout_rate=0.2, negative_slope=0.1)
            else:
                local_model = HeavyGRU(input_dim=38, hidden_dim=64, dropout_rate=0.2)

            # Assemble client model state:
            # (a) Load current global shared parameters
            local_full_state = local_model.state_dict()
            for key, val in current_shared_state.items():
                local_full_state[key] = val.clone()

            # (b) Load local persistent BatchNorm buffers (MLP FedBN)
            if is_mlp and client_local_bn_buffers[k] is not None:
                for b_key, b_val in client_local_bn_buffers[k].items():
                    local_full_state[b_key] = b_val.clone()

            # (c) Load local persistent private head (FedPer)
            if is_fedper and client_private_heads[k] is not None:
                for h_key, h_val in client_private_heads[k].items():
                    local_full_state[h_key] = h_val.clone()

            local_model.load_state_dict(local_full_state)

            # Re-initialize fresh Adam optimizer with learning rate eta^{(r)}
            optimizer = torch.optim.Adam(
                local_model.parameters(),
                lr=eta_r,
                weight_decay=weight_decay,
            )

            # Local loss with class weighting
            pos_weight_t = torch.tensor([client_w_pos[k]], dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

            # Train for E local epochs
            local_model.train()
            running_tr_loss = 0.0
            n_tr_samples = 0

            for epoch in range(local_epochs):
                for batch in client_loaders[k]:
                    x_b = batch["x"]
                    y_b = batch["y"]
                    optimizer.zero_grad()

                    logits = local_model(x_b)
                    loss = criterion(logits, y_b)
                    loss.backward()
                    optimizer.step()

                    bs = len(y_b)
                    running_tr_loss += loss.item() * bs
                    n_tr_samples += bs

            c_train_loss = running_tr_loss / n_tr_samples if n_tr_samples > 0 else 0.0
            round_train_losses.append(c_train_loss)

            # Evaluate local validation loss
            local_model.eval()
            running_val_loss = 0.0
            n_val_samples = 0
            with torch.no_grad():
                for batch in client_val_loaders[k]:
                    x_b = batch["x"]
                    y_b = batch["y"]
                    logits = local_model(x_b)
                    loss = criterion(logits, y_b)
                    bs = len(y_b)
                    running_val_loss += loss.item() * bs
                    n_val_samples += bs

            c_val_loss = running_val_loss / n_val_samples if n_val_samples > 0 else 0.0
            round_val_losses.append(c_val_loss)

            # In-place update of local persistent states:
            new_local_state = local_model.state_dict()

            # (1) Update persistent local BatchNorm buffers (MLP)
            if is_mlp:
                client_local_bn_buffers[k] = {
                    b_k: new_local_state[b_k].clone() for b_k in new_local_state if is_bn_buffer(b_k)
                }

            # (2) Update persistent local private classification head (FedPer)
            if is_fedper:
                client_private_heads[k] = {
                    h_k: new_local_state[h_k].clone() for h_k in new_local_state if is_private_head(h_k)
                }

            # (3) Collect updated shared parameters to upload to server
            client_updated_states.append(new_local_state)

        # Server Aggregation Phase
        current_shared_state = aggregate_weighted_fedavg(
            client_states=client_updated_states,
            client_sample_counts=client_sample_counts,
            is_mlp=is_mlp,
            is_fedper=is_fedper,
        )

        # Global weighted loss logging
        global_tr_loss = sum(w * l for w, l in zip(client_weights, round_train_losses))
        global_val_loss = sum(w * l for w, l in zip(client_weights, round_val_losses))

        round_record = {
            "round": r,
            "lr": float(eta_r),
            "global_train_loss": float(global_tr_loss),
            "global_val_loss": float(global_val_loss),
            "client_train_losses": [float(l) for l in round_train_losses],
            "client_val_losses": [float(l) for l in round_val_losses],
        }
        round_history.append(round_record)

        if (r + 1) % 10 == 0 or r == 0 or r == num_rounds - 1:
            print(
                f"  Round [{r+1:02d}/{num_rounds}] | LR: {eta_r:.6e} | "
                f"Global Train Loss: {global_tr_loss:.4f} | Global Val Loss: {global_val_loss:.4f}"
            )

    total_training_time = time.perf_counter() - t_start
    print(f"Completed {num_rounds} communication rounds in {total_training_time:.2f}s.")

    # 4. Save Final R=50 Checkpoints
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    saved_checkpoints = {}

    if not is_fedper:
        # Standard FedAvg Checkpoint
        ckpt_filename = f"{run_name}_R50.pt"
        ckpt_path = ckpt_dir / ckpt_filename

        # If MLP: load shared state into a model and also save the client-specific BN buffers companion
        final_model_state = copy.deepcopy(current_shared_state)
        torch.save(final_model_state, ckpt_path)
        saved_checkpoints["global_checkpoint"] = str(ckpt_path)

        if is_mlp:
            bn_companion = ckpt_dir / f"{run_name}_R50_client_bn.pt"
            bn_dict = {f"client_{k:02d}": client_local_bn_buffers[k] for k in range(num_clients)}
            torch.save(bn_dict, bn_companion)
            saved_checkpoints["client_bn_buffers"] = str(bn_companion)
    else:
        # FedPer Checkpoint: Global body + 10 client private heads
        body_filename = f"{run_name}_R50_body.pt"
        body_path = ckpt_dir / body_filename
        torch.save(current_shared_state, body_path)
        saved_checkpoints["global_body"] = str(body_path)

        head_filename = f"client00_09_{run_name}_heads.pt"
        head_path = ckpt_dir / head_filename
        heads_dict = {
            f"client_{k:02d}": {
                "private_head": client_private_heads[k],
                "bn_buffers": client_local_bn_buffers[k] if is_mlp else None,
            }
            for k in range(num_clients)
        }
        torch.save(heads_dict, head_path)
        saved_checkpoints["client_heads"] = str(head_path)

    run_summary = {
        "run_name": run_name,
        "model_type": model_type.upper(),
        "quantile": quantile,
        "is_fedper": is_fedper,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "lr_0": lr_0,
        "lr_min": lr_min,
        "weight_decay": weight_decay,
        "seed": seed,
        "total_training_time_seconds": float(total_training_time),
        "final_global_train_loss": float(round_history[-1]["global_train_loss"]),
        "final_global_val_loss": float(round_history[-1]["global_val_loss"]),
        "traffic_accounting": traffic_stats,
        "saved_checkpoints": saved_checkpoints,
        "round_history": round_history,
    }

    return run_summary


def execute_all_federated_runs() -> Dict[str, Any]:
    """Execute all 6 federated training runs required in Phase 5.3."""
    metrics_dir = Path("outputs/metrics/stage5")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    matrix = [
        ("global_fedavg_gru_Q99", "gru", "Q99", False),
        ("global_fedavg_mlp_Q99", "mlp", "Q99", False),
        ("global_fedavg_gru_Q95", "gru", "Q95", False),
        ("global_fedavg_mlp_Q95", "mlp", "Q95", False),
        ("fedper_gru_Q99", "gru", "Q99", True),
        ("fedper_mlp_Q99", "mlp", "Q99", True),
    ]

    all_metadata = []
    for run_name, m_type, q, fedper in matrix:
        res = train_federated_run(
            run_name=run_name,
            model_type=m_type,
            quantile=q,
            is_fedper=fedper,
            num_rounds=50,
            local_epochs=5,
            batch_size=64,
            lr_0=1e-3,
            lr_min=1e-5,
            weight_decay=1e-4,
            seed=42,
        )
        all_metadata.append(res)

    metadata_path = metrics_dir / "phase5.3_federated_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n[OK] All 6 federated training runs completed successfully.")
    print(f"[OK] Training metadata saved to: {metadata_path}")
    return {"runs": all_metadata}


if __name__ == "__main__":
    execute_all_federated_runs()


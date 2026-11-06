"""
Stage 5: Federated Training Convergence Curves Visualization.

Parses outputs/metrics/stage5/phase5.3_federated_metadata.json to extract
50-round global training and validation loss trajectories across all 6 federated runs:
- global_fedavg_gru_Q99
- global_fedavg_mlp_Q99
- fedper_gru_Q99
- fedper_mlp_Q99
- global_fedavg_gru_Q95
- global_fedavg_mlp_Q95

Generates a 300 DPI publication figure:
- outputs/plots/stage5/federated_convergence_curves.png
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np


def plot_federated_convergence():
    """Extract round-level loss histories and render 300 DPI convergence curves."""
    print("=" * 75)
    print("Generating Federated Convergence Curves (300 DPI)")
    print("=" * 75)

    meta_path = Path("outputs/metrics/stage5/phase5.3_federated_metadata.json")
    if not meta_path.exists():
        print("Loss progression data not preserved in metadata JSON: file not found.")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    # Run map: run_name -> dict with train_loss, val_loss
    run_histories = {}
    for entry in meta_list:
        r_name = entry.get("run_name")
        rh = entry.get("round_history", [])
        if rh:
            rounds = [r["round"] + 1 for r in rh]
            tr_losses = [r["global_train_loss"] for r in rh]
            va_losses = [r["global_val_loss"] for r in rh]
            run_histories[r_name] = {
                "rounds": rounds,
                "train_loss": tr_losses,
                "val_loss": va_losses,
                "model_type": entry.get("model_type"),
                "quantile": entry.get("quantile"),
                "is_fedper": entry.get("is_fedper", False),
            }

    if not run_histories:
        print("Loss progression data not preserved in metadata JSON: empty round history.")
        return

    output_dir = Path("outputs/plots/stage5")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "federated_convergence_curves.png"

    # Style setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # -----------------------------------------------------------------------
    # Subplot 1: Q0.99 Baseline Operating Point (FedAvg & FedPer)
    # -----------------------------------------------------------------------
    ax1 = axes[0]
    color_map = {
        "global_fedavg_gru_Q99": ("#1f77b4", "FedAvg HeavyGRU"),
        "global_fedavg_mlp_Q99": ("#ff7f0e", "FedAvg DenseMLP"),
        "fedper_gru_Q99": ("#2ca02c", "FedPer HeavyGRU"),
        "fedper_mlp_Q99": ("#d62728", "FedPer DenseMLP (Strategy B)"),
    }

    for r_name, (col, label) in color_map.items():
        if r_name in run_histories:
            d = run_histories[r_name]
            rounds = d["rounds"]
            ax1.plot(rounds, d["val_loss"], color=col, linewidth=1.8, label=f"{label} (Val)")
            ax1.plot(rounds, d["train_loss"], color=col, linewidth=1.0, linestyle="--", alpha=0.6)

    ax1.set_title("A. Global Convergence ($Q_{0.99}$ Operating Baseline)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Federated Communication Round ($r$)", fontsize=11)
    ax1.set_ylabel("Global Binary Cross-Entropy Loss", fontsize=11)
    ax1.set_xlim(1, 50)
    ax1.set_ylim(0.10, 0.65)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # -----------------------------------------------------------------------
    # Subplot 2: Q0.95 Candidate Operating Point
    # -----------------------------------------------------------------------
    ax2 = axes[1]
    color_map_q95 = {
        "global_fedavg_gru_Q95": ("#1f77b4", "FedAvg HeavyGRU"),
        "global_fedavg_mlp_Q95": ("#ff7f0e", "FedAvg DenseMLP"),
    }

    for r_name, (col, label) in color_map_q95.items():
        if r_name in run_histories:
            d = run_histories[r_name]
            rounds = d["rounds"]
            ax2.plot(rounds, d["val_loss"], color=col, linewidth=1.8, label=f"{label} (Val)")
            ax2.plot(rounds, d["train_loss"], color=col, linewidth=1.0, linestyle="--", alpha=0.6)

    ax2.set_title("B. Global Convergence ($Q_{0.95}$ Candidate Point)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Federated Communication Round ($r$)", fontsize=11)
    ax2.set_ylabel("Global Binary Cross-Entropy Loss", fontsize=11)
    ax2.set_xlim(1, 50)
    ax2.set_ylim(0.05, 0.45)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    # Add shared footnote
    fig.text(
        0.5, 0.01,
        "Solid curves indicate Global Validation Loss; dashed curves indicate Global Training Loss across 10 non-IID clients (E=5 local epochs, B=64, Cosine LR).",
        ha="center", fontsize=9, style="italic"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Exported 300 DPI convergence curves to {out_png}")


if __name__ == "__main__":
    plot_federated_convergence()


"""
Stage 5: Phase 5.4 Publication-Grade Visualizations.

Generates 300 DPI publication figures:
1. outputs/plots/stage5/e2e_recall_vs_load.png:
   Offered load rho in [0.1, 3.0] vs End-to-End Recall and Load-Shedding Ratio.
2. outputs/plots/stage5/e2e_routing_breakdown.png:
   10-client stacked bar chart of the 4 terminal routing outcomes under Q0.99 and Q0.95.
3. outputs/plots/stage5/config_comparison_bar.png:
   Macro performance bar chart comparing Config A (HeavyGRU FedAvg), Config B (DenseMLP FedPer),
   Config C (DenseMLP FedAvg), and Config D (Light AE Standalone) across Tiers 1-3.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style():
    """Configure publication plot styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })


def plot_e2e_recall_vs_load(df: pd.DataFrame, out_path: Path):
    """
    Figure 1: Offered Load rho in [0.1, 3.0] vs End-to-End Recall and Load-Shedding Ratio.
    """
    sub = df[df["scenario_id"] == "S2_Load_Sweep"]
    if len(sub) == 0:
        print("[WARN] S2_Load_Sweep data missing, skipping plot_e2e_recall_vs_load.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    configs_to_plot = [
        ("Config A", "Config A: HeavyGRU FedAvg", "#1f77b4", "-"),
        ("Config B", "Config B: DenseMLP FedPer", "#2ca02c", "--"),
        ("Config C", "Config C: DenseMLP FedAvg", "#ff7f0e", ":"),
        ("Config D", "Config D: Light AE Standalone", "#d62728", "-."),
    ]

    # Subplot 1: End-to-End Recall vs Offered Load
    for cfg_short, cfg_label, color, ls in configs_to_plot:
        s_cfg = sub[sub["config_short"] == cfg_short]
        if len(s_cfg) == 0:
            continue
        agg = s_cfg.groupby("offered_load_rho").agg({
            "recall_e2e": "mean",
            "r_load_shed_ae": "mean",
        }).reset_index().sort_values("offered_load_rho")

        ax1.plot(
            agg["offered_load_rho"],
            agg["recall_e2e"],
            label=cfg_short,
            color=color,
            linestyle=ls,
            marker="o",
            markersize=4,
            linewidth=1.8,
        )

        # Subplot 2: Load Shedding Ratio vs Offered Load
        ax2.plot(
            agg["offered_load_rho"],
            agg["r_load_shed_ae"] * 100.0,
            label=cfg_short,
            color=color,
            linestyle=ls,
            marker="s",
            markersize=4,
            linewidth=1.8,
        )

    # Reference capacity line rho = 1.0
    ax1.axvline(x=1.0, color="#888888", linestyle=":", linewidth=1.2, label="Server Capacity (rho=1.0)")
    ax2.axvline(x=1.0, color="#888888", linestyle=":", linewidth=1.2, label="Server Capacity (rho=1.0)")

    ax1.set_title("(a) End-to-End Recall vs. Offered Load", fontweight="bold")
    ax1.set_xlabel("Offered Load (rho = lambda / mu_heavy)")
    ax1.set_ylabel("End-to-End Recall (Recall_e2e)")
    ax1.set_ylim(0.2, 0.45)
    ax1.grid(True)
    ax1.legend(loc="lower left", framealpha=0.9)

    ax2.set_title("(b) Pre-Admission Load-Shedding Ratio vs. Offered Load", fontweight="bold")
    ax2.set_xlabel("Offered Load (rho = lambda / mu_heavy)")
    ax2.set_ylabel("Load-Shedding Ratio (%)")
    ax2.grid(True)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Generated: {out_path}")


def plot_e2e_routing_breakdown(df: pd.DataFrame, out_path: Path):
    """
    Figure 2: 10-Client Stacked Bar Chart of 4 Terminal Routing Outcomes (Q99 vs Q95).
    """
    sub = df[(df["scenario_id"] == "S1_Nominal") & (df["config_short"] == "Config A") & (df["tier"] == 1)]
    if len(sub) == 0:
        print("[WARN] S1_Nominal data missing, skipping plot_e2e_routing_breakdown.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    clients = [f"C{i:02d}" for i in range(10)]
    colors = ["#d62728", "#aec7e8", "#ffbb78", "#1f77b4"]
    labels = ["Front-end Drop", "Gate Reject (AE)", "Load Shed (AE)", "Heavy Admitted"]

    for ax, q_val, title in [(ax1, "Q99", "(a) Baseline Q0.99 Operating Point"), (ax2, "Q95", "(b) Candidate Q0.95 Operating Point")]:
        sub_q = sub[sub["quantile"] == q_val].sort_values("client_id")
        if len(sub_q) == 0:
            continue

        fe = sub_q["r_frontend_drop"].to_numpy() * 100.0
        gr = sub_q["r_gate_reject_ae"].to_numpy() * 100.0
        ls = sub_q["r_load_shed_ae"].to_numpy() * 100.0
        ha = sub_q["r_heavy_admitted"].to_numpy() * 100.0

        ind = np.arange(10)
        width = 0.65

        b1 = ax.bar(ind, fe, width, color=colors[0], label=labels[0])
        b2 = ax.bar(ind, gr, width, bottom=fe, color=colors[1], label=labels[1])
        b3 = ax.bar(ind, ls, width, bottom=fe+gr, color=colors[2], label=labels[2])
        b4 = ax.bar(ind, ha, width, bottom=fe+gr+ls, color=colors[3], label=labels[3])

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Federated Client")
        ax.set_ylabel("Routing Proportion (%)")
        ax.set_xticks(ind)
        ax.set_xticklabels(clients)
        ax.set_ylim(0, 105)
        ax.axhline(y=100.0, color="#333333", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.grid(axis="y")
        if q_val == "Q99":
            ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Generated: {out_path}")


def plot_config_comparison_bar(df: pd.DataFrame, out_path: Path):
    """
    Figure 3: Macro performance bar chart comparing Configs A, B, C, D across Tiers 1-3.
    """
    sub = df[(df["scenario_id"] == "S1_Nominal") & (df["quantile"] == "Q99")]
    if len(sub) == 0:
        print("[WARN] S1_Nominal data missing, skipping plot_config_comparison_bar.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)

    cfg_names = ["Config A", "Config B", "Config C", "Config D"]
    cfg_labels = ["A: GRU FedAvg", "B: MLP FedPer", "C: MLP FedAvg", "D: Light AE"]
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    metrics = [
        ("recall_e2e", "Recall", (0.2, 0.40)),
        ("precision", "Precision", (0.35, 0.75)),
        ("f1_score", "F1-Score", (0.2, 0.48)),
    ]

    for ax, (m_col, m_name, y_lim) in zip(axes, metrics):
        ind = np.arange(3)  # Tiers 1, 2, 3
        bar_w = 0.18

        for i, (cfg_short, label, col) in enumerate(zip(cfg_names, cfg_labels, palette)):
            vals = []
            for t in [1, 2, 3]:
                s = sub[(sub["config_short"] == cfg_short) & (sub["tier"] == t)]
                vals.append(float(s[m_col].mean()) if len(s) > 0 else 0.0)

            ax.bar(ind + i * bar_w, vals, bar_w, label=label, color=col, edgecolor="#333333", linewidth=0.6)

        ax.set_title(f"Macro {m_name} across Lineage Tiers", fontweight="bold")
        ax.set_xticks(ind + 1.5 * bar_w)
        ax.set_xticklabels(["Tier 1 (All)", "Tier 2 (Unique)", "Tier 3 (Exclusive)"])
        ax.set_ylabel(m_name)
        ax.set_ylim(y_lim)
        ax.grid(axis="y")
        if m_col == "f1_score":
            ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Generated: {out_path}")


def generate_all_plots():
    """Load results and generate all publication plots."""
    set_plot_style()
    csv_path = Path("outputs/metrics/stage5/phase5.4_e2e_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing Phase 5.4 results CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    out_dir = Path("outputs/plots/stage5")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_e2e_recall_vs_load(df, out_dir / "e2e_recall_vs_load.png")
    plot_e2e_routing_breakdown(df, out_dir / "e2e_routing_breakdown.png")
    plot_config_comparison_bar(df, out_dir / "config_comparison_bar.png")
    print(f"[OK] All 3 Stage 5 publication plots successfully generated in {out_dir}")


if __name__ == "__main__":
    generate_all_plots()


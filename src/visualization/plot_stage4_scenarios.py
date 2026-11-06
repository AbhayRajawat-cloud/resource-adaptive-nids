"""
Stage 4: Publication Visualization Script.

Generates 300 DPI publication-quality figures using matplotlib/seaborn:
1. outputs/plots/stage4/S2_load_sweep_delay_and_shedding.png:
   Dual-axis plot of offered load rho in [0.1, 3.0] vs. Mean Queue Delay W_q (ms) and Load-Shedding Ratio R_shed (%).
2. outputs/plots/stage4/S5_sla_delay_sweep.png:
   Delay threshold W_shed in [1.0, 100.0] ms vs. R_shed (%) and Effective Attack Recall TPR_Heavy^eff.
3. outputs/plots/stage4/S6_Q99_vs_Q95_crossover.png:
   Empirical Scenario S6 comparison of Q99 vs Q95 over rho in [0.1, 3.0] with visual annotation
   of theoretical critical stability limits (lambda_crit_tilde^(Q95) ~ 15.24, lambda_crit_tilde^(Q99) ~ 80.65).
4. outputs/plots/stage4/S7_client_heterogeneity_queue.png:
   Comparative analysis of Client 02 (threshold-limited), Client 04 (separability-limited),
   and standard client bundle under bursty non-IID traffic.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set overall publication aesthetics
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})


def plot_s2_load_sweep(df_s2: pd.DataFrame, output_dir: Path):
    """Plot S2: Offered Load Sweep vs Delay & Load-Shedding Ratio."""
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0), dpi=300)

    # Aggregate by offered load step
    agg = df_s2.groupby("target_rho").agg({
        "mean_waiting_time_ms": ["mean", "std"],
        "p95_waiting_time_ms": "mean",
        "load_shed_ae_pct": "mean",
        "heavy_admitted_pct": "mean",
    }).reset_index()

    rhos = agg["target_rho"].to_numpy()
    mean_delay = agg["mean_waiting_time_ms"]["mean"].to_numpy()
    std_delay = agg["mean_waiting_time_ms"]["std"].fillna(0).to_numpy()
    p95_delay = agg["p95_waiting_time_ms"].to_numpy()
    load_shed = agg["load_shed_ae_pct"].to_numpy()

    # Left axis: Queue Delay (ms)
    color1 = "#1f77b4"
    ax1.set_xlabel(r"Offered Load to Edge Router $\rho_{\mathrm{offered}} = \lambda_{\mathrm{arr}} / \mu_{\mathrm{heavy}}$", fontweight="bold")
    ax1.set_ylabel(r"Queue Delay $\bar{W}_q$ [ms]", color=color1, fontweight="bold")
    line1 = ax1.plot(rhos, mean_delay, color=color1, marker="o", label=r"Mean Delay $\bar{W}_q$")
    line1_p95 = ax1.plot(rhos, p95_delay, color="#aec7e8", linestyle="--", marker="s", label=r"95th Percentile Delay $W_{q, 95}$")
    ax1.fill_between(rhos, np.maximum(0, mean_delay - std_delay), mean_delay + std_delay, color=color1, alpha=0.15)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(-0.001, max(0.015, np.max(p95_delay) * 1.5))

    # Right axis: Load Shedding Ratio R_shed (%)
    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel(r"Load Shedding Ratio $R_{\mathrm{shed}}$ [%]", color=color2, fontweight="bold")
    line2 = ax2.plot(rhos, load_shed, color=color2, marker="^", linestyle="-.", label=r"Load-Shedding Ratio $R_{\mathrm{shed}}$")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-0.5, 10.0)

    # Combined legend
    lines = line1 + line1_p95 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)

    ax1.set_title(r"Scenario S2: Offered Load Sweep vs. Queue Delay and Load Shedding ($Q_{0.99}$)", pad=12)
    plt.tight_layout()
    out_file = output_dir / "S2_load_sweep_delay_and_shedding.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[OK] Generated: {out_file}")


def plot_s5_sla_delay_sweep(df_s5: pd.DataFrame, output_dir: Path):
    """Plot S5: SLA Delay Threshold Sweep vs Shedding & Attack Recall."""
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0), dpi=300)

    agg = df_s5.groupby("sweep_W_shed_ms").agg({
        "load_shed_ae_pct": "mean",
        "effective_heavy_passage_recall": "mean",
        "ideal_routing_yield": "mean",
        "max_waiting_time_ms": "mean",
    }).reset_index().sort_values("sweep_W_shed_ms")

    w_shed_ms = agg["sweep_W_shed_ms"].to_numpy()
    shed_pct = agg["load_shed_ae_pct"].to_numpy()
    eff_recall = agg["effective_heavy_passage_recall"].to_numpy()
    ideal_yield = agg["ideal_routing_yield"].to_numpy()

    # Left axis: Recall / Yield
    color1 = "#2ca02c"
    ax1.set_xlabel(r"SLA Delay Threshold $W_{\mathrm{shed}}$ [ms] (at $\rho_{\mathrm{offered}} = 2.0$)", fontweight="bold")
    ax1.set_ylabel(r"Attack Passage & Yield Ratio", color=color1, fontweight="bold")
    l1 = ax1.plot(w_shed_ms, ideal_yield, color=color1, marker="o", label=r"Ideal Routing Yield $\mathrm{TPR}_{\mathrm{ideal}}$")
    l2 = ax1.plot(w_shed_ms, eff_recall, color="#17becf", marker="s", linestyle="--", label=r"Effective Heavy Recall $\mathrm{TPR}_{\mathrm{Heavy}}^{\mathrm{eff}}$")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.0, 0.5)

    # Right axis: Load Shedding Ratio
    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel(r"Load Shedding Ratio $R_{\mathrm{shed}}$ [%]", color=color2, fontweight="bold")
    l3 = ax2.plot(w_shed_ms, shed_pct, color=color2, marker="^", linestyle="-.", label=r"Load-Shedding Ratio $R_{\mathrm{shed}}$")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-0.5, 5.0)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    ax1.set_title(r"Scenario S5: SLA Delay Threshold $W_{\mathrm{shed}}$ Sweep ($\rho_{\mathrm{offered}} = 2.0$, Overload)", pad=12)
    plt.tight_layout()
    out_file = output_dir / "S5_sla_delay_sweep.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[OK] Generated: {out_file}")


def plot_s6_q99_vs_q95_crossover(df_s6: pd.DataFrame, output_dir: Path):
    """
    Plot S6: Empirical Q99 vs Q95 comparison over simulated range rho in [0.1, 3.0]
    with visual annotations of theoretical critical queue stability limits.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=300)

    agg = df_s6.groupby(["quantile", "target_rho"]).agg({
        "effective_heavy_passage_recall": "mean",
        "effective_heavy_benign_fpr": "mean",
        "ideal_routing_yield": "mean",
        "heavy_admitted_pct": "mean",
        "mean_waiting_time_ms": "mean",
    }).reset_index()

    q99 = agg[agg["quantile"] == "Q99"].sort_values("target_rho")
    q95 = agg[agg["quantile"] == "Q95"].sort_values("target_rho")

    rhos = q99["target_rho"].to_numpy()

    # Panel 1: Empirical Attack Capture Yield vs Benign FPR
    ax1.plot(rhos, q99["ideal_routing_yield"], color="#1f77b4", marker="o", label=r"$Q_{0.99}$ (Baseline) Ideal Yield")
    ax1.plot(rhos, q95["ideal_routing_yield"], color="#ff7f0e", marker="s", label=r"$Q_{0.95}$ (Candidate) Ideal Yield")
    ax1.plot(rhos, q99["effective_heavy_passage_recall"], color="#1f77b4", linestyle="--", marker="x", label=r"$Q_{0.99}$ Heavy Recall")
    ax1.plot(rhos, q95["effective_heavy_passage_recall"], color="#ff7f0e", linestyle="--", marker="+", label=r"$Q_{0.95}$ Heavy Recall")

    ax1.set_xlabel(r"Normalized Edge Load $\rho_{\mathrm{offered}} = \lambda_{\mathrm{arr}} / \mu_{\mathrm{heavy}}$", fontweight="bold")
    ax1.set_ylabel(r"Attack Passage Recall & Yield", fontweight="bold")
    ax1.set_title("Empirical Attack Recall Gain ($Q_{0.99}$ vs. $Q_{0.95}$)", pad=10)
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc="center right", frameon=True)

    # Panel 2: Heavy Path Admission Rate and Analytical Stability Boundaries
    ax2.plot(rhos, q99["heavy_admitted_pct"], color="#1f77b4", marker="o", label=r"$Q_{0.99}$ Heavy Admission Rate ($\approx 1.40\%$)")
    ax2.plot(rhos, q95["heavy_admitted_pct"], color="#ff7f0e", marker="s", label=r"$Q_{0.95}$ Heavy Admission Rate ($\approx 7.46\%$)")

    # Annotate analytical stability limits
    ax2.axvline(x=15.24, color="#d62728", linestyle=":", linewidth=2, label=r"Analytical Limit $Q_{0.95}$ ($\tilde{\lambda}_{\mathrm{crit}} \approx 15.24$)")
    ax2.axvline(x=80.65, color="#2ca02c", linestyle=":", linewidth=2, label=r"Analytical Limit $Q_{0.99}$ ($\tilde{\lambda}_{\mathrm{crit}} \approx 80.65$)")

    # Shaded analytical transition region
    ax2.axvspan(15.24, 80.65, color="#ffbb78", alpha=0.25, label=r"Theoretical Stability Region $\tilde{\lambda} \in (15.24, 80.65)$")

    ax2.set_xlabel(r"Normalized Edge Load $\tilde{\lambda} = \lambda_{\mathrm{arr}} / \mu_{\mathrm{heavy}}$", fontweight="bold")
    ax2.set_ylabel(r"Heavy Path Admission $p_{\mathrm{Gate}}$ [%]", fontweight="bold")
    ax2.set_title("Admission Rate & Analytical Stability Boundaries", pad=10)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 12)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.suptitle("Scenario S6: Trade-off & Analytical Queue Stability Limits ($Q_{0.99}$ vs. $Q_{0.95}$)", y=0.98, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_file = output_dir / "S6_Q99_vs_Q95_crossover.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[OK] Generated: {out_file}")


def plot_s7_client_heterogeneity(df: pd.DataFrame, output_dir: Path):
    """Plot S7: Client 02 vs Client 04 vs Standard Client bundle under non-IID bursty traffic."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=300)

    # Filter S7 records
    s7 = df[df["scenario_id"] == "S7"].copy()
    
    # Compute baseline reference bundle (mean across clients 00, 01, 03, 05-09 in S6)
    s6_standard = df[(df["scenario_id"] == "S6") & (~df["client_id"].isin(["client_02", "client_04"]))].groupby("quantile").agg({
        "effective_heavy_passage_recall": "mean",
        "ae_fallback_recall": "mean",
        "ideal_routing_yield": "mean",
        "heavy_admitted_pct": "mean",
        "mean_waiting_time_ms": "mean",
    }).reset_index()

    # Client 02 and 04 S7 averages
    c2 = s7[s7["client_id"] == "client_02"].groupby("quantile").agg({
        "effective_heavy_passage_recall": "mean",
        "ae_fallback_recall": "mean",
        "ideal_routing_yield": "mean",
        "heavy_admitted_pct": "mean",
        "mean_waiting_time_ms": "mean",
    }).reset_index()

    c4 = s7[s7["client_id"] == "client_04"].groupby("quantile").agg({
        "effective_heavy_passage_recall": "mean",
        "ae_fallback_recall": "mean",
        "ideal_routing_yield": "mean",
        "heavy_admitted_pct": "mean",
        "mean_waiting_time_ms": "mean",
    }).reset_index()

    labels = ["Standard Bundle", "Client 04 (Diffused)", "Client 02 (Extreme Non-IID)"]
    x = np.arange(len(labels))
    width = 0.35

    # Panel 1: Ideal Routing Yield Comparison (Q99 vs Q95)
    yield_q99 = [
        s6_standard[s6_standard["quantile"] == "Q99"]["ideal_routing_yield"].iloc[0],
        c4[c4["quantile"] == "Q99"]["ideal_routing_yield"].iloc[0],
        c2[c2["quantile"] == "Q99"]["ideal_routing_yield"].iloc[0],
    ]
    yield_q95 = [
        s6_standard[s6_standard["quantile"] == "Q95"]["ideal_routing_yield"].iloc[0],
        c4[c4["quantile"] == "Q95"]["ideal_routing_yield"].iloc[0],
        c2[c2["quantile"] == "Q95"]["ideal_routing_yield"].iloc[0],
    ]

    rects1 = ax1.bar(x - width/2, yield_q99, width, label=r"$Q_{0.99}$ (Baseline)", color="#1f77b4")
    rects2 = ax1.bar(x + width/2, yield_q95, width, label=r"$Q_{0.95}$ (Candidate)", color="#ff7f0e")

    ax1.set_ylabel("Ideal Attack Routing Yield", fontweight="bold")
    ax1.set_title("Total Security Attack Yield under Non-IID Stress", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc="upper left", frameon=True)

    # Add direct percentage labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h*100:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h*100:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: Heavy Path Admission Rate (%)
    adm_q99 = [
        s6_standard[s6_standard["quantile"] == "Q99"]["heavy_admitted_pct"].iloc[0],
        c4[c4["quantile"] == "Q99"]["heavy_admitted_pct"].iloc[0],
        c2[c2["quantile"] == "Q99"]["heavy_admitted_pct"].iloc[0],
    ]
    adm_q95 = [
        s6_standard[s6_standard["quantile"] == "Q95"]["heavy_admitted_pct"].iloc[0],
        c4[c4["quantile"] == "Q95"]["heavy_admitted_pct"].iloc[0],
        c2[c2["quantile"] == "Q95"]["heavy_admitted_pct"].iloc[0],
    ]

    rects3 = ax2.bar(x - width/2, adm_q99, width, label=r"$Q_{0.99}$ (Baseline)", color="#2ca02c")
    rects4 = ax2.bar(x + width/2, adm_q95, width, label=r"$Q_{0.95}$ (Candidate)", color="#d62728")

    ax2.set_ylabel("Heavy Path Admission Rate [%]", fontweight="bold")
    ax2.set_title("Heavy Path Queue Triage Rate under Non-IID Stress", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0.0, 18.0)
    ax2.legend(loc="upper left", frameon=True)

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Scenario S7: Edge Client Heterogeneity & Non-IID Resilience", y=0.98, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_file = output_dir / "S7_client_heterogeneity_queue.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[OK] Generated: {out_file}")


def generate_all_plots():
    """Generate all Phase 4.4 publication figures."""
    print("Generating Stage 4 publication plots...")
    metrics_dir = Path("outputs/metrics/stage4")
    plots_dir = Path("outputs/plots/stage4")
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "stage4_scenario_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. S2 Load Sweep
    df_s2 = df[df["scenario_id"] == "S2"]
    plot_s2_load_sweep(df_s2, plots_dir)

    # 2. S5 SLA Delay Sweep
    df_s5 = df[df["scenario_id"] == "S5"]
    plot_s5_sla_delay_sweep(df_s5, plots_dir)

    # 3. S6 Q99 vs Q95 Crossover
    df_s6 = df[df["scenario_id"] == "S6"]
    plot_s6_q99_vs_q95_crossover(df_s6, plots_dir)

    # 4. S7 Client Heterogeneity
    plot_s7_client_heterogeneity(df, plots_dir)

    print(f"[DONE] All 4 publication plots successfully saved to: {plots_dir}")


if __name__ == "__main__":
    generate_all_plots()


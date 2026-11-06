"""
Stage 5: Client-Level Performance Dispersion & Sample-Weighted Pooled Metrics.

Computes unweighted macro distribution statistics (Mean, Std, Median, IQR, Min, Max, Range)
and sample-weighted pooled metrics (via raw confusion matrix summation: sum(TP), sum(FP), sum(FN), sum(TN))
across all 10 federated clients.

Covers:
1. Stage 5 Phase 5.4 End-to-End Integrated Pipeline (Nominal Operating Baseline S1)
2. Stage 5 Phase 5.3 Federated Heavy Models

Output:
- outputs/metrics/stage5/validation_dispersion_metrics.csv
"""

from pathlib import Path
import sys
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd


def compute_dispersion_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute Mean, Std, Median, IQR, Min, Max, and Range for an array."""
    if len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "min": 0.0,
            "max": 0.0,
            "range": 0.0,
        }
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    median = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    v_range = float(v_max - v_min)
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "median": round(median, 6),
        "iqr": round(iqr, 6),
        "min": round(v_min, 6),
        "max": round(v_max, 6),
        "range": round(v_range, 6),
    }


def evaluate_all_dispersion() -> pd.DataFrame:
    """Execute complete dispersion and pooled metrics analysis."""
    print("=" * 75)
    print("Computing Client-Level Performance Dispersion & Pooled Confusion Matrices")
    print("=" * 75)

    output_dir = Path("outputs/metrics/stage5")
    e2e_csv = output_dir / "phase5.4_e2e_results.csv"
    fed_csv = output_dir / "phase5.3_federated_results.csv"
    fedper_csv = output_dir / "phase5.3_fedper_results.csv"

    records: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Part 1: Phase 5.4 End-to-End Pipeline Evaluation (Nominal S1 Baseline)
    # -----------------------------------------------------------------------
    if e2e_csv.exists():
        df_e2e = pd.read_csv(e2e_csv)
        s1_df = df_e2e[df_e2e["scenario_id"] == "S1_Nominal"]

        # Group by config_short, quantile, tier
        grouped = s1_df.groupby(["config_short", "config_full", "quantile", "tier", "tier_label"])

        for (cfg_short, cfg_full, q, tier, tier_label), group in grouped:
            # Check availability of raw confusion matrix counts
            has_raw_counts = all(col in group.columns for col in ["tp", "fp", "fn", "tn"])

            f1_vals = group["f1_score"].dropna().to_numpy()
            prec_vals = group["precision"].dropna().to_numpy()
            rec_vals = group["recall_e2e"].dropna().to_numpy()

            f1_stats = compute_dispersion_stats(f1_vals)
            prec_stats = compute_dispersion_stats(prec_vals)
            rec_stats = compute_dispersion_stats(rec_vals)

            if has_raw_counts:
                tp_tot = int(group["tp"].sum())
                fp_tot = int(group["fp"].sum())
                fn_tot = int(group["fn"].sum())
                tn_tot = int(group["tn"].sum())

                prec_pooled = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
                rec_pooled = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
                f1_pooled = (
                    (2.0 * prec_pooled * rec_pooled) / (prec_pooled + rec_pooled)
                    if (prec_pooled + rec_pooled) > 0
                    else 0.0
                )
                acc_pooled = (tp_tot + tn_tot) / (tp_tot + fp_tot + fn_tot + tn_tot) if (tp_tot + fp_tot + fn_tot + tn_tot) > 0 else 0.0
                pooled_status = "CALCULATED"
            else:
                tp_tot = fp_tot = fn_tot = tn_tot = None
                prec_pooled = rec_pooled = f1_pooled = acc_pooled = None
                pooled_status = "NOT AVAILABLE"

            records.append({
                "benchmark_scope": "Phase 5.4 End-to-End Pipeline (S1 Nominal)",
                "config_short": cfg_short,
                "config_full": cfg_full,
                "quantile": q,
                "tier": tier,
                "tier_label": tier_label,
                "num_clients": len(group),
                # Macro F1 stats
                "macro_f1_mean": f1_stats["mean"],
                "macro_f1_std": f1_stats["std"],
                "macro_f1_median": f1_stats["median"],
                "macro_f1_iqr": f1_stats["iqr"],
                "macro_f1_min": f1_stats["min"],
                "macro_f1_max": f1_stats["max"],
                "macro_f1_range": f1_stats["range"],
                # Macro Precision stats
                "macro_prec_mean": prec_stats["mean"],
                "macro_prec_std": prec_stats["std"],
                "macro_prec_median": prec_stats["median"],
                "macro_prec_iqr": prec_stats["iqr"],
                # Macro Recall stats
                "macro_rec_mean": rec_stats["mean"],
                "macro_rec_std": rec_stats["std"],
                "macro_rec_median": rec_stats["median"],
                "macro_rec_iqr": rec_stats["iqr"],
                # Pooled Confusion Matrix & Metrics
                "pooled_status": pooled_status,
                "tp_pooled": tp_tot,
                "fp_pooled": fp_tot,
                "fn_pooled": fn_tot,
                "tn_pooled": tn_tot,
                "pooled_precision": round(prec_pooled, 6) if prec_pooled is not None else None,
                "pooled_recall": round(rec_pooled, 6) if rec_pooled is not None else None,
                "pooled_f1": round(f1_pooled, 6) if f1_pooled is not None else None,
                "pooled_accuracy": round(acc_pooled, 6) if acc_pooled is not None else None,
            })

    # -----------------------------------------------------------------------
    # Part 2: Phase 5.3 Federated Heavy Models Evaluation
    # -----------------------------------------------------------------------
    dfs_to_combine = []
    if fed_csv.exists():
        dfs_to_combine.append(pd.read_csv(fed_csv))
    if fedper_csv.exists():
        dfs_to_combine.append(pd.read_csv(fedper_csv))

    if dfs_to_combine:
        df_fed = pd.concat(dfs_to_combine, ignore_index=True)
        # Drop duplicates if any
        df_fed = df_fed.drop_duplicates(subset=["client_id", "quantile", "model_type", "strategy", "tier"])

        grouped_fed = df_fed.groupby(["model_type", "strategy", "quantile", "tier", "tier_label"])

        for (m_type, strat, q, tier, tier_label), group in grouped_fed:
            cfg_name = f"{strat} {m_type}"
            has_raw_counts = all(col in group.columns for col in ["tp", "fp", "fn", "tn"])

            f1_vals = group["f1_score"].dropna().to_numpy()
            prec_vals = group["precision"].dropna().to_numpy()
            rec_vals = group["recall"].dropna().to_numpy()

            f1_stats = compute_dispersion_stats(f1_vals)
            prec_stats = compute_dispersion_stats(prec_vals)
            rec_stats = compute_dispersion_stats(rec_vals)

            if has_raw_counts:
                tp_tot = int(group["tp"].sum())
                fp_tot = int(group["fp"].sum())
                fn_tot = int(group["fn"].sum())
                tn_tot = int(group["tn"].sum())

                prec_pooled = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0.0
                rec_pooled = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
                f1_pooled = (
                    (2.0 * prec_pooled * rec_pooled) / (prec_pooled + rec_pooled)
                    if (prec_pooled + rec_pooled) > 0
                    else 0.0
                )
                acc_pooled = (tp_tot + tn_tot) / (tp_tot + fp_tot + fn_tot + tn_tot) if (tp_tot + fp_tot + fn_tot + tn_tot) > 0 else 0.0
                pooled_status = "CALCULATED"
            else:
                tp_tot = fp_tot = fn_tot = tn_tot = None
                prec_pooled = rec_pooled = f1_pooled = acc_pooled = None
                pooled_status = "NOT AVAILABLE"

            records.append({
                "benchmark_scope": "Phase 5.3 Heavy Path Classification",
                "config_short": cfg_name,
                "config_full": f"Federated {cfg_name}",
                "quantile": q,
                "tier": tier,
                "tier_label": tier_label,
                "num_clients": len(group),
                # Macro F1 stats
                "macro_f1_mean": f1_stats["mean"],
                "macro_f1_std": f1_stats["std"],
                "macro_f1_median": f1_stats["median"],
                "macro_f1_iqr": f1_stats["iqr"],
                "macro_f1_min": f1_stats["min"],
                "macro_f1_max": f1_stats["max"],
                "macro_f1_range": f1_stats["range"],
                # Macro Precision stats
                "macro_prec_mean": prec_stats["mean"],
                "macro_prec_std": prec_stats["std"],
                "macro_prec_median": prec_stats["median"],
                "macro_prec_iqr": prec_stats["iqr"],
                # Macro Recall stats
                "macro_rec_mean": rec_stats["mean"],
                "macro_rec_std": rec_stats["std"],
                "macro_rec_median": rec_stats["median"],
                "macro_rec_iqr": rec_stats["iqr"],
                # Pooled Confusion Matrix & Metrics
                "pooled_status": pooled_status,
                "tp_pooled": tp_tot,
                "fp_pooled": fp_tot,
                "fn_pooled": fn_tot,
                "tn_pooled": tn_tot,
                "pooled_precision": round(prec_pooled, 6) if prec_pooled is not None else None,
                "pooled_recall": round(rec_pooled, 6) if rec_pooled is not None else None,
                "pooled_f1": round(f1_pooled, 6) if f1_pooled is not None else None,
                "pooled_accuracy": round(acc_pooled, 6) if acc_pooled is not None else None,
            })

    out_df = pd.DataFrame(records)
    out_csv = output_dir / "validation_dispersion_metrics.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Exported {len(out_df)} dispersion summary rows to {out_csv}")
    return out_df


if __name__ == "__main__":
    evaluate_all_dispersion()


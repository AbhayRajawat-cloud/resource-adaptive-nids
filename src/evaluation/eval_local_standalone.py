"""
Stage 5: Phase 5.2 Local Standalone Model Evaluation & Statistical Comparison Engine.

Evaluates trained local HeavyGRU and SupervisedDenseMLP models across all 10 clients on:
- Tier 1: All Gate-passed test flows (N_test_passed).
- Tier 2: Unique Gate-passed test vectors (tier2_mask_test).
- Tier 3: Training/validation-exclusive unique test vectors (tier3_mask_test).

Computes:
- Accuracy, Precision, Recall, F1-Score, PR-AUC, ROC-AUC, Confusion Matrix (TP, TN, FP, FN).
- Paired statistical comparison between HeavyGRU (W=1) and SupervisedDenseMLP:
  - Per-client F1 differences (Delta F1 = F1_GRU - F1_MLP)
  - Mean and median F1 differences
  - Paired Wilcoxon signed-rank test
  - Hodges-Lehmann effect-size estimate
  - 95% bootstrap confidence interval
  - Formal statistical limitations documentation (N=10 clients)

Outputs:
- outputs/metrics/stage5/phase5.2_local_training_results.json
- outputs/metrics/stage5/phase5.2_local_training_results.csv
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import torch

from src.models.heavy_models import HeavyGRU, SupervisedDenseMLP


def compute_hodges_lehmann_and_ci(
    diffs: np.ndarray,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute Hodges-Lehmann median estimator (median of all Walsh averages)
    and 95% bootstrap confidence interval of the mean difference.
    """
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Walsh averages (D_i + D_j) / 2 for i <= j
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((diffs[i] + diffs[j]) / 2.0)
    hl_estimate = float(np.median(walsh))

    # Bootstrap CI of mean difference
    rng = np.random.default_rng(seed)
    boot_means = np.empty(num_bootstrap, dtype=np.float64)
    for b in range(num_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return hl_estimate, ci_low, ci_high


def evaluate_binary_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for a binary prediction vector.
    Explicitly handles single-class edge cases where ROC-AUC/PR-AUC are mathematically undefined.
    """
    y_pred = (probs >= threshold).astype(int)
    n_samples = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_samples == 0:
        return {
            "n_samples": 0,
            "n_pos": 0,
            "n_neg": 0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
            "roc_auc": None, "pr_auc": None,
        }

    # Confusion matrix
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC & PR-AUC: undefined if only 1 class is present in y_true
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        roc_auc = float(roc_auc_score(y_true, probs))
        pr_auc = float(average_precision_score(y_true, probs))
    else:
        roc_auc = None
        pr_auc = None

    return {
        "n_samples": n_samples,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def evaluate_all_local_models(
    data_dir: Path = Path("data/processed/gate_passed"),
    ckpt_dir: Path = Path("outputs/checkpoints/stage5/phase5.2"),
    metrics_dir: Path = Path("outputs/metrics/stage5"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run full local evaluation across 10 clients, 2 quantiles, 2 model architectures, and 3 tiers.
    """
    print("=" * 70)
    print("Starting Stage 5 Phase 5.2: Local Model Evaluation & Statistical Comparison")
    print("=" * 70)

    quantiles = ["Q99", "Q95"]
    model_types = ["gru", "mlp"]
    rows: List[Dict[str, Any]] = []

    # Load training metadata for contextual logging
    meta_json_path = metrics_dir / "phase5.2_training_metadata.json"
    train_metadata = {}
    if meta_json_path.exists():
        with open(meta_json_path, "r", encoding="utf-8") as f:
            runs_meta = json.load(f)
            for rm in runs_meta:
                key = (rm["client_id"], rm["quantile"], rm["model_type"].lower())
                train_metadata[key] = rm

    # Master evaluation loop
    for q in quantiles:
        for cid in range(10):
            cid_str = f"client_{cid:02d}"
            npz_path = data_dir / q / f"{cid_str}_passed.npz"
            npz_data = np.load(npz_path)

            x_test = npz_data["x_test_passed"]
            y_test = npz_data["y_test_passed"].astype(int)
            tier2_mask = npz_data["tier2_mask_test"]
            tier3_mask = npz_data["tier3_mask_test"]

            for m_type in model_types:
                ckpt_file = ckpt_dir / q / f"{cid_str}_local_{m_type}.pt"
                meta_key = (cid, q, m_type)
                rm = train_metadata.get(meta_key, {})

                # Load trained model
                if m_type == "gru":
                    model = HeavyGRU(input_dim=38, hidden_dim=64)
                    model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
                    model.eval()
                    with torch.no_grad():
                        t_x = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) # (N, 1, 38)
                        logits = model(t_x)
                        probs = torch.sigmoid(logits).numpy()
                else:
                    model = SupervisedDenseMLP(input_dim=38)
                    model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
                    model.eval()
                    with torch.no_grad():
                        t_x = torch.tensor(x_test, dtype=torch.float32) # (N, 38)
                        logits = model(t_x)
                        probs = torch.sigmoid(logits).numpy()

                # Evaluate across Tier 1, Tier 2, Tier 3
                tiers = [
                    (1, "Tier 1 (All Passed)", np.ones(len(y_test), dtype=bool)),
                    (2, "Tier 2 (Unique Passed)", tier2_mask),
                    (3, "Tier 3 (Train/Val-Exclusive Unique)", tier3_mask),
                ]

                for tier_num, tier_label, mask in tiers:
                    y_sub = y_test[mask]
                    probs_sub = probs[mask]
                    metrics = evaluate_binary_predictions(y_sub, probs_sub, threshold=0.5)

                    row = {
                        "client_id": cid,
                        "client_str": cid_str,
                        "quantile": q,
                        "model_type": m_type.upper(),
                        "tier": tier_num,
                        "tier_label": tier_label,
                        "checkpoint_path": str(ckpt_file),
                        "w_pos": rm.get("dataset_statistics", {}).get("w_pos", None),
                        "best_epoch": rm.get("training_outcome", {}).get("best_epoch", None),
                        "best_val_loss": rm.get("training_outcome", {}).get("best_val_loss", None),
                        "n_samples": metrics["n_samples"],
                        "n_attacks": metrics["n_pos"],
                        "n_benign": metrics["n_neg"],
                        "tp": metrics["tp"],
                        "tn": metrics["tn"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"],
                    }
                    rows.append(row)

    df_results = pd.DataFrame(rows)

    # -------------------------------------------------------------
    # Paired Statistical Comparison: HeavyGRU vs SupervisedDenseMLP
    # -------------------------------------------------------------
    stat_comparisons = {}

    for q in quantiles:
        stat_comparisons[q] = {}
        for tier_num in [1, 2, 3]:
            tier_key = f"tier_{tier_num}"
            df_q_t = df_results[(df_results["quantile"] == q) & (df_results["tier"] == tier_num)]

            gru_df = df_q_t[df_q_t["model_type"] == "GRU"].sort_values("client_id")
            mlp_df = df_q_t[df_q_t["model_type"] == "MLP"].sort_values("client_id")

            f1_gru = gru_df["f1_score"].to_numpy()
            f1_mlp = mlp_df["f1_score"].to_numpy()
            diffs = f1_gru - f1_mlp

            mean_diff = float(np.mean(diffs))
            median_diff = float(np.median(diffs))
            hl_est, ci_low, ci_high = compute_hodges_lehmann_and_ci(diffs, num_bootstrap=1000, seed=42)

            # Paired Wilcoxon signed-rank test
            # If all diffs are identical / zero, Wilcoxon test handles ties via 'pratt' or zero_method
            try:
                if np.all(diffs == 0.0):
                    w_stat, w_pval = 0.0, 1.0
                else:
                    w_res = wilcoxon(f1_gru, f1_mlp, zero_method="pratt", alternative="two-sided")
                    w_stat = float(w_res.statistic)
                    w_pval = float(w_res.pvalue)
            except Exception as e:
                w_stat, w_pval = float("nan"), float("nan")

            per_client_diffs = []
            for i in range(10):
                per_client_diffs.append({
                    "client_id": i,
                    "f1_gru": float(f1_gru[i]),
                    "f1_mlp": float(f1_mlp[i]),
                    "delta_f1": float(diffs[i]),
                })

            stat_comparisons[q][tier_key] = {
                "quantile": q,
                "tier": tier_num,
                "mean_f1_gru": float(np.mean(f1_gru)),
                "mean_f1_mlp": float(np.mean(f1_mlp)),
                "mean_difference": mean_diff,
                "median_difference": median_diff,
                "hodges_lehmann_estimate": hl_est,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "wilcoxon_statistic": w_stat,
                "wilcoxon_pvalue": w_pval,
                "null_hypothesis_rejected_at_005": bool(w_pval < 0.05) if not np.isnan(w_pval) else False,
                "per_client_differences": per_client_diffs,
            }

    # Statistical limitations documentation
    limitations_doc = (
        "Statistical comparisons are based on N=10 paired client observations. "
        "With N=10, the paired Wilcoxon signed-rank test has limited statistical power (minimum two-sided p-value 0.00195). "
        "Failure to reject the null hypothesis (p >= 0.05) must NOT be interpreted as evidence of equivalence. "
        "Furthermore, W=1 GRU input vectors do not model physical temporal sequence dynamics; "
        "the comparison strictly evaluates the non-linear feature extraction and gating capacity of the two architectures."
    )

    full_output_package = {
        "metadata": {
            "stage": "5.2",
            "description": "Local Standalone HeavyGRU vs SupervisedDenseMLP Baselines",
            "decision_threshold": 0.5,
            "total_evaluations": len(df_results),
            "quantiles": quantiles,
            "models": ["HeavyGRU (W=1)", "SupervisedDenseMLP"],
            "statistical_limitations": limitations_doc,
        },
        "statistical_comparison": stat_comparisons,
        "evaluations": rows,
    }

    # Save outputs
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_out = metrics_dir / "phase5.2_local_training_results.csv"
    json_out = metrics_dir / "phase5.2_local_training_results.json"

    df_results.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(full_output_package, f, indent=2)

    print(f"[OK] Saved detailed evaluation CSV to: {csv_out}")
    print(f"[OK] Saved statistical comparison JSON to: {json_out}")
    print("=" * 70)

    return df_results, full_output_package


if __name__ == "__main__":
    evaluate_all_local_models()


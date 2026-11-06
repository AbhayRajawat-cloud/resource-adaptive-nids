"""
Stage 5: Phase 5.3 Multi-Tier Federated Evaluation & Statistical Comparison Engine.

Evaluates trained federated checkpoints across all 10 clients on:
- Tier 1: All Gate-passed test flows
- Tier 2: Unique Gate-passed test vectors (tier2_mask_test)
- Tier 3: Train/val-exclusive unique test vectors (tier3_mask_test)

Evaluation Coverage:
1. Standard FedAvg (HeavyGRU & SupervisedDenseMLP under Q0.99 and Q0.95) -> 120 evaluation rows
2. Strategy B (FedPer) (HeavyGRU & SupervisedDenseMLP under Q0.99) -> 60 evaluation rows

Statistical Testing & Multiple Testing Corrections:
- Paired Wilcoxon signed-rank tests with Hodges-Lehmann median differences and 95% bootstrap CIs.
- Benjamini-Hochberg (BH) False Discovery Rate (FDR) procedure (alpha=0.05) applied across 3 families:
    Family 1: HeavyGRU vs SupervisedDenseMLP (Federated)
    Family 2: Federated vs Local Standalone Baseline (FL Gain Delta_FL)
    Family 3: Multi-Tier Lineage (Tier 1 vs Tier 2, Tier 1 vs Tier 3)
- Strategy A (FedAvg) vs Strategy B (FedPer) Ablation analysis (with Client 02 deep dive).
- Formal communication traffic accounting in binary KiB / MiB units.

Outputs:
- outputs/metrics/stage5/phase5.3_federated_results.csv (120 rows)
- outputs/metrics/stage5/phase5.3_fedper_results.csv (60 rows)
- outputs/metrics/stage5/phase5.3_federated_results.json
"""

import copy
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
)
import torch

from src.models.heavy_models import HeavyGRU, SupervisedDenseMLP
from src.federated.fedavg_aggregator import is_bn_buffer, is_private_head
from src.evaluation.eval_local_standalone import evaluate_binary_predictions, compute_hodges_lehmann_and_ci


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply Benjamini-Hochberg (BH) False Discovery Rate (FDR) procedure.
    Handles NaN values safely by keeping them unchanged.

    Returns:
        (adjusted_p_values, reject_flags)
    """
    m = len(p_values)
    if m == 0:
        return [], []

    # Filter valid vs nan
    valid_entries = [(idx, p) for idx, p in enumerate(p_values) if not np.isnan(p)]
    m_valid = len(valid_entries)

    q_result = [float("nan")] * m
    reject_result = [False] * m

    if m_valid == 0:
        return q_result, reject_result

    # Sort valid p-values ascending
    valid_entries.sort(key=lambda x: x[1])
    sorted_p = [p for _, p in valid_entries]
    orig_indices = [idx for idx, _ in valid_entries]

    q_vals = [0.0] * m_valid
    accum_min = 1.0
    for i in range(m_valid - 1, -1, -1):
        rank = i + 1
        q_raw = (m_valid / rank) * sorted_p[i]
        accum_min = min(accum_min, q_raw)
        q_vals[i] = min(1.0, max(0.0, accum_min))

    for sort_idx, orig_idx in enumerate(orig_indices):
        q_result[orig_idx] = q_vals[sort_idx]
        reject_result[orig_idx] = q_vals[sort_idx] < alpha

    return q_result, reject_result


def load_model_for_client(
    model_type: str,
    checkpoint_path: Path,
    client_id: int,
    is_fedper: bool = False,
    bn_companion_path: Optional[Path] = None,
    heads_path: Optional[Path] = None,
) -> torch.nn.Module:
    """
    Load and assemble a model for client evaluation with appropriate weights,
    BatchNorm running buffers (FedBN), and private head (FedPer).
    """
    is_mlp = (model_type.lower() == "mlp")
    cid_str = f"client_{client_id:02d}"

    if is_mlp:
        model = SupervisedDenseMLP(input_dim=38)
    else:
        model = HeavyGRU(input_dim=38, hidden_dim=64)

    state = model.state_dict()

    if not is_fedper:
        # Standard FedAvg
        global_state = torch.load(checkpoint_path, map_location="cpu")
        for k, v in global_state.items():
            state[k] = v.clone()

        # Load client-specific persistent BatchNorm buffers if MLP
        if is_mlp and bn_companion_path is not None and bn_companion_path.exists():
            bn_data = torch.load(bn_companion_path, map_location="cpu")
            client_bn = bn_data.get(cid_str, {})
            for b_key, b_val in client_bn.items():
                state[b_key] = b_val.clone()
    else:
        # FedPer Strategy B: Global body + client private head
        body_state = torch.load(checkpoint_path, map_location="cpu")
        for k, v in body_state.items():
            state[k] = v.clone()

        if heads_path is not None and heads_path.exists():
            heads_data = torch.load(heads_path, map_location="cpu")
            c_info = heads_data.get(cid_str, {})
            c_head = c_info.get("private_head", {})
            for h_key, h_val in c_head.items():
                state[h_key] = h_val.clone()

            # Also load BatchNorm buffers if MLP
            if is_mlp and c_info.get("bn_buffers") is not None:
                for b_key, b_val in c_info["bn_buffers"].items():
                    state[b_key] = b_val.clone()

    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_federated_pipeline(
    data_dir: Path = Path("data/processed/gate_passed"),
    ckpt_dir: Path = Path("outputs/checkpoints/stage5/phase5.3"),
    metrics_dir: Path = Path("outputs/metrics/stage5"),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run full multi-tier evaluation for Standard FedAvg and Strategy B (FedPer).
    """
    print(f"\n{'='*75}")
    print("Starting Stage 5 Phase 5.3: Federated Multi-Tier Evaluation Engine")
    print(f"{'='*75}")

    # Load Phase 5.2 local baseline results for FL gain comparison
    phase52_csv = metrics_dir / "phase5.2_local_training_results.csv"
    local_baseline_dict = {}
    if phase52_csv.exists():
        df_local = pd.read_csv(phase52_csv)
        for _, row in df_local.iterrows():
            key = (int(row["client_id"]), str(row["quantile"]), str(row["model_type"]).upper(), int(row["tier"]))
            local_baseline_dict[key] = {
                "f1_score": float(row["f1_score"]),
                "accuracy": float(row["accuracy"]),
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
            }

    # Load Phase 5.3 training metadata
    meta_path = metrics_dir / "phase5.3_federated_metadata.json"
    fl_metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            runs_meta = json.load(f)
            for rm in runs_meta:
                fl_metadata[rm["run_name"]] = rm

    # -------------------------------------------------------------------
    # Part 1: Standard FedAvg Evaluation (120 rows)
    # -------------------------------------------------------------------
    fedavg_rows = []
    quantiles = ["Q99", "Q95"]
    models = ["gru", "mlp"]

    for q in quantiles:
        for cid in range(10):
            cid_str = f"client_{cid:02d}"
            npz_path = data_dir / q / f"{cid_str}_passed.npz"
            npz_data = np.load(npz_path)

            x_test = npz_data["x_test_passed"]
            y_test = npz_data["y_test_passed"].astype(int)
            tier2_mask = npz_data["tier2_mask_test"]
            tier3_mask = npz_data["tier3_mask_test"]

            for m_type in models:
                run_name = f"global_fedavg_{m_type}_{q}"
                ckpt_path = ckpt_dir / f"{run_name}_R50.pt"
                bn_path = ckpt_dir / f"{run_name}_R50_client_bn.pt" if m_type == "mlp" else None

                model = load_model_for_client(
                    model_type=m_type,
                    checkpoint_path=ckpt_path,
                    client_id=cid,
                    is_fedper=False,
                    bn_companion_path=bn_path,
                )

                # Predict probabilities
                with torch.no_grad():
                    if m_type == "gru":
                        t_x = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
                    else:
                        t_x = torch.tensor(x_test, dtype=torch.float32)
                    logits = model(t_x)
                    probs = torch.sigmoid(logits).numpy()

                tiers = [
                    (1, "Tier 1 (All Passed)", np.ones(len(y_test), dtype=bool)),
                    (2, "Tier 2 (Unique Passed)", tier2_mask),
                    (3, "Tier 3 (Train/Val-Exclusive Unique)", tier3_mask),
                ]

                for tier_num, tier_label, mask in tiers:
                    y_sub = y_test[mask]
                    probs_sub = probs[mask]
                    metrics = evaluate_binary_predictions(y_sub, probs_sub, threshold=0.5)

                    local_info = local_baseline_dict.get((cid, q, m_type.upper(), tier_num), {})
                    local_f1 = local_info.get("f1_score", None)
                    delta_fl_f1 = (metrics["f1_score"] - local_f1) if local_f1 is not None else None

                    row = {
                        "client_id": cid,
                        "client_str": cid_str,
                        "quantile": q,
                        "model_type": m_type.upper(),
                        "strategy": "FedAvg",
                        "tier": tier_num,
                        "tier_label": tier_label,
                        "checkpoint_path": str(ckpt_path),
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
                        "local_f1_score": local_f1,
                        "fl_gain_delta_f1": delta_fl_f1,
                    }
                    fedavg_rows.append(row)

    df_fedavg = pd.DataFrame(fedavg_rows)
    assert len(df_fedavg) == 120, f"Expected 120 FedAvg rows, got {len(df_fedavg)}"

    # -------------------------------------------------------------------
    # Part 2: Strategy B (FedPer) Evaluation (60 rows at Q0.99)
    # -------------------------------------------------------------------
    fedper_rows = []
    q_fedper = "Q99"

    for cid in range(10):
        cid_str = f"client_{cid:02d}"
        npz_path = data_dir / q_fedper / f"{cid_str}_passed.npz"
        npz_data = np.load(npz_path)

        x_test = npz_data["x_test_passed"]
        y_test = npz_data["y_test_passed"].astype(int)
        tier2_mask = npz_data["tier2_mask_test"]
        tier3_mask = npz_data["tier3_mask_test"]

        for m_type in models:
            run_name = f"fedper_{m_type}_{q_fedper}"
            body_path = ckpt_dir / f"{run_name}_R50_body.pt"
            heads_path = ckpt_dir / f"client00_09_{run_name}_heads.pt"

            model = load_model_for_client(
                model_type=m_type,
                checkpoint_path=body_path,
                client_id=cid,
                is_fedper=True,
                heads_path=heads_path,
            )

            with torch.no_grad():
                if m_type == "gru":
                    t_x = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
                else:
                    t_x = torch.tensor(x_test, dtype=torch.float32)
                logits = model(t_x)
                probs = torch.sigmoid(logits).numpy()

            tiers = [
                (1, "Tier 1 (All Passed)", np.ones(len(y_test), dtype=bool)),
                (2, "Tier 2 (Unique Passed)", tier2_mask),
                (3, "Tier 3 (Train/Val-Exclusive Unique)", tier3_mask),
            ]

            for tier_num, tier_label, mask in tiers:
                y_sub = y_test[mask]
                probs_sub = probs[mask]
                metrics = evaluate_binary_predictions(y_sub, probs_sub, threshold=0.5)

                local_info = local_baseline_dict.get((cid, q_fedper, m_type.upper(), tier_num), {})
                local_f1 = local_info.get("f1_score", None)
                delta_fl_f1 = (metrics["f1_score"] - local_f1) if local_f1 is not None else None

                # Find matching standard FedAvg F1 for Strategy A vs B comparison
                match_fedavg = df_fedavg[
                    (df_fedavg["client_id"] == cid)
                    & (df_fedavg["quantile"] == q_fedper)
                    & (df_fedavg["model_type"] == m_type.upper())
                    & (df_fedavg["tier"] == tier_num)
                ]
                fedavg_f1 = float(match_fedavg["f1_score"].values[0]) if len(match_fedavg) > 0 else None
                delta_per_vs_fedavg = (metrics["f1_score"] - fedavg_f1) if fedavg_f1 is not None else None

                row = {
                    "client_id": cid,
                    "client_str": cid_str,
                    "quantile": q_fedper,
                    "model_type": m_type.upper(),
                    "strategy": "FedPer",
                    "tier": tier_num,
                    "tier_label": tier_label,
                    "checkpoint_path": str(body_path),
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
                    "local_f1_score": local_f1,
                    "fl_gain_delta_f1": delta_fl_f1,
                    "fedavg_f1_score": fedavg_f1,
                    "fedper_gain_vs_fedavg": delta_per_vs_fedavg,
                }
                fedper_rows.append(row)

    df_fedper = pd.DataFrame(fedper_rows)
    assert len(df_fedper) == 60, f"Expected 60 FedPer rows, got {len(df_fedper)}"

    # -------------------------------------------------------------------
    # Part 3: Statistical Package with Benjamini-Hochberg FDR Corrections
    # -------------------------------------------------------------------
    print("\nExecuting paired statistical comparison and FDR adjustments...")

    # Helper function for paired Wilcoxon and bootstrap
    def compute_paired_test(v1: np.ndarray, v2: np.ndarray) -> Dict[str, Any]:
        diffs = v1 - v2
        mean_d = float(np.mean(diffs))
        median_d = float(np.median(diffs))
        hl_est, ci_l, ci_h = compute_hodges_lehmann_and_ci(diffs, num_bootstrap=1000, seed=42)

        try:
            if np.all(diffs == 0.0):
                w_stat, w_p = 0.0, 1.0
            else:
                w_res = wilcoxon(v1, v2, zero_method="pratt", alternative="two-sided")
                w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")

        return {
            "mean_diff": mean_d,
            "median_diff": median_d,
            "hodges_lehmann_estimate": hl_est,
            "ci_95_low": ci_l,
            "ci_95_high": ci_h,
            "wilcoxon_stat": w_stat,
            "raw_pvalue": w_p,
        }

    # Family 1: Federated HeavyGRU vs Federated SupervisedDenseMLP
    family1_tests = []
    for q in quantiles:
        for tier_num in [1, 2, 3]:
            sub_gru = df_fedavg[(df_fedavg["quantile"] == q) & (df_fedavg["model_type"] == "GRU") & (df_fedavg["tier"] == tier_num)].sort_values("client_id")
            sub_mlp = df_fedavg[(df_fedavg["quantile"] == q) & (df_fedavg["model_type"] == "MLP") & (df_fedavg["tier"] == tier_num)].sort_values("client_id")

            v_gru = sub_gru["f1_score"].to_numpy()
            v_mlp = sub_mlp["f1_score"].to_numpy()
            res = compute_paired_test(v_gru, v_mlp)
            res.update({
                "comparison": f"GRU_vs_MLP_{q}_Tier{tier_num}",
                "quantile": q,
                "tier": tier_num,
                "mean_f1_gru": float(np.mean(v_gru)),
                "mean_f1_mlp": float(np.mean(v_mlp)),
            })
            family1_tests.append(res)

    pvals_f1 = [t["raw_pvalue"] for t in family1_tests]
    qvals_f1, rejects_f1 = benjamini_hochberg(pvals_f1, alpha=0.05)
    for i, t in enumerate(family1_tests):
        t["bh_qvalue"] = qvals_f1[i]
        t["bh_reject_h0"] = rejects_f1[i]

    # Family 2: Federated vs Phase 5.2 Local Baseline (FL Gain Delta_FL)
    family2_tests = []
    for q in quantiles:
        for m_type in ["GRU", "MLP"]:
            for tier_num in [1, 2, 3]:
                sub_fl = df_fedavg[(df_fedavg["quantile"] == q) & (df_fedavg["model_type"] == m_type) & (df_fedavg["tier"] == tier_num)].sort_values("client_id")
                v_fl = sub_fl["f1_score"].to_numpy()
                v_local = sub_fl["local_f1_score"].to_numpy()

                res = compute_paired_test(v_fl, v_local)
                res.update({
                    "comparison": f"FL_vs_Local_{m_type}_{q}_Tier{tier_num}",
                    "quantile": q,
                    "model_type": m_type,
                    "tier": tier_num,
                    "mean_f1_fl": float(np.mean(v_fl)),
                    "mean_f1_local": float(np.mean(v_local)),
                })
                family2_tests.append(res)

    pvals_f2 = [t["raw_pvalue"] for t in family2_tests]
    qvals_f2, rejects_f2 = benjamini_hochberg(pvals_f2, alpha=0.05)
    for i, t in enumerate(family2_tests):
        t["bh_qvalue"] = qvals_f2[i]
        t["bh_reject_h0"] = rejects_f2[i]

    # Family 3: Multi-Tier Lineage (Tier 1 vs Tier 2, Tier 1 vs Tier 3)
    family3_tests = []
    for q in quantiles:
        for m_type in ["GRU", "MLP"]:
            sub_t1 = df_fedavg[(df_fedavg["quantile"] == q) & (df_fedavg["model_type"] == m_type) & (df_fedavg["tier"] == 1)].sort_values("client_id")
            sub_t2 = df_fedavg[(df_fedavg["quantile"] == q) & (df_fedavg["model_type"] == m_type) & (df_fedavg["tier"] == 2)].sort_values("client_id")
            sub_t3 = df_fedavg[(df_fedavg["quantile"] == q) & (df_fedavg["model_type"] == m_type) & (df_fedavg["tier"] == 3)].sort_values("client_id")

            v_t1 = sub_t1["f1_score"].to_numpy()
            v_t2 = sub_t2["f1_score"].to_numpy()
            v_t3 = sub_t3["f1_score"].to_numpy()

            # T1 vs T2
            res_12 = compute_paired_test(v_t1, v_t2)
            res_12.update({
                "comparison": f"Tier1_vs_Tier2_{m_type}_{q}",
                "quantile": q,
                "model_type": m_type,
                "mean_f1_base": float(np.mean(v_t1)),
                "mean_f1_target": float(np.mean(v_t2)),
            })
            family3_tests.append(res_12)

            # T1 vs T3
            res_13 = compute_paired_test(v_t1, v_t3)
            res_13.update({
                "comparison": f"Tier1_vs_Tier3_{m_type}_{q}",
                "quantile": q,
                "model_type": m_type,
                "mean_f1_base": float(np.mean(v_t1)),
                "mean_f1_target": float(np.mean(v_t3)),
            })
            family3_tests.append(res_13)

    pvals_f3 = [t["raw_pvalue"] for t in family3_tests]
    qvals_f3, rejects_f3 = benjamini_hochberg(pvals_f3, alpha=0.05)
    for i, t in enumerate(family3_tests):
        t["bh_qvalue"] = qvals_f3[i]
        t["bh_reject_h0"] = rejects_f3[i]

    # Strategy A (FedAvg) vs Strategy B (FedPer) Ablation
    fedper_ablation = []
    for m_type in ["GRU", "MLP"]:
        for tier_num in [1, 2, 3]:
            sub_per = df_fedper[(df_fedper["model_type"] == m_type) & (df_fedper["tier"] == tier_num)].sort_values("client_id")
            sub_avg = df_fedavg[(df_fedavg["quantile"] == "Q99") & (df_fedavg["model_type"] == m_type) & (df_fedavg["tier"] == tier_num)].sort_values("client_id")

            v_per = sub_per["f1_score"].to_numpy()
            v_avg = sub_avg["f1_score"].to_numpy()

            res = compute_paired_test(v_per, v_avg)
            res.update({
                "comparison": f"FedPer_vs_FedAvg_{m_type}_Q99_Tier{tier_num}",
                "model_type": m_type,
                "tier": tier_num,
                "mean_f1_fedper": float(np.mean(v_per)),
                "mean_f1_fedavg": float(np.mean(v_avg)),
            })
            fedper_ablation.append(res)

    # Compile Full JSON Package
    full_json_package = {
        "metadata": {
            "stage": "5.3",
            "description": "Federated Heavy Model Training using FedAvg & FedPer",
            "decision_threshold": 0.5,
            "num_clients": 10,
            "quantiles": quantiles,
            "models": ["HeavyGRU", "SupervisedDenseMLP"],
            "strategies": ["Standard FedAvg (FedBN)", "Strategy B (FedPer)"],
            "statistical_families": {
                "family1": "HeavyGRU vs SupervisedDenseMLP (Federated)",
                "family2": "Federated vs Local Baseline (FL Gain Delta_FL)",
                "family3": "Multi-Tier Lineage Comparison",
            },
        },
        "statistical_tests": {
            "family1_gru_vs_mlp": family1_tests,
            "family2_fl_vs_local": family2_tests,
            "family3_multitier_lineage": family3_tests,
            "fedper_ablation": fedper_ablation,
        },
        "communication_traffic_summary": {
            run_name: meta.get("traffic_accounting", {})
            for run_name, meta in fl_metadata.items()
        },
        "evaluation_summary_fedavg": fedavg_rows,
        "evaluation_summary_fedper": fedper_rows,
    }

    # Save to disk
    csv_fedavg_path = metrics_dir / "phase5.3_federated_results.csv"
    csv_fedper_path = metrics_dir / "phase5.3_fedper_results.csv"
    json_path = metrics_dir / "phase5.3_federated_results.json"

    df_fedavg.to_csv(csv_fedavg_path, index=False)
    df_fedper.to_csv(csv_fedper_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json_package, f, indent=2)

    print(f"\n[OK] Saved FedAvg results CSV (120 rows) to: {csv_fedavg_path}")
    print(f"[OK] Saved FedPer results CSV (60 rows) to: {csv_fedper_path}")
    print(f"[OK] Saved statistical package JSON to: {json_path}")
    print(f"{'='*75}\n")

    return df_fedavg, df_fedper, full_json_package


if __name__ == "__main__":
    evaluate_federated_pipeline()


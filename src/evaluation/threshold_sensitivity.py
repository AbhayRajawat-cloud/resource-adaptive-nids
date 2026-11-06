"""
Stage 3: Zero-Retraining Threshold Sensitivity Experiment.

Evaluates the effect of benign calibration quantiles (Q99, Q95, Q90, Q80)
on the Local Autoencoder and Self-Supervised Gate across 10 non-IID clients
and across three evaluation tiers (Tier 1, Tier 2, Tier 3).

Strict constraints:
- No model retraining or fine-tuning.
- Checkpoints loaded in eval() mode with torch.no_grad().
- Thresholds calibrated strictly on local benign validation data (y_val == 0).
- Local client-specific thresholds applied (no global threshold averaging).
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Ensure headless matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.autoencoder import Autoencoder
from src.models.gate import GatingRegressor
from src.evaluation.stage3_eval import construct_test_tiers


QUANTILES = [0.99, 0.95, 0.90, 0.80]
QUANTILE_NAMES = {0.99: "Q99", 0.95: "Q95", 0.90: "Q90", 0.80: "Q80"}


def calibrate_quantiles(
    val_benign_mse: np.ndarray,
    val_benign_e_hat: np.ndarray,
    quantiles: List[float] = QUANTILES,
) -> Dict[str, Dict[str, float]]:
    """
    Compute empirical quantile thresholds strictly on local benign validation scores.
    """
    thresholds = {}
    for q in quantiles:
        q_name = QUANTILE_NAMES[q]
        tau_ae = float(np.percentile(val_benign_mse, q * 100))
        tau_gate = float(np.percentile(val_benign_e_hat, q * 100))
        thresholds[q_name] = {
            "quantile": q,
            "tau_ae": tau_ae,
            "tau_gate": tau_gate,
        }
    return thresholds


def evaluate_tier_metrics(
    y_true: np.ndarray,
    mse_ae: np.ndarray,
    e_hat_gate: np.ndarray,
    tau_ae: float,
    tau_gate: float,
) -> Dict[str, Any]:
    """
    Compute complete performance and downstream routing metrics for a given tier and threshold pair.
    """
    n_samples = len(y_true)
    benign_mask = (y_true == 0)
    attack_mask = (y_true == 1)

    n_benign = int(benign_mask.sum())
    n_attack = int(attack_mask.sum())

    # Autoencoder binary predictions
    ae_pred = (mse_ae >= tau_ae).astype(int)
    tp_ae = int(((ae_pred == 1) & attack_mask).sum())
    fp_ae = int(((ae_pred == 1) & benign_mask).sum())
    tn_ae = int(((ae_pred == 0) & benign_mask).sum())
    fn_ae = int(((ae_pred == 0) & attack_mask).sum())

    ae_recall = float(tp_ae / n_attack) if n_attack > 0 else 0.0
    ae_fpr = float(fp_ae / n_benign) if n_benign > 0 else 0.0

    # Gate binary predictions (Heavy Path routing)
    gate_pred = (e_hat_gate >= tau_gate).astype(int)
    tp_gate = int(((gate_pred == 1) & attack_mask).sum())
    fp_gate = int(((gate_pred == 1) & benign_mask).sum())
    tn_gate = int(((gate_pred == 0) & benign_mask).sum())
    fn_gate = int(((gate_pred == 0) & attack_mask).sum())

    gate_recall = float(tp_gate / n_attack) if n_attack > 0 else 0.0
    gate_fpr = float(fp_gate / n_benign) if n_benign > 0 else 0.0

    # Heavy Path traffic characteristics
    passed_total = tp_gate + fp_gate
    suppressed_total = tn_gate + fn_gate

    heavy_path_invocation_rate_pct = float(passed_total / n_samples * 100.0) if n_samples > 0 else 0.0
    filtered_pct = float(suppressed_total / n_samples * 100.0) if n_samples > 0 else 0.0
    attack_concentration_pct = float(tp_gate / passed_total * 100.0) if passed_total > 0 else 0.0

    # Gate - AE Agreement
    agree_overall_count = int((gate_pred == ae_pred).sum())
    agree_overall_pct = float(agree_overall_count / n_samples * 100.0) if n_samples > 0 else 0.0

    agree_benign_count = int(((gate_pred == ae_pred) & benign_mask).sum())
    agree_benign_pct = float(agree_benign_count / n_benign * 100.0) if n_benign > 0 else 0.0

    agree_attack_count = int(((gate_pred == ae_pred) & attack_mask).sum())
    agree_attack_pct = float(agree_attack_count / n_attack * 100.0) if n_attack > 0 else 0.0

    # Joint Routing Confusion Matrix (AE vs Gate)
    both_trigger = int(((ae_pred == 1) & (gate_pred == 1)).sum())
    ae_only = int(((ae_pred == 1) & (gate_pred == 0)).sum())
    gate_only = int(((ae_pred == 0) & (gate_pred == 1)).sum())
    neither_triggers = int(((ae_pred == 0) & (gate_pred == 0)).sum())

    # Class-conditioned joint routing
    both_trigger_b = int(((ae_pred == 1) & (gate_pred == 1) & benign_mask).sum())
    ae_only_b = int(((ae_pred == 1) & (gate_pred == 0) & benign_mask).sum())
    gate_only_b = int(((ae_pred == 0) & (gate_pred == 1) & benign_mask).sum())
    neither_b = int(((ae_pred == 0) & (gate_pred == 0) & benign_mask).sum())

    both_trigger_a = int(((ae_pred == 1) & (gate_pred == 1) & attack_mask).sum())
    ae_only_a = int(((ae_pred == 1) & (gate_pred == 0) & attack_mask).sum())
    gate_only_a = int(((ae_pred == 0) & (gate_pred == 1) & attack_mask).sum())
    neither_a = int(((ae_pred == 0) & (gate_pred == 0) & attack_mask).sum())

    return {
        "n_samples": n_samples,
        "n_benign": n_benign,
        "n_attack": n_attack,
        "ae": {
            "tp": tp_ae,
            "fp": fp_ae,
            "tn": tn_ae,
            "fn": fn_ae,
            "recall": ae_recall,
            "fpr": ae_fpr,
        },
        "gate": {
            "tp": tp_gate,
            "fp": fp_gate,
            "tn": tn_gate,
            "fn": fn_gate,
            "passage_recall": gate_recall,
            "benign_fpr": gate_fpr,
            "heavy_path_invocation_rate_pct": heavy_path_invocation_rate_pct,
            "filtered_pct": filtered_pct,
            "attack_concentration_pct": attack_concentration_pct,
            "total_passed": passed_total,
            "benign_passed": fp_gate,
            "attack_passed": tp_gate,
            "total_suppressed": suppressed_total,
            "benign_suppressed": tn_gate,
            "attack_suppressed": fn_gate,
        },
        "agreement": {
            "overall_pct": agree_overall_pct,
            "benign_pct": agree_benign_pct,
            "attack_pct": agree_attack_pct,
            "overall_count": agree_overall_count,
            "benign_count": agree_benign_count,
            "attack_count": agree_attack_count,
        },
        "joint_matrix": {
            "both_trigger": both_trigger,
            "ae_only": ae_only,
            "gate_only": gate_only,
            "neither": neither_triggers,
            "benign": {
                "both_trigger": both_trigger_b,
                "ae_only": ae_only_b,
                "gate_only": gate_only_b,
                "neither": neither_b,
            },
            "attack": {
                "both_trigger": both_trigger_a,
                "ae_only": ae_only_a,
                "gate_only": gate_only_a,
                "neither": neither_a,
            },
        },
    }


def run_threshold_sensitivity_experiment(
    data_dir: str = "data/processed/alpha_0.5_seed_42",
    ckpt_dir: str = "outputs/checkpoints/stage3",
    output_dir: str = "outputs/experiments/stage3_threshold_sensitivity",
    n_clients: int = 10,
    batch_size: int = 1024,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    client_thresholds_record = {}
    all_client_rows = []
    tier3_joint_matrices_record = {}

    # Accumulators for sample-weighted pooled metrics
    # structure: pooled_accum[(tier, q_name)] = { ... }
    tiers_list = ["tier1", "tier2", "tier3"]
    pooled_accum = {}
    for t in tiers_list:
        for q_name in QUANTILE_NAMES.values():
            pooled_accum[(t, q_name)] = {
                "n_samples": 0,
                "n_benign": 0,
                "n_attack": 0,
                "ae_tp": 0, "ae_fp": 0, "ae_tn": 0, "ae_fn": 0,
                "gate_tp": 0, "gate_fp": 0, "gate_tn": 0, "gate_fn": 0,
                "agree_overall_count": 0,
                "agree_benign_count": 0,
                "agree_attack_count": 0,
                "joint_both": 0, "joint_ae_only": 0, "joint_gate_only": 0, "joint_neither": 0,
                "joint_both_b": 0, "joint_ae_only_b": 0, "joint_gate_only_b": 0, "joint_neither_b": 0,
                "joint_both_a": 0, "joint_ae_only_a": 0, "joint_gate_only_a": 0, "joint_neither_a": 0,
            }

    print("=" * 80)
    print("STAGE 3: ZERO-RETRAINING THRESHOLD SENSITIVITY EXPERIMENT")
    print("=" * 80)

    for c_idx in range(n_clients):
        cid = f"client_{c_idx:02d}"
        print(f"\nProcessing {cid}...")

        # 1. Load data
        npz_path = Path(data_dir) / cid / "data.npz"
        data = np.load(npz_path, allow_pickle=True)
        x_train, y_train = data["X_train"], data["y_train"]
        x_val, y_val = data["X_val"], data["y_val"]
        x_test, y_test = data["X_test"], data["y_test"]
        attack_test = data["attack_test"] if "attack_test" in data else np.zeros_like(y_test)

        # 2. Extract strictly benign validation data
        val_benign_mask = (y_val == 0)
        x_val_benign = x_val[val_benign_mask]
        n_benign_val = len(x_val_benign)

        # 3. Load frozen checkpoints
        ae = Autoencoder(input_dim=38)
        ae.load_state_dict(torch.load(f"{ckpt_dir}/{cid}_autoencoder.pt", map_location="cpu"))
        ae.eval()

        gate = GatingRegressor(input_dim=38)
        gate.load_state_dict(torch.load(f"{ckpt_dir}/{cid}_gate.pt", map_location="cpu"))
        gate.eval()

        # 4. Infer validation scores
        with torch.no_grad():
            t_val = torch.tensor(x_val_benign, dtype=torch.float32)
            val_recon = ae(t_val)
            val_mse = Autoencoder.compute_mse(t_val, val_recon).numpy()
            val_e_hat = gate(t_val).squeeze(1).numpy()

        # 5. Compute quantile thresholds & verify monotonicity
        client_thresh = calibrate_quantiles(val_mse, val_e_hat, QUANTILES)

        # Monotonicity checks: Q80 <= Q90 <= Q95 <= Q99
        assert client_thresh["Q80"]["tau_ae"] <= client_thresh["Q90"]["tau_ae"] <= client_thresh["Q95"]["tau_ae"] <= client_thresh["Q99"]["tau_ae"], f"AE Monotonicity violation in {cid}"
        assert client_thresh["Q80"]["tau_gate"] <= client_thresh["Q90"]["tau_gate"] <= client_thresh["Q95"]["tau_gate"] <= client_thresh["Q99"]["tau_gate"], f"Gate Monotonicity violation in {cid}"

        # Record thresholds
        client_thresholds_record[cid] = {
            "client_id": c_idx,
            "n_benign_val": n_benign_val,
            "thresholds": client_thresh,
        }

        # Print threshold summary
        print(f"  Benign val samples: {n_benign_val}")
        for q_name in ["Q99", "Q95", "Q90", "Q80"]:
            print(f"  {q_name}: tau_AE={client_thresh[q_name]['tau_ae']:.6f}, tau_Gate={client_thresh[q_name]['tau_gate']:.6f}")

        # 6. Infer test scores across full test set
        t_test = torch.tensor(x_test, dtype=torch.float32)
        loader = DataLoader(TensorDataset(t_test), batch_size=batch_size, shuffle=False)
        test_mses = []
        test_e_hats = []
        with torch.no_grad():
            for (bx,) in loader:
                recon = ae(bx)
                mse = Autoencoder.compute_mse(bx, recon)
                e_hat = gate(bx)
                test_mses.append(mse.numpy())
                test_e_hats.append(e_hat.squeeze(1).numpy())

        test_mse_arr = np.concatenate(test_mses)
        test_e_hat_arr = np.concatenate(test_e_hats)

        # 7. Construct 3 tiers
        tiers = construct_test_tiers(x_train, x_test, y_test, attack_test)

        # Tier indices
        # Tier 1: all
        # Tier 2: unique test
        df_test = pd.DataFrame(np.round(x_test, 7))
        t2_idx = df_test.drop_duplicates().index.values

        # Tier 3: unique test not in train
        df_train = pd.DataFrame(np.round(x_train, 7)).drop_duplicates()
        df_unique_test = pd.DataFrame(np.round(tiers["tier2"]["x"], 7))
        df_unique_test["_orig_idx"] = np.arange(len(tiers["tier2"]["x"]))
        merged = pd.merge(
            df_unique_test,
            df_train.assign(_in_train=1),
            on=list(range(x_test.shape[1])),
            how="left",
        )
        t3_mask = merged["_in_train"].isna().values
        t3_idx = t2_idx[t3_mask]

        tier_data_map = {
            "tier1": {
                "y": y_test,
                "mse": test_mse_arr,
                "e_hat": test_e_hat_arr,
            },
            "tier2": {
                "y": y_test[t2_idx],
                "mse": test_mse_arr[t2_idx],
                "e_hat": test_e_hat_arr[t2_idx],
            },
            "tier3": {
                "y": y_test[t3_idx],
                "mse": test_mse_arr[t3_idx],
                "e_hat": test_e_hat_arr[t3_idx],
            },
        }

        # Verify subset invariant
        assert len(t3_idx) <= len(t2_idx) <= len(x_test), f"Subset violation in {cid}"

        # 8. Evaluate all quantiles and tiers
        tier3_joint_matrices_record[cid] = {}

        for tier_name in tiers_list:
            t_data = tier_data_map[tier_name]
            y_t = t_data["y"]
            mse_t = t_data["mse"]
            e_t = t_data["e_hat"]

            for q_name in ["Q99", "Q95", "Q90", "Q80"]:
                tau_ae = client_thresh[q_name]["tau_ae"]
                tau_gate = client_thresh[q_name]["tau_gate"]

                metrics = evaluate_tier_metrics(y_t, mse_t, e_t, tau_ae, tau_gate)

                # Record flat row for CSV
                row = {
                    "client": cid,
                    "tier": tier_name,
                    "quantile": q_name,
                    "quantile_val": client_thresh[q_name]["quantile"],
                    "n_benign_val": n_benign_val,
                    "tau_ae": tau_ae,
                    "tau_gate": tau_gate,
                    "total_samples": metrics["n_samples"],
                    "benign_samples": metrics["n_benign"],
                    "attack_samples": metrics["n_attack"],
                    "ae_recall": metrics["ae"]["recall"],
                    "ae_fpr": metrics["ae"]["fpr"],
                    "ae_tp": metrics["ae"]["tp"],
                    "ae_fp": metrics["ae"]["fp"],
                    "ae_tn": metrics["ae"]["tn"],
                    "ae_fn": metrics["ae"]["fn"],
                    "gate_passage_recall": metrics["gate"]["passage_recall"],
                    "gate_benign_fpr": metrics["gate"]["benign_fpr"],
                    "gate_tp": metrics["gate"]["tp"],
                    "gate_fp": metrics["gate"]["fp"],
                    "gate_tn": metrics["gate"]["tn"],
                    "gate_fn": metrics["gate"]["fn"],
                    "heavy_path_invocation_rate_pct": metrics["gate"]["heavy_path_invocation_rate_pct"],
                    "filtered_pct": metrics["gate"]["filtered_pct"],
                    "attack_concentration_pct": metrics["gate"]["attack_concentration_pct"],
                    "gate_total_passed": metrics["gate"]["total_passed"],
                    "gate_benign_passed": metrics["gate"]["benign_passed"],
                    "gate_attack_passed": metrics["gate"]["attack_passed"],
                    "gate_total_suppressed": metrics["gate"]["total_suppressed"],
                    "gate_benign_suppressed": metrics["gate"]["benign_suppressed"],
                    "gate_attack_suppressed": metrics["gate"]["attack_suppressed"],
                    "gate_ae_agreement_pct": metrics["agreement"]["overall_pct"],
                    "gate_ae_benign_agreement_pct": metrics["agreement"]["benign_pct"],
                    "gate_ae_attack_agreement_pct": metrics["agreement"]["attack_pct"],
                }
                all_client_rows.append(row)

                # Accumulate for pooled metrics
                acc = pooled_accum[(tier_name, q_name)]
                acc["n_samples"] += metrics["n_samples"]
                acc["n_benign"] += metrics["n_benign"]
                acc["n_attack"] += metrics["n_attack"]
                acc["ae_tp"] += metrics["ae"]["tp"]
                acc["ae_fp"] += metrics["ae"]["fp"]
                acc["ae_tn"] += metrics["ae"]["tn"]
                acc["ae_fn"] += metrics["ae"]["fn"]
                acc["gate_tp"] += metrics["gate"]["tp"]
                acc["gate_fp"] += metrics["gate"]["fp"]
                acc["gate_tn"] += metrics["gate"]["tn"]
                acc["gate_fn"] += metrics["gate"]["fn"]
                acc["agree_overall_count"] += metrics["agreement"]["overall_count"]
                acc["agree_benign_count"] += metrics["agreement"]["benign_count"]
                acc["agree_attack_count"] += metrics["agreement"]["attack_count"]

                # Joint matrix accumulation
                jm = metrics["joint_matrix"]
                acc["joint_both"] += jm["both_trigger"]
                acc["joint_ae_only"] += jm["ae_only"]
                acc["joint_gate_only"] += jm["gate_only"]
                acc["joint_neither"] += jm["neither"]
                acc["joint_both_b"] += jm["benign"]["both_trigger"]
                acc["joint_ae_only_b"] += jm["benign"]["ae_only"]
                acc["joint_gate_only_b"] += jm["benign"]["gate_only"]
                acc["joint_neither_b"] += jm["benign"]["neither"]
                acc["joint_both_a"] += jm["attack"]["both_trigger"]
                acc["joint_ae_only_a"] += jm["attack"]["ae_only"]
                acc["joint_gate_only_a"] += jm["attack"]["gate_only"]
                acc["joint_neither_a"] += jm["attack"]["neither"]

                if tier_name == "tier3":
                    tier3_joint_matrices_record[cid][q_name] = jm

    # Save Client Thresholds JSON
    with open(out_path / "sensitivity_client_thresholds.json", "w") as f:
        json.dump(client_thresholds_record, f, indent=2)

    # Convert client rows to DataFrame and save CSV
    df_clients = pd.DataFrame(all_client_rows)
    df_clients.to_csv(out_path / "sensitivity_per_client_metrics.csv", index=False)
    print(f"\nSaved {out_path / 'sensitivity_per_client_metrics.csv'}")

    # Compute Sample-Weighted Pooled and Macro Summaries
    pooled_summary = {
        "sample_weighted_pooled": {},
        "macro_client_averaging": {},
    }

    tier3_joint_confusion_matrices_all = {
        "per_client": tier3_joint_matrices_record,
        "pooled": {},
    }

    tier3_joint_rows = []

    for tier_name in tiers_list:
        pooled_summary["sample_weighted_pooled"][tier_name] = {}
        pooled_summary["macro_client_averaging"][tier_name] = {}

        for q_name in ["Q99", "Q95", "Q90", "Q80"]:
            acc = pooled_accum[(tier_name, q_name)]

            n_tot = acc["n_samples"]
            n_b = acc["n_benign"]
            n_a = acc["n_attack"]

            ae_rec = float(acc["ae_tp"] / n_a) if n_a > 0 else 0.0
            ae_fpr = float(acc["ae_fp"] / n_b) if n_b > 0 else 0.0

            gate_rec = float(acc["gate_tp"] / n_a) if n_a > 0 else 0.0
            gate_fpr = float(acc["gate_fp"] / n_b) if n_b > 0 else 0.0

            passed = acc["gate_tp"] + acc["gate_fp"]
            suppressed = acc["gate_tn"] + acc["gate_fn"]

            hp_inv_rate = float(passed / n_tot * 100.0) if n_tot > 0 else 0.0
            filtered = float(suppressed / n_tot * 100.0) if n_tot > 0 else 0.0
            concentration = float(acc["gate_tp"] / passed * 100.0) if passed > 0 else 0.0

            agree_overall = float(acc["agree_overall_count"] / n_tot * 100.0) if n_tot > 0 else 0.0
            agree_b = float(acc["agree_benign_count"] / n_b * 100.0) if n_b > 0 else 0.0
            agree_a = float(acc["agree_attack_count"] / n_a * 100.0) if n_a > 0 else 0.0

            pooled_summary["sample_weighted_pooled"][tier_name][q_name] = {
                "n_samples": n_tot,
                "n_benign": n_b,
                "n_attack": n_a,
                "ae": {
                    "tp": acc["ae_tp"],
                    "fp": acc["ae_fp"],
                    "tn": acc["ae_tn"],
                    "fn": acc["ae_fn"],
                    "recall": ae_rec,
                    "fpr": ae_fpr,
                },
                "gate": {
                    "tp": acc["gate_tp"],
                    "fp": acc["gate_fp"],
                    "tn": acc["gate_tn"],
                    "fn": acc["gate_fn"],
                    "passage_recall": gate_rec,
                    "benign_fpr": gate_fpr,
                    "heavy_path_invocation_rate_pct": hp_inv_rate,
                    "filtered_pct": filtered,
                    "attack_concentration_pct": concentration,
                    "total_passed": passed,
                    "benign_passed": acc["gate_fp"],
                    "attack_passed": acc["gate_tp"],
                    "total_suppressed": suppressed,
                    "benign_suppressed": acc["gate_tn"],
                    "attack_suppressed": acc["gate_fn"],
                },
                "agreement": {
                    "overall_pct": agree_overall,
                    "benign_pct": agree_b,
                    "attack_pct": agree_a,
                },
                "joint_matrix": {
                    "both_trigger": acc["joint_both"],
                    "ae_only": acc["joint_ae_only"],
                    "gate_only": acc["joint_gate_only"],
                    "neither": acc["joint_neither"],
                    "benign": {
                        "both_trigger": acc["joint_both_b"],
                        "ae_only": acc["joint_ae_only_b"],
                        "gate_only": acc["joint_gate_only_b"],
                        "neither": acc["joint_neither_b"],
                    },
                    "attack": {
                        "both_trigger": acc["joint_both_a"],
                        "ae_only": acc["joint_ae_only_a"],
                        "gate_only": acc["joint_gate_only_a"],
                        "neither": acc["joint_neither_a"],
                    },
                },
            }

            if tier_name == "tier3":
                tier3_joint_confusion_matrices_all["pooled"][q_name] = pooled_summary["sample_weighted_pooled"][tier_name][q_name]["joint_matrix"]
                tier3_joint_rows.append({
                    "entity": "POOLED",
                    "quantile": q_name,
                    "total_samples": n_tot,
                    "both_trigger": acc["joint_both"],
                    "ae_only": acc["joint_ae_only"],
                    "gate_only": acc["joint_gate_only"],
                    "neither": acc["joint_neither"],
                    "benign_both": acc["joint_both_b"],
                    "benign_ae_only": acc["joint_ae_only_b"],
                    "benign_gate_only": acc["joint_gate_only_b"],
                    "benign_neither": acc["joint_neither_b"],
                    "attack_both": acc["joint_both_a"],
                    "attack_ae_only": acc["joint_ae_only_a"],
                    "attack_gate_only": acc["joint_gate_only_a"],
                    "attack_neither": acc["joint_neither_a"],
                    "benign_agree_pct": agree_b,
                    "attack_agree_pct": agree_a,
                    "overall_agree_pct": agree_overall,
                })

            # Macro averaging across clients for this tier and quantile
            sub = df_clients[(df_clients["tier"] == tier_name) & (df_clients["quantile"] == q_name)]
            pooled_summary["macro_client_averaging"][tier_name][q_name] = {
                "ae_recall_macro": float(sub["ae_recall"].mean()),
                "ae_fpr_macro": float(sub["ae_fpr"].mean()),
                "gate_passage_recall_macro": float(sub["gate_passage_recall"].mean()),
                "gate_benign_fpr_macro": float(sub["gate_benign_fpr"].mean()),
                "heavy_path_invocation_rate_macro_pct": float(sub["heavy_path_invocation_rate_pct"].mean()),
                "filtered_macro_pct": float(sub["filtered_pct"].mean()),
                "attack_concentration_macro_pct": float(sub["attack_concentration_pct"].mean()),
                "gate_ae_agreement_macro_pct": float(sub["gate_ae_agreement_pct"].mean()),
                "gate_ae_benign_agreement_macro_pct": float(sub["gate_ae_benign_agreement_pct"].mean()),
                "gate_ae_attack_agreement_macro_pct": float(sub["gate_ae_attack_agreement_pct"].mean()),
            }

    # Add per-client tier 3 joint matrix rows
    for cid in sorted(tier3_joint_matrices_record.keys()):
        for q_name in ["Q99", "Q95", "Q90", "Q80"]:
            jm = tier3_joint_matrices_record[cid][q_name]
            sub = df_clients[(df_clients["client"] == cid) & (df_clients["tier"] == "tier3") & (df_clients["quantile"] == q_name)].iloc[0]
            tier3_joint_rows.append({
                "entity": cid,
                "quantile": q_name,
                "total_samples": int(sub["total_samples"]),
                "both_trigger": jm["both_trigger"],
                "ae_only": jm["ae_only"],
                "gate_only": jm["gate_only"],
                "neither": jm["neither"],
                "benign_both": jm["benign"]["both_trigger"],
                "benign_ae_only": jm["benign"]["ae_only"],
                "benign_gate_only": jm["benign"]["gate_only"],
                "benign_neither": jm["benign"]["neither"],
                "attack_both": jm["attack"]["both_trigger"],
                "attack_ae_only": jm["attack"]["ae_only"],
                "attack_gate_only": jm["attack"]["gate_only"],
                "attack_neither": jm["attack"]["neither"],
                "benign_agree_pct": float(sub["gate_ae_benign_agreement_pct"]),
                "attack_agree_pct": float(sub["gate_ae_attack_agreement_pct"]),
                "overall_agree_pct": float(sub["gate_ae_agreement_pct"]),
            })

    # Save Tier 3 joint confusion matrices CSV & JSON
    pd.DataFrame(tier3_joint_rows).to_csv(out_path / "tier3_joint_confusion_matrices.csv", index=False)
    with open(out_path / "tier3_joint_confusion_matrices.json", "w") as f:
        json.dump(tier3_joint_confusion_matrices_all, f, indent=2)

    # Save Pooled Summary JSON
    with open(out_path / "sensitivity_pooled_summary.json", "w") as f:
        json.dump(pooled_summary, f, indent=2)
    print(f"Saved {out_path / 'sensitivity_pooled_summary.json'}")

    # 9. Generate Trade-off Plots
    print("\nGenerating Trade-off plots...")
    generate_tradeoff_plots(pooled_summary, df_clients, out_path)

    print("\nExperiment completed successfully!")


def generate_tradeoff_plots(pooled_summary: Dict[str, Any], df_clients: pd.DataFrame, out_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    quantiles_labels = ["Q99", "Q95", "Q90", "Q80"]

    # -------------------------------------------------------------
    # Plot 1: Attack Passage Recall vs Heavy Path Invocation Rate
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Tier 1 Gate
    hp_t1_gate = [pooled_summary["sample_weighted_pooled"]["tier1"][q]["gate"]["heavy_path_invocation_rate_pct"] for q in quantiles_labels]
    rec_t1_gate = [pooled_summary["sample_weighted_pooled"]["tier1"][q]["gate"]["passage_recall"] * 100 for q in quantiles_labels]
    
    # Tier 3 Gate
    hp_t3_gate = [pooled_summary["sample_weighted_pooled"]["tier3"][q]["gate"]["heavy_path_invocation_rate_pct"] for q in quantiles_labels]
    rec_t3_gate = [pooled_summary["sample_weighted_pooled"]["tier3"][q]["gate"]["passage_recall"] * 100 for q in quantiles_labels]

    # Tier 1 AE (for comparison)
    rec_t1_ae = [pooled_summary["sample_weighted_pooled"]["tier1"][q]["ae"]["recall"] * 100 for q in quantiles_labels]
    rec_t3_ae = [pooled_summary["sample_weighted_pooled"]["tier3"][q]["ae"]["recall"] * 100 for q in quantiles_labels]

    ax.plot(hp_t1_gate, rec_t1_gate, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Gate Passage (Tier 1 Full Test)")
    ax.plot(hp_t3_gate, rec_t3_gate, "s--", color="#ff7f0e", linewidth=2.5, markersize=8, label="Gate Passage (Tier 3 Train-Exclusive)")
    ax.plot(hp_t1_gate, rec_t1_ae, "^:", color="#2ca02c", linewidth=2.0, markersize=8, label="Autoencoder (Tier 1 Full Test)")
    ax.plot(hp_t3_gate, rec_t3_ae, "d-.", color="#d62728", linewidth=2.0, markersize=8, label="Autoencoder (Tier 3 Train-Exclusive)")

    for i, q in enumerate(quantiles_labels):
        ax.annotate(f"{q}", (hp_t1_gate[i], rec_t1_gate[i]), textcoords="offset points", xytext=(-10, 10), fontweight="bold", color="#1f77b4")
        ax.annotate(f"{q}", (hp_t3_gate[i], rec_t3_gate[i]), textcoords="offset points", xytext=(8, -12), fontweight="bold", color="#ff7f0e")

    ax.set_xlabel("Heavy Path Invocation Rate (%) [Passed to GRU Classifier]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attack Passage / Detection Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title("Trade-off: Attack Passage Recall vs. Heavy Path Invocation Rate", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path / "tradeoff_recall_vs_heavypath.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 2: Benign False Positive Rate across Tiers (FPR Amplification)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    x_indices = np.arange(len(quantiles_labels))
    width = 0.25

    fpr_t1 = [pooled_summary["sample_weighted_pooled"]["tier1"][q]["gate"]["benign_fpr"] * 100 for q in quantiles_labels]
    fpr_t2 = [pooled_summary["sample_weighted_pooled"]["tier2"][q]["gate"]["benign_fpr"] * 100 for q in quantiles_labels]
    fpr_t3 = [pooled_summary["sample_weighted_pooled"]["tier3"][q]["gate"]["benign_fpr"] * 100 for q in quantiles_labels]

    rects1 = ax.bar(x_indices - width, fpr_t1, width, label="Tier 1 (Full Test)", color="#3498db")
    rects2 = ax.bar(x_indices, fpr_t2, width, label="Tier 2 (Dedup Unique)", color="#9b59b6")
    rects3 = ax.bar(x_indices + width, fpr_t3, width, label="Tier 3 (Train-Exclusive)", color="#e74c3c")

    ax.set_xlabel("Calibration Quantile", fontsize=12, fontweight="bold")
    ax.set_ylabel("Benign False Positive Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Benign False Positive Rate across Evaluation Tiers", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(quantiles_labels, fontweight="bold")
    ax.legend(frameon=True, fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Label heights
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path / "tradeoff_fpr_across_tiers.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 3: Downstream Traffic Characteristics (Attack Concentration)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    conc_t1 = [pooled_summary["sample_weighted_pooled"]["tier1"][q]["gate"]["attack_concentration_pct"] for q in quantiles_labels]
    conc_t3 = [pooled_summary["sample_weighted_pooled"]["tier3"][q]["gate"]["attack_concentration_pct"] for q in quantiles_labels]

    ax.plot(quantiles_labels, conc_t1, "o-", color="#27ae60", linewidth=2.5, markersize=8, label="Tier 1 (Full Test)")
    ax.plot(quantiles_labels, conc_t3, "s--", color="#e67e22", linewidth=2.5, markersize=8, label="Tier 3 (Train-Exclusive Unique)")

    # Baseline input attack prevalence
    t1_prevalence = pooled_summary["sample_weighted_pooled"]["tier1"]["Q99"]["n_attack"] / pooled_summary["sample_weighted_pooled"]["tier1"]["Q99"]["n_samples"] * 100
    t3_prevalence = pooled_summary["sample_weighted_pooled"]["tier3"]["Q99"]["n_attack"] / pooled_summary["sample_weighted_pooled"]["tier3"]["Q99"]["n_samples"] * 100

    ax.axhline(t1_prevalence, color="#27ae60", linestyle=":", alpha=0.7, label=f"Tier 1 Raw Prevalence ({t1_prevalence:.2f}%)")
    ax.axhline(t3_prevalence, color="#e67e22", linestyle=":", alpha=0.7, label=f"Tier 3 Raw Prevalence ({t3_prevalence:.2f}%)")

    for i, q in enumerate(quantiles_labels):
        ax.annotate(f"{conc_t1[i]:.2f}%", (q, conc_t1[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color="#27ae60")
        ax.annotate(f"{conc_t3[i]:.2f}%", (q, conc_t3[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", color="#e67e22")

    ax.set_xlabel("Calibration Quantile", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attack Concentration in Passed Traffic (%)", fontsize=12, fontweight="bold")
    ax.set_title("Attack Enrichment Ratio in Heavy Path Traffic", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path / "tradeoff_attack_concentration.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 4: Per-Client Comparative Passage Recall (Highlighting 02 and 04)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Filter Tier 1
    t1_clients = df_clients[df_clients["tier"] == "tier1"]
    
    for cid in sorted(t1_clients["client"].unique()):
        sub_c = t1_clients[t1_clients["client"] == cid]
        # Sort by quantile order: Q99, Q95, Q90, Q80
        sub_c = sub_c.set_index("quantile").loc[quantiles_labels].reset_index()
        
        recs = sub_c["gate_passage_recall"] * 100.0
        
        if cid == "client_02":
            ax.plot(quantiles_labels, recs, "o-", color="#e74c3c", linewidth=3.5, markersize=10, label="Client 02 (64% attack skew, small N_b)", zorder=5)
        elif cid == "client_04":
            ax.plot(quantiles_labels, recs, "s-", color="#e67e22", linewidth=3.5, markersize=9, label="Client 04 (Moderate attack skew)", zorder=4)
        else:
            ax.plot(quantiles_labels, recs, ".-", color="#95a5a6", linewidth=1.2, alpha=0.6, label="Other Standard Non-IID Clients" if cid == "client_00" else "")

    ax.set_xlabel("Calibration Quantile", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gate Attack Passage Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title("Per-Client Gate Attack Passage Recall Across Quantiles (Tier 1)", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=11, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path / "client_comparative_recall.png")
    plt.close(fig)


if __name__ == "__main__":
    run_threshold_sensitivity_experiment()


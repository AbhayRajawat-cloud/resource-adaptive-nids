"""
Performance evaluation metrics and reporting for Stage 3.

Computes:
- Gate Regression Performance: Huber Loss, RMSE, R^2 score, Spearman rank correlation (rho).
- AE Reconstruction Error Statistics: Mean, Std, Median MSE for Benign vs. Attack flows.
- AE Anomaly Detection Metrics: Precision, Recall, F1-Score, False Positive Rate (FPR) at threshold tau_AE_k.
- Gate Routing & Hypothesis H1 Metrics: Heavy Path Trigger Rate (% where e_hat >= tau_gate_k),
  Attack Passage Recall (% of attack flows triggering Heavy Path),
  Benign Passage Rate (% of benign flows triggering Heavy Path).
- Pipeline Agreement Metric: Gate-AE Agreement Rate (% of test flows where Gate routing decision
  matches AE decision).
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn.functional as F


def compute_stage3_tier_metrics(
    y_true: np.ndarray,
    mse_ae: np.ndarray,
    e_hat_gate: np.ndarray,
    tau_ae: float,
    tau_gate: float,
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics for a single test tier.
    
    Args:
        y_true: Ground truth binary labels (0=Benign, 1=Attack)
        mse_ae: Reconstructed MSE values from Autoencoder
        e_hat_gate: Predicted log-MSE values from Gating Regressor
        tau_ae: Calibrated AE threshold
        tau_gate: Calibrated Gate threshold
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return {"error": "Empty sample set"}

    n_benign = int((y_true == 0).sum())
    n_attack = int((y_true == 1).sum())

    # Target teacher log-MSE
    y_gate_teacher = np.log(mse_ae + 1e-6)

    # 1. Gate Regression Performance
    rmse = float(np.sqrt(mean_squared_error(y_gate_teacher, e_hat_gate)))
    r2 = float(r2_score(y_gate_teacher, e_hat_gate))
    # Smooth L1 / Huber loss
    t_err = torch.tensor(e_hat_gate) - torch.tensor(y_gate_teacher)
    huber = float(F.smooth_l1_loss(torch.tensor(e_hat_gate), torch.tensor(y_gate_teacher), beta=1.0).item())
    
    # Spearman rank correlation
    if np.all(e_hat_gate == e_hat_gate[0]) or np.all(y_gate_teacher == y_gate_teacher[0]):
        spearman_rho = 0.0
    else:
        spearman_rho = float(spearmanr(y_gate_teacher, e_hat_gate).correlation)
        if np.isnan(spearman_rho):
            spearman_rho = 0.0

    # 2. AE Reconstruction Statistics (Benign vs Attack)
    benign_mask = (y_true == 0)
    attack_mask = (y_true == 1)

    benign_mse = mse_ae[benign_mask] if n_benign > 0 else np.array([])
    attack_mse = mse_ae[attack_mask] if n_attack > 0 else np.array([])

    ae_stats = {
        "benign_mean_mse": float(np.mean(benign_mse)) if len(benign_mse) > 0 else 0.0,
        "benign_std_mse": float(np.std(benign_mse)) if len(benign_mse) > 0 else 0.0,
        "benign_median_mse": float(np.median(benign_mse)) if len(benign_mse) > 0 else 0.0,
        "attack_mean_mse": float(np.mean(attack_mse)) if len(attack_mse) > 0 else 0.0,
        "attack_std_mse": float(np.std(attack_mse)) if len(attack_mse) > 0 else 0.0,
        "attack_median_mse": float(np.median(attack_mse)) if len(attack_mse) > 0 else 0.0,
    }

    # 3. AE Anomaly Detection Performance at tau_ae
    y_pred_ae = (mse_ae >= tau_ae).astype(int)
    
    if n_attack > 0 and n_benign > 0:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_ae, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred_ae, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    else:
        prec = float(np.mean(y_true == y_pred_ae))
        rec = 1.0 if n_attack == 0 else 0.0
        f1 = 0.0
        fpr = 0.0

    # 4. Gate Routing Decisions & Hypothesis H1
    gate_decision = (e_hat_gate >= tau_gate).astype(int)
    heavy_path_trigger_rate = float(np.mean(gate_decision)) * 100.0

    if n_attack > 0:
        attack_passage_recall = float(np.mean(gate_decision[attack_mask])) * 100.0
    else:
        attack_passage_recall = 0.0

    if n_benign > 0:
        benign_passage_rate = float(np.mean(gate_decision[benign_mask])) * 100.0
    else:
        benign_passage_rate = 0.0

    # 5. Pipeline Agreement Metric: Gate-AE Agreement Rate
    ae_decision = (mse_ae >= tau_ae).astype(int)
    agreement_rate = float(np.mean(gate_decision == ae_decision)) * 100.0

    return {
        "n_samples": n_samples,
        "n_benign": n_benign,
        "n_attack": n_attack,
        "gate_regression": {
            "huber_loss": huber,
            "rmse": rmse,
            "r2_score": r2,
            "spearman_rho": spearman_rho,
        },
        "ae_reconstruction_stats": ae_stats,
        "ae_classification": {
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "false_positive_rate": float(fpr),
        },
        "gate_routing": {
            "heavy_path_trigger_rate_pct": heavy_path_trigger_rate,
            "attack_passage_recall_pct": attack_passage_recall,
            "benign_passage_rate_pct": benign_passage_rate,
            "h1_benchmark_met_70pct": bool(attack_passage_recall >= 70.0),
        },
        "agreement": {
            "gate_ae_agreement_rate_pct": agreement_rate,
        },
    }

"""
Three-tier test set construction and evaluation runner for Stage 3.

Tiers:
  - Tier 1 (Full Test Set): Complete test split (x_test, y_test).
  - Tier 2 (Deduplicated Unique Test Set): Unique 38-D feature vectors in x_test.
  - Tier 3 (Training-Exclusive Unique Test Set): Unique 38-D feature vectors in x_test
    that DO NOT appear in x_train.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.autoencoder import Autoencoder
from src.models.gate import GatingRegressor
from src.evaluation.metrics import compute_stage3_tier_metrics


def construct_test_tiers(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    attack_test: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Construct Tier 1, Tier 2, and Tier 3 test sets for a single client.
    """
    # Tier 1: Full Test Set
    tier1 = {
        "x": x_test,
        "y": y_test,
        "attack": attack_test,
        "n_samples": len(x_test),
    }

    # Tier 2: Deduplicated Unique in x_test
    # Round to avoid floating point hash issues
    df_test = pd.DataFrame(np.round(x_test, 7))
    unique_test_idx = df_test.drop_duplicates().index.values

    tier2 = {
        "x": x_test[unique_test_idx],
        "y": y_test[unique_test_idx],
        "attack": attack_test[unique_test_idx],
        "n_samples": len(unique_test_idx),
    }

    # Tier 3: Unique in x_test that DO NOT appear in x_train
    df_train = pd.DataFrame(np.round(x_train, 7))
    # Deduplicate train first to make lookup efficient
    unique_train = df_train.drop_duplicates()
    
    # Check membership using merge/indicator
    df_unique_test = pd.DataFrame(np.round(tier2["x"], 7))
    df_unique_test["_orig_idx"] = np.arange(len(tier2["x"]))

    merged = pd.merge(
        df_unique_test,
        unique_train.assign(_in_train=1),
        on=list(range(x_test.shape[1])),
        how="left",
    )
    
    # Rows with NaN in _in_train are training-exclusive
    tier3_mask = merged["_in_train"].isna().values
    tier3_idx = merged["_orig_idx"].values[tier3_mask]

    tier3 = {
        "x": tier2["x"][tier3_idx],
        "y": tier2["y"][tier3_idx],
        "attack": tier2["attack"][tier3_idx],
        "n_samples": len(tier3_idx),
        "retained_pct_of_tier1": float(len(tier3_idx) / len(x_test) * 100.0),
        "retained_pct_of_tier2": float(len(tier3_idx) / len(unique_test_idx) * 100.0) if len(unique_test_idx) > 0 else 0.0,
    }

    return {"tier1": tier1, "tier2": tier2, "tier3": tier3}


def evaluate_client_models(
    model_ae: Autoencoder,
    model_gate: GatingRegressor,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    attack_test: np.ndarray,
    tau_ae: float,
    tau_gate: float,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """
    Run inference across all test tiers and return complete evaluation metrics.
    """
    model_ae.eval()
    model_gate.eval()

    # Pre-compute full test inference first
    t_test = torch.tensor(x_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(t_test), batch_size=batch_size, shuffle=False)

    full_mses = []
    full_e_hats = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon = model_ae(batch_x)
            mse = Autoencoder.compute_mse(batch_x, recon)
            e_hat = model_gate(batch_x)

            full_mses.append(mse.cpu().numpy())
            full_e_hats.append(e_hat.squeeze(1).cpu().numpy())

    full_mses_arr = np.concatenate(full_mses)
    full_e_hats_arr = np.concatenate(full_e_hats)

    # Construct tiers
    tiers = construct_test_tiers(x_train, x_test, y_test, attack_test)

    # Map MSE and e_hat to each tier
    results = {}

    # Tier 1
    results["tier1"] = compute_stage3_tier_metrics(
        y_true=tiers["tier1"]["y"],
        mse_ae=full_mses_arr,
        e_hat_gate=full_e_hats_arr,
        tau_ae=tau_ae,
        tau_gate=tau_gate,
    )
    results["tier1"]["tier_info"] = {"n_samples": tiers["tier1"]["n_samples"]}

    # Tier 2
    # Find indices for tier 2
    df_test = pd.DataFrame(np.round(x_test, 7))
    unique_test_idx = df_test.drop_duplicates().index.values

    results["tier2"] = compute_stage3_tier_metrics(
        y_true=tiers["tier2"]["y"],
        mse_ae=full_mses_arr[unique_test_idx],
        e_hat_gate=full_e_hats_arr[unique_test_idx],
        tau_ae=tau_ae,
        tau_gate=tau_gate,
    )
    results["tier2"]["tier_info"] = {
        "n_samples": tiers["tier2"]["n_samples"],
        "pct_of_tier1": float(tiers["tier2"]["n_samples"] / len(x_test) * 100.0),
    }

    # Tier 3
    if tiers["tier3"]["n_samples"] > 0:
        # Match indices in tier 2
        df_train = pd.DataFrame(np.round(x_train, 7)).drop_duplicates()
        df_unique_test = pd.DataFrame(np.round(tiers["tier2"]["x"], 7))
        df_unique_test["_orig_idx"] = np.arange(len(tiers["tier2"]["x"]))
        merged = pd.merge(
            df_unique_test,
            df_train.assign(_in_train=1),
            on=list(range(x_test.shape[1])),
            how="left",
        )
        tier3_mask = merged["_in_train"].isna().values
        tier3_sub_idx = unique_test_idx[tier3_mask]

        results["tier3"] = compute_stage3_tier_metrics(
            y_true=tiers["tier3"]["y"],
            mse_ae=full_mses_arr[tier3_sub_idx],
            e_hat_gate=full_e_hats_arr[tier3_sub_idx],
            tau_ae=tau_ae,
            tau_gate=tau_gate,
        )
        results["tier3"]["tier_info"] = {
            "n_samples": tiers["tier3"]["n_samples"],
            "pct_of_tier1": tiers["tier3"]["retained_pct_of_tier1"],
            "pct_of_tier2": tiers["tier3"]["retained_pct_of_tier2"],
        }
    else:
        results["tier3"] = {
            "error": "No training-exclusive test samples remaining",
            "tier_info": {"n_samples": 0, "pct_of_tier1": 0.0, "pct_of_tier2": 0.0},
        }

    # Store raw test predictions for global pooling across clients
    results["raw_test_preds"] = {
        "y_true": y_test,
        "mse_ae": full_mses_arr,
        "e_hat_gate": full_e_hats_arr,
    }

    return results


"""
Stage 5: Gate-Passed Dataset Extraction & Three-Tier Lineage Preservation.

Extracts 38-dimensional feature vectors that pass the Stage 3 lightweight Gating Regressor
(g_phi(x) >= tau_Gate) across all 10 federated clients under both Q0.99 (baseline) and Q0.95 (candidate).

Constructs Three-Tier test evaluation lineage masks:
- Tier 1: Complete Gate-passed test set (all N_test_passed samples).
- Tier 2: Unique Gate-passed test vectors (first occurrence of each unique 38-D vector).
- Tier 3: Training-exclusive unique test vectors (unique in test AND strictly absent from train and val).

Outputs:
- data/processed/gate_passed/Q99/client_{k:02d}_passed.npz
- data/processed/gate_passed/Q95/client_{k:02d}_passed.npz
- data/processed/gate_passed/extraction_summary.json
- data/processed/gate_passed/extraction_summary.csv
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.gate import GatingRegressor


def compute_three_tier_masks(
    x_test_passed: np.ndarray,
    x_train_raw: np.ndarray,
    x_val_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Compute boolean lineage masks for Tier 1, Tier 2, and Tier 3 on the Gate-passed test partition.
    
    Tier 1: Complete test set (N_test_passed).
    Tier 2: Unique feature vectors in test (first occurrence).
    Tier 3: Training-exclusive unique test vectors (strictly absent from train and val).
    
    Returns:
        tier2_mask_test: boolean array of shape [N_test_passed]
        tier3_mask_test: boolean array of shape [N_test_passed]
        counts: dictionary with tier1_test_count, tier2_test_count, tier3_test_count
    """
    n_test_passed = len(x_test_passed)
    if n_test_passed == 0:
        empty_mask = np.zeros(0, dtype=bool)
        return empty_mask, empty_mask, {
            "tier1_test_count": 0,
            "tier2_test_count": 0,
            "tier3_test_count": 0,
        }

    dim = x_test_passed.shape[1]
    dtype_void = np.dtype((np.void, x_test_passed.dtype.itemsize * dim))

    void_test = np.ascontiguousarray(x_test_passed).view(dtype_void).ravel()
    void_train = np.ascontiguousarray(x_train_raw).view(dtype_void).ravel()
    void_val = np.ascontiguousarray(x_val_raw).view(dtype_void).ravel()

    # Tier 2: Unique test vectors (mark first occurrence)
    _, u_idx = np.unique(void_test, return_index=True)
    tier2_mask = np.zeros(n_test_passed, dtype=bool)
    tier2_mask[u_idx] = True

    # Tier 3: Unique in test AND strictly absent from train and val
    void_train_val = np.unique(np.concatenate([void_train, void_val]))
    in_train_val = np.isin(void_test, void_train_val)
    tier3_mask = tier2_mask & (~in_train_val)

    # Invariant: Tier 3 is a strict subset of Tier 2
    assert np.all(tier3_mask <= tier2_mask), "Invariant violated: tier3_mask must imply tier2_mask"

    counts = {
        "tier1_test_count": int(n_test_passed),
        "tier2_test_count": int(np.sum(tier2_mask)),
        "tier3_test_count": int(np.sum(tier3_mask)),
    }

    return tier2_mask, tier3_mask, counts


def extract_gate_passed_for_client(
    client_id: int,
    quantile_name: str,
    stage2_dir: Path,
    ckpt_dir: Path,
    thresholds_dict: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Extract Gate-passed feature matrices, ground truth, and original indices for a client.
    """
    cid_str = f"client_{client_id:02d}"
    data_npz_path = stage2_dir / cid_str / "data.npz"
    gate_ckpt_path = ckpt_dir / f"{cid_str}_gate.pt"

    # 1. Load Stage 2 data arrays
    data = np.load(data_npz_path, allow_pickle=True)
    x_train = data["X_train"] if "X_train" in data else data["x_train"]
    y_train = data["y_train"]
    orig_idx_train = data["orig_idx_train"]

    x_val = data["X_val"] if "X_val" in data else data["x_val"]
    y_val = data["y_val"]
    orig_idx_val = data["orig_idx_val"]

    x_test = data["X_test"] if "X_test" in data else data["x_test"]
    y_test = data["y_test"]
    orig_idx_test = data["orig_idx_test"]

    n_train_raw = len(x_train)
    n_val_raw = len(x_val)
    n_test_raw = len(x_test)

    # 2. Load Gate model and threshold
    gate = GatingRegressor(input_dim=38).eval()
    gate.load_state_dict(torch.load(gate_ckpt_path, map_location="cpu"))

    tau_gate = float(thresholds_dict[cid_str]["thresholds"][quantile_name]["tau_gate"])

    # 3. Predict Gate anomaly scores independently
    with torch.no_grad():
        preds_train = gate(torch.tensor(x_train, dtype=torch.float32)).squeeze(1).numpy()
        preds_val = gate(torch.tensor(x_val, dtype=torch.float32)).squeeze(1).numpy()
        preds_test = gate(torch.tensor(x_test, dtype=torch.float32)).squeeze(1).numpy()

    # 4. Construct boolean extraction masks
    mask_train = preds_train >= tau_gate
    mask_val = preds_val >= tau_gate
    mask_test = preds_test >= tau_gate

    # 5. Extract Gate-passed samples
    x_train_passed = x_train[mask_train]
    y_train_passed = y_train[mask_train]
    orig_idx_train_passed = orig_idx_train[mask_train]

    x_val_passed = x_val[mask_val]
    y_val_passed = y_val[mask_val]
    orig_idx_val_passed = orig_idx_val[mask_val]

    x_test_passed = x_test[mask_test]
    y_test_passed = y_test[mask_test]
    orig_idx_test_passed = orig_idx_test[mask_test]

    n_train_passed = len(x_train_passed)
    n_val_passed = len(x_val_passed)
    n_test_passed = len(x_test_passed)

    # 6. Verify pairwise partition index exclusivity
    set_tr = set(orig_idx_train_passed.tolist())
    set_val = set(orig_idx_val_passed.tolist())
    set_te = set(orig_idx_test_passed.tolist())

    assert len(set_tr.intersection(set_val)) == 0, f"{cid_str} Train-Val index intersection non-empty"
    assert len(set_tr.intersection(set_te)) == 0, f"{cid_str} Train-Test index intersection non-empty"
    assert len(set_val.intersection(set_te)) == 0, f"{cid_str} Val-Test index intersection non-empty"

    # 7. Compute Three-Tier evaluation lineage masks for test partition
    tier2_mask_test, tier3_mask_test, tier_counts = compute_three_tier_masks(
        x_test_passed=x_test_passed,
        x_train_raw=x_train,
        x_val_raw=x_val,
    )

    # 8. Compute passage statistics
    p_rate_train = float(n_train_passed / n_train_raw) if n_train_raw > 0 else 0.0
    p_rate_val = float(n_val_passed / n_val_raw) if n_val_raw > 0 else 0.0
    p_rate_test = float(n_test_passed / n_test_raw) if n_test_raw > 0 else 0.0

    attack_count_train = int(np.sum(y_train_passed == 1))
    benign_count_train = int(np.sum(y_train_passed == 0))
    attack_conc_train = float(attack_count_train / n_train_passed) if n_train_passed > 0 else 0.0

    attack_count_test = int(np.sum(y_test_passed == 1))
    benign_count_test = int(np.sum(y_test_passed == 0))

    # 9. Export to .npz file
    out_file = output_dir / f"{cid_str}_passed.npz"
    np.savez_compressed(
        out_file,
        # Array keys
        x_train_passed=x_train_passed,
        y_train_passed=y_train_passed,
        orig_idx_train_passed=orig_idx_train_passed,
        x_val_passed=x_val_passed,
        y_val_passed=y_val_passed,
        orig_idx_val_passed=orig_idx_val_passed,
        x_test_passed=x_test_passed,
        y_test_passed=y_test_passed,
        orig_idx_test_passed=orig_idx_test_passed,
        tier2_mask_test=tier2_mask_test,
        tier3_mask_test=tier3_mask_test,
        # Metadata scalar keys
        client_id=client_id,
        quantile_name=quantile_name,
        tau_gate=tau_gate,
        n_train_raw=n_train_raw,
        n_train_passed=n_train_passed,
        passage_rate_train=p_rate_train,
        n_val_raw=n_val_raw,
        n_val_passed=n_val_passed,
        passage_rate_val=p_rate_val,
        n_test_raw=n_test_raw,
        n_test_passed=n_test_passed,
        passage_rate_test=p_rate_test,
        attack_count_train_passed=attack_count_train,
        benign_count_train_passed=benign_count_train,
        attack_concentration_train_passed=attack_conc_train,
        tier1_test_count=tier_counts["tier1_test_count"],
        tier2_test_count=tier_counts["tier2_test_count"],
        tier3_test_count=tier_counts["tier3_test_count"],
    )

    return {
        "client_id": client_id,
        "client_str": cid_str,
        "quantile": quantile_name,
        "tau_gate": tau_gate,
        "n_train_raw": n_train_raw,
        "n_train_passed": n_train_passed,
        "passage_rate_train_pct": p_rate_train * 100.0,
        "n_val_raw": n_val_raw,
        "n_val_passed": n_val_passed,
        "passage_rate_val_pct": p_rate_val * 100.0,
        "n_test_raw": n_test_raw,
        "n_test_passed": n_test_passed,
        "passage_rate_test_pct": p_rate_test * 100.0,
        "attack_count_train_passed": attack_count_train,
        "benign_count_train_passed": benign_count_train,
        "attack_concentration_train_passed_pct": attack_conc_train * 100.0,
        "attack_count_test_passed": attack_count_test,
        "benign_count_test_passed": benign_count_test,
        "tier1_test_count": tier_counts["tier1_test_count"],
        "tier2_test_count": tier_counts["tier2_test_count"],
        "tier3_test_count": tier_counts["tier3_test_count"],
        "tier2_pct_of_tier1": (tier_counts["tier2_test_count"] / n_test_passed * 100.0) if n_test_passed > 0 else 0.0,
        "tier3_pct_of_tier1": (tier_counts["tier3_test_count"] / n_test_passed * 100.0) if n_test_passed > 0 else 0.0,
        "exported_npz": str(out_file),
    }


def run_gate_passed_extraction() -> Dict[str, Any]:
    """Execute complete Gate-passed extraction across all 10 clients for Q99 and Q95."""
    print("=" * 70)
    print("Starting Stage 5 Phase 5.1: Gate-Passed Dataset Extraction")
    print("=" * 70)

    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    ckpt_dir = Path("outputs/checkpoints/stage3")
    sens_thresh_path = Path("outputs/experiments/stage3_threshold_sensitivity/sensitivity_client_thresholds.json")

    if not sens_thresh_path.exists():
        raise FileNotFoundError(f"Missing required sensitivity thresholds: {sens_thresh_path}")

    with open(sens_thresh_path, "r", encoding="utf-8") as f:
        thresholds_dict = json.load(f)

    base_out_dir = Path("data/processed/gate_passed")
    q99_out_dir = base_out_dir / "Q99"
    q95_out_dir = base_out_dir / "Q95"
    q99_out_dir.mkdir(parents=True, exist_ok=True)
    q95_out_dir.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, Any]] = []

    for q_name, out_dir in [("Q99", q99_out_dir), ("Q95", q95_out_dir)]:
        print(f"\nProcessing Quantile: {q_name} -> {out_dir}")
        for cid_int in range(10):
            res = extract_gate_passed_for_client(
                client_id=cid_int,
                quantile_name=q_name,
                stage2_dir=stage2_dir,
                ckpt_dir=ckpt_dir,
                thresholds_dict=thresholds_dict,
                output_dir=out_dir,
            )
            summary_records.append(res)
            print(
                f"  [{res['client_str']}] {q_name}: "
                f"Train Passed: {res['n_train_passed']:,}/{res['n_train_raw']:,} ({res['passage_rate_train_pct']:.2f}%) | "
                f"Test Passed: {res['n_test_passed']:,}/{res['n_test_raw']:,} ({res['passage_rate_test_pct']:.2f}%) | "
                f"T1: {res['tier1_test_count']} | T2: {res['tier2_test_count']} | T3: {res['tier3_test_count']}"
            )

    # Save summary tables
    df_summary = pd.DataFrame(summary_records)
    csv_path = base_out_dir / "extraction_summary.csv"
    json_path = base_out_dir / "extraction_summary.json"

    df_summary.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[OK] Extraction successfully completed for all 10 clients across Q99 and Q95.")
    print(f"[OK] Summary CSV exported to: {csv_path}")
    print(f"[OK] Summary JSON exported to: {json_path}")
    print("=" * 70)

    return {"summary_records": summary_records, "df": df_summary}


if __name__ == "__main__":
    run_gate_passed_extraction()


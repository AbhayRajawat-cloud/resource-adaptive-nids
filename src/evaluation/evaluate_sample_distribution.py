"""
Stage 5: Split-Wise Per-Client Sample-Size & Class Imbalance Audit.

Extracts complete split-wise sample distributions across Train, Validation, and Test sets
from Stage 2 raw partitions and Phase 5.1 Gate-passed extracted NPZ files.
Extracts exact training positive class weights (w_pos) from Phase 5.2 training metadata.

Output:
- outputs/metrics/stage5/validation_client_sample_distribution.csv
"""

import json
from pathlib import Path
import sys
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd


def extract_sample_distributions() -> pd.DataFrame:
    """Extract sample size metrics across all 10 clients and both quantiles."""
    print("=" * 75)
    print("Extracting Split-Wise Per-Client Sample-Size & Class Imbalance Metrics")
    print("=" * 75)

    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    gate_passed_dir = Path("data/processed/gate_passed")
    meta_path = Path("outputs/metrics/stage5/phase5.2_training_metadata.json")
    output_dir = Path("outputs/metrics/stage5")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load exact w_pos from Phase 5.2 metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    # Build lookup: (cid, quantile) -> (w_pos, fallback)
    w_pos_map: Dict[tuple, tuple] = {}
    for entry in meta_list:
        cid = entry["client_id"]
        q = entry["quantile"]
        ds = entry["dataset_statistics"]
        w_pos_map[(cid, q)] = (ds.get("w_pos"), ds.get("w_pos_fallback_used", False))

    records: List[Dict[str, Any]] = []

    for q in ["Q99", "Q95"]:
        for cid in range(10):
            cid_str = f"client_{cid:02d}"
            raw_path = stage2_dir / cid_str / "data.npz"
            passed_path = gate_passed_dir / q / f"{cid_str}_passed.npz"

            raw = np.load(raw_path, allow_pickle=True)
            passed = np.load(passed_path, allow_pickle=True)

            # Raw counts
            tr_b = int(np.sum(raw["y_train"] == 0))
            tr_a = int(np.sum(raw["y_train"] == 1))
            tr_tot = tr_b + tr_a

            va_b = int(np.sum(raw["y_val"] == 0))
            va_a = int(np.sum(raw["y_val"] == 1))
            va_tot = va_b + va_a

            te_b = int(np.sum(raw["y_test"] == 0))
            te_a = int(np.sum(raw["y_test"] == 1))
            te_tot = te_b + te_a

            # Passed counts
            ptr_b = int(np.sum(passed["y_train_passed"] == 0))
            ptr_a = int(np.sum(passed["y_train_passed"] == 1))
            ptr_tot = ptr_b + ptr_a

            pva_b = int(np.sum(passed["y_val_passed"] == 0))
            pva_a = int(np.sum(passed["y_val_passed"] == 1))
            pva_tot = pva_b + pva_a

            pte_b = int(np.sum(passed["y_test_passed"] == 0))
            pte_a = int(np.sum(passed["y_test_passed"] == 1))
            pte_tot = pte_b + pte_a

            # Passage rates
            tr_rate = (ptr_tot / tr_tot * 100.0) if tr_tot > 0 else 0.0
            va_rate = (pva_tot / va_tot * 100.0) if va_tot > 0 else 0.0
            te_rate = (pte_tot / te_tot * 100.0) if te_tot > 0 else 0.0

            # Gate-passed attack concentrations
            tr_conc = (ptr_a / ptr_tot * 100.0) if ptr_tot > 0 else 0.0
            va_conc = (pva_a / pva_tot * 100.0) if pva_tot > 0 else 0.0
            te_conc = (pte_a / pte_tot * 100.0) if pte_tot > 0 else 0.0

            # Training weights
            w_pos, w_fallback = w_pos_map.get((cid, q), (None, False))

            records.append({
                "client_id": cid,
                "client_str": cid_str,
                "quantile": q,
                # Train split
                "train_raw_benign": tr_b,
                "train_raw_attack": tr_a,
                "train_raw_total": tr_tot,
                "train_pass_benign": ptr_b,
                "train_pass_attack": ptr_a,
                "train_pass_total": ptr_tot,
                "train_passage_rate_pct": round(tr_rate, 4),
                "train_attack_concentration_pct": round(tr_conc, 4),
                # Val split
                "val_raw_benign": va_b,
                "val_raw_attack": va_a,
                "val_raw_total": va_tot,
                "val_pass_benign": pva_b,
                "val_pass_attack": pva_a,
                "val_pass_total": pva_tot,
                "val_passage_rate_pct": round(va_rate, 4),
                "val_attack_concentration_pct": round(va_conc, 4),
                # Test split
                "test_raw_benign": te_b,
                "test_raw_attack": te_a,
                "test_raw_total": te_tot,
                "test_pass_benign": pte_b,
                "test_pass_attack": pte_a,
                "test_pass_total": pte_tot,
                "test_passage_rate_pct": round(te_rate, 4),
                "test_attack_concentration_pct": round(te_conc, 4),
                # Training weights
                "w_pos_training": round(w_pos, 6) if w_pos is not None else None,
                "w_pos_fallback_used": w_fallback,
            })

    df = pd.DataFrame(records)
    out_csv = output_dir / "validation_client_sample_distribution.csv"
    df.to_csv(out_csv, index=False)
    print(f"Exported {len(df)} client distribution rows to {out_csv}")
    return df


if __name__ == "__main__":
    extract_sample_distributions()


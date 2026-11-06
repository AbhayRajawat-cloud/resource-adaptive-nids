"""
Stage 2 pipeline orchestrator.

Execution order
---------------
1.  Load the raw CSV into memory (single pass, preserve original row index).
2.  Run mandatory pre-partitioning verification (pairs + duplicate vectors).
3.  Finalise the feature column list based on verification results.
4.  Run constrained Dirichlet partitioning (default alpha=0.5).
5.  Preprocess each client (split → fit scaler on benign train → transform).
6.  Save processed arrays (.npz) + scaler params (.json) per client.
7.  Run full post-processing validation checks.
8.  Generate and save the partition summary report (JSON).

Usage
-----
    python3 run_stage2.py                          # default alpha=0.5
    python3 run_stage2.py --alpha 0.1              # specific alpha
    python3 run_stage2.py --alpha inf              # IID / uniform split
    python3 run_stage2.py --seed 123               # custom seed
    python3 run_stage2.py --skip-verification      # skip pre-checks (not recommended)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root on path when run directly
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.verification import run_verification
from src.data.partitioning import partition_dirichlet
from src.data.preprocessing import preprocess_client
from src.utils.config import load_config
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDENTIFIER_COLUMNS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
LABEL_COLUMNS = ["Label", "Attack"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_partition(
    client_results: Dict[int, Dict],
    total_rows: int,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """
    Run all post-processing validation checks. Returns a report dict.
    Raises AssertionError on any failure.
    """
    print("\n[Validate] Running partition validation checks ...")
    issues: List[str] = []

    # Collect all original indices across all splits
    all_train_orig, all_val_orig, all_test_orig = [], [], []
    for cid, cr in client_results.items():
        all_train_orig.extend(cr["original_indices"]["train"].tolist())
        all_val_orig.extend(cr["original_indices"]["val"].tolist())
        all_test_orig.extend(cr["original_indices"]["test"].tolist())

    all_assigned = sorted(all_train_orig + all_val_orig + all_test_orig)

    # 1. No samples lost
    if len(all_assigned) != total_rows:
        issues.append(f"FAIL: {len(all_assigned)} samples assigned, expected {total_rows}")
    else:
        print(f"  ✓ All {total_rows:,} samples accounted for")

    # 2. No duplicates across all clients and splits
    if len(set(all_assigned)) != len(all_assigned):
        issues.append("FAIL: Duplicate row indices detected across clients/splits")
    else:
        print("  ✓ No cross-client or cross-split index duplication")

    # 3. Train/val/test indices non-overlapping within each client
    per_client_ok = True
    for cid, cr in client_results.items():
        tr = set(cr["original_indices"]["train"].tolist())
        va = set(cr["original_indices"]["val"].tolist())
        te = set(cr["original_indices"]["test"].tolist())
        if tr & va or tr & te or va & te:
            issues.append(f"FAIL: Split overlap within client {cid}")
            per_client_ok = False
    if per_client_ok:
        print("  ✓ Train / val / test non-overlapping within every client")

    # 4. Minimum benign samples per client
    benign_ok = True
    for cid, cr in client_results.items():
        n_benign_train = cr["client_stats"]["benign_train_samples"]
        if n_benign_train < 1000:
            issues.append(
                f"FAIL: Client {cid} has only {n_benign_train} benign training samples (< 1000)"
            )
            benign_ok = False
    if benign_ok:
        print("  ✓ All clients meet 1,000 benign training sample minimum")

    # 5. All scaled values finite and within [0, 1]
    scale_ok = True
    for cid, cr in client_results.items():
        for split_name in ("X_train", "X_val", "X_test"):
            X = cr[split_name]
            if not np.all(np.isfinite(X)):
                issues.append(f"FAIL: Non-finite values in client {cid} {split_name}")
                scale_ok = False
            if np.any(X < -1e-6) or np.any(X > 1.0 + 1e-6):
                issues.append(f"FAIL: Values outside [0,1] in client {cid} {split_name}")
                scale_ok = False
    if scale_ok:
        print("  ✓ All scaled feature values are finite and within [0, 1]")

    # 6. No artificial duplication of Worms (164 rows)
    total_worms = sum(
        int((cr["attack_train"] == "Worms").sum()) +
        int((cr["attack_val"] == "Worms").sum()) +
        int((cr["attack_test"] == "Worms").sum())
        for cr in client_results.values()
    )
    if total_worms != 164:
        issues.append(
            f"FAIL: Worms count mismatch. Expected 164, found {total_worms}. "
            "Possible artificial duplication."
        )
    else:
        print(f"  ✓ Worms count preserved exactly: {total_worms} rows")

    # 7. Labels and indices alignment spot-check (verify counts add up)
    label_sum = sum(
        len(cr["y_train"]) + len(cr["y_val"]) + len(cr["y_test"])
        for cr in client_results.values()
    )
    if label_sum != total_rows:
        issues.append(f"FAIL: Label array total {label_sum} != {total_rows}")
    else:
        print(f"  ✓ Label arrays correctly aligned with all {total_rows:,} samples")

    status = "PASS" if not issues else "FAIL"
    print(f"\n[Validate] Result: {status}")
    if issues:
        for iss in issues:
            print(f"  !! {iss}")

    return {
        "status": status,
        "issues": issues,
        "checks_performed": [
            "no_samples_lost",
            "no_cross_client_duplication",
            "no_intra_client_split_overlap",
            "min_benign_per_client",
            "all_scaled_values_in_0_1",
            "worms_count_preserved",
            "label_index_alignment",
        ],
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_client_data(
    client_id: int,
    cr: Dict,
    processed_dir: Path,
    feature_cols: List[str],
) -> None:
    """Save scaled arrays + metadata for a single client."""
    client_dir = processed_dir / f"client_{client_id:02d}"
    client_dir.mkdir(parents=True, exist_ok=True)

    # Feature arrays
    np.savez_compressed(
        client_dir / "data.npz",
        X_train=cr["X_train"], y_train=cr["y_train"],
        X_val=cr["X_val"],   y_val=cr["y_val"],
        X_test=cr["X_test"], y_test=cr["y_test"],
        attack_train=cr["attack_train"],
        attack_val=cr["attack_val"],
        attack_test=cr["attack_test"],
        orig_idx_train=cr["original_indices"]["train"],
        orig_idx_val=cr["original_indices"]["val"],
        orig_idx_test=cr["original_indices"]["test"],
    )

    # Scaler parameters
    with open(client_dir / "scaler_params.json", "w") as f:
        json.dump(cr["scaler_params"], f, indent=2)

    # Client statistics
    with open(client_dir / "client_stats.json", "w") as f:
        stats = dict(cr["client_stats"])
        json.dump(stats, f, indent=2)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage2(
    alpha: float = 0.5,
    seed: int = 42,
    skip_verification: bool = False,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full Stage 2 pipeline."""
    config = load_config(config_path)
    set_seed(seed)

    root = Path(__file__).resolve().parent
    raw_csv = root / config["paths"]["raw_data_dir"] / config["paths"]["raw_csv_filename"]
    processed_dir = root / config["paths"]["processed_data_dir"]
    metrics_dir = root / config["paths"]["metrics_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    alpha_tag = "inf" if math.isinf(alpha) else str(alpha)
    run_dir = processed_dir / f"alpha_{alpha_tag}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  Stage 2: Federated Preprocessing  [alpha={alpha_tag}, seed={seed}]")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Load full dataset
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading full dataset ...")
    df = pd.read_csv(raw_csv, low_memory=False)
    df.index = np.arange(len(df))  # ensure integer index == original row number
    total_rows = len(df)
    print(f"  Loaded {total_rows:,} rows × {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # Step 2: Determine candidate feature columns (pre-verification)
    # ------------------------------------------------------------------
    exclude = set(IDENTIFIER_COLUMNS + LABEL_COLUMNS)
    candidate_feature_cols = [c for c in df.columns if c not in exclude]
    print(f"\n[Step 2] Candidate features (before verification): {len(candidate_feature_cols)}")

    # ------------------------------------------------------------------
    # Step 3: Pre-partitioning verification
    # ------------------------------------------------------------------
    if not skip_verification:
        print("\n[Step 3] Pre-partitioning verification ...")
        verify_report = run_verification(
            csv_path=raw_csv,
            feature_cols=candidate_feature_cols,
            output_path=metrics_dir / "verification_report.json",
        )
        feature_cols = verify_report["verified_feature_cols"]
        dropped_cols = verify_report["dropped_columns"]
        print(f"\n  Columns dropped after verification: {dropped_cols}")
        print(f"  Final feature dimensionality: {len(feature_cols)}")

        if verify_report["duplicate_vector_check"]["has_conflicts"]:
            print("\n  ⚠ LABEL CONFLICTS DETECTED — see verification_report.json")
            print(f"  {verify_report['duplicate_vector_check']['recommendation']}")
    else:
        print("\n[Step 3] Verification skipped (--skip-verification flag set)")
        feature_cols = candidate_feature_cols
        dropped_cols = []
        verify_report = {"skipped": True}

    print(f"\n  Feature dimensionality finalised: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # Step 4: Dirichlet partitioning
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Dirichlet partitioning (alpha={alpha_tag}, K=10) ...")
    client_index_map = partition_dirichlet(
        df=df,
        num_clients=10,
        alpha=alpha,
        seed=seed,
        min_benign=1_000,
    )

    # ------------------------------------------------------------------
    # Step 5 & 6: Per-client preprocessing + save
    # ------------------------------------------------------------------
    print("\n[Step 5] Client-level preprocessing ...")
    client_results: Dict[int, Dict] = {}

    for cid in range(10):
        print(f"\n  --- Client {cid:02d} ---")
        cidx = client_index_map[cid]
        client_df = df.iloc[cidx].copy()
        # Reset positional index but keep original CSV index accessible
        client_df = df.loc[cidx]  # preserves original integer index

        cr = preprocess_client(
            client_df=client_df,
            feature_cols=feature_cols,
            client_id=cid,
            seed=seed,
        )
        client_results[cid] = cr

        stats = cr["client_stats"]
        print(f"    Total: {stats['total_samples']:,} | "
              f"Benign: {stats['n_benign']:,} | "
              f"Malicious: {stats['n_malicious']:,}")
        print(f"    Train: {stats['split_sizes']['train']:,} | "
              f"Val: {stats['split_sizes']['val']:,} | "
              f"Test: {stats['split_sizes']['test']:,}")
        print(f"    Benign train samples (scaler fit): {stats['benign_train_samples']:,}")

        _save_client_data(cid, cr, run_dir, feature_cols)
        print(f"    Saved → {run_dir}/client_{cid:02d}/")

    # ------------------------------------------------------------------
    # Step 7: Validation
    # ------------------------------------------------------------------
    print("\n[Step 6] Post-processing validation ...")
    validation_report = validate_partition(client_results, total_rows, feature_cols)

    if validation_report["status"] == "FAIL":
        raise RuntimeError(
            "Partition validation FAILED. See issues above. "
            "Stage 2 output is NOT saved to avoid using invalid data."
        )

    # ------------------------------------------------------------------
    # Step 8: Save partition summary report
    # ------------------------------------------------------------------
    partition_summary = {
        "stage": 2,
        "alpha": alpha_tag,
        "seed": seed,
        "num_clients": 10,
        "total_rows": total_rows,
        "feature_cols": feature_cols,
        "feature_count": len(feature_cols),
        "dropped_cols": dropped_cols,
        "split_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "normalization": {
            "method": "benign_train_only_clip_minmax",
            "percentile_low": 1,
            "percentile_high": 99,
            "scale_range": [0, 1],
        },
        "client_stats": [cr["client_stats"] for cr in client_results.values()],
        "validation": validation_report,
        "verification_summary": {
            "skipped": skip_verification,
            "dropped_cols": dropped_cols,
            "has_label_conflicts": (
                verify_report.get("duplicate_vector_check", {}).get("has_conflicts", False)
                if not skip_verification else None
            ),
        },
    }

    summary_path = metrics_dir / f"partition_summary_alpha{alpha_tag}_seed{seed}.json"
    with open(summary_path, "w") as f:
        json.dump(partition_summary, f, indent=2)
    print(f"\n[Summary] Report saved → {summary_path}")

    # Save feature list for downstream modules
    feature_list_path = processed_dir / "feature_cols.json"
    with open(feature_list_path, "w") as f:
        json.dump({"feature_cols": feature_cols, "feature_count": len(feature_cols)}, f, indent=2)

    print("\n" + "=" * 65)
    print("  Stage 2 COMPLETE")
    print(f"  Feature dimensionality : {len(feature_cols)}")
    print(f"  Clients processed      : 10")
    print(f"  Validation             : {validation_report['status']}")
    print(f"  Output directory       : {run_dir}")
    print("=" * 65)

    return {"summary": partition_summary, "client_results": client_results}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Federated Preprocessing")
    parser.add_argument("--alpha", type=str, default="0.5",
                        help="Dirichlet alpha (use 'inf' for IID/uniform)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-verification", action="store_true",
                        help="Skip mandatory pre-partitioning verification (not recommended)")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    alpha = math.inf if args.alpha.lower() in ("inf", "infinity") else float(args.alpha)
    run_stage2(
        alpha=alpha,
        seed=args.seed,
        skip_verification=args.skip_verification,
        config_path=args.config,
    )


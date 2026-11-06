"""
Stage 2 — Pre-partitioning verification checks.

Performs two mandatory full-dataset scans before the feature dimensionality
is finalised:

  A. Perfect-correlation check for candidate redundant pairs.
  B. Duplicate feature-vector and label-conflict audit.

All checks are read-only: nothing is modified or deleted here.
Results drive decisions in the pipeline orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDENTIFIER_COLUMNS: List[str] = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
]
LABEL_COLUMNS: List[str] = ["Label", "Attack"]

# Candidate pairs to check for perfect (r = 1, bit-exact equality) redundancy.
CANDIDATE_REDUNDANT_PAIRS: List[Tuple[str, str]] = [
    ("LONGEST_FLOW_PKT", "MAX_IP_PKT_LEN"),
    ("ICMP_TYPE", "ICMP_IPV4_TYPE"),
]

CHUNK_SIZE = 200_000


# ---------------------------------------------------------------------------
# A. Perfect-correlation check
# ---------------------------------------------------------------------------

def verify_redundant_pairs(
    csv_path: Path,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Check whether candidate feature pairs are bit-exact across all rows.

    Strategy
    --------
    For each pair (A, B) accumulate the count of rows where A != B.
    If that count == 0 the columns are perfectly identical and one
    can be safely removed.

    Returns
    -------
    dict with keys:
      - pair_results : list of per-pair dicts
      - columns_to_drop : list of column names recommended for removal
      - columns_to_keep_from_pairs : list of retained column per pair
    """
    if pairs is None:
        pairs = CANDIDATE_REDUNDANT_PAIRS

    # Read all relevant columns in one pass
    cols_needed = list({c for pair in pairs for c in pair})
    mismatch_counts: Dict[Tuple[str, str], int] = {p: 0 for p in pairs}
    total_rows = 0

    for chunk in pd.read_csv(
        csv_path, chunksize=CHUNK_SIZE, usecols=cols_needed, low_memory=False
    ):
        total_rows += len(chunk)
        for pair in pairs:
            a, b = pair
            if a in chunk.columns and b in chunk.columns:
                # Cast both to float64 to handle mixed int/float edge cases
                col_a = chunk[a].astype("float64")
                col_b = chunk[b].astype("float64")
                mismatch_counts[pair] += int((col_a != col_b).sum())

    pair_results: List[Dict] = []
    columns_to_drop: List[str] = []

    for pair, mismatch in mismatch_counts.items():
        a, b = pair
        is_identical = mismatch == 0
        retained = a        # keep the first; drop the second
        dropped = b if is_identical else None

        if is_identical:
            columns_to_drop.append(b)

        pair_results.append({
            "feature_a": a,
            "feature_b": b,
            "total_rows_checked": total_rows,
            "mismatch_rows": mismatch,
            "are_bit_exact_identical": is_identical,
            "decision": f"DROP '{dropped}', KEEP '{retained}'" if is_identical
                        else f"KEEP BOTH (differ in {mismatch} rows)",
        })

    return {
        "pair_results": pair_results,
        "columns_to_drop": columns_to_drop,
        "total_rows_verified": total_rows,
    }


# ---------------------------------------------------------------------------
# B. Duplicate feature-vector and label-conflict audit
# ---------------------------------------------------------------------------

def verify_duplicate_vectors(
    csv_path: Path,
    feature_cols: List[str],
    label_col: str = "Label",
) -> Dict[str, Any]:
    """
    Identify duplicate feature vectors (excluding identifiers) and detect
    whether any identical vectors carry conflicting labels.

    Strategy
    --------
    - Hash each feature vector row using pandas hash_pandas_object.
    - Track (hash -> set of unique label values).
    - Conflicts exist where len(label_set) > 1.

    Returns
    -------
    dict describing duplicate counts and conflict counts.
    NOTE: No rows are deleted here. Recommendations are returned only.
    """
    hash_to_labels: Dict[int, set] = {}
    total_rows = 0
    duplicate_rows = 0

    usecols = feature_cols + [label_col]

    for chunk in pd.read_csv(
        csv_path, chunksize=CHUNK_SIZE, usecols=usecols, low_memory=False
    ):
        total_rows += len(chunk)
        feature_chunk = chunk[feature_cols]
        hashes = pd.util.hash_pandas_object(feature_chunk, index=False)

        for h, lbl in zip(hashes, chunk[label_col]):
            h = int(h)
            if h in hash_to_labels:
                duplicate_rows += 1
                hash_to_labels[h].add(int(lbl))
            else:
                hash_to_labels[h] = {int(lbl)}

    # Classify hashes
    unique_vectors = len(hash_to_labels)
    conflict_hashes = {h: list(lbls) for h, lbls in hash_to_labels.items()
                       if len(lbls) > 1}

    # Count rows involved in conflicts
    conflict_row_count = 0
    if conflict_hashes:
        conflict_hash_set = set(conflict_hashes.keys())
        for chunk in pd.read_csv(
            csv_path, chunksize=CHUNK_SIZE, usecols=feature_cols, low_memory=False
        ):
            hashes = pd.util.hash_pandas_object(chunk, index=False)
            conflict_row_count += int(
                sum(1 for h in hashes if int(h) in conflict_hash_set)
            )

    recommendation = (
        "No label conflicts detected. Duplicate vectors with identical labels "
        "may be retained safely; they represent repeated identical network flows."
        if not conflict_hashes
        else (
            f"{len(conflict_hashes)} unique feature vectors map to BOTH Label=0 and Label=1. "
            f"These {conflict_row_count} rows introduce irreducible label noise. "
            "Recommended handling: retain all rows (removing them would bias class balance); "
            "document the noise rate in the paper."
        )
    )

    return {
        "total_rows": total_rows,
        "duplicate_feature_vector_rows": duplicate_rows,
        "unique_feature_vectors": unique_vectors,
        "label_conflict_vectors": len(conflict_hashes),
        "label_conflict_rows": conflict_row_count,
        "conflict_details_sample": dict(list(conflict_hashes.items())[:10]),
        "recommendation": recommendation,
        "has_conflicts": len(conflict_hashes) > 0,
    }


# ---------------------------------------------------------------------------
# Orchestrator: run both checks and save report
# ---------------------------------------------------------------------------

def run_verification(
    csv_path: Path,
    feature_cols: List[str],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run both pre-partitioning verification checks and save a JSON report.

    Parameters
    ----------
    csv_path    : path to raw CSV (never modified).
    feature_cols: final candidate feature columns (after identifier removal).
    output_path : where to write the JSON report.

    Returns
    -------
    Full verification report dict. Also saves JSON to output_path.
    """
    print("[Verify] A. Checking perfectly correlated / bit-exact redundant pairs ...")
    pair_report = verify_redundant_pairs(csv_path)

    for r in pair_report["pair_results"]:
        print(f"    {r['feature_a']} vs {r['feature_b']}: {r['decision']}")

    print("[Verify] B. Duplicate feature-vector and label-conflict audit ...")
    # Use the feature cols AFTER applying the drop recommendation
    cols_to_drop = pair_report["columns_to_drop"]
    verified_feature_cols = [c for c in feature_cols if c not in cols_to_drop]
    dup_report = verify_duplicate_vectors(csv_path, verified_feature_cols)

    print(f"    Duplicate vectors : {dup_report['duplicate_feature_vector_rows']:,} rows")
    print(f"    Label conflicts   : {dup_report['label_conflict_vectors']} unique vectors")
    print(f"    Conflicts row count: {dup_report['label_conflict_rows']}")

    report = {
        "redundant_pair_check": pair_report,
        "duplicate_vector_check": dup_report,
        "verified_feature_cols": verified_feature_cols,
        "final_feature_count": len(verified_feature_cols),
        "dropped_columns": cols_to_drop,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[Verify] Report saved → {output_path}")

    return report


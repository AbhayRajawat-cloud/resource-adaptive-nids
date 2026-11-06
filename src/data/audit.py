"""
Dataset audit and profiling routines for NF-UNSW-NB15-v2.

This module performs a comprehensive, read-only audit of the raw dataset.
It never modifies the source CSV. All analyses are done in chunks to
stay within reasonable RAM constraints on both macOS and Kaggle kernels.

Sections
--------
1. File-level metadata (size, row count, column schema)
2. Missing-value and infinite-value survey
3. Duplicate record detection
4. Constant-feature (zero-variance) detection
5. Numerical distribution summary (per-feature statistics)
6. Extreme-value analysis
7. High-correlation feature pairs
8. Label semantics (Label vs Attack columns)
9. Temporal / chronological ordering investigation
10. Expected-feature completeness check
"""

from __future__ import annotations

import json
import math
import os
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

# Known temporal candidate column names to probe (case-insensitive).
TEMPORAL_CANDIDATES: List[str] = [
    "FIRST_SWITCHED",
    "LAST_SWITCHED",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "FLOW_START_SECONDS",
    "FLOW_END_SECONDS",
    "TIMESTAMP",
    "DATE_FIRST_SEEN",
    "DATE_LAST_SEEN",
    "START_TIME",
    "STOP_TIME",
]

CHUNK_SIZE: int = 200_000          # rows per chunk
CORRELATION_THRESHOLD: float = 0.95


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_serialisable(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return _to_serialisable(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    return obj


def _resolve_csv_path(csv_path: Optional[str], config: Optional[Dict] = None) -> Path:
    """Resolve the absolute path to the raw CSV file."""
    if csv_path is not None:
        return Path(csv_path).resolve()

    if config is not None:
        raw_dir = config.get("paths", {}).get("raw_data_dir", "data/raw")
        filename = config.get("paths", {}).get("raw_csv_filename", "NF-UNSW-NB15-v2.csv")
        return Path(raw_dir, filename).resolve()

    # Fallback: walk up from this file to project root
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "raw" / "NF-UNSW-NB15-v2.csv"


# ---------------------------------------------------------------------------
# Section 1 – File-level metadata
# ---------------------------------------------------------------------------

def audit_file_metadata(csv_path: Path) -> Dict[str, Any]:
    """Return file size, row count (exact), and column schema via chunked scan."""
    file_size_bytes = csv_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 ** 2), 2)

    # Read header only for column names and dtypes
    header_df = pd.read_csv(csv_path, nrows=0)
    columns = list(header_df.columns)
    num_columns = len(columns)

    # Count rows in chunks (avoid loading entire file)
    row_count = 0
    dtype_sample: Dict[str, str] = {}
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
        if not dtype_sample:
            dtype_sample = {col: str(dtype) for col, dtype in chunk.dtypes.items()}
        row_count += len(chunk)

    return {
        "file_path": str(csv_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "total_rows": row_count,
        "total_columns": num_columns,
        "column_names": columns,
        "column_dtypes": dtype_sample,
    }


# ---------------------------------------------------------------------------
# Section 2 – Missing values and infinite values
# ---------------------------------------------------------------------------

def audit_missing_and_infinite(csv_path: Path) -> Dict[str, Any]:
    """Count NaN/None and ±Inf values per column via chunked scan."""
    missing_counts: Dict[str, int] = {}
    inf_counts: Dict[str, int] = {}
    total_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
        total_rows += len(chunk)

        # Missing values (NaN/None)
        for col, cnt in chunk.isnull().sum().items():
            missing_counts[col] = missing_counts.get(col, 0) + int(cnt)

        # Infinite values (numeric columns only)
        num_cols = chunk.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            inf_cnt = int(np.isinf(chunk[col]).sum())
            inf_counts[col] = inf_counts.get(col, 0) + inf_cnt

    missing_features = {k: v for k, v in missing_counts.items() if v > 0}
    inf_features = {k: v for k, v in inf_counts.items() if v > 0}

    return {
        "total_rows_scanned": total_rows,
        "columns_with_missing_values": missing_features,
        "total_missing_cells": sum(missing_counts.values()),
        "columns_with_infinite_values": inf_features,
        "total_infinite_cells": sum(inf_counts.values()),
        "missing_value_free": len(missing_features) == 0,
        "infinite_value_free": len(inf_features) == 0,
    }


# ---------------------------------------------------------------------------
# Section 3 – Duplicate record detection
# ---------------------------------------------------------------------------

def audit_duplicates(csv_path: Path) -> Dict[str, Any]:
    """Detect exact duplicate rows. Uses a hash-based chunked approach."""
    seen_hashes: set = set()
    duplicate_count = 0
    total_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
        total_rows += len(chunk)
        # Hash each row tuple for memory efficiency
        row_hashes = pd.util.hash_pandas_object(chunk, index=False)
        for h in row_hashes:
            if h in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(h)

    return {
        "total_rows": total_rows,
        "duplicate_rows": duplicate_count,
        "unique_rows": total_rows - duplicate_count,
        "duplicate_percentage": round(duplicate_count / total_rows * 100, 4) if total_rows else 0.0,
    }


# ---------------------------------------------------------------------------
# Section 4 – Constant-value (zero-variance) feature detection
# ---------------------------------------------------------------------------

def audit_constant_features(csv_path: Path) -> Dict[str, Any]:
    """Identify features where all observed values are identical."""
    # Track min and max per column across chunks to detect constants efficiently
    col_min: Dict[str, Any] = {}
    col_max: Dict[str, Any] = {}

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
        num_cols = chunk.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            cmin = chunk[col].min()
            cmax = chunk[col].max()
            col_min[col] = cmin if col not in col_min else min(col_min[col], cmin)
            col_max[col] = cmax if col not in col_max else max(col_max[col], cmax)

    constant_features = [
        col for col in col_min if col_min[col] == col_max[col]
    ]
    near_constant = [
        col for col in col_min
        if col not in constant_features
        and col_max[col] != 0
        and abs(col_max[col] - col_min[col]) / (abs(col_max[col]) + 1e-12) < 1e-6
    ]

    return {
        "constant_features": constant_features,
        "near_constant_features": near_constant,
        "num_constant_features": len(constant_features),
    }


# ---------------------------------------------------------------------------
# Section 5 – Numerical distribution summary
# ---------------------------------------------------------------------------

def audit_numerical_distributions(csv_path: Path) -> Dict[str, Any]:
    """Compute per-feature descriptive statistics using chunked Welford-style accumulators."""
    # Accumulate: count, sum, sum_of_squares, min, max per numeric col
    count: Dict[str, int] = {}
    total: Dict[str, float] = {}
    sum_sq: Dict[str, float] = {}
    col_min: Dict[str, float] = {}
    col_max: Dict[str, float] = {}
    zero_counts: Dict[str, int] = {}

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
        num_cols = chunk.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            s = chunk[col].dropna()
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                continue
            n = len(s)
            count[col] = count.get(col, 0) + n
            total[col] = total.get(col, 0.0) + float(s.sum())
            sum_sq[col] = sum_sq.get(col, 0.0) + float((s ** 2).sum())
            col_min[col] = min(col_min.get(col, float("inf")), float(s.min()))
            col_max[col] = max(col_max.get(col, float("-inf")), float(s.max()))
            zero_counts[col] = zero_counts.get(col, 0) + int((s == 0).sum())

    stats: Dict[str, Dict] = {}
    for col in count:
        n = count[col]
        mean = total[col] / n if n > 0 else 0.0
        variance = (sum_sq[col] / n) - (mean ** 2) if n > 0 else 0.0
        std = math.sqrt(max(variance, 0.0))
        stats[col] = {
            "count": n,
            "mean": round(mean, 6),
            "std": round(std, 6),
            "min": round(col_min[col], 6),
            "max": round(col_max[col], 6),
            "zero_count": zero_counts[col],
            "zero_percentage": round(zero_counts[col] / n * 100, 4) if n > 0 else 0.0,
        }

    return {"feature_statistics": stats}


# ---------------------------------------------------------------------------
# Section 6 – Extreme-value analysis
# ---------------------------------------------------------------------------

def audit_extreme_values(
    csv_path: Path, feature_stats: Dict[str, Dict]
) -> Dict[str, Any]:
    """Flag features where max > mean + 5*std (potential outlier magnitude)."""
    extreme_features: Dict[str, Dict] = {}
    for col, s in feature_stats.items():
        if s["std"] == 0:
            continue
        upper_fence = s["mean"] + 5 * s["std"]
        if s["max"] > upper_fence:
            ratio = s["max"] / (s["mean"] + 1e-12) if s["mean"] != 0 else float("inf")
            extreme_features[col] = {
                "max": s["max"],
                "mean": s["mean"],
                "std": s["std"],
                "upper_fence_5sigma": round(upper_fence, 6),
                "max_to_mean_ratio": round(ratio, 4),
            }
    return {
        "extreme_value_features": extreme_features,
        "num_extreme_features": len(extreme_features),
    }


# ---------------------------------------------------------------------------
# Section 7 – High-correlation feature pairs (sample-based)
# ---------------------------------------------------------------------------

def audit_high_correlations(
    csv_path: Path, sample_rows: int = 100_000, threshold: float = CORRELATION_THRESHOLD
) -> Dict[str, Any]:
    """Identify highly correlated numeric feature pairs on a stratified sample."""
    sample_chunks = []
    rows_collected = 0

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
        num_cols = chunk.select_dtypes(include=[np.number]).columns
        # Exclude label columns from correlation matrix
        feature_cols = [c for c in num_cols if c not in LABEL_COLUMNS]
        sample_chunks.append(chunk[feature_cols].head(max(1, sample_rows // 10)))
        rows_collected += len(sample_chunks[-1])
        if rows_collected >= sample_rows:
            break

    if not sample_chunks:
        return {"high_correlation_pairs": [], "sample_size_used": 0}

    sample_df = pd.concat(sample_chunks, ignore_index=True)
    # Replace infs before correlation
    sample_df = sample_df.replace([np.inf, -np.inf], np.nan)
    corr_matrix = sample_df.corr(method="pearson", numeric_only=True)

    high_pairs: List[Dict] = []
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if not math.isnan(val) and abs(val) >= threshold:
                high_pairs.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "pearson_r": round(float(val), 6),
                })

    high_pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    return {
        "correlation_threshold": threshold,
        "sample_size_used": len(sample_df),
        "high_correlation_pairs": high_pairs,
        "num_high_correlation_pairs": len(high_pairs),
    }


# ---------------------------------------------------------------------------
# Section 8 – Label semantics
# ---------------------------------------------------------------------------

def audit_label_semantics(csv_path: Path) -> Dict[str, Any]:
    """
    Inspect Label and Attack column distributions and their mutual relationship.

    Observations are derived strictly from data; no prior assumptions are made
    about which column is binary vs. multiclass.
    """
    label_counts: Dict[str, Dict[Any, int]] = {"Label": {}, "Attack": {}}
    cross_tab_accum: Dict[Tuple, int] = {}   # (label_val, attack_val) -> count
    total_rows = 0

    for chunk in pd.read_csv(
        csv_path, chunksize=CHUNK_SIZE, usecols=["Label", "Attack"], low_memory=False
    ):
        total_rows += len(chunk)

        for col in ["Label", "Attack"]:
            if col in chunk.columns:
                for val, cnt in chunk[col].value_counts(dropna=False).items():
                    label_counts[col][str(val)] = (
                        label_counts[col].get(str(val), 0) + int(cnt)
                    )

        # Cross-tabulation: vectorized groupby (no row iteration)
        chunk["_lv"] = chunk["Label"].astype(str)
        chunk["_av"] = chunk["Attack"].astype(str)
        grp = chunk.groupby(["_lv", "_av"], observed=True).size()
        for (lv, av), cnt in grp.items():
            cross_tab_accum[(lv, av)] = cross_tab_accum.get((lv, av), 0) + int(cnt)

    # Convert flat cross_tab_accum to nested dict
    cross_tab: Dict[Any, Dict[Any, int]] = {}
    for (lv, av), cnt in cross_tab_accum.items():
        if lv not in cross_tab:
            cross_tab[lv] = {}
        cross_tab[lv][av] = cnt

    # Detect inconsistencies: rows where Label=0 but Attack != 'Benign' (or vice versa)
    # We observe without assuming semantic meaning
    label_unique_count = len(label_counts["Label"])
    attack_unique_count = len(label_counts["Attack"])

    # Determine which column looks binary vs multiclass
    binary_col = None
    multiclass_col = None
    if label_unique_count <= 2 < attack_unique_count:
        binary_col = "Label"
        multiclass_col = "Attack"
    elif attack_unique_count <= 2 < label_unique_count:
        binary_col = "Attack"
        multiclass_col = "Label"

    # Inconsistency check: for each (Label,Attack) pair where Label appears binary,
    # flag cross-tab cells that seem inconsistent (e.g. label=0 paired with a non-benign class)
    inconsistency_notes: List[str] = []
    if binary_col == "Label":
        # Find which Label value corresponds to the more frequent class
        sorted_label = sorted(label_counts["Label"].items(), key=lambda x: x[1], reverse=True)
        majority_label_val = sorted_label[0][0]  # likely benign
        # Find which Attack values appear under both label values
        for lv, av_map in cross_tab.items():
            if lv != majority_label_val:
                for av, cnt in av_map.items():
                    if av in cross_tab.get(majority_label_val, {}):
                        inconsistency_notes.append(
                            f"Attack category '{av}' appears under both Label='{majority_label_val}' "
                            f"and Label='{lv}' ({cnt} rows in Label='{lv}')"
                        )

    return {
        "total_rows": total_rows,
        "label_column": {
            "unique_values": label_unique_count,
            "distribution": label_counts["Label"],
        },
        "attack_column": {
            "unique_values": attack_unique_count,
            "distribution": label_counts["Attack"],
        },
        "inferred_binary_column": binary_col,
        "inferred_multiclass_column": multiclass_col,
        "label_attack_cross_tab": cross_tab,
        "inconsistency_notes": inconsistency_notes[:20],  # cap for JSON readability
    }


# ---------------------------------------------------------------------------
# Section 9 – Temporal provenance investigation
# ---------------------------------------------------------------------------

def audit_temporal_provenance(csv_path: Path) -> Dict[str, Any]:
    """
    Investigate whether chronological ordering can be established.

    Strategy:
      1. Check column headers for any temporal candidate names.
      2. If found, inspect value ranges, monotonicity, and sample values.
      3. If not found, explicitly report that temporal ordering is unverified.
    """
    header_df = pd.read_csv(csv_path, nrows=0)
    actual_cols = [c.upper() for c in header_df.columns]
    col_map = {c.upper(): c for c in header_df.columns}

    found_temporal_cols: List[str] = []
    for candidate in TEMPORAL_CANDIDATES:
        if candidate.upper() in actual_cols:
            found_temporal_cols.append(col_map[candidate.upper()])

    if not found_temporal_cols:
        # Check for partial matches (e.g. 'SWITCH', 'TIME', 'DATE', 'STAMP')
        partial_hits = [
            orig for up, orig in col_map.items()
            if any(kw in up for kw in ["TIME", "DATE", "STAMP", "SWITCH", "FLOW_START", "FLOW_END"])
        ]
        return {
            "explicit_temporal_columns_found": [],
            "partial_name_matches": partial_hits,
            "temporal_ordering_established": False,
            "notes": (
                "No explicit timestamp or flow-time column was identified in the dataset header. "
                "Temporal ordering of rows cannot be verified from available data alone. "
                "Row index MUST NOT be treated as a chronological proxy without external evidence. "
                "This has direct implications for the GRU sequential model: "
                "sliding windows formed from raw row order may not reflect true temporal sequences."
            ),
        }

    # If temporal columns exist, inspect them
    temporal_stats: Dict[str, Any] = {}
    sample_vals: Dict[str, List] = {}
    monotonic_info: Dict[str, bool] = {}

    first_chunk = True
    prev_vals: Dict[str, Any] = {}
    is_monotonic: Dict[str, bool] = {c: True for c in found_temporal_cols}

    for chunk in pd.read_csv(
        csv_path, chunksize=CHUNK_SIZE, usecols=found_temporal_cols, low_memory=False
    ):
        for col in found_temporal_cols:
            if col not in chunk.columns:
                continue
            col_data = chunk[col].dropna()
            if col_data.empty:
                continue

            if col not in temporal_stats:
                temporal_stats[col] = {"min": None, "max": None, "null_count": 0}
                sample_vals[col] = []

            temporal_stats[col]["null_count"] += int(chunk[col].isnull().sum())

            try:
                numeric = pd.to_numeric(col_data, errors="coerce").dropna()
                if not numeric.empty:
                    cmin = float(numeric.min())
                    cmax = float(numeric.max())
                    if temporal_stats[col]["min"] is None:
                        temporal_stats[col]["min"] = cmin
                        temporal_stats[col]["max"] = cmax
                    else:
                        temporal_stats[col]["min"] = min(temporal_stats[col]["min"], cmin)
                        temporal_stats[col]["max"] = max(temporal_stats[col]["max"], cmax)

                    # Check monotonicity across chunks
                    if col in prev_vals:
                        if float(numeric.iloc[0]) < prev_vals[col]:
                            is_monotonic[col] = False
                    prev_vals[col] = float(numeric.iloc[-1])

            except Exception:
                pass

            if len(sample_vals[col]) < 10:
                sample_vals[col].extend(col_data.head(10).tolist())

        first_chunk = False

    ordering_established = any(
        temporal_stats.get(c, {}).get("min") is not None for c in found_temporal_cols
    )

    return {
        "explicit_temporal_columns_found": found_temporal_cols,
        "temporal_column_stats": temporal_stats,
        "sample_values": {k: [str(v) for v in vals] for k, vals in sample_vals.items()},
        "monotonic_within_scan": is_monotonic,
        "temporal_ordering_established": ordering_established,
        "notes": (
            "Temporal columns were found. Monotonicity is checked sequentially across chunks. "
            "If is_monotonic=True, row order may align with time, but this is not guaranteed "
            "without external dataset documentation. Verify before using raw row order for GRU windows."
            if ordering_established
            else "Temporal columns present but could not be parsed as numeric timestamps."
        ),
    }


# ---------------------------------------------------------------------------
# Section 10 – Expected-feature completeness check
# ---------------------------------------------------------------------------

def audit_expected_features(csv_path: Path) -> Dict[str, Any]:
    """Verify that exactly 39 model-input features are present (non-identifier, non-label)."""
    header_df = pd.read_csv(csv_path, nrows=0)
    all_cols = list(header_df.columns)

    exclude = set(IDENTIFIER_COLUMNS) | set(LABEL_COLUMNS)
    feature_cols = [c for c in all_cols if c not in exclude]
    num_features = len(feature_cols)

    # Check for identifier columns
    found_identifiers = [c for c in IDENTIFIER_COLUMNS if c in all_cols]
    missing_identifiers = [c for c in IDENTIFIER_COLUMNS if c not in all_cols]
    found_labels = [c for c in LABEL_COLUMNS if c in all_cols]
    missing_labels = [c for c in LABEL_COLUMNS if c not in all_cols]

    return {
        "expected_num_features": 39,
        "observed_num_features": num_features,
        "feature_count_matches": num_features == 39,
        "feature_columns": feature_cols,
        "found_identifier_columns": found_identifiers,
        "missing_identifier_columns": missing_identifiers,
        "found_label_columns": found_labels,
        "missing_label_columns": missing_labels,
        "extra_feature_count": max(0, num_features - 39),
        "deficit_feature_count": max(0, 39 - num_features),
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_full_audit(
    csv_path: Optional[str] = None,
    config: Optional[Dict] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run all audit sections in sequence.

    Parameters
    ----------
    csv_path : str, optional
        Absolute or relative path to the raw CSV. Resolved via config or defaults if None.
    config : dict, optional
        Loaded experiment config (from load_config()).
    output_path : str, optional
        Path to save the JSON report. Resolved from config if None.

    Returns
    -------
    dict
        Full audit report as a nested dictionary.
    """
    resolved_path = _resolve_csv_path(csv_path, config)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {resolved_path}")

    print(f"[Audit] Starting full dataset audit.")
    print(f"[Audit] File: {resolved_path}")

    print("[Audit] Section 1/10: File metadata ...")
    meta = audit_file_metadata(resolved_path)
    print(f"        → {meta['total_rows']:,} rows | {meta['total_columns']} columns | {meta['file_size_mb']} MB")

    print("[Audit] Section 2/10: Missing & infinite values ...")
    missing = audit_missing_and_infinite(resolved_path)
    print(f"        → Missing cells: {missing['total_missing_cells']} | Infinite cells: {missing['total_infinite_cells']}")

    print("[Audit] Section 3/10: Duplicate records ...")
    dupes = audit_duplicates(resolved_path)
    print(f"        → Duplicates: {dupes['duplicate_rows']:,} ({dupes['duplicate_percentage']}%)")

    print("[Audit] Section 4/10: Constant-value features ...")
    constants = audit_constant_features(resolved_path)
    print(f"        → Constant features: {constants['num_constant_features']}")

    print("[Audit] Section 5/10: Numerical distributions ...")
    distributions = audit_numerical_distributions(resolved_path)
    print(f"        → Profiled {len(distributions['feature_statistics'])} numeric features.")

    print("[Audit] Section 6/10: Extreme values ...")
    extremes = audit_extreme_values(resolved_path, distributions["feature_statistics"])
    print(f"        → Features with extreme values (>5σ): {extremes['num_extreme_features']}")

    print("[Audit] Section 7/10: High-correlation pairs ...")
    correlations = audit_high_correlations(resolved_path)
    print(f"        → High-correlation pairs (|r|≥{CORRELATION_THRESHOLD}): {correlations['num_high_correlation_pairs']}")

    print("[Audit] Section 8/10: Label semantics ...")
    labels = audit_label_semantics(resolved_path)
    print(f"        → Label unique values: {labels['label_column']['unique_values']} | Attack unique values: {labels['attack_column']['unique_values']}")

    print("[Audit] Section 9/10: Temporal provenance ...")
    temporal = audit_temporal_provenance(resolved_path)
    print(f"        → Temporal ordering established: {temporal['temporal_ordering_established']}")

    print("[Audit] Section 10/10: Expected feature completeness ...")
    features = audit_expected_features(resolved_path)
    print(f"        → Observed features: {features['observed_num_features']} | Expected: 39 | Match: {features['feature_count_matches']}")

    # Compose full report
    report = {
        "audit_metadata": {
            "dataset_file": str(resolved_path),
            "packages_used": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            "chunk_size_rows": CHUNK_SIZE,
            "correlation_sample_rows": 100_000,
        },
        "file_metadata": meta,
        "missing_and_infinite": missing,
        "duplicates": dupes,
        "constant_features": constants,
        "numerical_distributions": distributions,
        "extreme_values": extremes,
        "correlations": correlations,
        "label_semantics": labels,
        "temporal_provenance": temporal,
        "expected_features": features,
    }

    # Resolve output path
    if output_path is None and config is not None:
        metrics_dir = config.get("paths", {}).get("metrics_dir", "outputs/metrics")
        root = Path(__file__).resolve().parents[2]
        output_path = str(root / metrics_dir / "dataset_audit_report.json")
    elif output_path is None:
        root = Path(__file__).resolve().parents[2]
        output_path = str(root / "outputs" / "metrics" / "dataset_audit_report.json")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_serialisable(report), f, indent=2)

    print(f"\n[Audit] ✓ Report saved → {out_path}")
    return report


# ---------------------------------------------------------------------------
# Convenience summary printer
# ---------------------------------------------------------------------------

def print_audit_summary(report: Dict[str, Any]) -> None:
    """Print a human-readable summary of the audit report to stdout."""
    sep = "=" * 65

    print(f"\n{sep}")
    print("  DATASET AUDIT SUMMARY — NF-UNSW-NB15-v2")
    print(sep)

    fm = report["file_metadata"]
    print(f"\n📁 FILE")
    print(f"   Size        : {fm['file_size_mb']} MB")
    print(f"   Rows        : {fm['total_rows']:,}")
    print(f"   Columns     : {fm['total_columns']}")

    mi = report["missing_and_infinite"]
    print(f"\n🔍 DATA QUALITY")
    print(f"   Missing cells  : {mi['total_missing_cells']}")
    print(f"   Infinite cells : {mi['total_infinite_cells']}")
    print(f"   Duplicate rows : {report['duplicates']['duplicate_rows']:,} ({report['duplicates']['duplicate_percentage']}%)")
    print(f"   Constant cols  : {report['constant_features']['num_constant_features']}")

    fe = report["expected_features"]
    print(f"\n📊 FEATURES")
    print(f"   Expected model inputs : 39")
    print(f"   Observed model inputs : {fe['observed_num_features']}")
    print(f"   Schema match          : {fe['feature_count_matches']}")
    print(f"   High-correlation pairs: {report['correlations']['num_high_correlation_pairs']}")
    print(f"   Extreme-value features: {report['extreme_values']['num_extreme_features']}")

    ls = report["label_semantics"]
    print(f"\n🏷️  LABELS")
    print(f"   Label unique values  : {ls['label_column']['unique_values']}")
    print(f"   Attack unique values : {ls['attack_column']['unique_values']}")
    print(f"   Inferred binary col  : {ls['inferred_binary_column']}")
    print(f"   Inferred multi col   : {ls['inferred_multiclass_column']}")
    print(f"   Label distribution   : {ls['label_column']['distribution']}")
    print(f"   Attack distribution  : {dict(list(ls['attack_column']['distribution'].items())[:10])}")

    tp = report["temporal_provenance"]
    print(f"\n⏱️  TEMPORAL PROVENANCE")
    print(f"   Explicit time cols found  : {tp['explicit_temporal_columns_found']}")
    print(f"   Temporal order established: {tp['temporal_ordering_established']}")
    print(f"   Note: {tp['notes'][:200]}...")

    print(f"\n{sep}\n")

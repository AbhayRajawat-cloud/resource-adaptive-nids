"""
Stage 2 — Leak-free client preprocessing.

Implements the following pipeline per client, in strict order:

  1. Split client data into train (70%) / val (15%) / test (15%) with
     stratification on ``Label`` and a reproducible random seed.

  2. Identify benign rows in the training split.

  3. Fit per-feature 1st and 99th percentile bounds using ONLY
     benign training rows.

  4. Clip ALL splits using those fitted bounds.

  5. Apply per-feature Min-Max scaling to [0, 1] using the min/max
     values of the benign-training clip.

  6. Validate that all output values are finite and in [0, 1].

No validation or test data is ever seen during fitting.
No other client's data is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------

def stratified_split_indices(
    labels: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split of a label array into train / val / test index arrays.

    Stratification is performed per class so that each split mirrors
    the client's class distribution.

    Parameters
    ----------
    labels      : 1-D integer array of Label values.
    train_frac  : Fraction for training (default 0.70).
    val_frac    : Fraction for validation (default 0.15).
    seed        : Random seed.

    Returns
    -------
    Three index arrays (train_idx, val_idx, test_idx) expressed as
    positions in the supplied labels array.
    """
    test_frac = 1.0 - train_frac - val_frac
    assert abs(test_frac - 0.15) < 1e-9, "Fractions must sum to 1.0"

    rng = np.random.default_rng(seed)
    train_idx_list, val_idx_list, test_idx_list = [], [], []

    for cls in np.unique(labels):
        cls_positions = np.where(labels == cls)[0]
        perm = rng.permutation(len(cls_positions))
        cls_positions = cls_positions[perm]

        n = len(cls_positions)
        n_train = max(1, int(round(n * train_frac)))
        n_val   = max(1, int(round(n * val_frac)))
        n_test  = n - n_train - n_val
        if n_test < 0:
            # Edge case: very small class — give at least 1 to train
            n_train = n
            n_val = 0
            n_test = 0

        train_idx_list.append(cls_positions[:n_train])
        val_idx_list.append(cls_positions[n_train: n_train + n_val])
        test_idx_list.append(cls_positions[n_train + n_val:])

    return (
        np.sort(np.concatenate(train_idx_list)),
        np.sort(np.concatenate(val_idx_list)),
        np.sort(np.concatenate(test_idx_list)),
    )


# ---------------------------------------------------------------------------
# Benign-training-only scaler
# ---------------------------------------------------------------------------

class BenignTrainScaler:
    """
    Per-feature scaler fitted on benign training samples only.

    Procedure
    ---------
    1. Compute 1st and 99th percentile of each feature using benign train rows.
    2. Clip input values to [p01, p99].
    3. Scale to [0, 1] via min-max on the clipped benign-train distribution.

    The fitted parameters (p01, p99, clip_min, clip_max) are serialisable
    as a plain dict and saved alongside each client's processed data.
    """

    def __init__(self) -> None:
        self.p01: Optional[np.ndarray] = None
        self.p99: Optional[np.ndarray] = None
        self.clip_min: Optional[np.ndarray] = None
        self.clip_max: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    def fit(self, X_benign_train: np.ndarray, feature_names: List[str]) -> "BenignTrainScaler":
        """
        Fit scaler on benign training samples.

        Parameters
        ----------
        X_benign_train : 2D array (n_benign, n_features).
        feature_names  : List of feature column names (for serialisation).
        """
        if X_benign_train.shape[0] == 0:
            raise ValueError("Cannot fit scaler on empty benign training set.")

        self.feature_names = list(feature_names)
        self.p01 = np.percentile(X_benign_train, 1, axis=0)
        self.p99 = np.percentile(X_benign_train, 99, axis=0)

        # After clipping benign-train, compute min/max for scaling
        clipped = np.clip(X_benign_train, self.p01, self.p99)
        self.clip_min = clipped.min(axis=0)
        self.clip_max = clipped.max(axis=0)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply clip + min-max scaling. Fitted parameters must already be set."""
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted before transform.")
        X_clipped = np.clip(X, self.p01, self.p99)
        denom = self.clip_max - self.clip_min
        # Features with zero range (constant after clipping) → set to 0
        safe_denom = np.where(denom == 0, 1.0, denom)
        X_scaled = (X_clipped - self.clip_min) / safe_denom
        # Clamp to [0, 1] to handle any floating-point overshoot
        return np.clip(X_scaled, 0.0, 1.0)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise scaler parameters to a JSON-compatible dict."""
        if not self._fitted:
            raise RuntimeError("Scaler has not been fitted.")
        return {
            "feature_names": self.feature_names,
            "p01": self.p01.tolist(),
            "p99": self.p99.tolist(),
            "clip_min": self.clip_min.tolist(),
            "clip_max": self.clip_max.tolist(),
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenignTrainScaler":
        """Restore a scaler from a previously saved parameter dict."""
        obj = cls()
        obj.feature_names = d["feature_names"]
        obj.p01 = np.array(d["p01"])
        obj.p99 = np.array(d["p99"])
        obj.clip_min = np.array(d["clip_min"])
        obj.clip_max = np.array(d["clip_max"])
        obj._fitted = True
        return obj


# ---------------------------------------------------------------------------
# Per-client preprocessing orchestrator
# ---------------------------------------------------------------------------

def preprocess_client(
    client_df: pd.DataFrame,
    feature_cols: List[str],
    client_id: int,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Dict[str, Any]:
    """
    Full preprocessing for a single client's data partition.

    Parameters
    ----------
    client_df   : DataFrame slice for this client (original indices preserved).
    feature_cols: Verified feature column names.
    client_id   : Integer client identifier.
    seed        : Per-client reproducible seed (derived from global seed + client_id).
    train_frac  : Training fraction.
    val_frac    : Validation fraction.

    Returns
    -------
    Dict with keys:
      - split_indices       : {train, val, test} → arrays of positions within client_df
      - original_indices    : {train, val, test} → arrays of original CSV row indices
      - X_train, X_val, X_test          : float32 numpy arrays (scaled)
      - y_train, y_val, y_test          : int numpy arrays (Label)
      - attack_train, attack_val, attack_test : string arrays (Attack)
      - scaler                          : BenignTrainScaler (fitted)
      - scaler_params                   : dict of serialisable scaler params
      - client_stats                    : summary dict
    """
    labels = client_df["Label"].values.astype(np.int64)
    attacks = client_df["Attack"].values.astype(str)
    X_raw = client_df[feature_cols].values.astype(np.float64)
    original_idx = client_df.index.values  # original CSV row numbers

    # --- Split ---
    local_seed = seed + client_id  # unique per client, reproducible
    pos_train, pos_val, pos_test = stratified_split_indices(
        labels, train_frac=train_frac, val_frac=val_frac, seed=local_seed
    )

    # Verify no overlap between splits
    assert len(np.intersect1d(pos_train, pos_val)) == 0
    assert len(np.intersect1d(pos_train, pos_test)) == 0
    assert len(np.intersect1d(pos_val, pos_test)) == 0
    assert len(pos_train) + len(pos_val) + len(pos_test) == len(client_df)

    X_train_raw = X_raw[pos_train]
    X_val_raw   = X_raw[pos_val]
    X_test_raw  = X_raw[pos_test]

    y_train = labels[pos_train]
    y_val   = labels[pos_val]
    y_test  = labels[pos_test]

    atk_train = attacks[pos_train]
    atk_val   = attacks[pos_val]
    atk_test  = attacks[pos_test]

    # --- Fit scaler on BENIGN training rows ONLY ---
    benign_train_mask = y_train == 0
    X_benign_train = X_train_raw[benign_train_mask]
    n_benign_train = benign_train_mask.sum()

    if n_benign_train < 10:
        raise RuntimeError(
            f"Client {client_id}: too few benign training samples ({n_benign_train}) "
            "to fit a reliable scaler."
        )

    scaler = BenignTrainScaler()
    scaler.fit(X_benign_train, feature_cols)

    # --- Scale all splits ---
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val   = scaler.transform(X_val_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)

    # --- Compile client stats ---
    client_stats = {
        "client_id": client_id,
        "total_samples": len(client_df),
        "n_benign": int((labels == 0).sum()),
        "n_malicious": int((labels == 1).sum()),
        "benign_pct": round(float((labels == 0).mean()) * 100, 4),
        "split_sizes": {
            "train": int(len(pos_train)),
            "val":   int(len(pos_val)),
            "test":  int(len(pos_test)),
        },
        "benign_train_samples": int(n_benign_train),
        "attack_distribution": {
            k: int(v)
            for k, v in zip(*np.unique(attacks, return_counts=True))
        },
    }

    return {
        "split_indices": {
            "train": pos_train,
            "val":   pos_val,
            "test":  pos_test,
        },
        "original_indices": {
            "train": original_idx[pos_train],
            "val":   original_idx[pos_val],
            "test":  original_idx[pos_test],
        },
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "attack_train": atk_train, "attack_val": atk_val, "attack_test": atk_test,
        "scaler": scaler,
        "scaler_params": scaler.to_dict(),
        "client_stats": client_stats,
    }

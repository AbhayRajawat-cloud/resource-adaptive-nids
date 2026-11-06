"""
Stage 2 — Constrained Dirichlet federated data partitioning.

Implements sample-exclusive, constrained Dirichlet partitioning across K clients.

Design decisions
----------------
* Every record belongs to exactly ONE client (sample exclusivity).
* Partitioning uses the binary ``Label`` column for class-aware Dirichlet draws.
* A minimum of 1 000 benign samples per client is enforced via a
  constrained-retry redistribution procedure.
* alpha = inf is handled as uniform / IID allocation.
* The rare Worms class (164 rows) is NOT artificially distributed; it falls
  wherever the Dirichlet draw naturally places it.
* All operations are reproducible via the supplied random seed.
* Row indices from the original CSV are preserved for auditing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLIENTS = 10
MIN_BENIGN_PER_CLIENT = 1_000
MAX_REDISTRIBUTION_RETRIES = 50


# ---------------------------------------------------------------------------
# Core partitioning
# ---------------------------------------------------------------------------

def _dirichlet_split_class(
    indices: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """
    Split a class's index array across clients using a Dirichlet draw.

    For alpha = inf the distribution is uniform; otherwise it is sampled
    from Dir(alpha * ones_K).
    """
    n = len(indices)
    if n == 0:
        return [np.array([], dtype=np.int64) for _ in range(num_clients)]

    if math.isinf(alpha):
        # Uniform / IID split
        proportions = np.ones(num_clients) / num_clients
    else:
        proportions = rng.dirichlet(np.full(num_clients, alpha))

    # Convert proportions to integer counts (floor + remainder to last client)
    counts = (proportions * n).astype(int)
    remainder = n - counts.sum()
    counts[-1] += remainder  # assign leftover to last client

    # Shuffle indices then split
    perm = rng.permutation(n)
    shuffled = indices[perm]

    splits: List[np.ndarray] = []
    start = 0
    for c in counts:
        splits.append(shuffled[start: start + c])
        start += c
    return splits


def _enforce_benign_minimum(
    client_benign_splits: List[np.ndarray],
    client_malicious_splits: List[np.ndarray],
    min_benign: int,
    rng: np.random.Generator,
    max_retries: int = MAX_REDISTRIBUTION_RETRIES,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    """
    Ensure every client has at least min_benign benign samples.

    Strategy
    --------
    Clients that are below quota receive samples donated from the client with
    the largest benign surplus. Donations are tracked and reported.
    Sample exclusivity is maintained throughout (no duplication).

    Raises
    ------
    RuntimeError if the minimum cannot be satisfied after max_retries.
    """
    benign = [list(s) for s in client_benign_splits]
    malicious = [list(s) for s in client_malicious_splits]
    redistribution_log: List[str] = []

    for attempt in range(max_retries):
        deficits = [
            (i, min_benign - len(benign[i]))
            for i in range(len(benign))
            if len(benign[i]) < min_benign
        ]
        if not deficits:
            break   # all clients satisfy quota

        for deficit_client, needed in deficits:
            # Find the client with the largest surplus
            surpluses = [
                (i, len(benign[i]) - min_benign)
                for i in range(len(benign))
                if i != deficit_client and len(benign[i]) - min_benign >= needed
            ]
            if not surpluses:
                raise RuntimeError(
                    f"[Partition] Cannot satisfy min_benign={min_benign} for "
                    f"client {deficit_client} after attempt {attempt}. "
                    "Reduce alpha or lower min_benign threshold."
                )
            # Donate from the richest
            donor = max(surpluses, key=lambda x: x[1])[0]
            donate = rng.choice(len(benign[donor]), size=needed, replace=False)
            donate_indices = [benign[donor][j] for j in donate]
            # Remove from donor
            donor_set = list(benign[donor])
            for j in sorted(donate, reverse=True):
                donor_set.pop(j)
            benign[donor] = donor_set
            benign[deficit_client].extend(donate_indices)
            redistribution_log.append(
                f"Donated {needed} benign samples from client {donor} "
                f"to client {deficit_client}"
            )
    else:
        raise RuntimeError(
            f"[Partition] Failed to satisfy benign minimum in {max_retries} retries."
        )

    return (
        [np.array(s, dtype=np.int64) for s in benign],
        [np.array(s, dtype=np.int64) for s in malicious],
        {"redistribution_steps": redistribution_log},
    )


def partition_dirichlet(
    df: pd.DataFrame,
    num_clients: int = NUM_CLIENTS,
    alpha: float = 0.5,
    seed: int = 42,
    min_benign: int = MIN_BENIGN_PER_CLIENT,
) -> Dict[int, np.ndarray]:
    """
    Partition the full dataset into num_clients non-overlapping index arrays.

    Parameters
    ----------
    df          : Full dataset DataFrame (must contain 'Label' column).
    num_clients : Number of federated clients (K).
    alpha       : Dirichlet concentration. Use math.inf for IID/uniform split.
    seed        : Random seed for reproducibility.
    min_benign  : Minimum benign samples required per client.

    Returns
    -------
    Dict mapping client_id (int) -> np.ndarray of original row indices.
    """
    rng = np.random.default_rng(seed)

    benign_idx = np.where(df["Label"].values == 0)[0]
    malicious_idx = np.where(df["Label"].values == 1)[0]

    print(f"[Partition] alpha={alpha}, K={num_clients}, seed={seed}")
    print(f"[Partition] Benign: {len(benign_idx):,}  |  Malicious: {len(malicious_idx):,}")

    benign_splits = _dirichlet_split_class(benign_idx, num_clients, alpha, rng)
    malicious_splits = _dirichlet_split_class(malicious_idx, num_clients, alpha, rng)

    # Enforce minimum benign quota
    benign_splits, malicious_splits, redist_log = _enforce_benign_minimum(
        benign_splits, malicious_splits, min_benign, rng
    )
    if redist_log["redistribution_steps"]:
        print(f"[Partition] Redistribution applied ({len(redist_log['redistribution_steps'])} steps)")

    # Merge benign + malicious per client
    client_indices: Dict[int, np.ndarray] = {}
    for i in range(num_clients):
        merged = np.concatenate([benign_splits[i], malicious_splits[i]])
        client_indices[i] = np.sort(merged).astype(np.int64)

    # Verify sample exclusivity
    all_assigned = np.concatenate(list(client_indices.values()))
    assert len(all_assigned) == len(df), \
        f"Sample count mismatch: {len(all_assigned)} assigned vs {len(df)} total"
    assert len(np.unique(all_assigned)) == len(all_assigned), \
        "Duplicate indices detected — sample exclusivity violated"

    # Per-client statistics
    for i in range(num_clients):
        cidx = client_indices[i]
        n_benign = int((df["Label"].values[cidx] == 0).sum())
        n_mal = int((df["Label"].values[cidx] == 1).sum())
        print(f"  Client {i:2d}: {len(cidx):7,} rows | benign={n_benign:7,} | malicious={n_mal:5,}")

    return client_indices

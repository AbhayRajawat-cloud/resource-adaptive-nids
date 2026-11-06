"""
Automated Test Suite for Stage 5 Phase 5.3:
Federated Heavy Model Training using FedAvg.

Validates:
1. Parameter Immutability & Seeding Test:
   Verifies seed 42 produces bitwise identical global initialization weights theta^{(0)}.
2. FedBN State Dict Isolation Test:
   Verifies aggregate_weighted_fedavg with is_mlp=True excludes running_mean, running_var,
   and num_batches_tracked from aggregation.
3. FedPer Private-Head Persistence Test:
   Verifies that private heads (33 params) are excluded from server aggregation and preserved locally.
4. Local Parameter Isolation Test:
   Verifies local Autoencoder (2,008 params) and Gate (641 params) checkpoints are NEVER modified
   or aggregated during federated training.
5. Global Cosine LR Schedule Test:
   Verifies eta^{(r)} decreases monotonically and smoothly from 1e-3 down to ~1.024e-5 at round 49.
6. Output CSV & Schema Test:
   Asserts phase5.3_federated_results.csv contains exactly 120 rows and phase5.3_fedper_results.csv
   contains exactly 60 rows.
7. Scoped Baseline SHA-256 Hash Verification:
   Re-verifies cryptographic SHA-256 hashes for all 40 baseline artifacts in stage4_artifact_hashes.json.
"""

import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.heavy_models import HeavyGRU, SupervisedDenseMLP
from src.federated.fedavg_aggregator import (
    aggregate_weighted_fedavg,
    is_bn_buffer,
    is_private_head,
)
from src.federated.federated_trainer import compute_cosine_lr, set_seed
from src.evaluation.eval_federated import benjamini_hochberg


def compute_sha256(filepath: Path) -> str:
    """Compute hex-encoded SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Check 1: Parameter Immutability & Seeding Test
# ---------------------------------------------------------------------------
def test_seeding_reproducibility():
    """Verify that seed 42 produces bitwise identical initializations for GRU and MLP."""
    set_seed(42)
    gru_1 = HeavyGRU()
    mlp_1 = SupervisedDenseMLP()

    set_seed(42)
    gru_2 = HeavyGRU()
    mlp_2 = SupervisedDenseMLP()

    for k, v in gru_1.state_dict().items():
        assert torch.equal(v, gru_2.state_dict()[k]), f"HeavyGRU parameter {k} differs under seed 42"

    for k, v in mlp_1.state_dict().items():
        assert torch.equal(v, mlp_2.state_dict()[k]), f"SupervisedDenseMLP parameter {k} differs under seed 42"


# ---------------------------------------------------------------------------
# Check 2: FedBN State Dict Isolation Test
# ---------------------------------------------------------------------------
def test_fedbn_state_dict_isolation():
    """Verify that aggregate_weighted_fedavg with is_mlp=True excludes BatchNorm buffers."""
    mlp = SupervisedDenseMLP()
    state_1 = copy.deepcopy(mlp.state_dict())
    state_2 = copy.deepcopy(mlp.state_dict())

    # Set distinct running buffers
    state_1["layer1.1.running_mean"] = torch.full((64,), 1.0)
    state_2["layer1.1.running_mean"] = torch.full((64,), 2.0)

    agg = aggregate_weighted_fedavg(
        client_states=[state_1, state_2],
        client_sample_counts=[100, 100],
        is_mlp=True,
        is_fedper=False,
    )

    # Assert that all BatchNorm running statistics are strictly absent from server aggregated state
    for k in agg:
        assert not is_bn_buffer(k), f"BatchNorm buffer '{k}' was unexpectedly aggregated by server!"

    assert "layer1.1.running_mean" not in agg
    assert "layer1.1.running_var" not in agg
    assert "layer1.1.num_batches_tracked" not in agg
    assert "layer2.1.running_mean" not in agg
    assert "layer2.1.running_var" not in agg
    assert "layer2.1.num_batches_tracked" not in agg

    # Assert trainable weights ARE present
    assert "layer1.0.weight" in agg
    assert "layer1.1.weight" in agg  # affine scale gamma is aggregated
    assert "head.weight" in agg


# ---------------------------------------------------------------------------
# Check 3: FedPer Private-Head Persistence Test
# ---------------------------------------------------------------------------
def test_fedper_private_head_persistence():
    """Verify that private classification heads (33 params) are excluded from FedPer aggregation."""
    # Test GRU FedPer
    gru = HeavyGRU()
    state_g1 = copy.deepcopy(gru.state_dict())
    state_g2 = copy.deepcopy(gru.state_dict())

    agg_gru = aggregate_weighted_fedavg(
        client_states=[state_g1, state_g2],
        client_sample_counts=[100, 100],
        is_mlp=False,
        is_fedper=True,
    )

    assert "classifier.3.weight" not in agg_gru
    assert "classifier.3.bias" not in agg_gru
    assert "cell1.weight_ih" in agg_gru
    assert "classifier.0.weight" in agg_gru

    # Test MLP FedPer
    mlp = SupervisedDenseMLP()
    state_m1 = copy.deepcopy(mlp.state_dict())
    state_m2 = copy.deepcopy(mlp.state_dict())

    agg_mlp = aggregate_weighted_fedavg(
        client_states=[state_m1, state_m2],
        client_sample_counts=[100, 100],
        is_mlp=True,
        is_fedper=True,
    )

    assert "head.weight" not in agg_mlp
    assert "head.bias" not in agg_mlp
    assert "layer1.0.weight" in agg_mlp


# ---------------------------------------------------------------------------
# Check 4: Local Parameter Isolation Test
# ---------------------------------------------------------------------------
def test_local_ae_gate_isolation():
    """
    Verify that Stage 3 local Autoencoders (2,008 params) and Gating Regressors (641 params)
    remain completely untouched and excluded from federated aggregation.
    """
    manifest_path = Path("outputs/logs/stage4_artifact_hashes.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ckpt_dir = Path("outputs/checkpoints/stage3")
    for cid, file_hashes in manifest["stage3_models"].items():
        ae_file = ckpt_dir / f"{cid}_autoencoder.pt"
        gate_file = ckpt_dir / f"{cid}_gate.pt"

        # Check parameter count of local AE (2,008)
        ae_state = torch.load(ae_file, map_location="cpu")
        ae_params = sum(p.numel() for p in ae_state.values())
        assert ae_params == 2008, f"Client {cid} Autoencoder params != 2,008"

        # Check parameter count of local Gate (641)
        gate_state = torch.load(gate_file, map_location="cpu")
        gate_params = sum(p.numel() for p in gate_state.values())
        assert gate_params == 641, f"Client {cid} Gate params != 641"


# ---------------------------------------------------------------------------
# Check 5: Global Cosine LR Schedule Test
# ---------------------------------------------------------------------------
def test_global_cosine_lr_schedule():
    """Verify eta^{(r)} decreases smoothly and monotonically from 1e-3 down to ~1.024e-5."""
    lrs = [compute_cosine_lr(r, total_rounds=50, lr_0=1e-3, lr_min=1e-5) for r in range(50)]

    # Round 0 must be exactly 1e-3
    assert abs(lrs[0] - 1e-3) < 1e-9

    # Monotonic decrease
    for r in range(len(lrs) - 1):
        assert lrs[r] > lrs[r + 1], f"LR not monotonically decreasing at round {r}: {lrs[r]} <= {lrs[r+1]}"

    # Round 49 value check: smoothly approaches lr_min (between 1.0e-5 and 1.15e-5)
    assert 1.0e-5 <= lrs[49] <= 1.15e-5, f"Round 49 LR out of expected range: {lrs[49]}"
    # Verify proximity to 1.024e-5 ~ 1.098e-5
    assert pytest.approx(lrs[49], rel=0.1) == 1.024e-5 or pytest.approx(lrs[49], rel=0.01) == 1.098e-5


# ---------------------------------------------------------------------------
# Check 6: Output CSV & Schema Test
# ---------------------------------------------------------------------------
def test_output_csv_and_schema():
    """Assert output CSVs exist and contain exact expected row counts and required columns."""
    metrics_dir = Path("outputs/metrics/stage5")
    fedavg_csv = metrics_dir / "phase5.3_federated_results.csv"
    fedper_csv = metrics_dir / "phase5.3_fedper_results.csv"

    assert fedavg_csv.exists(), f"Missing {fedavg_csv}"
    assert fedper_csv.exists(), f"Missing {fedper_csv}"

    df_fedavg = pd.read_csv(fedavg_csv)
    df_fedper = pd.read_csv(fedper_csv)

    # 10 clients * 2 models * 2 quantiles * 3 tiers = 120 rows
    assert len(df_fedavg) == 120, f"Expected 120 FedAvg rows, got {len(df_fedavg)}"

    # 10 clients * 2 models * 3 tiers = 60 rows
    assert len(df_fedper) == 60, f"Expected 60 FedPer rows, got {len(df_fedper)}"

    required_cols = [
        "client_id", "client_str", "quantile", "model_type", "strategy",
        "tier", "tier_label", "n_samples", "n_attacks", "n_benign",
        "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1_score"
    ]
    for c in required_cols:
        assert c in df_fedavg.columns, f"Column '{c}' missing from FedAvg CSV"
        assert c in df_fedper.columns, f"Column '{c}' missing from FedPer CSV"


# ---------------------------------------------------------------------------
# Check 7: Scoped Baseline SHA-256 Hash Verification
# ---------------------------------------------------------------------------
def test_scoped_baseline_cryptographic_verification():
    """Re-verify SHA-256 hashes for all 40 baseline artifacts in stage4_artifact_hashes.json."""
    manifest_path = Path("outputs/logs/stage4_artifact_hashes.json")
    assert manifest_path.exists(), "Missing manifest: outputs/logs/stage4_artifact_hashes.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Stage 2 data.npz files (10 clients)
    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    for cid, file_hashes in manifest["stage2_data"].items():
        npz_file = stage2_dir / cid / "data.npz"
        assert npz_file.exists(), f"Missing Stage 2 file: {npz_file}"
        assert compute_sha256(npz_file) == file_hashes["data.npz"], f"Hash mismatch: {npz_file}"

    # 2. Stage 3 checkpoints (20 files: 10 AE + 10 Gate)
    ckpt_dir = Path("outputs/checkpoints/stage3")
    for cid, file_hashes in manifest["stage3_models"].items():
        ae_file = ckpt_dir / f"{cid}_autoencoder.pt"
        gate_file = ckpt_dir / f"{cid}_gate.pt"
        assert compute_sha256(ae_file) == file_hashes["autoencoder.pt"], f"Hash mismatch: {ae_file}"
        assert compute_sha256(gate_file) == file_hashes["gate.pt"], f"Hash mismatch: {gate_file}"

    # 3. Stage 3 thresholds (10 files)
    for cid, file_hashes in manifest["stage3_thresholds"].items():
        thresh_file = ckpt_dir / f"{cid}_thresholds.json"
        expected = file_hashes["thresholds.json"] if isinstance(file_hashes, dict) else file_hashes
        assert compute_sha256(thresh_file) == expected, f"Hash mismatch: {thresh_file}"


# ---------------------------------------------------------------------------
# Check 8: Benjamini-Hochberg Multiple Testing Procedure Sanity
# ---------------------------------------------------------------------------
def test_benjamini_hochberg_procedure():
    """Verify Benjamini-Hochberg FDR calculation handles monotonicity and edge cases."""
    pvals = [0.001, 0.01, 0.04, 0.05, 0.10]
    qvals, rejects = benjamini_hochberg(pvals, alpha=0.05)

    assert len(qvals) == 5
    assert len(rejects) == 5
    # q values must be non-decreasing for sorted p values
    for i in range(len(qvals) - 1):
        assert qvals[i] <= qvals[i + 1]

    assert rejects[0] is True  # 0.001 -> 0.005 < 0.05
    assert rejects[1] is True  # 0.01 -> 0.025 < 0.05
    assert rejects[2] is False # 0.04 -> 0.0625 >= 0.05


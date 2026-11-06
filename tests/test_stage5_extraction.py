"""
Automated Validation Unit Test Suite for Stage 5 Phase 5.1:
Gate-Passed Dataset Extraction & Three-Tier Lineage Preservation.

Validates:
1. Independent Gate Prediction & Mask Alignment:
   Recomputes Gate inference on raw Stage 2 arrays and verifies exact bitwise match with extracted arrays.
2. Pairwise Partition Exclusivity:
   Verifies that original dataset indices have pairwise empty intersections across train, val, and test.
3. Three-Tier Lineage & Vector Uniqueness Checks:
   Verifies Tier 1, Tier 2, Tier 3 counts, masks, and strict subset property (Tier 3 => Tier 2).
4. Client 02 Integrity Check:
   Verifies Client 02 extracted partitions exist for Q99 and Q95 with non-zero counts and intact 38-D features.
5. Scoped Baseline Cryptographic Verification:
   Validates SHA-256 hashes of all artifacts listed in outputs/logs/stage4_artifact_hashes.json.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.gate import GatingRegressor


def compute_sha256(filepath: Path) -> str:
    """Compute hex-encoded SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_scoped_baseline_cryptographic_verification():
    """Verify SHA-256 hashes for all files explicitly listed in stage4_artifact_hashes.json."""
    manifest_path = Path("outputs/logs/stage4_artifact_hashes.json")
    assert manifest_path.exists(), "Missing manifest: outputs/logs/stage4_artifact_hashes.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Stage 2 data.npz files
    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    for cid, file_hashes in manifest["stage2_data"].items():
        npz_file = stage2_dir / cid / "data.npz"
        assert npz_file.exists(), f"Missing Stage 2 file: {npz_file}"
        assert compute_sha256(npz_file) == file_hashes["data.npz"], f"Hash mismatch: {npz_file}"

    # 2. Stage 3 checkpoints
    ckpt_dir = Path("outputs/checkpoints/stage3")
    for cid, file_hashes in manifest["stage3_models"].items():
        ae_file = ckpt_dir / f"{cid}_autoencoder.pt"
        gate_file = ckpt_dir / f"{cid}_gate.pt"
        assert compute_sha256(ae_file) == file_hashes["autoencoder.pt"], f"Hash mismatch: {ae_file}"
        assert compute_sha256(gate_file) == file_hashes["gate.pt"], f"Hash mismatch: {gate_file}"

    # 3. Stage 3 thresholds
    for cid, file_hashes in manifest["stage3_thresholds"].items():
        thresh_file = ckpt_dir / f"{cid}_thresholds.json"
        expected = file_hashes["thresholds.json"] if isinstance(file_hashes, dict) else file_hashes
        assert compute_sha256(thresh_file) == expected, f"Hash mismatch: {thresh_file}"


@pytest.mark.parametrize("quantile", ["Q99", "Q95"])
def test_independent_gate_prediction_alignment(quantile: str):
    """
    Verify that independently recomputing Gate forward-pass on raw Stage 2 data
    reproduces extracted arrays bitwise.
    """
    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    ckpt_dir = Path("outputs/checkpoints/stage3")
    gate_passed_dir = Path(f"data/processed/gate_passed/{quantile}")

    for cid_int in [0, 2, 4, 9]: # Spot-check representative clients including Client 02 and 04
        cid_str = f"client_{cid_int:02d}"
        raw_data = np.load(stage2_dir / cid_str / "data.npz", allow_pickle=True)
        x_test_raw = raw_data["X_test"]
        y_test_raw = raw_data["y_test"]
        orig_test_raw = raw_data["orig_idx_test"]

        passed_data = np.load(gate_passed_dir / f"{cid_str}_passed.npz")
        tau_gate = float(passed_data["tau_gate"])

        # Independent Gate inference
        gate = GatingRegressor(input_dim=38).eval()
        gate.load_state_dict(torch.load(ckpt_dir / f"{cid_str}_gate.pt", map_location="cpu"))
        with torch.no_grad():
            preds = gate(torch.tensor(x_test_raw, dtype=torch.float32)).squeeze(1).numpy()

        expected_mask = preds >= tau_gate

        np.testing.assert_array_equal(passed_data["x_test_passed"], x_test_raw[expected_mask])
        np.testing.assert_array_equal(passed_data["y_test_passed"], y_test_raw[expected_mask])
        np.testing.assert_array_equal(passed_data["orig_idx_test_passed"], orig_test_raw[expected_mask])


@pytest.mark.parametrize("quantile", ["Q99", "Q95"])
def test_pairwise_partition_exclusivity(quantile: str):
    """Verify original dataset indices maintain pairwise empty intersections across train, val, and test."""
    gate_passed_dir = Path(f"data/processed/gate_passed/{quantile}")

    for cid_int in range(10):
        cid_str = f"client_{cid_int:02d}"
        data = np.load(gate_passed_dir / f"{cid_str}_passed.npz")

        idx_tr = set(data["orig_idx_train_passed"].tolist())
        idx_val = set(data["orig_idx_val_passed"].tolist())
        idx_te = set(data["orig_idx_test_passed"].tolist())

        assert len(idx_tr.intersection(idx_val)) == 0, f"{cid_str} {quantile} Train-Val index intersection non-empty"
        assert len(idx_tr.intersection(idx_te)) == 0, f"{cid_str} {quantile} Train-Test index intersection non-empty"
        assert len(idx_val.intersection(idx_te)) == 0, f"{cid_str} {quantile} Val-Test index intersection non-empty"


@pytest.mark.parametrize("quantile", ["Q99", "Q95"])
def test_three_tier_lineage_and_uniqueness(quantile: str):
    """Verify Three-Tier lineage hierarchy, subset relations, and uniqueness properties."""
    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    gate_passed_dir = Path(f"data/processed/gate_passed/{quantile}")

    for cid_int in range(10):
        cid_str = f"client_{cid_int:02d}"
        passed_data = np.load(gate_passed_dir / f"{cid_str}_passed.npz")
        raw_data = np.load(stage2_dir / cid_str / "data.npz", allow_pickle=True)

        x_test_passed = passed_data["x_test_passed"]
        tier2_mask = passed_data["tier2_mask_test"]
        tier3_mask = passed_data["tier3_mask_test"]

        n_pass = len(x_test_passed)
        t1_count = int(passed_data["tier1_test_count"])
        t2_count = int(passed_data["tier2_test_count"])
        t3_count = int(passed_data["tier3_test_count"])

        # 1. Counts match masks
        assert t1_count == n_pass
        assert t2_count == int(np.sum(tier2_mask))
        assert t3_count == int(np.sum(tier3_mask))

        # 2. Strict hierarchy: Tier 3 <= Tier 2 <= Tier 1
        assert t3_count <= t2_count <= t1_count

        # 3. Implication: tier3_mask[i] => tier2_mask[i]
        assert np.all(tier3_mask <= tier2_mask)

        # 4. Rigorous vector check for Tier 3:
        # Every True vector in tier3_mask must NOT appear anywhere in x_train or x_val
        if t3_count > 0:
            dim = 38
            dtype_void = np.dtype((np.void, x_test_passed.dtype.itemsize * dim))
            void_train_val = np.unique(np.concatenate([
                np.ascontiguousarray(raw_data["X_train"]).view(dtype_void).ravel(),
                np.ascontiguousarray(raw_data["X_val"]).view(dtype_void).ravel(),
            ]))

            tier3_vectors_void = np.ascontiguousarray(x_test_passed[tier3_mask]).view(dtype_void).ravel()
            assert not np.any(np.isin(tier3_vectors_void, void_train_val)), (
                f"{cid_str} {quantile}: Detected Tier 3 test vector appearing in train/val partition!"
            )


def test_client_02_integrity_check():
    """Verify Client 02 integration: files exist, non-zero samples, correct schema."""
    for q_name in ["Q99", "Q95"]:
        p = Path(f"data/processed/gate_passed/{q_name}/client_02_passed.npz")
        assert p.exists(), f"Client 02 {q_name} file missing: {p}"

        data = np.load(p)
        assert data["client_id"] == 2
        assert data["quantile_name"] == q_name
        assert data["x_train_passed"].shape[1] == 38
        assert data["x_test_passed"].shape[1] == 38
        assert len(data["x_train_passed"]) > 0, f"Client 02 {q_name} train passed is unexpectedly empty"
        assert len(data["x_test_passed"]) > 0, f"Client 02 {q_name} test passed is unexpectedly empty"
        assert int(data["tier3_test_count"]) > 0, f"Client 02 {q_name} Tier 3 count must be > 0"


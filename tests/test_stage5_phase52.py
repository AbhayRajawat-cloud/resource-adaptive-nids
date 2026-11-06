"""
Automated Validation Suite for Stage 5 Phase 5.2:
Local Standalone GRU & MLP Baseline Training.

Validates the 10 core verification criteria specified in Section 9:
1. Scoped Baseline Cryptographic Immutability (all 40 Stage 1-4 baseline files).
2. Phase 5.1 NPZ File Existence and Schema Integrity (all 20 extraction files).
3. Exact Parameter Counts for HeavyGRU (47,041) and SupervisedDenseMLP (4,801).
4. Correct Input Shapes for W=1 and Structural W=10 Benchmark.
5. Correct Class-Weight Computation and Zero-Class Safeguards.
6. BatchNorm Safety for Small Final Batches and SafeBatchSampler.
7. Train / Validation / Test Partition Index Disjointness.
8. Exact Three-Tier Lineage Correspondence (Tier 3 <= Tier 2 <= Tier 1).
9. Robust Evaluation Metric Handling for Single-Class Subsets (ROC-AUC / PR-AUC -> None).
10. Training Reproducibility & Checkpoint Integrity (seed 42 exact match, best epoch loss).
"""

import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.heavy_models import (
    HeavyGRU,
    SupervisedDenseMLP,
    SafeBatchNorm1d,
    count_parameters,
)
from src.data.phase52_dataset import (
    compute_positive_class_weight,
    SafeBatchSampler,
    GatePassedPartitionDataset,
    load_client_gate_passed_data,
    create_safe_dataloader,
)
from src.evaluation.eval_local_standalone import (
    evaluate_binary_predictions,
    compute_hodges_lehmann_and_ci,
)


def compute_sha256(filepath: Path) -> str:
    """Compute hex-encoded SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Check 1: Scoped Baseline Cryptographic Immutability
# ---------------------------------------------------------------------------
def test_scoped_baseline_cryptographic_immutability():
    """Verify SHA-256 hashes for all 40 baseline files listed in stage4_artifact_hashes.json."""
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
# Check 2: Phase 5.1 NPZ File Existence and Schema Integrity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quantile", ["Q99", "Q95"])
def test_phase51_npz_existence_and_schema(quantile: str):
    """Verify existence and complete schema for all 10 client NPZ files per quantile."""
    base_dir = Path("data/processed/gate_passed") / quantile
    assert base_dir.exists(), f"Missing gate_passed directory: {base_dir}"

    required_keys = [
        "client_id", "quantile_name", "tau_gate",
        "x_train_passed", "y_train_passed", "orig_idx_train_passed",
        "benign_count_train_passed", "attack_count_train_passed",
        "x_val_passed", "y_val_passed", "orig_idx_val_passed",
        "x_test_passed", "y_test_passed", "orig_idx_test_passed",
        "tier1_test_count", "tier2_test_count", "tier3_test_count",
        "tier2_mask_test", "tier3_mask_test",
    ]

    for cid in range(10):
        npz_path = base_dir / f"client_{cid:02d}_passed.npz"
        assert npz_path.exists(), f"Missing Phase 5.1 dataset: {npz_path}"

        data = np.load(npz_path)
        for k in required_keys:
            assert k in data, f"Key '{k}' missing from {npz_path}"

        # Feature dimensions must be strictly 38
        assert data["x_train_passed"].shape[1] == 38, f"Feature dim != 38 in train for {npz_path}"
        assert data["x_val_passed"].shape[1] == 38, f"Feature dim != 38 in val for {npz_path}"
        assert data["x_test_passed"].shape[1] == 38, f"Feature dim != 38 in test for {npz_path}"

        # Labels and indices alignment
        assert len(data["x_train_passed"]) == len(data["y_train_passed"]) == len(data["orig_idx_train_passed"])
        assert len(data["x_val_passed"]) == len(data["y_val_passed"]) == len(data["orig_idx_val_passed"])
        assert len(data["x_test_passed"]) == len(data["y_test_passed"]) == len(data["orig_idx_test_passed"])

        # Lineage mask length
        assert len(data["tier2_mask_test"]) == len(data["x_test_passed"])
        assert len(data["tier3_mask_test"]) == len(data["x_test_passed"])


# ---------------------------------------------------------------------------
# Check 3: Exact Parameter Counts
# ---------------------------------------------------------------------------
def test_exact_model_parameter_counts():
    """Verify exact parameter counts: HeavyGRU = 47,041, SupervisedDenseMLP = 4,801."""
    gru = HeavyGRU(input_dim=38, hidden_dim=64, dropout_rate=0.2)
    mlp = SupervisedDenseMLP(input_dim=38, dropout_rate=0.2, negative_slope=0.1)

    gru_params = count_parameters(gru)
    mlp_params = count_parameters(mlp)

    assert gru_params == 47041, f"HeavyGRU parameter count mismatch: expected 47,041, got {gru_params:,}"
    assert mlp_params == 4801, f"SupervisedDenseMLP parameter count mismatch: expected 4,801, got {mlp_params:,}"


# ---------------------------------------------------------------------------
# Check 4: Input Shapes (W=1 and Structural W=10 Benchmark)
# ---------------------------------------------------------------------------
def test_model_input_shapes_and_outputs():
    """Verify forward pass under W=1 (2D and 3D) and structural W=10 benchmark."""
    gru = HeavyGRU(input_dim=38, hidden_dim=64)
    mlp = SupervisedDenseMLP(input_dim=38)

    gru.eval()
    mlp.eval()

    batch_size = 8

    # 1. W=1 formulated as 2D (batch_size, 38)
    x_2d = torch.randn(batch_size, 38)
    out_gru_2d = gru(x_2d)
    out_mlp_2d = mlp(x_2d)

    assert out_gru_2d.shape == (batch_size,), f"GRU 2D output shape mismatch: {out_gru_2d.shape}"
    assert out_mlp_2d.shape == (batch_size,), f"MLP 2D output shape mismatch: {out_mlp_2d.shape}"

    # 2. W=1 formulated as 3D (batch_size, 1, 38)
    x_3d_w1 = torch.randn(batch_size, 1, 38)
    out_gru_3d_w1 = gru(x_3d_w1)
    out_mlp_3d_w1 = mlp(x_3d_w1)

    assert out_gru_3d_w1.shape == (batch_size,), f"GRU 3D W=1 output shape mismatch: {out_gru_3d_w1.shape}"
    assert out_mlp_3d_w1.shape == (batch_size,), f"MLP 3D W=1 output shape mismatch: {out_mlp_3d_w1.shape}"

    # 3. Structural W=10 benchmark for GRU: shape (batch_size, 10, 38)
    x_3d_w10 = torch.randn(batch_size, 10, 38)
    out_gru_3d_w10 = gru(x_3d_w10)

    assert out_gru_3d_w10.shape == (batch_size,), f"GRU W=10 benchmark shape mismatch: {out_gru_3d_w10.shape}"


# ---------------------------------------------------------------------------
# Check 5: Class-Weight Computation and Zero-Class Safeguards
# ---------------------------------------------------------------------------
def test_class_weight_computation_and_safeguards():
    """Verify positive class weight formula w_pos = min(10.0, N_benign / N_attack) and edge cases."""
    # Standard unbalanced case: 100 benign, 20 attack -> ratio = 5.0 <= 10.0
    w, fb = compute_positive_class_weight(100, 20, max_weight=10.0)
    assert w == 5.0
    assert fb is False

    # Highly unbalanced case: 200 benign, 10 attack -> ratio = 20.0 > 10.0 -> cap at 10.0
    w_cap, fb_cap = compute_positive_class_weight(200, 10, max_weight=10.0)
    assert w_cap == 10.0
    assert fb_cap is False

    # Attack heavy case: 50 benign, 100 attack -> ratio = 0.5
    w_atk, fb_atk = compute_positive_class_weight(50, 100, max_weight=10.0)
    assert pytest.approx(w_atk, 1e-5) == 0.5
    assert fb_atk is False

    # Zero attack safeguard
    w_zero_atk, fb_zero_atk = compute_positive_class_weight(100, 0, max_weight=10.0)
    assert w_zero_atk == 1.0
    assert fb_zero_atk is True

    # Zero benign safeguard
    w_zero_ben, fb_zero_ben = compute_positive_class_weight(0, 100, max_weight=10.0)
    assert w_zero_ben == 1.0
    assert fb_zero_ben is True


# ---------------------------------------------------------------------------
# Check 6: BatchNorm Safety for Small Final Batches and SafeBatchSampler
# ---------------------------------------------------------------------------
def test_batchnorm_safety_and_sampler():
    """Verify SafeBatchNorm1d does not crash on N=1 and SafeBatchSampler merges remainder."""
    bn = SafeBatchNorm1d(32)
    bn.train()

    # Pass an initial multi-sample batch so running stats are populated
    x_multi = torch.randn(8, 32)
    bn(x_multi)

    # Now pass a single sample in train mode: standard nn.BatchNorm1d raises ValueError
    x_single = torch.randn(1, 32)
    try:
        out_single = bn(x_single)
        assert out_single.shape == (1, 32)
    except ValueError as e:
        pytest.fail(f"SafeBatchNorm1d crashed on single-sample batch: {e}")

    # SafeBatchSampler tests
    # Case A: N=33 with batch_size=32 -> remainder 1 -> merges into preceding batch -> 1 batch of size 33
    sampler_33 = SafeBatchSampler(data_source_len=33, batch_size=32, shuffle=False)
    batches_33 = list(sampler_33)
    assert len(batches_33) == 1
    assert len(batches_33[0]) == 33
    assert len(sampler_33) == 1

    # Case B: N=65 with batch_size=32 -> batches of 32, 33
    sampler_65 = SafeBatchSampler(data_source_len=65, batch_size=32, shuffle=False)
    batches_65 = list(sampler_65)
    assert len(batches_65) == 2
    assert len(batches_65[0]) == 32
    assert len(batches_65[1]) == 33
    assert len(sampler_65) == 2

    # Case C: N=1 with batch_size=32 -> 1 batch of size 1 (cannot merge into prior batch)
    sampler_1 = SafeBatchSampler(data_source_len=1, batch_size=32, shuffle=False)
    batches_1 = list(sampler_1)
    assert len(batches_1) == 1
    assert len(batches_1[0]) == 1


# ---------------------------------------------------------------------------
# Check 7: Partition Isolation (Index Disjointness)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quantile", ["Q99", "Q95"])
def test_partition_isolation_pairwise_disjointness(quantile: str):
    """Verify train, val, and test partitions have pairwise disjoint original indices."""
    base_dir = Path("data/processed/gate_passed") / quantile

    for cid in range(10):
        npz_path = base_dir / f"client_{cid:02d}_passed.npz"
        data = np.load(npz_path)

        train_idx = set(data["orig_idx_train_passed"].tolist())
        val_idx = set(data["orig_idx_val_passed"].tolist())
        test_idx = set(data["orig_idx_test_passed"].tolist())

        assert len(train_idx.intersection(val_idx)) == 0, f"Train-Val index overlap in Client {cid:02d} ({quantile})"
        assert len(train_idx.intersection(test_idx)) == 0, f"Train-Test index overlap in Client {cid:02d} ({quantile})"
        assert len(val_idx.intersection(test_idx)) == 0, f"Val-Test index overlap in Client {cid:02d} ({quantile})"


# ---------------------------------------------------------------------------
# Check 8: Exact Three-Tier Lineage Correspondence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("quantile", ["Q99", "Q95"])
def test_three_tier_lineage_correspondence(quantile: str):
    """Verify Tier 3 <= Tier 2 <= Tier 1 lineage hierarchy and mask consistency."""
    base_dir = Path("data/processed/gate_passed") / quantile

    for cid in range(10):
        npz_path = base_dir / f"client_{cid:02d}_passed.npz"
        data = np.load(npz_path)

        t1_count = int(data["tier1_test_count"])
        t2_count = int(data["tier2_test_count"])
        t3_count = int(data["tier3_test_count"])

        assert t3_count <= t2_count <= t1_count == len(data["x_test_passed"]), (
            f"Lineage count violation in Client {cid:02d} ({quantile}): T3={t3_count}, T2={t2_count}, T1={t1_count}"
        )

        t2_mask = data["tier2_mask_test"]
        t3_mask = data["tier3_mask_test"]

        # Strict subset property: any sample in Tier 3 MUST also be in Tier 2
        # (t3_mask & ~t2_mask) must be all False
        violations = np.sum(t3_mask & (~t2_mask))
        assert violations == 0, f"Tier 3 contains samples outside Tier 2 in Client {cid:02d} ({quantile})"
        assert int(np.sum(t2_mask)) == t2_count
        assert int(np.sum(t3_mask)) == t3_count


# ---------------------------------------------------------------------------
# Check 9: Single-Class Evaluation Metric Handling
# ---------------------------------------------------------------------------
def test_single_class_metric_handling():
    """Verify evaluate_binary_predictions safely returns None for ROC-AUC/PR-AUC on single-class subsets."""
    # All benign (class 0)
    y_all_zero = np.zeros(50, dtype=int)
    probs_all_zero = np.full(50, 0.1, dtype=float)
    res_zero = evaluate_binary_predictions(y_all_zero, probs_all_zero)

    assert res_zero["roc_auc"] is None
    assert res_zero["pr_auc"] is None
    assert res_zero["n_samples"] == 50
    assert res_zero["n_pos"] == 0
    assert res_zero["n_neg"] == 50
    assert res_zero["accuracy"] == 1.0

    # All attack (class 1)
    y_all_one = np.ones(50, dtype=int)
    probs_all_one = np.full(50, 0.9, dtype=float)
    res_one = evaluate_binary_predictions(y_all_one, probs_all_one)

    assert res_one["roc_auc"] is None
    assert res_one["pr_auc"] is None
    assert res_one["n_samples"] == 50
    assert res_one["n_pos"] == 50
    assert res_one["n_neg"] == 0
    assert res_one["accuracy"] == 1.0

    # Mixed classes: ROC-AUC and PR-AUC must be floats
    y_mixed = np.array([0, 0, 1, 1], dtype=int)
    probs_mixed = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    res_mixed = evaluate_binary_predictions(y_mixed, probs_mixed)

    assert isinstance(res_mixed["roc_auc"], float)
    assert isinstance(res_mixed["pr_auc"], float)
    assert res_mixed["roc_auc"] == 1.0


# ---------------------------------------------------------------------------
# Check 10: Training Reproducibility & Checkpoint Integrity
# ---------------------------------------------------------------------------
def test_training_reproducibility_and_checkpoint_integrity():
    """
    Verify reproducibility with seed 42, best validation loss recording,
    and checkpoint integrity across saved Phase 5.2 models.
    """
    # 1. Reproducibility test with synthetic data
    def run_dummy_train(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SupervisedDenseMLP(input_dim=38)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(16, 38)
        y = torch.randint(0, 2, (16,)).float()
        criterion = nn.BCEWithLogitsLoss()

        losses = []
        for _ in range(3):
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses, [p.clone().detach() for p in model.parameters()]

    losses1, params1 = run_dummy_train(42)
    losses2, params2 = run_dummy_train(42)

    assert np.allclose(losses1, losses2), "Loss trajectories differ under identical seed 42!"
    for p1, p2 in zip(params1, params2):
        assert torch.allclose(p1, p2), "Model parameters differ under identical seed 42!"

    # 2. Checkpoint and Metadata Consistency
    meta_path = Path("outputs/metrics/stage5/phase5.2_training_metadata.json")
    assert meta_path.exists(), "Missing phase5.2_training_metadata.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    assert len(meta_list) == 40, f"Expected 40 training records, found {len(meta_list)}"

    for record in meta_list:
        ckpt_path = Path(record["checkpoint_path"])
        assert ckpt_path.exists(), f"Saved checkpoint missing: {ckpt_path}"

        # Load weights and check architecture compatibility
        state = torch.load(ckpt_path, map_location="cpu")
        m_type = record["model_type"]
        if m_type == "GRU":
            m = HeavyGRU(input_dim=38, hidden_dim=64)
        else:
            m = SupervisedDenseMLP(input_dim=38)

        m.load_state_dict(state)

        # Check recorded outcome
        outcome = record["training_outcome"]
        assert outcome["best_epoch"] >= 1
        assert outcome["best_val_loss"] >= 0.0
        assert outcome["final_epoch"] >= outcome["best_epoch"]

    # 3. CSV results table consistency
    csv_path = Path("outputs/metrics/stage5/phase5.2_local_training_results.csv")
    assert csv_path.exists(), "Missing phase5.2_local_training_results.csv"

    import pandas as pd
    df = pd.read_csv(csv_path)
    # 10 clients * 2 quantiles * 2 models * 3 tiers = 120 rows
    assert len(df) == 120, f"Expected 120 evaluation rows, got {len(df)}"

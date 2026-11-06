"""
Phase 4.1 Initialization and Artifact Integrity Test Suite.

Verifies:
1. Stage 4 configuration validity and queue parameter definitions (Q_max = B, K = Q_max + 1).
2. Cryptographic immutability of Stage 2 data partitions (SHA-256 match).
3. Array shapes (38 features) across train, val, and test splits for all 10 clients.
4. Frozen Stage 3 model architecture parameter counts (AE: 2,008, Gate: 641) and checkpoint loading.
5. Frozen Stage 3 threshold JSON contents and CI bounds for all 10 clients.
6. Completeness of outputs/logs/stage4_artifact_hashes.json manifest.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import torch
import yaml

from src.models.autoencoder import Autoencoder
from src.models.gate import GatingRegressor


def test_stage4_config_validity():
    """Verify configs/stage4_config.yaml structure, invariants, and safeguard definitions."""
    cfg_path = Path("configs/stage4_config.yaml")
    assert cfg_path.exists(), "configs/stage4_config.yaml not found"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Required top-level sections
    required_sections = [
        "queue_parameters",
        "front_end_rate_limiter",
        "workload_estimator",
        "thresholds",
        "stochastic_arrival_processes",
        "proxy_benchmark",
        "security_accounting",
        "scenarios",
    ]
    for sec in required_sections:
        assert sec in cfg, f"Missing required configuration section: {sec}"

    # Queue capacity invariants
    q_params = cfg["queue_parameters"]
    q_max = q_params["buffer_capacity_Q_max"]
    assert q_max > 0, "buffer_capacity_Q_max must be positive"
    num_workers = q_params["num_workers"]
    assert num_workers == 1, "Phase 4 single-worker model expects num_workers == 1"
    system_capacity_k = q_max + num_workers
    assert system_capacity_k == q_max + 1, "System capacity K must equal Q_max + 1"

    # Front-end rate limiter parameters
    fe_limiter = cfg["front_end_rate_limiter"]
    assert 0.0 < fe_limiter["epsilon_cpu"] < 1.0, "epsilon_cpu must be in (0, 1)"
    assert fe_limiter["burst_tolerance_tokens"] > 0, "burst_tolerance_tokens must be positive"
    assert fe_limiter["continuous_replenishment"] is True
    assert 0.0 < fe_limiter["ewma_alpha"] <= 1.0, "ewma_alpha must be in (0, 1]"

    # Workload estimator: Active-Stall Residual Heuristic
    wl_est = cfg["workload_estimator"]
    assert wl_est["use_active_stall_heuristic"] is True
    assert wl_est["heuristic_name"] == "Active-Stall Residual Heuristic"

    # Security accounting: decoupling and deferral to Stage 5
    sec_acc = cfg["security_accounting"]
    assert sec_acc["track_effective_passage_recall"] is True
    assert sec_acc["track_ae_fallback_recall"] is True
    assert sec_acc["track_ideal_routing_yield"] is True
    assert sec_acc["defer_e2e_classification_to_stage5"] is True

    # Scenarios S1-S7 presence
    scenarios = cfg["scenarios"]
    for s_id in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        assert s_id in scenarios, f"Scenario {s_id} missing from configs/stage4_config.yaml"


def test_stage2_data_integrity_and_shapes():
    """Verify SHA-256 hashes and 38-feature tensor shapes for all 10 clients."""
    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    assert stage2_dir.exists(), "Stage 2 directory not found"

    expected_hashes = {
        "client_00": "47ce83a768f7383d",
        "client_01": "84db86d3df11e928",
        "client_02": "12ac19590a642487",
        "client_03": "6e048522d756877c",
        "client_04": "de85593a198aef58",
        "client_05": "7160207b5e299826",
        "client_06": "1a290dc2bc82d94b",
        "client_07": "43a6edf0191083d2",
        "client_08": "45a14143cd699212",
        "client_09": "75c866640def45db",
    }

    for c_idx in range(10):
        cid = f"client_{c_idx:02d}"
        c_path = stage2_dir / cid / "data.npz"
        assert c_path.exists(), f"data.npz missing for {cid}"

        # SHA-256 prefix check
        with open(c_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        assert h.startswith(expected_hashes[cid]), f"SHA-256 mismatch for {cid}: {h}"

        # Array shape checks
        data = np.load(c_path, allow_pickle=True)
        assert data["X_train"].ndim == 2 and data["X_train"].shape[1] == 38, f"{cid} X_train shape invalid"
        assert data["X_val"].ndim == 2 and data["X_val"].shape[1] == 38, f"{cid} X_val shape invalid"
        assert data["X_test"].ndim == 2 and data["X_test"].shape[1] == 38, f"{cid} X_test shape invalid"

        # Label shape checks
        assert len(data["y_train"]) == len(data["X_train"])
        assert len(data["y_val"]) == len(data["X_val"])
        assert len(data["y_test"]) == len(data["X_test"])


def test_stage3_model_checkpoint_shapes_and_parameters():
    """Verify frozen Stage 3 Autoencoders and Gate Regressors parameter counts and tensor shapes."""
    ckpt_dir = Path("outputs/checkpoints/stage3")
    assert ckpt_dir.exists(), "outputs/checkpoints/stage3 directory missing"

    for c_idx in range(10):
        cid = f"client_{c_idx:02d}"
        ae_path = ckpt_dir / f"{cid}_autoencoder.pt"
        gate_path = ckpt_dir / f"{cid}_gate.pt"

        assert ae_path.exists(), f"Autoencoder checkpoint missing for {cid}"
        assert gate_path.exists(), f"Gate checkpoint missing for {cid}"

        # Instantiate Autoencoder (38 -> 20 -> 10 -> 20 -> 38, 2,008 params)
        ae = Autoencoder(input_dim=38)
        ae_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
        assert ae_params == 2008, f"Expected 2,008 AE parameters, got {ae_params}"

        ae_ckpt = torch.load(ae_path, map_location="cpu")
        load_res_ae = ae.load_state_dict(ae_ckpt)
        assert len(load_res_ae.unexpected_keys) == 0 and len(load_res_ae.missing_keys) == 0

        # Instantiate Gating Regressor (38 -> 16 -> 1, 641 params)
        gate = GatingRegressor(input_dim=38)
        gate_params = sum(p.numel() for p in gate.parameters() if p.requires_grad)
        assert gate_params == 641, f"Expected 641 Gate parameters, got {gate_params}"

        gate_ckpt = torch.load(gate_path, map_location="cpu")
        load_res_gate = gate.load_state_dict(gate_ckpt)
        assert len(load_res_gate.unexpected_keys) == 0 and len(load_res_gate.missing_keys) == 0

        # Forward pass tensor shape validation
        b_size = 16
        dummy_x = torch.randn(b_size, 38)
        ae.eval()
        gate.eval()
        with torch.no_grad():
            x_recon = ae(dummy_x)
            assert x_recon.shape == (b_size, 38), f"AE output shape mismatch: {x_recon.shape}"
            e_hat = gate(dummy_x)
            assert e_hat.shape == (b_size, 1), f"Gate output shape mismatch: {e_hat.shape}"


def test_stage3_threshold_contents_and_parity():
    """Verify frozen Stage 3 Q99 threshold JSON files and finite bounds for all 10 clients."""
    ckpt_dir = Path("outputs/checkpoints/stage3")

    for c_idx in range(10):
        cid = f"client_{c_idx:02d}"
        th_path = ckpt_dir / f"{cid}_thresholds.json"
        assert th_path.exists(), f"Threshold JSON missing for {cid}"

        with open(th_path, "r", encoding="utf-8") as f:
            th_data = json.load(f)

        assert "tau_ae" in th_data and "tau_gate" in th_data
        tau_ae = th_data["tau_ae"]
        tau_gate = th_data["tau_gate"]

        assert np.isfinite(tau_ae) and tau_ae > 0.0, f"{cid} tau_ae invalid: {tau_ae}"
        assert np.isfinite(tau_gate), f"{cid} tau_gate invalid: {tau_gate}"

        # Confidence interval sanity
        ci_ae = th_data.get("tau_ae_95ci")
        if ci_ae is not None:
            assert ci_ae[0] <= tau_ae <= ci_ae[1], f"{cid} tau_ae outside 95% CI"

        ci_gate = th_data.get("tau_gate_95ci")
        if ci_gate is not None:
            assert ci_gate[0] <= tau_gate <= ci_gate[1], f"{cid} tau_gate outside 95% CI"


def test_stage4_artifact_hash_manifest():
    """Verify existence and completeness of outputs/logs/stage4_artifact_hashes.json."""
    manifest_path = Path("outputs/logs/stage4_artifact_hashes.json")
    assert manifest_path.exists(), "stage4_artifact_hashes.json missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "stage2_data" in manifest
    assert "stage3_models" in manifest
    assert "stage3_thresholds" in manifest

    assert len(manifest["stage2_data"]) == 10
    assert len(manifest["stage3_models"]) == 10
    assert len(manifest["stage3_thresholds"]) == 10

    for c_idx in range(10):
        cid = f"client_{c_idx:02d}"
        assert cid in manifest["stage2_data"]
        assert manifest["stage2_data"][cid]["baseline_verified"] is True
        assert cid in manifest["stage3_models"]
        assert "autoencoder.pt" in manifest["stage3_models"][cid]
        assert "gate.pt" in manifest["stage3_models"][cid]
        assert cid in manifest["stage3_thresholds"]


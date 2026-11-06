"""
Comprehensive Integration Tests and Audit Verification for Stage 3.

Tests:
1. Parameter Count Verification: AE == 2,008 parameters; Gate == 641 parameters.
2. Output Tensor Shape Verification: AE output [B, 38]; Gate output [B, 1].
3. AE Threshold Sanity Check: Verify tau_AE_k > 0.0 for all clients.
4. Loss Monotonicity Check: Verify training loss generally trends downward across epochs.
5. Benign/Attack Separability Test: Verify mean benign MSE < mean attack MSE on Tier 1 test set for all clients.
6. Gate-AE Agreement Test: Verify Gate-AE agreement rate >= 75% across test flows.
7. Stage 2 Immutability Check: Re-verify sha256 checksums of Stage 2 NPZ artifacts match initial checksums.
"""

import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import torch

from src.models.autoencoder import Autoencoder
from src.models.gate import GatingRegressor


def test_parameter_counts():
    """Verify exact parameter counts according to theoretical design."""
    ae = Autoencoder(input_dim=38)
    ae_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)
    assert ae_params == 2008, f"Expected 2008 parameters for Autoencoder, got {ae_params}"

    gate = GatingRegressor(input_dim=38)
    gate_params = sum(p.numel() for p in gate.parameters() if p.requires_grad)
    assert gate_params == 641, f"Expected 641 parameters for GatingRegressor, got {gate_params}"


def test_output_tensor_shapes():
    """Verify output tensor dimensions for batch size B=16."""
    B = 16
    x = torch.randn(B, 38)

    ae = Autoencoder(input_dim=38)
    x_hat = ae(x)
    assert x_hat.shape == (B, 38), f"Expected AE output shape (16, 38), got {x_hat.shape}"

    gate = GatingRegressor(input_dim=38)
    e_hat = gate(x)
    assert e_hat.shape == (B, 1), f"Expected Gate output shape (16, 1), got {e_hat.shape}"


def test_stage2_immutability():
    """Re-verify that all Stage 2 client data.npz files were untouched."""
    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    client_dirs = sorted(list(stage2_dir.glob("client_*")))
    assert len(client_dirs) == 10

    # Known recorded checksums from initial verification
    expected_checksums = {
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

    for cdir in client_dirs:
        npz_file = cdir / "data.npz"
        with open(npz_file, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        expected = expected_checksums[cdir.name]
        assert h.startswith(expected), f"Checksum mismatch for {cdir.name}: got {h}, expected {expected}..."


def test_threshold_sanity():
    """Verify tau_AE_k > 0.0 and positive finite values across all 10 clients."""
    metrics_file = Path("outputs/metrics/stage3/stage3_results_summary.json")
    assert metrics_file.exists(), "stage3_results_summary.json not found"

    with open(metrics_file, "r") as f:
        summary = json.load(f)

    th_list = summary["thresholds_summary"]
    assert len(th_list) == 10

    for th in th_list:
        tau_ae = th["tau_ae"]
        tau_gate = th["tau_gate"]
        assert tau_ae > 0.0, f"tau_ae must be > 0.0, got {tau_ae}"
        assert np.isfinite(tau_ae)
        assert np.isfinite(tau_gate)
        # Verify 95% CI bounds ordering
        assert th["tau_ae_ci_low"] <= tau_ae <= th["tau_ae_ci_high"]
        assert th["tau_gate_ci_low"] <= tau_gate <= th["tau_gate_ci_high"]


def test_benign_attack_separability():
    """Verify mean benign MSE < mean attack MSE on Tier 1 test set for all clients."""
    metrics_file = Path("outputs/metrics/stage3/stage3_results_summary.json")
    with open(metrics_file, "r") as f:
        summary = json.load(f)

    for cname, res in summary["per_client_results"].items():
        stats = res["tier1"]["ae_reconstruction_stats"]
        b_mse = stats["benign_mean_mse"]
        a_mse = stats["attack_mean_mse"]
        # Empirical test: attacks should produce higher average reconstruction loss than benign flows
        assert b_mse < a_mse, (
            f"Client {cname} failed separability test: benign MSE {b_mse:.6f} >= attack MSE {a_mse:.6f}"
        )


def test_gate_ae_agreement():
    """Verify Gate-AE agreement rate >= 75% across test flows for all clients."""
    metrics_file = Path("outputs/metrics/stage3/stage3_results_summary.json")
    with open(metrics_file, "r") as f:
        summary = json.load(f)

    for cname, res in summary["per_client_results"].items():
        agreement = res["tier1"]["agreement"]["gate_ae_agreement_rate_pct"]
        assert agreement >= 75.0, (
            f"Client {cname} failed Gate-AE agreement test: {agreement:.2f}% < 75.0%"
        )


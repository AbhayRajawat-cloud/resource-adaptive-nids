"""
Automated Validation Unit Test Suite for Stage 4 Report & Phase 4.4 Deliverables.

Validates:
1. Scoped Cryptographic Manifest Check: SHA-256 verification of files in outputs/logs/stage4_artifact_hashes.json
   (Stage 2 data.npz partitions, Stage 3 .pt checkpoints, and _thresholds.json).
2. Generated Artifact Validation:
   - Existence and non-zero size of outputs/reports/stage4_final_evidence_report.md
   - Existence, non-zero size, and schema of outputs/metrics/stage4/admitted_run_lengths.json
   - Existence, non-zero size, and schema of outputs/metrics/stage4/stage4_aggregated_metrics.json
   - Existence and non-zero size of all 4 publication PNG plots (outputs/plots/stage4/*.png)
3. Mathematical Conservation Laws:
   - Flow conservation holds across all aggregated scenarios:
       heavy_admitted_pct + gate_reject_ae_pct + load_shed_ae_pct + frontend_drop_pct == 100.0% (+/- 0.01%)
   - Effective passage recall and ideal routing yield positivity and upper-bounding (0 <= yield <= 1.0).
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def compute_sha256(filepath: Path) -> str:
    """Compute hex-encoded SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_scoped_manifest_cryptographic_check():
    """Verify SHA-256 hashes strictly for files recorded in outputs/logs/stage4_artifact_hashes.json."""
    manifest_path = Path("outputs/logs/stage4_artifact_hashes.json")
    assert manifest_path.exists(), "Missing manifest: outputs/logs/stage4_artifact_hashes.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Stage 2 data.npz files
    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    for cid, file_hashes in manifest["stage2_data"].items():
        npz_file = stage2_dir / cid / "data.npz"
        assert npz_file.exists(), f"Missing Stage 2 file: {npz_file}"
        curr_hash = compute_sha256(npz_file)
        assert curr_hash == file_hashes["data.npz"], f"Hash mismatch for {npz_file}"

    # 2. Stage 3 model checkpoints
    ckpt_dir = Path("outputs/checkpoints/stage3")
    for cid, file_hashes in manifest["stage3_models"].items():
        ae_file = ckpt_dir / f"{cid}_autoencoder.pt"
        gate_file = ckpt_dir / f"{cid}_gate.pt"
        assert ae_file.exists(), f"Missing checkpoint: {ae_file}"
        assert gate_file.exists(), f"Missing checkpoint: {gate_file}"
        assert compute_sha256(ae_file) == file_hashes["autoencoder.pt"], f"Hash mismatch for {ae_file}"
        assert compute_sha256(gate_file) == file_hashes["gate.pt"], f"Hash mismatch for {gate_file}"

    # 3. Stage 3 threshold files
    for cid, file_hashes in manifest["stage3_thresholds"].items():
        thresh_file = ckpt_dir / f"{cid}_thresholds.json"
        assert thresh_file.exists(), f"Missing threshold file: {thresh_file}"
        expected = file_hashes["thresholds.json"] if isinstance(file_hashes, dict) else file_hashes
        assert compute_sha256(thresh_file) == expected, f"Hash mismatch for {thresh_file}"


def test_generated_artifact_validation():
    """Verify existence, non-emptiness, and schema validity for all generated Phase 4.4 artifacts."""
    # 1. Evidence Report
    report_path = Path("outputs/reports/stage4_final_evidence_report.md")
    assert report_path.exists(), f"Report file missing: {report_path}"
    assert report_path.stat().st_size > 1000, "Report file is unexpectedly small or empty"

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()
    assert "STAGE 4 FINAL EVIDENCE & AUDIT REPORT" in report_text
    assert "Mathematical Queueing Invariant Audit" in report_text
    assert "Analytical Queue Stability Boundaries" in report_text
    assert "Client Heterogeneity Deep-Dive" in report_text

    # 2. Admitted Run Lengths JSON
    run_len_path = Path("outputs/metrics/stage4/admitted_run_lengths.json")
    assert run_len_path.exists(), f"Missing file: {run_len_path}"
    assert run_len_path.stat().st_size > 200, "Admitted run lengths JSON is empty"

    with open(run_len_path, "r", encoding="utf-8") as f:
        run_len_data = json.load(f)
    assert "pooled_summary" in run_len_data
    assert "per_client" in run_len_data
    assert "Q99" in run_len_data["pooled_summary"]
    assert "Q95" in run_len_data["pooled_summary"]
    assert "synthetic routing statistics" in run_len_data["metadata"]["methodological_caveat"].lower()

    # 3. Aggregated Metrics JSON & CSV
    agg_json = Path("outputs/metrics/stage4/stage4_aggregated_metrics.json")
    agg_csv = Path("outputs/metrics/stage4/stage4_aggregated_summary.csv")
    assert agg_json.exists(), f"Missing file: {agg_json}"
    assert agg_csv.exists(), f"Missing file: {agg_csv}"

    with open(agg_json, "r", encoding="utf-8") as f:
        agg_data = json.load(f)
    assert "single_server_stability_analysis" in agg_data
    assert "scenarios" in agg_data
    assert "heterogeneity_deep_dive" in agg_data

    # 4. Publication PNG Plots
    plot_dir = Path("outputs/plots/stage4")
    expected_plots = [
        "S2_load_sweep_delay_and_shedding.png",
        "S5_sla_delay_sweep.png",
        "S6_Q99_vs_Q95_crossover.png",
        "S7_client_heterogeneity_queue.png",
    ]
    for p_name in expected_plots:
        p_path = plot_dir / p_name
        assert p_path.exists(), f"Missing plot: {p_path}"
        assert p_path.stat().st_size > 50000, f"Plot {p_path} is suspiciously small ({p_path.stat().st_size} bytes)"


def test_mathematical_identity_and_conservation_laws():
    """Verify routing flow conservation identity and security probability bounds on aggregated outputs."""
    csv_path = Path("outputs/metrics/stage4/stage4_scenario_summary.csv")
    df = pd.read_csv(csv_path)

    # 1. Flow conservation across all 332 runs
    assert df["conservation_verified"].all(), "Flow conservation identity failed on some runs"

    # Percentage sum identity
    pct_sum = (
        df["frontend_drop_pct"] +
        df["gate_reject_ae_pct"] +
        df["load_shed_ae_pct"] +
        df["heavy_admitted_pct"]
    )
    np.testing.assert_allclose(pct_sum.to_numpy(), 100.0, atol=1e-3, err_msg="Percentage sum != 100%")

    # 2. Probability bounds
    assert (df["effective_heavy_passage_recall"] >= 0.0).all()
    assert (df["effective_heavy_passage_recall"] <= 1.0).all()
    assert (df["ideal_routing_yield"] >= 0.0).all()
    assert (df["ideal_routing_yield"] <= 1.0).all()
    assert (df["ideal_routing_yield"] >= df["effective_heavy_passage_recall"]).all(), (
        "Ideal yield must be >= effective heavy passage recall (monotone coupling)"
    )

    # 3. Stability boundary bounds
    agg_json = Path("outputs/metrics/stage4/stage4_aggregated_metrics.json")
    with open(agg_json, "r", encoding="utf-8") as f:
        agg = json.load(f)
    stab = agg["single_server_stability_analysis"]["critical_stability_limits"]
    assert stab["Q95_lambda_crit_tilde"] < stab["Q99_lambda_crit_tilde"], (
        "Q95 stability limit must be strictly tighter than Q99 stability limit"
    )
    assert 10.0 <= stab["Q95_lambda_crit_tilde"] <= 25.0
    assert 50.0 <= stab["Q99_lambda_crit_tilde"] <= 100.0

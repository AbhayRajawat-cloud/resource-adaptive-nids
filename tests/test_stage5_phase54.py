"""
Stage 5: Phase 5.4 Automated Verification Test Suite.

Validates:
1. Flow Conservation Exactness Test:
   Verifies N_total == N_frontend_drop + N_gate_reject_ae + N_load_shed_ae + N_heavy_admitted
   across all rows in phase5.4_e2e_results.csv.
2. Combined Recall Identity Test:
   Verifies Recall_e2e == TP / Total_Attacks and FPR_e2e == FP / Total_Benign.
3. Model Output Persistence Check:
   Verifies existence, non-emptiness, and schema validity of all Phase 5.4 deliverables:
   - CSV: outputs/metrics/stage5/phase5.4_e2e_results.csv
   - JSON: outputs/metrics/stage5/phase5.4_e2e_results.json
   - PNG plots: outputs/plots/stage5/e2e_recall_vs_load.png, e2e_routing_breakdown.png, config_comparison_bar.png
   - Report: outputs/reports/stage5_final_evidence_report.md
4. Scoped Cryptographic Baseline Audit:
   Re-verifies SHA-256 hashes for all 40 baseline artifacts in stage4_artifact_hashes.json.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def compute_sha256(filepath: Path) -> str:
    """Compute hex-encoded SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Check 1: Flow Conservation Exactness Test
# ---------------------------------------------------------------------------
def test_flow_conservation_exactness():
    """Verify that every evaluated row strictly satisfies N_total == sum(4 outcomes)."""
    csv_path = Path("outputs/metrics/stage5/phase5.4_e2e_results.csv")
    assert csv_path.exists(), f"Missing {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) > 0, "Empty E2E results CSV"

    for idx, row in df.iterrows():
        n_tot = row["n_total"]
        sum_routing = (
            row["n_frontend_drop"]
            + row["n_gate_reject_ae"]
            + row["n_load_shed_ae"]
            + row["n_heavy_admitted"]
        )
        assert n_tot == sum_routing, (
            f"Flow conservation violated at row {idx} ({row['scenario_id']}, {row['config_short']}, {row['client_str']}): "
            f"N_total={n_tot} != sum_routing={sum_routing}"
        )


# ---------------------------------------------------------------------------
# Check 2: Combined Security Identity Test
# ---------------------------------------------------------------------------
def test_combined_security_identity():
    """Verify that Recall_e2e and FPR_e2e exactly match true positive / false positive identities."""
    csv_path = Path("outputs/metrics/stage5/phase5.4_e2e_results.csv")
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        n_attacks = row["n_attacks"]
        n_benign = row["n_benign"]
        tp = row["tp"]
        fp = row["fp"]
        tn = row["tn"]
        fn = row["fn"]

        # Confusion matrix sums
        assert tp + fn == n_attacks, f"Row {idx}: TP + FN != Attacks"
        assert fp + tn == n_benign, f"Row {idx}: FP + TN != Benign"
        assert n_attacks + n_benign == row["n_total"], f"Row {idx}: Attacks + Benign != Total"

        # Recall identity
        expected_recall = (tp / n_attacks) if n_attacks > 0 else 0.0
        assert pytest.approx(row["recall_e2e"], 1e-5) == expected_recall, f"Row {idx}: Recall identity failed"

        # FPR identity
        expected_fpr = (fp / n_benign) if n_benign > 0 else 0.0
        assert pytest.approx(row["fpr_e2e"], 1e-5) == expected_fpr, f"Row {idx}: FPR identity failed"


# ---------------------------------------------------------------------------
# Check 3: Deliverables Persistence and Schema Check
# ---------------------------------------------------------------------------
def test_deliverables_persistence_and_schema():
    """Verify that all generated CSV, JSON, PNG, and MD report deliverables exist and are valid."""
    metrics_dir = Path("outputs/metrics/stage5")
    plots_dir = Path("outputs/plots/stage5")
    reports_dir = Path("outputs/reports")

    csv_file = metrics_dir / "phase5.4_e2e_results.csv"
    json_file = metrics_dir / "phase5.4_e2e_results.json"
    plot_1 = plots_dir / "e2e_recall_vs_load.png"
    plot_2 = plots_dir / "e2e_routing_breakdown.png"
    plot_3 = plots_dir / "config_comparison_bar.png"
    report_file = reports_dir / "stage5_final_evidence_report.md"

    for f in [csv_file, json_file, plot_1, plot_2, plot_3, report_file]:
        assert f.exists(), f"Deliverable missing: {f}"
        assert f.stat().st_size > 0, f"Deliverable is empty: {f}"

    # Verify JSON structure
    with open(json_file, "r", encoding="utf-8") as f:
        d = json.load(f)
    assert "metadata" in d
    assert "statistical_comparison_config_a_vs_config_b" in d
    assert "results_summary" in d
    assert len(d["statistical_comparison_config_a_vs_config_b"]) == 3  # Tiers 1, 2, 3


# ---------------------------------------------------------------------------
# Check 4: Scoped Baseline SHA-256 Cryptographic Audit
# ---------------------------------------------------------------------------
def test_scoped_baseline_cryptographic_audit():
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


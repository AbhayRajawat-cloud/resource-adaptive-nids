"""
Stage 5: Final System Validation, Code Audit & Threat Taxonomy Test Suite.

Validates:
1. CSV Existence & Schema Validity:
   - outputs/metrics/stage5/validation_attack_family_breakdown.csv
   - outputs/metrics/stage5/validation_client_sample_distribution.csv
   - outputs/metrics/stage5/validation_dispersion_metrics.csv
2. Audit Report Completeness:
   - outputs/reports/stage5_system_validation_report.md exists and contains all required section headings.
3. Scoped Cryptographic Baseline Audit:
   - Re-verifies SHA-256 digests for all 40 baseline artifacts explicitly listed in stage4_artifact_hashes.json.
4. Deliverable Plot Persistence:
   - outputs/plots/stage5/federated_convergence_curves.png
"""

import hashlib
import json
from pathlib import Path

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
# Test 1: CSV Deliverables Existence and Schema Validity
# ---------------------------------------------------------------------------
def test_validation_csv_existence_and_schema():
    """Verify that all 3 generated validation CSVs exist, are non-empty, and adhere to schema."""
    metrics_dir = Path("outputs/metrics/stage5")

    # 1. Attack Family Breakdown
    af_csv = metrics_dir / "validation_attack_family_breakdown.csv"
    assert af_csv.exists(), f"Missing {af_csv}"
    df_af = pd.read_csv(af_csv)
    assert len(df_af) > 0, "Empty attack family breakdown CSV"
    expected_af_cols = [
        "client_id", "client_str", "config_short", "config_full", "quantile",
        "attack_family", "is_attack", "status", "n_total", "n_fe_drop", "n_gate_rej",
        "n_load_shed", "n_heavy_adm", "gate_passage_rate", "heavy_admission_rate",
        "tp_ae", "n_processed_ae", "conditional_ae_recall", "tp_heavy",
        "n_admitted_heavy", "conditional_heavy_recall", "e2e_ae_contribution",
        "e2e_heavy_contribution", "recall_e2e"
    ]
    for col in expected_af_cols:
        assert col in df_af.columns, f"Missing column {col} in {af_csv}"

    # Verify Config B Q95 is marked NOT AVAILABLE
    b_q95 = df_af[(df_af["config_short"] == "Config B") & (df_af["quantile"] == "Q95")]
    assert len(b_q95) > 0, "Missing Config B Q95 entries"
    assert (b_q95["status"] == "NOT AVAILABLE").all(), "Config B Q95 must be marked NOT AVAILABLE"

    # 2. Client Sample Distribution
    sd_csv = metrics_dir / "validation_client_sample_distribution.csv"
    assert sd_csv.exists(), f"Missing {sd_csv}"
    df_sd = pd.read_csv(sd_csv)
    assert len(df_sd) == 20, f"Expected exactly 20 client-quantile rows, got {len(df_sd)}"
    expected_sd_cols = [
        "client_id", "client_str", "quantile",
        "train_raw_benign", "train_raw_attack", "train_raw_total",
        "train_pass_benign", "train_pass_attack", "train_pass_total",
        "train_passage_rate_pct", "train_attack_concentration_pct",
        "val_raw_benign", "val_raw_attack", "val_raw_total",
        "val_pass_benign", "val_pass_attack", "val_pass_total",
        "val_passage_rate_pct", "val_attack_concentration_pct",
        "test_raw_benign", "test_raw_attack", "test_raw_total",
        "test_pass_benign", "test_pass_attack", "test_pass_total",
        "test_passage_rate_pct", "test_attack_concentration_pct",
        "w_pos_training", "w_pos_fallback_used"
    ]
    for col in expected_sd_cols:
        assert col in df_sd.columns, f"Missing column {col} in {sd_csv}"

    # 3. Dispersion Metrics
    disp_csv = metrics_dir / "validation_dispersion_metrics.csv"
    assert disp_csv.exists(), f"Missing {disp_csv}"
    df_disp = pd.read_csv(disp_csv)
    assert len(df_disp) > 0, "Empty dispersion metrics CSV"
    expected_disp_cols = [
        "benchmark_scope", "config_short", "config_full", "quantile", "tier",
        "tier_label", "num_clients", "macro_f1_mean", "macro_f1_std", "macro_f1_median",
        "macro_f1_iqr", "macro_f1_min", "macro_f1_max", "macro_f1_range",
        "macro_prec_mean", "macro_prec_std", "macro_prec_median", "macro_prec_iqr",
        "macro_rec_mean", "macro_rec_std", "macro_rec_median", "macro_rec_iqr",
        "pooled_status", "tp_pooled", "fp_pooled", "fn_pooled", "tn_pooled",
        "pooled_precision", "pooled_recall", "pooled_f1", "pooled_accuracy"
    ]
    for col in expected_disp_cols:
        assert col in df_disp.columns, f"Missing column {col} in {disp_csv}"


# ---------------------------------------------------------------------------
# Test 2: Audit Report Completeness
# ---------------------------------------------------------------------------
def test_audit_report_completeness():
    """Verify that stage5_system_validation_report.md exists and contains all required sections."""
    report_path = Path("outputs/reports/stage5_system_validation_report.md")
    assert report_path.exists(), f"Missing report deliverable: {report_path}"
    assert report_path.stat().st_size > 2000, "Validation report is unexpectedly small"

    content = report_path.read_text(encoding="utf-8")
    required_sections = [
        "1. Stage 4/5 Queue & Service Rate Code Audit",
        "2. NF-UNSW-NB15-v2 Threat Taxonomy Profiling & Denominator Accounting",
        "3. Split-Wise Per-Client Sample-Size & Class Imbalance Audit",
        "4. Client-Level Performance Dispersion & Sample-Weighted Pooled Analysis",
        "5. Federated Training Convergence Progression",
        "6. Scoped Cryptographic Baseline Audit & File Categorization",
        "7. Final Systems Engineering Synthesis & Sign-off",
    ]
    for sec in required_sections:
        assert sec in content, f"Missing section heading in report: '{sec}'"


# ---------------------------------------------------------------------------
# Test 3: Scoped Cryptographic Baseline Audit
# ---------------------------------------------------------------------------
def test_scoped_cryptographic_baseline_audit():
    """Verify SHA-256 digests for all files explicitly listed in stage4_artifact_hashes.json."""
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
# Test 4: Convergence Plot Deliverable Persistence
# ---------------------------------------------------------------------------
def test_convergence_plot_persistence():
    """Verify existence and size of federated convergence curves PNG."""
    plot_file = Path("outputs/plots/stage5/federated_convergence_curves.png")
    assert plot_file.exists(), f"Missing convergence plot: {plot_file}"
    assert plot_file.stat().st_size > 50000, f"Plot file unexpectedly small: {plot_file.stat().st_size} bytes"


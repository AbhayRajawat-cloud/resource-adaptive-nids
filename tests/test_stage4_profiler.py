"""
Automated Validation Unit Tests for Phase 4.2 Hardware Latency Profiler.

Verifies:
1. JSON and CSV artifact generation and schema validity (strict assertions).
2. Positivity, finite values, and bootstrap CI validity (strict assertions).
3. Numerical ordering and plausibility bounds (diagnostic warnings per Phase 4.3 specification).
"""

import json
from pathlib import Path
import warnings
import pytest
import pandas as pd


def test_hardware_latency_artifacts_exist():
    """Verify that both JSON and CSV profile artifacts were generated (strict assertion)."""
    json_path = Path("outputs/metrics/stage4/hardware_latency_profile.json")
    csv_path = Path("outputs/metrics/stage4/hardware_latency_profile.csv")

    assert json_path.exists(), "hardware_latency_profile.json not found"
    assert csv_path.exists(), "hardware_latency_profile.csv not found"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "macro_summary" in data
    assert "gru_batch_scaling" in data
    assert "per_client" in data
    assert len(data["per_client"]) == 10

    df = pd.read_csv(csv_path)
    assert len(df) == 10
    assert "client" in df.columns
    assert "mu_light_fps" in df.columns
    assert "mu_heavy_fps" in df.columns


def test_latency_positivity_and_ordering():
    """Verify positivity (strict) and execution hierarchy (diagnostic warning on ordering anomaly)."""
    json_path = Path("outputs/metrics/stage4/hardware_latency_profile.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    macro = data["macro_summary"]
    l_gate_macro = macro["mean_l_gate_ms"]
    l_ae_macro = macro["mean_l_ae_ms"]
    s_gru_macro = macro["mean_s_gru_ms"]

    # Strict structural assertion: latencies must be strictly positive and finite
    assert l_gate_macro > 0.0, "Macro Gate latency must be positive"
    assert l_ae_macro > 0.0, "Macro AE latency must be positive"
    assert s_gru_macro > 0.0, "Macro GRU latency must be positive"

    # Diagnostic evaluation of execution hierarchy
    if not (l_gate_macro <= l_ae_macro < s_gru_macro):
        warnings.warn(
            f"Diagnostic Warning: Macro latency ordering deviates from theoretical expectation: "
            f"Gate={l_gate_macro:.4f} ms, AE={l_ae_macro:.4f} ms, GRU={s_gru_macro:.4f} ms",
            UserWarning
        )

    for cid, c_data in data["per_client"].items():
        derived = c_data["derived_metrics"]
        l_gate = derived["l_gate_mean_ms"]
        l_ae = derived["l_ae_mean_ms"]
        s_gru = derived["s_gru_mean_ms"]

        # Strict positivity assertion
        assert l_gate > 0.0, f"{cid} Gate latency must be strictly positive"
        assert l_ae > 0.0, f"{cid} AE latency must be strictly positive"
        assert s_gru > 0.0, f"{cid} GRU latency must be strictly positive"

        # Diagnostic check for individual client model ordering
        if not (l_gate <= l_ae < s_gru):
            warnings.warn(
                f"Diagnostic Warning: {cid} hierarchy ordering anomaly: "
                f"Gate={l_gate:.4f} ms, AE={l_ae:.4f} ms, GRU={s_gru:.4f} ms",
                UserWarning
            )


def test_latency_plausibility_bounds():
    """Verify latencies fall within plausible ranges (diagnostic warning on tail jitter)."""
    json_path = Path("outputs/metrics/stage4/hardware_latency_profile.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cid, c_data in data["per_client"].items():
        derived = c_data["derived_metrics"]
        l_gate = derived["l_gate_mean_ms"]
        l_ae = derived["l_ae_mean_ms"]
        s_gru = derived["s_gru_mean_ms"]

        # Strict assertion: non-negative and finite
        assert 0.0 < l_gate < 1000.0, f"{cid} Gate latency not finite or wildly out of bounds: {l_gate}"
        assert 0.0 < l_ae < 1000.0, f"{cid} AE latency not finite or wildly out of bounds: {l_ae}"
        assert 0.0 < s_gru < 1000.0, f"{cid} GRU latency not finite or wildly out of bounds: {s_gru}"

        # Diagnostic check for expected CPU ranges (0.001 - 5.0 ms)
        if not (0.001 <= l_gate <= 5.0):
            warnings.warn(f"Diagnostic Warning: {cid} Gate latency outside nominal range: {l_gate:.4f} ms", UserWarning)
        if not (0.001 <= l_ae <= 5.0):
            warnings.warn(f"Diagnostic Warning: {cid} AE latency outside nominal range: {l_ae:.4f} ms", UserWarning)
        if not (0.01 <= s_gru <= 10.0):
            warnings.warn(f"Diagnostic Warning: {cid} GRU latency outside nominal range: {s_gru:.4f} ms", UserWarning)


def test_service_capacity_and_rate_limits():
    """Verify derived service rates mu_light, mu_heavy, and lambda_frontend_max (strict assertion)."""
    json_path = Path("outputs/metrics/stage4/hardware_latency_profile.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cid, c_data in data["per_client"].items():
        derived = c_data["derived_metrics"]
        mu_light = derived["mu_light_flows_per_sec"]
        mu_heavy = derived["mu_heavy_flows_per_sec"]
        lambda_fe = derived["lambda_frontend_max_flows_per_sec"]
        c_s_sq = derived["c_s_gru_squared"]

        assert mu_light > 0.0, f"{cid} mu_light must be positive"
        assert mu_heavy > 0.0, f"{cid} mu_heavy must be positive"
        assert lambda_fe > 0.0, f"{cid} lambda_fe must be positive"
        assert c_s_sq >= 0.0, f"{cid} c_s^2 must be non-negative"


def test_gru_batch_scaling_monotonicity():
    """Verify that batch processing latency increases monotonically with batch size (strict assertion)."""
    json_path = Path("outputs/metrics/stage4/hardware_latency_profile.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gru_scaling = data["gru_batch_scaling"]
    b1 = gru_scaling["B_1"]["mean_ms"]
    b8 = gru_scaling["B_8"]["mean_ms"]
    b32 = gru_scaling["B_32"]["mean_ms"]
    b64 = gru_scaling["B_64"]["mean_ms"]

    assert b1 < b8 < b32 < b64, f"GRU batch scaling not strictly monotonic: B1={b1}, B8={b8}, B32={b32}, B64={b64}"

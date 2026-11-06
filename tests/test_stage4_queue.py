"""
Stage 4 Phase 4.3 Validation Suite: Mathematical Invariant Unit Tests V1–V6.

Tests:
- V1 (Little's Law Validation): Bar{N}_q = lambda_adm * Bar{W}_q over steady-state window.
- V2 (M/D/1 Closed-Form Benchmark): Bar{W}_q = (rho * S_bar) / (2 * (1 - rho)).
- V3 (M/M/1/K Blocking Benchmark): P_K = (1 - rho)*rho^K / (1 - rho^{K+1}) using K = Q_max + 1.
- V4 (Flow Conservation Identity): N_tot == N_frontend_drop + N_gate_reject + N_load_shed + N_heavy_admitted.
- V5 (Cryptographic Immutability Audit): SHA-256 baseline verification of Stage 1–3 artifacts.
- V6 (SI Unit Consistency Audit): Verify time [s], [ms], rates [flows/s], and dimensionless ratios.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from src.simulation.queue_simulator import WorkloadQueueSimulator, FlowArrival
from src.simulation.traffic_generator import PoissonTrafficGenerator, ClientTrafficTraceSampler


def test_v1_littles_law_steady_state():
    """
    V1: Validate Little's Law Bar{N}_q = lambda_adm * Bar{W}_q on steady-state queueing.
    
    Admitted flows with Poisson arrivals at rate lambda_adm = 2000 flows/s,
    mean service time S_bar = 0.0001799 s (rho = 0.3598 < 1.0).
    """
    s_bar = 0.0001799  # ~0.18 ms
    lambda_arr = 2000.0  # flows/s -> rho ~ 0.36
    n_flows = 20000

    # Queue simulator with permissive thresholds so all arrivals trigger heavy path
    sim = WorkloadQueueSimulator(
        q_max=100,
        w_shed=1.0,          # 1 second (no load shedding)
        s_gru_bar=s_bar,
        token_bucket_rate=100000.0, # no frontend drops
        tau_gate=-100.0,     # always triggers
        deterministic_service=True,
    )

    gen = PoissonTrafficGenerator(rate=lambda_arr, seed=123)
    timestamps = gen.generate_timestamps(n_flows)

    for i in range(n_flows):
        flow = FlowArrival(
            flow_id=i,
            arrival_time=float(timestamps[i]),
            x=np.zeros(38),
            y_true=0,
            attack_family="Benign",
            service_time=s_bar,
        )
        sim.process_arrival(flow, gate_pred=0.0)

    # Discard warmup transient (first 20% of flows)
    warmup_cutoff = int(n_flows * 0.2)
    steady_waits = sim.waiting_times_admitted[warmup_cutoff:]
    steady_queue_lengths = sim.queue_lengths_at_arrival[warmup_cutoff:]

    w_q_mean = float(np.mean(steady_waits))
    n_q_mean = float(np.mean(steady_queue_lengths))

    # Little's Law predicted queue length: N_q = lambda * W_q
    predicted_n_q = lambda_arr * w_q_mean

    # Validate within 15% relative error (standard stochastic tolerance for 16,000 steady-state flows)
    rel_error = abs(n_q_mean - predicted_n_q) / max(n_q_mean, 0.05)
    assert rel_error < 0.15, f"Little's Law violated: simulated N_q={n_q_mean:.4f}, predicted={predicted_n_q:.4f}, rel_err={rel_error:.4f}"


def test_v2_md1_closed_form_benchmark():
    """
    V2: Validate M/D/1 Pollaczek-Khinchine formula Bar{W}_q = (rho * S_bar) / (2 * (1 - rho)).
    """
    s_bar = 0.0001799
    # Choose lambda to achieve rho = 0.50
    rho_target = 0.50
    lambda_arr = rho_target / s_bar  # ~2779.3 flows/s
    n_flows = 30000

    # Theoretical M/D/1 waiting time
    w_q_theoretical = (rho_target * s_bar) / (2.0 * (1.0 - rho_target))

    sim = WorkloadQueueSimulator(
        q_max=500,           # large buffer to approximate infinite M/D/1
        w_shed=10.0,
        s_gru_bar=s_bar,
        token_bucket_rate=100000.0,
        tau_gate=-100.0,
        deterministic_service=True,
    )

    gen = PoissonTrafficGenerator(rate=lambda_arr, seed=42)
    timestamps = gen.generate_timestamps(n_flows)

    for i in range(n_flows):
        flow = FlowArrival(
            flow_id=i,
            arrival_time=float(timestamps[i]),
            x=np.zeros(38),
            y_true=0,
            attack_family="Benign",
            service_time=s_bar,
        )
        sim.process_arrival(flow, gate_pred=0.0)

    # Steady state (discard first 5,000 flows)
    steady_waits = sim.waiting_times_admitted[5000:]
    w_q_simulated = float(np.mean(steady_waits))

    rel_error = abs(w_q_simulated - w_q_theoretical) / w_q_theoretical
    # 15% tolerance for finite simulation of 25,000 steady-state flows
    assert rel_error < 0.15, (
        f"M/D/1 closed form violated: simulated W_q={w_q_simulated*1000:.4f} ms, "
        f"theoretical={w_q_theoretical*1000:.4f} ms, rel_err={rel_error:.4f}"
    )


def test_v3_mm1k_blocking_benchmark():
    """
    V3: Validate M/M/1/K Blocking Benchmark P_K = (1 - rho)*rho^K / (1 - rho^{K+1})
    using Q_max = B and K = Q_max + 1.
    """
    q_max = 5            # Waiting buffer capacity B = 5
    k_system = q_max + 1 # Total system capacity K = 6
    s_bar = 0.001        # 1 ms mean service
    rho_target = 0.80
    lambda_arr = rho_target / s_bar # 800 flows/s
    n_flows = 50000

    # Theoretical M/M/1/K blocking probability
    p_k_theoretical = ((1.0 - rho_target) * (rho_target ** k_system)) / (1.0 - (rho_target ** (k_system + 1)))

    sim = WorkloadQueueSimulator(
        q_max=q_max,
        w_shed=10.0,         # large w_shed so drops are driven strictly by buffer overflow N_q == Q_max
        s_gru_bar=s_bar,
        token_bucket_rate=100000.0,
        tau_gate=-100.0,
        deterministic_service=False, # Exponential service times for M/M/1/K
    )

    # Use distinct independent seeds for arrival process and service duration
    gen = PoissonTrafficGenerator(rate=lambda_arr, seed=42)
    timestamps = gen.generate_timestamps(n_flows)
    rng_svc = np.random.default_rng(123)
    # Exponential service times
    service_times = rng_svc.exponential(scale=s_bar, size=n_flows)

    for i in range(n_flows):
        flow = FlowArrival(
            flow_id=i,
            arrival_time=float(timestamps[i]),
            x=np.zeros(38),
            y_true=0,
            attack_family="Benign",
            service_time=float(service_times[i]),
        )
        sim.process_arrival(flow, gate_pred=0.0)

    # In our 4 outcomes, when w_shed is large, flows failing buffer capacity are ACTION_LOAD_SHED_AE
    dropped_or_shed = sim.n_load_shed_ae
    p_k_simulated = float(dropped_or_shed / n_flows)

    rel_error = abs(p_k_simulated - p_k_theoretical) / p_k_theoretical
    assert rel_error < 0.20, (
        f"M/M/1/K blocking benchmark violated: simulated P_K={p_k_simulated:.4f}, "
        f"theoretical={p_k_theoretical:.4f}, rel_err={rel_error:.4f}"
    )


def test_v4_flow_conservation_identity():
    """
    V4: Verify exact 4-outcome Flow Conservation Identity across random workloads:
      N_tot == N_frontend_drop + N_gate_reject_ae + N_load_shed_ae + N_heavy_admitted
    """
    sim = WorkloadQueueSimulator(
        q_max=10,
        w_shed=0.001,             # 1 ms SLA threshold
        s_gru_bar=0.00018,
        token_bucket_rate=12000.0,# rate limiting to produce frontend drops
        burst_capacity=10.0,
        tau_gate=-0.5,            # threshold allowing ~70% gate triggers
    )

    n_flows = 5000
    gen = PoissonTrafficGenerator(rate=15000.0, seed=777)
    timestamps = gen.generate_timestamps(n_flows)
    rng = np.random.default_rng(888)

    for i in range(n_flows):
        flow = FlowArrival(
            flow_id=i,
            arrival_time=float(timestamps[i]),
            x=np.zeros(38),
            y_true=1 if (i % 5 == 0) else 0,
            attack_family="Exploit" if (i % 5 == 0) else "Benign",
            service_time=0.00018,
        )
        # Random gate score spanning both sides of tau_gate = -0.5
        gate_pred = float(rng.normal(loc=0.0, scale=1.0))
        ae_mse = float(rng.exponential(scale=0.01))
        sim.process_arrival(flow, gate_pred=gate_pred, ae_mse=ae_mse)

    summary = sim.get_summary_metrics()
    counts = summary["flow_counts"]

    total = counts["total"]
    fe_drop = counts["frontend_drop"]
    gate_rej = counts["gate_reject_ae"]
    load_shed = counts["load_shed_ae"]
    heavy_adm = counts["heavy_admitted"]

    assert total == n_flows, f"Total flow count mismatch: {total} != {n_flows}"
    assert total == (fe_drop + gate_rej + load_shed + heavy_adm), (
        f"Conservation violated: total={total} vs sum={fe_drop + gate_rej + load_shed + heavy_adm}"
    )
    assert counts["conservation_verified"] is True

    # Check non-trivial distribution across outcomes
    assert fe_drop > 0, "Expected non-zero frontend drops under overload"
    assert gate_rej > 0, "Expected non-zero gate rejections"
    assert load_shed > 0, "Expected non-zero load shedding"
    assert heavy_adm > 0, "Expected non-zero heavy admissions"
    assert heavy_adm > 0, "Expected non-zero heavy admissions"


def test_v5_cryptographic_immutability_audit():
    """
    V5: Re-verify SHA-256 hashes of Stage 1–3 artifacts against manifest.
    """
    manifest_path = Path("outputs/logs/stage4_artifact_hashes.json")
    assert manifest_path.exists(), "stage4_artifact_hashes.json manifest missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    for cid, hashes in manifest["stage2_data"].items():
        npz_file = stage2_dir / cid / "data.npz"
        with open(npz_file, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        assert current_hash == hashes["data.npz"], f"Immutability check failed for {cid} data.npz"


def test_v6_si_unit_consistency_audit():
    """
    V6: Verify physical units ([s], [ms], [flows/s], Erlangs) across summary metrics.
    """
    sim = WorkloadQueueSimulator(
        q_max=20,
        w_shed=0.050,  # 50 ms = 0.050 s
        s_gru_bar=0.0001799,
        token_bucket_rate=34943.4,
    )

    flow = FlowArrival(
        flow_id=1,
        arrival_time=0.001,
        x=np.zeros(38),
        y_true=1,
        attack_family="Generic",
        service_time=0.0001799,
    )
    sim.process_arrival(flow, gate_pred=10.0, ae_mse=0.05)
    summary = sim.get_summary_metrics()
    qt = summary["queue_telemetry"]

    # SI time units: s vs ms consistency
    assert np.isclose(qt["mean_waiting_time_ms"], qt["mean_waiting_time_s"] * 1000.0)
    assert np.isclose(qt["W_shed_ms"], qt["W_shed_s"] * 1000.0)

    # Rates: flows per second >= 0
    assert qt["lambda_arrival_fps"] >= 0.0
    assert qt["lambda_admitted_fps"] >= 0.0

    # Dimensionless: offered load in Erlangs >= 0
    assert qt["rho_offered"] >= 0.0

    # Probabilities / Fractions: in [0, 100] %
    rf = summary["routing_fractions_pct"]
    for k in ["frontend_drop_pct", "gate_reject_ae_pct", "load_shed_ae_pct", "heavy_admitted_pct"]:
        assert 0.0 <= rf[k] <= 100.0

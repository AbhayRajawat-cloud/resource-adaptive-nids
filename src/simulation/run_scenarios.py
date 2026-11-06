"""
Stage 4: Scenario Execution Orchestrator (Scenarios S1 to S7).

Executes the WorkloadQueueSimulator across all 10 federated clients under:
- S1 (Nominal Operating Baseline): Poisson arrivals with rho in {0.10, 0.25, 0.40}.
- S2 (Offered Load Sweep): Poisson arrivals with rho in [0.1, 3.0] across 15 steps.
- S3 (Transient Surge / Flash Crowd Burst): 3-phase step burst (low -> 5x mu_heavy surge -> low).
- S4 (Pareto Microbursts): Heavy-tailed Pareto arrivals (alpha=1.2) with infinite variance.
- S5 (SLA Delay Sweep): W_shed in [1.0, 100.0] ms under overload (rho=2.0).
- S6 (Threshold Trade-off Comparison): Common Random Numbers (CRN) comparing Q99 vs Q95.
- S7 (Extreme Non-IID Edge Stress Test): Targeted deep dive on Client 02 and Client 04.

Outputs:
- outputs/metrics/stage4/stage4_simulation_results.json
- outputs/metrics/stage4/stage4_scenario_summary.csv
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.simulation.queue_simulator import WorkloadQueueSimulator, FlowArrival
from src.simulation.traffic_generator import (
    PoissonTrafficGenerator,
    MMPP2TrafficGenerator,
    ParetoTrafficGenerator,
    ClientTrafficTraceSampler,
)


def load_configurations() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load configuration YAML, hardware latency profile, and Stage 3 thresholds."""
    config_path = Path("configs/stage4_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    hw_profile_path = Path("outputs/metrics/stage4/hardware_latency_profile.json")
    with open(hw_profile_path, "r", encoding="utf-8") as f:
        hw_profile = json.load(f)

    thresh_path = Path("outputs/experiments/stage3_threshold_sensitivity/sensitivity_client_thresholds.json")
    with open(thresh_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    return config, hw_profile, thresholds


def initialize_samplers(seed: int = 42) -> Dict[str, ClientTrafficTraceSampler]:
    """Pre-initialize and score samplers for all 10 clients once."""
    print("Pre-initializing ClientTrafficTraceSamplers for all 10 clients...")
    samplers = {}
    stage2_dir = Path("data/processed/alpha_0.5_seed_42")
    ckpt_dir = Path("outputs/checkpoints/stage3")

    for cid_int in range(10):
        cid_str = f"client_{cid_int:02d}"
        npz_path = str(stage2_dir / cid_str / "data.npz")
        ae_path = str(ckpt_dir / f"{cid_str}_autoencoder.pt")
        gate_path = str(ckpt_dir / f"{cid_str}_gate.pt")

        samplers[cid_str] = ClientTrafficTraceSampler(
            client_id=cid_int,
            data_npz_path=npz_path,
            ae_checkpoint_path=ae_path,
            gate_checkpoint_path=gate_path,
            seed=seed + cid_int,
        )
    print("All 10 client samplers successfully initialized.")
    return samplers


def run_simulation_instance(
    sampler: ClientTrafficTraceSampler,
    timestamps: np.ndarray,
    tau_gate: float,
    tau_ae: float,
    q_max: int,
    w_shed: float,
    s_gru_bar: float,
    token_bucket_rate: float,
    service_scv: float = 0.0105,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute a single queue simulation instance over an arrival trace."""
    sim = WorkloadQueueSimulator(
        q_max=q_max,
        w_shed=w_shed,
        s_gru_bar=s_gru_bar,
        token_bucket_rate=token_bucket_rate,
        burst_capacity=float(q_max),
        tau_gate=tau_gate,
        tau_ae=tau_ae,
        deterministic_service=False,
    )

    arrivals = sampler.sample_arrivals(
        timestamps=timestamps,
        s_gru_bar=s_gru_bar,
        service_scv=service_scv,
    )

    for flow in arrivals:
        sim.process_arrival(
            flow=flow,
            gate_pred=flow.precomputed_gate_pred,
            ae_mse=flow.precomputed_ae_mse,
        )

    return sim.get_summary_metrics()


def generate_step_burst_timestamps(
    lambda_low: float,
    lambda_surge: float,
    t_phase1: float = 5.0,
    t_phase2: float = 10.0,
    t_phase3: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate 3-phase step-burst arrival timestamps: low -> surge -> recovery."""
    rng = np.random.default_rng(seed)
    timestamps = []
    t = 0.0

    # Phase 1: Low baseline
    while t < t_phase1:
        t += rng.exponential(1.0 / lambda_low)
        if t < t_phase1:
            timestamps.append(t)

    # Phase 2: High surge
    t_end_p2 = t_phase1 + t_phase2
    while t < t_end_p2:
        t += rng.exponential(1.0 / lambda_surge)
        if t < t_end_p2:
            timestamps.append(t)

    # Phase 3: Recovery
    t_end_p3 = t_end_p2 + t_phase3
    while t < t_end_p3:
        t += rng.exponential(1.0 / lambda_low)
        if t < t_end_p3:
            timestamps.append(t)

    return np.array(timestamps, dtype=np.float64)


def extract_row_record(
    scenario_id: str,
    scenario_name: str,
    client_id: str,
    arrival_process: str,
    quantile: str,
    metrics: Dict[str, Any],
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flatten metrics into a standardized tabular CSV row record."""
    counts = metrics["flow_counts"]
    pcts = metrics["routing_fractions_pct"]
    sec = metrics["security_accounting"]
    qt = metrics["queue_telemetry"]

    row = {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "client_id": client_id,
        "arrival_process": arrival_process,
        "quantile": quantile,
        "rho_offered": qt["rho_offered"],
        "lambda_arrival_fps": qt["lambda_arrival_fps"],
        "lambda_admitted_fps": qt["lambda_admitted_fps"],
        "W_shed_ms": qt["W_shed_ms"],
        "buffer_capacity_Q_max": qt["buffer_capacity_Q_max"],
        "total_flows": counts["total"],
        "frontend_drop_count": counts["frontend_drop"],
        "frontend_drop_pct": pcts["frontend_drop_pct"],
        "gate_reject_ae_count": counts["gate_reject_ae"],
        "gate_reject_ae_pct": pcts["gate_reject_ae_pct"],
        "load_shed_ae_count": counts["load_shed_ae"],
        "load_shed_ae_pct": pcts["load_shed_ae_pct"],
        "heavy_admitted_count": counts["heavy_admitted"],
        "heavy_admitted_pct": pcts["heavy_admitted_pct"],
        "conservation_verified": counts["conservation_verified"],
        "effective_heavy_passage_recall": sec["effective_heavy_passage_recall"],
        "effective_heavy_benign_fpr": sec["effective_heavy_benign_fpr"],
        "ae_fallback_recall": sec["ae_fallback_recall"],
        "ae_fallback_fpr": sec["ae_fallback_fpr"],
        "ideal_routing_yield": sec["ideal_routing_yield"],
        "mean_waiting_time_ms": qt["mean_waiting_time_ms"],
        "median_waiting_time_ms": qt["median_waiting_time_ms"],
        "p95_waiting_time_ms": qt["p95_waiting_time_ms"],
        "p99_waiting_time_ms": qt["p99_waiting_time_ms"],
        "max_waiting_time_ms": qt["max_waiting_time_ms"],
        "mean_queue_length": qt["mean_queue_length"],
        "p95_queue_length": qt["p95_queue_length"],
        "max_queue_length": qt["max_queue_length"],
        "sim_duration_s": qt["sim_duration_s"],
    }
    if extra_params:
        row.update(extra_params)
    return row


def run_all_scenarios():
    """Main orchestration loop executing Scenarios S1 to S7 across all clients."""
    t_start = time.perf_counter()
    config, hw_profile, thresholds = load_configurations()
    samplers = initialize_samplers(seed=config.get("seed", 42))

    # Hardware profile parameters
    macro = hw_profile["macro_summary"]
    s_gru_bar = macro["mean_s_gru_ms"] / 1000.0   # seconds (~0.180 ms)
    mu_heavy = macro["mu_heavy_fps"]               # ~5,559.8 flows/s
    lambda_frontend = macro["mean_lambda_frontend_max_fps"] # ~34,943 flows/s
    service_scv = macro["c_s_gru_squared"]         # ~0.0105

    default_q_max = config["queue_parameters"]["buffer_capacity_Q_max"] # 50
    default_w_shed = config["queue_parameters"]["W_shed_seconds"]       # 0.050 s

    all_csv_rows: List[Dict[str, Any]] = []
    simulation_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "host_hardware": "Host CPU",
            "s_gru_bar_s": s_gru_bar,
            "mu_heavy_fps": mu_heavy,
            "lambda_frontend_max_fps": lambda_frontend,
            "service_scv": service_scv,
            "default_q_max": default_q_max,
            "default_w_shed_s": default_w_shed,
        },
        "scenarios": {},
    }

    client_list = [f"client_{i:02d}" for i in range(10)]

    # ---------------------------------------------------------
    # Scenario S1: Nominal Operating Baseline
    # ---------------------------------------------------------
    print("\n--- Running Scenario S1: Nominal Operating Baseline ---")
    s1_results = {}
    rho_s1_list = [0.10, 0.25, 0.40]
    n_flows_s1 = 20000

    for rho in rho_s1_list:
        sub_id = f"rho_{rho:.2f}"
        s1_results[sub_id] = {}
        lam = rho * mu_heavy

        for cid in client_list:
            gen = PoissonTrafficGenerator(rate=lam, seed=42)
            ts = gen.generate_timestamps(n_flows_s1)
            t_gate = thresholds[cid]["thresholds"]["Q99"]["tau_gate"]
            t_ae = thresholds[cid]["thresholds"]["Q99"]["tau_ae"]

            m = run_simulation_instance(
                sampler=samplers[cid],
                timestamps=ts,
                tau_gate=t_gate,
                tau_ae=t_ae,
                q_max=default_q_max,
                w_shed=default_w_shed,
                s_gru_bar=s_gru_bar,
                token_bucket_rate=lambda_frontend,
                service_scv=service_scv,
            )
            s1_results[sub_id][cid] = m
            all_csv_rows.append(extract_row_record(
                scenario_id="S1",
                scenario_name="Nominal Operating Baseline",
                client_id=cid,
                arrival_process="poisson",
                quantile="Q99",
                metrics=m,
                extra_params={"target_rho": rho},
            ))

    simulation_results["scenarios"]["S1"] = s1_results
    print("Scenario S1 completed across all 10 clients.")

    # ---------------------------------------------------------
    # Scenario S2: Offered Load Sweep (rho in [0.1, 3.0] in 15 steps)
    # ---------------------------------------------------------
    print("\n--- Running Scenario S2: Offered Load Sweep (15 steps) ---")
    s2_results = {}
    rho_steps = np.linspace(0.1, 3.0, 15)
    n_flows_s2 = 15000

    for step_idx, rho in enumerate(rho_steps):
        step_key = f"step_{step_idx:02d}_rho_{rho:.2f}"
        s2_results[step_key] = {}
        lam = rho * mu_heavy

        for cid in client_list:
            gen = PoissonTrafficGenerator(rate=lam, seed=100 + step_idx)
            ts = gen.generate_timestamps(n_flows_s2)
            t_gate = thresholds[cid]["thresholds"]["Q99"]["tau_gate"]
            t_ae = thresholds[cid]["thresholds"]["Q99"]["tau_ae"]

            m = run_simulation_instance(
                sampler=samplers[cid],
                timestamps=ts,
                tau_gate=t_gate,
                tau_ae=t_ae,
                q_max=default_q_max,
                w_shed=default_w_shed,
                s_gru_bar=s_gru_bar,
                token_bucket_rate=lambda_frontend,
                service_scv=service_scv,
            )
            s2_results[step_key][cid] = m
            all_csv_rows.append(extract_row_record(
                scenario_id="S2",
                scenario_name="Offered Load Sweep",
                client_id=cid,
                arrival_process="poisson",
                quantile="Q99",
                metrics=m,
                extra_params={"target_rho": float(rho), "sweep_step": step_idx},
            ))

    simulation_results["scenarios"]["S2"] = s2_results
    print("Scenario S2 completed across all 10 clients (15 steps).")

    # ---------------------------------------------------------
    # Scenario S3: Transient Surge / Flash Crowd Burst
    # ---------------------------------------------------------
    print("\n--- Running Scenario S3: Transient Surge (3-Phase Step Burst) ---")
    s3_results = {}
    lam_low = 0.30 * mu_heavy     # 30% background load
    lam_surge = 5.0 * mu_heavy    # 500% flash crowd burst for 10 s

    for cid in client_list:
        ts = generate_step_burst_timestamps(
            lambda_low=lam_low,
            lambda_surge=lam_surge,
            t_phase1=5.0,
            t_phase2=10.0,
            t_phase3=5.0,
            seed=300,
        )
        t_gate = thresholds[cid]["thresholds"]["Q99"]["tau_gate"]
        t_ae = thresholds[cid]["thresholds"]["Q99"]["tau_ae"]

        m = run_simulation_instance(
            sampler=samplers[cid],
            timestamps=ts,
            tau_gate=t_gate,
            tau_ae=t_ae,
            q_max=default_q_max,
            w_shed=default_w_shed,
            s_gru_bar=s_gru_bar,
            token_bucket_rate=lambda_frontend,
            service_scv=service_scv,
        )
        s3_results[cid] = m
        all_csv_rows.append(extract_row_record(
            scenario_id="S3",
            scenario_name="Transient Surge Burst",
            client_id=cid,
            arrival_process="step_burst",
            quantile="Q99",
            metrics=m,
            extra_params={"lambda_surge_multiplier": 5.0, "surge_duration_s": 10.0},
        ))

    simulation_results["scenarios"]["S3"] = s3_results
    print("Scenario S3 completed across all 10 clients.")

    # ---------------------------------------------------------
    # Scenario S4: Heavy-Tailed Pareto Microbursts (alpha=1.2)
    # ---------------------------------------------------------
    print("\n--- Running Scenario S4: Heavy-Tailed Pareto Microbursts (alpha=1.2) ---")
    s4_results = {}
    pareto_gen = ParetoTrafficGenerator(alpha=1.2, mean_rate=0.50 * mu_heavy, seed=400)
    n_flows_s4 = 25000
    ts_pareto, empirical_scv_ca2 = pareto_gen.generate_timestamps(n_flows_s4)

    for cid in client_list:
        t_gate = thresholds[cid]["thresholds"]["Q99"]["tau_gate"]
        t_ae = thresholds[cid]["thresholds"]["Q99"]["tau_ae"]

        m = run_simulation_instance(
            sampler=samplers[cid],
            timestamps=ts_pareto,
            tau_gate=t_gate,
            tau_ae=t_ae,
            q_max=default_q_max,
            w_shed=default_w_shed,
            s_gru_bar=s_gru_bar,
            token_bucket_rate=lambda_frontend,
            service_scv=service_scv,
        )
        s4_results[cid] = m
        all_csv_rows.append(extract_row_record(
            scenario_id="S4",
            scenario_name="Pareto Microbursts",
            client_id=cid,
            arrival_process="pareto",
            quantile="Q99",
            metrics=m,
            extra_params={"pareto_alpha": 1.2, "empirical_c_a_squared": empirical_scv_ca2},
        ))

    simulation_results["scenarios"]["S4"] = {
        "empirical_c_a_squared": empirical_scv_ca2,
        "theoretical_variance": "infinity",
        "clients": s4_results,
    }
    print(f"Scenario S4 completed (empirical arrival SCV c_a^2 = {empirical_scv_ca2:.4f}).")

    # ---------------------------------------------------------
    # Scenario S5: SLA Delay Sweep under Overload (rho=2.0)
    # ---------------------------------------------------------
    print("\n--- Running Scenario S5: SLA Delay Sweep under Overload (rho=2.0) ---")
    s5_results = {}
    w_shed_sweep_ms = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    lam_overload = 2.0 * mu_heavy
    n_flows_s5 = 20000

    for w_ms in w_shed_sweep_ms:
        sub_key = f"w_shed_{w_ms:.1f}ms"
        s5_results[sub_key] = {}
        w_sec = w_ms / 1000.0

        for cid in client_list:
            gen = PoissonTrafficGenerator(rate=lam_overload, seed=500 + int(w_ms))
            ts = gen.generate_timestamps(n_flows_s5)
            t_gate = thresholds[cid]["thresholds"]["Q99"]["tau_gate"]
            t_ae = thresholds[cid]["thresholds"]["Q99"]["tau_ae"]

            m = run_simulation_instance(
                sampler=samplers[cid],
                timestamps=ts,
                tau_gate=t_gate,
                tau_ae=t_ae,
                q_max=default_q_max,
                w_shed=w_sec,
                s_gru_bar=s_gru_bar,
                token_bucket_rate=lambda_frontend,
                service_scv=service_scv,
            )
            s5_results[sub_key][cid] = m
            all_csv_rows.append(extract_row_record(
                scenario_id="S5",
                scenario_name="SLA Delay Sweep",
                client_id=cid,
                arrival_process="poisson",
                quantile="Q99",
                metrics=m,
                extra_params={"target_rho": 2.0, "sweep_W_shed_ms": w_ms},
            ))

    simulation_results["scenarios"]["S5"] = s5_results
    print("Scenario S5 completed across all 10 clients.")

    # ---------------------------------------------------------
    # Scenario S6: Threshold Trade-off (Q99 Baseline vs Q95 Candidate)
    # ---------------------------------------------------------
    print("\n--- Running Scenario S6: Threshold Trade-off (Q99 Baseline vs Q95 Candidate) ---")
    s6_results = {"Q99": {}, "Q95": {}}
    rho_s6_list = [0.50, 1.00, 1.50]
    n_flows_s6 = 25000

    for rho in rho_s6_list:
        sub_key = f"rho_{rho:.2f}"
        s6_results["Q99"][sub_key] = {}
        s6_results["Q95"][sub_key] = {}
        lam = rho * mu_heavy

        # Common Random Numbers (CRN) across quantiles
        gen = PoissonTrafficGenerator(rate=lam, seed=600 + int(rho * 100))
        ts_crn = gen.generate_timestamps(n_flows_s6)

        for cid in client_list:
            for q_name in ["Q99", "Q95"]:
                t_gate = thresholds[cid]["thresholds"][q_name]["tau_gate"]
                t_ae = thresholds[cid]["thresholds"][q_name]["tau_ae"]

                m = run_simulation_instance(
                    sampler=samplers[cid],
                    timestamps=ts_crn,
                    tau_gate=t_gate,
                    tau_ae=t_ae,
                    q_max=default_q_max,
                    w_shed=default_w_shed,
                    s_gru_bar=s_gru_bar,
                    token_bucket_rate=lambda_frontend,
                    service_scv=service_scv,
                )
                s6_results[q_name][sub_key][cid] = m
                all_csv_rows.append(extract_row_record(
                    scenario_id="S6",
                    scenario_name="Threshold Comparison Q99 vs Q95",
                    client_id=cid,
                    arrival_process="poisson_crn",
                    quantile=q_name,
                    metrics=m,
                    extra_params={"target_rho": rho, "common_random_numbers": True},
                ))

    simulation_results["scenarios"]["S6"] = s6_results
    print("Scenario S6 completed across all 10 clients.")

    # ---------------------------------------------------------
    # Scenario S7: Extreme Non-IID Edge Stress Test (Clients 02 and 04)
    # ---------------------------------------------------------
    print("\n--- Running Scenario S7: Extreme Non-IID Stress Test (Client 02 and Client 04) ---")
    s7_results = {}
    target_s7_clients = ["client_02", "client_04"]
    mmpp_gen = MMPP2TrafficGenerator(
        lambda_1=0.20 * mu_heavy,  # Normal: ~1,112 fps
        lambda_2=2.50 * mu_heavy,  # Surge: ~13,900 fps
        q_12=0.5,
        q_21=1.5,
        seed=700,
    )
    n_flows_s7 = 30000
    ts_s7 = mmpp_gen.generate_timestamps(n_flows_s7)
    buffer_sweep = [20, 50, 100]

    for cid in target_s7_clients:
        s7_results[cid] = {}
        for q_cap in buffer_sweep:
            buf_key = f"Q_max_{q_cap}"
            s7_results[cid][buf_key] = {}

            for q_name in ["Q99", "Q95"]:
                t_gate = thresholds[cid]["thresholds"][q_name]["tau_gate"]
                t_ae = thresholds[cid]["thresholds"][q_name]["tau_ae"]

                m = run_simulation_instance(
                    sampler=samplers[cid],
                    timestamps=ts_s7,
                    tau_gate=t_gate,
                    tau_ae=t_ae,
                    q_max=q_cap,
                    w_shed=default_w_shed,
                    s_gru_bar=s_gru_bar,
                    token_bucket_rate=lambda_frontend,
                    service_scv=service_scv,
                )
                s7_results[cid][buf_key][q_name] = m
                all_csv_rows.append(extract_row_record(
                    scenario_id="S7",
                    scenario_name="Non-IID Client Stress Test",
                    client_id=cid,
                    arrival_process="mmpp2",
                    quantile=q_name,
                    metrics=m,
                    extra_params={"target_client": cid, "test_Q_max": q_cap},
                ))

    simulation_results["scenarios"]["S7"] = s7_results
    print("Scenario S7 completed for Client 02 and Client 04.")

    # ---------------------------------------------------------
    # Export Results
    # ---------------------------------------------------------
    metrics_dir = Path("outputs/metrics/stage4")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    json_path = metrics_dir / "stage4_simulation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(simulation_results, f, indent=2)
    print(f"\n[OK] Successfully exported simulation JSON to: {json_path}")

    csv_path = metrics_dir / "stage4_scenario_summary.csv"
    df_summary = pd.DataFrame(all_csv_rows)
    df_summary.to_csv(csv_path, index=False)
    print(f"[OK] Successfully exported scenario summary CSV ({len(df_summary)} rows) to: {csv_path}")

    t_end = time.perf_counter()
    print(f"[DONE] Full Phase 4.3 simulation execution completed in {t_end - t_start:.2f} seconds.")


if __name__ == "__main__":
    run_all_scenarios()


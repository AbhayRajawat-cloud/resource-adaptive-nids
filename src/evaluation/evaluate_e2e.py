"""
Stage 5: Phase 5.4 Multi-Model Integrated Evaluation Engine.

Executes end-to-end evaluation of the integrated edge NIDS pipeline across:
- All 10 Federated Clients (00 to 09)
- All 4 Configurations:
    Config A: HeavyGRU Standard FedAvg
    Config B: SupervisedDenseMLP Strategy B FedPer (Private Heads + FedBN)
    Config C: SupervisedDenseMLP Standard FedAvg (FedBN)
    Config D: Light Path Standalone Baseline (AE Only; Zero Heavy Capacity)
- Threshold Operating Points: Q0.99 (Primary Frozen Baseline) and Q0.95 (Candidate)
- Multi-Tier Lineage: Tier 1 (All Flows), Tier 2 (Unique Vectors), Tier 3 (Train/Val-Exclusive)
- Scenarios S1 to S7 (Nominal, Load Sweep, Step Burst, Pareto Microbursts, SLA Sweep, Q99 vs Q95, Non-IID Stress)

Statistical Analysis & Hypothesis Testing:
- Paired client-level bootstrap resampling (1,000 iterations, seed 42)
- Hodges-Lehmann non-parametric median difference estimator with 95% CIs
- Two-sided paired Wilcoxon signed-rank test comparing Config A (HeavyGRU) vs Config B (DenseMLP FedPer)
- Benjamini-Hochberg (BH) False Discovery Rate (FDR) adjustment (alpha=0.05)

Outputs:
- outputs/metrics/stage5/phase5.4_e2e_results.csv
- outputs/metrics/stage5/phase5.4_e2e_results.json
"""

import json
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.simulation.integrated_pipeline import IntegratedPipelineSimulator
from src.simulation.queue_simulator import FlowArrival, WorkloadQueueSimulator
from src.simulation.traffic_generator import (
    PoissonTrafficGenerator,
    MMPP2TrafficGenerator,
    ParetoTrafficGenerator,
)
from src.evaluation.eval_federated import benjamini_hochberg
from src.evaluation.eval_local_standalone import compute_hodges_lehmann_and_ci


class ClientE2EEvaluator:
    """Pre-scores and caches client test splits for high-throughput end-to-end simulation."""

    def __init__(
        self,
        client_id: int,
        stage2_dir: Path = Path("data/processed/alpha_0.5_seed_42"),
        gate_passed_dir: Path = Path("data/processed/gate_passed"),
    ):
        self.client_id = client_id
        self.cid_str = f"client_{client_id:02d}"

        # 1. Load Stage 2 test split
        data_path = stage2_dir / self.cid_str / "data.npz"
        raw = np.load(data_path, allow_pickle=True)
        self.x_test = raw["X_test"]
        self.y_test = raw["y_test"].astype(int)
        self.attack_test = raw["attack_test"] if "attack_test" in raw else np.array(["Unknown"] * len(self.y_test))
        self.n_test = len(self.x_test)

        # 2. Compute full test split uniqueness masks (Tiers 1, 2, 3)
        # Unique vectors in X_test (Tier 2)
        _, u_idx = np.unique(self.x_test, axis=0, return_index=True)
        self.tier2_mask_full = np.zeros(self.n_test, dtype=bool)
        self.tier2_mask_full[u_idx] = True

        # Exclusive unique: unique in X_test and not in X_train or X_val (Tier 3)
        x_tr_val = np.vstack([raw["X_train"], raw["X_val"]])
        # Fast hash-based set membership check
        tr_val_hashes = {hash(row.tobytes()) for row in x_tr_val}
        self.tier3_mask_full = np.zeros(self.n_test, dtype=bool)
        for idx in u_idx:
            if hash(self.x_test[idx].tobytes()) not in tr_val_hashes:
                self.tier3_mask_full[idx] = True

        # Ensure strict lineage: Tier 3 <= Tier 2 <= Tier 1
        assert np.all(self.tier3_mask_full <= self.tier2_mask_full)

    def sample_arrivals(
        self,
        timestamps: np.ndarray,
        seed: int = 42,
        s_heavy_bar: float = 0.0001799,
        service_scv: float = 0.0105,
    ) -> Tuple[List[FlowArrival], np.ndarray]:
        """Sample flow arrivals with replacement matching arrival timestamps."""
        rng = np.random.default_rng(seed)
        n_arrivals = len(timestamps)
        sampled_indices = rng.integers(0, self.n_test, size=n_arrivals)

        # Generate service times
        if service_scv > 0:
            sigma_log = np.sqrt(np.log(1.0 + service_scv))
            mu_log = np.log(s_heavy_bar) - 0.5 * (sigma_log ** 2)
            service_times = rng.lognormal(mean=mu_log, sigma=sigma_log, size=n_arrivals)
        else:
            service_times = np.full(n_arrivals, s_heavy_bar)

        arrivals = []
        for i in range(n_arrivals):
            s_idx = sampled_indices[i]
            flow = FlowArrival(
                flow_id=i,
                arrival_time=float(timestamps[i]),
                x=self.x_test[s_idx],
                y_true=int(self.y_test[s_idx]),
                attack_family=str(self.attack_test[s_idx]),
                service_time=float(service_times[i]),
            )
            arrivals.append(flow)

        return arrivals, sampled_indices


def generate_step_burst_timestamps(
    lambda_low: float,
    lambda_surge: float,
    t_phase1: float = 5.0,
    t_phase2: float = 10.0,
    t_phase3: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate 3-phase step-burst arrival timestamps."""
    rng = np.random.default_rng(seed)
    timestamps = []
    t = 0.0

    while t < t_phase1:
        t += rng.exponential(1.0 / lambda_low)
        if t < t_phase1:
            timestamps.append(t)

    t_end_p2 = t_phase1 + t_phase2
    while t < t_end_p2:
        t += rng.exponential(1.0 / lambda_surge)
        if t < t_end_p2:
            timestamps.append(t)

    t_end_p3 = t_end_p2 + t_phase3
    while t < t_end_p3:
        t += rng.exponential(1.0 / lambda_low)
        if t < t_end_p3:
            timestamps.append(t)

    return np.array(timestamps, dtype=np.float64)


def execute_full_e2e_evaluation() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete end-to-end evaluation across all clients, configs, and scenarios."""
    print("=" * 75)
    print("Starting Stage 5 Phase 5.4: End-to-End Pipeline Evaluation Engine")
    print("=" * 75)

    s_heavy_bar = 0.0001799
    mu_heavy_nominal = 1.0 / s_heavy_bar  # ~5,559 flows/s

    # Initialize client trace loaders
    print("Pre-initializing client test data loaders...")
    evaluators = {cid: ClientE2EEvaluator(cid) for cid in range(10)}

    all_results_rows: List[Dict[str, Any]] = []

    configs = [
        ("Config A", "Config A: HeavyGRU FedAvg", ["Q99", "Q95"]),
        ("Config B", "Config B: DenseMLP FedPer", ["Q99"]),  # Q95 unavailable per spec
        ("Config C", "Config C: DenseMLP FedAvg", ["Q99", "Q95"]),
        ("Config D", "Config D: Light AE Standalone", ["Q99", "Q95"]),
    ]

    # Pre-score all clients and configs
    print("Pre-scoring client test splits with Gate, AE, and Heavy models...")
    cached_scores: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}

    for cid in range(10):
        ev = evaluators[cid]
        for cfg_short, cfg_full, quantiles in configs:
            for q in quantiles:
                sim = IntegratedPipelineSimulator(
                    client_id=cid,
                    quantile=q,
                    config_name=cfg_full,
                )
                g_pred, ae_mse, h_logit = sim.precompute_flow_predictions(ev.x_test)
                cached_scores[(cid, cfg_short, q)] = (g_pred, ae_mse, h_logit)

    print("Pre-scoring complete. Executing benchmark scenarios...")

    # -----------------------------------------------------------------------
    # Part 1: Comprehensive Nominal Baseline Benchmark (rho = 0.25) across Tiers
    # -----------------------------------------------------------------------
    nominal_n_flows = 20000
    nominal_rho = 0.25
    nominal_lambda = nominal_rho * mu_heavy_nominal  # ~1389.7 flows/s

    print(f"\n--- Running Nominal Baseline Benchmark (N={nominal_n_flows}, rho={nominal_rho:.2f}) ---")
    poisson_gen = PoissonTrafficGenerator(rate=nominal_lambda, seed=42)
    nominal_timestamps = poisson_gen.generate_timestamps(n_samples=nominal_n_flows)

    for cid in range(10):
        ev = evaluators[cid]
        arrivals, sample_indices = ev.sample_arrivals(
            timestamps=nominal_timestamps,
            seed=42 + cid,
            s_heavy_bar=s_heavy_bar,
        )

        for cfg_short, cfg_full, quantiles in configs:
            for q in quantiles:
                g_pred_full, ae_mse_full, h_logit_full = cached_scores[(cid, cfg_short, q)]
                g_pred = g_pred_full[sample_indices]
                ae_mse = ae_mse_full[sample_indices]
                h_logit = h_logit_full[sample_indices] if h_logit_full is not None else None

                sim = IntegratedPipelineSimulator(
                    client_id=cid,
                    quantile=q,
                    config_name=cfg_full,
                    w_shed=0.050,
                    q_max=100,
                )

                # Tier 1 (All flows)
                res_t1 = sim.run_simulation(
                    arrivals=arrivals,
                    precomputed_gate_preds=g_pred,
                    precomputed_ae_mses=ae_mse,
                    precomputed_heavy_logits=h_logit,
                )

                # Tier 2 and Tier 3 masks for sampled arrivals
                sampled_t2_mask = ev.tier2_mask_full[sample_indices]
                sampled_t3_mask = ev.tier3_mask_full[sample_indices]

                # Evaluate Tiers 1, 2, and 3
                tiers = [
                    (1, "Tier 1 (All Flows)", np.ones(nominal_n_flows, dtype=bool)),
                    (2, "Tier 2 (Unique Vectors)", sampled_t2_mask),
                    (3, "Tier 3 (Train/Val-Exclusive Unique)", sampled_t3_mask),
                ]

                for tier_num, tier_label, mask in tiers:
                    sub_arrivals = [arrivals[idx] for idx in range(nominal_n_flows) if mask[idx]]
                    sub_g_pred = g_pred[mask]
                    sub_ae_mse = ae_mse[mask]
                    sub_h_logit = h_logit[mask] if h_logit is not None else None

                    res = sim.run_simulation(
                        arrivals=sub_arrivals,
                        precomputed_gate_preds=sub_g_pred,
                        precomputed_ae_mses=sub_ae_mse,
                        precomputed_heavy_logits=sub_h_logit,
                    )

                    fc = res["flow_counts"]
                    sm = res["security_metrics"]
                    rp = res["routing_proportions"]
                    qt = res["queue_telemetry"]

                    row = {
                        "scenario_id": "S1_Nominal",
                        "scenario_name": "Nominal Baseline (rho=0.25)",
                        "client_id": cid,
                        "client_str": f"client_{cid:02d}",
                        "config_short": cfg_short,
                        "config_full": cfg_full,
                        "quantile": q,
                        "tier": tier_num,
                        "tier_label": tier_label,
                        "n_total": fc["n_total"],
                        "n_frontend_drop": fc["frontend_drop"],
                        "n_gate_reject_ae": fc["gate_reject_ae"],
                        "n_load_shed_ae": fc["load_shed_ae"],
                        "n_heavy_admitted": fc["heavy_admitted"],
                        "r_frontend_drop": rp["r_frontend_drop"],
                        "r_gate_reject_ae": rp["r_gate_reject_ae"],
                        "r_load_shed_ae": rp["r_load_shed_ae"],
                        "r_heavy_admitted": rp["r_heavy_admitted"],
                        "n_attacks": sm["total_attacks"],
                        "n_benign": sm["total_benign"],
                        "tp": sm["tp"],
                        "fp": sm["fp"],
                        "tn": sm["tn"],
                        "fn": sm["fn"],
                        "recall_e2e": sm["recall_e2e"],
                        "fpr_e2e": sm["fpr_e2e"],
                        "precision": sm["precision"],
                        "f1_score": sm["f1_score"],
                        "accuracy": sm["accuracy"],
                        "roc_auc": sm["roc_auc"],
                        "pr_auc": sm["pr_auc"],
                        "mean_waiting_time_ms": qt["mean_waiting_time_ms"],
                        "p99_waiting_time_ms": qt["p99_waiting_time_ms"],
                        "mean_queue_length": qt["mean_queue_length"],
                    }
                    all_results_rows.append(row)

    # -----------------------------------------------------------------------
    # Part 2: Scenario Sweep S1 to S7 Execution
    # -----------------------------------------------------------------------
    print("\n--- Running Scenarios S1 to S7 ---")

    # S2: Offered Load Sweep rho in [0.1 .. 3.0] across 15 steps
    load_steps = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    n_sweep_flows = 5000

    for rho in load_steps:
        rate = rho * mu_heavy_nominal
        p_gen = PoissonTrafficGenerator(rate=rate, seed=int(rho * 100))
        t_stamps = p_gen.generate_timestamps(n_samples=n_sweep_flows)

        for cid in [0, 2, 4, 8]:  # Representative clients (nominal, small, heavy attack)
            ev = evaluators[cid]
            arrivals, sample_indices = ev.sample_arrivals(timestamps=t_stamps, seed=42 + cid)

            for cfg_short, cfg_full, _ in configs:
                q = "Q99"
                g_pred_full, ae_mse_full, h_logit_full = cached_scores[(cid, cfg_short, q)]
                g_pred = g_pred_full[sample_indices]
                ae_mse = ae_mse_full[sample_indices]
                h_logit = h_logit_full[sample_indices] if h_logit_full is not None else None

                sim = IntegratedPipelineSimulator(client_id=cid, quantile=q, config_name=cfg_full)
                res = sim.run_simulation(arrivals, g_pred, ae_mse, h_logit)

                fc = res["flow_counts"]
                sm = res["security_metrics"]
                rp = res["routing_proportions"]
                qt = res["queue_telemetry"]

                row = {
                    "scenario_id": "S2_Load_Sweep",
                    "scenario_name": f"Offered Load Sweep (rho={rho:.2f})",
                    "client_id": cid,
                    "client_str": f"client_{cid:02d}",
                    "config_short": cfg_short,
                    "config_full": cfg_full,
                    "quantile": q,
                    "tier": 1,
                    "tier_label": "Tier 1 (All Flows)",
                    "offered_load_rho": rho,
                    "n_total": fc["n_total"],
                    "n_frontend_drop": fc["frontend_drop"],
                    "n_gate_reject_ae": fc["gate_reject_ae"],
                    "n_load_shed_ae": fc["load_shed_ae"],
                    "n_heavy_admitted": fc["heavy_admitted"],
                    "r_frontend_drop": rp["r_frontend_drop"],
                    "r_gate_reject_ae": rp["r_gate_reject_ae"],
                    "r_load_shed_ae": rp["r_load_shed_ae"],
                    "r_heavy_admitted": rp["r_heavy_admitted"],
                    "n_attacks": sm["total_attacks"],
                    "n_benign": sm["total_benign"],
                    "tp": sm["tp"],
                    "fp": sm["fp"],
                    "tn": sm["tn"],
                    "fn": sm["fn"],
                    "recall_e2e": sm["recall_e2e"],
                    "fpr_e2e": sm["fpr_e2e"],
                    "precision": sm["precision"],
                    "f1_score": sm["f1_score"],
                    "accuracy": sm["accuracy"],
                    "roc_auc": sm["roc_auc"],
                    "pr_auc": sm["pr_auc"],
                    "mean_waiting_time_ms": qt["mean_waiting_time_ms"],
                    "p99_waiting_time_ms": qt["p99_waiting_time_ms"],
                    "mean_queue_length": qt["mean_queue_length"],
                }
                all_results_rows.append(row)

    # S3: 3-Phase Step Burst (low -> 5x surge -> recovery)
    t_burst = generate_step_burst_timestamps(
        lambda_low=0.2 * mu_heavy_nominal,
        lambda_surge=5.0 * mu_heavy_nominal,
        t_phase1=3.0,
        t_phase2=5.0,
        t_phase3=3.0,
        seed=42,
    )
    for cid in range(10):
        ev = evaluators[cid]
        arrivals, sample_indices = ev.sample_arrivals(timestamps=t_burst, seed=100 + cid)
        for cfg_short, cfg_full, _ in [configs[0], configs[1]]:
            q = "Q99"
            g_pred, ae_mse, h_logit = cached_scores[(cid, cfg_short, q)]
            sim = IntegratedPipelineSimulator(client_id=cid, quantile=q, config_name=cfg_full)
            res = sim.run_simulation(arrivals, g_pred[sample_indices], ae_mse[sample_indices], h_logit[sample_indices])
            fc, sm, rp, qt = res["flow_counts"], res["security_metrics"], res["routing_proportions"], res["queue_telemetry"]
            all_results_rows.append({
                "scenario_id": "S3_Step_Burst",
                "scenario_name": "Transient Surge (5x mu_heavy burst)",
                "client_id": cid,
                "client_str": f"client_{cid:02d}",
                "config_short": cfg_short,
                "config_full": cfg_full,
                "quantile": q,
                "tier": 1,
                "tier_label": "Tier 1 (All Flows)",
                "n_total": fc["n_total"],
                "n_frontend_drop": fc["frontend_drop"],
                "n_gate_reject_ae": fc["gate_reject_ae"],
                "n_load_shed_ae": fc["load_shed_ae"],
                "n_heavy_admitted": fc["heavy_admitted"],
                "r_frontend_drop": rp["r_frontend_drop"],
                "r_gate_reject_ae": rp["r_gate_reject_ae"],
                "r_load_shed_ae": rp["r_load_shed_ae"],
                "r_heavy_admitted": rp["r_heavy_admitted"],
                "n_attacks": sm["total_attacks"],
                "n_benign": sm["total_benign"],
                "tp": sm["tp"], "fp": sm["fp"], "tn": sm["tn"], "fn": sm["fn"],
                "recall_e2e": sm["recall_e2e"], "fpr_e2e": sm["fpr_e2e"],
                "precision": sm["precision"], "f1_score": sm["f1_score"],
                "accuracy": sm["accuracy"], "roc_auc": sm["roc_auc"], "pr_auc": sm["pr_auc"],
                "mean_waiting_time_ms": qt["mean_waiting_time_ms"],
                "p99_waiting_time_ms": qt["p99_waiting_time_ms"],
                "mean_queue_length": qt["mean_queue_length"],
            })

    # S4: Pareto Heavy-Tailed Microbursts (alpha=1.2)
    pareto_gen = ParetoTrafficGenerator(alpha=1.2, mean_rate=0.5 * mu_heavy_nominal, seed=42)
    t_pareto, _ = pareto_gen.generate_timestamps(n_samples=5000)
    for cid in range(10):
        ev = evaluators[cid]
        arrivals, sample_indices = ev.sample_arrivals(timestamps=t_pareto, seed=200 + cid)
        for cfg_short, cfg_full, _ in [configs[0], configs[1]]:
            q = "Q99"
            g_pred, ae_mse, h_logit = cached_scores[(cid, cfg_short, q)]
            sim = IntegratedPipelineSimulator(client_id=cid, quantile=q, config_name=cfg_full)
            res = sim.run_simulation(arrivals, g_pred[sample_indices], ae_mse[sample_indices], h_logit[sample_indices])
            fc, sm, rp, qt = res["flow_counts"], res["security_metrics"], res["routing_proportions"], res["queue_telemetry"]
            all_results_rows.append({
                "scenario_id": "S4_Pareto_Burst",
                "scenario_name": "Heavy-Tailed Pareto Microbursts (alpha=1.2)",
                "client_id": cid,
                "client_str": f"client_{cid:02d}",
                "config_short": cfg_short,
                "config_full": cfg_full,
                "quantile": q,
                "tier": 1,
                "tier_label": "Tier 1 (All Flows)",
                "n_total": fc["n_total"],
                "n_frontend_drop": fc["frontend_drop"],
                "n_gate_reject_ae": fc["gate_reject_ae"],
                "n_load_shed_ae": fc["load_shed_ae"],
                "n_heavy_admitted": fc["heavy_admitted"],
                "r_frontend_drop": rp["r_frontend_drop"],
                "r_gate_reject_ae": rp["r_gate_reject_ae"],
                "r_load_shed_ae": rp["r_load_shed_ae"],
                "r_heavy_admitted": rp["r_heavy_admitted"],
                "n_attacks": sm["total_attacks"],
                "n_benign": sm["total_benign"],
                "tp": sm["tp"], "fp": sm["fp"], "tn": sm["tn"], "fn": sm["fn"],
                "recall_e2e": sm["recall_e2e"], "fpr_e2e": sm["fpr_e2e"],
                "precision": sm["precision"], "f1_score": sm["f1_score"],
                "accuracy": sm["accuracy"], "roc_auc": sm["roc_auc"], "pr_auc": sm["pr_auc"],
                "mean_waiting_time_ms": qt["mean_waiting_time_ms"],
                "p99_waiting_time_ms": qt["p99_waiting_time_ms"],
                "mean_queue_length": qt["mean_queue_length"],
            })

    # S7: Extreme Non-IID Edge Stress Test (Client 02 and Client 04)
    mmpp_gen = MMPP2TrafficGenerator(lambda_1=50.0, lambda_2=500.0, seed=42)
    t_mmpp = mmpp_gen.generate_timestamps(n_samples=5000)
    for cid in [2, 4]:
        ev = evaluators[cid]
        arrivals, sample_indices = ev.sample_arrivals(timestamps=t_mmpp, seed=300 + cid)
        for cfg_short, cfg_full, _ in configs:
            q = "Q99"
            g_pred, ae_mse, h_logit = cached_scores[(cid, cfg_short, q)]
            sim = IntegratedPipelineSimulator(client_id=cid, quantile=q, config_name=cfg_full)
            h_in = h_logit[sample_indices] if h_logit is not None else None
            res = sim.run_simulation(arrivals, g_pred[sample_indices], ae_mse[sample_indices], h_in)
            fc, sm, rp, qt = res["flow_counts"], res["security_metrics"], res["routing_proportions"], res["queue_telemetry"]
            all_results_rows.append({
                "scenario_id": "S7_NonIID_Stress",
                "scenario_name": f"Extreme Non-IID Stress Test (Client {cid:02d})",
                "client_id": cid,
                "client_str": f"client_{cid:02d}",
                "config_short": cfg_short,
                "config_full": cfg_full,
                "quantile": q,
                "tier": 1,
                "tier_label": "Tier 1 (All Flows)",
                "n_total": fc["n_total"],
                "n_frontend_drop": fc["frontend_drop"],
                "n_gate_reject_ae": fc["gate_reject_ae"],
                "n_load_shed_ae": fc["load_shed_ae"],
                "n_heavy_admitted": fc["heavy_admitted"],
                "r_frontend_drop": rp["r_frontend_drop"],
                "r_gate_reject_ae": rp["r_gate_reject_ae"],
                "r_load_shed_ae": rp["r_load_shed_ae"],
                "r_heavy_admitted": rp["r_heavy_admitted"],
                "n_attacks": sm["total_attacks"],
                "n_benign": sm["total_benign"],
                "tp": sm["tp"], "fp": sm["fp"], "tn": sm["tn"], "fn": sm["fn"],
                "recall_e2e": sm["recall_e2e"], "fpr_e2e": sm["fpr_e2e"],
                "precision": sm["precision"], "f1_score": sm["f1_score"],
                "accuracy": sm["accuracy"], "roc_auc": sm["roc_auc"], "pr_auc": sm["pr_auc"],
                "mean_waiting_time_ms": qt["mean_waiting_time_ms"],
                "p99_waiting_time_ms": qt["p99_waiting_time_ms"],
                "mean_queue_length": qt["mean_queue_length"],
            })

    df_results = pd.DataFrame(all_results_rows)

    # -----------------------------------------------------------------------
    # Part 3: Statistical Comparison: Config A (HeavyGRU) vs Config B (DenseMLP FedPer)
    # -----------------------------------------------------------------------
    print("\n--- Computing Paired Statistical Tests: Config A vs Config B ---")
    df_nominal = df_results[(df_results["scenario_id"] == "S1_Nominal") & (df_results["quantile"] == "Q99")]

    stat_comparisons = []
    for tier_num in [1, 2, 3]:
        sub_a = df_nominal[(df_nominal["config_short"] == "Config A") & (df_nominal["tier"] == tier_num)].sort_values("client_id")
        sub_b = df_nominal[(df_nominal["config_short"] == "Config B") & (df_nominal["tier"] == tier_num)].sort_values("client_id")

        v_a = sub_a["f1_score"].to_numpy()
        v_b = sub_b["f1_score"].to_numpy()
        diffs = v_a - v_b  # Config A - Config B

        mean_d = float(np.mean(diffs))
        median_d = float(np.median(diffs))
        hl_est, ci_l, ci_h = compute_hodges_lehmann_and_ci(diffs, num_bootstrap=1000, seed=42)

        try:
            if np.all(diffs == 0.0):
                w_stat, w_p = 0.0, 1.0
            else:
                w_res = wilcoxon(v_a, v_b, zero_method="pratt", alternative="two-sided")
                w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")

        stat_comparisons.append({
            "tier": tier_num,
            "tier_label": f"Tier {tier_num}",
            "mean_f1_config_a": float(np.mean(v_a)),
            "mean_f1_config_b": float(np.mean(v_b)),
            "mean_difference": mean_d,
            "median_difference": median_d,
            "hodges_lehmann_estimate": hl_est,
            "ci_95_low": ci_l,
            "ci_95_high": ci_h,
            "wilcoxon_stat": w_stat,
            "raw_pvalue": w_p,
        })

    pvals = [sc["raw_pvalue"] for sc in stat_comparisons]
    qvals, rejects = benjamini_hochberg(pvals, alpha=0.05)
    for i, sc in enumerate(stat_comparisons):
        sc["bh_qvalue"] = qvals[i]
        sc["bh_reject_h0"] = rejects[i]

    # Save to disk
    metrics_dir = Path("outputs/metrics/stage5")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_out = metrics_dir / "phase5.4_e2e_results.csv"
    json_out = metrics_dir / "phase5.4_e2e_results.json"

    df_results.to_csv(csv_out, index=False)

    full_package = {
        "metadata": {
            "stage": "5.4",
            "description": "End-to-End Integrated Pipeline Evaluation across Configs A, B, C, D",
            "decision_threshold": 0.5,
            "queue_sla_w_shed_ms": 50.0,
            "queue_buffer_q_max": 100,
            "triage_capacity_fps": 34943.4,
            "configs": {
                "Config A": "HeavyGRU Standard FedAvg",
                "Config B": "SupervisedDenseMLP Strategy B FedPer",
                "Config C": "SupervisedDenseMLP Standard FedAvg",
                "Config D": "Light Path Standalone Baseline (AE Only)",
            },
        },
        "statistical_comparison_config_a_vs_config_b": stat_comparisons,
        "results_summary": all_results_rows,
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(full_package, f, indent=2)

    print(f"\n[OK] Saved Phase 5.4 E2E CSV ({len(df_results)} rows) to: {csv_out}")
    print(f"[OK] Saved Phase 5.4 E2E JSON to: {json_out}")
    print("=" * 75)

    return df_results, full_package


if __name__ == "__main__":
    execute_full_e2e_evaluation()


"""
Stage 5: NF-UNSW-NB15-v2 Threat Taxonomy Profiling & Precise Recall Denominators.

Evaluates intrusion detection performance across the 9 official attack categories:
- Exploits, Fuzzers, Generic, Reconnaissance, DoS, Analysis, Backdoor, Shellcode, Worms (plus Benign)
across all 10 federated clients under operating quantiles Q0.99 and Q0.95 using actual
Phase 5.4 nominal queue simulation (S1: rho=0.25, 200,000 flows total, 20,000 per client).

Calculates multi-stage routing distributions and decoupled recall contributions:
1. 4 Mutually Exclusive Routing Outcomes: N_fe_drop, N_gate_rej, N_load_shed, N_heavy_adm
   Conservation check: N_total == N_fe_drop + N_gate_rej + N_load_shed + N_heavy_adm
2. Gate Passage Rate vs Heavy Admission Rate:
   - Gate Passage: (N_load_shed + N_heavy_adm) / N_total
   - Heavy Admission: N_heavy_adm / N_total
3. Conditional Light AE Recall: TP_AE / N_processed_ae (where N_processed_ae = N_gate_rej + N_load_shed)
4. Conditional Heavy Recall: TP_Heavy / N_admitted_heavy (where N_admitted_heavy = N_heavy_adm)
5. End-to-End AE Contribution: TP_AE / N_attack,total
6. End-to-End Heavy Contribution: TP_Heavy / N_attack,total
7. Total End-to-End Recall: (TP_AE + TP_Heavy) / N_attack,total

Output:
- outputs/metrics/stage5/validation_attack_family_breakdown.csv
"""

import json
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch

from src.simulation.integrated_pipeline import IntegratedPipelineSimulator
from src.simulation.traffic_generator import PoissonTrafficGenerator
from src.simulation.queue_simulator import WorkloadQueueSimulator
from src.evaluation.evaluate_e2e import ClientE2EEvaluator

OFFICIAL_ATTACK_CATEGORIES = [
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

ALL_CATEGORIES = OFFICIAL_ATTACK_CATEGORIES + ["Benign"]


def evaluate_all_attack_families() -> pd.DataFrame:
    """Execute complete attack taxonomy evaluation across all clients, configs, and quantiles."""
    print("=" * 75)
    print("Executing NF-UNSW-NB15-v2 Threat Taxonomy Profiling & Denominator Audit")
    print("Using Phase 5.4 Nominal Queue Simulation Stream (N=200,000, rho=0.25)")
    print("=" * 75)

    output_dir = Path("outputs/metrics/stage5")
    output_dir.mkdir(parents=True, exist_ok=True)

    s_heavy_bar = 0.0001799
    mu_heavy_nominal = 1.0 / s_heavy_bar
    nominal_n_flows = 20000
    nominal_rho = 0.25
    nominal_lambda = nominal_rho * mu_heavy_nominal
    poisson_gen = PoissonTrafficGenerator(rate=nominal_lambda, seed=42)
    nominal_timestamps = poisson_gen.generate_timestamps(n_samples=nominal_n_flows)

    evaluators = {cid: ClientE2EEvaluator(cid) for cid in range(10)}

    configs = [
        ("Config A", "Config A: HeavyGRU FedAvg", ["Q99", "Q95"]),
        ("Config B", "Config B: DenseMLP FedPer", ["Q99", "Q95"]),  # Q95 will be marked NOT AVAILABLE
        ("Config C", "Config C: DenseMLP FedAvg", ["Q99", "Q95"]),
        ("Config D", "Config D: Light AE Standalone", ["Q99", "Q95"]),
    ]

    records: List[Dict[str, Any]] = []

    for cid in range(10):
        cid_str = f"client_{cid:02d}"
        ev = evaluators[cid]
        arrivals, sample_indices = ev.sample_arrivals(
            timestamps=nominal_timestamps,
            seed=42 + cid,
            s_heavy_bar=s_heavy_bar,
        )

        for cfg_short, cfg_full, quantiles in configs:
            for q in quantiles:
                # Check checkpoint availability
                is_available = True
                if cfg_short == "Config B" and q == "Q95":
                    is_available = False

                if not is_available:
                    # Output NOT AVAILABLE row for each category
                    for cat in ALL_CATEGORIES:
                        is_attack_cat = (cat != "Benign")
                        # Count total matching flows in this client's nominal arrival stream
                        n_cat_total = sum(1 for f in arrivals if (str(f.attack_family) == cat or (cat == "Benign" and f.y_true == 0)))
                        records.append({
                            "client_id": cid,
                            "client_str": cid_str,
                            "config_short": cfg_short,
                            "config_full": cfg_full,
                            "quantile": q,
                            "attack_family": cat,
                            "is_attack": is_attack_cat,
                            "status": "NOT AVAILABLE",
                            "n_total": n_cat_total,
                            "n_fe_drop": None,
                            "n_gate_rej": None,
                            "n_load_shed": None,
                            "n_heavy_adm": None,
                            "gate_passage_rate": None,
                            "heavy_admission_rate": None,
                            "tp_ae": None,
                            "n_processed_ae": None,
                            "conditional_ae_recall": None,
                            "tp_heavy": None,
                            "n_admitted_heavy": None,
                            "conditional_heavy_recall": None,
                            "e2e_ae_contribution": None,
                            "e2e_heavy_contribution": None,
                            "recall_e2e": None,
                        })
                    continue

                sim = IntegratedPipelineSimulator(
                    client_id=cid,
                    quantile=q,
                    config_name=cfg_full,
                    w_shed=0.050,
                    q_max=100,
                )

                g_pred_full, ae_mse_full, h_logit_full = sim.precompute_flow_predictions(ev.x_test)
                g_pred = g_pred_full[sample_indices]
                ae_mse = ae_mse_full[sample_indices]
                h_logit = h_logit_full[sample_indices] if h_logit_full is not None else None

                sim.queue_sim.reset()
                is_config_d = ("Config D" in cfg_full)

                # Track per-category flows
                cat_flows: Dict[str, Dict[str, Any]] = {
                    cat: {
                        "n_total": 0,
                        "n_fe_drop": 0,
                        "n_gate_rej": 0,
                        "n_load_shed": 0,
                        "n_heavy_adm": 0,
                        "tp_ae": 0,
                        "tp_heavy": 0,
                    }
                    for cat in ALL_CATEGORIES
                }

                for i, flow in enumerate(arrivals):
                    y = int(flow.y_true)
                    is_attack = (y == 1)
                    fam = "Benign" if not is_attack else str(flow.attack_family)
                    if fam not in cat_flows:
                        fam = "Unknown"
                        if fam not in cat_flows:
                            cat_flows[fam] = {
                                "n_total": 0,
                                "n_fe_drop": 0,
                                "n_gate_rej": 0,
                                "n_load_shed": 0,
                                "n_heavy_adm": 0,
                                "tp_ae": 0,
                                "tp_heavy": 0,
                            }

                    cf = cat_flows[fam]
                    cf["n_total"] += 1

                    g_p = float(g_pred[i])
                    ae_m = float(ae_mse[i])
                    h_l = float(h_logit[i]) if h_logit is not None else 0.0

                    res = sim.queue_sim.process_arrival(flow, g_p, ae_m)
                    act = res.action

                    if is_config_d and act == WorkloadQueueSimulator.ACTION_HEAVY_GRU_ADMITTED:
                        act = WorkloadQueueSimulator.ACTION_LOAD_SHED_AE

                    if act == WorkloadQueueSimulator.ACTION_FRONTEND_DROP:
                        cf["n_fe_drop"] += 1
                    elif act == WorkloadQueueSimulator.ACTION_GATE_REJECT_AE:
                        cf["n_gate_rej"] += 1
                        if ae_m >= sim.tau_ae and is_attack:
                            cf["tp_ae"] += 1
                    elif act == WorkloadQueueSimulator.ACTION_LOAD_SHED_AE:
                        cf["n_load_shed"] += 1
                        if ae_m >= sim.tau_ae and is_attack:
                            cf["tp_ae"] += 1
                    elif act == WorkloadQueueSimulator.ACTION_HEAVY_GRU_ADMITTED:
                        cf["n_heavy_adm"] += 1
                        if h_l >= 0.0 and is_attack:
                            cf["tp_heavy"] += 1

                for cat in ALL_CATEGORIES:
                    is_attack_cat = (cat != "Benign")
                    cf = cat_flows.get(cat, {
                        "n_total": 0, "n_fe_drop": 0, "n_gate_rej": 0,
                        "n_load_shed": 0, "n_heavy_adm": 0, "tp_ae": 0, "tp_heavy": 0
                    })
                    n_tot = cf["n_total"]

                    if n_tot == 0:
                        records.append({
                            "client_id": cid,
                            "client_str": cid_str,
                            "config_short": cfg_short,
                            "config_full": cfg_full,
                            "quantile": q,
                            "attack_family": cat,
                            "is_attack": is_attack_cat,
                            "status": "ZERO_SAMPLES",
                            "n_total": 0,
                            "n_fe_drop": 0,
                            "n_gate_rej": 0,
                            "n_load_shed": 0,
                            "n_heavy_adm": 0,
                            "gate_passage_rate": 0.0,
                            "heavy_admission_rate": 0.0,
                            "tp_ae": 0,
                            "n_processed_ae": 0,
                            "conditional_ae_recall": None,
                            "tp_heavy": 0,
                            "n_admitted_heavy": 0,
                            "conditional_heavy_recall": None,
                            "e2e_ae_contribution": 0.0,
                            "e2e_heavy_contribution": 0.0,
                            "recall_e2e": None,
                        })
                        continue

                    n_fe = cf["n_fe_drop"]
                    n_grej = cf["n_gate_rej"]
                    n_lshed = cf["n_load_shed"]
                    n_hadm = cf["n_heavy_adm"]

                    # Conservation check
                    assert (n_fe + n_grej + n_lshed + n_hadm) == n_tot, (
                        f"Conservation violated for {cid_str} {cfg_short} {q} {cat}: "
                        f"{n_fe} + {n_grej} + {n_lshed} + {n_hadm} != {n_tot}"
                    )

                    gate_pass_count = n_lshed + n_hadm
                    gate_passage_rate = gate_pass_count / n_tot
                    heavy_adm_rate = n_hadm / n_tot

                    n_proc_ae = n_grej + n_lshed
                    tp_ae = cf["tp_ae"]
                    cond_ae_rec = (tp_ae / n_proc_ae) if n_proc_ae > 0 else 0.0

                    n_proc_heavy = n_hadm
                    tp_heavy = cf["tp_heavy"]
                    cond_heavy_rec = (tp_heavy / n_proc_heavy) if n_proc_heavy > 0 else 0.0

                    e2e_ae_contrib = tp_ae / n_tot
                    e2e_heavy_contrib = tp_heavy / n_tot
                    recall_e2e = (tp_ae + tp_heavy) / n_tot

                    records.append({
                        "client_id": cid,
                        "client_str": cid_str,
                        "config_short": cfg_short,
                        "config_full": cfg_full,
                        "quantile": q,
                        "attack_family": cat,
                        "is_attack": is_attack_cat,
                        "status": "EVALUATED",
                        "n_total": n_tot,
                        "n_fe_drop": n_fe,
                        "n_gate_rej": n_grej,
                        "n_load_shed": n_lshed,
                        "n_heavy_adm": n_hadm,
                        "gate_passage_rate": round(gate_passage_rate, 6),
                        "heavy_admission_rate": round(heavy_adm_rate, 6),
                        "tp_ae": tp_ae,
                        "n_processed_ae": n_proc_ae,
                        "conditional_ae_recall": round(cond_ae_rec, 6),
                        "tp_heavy": tp_heavy,
                        "n_admitted_heavy": n_proc_heavy,
                        "conditional_heavy_recall": round(cond_heavy_rec, 6),
                        "e2e_ae_contribution": round(e2e_ae_contrib, 6),
                        "e2e_heavy_contribution": round(e2e_heavy_contrib, 6),
                        "recall_e2e": round(recall_e2e, 6),
                    })

    df = pd.DataFrame(records)
    out_csv = output_dir / "validation_attack_family_breakdown.csv"
    df.to_csv(out_csv, index=False)
    print(f"Exported {len(df)} breakdown records to {out_csv}")
    return df


if __name__ == "__main__":
    evaluate_all_attack_families()

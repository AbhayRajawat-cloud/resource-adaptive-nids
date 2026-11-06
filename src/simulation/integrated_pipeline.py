"""
Stage 5: Phase 5.4 Integrated Pipeline Simulator & Pre-Admission Switch.

Implements the complete end-to-end edge inference pipeline combining:
1. Front-end Light Path socket-layer Token Bucket triage (Safeguard F2).
2. Private Local Gate Regressor anomaly screening (g_{phi_k}).
3. Queue Telemetry Workload Monitor with Active-Stall Residual Heuristic.
4. Pre-admission conjunctive dual-path routing switch (4 mutually exclusive outcomes):
     - ACTION_FRONTEND_DROP: Dropped at socket layer due to front-end CPU saturation.
     - ACTION_GATE_REJECT_AE: Gate anomaly error < tau_Gate -> routed to Light Path AE.
     - ACTION_LOAD_SHED_AE: Gate anomaly error >= tau_Gate, but queue delay > W_shed
       or buffer full (N_q >= Q_max) -> load shed to Light Path AE fallback.
     - ACTION_HEAVY_ADMITTED: Admitted to Heavy Path for high-assurance classification.
5. Light Path Autoencoder fallback screening (MSE >= tau_AE -> alert).
6. Heavy Path model classification (raw logit z >= 0 -> alert).
7. Unified security and routing accounting satisfying:
     N_total == N_fe_drop + N_gate_rej + N_load_shed + N_heavy_adm
     Recall_e2e = (TP_heavy + TP_ae_fallback) / Total_Attacks

Configurations Supported:
- Config A: HeavyGRU Standard FedAvg
- Config B: SupervisedDenseMLP Strategy B FedPer (Private Heads + FedBN Buffers)
- Config C: SupervisedDenseMLP Standard FedAvg (FedBN Buffers)
- Config D: Light Path Standalone Baseline (Private AE Only; Zero Heavy Capacity)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.models.autoencoder import Autoencoder
from src.models.gate import GatingRegressor
from src.models.heavy_models import HeavyGRU, SupervisedDenseMLP
from src.simulation.queue_simulator import (
    WorkloadQueueSimulator,
    FlowArrival,
    FlowRoutingResult,
    TokenBucketController,
)


class IntegratedPipelineSimulator:
    """
    End-to-End Integrated Pipeline Simulator combining Light Path screening,
    queue telemetry, dynamic pre-admission routing, and Heavy Path classification.
    """

    CONFIG_A = "Config A: HeavyGRU FedAvg"
    CONFIG_B = "Config B: DenseMLP FedPer"
    CONFIG_C = "Config C: DenseMLP FedAvg"
    CONFIG_D = "Config D: Light AE Standalone"

    def __init__(
        self,
        client_id: int,
        quantile: str = "Q99",
        config_name: str = "Config A: HeavyGRU FedAvg",
        w_shed: float = 0.050,                  # 50 ms SLA threshold
        q_max: int = 100,                       # Waiting buffer capacity (slots)
        s_heavy_bar: float = 0.0001799,         # Mean service time (0.1799 ms from Stage 4 profile)
        token_bucket_rate: float = 34943.4,     # Front-end triage capacity (flows/s)
        burst_capacity: float = 100.0,          # Matching buffer capacity
        stage2_dir: Path = Path("data/processed/alpha_0.5_seed_42"),
        stage3_ckpt_dir: Path = Path("outputs/checkpoints/stage3"),
        stage5_ckpt_dir: Path = Path("outputs/checkpoints/stage5/phase5.3"),
        thresholds_dir: Path = Path("outputs/experiments/stage3_threshold_sensitivity"),
    ):
        self.client_id = int(client_id)
        self.cid_str = f"client_{self.client_id:02d}"
        self.quantile = quantile.upper()
        self.config_name = config_name
        self.w_shed = float(w_shed)
        self.q_max = int(q_max)
        self.s_heavy_bar = float(s_heavy_bar)
        self.token_bucket_rate = float(token_bucket_rate)
        self.burst_capacity = float(burst_capacity)

        # 1. Load Client Local Thresholds (tau_Gate, tau_AE)
        thresh_json = thresholds_dir / "sensitivity_client_thresholds.json"
        if thresh_json.exists():
            with open(thresh_json, "r", encoding="utf-8") as f:
                all_thresh = json.load(f)
                client_entry = all_thresh[self.cid_str]
                if "thresholds" in client_entry and self.quantile in client_entry["thresholds"]:
                    c_thresh = client_entry["thresholds"][self.quantile]
                elif self.quantile in client_entry:
                    c_thresh = client_entry[self.quantile]
                else:
                    raise KeyError(f"Quantile '{self.quantile}' not found for {self.cid_str}")
                self.tau_gate = float(c_thresh["tau_gate"])
                self.tau_ae = float(c_thresh["tau_ae"])
        else:
            # Fallback to Stage 3 threshold json
            stage3_thresh_file = stage3_ckpt_dir / f"{self.cid_str}_thresholds.json"
            with open(stage3_thresh_file, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                self.tau_gate = float(t_data["tau_gate"])
                self.tau_ae = float(t_data["tau_ae"])

        # 2. Load Private Local Light Path Models (AE and Gate)
        ae_path = stage3_ckpt_dir / f"{self.cid_str}_autoencoder.pt"
        gate_path = stage3_ckpt_dir / f"{self.cid_str}_gate.pt"

        self.ae = Autoencoder(input_dim=38).eval()
        self.ae.load_state_dict(torch.load(ae_path, map_location="cpu"))

        self.gate = GatingRegressor(input_dim=38).eval()
        self.gate.load_state_dict(torch.load(gate_path, map_location="cpu"))

        # 3. Load Heavy Path Model based on Config Name
        self.heavy_model: Optional[nn.Module] = None
        self._load_heavy_model(stage5_ckpt_dir)

        # 4. Initialize Queue Simulator
        # For Config D (Light AE Standalone): Heavy Path capacity is zero
        effective_q_max = 0 if "Config D" in self.config_name else self.q_max

        self.queue_sim = WorkloadQueueSimulator(
            q_max=effective_q_max,
            w_shed=self.w_shed,
            s_gru_bar=self.s_heavy_bar,
            token_bucket_rate=self.token_bucket_rate,
            burst_capacity=self.burst_capacity,
            tau_gate=self.tau_gate,
            tau_ae=self.tau_ae,
            deterministic_service=False,
        )

    def _load_heavy_model(self, ckpt_dir: Path):
        """Load and configure the appropriate heavy model checkpoint."""
        if "Config D" in self.config_name:
            # Standalone AE baseline: zero Heavy Path capacity
            self.heavy_model = None
            return

        q_key = self.quantile  # "Q99" or "Q95"

        if "Config A" in self.config_name:
            # HeavyGRU Standard FedAvg
            ckpt_path = ckpt_dir / f"global_fedavg_gru_{q_key}_R50.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing Config A checkpoint: {ckpt_path}")
            model = HeavyGRU(input_dim=38, hidden_dim=64)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            self.heavy_model = model.eval()

        elif "Config B" in self.config_name:
            # SupervisedDenseMLP Strategy B FedPer (only Q99 trained)
            if q_key != "Q99":
                raise ValueError(f"Config B (FedPer) was trained exclusively for Q99; unavailable for {q_key}")
            body_path = ckpt_dir / f"fedper_mlp_Q99_R50_body.pt"
            heads_path = ckpt_dir / f"client00_09_fedper_mlp_Q99_heads.pt"
            if not body_path.exists() or not heads_path.exists():
                raise FileNotFoundError(f"Missing Config B checkpoints: {body_path} or {heads_path}")

            model = SupervisedDenseMLP(input_dim=38)
            full_state = model.state_dict()
            # (a) Load global shared body
            body_state = torch.load(body_path, map_location="cpu")
            for k, v in body_state.items():
                full_state[k] = v.clone()

            # (b) Load client private head and BN buffers
            heads_data = torch.load(heads_path, map_location="cpu")
            c_info = heads_data.get(self.cid_str, {})
            c_head = c_info.get("private_head", {})
            for hk, hv in c_head.items():
                full_state[hk] = hv.clone()
            if c_info.get("bn_buffers") is not None:
                for bk, bv in c_info["bn_buffers"].items():
                    full_state[bk] = bv.clone()

            model.load_state_dict(full_state)
            self.heavy_model = model.eval()

        elif "Config C" in self.config_name:
            # SupervisedDenseMLP Standard FedAvg + Client BN buffers
            ckpt_path = ckpt_dir / f"global_fedavg_mlp_{q_key}_R50.pt"
            bn_path = ckpt_dir / f"global_fedavg_mlp_{q_key}_R50_client_bn.pt"
            if not ckpt_path.exists() or not bn_path.exists():
                raise FileNotFoundError(f"Missing Config C checkpoints: {ckpt_path} or {bn_path}")

            model = SupervisedDenseMLP(input_dim=38)
            full_state = model.state_dict()
            # (a) Load global shared weights
            global_state = torch.load(ckpt_path, map_location="cpu")
            for k, v in global_state.items():
                full_state[k] = v.clone()
            # (b) Load client local BN buffers
            bn_data = torch.load(bn_path, map_location="cpu")
            c_bn = bn_data.get(self.cid_str, {})
            for bk, bv in c_bn.items():
                full_state[bk] = bv.clone()

            model.load_state_dict(full_state)
            self.heavy_model = model.eval()

        else:
            raise ValueError(f"Unknown config_name: {self.config_name}")

    def precompute_flow_predictions(
        self,
        x_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Precompute Gate error, AE MSE, and Heavy logits for an array of 38-D feature vectors.

        Returns:
            gate_preds: (N,) np.ndarray
            ae_mses: (N,) np.ndarray
            heavy_logits: (N,) np.ndarray or None (if Config D)
        """
        n_flows = len(x_array)
        t_x = torch.tensor(x_array, dtype=torch.float32)

        with torch.no_grad():
            # 1. Gate anomaly score
            gate_preds = self.gate(t_x).squeeze(-1).numpy()

            # 2. AE reconstruction MSE
            recon = self.ae(t_x)
            ae_mses = Autoencoder.compute_mse(t_x, recon).numpy()

            # 3. Heavy path logits
            if self.heavy_model is not None:
                if isinstance(self.heavy_model, HeavyGRU):
                    # W=1 single flow formulated as 3D (N, 1, 38)
                    t_x_gru = t_x.unsqueeze(1)
                    heavy_logits = self.heavy_model(t_x_gru).numpy()
                else:
                    # MLP: 2D (N, 38)
                    heavy_logits = self.heavy_model(t_x).numpy()
            else:
                heavy_logits = None

        return gate_preds, ae_mses, heavy_logits

    def run_simulation(
        self,
        arrivals: List[FlowArrival],
        precomputed_gate_preds: Optional[np.ndarray] = None,
        precomputed_ae_mses: Optional[np.ndarray] = None,
        precomputed_heavy_logits: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end simulation across an arrival stream.

        Returns a comprehensive dictionary containing exact routing counts,
        flow conservation verification, and combined end-to-end security metrics.
        """
        self.queue_sim.reset()
        n_total = len(arrivals)

        # Detailed tracking records
        routing_actions = []
        is_attack_list = []
        is_alert_list = []
        continuous_scores = []
        waiting_times = []

        # Classification & routing breakdown counters
        counts = {
            "n_total": n_total,
            "frontend_drop": 0,
            "gate_reject_ae": 0,
            "load_shed_ae": 0,
            "heavy_admitted": 0,
            "attacks_total": 0,
            "benign_total": 0,
            "attacks_fe_drop": 0,
            "attacks_gate_rej": 0,
            "attacks_load_shed": 0,
            "attacks_heavy_adm": 0,
            "benign_fe_drop": 0,
            "benign_gate_rej": 0,
            "benign_load_shed": 0,
            "benign_heavy_adm": 0,
            "tp_heavy": 0,
            "tp_ae_gate_rej": 0,
            "tp_ae_load_shed": 0,
            "fp_heavy": 0,
            "fp_ae_gate_rej": 0,
            "fp_ae_load_shed": 0,
        }

        is_config_d = ("Config D" in self.config_name)

        for i, flow in enumerate(arrivals):
            y = int(flow.y_true)
            is_attack = bool(y == 1)
            is_attack_list.append(y)

            if is_attack:
                counts["attacks_total"] += 1
            else:
                counts["benign_total"] += 1

            # Fetch precomputed or flow-attached scores
            g_pred = precomputed_gate_preds[i] if precomputed_gate_preds is not None else flow.precomputed_gate_pred
            ae_mse = precomputed_ae_mses[i] if precomputed_ae_mses is not None else flow.precomputed_ae_mse
            h_logit = precomputed_heavy_logits[i] if precomputed_heavy_logits is not None else None

            # Process arrival through Queue Simulator
            res = self.queue_sim.process_arrival(
                flow=flow,
                gate_pred=float(g_pred) if g_pred is not None else 0.0,
                ae_mse=float(ae_mse) if ae_mse is not None else 0.0,
            )

            # In Config D: Heavy Path capacity is zero -> any heavy admission redirected to load-shed AE
            action = res.action
            if is_config_d and action == WorkloadQueueSimulator.ACTION_HEAVY_GRU_ADMITTED:
                action = WorkloadQueueSimulator.ACTION_LOAD_SHED_AE

            routing_actions.append(action)
            waiting_times.append(res.actual_waiting_time)

            # -------------------------------------------------------------
            # Combined Security Accounting across the 4 Terminal Outcomes
            # -------------------------------------------------------------
            alerted = False
            score = 0.0

            if action == WorkloadQueueSimulator.ACTION_FRONTEND_DROP:
                counts["frontend_drop"] += 1
                if is_attack:
                    counts["attacks_fe_drop"] += 1
                else:
                    counts["benign_fe_drop"] += 1
                # Frontend drop counts as undetected (score 0.0)
                alerted = False
                score = 0.0

            elif action == WorkloadQueueSimulator.ACTION_GATE_REJECT_AE:
                counts["gate_reject_ae"] += 1
                if is_attack:
                    counts["attacks_gate_rej"] += 1
                else:
                    counts["benign_gate_rej"] += 1

                # Light Path Autoencoder fallback screening
                # Alert if MSE >= tau_AE
                alerted = bool(ae_mse >= self.tau_ae)
                # Map continuous score so that score >= 0.5 iff MSE >= tau_ae
                # Scaled logistic: score = sigmoid((MSE - tau_ae) / tau_ae * 5.0)
                rel_diff = (float(ae_mse) - self.tau_ae) / (self.tau_ae if self.tau_ae > 0 else 1.0)
                score = 1.0 / (1.0 + np.exp(-np.clip(rel_diff * 5.0, -10.0, 10.0)))

                if alerted:
                    if is_attack:
                        counts["tp_ae_gate_rej"] += 1
                    else:
                        counts["fp_ae_gate_rej"] += 1

            elif action == WorkloadQueueSimulator.ACTION_LOAD_SHED_AE:
                counts["load_shed_ae"] += 1
                if is_attack:
                    counts["attacks_load_shed"] += 1
                else:
                    counts["benign_load_shed"] += 1

                # Light Path Autoencoder fallback screening
                alerted = bool(ae_mse >= self.tau_ae)
                rel_diff = (float(ae_mse) - self.tau_ae) / (self.tau_ae if self.tau_ae > 0 else 1.0)
                score = 1.0 / (1.0 + np.exp(-np.clip(rel_diff * 5.0, -10.0, 10.0)))

                if alerted:
                    if is_attack:
                        counts["tp_ae_load_shed"] += 1
                    else:
                        counts["fp_ae_load_shed"] += 1

            elif action == WorkloadQueueSimulator.ACTION_HEAVY_GRU_ADMITTED:
                counts["heavy_admitted"] += 1
                if is_attack:
                    counts["attacks_heavy_adm"] += 1
                else:
                    counts["benign_heavy_adm"] += 1

                # Heavy Path Model Classification
                if h_logit is not None:
                    # Alert if raw logit z >= 0 (sigmoid probability >= 0.5)
                    alerted = bool(h_logit >= 0.0)
                    score = float(1.0 / (1.0 + np.exp(-np.clip(h_logit, -20.0, 20.0))))
                else:
                    alerted = False
                    score = 0.0

                if alerted:
                    if is_attack:
                        counts["tp_heavy"] += 1
                    else:
                        counts["fp_heavy"] += 1

            is_alert_list.append(int(alerted))
            continuous_scores.append(float(score))

        # -------------------------------------------------------------
        # Invariant V4 & Flow Conservation Identity Verification
        # -------------------------------------------------------------
        sum_routing = (
            counts["frontend_drop"]
            + counts["gate_reject_ae"]
            + counts["load_shed_ae"]
            + counts["heavy_admitted"]
        )
        assert sum_routing == n_total, (
            f"Flow conservation violation: N_total={n_total} != sum_routing={sum_routing}"
        )

        # -------------------------------------------------------------
        # Combined Classification Metrics
        # -------------------------------------------------------------
        y_true_arr = np.array(is_attack_list, dtype=int)
        y_pred_arr = np.array(is_alert_list, dtype=int)
        scores_arr = np.array(continuous_scores, dtype=float)

        total_attacks = counts["attacks_total"]
        total_benign = counts["benign_total"]

        tp_total = counts["tp_heavy"] + counts["tp_ae_gate_rej"] + counts["tp_ae_load_shed"]
        fp_total = counts["fp_heavy"] + counts["fp_ae_gate_rej"] + counts["fp_ae_load_shed"]
        fn_total = total_attacks - tp_total
        tn_total = total_benign - fp_total

        # Combined Recall & FPR
        recall_e2e = float(tp_total / total_attacks) if total_attacks > 0 else 0.0
        fpr_e2e = float(fp_total / total_benign) if total_benign > 0 else 0.0
        prec_e2e = float(tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
        f1_e2e = (
            float(2.0 * prec_e2e * recall_e2e / (prec_e2e + recall_e2e))
            if (prec_e2e + recall_e2e) > 0
            else 0.0
        )
        acc_e2e = float((tp_total + tn_total) / n_total) if n_total > 0 else 0.0

        # Continuous AUC metrics
        if total_attacks > 0 and total_benign > 0:
            roc_auc = float(roc_auc_score(y_true_arr, scores_arr))
            pr_auc = float(average_precision_score(y_true_arr, scores_arr))
        else:
            roc_auc = None
            pr_auc = None

        # Queue Telemetry & Latencies
        q_admitted_wait = self.queue_sim.waiting_times_admitted
        mean_wait_s = float(np.mean(q_admitted_wait)) if len(q_admitted_wait) > 0 else 0.0
        p99_wait_s = float(np.percentile(q_admitted_wait, 99.0)) if len(q_admitted_wait) > 0 else 0.0
        q_lens = self.queue_sim.queue_lengths_at_arrival
        mean_q_len = float(np.mean(q_lens)) if len(q_lens) > 0 else 0.0

        return {
            "client_id": self.client_id,
            "client_str": self.cid_str,
            "quantile": self.quantile,
            "config_name": self.config_name,
            "w_shed_seconds": self.w_shed,
            "q_max": self.q_max,
            "flow_counts": counts,
            "routing_proportions": {
                "r_frontend_drop": counts["frontend_drop"] / n_total if n_total > 0 else 0.0,
                "r_gate_reject_ae": counts["gate_reject_ae"] / n_total if n_total > 0 else 0.0,
                "r_load_shed_ae": counts["load_shed_ae"] / n_total if n_total > 0 else 0.0,
                "r_heavy_admitted": counts["heavy_admitted"] / n_total if n_total > 0 else 0.0,
            },
            "security_metrics": {
                "total_flows": n_total,
                "total_attacks": total_attacks,
                "total_benign": total_benign,
                "tp": tp_total,
                "fp": fp_total,
                "tn": tn_total,
                "fn": fn_total,
                "recall_e2e": recall_e2e,
                "fpr_e2e": fpr_e2e,
                "precision": prec_e2e,
                "f1_score": f1_e2e,
                "accuracy": acc_e2e,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            },
            "queue_telemetry": {
                "mean_waiting_time_ms": mean_wait_s * 1000.0,
                "p99_waiting_time_ms": p99_wait_s * 1000.0,
                "mean_queue_length": mean_q_len,
                "total_heavy_admitted": counts["heavy_admitted"],
            },
        }

"""
Stage 4: Workload-based Queue Simulator & Dynamic Routing Switch Engine.

Implements:
1. TokenBucketController: Front-end Light Path socket-layer rate limiter enforcing
   triage capacity with continuous replenishment and diagnostic EWMA rate tracking.
2. WorkloadQueueSimulator: FIFO edge queue simulator implementing:
   - Admitted-flow Lindley workload recursions (Eq. 6).
   - Telemetry workload estimator with Active-Stall Residual Heuristic.
   - Pre-admission dual-path conjunctive routing decision logic.
   - Exact 4-outcome mutually exclusive flow accounting and security tracking.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class FlowArrival:
    """Represents an incoming network flow arrival at the edge router."""
    flow_id: int
    arrival_time: float      # Arrival timestamp t_{k, i} in seconds
    x: np.ndarray            # 38-D feature vector
    y_true: int              # Ground truth label (0=Benign, 1=Attack)
    attack_family: str       # Attack category name
    service_time: float      # Profiling service time S_{GRU, k, i} in seconds
    precomputed_gate_pred: Optional[float] = None
    precomputed_ae_mse: Optional[float] = None


@dataclass
class FlowRoutingResult:
    """Outcome and latency telemetry for an individual flow."""
    flow_id: int
    arrival_time: float
    y_true: int
    attack_family: str
    action: str              # One of the 4 mutually exclusive terminal outcomes
    gate_prediction: float
    gate_trigger: bool
    estimated_delay_w_pre: float
    actual_waiting_time: float
    actual_service_time: float
    departure_time: float
    ae_mse: Optional[float] = None
    ae_detected: Optional[bool] = None


class TokenBucketController:
    """
    Front-end Light Path Triage Token Bucket Controller (Safeguard F2).
    
    Enforces maximum front-end socket ingestion capacity:
      lambda_light_triage_max = (1 - epsilon_cpu) / (E[L_Gate] + E[L_AE])
    Continuous token replenishment:
      tokens(t_i^-) = min(beta_burst, tokens(t_{i-1}^+) + r * (t_i - t_{i-1}))
    Maintains independent diagnostic EWMA inter-arrival estimator:
      A_hat_i = (1 - alpha) * A_hat_{i-1} + alpha * delta_t_i
      lambda_hat_i = 1.0 / A_hat_i
    """
    def __init__(
        self,
        rate_r: float = 34943.4,     # flows/second
        burst_capacity: float = 50.0, # tokens
        ewma_alpha: float = 0.05,
        initial_tokens: Optional[float] = None,
    ):
        self.rate_r = float(rate_r)
        self.burst_capacity = float(burst_capacity)
        self.ewma_alpha = float(ewma_alpha)

        self.tokens = float(burst_capacity if initial_tokens is None else initial_tokens)
        self.last_update_time = 0.0
        self.smoothed_inter_arrival = 1.0 / self.rate_r if self.rate_r > 0 else 0.001
        self.total_checked = 0
        self.total_dropped = 0

    def reset(self, start_time: float = 0.0):
        self.tokens = self.burst_capacity
        self.last_update_time = start_time
        self.smoothed_inter_arrival = 1.0 / self.rate_r if self.rate_r > 0 else 0.001
        self.total_checked = 0
        self.total_dropped = 0

    def try_consume(self, arrival_time: float) -> Tuple[bool, float, float]:
        """
        Attempt to consume 1 token for an incoming flow.
        
        Returns:
            admitted: bool (True if token available, False if dropped)
            current_tokens: float
            estimated_arrival_rate: float (from diagnostic EWMA)
        """
        self.total_checked += 1
        delta_t = max(0.0, arrival_time - self.last_update_time)

        # 1. Continuous token replenishment
        replenished = self.tokens + self.rate_r * delta_t
        self.tokens = min(self.burst_capacity, replenished)
        self.last_update_time = arrival_time

        # 2. Independent Diagnostic EWMA Inter-Arrival Estimator
        if delta_t > 0:
            self.smoothed_inter_arrival = (
                (1.0 - self.ewma_alpha) * self.smoothed_inter_arrival
                + self.ewma_alpha * delta_t
            )
        estimated_lambda = 1.0 / self.smoothed_inter_arrival if self.smoothed_inter_arrival > 0 else self.rate_r

        # 3. Token check
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, self.tokens, estimated_lambda
        else:
            self.total_dropped += 1
            return False, self.tokens, estimated_lambda


class WorkloadQueueSimulator:
    """
    Event-driven FIFO Workload Queue Simulator and Dynamic Pre-Admission Switch.
    
    Implements:
    - Waiting buffer capacity Q_max = B, total system capacity K = Q_max + 1.
    - Admitted-flow Lindley workload recursions:
        W_{k, i}^- = max(0, W_{k, i-1}^- + a_{k, i-1} S_{GRU, k, i-1} - (t_{k, i} - t_{k, i-1}))
    - Runtime Telemetry Estimator with Active-Stall Residual Heuristic:
        W_hat^{pre}(t_i^-) = b_k * max(0, S_bar - E_k) + N_q * S_bar
        Fallback: if E_k > S_bar, W_hat^{upper} = [b_k + N_q] * S_bar
    - 4 mutually exclusive terminal outcomes per arrival:
        1. ACTION_FRONTEND_DROP
        2. ACTION_GATE_REJECT_AE
        3. ACTION_LOAD_SHED_AE
        4. ACTION_HEAVY_GRU_ADMITTED
    """

    ACTION_FRONTEND_DROP = "ACTION_FRONTEND_DROP"
    ACTION_GATE_REJECT_AE = "ACTION_GATE_REJECT_AE"
    ACTION_LOAD_SHED_AE = "ACTION_LOAD_SHED_AE"
    ACTION_HEAVY_GRU_ADMITTED = "ACTION_HEAVY_GRU_ADMITTED"

    def __init__(
        self,
        q_max: int = 50,                        # Buffer slots (B), excluding worker
        w_shed: float = 0.050,                  # Delay threshold in seconds (50 ms)
        s_gru_bar: float = 0.0001799,           # Mean service time in seconds (0.1799 ms)
        token_bucket_rate: float = 34943.4,     # Front-end triage capacity (flows/s)
        burst_capacity: float = 50.0,
        tau_gate: float = -4.031378,
        tau_ae: float = 0.017162,
        deterministic_service: bool = False,
    ):
        self.q_max = int(q_max)
        self.system_capacity_k = self.q_max + 1
        self.w_shed = float(w_shed)
        self.s_gru_bar = float(s_gru_bar)
        self.tau_gate = float(tau_gate)
        self.tau_ae = float(tau_ae)
        self.deterministic_service = deterministic_service

        self.token_bucket = TokenBucketController(
            rate_r=token_bucket_rate,
            burst_capacity=burst_capacity,
        )

        self.reset()

    def reset(self):
        """Reset queue state and statistics."""
        self.token_bucket.reset()

        # State variables
        self.current_time = 0.0
        self.lindley_workload = 0.0    # W_{i}^- [s]
        self.last_arrival_time = 0.0

        # Physical FIFO Queue structures:
        # waiting_queue contains tuples: (flow_id, arrival_time, service_time, y_true, attack_family)
        self.waiting_queue = deque()
        self.active_job: Optional[Dict[str, Any]] = None  # in-service job details

        # Flow counters for the 4 mutually exclusive terminal outcomes
        self.n_total = 0
        self.n_frontend_drop = 0
        self.n_gate_reject_ae = 0
        self.n_load_shed_ae = 0
        self.n_heavy_admitted = 0

        # Sub-counters by class (Benign=0, Attack=1)
        self.counts_by_class = {
            0: {"total": 0, "frontend_drop": 0, "gate_reject_ae": 0, "load_shed_ae": 0, "heavy_admitted": 0},
            1: {"total": 0, "frontend_drop": 0, "gate_reject_ae": 0, "load_shed_ae": 0, "heavy_admitted": 0},
        }

        # AE detection on Light Path flows (fallback tracking)
        self.ae_detections = {
            "gate_reject_ae": {"benign": 0, "attack": 0},
            "load_shed_ae": {"benign": 0, "attack": 0},
        }

        # Telemetry recording
        self.records: List[FlowRoutingResult] = []
        self.waiting_times_admitted: List[float] = []
        self.queue_lengths_at_arrival: List[int] = []

    def _advance_service_to(self, target_time: float):
        """
        Advance the discrete server to target_time, completing in-service and queued jobs FIFO.
        """
        while self.active_job is not None:
            departure_time = self.active_job["departure_time"]
            if departure_time <= target_time:
                # Active job completes
                self.current_time = departure_time
                if len(self.waiting_queue) > 0:
                    # Dequeue next waiting job into service
                    next_flow = self.waiting_queue.popleft()
                    w_time = departure_time - next_flow["arrival_time"]
                    s_time = next_flow["service_time"]
                    self.active_job = {
                        "flow_id": next_flow["flow_id"],
                        "start_time": departure_time,
                        "service_time": s_time,
                        "departure_time": departure_time + s_time,
                        "waiting_time": w_time,
                    }
                else:
                    self.active_job = None
            else:
                # Active job is still in service at target_time
                break

        self.current_time = target_time

    def estimate_pre_admission_delay(self, arrival_time: float) -> Tuple[float, float, int, int]:
        """
        Compute runtime workload estimator with Active-Stall Residual Heuristic:
          W_hat^{pre}(t_i^-) = b_k * max(0, S_bar - E_k) + N_q * S_bar
        Fallback: if E_k > S_bar, W_hat^{upper} = [b_k + N_q] * S_bar
        
        Returns:
            w_hat_pre: float (estimated queue delay in seconds)
            elapsed_time: float (E_k in seconds)
            b_server: int (1 if busy, 0 if idle)
            n_queue: int (current waiting queue length)
        """
        b_server = 1 if self.active_job is not None else 0
        n_queue = len(self.waiting_queue)

        if b_server == 1:
            elapsed_time = max(0.0, arrival_time - self.active_job["start_time"])
            if elapsed_time > self.s_gru_bar:
                # Active-Stall Residual Heuristic fallback
                w_hat_pre = (b_server + n_queue) * self.s_gru_bar
            else:
                remaining_active = max(0.0, self.s_gru_bar - elapsed_time)
                w_hat_pre = remaining_active + (n_queue * self.s_gru_bar)
        else:
            elapsed_time = 0.0
            w_hat_pre = 0.0

        return w_hat_pre, elapsed_time, b_server, n_queue

    def process_arrival(
        self,
        flow: FlowArrival,
        gate_pred: Optional[float] = None,
        ae_mse: Optional[float] = None,
    ) -> FlowRoutingResult:
        """
        Process a single flow arrival through the conjunctive dynamic routing switch.
        """
        if gate_pred is None:
            gate_pred = flow.precomputed_gate_pred if flow.precomputed_gate_pred is not None else 0.0
        if ae_mse is None:
            ae_mse = flow.precomputed_ae_mse

        t_arr = flow.arrival_time
        y = int(flow.y_true)
        s_time = self.s_gru_bar if self.deterministic_service else flow.service_time

        # 1. Advance server state up to arrival time t_arr
        self._advance_service_to(t_arr)

        # 2. Update Admitted-Flow Lindley Workload Recursion (Eq. 6)
        delta_t_arrival = max(0.0, t_arr - self.last_arrival_time) if self.n_total > 0 else 0.0
        # W_{i}^- = max(0, W_{i-1}^- - delta_t)  (service term a_{i-1}*S_{i-1} was added at admission)
        self.lindley_workload = max(0.0, self.lindley_workload - delta_t_arrival)
        self.last_arrival_time = t_arr

        self.n_total += 1
        self.counts_by_class[y]["total"] += 1

        # 3. Check Front-End Light Path Triage Token Bucket Controller (Safeguard F2)
        admitted_frontend, _, _ = self.token_bucket.try_consume(t_arr)
        if not admitted_frontend:
            # Action 1: Dropped at socket layer due to front-end CPU saturation
            action = self.ACTION_FRONTEND_DROP
            self.n_frontend_drop += 1
            self.counts_by_class[y]["frontend_drop"] += 1

            result = FlowRoutingResult(
                flow_id=flow.flow_id,
                arrival_time=t_arr,
                y_true=y,
                attack_family=flow.attack_family,
                action=action,
                gate_prediction=gate_pred,
                gate_trigger=False,
                estimated_delay_w_pre=0.0,
                actual_waiting_time=0.0,
                actual_service_time=0.0,
                departure_time=t_arr,
                ae_mse=ae_mse,
                ae_detected=False,
            )
            self.records.append(result)
            return result

        # 4. Evaluate Gate Anomaly Score
        gate_trigger = bool(gate_pred >= self.tau_gate)

        # 5. Compute Runtime Workload Telemetry & Active-Stall Heuristic
        w_hat_pre, elapsed_k, b_server, n_queue = self.estimate_pre_admission_delay(t_arr)
        self.queue_lengths_at_arrival.append(n_queue)

        # 6. Pre-Admission Conjunctive Routing Decision
        if not gate_trigger:
            # Action 2: Gate rejection -> triaged directly to Light Path AE
            action = self.ACTION_GATE_REJECT_AE
            self.n_gate_reject_ae += 1
            self.counts_by_class[y]["gate_reject_ae"] += 1
            a_k_i = 0

        elif gate_trigger and (w_hat_pre > self.w_shed or n_queue >= self.q_max):
            # Action 3: Load shedding -> delay limit exceeded OR waiting buffer full -> Light Path AE fallback
            action = self.ACTION_LOAD_SHED_AE
            self.n_load_shed_ae += 1
            self.counts_by_class[y]["load_shed_ae"] += 1
            a_k_i = 0

        else:
            # Action 4: Heavy Path Admission (gate_trigger and w_hat_pre <= w_shed and n_queue < Q_max)
            action = self.ACTION_HEAVY_GRU_ADMITTED
            self.n_heavy_admitted += 1
            self.counts_by_class[y]["heavy_admitted"] += 1
            a_k_i = 1

        # 7. Execute routing action
        actual_wait = 0.0
        departure_t = t_arr
        actual_svc = 0.0

        if a_k_i == 1:
            # Flow admitted to Heavy GRU Path
            actual_svc = s_time
            # Update Lindley recursion: add admitted flow's service time
            self.lindley_workload += actual_svc

            if self.active_job is None:
                # Server is idle -> job enters service immediately
                actual_wait = 0.0
                departure_t = t_arr + actual_svc
                self.active_job = {
                    "flow_id": flow.flow_id,
                    "start_time": t_arr,
                    "service_time": actual_svc,
                    "departure_time": departure_t,
                    "waiting_time": 0.0,
                }
            else:
                # Server is busy -> job joins FIFO waiting buffer
                # Job will start after the last waiting job completes
                if len(self.waiting_queue) == 0:
                    projected_start = self.active_job["departure_time"]
                else:
                    last_queued = self.waiting_queue[-1]
                    projected_start = last_queued["projected_departure"]

                actual_wait = projected_start - t_arr
                departure_t = projected_start + actual_svc
                self.waiting_queue.append({
                    "flow_id": flow.flow_id,
                    "arrival_time": t_arr,
                    "service_time": actual_svc,
                    "projected_start": projected_start,
                    "projected_departure": departure_t,
                    "y_true": y,
                    "attack_family": flow.attack_family,
                })

            self.waiting_times_admitted.append(actual_wait)

        # 8. Evaluate Light Path AE Fallback Detection
        ae_detected = None
        if action in [self.ACTION_GATE_REJECT_AE, self.ACTION_LOAD_SHED_AE]:
            if ae_mse is not None:
                ae_detected = bool(ae_mse >= self.tau_ae)
                class_key = "attack" if y == 1 else "benign"
                path_key = "gate_reject_ae" if action == self.ACTION_GATE_REJECT_AE else "load_shed_ae"
                if ae_detected:
                    self.ae_detections[path_key][class_key] += 1

        result = FlowRoutingResult(
            flow_id=flow.flow_id,
            arrival_time=t_arr,
            y_true=y,
            attack_family=flow.attack_family,
            action=action,
            gate_prediction=gate_pred,
            gate_trigger=gate_trigger,
            estimated_delay_w_pre=w_hat_pre,
            actual_waiting_time=actual_wait,
            actual_service_time=actual_svc,
            departure_time=departure_t,
            ae_mse=ae_mse,
            ae_detected=ae_detected,
        )
        self.records.append(result)
        return result

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Compute flow conservation, queueing telemetry, and decoupled security metrics.
        """
        # Flow Conservation Invariant check
        n_sum = self.n_frontend_drop + self.n_gate_reject_ae + self.n_load_shed_ae + self.n_heavy_admitted
        assert n_sum == self.n_total, f"Flow conservation identity violated: {n_sum} != {self.n_total}"

        # Arrival breakdown
        n_attack_tot = self.counts_by_class[1]["total"]
        n_benign_tot = self.counts_by_class[0]["total"]

        # 1. Effective Heavy Path Attack Passage Recall
        heavy_attacks_admitted = self.counts_by_class[1]["heavy_admitted"]
        heavy_benign_admitted = self.counts_by_class[0]["heavy_admitted"]
        tpr_heavy_eff = float(heavy_attacks_admitted / n_attack_tot) if n_attack_tot > 0 else 0.0
        fpr_heavy_eff = float(heavy_benign_admitted / n_benign_tot) if n_benign_tot > 0 else 0.0

        # 2. Stage 3 AE Fallback Recall on Light Path flows
        ae_attacks_detected = (
            self.ae_detections["gate_reject_ae"]["attack"] +
            self.ae_detections["load_shed_ae"]["attack"]
        )
        ae_benign_false_alarms = (
            self.ae_detections["gate_reject_ae"]["benign"] +
            self.ae_detections["load_shed_ae"]["benign"]
        )
        tpr_ae_fallback = float(ae_attacks_detected / n_attack_tot) if n_attack_tot > 0 else 0.0
        fpr_ae_fallback = float(ae_benign_false_alarms / n_benign_tot) if n_benign_tot > 0 else 0.0

        # 3. Ideal Routing Yield (assuming perfect downstream classification for heavy flows)
        tpr_ideal_yield = float((heavy_attacks_admitted + ae_attacks_detected) / n_attack_tot) if n_attack_tot > 0 else 0.0

        # Queue Delay Statistics for Admitted Flows
        if len(self.waiting_times_admitted) > 0:
            w_arr = np.array(self.waiting_times_admitted)
            w_mean_s = float(np.mean(w_arr))
            w_median_s = float(np.median(w_arr))
            w_p95_s = float(np.percentile(w_arr, 95.0))
            w_p99_s = float(np.percentile(w_arr, 99.0))
            w_max_s = float(np.max(w_arr))
        else:
            w_mean_s = w_median_s = w_p95_s = w_p99_s = w_max_s = 0.0

        # Queue Length Statistics at Arrival
        if len(self.queue_lengths_at_arrival) > 0:
            q_arr = np.array(self.queue_lengths_at_arrival)
            q_mean = float(np.mean(q_arr))
            q_p95 = float(np.percentile(q_arr, 95.0))
            q_max = int(np.max(q_arr))
        else:
            q_mean = q_p95 = 0.0
            q_max = 0

        # Throughput and Offered Load
        sim_duration = max(0.001, self.current_time)
        lambda_arrival = float(self.n_total / sim_duration)
        lambda_admitted = float(self.n_heavy_admitted / sim_duration)
        # Offered utilization to the heavy server
        rho_offered = (lambda_arrival * self.s_gru_bar)

        return {
            "flow_counts": {
                "total": self.n_total,
                "frontend_drop": self.n_frontend_drop,
                "gate_reject_ae": self.n_gate_reject_ae,
                "load_shed_ae": self.n_load_shed_ae,
                "heavy_admitted": self.n_heavy_admitted,
                "conservation_verified": (n_sum == self.n_total),
            },
            "routing_fractions_pct": {
                "frontend_drop_pct": float(self.n_frontend_drop / self.n_total * 100.0) if self.n_total > 0 else 0.0,
                "gate_reject_ae_pct": float(self.n_gate_reject_ae / self.n_total * 100.0) if self.n_total > 0 else 0.0,
                "load_shed_ae_pct": float(self.n_load_shed_ae / self.n_total * 100.0) if self.n_total > 0 else 0.0,
                "heavy_admitted_pct": float(self.n_heavy_admitted / self.n_total * 100.0) if self.n_total > 0 else 0.0,
            },
            "security_accounting": {
                "total_attacks": n_attack_tot,
                "total_benign": n_benign_tot,
                "heavy_attacks_admitted": heavy_attacks_admitted,
                "heavy_benign_admitted": heavy_benign_admitted,
                "effective_heavy_passage_recall": tpr_heavy_eff,
                "effective_heavy_benign_fpr": fpr_heavy_eff,
                "ae_fallback_attacks_detected": ae_attacks_detected,
                "ae_fallback_benign_false_alarms": ae_benign_false_alarms,
                "ae_fallback_recall": tpr_ae_fallback,
                "ae_fallback_fpr": fpr_ae_fallback,
                "ideal_routing_yield": tpr_ideal_yield,
            },
            "queue_telemetry": {
                "mean_waiting_time_s": w_mean_s,
                "mean_waiting_time_ms": w_mean_s * 1000.0,
                "median_waiting_time_ms": w_median_s * 1000.0,
                "p95_waiting_time_ms": w_p95_s * 1000.0,
                "p99_waiting_time_ms": w_p99_s * 1000.0,
                "max_waiting_time_ms": w_max_s * 1000.0,
                "mean_queue_length": q_mean,
                "p95_queue_length": q_p95,
                "max_queue_length": q_max,
                "buffer_capacity_Q_max": self.q_max,
                "system_capacity_K": self.system_capacity_k,
                "W_shed_s": self.w_shed,
                "W_shed_ms": self.w_shed * 1000.0,
                "lambda_arrival_fps": lambda_arrival,
                "lambda_admitted_fps": lambda_admitted,
                "rho_offered": rho_offered,
                "sim_duration_s": sim_duration,
            },
        }

"""
Stage 4: Stochastic Arrival Generators & Client Traffic Trace Sampler.

Implements:
1. PoissonTrafficGenerator: Inter-arrivals Delta_t ~ Exp(lambda).
2. MMPP2TrafficGenerator: Two-state Markov Modulated Poisson Process with states (lambda_1, lambda_2)
   and transition matrix R.
3. ParetoTrafficGenerator: Heavy-tailed Pareto arrivals (alpha=1.2). Explicitly documents
   infinite theoretical variance (sigma^2 = infinity) and computes finite empirical SCV c_a^2.
4. ClientTrafficTraceSampler: Samples 38-D flows with replacement from client test splits (data.npz),
   yielding arrival streams coupled with frozen Gate anomaly scores and AE reconstruction errors.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from src.models.autoencoder import Autoencoder
from src.models.gate import GatingRegressor
from src.simulation.queue_simulator import FlowArrival


class PoissonTrafficGenerator:
    """Generates homogeneous Poisson process arrivals with rate lambda."""

    def __init__(self, rate: float = 100.0, seed: int = 42):
        self.rate = float(rate)
        self.rng = np.random.default_rng(seed)

    def generate_inter_arrivals(self, n_samples: int) -> np.ndarray:
        """Delta_t ~ Exp(rate) = -ln(U) / rate."""
        if self.rate <= 0:
            raise ValueError("Poisson rate must be strictly positive")
        return self.rng.exponential(scale=1.0 / self.rate, size=n_samples)

    def generate_timestamps(self, n_samples: int, t_start: float = 0.0) -> np.ndarray:
        dt = self.generate_inter_arrivals(n_samples)
        return t_start + np.cumsum(dt)


class MMPP2TrafficGenerator:
    """
    Two-state Markov Modulated Poisson Process (MMPP(2)).
    
    States:
      State 1 (Normal / Background): rate lambda_1
      State 2 (Burst / Surge): rate lambda_2
    Transition rates:
      q_12: transition rate from state 1 -> 2 (/s)
      q_21: transition rate from state 2 -> 1 (/s)
    """

    def __init__(
        self,
        lambda_1: float = 30.0,
        lambda_2: float = 300.0,
        q_12: float = 0.1,
        q_21: float = 0.5,
        seed: int = 42,
    ):
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.q_12 = float(q_12)
        self.q_21 = float(q_21)
        self.rng = np.random.default_rng(seed)

    def generate_timestamps(self, n_samples: int, t_start: float = 0.0) -> np.ndarray:
        """Generate n_samples timestamps under continuous-time Markov modulation."""
        timestamps = np.empty(n_samples, dtype=np.float64)
        current_t = float(t_start)
        current_state = 1
        generated = 0

        # Stationary probabilities
        pi_1 = self.q_21 / (self.q_12 + self.q_21)
        current_state = 1 if self.rng.random() < pi_1 else 2

        while generated < n_samples:
            rate_lambda = self.lambda_1 if current_state == 1 else self.lambda_2
            transition_rate = self.q_12 if current_state == 1 else self.q_21

            # Competing exponentials: next arrival vs next state transition
            time_to_arrival = self.rng.exponential(1.0 / rate_lambda)
            time_to_transition = self.rng.exponential(1.0 / transition_rate)

            if time_to_arrival < time_to_transition:
                # Arrival occurs before state transition
                current_t += time_to_arrival
                timestamps[generated] = current_t
                generated += 1
            else:
                # State transition occurs
                current_t += time_to_transition
                current_state = 2 if current_state == 1 else 1

        return timestamps


class ParetoTrafficGenerator:
    """
    Heavy-Tailed Pareto Traffic Generator.
    
    Density:
      f(t) = (alpha * t_m^alpha) / (t^{alpha + 1})  for t >= t_m
    
    Theoretical Note:
      When alpha <= 2 (e.g. alpha = 1.2), the theoretical variance is infinite (sigma^2 = inf),
      causing intense microbursts and heavy-tailed inter-arrival intervals.
      The empirical squared coefficient of variation (c_a^2) is recorded over finite runs.
    """

    def __init__(
        self,
        alpha: float = 1.2,
        mean_rate: float = 100.0,
        seed: int = 42,
    ):
        if not (1.0 < alpha <= 2.0):
            raise ValueError("alpha must be in (1.0, 2.0] for finite mean and infinite variance")
        self.alpha = float(alpha)
        self.mean_rate = float(mean_rate)
        self.rng = np.random.default_rng(seed)

        # Mean E[T] = (alpha * t_m) / (alpha - 1) = 1.0 / mean_rate
        # => t_m = (alpha - 1) / (alpha * mean_rate)
        self.t_m = (self.alpha - 1.0) / (self.alpha * self.mean_rate)

    def generate_inter_arrivals(self, n_samples: int) -> Tuple[np.ndarray, float]:
        """
        Generate Pareto inter-arrivals via inverse transform:
          Delta_t = t_m / (1 - U)^(1 / alpha)
        
        Returns:
            dt: np.ndarray of inter-arrival times [s]
            empirical_scv: float (c_a^2 computed empirically over the sample)
        """
        u = self.rng.random(size=n_samples)
        # Avoid u == 1.0 boundary
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        dt = self.t_m / ((1.0 - u) ** (1.0 / self.alpha))

        # Compute empirical SCV
        dt_mean = float(np.mean(dt))
        dt_var = float(np.var(dt, ddof=1))
        empirical_scv = float(dt_var / (dt_mean ** 2)) if dt_mean > 0 else 0.0

        return dt, empirical_scv

    def generate_timestamps(self, n_samples: int, t_start: float = 0.0) -> Tuple[np.ndarray, float]:
        dt, empirical_scv = self.generate_inter_arrivals(n_samples)
        timestamps = t_start + np.cumsum(dt)
        return timestamps, empirical_scv


class ClientTrafficTraceSampler:
    """
    Samples network flows with replacement from a client's Stage 2 test split.
    
    Pre-computes Gate predictions and Autoencoder reconstruction errors on the test split
    to enable high-throughput simulation of hundreds of thousands of arrivals.
    """

    def __init__(
        self,
        client_id: int,
        data_npz_path: str,
        ae_checkpoint_path: str,
        gate_checkpoint_path: str,
        seed: int = 42,
    ):
        self.client_id = client_id
        self.rng = np.random.default_rng(seed)

        # 1. Load client Stage 2 test partition
        data = np.load(data_npz_path, allow_pickle=True)
        self.x_test = data["X_test"]
        self.y_test = data["y_test"]
        self.attack_test = data["attack_test"] if "attack_test" in data else np.array(["Unknown"] * len(self.y_test))
        self.n_test_samples = len(self.x_test)

        # 2. Load frozen models and pre-score test split
        ae = Autoencoder(input_dim=38).eval()
        ae.load_state_dict(torch.load(ae_checkpoint_path, map_location="cpu"))

        gate = GatingRegressor(input_dim=38).eval()
        gate.load_state_dict(torch.load(gate_checkpoint_path, map_location="cpu"))

        with torch.no_grad():
            t_x = torch.tensor(self.x_test, dtype=torch.float32)
            recon = ae(t_x)
            self.precomputed_mse = Autoencoder.compute_mse(t_x, recon).numpy()
            self.precomputed_gate_pred = gate(t_x).squeeze(1).numpy()

    def sample_arrivals(
        self,
        timestamps: np.ndarray,
        s_gru_bar: float = 0.0001799,
        service_scv: float = 0.0105,
    ) -> List[FlowArrival]:
        """
        Sample n arrivals with replacement coupled with stochastic timestamps and service times.
        """
        n_arrivals = len(timestamps)
        sampled_indices = self.rng.integers(0, self.n_test_samples, size=n_arrivals)

        # Generate service times (lognormal or gamma matching mean and profiled SCV)
        if service_scv > 0:
            sigma_s = np.sqrt(service_scv * (s_gru_bar ** 2))
            # Lognormal parameterization
            sigma_log = np.sqrt(np.log(1.0 + service_scv))
            mu_log = np.log(s_gru_bar) - 0.5 * (sigma_log ** 2)
            service_times = self.rng.lognormal(mean=mu_log, sigma=sigma_log, size=n_arrivals)
        else:
            service_times = np.full(n_arrivals, s_gru_bar)

        arrivals = []
        for i in range(n_arrivals):
            idx = sampled_indices[i]
            flow = FlowArrival(
                flow_id=i,
                arrival_time=float(timestamps[i]),
                x=self.x_test[idx],
                y_true=int(self.y_test[idx]),
                attack_family=str(self.attack_test[idx]),
                service_time=float(service_times[i]),
            )
            # Attach pre-scored evaluations as attributes for instant retrieval
            flow.precomputed_gate_pred = float(self.precomputed_gate_pred[idx])
            flow.precomputed_ae_mse = float(self.precomputed_mse[idx])
            arrivals.append(flow)

        return arrivals


"""
Phase 4.2: Hardware Latency Profiler & Service Capacity Measurement.

Measures host CPU execution times for:
- Frozen Stage 3 Gating Regressors (Light triage)
- Frozen Stage 3 Autoencoders (Light anomaly detection)
- 2-layer GRU execution stub (Heavy path proxy benchmark)

Computes empirical service rates, coefficient of variation, front-end rate limits,
and bootstrap confidence intervals across batch sizes B in {1, 8, 32, 64}.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.autoencoder import Autoencoder
from src.models.gate import GatingRegressor


class GRUExecutionStub(nn.Module):
    """
    Two-layer GRU proxy execution stub for Phase 4 queue telemetry profiling.
    Matches anticipated Stage 5 model complexity:
      input_dim=38, hidden_dim=64, num_layers=2, sequence_window=10.
    """
    def __init__(self, input_dim: int = 38, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [B, W, D] where B is batch size, W is sequence length (10), D is feature dim (38).
        """
        gru_out, _ = self.gru(x)
        # Classification from last sequence step
        last_step = gru_out[:, -1, :]
        logits = self.classifier(last_step)
        return logits


def profile_model_latencies(
    model: nn.Module,
    sample_input: torch.Tensor,
    warmup_iters: int = 1000,
    timed_iters: int = 10000,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Profile high-resolution execution latency on CPU with warmup and bootstrap CIs.
    """
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    sample_input = sample_input.to(device)

    # 1. Warmup passes
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(sample_input)

    # 2. Timed inference passes
    durations = np.empty(timed_iters, dtype=np.float64)
    with torch.no_grad():
        for i in range(timed_iters):
            t0 = time.perf_counter()
            _ = model(sample_input)
            t1 = time.perf_counter()
            durations[i] = t1 - t0

    # Convert to milliseconds
    durations_ms = durations * 1000.0

    mean_ms = float(np.mean(durations_ms))
    std_ms = float(np.std(durations_ms, ddof=1))
    median_ms = float(np.median(durations_ms))
    p95_ms = float(np.percentile(durations_ms, 95.0))
    p99_ms = float(np.percentile(durations_ms, 99.0))
    min_ms = float(np.min(durations_ms))
    max_ms = float(np.max(durations_ms))

    # Variance and squared coefficient of variation (c_s^2)
    var_ms = float(np.var(durations_ms, ddof=1))
    c_s_squared = float(var_ms / (mean_ms ** 2)) if mean_ms > 0 else 0.0

    # 3. 1,000-sample block-bootstrap for 95% Confidence Interval on mean latency
    rng = np.random.default_rng(random_seed)
    block_size = max(10, timed_iters // 100)
    n_blocks = timed_iters // block_size
    blocks = durations_ms[: n_blocks * block_size].reshape(n_blocks, block_size)

    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled_block_indices = rng.integers(0, n_blocks, size=n_blocks)
        boot_sample = blocks[sampled_block_indices].ravel()
        boot_means[b] = np.mean(boot_sample)

    ci_low_ms = float(np.percentile(boot_means, 2.5))
    ci_high_ms = float(np.percentile(boot_means, 97.5))

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "ci_95_low_ms": ci_low_ms,
        "ci_95_high_ms": ci_high_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "var_ms": var_ms,
        "c_s_squared": c_s_squared,
        "timed_iterations": timed_iters,
        "warmup_iterations": warmup_iters,
    }


def run_hardware_latency_profiling(
    ckpt_dir: str = "outputs/checkpoints/stage3",
    output_dir: str = "outputs/metrics/stage4",
    warmup_iters: int = 1000,
    timed_iters: int = 10000,
    n_clients: int = 10,
    epsilon_cpu: float = 0.05,
    batch_sizes: List[int] = [1, 8, 32, 64],
) -> Dict[str, Any]:
    """
    Execute empirical CPU latency profiling across all 10 clients and export summary profiles.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(ckpt_dir)

    print("=" * 80)
    print("STAGE 4 — PHASE 4.2: HARDWARE LATENCY PROFILING")
    print(f"Warmup: {warmup_iters} iterations | Timed: {timed_iters} iterations | CPU Device")
    print("=" * 80)

    # 1. Profile 2-Layer GRU Execution Stub across batch sizes
    gru_model = GRUExecutionStub(input_dim=38, hidden_dim=64, num_layers=2, num_classes=2)
    gru_batch_profiles = {}

    print("\nProfiling 2-Layer GRU Proxy Execution Stub...")
    for b in batch_sizes:
        x_gru = torch.randn(b, 10, 38)
        prof = profile_model_latencies(gru_model, x_gru, warmup_iters, timed_iters)
        gru_batch_profiles[f"B_{b}"] = prof
        print(f"  Batch B={b:2d}: Mean = {prof['mean_ms']:.4f} ms (95% CI: [{prof['ci_95_low_ms']:.4f}, {prof['ci_95_high_ms']:.4f}]) | c_s^2 = {prof['c_s_squared']:.4f}")

    gru_single = gru_batch_profiles["B_1"]
    s_gru_bar_ms = gru_single["mean_ms"]
    mu_heavy_fps = 1000.0 / s_gru_bar_ms if s_gru_bar_ms > 0 else 0.0

    # 2. Profile per-client Autoencoder and Gate models
    client_profiles = {}
    csv_rows = []

    print("\nProfiling Per-Client Stage 3 Autoencoders and Gating Regressors (Single-Flow B=1)...")
    for c_idx in range(n_clients):
        cid = f"client_{c_idx:02d}"

        # Load Autoencoder
        ae = Autoencoder(input_dim=38)
        ae.load_state_dict(torch.load(ckpt_path / f"{cid}_autoencoder.pt", map_location="cpu"))

        # Load Gate
        gate = GatingRegressor(input_dim=38)
        gate.load_state_dict(torch.load(ckpt_path / f"{cid}_gate.pt", map_location="cpu"))

        # Profile single flow B=1
        x_single = torch.randn(1, 38)
        prof_ae = profile_model_latencies(ae, x_single, warmup_iters, timed_iters)
        prof_gate = profile_model_latencies(gate, x_single, warmup_iters, timed_iters)

        l_gate_ms = prof_gate["mean_ms"]
        l_ae_ms = prof_ae["mean_ms"]

        # Derived Service Rates (in flows/second)
        mu_light_fps = 1000.0 / l_ae_ms if l_ae_ms > 0 else 0.0

        # Front-End Rate Limit (Safeguard F2, using epsilon_cpu = 0.05)
        # Latency in denominator converted from ms to seconds: (L_gate + L_ae) * 1e-3
        triage_latency_sec = (l_gate_ms + l_ae_ms) / 1000.0
        lambda_frontend_max_fps = (1.0 - epsilon_cpu) / triage_latency_sec if triage_latency_sec > 0 else 0.0

        # Ordering check
        ordering_valid = (0.0 < l_gate_ms <= l_ae_ms < s_gru_bar_ms)

        client_profiles[cid] = {
            "client_id": c_idx,
            "gate": prof_gate,
            "ae": prof_ae,
            "gru_proxy": gru_single,
            "derived_metrics": {
                "l_gate_mean_ms": l_gate_ms,
                "l_ae_mean_ms": l_ae_ms,
                "s_gru_mean_ms": s_gru_bar_ms,
                "mu_light_flows_per_sec": mu_light_fps,
                "mu_heavy_flows_per_sec": mu_heavy_fps,
                "c_s_gru_squared": gru_single["c_s_squared"],
                "lambda_frontend_max_flows_per_sec": lambda_frontend_max_fps,
                "epsilon_cpu": epsilon_cpu,
                "ordering_valid": ordering_valid,
            },
        }

        print(f"  {cid}: L_Gate={l_gate_ms:.4f} ms | L_AE={l_ae_ms:.4f} ms | S_GRU={s_gru_bar_ms:.4f} ms | mu_light={mu_light_fps:.1f} f/s | lambda_front_max={lambda_frontend_max_fps:.1f} f/s")

        csv_rows.append({
            "client": cid,
            "gate_mean_ms": l_gate_ms,
            "gate_ci_95_low_ms": prof_gate["ci_95_low_ms"],
            "gate_ci_95_high_ms": prof_gate["ci_95_high_ms"],
            "gate_std_ms": prof_gate["std_ms"],
            "ae_mean_ms": l_ae_ms,
            "ae_ci_95_low_ms": prof_ae["ci_95_low_ms"],
            "ae_ci_95_high_ms": prof_ae["ci_95_high_ms"],
            "ae_std_ms": prof_ae["std_ms"],
            "gru_proxy_mean_ms": s_gru_bar_ms,
            "gru_proxy_ci_95_low_ms": gru_single["ci_95_low_ms"],
            "gru_proxy_ci_95_high_ms": gru_single["ci_95_high_ms"],
            "gru_proxy_std_ms": gru_single["std_ms"],
            "mu_light_fps": mu_light_fps,
            "mu_heavy_fps": mu_heavy_fps,
            "c_s_gru_squared": gru_single["c_s_squared"],
            "lambda_frontend_max_fps": lambda_frontend_max_fps,
            "ordering_valid": ordering_valid,
        })

    # Summary Statistics across all clients
    mean_l_gate = float(np.mean([cp["derived_metrics"]["l_gate_mean_ms"] for cp in client_profiles.values()]))
    mean_l_ae = float(np.mean([cp["derived_metrics"]["l_ae_mean_ms"] for cp in client_profiles.values()]))
    mean_mu_light = float(np.mean([cp["derived_metrics"]["mu_light_flows_per_sec"] for cp in client_profiles.values()]))
    mean_lambda_fe_max = float(np.mean([cp["derived_metrics"]["lambda_frontend_max_flows_per_sec"] for cp in client_profiles.values()]))

    full_profile = {
        "metadata": {
            "stage": "4.2",
            "host_hardware": "Host CPU",
            "warmup_iterations": warmup_iters,
            "timed_iterations": timed_iters,
            "bootstrap_samples": 1000,
            "epsilon_cpu": epsilon_cpu,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "macro_summary": {
            "mean_l_gate_ms": mean_l_gate,
            "mean_l_ae_ms": mean_l_ae,
            "mean_s_gru_ms": s_gru_bar_ms,
            "mean_mu_light_fps": mean_mu_light,
            "mu_heavy_fps": mu_heavy_fps,
            "c_s_gru_squared": gru_single["c_s_squared"],
            "mean_lambda_frontend_max_fps": mean_lambda_fe_max,
        },
        "gru_batch_scaling": gru_batch_profiles,
        "per_client": client_profiles,
    }

    # Save JSON and CSV
    with open(out_path / "hardware_latency_profile.json", "w") as f:
        json.dump(full_profile, f, indent=2)

    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(out_path / "hardware_latency_profile.csv", index=False)

    print(f"\nSaved hardware latency profile to:")
    print(f"  - {out_path / 'hardware_latency_profile.json'}")
    print(f"  - {out_path / 'hardware_latency_profile.csv'}")

    return full_profile


if __name__ == "__main__":
    run_hardware_latency_profiling()


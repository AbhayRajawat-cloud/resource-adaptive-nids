"""
Stage 3 Pipeline Orchestrator: Local Autoencoder and Self-Supervised Gate Development.

Workflow:
1. Inspect and record initial SHA256 checksums of Stage 2 NPZ artifacts.
2. For each client (00 to 09):
    a. Load client k data from data/processed/alpha_0.5_seed_42/client_{k:02d}/data.npz.
    b. Extract local benign training split (x_train[y_train == 0]) and benign val split.
    c. Train local Autoencoder (train_autoencoder.py) and calibrate tau_AE_k with 95% bootstrap CI.
    d. Freeze AE, generate teacher targets, train student Gating Regressor (train_gate.py).
    e. Calibrate tau_gate_k with 95% bootstrap CI.
    f. Save models and threshold artifacts to outputs/checkpoints/stage3/.
    g. Evaluate models on Tier 1 (Full), Tier 2 (Deduplicated Unique), Tier 3 (Train-Exclusive Unique).
3. Compute globally pooled metrics across all 10 clients.
4. Verify Stage 2 artifact immutability (re-verify SHA256 checksums).
5. Export full summaries to outputs/metrics/stage3/stage3_results_summary.json and .csv.
"""

from pathlib import Path
from typing import Dict, Any, List
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch

from src.training.train_autoencoder import train_client_autoencoder
from src.training.train_gate import train_client_gate
from src.evaluation.stage3_eval import evaluate_client_models
from src.evaluation.metrics import compute_stage3_tier_metrics
from src.utils.reproducibility import set_seed, get_device
from src.utils.config import load_config


def compute_sha256(path: Path) -> str:
    """Compute sha256 checksum of file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_stage3(
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute Stage 3 training and evaluation across all 10 clients."""
    print("=" * 70)
    print("STAGE 3: LOCAL AUTOENCODER & SELF-SUPERVISED GATE DEVELOPMENT")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Resolved Device: {device}")

    # Verify Stage 2 artifacts and record initial checksums
    print("\n--- STEP 1: VERIFYING STAGE 2 NPZ IMMUTABILITY (INITIAL CHECKSUMS) ---")
    initial_checksums = {}
    client_dirs = sorted(list(data_dir.glob("client_*")))
    assert len(client_dirs) == 10, f"Expected 10 client directories, found {len(client_dirs)}"

    for cdir in client_dirs:
        npz_file = cdir / "data.npz"
        assert npz_file.exists(), f"Missing {npz_file}"
        c_hash = compute_sha256(npz_file)
        initial_checksums[cdir.name] = c_hash
        print(f"  {cdir.name}: sha256 = {c_hash[:16]}...")

    checkpoints_dir = output_dir / "checkpoints" / "stage3"
    metrics_dir = output_dir / "metrics" / "stage3"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    client_results = {}
    pooled_test_preds = {"y_true": [], "mse_ae": [], "e_hat_gate": []}
    threshold_records = []

    # Iterate over all 10 clients
    print("\n--- STEP 2 & 3: LOCAL TRAINING & CALIBRATION (CLIENTS 00 TO 09) ---")
    start_time = time.time()

    for cdir in client_dirs:
        cid = int(cdir.name.split("_")[-1])
        print(f"\n>>> PROCESSING CLIENT {cid:02d} <<<")

        # Load data.npz
        npz_path = cdir / "data.npz"
        data = np.load(npz_path, allow_pickle=True)

        # Robust case-agnostic extraction
        keys = list(data.keys())
        x_tr_key = "X_train" if "X_train" in keys else "x_train"
        y_tr_key = "y_train" if "y_train" in keys else "y_train"
        x_va_key = "X_val" if "X_val" in keys else "x_val"
        y_va_key = "y_val" if "y_val" in keys else "y_val"
        x_te_key = "X_test" if "X_test" in keys else "x_test"
        y_te_key = "y_test" if "y_test" in keys else "y_test"
        atk_te_key = "attack_test" if "attack_test" in keys else "attack_test"

        x_train = data[x_tr_key]
        y_train = data[y_tr_key]
        x_val = data[x_va_key]
        y_val = data[y_va_key]
        x_test = data[x_te_key]
        y_test = data[y_te_key]
        attack_test = data[atk_te_key]

        # Extract benign-only partitions for training
        benign_tr_mask = (y_train == 0)
        benign_va_mask = (y_val == 0)
        x_train_benign = x_train[benign_tr_mask]
        x_val_benign = x_val[benign_va_mask]

        print(f"  Benign Train: {len(x_train_benign):,} | Benign Val: {len(x_val_benign):,} | Total Test: {len(x_test):,}")

        # Train Autoencoder
        set_seed(42 + cid)
        ae_res = train_client_autoencoder(
            client_id=cid,
            x_train_benign=x_train_benign,
            x_val_benign=x_val_benign,
            device=device,
            config=config.get("models", {}).get("autoencoder", {}),
            save_dir=checkpoints_dir,
        )
        print(f"  [AE] Best Epoch: {ae_res['best_epoch']} | Val MSE: {ae_res['best_val_loss']:.6f}")
        print(f"  [AE] Threshold tau_AE = {ae_res['tau_ae']:.6f} (95% CI: [{ae_res['tau_ae_ci_low']:.6f}, {ae_res['tau_ae_ci_high']:.6f}])")

        # Train Student Gating Regressor
        set_seed(42 + cid)
        gate_res = train_client_gate(
            client_id=cid,
            model_ae=ae_res["model"],
            x_train_benign=x_train_benign,
            x_val_benign=x_val_benign,
            tau_ae=ae_res["tau_ae"],
            tau_ae_ci_low=ae_res["tau_ae_ci_low"],
            tau_ae_ci_high=ae_res["tau_ae_ci_high"],
            device=device,
            config=config.get("models", {}).get("gate", {}),
            save_dir=checkpoints_dir,
        )
        print(f"  [Gate] Best Epoch: {gate_res['best_epoch']} | Val Huber Loss: {gate_res['best_val_loss']:.6f}")
        print(f"  [Gate] Threshold tau_gate = {gate_res['tau_gate']:.6f} (95% CI: [{gate_res['tau_gate_ci_low']:.6f}, {gate_res['tau_gate_ci_high']:.6f}])")

        threshold_records.append({
            "client_id": cid,
            "tau_ae": ae_res["tau_ae"],
            "tau_ae_ci_low": ae_res["tau_ae_ci_low"],
            "tau_ae_ci_high": ae_res["tau_ae_ci_high"],
            "tau_gate": gate_res["tau_gate"],
            "tau_gate_ci_low": gate_res["tau_gate_ci_low"],
            "tau_gate_ci_high": gate_res["tau_gate_ci_high"],
        })

        # Evaluate across 3 test tiers
        eval_res = evaluate_client_models(
            model_ae=ae_res["model"],
            model_gate=gate_res["model"],
            x_train=x_train,
            x_test=x_test,
            y_test=y_test,
            attack_test=attack_test,
            tau_ae=ae_res["tau_ae"],
            tau_gate=gate_res["tau_gate"],
            device=device,
        )

        client_results[f"client_{cid:02d}"] = {
            "thresholds": threshold_records[-1],
            "tier1": eval_res["tier1"],
            "tier2": eval_res["tier2"],
            "tier3": eval_res["tier3"],
        }

        # Accumulate for global pooling
        pooled_test_preds["y_true"].append(eval_res["raw_test_preds"]["y_true"])
        pooled_test_preds["mse_ae"].append(eval_res["raw_test_preds"]["mse_ae"])
        pooled_test_preds["e_hat_gate"].append(eval_res["raw_test_preds"]["e_hat_gate"])

        # Quick logging of key metrics
        t1 = eval_res["tier1"]
        print(f"  [Eval Tier 1] Attack Passage Recall: {t1['gate_routing']['attack_passage_recall_pct']:.2f}% (H1 Met: {t1['gate_routing']['h1_benchmark_met_70pct']})")
        print(f"  [Eval Tier 1] Gate-AE Agreement: {t1['agreement']['gate_ae_agreement_rate_pct']:.2f}% | Gate RMSE: {t1['gate_regression']['rmse']:.4f}")

    elapsed = time.time() - start_time
    print(f"\nAll 10 clients trained and evaluated in {elapsed:.1f}s.")

    # Global Pooling Evaluation
    print("\n--- STEP 4: GLOBALLY POOLED EVALUATION ---")
    pooled_y = np.concatenate(pooled_test_preds["y_true"])
    pooled_mse = np.concatenate(pooled_test_preds["mse_ae"])
    pooled_ehat = np.concatenate(pooled_test_preds["e_hat_gate"])

    # Global thresholds: mean across clients
    mean_tau_ae = float(np.mean([r["tau_ae"] for r in threshold_records]))
    mean_tau_gate = float(np.mean([r["tau_gate"] for r in threshold_records]))

    pooled_tier1 = compute_stage3_tier_metrics(
        y_true=pooled_y,
        mse_ae=pooled_mse,
        e_hat_gate=pooled_ehat,
        tau_ae=mean_tau_ae,
        tau_gate=mean_tau_gate,
    )

    print(f"  Pooled Tier 1 Total Samples: {len(pooled_y):,}")
    print(f"  Pooled Attack Passage Recall: {pooled_tier1['gate_routing']['attack_passage_recall_pct']:.2f}%")
    print(f"  Pooled Gate-AE Agreement Rate: {pooled_tier1['agreement']['gate_ae_agreement_rate_pct']:.2f}%")

    # Step 5: Check Stage 2 Immutability
    print("\n--- STEP 5: VERIFYING STAGE 2 ARTIFACT IMMUTABILITY (FINAL CHECKSUMS) ---")
    immutability_passed = True
    for cdir in client_dirs:
        npz_file = cdir / "data.npz"
        final_hash = compute_sha256(npz_file)
        initial_hash = initial_checksums[cdir.name]
        is_match = (final_hash == initial_hash)
        if not is_match:
            immutability_passed = False
            print(f"  [FAIL] {cdir.name} checksum mismatch! Original: {initial_hash}, Current: {final_hash}")
        else:
            print(f"  [PASS] {cdir.name}: Checksum verified identical.")

    assert immutability_passed, "STAGE 2 IMMUTABILITY VIOLATED!"

    # Compile Final Results
    summary_report = {
        "stage": 3,
        "device": str(device),
        "execution_time_seconds": elapsed,
        "immutability_verified": immutability_passed,
        "thresholds_summary": threshold_records,
        "pooled_metrics": {
            "mean_tau_ae": mean_tau_ae,
            "mean_tau_gate": mean_tau_gate,
            "tier1": pooled_tier1,
        },
        "per_client_results": client_results,
    }

    # Save JSON summary
    json_path = metrics_dir / "stage3_results_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\n[Saved] Full JSON summary -> {json_path}")

    # Build and Save Flattened CSV Summary
    rows = []
    for cname, res in client_results.items():
        th = res["thresholds"]
        for tier_name in ["tier1", "tier2", "tier3"]:
            tdat = res[tier_name]
            if "error" in tdat:
                continue
            rows.append({
                "client": cname,
                "tier": tier_name,
                "n_samples": tdat["n_samples"],
                "n_attack": tdat["n_attack"],
                "tau_ae": th["tau_ae"],
                "tau_gate": th["tau_gate"],
                "gate_rmse": tdat["gate_regression"]["rmse"],
                "gate_r2": tdat["gate_regression"]["r2_score"],
                "gate_spearman_rho": tdat["gate_regression"]["spearman_rho"],
                "ae_benign_mean_mse": tdat["ae_reconstruction_stats"]["benign_mean_mse"],
                "ae_attack_mean_mse": tdat["ae_reconstruction_stats"]["attack_mean_mse"],
                "ae_precision": tdat["ae_classification"]["precision"],
                "ae_recall": tdat["ae_classification"]["recall"],
                "ae_f1": tdat["ae_classification"]["f1_score"],
                "heavy_path_trigger_rate_pct": tdat["gate_routing"]["heavy_path_trigger_rate_pct"],
                "attack_passage_recall_pct": tdat["gate_routing"]["attack_passage_recall_pct"],
                "benign_passage_rate_pct": tdat["gate_routing"]["benign_passage_rate_pct"],
                "gate_ae_agreement_rate_pct": tdat["agreement"]["gate_ae_agreement_rate_pct"],
            })

    csv_path = metrics_dir / "stage3_results_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[Saved] CSV summary -> {csv_path}")

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Local AE and Gate Training & Evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dev = get_device(args.device)

    data_path = Path("data/processed/alpha_0.5_seed_42")
    out_path = Path("outputs")

    run_stage3(data_dir=data_path, output_dir=out_path, device=dev, config=cfg)


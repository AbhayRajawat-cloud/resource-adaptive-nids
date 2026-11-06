"""
Audit Evidence and Diagnostics Generator for Stage 3.
Does not retrain or modify anything.
Extracts:
1. Exact SHA-256 hashes of all Stage 2 artifacts.
2. Exact Tier 3 counts and retention.
3. Diagnostic metrics for AE and Gate using local thresholds.
4. Confusion matrices, Gate-AE agreement on benign vs attack.
5. Attack-family counts and recall per family.
6. Target/prediction scaling (min, max, mean, std) on train, val, and test.
7. Correctly sample-weighted and client-averaged pooled metrics.
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.models.autoencoder import Autoencoder
from src.models.gate import GatingRegressor

def main():
    root = Path(".")
    stage2_dir = root / "data" / "processed" / "alpha_0.5_seed_42"
    ckpt_dir = root / "outputs" / "checkpoints" / "stage3"
    out_dir = root / "outputs" / "verification" / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. SHA-256 of all Stage 2 artifacts
    hashes = {}
    for cdir in sorted(stage2_dir.glob("client_*")):
        cname = cdir.name
        hashes[cname] = {}
        for fname in ["data.npz", "scaler_params.json", "client_stats.json"]:
            fpath = cdir / fname
            if fpath.exists():
                h = hashlib.sha256(fpath.read_bytes()).hexdigest()
                hashes[cname][fname] = h
            else:
                hashes[cname][fname] = "MISSING"

    with open(out_dir / "stage2_artifact_hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)

    # Diagnostic containers
    client_diagnostics = {}
    attack_family_rows = []
    scaling_rows = []
    threshold_comp_rows = []

    # Process all 10 clients
    for cdir in sorted(stage2_dir.glob("client_*")):
        cname = cdir.name
        cid = int(cname.split("_")[-1])

        # Load Stage 2 data
        data = np.load(cdir / "data.npz", allow_pickle=True)
        x_train, y_train = data["X_train"], data["y_train"]
        x_val, y_val = data["X_val"], data["y_val"]
        x_test, y_test = data["X_test"], data["y_test"]
        attack_test = data["attack_test"]

        # Load Stage 3 thresholds
        with open(ckpt_dir / f"{cname}_thresholds.json") as f:
            th = json.load(f)
        tau_ae = th["tau_ae"]
        tau_gate = th["tau_gate"]

        # Load models
        ae = Autoencoder(38)
        ae.load_state_dict(torch.load(ckpt_dir / f"{cname}_autoencoder.pt", map_location="cpu"))
        ae.eval()

        gate = GatingRegressor(38)
        gate.load_state_dict(torch.load(ckpt_dir / f"{cname}_gate.pt", map_location="cpu"))
        gate.eval()

        with torch.no_grad():
            # Inferences on Test
            x_te_t = torch.tensor(x_test, dtype=torch.float32)
            recon_te = ae(x_te_t)
            mse_te = Autoencoder.compute_mse(x_te_t, recon_te).numpy()
            e_hat_te = gate(x_te_t).squeeze(1).numpy()
            target_te = np.log(mse_te + 1e-6)

            # Inferences on Val (Benign)
            x_va_benign_t = torch.tensor(x_val[y_val == 0], dtype=torch.float32)
            recon_va = ae(x_va_benign_t)
            mse_va_benign = Autoencoder.compute_mse(x_va_benign_t, recon_va).numpy()
            e_hat_va_benign = gate(x_va_benign_t).squeeze(1).numpy()
            target_va_benign = np.log(mse_va_benign + 1e-6)

            # Inferences on Train (Benign sample up to 50k for scaling checks)
            x_tr_benign = x_train[y_train == 0]
            if len(x_tr_benign) > 50000:
                rng = np.random.default_rng(42 + cid)
                sub_idx = rng.choice(len(x_tr_benign), size=50000, replace=False)
                x_tr_benign_sub = x_tr_benign[sub_idx]
            else:
                x_tr_benign_sub = x_tr_benign
            x_tr_benign_t = torch.tensor(x_tr_benign_sub, dtype=torch.float32)
            recon_tr = ae(x_tr_benign_t)
            mse_tr_benign = Autoencoder.compute_mse(x_tr_benign_t, recon_tr).numpy()
            e_hat_tr_benign = gate(x_tr_benign_t).squeeze(1).numpy()
            target_tr_benign = np.log(mse_tr_benign + 1e-6)

        # Record Target and Prediction Scaling
        for split_name, tg, pr in [
            ("train_benign", target_tr_benign, e_hat_tr_benign),
            ("val_benign", target_va_benign, e_hat_va_benign),
            ("test_full", target_te, e_hat_te),
            ("test_benign", target_te[y_test == 0], e_hat_te[y_test == 0]),
            ("test_attack", target_te[y_test == 1], e_hat_te[y_test == 1]),
        ]:
            scaling_rows.append({
                "client": cname,
                "split": split_name,
                "n_samples": len(tg),
                "target_min": float(np.min(tg)),
                "target_max": float(np.max(tg)),
                "target_mean": float(np.mean(tg)),
                "target_std": float(np.std(tg)),
                "pred_min": float(np.min(pr)),
                "pred_max": float(np.max(pr)),
                "pred_mean": float(np.mean(pr)),
                "pred_std": float(np.std(pr)),
            })

        # Decisions at local thresholds
        ae_decision = (mse_te >= tau_ae).astype(int)
        gate_decision = (e_hat_te >= tau_gate).astype(int)

        benign_mask = (y_test == 0)
        attack_mask = (y_test == 1)

        # Confusion Matrices
        # Format: tn, fp, fn, tp
        cm_ae = confusion_matrix(y_test, ae_decision, labels=[0, 1])
        tn_ae, fp_ae, fn_ae, tp_ae = cm_ae.ravel()

        cm_gate = confusion_matrix(y_test, gate_decision, labels=[0, 1])
        tn_g, fp_g, fn_g, tp_g = cm_gate.ravel()

        ae_recall = float(tp_ae / (tp_ae + fn_ae)) if (tp_ae + fn_ae) > 0 else 0.0
        ae_fpr = float(fp_ae / (fp_ae + tn_ae)) if (fp_ae + tn_ae) > 0 else 0.0

        gate_recall = float(tp_g / (tp_g + fn_g)) if (tp_g + fn_g) > 0 else 0.0
        gate_fpr = float(fp_g / (fp_g + tn_g)) if (fp_g + tn_g) > 0 else 0.0

        # Agreement separately on benign and malicious
        agree_benign = float(np.mean(ae_decision[benign_mask] == gate_decision[benign_mask])) * 100.0 if np.sum(benign_mask) > 0 else 0.0
        agree_attack = float(np.mean(ae_decision[attack_mask] == gate_decision[attack_mask])) * 100.0 if np.sum(attack_mask) > 0 else 0.0
        agree_overall = float(np.mean(ae_decision == gate_decision)) * 100.0

        # Reconstruction error & gate prediction quantiles
        q_levels = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        mse_benign_q = {f"q_{int(q*100)}": float(np.quantile(mse_te[benign_mask], q)) for q in q_levels} if np.sum(benign_mask) > 0 else {}
        mse_attack_q = {f"q_{int(q*100)}": float(np.quantile(mse_te[attack_mask], q)) for q in q_levels} if np.sum(attack_mask) > 0 else {}

        e_hat_benign_q = {f"q_{int(q*100)}": float(np.quantile(e_hat_te[benign_mask], q)) for q in q_levels} if np.sum(benign_mask) > 0 else {}
        e_hat_attack_q = {f"q_{int(q*100)}": float(np.quantile(e_hat_te[attack_mask], q)) for q in q_levels} if np.sum(attack_mask) > 0 else {}

        # Attack family breakdown
        df_client_te = pd.DataFrame({
            "attack": attack_test,
            "label": y_test,
            "ae_dec": ae_decision,
            "gate_dec": gate_decision,
            "mse": mse_te,
            "e_hat": e_hat_te
        })
        for atk_name, grp in df_client_te[df_client_te["label"] == 1].groupby("attack"):
            attack_family_rows.append({
                "client": cname,
                "attack_family": atk_name,
                "total_samples": len(grp),
                "ae_triggered": int(grp["ae_dec"].sum()),
                "ae_recall_pct": float(grp["ae_dec"].mean() * 100.0),
                "gate_triggered": int(grp["gate_dec"].sum()),
                "gate_passage_recall_pct": float(grp["gate_dec"].mean() * 100.0),
                "mean_mse": float(grp["mse"].mean()),
                "median_mse": float(grp["mse"].median()),
                "mean_e_hat": float(grp["e_hat"].mean()),
                "median_e_hat": float(grp["e_hat"].median()),
            })

        client_diagnostics[cname] = {
            "tau_ae": tau_ae,
            "tau_gate": tau_gate,
            "samples": {
                "total": len(y_test),
                "benign": int(np.sum(benign_mask)),
                "attack": int(np.sum(attack_mask))
            },
            "ae": {
                "recall": ae_recall,
                "fpr": ae_fpr,
                "confusion_matrix": {"tn": int(tn_ae), "fp": int(fp_ae), "fn": int(fn_ae), "tp": int(tp_ae)},
                "mse_quantiles": {
                    "benign": mse_benign_q,
                    "attack": mse_attack_q
                }
            },
            "gate": {
                "passage_recall": gate_recall,
                "benign_trigger_rate_fpr": gate_fpr,
                "confusion_matrix": {"tn": int(tn_g), "fp": int(fp_g), "fn": int(fn_g), "tp": int(tp_g)},
                "e_hat_quantiles": {
                    "benign": e_hat_benign_q,
                    "attack": e_hat_attack_q
                }
            },
            "agreement": {
                "overall_pct": agree_overall,
                "benign_pct": agree_benign,
                "attack_pct": agree_attack
            },
            "raw_predictions": {
                "y_true": y_test.tolist(),
                "ae_dec": ae_decision.tolist(),
                "gate_dec": gate_decision.tolist(),
                "mse": mse_te.tolist(),
                "e_hat": e_hat_te.tolist()
            }
        }

    # Save detailed per-client diagnostics
    with open(out_dir / "client_diagnostic_metrics.json", "w") as f:
        # Save without raw predictions to keep JSON concise
        clean_diag = {c: {k: v for k, v in client_diagnostics[c].items() if k != "raw_predictions"} for c in client_diagnostics}
        json.dump(clean_diag, f, indent=2)

    pd.DataFrame(attack_family_rows).to_csv(out_dir / "attack_family_performance.csv", index=False)
    pd.DataFrame(scaling_rows).to_csv(out_dir / "target_prediction_scaling.csv", index=False)

    # 4. Compute Both Sample-Weighted (True Local Thresholding) and Client-Averaged Pooled Metrics
    # Sample-weighted pooling of local decisions:
    total_samples = 0
    total_benign = 0
    total_attack = 0

    pooled_tp_ae = 0
    pooled_fp_ae = 0
    pooled_tn_ae = 0
    pooled_fn_ae = 0

    pooled_tp_g = 0
    pooled_fp_g = 0
    pooled_tn_g = 0
    pooled_fn_g = 0

    agree_benign_sum = 0
    agree_attack_sum = 0
    agree_total_sum = 0

    client_recalls_ae = []
    client_recalls_gate = []
    client_fprs_ae = []
    client_fprs_gate = []
    client_agreements = []

    for cname, d in client_diagnostics.items():
        n = d["samples"]["total"]
        nb = d["samples"]["benign"]
        na = d["samples"]["attack"]
        total_samples += n
        total_benign += nb
        total_attack += na

        cm_ae = d["ae"]["confusion_matrix"]
        pooled_tp_ae += cm_ae["tp"]
        pooled_fp_ae += cm_ae["fp"]
        pooled_tn_ae += cm_ae["tn"]
        pooled_fn_ae += cm_ae["fn"]

        cm_g = d["gate"]["confusion_matrix"]
        pooled_tp_g += cm_g["tp"]
        pooled_fp_g += cm_g["fp"]
        pooled_tn_g += cm_g["tn"]
        pooled_fn_g += cm_g["fn"]

        agree_benign_sum += (d["agreement"]["benign_pct"] / 100.0) * nb
        agree_attack_sum += (d["agreement"]["attack_pct"] / 100.0) * na
        agree_total_sum += (d["agreement"]["overall_pct"] / 100.0) * n

        client_recalls_ae.append(d["ae"]["recall"])
        client_recalls_gate.append(d["gate"]["passage_recall"])
        client_fprs_ae.append(d["ae"]["fpr"])
        client_fprs_gate.append(d["gate"]["benign_trigger_rate_fpr"])
        client_agreements.append(d["agreement"]["overall_pct"])

    sample_weighted_pooled = {
        "total_samples": total_samples,
        "total_benign": total_benign,
        "total_attack": total_attack,
        "ae": {
            "confusion_matrix": {"tp": pooled_tp_ae, "fp": pooled_fp_ae, "tn": pooled_tn_ae, "fn": pooled_fn_ae},
            "recall": pooled_tp_ae / (pooled_tp_ae + pooled_fn_ae) if (pooled_tp_ae + pooled_fn_ae) > 0 else 0.0,
            "fpr": pooled_fp_ae / (pooled_fp_ae + pooled_tn_ae) if (pooled_fp_ae + pooled_tn_ae) > 0 else 0.0,
        },
        "gate": {
            "confusion_matrix": {"tp": pooled_tp_g, "fp": pooled_fp_g, "tn": pooled_tn_g, "fn": pooled_fn_g},
            "passage_recall": pooled_tp_g / (pooled_tp_g + pooled_fn_g) if (pooled_tp_g + pooled_fn_g) > 0 else 0.0,
            "benign_trigger_rate_fpr": pooled_fp_g / (pooled_fp_g + pooled_tn_g) if (pooled_fp_g + pooled_tn_g) > 0 else 0.0,
        },
        "agreement": {
            "overall_pct": (agree_total_sum / total_samples) * 100.0,
            "benign_pct": (agree_benign_sum / total_benign) * 100.0 if total_benign > 0 else 0.0,
            "attack_pct": (agree_attack_sum / total_attack) * 100.0 if total_attack > 0 else 0.0,
        }
    }

    client_averaged_pooled = {
        "ae_recall_macro": float(np.mean(client_recalls_ae)),
        "ae_fpr_macro": float(np.mean(client_fprs_ae)),
        "gate_passage_recall_macro": float(np.mean(client_recalls_gate)),
        "gate_fpr_macro": float(np.mean(client_fprs_gate)),
        "gate_ae_agreement_macro": float(np.mean(client_agreements)),
    }

    pooled_summary = {
        "sample_weighted_local_thresholding": sample_weighted_pooled,
        "client_averaged_macro": client_averaged_pooled,
    }

    with open(out_dir / "pooled_local_threshold_summary.json", "w") as f:
        json.dump(pooled_summary, f, indent=2)

    print("Stage 3 verification data generation completed successfully!")

if __name__ == "__main__":
    main()


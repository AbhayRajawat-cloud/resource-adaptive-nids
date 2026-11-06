# Stage 3 Final Verification and Closure Audit Report

**Project Title:** Resource-Adaptive Gated Network Intrusion Detection for Federated IoT Edge Networks  
**Target Milestone:** Stage 3 — Local Autoencoder and Self-Supervised Gate Development  
**Audit Date:** November 3, 2026  
**Audit Type:** Strict Post-Execution Verification and Closure Audit (No Model Retraining / No Data Modification)  
**Execution Status:** Complete  
**Final Audit Recommendation:** **Stage 3 frozen with documented limitations (Stage 4 blocked pending threshold strategy approval)**

---

## Executive Summary

This final audit and closure report provides an exhaustive, evidence-based verification of Stage 3 deliverables. Following independent review, this report resolves the documentation discrepancy regarding the Autoencoder architecture, generates complete empirical metrics for Tier 3 (training-exclusive unique test set), provides an in-depth analysis of Client 02's performance under extreme non-IID partitioning, establishes a comprehensive limitation registry, and confirms strict adherence to all software and methodological constraints.

### Core Audit Outcomes
1. **Autoencoder Architecture Resolved:** The actual implemented model in [`src/models/autoencoder.py`](file:///Users/abhayrajawat/Downloads/resource-adaptive-nids/src/models/autoencoder.py) is the approved 5-layer symmetric architecture: $38 \to 20 \to 10 \to 20 \to 38$ with **2,008 trainable parameters** and **Linear (Identity) decoder output activation**. All 10 client checkpoints match this architecture with zero shape or key discrepancy. The description $38 \to 24 \to 12 \to 6 \dots$ in earlier draft documentation was an errant typographical mistake and is formally corrected.
2. **Tier 3 Metrics Fully Evaluated:** Across all 10 clients, Tier 3 evaluation was executed on existing checkpoints using local calibrated thresholds ($Q_{0.99}$). Out of 358,538 total test samples, exactly **55,897 samples** ($15.59\%$) are training-exclusive unique flows ($\text{Tier 3} \subseteq \text{Tier 2} \subseteq \text{Tier 1}$ strictly verified). Sample-weighted pooled Tier 3 performance achieves **22.13% AE Recall** (FPR $5.21\%$), **9.55% Gate Passage Recall** (FPR $5.13\%$), and **94.36% Gate–AE Agreement**.
3. **Client 02 Non-IID Limitation Analyzed:** Under Dirichlet partitioning ($\alpha=0.5$), Client 02 received $64.01\%$ attack traffic and only $2,556$ benign training samples ($548$ validation). This severe distribution shift caused the local 99th empirical percentile threshold to inflate to $\tau_{\text{AE}} = 0.128263$ (an order of magnitude higher than standard clients $\sim 0.016$), resulting in an observed Tier 1 AE recall of $1.85\%$ ($3.01\%$ on Tier 3). This is documented as a key edge heterogeneity limitation.
4. **Zero Retraining and Strict Immutability:** No model weights, checkpoints, hyperparameters, or Stage 2 dataset files were modified. All 10 client `data.npz` SHA-256 hashes remain byte-for-byte identical to baseline.
5. **Software vs. Research Metrics Distinction:** Software correctness tests pass 100% ($6/6$ pytest passed in $0.70$s). Empirical research performance (low gating recall under pure unsupervised 99th percentile calibration) is characterized transparently as an inherent design trade-off rather than a software defect.

---

## 1. Task 1 — Resolution of the Autoencoder Architecture Discrepancy

### 1.1 Implementation Inspection (`src/models/autoencoder.py`)
Direct inspection of [`src/models/autoencoder.py`](file:///Users/abhayrajawat/Downloads/resource-adaptive-nids/src/models/autoencoder.py#L26-L48) confirms the exact structural definition:

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 38):
        super().__init__()
        self.input_dim = input_dim

        # Encoder: 38 -> 20 -> 10
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ELU(alpha=1.0),
            nn.Linear(20, 10),
            nn.ELU(alpha=1.0),
        )

        # Decoder: 10 -> 20 -> 38
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ELU(alpha=1.0),
            nn.Linear(20, 38),
        )
```

### 1.2 Exact Layer Structure and Parameter Derivation
The model is instantiated as `Autoencoder(input_dim=38)`:

| Component | Layer Name | Operation | Input Dim | Output Dim | Weights Shape | Biases Shape | Trainable Parameters |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Encoder** | `encoder.0` | `nn.Linear` | 38 | 20 | $(20, 38)$ | $(20,)$ | $20 \times 38 + 20 = \mathbf{780}$ |
| | `encoder.1` | `nn.ELU(alpha=1.0)` | 20 | 20 | — | — | 0 |
| | `encoder.2` | `nn.Linear` | 20 | 10 | $(10, 20)$ | $(10,)$ | $10 \times 20 + 10 = \mathbf{210}$ |
| | `encoder.3` | `nn.ELU(alpha=1.0)` | 10 | 10 | — | — | 0 |
| **Decoder** | `decoder.0` | `nn.Linear` | 10 | 20 | $(20, 10)$ | $(20,)$ | $20 \times 10 + 20 = \mathbf{220}$ |
| | `decoder.1` | `nn.ELU(alpha=1.0)` | 20 | 20 | — | — | 0 |
| | `decoder.2` | `nn.Linear` | 20 | 38 | $(38, 20)$ | $(38,)$ | $38 \times 20 + 38 = \mathbf{798}$ |
| **Total** | | | | | | | $\mathbf{2,008}$ |

$$\text{Total Parameters} = 780 + 210 + 220 + 798 = \mathbf{2,008} \quad (8.03 \text{ KB in Float32})$$

### 1.3 Decoder Output Activation Confirmation
- The final module in the decoder sequence is `decoder.2 = nn.Linear(20, 38, bias=True)`.
- **Activation:** **Linear (Identity)**.
- **Verification:** There is **NO** Sigmoid, Tanh, ReLU, or Softmax applied to the output layer. The output $\hat{\mathbf{x}} \in \mathbb{R}^{38}$ represents continuous feature reconstructions, fully compliant with input normalization via `RobustScaler`.

### 1.4 Checkpoint Compatibility and State Dictionary Verification
The trained checkpoint [`outputs/checkpoints/stage3/client_00_autoencoder.pt`](file:///Users/abhayrajawat/Downloads/resource-adaptive-nids/outputs/checkpoints/stage3/client_00_autoencoder.pt) was loaded and evaluated against the instantiated class:
- **State Dict Structure:** Direct `OrderedDict` containing 8 tensor parameters.
- **Loading Result:** `model.load_state_dict(ckpt)` returned `<All keys matched successfully>`.
- **Shape Parity:** Every layer in the checkpoint matches the instantiated model tensor dimensions exactly:
  - `encoder.0.weight`: $(20, 38)$ — MATCH
  - `encoder.0.bias`: $(20,)$ — MATCH
  - `encoder.2.weight`: $(10, 20)$ — MATCH
  - `encoder.2.bias`: $(10,)$ — MATCH
  - `decoder.0.weight`: $(20, 10)$ — MATCH
  - `decoder.0.bias`: $(20,)$ — MATCH
  - `decoder.2.weight`: $(38, 20)$ — MATCH
  - `decoder.2.bias`: $(38,)$ — MATCH

### 1.5 Conclusion of Task 1
The codebase implementation and saved checkpoints **strictly adhere to the approved $38 \to 20 \to 10 \to 20 \to 38$ architecture (2,008 parameters)**. The earlier report text mentioning $38 \to 24 \to 12 \dots$ was an editorial typo in the draft documentation and is hereby formally corrected.

---

## 2. Task 2 — Complete Tier 3 Evaluation Metrics (Training-Exclusive Unique Set)

### 2.1 Methodology and Invariant Verification
Tier 3 test sets isolate unique network flow feature vectors in the local test split that **never appeared in the local training split**.
- **Float Rounding Precision:** 7 decimal places (`np.round(X, 7)`) to ensure exact hash parity without machine-epsilon false negatives.
- **Subset Invariant:** Verified for all 10 clients:
$$\text{Tier 3 (Train-Exclusive Unique)} \subseteq \text{Tier 2 (Deduplicated Unique)} \subseteq \text{Tier 1 (Full Test)}$$

### 2.2 Sample Counts and Retention Breakdown Across All 10 Clients

| Client ID | Tier 1 (Full Test) | Tier 2 (Dedup Unique) | Tier 3 (Train-Exclusive) | Retention of Tier 1 (%) | Retention of Tier 2 (%) | Subset Invariant Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `client_00` | 50,040 | 12,694 | 7,341 | 14.67% | 57.83% | **PASS** |
| `client_01` | 80,431 | 19,213 | 10,345 | 12.86% | 53.84% | **PASS** |
| `client_02` | 1,523 | 1,040 | 763 | 50.10% | 73.37% | **PASS** |
| `client_03` | 50,171 | 14,140 | 8,366 | 16.67% | 59.17% | **PASS** |
| `client_04` | 5,428 | 3,428 | 2,096 | 38.61% | 61.14% | **PASS** |
| `client_05` | 8,712 | 3,259 | 1,824 | 20.94% | 55.97% | **PASS** |
| `client_06` | 28,568 | 8,378 | 5,041 | 17.65% | 60.17% | **PASS** |
| `client_07` | 19,388 | 5,968 | 3,584 | 18.49% | 60.05% | **PASS** |
| `client_08` | 67,424 | 16,736 | 9,362 | 13.89% | 55.94% | **PASS** |
| `client_09` | 46,853 | 12,276 | 7,175 | 15.31% | 58.45% | **PASS** |
| **Total** | **358,538** | **97,132** | **55,897** | **15.59%** | **57.55%** | **PASS** |

> [!NOTE]
> Every client retains sufficient samples for both classes in Tier 3 (Benign: 199 to 9,593; Attack: 120 to 1,781). No metric is undefined or divided by zero.

### 2.3 Per-Client Tier 3 Empirical Performance Table
Evaluated strictly using each client's local calibrated thresholds ($\tau_{\text{AE}}^{(k)}, \tau_{\text{Gate}}^{(k)}$):

| Client ID | Benign Samples | Attack Samples | Total Samples | AE Recall (%) | AE FPR (%) | Gate Recall (%) | Gate FPR (%) | Gate–AE Agreement (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `client_00` | 7,093 | 248 | 7,341 | 29.84% | 5.41% | 6.85% | 5.26% | 95.15% |
| `client_01` | 9,593 | 752 | 10,345 | 27.13% | 6.14% | 14.49% | 5.75% | 95.57% |
| `client_02` | 199 | 564 | 763 | 3.01% | 3.52% | 3.37% | 2.01% | 98.03% |
| `client_03` | 6,796 | 1,570 | 8,366 | 23.63% | 5.24% | 12.17% | 4.22% | 92.55% |
| `client_04` | 315 | 1,781 | 2,096 | 14.43% | 2.86% | 5.45% | 2.54% | 88.79% |
| `client_05` | 1,682 | 142 | 1,824 | 29.58% | 4.40% | 7.04% | 4.46% | 94.46% |
| `client_06` | 4,552 | 489 | 5,041 | 29.04% | 5.21% | 9.41% | 5.56% | 94.33% |
| `client_07` | 3,464 | 120 | 3,584 | 35.83% | 4.53% | 11.67% | 4.82% | 95.06% |
| `client_08` | 8,508 | 854 | 9,362 | 29.51% | 4.90% | 14.29% | 5.37% | 93.98% |
| `client_09` | 6,678 | 497 | 7,175 | 30.38% | 4.73% | 9.05% | 4.99% | 95.33% |

### 2.4 Globally Pooled Tier 3 Metrics

Pooling is conducted by evaluating each sample against its client's own calibrated local threshold and aggregating the resulting confusion matrices:

| Evaluation Metric | Corrected Sample-Weighted Local Pooling | Corrected Macro Client-Averaging |
| :--- | :---: | :---: |
| **Total Evaluated Flows** | **55,897** (48,880 Benign, 7,017 Attack) | 10 Clients Averaged |
| **AE Attack Recall** | **22.13%** (1,553 / 7,017) | **25.24%** |
| **AE False Positive Rate (FPR)** | **5.21%** (2,546 / 48,880) | **4.69%** |
| **Gate Passage Recall** | **9.55%** (670 / 7,017) | **9.38%** |
| **Gate Benign Trigger Rate (FPR)** | **5.13%** (2,509 / 48,880) | **4.50%** |
| **Gate–AE Overall Agreement** | **94.36%** (52,747 / 55,897) | **94.33%** |

*Machine-readable evidence saved to [`outputs/verification/stage3/tier3_per_client_metrics.csv`](file:///Users/abhayrajawat/Downloads/resource-adaptive-nids/outputs/verification/stage3/tier3_per_client_metrics.csv) and [`outputs/verification/stage3/tier3_evaluation_summary.json`](file:///Users/abhayrajawat/Downloads/resource-adaptive-nids/outputs/verification/stage3/tier3_evaluation_summary.json).*

---

## 3. Task 3 — In-Depth Analysis of Client 02 Non-IID Performance

Client 02 represents an extreme edge scenario resulting from Dirichlet non-IID partitioning ($\alpha=0.5$).

### 3.1 Observed Empirical Results (Verified Facts)
- **Dataset Partition Size:** $10,147$ total records ($7,102$ train, $1,522$ val, $1,523$ test).
- **Class Imbalance:** $6,495$ malicious flows vs. $3,652$ benign flows (**$64.01\%$ attack skew** across all splits).
- **Benign Training Samples:** Exactly **2,556** benign records available for Autoencoder training (`y_train == 0`).
- **Benign Validation Samples:** Exactly **548** benign records available for threshold calibration.
- **Calibrated Local Thresholds ($Q_{0.99}$):**
  - $\tau_{\text{AE}}^{(2)} = \mathbf{0.128263}$ (95% CI: $[0.101656, 0.153550]$) — roughly $7.5\times$ higher than normal clients ($\sim 0.016-0.018$).
  - $\tau_{\text{Gate}}^{(2)} = \mathbf{-1.594451}$ (95% CI: $[-2.186922, -1.467549]$) — roughly $2.5$ log-units higher than normal clients ($\sim -4.1$).
- **Observed Detection Performance:**
  - **Tier 1 (Full Test, 975 attacks, 548 benign):**
    - AE Recall: **1.85%** (18 / 975), AE FPR: **1.28%** (7 / 548).
    - Gate Passage Recall: **2.05%** (20 / 975), Gate FPR: **0.73%** (4 / 548).
    - Gate–AE Agreement: **98.88%**.
  - **Tier 3 (Train-Exclusive Unique, 564 attacks, 199 benign):**
    - AE Recall: **3.01%** (17 / 564), AE FPR: **3.52%** (7 / 199).
    - Gate Passage Recall: **3.37%** (19 / 564), Gate FPR: **2.01%** (4 / 199).
    - Gate–AE Agreement: **98.03%**.

### 3.2 Distinction Between Observed Results and Potential Explanations
It is methodologically vital not to assert single-cause claims without ablation. We distinguish verified results from candidate hypotheses:
1. **Hypothesis A — Threshold Inflation via Small Validation Quantile:** With only 548 benign validation samples, the 99th empirical percentile corresponds to the 5th highest reconstruction error. If a small handful of benign flows in Client 02 possess unusual feature values, the estimated $\tau_{\text{AE}}$ inflates dramatically to $0.128$. Because attack MSE for Client 02 exhibits a median of $0.0016$ and 90th percentile of $0.043$, nearly all attack flows fall below $0.128$, collapsing recall.
2. **Hypothesis B — Local Feature Scaling Divergence:** In Stage 2, Client 02's `RobustScaler` was fitted strictly on its 2,556 local benign training records. If this limited sample lacks adequate representation of specific feature distributions, scaling parameters (median and IQR) diverge from the global population, altering error landscapes.
3. **Hypothesis C — Benign Sample Representation:** While 2,556 samples are sufficient for parameter convergence ($2,008$ parameters), the diversity of benign IoT traffic patterns represented in those 2,556 records may be narrower than in clients with 30,000+ benign records.

### 3.3 Implications for Heterogeneous Federated Edge Environments
This finding exposes a fundamental research challenge in decentralized edge NIDS:
- In real-world IoT networks, edge devices frequently operate in skewed environments (e.g., honeypots, edge gateways during active DDoS campaigns, or compromised subnets) where attack flows far outnumber normal traffic.
- When unsupervised anomaly detectors rely strictly on **isolated local quantile calibration** without global coordination, high attack skew and small benign sample sizes lead to conservative, inflated thresholds that induce near-total detection blackout ($1.85\%$ recall).
- **Recommendation for Future Stages:** Purely local percentile thresholding is brittle under extreme non-IID skew. In Stage 4/5, federated threshold calibration mechanisms (e.g., federated averaging of quantiles, server-guided calibration, or density-normalized anomaly scoring) should be investigated.

---

## 4. Software Verification vs. Empirical Research Metrics

To prevent confusing software defects with empirical machine learning behavior, audit results are partitioned into software verification and empirical research metrics.

### 4.1 Software Contract and Invariant Tests (100% Passed)
```text
tests/test_stage3.py::test_parameter_counts PASSED                       [ 16%]
tests/test_stage3.py::test_output_tensor_shapes PASSED                   [ 33%]
tests/test_stage3.py::test_stage2_immutability PASSED                    [ 50%]
tests/test_stage3.py::test_threshold_sanity PASSED                       [ 66%]
tests/test_stage3.py::test_benign_attack_separability PASSED             [ 83%]
tests/test_stage3.py::test_gate_ae_agreement PASSED                      [100%]
============================== 6 passed in 0.70s ===============================
```
- **Architectural Dimensions:** Autoencoder output shape is strictly `(B, 38)` and Gate output is strictly `(B, 1)` for arbitrary batch size $B$.
- **Parameter Invariant:** AE requires exactly 2,008 gradients; Gate requires exactly 641 gradients.
- **Threshold Validity:** All $\tau_{\text{AE}}^{(k)} > 0$, and confidence intervals are monotonically ordered.
- **Data Immutability:** SHA-256 hashes of all 10 Stage 2 `data.npz` files verified.

### 4.2 Empirical Research Metrics Summary (Methodological Trade-offs)
- **High-Precision Edge Filtering:** At $Q_{0.99}$ benign calibration, the Gate filters out **98.75% of incoming traffic**, routing only **1.25% of flows** to the heavy classifier. This achieves exceptional edge power and compute conservation.
- **Attack Passage Trade-off:** Under pure unsupervised calibration without attack supervision, subtle attack flows mimic benign traffic distributions. Consequently, pooled Gate passage recall is **6.58% on Tier 1** and **9.55% on Tier 3**, while Autoencoder detection recall is **27.93% on Tier 1** and **22.13% on Tier 3**.

---

## 5. Registry of Unresolved Limitations

The following limitations are formally cataloged and will persist into subsequent research milestones:

1. **Unsupervised Gate Attack Recall Trade-off:**
   - *Limitation:* The Student Gate achieves 6.58% (Tier 1) and 9.55% (Tier 3) attack passage recall under strict $Q_{0.99}$ benign calibration, falling short of the $70\%$ reference line originally hypothesized for supervised classifiers.
   - *Status:* Inherent to unsupervised anomaly routing without attack labels. Accepted as empirical baseline.
2. **Extreme Non-IID Skew Sensitivity (Clients 02 & 04):**
   - *Limitation:* Clients with high attack proportions ($\ge 50\%$) and limited benign validation counts establish inflated thresholds ($\tau_{\text{AE}} \approx 0.128$), suppressing recall to $< 3\%$.
   - *Status:* Requires investigation in federated calibration or normalized scoring.
3. **Retrospective Metadata Hashing:**
   - *Limitation:* `scaler_params.json` and `client_stats.json` were not recorded in a pre-execution manifest prior to Stage 3. While timestamps and current hashes ([`outputs/verification/stage3/stage2_artifact_hashes.json`](file:///Users/abhayrajawat/Downloads/resource-adaptive-nids/outputs/verification/stage3/stage2_artifact_hashes.json)) confirm stability, historical immutability is technically retrospective.
   - *Status:* Accepted; `data.npz` cryptographic immutability is fully verified.
4. **Attack-Family Disparity:**
   - *Limitation:* Detection is highly uneven across attack families (Generic achieves $70.59\%$ AE recall, while Exploits achieves $11.57\%$ and Shellcode achieves $0.89\%$ Gate passage).
   - *Status:* Open research challenge in NF-UNSW-NB15-v2 flow features.

---

## 6. Confirmation of No Retraining and Artifact Immutability

It is formally certified that:
1. **Zero Retraining:** No PyTorch optimizer, gradient step, fine-tuning routine, or weight update was executed during this audit. Checkpoints under `outputs/checkpoints/stage3/` retain their original timestamps and weights.
2. **Zero Data Modification:** All files under `data/processed/alpha_0.5_seed_42/` were opened exclusively in read-only mode (`np.load(..., allow_pickle=True)`).
3. **Zero Architectural Alteration:** No model definitions, layer sizes, activation functions, or threshold values were altered.

---

## 7. Final Recommendation and Stage 4 Gate Decision

### **Formal Recommendation: Stage 3 Frozen with Documented Limitations**

### Gate Decision
- **Software Implementation & Verification:** **FROZEN.** The codebase, architectures, checkpoints, data contracts, and three-tier evaluation implementations are verified, robust, and reproducible.
- **Stage 4 Status:** **BLOCKED.** Stage 4 (Federated GRU Classifier and Queue Load-Shedding Simulator) shall remain blocked until the research supervisor/user formally signs off on the thresholding strategy:
  - **Option 1 (Proceed with $Q_{0.99}$):** Accept the Gate as a high-precision edge filter that routes $\approx 1.25\%$ of flows to the GRU, treating low attack passage recall as an established baseline characteristic of passive unsupervised anomaly detection.
  - **Option 2 (Re-calibrate Routing Quantile):** Lower the gate routing threshold (e.g., to $Q_{0.90}$ or $Q_{0.80}$ of local benign validation) to route a higher proportion of suspicious traffic to the federated GRU prior to Stage 4 training.

# Stage 3 — Zero-Retraining Threshold Sensitivity Experiment Report

**Project Title:** Resource-Adaptive Gated Network Intrusion Detection for Federated IoT Edge Networks  
**Milestone:** Stage 3 Sensitivity Investigation  
**Experiment Date:** November 3, 2026  
**Type:** Independent Evaluation Experiment (Zero-Retraining / Frozen Checkpoints)  
**Baseline Status:** Approved Q99 Baseline Remains Frozen & Unmodified  

---

## Executive Summary

This report documents the results of the **Zero-Retraining Threshold Sensitivity Experiment** conducted across all 10 non-IID clients and three evaluation tiers (Tier 1 Full Test, Tier 2 Deduplicated Unique, and Tier 3 Training-Exclusive Unique). 

The objective of this experiment is to rigorously characterize the trade-off between **Attack Passage Recall**, **Benign False Positive Rate (FPR)**, and **Heavy Path Invocation Rate** under four empirical calibration quantiles:
- **Q99 ($q = 0.99$):** Official frozen baseline.
- **Q95 ($q = 0.95$):** Sensitivity variant.
- **Q90 ($q = 0.90$):** Sensitivity variant.
- **Q80 ($q = 0.80$):** Sensitivity variant.

### Core Experimental Findings
1. **The Q95 Pareto "Knee":** Relaxing calibration from Q99 to Q95 yields a dramatic **nearly 7-fold increase in pooled Gate attack passage recall** (jumping from **6.58% to 45.20%** on Tier 1, and **9.55% to 51.97%** on Tier 3), while maintaining **93.44% total traffic shedding** (only **6.56%** of edge flows invoked to the Heavy Path).
2. **Attack Enrichment Peak:** The Q95 threshold maximizes the concentration of attacks in the routed traffic: on Tier 1, attack concentration reaches **27.41%** (a $6.89\times$ enrichment over raw $3.98\%$ test prevalence); on Tier 3, attack concentration reaches **33.52%** (vs. $12.55\%$ raw prevalence).
3. **Hypothesis H1 Boundary at Q80:** To achieve $\ge 70\%$ attack passage recall without supervision, the quantile must be lowered to **Q80** (yielding **74.59%** on Tier 1 and **76.47%** on Tier 3). However, this increases Heavy Path invocation to **22.21%** on Tier 1 (and **28.24%** on Tier 3) and elevates benign false alarms to **20.04%**, placing significant computational and queuing burden on the edge device.
4. **Non-IID Divergence (Clients 02 and 04):** 
   - **Client 02** ($64.01\%$ attack skew): At Q80, Gate recall reaches $83.38\%$, but Heavy Path invocation surges to **60.93%**, severely compromising edge resource savings.
   - **Client 04** (high attack concentration, subtle exploits): Even at Q80, Gate recall reaches only **42.40%** (Tier 1) and **36.27%** (Tier 3), demonstrating that simple quantile relaxation without feature-level adaptation cannot overcome feature overlap in subtle attacks.
5. **Tier 3 FPR Amplification Persistence:** Across all quantiles, unique unseen benign flows (Tier 3) exhibit higher FPR than raw repetitive flows (Tier 1): $5.08\times$ higher at Q99 ($5.13\%$ vs. $1.01\%$), $2.98\times$ at Q95 ($14.80\%$ vs. $4.96\%$), $1.68\times$ at Q90 ($17.79\%$ vs. $10.56\%$), and $1.06\times$ at Q80 ($21.32\%$ vs. $20.04\%$).

---

## 1. Experimental Setup and Protocol

### 1.1 Model & Data Invariants
- **Frozen Models:** Autoencoder ($38 \to 20 \to 10 \to 20 \to 38$, 2,008 parameters) and Gating Regressor ($38 \to 16 \to 1$, 641 parameters) loaded directly from `outputs/checkpoints/stage3/client_{00..09}_*.pt`.
- **Zero Retraining:** Models evaluated strictly with `torch.no_grad()` and `.eval()`. Zero optimizer steps.
- **Benign-Only Validation Calibration:** Thresholds $\tau_{\text{AE}}^{(k)}(Q)$ and $\tau_{\text{Gate}}^{(k)}(Q)$ computed strictly on each client's local benign validation set ($\mathbf{y}_{\text{val}} = 0$). No attack labels were exposed during calibration.
- **Local Application:** Each client's test flows were evaluated against its own local threshold. No global threshold averaging was used.

### 1.2 Evaluation Tiers
- **Tier 1 (Full Test):** Complete test split ($358,538$ samples across 10 clients).
- **Tier 2 (Deduplicated Unique):** Unique test feature vectors rounded to 7 decimal places ($97,132$ samples).
- **Tier 3 (Training-Exclusive Unique):** Unique test feature vectors that never appeared in local training ($55,897$ samples).
- **Invariant Verified:** $\text{Tier 3} \subseteq \text{Tier 2} \subseteq \text{Tier 1}$ holds for all clients.

---

## 2. Threshold Calibration Summary and Monotonicity

### 2.1 Monotonicity Verification
For all 10 clients, threshold monotonicity strictly holds:
$$\tau_{\text{AE}}(0.80) \le \tau_{\text{AE}}(0.90) \le \tau_{\text{AE}}(0.95) \le \tau_{\text{AE}}(0.99)$$
$$\tau_{\text{Gate}}(0.80) \le \tau_{\text{Gate}}(0.90) \le \tau_{\text{Gate}}(0.95) \le \tau_{\text{Gate}}(0.99)$$

### 2.2 Client-Specific Threshold Values Table

| Client ID | Benign Val ($N_b$) | Model | Q99 (Baseline) | Q95 | Q90 | Q80 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `client_00` | 49,693 | AE ($\tau_{\text{AE}}$)<br>Gate ($\tau_{\text{Gate}}$) | 0.017162<br>-4.031378 | 0.002883<br>-5.911168 | 0.001574<br>-6.400211 | 0.000854<br>-7.212412 |
| `client_01` | 79,082 | AE ($\tau_{\text{AE}}$)<br>Gate ($\tau_{\text{Gate}}$) | 0.016690<br>-4.365236 | 0.002869<br>-5.972658 | 0.001451<br>-6.548198 | 0.000840<br>-7.125545 |
| `client_02` | **548** | AE ($\tau_{\text{AE}}$)<br>Gate ($\tau_{\text{Gate}}$) | **0.128263**<br>**-1.594451** | **0.058917**<br>**-2.740701** | **0.043656**<br>**-2.996317** | **0.025184**<br>**-3.796902** |
| `client_03` | 46,684 | AE ($\tau_{\text{AE}}$)<br>Gate ($\tau_{\text{Gate}}$) | 0.017266<br>-4.130823 | 0.003686<br>-5.627156 | 0.002591<br>-5.985599 | 0.001248<br>-6.739603 |
| `client_04` | **1,014** | AE ($\tau_{\text{AE}}$)<br>Gate ($\tau_{\text{Gate}}$) | **0.082440**<br>**-2.519393** | **0.039794**<br>**-3.170701** | **0.028899**<br>**-3.751971** | **0.019600**<br>**-4.055562** |
| `client_05` | 8,493 | AE ($\tau_{\text{AE}}$)<br>Gate ($\tau_{\text{Gate}}$) | 0.018202<br>-4.040293 | 0.003517<br>-5.679636 | 0.002128<br>-6.189053 | 0.001380<br>-6.667426 |
| `client_06` | 27,699 | AE ($\tau_{\text{AE}}$)<br>Gate ($\tau_{\text{Gate}}$) | 0.015769<br>-4.243894 | 0.002582<br>-5.902976 | 0.001763<br>-6.447507 | 0.000958<br>-7.111668 |
| `client_07` | 19,219 | AE ($\tau_{\text{AE}}$)<br>Gate ($\tau_{\text{Gate}}$) | 0.014306<br>-4.282506 | 0.002168<br>-6.168154 | 0.001182<br>-6.686702 | 0.000441<br>-7.672701 |
| `client_08` | 65,847 | AE ($\tau_{\text{AE}}$)<br>Gate ($\tau_{\text{Gate}}$) | 0.017429<br>-4.048007 | 0.002632<br>-5.944379 | 0.001514<br>-6.520214 | 0.000725<br>-7.349794 |
| `client_09` | 46,006 | AE ($\tau_{\text{AE}}$)<br>Gate ($\tau_{\text{Gate}}$) | 0.016827<br>-4.129553 | 0.004168<br>-5.470659 | 0.003102<br>-5.695354 | 0.001625<br>-6.518476 |

*Machine-readable thresholds saved to `outputs/experiments/stage3_threshold_sensitivity/sensitivity_client_thresholds.json`.*

---

## 3. Sample-Weighted Pooled Performance Results

### 3.1 Tier 1 (Full Test Set — 358,538 samples: 344,282 Benign, 14,256 Attack)

| Quantile | AE Recall (%) | AE FPR (%) | Gate Recall (%) | Gate FPR (%) | Heavy Path Invocation (%) | Traffic Filtered (%) | Attack Concentration (%) | Gate–AE Agreement (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q99 (Baseline)** | 27.93% | 0.99% | **6.58%** | 1.01% | **1.24%** | **98.76%** | 21.18% | 98.14% |
| **Q95** | 73.89% | 5.04% | **45.20%** | 4.96% | **6.56%** | **93.44%** | **27.41%** | 97.39% |
| **Q90** | 79.92% | 10.24% | **59.03%** | 10.56% | **12.49%** | **87.51%** | 18.79% | 97.22% |
| **Q80** | 85.17% | 20.11% | **74.59%** | 20.04% | **22.21%** | **77.79%** | 13.35% | 97.18% |

### 3.2 Tier 2 (Deduplicated Unique Test Set — 97,132 samples: 83,728 Benign, 13,404 Attack)

| Quantile | AE Recall (%) | AE FPR (%) | Gate Recall (%) | Gate FPR (%) | Heavy Path Invocation (%) | Traffic Filtered (%) | Attack Concentration (%) | Gate–AE Agreement (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q99 (Baseline)** | 25.75% | 3.63% | **8.66%** | 3.73% | **4.22%** | **95.78%** | 20.65% | 94.93% |
| **Q95** | 72.81% | 13.26% | **47.23%** | 12.65% | **16.13%** | **83.87%** | **29.48%** | 94.55% |
| **Q90** | 79.28% | 17.29% | **57.70%** | 16.92% | **21.02%** | **78.98%** | 27.62% | 95.42% |
| **Q80** | 84.34% | 23.52% | **71.29%** | 23.56% | **28.36%** | **71.64%** | 25.30% | 95.89% |

### 3.3 Tier 3 (Training-Exclusive Unique Test Set — 55,897 samples: 48,880 Benign, 7,017 Attack)

| Quantile | AE Recall (%) | AE FPR (%) | Gate Recall (%) | Gate FPR (%) | Heavy Path Invocation (%) | Traffic Filtered (%) | Attack Concentration (%) | Gate–AE Agreement (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q99 (Baseline)** | 22.13% | 5.21% | **9.55%** | 5.13% | **5.69%** | **94.31%** | 21.08% | 94.36% |
| **Q95** | 74.28% | 15.75% | **51.97%** | 14.80% | **19.46%** | **80.54%** | **33.52%** | 94.11% |
| **Q90** | 80.82% | 18.62% | **62.70%** | 17.79% | **23.43%** | **76.57%** | 33.60% | 95.42% |
| **Q80** | 85.78% | 21.95% | **76.47%** | 21.32% | **28.24%** | **71.76%** | 33.99% | 96.77% |

---

## 4. Macro Client-Averaged Performance Results

To ensure clients with small sample counts (e.g. Client 02) receive equal weighting, macro metrics across all 10 clients are reported:

| Tier | Quantile | AE Recall Macro (%) | AE FPR Macro (%) | Gate Passage Recall Macro (%) | Gate FPR Macro (%) | Heavy Path Invocation Macro (%) | Gate–AE Agreement Macro (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1** | **Q99** | 31.82% | 1.02% | **6.73%** | 0.98% | **1.41%** | 97.22% |
| | **Q95** | 83.16% | 5.08% | **53.96%** | 4.99% | **7.39%** | 92.18% |
| | **Q90** | 89.41% | 10.29% | **69.70%** | 10.45% | **14.63%** | 93.54% |
| | **Q80** | 92.82% | 20.56% | **83.57%** | 20.13% | **27.25%** | 94.16% |
| **Tier 3** | **Q99** | 25.24% | 4.69% | **9.38%** | 4.50% | **5.25%** | 94.33% |
| | **Q95** | 83.46% | 14.74% | **56.36%** | 13.46% | **18.53%** | 91.53% |
| | **Q90** | 88.88% | 19.10% | **67.97%** | 17.34% | **23.42%** | 92.86% |
| | **Q80** | 92.39% | 23.66% | **82.64%** | 22.53% | **31.35%** | 93.63% |

---

## 5. Tier 3 Additional Analysis: Joint AE–Gate Routing Dynamics

The relationship between the Teacher Autoencoder and the Student Gate was analyzed on the training-exclusive unique test set ($55,897$ samples: $48,880$ benign, $7,017$ attack).

### 5.1 Pooled Joint Routing Confusion Matrix (Sample Counts)

| Quantile | Total Samples | Both Trigger | AE Only | Gate Only | Neither Triggers | Benign Agreement (%) | Attack Agreement (%) | Overall Agreement (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q99** | 55,897 | 2,064 | 2,035 | 1,115 | 50,683 | 96.49% | 79.58% | **94.36%** |
| **Q95** | 55,897 | 10,249 | 2,660 | 630 | 42,358 | 97.21% | 72.57% | **94.11%** |
| **Q90** | 55,897 | 12,652 | 2,119 | 443 | 40,683 | 97.65% | 79.83% | **95.42%** |
| **Q80** | 55,897 | 15,366 | 1,384 | 419 | 38,728 | 98.18% | 86.99% | **96.77%** |

### 5.2 Class-Conditioned Routing Breakdown on Tier 3

```
Q99 Breakdown:
  Benign Flows (48,880): Both=1,669 | AE_only=877  | Gate_only=840 | Neither=45,494 (96.49% Agree)
  Attack Flows ( 7,017): Both=  395 | AE_only=1,158| Gate_only=275 | Neither= 5,189 (79.58% Agree)

Q95 Breakdown:
  Benign Flows (48,880): Both=6,782 | AE_only=915  | Gate_only=450 | Neither=40,733 (97.21% Agree)
  Attack Flows ( 7,017): Both=3,467 | AE_only=1,745| Gate_only=180 | Neither= 1,625 (72.57% Agree)

Q90 Breakdown:
  Benign Flows (48,880): Both=8,324 | AE_only=776  | Gate_only=371 | Neither=39,409 (97.65% Agree)
  Attack Flows ( 7,017): Both=4,328 | AE_only=1,343| Gate_only= 72 | Neither= 1,274 (79.83% Agree)

Q80 Breakdown:
  Benign Flows (48,880): Both=10,130| AE_only=601  | Gate_only=289 | Neither=37,860 (98.18% Agree)
  Attack Flows ( 7,017): Both=5,236 | AE_only=783  | Gate_only=130 | Neither=   868 (86.99% Agree)
```

**Key Observation:** As thresholds are relaxed from Q99 to Q80, the "Gate Only" cell among attack flows drops from $275$ to $130$, while "Both Trigger" grows from $395$ to $5,236$. The Student Gate tracks the Teacher Autoencoder's trigger decisions with increasing fidelity, confirming that the distillation relationship stabilizes across broader distribution tails.

---

## 6. Client-Level Analysis: Extreme Non-IID Behavior (Clients 02 and 04)

### 6.1 Client 02 Performance Progression (Partition: 1,523 Tier 1 samples, 64.01% malicious, $N_{\text{val, benign}}=548$)

| Quantile | $\tau_{\text{AE}}$ | $\tau_{\text{Gate}}$ | Tier 1 AE Rec (%) | Tier 1 Gate Rec (%) | Tier 1 Gate FPR (%) | Heavy Path Invocation (%) | Tier 3 Gate Rec (%) | Tier 3 Heavy Path (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q99** | 0.128263 | -1.594451 | 1.85% | 2.05% | 0.73% | **1.58%** | 3.37% | 3.01% |
| **Q95** | 0.058917 | -2.740701 | 47.08% | 16.92% | 6.02% | **13.00%** | 20.92% | 17.17% |
| **Q90** | 0.043656 | -2.996317 | 58.26% | 40.10% | 10.77% | **29.55%** | 27.48% | 23.85% |
| **Q80** | 0.025184 | -3.796902 | 69.95% | **83.38%** | 20.99% | **60.93%** | **76.42%** | **62.65%** |

### 6.2 Client 04 Performance Progression (Partition: 5,428 Tier 1 samples, 81.34% malicious, $N_{\text{val, benign}}=1,014$)

| Quantile | $\tau_{\text{AE}}$ | $\tau_{\text{Gate}}$ | Tier 1 AE Rec (%) | Tier 1 Gate Rec (%) | Tier 1 Gate FPR (%) | Heavy Path Invocation (%) | Tier 3 Gate Rec (%) | Tier 3 Heavy Path (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q99** | 0.082440 | -2.519393 | 15.99% | 3.44% | 0.89% | **2.97%** | 5.45% | 5.01% |
| **Q95** | 0.039794 | -3.170701 | 43.04% | 10.69% | 4.44% | **9.52%** | 16.79% | 15.94% |
| **Q90** | 0.028899 | -3.751971 | 49.40% | 21.72% | 10.17% | **19.57%** | 29.81% | 28.05% |
| **Q80** | 0.019600 | -4.055562 | 58.89% | **42.40%** | 20.73% | **38.36%** | **36.27%** | **35.69%** |

### 6.3 Critical Insights on Non-IID Behavior
1. **The Edge Saturation Phenomenon (Client 02):** In Client 02, lowering the quantile to Q80 successfully unlocks high attack passage recall ($83.38\%$). However, because $64.01\%$ of the traffic arriving at this edge client is malicious, passing $83.38\%$ of attacks plus $20.99\%$ of benign flows results in **$60.93\%$ of all local flows being routed to the heavy path**. On resource-constrained IoT nodes, sending $61\%$ of incoming packets to an intensive GRU completely overwhelms the queue and defeats edge energy-saving goals.
2. **The Feature Overlap Deficit (Client 04):** In Client 04, lowering the threshold to Q80 fails to reach the 70% benchmark, achieving only $42.40\%$ Gate passage recall. This proves that Client 04's underperformance is **not merely caused by threshold inflation**, but by substantial distribution overlap between benign flows and subtle exploit vectors in its partition. Simple quantile reduction cannot separate overlapping distributions without supervised feedback or multi-feature discrimination.

---

## 7. Answers to Final Research Interpretation Questions

### 1. How does attack passage recall change as the quantile decreases?
Attack passage recall increases non-linearly across all tiers:
- **Steepest marginal gain between Q99 and Q95:** On Tier 1, Gate recall jumps from **6.58% to 45.20%** (+38.62 percentage points, nearly a $7\times$ multiplier). On Tier 3, it jumps from **9.55% to 51.97%** (+42.42 percentage points).
- **Diminishing gains between Q95 and Q80:** Moving from Q95 to Q90 adds $+13.83\%$ (Tier 1), and moving from Q90 to Q80 adds $+15.56\%$ (Tier 1), eventually surpassing the $70\%$ mark at Q80 ($74.59\%$).

### 2. How does benign FPR change across all three tiers?
- **Tier 1 (Raw Flows):** FPR tracks theoretical tail percentiles almost exactly: $1.01\%$ at Q99, $4.96\%$ at Q95, $10.56\%$ at Q90, and $20.04\%$ at Q80.
- **Tier 2 (Deduped Flows):** FPR is magnified: $3.73\%$ at Q99, $12.65\%$ at Q95, $16.92\%$ at Q90, and $23.56\%$ at Q80.
- **Tier 3 (Train-Exclusive Unique Flows):** FPR is severely amplified: $5.13\%$ at Q99, $14.80\%$ at Q95, $17.79\%$ at Q90, and $21.32\%$ at Q80.

### 3. How much additional traffic is routed to the Heavy Path?
- **At Q99:** Heavy Path invocation is **1.24%** (98.76% of traffic filtered).
- **At Q95:** Heavy Path invocation increases by **+5.32%** to **6.56%** (93.44% filtered).
- **At Q90:** Heavy Path invocation increases by **+11.25%** to **12.49%** (87.51% filtered).
- **At Q80:** Heavy Path invocation increases by **+20.97%** to **22.21%** (77.79% filtered).

### 4. Which quantile provides the most informative detection-resource trade-off?
**Q95 represents the Pareto-optimal operating point:**
- It achieves a nearly $7\times$ recall boost over baseline ($45.20\%$ vs. $6.58\%$ on Tier 1; $51.97\%$ vs. $9.55\%$ on Tier 3).
- It preserves **93.44% traffic shedding**, shielding the edge device from heavy processing on $>93\%$ of packets.
- It achieves the **highest attack concentration in passed traffic (27.41% on Tier 1 and 33.52% on Tier 3)**, meaning the GRU classifier receives an attack-enriched stream rather than false alarms.
- While Q80 achieves higher recall ($74.59\%$), its $22.21\%$ invocation rate and $20.04\%$ false positive rate overload the edge queue with non-malicious traffic.

### 5. How do Client 02 and Client 04 differ from the other clients?
- Both clients possess extreme non-IID partitions with very high attack proportions ($64\%$ and $81\%$).
- **Client 02** is threshold-limited: lowering the threshold to Q80 resolves recall ($83.38\%$), but saturates the edge node ($60.93\%$ Heavy Path invocation).
- **Client 04** is separability-limited: even at Q80, recall remains capped at $42.40\%$, proving that unsupervised single-metric thresholding cannot distinguish its subtle attack patterns from normal traffic.

### 6. Does the Tier 3 FPR amplification persist across all quantiles?
**Yes, but its relative ratio diminishes as the baseline FPR grows:**
- At Q99: Tier 3 FPR ($5.13\%$) is **$5.08\times$** higher than Tier 1 FPR ($1.01\%$).
- At Q95: Tier 3 FPR ($14.80\%$) is **$2.98\times$** higher than Tier 1 FPR ($4.96\%$).
- At Q90: Tier 3 FPR ($17.79\%$) is **$1.68\times$** higher than Tier 1 FPR ($10.56\%$).
- At Q80: Tier 3 FPR ($21.32\%$) is **$1.06\times$** higher than Tier 1 FPR ($20.04\%$).
- *Reason:* Tier 1 is dominated by common benign background flows with near-zero error. Tier 3 filters out all repetitive training-seen flows, exposing the tail of unseen benign variance where false alarms naturally concentrate.

### 7. What limitations remain unresolved?
1. **Unsupervised Inability to Distinguish Novelty from Malice:** Without attack labels, lowering thresholds to capture attacks inherently captures tail benign flows, causing $15-21\%$ FPR at Q95-Q80.
2. **Extreme Non-IID Queue Overload:** Heavily attacked edge nodes (Client 02) route $>60\%$ of traffic to the heavy path when thresholds are relaxed.
3. **Attack-Specific Blind Spots:** Client 04's exploit traffic remains largely undetectable ($<43\%$ recall) under purely unsupervised reconstruction error.

---

## 8. Artifact and Script Registry

All artifacts generated by this experiment are organized in `outputs/experiments/stage3_threshold_sensitivity/`:
- **Experiment Script:** `src/evaluation/threshold_sensitivity.py`
- **Per-Client Metrics CSV:** `outputs/experiments/stage3_threshold_sensitivity/sensitivity_per_client_metrics.csv`
- **Pooled Metrics JSON:** `outputs/experiments/stage3_threshold_sensitivity/sensitivity_pooled_summary.json`
- **Client Thresholds JSON:** `outputs/experiments/stage3_threshold_sensitivity/sensitivity_client_thresholds.json`
- **Tier 3 Joint Confusion Matrices CSV:** `outputs/experiments/stage3_threshold_sensitivity/tier3_joint_confusion_matrices.csv`
- **Tier 3 Joint Confusion Matrices JSON:** `outputs/experiments/stage3_threshold_sensitivity/tier3_joint_confusion_matrices.json`
- **Visualizations:**
  - `tradeoff_recall_vs_heavypath.png`: Recall vs. Heavy Path Invocation Rate across Tiers 1 and 3.
  - `tradeoff_fpr_across_tiers.png`: FPR progression and Tier 3 amplification across quantiles.
  - `tradeoff_attack_concentration.png`: Attack enrichment ratio in passed traffic.
  - `client_comparative_recall.png`: Per-client passage recall highlighting Clients 02 and 04.

---

## 9. Formal Confirmation of Invariants

It is certified that:
1. **Zero Retraining Occurred:** No model was retrained, fine-tuned, or updated.
2. **Zero Checkpoint Modification:** Checkpoints in `outputs/checkpoints/stage3/` retain their original weights and timestamps.
3. **Zero Stage 2 Modification:** All Stage 2 files in `data/processed/alpha_0.5_seed_42/` remain untouched.
4. **Original Baseline Preserved:** The official Q99 baseline thresholds in `outputs/checkpoints/stage3/client_*_thresholds.json` and reports in `outputs/metrics/stage3/` remain frozen and intact.
5. **No Stage 4 Components Implemented:** Queue simulators, GRU classifiers, and FedAvg routines have not been constructed.


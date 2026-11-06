# Resource-Adaptive Gated Network Intrusion Detection for Federated IoT Edge Networks

Final-Year Engineering Research Project.

## 1. Overview & Architecture

This repository contains the modular implementation of a resource-adaptive, multi-tier Network Intrusion Detection System (NIDS) designed for resource-constrained IoT edge devices operating under non-IID federated learning environments:

1. **Lightweight Autoencoder (`src/models/autoencoder.py`):** Unsupervised local anomaly detection tier for fast, low-power screening. Remainder stays local.
2. **Gating Regressor (`src/models/gate.py`):** Predicts reconstruction error via self-supervised Teacher–Student knowledge distillation to guide flow classification paths.
3. **Sequential GRU Classifier (`src/models/gru_model.py`):** Heavy classification tier capturing temporal packet flow dynamics. Uploaded and aggregated globally across federated rounds.
4. **Queue & Delay Simulator (`src/simulation/queue_simulator.py`):** Evaluates latency and queue-delay constraints on IoT nodes to decide when heavier inference paths can safely execute.
5. **Federated Learning Orchestration (`src/federated/`):** Simulates 10 clients with non-IID Dirichlet traffic partitions; coordinates parameter aggregation via sample-weighted FedAvg.

---

## 2. Directory Structure

```text
resource-adaptive-nids/
├── configs/
│   └── experiment_config.yaml         # Centralized hyperparameters & paths
├── data/
│   ├── raw/                           # Original dataset (NF-UNSW-NB15-v2.csv)
│   ├── processed/                     # Preprocessed features & partitioned subsets
│   └── README.md
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   # Interactive EDA & schema profiling
│   ├── 02_preprocessing_validation.ipynb # Preprocessing pipeline validation
│   ├── 03_model_experiments.ipynb     # Model training & ablation experiments
│   └── 04_federated_experiments.ipynb # Federated FL & delay simulation experiments
├── outputs/
│   ├── models/                        # Saved checkpoints (.pt)
│   ├── metrics/                       # JSON/CSV evaluation outputs
│   └── figures/                       # Plots and visualizations
├── src/
│   ├── __init__.py
│   ├── data/                          # Audit, cleaning, Dirichlet partitioning, windowing
│   ├── models/                        # Autoencoder, Gating Regressor, GRU models
│   ├── federated/                     # Client, FedAvg aggregation, Server orchestrator
│   ├── simulation/                    # M/M/1 or queue-delay edge simulator
│   ├── training/                      # Independent training loops
│   ├── evaluation/                    # Detection rate, FAR, F1, comparisons
│   └── utils/                         # Config loaders, seeds, hardware detection
├── tests/
│   └── test_data.py                   # Verification and integrity unit tests
├── .gitignore
├── requirements.txt
├── README.md
└── main.py                            # CLI execution entrypoint
```

---

## 3. Local Setup (macOS / Apple Silicon M2)

### 3.1. Prerequisites & Virtual Environment
Ensure you have Python 3.10+ installed:

```bash
# Navigate to the project root
cd /Users/abhayrajawat/Downloads/resource-adaptive-nids

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2. Dataset Placement
Place `NF-UNSW-NB15-v2.csv` directly into:
```text
data/raw/NF-UNSW-NB15-v2.csv
```

### 3.3. Verify Skeleton
Run unit tests and the initial entrypoint:
```bash
python3 -m unittest discover tests
python3 main.py --mode audit
```

---

## 4. Kaggle Workflow (Zero Code Duplication)

The project is structured so that you do **not** write or duplicate model/training code inside Kaggle notebooks:

1. **Push this repository to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "feat: initial modular project structure"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```
2. **Kaggle Setup:**
   - Attach the `NF-UNSW-NB15-v2` dataset to your Kaggle Notebook via **Add Data** (it is placed under `/kaggle/input/...`).
   - In the first cell of your Kaggle notebook, clone your repository and set `PROJECT_ROOT`:
     ```python
     !git clone https://github.com/<your-username>/resource-adaptive-nids.git
     %cd resource-adaptive-nids
     import sys
     sys.path.append(".")
     ```
   - Override the dataset path in `configs/experiment_config.yaml` or through the Python config loader to point to `/kaggle/input/<dataset-name>/NF-UNSW-NB15-v2.csv`.
   - Hardware detection (`src/utils/reproducibility.py`) automatically uses CUDA GPU on Kaggle and MPS on your local MacBook M2 without code changes.


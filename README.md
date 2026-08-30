# ICU Clinical Decision Support System
### Multimodal Intelligence System with Federated Learning for Continuous Patient Monitoring and Early Deterioration Detection

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Configuration](#configuration)
7. [Running the Application](#running-the-application)
8. [Federated Learning Training](#federated-learning-training)
9. [Real FL Demo (Server + Clients)](#real-fl-demo-server--clients)
10. [Application Features](#application-features)
11. [Model Details](#model-details)
12. [Dataset Information](#dataset-information)
13. [Technology Stack](#technology-stack)
14. [Troubleshooting](#troubleshooting)

---

## Project Overview

This system is an AI-powered ICU Clinical Decision Support System (CDSS) that:

- Continuously monitors ICU patients using multimodal data (vital signs, computer vision features, clinical notes)
- Predicts the patient's **SOFA score** (Sequential Organ Failure Assessment, 0–24) as a clinical severity measure
- Classifies risk as **Low** (SOFA < 5) / **Moderate** (5–9) / **High** (≥ 10)
- Fires a clinical alert when predicted SOFA ≥ 8 (threshold lowered to compensate for model under-prediction)
- Provides **SHAP explainability** showing which features drove the prediction
- Generates a structured **LLM clinical assessment** (condition, cause, forecast, actions) using Groq
- Validates LLM reliability via a **self-consistency check** (3 responses + cosine similarity)
- Preserves patient privacy through **Federated Learning** (model trained across 3 hospital nodes without sharing raw data)

> **Important:** This system is a decision support tool only. It does not diagnose disease or replace clinical judgment.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Multimodal Input                           │
│  IoMT Vitals │ Computer Vision Features │ Clinical Notes     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Real-Time Processing Pipeline                   │
│  Sliding Window (20 readings) → Trend Features (9)          │
│  Latest Vitals (7) │ TF-IDF (600) │ CV Features (2)        │
└──────────────────────┬──────────────────────────────────────┘
                       │  618 features
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Federated DNN  (618→256→128→64→1)                  │
│          Trained via Flower FedAvg across 3 hospitals        │
│          AdamW + Linear Weighted MSE Loss                    │
└───────────┬──────────────────────────────────────────────────┘
            │  SOFA Score (0–24)
            ▼
┌───────────────────────┐   ┌─────────────────────────────────┐
│  Risk Classification  │   │     SHAP DeepExplainer          │
│  Alert if SOFA ≥ 8   │   │  Top-7 clinical features        │
└───────────┬───────────┘   └──────────────┬──────────────────┘
            └──────────────┬───────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         Groq LLaMA 3.3 70B — Clinical Explanation           │
│  3 responses → TF-IDF cosine similarity → Reliability score │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12+ |
| PyTorch | 2.0+ |
| scikit-learn | 1.6+ |
| Flower | 1.8+ |
| Streamlit | 1.35+ |
| SHAP | 0.44+ |
| Groq API Key | Required for LLM |

---

## Installation

### Step 1 — Clone / Download the project

Ensure the `icu_monitor/` folder is on your machine.

### Step 2 — Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

### Step 3 — Install dependencies

```bash
cd icu_monitor
pip install -r requirements.txt
```

The `requirements.txt` installs:

```
streamlit>=1.35.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.6.0,<2.0.0
joblib>=1.3.0
shap>=0.44.0
groq>=0.9.0
python-dotenv>=1.0.0
plotly>=5.18.0
flwr[simulation]>=1.8.0
```

> Note: `flwr[simulation]` installs Flower with Ray-based simulation support. This may take a few minutes.

### Step 4 — Configure the Groq API key

Create a `.env` file in the `icu_monitor/` directory:

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free API key at [https://console.groq.com](https://console.groq.com)

> If no `.env` is set, the app shows an API key input field in the sidebar.

---

## Project Structure

```
icu_monitor/
│
├── app.py                    ← Main Streamlit application
├── model_utils.py            ← ICUModel architecture + training utilities
├── train_federated.py        ← FL training script (simulation mode)
├── server.py                 ← Flower FL server (real FL demo)
├── client.py                 ← Flower FL client (real FL demo)
│
├── requirements.txt
├── .env                      ← Groq API key (create this)
├── .env.example              ← Template
├── README.md
├── SKILL.md                  ← Complete technical context document
│
├── models/
│   ├── federated_model.pth   ← Trained PyTorch DNN weights
│   ├── scaler.pkl            ← StandardScaler (618 features)
│   ├── tfidf_vectorizer.pkl  ← TF-IDF vectorizer (600 terms, ngram 1-2)
│   ├── feature_columns.pkl   ← Ordered list of 618 feature names
│   ├── shap_background.npy   ← 300 background samples for SHAP
│   ├── patient_vitals.csv    ← Sliding window vitals history (20 rows)
│   ├── prediction_history.csv← SOFA prediction log (last 50)
│   └── training_metadata.json← Training config and performance metrics
│
├── data/
│   ├── client_0.csv          ← Hospital 0 training data (~15,889 rows)
│   ├── client_1.csv          ← Hospital 1 training data (~15,890 rows)
│   └── client_2.csv          ← Hospital 2 training data (~16,372 rows)
│
└── notebooks/
    └── federated_learning.ipynb  ← Original training notebook (Colab/BigQuery)
```

---

## Configuration

### App Configuration (`app.py`)

| Constant | Default | Description |
|---|---|---|
| `ALERT_THRESHOLD` | `8.0` | SOFA score at which the red alert fires. Set lower than the clinical ≥ 10 boundary to compensate for model under-prediction (doubles High Risk recall from 28% → 52%) |

### Training Configuration (`train_federated.py`)

| Constant | Default | Description |
|---|---|---|
| `NUM_ROUNDS` | `20` | Number of FL aggregation rounds |
| `EPOCHS_PER_ROUND` | `10` | Local training epochs per hospital per round |
| `BATCH_SIZE` | `64` | Mini-batch size for DataLoader |
| `BASE_LR` | `0.001` | AdamW learning rate |
| `LR_DECAY` | `1.0` | Set < 1.0 to enable per-round lr decay (tested: 0.97 hurt performance) |
| `GRAD_CLIP` | `None` | Gradient clipping max norm (tested: disabled is better for this problem) |
| `OVERSAMPLE` | `False` | WeightedRandomSampler for high-risk oversampling (tested: caused overfitting) |
| `USE_NONIID_SPLIT` | `False` | Enable biased hospital split by SOFA severity (tested: causes client drift, lower R²) |
| `USE_DP` | `False` | Enable Differential Privacy (clip + add Gaussian noise before sending weights) |
| `DP_SENSITIVITY` | `1.0` | DP clipping threshold (max L2 norm of model update) |
| `DP_SIGMA` | `1.0` | DP noise multiplier (noise std = sigma × sensitivity) |
| `SHAP_BG_SAMPLES` | `300` | Number of background samples for SHAP DeepExplainer |

### Real FL Configuration (`client.py`)

| Constant | Default | Description |
|---|---|---|
| `USE_DP` | `False` | Enable DP in real FL clients (must match server intent) |
| `DP_SENSITIVITY` | `1.0` | Clipping threshold |
| `DP_SIGMA` | `1.0` | Noise multiplier |
| `EPOCHS` | `10` | Local training epochs per round |

---

## Running the Application

### Launch the Streamlit app

```bash
cd icu_monitor
streamlit run app.py
```

Opens at **http://localhost:8501**

### Using the app

1. Enter patient data in the **sidebar**:
   - Clinical notes (free text: history, medications, observations)
   - Vital signs: HR, RR, SpO₂, Temperature, SBP, DBP, MAP
   - GCS Eye Opening (1–4)
   - Stress Score (0–10)

2. Click **🚀 Run Prediction**

3. Review results across 4 tabs:
   - **📊 Risk Assessment** — SOFA score, severity bar, vital sign trend chart, prediction history
   - **🔍 Explainability** — SHAP feature importance, clinical interpretations, key risk factors
   - **🧠 AI Clinical Report** — LLM assessment with consistency score and reliability rating
   - **🔒 Federated Learning** — FL training configuration, model architecture, DP status

### Utility buttons (sidebar)

- **🔄 Reset Patient History** — Clears the 20-reading sliding window. Use when switching to a new patient.
- **ℹ️ Model Information** — Expandable panel showing MAE, R², training details.

---

## Federated Learning Training

### Retrain the model locally (simulation mode)

Uses the existing client CSV files in `data/`. No Google BigQuery or internet connection required.

```bash
cd icu_monitor
python train_federated.py
```

Training takes approximately **1–3 minutes** on a modern CPU.

**What it does:**
1. Loads `data/client_0/1/2.csv` (3 simulated hospitals, ~48,000 samples total)
2. Saves 300 SHAP background samples → `models/shap_background.npy`
3. Runs Flower FL simulation: 20 rounds × 10 epochs × 3 hospitals
4. Saves best global model → `models/federated_model.pth`
5. Evaluates on held-out test set and prints MAE / R²
6. Saves training metadata → `models/training_metadata.json`

**After retraining, restart the app** to load the new model:

```bash
streamlit run app.py
```

### Enable experimental options before retraining

Open `train_federated.py` and change any config constant, then run:

```python
# Enable Differential Privacy
USE_DP = True

# Enable Non-IID hospital split (note: reduces global R² due to client drift)
USE_NONIID_SPLIT = True

# More training
NUM_ROUNDS = 30
EPOCHS_PER_ROUND = 15
```

---

## Real FL Demo (Server + Clients)

This demonstrates the actual Flower federated learning protocol with separate server and client processes — simulating hospitals on different machines (all on localhost for demo).

### Requirements

4 terminal windows, all inside `icu_monitor/`

### Step 1 — Start the server (Terminal 1)

```bash
cd icu_monitor
python server.py
```

Output:
```
==================================================
  ICU Federated Learning Server
==================================================
  Address : 127.0.0.1:8080
  Rounds  : 20
  Waiting for 3 hospital clients...
```

### Step 2 — Connect Hospital 0 (Terminal 2)

```bash
python client.py --client_id 0
```

### Step 3 — Connect Hospital 1 (Terminal 3)

```bash
python client.py --client_id 1
```

### Step 4 — Connect Hospital 2 (Terminal 4)

```bash
python client.py --client_id 2
```

Training begins automatically once all 3 clients connect. The server saves:
- `models/federated_model.pth` — latest model (updated every round)
- `models/federated_model_best.pth` — best model by eval loss
- `models/training_metadata.json` — configuration snapshot

### Enable DP for the real FL demo

In `client.py`, set `USE_DP = True` before running. Each client will clip its model update and add Gaussian noise before transmitting to the server.

---

## Application Features

### Input Validation

All vital signs are validated against physiologically possible ranges:

| Vital | Valid Range |
|---|---|
| HR | 30–220 bpm |
| RR | 5–60 breaths/min |
| SpO₂ | 50–100 % |
| Temperature | 30–43 °C |
| SBP | 40–250 mmHg |
| DBP | 20–150 mmHg |
| MAP | 30–200 mmHg |
| GCS Eye | 1–4 |
| Stress Score | 0–10 |

Invalid inputs block the prediction and show an error.

### Sliding Window

The app maintains the last 20 vital sign readings in `models/patient_vitals.csv`.

- Each prediction appends the current vitals and removes the oldest row
- Trend features (mean, std, min) are recomputed from this 20-row window
- **Reset button** clears the history for a new patient

### Alert System

| Predicted SOFA | Alert |
|---|---|
| ≥ 8 | 🔴 High Risk — red banner + clinical advisory panel |
| 5–7.9 | 🟡 Moderate Risk — yellow warning banner |
| < 5 | No alert |

The alert fires at ≥ 8 (not ≥ 10) because the model under-predicts severe cases by ~2–3 SOFA points due to limited high-risk training data.

### LLM Self-Consistency Check

To assess LLM reliability:
1. The same prompt is sent to Groq 3 times (temperature=0.7)
2. Responses are vectorised with TF-IDF
3. Pairwise cosine similarity → average = consistency score

| Score | Label |
|---|---|
| ≥ 0.85 | ✅ High Reliability |
| 0.65–0.85 | ⚠️ Moderate Reliability |
| < 0.65 | ❌ Low Reliability |

### Prediction History

Every prediction is logged to `models/prediction_history.csv` (keeps last 50). Visible as a table in Tab 1 under the trend chart once 2+ predictions have been made.

---

## Model Details

### Architecture

```
Input: 618 features
  ├─ 9 trend features  (HR_mean, HR_std, RR_mean, SpO₂_mean, SpO₂_min,
  │                     Temp_mean, SBP_mean, DBP_mean, MAP_mean)
  ├─ 7 latest vitals   (latest_HR, latest_RR, latest_SpO₂, latest_Temp,
  │                     latest_SBP, latest_DBP, latest_MAP)
  ├─ 2 CV features     (GCS_eye_opening, stress_score)
  └─ 600 TF-IDF        (clinical note terms + bigrams)

DNN:  618 → Linear(256) → ReLU
           → Linear(128) → ReLU
           → Linear(64)  → ReLU
           → Linear(1)
           → SOFA Score (0–24, direct output, no scaling)
```

### Performance (held-out test set)

| Metric | Value |
|---|---|
| MAE | ~2.05 SOFA points |
| R² | ~0.25 |
| High Risk recall (SOFA ≥ 10, alert ≥ 8) | ~52% |
| Prediction range | 0.4 – 18.5 |

### Training

- **Optimizer:** AdamW (`lr=0.001`, `weight_decay=1e-4`)
- **Loss:** Linear weighted MSE — `weight = 1 + SOFA × 3` (high-SOFA patients weighted ~2-5×)
- **FL Algorithm:** FedAvg (Flower framework)
- **Data:** 48,151 MIMIC-III ICU samples split across 3 simulated hospitals

### SHAP Explainability

Uses `shap.DeepExplainer` with 300 background samples.

- Model is set to `eval()` mode — Dropout (if any) is disabled, SHAP is deterministic
- Top 7 clinically relevant features are selected by SHAP absolute impact
- Each feature is mapped to a human-readable clinical interpretation

---

## Dataset Information

### Source

MIMIC-III (Medical Information Mart for Intensive Care III), accessed via Google BigQuery.

**BigQuery project:** `mimic-icu-risk-prediction`
**Processed table:** `mimic-icu-risk-prediction.processed_mimic.ml_dataset_final`

### What the client CSVs contain

Each `data/client_N.csv` file contains pre-processed, scaled training data:

- **Rows:** ~15,890–16,372 ICU measurement windows
- **Columns:** 618 feature columns (already StandardScaler-normalised) + `sofa_score` target
- The scaler was fit during preprocessing — do not re-scale these CSVs
- **Size:** ~191–197 MB each

### Cannot be regenerated locally

The `models/scaler.pkl` and `models/tfidf_vectorizer.pkl` were fitted on the full MIMIC-III dataset via Google BigQuery (SQL queries in `sql_queries.pdf`). They cannot be regenerated without BigQuery access. The client CSVs are the training data.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| DNN Framework | PyTorch 2.x |
| Federated Learning | Flower (flwr) 1.8+ |
| Explainability | SHAP 0.44+ |
| LLM Provider | Groq (`llama-3.3-70b-versatile`) |
| LLM Self-Consistency | scikit-learn TF-IDF + cosine similarity |
| Web Framework | Streamlit 1.35+ |
| Visualisation | Plotly |
| Data Processing | pandas, numpy |
| Environment Config | python-dotenv |
| Dataset | MIMIC-III (Google BigQuery) |

---

## Troubleshooting

### App fails to start — `ModuleNotFoundError`

```bash
pip install -r requirements.txt
```

If `flwr[simulation]` fails to install:

```bash
pip install flwr
pip install "flwr[simulation]"
```

### sklearn version warning on startup

```
InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.6.1 when using version X.X
```

This is expected and harmless. The scaler was fitted with sklearn 1.6.1 but StandardScaler is stable across 1.x versions. The app functions correctly.

### SHAP tab shows "SHAP background data not found"

Run the training script once to generate it:

```bash
python train_federated.py
```

Then restart the app.

### Groq API error / LLM unavailable

- Check your `.env` file has the correct `GROQ_API_KEY`
- Verify your key is active at [https://console.groq.com](https://console.groq.com)
- The app continues to work without LLM (SOFA score and SHAP still function)

### App loads but SOFA prediction is 0 or negative

The sliding window `models/patient_vitals.csv` may be corrupted. Reset it:

```bash
rm models/patient_vitals.csv
```

The app will recreate it with default values on next startup.

### Training crashes with `RuntimeError: Simulation crashed`

Flower's simulation requires Ray. Ensure `flwr[simulation]` (not just `flwr`) is installed:

```bash
pip install "flwr[simulation]"
```

### Real FL training — clients cannot connect to server

Ensure you start `server.py` FIRST and wait for the "Waiting for 3 hospital clients..." message before launching clients. All 4 processes must run from the `icu_monitor/` directory.

### Training gives R² < 0 (diverging model)

This can happen due to FL stochasticity. Simply run training again:

```bash
python train_federated.py
```

The `SaveBestStrategy` always saves the round with the lowest validation loss, so the saved model is the best checkpoint even if later rounds diverge.

---

## Quick Start Summary

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Run the app
streamlit run app.py

# 4. (Optional) Retrain the federated model
python train_federated.py

# 5. (Optional) Real FL demo — run in 4 separate terminals
python server.py
python client.py --client_id 0
python client.py --client_id 1
python client.py --client_id 2
```

---

*This project was developed as a Final Year Capstone in Computer Science Engineering.*
*Dataset: MIMIC-III | Framework: Flower | LLM: Groq Llama 3.3 70B*

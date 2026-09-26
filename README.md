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
8. [Computer Vision Monitor](#computer-vision-monitor)
9. [Federated Learning Training](#federated-learning-training)
10. [Real FL Demo (Server + Clients)](#real-fl-demo-server--clients)
11. [Application Features](#application-features)
12. [Model Details](#model-details)
13. [Dataset Information](#dataset-information)
14. [Technology Stack](#technology-stack)
15. [Troubleshooting](#troubleshooting)

---

## Project Overview

This system is an AI-powered ICU Clinical Decision Support System (CDSS) that:

- Continuously monitors ICU patients using multimodal data (vital signs, computer vision features, clinical notes)
- Predicts the patient's **SOFA score** (Sequential Organ Failure Assessment, 0–24) as a clinical severity measure
- Classifies risk as **Low** (SOFA < 5) / **Moderate** (5–9) / **High** (≥ 10)
- Fires a clinical alert when predicted SOFA ≥ 8 (threshold lowered to compensate for model under-prediction)
- Provides **SHAP explainability** showing which features drove the prediction
- Generates a structured **LLM clinical assessment** (condition, cause, forecast, actions) using Groq
- Validates LLM reliability via a **self-consistency check** (3 responses + 3-component consistency metric)
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
│  Latest Vitals (7) │ CV Features (2) │ TF-IDF SOFA vocab (90)│
└──────────────────────┬──────────────────────────────────────┘
                       │  108 features
                       ▼
┌─────────────────────────────────────────────────────────────┐
│       Federated DNN  (108→128→64→32→1)                      │
│  Trained via Flower FedYogi + FedProx across 3 hospitals    │
│  AdamW + Linear Weighted MSE Loss (1 + SOFA × 0.5)         │
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
│         Groq openai/gpt-oss-120b — Clinical Explanation      │
│  3 responses → 3-component consistency → Reliability score  │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version | Used by |
|---|---|---|
| Python | 3.12+ | All |
| PyTorch | 2.0+ | Model inference + FL training |
| scikit-learn | 1.6+ | Scaler + TF-IDF |
| Flower | 1.8+ | Federated Learning |
| Streamlit | 1.35+ | Main web app |
| SHAP | 0.44+ | Explainability |
| Groq API Key | Required | LLM clinical report |
| OpenCV | 5.0+ | CV Monitor (webcam display) |
| MediaPipe | 0.10.35+ | CV Monitor (face landmark + blendshapes) |

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
# Core app
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

# Federated Learning
flwr[simulation]>=1.8.0

# Computer Vision Monitor
opencv-python>=5.0.0
mediapipe>=0.10.35

# BigQuery (Phase 0 only)
google-cloud-bigquery>=3.11.0
google-cloud-bigquery-storage>=2.20.0
```

> Note: `flwr[simulation]` installs Flower with Ray-based simulation support. This may take a few minutes.

> Note: `opencv-python` and `mediapipe` are required for `cv_monitor.py` only. The main Streamlit app runs without them.

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
├── app.py                    ← Main Streamlit application (CDSS)
├── fl_dashboard.py           ← Federated Learning dashboard (Streamlit)
├── cv_monitor.py             ← Computer Vision Monitor — real-time Stress + GCS (standalone)
├── model_utils.py            ← ICUModel architecture + training utilities
├── train_federated.py        ← FL training script (simulation mode)
├── server.py                 ← Flower FL server (real FL demo)
├── client.py                 ← Flower FL client (real FL demo)
│
├── requirements.txt
├── .env                      ← Groq API key (create this — never push to git)
├── .env.example              ← Template
├── README.md
│
├── docs/                     ← Technical documentation
│   ├── CV_MONITOR.md         ← Stress Score + GCS Eye Score: methods, algorithms, pipeline
│   ├── TRAIN_FEDERATED.md    ← Complete training flow + optimization journey
│   ├── Project_Context.md    ← Full project context and design decisions
│   ├── MODEL_DISCUSSION.md   ← Model evolution and alternatives
│   ├── VALIDATION.md         ← SOFA and LLM validation framework
│   ├── INPUT_OUTPUT.md       ← Complete data flow documentation
│   └── SKILL.md              ← Technical reference document
│
├── models/
│   ├── federated_model.pth   ← Trained PyTorch DNN weights (~0.3 MB)
│   ├── scaler.pkl            ← StandardScaler (108 features, ~8 KB)
│   ├── tfidf_vectorizer.pkl  ← TF-IDF vectorizer (90 SOFA-vocab terms, ~2 MB)
│   ├── feature_columns.pkl   ← Ordered list of 108 feature names
│   ├── shap_background.npy   ← 300 background samples for SHAP, shape (300, 108)
│   ├── patient_vitals/       ← Per-patient sliding window vitals CSVs
│   ├── prediction_history/   ← Per-patient SOFA prediction logs
│   └── training_metadata.json← Training config and performance metrics
│
├── face_landmarker.task      ← MediaPipe face landmark model (~12 MB, auto-downloaded)
│                               gitignored — regenerated by cv_monitor.py on first run
│
└── data/
    └── fl_training/          ← Hospital client datasets for FL
        ├── client_0.csv      ← Hospital 0: ~15,889 rows × 109 cols
        ├── client_1.csv      ← Hospital 1: ~15,890 rows × 109 cols
        └── client_2.csv      ← Hospital 2: ~16,371 rows × 109 cols
```

---

## Configuration

### App Configuration (`app.py`)

| Constant | Default | Description |
|---|---|---|
| `ALERT_THRESHOLD` | `8.0` | SOFA score at which the red alert fires. Set lower than the clinical ≥ 10 boundary to compensate for model under-prediction |

### Training Configuration (`train_federated.py`)

| Constant | Default | Description |
|---|---|---|
| `NUM_ROUNDS` | `100` | Number of FL aggregation rounds |
| `EPOCHS_PER_ROUND` | `3` | Local training epochs per hospital per round |
| `BATCH_SIZE` | `64` | Mini-batch size for DataLoader |
| `BASE_LR` | `0.001` | AdamW starting learning rate |
| `LR_DECAY` | `0.99` | Per-round LR multiplier (LR decays slowly each round) |
| `GRAD_CLIP` | `1.0` | Gradient L2 norm clipping threshold |
| `MU_FEDPROX` | `0.5` | FedProx proximal term weight (prevents client drift) |
| `SERVER_ETA` | `0.01` | FedYogi server-side step size |
| `SERVER_BETA1` | `0.9` | FedYogi first-moment decay (momentum) |
| `SERVER_BETA2` | `0.99` | FedYogi second-moment decay (adaptive rate) |
| `SERVER_TAU` | `0.001` | FedYogi stability constant |
| `NOTES_SAMPLE_SIZE` | `283208` | Number of clinical notes sampled for TF-IDF |
| `NOTES_TEXT_LIMIT` | `10000` | Max characters per admission in TF-IDF |
| `OVERSAMPLE` | `False` | WeightedRandomSampler for high-risk oversampling (tested: caused overfitting) |
| `USE_NONIID_SPLIT` | `False` | Enable biased hospital split by SOFA severity (tested: causes client drift, lower R²) |
| `USE_DP` | `False` | Enable Differential Privacy (clip + add Gaussian noise before sending weights) |
| `DP_SENSITIVITY` | `1.0` | DP clipping threshold (max L2 norm of model update) |
| `DP_SIGMA` | `1.0` | DP noise multiplier (noise std = sigma × sensitivity) |
| `SHAP_BG_SAMPLES` | `300` | Number of background samples for SHAP DeepExplainer |

### Real FL Configuration (`client.py`)

| Constant | Default | Description |
|---|---|---|
| `BASE_LR` | `0.001` | Client-side AdamW learning rate |
| `LR_DECAY` | `0.99` | Per-round LR multiplier (mirrors train_federated.py) |
| `EPOCHS` | `3` | Local training epochs per round |
| `GRAD_CLIP` | `1.0` | Gradient clipping threshold |
| `MU_FEDPROX` | `0.5` | FedProx proximal term weight |
| `USE_DP` | `False` | Enable DP in real FL clients (must match server intent) |
| `DP_SENSITIVITY` | `1.0` | Clipping threshold |
| `DP_SIGMA` | `1.0` | Noise multiplier |

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

## Computer Vision Monitor

`cv_monitor.py` is a **standalone real-time computer vision module** that measures two of the ICU model's input features from a live webcam feed — no physical contact with the patient required.

### What it measures

| Feature | Output | Clinical meaning |
|---|---|---|
| **Stress Score** | 0.0 – 10.0 | Facial distress / pain level measured via PSPI |
| **GCS Eye Score** | E1 – E4 (1 – 4) | Eye opening response — proxy for consciousness level |

### Scoring method

**Stress Score** uses the **PSPI (Prkachin-Solomon Pain Intensity)** — a peer-reviewed clinical formula:
```
PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
```
Computed from MediaPipe blendshapes (browDown, cheekSquint, eyeSquint, noseSneer, mouthUpperUp, eyeBlink). Smile-corrected to avoid false readings from laughing. Head-movement agitation blended in at 20%. EMA-smoothed (α = 0.15).

**GCS Eye Score** uses a **30-second rolling behavioral window** with personal calibration:
- The system learns each patient's natural open-eye EAR in ~3 seconds
- Classifies each frame as `open` / `partial` / `closed` using both EAR and eyeBlink blendshape
- Maps the open fraction to E4 / E3 / E2 / E1 following clinical GCS criteria

For full technical details see `docs/CV_MONITOR.md`.

### Running the CV Monitor

```bash
cd icu_monitor

# Default webcam (built-in camera, index 0)
python cv_monitor.py

# External / USB webcam
python cv_monitor.py --cam 1

# Press Q or ESC to quit
```

**First run only:** The script downloads `face_landmarker.task` (~12 MB) from Google MediaPipe CDN and saves it to `icu_monitor/`. All subsequent runs start instantly.

### What you see

```
┌─────────────────────────────────────────────────────────────────────┐
│  ICU CDSS  |  Computer Vision Monitor  |  Stress Score & GCS Eye   │
├──────────────────────────────── Camera feed ────────────────────────┤
│  [Live face with eye-contour cyan overlay]                          │
├────────────────────────┬────────────────────────────────────────────┤
│  STRESS SCORE          │  GCS EYE SCORE  (CV proxy)                 │
│  2.4  / 10             │  E4   Spontaneous                          │
│  LOW                   │       Score: 4 / 4                         │
│  [gauge bar]           │  [E1] [E2] [E3] [E4●]                     │
│  Session peak: 3.1     │  EAR: 0.311                                │
│  Brow Furrow  ▌  4%   │  Thresholds: open≥0.23  partial≥0.13      │
│  Eye Tension  ███ 28%  │             (personal)                     │
│  Nose / Lip   ▌  2%   │  E4:open≥70%  E3:any≥30%  E2:flicker      │
└────────────────────────┴────────────────────────────────────────────┘
```

### 3-second calibration on startup

When you first run the monitor, the panel shows a calibration progress bar for ~3 seconds. During this time:
- Keep your eyes naturally open
- Keep a neutral expression
- The system learns your personal open-eye EAR baseline

After calibration: GCS thresholds are personalised, stress scoring begins from frame 1.

### Integration plan (future)

The scores from `cv_monitor.py` map directly to the two CV input features of the ICU model:

| cv_monitor.py output | ICU model input feature |
|---|---|
| `stress_v` (float 0–10) | `stress_score` |
| `gcs_v` (int 1–4) | `GCS_eye_opening` |

These will be fed into `app.py`'s continuous monitoring pipeline in a future sprint.

---

## Federated Learning Training

### Retrain the model (simulation mode)

This runs the complete preprocessing from BigQuery **plus** FL training in a single script.

```bash
cd icu_monitor
python train_federated.py
```

**Expected total time:** ~40–60 minutes
- Phase 0 (BigQuery + TF-IDF, all 283k notes): ~15–20 minutes
- Phase 1–6 (FL training, 100 rounds × 3 epochs): ~25–40 minutes

**What it does:**
1. Queries BigQuery (`ml_dataset_final` and `clinical_notes`)
2. Builds TF-IDF with 90-term SOFA vocabulary whitelist
3. StandardScaler + clip(±10) on training data
4. Splits 48,150 training samples into 3 hospital CSVs (IID 33/33/34)
5. Saves 300 SHAP background samples → `models/shap_background.npy`
6. Runs Flower FL simulation: 100 rounds × 3 epochs × 3 hospitals with FedYogi server + FedProx client
7. Saves best global model → `models/federated_model.pth`
8. Evaluates on held-out test set (12,038 patients) and prints MAE / R²
9. Saves training metadata → `models/training_metadata.json`

**Re-run without BigQuery (CSVs already exist):**

```bash
# Phase 0 is automatically skipped if client_0/1/2.csv already exist
python train_federated.py
```

**Force Phase 0 re-run (delete CSVs and pkl artifacts):**

```bash
rm data/fl_training/client_0.csv data/fl_training/client_1.csv data/fl_training/client_2.csv
rm models/scaler.pkl models/tfidf_vectorizer.pkl models/feature_columns.pkl
python train_federated.py
```

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
  Rounds  : 100
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

The app maintains the last 20 vital sign readings per patient.

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
1. The same prompt is sent to Groq 3 times (temperature=0.2)
2. A 3-component consistency score is computed:
   - TF-IDF cosine similarity: 20% weight
   - Clinical intervention agreement (vasopressors, antibiotics, etc.): 50% weight
   - Clinical condition agreement (sepsis, AKI, hypoxemia, etc.): 30% weight

| Score | Label |
|---|---|
| ≥ 0.80 | ✅ High Reliability |
| 0.60–0.80 | ⚠️ Moderate Reliability |
| < 0.60 | ❌ Low Reliability |

### Prediction History

Every prediction is logged per patient (keeps last 50). Visible as a table in Tab 1 under the trend chart once 2+ predictions have been made.

---

## Model Details

### Architecture

```
Input: 108 features
  ├─ 9 trend features  (HR_mean, HR_std, RR_mean, SpO₂_mean, SpO₂_min,
  │                     Temp_mean, SBP_mean, DBP_mean, MAP_mean)
  ├─ 7 latest vitals   (latest_HR, latest_RR, latest_SpO₂, latest_Temp,
  │                     latest_SBP, latest_DBP, latest_MAP)
  ├─ 2 CV features     (GCS_eye_opening, stress_score)
  └─ 90 TF-IDF         (SOFA-vocabulary whitelist: 6 organ components)

DNN:  108 → Linear(128) → ReLU
           → Linear(64)  → ReLU
           → Linear(32)  → ReLU
           → Linear(1)
           → SOFA Score (0–24, direct output, no scaling)

Total parameters: ~23,000
```

### Performance (held-out test set, 12,038 patients)

| Metric | Value |
|---|---|
| MAE | **1.9608 SOFA points** |
| R² | **0.3357** |
| Prediction range | -0.26 – 13.91 |
| Best FL round | 24 (of 100) |

Per-segment performance:

| Segment | n | MAE | R² |
|---|---|---|---|
| Low Risk (SOFA < 5) | 7,546 | 1.555 | −1.093 |
| Moderate (SOFA 5–9) | 3,754 | 2.103 | −2.599 |
| High Risk (SOFA ≥ 10) | 738 | 5.385 | −6.601 |

> Note: Within-segment R² is negative because the model correctly distinguishes between risk tiers (positive global R²) but cannot precisely rank patients within the same tier without direct lab values (bilirubin, creatinine, platelets). This is explained in `docs/VALIDATION.md`.

### Training

- **Server Optimizer:** FedYogi (η=0.01, β1=0.9, β2=0.99, τ=0.001) — adaptive server-side optimization
- **Client Regularisation:** FedProx (μ=0.5) — prevents client drift
- **Loss:** Linear weighted MSE — `weight = 1 + SOFA × 0.5` (high-SOFA patients weighted ~2×)
- **Optimizer:** AdamW (`lr=0.001`, `weight_decay=1e-4`)
- **Data:** 48,150 MIMIC-III ICU samples split across 3 simulated hospitals
- **Rounds:** 100 FL rounds, 3 local epochs per round

### SHAP Explainability

Uses `shap.DeepExplainer` with 300 background samples (shape: 300 × 108).

- Model is set to `eval()` mode — deterministic forward pass
- Top 7 clinically relevant features are selected by SHAP absolute impact
- Each feature is mapped to a human-readable clinical interpretation

---

## Dataset Information

### Source

MIMIC-III (Medical Information Mart for Intensive Care III), accessed via Google BigQuery.

**BigQuery project:** `mimic-project-2`  
**BigQuery dataset:** `Dataset`  
**Processed table:** `mimic-project-2.Dataset.ml_dataset_final`

### What the client CSVs contain

Each `data/fl_training/client_N.csv` file contains pre-processed, scaled training data:

- **Rows:** ~15,889–16,371 ICU measurement windows
- **Columns:** 108 feature columns (already StandardScaler-normalised + clipped to ±10) + `sofa_score` target = **109 columns total**
- The scaler was fit during Phase 0 of `train_federated.py` — do not re-scale these CSVs
- **Can be regenerated** by running `train_federated.py` with BigQuery access

### TF-IDF Vocabulary

The TF-IDF vectorizer uses an **explicit 90-term SOFA vocabulary whitelist** — not data-driven feature selection. Every term maps directly to one of the 6 SOFA organ components:

| SOFA Component | Example terms |
|---|---|
| Respiratory | intubated, ventilator, bipap, hypoxia, respiratory failure |
| Coagulation | plt, platelets, coagulopathy, inr, hemorrhage |
| Hepatic | bilirubin, totbili, bili, jaundice, cirrhosis |
| Cardiovascular | vasopressor, septic shock, levophed, hypotension |
| CNS | sedated, coma, encephalopathy, gcs, altered mental |
| Renal | creatinine, dialysis, oliguria, renal failure, aki |

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| DNN Framework | PyTorch 2.x |
| Federated Learning | Flower (flwr) 1.8+ — FedYogi + FedProx |
| Explainability | SHAP 0.44+ |
| LLM Provider | Groq (`openai/gpt-oss-120b`) |
| LLM Self-Consistency | 3-component score (TF-IDF + interventions + conditions) |
| Web Framework | Streamlit 1.35+ |
| Visualisation | Plotly |
| Data Processing | pandas, numpy |
| Environment Config | python-dotenv |
| Dataset | MIMIC-III (Google BigQuery) |
| CV — Camera Capture & Display | OpenCV 5.0+ |
| CV — Face Landmark + Blendshapes | MediaPipe 0.10.35+ (FaceLandmarker Tasks API) |
| CV — Stress Scoring | PSPI (Prkachin-Solomon Pain Intensity, 2008) |
| CV — GCS Eye Scoring | Rolling 30-second behavioral window + personal calibration |

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

The sliding window file may be corrupted. Reset it via the sidebar Reset button, or delete the patient vitals file and let the app recreate it.

### Training crashes with `RuntimeError: Simulation crashed`

Flower's simulation requires Ray. Ensure `flwr[simulation]` (not just `flwr`) is installed:

```bash
pip install "flwr[simulation]"
```

### Real FL training — clients cannot connect to server

Ensure you start `server.py` FIRST and wait for the "Waiting for 3 hospital clients..." message before launching clients. All 4 processes must run from the `icu_monitor/` directory.

### Training gives oscillating loss (FedYogi exploration phase)

This is expected. FedYogi causes an initial "cold-start exploration" phase (rounds 5–12) where server loss increases before recovering to a deeper minimum. The `SaveBestStrategy` always saves the model from the round with the lowest validation loss, so the final saved model is the best checkpoint regardless of later oscillations.

---

## Quick Start Summary

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Set Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Run the main CDSS app
streamlit run app.py

# 4. Run the Computer Vision Monitor (separate standalone window)
#    First run downloads face_landmarker.task (~12 MB) automatically
python cv_monitor.py

# 5. (Optional) Retrain the federated model (requires BigQuery access)
python train_federated.py

# 6. (Optional) Real FL demo — run in 4 separate terminals
python server.py
python client.py --client_id 0
python client.py --client_id 1
python client.py --client_id 2
```

---

*This project was developed as a Final Year Capstone in Computer Science Engineering.*
*Dataset: MIMIC-III | Framework: Flower (FedYogi + FedProx) | LLM: Groq openai/gpt-oss-120b*

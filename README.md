# ClinIQ — ICU Clinical Decision Support System
### AI-Powered Real-Time Patient Monitoring with Federated Learning

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Running the Application](#running-the-application)
7. [Dashboard Features](#dashboard-features)
8. [AI Clinical Report](#ai-clinical-report)
9. [Model Details](#model-details)
10. [Federated Learning Training](#federated-learning-training)
11. [Dataset Information](#dataset-information)
12. [Technology Stack](#technology-stack)
13. [Database Inspection (SQLite CLI)](#database-inspection-sqlite-cli)
14. [Troubleshooting](#troubleshooting)

---

## Project Overview

**ClinIQ** is an AI-powered ICU Clinical Decision Support System (CDSS) built as a full-stack web application. It continuously monitors ICU patients in real time, predicts clinical deterioration, and generates structured LLM clinical assessments.

**What it does:**

- Simulates continuous ICU monitoring — fetches a new patient vitals reading every **90 seconds**, cycling through 30 pre-loaded MIMIC-III readings per patient
- Predicts the **SOFA score** (Sequential Organ Failure Assessment, 0–24) using a Federated DNN trained across 3 simulated hospital nodes
- Classifies risk as **Low** (SOFA < 5) / **Moderate** (5–9) / **High** (≥ 10) and fires a clinical alert when SOFA ≥ 8
- Shows **SHAP feature attribution** — top 5 clinical drivers of each prediction with animated bar chart
- Generates a 4-section **AI clinical assessment** (Condition → Cause → Forecast → Actions) using `openai/gpt-oss-120b` via Groq, validated by a judge LLM for reliability
- Visualises **vital sign trends** over the last 20 readings with interactive charts
- Supports two care modes: **Hospital** and **Home Care** — the LLM generates mode-specific recommendations

> **Important:** This is a research prototype and decision support tool only. It does not diagnose disease or replace clinical judgment.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  React 19 + Vite Frontend  (port 5174)                          │
│  SelectPatient → MonitorDashboard                                │
│  Framer Motion · anime.js · Recharts · TanStack Query           │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP (Vite proxy → :8000)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                                    │
│  /api/patients  /api/predict  /api/report                        │
│  /api/history   /api/vitals                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐     ┌──────────────────────────────────┐
│  ML Service          │     │  LLM Service                      │
│  PyTorch DNN         │     │  openai/gpt-oss-120b (main)       │
│  108 features        │     │  openai/gpt-oss-20b  (judge)      │
│  SOFA prediction     │     │  90s TTL in-process cache         │
│  SHAP DeepExplainer  │     │  Threading timeout (12s judge)    │
└──────────┬──────────┘     └──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  SQLite Database  (models/icu_monitor.db)                        │
│  sample_patients · patient_vitals · prediction_history           │
│  fl_training (48,150 rows — MIMIC-III derived)                  │
└─────────────────────────────────────────────────────────────────┘
```

**Predict pipeline (per reading):**

```
sample_patients row  →  Sliding window (20 readings)
  →  Trend features (9)  +  Latest vitals (7)  +  CV features (2)  +  TF-IDF (90)
  →  108-feature vector  →  Federated DNN  →  SOFA score
  →  Risk classification  +  SHAP top-5  +  Conformal interval
```

**LLM report pipeline (triggered on new vitals or care-mode switch):**

```
SOFA + vitals + SHAP top-5  →  openai/gpt-oss-120b  (main assessment, 3500 tokens)
                             →  openai/gpt-oss-20b   (judge: G-Eval + RAGAS, 300 tokens)
                             →  9-method reliability score  →  ReliabilityBadge
```

---

## Prerequisites

| Requirement | Version | Used by |
|---|---|---|
| Python | 3.12+ | Backend |
| Node.js | 18+ | Frontend |
| PyTorch | 2.0+ | ML inference |
| scikit-learn | 1.6+ | Scaler + TF-IDF |
| FastAPI + uvicorn | 0.115+ / 0.30+ | API server |
| Groq API Key | — | LLM clinical report |
| Flower | 1.8+ | FL training (optional) |

---

## Installation

### Step 1 — Clone / Download the project

Ensure the `icu_monitor/` folder is on your machine.

### Step 2 — Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows
```

### Step 3 — Install Python dependencies

```bash
cd icu_monitor
pip install -r requirements.txt
```

### Step 4 — Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 5 — Set up the SQLite database

```bash
cd icu_monitor
python migrate_to_db.py
```

Expected output:
```
  patient_vitals                  60 rows
  prediction_history             112 rows
  sample_patients                 90 rows
  fl_training                 48,150 rows
  Database size: ~23 MB
```

> This is a one-time step. The database is gitignored. After a fresh clone, always run `migrate_to_db.py` first.

### Step 6 — Configure the Groq API key

```bash
cp .env.example .env
```

Open `.env` and set your key:

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [https://console.groq.com](https://console.groq.com)

---

## Project Structure

```
icu_monitor/
│
├── backend/                      ← FastAPI application
│   ├── main.py                   ← App entry point, CORS, router registration
│   ├── api/
│   │   ├── patients.py           ← GET /api/patients, GET /api/patients/{id}
│   │   ├── predict.py            ← POST /api/predict — SOFA + SHAP + risk
│   │   ├── report.py             ← POST /api/report — LLM clinical assessment
│   │   ├── history.py            ← GET /api/history/{id} — prediction log
│   │   └── vitals.py             ← GET /api/vitals/{id} — last 20 readings
│   └── services/
│       ├── ml_service.py         ← PyTorch DNN inference + SHAP + validation
│       └── llm_service.py        ← Groq LLM calls + judge + 90s TTL cache
│
├── frontend/                     ← React 19 + Vite application
│   ├── src/
│   │   ├── pages/
│   │   │   ├── SelectPatient.jsx ← Patient selection screen
│   │   │   └── MonitorDashboard.jsx ← Main monitoring dashboard
│   │   ├── components/
│   │   │   ├── layout/           ← HeaderBar, AlertBanner, ThemeToggle
│   │   │   ├── overview/         ← SOFAGauge, VitalsGrid, VitalCard
│   │   │   ├── report/           ← ReportContainer, ReportSection,
│   │   │   │                        ReliabilityBadge, SHAPPanel, ReportToggle
│   │   │   └── trends/           ← TrendsSection, TrendChart
│   │   ├── hooks/
│   │   │   └── useMonitor.js     ← 90s polling loop + countdown timer
│   │   ├── contexts/
│   │   │   └── ThemeContext.jsx  ← Light / dark theme
│   │   ├── api/
│   │   │   └── client.js         ← Axios API client (proxied to :8000)
│   │   ├── App.jsx               ← Router setup
│   │   ├── main.jsx              ← React entry point
│   │   └── index.css             ← Design tokens + global styles
│   ├── index.html
│   ├── vite.config.js            ← Proxy /api → localhost:8000
│   ├── tailwind.config.js
│   └── package.json
│
├── models/
│   ├── icu_monitor.db            ← SQLite database (~23 MB, gitignored)
│   │                                Tables: sample_patients · patient_vitals
│   │                                        prediction_history · fl_training
│   ├── federated_model.pth       ← Trained PyTorch DNN weights (~0.3 MB)
│   ├── scaler.pkl                ← StandardScaler (108 features)
│   ├── tfidf_vectorizer.pkl      ← TF-IDF vectorizer (90 SOFA-vocab terms)
│   ├── feature_columns.pkl       ← Ordered list of 108 feature names
│   ├── shap_background.npy       ← 300 background samples for SHAP (300 × 108)
│   ├── training_metadata.json    ← Training config + performance metrics
│   ├── patient_vitals/           ← Gitignored legacy CSVs (.gitkeep only)
│   └── prediction_history/       ← Gitignored legacy CSVs (.gitkeep only)
│
├── data/
│   ├── sample_patients/          ← Source CSVs for migrate_to_db.py
│   │   ├── sample_patient01_data.csv
│   │   ├── sample_patient02_data.csv
│   │   └── sample_patient03_data.csv
│   └── fl_training/              ← MIMIC-III FL training CSVs (gitignored)
│       └── README.md
│
├── tests/                        ← pytest test suite
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_patients.py
│   ├── test_predict.py
│   ├── test_report.py
│   ├── test_history.py
│   ├── test_vitals.py
│   └── test_integration.py
│
├── docs/                         ← Technical documentation
│   ├── Project_Context.md
│   ├── MODEL_DISCUSSION.md
│   ├── VALIDATION.md
│   ├── INPUT_OUTPUT.md
│   ├── TRAIN_FEDERATED.md
│   ├── CV_MONITOR.md
│   └── FL_DASHBOARD_PLAN.md
│
├── db.py                         ← SQLite read/write helpers (used by backend)
├── model_utils.py                ← ICUModel architecture (used by ml_service.py)
├── migrate_to_db.py              ← One-time CSV → SQLite migration
├── train_federated.py            ← FL training script (simulation mode)
├── server.py                     ← Flower FL server (real FL demo)
├── client.py                     ← Flower FL client (real FL demo)
├── cv_monitor.py                 ← Computer Vision Monitor (standalone)
│
├── requirements.txt
├── pytest.ini
├── .env                          ← Groq API key (never push to git)
├── .env.example                  ← Template
└── README.md
```

---

## Running the Application

Two processes must run simultaneously — the FastAPI backend and the Vite frontend.

### Terminal 1 — Backend

```bash
cd icu_monitor
uvicorn backend.main:app --port 8000 --reload
```

Backend is available at **http://localhost:8000**  
API docs (auto-generated): **http://localhost:8000/docs**

### Terminal 2 — Frontend

```bash
cd icu_monitor/frontend
npm run dev
```

Dashboard opens at **http://localhost:5174**

### Restart commands (if port 8000 is already in use)

```bash
lsof -ti :8000 | xargs kill -9
uvicorn backend.main:app --port 8000 --reload
```

---

## Dashboard Features

### Patient Selection (`/`)

- Three ICU patient profiles loaded from `sample_patients` table
- Each card shows name, description, and a live ECG waveform animation
- Dark / light theme toggle persists across navigation

### Monitor Dashboard (`/monitor/:id`)

**Header bar** — ClinIQ brand, LIVE chip with ECG animation, 90s countdown timer, patient info pill, risk-color reactive accent strip.

**Patient Context Strip** — risk level badge (color-coded), SOFA score, AI confidence — dims during refresh.

**Current Status section**
- `SOFAGauge` — circular radial gauge, risk-colored, animated on new data
- `VitalsGrid` — 7 vital sign cards (HR, RR, SpO₂, Temp, SBP, DBP, MAP) with range classification and spark-style value display

**Vital Trends section** — Recharts line charts for the last 20 readings per vital, with normal-range bands and abnormal-value highlighting.

**AI Clinical Report section**
- Hospital / Home Care toggle — switches care mode and triggers a new LLM report
- `ReliabilityBadge` — circular score dial (0–100%) with label and caution text
- 4-section report grid: Current Condition · Probable Cause · Risk Forecast · Immediate Actions
- `SHAPPanel` — "Key Risk Drivers" bar chart with animated bars (anime.js spring physics), direction arrows (↑/↓ risk), impact badges

### 90-second refresh cycle

`useMonitor.js` drives the cycle:
1. Timer counts down from 90 → 1 (freezes at 1 during the in-flight predict call)
2. `runPredict` → backend fetches the next `sample_patients` row, runs DNN + SHAP
3. Timer resets to 90 only after the response arrives
4. `ReportContainer` compares SOFA + key vitals against the last LLM trigger — only calls the LLM if clinical values actually changed, or if care mode was switched

---

## AI Clinical Report

### LLM models used

| Role | Model | Tokens | Purpose |
|---|---|---|---|
| Main | `openai/gpt-oss-120b` | 3500 | Generate 4-section clinical assessment |
| Judge | `openai/gpt-oss-20b` | 300 | G-Eval + RAGAS reliability evaluation |

### 9-method reliability score

| # | Method | Weight | What it checks |
|---|---|---|---|
| 1 | Factual Grounding | 13% | Does the response acknowledge each abnormal vital sign? |
| 2 | Clinical Hard Rules | 13% | MAP < 65 → vasopressors, SpO₂ < 90 → oxygen mentioned? |
| 3 | SHAP-LLM Coherence | 9% | Does the response address the top SHAP prediction drivers? |
| 4 | Response Structure | 9% | All 4 required sections present and detailed? |
| 5 | Severity Calibration | 8% | Urgency language matches the SOFA level? |
| 6 | Contraindication Check | 5% | No dangerous drug recommendations for this patient's vitals? |
| 7 | Numeric Accuracy | 8% | Numbers stated in the response match actual vital values? |
| 8 | G-Eval Quality | 13% | Judge LLM: factual accuracy + clinical appropriateness (1–5 each) |
| 9 | RAGAS Faithfulness | 22% | Judge LLM: statements supported by actual patient vitals? |

| Score | Label |
|---|---|
| ≥ 80% | High Reliability |
| 60–79% | Moderate Reliability |
| < 60% | Low Reliability |

### Caching

Reports are cached in-process with a 90-second TTL keyed by `care_mode|sofa_score|HR|SpO2`. If the same patient with the same vitals is re-queried within 90 seconds, the cached report is returned instantly.

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
  └─ 90 TF-IDF         (SOFA vocabulary whitelist, 6 organ components)

DNN:  108 → Linear(128) → ReLU
           → Linear(64)  → ReLU
           → Linear(32)  → ReLU
           → Linear(1)
           → SOFA Score (0–24)

Total parameters: ~23,000
```

### Performance (held-out test set, 2,559 patients)

| Metric | Value |
|---|---|
| MAE | 1.8968 SOFA points |
| R² | 0.5995 |
| AUROC | 0.9461 |
| PR-AUC | 0.6840 |

> AUROC 0.9461 exceeds the published ICU ML benchmark of 0.82–0.84 (Lancet Digital Health 2025).

### Training

- **Server optimizer:** FedYogi (η=0.01, β1=0.9, β2=0.99)
- **Client regularisation:** FedProx (μ=0.5)
- **Loss:** Linear weighted MSE — `weight = 1 + SOFA × 0.5`
- **Optimizer:** AdamW (lr=0.001, weight_decay=1e-4)
- **Rounds:** 100 FL rounds × 3 local epochs × 3 hospital nodes

---

## Federated Learning Training

### Retrain the model (simulation mode)

```bash
cd icu_monitor
python train_federated.py
```

Takes ~40–60 minutes (Phase 0: BigQuery + TF-IDF, Phase 1–6: FL training).

**Force Phase 0 re-run:**

```bash
sqlite3 models/icu_monitor.db "DELETE FROM fl_training;"
rm models/scaler.pkl models/tfidf_vectorizer.pkl models/feature_columns.pkl
python train_federated.py
```

After retraining, restart the backend to load the new model weights.

### Real FL Demo (4 terminals)

```bash
# Terminal 1 — start the Flower server
python server.py

# Terminal 2–4 — connect one hospital client each
python client.py --client_id 0
python client.py --client_id 1
python client.py --client_id 2
```

---

## Dataset Information

**Source:** MIMIC-III (Medical Information Mart for Intensive Care III), accessed via Google BigQuery.

**BigQuery project:** `mimic-project-2`  
**Processed table:** `mimic-project-2.Dataset.ml_dataset_final`

The `sample_patients` table (90 rows, 3 patients × 30 readings) is the data source for the dashboard's live simulation. These are pre-processed MIMIC-III ICU windows with realistic vital sign progressions.

The `fl_training` table holds the full 48,150 training rows used by `train_federated.py` — StandardScaler-normalised, clipped to ±10, split across 3 hospital clients.

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19, Vite 8, Tailwind CSS 3, Framer Motion 14 |
| **Animations** | anime.js 4 (SHAP bars), AOS 2 (scroll-reveal), Framer Motion (page transitions) |
| **Charts** | Recharts 3 |
| **Data fetching** | TanStack Query 5, Axios |
| **Backend** | FastAPI 0.115, uvicorn |
| **ML framework** | PyTorch 2.x |
| **Federated Learning** | Flower (flwr) 1.8 — FedYogi + FedProx |
| **Explainability** | SHAP 0.44 — DeepExplainer, top-5 features |
| **LLM (main)** | Groq `openai/gpt-oss-120b` — clinical assessment |
| **LLM (judge)** | Groq `openai/gpt-oss-20b` — G-Eval + RAGAS reliability |
| **Database** | SQLite 3 (stdlib) |
| **Environment** | python-dotenv |
| **Dataset** | MIMIC-III (Google BigQuery) |

---

## Database Inspection (SQLite CLI)

```bash
sqlite3 models/icu_monitor.db
```

```sql
.headers on
.mode column

-- Row counts across all tables
SELECT 'patient_vitals'    AS tbl, COUNT(*) AS rows FROM patient_vitals
UNION ALL SELECT 'prediction_history', COUNT(*) FROM prediction_history
UNION ALL SELECT 'sample_patients',    COUNT(*) FROM sample_patients
UNION ALL SELECT 'fl_training',        COUNT(*) FROM fl_training;

-- Latest reading per patient
SELECT patient, MAX(time) AS latest FROM patient_vitals GROUP BY patient;

-- Last 5 SOFA predictions for patient 1
SELECT Timestamp, SOFA, Risk, HR, SpO2, MAP
FROM prediction_history WHERE patient = 1 ORDER BY id DESC LIMIT 5;

-- Verify 90-row simulation data (3 patients × 30 readings)
SELECT patient, COUNT(*) AS rows FROM sample_patients GROUP BY patient;

.quit
```

---

## Troubleshooting

**`ModuleNotFoundError` on backend start**
```bash
pip install -r requirements.txt
```

**Port 8000 already in use**
```bash
lsof -ti :8000 | xargs kill -9
uvicorn backend.main:app --port 8000 --reload
```

**Report shows "No assessment available" in all sections**
- Confirm `GROQ_API_KEY` is set in `.env`
- Verify the key is active at [https://console.groq.com](https://console.groq.com)
- Check backend terminal for error output — the model used is `openai/gpt-oss-120b`

**SOFA score is 0 or very low on first load**
- The sliding window needs at least 1 reading. Wait for the first 90-second cycle to complete, or restart the backend to reset state.

**sklearn version warning on startup**
```
InconsistentVersionWarning: Trying to unpickle estimator StandardScaler...
```
Expected and harmless — `StandardScaler` is stable across 1.x versions.

**Database not found**
```bash
python migrate_to_db.py
```

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install frontend dependencies
cd frontend && npm install && cd ..

# 3. Set up database (one-time)
python migrate_to_db.py

# 4. Set Groq API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Start backend (Terminal 1)
uvicorn backend.main:app --port 8000 --reload

# 6. Start frontend (Terminal 2)
cd frontend && npm run dev

# Dashboard → http://localhost:5174
```

---

*Final Year Capstone — Computer Science Engineering*  
*Dataset: MIMIC-III · ML: PyTorch Federated DNN · LLM: Groq openai/gpt-oss-120b · Frontend: React 19 + Vite*

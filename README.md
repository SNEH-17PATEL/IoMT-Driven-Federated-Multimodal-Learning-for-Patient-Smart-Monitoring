# ClinIQ — ICU Clinical Decision Support System
### AI-Powered Real-Time Patient Monitoring with Federated Learning

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [COMPLETE DEMO GUIDE](#complete-demo-guide)
7. [Dashboard Features](#dashboard-features)
8. [AI Clinical Report](#ai-clinical-report)
9. [Model Details](#model-details)
10. [Federated Learning Training](#federated-learning-training)
11. [Dataset Information](#dataset-information)
12. [Technology Stack](#technology-stack)
13. [Database Inspection (SQLite CLI)](#database-inspection-sqlite-cli)
14. [Test Suite](#test-suite)
15. [Troubleshooting](#troubleshooting)

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
| Webcam | — | CV Monitor (optional) |

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
│   └── training_metadata.json    ← Training config + performance metrics
│
├── data/
│   ├── sample_patients/          ← Source CSVs for migrate_to_db.py
│   └── fl_training/              ← MIMIC-III FL training CSVs (gitignored)
│
├── tests/                        ← Full test suite (701 tests)
│   ├── fl/                       ← 287 FL tests (unit + integration + system)
│   └── cliniq/
│       ├── backend/              ← 207 FastAPI backend tests
│       └── frontend/             ← 207 Vitest frontend tests
│
├── docs/                         ← Technical documentation
│
├── db.py                         ← SQLite read/write helpers
├── model_utils.py                ← ICUModel architecture
├── migrate_to_db.py              ← One-time CSV → SQLite migration
├── train_federated.py            ← FL training script (simulation mode)
├── fl_dashboard.py               ← Streamlit FL Simulation Dashboard
├── app.py                        ← Streamlit Legacy ICU Monitor
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

## COMPLETE DEMO GUIDE

This section covers every demo-able component of the capstone project with full run commands. All commands run from `icu_monitor/` unless noted otherwise.

**Recommended demo order:**
1. ClinIQ Main Dashboard (most impressive — full-stack AI application)
2. API Swagger Docs (shows the backend API surface)
3. FL Simulation Dashboard (Streamlit visualisation of Federated Learning)
4. FL Training in Terminal (shows real training logs)
5. Real Flower FL Demo (4 terminals — actual federated clients connecting to server)
6. Computer Vision Monitor (webcam-based stress detection)
7. Legacy Streamlit ICU Monitor (original prototype)
8. Trial ICU Inference App (early research prototype)
9. Federated Learning Prototype App (early FL prototype)
10. SQLite Database Inspection (shows raw data and model outputs)
11. Full Test Suite (demonstrates 701 passing tests)

---

### Demo 1 — ClinIQ Main Dashboard (Full-Stack Web App)

The primary deliverable: React 19 + FastAPI + PyTorch + Groq LLM, all wired together.

**Terminal 1 — Start the FastAPI backend:**
```bash
cd icu_monitor
source venv/bin/activate
uvicorn backend.main:app --port 8000 --reload
```

**Terminal 2 — Start the React frontend:**
```bash
cd icu_monitor/frontend
npm run dev
```

**Open in browser:** http://localhost:5174

**What to show:**
- Patient selection screen — 3 ICU patients with ECG animation
- Click a patient — Monitor Dashboard loads
- Live 90-second countdown timer in the header
- SOFA score gauge (circular, color-coded: green/yellow/red)
- Vitals Grid — 7 vital signs (HR, RR, SpO₂, Temp, SBP, DBP, MAP) with range classification
- Vital Trends — Recharts line charts for last 20 readings with normal-range bands
- AI Clinical Report — 4 sections (Condition, Cause, Forecast, Actions) generated by Groq LLM
- Reliability Badge — circular score showing 0–100% LLM reliability via 9-method scoring
- SHAP Panel — top-5 risk drivers with animated bars (anime.js spring physics)
- Toggle between Hospital and Home Care mode — LLM regenerates report instantly
- Dark/light theme toggle (top-right)
- Wait for 90-second cycle — watch vitals update live and SOFA gauge animate

**If port 8000 is busy:**
```bash
lsof -ti :8000 | xargs kill -9
uvicorn backend.main:app --port 8000 --reload
```

---

### Demo 2 — API Swagger Docs (Interactive REST API)

Shows all backend endpoints with live testing capability. Backend from Demo 1 must be running.

**Open in browser:** http://localhost:8000/docs

**What to show:**
- `GET /api/patients` — list all 3 ICU patients (click "Try it out" → Execute)
- `GET /api/patients/1` — single patient details
- `POST /api/predict` — run SOFA prediction + SHAP (paste patient id: 1)
- `POST /api/report` — trigger LLM clinical assessment
- `GET /api/vitals/1` — last 20 vital sign readings
- `GET /api/history/1` — full prediction history log
- Alternatively view the raw JSON at http://localhost:8000/api/patients

---

### Demo 3 — FL Simulation Dashboard (Streamlit)

Interactive visualisation of the Federated Learning process: pick an algorithm, watch rounds train, and explore weight evolution charts.

```bash
cd icu_monitor
source venv/bin/activate
streamlit run fl_dashboard.py
```

**Open in browser:** http://localhost:8501

**What to show:**
- Sidebar controls: number of rounds (1–100), algorithm selector (FedAvg / FedYogi / FedProx), number of hospital clients, simulation speed
- Click **Start FL Simulation** — watch per-round progress bars update live
- Global Loss chart — training loss curve across rounds
- Client Loss chart — per-hospital loss (Hospital 0, 1, 2)
- Global MAE and R² curves — validation metrics improving per round
- Round Log table — round-by-round summary (H0 Loss, H1 Loss, H2 Loss, Global)
- Weight Statistics heatmap — mean/std/delta per client per round
- Layer 1 weights histogram — distribution shift before vs after aggregation
- Best Round indicator — round with lowest validation MAE highlighted
- Try FedYogi vs FedAvg: run 20 rounds of each and compare convergence curves

---

### Demo 4 — FL Training in Terminal (Simulation Mode)

Runs the complete federated training pipeline with live logs: BigQuery-sourced MIMIC-III data → 100 FL rounds → saves model weights.

```bash
cd icu_monitor
source venv/bin/activate
python train_federated.py
```

**What you see in the terminal:**
```
Phase 0: Loading FL training data from SQLite...
  Client 0: 16,050 rows loaded
  Client 1: 16,050 rows loaded
  Client 2: 16,050 rows loaded

Phase 1: Initialising global model (108 features, 3 clients)...
Phase 2: FL Training — FedYogi server, FedProx clients
  Round   1/100 | Global Loss: 47.23 | MAE: 6.21 | R²: -0.12
  Round   2/100 | Global Loss: 43.81 | MAE: 5.98 | R²:  0.05
  ...
  Round 100/100 | Global Loss: 12.44 | MAE: 1.90 | R²:  0.60

Phase 3: Saving model artifacts...
  federated_model.pth    saved
  scaler.pkl             saved
  tfidf_vectorizer.pkl   saved
  feature_columns.pkl    saved
  shap_background.npy    saved
  training_metadata.json saved

Training complete. MAE: 1.8968 | R²: 0.5995 | AUROC: 0.9461
```

> Full training takes ~40–60 minutes. For a quick demo, interrupt with Ctrl+C after a few rounds to show the logs.

---

### Demo 5 — Real Flower Federated Learning (4 Terminals)

Demonstrates actual federated learning: a Flower server coordinates 3 independent hospital clients that each train locally and send only weight updates — raw patient data never leaves the client.

**Terminal 1 — Start the FL server:**
```bash
cd icu_monitor
source venv/bin/activate
python server.py
```

Expected output:
```
INFO flwr 2024 - Starting Flower server, config: ServerConfig(num_rounds=10)
INFO flwr 2024 - Flower ECE: gRPC server running (insecure), listening on 0.0.0.0:8080
```

**Terminal 2 — Hospital 0:**
```bash
cd icu_monitor
source venv/bin/activate
python client.py --client_id 0
```

**Terminal 3 — Hospital 1:**
```bash
cd icu_monitor
source venv/bin/activate
python client.py --client_id 1
```

**Terminal 4 — Hospital 2:**
```bash
cd icu_monitor
source venv/bin/activate
python client.py --client_id 2
```

**What to show:**
- Server terminal: waits for 3 clients, then starts Round 1
- Each client terminal: shows local training progress per round
- Server terminal: aggregates weights after each round, prints global metrics
- Key point: each hospital trains on its own data — only weight updates are shared, never raw patient records

---

### Demo 6 — Computer Vision Monitor (Webcam Stress Detection)

Standalone OpenCV + MediaPipe application that detects facial stress indicators (blink rate, eye openness, gaze deviation) and feeds a stress score into the ICU model as a supplementary feature.

```bash
cd icu_monitor
source venv/bin/activate
python cv_monitor.py
```

**What to show:**
- Opens webcam feed in a window
- MediaPipe face mesh overlay — 468 facial landmarks tracked in real-time
- Blink rate counter (blinks per minute)
- Eye Aspect Ratio (EAR) — drops sharply on each blink
- Gaze deviation score — measures how much the patient looks away from center
- Stress score composite — combines blink rate, EAR, and gaze into 0–1 score
- This score populates `stress_score` in the 108-feature vector for the DNN

Press **Q** to quit.

---

### Demo 7 — Legacy Streamlit ICU Monitor

The original Streamlit prototype that preceded the full-stack React/FastAPI version. Runs as a single-page app.

```bash
cd icu_monitor
source venv/bin/activate
streamlit run app.py --server.port 8502
```

**Open in browser:** http://localhost:8502

**What to show:**
- Patient selector sidebar
- Real-time vitals display with color-coded risk level
- SOFA score prediction with the DNN model
- Prediction history chart
- Compares to the ClinIQ React app to show the evolution of the project

---

### Demo 8 — Trial ICU Inference App (Research Prototype)

Early research prototype from the `trial/` directory, built during initial model development before federated learning was added.

```bash
cd trial
streamlit run ICU_Inference.py --server.port 8503
```

**Open in browser:** http://localhost:8503

**What to show:**
- Manual vital sign input sliders
- Ensemble model inference (Random Forest + XGBoost + LightGBM)
- SHAP waterfall plots — visual breakdown of each feature's contribution
- Risk classification output
- Compares to the final federated DNN to show the progression from ensemble to FL

---

### Demo 9 — Federated Learning Prototype App

Early Streamlit FL prototype from the `federated learning/` directory, before the full FL dashboard and Flower integration were built.

```bash
cd "federated learning"
streamlit run app.py --server.port 8504
```

**Open in browser:** http://localhost:8504

**What to show:**
- Simulated FL training across 3 hospital nodes
- Per-round loss and accuracy charts
- Shows the early concept that evolved into the full FL simulation dashboard

---

### Demo 10 — SQLite Database Inspection

Live inspection of the production database — shows MIMIC-III-derived data, prediction history, and FL training splits.

```bash
cd icu_monitor
sqlite3 models/icu_monitor.db
```

Run these queries inside the SQLite shell:

```sql
.headers on
.mode column

-- ── Step 1: List every table in the database ─────────────────────────────────
.tables

-- ── Step 2: Row counts across all tables ─────────────────────────────────────
SELECT 'patient_vitals'    AS table_name, COUNT(*) AS total_rows FROM patient_vitals
UNION ALL SELECT 'prediction_history', COUNT(*) FROM prediction_history
UNION ALL SELECT 'sample_patients',    COUNT(*) FROM sample_patients
UNION ALL SELECT 'fl_training',        COUNT(*) FROM fl_training;

-- ── Step 3: Schema of every table ────────────────────────────────────────────
.schema patient_vitals
.schema prediction_history
.schema sample_patients
.schema fl_training

-- ── Step 4: First 10 rows — patient_vitals ───────────────────────────────────
-- 60 rows (3 patients × 20 readings). Real-time vital signs from MIMIC-III.
SELECT * FROM patient_vitals LIMIT 10;

-- ── Step 5: First 10 rows — prediction_history ───────────────────────────────
-- SOFA predictions logged over time. Written by the backend on every /api/predict call.
SELECT * FROM prediction_history LIMIT 10;

-- ── Step 6: First 10 rows — sample_patients ──────────────────────────────────
-- 90 rows (3 patients × 30 readings). Powers the 90-second live monitoring cycle.
SELECT * FROM sample_patients LIMIT 10;

-- ── Step 7: First 10 rows — fl_training ──────────────────────────────────────
-- 48,150 rows (MIMIC-III derived). Split across 3 hospital clients for federated training.
SELECT * FROM fl_training LIMIT 10;

-- ── Step 8: Per-patient breakdown ────────────────────────────────────────────
SELECT patient, COUNT(*) AS readings FROM patient_vitals GROUP BY patient;
SELECT patient, COUNT(*) AS readings FROM sample_patients GROUP BY patient;
SELECT patient, COUNT(*) AS predictions FROM prediction_history GROUP BY patient;

-- ── Step 9: Per-client FL training split ─────────────────────────────────────
SELECT client_id, COUNT(*) AS rows FROM fl_training GROUP BY client_id;

-- ── Step 10: Latest SOFA predictions per patient ─────────────────────────────
SELECT patient, Timestamp, SOFA, Risk, HR, SpO2, MAP
FROM prediction_history
WHERE id IN (
    SELECT MAX(id) FROM prediction_history GROUP BY patient
);

-- ── Step 11: Database file size ───────────────────────────────────────────────
SELECT ROUND(page_count * page_size / 1024.0 / 1024.0, 2) AS db_size_mb
FROM pragma_page_count(), pragma_page_size();

.quit
```

**What to show:**
- `.tables` — 4 tables: `patient_vitals`, `prediction_history`, `sample_patients`, `fl_training`
- Row counts — 48,150 FL training rows + 90 simulation rows + 60 vitals rows
- `sample_patients` schema — 108-feature rows including `stress_score`, `clinical_note`, `GCS_eye_opening`
- `fl_training` schema — `client_id` column + all feature + `sofa_score` target columns
- Per-client split — exactly 16,050 rows per hospital (clients 0, 1, 2)
- `prediction_history` — shows SOFA, Risk, all vitals written by the live backend
- The database is the single source of truth for both inference and FL training

---

### Demo 11 — Full Test Suite (701 Tests)

Shows the complete automated test coverage: FL system, ClinIQ backend API, and React frontend.

**FL System Tests (287 tests — all phases):**
```bash
cd icu_monitor
source venv/bin/activate

# Unit tests (fast)
python -m pytest tests/fl/unit/ -v

# Integration tests
python -m pytest tests/fl/integration/ -v

# System tests (end-to-end fl_thread simulation)
python -m pytest tests/fl/system/ -v -m slow

# All FL tests at once
python -m pytest tests/fl/ -v
```

**ClinIQ Backend Tests (207 tests):**
```bash
cd icu_monitor
source venv/bin/activate
python -m pytest tests/cliniq/backend/ -v
```

**React Frontend Tests (207 Vitest tests):**
```bash
cd icu_monitor/frontend
npx vitest run
```

**All runnable tests in one command:**
```bash
cd icu_monitor
python -m pytest tests/ -v
cd frontend && npx vitest run && cd ..
```

**Expected results:**
```
tests/fl/           287 passed
tests/cliniq/backend/  207 passed
frontend Vitest        207 passed
─────────────────────────────────
Total                  701 passed
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

Takes ~40–60 minutes (Phase 0: data loading, Phase 1–6: FL training, Phase 7: evaluation + artifact saving).

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
| **FL Dashboard** | Streamlit (fl_dashboard.py) |
| **Computer Vision** | OpenCV + MediaPipe (cv_monitor.py) |
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

-- FL training split per hospital client
SELECT client_id, COUNT(*) AS rows FROM fl_training GROUP BY client_id;

.quit
```

---

## Test Suite

### FL Tests — 287 tests across 10 phases

```bash
cd icu_monitor
source venv/bin/activate

# All FL tests
python -m pytest tests/fl/ -v

# By phase group
python -m pytest tests/fl/unit/       -v          # Phases 1–6 (fast unit tests)
python -m pytest tests/fl/integration/ -v         # Phases 7–9 (integration)
python -m pytest tests/fl/system/     -v -m slow  # Phase 10 (end-to-end system)
```

### ClinIQ Backend Tests — 207 tests

```bash
cd icu_monitor
python -m pytest tests/cliniq/backend/ -v
```

### React Frontend Tests — 207 Vitest tests

```bash
cd icu_monitor/frontend
npx vitest run
```

### Run everything

```bash
# Python tests (494 total)
cd icu_monitor && python -m pytest tests/ -v

# Frontend tests (207 total)
cd icu_monitor/frontend && npx vitest run
```

**Grand total: 701 passing tests.**

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

**Port 8501/8502/8503 already in use (Streamlit)**
```bash
lsof -ti :8501 | xargs kill -9
streamlit run fl_dashboard.py
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

**Flower server: "address already in use"**
```bash
lsof -ti :8080 | xargs kill -9
python server.py
```

**CV Monitor: no webcam detected**
- Ensure no other application (FaceTime, Zoom) is using the webcam
- macOS: grant camera access to Terminal in System Preferences → Privacy → Camera

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
# API docs  → http://localhost:8000/docs
```

---

*Final Year Capstone — Computer Science Engineering*  
*Dataset: MIMIC-III · ML: PyTorch Federated DNN · LLM: Groq openai/gpt-oss-120b · Frontend: React 19 + Vite*

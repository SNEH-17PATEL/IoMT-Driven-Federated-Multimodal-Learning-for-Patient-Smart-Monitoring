# ClinIQ — ICU Clinical Decision Support System
### AI-Powered Real-Time Patient Monitoring with Federated Learning

*Last updated: 2026-11-27 · matches the final trained model (round 37 of 100)*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Read This First — What You Get From a Fresh Clone](#read-this-first--what-you-get-from-a-fresh-clone)
3. [System Architecture](#system-architecture)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Project Structure](#project-structure)
7. [COMPLETE DEMO GUIDE](#complete-demo-guide)
8. [Dashboard Features](#dashboard-features)
9. [Camera Monitor (Stress + GCS Eye)](#camera-monitor-stress--gcs-eye)
10. [AI Clinical Report](#ai-clinical-report)
11. [REST API Reference](#rest-api-reference)
12. [Model Details](#model-details)
13. [Federated Learning Training](#federated-learning-training)
14. [Dataset Information](#dataset-information)
15. [Technology Stack](#technology-stack)
16. [Database Inspection (SQLite CLI)](#database-inspection-sqlite-cli)
17. [Test Suite](#test-suite)
18. [Documentation Index](#documentation-index)
19. [Troubleshooting](#troubleshooting)
20. [Quick Start](#quick-start)

---

## Project Overview

**ClinIQ** is an AI-powered ICU Clinical Decision Support System (CDSS) built as a full-stack web application. It continuously monitors ICU patients in real time, predicts how sick they are, explains why, and writes a structured clinical assessment using a large language model (LLM).

**What it does, in plain words:**

- **Simulates continuous ICU monitoring.** Every **90 seconds** it fetches the next vitals reading for the selected patient, cycling through 30 pre-loaded MIMIC-III readings per patient (3 patients, 90 rows in total).
- **Predicts the SOFA score.** SOFA (Sequential Organ Failure Assessment, 0–24) is the standard ICU measure of organ failure; higher means sicker. The prediction comes from a **Federated Deep Neural Network** (108 → 256 → 128 → 64 → 1, 69,121 parameters) trained with **FedAvg + FedProx** across 3 simulated hospitals on **41,798 MIMIC-III ICU stays**.
- **Classifies risk** as **Low** (SOFA < 5), **Moderate** (5–9) or **High** (≥ 10), and **raises an alert when the predicted SOFA is ≥ 7** (a deliberately lower trigger than 10, because the model under-predicts the sickest patients — see [Model Details](#model-details)).
- **Explains each prediction** with **SHAP**: the top 5 clinical drivers shown as an animated bar chart.
- **Shows an uncertainty range**: a 90% conformal interval of about ± 3.28 SOFA points.
- **Writes a 4-section AI clinical assessment** (Current Condition → Probable Cause → Risk Forecast → Immediate Actions) with `openai/gpt-oss-120b` via Groq, then **grades its own reliability** with a 9-method score.
- **Visualises vital-sign trends** over the last 20 readings.
- **Supports two care modes**: **Hospital** and **Home Care** — the LLM writes mode-specific recommendations.
- **Optional camera monitor**: the browser webcam estimates a **stress score** and **GCS eye-opening** and feeds them into the prediction (it moves the SOFA by only about 0.3–1.4 points and never changes the risk band or alert). The camera window has a **live panel** on its right that shows the predicted SOFA, a countdown, the two camera inputs next to the values the model actually used, the clinical note and the vitals, so you can see the camera values go in and the result come out.

> **Important:** this is a research prototype and decision-support tool only. It does not diagnose disease and does not replace clinical judgment. All results are retrospective MIMIC-III results, not independent clinical validation.

---

## Read This First — What You Get From a Fresh Clone

The patient database is derived from **MIMIC-III**, which may only be shared with people who hold PhysioNet credentialed access. For that reason **the database and training data are NOT stored on GitHub** (`models/icu_monitor.db` is git-ignored). `.env` (your API key) is also never pushed.

| Item | Fresh clone from GitHub | Your original working folder / credentialed hand-off |
|---|---|---|
| Source code, tests, docs | ✅ | ✅ |
| Trained model + scaler + TF-IDF + SHAP background (`models/*.pth`, `*.pkl`, `shap_background.npy`, `training_metadata.json`) | ✅ | ✅ |
| 3 sample patient CSVs (`data/sample_patients/*.csv`) | ✅ | ✅ |
| `models/icu_monitor.db` | ❌ built by `setup.py` from the CSVs (~0.1 MB, **90 `sample_patients` rows only**) | ✅ full (~56 MB) |
| `fl_training` table (41,798 training rows) | ❌ empty | ✅ |
| `models/fl_test_X.npy`, `fl_test_y.npy` (held-out test set) | ❌ | ✅ |
| `.env` with `GROQ_API_KEY` | ❌ you create it | you already have it |

**What this means in practice**

| Feature | Fresh clone | Full folder |
|---|---|---|
| Dashboard: patient list, SOFA, SHAP, risk, alert, trends, history, camera, validation drawers | ✅ works | ✅ |
| AI Clinical Report | ✅ with your own Groq key (without one the API answers `500 GROQ_API_KEY not configured`) | ✅ |
| Uncertainty interval | uses a fallback of **± 2.9** (the real value **± 3.28** needs `fl_training`) | ± 3.28 |
| New tests added 2026-11-27 (backend branches, training script, Streamlit page, migration) | ✅ all pass: they build their own temporary data | ✅ |
| Older API and FL tests | ⚠ need the sample database that `setup.py` creates (and 4 FL files need the `fl_training` table) | ✅ |
| Retraining (`train_federated.py`), real Flower demo (`server.py` / `client.py`) | ❌ need the `fl_training` table (BigQuery access for credentialed users — see `docs/SQL_COMMANDS.md`) | ✅ |
| Streamlit simulator (`fl_dashboard.py`) | ✅ (it is a simulation; it does not need the data) | ✅ |
| Database demo with 41,798 rows | ❌ | ✅ |

A normal reviewer can therefore run the **whole dashboard** from a fresh clone. Training-related demos need the full folder.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  React 19 + Vite Frontend  (port 5173)                          │
│  SelectPatient → MonitorDashboard                               │
│  Framer Motion · anime.js · Recharts · TanStack Query           │
│  MediaPipe FaceLandmarker (camera monitor, runs in the browser) │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP (Vite proxy  /api → localhost:8000)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                                   │
│  /health  /api/patients  /api/predict  /api/report              │
│  /api/history  /api/vitals                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐     ┌──────────────────────────────────┐
│  ML Service         │     │  LLM Service                     │
│  PyTorch DNN        │     │  openai/gpt-oss-120b (writer)    │
│  108 features       │     │  openai/gpt-oss-120b (judge)     │
│  SOFA prediction    │     │  90 s in-process report cache    │
│  SHAP DeepExplainer │     │  12 s judge timeout              │
│  Conformal interval │     │                                  │
└──────────┬──────────┘     └──────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  SQLite Database  (models/icu_monitor.db)                       │
│  sample_patients · patient_vitals · prediction_history          │
│  fl_training (MIMIC-III derived FL training splits)             │
└─────────────────────────────────────────────────────────────────┘
```

**Predict pipeline (one reading):**

```
sample_patients row → sliding window (last 20 readings)
  → 9 trend features + 7 latest vitals + 2 camera features + 90 TF-IDF note features
  → 108-feature vector → StandardScaler → Federated DNN → SOFA score
  → risk band + alert if SOFA ≥ 7 + SHAP top-5 + 90% conformal interval
```

**LLM report pipeline (runs when vitals change materially or the care mode is switched):**

```
SOFA + vitals + SHAP top-5 → openai/gpt-oss-120b (writer, temp 0.15, up to 6,000 tokens)
                           → openai/gpt-oss-120b (judge, temp 0, up to 1,200 tokens,
                                                  12 s timeout, first 900 chars of the report)
                           → 9-method reliability score → ReliabilityBadge
```

**Federated learning (training time, not used by the dashboard at run time):**

```
Hospital 0 ┐
Hospital 1 ├─ each trains locally (3 epochs/round) → sends weights only → FedAvg + FedProx
Hospital 2 ┘   raw patient rows never leave a hospital                      (100 rounds)
```

---

## Prerequisites

| Requirement | Version | Used by |
|---|---|---|
| Python | 3.12+ (developed on 3.12.6) | Backend, training, tests |
| Node.js + npm | 18+ | Frontend |
| Groq API key | — | AI Clinical Report (free at https://console.groq.com) |
| Modern browser (Chrome / Edge / Safari / Firefox) | — | Dashboard |
| Webcam | optional | Camera monitor. First use downloads a ~12 MB face model, so it needs internet |
| SQLite3 CLI | optional | Database inspection demo |

Python and JavaScript libraries are installed automatically by `setup.py`. The versions the project was built and tested with:

- **Python:** fastapi 0.138.2, uvicorn 0.49.0, torch 2.13.0, scikit-learn 1.9.0, shap 0.52.0, flwr 1.32.1, groq 1.5.0, streamlit 1.59.1
- **Frontend:** React 19.3.1, Vite 8.3.4, framer-motion 14.5.0, recharts 3.10.1, tailwindcss 3.4.19, TanStack Query 5.104.5, animejs 4.5.0, aos 2.3.4, react-router-dom 7.18.4, @mediapipe/tasks-vision 1.1.1, axios 1.20.0

(`requirements.txt` and `frontend/package.json` hold the exact constraints.)

---

## Installation

### Step 1 — Get the project

```bash
git clone <repository-url>
cd <project-folder>        # the folder that contains setup.py and requirements.txt
```

Or extract the ZIP and `cd` into the extracted folder.

### Step 2 — Run the automated setup

```bash
python setup.py            # same command on macOS / Linux / Windows
```

(On some systems the command is `python3 setup.py`.)

Add `--no-venv` to install into your system Python instead of creating `venv/`.

### Step 3 — Configure the Groq API key

`setup.py` copies `.env.example` to `.env`. Open `.env` and set your key:

```
GROQ_API_KEY=gsk_your_key_here
```

Never commit `.env`; it is git-ignored. Never paste the key into the code.

### Step 4 — Activate the virtual environment (every new terminal)

```bash
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows Command Prompt
venv\Scripts\Activate.ps1         # Windows PowerShell
```

You should see `(venv)` at the start of your prompt.

---

## Project Structure

```
icu_monitor/
│
├── backend/                      ← FastAPI application
│   ├── main.py                   ← App entry, CORS (ports 5173–5175), routers, /health
│   ├── api/
│   │   ├── patients.py           ← GET /api/patients, GET /api/patients/{id}
│   │   ├── predict.py            ← POST /api/predict — SOFA + SHAP + risk + alert
│   │   ├── report.py             ← POST /api/report — LLM assessment + reliability
│   │   ├── history.py            ← GET /api/history/{id}
│   │   └── vitals.py             ← GET /api/vitals/{id}
│   └── services/
│       ├── ml_service.py         ← DNN inference, SHAP, conformal interval, validation
│       └── llm_service.py        ← Groq calls, judge, 9-method score, 90 s cache
│
├── frontend/                     ← React 19 + Vite application
│   ├── src/
│   │   ├── pages/                ← SelectPatient.jsx, MonitorDashboard.jsx
│   │   ├── components/
│   │   │   ├── layout/           ← HeaderBar, AlertBanner, ThemeToggle, SystemStatusBar …
│   │   │   ├── overview/         ← SOFAGauge, VitalsGrid, VitalCard
│   │   │   ├── report/           ← ReportContainer, ReportSection, ReliabilityBadge,
│   │   │   │                        SHAPPanel, ReportToggle
│   │   │   ├── trends/           ← TrendsSection, TrendChart
│   │   │   ├── history/          ← HistoryDrawer, HistoryFAB
│   │   │   ├── cv/               ← CVMonitorModal (in-browser webcam), CVLivePanel (live right-hand panel), CVMonitorFAB
│   │   │   └── validation/       ← SOFA + LLM validation drawers and FABs
│   │   ├── hooks/useMonitor.js   ← 90 s polling loop + countdown
│   │   ├── contexts/             ← ThemeContext (light/dark)
│   │   ├── api/client.js         ← Axios client (proxied to :8000)
│   │   ├── App.jsx, main.jsx, index.css
│   ├── public/mediapipe/         ← MediaPipe WASM files (served locally)
│   ├── vite.config.js            ← proxies /api → localhost:8000 (default dev port 5173)
│   ├── playwright.config.js, vitest.config.js, vitest-setup.js
│   └── package.json
│
├── models/
│   ├── icu_monitor.db            ← SQLite DB (~56 MB, GIT-IGNORED, MIMIC-III derived)
│   ├── federated_model.pth       ← Best-round model weights (round 37)
│   ├── federated_model_latest.pth← Last round (written by server.py; not needed by the app)
│   ├── scaler.pkl                ← StandardScaler (108 features)
│   ├── tfidf_vectorizer.pkl      ← TF-IDF (90 SOFA-vocabulary terms)
│   ├── feature_columns.pkl       ← Ordered list of the 108 feature names
│   ├── shap_background.npy       ← 300 background samples for SHAP (300 × 108)
│   ├── training_metadata.json    ← Training config + final metrics
│   └── fl_test_X.npy / fl_test_y.npy ← Held-out test set (git-ignored)
│
├── data/
│   ├── sample_patients/          ← 3 source CSVs for migrate_to_db.py
│   └── fl_training/              ← Local training CSVs (git-ignored)
│
├── tests/
│   ├── fl/                       ← FL tests: unit + integration + system (360 tests)
│   └── cliniq/
│       ├── backend/              ← FastAPI tests (390 tests)
│       └── frontend/             ← Vitest (664) + Playwright E2E (33)
│
├── docs/                         ← Technical documentation (see Documentation Index)
│
├── db.py                         ← SQLite helpers
├── model_utils.py                ← ICUModel, ALERT_THRESHOLD, training/eval helpers
├── migrate_to_db.py              ← CSV → SQLite migration
├── train_federated.py            ← FL training (simulation mode)
├── server.py                     ← Flower FL server (real FL demo, 100 rounds)
├── client.py                     ← Flower FL client
├── fl_dashboard.py               ← Streamlit FL simulator
├── setup.py                      ← One-command setup
├── requirements.txt, pytest.ini
├── .env                          ← Your Groq key (never pushed)
├── .env.example                  ← Template
└── README.md
```

---

## COMPLETE DEMO GUIDE

This guide starts from a **brand-new machine**. Follow it in order.

**Recommended demo order**

1. ClinIQ Main Dashboard (the main deliverable)
2. API Swagger docs
3. FL Simulation Dashboard (Streamlit)
4. FL training in the terminal *(full folder only)*
5. Real Flower FL, 4 terminals *(full folder only)*
6. SQLite database inspection *(full folder for the big tables)*
7. Full test suite

The webcam monitor is part of Demo 1 (the camera button on the dashboard).

---

### Step 0 — Install system requirements (one-time)

`setup.py` cannot install these for you.

#### Python 3.12+

**macOS:** download from https://www.python.org/downloads/ or `brew install python@3.12`. Verify with `python3 --version`.

**Windows:** download the Windows installer from https://www.python.org/downloads/, tick **"Add Python to PATH"**, install, open a new terminal and run `python --version`.

#### Node.js 18+ and npm

**macOS:** download the LTS installer from https://nodejs.org/ or `brew install node`.
**Windows:** run the LTS `.msi` installer with the defaults (npm is included).
Verify: `node --version` (v18+) and `npm --version`.

> Node.js is only needed for the web dashboard (Demo 1). Everything else needs Python only.

#### SQLite3 CLI (optional, Demo 6)

macOS ships with it (`sqlite3 --version`). On Windows download `sqlite-tools-win-x64-*.zip` from https://www.sqlite.org/download.html and put `sqlite3.exe` in a folder on your PATH.

---

### Steps 1–4 — Project, setup, API key, venv

Do [Installation](#installation) Steps 1–4 above. `setup.py` performs these 7 checks and actions:

1. Python ≥ 3.12
2. Node ≥ 18 and npm
3. Create `venv/`
4. `pip install -r requirements.txt`
5. `npm install` in `frontend/`
6. Create `models/icu_monitor.db` with `migrate_to_db.py` **only if it does not already exist** (a fresh clone gets the 90-row sample database; you will see a harmless warning that `patient_vitals` is empty — it fills up when you use the dashboard)
7. Create `.env` from `.env.example` if missing

> **Dashboard address:** `setup.py` finishes by printing **http://localhost:5173** (Vite's default; an older version printed 5174 by mistake, now fixed). If 5173 is busy, Vite picks the next free port and tells you in its own output; the backend accepts 5173, 5174 and 5175.

---

### Demo 1 — ClinIQ Main Dashboard (full-stack web app)

> Requires: venv activated, Node.js installed, and a Groq key in `.env` for the AI report (the rest works without it).

**Terminal 1 — backend**

```bash
source venv/bin/activate                   # Windows: venv\Scripts\activate
uvicorn backend.main:app --port 8000 --reload
```

**Terminal 2 — frontend**

```bash
cd frontend
npm run dev
```

**Open:** http://localhost:5173  ·  **API docs:** http://localhost:8000/docs

> **Port 8000 must be free *and* be this project.** If another program already answers on port 8000 (another FastAPI app, for example) the dashboard will load but show errors or empty data, because the Vite proxy sends `/api` to whatever is on 8000. Check with `curl http://localhost:8000/health` and `curl http://localhost:8000/api/patients` — the second must list 3 patients. See [Troubleshooting](#troubleshooting).

**What to show**

- **Patient selection screen** — 3 ICU patients, live clock, ECG animation, system status bar.
- Click a patient — the Monitor Dashboard opens.
- **Header** — live 90-second countdown, LIVE chip, patient pill, risk-colored accent strip.
- **Patient context strip** — risk badge, SOFA score, **AI Reliability**.
- **Current Status** — circular SOFA gauge (green / yellow / red) and a vitals grid (HR, RR, SpO₂, Temp, SBP, DBP, MAP) with range classification.
- **Vital Trends (last 20 readings)** — Recharts line charts with normal-range bands.
- **AI Clinical Report** — four section cards (Condition, Cause, Forecast, Actions), the **Hospital ↔ Home Care** toggle, **SHAP panel** (top-5 drivers with animated bars) and the **Reliability badge** (9-method score).
- **Four floating buttons** on the right edge (bottom to top): **History** (past predictions), **Camera** (stress + GCS eye), **LLM validation** and **SOFA validation** (the evidence behind the model and the report).
- **Camera window with the live panel:** click the camera button. The picture is on the left; the **panel on the right** shows the predicted SOFA, a countdown to the next prediction, the two camera inputs (**live from camera** next to **used in last SOFA**, with a badge), the clinical note and the vitals. A line under the timer says whether the next prediction will use **camera** or **database** values. Close it with Esc, the ✕ or by clicking outside.
- **Dark / light** theme toggle.
- Wait one 90-second cycle: vitals, gauge, SHAP and (if values changed) the report all refresh.

---

### Demo 2 — API Swagger docs

Backend from Demo 1 must be running. Open http://localhost:8000/docs and use **Try it out → Execute**:

- `GET /health`
- `GET /api/patients` — the 3 ICU patients
- `GET /api/patients/1` — one patient
- `POST /api/predict` — body **must contain both `patient_id` and `row_idx`**:
  ```json
  {"patient_id": 1, "row_idx": 0}
  ```
  Optional camera fields: `cv_gcs_eye` (1–4) and `cv_stress` (0–10). Omitting `row_idx` returns `422`.
- `POST /api/report` — body `{"patient_id": 1, "care_mode": "hospital"}` (or `"homecare"`; any other value returns `400`). It must come **after** a predict for that patient (otherwise `409`), and it needs the Groq key (otherwise `500`).
- `GET /api/vitals/1?limit=20` — last readings (limit 1–20)
- `GET /api/history/1?limit=50` — prediction log (limit 1–50)

Full field descriptions: [REST API Reference](#rest-api-reference).

---

### Demo 3 — FL Simulation Dashboard (Streamlit)

An **animated simulation** of how federated learning works (it illustrates the process; it does not run the real trainer).

```bash
source venv/bin/activate
streamlit run fl_dashboard.py
```

**Open:** http://localhost:8501

**Controls (sidebar)**

| Control | Range / options | Default |
|---|---|---|
| Rounds | 5 – 15 | 10 |
| Speed (s / phase) | 0.5 – 5.0 | 1.5 |
| Aggregation Algorithm | FedAvg · FedYogi · FedAvg + FedProx · FedYogi + FedProx | FedYogi + FedProx |
| Buttons | ▶ Start · ⏹ Stop · ↺ Reset | |

**What to show:** per-round progress, global and per-hospital loss curves, MAE / R² curves, round log, weight statistics, layer-1 weight histogram, and the best-round marker. Try *FedAvg* versus *FedYogi* and compare convergence.

> **Note:** the simulator's default (and help text) mention "FedYogi + FedProx". The **final trained model used FedAvg + FedProx** (`USE_FEDYOGI = False`). Pick **FedAvg + FedProx** to match the real run. The real run also uses 100 rounds, which the simulator caps at 15 for speed.

---

### Demo 4 — FL training in the terminal *(full folder only)*

```bash
source venv/bin/activate
python train_federated.py
```

Requires the `fl_training` table and the held-out test files. Typical output:

```
[1/6] Loading hospital datasets from database (fl_training table)...
  Hospital 0 (General ICU): 13,932 samples — ...
  Hospital 1 (Mixed ICU): 13,933 samples — ...
  Hospital 2 (Cardiac/Trauma ICU): 13,933 samples — ...
[2/6] Saving SHAP background samples...
[3/6] Starting FL simulation...
  Model              : 108 → 256 → 128 → 64 → 1  (ReLU, Dropout 0.3)
  Server aggregation : FedAvg + FedProx (μ=0.5)
  Loss               : Plain MSE
  [Server] Round 001 — lr=0.00100 | ...
  ...
[4/6] Saving model (best was round 37)...
[5/6] Evaluating on held-out test set (10,450 ICU stays)...
  MAE  : 1.8006 SOFA points
  R²   : 0.4806
  High-risk alert  AUROC   : 0.9181
  High-risk alert  PR-AUC  : 0.6466
[6/6] Saving training metadata...
```

A full run is 100 rounds and takes a long time (tens of minutes to about an hour depending on the machine). **Back up `models/` first — this overwrites the trained model.** For a demo, press Ctrl+C after a few rounds to show the live logs. Details: `docs/TRAIN_FEDERATED.md`.

---

### Demo 5 — Real Flower federated learning (4 terminals) *(full folder only)*

A real Flower gRPC server coordinates 3 hospital clients. Each client trains on its own rows and sends **only weights**.

```bash
# Terminal 1
python server.py                 # listens on 127.0.0.1:8080, NUM_ROUNDS = 100
# Terminals 2, 3, 4
python client.py --client_id 0
python client.py --client_id 1
python client.py --client_id 2
```

The server waits for all 3 clients, then runs 100 rounds, saves the best round to `models/federated_model.pth` and every round to `models/federated_model_latest.pth`. Use a copy of the project for this demo if you want to keep the trained model untouched. Details: `docs/FL_SERVER_CLIENT_DB_TESTING.md`.

---

### Demo 6 — SQLite database inspection

```bash
sqlite3 models/icu_monitor.db        # Windows: sqlite3 models\icu_monitor.db
```

```sql
.headers on
.mode column
.tables

-- Row counts
SELECT 'patient_vitals' AS table_name, COUNT(*) AS total_rows FROM patient_vitals
UNION ALL SELECT 'prediction_history', COUNT(*) FROM prediction_history
UNION ALL SELECT 'sample_patients',    COUNT(*) FROM sample_patients
UNION ALL SELECT 'fl_training',        COUNT(*) FROM fl_training;

.schema sample_patients
.schema fl_training

SELECT * FROM sample_patients LIMIT 10;
SELECT patient, COUNT(*) FROM sample_patients GROUP BY patient;
SELECT client_id, COUNT(*) FROM fl_training GROUP BY client_id;

-- Latest prediction per patient
SELECT patient, Timestamp, SOFA, Risk, HR, SpO2, MAP
FROM prediction_history
WHERE id IN (SELECT MAX(id) FROM prediction_history GROUP BY patient);

.quit
```

**Expected row counts**

| Table | Full folder | Fresh clone (new DB) | Meaning |
|---|---|---|---|
| `sample_patients` | 90 | 90 | 3 patients × 30 readings — feed the 90 s cycle |
| `patient_vitals` | 60 | 0 until you use the dashboard | Sliding window: 20 newest readings per patient |
| `prediction_history` | 150 | 0 until you use the dashboard | Sliding window: 50 newest predictions per patient |
| `fl_training` | 41,798 | 0 | One row per training ICU stay; hospital split 13,932 / 13,933 / 13,933 |

The full-folder database is about **56 MB**; a fresh-clone database is about 0.1 MB.

---

### Demo 7 — Full test suite

Activate the venv and run from the project root. Details and measured coverage: `docs/CLINIQ_TESTING.md`.

```bash
python -m pytest tests/fl/unit/ -v                  # FL unit tests (fast)
python -m pytest tests/fl/integration/ -v           # FL integration
python -m pytest tests/fl/system/ -v -m slow        # FL end-to-end system tests
python -m pytest tests/cliniq/backend/ -v           # ClinIQ backend API tests
python -m pytest tests/ -v                          # all 750 Python tests (about 2 minutes)
python -m pytest tests/ -m "not slow" -v            # skip the 83 slow ones (667 tests)

cd frontend && npx vitest run && cd ..              # 664 Vitest tests (about 16 seconds)

cd frontend
npx playwright install chromium                     # one-time
npm run test:e2e                                    # 33 Playwright tests (3–4 minutes)
cd ..
```

**Expected results**

```
Python (pytest)      750 passed   (360 FL + 390 backend; 83 marked slow, 10 of those call the Groq API)
Frontend (Vitest)    664 passed   (53 test files)
─────────────────────────────────
Unit/integration    1,414 passed
Browser (Playwright)  33 tests    (6 need the Groq API; 2 of those can time out in a full run)
─────────────────────────────────
Grand total        1,447 tests
```

> Some tests need a full folder (the held-out test files / `fl_training`). On a fresh clone those are skipped or unavailable; see `docs/CLINIQ_TESTING.md`.
> The Playwright tests need the ClinIQ backend on **port 8000**. If something else already uses that port they time out.
> In a full Playwright run the two slowest report tests (`report section shows 4 content cards after load` and `reliability badge appears with report`) can time out at 30 seconds when Groq is slow after many calls in a row; they pass when re-run on their own.

---

## Dashboard Features

### Patient Selection (`/`)

- Three ICU patient profiles from the `sample_patients` table.
- Live clock, ECG waveform animation and a system status bar.
- Dark / light theme persists across navigation.

### Monitor Dashboard (`/monitor/:id`)

**Header bar** — ClinIQ brand, LIVE chip with ECG animation, 90-second countdown, patient pill, risk-colored accent strip.

**Patient Context Strip** — risk badge, SOFA score, AI Reliability (the same score as the report). Dims while refreshing.

**Current Status**
- `SOFAGauge` — circular gauge, risk-colored, animated on new data.
- `VitalsGrid` — 7 vital-sign cards (HR, RR, SpO₂, Temp, SBP, DBP, MAP) with range classification.

**Vital Trends (last 20 readings)** — line charts with normal-range bands and abnormal highlighting.

**AI Clinical Report**
- Hospital / Home Care toggle.
- `ReliabilityBadge` — score dial (0–100%), label and caution text.
- Four section cards: Current Condition · Probable Cause · Risk Forecast · Immediate Actions.
- `SHAPPanel` — "Key Risk Drivers" with animated bars (anime.js), ↑/↓ risk arrows and impact badges.

**Floating buttons (right edge, bottom to top)**

| Button | Opens |
|---|---|
| History | Drawer with the patient's past predictions (up to 50) |
| Camera | Webcam stress + GCS-eye monitor |
| LLM validation | How the AI report is graded |
| SOFA validation | Evidence for the SOFA model (metrics, thresholds, calibration) |

### 90-second refresh cycle

`useMonitor.js` drives it:

1. The timer counts down 90 → 1 and holds at 1 while a request is in flight.
2. `runPredict` asks the backend for the next `sample_patients` row; the backend runs the DNN + SHAP.
3. The timer resets to 90 only after the response arrives.
4. `ReportContainer` calls the LLM only if SOFA or key vitals actually changed since the last report, or the care mode was switched.

---

## Camera Monitor (Stress + GCS Eye)

- Opens from the **Camera** floating button, in a window with the camera picture on the left and a **live model panel** on the right (see below).
- **Runs entirely in your browser** (MediaPipe FaceLandmarker compiled to WebAssembly). Video is never uploaded. The camera permission is a **browser** permission, not a Terminal/OS-app one.
- Estimates a **stress score (0–10)** from facial action units (PSPI) and a **GCS eye-opening score (1–4)** over a **30-second window**, then sends them to `/api/predict` as `cv_stress` and `cv_gcs_eye`.
- They are 2 of the model's 108 inputs; effect on SOFA is small (about 0.3–1.4 points) and does not change the risk band or the alert.
- First use downloads a ~12 MB face model from the internet.
- Full explanation: `docs/CV_MONITOR.md`.

### The live model panel (right-hand side of the camera window)
It exists to show that the camera values really go into the model and what comes out. From top to bottom:

| Block | What it shows |
|---|---|
| **Predicted SOFA (output)** | The score, risk level, 90% range and an ALERT tag |
| **Next SOFA prediction (timer)** | The 90-second countdown (`m:ss`) and a line: **"At 0 → CAMERA values go into the model"** while a calibrated camera sees a face, otherwise **"At 0 → DATABASE values go into the model"** |
| **Input 1 and 2: camera features** | Stress and GCS eye, each as **Live from camera** next to **Used in last SOFA**, with a badge: *WAITING FOR CAMERA*, *SENDS NEXT CYCLE* or *IN MODEL ✓* |
| **Input 3: clinical note** | The note text behind the 90 text features |
| **Other inputs: vitals** | HR, RR, SpO₂, Temp, SBP, DBP, MAP |

**To demonstrate it:** open the camera with no one in front of it (badges say WAITING, the line says DATABASE) → face the camera until calibration ends (the line turns purple) → wait for the timer to reach zero (the SOFA updates and the badge turns **IN MODEL ✓**) → close the camera (the next prediction uses the database values again). The SOFA changes only a little (about 0.3–1.4 points), by design of the trained model.

---

## AI Clinical Report

### LLM models used

| Role | Model | Settings | Purpose |
|---|---|---|---|
| Writer | `openai/gpt-oss-120b` (Groq) | temperature 0.15, up to 6,000 tokens | Writes the 4-section assessment |
| Judge | `openai/gpt-oss-120b` (Groq) | temperature 0, up to 1,200 tokens, 12 s timeout, first 900 characters | G-Eval + RAGAS reliability scoring |

### 9-method reliability score

| # | Method | Weight | What it checks |
|---|---|---|---|
| 1 | Factual Grounding | 13% | Does the response acknowledge each abnormal vital sign? |
| 2 | Clinical Hard Rules | 13% | e.g. MAP < 65 → vasopressors, SpO₂ < 90 → oxygen mentioned? |
| 3 | SHAP-LLM Coherence | 9% | Does it address the top SHAP drivers? |
| 4 | Response Structure | 9% | All 4 sections present and detailed? |
| 5 | Severity Calibration | 8% | Urgency language matches the SOFA level? |
| 6 | Contraindication Check | 5% | No dangerous recommendations for these vitals? |
| 7 | Numeric Accuracy | 8% | Numbers quoted match the real vitals? |
| 8 | G-Eval Quality | 13% | Judge LLM: factual accuracy + clinical appropriateness |
| 9 | RAGAS Faithfulness | 22% | Judge LLM: statements supported by the patient's data? |

| Score | Label |
|---|---|
| ≥ 75% | High Reliability |
| 55–74% | Moderate Reliability |
| < 55% | Low Reliability |

A score below **60%** also shows a caution message. Full explanation: `docs/VALIDATION.md`.

### Caching

- **Backend:** reports are cached for 90 seconds, keyed by care mode + patient + SOFA + every vital quoted in the prompt. Identical concurrent requests share one LLM call; failed reports are not cached.
- **Frontend:** once the current mode's report loads, the other care mode is generated in the background so switching is instant. A new reading clears both.

---

## REST API Reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/api/patients` | List the 3 patients |
| GET | `/api/patients/{id}` | One patient |
| POST | `/api/predict` | Body: `patient_id`, `row_idx` (both required); optional `cv_gcs_eye` (1–4), `cv_stress` (0–10). Returns SOFA, risk band, alert flag, SHAP top-5, conformal interval, vitals |
| POST | `/api/report` | Body: `patient_id`, `care_mode` (`hospital` or `homecare`, else `400`). `409` if no earlier predict, `500` if no Groq key |
| GET | `/api/history/{id}?limit=` | Prediction log, `limit` 1–50 |
| GET | `/api/vitals/{id}?limit=` | Latest vitals, `limit` 1–20 |

CORS accepts the frontend on ports 5173, 5174 and 5175.

---

## Model Details

### Architecture

```
Input: 108 features
  ├─ 9 trend features  (HR_mean, HR_std, RR_mean, SpO₂_mean, SpO₂_min,
  │                     Temp_mean, SBP_mean, DBP_mean, MAP_mean)
  ├─ 7 latest vitals   (latest_HR, latest_RR, latest_SpO₂, latest_Temp,
  │                     latest_SBP, latest_DBP, latest_MAP)
  ├─ 2 camera features (GCS_eye_opening, stress_score)
  └─ 90 TF-IDF         (SOFA vocabulary whitelist, 6 organ components)

DNN:  108 → Linear(256) → ReLU → Dropout(0.3)
           → Linear(128) → ReLU → Dropout(0.3)
           → Linear(64)  → ReLU → Dropout(0.3)
           → Linear(1)   → SOFA score (0–24)

Total parameters: 69,121
```

Dropout switches off 30% of neurons randomly **during training only**, so the model cannot lean on a few neurons. At prediction time (backend, SHAP, evaluation) dropout is off, so the same input always gives the same SOFA.

The 108 inputs are fixed. Most SOFA points come from labs (bilirubin, creatinine, platelets, PaO₂/FiO₂) and vasopressor doses, which are deliberately **not** inputs (non-invasive design); this limits accuracy.

### Performance (held-out test set — 10,450 ICU stays: 602 High, 2,821 Moderate, 7,027 Low)

| Metric | Value |
|---|---|
| MAE | **1.8006** SOFA points |
| R² | **0.4806** |
| Prediction range | 1.78 – 23.07 |
| AUROC (high-risk alert) | **0.9181** |
| PR-AUC (high-risk alert) | **0.6466** |

For comparison, a centralised gradient-boosting model on the same inputs scores MAE 1.79, R² 0.50, AUROC 0.922 — federated training costs almost nothing in accuracy while keeping data at each hospital.

**MAE per risk band:** Low 1.33 · Moderate 2.44 · High 4.36. Per-band R² is negative by design (little variance inside one band), so only the overall R² is meaningful.

**High-risk alert** — fires when predicted SOFA ≥ 7 (`ALERT_THRESHOLD = 7.0` in `model_utils.py`), scored against actual SOFA ≥ 10:

| Metric | Value |
|---|---|
| Recall (sensitivity) | **67.8%** (408 of 602 high-risk stays caught) |
| Specificity | 96.6% |
| Precision | 54.9% |
| F1 | 0.607 |
| Missed (false negatives) | 194 |
| False alarms (false positives) | 335 |
| Correct no-alarm (true negatives) | 9,513 |

**Why 7 and not 10?** The model under-predicts the sickest patients: for truly high-risk stays the mean prediction is about **9.5** while the true mean is about **13.1**. A trigger of 10 would miss more than half of them.

| Threshold | Recall | Specificity |
|---|---|---|
| SOFA ≥ 10 | 43.9% | 99.3% |
| SOFA ≥ 9 | 51.5% | 98.8% |
| SOFA ≥ 8 | 59.6% | 97.9% |
| SOFA ≥ 7.5 | 64.3% | 97.3% |
| **SOFA ≥ 7 (current)** | **67.8%** | **96.6%** |
| SOFA ≥ 6.5 | 70.6% | 95.6% |
| SOFA ≥ 6 | 73.1% | 94.1% |

**3-band confusion matrix** (rows = actual, columns = predicted Low / Moderate / High): 

| Actual \ Predicted | Low | Moderate | High |
|---|---|---|---|
| Low | 6,714 | 306 | 7 |
| Moderate | 2,086 | 669 | 66 |
| High | 119 | 219 | 264 |

3-band accuracy is **73.2%**. The 90% conformal interval shown on the dashboard is **± 3.28** SOFA points (the 90th percentile of absolute training errors; the backend falls back to ± 2.9 if the `fl_training` table is missing).

> **Limitations:** one row per ICU stay, split 80/20 by ICU stay, so a patient with several stays can appear in both train and test. Results are retrospective MIMIC-III numbers, not independent clinical validation. See `docs/MODEL_DISCUSSION.md`.

### Training configuration

- **Server aggregation:** FedAvg (`USE_FEDYOGI = False` in `train_federated.py`; if switched on it uses η = 0.001)
- **Client regularisation:** FedProx, μ = 0.5
- **Loss:** plain MSE
- **Optimizer:** AdamW, lr 0.001 decaying ×0.99 per round (floor 1e-4), weight decay 1e-4, gradient clip 1.0
- **Schedule:** 100 FL rounds × 3 local epochs × 3 hospitals, batch size 64
- **Data:** 41,798 training stays split IID across 3 hospitals (13,932 / 13,933 / 13,933)
- **Model selection:** the round with the lowest average server validation MSE is kept — **round 37** in the final run
- Oversampling and differential privacy are off

---

## Federated Learning Training

### Retrain (simulation mode) — full folder only

```bash
python train_federated.py
```

Phase 0 (BigQuery download and note processing) runs only if `fl_training` is empty; Phases 1–6 train, evaluate and save. Running BigQuery requires PhysioNet-credentialed access; the SQL is in `docs/SQL_COMMANDS.md`. Full explanation: `docs/TRAIN_FEDERATED.md`.

After retraining, restart the backend to load the new weights.

### Real FL demo

See [Demo 5](#demo-5--real-flower-federated-learning-4-terminals-full-folder-only). Server: `python server.py`; clients: `python client.py --client_id 0|1|2`.

---

## Dataset Information

**Source:** MIMIC-III, accessed through Google BigQuery (credentialed access).

- **Project / dataset:** `mimic-project-2` / `mimic-project-2.dataset_v2` (SQL in `docs/SQL_COMMANDS.md`)
- **Processed view:** `ml_dataset_final` — 5,089,857 hourly rows
- **Cohort:** 52,248 adult ICU stays (≥ 6 h) from 38,304 patients
- **Split:** 80/20, stratified by SOFA ≥ 10 → **41,798 train / 10,450 test**
- **`sample_patients`** (90 rows): 3 patients × 30 readings; drives the live simulation
- **`fl_training`** (41,798 rows): one row per training stay (the newest hour), SOFA label at that hour, StandardScaler-normalised and clipped to ±10, split IID across 3 hospital clients

**Privacy:** MIMIC-III-derived files (`icu_monitor.db`, `fl_training`, test arrays) are git-ignored and may only be handed to PhysioNet-credentialed recipients.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite 8, Tailwind CSS 3, Framer Motion 14 |
| Animations | anime.js 4 (SHAP bars), AOS 2 (scroll reveal), Framer Motion (transitions) |
| Charts | Recharts 3 |
| Data fetching | TanStack Query 5, Axios |
| Routing | react-router-dom 7 |
| Backend | FastAPI, uvicorn |
| ML framework | PyTorch |
| Federated learning | Flower (flwr 1.x) — FedAvg + FedProx |
| Explainability | SHAP DeepExplainer (top-5) |
| FL simulator | Streamlit (`fl_dashboard.py`) |
| Computer vision | MediaPipe Tasks Vision (browser, WASM) |
| LLM (writer + judge) | Groq `openai/gpt-oss-120b` |
| Database | SQLite 3 |
| Config | python-dotenv |
| Testing | pytest, Vitest, Playwright |
| Dataset | MIMIC-III (Google BigQuery) |

---

## Database Inspection (SQLite CLI)

```bash
sqlite3 models/icu_monitor.db
```

```sql
.headers on
.mode column

SELECT 'patient_vitals' AS tbl, COUNT(*) AS rows FROM patient_vitals
UNION ALL SELECT 'prediction_history', COUNT(*) FROM prediction_history
UNION ALL SELECT 'sample_patients',    COUNT(*) FROM sample_patients
UNION ALL SELECT 'fl_training',        COUNT(*) FROM fl_training;

SELECT patient, MAX(time) AS latest FROM patient_vitals GROUP BY patient;

SELECT Timestamp, SOFA, Risk, HR, SpO2, MAP
FROM prediction_history WHERE patient = 1 ORDER BY id DESC LIMIT 5;

SELECT patient, COUNT(*) AS rows FROM sample_patients GROUP BY patient;
SELECT client_id, COUNT(*) AS rows FROM fl_training GROUP BY client_id;

.quit
```

---

## Test Suite

> Activate the venv and run from the project root.

| Layer | Command | Tests |
|---|---|---|
| FL (unit + integration + system) | `python -m pytest tests/fl/ -v` | 360 |
| ClinIQ backend | `python -m pytest tests/cliniq/backend/ -v` | 390 |
| **All Python** | `python -m pytest tests/ -v` (add `-m "not slow"` to skip 83) | **750** |
| React (Vitest) | `cd frontend && npx vitest run` | **664** (53 files) |
| Browser E2E (Playwright) | `cd frontend && npm run test:e2e` | **33** |

**1,414** unit, integration and system tests (750 + 664) plus **33** browser tests = **1,447** in total.

**Measured coverage (2026-11-27):**

| Area | Coverage |
|---|---|
| Python, whole suite (750 tests) | **99%** of statements; 100% of every backend file, 99% of `train_federated.py`, 97% of `fl_dashboard.py` |
| Python without the 83 slow tests | 85% (the Streamlit simulator and the full training path need the slow tests) |
| Frontend (664 tests) | **100%** of statements, **96.7%** of branches, **100%** of functions; every source file is at or above 90% |

These were 83% (Python) and 79.8% / 71.0% (frontend statements / branches) before 2026-11-27. The 17 uncovered Python lines are edge branches that the synthetic test data never reaches. Full report, what each test checks, and the known gaps: `docs/CLINIQ_TESTING.md` (ClinIQ) and `docs/FL_SERVER_CLIENT_DB_TESTING.md` (federated learning).

**E2E prerequisites:** the ClinIQ backend must run on port 8000, and the specs (which live outside `frontend/`) need a symlink: from the project root run `ln -s frontend/node_modules node_modules` (git-ignored).

---

## Documentation Index

| File | What it explains |
|---|---|
| `docs/Project_Context.md` | Whole-project background and decisions |
| `docs/MODEL_DISCUSSION.md` | Model design, results, limitations |
| `docs/INPUT_OUTPUT.md` | Every model input and output |
| `docs/TRAIN_FEDERATED.md` | `train_federated.py` phase by phase |
| `docs/FL_SERVER_CLIENT_DB_TESTING.md` | Flower server/client, database, FL tests |
| `docs/VALIDATION.md` | SOFA validation and the 9-method LLM score |
| `docs/CV_MONITOR.md` | Camera monitor |
| `docs/CLINIQ_TESTING.md` | Backend/frontend/E2E tests and coverage |
| `docs/SQL_COMMANDS.md` | BigQuery SQL used to build the dataset |
| `docs/SKILL.md` | Project skill/assistant notes |

---

## Troubleshooting

**`ModuleNotFoundError` on backend start** — activate the venv, then `pip install -r requirements.txt`.

**Dashboard loads but shows errors / no patients** — something else is probably using port **8000**. Check:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/patients     # must list 3 patients
```
If `/api/patients` gives 404, another app owns the port. Start ClinIQ on a free port (e.g. `uvicorn backend.main:app --port 8001`) and point the Vite proxy in `frontend/vite.config.js` to it, or stop the other app **yourself**. Do not kill processes you do not recognise.

**Port already in use (5173 / 8000 / 8080 / 8501)** — find the owner first, then stop only your own process:
```bash
lsof -i :8000                       # macOS / Linux: shows the PID and program
kill <pid>                          # then retry
```
Windows:
```cmd
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

**Report fails / "No assessment available"** — confirm `GROQ_API_KEY` is set in `.env` (the API says `500 GROQ_API_KEY not configured` otherwise), the key is active at https://console.groq.com, restart the backend after editing `.env`, and check the backend terminal.

**`409` from `/api/report`** — call `/api/predict` for that patient first (the dashboard does this automatically).

**`422` from `/api/predict`** — the body needs both `patient_id` and `row_idx`.

**Uncertainty interval is ± 2.9 instead of ± 3.28** — the `fl_training` table is empty (fresh clone). Expected; see [Read This First](#read-this-first--what-you-get-from-a-fresh-clone).

**Warning "patient_vitals is empty" during setup** — harmless; the table fills when the dashboard runs.

**Dashboard does not open at the address `setup.py` printed** — it prints **http://localhost:5173**. If another program already uses 5173, Vite starts on the next free port (5174, 5175…) and prints the real address in its own output; use that one.

**sklearn `InconsistentVersionWarning` on startup** — harmless; the scaler and TF-IDF were pickled with scikit-learn 1.9.0 (`requirements.txt` pins a compatible version).

**Database not found** — `python migrate_to_db.py` (creates a sample-only database from the CSVs).

**Retraining or Flower demo fails with "fl_training empty" / missing test files** — these need the full folder (see the table at the top), or credentialed BigQuery access.

**Flower "address already in use" on 8080** — `lsof -i :8080`, stop that process, then rerun `python server.py`.

**Camera monitor: no webcam / permission denied** — close apps that hold the webcam (FaceTime, Zoom); allow the camera for the **browser tab** (address-bar lock icon; on macOS also System Settings → Privacy & Security → Camera → your browser). The first use needs internet to download the face model.

**E2E tests time out** — port 8000 is not the ClinIQ backend, or the `node_modules` symlink is missing (see [Test Suite](#test-suite)).

---

## Quick Start

```bash
# 1. Setup (venv, pip, npm, sample DB, .env)
cd <project-folder>
python setup.py

# 2. Put your Groq key in .env
#    GROQ_API_KEY=gsk_your_key_here

# 3. Terminal 1 — backend
source venv/bin/activate            # Windows: venv\Scripts\activate
uvicorn backend.main:app --port 8000 --reload

# 4. Terminal 2 — frontend
cd frontend && npm run dev

# Dashboard → http://localhost:5173
# API docs  → http://localhost:8000/docs
```

**Manual setup** (instead of `setup.py`):

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd frontend && npm install && cd ..
python migrate_to_db.py                              # only if models/icu_monitor.db is missing
cp .env.example .env                                 # Windows: copy .env.example .env
```

---

*Final Year Capstone — Computer Science Engineering*
*Dataset: MIMIC-III · ML: PyTorch Federated DNN · LLM: Groq openai/gpt-oss-120b · Frontend: React 19 + Vite*

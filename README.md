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

```bash
git clone <repository-url>
cd <project-folder>        # the folder name can be anything
```

Or download and extract the ZIP, then `cd` into the extracted folder.

### Step 2 — Run the automated setup

```bash
python setup.py            # macOS / Linux / Windows — same command
```

This handles everything: creates `venv/`, installs pip packages, installs npm packages, initialises the database, and creates `.env`.  
See the [COMPLETE DEMO GUIDE](#complete-demo-guide) for full details and the `--no-venv` option.

### Step 3 — Configure the Groq API key

Open the `.env` file and set your key:

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [https://console.groq.com](https://console.groq.com)

### Step 4 — Activate the virtual environment

macOS / Linux:
```bash
source venv/bin/activate
```

Windows:
```cmd
venv\Scripts\activate
```

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

This guide starts from a **brand-new machine** — it covers installing every tool, setting up the project, and running all 11 demo components. Follow the steps in order on a fresh system.

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

### Step 0 — Install System Requirements (one-time, new machine only)

These tools must be installed at the system level before anything else. `setup.py` cannot install them for you.

#### Python 3.12+

**macOS:**
```bash
# Option A — download the installer (easiest)
# Go to https://www.python.org/downloads/ and install the latest 3.12.x pkg

# Option B — via Homebrew
brew install python@3.12

# Verify
python3 --version    # should print Python 3.12.x or later
```

**Windows:**
```
1. Go to https://www.python.org/downloads/
2. Download the latest Python 3.12.x Windows installer
3. Run it — check the box "Add Python to PATH" before clicking Install
4. Open a new Command Prompt or PowerShell and verify:
```
```powershell
python --version    # should print Python 3.12.x or later
```

---

#### Node.js 18+ and npm

**macOS:**
```bash
# Option A — download the installer (easiest)
# Go to https://nodejs.org/ and download the LTS version

# Option B — via Homebrew
brew install node

# Verify
node --version    # v18.x or later
npm  --version    # 9.x or later
```

**Windows:**
```
1. Go to https://nodejs.org/
2. Download the LTS installer (Windows Installer .msi)
3. Run the installer — keep all defaults (includes npm automatically)
4. Open a new Command Prompt or PowerShell and verify:
```
```powershell
node --version    # v18.x or later
npm  --version    # 9.x or later
```

> **Note:** Node.js is only needed for the React frontend (Demo 1). All other demos (backend, FL, CV, tests) work with Python only.

---

#### SQLite3 CLI (for Demo 10 — database inspection)

**macOS:** Already installed. Verify:
```bash
sqlite3 --version    # 3.x.x
```

**Windows:**
```
1. Go to https://www.sqlite.org/download.html
2. Under "Precompiled Binaries for Windows", download sqlite-tools-win-x64-*.zip
3. Extract it — you get sqlite3.exe
4. Move sqlite3.exe to C:\Windows\ (or any folder that is in your PATH)
5. Open a new Command Prompt and verify:
```
```powershell
sqlite3 --version    # 3.x.x
```

---

### Step 1 — Get the Project

**If you have Git:**
```bash
git clone <repository-url>
cd <project-folder>          # the folder name can be anything
```

**If you downloaded a ZIP:**
```bash
# Extract the ZIP, then cd into the extracted folder
cd <extracted-folder>        # macOS / Linux
cd <extracted-folder>        # Windows (same command in Command Prompt / PowerShell)
```

> All commands from here run from inside this project folder (the one that contains `requirements.txt` and `setup.py`).

---

### Step 2 — Run the Automated Setup

A single script handles everything: creates a virtual environment, installs all Python and npm packages, initialises the SQLite database, and creates the `.env` file.

```bash
python setup.py              # macOS / Linux
python setup.py              # Windows (same command)
```

What it does automatically:
1. Checks Python ≥ 3.12
2. Checks Node.js ≥ 18 and npm
3. Creates `venv/` virtual environment
4. Installs all Python packages from `requirements.txt`
5. Installs all npm packages from `frontend/package.json`
6. Initialises the SQLite database (runs `migrate_to_db.py`)
7. Creates `.env` from `.env.example`

To skip the virtual environment and install into system Python instead:
```bash
python setup.py --no-venv
```

---

### Step 3 — Add Your Groq API Key

Open the `.env` file that setup.py just created and add your key:

```
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at https://console.groq.com  
(The AI Clinical Report in Demo 1 won't work without this. All other demos work fine without it.)

---

### Step 4 — Activate the Virtual Environment

**Do this once in every new terminal window before running any demo.**

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script with an execution policy error, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> then try activating again.

You will see `(venv)` at the start of your prompt when it is active. Every terminal you open for the demos below needs this activation step.

---

---

### Demo 1 — ClinIQ Main Dashboard (Full-Stack Web App)

The primary deliverable: React 19 + FastAPI + PyTorch + Groq LLM, all wired together.

> Requires: venv activated, Groq API key in `.env`, Node.js installed.

**Terminal 1 — Activate venv and start the FastAPI backend:**

macOS / Linux:
```bash
source venv/bin/activate
uvicorn backend.main:app --port 8000 --reload
```

Windows:
```cmd
venv\Scripts\activate
uvicorn backend.main:app --port 8000 --reload
```

**Terminal 2 — Start the React frontend (open a new terminal window):**

macOS / Linux:
```bash
source venv/bin/activate
cd frontend
npm run dev
```

Windows:
```cmd
venv\Scripts\activate
cd frontend
npm run dev
```

**Open in browser:** http://localhost:5174  
**API docs:** http://localhost:8000/docs

**What to show:**
- Patient selection screen — 3 ICU patients with ECG animation
- Click a patient — Monitor Dashboard loads
- Live 90-second countdown timer in the header
- SOFA score gauge (circular, color-coded: green / yellow / red)
- Vitals Grid — 7 vital signs (HR, RR, SpO₂, Temp, SBP, DBP, MAP) with range classification
- Vital Trends — Recharts line charts for last 20 readings with normal-range bands
- AI Clinical Report — 4 sections (Condition, Cause, Forecast, Actions) generated by Groq LLM
- Reliability Badge — 0–100% LLM reliability score via 9-method evaluation
- SHAP Panel — top-5 risk drivers with animated bars (anime.js spring physics)
- Toggle Hospital ↔ Home Care mode — LLM regenerates report instantly
- Dark / light theme toggle (top-right corner)
- Wait for the 90-second cycle — watch vitals update live and SOFA gauge animate

**If port 8000 is already in use:**

macOS / Linux:
```bash
lsof -ti :8000 | xargs kill -9
```

Windows (Command Prompt — find PID first, then kill):
```cmd
netstat -ano | findstr :8000
taskkill /PID <pid-from-above> /F
```

---

### Demo 2 — API Swagger Docs (Interactive REST API)

Shows all backend endpoints with live request testing. Backend from Demo 1 must be running.

**Open in browser:** http://localhost:8000/docs

**What to show:**
- `GET /api/patients` — list all 3 ICU patients (click "Try it out" → Execute)
- `GET /api/patients/1` — single patient details
- `POST /api/predict` — run SOFA prediction + SHAP (body: `{"patient_id": 1}`)
- `POST /api/report` — trigger LLM clinical assessment
- `GET /api/vitals/1` — last 20 vital sign readings
- `GET /api/history/1` — full prediction history log
- Raw JSON: http://localhost:8000/api/patients

---

### Demo 3 — FL Simulation Dashboard (Streamlit)

Interactive visualisation of the Federated Learning process: pick an algorithm, watch rounds train, and explore weight evolution charts.

> Open a new terminal window for this. Keep Demo 1 backend running if you want both at once.

macOS / Linux:
```bash
source venv/bin/activate
streamlit run fl_dashboard.py
```

Windows:
```cmd
venv\Scripts\activate
streamlit run fl_dashboard.py
```

**Open in browser:** http://localhost:8501

**What to show:**
- Sidebar: number of rounds (1–100), algorithm (FedAvg / FedYogi / FedProx), clients, speed
- Click **Start FL Simulation** — per-round progress bars update live
- Global Loss chart — training loss curve across rounds
- Client Loss chart — per-hospital loss (Hospital 0, 1, 2)
- Global MAE and R² curves — validation metrics improving per round
- Round Log table — H0 Loss, H1 Loss, H2 Loss, Global per round
- Weight Statistics heatmap — mean / std / delta per client per round
- Layer 1 weights histogram — distribution shift before vs after aggregation
- Best Round indicator — round with lowest validation MAE highlighted
- Try FedYogi vs FedAvg: run 20 rounds each and compare convergence curves

**If port 8501 is busy:**

macOS / Linux:
```bash
lsof -ti :8501 | xargs kill -9
```

Windows:
```cmd
netstat -ano | findstr :8501
taskkill /PID <pid> /F
```

---

### Demo 4 — FL Training in Terminal (Simulation Mode)

Runs the complete federated training pipeline end-to-end with live logs.

macOS / Linux:
```bash
source venv/bin/activate
python train_federated.py
```

Windows:
```cmd
venv\Scripts\activate
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
  federated_model.pth  · scaler.pkl  · tfidf_vectorizer.pkl
  feature_columns.pkl  · shap_background.npy  · training_metadata.json

Training complete. MAE: 1.8968 | R²: 0.5995 | AUROC: 0.9461
```

> Full training takes ~40–60 minutes. For a quick demo, press Ctrl+C after a few rounds to show the live logs, then explain the full run produces the model that powers the dashboard.

---

### Demo 5 — Real Flower Federated Learning (4 Terminals)

Demonstrates true federated learning: a Flower gRPC server coordinates 3 hospital clients that each train locally and share only weight updates — raw patient data never leaves the client.

Open 4 separate terminal windows. Activate venv in each one first.

**All platforms — activate in each terminal:**

macOS / Linux: `source venv/bin/activate`  
Windows: `venv\Scripts\activate`

**Terminal 1 — FL server:**
```bash
python server.py
```
Expected output:
```
INFO flwr - Starting Flower server, config: ServerConfig(num_rounds=10)
INFO flwr - Flower ECE: gRPC server running (insecure), listening on 0.0.0.0:8080
INFO flwr - Waiting for 3 clients to connect...
```

**Terminal 2 — Hospital 0:**
```bash
python client.py --client_id 0
```

**Terminal 3 — Hospital 1:**
```bash
python client.py --client_id 1
```

**Terminal 4 — Hospital 2:**
```bash
python client.py --client_id 2
```

**What to show:**
- Server terminal: shows "Waiting for clients…" then starts Round 1 once all 3 connect
- Each client terminal: shows local training loss per round
- Server terminal: aggregates weights each round and prints global metrics
- Key point to explain: each hospital trains on its own local data — only gradient updates cross the network, never patient records

**If port 8080 is busy:**

macOS / Linux:
```bash
lsof -ti :8080 | xargs kill -9
```

Windows:
```cmd
netstat -ano | findstr :8080
taskkill /PID <pid> /F
```

---

### Demo 6 — Computer Vision Monitor (Webcam Stress Detection)

Standalone OpenCV + MediaPipe application — tracks 468 facial landmarks in real-time and computes a stress score used as a feature in the ICU prediction model.

> Requires a webcam. macOS: grant camera access to Terminal in System Preferences → Privacy & Security → Camera.

macOS / Linux:
```bash
source venv/bin/activate
python cv_monitor.py
```

Windows:
```cmd
venv\Scripts\activate
python cv_monitor.py
```

**What to show:**
- Webcam feed opens in a window with MediaPipe face mesh overlay (468 landmarks)
- Blink rate counter — blinks per minute
- Eye Aspect Ratio (EAR) — dips sharply on each blink
- Gaze deviation score — measures how far the gaze drifts from center
- Stress score composite (0–1) — combines blink rate, EAR, and gaze deviation
- Explain: this `stress_score` feeds into the 108-feature vector for the DNN alongside vitals

Press **Q** to quit.

---

### Demo 7 — Legacy Streamlit ICU Monitor

The original Streamlit prototype that preceded the full-stack React/FastAPI version — shows the project's evolution.

macOS / Linux:
```bash
source venv/bin/activate
streamlit run app.py --server.port 8502
```

Windows:
```cmd
venv\Scripts\activate
streamlit run app.py --server.port 8502
```

**Open in browser:** http://localhost:8502

**What to show:**
- Patient selector sidebar
- Real-time vitals display with color-coded risk level
- SOFA score prediction using the same DNN model
- Prediction history chart
- Compare with Demo 1 to illustrate how the project evolved from a single-file Streamlit app to a full React/FastAPI system

---

### Demo 8 — Trial ICU Inference App (Research Prototype)

Early ensemble model prototype from the `trial/` folder — built before federated learning was introduced.

> This app lives one level up from the main project folder. Navigate there from the parent directory.

macOS / Linux:
```bash
# From the parent directory (one level above the project folder)
cd ../trial
streamlit run ICU_Inference.py --server.port 8503
```

Windows:
```cmd
cd ..\trial
streamlit run ICU_Inference.py --server.port 8503
```

**Open in browser:** http://localhost:8503

**What to show:**
- Manual vital sign input sliders (HR, RR, SpO₂, Temp, SBP, MAP, GCS)
- Ensemble model inference — Random Forest + XGBoost + LightGBM weighted average
- SHAP waterfall plot — visual breakdown of each feature's contribution to the SOFA prediction
- Risk classification output (Low / Moderate / High)
- Compare with Demo 1: this uses hand-crafted features and ensemble models; the final system uses a Federated DNN with 108 features and TF-IDF

---

### Demo 9 — Federated Learning Prototype App

Early Streamlit FL prototype from the `federated learning/` folder — the concept that became the full FL simulation dashboard.

> Also lives one level up from the main project folder.

macOS / Linux:
```bash
# From the parent directory
cd "../federated learning"
streamlit run app.py --server.port 8504
```

Windows:
```cmd
cd "..\federated learning"
streamlit run app.py --server.port 8504
```

**Open in browser:** http://localhost:8504

**What to show:**
- Simulated FL training across 3 hospital nodes
- Per-round loss and accuracy charts
- Compare with Demo 3: this is the early prototype; the full FL dashboard (Demo 3) adds real algorithms (FedYogi, FedProx), weight statistics heatmaps, and live visualisation

---

### Demo 10 — SQLite Database Inspection

Live inspection of the production database — shows MIMIC-III-derived data, prediction history, and FL training splits.

macOS / Linux:
```bash
sqlite3 models/icu_monitor.db
```

Windows (if sqlite3.exe is in PATH):
```cmd
sqlite3 models\icu_monitor.db
```

Run these queries inside the SQLite shell:

```sql
.headers on
.mode column

-- ── Step 1: List every table in the database ─────────────────────────────
.tables

-- ── Step 2: Row counts across all tables ─────────────────────────────────
SELECT 'patient_vitals'    AS table_name, COUNT(*) AS total_rows FROM patient_vitals
UNION ALL SELECT 'prediction_history', COUNT(*) FROM prediction_history
UNION ALL SELECT 'sample_patients',    COUNT(*) FROM sample_patients
UNION ALL SELECT 'fl_training',        COUNT(*) FROM fl_training;

-- ── Step 3: Schema of every table ────────────────────────────────────────
.schema patient_vitals
.schema prediction_history
.schema sample_patients
.schema fl_training

-- ── Step 4: First 10 rows — patient_vitals ───────────────────────────────
-- 60 rows total (3 patients × 20 readings). Real-time vital signs from MIMIC-III.
SELECT * FROM patient_vitals LIMIT 10;

-- ── Step 5: First 10 rows — prediction_history ───────────────────────────
-- Written by the backend on every /api/predict call during live monitoring.
SELECT * FROM prediction_history LIMIT 10;

-- ── Step 6: First 10 rows — sample_patients ──────────────────────────────
-- 90 rows (3 patients × 30 readings). Powers the 90-second monitoring cycle.
SELECT * FROM sample_patients LIMIT 10;

-- ── Step 7: First 10 rows — fl_training ──────────────────────────────────
-- 48,150 rows (MIMIC-III derived). The training data for federated learning.
SELECT * FROM fl_training LIMIT 10;

-- ── Step 8: Per-patient row counts ───────────────────────────────────────
SELECT patient, COUNT(*) AS readings    FROM patient_vitals    GROUP BY patient;
SELECT patient, COUNT(*) AS readings    FROM sample_patients   GROUP BY patient;
SELECT patient, COUNT(*) AS predictions FROM prediction_history GROUP BY patient;

-- ── Step 9: FL training split per hospital client ────────────────────────
SELECT client_id, COUNT(*) AS rows FROM fl_training GROUP BY client_id;

-- ── Step 10: Latest SOFA prediction per patient ──────────────────────────
SELECT patient, Timestamp, SOFA, Risk, HR, SpO2, MAP
FROM prediction_history
WHERE id IN (
    SELECT MAX(id) FROM prediction_history GROUP BY patient
);

-- ── Step 11: Database file size ──────────────────────────────────────────
SELECT ROUND(page_count * page_size / 1024.0 / 1024.0, 2) AS db_size_mb
FROM pragma_page_count(), pragma_page_size();

.quit
```

**What to show:**
- `.tables` — 4 tables: `patient_vitals`, `prediction_history`, `sample_patients`, `fl_training`
- Row counts — 48,150 FL training rows + 90 simulation rows + 60 vitals rows
- `sample_patients` schema — 108-feature rows including `stress_score`, `GCS_eye_opening`, `clinical_note`
- `fl_training` schema — `client_id` column + all features + `sofa_score` target
- FL split — exactly 16,050 rows per hospital client (0, 1, 2)
- `prediction_history` — SOFA, Risk, and all vitals written by the live backend

---

### Demo 11 — Full Test Suite (701 Tests)

Shows the complete automated test coverage across all layers of the system.

> Activate venv first. Run from the project root folder (same folder as `requirements.txt`).

macOS / Linux:
```bash
source venv/bin/activate
```

Windows:
```cmd
venv\Scripts\activate
```

**FL unit tests — Phases 1–6 (fast, ~30 seconds):**
```bash
python -m pytest tests/fl/unit/ -v
```

**FL integration tests — Phases 7–9:**
```bash
python -m pytest tests/fl/integration/ -v
```

**FL system tests — Phase 10 (end-to-end `fl_thread` simulation):**
```bash
python -m pytest tests/fl/system/ -v -m slow
```

**All FL tests at once (287 tests):**
```bash
python -m pytest tests/fl/ -v
```

**ClinIQ backend API tests (207 tests):**
```bash
python -m pytest tests/cliniq/backend/ -v
```

**All Python tests at once (494 tests):**
```bash
python -m pytest tests/ -v
```

**React frontend tests — Vitest (207 tests):**
```bash
cd frontend
npx vitest run
cd ..
```

**Expected results:**
```
tests/fl/unit/           ~256 passed
tests/fl/integration/     ~14 passed
tests/fl/system/          ~17 passed  (slow mark)
tests/cliniq/backend/    207 passed
──────────────────────────────────────
Python total             494 passed

frontend Vitest          207 passed
──────────────────────────────────────
Grand total              701 passed
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
python train_federated.py
```

Takes ~40–60 minutes (Phase 0: data loading, Phase 1–6: FL training, Phase 7: evaluation + artifact saving).

**Force Phase 0 re-run:**

macOS / Linux:
```bash
sqlite3 models/icu_monitor.db "DELETE FROM fl_training;"
rm models/scaler.pkl models/tfidf_vectorizer.pkl models/feature_columns.pkl
python train_federated.py
```

Windows:
```cmd
sqlite3 models\icu_monitor.db "DELETE FROM fl_training;"
del models\scaler.pkl models\tfidf_vectorizer.pkl models\feature_columns.pkl
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

> Activate your venv first (`source venv/bin/activate` on macOS/Linux, `venv\Scripts\activate` on Windows), then run all commands from the project root.

### FL Tests — 287 tests across 10 phases

```bash
# All FL tests
python -m pytest tests/fl/ -v

# By phase group
python -m pytest tests/fl/unit/        -v          # Phases 1–6 (fast unit tests)
python -m pytest tests/fl/integration/ -v          # Phases 7–9 (integration)
python -m pytest tests/fl/system/      -v -m slow  # Phase 10 (end-to-end system)
```

### ClinIQ Backend Tests — 207 tests

```bash
python -m pytest tests/cliniq/backend/ -v
```

### React Frontend Tests — 207 Vitest tests

```bash
cd frontend
npx vitest run
cd ..
```

### Run everything

```bash
# All Python tests (494 total)
python -m pytest tests/ -v

# Frontend tests (207 total)
cd frontend && npx vitest run && cd ..
```

**Grand total: 701 passing tests.**

---

## Troubleshooting

**`ModuleNotFoundError` on backend start**
```bash
pip install -r requirements.txt
```

**Port already in use (8000 / 8080 / 8501 / 8502 / 8503 / 8504)**

macOS / Linux:
```bash
lsof -ti :8000 | xargs kill -9    # replace 8000 with the port that is busy
```

Windows:
```cmd
netstat -ano | findstr :8000       # find the PID in the last column
taskkill /PID <pid> /F             # replace <pid> with that number
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

### Automated setup (recommended — works on macOS, Linux, Windows)

```bash
# Clone / download the project, then cd into it (any folder name works)
cd <project-folder>

# One command sets up everything: venv, pip, npm, DB, .env
python setup.py

# Skip the virtual environment if you prefer system Python
python setup.py --no-venv
```

`setup.py` handles all 7 steps automatically:
checks Python ≥ 3.12, checks Node ≥ 18, creates `venv/`, installs pip packages,
installs npm packages, initialises the SQLite database, and creates `.env` from `.env.example`.

After setup, add your Groq API key to `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```
Get a free key at https://console.groq.com

### Manual setup (step by step)

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend && npm install && cd ..

# 4. Set up the database (one-time migration)
python migrate_to_db.py

# 5. Set your Groq API key
cp .env.example .env            # macOS / Linux
copy .env.example .env          # Windows
# Then open .env and set: GROQ_API_KEY=gsk_your_key_here

# 6. Start the backend (Terminal 1)
uvicorn backend.main:app --port 8000 --reload

# 7. Start the frontend (Terminal 2)
cd frontend && npm run dev

# Dashboard → http://localhost:5174
# API docs  → http://localhost:8000/docs
```

---

*Final Year Capstone — Computer Science Engineering*  
*Dataset: MIMIC-III · ML: PyTorch Federated DNN · LLM: Groq openai/gpt-oss-120b · Frontend: React 19 + Vite*

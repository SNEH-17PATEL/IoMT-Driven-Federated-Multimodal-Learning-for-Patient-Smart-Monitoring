# ClinIQ — ICU Intelligence Platform
## Redesign Plan: FastAPI + React
### STATUS: IMPLEMENTED ✅ (October 2026)
> This document captures what was planned and what was actually built.
> All phases through Phase 12 are complete. The ClinIQ React+FastAPI dashboard is live.

---

## 1. Platform Name

**ClinIQ** *(Clinical Intelligence)* — AI-Powered ICU Decision Support

> Tagline: *"Real-time clinical intelligence for every care setting."*

The name captures: clinical focus, AI intelligence, brevity, and works equally well for hospital and home care contexts.

---

## 2. What Is Wrong With the Current app.py Dashboard

After a full read-through of all ~3000 lines, here are the specific problems:

| Problem | Where |
|---|---|
| 8 vital sign cards shown as a 2-row grid **immediately** on load — overwhelming | Lines 666–692 |
| Tab 1 shows: SOFA gauge + severity scale + 4 model accuracy tiles + AUROC table + SOFA validation panel (4 rows) + High Risk Advisory + trend chart + trend summary + prediction history table **all at once** | Tab 1 |
| 9-method validation breakdown is a full scrollable grid — meaningless to doctors/nurses/family | Tab 3, Lines 2678–2731 |
| AUROC, PR-AUC, per-class confusion matrix shown to clinical users who have no use for it | Tab 1, Lines 2031–2095 |
| Reading counter / cycle counter / 30-second countdown bar is shown prominently — adds noise | Lines 619–634 |
| Tab 3 shows the reliability banner + violation list + 9-method grid + LLM report + Judge details — 5 separate layers before the actual report | Tab 3 |
| Sidebar with "Model Information" expander shows MAE, R², training samples, FL rounds — not useful to clinical users | Lines 423–436 |
| "About SHAP" expander explains model internals — unnecessary for clinical audience | Tab 2 |
| The SOFA validation panel explains conformal prediction methodology — researcher content, not clinical | Tab 1, Lines 2097–2196 |
| Trend summary shown as a table of badges AND a full Plotly subplot chart — redundant | Tab 1 |
| G-Eval + RAGAS Judge Details expander | Tab 3, Lines 2910–2945 |

---

## 3. What We Keep (Core Clinical Value)

- ✅ Patient selector — 3 patients, clean cards
- ✅ SOFA score and risk level (the single most important output)
- ✅ Vital signs (HR, RR, SpO₂, Temp, SBP/DBP, MAP) — essential for all users
- ✅ Trend charts for vitals over last 20 readings
- ✅ SHAP top-5 risk factors — simplified to plain cards, no technical jargon
- ✅ LLM clinical report — the 4 sections with Hospital/Home Care toggle
- ✅ Overall reliability score (single number + label only — not the 9-method breakdown)
- ✅ Alert banners (HIGH RISK / MODERATE RISK)
- ✅ Auto-refresh every 90 seconds (upgraded from 30s — LLM generation time)
- ✅ DB read/write (all via existing `db.py`)
- ✅ Clinical disclaimer

---

## 4. What We Remove (Clutter)

| Removed | Why |
|---|---|
| 9-method validation breakdown grid | Meaningless to doctors/nurses/family; keep only overall score |
| AUROC, PR-AUC, confusion matrix, per-class metrics | Researcher-only content; clinical users have no use |
| SOFA validation panel (conformal interval, physiological floor, trajectory coherence rows) | Methodology explanation belongs in research report, not clinical dashboard |
| High Risk Advisory expander (uncertainty explanation) | Replaced by a simple "uncertainty ± X pts" inline with SOFA |
| G-Eval + RAGAS Judge Details expander | Internal QA detail; keep only the reliability number |
| Violation list (9 validation violations) | Replaced by a simple "Caution" banner if reliability < 0.60 |
| Sidebar | Eliminated — all navigation moves to top bar |
| Reading counter / cycle counter | Removed — just show "LIVE" + last updated time |
| "About SHAP" expander | Removed — the chart speaks for itself |
| Trend text summary (badges duplicating the chart) | Chart is enough |
| Prediction history table (50-row, 12-column) | Replaced by a compact "Last 5 readings" mini-summary |
| Model accuracy tiles (MAE / R² / 48K patients / 100 rounds) | Move to an "About Model" info page; not needed in monitoring view |
| Stress Score in vitals (low clinical relevance for family members) | Folded into clinical notes display |

---

## 5. New Information Architecture (ClinIQ UX)

### Page A — Patient Selection
- Clean full-screen landing with 3 patient cards
- Each card: name, condition, risk level, brief description, "Begin Monitoring" button
- No clutter, no metrics, no sidebar

### Page B — Monitoring Dashboard (single-page, no tabs)

The new layout has **3 clearly separated sections** on one scrollable page:

```
┌────────────────────────────────────────────────────────────┐
│  HEADER BAR                                                │
│  ClinIQ logo | Patient name & risk badge | LIVE | Stop     │
│  Last updated: 14:32:05 | Next refresh: 28s               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  [ALERT BANNER — only if SOFA ≥ 8]                        │
│                                                            │
│  SECTION 1 — PATIENT OVERVIEW                              │
│  ┌──────────────┐  ┌────────────────────────────────────┐  │
│  │  SOFA Score  │  │  6 Vital Sign Cards (2 rows × 3)   │  │
│  │  Big number  │  │  HR | RR | SpO₂                    │  │
│  │  Risk level  │  │  Temp | SBP | MAP                   │  │
│  │  ± interval  │  │  (color coded normal/abnormal)     │  │
│  └──────────────┘  └────────────────────────────────────┘  │
│                                                            │
│  [Clinical Notes — collapsible, collapsed by default]      │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  SECTION 2 — VITAL TRENDS                                  │
│  [6 trend charts in 2×3 grid — Recharts, clean lines]      │
│  [Last 5 readings summary — compact row of badges]         │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  SECTION 3 — AI CLINICAL REPORT                            │
│  [Hospital / Home Care toggle — top right of section]      │
│  [Reliability badge — single colored pill, not a grid]     │
│  [4 Section Cards: CONDITION | CAUSE | FORECAST | ACTIONS] │
│  [Top 5 SHAP Factors — simple horizontal bars]             │
│  [Clinical Disclaimer — footer]                            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 6. Technology Stack

### Backend — FastAPI (Python)
- Keeps **all ML logic unchanged** (model, SHAP, TF-IDF, SOFA pipeline)
- Keeps **all DB logic unchanged** (`db.py`)
- Keeps **all LLM logic unchanged** (Groq, system messages, validation)
- Adds HTTP layer on top

### Frontend — React
- **Vite** build tool (fast, modern)
- **TailwindCSS** for styling (utility-first, dark theme easy)
- **shadcn/ui** for pre-built components (cards, badges, buttons, tabs)
- **Recharts** for vital sign trend charts
- **React Router** for page navigation
- **Custom useMonitor hook** for 90-second polling (chosen over React Query for countdown control)
- **Framer Motion v14.5** for page/component animations
- **anime.js v4.5** (named exports API) for SHAP bar spring animations
- **AOS v2** for scroll reveal

---

## 7. Folder Structure

```
icu_monitor/
│
│  ── EXISTING (DO NOT TOUCH) ──────────────────────────────
├── app.py                    # Streamlit app — keep until migration done
├── fl_dashboard.py           # Streamlit FL dashboard — keep
├── db.py                     # SQLite DB layer — unchanged
├── model_utils.py            # ML helpers — unchanged
├── migrate_to_db.py          # DB seeder — unchanged
├── train_federated.py        # FL training — unchanged
├── models/                   # Trained model artifacts — unchanged
├── data/                     # Raw data — unchanged
│
│  ── NEW: BACKEND ─────────────────────────────────────────
├── backend/
│   ├── main.py               # FastAPI app + CORS + route registration
│   ├── requirements.txt      # fastapi, uvicorn, python-dotenv
│   │
│   ├── api/                  # Route handlers (thin — just call services)
│   │   ├── __init__.py
│   │   ├── patients.py       # GET /api/patients
│   │   ├── predict.py        # POST /api/predict  (main pipeline)
│   │   ├── report.py         # POST /api/report   (LLM)
│   │   └── history.py        # GET /api/history/{patient_id}
│   │
│   └── services/             # Business logic (ported from app.py)
│       ├── __init__.py
│       ├── ml_service.py     # Model load, SHAP, SOFA pipeline
│       └── llm_service.py    # Groq calls, validation, reliability
│
│  ── NEW: FRONTEND ────────────────────────────────────────
└── frontend/
    ├── package.json
    ├── vite.config.js        # Proxy: /api → http://localhost:8000
    ├── tailwind.config.js
    ├── index.html
    │
    └── src/
        ├── main.jsx
        ├── App.jsx            # Router setup
        │
        ├── api/
        │   └── client.js     # Axios instance + all API functions
        │
        ├── components/
        │   ├── layout/
        │   │   ├── HeaderBar.jsx       # Patient name, LIVE, Stop, timer
        │   │   └── AlertBanner.jsx     # HIGH/MODERATE risk banner
        │   │
        │   ├── overview/
        │   │   ├── SOFAGauge.jsx       # Big number + risk level + ± interval
        │   │   ├── VitalCard.jsx       # Single vital tile (color coded)
        │   │   └── VitalsGrid.jsx      # 2×3 grid of VitalCards
        │   │
        │   ├── trends/
        │   │   └── TrendChart.jsx      # Recharts line chart for one vital
        │   │
        │   └── report/
        │       ├── ReportToggle.jsx    # Hospital / Home Care pill toggle
        │       ├── ReliabilityBadge.jsx # Single colored score pill
        │       ├── ReportSection.jsx   # One of the 4 LLM sections
        │       └── SHAPPanel.jsx       # Top 5 SHAP factors
        │
        └── pages/
            ├── SelectPatient.jsx    # Patient selection landing
            └── MonitorDashboard.jsx # Full monitoring page
```

---

## 8. FastAPI Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/api/patients` | Returns patient config (name, description, risk, color) |
| `POST` | `/api/predict` | Runs full pipeline: reads DB row → vitals → SOFA → SHAP → saves history → returns JSON |
| `POST` | `/api/report` | Calls Groq LLM → runs reliability check → returns report + score |
| `GET` | `/api/history/{patient_id}` | Returns last 5 predictions from DB |
| `GET` | `/api/vitals/{patient_id}` | Returns last 20 vitals rows from DB |

### `/api/predict` Response Shape
```json
{
  "patient_id": 2,
  "sofa_score": 7.3,
  "risk_level": "Moderate Risk",
  "risk_color": "#e6a817",
  "conformal_interval": [4.4, 10.2],
  "vitals": { "HR": 108, "RR": 22, "SpO2": 92.1, "Temp": 38.2, "SBP": 98, "DBP": 62, "MAP": 74 },
  "vitals_status": { "HR": "high", "RR": "high", "SpO2": "low", "Temp": "high", "SBP": "low", "DBP": "normal", "MAP": "normal" },
  "shap_top5": [
    { "feature": "SpO₂", "value": 92.1, "unit": "%", "impact": 1.82, "direction": "increases_risk" },
    ...
  ],
  "key_risks": ["Low oxygen levels", "Respiratory distress"],
  "clinical_note": "Patient is...",
  "trend_text": "...",
  "reading_index": 14
}
```

### `/api/report` Request + Response
```json
// Request
{ "patient_id": 2, "care_mode": "hospital", "predict_result": { ...above... } }

// Response
{
  "sections": {
    "CURRENT CONDITION": "...",
    "PROBABLE CAUSE": "...",
    "RISK FORECAST": "...",
    "IMMEDIATE ACTIONS": "..."
  },
  "reliability_score": 0.79,
  "reliability_label": "Moderate Reliability",
  "caution": false
}
```

---

## 9. React Data Flow (As Built)

```
MonitorDashboard
  ├── useMonitor(patientId)  →  POST /api/predict every 90s
  │     ├── fetchingRef guard (no overlapping fetches)
  │     ├── countdown freezes at 1 during in-flight fetch
  │     ├── setCountdown(90) only in finally block
  │     └── on success → updates predictData, lastUpdated, countdown
  │
  └── ReportContainer  →  POST /api/report
        ├── fires only when reportKey() changes or careMode switches
        ├── reportKey() = "patient_id|sofa|HR|SpO2|SBP|GCS"
        ├── lastKeyRef + lastCareModeRef guard redundant LLM calls
        └── on success → updates reportData, sections, reliability
```

Data flow handles:
- Loading states: previous data shown while new fetch in progress
- 90-second poll: countdown visible in header status strip
- Error states: "Refresh failed — showing last reading. Retrying in 90s."
- LLM gating: report only regenerates on genuine clinical value change

---

## 10. Design Principles for ClinIQ

### Visual Design
- **Dark navy base** (#0a1628) — matches current ICU aesthetic, familiar to clinical staff
- **High contrast text** — white/light on dark, never grey-on-grey
- **Color-coded risk** — green (Low), amber (Moderate), red (High) — consistent everywhere
- **Large, readable numbers** — SOFA score, vital values; clinicians read at a glance under pressure
- **Generous whitespace** — cards don't crowd each other; one focus per section

### UX Principles
- **Progressive disclosure** — most critical info (SOFA + risk + alerts) is first, details require scroll
- **No hidden tabs** — single scrollable page; nothing requires clicking to find critical information
- **Collapsed by default** — clinical notes starts collapsed; trends are visible but not overwhelming
- **Auto-refresh is silent** — loading spinner only at top; page doesn't flicker or jump on refresh
- **Clear section labels** — "Current Status", "Vital Trends", "AI Clinical Report" — plain language

### Audience-Specific
- **Doctors/nurses** — SOFA, vitals, SHAP, AI report (Hospital mode)
- **Family members** — simplified vitals (just normal/abnormal labels), Home Care report (plain English)
- **No researcher content** — AUROC, confusion matrix, conformal prediction math → removed

---

## 11. Migration Steps (Implementation Order)

> **Do not delete app.py until Step 6 is complete and verified.**

### Step 1 — Backend skeleton
- Create `backend/` folder
- `main.py` with FastAPI app, CORS enabled for `localhost:5173`
- `GET /api/patients` hardcoded
- Run `uvicorn backend.main:app --reload`

### Step 2 — ML service
- Port `load_artifacts()`, `compute_conformal_q_hat()`, SOFA pipeline to `backend/services/ml_service.py`
- Wire `POST /api/predict` — reads DB, runs pipeline, saves prediction, returns JSON
- Test with curl/Postman

### Step 3 — LLM service
- Port `get_multiple_llm_responses()`, reliability computation to `backend/services/llm_service.py`
- Wire `POST /api/report`
- Test with curl/Postman

### Step 4 — React scaffold
- `npm create vite@latest frontend -- --template react`
- Install: `tailwindcss`, `shadcn/ui`, `axios`, `@tanstack/react-query`, `recharts`, `react-router-dom`
- Vite proxy: `/api` → `http://localhost:8000`
- Basic `App.jsx` with routes

### Step 5 — Patient Selection page
- `SelectPatient.jsx` — 3 clean cards, fetch from `/api/patients`
- On click → navigate to `/monitor/:patientId`

### Step 6 — Monitoring Dashboard
- Build from top to bottom:
  1. `HeaderBar` (patient info, LIVE, stop)
  2. `AlertBanner` (conditional)
  3. `SOFAGauge` + `VitalsGrid` side by side
  4. `TrendChart` grid (Recharts)
  5. `ReportToggle` + `ReliabilityBadge` + `ReportSection` cards + `SHAPPanel`
- Wire React Query polling every 30s to `/api/predict` then `/api/report`

### Step 7 — Test end-to-end
- All 3 patients
- Hospital mode + Home Care mode
- Let it run through multiple auto-refresh cycles
- Verify DB reads and writes are happening correctly

### Step 8 — Polish
- Loading skeletons
- Error handling (API down, LLM timeout)
- Responsive layout (works on tablet)
- Final styling pass

### Step 9 — Retire app.py
- Once Step 7 is complete and verified → `app.py` can be archived (move to `archive/` or delete)

---

## 12. DB Layer — No Changes

All reads and writes go through the existing `db.py` functions, called from the FastAPI backend services:

| DB function | Called from |
|---|---|
| `get_sample_patient_row()` | `ml_service.py` → `/api/predict` |
| `get_sample_patient_count()` | `ml_service.py` → `/api/predict` |
| `append_patient_vitals()` | `ml_service.py` → `/api/predict` |
| `get_patient_vitals()` | `ml_service.py` → `/api/predict` + `/api/vitals/{id}` |
| `append_prediction()` | `ml_service.py` → `/api/predict` |
| `get_prediction_history()` | `history.py` → `/api/history/{id}` |
| `fl_training_exists()` | `ml_service.py` (conformal q_hat computation) |
| `get_fl_client_data()` | `ml_service.py` (conformal q_hat computation) |

The database file (`models/icu_monitor.db`) is **never touched** — same file, same schema.

---

## 13. Environment & Running

```bash
# Terminal 1 — Backend
cd icu_monitor
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
cd icu_monitor/frontend
npm run dev         # Vite dev server on http://localhost:5174

# .env stays in icu_monitor/ (backend loads it with python-dotenv)
```

ClinIQ is live at `http://localhost:5174`.
`app.py` (Streamlit original) has been archived to `archive/` — ClinIQ is now the primary dashboard.

---

## 14. What ClinIQ Will Look Like vs. Current

| Aspect | Current (Streamlit) | New (ClinIQ / React) |
|---|---|---|
| Navigation | 3 tabs + sidebar | Single scrollable page, top header |
| First thing user sees | 8 vital cards + live banner + countdown | SOFA score + 6 vitals side by side, clean |
| Information density | Everything at once, overwhelming | Progressive: critical → supporting → deep-dive |
| SHAP | 7-row impact bar list + 2-col interpretation + chip grid | 5 compact bars with plain-language label |
| LLM report | Buried below reliability grid (scroll past 4 heavy sections) | Prominent section with toggle at top |
| Reliability | 9-method grid with weights, scores, icons | Single colored badge (High / Moderate / Low) |
| Validation violations | Red boxes listed below report | Caution pill on reliability badge if < 0.60 |
| Model metrics | Tiles in Tab 1 + sidebar expander | Removed from monitoring view |
| Auto-refresh | Streamlit full page rerun (visible flash) | useMonitor 90s hook — silent, no flicker |
| Family member usability | Not designed for them | Home Care mode + plain English vitals labels |

---

---

# IMPLEMENTATION PHASES — ClinIQ Redesign

> Each phase has: **Goal**, **Files created/modified**, **Exact tasks**, **How to verify it works**, **Done when** checklist.
> Phases are sequential. Do not start the next phase until the current one passes its verification step.

---

## PHASE 0 — Prerequisites & Environment Verification

**Goal:** Confirm the existing system is fully working before touching anything, and install all new dependencies so they are ready.

### Tasks

**0.1 — Verify the existing DB and ML stack works**
- Run `python migrate_to_db.py` and confirm no errors
- Run `python -c "from db import init_db; init_db(); print('DB OK')"` — should print `DB OK`
- Confirm `models/icu_monitor.db` exists and is not empty
- Confirm `models/federated_model.pth`, `models/scaler.pkl`, `models/tfidf_vectorizer.pkl`, `models/feature_columns.pkl` all exist
- Confirm `.env` file exists with `GROQ_API_KEY=...`

**0.2 — Verify GROQ API key is live**
```bash
python -c "
from dotenv import load_dotenv; import os; from groq import Groq
load_dotenv()
c = Groq(api_key=os.getenv('GROQ_API_KEY'))
r = c.chat.completions.create(model='llama-3.1-8b-instant',
    messages=[{'role':'user','content':'say OK'}], max_tokens=5)
print('GROQ OK:', r.choices[0].message.content)
"
```

**0.3 — Install backend dependencies**
```bash
pip install fastapi uvicorn[standard] python-multipart
# All other deps (torch, joblib, shap, groq, etc.) already installed
```

**0.4 — Install frontend dependencies**
```bash
cd icu_monitor
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install -D tailwindcss@3 postcss autoprefixer
npm install axios @tanstack/react-query react-router-dom recharts
npm install -D @types/react
npx tailwindcss init -p
```

**0.5 — Install shadcn/ui**
```bash
cd icu_monitor/frontend
npx shadcn@latest init
# Choose: Dark theme, CSS variables, default style
```

### Verify Phase 0 Complete
- [ ] `models/icu_monitor.db` exists and has rows in `sample_patients`, `fl_training`
- [ ] All 5 model files exist in `models/`
- [ ] GROQ API test prints `GROQ OK`
- [ ] `icu_monitor/frontend/` folder exists with `package.json` and `node_modules/`
- [ ] `pip show fastapi uvicorn` shows installed versions
- [ ] Old `app.py` still runs: `streamlit run app.py` works (do not break it)

---

## PHASE 1 — Backend Foundation

**Goal:** A running FastAPI server that returns patient config and passes a health check. No ML yet — just the skeleton.

### Files Created
```
icu_monitor/backend/__init__.py
icu_monitor/backend/main.py
icu_monitor/backend/api/__init__.py
icu_monitor/backend/api/patients.py
icu_monitor/backend/services/__init__.py
```

### Tasks

**1.1 — Create `backend/main.py`**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import patients, predict, report, history

app = FastAPI(title="ClinIQ API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router)
app.include_router(predict.router)
app.include_router(report.router)
app.include_router(history.router)

@app.get("/health")
def health(): return {"status": "ok", "service": "ClinIQ"}
```

**1.2 — Create `backend/api/patients.py`**
- Hardcoded PATIENTS dict (same 3 patients from app.py)
- `GET /api/patients` → returns list of patient configs
- `GET /api/patients/{patient_id}` → returns single patient

**1.3 — Create stub routers** for predict, report, history
- Each returns `{"status": "not implemented"}` for now
- Required so main.py can import them without errors

**1.4 — Create `__init__.py` files** in `backend/`, `backend/api/`, `backend/services/`

**1.5 — Add `.env` loading in `main.py`**
```python
from dotenv import load_dotenv
load_dotenv()   # reads icu_monitor/.env
```

### Verify Phase 1 Complete
```bash
# From icu_monitor/ directory:
uvicorn backend.main:app --reload --port 8000
```
- [ ] `curl http://localhost:8000/health` → `{"status":"ok","service":"ClinIQ"}`
- [ ] `curl http://localhost:8000/api/patients` → JSON array with 3 patient objects
- [ ] `curl http://localhost:8000/api/patients/1` → single patient JSON
- [ ] `curl http://localhost:8000/api/predict` → `{"status":"not implemented"}`
- [ ] No import errors in uvicorn output
- [ ] `app.py` still runs on port 8501 without any issue

---

## PHASE 2 — ML Service & Prediction Endpoint

**Goal:** `POST /api/predict` runs the complete SOFA pipeline (reads DB, computes features, predicts, runs SHAP, saves to DB, returns JSON). This is the most critical backend phase.

### Files Created/Modified
```
icu_monitor/backend/services/ml_service.py   ← NEW (all ML logic)
icu_monitor/backend/api/predict.py           ← REPLACE stub
```

### Tasks

**2.1 — Create `backend/services/ml_service.py`**

Port the following from `app.py` into a class/functions:

- `load_artifacts()` — model, scaler, tfidf, feature_cols, background. Use `@functools.lru_cache` instead of `@st.cache_resource`
- `load_shap_explainer()` — same logic, cached
- `compute_conformal_q_hat()` — same logic, cached
- `load_training_metadata()` — reads `training_metadata.json`, cached
- `PATIENTS` dict — same 3 patients
- `NORMAL_RANGES`, `VALID_RANGES`, `ALERT_THRESHOLD` constants
- `risk_label(sofa)` function
- `compute_sofa_floor()`, `vital_consistency_flags()`, `check_trajectory()` — validation helpers (can be simplified: only keep conformal interval, drop the verbose methodology text)
- Full feature building pipeline: sliding window → trend features → latest features → TF-IDF → combine → scale → predict → clip

Signature:
```python
def run_prediction(patient_id: int, row_idx: int) -> dict:
    """
    Reads DB, runs full SOFA pipeline, saves vitals + prediction to DB.
    Returns the complete prediction result dict.
    """
```

Returns:
```python
{
    "patient_id": int,
    "row_idx": int,
    "total_rows": int,
    "sofa_score": float,
    "risk_level": str,        # "Low Risk" / "Moderate Risk" / "High Risk"
    "risk_color": str,        # hex color
    "risk_icon": str,         # emoji
    "is_alert": bool,         # sofa >= ALERT_THRESHOLD
    "conformal_lo": float,
    "conformal_hi": float,
    "conformal_q": float,
    "vitals": {               # raw values
        "HR": float, "RR": float, "SpO2": float, "Temp": float,
        "SBP": float, "DBP": float, "MAP": float,
        "GCS_eye": int, "stress": int
    },
    "vitals_status": {        # "normal" / "high" / "low" per vital
        "HR": str, ...
    },
    "shap_top5": [            # top 5 SHAP features
        {"feature": str, "label": str, "value": float, "unit": str,
         "impact": float, "abs_impact": float, "direction": str}
    ],
    "key_risks": [str],
    "clinical_note": str,
    "trend_text": str,
    "vitals_df_json": str,    # JSON of last 20 vitals (for charts)
    "clinical_explanations": [str],
    "alert_threshold": float,
}
```

**2.2 — Update `backend/api/predict.py`**

```python
@router.post("/api/predict")
def predict(body: PredictRequest):
    result = run_prediction(body.patient_id, body.row_idx)
    return result
```

Request body:
```python
class PredictRequest(BaseModel):
    patient_id: int   # 1, 2, or 3
    row_idx: int      # current simulation row index (managed by frontend)
```

**2.3 — Session row index management**

In `app.py`, Streamlit session state holds `row_indices[patient_id]`. In the new system, the **frontend** holds this state (starts at 0, increments by 1 each successful prediction call).

The API does NOT maintain state between calls — it is stateless. The frontend sends the current `row_idx` and the API uses it.

### DB Interactions in This Phase
| DB function | When called |
|---|---|
| `get_sample_patient_count(patient_id)` | At start of `run_prediction()` |
| `get_sample_patient_row(patient_id, cur_idx)` | Read current simulation row |
| `get_patient_vitals(patient_id, limit=1)` | Check if vitals exist (seed if not) |
| `get_patient_vitals(patient_id, limit=20)` | Build sliding window |
| `append_patient_vitals(patient_id, row)` | Write new vitals reading |
| `append_prediction(patient_id, row)` | Write new prediction to history |
| `fl_training_exists()` | Conformal q_hat computation |
| `get_fl_client_data(i)` | Conformal q_hat computation |

### Verify Phase 2 Complete
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id": 2, "row_idx": 0}'
```
- [ ] Response has `sofa_score` (float between 0–24)
- [ ] Response has `vitals` with all 7 keys
- [ ] Response has `vitals_status` with correct normal/high/low for each vital
- [ ] Response has `shap_top5` list (5 items)
- [ ] Response has `conformal_lo`, `conformal_hi`
- [ ] Check SQLite DB: new row appears in `patient_vitals` and `prediction_history`
  ```bash
  sqlite3 models/icu_monitor.db "SELECT * FROM patient_vitals ORDER BY id DESC LIMIT 3;"
  sqlite3 models/icu_monitor.db "SELECT * FROM prediction_history ORDER BY id DESC LIMIT 3;"
  ```
- [ ] Call with `row_idx: 1`, `row_idx: 2` — different vitals come back each time
- [ ] Call for patient 1, 2, 3 — all work, different SOFA scores
- [ ] Response time < 5 seconds (SHAP is the slow part)

---

## PHASE 3 — LLM Service & Report Endpoint

**Goal:** `POST /api/report` generates the LLM report for a given patient + care mode, runs the reliability check, and returns the structured 4-section report with a reliability score.

### Files Created/Modified
```
icu_monitor/backend/services/llm_service.py   ← NEW (all LLM logic)
icu_monitor/backend/api/report.py             ← REPLACE stub
```

### Tasks

**3.1 — Create `backend/services/llm_service.py`**

Port the following from `app.py`:
- `_HOSPITAL_SYSTEM_MSG`, `_HOMECARE_SYSTEM_MSG` constants
- `_strip_thinking()` function
- `_FALLBACK_MSG` constant
- `get_llm_response()` — single call (was `get_multiple_llm_responses` with n=1)
- All prompt building logic (hospital alert, hospital non-alert, homecare alert, homecare non-alert)
- All 7 deterministic validation functions: `factual_grounding_score`, `llm_clinical_rules_score`, `shap_coherence_score`, `response_structure_score`, `severity_calibration_score`, `contraindication_score`, `numeric_hallucination_score`
- `call_judge_llm()` — 2nd LLM call (llama-3.1-8b-instant)
- `compute_reliability_new()` — 9-method combined score
- `_parse_llm_sections()` — splits LLM text into 4 section dict

Signature:
```python
def generate_report(predict_result: dict, care_mode: str, api_key: str) -> dict:
    """
    Builds prompt from predict_result, calls Groq, validates, returns report.
    care_mode: "hospital" or "homecare"
    """
```

Returns:
```python
{
    "sections": {
        "CURRENT CONDITION": str,
        "PROBABLE CAUSE": str,
        "RISK FORECAST": str,
        "IMMEDIATE ACTIONS": str,
    },
    "raw_response": str,
    "reliability_score": float,   # 0.0 – 1.0
    "reliability_label": str,     # "High" / "Moderate" / "Low"
    "reliability_color": str,     # hex color for frontend badge
    "caution": bool,              # True if reliability < 0.60
    "care_mode": str,
}
```

**3.2 — Update `backend/api/report.py`**

```python
class ReportRequest(BaseModel):
    patient_id: int
    care_mode: str         # "hospital" or "homecare"
    predict_result: dict   # full result from /api/predict

@router.post("/api/report")
def report(body: ReportRequest):
    api_key = os.getenv("GROQ_API_KEY", "")
    result = generate_report(body.predict_result, body.care_mode, api_key)
    return result
```

**3.3 — Error handling**
- If Groq API fails → return `{"error": "LLM unavailable", "sections": null, "reliability_score": null}`
- If LLM returns fallback (token cutoff) → include in response so frontend can show a retry message
- If judge LLM fails → reliability still computes with 7 deterministic methods only (geval=0.5, ragas=0.5 fallback)

### Verify Phase 3 Complete
```bash
# First get a predict result:
PREDICT=$(curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id": 2, "row_idx": 0}')

# Then call report:
curl -X POST http://localhost:8000/api/report \
  -H "Content-Type: application/json" \
  -d "{\"patient_id\": 2, \"care_mode\": \"hospital\", \"predict_result\": $PREDICT}"
```
- [ ] Response has all 4 section keys with non-empty string content
- [ ] `reliability_score` is between 0.0 and 1.0
- [ ] `reliability_label` is one of "High", "Moderate", "Low"
- [ ] No sections are `null` or empty
- [ ] Test with `care_mode: "homecare"` — different response tone
- [ ] Test with patient 3 (High Risk) — sections mention urgency
- [ ] Test with patient 1 (Low Risk) — sections are calmer
- [ ] Response time < 30 seconds (LLM call takes ~10–15s)

---

## PHASE 4 — History & Vitals Endpoints

**Goal:** Complete the remaining two API endpoints that the frontend needs.

### Files Created/Modified
```
icu_monitor/backend/api/history.py   ← REPLACE stub
icu_monitor/backend/api/vitals.py    ← NEW (add to main.py router list)
```

### Tasks

**4.1 — `GET /api/history/{patient_id}`**

```python
@router.get("/api/history/{patient_id}")
def get_history(patient_id: int, limit: int = 5):
    df = get_prediction_history(patient_id, limit=limit)
    return df.to_dict(orient="records")
```

Returns last `limit` prediction rows (timestamp, SOFA, risk, HR, RR, SpO2) — used by the "Last 5 readings" mini summary on the dashboard.

**4.2 — `GET /api/vitals/{patient_id}`**

```python
@router.get("/api/vitals/{patient_id}")
def get_vitals(patient_id: int, limit: int = 20):
    df = get_patient_vitals(patient_id, limit=limit)
    return df.to_dict(orient="records")
```

Returns last 20 vitals rows for trend charts.

**4.3 — Add vitals router to `main.py`**

### Verify Phase 4 Complete
```bash
curl http://localhost:8000/api/history/2?limit=5
curl http://localhost:8000/api/vitals/2?limit=20
```
- [ ] `/api/history/2` returns an array of ≤ 5 objects with `SOFA`, `Risk`, `Timestamp` keys
- [ ] `/api/vitals/2` returns an array of ≤ 20 objects with `HR`, `RR`, `SpO2`, `Temp`, `SBP`, `DBP`, `MAP` keys
- [ ] After calling `/api/predict` for patient 2 a few times, the history endpoint shows new rows
- [ ] All 3 patient IDs work for both endpoints

---

## PHASE 5 — React Frontend Foundation

**Goal:** React app runs, connects to the API, and the routing skeleton is in place. No UI components yet — just the plumbing.

### Files Created/Modified
```
frontend/vite.config.js              ← add API proxy
frontend/tailwind.config.js          ← configure paths
frontend/src/index.css               ← Tailwind directives
frontend/src/main.jsx                ← React Query provider + Router
frontend/src/App.jsx                 ← Route definitions
frontend/src/api/client.js           ← Axios instance + all API calls
```

### Tasks

**5.1 — Configure Vite proxy** in `vite.config.js`
```js
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
})
```
This means `axios.get('/api/patients')` in the React code automatically hits `http://localhost:8000/api/patients` in dev. No hardcoded URLs anywhere in the frontend.

**5.2 — Configure Tailwind** in `tailwind.config.js`
```js
content: ["./index.html", "./src/**/*.{js,jsx}"]
```

Add Tailwind directives to `src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

Set dark background on `body` in `index.css`:
```css
body { background-color: #0a1628; color: #d0e0ec; }
```

**5.3 — Create `src/api/client.js`**

```js
import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const getPatients    = ()                 => api.get('/patients')
export const getPatient     = (id)               => api.get(`/patients/${id}`)
export const runPredict     = (patientId, rowIdx)=> api.post('/predict', { patient_id: patientId, row_idx: rowIdx })
export const generateReport = (patientId, careMode, predictResult) =>
  api.post('/report', { patient_id: patientId, care_mode: careMode, predict_result: predictResult })
export const getHistory     = (patientId, limit=5)  => api.get(`/history/${patientId}?limit=${limit}`)
export const getVitals      = (patientId, limit=20) => api.get(`/vitals/${patientId}?limit=${limit}`)
```

**5.4 — Create `src/main.jsx`**
```jsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
const queryClient = new QueryClient()
// Wrap App in QueryClientProvider + BrowserRouter
```

**5.5 — Create `src/App.jsx`** with two routes:
- `/` → `<SelectPatient />`
- `/monitor/:patientId` → `<MonitorDashboard />`

Both are placeholder components returning `<div>Coming soon</div>` for now.

### Verify Phase 5 Complete
```bash
cd frontend && npm run dev
```
- [ ] Vite dev server starts on `http://localhost:5173`
- [ ] Browser shows `http://localhost:5173` without console errors
- [ ] Browser console: `fetch('/api/patients')` returns patient JSON (test in browser DevTools)
- [ ] Navigate to `http://localhost:5173/monitor/2` — shows "Coming soon" without crashing
- [ ] Tailwind works: add `className="bg-red-500 text-white p-4"` to App.jsx temporarily → red box appears

---

## PHASE 6 — Patient Selection Page

**Goal:** A clean, professional patient selection landing page. Users click a patient card to navigate to monitoring.

### Files Created
```
frontend/src/pages/SelectPatient.jsx
frontend/src/components/layout/PageHeader.jsx
```

### Tasks

**6.1 — `SelectPatient.jsx`**
- Fetches `/api/patients` using React Query (`useQuery`)
- Shows loading skeleton while fetching (3 grey placeholder cards)
- Renders 3 patient cards in a responsive 3-column grid

**6.2 — Patient Card design**

Each card contains:
- Risk color accent (left border + background tint)
- Risk badge pill (LOW RISK / MODERATE RISK / HIGH RISK) — top right
- Patient icon (🟢 / 🟡 / 🔴) — large, top left
- Patient name — bold, large
- Condition description — one line
- "Expected SOFA" range — subtle tag
- "Begin Monitoring →" button — full width, prominent

Card visual spec:
- Dark navy card background (`#0d1f3c`) with 1px border matching risk color
- Left border 4px in risk color
- On hover: card lifts slightly (transform: translateY -3px), border brightens
- Badge: small pill, risk color background at 25% opacity, risk color text, risk color border

**6.3 — Page header**

Above the cards:
- ClinIQ logo text + tagline
- "Select a patient to begin real-time monitoring" subtitle

### Verify Phase 6 Complete
- [ ] Page loads at `http://localhost:5173`
- [ ] 3 patient cards are visible with correct risk colors (green/amber/red)
- [ ] Clicking "Begin Monitoring" on any card navigates to `/monitor/1`, `/monitor/2`, or `/monitor/3`
- [ ] Loading state shows placeholder cards before data arrives (test by throttling network in DevTools)
- [ ] No console errors
- [ ] Looks clean — not cluttered, good spacing, readable at a glance

---

## PHASE 7 — Monitoring Dashboard: Header + Patient Overview

**Goal:** The monitoring page shows a live header bar and the top "Current Status" section — SOFA gauge, vitals grid, and alert banner.

### Files Created
```
frontend/src/pages/MonitorDashboard.jsx
frontend/src/components/layout/HeaderBar.jsx
frontend/src/components/layout/AlertBanner.jsx
frontend/src/components/overview/SOFAGauge.jsx
frontend/src/components/overview/VitalCard.jsx
frontend/src/components/overview/VitalsGrid.jsx
frontend/src/hooks/useMonitor.js
```

### Tasks

**7.1 — Create `src/hooks/useMonitor.js`**

This hook owns all data fetching for the monitoring page:
```js
export function useMonitor(patientId) {
  const [rowIdx, setRowIdx] = useState(0)
  const [predictData, setPredictData] = useState(null)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const fetchNext = async () => {
    setIsRefreshing(true)
    const result = await runPredict(patientId, rowIdx)
    setPredictData(result.data)
    setRowIdx(i => i + 1)
    setIsRefreshing(false)
  }

  // Initial fetch on mount; then every 30s
  useEffect(() => {
    fetchNext()
    const interval = setInterval(fetchNext, 30000)
    return () => clearInterval(interval)
  }, [patientId])

  return { predictData, isRefreshing, rowIdx }
}
```

**7.2 — `MonitorDashboard.jsx`**
- Gets `patientId` from URL params (`useParams`)
- Fetches patient config from `/api/patients/:id`
- Uses `useMonitor(patientId)` hook for live data
- Renders: `HeaderBar` → `AlertBanner` → Section 1 (SOFA + Vitals) → [Sections 2 & 3 stubs]

**7.3 — `HeaderBar.jsx`**

Fixed top bar (sticky) with:
- Left: ClinIQ logo | patient name | risk badge
- Center: `● LIVE` pulsing red dot + "Updating every 30s"
- Right: last-updated timestamp | "Stop Monitoring" button (navigates back to `/`)

**7.4 — `AlertBanner.jsx`**

Conditional — only renders if `predictData.is_alert === true`:
- Red background (`rgba(220,53,69,0.15)`), red border
- "⚠ HIGH RISK — Predicted SOFA {score}. Immediate clinical attention required."
- If SOFA 5–8: amber background, "MODERATE RISK — Close monitoring required."

**7.5 — `SOFAGauge.jsx`**

```
┌────────────────────────────────┐
│  PREDICTED SOFA SCORE          │
│                                │
│        7.3                     │  ← huge number, risk color
│        /24                     │
│                                │
│  🟡 Moderate Risk              │
│  90% interval: [4.4 – 10.2]   │  ← small, subtle
└────────────────────────────────┘
```

**7.6 — `VitalCard.jsx`**

Single vital sign tile:
```
┌──────────────────────┐
│  ❤️ Heart Rate       │
│                      │
│       108            │  ← big number
│       bpm            │
│                      │
│  ⚠ Above Normal     │  ← status line, color coded
│  Normal: 60–100      │  ← small grey text
└──────────────────────┘
```

Status colors:
- Normal → green border + green "✓ Normal"
- High → red border + red "⚠ Above Normal"
- Low → orange border + orange "⚠ Below Normal"

**7.7 — `VitalsGrid.jsx`**

2 rows × 3 cols grid of VitalCards:
- Row 1: HR, RR, SpO₂
- Row 2: Temp, SBP, MAP
- (Removed: DBP — less clinically actionable than MAP for this audience)

### Verify Phase 7 Complete
- [ ] Navigate to `/monitor/2` → page loads without crashing
- [ ] Header shows patient 2 name + "Moderate Risk" badge
- [ ] SOFA score appears as a large number with risk color
- [ ] 6 vital cards show correct values from the API
- [ ] Vitals with abnormal values show red/orange styling
- [ ] `AlertBanner` appears for patient 3 (High Risk), absent for patient 1
- [ ] After 30 seconds, the data silently refreshes (values may change slightly)
- [ ] "Stop Monitoring" button navigates back to patient selection
- [ ] Check DevTools Network tab: `POST /api/predict` fires on load + every 30s
- [ ] Check SQLite: `SELECT COUNT(*) FROM patient_vitals WHERE patient=2;` increases by 1 each cycle

---

## PHASE 8 — Vital Trends Section

**Goal:** Show the last 20 readings as trend charts — one chart per vital, in a clean 2×3 grid.

### Files Created
```
frontend/src/components/trends/TrendChart.jsx
frontend/src/components/trends/TrendsSection.jsx
```

### Tasks

**8.1 — `TrendChart.jsx`**

Single Recharts line chart for one vital:
- `ResponsiveContainer` (fills parent width)
- `LineChart` with `Line`, `XAxis`, `YAxis`, `Tooltip`, `ReferenceLine` (normal range band)
- Color: matches vital card color (HR=red, RR=blue, SpO2=green, Temp=orange, SBP=purple, MAP=teal)
- Green shaded reference area between normal lo and hi values
- Dark background (`#0a1628`) with subtle grid lines
- Tooltip: shows value + unit on hover
- No legend (label is in the chart title above)

Props:
```jsx
<TrendChart
  data={vitalsHistory}   // array of {time, HR, RR, SpO2, ...}
  dataKey="HR"           // which column to plot
  label="Heart Rate"
  unit="bpm"
  color="#e74c3c"
  normalLo={60}
  normalHi={100}
/>
```

**8.2 — `TrendsSection.jsx`**

- Fetches `/api/vitals/{patientId}` with React Query (refetches every 30s)
- Renders "Vital Trends — Last 20 Readings" section header
- 2×3 grid of TrendChart components:
  - Row 1: HR, RR, SpO₂
  - Row 2: Temp, SBP, MAP
- On mobile: stacks to 1 column

### Verify Phase 8 Complete
- [ ] 6 charts visible in a clean 2×3 grid below the vitals section
- [ ] Each chart shows a line with multiple data points (not just 1 point)
- [ ] Normal range band is visible as a green shaded area
- [ ] Hovering over a data point shows the value + unit in tooltip
- [ ] Charts update silently after 30 seconds
- [ ] On narrow viewport (< 768px): charts stack vertically, no horizontal scroll
- [ ] Charts with all-normal values show the line within the green band
- [ ] Patient 3 (High Risk): SpO₂ and MAP charts show line outside the normal band

---

## PHASE 9 — AI Clinical Report Section

**Goal:** The AI report section renders below the charts. It includes the Hospital/Home Care toggle, reliability badge, 4 LLM section cards, and the top 5 SHAP factors.

### Files Created
```
frontend/src/components/report/ReportToggle.jsx
frontend/src/components/report/ReliabilityBadge.jsx
frontend/src/components/report/ReportSection.jsx
frontend/src/components/report/SHAPPanel.jsx
frontend/src/components/report/ReportContainer.jsx
```

### Tasks

**9.1 — Care mode state**

In `MonitorDashboard.jsx`, add:
```jsx
const [careMode, setCareMode] = useState('hospital')
```

Pass down to `ReportContainer`.

**9.2 — `ReportToggle.jsx`**

Two pill buttons: "🏥 Hospital" and "🏠 Home Care"
- Active pill: solid background in mode color (blue for hospital, green for home care)
- Inactive: dark background, muted text
- On click → calls `setCareMode('hospital'|'homecare')` which triggers a new `/api/report` call

**9.3 — `ReportContainer.jsx`**

- Manages report data fetching: when `predictData` updates OR `careMode` changes → calls `generateReport()`
- Shows loading state: skeleton placeholder cards while report generates
- Shows error state: "Report generation failed. Will retry on next reading."

**9.4 — `ReliabilityBadge.jsx`**

A single prominent badge:
```
┌─────────────────────────────────────────────┐
│  🧠 AI Clinical Assessment                  │
│                                             │
│  ✅ High Reliability     0.84               │
│  Response validated across all checks.      │
└─────────────────────────────────────────────┘
```

- Color matches reliability level (green/amber/red)
- `caution: true` → shows "⚠ Review carefully before acting" in amber
- Does NOT show the 9-method breakdown grid — just the single score and label

**9.5 — `ReportSection.jsx`**

One of the 4 LLM section cards. Props: `name`, `content`, `icon`, `color`, `subtitle`

Section header:
```
┌─────────────────────────────────────────────┐
│  📋  CURRENT CONDITION                      │
│       Patient status right now              │
└─────────────────────────────────────────────┘
```

Content area renders the section text:
- Lines starting with `•` → render as styled bullet rows
- Lines ending with `:` (subheadings like "Respiratory:") → render as cyan divider label
- Lines with `⚡` → render as priority action box with amber border
- `**bold**` → render as `<strong>`
- All other lines → styled bullet rows

**9.6 — `SHAPPanel.jsx`**

"What's driving this prediction" — top 5 SHAP factors:

```
┌─────────────────────────────────────────────┐
│  🔬 Key Risk Drivers                        │
│                                             │
│  💧 SpO₂   92.1%    ████████░░  +1.82  ↑   │
│  ❤️ HR     108 bpm  ██████░░░░  +1.41  ↑   │
│  🫁 RR     22 br/m  █████░░░░░  +1.08  ↑   │
│  📉 MAP    74 mmHg  ███░░░░░░░  +0.67  ↑   │
│  🌡️ Temp  38.2°C   ██░░░░░░░░  +0.34  ↑   │
└─────────────────────────────────────────────┘
```

Each row:
- Icon + feature label
- Current value + unit
- Horizontal bar (width proportional to relative impact)
- Impact number + ↑/↓ direction
- Red bar for increases-risk, green for reduces-risk

No "SHAP-identified risk factor" jargon — just "what's driving this prediction"

**9.7 — Render order in MonitorDashboard**

```
<ReportToggle />
<ReportContainer>
  <ReliabilityBadge />
  {sections.map(s => <ReportSection key={s.name} {...s} />)}
  <SHAPPanel factors={predictData.shap_top5} />
</ReportContainer>
<ClinicalDisclaimer />
```

### Verify Phase 9 Complete
- [ ] "Hospital" and "Home Care" toggle renders in top-right of the report section
- [ ] Clicking "Home Care" triggers a new report fetch and the content changes to plain English
- [ ] Reliability badge shows correct color and label
- [ ] All 4 section cards render with content (no empty cards)
- [ ] Section bullets render cleanly — no raw `•` characters, no `**bold**` markdown leaking through
- [ ] ⚡ Priority actions appear in amber boxes (patient 3, Hospital mode)
- [ ] SHAP panel shows 5 bars with correct colors (red for risk-increasing, green for risk-reducing)
- [ ] Report skeleton shows while API is loading (first load)
- [ ] Clinical disclaimer appears at the bottom

---

## PHASE 10 — Auto-Refresh, Loading States & Error Handling

**Goal:** The dashboard auto-refreshes every 30 seconds silently, shows meaningful loading states, and handles API errors gracefully.

### Files Modified
```
frontend/src/hooks/useMonitor.js    ← add countdown, error state
frontend/src/components/layout/HeaderBar.jsx  ← add countdown display
frontend/src/pages/MonitorDashboard.jsx       ← add error handling
```

### Tasks

**10.1 — Countdown timer in `useMonitor`**

```js
const [countdown, setCountdown] = useState(90)

// Tick countdown every second — freezes at 1 while fetch is in flight
useEffect(() => {
  const tick = setInterval(() => {
    setCountdown(c => {
      if (c <= 1) { fetchNext(); return 1 }  // freeze at 1, not 0
      return c - 1
    })
  }, 1000)
  return () => clearInterval(tick)
}, [patientId, fetchNext])

// Reset only after data arrives (in finally block of fetchNext)
// setCountdown(90) lives inside the finally{} of fetchNext
```

Show countdown in `HeaderBar` status strip. fetchingRef prevents overlapping fetches.

**10.2 — Loading state — skeleton cards**

While `isRefreshing === true`:
- SOFA gauge shows a subtle pulsing animation on the number (not a spinner)
- Vital cards dim slightly (opacity 0.7)
- Report section shows "Generating new report…" below the old content

The key UX principle: **never blank out previous data while loading**. Always show last-known-good values with a subtle indicator that a refresh is happening.

**10.3 — Error states**

If `POST /api/predict` fails:
- Show a non-blocking banner: "⚠ Refresh failed — showing last reading. Retrying in 30s."
- Keep previous data visible
- Auto-retry on next cycle

If `POST /api/report` fails:
- Show in report section: "Report unavailable for this reading. Previous report shown below."
- Keep previous report sections visible

If the backend is completely down:
- Header shows "⚠ Connection lost"
- All data stays frozen at last reading
- Retry continues every 30s silently

**10.4 — Row index overflow**

When `rowIdx` exceeds `totalRows` (end of simulation data), it wraps:
```js
const actualIdx = rowIdx % totalRows
```
The backend already handles this (`cur_idx = row_idx % total_rows`). Frontend just keeps incrementing.

**10.5 — Stop monitoring**

"Stop Monitoring" button in HeaderBar:
- Clears the interval
- Navigates back to `/`
- Resets `rowIdx` to 0 for that patient (so next time monitoring starts fresh)

### Verify Phase 10 Complete
- [ ] Countdown displays in header and counts down from 30 to 1, then resets
- [ ] After countdown hits 0: data refreshes, SOFA may change slightly
- [ ] During refresh: vitals dim slightly but remain visible
- [ ] Simulate API error: stop uvicorn backend → "Connection lost" appears in header
- [ ] Restart uvicorn → banner disappears, data resumes
- [ ] Let it run for 30+ readings (row index > 30 = wraps to 0 mod total_rows) — no crash
- [ ] "Stop Monitoring" button: clears timer, navigates to `/`, backend cycles stop
- [ ] Check SQLite: `SELECT COUNT(*) FROM prediction_history WHERE patient=2;` grows over time (confirms DB writes are accumulating)

---

## PHASE 11 — End-to-End Integration Testing

**Goal:** Test the complete system with all 3 patients, both care modes, multiple refresh cycles. Verify DB integrity throughout.

### Test Matrix

| Test | Patient | Care Mode | Expected |
|---|---|---|---|
| T01 | Patient 1 (Low Risk) | Hospital | SOFA 0–4, green gauge, no alert banner |
| T02 | Patient 2 (Moderate Risk) | Hospital | SOFA 5–9, amber gauge, moderate risk banner |
| T03 | Patient 3 (High Risk) | Hospital | SOFA ≥ 8, red gauge, HIGH RISK alert banner |
| T04 | Patient 2 (Moderate Risk) | Home Care | Same SOFA, report in plain English |
| T05 | Patient 3 (High Risk) | Home Care | Report shows home remedies + hospital escalation signs |
| T06 | Patient 1 | Hospital | Run 5 auto-refresh cycles, DB row count increases by 5 |
| T07 | All patients | Both modes | No JS console errors throughout |
| T08 | Patient 2 | Hospital | Reliability badge shows correct color |

### DB Integrity Checks (run after T06)

```bash
sqlite3 icu_monitor/models/icu_monitor.db << 'EOF'
-- Check vitals are accumulating
SELECT patient, COUNT(*) as count FROM patient_vitals GROUP BY patient;

-- Check predictions are accumulating
SELECT patient, COUNT(*) as count FROM prediction_history GROUP BY patient;

-- Check vitals have valid ranges (no NULL, no out-of-range)
SELECT COUNT(*) as total,
       SUM(CASE WHEN HR IS NULL OR HR < 30 OR HR > 220 THEN 1 ELSE 0 END) as bad_HR,
       SUM(CASE WHEN SpO2 IS NULL OR SpO2 < 50 OR SpO2 > 100 THEN 1 ELSE 0 END) as bad_SpO2
FROM patient_vitals;

-- Check SOFA predictions are in valid range
SELECT COUNT(*) as total,
       SUM(CASE WHEN SOFA < 0 OR SOFA > 24 THEN 1 ELSE 0 END) as invalid_SOFA
FROM prediction_history;

-- Check no prediction was saved with NULL values
SELECT COUNT(*) FROM prediction_history WHERE SOFA IS NULL OR Risk IS NULL;
EOF
```

All counts for `bad_HR`, `bad_SpO2`, `invalid_SOFA`, NULL checks should be **0**.

### UI Regression Checks

- [ ] Patient selection page: all 3 cards show correct risk colors
- [ ] Patient 1 monitoring: SOFA shows in green, no alert banner
- [ ] Patient 3 monitoring: large red SOFA number, red alert banner appears
- [ ] Toggle Home Care on patient 2: report content changes to plain language (no medical jargon, no IV lines)
- [ ] Toggle back to Hospital: report reverts to clinical language
- [ ] SHAP bars: top factor has the widest bar (proportional display)
- [ ] Trend charts: after 5+ readings, charts show a line (not just a dot)
- [ ] "Stop Monitoring" → returns to patient selection, no crashed state
- [ ] Can re-select same patient and start monitoring again
- [ ] Report section has no raw `•` characters, no `**` markdown artifacts
- [ ] Reliability badge is green for patient 1 (low risk, stable), may be amber for patient 3

---

## PHASE 12 — Polish, Responsive Design & Final QA

**Goal:** The dashboard looks and feels like professional clinical software. Works on all screen sizes. No rough edges.

### Tasks

**12.1 — Responsive layout**

- `< 640px` (mobile): vitals stack to 2 columns (3+3 → 2+2+2), SOFA gauge full width, trend charts 1 column
- `640px – 1024px` (tablet): 3-column vitals, 2-column trends, report full width
- `> 1024px` (desktop): current layout (SOFA + 3-col vitals side by side, 3-col trends)

Test in browser DevTools with device presets.

**12.2 — Typography & spacing pass**

- All section headers: `text-xl font-bold` with left border accent
- All vital numbers: `text-4xl font-black`
- All SOFA number: `text-7xl font-black` with risk color
- Section spacing: 24px between major sections, 16px between cards
- No horizontal scroll on any viewport width

**12.3 — Accessibility**

- All interactive elements have focus styles
- Risk status conveys meaning via both color AND text (not color alone)
- Font sizes: minimum 13px for body text, 11px only for labels/captions
- SOFA gauge has `aria-label="Predicted SOFA score {value} out of 24"`
- Vital cards have `aria-label="Heart rate {value} bpm, {status}"`

**12.4 — Performance**

- Report section: while LLM generates (10–20s), show previous report content + subtle "Updating…" spinner at top
- Charts: use `isAnimationActive={false}` on subsequent renders (animation only on first load)
- Images/icons: use emoji only, no external image assets
- Lighthouse performance score target: > 85

**12.5 — Final visual QA checklist**

- [ ] Dark navy background consistent throughout (`#0a1628`)
- [ ] No white backgrounds bleeding through anywhere
- [ ] No unstyled default browser components visible
- [ ] All badges/pills have rounded corners (`border-radius: 9999px`)
- [ ] All card borders are exactly 1px, not 2px or 0px
- [ ] LIVE dot pulses (CSS animation)
- [ ] Report section: section header cards have colored left border
- [ ] SHAP bars: all start from 0 (no negative bars displayed to user)
- [ ] Home Care mode: no hospital equipment mentioned in report (check with patient 3)
- [ ] Hospital mode: includes drug names, doses, routes (check with patient 3)

---

## PHASE 13 — fl_dashboard.py Migration (Separate, After Phase 12)

**Goal:** Build the ClinIQ FL Training Dashboard as a second React page, replacing `fl_dashboard.py`.

> This is a separate effort from the app.py migration. Phases 0–12 cover only the app.py dashboard. Tackle this after Phase 12 is verified complete.

### Scope
- New route: `/fl` or `/training`
- Shows: federated learning simulation + model info (equivalent of current fl_dashboard.py two-tab layout)
- Backend: `GET /api/fl/metadata`, `GET /api/fl/client/{id}`, `POST /api/fl/run-round`
- Kept entirely separate from patient monitoring flow

---

## PHASE 14 — Retire app.py

**Goal:** Archive the old Streamlit app. ClinIQ is the production system.

### Checklist Before Retirement
- [ ] All Phase 11 DB integrity checks pass
- [ ] Both care modes work correctly for all 3 patients
- [ ] Auto-refresh has run for 10+ cycles without errors
- [ ] No console errors in the browser
- [ ] Backend has been tested with uvicorn restart (models reload correctly)
- [ ] `.env` file is NOT committed to git (gitignore check)
- [ ] `models/icu_monitor.db` is NOT committed to git (gitignore check)

### Retirement Steps
```bash
cd icu_monitor
mkdir -p archive
mv app.py archive/app_streamlit_original.py
mv fl_dashboard.py archive/fl_dashboard_streamlit_original.py
# Update README to reflect new stack
```

> Do NOT delete these files permanently — move to `archive/`. They contain reference implementations of the ML pipeline and can be consulted if any backend service needs debugging.

---

## Phase Summary Table

| Phase | What Gets Built | Status |
|---|---|---|
| **0** | Prerequisites, deps installed | ✅ Complete |
| **1** | FastAPI skeleton, `/api/patients` | ✅ Complete |
| **2** | ML service, `/api/predict` | ✅ Complete |
| **3** | LLM service, `/api/report` | ✅ Complete |
| **4** | `/api/history`, `/api/vitals` | ✅ Complete |
| **5** | React scaffold, routing, API client | ✅ Complete |
| **6** | Patient selection page | ✅ Complete |
| **7** | Header + SOFA gauge + vitals grid | ✅ Complete |
| **8** | Vital trend charts | ✅ Complete |
| **9** | AI report section (toggle, sections, SHAP) | ✅ Complete |
| **10** | Auto-refresh 90s, error handling, countdown | ✅ Complete |
| **11** | End-to-end integration + DB integrity | ✅ Complete |
| **12** | Polish, glassmorphism design, final QA | ✅ Complete |
| **13** | FL dashboard migration | ⏳ Future (after Phase 12) |
| **14** | Retire app.py | ✅ Archived to archive/ |

### Key Deviations From Original Plan

| Plan | Actual |
|---|---|
| React Query for polling | Custom `useMonitor` hook (needed countdown control) |
| 30-second refresh | 90-second refresh (LLM generation takes ~15s) |
| shadcn/ui components | Custom glassmorphism components with Figtree font |
| Recharts only | Recharts + anime.js v4.5 (SHAP bars) + Framer Motion v14.5 |
| openai/gpt-oss-120b planned from start | Same model confirmed (quality over speed) |
| Cache TTL 60s | Cache TTL 90s (matches frontend timer) |


# ClinIQ Dashboard — Complete Testing Implementation Plan

> **Scope:** Everything inside `icu_monitor/` related to ClinIQ dashboard  
> **Not in scope (yet):** FL dashboard, Flower server/client, Streamlit app  
> **All test files live in:** `icu_monitor/tests/cliniq/`

---

## Directory Layout

```
tests/
├── cliniq/
│   ├── backend/
│   │   ├── conftest.py              ← shared TestClient + predict_p1 fixture
│   │   ├── unit/
│   │   │   ├── test_db_unit.py      ← db.py — schema, CRUD, sliding windows
│   │   │   ├── test_ml_service_unit.py  ← ml_service.py pure functions
│   │   │   └── test_llm_service_unit.py ← llm_service.py pure functions
│   │   ├── api/
│   │   │   ├── test_api_health.py
│   │   │   ├── test_api_patients.py
│   │   │   ├── test_api_predict.py
│   │   │   ├── test_api_vitals.py
│   │   │   ├── test_api_history.py
│   │   │   └── test_api_report.py
│   │   └── system/
│   │       └── test_system_flows.py ← end-to-end backend flows
│   └── frontend/
│       ├── setup.js                 ← @testing-library/jest-dom matchers
│       ├── unit/
│       │   ├── components/
│       │   │   ├── layout/          AlertBanner, HeaderBar, ThemeToggle
│       │   │   ├── overview/        SOFAGauge, VitalCard, VitalsGrid
│       │   │   ├── report/          ReportContainer, ReliabilityBadge,
│       │   │   │                    ReportSection, ReportToggle, SHAPPanel
│       │   │   ├── trends/          TrendChart, TrendsSection
│       │   │   ├── validation/      LLMValidationFAB, LLMValidationDrawer,
│       │   │   │                    SOFAValidationFAB, SOFAValidationDrawer
│       │   │   ├── cv/              CVMonitorFAB, CVMonitorModal
│       │   │   └── history/         HistoryFAB, HistoryDrawer
│       │   ├── hooks/
│       │   │   └── useMonitor.test.js
│       │   ├── contexts/
│       │   │   └── ThemeContext.test.jsx
│       │   └── api/
│       │       └── client.test.js
│       ├── integration/
│       │   ├── SelectPatient.test.jsx
│       │   └── MonitorDashboard.test.jsx
│       └── e2e/
│           ├── patient_selection.spec.js
│           ├── monitoring_flow.spec.js
│           └── report_generation.spec.js
└── fl/
    └── .gitkeep                     ← placeholder for future FL tests
```

---

## Tool Stack

| Layer | Tool | Config file |
|---|---|---|
| Backend tests | pytest + httpx | `pytest.ini` (root) |
| Frontend unit/integration | Vitest + React Testing Library | `frontend/vitest.config.js` |
| Frontend mocking | vi.mock + msw | inside test files |
| E2E | Playwright | `frontend/playwright.config.js` |

### Install commands

```bash
# Backend (already in venv)
pip install pytest pytest-asyncio httpx

# Frontend testing libraries
cd icu_monitor/frontend
npm install --save-dev \
  vitest \
  @vitest/coverage-v8 \
  @testing-library/react \
  @testing-library/jest-dom \
  @testing-library/user-event \
  jsdom \
  msw

# E2E (optional — Phase 6)
npm install --save-dev @playwright/test
npx playwright install
```

### `pytest.ini` (place at `icu_monitor/` root)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests that call the real LLM (deselect with -m "not slow")
    unit: pure-function tests with no I/O
    api: FastAPI endpoint tests (need TestClient)
    system: full end-to-end backend flows
```

### `frontend/vitest.config.js`

```js
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: [path.resolve(__dirname, '../tests/cliniq/frontend/setup.js')],
    include: [path.resolve(__dirname, '../tests/cliniq/frontend/**/*.test.{js,jsx}')],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/**/*.{js,jsx}'],
    },
  },
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
})
```

Add to `frontend/package.json` scripts:

```json
"test":          "vitest run",
"test:watch":    "vitest",
"test:coverage": "vitest run --coverage",
"test:e2e":      "playwright test"
```

---

## Phase 1 — Backend Unit Tests

**Goal:** Test every pure function and database helper in isolation.  
**Runner:** `pytest tests/cliniq/backend/unit/`  
**No external calls, no real ML model load required.**

### 1A — `test_db_unit.py`

Tests the `db.py` module against a **temp in-memory SQLite** database.

| # | Test | What it covers |
|---|---|---|
| 1 | `init_db()` creates `patient_vitals` table | Schema bootstrap |
| 2 | `init_db()` creates `prediction_history` table | Schema bootstrap |
| 3 | `init_db()` creates `sample_patients` table | Schema bootstrap |
| 4 | `init_db()` is idempotent (called twice, no error) | IF NOT EXISTS guard |
| 5 | `append_patient_vitals()` inserts a row correctly | CRUD |
| 6 | `get_patient_vitals()` returns rows ascending by id | Order |
| 7 | Sliding window — after 21 inserts, exactly 20 remain | DELETE logic |
| 8 | Cross-patient isolation — patient 1 row not in patient 2 | WHERE patient |
| 9 | `append_prediction()` inserts a row correctly | CRUD |
| 10 | `get_prediction_history()` limit respected | SELECT LIMIT |
| 11 | Prediction sliding window — after 51 inserts, exactly 50 remain | DELETE logic |
| 12 | `get_sample_patient_count()` returns 0 on empty table | COUNT query |
| 13 | `get_sample_patient_row()` returns None for missing idx | Null safety |
| 14 | `get_conn()` sets WAL journal mode | SQLite PRAGMA |

### 1B — `test_ml_service_unit.py`

Tests pure helper functions inside `ml_service.py` without loading ML models.

| # | Test | What it covers |
|---|---|---|
| 1 | `_is_clinical('HR_mean')` → True | VITAL_UNITS membership |
| 2 | `_is_clinical('latest_SpO2')` → True | VITAL_UNITS membership |
| 3 | `_is_clinical('septic')` → True | CLINICAL_KEYWORDS |
| 4 | `_is_clinical('weight')` → False | Not vital, not keyword |
| 5 | `_is_clinical('epinephrine')` → False for 'hr' | Boundary guard: substring trap |
| 6 | `_is_clinical('cirrhosis')` → False for 'rr' | Boundary guard: substring trap |
| 7 | `_is_clinical('crrt')` → False for 'rr' | Boundary guard: substring trap |
| 8 | `_is_clinical('latest_hr')` → True (word boundary match) | Non-letter boundary regex |
| 9 | `_VITAL_ABBR` contains exactly {"hr","rr","map","sbp","dbp","gcs","spo2"} | Set completeness |
| 10 | `load_training_metadata()['final_mae']` == 1.8968 | Fallback / real JSON values |
| 11 | `load_training_metadata()['final_r2']` == 0.5995 | Fallback / real JSON values |
| 12 | `load_training_metadata()['test_samples']` == 2559 | Test split correctness |
| 13 | Seed timestamps contain no hard-coded '2026-08-29' | Relative timestamp fix |
| 14 | Seed timestamps are parseable datetime strings | Format check |

### 1C — `test_llm_service_unit.py`

Tests `_cache_key()` and `judge_used` logic.

| # | Test | What it covers |
|---|---|---|
| 1 | `_cache_key(pred, 'hospital')` returns a non-empty string | Basic output |
| 2 | Different patients → different cache keys | C2 fix: patient_id included |
| 3 | Different care_modes → different cache keys | Mode separates cache |
| 4 | Different SOFA scores → different cache keys | Score separates cache |
| 5 | Different SBP values → different cache keys | C2 fix: SBP included |
| 6 | Different RR values → different cache keys | C2 fix: RR included |
| 7 | Identical inputs → same cache key | Deterministic |
| 8 | `generate_report()` result dict has `judge_used` key | C6 fix verification |
| 9 | `judge_used` is a bool | Type check |

---

## Phase 2 — Backend API Tests (Integration)

**Goal:** Test every HTTP endpoint through FastAPI TestClient.  
**Runner:** `pytest tests/cliniq/backend/api/`  
**Uses real SQLite DB (same DB as the running app).**

### 2A — `test_api_health.py`

| Test | Expected |
|---|---|
| `GET /health` → 200 | `{"status":"ok","service":"ClinIQ"}` |
| Service field is "ClinIQ" | Exact string match |

### 2B — `test_api_patients.py`

| Test | Expected |
|---|---|
| `GET /api/patients` → 200 with 3 items | List length == 3 |
| All required fields present per patient | `{id,name,short,icon,description,color,risk_level,sofa_range}` |
| IDs are exactly `{1,2,3}` | Set equality |
| Colors are 7-char hex strings | `startswith('#')` + `len==7` |
| Patient 1 risk level is "Low Risk" | Correct mapping |
| Patient 2 risk level is "Moderate Risk" | Correct mapping |
| Patient 3 risk level is "High Risk" | Correct mapping |
| `GET /api/patients/1` → 200, id==1 | Single patient fetch |
| `GET /api/patients/4` → 404 | Not found |
| `GET /api/patients/0` → 404 | Boundary |
| `GET /api/patients/-1` → 404 | Negative |

### 2C — `test_api_predict.py`

| Test | Expected |
|---|---|
| Required keys in response | `{patient_id,row_idx,cur_idx,total_rows,sofa_score,risk_level,...}` |
| `patient_id` echoed correctly | `body["patient_id"] == 2` |
| SOFA in `[0.0, 24.0]` | Range check |
| risk_level in known set | `{"Low Risk","Moderate Risk","High Risk"}` |
| risk_color is hex | `#RRGGBB` |
| conformal interval brackets SOFA | `lo ≤ sofa ≤ hi` |
| conformal lo ≥ 0 | Non-negative |
| conformal hi ≤ 24 | Max SOFA |
| Vitals keys all present | `{HR,RR,SpO2,Temp,SBP,DBP,MAP}` |
| vitals_status values valid | `{"normal","high","low"}` |
| SHAP is a list of ≤ 5 items | Length check |
| SHAP factor structure | `{feature, direction, rel_impact}` |
| direction in `{"up","down"}` | Enum check |
| rel_impact in `[0,1]` | Range check |
| cur_idx within total_rows | Bounds |
| Large row_idx wraps correctly | Modulo cycling |
| Consecutive row_idx increments cur_idx | Monotone |
| All 3 patients succeed | patient_id ∈ {1,2,3} |
| patient_id=4 → 400 | Validation |
| patient_id=0 → 400 | Validation |
| patient_id=-1 → 400 | Validation |
| Empty body → 422 | Pydantic validation |

### 2D — `test_api_vitals.py`

| Test | Expected |
|---|---|
| `GET /api/vitals/1` → 200, list | Type check |
| Default limit ≤ 20 | Sliding window cap |
| `?limit=5` → ≤ 5 rows | Custom limit |
| Required columns present | `{time,HR,RR,SpO2,Temp,SBP,DBP,MAP}` |
| HR in plausible range `[20,300]` | Clinical sanity |
| SpO2 in `[50,100]` | Clinical sanity |
| All patients succeed | patient_id ∈ {1,2,3} |
| patient_id=99 → 400 | Validation |
| patient_id=0 → 400 | Validation |
| `?limit=21` → 400 | Limit cap |
| `?limit=0` → 400 | Min limit |

### 2E — `test_api_history.py`

| Test | Expected |
|---|---|
| `GET /api/history/1` → 200, list | Type check |
| Default limit ≤ 5 | Default cap |
| `?limit=2` → ≤ 2 rows | Custom limit |
| Required columns present | `{Timestamp,SOFA,Risk}` |
| SOFA values numeric in `[0,24]` | Range check |
| All patients succeed | patient_id ∈ {1,2,3} |
| patient_id=99 → 400 | Validation |
| patient_id=0 → 400 | Validation |
| `?limit=0` → 400 | Min limit |
| `?limit=51` → 400 | Max limit |

### 2F — `test_api_report.py`

| Test | Expected |
|---|---|
| Valid request → 200 with required keys | `{sections,reliability_score,...}` |
| All 4 section keys present | `{CURRENT CONDITION,PROBABLE CAUSE,...}` |
| Sections have non-trivial content | `len(section) > 20` |
| reliability_score in `[0.0, 1.0]` | Range check |
| homecare mode works | status == 200 |
| `judge_used` key in response | C6 fix |
| `judge_used` is bool | Type check |
| Invalid care_mode → 400 | Validation |
| care_mode missing → 422 | Pydantic |
| patient_id=4 → 400 | Out of range |
| predict_result.patient_id mismatch → 400 | C5 fix |
| predict_result.vitals not dict → 400 | C5 fix |
| predict_result.sofa_score missing → 400 | C5 fix |
| Empty body → 422 | Pydantic |

---

## Phase 3 — Backend System Tests

**Goal:** Test complete multi-step workflows that simulate the React app's behaviour.  
**Runner:** `pytest tests/cliniq/backend/system/`

| Test | What it simulates |
|---|---|
| Full patient select → predict → vitals → history flow | SelectPatient + MonitorDashboard initial load |
| All 3 patients predict independently, no cross-contamination | Concurrent patients |
| Row cycling stays in bounds for any row_idx | 90-second refresh loop |
| Vitals table grows after predict calls (sliding window) | TrendsSection data source |
| Cross-patient history isolation | Patient 2 data doesn't appear in patient 1 history |
| Health endpoint stays 200 throughout a monitoring session | Uptime |
| Vitals from predict match vitals endpoint latest row | Consistency between endpoints |

---

## Phase 4 — Frontend Unit Tests

**Goal:** Test every React component, hook, context, and API function in isolation.  
**Runner:** `npx vitest run` (from `frontend/` directory)  
**Environment:** jsdom. Framer-motion, animejs, recharts, and axios are mocked.

### Global Mocks (in `setup.js`)

```js
// @testing-library/jest-dom matchers
import '@testing-library/jest-dom'

// Mock animation libraries
vi.mock('framer-motion', ...)
vi.mock('animejs', ...)
vi.mock('aos', ...)

// Mock recharts (chart renders nothing in jsdom)
vi.mock('recharts', ...)

// Mock @mediapipe/tasks-vision
vi.mock('@mediapipe/tasks-vision', ...)
```

### 4A — Layout Components

**`AlertBanner.test.jsx`**
| Test |
|---|
| Returns null when predictData is null |
| Returns null when risk_level is "Low Risk" |
| Renders alert when risk_level is "Moderate Risk" |
| Renders alert when risk_level is "High Risk" |
| Shows SOFA score in message |
| Shows "ELEVATED RISK" label for moderate |
| Shows "CRITICAL ALERT" label for high risk |
| Has `role="alert"` for accessibility |

**`ThemeToggle.test.jsx`**
| Test |
|---|
| Renders a button |
| Calls toggle function on click |
| Shows sun icon in dark mode |
| Shows moon icon in light mode |

**`HeaderBar.test.jsx`**
| Test |
|---|
| Renders patient name |
| Renders countdown value |
| Renders last-updated time when provided |
| Shows "—" or placeholder when no last-updated |
| Renders ThemeToggle |
| Shows "LIVE" indicator |

### 4B — Overview Components

**`VitalCard.test.jsx`**
| Test |
|---|
| Renders vital label and value |
| Renders unit string |
| Applies correct status color for "normal" |
| Applies correct status color for "high" |
| Applies correct status color for "low" |
| Renders "—" when value is null/undefined |
| Renders trend arrow up when status is "high" |
| Renders trend arrow down when status is "low" |

**`VitalsGrid.test.jsx`**
| Test |
|---|
| Renders exactly 7 VitalCard children |
| Passes HR, RR, SpO2, Temp, SBP, DBP, MAP |
| Handles null vitals gracefully (no crash) |

**`SOFAGauge.test.jsx`**
| Test |
|---|
| Renders SOFA score as text |
| Renders risk level label |
| SVG arc element is present |
| Shows zone markers at 6, 9, 12 |
| Handles score=0 without crash |
| Handles score=24 (max) without crash |

### 4C — Report Components

**`ReportSection.test.jsx`**
| Test |
|---|
| Renders section name |
| Renders content text |
| Renders subtitle |
| Empty content renders without crash |

**`ReliabilityBadge.test.jsx`**
| Test |
|---|
| Renders reliability score as percentage |
| Renders "High" label correctly |
| Renders "Moderate" label correctly |
| Renders "Low" label correctly |
| Shows caution text when caution prop provided |
| Omits caution block when caution is null |

**`ReportToggle.test.jsx`**
| Test |
|---|
| Renders "Hospital" button |
| Renders "Home Care" button |
| Active button has different style |
| Clicking inactive button calls setCareMode |
| Clicking active button does not re-fire unnecessarily |

**`SHAPPanel.test.jsx`**
| Test |
|---|
| Renders panel when factors are provided |
| Does not render when factors is empty/null |
| Shows feature name for each factor |
| Shows direction indicator (up/down) |
| Shows relative impact bar |
| Keeps last factors during refresh (factors=[]) |

**`ReportContainer.test.jsx`**
| Test |
|---|
| Renders skeleton when loading (no previous data) |
| Calls generateReport when predictData changes |
| Does NOT call generateReport when only non-clinical values change |
| Shows stale notice when refreshing with existing data |
| Shows error banner when API call fails (no previous data) |
| Shows stale notice + previous data when API call fails with existing data |
| 401 error shows "Invalid API key" message |
| 429 error shows "Rate limit reached" message |
| 503 error shows "AI service temporarily unavailable" message |
| Does not show report when predictData is null |
| reportKey changes when sofa_score changes |
| reportKey changes when GCS_eye changes |
| reportKey changes when stress changes |
| reportKey stays same when only timestamp changes |

### 4D — Trends Components

**`TrendChart.test.jsx`**
| Test |
|---|
| Renders chart container |
| Handles empty data array gracefully |
| Renders with valid vitals data |

**`TrendsSection.test.jsx`**
| Test |
|---|
| Renders vital selector tabs |
| Default selected vital is shown |
| Clicking another vital updates selection |
| Renders TrendChart |

### 4E — Validation Components

**`LLMValidationFAB.test.jsx`**
| Test |
|---|
| Renders button with aria-label "Open AI Report Validation" |
| Calls onClick when clicked and hasData=true |
| Is disabled (pointer-events: none, opacity 0.45) when hasData=false |

**`LLMValidationDrawer.test.jsx`**
| Test |
|---|
| Does not render when open=false |
| Renders when open=true |
| Shows "✓ 9 / 9 methods ran" when judge_used=true |
| Shows "⚠ 7 / 9 methods ran" when judge_used=false |
| All metric tabs are present |
| Calls onClose when close button clicked |

**`SOFAValidationFAB.test.jsx`**
| Test |
|---|
| Renders button with aria-label "Open SOFA Score Validation" |
| Calls onClick when clicked |

**`SOFAValidationDrawer.test.jsx`**
| Test |
|---|
| Does not render when closed |
| Renders when open |
| Shows MAE = 1.8968 |
| Shows R² = 0.5995 |
| Shows AUROC = 0.9461 |
| Shows PR-AUC = 0.6840 |
| Shows 3-class accuracy = 70.2% |
| Shows test split "2,559" |
| Confusion matrix renders 3 rows |
| High-risk recall shows 19.8% |
| Calls onClose when close button clicked |

### 4F — CV Components

**`CVMonitorFAB.test.jsx`**
| Test |
|---|
| Renders button with aria-label "Open Live CV Monitor" |
| Calls onClick when clicked |
| isActive prop changes visual style |

**`CVMonitorModal.test.jsx`**
| Test |
|---|
| Does not render when open=false |
| Renders when open=true |
| Shows supplementary label text |
| Shows "SOFA prediction uses dataset values" text |

### 4G — History Components

**`HistoryFAB.test.jsx`**
| Test |
|---|
| Renders button with aria-label "Open patient history" |
| Calls onClick when clicked |
| Shows recordCount when > 0 |

**`HistoryDrawer.test.jsx`**
| Test |
|---|
| Does not render when open=false |
| Renders table when open=true |
| Shows Timestamp, SOFA, Risk columns |
| Renders each history row |
| Shows empty state when data=[] |
| Calls onClose when close button clicked |

### 4H — Hook Tests

**`useMonitor.test.js`**
| Test |
|---|
| Calls runPredict on mount with patient_id and rowIdx=0 |
| predictData is set after successful fetch |
| rowIdx increments after each successful fetch |
| countdown resets to 90 after fetch completes |
| error state set on failed fetch |
| Changing patientId resets rowIdx to 0 |
| Changing patientId triggers new fetch |
| Concurrent fetches are guarded (fetchingRef) |
| StrictMode double-invoke guard (lastPatientRef) |

### 4I — Context Tests

**`ThemeContext.test.jsx`**
| Test |
|---|
| Provides isDark and toggleTheme |
| Default isDark state is boolean |
| toggleTheme flips isDark |
| Calling toggleTheme twice returns to original |

### 4J — API Client Tests

**`client.test.js`**
| Test |
|---|
| `getPatients()` calls `GET /api/patients` |
| `getPatient(2)` calls `GET /api/patients/2` |
| `runPredict(1, 5)` calls `POST /api/predict` with `{patient_id:1, row_idx:5}` |
| `generateReport(1,'hospital',pred)` calls `POST /api/report` |
| `getHistory(1,5)` calls `GET /api/history/1?limit=5` |
| `getVitals(1,20)` calls `GET /api/vitals/1?limit=20` |

---

## Phase 5 — Frontend Integration Tests

**Goal:** Test full page components with mocked API calls.  
**Runner:** Vitest with msw for API mocking.

### `SelectPatient.test.jsx`

| Test |
|---|
| Calls `getPatients` on mount |
| Shows loading state while fetching |
| Renders 3 patient cards after successful load |
| Shows error state on API failure |
| Clicking a patient card navigates to `/monitor/:id` |
| Patient name/description rendered per card |

### `MonitorDashboard.test.jsx`

| Test |
|---|
| Redirects to "/" if no patientId in route |
| Calls `getPatient(id)` on mount |
| Shows HeaderBar with patient name |
| Shows AlertBanner when risk_level is "High Risk" |
| Hides AlertBanner when risk_level is "Low Risk" |
| VitalsGrid receives predictData.vitals |
| SOFAGauge receives sofa_score |
| HistoryFAB opens HistoryDrawer on click |
| CVMonitorFAB opens CVMonitorModal on click |
| SOFAValidationFAB opens SOFAValidationDrawer on click |
| LLMValidationFAB opens LLMValidationDrawer on click |
| Tab switching between Overview / Trends / Report |

---

## Phase 6 — End-to-End Tests

**Goal:** Full browser-driven tests that spin up the real backend + frontend.  
**Tool:** Playwright  
**Requires:** Backend running on `localhost:8000`, frontend on `localhost:5173`

### `patient_selection.spec.js`

| Test |
|---|
| Navigate to `/` — patient cards load |
| Each card shows patient name, description, risk level |
| Clicking "Patient 1" navigates to `/monitor/1` |
| Dashboard header shows "Patient 1" |

### `monitoring_flow.spec.js`

| Test |
|---|
| Dashboard shows SOFA gauge |
| VitalsGrid shows 7 vital cards |
| Countdown timer visible in header |
| History FAB opens history drawer |
| History drawer shows table rows |
| CV FAB opens CV modal |
| CV modal shows supplementary disclaimer |
| SOFA Validation FAB opens drawer |
| SOFA drawer shows MAE/R² values |

### `report_generation.spec.js`

| Test |
|---|
| Report section shows loading skeleton on first load |
| Report section shows content after LLM call |
| 4 report cards rendered (Current Condition, Probable Cause, Risk Forecast, Immediate Actions) |
| Reliability badge rendered |
| Toggling "Home Care" mode triggers new report |
| LLM Validation FAB opens after report loads |
| LLM drawer shows method count |

---

## Running the Tests

```bash
# All backend tests
cd icu_monitor
pytest tests/cliniq/backend/ -v

# Backend unit tests only (fast, no ML load)
pytest tests/cliniq/backend/unit/ -v -m unit

# Skip slow LLM tests
pytest tests/cliniq/backend/ -v -m "not slow"

# All frontend unit + integration tests
cd icu_monitor/frontend
npx vitest run

# Frontend with coverage report
npx vitest run --coverage

# E2E tests (backend + frontend must be running)
npx playwright test
```

---

## Coverage Targets

| Layer | Target |
|---|---|
| `db.py` | ≥ 90% |
| `backend/api/*.py` | ≥ 85% |
| `backend/services/ml_service.py` (pure fns) | ≥ 80% |
| `backend/services/llm_service.py` (pure fns) | ≥ 75% |
| Frontend components | ≥ 70% |
| `hooks/useMonitor.js` | ≥ 80% |
| `contexts/ThemeContext.jsx` | ≥ 95% |
| `api/client.js` | ≥ 95% |

---

## Implementation Status

| Phase | Status | Files |
|---|---|---|
| 1A — DB unit | ✅ Implemented | `tests/cliniq/backend/unit/test_db_unit.py` |
| 1B — ML service unit | ✅ Implemented | `tests/cliniq/backend/unit/test_ml_service_unit.py` |
| 1C — LLM service unit | ✅ Implemented | `tests/cliniq/backend/unit/test_llm_service_unit.py` |
| 2A — Health API | ✅ Implemented | `tests/cliniq/backend/api/test_api_health.py` |
| 2B — Patients API | ✅ Implemented | `tests/cliniq/backend/api/test_api_patients.py` |
| 2C — Predict API | ✅ Implemented | `tests/cliniq/backend/api/test_api_predict.py` |
| 2D — Vitals API | ✅ Implemented | `tests/cliniq/backend/api/test_api_vitals.py` |
| 2E — History API | ✅ Implemented | `tests/cliniq/backend/api/test_api_history.py` |
| 2F — Report API | ✅ Implemented | `tests/cliniq/backend/api/test_api_report.py` |
| 3 — System flows | ✅ Implemented | `tests/cliniq/backend/system/test_system_flows.py` |
| 4 — Frontend unit | ✅ Implemented | `tests/cliniq/frontend/unit/**/*.test.{js,jsx}` |
| 5 — Frontend integration | ✅ Implemented | `tests/cliniq/frontend/integration/*.test.jsx` |
| 6 — E2E (Playwright) | ✅ Implemented | `tests/cliniq/frontend/e2e/*.spec.js` |

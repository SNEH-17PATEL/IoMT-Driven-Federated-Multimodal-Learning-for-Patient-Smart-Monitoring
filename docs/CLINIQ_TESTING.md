# ClinIQ Dashboard — Complete Testing Guide
**What every test does, why it exists, and how the whole suite fits together**

**Last updated:** 2026-11-24
**Result of the last full run (2026-11-24):** backend **218 passed** · frontend (Vitest) **232 passed** in 27 files · end-to-end (Playwright) **21 passed** → **471 tests, all passing**.
**Measured coverage:** backend 87% of lines · frontend 79% of lines (Section 13).

> **Scope:** the ClinIQ React dashboard and its FastAPI backend: everything under `icu_monitor/tests/cliniq/`.
> **Not covered here:** federated learning (Flower, `train_federated.py`, the Streamlit simulator). Those are in `FL_SERVER_CLIENT_DB_TESTING.md`.
> **How this document was checked:** every count, test name and number below comes from the real test files and a real run on 2026-11-24, not from memory. Where the suite has a gap or a quirk, it is stated plainly (Section 14).

---

## How to Read This Document

- **Sections 1–2** explain why tests matter and how the four layers (unit, integration, system, end-to-end) fit together.
- **Sections 3–5** give the directory layout, the tools, and the configuration.
- **Sections 6–11** describe the six phases of tests, one section each, with the reason each group of tests exists.
- **Section 12** lists the run commands, including two things that trip people up.
- **Section 13** gives measured coverage. **Section 14** lists known gaps and quirks. **Section 15** is the status table.

If you only want to run the tests, go straight to [Section 12](#12-running-the-tests).

---

## Table of Contents

1. [Why Testing Matters](#1-why-testing-matters)
2. [The Testing Pyramid — Four Layers, Four Purposes](#2-the-testing-pyramid--four-layers-four-purposes)
3. [Directory Layout](#3-directory-layout)
4. [Tool Stack](#4-tool-stack)
5. [Setup and Configuration Files](#5-setup-and-configuration-files)
6. [Phase 1 — Backend Unit Tests (85 tests)](#6-phase-1--backend-unit-tests-85-tests)
7. [Phase 2 — Backend API Tests (124 tests)](#7-phase-2--backend-api-tests-124-tests)
8. [Phase 3 — Backend System Tests (9 tests)](#8-phase-3--backend-system-tests-9-tests)
9. [Phase 4 — Frontend Unit Tests (209 tests)](#9-phase-4--frontend-unit-tests-209-tests)
10. [Phase 5 — Frontend Integration Tests (23 tests)](#10-phase-5--frontend-integration-tests-23-tests)
11. [Phase 6 — End-to-End Tests (21 tests)](#11-phase-6--end-to-end-tests-21-tests)
12. [Running the Tests](#12-running-the-tests)
13. [Coverage — Measured, Not Guessed](#13-coverage--measured-not-guessed)
14. [Known Gaps and Quirks](#14-known-gaps-and-quirks)
15. [Implementation Status](#15-implementation-status)

---

## 1. Why Testing Matters

**The core problem: confidence without tests is false confidence.**

Imagine a function that classifies SOFA risk:
```python
def classify_risk(sofa_score):
    if sofa_score < 5:  return "Low Risk"
    if sofa_score < 10: return "Moderate Risk"
    return "High Risk"
```
You try the dashboard by hand and it looks fine. Weeks later someone changes `< 10` to `<= 10` while tidying code. A patient with SOFA exactly 10 is now "Moderate Risk" instead of "High Risk". Nothing crashes and the dashboard looks identical, but the clinical guidance is silently wrong. A single test catches it at once:
```python
def test_sofa_10_is_high_risk():
    assert classify_risk(10) == "High Risk"      # fails the moment the bug appears
```
Tests turn silent regressions into loud, immediate failures. That matters most in a medical decision-support system, where wrong output has consequences.

**Three specific jobs the tests do in this project**
1. **Regression prevention.** What works today keeps working after tomorrow's changes.
2. **Interface contracts.** The backend and frontend were built separately. The API tests state exactly which fields each endpoint returns (for example `/api/predict` must return `sofa_score`, `risk_level`, `vitals`, `shap_top5`). If a field is renamed, a test fails before anyone tries to integrate.
3. **Edge-case documentation.** "patient_id = −1 → 400", "SOFA = 24 renders without crashing", "a report is never built from browser-supplied data". Without tests these boundaries are forgotten until a user hits them.

---

## 2. The Testing Pyramid — Four Layers, Four Purposes

```
                    ┌──────────────┐
                    │  E2E Tests   │  ← a real browser, the real app
                    │  (Phase 6)   │    slowest, most realistic, fewest
                    └──────┬───────┘
                   ┌───────┴────────┐
                   │  Integration   │  ← parts working together
                   │  (Phases 2, 5) │
                   └───────┬────────┘
               ┌───────────┴────────────┐
               │      Unit Tests        │  ← one function or component at a time
               │     (Phases 1, 4)      │    fastest, most numerous
               └────────────────────────┘
        (System tests, Phase 3, sit between integration and E2E)
```

| Layer | What it checks | Where | Count |
|---|---|---|---|
| **Unit** | One function or one component in isolation; everything external is replaced by a stand-in ("mock") | Phases 1 and 4 | 85 + 209 = 294 |
| **Integration** | Several parts together. Backend: the API router, database and ML code, called over HTTP. Frontend: a whole page with its API calls mocked | Phases 2 and 5 | 124 + 23 = 147 |
| **System** | A multi-step clinical session through the real backend ("select a patient → predict → vitals → history") | Phase 3 | 9 |
| **End-to-end** | A real browser, a real frontend and a real backend, no mocks | Phase 6 | 21 |

**Why all layers are needed:** each catches a different bug.
```
A typo in a helper                      → unit test
An API field renamed                    → integration test
Row cycling breaks after many readings  → system test
The report tab does not render in Chrome → end-to-end test
```

---

## 3. Directory Layout

```
icu_monitor/tests/cliniq/
├── __init__.py
├── backend/                                   ← 218 tests (pytest)
│   ├── conftest.py                            ← 4 shared fixtures
│   ├── unit/                                  ← 85 tests
│   │   ├── test_db_unit.py                    ← 17  database helpers (temporary database)
│   │   ├── test_ml_service_unit.py            ← 43  ML helper functions (no model loaded)
│   │   └── test_llm_service_unit.py           ← 25  report cache, concurrency, judge_used
│   ├── api/                                   ← 124 tests
│   │   ├── test_api_health.py                 ←  4  GET /health
│   │   ├── test_api_patients.py               ← 26  GET /api/patients and /{id}
│   │   ├── test_api_predict.py                ← 39  POST /api/predict
│   │   ├── test_api_vitals.py                 ← 20  GET /api/vitals/{id}
│   │   ├── test_api_history.py                ← 17  GET /api/history/{id}
│   │   └── test_api_report.py                 ← 18  POST /api/report  (10 are slow: real Groq calls)
│   └── system/
│       └── test_system_flows.py               ←  9  multi-step backend workflows
│
└── frontend/                                  ← 232 + 21 tests
    ├── unit/                                  ← 209 tests (Vitest)
    │   ├── components/
    │   │   ├── layout/        AlertBanner (9), HeaderBar (12), ThemeToggle (7)
    │   │   ├── overview/      SOFAGauge (8), VitalCard (14), VitalsGrid (6)
    │   │   ├── report/        ReliabilityBadge (8), ReportContainer (22),
    │   │   │                  ReportSection (5), ReportToggle (7), SHAPPanel (9)
    │   │   ├── trends/        TrendChart (5), TrendsSection (6)
    │   │   ├── validation/    LLMValidationDrawer (8), LLMValidationFAB (4),
    │   │   │                  SOFAValidationDrawer (13), SOFAValidationFAB (3)
    │   │   ├── cv/            CVMonitorFAB (4), CVMonitorModal (6), cvScoring (16)
    │   │   └── history/       HistoryDrawer (8), HistoryFAB (5)
    │   ├── hooks/             useMonitor (11)
    │   ├── contexts/          ThemeContext (5)
    │   └── api/               client (8)
    ├── integration/           ← 23 tests (Vitest)
    │   ├── SelectPatient.test.jsx   (7)
    │   └── MonitorDashboard.test.jsx (16)
    └── e2e/                   ← 21 tests (Playwright)
        ├── patient_selection.spec.js   (7)
        ├── monitoring_flow.spec.js     (8)
        └── report_generation.spec.js   (6)
```
Plus the shared frontend setup file `frontend/vitest-setup.js` (it lives inside `frontend/`, not in `tests/`; Section 5 explains why).

**Why `conftest.py` is special.** It is a pytest convention: pytest finds it automatically and makes its fixtures (reusable test objects) available to every test file in the same folder and below, with no `import`. The backend `conftest.py` provides `client`, `predict_p1`, `predict_p2` and `predict_p3`.

---

## 4. Tool Stack

### pytest (backend)
Python's standard testing framework. It finds files named `test_*.py`, runs functions named `test_*`, and reports failures with a clear diff.
```python
def test_sofa_10_is_high_risk():
    assert classify_risk(10) == "High Risk"
# pytest handles discovery, running, reporting and the exit code
```
**Fixtures** are reusable setup. Instead of building a test client in every test, define it once and ask for it by name:
```python
@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:        # a test HTTP client wired to the FastAPI app
        yield c

def test_health_returns_200(client):  # pytest injects the fixture by name
    assert client.get("/health").status_code == 200
```
Markers (in `pytest.ini`) label tests so groups can be selected: `slow`, `unit`, `api`, `system`.

**Also used in the backend tests:** `monkeypatch` (temporarily replaces a variable or function and restores it afterwards), `tmp_path` (a temporary folder), `threading` (the concurrency test), and `inspect.getsource` (three "source-level" tests; Section 14).

> **Corrected from earlier versions of this document.** It said the backend tests use `pytest-asyncio` and `httpx.AsyncClient` to call async endpoints, and that the frontend integration tests use **MSW (Mock Service Worker)**. Neither is true: **no test uses `async def`, `AsyncClient`, or MSW**. The backend tests call the API with FastAPI's synchronous `TestClient`, and the frontend tests mock the API module with `vi.mock`. (`pytest-asyncio` and `httpx` happen to be installed, because `TestClient` needs `httpx`.)

### FastAPI `TestClient`
Runs the **real** FastAPI app inside the test process and sends it HTTP requests without opening a network port. The real routers, real SQLite database and real ML code all run, so API tests are genuine integration tests. It is session-scoped, so the model files load once for the whole run.

### Vitest (frontend unit and integration)
The frontend equivalent of pytest. It is built for Vite (the frontend's build tool), so it shares the same import aliases and settings (`@` means `frontend/src`). Version 3.2.7.

### React Testing Library (`@testing-library/react`)
Provides `render()` and `screen`. **Philosophy: test what the user sees, not internal implementation.**
```js
// Brittle (breaks on refactor):       expect(component.state.isHigh).toBe(true)
// Robust (tests what the user sees):  expect(screen.getByText('CRITICAL ALERT')).toBeInTheDocument()
```
Tests written this way survive refactors, because they only care that the right thing is on screen.

### jsdom (the virtual browser)
Frontend unit tests run in Node.js, not a real browser. jsdom simulates the browser's DOM (`document`, `window`, elements), so Vitest runs with `environment: 'jsdom'`. **What it cannot do:** real CSS layout, WebGL, GPU, browser quirks. Those need the end-to-end tests.

### `vi.mock` (replacing a module with a fake)
Libraries that need a real browser are replaced with simple stand-ins so the component logic can still be tested. Globally (in `vitest-setup.js`): `framer-motion`, `animejs`, `aos`, `recharts` and `@mediapipe/tasks-vision`. Per test file: the API client and the `useMonitor` hook, where a test needs to control what comes back.

### Playwright (end-to-end)
Controls a real Chromium browser: it clicks, waits and reads the page like a user. It needs a real backend and frontend running. The project uses **one worker, Chromium only**, a 120-second test timeout and a trace kept on failure.

---

## 5. Setup and Configuration Files

### `pytest.ini` (in `icu_monitor/`)
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests that call the real LLM API (deselect with -m "not slow")
    unit: pure-function tests with no I/O
    api: FastAPI endpoint tests via TestClient
    system: full end-to-end backend flows
```
- `testpaths = tests`: pytest only looks inside `tests/`.
- `addopts = -v --tb=short`: verbose names and short failure traces by default.
- `-m "not slow"` skips the 10 tests that call the real Groq API (saves time and API credits).

### `frontend/vitest.config.js`
- `plugins: [react()]` handles JSX.
- `globals: true` makes `describe`, `it`, `expect`, `vi` available without imports.
- `environment: 'jsdom'` simulates the browser DOM.
- `setupFiles: ['./vitest-setup.js']` runs the setup file before every test file.
- `include`: `tests/cliniq/frontend/**/*.test.{js,jsx}`. Only `.test.` files run, so the Playwright `.spec.js` files in `e2e/` are not picked up by Vitest.
- `server.fs.allow` lets Vite read test files that live **outside** `frontend/`.
- `coverage`: provider `v8`, reporters `text` and `html`, measuring `src/**/*.{js,jsx}` except `src/main.jsx`.
- `resolve.alias`: `'@'` → `frontend/src`.

### `frontend/vitest-setup.js` (the global setup)
> **Corrected:** earlier versions called this `tests/cliniq/frontend/setup.js`. The real file is **`frontend/vitest-setup.js`**. It has to live inside `frontend/` so Vite can find `node_modules` (for example `@testing-library/jest-dom`).

What it does before every test file:
1. Loads the jest-dom matchers (`toBeInTheDocument`, `toBeVisible`, `toHaveTextContent`…).
2. Makes React global (`globalThis.React`) for JSX in files outside the Vite root.
3. Sets the CSS custom properties components read (`--t1`, `--s1`, `--ok-l`, `--danger-l`, `--warn-l`…) so styles resolve.
4. **Mocks** `framer-motion` (every `motion.*` becomes a plain element; `AnimatePresence` returns its children), `animejs` (`animate` and `createTimeline` chain correctly), `aos`, `recharts` (charts become simple `<div data-testid="chart">`), and `@mediapipe/tasks-vision`.
5. **Fakes browser APIs** jsdom lacks: `IntersectionObserver`, `ResizeObserver` and `window.matchMedia`.

### `frontend/playwright.config.js`
- `testDir: ../tests/cliniq/frontend/e2e`, `testMatch: **/*.spec.js`.
- `timeout: 120_000`, `fullyParallel: false`, `workers: 1`.
- `baseURL: http://localhost:5173`, `trace: retain-on-failure`, Chromium only.
- `webServer`: starts `npx vite --port 5173 --strictPort` automatically, or **reuses** one already running. **It does not start the backend**: that must already be running on port 8000.

### `frontend/package.json` test scripts
```json
"test":          "vitest run",              // run all frontend tests once
"test:watch":    "vitest",                  // re-run on file changes
"test:coverage": "vitest run --coverage",   // with a coverage report
"test:e2e":      "playwright test"          // Playwright end-to-end tests
```

---

## 6. Phase 1 — Backend Unit Tests (85 tests)

**Goal:** test helper functions and database helpers in isolation: no network, no ML model load, no external service. `test_db_unit.py` has 17 tests, `test_ml_service_unit.py` 43, `test_llm_service_unit.py` 25.

### 1A — `test_db_unit.py` (17 tests)
It redirects `db.DB_PATH` to a fresh **temporary** database for each test (the `tmp_db` fixture), so the real `models/icu_monitor.db` is never touched.

| Group | Tests | What is checked and why |
|---|---|---|
| Schema (4) | `init_db` creates `patient_vitals`, `prediction_history` and `sample_patients`; is **idempotent** (safe to run twice) | A fresh clone must start with a working database, and the backend runs `init_db()` on every start |
| Vitals (5) | An insert works; rows come back in **ascending time order**; the **sliding window keeps only 20** after 21 inserts; patient 2 does not see patient 1's rows; `limit` is respected | The 20-reading window drives all trend features; isolation prevents mixing patients |
| Prediction history (5) | Insert; `limit`; the window keeps **50** after 51 inserts; cross-patient isolation; the stored SOFA value round-trips | History is capped at 50 rows per patient |
| Sample patients (2) | An empty table gives count 0; a missing row returns `None` | The replay logic must cope with missing data |
| Connection (1) | `get_conn()` uses **WAL** journal mode | WAL lets the API read while it writes without "database is locked" errors |

### 1B — `test_ml_service_unit.py` (43 tests)
It imports only constants and pure functions from `ml_service.py`, so no model is loaded.

| Group | Tests | What is checked and why |
|---|---|---|
| `_is_clinical` with known vitals (4) | `HR_mean`, `latest_SpO2`, `GCS_eye_opening`, `stress_score` are clinical | These are the inputs SHAP is allowed to show |
| `_is_clinical` with keywords (3) | `septic_shock_flag`, `respiratory_distress`, `vasopressor_use` are clinical | Clinical note words are recognised too |
| Non-clinical (3) | `weight`, `age`, `glucose` are not | Unrelated features are filtered out |
| **Boundary guards (8)** | `epinephrine`, `norepinephrine`, `thrombocytopenia` do **not** match "hr"; `cirrhosis`, `crrt`, `hemorrhage` do **not** match "rr"; `latest_hr` and `trend_rr` **do** match | "hr" is a substring of "epinephrine". Without a whole-word match, a drug name would be reported as "Abnormal heart rate" in the SHAP explanation. This was a real bug (fix "C3") and these tests lock it in |
| `_VITAL_ABBR` (8) | Contains `hr`, `rr`, `map`, `sbp`, `dbp`, `gcs`, `spo2` and **exactly** those seven | The whole-word matching depends on this list |
| `NORMAL_RANGES` (3) | Has all 7 vitals; HR is 60–100; SpO₂ is 95–100 | Status badges depend on these ranges |
| **`ALERT_THRESHOLD` (2)** | It equals **7.0**, and equals `model_utils.ALERT_THRESHOLD` | The dashboard and the training report must use one shared number. If either side changes it alone, a test fails |
| `load_training_metadata` (6) | Returns a dict; `final_mae` is between 0 and 5; `final_r2` between 0 and 1; `test_samples` is a positive int; `train_samples` ≈ 4 × `test_samples` (the 80/20 split); `best_round` is an int | These read the **live** `models/training_metadata.json`, so they check the values are *sane*, not that they equal one training run; the file changes on every retrain |
| Seed timestamps (3, source-level) | The source has no hard-coded `2026-08-29`; it uses `timedelta`; the seed loop starts at `range(1, min(`) | Seed rows must use relative timestamps, and must start at row 1 so row 0 (the current reading) is not duplicated (fix "C7"). **These three read the source code, not behaviour** (Section 14) |
| `RISK_MAP` (3) | Has `GCS` and `HR` keys; all values are strings | The key-risk labels are built from it |

### 1C — `test_llm_service_unit.py` (25 tests)
No real AI call is made.

| Group | Tests | What is checked and why |
|---|---|---|
| Cache key basics (2) | A non-empty string; **deterministic** | Same reading → same key |
| What separates a key (8) | Different patient, care mode, SOFA, SBP, RR, HR, SpO₂ give **different** keys; the key contains the patient id | The cache must never serve patient 1's report for patient 2, or an old reading's report for a new reading (fix "C2") |
| **Every vital in the prompt (5, parametrised)** | DBP, MAP, Temp, GCS eye and stress each change the key | If a vital appears in the AI prompt but not in the key, a changed reading would get a report quoting the **old** numbers. The frontend `reportKey` mirrors this |
| Missing data (2) | A missing vital field, or a missing `vitals` key, does not crash | Robustness |
| `judge_used` (3, source-level) | The result dict includes `judge_used`; it is set `False` when the judge fails; failures are logged to stderr | The dashboard shows "7 / 9 methods ran" when the judge failed (fix "C6"). **Source-level checks** (Section 14) |
| **TTL cache (5)** | An entry is served until the 90-second TTL then expires; setting a new entry **evicts every expired entry**; a repeat request is served from cache; **4 simultaneous identical requests make exactly one AI call**; a **failed report is not cached** and the next request retries | The cache limits cost and avoids stale text; the concurrency test uses 4 real threads and a gate |

---

## 7. Phase 2 — Backend API Tests (124 tests)

**Goal:** call each endpoint over HTTP with FastAPI's `TestClient`. The real routers, database and ML code all run. These tests need the model files and the database to exist.

The fixtures (`conftest.py`): `client` (session-scoped, opens the app once so the models load once), and `predict_p1`, `predict_p2`, `predict_p3` (a cached `/api/predict` result per patient, reused by report and system tests).

### 2A — `test_api_health.py` (4)
`GET /health` returns 200, `status == "ok"`, `service == "ClinIQ"`, and a JSON content type.

### 2B — `test_api_patients.py` (26)
| Group | What is checked |
|---|---|
| List (9) | 200; exactly 3 patients; each has `id`, `name`, `short`, `icon`, `description`, `color`, `risk_level`, `sofa_range`; ids are {1, 2, 3}; colours are 7-character hex; risk levels are valid; names, icons and SOFA ranges are non-empty |
| Risk mapping (3) | Patient 1 is Low Risk, patient 2 Moderate, patient 3 High |
| One patient (9) | For ids 1, 2, 3: returns 200, echoes the id, has the required fields |
| Not found (5) | Ids 4, 5, 99, 0 and −1 return **404** |

### 2C — `test_api_predict.py` (39)
| Group | Tests | What is checked |
|---|---|---|
| Response shape (8) | Required keys; `patient_id` and `row_idx` echoed; SOFA in 0–24; risk level known; risk colour is hex; `is_alert` is a bool | The contract the frontend relies on |
| Conformal interval (4) | `conformal_lo ≤ sofa ≤ conformal_hi`; lo ≥ 0; hi ≤ 24; both numeric | The 90% range is always valid |
| Vitals (7) | Keys present; HR numeric; SpO₂ in 50–100; HR in 20–300; temperature in range; `vitals_status` keys and values (`normal`/`high`/`low`) | Plausible values reach the UI |
| SHAP (6) | A list; at most 5; each has `feature`, `direction`, `rel_impact`; direction is `up` or `down`; `rel_impact` in 0–1; `feature` is a string | The SHAP panel's contract |
| Row cycling (4) | `cur_idx` within `total_rows`; a huge `row_idx` (99,999) wraps; consecutive rows advance `cur_idx` by one (mod total); `total_rows > 0` | The 30 recorded readings replay in a loop |
| All patients (3, parametrised) | Patients 1, 2, 3 succeed | |
| Errors (7) | Patient ids 4, 5, 99, 0, −1 → **400**; an empty body and a missing `row_idx` → **422** | Input validation |

> **Quirk:** `test_predict_is_alert_true_for_high_sofa` only checks that `is_alert` is a boolean, despite its name. The comment says so ("we do not guarantee it but verify type"). See Section 14.

### 2D — `test_api_vitals.py` (20)
`GET /api/vitals/{id}`: 200; a list; the default is at most 20 rows; `limit=5` and `limit=1` are respected; columns `time`, `HR`, `RR`, `SpO2`, `Temp`, `SBP`, `DBP`, `MAP` present; `time` is a string; HR, SpO₂, SBP and RR are in plausible ranges; patients 1–3 succeed; errors: ids 4, 99, 0 → 400, and `limit` of 21, 0 or −1 → **400** (the limit must be 1–20).

### 2E — `test_api_history.py` (17)
`GET /api/history/{id}`: 200; a list; the default is at most 5; `limit=2` is respected; columns `Timestamp`, `SOFA`, `Risk`; SOFA values are numeric and in 0–24; risk values are non-empty; patients 1–3 succeed; errors: ids 4, 99, 0 → 400, and `limit` of 0, 51 or −1 → **400** (the limit must be 1–50).

### 2F — `test_api_report.py` (18)
| Group | Tests | What is checked |
|---|---|---|
| **Slow happy path (10)** | Hospital mode returns 200; required keys; all 4 sections present; each section is a string longer than 20 characters; reliability score is a float in 0–1; label is High/Moderate/Low; **Home Care mode** returns 200 with all 4 sections; `judge_used` is present and a bool | These call the **real Groq API** (about 16 seconds in total). Marked `slow` |
| Validation, no AI call (4) | A bad care mode → 400; a missing care mode → 422; patient 4 → 400; patient 0 → 400 | |
| **Server-side data (3)** | With no earlier prediction, the report returns **409**; **client-supplied data is ignored**: even a malformed `predict_result` for another patient is never used, and the server's own stored prediction is what reaches the report builder; the minimal body `{patient_id, care_mode}` is enough | The report is built **only** from the server's own last prediction, never from browser data |
| Empty body (1) | → 422 | |

> **Quirk:** the 10 slow tests still send `predict_result` in the request body (the old request shape). The API ignores that field now, so the tests work, but they do not exercise the "minimal body" path in the happy case. The three "server-side data" tests above do (Section 14).

---

## 8. Phase 3 — Backend System Tests (9 tests)

**Goal:** simulate real multi-step sessions through the real database and ML stack. Each test is independent.

| Test | The flow it checks |
|---|---|
| `full_patient_select_and_monitor_flow` | List patients (3) → pick patient 2 → predict row 0 (`cur_idx` 0) → predict row 1 (`cur_idx` 1) → vitals (≥ 1 row) → history (SOFA in 0–24). Mirrors SelectPatient → MonitorDashboard → History drawer |
| `all_patients_predict_independently` | Patients 1, 2, 3 each return their own valid prediction |
| `row_cycling_stays_in_bounds` | Row indices 0, 1, 50, 500, 10,000, 99,999 always give `0 ≤ cur_idx < total_rows` |
| `vitals_accumulate_after_predict_calls` | Predict calls append to the vitals window |
| **`cross_patient_history_isolation`** | Two predictions for patient 2 do **not** change patient 1's history count |
| `predict_then_vitals_consistency` | The HR returned by `/api/predict` appears in `/api/vitals` |
| `health_endpoint_stays_responsive_throughout` | `/health` is 200 before and after a monitoring sequence |
| `history_contains_latest_prediction_data` | After a predict, the history count does not shrink |
| `vitals_endpoint_returns_valid_sofa_consistent_data` | Returned vitals are numeric and in range |

> **Note:** a few of these assertions are deliberately loose because the sliding windows are capped (20 vitals, 50 history rows) and already full: for example "count ≥ before" rather than "count + 1".

---

## 9. Phase 4 — Frontend Unit Tests (209 tests)

**Goal:** test each React component, hook and helper in isolation with `render()`; the animation, chart and camera libraries are mocked by `vitest-setup.js`.

### 4A — Layout (28)
- **AlertBanner (9):** renders nothing when `predictData` is null or the risk is Low; renders for Moderate and High; shows the SOFA score; label **ELEVATED RISK** for Moderate and **CRITICAL ALERT** for High; `aria-live` is assertive for High and polite for Moderate (so screen readers announce urgency correctly).
- **HeaderBar (12):** renders; the LIVE indicator; the countdown in seconds; "···" while refreshing; the patient name; the Exit button; the ClinIQ brand mark; the last-updated time; "connection lost" on error; "All Systems Operational" otherwise; no timestamp when `lastUpdated` is null; the theme toggle is an accessible button.
- **ThemeToggle (7):** is a button with an `aria-label`; shows a text label unless `compact`; clicking toggles the theme; sun icon in dark mode, moon in light.

### 4B — Overview (28)
- **SOFAGauge (8):** renders; shows the score and risk level; has the SVG arc; handles score 0 and score 24 without crashing; shows the conformal interval values; draws **3 zone-marker lines** at SOFA 6, 9 and 12.
- **VitalCard (14):** label, value and unit; a null value shows "—" without crashing; status labels **Normal**, **Above Normal**, **Below Normal**; up-arrow for high and down-arrow for low; the trend section.
- **VitalsGrid (6):** renders with valid vitals; shows HR and SpO₂; handles null vitals and undefined `predictData`; renders all 7 vital card labels (including DBP).

### 4C — Report (51)
- **ReliabilityBadge (8):** the percentage; the High, Moderate and Low labels; the caution block appears only when `caution` is truthy; the score-ring SVG; the "AI Clinical Assessment" header.
- **ReportContainer (22)** is the most heavily tested component, because it decides *when* to call the AI:
  - skeleton while loading; calls `generateReport` when data arrives; shows content and **all 4 sections**; clear messages for errors **401, 429 and 503**;
  - does **not** call again when non-clinical fields change, and does call again when `sofa_score`, **`GCS_eye`** or **`stress`** change;
  - **requests a report for each care mode**, then **prefetches the other mode** once the first has loaded, and **switches mode instantly from cache** with no extra request and no skeleton;
  - never shows one mode's report under the other tab while loading;
  - **a new reading empties the cache**, so both modes are regenerated;
  - **a failed report is not cached**, so the same reading retries;
  - renders nothing when `predictData` is null; calls `onReportData` after load; shows a "stale" notice while refreshing, or if the API fails after data exists.
- **ReportSection (5):** the title-case section name; the content; the subtitle; no crash on empty content; works with no colour.
- **ReportToggle (7):** the Hospital and Home Care buttons; clicking calls `setCareMode` with `"hospital"` or `"homecare"`; two buttons; the active one has a gradient; clicking the active one still fires.
- **SHAPPanel (9):** renders with factors, nothing for empty or null; shows feature names; **preserves the last factors when an empty array arrives** (the panel does not blank out if SHAP fails once); at most 5 factors; ↑ and ↓ indicators; bars carry a `data-target-width`.

### 4D — Trends (11)
- **TrendChart (5):** renders with data; the chart container; handles an empty array and null; renders for SpO₂.
- **TrendsSection (6):** renders; shows skeleton cards while loading; after data loads shows **7 charts** and all 7 vital labels.

### 4E — Validation (28)
- **LLMValidationDrawer (8):** hidden when closed; shown when open; **"9 / 9 methods ran"** when the judge was used and **"7 / 9 methods ran"** with a warning when it was not; no crash when `reportData` is null; the three tabs (Overview, 9 Methods, Findings); the close button.
- **LLMValidationFAB (4):** the `aria-label`; clicks work when data exists; it is visually disabled (low opacity, no pointer events) when there is no report yet, and full opacity otherwise.
- **SOFAValidationDrawer (13)** checks the drawer reads **live** from the training metadata (`_training_meta`) and never shows hard-coded numbers: MAE, R², AUROC and PR-AUC come from the metadata; the **3-class accuracy is computed from the confusion matrix**; the test and total sample counts; the live alert threshold and alert recall; the High-risk recall from `per_class`; the confusion-matrix rows; **placeholders ("—") instead of a crash when the metadata is missing**; the close button.
- **SOFAValidationFAB (3):** the `aria-label`; clicking; renders.

### 4F — Camera monitor (26)
- **CVMonitorFAB (4):** the `aria-label`; clicking; renders when active; the active state renders different HTML.
- **CVMonitorModal (6):** hidden when closed, shown when open; the supplementary label; the footer says live readings feed the next SOFA prediction; no score cards before the camera runs; **live readings are cleared (`null` sent) when the monitor closes**.
- **cvScoring (16)** tests the exported scoring maths directly (no camera needed):
  - **PSPI (4):** a neutral face gives 0; the formula adds correctly (example 7.5); the maximum is 16; a full smile cancels the score;
  - **stress (4):** the first frame seeds the average; maximal pain and agitation give 10; smoothing uses α = 0.15; the bands LOW < 3.5 ≤ MODERATE < 6.5 ≤ HIGH (checked at 3.4, 3.5, 6.4, 6.5);
  - **eye aspect ratio (1)** and **eye state (2):** the blink value overrides the ratio; open, partial and closed thresholds;
  - **GCS window (5):** E4 at ≥ 70% open (blinks allowed); E3 at ≥ 30% open or partly open; E2 for a flicker of at least 0.1 s; E1 when eyes stay closed; a one-frame twitch is not a flicker.

  Details of the algorithm: `docs/CV_MONITOR.md`.

### 4G — History (13)
- **HistoryDrawer (8):** no table when closed; renders when open; the column headers (including Timestamp and Risk); the close button; a SOFA of 5.5 shows as "5.5"; an empty history shows an empty-state message.
- **HistoryFAB (5):** the `aria-label`; clicking; renders with `recordCount` 0 and above 0; has an SVG icon.

### 4H — `useMonitor` hook (11)
This hook runs the 90-second polling loop, so these tests protect the live behaviour:
calls `runPredict` on mount with the patient and row 0; sets `predictData` after a successful fetch; sets `error` on a failed fetch; `isRefreshing` is true during a fetch; the countdown **resets to 90** after a fetch; changing the patient resets the data and triggers a new fetch; `lastUpdated` is set; the countdown **decrements** with the timer; the row index goes to 1 on the second fetch; **the concurrent-fetch guard prevents overlapping calls**; **the React StrictMode double-invoke guard prevents a duplicate first fetch**.

### 4I — `ThemeContext` (5)
Provides the theme and `toggleTheme`; toggling flips the theme and twice returns to the original; sets `data-theme` on the document element.

### 4J — API client (8)
With `axios` mocked, checks each call: `getPatients` → `GET /patients`; `getPatient(2)` → `/patients/2`; `runPredict(1, 5)` → `POST /predict` with `{patient_id: 1, row_idx: 5}`; **`generateReport` sends only `patient_id` and `care_mode`**; `getHistory(1, 5)` → `/history/1?limit=5`; `getVitals(1, 20)` → `/vitals/1?limit=20`; the default limits are 50 (history) and 20 (vitals).

> **Not tested:** `runPredict` with camera data (`cv_gcs_eye`, `cv_stress`) is never exercised in `client.test.js` (Section 14).

---

## 10. Phase 5 — Frontend Integration Tests (23 tests)

**Goal:** render a **whole page** with the real components, but with its API calls replaced by mocks (`vi.mock` on `@/api/client.js`), inside a real `MemoryRouter`, `QueryClientProvider` and `ThemeProvider`. (There is no MSW; Section 4.)

### `SelectPatient.test.jsx` (7)
Calls `getPatients` on mount; renders 3 patient cards after load; shows the descriptions; **clicking a card navigates to the monitor page**; shows an error state when the API fails; shows ClinIQ branding; shows no cards while loading.

### `MonitorDashboard.test.jsx` (16)
`getPatient`/`getVitals`/`getHistory`/`generateReport` are mocked, and **`useMonitor` is mocked** so each test controls the prediction data directly.

| What is checked |
|---|
| Renders; fetches patient data on mount; shows the SOFA score and risk level from `predictData`; shows the patient name and the HR value in the vitals grid; shows the countdown from the hook; renders **all three section headings** |
| The alert banner is **absent** for Low Risk and **present** for High Risk |
| An invalid patient id **redirects to `/`** (two tests: the redirect, and that the home element is then shown) |
| **Each drawer opens from its floating button:** History, SOFA validation, **Camera monitor** (CV), and LLM validation (after the report loads) |

---

## 11. Phase 6 — End-to-End Tests (21 tests)

**Goal:** a real Chromium browser drives the real frontend against the real backend, with no mocks. These test what the clinician actually experiences.

**Requirements:** the ClinIQ backend running on **port 8000**; Playwright starts (or reuses) the Vite server on port 5173. The report tests wait on **real Groq calls** and need a valid `GROQ_API_KEY`.

### `patient_selection.spec.js` (7)
The home page shows 3 patient cards; each shows its risk level and description (post-surgical, pneumonia, septic shock); ClinIQ branding; clicking Patient 1 and Patient 3 each navigates to the right monitor page; the header shows a live clock; the dashboard header shows the patient name.

### `monitoring_flow.spec.js` (8)
Starting from patient 2: the SOFA gauge is visible; the LIVE indicator; the countdown; the vitals grid shows the vital cards; the **History button** opens the drawer; the **Camera button** opens the camera modal; the **SOFA Validation button** opens its drawer; the Exit button returns to patient selection.

### `report_generation.spec.js` (6)
Starting from patient 1: the loading skeleton does not crash the page; **the 4 report sections appear after load** (up to 30 s with a real AI); the reliability badge appears; **the Hospital → Home Care toggle works** (about 30 s, because the other mode is generated); the LLM Validation button is enabled after the report loads; the LLM drawer opens and shows the method count.

**Last full run (2026-11-24):** all 21 passed in 1.4 minutes.

> **A real pitfall met while running these.** On the machine used for this check, **a different project's server was already listening on port 8000** (a "Prism Dashboard API"). It answers `/health` with 200, so a simple health check looks fine, yet `/api/patients` returns 404 and every E2E test fails after a 2-minute timeout in the setup step (the page shows "Unable to load patients — ensure the API server is running on port 8000"). The ClinIQ backend has to be the thing on port 8000, so before running E2E, check that `curl localhost:8000/api/patients` returns the three patients, not just `/health`. See Section 12.

---

## 12. Running the Tests

**All commands run from `icu_monitor/` unless a `cd` is shown.**

### Backend (pytest)
```bash
pytest tests/cliniq/backend                  # all 218 backend tests          (≈ 20 s with the slow ones)
pytest tests/cliniq/backend -m "not slow"    # 208 tests, no real AI calls    (≈ 4–10 s)
pytest tests/cliniq/backend -m slow          # only the 10 real-Groq tests    (≈ 16 s, needs GROQ_API_KEY)
pytest tests/cliniq/backend/unit             # 85 unit tests
pytest tests/cliniq/backend/api              # 124 API tests
pytest tests/cliniq/backend/system           # 9 system tests

pytest tests/cliniq/backend/api/test_api_predict.py -v                              # one file
pytest tests/cliniq/backend/unit/test_ml_service_unit.py::test_alert_threshold_is_7 # one test

pytest tests                                 # whole Python suite: 513 (218 backend + 295 FL)
pytest tests -m "not slow"                   # 472 (skips 41 slow: 10 real-Groq + 31 FL simulation)
```
The API and system tests load the real model files and the real database (read-only in effect: the sliding windows are capped), so `models/` must be populated. The API tests do **not** need a server running; `TestClient` runs the app in-process.

### Frontend unit and integration (Vitest)
**One-time setup.** The frontend test files live in `icu_monitor/tests/`, **outside** `frontend/`, so Node cannot find packages such as `react` and `vitest` from them. Create a symlink once (it is git-ignored):
```bash
cd icu_monitor
ln -s frontend/node_modules node_modules
```
Then:
```bash
cd frontend
npx vitest run                  # all 232 tests in 27 files          (≈ 6 s)
npx vitest                      # watch mode, re-runs on changes
npx vitest run --coverage       # with a coverage report (text + html in frontend/coverage/)
npx vitest run ../tests/cliniq/frontend/unit/components/cv      # one folder
```

### End-to-end (Playwright)
The **same symlink is required**, for the same reason (the specs import `@playwright/test`). Without it every spec fails with "Cannot find module '@playwright/test'".
```bash
# 1. Start the ClinIQ backend on port 8000 (and check it is really ClinIQ):
cd icu_monitor && uvicorn backend.main:app --port 8000
curl localhost:8000/api/patients        # must list 3 patients (not just /health)

# 2. Symlink (once), then run. Playwright starts/reuses the Vite server on 5173:
ln -s frontend/node_modules node_modules
cd frontend
npx playwright test                                   # all 21 (≈ 1.5 min; needs a valid GROQ_API_KEY)
npx playwright test ../tests/cliniq/frontend/e2e/patient_selection.spec.js
npx playwright test --headed                          # watch the browser
npx playwright show-report                            # HTML report
```
**If port 8000 is taken by another program**, ClinIQ cannot use it and the Vite proxy (fixed at `localhost:8000`) will reach the wrong server. Stop the other program, or run ClinIQ elsewhere and point a temporary copy of the Vite config's `/api` proxy at it.

### Recommended workflow
```
While editing a backend helper:        pytest tests/cliniq/backend/unit
After changing an endpoint:            pytest tests/cliniq/backend/api -m "not slow"
After editing a React component:       cd frontend && npx vitest run   (or watch mode)
Before committing:                     pytest tests -m "not slow"  and  npx vitest run
Before a demo or submission:           pytest tests  and  npx vitest run  and  npx playwright test
```

---

## 13. Coverage — Measured, Not Guessed

Earlier versions of this document listed *target* percentages. The numbers below were **measured on 2026-11-24**. They are **line coverage**.

### Backend (the ClinIQ backend and `db.py`, from the 218 backend tests)
| File | Lines | Missed | Coverage |
|---|---|---|---|
| `backend/api/patients.py` | 11 | 0 | **100%** |
| `backend/main.py` | 27 | 1 | **96%** |
| `db.py` | 97 | 7 | **93%** |
| `backend/services/ml_service.py` | 282 | 33 | **88%** |
| `backend/services/llm_service.py` | 279 | 43 | **85%** |
| `backend/api/report.py` | 28 | 5 | **82%** |
| `backend/api/history.py` | 25 | 5 | **80%** |
| `backend/api/vitals.py` | 25 | 5 | **80%** |
| `backend/api/predict.py` | 29 | 8 | **72%** |
| **Total** | **803** | **107** | **87%** |

**Without the 10 slow tests** (no real AI calls) the total is **67%**, because `llm_service.py` falls from **85% to 28%**: its nine-method reliability scoring only runs when a real report is generated. So the slow tests carry most of the AI-report coverage.

### Frontend (from the 232 Vitest tests; `src/main.jsx` is excluded)
**Overall: 79.2% of lines, 73.4% of functions, 72.5% of branches.**

| Coverage | Files |
|---|---|
| **100%** | `AlertBanner`, `ReliabilityBadge`, `ReportToggle`, `ThemeContext`, `useMonitor`, `api/client` |
| **95–99%** | `SHAPPanel`, `SOFAGauge`, `HistoryDrawer`, `VitalsGrid`, `VitalCard`, `HeaderBar`, `SelectPatient` |
| **85–94%** | `ReportContainer` (89%), `TrendsSection` (89%), `MonitorDashboard` (88%), `ThemeToggle` (86%) |
| **70–84%** | `CVMonitorFAB` (84%), `ReportSection` (77%), `TrendChart` (76%) |
| **50–66%** | `LLMValidationFAB`, `SOFAValidationFAB`, `HistoryFAB`, `LLMValidationDrawer` (57%), `SOFAValidationDrawer` (56%), `CVMonitorModal` (51%) |
| **0%** | **`App.jsx`** (the router) and **`components/ErrorBoundary.jsx`** have no unit tests |

**Reading the numbers**
- **`ReportContainer` (89%) and `useMonitor` (100%)** are the logic-heavy parts that decide when to call the AI and when to refresh; they are the best covered, as intended.
- **`CVMonitorModal` (51%)**: the scoring functions are tested directly (Section 9, 4F), but the camera loop, the model loading and the on-screen layout need a real camera and browser, so they are not unit-tested.
- **The two validation drawers (about 56%)**: the tests cover the metrics, the placeholders and closing; much of the file is layout for the other tabs.
- **`App.jsx` and `ErrorBoundary.jsx` at 0%:** the routes are exercised only indirectly (the page tests mount pages directly, and the E2E tests use the real router). `ErrorBoundary` is used in `main.jsx` and is **not tested at all**.
- Coverage of the Python federated-learning code is in `FL_SERVER_CLIENT_DB_TESTING.md` (80%).

---

## 14. Known Gaps and Quirks

Stating limits honestly is part of testing well.

**Gaps (behaviour that is not tested)**
- **The live camera inputs at the API.** No backend test sends `cv_gcs_eye` or `cv_stress` to `/api/predict`, so the 400 errors for out-of-range values ("cv_gcs_eye must be 1–4", "cv_stress must be 0–10") and the "live values replace recorded values" path are not covered by a test. `client.test.js` does not test `runPredict` with camera data either. (An earlier note in `CV_MONITOR.md` implied the API limits are covered; they are not.)
- **`App.jsx` (routes) and `ErrorBoundary.jsx`:** 0% unit coverage (Section 13).
- **No test checks the exact content of a report** beyond structure and length; the AI's wording is not deterministic.
- **The real camera and face model** are never exercised by automated tests (they need a webcam and browser).

**Quirks in the existing tests**
- **Three "source-level" tests read code instead of running it.** `test_generate_report_response_has_judge_used_key`, `test_judge_used_is_set_when_judge_fails` and `test_judge_failure_logged_to_stderr` use `inspect.getsource` and check that certain text appears in `llm_service.py`; the three seed-timestamp tests in `test_ml_service_unit.py` do the same for `ml_service.py`. They guard against a regression in a particular fix, but they would still pass if the text stayed and the behaviour broke. (The live API tests do check `judge_used` at run time.)
- **`test_predict_is_alert_true_for_high_sofa`** only checks that `is_alert` is a boolean (its own comment says "we do not guarantee it"). It does not assert the alert fires for patient 3.
- **The 10 slow report tests send the old request shape** (a `predict_result` body the server now ignores). They pass, but the three "server-side data" tests are the ones that cover the current contract.
- **Some system-test assertions are intentionally loose** (for example `after >= before`), because the sliding windows are capped and already full.
- **`test_api_predict.py` and the system tests write to the real database** (predictions are appended; the sliding windows keep counts constant at 20 and 50 rows per patient). They do not change the model files. The backend unit tests use a temporary database and never touch it.
- **Vitest prints a harmless warning** about a duplicate `borderBottom` key in the two validation drawers' source (`SOFAValidationDrawer.jsx`, `LLMValidationDrawer.jsx`). It is a small code-style issue in the components, not a test failure.

**Environment notes**
- E2E needs the real ClinIQ backend on port 8000 and the `node_modules` symlink (Section 12).
- The 10 slow backend tests and the report E2E tests spend a few Groq API calls each run.

---

## 15. Implementation Status

| Phase | Tests | Status | Location |
|---|---|---|---|
| 1 — Backend unit | 85 | ✅ Passing | `tests/cliniq/backend/unit/` |
| 2 — Backend API | 124 | ✅ Passing (10 slow) | `tests/cliniq/backend/api/` |
| 3 — Backend system | 9 | ✅ Passing | `tests/cliniq/backend/system/` |
| 4 — Frontend unit | 209 | ✅ Passing | `tests/cliniq/frontend/unit/` |
| 5 — Frontend integration | 23 | ✅ Passing | `tests/cliniq/frontend/integration/` |
| 6 — End-to-end | 21 | ✅ Passing | `tests/cliniq/frontend/e2e/` |
| **Total ClinIQ** | **471** | **✅ all passing** | |

Related totals: **Python suite 513** (218 ClinIQ backend + 295 federated-learning tests) · **Vitest 232** (Phases 4 and 5) · **Playwright 21** (Phase 6) · overall **745 automated unit/integration/system tests** (513 + 232) plus the 21 end-to-end tests.

---

*Documentation for the ClinIQ ICU CDSS capstone — last updated 2026-11-24.*
*ClinIQ test suite: 85 backend unit + 124 backend API + 9 backend system + 209 frontend unit + 23 frontend integration + 21 end-to-end = **471 tests, all passing**.*
*Tools actually used: pytest, FastAPI `TestClient`, `monkeypatch`, Vitest, React Testing Library, jsdom, `vi.mock`, Playwright. Not used: pytest-asyncio, httpx.AsyncClient, MSW.*
*Measured coverage: backend 87%, frontend 79% (Section 13).*

# ClinIQ Dashboard — Complete Testing Guide
**What every test does, why it exists, and how the whole suite fits together**

**Last updated:** 2026-11-27
**Result of the last full run (2026-11-27):** backend **390 passed** · frontend (Vitest) **664 passed** in 53 files · end-to-end (Playwright) **33 tests** (31 passed in the full run, 2 slow report tests timed out under load and passed when re-run alone; Section 11) → **1,087 ClinIQ tests**.
**Measured coverage:** backend **99%** of statements · frontend **100%** of statements, **96.7%** of branches, **100%** of functions, and every one of the 29 source files is at or above 90% on every measure (Section 13).

> **Scope:** the ClinIQ React dashboard and its FastAPI backend: everything under `icu_monitor/tests/cliniq/`.
> **Not covered here:** federated learning (Flower, `train_federated.py`, the Streamlit simulator). Those are in `FL_SERVER_CLIENT_DB_TESTING.md`.
> **How this document was checked:** every count, test name and number below comes from the real test files and a real run on 2026-11-27, not from memory. Where the suite has a gap or a quirk, it is stated plainly (Section 14).

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
6. [Phase 1 — Backend Unit Tests (237 tests)](#6-phase-1--backend-unit-tests-237-tests)
7. [Phase 2 — Backend API Tests (144 tests)](#7-phase-2--backend-api-tests-144-tests)
8. [Phase 3 — Backend System Tests (9 tests)](#8-phase-3--backend-system-tests-9-tests)
9. [Phase 4 — Frontend Unit Tests (591 tests)](#9-phase-4--frontend-unit-tests-591-tests)
10. [Phase 5 — Frontend Integration Tests (73 tests)](#10-phase-5--frontend-integration-tests-73-tests)
11. [Phase 6 — End-to-End Tests (33 tests)](#11-phase-6--end-to-end-tests-33-tests)
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
| **Unit** | One function or one component in isolation; everything external is replaced by a stand-in ("mock") | Phases 1 and 4 | 237 + 591 = 828 |
| **Integration** | Several parts together. Backend: the API router, database and ML code, called over HTTP. Frontend: a whole page with its API calls mocked | Phases 2 and 5 | 144 + 73 = 217 |
| **System** | A multi-step clinical session through the real backend ("select a patient → predict → vitals → history") | Phase 3 | 9 |
| **End-to-end** | A real browser, a real frontend and a real backend, no mocks | Phase 6 | 33 |

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
├── backend/                                   ← 390 tests (pytest)
│   ├── conftest.py                            ← 4 shared fixtures
│   ├── unit/                                  ← 237 tests
│   │   ├── test_db_unit.py                    ← 17  database helpers (temporary database)
│   │   ├── test_ml_service_unit.py            ← 43  ML helper functions (no model loaded)
│   │   ├── test_llm_service_unit.py           ← 25  report cache, concurrency, judge_used
│   │   ├── test_llm_service_branches.py       ← 66  AI report logic with a FAKE Groq client   (added 2026-11-27)
│   │   └── test_ml_service_branches.py        ← 86  conformal interval, SHAP shapes, run_prediction on a temp DB  (added)
│   ├── api/                                   ← 144 tests
│   │   ├── test_api_health.py                 ←  4  GET /health
│   │   ├── test_api_patients.py               ← 26  GET /api/patients and /{id}
│   │   ├── test_api_predict.py                ← 39  POST /api/predict
│   │   ├── test_api_vitals.py                 ← 20  GET /api/vitals/{id}
│   │   ├── test_api_history.py                ← 17  GET /api/history/{id}
│   │   ├── test_api_report.py                 ← 18  POST /api/report  (10 are slow: real Groq calls)
│   │   └── test_api_error_paths.py            ← 20  bad camera values, service failures, missing key  (added)
│   └── system/
│       └── test_system_flows.py               ←  9  multi-step backend workflows
│
└── frontend/                                  ← 664 + 33 tests
    ├── unit/                                  ← 591 tests (Vitest), 48 files
    │   ├── App.test.jsx (4), ErrorBoundary.test.jsx (5)                       ← both were 0% before
    │   ├── components/
    │   │   ├── FloatingButtons.test.jsx (12), FloatingButtonsHover.test.jsx (9)
    │   │   ├── layout/        AlertBanner (9), HeaderBar (12), HeaderBarStates (18),
    │   │   │                  ThemeToggle (7), ThemeToggleBehaviour (4)
    │   │   ├── overview/      SOFAGauge (8), SOFAGaugeBranches (14), VitalCard (14),
    │   │   │                  VitalCardVariants (27), VitalsGrid (6), VitalsGridStatus (28)
    │   │   ├── report/        ReliabilityBadge (8), ReliabilityAndShap (14), ReportContainer (22),
    │   │   │                  ReportContainerBranches (15), ReportSection (5),
    │   │   │                  ReportSectionRendering (25), ReportToggle (7), SHAPPanel (9)
    │   │   ├── trends/        TrendChart (5), TrendChartBranches (16), TrendChartTooltip (7),
    │   │   │                  TrendsSection (6), TrendsSectionBranches (5)
    │   │   ├── validation/    LLMValidationDrawer (8), LLMValidationDrawerTabs (26),
    │   │   │                  LLMValidationFAB (4), SOFAValidationDrawer (13),
    │   │   │                  SOFAValidationDrawerTabs (28), SOFAValidationFAB (3)
    │   │   ├── cv/            CVMonitorFAB (4), CVMonitorModal (6), CVMonitorModalFlow (35),
    │   │   │                  CVLivePanel (33), cvScoring (16)
    │   │   └── history/       HistoryDrawer (8), HistoryDrawerRows (25), HistoryFAB (5)
    │   ├── hooks/             useMonitor (11), useMonitorEdges (15)
    │   ├── contexts/          ThemeContext (5), ThemeFallbacks (9)
    │   └── api/               client (8), clientCameraFields (8)
    ├── integration/           ← 73 tests (Vitest), 5 files
    │   ├── SelectPatient.test.jsx (7), SelectPatientBranches.test.jsx (21)
    │   └── MonitorDashboard.test.jsx (16), MonitorDashboardStates.test.jsx (20),
    │       MonitorDashboardDrawers.test.jsx (9)
    └── e2e/                   ← 33 tests (Playwright)
        ├── patient_selection.spec.js   (7)
        ├── monitoring_flow.spec.js     (8)
        ├── report_generation.spec.js   (6)
        └── cv_live_panel.spec.js       (12)   ← camera window and the live panel  (added)
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
- `-m "not slow"` skips the **83 slow tests**: the 10 ClinIQ tests that call the real Groq API, plus 73 federated-learning tests that run a full simulation or system flow. (The text of the `slow` marker above only mentions the LLM, but the FL tests use it too.)

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

## 6. Phase 1 — Backend Unit Tests (237 tests)

**Goal:** test helper functions and database helpers in isolation: no network, no external service, and no real AI call. `test_db_unit.py` has 17 tests, `test_ml_service_unit.py` 43, `test_llm_service_unit.py` 25, and (added 2026-11-27) `test_llm_service_branches.py` 66 and `test_ml_service_branches.py` 86.

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


### 1D — `test_llm_service_branches.py` (66 tests, added 2026-11-27)
These cover the AI-report code that the earlier tests only reached through the slow, real-Groq tests. **No network and no `GROQ_API_KEY` are needed**: a small fake stands in for the Groq client and returns whatever text each test scripts (including errors).

| Group | Tests | What is checked and why |
|---|---|---|
| `<think>` stripping (4) | A closed block is removed; an **unclosed** block is cut (the model ran out of tokens while reasoning); a **stray closing tag** drops everything before it; plain text is only trimmed | Some models print their reasoning. It must never reach a clinician |
| Section parsing (3) | Finds all four headings; accepts markdown and numbering (`## 1. **CURRENT CONDITION**`); returns nothing when there are no headings | The frontend needs exactly four sections |
| Prompt building (5) | Hospital alert starts with `CLINICAL ALERT`; home-care alert with `PATIENT ALERT`; no alert text below the threshold; the threshold is **inclusive** (7.0 fires); the right system message per mode | The alert wording is what a clinician reads first |
| **Methods 1–7, one group each (about 33)** | Factual grounding, clinical hard rules, SHAP coherence, response structure, severity calibration, contraindication and numeric accuracy: each is checked at its edges, for example "SpO₂ below 90 must mention oxygen", "an NSAID is always flagged", "a made-up heart rate of 187 is reported as a hallucinated number, and a repeated number is counted once", "a standard threshold such as MAP 65 is not treated as a measurement" | These seven checks are the deterministic part of the 9-method reliability score (`docs/VALIDATION.md`) |
| Judge call (4) | Parses the judge's JSON even when it is wrapped in chatter; swallows an exception and returns nothing; returns nothing when there is no JSON; sends only the **first 900 characters** at temperature 0 | The judge must never block or crash the report |
| Reliability score (6) | Works with no judge (neutral 0.5 defaults); a perfect judge scores higher; unsupported claims lower the score and **only the first two are listed**; a "dangerous recommendation" zeroes the contraindication part and adds an `AI Judge:` flag; hallucinated numbers are listed | Locks in how the score is combined |
| Label thresholds (6) | 0.90 and 0.75 are High; 0.74 and 0.55 are Moderate; 0.54 and 0 are Low | The label edges match the dashboard |
| `generate_report` end to end (9) | Success path (writer at temperature 0.15, then judge at 0); **cached** for identical input (second call makes no AI call); an LLM failure returns an error and is **not cached**; empty text uses the fallback message; a failed judge falls back to neutral; missing sections are filled with empty strings; a bad report sets `caution`; no API key skips the judge | The whole report pipeline, with a fake Groq |

### 1E — `test_ml_service_branches.py` (86 tests, added 2026-11-27)
These run the real prediction code against a **temporary database** (the real `icu_monitor.db` is never read or written) and replace only the slow or random parts.

| Group | Tests | What is checked and why |
|---|---|---|
| `_safe_float` (8) | `None`, text, `NaN`, `inf` and a list all give the default; numeric text converts | A bad value in the database must not crash a prediction |
| Risk label, vital status, SOFA floor (about 26) | Band edges (4.99 Low, 5.0 Moderate, 9.99 Moderate, 10.0 High); low, normal and high vitals; each SOFA floor contribution (MAP, SpO₂, GCS eye 1 and 2) | These feed the colours and the "validation" checks |
| Consistency flags (5) | A shock pattern (HR over 130, SBP under 90, SOFA under 4) and multi-organ stress (SpO₂ under 88, MAP under 65, SOFA under 6) are flagged; neither is flagged when SOFA is already high | The Runtime Checks tab shows these |
| Trajectory (7) | History unavailable; first reading; duplicate rows count as first reading; a **big rise or fall with no vital-sign change is flagged**; the same jump with a real vital change is not; a small change is not | A sudden unexplained SOFA jump means "check the input" |
| SHAP wording (about 23) | Every sentence the system can write for a feature (low or elevated heart rate, fever, hypothermia, impaired consciousness, high or low stress, and so on), the direction words, and a clinical-note word | The plain-English explanations clinicians read |
| Cached loaders (3) | No SHAP explainer when there is no background file; the training-metadata fallback numbers; reading the file when present | |
| **Conformal interval (6)** | Falls back to **2.9** when the training table is missing, when there are fewer than 50 samples, or on any error; otherwise it is the **90th percentile of the absolute errors**; empty or unlabelled hospitals are skipped; missing feature columns are zero-filled | This is the ±3.28 range on the dashboard, and the reason a fresh clone shows ±2.9 |
| **`run_prediction` on a temporary database (about 10)** | No sample rows raises a clear error; the vitals window is **seeded** when empty; row numbers wrap; **dataset camera values are used by default and live camera values override them**; the live camera value **changes the model input** (the SOFA differs); the prediction is logged with the camera inputs; the result has all expected fields | This is the test of your question "do the camera values go into the model?": yes |
| SHAP result shapes (5) | A list, a 2-D array and a 1-D array are all accepted; a SHAP failure or no explainer leaves the prediction working with empty explanations | SHAP versions return different shapes |
| Missing feature column (1) | A column the pipeline never builds is zero-filled | |
| Start-up path (5) | Each backend module adds the project root to the path when it is missing | The app must start however it is launched |

---

## 7. Phase 2 — Backend API Tests (144 tests)

**Goal:** call each endpoint over HTTP with FastAPI's `TestClient`. The real routers, database and ML code all run. These tests need the model files and the database to exist. (The new `test_api_error_paths.py` is the exception: it replaces the service behind each route with a small fake so that failures can be forced.)

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


### 2G — `test_api_error_paths.py` (20 tests, added 2026-11-27)
These reach the branches that only run when something goes wrong. Nothing here calls the real model, database writer or Groq.

| Group | Tests | What is checked |
|---|---|---|
| **Bad camera values (8)** | `cv_gcs_eye` of 0, 5, −1 or 99 → **400** ("cv_gcs_eye must be 1–4"); `cv_stress` of −0.1, 10.1, 11 or −5 → **400** ("cv_stress must be 0–10") | This closes the gap noted in Section 14: the camera input limits are now tested |
| Unknown patient (1) | Patient 9 → 400 | |
| Service failures (2) | A service `ValueError` becomes **400** with its message; any other exception becomes **500** with only "Prediction failed", **never the internal detail** | A stack trace must not leak to the browser |
| Camera values reach the service (1) | `cv_gcs_eye=3, cv_stress=6.5` are passed to the prediction code unchanged | |
| Response clean-up (2) | The four LLM prompt fields are stripped from the response, and the full result (with them) is stored for `/api/report` | |
| History and vitals failures (2) | A database error → **500** "query failed", without the internal message | |
| **Report errors (4)** | No `GROQ_API_KEY` → 500; a report failure → 500 without the detail; success returns the service result; the report is built from the **server's own stored prediction**, ignoring a `predict_result` the client sends | |

A fixture saves and restores the shared "last prediction" store around every test, so these fakes cannot leak into other tests.

> **What this section does not cover.** The slow tests still send the old request shape (Section 14).

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

## 9. Phase 4 — Frontend Unit Tests (591 tests)

**Goal:** test each React component, hook and helper in isolation with `render()`; the animation, chart and camera libraries are mocked by `vitest-setup.js`. The original 209 tests are described first in each group; the **382 tests added on 2026-11-27** follow, marked *added*.

### 4A — Layout (50)
- **AlertBanner (9):** renders nothing when `predictData` is null or the risk is Low; renders for Moderate and High; shows the SOFA score; label **ELEVATED RISK** for Moderate and **CRITICAL ALERT** for High; `aria-live` is assertive for High and polite for Moderate (so screen readers announce urgency correctly).
- **HeaderBar (12):** renders; the LIVE indicator; the countdown in seconds; "···" while refreshing; the patient name; the Exit button; the ClinIQ brand mark; the last-updated time; "connection lost" on error; "All Systems Operational" otherwise; no timestamp when `lastUpdated` is null; the theme toggle is an accessible button.
- **ThemeToggle (7):** is a button with an `aria-label`; shows a text label unless `compact`; clicking toggles the theme; sun icon in dark mode, moon in light.
- **HeaderBarStates (18, added):** the LIVE chip shows the countdown, shows "···" while refreshing, shows **"Connection lost"** on an error and keeps showing LIVE while a retry is in flight; the patient pill shows name, initial (or "P" for an empty name), age and gender; no pill before the patient loads and no risk badge before a prediction; the last-updated time appears only when known; **Exit** returns to the patient list and highlights on hover; the top accent strip is neutral for Low Risk and risk-coloured otherwise.
- **ThemeToggleBehaviour (4, added):** hover highlights and restores; click flips the label and `aria-label`; `title` and `aria-label` always agree; compact mode has tighter padding.

### 4B — Overview (97)
- **SOFAGauge (8):** renders; shows the score and risk level; has the SVG arc; handles score 0 and score 24 without crashing; shows the conformal interval values; draws **3 zone-marker lines** at SOFA 6, 9 and 12.
- **VitalCard (14):** label, value and unit; a null value shows "—" without crashing; status labels **Normal**, **Above Normal**, **Below Normal**; up-arrow for high and down-arrow for low; the trend section.
- **VitalsGrid (6):** renders with valid vitals; shows HR and SpO₂; handles null vitals and undefined `predictData`; renders all 7 vital card labels (including DBP).
- **SOFAGaugeBranches (14, added):** the **interval bar** positions the score inside the 90% range, caps at 100%, and is empty for missing, empty or inverted bounds; the glow ring appears only for Moderate and High risk; the score **animates from the previous value**, restarts from 0 on a patient switch, and pauses on unmount; the arc fills straight away for users who prefer reduced motion.
- **VitalCardVariants (27, added):** standard and wide cards; status labels; decimals to one place; a dash for a missing value; the **icon animation chosen per vital** (heartbeat, drop sway, breathe, float, blood-pressure nudges, spin) and that **abnormal values animate faster**; an unknown status is treated as normal.
- **VitalsGridStatus (28, added):** the two cards the grid works out itself. **GCS eye** shows None / To Pain / To Voice / Spontaneous and highlights scores 1 and 2. **Stress** is Low below 3.5, Elevated from 3.5, High from 6.5; the 10-dot scale; a dash when missing. The clinical-note card shows the note or a fallback; API statuses pass through; an empty prediction does not crash.

### 4C — Report (105)
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
- **ReliabilityAndShap (14, added):** the badge wording for High / Moderate / Low and an unknown label, the rounded percentage, a missing score shown as 0, the caution line only when asked; the SHAP panel's labels, signed impacts, values with and without a unit, the legend, empty input, and bar animation.
- **ReportContainerBranches (15, added):** the previous patient's report is **cleared at once** when the patient changes; a report-level error screen with the rate-limit wording for "429", "rate_limit" and "Rate limit"; the error wording for a failure with a body, a plain `Error` and nothing at all; **a hospital report is never shown under the home-care tab**; the badge appears only when a score exists. **A soft failure (a reply with no sections) keeps the last good report on screen** (fixed 2026-11-27; it used to replace it with the "unavailable" screen). It still shows the "unavailable" screen when there is no earlier report to keep, and it never keeps the *other* care mode's report.
- **ReportSectionRendering (25, added):** how AI text becomes UI: `Heading:` lines become small upper-case sub-headings (not when long, and not when they are bullets); `•`, `-` and `*` bullets lose their marker; `**bold**` becomes `<strong>` (several per line, an unmatched `**` is left alone); a `⚡` line becomes a highlighted callout; blank lines are skipped; empty content shows "No assessment available."; every section has its icon; the card lifts on hover.

### 4D — Trends (39)
- **TrendChart (5):** renders with data; the chart container; handles an empty array and null; renders for SpO₂.
- **TrendsSection (6):** renders; shows skeleton cards while loading; after data loads shows **7 charts** and all 7 vital labels.
- **TrendChartBranches (16, added):** the real props (`dataKey`, `label`, `unit`, `color`, `normalLo`, `normalHi`; the older test passes a `vital` prop the component ignores); the latest value as an integer or to one decimal; the **abnormal flag** (↑ above, ↓ below, none at the range edges or when only an earlier reading was abnormal); the tinted border; empty, null, single-reading and timestamp-less data.
- **TrendChartTooltip (7, added):** the hover tooltip is captured from the element handed to the chart and rendered directly (the chart library is stubbed): time label, value to one decimal with the unit, a dash for no value, nothing when inactive or empty, the chart colour.
- **TrendsSectionBranches (5, added):** seven skeletons while loading, an error message when the request fails, seven labelled charts on success, the right patient and limit (20), and a null body.

### 4E — Validation (82)
- **LLMValidationDrawer (8):** hidden when closed; shown when open; **"9 / 9 methods ran"** when the judge was used and **"7 / 9 methods ran"** with a warning when it was not; no crash when `reportData` is null; the three tabs (Overview, 9 Methods, Findings); the close button.
- **LLMValidationFAB (4):** the `aria-label`; clicks work when data exists; it is visually disabled (low opacity, no pointer events) when there is no report yet, and full opacity otherwise.
- **SOFAValidationDrawer (13)** checks the drawer reads **live** from the training metadata (`_training_meta`) and never shows hard-coded numbers: MAE, R², AUROC and PR-AUC come from the metadata; the **3-class accuracy is computed from the confusion matrix**; the test and total sample counts; the live alert threshold and alert recall; the High-risk recall from `per_class`; the confusion-matrix rows; **placeholders ("—") instead of a crash when the metadata is missing**; the close button.
- **SOFAValidationFAB (3):** the `aria-label`; clicking; renders.
- **SOFAValidationDrawerTabs (28, added):** the header, the three tabs, Escape and ✕ to close, the first tab again on reopen. **Runtime Checks:** the lower-bound check lists every floor contribution and flags a prediction below the floor; the consistency check flags the shock and multi-organ patterns and stays quiet when SOFA is already high; the trajectory check shows "First reading", ↑ and ↓ changes, and "Verify input" for a flagged jump. **Uncertainty:** the interval, the ±q̂ margin, the real MAE comparison, the 0–24 scale, and the **±2.9 fallback clamped to 0 and 24** when the backend sent no interval.
- **LLMValidationDrawerTabs (26, added):** the header badge, care mode, singular and plural violation counts, the warning dot on the Findings tab, Escape and ✕; **Overview** (the three explanations, thresholds, nine weight segments adding to 100); **9 Methods** (all nine named with their weights and a hint per score band); **Findings** (the all-clear message, each violation listed, the trust table and the disclaimer).

### 4F — Camera monitor (94)
- **CVMonitorFAB (4):** the `aria-label`; clicking; renders when active; the active state renders different HTML.
- **CVMonitorModal (6):** hidden when closed, shown when open; the supplementary label; the footer says live readings feed the next SOFA prediction; no score cards before the camera runs; **live readings are cleared (`null` sent) when the monitor closes**.
- **cvScoring (16)** tests the exported scoring maths directly (no camera needed):
  - **PSPI (4):** a neutral face gives 0; the formula adds correctly (example 7.5); the maximum is 16; a full smile cancels the score;
  - **stress (4):** the first frame seeds the average; maximal pain and agitation give 10; smoothing uses α = 0.15; the bands LOW < 3.5 ≤ MODERATE < 6.5 ≤ HIGH (checked at 3.4, 3.5, 6.4, 6.5);
  - **eye aspect ratio (1)** and **eye state (2):** the blink value overrides the ratio; open, partial and closed thresholds;
  - **GCS window (5):** E4 at ≥ 70% open (blinks allowed); E3 at ≥ 30% open or partly open; E2 for a flicker of at least 0.1 s; E1 when eyes stay closed; a one-frame twitch is not a flicker.

  Details of the algorithm: `docs/CV_MONITOR.md`.

- **CVMonitorModalFlow (35, added):** drives the real detection loop with a **fake camera, a fake face detector and a fake clock**, so nothing needs a webcam or the 12 MB model.
  - **Start-up (9):** the loading messages in order; `NotAllowedError` and `NotFoundError` become friendly messages; a download failure explains "internet connection required"; any other message passes through; Retry starts again.
  - **Closing (6):** Escape, ✕ and the backdrop close it; the camera and the model are released and live readings cleared on unmount; closing during start-up never starts the camera, and a stream that arrives late is released.
  - **Calibration (6):** progress while calibrating; "centre your face" when none is seen; **nothing is sent to the dashboard during calibration**; duplicate frames are skipped; a frame where detection throws does not stop the loop.
  - **After calibration (8):** the stress and GCS cards appear; **nothing is sent until the 10-second observation window is full**, then `{stress, gcs_eye}` is; open eyes give E4, closed eyes give E1 (and E1 is what is sent); a strained face raises stress; an unchanged reading is not resent; the default thresholds are used if the eyes never open enough to calibrate.
  - **Face lost (3):** the warning appears; after 3 seconds without a face the reading is cleared (`null`); a reading never sent is never cleared.
  - **Live panel (2):** the panel receives the prediction, countdown and camera values.
- **CVLivePanel (33, added):** the new right-hand panel. **Output:** SOFA, risk level, 90% range, ALERT. **Timer:** `m:ss`, "Predicting now", the progress bar (it spans 90 → 1, so it is (90 − c) / 89), clamping. **Source line:** "At 0 → DATABASE values" with no camera reading and "At 0 → CAMERA values" with one. **Camera inputs:** WAITING FOR CAMERA, **IN MODEL ✓** when the live value equals what the last SOFA used, **SENDS NEXT CYCLE** when it differs, each feature decided on its own. **Note and vitals:** the note text or the empty message, and all seven vitals.

### 4G — History (38)
- **HistoryDrawer (8):** no table when closed; renders when open; the column headers (including Timestamp and Risk); the close button; a SOFA of 5.5 shows as "5.5"; an empty history shows an empty-state message.
- **HistoryFAB (5):** the `aria-label`; clicking; renders with `recordCount` 0 and above 0; has an SVG icon.
- **HistoryDrawerRows (25, added):** loading, error, empty and null states and the recording count; the request is for the 50 latest rows of the right patient and not made while closed; rows are **newest first**; every risk badge; dashes for missing values; **↑ and ↓ marks** outside the normal range; row highlight on hover; closing from the backdrop and ✕. **Escape closes the drawer** (added 2026-11-27, like the other drawers); it ignores other keys, does not listen while closed, and stops listening once unmounted.

### 4H — `useMonitor` hook (26)
This hook runs the 90-second polling loop, so these tests protect the live behaviour:
calls `runPredict` on mount with the patient and row 0; sets `predictData` after a successful fetch; sets `error` on a failed fetch; `isRefreshing` is true during a fetch; the countdown **resets to 90** after a fetch; changing the patient resets the data and triggers a new fetch; `lastUpdated` is set; the countdown **decrements** with the timer; the row index goes to 1 on the second fetch; **the concurrent-fetch guard prevents overlapping calls**; **the React StrictMode double-invoke guard prevents a duplicate first fetch**.

**Added (`useMonitorEdges`, 15):** the camera values passed at call time reach the request, and the **newest** camera values are used for the next request, not those from the first render; the three error wordings; the previous reading is kept when a later refresh fails and the error clears after a good one; **the row resumes from `sessionStorage`**, is saved after each success, is not advanced on a failure, and is separate per patient; a reply for a different patient is ignored; the countdown ticks and fires a new request at 1; the timer stops on unmount.

### 4I — `ThemeContext` (14)
Provides the theme and `toggleTheme`; toggling flips the theme and twice returns to the original; sets `data-theme` on the document element.

**Added (`ThemeFallbacks`, 9):** which theme wins at start-up (**saved choice > operating-system preference > light**, with a garbage saved value ignored); the choice is saved and `data-theme` set on every change; it still works when **reading or writing storage throws** (private windows); a harmless default with no provider.

### 4J — API client (16)
With `axios` mocked, checks each call: `getPatients` → `GET /patients`; `getPatient(2)` → `/patients/2`; `runPredict(1, 5)` → `POST /predict` with `{patient_id: 1, row_idx: 5}`; **`generateReport` sends only `patient_id` and `care_mode`**; `getHistory(1, 5)` → `/history/1?limit=5`; `getVitals(1, 20)` → `/vitals/1?limit=20`; the default limits are 50 (history) and 20 (vitals).

**Added (`clientCameraFields`, 8):** the camera fields are sent **only when the camera has a value**, because the backend treats a missing field as "use the stored database value": none or `null` sends just patient and row; both, GCS only, and stress only are each sent correctly; **a stress of exactly 0 is sent** (zero is a real reading, not "missing"); `undefined` and `null` values are not sent. This closes the gap noted earlier.

### 4K — App and ErrorBoundary (9, added)
Both files were at 0% before. **App (4):** `/` shows patient selection; `/monitor/2` shows the dashboard and passes the id; an unknown URL and `/monitor` with no id redirect to `/`. **ErrorBoundary (5):** children render normally; a throwing child shows "Something went wrong" and is no longer rendered; the error is logged with the `[ClinIQ]` prefix; **Reload** calls `window.location.reload`.

### 4L — Floating buttons (21, added)
`FloatingButtons` (12): the four buttons each render one accessible button with an icon; clicks work; the camera button shows its live dot and blinks faster only while active; the LLM button is dimmed with pointer events off until a report exists; and the **stacking order from the bottom is History 28, Camera 88, LLM 148, SOFA 208**. `FloatingButtonsHover` (9) uses the **real** animation library (the global setup stubs it) to prove each button **expands to show its label on hover** ("Patient History" with the record count, "Live CV Monitor" or "CV Live — Click to Close", "AI Validation" or "Awaiting Report…", "SOFA Validation") and collapses again.

---

## 10. Phase 5 — Frontend Integration Tests (73 tests)

**Goal:** render a **whole page** with the real components, but with its API calls replaced by mocks (`vi.mock` on `@/api/client.js`), inside a real `MemoryRouter`, `QueryClientProvider` and `ThemeProvider`. (There is no MSW; Section 4.)

### `SelectPatient.test.jsx` (7)
Calls `getPatients` on mount; renders 3 patient cards after load; shows the descriptions; **clicking a card navigates to the monitor page**; shows an error state when the API fails; shows ClinIQ branding; shows no cards while loading.

### `SelectPatientBranches.test.jsx` (21, added)
**Theme:** the toggle flips between dark and light and the page background follows. **Cards:** Enter on a focused card opens that patient, other keys do not; cards are keyboard-focusable; the "Open Monitoring Dashboard" button navigates without firing the card click twice; hovering raises the card's shadow and the button fills; risk, SOFA range and bay are shown. **Status bar and clock:** the patient count (3 while loading, the real count afterwards); the clock ticks every second and stops on unmount. **Reduced motion:** the intro animation plays when motion is allowed and is skipped when it is reduced. **Loading and errors:** skeleton cards, the API-error banner, and an empty list.

### `MonitorDashboard.test.jsx` (16)
`getPatient`/`getVitals`/`getHistory`/`generateReport` are mocked, and **`useMonitor` is mocked** so each test controls the prediction data directly.

| What is checked |
|---|
| Renders; fetches patient data on mount; shows the SOFA score and risk level from `predictData`; shows the patient name and the HR value in the vitals grid; shows the countdown from the hook; renders **all three section headings** |
| The alert banner is **absent** for Low Risk and **present** for High Risk |
| An invalid patient id **redirects to `/`** (two tests: the redirect, and that the home element is then shown) |
| **Each drawer opens from its floating button:** History, SOFA validation, **Camera monitor** (CV), and LLM validation (after the report loads) |

### `MonitorDashboardStates.test.jsx` (20, added)
**First load:** the loading skeleton, no floating buttons, and the trends section still shown. **Refresh:** the refresh-error banner keeps the last reading; the status area dims while refreshing. **Patient strip:** identity, risk and SOFA; the reliability shows "—" until the report arrives and then a percentage; not drawn before the patient record loads; Low Risk does not pulse and Moderate does. **Care mode:** switching to home care requests a home-care report. **Invalid patient:** ids 0, 4, abc and −1 redirect, with no server request. **Camera hand-off:** the page passes `null` camera data to the monitor hook until the camera reports, and opening the camera window shows the live panel with the prediction and the countdown.

### `MonitorDashboardDrawers.test.jsx` (9, added)
Opens and closes the SOFA validation drawer, the LLM validation drawer (with its care mode) and the history drawer from the page; the ✕ buttons turn red on hover; the camera window opens, closes from its ✕ and from the camera button. These reach the open and close handlers wired in the page itself.

> **A lesson from writing these.** The page re-mounts parts of its tree while the patient and report requests settle, so a button found early can be a detached node by the time it is clicked. The tests therefore wait until the report is on screen and look the button up again at click time.

---

## 11. Phase 6 — End-to-End Tests (33 tests)

**Goal:** a real Chromium browser drives the real frontend against the real backend, with no mocks. These test what the clinician actually experiences.

**Requirements:** the ClinIQ backend running on **port 8000**; Playwright starts (or reuses) the Vite server on port 5173. The report tests wait on **real Groq calls** and need a valid `GROQ_API_KEY`.

### `patient_selection.spec.js` (7)
The home page shows 3 patient cards; each shows its risk level and description (post-surgical, pneumonia, septic shock); ClinIQ branding; clicking Patient 1 and Patient 3 each navigates to the right monitor page; the header shows a live clock; the dashboard header shows the patient name.

### `monitoring_flow.spec.js` (8)
Starting from patient 2: the SOFA gauge is visible; the LIVE indicator; the countdown; the vitals grid shows the vital cards; the **History button** opens the drawer; the **Camera button** opens the camera modal; the **SOFA Validation button** opens its drawer; the Exit button returns to patient selection.

### `report_generation.spec.js` (6)
Starting from patient 1: the loading skeleton does not crash the page; **the 4 report sections appear after load** (up to 30 s with a real AI); the reliability badge appears; **the Hospital → Home Care toggle works** (about 30 s, because the other mode is generated); the LLM Validation button is enabled after the report loads; the LLM drawer opens and shows the method count.

### `cv_live_panel.spec.js` (12, added 2026-11-27)
Chromium is given a **synthetic webcam** and the MediaPipe model download is answered locally, so no real camera and no internet are needed. The face detector therefore never reports a face, which is exactly the state under test.

**Camera window and live panel (7):** it opens with the panel on its right; the panel shows the predicted SOFA, the two camera inputs (stress and GCS eye), the clinical note and the vitals; **with no camera reading it says "At 0 → DATABASE values go into the model" and WAITING FOR CAMERA**; the timer shows the dashboard countdown; Escape closes it; clicking the backdrop closes it; and **the floating camera button is covered while the window is open** (a real user cannot click it again, so the window is closed with Escape, ✕ or the backdrop).

**Other overlays (5):** the history drawer opens; the SOFA validation drawer shows its three tabs and switches between them (lower-bound check, 90% interval); Escape closes it; an invalid patient id and an unknown URL both redirect to the selection page.

**Last full run (2026-11-27):** the 12 new tests passed in 38 seconds. In the **full 33-test run**, 31 passed and the 2 slowest report tests (`report section shows 4 content cards after load` and `reliability badge appears with report`) **timed out at 30 seconds** because the run had just made many real Groq calls in a row; **both passed when re-run on their own** (all 6 report tests passed in 30 seconds). Earlier, on 2026-11-24, all 21 original tests passed together in 1.4 minutes. Treat these two as sensitive to Groq's speed, not as a defect.

> **A real pitfall met while running these.** On the machine used for this check, **a different project's server was already listening on port 8000** (a "Prism Dashboard API"). It answers `/health` with 200, so a simple health check looks fine, yet `/api/patients` returns 404 and every E2E test fails after a 2-minute timeout in the setup step (the page shows "Unable to load patients — ensure the API server is running on port 8000"). The ClinIQ backend has to be the thing on port 8000, so before running E2E, check that `curl localhost:8000/api/patients` returns the three patients, not just `/health`. See Section 12.

---

## 12. Running the Tests

**All commands run from `icu_monitor/` unless a `cd` is shown.**

### Backend (pytest)
```bash
pytest tests/cliniq/backend                  # all 390 backend tests          (≈ 15 s with the slow ones)
pytest tests/cliniq/backend -m "not slow"    # 380 tests, no real AI calls    (≈ 8–15 s)
pytest tests/cliniq/backend -m slow          # only the 10 real-Groq tests    (≈ 16 s, needs GROQ_API_KEY)
pytest tests/cliniq/backend/unit             # 237 unit tests
pytest tests/cliniq/backend/api              # 144 API tests
pytest tests/cliniq/backend/system           # 9 system tests

pytest tests/cliniq/backend/api/test_api_predict.py -v                              # one file
pytest tests/cliniq/backend/unit/test_ml_service_unit.py::test_alert_threshold_is_7 # one test

pytest tests                                 # whole Python suite: 750 (390 backend + 360 FL)   (≈ 2 min)
pytest tests -m "not slow"                   # 667 (skips 83 slow: 10 real-Groq + 73 FL simulation)  (≈ 1.7 min)
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
npx vitest run                  # all 664 tests in 53 files          (≈ 16 s)
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
npx playwright test                                   # all 33 (≈ 3–4 min; needs a valid GROQ_API_KEY)
npx playwright test ../tests/cliniq/frontend/e2e/patient_selection.spec.js
npx playwright test cv_live_panel                     # only the camera-window tests (no Groq key needed)
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

Earlier versions of this document listed *target* percentages, and the first measured version (2026-11-24) showed real gaps: backend **87%**, frontend **79%**, with `App.jsx` and `ErrorBoundary.jsx` at 0%. The numbers below were **measured again on 2026-11-27 after the gaps were closed**.

> **How it was measured.** Python: `coverage run -m pytest tests` (statement coverage, with `source = .` and `tests/`, `frontend/` and `setup.py` left out). Frontend: `vitest run --coverage` with the v8 provider. Both tools were run from a temporary location, so nothing was added to the project. To measure again yourself, install `coverage` (Python) and `@vitest/coverage-v8` (frontend) first; neither is in the requirements.

### Backend and scripts (Python)
Whole suite, **750 tests**, statement coverage:

| File | Statements | Missed | Coverage |
|---|---|---|---|
| `backend/api/history.py`, `patients.py`, `predict.py`, `report.py`, `vitals.py` | 118 | 0 | **100%** each |
| `backend/main.py` | 27 | 0 | **100%** |
| `backend/services/llm_service.py` | 279 | 0 | **100%** |
| `backend/services/ml_service.py` | 282 | 0 | **100%** |
| `db.py` | 97 | 0 | **100%** |
| `model_utils.py` | 159 | 0 | **100%** |
| `migrate_to_db.py` | 158 | 0 | **100%** |
| `client.py`, `server.py` | 134 | 0 | **100%** |
| `train_federated.py` | 400 | 4 | **99%** (lines 214, 561–563) |
| `fl_dashboard.py` | 458 | 13 | **97%** (357, 378–384, 418, 426, 467, 496, 1532) |
| **Total** | **2,112** | **17** | **99%** |

**What the 17 missed lines are:** 13 in the Streamlit simulator (a zero-fill of missing feature columns, some empty-data branches in its data-loading helper, and the `sleep(1)` that ends a running page, which the tests deliberately interrupt) and 4 in the training script (a "note is not text" guard in the text cleaner and a fallback inside the non-IID branch). None needs a real connection; they are edge branches the synthetic data never reaches, and not worth faking.

**The slow tests matter.** With the 83 slow tests skipped (`-m "not slow"`) the whole-suite total falls from **99% to 85%**, because the Streamlit simulator and the training script's full path (`fl_dashboard.py` 62%, `train_federated.py` 65%) are only exercised by slow tests. For the **ClinIQ backend files** alone, skipping the 10 real-Groq tests changes nothing now: `llm_service.py` is **100% either way**, because the new `test_llm_service_branches.py` tests the report code with a fake Groq client. (Before 2026-11-27 it fell from 85% to 28% without the slow tests, so the slow tests carried the AI-report coverage.)

Coverage of the federated-learning code in detail: `FL_SERVER_CLIENT_DB_TESTING.md`.

### Frontend (from the 664 Vitest tests; `src/main.jsx` is excluded)
**Overall: 100% of statements, 96.7% of branches, 100% of functions** (29 source files).

**Every one of the 29 source files is at or above 90% on statements, branches and functions, and every file is at 100% of statements and functions.** The lowest branch figures are `CVMonitorModal` (94.1%), `HeaderBar` (94.2%), `ReportContainer` (94.4%), `SHAPPanel` (94.4%), `SelectPatient` (93.5%) and `MonitorDashboard` (95.3%). `App.jsx`, `ErrorBoundary.jsx`, `api/client.js`, `ThemeContext`, `VitalsGrid`, `ReportSection`, `TrendChart`, `TrendsSection`, `LLMValidationDrawer` and all four floating buttons are at **100%** of statements, branches and functions.

| Before (2026-11-24) | After (2026-11-27) |
|---|---|
| `App.jsx` **0%**, `ErrorBoundary.jsx` **0%** | **100%** |
| `CVMonitorModal` 51% | **99.8%** |
| `SOFAValidationDrawer` 56%, `LLMValidationDrawer` 57% | **100%** of statements (branches 97.5% and 100%) |
| `HistoryFAB`, `SOFAValidationFAB`, `LLMValidationFAB` 58–66% | **100%** |
| `TrendChart` 76%, `ReportSection` 77% | **100%** |
| `MonitorDashboard` 88%, `ReportContainer` 89% | **100%** of statements |
| Total: 79.2% lines, 72.5% branches, 73.4% functions | **100% / 96.7% / 100%** (statements / branches / functions) |

**Dead code removed (2026-11-27).** `MethodBar` (in `LLMValidationDrawer.jsx`) and `Eyebrow` (in `SOFAValidationDrawer.jsx`) were defined but never used anywhere, so no test could reach them and they held those two drawers at 90.2% and 97.9% of lines. They have been deleted; both drawers are now at 100% of statements.

**What coverage does not prove.** 99% means the tests *run* nearly every line, not that they check every result. The camera window is tested with a **fake** camera and a **fake** face detector, so the real MediaPipe model and a real webcam are still checked only by hand and by the E2E spec's synthetic stream (Section 14).

### End-to-end
Playwright has no code-coverage figure here, so the 33 browser tests are not part of the percentages above.

---

## 14. Known Gaps and Quirks

Stating limits honestly is part of testing well.

**Gaps (behaviour that is not tested)**
- **No test checks the exact content of a report** beyond structure and length; the AI's wording is not deterministic. (The new tests check the *logic* around a report with a fake AI: the reliability scoring, caching and error handling.)
- **The real camera and face model** are never exercised by automated tests. The unit tests use a fake camera and fake detector, and the E2E spec uses a synthetic webcam with the model download stubbed, so the real MediaPipe model and a real face are checked only by hand.
- **The 17 uncovered Python lines** (Section 13) are edge branches in the training script and the Streamlit simulator that the synthetic test data never reaches.

**Gaps that were closed on 2026-11-27** (listed in the earlier version of this document)
- *The live camera inputs at the API:* `test_api_error_paths.py` now sends out-of-range `cv_gcs_eye` and `cv_stress` (both → 400), and `test_ml_service_branches.py` proves live camera values **replace** the stored ones and **change the model input**. `clientCameraFields.test.js` covers `runPredict` with camera data.
- *`App.jsx` and `ErrorBoundary.jsx` at 0%:* both are now tested (Section 9, 4K).

**Things found while writing the new tests, and what was done** (each was recorded by a test when found, and the first four were **fixed on 2026-11-27**; the tests now check the fixed behaviour)
- **A soft report failure replaced the last good report. Fixed.** A comment in `ReportContainer.jsx` says a reply with no sections "keeps the last good report visible", but the page showed "AI Assessment Unavailable" because the newest reply was preferred. `ReportContainer` now prefers a reply that has content, then the last good report **of the same care mode**, and only then the failed reply. The "unavailable" screen still appears when there is nothing earlier to keep, and one care mode's report is never shown under the other tab. (`ReportContainerBranches`, 3 tests.)
- **The history drawer did not close on Escape. Fixed.** The SOFA validation drawer, the LLM validation drawer and the camera window all did. `HistoryDrawer` now listens for Escape while open and stops when closed or unmounted. (`HistoryDrawerRows`, 4 tests.)
- **Dead code. Fixed.** `MethodBar` and `Eyebrow` were deleted (Section 13).
- **`setup.py` printed the wrong dashboard address. Fixed.** It printed `http://localhost:5174`; Vite's default, and the address in the Playwright config and the backend's allowed origins, is **5173**. It now prints 5173. (`test_script_entrypoints.py`, 2 tests, which also check that `setup.py`, `playwright.config.js` and `backend/main.py` agree.)
- **The camera window covers its own button.** While it is open, its backdrop sits over the floating camera button, so the button cannot be clicked again; close it with Escape, ✕ or the backdrop. This is how the window is designed to work, so it was left as it is. (`cv_live_panel.spec.js`.)
- **The old `TrendChart.test.jsx` passes a `vital` prop** that the component does not have (the real props are `dataKey`, `label`, `unit`, `color`, `normalLo`, `normalHi`). Those five old tests still pass, but they check little. `TrendChartBranches.test.jsx` uses the real props.

**Quirks in the existing tests**
- **Three "source-level" tests read code instead of running it.** `test_generate_report_response_has_judge_used_key`, `test_judge_used_is_set_when_judge_fails` and `test_judge_failure_logged_to_stderr` use `inspect.getsource` and check that certain text appears in `llm_service.py`; the three seed-timestamp tests in `test_ml_service_unit.py` do the same for `ml_service.py`. They guard against a regression in a particular fix, but they would still pass if the text stayed and the behaviour broke. (The live API tests do check `judge_used` at run time.)
- **`test_predict_is_alert_true_for_high_sofa`** only checks that `is_alert` is a boolean (its own comment says "we do not guarantee it"). It does not assert the alert fires for patient 3.
- **The 10 slow report tests send the old request shape** (a `predict_result` body the server now ignores). They pass, but the three "server-side data" tests are the ones that cover the current contract.
- **Some system-test assertions are intentionally loose** (for example `after >= before`), because the sliding windows are capped and already full.
- **`test_api_predict.py` and the system tests write to the real database** (predictions are appended; the sliding windows keep counts constant at 20 and 50 rows per patient). They do not change the model files. The backend unit tests use a temporary database and never touch it.
- **Vitest prints a harmless warning** about a duplicate `borderBottom` key in the two validation drawers' source (`SOFAValidationDrawer.jsx`, `LLMValidationDrawer.jsx`). It is a small code-style issue in the components, not a test failure.
- **`SOFAValidationDrawer.test.jsx` still uses numbers from the previous model run** (MAE 1.8247, AUROC 0.9254 and so on) as its test data. They are fixtures that the test feeds in and reads back, so the test passes regardless of the real model; they are not a claim about the current model (MAE 1.8006, AUROC 0.9181).
- **The new tests that write to a database or to `models/`** (`test_ml_service_branches.py`, `test_migrate_to_db.py`, `test_script_entrypoints.py`, `test_train_federated_script.py`, `test_fl_dashboard_app.py`) always redirect to a temporary file or folder; the real `icu_monitor.db` and the trained model files are never written by them.

**Environment notes**
- E2E needs the real ClinIQ backend on port 8000 and the `node_modules` symlink (Section 12).
- The 10 slow backend tests and the report E2E tests spend a few Groq API calls each run.
- **Two report E2E tests can time out in a full run.** They wait up to 30 seconds for a real Groq reply; after many calls in a row Groq can be slower than that. They pass when re-run alone (Section 11).
- **`test_fl_dashboard_app.py` swaps in the real `streamlit` for its own module.** Three other FL test files replace `sys.modules["streamlit"]` with a mock when pytest collects them; this file puts the real library back for its own tests and restores the previous state afterwards, so the order of files does not matter.

---

## 15. Implementation Status

| Phase | Tests | Status | Location |
|---|---|---|---|
| 1 — Backend unit | 237 | ✅ Passing | `tests/cliniq/backend/unit/` |
| 2 — Backend API | 144 | ✅ Passing (10 slow) | `tests/cliniq/backend/api/` |
| 3 — Backend system | 9 | ✅ Passing | `tests/cliniq/backend/system/` |
| 4 — Frontend unit | 591 | ✅ Passing | `tests/cliniq/frontend/unit/` |
| 5 — Frontend integration | 73 | ✅ Passing | `tests/cliniq/frontend/integration/` |
| 6 — End-to-end | 33 | ✅ 31 in the full run; the other 2 pass when re-run alone | `tests/cliniq/frontend/e2e/` |
| **Total ClinIQ** | **1,087** | | |

**Overall totals (2026-11-27)**

| Group | Count |
|---|---|
| Python suite (`pytest tests`) | **750** = 390 ClinIQ backend + 360 federated-learning (83 marked slow) |
| Frontend (Vitest) | **664** in 53 files |
| **Automated unit, integration and system tests** | **1,414** (750 + 664) |
| Browser end-to-end (Playwright) | **33** |
| **Everything** | **1,447** |

Added on 2026-11-27: **+237 Python tests** (513 → 750), **+432 Vitest tests** (232 → 664) and **+12 Playwright tests** (21 → 33), **+681 in all**.

---

*Documentation for the ClinIQ ICU CDSS capstone — last updated 2026-11-27.*
*ClinIQ test suite: 237 backend unit + 144 backend API + 9 backend system + 591 frontend unit + 73 frontend integration + 33 end-to-end = **1,087 tests**.*
*Tools actually used: pytest, FastAPI `TestClient`, `monkeypatch`, Vitest, React Testing Library, jsdom, `vi.mock`, Playwright. Not used: pytest-asyncio, httpx.AsyncClient, MSW.*
*Measured coverage: Python 99% of statements (85% without the slow tests); frontend 100% of statements, 96.7% of branches, 100% of functions (Section 13).*

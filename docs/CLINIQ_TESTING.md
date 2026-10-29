# ClinIQ Dashboard — Complete Testing Implementation Guide
**What every test does, why it exists, and how the whole suite fits together**

> **Scope:** Everything inside `icu_monitor/` related to the ClinIQ React dashboard and its FastAPI backend
> **Not in scope (yet):** FL dashboard, Flower server/client, Streamlit app
> **All test files live in:** `icu_monitor/tests/cliniq/`

---

## Table of Contents

1. [Why Testing Matters — The First Principles](#1-why-testing-matters--the-first-principles)
2. [The Testing Pyramid — Four Layers, Four Purposes](#2-the-testing-pyramid--four-layers-four-purposes)
3. [Directory Layout](#3-directory-layout)
4. [Tool Stack — What Each Tool Does and Why We Chose It](#4-tool-stack--what-each-tool-does-and-why-we-chose-it)
5. [Setup and Configuration Files](#5-setup-and-configuration-files)
6. [Phase 1 — Backend Unit Tests](#6-phase-1--backend-unit-tests)
7. [Phase 2 — Backend API Tests (Integration)](#7-phase-2--backend-api-tests-integration)
8. [Phase 3 — Backend System Tests](#8-phase-3--backend-system-tests)
9. [Phase 4 — Frontend Unit Tests](#9-phase-4--frontend-unit-tests)
10. [Phase 5 — Frontend Integration Tests](#10-phase-5--frontend-integration-tests)
11. [Phase 6 — End-to-End Tests (Playwright)](#11-phase-6--end-to-end-tests-playwright)
12. [Running the Tests](#12-running-the-tests)
13. [Coverage Targets and What They Mean](#13-coverage-targets-and-what-they-mean)
14. [Implementation Status](#14-implementation-status)

---

## 1. Why Testing Matters — The First Principles

Before looking at any specific test, it helps to understand the fundamental problem testing solves.

**The core problem: confidence without testing is false confidence.**

Imagine you write a function that calculates SOFA risk classification:
```python
def classify_risk(sofa_score):
    if sofa_score < 5:
        return "Low Risk"
    if sofa_score < 10:
        return "Moderate Risk"
    return "High Risk"
```

You run the dashboard manually, enter a few values, it looks fine. Three weeks later, a teammate refactors the function to fix a formatting issue and accidentally changes `< 10` to `<= 10`:
```python
    if sofa_score <= 10:   # off-by-one error introduced
        return "Moderate Risk"
```

Now a patient with SOFA = 10 gets "Moderate Risk" instead of "High Risk." There is no error message, no crash — the dashboard looks identical. The bug silently gives wrong clinical guidance.

**A test catches this immediately:**
```python
def test_sofa_10_is_high_risk():
    assert classify_risk(10) == "High Risk"    # ← fails the moment the bug is introduced
```

The test suite turns silent regressions into loud, immediate failures. This is especially important in a medical decision-support system where wrong output has clinical consequences.

**Three specific problems testing solves in this project:**

1. **Regression prevention:** Every feature that works today should still work after tomorrow's changes. Tests lock in correct behaviour.

2. **Interface contracts:** When the backend and frontend were developed independently, tests define exactly what each side expects. The `/api/predict` test specifies that the response must include `sofa_score`, `risk_level`, `vitals`, and `shap` — if the backend developer renames a field, the API test fails before the frontend developer even tries to integrate.

3. **Edge case documentation:** Tests like "patient_id=-1 → 400" and "SOFA=24 renders without crash" document the system's behaviour at boundaries. Without tests, these edge cases are forgotten and only discovered when a real user triggers them.

---

## 2. The Testing Pyramid — Four Layers, Four Purposes

The tests in this project are organised into four layers, each serving a different purpose:

```
                    ┌──────────────┐
                    │  E2E Tests   │  ← Tests the real app in a real browser
                    │  (Phase 6)   │    Slowest, most realistic, fewest tests
                    └──────┬───────┘
                   ┌───────┴────────┐
                   │  Integration   │  ← Tests components working together
                   │  (Phase 2,5)   │    Medium speed, medium scope
                   └───────┬────────┘
               ┌───────────┴────────────┐
               │     Unit Tests         │  ← Tests individual functions in isolation
               │    (Phase 1, 4)        │    Fastest, most numerous, most specific
               └────────────────────────┘
```

**Layer 1 — Unit Tests (Phases 1 and 4):**
- Test one function or one component at a time
- Everything external (databases, APIs, network, animation libraries) is replaced with fake versions (called "mocks")
- When a unit test fails, you know EXACTLY which function broke — there is nowhere else to look
- Run in milliseconds — you can run them hundreds of times per day

**Layer 2 — Integration Tests (Phases 2 and 5):**
- Test multiple components working together
- Backend integration: the FastAPI router + the database + the business logic all running together, called via HTTP
- Frontend integration: a full page component + its API calls (API calls are mocked, but the component is real)
- When these fail, the bug is in how components communicate — the interface between two parts

**Layer 3 — System Tests (Phase 3):**
- Test complete multi-step workflows, like a real clinical session
- "Patient selects → predict → vitals populate → history updates → data is consistent"
- When these fail, the workflow logic is broken even if individual components work

**Layer 4 — E2E Tests (Phase 6):**
- A real browser, a real backend, a real frontend — no mocks at all
- Tests what the clinician actually experiences
- Slowest (seconds per test) but most realistic
- Used for the most important user journeys only

**Why we need all four layers:**

Each layer catches different bugs:
```
A typo in classify_risk()           → Caught by Unit Tests
API field renamed "sofa" → "score"  → Caught by Integration Tests
Row cycling breaks after 50 reads   → Caught by System Tests
Report tab doesn't render in Chrome → Caught by E2E Tests
```

No single layer catches everything. Running all four gives comprehensive confidence.

---

## 3. Directory Layout

```
icu_monitor/
├── tests/
│   ├── cliniq/
│   │   ├── backend/
│   │   │   ├── conftest.py                  ← Shared fixtures for all backend tests
│   │   │   ├── unit/
│   │   │   │   ├── test_db_unit.py          ← Database helper tests (in-memory SQLite)
│   │   │   │   ├── test_ml_service_unit.py  ← ML pure-function tests (no model load)
│   │   │   │   └── test_llm_service_unit.py ← LLM cache key and judge_used tests
│   │   │   ├── api/
│   │   │   │   ├── test_api_health.py       ← GET /health
│   │   │   │   ├── test_api_patients.py     ← GET /api/patients and /api/patients/{id}
│   │   │   │   ├── test_api_predict.py      ← POST /api/predict
│   │   │   │   ├── test_api_vitals.py       ← GET /api/vitals/{patient_id}
│   │   │   │   ├── test_api_history.py      ← GET /api/history/{patient_id}
│   │   │   │   └── test_api_report.py       ← POST /api/report
│   │   │   └── system/
│   │   │       └── test_system_flows.py     ← End-to-end backend workflows
│   │   └── frontend/
│   │       ├── setup.js                     ← Global test setup and mocks
│   │       ├── unit/
│   │       │   ├── components/
│   │       │   │   ├── layout/              ← AlertBanner, HeaderBar, ThemeToggle
│   │       │   │   ├── overview/            ← SOFAGauge, VitalCard, VitalsGrid
│   │       │   │   ├── report/              ← ReportContainer, ReliabilityBadge,
│   │       │   │   │                           ReportSection, ReportToggle, SHAPPanel
│   │       │   │   ├── trends/              ← TrendChart, TrendsSection
│   │       │   │   ├── validation/          ← LLMValidationFAB, LLMValidationDrawer,
│   │       │   │   │                           SOFAValidationFAB, SOFAValidationDrawer
│   │       │   │   ├── cv/                  ← CVMonitorFAB, CVMonitorModal
│   │       │   │   └── history/             ← HistoryFAB, HistoryDrawer
│   │       │   ├── hooks/
│   │       │   │   └── useMonitor.test.js   ← Core data-fetching hook
│   │       │   ├── contexts/
│   │       │   │   └── ThemeContext.test.jsx ← Dark/light mode context
│   │       │   └── api/
│   │       │       └── client.test.js       ← API call functions
│   │       ├── integration/
│   │       │   ├── SelectPatient.test.jsx   ← Patient selection page
│   │       │   └── MonitorDashboard.test.jsx ← Main monitoring dashboard
│   │       └── e2e/
│   │           ├── patient_selection.spec.js  ← Patient selection user flow
│   │           ├── monitoring_flow.spec.js    ← Dashboard monitoring flow
│   │           └── report_generation.spec.js ← AI report generation flow
└── fl/
    └── .gitkeep    ← Empty placeholder for future federated learning tests
```

**Why is `conftest.py` special?**

`conftest.py` is a pytest convention — it is automatically discovered by pytest and its contents are made available to all test files in the same directory and all subdirectories. Rather than copying the same setup code into every test file, `conftest.py` defines shared "fixtures" (reusable test objects) once. The `TestClient` and `predict_p1` fixtures defined there are available in every test file under `backend/` without any import statement.

---

## 4. Tool Stack — What Each Tool Does and Why We Chose It

### pytest (Backend Testing)

pytest is Python's most widely used testing framework. It discovers test files automatically (any file named `test_*.py`), runs functions named `test_*`, and reports failures with a clear, readable diff.

**What makes pytest powerful:**

```python
# Without pytest: you write this boilerplate yourself
try:
    result = classify_risk(10)
    assert result == "High Risk"
    print("PASS")
except AssertionError:
    print("FAIL: got", result)

# With pytest: just write the assertion
def test_sofa_10_is_high_risk():
    assert classify_risk(10) == "High Risk"
# pytest handles discovery, running, reporting, and exit codes automatically
```

**Fixtures — the most important pytest concept:**

A fixture is a reusable piece of test setup. Instead of creating a database connection, a test client, and sample data in every test function, you define them once as fixtures and request them by name:

```python
# In conftest.py — defined once
@pytest.fixture
def client():
    # Create a test HTTP client connected to our FastAPI app
    with TestClient(app) as tc:
        yield tc  # "yield" means: this object is available during the test

# In test_api_health.py — used by name in the function signature
def test_health_returns_200(client):          # pytest injects the client fixture
    response = client.get("/health")
    assert response.status_code == 200
```

The `predict_p1` fixture specifically does a POST to `/api/predict` for Patient 1 before the test starts, so tests that need a prediction result already have one ready without repeating the predict call.

### pytest-asyncio and httpx

FastAPI routes can be `async` functions. The standard Python `requests` library cannot call async endpoints. `httpx.AsyncClient` can — and `pytest-asyncio` allows writing `async def test_*` functions. Together they enable testing async FastAPI endpoints.

### Vitest (Frontend Unit and Integration Testing)

Vitest is the frontend equivalent of pytest — a test runner for JavaScript and React. It integrates directly with Vite (the build tool used by the frontend), so it understands the same import aliases, module system, and configuration.

```js
// A Vitest frontend test looks just like a pytest backend test
import { render, screen } from '@testing-library/react'
import VitalCard from '@/components/VitalCard'

test('shows vital value and unit', () => {
    render(<VitalCard label="HR" value={122} unit="bpm" status="high" />)
    expect(screen.getByText('122')).toBeInTheDocument()
    expect(screen.getByText('bpm')).toBeInTheDocument()
})
```

**Why Vitest instead of Jest?**

Jest is the traditional React testing framework but has a known friction point with Vite-based projects — the module systems and configurations differ, requiring significant translation. Vitest was designed specifically for Vite and uses identical configuration, making setup and maintenance much simpler.

### React Testing Library (@testing-library/react)

React Testing Library provides the `render()` function and the `screen` query object. These simulate rendering a React component in a test environment and let you query what was rendered.

**The philosophy of React Testing Library:**

Test what the user sees, not internal implementation details.

```js
// Bad test (implementation-focused — breaks when you refactor):
expect(component.state.isHigh).toBe(true)  // tests internal state

// Good test (user-focused — tests what the user actually sees):
expect(screen.getByText('CRITICAL ALERT')).toBeInTheDocument()
expect(screen.getByRole('alert')).toBeVisible()
```

If the component shows the right text to the user, the test passes — regardless of whether the implementation uses `useState`, `useReducer`, or a third-party state manager. Tests written this way survive refactors.

### jsdom (The Virtual Browser)

Frontend tests run in Node.js, not a real browser. jsdom is a JavaScript implementation of the browser's DOM (Document Object Model) — it simulates `document`, `window`, `HTMLElement`, and all the APIs that React uses to render. Vitest is configured with `environment: 'jsdom'`, which means every test file runs as if it is inside a browser, even though it is running in Node.js on the command line.

**What jsdom cannot do:** Render visual layouts with actual CSS, run WebGL (used by some chart libraries), or fully replicate browser-specific quirks. For those, you need E2E tests.

### vi.mock — The Mocking System

`vi.mock` replaces a module with a fake version for the duration of the test. This is essential because real libraries cannot run in jsdom:

```js
// framer-motion requires browser APIs that jsdom doesn't have
// Without mocking: test crashes with "ResizeObserver is not defined"

vi.mock('framer-motion', () => ({
    motion: {
        div: ({ children, ...props }) => <div {...props}>{children}</div>,
        // Replace animated div with a plain div — same structure, no animations
    },
    AnimatePresence: ({ children }) => children,
}))

// Now framer-motion components render as plain HTML — tests work
```

Other mocked libraries:
- **animejs**: Uses `requestAnimationFrame` — not in jsdom
- **recharts**: SVG rendering requires browser layout engine — not in jsdom
- **@mediapipe/tasks-vision**: Uses WebAssembly and GPU APIs — not in jsdom

The mocks let us test the React component logic (does it render the right text, does it call the right function on click?) without needing a real browser.

### MSW (Mock Service Worker) — Frontend Integration Test API Mocking

MSW is used in integration tests to intercept HTTP requests and return fake responses without hitting the real backend. This means integration tests can run without the backend server running.

```js
// Without MSW: test calls real backend → slow, unreliable, needs server running
// With MSW: test intercepts the call and returns controlled fake data

const server = setupServer(
    rest.get('/api/patients', (req, res, ctx) =>
        res(ctx.json([
            { id: 1, name: "Patient 1", risk_level: "Low Risk" },
            { id: 2, name: "Patient 2", risk_level: "Moderate Risk" },
            { id: 3, name: "Patient 3", risk_level: "High Risk" },
        ]))
    )
)
// Now SelectPatient.test.jsx can test that it renders 3 cards,
// without needing the backend to be running
```

### Playwright (E2E Testing)

Playwright is a browser automation tool. It controls a real Chrome, Firefox, or Safari browser — clicking buttons, typing text, waiting for elements to appear — exactly as a user would.

```js
// Playwright test: a real browser clicks through the real app
test('clicking Patient 1 navigates to dashboard', async ({ page }) => {
    await page.goto('http://localhost:5173')          // real browser opens real URL
    await page.click('text=Patient 1')               // real click on real element
    await expect(page).toHaveURL('/monitor/1')       // real URL change verified
    await expect(page.locator('h1')).toContainText('Patient 1')  // real DOM checked
})
```

E2E tests require both the backend (`localhost:8000`) and frontend (`localhost:5173`) to be running. They are the slowest but most complete tests.

---

## 5. Setup and Configuration Files

### `pytest.ini` (at `icu_monitor/` root)

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

**What each line does:**

- `testpaths = tests` — pytest only looks inside the `tests/` directory. Without this, pytest might try to collect test files from `src/`, `models/`, or other directories unnecessarily.
- `python_files = test_*.py` — only files starting with `test_` are treated as test files
- `markers` — custom labels you can use to select or deselect specific test groups:
  ```bash
  pytest -m "not slow"    # skip tests that call the real Groq API (saves time and API credits)
  pytest -m unit          # run only fast, pure function tests
  pytest -m api           # run only the HTTP endpoint tests
  ```

### `frontend/vitest.config.js`

```js
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
    plugins: [react()],          // Enable JSX and React-specific transforms
    test: {
        globals: true,           // Makes test/expect/describe/vi available without imports
        environment: 'jsdom',    // Simulate browser DOM in Node.js
        setupFiles: [path.resolve(__dirname, '../tests/cliniq/frontend/setup.js')],
        // Run setup.js before every test file (loads jest-dom matchers, mocks)
        include: [path.resolve(__dirname, '../tests/cliniq/frontend/**/*.test.{js,jsx}')],
        // Only run files matching *.test.js or *.test.jsx inside the cliniq/frontend/ directory
        coverage: {
            provider: 'v8',      // V8 is Node.js's built-in coverage engine — fast and accurate
            reporter: ['text', 'html'],  // Show in terminal AND generate browsable HTML report
            include: ['src/**/*.{js,jsx}'],  // Only measure coverage of production code
        },
    },
    resolve: {
        alias: { '@': path.resolve(__dirname, './src') },
        // Allows: import VitalCard from '@/components/VitalCard'
        // instead of: import VitalCard from '../../components/VitalCard'
    },
})
```

### `frontend/tests/cliniq/frontend/setup.js`

This file runs before every test file. It loads the jest-dom matchers (which add assertions like `toBeInTheDocument()`, `toBeVisible()`, `toHaveTextContent()`) and sets up global mocks that are needed by almost every component:

```js
import '@testing-library/jest-dom'  // Loads custom matchers

// Mock framer-motion (animation library — not compatible with jsdom)
vi.mock('framer-motion', () => ({ ... }))

// Mock animejs (animation library — uses requestAnimationFrame)
vi.mock('animejs', () => ({ ... }))

// Mock AOS (scroll animation library)
vi.mock('aos', () => ({ ... }))

// Mock recharts (chart library — SVG layout needs real browser)
vi.mock('recharts', () => ({ ... }))

// Mock @mediapipe/tasks-vision (computer vision — needs WebAssembly + GPU)
vi.mock('@mediapipe/tasks-vision', () => ({ ... }))
```

By placing these in `setup.js` rather than in each test file, we avoid repeating 50 lines of mock code in every component test.

### `frontend/package.json` scripts

```json
"test":          "vitest run",           // Run all tests once and exit
"test:watch":    "vitest",               // Run tests and re-run on file changes (dev mode)
"test:coverage": "vitest run --coverage", // Run tests + generate coverage report
"test:e2e":      "playwright test"        // Run Playwright E2E tests
```

---

## 6. Phase 1 — Backend Unit Tests

**Goal:** Test every pure function and database helper in complete isolation — no real network calls, no ML model loading, no external services.

**What "pure function" means:** A function that takes inputs, does computation, and returns output — with no side effects (no file writes, no network calls, no randomness). Pure functions are the easiest to test because the same inputs always produce the same output.

**Runner:** `pytest tests/cliniq/backend/unit/ -v`

---

### 1A — `test_db_unit.py` — Database Helper Tests

**The setup: a temporary in-memory SQLite database**

```python
@pytest.fixture
def test_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    init_db(conn)    # Run the schema creation (same function the real app uses)
    yield conn
    conn.close()
```

Every test gets its own fresh, empty database. `tmp_path` is a pytest built-in fixture that provides a temporary directory that is automatically deleted after the test. This means:
- Tests cannot interfere with each other through shared database state
- Tests cannot accidentally modify the real production database
- Tests always start from a clean, known state

**The 14 tests and what each verifies:**

| # | Test | What it verifies | Why it matters |
|---|---|---|---|
| 1 | `init_db()` creates `patient_vitals` table | Schema bootstrap | Without this table, vital sign storage fails |
| 2 | `init_db()` creates `prediction_history` table | Schema bootstrap | Without this table, prediction logging fails |
| 3 | `init_db()` creates `sample_patients` table | Schema bootstrap | Without this table, patient data cannot be seeded |
| 4 | `init_db()` is idempotent (called twice, no error) | `IF NOT EXISTS` guard | The app calls `init_db()` on every startup — it must not crash if tables already exist |
| 5 | `append_patient_vitals()` inserts a row correctly | CRUD correctness | Verifies the INSERT SQL is correct and data is stored |
| 6 | `get_patient_vitals()` returns rows ascending by id | Ordering | If rows come back in random order, trend charts show scrambled history |
| 7 | Sliding window — after 21 inserts, exactly 20 remain | DELETE logic | The 20-reading window must stay at 20; if this fails, the window grows unbounded |
| 8 | Cross-patient isolation — patient 1 row not in patient 2 | WHERE patient_id | Critical safety: patient 1's data must never contaminate patient 2's view |
| 9 | `append_prediction()` inserts a row correctly | CRUD correctness | The prediction log must actually save |
| 10 | `get_prediction_history()` limit respected | SELECT LIMIT | The dashboard requests 5 most recent predictions — limit must be enforced |
| 11 | Prediction sliding window — after 51 inserts, exactly 50 remain | DELETE logic | Same as vital sliding window but for prediction history |
| 12 | `get_sample_patient_count()` returns 0 on empty table | COUNT query | Used to decide whether to seed sample patients at startup |
| 13 | `get_sample_patient_row()` returns None for missing idx | Null safety | Used when cycling through patient data — must not crash on missing rows |
| 14 | `get_conn()` sets WAL journal mode | SQLite PRAGMA | WAL (Write-Ahead Logging) allows concurrent reads and writes — without it, the dashboard would hang when reading and writing simultaneously |

**Example — understanding the sliding window test:**

```python
def test_sliding_window_keeps_last_20(test_db):
    # Insert 21 vital sign rows for patient 1
    for i in range(21):
        append_patient_vitals(test_db, patient_id=1, hr=60+i, rr=15, ...)
    
    # Fetch all rows for patient 1
    rows = get_patient_vitals(test_db, patient_id=1)
    
    # Exactly 20 must remain — the oldest one was deleted
    assert len(rows) == 20
    
    # The oldest row (hr=60) must be gone; the newest (hr=80) must be present
    hr_values = [row['hr'] for row in rows]
    assert 60 not in hr_values   # row 0 (oldest) was dropped
    assert 80 in hr_values       # row 20 (newest) survived
```

**Why test cross-patient isolation (test #8) specifically?**

In a real ICU, the system monitors multiple patients simultaneously. If a bug caused patient 1's vitals to appear in patient 2's trend chart, clinicians would make decisions based on the wrong patient's data — a potentially fatal error. The isolation test is a safety test:

```python
def test_cross_patient_isolation(test_db):
    append_patient_vitals(test_db, patient_id=1, hr=90, ...)
    append_patient_vitals(test_db, patient_id=2, hr=130, ...)
    
    p1_rows = get_patient_vitals(test_db, patient_id=1)
    p2_rows = get_patient_vitals(test_db, patient_id=2)
    
    # Patient 1 should NOT see patient 2's HR=130
    assert all(row['hr'] != 130 for row in p1_rows)
    
    # Patient 2 should NOT see patient 1's HR=90
    assert all(row['hr'] != 90 for row in p2_rows)
```

---

### 1B — `test_ml_service_unit.py` — ML Pure Function Tests

These tests check helper functions inside `ml_service.py` without loading the ML model itself. Loading the model requires `models/federated_model.pth`, `models/scaler.pkl`, and `models/tfidf_vectorizer.pkl` — expensive file operations that would slow down every test. The pure functions are extracted and tested in isolation.

**The most important function tested: `_is_clinical(feature_name)`**

This function determines whether a feature name is clinically meaningful — it is used when filtering SHAP explanations to show only understandable clinical factors (like "SpO₂" or "creatinine") rather than internal feature names (like "feature_82" or "weight").

The function has a subtle bug risk: substring matching. The vital abbreviation "RR" (respiratory rate) is a 2-letter string. If we check `"rr" in feature_name`, then:
- `"crrt"` (continuous renal replacement therapy) would incorrectly match "rr"
- `"cirrhosis"` (liver disease) would incorrectly match "rr"

Tests #5, #6, and #7 specifically catch this class of substring traps:

| # | Test | Why it's a trap | Why it matters |
|---|---|---|---|
| 5 | `_is_clinical('epinephrine')` → False for 'hr' | 'hr' is inside 'epinephrine' | 'epinephrine' should not be treated as a heart-rate feature |
| 6 | `_is_clinical('cirrhosis')` → False for 'rr' | 'rr' is inside 'cirrhosis' | Cirrhosis is a TF-IDF text feature, not a respiratory rate feature |
| 7 | `_is_clinical('crrt')` → False for 'rr' | 'rr' is inside 'crrt' | CRRT is renal therapy — must not match respiratory abbreviation |
| 8 | `_is_clinical('latest_hr')` → True (word boundary match) | 'hr' appears but is correct | Should match because 'hr' is a complete word segment |

The correct implementation uses a word-boundary regex check:
```python
import re
def _is_clinical(name):
    lower = name.lower()
    for abbr in _VITAL_ABBR:
        # Check with non-letter boundaries — prevents substring traps
        if re.search(r'(?<![a-z])' + re.escape(abbr) + r'(?![a-z])', lower):
            return True
    return name in CLINICAL_KEYWORDS
```

**Training metadata tests (#10–12):**

```
load_training_metadata()['final_mae']       == 1.8968
load_training_metadata()['final_r2']        == 0.5995
load_training_metadata()['test_samples']    == 2559
```

These specific numbers come from the actual federated training run. The tests lock in these values so that:
1. If anyone accidentally overwrites `training_metadata.json` with wrong values, the test fails immediately
2. The values displayed in the SOFA Validation Drawer are guaranteed to match what the model actually achieved
3. The test documents the model's verified performance for the capstone panel

**What these numbers mean clinically:**
- `final_mae = 1.8968`: On average, the model's SOFA predictions are off by 1.90 points. Since SOFA is scored 0–24, this is a 7.9% average error — acceptable for a decision-support tool.
- `final_r2 = 0.5995`: The model explains 60% of the variance in SOFA scores. This means patient-to-patient variation in SOFA is 60% predictable from the input features.
- `test_samples = 2559`: The performance metrics were computed on 2,559 patients held out from training — this is the validation set size that gives confidence the metrics are not inflated.

**Seed timestamp test (#13):**

```python
def test_seed_timestamps_not_hardcoded_2026_08_29(test_db):
    timestamps = get_seed_timestamps(test_db)
    for ts in timestamps:
        assert '2026-08-29' not in ts
```

This test exists because during development, seed data was initially seeded with a hardcoded date (`2026-08-29`). When the capstone is presented in late October 2026, a history table showing "last reading: 2 months ago" would look wrong to the panel. The seed timestamps must be relative to the current date. This test ensures that hardcoded date never reappears during a refactor.

---

### 1C — `test_llm_service_unit.py` — LLM Cache Key Tests

The LLM service implements caching: if the same patient has the same vitals, SOFA, and notes as a previous request, the system returns the cached response rather than making another expensive API call to Groq. The cache key is a string that uniquely identifies a combination of inputs.

**The C2 fix — why patient_id must be in the cache key:**

A critical bug that was fixed: the original cache key did not include `patient_id`. This meant if Patient 1 (stable) and Patient 3 (critical) happened to have similar SOFA scores and some matching vitals, they would get the same cache key — and Patient 3 would receive Patient 1's clinical report. The test locks in the fix:

```python
def test_different_patients_get_different_cache_keys():
    key1 = _cache_key(pred_patient1, 'hospital')
    key3 = _cache_key(pred_patient3, 'hospital')
    assert key1 != key3   # Must be different — the patient identity separates them
```

Similarly, tests for SBP and RR in the cache key (C2 fix verification) ensure that two patients with different vital signs never accidentally share a cached response:

```python
def test_different_sbp_gives_different_key():
    pred_high_sbp = {**base_pred, 'vitals': {**base_vitals, 'SBP': 140}}
    pred_low_sbp  = {**base_pred, 'vitals': {**base_vitals, 'SBP': 80}}
    assert _cache_key(pred_high_sbp, 'hospital') != _cache_key(pred_low_sbp, 'hospital')
```

**The C6 fix — `judge_used` in the response:**

The `generate_report()` function was enhanced to include a `judge_used` field indicating whether the LLM-as-judge consistency verification ran all 9 methods (vs 7). Tests #8 and #9 verify this:

```python
def test_generate_report_has_judge_used_key(mock_groq):
    result = generate_report(patient_id=1, pred={...}, care_mode='hospital')
    assert 'judge_used' in result

def test_judge_used_is_bool(mock_groq):
    result = generate_report(patient_id=1, pred={...}, care_mode='hospital')
    assert isinstance(result['judge_used'], bool)
```

This matters for the LLM Validation Drawer in the frontend, which shows "✓ 9/9 methods ran" when `judge_used=True` or "⚠ 7/9 methods ran" when `judge_used=False`. The tests guarantee this field is always present and always a boolean, so the frontend can safely read it.

---

## 7. Phase 2 — Backend API Tests (Integration)

**Goal:** Test every HTTP endpoint through a real (but in-process) HTTP client.

**What TestClient does:**

FastAPI's `TestClient` wraps the ASGI app and lets you make HTTP requests without starting an actual server on a port. Requests go in-process: your test → TestClient → FastAPI router → service layer → database → response back.

```python
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# This is a REAL HTTP request processed by the REAL FastAPI app
response = client.get("/health")
assert response.status_code == 200
```

This tests the full request-response lifecycle: routing, request parsing, validation, business logic, database interaction, response serialisation — all in one test. The difference from unit tests is that nothing is mocked.

---

### 2A — `test_api_health.py`

The health endpoint is the simplest possible test — but it is critically important for monitoring.

```
GET /health → 200 {"status": "ok", "service": "ClinIQ"}
```

In production deployments, a load balancer or container orchestrator (like Kubernetes) pings `/health` every 10 seconds to determine whether the backend is healthy. If this endpoint returns anything other than 200, the orchestrator restarts the container.

The test checks the `service` field specifically ("ClinIQ" not "ICU Monitor" or "Backend") because the frontend uses this field to verify it is talking to the right service. A mismatch would indicate the wrong backend version is running.

---

### 2B — `test_api_patients.py`

The 11 patient tests verify the full patient listing and individual patient retrieval.

**Why check all required fields (`{id, name, short, icon, description, color, risk_level, sofa_range}`) rather than just that the request succeeded?**

An HTTP 200 response only means "the server responded." It does not mean the response contains the data the frontend needs. The frontend destructures these specific fields:

```js
// In SelectPatient.jsx
const { id, name, short, icon, description, color, risk_level } = patient
```

If any field is missing or renamed, the frontend renders `undefined` — no crash, no error, just blank cards that silently confuse the clinician. The API test verifies every field is present.

**Why test all 3 patients' risk levels specifically?**

```
Patient 1 → "Low Risk"      (stable reference patient)
Patient 2 → "Moderate Risk" (intermediate patient)
Patient 3 → "High Risk"     (critical patient)
```

The three demo patients were chosen to represent each risk tier. If anyone changes the seed data or the risk classification thresholds, these tests immediately flag that the demonstration no longer covers all three tiers.

**Why test boundary patient IDs (-1, 0, 4)?**

```
GET /api/patients/0  → 404
GET /api/patients/-1 → 404
GET /api/patients/4  → 404
```

Only patients 1, 2, and 3 exist. The system should return 404 (Not Found) for any other ID. Without these tests, a developer might accidentally return 500 (server error) for invalid IDs — which would break the frontend's error handling and potentially expose internal error messages.

**Why do hex color tests matter?**

```python
assert patient['color'].startswith('#')
assert len(patient['color']) == 7   # e.g. "#4A90D9" — 6 hex digits + #
```

The patient colors (`color` field) are used for the risk-level color coding throughout the UI. If the backend returns `"blue"` (CSS name) instead of `"#4A90D9"` (hex), the frontend's inline style would receive an unsupported format and render a default color — making all patient cards look identical.

---

### 2C — `test_api_predict.py`

The predict endpoint is the most complex and most important in the entire system. It has the most tests (22) for this reason.

**The full test list with explanations:**

**Response structure tests:**
```
Required keys: {patient_id, row_idx, cur_idx, total_rows, sofa_score,
                risk_level, risk_color, vitals, vitals_status, shap, conformal}
```
Every field is used by some part of the frontend. Missing a field means some dashboard component silently renders nothing.

**Clinical value range tests:**
```
SOFA in [0.0, 24.0]    — SOFA is a medical scale, cannot be outside this
risk_level in {"Low Risk", "Moderate Risk", "High Risk"}
risk_color is #RRGGBB format
```

**Conformal prediction interval tests:**
```
conformal_lo ≤ sofa_score ≤ conformal_hi   — interval must bracket the prediction
conformal_lo ≥ 0                            — SOFA cannot be negative
conformal_hi ≤ 24                           — SOFA cannot exceed 24
```

The conformal prediction interval is a statistically rigorous uncertainty range around the SOFA prediction. For example, if predicted SOFA = 8.4, the conformal interval might be [6.1, 10.7] — meaning "with 90% confidence, the true SOFA is somewhere in this range." These tests verify the interval is both mathematically valid (brackets the prediction) and clinically valid (within the SOFA scale bounds).

**SHAP structure tests:**
```
shap is a list of ≤ 5 items
Each item: {feature, direction, rel_impact}
direction ∈ {"up", "down"}
rel_impact ∈ [0, 1]      ← relative impact, 1.0 = largest factor, 0.0 = no impact
```

The SHAP panel in the dashboard renders these as direction arrows with proportional impact bars. If `rel_impact` is outside [0, 1], the bar's width calculation produces a visual overflow. If `direction` is an unexpected value, the arrow icon lookup fails silently.

**Row cycling tests:**
```
Large row_idx wraps correctly          — modulo cycling must stay in bounds
Consecutive row_idx increments cur_idx — 90-second polling advances through data
```

The system cycles through 48,150 MIMIC-III patient rows, using each row's vitals as a simulated real-time reading. The modulo cycling test checks that row_idx=48,200 correctly wraps to a valid row index (48,200 mod 48,150 = 50). Without this test, a row_idx larger than the dataset would cause an array-out-of-bounds error.

**Patient validation tests:**
```
patient_id=4  → 400   (only 3 patients exist)
patient_id=0  → 400   (IDs start at 1)
patient_id=-1 → 400   (negative IDs are nonsensical)
Empty body    → 422   (Pydantic model validation rejects missing required fields)
```

The difference between 400 (Bad Request) and 422 (Unprocessable Entity) matters for the frontend's error handling:
- 422 means Pydantic caught a malformed request before it reached any business logic — the client sent wrong data types or missing fields
- 400 means the request was well-formed but the values were invalid — the patient ID doesn't exist

The frontend shows different error messages for each case.

---

### 2D — `test_api_vitals.py`

**Why test `?limit=21 → 400` and `?limit=0 → 400`?**

The vitals endpoint is `GET /api/vitals/1?limit=20`. The limit parameter controls how many rows the frontend charts plot. Valid range is 1–20.

If `limit=0` is accepted: the query would return 0 rows, the chart would show no data, and the clinician might think the patient has no vital history.

If `limit=21` is accepted: the endpoint would try to return more rows than exist in the sliding window, which is capped at 20. The response would be inconsistent with the sliding window guarantees documented in the system.

**Why test HR ∈ [20, 300] and SpO₂ ∈ [50, 100]?**

These are "clinical sanity" checks — they verify the endpoint returns physiologically plausible values, not arbitrary numbers. If the database has a corrupt row (e.g., HR=999 from a hardware glitch), the clinical sanity test catches it. The ranges are wider than the clinical normal ranges (to allow for abnormal-but-real values) but narrower than the universe of all floats (to catch clearly corrupt data).

---

### 2E — `test_api_history.py`

Similar structure to the vitals tests. Key differences:

**Default limit ≤ 5 (not 20):**

The history drawer shows the 5 most recent predictions by default — enough for a clinician to see recent trends without overwhelming the UI.

**`?limit=51 → 400`:**

The sliding window for prediction history is 50 rows maximum. Requesting limit=51 would ask for more rows than the system stores. This test verifies the cap is enforced at the API level.

**Required columns: `{Timestamp, SOFA, Risk}`:**

These are the three columns displayed in the HistoryDrawer table. The test verifies the exact column names, because the frontend accesses them by name:
```js
{history.map(row => <tr>
    <td>{row.Timestamp}</td>
    <td>{row.SOFA}</td>
    <td>{row.Risk}</td>
</tr>)}
```

A rename from `SOFA` to `sofa_score` in the backend would break the table silently — the test catches it.

---

### 2F — `test_api_report.py`

The report endpoint is the most complex API test because it calls the Groq LLM, validates a multi-step response, and has several specific security checks.

**The 4 required section keys:**

```
{CURRENT CONDITION, PROBABLE CAUSE, RISK FORECAST, IMMEDIATE ACTIONS}
```

The frontend renders exactly these four sections in separate cards. A missing key → empty card. A renamed key → empty card. The test verifies all four are present and non-trivial:

```python
assert len(report['sections']['CURRENT CONDITION']) > 20
# Prevents a "correct structure, empty content" failure
# (e.g., the LLM returns a section with just whitespace)
```

**The C5 fix tests — cross-patient ID mismatch:**

```
predict_result.patient_id mismatch → 400
predict_result.vitals not dict     → 400
predict_result.sofa_score missing  → 400
```

The C5 fix addresses a scenario where the report endpoint receives a `predict_result` that was generated for a different patient than the current `patient_id`. This could happen if:
- The frontend has a stale cached prediction for a previous patient
- A race condition in rapid patient switching

Without the C5 fix, the backend would generate a report for Patient 3 using Patient 1's SOFA and vitals — a dangerous mixing of patient data. The test verifies the backend rejects mismatched IDs with a 400 error before generating any LLM content.

**The `judge_used` test:**

```python
assert 'judge_used' in response.json()
assert isinstance(response.json()['judge_used'], bool)
```

This is the C6 fix verification. The `judge_used` field tells the LLM Validation Drawer whether the full 9-method consistency check ran (True) or the fallback 7-method check ran (False). If this field is missing, the frontend Drawer shows "—" instead of a method count.

---

## 8. Phase 3 — Backend System Tests

**Goal:** Simulate complete multi-step monitoring sessions to verify that sequences of API calls work correctly together.

**Why system tests catch bugs that unit and API tests miss:**

Individual endpoint tests verify that each endpoint works in isolation. But real bugs often emerge from the interaction between endpoints:

```
Example — the "vitals from predict" consistency bug:

API test for /api/predict: PASS (returns latest vitals correctly)
API test for /api/vitals/1: PASS (returns vitals from DB correctly)

But a system test might reveal:
  After a predict call, the vitals endpoint returns DIFFERENT HR value
  because predict writes to the DB with different rounding than vitals reads
```

**The 7 system tests:**

| Test | What it simulates | Why it matters |
|---|---|---|
| Full predict → vitals → history flow | The complete dashboard initialisation sequence | Verifies the app works correctly from first load |
| All 3 patients predict independently | Clinician switches between patient views | Verifies no cross-contamination between concurrent sessions |
| Row cycling stays in bounds | The 90-second automatic data refresh loop over the full 48,150-row dataset | Cycling must never crash regardless of row count or wrap-around point |
| Vitals table grows after predict | Each predict call should add one row to the vitals history | Verifies the sliding window write is actually happening |
| Cross-patient history isolation | Patient 2's predictions must not appear in Patient 1's history table | Critical patient safety test — wrong patient data in history is dangerous |
| Health endpoint stays 200 | Backend uptime during an active monitoring session | Verifies that predict/vitals/history calls don't put the backend into a broken state |
| Vitals from predict match vitals endpoint | The latest row returned by /api/vitals must match the values just returned by /api/predict | Verifies consistency between the two most important data sources |

**The cross-patient history isolation test in detail:**

```python
def test_cross_patient_history_isolation(client):
    # Make predictions for all 3 patients
    client.post('/api/predict', json={'patient_id': 1, 'row_idx': 0})
    client.post('/api/predict', json={'patient_id': 2, 'row_idx': 0})
    client.post('/api/predict', json={'patient_id': 3, 'row_idx': 0})
    
    # Get patient 2's SOFA score
    p2_predict = client.post('/api/predict', json={'patient_id': 2, 'row_idx': 1}).json()
    p2_sofa = p2_predict['sofa_score']
    
    # Get patient 1's history
    p1_history = client.get('/api/history/1').json()
    
    # Patient 1's history must not contain patient 2's SOFA score
    p1_sofa_values = [row['SOFA'] for row in p1_history]
    assert p2_sofa not in p1_sofa_values
```

---

## 9. Phase 4 — Frontend Unit Tests

**Goal:** Test every React component, hook, context, and API function in isolation, using jsdom as the browser environment and mocking everything external.

**Key principle:** Each test renders only the component being tested. Child components that make network calls or use browser-only APIs are either mocked or simplified.

---

### 4A — Layout Components

**`AlertBanner.test.jsx`**

The AlertBanner appears at the top of the dashboard when the patient's risk level is Moderate or High. It shows the SOFA score and uses `role="alert"` for screen reader accessibility.

```jsx
// What the component renders for High Risk:
<div role="alert" className="alert-banner alert-critical">
    <span>CRITICAL ALERT</span>
    <span>SOFA Score: 8.4 — Immediate clinical review required</span>
</div>
```

Why the null-return tests for `predictData=null` and `risk_level="Low Risk"` matter:
- `predictData=null` is the initial state before the first API call completes — the banner must not crash during loading
- For "Low Risk", showing an alert would desensitise clinicians to alerts over time (alert fatigue). No alert for stable patients is the correct clinical behaviour.

The `role="alert"` test is an accessibility test — screen readers used by visually impaired clinicians announce `role="alert"` elements automatically. Without this attribute, visually impaired users would not be aware of critical alerts.

**`HeaderBar.test.jsx`**

The header shows the patient name, a countdown timer (seconds until next prediction), and the last-updated timestamp.

Why test the "—" placeholder for missing last-updated?

When the app first loads, no prediction has run yet — `lastUpdated` is null. The header should show "—" (an em dash indicating absence of data) rather than crashing or showing "Invalid Date" from a failed `Date.format()` call on a null value.

**`ThemeToggle.test.jsx`**

Simple toggle tests, but the icon direction matters: Sun icon in dark mode (click to go to light/brighter), Moon icon in light mode (click to go to darker). Reversed icons would confuse users who expect industry-standard toggle conventions.

---

### 4B — Overview Components

**`VitalCard.test.jsx`**

The VitalCard is the most frequently rendered component — 7 instances on the dashboard simultaneously (one per vital). The status color tests are critical:

```
status="normal" → green color token applied
status="high"   → amber/red color token applied
status="low"    → blue/amber color token applied
```

Wrong colors would convey incorrect clinical signals. A patient with critically low SpO₂ shown in green would look fine to a clinician glancing at the dashboard.

The `value=null` test (renders "—" without crash) handles the loading state and also handles the case where a vital sign is not being monitored (e.g., some patients don't have Temp readings in all rows).

**`SOFAGauge.test.jsx`**

The SOFA gauge is an SVG arc gauge showing the score visually (0–24). Critical tests:

- **Zone markers at 6, 9, 12:** These are the visual risk zone boundaries on the gauge. If they are wrong, the gauge gives incorrect visual guidance about where the current SOFA score falls relative to risk thresholds. (Note: 5 and 10 are the classification thresholds, but the visual gauge uses slightly different zone positions for design reasons — the test locks in the actual implemented values.)
- **score=0 without crash:** The gauge calculates an arc angle from the score — `angle = (score / 24) × 270`. For score=0, the arc has zero length, which could produce NaN in some SVG implementations.
- **score=24 without crash:** Maximum score — ensures the arc doesn't overflow its container.

**`VitalsGrid.test.jsx`**

Renders exactly 7 VitalCard children. The "exactly 7" test is a snapshot of the system's architecture: the 7 vitals are HR, RR, SpO₂, Temp, SBP, DBP, MAP. If someone adds an 8th vital to the backend but forgets to add its VitalCard, the count test would still pass — but if someone removes a vital from the grid accidentally, the test catches it.

---

### 4C — Report Components

**`ReportContainer.test.jsx` — The Most Complex Frontend Test**

ReportContainer is the most complex component in the system — it manages the LLM report lifecycle: detecting when to regenerate, showing loading/stale/error states, and displaying the final report. It has 13 tests.

**The reportKey mechanism — when to regenerate:**

The component uses a `reportKey` value to decide when to trigger a new LLM API call. The key is computed from clinical values that actually affect the SOFA score:

```js
const reportKey = `${sofa_score}-${GCS_eye}-${stress}`
```

Tests verify the key changes appropriately:
```
reportKey changes when sofa_score changes   ✓ (the main prediction changed)
reportKey changes when GCS_eye changes      ✓ (neurological state changed)
reportKey changes when stress changes       ✓ (patient comfort changed)
reportKey stays SAME when only timestamp changes  ✓ (cosmetic update — no new report needed)
```

The timestamp-invariance test is subtle but important. The predict endpoint runs every 90 seconds — every call includes a new timestamp. If the report re-generated on every timestamp change, it would trigger a new (expensive, slow) LLM API call every 90 seconds, burning API credits and showing a loading skeleton every minute and a half. The test verifies that cosmetic changes (timestamp) don't trigger unnecessary regeneration.

**Error state tests:**

```
401 → "Invalid API key" message
429 → "Rate limit reached" message
503 → "AI service temporarily unavailable" message
```

When the LLM call fails, the frontend needs to show the clinician a specific, actionable message — not a generic "error" that leaves them confused:
- 401: "Your Groq API key is wrong or expired — check the `.env` file"
- 429: "Too many requests in a short time — wait a minute and try again"
- 503: "Groq's servers are having issues — this will resolve automatically"

Each status code maps to a specific user-facing message. The tests verify these mappings are correct.

**`SHAPPanel.test.jsx`**

The "Keeps last factors during refresh" test is important UX: when the 90-second polling cycle refreshes the prediction, there is a brief moment where `factors=[]` (the new SHAP values haven't loaded yet). Without this behaviour, the SHAP panel would briefly disappear every 90 seconds — a jarring flash. The test verifies that the component keeps showing the previous SHAP factors until new ones arrive.

---

### 4D — Trends Components

**`TrendsSection.test.jsx`**

The vital selector tabs let the clinician choose which of the 7 vitals to see in the trend chart. The "Clicking another vital updates selection" test verifies that tab switching actually changes what is displayed — not just what appears to be highlighted.

```jsx
// What the test does:
render(<TrendsSection vitals={mockData} />)
expect(screen.getByText('HR')).toBeVisible()  // default: HR selected

await userEvent.click(screen.getByRole('tab', { name: 'SpO2' }))
// Wait for state update
expect(screen.getByText('SpO₂ (%)')).toBeInTheDocument()  // SpO2 chart now shown
```

---

### 4E — Validation Components

**`SOFAValidationDrawer.test.jsx` — Model Performance Values**

The SOFA Validation Drawer shows the model's validated performance metrics:
```
MAE = 1.8968
R² = 0.5995
AUROC = 0.9461
PR-AUC = 0.6840
3-class accuracy = 70.2%
Test split: 2,559 patients
High-risk recall: 19.8%
```

**Why these specific values?**

These are the actual metrics from the final federated model evaluation:

- **AUROC = 0.9461:** The model correctly ranks 94.6% of random (High Risk, non-High Risk) patient pairs. This means if you pick any two patients — one critical and one stable — the model gives the critical patient a higher predicted SOFA 94.6% of the time. This is excellent discrimination.

- **PR-AUC = 0.6840:** The area under the Precision-Recall curve, which is more informative than AUROC when the positive class (High Risk) is rare (about 6% of patients). A random classifier achieves PR-AUC ≈ 0.06 (the base rate). At 0.684, the model is 11× better than random for identifying critical patients.

- **High-risk recall = 19.8%:** At the default SOFA threshold (≥10 for High Risk), the model correctly identifies 19.8% of true critical patients. This seems low, but is consistent with the DNN's known systematic under-prediction. The practical workaround is the ALERT_THRESHOLD=8.0 (explained in INPUT_OUTPUT.md), which improves recall to 52.5%.

- **Confusion matrix renders 3 rows:** The 3×3 confusion matrix (Low/Moderate/High Risk actual vs predicted) has exactly 3 rows. The test verifies the rendering because a wrong row count would indicate the matrix data structure is malformed.

The tests lock these values in so the Drawer always shows verified metrics, not placeholder zeros or accidentally overwritten values.

**`LLMValidationDrawer.test.jsx`**

```
judge_used=True  → "✓ 9 / 9 methods ran"
judge_used=False → "⚠ 7 / 9 methods ran"
```

The 9 "methods" in LLM validation refer to 9 consistency measurement dimensions (TF-IDF similarity, intervention agreement for 7 categories). When `judge_used=True`, all 9 ran. When `judge_used=False`, 2 of the 7 intervention categories could not be evaluated (usually because the LLM response didn't discuss those specific interventions), reducing the total to 7.

---

### 4F — CV Components

**`CVMonitorModal.test.jsx`**

The CV Monitor modal shows the computer vision feed for GCS Eye Opening and Stress Score assessment. Two key text checks:

```
Shows supplementary label text
Shows "SOFA prediction uses dataset values" text
```

The second check is a crucial disclaimer: in the deployed dashboard, the CV module provides real-time readings, but the SOFA DNN prediction itself uses cycling MIMIC-III dataset rows (not the live vitals). This disclaimer prevents clinicians from thinking the SOFA score reflects the live camera-derived values. Testing this text ensures the disclaimer cannot be accidentally removed.

---

### 4G — History Components

**`HistoryFAB.test.jsx`**

The HistoryFAB (Floating Action Button) shows a badge with `recordCount` when > 0. This badge tells the clinician "there are N previous predictions in the history" at a glance, without needing to open the drawer. The test verifies the count appears when records exist and does not appear when count is zero.

**`HistoryDrawer.test.jsx`**

The empty state test ("Shows empty state when data=[]") is important for new patients — a patient with no prediction history should see a friendly "No history yet" message, not a blank table with no headers or a JavaScript error from trying to `.map()` over an empty array.

---

### 4H — Hook Tests (`useMonitor.test.js`)

`useMonitor` is the most important hook in the application — it drives the entire monitoring loop. It manages:
- Making the initial prediction on mount
- Auto-refreshing every 90 seconds
- Incrementing `rowIdx` on each successful fetch
- Resetting when the patient changes
- Preventing race conditions

**The `fetchingRef` concurrent fetch guard:**

```
Concurrent fetches are guarded (fetchingRef)
```

This tests a subtle race condition. The hook fires a prediction fetch every 90 seconds. What if the Groq LLM takes 95 seconds to respond? Without the `fetchingRef` guard, the next 90-second tick would fire a second prediction fetch WHILE the first one is still in progress. Two simultaneous fetches could overwrite each other's results in random order, producing inconsistent state.

`fetchingRef` is a ref (not state) that acts as a mutex: "if I'm already fetching, do not start another fetch." The test verifies this guard works.

**The StrictMode double-invoke guard:**

```
StrictMode double-invoke guard (lastPatientRef)
```

React's Strict Mode (enabled in development) intentionally renders components twice — once to find side effects and once for the real render. Without a guard, `useMonitor` would fire TWO prediction fetches on mount: one from the Strict Mode test render and one from the real render. This would waste one API call and could cause state to update before the component is ready.

`lastPatientRef` stores the last patient ID that triggered a fetch. On the second Strict Mode invoke with the same patient ID, the guard says "I already fetched for this patient — don't fetch again." The test verifies only ONE fetch fires, not two.

**The `rowIdx` increment test:**

```
rowIdx increments after each successful fetch
```

This verifies the 48,150-row cycling mechanism. Each prediction call uses a different row of MIMIC-III data. If `rowIdx` never incremented, the same patient row would be used for every prediction — the dashboard would appear "stuck" showing the same vitals indefinitely.

---

### 4I — Context Tests (`ThemeContext.test.jsx`)

```
Default isDark state is boolean   ← not undefined, not null
toggleTheme flips isDark          ← false→true or true→false
Calling toggleTheme twice returns to original  ← toggle, not one-way setter
```

The "toggle twice returns to original" test is the most important: it verifies that `toggleTheme` is genuinely a toggle (alternates between states), not a setter that always goes in one direction. A bug like:

```js
const toggleTheme = () => setIsDark(true)   // bug: always sets to dark, never toggles back
```

would pass all tests except the "toggle twice" test, which would see: `isDark` → `true` → `true` (failed to return to original).

---

### 4J — API Client Tests (`client.test.js`)

These tests verify that each API function calls the correct HTTP method and URL with the correct parameters. They use `vi.spyOn(axios, 'get')` and `vi.spyOn(axios, 'post')` to intercept the actual HTTP calls.

```js
test('runPredict sends correct body', async () => {
    const postSpy = vi.spyOn(axios, 'post').mockResolvedValue({ data: mockPred })
    await runPredict(1, 5)
    expect(postSpy).toHaveBeenCalledWith('/api/predict', { patient_id: 1, row_idx: 5 })
})
```

Why test these simple wrapper functions? Because the URL is a contract between frontend and backend. If `getHistory` were accidentally changed to call `/api/prediction_history` instead of `/api/history`, the backend would return 404 — and the bug would only be caught when the two sides are tested together. The API client tests catch this class of URL mismatch immediately.

---

## 10. Phase 5 — Frontend Integration Tests

**Goal:** Test complete page components (`SelectPatient` and `MonitorDashboard`) with API calls mocked using MSW.

**The difference from unit tests:**

Unit tests render a single isolated component. Integration tests render an entire page, including all its child components, with real React routing and real React context providers. The only thing mocked is the network layer.

### `SelectPatient.test.jsx`

**Loading state test:**

```js
// Before the mock API responds, the component should show a loading state
render(<SelectPatient />)
expect(screen.getByRole('progressbar')).toBeInTheDocument()
// ...wait for API mock to respond...
expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
```

The loading state test verifies that the user sees feedback ("loading...") rather than a blank page while the patient list is fetching. Without this test, a developer might accidentally remove the loading state check and produce a blank screen flash on every app load.

**Navigation test:**

```js
// Clicking a patient card should navigate to /monitor/1
await userEvent.click(screen.getByText('Patient 1'))
expect(mockNavigate).toHaveBeenCalledWith('/monitor/1')
```

This is the critical integration between the SelectPatient page and React Router. The test verifies the entire "click patient → navigate" path, not just that the button exists.

### `MonitorDashboard.test.jsx`

The MonitorDashboard integration test is the largest single test file — it exercises the entire dashboard at once, verifying that all sub-components receive and display data correctly.

**The redirect test:**

```js
// If no patientId in the URL, redirect to "/"
render(<MonitorDashboard />, { route: '/monitor/' })
expect(mockNavigate).toHaveBeenCalledWith('/')
```

A clinician should not be able to navigate directly to `/monitor/` without a patient ID. The redirect test ensures this protection is working.

**The FAB → Drawer integration tests:**

```
HistoryFAB opens HistoryDrawer on click
CVMonitorFAB opens CVMonitorModal on click
SOFAValidationFAB opens SOFAValidationDrawer on click
LLMValidationFAB opens LLMValidationDrawer on click
```

Each of these floating action buttons must open its corresponding drawer or modal. Unit tests verify each component renders correctly in isolation — but they cannot verify that `MonitorDashboard` correctly connects the FAB's `onClick` to the Drawer's `open` state. Integration tests verify this wiring.

**The tab switching test:**

```js
await userEvent.click(screen.getByRole('tab', { name: 'Trends' }))
expect(screen.getByTestId('trends-section')).toBeVisible()
expect(screen.queryByTestId('overview-section')).not.toBeVisible()
```

Verifies that only one tab's content is visible at a time, and that switching tabs shows the correct content.

---

## 11. Phase 6 — End-to-End Tests (Playwright)

**Goal:** Tests a real user interacting with the real, fully running application in a real browser.

**Requirements before running:**
- Backend server running on `localhost:8000` (`python -m uvicorn backend.main:app --port 8000`)
- Frontend dev server running on `localhost:5173` (`npm run dev` from `frontend/`)
- `.env` file with valid `GROQ_API_KEY` (report generation tests need it)

**Why E2E tests catch bugs that lower-level tests cannot:**

- CSS that incorrectly hides an element (the element is in the DOM but not visible)
- JavaScript that works in jsdom but fails in Chrome's V8 engine
- Timing issues (a button is clickable but the state hasn't updated yet)
- Browser-specific rendering quirks (Safari renders SVGs differently than Chrome)

### `patient_selection.spec.js`

```js
test('patient cards load and clicking navigates', async ({ page }) => {
    await page.goto('http://localhost:5173')
    
    // Wait for cards to appear (not just the page to load)
    await expect(page.locator('.patient-card')).toHaveCount(3)
    
    // Click Patient 1 by visible text
    await page.click('text=Patient 1')
    
    // Verify navigation happened
    await expect(page).toHaveURL('http://localhost:5173/monitor/1')
    
    // Verify the dashboard loaded with the right patient
    await expect(page.locator('h1')).toContainText('Patient 1')
})
```

### `monitoring_flow.spec.js`

These tests simulate a complete monitoring session from the clinician's perspective:

1. Navigate to a patient's dashboard
2. Verify the SOFA gauge and vital cards appear
3. Open the History drawer via FAB
4. Verify history rows appear in the table
5. Open the CV Monitor modal
6. Verify the disclaimer text is present
7. Open the SOFA Validation drawer
8. Verify the MAE/R² values are displayed (these come from the real backend!)

The SOFA validation drawer test is especially valuable in E2E context — it verifies that the real backend correctly loads `training_metadata.json` and returns the right values through the real API, not just that the frontend renders whatever values it receives.

### `report_generation.spec.js`

The most time-sensitive E2E test — this calls the real Groq API and must wait for a real LLM response.

```js
test('report shows content after LLM call', async ({ page }) => {
    await page.goto('http://localhost:5173/monitor/1')
    
    // Navigate to Report tab
    await page.click('text=Report')
    
    // Loading skeleton should appear immediately
    await expect(page.locator('[data-testid="report-skeleton"]')).toBeVisible()
    
    // Wait for the real LLM response (can take 10-30 seconds)
    await expect(page.locator('text=CURRENT CONDITION')).toBeVisible({ timeout: 45000 })
    
    // All 4 section cards should be present
    await expect(page.locator('.report-section')).toHaveCount(4)
    
    // Reliability badge should appear
    await expect(page.locator('[data-testid="reliability-badge"]')).toBeVisible()
})
```

The 45-second timeout is explicitly set for LLM response tests. The default Playwright timeout is 5 seconds — waiting for a real LLM call requires a custom timeout. The test is marked `@slow` in pytest (or equivalent in Playwright) so it can be skipped during rapid development cycles and only run in CI or before presentations.

---

## 12. Running the Tests

### Install Dependencies

```bash
# Backend test dependencies (activate virtualenv first)
pip install pytest pytest-asyncio httpx

# Frontend test dependencies
cd icu_monitor/frontend
npm install --save-dev \
  vitest \
  @vitest/coverage-v8 \
  @testing-library/react \
  @testing-library/jest-dom \
  @testing-library/user-event \
  jsdom \
  msw

# E2E dependencies (optional — only for Phase 6)
npm install --save-dev @playwright/test
npx playwright install chromium   # Install the browser Playwright will drive
```

### Run Commands

```bash
# ── BACKEND TESTS ──────────────────────────────────────────────────

# All backend tests (unit + API + system)
cd icu_monitor
pytest tests/cliniq/backend/ -v

# Unit tests only (fast — no database or ML load needed)
pytest tests/cliniq/backend/unit/ -v

# API tests only (need the DB to exist — run init_db first)
pytest tests/cliniq/backend/api/ -v

# System tests only
pytest tests/cliniq/backend/system/ -v

# Skip slow tests (LLM tests that call real Groq API)
pytest tests/cliniq/backend/ -v -m "not slow"

# Run specific test file
pytest tests/cliniq/backend/unit/test_db_unit.py -v

# Run specific test by name
pytest tests/cliniq/backend/api/test_api_predict.py::test_sofa_in_valid_range -v

# ── FRONTEND TESTS ─────────────────────────────────────────────────

# All frontend unit + integration tests
cd icu_monitor/frontend
npx vitest run

# Watch mode (re-runs tests on file changes — use during development)
npx vitest

# With coverage report
npx vitest run --coverage
# Coverage HTML report: open frontend/coverage/index.html in browser

# Run specific test file
npx vitest run tests/cliniq/frontend/unit/components/overview/SOFAGauge.test.jsx

# ── E2E TESTS ──────────────────────────────────────────────────────
# (Start backend and frontend servers first in separate terminals)

# Run all E2E tests
npx playwright test

# Run one E2E spec file
npx playwright test tests/cliniq/frontend/e2e/patient_selection.spec.js

# Run E2E with visible browser (helpful for debugging)
npx playwright test --headed

# Show Playwright trace/report (HTML page with test recording)
npx playwright show-report
```

### The Recommended Development Workflow

When actively coding, run tests in this sequence:

```
1. After editing a backend function:
   pytest tests/cliniq/backend/unit/ -v -m unit
   → Fast feedback (<5 seconds)

2. After editing a backend route:
   pytest tests/cliniq/backend/api/test_api_[changed_endpoint].py -v
   → Medium feedback (<30 seconds)

3. After a significant backend change:
   pytest tests/cliniq/backend/ -v -m "not slow"
   → Full backend feedback (<60 seconds, skips LLM calls)

4. After editing a frontend component:
   npx vitest run tests/cliniq/frontend/unit/components/[changed_component].test.jsx
   → Fast feedback (<5 seconds)

5. Before a presentation or demo:
   pytest tests/cliniq/backend/ -v     (ALL backend tests, including slow)
   npx vitest run --coverage           (ALL frontend tests + coverage report)
   npx playwright test                 (ALL E2E tests — need servers running)
```

---

## 13. Coverage Targets and What They Mean

Coverage measures what percentage of your production code is exercised by at least one test.

```
Line coverage 80% means: 80% of the lines in the file were executed
                         during the test run. 20% were never touched.
```

**The coverage targets:**

| Layer | Target | Why This Level |
|---|---|---|
| `db.py` | ≥ 90% | Database bugs are the most dangerous — wrong data in → wrong prediction out |
| `backend/api/*.py` | ≥ 85% | Route handlers are the interface contract — most paths must be tested |
| `ml_service.py` (pure fns) | ≥ 80% | Model loading and inference are tested through API tests, not unit tests |
| `llm_service.py` (pure fns) | ≥ 75% | LLM calls are expensive — heavy mocking is expected, reducing measurable coverage |
| Frontend components | ≥ 70% | UI components have many rendering variants — 70% is realistic without screenshot tests |
| `hooks/useMonitor.js` | ≥ 80% | The monitoring loop is critical — most timing and state branches must be tested |
| `contexts/ThemeContext.jsx` | ≥ 95% | Simple context with 2 values and 1 toggle — almost everything is testable |
| `api/client.js` | ≥ 95% | Thin wrappers around axios — every function must be called in at least one test |

**Coverage is a minimum bar, not a quality measure:**

100% coverage does not mean the code is correct — it means every line was executed. You could have 100% coverage and still have wrong output if your test assertions don't verify the right things. Coverage tells you "this code was run during tests." Assertions tell you "this code produced the right result."

Think of coverage as a **floor** (below 70% means significant untested areas), not a **ceiling** (100% doesn't mean done).

---

## 14. Implementation Status

| Phase | Status | Files |
|---|---|---|
| 1A — DB unit tests | ✅ Implemented | `tests/cliniq/backend/unit/test_db_unit.py` |
| 1B — ML service unit tests | ✅ Implemented | `tests/cliniq/backend/unit/test_ml_service_unit.py` |
| 1C — LLM service unit tests | ✅ Implemented | `tests/cliniq/backend/unit/test_llm_service_unit.py` |
| 2A — Health API tests | ✅ Implemented | `tests/cliniq/backend/api/test_api_health.py` |
| 2B — Patients API tests | ✅ Implemented | `tests/cliniq/backend/api/test_api_patients.py` |
| 2C — Predict API tests | ✅ Implemented | `tests/cliniq/backend/api/test_api_predict.py` |
| 2D — Vitals API tests | ✅ Implemented | `tests/cliniq/backend/api/test_api_vitals.py` |
| 2E — History API tests | ✅ Implemented | `tests/cliniq/backend/api/test_api_history.py` |
| 2F — Report API tests | ✅ Implemented | `tests/cliniq/backend/api/test_api_report.py` |
| 3 — Backend system flows | ✅ Implemented | `tests/cliniq/backend/system/test_system_flows.py` |
| 4 — Frontend unit tests | ✅ Implemented | `tests/cliniq/frontend/unit/**/*.test.{js,jsx}` |
| 5 — Frontend integration tests | ✅ Implemented | `tests/cliniq/frontend/integration/*.test.jsx` |
| 6 — E2E tests (Playwright) | ✅ Implemented | `tests/cliniq/frontend/e2e/*.spec.js` |
| FL tests | 🔜 Future | `tests/fl/` (placeholder only) |

---

*Documentation for ICU CDSS Capstone Project*
*Test suite: 14 database tests + 60+ API tests + 7 system tests + 80+ frontend unit tests + 17 integration tests + 20 E2E tests*
*Tools: pytest + httpx (backend) · Vitest + RTL + MSW (frontend) · Playwright (E2E)*

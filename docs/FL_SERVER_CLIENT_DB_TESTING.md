# FL System — Complete Testing Implementation Guide
**What every test does, why it exists, and how the FL test suite fits together**

> **Scope:** `train_federated.py`, `fl_dashboard.py`, `server.py`, `client.py`,
> `db.py` (FL-specific helpers), `model_utils.py`, `migrate_to_db.py`
> **All test files live in:** `icu_monitor/tests/fl/`
> **Not in scope:** ClinIQ dashboard (covered in `CLINIQ_TESTING.md`)

---

## Table of Contents

1. [Why FL Testing Is Different From ClinIQ Testing](#1-why-fl-testing-is-different-from-cliniq-testing)
2. [Files Under Test — What Each File Does](#2-files-under-test--what-each-file-does)
3. [Directory Layout](#3-directory-layout)
4. [Tool Stack](#4-tool-stack)
5. [Shared Fixtures — The Foundation Every Test Relies On](#5-shared-fixtures--the-foundation-every-test-relies-on)
6. [Phase 1 — Database FL Helpers](#6-phase-1--database-fl-helpers)
7. [Phase 2 — Model Utilities](#7-phase-2--model-utilities)
8. [Phase 3 — FL Dashboard Pure Functions](#8-phase-3--fl-dashboard-pure-functions)
9. [Phase 4 — Client Logic](#9-phase-4--client-logic)
10. [Phase 5 — Server Strategy](#10-phase-5--server-strategy)
11. [Phase 6 — Train Federated Helpers](#11-phase-6--train-federated-helpers)
12. [Phase 7 — Database ↔ Model Integration](#12-phase-7--database--model-integration)
13. [Phase 8 — FL Round Integration](#13-phase-8--fl-round-integration)
14. [Phase 9 — Migration Integration](#14-phase-9--migration-integration)
15. [Phase 10 — System Tests](#15-phase-10--system-tests)
16. [Run Commands](#16-run-commands)
17. [Coverage Targets](#17-coverage-targets)
18. [Implementation Status](#18-implementation-status)

---

## 1. Why FL Testing Is Different From ClinIQ Testing

The ClinIQ tests (in `CLINIQ_TESTING.md`) deal with a web application — HTTP requests, React components, database CRUD. The patterns are well-established: send a request, check the response.

The FL tests deal with something fundamentally harder to test:

**Problem 1 — Distributed computation:**
In real federated learning, 3 hospitals each run their own Flower client process, communicate with a central Flower server over gRPC (a network protocol), and exchange model weights that are gigabytes in size per round over 100 rounds. You cannot run all of this in a unit test — it would take hours, require network infrastructure, and make failures impossible to diagnose.

**Solution:** Test the mathematical logic in isolation. Extract the aggregation functions (`_fedavg`, `_fedyogi`), the training functions (`train_model`, `evaluate_model`), and the client/server strategy classes — and test each with tiny synthetic data and mocked Flower internals.

**Problem 2 — External library dependencies:**
Flower (the FL framework) uses gRPC for network communication and has its own event loop. Streamlit has a global state model that crashes if you try to import and call `st.write()` in a pytest test. PyTorch loads GPU drivers. MediaPipe needs WASM.

**Solution:** Monkeypatching and mocking at import time. Before the module loads, replace `fl.client.NumPyClient` with a stub, replace `st.cache_resource` with a passthrough decorator, replace `torch.save` with a no-op.

**Problem 3 — Non-determinism:**
FL training is stochastic — gradient descent uses random mini-batch sampling, weight initialisation is random, the order of hospital contributions varies. A test that checks "loss is exactly 4.213" would break on a different machine or Python version.

**Solution:** Test mathematical invariants and ordering, not exact values:
- "Loss after 5 epochs < loss after 1 epoch" (learning happens)
- "FedYogi loss < FedAvg loss after 5 rounds" (better algorithm wins)
- "FedProx delta_mean < FedAvg delta_mean" (drift is reduced)

These relationships must hold regardless of random seed, hardware, or library version.

---

## 2. Files Under Test — What Each File Does

| File | Role | What it contains |
|---|---|---|
| `db.py` | SQLite data layer | Functions for reading and writing all database tables — both ClinIQ dashboard tables AND the FL training data table (`fl_training`) |
| `model_utils.py` | ML core utilities | The DNN architecture (`ICUModel`), dataset class (`ICUDataset`), loss function, training loop, evaluation, SHAP preparation, weight serialisation for Flower, trend analysis, differential privacy |
| `fl_dashboard.py` | Streamlit FL simulator | The interactive FL simulation UI — implements `_fedavg` and `_fedyogi` in Python, manages simulation state (`_blank_state`, `_reset_state`, `_read_state`), builds Plotly charts |
| `client.py` | Flower client | The `HospitalClient` class — each hospital runs one instance. Implements `get_parameters`, `fit`, `evaluate` per Flower's protocol. Handles LR decay, FedProx regularisation, differential privacy |
| `server.py` | Flower server | The `SaveBestStrategy` class — extends Flower's `FedYogi` strategy. Aggregates weights from 3 hospitals, tracks the best model, saves checkpoints, writes `training_metadata.json` |
| `train_federated.py` | Unified FL script | Standalone script that runs the full 100-round federated training. Contains `make_noniid_split` for non-IID data distribution, its own `HospitalClient` variant, and its own `SaveBestStrategy` (FedYogi variant) |
| `migrate_to_db.py` | CSV → SQLite import | One-time migration that converts all CSV data files (patient vitals, prediction history, FL training data from 3 hospitals) into the SQLite database |

**Why are there two `HospitalClient` implementations and two `SaveBestStrategy` implementations?**

`client.py` + `server.py` are used with the Flower framework for live distributed training (where Flower manages network communication between real processes). `train_federated.py` contains variants of both that are used in the standalone unified script — this script runs all 3 hospitals in a single process using Python threads rather than separate gRPC-connected processes.

The logic is nearly identical but the entry points differ. Both are tested, because a bug in one variant will not be caught by testing the other.

---

## 3. Directory Layout

```
icu_monitor/tests/fl/
├── __init__.py                          ← Makes tests/fl/ a Python package
├── conftest.py                          ← Shared fixtures for ALL fl tests
│
├── unit/                                ← Tests for individual functions/classes
│   ├── __init__.py
│   ├── test_db_fl.py                    ← DB helpers specific to FL (fl_training table)
│   ├── test_model_utils.py              ← Every function in model_utils.py
│   ├── test_fl_dashboard_fns.py         ← Pure functions from fl_dashboard.py
│   ├── test_client_logic.py             ← HospitalClient class (client.py)
│   ├── test_server_strategy.py          ← SaveBestStrategy class (server.py)
│   └── test_train_federated_fns.py      ← make_noniid_split + config constants
│
├── integration/                         ← Tests for multiple components working together
│   ├── __init__.py
│   ├── test_db_model_integration.py     ← DB read → ICUDataset → train_model pipeline
│   ├── test_fl_round_integration.py     ← Full FL round without Flower gRPC
│   └── test_migrate_to_db.py            ← migrate() with temp DB + CSV stubs
│
└── system/                              ← End-to-end FL simulation tests
    ├── __init__.py
    └── test_fl_system.py                ← fl_thread logic with real synthetic data
```

**Why three levels (unit / integration / system)?**

- **Unit tests** answer: "Does this one function do what I think it does?"
- **Integration tests** answer: "Do these two functions work correctly when connected?"
- **System tests** answer: "Does the complete FL simulation run from start to finish and produce sensible results?"

Each level answers a different question and catches a different class of bug.

---

## 4. Tool Stack

### pytest (same as ClinIQ)

Same `pytest.ini` at `icu_monitor/` root — test discovery, fixtures, markers. See `CLINIQ_TESTING.md` Section 4 for a full explanation.

### `unittest.mock` and `pytest-mock`

These are Python's built-in and extended mocking tools. They allow replacing a real object (a function, a class, a module) with a fake version ("mock") for the duration of a test.

**Why mocking is essential for FL tests:**

```python
# Without mocking: importing client.py triggers this chain:
import flwr as fl          # Flower initialises gRPC client infrastructure
import torch               # PyTorch initialises CUDA/GPU subsystem
from db import get_conn    # Opens the real production database

# Your unit test now:
# - Might fail if GPU drivers aren't installed
# - Reads the real DB (tests should never touch production data)
# - Takes 10 seconds just to import

# With mocking: replace before import
import unittest.mock as mock
with mock.patch('flwr.client.NumPyClient'):
    from client import HospitalClient
    # Flower is now a stub — no gRPC, no network, no GPU
    # Test runs in milliseconds
```

**`pytest-mock`** provides the `mocker` fixture — a cleaner way to do the same thing:

```python
def test_fit_calls_train_model(mocker):
    mock_train = mocker.patch('client.train_model', return_value=0.5)
    client = HospitalClient(client_id=0, X_train=X, y_train=y)
    client.fit(initial_params, config={'server_round': 1})
    mock_train.assert_called_once()   # verify train_model was actually called
```

### `monkeypatch` (pytest built-in)

`monkeypatch` is a pytest fixture for temporarily replacing global variables, environment variables, or module attributes. Unlike `mock.patch`, `monkeypatch` changes are automatically reversed after each test — you never need to remember to clean up.

```python
def test_db_uses_temp_file(monkeypatch, tmp_path):
    # Replace the real DB path with a temp file — ONLY for this test
    monkeypatch.setattr('db.DB_PATH', str(tmp_path / 'test.db'))
    
    # Now every db.get_conn() call opens the temp file, not the real DB
    conn = get_conn()
    
    # After the test: DB_PATH is restored to its original value
    # The temp file is deleted automatically
```

### `numpy.testing.assert_allclose`

Floating-point arithmetic is not exact. `0.1 + 0.2 == 0.3` is `False` in Python (the actual result is `0.30000000000000004`). Using `==` to compare floats in tests causes random failures that have nothing to do with bugs.

`assert_allclose` compares floats with a tolerance:

```python
import numpy.testing as npt

# Wrong — fails due to floating-point representation:
assert weighted_loss == 1.5

# Correct — passes if result is within 0.1% of expected:
npt.assert_allclose(weighted_loss, 1.5, rtol=1e-3)
```

This is especially important for FL tests where weight values go through many floating-point operations (training → serialisation → network transfer → deserialisation → aggregation → next round's training).

### `@pytest.mark.slow`

Long-running tests are marked `slow`. The system tests run real FL simulations — even with tiny synthetic data, training a DNN for 3 rounds across 3 hospitals takes 20–60 seconds. During development, you skip these:

```bash
pytest tests/fl/ -m "not slow"   # skip slow tests — fast feedback loop
pytest tests/fl/ -m slow          # run ONLY slow tests — before a presentation
```

---

## 5. Shared Fixtures — The Foundation Every Test Relies On

All fixtures are defined in `tests/fl/conftest.py` and are automatically available to every test file under `tests/fl/` without any import statement.

### `tmp_db` — Isolated Test Database

```python
@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    # Create a fresh SQLite database in a temporary directory
    db_path = tmp_path / "test_fl.db"
    
    # Redirect db.DB_PATH to this temp file
    # (monkeypatch automatically restores the original value after the test)
    monkeypatch.setattr('db.DB_PATH', str(db_path))
    
    # Run schema creation — same init_db() the real app uses
    conn = sqlite3.connect(str(db_path))
    init_db(conn)
    
    yield conn   # provide the connection to the test
    conn.close() # cleanup after test completes
```

**Why every FL test needs this:**

The production database at `models/icu_monitor.db` contains real MIMIC-III derived data (~23 MB). Tests must NEVER read or write to it because:
- Tests that write could corrupt or pollute the real data
- Tests that read are slow (23 MB load) and environment-dependent (fails on machines without the file)
- Test outcomes would depend on the contents of the production database — tests must be deterministic

With `tmp_db`, every test gets its own empty database that:
- Starts with the correct schema (tables created by `init_db()`)
- Is completely empty (no leftover data from other tests)
- Is automatically deleted when the test finishes
- Has no connection to the production database

### `synthetic_client_data(n=200, input_dim=10)` — Fake Training Data

```python
@pytest.fixture
def synthetic_client_data():
    np.random.seed(42)   # fixed seed for reproducibility
    
    n = 200
    input_dim = 10
    
    # Generate X: n samples × input_dim features
    X = np.random.randn(n, input_dim).astype(np.float32)
    
    # Generate y: SOFA scores with realistic class distribution
    # 70% Low Risk (SOFA 0-4), 24% Moderate (5-9), 6% High (10-24)
    low_count      = int(n * 0.70)   # 140 patients
    moderate_count = int(n * 0.24)   # 48 patients
    high_count     = n - low_count - moderate_count  # 12 patients
    
    y_low      = np.random.uniform(0, 5, low_count)
    y_moderate = np.random.uniform(5, 10, moderate_count)
    y_high     = np.random.uniform(10, 24, high_count)
    y          = np.concatenate([y_low, y_moderate, y_high]).astype(np.float32)
    np.random.shuffle(y)   # mix so early samples aren't all Low Risk
    
    return X, y
```

**Why 70/24/6 class distribution?**

This mirrors the real MIMIC-III distribution: most ICU patients are stable (Low Risk), a minority are deteriorating (Moderate), and very few are in critical multi-organ failure (High). The model was trained on this imbalanced distribution — tests should use the same imbalance.

**Why `input_dim=10` instead of 108?**

The full production model has 108 input features. Using 108 in every test would require 200×108 = 21,600 float32 values per test, and a 10×(108→128→64→32→1) DNN which is much slower to train than 10→128→64→32→1. The `mini_model` and `synthetic_client_data` use `input_dim=10` to make tests run in milliseconds instead of seconds. The architecture logic (layer counts, activations, weight shapes) is identical regardless of input dimension.

### `mini_model(input_dim=10)` — Tiny DNN for Fast Testing

```python
@pytest.fixture
def mini_model():
    model = ICUModel(input_dim=10)
    return model
```

This is an `ICUModel` with input dimension 10 instead of 108. The architecture is identical — same 4 layers (10→128→64→32→1), same ReLU activations, same output layer. Using this in tests means training completes in a fraction of a second.

### `mini_weights(model)` — Serialised Weight List

```python
@pytest.fixture
def mini_weights(mini_model):
    return get_weights(mini_model)
```

`get_weights(model)` returns the model's parameters as a list of numpy arrays — this is the format Flower uses to transmit model weights between client and server. Having this pre-computed as a fixture saves repeating the call in every test that needs a valid weight list.

### `fl_training_rows(client_id, n=50)` — Synthetic FL Database Rows

```python
@pytest.fixture
def fl_training_rows(tmp_db):
    def _insert(client_id=0, n=50):
        # Generate n rows with 108 feature columns + sofa_score
        np.random.seed(42)
        X = np.random.randn(n, 108).astype(np.float32)
        y = np.random.uniform(0, 24, n).astype(np.float32)
        
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(108)])
        df['sofa_score'] = y
        df['client_id'] = client_id
        
        # Write to the fl_training table in tmp_db
        df.to_sql('fl_training', tmp_db, if_exists='append', index=False)
    
    return _insert
```

This fixture is a "factory" — it returns a function `_insert()` rather than the data directly. The test calls `fl_training_rows(client_id=0, n=50)` to insert data for Hospital 0, then `fl_training_rows(client_id=1, n=50)` for Hospital 1. This flexibility allows each test to set up exactly the data distribution it needs.

---

## 6. Phase 1 — Database FL Helpers (`test_db_fl.py`)

**Goal:** Test every `db.py` function related to the FL training table that is NOT already covered by `test_db_unit.py`. The ClinIQ DB tests cover `patient_vitals`, `prediction_history`, and `sample_patients`. This file covers `fl_training`, the table that stores the 48,150 MIMIC-III patient rows used for federated training.

**No ML model load. No real database touched.**

### 1A — `fl_training_exists()` (Tests 1–4)

This function answers one question: "Has FL training data been migrated into the database yet?" It is called at application startup to decide whether to prompt the user to run the data migration.

```python
def fl_training_exists(conn):
    cursor = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='fl_training'"
    )
    if cursor.fetchone()[0] == 0:
        return False    # table doesn't even exist yet
    cursor = conn.execute("SELECT COUNT(*) FROM fl_training")
    return cursor.fetchone()[0] > 0   # True only if at least 1 row exists
```

| # | Test | Why this specific case matters |
|---|---|---|
| 1 | Returns `False` when `fl_training` table is absent | The table is created dynamically by `migrate_to_db.py` — on a fresh install, it doesn't exist yet. The function must handle this without crashing ("table doesn't exist" SQLite error) |
| 2 | Returns `False` when table exists but has 0 rows | An interrupted migration could leave an empty table. A table with no data is equivalent to no migration having happened |
| 3 | Returns `True` after inserting at least 1 row | The normal post-migration case |
| 4 | Returns `True` for client_id=2 rows only | `fl_training_exists` should return True regardless of which client_id the rows belong to — it checks for ANY FL data, not data for a specific hospital |

### 1B — `get_fl_client_data(client_id)` (Tests 5–15)

This is the most important FL database function. It is called by each hospital's training client to load its portion of the training data:

```python
df = get_fl_client_data(client_id=0)   # Hospital 0 loads its 16,050 rows
X = df.drop('sofa_score', axis=1).values   # 108 feature columns
y = df['sofa_score'].values                 # target column
```

**All 11 tests with explanations:**

| # | Test | What it verifies | Why it matters |
|---|---|---|---|
| 5 | Returns a `pd.DataFrame` | Return type | The downstream code does `.drop()`, `.values`, `.fillna()` — all DataFrame operations. A dict or list would crash immediately |
| 6 | Drops `id` column from result | DB-only column removed | `id` is the SQLite primary key — it is meaningless as a training feature. If left in, the model would learn to use row number as a predictor, which is pure memorisation |
| 7 | Drops `client_id` column from result | DB-only column removed | `client_id` encodes which hospital the row came from. If left in, the model learns "Hospital 0 patients are Low Risk" instead of learning clinical patterns — a form of data leakage |
| 8 | Only returns rows for requested `client_id` | WHERE clause correctness | Hospital 0 must only train on its own data — privacy requirement. Returning all hospitals' data would defeat the entire purpose of federated learning |
| 9 | Returns 0 rows for client_id=99 (unknown) | Empty result safety | An invalid client ID should return an empty DataFrame, not crash |
| 10 | Rows with NULL `sofa_score` are dropped | `dropna(subset=["sofa_score"])` | SOFA is the training target. A row with no target is useless for supervised learning and would crash the PyTorch DataLoader when it tries to compute loss against a NaN |
| 11 | Warning is emitted when NULL rows are dropped | `warnings.warn` triggered | The caller should know that data was silently discarded — a silent discard could mask data corruption |
| 12 | NaN in feature columns replaced with 0 | `fillna(0)` applied | Feature NaNs crash PyTorch tensor creation. Replacing with 0 (the scaled mean) is the correct imputation strategy for normalised data |
| 13 | `sofa_score` column is present in result | Label column kept | The downstream code extracts `y = df['sofa_score'].values` — if the column is missing, this crashes |
| 14 | Returns same row count as inserted (no NULL SOFA) | Count correctness | With clean data, no rows should be dropped |
| 15 | Two clients return independent DataFrames | Cross-client isolation | Hospital 0's DataFrame must not contain Hospital 1's rows — this is the fundamental FL data isolation guarantee |

**Example — what the isolation test does:**

```python
def test_cross_client_isolation(tmp_db, fl_training_rows):
    fl_training_rows(client_id=0, n=30)   # Hospital 0: 30 rows
    fl_training_rows(client_id=1, n=40)   # Hospital 1: 40 rows
    
    df0 = get_fl_client_data(conn=tmp_db, client_id=0)
    df1 = get_fl_client_data(conn=tmp_db, client_id=1)
    
    assert len(df0) == 30   # Hospital 0 gets exactly its 30 rows
    assert len(df1) == 40   # Hospital 1 gets exactly its 40 rows
    
    # The DataFrames share no rows
    assert not df0.index.intersection(df1.index).any()
```

### 1C — `fl_training` Table Schema and Triggers (Tests 16–19)

Unlike the ClinIQ tables (`patient_vitals`, etc.), the `fl_training` table is created dynamically by pandas' `to_sql()` during migration — it does not have a pre-written `CREATE TABLE` statement. These tests verify the table is created with the right structure.

| # | Test | Why it matters |
|---|---|---|
| 16 | `fl_training` table created when populated via pandas `to_sql` | Verifies that the migration's `df.to_sql('fl_training', conn)` call actually creates the table with the right columns |
| 17 | `client_id` index (`idx_ft_client`) exists after first insert | `get_fl_client_data` uses `WHERE client_id = ?`. Without an index, this query scans ALL 48,150 rows for every client data load — 3× per training round × 100 rounds = 300 full table scans. The index makes this query ~100× faster |
| 18 | `created_at` timestamp auto-set on INSERT (trigger) | The `trg_ft_created_at` trigger automatically sets the timestamp. Without it, every insert would need to manually include the current time |
| 19 | `init_db()` safely skips `fl_training` trigger if table absent | `init_db()` is called before migration. If the trigger creation references the `fl_training` table and the table doesn't exist yet, SQLite would throw an error. The `IF NOT EXISTS` guard prevents this |

### 1D — `get_conn()` Extended (Tests 20–21)

| # | Test | Why it matters |
|---|---|---|
| 20 | `PRAGMA foreign_keys` is ON | Foreign key constraints are disabled by default in SQLite. The FL schema uses `client_id` as a logical foreign key to ensure data belongs to a valid hospital. Without this PRAGMA, SQLite silently accepts invalid values |
| 21 | Multiple connections don't corrupt (WAL mode) | In the real system, the Flower server and all 3 client processes open the database simultaneously. WAL (Write-Ahead Logging) allows concurrent readers while a writer is active — without it, the second process to open the DB would get "database is locked" errors, crashing the FL round |

---

## 7. Phase 2 — Model Utilities (`test_model_utils.py`)

**Goal:** Test every function in `model_utils.py` as pure Python with small synthetic data. No database, no file I/O. Uses the `mini_model` fixture (10-dimensional input instead of 108).

This is the largest single test file — 87 tests across 11 sub-sections.

### 2A — `ICUModel` Architecture Tests (Tests 1–9)

`ICUModel` is the PyTorch DNN. These tests verify its architecture is exactly correct.

```python
class ICUModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        return self.net(x)
```

| # | Test | Why this specific check matters |
|---|---|---|
| 1 | `ICUModel(10).forward(x)` returns tensor of shape `(n, 1)` | The output shape `(n, 1)` not `(n,)` is critical — the loss function expects `(batch_size, 1)` predictions and `(batch_size, 1)` targets. A shape mismatch causes a broadcasting error during loss computation |
| 2 | `ICUModel(108).forward(x)` works with real input dim | Verifies that the production 108-dimensional model can do a forward pass without shape errors |
| 3 | Architecture: 4 Linear layers + 3 ReLU in `.net` | Documents the intended architecture. If someone accidentally adds a Dropout or LayerNorm (which we know are incompatible with FL — explained in MODEL_DISCUSSION.md), this test fails |
| 4–7 | Layer dimensions: `(input→128)`, `(128→64)`, `(64→32)`, `(32→1)` | Each dimension is load-bearing. If anyone changes `128` to `64` to "speed up training," the model's weights become incompatible with the saved `federated_model.pth`, causing a shape mismatch when loading |
| 8 | Eval mode outputs deterministic values | `model.eval()` ensures no stochastic behaviour (e.g., Dropout — though we have none). Verifies that the same input always produces the same output in inference mode |
| 9 | Parameter count matches architecture formula | For `input_dim=10`: `(10×128+128) + (128×64+64) + (64×32+32) + (32×1+1) = 1408 + 8256 + 2080 + 33 = 11,777`. For `input_dim=108`: `(108×128+128) + ... ≈ 22,833`. This locks in the exact model size |

**Why the parameter count test matters:**

When Flower serialises model weights for transmission between client and server, it sends flat numpy arrays. If the server's model and the client's model have different parameter counts (because one was changed), `set_weights()` fails with a shape mismatch. The parameter count test immediately catches any architecture change that would break FL communication.

### 2B — `ICUDataset` (Tests 10–13)

`ICUDataset` is the PyTorch Dataset that wraps the numpy X and y arrays for use with a DataLoader.

```python
class ICUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

| # | Test | What it verifies |
|---|---|---|
| 10 | `len(dataset)` == number of samples | DataLoader uses `len()` to compute the number of batches per epoch |
| 11 | `dataset[0]` returns `(X_tensor, y_tensor)` tuple | DataLoader calls `dataset[i]` for each sample — must return a 2-tuple |
| 12 | X tensor dtype is `float32` | PyTorch requires matching dtypes. The model weights are `float32`; if X were `float64`, the matrix multiplication would fail or implicitly convert, losing precision |
| 13 | y tensor shape is `(1,)` not `()` | `.view(-1, 1)` is critical. Without it, y has shape `(batch_size,)`. The loss function computes `predictions - targets` where predictions have shape `(batch_size, 1)` and targets `(batch_size,)` — PyTorch broadcasts this incorrectly, producing a `(batch_size, batch_size)` matrix instead of a `(batch_size, 1)` vector |

**The shape bug explained:**

```
Without .view(-1, 1):
  predictions shape: (64, 1)
  targets shape: (64,)
  
  PyTorch broadcasts: (64, 1) - (64,) → (64, 64) matrix!
  Instead of element-wise differences, you get:
    pred[0]-target[0], pred[0]-target[1], ..., pred[0]-target[63]
    pred[1]-target[0], pred[1]-target[1], ..., pred[1]-target[63]
    ...
  
  Loss becomes an average over 64×64=4096 numbers, most of which are wrong.
  The model "trains" but learns nothing useful — loss never decreases.

With .view(-1, 1):
  predictions shape: (64, 1)
  targets shape: (64, 1)
  (64, 1) - (64, 1) → (64, 1) ← correct: one difference per sample
```

### 2C — `weighted_mse_loss()` (Tests 14–18)

The custom loss function that gives more training weight to high-SOFA patients:

```python
def weighted_mse_loss(predictions, targets):
    weights = 1.0 + targets * 0.5     # SOFA=0 → weight=1.0, SOFA=24 → weight=13.0
    weights = weights / weights.mean() # Normalize so mean weight = 1.0
    squared_errors = (predictions - targets) ** 2
    return (weights * squared_errors).mean()
```

| # | Test | What it verifies | Why it matters |
|---|---|---|---|
| 14 | Perfect predictions → loss == 0 | Identity case | If this fails, the loss function has a bug that causes non-zero loss even when correct |
| 15 | Higher SOFA targets get larger weights | Core property | High-SOFA patients must influence training more heavily than low-SOFA patients |
| 16 | `weight = 1 + target * 0.5` before normalization | Formula correctness | This exact formula was chosen to balance sensitivity — locks it in against accidental changes |
| 17 | Loss is positive for non-zero error | Positivity | Negative loss would cause the optimizer to minimise in the wrong direction |
| 18 | Loss is batch-mean normalized | Normalization check | Without normalization, a batch of high-SOFA patients would have ~13× higher loss than a low-SOFA batch, causing wildly variable gradient magnitudes across rounds |

**Worked example — why weighting helps:**

```
Without weighting (plain MSE):
  Patient A: true SOFA=2,  predicted=3   → error²=1   → gradient pulls model toward SOFA=2
  Patient B: true SOFA=20, predicted=12  → error²=64  → gradient pulls model toward SOFA=20
  Average: Patient B gets 64× more influence than Patient A (proportional to error²)
  
  BUT: Patient B is one of the rare 6% high-SOFA patients.
  Over a whole batch: 57 Low-Risk patients dominate, 3 High-Risk patients are drowned out.
  
With weighting (custom):
  Patient A: weight = 1 + 2×0.5 = 2.0
  Patient B: weight = 1 + 20×0.5 = 11.0
  
  Patient B now has 5.5× more explicit weight than Patient A.
  Combined with the error² term: 5.5 × (64/1) = 352× more influence.
  High-SOFA patients drive learning far more aggressively.
```

### 2D — `get_sample_weights()` (Tests 19–23)

This function provides sample weights for `WeightedRandomSampler` — it controls how often each sample is drawn during mini-batch creation.

```python
def get_sample_weights(y):
    weights = np.where(y < 5, 1.0,
              np.where(y < 10, 2.0, 5.0))
    return weights.astype(np.float32)
```

| # | Test | What it verifies |
|---|---|---|
| 19 | Low SOFA (< 5) → weight == 1.0 | Base sampling rate for majority class |
| 20 | Moderate SOFA (5–9) → weight == 2.0 | Drawn 2× more often than Low-Risk |
| 21 | High SOFA (≥ 10) → weight == 5.0 | Drawn 5× more often than Low-Risk |
| 22 | Output length equals input length | Each sample gets exactly one weight |
| 23 | Output dtype is `float32` | PyTorch's `WeightedRandomSampler` requires float32 |

**Why these exact weight values (1.0, 2.0, 5.0)?**

The dataset has roughly 70% Low / 24% Moderate / 6% High. With uniform sampling, a 64-sample batch contains on average:
- 45 Low-Risk patients
- 15 Moderate patients  
- 4 High-Risk patients

With weights 1/2/5:
- Effective probability: Low=70×1=70, Moderate=24×2=48, High=6×5=30
- Normalized: Low=47%, Moderate=32%, High=20%
- A 64-sample batch now has: 30 Low, 21 Moderate, 13 High patients

This dramatically increases the High-Risk representation in each batch without removing Low-Risk patients entirely.

### 2E — `train_model()` (Tests 24–34)

This is the core training function — it runs one epoch of gradient descent on a hospital's local data.

```python
def train_model(model, X, y, optimizer, global_params=None, mu=0.0,
                oversample=False, grad_clip=None):
    # Setup DataLoader (with or without WeightedRandomSampler)
    # For each batch:
    #   Forward pass → predictions
    #   Compute weighted MSE loss
    #   If FedProx (mu > 0): add proximal term
    #   Backward pass → gradients
    #   If gradient clipping: clip
    #   Optimizer step
    # Return: mean batch loss
```

| # | Test | Why this specific case is tested |
|---|---|---|
| 24 | Returns a `float` | The return value is used to track training progress — must be a Python float, not a PyTorch tensor |
| 25 | Returned loss is finite (not NaN/inf) | NaN loss means gradient explosion. The test catches this immediately — without it, NaN propagates silently through 100 training rounds |
| 26 | Loss decreases from epoch 1 to epoch 5 on same data | Verifies that gradient descent is actually working. If this fails, the optimizer, loss function, or model forward pass is broken |
| 27–28 | Works with `global_params=None, mu=0` AND with `global_params=weights, mu=0.5` | Tests both code paths — FedAvg (no proximal term) and FedProx (with proximal term). Both must work |
| 29–30 | Works with `oversample=True` AND `oversample=False` | Tests both DataLoader configurations |
| 31–32 | Works with `grad_clip=1.0` AND `grad_clip=None` | Tests both gradient clipping configurations |
| 33 | `lr=1e-5` still trains without NaN | The LR decay formula can reduce LR to very small values over 100 rounds. At `lr=1e-5`, gradient updates are tiny but must not underflow to NaN |
| 34 | FedProx proximal term is added to loss when `mu > 0` | The FedProx term `(mu/2) × ||w - w_global||²` must visibly increase the loss compared to no proximal term on the same data. This verifies the term is actually being added, not silently skipped |

**The FedProx proximal term test explained:**

```python
def test_fedprox_proximal_term_added(mini_model, synthetic_client_data):
    X, y = synthetic_client_data
    optimizer = torch.optim.AdamW(mini_model.parameters())
    
    global_weights = get_weights(mini_model)   # initial weights as "global"
    
    # Train WITHOUT FedProx
    loss_no_prox = train_model(mini_model, X, y, optimizer,
                               global_params=None, mu=0.0)
    
    # Reset model weights
    set_weights(mini_model, global_weights)
    
    # Train WITH FedProx (proximal term pulls weights toward global)
    loss_with_prox = train_model(mini_model, X, y, optimizer,
                                 global_params=global_weights, mu=0.5)
    
    # With proximal term: loss includes both MSE and the proximity penalty
    assert loss_with_prox > loss_no_prox   # proximal term adds a positive penalty
```

### 2F — `evaluate_model()` (Tests 35–42)

```python
def evaluate_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor)
    
    mse = mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    r2  = r2_score(y_val, preds)
    return {"mse": mse, "mae": mae, "r2": r2}
```

| # | Test | The mathematical invariant being verified |
|---|---|---|
| 35 | Returns dict with keys `mse`, `mae`, `r2` | The Flower `evaluate` method passes these to the server for logging |
| 36 | All values are Python `float` | Flower serialises metrics as floats for transmission — PyTorch tensors or numpy scalars would fail |
| 37 | MSE is non-negative | MSE = mean of squared errors. Squared numbers are always ≥ 0. If this fails, the formula is wrong |
| 38 | MAE is non-negative | MAE = mean of absolute errors. Absolute values are always ≥ 0. Same reasoning |
| 39 | R² ≤ 1.0 always | R² = 1 − (SS_res / SS_tot). Since SS_res ≥ 0, R² ≤ 1. Can be negative (model worse than mean), but never > 1 |
| 40 | Perfect predictions → MAE == 0, R² == 1.0 | The identity case — verifies all three metrics compute correctly for the trivial case |
| 41 | Constant predictions → R² ≤ 0 | If the model always predicts the mean SOFA (a degenerate model), R² = 0. If it always predicts ANY constant, R² ≤ 0. This test catches a bug where R² is erroneously positive for bad models |
| 42 | Model set to eval mode | `model.eval()` is needed so that if Dropout or BatchNorm ever get added, they don't interfere with evaluation. Calling `evaluate_model` with a training-mode model could give different results on each call |

**The constant predictions (degenerate) test explained:**

```python
def test_constant_predictions_r2_leq_zero(mini_model, synthetic_client_data):
    X, y = synthetic_client_data
    
    # Bias all output biases to produce constant predictions
    with torch.no_grad():
        mini_model.net[-1].bias.fill_(5.0)     # always predict SOFA=5
        mini_model.net[-1].weight.fill_(0.0)   # ignore all input
    
    metrics = evaluate_model(mini_model, X, y)
    assert metrics['r2'] <= 0   # a model that ignores inputs is useless
```

### 2G — `compute_sofa_risk_metrics()` (Tests 43–58)

This is the most comprehensive single function test in the entire test suite — 16 tests covering return structure, mathematical invariants, edge cases, and input validation.

`compute_sofa_risk_metrics(y_true, y_score)` takes actual SOFA scores and predicted SOFA scores, and computes the full clinical performance report shown in the SOFA Validation Drawer.

**Return structure tests (43–46):**

```
risk_bands:
  labels: ["Low (<5)", "Moderate (5-<10)", "High (>=10)"]
  confusion_matrix: 3×3 list of lists
  per_class: {Low: {...}, Moderate: {...}, High: {...}}
  macro: {precision, recall, f1, ...}

high_risk_alert:
  predicted_sofa_threshold: 8.0
  sensitivity: ...
  specificity: ...
```

| # | Test | Why this exact structure |
|---|---|---|
| 43 | Returns dict with `risk_bands` and `high_risk_alert` keys | Two-level structure: classification metrics and alert-threshold metrics |
| 44 | `risk_bands.labels` in exact order: Low, Moderate, High | The confusion matrix rows/columns must correspond to this label order. Wrong order means a "High" row is labelled "Low" |
| 45 | `confusion_matrix` is 3×3 list | Three classes = 3×3 matrix. Wrong dimensions mean a class was merged or added |
| 46 | Per-class dict has all 5 keys: `precision`, `recall`, `sensitivity`, `specificity`, `f1` | All 5 are displayed in the SOFA Validation Drawer |
| 47 | `sensitivity == recall` for all classes | `sensitivity` and `recall` are the same metric (TP / (TP + FN)). They are aliased in the return value — the test verifies the alias is consistent |
| 48 | `high_risk_alert.predicted_sofa_threshold == 8.0` | Locks in the ALERT_THRESHOLD=8.0 value. If anyone changes this threshold without understanding why it was chosen (explained in INPUT_OUTPUT.md), this test fails immediately |

**Mathematical invariant tests (49–51, 55–58):**

| # | Test | The invariant |
|---|---|---|
| 49 | AUROC is `float` when both classes present | AUROC requires at least one positive and one negative sample to be defined |
| 50 | AUROC is `None` when only one class in `y_true` | If all patients are Low Risk, AUROC is undefined (can't rank positives vs negatives). `None` is the correct sentinel — not 0.0, not NaN |
| 51 | PR-AUC is `None` when only one class present | Same reasoning — Precision-Recall curve requires both classes |
| 55 | Predictions clipped to `[0, 24]` before band assignment | A predicted SOFA of 24.3 should be assigned to the "High" band, not create an out-of-range error. `np.clip` ensures this |
| 57 | Confusion matrix rows sum to actual class counts | A 3×3 confusion matrix must have each row summing to the true count of that class. If row sums don't match class counts, a sample was counted twice or not counted |
| 58 | TP + TN + FP + FN == len(y_true) | Every sample must be assigned to exactly one cell in the confusion matrix. If this doesn't hold, a sample is missing or double-counted |

**Input validation tests (52–54):**

| # | Test | Why fail-fast is correct |
|---|---|---|
| 52 | Raises `ValueError` for empty inputs | An empty y_true would produce division by zero in recall computation. Raising immediately with a clear message is better than propagating NaN |
| 53 | Raises `ValueError` when lengths differ | y_true[i] and y_score[i] must correspond to the same patient. Different lengths mean the pairing is broken |
| 54 | Raises `ValueError` for NaN/inf values | NaN SOFA scores propagate into all metrics, producing NaN everywhere. Catching it early gives a diagnostic error instead of a silent all-NaN result |

### 2H — `get_weights()` and `set_weights()` (Tests 59–64)

These two functions are the bridge between PyTorch and Flower. Flower transmits model parameters as lists of numpy arrays; PyTorch models store them as tensors. These functions convert between the two formats.

```python
def get_weights(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]

def set_weights(model, weights):
    state_dict = dict(zip(model.state_dict().keys(),
                         [torch.tensor(w) for w in weights]))
    model.load_state_dict(state_dict, strict=True)
```

| # | Test | Why it matters |
|---|---|---|
| 59–60 | Returns a list of numpy arrays | Flower's `NDArrays` type is literally `List[np.ndarray]`. Wrong type = Flower communication failure |
| 61 | Length equals number of parameter tensors | The state_dict has 8 entries (4 weight matrices + 4 bias vectors). Any mismatch means a layer was missed |
| 62–63 | Round-trip: `set_weights(model, get_weights(model))` → identical state | The most important test: serialize → deserialize must produce the same model. If this fails, every FL round corrupts the model slightly |
| 64 | `set_weights` with wrong layer count raises | If the server sends weights from a different architecture, the client must raise an error, not silently load wrong shapes |

**The round-trip test explained:**

```python
def test_weights_round_trip(mini_model):
    original_weights = get_weights(mini_model)
    
    # Simulate Flower transmission: weights become numpy arrays,
    # go through the network, arrive at another client
    received_weights = [w.copy() for w in original_weights]
    
    # Load received weights back into the model
    set_weights(mini_model, received_weights)
    
    # Get weights again
    restored_weights = get_weights(mini_model)
    
    # Must be bit-for-bit identical
    for orig, restored in zip(original_weights, restored_weights):
        npt.assert_array_equal(orig, restored)
```

### 2I — `get_trend()` and `classify_range()` (Tests 65–75)

These are pure utility functions used in the ClinIQ dashboard's trend analysis (Step 10 of the inference pipeline, documented in INPUT_OUTPUT.md).

**`get_trend()` — linear regression slope classifier:**

The function fits a linear regression line to the list of values and classifies the slope:

| # | Test | Edge case it covers |
|---|---|---|
| 65 | `[1,2,3,4,5]` → `"increasing"` | Clear positive slope |
| 66 | `[5,4,3,2,1]` → `"decreasing"` | Clear negative slope |
| 67 | `[3,3,3,3]` → `"stable"` | Zero slope |
| 68 | `[5]` → `"stable"` (single element) | A single point has undefined slope — must default to "stable" not crash |
| 69 | `[]` → `"stable"` (empty) | Empty list — must not crash with "division by zero" in linear regression |
| 70 | Slope exactly 0.1 → `"stable"` (boundary) | The threshold is `> 0.1` for "increasing", not `≥ 0.1`. Exactly 0.1 is "stable". This boundary test locks in the comparison operator |

**`classify_range()` — vital sign status classifier:**

| # | Test | Boundary being tested |
|---|---|---|
| 71 | `classify_range(70, 60, 100)` → `"normal"` | 70 is strictly between 60 and 100 |
| 72 | `classify_range(120, 60, 100)` → `"high"` | 120 exceeds the high boundary |
| 73 | `classify_range(40, 60, 100)` → `"low"` | 40 is below the low boundary |
| 74 | `classify_range(60, 60, 100)` → `"normal"` (not low) | Exactly AT the lower bound = normal, not low |
| 75 | `classify_range(100, 60, 100)` → `"normal"` (not high) | Exactly AT the upper bound = normal, not high |

Tests 74 and 75 are boundary tests that clarify the comparison operators used (closed bounds → `≥ low_val AND ≤ high_val`). Without these, a developer could accidentally use strict inequalities and misclassify patients at the exact boundary of the normal range.

### 2J — `apply_dp_to_update()` (Tests 76–81)

Differential Privacy is the optional mechanism that adds calibrated Gaussian noise to weight updates before sending them to the server, preventing the server from inferring individual patient data from weight changes.

```python
def apply_dp_to_update(local_weights, global_weights, sensitivity, sigma):
    # Compute update: delta = local - global
    delta = [l - g for l, g in zip(local_weights, global_weights)]
    
    # Compute L2 norm of the update
    total_norm = np.sqrt(sum(np.sum(d**2) for d in delta))
    
    # Clip: if update is too large, scale it down
    clip_factor = min(1.0, sensitivity / (total_norm + 1e-8))
    clipped_delta = [d * clip_factor for d in delta]
    
    # Add Gaussian noise
    noised_delta = [d + np.random.normal(0, sigma * sensitivity, d.shape)
                    for d in clipped_delta]
    
    # Apply noised update to global weights
    return [g + d for g, d in zip(global_weights, noised_delta)]
```

| # | Test | What it verifies |
|---|---|---|
| 76 | Returns list of arrays same shape as `local_weights` | DP must not change the shape of the weight update — the server expects exact weight arrays |
| 77 | When update L2 norm ≤ sensitivity, no clipping applied | If the update is already small enough, the `clip_factor = min(1.0, ...)` = 1.0, meaning no change |
| 78 | When update L2 norm > sensitivity, clipped to sensitivity | Verifies that large updates are scaled down correctly |
| 79 | Gaussian noise is added (output differs from input + zero noise) | The core DP guarantee — even if sigma is small, the output must differ from the un-noised update |
| 80 | Output dtype is `float32` | The noised weights go directly into Flower's transmission format — must be float32 |
| 81 | With `sigma=0`, noise std is 0 (no noise) | `sigma=0` is a special case meaning "no DP". The function must handle this gracefully without dividing by zero or adding NaN |

### 2K — `estimate_privacy_budget()` (Tests 82–87)

The privacy budget ε (epsilon) quantifies how much privacy is lost per round of FL. Smaller ε = more privacy.

```
ε = (√(2 × ln(1.25 / δ)) / σ) × rounds
```

Where `σ` is the noise multiplier and `δ` is the failure probability (typically 1e-5).

| # | Test | What it verifies |
|---|---|---|
| 82 | Returns a `float` | Used in the FL dashboard's privacy meter display |
| 83 | `estimate_privacy_budget(0, 1.0)` == 0 | Zero rounds → zero privacy cost (nothing was shared) |
| 84 | 100 rounds, σ=1.0 matches formula | Verifies the formula is implemented correctly — not an approximation |
| 85 | `sigma ≤ 0` returns `inf` | Zero or negative noise means infinite privacy cost (DP guarantee is violated completely). `inf` is the correct sentinel — training with σ=0 offers no privacy |
| 86 | Higher sigma → lower epsilon | More noise = better privacy = smaller ε. This is the fundamental DP monotonicity property |
| 87 | More rounds → higher epsilon | Each round leaks a bit more. After 100 rounds, ε is larger than after 10 rounds. Privacy degrades with training time |

---

## 8. Phase 3 — FL Dashboard Pure Functions (`test_fl_dashboard_fns.py`)

**Goal:** Test every extractable pure function from `fl_dashboard.py` without launching Streamlit.

**The challenge with Streamlit:**

`fl_dashboard.py` imports Streamlit at the top: `import streamlit as st`. When pytest tries to import this module, Streamlit initialises its server, tries to connect to the Streamlit runtime, and either crashes or blocks indefinitely.

**The solution — mock before import:**

```python
# At the top of the test file — before any imports of fl_dashboard
import sys
import unittest.mock as mock

# Create a fake streamlit module
fake_st = mock.MagicMock()
fake_st.cache_resource = lambda f: f   # passthrough — don't cache, just call
sys.modules['streamlit'] = fake_st

# NOW it's safe to import
from fl_dashboard import _blank_state, _reset_state, _fedavg, _fedyogi, ...
```

After this setup, every `st.write()`, `st.sidebar.slider()`, etc. in `fl_dashboard.py` calls the MagicMock, which accepts any arguments and returns a mock object — no crash, no output.

### 3A — State Management (Tests 1–11)

The FL simulation state is a Python dictionary stored in a module-level variable `_STATE`. All functions read and write this dict through controlled functions.

**`_blank_state()` tests (1–6):**

| # | Test | Why this initial value matters |
|---|---|---|
| 1 | `status == "idle"` | The UI shows "idle" as the initial state — any other value would show a running or error indicator on load |
| 2 | `current_round == 0` | Round 0 means no rounds have run. The round counter drives the chart x-axis |
| 3 | `total_rounds == 10` | The default simulation runs 10 rounds. This is intentionally much less than the real 100 rounds — long enough for convergence trends to appear, short enough for a live demo |
| 4 | `client_losses == [[], [], []]` | Three empty lists — one per hospital. The list grows by one entry per round as each hospital's per-round loss is recorded |
| 5 | `best_loss == float("inf")` | Infinity is the correct initial "worst possible" value — any real loss will be less than infinity, so the first round always becomes the "best" |
| 6 | Has all 20 required keys | Completeness check — if a new key is added to `_blank_state()` but not to this test, the test needs updating. Prevents silently missing state fields |

**`_reset_state()` tests (7–9):**

`_reset_state(n_rounds, algorithm)` is called when the clinician clicks "Start Training" with new settings.

```python
def test_reset_clears_previous_losses():
    _reset_state(5, "FedAvg")
    _STATE["global_losses"].append(1.5)   # simulate one round
    _reset_state(5, "FedAvg")              # reset again
    assert _read_state()["global_losses"] == []   # must be empty again
```

If `_reset_state` doesn't clear `global_losses`, the second training run's chart would show the previous run's loss values before the new ones, confusing the clinician.

**`_read_state()` thread safety tests (10–11):**

`fl_thread` modifies `_STATE` from a background thread. The UI thread reads `_STATE` via `_read_state()`. If `_read_state()` returns a reference to `_STATE` itself (not a copy), the UI could see partial updates mid-modification — values from two different rounds simultaneously.

```python
def test_read_state_returns_copy():
    snapshot = _read_state()
    snapshot["status"] = "mutated"   # mutate the snapshot
    assert _read_state()["status"] != "mutated"   # original unchanged
```

If `_read_state()` returns the dict directly (not a copy), this test fails — mutations to the snapshot would corrupt `_STATE`. The deep copy guarantee is essential for thread-safe rendering.

### 3B — `_fedavg()` (Tests 12–17)

`_fedavg` implements Federated Averaging: combine the 3 hospitals' local weight updates into one global update by taking a weighted average proportional to each hospital's sample count.

```python
def _fedavg(client_weights, client_sizes):
    total = sum(client_sizes)
    weighted = [s/total for s in client_sizes]   # e.g., [0.33, 0.33, 0.34]
    
    # For each layer, compute the weighted average of all 3 clients' weights
    aggregated = []
    for layer_idx in range(len(client_weights[0])):
        layer_avg = sum(w * client_weights[i][layer_idx]
                        for i, w in enumerate(weighted))
        aggregated.append(layer_avg.astype(np.float32))
    
    return aggregated
```

| # | Test | Why it matters |
|---|---|---|
| 12 | Equal sample counts → arithmetic mean | With equal sizes (33/33/34), the average is simply the arithmetic mean of the 3 weight arrays |
| 13 | Different sample counts → weighted average | A hospital with 40,000 patients should influence the global model more than a hospital with 8,000 patients. This verifies proportional weighting |
| 14 | Single client → returns that client's weights unchanged | Trivial case but must not crash — used when only 1 hospital has data |
| 15 | Output shape matches input shape per layer | FedAvg must never change the number of layers or the dimensions of any layer |
| 16 | Output dtype is `float32` | The aggregated weights go directly into the global model via `set_weights()` |
| 17 | Result is between min and max of inputs | FedAvg is an average — it cannot produce values outside the range of its inputs. If this fails, the weighting formula has an error |

**The weighted average test explained:**

```python
def test_fedavg_weighted_by_sample_count():
    # Hospital 0: weights all 1.0, sample size 80
    # Hospital 1: weights all 3.0, sample size 20
    # Hospital 2: weights all 5.0, sample size 0 (not participating)
    
    # Expected: (80×1.0 + 20×3.0) / (80+20) = (80+60)/100 = 1.4
    
    client_weights = [
        [np.array([1.0, 1.0])],  # Hospital 0
        [np.array([3.0, 3.0])],  # Hospital 1
    ]
    client_sizes = [80, 20]
    
    result = _fedavg(client_weights, client_sizes)
    npt.assert_allclose(result[0], [1.4, 1.4], rtol=1e-5)
```

### 3C — `_fedyogi()` (Tests 18–25)

`_fedyogi` is more complex than `_fedavg`. It implements the FedYogi adaptive optimisation strategy:

```
delta    = global_weights - averaged_client_weights   (pseudo-gradient)
m_t      = β₁ × m_{t-1} + (1-β₁) × delta            (momentum)
v_t      = v_{t-1} - (1-β₂) × delta² × sign(v_{t-1} - delta²)  (Yogi variance)
new_w    = global_weights + η × m_t / (√v_t + τ)
```

Key test for numerical stability (Test 21):

```python
def test_fedyogi_v_t_stays_above_tau_squared():
    TAU = 1e-3   # stability constant
    _, _, v_t = _fedyogi(client_weights, client_sizes, global_weights,
                         m_t=None, v_t=None)
    
    for v_layer in v_t:
        # v_t is used as the denominator: √v_t + τ
        # If v_t → 0, the denominator → τ (safe)
        # But v_t must never be negative (√ of negative = NaN)
        assert np.all(v_t >= TAU**2)
```

The `np.maximum(v_t, tau²)` guard in the implementation prevents v_t from going negative due to floating-point cancellation. This test verifies the guard is in place — without it, a single round with near-zero gradients could produce NaN in all subsequent rounds.

Test 23 (zero gradient test):

```python
def test_fedyogi_zero_delta_no_update():
    # When all client weights equal global weights: delta = 0
    # FedYogi should produce unchanged global weights
    same_weights = [global_weights.copy(), global_weights.copy(), global_weights.copy()]
    new_weights, _, _ = _fedyogi(same_weights, [33,33,34], global_weights, None, None)
    npt.assert_allclose(new_weights[0], global_weights[0], rtol=1e-5)
```

If clients learned nothing (their weights stayed at the global weights), FedYogi should not change the global model. This verifies the algorithm doesn't introduce drift when there's nothing to learn.

### 3D — `load_training_metadata()` (Tests 26–32)

This function loads `models/training_metadata.json` if it exists, or returns default fallback values if it doesn't.

```python
def load_training_metadata():
    try:
        with open("models/training_metadata.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "num_rounds": 100,
            "final_mae": 1.8968,
            "final_r2": 0.5995,
            "hospitals": 3,
            ...
        }
```

Tests 27–30 lock in the fallback values. If someone runs the FL training again and the new model has worse performance, they might update `training_metadata.json` — but the fallback values (shown when no file exists) should remain as the documented best model performance. The tests prevent silently changing these reference values.

### 3E — Chart Builder Tests (Tests 33–41)

The chart functions build Plotly Figure objects for the training curves, weight histograms, and weight heatmaps. Each has a "null/empty input → None" test and a "valid input → Figure" test.

**The null guard tests (33, 36, 39) are important:**

When the FL simulation first starts, no training data exists yet. The UI code calls all three chart functions at initial render time. If any chart function crashes on empty input, the entire Streamlit page crashes with an error on load.

```python
def test_training_chart_returns_none_for_empty_data():
    result = _training_chart([], [], [])   # no rounds yet
    assert result is None   # UI checks for None and shows "No data yet" message
```

**The heatmap slice test (Test 41):**

```
_heatmap_chart slices: weights.T[:40, :64]
```

The weight heatmap shows the first layer's weight matrix as a colour grid. The full first layer matrix is `128 × 108`. Plotting all 13,824 pixels would be slow — the slice takes the first 40 neurons × first 64 features as a representative sample. The test verifies the correct slice is used — if the dimensions are wrong, the heatmap shows data from the wrong layer or wrong neurons.

### 3F — `_network_diagram()` (Tests 42–50)

`_network_diagram()` generates an HTML string containing an animated network diagram showing the FL communication in real time. The diagram shows 3 hospital nodes connected to a central server, with animated dots flowing up (aggregating) or down (distributing weights).

| # | Test | Why this is tested |
|---|---|---|
| 43 | Contains `<!DOCTYPE html>` | A valid HTML document that the browser can render — not a partial HTML fragment |
| 44 | `"distributing"` → `dotDown` in HTML | When the server sends weights to hospitals, dots animate downward |
| 45 | `"aggregating"` → `dotUp` in HTML | When hospitals send weights to server, dots animate upward |
| 46 | `"idle"` → no `dotDown`/`dotUp` | Static display when not training |
| 47 | `"done"` → `best_round` number in HTML | After training completes, the diagram shows which round was the best |
| 50 | `best_loss=inf` → displays `0.0` not `inf` | When `best_loss` is still infinity (no rounds completed), displaying "inf" in the HTML would look like a bug. The function should show `0.0` as the initial state |

---

## 9. Phase 4 — Client Logic (`test_client_logic.py`)

**Goal:** Test `HospitalClient` from `client.py` — the class that each hospital's process runs. Tests use mocked Flower internals and DB access.

**What `HospitalClient` does:**

```python
class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        # Load this hospital's data from DB
        df = get_fl_client_data(client_id)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(df, ...)
        self.model = ICUModel(108)
    
    def get_parameters(self, config):
        return get_weights(self.model)
    
    def fit(self, parameters, config):
        set_weights(self.model, parameters)   # receive global weights
        server_round = config["server_round"]
        lr = max(1e-4, 0.001 * LR_DECAY ** (server_round - 1))  # LR decay
        optimizer = AdamW(self.model.parameters(), lr=lr)
        loss = train_model(self.model, self.X_train, self.y_train, optimizer,
                           global_params=parameters, mu=MU_FEDPROX)
        return get_weights(self.model), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        metrics = evaluate_model(self.model, self.X_val, self.y_val)
        return metrics["mse"], len(self.X_val), {"mae": metrics["mae"], "r2": metrics["r2"]}
```

**The LR decay tests (7–9) — why these exact values:**

```
LR at round 1:    max(1e-4, 0.001 × 0.99⁰)  = max(1e-4, 0.001)    = 0.001
LR at round 50:   max(1e-4, 0.001 × 0.99⁴⁹) = max(1e-4, 0.000607) = 0.000607
LR at round 1000: max(1e-4, 0.001 × 0.99⁹⁹⁹) = max(1e-4, ≈0.00000)= 0.0001 (floor)
```

The LR decay floor (`1e-4`) prevents learning rate from going to zero, which would stop all learning. The tests lock in the exact formula values — if anyone changes `LR_DECAY = 0.99` to `0.95` for faster decay, the round-50 test immediately fails.

**The DP branch tests (10–11):**

```python
def test_dp_disabled_branch(mocker):
    mock_dp = mocker.patch('client.apply_dp_to_update')
    client = HospitalClient(client_id=0)
    client.fit(initial_params, {"server_round": 1})
    mock_dp.assert_not_called()   # USE_DP=False → DP function never called

def test_dp_enabled_branch(mocker, monkeypatch):
    monkeypatch.setattr('client.USE_DP', True)   # temporarily enable DP
    mock_dp = mocker.patch('client.apply_dp_to_update', return_value=initial_params)
    client = HospitalClient(client_id=0)
    client.fit(initial_params, {"server_round": 1})
    mock_dp.assert_called_once()  # USE_DP=True → DP function must be called
```

These two tests verify the `if USE_DP:` branch is correctly wired. Without them, a developer could accidentally move `apply_dp_to_update` outside the `if` block (always runs) or delete the `if` block (never runs) without any test catching it.

**Config constant tests (15–19):**

```
MU_FEDPROX == 0.5   (FedProx regularisation strength)
USE_DP == False     (DP disabled by default)
EPOCHS == 3         (local training epochs per round)
BATCH_SIZE == 64    (mini-batch size)
SERVER_ADDR == "127.0.0.1:8080"  (Flower server address)
```

These tests document and lock in the finalized hyperparameters. Each value was chosen after the optimization journey documented in `TRAIN_FEDERATED.md`. If anyone "tries a quick change" like `EPOCHS=5`, the test fails immediately — prompting them to understand why 3 epochs was the correct choice before changing it.

---

## 10. Phase 5 — Server Strategy (`test_server_strategy.py`)

**Goal:** Test `SaveBestStrategy` from `server.py` — the class that runs on the central server, aggregates client weights, and saves model checkpoints.

**The key responsibility of `SaveBestStrategy`:**

1. After each round: call the parent `FedYogi` aggregation (`super().aggregate_fit`)
2. Set the aggregated weights into the global model
3. Save the aggregated model as `federated_model.pth` (latest checkpoint)
4. After evaluating: if this round's loss < best loss seen so far, save `federated_model_best.pth`
5. Update `_best_round` and `_best_loss`
6. After training: write `training_metadata.json`

**The best-model tracking tests (7–12):**

```
Round 1: avg_loss = 5.2   → 5.2 < inf → new best (save federated_model_best.pth)
Round 2: avg_loss = 4.8   → 4.8 < 5.2 → new best (save federated_model_best.pth)
Round 3: avg_loss = 4.9   → 4.9 > 4.8 → NOT new best (do NOT save best model)
Round 4: avg_loss = 4.7   → 4.7 < 4.8 → new best (save federated_model_best.pth)
```

Test 8 ("Second round with higher loss → `_best_loss` unchanged") is especially important:

```python
def test_higher_loss_does_not_update_best(mocker):
    mock_save = mocker.patch('server.torch.save')
    strategy = SaveBestStrategy(global_model=mock_model)
    
    # Round 1: loss = 5.0 → becomes best
    strategy.aggregate_evaluate(1, [(client1_result, 50), ...])
    
    # Round 2: loss = 6.0 → worse than 5.0
    strategy.aggregate_evaluate(2, [(client1_result_worse, 50), ...])
    
    # torch.save should have been called ONCE (for round 1) not TWICE
    best_save_calls = [c for c in mock_save.call_args_list 
                       if 'best' in str(c)]
    assert len(best_save_calls) == 1   # only round 1 triggered a best save
```

Without this test, a bug could cause the server to save the round-2 (worse) model as the "best" model — overwriting the actual best checkpoint.

**The metadata merge test (Test 18):**

```python
def test_metadata_merge_with_existing_file(mocker, tmp_path):
    # Existing metadata.json already has final_mae and final_r2 from real training
    existing_metadata = {"final_mae": 1.8968, "final_r2": 0.5995, "best_round": 10}
    
    # Write existing file
    (tmp_path / "training_metadata.json").write_text(json.dumps(existing_metadata))
    
    # Now aggregate_evaluate finds a new best at round 24
    strategy.aggregate_evaluate(24, client_results)
    
    # The new metadata must PRESERVE final_mae and final_r2 from the existing file
    new_metadata = json.loads((tmp_path / "training_metadata.json").read_text())
    assert new_metadata["final_mae"] == 1.8968   # not overwritten
    assert new_metadata["best_round"] == 24      # updated
```

The server runs `aggregate_evaluate` after each round, before the final `final_mae` and `final_r2` (which are computed only at the end). If the metadata write doesn't merge with the existing file, it would overwrite `final_mae` and `final_r2` with zeros or None on every round — and the Validation Drawer would show 0.00 for all metrics.

---

## 11. Phase 6 — Train Federated Helpers (`test_train_federated_fns.py`)

**Goal:** Test the standalone `train_federated.py` script's helper functions — especially `make_noniid_split()`, which creates the intentionally skewed data distribution for non-IID federated learning experiments.

### 6A — `make_noniid_split()` (Tests 1–8)

In the real MIMIC-III data, hospitals specialise:
- A cardiac ICU sees mostly cardiovascular patients (lower SOFA variation)
- A trauma ICU sees high-acuity polytrauma patients (high SOFA)
- A general medical ICU sees a mix

`make_noniid_split` creates this specialisation artificially for the non-IID experiment:

```python
def make_noniid_split(X, y, seed=42):
    low_idx  = np.where(y < 5)[0]   # Low-Risk patients
    mod_idx  = np.where((y >= 5) & (y < 10))[0]
    high_idx = np.where(y >= 10)[0]
    
    # Hospital 0: specialises in Low-Risk (cardiac ICU)
    # Hospital 1: balanced mix (general medical ICU)
    # Hospital 2: specialises in High-Risk (trauma ICU)
    ...
```

| # | Test | Why this exactly |
|---|---|---|
| 1 | Returns list of exactly 3 tuples | Three hospitals — not 2, not 4 |
| 2 | Each tuple is `(X_array, y_array)` | The downstream `ICUDataset(X, y)` call requires this exact structure |
| 3 | Total samples across 3 splits == input total | No patients lost or duplicated during the split |
| 4 | Hospital 0 has ≥ 50% Low-SOFA samples | The specialisation bias must be meaningful — if Hospital 0 only has 35% Low-Risk, the non-IID effect is too weak to demonstrate |
| 5 | Hospital 2 has ≥ 50% High-SOFA samples | Same reasoning for the high-acuity specialisation |
| 6 | Same seed → identical splits | Reproducibility — the demo must produce the same splits every time |
| 7 | Different seed → different splits | Verifies the function actually uses the seed — not hardcoded splits |
| 8 | Works with empty High-SOFA class | Edge case: if the synthetic data has no High-SOFA patients (rare but possible with small test datasets), the function must not crash |

### 6B and 6C — HospitalClient and SaveBestStrategy (train variant)

These are nearly identical to the tests in Phases 4 and 5, but for the variants in `train_federated.py`. The key additional test:

**Test 16 — `configure_fit` adds `server_round` to config:**

```python
def test_configure_fit_adds_round_to_config(mocker):
    strategy = SaveBestStrategy(global_model=mock_model, ...)
    fit_instructions = strategy.configure_fit(server_round=5, parameters=..., 
                                               client_manager=...)
    
    # Each client's config dict must contain {"server_round": 5}
    for _, fit_config in fit_instructions:
        assert fit_config.config.get("server_round") == 5
```

Without this, `HospitalClient.fit()` would receive an empty config dict, and `config["server_round"]` would raise a `KeyError`, crashing the entire FL round.

### 6D — Config Constants (Tests 21–30)

```
NUM_ROUNDS == 100
EPOCHS_PER_ROUND == 3
BATCH_SIZE == 64
MU_FEDPROX == 0.5
SERVER_ETA == 0.01
SERVER_BETA1 == 0.9
SERVER_BETA2 == 0.99
USE_DP == False
OVERSAMPLE == False
LR_DECAY == 0.99
```

Every one of these constants was determined through the optimization journey documented in `TRAIN_FEDERATED.md`. The full rationale:

- `SERVER_ETA=0.01`: FedYogi's global learning rate. Too high → the explosion/recovery cycle becomes unrecoverable. Too low → convergence stalls.
- `SERVER_BETA1=0.9`: Momentum decay. Standard Adam value, tuned for the clinical data distribution.
- `SERVER_BETA2=0.99`: Adaptive variance decay. The `0.99` (not `0.999` as in standard Adam) causes faster adaptation to changing gradient statistics between rounds.
- `MU_FEDPROX=0.5`: Proximal weight. At 0.5, drift is reduced by 23% compared to FedAvg without significantly slowing convergence. Values above 1.0 over-constrain learning.

These tests are "configuration documentation" — they fail if anyone "tries" a different value, forcing them to understand the optimization history before making changes.

---

## 12. Phase 7 — Database ↔ Model Integration (`test_db_model_integration.py`)

**Goal:** Test the complete pipeline from `db.get_fl_client_data()` through model training and evaluation. This is the first integration test — it connects the DB layer to the ML layer.

**Why this integration is needed, beyond the unit tests:**

The DB unit tests (Phase 1) verify that `get_fl_client_data()` returns the right DataFrame. The model unit tests (Phase 2) verify that `train_model()` works on synthetic numpy arrays. But neither test verifies that the DataFrame produced by the DB function has the right shape and content to be fed directly into the training pipeline.

A real bug this catches:

> The DB function returns a DataFrame with columns in alphabetical order. The training pipeline's StandardScaler was fitted with columns in a different order. When the alphabetically-ordered DataFrame is normalised, feature 0 gets feature 17's statistics. R² collapses from 0.47 to 0.02. Neither the DB test nor the model test would catch this — only the integration test that runs both together would.

| # | Test | What it verifies |
|---|---|---|
| 1 | `get_fl_client_data(0)` → `ICUDataset` → `len(dataset) > 0` | The DB-produced DataFrame can be wrapped in ICUDataset without errors |
| 2 | X values from DB match what ICUDataset produces | Column alignment check — the feature order is preserved through the DB → DataFrame → ICUDataset chain |
| 3 | `get_fl_client_data` → `train_model` completes without error | The full "load data → train" path works end-to-end |
| 4 | `get_fl_client_data` → `evaluate_model` returns finite metrics | All three metrics (MSE, MAE, R²) are finite real numbers — not NaN/inf |
| 5 | Client_id=1 produces different data than client_id=0 | Data isolation holds through the full stack, not just at the DB level |
| 6 | Model trained on client_0 data has different weights than untrained model | Training actually modifies the model — verifies the optimizer is working through the full pipeline |
| 7 | `get_weights` → `set_weights` → `get_weights` round-trip on DB-trained model | Weight round-trip holds even after training has modified the weights |
| 8 | `compute_sofa_risk_metrics` works with SOFA values from DB | The clinical metrics function accepts real DB-sourced SOFA scores |
| 9 | NaN values from DB filled with 0 before feeding to ICUDataset | The `fillna(0)` in `get_fl_client_data` is applied before the data reaches training |
| 10 | `get_fl_client_data` + `train_test_split` → correct train/val sizes | 80/20 split produces the expected sizes — the DataFrames are compatible with sklearn's split function |

---

## 13. Phase 8 — FL Round Integration (`test_fl_round_integration.py`)

**Goal:** Simulate a complete FL round — distribute global weights to 3 hospitals, have each train locally, aggregate their updates — without using Flower's gRPC infrastructure.

**How this test works (no Flower needed):**

```python
# Step 1: Create global model
global_model = ICUModel(input_dim=10)
global_weights = get_weights(global_model)

# Step 2: Simulate 3 hospitals training in parallel
client_weights_after = []
for client_id in range(3):
    # Each hospital receives the global weights, trains locally
    local_model = ICUModel(input_dim=10)
    set_weights(local_model, global_weights)   # receive from server
    optimizer = AdamW(local_model.parameters())
    train_model(local_model, X_clients[client_id], y_clients[client_id], optimizer)
    client_weights_after.append(get_weights(local_model))

# Step 3: Server aggregates
client_sizes = [len(y) for y in y_clients]
global_weights_round2 = _fedavg(client_weights_after, client_sizes)

# This is one complete FL round without any Flower gRPC
```

The complete test list:

| # | Test | What it verifies |
|---|---|---|
| 1 | 3 hospitals each train → produce 3 weight lists | All 3 training runs complete successfully |
| 2 | `_fedavg` on 3 weight sets → correct layer shapes | Aggregation preserves all layer dimensions |
| 3 | After `_fedavg`, forward pass produces valid output | The aggregated model is a working DNN — not corrupted |
| 4 | `_fedyogi` on 3 weight sets → bounded weights | FedYogi numerical stability holds in practice |
| 5 | Round 2 loss < Round 1 loss | Convergence — the aggregated model improves with each round |
| 6 | `fl_thread` state transitions: idle→running→distributing→training→done | The simulation state machine progresses correctly through all phases |
| 7 | `_read_state()["current_round"]` == 1 after 1 round | State is updated after each round |
| 8 | `_read_state()["global_losses"]` has length 1 | Loss is recorded for each completed round |
| 9 | `_read_state()["client_losses"]` has 1 entry per client | Each hospital's per-round loss is recorded separately |
| 10 | `_STOP.set()` → status becomes "stopped" | The stop signal is respected — simulation can be halted |
| 11 | FedProx (μ=0.5) → smaller weight delta vs FedAvg (μ=0) | FedProx reduces client drift — the key clinical property it was chosen for |
| 12 | `_read_state()["best_round"]` == 1 after single round | The only round is always the best round |
| 13 | `layer1_weights` shape is `(128, input_dim)` | The heatmap visualization gets the first layer's weight matrix in the correct shape |
| 14 | `pre_agg_flat` and `post_agg_flat` have same length | Before/after aggregation histograms must have compatible shapes for comparison |

**The FedProx drift reduction test (Test 11) explained:**

```python
def test_fedprox_reduces_drift(synthetic_splits):
    global_weights = get_weights(ICUModel(10))
    
    # Without FedProx (mu=0.0): hospitals can drift far from global
    fedavg_deltas = []
    for X, y in synthetic_splits:
        local_model = ICUModel(10)
        set_weights(local_model, global_weights)
        train_model(local_model, X, y, optimizer, global_params=None, mu=0.0)
        delta = np.mean([np.mean(np.abs(lw - gw))
                         for lw, gw in zip(get_weights(local_model), global_weights)])
        fedavg_deltas.append(delta)
    
    # With FedProx (mu=0.5): proximal term pulls weights toward global
    fedprox_deltas = []
    for X, y in synthetic_splits:
        local_model = ICUModel(10)
        set_weights(local_model, global_weights)
        train_model(local_model, X, y, optimizer, global_params=global_weights, mu=0.5)
        delta = np.mean([np.mean(np.abs(lw - gw))
                         for lw, gw in zip(get_weights(local_model), global_weights)])
        fedprox_deltas.append(delta)
    
    # FedProx hospitals stay closer to global weights
    assert np.mean(fedprox_deltas) < np.mean(fedavg_deltas)
```

This test directly verifies the clinical motivation for FedProx: with heterogeneous ICU data (Hospital 0 sees mostly Low-Risk, Hospital 2 sees mostly High-Risk), unconstrained local training causes Hospital 2's model to drift heavily toward "always predict High." FedProx constrains this drift, making the aggregated global model more balanced.

---

## 14. Phase 9 — Migration Integration (`test_migrate_to_db.py`)

**Goal:** Test `migrate_to_db.migrate()` — the one-time import that converts all CSV data files into the SQLite database.

**What `migrate()` does:**

```
Input CSV files:                     Output SQLite tables:
  data/patient_vitals.csv       →      patient_vitals
  data/prediction_history.csv   →      prediction_history
  data/sample_patients.csv      →      sample_patients
  data/fl_training_0.csv        →      fl_training (client_id=0)
  data/fl_training_1.csv        →      fl_training (client_id=1)
  data/fl_training_2.csv        →      fl_training (client_id=2)
```

The test uses `tmp_path` to create fake CSV files and a temp database, then verifies the migration populates all tables correctly.

| # | Test | Why this is tested |
|---|---|---|
| 1 | `migrate()` completes without raising | Basic sanity — the function runs without crashing |
| 2–5 | All 4 tables have rows after migration | Each table is populated (not just created empty) |
| 6 | `migrate()` called twice → row counts unchanged | **Idempotency** — if the user accidentally runs the migration script twice, the database must not double all its rows |
| 7–9 | Column renames are applied | SpO₂ → SpO2, Temp (°C) → Temp, GCS Eye → GCS_Eye — the original CSV column names contain Unicode and spaces that SQLite and pandas handle inconsistently |
| 10–11 | `fl_training` rows have correct `client_id` | Each hospital's CSV gets assigned the right client_id (Hospital 0 CSV → client_id=0) |
| 12 | Missing CSV → skipped without crash | Some CSV files might not exist on a fresh install — the migration must skip them gracefully and continue |
| 13 | Empty CSV → skipped without crash | An empty CSV is a degenerate case that must not crash |
| 14 | `patient_vitals` rows have `patient` column values 1, 2, 3 | The patient ID assignment is correct — row counts per patient_id match the input distribution |
| 15 | `fl_training` index `idx_ft_client` exists | The performance index (created after FL data is inserted) must be present for fast query performance |
| 16 | `sample_patients` has `row_idx` column starting at 0 | The cycling mechanism uses `row_idx` to advance through the dataset — must start at 0, not at the CSV row number |
| 17 | `os.path.getsize()` call doesn't crash | The migration prints a summary including the DB file size — must not crash if the file system returns unexpected results |

**The idempotency test explained:**

```python
def test_migrate_idempotent(stub_csvs, tmp_db):
    # Run migration once
    migrate()
    
    # Count rows in each table
    counts_after_first = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ["patient_vitals", "prediction_history", "fl_training"]
    }
    
    # Run migration a second time
    migrate()
    
    # Counts must be identical — no rows added
    counts_after_second = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ["patient_vitals", "prediction_history", "fl_training"]
    }
    
    assert counts_after_first == counts_after_second
```

The implementation achieves idempotency by checking `fl_training_exists()` before inserting FL data, and by using `if_exists='replace'` (with pre-truncation) or checking existing row hashes for other tables. Without this test, a user running the migration script twice would end up with duplicate data — every vital sign reading appearing twice, the dashboard charts showing the same readings back-to-back.

**The column rename tests (7–9) explained:**

The original CSV files exported from the Streamlit version of the app used display-friendly column names:

```
CSV column name → SQLite column name
"SpO₂"         → "SpO2"      (remove Unicode ₂ subscript)
"Temp (°C)"     → "Temp"      (remove unit suffix)
"GCS Eye"       → "GCS_Eye"   (replace space with underscore)
```

The tests verify these renames because the ClinIQ backend's SQL queries use the renamed column names. If the migration fails to rename, every `SELECT SpO2 FROM patient_vitals` would return an error ("no such column: SpO2").

---

## 15. Phase 10 — System Tests (`test_fl_system.py`)

**Goal:** End-to-end FL simulation using the actual `fl_thread` function with real synthetic data. These are the most realistic tests — they run the full simulation loop from `status="idle"` through `status="done"`.

**All tests in this phase are `@pytest.mark.slow`.** A 1-round simulation takes 5–15 seconds; a 3-round simulation takes 15–45 seconds depending on hardware.

**How the test harness works:**

```python
def run_simulation(n_rounds, algorithm="FedAvg", mu=0.0, use_dp=False):
    # Reset state for a clean run
    _reset_state(n_rounds, algorithm)
    _STOP.clear()   # ensure stop signal is not set
    
    # Run fl_thread (the same function the Streamlit UI calls)
    # In tests, run synchronously (not in a background thread)
    fl_thread(algorithm=algorithm, mu=mu, use_dp=use_dp)
    
    return _read_state()   # return final state for assertions
```

| # | Test | What it verifies |
|---|---|---|
| 1 | 1-round sim → `status="done"` | The simulation completes normally without hanging or crashing |
| 2 | Global loss after round 1 is finite | Numerical stability with real training data and the real training loop |
| 3 | `best_round == 1` after 1-round sim | The only round is trivially the best round |
| 4 | Global MAE recorded after round 1 | MAE is computed and stored in state |
| 5 | Global R² recorded (can be negative on round 1) | R² is computed and stored — can be negative on the first round (model hasn't learned enough yet) |
| 6 | All 3 hospitals have `mae`, `r2`, `loss` in client metrics | Per-hospital metrics are tracked for all 3 hospitals |
| 7 | Weight stats recorded for all 3 hospitals | Mean, std, and delta_mean of weights are recorded per hospital |
| 8 | `layer1_weights` has shape `(128, input_dim)` | The first layer's weights (used for the heatmap) have the expected dimensions |
| 9 | `pre_agg_flat` and `post_agg_flat` are 1D arrays of same length | Histogram data arrays are compatible for before/after comparison |
| 10 | 3-round sim → `global_losses` has length 3 | One loss value per completed round |
| 11 | 3-round sim: global loss trend is non-increasing | Convergence is occurring — the simulation learns over multiple rounds |
| 12 | FedYogi sim outperforms FedAvg on same data after 5 rounds | FedYogi's adaptive optimisation produces lower final loss than plain averaging |
| 13 | FedProx (μ=0.5): each client's `delta_mean` < FedAvg delta_mean | FedProx reduces client drift — verified at the system level, not just the unit level |
| 14 | `_STOP.set()` before round 1 → status="stopped", no rounds run | The stop signal is checked at the start of each round — stopping before round 1 prevents any training |
| 15–17 | `round_log` has 1 entry with correct structure | The round log (used by the training chart) has one entry per round with the correct field names |

**The convergence test (Test 11) in detail:**

```python
@pytest.mark.slow
def test_global_loss_non_increasing_over_3_rounds(synthetic_data):
    state = run_simulation(n_rounds=3)
    losses = state["global_losses"]   # [loss_round1, loss_round2, loss_round3]
    
    assert len(losses) == 3
    
    # Overall trend should be downward (allow one non-improvement round)
    # Not every round must improve — the FedYogi explosion phase can cause a brief spike
    n_improvements = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
    assert n_improvements >= len(losses) // 2   # majority of rounds must improve
```

The test allows for one non-improving round because, as documented in `TRAIN_FEDERATED.md`, the FedYogi warm-start causes a brief exploration-explosion phase. Requiring strict monotone decrease would cause the test to fail on the first round of a 2-round sim if that round happened to land in the exploration phase.

**The FedYogi vs FedAvg comparison (Test 12) in detail:**

```python
@pytest.mark.slow
def test_fedyogi_outperforms_fedavg(synthetic_data):
    fedavg_state = run_simulation(n_rounds=5, algorithm="FedAvg")
    fedyogi_state = run_simulation(n_rounds=5, algorithm="FedYogi")
    
    fedavg_final_loss  = fedavg_state["global_losses"][-1]
    fedyogi_final_loss = fedyogi_state["global_losses"][-1]
    
    assert fedyogi_final_loss < fedavg_final_loss
```

This is a high-value test that verifies the algorithm choice is justified. In the real training run, FedYogi achieved R²=0.2229 compared to FedAvg's R²=0.2148 — a measurable improvement. The system test verifies this relationship holds on synthetic data too. If a bug in `_fedyogi` caused it to perform worse than `_fedavg`, this test would catch it immediately.

---

## 16. Run Commands

```bash
# ── FAST TESTS (development cycle) ──────────────────────────────────

# All FL unit tests (no DB, no model load needed)
cd icu_monitor
pytest tests/fl/unit/ -v

# All FL tests excluding slow system tests
pytest tests/fl/ -v -m "not slow"

# Specific test file
pytest tests/fl/unit/test_model_utils.py -v

# Specific test by name
pytest tests/fl/unit/test_model_utils.py::test_weighted_mse_loss_zero_for_perfect -v

# ── INTEGRATION TESTS (slightly slower) ──────────────────────────────

# DB ↔ Model pipeline
pytest tests/fl/integration/test_db_model_integration.py -v

# Full FL round (no Flower gRPC)
pytest tests/fl/integration/test_fl_round_integration.py -v -m "not slow"

# Migration tests
pytest tests/fl/integration/test_migrate_to_db.py -v

# ── SYSTEM TESTS (slow — run before demos) ────────────────────────────

# All system tests (real FL simulation)
pytest tests/fl/system/ -v -m slow

# Only the 1-round convergence test (fastest system test)
pytest tests/fl/system/test_fl_system.py::test_1round_sim_status_done -v

# ── COVERAGE ─────────────────────────────────────────────────────────

# Coverage for all FL modules (excluding slow tests)
pytest tests/fl/ -m "not slow" \
    --cov=db \
    --cov=model_utils \
    --cov=fl_dashboard \
    --cov=client \
    --cov=server \
    --cov=train_federated \
    --cov=migrate_to_db \
    --cov-report=term-missing \
    --cov-report=html:coverage_fl/

# Open HTML report
open coverage_fl/index.html

# ── COMBINED (ClinIQ + FL) ─────────────────────────────────────────

# All tests (backend + FL), skip slow LLM and FL system tests
pytest tests/ -v -m "not slow"

# Full suite including all slow tests (for final submission/demo)
pytest tests/ -v
```

**Recommended development workflow:**

```
While editing model_utils.py:
  pytest tests/fl/unit/test_model_utils.py -v     # fast unit feedback (<5s)

After adding a new DB function:
  pytest tests/fl/unit/test_db_fl.py -v           # DB unit tests (<3s)
  pytest tests/fl/integration/test_db_model_integration.py -v  # integration (<15s)

Before a demo or presentation:
  pytest tests/fl/ -v                              # everything including slow tests
  pytest tests/cliniq/ -v -m "not slow"           # ClinIQ tests
```

---

## 17. Coverage Targets

| Module | Target | Key areas that must be covered |
|---|---|---|
| `db.py` (FL helpers) | ≥ 90% | `get_fl_client_data`, `fl_training_exists`, both NULL-handling branches, trigger creation |
| `model_utils.py` | ≥ 85% | All public functions, both DP branches, both DataLoader branches, all metric computations |
| `fl_dashboard.py` (pure fns) | ≥ 70% | `_fedavg`, `_fedyogi`, all state management functions, all chart builders' null guards |
| `client.py` | ≥ 75% | `fit`, `evaluate`, `get_parameters`, LR decay formula, both DP branches |
| `server.py` | ≥ 75% | `aggregate_fit`, `aggregate_evaluate`, both best-model branches, metadata save/merge |
| `train_federated.py` | ≥ 70% | `make_noniid_split`, config constants, `HospitalClient`, `SaveBestStrategy` variants |
| `migrate_to_db.py` | ≥ 80% | `migrate()`, idempotency check, column rename logic, missing-file guard |

**Why these targets are set at these levels:**

- **`db.py` at ≥ 90%:** Database bugs are the most dangerous — silent data corruption is invisible until the system makes a wrong prediction. The 90% target requires testing almost every code path including error branches.
- **`model_utils.py` at ≥ 85%:** The ML core is the scientific heart of the system. Lower coverage would leave critical paths (like the FedProx proximal term or the AUROC single-class guard) untested.
- **`fl_dashboard.py` at ≥ 70%:** Streamlit UI code is difficult to test without a live browser — the 70% target reflects what is achievable by testing pure functions only. The remaining 30% (Streamlit render calls) is covered by manual testing during demos.
- **`client.py` and `server.py` at ≥ 75%:** Both require mocking Flower internals, which reduces testable coverage of the communication paths. 75% covers all the business logic while accepting that the gRPC binding code cannot be unit-tested.

---

## 18. Implementation Status

| Phase | Status | File |
|---|---|---|
| 1 — DB FL helpers | 🔲 Not started | `tests/fl/unit/test_db_fl.py` |
| 2 — model_utils unit | 🔲 Not started | `tests/fl/unit/test_model_utils.py` |
| 3 — FL dashboard fns | 🔲 Not started | `tests/fl/unit/test_fl_dashboard_fns.py` |
| 4 — Client logic | 🔲 Not started | `tests/fl/unit/test_client_logic.py` |
| 5 — Server strategy | 🔲 Not started | `tests/fl/unit/test_server_strategy.py` |
| 6 — Train federated fns | 🔲 Not started | `tests/fl/unit/test_train_federated_fns.py` |
| 7 — DB ↔ Model integration | 🔲 Not started | `tests/fl/integration/test_db_model_integration.py` |
| 8 — FL round integration | 🔲 Not started | `tests/fl/integration/test_fl_round_integration.py` |
| 9 — Migration integration | 🔲 Not started | `tests/fl/integration/test_migrate_to_db.py` |
| 10 — System tests | 🔲 Not started | `tests/fl/system/test_fl_system.py` |

---

*Documentation for ICU CDSS Capstone Project*
*FL test suite: 21 DB tests + 87 model_utils tests + 50 dashboard tests + 19 client tests + 19 server tests + 30 train_federated tests + 10 DB↔model integration tests + 14 FL round integration tests + 17 migration tests + 17 system tests*
*Tools: pytest + pytest-mock + monkeypatch + numpy.testing (all phases)*

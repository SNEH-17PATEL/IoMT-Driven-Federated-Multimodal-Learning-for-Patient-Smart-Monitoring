# FL System — Complete Testing Guide
**What every federated-learning test does, why it exists, and how the suite fits together**

**Last updated:** 2026-11-27
**Result:** **360 FL tests, all passing** (287 fast + 73 slow) · whole Python suite **750 passed** · Vitest **664 passed**
**Measured statement coverage of the FL modules:** **97%** with the FL tests alone, **99%** counting `db.py` at its whole-suite figure (details in Section 17)

> **Scope:** `model_utils.py`, `db.py` (the FL helpers), `fl_dashboard.py`, `client.py`, `server.py`, `train_federated.py`, `migrate_to_db.py`
> **All tests live in:** `icu_monitor/tests/fl/`
> **Not covered here:** the ClinIQ dashboard and its API (see `CLINIQ_TESTING.md`)
> **Everything below was checked against the real test files and a real run on 2026-11-27.** Every test name, count and number comes from the code, not from memory.

---

## How to Read This Document

- **Sections 1–5** explain *why* the tests are built the way they are, the tools, and the shared setup every test uses.
- **Sections 6–15** describe the **first 10 phases** of tests, one phase per section. Each phase lists its tests with the reason each one exists.
- **Sections 16–18** give the run commands, real coverage numbers and the status table.
- **Section 19** lists things the tests deliberately do **not** check, and known quirks, so nothing is hidden.
- **Section 20** (added 2026-11-27) describes **Phases 11–13**: the tests that run the project's scripts for real, in a sandbox, which is how the coverage rose from 80% to 97%.

If you only want to run the tests, jump to [Section 16](#16-run-commands).

---

## Table of Contents

1. [Why FL Testing Is Different From ClinIQ Testing](#1-why-fl-testing-is-different-from-cliniq-testing)
2. [Files Under Test — What Each File Does](#2-files-under-test--what-each-file-does)
3. [Directory Layout](#3-directory-layout)
4. [Tool Stack](#4-tool-stack)
5. [Shared Fixtures — The Foundation Every Test Relies On](#5-shared-fixtures--the-foundation-every-test-relies-on)
6. [Phase 1 — Database FL Helpers (21 tests)](#6-phase-1--database-fl-helpers-21-tests)
7. [Phase 2 — Model Utilities (91 tests)](#7-phase-2--model-utilities-91-tests)
8. [Phase 3 — FL Dashboard Pure Functions (50 tests)](#8-phase-3--fl-dashboard-pure-functions-50-tests)
9. [Phase 4 — Client Logic (22 tests)](#9-phase-4--client-logic-22-tests)
10. [Phase 5 — Server Strategy (19 tests)](#10-phase-5--server-strategy-19-tests)
11. [Phase 6 — Train Federated Helpers (32 tests)](#11-phase-6--train-federated-helpers-32-tests)
12. [Phase 7 — Database ↔ Model Integration (10 tests)](#12-phase-7--database--model-integration-10-tests)
13. [Phase 8 — FL Round Integration (14 tests)](#13-phase-8--fl-round-integration-14-tests)
14. [Phase 9 — Migration Integration (23 tests)](#14-phase-9--migration-integration-23-tests)
15. [Phase 10 — System Tests (17 tests)](#15-phase-10--system-tests-17-tests)
16. [Run Commands](#16-run-commands)
17. [Coverage — Measured, Not Guessed](#17-coverage--measured-not-guessed)
18. [Implementation Status](#18-implementation-status)
19. [What the Tests Do Not Cover, and Known Quirks](#19-what-the-tests-do-not-cover-and-known-quirks)
20. [Phases 11–13 — Script Entry Points, Training Script and Dashboard Page (added 2026-11-27)](#20-phases-1113--script-entry-points-training-script-and-dashboard-page-added-2026-11-27)

---

## 1. Why FL Testing Is Different From ClinIQ Testing

The ClinIQ tests deal with a web application: send an HTTP request, check the response. The federated-learning tests deal with something harder to test.

### Problem 1 — Distributed computation
In real federated learning, three hospitals each run a Flower client process, talk to a central Flower server over the network (gRPC), and exchange model weights for 100 rounds. A unit test cannot start all of that: it would be slow, need network setup, and failures would be impossible to diagnose.

**Solution:** test the logic in pieces. Pull out the aggregation maths (`_fedavg`, `_fedyogi`), the training functions (`train_model`, `evaluate_model`), and the client and server classes, and test each with tiny synthetic data and with Flower's internals replaced by stand-ins ("mocks").

### Problem 2 — Files that do work the moment they are imported
Several project files are **scripts, not libraries**. Importing them runs real work:
- `client.py` reads the command line, loads the real `scaler.pkl`, loads a hospital's rows from the database and would connect to a Flower server.
- `server.py` loads `scaler.pkl` and would start a server and run 100 rounds.
- `train_federated.py` would run the **entire** federated simulation, save a model and overwrite `training_metadata.json`.
- `fl_dashboard.py` imports Streamlit, which fails outside a running Streamlit app.

**Solution:** "guards" that are put in place *before* the import and removed straight after (Section 4 explains each one). They make the import fast and, most importantly, **stop the tests overwriting the real trained model files.**

### Problem 3 — Non-determinism
Training is random (mini-batch order, initial weights, the order hospitals finish). A test that checks "loss is exactly 4.213" would break on another machine.

**Solution:** test invariants, not exact values:
- "loss after 10 epochs is lower than after 1 epoch" (learning happens),
- "FedProx keeps hospitals closer to the shared model than plain FedAvg" (drift is reduced),
- "FedYogi and FedAvg give different final losses" (the two modes really are different algorithms).

These must hold on any seed, machine or library version.

---

## 2. Files Under Test — What Each File Does

| File | Role | What the tests exercise |
|---|---|---|
| `model_utils.py` | Shared ML code | `ICUModel` (108→256→128→64→1, Dropout 0.3), `ICUDataset`, `ALERT_THRESHOLD = 7.0`, optional `weighted_mse_loss` (off by default), `get_sample_weights`, `train_model` (AdamW, plain MSE, FedProx, gradient clipping), `evaluate_model`, `compute_sofa_risk_metrics`, `get_weights`/`set_weights`, `get_trend`, `classify_range`, differential privacy |
| `db.py` | SQLite layer | The federated-learning helpers: `fl_training_exists()`, `get_fl_client_data()`, the `fl_training` table, and `get_conn()` (WAL mode) |
| `fl_dashboard.py` | Streamlit simulator | `_blank_state`, `_reset_state`, `_read_state`, `_fedavg`, `_fedyogi`, `fl_thread`, the chart builders, `_network_diagram`, `load_training_metadata` |
| `client.py` | Real Flower client | The module-level hospital setup and the `HospitalClient` class: `get_parameters`, `fit` (learning-rate decay, FedProx, optional DP) and `evaluate` |
| `server.py` | Real Flower server | `SaveBestStrategy` (a **FedAvg** subclass): `aggregate_fit`, `aggregate_evaluate`, best-round tracking, the `federated_model_latest.pth` backup, and the partial metadata file |
| `train_federated.py` | Simulation script | `make_noniid_split`, its own `HospitalClient` and `SaveBestStrategy` (built on `_BaseStrategy` = **FedAvg** by default, FedYogi if `USE_FEDYOGI`), and the configuration constants |
| `migrate_to_db.py` | CSV → SQLite import | `migrate()`, idempotency, column renames, the `_check_cwd()` guard |

### Why two `HospitalClient` classes and two `SaveBestStrategy` classes?
- `client.py` + `server.py` are the **real** distributed demo: separate processes, TCP at `127.0.0.1:8080`. `HospitalClient()` there takes **no arguments**; the hospital is chosen by the command line (`--client_id 0/1/2`) and its data and model are module-level variables.
- `train_federated.py` has its own versions for the **simulation**, where all three hospitals run inside one process under Flower's simulation engine (which uses Ray). There `HospitalClient(X, y, hospital_id, name)` receives its data as arguments.

The logic is nearly the same but the entry points differ, so **both are tested**: a bug in one would not show up in tests of the other.

---

## 3. Directory Layout

```
icu_monitor/tests/fl/
├── __init__.py
├── conftest.py                      ← shared fixtures for every FL test (5 fixtures)
│
├── unit/                            ← one function or class at a time
│   ├── test_db_fl.py                ←  21 tests  fl_training helpers (db.py)
│   ├── test_model_utils.py          ←  91 tests  every function in model_utils.py
│   ├── test_fl_dashboard_fns.py     ←  50 tests  pure functions of fl_dashboard.py
│   ├── test_client_logic.py         ←  22 tests  client.py
│   ├── test_server_strategy.py      ←  19 tests  server.py
│   ├── test_train_federated_fns.py  ←  32 tests  train_federated.py
│   └── test_script_entrypoints.py   ←  19 tests  client.py, server.py run as scripts; setup.py checked; model_utils edges  (added)
│                                       (254 unit tests)
├── integration/                     ← several pieces working together
│   ├── test_db_model_integration.py ←  10 tests  DB → dataset → train → evaluate
│   ├── test_fl_round_integration.py ←  14 tests  a full FL round without Flower  [slow]
│   └── test_migrate_to_db.py        ←  23 tests  migrate() on a temporary database (4 added)
│                                       (47 integration tests)
└── system/                          ← the whole simulation from start to finish
    ├── test_fl_system.py            ←  17 tests  fl_thread with synthetic hospitals [slow]
    ├── test_train_federated_script.py ← 24 tests  train_federated.py run for real in a sandbox [slow]  (added)
    └── test_fl_dashboard_app.py     ←  18 tests  the Streamlit page driven by AppTest [slow]  (added)
                                        (59 system tests)

254 + 47 + 59 = 360 tests.   Slow tests: 14 (round integration) + 17 + 24 + 18 (system) = 73.
```

**Why three levels?**
- **Unit tests** ask: "does this one function do what I think?"
- **Integration tests** ask: "do these pieces work when connected?"
- **System tests** ask: "does the whole simulation run from start to finish and give sensible results?"

Each level catches a different kind of bug.

---

## 4. Tool Stack

### pytest
The test runner. The project's `pytest.ini` (in `icu_monitor/`) sets the test folder, file and function naming, `-v --tb=short`, and four markers: `slow`, `unit`, `api`, `system`. Only `slow` is used in the FL tests.

### `unittest.mock` and pytest's `monkeypatch` — the two tools actually used
These are the only mocking tools in the suite (`MagicMock`, `mock_open`, `patch` and `monkeypatch`).

- **`unittest.mock.MagicMock`** is a stand-in object that accepts any call and remembers how it was called. Tests use it to replace `train_model`, `evaluate_model` or Flower pieces and then check "was it called once?" or "with what arguments?".
- **`monkeypatch`** (a pytest built-in) temporarily replaces a global variable or function and **restores it automatically** after the test. Used for `db.DB_PATH`, module constants such as `USE_DP`, and functions inside `client`, `server` or `train_federated`.

> **Earlier versions of this document said the suite used the `pytest-mock` plugin (the `mocker` fixture). It does not:** `mocker` appears in zero tests, and `pytest-mock` is not installed. Likewise the `pytest-cov` plugin is not installed (see Section 17 for how coverage is measured).

```python
def test_db_uses_temp_file(monkeypatch, tmp_path):
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "test.db"))   # only for this test
    # every db.get_conn() now opens the temp file; the real database is untouched
    # after the test, DB_PATH is restored automatically
```

### Import-time guards (the most important trick in this suite)
Because `client.py`, `server.py`, `train_federated.py` and `fl_dashboard.py` do work when imported (Section 1, Problem 2), each test file sets up guards **before** importing the module and **restores everything immediately after**:

| File imported | Guard put in place before the import | Why |
|---|---|---|
| `client.py` | `sys.argv = ["client.py", "--client_id", "0"]`; `flwr.client.start_numpy_client` replaced with a no-op | The script needs a command-line argument and would otherwise try to connect to a server and block |
| `server.py` | `flwr.server.start_server` replaced with a no-op; `builtins.open` redirected to a throw-away `StringIO` whenever a `training_metadata` file is opened for writing | The script would run 100 rounds, and its closing metadata write contains `best_eval_loss = inf` (invalid JSON), which would **corrupt the real `models/training_metadata.json`** |
| `train_federated.py` | `flwr.simulation.start_simulation` replaced by a function that fills `_best_weights`/`_last_weights` with a fresh random model; `np.save` skipped for `shap_background.npy`; `torch.save` replaced by a no-op; `builtins.open` redirected for `training_metadata` writes | The script would run the full 100-round simulation and **overwrite the real trained model, SHAP background and metadata** |
| `fl_dashboard.py` | `sys.modules["streamlit"]` replaced by a `MagicMock` (with `cache_resource` as a pass-through decorator and `tabs`/`columns` returning proper-length lists) | Streamlit cannot run outside its own server; the module calls `st.*` at import time |

After the import the originals (`builtins.open`, `np.save`, `torch.save`) are put back so other test files are not affected.

**Safety check, done on 2026-11-24:** after running the whole FL suite, `models/federated_model.pth` and `models/training_metadata.json` still carry their training-day timestamps and the metadata still reads MAE 1.8006, R² 0.4806, best round 37. The tests do not change the real model.

### `numpy.testing`
Floating-point numbers are not exact (`0.1 + 0.2 != 0.3` in Python). `npt.assert_allclose` and `assert_array_equal`, and `pytest.approx`, compare with a tolerance, which matters here because weights go through many floating-point steps.

### `@pytest.mark.slow`
Tests that run real training are marked `slow`. Run only the fast ones during development:
```bash
pytest tests/fl -m "not slow"     # 287 tests, about 10 seconds
pytest tests/fl -m slow           # 73 tests, about 1.5 minutes
```
(The original 31 slow tests are quick because the synthetic networks are tiny. The 42 added on 2026-11-27 run whole scripts and a Streamlit page, so the slow group as a whole now takes about 1.5 minutes.)

---

## 5. Shared Fixtures — The Foundation Every Test Relies On

Defined in `tests/fl/conftest.py` and available to every FL test without an import. There are exactly **five** fixtures.

### `tmp_db` — an isolated temporary database
```python
@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    import db as db_mod
    db_path = str(tmp_path / "test_fl.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)   # redirect the database file
    db_mod.init_db()                                  # create the normal tables
    return db_path                                    # a path string, not a connection
```
**Why every database test needs it:** the real `models/icu_monitor.db` is about 56 MB of MIMIC-III-derived data. Tests must never read or write it: writes could corrupt it, reads are slow and machine-dependent, and results would depend on its contents. With `tmp_db` every test gets its **own empty database** with the right tables, which is deleted afterwards.

It returns the **path string** so a test can open its own connection. `fl_training` is *not* created by `init_db()`; tests create it with pandas where needed, exactly as `migrate_to_db.py` does.

### `synthetic_client_data` — fake training data
```python
@pytest.fixture
def synthetic_client_data():
    rng = np.random.default_rng(42)         # fixed seed → reproducible
    n, input_dim = 200, 10
    X = rng.standard_normal((n, input_dim)).astype(np.float32)
    # y: 70% Low (0–5), 24% Moderate (5–10), 6% High (10–24), shuffled
    ...
    return X, y        # X: (200, 10) float32,  y: (200,) float32, range 0–24
```
**Why 70/24/6?** It mirrors the real mix: most ICU stays are Low Risk and few are High. **Why 10 inputs and not 108?** Speed. The full 108-input network has 69,121 weights; the 10-input version has 44,033. The *structure* (4 linear layers, 3 ReLU, 3 dropout, same widths after the first layer) is identical, so what the tests prove about the architecture holds for the real model too.

### `mini_model` — a tiny network
`ICUModel(10)`: the same architecture as production but with 10 inputs.

### `mini_weights` — its weights as numpy arrays
`get_weights(mini_model)`: the list-of-arrays format Flower uses to send weights between hospital and server.

### `fl_training_rows` — a factory that inserts synthetic hospital rows
```python
def test_foo(tmp_db, fl_training_rows):
    fl_training_rows(tmp_db, client_id=0, n=50)                       # 50 rows for hospital 0
    fl_training_rows(tmp_db, client_id=1, n=40, with_null_sofa=True)  # one row has no SOFA answer
```
Signature: `_insert(db_path, client_id, n=50, with_null_sofa=False, with_nan_feature=False)`. It writes rows with columns `client_id`, `feature1`, `feature2` and `sofa_score` into `fl_training` (creating it, plus the `idx_ft_client` index, if needed). Two options create the two data problems the code must survive: a missing SOFA answer in row 0, and a NaN feature in row 1. The fixture is a *factory* (it returns a function) so each test can set up exactly the hospitals it needs.

> **Note:** these rows use two tiny feature columns, not the real 108. That is enough to test the database code, which does not care how many feature columns there are.

---

## 6. Phase 1 — Database FL Helpers (21 tests)

**File:** `tests/fl/unit/test_db_fl.py` · **Goal:** test the `db.py` functions that touch the `fl_training` table (the 41,798 hospital training rows). The ClinIQ tests cover the other tables. **No model is loaded and the real database is never touched.**

Helpers in the file: `_make_rows(client_id, n, with_null_sofa, with_nan_feature, with_timestamps)` builds row dictionaries and `_create_fl_table(db_path, rows)` creates `fl_training` with pandas `to_sql` (as the migration does) plus its index.

### 1A — `fl_training_exists()` (4 tests)
It answers: "has the training data been loaded into the database yet?" The code checks that the table exists, has at least one row, **and** has a `sofa_score` column.

| Test | Why it matters |
|---|---|
| `…false_when_table_absent` | On a fresh install the table does not exist; the function must return False, not crash |
| `…false_when_empty` | A table with no rows (for example an interrupted migration) counts as "no data" |
| `…true_after_one_row` | The normal post-migration case |
| `…true_for_any_client` | It checks for *any* FL data, not one hospital's |

### 1B — `get_fl_client_data(client_id)` (11 tests)
Each hospital's training code calls this to load **only its own rows**.

| Test | What it checks | Why it matters |
|---|---|---|
| `…returns_dataframe` | Returns a pandas DataFrame | Downstream code uses `.drop`, `.values`, `.fillna` |
| `…drops_id_column` | The database row number `id` is removed | It is not a medical feature; leaving it in would let the model learn the row number |
| `…drops_client_id_column` | `client_id` is removed | Otherwise the model could learn "hospital 0 = Low Risk", a form of leakage |
| `…filters_by_client_id` | Only that hospital's rows come back | The core privacy rule of federated learning |
| `…empty_for_unknown_client` | An unknown id gives an empty DataFrame | No crash on a bad id |
| `…drops_null_sofa_rows` | Rows with no SOFA answer are dropped | A row without a target would crash the loss computation |
| `…warns_on_null_sofa` | A warning tells how many rows were dropped | A silent drop could hide data corruption |
| `…fills_nan_features_with_zero` | NaN features become 0 | NaN would crash PyTorch tensor creation |
| `…sofa_score_present` | The label column is kept | Training reads `df["sofa_score"]` |
| `…correct_row_count` | Clean data: no row lost | Count correctness |
| `…cross_client_isolation` | Two hospitals get separate row sets (8 and 6 rows) | The fundamental FL data-isolation guarantee |

### 1C — The `fl_training` table (4 tests)
The table is **not** created by a `CREATE TABLE` statement. `train_federated.py` and `migrate_to_db.py` create it with pandas `to_sql`, so these tests check what that produces.

| Test | Why it matters |
|---|---|
| `…table_created_via_pandas_to_sql` | The table appears when written the way the project writes it |
| `…client_id_index_created` | `idx_ft_client` exists. Every hospital loads data with `WHERE client_id = ?`; without the index each load scans every row |
| `init_db_never_installs_fl_training_trigger` | `fl_training` has no timestamp columns, so `init_db()` must **not** put a timestamp trigger on it, and ordinary inserts must keep working |
| `init_db_skips_fl_trigger_when_table_absent` | `init_db()` runs before the data exists and must not fail or create a trigger |

> **Corrected from earlier versions:** an older version of this document described a trigger `trg_ft_created_at` that sets `created_at` automatically. **No such trigger exists any more**; the tests now assert the opposite (it must *not* be installed). Also, no test checks `PRAGMA foreign_keys`.

### 1D — `get_conn()` (2 tests)
| Test | Why it matters |
|---|---|
| `…uses_wal_journal_mode` | WAL (write-ahead logging) lets the API and training read and write at once without "database is locked" errors |
| `…concurrent_wal_writes` | Two connections can write and read in turn: a write committed on one is visible on the other, with no corruption |

---

## 7. Phase 2 — Model Utilities (91 tests)

**File:** `tests/fl/unit/test_model_utils.py` · **Goal:** test every public function in `model_utils.py` as plain Python with tiny data. No database, no file I/O. The largest test file.

> **The count is 91.** The file's own docstring header still says "87 tests" (it was not updated when four tests were added); this document uses the real count, which pytest confirms.

Group sizes: 2A ICUModel **11** · 2B ICUDataset **4** · 2C weighted_mse_loss **5** · 2D get_sample_weights **5** · 2E train_model **13** · 2F evaluate_model **8** · 2G compute_sofa_risk_metrics **16** · 2H get/set_weights **6** · 2I trend and range **11** · 2J differential privacy **6** · 2K privacy budget **6**. (Total 91.)

### 2A — `ICUModel` architecture (11 tests)
The current network is **108 → 256 → 128 → 64 → 1**: ReLU and Dropout 0.3 after each hidden layer, 69,121 weights. (It was 108→128→64→32→1 with no dropout in an earlier version.)

| Test | What it checks | Why it matters |
|---|---|---|
| `forward_output_shape` | Output shape is `(n, 1)` | The loss expects `(batch, 1)`; `(batch,)` would broadcast wrongly |
| `production_input_dim` | A 108-input model does a forward pass | The real model works |
| `layer_count` | 4 Linear, 3 ReLU and 3 Dropout layers (all p = 0.3) | If someone removes dropout or adds LayerNorm, this fails |
| `first/second/third/output_layer_dims` (4 tests) | Widths 10→256, 256→128, 128→64, 64→1 | Changing a width would make the model incompatible with the saved `federated_model.pth` |
| `dropout_off_in_eval_on_in_train` | Two train-mode passes on the same input differ; two eval-mode passes are identical | Dropout trains the network, but prediction must be repeatable (the backend and SHAP use `eval()`) |
| `custom_hidden_and_no_dropout` | `ICUModel(10, hidden=(8,4), dropout=0.0)` gives widths 8→4→1 and no dropout | Keeps the class usable for experiments |
| `eval_mode_deterministic` | Same input → same output in eval mode | Deterministic predictions |
| `parameter_count` | For 10 inputs: `(10·256+256)+(256·128+128)+(128·64+64)+(64·1+1) = 44,033` | Locks the exact model size. For 108 inputs the same formula gives **69,121** |

**Why the parameter count matters:** Flower sends weights as flat arrays. If server and client models differ in size, `set_weights()` fails. This test catches any architecture change that would break federated communication.

### 2B — `ICUDataset` (4 tests)
| Test | Why it matters |
|---|---|
| `len` | The DataLoader uses `len()` to count batches |
| `getitem_returns_tuple` | The DataLoader needs `(x, y)` |
| `x_dtype_float32` | The weights are float32; a mismatch fails or silently converts |
| `y_shape` | `y` is reshaped to a column `(n, 1)` so each item has shape `(1,)` |

**The shape bug this prevents:** with `y` of shape `(64,)` and predictions `(64, 1)`, PyTorch broadcasts `pred − y` into a `(64, 64)` matrix, so the loss averages 4,096 meaningless numbers and the model never learns. The `.view(-1, 1)` in `ICUDataset` avoids it.

### 2C — `weighted_mse_loss` (5 tests)
An **optional** loss that gives high-SOFA patients more weight (`weight = 1 + 0.5 × SOFA`, normalised by the batch mean). The **final model does not use it**: it trains with plain MSE, because on `dataset_v2` the weighted loss pushed every prediction upward and raised the error. The function is only used when `train_model(..., weighted_loss=True)`, and it stays tested so that option keeps working.

| Test | What it checks |
|---|---|
| `zero_for_perfect_predictions` | Perfect predictions give zero loss |
| `higher_sofa_gets_larger_weight` | High-SOFA errors count more |
| `formula_correctness` | One patient with target 10 and prediction 0: weight normalises to 1, loss = 100 |
| `positive_for_nonzero_error` | Loss is positive |
| `batch_mean_normalized` | After dividing by the batch mean, weights average 1.0, so the loss stays comparable to plain MSE |

### 2D — `get_sample_weights` (5 tests)
Used only for the optional oversampling (off in the final model, because oversampling the 6% High group caused overfitting). Weights: Low (< 5) → 1.0, Moderate (5–9) → 2.0, High (≥ 10) → 5.0; tests check each band, the output length and the `float32` type.

### 2E — `train_model` (13 tests)
The local training function (one hospital, one round). Real signature: `train_model(model, X, y, epochs, lr, batch_size, grad_clip, oversample, global_params, mu, weighted_loss)`; it builds its own AdamW optimiser (weight decay 1e-4) internally.

| Test | Why it matters |
|---|---|
| `returns_float` | The loss is a Python float, not a tensor |
| `loss_is_finite` | NaN/inf means exploding gradients |
| `loss_decreases_with_more_epochs` | 10 epochs on an easy constant target end lower than 1 epoch, so learning really happens |
| `no_fedprox` / `with_fedprox` | Both code paths work (plain and FedProx) |
| `oversample_true` / `oversample_false` | Both DataLoader branches work |
| `weighted_loss_true` | The optional weighted-loss branch still works |
| **`default_is_plain_mse`** | With `lr = 0`, one full-size batch and `dropout = 0.0`, the reported loss equals the batch-mean `(pred − y)²`. If someone silently switches the default back to the weighted loss, this fails |
| `grad_clip` / `no_grad_clip` | Both gradient-clipping branches work |
| `very_low_lr` | `lr = 1e-5` (the learning-rate floor region) still gives a finite loss |
| **`fedprox_increases_loss`** | With a large `mu = 100` and global weights far from the local ones (all zeros), the proximal penalty makes the reported loss larger than without it, which proves the FedProx term is really added |

### 2F — `evaluate_model` (8 tests)
Returns `{"mse", "mae", "r2"}` on the 0–24 scale (predictions are clipped to 0–24).

| Test | The invariant |
|---|---|
| `returns_dict_keys` | Keys `mse`, `mae`, `r2` are present |
| `values_are_float` | Plain floats (Flower serialises metrics as floats) |
| `mse_nonnegative`, `mae_nonnegative` | Squares and absolute values cannot be negative |
| `r2_at_most_one` | R² ≤ 1 by definition |
| `perfect_predictions` | A stand-in model that returns `y` exactly gives MAE 0 and R² 1 |
| `constant_mean_predictions_r2_zero` | Always predicting the true mean gives R² 0 |
| `sets_eval_mode` | Calling it leaves the model in eval mode, so dropout cannot make two evaluations differ |

### 2G — `compute_sofa_risk_metrics` (16 tests)
Builds the clinical performance report shown in the SOFA validation drawer. The default alert threshold is `ALERT_THRESHOLD` (**7.0**).

| Test | Why it matters |
|---|---|
| `return_keys` | Top-level keys `risk_bands` and `high_risk_alert` |
| `risk_band_labels` | Labels in the order Low, Moderate, High, matching the matrix rows |
| `confusion_matrix_is_3x3` | Three classes, 3×3 |
| `per_class_keys` | Each class has precision, recall, sensitivity, specificity, f1 |
| `sensitivity_equals_recall` | The two are aliases of the same number |
| **`default_alert_threshold`** | `predicted_sofa_threshold == 7.0` by default. If someone changes the line without reading why 7 was chosen (see `VALIDATION.md` Section 3.6), this fails. (The backend tests separately check that `ml_service.py` imports the same constant.) |
| `auroc_float_when_both_classes` | AUROC is a float when both classes exist |
| `auroc_none_when_single_class` | With one class AUROC is undefined, so the answer is `None` (not 0 or NaN) |
| `pr_auc_none_when_single_class` | Same for PR-AUC |
| `raises_empty_input` | Empty input raises `ValueError` rather than dividing by zero |
| `raises_length_mismatch` | `y_true` and `y_score` must pair up |
| `raises_non_finite` | NaN or infinity raises an error, not an all-NaN report |
| `predictions_clipped_to_0_24` | 24.3 is treated as 24 (High band), not an error |
| `macro_keys_match_per_class` | The macro summary has the same keys |
| `confusion_rows_sum_to_class_counts` | Rows sum to the true class sizes (40, 30, 30 on perfect predictions) |
| `binary_confusion_total` | TP + FP + TN + FN equals the number of patients |

### 2H — `get_weights` / `set_weights` (6 tests)
The bridge between PyTorch and Flower. Flower sends weights as a list of numpy arrays.

| Test | Why it matters |
|---|---|
| `get_weights_returns_list`, `…elements_are_ndarrays` | Flower's weight type is `List[np.ndarray]` |
| `…length_matches_state_dict` | One array per weight and bias tensor (8 for this network) |
| `set_weights_round_trip`, `get_set_weights_identity` | Send then receive gives back **exactly** the same model; otherwise every round would slowly corrupt it |
| `set_weights_wrong_count_raises` | Weights from a different architecture are refused, not loaded silently |

### 2I — `get_trend` and `classify_range` (11 tests)
Used for the trend text in the AI report. `get_trend` fits a straight line and calls a slope above +0.1 "increasing", below −0.1 "decreasing" and otherwise "stable".

| Test | Case |
|---|---|
| increasing / decreasing / stable_flat | Clear up, clear down, flat |
| `single_element`, `empty` | Too few points: "stable", no crash |
| **`slope_exactly_boundary`** | `[0, 0.1, 0.2, 0.3]` has slope exactly 0.1 and is "stable", locking in the "strictly greater than" rule |
| `classify_range` normal / high / low | 70 in 60–100 is normal, 120 high, 40 low |
| `…lower_boundary_is_normal`, `…upper_boundary_is_normal` | Exactly 60 and exactly 100 are **normal**, so a patient on the boundary is not mislabelled |

### 2J — `apply_dp_to_update` (6 tests)
Differential privacy (off by default) clips each hospital's update and adds Gaussian noise.

| Test | What it checks |
|---|---|
| `returns_list_of_ndarrays` | Same shape and type as the input weights |
| `no_clip_when_norm_small` | A small update is not rescaled |
| `clips_large_update` | A large update is scaled down to the sensitivity limit |
| `noise_is_added` | The output differs from the un-noised update |
| `output_dtype_float32` | float32 for Flower |
| `zero_sigma_no_noise` | `sigma = 0` adds no noise and does not crash |

### 2K — `estimate_privacy_budget` (6 tests)
Formula: `ε ≈ sqrt(2·ln(1.25/δ)) / σ × rounds` (δ = 1e-5).

| Test | Why it matters |
|---|---|
| `returns_float` | The value is shown in the dashboard |
| `zero_rounds` | 0 rounds → ε = 0 |
| `formula` | The implementation matches the formula (for σ = 1, 100 rounds: about 484.5, a very weak guarantee) |
| `sigma_nonpositive_returns_inf` | No noise means no privacy, so ε is infinite |
| `monotone_in_sigma` | More noise → smaller ε |
| `monotone_in_rounds` | More rounds → larger ε |

---

## 8. Phase 3 — FL Dashboard Pure Functions (50 tests)

**File:** `tests/fl/unit/test_fl_dashboard_fns.py` · **Goal:** test the pure functions of `fl_dashboard.py` (the Streamlit federated-learning simulator) **without launching Streamlit**.

**How:** Streamlit is replaced by a mock **before** `fl_dashboard` is imported (Section 4). An autouse fixture resets the shared simulation state before and after every test (`_reset_state(10, "FedYogi + FedProx")`).

Groups: 3A state **11** · 3B `_fedavg` **6** · 3C `_fedyogi` **8** · 3D `load_training_metadata` **7** · 3E chart builders **9** · 3F `_network_diagram` **9**.

### 3A — State management (11 tests)
The simulation runs in a background thread and writes into one shared dictionary, `_STATE`; the page reads it.

| Test | Why it matters |
|---|---|
| `blank_state_status_is_idle` | The page starts idle |
| `…current_round_is_zero` | Round 0 means nothing has run |
| `…total_rounds_is_ten` | The default demo is 10 rounds: long enough to show a trend, short enough for a live demo (the real run is 100) |
| `…client_losses_three_empty_lists` | One empty list per hospital |
| `…best_loss_is_inf` | Infinity is the correct "worst possible" start, so the first round always becomes the best |
| `…has_twenty_keys` | Locks the state's shape (20 keys) |
| `reset_state_sets_total_rounds`, `…sets_algorithm` | A new run uses the chosen settings |
| `reset_state_clears_global_losses` | A new run must not show the previous run's curve |
| `read_state_returns_copy_not_reference` | The page gets a snapshot; changing it must not change the real state |
| `read_state_client_losses_is_deep_copy` | Nested lists are copied too, so the background thread and the page never share a mutable object |

### 3B — `_fedavg` (6 tests)
Weighted average of the hospitals' weights, each hospital counting in proportion to its number of samples.

| Test | What it checks |
|---|---|
| `equal_samples_gives_arithmetic_mean` | Equal sizes give a plain average |
| `unequal_samples_weighted` | Example: weights 1.0 (80 samples) and 3.0 (20 samples) give (80·1 + 20·3)/100 = **1.4** |
| `single_client_unchanged` | One hospital returns its own weights |
| `output_shape_matches_input` | Layer shapes never change |
| `output_dtype_float32` | float32 |
| `result_bounded_by_inputs` | An average cannot fall outside min and max of its inputs |

### 3C — `_fedyogi` (8 tests)
FedYogi (momentum plus an adaptive step) is an **optional** algorithm in the simulator; the real training uses FedAvg + FedProx. The tests keep the option correct. Defaults in `_fedyogi`: η = 0.01, β1 = 0.9, β2 = 0.99, τ = 1e-3.

| Test | What it checks |
|---|---|
| `returns_triple` | Returns `(new_weights, m, v)` |
| `output_shape_matches_global` | Shapes are preserved |
| `cold_start_none_mt_vt` | `m_t = v_t = None` initialises to zeros and τ² |
| **`vt_never_below_tau_squared`** | The variance estimate never drops below τ², so the denominator `√v + τ` is never zero or NaN |
| `new_m_shape_matches_delta` | Momentum has the right shape |
| `zero_delta_no_weight_change` | If hospitals learned nothing, the global model does not move |
| `output_dtype_float32` | float32 |
| `warm_start_uses_previous_mt_vt` | A second round continues from the saved momentum and variance |

### 3D — `load_training_metadata` (7 tests)
Loads `models/training_metadata.json`, or returns built-in defaults if the file is missing.

| Test | What it checks |
|---|---|
| `returns_dict` | Always a dictionary |
| `fallback_num_rounds` (100), `fallback_final_mae` (1.8968), `fallback_final_r2` (0.5995), `fallback_hospitals` (3) | The fallback values are locked so they cannot change unnoticed |
| `reads_from_file`, `file_overrides_final_r2` | When the file exists, **its** values win over the fallback |

> **Historical placeholder, not the current model.** The fallback numbers (MAE 1.8968, R² 0.5995) belong to an **earlier model on the old dataset**. They are shown only if `training_metadata.json` is missing. The current model's real numbers (MAE 1.8006, R² 0.4806) always come from the file. The tests exist to stop these placeholders changing silently; they are not claims about the current model. (The fallback block in `fl_dashboard.py` itself still holds the old values.)

### 3E — Chart builders (9 tests)
Three Plotly builders: `_training_chart`, `_hist_chart`, `_heatmap_chart`.

| Test | Why it matters |
|---|---|
| `training_chart_empty_returns_none`, `hist_chart_none_inputs_returns_none`, `heatmap_chart_none_returns_none` | **Empty-input guards.** When the simulation first opens there is no data yet; the page calls every chart at once, so each builder must return `None` instead of crashing |
| `training_chart_with_data_returns_figure`, `hist_chart_with_data_returns_figure`, `heatmap_chart_with_weights_returns_figure` | With data they return a real figure |
| `training_chart_has_secondary_y_axis` | The training chart has a second axis (loss and R²) |
| `hist_chart_barmode_overlay` | Before and after aggregation histograms are overlaid for comparison |
| `heatmap_chart_slice_shape` | The heatmap shows a slice of the first-layer matrix, not all 27,648 weights, so the page stays fast |

### 3F — `_network_diagram` (9 tests)
Returns an HTML string with an animated picture of hospitals and the server.

| Test | Why it matters |
|---|---|
| `returns_nonempty_string`, `contains_doctype` | A complete HTML document the browser can render |
| `distributing_has_dotdown` | Weights flowing from server to hospitals animate downward |
| `aggregating_has_dotup` | Weights flowing back animate upward |
| `idle_no_animation` | No animation when idle |
| `done_contains_best_round` | After training, the best round number is shown |
| `hospital_names_present`, `algorithm_name_present` | The diagram labels the three hospitals and the algorithm |
| `inf_best_loss_no_crash` | An infinite best loss (nothing finished yet) does not crash the diagram |

---

## 9. Phase 4 — Client Logic (22 tests)

**File:** `tests/fl/unit/test_client_logic.py` · **Goal:** test `client.py`, the real federated client one hospital runs.

> **The count is 22.** The file's docstring says "19 tests" (its `fit()` group has 12 tests, not the number it lists); the real total is 22: 2 + 12 + 3 + 5.

### How the real `client.py` is built (important for understanding the tests)
`client.py` is a **script**. At import it does all of this at module level:
1. reads `--client_id` (0, 1 or 2) with `argparse`,
2. loads `scaler.pkl`,
3. loads that hospital's rows with `get_fl_client_data(CLIENT_ID)`,
4. splits them 85% train / 15% validation,
5. creates the module-level `model`, then defines `HospitalClient` and connects to the server.

So `HospitalClient()` takes **no arguments** and uses the module-level `X_train`, `y_train`, `X_val`, `y_val` and `model`. The constants are `EPOCHS = 3`, `BATCH_SIZE = 64`, `BASE_LR = 0.001`, `LR_DECAY = 0.99`, `GRAD_CLIP = 1.0`, `MU_FEDPROX = 0.5`, `USE_DP = False`, `SERVER_ADDR = "127.0.0.1:8080"`.

The tests set `sys.argv = ["client.py", "--client_id", "0"]` and replace `start_numpy_client` with a no-op before importing; the **real** scaler and the **real** hospital-0 rows are then used. Individual tests replace `train_model`, `evaluate_model`, `apply_dp_to_update`, `get_weights` and `set_weights` inside the `client` module with `MagicMock`s.

**What `fit()` does:** save a copy of the received global weights → load them into the model → compute `lr = max(1e-4, 0.001 × 0.99^(round−1))` → call `train_model(..., epochs=3, batch_size=64, grad_clip=1.0, oversample=False, global_params=parameters, mu=0.5)` → evaluate on the validation set → if DP is on, add noise to the update → return `(weights, n_train, {"loss", "mae", "r2"})`.

### 4A — `get_parameters` (2 tests)
Returns a list of numpy arrays whose length equals the model's number of weight arrays (the type and shape Flower needs).

### 4B — `fit` (12 tests)
| Test | Why it matters |
|---|---|
| `returns_three_element_tuple` | Flower expects `(weights, count, metrics)` |
| `n_samples_equals_x_train_len` | The server weights each hospital by this number in FedAvg |
| `calls_train_model_once`, `calls_evaluate_model_once` | One local training and one evaluation per round |
| `lr_round_1` | Round 1 → `lr = BASE_LR` (0.001) |
| `lr_round_50` | Round 50 → `max(1e-4, 0.001·0.99^49)` ≈ 0.00061 |
| `lr_clamps_to_floor` | Round 1,000 → exactly the 1e-4 floor, so learning never stops |
| `dp_disabled_no_dp_call` | `USE_DP = False` → `apply_dp_to_update` is never called |
| `dp_enabled_calls_dp` | `USE_DP = True` → it is called exactly once |
| `returns_training_metrics` | The third return value has exactly `loss`, `mae`, `r2`, all floats, so the server can log per-hospital stats |
| `weights_list_is_returned` | A non-empty list of arrays |
| `default_server_round_is_1` | If the config has no `server_round`, the client behaves as round 1 |

The two learning-rate tests lock the decay formula: if someone changes `LR_DECAY` from 0.99 to a faster decay, the round-50 test fails at once.

**The DP tests** guard the `if USE_DP:` branch. Without them a developer could move the privacy call outside the `if` (always on) or delete it (never on), and nothing else would notice.

### 4C — `evaluate` (3 tests)
| Test | What it checks |
|---|---|
| `returns_triple` | `(mse, n_val, {mae, r2})` as Flower expects |
| `calls_set_weights_before_evaluate_model` | The received **shared** weights are loaded first, so the hospital scores the shared model, not its own old one |
| `returned_mse_matches_metrics` | The returned loss equals the computed MSE |

### 4D — Configuration constants (5 tests)
`MU_FEDPROX == 0.5`, `USE_DP is False`, `EPOCHS == 3`, `BATCH_SIZE == 64`, `SERVER_ADDR == "127.0.0.1:8080"`. These lock the final settings (the reasons are in `TRAIN_FEDERATED.md`), so a "quick try" such as `EPOCHS = 5` is noticed.

---

## 10. Phase 5 — Server Strategy (19 tests)

**File:** `tests/fl/unit/test_server_strategy.py` · **Goal:** test `SaveBestStrategy` from `server.py`.

### What the real `server.py` does
`SaveBestStrategy` extends **`fl.server.strategy.FedAvg`**.
- **`aggregate_fit`** (after the hospitals train): call the parent FedAvg, load the averaged weights into the server's model and save them to **`federated_model_latest.pth`** every round (a crash-recovery backup only, never the production model), then print the round.
- **`aggregate_evaluate`** (after the hospitals score the new model): average the hospitals' losses; **if this is the lowest so far**, save the model to **`federated_model.pth`** (the file the dashboard loads) and record the round.
- After all rounds: write a **partial** `training_metadata.json` (`aggregation: "FedAvg"`, best round, best loss, architecture, source `server.py`). It **keeps** `final_mae`, `final_r2` and other fields from an earlier `train_federated.py` run, because the server has no test data to compute them itself.

> **Corrected from earlier versions:** the best model is saved to `federated_model.pth` and the per-round backup to `federated_model_latest.pth`. An older document described the reverse (`federated_model_best.pth`), which no longer exists.

### 5A — `aggregate_fit` (5 tests)
| Test | Why it matters |
|---|---|
| `calls_super` | The real FedAvg averaging is used |
| `calls_set_weights_when_params_not_none` | The server's model receives the averaged weights |
| **`saves_latest_model_to_pth`** | Saves exactly once, to a path ending `federated_model_latest.pth`, so the production model is never overwritten by a mediocre round |
| `prints_round_number` | The round number appears in the log |
| `null_params_skips_set_weights` | If aggregation returns nothing, no weights are loaded |

### 5B — `aggregate_evaluate` (7 tests)
| Test | Why it matters |
|---|---|
| `computes_correct_avg_loss` | Losses 1, 2, 3 average to 2.0 |
| `first_round_updates_best_loss` | The first round always beats the starting infinity |
| **`higher_loss_does_not_update_best`** | A worse round (5.0 vs best 1.0) leaves the best loss unchanged |
| `new_best_saves_best_model` | A new best writes `federated_model.pth` |
| **`not_best_does_not_save_best_model`** | A worse round does **not** call `torch.save` at all, so a good model is never overwritten by a worse one |
| `updates_best_round` | The best round number is recorded |
| `calls_super` | The parent evaluation is still run |

### 5C — Strategy settings (3 tests)
`fraction_fit == 1.0` (all hospitals every round), `min_fit_clients == 3`, `min_evaluate_clients == 3`.

### 5D — Constant (1 test)
`NUM_ROUNDS == 100` (matches `train_federated.py`).

### 5E — Metadata (3 tests)
| Test | Why it matters |
|---|---|
| `best_round_field_is_int` | A plain int is JSON-safe |
| `path_is_in_models_dir` | The file goes to `models/training_metadata.json` |
| **`merge_preserves_final_mae_and_r2`** | The merge keeps `final_mae` and `final_r2` from an existing file and drops unrelated keys. Without it, each real-FL run would erase the accuracy figures and the SOFA validation drawer would show nothing |

> The merge test replicates the merge loop of `server.py` on a temporary file, using the **historical** example numbers 1.8968 and 0.5995. They are only test data, not claims about the current model.

---

## 11. Phase 6 — Train Federated Helpers (32 tests)

**File:** `tests/fl/unit/test_train_federated_fns.py` · **Goal:** test `train_federated.py`: the non-IID split, its own `HospitalClient`, its own `SaveBestStrategy` and the configuration constants.

> **The count is 32** = 8 + 7 + 5 + 12. The file's docstring says "30 tests" (and lists 6C as 7 and 6D as 10); the real groups are 6A **8**, 6B **7**, 6C **5**, 6D **12**.

`train_federated.py` runs the *entire* simulation at import time. The guards in Section 4 (fake `start_simulation`, skipped `np.save`, no-op `torch.save`, redirected metadata write) keep that harmless. An autouse fixture resets `_best_loss`, `_best_round`, `_best_weights` and `_last_weights` before each test.

### 6A — `make_noniid_split` (8 tests)
Builds the **optional** specialty-biased hospital split (General / Mixed / Cardiac-Trauma). The default is IID; this is only for demonstrating federated-learning challenges.

| Test | Why it matters |
|---|---|
| `returns_exactly_3_tuples` | Three hospitals |
| `each_element_is_xy_pair` | `(X, y)` pairs |
| `total_samples_preserved` | No patient lost or duplicated |
| `hospital0_low_sofa_majority` | Hospital 0 is mostly Low SOFA, so the bias is real |
| `hospital2_high_sofa_majority` | Hospital 2 is mostly High SOFA |
| `is_deterministic` | The same seed gives the same split |
| `different_seeds_produce_different_results` | The seed is actually used |
| `handles_empty_high_sofa_class` | No High-SOFA patients at all must not crash it |

### 6B — `HospitalClient` (simulation variant) (7 tests)
Constructed as `HospitalClient(X, y, hospital_id, name)`.

| Test | Why it matters |
|---|---|
| `get_parameters_returns_model_weights` | List of arrays, one per weight tensor |
| `fit_calls_set_weights_before_train` | The received weights are loaded **first** (checked by recording call order) |
| `fit_passes_server_round_to_lr_formula` | Round 10 → `lr = max(1e-4, BASE_LR·LR_DECAY⁹)` |
| `fit_dp_enabled_calls_apply_dp` | `USE_DP = True` → DP applied once |
| `fit_returns_correct_layer_count` | Returns the right number of arrays; the extra metrics dict is empty here |
| `evaluate_calls_set_weights_before_eval` | Order is `set_weights`, then `evaluate_model` |
| `evaluate_returns_correct_structure` | `(mse, n_val, {mae, r2})` with `n_val` a positive int |

### 6C — `SaveBestStrategy` (simulation variant) (5 tests)
Built on `_BaseStrategy`, which is **FedAvg** by default and FedYogi only if `USE_FEDYOGI = True`. The tests patch `_BaseStrategy` so they hold for either parent.

| Test | Why it matters |
|---|---|
| **`configure_fit_adds_server_round`** | Every hospital's config contains `server_round` (checked with round 7). Without it `fit()` could not compute its learning rate |
| `aggregate_fit_stores_params_in_last_weights` | The averaged weights are kept in `_last_weights` |
| `aggregate_evaluate_updates_best_loss` | A new lowest loss is recorded |
| `aggregate_evaluate_copies_last_to_best_on_new_best` | On a new best, the stored weights are copied into `_best_weights`, which is what finally gets saved as `federated_model.pth` |
| `aggregate_evaluate_calls_super` | The parent strategy (FedAvg by default) is still called exactly once |

### 6D — Configuration constants (12 tests)
```
NUM_ROUNDS == 100          EPOCHS_PER_ROUND == 3       BATCH_SIZE == 64
MU_FEDPROX == 0.5          LR_DECAY == 0.99            USE_DP is False
OVERSAMPLE is False        USE_FEDYOGI is False        strategy is FedAvg, not FedYogi
SERVER_ETA == 0.001        SERVER_BETA1 == 0.9         SERVER_BETA2 == 0.99
```
- **`USE_FEDYOGI is False`** and **`strategy_is_fedavg_by_default`**: the final model aggregates with plain FedAvg (FedProx stays on at the clients). FedYogi with η = 0.01 made the server loss explode on `dataset_v2`.
- **`SERVER_ETA == 0.001`**: used only if FedYogi is switched back on (it was 0.01).
- **`SERVER_BETA1`, `SERVER_BETA2`**: FedYogi settings (also only used if it is on).
- **`MU_FEDPROX == 0.5`**: the proximal weight; 0.01 was far too weak and 1.0 too strong.

These tests are "configuration as documentation": changing a value fails the test, which forces the person to read why it was chosen.

---

## 12. Phase 7 — Database ↔ Model Integration (10 tests)

**File:** `tests/fl/integration/test_db_model_integration.py` · **Goal:** connect the database layer to the model layer.

**Why integration is needed beyond the unit tests.** Phase 1 shows `get_fl_client_data()` returns the right DataFrame; Phase 2 shows `train_model()` works on arrays. Neither proves the DataFrame from the database can be fed straight into training. A real class of bug this catches: columns arriving in a different order than the scaler was fitted with, which would make R² collapse while every unit test still passed.

All tests use `tmp_db` and `fl_training_rows` (so the real database is untouched). `_load(db_path, client_id)` is a small helper that reads a hospital's X and y.

| Test | What it verifies |
|---|---|
| `db_to_dataset_length` | A database DataFrame wraps into an `ICUDataset` of the right length |
| `db_to_dataset_x_values_match` | Feature values and order survive DB → DataFrame → dataset |
| `db_to_train_model_completes` | Load → train works end to end |
| `db_to_evaluate_model_returns_finite_metrics` | MSE, MAE and R² are finite |
| `different_clients_return_different_data` | Hospital isolation holds through the whole stack |
| `trained_model_weights_differ_from_untrained` | Training really changes the weights |
| `weights_round_trip_on_db_trained_model` | `get_weights` → `set_weights` is exact even after training |
| `compute_sofa_risk_metrics_with_db_data` | The clinical report accepts database-sourced SOFA values |
| `nan_features_filled_with_zero` | The `fillna(0)` in `get_fl_client_data` is applied before training |
| `train_test_split_correct_sizes` | An 80/20 split of a database DataFrame gives the expected sizes |

---

## 13. Phase 8 — FL Round Integration (14 tests, all slow)

**File:** `tests/fl/integration/test_fl_round_integration.py` · **Goal:** run a complete federated round (distribute → train → aggregate) **without Flower**, using the simulator's own functions (`_fedavg`, `_fedyogi`, `fl_thread`) and synthetic hospitals. Every test is `@pytest.mark.slow` because each trains real networks.

Setup: `INPUT_DIM = 10`; `_make_clients()` builds 3 hospitals (100 train + 25 validation rows, random SOFA); `_make_clients_structured()` builds hospitals whose answer depends on one feature, so convergence tests have a real signal to learn. An autouse fixture resets the simulator state (`_reset_state(1, "FedAvg")`, `_STOP.clear()`) before every test. Streamlit is mocked only if an earlier test file has not already done it.

Groups: 8A manual training and aggregation **5** · 8B `fl_thread` state tracking **7** · 8C FedProx vs FedAvg **1** · 8D data shapes **1**.

### 8A — Manual training and aggregation (5 tests)
| Test | What it verifies |
|---|---|
| `three_clients_produce_three_weight_lists` | Each of 3 hospitals trains and returns its own weights |
| `fedavg_aggregated_shapes_match_global` | Aggregation keeps every layer shape |
| `post_fedavg_model_forward_pass_is_valid` | The averaged model still produces finite predictions |
| `fedyogi_variance_floored_at_tau_squared` | The optional FedYogi keeps `v ≥ τ²` and gives finite weights in practice |
| `round2_global_loss_lower_than_round1` | On structured data, a second round of FedAvg gives a lower loss, so convergence works |

### 8B — `fl_thread` state tracking (7 tests)
| Test | What it verifies |
|---|---|
| `fl_thread_status_done_after_completion` | `status` and `current_phase` both end at `"done"` |
| `state_current_round_is_1_after_one_round` | The round counter advances |
| `state_global_losses_length_equals_rounds` | One loss per completed round |
| `state_client_losses_one_entry_per_client` | Each hospital's loss is recorded |
| `stop_event_before_round_yields_stopped` | Setting the stop signal first gives status `"stopped"` |
| `best_round_is_1_after_single_round` | The only round is the best |
| `round_log_has_one_entry_with_correct_keys` | Entry has `Round`, `H0 Loss`, `H1 Loss`, `H2 Loss`, `Global` |

### 8C — FedProx vs FedAvg (1 test)
`fedprox_produces_smaller_weight_delta_than_fedavg`: train the three hospitals for 10 epochs with and without the proximal term (μ = 0.5) and compare the average distance each hospital's weights moved from the shared model. **FedProx must move less.** This verifies the reason FedProx is used: it limits client drift.

### 8D — Data shapes (1 test)
`pre_post_agg_flat_and_layer1_weights_shapes`: after one round, the before- and after-aggregation weight vectors are 1-D and equal in length (for the histogram), and the first-layer matrix has shape `(256, INPUT_DIM)` (for the heatmap).

---

## 14. Phase 9 — Migration Integration (23 tests)

**File:** `tests/fl/integration/test_migrate_to_db.py` · **Goal:** test `migrate_to_db.migrate()`, the one-time import of the CSV files into the SQLite database.

> **The count is 23.** The file's docstring says "17 tests" and lists groups that add up to 17; the file also has the two `_check_cwd` tests (19), and on 2026-11-27 **four more** were added for the skip and failure paths (group 9G, below).

### What `migrate()` does
It first calls `_check_cwd()` (aborts unless run from `icu_monitor/`), then `init_db()`, then imports:

| CSV source (relative paths) | Database table |
|---|---|
| `models/patient_vitals/patient_vitals_01..03.csv` | `patient_vitals` (patient 1, 2, 3) |
| `models/prediction_history/prediction_history_01..03.csv` | `prediction_history` |
| `data/sample_patients/sample_patient01..03_data.csv` | `sample_patients` |
| `data/fl_training/client_0..2.csv` | `fl_training` (`client_id` 0, 1, 2) |

A missing file is reported and skipped. A hospital that already has rows in `fl_training` is skipped, which is what makes the import safe to repeat. After importing it (re)creates `idx_ft_client`.

> **Current project state:** the training data lives in the database already (41,798 rows), and training no longer writes `client_*.csv` files. The migration is therefore needed mainly to (re)create the **demo tables** from `data/sample_patients/`. The `fl_training` import paths are kept for people who receive the three CSVs.

### How the tests stay safe
`monkeypatch.chdir(tmp_path)` makes the relative CSV paths resolve inside a temporary folder; both `db.DB_PATH` and `migrate_to_db.DB_PATH` point to a temporary database; `_check_cwd` is switched off inside the temporary folder (it has its own tests). Small CSV builders (`_vitals_df`, `_hist_df`, `_sp_df`, `_fl_df`) and `_create_all_csvs()` create a full stub tree. The fixtures are `patched_env`, `migrated_db` (one `migrate()` call) and `conn`.

Groups: 9G skip and failure paths **4** (added) · 9A happy path **5** · 9B idempotency **1** · 9C column renames **3** · 9D `client_id` **2** · 9E edge cases **2** · 9F structure and working-directory guard **6** (4 structure checks plus the 2 `_check_cwd` tests). Total 19.

| Test | Why it matters |
|---|---|
| `migrate_completes_without_error` | Basic sanity |
| `patient_vitals_has_rows`, `prediction_history_has_rows`, `sample_patients_has_rows`, `fl_training_has_rows_for_all_three_clients` | Each table is actually filled, not just created |
| **`migrate_twice_leaves_row_counts_unchanged`** | **Idempotency.** Running the script twice must not double any table (checked on all four tables). Achieved by skipping a hospital's `fl_training` import when it already has rows |
| `spo2_column_renamed_from_special_char` | The CSV header `SpO₂` becomes `SpO2` |
| `temp_column_renamed_from_special_char` | `Temp (°C)` becomes `Temp` |
| `gcs_eye_column_renamed_from_space` | `GCS Eye` becomes `GCS_Eye` |
| `fl_training_client_id_column_has_values_0_1_2` | All three hospitals are present |
| `fl_training_client0_rows_assigned_client_id_0` | Each CSV gets the right `client_id` |
| `missing_csv_skipped_without_crash` | A missing file is skipped, not fatal |
| `empty_csv_skipped_without_crash` | An empty file is skipped, not fatal |
| `patient_vitals_patient_column_has_values_1_2_3` | Patient ids are correct |
| `fl_training_index_idx_ft_client_exists` | The performance index is rebuilt |
| `sample_patients_row_idx_starts_at_zero` | The replay position starts at 0 |
| `db_file_size_reported_without_crash` | The size report in the summary cannot crash |
| **`check_cwd_exits_outside_icu_monitor`** | Running from another folder aborts with exit code 1, instead of "succeeding" with empty tables (every "not found" is otherwise swallowed) |
| `check_cwd_passes_inside_icu_monitor` | From the project folder the guard allows the run |

**9G — skip and failure paths (4 tests, added 2026-11-27)**

| Test | Why it matters |
|---|---|
| `empty_prediction_history_csv_is_skipped_with_a_warning` | A header-only history file is reported ("… is empty") and skipped; the other patients are still imported |
| `sample_csv_missing_required_columns_is_skipped` | A sample-patient file missing required columns (for example `stress_score`, `GCS_eye_opening`) is **not imported**, and the message names the missing columns. Note that `clinical_note` is not a required column |
| `fl_import_survives_an_unreadable_table_info_query` | If the database's table-info query fails, the migration carries on and still imports all 9 FL rows |
| `running_the_file_as_a_script_hits_the_wrong_directory_guard` | `python migrate_to_db.py` reaches `migrate()` through its `__main__` guard; from a folder that is not `icu_monitor/` the safety check stops it with exit code 1 |

**Why the column renames matter:** the backend's SQL uses the clean names (`SELECT SpO2 …`). If the import left `SpO₂` in place, those queries would fail with "no such column".

---

## 15. Phase 10 — System Tests (17 tests, all slow)

**File:** `tests/fl/system/test_fl_system.py` · **Goal:** run the **whole simulation loop** (`fl_thread`) with synthetic hospitals, from "idle" to "done".

`fl_thread` is called directly (not in a background thread) with `speed = 0.0`, so runs are synchronous and repeatable. Signature: `fl_thread(num_rounds, speed, algorithm, clients, input_dim)`. A module-scoped fixture `one_round` runs **one** FedAvg round once and shares the result with the tests that only read it. An autouse fixture resets the state around every test. Three data builders exist: random, structured (one learnable signal), and non-IID (each hospital has a different feature and SOFA range; this builder exists in the file but no test currently uses it).

Groups: 10A single round **5** · 10B client and weight metrics **4** · 10C multi-round **2** · 10D algorithm comparison **2** · 10E stop and log **4**.

### 10A — Single round (5 tests)
| Test | What it verifies |
|---|---|
| `one_round_status_done` | The simulation finishes normally |
| `one_round_global_loss_is_finite` | No NaN with real training |
| `one_round_best_round_is_1` | The only round is the best |
| `one_round_global_mae_recorded` | A finite MAE is stored |
| `one_round_global_r2_recorded` | A finite R² is stored (may be negative in round 1) |

### 10B — Client and weight metrics (4 tests)
| Test | What it verifies |
|---|---|
| `all_three_client_metrics_present` | `mae`, `r2`, `loss` for each of the 3 hospitals |
| `weight_stats_all_three_clients_recorded` | `n_params`, `mean`, `std`, `delta_mean` per hospital, per round |
| `layer1_weights_correct_shape` | First-layer matrix is `(256, INPUT_DIM)` (heatmap data) |
| `pre_post_agg_flat_1d_same_length` | Histogram vectors are compatible |

### 10C — Multi-round (2 tests)
| Test | What it verifies |
|---|---|
| `three_round_global_losses_length` | Exactly 3 loss values after 3 rounds |
| `three_round_loss_generally_converging` | On structured data, the round-3 loss is **lower than** round 1 |

### 10D — Algorithm comparison (2 tests)
- **`fedyogi_5round_simulation_converges`:** runs 5 rounds of FedYogi and 5 of FedAvg on the same structured data and seed. Checks both run 5 rounds, that **FedYogi's final loss is lower than its first**, and that **the two final losses differ**, so they are really different algorithms. It does **not** claim FedYogi is better: five synthetic rounds cannot settle that, and on the real `dataset_v2` data FedAvg did better (FedYogi with η = 0.01 made the server loss explode). *Historical note:* an older version of this test asserted "FedYogi beats FedAvg" using old training numbers.
- **`fedprox_clients_smaller_delta_than_fedavg`:** same seed, so same starting weights. With FedProx (μ = 0.5) **each of the three hospitals'** `delta_mean` is smaller than under plain FedAvg. This confirms the drift-limiting property at system level.

### 10E — Stop and log (4 tests)
| Test | What it verifies |
|---|---|
| `stop_before_round_yields_stopped_status` | Setting the stop signal first gives status `"stopped"` and **no** completed rounds |
| `one_round_round_log_has_one_entry` | One log entry per round |
| `one_round_round_log_round_number` | The entry says `Round == 1` |
| `one_round_round_log_has_hospital_and_global_keys` | `H0 Loss`, `H1 Loss`, `H2 Loss`, `Global` |

---

## 16. Run Commands

**Always run from the `icu_monitor/` folder.** Running from a subfolder fails at collection: the client, server and training tests import real project files by relative path.

```bash
cd icu_monitor

# ── FL tests ────────────────────────────────────────────────────────────
pytest tests/fl                       # all 360 FL tests         (≈ 20 s)
pytest tests/fl -m "not slow"         # 287 fast tests           (≈ 10 s)
pytest tests/fl -m slow               # 73 slow tests            (≈ 1.5 min)

pytest tests/fl/unit                  # 254 unit tests
pytest tests/fl/integration           # 47 integration tests
pytest tests/fl/system                # 59 system tests

# one file, or one test
pytest tests/fl/unit/test_model_utils.py -v
pytest tests/fl/unit/test_model_utils.py::test_train_model_default_is_plain_mse -v

# ── Everything (ClinIQ backend + FL) ────────────────────────────────────
pytest tests                          # 750 passed   (≈ 2 min)
pytest tests -m "not slow"            # 667 passed, 83 deselected   (≈ 1.7 min)
#   (the 83 slow tests = 73 FL + 10 that call the real Groq LLM)

# ── Frontend (Vitest) ───────────────────────────────────────────────────
# Frontend test files live in icu_monitor/tests/, outside frontend/, so packages
# need a symlink (ignored by git; create it once):
ln -s frontend/node_modules node_modules
cd frontend && npx vitest run         # 664 passed (53 files)
```

**Recommended workflow**
```
While editing model_utils.py:      pytest tests/fl/unit/test_model_utils.py
After changing a DB function:      pytest tests/fl/unit/test_db_fl.py tests/fl/integration/test_db_model_integration.py
Before a demo or submission:       pytest tests            (and the Vitest run)
```

**Safe to run on your machine:** the FL tests do not write to the real database or overwrite the real trained model (Section 4, "Import-time guards"). They *do* read the real `models/scaler.pkl`, and the client tests read the real hospital-0 rows from `models/icu_monitor.db`, so those files must exist.

---

## 17. Coverage — Measured, Not Guessed

The first measured version (2026-11-24) gave **80%** for the FL modules, with `train_federated.py` at 65% and `fl_dashboard.py` at 78%. The numbers below were **measured again on 2026-11-27** after Phases 11–13 were added. They are **statement coverage**, measured with the `coverage` package run as `python -m coverage run -m pytest …` (the `pytest-cov` plugin is not installed).

| Module | Statements | Missed | `tests/fl` only (360) | Whole suite (750) | Before (2026-11-24, `tests/fl` only) |
|---|---|---|---|---|---|
| `model_utils.py` | 159 | 0 | **100%** | 100% | 99% |
| `migrate_to_db.py` | 158 | 0 | **100%** | 100% | 96% |
| `server.py` | 66 | 0 | **100%** | 100% | 95% |
| `client.py` | 68 | 0 | **100%** | 100% | 94% |
| `train_federated.py` | 400 | 4 | **99%** | 99% | 65% |
| `fl_dashboard.py` | 458 | 13 | **97%** | 97% | 78% |
| `db.py` | 97 | 29 | **70%** | 100% | 70% |
| **All seven** | **1,406** | **46** | **96.7%** | — | 80% |

**Reading the numbers**
- **`db.py` is 70% with the FL tests alone** because the ClinIQ database functions (vitals, history, sample patients) are tested in `tests/cliniq/`. Together the two suites cover **100%** of `db.py`. Counting `db.py` at its whole-suite figure of 100%, only 17 of the 1,406 statements are missed: **98.8%**.
- **`train_federated.py` went from 65% to 99%.** The new Phase 12 tests run the real script in a sandbox with a fake BigQuery, a temporary database and a fake simulation engine, so the Phase 0 download, TF-IDF, split and save steps, the server strategy, the non-IID and differential-privacy options and every error path are now exercised. The 4 missed lines are line 214 (the text cleaner's "if the note is not text, return an empty string" guard, which a fake table with only text notes never triggers) and lines 561–563 (a fallback to the combined training rows inside the non-IID branch, used only when no held-out pool exists).
- **`fl_dashboard.py` went from 78% to 97%.** The new Phase 13 tests run the real Streamlit page with Streamlit's own test runner (`AppTest`), putting it into the idle, running, stopped, done and error views. The 13 missed lines are the zero-fill of feature columns missing from the database (357), the data-loading helper's empty-data branches (378–384, 418, 426, 467, 496) and the `sleep(1)` that ends a running page (1532, which the tests deliberately interrupt).
- **`client.py`, `server.py` and `migrate_to_db.py` are 100%.** Phase 11 runs `client.py` and `server.py` as scripts, which reaches the argument check, the zero-fill warning and the metadata merge.
- **The slow tests carry this.** With the 73 slow FL tests skipped, `fl_dashboard.py` is **62%** and `train_federated.py` **65%** again (the whole non-slow suite is 85%, against 99% with everything).

---

## 18. Implementation Status

| Phase | Tests | Status | File |
|---|---|---|---|
| 1 — DB FL helpers | 21 | ✅ Passing | `tests/fl/unit/test_db_fl.py` |
| 2 — `model_utils` | 91 | ✅ Passing | `tests/fl/unit/test_model_utils.py` |
| 3 — FL dashboard functions | 50 | ✅ Passing | `tests/fl/unit/test_fl_dashboard_fns.py` |
| 4 — Client logic | 22 | ✅ Passing | `tests/fl/unit/test_client_logic.py` |
| 5 — Server strategy | 19 | ✅ Passing | `tests/fl/unit/test_server_strategy.py` |
| 6 — Train-federated helpers | 32 | ✅ Passing | `tests/fl/unit/test_train_federated_fns.py` |
| 7 — DB ↔ model integration | 10 | ✅ Passing | `tests/fl/integration/test_db_model_integration.py` |
| 8 — FL round integration | 14 | ✅ Passing (slow) | `tests/fl/integration/test_fl_round_integration.py` |
| 9 — Migration integration | 23 | ✅ Passing | `tests/fl/integration/test_migrate_to_db.py` |
| 10 — System tests | 17 | ✅ Passing (slow) | `tests/fl/system/test_fl_system.py` |
| 11 — Script entry points | 19 | ✅ Passing | `tests/fl/unit/test_script_entrypoints.py` |
| 12 — Training script | 24 | ✅ Passing (slow) | `tests/fl/system/test_train_federated_script.py` |
| 13 — Dashboard page | 18 | ✅ Passing (slow) | `tests/fl/system/test_fl_dashboard_app.py` |
| **Total FL** | **360** | **✅ all passing** | |

Totals: **360 FL + 390 ClinIQ backend = 750 Python tests**, plus **664 Vitest** frontend tests = **1,414 tests**. (The four Playwright specs in `tests/cliniq/frontend/e2e/` hold 33 more and need a running app; they are not in these counts.)

---

## 19. What the Tests Do Not Cover, and Known Quirks

Stating limits is part of testing honestly.

**Not covered by automated tests**
- **Real network federated learning.** No test starts a real Flower server and clients over gRPC. The real demo (`server.py` + three `client.py` processes) is tried by hand. The tests cover the logic of each piece, not the connection between them.
- **The real 100-round training run and a real BigQuery download.** Phase 12 runs the whole script, but with a fake BigQuery client returning a small table and a fake simulation engine, so the real download and the real 100 rounds are exercised only by actually running `python train_federated.py`.
- **How the Streamlit page looks in a browser.** Phase 13 runs the real page and checks what it writes, but not its colours, layout or animation; those are checked by hand during demos.
- **Model quality.** These tests check that the code is correct and stable, **not** that the model is accurate. Accuracy (MAE 1.80, R² 0.48, AUROC 0.918) is measured once at the end of training and stored in `models/training_metadata.json`; see `VALIDATION.md`.

**Known quirks (documented, not hidden)**
- **`test_fl_dashboard_app.py` needs the real `streamlit`, but three older files replace it with a mock** (`test_fl_dashboard_fns.py`, `test_fl_round_integration.py`, `test_fl_system.py` set `sys.modules["streamlit"]` when pytest collects them). The new file swaps the real one in for its own tests and restores the previous state afterwards, so the order of files does not matter.
- **The Streamlit page keeps its state in a cached singleton**, which a test cannot reach directly. The Phase 13 tests replace `st.cache_resource` with a memoiser they own, which also gives every test a fresh state.
- **Stale numbers in three test docstrings.** `test_model_utils.py` says "87 tests" (91), `test_client_logic.py` says "19" (22), `test_train_federated_fns.py` says "30" (32) and `test_migrate_to_db.py` says "17" (19). The tests themselves are correct; only the headings are out of date. This document uses the real counts.
- **Historical placeholder numbers inside tests.** A few tests use MAE 1.8968 / R² 0.5995 (the `load_training_metadata` fallback and the server metadata-merge example). They belong to an earlier model and are test data only.
- **`fl_dashboard.py` still shows "FedYogi + FedProx" as its default algorithm label and old fallback metadata** if `training_metadata.json` is missing. The tests lock those defaults, so changing them means updating the tests too.
- **Order of hospitals finishing is random in real Flower simulation**, so two training runs of the same settings differ slightly. The tests avoid exact-value checks for this reason.
- **Tests must be run from `icu_monitor/`.** Run from `tests/fl/` and the client, server and training test files fail to collect.

---

## 20. Phases 11–13 — Script Entry Points, Training Script and Dashboard Page (added 2026-11-27)

**Why these exist.** Before 2026-11-27 the FL guide recorded honestly that `train_federated.py` was 65% covered and `fl_dashboard.py` 78%, because both do their work the moment they are run and the old tests had to **replace** that work with stand-ins. These three phases take the opposite approach: **run the real script or page, but inside a sandbox**.

### The sandbox idea (different from the import-time guards in Section 4)
Instead of stubbing out the heavy steps, each test gives the script a **throw-away world** to run in:

| What the script touches | What the test gives it |
|---|---|
| The `models/` folder (`scaler.pkl`, `federated_model.pth`, `training_metadata.json`, …) | A **temporary folder** with its own copy of the sklearn files, set as the working directory, so every file the script writes lands there |
| The SQLite database | A **temporary database** (`db.DB_PATH` redirected) filled with small synthetic hospital rows |
| Google BigQuery | A **fake `bigquery` module** whose `Client().query(...)` returns small DataFrames |
| Flower's simulation engine | A **fake `start_simulation`** that drives the script's own server strategy and client factory for two scripted rounds |
| The network (`start_numpy_client`, `start_server`) | A no-op that records its arguments |

The script's source is run with `exec`, which lets one test **flip a setting** (for example `USE_DP = True`) without editing the real file. **The real `models/` folder and `icu_monitor.db` are never read for writing.** After the runs, the model files still carry their original timestamps and the metadata still reads MAE 1.8006, best round 37.

### Phase 11 — Script entry points (19 tests) — `tests/fl/unit/test_script_entrypoints.py`
Runs `client.py` and `server.py` as scripts, checks edge cases in `model_utils.py`, and (added 2026-11-27) checks what `setup.py` tells the user.

| Group | Tests | What is checked and why |
|---|---|---|
| `model_utils` edges (4) | `train_model` with no data returns 0.0 and **leaves the weights untouched**; a `None` or `NaN` vital is "normal"; the low / normal / high bands | An empty hospital must not break training |
| **`server.py` (5)** | `_fit_config` passes the round number to clients; Flower is started with **100 rounds, 3 clients** at `127.0.0.1:8080`; the metadata is written (into the sandbox only); **metrics from an earlier training run are kept** (and unknown keys are not); **a corrupt metadata file does not stop the server** | The server writes `training_metadata.json`, so it only ever runs in a temporary folder |
| **`setup.py` (2)** | The final lines print the real dashboard address, **`http://localhost:5173`** (never 5174); `setup.py`, `playwright.config.js` and the backend's allowed origins all agree on 5173 | `setup.py` used to print 5174, sending first-time users to a page that does not exist |
| **`client.py` (8)** | Hospital ids 3, 5 and −1 are refused with "must be 0, 1, or 2"; the id argument is required; ids 0, 1, 2 each connect to the local server; **a table missing most feature columns is zero-filled with a warning** and still has all 108 columns | The argument check and the schema-mismatch warning were the last uncovered lines |

### Phase 12 — The training script, run for real (24 tests, all slow) — `tests/fl/system/test_train_federated_script.py`
Runs the whole of `train_federated.py` in the sandbox.

| Group | Tests | What is checked and why |
|---|---|---|
| **Training from an existing table (9)** | End to end: `federated_model.pth`, `shap_background.npy` and `training_metadata.json` are written, with 108 inputs, 100 rounds and `FedAvg + FedProx`; the strategy keeps the **best round** (a loss of 1.0 beats a later 2.1); **no saved test set triggers the "inflated metrics" warning**, a saved one is loaded from disk; **missing feature columns are zero-filled with a warning**; each of the three sklearn files missing stops the run with a clear message; **no weights produced** raises a clear error; a failing metrics call is reported but is not fatal; the risk metrics are stored when they work | This is the real script, in every state a user can put it in |
| **Settings (3)** | `USE_DP = True` writes the privacy numbers and noises the client's upload; `USE_NONIID_SPLIT = True` gives hospital 2 the most high-SOFA patients; `USE_FEDYOGI = True` switches the strategy class | Each option the README mentions actually changes behaviour |
| **Non-IID guard (1)** | Too few rows for a non-IID split raises "empty client" | |
| **Strategy and client pieces (4)** | `client_fn` returns the right hospital; `configure_fit` sends the round number to every client; `aggregate_fit` stores the aggregated weights and, when nothing comes back, **keeps the previous weights** | |
| **Phase 0, with a fake BigQuery (6)** | With an **empty** `fl_training` table the script runs Phase 0: it connects to `mimic-project-2` / `US`; builds a TF-IDF vocabulary (cleaning MIMIC placeholders and numbers); merges notes and vitals; splits **80/20 stratified by high risk**; fits the scaler; splits the training rows into **three near-equal hospitals with the same high-risk share**; writes `scaler.pkl`, `tfidf_vectorizer.pkl`, `feature_columns.pkl`, the test arrays and the database table; **re-running replaces a hospital's rows** instead of doubling them; and **a missing BigQuery library gives an install hint** | This is the data pipeline that created the 41,798 training rows, now tested end to end on a small table |

### Phase 13 — The Streamlit page, run for real (18 tests, all slow) — `tests/fl/system/test_fl_dashboard_app.py`
Runs `fl_dashboard.py` with Streamlit's own test runner, `AppTest`, so every `st.*` call really executes.

| Group | Tests | What is checked and why |
|---|---|---|
| **Guard screens (2)** | With no training table and no scaler the page stops with a clear "Missing data" message and **shows no buttons**; if loading the data fails it reports "Failed to load training data" with the reason | A fresh clone has no `fl_training`, so this is what a first-time user sees |
| **Idle page (7)** | The controls are **Rounds 5–15 (default 10)**, **Speed**, the four **algorithms** (default FedYogi + FedProx) and **Start / Stop / Reset** (Stop disabled); the "How federated learning works" guide; the algorithm banner shows only for the default; the Model Info tab shows the metadata (and the fallback numbers without a file, and the **differential-privacy block** when enabled); the three hospitals are summarised | This pins the real control ranges, which the README now documents correctly |
| **Buttons (2)** | **Start launches the background run with the chosen settings** (6 rounds, 2.5 s, FedAvg) and marks the state "running"; **Reset** returns to idle | |
| **Result views (7)** | The done, stopped and error views render with a round history, weight statistics and the best round marked; a view with no completed round shows the "appear after Round 1" placeholder; **a running view disables the sliders, the algorithm picker, Start and Reset and enables Stop**; **clicking Stop sets the stop flag** | |

**Two tricks worth knowing.** (1) The page keeps its simulation state in a cached singleton that a test cannot reach, so the tests replace `st.cache_resource` with a memoiser they own; that also gives each test a fresh state. (2) A *running* page ends with `sleep(1)` then a rerun, which would loop forever, so the tests make `sleep` raise **only when called from `fl_dashboard.py`**; everything above that line has already been drawn by then.

**What these phases do not do:** they do not start a real Flower server and clients over the network, call the real BigQuery, run the real 100-round training, or judge how the page looks in a browser (Section 19).

---

*Documentation for the ClinIQ ICU CDSS capstone — last updated 2026-11-27.*
*FL test suite: 21 DB + 91 model_utils + 50 dashboard + 22 client + 19 server + 32 train_federated + 10 DB↔model + 14 FL round + 23 migration + 17 system + 19 script entry points + 24 training script + 18 dashboard page = **360 tests, all passing**.*
*Whole Python suite (`pytest tests`): 750 passed · Frontend Vitest: 664 passed (53 files) · Measured FL statement coverage: 96.7% (FL tests alone) / 98.8% (with `db.py` from the whole suite).*
*Tools actually used: pytest, `unittest.mock` (`MagicMock`, `patch`, `mock_open`), `monkeypatch`, `numpy.testing`. Not used: `pytest-mock`, `pytest-cov`.*

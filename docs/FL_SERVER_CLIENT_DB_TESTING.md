# FL System — Complete Testing Implementation Plan

> **Scope:** `train_federated.py`, `fl_dashboard.py`, `server.py`, `client.py`,
> `db.py` (FL-specific helpers), `model_utils.py`, `migrate_to_db.py`
> **All test files live in:** `icu_monitor/tests/fl/`
> **Not in scope:** ClinIQ dashboard (covered in `CLINIQ_TESTING.md`)

---

## Files Under Test

| File | Role | Testable surface |
|---|---|---|
| `db.py` | SQLite layer | `get_fl_client_data`, `fl_training_exists`, `get_conn`, `init_db`, triggers |
| `model_utils.py` | ML core | `ICUModel`, `ICUDataset`, `weighted_mse_loss`, `get_sample_weights`, `train_model`, `evaluate_model`, `compute_sofa_risk_metrics`, `get_weights`, `set_weights`, `get_trend`, `classify_range`, `apply_dp_to_update`, `estimate_privacy_budget` |
| `fl_dashboard.py` | Streamlit FL sim | `_blank_state`, `_reset_state`, `_read_state`, `_fedavg`, `_fedyogi`, `load_training_metadata`, `_training_chart`, `_hist_chart`, `_heatmap_chart`, `_network_diagram`, `fl_thread` state transitions |
| `client.py` | Flower client | `HospitalClient.get_parameters`, `.fit`, `.evaluate`; LR decay; FedProx; DP branch |
| `server.py` | Flower server | `SaveBestStrategy.aggregate_fit`, `.aggregate_evaluate`; best-model tracking; metadata save |
| `train_federated.py` | Unified FL script | `make_noniid_split`, `HospitalClient` (train variant), `SaveBestStrategy` (FedYogi variant); config constants |
| `migrate_to_db.py` | CSV→SQLite import | `migrate()`; idempotency; column mapping; fl_training schema |

---

## Directory Layout

```
tests/fl/
├── __init__.py
├── conftest.py                          ← shared fixtures: tmp_db, synthetic_client_data,
│                                           mini_model, sample_weights_np
├── unit/
│   ├── __init__.py
│   ├── test_db_fl.py                    ← fl_training helpers in db.py
│   ├── test_model_utils.py              ← ALL functions in model_utils.py
│   ├── test_fl_dashboard_fns.py         ← pure functions extracted from fl_dashboard.py
│   ├── test_client_logic.py             ← HospitalClient (client.py)
│   ├── test_server_strategy.py          ← SaveBestStrategy (server.py)
│   └── test_train_federated_fns.py      ← make_noniid_split + HospitalClient (train variant)
├── integration/
│   ├── __init__.py
│   ├── test_db_model_integration.py     ← db.get_fl_client_data → model_utils pipeline
│   ├── test_fl_round_integration.py     ← full FL round without Flower (manual simulate)
│   └── test_migrate_to_db.py            ← migrate_to_db.migrate() with temp DB + CSV stubs
└── system/
    ├── __init__.py
    └── test_fl_system.py                ← end-to-end simulation using fl_thread logic
```

---

## Tool Stack

| Layer | Tool | Notes |
|---|---|---|
| Runner | `pytest` | Same `pytest.ini` at `icu_monitor/` root |
| Mocking | `unittest.mock`, `pytest-mock` | Mock Streamlit, Flower gRPC, file I/O |
| DB isolation | `monkeypatch` + `tmp_path` | Redirect `db.DB_PATH` to temp file |
| Numeric | `numpy.testing.assert_allclose` | Float comparisons with tolerance |
| Markers | `@pytest.mark.slow` for real FL rounds | Deselect with `-m "not slow"` |

### Run commands
```bash
# All FL tests (fast only)
cd icu_monitor
pytest tests/fl/ -v -m "not slow"

# Unit tests only
pytest tests/fl/unit/ -v

# Integration tests
pytest tests/fl/integration/ -v

# System tests (slow — runs real FL simulation round)
pytest tests/fl/system/ -v -m slow

# With coverage
pytest tests/fl/ -m "not slow" --cov=db --cov=model_utils --cov-report=term-missing
```

---

## Shared Fixtures (`conftest.py`)

```python
# ── tmp_db ────────────────────────────────────────────────────────
# monkeypatches db.DB_PATH → fresh temp SQLite for every test
# calls init_db() after patching so all tables exist
# exposes both the path (str) and a direct sqlite3.Connection

# ── synthetic_client_data(n=200, input_dim=10) ───────────────────
# returns (X: np.ndarray[n,input_dim], y: np.ndarray[n]) with
# realistic SOFA range 0–24 (70% Low, 24% Moderate, 6% High)

# ── mini_model(input_dim=10) ─────────────────────────────────────
# returns ICUModel(10) — tiny for speed

# ── mini_weights(model) ──────────────────────────────────────────
# returns get_weights(model) — list of numpy arrays

# ── fl_training_rows(client_id, n=50) ───────────────────────────
# inserts n synthetic rows into fl_training table in tmp_db
# columns: client_id + 108 feature columns + sofa_score
```

---

## Phase 1 — Database FL Helpers (`test_db_fl.py`)

**Goal:** Test every `db.py` function not already covered by `test_db_unit.py`.
**No ML model load. No real DB touched.**

### 1A — `fl_training_exists()`

| # | Test | What it covers |
|---|---|---|
| 1 | Returns `False` when `fl_training` table is absent | Missing table safety |
| 2 | Returns `False` when table exists but has 0 rows | Empty table |
| 3 | Returns `True` after inserting at least 1 row | Normal case |
| 4 | Returns `True` for client_id=2 rows only | Any row, any client |

### 1B — `get_fl_client_data(client_id)`

| # | Test | What it covers |
|---|---|---|
| 5 | Returns a `pd.DataFrame` | Return type |
| 6 | Drops `id` column from result | DB-only col removed |
| 7 | Drops `client_id` column from result | DB-only col removed |
| 8 | Only returns rows for requested `client_id` | WHERE clause |
| 9 | Returns 0 rows for client_id=99 (unknown) | Empty result |
| 10 | Rows with NULL `sofa_score` are dropped | `dropna(subset=["sofa_score"])` |
| 11 | Warning is emitted when NULL rows are dropped | `warnings.warn` triggered |
| 12 | NaN in feature columns replaced with 0 | `fillna(0)` applied |
| 13 | `sofa_score` column is present in result | Label column kept |
| 14 | Returns same row count as inserted (no NULL SOFA) | Count correctness |
| 15 | Two clients return independent DataFrames | Cross-client isolation |

### 1C — `fl_training` table schema & triggers

| # | Test | What it covers |
|---|---|---|
| 16 | `fl_training` table created when populated via pandas `to_sql` | Dynamic schema |
| 17 | `client_id` index (`idx_ft_client`) exists after first insert | Index creation |
| 18 | `created_at` timestamp auto-set on INSERT (trigger) | `trg_ft_created_at` |
| 19 | `init_db()` safely skips `fl_training` trigger if table absent | Conditional trigger |

### 1D — `get_conn()` extended

| # | Test | What it covers |
|---|---|---|
| 20 | `PRAGMA foreign_keys` is ON | Foreign key enforcement |
| 21 | Multiple connections to same DB do not corrupt (WAL mode) | Concurrent write safety |

---

## Phase 2 — Model Utilities (`test_model_utils.py`)

**Goal:** Test every function in `model_utils.py` as pure Python with small synthetic data.
**No DB. No file I/O. Uses `mini_model` fixture.**

### 2A — `ICUModel`

| # | Test | What it covers |
|---|---|---|
| 1 | `ICUModel(10).forward(x)` returns tensor of shape `(n, 1)` | Forward pass |
| 2 | `ICUModel(108).forward(x)` works with real input dim | Production dim |
| 3 | Architecture: 4 Linear layers + 3 ReLU in `.net` | Layer count |
| 4 | First layer: `(input_dim → 128)` | Dimension check |
| 5 | Second layer: `(128 → 64)` | Dimension check |
| 6 | Third layer: `(64 → 32)` | Dimension check |
| 7 | Output layer: `(32 → 1)` | Dimension check |
| 8 | `ICUModel(10)` in eval mode outputs deterministic values | `model.eval()` |
| 9 | Parameter count matches architecture formula | Weight count |

### 2B — `ICUDataset`

| # | Test | What it covers |
|---|---|---|
| 10 | `len(dataset)` == number of samples | `__len__` |
| 11 | `dataset[0]` returns `(X_tensor, y_tensor)` tuple | `__getitem__` |
| 12 | X tensor dtype is `float32` | Type check |
| 13 | y tensor shape is `(1,)` not `()` | `.view(-1, 1)` applied |

### 2C — `weighted_mse_loss()`

| # | Test | What it covers |
|---|---|---|
| 14 | Perfect predictions → loss == 0 | Zero loss |
| 15 | Higher SOFA targets get larger weights | Weight formula |
| 16 | `weight = 1 + target * 0.5` before normalization | Formula correctness |
| 17 | Loss is positive for non-zero error | Positivity |
| 18 | Loss is batch-mean normalized (comparable to plain MSE) | Normalization |

### 2D — `get_sample_weights()`

| # | Test | What it covers |
|---|---|---|
| 19 | Low SOFA (< 5) → weight == 1.0 | Low-risk weight |
| 20 | Moderate SOFA (5–9) → weight == 2.0 | Moderate weight |
| 21 | High SOFA (≥ 10) → weight == 5.0 | High-risk weight |
| 22 | Output length equals input length | Shape correctness |
| 23 | Output dtype is `float32` | Type check |

### 2E — `train_model()`

| # | Test | What it covers |
|---|---|---|
| 24 | Returns a `float` | Return type |
| 25 | Returned loss is finite (not NaN/inf) | Numerical stability |
| 26 | Loss decreases from epoch 1 to epoch 5 on same data | Learning happens |
| 27 | Works with `global_params=None, mu=0.0` (no FedProx) | Base case |
| 28 | Works with `global_params=weights, mu=0.5` (FedProx) | FedProx branch |
| 29 | Works with `oversample=True` | WeightedRandomSampler branch |
| 30 | Works with `oversample=False` | Default DataLoader |
| 31 | Works with `grad_clip=1.0` | Gradient clipping branch |
| 32 | Works with `grad_clip=None` | No clipping branch |
| 33 | LR decay: `lr=1e-5` still trains without NaN | Minimum LR |
| 34 | FedProx proximal term is added to loss when `mu > 0` | Prox term integration |

### 2F — `evaluate_model()`

| # | Test | What it covers |
|---|---|---|
| 35 | Returns dict with keys `mse`, `mae`, `r2` | Return shape |
| 36 | All values are Python `float` | Return types |
| 37 | MSE is non-negative | Math invariant |
| 38 | MAE is non-negative | Math invariant |
| 39 | R² ≤ 1.0 always | Math invariant |
| 40 | Perfect predictions → MAE == 0, R² == 1.0 | Identity case |
| 41 | Constant predictions → R² ≤ 0 | Degenerate case |
| 42 | Model set to eval mode (no gradient computation) | `model.eval()` |

### 2G — `compute_sofa_risk_metrics()`

| # | Test | What it covers |
|---|---|---|
| 43 | Returns dict with `risk_bands` and `high_risk_alert` keys | Return shape |
| 44 | `risk_bands.labels` == `["Low (<5)", "Moderate (5-<10)", "High (>=10)"]` | Label order |
| 45 | `confusion_matrix` is 3×3 list | Matrix shape |
| 46 | Per-class dict has `precision`, `recall`, `sensitivity`, `specificity`, `f1` | Per-class keys |
| 47 | `sensitivity == recall` for all classes | Alias correctness |
| 48 | `high_risk_alert.predicted_sofa_threshold == 8.0` | Default threshold |
| 49 | AUROC is `float` when both classes present | Binary metric |
| 50 | AUROC is `None` when only one class in `y_true` | Single-class guard |
| 51 | PR-AUC is `None` when only one class present | Single-class guard |
| 52 | Raises `ValueError` for empty inputs | Input validation |
| 53 | Raises `ValueError` when `y_true` and `y_score` lengths differ | Shape mismatch |
| 54 | Raises `ValueError` for non-finite values (NaN/inf) | NaN guard |
| 55 | Predictions clipped to `[0, 24]` before band assignment | `np.clip` applied |
| 56 | `macro` dict has same keys as per-class | Consistency |
| 57 | Confusion matrix rows sum to actual class counts | Matrix correctness |
| 58 | `true_negative + false_positive + false_negative + true_positive == len(y_true)` | Confusion total |

### 2H — `get_weights()` and `set_weights()`

| # | Test | What it covers |
|---|---|---|
| 59 | `get_weights(model)` returns a list | Return type |
| 60 | Each element is a `numpy.ndarray` | Array type |
| 61 | Length equals number of parameter tensors in state_dict | Length check |
| 62 | `set_weights(model, weights)` restores identical state | Round-trip |
| 63 | After `set_weights`, `get_weights` returns identical values | Identity |
| 64 | `set_weights` with wrong layer count raises an error | Shape guard |

### 2I — `get_trend()` and `classify_range()`

| # | Test | What it covers |
|---|---|---|
| 65 | `get_trend([1,2,3,4,5])` == `"increasing"` | Positive slope |
| 66 | `get_trend([5,4,3,2,1])` == `"decreasing"` | Negative slope |
| 67 | `get_trend([3,3,3,3])` == `"stable"` | Zero slope |
| 68 | `get_trend([5])` == `"stable"` (single element) | Edge case |
| 69 | `get_trend([])` == `"stable"` (empty) | Empty guard |
| 70 | Slope exactly 0.1 is `"stable"` (boundary not exceeded) | Boundary |
| 71 | `classify_range(70, 60, 100)` == `"normal"` | Normal |
| 72 | `classify_range(120, 60, 100)` == `"high"` | High |
| 73 | `classify_range(40, 60, 100)` == `"low"` | Low |
| 74 | `classify_range(60, 60, 100)` == `"normal"` (boundary, not low) | Lower bound |
| 75 | `classify_range(100, 60, 100)` == `"normal"` (boundary, not high) | Upper bound |

### 2J — `apply_dp_to_update()`

| # | Test | What it covers |
|---|---|---|
| 76 | Returns list of numpy arrays same shape as `local_weights` | Return shape |
| 77 | When update L2 norm ≤ sensitivity, no clipping applied | No-clip branch |
| 78 | When update L2 norm > sensitivity, clipped to sensitivity | Clip branch |
| 79 | Gaussian noise is added (output differs from input + zero noise) | Noise present |
| 80 | Output dtype is `float32` | Type preservation |
| 81 | With `sigma=0`, noise std is 0 (no noise) | Zero noise edge |

### 2K — `estimate_privacy_budget()`

| # | Test | What it covers |
|---|---|---|
| 82 | Returns a `float` | Return type |
| 83 | `estimate_privacy_budget(0, 1.0)` == 0 | Zero rounds |
| 84 | `estimate_privacy_budget(100, 1.0)` matches formula: `sqrt(2*ln(1.25/1e-5))/1.0 * 100` | Formula correctness |
| 85 | `sigma <= 0` returns `inf` | Degenerate guard |
| 86 | Higher sigma → lower epsilon (more privacy) | Monotone |
| 87 | More rounds → higher epsilon (less privacy) | Monotone |

---

## Phase 3 — FL Dashboard Pure Functions (`test_fl_dashboard_fns.py`)

**Goal:** Test every extractable pure function from `fl_dashboard.py` without launching Streamlit.
**Strategy:** Import only the pure functions; mock `st.*` calls and `@st.cache_resource` at module load time.

### 3A — State Management

| # | Test | What it covers |
|---|---|---|
| 1 | `_blank_state()["status"]` == `"idle"` | Initial status |
| 2 | `_blank_state()["current_round"]` == 0 | Initial round |
| 3 | `_blank_state()["total_rounds"]` == 10 | Default rounds |
| 4 | `_blank_state()["client_losses"]` == `[[], [], []]` | 3 empty client lists |
| 5 | `_blank_state()["best_loss"]` == `float("inf")` | Initial best |
| 6 | `_blank_state()` has all required keys (20 keys) | Completeness |
| 7 | `_reset_state(5, "FedAvg")` sets `total_rounds=5` | Round override |
| 8 | `_reset_state(5, "FedAvg")` sets `algorithm="FedAvg"` | Algo override |
| 9 | `_reset_state` clears previous `global_losses` | Clear on reset |
| 10 | `_read_state()` returns a copy (mutating result doesn't affect `_STATE`) | Thread safety |
| 11 | `_read_state()["client_losses"]` is a list of lists (not references) | Deep copy |

### 3B — `_fedavg()`

| # | Test | What it covers |
|---|---|---|
| 12 | Equal sample counts → arithmetic mean | Uniform weighting |
| 13 | Different sample counts → weighted average | Proportional weight |
| 14 | Single client → returns that client's weights unchanged | Trivial case |
| 15 | Output shape matches input shape per layer | Shape preservation |
| 16 | Output dtype is `float32` | Type check |
| 17 | Result is between min and max of inputs (no extrapolation) | Bounds check |

### 3C — `_fedyogi()`

| # | Test | What it covers |
|---|---|---|
| 18 | Returns `(new_weights, new_m, new_v)` tuple | Return structure |
| 19 | Output weights shape matches input global_weights | Shape preservation |
| 20 | `m_t=None, v_t=None` initializes momentum/variance from scratch | Cold start |
| 21 | `v_t` values remain ≥ `tau²` (numerical stability) | `np.maximum` guard |
| 22 | `new_m` shape matches `delta` shape per layer | Momentum shape |
| 23 | With same client weights as global → no update (delta=0) | Zero gradient |
| 24 | Output dtype is `float32` | Type check |
| 25 | Warm start: passing previous `m_t, v_t` uses them | Momentum continuity |

### 3D — `load_training_metadata()`

| # | Test | What it covers |
|---|---|---|
| 26 | Returns a `dict` | Return type |
| 27 | Fallback (no file): `num_rounds == 100` | Default rounds |
| 28 | Fallback: `final_mae == 1.8968` | Default MAE |
| 29 | Fallback: `final_r2 == 0.5995` | Default R² |
| 30 | Fallback: `hospitals == 3` | Default hospital count |
| 31 | When JSON file exists: loads from file (mocked open) | File-read branch |
| 32 | When JSON file has custom `final_r2`, returns that | Override from file |

### 3E — Chart Builders

| # | Test | What it covers |
|---|---|---|
| 33 | `_training_chart([], [], [])` returns `None` | Empty data → None |
| 34 | `_training_chart` with 3 rounds returns Plotly Figure | Figure returned |
| 35 | Figure has secondary y-axis (R² axis) | `secondary_y=True` |
| 36 | `_hist_chart(None, None, 1)` returns `None` | Null inputs → None |
| 37 | `_hist_chart(pre, post, 3)` returns Plotly Figure | Figure returned |
| 38 | Histogram figure has `barmode="overlay"` | Chart config |
| 39 | `_heatmap_chart(None)` returns `None` | Null input → None |
| 40 | `_heatmap_chart(weights)` returns Plotly Figure | Figure returned |
| 41 | Heatmap data slice is `weights.T[:40, :64]` (correct shape) | Slice correctness |

### 3F — `_network_diagram()`

| # | Test | What it covers |
|---|---|---|
| 42 | Returns a non-empty `str` | Return type |
| 43 | Returned string contains `<!DOCTYPE html>` | Valid HTML doc |
| 44 | Phase `"distributing"` → dots flow down (contains `dotDown`) | Animate down |
| 45 | Phase `"aggregating"` → dots flow up (contains `dotUp`) | Animate up |
| 46 | Phase `"idle"` → static dashed line (no `dotDown`/`dotUp`) | Static line |
| 47 | `"done"` phase → contains `best_round` number in HTML | Best round shown |
| 48 | Each hospital name appears in the HTML | Node labels |
| 49 | Algorithm name appears in the HTML | Algorithm label |
| 50 | `best_loss=inf` → displays `0.0` not `inf` in HTML | Inf guard |

---

## Phase 4 — Client Logic (`test_client_logic.py`)

**Goal:** Test `HospitalClient` from `client.py` with mocked Flower and DB.
**Strategy:** Mock `fl.client.NumPyClient`, mock `get_fl_client_data`, mock `get_weights`/`set_weights`.

| # | Test | What it covers |
|---|---|---|
| 1 | `get_parameters({})` returns list of numpy arrays | Weight extraction |
| 2 | `get_parameters({})` length matches model layer count | Correct layer count |
| 3 | `fit(params, {"server_round": 1})` returns `(weights, n_samples, {})` | Return structure |
| 4 | Returned `n_samples` equals `len(X_train)` | Sample count |
| 5 | `fit` calls `train_model` exactly once per call | Training triggered |
| 6 | `fit` calls `evaluate_model` once per call | Eval triggered |
| 7 | LR at round 1: `lr = max(1e-4, 0.001 * 0.99**0)` == 0.001 | LR round 1 |
| 8 | LR at round 50: `lr = max(1e-4, 0.001 * 0.99**49)` | LR decay formula |
| 9 | LR at round 1000: clamps to `1e-4` | LR floor |
| 10 | `fit` with `USE_DP=False` → `apply_dp_to_update` NOT called | DP disabled branch |
| 11 | `fit` with `USE_DP=True` → `apply_dp_to_update` IS called | DP enabled branch |
| 12 | `evaluate(params, {})` returns `(mse, n_val, {"mae": ..., "r2": ...})` | Eval return structure |
| 13 | `evaluate` calls `set_weights` before `evaluate_model` | Correct order |
| 14 | Returned `mse` is `metrics["mse"]` | Correct metric used |
| 15 | `MU_FEDPROX == 0.5` (constant check) | Config value |
| 16 | `USE_DP == False` by default | Config default |
| 17 | `EPOCHS == 3` | Config value |
| 18 | `BATCH_SIZE == 64` | Config value |
| 19 | `SERVER_ADDR == "127.0.0.1:8080"` | Config value |

---

## Phase 5 — Server Strategy (`test_server_strategy.py`)

**Goal:** Test `SaveBestStrategy` from `server.py` with mocked Flower internals.
**Strategy:** Mock `parameters_to_ndarrays`, `torch.save`, the parent `FedAvg` methods.

| # | Test | What it covers |
|---|---|---|
| 1 | `aggregate_fit` calls `super().aggregate_fit(...)` | Parent delegation |
| 2 | `aggregate_fit` calls `set_weights(global_model, weights)` when result not None | Model update |
| 3 | `aggregate_fit` calls `torch.save(...)` to `federated_model.pth` | Latest model save |
| 4 | `aggregate_fit` prints round number | Console logging |
| 5 | `aggregate_fit` with `aggregated_params=None` → no set_weights call | Null guard |
| 6 | `aggregate_evaluate` computes average loss correctly | Avg formula |
| 7 | First round: avg_loss < `_best_loss` (inf) → `_best_loss` updated | Initial best |
| 8 | Second round with higher loss → `_best_loss` unchanged | No regression |
| 9 | New best → `torch.save(...)` to `federated_model_best.pth` | Best model save |
| 10 | Not best → `federated_model_best.pth` NOT saved | No unnecessary write |
| 11 | `_best_round` updated when new best found | Round tracking |
| 12 | `aggregate_evaluate` calls `super().aggregate_evaluate(...)` | Parent delegation |
| 13 | Strategy `fraction_fit == 1.0` | Config check |
| 14 | Strategy `min_fit_clients == 3` | Config check |
| 15 | Strategy `min_evaluate_clients == 3` | Config check |
| 16 | `NUM_ROUNDS == 100` | Config check |
| 17 | Metadata JSON save: `best_round` field is correct type (`int`) | Type check |
| 18 | Metadata JSON: preserves existing `final_mae`/`final_r2` if file already present | Merge logic |
| 19 | Metadata JSON: written to `models/training_metadata.json` | Path check |

---

## Phase 6 — Train Federated Helpers (`test_train_federated_fns.py`)

**Goal:** Test extractable pure functions from `train_federated.py`.
**Strategy:** Test `make_noniid_split`, `HospitalClient` (train variant), config constants.

### 6A — `make_noniid_split()`

| # | Test | What it covers |
|---|---|---|
| 1 | Returns a list of exactly 3 tuples | Hospital count |
| 2 | Each tuple is `(X_array, y_array)` | Return structure |
| 3 | Total samples across 3 splits == input total | No samples lost |
| 4 | Hospital 0 has ≥ 50% Low-SOFA samples (specialty bias) | Non-IID skew H0 |
| 5 | Hospital 2 has ≥ 50% High-SOFA samples (specialty bias) | Non-IID skew H2 |
| 6 | Same seed → identical splits (deterministic) | Reproducibility |
| 7 | Different seed → different splits | RNG variation |
| 8 | Works with empty High-SOFA class (all Low) | Edge case |

### 6B — `HospitalClient` (train variant)

| # | Test | What it covers |
|---|---|---|
| 9 | `get_parameters({})` returns weights from `self.model` | Weight getter |
| 10 | `fit` applies `set_weights` before training | Correct weight init |
| 11 | `fit` passes `server_round` from config to LR formula | LR scheduling |
| 12 | `fit` with `USE_DP=True` calls `apply_dp_to_update` | DP branch |
| 13 | `fit` returns weights with same layer count as model | Shape |
| 14 | `evaluate` applies received parameters to model first | Correct order |
| 15 | `evaluate` returns `(mse, n_val, {"mae":..., "r2":...})` | Return shape |

### 6C — `SaveBestStrategy` (FedYogi variant)

| # | Test | What it covers |
|---|---|---|
| 16 | `configure_fit` adds `{"server_round": rnd}` to config | Round in config |
| 17 | `aggregate_fit` stores result in `_last_weights["params"]` | Weight cache |
| 18 | `aggregate_evaluate` with lower loss updates `_best_loss` | Best tracking |
| 19 | `aggregate_evaluate` copies `_last_weights` to `_best_weights` on new best | Best snapshot |
| 20 | `aggregate_evaluate` calls `super().aggregate_evaluate(...)` | Parent call |

### 6D — Config Constants

| # | Test | What it covers |
|---|---|---|
| 21 | `NUM_ROUNDS == 100` | Training rounds |
| 22 | `EPOCHS_PER_ROUND == 3` | Local epochs |
| 23 | `BATCH_SIZE == 64` | Batch size |
| 24 | `MU_FEDPROX == 0.5` | Proximal weight |
| 25 | `SERVER_ETA == 0.01` | FedYogi step size |
| 26 | `SERVER_BETA1 == 0.9` | Momentum decay |
| 27 | `SERVER_BETA2 == 0.99` | Adaptive decay |
| 28 | `USE_DP == False` | DP disabled by default |
| 29 | `OVERSAMPLE == False` | No oversampling |
| 30 | `LR_DECAY == 0.99` | Per-round LR decay |

---

## Phase 7 — Database ↔ Model Integration (`test_db_model_integration.py`)

**Goal:** Verify the pipeline from `db.get_fl_client_data()` through `model_utils` functions.
**Uses:** `tmp_db` + `fl_training_rows` fixtures.

| # | Test | What it covers |
|---|---|---|
| 1 | `get_fl_client_data(0)` → `ICUDataset` → `len(dataset) > 0` | DB → Dataset |
| 2 | X values from DB match what ICUDataset produces | Column alignment |
| 3 | `get_fl_client_data` → `train_model` completes without error | DB → Training |
| 4 | `get_fl_client_data` → `evaluate_model` returns finite metrics | DB → Evaluation |
| 5 | `get_fl_client_data` for client_id=1 produces different data than client_id=0 | Client isolation |
| 6 | Model trained on client_0 data has different weights than untrained model | Learning verification |
| 7 | `get_weights` → `set_weights` → `get_weights` round-trip on DB-trained model | Weight persistence |
| 8 | `compute_sofa_risk_metrics` works with SOFA values from DB | Full pipeline |
| 9 | NaN values from DB filled with 0 before feeding to `ICUDataset` | NaN safe chain |
| 10 | `get_fl_client_data` + `train_test_split` → correct train/val sizes | Split integration |

---

## Phase 8 — FL Round Integration (`test_fl_round_integration.py`)

**Goal:** Simulate a full FL round (distribute → train → aggregate) without Flower gRPC.
**Uses:** synthetic data + `tmp_db` fixtures. Marked `@pytest.mark.slow`.

| # | Test | What it covers |
|---|---|---|
| 1 | 3 hospitals each train for 1 round → produce 3 weight lists | Parallel training |
| 2 | `_fedavg` on 3 trained weight sets returns correct layer shapes | Aggregation shape |
| 3 | After `_fedavg`, set global model → forward pass produces valid output | Post-aggregation model |
| 4 | `_fedyogi` on 3 trained weight sets produces bounded weights | Yogi stability |
| 5 | Round 2 loss < Round 1 loss (model improves with FedAvg) | Convergence |
| 6 | `fl_thread` state transitions: `idle → running → distributing → training_0 → ... → done` | Phase sequence |
| 7 | `_read_state()` after 1 round has `current_round == 1` | State update |
| 8 | `_read_state()` after 1 round has `global_losses` list of length 1 | Loss recorded |
| 9 | `_read_state()["client_losses"]` has 1 entry per client | Client losses |
| 10 | `_STOP.set()` during `fl_thread` → status eventually becomes `"stopped"` | Stop signal |
| 11 | FedProx (`mu=0.5`) produces smaller weight delta vs FedAvg (`mu=0`) | Drift control |
| 12 | `_read_state()["best_round"]` == 1 after first round (only round) | Best tracking |
| 13 | `layer1_weights` in state has shape `(128, input_dim)` (correct layer) | Heatmap data shape |
| 14 | `pre_agg_flat` and `post_agg_flat` have same length | Histogram data |

---

## Phase 9 — Migration Integration (`test_migrate_to_db.py`)

**Goal:** Test `migrate_to_db.migrate()` with a temp DB and stub CSV files.
**Uses:** `tmp_path` + monkeypatched `db.DB_PATH` + minimal in-memory CSV content.

| # | Test | What it covers |
|---|---|---|
| 1 | `migrate()` completes without raising | Happy path |
| 2 | After migrate: `patient_vitals` table has rows | Vitals imported |
| 3 | After migrate: `prediction_history` table has rows | History imported |
| 4 | After migrate: `sample_patients` table has rows | Sim data imported |
| 5 | After migrate: `fl_training` table has rows for 3 clients | FL data imported |
| 6 | `migrate()` called twice → row counts unchanged (idempotent) | Idempotency |
| 7 | Prediction history column rename: `SpO₂` → `SpO2` | Column mapping |
| 8 | Prediction history column rename: `Temp (°C)` → `Temp` | Column mapping |
| 9 | Prediction history column rename: `GCS Eye` → `GCS_Eye` | Column mapping |
| 10 | `fl_training` rows have `client_id` column with values 0, 1, 2 | Client ID assigned |
| 11 | `fl_training` rows from client_0 CSV are assigned `client_id=0` | Correct assignment |
| 12 | Missing CSV file → skipped without crash (`⚠` printed) | Missing file guard |
| 13 | Empty CSV file → skipped without crash | Empty file guard |
| 14 | `patient_vitals` rows have `patient` column with values 1, 2, 3 | Patient ID assigned |
| 15 | After migrate: `fl_training` index `idx_ft_client` exists | Index created |
| 16 | `sample_patients` has `row_idx` column starting at 0 | Row index added |
| 17 | DB file size is reported (no crash on `os.path.getsize`) | Summary output |

---

## Phase 10 — System Tests (`test_fl_system.py`)

**Goal:** End-to-end FL simulation using `fl_thread` logic with real synthetic data.
**All tests in this phase are `@pytest.mark.slow`.**

| # | Test | What it covers |
|---|---|---|
| 1 | 1-round FL sim completes with `status="done"` | Full round |
| 2 | Global loss after round 1 is finite | Numerical stability |
| 3 | Best round == 1 after 1-round simulation | Best tracking |
| 4 | Global MAE recorded after round 1 | MAE history |
| 5 | Global R² recorded (can be negative on first round) | R² history |
| 6 | Client metrics for all 3 hospitals have `mae`, `r2`, `loss` | Client stats |
| 7 | Weight stats for all 3 hospitals recorded (mean, std, delta_mean) | Weight tracking |
| 8 | `layer1_weights` has correct shape (128, input_dim) | Layer 1 shape |
| 9 | `pre_agg_flat` and `post_agg_flat` are 1D arrays of same length | Histogram data shape |
| 10 | 3-round sim: `global_losses` has length 3 | Round count |
| 11 | 3-round sim: global loss trend is non-increasing (generally converging) | Convergence |
| 12 | FedYogi sim outperforms FedAvg on same data after 5 rounds (lower loss) | Algorithm comparison |
| 13 | FedProx (mu=0.5) sim: each client's `delta_mean` < FedAvg delta_mean | Drift reduction |
| 14 | `_STOP.set()` before round 1 → status = "stopped", no rounds run | Stop before start |
| 15 | 1-round sim with `_read_state()`: `round_log` has 1 entry | Log entry |
| 16 | `round_log[0]["Round"] == 1` | Log fields |
| 17 | `round_log[0]` has H0/H1/H2/Global loss keys | Log structure |

---

## Coverage Targets

| Module | Target | Key areas |
|---|---|---|
| `db.py` (FL helpers) | ≥ 90% | `get_fl_client_data`, `fl_training_exists`, `fl_training` trigger |
| `model_utils.py` | ≥ 85% | All public functions, both branches of DP |
| `fl_dashboard.py` (pure fns) | ≥ 70% | `_fedavg`, `_fedyogi`, state mgmt, chart builders |
| `client.py` | ≥ 75% | `fit`, `evaluate`, LR decay, DP branch |
| `server.py` | ≥ 75% | `aggregate_fit`, `aggregate_evaluate`, metadata save |
| `train_federated.py` | ≥ 70% | `make_noniid_split`, `HospitalClient`, constants |
| `migrate_to_db.py` | ≥ 80% | `migrate()`, idempotency, column mapping |

---

## Implementation Status

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

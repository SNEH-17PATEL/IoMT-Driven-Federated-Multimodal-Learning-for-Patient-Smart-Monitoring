# DB Layer, FL Dashboard, Migration & Model Utilities — Bug & Defect Report

**Files reviewed (completely, line by line):**
[db.py](db.py) (377 lines), [fl_dashboard.py](fl_dashboard.py) (1,736 lines), [migrate_to_db.py](migrate_to_db.py) (178 lines),
[model_utils.py](model_utils.py) (405 lines). I also read their direct callers/callees where behaviour depends on them
(`client.py`, `server.py`, `train_federated.py`, `backend/services/ml_service.py`) and checked the committed artifacts and the local
`models/icu_monitor.db` (opened read-only).

**Branch / commit:** `main-copy` (= `origin/main` @ `1caee9a` plus the docs commits).
**Method:** static read, plus targeted checks run on throw-away temp databases / throw-away model objects (your real DB was not written to).
Confidence tags: **[Verified]** = reproduced by running code, **[Certain]** = follows directly from the code, **[Likely]** = depends on a runtime/library/environment detail, **[Verify]** = needs a check I couldn't do here.
**Status:** report only — nothing has been fixed. Findings already written up elsewhere are cross-referenced (CLINIQ_BUGS.md = `C#/H#/M#/L#`, TRAIN_BUGS.md = `T#`) instead of repeated in full.

ID prefixes: **D** = `db.py`, **G** = `migrate_to_db.py`, **F** = `fl_dashboard.py`, **U** = `model_utils.py`.

---

## Summary

| Severity | Count |
|---|---|
| Critical | 7 |
| High | 21 |
| Medium | 28 |
| Low | 15 |

Most important:
1. `fl_dashboard.py` crashes on startup on this machine's DB and on any DB without the `fl_training` table (F1, F2, D1).
2. The `fl_training` table lifecycle is inconsistent across `db.py` / `migrate_to_db.py` / `train_federated.py`: triggers reference columns the table doesn't have, `replace` vs `append` conflicts, a placeholder schema (D1–D3, G1–G3).
3. The FL simulation can hang in "running" forever on any exception and shows training loss labelled as validation loss (F3, F4).
4. `db.py` leaks connections on every error and does schema changes inside read paths (D4, D5).
5. DP code in `model_utils.py` gives no real privacy and the printed ε is meaningless (U1).

---

## db.py

### CRITICAL

#### D1 — `fl_training_exists()` / `get_fl_client_data()` crash or misbehave when the table is missing or is the placeholder `[Verified]`
- No table → `fl_training_exists()` raises `OperationalError: no such table: fl_training` (reproduced). It is called unguarded at import time by `train_federated.py:129` and at the top of `fl_dashboard.py:964`, so both crash instead of showing their own "missing data" message.
- Placeholder table (what your local `models/icu_monitor.db` has: `client_id, created_at, updated_at`, 0 rows) → `get_fl_client_data(0)` returns an empty frame with **no `sofa_score` column** (reproduced); every caller (`client.py:68`, `train_federated.py:637`, `fl_dashboard.load_client_data`) then does `df["sofa_score"]` → `KeyError`. The "drop rows with NULL sofa_score" guard in `get_fl_client_data` is skipped because the column doesn't exist.
- `init_db()` deliberately doesn't create `fl_training` ("created dynamically by migrate_to_db.py") yet `fl_training_exists()` and `get_fl_client_data()` assume it exists. Where did the 3-column placeholder come from? Nothing in the current `init_db()` creates it, so it is a leftover of an older schema (`migrate_to_db.py:has_full_schema` comment: "minimal placeholder has ≤3 cols") — the code carries compatibility for a schema it no longer produces.

#### D2 — The `fl_training` insert trigger references columns the real table doesn't have `[Verified]`
`init_db()` (l.186-201) installs `trg_ft_created_at` (`UPDATE fl_training SET created_at=…, updated_at=… WHERE rowid = NEW.rowid`) whenever the table exists. A `fl_training` created by pandas `to_sql` (as `migrate_to_db.py` and `train_federated.py` do) has **no `created_at`/`updated_at` columns**, so after `init_db()` runs, every subsequent insert fails with `no such column: created_at` (reproduced). Since `init_db()` is called from `train_federated.run_preprocessing`, `app.py` and (via migrate) the DB setup flow, ordering decides whether training data can ever be written again. The comment "Safe to re-run" is wrong.

#### D3 — `init_db()` is never called by the FastAPI backend (see CLINIQ_BUGS C1) and creates different schema than the importers expect `[Certain]`
Beyond C1: `sample_patients` has `UNIQUE`-less `(patient,row_idx)`, `patient_vitals`/`prediction_history` have `created_at/updated_at` via triggers only, and `fl_training` isn't created. A DB created by `init_db()` alone cannot run the dashboard (empty `sample_patients`) or training (no `fl_training`).

### HIGH

#### D4 — Connections are never closed on error (no `try/finally` / context manager) `[Certain]`
Every helper does `conn = get_conn() … conn.close()` linearly. Any exception between (e.g. `pd.read_sql` on a missing table, reproduced: `DatabaseError … no such table: patient_vitals`) leaves the connection open until garbage collection. On Windows (and OneDrive, see D8) an open handle keeps the file locked; the DB-wal/shm files persist. In a threaded server this is a slow leak plus "database is locked" under load.

#### D5 — Reads perform DDL; write paths swallow every error `[Certain]`
`get_prediction_history` (l.255-260) and `append_prediction` (l.277-282) run `ALTER TABLE prediction_history ADD COLUMN DBP` on every call inside `try/except Exception: pass`. That's a schema write inside a GET endpoint, redundant because `CREATE TABLE` already defines `DBP` (and `init_db()` has the same ALTER), and `except Exception` hides real failures (locked DB, read-only file, disk full) behind "column already exists". (Related: CLINIQ C9.)

#### D6 — `get_conn()` runs `PRAGMA journal_mode=WAL` on every connection and never sets `busy_timeout`/`synchronous` `[Certain]`
Switching journal mode is a write operation that can itself raise `database is locked` when another connection is mid-write; it's repeated per call. The only protection is Python's default 5-second `sqlite3.connect(timeout=5.0)` — too short for long writes such as `fl_training` imports, and not configurable. `check_same_thread=False` is unnecessary because connections are per-call and encourages accidental sharing across threads.

#### D7 — No transactions around multi-statement writes `[Certain]`
`append_patient_vitals` / `append_prediction` insert and then trim in separate statements with a single `commit()`; if the process stops between, or two requests interleave for the same patient (two browser tabs, two `predict` calls), the "sliding window" can exceed 20/50 rows or drop the wrong rows because trimming keys on `id` ordering at that instant. No `BEGIN IMMEDIATE`, no retry.

#### D8 — The WAL database lives in a OneDrive-synced folder `[Likely]`
`DB_PATH = <repo>/models/icu_monitor.db`, and this repo is under `OneDrive\Documents`. SQLite WAL (`-wal`, `-shm` files) on a cloud-synced directory is a known corruption/"database is locked"/sync-conflict risk because the sync client can lock or partially upload the sidecar files while SQLite is using them. (Also M28/M23 in CLINIQ_BUGS: the `-shm`/`-wal` files are tracked by git.)

#### D9 — Training data and operational state share one file with no backup `[Certain]`
`fl_training` (MIMIC-derived, rebuilt only from gitignored 190 MB CSVs or BigQuery) sits in the same `.db` as `patient_vitals`/`prediction_history`. "Delete the DB to reset the dashboard" (a natural fix for corrupt WAL/stale history) destroys the training data irrecoverably on any machine that doesn't have the CSVs. There is no `reset_operational_tables()` helper, no backup, no separate DB files.

### MEDIUM

#### D10 — `get_fl_client_data` masks data-quality problems `[Certain]`
`df.fillna(0)` on every feature column (l.367) turns NULLs in any column into 0 — after the data is already StandardScaler-normalised, 0 means "mean", so import gaps silently train as average patients. Rows with NULL labels are dropped with only a `warnings.warn(..., stacklevel=2)` that most scripts never surface. No `ORDER BY`, so row order (and any later `random_state=42` split) depends on physical order (TRAIN T49). `SELECT *` loads every column including `created_at/updated_at` if present (they'd need to be dropped; only `id`/`client_id` are).

#### D11 — `fl_training_exists()` counts the whole table `[Certain]`
`SELECT COUNT(*) FROM fl_training` full scan (~15–48 k rows × ~110 columns) where `SELECT 1 … LIMIT 1` would do; `fl_dashboard.py` calls it on **every Streamlit rerun** (every second while a simulation runs).

#### D12 — Missing constraints and indexes `[Certain]`
No `UNIQUE(patient,row_idx)` on `sample_patients` (re-running an import after a partial failure can duplicate rows; `get_sample_patient_row` uses `LIMIT 1` with no `ORDER BY`, so duplicates return arbitrary rows), no `CHECK(patient IN (1,2,3))`, no `NOT NULL` on vitals (NaN/NULL reach JSON, CLINIQ M2). `idx_pv_patient` indexes `patient` only while queries filter `patient` and order by `id DESC` (a composite `(patient,id)` would serve both). `fl_training` has no index created by `init_db()` (only by `migrate_to_db.py`, and it is lost whenever pandas recreates the table with `replace`).

#### D13 — `PRAGMA foreign_keys=ON` with no foreign keys `[Certain]`
Dead configuration; there is no patients table, so referential integrity isn't enforced anywhere.

#### D14 — Triggers do extra writes and have gaps `[Certain]`
- Each insert runs an extra `UPDATE` (doubles write cost; noticeable when pandas inserts tens of thousands of rows into a trigger-bearing table).
- `trg_ph_updated_at`'s column list omits `DBP` (CLINIQ M3), so DBP-only updates don't bump `updated_at`.
- `datetime('now','localtime')` stores local time without offset; the data are not comparable across machines/timezones and `Timestamp`/`time` columns written by Python are a different string format.
- `created_at/updated_at` triggers exist for 3 tables but nothing reads those columns.

#### D15 — `get_sample_patient_row` / `get_prediction_history` use pandas for single-row access `[Certain]`
`pd.read_sql` + `df.iloc[0]` for one row per request; the returned `Series` has object dtype when text columns are present, so numerics may become Python objects (callers cast with `float()`/`int()`; a NULL numeric becomes `NaN`/`None` → `int(NaN)` crash, CLINIQ M1).

#### D16 — Unicode column aliases in SQL `[Certain]`
`SpO2 as 'SpO₂'`, `Temp as 'Temp (°C)'`, `GCS_Eye as 'GCS Eye'` are defined in `get_prediction_history` and again as dict keys in `append_prediction` — a hidden contract between layers that only works because both sides copy the same string (CLINIQ M6).

### LOW

#### D17 — Annotation `pd.Series | None` needs Python ≥ 3.10 `[Likely]`
`def get_sample_patient_row(...) -> pd.Series | None:` is evaluated at definition time; on Python 3.9 the module fails to import. Fine on this machine (3.12.2), but `requirements.txt`/README don't state a minimum Python.

#### D18 — Module-level path and no directory creation `[Certain]`
`DB_PATH` is built at import from `__file__`; `sqlite3.connect` fails with `unable to open database file` if `models/` doesn't exist (e.g. a stripped checkout). No `os.makedirs`.

#### D19 — Magic numbers duplicated across layers `[Certain]`
Window sizes 20 and 50 appear as literals in `db.py` (twice each) and again in `backend/api/*.py` (`limit` caps) and the frontend; changing one silently desynchronises the others.

#### D20 — Docstring claims `[Certain]`
Header says tables "replace patient_vitals_01/02/03.csv …"; `get_conn` docstring promises "better concurrency" while per-call connections + WAL pragma (D6) undermine it; `init_db` comment says the fl trigger is "Safe to re-run" (false, D2).

---

## migrate_to_db.py

### CRITICAL

#### G1 — Silent "success" when run from anywhere but the repo root `[Certain]`
CSV directories are relative strings (`"models/patient_vitals"`, `"data/sample_patients"`, `"data/fl_training"`) while the DB path is absolute (`db.DB_PATH`). Run from another directory and every file prints `⚠ … not found — skipping`, the script still creates/touches the DB, prints "Migration complete — row counts", and exits 0 with empty tables. The user believes the migration worked.

### HIGH

#### G2 — Fresh-clone path can't produce `fl_training` `[Certain]`
`data/fl_training/*.csv` is gitignored (190 MB each). On a fresh clone `migrate_to_db.py` skips all three with a warning, so `fl_training` stays absent/placeholder → D1 crashes downstream. The README/summary doesn't tell the user the DB can't be fully built without BigQuery or the large CSVs.

#### G3 — `to_sql(if_exists="replace")` / `append` logic is fragile `[Certain]`
- First import uses `replace` only for `client_id == 0 and not has_full_schema`. If `client_0.csv` is missing but `client_1.csv` exists, client 1 is `append`ed into the 3-column placeholder → `table fl_training has no column named …` (same failure as TRAIN T4).
- `has_full_schema = len(cols) > 5` is a heuristic for "placeholder vs full table".
- `replace` drops the table, its triggers and `idx_ft_client`; the index is re-created at the end but `init_db()` will later re-create `trg_ft_created_at`, which breaks inserts (D2).
- No `chunksize`/`method="multi"` for 3 × ~16 k rows × 110 columns; with triggers present each row triggers an `UPDATE`.

### MEDIUM

#### G4 — No overall transaction; partial migrations are indistinguishable from complete ones `[Certain]`
Each table is committed separately. If the script dies while importing `fl_training` (memory, schema mismatch), the earlier tables are populated and `fl_training` partially so. The "already has N rows — skipping" rule then treats the partial table as done on re-run (`existing > 0`), with no per-file checksum, row-count check or `--force`.

#### G5 — Memory: whole 190 MB CSVs are read at once `[Certain]`
`pd.read_csv(path)` on each client file, then `pd.concat` with a `client_id` frame (a second full copy) before `to_sql`. Peak memory is several × the CSV size (hundreds of MB to GBs for 110 float64 columns); no `chunksize` streaming.

#### G6 — Schema assumptions about legacy CSVs `[Certain]`
`patient_vitals` and `prediction_history` imports do `df.insert(0,"patient",…)` and `to_sql(append)` with whatever columns the CSV has; an extra column (e.g. a stray `Unnamed: 0` from `to_csv` without `index=False`) raises `no column named …`. `prediction_history` rename map handles only three columns; `DBP` may be missing. The legacy CSVs are gitignored so this path is untestable on a fresh checkout.

#### G7 — Sample-patient import assumptions `[Certain]`
`row_idx = range(len(df))` — ordering depends on CSV row order; NaN notes become SQL NULL and later `str(None)`/`"nan"` is fed to TF-IDF (`ml_service`); no validation that required columns exist; stress/GCS dtype not coerced (stress stored REAL, read as `int()` in the backend, CLINIQ M1).

### LOW

#### G8 — Idempotency only by row count `[Certain]`
Changed CSVs are ignored on re-run (`already has rows — skipping`); there's no `--force`, `--reset`, or content hash.

#### G9 — Cosmetic/robustness `[Certain]`
Step labels `[1/4] … [4a/4] … [4b/4]` are inconsistent; unused `import sqlite3`; summary query uses f-string table names (safe constants); the final message says "delete CSVs only after verifying" but doesn't verify anything (no row-count comparison against the CSVs); prints "~48k rows" while the committed metadata shows ~12.8 k.

#### G10 — Hard-coded patient↔file mapping and paths `[Certain]`
Three file names per table are listed inline; adding a patient or hospital means editing multiple blocks. No CLI arguments.

---

## fl_dashboard.py

### CRITICAL

#### F1 — Page crashes on this machine's DB (placeholder `fl_training`) and when the table is missing `[Verified]`
`fl_dashboard.py:963-974` guard calls `fl_training_exists()` — raises on a missing table (D1) — and, for an existing-but-empty table, shows the intended "Missing data" error only if the table has no rows. If rows exist but lack `sofa_score` (partial/placeholder import), `load_client_data()` (l.346-370) raises `KeyError: 'sofa_score'` (reproduced) and Streamlit shows a stack trace. The guard also only checks `scaler.pkl`, not `feature_columns.pkl`/`federated_model.pth`/`training_metadata.json` that the Model Info tab needs.

#### F2 — `training_metadata.json` written by `server.py` crashes the Model Info tab `[Certain]`
The Info tab indexes `m["final_r2"]` (l.1031), `m["final_mae"]`, `m["pred_range_min"]`, `m["train_samples"]`, `m["test_samples"]`, `m["epochs_per_round"]`, `m["split_type"]`, `m["best_round"]`, `m["num_rounds"]`, `m["hospitals"]`, `m["aggregation"]` with no `.get`. `server.py` (real FL) rewrites the file keeping only a subset of keys (and none at all of `final_*` if no previous file existed), so after a real-FL run the whole page dies with `KeyError`. The fallback dict is used only when the file is *missing*, not when it is incomplete.

### HIGH

#### F3 — Any exception in the FL thread leaves the UI in "running" forever `[Certain]`
`fl_thread` has no `try/except`. If `train_model`, `evaluate_model`, aggregation, or shape handling raises, the daemon thread dies silently; `_STATE["status"]` stays `"running"`, `_STATE["error"]` is never written (the field exists but nothing sets it; the badge config for `"error"` is dead code), `is_running` stays True, the page re-runs every second forever, Start/Reset stay disabled, and the only escape is Stop (which sets an event nobody is listening to — the status is never updated to "stopped" because only the dead thread would do it) → the user must restart Streamlit.

#### F4 — Losses are mislabelled: training loss shown as "validation loss", "weighted MSE", and "Best" `[Certain]`
- `c_losses` come from `train_model()` = mean weighted-MSE **plus the FedProx term** over the local *training* epochs, not validation. The history table footer says "H0/H1/H2 = per-hospital **validation** loss", the card says "Best Loss — weighted MSE".
- `g_loss` is a sample-weighted average of those training losses, computed **before** the new aggregate is evaluated; "Best round" is chosen by it. So the best round isn't the round whose *global model* is best (no evaluation of the aggregate is involved in the selection, unlike Flower's `aggregate_evaluate` in `train_federated.py`).
- The completion card then shows `Best Round` (by training loss), `Best MAE = min(gm)` and `Best R² = max(gr)` — three independent optima over different rounds, presented side by side as one result. The Model Info tab says "Best Round — lowest eval loss".
- With FedProx the logged loss contains the proximal penalty, so it rises when clients drift even if the model is fine.

#### F5 — "Global" metrics are measured on data the model trains on `[Certain]`
`combined_X_val` is the union of each client's 15 % validation slice carved out of that client's rows (`load_client_data`), and the global eval each round (`g_mae`, `g_r2`) uses it. The dashboard presents this as the global model quality ("mirrors app_test.py approach"), yet the validation rows are a subset of the same pool used for hospital training in `train_federated.py` and (via T1) for the reported test metrics. Early rounds look worse partly because the simulation re-initialises random weights, not because of true generalisation.

#### F6 — The simulation does not simulate what the repo trains/deploys `[Certain]`
- Always starts from a fresh random model and runs 5–15 rounds (default 10) while the real run is 100 rounds; the banner states "this is the exact algorithm used in your `train_federated.py` run that achieved R² = {final_r2}" — a different procedure (different init, rounds, no seed, no DP, `eta` fixed) with the (leaked, T1) R² from the metadata.
- Hyperparameters (`mu=0.5`, `lr=0.001·0.99^(r−1)`, `epochs=3`, `batch 64`, Yogi constants) are re-hard-coded rather than read from `training_metadata.json`; they already drift from `client.py` (which doesn't even send LR schedule, T5).
- DP is not simulated at all, although Model Info shows a DP explainer.
- The "Aggregation" options `FedAvg + FedProx` etc. map only to `use_yogi`/`use_prox` flags.

#### F7 — Shared global state is per-server, not per-session `[Certain]`
`@st.cache_resource def _shared()` creates one `_STATE`/`_STOP`/`_LOCK` for the whole Streamlit process. Two browser tabs/users see and control the **same** simulation; a second user's Start/Stop/Reset changes everyone's view. `_reset_state` is callable only when `not is_running` per-tab, but "running" is global so the UI can be disabled for one user by another. Restarting the script (code edits) orphans a still-running daemon thread whose writes go to the old state object.

#### F8 — Stop is not honoured promptly and can race with Start `[Certain]`
`_STOP` is checked only between phases (and during `time.sleep(speed)` implicitly not at all). A training step (3 epochs × 3 hospitals) runs to completion; `status` flips to `"stopped"` only when the thread exits. Between clicking Stop and that moment `is_running` remains True. If the thread is in the aggregation block when Stop is set it exits via the `break` after eval, skipping the round log → "stopped" with the last round's metrics missing. `_STOP.clear()` is executed on Start, so a Start immediately after a quick Stop (once status returns) is fine, but a stale thread from a previous script version (F7) can still observe the cleared event and keep running.

#### F9 — Data in the diagram and tiles is hard-coded and wrong `[Certain]`
- `~15,889 / ~15,890 / ~16,372 patients` per hospital are literals in the diagram (l.888-900); actual client sizes come from `load_client_data()` (e.g. ≈ 4.2 k rows each for the committed 12,795-row set) and are displayed correctly elsewhere ("Hospital Dataset Summary"), so two numbers disagree on the same page.
- Model Info: "Parameters: ~23,000" (actual **24,321**, verified), "Loss: … up to 4.7× gradient weight" (code comments elsewhere say ~3.7× / ~2× / "1+SOFA×3 → 2.4×"; the formula `1+0.5·SOFA` normalised by batch mean gives ratios that depend on the batch), "Training Samples — real ICU patients", "Test Samples — held-out patients" (these are rows, from a split that overlaps training; T1/CLINIQ C10), "Pred Range Min — Lowest predicted SOFA across test set" shows −0.2 (SOFA can't be negative).
- DP section says "Model trained with plain FedAvg (no noise)" while the same page says FedYogi + FedProx.

#### F10 — Streamlit/Plotly missing from `requirements.txt`; CSS depends on private DOM `[Certain]`
`requirements.txt` has **no `streamlit` or `plotly`** (verified: neither appears), although `app.py`/`fl_dashboard.py` import both. A clean `pip install -r requirements.txt` can't run this dashboard. The styling relies on `data-testid="baseButton-primary"`, `stExpanderDetails`, `stCustomComponentV1`, `stDataFrameResizable` etc. — private Streamlit DOM attributes that change between Streamlit releases (the file's own comments admit "belt+suspenders" hacks), so a version bump silently breaks readability (white-on-white, dark text on dark buttons).

### MEDIUM

#### F11 — Stale resource caching `[Certain]`
`load_client_data` is `@st.cache_resource` with no key/TTL: after retraining or re-migrating the DB, the page keeps the old arrays until Streamlit restarts. `_shared()` likewise (stale histories from a previous run). `load_training_metadata()` is uncached and re-reads the JSON every rerun (every second during a run).

#### F12 — Auto-refresh by `time.sleep(1); st.rerun()` `[Certain]`
The script thread sleeps while holding the Streamlit run; every second the whole page (including all Plotly charts and the iframe `components.html` diagram) is rebuilt and re-sent, causing flicker, scroll resets and wasted CPU; button clicks are delayed up to 1 s. `st.fragment(run_every=…)` or `st_autorefresh` is the supported way. `components.html` re-renders reset the CSS animations each time so the "flowing dots" restart every second.

#### F13 — Threading/lock gaps `[Likely]`
`is_best = g_loss < _STATE["best_loss"]` reads shared state outside the lock; `_STATE["global_weights"]` is assigned inside the lock but is absent from `_blank_state()`/`_read_state()` (stray key that survives until `_STATE.clear()`), `pre_agg_flat`/`post_agg_flat` (full 24 k-float arrays) and `layer1_weights` are shared by reference and copied by `_read_state` only shallowly (`list()` of lists of floats is fine, numpy arrays are shared), so the UI thread can read arrays the worker mutates. `torch` CPU threads inside a Streamlit server thread compete with Streamlit's own.

#### F14 — FedYogi/FedAvg implementation details `[Certain]`
- `_fedyogi` initialises `v_t = τ²` and floors `v` at `τ²`; Flower initialises `v` from `initial_parameters`-based zeros plus τ², and uses the same sign update, so results differ subtly. No bias correction (same as Flower) but not documented.
- `eta=0.01, β1=0.9, β2=0.99, τ=1e-3` hard-coded and duplicated from `train_federated.py` (`SERVER_*`).
- `_fedavg` weights by `n_train` — correct for FedAvg, but each client's returned weights and `n_train` are lists captured before the STOP check; a partial round (`break` after hospital 1) never reaches aggregation so no partial update, fine.
- When `use_yogi` is False the code sets `global_weights = _fedavg(...)` but still carries stale `m_t, v_t`; irrelevant but dead.

#### F15 — Reproducibility and fairness of the "demo" `[Certain]`
No seed for `ICUModel(input_dim)` init, `DataLoader` shuffling, or sampler: every run shows different curves; claims such as "R² < 0 = model still warming up" are state-dependent. Each client trains once per round on the *same* 85 % split (`random_state=42`) so IID clients see disjoint slices of one pool — "hospital drift" arises only from disjoint random subsets, not from different populations, yet the explainer text attributes drift to "different case mixes, SOFA distributions, and clinical note vocabularies" (true only for Non-IID, which this page doesn't offer).

#### F16 — Charts: edge cases `[Certain]`
`_training_chart` uses `range=[0.5, len(rounds)+0.5]`, `dtick=1/2`; `r2_lo/r2_hi` computed from `min`/`max` with fixed floors (−0.1, 0.5) so strongly negative early R² values (e.g. −5) are clipped off the axis; the annotation "R² < 0 = model still warming up" is placed on the figure whenever any value is negative; the histogram samples 5,000 of 24,321 weights with a fixed RNG seed (always the same indices), so it can't show new tails; `_heatmap_chart` slices `[:40,:64]` (first 40 inputs × 64 neurons only) while the heading says "(108 → 128)".

#### F17 — HTML built with f-strings and `unsafe_allow_html=True` `[Certain]`
Values from `training_metadata.json` (`m["aggregation"]`, `hospital_names`, `split_type`) and from state are interpolated into HTML without escaping. The JSON is a local trusted file, but anything that can write it (e.g. an untrusted retrain) can inject markup/script into the iframe/markdown. The long `aggregation` string in a 26 px tile also overflows the tile (no truncation).

#### F18 — Error/edge handling in layout code `[Certain]`
`zip([wc0,wc1,wc2], range(3), latest)` silently shows fewer cards if `latest` is shorter; `S["client_metrics"][cid]` is `{}` until the first round for each hospital (renders "—"); `st.progress(pct)` receives `rnd/total` where `rnd` counts the *current* round (progress shows 100 % while the last round is still running); the "Stop" → "stopped" state keeps all charts but offers no way to download results; `Reset` always resets to `10 / "FedYogi + FedProx"`, ignoring the sliders.

#### F19 — Misleading status/phase text `[Certain]`
Phase ⑥ is shown for `speed×0.5` seconds; the "done" badge's `best_loss` is replaced by `0.0` when no round finished (`bl if bl < 1e9 else 0.0`) and displays "loss: 0.0000"; the summary card's "weighted MSE" label is wrong (F4); the header says "Real FL Training Visualizer · MIMIC-III Data" for what is a single-process simulation; "Simulates server.py + 3 × client.py … no terminals needed" implies fidelity it doesn't have (F6, TRAIN T5/T7).

### LOW

#### F20 — Dead/unused code and imports `[Certain]`
`pandas` (`pd`) is imported and unused; the `"error"` branch of the badge/status dicts is unreachable (F3); `_STATE["global_weights"]` is never read; `HOSP_DIM`, `HOSP_RGBA` are used but duplicated per-hospital constants; CSS contains many `!important` blocks for components not used on the page (`stDataFrame…`, `iframe.stDataFrame`).

#### F21 — Large inline CSS/HTML (≈ 900 lines) in the script `[Certain]`
Maintainability: styling, state management, algorithms, charts and layout are one 1,700-line file; two near-identical HTML card patterns repeated for tiles; no tests.

#### F22 — Docstring/comment drift `[Certain]`
Docstring says `cd icu_monitor`; mentions "mirrors app_test.py" (a file that isn't in the repo); `load_training_metadata` fallback numbers (12,795 / 2,559 / 18 / 1.8968 / 0.5995) differ from `backend/services/ml_service.py`'s fallback (48,150 / 12,038 / 24 / 1.7588 / 0.4724), so the two dashboards show different "facts" when the file is missing; text says "Uses your existing fl_training table — no BigQuery or internet" but the page can't start without it (F1).

#### F23 — Encoding and paths `[Certain]`
`open(path)` without encoding (JSON is ASCII-escaped, so safe today); `MODEL_PATH = "models/"` is cwd-relative, so `streamlit run` from another directory reports "Missing data" even though files exist.

---

## model_utils.py

### CRITICAL

#### U1 — Differential-privacy utilities don't provide the stated guarantee `[Certain]`
(See TRAIN_BUGS T15; summarised here because the code lives in this file.) `apply_dp_to_update`:
- adds `N(0, (σ·S)²)` noise to **every** parameter of a ≈ 24,321-parameter model (verified count) with σ = S = 1 → noise L2 norm ≈ √24,321 ≈ 156 against a clipped update of norm ≤ 1; the transmitted update is essentially noise;
- there is no per-sample clipping/privacy accountant (it is client-update clipping, "client-level DP", not the "per patient" bound the docstring claims);
- uses the global unseeded `np.random` (not a CSPRNG).
`estimate_privacy_budget` applies the classical Gaussian-mechanism formula ε = √(2 ln(1.25/δ))/σ, which is only valid for ε < 1 (here ≈ 4.8 per round) and composes linearly (≈ 480 after 100 rounds) without composing δ — the reported "ε" is not a meaningful privacy budget.

### HIGH

#### U2 — `get_weights` returns live views of the model's parameters `[Verified]`
`val.cpu().numpy()` on CPU tensors shares memory: after `w = get_weights(m)`, an in-place update of the model changes `w` (reproduced). Callers that keep the returned list across further training — e.g. a client that stores `global_weights = get_weights(model)` and then trains the same model — silently see their "snapshot" move. The Flower callers copy (`[w.copy() for w in parameters]`) only because the parameters come from the server; `fl_dashboard` relies on creating a new model per client for the same reason.

#### U3 — `set_weights` silently truncates and gives cryptic errors `[Verified]`
`zip(state_dict.keys(), weights)` stops at the shorter list; with fewer arrays the later `load_state_dict(strict=True)` raises a `RuntimeError` about missing keys instead of a clear "expected N arrays, got M". Shapes are not validated (a same-size-different-shape array is caught only by torch). No dtype check (float64 arrays become float64 tensors and are cast on load).

#### U4 — `train_model` can't handle empty/degenerate data and mutates caller state `[Verified]`
Empty `X` → `ValueError: num_samples should be a positive integer value, but got num_samples=0` from `DataLoader`/sampler (reproduced; also `len(loader)` division if it got further). It leaves the model in `train()` mode, creates a fresh `AdamW` every call (optimizer state lost each FL round, TRAIN T23), and silently ignores `global_params` when `mu == 0` (by design) or when `global_params` is a list of arrays whose order differs from `model.parameters()` (relies on `state_dict` order = parameter order). `oversample=True` is the default (TRAIN T24).

#### U5 — `get_trend` is unit-agnostic and NaN-blind `[Verified]`
Fixed ±0.1 slope threshold regardless of vital (Temp 36.5→37.4 °C is "stable", HR 80→80.9 "increasing"; CLINIQ M25). With a NaN in the series `np.polyfit` returns NaN, the comparisons are false, and the function returns `"stable"` (reproduced) — a missing reading is reported as a stable trend instead of "unknown". Used for the LLM prompt's "Vital Sign Trends".

### MEDIUM

#### U6 — `evaluate_model` and the metrics functions use raw (unclipped) predictions `[Certain]`
`evaluate_model` returns MSE/MAE/R² on raw outputs (can be negative or > 24) while the dashboard clips to 0–24 and `compute_sofa_risk_metrics` clips internally (TRAIN T16). `r2_score` on a constant/tiny validation slice returns 0.0/NaN; no guard. MSE is computed twice (torch `MSELoss` and sklearn MAE separately) and returned as NumPy floats (fine for Flower but inconsistent types).

#### U7 — Training objective vs. selection metric mismatch `[Certain]`
`weighted_mse_loss` (emphasises high SOFA) is used for training while Flower's `evaluate` returns **unweighted** MSE, so the "best round" criterion favours the majority low-SOFA class that the weighting deliberately de-emphasises (the docstring itself describes this conflict). The weight doc numbers (0.33 / 1.0 / 2.0 / 3.7) are for a "mean target ≈ 4" batch, not the formula's guarantee.

#### U8 — `compute_sofa_risk_metrics` details `[Certain]`
- Bands use fixed edges `[5, 10]` and alert threshold 8 / "high ≥ 10" defined here, again in `ml_service.py` (`ALERT_THRESHOLD`, `risk_label`), in the LLM prompt (0–6 / 7–9 / ≥10) and the frontend (CLINIQ H10) — four copies drifting.
- Specificity for bands with no negatives returns 0.0 (not N/A), which feeds the macro average (`np.mean`), biasing it downward when a class is absent.
- `high_risk_alert.auroc` uses the continuous predicted score against "actual ≥ 10" — correct, but the alert *threshold* metrics (precision/recall) use 8 vs 10, mixing definitions in one block; the docstring claims the result is "displayed in the app's Tab 1" (Streamlit) which no longer exists in the FastAPI/React app.
- Duplicate keys `sensitivity`/`recall` (same value).
- `precision_recall_fscore_support(..., zero_division=0)` hides classes the model never predicts (the committed metadata shows High-class recall 19.8 %).

#### U9 — Docstring/comments inconsistent with code `[Certain]`
`ICUModel` docstring says "Input: 108 features — 9 trend + 7 latest + 2 CV + 90 SOFA-vocab TF-IDF" (matches the artifacts) while `train_federated.py` metadata says 200 TF-IDF features (TRAIN T8); `train_model` docstring and the comment block in `get_sample_weights` say oversampling is unused while the default argument enables it; module comment "shared between app and training" refers to the removed Streamlit app.

#### U10 — Hyperparameters embedded in function bodies `[Certain]`
`weight_decay=1e-4`, loss weight multiplier `0.5`, `AdamW`, `reduction`, and `DataLoader` options aren't parameters; clip value and `mu` are. Any experiment requires editing the library file, and `train_federated`/`client`/`fl_dashboard` can't vary them consistently.

#### U11 — Weighted sampler default semantic risks `[Certain]`
When `oversample=True`, `WeightedRandomSampler(replacement=True, num_samples=len(sample_w))` over-represents high-SOFA rows by 2×/5×; combined with the weighted loss (also emphasising high SOFA) this double-counts the minority class — acknowledged in comments but unprotected by an assertion/warning.

### LOW

#### U12 — Unused imports and style `[Certain]`
`import pandas as pd` is unused; `super(ICUModel, self)` legacy style; `import math` used only once; no type hints or tests for any function (`apply_dp_to_update`, `compute_sofa_risk_metrics`, `get_trend`, `classify_range` have no unit tests).

#### U13 — `classify_range` and `get_trend` edge cases `[Certain]`
`classify_range(value, low, high)` assumes `low ≤ high` and non-NaN `value` (NaN → "normal" because both comparisons are False); `get_trend` returns "stable" for series shorter than 2 even when the one reading is abnormal.

#### U14 — Model has no versioning/metadata `[Certain]`
`ICUModel.__init__(input_dim)` doesn't record the feature list/order; `federated_model.pth` stores only a `state_dict` (no architecture hash, no feature-column hash, no sklearn/torch version, no training metadata link), so mismatched scaler/model pairs (TRAIN T32) can't be detected at load time.

#### U15 — `get_sample_weights` thresholds duplicate risk bands `[Certain]`
Hard-coded 5/10 thresholds again (see U8).

---

## Cross-file issues

#### X1 — `fl_training` lifecycle is owned by three files that disagree `[Verified]`
`db.init_db()` (no table; conditional trigger), `migrate_to_db.py` (`replace`/`append`, creates index), `train_federated.py` (`DELETE` + `append`, calls `init_db()` first). The combination of D1–D3, G3 and TRAIN T2–T4 means there is currently **no sequence of commands on a clean checkout that populates `fl_training` and keeps it writable**. Suggested single owner: one `ensure_fl_training_schema()` in `db.py` with explicit columns (`client_id`, 108 feature columns from `feature_columns.pkl`, `sofa_score`), no created_at trigger, and a shared writer used by both scripts.

#### X2 — Two dashboards, two sets of "facts" `[Certain]`
`ml_service.load_training_metadata` fallback (48,150 / 12,038 / 24 / 1.7588 / 0.4724), `fl_dashboard.load_training_metadata` fallback (12,795 / 2,559 / 18 / 1.8968 / 0.5995), the SOFA-validation drawer literals and `training_metadata.json` all disagree (CLINIQ C10). None is generated from the artifact actually deployed.

#### X3 — The legacy Streamlit `app.py`/`fl_dashboard.py` and the new React/FastAPI dashboard share `models/` and `db.py` `[Certain]`
Both stacks write `prediction_history`/`patient_vitals`; Streamlit scripts call `init_db()` and the migration comments; the FastAPI backend doesn't. Running both against the same DB interleaves two writers with different assumptions (e.g. different `Timestamp` formats), and any `db.py` change must be tested against both UIs.

---

## Suggested fix order
1. **D1–D3 / G1–G3 / X1** — one owner for the `fl_training` schema (explicit columns, no incompatible trigger, `replace` for the first load, existence checks that don't raise), `init_db()` in the FastAPI lifespan, absolute CSV paths + non-zero exit when nothing was imported.
2. **F1–F3** — guard and `.get()` defaults for the dashboard, `try/except` in the FL thread that sets `status="error"`.
3. **D4–D7** — context-managed connections, `busy_timeout`, transactions around insert+trim, remove DDL from reads.
4. **F4–F6, F9, X2** — relabel losses honestly, evaluate the aggregate for "best round", read hyperparameters/metadata from the artifacts instead of literals.
5. **U1–U5** — decide whether DP is real (then implement an accountant) or remove the claims; copy in `get_weights`; validate in `set_weights`; unit-aware, NaN-aware `get_trend`.
6. Remaining Medium/Low as polish.

*Static review plus targeted checks on temp DBs/model objects; Streamlit pages were not rendered and no training/simulation was executed.*

# Federated Training Files — Bug & Defect Report

**Files reviewed (completely, line by line):** [client.py](client.py) (155 lines), [server.py](server.py) (147 lines), [train_federated.py](train_federated.py) (1,029 lines).
Because they call straight into them, I also read the relevant parts of `model_utils.py` (training loop, FedProx, DP, metrics) and
`db.py` (`fl_training` helpers), and checked the committed artifacts (`models/training_metadata.json`, `scaler.pkl`, `tfidf_vectorizer.pkl`,
`feature_columns.pkl`, `shap_background.npy`) and the local `models/icu_monitor.db`.

**Branch / commit:** `main-copy` @ `1caee9a` (= `origin/main`).
**Method:** static read plus targeted verification. Confidence tags: **[Verified]** = reproduced by running code on a throw-away
temp database / inspecting the committed artifacts (your real DB was only opened read-only), **[Certain]** = follows directly from the code,
**[Likely]** = depends on a library/runtime detail, **[Verify]** = needs a check I couldn't do here.
**Not verified:** a full training run. It needs BigQuery access (Phase 0), `flwr` (not installed in this environment), and the
gitignored `data/fl_training/*.csv`. So Flower API behaviour is judged from the code and version pins only.
**Status:** report only — nothing has been fixed.

---

## Summary

| Severity | Count |
|---|---|
| Critical (script can't run / metrics invalid / results misreported) | 7 |
| High (wrong behaviour or misleading output) | 13 |
| Medium (robustness, reproducibility, correctness details) | 19 |
| Low (stale docs, dead code, polish) | 16 |

Most important:
1. **The reported test metrics are computed on data the model trained on** — the "held-out" set is a random 20 % of the clients' own training pool (T1).
2. **A fresh run cannot get through Phase 0 / re-training at all** — three separate crashes on the SQLite `fl_training` table (T2, T3, T4).
3. **Real FL (`server.py` + `client.py`) behaves differently from the simulation** — no LR decay, FedAvg instead of FedYogi, and the server overwrites the deployed model every round (T5, T6, T7).
4. Metadata written by training misdescribes the model (200 TF-IDF features vs 90 actual, "train_samples" = all data, etc.) (T8).

---

## CRITICAL

### T1 — "Held-out" test set is drawn from the training pool → metrics are inflated `[Verified]`
`train_federated.py:664-677`. When Phase 0 is skipped (the normal case once `fl_training` has data), the script does
`train_test_split(X_all, y_all, test_size=0.2, random_state=42)` where `X_all` is the **union of all three clients' training data**.
The clients then train on (their share of) that same data, so the "test" rows are rows that were trained on.
- Evidence: the committed `models/training_metadata.json` has `train_samples = 12,795` (= `X_all.shape[0]`, i.e. *all* client rows) and
  `test_samples = 2,559`, and **12,795 × 0.2 = 2,559 exactly**. The reported MAE 1.8968 / R² 0.5995 / AUROC 0.946 are therefore evaluated on training data.
- The same file's `evaluation_limitation` ("row-level split…") understates it; this is not just patient leakage, it is train/test row overlap.
- Even in the Phase-0 path the split is row-level over ICU stays, so repeat admissions/stays of one patient land on both sides (T13).
- **Fix direction:** persist the held-out test set from Phase 0 (e.g. a `fl_test` table / `.npy`) and always load it; never rebuild a test split from client data.

### T2 — Script crashes before Phase 0 on a fresh DB: `fl_training_exists()` has no table `[Verified]`
`train_federated.py:129` runs `_db_has_data = fl_training_exists()` at import time, and `db.fl_training_exists` does an unguarded
`SELECT COUNT(*) FROM fl_training`. `init_db()` deliberately does **not** create `fl_training`. On a clean DB the very first line of the pipeline raises
`OperationalError: no such table: fl_training` (reproduced on a temp DB), so Phase 0 — whose whole job is to create that table — can never start.
`init_db` is imported (l.51) but only called *inside* `run_preprocessing()`, after this check.

### T3 — Re-running preprocessing fails after `init_db()`: the `created_at` trigger breaks every insert `[Verified]`
`run_preprocessing()` calls `_init_db()` (l.577) then `DELETE`/`to_sql(..., if_exists="append")` into `fl_training`.
When `fl_training` already exists, `init_db()` installs `trg_ft_created_at`, which does `UPDATE fl_training SET created_at = …`. The table written by pandas
(`migrate_to_db.py` / this script) has **no** `created_at` column, so every insert fires the trigger and fails with
`OperationalError: no such column: created_at` (reproduced). The trigger and the pandas-created schema are mutually incompatible.

### T4 — `to_sql(append)` into the placeholder `fl_training` table fails `[Verified]`
Your local `models/icu_monitor.db` contains `fl_training` as a **3-column placeholder** (`client_id, created_at, updated_at`) with **0 rows**
(read-only inspection). `train_federated.py:585-590` deletes this client's rows and appends a 109-column frame; pandas `append` into an existing table
with a different schema raises `table fl_training has no column named HR_mean` (reproduced). `migrate_to_db.py` handles this with `if_exists="replace"` for the first client;
`train_federated.py` does not. Consequence on this machine: with 0 rows `fl_training_exists()` is False → Phase 0 runs → BigQuery → then this crash.

### T5 — Real-FL client never decays the learning rate (config key is never sent) `[Certain]`
`client.py:103`: `server_round = config.get("server_round", 1)`. In `train_federated.py` the strategy overrides `configure_fit` to inject `{"server_round": …}`
(l.843-852), but `server.py` uses a stock `FedAvg` with no `on_fit_config_fn`, so `config` is empty → every round uses round 1's LR (0.001), contradicting the
"mirrors train_federated.py" claims and the per-round LR printed by the client. The same missing config also means any other per-round setting can't be passed.

### T6 — `server.py` overwrites the deployed model with the *latest* round, not the best `[Certain]`
`server.py:50-60` saves the aggregate to `models/federated_model.pth` after **every** round — the file the dashboard loads — and writes the "best" model to
`federated_model_best.pth`, which is gitignored and never used. So (a) a run that is stopped/crashes/diverges leaves a half-trained or diverged model in production position
from round 1 onwards, with no backup; (b) after a full run the dashboard serves the last round even though the metadata reports `best_round`;
(c) `train_federated.py` does the opposite (saves the best-round weights to `federated_model.pth`), so the two entry points disagree on the artifact semantics.

### T7 — The two training paths use different algorithms but share one set of artifacts/metadata `[Certain]`
`server.py` = plain **FedAvg**, no initial parameters, no `server_round`, evaluation by unweighted mean; `train_federated.py` = **FedYogi** (η=0.01, β₁=0.9, β₂=0.99, τ=1e-3) with a custom `configure_fit`.
Both write `models/federated_model.pth` and `training_metadata.json`; `server.py` then overwrites the metadata with `aggregation: "FedAvg"` while the architecture string
still says "FedProx" and the client hyperparameters (μ=0.5, LR 0.001, "sweet spot with FedYogi") were tuned for FedYogi. Results from the two paths are not comparable but are stored as if they were.

---

## HIGH

### T8 — Metadata written by training is wrong or misleading `[Verified]`
`train_federated.py:984-1022` vs reality:
- `tfidf_features = TFIDF_MAX_FEATURES` (**200**) but `vocabulary=` mode ignores `max_features`; the real vocabulary has **90** terms (matches `tfidf_vectorizer.pkl`). The `input_features` (108) is right, the TF-IDF count is not.
- `train_samples = X_all.shape[0]` is *all* client data (including the 15 % per-client validation slices), not the training portion.
- `test_samples`/`final_*` describe the leaked test set (T1).
- `notes_sample_size = 283_208` is a *cap* (`min(NOTES_SAMPLE_SIZE, len(notes))`), not the actual number sampled.
- `optimizer`, `loss_weights`, `split_type` are hard-coded strings (e.g. loss text says "mild emphasis (~2× at SOFA=10)") rather than derived from the code path actually used.
- `server.py` rewrites this file keeping only a whitelist of keys, dropping `base_lr`, `lr_decay`, `oversample`, `loss_weights`, `optimizer`, `risk_classification_metrics`, `evaluation_limitation`, `bigquery_project`, `notes_sample_size`, `tfidf_ngram`, `dp_*`, `best_round` semantics etc. (T6/T7). If the server runs without a previous good file, `best_eval_loss: Infinity` (invalid JSON) is written when no evaluate round completed.

### T9 — Dashboard-facing stale numbers: defaults/UI disagree with what this script produces `[Verified]`
`ml_service.load_training_metadata` fallback (`train_samples 48150`, `test_samples 12038`, `best_round 24`, `final_mae 1.7588`, `final_r2 0.4724`, 108 features) and the dashboard's SOFA-validation drawer literals do not match
`training_metadata.json` produced by this script (12,795 / 2,559 / 18 / 1.8968 / 0.5995). See CLINIQ_BUGS.md (C10). Nothing in the training script writes the numbers the UI shows, so any retrain silently desynchronises them.

### T10 — First-run auto-skip logic can leave the script half-configured `[Certain]`
`skip = SKIP_PREPROCESSING or _db_has_data` (l.621): if `fl_training` has rows but `scaler.pkl` / `feature_columns.pkl` / `tfidf_vectorizer.pkl` are missing or from another vocabulary version,
Phase 1 loads `scaler.pkl` (l.629) and either crashes with `FileNotFoundError` or silently trains on a mismatched feature order. Conversely, once data exists there is no way to force Phase 0 except editing the constant or emptying the table.
`SKIP_PREPROCESSING=True` with an empty DB crashes on `get_fl_client_data(i)` / `df["sofa_score"]`.

### T11 — Feature alignment silently zero-fills missing columns `[Certain]`
`train_federated.py:641-644` and `client.py:71-74`: any feature in `scaler.feature_names_in_` missing from the table is silently set to `0` — and with the StandardScaler already applied, `0` means "the mean". A schema/vocabulary mismatch (e.g. after changing `_SOFA_VOCABULARY`) trains on mean-imputed garbage with no warning, and `db.get_fl_client_data` additionally `fillna(0)`s any NULL feature and drops NULL-label rows (only a `warnings.warn`). Extra unknown columns are dropped without notice.

### T12 — Missing values are imputed with 0 *before* scaling `[Certain]`
`train_federated.py:526`: `dataset.fillna(0)` runs on raw vitals before the split/scaler. A missing HR/MAP/SpO₂ becomes a physiologically impossible 0 (HR = 0, MAP = 0), contaminates the scaler's mean/std, and produces extreme z-scores that are then clipped to ±10. Note-less admissions (left join) also get all-zero TF-IDF rows, i.e. "no note" is indistinguishable from "note with none of the terms".

### T13 — Row-level split over ICU stays/admissions → patient leakage even in Phase 0 `[Likely]`
`notes_combined` is per `hadm_id`; `ml_data` is one row per `icustay_id`. Several ICU stays per admission (and several admissions per patient) share identical note text and often near-identical vitals, and `train_test_split` splits **rows**, so the same patient/admission appears on both sides. The script drops `subject_id`/`hadm_id` before splitting and has no group split (`GroupShuffleSplit`) — the metadata admits patient IDs were "unavailable", but they were in the frame (`drop_cols`, l.523).

### T14 — Possible target leakage from notes, and TF-IDF fit on all notes `[Likely]`
Notes are aggregated for the whole admission (`groupby("hadm_id")`, no `charttime` filter against `window_time`), so they can include discharge summaries and post-window text that describe organ failure, "septic shock", dialysis, etc. — information not available at prediction time. The vocabulary is a hand-picked whitelist of SOFA-related terms, which makes this more likely to leak outcome. Also `tfidf_vec.fit_transform` fits IDF on all sampled notes **before** the train/test split (IDF leakage from test into train).

### T15 — DP implementation does not give the stated guarantee and destroys the model `[Certain]`
`model_utils.apply_dp_to_update` (used by both client scripts; `USE_DP` defaults to False):
- Noise `N(0, (σ·S)²)` is added to every one of ≈ 25 k weights with σ=S=1 → noise L2 norm ≈ √25,000 ≈ 158 versus a clipped update norm ≤ 1; the update is pure noise.
- `estimate_privacy_budget` uses the classic Gaussian formula ε = √(2 ln(1.25/δ))/σ ≈ 4.8 per round (which is only valid for ε < 1) and linear composition → ε ≈ 480 after 100 rounds. That is not meaningful privacy, yet it is printed as "ε≈…" and written to metadata; δ is also not composed (should be rounds·δ).
- The mechanism is client-level, not "per patient" as the docstring says; no per-sample clipping, no secure aggregation, no privacy accountant; noise uses the global `np.random` (unseeded, not a CSPRNG).
- In `client.py`/`train_federated.py` the printed validation metrics are from the **pre-noise** local model, hiding the damage.
- `DP_SENSITIVITY`/`DP_SIGMA` must be edited identically in three files by hand.

### T16 — Evaluation reports unclipped predictions and unweighted selection signal `[Certain]`
`train_federated.py:942-956`: MAE/R² and per-band metrics are computed on raw network outputs (range −0.22 … 14.38 in the metadata), but the dashboard clips to 0…24 and `compute_sofa_risk_metrics` clips internally — so printed/stored MAE/R² and the risk metrics are computed on different prediction vectors. Best-round selection uses `avg_loss = sum(r.loss)/len(results)` (unweighted over clients) on the **clients' own validation slices** (`aggregate_evaluate`), not on the held-out set; with unequal client sizes (33/33/34 %, or Non-IID) it isn't a sample-weighted loss. The model also never predicts above 14.4, i.e. the high-risk tail is under-fit (alert recall 56.9 %, high-class recall 19.8 % in the committed metadata) — not surfaced anywhere in the script's summary.

### T17 — Non-IID branch uses different data than the IID branch `[Certain]`
`train_federated.py:716-724`: Non-IID path re-splits `X_train_pool`; that variable is only defined in the "reconstructed" branch (an 80 % subset, and in the Phase-0 path it is undefined and the code falls back to `X_all`). So with `USE_NONIID_SPLIT=True` the clients train on 80 % of the data in one path and 100 % in the other, and the SHAP background pool differs too (`X_bg_pool = X_train_pool` vs `X_all`).

### T18 — Hospitals are not private: every client reads the shared DB `[Certain]`
`client.py:66` calls `get_fl_client_data(CLIENT_ID)` which queries the **same** `models/icu_monitor.db` that holds all three hospitals' rows. "Loads its own private dataset … NO patient data leaves" is only true by convention; each process can read all clients' data (and the dashboard DB with patient vitals). Nothing validates `--client_id` (1000 → empty frame → `train_test_split` error), and nothing stops two terminals from using the same id while a third hospital never joins.

### T19 — No transport security or client authentication in real FL `[Certain]`
`server.py`/`client.py` use an insecure gRPC channel on `127.0.0.1:8080`, with no TLS, tokens or client allow-list. Any process that connects counts as a "hospital" (`min_available_clients=3`), can send arbitrary weights (model poisoning), and weights are visible in clear text if the address is ever changed from loopback. Aggregation is un-robust (plain mean) so one bad client controls the model.

### T20 — Flower API usage likely incompatible with current releases `[Verify]`
`requirements.txt` pins only `flwr[simulation]>=1.8.0`. The scripts use `fl.simulation.start_simulation`, `fl.client.start_numpy_client`, `fl.server.start_server`, `NumPyClient.to_client()` and subclass `strategy.FedYogi/FedAvg` with overridden `configure_fit` — APIs Flower has deprecated in favour of `flwr run` / `ServerApp` / `ClientApp` (and `client_resources`/Ray simulation backend requirements). An unpinned install may fail or change behaviour; `flwr` isn't installed in this environment so I couldn't confirm which APIs still exist.

---

## MEDIUM

### T21 — No `if __name__ == "__main__":` guards; everything runs on import `[Certain]`
All three scripts execute at import time (argument parsing in `client.py`, model loading + `start_server` in `server.py`, the entire 1,000-line pipeline including BigQuery/FL simulation in `train_federated.py`). `import train_federated` (e.g. from a test, notebook or a spawned worker process on Windows) starts training. Ray/multiprocessing "spawn" on Windows re-imports the main module, making accidental recursion a real risk.

### T22 — Nothing is seeded → runs are not reproducible `[Certain]`
No `torch.manual_seed`, `np.random.seed`, `random.seed`. Initial weights (`ICUModel(input_dim)` in `_init_model` and in each `HospitalClient`), `DataLoader(shuffle=True)`, `WeightedRandomSampler`, DP noise (`np.random.normal`) and Ray scheduling are all unseeded (only the sklearn splits use `random_state=42`). Comments like "sweet spot", "best at round 24-29", "R²=0.2229" cannot be reproduced; the committed metadata (best_round 18) already contradicts the comments.

### T23 — Optimiser state is reset every round `[Certain]`
`model_utils.train_model` creates a fresh `AdamW` each call, so Adam's moments restart every FL round (together with the weight decay applying to a global model reloaded each round). Combined with the server-side Yogi optimiser this is a double-adaptive scheme that isn't discussed, and `eta_l=BASE_LR` passed to `FedYogi` is informational only while the actual client LR follows a different schedule.

### T24 — `model_utils.train_model` default `oversample=True` contradicts docs `[Certain]`
Default `oversample=True`, while the docstring of `get_sample_weights` says oversampling is unused/harmful; callers must remember to pass `False` (both do). `WeightedRandomSampler(... replacement=True)` also changes the number of distinct samples per epoch. The proximal term is recomputed with a Python-level `sum(...)` over all parameters each batch (slow) and its loss is included in the reported `loss`, so "loss" printed per round mixes data and prox terms.

### T25 — Validation split is taken *inside* each client's already-split data `[Certain]`
`HospitalClient.__init__` (both files) uses `train_test_split(X, y, test_size=0.15, random_state=42)` — a 15 % validation set of the client's training share. Reported per-client MAE/R² in the logs and `best_round` selection use this validation set, which is also part of `X_all` → part of the leaked "test" set (T1) and the SHAP background. There are effectively three overlapping evaluation sets.

### T26 — Class imbalance is only handled by a mild loss weighting `[Certain]`
Training data are ≈ 6 % high-risk, loss weight `1 + 0.5·y` normalised by batch mean; no stratification of client splits in the IID path (`sk_shuffle` + slice, l.564-571) or of the client validation split; client class mix can differ and the stratified Phase-0 split only protects the global train/test, not hospitals.

### T27 — Hard-coded personal infrastructure and secrets-adjacent config `[Certain]`
`BIGQUERY_PROJECT = "mimic-project-2"`, dataset `"Dataset"`, location `"US"` are hard-coded; no environment/CLI overrides. MIMIC-III is credentialed data; the script pulls **all** notes for the project into memory (`SELECT subject_id, hadm_id, text FROM …clinical_notes` → `.to_dataframe()`), which is multi-GB, and `to_dataframe()` needs the `db-dtypes` (and usually `pyarrow`) package, which `requirements.txt` doesn't list (`google-cloud-bigquery` alone isn't enough on recent versions) `[Likely]`.

### T28 — Sampling notes then grouping loses data and biases coverage `[Certain]`
`notes.sample(n=…)` samples **notes**, not admissions, before grouping by `hadm_id`; admissions end up with a random subset of their notes (or none → all-zero TF-IDF after the left join). `NOTES_TEXT_LIMIT` truncation (10,000 chars) is applied after concatenation in arbitrary order, so what survives depends on the sampling order, not clinical relevance. Notes with NULL `hadm_id` are silently dropped by `groupby`.

### T29 — Cleaning regex removes clinically meaningful tokens `[Certain]`
`_clean_clinical_text` deletes all numbers and every non-letter (the `[^a-z\s]` substitution), turning `PaO2`, `FiO2`, `O2`, `SpO2`, `T-bili`, `K+`, `pH`, `HCO3`, `CO2` into fragments (`pao`, `fio`, `hco`, `bili`) or dropping them; `token_pattern` requires ≥ 3 letters so `o2`, `ph`, `na`, `k` vanish. The whitelist even contains such leftovers (e.g. `hco`, which is `HCO3` with its digit stripped), so the vocabulary is partly shaped by cleaning artefacts rather than clinical meaning.

### T30 — `stratify` and splits can raise on small data `[Certain]`
`train_test_split(..., stratify=(y >= 10))` raises if a class has < 2 members; `rng.choice(len(X_bg_pool), SHAP_BG_SAMPLES, replace=False)` raises if there are fewer than 300 rows; `np.polyfit`-style helpers and `r2_score` on tiny per-band masks give NaN/negative values. No guards.

### T31 — Misleading printed/progress statistics `[Certain]`
Server log LR (`BASE_LR * LR_DECAY**(round-1)`, l.871) ignores the `max(1e-4, …)` floor used by clients; `loss` printed by clients is the weighted MSE plus the prox term, labelled just "loss"; AUROC/PR-AUC print uses truthiness (`if _alert["auroc"]`) so a legitimate value of 0.0 prints "N/A"; the final banner says "Restart the app … `streamlit run app.py`" (the dashboard is FastAPI/React now).

### T32 — Overwrites artifacts with no backup, non-atomic writes `[Certain]`
`joblib.dump(scaler/tfidf/feature_columns)`, `np.save(shap_background)`, `torch.save(federated_model.pth)` and `training_metadata.json` are overwritten in place sequentially. A crash mid-run (e.g. after Phase 0 but before FL finishes) leaves **new scaler/vocabulary with the old model** — the dashboard then loads a model whose input features no longer match its scaler/TF-IDF. There is no versioning or hash tying the model to its preprocessors.

### T33 — Preprocessors/artifacts not versioned against sklearn `[Verified]`
`scaler.pkl`/`tfidf_vectorizer.pkl` committed in the repo were pickled with scikit-learn **1.9.0** (loading in 1.7.1 emits `InconsistentVersionWarning`), while `requirements.txt` says "fitted with 1.6.1" and caps `<2.0.0`. `joblib.dump` doesn't record the version in the metadata; retraining with a different sklearn than inference will silently change behaviour.

### T34 — `client.py` metrics/behaviour inconsistencies `[Certain]`
- `fit` returns an empty metrics dict, so the server can't log client losses/MAE; `evaluate` returns the same `mae/r2` but the server ignores them (`aggregate_evaluate` only averages `loss`).
- The client trains on `parameters` supplied by the server but never verifies shapes: a feature-count mismatch between server (`scaler.n_features_in_`) and client (`feature_columns`) surfaces as a cryptic `load_state_dict` size error.
- `set_weights` relies on `state_dict()` order/keys; any model change breaks silently if shapes coincide.
- `get_parameters(self, config)` returns the *current random init* of each client's own model: with FedAvg (no `initial_parameters`) the server asks one arbitrary client, so the starting point depends on which client Flower picks.
- Unused imports (`pandas`, `estimate_privacy_budget`).

### T35 — `server.py` evaluation bookkeeping `[Certain]`
`aggregate_evaluate` averages `r.loss` unweighted and prints with `end=""` string concatenation; `_best_round`/`_best_loss` are module globals; if `results` is empty the "best" never updates and the final line formats `inf`. `strategy` is constructed at import with `min_*_clients=3` hard-coded, `fraction_*=1.0` — one slow/dead client blocks the round forever (no timeout/`round_timeout` in `ServerConfig`).

### T36 — Docstring "usage" instructions refer to a different layout `[Certain]`
`server.py` header says `cd icu_monitor`, but the repo root isn't called that; `client.py` paths (`models/`) only work when run from the repo root. `MODEL_PATH = "models/"` is cwd-relative in all three files, while the backend resolves paths from `__file__` — running from another directory writes artifacts to the wrong place.

### T37 — Fixed 3-hospital assumption everywhere `[Certain]`
`range(3)`, `num_clients=3`, `min_*_clients=3`, `hospital_names` (3 entries), `% 3` indexing in splits and `fractions` arrays. In IID mode the "General/Mixed/Cardiac" names are meaningless (random split), but they're stored in metadata as if they described the sites.

### T38 — Non-IID helper has edge-case bugs `[Certain]`
`make_noniid_split._split` clips cumulative cuts to `[0, n]` and slices `indices[prev:cut]`; with rounding the last part can be empty/short, and tiny strata produce empty clients (an empty client then crashes in `train_test_split`). Fractions per class are hard-coded and the function mutates nothing but is only reachable via the global flag.

### T39 — Re-importing heavy modules inside functions `[Certain]`
`import re`, `TfidfVectorizer`, `StandardScaler`, `sk_shuffle`, `sqlite3`, `pandas as _pd_local`, `db as _init_db/_DB_PATH` are imported inside `run_preprocessing`, alongside module-level `import os` placed mid-file (l.128, after code that uses it conceptually). `pandas` is imported twice under two names. It works, but it hides dependency errors until Phase 0.

---

## LOW

### T40 — Stale docstring and comments contradict the code `[Certain]`
`train_federated.py` docstring and comments say: "600 bigram features, 80K note sample", "20 rounds", "Weighted MSE loss (1 + SOFA×3 → High-Risk ~2.4×)", "LR_DECAY=1.0 = no decay", "top 600 most informative clean terms", `NOTES_SAMPLE_SIZE … increased from 80k → 200k` (value is 283,208), "all 3000 chars" (limit is 10,000), "FedAdam" vs FedYogi in section headers/`initial_parameters # required by FedAdam`, R² figures (0.2229 / 0.28 / 0.08) vs the committed 0.5995. The current code uses 200/90 features, 100 rounds, `1 + 0.5·SOFA`, `LR_DECAY=0.99`.

### T41 — ~250 lines of dead code in `run_preprocessing` `[Certain]`
`_MEDICAL_VOCABULARY_REF` (~130 terms with duplicates such as `"cardiac arrest"`, `"hypertension"`, `"blood"`), `_ADMIN_STOP` (~250 words with many duplicates: `"general"`, `"multiple"`, `"unable"`, `"prior"`, `"valuables"`…) and `_all_stop_words` are built and never used (the code even says "kept for reference"). They also make the key whitelist (`_SOFA_VOCABULARY`) hard to find.

### T42 — Whitelist contains terms that cannot match under the tokenizer `[Verify]`
`token_pattern` is `\b[a-zA-Z][a-zA-Z]{2,}\b` with `ngram_range=(1,2)`; vocabulary entries like `"plt blood"`, `"ptt inr"`, `"blood cultures"` rely on adjacent tokens after cleaning. Several single-word terms (`"vent"`, `"levophed"`, `"totbili"`, `"gcs"`, `"unresponsive"`, `"delirium"`, `"aki"`) may be extremely rare in MIMIC text after cleaning; the script never reports which of the 90 terms are all-zero columns (those become constant features that StandardScaler handles but that waste inputs and SHAP slots).

### T43 — Variables defined and never used `[Certain]`
`train_federated.py`: `skip` (after use), `_all_stop_words`, `X_train_pool`/`y_train_pool` when IID, `y_train_pool` in the else path, `global_weights` in `fit` when `USE_DP` is False, `warnings.filterwarnings` placed after most imports. `client.py`: `pd`, `estimate_privacy_budget`. `server.py`: nested `import os` inside `try`.

### T44 — Print/log formatting `[Certain]`
Mixed use of `print` with emoji/arrow characters (`←`, `✓`, `→`, `σ`, `η`, `μ`) in a Windows cp1252 console raises `UnicodeEncodeError` unless `PYTHONIOENCODING=utf-8` is set (I hit this when printing the dashboard's predict output; the same applies to these scripts' banners). JSON metadata itself is safe (`json.dump` escapes non-ASCII).

### T45 — No CLI options / configuration `[Certain]`
All hyperparameters are module constants duplicated in `client.py` and `train_federated.py` ("keep in sync" comments), with no shared config file, argparse flags or environment overrides, so the real-FL clients can silently drift from the simulation (they already do: T5, T7).

### T46 — Misc correctness nits `[Certain]`
- `HospitalClient.fit` returns `len(self.X_train)` as the weight but evaluation weight is `len(X_val)`; fine, but `train_samples` in metadata counts neither.
- `final_loss` returned by `train_model` is divided by `len(loader)` — with `WeightedRandomSampler` that's the number of batches, OK; with `drop_last` absent the last small batch has the same weight as full batches.
- `compute_sofa_risk_metrics` uses `np.digitize` bin edges `[5, 10]` (risk bands) but the alert threshold 8 and "high ≥ 10" are independent constants, defined again in `ml_service`/frontend (see CLINIQ_BUGS H10).
- `get_trend` (shared with the dashboard) uses a fixed 0.1 slope regardless of unit (CLINIQ_BUGS M25).
- `estimate_privacy_budget` returns `inf` for σ ≤ 0 but `train_federated` still prints "ε≈inf".

### T47 — `models/_backup/` and tracked artifacts `[Certain]`
The training scripts write straight into the tracked `models/` directory (`federated_model.pth`, `scaler.pkl`, `tfidf_vectorizer.pkl`, `feature_columns.pkl`, `shap_background.npy`, `training_metadata.json`), so every retrain dirties tracked binary files; there's no run-ID/timestamped output directory, no model card, and nothing records the git commit / data version used.

### T48 — Real-FL documentation promises not implemented `[Certain]`
Docstrings claim "Best model is saved after every improvement round" (it also saves *every* round to the production path, T6), "Sends only model weights … NO patient data leaves" (true on the wire, but see T18), and "All hyperparameters below mirror train_federated.py" (not true: no LR schedule delivery, no FedYogi, `OVERSAMPLE` hard-coded False in the client instead of the shared constant).

### T49 — `fl_training` helper gaps affecting training `[Certain]`
`db.get_fl_client_data` has no `ORDER BY`, so row order (and therefore the `random_state=42` split) depends on SQLite's physical order; after a re-import the validation set changes. It also relies on `idx_ft_client`, which is only created by `migrate_to_db.py` (`CREATE INDEX` after replace) and is lost whenever pandas recreates the table with `replace`.

### T50 — Summary printing assumes data in every band `[Certain]`
Per-hospital print divides by `len(y_h)`; an empty client (Non-IID edge, or a bad `--client_id`) raises `ZeroDivisionError` before the real error.

### T51 — `evaluate_model` returns plain `mse` as Flower loss `[Certain]`
Flower's `loss` is the unweighted MSE while training optimises the weighted MSE (+ prox), so "best round" is chosen by a metric that emphasises low-SOFA rows — exactly the class the weighting de-emphasises (the training/evaluation mismatch the `weighted_mse_loss` docstring itself describes) and the opposite of the stated high-risk priority.

### T52 — Windows/paths and encoding `[Certain]`
`open()` calls (`server.py:134,145`, `train_federated.py:1024`) use the locale encoding; harmless today only because `json.dump` escapes non-ASCII — any literal non-ASCII text added later would break on Windows cp1252. Paths are built with string concatenation (`MODEL_PATH + "x"`) instead of `os.path.join`.

### T53 — `import os` and ordering `[Certain]`
`os` is imported at l.128, after `fl_training_exists()` is already used conceptually at the top and after functions referencing it are defined; harmless but linter-flagged (E402) alongside the in-function imports (T39).

### T54 — Result summary excludes crucial failure modes `[Certain]`
The final summary prints overall MAE/R² and band MAEs but not the confusion matrix, high-risk recall, or the leakage caveat; the training script never warns that predictions top out near 14 (SOFA ≥ 15 patients are never predicted), nor that R² on the high band is negative.

### T55 — Unused/duplicated configuration constants `[Certain]`
`TFIDF_MAX_DF`, `TFIDF_MIN_DF`, `TFIDF_MAX_FEATURES`, `NOTES_TEXT_LIMIT` semantics, `SERVER_TAU`, `SERVER_BETA*` are defined but several have no effect (vocabulary mode ignores the TF-IDF ones), yet they are printed and stored as if active.

---

## Suggested fix order
1. **T1** (persist a true held-out set) — until this is fixed every reported metric is unreliable; then regenerate `training_metadata.json`.
2. **T2–T4** (the `fl_training` table lifecycle: create schema in `init_db`, drop the incompatible trigger or add the columns, use `replace` for the first load, wrap `fl_training_exists`).
3. **T5–T7** (decide what the real-FL path is: either implement the same FedYogi strategy + `server_round` config in `server.py`, or document it as a toy; never write the production model every round).
4. **T8–T9, T33** (metadata derived from the actual run, versions recorded; keep the dashboard's numbers in sync from that file).
5. **T11–T14** (explicit schema checks instead of zero-fill, group-aware split, time-bounded notes, fit IDF on train only).
6. **T15** (either remove the DP claims or implement a proper accountant), T18–T20 (privacy/auth/Flower API).
7. Everything else as polish; add `__main__` guards and seeds first since they make the rest testable.

*Static review plus targeted verification on temp databases; no full training run was possible here (no BigQuery access, `flwr` not installed, FL CSVs not in the repo).*

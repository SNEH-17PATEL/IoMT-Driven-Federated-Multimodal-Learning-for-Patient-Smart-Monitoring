# Model Improvement Report — raising SOFA R² on the federated model

**Branch:** `feature/improve-training-r2` (local only, never pushed to `main`)
**Date:** 2026-11-22
**Result in one line:** federated R² **0.466 → 0.518** (MAE 1.81 → 1.69) on the same patient-grouped test split,
with the same federated neural network and SHAP still naming real words. The 0.6 target was **not** reached, and the
evidence below shows why it is out of reach with the data we have.

---

## 1. Goal and constraints

The task was to raise R² from ~0.48 to above 0.6. The constraints given were:

| Constraint | How it was respected |
|---|---|
| Do not tamper with the dataset to flatter the metric (e.g. removing high-risk patients) | The label distribution was never touched. Train and test keep the same ~6 % High-risk share. |
| Keep the technology and methodology | Same federated MLP, FedAvg aggregation, FedProx local training, plain MSE, AdamW, IID 3-hospital split. |
| No tree-based models | XGBoost appears only as a *measuring stick* for how much signal the data holds, and as a *ranking tool* to choose note words (preprocessing). It is never the deployed model. |
| Keep SHAP explainability | Every model input is still a named quantity (a vital or a real note word). An SVD/embedding approach was tried and **rejected** for this reason (section 7). |
| Work on a feature branch, never push | All work is on `feature/improve-training-r2`. Nothing was pushed. |

## 2. Data used

A hand-off package (`handoff_full_dataset.zip`) containing MIMIC-III derived data was supplied. It is **restricted data
(PhysioNet credentialed access)** and is *not* committed to git.

| File | Content |
|---|---|
| `models/icu_monitor.db` → `fl_training` | 41,798 training rows = 108 scaled inputs + `sofa_score` + `client_id` (3 hospitals) |
| `models/fl_test_X.npy`, `fl_test_y.npy` | 10,450-row held-out test set from the *old* split |
| `bigquery_tables/ml_dataset_final_last_row_per_stay.parquet` | 52,248 rows × 24 cols: last hourly row of each ICU stay (18 vitals/CV features + `sofa_score` + ids) |
| `bigquery_tables/clinical_notes.parquet` | 35,799 rows: bedside notes (24 h before the last hour), one row per admission |

The parquet vitals table contains **only** the 18 numeric features the model already used (HR, RR, SpO₂, Temp, SBP/DBP/MAP,
their trends, GCS eye-opening, stress score). There are **no** creatinine, bilirubin, platelet or PaO₂ columns — i.e. none
of the organ-function measurements that actually define SOFA.

## 3. Baselines

### 3.1 Measured on the project's own test set (10,450 stays)

| Baseline | R² |
|---|---|
| Predict the mean SOFA for everyone | −0.0001 |
| Ridge regression on the 108 features | 0.394 |
| Federated MLP (original setup), reproduced | **0.478** (MAE 1.811, best round 21 of 40) |

### 3.2 Documented in the project history (`docs/Project_Context.md`, `docs/MODEL_DISCUSSION.md`)

Original FL notebook 0.04 → … → 0.4725 for the previous documented best. Linear regression was documented as ≈0.15 (old
dataset; on the current data it is 0.394). Gradient-boosting ensembles were dropped earlier because trees cannot be
averaged across hospitals (FedAvg) — hence "no tree models" here.

### 3.3 Published research (from search results; papers were not read in full — verify before citing)

| Paper | Predicts | Reported |
|---|---|---|
| CNN + Random Forest on 7 bedside vitals, MIMIC-III (PubMed 33157471) | SOFA value before sepsis onset | MAE 0.66, RMSE 1.23, r = 0.863; no R² |
| Temporal CNN on MIMIC-IV (arXiv 2608.29301) | Future SOFA | MAE 1.46, R² 0.715; a "repeat last SOFA" baseline got R² 0.707 |
| DeepSOFA (arXiv 1802.10238) | Mortality from hourly data | AUC only |
| FL on MIMIC-III (arXiv 2005.11756), FLICU (arXiv 2205.15104) | Mortality | AUROC ≈ 0.85 federated vs 0.855 centralized |

No paper was found that regresses SOFA from vitals + notes in a federated setup. The literature's R² ≈ 0.7 comes from
models that see hours of prior time series (and a copy-last-value baseline already gets 0.707), so it is **not** a
like-for-like target for a model that sees one summary row per stay.

## 4. Ceiling analysis — can tuning alone reach 0.6? (No.)

Centralised (non-federated) models on the same 108 features and the official test set:

| Model | R² |
|---|---|
| XGBoost depth 4, 400 trees | 0.487 |
| XGBoost depth 6, 600 trees | 0.503 |
| XGBoost, heavily tuned (1,500 trees, regularised) | 0.505 |
| XGBoost on a √SOFA target | 0.483 |
| MLP (256-128-64, dropout 0.3), 5–30 epochs | 0.464 – 0.484 |

A strong boosted model with no federation constraint tops out at ≈0.50 on these inputs, so the information, not the
tuning, is the bottleneck (scripts `ceil2.py`, `ceil3.py`).

## 5. Feature experiments (patient-grouped split, centralised XGBoost as the probe)

To avoid train/test leakage, experiments used `GroupShuffleSplit` by `subject_id` (80/20, seed 42) on the 52,248
parquet rows. XGBoost is used here purely to measure how much signal each feature set holds.

| Feature set | R² |
|---|---|
| 18 vitals only | 0.392 |
| + 90-word SOFA whitelist (the old pipeline) | 0.485 |
| + 3,000 learned TF-IDF terms (uni+bigrams, 30k chars/note) | **0.549** |
| + note length | 0.548 |
| + 56 numeric lab values regex-extracted from notes (creatinine, bilirubin, platelets, INR, lactate, …) | 0.549 (no gain; only 7 % of notes contain a creatinine value) |
| lab regex features alone, no TF-IDF | 0.438 |
| + 10,000 learned TF-IDF terms | 0.556 |

Takeaways: the notes carry real extra signal; the hand-picked 90-word whitelist was leaving ≈0.06 R² on the table;
regex lab extraction adds nothing; ≈0.55 is the ceiling even for a tree model (scripts `feat1.py`, `feat2.py`, `xgb10.py`).

## 6. Can the *neural network* use that signal?

The deployed model must stay an MLP. The same 3,000 learned terms fed straight to the MLP only reached ≈0.49 (it overfits
thousands of sparse columns).

| MLP input | R² |
|---|---|
| 3,000 TF-IDF terms ×10 | 0.482 – 0.489 |
| 3,000 terms standardised | 0.465 – 0.483 |
| top-300 terms by correlation | 0.477 – 0.491 |
| 3,000 terms → SVD 64 / 128 / 256 components | 0.511 / **0.523** / 0.507 |
| 10,000 terms → SVD 128 / 256 | 0.527 / 0.520 |
| SVD with dropout 0.5 | worse than dropout 0.3 every time |

## 7. Why SVD was rejected despite the gain

Compressing the notes with SVD lifted the MLP to ≈0.52, but each SVD component is a blend of thousands of words and
has no readable name. SHAP could then no longer say "levophed raised the score" — it would say "note_svd_17 did". The
requirement was to keep SHAP explanations integral, so a first SVD-based implementation was **built, tested and
reverted** (`git checkout`) and is not part of this branch.

## 8. The adopted approach — data-chosen note words (SHAP-compatible)

Instead of compressing, keep individual words but choose them from data. On the same patient-grouped split, MLP
(centralised, 20 epochs):

| Note inputs | R² |
|---|---|
| 90-word whitelist (current) | 0.451 – 0.460 |
| 90-word whitelist + weight decay 1e-2 + 20 % input dropout | 0.449 – 0.469 |
| **top-200 XGBoost-ranked words** | **0.506 – 0.521** |
| top-200 + weight decay/input dropout | 0.479 – 0.526 |
| top-500 words | 0.490 – 0.510 |
| whitelist + top-200 | 0.498 – 0.525 |
| whitelist + top-500 | 0.493 – 0.516 |

200 words was the sweet spot (more words overfit). Script: `words1.py`.

**Pipeline implemented (Phase 0 of `train_federated.py`, `NOTE_VOCAB = "data"`):**

1. Split ICU stays 80/20 **grouped by patient** (`subject_id`) — repeat stays of one patient never straddle train/test.
2. Fit a 3,000-term candidate TF-IDF (uni+bigrams, English stop-words removed) on **training notes only**.
3. Rank candidates with an XGBoost regressor on training rows only (gain importance); keep the top 200 words.
4. Build the final `TfidfVectorizer(vocabulary=<200 words>, preprocessor=clean_clinical_text, stop_words="english")`,
   fit on training notes, and save it as `tfidf_vectorizer.pkl`.
5. Scale (StandardScaler, clip ±10), split into 3 IID hospital datasets (stratified by High-risk), save as before.

The first attempt picked filler words ("and", "from", "admitted from"); adding English stop-word removal fixed that, and
the final list is clinical (`acidosis`, `acute distress`, `alert oriented`, `abd pain`, …).

## 9. Federated confirmation (identical split, same algorithm)

All federated numbers below use the harness `tune_training.py`, which re-implements the exact loop of
`train_federated.py` (3 IID hospitals, FedAvg weighted by sample count, FedProx μ, AdamW, LR decay 0.99, grad-clip 1.0,
best round chosen by clients' validation MSE) without Flower.

| Run (40 rounds unless noted) | R² | MAE | Best round |
|---|---|---|---|
| Old: 90-word whitelist (patient-grouped split) | 0.466 | 1.810 | 31 |
| **New: 200 data-chosen words** | **0.518** | **1.693** | 30 |
| New, 5 local epochs/round | 0.518 | 1.705 | 31 |
| New, batch 128 + lr 0.002 | 0.519 | 1.705 | 30 |
| New, FedProx μ = 0.05 | 0.505 | 1.708 | 11 |
| New, FedProx μ = 1.0 | 0.513 | 1.705 | 40 |
| New, **100 rounds** | 0.518 | 1.693 | 30 (identical — extra rounds never improve) |
| New, wider net 512-256-128, dropout 0.4 | 0.525 | 1.688 | 34 |

Decisions: no setting moves R² by more than 0.01, so **training hyper-parameters are unchanged**. The wider network's
+0.007 is within single-run noise and does not meet the "change the model only for a significant gain" bar, so the
architecture stays 256-128-64.

## 10. Final model

Trained for 100 rounds with the new pipeline (best round 30), evaluated on the 10,503-stay patient-grouped test set:

| Metric | Value |
|---|---|
| **R²** | **0.5184** |
| **MAE** | **1.693** SOFA points |
| High-risk (SOFA ≥ 10) AUROC | 0.9326 |
| High-risk PR-AUC | 0.6550 |
| Alert recall @ predicted SOFA ≥ 7 | 67.7 % |
| Alert specificity | 96.3 % |
| Input features | 218 (18 vitals/CV + 200 note words) |

## 11. What changed in the repository

| File | Change |
|---|---|
| `train_federated.py` | New config block (`NOTE_VOCAB`, `NOTE_CANDIDATE_TERMS`, `NOTE_TOP_TERMS`, `PARQUET_DIR`); Phase 0 can read the two parquet tables instead of BigQuery; new `_data_word_preprocessing()` (grouped split + train-only word selection); tail of Phase 0 factored into `_scale_split_save()`. Old 90-word path kept as `NOTE_VOCAB = "whitelist"`. Phases 1–6 (FL training) untouched. |
| `model_utils.py` | New module-level `clean_clinical_text()` — saved inside the vectorizer so the backend cleans a live note exactly like the training notes (also fixes the previous train/serve text-cleaning mismatch). |
| `tune_training.py` | Flower-free federated harness; `--data-dir` option; optional `return_model`. |
| `models/*` | Retrained artifacts: `federated_model.pth` (218→256→128→64→1), `scaler.pkl`, `tfidf_vectorizer.pkl`, `feature_columns.pkl`, `shap_background.npy`, `fl_test_X/y.npy`, `training_metadata.json`. |
| `fl_dashboard.py`, `server.py`, `backend/services/ml_service.py` | Stale "108 features / 90 words" display text → 218 / 200. No logic change. |
| `experiments/model_r2/` | Every exploration script that produced the numbers in this report. |
| `docs/MODEL_IMPROVEMENT.md` | This report. |

The backend needed **no logic change**: it already does `tfidf.transform([note])`, `get_feature_names_out()` and
`scaler.feature_names_in_`, and builds `ICUModel(len(feature_cols))`.

## 12. Verification performed

* **Backend end-to-end** (`experiments/model_r2/verify_backend.py`): `ml_service.load_artifacts()` loads the new model
  (input_dim 218, first layer 256×218). A synthetic severe patient (intubated, levophed, septic shock, hypotension, low
  GCS) scores **SOFA 9.6**, and SHAP's top features are `levophed`, `GCS_eye_opening`, `sedated`, `acute distress`,
  `acidosis`, `intubated` — real words, as required.
* **Tests:** 131 FL unit tests pass (`tests/fl/unit`, excluding 3 files that need the filled local DB); 211 of 218
  backend tests pass. The 7 failures are all `test_api_report.py` and fail with `LLM error 401 – Invalid API Key` (the
  Groq key is a placeholder in this environment) — unrelated to the model. `test_client_logic.py` fails identically on
  the untouched code because the local `fl_training` table is empty (pre-existing).

## 13. Caveats and honest limitations

1. **0.6 was not reached.** Final R² is 0.518. Even an unconstrained XGBoost reaches only ≈0.556 on these inputs, so
   0.6–0.7 needs new information (real lab values, or time-series history), not further tuning.
2. **Numbers are not comparable to the old 0.48.** The old test split was row-level (a patient could appear in train
   and test). The new split is patient-grouped and stricter. The fair before/after is 0.466 → 0.518 on the *same* new
   test patients. `fl_test_X/y.npy` therefore changed (10,450 → 10,503 rows).
3. **Not trained with Flower.** Flower's Ray backend crashes on Windows (`access violation`). The model was trained with
   the equivalent harness; `training_metadata.json` says so. Re-run `python train_federated.py` on Linux/macOS (set
   `PARQUET_DIR` to the folder with the two parquet files) to get the true Flower number — expect it to be within ±0.02.
4. **Single seed.** Differences below ≈0.01 (e.g. the wider network) are within run-to-run noise.
5. **XGBoost dependency.** Phase 0 now needs `xgboost` to rank words (preprocessing only). Add it to the environment
   (`pip install xgboost`).
6. **SHAP keyword filter.** The backend displays only SHAP features matching `CLINICAL_KEYWORDS`
   (`backend/services/ml_service.py`); the new data-chosen words may not match and then appear only through the existing
   fallback (shown when fewer than 3 clinical features match). Widening that list is a small follow-up.
7. **Local database.** The local `models/icu_monitor.db` has an empty `fl_training` table; the new training rows were
   *not* copied into it (restricted data; gitignored). The backend's conformal interval therefore keeps its default
   (±2.9). To refresh it, run Phase 0 against your DB.
8. **Interpretability trade-off.** Selected words (e.g. `admission note`, `abd pain`) reflect documentation patterns as
   well as severity. They are legitimate, readable predictors, but SHAP attributions on them should be read with care.

## 14. How to reproduce

```powershell
# 1. data: unzip the hand-off package to a folder, e.g. D:\handoff\handoff_full
# 2. in train_federated.py set:  PARQUET_DIR = r"D:\handoff\handoff_full\bigquery_tables"
# 3. Phase 0 + federated training (Linux/macOS for Flower; Windows: use the harness below)
pip install xgboost google-cloud-bigquery flwr[simulation]
python train_federated.py
# Windows / Flower-free:
python experiments/model_r2/run_phase0.py      # builds the 218-feature dataset in a scratch copy
python experiments/model_r2/build_final.py     # trains + writes all artifacts
python experiments/model_r2/verify_backend.py  # loads them through the backend and runs SHAP
```

The scripts in `experiments/model_r2/` contain `<WORKDIR>` / `<REPO>` placeholders for local paths — set them to your own
folders before running. See `experiments/model_r2/README.md` for what each script shows.

## 15. Backups and rollback

The previous model artifacts are in git history (parent of the improvement commit) and were copied aside before being
replaced. To roll back: `git checkout main -- models/` (or the commit before `f9a3b55`).

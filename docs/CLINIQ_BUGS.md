# ClinIQ Dashboard — Bug & Defect Report

**Scope:** the ClinIQ dashboard as it exists on `origin/main` @ `1caee9a` ("solved the bugs/errors"):
React + Vite frontend (`frontend/`), FastAPI backend (`backend/`), SQLite layer (`db.py`, `migrate_to_db.py`),
model code (`model_utils.py`), tests (`tests/`), config and docs.
**Method:** line-by-line read of every file in those areas, then a verification pass on branch `main-copy` (identical to
`origin/main`) that loaded the real model artifacts and ran the pure functions / a predict call against the local DB.
Each item carries a confidence tag: **[Verified]** = reproduced by running code or inspecting the committed artifacts,
**[Certain]** = follows directly from the code, **[Likely]** = depends on a runtime/library detail, **[Verify]** = needs a
quick check before acting. The 190 MB FL training CSVs are gitignored, so training/calibration paths and a clean-clone start
could not be exercised end-to-end.
**Status:** report only — nothing has been fixed.

Line numbers refer to `origin/main` @ `1caee9a`.

---

## Summary

| Severity | Count |
|---|---|
| Critical (wrong clinical output / app can't run / data leakage / false claims) | 10 |
| High (functional bug visible to users) | 18 |
| Medium (incorrect/misleading behaviour, robustness) | 28 |
| Low (polish, dead code, inconsistencies) | 23 |

Top things to fix first:
1. Backend never calls `init_db()` → fresh clone returns 500 on every predict (**C1**).
2. LLM report cache key ignores `patient_id` → one patient can be shown another patient's AI report (**C2**).
3. Clinical-note keyword/risk matching uses substring on TF-IDF tokens → fabricated "key risks" (**C3**).
4. The "multimodal" CV values (GCS/stress) are captured in the browser but never sent to the model (**C4**).
5. `/api/report` trusts a client-supplied `predict_result` (forgeable vitals, prompt injection, no server-side truth) (**C5**).
6. Soft LLM failures overwrite the last good report on screen (**H1**).

---

## CRITICAL

### C1 — Backend never initialises the database; fresh clone crashes `[Certain]`
- `backend/main.py` never calls `init_db()`. Only legacy `app.py:35` does. `db.get_conn()` (`db.py:26`) happily creates an
  empty `icu_monitor.db`, then every query fails with `no such table: sample_patients`.
- `/api/predict` → 500 "Prediction failed: no such table…", `/api/vitals` and `/api/history` → 500. The dashboard sits on the
  loading skeleton forever.
- Even after `init_db()`, `sample_patients` is empty until `migrate_to_db.py` is run manually; `run_prediction` then raises
  `ValueError("No sample data… Run migrate_to_db.py first")` (a 400 to the browser, message shown as "Refresh failed").
- `migrate_to_db.py` also relies on relative paths (`"data/sample_patients"`, `"models/..."`) so it only works when run from
  the repo root.
- **Fix direction:** call `init_db()` in a FastAPI lifespan handler and auto-seed `sample_patients` from `data/sample_patients/*.csv`
  when the table is empty (those CSVs are tiny and committed). Resolve paths from `__file__`.

### C2 — LLM report cache key omits the patient → cross-patient report leakage `[Verified]`
- `llm_service._cache_key` (`llm_service.py:16-21`) = `care_mode|sofa|round(HR)|round(SpO2)`. No `patient_id`.
- Two patients whose SOFA (1 decimal), HR and SpO₂ round to the same values within 90 s get the **same cached report**,
  including the other patient's note-derived text. In a clinical tool this is the worst class of bug.
- The key also ignores RR, Temp, SBP, DBP, MAP, GCS, stress, the clinical note and SHAP output, so a clinically different
  reading (e.g. MAP drops from 80 → 55 with the same HR/SpO₂/SOFA) re-serves a stale report that doesn't mention the hypotension.
- Reproduced: two payloads with different SOFA decimals, HR/SpO₂ and **MAP 80 vs 50** produce the identical key `hospital|6.0|90|96`.
- Cache TTL (90 s) equals the poll interval, so the cache rarely helps legitimately but can still bite.
- **Fix direction:** key on `patient_id` + a hash of the entire prompt (or the full `predict_result`) + `care_mode`.

### C3 — Substring keyword matching on TF-IDF features produces spurious "key risks" `[Verified]`
- `ml_service.py` (`_is_clinical`, ~l.433; RISK_MAP loop, ~l.446-449): `any(k.lower() in f.lower() for k in CLINICAL_KEYWORDS)` and
  `k.lower() in r["feature"].lower()` are plain substring tests. The vectorizer vocabulary is 90 curated terms; I ran the real
  vocabulary through the real tables:
  - `epinephrine`, `norepinephrine`, `thrombocytopenia` contain `"hr"` → **"Abnormal heart rate"**
  - `cirrhosis`, `crrt`, `hemorrhage` contain `"rr"` → **"Respiratory distress"**
  - (all six also pass the "clinical keyword" filter for the same accidental reason)
- When any of these note terms is among the top SHAP features the spurious risk is injected into the LLM prompt under
  "Key Risk Factors (SHAP-identified)" and shown as clinical truth. The scope is narrower than a no-op filter (about 25 of 90 terms pass,
  most legitimately), but the false positives are real.
- **Fix direction:** match whole tokens / exact feature names (`"hr"` should only match the HR vital features, not note words).

### C4 — CV (GCS Eye / Stress) results never reach the prediction `[Certain]`
- `CVMonitorModal` calls `onCVResults({stress, gcs_eye})` every frame; `MonitorDashboard` stores it in `cvLiveData`
  (`MonitorDashboard.jsx:275,415`) and **never reads it**. `/api/predict` takes only `patient_id`+`row_idx`; GCS and stress come
  from the static CSV. The product is presented as multimodal/CV-driven, but the live webcam has zero effect on SOFA, the
  vitals grid, the report or history.
- Consequently the CV modal is purely cosmetic, and every GCS/stress shown in the vitals grid is simulated.
- **Fix direction:** either wire `cvLiveData` into the predict request (and validate server-side), or label the modal as a demo.

### C5 — `/api/report` trusts a client-supplied `predict_result` `[Certain]`
- `backend/api/report.py:13-24` accepts an arbitrary `predict_result: dict`. The server never recomputes it.
  - Anyone can POST fabricated vitals/SOFA and get an authoritative-looking clinical report + reliability score.
  - `clinical_note`, `_vitals_block`, `_risk_block`, `_explanation_block`, `trend_text` are client strings concatenated straight into
    the LLM prompt → **prompt injection** and free use of your Groq key.
  - Missing keys → `KeyError` inside `_build_prompt` → 500 with `str(e)` leaked to the client.
  - `patient_id` in the body is never checked against `predict_result["patient_id"]`, and `patient_id` is not validated to 1-3 here
    (unlike predict/vitals/history).
- No authentication/rate limiting on any endpoint; `/api/report` can burn the Groq quota.
- **Fix direction:** have `/api/report` take `{patient_id}` only and read the latest stored prediction server-side; add auth/rate limit.

### C6 — Reliability score can be silently capped because the judge model id is probably invalid `[Verify]`
- `llm_service.py:538` uses `model="qwen/qwen3.8-27b"`. I can't confirm that id exists on Groq; if it doesn't, the call raises,
  `except Exception: pass` swallows it, `judge_result` is `None`, and `compute_reliability` falls back to `geval = ragas = 0.5`.
  Those two terms carry **35 %** of the weight (0.13 + 0.22), so the badge is mechanically pinned near ~0.8 max and never
  "High" reliably, and the UI/LLM-validation drawer present these defaults as if the judge had scored the report.
- Even when the judge works, failures are invisible (no log, no flag in the response that the judge was skipped).
- **Fix direction:** verify the model id, log the exception, and return `judge_used: false` so the UI can say "judge unavailable".

### C7 — Seed data and the "current" reading are duplicated, and seed timestamps are fabricated `[Certain]`
- `run_prediction` step 2 (`ml_service.py:311-321`): when `patient_vitals` is empty it seeds rows `0..19` from `sample_patients`
  with **hard-coded date `2026-08-29`** and synthetic times, then step 3 appends the current row (usually row 0 again).
  So the first trend window contains the same reading twice, every "trend" chart starts with a hard-coded past date, and
  `HR_std`/`*_mean` features fed to the model are computed over this synthetic history, not real readings.
- Trend features at inference therefore don't match how the model was trained, and change depending on whether it's the first call.

### C8 — Model input distribution mismatch for vitals-window features `[Likely]`
- Training used window statistics (`HR_mean`, `HR_std`, `SpO2_min`…); at inference `HR_std` is `vitals_df["HR"].std()` over a sliding
  window of **up to 20 readings that include seeded/duplicated rows** (C7) and repeats of the same 30 sample rows after wrap-around
  (`cur_idx = row_idx % total_rows`). Predictions are deterministic per row only on the first lap; later laps differ because the
  window contents differ.

### C9 — The DB write on every GET and per-request connections can corrupt/lose data under concurrency `[Likely]`
- `get_prediction_history` runs `ALTER TABLE … ADD COLUMN DBP` (a schema write) on every GET (`db.py:256`), inside a read endpoint,
  and swallows every exception. `append_prediction` does the same.
- `get_conn()` relies on sqlite3's default 5 s timeout (no explicit `busy_timeout`); FastAPI runs sync endpoints in a thread pool, so concurrent
  predict + history + vitals calls can raise `database is locked` (WAL helps but not for the writer/ALTER case).
- `append_patient_vitals` is insert-then-trim in separate statements without a transaction guard across the two calls in
  `run_prediction`; two overlapping predicts for the same patient interleave and leave a window < 20 or with duplicates.

### C10 — Validation drawer shows performance numbers that contradict the committed `training_metadata.json` `[Verified]`
Committed `models/training_metadata.json` vs what the dashboard displays/claims:

| Item | Drawer / defaults say | Metadata file says |
|---|---|---|
| Test set | "12,038 MIMIC-III patients" | `test_samples` 2,559 **rows**; `train_samples` 12,795 |
| MAE | 1.7588 | `final_mae` 1.8968 |
| R² | 0.4724 | `final_r2` 0.5995 |
| High-risk AUROC | 0.9254 | 0.9461 (`high_risk_alert.auroc`) |
| PR-AUC | 0.5703 | 0.6840 |
| Best round | 24 (`load_training_metadata` fallback) | 18 |
| TF-IDF size | "90 SOFA-vocab TF-IDF" (docs/model docstring) | `tfidf_features: 200` (the vectorizer on disk has 90) |
| Confusion matrix | 7210/420/15 · 3170/390/95 · 378/348/12 | [[1304,178,3],[378,444,4],[7,192,49]] |

- The metadata also states `evaluation_limitation`: *row-level train/test split — patient/ICU-stay IDs unavailable; not independent clinical validation.* The drawer calls it a "Held-out test set · 12,038 MIMIC-III patients" with a "benchmark avg 0.82–0.84" comparison, which overstates the evidence (a row-level split means probable patient leakage).
- Actual high-risk (SOFA ≥ 10) **recall is 19.8 %** and the ≥ 8 alert recall is 56.9 % (107 of 248 high-risk cases missed). The UI surfaces neither; it shows only AUROC.
- The numbers probably come from a different/earlier model than the `federated_model.pth` actually deployed, or the metadata was regenerated and the UI never updated. Either way the validation page can't be trusted.

---

## HIGH

### H1 — A soft LLM failure replaces the last good report with an error card `[Certain]`
- `generate_report` returns HTTP **200** with `sections: null, error: "..."` for any LLM failure/empty output
  (`llm_service.py:668-678`). `ReportContainer.jsx:140-141` does `setReportData(r.data); setPrevReportData(r.data)` unconditionally,
  so the previous good assessment is overwritten and the UI shows "AI Assessment Unavailable". The "Showing previous assessment"
  fallback (only wired to HTTP errors) is bypassed. `onReportData(r.data)` also pushes the error payload to the LLM validation drawer
  (reliability = null).

### H2 — Stale report from the previous patient/care-mode is shown while the new one loads `[Certain]`
- `ReportContainer` keeps `prevReportData` forever and displays `reportData ?? prevReportData` (l.176). After switching patient
  (same component instance via route param) or care mode, the old patient's report remains visible under the new patient's header
  until the new call returns, and permanently if it fails. `llmReportData` in `MonitorDashboard` is likewise never cleared.

### H3 — `reportKey` reads `v.GCS`, which doesn't exist `[Certain]`
- `ReportContainer.jsx:19` uses `Math.round(v.GCS ?? 0)`; the API field is `GCS_eye`. GCS never participates in change detection, so a
  GCS drop (e.g. 4 → 1) with unchanged SOFA/HR/SpO₂/SBP won't regenerate the report. Also MAP, DBP, RR, Temp, stress and the note are
  omitted, so many clinically significant changes don't refresh the report (compounded by C2).

### H4 — Rules of Hooks violation in `MonitorDashboard` `[Certain]`
- `MonitorDashboard.jsx:261` returns `<Navigate>` **before** `useQuery`, `useMonitor`, `useState`… (l.263+). If `patientId` changes
  between a valid and an invalid value in the same mounted instance, React throws "Rendered fewer hooks than expected". Also lint
  failure (`react-hooks/rules-of-hooks`).

### H5 — Side effects inside a `setState` updater (countdown) `[Certain]`
- `useMonitor.js:62-66` calls `fetchNext()` (which calls several setStates) from inside `setCountdown(c => …)`. Updaters must be pure;
  React 19 StrictMode double-invokes them (dev) and React warns about updating during render. It only "works" because of
  `fetchingRef`. If the tab is throttled in the background, the 1 s interval drifts and polling is irregular.

### H6 — Race on patient change / overlapping fetches in `useMonitor` `[Likely]`
- `useMonitor.js:47-56` force-resets `fetchingRef.current = false` and starts a new fetch while the previous patient's request may still
  be in flight. When the old one resolves it calls `setPredictData(oldPatientData)` and `rowIdxRef.current += 1`, overwriting the new
  patient's state with the old patient's reading. No `AbortController`, no check that the response `patient_id` matches the current one.

### H7 — `rowIdx` restarts at 0 on every page visit while the DB keeps its state `[Certain]`
- `rowIdxRef` is in-memory and starts at 0 each mount, but vitals/history persist. Each visit replays rows 0,1,2… and appends duplicates
  to `prediction_history` (capped at 50) and the vitals window — history shows the same 30 readings repeating, and the trajectory
  check compares unrelated points. Reloading the page produces a duplicate row 0 reading every time.

### H8 — Trend charts race the first prediction / no error state `[Likely]`
- `TrendsSection` fires `GET /api/vitals/:id` immediately, in parallel with the first `POST /api/predict` that seeds the vitals table.
  On a fresh patient the first response is `[]` → empty charts until the 30 s refetch. No `isError` handling (failure renders empty
  charts silently); `refetchInterval: 30_000` doesn't match the 90 s data cadence (3 of 4 refetches return identical data).

### H9 — `isHighRisk`/glow logic compares against colours the API never returns `[Certain]`
- `HeaderBar.jsx:285` and `SOFAGauge.jsx` (`ArcGauge`, l.41) test `riskColor !== '#0d8a52' && !== '#18c872'` to decide "not high risk".
  The backend's Low-Risk colour is `#28a745` (`risk_label`), so **low-risk patients are always treated as high-risk**: ambient glow ring on
  the gauge, accent styling in the header. (Patient-context strip uses `risk !== 'Low Risk'` which is correct — the three disagree.)

### H10 — Inconsistent risk thresholds across backend, prompts, UI and API `[Certain]`
- Backend `risk_label`: Low <5, Moderate 5–<10, High ≥10; alert at ≥8 (`ALERT_THRESHOLD`).
- Patients API ranges: "0–4", "5–9", "10–24" (consistent with backend), but the **LLM prompt** says "0–6 = low, 7–9 = moderate, ≥10 high,
  ≥15 critical" (`llm_service.py:189`) and the home-care prompt says "above 6 = concerning" (l.278); the LLM then calls a "Moderate"-labelled
  patient low risk or vice-versa. `HistoryDrawer` flags SOFA >9 only. `RISK_STYLES` has a "Critical Risk" the backend never emits. `ALERT_THRESHOLD=8`
  puts "alert" prompts inside the Moderate band while the banner says "Elevated risk".

### H11 — Vital normal ranges disagree between layers `[Certain]`
- Backend `NORMAL_RANGES`: Temp 36.5–37.5, SBP 100–120. Frontend `VitalsGrid`/`TrendsSection`/`HistoryDrawer`: Temp **36.0**–37.5, SBP **90**–120.
  Values in 36.0–36.5 or SBP 90–100 are "low" in the status chip (`vitals_status` from backend) but plotted/coloured as normal in charts and
  history, and the LLM prompt quotes different ranges again ("~120/80"). Each component hard-codes its own table.

### H12 — Trajectory check logic is wrong and disables itself `[Certain]`
- `check_trajectory` (`ml_service.py:220-249`) fetches the last two history rows and compares `hist.iloc[-1]` to `hist.iloc[0]` to detect
  "duplicate" rows: if the two most recent predictions share SOFA **and** `Timestamp` (minute resolution, SOFA rounded to 0.1 — very
  likely for stable patients or a page reload) it returns "First reading" — trajectory validation is skipped precisely for stable data.
  It also always compares to the single previous row, so with the duplicate rows from H7 the "previous" is often the same reading.
- `jump > 4 and sig == 0` ignores Temp/DBP/GCS/stress; threshold 4 points is hard-coded.

### H13 — `sofa_floor` check passes under-prediction by construction `[Certain]`
- `SOFAValidationDrawer.jsx:215`: `floorPass = sofa >= floor - 1`, so a prediction can be up to 1 point below the physiological floor and
  still show "Consistent ✓". Backend `compute_sofa_floor` is computed but **never returned**; the frontend re-implements the same logic
  (duplicate rule set that can drift). The floor score also double-counts nothing for HR/RR/Temp, so it validates only 3 vitals.

### H14 — SOFA Validation "Layer 1" values are hard-coded and never read from the model metadata `[Verified]`
- `SOFAValidationDrawer.jsx` Layer1 hard-codes AUROC `0.9254`, PR-AUC `0.5703`, the per-class MAE/R² and confusion counts, "12,038 patients" and "ceiling ~0.55".
  MAE/R² try `predictData._training_meta`, which the backend **never** sends, so they always fall back to literals (`1.7588`, `0.4724`).
  `training_metadata.json` is never served to the UI. See **C10** for how far these literals are from the committed metadata.

### H15 — Conformal interval fallback masks failures and mixes training data into calibration `[Certain]`
- `_compute_conformal_q_hat` (`ml_service.py:132-156`): bare `except Exception: return 2.9` hides DB/schema errors; q̂ is computed from
  `fl_training` **training** rows, not a held-out calibration set, so the "90% conformal interval" is optimistic and not valid conformal
  prediction. The value is `lru_cache`d for the process lifetime, and `confidence = 1 - q/12` (l.559) is an arbitrary transform labelled
  "AI Confidence" in the UI.
- Because MAE ≈ 1.76 and R² ≈ 0.47, a fixed ±2.9 band on a 0–24 scale is shown on every patient; for SOFA near 0 the lower bound is clamped so the
  interval is asymmetric but still described as ±.

### H16 — Concurrency-unsafe shared model/SHAP explainer `[Likely]`
- `load_artifacts` returns one shared `ICUModel`; `shap.DeepExplainer` (cached via `lru_cache`) registers forward hooks on that model and is
  not thread-safe. FastAPI runs the sync `predict` handler in a thread pool, so two concurrent predicts (two browser tabs, or predict +
  tests) can interleave `model.eval()`/hook state. SHAP failure is swallowed (`except Exception: shap_vals = None`) and the UI then
  quietly hides the whole "Key Risk Drivers" panel.

### H17 — Judge/LLM calls can exceed any reasonable latency and are never time-limited at the HTTP layer `[Certain]`
- `generate_report` = main Groq call (no client timeout, `max_tokens=3500`) + judge thread joined for up to 12 s (the thread keeps running afterwards,
  `daemon=True`, no cancel). Axios in `client.js` has **no timeout**, so a hung Groq request freezes the report section ("~5s" skeleton text is
  inaccurate; real time is 10–30 s+). `finish_reason == "length"` is never checked: truncated reports are accepted, scored, and cached as OK
  (hence the prompt's repeated "never truncate" warnings). gpt-oss reasoning tokens count against `max_tokens`, which is what produces the
  `_FALLBACK_MSG` path.

### H18 — `server.py` (real FL) overwrites the deployed model with the *latest* round, not the best `[Certain]`
`SaveBestStrategy.aggregate_fit` saves every round's aggregate to `models/federated_model.pth` — the file the dashboard loads — while the "best" checkpoint goes to `federated_model_best.pth` (gitignored). After a run the dashboard serves the last-round model, not the best-round one the metadata claims (`best_round`). At the end the script overwrites `training_metadata.json` with `aggregation: "FedAvg"` and drops fields (`base_lr`, `risk_classification_metrics`, `evaluation_limitation`…) that the dashboard/docs expect.

---

## MEDIUM

### M1 — `int()` truncation / NaN handling on sample rows `[Certain]`
`ml_service.py:306-307`: `GCS_eye = int(...)`, `stress = int(current_row["stress_score"])`. Stress is a float score (shown /10); truncation loses
precision (2.9 → 2), and a NULL/NaN in the CSV raises `ValueError: cannot convert float NaN to integer` → 500. Same for any NaN vital
(`float(NaN)` passes through and later breaks JSON).

### M2 — NaN in DB rows breaks JSON serialisation `[Likely]`
`/api/vitals` uses `df.to_dict(orient="records")` (`vitals.py:28`). A NULL vital becomes `NaN`; Starlette's JSONResponse rejects NaN → 500 for the entire
trends panel. `/api/history` works around this with `json.loads(df.to_json())`; `vitals` doesn't.

### M3 — Prediction history trigger omits `DBP` `[Certain]`
`trg_ph_updated_at` (`db.py:158`) lists data columns without `DBP`; editing only DBP doesn't bump `updated_at`. `fl_training` gets no `updated_at`
trigger at all (only `created_at`), contradicting the comment block. The trigger for `created_at` runs an extra UPDATE per insert.

### M4 — `DELETE … NOT IN (SELECT … LIMIT 20)` windowing is per `id` not per time `[Certain]`
Sliding window is by autoincrement id. Seeded rows (C7) and real rows are mixed; chart x-axis labels mix synthetic `08:00:00` with wall-clock
`21:41:07`, so the time axis is non-monotonic/meaningless (`TrendChart.formatTime` strips the date).

### M5 — `Timestamp`/`Reading` columns are text with minute resolution `[Certain]`
`append_prediction` stores `%Y-%m-%d %H:%M` — two predictions in the same minute are indistinguishable (feeds H12). `Reading` is `"3/30"` text.

### M6 — Column-name aliasing with unicode (`SpO₂`, `Temp (°C)`) leaks across layers `[Certain]`
API JSON keys are `"SpO₂"`, `"Temp (°C)"`, `"GCS Eye"` (aliased in SQL), while `/api/vitals` uses `SpO2`/`Temp`. Frontend hard-codes both. `migrate_to_db.py`
has a separate `col_map`. Any rename breaks silently (undefined → "—").

### M7 — `_interpret_shap` text is wrong for several features `[Certain]`
`ml_service.py:253-275`: always says "low oxygen levels" for any `spo2` feature even when SpO₂ is high and reducing risk; "possible infection / fever"
for any Temp feature; "abnormal heart rate" for `HR_std` etc. Direction text ("increasing/reducing risk") is right but the clause before it isn't conditioned on the value.
`"hr" in f` also matches `GCS`-free names like `"latest_HR"` fine, but `"rr" in f` matches `latest_RR` only by luck of the vocabulary; `"map"` matches `MAP_mean` and anything containing "map".

### M8 — SHAP panel/`shap_top5` mixing and staleness `[Certain]`
- `SHAPPanel` keeps the last non-empty list (`displayFactors`), so after a patient switch or SHAP failure it shows the **previous patient's drivers**.
- `ReportContainer` renders SHAP from the *current* `predictData` while the report text on screen may be from an older reading (see H2/H3).
- `rel_impact` is normalised against the top-7 max but only 5 are shown; bars for the displayed top-5 may all be <1.
- `animejs` mutates bar widths in an `IntersectionObserver`; if the panel is already in view when data refreshes, `animated.current` is reset but bars re-mount at width 0 and rely on the observer firing again (it won't re-fire for an already-intersecting element) → bars can stay at 0 after the first refresh. `[Likely]`

### M9 — `SOFAGauge` mutates React-owned text `[Certain]`
`ref.textContent = obj.val.toFixed(1)` (animejs) fights React rendering `{sofa_score?.toFixed(1)}` in the same node; React may overwrite or the node can desync.
`prevScore` persists across patient switches so the animation counts from the previous patient's SOFA. The score initially animates from 0 on every mount.

### M10 — Stress thresholds inconsistent within one card `[Certain]`
`VitalsGrid.jsx` `stressStatus`: abnormal (red/"high") at ≥4, but the label says "Low Stress" ≤3, "Elevated" 4–6, "High" ≥7, and dots colour 1–3 green / 4–6 amber / 7+ red.
So stress 4–6 shows an amber label but red/abnormal card styling. Backend LLM rule uses `stress > 7`; `HistoryDrawer` flags `> 3`; CV modal uses 3.5/6.5 cut-offs. GCS 3 is "normal"
in the grid but history flags `<3`.

### M11 — `HistoryFAB` badge is hard-wired to 0 `[Certain]`
`MonitorDashboard.jsx:398` passes `recordCount={0}`; the counter UI in `HistoryFAB` can never display.

### M12 — History drawer data freshness `[Likely]`
`HistoryDrawer` query key `['history', patientId]`, `staleTime: 30_000`, never invalidated after a new predict. Opening the drawer shows up to 30 s-old data; count header "recordings"
may lag the latest reading. Rows keyed by array index (`key={idx}`) with `reverse()` → animations replay oddly on refetch. Empty-state text says "first 90-second poll" (cadence hard-coded in copy).

### M13 — `RefreshErrorBanner` and several styles use invalid CSS `[Certain]`
Colour tokens are CSS variables (`var(--x)`) but code appends hex alpha: `'1px solid var(--warn)50'` (`MonitorDashboard.jsx:247`), `${ac}50` (SectionHeading box-shadow),
`${color}1e`/`${color}48`/`${accentColor}30`/`${accentColor}08` in `ReportSection.jsx` where `color` is `var(--blue-l)` etc. (`ReportContainer.SECTIONS`).
Browsers drop the whole declaration → report card borders, glows, ⚡ highlight backgrounds and the error banner border silently don't render. (Works only for hex colours like `#9b59b6`.)

### M14 — `ReportSection` subheading detection is fragile `[Certain]`
`renderLine`: any line ending in `:` under 60 chars and not starting with `•` becomes a heading; a bullet-less sentence ending with a colon, or "Home Remedies:" typed inline, is styled as a heading.
`renderBold` splits on `**…**` only (no nested/escaped); stray `*` list markers from the model are stripped by `^[•\-\*]` (a leading `- 5%` would lose its dash). Content is rendered as React text (safe against XSS) — but `text.includes('⚡')` styling is applied to any line.

### M15 — `_parse_llm_sections` regex edge cases `[Likely]`
`llm_service.py:97-115`: finds only the **first** line-start match per heading (verified: bullets like "• RISK FORECAST:" are correctly ignored, but any other line that begins with a heading phrase is not); anything before the first heading is dropped; such a line truncates the preceding section. Missing sections become `""` and the UI shows "No assessment available." without flagging that parsing failed (reliability is computed on raw text, not parsed sections).

### M16 — `response_structure_score` & other checks are keyword heuristics presented as validation `[Certain]`
- `factual_grounding_score`, `llm_clinical_rules_score`, `severity_calibration_score` pass if the text merely *contains* "oxygen"/"monitor"/"urgent" anywhere (including negations "no oxygen needed", or the model echoing the prompt).
- `severity_calibration_score` returns 1.0 for every 5 ≤ SOFA < 10 regardless of content.
- `_CLINICAL_REFS` whitelist in `numeric_hallucination_score` contains values (1–6, 8, 10, 12, 15…) so most invented small numbers (doses, timeframes) pass; `_is_allowed` tolerance `max(10 %, 3.0)` is huge for SpO₂/Temp (36 ± 3.6 …).
- `contraindication_score`: "NSAID" rule fires on any ICU note that *warns against* NSAIDs (substring `nsaid`); `morphine/opioid` rule penalises correct analgesia advice in hypoxic patients; benzodiazepine+hypoxia flagged even when the report says "avoid".
- Hard rule "SpO₂ < 90 → must mention oxygen" is checked against `vitals["SpO2"]`, but home-care mode forbids IV/hospital equipment; the same hospital-oriented keyword lists are used for both modes.
- `compute_reliability` mixes 9 methods with hand-picked weights that sum to 1.00 (0.13+0.13+0.09+0.09+0.08+0.05+0.08+0.13+0.22) — fine arithmetically but the 0.05 weight on **contraindication** means a dangerous recommendation barely moves the score unless the judge flags it; `caution` fires only <0.60.

### M17 — `_strip_thinking` edge cases `[Certain]`
Handles `<think>…</think>` and an unclosed `<think>` (returns everything *before* it → often `""`), but a stray `</think>` without an opening tag is left in the output, and reasoning text before `</think>` is shown to clinicians in that case.

### M18 — Error handling leaks internals `[Certain]`
All routes return `f"... failed: {str(e)}"` in 500 details (stack/driver/DB messages, Groq error bodies). `generate_report` puts `f"⚠️ LLM unavailable: {e}"` into the **response body** that the UI renders. `history.py` has dead `import json/os/sys` hacks (`sys.path.insert`) in every router.

### M19 — Polling/“LIVE” indicator claims don't match reality `[Certain]`
`HeaderBar` shows a red "LIVE" chip + countdown; data is a replay of 30 CSV rows with synthetic timestamps. After an error, `connectionLost = Boolean(error) && !isRefreshing` shows
"Connection lost" even for non-connection errors (400 "No sample data", 500 model error). Error text always says "Retrying in 90s" even if the interval is blocked.

### M20 — Countdown semantics `[Certain]`
`useMonitor` clamps to 1 and waits for the fetch to reset to 90; if the fetch is slow (SHAP+DB can take seconds; LLM isn't involved) the UI shows "1 s" for the entire duration.
The next fetch starts 90 s after the **last tick**, not the last completion, so polling period = 90 s + latency. Doc/test comments say "30-second refresh" (`test_integration.py`, `_FALLBACK_MSG`).

### M21 — Query/cache hygiene in React Query `[Likely]`
`getPatient` / `getPatients` have default staleTime 10 s and `retry: 1`; a 400/404 is retried. `queryKey: ['patient', id]` with NaN id never occurs (guarded) but `HistoryDrawer patientId={id ? Number(id) : null}`
is redundant. No global error boundary: any render exception (e.g. `predictData.vitals` shape change) white-screens the app.

### M22 — CV monitor reliability/perf issues `[Certain]`
- ~10 `setState` calls per animation frame (≈60/s) → whole modal re-renders every frame; `onCVResults` fires every frame.
- Model file loaded from `storage.googleapis.com` on every first use (not bundled; offline = error), `delegate: 'GPU'` with no CPU fallback (fails on machines without WebGL2).
- `detectForVideo(video, Date.now())` – timestamps must be monotonically increasing; `Date.now()` can repeat within the same ms on fast frames → MediaPipe throws.
- `requestAnimationFrame` loop keeps running when the tab is hidden; on tab-hide the EAR/GCS windows get huge gaps (window is wall-clock 30 s).
- `gcsFromWindow` returns score 4 ("Spontaneous") when there's no history, so with no face detected nothing alerts; `GCS "To Pain"` is inferred from "eyes closed with flicker" — not a valid GCS assessment (a webcam can't elicit pain/voice response). The UI labels these as GCS scores.
- Calibration assumes the user is looking at the camera with eyes open for the first 90 frames; if not, the baseline is rejected silently and defaults (0.22/0.12) are used without telling the user.
- Escape handler effect depends on inline `onClose` → re-subscribes each render.
- Camera permission/streams: on `handleRetry` the effect re-runs but the previous `flRef` close happens in cleanup, fine; however `status` stays `'idle'` until the effect runs so the status panel flashes empty.

### M23 — `.gitignore` vs tracked files `[Certain]`
`models/icu_monitor.db-shm` and `-wal` are **tracked** in git (see `git ls-files models`) even though `.gitignore` lists them; they're binary runtime files that will conflict/corrupt on every pull and can contain
MIMIC-derived data. `federated_model.pth`, `scaler.pkl`, `tfidf_vectorizer.pkl` (joblib/pickle) are committed — fine for a prototype but loading unpickled artifacts from a repo is a code-execution risk if tampered.

### M24 — Missing/incorrect dependencies and docs `[Certain]`
- `requirements.txt` lacks `pytest` and `httpx` (required by `fastapi.testclient`), so `pytest` fails on a clean env.
- The backend requirement set drags in `flwr[simulation]`, `opencv-python>=5.0.0` (verify this version exists), `mediapipe`, `google-cloud-bigquery*` even though the dashboard backend uses none of them → huge, slow install and a larger attack surface. `python-dotenv` is only loaded in `main.py`.
- README says the dashboard opens at **:5174** but `vite.config.js` doesn't set a port (default 5173; 5174 only if 5173 is taken). README troubleshooting uses `lsof` (not available on Windows, the primary dev OS here).
- Vite proxy `'/api' → localhost:8000` is hard-coded; no `.env` for the API URL; CORS allow-list omits `127.0.0.1:5175`, and production (served build) has no config at all. `frontend/README.md` / `oxlint` have no config checked in (`.oxlintrc.json` is gitignored).

### M25 — `get_trend` is unit-agnostic `[Verified]`
`model_utils.get_trend` labels any slope beyond ±0.1 per reading as increasing/decreasing. Temperature climbing 36.5 → 37.4 °C over 10 readings is reported **"stable"**, whereas HR creeping 80 → 80.9 bpm is **"increasing"**. These strings go into the LLM prompt as "Vital Sign Trends".

### M26 — Pickled preprocessors were built with a different scikit-learn than documented `[Verified]`
Loading `tfidf_vectorizer.pkl` raises `InconsistentVersionWarning: … from version 1.9.0 when using version 1.7.1`, but `requirements.txt` says they were fitted with 1.6.1 and pins `<2.0.0`. Unpickling across sklearn versions is explicitly unsupported and can silently change TF-IDF output; the documented pin is wrong.

### M27 — Windows console/encoding pitfalls `[Verified]`
`open()` is called without `encoding="utf-8"` in `server.py:134,145`, `train_federated.py:1024`, `ml_service.py:164` (and legacy `app.py:370`, `fl_dashboard.py:914`). This is currently harmless for `training_metadata.json` because `json.dump` escapes non-ASCII (`→`, `η`, `μ` are stored as `\uXXXX` escapes; verified), but any literal non-ASCII text added later would break on Windows cp1252. The reproduced problem is console output: printing predict results containing emoji (`risk_icon` 🟢🟡🔴), or the scripts' `→`/`✓`/`σ` banners, raises `UnicodeEncodeError` unless `PYTHONIOENCODING=utf-8` is set.

### M28 — Running the API mutates tracked repository files `[Verified]`
`models/icu_monitor.db-shm` / `-wal` are tracked in git. Simply exercising the backend checkpointed SQLite and **deleted both files from the working tree** (`git status` showed ` D`). They'll keep appearing as modifications/deletions and can be committed by accident (see M23).

---

## LOW

### L1 — Patient API has no age/gender, header code expects them
`HeaderBar.PatientPill` renders `patient.age`/`gender` conditionally; backend `PATIENTS` never provides them, so that branch is dead. Avatar initial is `patient.name[0]` = "P" for all three patients.

### L2 — Patients are hard-coded in two places
`PATIENTS` dict (`patients.py`) vs `_VALID_PATIENTS = {1,2,3}` in history/vitals and `(1, 2, 3)` in predict and `[1,2,3]` in `MonitorDashboard`. Adding a patient requires 5 edits. Also DB patient ↔ CSV mapping hard-coded in `migrate_to_db.py`.

### L3 — `limit` validation inconsistent
`/api/vitals` caps at 20 (`vitals.py:22`), `/api/history` at 50; `limit` bad values return 400 instead of FastAPI's standard 422 (tests encode this). `row_idx` is unbounded (negative/huge accepted).

### L4 — `get_history` default limit = 5 vs frontend 50; client default mismatch
`api/client.js` `getHistory(patientId, limit = 5)` while the drawer asks 50; harmless but unused default.

### L5 — Dead/legacy code left in the dashboard repo
`app.py` (2,999 lines, Streamlit) and `fl_dashboard.py` (1,736 lines) still contain a full second copy of the prediction/validation/LLM logic that has already diverged from `backend/services/*`
(duplicated `compute_sofa_floor`, `vital_consistency_flags`, `_interpret_shap`, LLM prompts). Future fixes will land in only one copy. `docs/` (≈8,000 lines) still describes the Streamlit architecture.

### L6 — `CLINICAL_KEYWORDS`/`RISK_MAP` duplication and ordering
`RISK_MAP` contains both `"SpO2"` and `"RR"` keys matched case-insensitively against *every* feature; dict iteration order decides which message appears first.

### L7 — Misleading UI copy
"Generating AI assessment — this may take ~5s…" (actual 10–30 s), "Static between readings" on the clinical note, "AI Confidence" (derived from conformal width, not a probability),
"90% prediction interval" (see H15), "nursing · physician · discharge summary" label though notes are synthetic one-liners, "Validated across all quality checks" for any High badge.

### L8 — Hospital/Home-care toggle triggers an extra LLM call + judge call every flip
No debounce; rapid toggling spawns multiple in-flight requests (`reqIdRef` ignores stale *results* but the server still performs and bills each call). Each toggle also re-runs the judge.

### L9 — `ReportToggle`/FABs accessibility
Buttons lack `aria-pressed`; drawers/modals lack `role="dialog"`, `aria-modal`, focus trap and focus restore; FAB stack at fixed `bottom: 28/88/148/208` can overlap the footer text and collides with the drawer close buttons on small screens;
four floating buttons on mobile cover content. Hover-only labels (`onHoverStart`) are not keyboard-accessible.

### L10 — Colour-only status encoding and dark-only headers
Status relies on colour/arrow; report cards, charts and reliability badge headers are hard-coded dark gradients (`#07111e`) inside the light theme, so text-contrast tokens (`var(--t1)`) are never used there; some `rgba(255,255,255,0.35–0.44)` text on dark has ratio < 4.5:1.

### L11 — Font loading
`index.html` loads four Google font families (Figtree, Lexend, Inter, JetBrains Mono) from the network on every visit — blocks first paint offline/hospital intranet; no `font-display` fallback; MediaPipe model also remote (M22).

### L12 — `AOS.refresh()` after `setTimeout` without cleanup
`MonitorDashboard.jsx:278-282` schedules a timeout per prediction with no `clearTimeout` on unmount/re-run.

### L13 — `useClock` / SelectPatient re-render every second
The whole `TopNav` re-renders each second; harmless but `SelectPatient` ECG/hero animations don't pause on hidden tab. `SelectPatient` footer says "MIMIC-III dataset" though patient data are hand-written CSVs.

### L14 — Test-suite defects
- `tests/` run against the **real** DB (no isolation/fixtures). `test_cross_patient_history_isolation`, `test_vitals_accumulate…` mutate and read live state, so they're order-dependent and pollute the dev database (and fail on a fresh clone due to C1).
- `test_predict_row_wraps_large_index` contains a no-op assertion (`body["status_code"] if "status_code" in body else True`).
- `test_vitals_accumulate_after_predict_calls` asserts only `after >= 1` (can't fail).
- No tests for: the frontend, LLM validators (pure functions, easy to unit-test), `_parse_llm_sections`, `_cache_key` collisions (C2), `check_trajectory`, the 429/401 error mapping, or `/api/report` happy path without a real Groq key (tests tagged `slow` hit the live API).
- `conftest.py` imports `backend.main` at collection, so a missing `GROQ_API_KEY` or model file breaks *all* tests rather than skipping.

### L15 — `lru_cache` on loaders never invalidates
`load_artifacts`, `_load_shap_explainer`, `_compute_conformal_q_hat`, `load_training_metadata` are cached for the process lifetime; replacing `federated_model.pth` needs a restart; uvicorn `--reload` only watches `.py`. `load_training_metadata` is defined but unused by the API.

### L16 — `ml_service` module side effects
Imports torch/shap/joblib and loads nothing until first request, so the first `/api/predict` after start takes many seconds (model + SHAP explainer init) while the UI shows the skeleton; no startup warm-up or readiness endpoint (`/health` returns ok before the model is loaded).

### L17 — `weights_only=True` torch load vs older checkpoints
`torch.load(..., weights_only=True)` will fail on checkpoints saved with extra objects; `requirements.txt` allows `torch>=2.0.0` (the flag/behaviour differs before 2.6).

### L18 — Unused imports and dead values
`llm_service.py` imports `numpy`, `Decimal`; `report.py` imports `os` only for the key; `predict.py` result passes through `_vitals_block` etc. to the browser (private `_…` fields exposed publicly and echoed back in `/api/report` — leaks prompt scaffolding and doubles the payload).

### L19 — Naming/units
`GCS_eye` vs `GCS Eye` vs `GCS_Eye` vs `GCS_eye_opening` vs `gcs_eye` (CV) across DB/API/frontend; `Temp` vs `Temp (°C)`; Stress is float in CSV, int in API (`stress`), `Stress` in history. Time columns: `time`, `Timestamp`, `timestamp`.

### L20 — Date/time handling
`datetime.now()` naive local time is stored without timezone; hard-coded `2026-08-29` seed date; `formatTime` assumes "YYYY-MM-DD HH:MM:SS" strings. History shows `Timestamp` as local-without-seconds.

### L21 — Security hygiene
No auth, rate-limiting or request-size limits; CORS fixed to localhost (fine for dev); `GROQ_API_KEY` read per request from env (fine) but `load_dotenv()` runs *after* routers import `llm_service` (no issue today, but fragile); patient data/PHI-like clinical notes are sent to a third-party LLM with no consent/redaction step; `/docs` Swagger is enabled in all environments.

### L22 — Repo hygiene
`PR_DESCRIPTION.md`, `reports/`, `models/_backup/` in the user's branch are untracked; `models/patient_vitals/.gitkeep` and `prediction_history/.gitkeep` exist for CSV paths the app no longer uses.

### L23 — Reproduction note: side effects of exercising the API
Every `POST /api/predict` writes to `patient_vitals` and `prediction_history` (and `/api/history` can `ALTER TABLE`). There is no dry-run/read-only mode, so smoke tests and the pytest suite permanently alter the developer's DB.

---

## Suggested fix order
1. **C1** (init + auto-seed) → app works on a clean clone.
2. **C2, C5, C10, H1–H3** (report correctness and cache identity) → report always matches the patient shown.
3. **C3, C7, H7, H12** (note-keyword matching, seeding/replay/trajectory) → trustworthy model inputs and validation.
4. **C4** (either wire CV values or relabel), **C6** (verify judge model id, expose `judge_used`).
5. **H4–H6, H9–H11** (hook order, polling race, thresholds/colours in one shared constants module served from the API).
6. Medium/Low items as polish, plus remove legacy `app.py`/`fl_dashboard.py` or mark clearly as deprecated.

*Static review plus a partial runtime verification (model artifacts, pure functions, one predict call). A full browser/runtime pass with a seeded DB is still recommended to confirm the **[Likely]**/**[Verify]** items before fixing.*

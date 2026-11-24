# Project Context — Complete Implementation Guide
## ClinIQ: ICU Clinical Decision Support System with Federated Learning

**Last updated:** 2026-11-24

> This document explains the whole project from start to finish: what was built, how it was built, why each design decision was made, what was tried and failed, and what the final approach is. A reader who has never seen the project should understand all of it. Every number for the current model comes from `models/training_metadata.json` (the final training run) and from the current code.

---

## How to Read This Document

- **Sections 1–3** say what the project is and what it does.
- **Sections 4–9** cover the data, the 108 inputs and the SOFA target.
- **Sections 10–12** cover the model, federated learning and the history of training improvements.
- **Sections 13–16** cover explanations (SHAP), the AI report, how that report is checked, and the alert.
- **Sections 17–19** cover how the whole app works, every file, and every setting.
- **Sections 20–22** cover results, what did not work, and privacy.
- **Section 23** is the camera monitor, **24** the limitations, **25** the glossary.

Parts marked **historical** describe earlier versions. They are kept because understanding what failed matters as much as understanding what works.

Other documents go deeper on single topics:

| Topic | Document |
|---|---|
| Building the dataset in BigQuery | `docs/SQL_COMMANDS.md` |
| Training script and every setting | `docs/TRAIN_FEDERATED.md` |
| Checking the SOFA score and the AI report | `docs/VALIDATION.md` |
| Model inputs and outputs | `docs/INPUT_OUTPUT.md` |
| Why the model looks the way it does | `docs/MODEL_DISCUSSION.md` |
| Camera monitor | `docs/CV_MONITOR.md` |
| Short whole-project reference and viva questions | `docs/SKILL.md` |

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Problem Being Solved](#2-problem-being-solved)
3. [What the System Does — End to End](#3-what-the-system-does--end-to-end)
4. [Dataset — MIMIC-III and BigQuery](#4-dataset--mimic-iii-and-bigquery)
5. [Preprocessing — Building `dataset_v2`](#5-preprocessing--building-dataset_v2)
6. [Feature Engineering — All 108 Features](#6-feature-engineering--all-108-features)
7. [Sliding Window Mechanism](#7-sliding-window-mechanism)
8. [Clinical NLP — TF-IDF](#8-clinical-nlp--tf-idf)
9. [Target Variable — SOFA Score](#9-target-variable--sofa-score)
10. [Machine Learning Model — Evolution and Final Choice](#10-machine-learning-model--evolution-and-final-choice)
11. [Federated Learning — Architecture and Evolution](#11-federated-learning--architecture-and-evolution)
12. [Training Improvements — What We Tried and Why](#12-training-improvements--what-we-tried-and-why)
13. [SHAP Explainability](#13-shap-explainability)
14. [LLM Integration — Groq](#14-llm-integration--groq)
15. [Checking the AI Report — The Reliability Score](#15-checking-the-ai-report--the-reliability-score)
16. [Alert System and Threshold Decision](#16-alert-system-and-threshold-decision)
17. [Complete App Pipeline — Step by Step](#17-complete-app-pipeline--step-by-step)
18. [All Files and Their Roles](#18-all-files-and-their-roles)
19. [Configuration Reference](#19-configuration-reference)
20. [Model Performance Summary](#20-model-performance-summary)
21. [What Did Not Work and Why](#21-what-did-not-work-and-why)
22. [Privacy — Differential Privacy Layer](#22-privacy--differential-privacy-layer)
23. [Camera Monitor — Stress and Eye-Opening Scores](#23-camera-monitor--stress-and-eye-opening-scores)
24. [Limitations](#24-limitations)
25. [Glossary](#25-glossary)
- [Appendix — Key Numbers at a Glance](#appendix--key-numbers-at-a-glance)

---

## 1. Project Identity

**Full title:** Multimodal Intelligence System with Federated Learning for Continuous Patient Monitoring and Early Deterioration Detection Leveraging IoMT, Computer Vision and Clinical NLP

**Product name:** **ClinIQ** (the dashboard and its API)

**Type:** Final-year engineering capstone project in Computer Science

**Scope:** An AI-powered ICU **clinical decision support system (CDSS)**

**What it is NOT:** It does not diagnose any disease, does not prescribe treatment and does not make decisions on its own. It surfaces AI-generated insights that clinicians review and act on with their own judgment. All results are retrospective MIMIC-III results, not clinical validation.

---

## 2. Problem Being Solved

ICU patients can deteriorate rapidly. Traditional monitoring fails in three specific ways.

**Problem 1 — Alarm fatigue.** Current ICU systems sound an alarm when a single vital crosses a fixed line (heart rate above 120). This produces hundreds of false alarms a day. Staff begin to ignore alarms and real emergencies get missed. This is a documented patient-safety problem.

**Problem 2 — No combining of signals.** Each vital is judged on its own. A heart rate of 105 means little without knowing what SpO₂ is doing, what the blood-pressure trend looks like and what the doctor's notes say. Current systems cannot join these into one picture.

**Problem 3 — Patient data cannot be shared.** Normal machine learning needs all data in one place. Patient data is legally protected (HIPAA, GDPR), so a hospital cannot send records to another institution or to a central server. That makes shared AI training impossible with conventional methods.

**How ClinIQ answers all three:**
- One predicted **SOFA** organ-failure score, built from vitals, trends, notes and camera cues, replaces single-vital alarms.
- Trend analysis, SHAP explanations and an AI report give a complete, explainable picture.
- **Federated learning** lets hospitals train a shared model without sharing patient data.

---

## 3. What the System Does — End to End

```
INPUTS (read automatically every 90 seconds — no manual entry)
  ├─ Clinical note for the patient (from the sample_patients table)
  ├─ 7 vital signs: HR, RR, SpO₂, Temperature, SBP, DBP, MAP
  ├─ GCS Eye Opening (1–4): recorded value, or LIVE from the webcam
  └─ Stress Score (0–10): recorded value, or LIVE from the webcam

PREPROCESSING (FastAPI backend — ml_service.py)
  ├─ Read the next recorded row for the patient (30 rows replayed in a loop)
  ├─ Append the vitals to the rolling 20-reading window in SQLite
  ├─ Compute 9 trend features (mean, std, min over the last 20 readings)
  ├─ Take the 7 latest vital values
  ├─ Turn the note into 90 TF-IDF features (SOFA word list)
  └─ Combine into a 108-number vector → StandardScaler → clip to ±10

PREDICTION (ml_service.py)
  └─ Federated PyTorch network (108→256→128→64→1, dropout in training only)
     → SOFA score (0–24)
     → Risk level: Low (<5) / Moderate (5–9) / High (≥10)
     → Alert if predicted SOFA ≥ 7 (ALERT_THRESHOLD = 7.0)
     → 90% conformal range: prediction ± 3.28
     → 3 runtime safety checks (SOFA floor, vital combinations, sudden jumps)
     → SHAP DeepExplainer → top 5 clinical drivers

AI CLINICAL REPORT (llm_service.py)
  ├─ Groq model openai/gpt-oss-120b writes a 4-section report
  │    (Hospital mode: clinical;  Home Care mode: plain language)
  ├─ If SOFA ≥ 7 the report puts IMMEDIATE ACTIONS first
  ├─ 9-method reliability check (7 rule checks + 2 AI-judge checks) → score 0–1
  ├─ 90-second cache, keyed by patient, mode, SOFA and every vital
  └─ Fires only when a clinical value changes (reportKey in the frontend)

DISPLAY (ClinIQ React dashboard — one scrolling page)
  ├─ Header, alert banner (Moderate / High risk)
  ├─ SOFA gauge (with conformal range) + vital cards (7 vitals, GCS eye, stress)
  ├─ 7 vital-trend charts (last 20 readings)
  ├─ AI report: Hospital / Home Care toggle → 4 section cards → SHAP bars
  │    → reliability badge → disclaimer
  └─ Floating buttons: History · Camera monitor · LLM validation · SOFA validation
```

---

## 4. Dataset — MIMIC-III and BigQuery

**Dataset:** MIMIC-III (Medical Information Mart for Intensive Care III), a public, de-identified database of real ICU stays at Beth Israel Deaconess Medical Center, Boston. Access requires PhysioNet credentials, so the data is never published on GitHub.

**Why MIMIC-III and not MIMIC-IV:** MIMIC-III is more stable for preprocessing and has well-documented derived tables, including pre-computed hourly SOFA scores.

**Access method:** Google BigQuery.
- **Project:** `mimic-project-2`
- **Dataset:** `dataset_v2` (rebuilt from scratch; every command is in `docs/SQL_COMMANDS.md`; the old dataset `Dataset` was deleted)
- **Tables:** `cohort`, `vitals_clean`, `vitals_hourly`, `vital_trends`, `latest_vitals`, `gcs_hourly`, `stress_hourly`, `labels`
- **Views training reads:** `ml_dataset_final` (5,089,857 hourly rows) and `clinical_notes` (35,799 admissions)

**Cohort:** adults (age ≥ 16) with an ICU stay of at least 6 hours: **52,248 ICU stays from 38,304 patients**.

**Main source tables:**

| Table | Purpose |
|---|---|
| `mimiciii_derived.icustay_detail` | Which stays are in the cohort (age, ICU in/out times) |
| `mimiciii_derived.pivoted_vital` | Heart rate, breathing rate, SpO₂, temperature, blood pressures (both charting systems) |
| `mimiciii_derived.pivoted_gcs` | Glasgow Coma Scale eye opening |
| `mimiciii_clinical.chartevents` | Nurse-recorded pain score (used as the stress score) |
| `mimiciii_derived.pivoted_sofa` | Hourly SOFA; `SOFA_24hours` at the matching hour is the target label |
| `mimiciii_notes.noteevents` | Clinical notes |

**Dataset size, step by step:**

| Step | Result |
|---|---|
| Phase 0 of `train_federated.py` keeps the newest row of each ICU stay | 52,248 rows, each with 108 inputs and one SOFA answer |
| Train / test split, 80/20, stratified by High Risk (SOFA ≥ 10) | **41,798 train / 10,450 test** |
| Test set | 602 High, 2,821 Moderate, 7,027 Low |
| Three simulated hospitals (IID, stratified) | 13,932 / 13,933 / 13,933 rows |

*(Historical: the earlier `Dataset` build used 10-minute windows and a stay-level SOFA label, and had 60,188 rows. Its numbers are no longer used.)*

---

## 5. Preprocessing — Building `dataset_v2`

The data was rebuilt in BigQuery with numbered commands. **Every command, the reason for each part and the test results are in `docs/SQL_COMMANDS.md`.** In short:

| Stage | Table | What it does, and what it fixed |
|---|---|---|
| 1 | `cohort` | Adults only, ICU stay ≥ 6 h. (The first build included about 8,000 newborns.) |
| 2 | `vitals_clean` | Uses both ICU charting systems (CareVue and MetaVision); removes impossible values and readings outside the stay |
| 3 | `vitals_hourly` | One row per stay per hour, lined up with the hourly SOFA timeline |
| 4 | `vital_trends` | The 9 trend columns over the previous 6 hours of time. (The first build used "the last 20 rows", which could span days.) |
| 5 | `latest_vitals` | The 7 latest-value columns, carried forward for a limited time instead of being empty |
| 6 | `gcs_hourly` | GCS eye opening from both systems. (The first build used one item and was 83% empty.) |
| 7 | `stress_hourly` | Nurse-recorded pain score from both systems, converted to 0–10 |
| 8 | `labels` | **SOFA for the same hour** (`pivoted_sofa.SOFA_24hours`). The first build copied a single first-day SOFA onto every window of a stay. |
| 9 | `ml_dataset_final` | Joins everything into the same 23 columns as before |
| 10 | `clinical_notes` | Bedside notes only (Nursing, Nursing/other, Physician, Respiratory), written during the ICU stay, error notes removed |

**Rule that was kept:** only *how* the data is built changed. The names and meaning of the 108 inputs stayed identical, so no model, backend or dashboard code had to change.

**Historical — the original 12 queries.** The first build ran 12 queries: raw vitals from chart events (Fahrenheit temperature converted to Celsius), pivot to one row per timestamp, 10-minute windows, 20-row trend features, latest values, notes (Nursing, Physician, Nursing/Other, Discharge summary), SOFA, joins, GCS eye (item 220739), stress score (item 223791, renamed from "pain score") and a final join into `ml_dataset_final`. It is superseded by `dataset_v2`.

---

## 6. Feature Engineering — All 108 Features

The 108 inputs fall into 4 groups.

### Group 1 — Trend features (9)

Computed from the sliding window of the last 20 readings. They capture the patient's trajectory, not just the current state.

| Feature | Description | Why it is there |
|---|---|---|
| HR_mean | Average heart rate | Sustained elevation |
| HR_std | Spread of heart rate | Instability, rhythm problems |
| RR_mean | Average breathing rate | Sustained respiratory distress |
| SpO2_mean | Average oxygen saturation | Sustained low oxygen |
| SpO2_min | Lowest SpO₂ seen | Catches even brief dangerous dips |
| Temp_mean | Average temperature | Fever or low temperature |
| SBP_mean | Average systolic BP | Sustained circulatory compromise |
| DBP_mean | Average diastolic BP | Diastolic pressure |
| MAP_mean | Average mean arterial pressure | The most important BP measure |

**Why trends and not just latest values:** one reading of HR 105 says little. If HR went 82 → 85 → 90 → 98 → 105, the patient is clearly getting worse. Without trends, early deterioration is invisible.

Slope features (HR_slope and so on) were deliberately left out: they were not in the processed dataset, and adding them would have meant reprocessing all of MIMIC-III.

### Group 2 — Latest vital values (7)

latest_HR, latest_RR, latest_SpO2, latest_Temp, latest_SBP, latest_DBP, latest_MAP. These are the patient's current state. Trends plus latest values give the model both history and the present.

### Group 3 — Camera / observation features (2)

| Feature | Range | Meaning |
|---|---|---|
| GCS_eye_opening | 1–4 | Eye response of the Glasgow Coma Scale. 1 = no response, 4 = spontaneous. Lower is worse. |
| stress_score | 0–10 | Level of visible distress or pain. Higher is worse. |

In the training data these come from MIMIC-III charting (nurse-recorded). In the dashboard they can come **live from the webcam** (Section 23), and then replace the recorded values in the model input.

### Group 4 — TF-IDF SOFA vocabulary features (90)

Clinical notes become 90 numbers using a **fixed list of 90 words**, each tied to one of the six SOFA organs. This replaced an automatic 600-word choice that picked paperwork words ("tablet", "sig", "refills") that measured how much was written, not how ill the patient was.

The list covers: respiratory (intubated, ventilator, bipap, hypoxia…), coagulation (plt, platelets, inr, coagulopathy…), hepatic (bilirubin, totbili, jaundice…), cardiovascular (vasopressor, levophed, septic shock, hypotension…), brain (sedated, coma, encephalopathy, gcs…), kidney (creatinine, dialysis, oliguria, aki…), infection (sepsis, bacteremia, cultures…), lab words (lactate, wbc, hco, angap…) and severity words (failure, ards, pneumonia…).

**Total: 9 + 7 + 2 + 90 = 108 features.**

### What the training data looks like for the two camera inputs
In `dataset_v2`, GCS eye opening is filled for about 96% of stays (84% are 4, "eyes open on their own"). Stress comes from nurse pain scores, and 71% of stays have a stress of 0. So these two inputs carry less weight than the vitals, and live camera readings change the predicted SOFA only a little. (Earlier documents described them as "mostly empty"; that was the old dataset.)

### Feature scaling
Missing values are filled with 0, then all 108 inputs are rescaled with a `StandardScaler` fitted on the training set (average 0, spread 1) and clipped to −10…+10. The scaler is saved as `models/scaler.pkl`.

**Important:** the rows stored in the `fl_training` table (the three hospitals' data) are **already scaled**. Do not scale them again.

### Feature order
The exact order of the 108 inputs matters. `models/feature_columns.pkl` stores it (identical to `scaler.feature_names_in_`). At the bedside the input is always reordered to this sequence before scaling and prediction.

---

## 7. Sliding Window Mechanism

**Why a window:** deterioration is a process, not a single event. A patient developing sepsis may show heart rate climbing for two hours before becoming critical. A snapshot misses that; a window shows it.

**Window size:** 20 readings.

**How it works in the live app:**
1. Each prediction appends the new reading's 7 vitals to the `patient_vitals` table in SQLite.
2. Only the newest 20 rows per patient are kept.
3. All 9 trend features are recomputed from those 20 rows.

**Starting up:** if a patient has no stored vitals yet, the app seeds the window with earlier readings from the sample data, so trends are meaningful from the first prediction. (The old design used a CSV file and a "Reset Patient History" button in a Streamlit sidebar; both are gone. The database replaced the CSV.)

**Trend direction** (shown on the charts and sent to the AI report): the slope of a straight line through the window (`numpy.polyfit`). Above +0.1 = "increasing", below −0.1 = "decreasing", otherwise "stable". The 0.1 cut-off filters out noise. NaN values are ignored.

**Normal ranges:**

| Vital | Low | High |
|---|---|---|
| Heart rate | 60 | 100 |
| Respiratory rate | 12 | 20 |
| SpO₂ | 95 | 100 |
| Temperature (°C) | 36.0 on the dashboard (36.5 in the AI trend text) | 37.5 |
| Systolic BP | 90 on the dashboard (100 in the AI trend text) | 120 |
| Diastolic BP | 60 | 80 |
| MAP | 70 | 100 |

Each vital's trend is reported as "{status} & {direction}", for example "SpO₂ → low & decreasing" (below normal and still falling).

---

## 8. Clinical NLP — TF-IDF

**What TF-IDF is:** it turns text into numbers. A word scores high for a patient if it appears often in that patient's notes (term frequency) and is rare across all patients (inverse document frequency).

**Why TF-IDF and not ClinicalBERT:**
1. ClinicalBERT needs a separate tokenizer and embedding step.
2. It produces 768 numbers per note, far more than 90.
3. It needs a GPU to be efficient.
4. Accuracy was similar for this task.
5. A TF-IDF input named "creatinine" has a clear medical meaning; a BERT dimension does not.
6. TF-IDF inputs appear in SHAP with that meaning.

**Approach: a fixed 90-word list, not automatic selection.** Automatic selection of the 600 most variable words chose paperwork words. Sicker patients have longer records, so those words only reflected the amount of writing. The fixed list guarantees every input is a real medical idea.

```python
TfidfVectorizer(
    vocabulary=list(_SOFA_VOCABULARY),             # the 90 words
    ngram_range=(1, 2),                            # single words and two-word phrases
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b', # words of 3+ letters
    sublinear_tf=True,                             # 1 + log(count)
    norm="l2",                                     # long notes don't score higher by length
)
# max_features / min_df / max_df / stop_words are ignored when vocabulary= is set
```

**Corpus used in training:** all **283,208** notes (no sampling), grouped by admission, cleaned (privacy placeholders, numbers and symbols removed), then cut to the first **10,000 characters** per admission so lab mentions such as "creatinine rising" are kept. The fitted vectorizer is saved as `models/tfidf_vectorizer.pkl`.

**At the bedside:** `tfidf.transform([clinical_note])` with the saved vectorizer. Words outside the list are ignored. An empty or word-free note gives an all-zero vector, which is valid.

---

## 9. Target Variable — SOFA Score

**What SOFA is:** the Sequential Organ Failure Assessment, an international ICU score for multi-organ failure. Range **0 to 24**.

It scores six organ systems from 0 to 4 and adds them:
1. Respiratory (PaO₂/FiO₂)
2. Coagulation (platelets)
3. Liver (bilirubin)
4. Cardiovascular (MAP and blood-pressure drugs)
5. Central nervous system (Glasgow Coma Scale)
6. Renal (creatinine and urine output)

**Source:** `SOFA_24hours` from `mimiciii_derived.pivoted_sofa`, taken **at the same hour** as each row, not one value per stay. *(Historical: the earlier build used `mimiciii_derived.sofa`, one value per stay.)*

**Why SOFA is the target:**
- It is clinically validated and used worldwide.
- It covers six organ systems.
- It is available as a reliable pre-computed label.
- A continuous score gives more nuance than "stable / deteriorating".

**Risk bands:**

| SOFA | Risk level |
|---|---|
| 0 – 4 | Low Risk |
| 5 – 9 | Moderate Risk |
| ≥ 10 | High Risk (clinical definition) |

**Alert line:** the red alert fires at **predicted SOFA ≥ 7** (`ALERT_THRESHOLD = 7.0`), not ≥ 10. See Section 16. (It was ≥ 8 earlier.)

**Model output:** raw SOFA (0–24), with no scaling. The app clips the output to 0–24. An early version wrongly multiplied by 24 and produced values above 70; that bug was fixed.

**Why the model cannot be perfect:** SOFA is mostly computed from lab values and drug doses. ClinIQ is non-invasive, so those are not inputs.

---

## 10. Machine Learning Model — Evolution and Final Choice

### Stage 1 — Ensemble of trees (LightGBM + XGBoost + Random Forest) — historical

The first implementation trained an ensemble on the old 618-feature dataset. A grid search over weights found LightGBM 0.52 + XGBoost 0.48 + Random Forest 0.00. Performance: MAE 1.86, RMSE 2.44, R² 0.40.

**Why it was not the final model:** federated averaging averages weight matrices. Trees have none; each hospital's trees look different, so they cannot be averaged. That is why the project moved to a neural network. The ensemble code is kept for reference in the older folders outside `icu_monitor/` (`trial/`, `model training/`).

### Stage 2 — PyTorch neural network (current)

Neural networks fit federated learning naturally: their weights are tensors of the same shape at every hospital, and FedAvg averages them element by element.

**Architecture:**
```
Input: 108 scaled numbers
  ↓ Linear(108 → 256) → ReLU → Dropout(0.3)
  ↓ Linear(256 → 128) → ReLU → Dropout(0.3)
  ↓ Linear(128 →  64) → ReLU → Dropout(0.3)
  ↓ Linear( 64 →   1)
Output: SOFA score (0–24, direct)
```
Total trainable parameters: **69,121**. Dropout is active only during training; evaluation, SHAP and the backend call `model.eval()`, so predictions are deterministic. Class: `ICUModel` in `model_utils.py` (defaults `hidden=(256, 128, 64)`, `dropout=0.3`).

*(Historical: 108→128→64→32→1 with no dropout, 24,321 weights; before that, 618→256→128→64→1 with 199,681 weights from before the 90-word list.)*

**Why this architecture:** on `dataset_v2`, with plain MSE and FedAvg, the small 128-64-32 network reached its best round early and then overfit. Each hospital's training loss kept falling while the server's validation loss rose. Dropout 0.3 stops the memorising, and the extra width gives room to use it. On the same test set this improved MAE 1.850 → 1.825, R² 0.457 → 0.473 and AUROC 0.921 → 0.925 (first run of this setup). A still wider 512-256-128 gave no further gain.

**Dropout history (historical).** In an earlier setup (old dataset, FedYogi, weighted MSE, 128-64-32) dropout was tested and failed (R² 0.057 with p=0.3/0.2; −0.067 with p=0.1/0.05). It was blamed on hospitals using different random dropout masks, so their averaged updates disagreed. With the current setup (`dataset_v2`, plain MSE, FedAvg + FedProx) that problem did not appear and dropout helps. AdamW weight decay 1e-4 is still used alongside it.

**No BatchNorm or LayerNorm:** BatchNorm learns batch statistics that differ between hospitals and break when averaged. LayerNorm collapsed the prediction range to 0.22–7.04 because it rescales every patient the same way and hides how severe a high-SOFA patient is.

### Why not a tree model now, and how close are we to the ceiling?
A centralised gradient-boosting model on the same 108 inputs and the same test stays reaches MAE 1.79, R² 0.50, AUROC 0.922. Our federated network gets MAE 1.80, R² 0.48, AUROC 0.918. Choosing a network for federated learning cost almost nothing.

---

## 11. Federated Learning — Architecture and Evolution

### Why federated learning
Patient data cannot leave hospitals. Federated learning (FL) sends the **model** to the data instead:
- Each hospital trains locally on its own patients.
- Only the model's weights (arrays of numbers) are sent to the server.
- The server combines them with FedAvg (an average weighted by number of patients).
- The improved model goes back to every hospital.
- No patient record ever leaves a hospital.

### Framework: Flower (`flwr`)
Flower was chosen because it supports PyTorch, has a built-in FedAvg, runs both simulation and real multi-process FL, is well documented and is free.

### What is real and what is simulated

| Real | Simulated |
|---|---|
| Flower runs the real FL protocol | The 3 hospitals are made by splitting one MIMIC-III dataset |
| Only weights are exchanged | All clients run on one computer |
| Each client trains only on its own rows | The split is random (IID), so the hospital names (General / Mixed / Cardiac-Trauma ICU) are only labels |

### Two modes

**Mode 1 — Simulation (`train_federated.py`).** `fl.simulation.start_simulation()`; the 3 hospitals run inside one Python process, coordinated by Ray. This script produces the model the dashboard uses. It takes roughly 25–40 minutes on a modern CPU (100 rounds × 3 epochs), plus the first-time data preparation.

**Mode 2 — Real FL (`server.py` + `client.py`).** Separate processes over TCP at `127.0.0.1:8080`. Start the server, then three clients (`--client_id 0/1/2`). Each client reads its own rows from the `fl_training` table. The server uses the same FedAvg, runs 100 rounds, saves the best round to `federated_model.pth` and every round to `federated_model_latest.pth`, and writes partial metadata at the end.

**Extra tool — `fl_dashboard.py`.** A Streamlit app (port 8501) that shows a federated run live with charts. It runs FedAvg (FedYogi as an option).

### Server aggregation — FedAvg (current)
After each round:
1. The server sends the current weights to all clients.
2. Each client trains locally for 3 epochs (AdamW, plain MSE, FedProx, dropout).
3. Each client returns its weights.
4. The server computes FedAvg: `w = Σ(nᵢ × wᵢ) / Σnᵢ`.
5. `SaveBestStrategy` remembers the round with the lowest average validation error. In the final run this was **round 37**.

**History:** an earlier version used **FedYogi** on top of the average (η = 0.01). On `dataset_v2` it made the server validation loss explode in rounds 5–10 (from about 8 to 123), so it is now off (`USE_FEDYOGI = False`). Turning it on again uses a safer `SERVER_ETA = 0.001`.

### FedProx client regularisation
Each client's loss is `MSE + (μ/2) × ‖w_local − w_global‖²` with μ = 0.5. Clients are penalised for drifting from the shared model. μ = 0.01 was far too weak and 1.0 too strong.

### SaveBestStrategy
A custom Flower strategy on top of FedAvg (or FedYogi if switched on). It:
- averages the 3 hospitals' validation errors after every round,
- remembers the weights of the lowest one, and the final model saved to `federated_model.pth` is that **best round**, not the last,
- sends the round number to clients so each decays its learning rate (`lr = 0.001 × 0.99^(round−1)`, floor 1e-4).

Each hospital keeps a private 15% validation set of its own data.

### IID vs non-IID split
**IID (current):** the 41,798 training stays are split into three stratified groups of 13,932 / 13,933 / 13,933. Each has the same mix: Low 67%, Moderate 27%, High 6% (802 High Risk stays each).

**Non-IID (optional, `USE_NONIID_SPLIT = True`):** `make_noniid_split()` gives each hospital a different mix (General ICU 70% Low, Mixed ICU 60% Moderate, Cardiac/Trauma ICU 70% High). In an earlier test on the old dataset this dropped R² from about 0.28 to 0.08. Each hospital's model learned its own population and FedAvg could not reconcile them (**client drift**). It stays as an option to demonstrate FL challenges; IID is the default.

---

## 12. Training Improvements — What We Tried and Why

> **Read this section as history.** It lists improvements in the order they were made, most of them on the old `Dataset` build. Several were later reversed on `dataset_v2`: weighted MSE (Improvement 1) became plain MSE, FedYogi (Improvement 10) became FedAvg, and the network became 256-128-64 with dropout. The current setup is Improvement 12 and Section 20.

The original notebook trained with 10 rounds, 3 epochs, plain MSE, plain Adam, no imbalance handling. Result: R² 0.04, MAE 2.38 — the model predicted the average SOFA for everyone.

### Improvement 1 — Weighted MSE loss (historical; replaced by plain MSE)
**Problem:** 63% of the training data was Low Risk, so the model predicted about 4 for everybody (that minimises plain MSE). The 6% High Risk patients were ignored.
**Solution:** weight each error by `1 + target × 0.5`, normalised by the batch mean: SOFA 0 ≈ 0.33×, SOFA 4 ≈ 1×, SOFA 10 ≈ 2×, SOFA 20 ≈ 3.7×.
**Why 0.5 and not 3.0:** with 3.0, low-SOFA patients (63% of data) received only 7% of a fair signal, and training conflicted with the evaluation metric (unweighted, dominated by low-SOFA patients).
**Why linear:** a piecewise weight (Low 1×, Moderate 5×, High 20×) plus oversampling gave R² −0.822.
**Impact then:** R² 0.04 → 0.13–0.18.
**Later reversed:** on `dataset_v2` it pushed every prediction upward and raised MAE without catching more High Risk patients. Plain MSE together with FedAvg improved MAE 1.914 → 1.850 and R² 0.426 → 0.457. `train_model(..., weighted_loss=True)` still restores the old loss.

### Improvement 2 — AdamW optimiser
Changed `Adam` to `AdamW(lr=0.001, weight_decay=1e-4)`. Adam applies weight decay incorrectly because the adaptive learning rate interferes; AdamW decouples it. R² improved from 0.22 to 0.235–0.251.

### Improvement 3 — Training configuration
Rounds 10 → 20 → **100**; epochs 3 → 10 → **3** (fewer epochs mean less drift); learning-rate decay 0.99 per round. 100 rounds × 3 epochs gives enough total steps while limiting per-round drift.

### Improvement 4 — SaveBestStrategy
Save the round with the lowest validation loss instead of the last round. FL training is not steady (random batches, Ray scheduling, aggregation noise), so the last round is not necessarily the best.

### Improvement 5 — Learning-rate scheduling
The server passes the round number to clients through Flower's `configure_fit()`. Each client computes `lr = 0.001 × 0.99^(round−1)`, with a floor of 1e-4. At round 37 (the best round of the final run) the rate is about 0.0007.

### Improvement 6 — Oversampling (implemented, not active)
A `WeightedRandomSampler` draws High Risk patients about 10× more often (6% → 28% of each batch). With the weighted loss it gave R² −0.822: the few High Risk samples were repeated so often that the model overfit them and under-predicted the low-risk majority. The code remains in `model_utils.py` and `train_federated.py` (`OVERSAMPLE = False`).

### Improvement 7 — SHAP background samples
100 → 300 samples. With 100 samples (mostly low-SOFA patients), the baseline was biased; 300 gives more stable SHAP values.

### Improvement 8 — SOFA vocabulary list (breakthrough: +94% relative R²)
Replaced 600 automatic words with the fixed 90-word list (and the network at that time with 108→128→64→32→1). Automatic selection picked paperwork words. R² 0.13 → 0.1752, the biggest early jump.

### Improvement 9 — FedProx
Added `(μ/2)‖w_local − w_global‖²` to each hospital's loss. Without it, hospitals drifted and aggregation oscillated. R² about 0.13 → 0.18 (stable training).

### Improvement 10 — FedYogi on the server (historical; replaced by FedAvg)
FedAvg has no memory; each round is independent. FedYogi adds momentum and an adaptive step. On the old dataset: R² 0.18 (FedAvg) → 0.2148 (FedAdam) → 0.2229 (FedYogi). **Later reversed:** on `dataset_v2`, η = 0.01 made the loss explode, so FedAvg is used.

### Improvement 11 — More notes (breakthrough: +50% relative R²)
Notes 80,000 → all **283,208**; text per admission 5,000 → **10,000** characters. TF-IDF coverage of unique admissions rose from 22,245 to about 41,000, and longer text kept lab mentions ("creatinine 2.8") that were being cut off. R² on the old dataset 0.2229 → about 0.34 (later runs reached about 0.40–0.42).

### Improvement 12 — `dataset_v2`, FedAvg, plain MSE, wider network with dropout (current)
The data was rebuilt (Section 5): time-matched hourly SOFA labels, one row per ICU stay, 52,248 stays. Test set: 10,450 stays.

| Run | MAE | R² | AUROC | PR-AUC | Recall | Specificity |
|---|---|---|---|---|---|---|
| Old dataset (FedYogi, weighted MSE, 128-64-32), alert ≥ 8 — *historical* | 1.80 | 0.398 | 0.879 | — | 23% | — |
| `dataset_v2`, FedYogi η = 0.01, weighted MSE, alert ≥ 8 — *historical* | 1.914 | 0.426 | 0.915 | 0.599 | 53.8% | 98.0% |
| + FedAvg + plain MSE, alert ≥ 8 — *historical* | 1.850 | 0.457 | 0.921 | 0.628 | 59.3% | 97.7% |
| + 256-128-64 + dropout 0.3, alert ≥ 8 — *first run, historical* | 1.825 | 0.473 | 0.925 | 0.646 | 61.1% | 97.8% |
| **Final retrain, alert ≥ 7 (current)** | **1.801** | **0.481** | **0.918** | **0.647** | **67.8%** | **96.6%** |

The first row used the old dataset; its MAE is not directly comparable because the label changed. The last two rows are two runs of the same setup: federated simulation is not exactly repeatable, so they differ slightly. The recall jump comes mostly from the alert moving from 8 to 7.

---

## 13. SHAP Explainability

### Why SHAP
The model predicts a SOFA score but does not say why. SHAP answers: "for this patient, which inputs pushed the score up, and which pushed it down?" Clinicians will not act on a black-box number.

### Why DeepExplainer and not TreeExplainer
The first ensemble used `shap.TreeExplainer`, which only works for tree models. For a PyTorch network, `shap.DeepExplainer` is used; it attributes the output to each input by backpropagation (DeepLIFT).

### Background data
- **Size:** 300 typical training patients (raised from 100).
- **File:** `models/shap_background.npy`, shape **(300, 108)**.
- **Source:** randomly sampled from the training pool during `train_federated.py`; rewritten at every training run.
- **Why `eval()`:** it switches dropout off, so the explanation is deterministic.
- The explainer is created the first time a prediction is requested and cached.

### What the values mean
A positive SHAP value pushed the predicted SOFA up (worse). A negative one pushed it down.

### Clinical filtering
From all 108 values, only clinically meaningful inputs are kept by keyword: HR, RR, SpO2, Temp, SBP, DBP, MAP, GCS, stress, hypotension, respiratory, mental, septic, failure, intubated, vasopressor, shock, fever, infection, oxygen, ventilat, cardiac, renal, hepatic. Short abbreviations such as "hr" must match as a whole word, so "hr" inside "epinephrine" is not mistaken for heart rate. The **top 7** by size of effect are used for explanations and key risks, and the **top 5** are sent to the dashboard.

### Display uses original values
An early bug showed the scaled z-scores ("SpO₂_mean = −1.8"), meaningless to a clinician. The fix uses the pre-scaling value ("SpO₂ mean = 88.0%"). For note words the text says the note contains the word.

### A counter-intuitive result
Sometimes a high stress score for a deteriorating patient shows a *negative* SHAP effect. This is not a code bug. In MIMIC-III, stressed but responsive patients often had lower SOFA than unresponsive ones, because being responsive suggests less organ failure, and the model learned that pattern.

### Units mapping
`VITAL_UNITS` in `ml_service.py` maps each vital to a label and unit (for example `HR_mean → ("HR mean", "bpm")`, `GCS_eye_opening → ("GCS Eye Opening", "/4")`, `stress_score → ("Stress Score", "/10")`), and `_interpret_shap` turns each into a sentence such as "SpO₂ = 89% → low oxygen saturation, increasing risk".

---

## 14. LLM Integration — Groq

### Provider and model
- **Provider:** Groq, chosen for very fast inference, so a report appears in seconds.
- **Model:** `openai/gpt-oss-120b` for **both** jobs: writing the report and judging it.
- **API key:** read from `.env` (`GROQ_API_KEY`); never in code, never on GitHub.

| Job | Temperature | Max tokens | Notes |
|---|---|---|---|
| Write the report | 0.15 | 6,000 | Low temperature for factual, consistent text |
| Judge the report | 0.0 | 1,200 | Sees the first 900 characters; 12-second limit |

### What the prompt contains
The prompt holds, in order: an urgency header (alert only), the SOFA score and risk level, the ± uncertainty, the clinical note, the latest vitals with normal ranges, the neurological and stress values, the key risk factors from SHAP, the SHAP explanations (Hospital mode) and the vital-sign trends over the last 20 readings. The model is told to use only this data and not invent values.

### Two care modes

| | Hospital mode | Home Care mode |
|---|---|---|
| Audience | Clinicians | One home-care nurse and non-medical family members |
| Role given | Senior ICU physician following SCCM / Surviving Sepsis guidelines | Assistant writing for a nurse and anxious family |
| Style | Clinical; each drug has name, dose, route and target; each vital has value and normal range | Plain calm English; safe home steps; "go to hospital now if…" warning signs |

### Four sections
Every report has: **Current Condition**, **Probable Cause**, **Risk Forecast**, **Immediate Actions**.

### Risk-adaptive prompting
For alert patients (predicted SOFA ≥ 7, in both modes):
- the prompt opens with a clinical alert line,
- the sections are reordered so **Immediate Actions comes first**,
- the time horizon becomes the next 30 minutes,
- the actions ask for specific drugs, doses and procedures.

Non-alert patients get the normal order and a 2–4 hour horizon. For a critical patient, what to do *now* should be at the top, not buried under analysis.

### Practical details
- **Thinking text removed:** `<think>…</think>` blocks are stripped (including unclosed or stray tags).
- **Section parsing:** the reply is split into four sections even if headings use `#`, bold marks, numbers or dashes; all four keys always exist.
- **90-second cache** keyed by care mode, patient, SOFA and every vital in the prompt (HR, RR, SpO₂, SBP, DBP, MAP, temperature, GCS, stress). A changed reading is never given a report quoting old numbers. Expired entries are cleaned on every write.
- **One call for identical simultaneous requests:** a second request for the same report waits for the first.
- **Other mode prepared in the background**, so switching Hospital ↔ Home Care is instant.
- **Cost:** 2 calls per report; about 4 per new reading with both modes.
- **Failure:** the dashboard keeps the last report and shows a notice.

---

## 15. Checking the AI Report — The Reliability Score

> **The earlier "self-consistency check" was removed.** It sent the same prompt three times and compared the answers. That measures consistency, not correctness: a model can be consistently wrong ("patient is stable" three times while SpO₂ is 84%) and would still score High. It is replaced by checks against the real patient data.

The full explanation with worked examples is in **`docs/VALIDATION.md`**. Summary:

### Nine methods

| # | Method | Weight | What it checks |
|---|---|---|---|
| 1 | Factual grounding | 13% | Every abnormal vital is mentioned |
| 2 | Clinical hard rules | 13% | The standard treatment is mentioned (6 rules) |
| 3 | SHAP–LLM coherence | 9% | The report discusses what drove the prediction |
| 4 | Response structure | 9% | All four sections and enough length |
| 5 | Severity calibration | 8% | Tone matches how sick the patient is |
| 6 | Contraindication safety | 5% | No dangerous drug for these vitals (3 rules) |
| 7 | Numeric accuracy | 8% | Quoted numbers match real values |
| 8 | G-Eval (AI judge) | 13% | Overall quality, 4 aspects scored 1–5 |
| 9 | RAGAS faithfulness (AI judge) | 22% | Each unsupported claim lowers the score |

Methods 1–7 are instant rule checks (65% of the weight). Methods 8–9 use the AI judge (35%). If the judge is late (12 s) or fails, those two score a neutral 0.5 and the report is still shown.

### Labels
≥ 0.75 **High** (green) · 0.55–0.75 **Moderate** (amber) · < 0.55 **Low** (red). A separate **caution** flag is raised below 0.60. Every specific problem found (broken rules, dangerous drugs, wrong numbers, unsupported claims) is listed for the clinician.

### Where it shows
The reliability badge on the report, the same percentage in the top patient bar, and the LLM validation drawer with the per-method breakdown.

### Honest limits
The same model writes and grades the report, and the judge sees only the first 900 characters; the seven rule checks are the independent part. Keyword checks look for words, not meaning.

### Why this is a contribution
Most healthcare AI that uses an LLM shows one answer with no quality measure. Here every report is scored against the real data and ICU rules, with the reasons shown.

---

## 16. Alert System and Threshold Decision

### Risk bands vs the alert line
The risk bands follow clinical standards: Low < 5, Moderate 5–9, High ≥ 10. The **alert** fires earlier, at **predicted SOFA ≥ 7**. `ALERT_THRESHOLD = 7.0` is defined once in `model_utils.py`; `ml_service.py` imports it and `compute_sofa_risk_metrics()` uses it by default, so the dashboard and the training metrics always use the same value. (It was 8.0, defined in `ml_service.py`, earlier.)

### Why the lower line
The model **under-predicts severe patients**. For patients whose true SOFA is ≥ 10 the average true SOFA is about 13.1, but the average prediction is only about 9.5. Most of SOFA comes from labs and drug doses that are not inputs.

**Sweep on the final model** (10,450 test stays; 602 truly High Risk; false alarms are among the 9,848 others):

| Alert when predicted SOFA ≥ | Recall | Specificity | Missed | False alarms |
|---|---|---|---|---|
| 10 | 43.9% | 99.3% | 338 | 73 |
| 9 | 51.5% | 98.8% | 292 | 117 |
| 8 | 59.6% | 97.9% | 243 | 207 |
| 7.5 | 64.3% | 97.3% | 215 | 263 |
| **7 (current)** | **67.8%** | **96.6%** | **194** | **335** |
| 6.5 | 70.6% | 95.6% | 177 | 436 |
| 6 | 73.1% | 94.1% | 162 | 578 |

### Decision: alert at ≥ 7
- Moving from 8 to 7 catches 49 more critical patients (missed 243 → 194) for 128 more false alarms (207 → 335).
- Specificity stays high at 96.6%. Precision is 54.9% (about one alert in two is a truly High Risk patient).
- Below 7 the false alarms grow quickly for small gains (7 → 6 adds 243 false alarms to catch 32 more patients).
- AUROC, PR-AUC and MAE do not change; only the decision point moves.
- SOFA 7–9 is in the moderate-to-high range, so even a "false alarm" there deserves attention.

An alert at ≥ 7 is an early warning, not a High Risk classification: a prediction of 7.0–9.9 is Moderate by the SOFA definition. **Even at 7, 194 of 602 High Risk patients (32%) get no alert**, which is why ClinIQ supports clinicians and does not replace them.

### What the alert controls
The `is_alert` flag in the predict response and the urgent "Immediate Actions first" AI-report format, in both Hospital and Home Care modes.

### What the top banner shows
The banner uses the **risk level**, not the alert line: High Risk (≥ 10) → red **CRITICAL ALERT**; Moderate (5–9) → amber **ELEVATED RISK**; Low (< 5) → no banner and a green gauge.

---

## 17. Complete App Pipeline — Step by Step

### Backend (FastAPI, port 8000)

| Endpoint | What it does |
|---|---|
| `GET /health` | Returns `{"status": "ok", "service": "ClinIQ"}` |
| `GET /api/patients`, `GET /api/patients/{id}` | The 3 demo patients (404 if unknown) |
| `POST /api/predict` | Runs the full prediction. Optional live `cv_gcs_eye` (1–4) and `cv_stress` (0–10) |
| `POST /api/report` | Builds the AI report for that patient's latest prediction; `care_mode` is `hospital` or `homecare` |
| `GET /api/history/{id}?limit=` | Recent predictions (limit 1–50, default 5) |
| `GET /api/vitals/{id}?limit=` | Recent vitals for the charts (limit 1–20, default 20) |

Patient IDs are 1, 2 or 3. Bad input returns a clear 400. `/api/report` returns **409** if no prediction has run yet and **500** if `GROQ_API_KEY` is missing. The server stores each patient's latest prediction itself and **never trusts data sent by the browser** when writing a report. At app start the database tables are created if missing.

### What happens every 90 seconds (the `useMonitor.js` timer fires)

1. **Read the sample row.** The next of the patient's 30 recorded readings is read from `sample_patients` (looped).
2. **Choose camera values.** Live webcam GCS eye and stress replace the recorded values when present.
3. **Update the window.** Seed the vitals window if empty, append this reading, query the last 20.
4. **Trend features.** Compute the 9 trend features.
5. **Latest features.** The 7 latest vitals plus GCS and stress.
6. **TF-IDF.** `tfidf.transform([note])` → 90 columns.
7. **Combine and align.** 9 + 7 + 2 + 90 = **108** columns, missing ones filled with 0, reordered to `feature_columns.pkl`.
8. **Scale.** `np.clip(scaler.transform(...), -10, 10)`.
9. **Predict.** `ICUModel` in `eval()` with `torch.no_grad()`; clip to 0–24.
10. **Classify.** Risk band, `is_alert = sofa ≥ ALERT_THRESHOLD`, conformal range (sofa ± q̂), confidence.
11. **Runtime checks.** SOFA floor (abnormal MAP, SpO₂ or GCS imply a minimum), dangerous vital combinations (HR > 130 + SBP < 90; SpO₂ < 88 + MAP < 65), unexplained jumps (SOFA moves by more than 4 with no real vital change). They flag a problem and never change the predicted number.
12. **Save history.** Append to `prediction_history` (newest 50 kept).
13. **SHAP.** Compute 108 values → top 7 → top 5 for the dashboard, plus explanations and key risks.
14. **Trend text and chart data.**
15. **Return JSON** (including `_training_meta`, which the SOFA validation drawer reads).

### Then the report (frontend decides, backend writes)

16. **Trigger check (frontend).** `reportKey()` joins patient, SOFA and every vital (HR, RR, SpO₂, SBP, DBP, MAP, temperature, GCS, stress). If the key or the care mode changed, `POST /api/report` is called; otherwise the cached report is shown. A new reading discards reports for older readings.
17. **Cache check (backend).** The 90-second cache returns an identical earlier report.
18. **Prompt and call.** Build the care-mode prompt (alert-first if SOFA ≥ 7), call `openai/gpt-oss-120b`, strip thinking text, parse the 4 sections.
19. **Judge.** Call the judge (12-second limit).
20. **Score.** Run the nine-method reliability score; return sections, score, label, violations, caution flag, care mode and whether the judge answered.
21. **Prefetch.** After this mode's report is ready, the other mode is generated in the background.
22. **Display.** The one-page dashboard updates.

### Polling details (`useMonitor.js`)
- 90-second countdown; a new prediction is fetched when it reaches 1.
- `fetchingRef` prevents overlapping requests; the countdown resets only after data arrives.
- A response is discarded if the patient changed while it was in flight.
- The row position is kept in the browser's session storage so a refresh resumes from the same reading.

### What the user sees
- **Select-patient page:** three patient cards (Low, Moderate, High risk).
- **Monitor page:** header; alert banner; **SOFA gauge** and **vital cards**; **7 trend charts**; **AI report** with Hospital / Home Care toggle, four section cards, SHAP bars (top 5), reliability badge and disclaimer.
- **Four floating buttons:** History (last predictions), Camera monitor, LLM validation drawer, SOFA validation drawer (offline metrics read live from `training_metadata.json`, the three runtime checks, the conformal range).
- Dark and light themes.

### The three demo patients

| ID | Patient | Story |
|---|---|---|
| 1 | Low Risk | Post-surgical recovery, stable, improving |
| 2 | Moderate Risk | Community-acquired pneumonia on supplemental oxygen |
| 3 | High Risk | Septic shock, vasopressors, intubated, multi-organ failure |

### Database (`models/icu_monitor.db`, SQLite, gitignored)

| Table | Purpose | Rows now |
|---|---|---|
| `patient_vitals` | Sliding window of vitals (newest 20 per patient) | 60 |
| `prediction_history` | Prediction log (newest 50 per patient) | 150 |
| `sample_patients` | The recorded readings that are replayed (30 per patient) | 90 |
| `fl_training` | The 3 hospitals' scaled training rows | 41,798 |

---

## 18. All Files and Their Roles

### Python — training and shared code

**`model_utils.py` — shared code.** Imported by the backend, `train_federated.py`, `server.py`, `client.py` and `fl_dashboard.py`. Contains:
- `ALERT_THRESHOLD = 7.0` — the single source of the alert line
- `ICUModel` — the network (default `hidden=(256, 128, 64)`, `dropout=0.3`)
- `ICUDataset` — PyTorch dataset wrapper
- `weighted_mse_loss` — optional weighted loss (only with `weighted_loss=True`; default is plain MSE)
- `get_sample_weights` — oversampling infrastructure (off)
- `train_model` — mini-batch training: AdamW, plain MSE by default, optional FedProx term, gradient clipping
- `compute_sofa_risk_metrics` — band and alert confusion matrices, AUROC, PR-AUC, recall, specificity
- `evaluate_model` — MAE, R²
- `get_weights` / `set_weights` — weight serialisation for FL
- `get_trend`, `classify_range` — trend direction and normal/low/high
- `apply_dp_to_update`, `estimate_privacy_budget` — differential privacy (off)

One shared file keeps the client and the simulation on identical training, DP and model code.

**`train_federated.py` — the training script.** Phase 0 (BigQuery → notes → TF-IDF → scaling → hospital split → `fl_training` table) then phases 1–6 (load hospitals, SHAP background, federated training, save best model, test, save metadata). Produces `federated_model.pth`, `shap_background.npy`, `training_metadata.json`, `scaler.pkl`, `tfidf_vectorizer.pkl`, `feature_columns.pkl`, `fl_test_X.npy`, `fl_test_y.npy`. All settings are constants at the top.

**`server.py` — real Flower server.** Listens on 127.0.0.1:8080, waits for 3 clients, runs 100 rounds of FedAvg, saves the best round to `federated_model.pth` and every round to `federated_model_latest.pth`, then writes partial metadata (`source: server.py`, no MAE/R²).

**`client.py` — real Flower client.** `--client_id 0/1/2`. Loads its rows from `fl_training`, keeps a private 15% validation set, trains 3 epochs per round with FedProx (and optional DP).

**`fl_dashboard.py` — Streamlit simulation dashboard.** Shows a federated run live (port 8501).

**`db.py` — database layer.** Connection in WAL mode, `init_db`, vitals, history, sample rows, `get_fl_client_data`, `fl_training_exists`. Window sizes: 20 vitals, 50 history rows.

**`migrate_to_db.py` — imports the demo CSVs** (`data/sample_patients/*.csv` and any legacy vitals/history CSVs) into the database.

**`setup.py` — one-command setup:** checks Python ≥ 3.12 and Node ≥ 18, makes a virtual environment, installs dependencies, initialises the database, creates `.env` from `.env.example`.

### Backend

| File | Role |
|---|---|
| `backend/main.py` | App entry, CORS for ports 5173/5174/5175, 5 routers, `/health`, database init |
| `backend/api/predict.py` | Validates input, calls `run_prediction`, stores the full result server-side for the report, returns a clean copy |
| `backend/api/report.py` | Uses the stored prediction (never browser data), calls `generate_report` |
| `backend/api/patients.py`, `history.py`, `vitals.py` | Patient list, prediction history, vitals window |
| `backend/services/ml_service.py` | Whole inference pipeline, SHAP, risk label, SOFA floor, vital-combination flags, trajectory check, conformal q̂ (computed from the training data, fallback 2.9), training-metadata loader |
| `backend/services/llm_service.py` | Prompts for both modes, Groq calls, cache, section parser, nine reliability methods, judge, labels |

### Frontend (React 19 + Vite 8)

| File | Role |
|---|---|
| `hooks/useMonitor.js` | 90-second loop with guards and session-stored row index |
| `pages/MonitorDashboard.jsx`, `SelectPatient.jsx` | The monitor page and the landing page |
| `components/report/ReportContainer.jsx` | Per-mode report fetching, per-reading cache, prefetch of the other mode |
| `components/report/SHAPPanel.jsx` | Animated SHAP bars (anime.js); keeps last good factors if SHAP fails once |
| `components/cv/CVMonitorModal.jsx` | Browser camera monitor (Section 23) |
| `components/validation/SOFAValidationDrawer.jsx` | Offline metrics read live from `_training_meta`, runtime checks, conformal range; "—" if metadata missing |
| `components/validation/LLMValidationDrawer.jsx` | Nine-method breakdown and every violation |
| `components/layout/HeaderBar.jsx`, `AlertBanner.jsx`, `ThemeToggle.jsx` | Header, risk banner, theme switch |
| `components/overview/…`, `trends/…`, `history/…` | Gauge, vital cards, charts, history drawer |

*(The old Streamlit `app.py` and the old CSV window files are gone; ClinIQ is the only dashboard.)*

### Saved model files in `models/`

| File | Contents |
|---|---|
| `federated_model.pth` | PyTorch state dict: keys `net.0`, `net.3`, `net.6`, `net.9` (weight and bias; ReLU and dropout layers have none). A checkpoint from a different network size will not load, so retrain instead. |
| `scaler.pkl` | StandardScaler fitted on the 108 inputs; `feature_names_in_` is the authoritative order |
| `tfidf_vectorizer.pkl` | TF-IDF with the 90-word list, fitted on 283,208 notes |
| `feature_columns.pkl` | The 108 names in training order |
| `shap_background.npy` | 300 training rows, shape (300, 108); rewritten at each training run |
| `training_metadata.json` | Settings and test results of the last full training run; the backend sends it as `_training_meta` and the SOFA drawer reads every number from it. Its `evaluation_limitation` text reads: "One row per ICU stay, split 80/20 by ICU stay — a patient with several stays can appear in both train and test. Retrospective MIMIC-III results, not independent clinical validation." |
| `fl_test_X.npy`, `fl_test_y.npy` | The held-out test set (10,450 stays) |
| `icu_monitor.db` | SQLite database (MIMIC-III derived; gitignored) |

### Other
`data/sample_patients/` holds the 3 demo CSVs (30 recorded readings each). `tests/` holds 745 tests: 513 Python (295 federated learning + 218 backend) and 232 Vitest (27 files), plus 3 Playwright e2e specs that need a running app.

---

## 19. Configuration Reference

### `model_utils.py`
```python
ALERT_THRESHOLD = 7.0     # alert fires when predicted SOFA >= this value
```

### `backend/services/ml_service.py`
```python
from model_utils import ALERT_THRESHOLD       # imported, not redefined
MODEL_PATH = <icu_monitor>/models/
# conformal q̂ = 90th percentile of |prediction - truth| over the training stays
# (computed at first use; fallback 2.9 if the training data is unavailable)
```

### `backend/services/llm_service.py`
```python
_CACHE_TTL = 90                       # seconds
model      = "openai/gpt-oss-120b"    # writer AND judge
writer: temperature 0.15, max_tokens 6000
judge:  temperature 0.0,  max_tokens 1200, timeout 12 s, sees first 900 characters
reliability labels: >= 0.75 High, >= 0.55 Moderate, else Low;  caution < 0.60
weights: 0.13 0.13 0.09 0.09 0.08 0.05 0.08 0.13 0.22
```

### `frontend/src/hooks/useMonitor.js`
```js
const [countdown, setCountdown] = useState(90)   // 90-second cycle; resets to 90 after data arrives
// countdown never goes below 1; a fetch fires when it reaches 1
```

### `train_federated.py`
```python
BIGQUERY_PROJECT = "mimic-project-2";  BIGQUERY_DATASET = "dataset_v2"
NOTES_SAMPLE_SIZE = 283_208;  NOTES_TEXT_LIMIT = 10_000
NUM_ROUNDS = 100;  EPOCHS_PER_ROUND = 3;  BATCH_SIZE = 64;  SHAP_BG_SAMPLES = 300
BASE_LR = 0.001;  LR_DECAY = 0.99          # floor 1e-4
GRAD_CLIP = 1.0;  MU_FEDPROX = 0.5
USE_FEDYOGI = False                         # False = FedAvg (current)
SERVER_ETA = 0.001                          # only if USE_FEDYOGI
OVERSAMPLE = False                          # disabled (overfits)
USE_NONIID_SPLIT = False                    # disabled (client drift)
USE_DP = False;  DP_SENSITIVITY = 1.0;  DP_SIGMA = 1.0
```

### `client.py`
```python
EPOCHS = 3;  BATCH_SIZE = 64;  BASE_LR = 0.001;  LR_DECAY = 0.99
GRAD_CLIP = 1.0;  MU_FEDPROX = 0.5
USE_DP = False;  DP_SENSITIVITY = 1.0;  DP_SIGMA = 1.0     # keep in sync with train_federated.py
SERVER_ADDR = "127.0.0.1:8080"
```

### Normal ranges (`ml_service.py`)
```python
NORMAL_RANGES = {
    "HR": (60, 100), "RR": (12, 20), "SpO2": (95, 100),
    "Temp": (36.0, 37.5), "SBP": (90, 120), "DBP": (60, 80), "MAP": (70, 100),
}
# The short trend summary sent to the AI report uses Temp 36.5 and SBP 100 as lower limits.
```

---

## 20. Model Performance Summary

### Current model (`federated_model.pth`, final run, 2026-11-24)

| Metric | Value |
|---|---|
| MAE | **1.80** SOFA points (1.8006) |
| R² | **0.48** (0.4806) |
| AUROC (High Risk) | **0.918** (0.9181) |
| PR-AUC (High Risk) | **0.647** (0.6466) |
| Prediction range on the test set | 1.78 – 23.07 |
| Training / test stays | 41,798 (13,932 / 13,933 / 13,933) / 10,450 (602 High, 2,821 Moderate, 7,027 Low) |
| Data | BigQuery `dataset_v2`, newest row per ICU stay, 80/20 split stratified by High Risk |
| Architecture | 108 → 256 → 128 → 64 → 1, ReLU, Dropout 0.3 — 69,121 parameters |
| FL rounds | 100 (best = **round 37**) |
| Server / client method | FedAvg (`USE_FEDYOGI = False`) + FedProx (μ = 0.5) |
| Loss | plain MSE |
| Optimiser | AdamW lr 0.001 (×0.99 per round, floor 1e-4), weight decay 1e-4, gradient clip 1.0 |
| Conformal 90% interval | ± 3.28 SOFA points |

**Evaluation limitation:** one row per ICU stay, split 80/20 by ICU stay. A patient with several stays can appear in both train and test, which can flatter the scores. These are retrospective MIMIC-III results, not independent clinical validation.

### Per risk level

| Risk level | Stays | MAE | Share placed in the right band | Notes |
|---|---|---|---|---|
| Low (< 5) | 7,027 | 1.33 | 95.5% | Most patients; small errors |
| Moderate (5–9) | 2,821 | 2.44 | 23.7% | About 74% are predicted as Low |
| High (≥ 10) | 602 | 4.36 | 43.9% | Wide SOFA range; model under-predicts (mean prediction ≈ 9.5, true mean ≈ 13.1) |

**Per-band R² is negative by design** (Low −0.71, Moderate −3.53, High −1.75): there is little variance inside one band, so only the overall R² (0.48) is meaningful. Ranking patients inside a band would need lab values.

**3-band confusion matrix** (rows actual, columns predicted: Low / Moderate / High):
```
[[6714,  306,   7],
 [2086,  669,  66],
 [ 119,  219, 264]]
```
3-band accuracy **73.2%** (7,647 of 10,450). High-band precision 0.78, recall 0.44.

### Alert performance (predicted SOFA ≥ 7 vs actual SOFA ≥ 10)

| Metric | Value |
|---|---|
| Recall | **67.8%** (408 of 602) |
| Specificity | **96.6%** |
| Precision | 54.9% |
| F1 | 0.607 |
| Missed (FN) | 194 |
| False alarms (FP) | 335 |
| Correctly quiet (TN) | 9,513 |

### Comparison across versions

| Version | R² | Key change |
|---|---|---|
| Original FL notebook (10 rounds, plain MSE) | 0.04 | Baseline *(historical)* |
| Weighted MSE | 0.11–0.13 | Loss improvement *(historical)* |
| AdamW + 300 SHAP samples | 0.13–0.18 | Optimiser *(historical)* |
| Piecewise loss + oversampling | −0.822 | Failed *(historical)* |
| Dropout in the old setup | 0.057 / −0.067 | Failed with FedYogi *(historical)* |
| SOFA vocabulary list (90 words) | 0.1752 | Feature breakthrough *(historical)* |
| FedAdam | 0.2148 | Server optimisation *(historical)* |
| FedYogi | 0.2229 | Adaptive step *(historical)* |
| All notes + 10k characters + FedYogi | ≈ 0.34–0.42 | Notes expansion *(old dataset, historical)* |
| Old dataset final | 0.398 | Last old-dataset model *(historical)* |
| `dataset_v2`, FedYogi, weighted MSE | 0.426 | New dataset *(historical)* |
| + FedAvg + plain MSE | 0.457 | Simpler server and loss |
| + 256-128-64 + dropout 0.3 (first run) | 0.473 | Wider network with dropout |
| **Final retrain** | **0.481** | **Current model** |

R² on the old dataset and on `dataset_v2` is not directly comparable: the label changed.

### How close to the ceiling?
Centralised gradient boosting on the same 108 inputs and the same test stays: MAE 1.79, R² 0.50, AUROC 0.922. The federated network is close; more accuracy would need lab values as inputs.

---

## 21. What Did Not Work and Why

Understanding what failed is as important for the viva as understanding what works.

### 1. Piecewise weighted loss + oversampling
**Tried:** Low 1×, Moderate 5×, High 20× weights plus a sampler drawing High Risk 10× more often.
**Result:** R² −0.822; loss rose from 12 to 110 over 20 rounds.
**Why:** the few High Risk samples were repeated so often that the model overfit them. Three hospitals each biased high, averaged, gave a model predicting high SOFA for everyone, ruining low-risk accuracy.

### 2. Dropout in FL (failed in the old setup — used now)
> **Update:** dropout 0.3 is part of the **current** model. The failure below was in the old setup (old dataset, FedYogi, weighted MSE, 128-64-32).

**Tried:** p = 0.3 / 0.2, then 0.1 / 0.05. **Result:** R² 0.057 and −0.067.
**Why (as understood then):** each hospital zeroes different random neurons, so their updates disagree and averaging fails. On `dataset_v2` with plain MSE and FedAvg + FedProx this did not happen; dropout fixed the overfitting. AdamW weight decay (1e-4) is still used too.

### 3. Non-IID hospital split
**Tried:** Hospital 0 with 70% Low SOFA, Hospital 2 with 70% High.
**Result:** R² 0.084 vs 0.28 with IID.
**Why:** client drift. Each local model fits its own population; the average is mediocre for all. This is the fundamental non-IID challenge in FL.

### 4. Learning-rate decay 0.97 (historical; now 0.99)
**Result:** worse overall, with more oscillation late. A very low rate late on cannot recover from a bad aggregation round. The current 0.99 is slow and stable.

### 5. Gradient clipping (historical finding)
**Earlier result:** clipping at 1.0 or 5.0 reduced R². **Why:** the weighted loss deliberately produced large gradients for high-SOFA patients and clipping cut exactly those. The current plain-MSE model uses `GRAD_CLIP = 1.0`.

### 6. ClinicalBERT for notes
**Result:** similar accuracy to TF-IDF with far more compute. **Decision:** stay with TF-IDF (simpler, faster, interpretable in SHAP).

### 7. Training on all of the old automatic 600 words
Paperwork words ("tablet", "sig", "refills") measured the amount of writing, not illness. Replaced by the 90-word list (+94% relative R²).

### 8. Experiments on `dataset_v2` that did not help
Lower learning rate 5e-4, stronger weight decay 1e-3, Huber loss, a wider 512-256-128 network, averaging 8 models, isotonic recalibration, missing-value indicator flags, and several hourly rows per stay (+0.007 R² for 9× more data).

**The ceiling:** centralised gradient boosting on the same inputs gives MAE 1.79, R² 0.50, AUROC 0.922. The main limit is the input design: SOFA is mostly computed from labs and drug doses, which are deliberately not inputs.

---

## 22. Privacy — Differential Privacy Layer

### What DP adds
Without DP, each hospital sends its weights directly, and a clever server could try to reconstruct training data from them (model inversion). With DP, before sending, each hospital:
1. computes `update = local weights − global weights`;
2. clips the update's L2 size to `DP_SENSITIVITY` (so no single patient can influence it much);
3. adds Gaussian noise of spread `DP_SIGMA × DP_SENSITIVITY` to every weight;
4. sends `global weights + clipped noisy update`.

The server therefore cannot recover any individual patient's contribution.

### Privacy budget
```
ε per round ≈ sqrt(2 × ln(1.25 / δ)) / σ        (δ = 1e-5, σ = 1.0 → about 4.85)
ε total (simple composition) = ε per round × number of rounds
```
- **100 rounds (current setting): ε ≈ 485**
- 20 rounds (the earlier real-demo setting): ε ≈ 97

Smaller ε is more private; below about 1 is strong. At ε ≈ 485 the guarantee is **very weak**, so the feature demonstrates the mechanism rather than giving real protection. Strong privacy would need far more noise and fewer rounds, which would hurt accuracy a lot.

### Current status
DP is implemented in `train_federated.py` (simulation) and `client.py` (real FL), with the same logic in `model_utils.py: apply_dp_to_update()`. It is **off** (`USE_DP = False`). To enable it, set `USE_DP = True` in both files (keep `DP_SENSITIVITY` and `DP_SIGMA` in sync) and retrain.

---

## 23. Camera Monitor — Stress and Eye-Opening Scores

The camera monitor lives inside the dashboard (the 2nd floating button). It runs **entirely in the browser**: React plus **MediaPipe FaceLandmarker** compiled to WebAssembly (files in `frontend/public/mediapipe/`). The first use downloads a ~12 MB face model. Only the two resulting numbers are sent to the backend; no video leaves the device. (There is no standalone `cv_monitor.py` any more.) Full details: `docs/CV_MONITOR.md`.

### Stress score (0–10)
Built from the **PSPI** pain formula (Prkachin & Solomon, 2008): brow lowering, cheek or eye squeeze, nose and upper-lip raising, eye closing, reduced by smiling. It is mixed 80% facial and 20% head movement, then smoothed with an exponential average (α = 0.15). Bands: **LOW < 3.5**, **MODERATE < 6.5**, **HIGH ≥ 6.5**. These same bands are used in the vital cards, the SHAP explanations and the AI-report rules.

### GCS eye score (1–4)
Based on how often the eyes are open over a rolling **30-second** window:

| Result | Rule | Score |
|---|---|---|
| Spontaneous | Eyes open at least 70% of the time | 4 |
| To voice | Eyes at least partly open ≥ 30% of the time | 3 |
| To pain | A brief flicker of opening (≥ 0.1 s) | 2 |
| No response | Otherwise | 1 |

A short **personal calibration** (about 3 seconds, at most about 10 seconds for a patient whose eyes stay closed) sets this person's open-eye level. A blink score above 0.55 counts as closed.

### Behaviour
- Readings are sent at most once a second and stop if no face is seen for 3 seconds.
- When active, the live values replace the recorded GCS and stress in the next prediction.
- Because these two inputs carry modest weight in the trained model (Section 6), the SOFA changes only slightly with them.

---

## 24. Limitations

- **No lab values or drug doses as inputs.** This limits R² to about 0.48 and makes the Moderate band hard (only 23.7% placed correctly).
- **Under-prediction of severe patients.** Even at alert line 7, 194 of 602 High Risk stays (32%) get no alert.
- **Split by ICU stay, not by patient.** A patient with several stays can be in both train and test. Results are retrospective, not independent clinical validation.
- **Artificial hospitals.** The split is random (IID); there is no real hospital-to-hospital difference.
- **Conformal range measured on training data.** Real coverage on new patients is probably a little under 90%.
- **The judge also wrote the report** and sees only 900 characters; the keyword checks look for words, not meaning.
- **Replayed demo data.** The dashboard replays 30 recorded readings per sample patient; the model does not retrain itself.
- **Camera cues are lightly weighted** in training.
- **Differential privacy is off** and would be weak (ε ≈ 485).
- **The AI report needs internet and a Groq key.**
- **Run-to-run variation.** Two training runs of the same settings differ slightly (federated simulation is not exactly repeatable); quote `models/training_metadata.json`.

---

## 25. Glossary

| Term | Plain meaning |
|---|---|
| **SOFA** | A 0–24 score of how badly six organs are failing; higher = sicker |
| **ICU stay** | One intensive-care admission; the unit of one training row |
| **MIMIC-III** | A public, de-identified ICU database from Boston |
| **BigQuery** | Google's cloud database holding our MIMIC-III copy |
| **Federated learning (FL)** | Training one model across sites without moving their data |
| **Flower** | The framework that runs the federated rounds |
| **FedAvg** | Server averages the hospitals' weights, bigger hospitals counting more |
| **FedProx** | A penalty that keeps each hospital close to the shared model |
| **FedYogi** | Averaging plus momentum and an adaptive step; optional, off by default |
| **IID / non-IID** | Hospitals have the same / different patient mixes |
| **Client drift** | Hospitals' models moving in different directions during local training |
| **Dropout** | Randomly switching off neurons in training so the model cannot memorise |
| **Overfitting** | Better on training data, worse on new data |
| **MAE** | Average size of the prediction error |
| **R²** | How much better than "guess the average for everyone" (1 = perfect) |
| **AUROC** | Chance the model ranks a sicker patient above a less sick one |
| **PR-AUC** | Precision and recall combined; good for rare conditions |
| **Recall / specificity** | Share of sick patients caught / share of healthy patients left alone |
| **TF-IDF** | A score for how important a word is in one document compared with all |
| **StandardScaler** | Rescales each input to average 0 and spread 1 |
| **SHAP** | Explains a prediction as contributions from each input |
| **Conformal interval** | A range from past errors that holds the truth about 90% of the time |
| **q̂ (q-hat)** | The error size 90% of past patients stayed within (3.28) |
| **LLM / LLM judge** | An AI that writes text / an AI call that grades another AI's output |
| **Hallucination** | AI text that sounds right but is not supported by the data |
| **GCS eye opening** | Eye part of the Glasgow Coma Scale: 4 = opens by itself, 1 = never |
| **PSPI** | A facial pain formula used to score stress from facial movements |
| **MAP** | Mean arterial pressure; below 65 is dangerous |
| **Alert threshold** | The predicted SOFA at which the red alert fires (7) |
| **CDSS** | Clinical decision support system: assists, does not decide |
| **Differential privacy** | Adding noise so individual records cannot be recovered |

---

## Appendix — Key Numbers at a Glance

| Parameter | Value |
|---|---|
| Total features | 108 (9 trend + 7 latest + 2 camera + 90 TF-IDF) |
| Sliding window | 20 readings |
| TF-IDF approach | 90-word SOFA list (not data-driven) |
| Notes used / text limit | 283,208 / 10,000 characters |
| BigQuery dataset | `dataset_v2` |
| Cohort | 52,248 ICU stays / 38,304 patients |
| Training / test samples | 41,798 / 10,450 |
| Hospital clients | 3 (13,932 / 13,933 / 13,933) |
| Architecture | 108 → 256 → 128 → 64 → 1, Dropout 0.3 |
| Trainable parameters | 69,121 |
| Loss | plain MSE |
| FL rounds / epochs per round / batch | 100 / 3 / 64 |
| Best round (final run) | 37 |
| MAE / R² | 1.80 / 0.48 |
| AUROC / PR-AUC (High Risk) | 0.918 / 0.647 |
| 3-band accuracy | 73.2% |
| Alert threshold | SOFA ≥ 7 (`ALERT_THRESHOLD = 7.0`) |
| Alert recall / specificity / precision | 67.8% / 96.6% / 54.9% |
| Conformal 90% interval | ± 3.28 |
| SHAP background / shown | 300 / top 5 |
| AI model (write and judge) | `openai/gpt-oss-120b` |
| Write: temperature / max tokens | 0.15 / 6,000 |
| Judge: temperature / max tokens / timeout | 0 / 1,200 / 12 s |
| Reliability labels | ≥ 0.75 High, 0.55–0.75 Moderate, < 0.55 Low |
| AI calls per report | 2 (1 write + 1 judge) |
| Report cache / frontend refresh | 90 s / 90 s |
| Prediction history kept | newest 50 per patient |
| Server aggregation | FedAvg (`USE_FEDYOGI = False`; optional FedYogi η = 0.001) |
| FedProx μ | 0.5 |
| Tests | 745 (513 Python + 232 Vitest) |

---

*End of Project Context Document.*
*Last updated: 2026-11-24 — final model: `dataset_v2`, FedAvg + FedProx, plain MSE, 256-128-64 with dropout 0.3, alert ≥ 7. ClinIQ (React + FastAPI) is the production dashboard; the old Streamlit `app.py` is gone. When the model is retrained, update Sections 20 and 16 and the Appendix from `models/training_metadata.json`.*

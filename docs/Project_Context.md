# Project Context — Complete Implementation Guide
## ICU Clinical Decision Support System with Federated Learning

> This document contains the **complete technical context** of the project — what was built, how it was built, why every design decision was made, what was tried and failed, and what the final approach is. Any person reading this should understand the full scope and implementation of the project from start to finish.

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Problem Being Solved](#2-problem-being-solved)
3. [What the System Does — End to End](#3-what-the-system-does--end-to-end)
4. [Dataset — MIMIC-III and BigQuery](#4-dataset--mimic-iii-and-bigquery)
5. [Preprocessing — SQL Pipeline](#5-preprocessing--sql-pipeline)
6. [Feature Engineering — All 618 Features](#6-feature-engineering--all-618-features)
7. [Sliding Window Mechanism](#7-sliding-window-mechanism)
8. [Clinical NLP — TF-IDF](#8-clinical-nlp--tf-idf)
9. [Target Variable — SOFA Score](#9-target-variable--sofa-score)
10. [Machine Learning Model — Evolution and Final Choice](#10-machine-learning-model--evolution-and-final-choice)
11. [Federated Learning — Architecture and Evolution](#11-federated-learning--architecture-and-evolution)
12. [Training Improvements — What We Tried and Why](#12-training-improvements--what-we-tried-and-why)
13. [SHAP Explainability](#13-shap-explainability)
14. [LLM Integration — Groq](#14-llm-integration--groq)
15. [LLM Self-Consistency Check](#15-llm-self-consistency-check)
16. [Alert System and Threshold Decision](#16-alert-system-and-threshold-decision)
17. [Complete App Pipeline — Step by Step](#17-complete-app-pipeline--step-by-step)
18. [All Files and Their Roles](#18-all-files-and-their-roles)
19. [Configuration Reference](#19-configuration-reference)
20. [Model Performance Summary](#20-model-performance-summary)
21. [What Did Not Work and Why](#21-what-did-not-work-and-why)
22. [Privacy — Differential Privacy Layer](#22-privacy--differential-privacy-layer)

---

## 1. Project Identity

**Full Title:**
Multimodal Intelligence System with Federated Learning for Continuous Patient Monitoring and Early Deterioration Detection Leveraging IoMT, Computer Vision and Clinical NLP

**Type:** Final Year Engineering Capstone Project in Computer Science

**Scope:** AI-powered ICU Clinical Decision Support System (CDSS)

**What it is NOT:** The system does NOT diagnose any disease, does NOT prescribe treatment, and does NOT make autonomous decisions. It is a decision support tool — it surfaces AI-generated insights that clinicians review and act upon with their own judgment.

---

## 2. Problem Being Solved

ICU patients can deteriorate rapidly. Traditional monitoring fails in three specific ways:

**Problem 1 — Alarm Fatigue:**
Current ICU systems fire alarms when a single vital sign crosses a fixed threshold (HR > 120 → alarm). This produces hundreds of false alarms per day. Clinical staff start ignoring alarms. Real emergencies get missed. This is called alarm fatigue and is a documented patient safety problem.

**Problem 2 — No Multimodal Integration:**
Each vital sign is assessed in isolation. A heart rate of 105 bpm means nothing without knowing what the SpO₂ is doing, what the blood pressure trend looks like, and what the doctor's notes say. Current systems cannot combine all of these into a unified picture.

**Problem 3 — Patient Data Cannot Be Shared:**
Traditional machine learning requires centralising all hospital data in one place to train a model. In healthcare, patient data is legally protected (HIPAA, GDPR). No hospital can send its patient records to another institution or to a central server. This makes collaborative AI training ethically and legally impossible with conventional ML.

**Our System Solves All Three:**
- Multimodal fusion (vitals + CV + clinical text) replaces threshold-based alarms
- Trend analysis + SHAP + LLM gives a complete, explainable patient picture
- Federated Learning allows hospitals to train a shared model without sharing patient data

---

## 3. What the System Does — End to End

```
INPUTS (entered by nurse/doctor in Streamlit sidebar)
  ├─ Clinical Notes (free text: history, medications, observations)
  ├─ 7 Vital Signs: HR, RR, SpO₂, Temperature, SBP, DBP, MAP
  ├─ GCS Eye Opening (1–4) — neurological indicator
  └─ Stress Score (0–10) — behavioural/physiological distress

PREPROCESSING
  ├─ Append new vitals to rolling 20-reading sliding window
  ├─ Compute 9 trend features (mean, std, min from last 20 readings)
  ├─ Extract 7 latest vital values (current state)
  ├─ Convert clinical notes to 600 TF-IDF features
  └─ Combine into 618-feature vector → StandardScaler

PREDICTION
  └─ Federated PyTorch DNN (618→256→128→64→1)
     → Raw SOFA score (0–24)
     → Risk level: Low (<5) / Moderate (5–9) / High (≥10)
     → Alert fires if predicted SOFA ≥ 8

EXPLAINABILITY
  ├─ SHAP DeepExplainer → top 7 clinical features driving the prediction
  ├─ Clinical interpretation of each feature
  └─ Key risk factors mapped from SHAP

LLM CLINICAL REPORT
  ├─ Groq Llama 3.3 70B called 3 times
  ├─ For High Risk: IMMEDIATE ACTIONS listed first (30-min horizon)
  ├─ For normal: Current condition, cause, forecast, actions
  ├─ TF-IDF cosine similarity → consistency score (0–1)
  └─ Reliability label: High (≥0.85) / Moderate / Low

DISPLAY (4 tabs in Streamlit)
  ├─ Tab 1: Risk Assessment — SOFA, severity, vital trend charts, prediction history
  ├─ Tab 2: Explainability — SHAP table, clinical interpretations, key risk factors
  ├─ Tab 3: AI Clinical Report — LLM output, consistency score, all 3 responses
  └─ Tab 4: Federated Learning — FL config, model architecture, DP status
```

---

## 4. Dataset — MIMIC-III and BigQuery

**Dataset:** MIMIC-III (Medical Information Mart for Intensive Care III)

**Why MIMIC-III and not MIMIC-IV:**
MIMIC-IV was available but MIMIC-III was chosen because it is more stable for preprocessing, has more documented derived tables (including pre-computed SOFA scores in `mimiciii_derived.sofa`), and is better-tested for research use.

**Access method:** Google BigQuery

**BigQuery Project:** `mimic-project-2`
**BigQuery Dataset:** `Dataset`
**Final processed table:** `mimic-project-2.Dataset.ml_dataset_final`

**Main tables used:**

| Table | Purpose |
|---|---|
| `physionet-data.mimiciii_clinical.chartevents` | All vital signs, GCS Eye Opening, Stress Score |
| `physionet-data.mimiciii_notes.noteevents` | Clinical notes (nursing, physician) |
| `physionet-data.mimiciii_derived.sofa` | Pre-computed SOFA scores (the target label) |
| `physionet-data.mimiciii_clinical.icustays` | ICU stay metadata |

**Final dataset size:** 60,188 ICU measurement windows, each with 108 features and one SOFA score target

---

## 5. Preprocessing — SQL Pipeline

12 BigQuery SQL queries were run in sequence to transform raw MIMIC-III data into an ML-ready dataset.

### Query 1 — Extract Raw Vitals
Extracts 7 vital signs from CHARTEVENTS using specific item IDs.
Temperature measured in Fahrenheit (itemid=223761) is converted to Celsius: `(value - 32) × 5/9`

MIMIC-III item IDs used:
- HR: 211, 220045
- RR: 618, 220210
- SpO₂: 646, 220277
- Temperature: 223762, 678, 223761 (Fahrenheit → converted)
- SBP: 51, 220179
- DBP: 8368, 220180
- MAP: 52, 220181

### Query 2 — Pivot to Wide Format
Groups by (subject_id, hadm_id, icustay_id, charttime) and creates one row per timestamp with separate columns for each vital sign.

### Query 3 — Aggregate to 10-Minute Windows
ICU measurements are irregular (not taken at fixed intervals). This query creates 10-minute time buckets using:
`TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(charttime), 600) * 600)`
and computes AVG of each vital within each window. This regularises the time series.

### Query 4 — Compute Sliding Window Trend Features
Uses SQL window function `ROWS BETWEEN 19 PRECEDING AND CURRENT ROW` (20-row window) to compute:
- HR_mean, HR_std
- RR_mean
- SpO2_mean, SpO2_min
- Temp_mean
- SBP_mean, DBP_mean, MAP_mean

### Query 5 — Extract Latest Vital Values
From the 10-minute windows, extracts the raw current values as `latest_HR`, `latest_RR`, etc.

### Query 6 — Extract Clinical Notes
From NOTEEVENTS, keeps only relevant categories:
- Nursing, Physician, Nursing/Other, Discharge summary
Avoids Radiology and Echo (post-event summaries, not continuous monitoring).

### Query 7 — Extract SOFA Scores
From `physionet-data.mimiciii_derived.sofa` — this pre-computed table saves us from manually calculating SOFA from lab values. SOFA = target label.

### Query 8 — Join Trends + Latest Vitals + SOFA
Combines trend features (Q4), latest vitals (Q5), and SOFA target (Q7) into a single ML dataset.

### Query 9 — Extract GCS Eye Opening
itemid=220739 — Glasgow Coma Scale eye response (1–4).
This is a neurological indicator originally from clinical assessments, repurposed as a Computer Vision feature (conceptually: a camera-based CV model would estimate this from patient behaviour).

### Query 10 — Extract Stress Score
itemid=223791 — originally called "Pain Score" in MIMIC-III.
**Renamed to `stress_score`** in the BigQuery database before the notebook was run. All saved artifacts (scaler.pkl, client CSVs) use `stress_score`, not `pain_score`.

### Query 11 — Combine CV Features into 10-Minute Windows
UNIONs GCS Eye Opening and Stress Score tables, then aggregates into 10-minute time windows.

### Query 12 — Create Final ML Dataset
LEFT JOINs the base ml_dataset (Q8) with CV features (Q11) on (icustay_id, window_time).
This is the final table: `ml_dataset_final`

---

## 6. Feature Engineering — All 618 Features

The 618 features fed into the model are split into 4 groups:

### Group 1 — Trend Features (9 features)

Computed from the sliding window of the last 20 vital sign readings. These capture the patient's physiological trajectory, not just the current state.

| Feature | Description | Why This Feature |
|---|---|---|
| HR_mean | Average heart rate over last 20 readings | Captures sustained elevation |
| HR_std | Standard deviation of HR | Captures instability / arrhythmia signals |
| RR_mean | Average respiratory rate | Sustained respiratory distress |
| SpO2_mean | Average oxygen saturation | Sustained hypoxaemia |
| SpO2_min | Minimum SpO₂ value seen | Catches even brief dangerous dips |
| Temp_mean | Average temperature | Fever / hypothermia detection |
| SBP_mean | Average systolic BP | Sustained haemodynamic compromise |
| DBP_mean | Average diastolic BP | Diastolic function |
| MAP_mean | Average mean arterial pressure | Most clinically important BP metric |

**Why trend and not just latest values:** A single reading of HR=105 tells you nothing. But if HR was 82→85→90→98→105 over 20 readings, the patient is clearly deteriorating. Trend analysis captures this temporal information. Without it, you cannot detect early deterioration.

**Slope features were intentionally excluded:** The original design specification (in `project_context_4.md`) included slope features (HR_slope, RR_slope, etc.). These were not included because they were not in the final processed BigQuery dataset (the SQL windows only computed mean/std/min). Adding slopes would have required reprocessing the full MIMIC-III data in BigQuery.

### Group 2 — Latest Vital Values (7 features)

| Feature | Clinical Meaning |
|---|---|
| latest_HR | Most recent heart rate (bpm) |
| latest_RR | Most recent respiratory rate (breaths/min) |
| latest_SpO2 | Most recent oxygen saturation (%) |
| latest_Temp | Most recent temperature (°C) |
| latest_SBP | Most recent systolic BP (mmHg) |
| latest_DBP | Most recent diastolic BP (mmHg) |
| latest_MAP | Most recent MAP (mmHg) |

These represent the patient's current state. The combination of trend features AND latest values gives the model both historical context and present status.

### Group 3 — Computer Vision Features (2 features)

| Feature | Range | Clinical Meaning |
|---|---|---|
| GCS_eye_opening | 1–4 | Glasgow Coma Scale eye response. 1=no response, 4=spontaneous. Lower=worse. |
| stress_score | 0–10 | Physiological and behavioural stress/pain level. Higher=more distress. |

**Why these are called Computer Vision features:** In a real hospital deployment, a bedside camera connected to a CV model would continuously estimate these neurological and behavioural indicators without any clinician having to manually assess them. In this project, the values come from MIMIC-III (manually recorded by clinicians) and are entered manually in the Streamlit sidebar. The conceptual pipeline represents what would happen with camera-based monitoring.

### Group 4 — TF-IDF SOFA Vocabulary Features (90 features)

Clinical notes are converted to 90 numerical TF-IDF features using an **explicit SOFA vocabulary whitelist** — every term directly corresponds to one of the 6 SOFA organ components. This replaced the original data-driven 600-feature approach, which selected documentation-style terms ("tablet", "sig", "refills") that correlated with note volume rather than patient severity.

The 90-term vocabulary covers: respiratory (intubated, ventilator, bipap, hypoxia...), coagulation (plt, platelets, inr, coagulopathy...), hepatic (bilirubin, totbili, bili, jaundice...), cardiovascular (vasopressor, levophed, septic shock, hypotension...), CNS (sedated, coma, encephalopathy, gcs...), renal (creatinine, dialysis, oliguria, aki...), sepsis (sepsis, bacteremia, cultures...), lab values (lactate, wbc, hco, angap...), and severity terms (failure, ards, pneumonia...).

**Total: 9 + 7 + 2 + 90 = 108 features**

### Feature Scaling

All 108 features are scaled using `StandardScaler` (zero mean, unit variance) fitted on the training dataset, then clipped to [-10, 10]. The fitted scaler is saved as `models/scaler.pkl`.

**Important:** The `data/fl_training/client_0/1/2.csv` files contain ALREADY SCALED data. The scaler was applied during Phase 0 of `train_federated.py`. Do not re-apply the scaler to these CSVs.

### Feature Order

The exact order of all 108 features is critical. The saved `models/feature_columns.pkl` stores the authoritative column order (identical to `scaler.feature_names_in_`). During inference, the feature vector is always reordered to match this exact sequence before scaling and prediction.

---

## 7. Sliding Window Mechanism

**Why a sliding window:**
Health deterioration is not a single event — it is a process that unfolds over time. A patient deteriorating from sepsis will show HR gradually climbing over 2 hours before becoming critical. A single snapshot misses this. The sliding window captures the recent trajectory.

**Window size: 20 readings**

Each reading = one 10-minute aggregation window from the original MIMIC-III preprocessing. 20 readings × 10 minutes = approximately the last 3 hours of patient monitoring.

**How it works in the app:**
1. On each prediction click, the new vital signs are appended to `models/patient_vitals.csv`
2. The CSV is trimmed to the last 20 rows (`tail(20)`) — oldest reading is dropped
3. All 9 trend features are recomputed from this updated window
4. The updated window is saved back to CSV

**Initialisation:**
If `patient_vitals.csv` does not exist, a default 20-row dataset is created showing a gradually deteriorating patient (HR climbing from 88 to 109, SpO₂ dropping from 97 to 88, etc.). This ensures the trend features are always available from the first prediction.

**Reset functionality:**
The sidebar has a "🔄 Reset Patient History" button that recreates the default 20-row CSV. This should be used when switching from one patient to another.

**Trend direction calculation:**
Uses `numpy.polyfit` to fit a line through the 20 readings. Slope > 0.1 = "increasing", slope < -0.1 = "decreasing", otherwise "stable". The 0.1 threshold was chosen to filter out measurement noise.

**Normal range classification:**

| Vital | Normal Low | Normal High |
|---|---|---|
| HR | 60 | 100 |
| RR | 12 | 20 |
| SpO₂ | 95 | 100 |
| Temperature | 36.5 | 37.5 |
| SBP | 100 | 120 |
| DBP | 60 | 80 |
| MAP | 70 | 100 |

Each vital's trend is reported as "{status} & {direction}" — for example "SpO₂ → low & decreasing" means the oxygen is below normal AND still falling.

---

## 8. Clinical NLP — TF-IDF

**What is TF-IDF:**
Term Frequency-Inverse Document Frequency. Converts text into numbers. Each word/phrase gets a score based on how often it appears in this document vs how common it is across all documents. Rare but document-specific words get higher scores.

**Why TF-IDF and not ClinicalBERT or other embeddings:**
ClinicalBERT was explored. It was abandoned because:
1. More complex integration (separate tokenizer and embedding step)
2. Produces 768-dimensional embeddings, much larger than 90 SOFA vocab features
3. Requires GPU for efficient computation
4. Performance was similar to TF-IDF for this specific task
5. TF-IDF features are directly interpretable (a feature named "creatinine" is clinically meaningful; a BERT embedding dimension is not)
6. TF-IDF features appear in SHAP analysis with clinical meaning

**Approach: SOFA Vocabulary Whitelist (NOT data-driven)**

Rather than letting TF-IDF pick the 600 most statistically variable terms (which selects noise terms like "tablet", "sig", "refills"), we use an explicit 90-term whitelist where every term directly maps to a SOFA organ component.

**Vectorizer configuration:**
```python
TfidfVectorizer(
    vocabulary=list(_SOFA_VOCABULARY),   # 90-term SOFA whitelist
    ngram_range=(1, 2),                  # unigrams + bigrams
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b',
    sublinear_tf=True,
    norm="l2",
)
# max_features/min_df/max_df/stop_words are ignored when vocabulary= is set
```

**Corpus:**
- NOTES_SAMPLE_SIZE = 283,208 (all available notes — no sampling)
- NOTES_TEXT_LIMIT = 10,000 characters per admission (captures lab values in progress notes)
- Notes grouped by `hadm_id`, cleaned, then truncated
- Fitted vectorizer saved as `models/tfidf_vectorizer.pkl`

**At inference:**
The user's clinical note is transformed using the saved vectorizer (`tfidf.transform([note])`). The same 600 vocabulary terms are used — no re-fitting. If the note contains words not in the vocabulary, those words are simply ignored.

**Handling empty notes:**
If no clinical note is entered, the app substitutes "No clinical notes provided" which produces a valid but sparse vector.

---

## 9. Target Variable — SOFA Score

**What is SOFA:**
Sequential Organ Failure Assessment. An internationally validated ICU severity scoring system that quantifies multi-organ dysfunction.

**Range:** 0 to 24

**How SOFA is computed (6 organ systems, each scored 0–4):**
1. Respiratory — PaO₂/FiO₂ ratio
2. Coagulation — Platelet count
3. Liver — Bilirubin
4. Cardiovascular — MAP and vasopressor requirement
5. Central Nervous System — Glasgow Coma Scale
6. Renal — Creatinine and urine output

**Source in this project:** Pre-computed SOFA from `physionet-data.mimiciii_derived.sofa`. This saves the considerable work of computing it from raw lab values.

**Why SOFA was chosen as the target:**
- It is clinically validated and used worldwide by ICU clinicians
- It captures multi-organ failure across 6 systems — comprehensive
- It is available as a pre-computed, reliable label in MIMIC-III
- It gives a continuous severity measure rather than a binary label
- Predicting a continuous SOFA gives clinicians a nuanced picture, not just "deteriorating/stable"

**Risk classification thresholds:**

| Predicted SOFA | Clinical Label |
|---|---|
| 0 – 4 | Low Risk |
| 5 – 9 | Moderate Risk |
| ≥ 10 | High Risk (clinical definition) |

**Alert threshold:** The alert fires at **predicted SOFA ≥ 8**, not ≥ 10. See Section 16 for why.

**Model output:** The model outputs raw SOFA values (0–24 range) directly. There is NO multiplication by 24 at inference — the model was trained on raw SOFA values, not normalised ones. This was a bug in earlier versions that caused predicted SOFA values of 70+. It was fixed.

---

## 10. Machine Learning Model — Evolution and Final Choice

### Phase 1 — Ensemble ML (LightGBM + XGBoost + Random Forest)

**What was done:** The first implementation trained an ensemble of three gradient boosting models on the 618-feature dataset.

**Grid search for optimal weights:** Used an exhaustive grid search over all combinations of (w_XGB, w_LGB, w_RF) with step 0.01. Best weights found: LightGBM 0.52 + XGBoost 0.48 + Random Forest 0.00 (RF contributed nothing useful).

**Ensemble performance:**
- MAE: 1.86 SOFA points
- RMSE: 2.44
- R²: 0.40

**Why the ensemble was NOT used as the final model:**
Federated Learning with FedAvg works by averaging model weight matrices across clients. Gradient boosting models (LightGBM, XGBoost, Random Forest) do not have a weight matrix that can be averaged — each tree has a different structure at each hospital. You literally cannot apply FedAvg to tree-based models. This is the fundamental reason the project moved from the ensemble to a neural network.

**The ensemble is kept as reference** in `notebooks/Final_ICU.ipynb` (original BigQuery notebook, not executable locally).

### Phase 2 — PyTorch Deep Neural Network (Current)

**Why PyTorch DNN:**
Neural networks are the natural fit for Federated Learning. Their parameters (weight matrices) are real-valued tensors of the same shape on every client. FedAvg simply averages these tensors element-wise. This makes FL mathematically straightforward for DNNs.

**Architecture:**

```
Input: 108 scaled features
    ↓
Linear(108 → 128)
    ↓
ReLU
    ↓
Linear(128 → 64)
    ↓
ReLU
    ↓
Linear(64 → 32)
    ↓
ReLU
    ↓
Linear(32 → 1)
    ↓
Output: SOFA score (0–24, direct, no scaling)
```

Total parameters: ~23,000 (vs 199,681 with the old 618→256→128→64→1 architecture)

**Why this specific architecture:**
With ~100 input features, a 2:1 samples-per-parameter ratio (48k samples / 23k params)
is appropriate for tabular data. The old 618→256 first layer was over-parameterised
at only 0.5:1, causing poor generalisation. The 108→128→64→32→1 architecture
progressively compresses features into a single SOFA prediction.

**Why no Dropout:**
Dropout was tested at two settings:
- p=0.3 (after layer 1), p=0.2 (after layer 2): R² = 0.057 — catastrophic failure
- p=0.1, p=0.05 (conservative): R² = -0.067 — worse than no dropout

**Root cause:** Each hospital trains with different random Dropout masks. Hospital A zeros out neurons {1, 5, 23, ...}. Hospital B zeros out neurons {3, 7, 19, ...}. Their gradients point in completely different directions. FedAvg averages these divergent gradients — the global model cannot converge. This is a documented incompatibility between Dropout and FedAvg in the FL research literature.

**Regularisation without Dropout:** The AdamW optimizer's `weight_decay=1e-4` provides L2 regularisation that achieves a similar effect without the FL incompatibility.

**Output behaviour:**
The model outputs raw SOFA (0–24) directly. The `np.clip(raw_pred, 0, 24)` in app.py ensures predictions stay in the valid clinical range.

---

## 11. Federated Learning — Architecture and Evolution

### Why Federated Learning

Patient data cannot leave hospitals. Traditional ML requires centralising all data in one place. Federated Learning solves this:
- Each hospital trains the model locally on its own private patient data
- Only model weights (numpy arrays of floats) are transmitted to a central server
- The server aggregates the weights using FedAvg (weighted average by dataset size)
- The improved global model is distributed back to all hospitals
- No raw patient data ever leaves any hospital

### Framework: Flower (flwr)

Flower was chosen because:
- Native support for PyTorch (our model framework)
- Built-in FedAvg strategy
- Supports both simulation and real multi-process FL
- Well-documented and actively maintained
- Free and open-source

### Two Modes of Operation

**Mode 1 — Simulation (train_federated.py):**
Uses `fl.simulation.start_simulation()`. All 3 hospital clients run in the same Python process, coordinated by Ray. Used for training the model locally. Takes ~25–40 minutes on a modern CPU (100 rounds × 3 epochs).

**Mode 2 — Real FL (server.py + client.py):**
True separate processes communicating over TCP sockets on 127.0.0.1:8080. Server runs first, then 3 separate client processes connect. Each client reads its own private CSV. Demonstrates the actual FL communication protocol for demo purposes.

### FedYogi Server Aggregation

After each round:
1. Server distributes current global model weights to all clients
2. Each client performs local training (3 epochs with AdamW + weighted MSE + FedProx)
3. Each client returns its updated weights to the server
4. Server computes FedAvg: `w_avg = sum(n_i × w_i) / sum(n_i)`
5. Server applies FedYogi update: accumulates momentum and applies adaptive update
6. SaveBestStrategy saves the round with the lowest average validation loss

### FedProx Client Regularisation

Each client's local loss = `L_data + (μ/2) × ||w_local - w_global||²`

With μ=0.5, clients are penalised for drifting too far from the global model during
local training, preventing the oscillation that plagued plain FedAvg.

### The SaveBestStrategy

A custom Flower strategy extending FedAvg. In addition to standard aggregation, it:
- Tracks the best (lowest) evaluation loss across all rounds
- Saves the best model as `federated_model_best.pth`
- Saves the latest model (every round) as `federated_model.pth`
- Passes the current round number to clients via `configure_fit()` for LR scheduling

### IID vs Non-IID Hospital Split

**What IID means:** The 48,151 training samples are split randomly (33%/33%/34%) across 3 hospitals. Each hospital sees the same distribution of SOFA scores.

**What was implemented:** `data/client_0/1/2.csv` are IID splits produced by the original BigQuery notebook.

**Non-IID was implemented and tested:**
A `make_noniid_split()` function was written that creates a biased split:
- Hospital 0 (General ICU): 70% Low SOFA / 20% Moderate / 10% High
- Hospital 1 (Mixed ICU): 20% Low / 60% Moderate / 20% High
- Hospital 2 (Cardiac/Trauma): 10% Low / 20% Moderate / 70% High

**Non-IID result:** R²=0.084 vs IID R²=0.28. The non-IID split causes **client drift** — each hospital's local model optimises for its own patient population, and FedAvg cannot reconcile these conflicting updates into a good global model. This is a known limitation of standard FedAvg, which motivates advanced FL algorithms like FedProx and SCAFFOLD.

**Decision:** Non-IID remains as an option (`USE_NONIID_SPLIT=True`) for viva demonstration purposes, but IID is the default for production use.

---

## 12. Training Improvements — What We Tried and Why

The original notebook trained the FL model with:
- 10 FL rounds
- 3 epochs per client per round
- Plain MSE loss
- Plain Adam optimizer
- No class imbalance handling
- Result: R²=0.04, MAE=2.38 — the model predicted the mean SOFA for all patients

### Improvement 1 — Weighted MSE Loss

**Problem:** 63.3% of training data is Low Risk (SOFA < 5). The model learns to predict ~4.0 for every patient (the class mean) because this minimises plain MSE. The 6.0% High Risk patients are effectively ignored.

**Solution:** Weight the MSE loss so high-SOFA patients contribute more to the gradient.

**Final formula: `weight = 1 + target × 0.5`**

This gives:
- SOFA=0: weight ≈ 0.33× (after batch-mean normalisation)
- SOFA=4 (mean): weight ≈ 1.0×
- SOFA=10 (High Risk): weight ≈ 2.0×
- SOFA=20 (severe): weight ≈ 3.7×

**Why 0.5 multiplier (reduced from the original 3.0):**
With multiplier=3.0, low-SOFA patients (63% of data) received only 7% of fair gradient signal.
The model optimised for high-SOFA but the server evaluation metric (unweighted MSE) was
dominated by low-SOFA patients — training and evaluation objectives conflicted.
Reducing to 0.5 gives low-SOFA patients 33% of fair signal while still emphasising High Risk.

**Why linear and not piecewise:**
Piecewise weighting (Low=1×, Moderate=5×, High=20×) was tested. Combined with oversampling
it produced R²=-0.822. The linear formula provides a smooth, stable gradient.

**Impact:** R² improved from 0.04 to 0.13–0.18.

### Improvement 2 — AdamW Optimizer

**Previous:** `torch.optim.Adam(model.parameters(), lr=0.001)`

**Changed to:** `torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)`

**Why AdamW:**
Standard Adam has a known issue — weight decay is applied incorrectly (it gets affected by the adaptive learning rate). AdamW fixes this by applying weight decay separately (decoupled weight decay). For tabular data regression tasks, AdamW consistently outperforms Adam.

**Impact:** R² improved from 0.22 to 0.235–0.251.

### Improvement 3 — Training Configuration Tuning

**Rounds:** 10 → 20 → **100** FL rounds
**Epochs:** 3 → 10 → **3** (optimal sweet spot — fewer epochs reduce client drift)
**LR decay:** Added 0.99 per-round decay for stable convergence

**Reason:** 100 rounds with 3 epochs each gives sufficient total gradient steps while
limiting per-round drift. The FedYogi server optimizer compensates for fewer local
epochs by accumulating server-side momentum.

### Improvement 4 — SaveBestStrategy

**Previous:** Save the model from the final round (round 10/20)
**Changed to:** Save the model from the round with the lowest average validation loss

**Why:** FL training is non-monotonic — due to stochasticity (random batch sampling, Ray parallelism, FL aggregation noise), the model quality oscillates across rounds. Round 20 is not necessarily the best round. The SaveBestStrategy tracks validation loss across all rounds and saves the checkpoint with the lowest loss.

### Improvement 5 — LR Scheduling

**What was built:** The server passes the current round number to each client via the Flower `configure_fit()` config mechanism. Clients compute:
`lr = BASE_LR × (LR_DECAY ^ (round-1))`

**Current setting: LR_DECAY = 0.99** (1% decay per round).
At round 24 (best): LR = 0.001 × 0.99^24 ≈ 0.000787.
This is a slow, stable decay that allows fine-tuning in later rounds.

### Improvement 6 — Oversampling (Implemented, Not Active)

**What was built:** `WeightedRandomSampler` that draws High Risk patients 10× more often per epoch, bringing them from 6% to 28% of each training batch.

**Why it was disabled:**
Tested: `OVERSAMPLE=True` with the weighted loss combined caused R²=-0.822. The problem: the 2,904 high-risk samples get repeated so many times per epoch that the model overfits to them. The globally averaged model under-predicts for the 63% low-risk majority. The WeightedRandomSampler infrastructure remains in `model_utils.py` and `train_federated.py` for future experimentation.

### Improvement 7 — SHAP Background Samples

**Previous:** 100 background samples for SHAP DeepExplainer
**Changed to:** 300 background samples

**Why:** SHAP DeepExplainer uses a background/reference dataset to compute baseline attributions. With only 100 samples (mostly low-SOFA patients since 63% of data is low-risk), the baseline is biased. 300 samples gives more stable, representative SHAP values.

### Improvement 8 — SOFA Vocabulary Whitelist (Breakthrough: +94% relative R²)

**Previous:** Data-driven 600-feature TF-IDF selected statistically variable terms
**Changed to:** Explicit 90-term SOFA vocabulary whitelist, 108→128→64→32→1 architecture

**Why:** Data-driven TF-IDF selected documentation-style noise terms ("tablet", "sig",
"refills", "wallet") that correlated with note volume rather than patient severity.
The whitelist ensures every feature corresponds directly to a SOFA organ component.

**Impact:** R² improved from 0.13 to 0.1752 (+94% relative improvement — the single
biggest improvement in the entire project).

### Improvement 9 — FedProx Client Regularisation

**Added:** `(μ/2) × ||w_local - w_global||²` proximal term to each hospital's loss.

**Why:** Without FedProx, hospitals drifted far from the global model during local
training, causing FedAvg aggregation to oscillate. FedProx with μ=0.5 keeps local
models within a bounded distance from the global, stabilising FL convergence.

**Impact:** R² improved from ~0.13 to ~0.18 (stable training).

### Improvement 10 — FedYogi Server Optimizer

**Previous:** FedAvg (plain average, no server learning)
**Changed to:** FedYogi (adaptive server-side optimization with momentum)

**Why:** FedAvg has no memory — each round is independent. FedYogi accumulates
server-side momentum in the direction of consistent improvement, allowing the global
model to find deeper minima even when individual rounds are noisy.

**Impact:** R² from ~0.18 (FedAvg) → 0.2148 (FedAdam) → 0.2229 (FedYogi).

### Improvement 11 — Notes Corpus Expansion (Breakthrough: +50% relative R²)

**Previous:** NOTES_SAMPLE_SIZE=80,000, NOTES_TEXT_LIMIT=5,000 chars
**Changed to:** NOTES_SAMPLE_SIZE=283,208 (all notes), NOTES_TEXT_LIMIT=10,000 chars

**Why:** With 80k notes, TF-IDF covered only 22,245 unique admissions. With all 283k
notes, it covers 41,179+ admissions — 85% more. Longer text captures lab value mentions
("creatinine 2.8", "bilirubin elevated at 4.5") in progress notes that were cut off.

**Impact:** R² from 0.2229 to **0.3357** (+50% relative improvement).

---

## 13. SHAP Explainability

### Why SHAP

The model predicts a SOFA score but does not inherently explain which features caused it. SHAP (SHapley Additive exPlanations) answers: "For this specific patient, which features pushed the prediction toward a higher SOFA (worse outcome) and which pushed it toward a lower SOFA?"

This is essential for clinical trust — clinicians will not act on a black-box number. They need to understand what the model saw.

### Why DeepExplainer and not TreeExplainer

The original ensemble implementation used `shap.TreeExplainer`. TreeExplainer only works with tree-based models (LightGBM, XGBoost, Random Forest). After switching to a PyTorch DNN, TreeExplainer cannot be used.

`shap.DeepExplainer` is specifically designed for deep learning models. It uses a backpropagation-based attribution method (DeepLIFT) to compute how much each input feature contributed to the output.

### Background Data

DeepExplainer requires a reference "background" dataset to establish baseline attribution. This is a set of representative training samples against which the model's response to the input is compared.

- **Size:** 300 samples (originally 100, increased for stability)
- **File:** `models/shap_background.npy` — shape (300, 618)
- **Source:** Randomly sampled from the combined training pool during `train_federated.py`
- **Why the model must be in `eval()` mode for SHAP:** This ensures any stochastic layers (Dropout if present) are disabled. Deterministic forward pass is required for reliable attributions.

### What SHAP Values Mean

A positive SHAP value for a feature means that feature pushed the prediction toward a higher SOFA (worse condition). A negative value means it pushed toward a lower SOFA.

### Clinical Feature Filtering

From all 618 SHAP values, clinically relevant features are identified using keyword matching:
- Vital sign keywords: HR, RR, SpO2, Temp, SBP, DBP, MAP, GCS, stress
- Clinical text keywords: hypotension, respiratory, mental, septic, failure, intubated, vasopressor, shock, fever, infection, oxygen, ventilat, cardiac, renal, hepatic

Top 7 features by absolute SHAP impact from the filtered set are selected.

### Feature Display — Original Values Not Z-Scores

**Previous (wrong):** The SHAP table showed `raw_value = final_scaled[0]` — the StandardScaler z-scores. A clinician seeing "SpO₂_mean = -1.8" has no idea what that means.

**Fixed:** The SHAP table uses `original_value = final_df.iloc[0].values` — the pre-scaling values. Now shows "SpO₂ mean = 88.0%" which is immediately clinically meaningful.

For TF-IDF features: shows "detected" if the term appeared in the note, "absent" if it did not.

### SHAP Counterintuitive Results Disclaimer

A known issue: stress_score=8 for a clearly deteriorating patient sometimes shows a negative SHAP impact (reducing risk). This is not a code bug — it reflects a MIMIC-III training data correlation. In MIMIC-III, stressed but responsive patients (high stress score) sometimes had lower SOFA than unresponsive patients, because consciousness implies less organ failure. The model learned this correlation. The Tab 2 disclaimer explains this to clinicians.

### VITAL_UNITS Mapping

To display values with proper units, a mapping was created:

```python
VITAL_UNITS = {
    "HR_mean": ("HR mean", "bpm"),
    "HR_std":  ("HR variability", "bpm"),
    "SpO2_mean": ("SpO₂ mean", "%"),
    "SpO2_min":  ("SpO₂ minimum", "%"),
    "GCS_eye_opening": ("GCS Eye Opening", "/4"),
    "stress_score":    ("Stress Score", "/10"),
    # ... etc
}
```

---

## 14. LLM Integration — Groq

### Provider and Model

- **Provider:** Groq (fast inference API)
- **Model:** `llama-3.3-70b-versatile`
- **Why Groq:** Extremely fast inference compared to OpenAI. Near-real-time response for ICU decision support where speed matters.
- **Why this model:** Llama 3.3 70B offers strong clinical reasoning capability at a scale appropriate for structured medical context analysis.

### LLM Prompt Structure

The prompt is constructed from 7 sections:
1. Urgency header (only for High Risk — see below)
2. SOFA score and risk level
3. Clinical notes (user input)
4. Latest vital signs with normal ranges
5. Neurological and stress indicators (GCS, stress) with interpretation scale
6. Key risk factors (from SHAP analysis)
7. SHAP feature explanations
8. Vital sign trend summary (direction + status for all 7 vitals)

### Risk-Adaptive Prompting

**This is a key innovation added during implementation.**

For High Risk patients (predicted SOFA ≥ 8):
- The prompt opens with `⚠️ CLINICAL ALERT — Predicted SOFA X.X`
- The 4 sections are **reordered**: IMMEDIATE ACTIONS comes FIRST (not last)
- Time horizon changes from "next 2–4 hours" to "next 30 minutes"
- IMMEDIATE ACTIONS requests specific drug names, doses, procedures

For non-alert patients:
- Standard section order: Current Condition → Probable Cause → Risk Forecast → Immediate Actions
- Standard 2–4 hour time horizon

**Why this matters:** For a genuinely critical patient, you want the actionable output (what to do RIGHT NOW) at the top of the response, not buried at the bottom after paragraphs of analysis.

### LLM Parameters

- **Temperature:** 0.7 (for self-consistency calls — variation needed to test reliability)
- **Max tokens:** 800
- **System prompt:** "You are an expert ICU clinical decision support assistant. Provide concise, structured, and clinically accurate reasoning."

---

## 15. LLM Self-Consistency Check

### The Problem It Solves

LLMs can hallucinate — produce confident but incorrect outputs. In a clinical setting, an unreliable AI recommendation is dangerous. A single response cannot be assessed for reliability.

### The Mechanism

1. The same prompt is sent to Groq **3 times** (temperature=0.7 allows natural variation)
2. The 3 responses are collected
3. Each response is vectorised using `sklearn.feature_extraction.text.TfidfVectorizer` (separate from the clinical note TF-IDF — this vectorizer is created fresh each inference)
4. Pairwise cosine similarity is computed between all response pairs
5. Average similarity across all pairs = **consistency score**

**Formula:**
```
consistency_score = (similarity_matrix.sum() - n) / (n × (n-1))
```
Where n=3, and diagonal values (self-similarity=1.0) are excluded.

### Reliability Labels

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.85 | ✅ High Reliability | All 3 responses agree — trust the output |
| 0.65–0.85 | ⚠️ Moderate Reliability | Some variation — review carefully |
| < 0.65 | ❌ Low Reliability | Significant variation — apply extra clinical judgment |

### What Gets Displayed

- The first response (Response 1) is the primary output shown prominently
- The consistency score and reliability label are displayed as metrics
- An expandable "View all 3 LLM responses" section shows all three for comparison

### Why This Is a Novel Contribution

Most healthcare AI systems that use LLMs call them once and display the single result. There is no quality measure on the output. This self-consistency mechanism provides a quantitative reliability estimate using only the model's own outputs — no ground truth needed.

---

## 16. Alert System and Threshold Decision

### Clinical SOFA Thresholds vs Alert Threshold

The SOFA score risk classification follows clinical standards:
- **Low Risk:** SOFA < 5
- **Moderate Risk:** SOFA 5–9
- **High Risk:** SOFA ≥ 10 (clinical definition)

However, the system alert fires at **predicted SOFA ≥ 8**, not ≥ 10.

### Why the Lower Alert Threshold

The model systematically under-predicts severe cases. This is a direct consequence of the class imbalance: 63% of training data is SOFA < 5, so the model is biased toward lower predictions.

**Empirical analysis on the test set:**

| Alert at | High Risk Recall | False Alarm Rate on Low Risk |
|---|---|---|
| Predicted SOFA ≥ 10 | 28.4% | 0.4% |
| **Predicted SOFA ≥ 8** | **52.5%** | **1.4%** |
| Predicted SOFA ≥ 7 | 59.7% | 3.1% |

**Decision: alert at ≥ 8**
- Doubles recall from 28.4% → 52.5% (catches twice as many truly critical patients)
- False alarm rate stays very low (only 1.4% of genuinely stable patients get flagged)
- SOFA=8 is clinically in the moderate-high range — even a "false alarm" at this level warrants attention

The alert at ≥ 8 fires when the true clinical risk level is either "Moderate" (5-9) or "High" (≥10) by SOFA standards. If the prediction is 8.0–9.9 (technically moderate by SOFA definition), the banner includes a note: "(Clinical High Risk threshold is SOFA≥10)".

### High Risk Clinical Advisory Panel

When the alert fires, an expanded advisory panel appears automatically in Tab 1 showing:
- Prediction uncertainty for High Risk patients (typical error ±3.9 SOFA points)
- Explanation of why the alert threshold is ≥ 8 instead of ≥ 10
- Guidance: reassess in 30 minutes, track SOFA trajectory, use LLM as supporting context

---

## 17. Complete App Pipeline — Step by Step

When the "🚀 Run Prediction" button is clicked, the following sequence executes in `app.py`:

### Step 1 — Input Validation
All 7 vital signs are checked against physiologically possible ranges (HR: 30–220, SpO₂: 50–100, etc.). Invalid inputs stop execution and display an error. The Groq API key is checked.

### Step 2 — Sliding Window Update
- Load `models/patient_vitals.csv`
- Append current vitals as a new row with timestamp
- Apply `tail(20)` to keep last 20 rows
- Save updated CSV

### Step 3 — Trend Feature Computation
Compute 9 trend features from the 20-row window: HR_mean, HR_std, RR_mean, SpO2_mean, SpO2_min, Temp_mean, SBP_mean, DBP_mean, MAP_mean

### Step 4 — Latest Feature Collection
Build a dictionary with the 7 latest vitals + GCS eye + stress score from sidebar inputs.

### Step 5 — TF-IDF Transformation
Apply `tfidf.transform([clinical_note])` → sparse matrix → dense DataFrame with 600 columns.

### Step 6 — Feature Combination and Alignment
Concatenate trend_df + latest_df + tfidf_df into a 618-column DataFrame. For any missing columns, fill with 0. Reorder to exactly match `feature_cols` (loaded from `models/feature_columns.pkl`).

### Step 7 — Scaling
Apply `scaler.transform(final_df)` → StandardScaler-normalised values.

### Step 8 — Model Inference
Convert to `torch.FloatTensor`. Run through `ICUModel` in `eval()` mode with `torch.no_grad()`. Get scalar output = raw SOFA prediction. Clip to [0, 24].

### Step 9 — Risk Classification
- SOFA < 5 → Low Risk 🟢
- SOFA 5–9 → Moderate Risk 🟡
- SOFA ≥ 10 → High Risk 🔴 (clinical definition)

### Step 10 — Alert Banner
If SOFA ≥ 8 (ALERT_THRESHOLD): display red `st.error()` banner + auto-expanded High Risk Advisory panel.
If SOFA ≥ 5: yellow `st.warning()` banner.

### Step 11 — Save Prediction History
Append current prediction to `models/prediction_history.csv`: timestamp, SOFA, risk level, HR, RR, SpO₂, Temp, SBP, MAP, GCS, Stress. Keep last 50 rows.

### Step 12 — SHAP Computation
With spinner: `explainer.shap_values(X_tensor)` → 618 SHAP values. Filter to clinically relevant features → top 7 → clinical interpretations + key risk factors.

### Step 13 — Trend Analysis
For each of 7 vitals: compute trend direction (increasing/stable/decreasing from polyfit slope) and status (low/normal/high from normal range comparison). Combine: "SpO₂ → low & decreasing".

### Step 14 — LLM Prompt Construction
Build the risk-adaptive prompt (urgency header for High Risk, standard for others). Include all 7 sections of patient data.

### Step 15 — LLM Self-Consistency (3 Groq calls)
With spinner: call Groq 3 times (temperature=0.7). Compute TF-IDF cosine similarity → consistency score → reliability label.

### Step 16 — Display
Render across 4 tabs with all computed information.

---

## 18. All Files and Their Roles

### `app.py` — The Main Application
The complete Streamlit web application. Contains the entire inference pipeline from user input to LLM output. All display logic. Imports `ICUModel`, `get_trend`, `classify_range` from `model_utils.py`.

### `model_utils.py` — Shared Utilities
Single file imported by app.py, train_federated.py, server.py, and client.py. Contains:
- `ICUModel` — the DNN architecture class
- `ICUDataset` — PyTorch Dataset wrapper
- `weighted_mse_loss` — linear weighted MSE loss function
- `get_sample_weights` — per-sample weights for oversampling (infrastructure only)
- `train_model` — mini-batch training with AdamW
- `evaluate_model` — MAE + R² computation
- `get_weights` / `set_weights` — FL weight serialisation utilities
- `get_trend` — trend direction from polyfit slope
- `classify_range` — normal/low/high classification
- `apply_dp_to_update` — differential privacy: clip + noise
- `estimate_privacy_budget` — approximate ε via Gaussian mechanism

**Why centralised:** Having all shared logic in one file prevents code duplication and ensures client.py and train_federated.py use identical DP logic, identical training functions, and identical model architecture.

### `train_federated.py` — FL Training Script (Simulation)
Runs the full FL training pipeline in simulation mode using Ray. All 3 hospitals run in the same Python process. Produces:
- `models/federated_model.pth` — best global model weights
- `models/shap_background.npy` — 300 background samples
- `models/training_metadata.json` — training config + performance metrics

Key configurable parameters are all at the top of the file.

### `server.py` — Flower FL Server (Real Demo)
Runs a Flower server listening on 127.0.0.1:8080. Waits for exactly 3 clients. Runs 20 FL rounds. Saves model after every round (latest) and whenever a new best eval loss is achieved (best). After training, saves partial metadata to training_metadata.json.

### `client.py` — Flower FL Client (Real Demo)
Takes `--client_id` argument (0, 1, or 2). Loads its private dataset from `data/client_{id}.csv`. Trains locally for 10 epochs per round. Optionally applies Differential Privacy to model updates before sending to server (`USE_DP = True/False`).

### `models/federated_model.pth`
PyTorch `state_dict` containing the trained DNN weights. Keys: `net.0.weight`, `net.0.bias`, `net.2.weight`, `net.2.bias`, `net.4.weight`, `net.4.bias`, `net.6.weight`, `net.6.bias`.

### `models/scaler.pkl`
`sklearn.StandardScaler` fitted on all 618 features from the training data. Cannot be regenerated without the raw BigQuery data. Used to transform inference inputs to the same scale as training data.

### `models/tfidf_vectorizer.pkl`
`sklearn.TfidfVectorizer` fitted on 283,208 MIMIC-III clinical notes. Cannot be regenerated without the raw BigQuery data. Used to transform clinical note text to 600-dimensional vectors.

### `models/feature_columns.pkl`
Python list of 618 column names in the exact training order. Identical to `scaler.feature_names_in_`. Used as the authoritative column list in app.py to ensure feature alignment. This makes `feature_names_in_` independent of sklearn version.

### `models/shap_background.npy`
Numpy array of shape (300, 618) — 300 randomly sampled training rows used as the SHAP reference distribution. Regenerated every time `train_federated.py` is run.

### `models/patient_vitals.csv`
20-row CSV with columns: time, HR, RR, SpO2, Temp, SBP, DBP, MAP. The sliding window history. Updated every time "Run Prediction" is clicked. Reset by the sidebar button.

### `models/prediction_history.csv`
Log of all predictions made: timestamp, SOFA, risk level, HR, RR, SpO₂, Temp, SBP, MAP, GCS Eye, Stress. Keeps last 50 predictions. Displayed as a table in Tab 1.

### `models/training_metadata.json`
JSON file recording the last training configuration and results. Read by app.py to populate Tab 4 (Federated Learning info panel). Written by both `train_federated.py` (full metrics) and `server.py` (partial metrics, preserves existing MAE/R²).

### `data/fl_training/client_0/1/2.csv`
Each CSV contains ~15,889–16,371 rows × 109 columns (108 features + sofa_score).
The features are **already StandardScaler-normalised and clipped to ±10**.
The sofa_score column contains raw SOFA values (0–24).
These represent the private hospital datasets for FL training.
Generated by Phase 0 of `train_federated.py`.

---

## 19. Configuration Reference

### `app.py` Constants

```python
MODEL_PATH    = "models/"
VITALS_FILE   = MODEL_PATH + "patient_vitals.csv"
HISTORY_FILE  = MODEL_PATH + "prediction_history.csv"
ALERT_THRESHOLD = 8.0   # alert fires when predicted SOFA ≥ this value
```

### `train_federated.py` Constants

```python
NUM_ROUNDS       = 20      # FL aggregation rounds
EPOCHS_PER_ROUND = 10      # local training epochs per hospital per round
BATCH_SIZE       = 64      # mini-batch size
SHAP_BG_SAMPLES  = 300     # background samples for SHAP DeepExplainer

BASE_LR          = 0.001   # AdamW learning rate
LR_DECAY         = 1.0     # 1.0 = no decay; <1.0 enables per-round decay

GRAD_CLIP        = None    # gradient clip max norm; None = disabled
OVERSAMPLE       = False   # WeightedRandomSampler; disabled (causes overfitting)
USE_NONIID_SPLIT = False   # biased hospital split; disabled (causes client drift)

USE_DP           = False   # Differential Privacy; True = add noise to weights
DP_SENSITIVITY   = 1.0     # max L2 norm of model update
DP_SIGMA         = 1.0     # Gaussian noise multiplier
```

### `client.py` Constants

```python
EPOCHS       = 10          # local training epochs per round
BATCH_SIZE   = 64
USE_DP       = False       # must match server's intent
DP_SENSITIVITY = 1.0
DP_SIGMA     = 1.0
```

### Normal Ranges Used for Trend Classification

```python
NORMAL_RANGES = {
    "HR":   (60, 100),    # bpm
    "RR":   (12, 20),     # breaths/min
    "SpO2": (95, 100),    # %
    "Temp": (36.5, 37.5), # °C
    "SBP":  (100, 120),   # mmHg
    "DBP":  (60, 80),     # mmHg
    "MAP":  (70, 100),    # mmHg
}
```

---

## 20. Model Performance Summary

### Current Model (federated_model.pth — best run)

| Metric | Value |
|---|---|
| Overall MAE | **1.9608 SOFA points** |
| Overall R² | **0.3357** |
| Prediction range | -0.26 – 13.91 |
| Training samples | 48,150 |
| Test samples | 12,038 |
| FL rounds | 100 (best was round 24) |
| Server optimizer | FedYogi (η=0.01, β1=0.9, β2=0.99, τ=0.001) |
| Client regularisation | FedProx (μ=0.5) |
| Loss | weighted MSE (weight = 1 + SOFA × 0.5) |

### Per-Risk-Level Performance

| Risk Level | n samples | MAE | R² | Notes |
|---|---|---|---|---|
| Low (<5) | 7,546 | 1.555 | -1.093 | Within-segment ranking limited by absent lab values |
| Moderate (5-9) | 3,754 | 2.103 | -2.599 | Between-segment discrimination is strong |
| High (≥10) | 738 | 5.385 | -6.601 | Wide SOFA range (10-24) + sparse training data |

**Note on negative per-class R²:** Global R²=0.3357 is positive because the model
correctly discriminates between risk tiers. Within-segment R² is negative because
ranking patients within the same SOFA tier requires direct lab values (bilirubin,
creatinine, platelets) not present in the feature set.

### Alert System Performance

| Alert threshold | High Risk Recall | False Alarm Rate |
|---|---|---|
| SOFA ≥ 10 | 28.4% | 0.4% |
| **SOFA ≥ 8 (current)** | **52.5%** | **1.4%** |

### Comparison to Previous Versions

| Version | R² | Key Change |
|---|---|---|
| Original FL notebook (10R, 3E, plain MSE) | 0.04 | Baseline |
| After weighted MSE (1+y×3) | 0.11–0.13 | Loss improvement |
| After AdamW + 300 SHAP samples | 0.13–0.18 | Optimizer improvement |
| Piecewise loss + oversampling (failed) | -0.822 | Failed |
| Dropout (failed) | 0.057 / -0.067 | FL incompatible |
| **SOFA vocabulary whitelist (90 terms)** | **0.1752** | **Feature breakthrough** |
| FedAdam server optimizer | 0.2148 | Server optimization |
| FedYogi server optimizer | 0.2229 | Better adaptive rate |
| **All notes (283k) + 10k chars + FedYogi** | **0.3357** | **Current best** |

---

## 21. What Did Not Work and Why

This section is critical for viva preparation — understanding what was tried and why it failed is as important as understanding what works.

### 1. Piecewise Weighted Loss + Oversampling

**What was tried:** Loss weights Low=1×, Moderate=5×, High=20× combined with WeightedRandomSampler (High Risk samples 10× more frequent per batch).

**What happened:** R²=-0.822, loss diverged from 12 → 110 over 20 rounds.

**Root cause:** The 2,904 high-risk samples being repeated 10× caused the model to overfit specifically to those patients. The FedAvg aggregation of 3 hospitals each biased toward high-SOFA predictions produced a global model that predicted high SOFA for all patients — destroying low-risk accuracy while only marginally improving high-risk.

### 2. Dropout in FL

**What was tried:** Dropout p=0.3 after layer 1, p=0.2 after layer 2. Then p=0.1, p=0.05.

**What happened:** Both configurations caused training divergence. R²=0.057 and -0.067 respectively.

**Root cause:** FL + Dropout is fundamentally incompatible with FedAvg. Each hospital trains with a different random Dropout mask. Hospital A zeros neurons {1,5,23...}, Hospital B zeros {3,7,19...}. Their weight updates point in different directions. FedAvg averages these conflicting signals — the global model cannot converge. This is a documented FL research finding. Solutions exist (FedProx, SCAFFOLD) but they require significantly more complex FL algorithms.

**Solution implemented:** AdamW weight decay (`weight_decay=1e-4`) provides L2 regularisation without the FL incompatibility.

### 3. Non-IID Hospital Split

**What was tried:** Biased split where Hospital 0 gets 70% Low SOFA, Hospital 2 gets 70% High SOFA.

**What happened:** R²=0.084 vs R²=0.28 with IID split.

**Root cause:** Client drift. Each hospital's local model optimises for its own patient distribution. Hospital 0's model becomes excellent at predicting Low SOFA, Hospital 2's at High SOFA. FedAvg averaging these conflicting models produces a global model that is mediocre at everything. This is the fundamental non-IID challenge in FL.

### 4. Learning Rate Decay (LR_DECAY=0.97)

**What was tried:** Per-round LR decay of 3% — lr=0.001 in round 1, lr≈0.00056 in round 20.

**What happened:** Overall worse performance. Loss showed more oscillation in late rounds.

**Root cause:** With a low LR in later rounds, the model cannot recover from bad FedAvg aggregation rounds. Fixed lr=0.001 allows the model to continue updating meaningfully even if a particular aggregation round produces a poor global model.

### 5. Gradient Clipping

**What was tried:** Clip=1.0 and Clip=5.0.

**What happened:** Both reduced R² compared to no clipping.

**Root cause:** The weighted MSE loss intentionally produces large gradients for high-SOFA patients. Gradient clipping cuts exactly these large (but informative) gradients, removing the benefit of the weighted loss.

### 6. ClinicalBERT for NLP

**What was tried:** Replace TF-IDF with ClinicalBERT embeddings (768-dimensional per note).

**What happened:** Similar performance to TF-IDF, significantly more computational overhead.

**Decision:** Stayed with TF-IDF. It is simpler, faster, directly interpretable in SHAP, and achieves comparable accuracy for this specific task.

---

## 22. Privacy — Differential Privacy Layer

### What DP Adds

Without DP: each hospital sends its model weights directly. The server could potentially use model inversion attacks to partially reconstruct training data from the weights.

With DP: before sending weights, each hospital:
1. Computes `update = local_weights - global_weights`
2. Clips the update L2 norm to `DP_SENSITIVITY` — this bounds the maximum influence any single patient can have on the transmitted weights (sensitivity clipping)
3. Adds i.i.d. Gaussian noise `N(0, (DP_SIGMA × DP_SENSITIVITY)²)` to every weight parameter
4. Returns `global_weights + clipped_noisy_update`

The server receives noisy, clipped updates. It cannot reconstruct any individual patient's contribution.

### Privacy Budget Estimate

Uses the Gaussian mechanism formula:
```
ε_per_round ≈ sqrt(2 × ln(1.25/δ)) / σ
ε_total = ε_per_round × num_rounds  (simple composition)
```

With σ=1.0 and 20 rounds, ε≈96.9 (δ=1e-5). This is a conservative bound — advanced composition (Rényi DP) gives tighter bounds.

### Current Status

DP is implemented in both `train_federated.py` (simulation) and `client.py` (real FL), with identical logic centralised in `model_utils.py:apply_dp_to_update()`. It is disabled by default (`USE_DP=False`) because enabling it with σ=1.0 would require more training rounds to recover from the noise.

To enable: set `USE_DP=True` in `train_federated.py` (and `client.py` for real FL), then retrain.

---

## Appendix — Key Numbers at a Glance

| Parameter | Value |
|---|---|
| Total features | 108 |
| Trend features | 9 |
| Latest vital features | 7 |
| CV features | 2 |
| TF-IDF features | 90 (SOFA vocabulary whitelist) |
| Sliding window size | 20 readings |
| TF-IDF vocab approach | 90-term SOFA whitelist (not data-driven) |
| Notes sample size | 283,208 (all notes) |
| Notes text limit | 10,000 chars per admission |
| Training samples | 48,150 |
| Test samples | 12,038 |
| Hospital clients | 3 |
| FL rounds | 100 |
| Epochs per round | 3 |
| Best round | 24 |
| Model MAE | 1.9608 SOFA pts |
| Model R² | 0.3357 |
| High Risk recall (threshold ≥ 8) | 52.5% |
| SHAP background samples | 300 |
| LLM calls per prediction | 3 |
| LLM model | openai/gpt-oss-120b |
| LLM temperature | 0.2 |
| LLM max tokens | 2500 |
| Alert threshold | SOFA ≥ 8 |
| Prediction history kept | 50 predictions |
| FedYogi η | 0.01 |
| FedProx μ | 0.5 |

---

*End of Project Context Document.*
*This document was generated after complete implementation of the ICU CDSS capstone project.*

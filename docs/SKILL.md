# COMPLETE PROJECT CONTEXT — ICU Clinical Decision Support System
# Multimodal Intelligence System with Federated Learning

> This document contains the complete, in-depth, and detailed understanding of the
> capstone project. Every design decision, every parameter, every file, every pipeline
> step, and every technical choice is documented here. This is the single source of
> truth for the entire project.

---

## TABLE OF CONTENTS

1. [Project Identity](#1-project-identity)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [What the System Does and Does NOT Do](#4-what-the-system-does-and-does-not-do)
5. [Dataset — MIMIC-III](#5-dataset--mimic-iii)
6. [BigQuery SQL Pipeline — All 12 Queries](#6-bigquery-sql-pipeline--all-12-queries)
7. [Three Data Modalities](#7-three-data-modalities)
8. [Feature Engineering — Complete Detail](#8-feature-engineering--complete-detail)
9. [Sliding Window Mechanism](#9-sliding-window-mechanism)
10. [TF-IDF Clinical NLP Pipeline](#10-tf-idf-clinical-nlp-pipeline)
11. [Target Variable — SOFA Score](#11-target-variable--sofa-score)
12. [Machine Learning Model](#12-machine-learning-model)
13. [Federated Learning Architecture](#13-federated-learning-architecture)
14. [SHAP Explainability](#14-shap-explainability)
15. [LLM Integration — Groq](#15-llm-integration--groq)
16. [LLM Self-Consistency Check](#16-llm-self-consistency-check)
17. [Alert System](#17-alert-system)
18. [ClinIQ React+FastAPI Dashboard — Complete Pipeline](#18-cliniq-reactfastapi-dashboard--complete-pipeline)
19. [Complete Folder Structure](#19-complete-folder-structure)
20. [File-by-File Description](#20-file-by-file-description)
21. [All Finalized Technical Parameters](#21-all-finalized-technical-parameters)
22. [Model Performance Metrics](#22-model-performance-metrics)
23. [Technology Stack](#23-technology-stack)
24. [How to Run the Project](#24-how-to-run-the-project)
25. [Project Evolution History](#25-project-evolution-history)
26. [Novel Contributions](#26-novel-contributions)
27. [Design Constraints and Assumptions](#27-design-constraints-and-assumptions)
28. [Viva Preparation — Key Points](#28-viva-preparation--key-points)

---

## 1. PROJECT IDENTITY

**Full Title:**
Multimodal Intelligence System with Federated Learning for Continuous Patient
Monitoring and Early Deterioration Detection Leveraging IoMT, Computer Vision
and Clinical NLP

**Type:** Final Year Engineering Capstone Project

**Domain:** Healthcare AI / Clinical Decision Support / Federated Learning

**Nature:** This is a Clinical Decision Support System (CDSS). It is NOT a
diagnostic system. It assists clinicians — it does not replace them.

**Student:** Sneh Patel, Final Year Computer Science Engineering

---

## 2. PROBLEM STATEMENT

ICU patients deteriorate rapidly. The existing monitoring infrastructure has
three critical failures:

**Failure 1 — Threshold-based alarms cause alarm fatigue.**
Current ICU systems fire an alarm whenever a single vital crosses a fixed
threshold (e.g., HR > 120). This produces hundreds of false alarms per day.
Clinical staff begins to ignore alarms. Genuine emergencies are missed.

**Failure 2 — No multimodal integration.**
Existing systems look at one vital at a time, in isolation. A heart rate of
105 bpm is clinically meaningless without context: What is the trend? What are
the blood pressure and oxygen saturation doing simultaneously? What does the
clinical note say? No existing bedside system combines all of these together
into a unified risk signal.

**Failure 3 — Patient data cannot leave hospitals.**
Traditional machine learning requires centralizing data in one place for
training. In healthcare, patient data is legally protected (HIPAA, GDPR).
Hospitals cannot share patient records with each other or with a central server.
This makes centralized collaborative ML ethically and legally impossible.

**Our system solves all three:**
- Multimodal fusion replaces threshold-based alarms
- Trend analysis + NLP + CV features give full patient context
- Federated Learning enables collaborative training without data sharing

---

## 3. PROJECT OBJECTIVES

1. Continuously monitor ICU patients using multimodal data streams
2. Detect early physiological deterioration before it becomes severe
3. Predict the patient's SOFA score (Sequential Organ Failure Assessment, 0–24)
4. Classify risk as Low, Moderate, or High
5. Generate an alert for high-risk patients
6. Provide explainable AI output using SHAP (feature-level reasoning)
7. Generate clinician-friendly natural language explanation using a pre-trained LLM
8. Validate LLM output reliability using a self-consistency check
9. Preserve patient privacy across hospitals using Federated Learning

---

## 4. WHAT THE SYSTEM DOES AND DOES NOT DO

### What it DOES:
- Predicts deterioration risk as a SOFA score
- Classifies severity (Low / Moderate / High)
- Shows which features are driving the prediction (SHAP)
- Generates a natural language clinical assessment (LLM)
- Trains collaboratively across hospitals without sharing patient data (FL)
- Maintains a sliding window of recent vitals for trend analysis

### What it does NOT do:
- Does NOT diagnose any disease
- Does NOT make clinical decisions autonomously
- Does NOT replace physician or nurse judgment
- Does NOT prescribe medications or treatments
- Does NOT access real-time bedside devices (demo uses manual input)

---

## 5. DATASET — MIMIC-III

**Dataset Name:** MIMIC-III (Medical Information Mart for Intensive Care III)

**Access Method:** Google BigQuery (project: `mimic-project-2`, dataset: `dataset_v2` — rebuilt from scratch, see `docs/SQL_COMMANDS.md`; the old dataset `Dataset` was deleted)

**Why MIMIC-III and not MIMIC-IV:**
MIMIC-III is more stable for preprocessing and experimentation. MIMIC-III has
well-documented derived tables including pre-computed SOFA scores in
`mimiciii_derived.sofa`.

**Key BigQuery Tables Used:**

| Table | Purpose |
|---|---|
| `physionet-data.mimiciii_clinical.chartevents` | Vital signs, GCS, Stress Score |
| `physionet-data.mimiciii_notes.noteevents` | Clinical notes (nursing, physician) |
| `physionet-data.mimiciii_derived.pivoted_sofa` | Hourly SOFA — `SOFA_24hours` at the matching hour is the target label |

**dataset_v2 tables:** `cohort`, `vitals_clean`, `vitals_hourly`, `vital_trends`, `latest_vitals`, `gcs_hourly`, `stress_hourly`, `labels`; VIEWs `ml_dataset_final` (5,089,857 hourly rows) and `clinical_notes` (35,799 admissions).

**Final Processed View:**
`mimic-project-2.dataset_v2.ml_dataset_final`

**Dataset Size (current):**
- Cohort: adults (age ≥ 16), ICU stay ≥ 6 h → 52,248 ICU stays / 38,304 patients
- Label: `SOFA_24hours` at the matching hour (from `pivoted_sofa`), not a stay-level value
- Phase 0 takes the newest row per ICU stay → 52,248 rows × 109 columns (108 features + sofa_score)
- Train/test split: 80/20 stratified by High Risk (SOFA ≥ 10) → 41,798 train, 10,450 test
- Test set: 602 High, 2,919 Moderate, 6,929 Low
- Federated split: 3 IID hospitals of 13,932 / 13,933 / 13,933 rows (each Low 67%, Moderate 27%, High 6%)
- Missing values filled with 0 before scaling; StandardScaler clipped to ±10

---

## 6. BIGQUERY SQL PIPELINE — ALL 12 QUERIES

> **Historical:** the 12 queries below describe the original `Dataset` build. The
> current data is `dataset_v2`, rebuilt with the commands in `docs/SQL_COMMANDS.md`
> (hourly vitals on the SOFA timeline, time-matched SOFA label). The 108 input
> columns are unchanged.

All 12 queries were executed sequentially in Google BigQuery to produce the
final ML-ready dataset.

**Query 1 — Extract raw vitals from CHARTEVENTS:**
Extracts Heart Rate, RR, SpO2, Temperature, SBP, DBP, MAP from CHARTEVENTS.
Fahrenheit temperatures (itemid=223761) are converted to Celsius using
formula: (value - 32) × 5/9
MIMIC item IDs used:
- HR: 211, 220045
- RR: 618, 220210
- SpO2: 646, 220277
- Temp: 223762, 678, 223761 (223761 = Fahrenheit, converted)
- SBP: 51, 220179
- DBP: 8368, 220180
- MAP: 52, 220181

Output table: `vitals_raw`

**Query 2 — Pivot long format to wide format:**
Groups by (subject_id, hadm_id, icustay_id, charttime)
Uses MAX(CASE WHEN itemid IN (...) THEN value END) to create one row per
timestamp with separate columns for each vital sign.

Output table: `vitals_pivoted`

**Query 3 — Aggregate into 10-minute windows:**
Uses TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(charttime), 600) * 600) to create
10-minute time buckets. Computes AVG for each vital within each 10-min window.

Output table: `vitals_10min`

**Query 4 — Compute sliding window trend features:**
Uses SQL window function with ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
(20 rows per window, partitioned by icustay_id).
Computes: HR_mean, HR_std, RR_mean, SpO2_mean, SpO2_min, Temp_mean,
SBP_mean, DBP_mean, MAP_mean

Output table: `vital_trends_20`

**Query 5 — Extract latest vital values:**
From `vitals_10min`, extracts the raw (latest) vital values and renames them:
HR → latest_HR, RR → latest_RR, SpO2 → latest_SpO2, Temp → latest_Temp,
SBP → latest_SBP, DBP → latest_DBP, MAP → latest_MAP

Output table: `latest_vitals`

**Query 6 — Extract clinical notes:**
From NOTEEVENTS, keeps only: 'Nursing', 'Physician', 'Nursing/Other',
'Discharge summary'. Filters out null text entries.

Output table: `clinical_notes`

**Query 7 — Extract SOFA scores:**
From `physionet-data.mimiciii_derived.sofa`, extracts pre-computed SOFA scores.

Output table: `sofa_scores`

**Query 8 — Join trends + latest vitals + SOFA:**
Creates the base ML dataset with all physiological features + target label.

Output table: `ml_dataset`

**Query 9 — Extract GCS Eye Opening:**
From CHARTEVENTS, extracts GCS Eye Opening scores (itemid=220739).

Output table: `gcs_eye`

**Query 10 — Extract Stress Score (originally Pain Score):**
From CHARTEVENTS, extracts Pain/Stress Score (itemid=223791).
RENAMED to `stress_score` in the BigQuery database. All saved artifacts use
`stress_score` as the column name.

Output table: `pain_score` (internal; column is stress_score)

**Query 11 — Combine CV features into 10-min windows:**
UNIONs GCS Eye and Stress Score tables, aggregates into 10-min windows.

Output table: `cv_features_10min`

**Query 12 — Create final ML dataset:**
LEFT JOINs `ml_dataset` with `cv_features_10min` on (icustay_id, window_time).
Adds GCS_eye_opening and stress_score to the feature set.

Output table: `ml_dataset_final` (THE FINAL DATASET)

---

## 7. THREE DATA MODALITIES

### Modality 1 — IoMT Physiological Vital Signs (Primary Modality)

7 vital signs: HR, RR, SpO2, Temperature, SBP, DBP, MAP
Valid physiological ranges: HR 30–220, RR 5–60, SpO2 50–100, Temp 30–43°C,
SBP 40–250, DBP 20–150, MAP 30–200

### Modality 2 — Computer Vision Features (Behavioral/Neurological)

2 CV features:
- **GCS Eye Opening** — Glasgow Coma Scale eye response. Range: 1–4
- **Stress Score** — Behavioral/physiological stress. Range: 0–10

Conceptual pipeline: in a real deployment, a bedside camera feeds a CV model
that estimates these. In the current project they are entered manually.

### Modality 3 — Clinical Notes (NLP)

Free-text nursing/physician notes converted to 90 SOFA-vocabulary TF-IDF features.
All 6 SOFA organ components are represented by specific whitelist terms.

---

## 8. FEATURE ENGINEERING — COMPLETE DETAIL

The final ML model receives **108 features** total, in this exact order:

### Group 1 — Trend Features (9 features)
Computed from the sliding window of last 20 vital sign readings.

| Feature | How Computed |
|---|---|
| HR_mean | Average HR over last 20 readings |
| HR_std | Standard deviation of HR over last 20 readings |
| RR_mean | Average RR over last 20 readings |
| SpO2_mean | Average SpO2 over last 20 readings |
| SpO2_min | Minimum SpO2 over last 20 readings (catches dips) |
| Temp_mean | Average Temperature over last 20 readings |
| SBP_mean | Average SBP over last 20 readings |
| DBP_mean | Average DBP over last 20 readings |
| MAP_mean | Average MAP over last 20 readings |

### Group 2 — Latest Vital Values (7 features)

latest_HR, latest_RR, latest_SpO2, latest_Temp, latest_SBP, latest_DBP, latest_MAP

### Group 3 — Computer Vision Features (2 features)

GCS_eye_opening (1–4), stress_score (0–10)

### Group 4 — TF-IDF SOFA Vocabulary Features (90 features)

**90-term SOFA vocabulary whitelist** — explicit, curated terms that directly
correspond to the 6 SOFA organ components. This replaced the original
data-driven 600-feature approach, which selected documentation-style terms
("tablet", "sig", "refills") that correlated with volume of charting rather
than patient severity.

The whitelist guarantees every text feature has a direct clinical meaning:

| SOFA Component | Example vocabulary terms |
|---|---|
| Respiratory | intubated, ventilator, bipap, cpap, hypoxia, respiratory failure, extubated |
| Coagulation | plt, platelets, coagulopathy, inr, ptt, hemorrhage, thrombocytopenia |
| Hepatic | bilirubin, bili, totbili, liver, jaundice, hepatic, cirrhosis, liver failure |
| Cardiovascular | vasopressor, pressors, levophed, norepinephrine, septic shock, hypotension |
| CNS | sedated, sedation, coma, altered, confused, gcs, encephalopathy, delirium |
| Renal | creatinine, creat, renal, kidney, dialysis, aki, oliguria, urine output |
| Sepsis/Infection | sepsis, septic, bacteremia, infection, cultures, antibiotics, fever |
| Lab values | lactate, wbc, hgb, hct, sodium, potassium, hco, angap, bun |
| Severity | failure, acute, pneumonia, ards, pulmonary, dyspnea, edema |

**Total: 9 + 7 + 2 + 90 = 108 features**

### Feature Scaling
All 108 features are scaled using StandardScaler (zero mean, unit variance)
fitted on the training dataset, then clipped to [-10, 10].
Saved as `models/scaler.pkl`.

### Feature Order
The exact order of all 108 features is critical. Saved in `models/feature_columns.pkl`.
At inference time, the feature vector is reordered to match this exact order.

---

## 9. SLIDING WINDOW MECHANISM

**Window Size:** 20 readings (fixed)

**Mechanics:**
1. Patient vitals CSV stores the last 20 readings
2. New vitals appended → oldest row dropped (tail(20))
3. Trend features recomputed from updated window
4. Window saved back

**Trend Direction:** numpy.polyfit slope — >0.1 "increasing", <-0.1 "decreasing", else "stable"

**Normal Ranges:**

| Vital | Normal Low | Normal High |
|---|---|---|
| HR | 60 | 100 |
| RR | 12 | 20 |
| SpO2 | 95 | 100 |
| Temperature | 36.5 | 37.5 |
| SBP | 100 | 120 |
| DBP | 60 | 80 |
| MAP | 70 | 100 |

---

## 10. TF-IDF CLINICAL NLP PIPELINE

**Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
**Approach:** Explicit 90-term SOFA vocabulary whitelist (NOT data-driven)

**Why whitelist instead of data-driven:**
Data-driven TF-IDF (max_features=600) selects the 600 most statistically variable
terms from the corpus. In MIMIC-III, these include documentation-style terms:
"tablet", "sig" (Latin for "take as directed"), "refills", "disp" (dispense),
"capsule". These correlate with volume of documentation (sicker patients get more
notes), not patient severity — creating spurious correlations.

The whitelist guarantees every feature has direct clinical meaning tied to a
specific SOFA organ component.

**Vectorizer Configuration:**
```python
TfidfVectorizer(
    vocabulary=list(_SOFA_VOCABULARY),   # 90-term SOFA whitelist
    ngram_range=(1, 2),                  # unigrams + bigrams
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b',
    sublinear_tf=True,    # log(1+tf) — grades severity smoothly
    norm="l2",            # L2-normalise per document
)
```

When `vocabulary=` is set, sklearn ignores `max_features`, `min_df`, `max_df`,
and `stop_words`.

**Corpus used:**
- NOTES_SAMPLE_SIZE = 283,208 (all available notes — no sampling)
- NOTES_TEXT_LIMIT = 10,000 characters per admission (captures full progress notes
  including lab value mentions: "creatinine 2.8", "bilirubin elevated at 4.5")
- Notes grouped by `hadm_id`, cleaned (remove MIMIC de-identification patterns,
  numbers, punctuation), then truncated

**At Inference:**
`tfidf.transform([clinical_note])` using the saved vectorizer. Same 90 vocabulary
terms always. Words not in the vocabulary are ignored.

**Handling Empty Notes:**
"No clinical notes provided" → valid but sparse vector (all near-zero).

---

## 11. TARGET VARIABLE — SOFA SCORE

**What is SOFA:**
Sequential Organ Failure Assessment — an internationally standardized ICU
severity scoring system used worldwide by clinicians.

**SOFA Range:** 0 to 24

**How SOFA is calculated (6 organ systems, each 0–4):**
1. Respiratory (PaO2/FiO2 ratio)
2. Coagulation (Platelet count)
3. Liver (Bilirubin)
4. Cardiovascular (MAP and vasopressors)
5. Central Nervous System (GCS score)
6. Renal (Creatinine and urine output)

**Source in MIMIC-III:** `SOFA_24hours` from `physionet-data.mimiciii_derived.pivoted_sofa`, at the matching hour (dataset_v2). (Earlier build: `mimiciii_derived.sofa`.)

**Risk Classification Thresholds (finalized):**

| SOFA Score | Risk Level |
|---|---|
| 0 – 4 | 🟢 Low Risk |
| 5 – 9 | 🟡 Moderate Risk |
| 10 – 24 | 🔴 High Risk |

**Alert threshold:** predicted SOFA ≥ 7 (`ALERT_THRESHOLD = 7.0`; lower than clinical ≥ 10 to compensate for under-prediction; was ≥ 8 in an earlier version)

**Model Output:**
The model predicts raw SOFA directly (0–24). No normalisation. Clipped to [0, 24]
using np.clip at inference.

---

## 12. MACHINE LEARNING MODEL

### Architecture — PyTorch Fully Connected DNN

```
Input: 108 features (scaled)
  ↓
Linear(108 → 256) + ReLU + Dropout(0.3)
  ↓
Linear(256 → 128) + ReLU + Dropout(0.3)
  ↓
Linear(128 → 64) + ReLU + Dropout(0.3)
  ↓
Linear(64 → 1)
  ↓
Output: SOFA score (0–24)
```

Class definition: `ICUModel` in `model_utils.py` (defaults `hidden=(256, 128, 64)`, `dropout=0.3`)
Total parameters: 69,121
Saved weights: `models/federated_model.pth`
Dropout is only active during training; evaluation, SHAP and the backend use `model.eval()`.

**Why this architecture (108→256→128→64→1 + Dropout):**
On dataset_v2 with plain MSE and FedAvg, the earlier 108→128→64→32→1 network
(no Dropout, 24,321 parameters — historical) peaked early and then overfit. Dropout 0.3
stops the memorisation and the extra width gives room to use it: MAE 1.850 → 1.825,
R² 0.457 → 0.473, AUROC 0.921 → 0.925 on the same test set. 512-256-128 gave no
further gain. (An even older 618→256→128→64→1 model, from before the SOFA vocabulary
whitelist, had 199,681 parameters.)

### Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| FL Rounds | 100 | Best round kept by SaveBestStrategy — round 26 in the current run |
| Epochs/round | 3 | Sweet spot — less drift than 5, more signal than 1 |
| Batch size | 64 | Mini-batch gradient updates |
| Base LR | 0.001 | AdamW starting learning rate |
| LR decay | 0.99 | Per-round multiplier (floor 1e-4) |
| Grad clip | 1.0 | L2 norm clip |
| Loss | Plain MSE | `weighted_loss=True` restores the old 1 + SOFA × 0.5 weighting |
| Dropout | 0.3 | After each hidden layer, training only |
| Server aggregation | FedAvg | `USE_FEDYOGI = False` (FedYogi was used earlier) |
| Oversampling / DP | Off / Off | `OVERSAMPLE = False`, `USE_DP = False` |
| Optimizer | AdamW | weight_decay=1e-4 |
| FedProx μ | 0.5 | Client-side proximal regularisation |

**Loss function (current: plain MSE):**
```python
loss = mean((pred - target)²)
```
Historical: earlier versions used a linear-weighted MSE (`weight = 1 + target × 0.5`,
normalised by the batch mean). On dataset_v2 it pushed every prediction upward and raised
MAE without catching more high-risk patients, so it was replaced by plain MSE. It is
still available as `train_model(..., weighted_loss=True)`.

**Why PyTorch DNN and not LightGBM/XGBoost:**
FedAvg works by averaging model weight matrices. Tree-based models do not have weight
matrices — each tree has a different structure per client. You cannot average trees.
Neural networks are the natural choice for FedAvg-based FL.

---

## 13. FEDERATED LEARNING ARCHITECTURE

### Concept
Multiple hospitals each have private patient data. Federated Learning allows
collaborative training without any raw patient data leaving any hospital.

### What IS Federated in This Project:
- The FL communication protocol is REAL (Flower server/client, TCP sockets)
- Weights only are transmitted — no patient data
- Each client trains on its own separate dataset
- The server aggregates the hospitals' weights with FedAvg (sample-size weighted average)

### What IS Simulated:
- Hospital data is artificially split from one MIMIC-III dataset
- All three clients run on the same machine (127.0.0.1)

### Framework: Flower (flwr)

### Server Aggregation: FedAvg (current)

```
w_global ← Σ n_i · w_i / Σ n_i      (sample-size weighted average of the 3 hospitals)
```

`USE_FEDYOGI = False` in `train_federated.py`. **Historical:** an earlier version applied
FedYogi on the server after averaging (η = 0.01):

```
Δ = FedAvg(local_models) − w_global         (pseudo-gradient)
m_t = β1·m_{t-1} + (1-β1)·Δ                (momentum)
v_t = v_{t-1} + (1-β2)·(Δ²-v_{t-1})·sign(Δ²-v_{t-1})  (Yogi adaptive)
w_global ← w_global + η·m_t / (√v_t + τ)   (server update)
```

On dataset_v2, FedYogi with η = 0.01 made the server loss explode in early rounds, and
FedAvg + plain MSE did better (MAE 1.914 → 1.850, R² 0.426 → 0.457). Setting
`USE_FEDYOGI = True` re-enables FedYogi, now with `SERVER_ETA = 0.001`.

### Client Regularisation: FedProx

Each hospital's local loss includes a proximal term:
```
L_local = L_data + (μ/2) × ||w_local - w_global||²
```
With μ=0.5, the proximal term prevents hospitals from drifting too far from the
global model during local training.

### FL Architecture:

```
Global Server (server.py)
    │
    │ sends global weights
    ▼
┌──────────────────────────────────────────────┐
│  Hospital 0       Hospital 1      Hospital 2 │
│  (client_0.csv)   (client_1.csv)  (client_2.csv)│
│  13,932 rows      13,933 rows     13,933 rows  │
│  Local Train      Local Train     Local Train  │
│  3 epochs + FedProx                            │
└──────────────────────────────────────────────┘
    │               │               │
    └───────────────┴───────────────┘
                    │
                    ▼
            FedAvg Aggregation
                    │
                    ▼
          Updated Global Model (w_global)
```

**FL Rounds:** 100
**Clients per round:** 3 (fraction_fit=1.0)
**Aggregation:** FedAvg (Flower `FedAvg` strategy; `FedYogi` only if `USE_FEDYOGI = True`)
**Server address:** 127.0.0.1:8080

### Two FL Modes:

**Mode 1 — Simulation (train_federated.py):**
`fl.simulation.start_simulation` — all in same Python process, uses Ray.
Run: `python train_federated.py`

**Mode 2 — Real FL Demo (server.py + client.py):**
True separate processes via TCP socket.
Run: `python server.py` then `python client.py --client_id 0/1/2`

### SaveBestStrategy:
Custom Flower strategy extending FedAvg (FedYogi in an earlier version). Tracks the lowest evaluation loss
across all rounds. Saves that round's model. Passes `server_round` to clients
for per-round LR scheduling.

---

## 14. SHAP EXPLAINABILITY

### Method: shap.DeepExplainer
Used because the model is a PyTorch neural network.

### Background Data:
300 randomly sampled training rows (shape: 300 × 108).
Saved as `models/shap_background.npy`.
Created at startup: `explainer = shap.DeepExplainer(model, background_tensor)`

### SHAP Inference:
`shap_values = explainer.shap_values(X_tensor)` → shape (108,)
Positive value = feature pushed SOFA prediction UP (worse outcome)
Negative value = feature pushed SOFA prediction DOWN (better/neutral)

### Clinical Feature Filtering:
From 108 SHAP values, filter to clinically relevant features via keyword matching:
"HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP", "GCS", "stress",
"hypotension", "respiratory", "septic", "failure", "intubated", "vasopressor",
"shock", "fever", "infection", "oxygen", "ventilat", "cardiac", "renal", "hepatic"

### Top 7 Features:
Top 7 by absolute SHAP impact from filtered features.

### SHAP Display:
Shows original clinical values (not z-scores). For TF-IDF features, shows
"detected" or "absent" depending on whether the term appeared in the note.

---

## 15. LLM INTEGRATION — GROQ

### Provider: Groq
### Main Model: openai/gpt-oss-120b — temperature 0.2, max_tokens 3500
### Judge Model: openai/gpt-oss-20b — temperature 0.2, max_tokens 300, 12s threading timeout
### Cache: 90s in-process dict keyed by care_mode|sofa|hr|spo2

**Why Groq:** Extremely fast inference compared to other providers. Near-real-time
response for ICU decision support.

**System Prompt:**
"You are an expert ICU clinical decision support assistant. Provide concise,
structured, and clinically accurate reasoning."

### LLM Prompt Structure (7 sections):
1. SOFA Score and Risk Level
2. Clinical Notes
3. Latest Vital Signs with normal ranges
4. Neurological & Stress Indicators
5. Key Risk Factors (SHAP-identified)
6. Feature-Level Explanations (SHAP-derived)
7. Vital Sign Trends

### LLM Output Structure (4 sections):
1. Current Condition
2. Probable Cause
3. Risk Forecast
4. Immediate Actions

### Risk-Adaptive Prompting:
For alert patients (predicted SOFA ≥ 7 = `ALERT_THRESHOLD`, in hospital and home-care modes): "IMMEDIATE ACTIONS" section is placed FIRST,
with a 30-minute time horizon. For non-alert patients: standard order, 2–4 hour
horizon.

### `<think>` Block Stripping:
The app strips any `<think>...</think>` blocks that some LLM models produce as
internal reasoning, so only the final structured output is shown.

---

## 16. LLM SELF-CONSISTENCY CHECK

**3 components, weighted average:**

| Component | Weight | What it measures |
|---|---|---|
| TF-IDF cosine similarity | 20% | Word-level phrasing overlap between 3 responses |
| Intervention agreement | 50% | Do all 3 agree on which treatments to recommend? |
| Condition agreement | 30% | Do all 3 identify the same clinical conditions? |

**Combined score = 0.20 × tfidf + 0.50 × interventions + 0.30 × conditions**

**Reliability Labels:**

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.80 | ✅ High Reliability | Strong clinical consensus |
| 0.60–0.80 | ⚠️ Moderate Reliability | Some variation — review carefully |
| < 0.60 | ❌ Low Reliability | Significant variation — extra clinical judgment |

**Display in App:**
- Response 1 shown prominently
- Consistency score and reliability label as metrics
- All 3 responses available in expandable section
- Clinical disclaimer mandatory

---

## 17. ALERT SYSTEM

**Alert Trigger:** predicted SOFA ≥ 7 (`ALERT_THRESHOLD = 7.0`, defined once in `model_utils.py`; `backend/services/ml_service.py` imports it and `compute_sofa_risk_metrics()` defaults to it. It was 8.0 in `ml_service.py` in an earlier version.)
**Moderate Warning:** predicted SOFA ≥ 5

**Why threshold 7.0 instead of clinical ≥ 10:**
The model under-predicts severe cases (mean prediction for truly High-Risk patients ≈ 9.7),
because most of SOFA comes from labs and vasopressor doses that are not inputs.

Sweep on the current model (test set, 602 truly High-Risk stays):
- ≥ 8: recall 61.1%, specificity 97.8%
- ≥ 7.5: recall 64.3%, specificity 97.1%
- **≥ 7 (current): recall 67.1%, specificity 96.3%**
- ≥ 6.5: recall 69.9%, specificity 95.3%
- ≥ 6: recall 73.1%, specificity 93.9%

Lowering 8 → 7 means 36 fewer missed high-risk patients (234 → 198) for 140 more false
alarms (220 → 360). AUROC and MAE are unaffected.

**What the alert drives:** red alert banner, `is_alert` flag, and the urgent
("immediate actions first") LLM report format in hospital and home-care modes.

**Display (ClinIQ React dashboard):**
- High Risk → red AlertBanner at top of page + red SOFA gauge
- Moderate Risk → amber AlertBanner at top of page + amber SOFA gauge
- Low Risk → no banner, green SOFA gauge

---

## 18. ClinIQ React+FastAPI Dashboard — COMPLETE PIPELINE

### Backend Startup (lru_cache on first request):
1. Load `models/scaler.pkl` → StandardScaler (108 features)
2. Load `models/tfidf_vectorizer.pkl` → TF-IDF vectorizer (90-term SOFA vocab)
3. Load `ICUModel(108)`, load weights from `models/federated_model.pth`
4. Set model to eval() mode
5. Load `models/shap_background.npy` → numpy array (300 × 108)
6. Create `shap.DeepExplainer(model, background_tensor)`
7. Seed patient vitals in SQLite if first run

### On POST /api/predict (90-second auto-refresh from frontend):

**Step 1: Read DB Row** — `get_sample_patient_row(patient_id, row_idx % total_rows)`.

**Step 2: Sliding Window Update** — append vitals to SQLite + keep last 20 rows.

**Step 3: Trend Features** — compute 9 statistics from 20-row window.

**Step 4: Latest Features** — package 7 latest vitals + GCS + stress from DB row.

**Step 5: TF-IDF Transform** — `tfidf.transform([note])` → 90 columns.

**Step 6: Feature Combination** — concatenate 9+7+2+90 = **108 columns**.

**Step 7: Feature Alignment** — reorder to match feature_columns.pkl.

**Step 8: Scaling** — `np.clip(scaler.transform(final_df), -10, 10)`.

**Step 9: Model Inference** — `ICUModel(108→256→128→64→1)` in eval() (Dropout off) with no_grad.

**Step 10: Risk Classification** — SOFA < 5 / 5-9 / ≥ 10.

**Step 11: Alert Flag** — `is_alert = sofa >= ALERT_THRESHOLD` (7.0, imported from model_utils.py).

**Step 12: Save Prediction** — append to prediction_history in SQLite.

**Step 13: SHAP Computation** — `explainer.shap_values(X_tensor)` → 108 values → top 5.

**Step 14: Trend Analysis** — `get_trend()` + `classify_range()` for 7 vitals.

**Step 15: Return JSON** — sofa_score, risk_level, vitals, shap_top5, key_risks, etc.

### On POST /api/report (triggered by frontend when predict data changes):

**Step 1: Cache Check** — 90s in-process cache keyed by `care_mode|sofa|hr|spo2`. Return cached if hit.

**Step 2: Build Prompt** — 7-section risk-adaptive prompt (IMMEDIATE ACTIONS first for High Risk).

**Step 3: LLM Call** — Groq `openai/gpt-oss-120b`, temp=0.2, max_tokens=3500. Strip `<think>` blocks.

**Step 4: Parse Sections** — split into 4 section dict (CURRENT CONDITION / PROBABLE CAUSE / RISK FORECAST / IMMEDIATE ACTIONS).

**Step 5: Judge Call** — Groq `openai/gpt-oss-20b`, 12s timeout, 300 tokens. Reliability score.

**Step 6: Return JSON** — sections, reliability_score, reliability_label, care_mode.

### Frontend Polling Loop (useMonitor.js):

- 90-second countdown timer; freezes at 1 during in-flight fetch
- `fetchingRef` guard — no overlapping fetch calls
- `setCountdown(90)` reset only in `finally` block after data arrives
- `reportKey()` + `lastKeyRef`/`lastCareModeRef` guard — LLM fires only on genuine clinical value change or care mode switch

### Display (single-page, no tabs):

**Section 1 — Patient Overview:** SOFA score (large, risk-colored) + 6 vital cards (color-coded normal/abnormal)

**Section 2 — Vital Trends:** 6 Recharts line charts (last 20 readings) with normal range bands

**Section 3 — AI Clinical Report:** Hospital / Home Care toggle → 4 LLM section cards → SHAP bars (anime.js springs) → reliability badge → clinical disclaimer

---

## 19. COMPLETE FOLDER STRUCTURE

```
icu_monitor/
│
├── model_utils.py             ← Shared ML utilities (ICUModel, get_trend, etc.)
├── train_federated.py         ← FL training simulation (Flower + Ray)
├── server.py                  ← Real Flower FL server (TCP, 127.0.0.1:8080)
├── client.py                  ← Real Flower FL client (--client_id 0/1/2)
├── db.py                      ← SQLite DB layer (all reads/writes)
├── migrate_to_db.py           ← Migrates CSVs → icu_monitor.db
├── cv_monitor.py              ← Computer Vision Monitor (Stress + GCS, standalone)
├── requirements.txt
├── .env                       ← NEVER committed (contains GROQ_API_KEY)
├── .env.example
├── README.md
│
├── backend/                   ← FastAPI backend (port 8000)
│   ├── main.py                ← App entry point, CORS, router registration
│   ├── api/
│   │   ├── patients.py        ← GET /api/patients
│   │   ├── predict.py         ← POST /api/predict
│   │   ├── report.py          ← POST /api/report
│   │   └── history.py         ← GET /api/history/{patient_id}
│   └── services/
│       ├── ml_service.py      ← Full SOFA inference pipeline
│       └── llm_service.py     ← Groq LLM calls + 90s cache + judge
│
├── frontend/                  ← React 19 + Vite 8 frontend (port 5174)
│   ├── package.json
│   ├── vite.config.js         ← Proxy /api → http://localhost:8000
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css          ← Design tokens (glassmorphism, Figtree font)
│       ├── api/client.js      ← Axios instance + all API functions
│       ├── hooks/useMonitor.js← 90s polling loop with fetchingRef guard
│       ├── components/
│       │   ├── layout/        ← HeaderBar, AlertBanner
│       │   ├── overview/      ← SOFAGauge, VitalCard, VitalsGrid
│       │   ├── trends/        ← TrendChart, TrendsSection
│       │   └── report/        ← ReportContainer, ReportSection, SHAPPanel,
│       │                         ReportToggle, ReliabilityBadge
│       └── pages/
│           ├── SelectPatient.jsx
│           └── MonitorDashboard.jsx
│
├── docs/
│   ├── CV_MONITOR.md
│   ├── TRAIN_FEDERATED.md
│   ├── Project_Context.md
│   ├── REDESIGN_PLAN.md
│   ├── MODEL_DISCUSSION.md
│   ├── VALIDATION.md
│   ├── INPUT_OUTPUT.md
│   └── SKILL.md
│
├── models/
│   ├── federated_model.pth    ← DNN weights (108→256→128→64→1, 69,121 params)
│   ├── scaler.pkl             ← StandardScaler (108 features, ~8 KB)
│   ├── tfidf_vectorizer.pkl   ← TF-IDF (90-term SOFA vocab, ~2 MB)
│   ├── feature_columns.pkl    ← Ordered list of 108 feature names
│   ├── shap_background.npy    ← 300 background samples (300 × 108)
│   ├── icu_monitor.db         ← SQLite DB (~23 MB, gitignored)
│   ├── patient_vitals/        ← Legacy CSV dir (now in DB, .gitkeep only)
│   ├── prediction_history/    ← Legacy CSV dir (now in DB, .gitkeep only)
│   └── training_metadata.json
│
└── data/
    └── fl_training/
        ├── client_0.csv       ← Hospital 0: 13,932 × 109 cols (gitignored)
        ├── client_1.csv       ← Hospital 1: 13,933 × 109 cols (gitignored)
        └── client_2.csv       ← Hospital 2: 13,933 × 109 cols (gitignored)
```

---

## 20. FILE-BY-FILE DESCRIPTION

### `backend/main.py`
FastAPI application entry point. CORS configured for ports 5173/5174/5175.
Registers routers: patients, predict, report, history.

### `backend/api/predict.py`
POST /api/predict — reads DB row, runs full SOFA pipeline, saves vitals + prediction, returns JSON.

### `backend/api/report.py`
POST /api/report — calls Groq LLM, runs judge reliability check, returns 4-section report.

### `backend/api/patients.py`
GET /api/patients — returns patient config (name, description, risk, color).

### `backend/api/history.py`
GET /api/history/{patient_id} — returns last N predictions from DB.

### `backend/services/ml_service.py`
ML inference pipeline. Loads artifacts with lru_cache. Runs: sliding window → trend features
→ TF-IDF → scale → DNN → SHAP. Wraps everything in run_prediction(patient_id, row_idx).

### `backend/services/llm_service.py`
Groq LLM calls (openai/gpt-oss-120b, main; openai/gpt-oss-20b, judge).
90s in-process cache keyed by care_mode|sofa|hr|spo2. Strips <think> blocks.
Returns 4-section dict + reliability score.

### `frontend/src/hooks/useMonitor.js`
90-second polling loop with fetchingRef guard (no overlapping fetches).
countdown freezes at 1 while fetch is in flight; resets to 90 in finally block.

### `frontend/src/components/report/ReportContainer.jsx`
Calls generateReport only when reportKey() changes or care mode switches.
reportKey() = "patient_id|sofa|HR|SpO2|SBP|GCS" — prevents redundant LLM calls.

### `frontend/src/components/report/SHAPPanel.jsx`
anime.js v4.5.0 spring-physics bar animation. displayFactors state preserves
last non-empty factors across backend SHAP failures.

### `frontend/src/components/layout/HeaderBar.jsx`
68px main bar + 32px status strip. BrandMark with glass glint, LiveChip with
ECG animation, PatientPill, ThemeToggle, Exit button.

### `frontend/src/pages/SelectPatient.jsx`
Landing page with TopNav (72px), SystemStatusBar (34px), and 3 patient cards.

### `model_utils.py`
Shared utility module imported by backend services, train_federated.py, server.py, client.py. Contains:
- `ALERT_THRESHOLD = 7.0` — single source of the alert threshold (imported by the backend)
- `ICUModel` class — 108→256→128→64→1 DNN with Dropout 0.3 (configurable `hidden=`, `dropout=`)
- `ICUDataset` class — PyTorch Dataset for DataLoader
- `weighted_mse_loss` — weight = 1 + SOFA × 0.5 (optional; only with `train_model(..., weighted_loss=True)`)
- `get_sample_weights` — oversampling infrastructure (currently disabled)
- `train_model` — AdamW training, plain MSE by default, + FedProx proximal term
- `compute_sofa_risk_metrics` — risk-band confusion matrix + alert metrics at `ALERT_THRESHOLD`
- `evaluate_model` — MSE, MAE, R² metrics
- `get_weights` / `set_weights` — FL weight serialisation
- `get_trend` — trend direction from polyfit slope
- `classify_range` — low/normal/high classification
- `apply_dp_to_update` — differential privacy: clip + noise
- `estimate_privacy_budget` — approximate ε via Gaussian mechanism

### `train_federated.py`
Complete FL pipeline: Phase 0 (BigQuery → TF-IDF → scale → CSVs) + Phases 1-6 (FL).
Outputs: `federated_model.pth`, `shap_background.npy`, `training_metadata.json`

### `server.py`
Real Flower FL server. Listens on 127.0.0.1:8080, runs 100 rounds.
Uses FedAvg — the same aggregation as the simulation (`train_federated.py`, `USE_FEDYOGI = False`).

### `client.py`
Real Flower FL client. Takes `--client_id` (0, 1, 2).
Loads from `data/fl_training/client_{id}.csv`. Trains with FedProx.

### `models/federated_model.pth`
PyTorch state_dict. Keys: `net.0.weight`, `net.0.bias`, `net.3.weight`, `net.3.bias`,
`net.6.weight`, `net.6.bias`, `net.9.weight`, `net.9.bias` (ReLU/Dropout layers in between have no weights).

### `models/scaler.pkl`
sklearn StandardScaler fitted on 108 features from training data.
`scaler.feature_names_in_` holds the authoritative 108-column order.

### `models/tfidf_vectorizer.pkl`
sklearn TfidfVectorizer with 90-term SOFA vocabulary whitelist.
Fitted on all 283,208 MIMIC-III clinical notes with 10,000-char limit per admission.

### `models/feature_columns.pkl`
Python list of 108 column names in exact training order.

### `models/shap_background.npy`
Numpy array shape (300, 108) — 300 randomly selected training rows.

### `data/fl_training/client_0/1/2.csv`
Each: 13,932–13,933 rows × 109 columns (108 features already scaled + sofa_score).
Features are already StandardScaler-normalised and clipped to ±10.

---

## 21. ALL FINALIZED TECHNICAL PARAMETERS

| Parameter | Value | Where Used |
|---|---|---|
| Feature vector size | 108 | model, scaler, all files |
| TF-IDF vocab size | 90 (SOFA whitelist) | tfidf_vectorizer.pkl |
| TF-IDF ngram_range | (1, 2) | tfidf_vectorizer.pkl |
| Notes sample size | 283,208 (all) | train_federated.py |
| Notes text limit | 10,000 chars | train_federated.py |
| Sliding window size | 20 | app.py, patient_vitals |
| Trend slope threshold | 0.1 | get_trend() in model_utils.py |
| Normal HR range | 60–100 | trend display, LLM prompt |
| Normal RR range | 12–20 | trend display, LLM prompt |
| Normal SpO2 range | 95–100 | trend display, LLM prompt |
| Normal Temp range | 36.5–37.5 | trend display, LLM prompt |
| Normal SBP range | 100–120 | trend display, LLM prompt |
| Normal DBP range | 60–80 | trend display, LLM prompt |
| Normal MAP range | 70–100 | trend display, LLM prompt |
| SOFA Low threshold | < 5 | risk_label() in app.py |
| SOFA Moderate threshold | 5–9 | risk_label() in app.py |
| SOFA High threshold | ≥ 10 | risk_label() in app.py |
| Alert threshold | SOFA ≥ 7 | ALERT_THRESHOLD in model_utils.py (imported by ml_service.py) |
| DNN architecture | 108→256→128→64→1, Dropout 0.3 | ICUModel in model_utils.py |
| Trainable parameters | 69,121 | ICUModel |
| FL rounds | 100 | train_federated.py, server.py |
| Epochs per round | 3 | train_federated.py, client.py |
| Batch size | 64 | train_model() in model_utils.py |
| Base learning rate | 0.001 | train_federated.py, client.py |
| LR decay | 0.99 | train_federated.py, client.py |
| Gradient clip | 1.0 | train_federated.py, client.py |
| FedProx μ | 0.5 | train_federated.py, client.py |
| Server aggregation | FedAvg (`USE_FEDYOGI = False`) | train_federated.py |
| FedYogi η (optional, only if USE_FEDYOGI) | 0.001 | train_federated.py |
| FedYogi β1 / β2 / τ (optional) | 0.9 / 0.99 / 0.001 | train_federated.py |
| Loss | plain MSE (`weighted_loss=True` → weight=1+SOFA×0.5, optional) | train_model() |
| Conformal 90% interval q̂ | 3.34 (computed at backend start) | ml_service.py |
| SHAP background size | 300 | shap_background.npy |
| LLM main model | openai/gpt-oss-120b | backend/services/llm_service.py |
| LLM judge model | openai/gpt-oss-20b | backend/services/llm_service.py |
| LLM temperature | 0.2 | llm_service.py |
| LLM main max_tokens | 3500 | llm_service.py |
| LLM judge max_tokens | 300 | llm_service.py |
| LLM judge timeout | 12s (threading) | llm_service.py |
| LLM cache TTL | 90s | llm_service.py (_CACHE_TTL) |
| High reliability threshold | ≥ 0.80 | llm_service.py |
| Moderate reliability threshold | 0.60–0.80 | llm_service.py |
| Frontend refresh interval | 90s | useMonitor.js |
| GCS Eye range | 1–4 | sidebar input |
| Stress Score range | 0–10 | sidebar input |
| Hospital clients | 3 | server.py, train_federated.py |
| Server address | 127.0.0.1:8080 | server.py, client.py |
| SHAP top features | 5 | backend/services/ml_service.py |

---

## 22. MODEL PERFORMANCE METRICS

### Current Federated DNN (federated_model.pth — 2026-11-12)

| Metric | Value |
|---|---|
| Overall MAE | **1.8247 SOFA points** |
| Overall R² | **0.4725** |
| Prediction range | 1.72 – 23.45 |
| Training samples | 41,798 ICU stays |
| Test samples | 10,450 ICU stays |
| FL rounds | 100 (best was round 26; server validation loss flat ~5.8–6.1) |
| Server aggregation | FedAvg + FedProx (μ=0.5) |
| Loss | plain MSE |
| Architecture | 108→256→128→64→1, Dropout 0.3, 69,121 params |
| Conformal 90% interval | ± 3.34 SOFA points |

**Evaluation limitation:** one row per ICU stay, split 80/20 by ICU stay — a patient with
several stays can appear in both train and test. Retrospective MIMIC-III results, not
independent clinical validation.

### Per-Risk-Level Performance

| Risk Level | n samples | MAE |
|---|---|---|
| Low (<5) | 6,929 | 1.331 |
| Moderate (5-9) | 2,919 | 2.481 |
| High (≥10) | 602 | 4.328 |

**Note on per-class R²:** per-band R² is negative by design — there is little variance
inside one band, so only the overall R² is meaningful. Ranking patients within the same
band needs lab values (bilirubin, creatinine, platelets) that are not inputs.

**3-class confusion matrix** (rows true Low/Mod/High, cols predicted Low/Mod/High):
`[[6617, 305, 7], [2159, 669, 91], [112, 203, 287]]` → 3-class accuracy 72.5%;
High-band recall 47.7%, High-band precision 74.5%.

### High-Risk Alert (predicted SOFA ≥ 7 vs actual SOFA ≥ 10)

| Metric | Value |
|---|---|
| AUROC | 0.9254 |
| PR-AUC | 0.6456 |
| Recall | 67.1% (404/602) |
| Specificity | 96.3% |
| Precision | 52.9% |
| F1 | 0.5915 |
| Missed (FN) / False alarms (FP) / TN | 198 / 360 / 9,488 |

The SOFA Validation Drawer reads all of these numbers live from `training_metadata.json`
(sent by the backend as `_training_meta`), so it always matches the loaded model.

### Comparison to Previous Versions

| Version | R² | Key Change |
|---|---|---|
| Original FL notebook (10R, 3E, plain MSE) | 0.04 | Baseline (historical) |
| After weighted MSE (1+y×3) | 0.11–0.13 | Loss improvement (historical) |
| SOFA vocabulary whitelist (90 terms) | 0.1752 | Feature breakthrough (historical) |
| FedAdam server optimizer | 0.2148 | Server optimization (historical) |
| FedYogi server optimizer | 0.2229 | Adaptive rate (historical) |
| Old dataset final (FedYogi, weighted MSE, 128-64-32), alert ≥ 8 — historical | 0.398 | Last old-dataset model (historical; recall 23%) |
| dataset_v2, FedYogi η=0.01, weighted MSE, alert ≥ 8 — historical | 0.426 | New dataset (historical; AUROC 0.915) |
| + FedAvg + plain MSE | 0.457 | Simpler server + loss (AUROC 0.921) |
| + 256-128-64 + Dropout 0.3 | 0.473 | Wider network + Dropout (AUROC 0.925) |
| **Same model, alert ≥ 7** | **0.4725** | **Current — alert recall 67.1%, specificity 96.3%** |

(Old-dataset numbers are not directly comparable: the label definition changed.)

**Experiments on dataset_v2 that did not help:** lower LR 5e-4, weight decay 1e-3, Huber
loss, 512-256-128, averaging 8 models, isotonic recalibration, missing-value flags,
several hourly rows per stay (+0.007 R² for 9× data). Reference ceiling (centralized
gradient boosting, same 108 columns): MAE ~1.81, R² ~0.49, AUROC ~0.927.

---

## 23. TECHNOLOGY STACK

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML Framework | PyTorch ≥ 2.0 |
| Federated Learning | Flower (flwr) ≥ 1.8 — FedAvg + FedProx |
| Explainability | SHAP ≥ 0.44 (DeepExplainer) |
| Data Processing | Pandas, NumPy |
| NLP | scikit-learn TfidfVectorizer (90-term SOFA vocabulary) |
| LLM Provider | Groq — main: openai/gpt-oss-120b (3500 tokens), judge: openai/gpt-oss-20b (300 tokens) |
| LLM Cache | 90s in-process Python dict (care_mode|sofa|hr|spo2 key) |
| Backend Framework | FastAPI ≥ 0.115 + uvicorn (port 8000) |
| Frontend Framework | React 19 + Vite 8 (port 5174) |
| Frontend UI | Framer Motion v14.5 + anime.js v4.5 + AOS v2 |
| Frontend Charts | Recharts (vital trend line charts) |
| Design System | "Modern Dark Cinema Mobile" glassmorphism, Figtree font, CSS tokens |
| Database | SQLite via `models/icu_monitor.db` (~23 MB, gitignored) |
| Dataset | MIMIC-III |
| Data Access | Google BigQuery (mimic-project-2, dataset_v2) |
| CV Monitor — Camera Capture | OpenCV ≥ 5.0 |
| CV Monitor — Face AI | MediaPipe ≥ 0.10.35 (FaceLandmarker + 52 blendshapes) |
| CV Stress Scoring | PSPI (Prkachin-Solomon Pain Intensity, 2008) |
| CV GCS Scoring | 30-second rolling behavioral window + personal calibration |
| Model Serialization | torch.save / joblib |
| Environment Config | python-dotenv |

---

## 24. HOW TO RUN THE PROJECT

### Prerequisites:
```bash
cd /Users/sneh.patel/Desktop/capstone/icu_monitor
pip install -r requirements.txt
```

### Ensure `.env` has the Groq API key:
```
GROQ_API_KEY=gsk_...your_key_here...
```

### Run the ClinIQ Dashboard (main demo — 2 terminals):

**Terminal 1 — FastAPI backend:**
```bash
cd /Users/sneh.patel/Desktop/capstone/icu_monitor
uvicorn backend.main:app --reload --port 8000
# API available at http://localhost:8000
```

**Terminal 2 — React frontend:**
```bash
cd /Users/sneh.patel/Desktop/capstone/icu_monitor/frontend
npm run dev
# Opens at http://localhost:5174
```

### Seed the database (first run only):
```bash
cd /Users/sneh.patel/Desktop/capstone/icu_monitor
python migrate_to_db.py
# Creates models/icu_monitor.db from fl_training CSVs
```

### Run the CV Monitor (standalone — stress score + GCS eye score from webcam):
```bash
python cv_monitor.py         # built-in webcam
python cv_monitor.py --cam 1 # USB webcam
# First run downloads face_landmarker.task (~12 MB automatically)
# Press Q or ESC to quit
```

### Run the tests:
```bash
cd /Users/sneh.patel/Desktop/capstone/icu_monitor
pytest tests                      # Python (ClinIQ backend + FL): 513 passed

# Frontend tests live in tests/, outside frontend/, so they need this
# symlink to resolve packages (gitignored, create once):
ln -s frontend/node_modules node_modules
cd frontend && npx vitest run     # Vitest: 232 passed (27 files)
```

### Retrain the Federated Model (requires BigQuery access):
```bash
python train_federated.py
# Phase 0: BigQuery + TF-IDF (~15-20 min)
# Phase 1-6: FL training, 100 rounds (~25-40 min)
# Total: ~40-60 minutes
```

### Demo Real Federated Learning (4 terminals):
```bash
# Terminal 1 — start server first:
python server.py

# Terminal 2:
python client.py --client_id 0

# Terminal 3:
python client.py --client_id 1

# Terminal 4:
python client.py --client_id 2
```

---

## 25. PROJECT EVOLUTION HISTORY

**Phase 1 — Architecture and Design:**
Defined system architecture, chose MIMIC-III, Google BigQuery. Drew architecture diagram.

**Phase 2 — BigQuery Preprocessing:**
Ran all 12 SQL queries to extract and transform MIMIC-III data. Produced ml_dataset_final
with 60,188 rows (old `Dataset` build — historical; replaced by dataset_v2 in Phase 9).

**Phase 3 — Ensemble ML (trial/):**
LightGBM + XGBoost + Random Forest. SHAP TreeExplainer. Groq LLM with self-consistency.
Performance: R²=0.40, MAE=1.86. Could not do FL (tree models not compatible with FedAvg).

**Phase 4 — Federated Learning Integration:**
Switched to PyTorch DNN for FL compatibility. Flower framework. First FL training.
Performance: R²=0.04–0.18 (undertrained).

**Phase 5 — Training Improvements:**
Added weighted MSE (later replaced by plain MSE in Phase 9), AdamW, SaveBestStrategy, SHAP DeepExplainer.
Performance improved to R²=0.13–0.25.

**Phase 6 — SOFA Vocabulary Whitelist Breakthrough:**
Replaced 600 data-driven TF-IDF features with 90-term SOFA vocabulary whitelist.
Reduced model from 618→256→128→64→1 to 108→128→64→32→1 (later widened to 256-128-64 + Dropout in Phase 9).
Performance: R²=0.1752 (biggest single jump — +94% relative improvement).

**Phase 7 — FedAdam/FedYogi Server Optimization (historical; FedYogi replaced by FedAvg in Phase 9):**
Added server-side adaptive optimization. FedAdam → R²=0.2148. FedYogi → R²=0.2229.
Added FedProx client regularisation. Tuned loss weight from 3.0 to 0.5.

**Phase 8 — Notes Corpus Expansion:**
Increased NOTES_SAMPLE_SIZE from 80k to all 283k notes.
Increased NOTES_TEXT_LIMIT from 5k to 10k chars.
Performance: R²=0.4171 (50% relative improvement over FedYogi alone; old dataset).

**Phase 9 — dataset_v2 rebuild and model update (current, 2026-11-12):**
Rebuilt the data in BigQuery as `dataset_v2` (time-matched hourly SOFA label, 52,248 ICU
stays). Replaced FedYogi with FedAvg (FedYogi η=0.01 exploded on the new data), weighted
MSE with plain MSE, and 128-64-32 with 256-128-64 + Dropout 0.3. Lowered the alert
threshold from 8 to 7. Performance: **MAE 1.8247, R²=0.4725, AUROC 0.9254, alert recall
67.1% at 96.3% specificity**.

---

## 26. NOVEL CONTRIBUTIONS

1. **Multimodal fusion of IoMT + CV + Clinical NLP in a unified risk predictor.**

2. **SOFA-vocabulary TF-IDF whitelist:** 90 terms directly mapped to 6 SOFA organ
   components, replacing noise-prone data-driven selection.

3. **Trend-based temporal feature engineering with sliding window.**

4. **FedAvg + FedProx Federated Learning:** FedAvg server aggregation combined with
   FedProx client proximal regularisation (μ=0.5) and Dropout 0.3 — stable FL with
   R²=0.4725, AUROC 0.9254 (FedYogi was used in earlier versions).

5. **LLM Self-Consistency Check:** 3-component consistency metric (TF-IDF + clinical
   interventions + clinical conditions) for quantitative LLM reliability assessment.

6. **SHAP DeepExplainer on a Federated DNN:** Feature-level explainability for a
   privacy-preserving federated neural network.

---

## 27. DESIGN CONSTRAINTS AND ASSUMPTIONS

### Constraints:
- No patient data can leave hospitals → Federated Learning
- Must run on standard hardware (no GPU required for inference)
- Must be explainable to clinicians → SHAP + LLM
- MIMIC-III only (no real-time patient data access)

### Assumptions:
- Vital signs are continuously available and reasonably accurate
- A 20-reading sliding window captures sufficient temporal context
- SOFA score from MIMIC-III is the correct target label
- The simulated 3-hospital FL split is a valid proxy for real multi-hospital FL
- Federated nodes behave honestly (no adversarial clients)

### Limitations:
- R² = 0.4725 — limited by missing direct lab values (bilirubin, creatinine, platelets) and vasopressor doses
- One row per ICU stay, split by ICU stay — a patient with several stays can appear in both train and test; retrospective MIMIC-III results only
- The "hospitals" are artificial splits of one MIMIC-III dataset
- The CV features (GCS, stress) are manually entered, not from a real camera
- The model cannot retrain on new patient data automatically

---

## 28. VIVA PREPARATION — KEY POINTS

### Q: Why did you choose MIMIC-III instead of MIMIC-IV?
MIMIC-III is more stable for preprocessing and has well-tested derived tables
including pre-computed SOFA scores in mimiciii_derived.sofa. MIMIC-III was
sufficient for our requirements.

### Q: Why TF-IDF instead of ClinicalBERT?
TF-IDF is simpler, faster, interpretable, and produces features directly usable
by the ML model. With the SOFA vocabulary whitelist approach, TF-IDF achieves
strong performance (contributes to R²=0.4725) because every feature has a
clinically grounded meaning.

### Q: Why PyTorch DNN instead of LightGBM for the federated model?
Federated Learning with FedAvg works by averaging numerical weight matrices.
Tree-based models do not have weight matrices — each tree has a different structure
per hospital. Neural networks are natural candidates for FedAvg.

### Q: Is your Federated Learning real or simulated?
The FL protocol is real: Flower server/client, TCP sockets, weight-only
transmission, FedAvg aggregation. What is simulated is the hospital data
(artificially split from MIMIC-III). This is the standard approach in all
published FL research.

### Q: Why is R² 0.4725 and not higher? Is that acceptable?
SOFA prediction from vital signs + clinical text alone is inherently limited
because SOFA is mostly computed from lab values (bilirubin, creatinine, platelets,
PaO2/FiO2) and vasopressor doses, which are deliberately not inputs (non-invasive
design). R²=0.4725 is close to the ceiling for these inputs: centralized gradient
boosting on the same 108 columns reaches only R² ~0.49. For the clinical decision
that matters — flagging high-risk patients — the model has AUROC 0.9254.

### Q: Why FedAvg and not FedYogi?
Earlier versions used FedYogi, which applies the Yogi adaptive optimizer at the server
after averaging. On the rebuilt dataset_v2 data, FedYogi with η=0.01 made the server
loss explode in early rounds, and plain FedAvg (with FedProx on the clients) gave
better results (MAE 1.914 → 1.850, R² 0.426 → 0.457, together with switching to plain
MSE). FedYogi is still available (`USE_FEDYOGI = True`, now with η=0.001).

### Q: Why does the alert fire at predicted SOFA ≥ 7 instead of ≥ 10?
The model under-predicts severe patients (mean prediction ≈ 9.7 for true SOFA ≥ 10).
At ≥ 7 it catches 67.1% of truly high-risk patients at 96.3% specificity; moving
from 8 to 7 catches 36 more of them for 140 more false alarms.

### Q: What is FedProx and why did you use it?
FedProx adds a proximal term (μ/2)×||w_local - w_global||² to each hospital's
training loss. This prevents hospitals from drifting too far from the global model
during local training — reducing the aggregation noise that caused plain FedAvg
to oscillate.

### Q: What is the self-consistency check and why is it important?
LLMs can hallucinate. We send the same prompt 3 times and measure how similar
the responses are using a 3-component score: TF-IDF phrasing similarity (20%),
clinical intervention agreement (50%), clinical condition agreement (30%).
A high consistency score means the LLM is producing stable, reproducible clinical
reasoning — a proxy for reliability. Low score means the clinical picture may be
genuinely ambiguous.

### Q: How does SHAP work with a neural network?
We use shap.DeepExplainer with 300 background samples. DeepExplainer uses
backpropagation-based attribution to compute how much each of the 108 input
features contributed to the SOFA prediction for a specific patient. The model
must be in eval() mode for deterministic output.

### Q: What happens to patient data in Federated Learning?
Patient data never leaves the hospital. Only model weight matrices (numpy arrays
of floating point numbers) are transmitted. These weights contain no
patient-identifiable information.

---

*End of SKILL.md — Complete Project Context Document*
*Last updated: 2026-11-12 — current model: dataset_v2, FedAvg + FedProx, plain MSE, 256-128-64 + Dropout 0.3, alert ≥ 7*

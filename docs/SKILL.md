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
18. [Streamlit Application — Complete Pipeline](#18-streamlit-application--complete-pipeline)
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

**Access Method:** Google BigQuery (project: `mimic-icu-risk-prediction`)

**Why MIMIC-III and not MIMIC-IV:**
MIMIC-IV was available but MIMIC-III was chosen because it was more stable for
preprocessing and experimentation. MIMIC-III has well-documented derived tables
including pre-computed SOFA scores in `mimiciii_derived.sofa`.

**Key BigQuery Tables Used:**

| Table | Purpose |
|---|---|
| `physionet-data.mimiciii_clinical.chartevents` | Vital signs, GCS, Stress Score |
| `physionet-data.mimiciii_notes.noteevents` | Clinical notes (nursing, physician) |
| `physionet-data.mimiciii_derived.sofa` | Pre-computed SOFA scores (target label) |

**Final Processed Table:**
`mimic-icu-risk-prediction.processed_mimic.ml_dataset_final`

**Dataset Size:**
- Total ICU samples: 60,189 (one per ICU stay window — latest window selected)
- Clinical notes: 283,208 raw notes (combined per admission)
- After TF-IDF merge: 60,189 rows × 619 columns (618 features + sofa_score)
- Train/test split: 80/20 → ~48,151 train, ~12,038 test
- Federated split: 3 hospitals of ~15,889 / 15,890 / 16,372 rows each

---

## 6. BIGQUERY SQL PIPELINE — ALL 12 QUERIES

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
This converts irregular ICU measurements into consistent time intervals.

Output table: `vitals_10min`

**Query 4 — Compute sliding window trend features:**
Uses SQL window function with ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
(20 rows per window, partitioned by icustay_id).
Computes for each window:
- HR_mean, HR_std (AVG and STDDEV of HR over last 20 readings)
- RR_mean
- SpO2_mean, SpO2_min (AVG and MIN of SpO2)
- Temp_mean
- SBP_mean, DBP_mean, MAP_mean

Output table: `vital_trends_20`

**Query 5 — Extract latest vital values:**
From `vitals_10min`, extracts the raw (latest) vital values and renames them:
HR → latest_HR, RR → latest_RR, SpO2 → latest_SpO2, Temp → latest_Temp,
SBP → latest_SBP, DBP → latest_DBP, MAP → latest_MAP

Output table: `latest_vitals`

**Query 6 — Extract clinical notes:**
From NOTEEVENTS, keeps only clinically relevant categories:
- 'Nursing'
- 'Physician'
- 'Nursing/Other'
- 'Discharge summary'
Filters out null text entries.

Output table: `clinical_notes`

**Query 7 — Extract SOFA scores:**
From `physionet-data.mimiciii_derived.sofa`, extracts pre-computed SOFA scores
joined by (subject_id, hadm_id, icustay_id).

Output table: `sofa_scores`

**Query 8 — Join trends + latest vitals + SOFA:**
Joins `vital_trends_20` with `latest_vitals` on (icustay_id, window_time).
LEFT JOINs `sofa_scores` on icustay_id.
Creates the base ML dataset with all physiological features + target label.

Output table: `ml_dataset`

**Query 9 — Extract GCS Eye Opening:**
From CHARTEVENTS, extracts GCS Eye Opening scores (itemid=220739).
Renames valuenum → GCS_eye_opening.

Output table: `gcs_eye`

**Query 10 — Extract Stress Score (originally Pain Score):**
From CHARTEVENTS, extracts Pain/Stress Score (itemid=223791).
Renames valuenum → stress_score.
NOTE: This was called "pain_score" in the original design but was RENAMED to
"stress_score" in the BigQuery database before the final run. In all saved
artifacts (scaler.pkl, tfidf_vectorizer.pkl, client CSVs), the column is
named "stress_score". This is the final name.

Output table: `pain_score` (internal table name, column is stress_score)

**Query 11 — Combine CV features into 10-min windows:**
UNIONs GCS Eye and Stress Score tables, then aggregates into 10-min windows
using AVG per window_time per icustay_id.

Output table: `cv_features_10min`

**Query 12 — Create final ML dataset:**
LEFT JOINs `ml_dataset` with `cv_features_10min` on (icustay_id, window_time).
This adds GCS_eye_opening and stress_score to the existing feature set.

Output table: `ml_dataset_final` (THE FINAL DATASET)

---

## 7. THREE DATA MODALITIES

The system integrates exactly three data modalities. Each contributes a distinct
type of clinical information.

### Modality 1 — IoMT Physiological Vital Signs (Primary Modality)

Source: MIMIC-III CHARTEVENTS → BigQuery pipeline → client CSVs
In inference: manually entered by nurse/doctor in the Streamlit sidebar

7 vital signs:
- **HR (Heart Rate)** — beats per minute. Normal: 60–100
- **RR (Respiratory Rate)** — breaths per minute. Normal: 12–20
- **SpO2 (Oxygen Saturation)** — percentage. Normal: 95–100
- **Temperature** — Celsius. Normal: 36.5–37.5
- **SBP (Systolic Blood Pressure)** — mmHg. Normal: 100–120
- **DBP (Diastolic Blood Pressure)** — mmHg. Normal: 60–80
- **MAP (Mean Arterial Pressure)** — mmHg. Normal: 70–100

Valid physiological ranges (used for input validation):
- HR: 30–220, RR: 5–60, SpO2: 50–100, Temp: 30–43°C
- SBP: 40–250, DBP: 20–150, MAP: 30–200

### Modality 2 — Computer Vision Features (Behavioral/Neurological)

Source: MIMIC-III CHARTEVENTS (same table, different itemids)
In inference: manually entered in sidebar (simulates a bedside camera CV model)

Conceptual explanation: In a real hospital deployment, a camera above the
patient's bed feeds into a computer vision model that estimates neurological
status and behavioral distress. In this project, the values come from
MIMIC-III (where clinicians recorded these manually) and are entered manually
at inference time.

2 CV features:
- **GCS Eye Opening** — Glasgow Coma Scale eye response. Range: 1–4
  - 1 = No response (worst)
  - 2 = To pain
  - 3 = To voice
  - 4 = Spontaneous (best/normal)
  Lower score = worse neurological state

- **Stress Score** — Behavioral/physiological stress (originally Pain Score
  in MIMIC, renamed to stress_score in the BigQuery database).
  Range: 0–10. Higher = more distress.

### Modality 3 — Clinical Notes (NLP)

Source: MIMIC-III NOTEEVENTS → TF-IDF vectorization
In inference: free-text entered by the clinician in the sidebar

Categories used: Nursing, Physician, Nursing/Other, Discharge summary

The raw clinical text contains information no vital sign can capture:
patient history, medications, physician observations, clinical context.

This text is converted to 600 numerical TF-IDF features before being
fed to the ML model.

---

## 8. FEATURE ENGINEERING — COMPLETE DETAIL

The final ML model receives 618 features total, in this exact order:

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

WHY these specific features: Originally all statistics (mean, std, min, max,
slope) were computed for every vital. This was reduced to only clinically
meaningful features to avoid feature explosion. HR_std captures instability.
SpO2_min catches any dangerous oxygen dip even if the mean looks acceptable.

### Group 2 — Latest Vital Values (7 features)
The most recent reading of each vital sign, representing current patient state.

| Feature | Value |
|---|---|
| latest_HR | Current Heart Rate |
| latest_RR | Current Respiratory Rate |
| latest_SpO2 | Current SpO2 |
| latest_Temp | Current Temperature |
| latest_SBP | Current Systolic BP |
| latest_DBP | Current Diastolic BP |
| latest_MAP | Current Mean Arterial Pressure |

### Group 3 — Computer Vision Features (2 features)

| Feature | Value |
|---|---|
| GCS_eye_opening | Glasgow Coma Scale Eye Opening (1–4) |
| stress_score | Physiological/Behavioral Stress Score (0–10) |

### Group 4 — TF-IDF Clinical NLP Features (600 features)
Generated by applying the saved TF-IDF vectorizer to the clinical note text.
Each of the 600 features represents the importance of a specific word or
bigram from the clinical notes vocabulary.

**Total: 9 + 7 + 2 + 600 = 618 features**

### Feature Scaling
All 618 features are scaled using a StandardScaler (zero mean, unit variance)
that was fitted on the full training dataset. The fitted scaler is saved as
`models/scaler.pkl`.

### Feature Order
The exact order of all 618 features is critical. The saved scaler records
the expected feature order in `scaler.feature_names_in_`. At inference time,
the input DataFrame is reordered to exactly match this order before scaling.

---

## 9. SLIDING WINDOW MECHANISM

**Window Size:** 20 readings (fixed)

**Purpose:** Health deterioration is a temporal process. A single reading cannot
show whether a patient is stable or deteriorating. The sliding window captures
the patient's recent physiological trajectory.

**Mechanics:**
1. `models/patient_vitals.csv` stores the last 20 vital sign readings
2. When a new prediction is run, the new vitals are appended as a new row
3. The oldest row is dropped (tail(20))
4. The updated window is saved back to CSV
5. Trend features are recomputed from the 20 current rows

**Initialization:**
If `patient_vitals.csv` does not exist, a default 20-row dataset is created
(simulating a patient with gradually worsening vitals — used for demo).

**Trend Direction Calculation:**
Uses numpy.polyfit to fit a line through the 20 readings.
- Slope > 0.1 → "increasing"
- Slope < -0.1 → "decreasing"
- -0.1 ≤ slope ≤ 0.1 → "stable"

**Trend Status Classification:**
For each vital, the mean is compared against clinical normal ranges:

| Vital | Normal Low | Normal High |
|---|---|---|
| HR | 60 | 100 |
| RR | 12 | 20 |
| SpO2 | 95 | 100 |
| Temperature | 36.5 | 37.5 |
| SBP | 100 | 120 |
| DBP | 60 | 80 |
| MAP | 70 | 100 |

Result: each vital gets a status ("low", "normal", "high") and a direction
("increasing", "stable", "decreasing"). Combined: "SpO2 → low & decreasing"

---

## 10. TF-IDF CLINICAL NLP PIPELINE

**Method:** TF-IDF (Term Frequency-Inverse Document Frequency)
**Why TF-IDF and not ClinicalBERT:** ClinicalBERT was explored in
`New_ICU_ClinicalBERT.ipynb` but TF-IDF was chosen for the final implementation
because it is simpler, faster, interpretable, and produces features directly
usable by the ML model without a separate embedding step.

**Vectorizer Parameters (finalized):**
- max_features = 600 (vocabulary limited to 600 most important terms)
- ngram_range = (1, 2) (single words AND two-word phrases like "chest pain")
- stop_words = "english" (common words like "the", "a" removed)
- max_df = 0.9 (ignore terms appearing in >90% of documents)
- min_df = 10 (ignore terms appearing in <10 documents)

**Training:**
- Clinical notes grouped by hadm_id (admission ID)
- Text truncated to 3000 characters
- TF-IDF fitted on the combined note corpus
- Fitted vectorizer saved as `models/tfidf_vectorizer.pkl`

**At Inference:**
- User's clinical note → tfidf.transform([note]) → sparse matrix
- Converted to dense DataFrame with 600 columns
- These 600 columns are appended to the vital features
- IMPORTANT: The same saved vectorizer is used. No re-fitting at inference.

**Handling Empty Notes:**
If no clinical note is entered (""), the app substitutes
"No clinical notes provided" — this produces a valid but sparse TF-IDF vector.

---

## 11. TARGET VARIABLE — SOFA SCORE

**What is SOFA:**
Sequential Organ Failure Assessment — an internationally standardized ICU
severity scoring system used worldwide by clinicians.

**SOFA Range:** 0 to 24

**How SOFA is calculated (in MIMIC):**
SOFA assesses the function of 6 organ systems:
1. Respiratory (PaO2/FiO2 ratio)
2. Coagulation (Platelet count)
3. Liver (Bilirubin)
4. Cardiovascular (MAP and vasopressors)
5. Central Nervous System (GCS score)
6. Renal (Creatinine and urine output)
Each contributes 0–4 points. Total = 0–24.

**Source in MIMIC-III:** `physionet-data.mimiciii_derived.sofa` (pre-computed)

**Risk Classification Thresholds (finalized):**

| SOFA Score | Risk Level |
|---|---|
| 0 – 4 | 🟢 Low Risk |
| 5 – 9 | 🟡 Moderate Risk |
| 10 – 24 | 🔴 High Risk |

**Why SOFA:**
SOFA is clinically validated. It is not an arbitrary label — it is the real
severity metric used by ICU clinicians worldwide. Using it makes the project
medically grounded and defensible in a viva or clinical context.

**Model Output:**
The model predicts SOFA directly (0–24 range). No normalization is applied
at inference (the model was trained on raw SOFA values). The prediction is
clipped to [0, 24] using np.clip.

---

## 12. MACHINE LEARNING MODEL

### Architecture — PyTorch Fully Connected DNN

```
Input: 618 features (scaled)
  ↓
Linear(618 → 256) + ReLU
  ↓
Linear(256 → 128) + ReLU
  ↓
Linear(128 → 64) + ReLU
  ↓
Linear(64 → 1)
  ↓
Output: SOFA score (0–24)
```

Class definition: `ICUModel` in `model_utils.py`
Saved weights: `models/federated_model.pth`

### Training Configuration (in train_federated.py)
- Loss function: MSE (Mean Squared Error)
- Optimizer: Adam (lr=0.001)
- Batch size: 64 (mini-batch via DataLoader)
- Epochs per round per client: 10
- FL Rounds: 20
- Aggregation: FedAvg (Federated Averaging)

### Why PyTorch DNN and not LightGBM/XGBoost for the final model:
Federated Learning (specifically FedAvg) works by averaging model weight
matrices across clients. This only works natively with neural networks, where
the model is defined by numerical weight tensors. Gradient boosting models
(LightGBM, XGBoost, Random Forest) do not have a weight vector that can be
averaged — each tree has a different structure per client.

Therefore, the PyTorch DNN is used as the federated model. LightGBM and XGBoost
were trained separately (in `Final_ICU.ipynb`) as the ensemble approach, but
the final app uses the federated DNN.

### Ensemble ML (reference, not used in final app):
Trained in `notebooks/Final_ICU.ipynb` (requires BigQuery/Colab):
- LightGBM + XGBoost + Random Forest
- Grid search for optimal weights
- Final weights: 0.52 × LightGBM + 0.48 × XGBoost + 0.00 × RandomForest
- Feature selection: top 220 features using LightGBM importance
- Performance: MAE=1.86, RMSE=2.44, R²=0.40

---

## 13. FEDERATED LEARNING ARCHITECTURE

### Concept
Multiple hospitals each have their own private patient data. They cannot share
it. Federated Learning allows them to collaboratively train a global model
without any patient data leaving any hospital.

### What IS Federated in This Project:
- The FL communication protocol is REAL (Flower server/client, TCP sockets)
- Weights are the only thing transmitted — no patient data
- Each client trains on its own separate dataset
- FedAvg aggregation produces a global model

### What IS Simulated:
- The "hospital data" is artificially created by splitting one MIMIC-III dataset
  into 3 parts. In real life, each hospital would have its own patients.
- All three clients run on the same machine (127.0.0.1)

### Framework: Flower (flwr)

**Architecture:**

```
Global Server (server.py)
    │
    │ sends global weights
    ▼
┌───────────────────────────────────────┐
│  Hospital 0      Hospital 1      Hospital 2
│  (client_0.csv)  (client_1.csv)  (client_2.csv)
│  ~15,889 rows    ~15,890 rows    ~16,372 rows
│       │               │               │
│   Local Train     Local Train     Local Train
│   (10 epochs)    (10 epochs)     (10 epochs)
│       │               │               │
│   Updated         Updated         Updated
│   Weights         Weights         Weights
└───────────────────────────────────────┘
    │               │               │
    └───────────────┴───────────────┘
                    │
                    ▼
            FedAvg Aggregation
            (weighted average of weights
             proportional to dataset size)
                    │
                    ▼
            Updated Global Model
                    │
                    ▼
          Save federated_model.pth
```

**FL Rounds:** 20
**Clients per round:** 3 (fraction_fit=1.0 — all clients participate every round)
**Aggregation:** FedAvg (standard Flower implementation)
**Server address:** 127.0.0.1:8080

### Two FL Modes Available:

**Mode 1 — Simulation (train_federated.py):**
Uses `fl.simulation.start_simulation`. All clients run in the same Python
process. Used for retraining the model locally using data/ CSVs.
Run with: `python train_federated.py`

**Mode 2 — Real FL Demo (server.py + client.py):**
Real separate processes connected via TCP socket. Used for demonstrating
that the FL protocol actually works with separate "hospital" nodes.
Run with: `python server.py` (Terminal 1)
           `python client.py --client_id 0` (Terminal 2)
           `python client.py --client_id 1` (Terminal 3)
           `python client.py --client_id 2` (Terminal 4)

### SaveBestStrategy:
A custom Flower strategy that extends FedAvg. After each FL round:
- Saves the latest aggregated model to `federated_model.pth`
- Tracks the round with the best (lowest) evaluation loss
- Saves the best-round model separately as `federated_model_best.pth`

---

## 14. SHAP EXPLAINABILITY

### Why SHAP:
SHAP (SHapley Additive exPlanations) computes the contribution of each feature
to the model's prediction for a specific input. This makes the AI's reasoning
transparent to clinicians — they can see WHY the model predicted a high SOFA
score, not just that it did.

### Method: shap.DeepExplainer
Used because the model is a PyTorch neural network. DeepExplainer uses
backpropagation-based attribution to compute feature importance.

TreeExplainer (used in old ensemble version) CANNOT work with neural networks.
DeepExplainer is specifically designed for deep learning models.

### Background Data:
DeepExplainer requires a reference "background" dataset to compute SHAP values.
This is 100 randomly sampled rows from the training data, saved as
`models/shap_background.npy` (shape: 100 × 618).

At startup, app.py loads the background and creates:
`explainer = shap.DeepExplainer(model, background_tensor)`

### SHAP Inference:
`shap_values = explainer.shap_values(X_tensor)`
Returns shape (618,) — one SHAP value per feature.
Positive value = feature pushes prediction toward higher SOFA (worse outcome)
Negative value = feature pushes prediction toward lower SOFA (better outcome)

### Clinical Feature Filtering:
After computing all 618 SHAP values, we filter to only clinically relevant
features using keyword matching:
Keywords: "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP",
          "GCS", "stress", "hypotension", "respiratory", "mental"

### Top 7 Features:
From the filtered clinical features, the top 7 by absolute SHAP impact are
selected. If fewer than 3 clinical features pass the filter, fall back to
top 7 from all features.

### Clinical Interpretation Mapping:
Each SHAP feature is translated to a human-readable clinical statement:

| Feature contains | Clinical meaning |
|---|---|
| SpO2 | Low oxygen levels |
| HR | Abnormal heart rate pattern |
| SBP / DBP / MAP | Blood pressure instability |
| Temp | Possible infection / fever |
| RR | Respiratory distress |
| GCS | Neurological deterioration |
| stress | Elevated physiological stress |

### Key Risk Factor Mapping:
Top SHAP features are further mapped to standard risk categories:

| Feature keyword | Risk Factor |
|---|---|
| SpO2 | Low oxygen levels |
| RR / respiratory | Respiratory distress |
| SBP / DBP / MAP | Hypotension |
| HR | Abnormal heart rate |
| mental | Altered mental status |
| GCS | Neurological deterioration |
| stress | High physiological stress |

Duplicates are removed. These risk factors feed into the LLM prompt.

---

## 15. LLM INTEGRATION — GROQ

### Provider: Groq
### Model: llama-3.3-70b-versatile
### API: Groq Python SDK (`from groq import Groq`)

**Why Groq and not OpenAI:**
Groq offers extremely fast inference on the Llama model, which is important for
near-real-time clinical decision support. Cost is also lower.

**Temperature:** 0.7 (for self-consistency calls — variation needed)
**Max Tokens:** 800

**System Prompt:**
"You are an expert ICU clinical decision support assistant. Provide concise,
structured, and clinically accurate reasoning."

### LLM Prompt Structure:
The prompt contains 7 sections:

1. **SOFA Score and Risk Level** — the quantitative severity measure
2. **Clinical Notes** — raw text entered by the clinician
3. **Latest Vital Signs** — all 7 vitals with their normal ranges
4. **Neurological & Stress Indicators** — GCS Eye and Stress Score with ranges
5. **Key Risk Factors (SHAP-identified)** — top clinical risks from SHAP
6. **Feature-Level Explanations (SHAP-derived)** — specific feature impacts
7. **Vital Sign Trends** — direction + status for all 7 vitals

### LLM Output Structure (4 sections):
The LLM is instructed to produce exactly 4 sections:
1. **Current Condition** — What is happening with this patient right now
2. **Probable Cause** — Why this deterioration is occurring
3. **Risk Forecast** — What may happen in the next 2–4 hours if untreated
4. **Immediate Actions** — Specific clinical interventions required now

### API Key Management:
Loaded from `.env` file using `python-dotenv`.
If not in `.env`, a sidebar input field appears for the user to paste the key.
The key is NEVER hardcoded in any Python file in the final implementation.

---

## 16. LLM SELF-CONSISTENCY CHECK

**This is one of the novel contributions of the project.**

### Problem it solves:
LLMs can produce different (sometimes contradictory) outputs for the same
prompt. In a clinical setting, an unreliable AI output is dangerous.

### Mechanism:
1. The same prompt is sent to Groq **3 times** (with temperature=0.7 to allow
   natural variation in outputs)
2. All 3 responses are collected
3. Each response is converted to a TF-IDF vector (using sklearn TfidfVectorizer
   with stop_words="english")
4. Cosine similarity is computed between all pairs of response vectors
5. The average pairwise similarity is the **consistency score** (0 to 1)

Formula:
```
consistency_score = (similarity_matrix.sum() - n) / (n × (n - 1))
```
where n=3 (3 responses), diagonal values (self-similarity=1.0) are excluded.

### Reliability Labels:

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.85 | ✅ High Reliability | All 3 responses agree — trust the output |
| 0.65 – 0.85 | ⚠️ Moderate Reliability | Some variation — review carefully |
| < 0.65 | ❌ Low Reliability | Large variation — use clinical judgment |

### Display in App:
- Main response (Response 1) shown prominently
- Consistency score shown as a metric
- Reliability label shown as a colored status
- All 3 responses available in an expandable section

---

## 17. ALERT SYSTEM

**Trigger Condition:** SOFA ≥ 10 (High Risk)

**Moderate Risk:** SOFA ≥ 5 (shows a warning banner)

**Display:**
- High Risk → st.error() → red banner:
  "⚠️ HIGH RISK PATIENT DETECTED — Immediate clinical attention required."
- Moderate Risk → st.warning() → yellow banner:
  "⚠️ MODERATE RISK — Patient requires close monitoring."
- Low Risk → no banner, normal display

The alert appears at the top of the page, above all tabs, so it is immediately
visible to the clinician before they interact with any other section.

---

## 18. STREAMLIT APPLICATION — COMPLETE PIPELINE

### File: `app.py`

### Startup (cached with @st.cache_resource):
1. Load `models/scaler.pkl` → StandardScaler (618 features)
2. Load `models/tfidf_vectorizer.pkl` → TF-IDF vectorizer
3. Load `ICUModel` architecture, load weights from `models/federated_model.pth`
4. Set model to eval() mode
5. Load `models/shap_background.npy` → numpy array (100 × 618)
6. Create `shap.DeepExplainer(model, background_tensor)`
7. Initialize `models/patient_vitals.csv` if not present (20-row default)

### Sidebar Inputs:
- Groq API key (if not in .env)
- Clinical Notes (text area, placeholder with example text)
- Vital Signs: HR, RR, SpO2, Temp, SBP, DBP, MAP
  (with min/max validation enforced by Streamlit number_input)
- GCS Eye Opening (selectbox: 4/3/2/1 with labels)
- Stress Score (slider: 0–10)
- "Run Prediction" button

### On Button Click — Step-by-Step Pipeline:

**Step 1: Input Validation**
Check each vital against VALID_RANGES. If any is outside range, show error and
call st.stop(). Check API key is present.

**Step 2: Sliding Window Update**
- Load patient_vitals.csv
- Append new row with current timestamp and 7 vitals
- Keep tail(20) → drop oldest row
- Save back to CSV

**Step 3: Trend Features**
Compute 9 trend features from the updated 20-row window:
HR_mean, HR_std, RR_mean, SpO2_mean, SpO2_min, Temp_mean, SBP_mean,
DBP_mean, MAP_mean

**Step 4: Latest Features**
Create dict with latest_HR, latest_RR, latest_SpO2, latest_Temp, latest_SBP,
latest_DBP, latest_MAP, GCS_eye_opening, stress_score

**Step 5: TF-IDF Transform**
Apply saved tfidf.transform([clinical_note]) → sparse matrix → dense DataFrame
with 600 columns (one per vocabulary term)

**Step 6: Feature Combination**
Concatenate: trend_df + latest_df + tfidf_df → 618-column DataFrame

**Step 7: Feature Alignment**
For any of the 618 expected columns missing, set to 0.
Reorder all columns to exact order in scaler.feature_names_in_.

**Step 8: Scaling**
Apply scaler.transform(final_df) → scaled numpy array (1 × 618)

**Step 9: Model Inference**
Convert scaled array to torch.FloatTensor.
Run through ICUModel in eval() mode with torch.no_grad().
Get scalar output → raw SOFA prediction.
Clip to [0, 24].

**Step 10: Risk Classification**
- raw_pred < 5 → Low Risk 🟢
- raw_pred < 10 → Moderate Risk 🟡
- raw_pred ≥ 10 → High Risk 🔴

**Step 11: Alert Display**
Show error/warning banner if Moderate or High risk.

**Step 12: SHAP Computation**
explainer.shap_values(X_tensor) → 618 SHAP values
Filter to clinical keywords → top 7 features by |impact|
Generate clinical interpretations → generate key risk factors

**Step 13: Trend Analysis**
For each of 7 vitals:
- get_trend(series) → direction (increasing/stable/decreasing)
- classify_range(mean, lo, hi) → status (low/normal/high)
- Combine: "SpO2 → low & decreasing"

**Step 14: Build LLM Prompt**
Assemble 7-section prompt with SOFA, notes, vitals, CV features,
key risk factors (from SHAP), feature explanations (from SHAP), trend summary.

**Step 15: LLM Self-Consistency**
Call Groq 3 times with the prompt (temperature=0.7).
Compute TF-IDF cosine similarity → consistency score.
Determine reliability label.

**Step 16: Display (3 Tabs)**

**Tab 1 — Risk Assessment:**
- 4 metric cards: SOFA Score, Risk Level, Severity %, Readings in Window
- Progress bar (SOFA/24)
- Plotly line chart: last 20 readings for all 7 vitals
- Trend summary text

**Tab 2 — Explainability:**
- SHAP table: feature, absolute impact, direction (↑/↓)
- Clinical interpretations (bullet list)
- Key risk factors (warning cards)

**Tab 3 — AI Clinical Report:**
- Consistency score + reliability metric side by side
- Reliability status banner (green/yellow/red)
- Main LLM response (Response 1)
- Expandable section with all 3 responses
- Clinical disclaimer

---

## 19. COMPLETE FOLDER STRUCTURE

```
icu_monitor/
│
├── app.py                      ← Main Streamlit application (the demo)
├── model_utils.py              ← ICUModel class + training/eval utilities
├── train_federated.py          ← FL retraining script (simulation, local)
├── server.py                   ← Real Flower FL server
├── client.py                   ← Real Flower FL client
├── requirements.txt            ← All Python dependencies
├── .env                        ← Groq API key (not committed to version control)
├── .env.example                ← Template showing required env variables
│
├── models/                     ← All saved ML artifacts
│   ├── federated_model.pth     ← Trained PyTorch DNN weights
│   ├── scaler.pkl              ← StandardScaler fitted on 618 features
│   ├── tfidf_vectorizer.pkl    ← TF-IDF vectorizer (600 features, ngram 1-2)
│   ├── feature_columns.pkl     ← Ordered list of 618 feature column names
│   ├── shap_background.npy     ← 100 background samples for SHAP (100 × 618)
│   └── patient_vitals.csv      ← Sliding window vitals history (20 rows)
│
├── data/                       ← Hospital client datasets for FL
│   ├── client_0.csv            ← Hospital 0: ~15,889 rows × 619 cols
│   ├── client_1.csv            ← Hospital 1: ~15,890 rows × 619 cols
│   └── client_2.csv            ← Hospital 2: ~16,372 rows × 619 cols
│
└── notebooks/
    └── federated_learning.ipynb ← Original training notebook (BigQuery/Colab)
                                    Produced all artifacts in models/ and data/
```

**Files NOT in this folder (kept in original capstone/ root):**
- `trial/` — old ensemble app (reference only, superseded)
- `federated learning/` — old FL app (reference only, superseded)
- `model training/` — all original notebooks (reference only)
- `project_context_1-6.md` — ChatGPT session context files (not project code)
- `architecture.png` — system architecture diagram
- `project_report.pdf` — project report / presentation PDF
- `sql_queries.pdf` — all 12 BigQuery SQL queries as PDF

---

## 20. FILE-BY-FILE DESCRIPTION

### `app.py`
The complete Streamlit web application. This is the primary demo file.
Contains the entire inference pipeline from user input to LLM output.
Imports from: model_utils.py, .env, models/ folder.
Run with: `streamlit run app.py`

### `model_utils.py`
Shared utility module imported by app.py, train_federated.py, server.py, client.py.
Contains:
- `ICUModel` class — the PyTorch DNN architecture (618→256→128→64→1)
- `ICUDataset` class — PyTorch Dataset for DataLoader
- `train_model()` — mini-batch training function (Adam, MSE loss)
- `evaluate_model()` — returns MSE, MAE, R² metrics
- `get_weights()` — extracts model weights as list of numpy arrays (for Flower)
- `set_weights()` — loads list of numpy arrays back into model state_dict
- `get_trend()` — computes trend direction from a series
- `classify_range()` — classifies a value as low/normal/high

### `train_federated.py`
Standalone FL retraining script. Run locally to retrain the model.
Uses Flower simulation (fl.simulation.start_simulation) — no separate terminals.
Steps: Load client CSVs → Save SHAP background → Run FL simulation →
       Save best model → Print evaluation metrics.
Output: updates `models/federated_model.pth` and `models/shap_background.npy`
Run with: `python train_federated.py`

### `server.py`
Real Flower FL server. Must be run first (before clients).
Waits for exactly 3 clients to connect on 127.0.0.1:8080.
Runs 20 FL rounds.
Saves model after every round (latest) and whenever a new best is found.
Uses custom SaveBestStrategy extending FedAvg.
Run with: `python server.py`

### `client.py`
Real Flower FL client representing one hospital.
Takes --client_id argument (0, 1, or 2).
Loads its private dataset from data/client_<id>.csv.
Normalizes SOFA target. Splits into local train/val.
Trains locally for 10 epochs per round.
Sends only weights to server (NO patient data transmitted).
Run with: `python client.py --client_id 0` (and 1, and 2)

### `requirements.txt`
```
streamlit>=1.35.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
shap>=0.44.0
groq>=0.9.0
python-dotenv>=1.0.0
plotly>=5.18.0
flwr>=1.8.0
```

### `models/federated_model.pth`
PyTorch state_dict. Contains the trained weights of ICUModel.
Produced by: notebooks/federated_learning.ipynb (original)
Can be updated by: python train_federated.py (local retraining)

### `models/scaler.pkl`
sklearn StandardScaler fitted on the full 618-feature training set.
Knows the exact 618 feature names and their order (scaler.feature_names_in_).
CANNOT be retrained locally (requires raw BigQuery data).
Used by: app.py (for inference scaling), client.py (for feature alignment)

### `models/tfidf_vectorizer.pkl`
sklearn TfidfVectorizer fitted on the MIMIC clinical notes corpus.
Parameters: max_features=600, ngram=(1,2), max_df=0.9, min_df=10.
CANNOT be retrained locally (requires raw BigQuery notes).
Used by: app.py (to transform clinical note text)

### `models/feature_columns.pkl`
Python list of 618 column names in the correct order.
Used as a cross-check for feature alignment.

### `models/shap_background.npy`
100 randomly sampled training rows as a numpy array (100 × 618).
Used by: shap.DeepExplainer as background reference.
Generated from data/client_0/1/2.csv by the SHAP background generation script
or by train_federated.py.

### `models/patient_vitals.csv`
20-row CSV with columns: time, HR, RR, SpO2, Temp, SBP, DBP, MAP.
Maintains the sliding window of the last 20 vital readings.
Updated every time "Run Prediction" is clicked.
If deleted, a default 20-row dataset is recreated automatically.

### `data/client_0/1/2.csv`
Each file: ~15,889–16,372 rows × 619 columns (618 features + sofa_score).
The 618 features are ALREADY SCALED (StandardScaler was applied during
the original BigQuery notebook). The sofa_score column is raw (0–24).
These are the "hospital datasets" for federated learning.

### `notebooks/federated_learning.ipynb`
The original Google Colab notebook that:
- Connected to BigQuery and loaded the final dataset
- Trained TF-IDF and scaled the features
- Split data into 3 hospital datasets
- Ran Flower FL simulation (10 rounds)
- Saved all model artifacts to Google Drive
This notebook CANNOT be run locally (requires BigQuery auth and Colab).
It is kept for reference to understand how the artifacts were produced.

---

## 21. ALL FINALIZED TECHNICAL PARAMETERS

These parameters are FIXED in the current implementation. Do not change them
unless explicitly re-training from scratch.

| Parameter | Value | Where Used |
|---|---|---|
| Feature vector size | 618 | model, scaler, all files |
| TF-IDF max_features | 600 | tfidf_vectorizer.pkl |
| TF-IDF ngram_range | (1, 2) | tfidf_vectorizer.pkl |
| TF-IDF max_df | 0.9 | tfidf_vectorizer.pkl |
| TF-IDF min_df | 10 | tfidf_vectorizer.pkl |
| Sliding window size | 20 | app.py, patient_vitals.csv |
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
| DNN architecture | 618→256→128→64→1 | ICUModel in model_utils.py |
| FL rounds | 20 | train_federated.py, server.py |
| Epochs per round | 10 | train_federated.py, client.py |
| Batch size | 64 | train_model() in model_utils.py |
| Learning rate | 0.001 | train_model() in model_utils.py |
| SHAP background size | 100 | shap_background.npy |
| LLM model | llama-3.3-70b-versatile | app.py |
| LLM temperature | 0.7 | app.py (for self-consistency) |
| LLM max_tokens | 800 | app.py |
| Self-consistency calls | 3 | get_multiple_llm_responses() |
| High reliability threshold | ≥ 0.85 | compute_consistency() |
| Moderate reliability threshold | 0.65–0.85 | compute_consistency() |
| GCS Eye range | 1–4 | sidebar input, LLM prompt |
| Stress Score range | 0–10 | sidebar input, LLM prompt |
| Clients per FL round | 3 | server.py, train_federated.py |
| Server address | 127.0.0.1:8080 | server.py, client.py |
| SHAP top features | 7 | app.py |

---

## 22. MODEL PERFORMANCE METRICS

### Current Federated DNN (federated_model.pth):
Evaluated in notebooks/federated_learning.ipynb after 10 FL rounds:
- MAE: ~2.18 SOFA points
- R²: ~0.18

NOTE: This R² is lower than the ensemble because:
1. Only 10 FL rounds (not enough convergence)
2. Only 3 epochs per client per round (undertrained)
3. The model has less capacity than the optimized ensemble

### After Retraining with train_federated.py (20 rounds, 10 epochs):
Expected improvement to:
- MAE: ~1.90–2.00 SOFA points
- R²: ~0.30–0.38

### Ensemble Model (Final_ICU.ipynb — not used in final app):
- LightGBM: MAE=1.87, RMSE=2.46, R²=0.397
- XGBoost: MAE=1.86, RMSE=2.46, R²=0.396
- Random Forest: MAE=1.99, RMSE=2.60, R²=0.323
- Ensemble (0.52 LGB + 0.48 XGB): MAE=1.86, RMSE=2.44, R²=0.403

### Why R² ~0.35 is Acceptable:
ICU deterioration prediction is inherently difficult because:
- SOFA is determined by many factors the model doesn't see (lab values, medications)
- MIMIC-III data has inherent noise and missing patterns
- Even clinical experts cannot perfectly predict SOFA from vitals alone
- The model's purpose is early WARNING, not perfect SOFA prediction

---

## 23. TECHNOLOGY STACK

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML Framework | PyTorch ≥ 2.0 |
| Federated Learning | Flower (flwr) ≥ 1.8 |
| Explainability | SHAP ≥ 0.44 |
| Data Processing | Pandas, NumPy |
| Traditional ML (reference) | scikit-learn, LightGBM, XGBoost |
| NLP | scikit-learn TfidfVectorizer |
| LLM Provider | Groq (llama-3.3-70b-versatile) |
| Web Framework | Streamlit ≥ 1.35 |
| Visualization | Plotly |
| Dataset | MIMIC-III |
| Data Access | Google BigQuery |
| Training Environment | Google Colab |
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

### Run the Streamlit App (main demo):
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Retrain the Federated Model (local simulation):
```bash
python train_federated.py
# Takes 10–30 minutes depending on machine
# Updates models/federated_model.pth and models/shap_background.npy
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

### Reset patient vitals history (start fresh sliding window):
```bash
rm models/patient_vitals.csv
# The app will recreate it with default values on next run
```

---

## 25. PROJECT EVOLUTION HISTORY

**Phase 1 — Architecture and Design:**
Defined the system architecture, identified MIMIC-III as dataset, chose Google
BigQuery for data access. Drew the system architecture diagram (architecture.png).
Defined UML diagrams and requirements.

**Phase 2 — BigQuery Preprocessing:**
Ran all 12 SQL queries to extract and transform MIMIC-III data into ML-ready
format. Produced ml_dataset_final table with 60,189 rows.

**Phase 3 — Ensemble ML (trial/):**
Trained LightGBM + XGBoost + Random Forest on the processed data.
Used SHAP TreeExplainer for explainability.
Integrated Groq LLM with 3-response self-consistency check.
Implemented sliding window inference.
Built initial Streamlit app (trial/ICU_Inference.py).
Performance: R²=0.40, MAE=1.86

**Phase 4 — ClinicalBERT Experiment:**
Explored replacing TF-IDF with ClinicalBERT embeddings.
Result: more complex, similar performance.
Decision: stayed with TF-IDF.

**Phase 5 — Federated Learning Integration:**
Researched Flower framework. Discovered that ensemble models cannot be
federated natively (no weight averaging possible for trees).
Switched to PyTorch DNN as the federated model.
Trained DNN via Flower simulation (federated_learning.ipynb).
Implemented real server.py + client.py.
Performance: R²=0.18, MAE=2.18 (undertrained, needs more rounds)

**Phase 6 — Final Unified Project (icu_monitor/):**
Created clean final project folder with all components integrated:
- Improved FL training (20 rounds, 10 epochs, mini-batch)
- SHAP DeepExplainer for PyTorch model
- LLM self-consistency check brought back
- Full 3-tab Streamlit UI
- Input validation, alert system
- Fixed SOFA ×24 bug (model outputs raw SOFA, not normalized)
- Generated shap_background.npy from existing client CSVs

---

## 26. NOVEL CONTRIBUTIONS

1. **Multimodal fusion of IoMT + CV + Clinical NLP in a unified risk predictor.**
   Most existing CDSS systems use only vital signs. Combining behavioral CV
   features (GCS, stress) and clinical text into the same ML model is novel.

2. **Trend-based temporal feature engineering with sliding window.**
   Using statistical features over the last 20 readings rather than just the
   latest values captures physiological trajectory — crucial for early detection.

3. **SOFA score prediction from multimodal bedside data.**
   Using the internationally validated SOFA score as the prediction target
   (rather than a binary label) gives a continuous severity measure.

4. **Federated Learning for privacy-preserving ICU monitoring.**
   Training a neural network collaboratively across simulated hospital nodes
   using the Flower framework, without any patient data leaving any node.

5. **LLM Self-Consistency Check as reliability validation.**
   Calling the LLM 3 times and measuring agreement via TF-IDF cosine similarity
   to produce a quantitative reliability score for AI outputs. This addresses
   LLM hallucination concerns in a clinical setting.

6. **SHAP DeepExplainer on a Federated DNN.**
   Providing feature-level explainability for a privacy-preserving federated
   neural network using SHAP. This combination is not common in existing work.

---

## 27. DESIGN CONSTRAINTS AND ASSUMPTIONS

### Constraints:
- No patient data can leave hospitals → Federated Learning
- Must run on standard hardware (no GPU required for inference)
- Must be explainable to clinicians → SHAP + LLM
- MIMIC-III only (no real-time patient data access)
- TF-IDF vectorizer and StandardScaler cannot be retrained locally
  (require original BigQuery data)

### Assumptions:
- Vital signs are continuously available and reasonably accurate
- A 20-reading sliding window captures sufficient temporal context
- SOFA score from MIMIC-III `mimiciii_derived.sofa` is the correct target label
- The simulated 3-hospital FL split is a valid proxy for real multi-hospital FL
- The pre-trained Groq LLM provides clinically reasonable reasoning
- Federated nodes behave honestly (no adversarial or corrupted parameters)

### Limitations:
- R² of ~0.18–0.35 means the model is not highly accurate
  (acceptable given the difficulty of the task and the accepted norm in literature)
- The "hospitals" are artificial splits of one dataset, not real institutions
- The CV features (GCS, stress) are manually entered, not from a real camera
- The sliding window resets if patient_vitals.csv is deleted
- The model cannot retrain on new patient data automatically
- LLM outputs depend on prompt quality and the Groq API being available

---

## 28. VIVA PREPARATION — KEY POINTS

### Q: Why did you choose MIMIC-III instead of MIMIC-IV?
MIMIC-III is more stable for preprocessing and has well-tested derived tables
including pre-computed SOFA scores in mimiciii_derived.sofa. MIMIC-IV was
available but MIMIC-III was sufficient for our requirements.

### Q: Why TF-IDF instead of ClinicalBERT?
We explored ClinicalBERT (New_ICU_ClinicalBERT.ipynb). It was more complex
to integrate and produced similar performance while requiring significantly more
compute and a separate embedding step. TF-IDF is simpler, faster, directly
produces numerical features usable by any ML model, and is interpretable.

### Q: Why PyTorch DNN instead of LightGBM for the federated model?
Federated Learning with FedAvg works by averaging numerical weight matrices.
Tree-based models (LightGBM, XGBoost) do not have weight matrices — each tree
has a different structure per hospital. You cannot average trees. Neural
networks are natural candidates for FedAvg because all parameters are numerical.

### Q: Is your Federated Learning real or simulated?
The FL protocol is real: Flower server/client, TCP sockets, weight-only
transmission, FedAvg aggregation. What is simulated is the hospital data
(artificially split from MIMIC-III). This is the standard approach in all
published FL research — true multi-hospital data would require IRB approval
from multiple institutions.

### Q: Why is R² only 0.18–0.35? Is that acceptable?
SOFA prediction from vital signs alone is inherently difficult. SOFA depends on
organ function metrics (bilirubin, creatinine, PaO2) that we don't have in our
feature set. Published research on vital-sign-only SOFA prediction reports
similar R² values. The goal is early WARNING, not perfect prediction.

### Q: What is the self-consistency check and why is it important?
LLMs can hallucinate — produce confident but wrong or inconsistent outputs.
In a clinical setting, an unreliable AI recommendation could influence wrong
decisions. We send the same prompt to the LLM 3 times and measure how similar
the responses are using cosine similarity on TF-IDF vectors. A high consistency
score (≥0.85) means the LLM is producing stable, reproducible reasoning —
a proxy for reliability.

### Q: How does SHAP work with a neural network?
We use shap.DeepExplainer, which uses backpropagation-based attribution
(DeepLIFT algorithm) to compute how much each of the 618 input features
contributed to the SOFA prediction for a specific patient. Unlike TreeExplainer
(which works only with tree models), DeepExplainer is designed for deep learning.
It requires a background dataset (100 reference samples) to compute baseline
attribution.

### Q: Why does the system not diagnose diseases?
Medical diagnosis is a regulated clinical activity. An AI system that diagnoses
would require clinical validation, regulatory approval (FDA, CE marking), and
malpractice liability coverage. Our system only predicts severity risk and
generates explanations — acting as a decision SUPPORT tool, not a decision MAKER.

### Q: What happens to patient data in Federated Learning?
Patient data never leaves the hospital. Each hospital trains the model locally.
Only the model's weight matrices (numpy arrays of floating point numbers)
are sent to the aggregation server. These weights contain no patient-identifiable
information. This is the fundamental privacy guarantee of Federated Learning.

---

*End of SKILL.md — Complete Project Context Document*
*Last updated: July 2026*

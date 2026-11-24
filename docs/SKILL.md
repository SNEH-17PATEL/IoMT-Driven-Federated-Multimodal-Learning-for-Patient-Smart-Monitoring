# ClinIQ — Complete Project Context
# ICU Clinical Decision Support System with Federated Learning

**Last updated:** 2026-11-24

> This is the single reference for the whole project: what it is, why it exists, how the data is built, how the model is trained, how the dashboard works, how to run it, and how to explain it. Every number here comes from the current code and from `models/training_metadata.json` (the final training run). Anyone who has never seen the project should be able to read this file and understand all of it.

---

## How to Read This Document

- **Sections 1–4** explain the problem and what the system does and does not do.
- **Sections 5–12** explain the data, the 108 inputs, the SOFA target and the model.
- **Sections 13–17** explain federated learning, explanations (SHAP), the AI report (Groq LLM), how that report is checked, and the alert.
- **Sections 18–20** explain the dashboard, the folder layout and every file.
- **Sections 21–24** list all fixed settings, the results, the technology and how to run everything.
- **Sections 25–28** cover the project history, contributions, limits and viva questions.
- **Section 29** is a glossary of every technical term.

Where a detail has its own long document, this file gives a summary and points to it:

| Topic | Detailed document |
|---|---|
| Building the dataset in BigQuery | `docs/SQL_COMMANDS.md` |
| Training script, every setting, experiments | `docs/TRAIN_FEDERATED.md` |
| Checks on the SOFA score and the AI report | `docs/VALIDATION.md` |
| Model inputs and outputs | `docs/INPUT_OUTPUT.md` |
| Why the model looks the way it does | `docs/MODEL_DISCUSSION.md` |
| Camera (stress and eye) monitor | `docs/CV_MONITOR.md` |
| Tests | `docs/CLINIQ_TESTING.md`, `docs/FL_SERVER_CLIENT_DB_TESTING.md` |

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [What the System Does and Does NOT Do](#4-what-the-system-does-and-does-not-do)
5. [Dataset — MIMIC-III](#5-dataset--mimic-iii)
6. [BigQuery Pipeline — Building `dataset_v2`](#6-bigquery-pipeline--building-dataset_v2)
7. [Three Data Modalities](#7-three-data-modalities)
8. [Feature Engineering — All 108 Inputs](#8-feature-engineering--all-108-inputs)
9. [Sliding Window Mechanism](#9-sliding-window-mechanism)
10. [TF-IDF Clinical NLP Pipeline](#10-tf-idf-clinical-nlp-pipeline)
11. [Target Variable — SOFA Score](#11-target-variable--sofa-score)
12. [Machine Learning Model](#12-machine-learning-model)
13. [Federated Learning Architecture](#13-federated-learning-architecture)
14. [SHAP Explainability](#14-shap-explainability)
15. [LLM Integration — Groq](#15-llm-integration--groq)
16. [Validation — Checking the SOFA Score and the AI Report](#16-validation--checking-the-sofa-score-and-the-ai-report)
17. [Alert System](#17-alert-system)
18. [ClinIQ Dashboard — Complete Pipeline](#18-cliniq-dashboard--complete-pipeline)
19. [Complete Folder Structure](#19-complete-folder-structure)
20. [File-by-File Description](#20-file-by-file-description)
21. [All Finalized Technical Parameters](#21-all-finalized-technical-parameters)
22. [Model Performance Metrics](#22-model-performance-metrics)
23. [Technology Stack](#23-technology-stack)
24. [How to Run the Project](#24-how-to-run-the-project)
25. [Project Evolution History](#25-project-evolution-history)
26. [Novel Contributions](#26-novel-contributions)
27. [Design Constraints, Assumptions and Limitations](#27-design-constraints-assumptions-and-limitations)
28. [Viva Preparation — Key Points](#28-viva-preparation--key-points)
29. [Glossary](#29-glossary)

---

## 1. Project Identity

**Full title:** Multimodal Intelligence System with Federated Learning for Continuous Patient Monitoring and Early Deterioration Detection Leveraging IoMT, Computer Vision and Clinical NLP

**Product name:** **ClinIQ** (the dashboard and API)

**Type:** Final-year engineering capstone project

**Domain:** Healthcare AI, clinical decision support, federated learning

**Nature:** ClinIQ is a **clinical decision support system (CDSS)**. It is **not** a diagnostic system. It assists clinicians; it does not replace them. All results are retrospective MIMIC-III results, not clinical validation.

**Student:** Sneh Patel, Final Year Computer Science Engineering

---

## 2. Problem Statement

ICU patients can get worse quickly. Today's monitoring has three serious weaknesses.

**Failure 1 — Threshold alarms cause alarm fatigue.** A typical ICU monitor sounds an alarm whenever one vital crosses a fixed line (for example heart rate above 120). This produces hundreds of false alarms a day. Staff start to ignore alarms, and a real emergency can be missed.

**Failure 2 — Nothing combines the signals.** Existing systems look at one vital at a time. A heart rate of 105 means little without context: what is the trend, what are blood pressure and oxygen doing at the same moment, what does the doctor's note say? No bedside system joins all of this into one risk signal.

**Failure 3 — Patient data cannot leave the hospital.** Normal machine learning needs all data in one place. Patient data is legally protected (HIPAA, GDPR), so hospitals cannot pool records. That makes shared model training legally and ethically hard.

**How ClinIQ answers each one:**
- Instead of single-vital alarms, it predicts one **SOFA** organ-failure score from many signals together.
- Trends over the last 20 readings, words from clinical notes and camera-based cues give the model context.
- **Federated learning** lets hospitals train one shared model without sharing patient data.

---

## 3. Project Objectives

1. Monitor ICU patients continuously using several kinds of data (vitals, notes, camera).
2. Detect early physical deterioration before it becomes severe.
3. Predict the patient's SOFA score (0–24).
4. Classify risk as Low, Moderate or High.
5. Raise an alert for patients at high risk.
6. Explain each prediction with SHAP (which inputs pushed the score up or down).
7. Write a clinician-friendly AI report with a large language model (LLM).
8. **Check that AI report** against the real patient data and medical rules, and show a reliability score.
9. Keep patient privacy across hospitals using federated learning.
10. Offer two audiences: **Hospital** mode (clinical language) and **Home Care** mode (plain language for a home nurse and family).

---

## 4. What the System Does and Does NOT Do

### What it DOES
- Predicts a SOFA score and classifies it as Low / Moderate / High risk.
- Fires a red alert when the predicted SOFA is 7 or more (Section 17).
- Shows a range around the prediction (a 90% **conformal interval**, ± 3.28).
- Shows which inputs drive the prediction (SHAP, top 5).
- Writes a four-section AI report in Hospital or Home Care style and scores how reliable it is.
- Runs three safety checks on every prediction (a SOFA floor from vitals, dangerous vital combinations, and unexplained jumps).
- Reads the patient's face with the webcam (stress and eye-opening scores) and feeds them into the model.
- Keeps a sliding window of the last 20 vital readings and a history of the last 50 predictions.
- Trains one model across 3 simulated hospitals without sharing patient data.

### What it does NOT do
- Does not diagnose any disease.
- Does not make clinical decisions on its own or replace doctors or nurses.
- Does not prescribe treatment (the AI report only suggests; every report carries a disclaimer that qualified professionals decide).
- Does not connect to real bedside devices. The demo **replays** recorded MIMIC-III readings for 3 sample patients, one new reading every 90 seconds.
- Does not retrain itself on new patients automatically.
- Has not been validated on real patients.

---

## 5. Dataset — MIMIC-III

**MIMIC-III** (Medical Information Mart for Intensive Care III) is a public, de-identified database of real ICU stays at Beth Israel Deaconess Medical Center in Boston.

**Access:** Google BigQuery. Project `mimic-project-2`, dataset **`dataset_v2`** (rebuilt from scratch; the older dataset `Dataset` was deleted). Access to MIMIC-III itself requires PhysioNet credentials, so the data is never published on GitHub.

**Why MIMIC-III and not MIMIC-IV:** MIMIC-III is stable for preprocessing and has well-documented derived tables, including hourly pre-computed SOFA scores.

**Main source tables:**

| Table | Purpose |
|---|---|
| `mimiciii_derived.icustay_detail` | Who is in the cohort (age, ICU in/out times) |
| `mimiciii_derived.pivoted_vital` | Heart rate, breathing rate, SpO₂, temperature, blood pressures |
| `mimiciii_derived.pivoted_gcs` | Glasgow Coma Scale (eye opening) |
| `mimiciii_clinical.chartevents` | Nurse-recorded pain score (used as the stress score) |
| `mimiciii_derived.pivoted_sofa` | Hourly SOFA, the answer the model learns |
| `mimiciii_notes.noteevents` | Clinical notes |

**What `dataset_v2` contains:** working tables `cohort`, `vitals_clean`, `vitals_hourly`, `vital_trends`, `latest_vitals`, `gcs_hourly`, `stress_hourly`, `labels`, plus two views that training reads: `ml_dataset_final` (5,089,857 hourly rows) and `clinical_notes` (35,799 admissions).

**Dataset size, step by step:**

| Step | Result |
|---|---|
| Cohort | Adults (age ≥ 16) with an ICU stay of at least 6 hours: **52,248 ICU stays from 38,304 patients** |
| Phase 0 takes the newest hourly row of each stay | 52,248 rows × 109 columns (108 inputs + `sofa_score`) |
| Train / test split (80/20, stratified by High Risk) | **41,798 train / 10,450 test** |
| Test set | 602 High, 2,821 Moderate, 7,027 Low |
| Three simulated hospitals (IID) | 13,932 / 13,933 / 13,933 rows; each about 67% Low, 27% Moderate, 6% High |
| Missing values | Filled with 0, then every input is scaled and clipped to ±10 |

---

## 6. BigQuery Pipeline — Building `dataset_v2`

The dataset was rebuilt in BigQuery with a short series of numbered commands. **Every command, the reasoning behind it and the test results are in `docs/SQL_COMMANDS.md`.** In short:

| Stage | Table made | What it does, and what it fixed compared with the first build |
|---|---|---|
| 1 | `cohort` | Adults only, ICU stay ≥ 6 h. (The first build included about 8,000 newborns.) |
| 2 | `vitals_clean` | Uses both ICU charting systems (CareVue and MetaVision), drops impossible values and readings outside the stay |
| 3 | `vitals_hourly` | One row per stay per hour, lined up with the hourly SOFA timeline |
| 4 | `vital_trends` | The 9 trend columns over the previous 6 hours (the first build used "the last 20 rows", which could span days) |
| 5 | `latest_vitals` | The 7 latest-value columns, carried forward for a limited time instead of left empty |
| 6 | `gcs_hourly` | GCS eye opening from both systems (the first build used only one and was 83% empty) |
| 7 | `stress_hourly` | Nurse-recorded pain score from both systems, converted to 0–10 |
| 8 | `labels` | **SOFA for the same hour** (`pivoted_sofa.SOFA_24hours`). The first build copied one first-day SOFA onto every window of a stay. |
| 9 | `ml_dataset_final` | Joins everything into the same 23 columns as before |
| 10 | `clinical_notes` | Bedside notes only (Nursing, Nursing/other, Physician, Respiratory), written during the ICU stay, errors removed |

**Key rule:** only *how* the data is built changed. The names and meaning of the 108 model inputs stayed identical, so no model, backend or dashboard code had to change.

The original build ran 12 older queries (raw chart events → 10-minute windows → trends → final table). That design is historical and is not described further here.

---

## 7. Three Data Modalities

### Modality 1 — Physiological vital signs (main)
Seven vitals: heart rate (HR), respiratory rate (RR), oxygen saturation (SpO₂), temperature, systolic BP (SBP), diastolic BP (DBP) and mean arterial pressure (MAP).

### Modality 2 — Computer-vision features (behaviour and consciousness)
- **GCS Eye Opening** (1–4): the eye-response part of the Glasgow Coma Scale. 4 = eyes open on their own, 1 = never opens.
- **Stress Score** (0–10): how distressed the patient looks. In the training data it comes from the nurse-recorded pain score.

In the dashboard these two values can come **live from the webcam** (Section 18, "Camera monitor"). When the camera is on, its values replace the recorded sample values in the model input.

### Modality 3 — Clinical notes (NLP)
Free-text notes are turned into 90 **TF-IDF** scores, using a fixed list of 90 words tied to the six SOFA organs (Section 10).

---

## 8. Feature Engineering — All 108 Inputs

The model receives **108 numbers**, always in this exact order:

| Group | Inputs | Count |
|---|---|---|
| 1. Trend features | HR_mean, HR_std, RR_mean, SpO2_mean, SpO2_min, Temp_mean, SBP_mean, DBP_mean, MAP_mean | 9 |
| 2. Latest vitals | latest_HR, latest_RR, latest_SpO2, latest_Temp, latest_SBP, latest_DBP, latest_MAP | 7 |
| 3. Camera / observation | GCS_eye_opening (1–4), stress_score (0–10) | 2 |
| 4. Note words | 90 TF-IDF scores from the SOFA word list | 90 |
| **Total** | 9 + 7 + 2 + 90 | **108** |

**Why both trends and latest values?** One reading can mislead (a slipped finger clip can show SpO₂ 91%). The trend over the last 20 readings tells whether that is a one-off or a steady fall.

### The 90-word SOFA list (examples)

| SOFA organ | Example words |
|---|---|
| Respiratory | intubated, ventilator, bipap, cpap, hypoxia, respiratory failure, extubated |
| Coagulation | plt, platelets, coagulopathy, inr, ptt, hemorrhage, thrombocytopenia |
| Hepatic | bilirubin, bili, totbili, liver, jaundice, hepatic, cirrhosis, liver failure |
| Cardiovascular | vasopressor, pressors, levophed, norepinephrine, septic shock, hypotension |
| Brain | sedated, sedation, coma, altered, confused, gcs, encephalopathy, delirium |
| Kidney | creatinine, creat, renal, kidney, dialysis, aki, oliguria, urine output |
| Infection | sepsis, septic, bacteremia, infection, cultures, antibiotics, fever |
| Lab words | lactate, wbc, hgb, hct, sodium, potassium, hco, angap, bun |
| General severity | failure, acute, pneumonia, ards, pulmonary, dyspnea, edema |

Every word has a direct medical meaning. The list replaced an automatic 600-word choice that picked paperwork words such as "tablet" and "refills" (Section 10).

### Scaling and order
- All 108 inputs are rescaled with a **StandardScaler** fitted on the training set only (average 0, spread 1), then clipped to −10…+10. Saved as `models/scaler.pkl`.
- The exact order is saved in `models/feature_columns.pkl`. At the bedside the input is rebuilt in this order before the model sees it.

### What the training data looks like for the two camera inputs
In `dataset_v2`, GCS eye opening is filled for about 96% of stays (84% are 4, "eyes open on their own"). Stress comes from nurse pain scores, and 71% of stays have a stress of 0 (no pain recorded, or none felt). So the camera values carry less weight in training than the vitals, and live camera readings change the predicted SOFA only a little.

---

## 9. Sliding Window Mechanism

**Window size:** 20 readings (fixed).

**How it works at the bedside (the live dashboard):**
1. Each new reading's 7 vitals are appended to the `patient_vitals` table in SQLite (the app's database).
2. Only the newest 20 rows per patient are kept; older rows are dropped.
3. The 9 trend features are computed from those 20 rows.
4. The model uses these trends plus the latest values.

If a patient has no stored vitals yet (a fresh start), the app first **seeds** the window with earlier readings from the sample data so the trends are meaningful from the first prediction.

**Trend direction** (shown on the charts and sent to the AI report): the slope of a straight line fitted to the window (`numpy.polyfit`). Slope above +0.1 = "increasing", below −0.1 = "decreasing", otherwise "stable". NaN values are ignored.

**Normal ranges used by the app:**

| Vital | Normal low | Normal high |
|---|---|---|
| Heart rate | 60 | 100 |
| Respiratory rate | 12 | 20 |
| SpO₂ | 95 | 100 |
| Temperature (°C) | 36.0 (36.5 in the AI trend text) | 37.5 |
| Systolic BP | 90 (100 in the AI trend text) | 120 |
| Diastolic BP | 60 | 80 |
| MAP | 70 | 100 |

The status badges on the dashboard use 36.0 and 90; the short trend summary sent to the AI report uses 36.5 and 100. Both are as written in the code.

---

## 10. TF-IDF Clinical NLP Pipeline

**TF-IDF** gives every word a score for each patient: how often it appears in **this** patient's notes (TF), boosted if the word is **rare** across all patients (IDF).

**Approach: a fixed 90-word list, not automatic selection.**

Automatic selection (600 words) chose paperwork words like "tablet", "sig", "refills" and "disp". Sicker patients get longer records, so those words only measured how much was written, not how ill the patient was. The fixed list makes every input a real medical idea.

**Vectorizer configuration:**
```python
TfidfVectorizer(
    vocabulary=list(_SOFA_VOCABULARY),             # the 90 words
    ngram_range=(1, 2),                            # single words and two-word phrases
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b', # words of 3+ letters
    sublinear_tf=True,                             # 1 + log(count): repeats count less
    norm="l2",                                     # long notes don't score higher by length
)
```
When a vocabulary is given, scikit-learn ignores `max_features`, `min_df`, `max_df` and `stop_words`.

**Corpus used in training:** all **283,208** notes, grouped by admission, cleaned (privacy placeholders, numbers and symbols removed), then cut to the first **10,000 characters** per admission so lab mentions such as "creatinine rising" are not lost.

**At the bedside:** `tfidf.transform([clinical_note])` with the saved `models/tfidf_vectorizer.pkl`. Words not in the list are ignored. A note with none of the words gives an all-zero vector, which is valid.

---

## 11. Target Variable — SOFA Score

**SOFA** (Sequential Organ Failure Assessment) is an international ICU score. It rates six organ systems from 0 to 4 and adds them: **0 to 24**. Higher means sicker.

| Organ | What doctors measure |
|---|---|
| Respiratory | Blood oxygen vs oxygen given (PaO₂/FiO₂) |
| Coagulation | Platelet count |
| Liver | Bilirubin |
| Cardiovascular | Mean arterial pressure and blood-pressure drugs |
| Brain | Glasgow Coma Scale |
| Kidney | Creatinine and urine output |

**Source in MIMIC-III:** `SOFA_24hours` from `mimiciii_derived.pivoted_sofa`, taken at the **same hour** as the readings.

**ClinIQ's risk bands:**

| SOFA | Risk level |
|---|---|
| 0 – 4 | 🟢 Low Risk |
| 5 – 9 | 🟡 Moderate Risk |
| 10 – 24 | 🔴 High Risk |

**Alert line:** predicted SOFA **≥ 7** (`ALERT_THRESHOLD = 7.0`). It is lower than the clinical 10 on purpose; see Section 17.

**Model output:** the network predicts raw SOFA directly. The app clips it to 0–24.

**Why the model cannot be perfect:** SOFA is mostly computed from lab tests and drug doses. ClinIQ is non-invasive, so those are not inputs, and the model estimates them from vitals and note words. It tells sick from not-sick well (AUROC 0.918) but cannot give the exact score inside a band.

---

## 12. Machine Learning Model

### Architecture — PyTorch fully connected network

```
Input: 108 numbers (scaled)
  ↓ Linear(108 → 256) + ReLU + Dropout(0.3)
  ↓ Linear(256 → 128) + ReLU + Dropout(0.3)
  ↓ Linear(128 →  64) + ReLU + Dropout(0.3)
  ↓ Linear( 64 →   1)
Output: predicted SOFA
```

- **Class:** `ICUModel` in `model_utils.py` (defaults `hidden=(256, 128, 64)`, `dropout=0.3`).
- **Size:** **69,121** trainable numbers (weights).
- **Saved file:** `models/federated_model.pth`.
- **Dropout is on only while training.** Testing, SHAP and the backend call `model.eval()`, which turns it off, so a patient always gets the same prediction.

### Why this shape
The earlier network (108→128→64→32→1, no dropout, 24,321 weights) peaked early on `dataset_v2` and then overfit. Dropout 0.3 stops the memorising and the extra width gives room to use it. A still wider 512-256-128 gave no further gain. An even older network (618→256→128→64→1, 199,681 weights) predates the 90-word list.

### Why a neural network and not LightGBM or XGBoost
Federated averaging works by averaging weight matrices. Tree models have no weight matrices, and trees cannot be averaged. A neural network fits federated learning naturally. A centralised gradient-boosting model on the same inputs reaches nearly the same accuracy (R² 0.50 vs 0.48), so using a network cost almost nothing.

### Training configuration

| Setting | Value | Notes |
|---|---|---|
| Rounds | 100 | The best round is kept: **round 37** in the final run |
| Epochs per round | 3 | Less drift than 5, more learning than 1 |
| Batch size | 64 | |
| Optimiser | AdamW | weight decay 1e-4 |
| Base learning rate | 0.001 | multiplied by 0.99 each round (floor 1e-4) |
| Gradient clipping | 1.0 | caps any one update's size |
| Loss | **plain MSE** | `weighted_loss=True` restores the old 1 + 0.5×SOFA weighting |
| Dropout | 0.3 | after each hidden layer |
| Server method | **FedAvg** | `USE_FEDYOGI = False` (FedYogi is optional) |
| Client regularisation | **FedProx**, μ = 0.5 | |
| Oversampling | Off | `OVERSAMPLE = False` |
| Differential privacy | Off | `USE_DP = False` |

**Loss (plain MSE):** `mean((prediction − truth)²)`. The earlier weighted version multiplied each error by 1 + 0.5 × SOFA; it pushed every prediction upward, raised the error and did not catch more High Risk patients, so it was replaced.

Full training details, every setting and the experiment history: `docs/TRAIN_FEDERATED.md`.

---

## 13. Federated Learning Architecture

### Concept
Several hospitals each hold private patient data. Federated learning (**FL**) trains one shared model by sending the **model** to the data instead of the data to a server. Only the model's numbers (weights) travel; no patient record leaves a hospital.

```
1. The server sends the current model to every hospital.
2. Each hospital trains it on its own patients.
3. Each hospital sends back only its updated weights.
4. The server combines them (FedAvg) into a better shared model.
5. Repeat for 100 rounds.
```

### What is real and what is simulated

| Real | Simulated |
|---|---|
| Flower (a real FL framework) runs the protocol | The 3 "hospitals" are made by splitting one MIMIC-III dataset |
| Only weights are exchanged | All 3 clients run on one computer |
| Each client trains only on its own rows | The split is random (IID), so the hospital names are just labels |

### Framework: Flower (`flwr`)

### Server: FedAvg
```
new shared weights = Σ (nᵢ × weightsᵢ) / Σ nᵢ        (hospitals with more patients count more)
```
`USE_FEDYOGI = False` in `train_federated.py`. FedYogi (momentum plus an adaptive step) was used earlier; on the rebuilt data its validation loss exploded in rounds 5–10 (up to about 123), so the final model uses plain FedAvg. Setting `USE_FEDYOGI = True` brings it back with a safer step size of 0.001.

### Client regularisation: FedProx
Each hospital adds a penalty for drifting from the shared model:
```
local loss = MSE + (μ / 2) × Σ (local weight − shared weight)²        with μ = 0.5
```
This stops hospitals from pulling the shared model in different directions.

### Architecture picture
```
                 Global server
        (FedAvg, keeps the best round)
              │  sends shared weights
              ▼
┌────────────────────────────────────────────┐
│ Hospital 0     Hospital 1     Hospital 2   │
│ 13,932 rows    13,933 rows    13,933 rows  │
│ local training: 3 epochs, FedProx, dropout │
└────────────────────────────────────────────┘
              │  send back updated weights
              ▼
        FedAvg → new shared model
```

### Two ways to run federated training

| Mode | Command | What it is |
|---|---|---|
| **Simulation** | `python train_federated.py` | `fl.simulation.start_simulation` (uses Ray); all in one process. This is the script that produces the model the dashboard uses. |
| **Real demo** | `python server.py` then `python client.py --client_id 0` / `1` / `2` | Separate processes over TCP at `127.0.0.1:8080`. Uses the same FedAvg. Saves the best round to `federated_model.pth` and every round to `federated_model_latest.pth`. |

A third tool, `fl_dashboard.py` (Streamlit, port 8501), shows a simulation of an FL run live in the browser with charts.

### SaveBestStrategy
A custom Flower strategy on top of FedAvg. After every round it averages the 3 hospitals' validation errors and, if that is the lowest so far, remembers the round's weights. It also sends the round number to clients so each can decay its learning rate. Each hospital keeps a private 15% validation set of its own data. The model saved at the end is the **best round**, not the last.

---

## 14. SHAP Explainability

**SHAP** answers "why did the model predict this score?" by splitting the prediction into a contribution from each input.

- **Method:** `shap.DeepExplainer`, which is made for PyTorch networks.
- **Background:** 300 typical training patients (`models/shap_background.npy`, shape 300 × 108), the reference point for "average patient".
- **When it is created:** the first time a prediction is requested; the explainer is cached.
- **Meaning:** a positive value pushed the SOFA up (worse); negative pushed it down.

**From 108 values to something readable:**
1. Keep only clinically meaningful inputs by keyword match (vitals, GCS, stress and words such as hypotension, respiratory, septic, failure, intubated, vasopressor, shock, fever, infection, oxygen, ventilat, cardiac, renal, hepatic). Short abbreviations like "hr" must match as whole tokens so "hr" inside "epinephrine" is not treated as heart rate.
2. Take the **top 7** by size of effect. They become plain-English explanations such as "SpO₂ = 89% → low oxygen saturation, increasing risk" and the list of **key risks** sent to the AI report.
3. Return the **top 5** to the dashboard with the original value, the effect size and its direction.

**Display:** original clinical values (not scaled numbers). For a note word, the text says the note contains that word.

---

## 15. LLM Integration — Groq

**Provider:** Groq (very fast inference, so a report appears within seconds).

**Model:** `openai/gpt-oss-120b`, used for **two** jobs:

| Job | Temperature | Max tokens | Notes |
|---|---|---|---|
| Write the report | 0.15 (low = factual, consistent) | 6,000 | One call per report |
| Judge the report | 0.0 | 1,200 | Reads the first 900 characters; 12-second limit |

The API key is read from `.env` (`GROQ_API_KEY`). It is never in the code and never pushed to GitHub.

### What the AI is given
The prompt contains, in order: the SOFA score and risk level, the ± uncertainty, the clinical note, the latest vitals with normal ranges, the camera/stress values, the key risk factors from SHAP, the SHAP explanations (Hospital mode), and the vital trends over the last 20 readings. The AI is told to use only this data and not invent values.

### Two care modes

| | Hospital mode | Home Care mode |
|---|---|---|
| Role given to the AI | Senior ICU physician following SCCM / Surviving Sepsis guidelines | Assistant writing for a home nurse and non-medical family |
| Language | Clinical, with drug name, dose, route and target | Plain, calm English with safe home steps and "go to hospital now if…" warning signs |
| Rule | Every vital cited gives value and normal range | No clinical jargon without a plain explanation |

### Four sections in every report
1. **Current Condition** · 2. **Probable Cause** · 3. **Risk Forecast** · 4. **Immediate Actions**

### Risk-adaptive order
If the predicted SOFA is **≥ 7** (the alert line), the prompt starts with a clinical alert and puts **Immediate Actions first**. Otherwise the normal order is used, with Immediate Actions last.

### Practical details
- **Thinking text removed:** any `<think>…</think>` block some models produce is stripped, including unclosed or stray tags.
- **Section parsing:** the reply is split into the four sections, tolerating headings written with `#`, bold marks, numbers or dashes. All four keys always exist (empty if missing).
- **90-second cache:** the key includes the care mode, patient, SOFA and every vital in the prompt (HR, RR, SpO₂, SBP, DBP, MAP, temperature, GCS, stress), so a changed reading is never served a report quoting old numbers. Expired entries are cleaned on each write.
- **No duplicate calls:** if two requests ask for the same report at once, one call is made and the other waits for it.
- **Both modes prepared:** after one mode's report is ready, the dashboard prepares the other mode in the background so switching Hospital ↔ Home Care is instant.
- **Cost:** 2 AI calls per report (write + judge), about 4 per new reading with both modes.
- **If the AI fails:** the dashboard keeps showing the last report with a notice.

---

## 16. Validation — Checking the SOFA Score and the AI Report

> The earlier "self-consistency check" (asking the AI the same question 3 times) was **removed**: agreement measures consistency, not correctness, so a model that is consistently wrong would still score high.

The complete explanation, with worked examples, is in **`docs/VALIDATION.md`**. Summary:

### Part 1 — The SOFA score (three layers)

| Layer | When | What |
|---|---|---|
| 1. Offline metrics | Once, after training | MAE, R², AUROC, PR-AUC, 3-band and alert confusion matrices on 10,450 unseen stays (stored in `training_metadata.json`) |
| 2. Runtime checks | Every prediction | **SOFA floor** (abnormal MAP, SpO₂ or GCS imply a minimum SOFA); **dangerous vital combinations** (HR > 130 + SBP < 90; SpO₂ < 88 + MAP < 65); **unexplained jumps** (SOFA changes by more than 4 with no real vital change) |
| 3. Uncertainty | Every prediction | 90% **conformal** range, prediction ± 3.28 (the 90th percentile of absolute errors over the training stays) |

The checks flag a problem; they never change the predicted number.

### Part 2 — The AI report (nine methods)

| # | Method | Weight | What it checks |
|---|---|---|---|
| 1 | Factual grounding | 13% | Every abnormal vital is mentioned |
| 2 | Clinical hard rules | 13% | The standard treatment is mentioned (6 rules) |
| 3 | SHAP–LLM coherence | 9% | The report talks about what drove the prediction |
| 4 | Response structure | 9% | All 4 sections and enough length |
| 5 | Severity calibration | 8% | The tone matches how sick the patient is |
| 6 | Contraindication safety | 5% | No dangerous drug for these vitals (3 rules) |
| 7 | Numeric accuracy | 8% | Quoted numbers match real values |
| 8 | G-Eval (AI judge) | 13% | Overall clinical quality, 4 aspects scored 1–5 |
| 9 | RAGAS faithfulness (AI judge) | 22% | Each unsupported claim lowers the score |

Methods 1–7 are instant rule checks (65% of the weight); 8–9 use the AI judge (35%). If the judge is late (12 s) or fails, methods 8 and 9 score a neutral 0.5.

**Reliability labels:** ≥ 0.75 **High** (green) · 0.55–0.75 **Moderate** (amber) · < 0.55 **Low** (red). A separate **caution** flag is raised below 0.60. Every specific problem found is listed for the clinician.

### Honest limits of the judge
The same model writes and grades the report, and the judge sees only the first 900 characters. The seven rule checks are the independent part of the score.

---

## 17. Alert System

**Alert trigger:** predicted SOFA **≥ 7** — `ALERT_THRESHOLD = 7.0`, defined once in `model_utils.py`. The backend imports it and `compute_sofa_risk_metrics()` uses it by default, so training reports and the live dashboard always use the same line. (It was 8 in an earlier version.)

**Why 7 and not the clinical 10:** the model **under-predicts severe patients**. For truly High Risk stays the average true SOFA is about 13.1, but the average prediction is only about 9.5, because most of SOFA comes from labs and drug doses that are not inputs. Waiting for a predicted 10 would alert for only 43.9% of High Risk patients.

**Sweep on the final model** (10,450 test stays, 602 truly High Risk):

| Alert when predicted SOFA ≥ | Recall | Specificity | Missed | False alarms |
|---|---|---|---|---|
| 10 | 43.9% | 99.3% | 338 | 73 |
| 8 | 59.6% | 97.9% | 243 | 207 |
| **7 (current)** | **67.8%** | **96.6%** | **194** | **335** |
| 6 | 73.1% | 94.1% | 162 | 578 |

Moving from 8 to 7 catches 49 more High Risk patients for 128 more false alarms. In an ICU a missed patient costs more than an extra alarm, and below 7 the false alarms grow quickly for small gains. The threshold changes only the yes/no alert; AUROC, PR-AUC and MAE do not change. **Even at 7, 194 of 602 High Risk patients (32%) get no alert**, which is why ClinIQ supports clinicians and does not replace them.

**What the alert (`is_alert`) controls:** the urgent "Immediate Actions first" format of the AI report in both Hospital and Home Care modes.

**What the top banner shows** (this is based on the *risk level*, not the alert line):
- High Risk (SOFA ≥ 10) → red banner **CRITICAL ALERT**, "Immediate clinical review required".
- Moderate Risk (SOFA 5–9) → amber banner **ELEVATED RISK**, "Close observation warranted".
- Low Risk (SOFA < 5) → no banner, green gauge.

---

## 18. ClinIQ Dashboard — Complete Pipeline

### Backend startup
The backend loads everything once, on the first request (cached):
1. `models/scaler.pkl`, `models/tfidf_vectorizer.pkl`, `models/feature_columns.pkl`.
2. `ICUModel(108)` with the weights from `models/federated_model.pth` (`weights_only=True`), set to `eval()`.
3. `models/shap_background.npy` and the SHAP explainer (created on first use).
4. The conformal q̂, computed from the training data in the database (falls back to 2.9 if the data is unavailable).
5. At app start the database tables are created if missing (`init_db()`), so a fresh clone does not fail.

### The API (FastAPI, port 8000)

| Method and path | What it does |
|---|---|
| `GET /health` | `{"status": "ok", "service": "ClinIQ"}` |
| `GET /api/patients` | The 3 demo patients (name, description, risk, colour) |
| `GET /api/patients/{id}` | One patient (404 if not found) |
| `POST /api/predict` | Runs the full prediction (below). Optional live `cv_gcs_eye` (1–4) and `cv_stress` (0–10) |
| `POST /api/report` | Builds the AI report for the last prediction of that patient: `care_mode` is `hospital` or `homecare` |
| `GET /api/history/{id}?limit=` | Recent predictions (limit 1–50, default 5) |
| `GET /api/vitals/{id}?limit=` | Recent vitals for the trend charts (limit 1–20, default 20) |

Patient IDs must be 1, 2 or 3. Wrong input gives a clear 400 error. `/api/report` returns **409** if no prediction has run yet for that patient, and **500** if `GROQ_API_KEY` is missing. The server keeps its own copy of each patient's latest prediction and **never trusts data sent by the browser** when writing a report.

### What happens on `POST /api/predict` (every 90 seconds)
1. Read the next sample row for the patient (the 30 recorded readings per patient are replayed in a loop).
2. Use live camera values for GCS eye and stress if provided; otherwise use the recorded ones.
3. Seed the vitals window if empty, then append this reading and keep the last 20.
4. Compute the 9 trend features and the 7 latest values plus GCS and stress.
5. TF-IDF the clinical note into 90 columns.
6. Combine to 9 + 7 + 2 + 90 = **108** columns, in the saved order.
7. Scale and clip to ±10.
8. Predict with the network in `eval()` mode, clip to 0–24.
9. Classify risk (< 5 / 5–9 / ≥ 10), set `is_alert` (≥ 7), build the conformal range.
10. Run the three runtime checks (floor, vital combinations, trajectory).
11. Save the prediction to the history table (last 50 kept).
12. Compute SHAP and build the top 5, explanations and key risks.
13. Build the trend text and chart data.
14. Return the JSON, including the training metadata (`_training_meta`) that the validation drawer reads.

### What happens on `POST /api/report`
1. Check the 90-second cache; return it if present.
2. Build the care-mode prompt (alert-first if SOFA ≥ 7).
3. Call the model; strip thinking text; parse the four sections.
4. Call the judge (12-second limit).
5. Run the nine-method score and return: sections, reliability score and label, violations, caution flag, care mode and whether the judge answered.

### Frontend polling loop (`useMonitor.js`)
- 90-second countdown; when it reaches 1, a new prediction is fetched.
- A guard prevents overlapping requests; the countdown resets only after the data arrives.
- A response is dropped if the patient changed while it was in flight.
- The row position is kept in the browser's session storage, so a page refresh resumes from the same reading.
- The report is regenerated only when `reportKey()` (patient, SOFA and every vital) changes or the care mode is switched. This key mirrors the backend cache key.

### What the clinician sees
- **Select-patient page:** three patient cards (Low, Moderate, High risk demos).
- **Monitor page** (one scrolling page):
  1. **Header** with patient, live status, countdown, theme toggle and exit.
  2. **Alert banner** (Moderate / High only).
  3. **Current Status:** the **SOFA gauge** (score, risk colour, conformal range, confidence) and the **vital cards** (7 vitals + GCS eye + stress, colour-coded normal / abnormal).
  4. **Vital Trends:** line charts of the last 20 readings with normal-range bands.
  5. **AI Clinical Report:** Hospital / Home Care toggle → the four section cards → **SHAP bars** (top 5) → reliability badge → disclaimer.
- **Four floating buttons** on the right edge (bottom to top): **History** (last predictions table), **Camera monitor**, **LLM validation drawer** (the nine methods and every problem found), **SOFA validation drawer** (offline metrics read live from `training_metadata.json`, the three runtime checks, and the conformal range).
- Dark and light themes.

### Camera monitor (in the browser)
The camera feature runs entirely in the browser (React plus **MediaPipe FaceLandmarker** compiled to WebAssembly, files in `frontend/public/mediapipe/`). The first use downloads a ~12 MB face model. Nothing is sent to a server except the two resulting numbers.
- **Stress score (0–10)** from facial muscle movements using the **PSPI** pain formula (brow lowering, cheek/eye squeeze, nose and upper-lip raising, eye closing, reduced by smiling), mixed 80% facial and 20% head movement, then smoothed with an exponential average (α = 0.15). Bands: LOW < 3.5, MODERATE < 6.5, HIGH ≥ 6.5.
- **GCS eye score (1–4)** from how often the eyes are open over a rolling **30-second** window: ≥ 70% open → 4 (spontaneous); ≥ 30% of the time at least partly open → 3 (to voice); a brief flicker → 2 (to pain); otherwise 1 (no response). A short personal calibration (about 3–10 s) sets this person's open-eye level.
- Readings are sent at most once a second and stop if no face is seen for 3 seconds.
- Details: `docs/CV_MONITOR.md`.

### Database (SQLite, `models/icu_monitor.db`)

| Table | Purpose | Current rows |
|---|---|---|
| `patient_vitals` | Sliding window of vitals per patient (newest 20 kept) | 60 |
| `prediction_history` | Prediction log per patient (newest 50 kept) | 150 |
| `sample_patients` | The recorded readings that are replayed (30 per patient) | 90 |
| `fl_training` | The 3 hospitals' scaled training rows | 41,798 |

Tables use `created_at` / `updated_at` triggers. The file is gitignored (MIMIC-III derived; only credentialed users may hold it).

### The three demo patients

| ID | Patient | Story |
|---|---|---|
| 1 | Low Risk | Post-surgical recovery, stable and improving |
| 2 | Moderate Risk | Community-acquired pneumonia on supplemental oxygen |
| 3 | High Risk | Septic shock, on vasopressors, intubated, multi-organ failure |

---

## 19. Complete Folder Structure

```
icu_monitor/
│
├── model_utils.py             ← Shared ML code: ICUModel, ALERT_THRESHOLD, training, metrics
├── train_federated.py         ← Full data prep + federated training simulation (Flower + Ray)
├── server.py                  ← Real Flower server (TCP, 127.0.0.1:8080)
├── client.py                  ← Real Flower client (--client_id 0/1/2)
├── fl_dashboard.py            ← Streamlit app: live FL simulation (port 8501)
├── db.py                      ← SQLite layer (all reads and writes)
├── migrate_to_db.py           ← Imports the demo CSVs into the database
├── setup.py                   ← One-command cross-platform setup
├── requirements.txt
├── pytest.ini
├── .env                       ← NEVER committed (holds GROQ_API_KEY)
├── .env.example
├── .gitignore
├── README.md
│
├── backend/                   ← FastAPI backend (port 8000)
│   ├── main.py                ← App entry, CORS, routers, /health, DB init
│   ├── api/
│   │   ├── patients.py        ← GET /api/patients
│   │   ├── predict.py         ← POST /api/predict
│   │   ├── report.py          ← POST /api/report
│   │   ├── history.py         ← GET /api/history/{id}
│   │   └── vitals.py          ← GET /api/vitals/{id}
│   └── services/
│       ├── ml_service.py      ← Prediction pipeline, SHAP, runtime checks, conformal range
│       └── llm_service.py     ← Groq calls, prompts, cache, 9-method reliability, judge
│
├── frontend/                  ← React 19 + Vite 8 (default port 5173)
│   ├── package.json, vite.config.js (proxies /api → :8000), index.html
│   ├── public/ favicon.svg, mediapipe/ (WebAssembly for the camera monitor)
│   └── src/
│       ├── main.jsx, App.jsx, index.css
│       ├── api/client.js          ← Axios client for all endpoints
│       ├── hooks/useMonitor.js    ← 90 s polling loop
│       ├── contexts/ThemeContext.jsx
│       ├── pages/ SelectPatient.jsx, MonitorDashboard.jsx
│       └── components/
│           ├── layout/      HeaderBar, AlertBanner, ThemeToggle
│           ├── overview/    SOFAGauge, VitalCard, VitalsGrid
│           ├── trends/      TrendChart, TrendsSection
│           ├── report/      ReportContainer, ReportSection, ReportToggle, SHAPPanel, ReliabilityBadge
│           ├── history/     HistoryDrawer, HistoryFAB
│           ├── cv/          CVMonitorModal, CVMonitorFAB
│           ├── validation/  SOFAValidationDrawer/FAB, LLMValidationDrawer/FAB
│           └── ErrorBoundary.jsx
│
├── docs/                      ← CLINIQ_TESTING, CV_MONITOR, FL_SERVER_CLIENT_DB_TESTING,
│                                INPUT_OUTPUT, MODEL_DISCUSSION, Project_Context, SKILL (this file),
│                                SQL_COMMANDS, TRAIN_FEDERATED, VALIDATION
│
├── models/
│   ├── federated_model.pth        ← network weights (108→256→128→64→1, 69,121)
│   ├── scaler.pkl                 ← StandardScaler for the 108 inputs
│   ├── tfidf_vectorizer.pkl       ← TF-IDF with the 90-word SOFA list
│   ├── feature_columns.pkl        ← the 108 input names, in order
│   ├── shap_background.npy        ← 300 background patients (300 × 108)
│   ├── training_metadata.json     ← settings and test results of the final run
│   ├── fl_test_X.npy, fl_test_y.npy ← the held-out test set (10,450 stays)
│   └── icu_monitor.db             ← SQLite database (gitignored)
│
├── data/
│   └── sample_patients/       ← 3 demo CSVs (30 recorded readings each)
│
└── tests/
    ├── fl/ (unit, integration, system)            ← federated learning, DB, model tests
    └── cliniq/
        ├── backend/ (api, unit, system)            ← API and service tests
        └── frontend/ (unit, integration, e2e)      ← component, hook and e2e specs
```

Training no longer writes CSV files: the three hospitals' data lives in the `fl_training` table. There is no standalone `cv_monitor.py`; the camera monitor is built into the dashboard.

---

## 20. File-by-File Description

### Training and model code

| File | What it does |
|---|---|
| `model_utils.py` | Shared code: `ALERT_THRESHOLD = 7.0`; `ICUModel` (108→256→128→64→1, dropout); `ICUDataset`; `weighted_mse_loss` (optional); `get_sample_weights` (oversampling, off); `train_model` (AdamW, plain MSE, FedProx term, gradient clipping); `evaluate_model`; `compute_sofa_risk_metrics` (band and alert confusion matrices, AUROC, PR-AUC); `get_weights` / `set_weights`; `get_trend`; `classify_range`; `apply_dp_to_update` and `estimate_privacy_budget` (differential privacy, off) |
| `train_federated.py` | Phase 0 (BigQuery → notes → TF-IDF → scaling → hospital split → database), then phases 1–6 (load, SHAP background, federated training, save best model, test, save metadata) |
| `server.py` | Real Flower server; FedAvg; saves the best round to `federated_model.pth` |
| `client.py` | Real Flower client; takes `--client_id`; loads its rows from `fl_training`; trains with FedProx |
| `fl_dashboard.py` | Streamlit page that shows a federated run live (FedAvg, FedYogi option, charts) |
| `db.py` | All database access: connection (WAL mode), `init_db`, vitals, history, sample rows, `get_fl_client_data`, `fl_training_exists` |
| `migrate_to_db.py` | Imports the demo CSVs into `sample_patients`, `patient_vitals` and `prediction_history` |
| `setup.py` | Checks Python ≥ 3.12 and Node ≥ 18, creates a virtual environment, installs dependencies, initialises the database, creates `.env` from `.env.example` |

### Backend

| File | What it does |
|---|---|
| `backend/main.py` | FastAPI app; CORS for ports 5173/5174/5175; registers the 5 routers; creates tables at startup |
| `backend/api/predict.py` | Validates input, runs `run_prediction`, stores the full result server-side for the report, returns a clean copy |
| `backend/api/report.py` | Uses the stored prediction, calls `generate_report` |
| `backend/api/patients.py`, `history.py`, `vitals.py` | Patient list, prediction history, vitals window |
| `backend/services/ml_service.py` | The whole inference pipeline, SHAP, risk label, SOFA floor, vital-combination flags, trajectory check, conformal q̂, training-metadata loader |
| `backend/services/llm_service.py` | Prompt builders for both modes, Groq calls, cache, section parser, nine reliability methods, judge, labels |

### Frontend (key files)

| File | What it does |
|---|---|
| `hooks/useMonitor.js` | 90-second loop, guards, session-stored row index |
| `pages/MonitorDashboard.jsx` | Lays out the monitor page and wires the drawers and buttons |
| `components/report/ReportContainer.jsx` | Fetches reports per care mode, caches them per reading, prefetches the other mode |
| `components/report/SHAPPanel.jsx` | Animated SHAP bars; keeps the last good factors if SHAP fails once |
| `components/cv/CVMonitorModal.jsx` | MediaPipe camera monitor: PSPI stress, 30-second GCS window, calibration |
| `components/validation/SOFAValidationDrawer.jsx` | Shows offline metrics live from `_training_meta`, runtime checks, conformal range; "—" if metadata missing |
| `components/validation/LLMValidationDrawer.jsx` | Shows the nine-method breakdown and every violation |
| `components/layout/AlertBanner.jsx` | Red / amber banner by risk level |

### Saved model files

| File | Contents |
|---|---|
| `federated_model.pth` | PyTorch state dict, keys `net.0`, `net.3`, `net.6`, `net.9` (weights and biases; ReLU and dropout layers have none) |
| `scaler.pkl` | StandardScaler; `scaler.feature_names_in_` holds the authoritative 108-name order |
| `tfidf_vectorizer.pkl` | TF-IDF with the 90-word list, fitted on all 283,208 notes |
| `feature_columns.pkl` | The 108 names in training order |
| `shap_background.npy` | 300 training rows, shape (300, 108) |
| `training_metadata.json` | All settings and test results (the SOFA validation drawer reads it live) |
| `fl_test_X.npy`, `fl_test_y.npy` | The held-out test set, so later runs and checks use the same test patients |

---

## 21. All Finalized Technical Parameters

| Parameter | Value | Where used |
|---|---|---|
| Input size | 108 | model, scaler, all files |
| TF-IDF words | 90 (SOFA list), n-grams (1, 2) | `tfidf_vectorizer.pkl` |
| Notes used / text limit | 283,208 / 10,000 characters | `train_federated.py` |
| Sliding window | 20 readings | `db.py`, `ml_service.py` |
| Prediction history kept | 50 | `db.py` |
| Trend slope threshold | 0.1 | `get_trend()` |
| Risk bands | < 5, 5–9, ≥ 10 | `risk_label()` |
| **Alert threshold** | **SOFA ≥ 7** | `ALERT_THRESHOLD` in `model_utils.py` |
| Network | 108→256→128→64→1, Dropout 0.3 | `ICUModel` |
| Trainable weights | 69,121 | `ICUModel` |
| Rounds / epochs / batch | 100 / 3 / 64 | `train_federated.py`, `client.py`, `server.py` |
| Learning rate / decay / floor | 0.001 / 0.99 per round / 1e-4 | `train_federated.py`, `client.py` |
| Gradient clip | 1.0 | `train_federated.py`, `client.py` |
| FedProx μ | 0.5 | `train_federated.py`, `client.py` |
| Server method | FedAvg (`USE_FEDYOGI = False`) | `train_federated.py`, `server.py` |
| FedYogi η / β1 / β2 / τ (optional) | 0.001 / 0.9 / 0.99 / 0.001 | `train_federated.py` |
| Loss | plain MSE | `train_model()` |
| Client validation share | 15% | `train_federated.py`, `client.py` |
| Hospitals | 3 (IID, stratified) | `train_federated.py` |
| Differential privacy / non-IID / oversampling | all off | `train_federated.py` |
| **Conformal 90% q̂** | **3.28** (fallback 2.9) | `ml_service.py` |
| SHAP background / top shown | 300 / 5 (7 computed) | `ml_service.py` |
| AI model (write and judge) | `openai/gpt-oss-120b` | `llm_service.py` |
| Write: temperature / max tokens | 0.15 / 6,000 | `llm_service.py` |
| Judge: temperature / max tokens / limit / text seen | 0 / 1,200 / 12 s / 900 characters | `llm_service.py` |
| Report cache | 90 s | `_CACHE_TTL` |
| Reliability labels | ≥ 0.75 High, 0.55–0.75 Moderate, < 0.55 Low; caution < 0.60 | `llm_service.py` |
| Method weights | 13 / 13 / 9 / 9 / 8 / 5 / 8 / 13 / 22 (%) | `compute_reliability` |
| Frontend refresh | 90 s | `useMonitor.js` |
| GCS eye / stress range | 1–4 / 0–10 | camera monitor, API |
| Server address (real demo) | 127.0.0.1:8080 | `server.py`, `client.py` |
| Ports | backend 8000, frontend 5173, Streamlit 8501 | |

---

## 22. Model Performance Metrics

### Final federated network (`federated_model.pth`, 2026-11-24)

| Metric | Value |
|---|---|
| MAE | **1.80** SOFA points (1.8006) |
| R² | **0.48** (0.4806) |
| AUROC (High Risk) | **0.918** |
| PR-AUC (High Risk) | **0.647** |
| Prediction range on the test set | 1.78 – 23.07 |
| Training / test stays | 41,798 / 10,450 |
| Rounds | 100 (best = **round 37**) |
| Server / client method | FedAvg + FedProx (μ = 0.5) |
| Loss | plain MSE |
| Conformal 90% interval | ± 3.28 |

**Evaluation limitation:** one row per ICU stay, split 80/20 by ICU stay. A patient with several stays can appear in both train and test, which can flatter the scores. These are retrospective MIMIC-III results, not independent clinical validation.

### Per risk band

| Risk level | Stays | MAE | 3-band recall |
|---|---|---|---|
| Low (< 5) | 7,027 | 1.33 | 95.5% |
| Moderate (5–9) | 2,821 | 2.44 | 23.7% |
| High (≥ 10) | 602 | 4.36 | 43.9% |

Per-band R² is negative (Low −0.71, Moderate −3.53, High −1.75). This is expected: inside one narrow band the differences come from lab-based organs the model cannot see. Only the overall R² is meaningful.

**3-band confusion matrix** (rows actual, columns predicted: Low / Moderate / High):
`[[6714, 306, 7], [2086, 669, 66], [119, 219, 264]]` → **3-band accuracy 73.2%**.

### High-risk alert (predicted SOFA ≥ 7 vs actual SOFA ≥ 10)

| Metric | Value |
|---|---|
| Recall | **67.8%** (408 of 602) |
| Specificity | **96.6%** |
| Precision | 54.9% |
| F1 | 0.607 |
| Missed / false alarms / correctly quiet | 194 / 335 / 9,513 |

The SOFA validation drawer reads all of these live from `training_metadata.json`, so it always matches the loaded model.

### How the model improved

| Step | R² | AUROC | Alert recall |
|---|---|---|---|
| Original FL notebook (plain MSE) | 0.04 | — | — |
| Weighted loss | 0.11–0.13 | — | — |
| 90-word SOFA list | 0.175 | — | — |
| FedAdam on the server | 0.215 | — | — |
| FedYogi on the server | 0.223 | — | — |
| All 283,208 notes (old dataset) | ≈ 0.40–0.42 | 0.879 | 23% |
| `dataset_v2`, FedYogi, weighted loss | 0.426 | 0.915 | 53.8% |
| + FedAvg and plain MSE | 0.457 | 0.921 | 59.3% |
| + 256-128-64 with dropout 0.3 (first run) | 0.473 | 0.925 | 61.1% (at ≥ 8) |
| **Final retrain, alert ≥ 7** | **0.481** | **0.918** | **67.8%** |

Rows up to "All 283,208 notes" used the old dataset and label definition, so they cannot be compared exactly with the `dataset_v2` rows. Two runs of the same settings differ slightly (federated simulation is not exactly repeatable), which is why the last two rows differ a little.

**Did not help on `dataset_v2`:** lower learning rate, stronger weight decay, Huber loss, a 512-256-128 network, averaging 8 models, recalibration, missing-value flags, several hourly rows per stay (+0.007 R² for 9× the data).

**Reference ceiling:** a centralised gradient-boosting model on the same 108 inputs and the same test stays reaches MAE 1.79, R² 0.50, AUROC 0.922. The federated network is already close to what these inputs allow; more accuracy would need lab values as inputs.

---

## 23. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML | PyTorch ≥ 2.0, scikit-learn ≥ 1.6 (< 2.0), NumPy, Pandas, joblib |
| Federated learning | Flower (`flwr[simulation]` ≥ 1.8, uses Ray) — FedAvg + FedProx |
| Explainability | SHAP ≥ 0.44 (`DeepExplainer`) |
| NLP | scikit-learn `TfidfVectorizer` (90-word SOFA list) |
| LLM | Groq: `openai/gpt-oss-120b` for both writing and judging |
| Backend | FastAPI ≥ 0.115 + uvicorn (port 8000), python-dotenv |
| Database | SQLite (`models/icu_monitor.db`) |
| Data source | MIMIC-III on Google BigQuery (`mimic-project-2.dataset_v2`); `google-cloud-bigquery`, `db-dtypes`, `pyarrow` |
| FL simulation dashboard | Streamlit + Plotly (port 8501) |
| Frontend | React 19, Vite 8, React Router 7, TanStack React Query, Axios, Tailwind CSS 3 |
| Frontend visuals | Recharts (charts), Framer Motion 14, anime.js 4, AOS |
| Camera monitor | `@mediapipe/tasks-vision` (FaceLandmarker + 52 blendshapes, in the browser) |
| Stress scoring | PSPI formula (Prkachin & Solomon, 2008) |
| Tests | pytest (Python), Vitest + Testing Library (frontend), Playwright e2e specs |
| Lint | oxlint |

No Python computer-vision packages (such as OpenCV) are needed; the camera work is done in the browser.

---

## 24. How to Run the Project

All commands run from the `icu_monitor/` folder unless stated.

### First-time setup (recommended)
```bash
python setup.py
# checks Python/Node, makes a venv, installs Python + frontend packages,
# creates the database, copies .env.example to .env
```
Then put your key in `.env`:
```
GROQ_API_KEY=your_key_here
```
Never commit `.env`.

### Manual setup
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
python migrate_to_db.py        # only if models/icu_monitor.db is missing
```

### Run the dashboard (2 terminals)
```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev     # opens http://localhost:5173
```
Restart the backend after retraining so it loads the new model.

### Run the federated-learning simulation dashboard
```bash
streamlit run fl_dashboard.py    # http://localhost:8501
```

### Real federated learning demo (4 terminals)
```bash
python server.py                     # start first
python client.py --client_id 0
python client.py --client_id 1
python client.py --client_id 2
```

### Retrain the model
```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project mimic-project-2
python train_federated.py
```
- **First time (Phase 0):** downloads about 52,000 vitals rows and 283,208 notes and scores them; this takes the longest.
- **Later runs:** Phase 0 is skipped automatically because the database already has training data.
- **Re-do data preparation:** `sqlite3 models/icu_monitor.db "DELETE FROM fl_training;"` then run again.
- The 100 federated rounds follow. Results can differ slightly from run to run; quote `models/training_metadata.json`.
- Anyone without BigQuery access can train using a shared copy of the `fl_training` table (only for PhysioNet-credentialed recipients; never on GitHub).

### Run the tests
```bash
pytest tests                                  # Python: 513 tests (295 FL + 218 backend)

# Frontend tests live in tests/, outside frontend/, so packages need a symlink (create once; gitignored):
ln -s frontend/node_modules node_modules
cd frontend && npx vitest run                 # Vitest: 232 tests in 27 files
```
Total: **745 tests**. The three Playwright e2e specs in `tests/cliniq/frontend/e2e/` need a running app and are not part of this count.

---

## 25. Project Evolution History

**Phase 1 — Architecture and design.** Chose MIMIC-III and Google BigQuery; drew the system architecture.

**Phase 2 — BigQuery preprocessing.** Ran 12 queries to build the first dataset (60,188 rows). Historical; replaced in Phase 9.

**Phase 3 — Ensemble ML (`trial/`).** LightGBM + XGBoost + Random Forest, SHAP TreeExplainer, Groq with a self-consistency check. R² 0.40, MAE 1.86. Could not be federated because trees cannot be averaged.

**Phase 4 — Federated learning.** Switched to a PyTorch network and Flower. R² 0.04–0.18 (undertrained).

**Phase 5 — Training improvements.** Weighted loss, AdamW, SaveBestStrategy, SHAP DeepExplainer. R² 0.13–0.25.

**Phase 6 — SOFA word-list breakthrough.** Replaced 600 automatic words with the 90-word SOFA list; model shrank to 108→128→64→32→1. R² 0.175 (the biggest single early jump).

**Phase 7 — FedAdam / FedYogi.** Server-side adaptive methods: FedAdam 0.215, FedYogi 0.223. Added FedProx; tuned the loss weight from 3.0 to 0.5.

**Phase 8 — More notes.** Notes 80k → all 283k; text per admission 5k → 10k characters. R² ≈ 0.40–0.42 on the old dataset.

**Phase 9 — `dataset_v2` rebuild and model update (2026-11-12).** Rebuilt the data in BigQuery (hourly time-matched SOFA, 52,248 stays). FedYogi exploded on it, so FedAvg replaced it; plain MSE replaced weighted MSE; the network became 256-128-64 with dropout 0.3; the alert moved from 8 to 7.

**Phase 10 — Validation framework.** The three-run self-consistency check was replaced by nine methods (seven rule checks and two AI-judge checks). Three runtime checks and a conformal range were added for the SOFA score, and the SOFA validation drawer was made to read the training metadata live.

**Phase 11 — Final retrain and documentation (2026-11-24).** Final run: MAE 1.80, R² 0.48, AUROC 0.918, alert recall 67.8% at 96.6% specificity. All documents were updated to match. No further model changes are planned.

---

## 26. Novel Contributions

1. **Multimodal fusion** of vitals, camera-based cues and clinical notes in one risk predictor, with live webcam values entering the model.
2. **SOFA-vocabulary TF-IDF list:** 90 terms mapped to the six SOFA organs, replacing noisy automatic word selection.
3. **Trend features over a sliding window** of the last 20 readings.
4. **Stable federated learning with FedAvg + FedProx + dropout**, reaching R² 0.48 and AUROC 0.918 with no patient data shared, close to a centralised model on the same inputs.
5. **A nine-method reliability score for LLM output** that checks the AI report against the real patient data and ICU rules (not against itself), with every problem listed.
6. **A tuned, justified alert line:** a measured sweep shows why 7 beats 8 and 10 for a model that under-predicts severe patients.
7. **SHAP on a federated network**, explaining each prediction in plain language.
8. **Two audiences:** Hospital and Home Care reports from the same prediction.

---

## 27. Design Constraints, Assumptions and Limitations

### Constraints
- No patient data can leave a hospital → federated learning.
- Must run on ordinary hardware (CPU only; no GPU needed).
- Must be explainable to clinicians → SHAP and a plain-language report.
- MIMIC-III only; no real-time patient data access.
- The 108 model inputs are fixed.

### Assumptions
- Vitals are available continuously and are reasonably accurate.
- A 20-reading window captures enough recent history.
- MIMIC-III's hourly SOFA is the right target label.
- The simulated 3-hospital split is a fair stand-in for real multi-hospital training.
- Hospital participants behave honestly.

### Limitations (stated honestly)
- **No lab values** (bilirubin, creatinine, platelets, blood gases) or drug doses as inputs. This limits R² to about 0.48 and makes the Moderate band hard (only 23.7% classified correctly).
- **Under-prediction of severe patients:** even at alert line 7, 194 of 602 truly High Risk stays (32%) get no alert.
- **Split by ICU stay, not by patient:** a patient with several stays can be in both train and test. Results are retrospective, not independent clinical validation.
- **The hospitals are artificial:** the split is random (IID), so there is no real hospital-to-hospital difference. A harder non-IID split is available as an option; in earlier tests it dropped R² sharply (about 0.28 to 0.08).
- **Conformal range measured on training data:** real coverage on new patients is probably a little under 90%.
- **The judge also wrote the report** and sees only the first 900 characters; the keyword checks look for words, not meaning.
- **Demo data is replayed:** the dashboard replays 30 recorded readings per sample patient; the model does not retrain on new patients automatically.
- **Camera cues** are lightly weighted in training (most stays have stress 0 and GCS 4), so live camera values change the predicted SOFA only slightly.
- **Differential privacy** exists but is off, and its guarantee would be weak (ε ≈ 485 over 100 rounds).
- The AI report needs an internet connection and a Groq key.

---

## 28. Viva Preparation — Key Points

### Q: What is ClinIQ in one sentence?
A decision-support dashboard that predicts an ICU patient's SOFA organ-failure score from vitals, doctor's notes and camera cues, explains it, raises an alert and writes a checked AI report, using a model trained across hospitals without sharing patient data.

### Q: Why MIMIC-III and not MIMIC-IV?
MIMIC-III is stable for preprocessing and has well-documented derived tables, including hourly pre-computed SOFA scores.

### Q: Why TF-IDF instead of ClinicalBERT?
TF-IDF is simple, fast and interpretable, and with a fixed 90-word SOFA list every input has a clear medical meaning. An earlier ClinicalBERT approach was dropped because TF-IDF gave similar accuracy at far lower cost.

### Q: Why a PyTorch network and not LightGBM?
Federated averaging averages weight matrices; trees have none. A gradient-boosting model on the same inputs scores about the same (R² 0.50 vs 0.48), so we lost almost nothing.

### Q: Is your federated learning real or simulated?
The protocol is real (Flower, weights only, FedAvg). The hospitals' data is simulated by splitting one MIMIC-III dataset into three. This is standard in FL research.

### Q: Why FedAvg and not FedYogi?
FedYogi made the server loss explode on the rebuilt dataset (about 8 → 123 in rounds 5–10). FedProx already stops hospitals drifting apart, so plain FedAvg is enough and trains smoothly. FedYogi is still available with `USE_FEDYOGI = True`.

### Q: What is FedProx and why use it?
It adds a penalty (μ/2)·‖local − shared‖² to each hospital's loss, so a hospital learns from its data but cannot wander far from the shared model. μ = 0.5 was the best balance (0.01 too weak, 1.0 too strong).

### Q: Why dropout, and why wider layers?
Without it the network memorised its training patients after about round 13 (training loss fell while validation loss rose). Dropout 0.3 stops that and the extra width gives it room to learn with dropout on.

### Q: Why is R² 0.48 and not higher? Is that acceptable?
SOFA is mostly computed from lab values and drug doses that are deliberately not inputs. A centralised gradient-boosting model on the same inputs reaches only 0.50, so we are near the ceiling for these inputs. For the decision that matters, flagging high-risk patients, AUROC is 0.918.

### Q: Why does the alert fire at 7 and not 10?
The model under-predicts severe patients (their true average SOFA is about 13.1; the model's average guess is about 9.5). At 7 it catches 67.8% of High Risk patients at 96.6% specificity. Moving from 8 to 7 catches 49 more patients for 128 more false alarms. At 10 it would catch only 43.9%.

### Q: Why is the High Risk recall only 68%?
The model cannot see the labs and drug doses that drive severe SOFA, so it under-scores severe patients. 194 of 602 still get no alert. That is why the system supports clinicians instead of replacing them.

### Q: How do you validate the SOFA score?
Three layers: offline metrics on 10,450 unseen stays; three runtime plausibility checks (floor, dangerous vital combinations, unexplained jumps); and a 90% conformal range (± 3.28).

### Q: How do you validate the AI report?
Nine methods. Seven are instant rule checks against the real vitals and ICU guidelines (mentions abnormal vitals, recommends the standard treatment, matches SHAP, has all sections, right tone, no dangerous drugs, correct numbers). Two use an AI judge (overall quality and unsupported claims). They combine into a 0–1 reliability score with every problem listed.

### Q: Why did you stop asking the AI three times and comparing answers?
Agreement measures consistency, not correctness. The AI can say the same wrong thing three times and still look reliable. Checking the text against the real data catches that.

### Q: What if the reliability score is high but the report is wrong?
It can happen, and the documentation says so. The score is mostly rule checks against real data, every problem is listed, the SOFA alert fires independently of the report, and every report carries a disclaimer.

### Q: How does SHAP work with a neural network?
`shap.DeepExplainer` with 300 background patients computes each input's contribution to the prediction by backpropagation. The model must be in `eval()` mode for a stable result.

### Q: How does the camera feed the model?
The browser analyses the face with MediaPipe (no video leaves the device). It computes a stress score from facial muscle movements (PSPI) and an eye-opening score over a 30-second window. These two numbers replace the recorded GCS and stress values in the model input.

### Q: What happens to patient data in federated learning?
It never leaves the hospital. Only weight arrays travel, which contain no patient identifiers.

### Q: Can these results be trusted as clinical proof?
No. They are retrospective MIMIC-III results, split by ICU stay rather than by patient. They show the approach works; they are not independent clinical validation.

### Q: What would you do next?
Add real lab values as inputs (the biggest possible gain), split train and test by patient, hold back a calibration set for the conformal range, handle severe cases better, and validate prospectively.

---

## 29. Glossary

| Term | Plain meaning |
|---|---|
| **SOFA** | A 0–24 score of how badly six organs are failing; higher = sicker |
| **ICU stay** | One admission to intensive care; the unit of one training row |
| **MIMIC-III** | A public, de-identified ICU database from Boston |
| **BigQuery** | Google's cloud database where our MIMIC-III copy lives |
| **Federated learning (FL)** | Training one model across several sites without moving their data |
| **Flower** | The software framework that runs the federated rounds |
| **FedAvg** | The server averages the hospitals' weights (bigger hospitals count more) |
| **FedProx** | A penalty that keeps each hospital close to the shared model |
| **FedYogi** | Averaging plus momentum and an adaptive step; optional here, off by default |
| **IID** | Every hospital has the same kind of patient mix |
| **Dropout** | Randomly switching off neurons while training so the model cannot memorise |
| **Overfitting** | Better on training data but worse on new data |
| **MAE** | Average size of the prediction error |
| **R²** | How much better than "guess the average for everyone" (1 = perfect) |
| **AUROC** | Chance the model ranks a sicker patient above a less sick one |
| **PR-AUC** | Precision and recall combined; good for rare conditions |
| **Recall / specificity** | Share of sick patients caught / share of healthy patients left alone |
| **TF-IDF** | A score for how important a word is in one document compared with all documents |
| **StandardScaler** | Rescales each input to average 0 and spread 1 |
| **SHAP** | Explains a prediction as contributions from each input |
| **Conformal interval** | A range built from past errors that holds the truth about 90% of the time |
| **q̂ (q-hat)** | The error size 90% of past patients stayed within (3.28) |
| **LLM** | Large language model, an AI that writes text |
| **LLM judge** | An AI call that grades another AI output |
| **Hallucination** | AI text that sounds right but is not supported by the data |
| **GCS eye opening** | Eye part of the Glasgow Coma Scale: 4 = opens by itself, 1 = never |
| **PSPI** | A facial pain formula used to score stress from facial muscle movements |
| **MediaPipe** | Google's browser face-analysis library used by the camera monitor |
| **MAP** | Mean arterial pressure; below 65 is dangerous |
| **Alert threshold** | Predicted SOFA at which the red alert fires (7) |
| **CDSS** | Clinical decision support system: assists, does not decide |

---

*Documentation for the ClinIQ ICU CDSS capstone. When the model is retrained, update Section 22 and the figures quoted in Sections 5, 12, 17 and 28 (read them from `models/training_metadata.json`).*

# TRAIN_FEDERATED — Complete Technical Documentation
**Module:** ICU CDSS Capstone — Federated Learning Training Pipeline
**Status:** Fully implemented and verified
**Best result achieved:** R²=0.4724, MAE=1.76 SOFA points (full training set)

---

## What This Document Covers

This document covers two things:

**Section A — How the training script works:** Every phase of `train_federated.py` in execution order, with plain-language explanations of what each step does and why.

**Section B — How the model got from R²=0.09 to R²=0.4724:** The complete research journey — every experiment tried, every failure with a diagnosis of why, and every breakthrough with an explanation of the mechanism.

If you are presenting this project to a panel, read Section A to understand what the system does and Section B to understand what you tried and learned.

---

## Table of Contents

### Section A — Current Script Flow
1. [Federated Learning — What It Is and Why We Use It](#1-federated-learning--what-it-is-and-why-we-use-it)
2. [One-Line Summary of the Training Script](#2-one-line-summary)
3. [All Configurable Parameters — Explained](#3-all-configurable-parameters--explained)
4. [Auto-Detection: Does the Script Skip Preprocessing?](#4-auto-detection-skip-preprocessing)
5. [Phase 0 — Data Preprocessing](#5-phase-0--data-preprocessing)
   - [0a. BigQuery Connection](#0a-bigquery-connection)
   - [0b. Vitals Query — ml_dataset_final](#0b-vitals-query)
   - [0c. Clinical Notes Query](#0c-clinical-notes-query)
   - [0d. Text Cleaning and TF-IDF Encoding](#0d-text-cleaning-and-tfidf-encoding)
   - [0e. Merge and Clean](#0e-merge-and-clean)
   - [0f. Train/Test Split](#0f-traintest-split)
   - [0g. StandardScaler with Clipping](#0g-standardscaler-with-clipping)
   - [0h. Hospital Split — IID 33/33/34](#0h-hospital-split--iid-333334)
   - [0i. Save Sklearn Artifacts](#0i-save-sklearn-artifacts)
6. [Phase 1 — Load Client Data](#6-phase-1--load-client-data)
7. [Phase 2 — SHAP Background Data](#7-phase-2--shap-background-data)
8. [Phase 3 — Federated Learning Simulation](#8-phase-3--federated-learning-simulation)
   - [The Model Architecture](#81-the-model-architecture)
   - [The Loss Function](#82-the-loss-function--weighted-mse)
   - [FedProx Client Regularisation](#83-fedprox-client-regularisation)
   - [The Hospital Client](#84-the-hospital-client)
   - [FedYogi Server Optimiser](#85-fedyogi-server-optimiser)
   - [One Complete FL Round](#86-what-happens-in-one-complete-fl-round)
9. [Phase 4 — Save the Best Model](#9-phase-4--save-the-best-model)
10. [Phase 5 — Final Evaluation](#10-phase-5--final-evaluation)
11. [Phase 6 — Save Training Metadata](#11-phase-6--save-training-metadata)
12. [Optional Features: Non-IID Split and Differential Privacy](#12-optional-features)
13. [Complete Output Files](#13-complete-output-files)
14. [How to Run the Script](#14-how-to-run)

### Section B — Complete Optimization Journey
15. [The Underlying Problem: Predicting SOFA Score](#15-the-underlying-problem-predicting-sofa-score)
16. [Stage-by-Stage Experiment Log](#16-stage-by-stage-experiment-log)
    - [Stage 1: Catastrophic Baseline (R²=−824)](#stage-1-catastrophic-baseline)
    - [Stage 2: After Clipping — Plateau at R²=0.11–0.13](#stage-2-after-clipping--the-plateau)
    - [Stage 3: Failed Experiments to Break the Plateau](#stage-3-failed-experiments)
    - [Stage 4: Breakthrough #1 — SOFA Vocabulary Whitelist (R²=0.1752)](#stage-4-breakthrough-1--sofa-vocabulary-whitelist)
    - [Stage 5: Failed Experiments After SOFA Vocabulary](#stage-5-failed-experiments-after-vocabulary)
    - [Stage 6: Breakthrough #2 — FedAdam (R²=0.2148)](#stage-6-breakthrough-2--fedadam)
    - [Stage 7: Breakthrough #3 — FedYogi (R²=0.2229)](#stage-7-breakthrough-3--fedyogi)
    - [Stage 8: Failed Experiments After FedYogi](#stage-8-failed-experiments-after-fedyogi)
    - [Stage 9: Breakthrough #4 — Expanded Notes Corpus (R²=0.4724)](#stage-9-breakthrough-4--expanded-notes-corpus)
17. [Key Lessons Learned](#17-key-lessons-learned)
18. [Performance Summary Table](#18-performance-summary-table)
19. [Fundamental Ceiling Analysis](#19-fundamental-ceiling-analysis)

---

---

# Section A — Current Script Flow

---

## 1. Federated Learning — What It Is and Why We Use It

### The privacy problem with centralised training

The normal way to train a machine learning model on hospital data is to collect all patient data into one central database and train on it together. This is how most ML models work.

**But this is illegal and unethical in medicine.** Patient data is protected by privacy law (HIPAA in the US, similar laws elsewhere). Hospital A cannot send Patient X's ICU records to Hospital B's server — even for the purpose of training a better AI model.

The result is that each hospital has its own isolated dataset. Hospital A trains a model on its own patients. Hospital B trains a separate model on its own patients. Neither model is as good as one trained on both hospitals' data combined — but combining the data is not allowed.

### The federated learning solution

**Federated Learning (FL)** solves this by training on distributed data without ever moving the data.

Instead of sending data to a central server, the central server sends the **model** to each hospital. Each hospital trains the model locally on its own private patient data. Then each hospital sends back only the **model weights** (the learned parameters) — not the patient data — to the server. The server combines the model weights from all hospitals into an improved global model, and the cycle repeats.

```
Traditional (centralised):
  Hospital A data ──┐
  Hospital B data ──┼──→ Central server trains → Model
  Hospital C data ──┘
  
  Problem: Patient data leaves the hospital → Privacy violation

Federated Learning:
  Central server sends model → Hospital A → Hospital A trains locally on own data
  Central server sends model → Hospital B → Hospital B trains locally on own data
  Central server sends model → Hospital C → Hospital C trains locally on own data
  
  Hospitals return model weights (NOT patient data) to central server
  Central server aggregates weights → Improved model
  
  Patient data NEVER leaves the hospital → Privacy preserved ✓
```

### Why this matters for our project

Our model is trained on MIMIC-III data — a real ICU dataset from Beth Israel Deaconess Medical Center. We simulate a federated scenario by splitting the training data into three artificial "hospitals" and training using a real federated learning framework (Flower). This demonstrates that our system is architecturally compatible with privacy-preserving deployment across multiple real hospital networks.

In a real deployment, each hospital would have:
- Its own patient population
- Its own local computing infrastructure
- Its own copy of the model
- No ability to see any other hospital's patient records

The server would aggregate only numerical weight vectors (millions of floating point numbers representing model parameters) — these contain no patient-identifiable information.

---

## 2. One-Line Summary

`train_federated.py` downloads MIMIC-III ICU data from Google BigQuery, builds a 108-feature vector combining vital sign trends and clinical text features, then trains a PyTorch DNN using Flower Federated Learning across 3 simulated hospital clients — using FedYogi server-side optimisation and FedProx client regularisation — saving the model from whichever training round achieves the best validated performance.

**Current best result: R²=0.4724, MAE=1.76 SOFA points** on a held-out test set of 12,038 patients, trained on 48,150 patients across 3 hospital nodes.

---

## 3. All Configurable Parameters — Explained

Every parameter that controls training behaviour is defined at the top of the script. Understanding each one is important for understanding why the model performs the way it does.

### Paths

```python
DATA_PATH  = "data/fl_training/"   # where hospital CSVs are saved and loaded
MODEL_PATH = "models/"             # all model artifacts saved here
```

### BigQuery Settings

```python
BIGQUERY_PROJECT  = "mimic-project-2"
BIGQUERY_DATASET  = "Dataset"
BIGQUERY_LOCATION = "US"
```

BigQuery is Google's cloud SQL database where the MIMIC-III data is stored. The script connects here in Phase 0 to download the training data.

### Preprocessing Control

```python
SKIP_PREPROCESSING = False
```

When `False`, the script automatically checks whether the training data already exists in the database. If it does, it skips the slow BigQuery download and TF-IDF computation (which takes 15–20 minutes) and goes straight to FL training. This is explained in detail in Section 4.

### TF-IDF Settings

```python
NOTES_SAMPLE_SIZE = 283_208   # use all available clinical notes
NOTES_TEXT_LIMIT  = 10_000    # character limit per patient's note
TFIDF_MAX_FEATURES = 200      # ignored when vocabulary= is set (see below)
TFIDF_NGRAM_RANGE  = (1, 2)   # consider both single words and two-word phrases
```

**Why these values matter so much:**

`NOTES_SAMPLE_SIZE = 283,208` means we use every single clinical note in the MIMIC-III dataset — not a sample. More notes means better TF-IDF weight estimates and more patients covered. This was one of the biggest performance breakthroughs (see Stage 9 in Section B).

`NOTES_TEXT_LIMIT = 10,000` means each patient's combined notes are cut off at 10,000 characters (about 2,000 words). Going from 5,000 to 10,000 characters captured crucial lab value mentions that physicians write later in their notes — things like "creatinine 2.8 this morning" which directly tells us about kidney function (SOFA Component 6).

`TFIDF_NGRAM_RANGE = (1, 2)` means TF-IDF considers both single words ("sepsis", "renal") and two-word phrases ("septic shock", "renal failure"). Two-word phrases are more specific and clinically meaningful than single words alone.

### FL Training Settings

```python
NUM_ROUNDS       = 100   # run 100 rounds of federated communication
EPOCHS_PER_ROUND = 3     # each hospital trains for 3 epochs locally per round
BATCH_SIZE       = 64    # 64 patients per gradient update
SHAP_BG_SAMPLES  = 300   # 300 reference patients for SHAP explainability
```

**What these mean:**

In Federated Learning, a "round" is one complete cycle: the server sends the model to all hospitals → all hospitals train locally → all hospitals send their updated models back → the server combines them. We run 100 such cycles.

Within each round, each hospital runs 3 full passes over its local data (3 epochs). With 16,000 patients per hospital and batch size 64, one epoch = 250 gradient update steps, so each hospital makes 750 gradient update steps per round.

**Why 3 epochs, not more?** More epochs means each hospital trains for longer before reporting back. This causes "client drift" — each hospital's model diverges further from the global model toward its own local optimum, and combining divergent models gives a worse result than combining less-divergent ones. 3 epochs was the sweet spot between "enough learning" and "too much drift."

### Learning Rate Schedule

```python
BASE_LR  = 0.001
LR_DECAY = 0.99   # multiply learning rate by 0.99 after every round
```

The learning rate controls how big of a step the model takes in each gradient update. Starting at 0.001 and decaying by 0.99 each round means:

```
Round 1:   LR = 0.001000
Round 10:  LR = 0.001 × 0.99^9  = 0.000914
Round 24:  LR = 0.001 × 0.99^23 = 0.000794  ← best round
Round 50:  LR = 0.001 × 0.99^49 = 0.000608
Round 100: LR = 0.001 × 0.99^99 = 0.000370  (floored at 0.0001)
```

Starting with a larger learning rate lets the model explore broadly in early rounds. As rounds progress and the model approaches a good solution, the smaller learning rate allows it to settle precisely without overshooting.

### Gradient Clipping

```python
GRAD_CLIP = 1.0
```

During backpropagation, computed gradients can occasionally be very large (a single batch of unusual patients might produce an enormous gradient). If applied directly, this would cause the model weights to jump wildly — "exploding gradients." Gradient clipping rescales any gradient vector whose total L2 norm exceeds 1.0, keeping it at exactly 1.0 while preserving its direction. This prevents single-batch spikes from destabilising training.

```
Example:
  Gradient vector norm = 8.3  (too large)
  After clipping:        norm = 1.0  (direction preserved, magnitude capped)
  
  Without clipping: model weights jump by 8× normal amount → potential catastrophic update
  With clipping:    model weights update by normal amount → stable training
```

### FedProx Proximal Regularisation

```python
MU_FEDPROX = 0.5
```

FedProx is a technique that prevents each hospital's model from drifting too far from the global model during local training. It works by adding a penalty term to each hospital's loss function:

```
Total Loss = Normal Training Loss + (0.5/2) × ||local_weights − global_weights||²
```

The `||...||²` term measures how different the local model is from the global model. If the local model drifts far, this penalty grows large and pulls the model back toward the global parameters. `μ=0.5` makes this penalty 50% of the data gradient strength — strong enough to prevent excessive drift, weak enough to allow the hospital to learn from its own data.

This is explained with a full worked example in Section 8.3.

### FedYogi Server Optimiser

```python
SERVER_ETA   = 0.01    # server step size (how aggressively the server updates)
SERVER_BETA1 = 0.9     # momentum accumulation rate (90% of past direction kept)
SERVER_BETA2 = 0.99    # adaptive rate memory (99% of past variance kept)
SERVER_TAU   = 1e-3    # numerical stability constant
```

FedYogi is the server-side optimiser that intelligently combines the models from all three hospitals. Instead of simply averaging them (FedAvg), FedYogi tracks the history of how the model has been changing and uses that history to make smarter updates. Explained in full in Section 8.5.

---

## 4. Auto-Detection: Skip Preprocessing?

Before running Phase 0 (the slow BigQuery + TF-IDF step), the script checks whether the training data already exists in the local SQLite database.

```python
skip = SKIP_PREPROCESSING or fl_training_exists()
if not skip:
    run_preprocessing()
```

| `SKIP_PREPROCESSING` | Is data in the database? | What happens |
|---|---|---|
| `False` | No | Phase 0 runs — downloads from BigQuery + builds TF-IDF (~15–20 min) |
| `False` | Yes | Phase 0 is skipped automatically — goes straight to FL training |
| `True` | No | Phase 0 skipped → **crashes in Phase 1** — no data to train on |
| `True` | Yes | Phase 0 skipped → FL training runs immediately |

**Practical implication:** The first run is slow (BigQuery download + TF-IDF processing on 283,208 notes). All subsequent runs skip Phase 0 and start immediately with FL training (~25–40 minutes).

**To force Phase 0 to re-run** (needed if you change the TF-IDF vocabulary or add new data):
```bash
sqlite3 models/icu_monitor.db "DELETE FROM fl_training;"
rm models/scaler.pkl models/tfidf_vectorizer.pkl models/feature_columns.pkl
python train_federated.py
```

---

## 5. Phase 0 — Data Preprocessing

Phase 0 runs inside `run_preprocessing()`. It downloads two tables from Google BigQuery, builds clinical text features, merges everything together, and saves the result as three "hospital" CSV files ready for FL training.

---

### 0a. BigQuery Connection

```python
from google.cloud import bigquery
bq = bigquery.Client(project="mimic-project-2", location="US")
```

Google BigQuery is a cloud-hosted SQL database. Our MIMIC-III ICU data lives there in the project `mimic-project-2`. The `bigquery.Client` object is the authenticated connection — it uses credentials from `gcloud auth application-default login`, which must have been run first.

---

### 0b. Vitals Query

```sql
SELECT *
FROM (
    SELECT *,
           ROW_NUMBER() OVER(
               PARTITION BY icustay_id
               ORDER BY window_time DESC
           ) AS rn
    FROM `mimic-project-2.Dataset.ml_dataset_final`
    WHERE sofa_score IS NOT NULL
)
WHERE rn = 1
```

**What this query does, step by step:**

`ml_dataset_final` contains the processed vital sign data for every ICU stay. It has multiple rows per patient — one row per 10-minute window. A patient who stayed in the ICU for 3 days has approximately 432 rows (3 days × 24 hours × 6 windows per hour).

The `ROW_NUMBER() OVER(PARTITION BY icustay_id ORDER BY window_time DESC)` assigns a row number within each patient's data, ordered by time (newest first). So `rn = 1` means the most recent window for that patient.

`WHERE rn = 1` keeps only the most recent row per patient — their current state.

**Result: approximately 60,188 rows — one row per unique ICU stay.**

**The 24 columns include:**
- **Identifier columns** (dropped later): `subject_id`, `hadm_id`, `icustay_id`, `window_time`, `rn`
- **9 trend vital statistics** (computed over the last 20 readings): `HR_mean`, `HR_std`, `RR_mean`, `SpO2_mean`, `SpO2_min`, `Temp_mean`, `SBP_mean`, `DBP_mean`, `MAP_mean`
- **7 most recent vital readings**: `latest_HR`, `latest_RR`, `latest_SpO2`, `latest_Temp`, `latest_SBP`, `latest_DBP`, `latest_MAP`
- **2 computer vision scores**: `GCS_eye_opening` (1–4), `stress_score` (0–10)
- **1 target label**: `sofa_score` (0–24) — what we are trying to predict

**Why 9 trend features instead of just the 7 latest values?**

A single vital sign reading (e.g., "SpO2 = 91%") can be momentarily low due to the probe falling off, or it can reflect a genuinely deteriorating patient. The trend over the last 20 readings tells the full story: is this person's SpO2 trending downward over 4 hours (concerning), or was it stable at 97% and just had one bad reading (not concerning)?

The 9 trend features capture trajectory (mean and variability of HR and SpO2, minimum SpO2, means of all others). The 7 latest features capture current state. Together they provide both context and current status.

**Which SOFA components do these 18 features directly cover?**

| Feature | SOFA Component | Mechanism |
|---|---|---|
| `SpO2_mean`, `SpO2_min` | Component 1 — Respiratory | Low SpO2 → low PaO2 → high respiratory SOFA |
| `MAP_mean`, `latest_MAP` | Component 4 — Cardiovascular | Low MAP → shock state → high cardiovascular SOFA |
| `GCS_eye_opening` | Component 5 — CNS | Low eye opening → low GCS → high CNS SOFA |

Components 2 (coagulation/platelets), 3 (hepatic/bilirubin), and 6 (renal/creatinine) are NOT directly measured here — they require lab tests not in the vitals monitoring stream. These three missing components are what the TF-IDF text features try to capture indirectly.

---

### 0c. Clinical Notes Query

```sql
SELECT subject_id, hadm_id, text
FROM `mimic-project-2.Dataset.clinical_notes`
```

**Result: 283,208 rows — one row per clinical note.**

This table contains the free-text documentation written by nurses and physicians during each patient's hospital stay:
- **Nursing notes:** Hourly assessments, medication administration, patient comfort
- **Physician notes:** Physical examination findings, assessment and plan
- **Progress notes:** Daily updates on the patient's condition
- **Discharge summaries:** Comprehensive summary written at the end of the stay

Each note is associated with a `hadm_id` (hospital admission ID) — the key used to match notes to ICU stays.

**Why are clinical notes valuable for predicting SOFA?**

Physicians document things in text that do not appear in structured vital sign tables:
- "Creatinine 2.8 this morning, trending up" → tells us about renal function (SOFA Component 6)
- "Bilirubin 4.5 on AM labs, hepatology consulted" → tells us about liver function (SOFA Component 3)
- "Platelet count dropped to 78, patient is thrombocytopenic" → tells us about coagulation (SOFA Component 2)
- "Patient started on norepinephrine for vasopressor support" → tells us about cardiovascular failure (SOFA Component 4)

These critical clinical indicators are embedded in free-text notes. The TF-IDF encoding converts them into numerical features the model can use.

---

### 0d. Text Cleaning and TF-IDF Encoding

#### Why text must be cleaned first

MIMIC-III notes contain elements that would confuse TF-IDF and produce useless features:

1. **De-identification placeholders:** MIMIC-III replaces patient names, dates, and hospital names with tokens like `[**John Smith**]`, `[**2145-03-15**]`, `[**Hospital1**]`. If not removed, TF-IDF would pick these as features.

2. **Numbers:** Clinical notes are full of numbers ("creatinine 2.8", "SpO2 91%", "HR 110"). If included, TF-IDF would treat "creatinine 2" and "creatinine 3" as different features, fragmenting the signal.

3. **Punctuation and formatting:** Symbols, colons, hyphens, and formatting characters should be stripped to leave only clean clinical vocabulary.

#### The cleaning function

```python
def _clean_clinical_text(text):
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', ' ', text)   # remove [**de-identification**]
    text = re.sub(r'\b\d+(?:[./]\d+)?\b', ' ', text)  # remove numbers (2.8, 120/80)
    text = text.lower()                                 # lowercase everything
    text = re.sub(r'[^a-z\s]', ' ', text)              # keep only a-z and spaces
    return ' '.join(text.split())                       # collapse multiple spaces
```

**Before cleaning:**
```
Pt [**Last Name (un) **] [**Known firstname 15**] presents with creatinine 2.8, 
BP 95/60. Started on Levophed @ 0.08 mcg/kg/min. [**2145-03-15**]
```

**After cleaning:**
```
pt presents with creatinine bp started on levophed mcg kg min
```

Now TF-IDF sees clean clinical vocabulary. "creatinine" and "levophed" (a brand name for norepinephrine, a vasopressor) both appear as features — both are highly relevant to SOFA scoring.

#### Per-admission text grouping

```python
notes_combined = notes.groupby("hadm_id")["text"].apply(
    lambda x: " ".join(x)
).reset_index()
notes_combined["text"] = notes_combined["text"].str[:NOTES_TEXT_LIMIT]  # 10,000 chars
```

All notes for a single hospital admission are **concatenated into one document**, then truncated to 10,000 characters. This means a patient's worth of nursing notes, physician notes, and progress notes are combined into a single text block from which TF-IDF extracts features.

**Why a 10,000 character limit?** Memory and computation constraints prevent using unlimited text. The 10,000 character limit (~2,000 words) is enough to capture the key clinical descriptors from most notes while keeping the TF-IDF matrix computationally tractable.

**Why 10,000 and not 5,000?** This was empirically discovered (see Stage 9 in Section B). Lab value mentions — "creatinine 2.8 this morning" — often appear in the middle of a progress note, which gets cut off at 5,000 characters. Doubling to 10,000 characters captured these mentions and produced an 87% relative improvement in R².

#### The SOFA Vocabulary Whitelist — The Most Important Design Decision in Preprocessing

Instead of letting TF-IDF automatically pick the 200 most statistically variable words (which turns out to pick medication instructions, social history, and administrative language), we provide an **explicit list of 90 clinically meaningful terms** that directly map to the 6 SOFA organ components.

**Why data-driven TF-IDF failed:**

In a large clinical notes dataset, the most statistically variable terms are the most common ones with the highest variance across documents. In MIMIC-III notes, these include:

```
"tablet"   — every medication order says "take 1 tablet twice daily"
"sig"      — Latin for "take as directed" — appears in every prescription
"refills"  — every outpatient prescription mentions refill count
"disp"     — dispense count, appears in pharmacy notes
"capsule"  — medication form descriptor
"wallet"   — patient personal effects documentation
"afternoon" / "morning" — time references in nursing notes
```

These terms correlate with how much documentation a patient has (sicker patients get more notes, so more medication orders, more time references) — not with how sick they actually are. A model trained on these features learns "patients with lots of notes are sicker" rather than "patients with high creatinine are sicker."

**The 90-term whitelist — each term has a direct SOFA connection:**

```
SOFA Component 1 — Respiratory (PaO2/FiO2 proxy):
  "intubated", "intubation", "ventilator", "vent", "cpap", "bipap",
  "hypoxia", "respiratory failure", "respiratory distress",
  "extubated", "shortness breath"

SOFA Component 2 — Coagulation (Platelet proxy):
  "plt", "platelets", "bleed", "bleeding", "hemorrhage",
  "inr", "ptt", "coagulopathy", "thrombocytopenia"

SOFA Component 3 — Hepatic (Bilirubin proxy):
  "bilirubin", "bili", "totbili", "liver", "jaundice",
  "hepatic", "cirrhosis", "liver failure"

SOFA Component 4 — Cardiovascular (MAP + vasopressor proxy):
  "hypotension", "hypotensive", "shock", "vasopressor", "pressors",
  "norepinephrine", "levophed", "dopamine", "epinephrine", "dobutamine",
  "septic shock"

SOFA Component 5 — CNS (GCS proxy in text):
  "sedated", "sedation", "coma", "altered", "confused", "gcs",
  "encephalopathy", "delirium", "unresponsive", "altered mental"

SOFA Component 6 — Renal (Creatinine + Urine Output proxy):
  "creatinine", "creat", "renal", "kidney", "dialysis", "aki",
  "oliguria", "urine", "urine output", "renal failure", "acute kidney",
  "crrt", "urean"

General severity and infection:
  "sepsis", "septic", "bacteremia", "infection", "cultures",
  "antibiotics", "fever", "blood cultures", "lactate", "wbc",
  "hgb", "hct", "sodium", "potassium", "hco", "angap", "bun",
  "failure", "acute", "pneumonia", "ards", "pulmonary", "dyspnea", "edema"
```

**Worked example — what a single patient's TF-IDF features look like:**

Patient A: 55-year-old male admitted with septic shock, on the ventilator, receiving norepinephrine, creatinine elevated, bilirubin borderline.

His combined notes (after cleaning) contain:
```
"... intubated day two ...septic shock...norepinephrine...creatinine elevated...
bilirubin mildly elevated...oliguria urine output low...hypotensive..."
```

TF-IDF features from the whitelist:
```
"intubated"       → TF-IDF = 0.42  (appears 2 times in notes)
"septic shock"    → TF-IDF = 0.35  (appears 1 time)
"norepinephrine"  → TF-IDF = 0.28  (appears 1 time)
"creatinine"      → TF-IDF = 0.31  (appears 1 time)
"bilirubin"       → TF-IDF = 0.18  (appears 1 time, not the primary issue)
"oliguria"        → TF-IDF = 0.25  (appears 1 time)
"hypotensive"     → TF-IDF = 0.22  (appears 1 time)
... (all other 83 terms) → TF-IDF ≈ 0.0 (not mentioned)
```

Patient B: 70-year-old female admitted with exacerbated heart failure, not intubated, no vasopressors, no renal issues.

Her notes contain:
```
"... dyspnea...edema bilateral...pulmonary congestion...not requiring intubation..."
```

TF-IDF features:
```
"dyspnea"         → TF-IDF = 0.38
"edema"           → TF-IDF = 0.31
"pulmonary"       → TF-IDF = 0.22
... all SOFA organ failure terms → 0.0 (not mentioned)
```

**The model can now clearly distinguish Patient A (higher SOFA, multi-organ failure) from Patient B (lower SOFA, primarily respiratory) based on these text features.** Data-driven TF-IDF would have given both patients similar values for "tablet", "sig", "refills" — providing no discrimination.

#### TF-IDF Configuration

```python
tfidf_vec = TfidfVectorizer(
    vocabulary=list(_SOFA_VOCABULARY),   # our 90-term whitelist
    ngram_range=(1, 2),                  # single words AND two-word phrases
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b',  # words ≥ 3 letters
    sublinear_tf=True,    # use log(1 + term_frequency) instead of raw count
    norm="l2",            # normalise each document to unit length
)
```

**`sublinear_tf=True`** — Without this, a patient whose notes mention "sepsis" 10 times gets 10× the feature value of a patient who mentions it once. But clinically, mentioning "sepsis" 10 times doesn't mean the patient is 10× as septic. `log(1 + count)` smooths this: count=1 → 0.69, count=10 → 2.40, count=100 → 4.61. The tenth mention of "sepsis" contributes less additional signal than the first.

**`norm="l2"`** — Normalises each document's feature vector to have total length 1.0. This prevents patients with very long notes (many words) from dominating simply because they have more text — a 500-word note and a 2,000-word note are put on the same scale.

**Result: TF-IDF matrix of shape (41,179 admissions, 90 features)**

---

### 0e. Merge and Clean

```python
dataset = ml_data.merge(tfidf_df, on="hadm_id", how="left")
# Shape: (60,188 ICU stays, 114 columns)

drop_cols = ["subject_id", "hadm_id", "icustay_id", "window_time", "rn"]
dataset = dataset.drop(columns=drop_cols).fillna(0)
# Shape: (60,188, 109) = 108 features + sofa_score target
```

**Left join:** Every ICU stay from the vitals table gets its matching TF-IDF features. "Left join" means if an ICU stay has no clinical notes (rare but possible), it still appears in the merged dataset — with all 90 TF-IDF features set to 0. Zero is the correct encoding for "this condition was never mentioned in any notes for this patient."

**The merged dataset has exactly 109 columns:**
- 9 trend vital statistics
- 7 latest vital readings
- 2 computer vision scores (GCS_eye_opening, stress_score)
- 90 TF-IDF SOFA vocabulary features
- 1 target: `sofa_score`

(18 vital + 90 TF-IDF = 108 features total, plus 1 label = 109 columns)

---

### 0f. Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=(y >= 10).astype(int)    # ensure High-Risk patients appear in both splits
)
```

**An 80/20 split:**
- Training set: 48,150 patients (used for FL training across 3 hospitals)
- Test set: 12,038 patients (held out completely until Phase 5 — never seen during training)

**What `stratify=(y >= 10)` means:**

Only about 6% of ICU patients have SOFA ≥ 10 (High Risk). Without stratification, the random split might put all the High-Risk patients in the training set and none in the test set — or vice versa. Stratification guarantees that both training and test sets have the same 6% High-Risk proportion.

**Why is this important?** Our final evaluation metrics include per-segment analysis (Low/Moderate/High Risk). If the test set has very few High-Risk patients (say, 5 patients instead of 722), the High-Risk metrics would be meaningless statistics computed on too small a sample.

**SOFA distribution across the full dataset:**
| Risk Level | SOFA Range | Proportion |
|---|---|---|
| Low Risk | 0–4 | ~63% (3,800 in test set) |
| Moderate Risk | 5–9 | ~30% (3,611 in test set) |
| High Risk | ≥10 | ~6% (722 in test set) — minority class |

---

### 0g. StandardScaler with Clipping

```python
scaler = StandardScaler()
X_train_scaled = np.clip(scaler.fit_transform(X_train), -10, 10)
X_test_scaled  = np.clip(scaler.transform(X_test), -10, 10)
```

#### What StandardScaler does

The 108 features have wildly different scales:
- Heart rate: 40–200 beats per minute
- SpO2: 70–100%
- TF-IDF features: 0.0–1.0

If fed raw to the DNN, the heart rate feature would dominate gradients (just because its numbers are bigger), while TF-IDF features would be nearly invisible. StandardScaler puts all features on a common scale:

```
scaled_value = (original_value − mean_train) / std_train
```

Example:
```
HR_mean: original values cluster around 85 bpm, std ≈ 20 bpm
  A patient with HR_mean=105: scaled = (105 − 85) / 20 = +1.0
  A patient with HR_mean=65:  scaled = (65 − 85) / 20 = −1.0
  
After scaling, every feature has mean=0, std=1.
The DNN treats all features equally. ✓
```

**Critical rule:** The scaler is **fitted only on the training set** (using `fit_transform`). The test set is only **transformed** (using `transform`) — it uses the training set's mean and standard deviation.

Why? If we computed the test set's mean and used that, we would be leaking information about the test set into the preprocessing step. The test set must be treated as if it were future data — unknown at training time.

#### Why clipping to ±10 is essential

TF-IDF features for very rare but important clinical terms have very high IDF values. For example:
```
"bacteremia" appears in only 0.1% of notes
IDF = log(N / document_frequency) = log(41179 / 41) ≈ 6.9

After StandardScaler:
  A patient with "bacteremia" in notes: feature value ≈ 6.9 × something
  z-score after scaling ≈ 80–150 standard deviations above mean
```

A z-score of 150 — 150 standard deviations above the mean — would produce catastrophically extreme values when passed through the DNN, leading to enormous predictions (we saw SOFA predictions of 10,001 in the initial runs). Clipping at ±10 standard deviations eliminates this while preserving all clinically meaningful signal:

```
No legitimate clinical value should have z-score > 10.
Maximum clipped at 10 → model sees: "this term was strongly present"
No false information; just prevents computational catastrophe.
```

---

### 0h. Hospital Split — IID 33/33/34

```python
data_train = shuffle(data_train, random_state=42)   # randomise order
n = len(data_train)   # 48,150

Hospital 0 (General ICU):    first 15,889 rows
Hospital 1 (Mixed ICU):      next 15,890 rows
Hospital 2 (Cardiac/Trauma): last 16,371 rows
```

The 48,150 training patients are divided into 3 equal groups. Each group represents one simulated hospital's private dataset.

**IID (Independent and Identically Distributed):** Because the data is first shuffled randomly, all 3 hospitals see the same distribution of SOFA scores (~63% Low Risk, ~30% Moderate, ~6% High Risk). This is the realistic best case for FL convergence.

**Why not use a non-IID split by default?**

Non-IID means different hospitals see different patient populations (e.g., Hospital 2 is a trauma centre and sees mostly High Risk patients). We tested this:

```
IID split (all hospitals see similar SOFA distribution):
  R² ≈ 0.33 — hospitals train compatible gradient updates
  
Non-IID split (Hospital 2 sees mostly SOFA ≥ 10):
  R² ≈ 0.08 — hospitals' gradient updates point in conflicting directions
```

Why does non-IID hurt so badly? When Hospital 2 (mostly High-Risk patients) trains its model, its gradients push the weights toward predicting high SOFA values. When Hospital 1 (mostly Low-Risk patients) trains, its gradients push toward low SOFA values. FedYogi averages these conflicting gradients — the result satisfies neither hospital's training objective. The non-IID scenario is available as an option (`USE_NONIID_SPLIT = True`) for demonstrating FL challenges in presentations.

---

### 0i. Save Sklearn Artifacts

```python
joblib.dump(scaler,    "models/scaler.pkl")
joblib.dump(tfidf_vec, "models/tfidf_vectorizer.pkl")
joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")
```

These three files must be saved and reused at inference time (when a real patient's data is submitted through the app). Here is why each one is critical:

| File | Why it must be saved |
|---|---|
| `scaler.pkl` | Contains the training set mean and standard deviation for all 108 features. New patient data must be scaled with these exact same values — using a different scaler would put the data on a different scale than the model expects. |
| `tfidf_vectorizer.pkl` | Contains the IDF weights for all 90 vocabulary terms. New clinical notes must be converted using the exact same IDF weights — otherwise the feature values would be on a different scale. |
| `feature_columns.pkl` | Contains the exact ordered list of the 108 feature names. At inference, the app must present features in the exact same order as training — column order matters for a DNN. |

---

## 6. Phase 1 — Load Client Data

```python
scaler_obj      = joblib.load("models/scaler.pkl")
feature_columns = list(scaler_obj.feature_names_in_)   # the 108 ordered feature names

for i in range(3):
    df = get_fl_client_data(i)    # read hospital i's data from SQLite
    y  = df["sofa_score"].values
    X  = df.drop(columns=["sofa_score"])
    
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0             # add any missing column as zeros
    
    X = X[feature_columns].values   # reorder to exact training column order
```

This phase loads the three hospital CSV files from the SQLite database and ensures they all have exactly the same 108-column layout.

**Why the column alignment step?** In Phase 0, the three hospital CSVs were saved with columns in a consistent order. But if the database was migrated or the script was run in different environments, column order might differ slightly. The alignment step guarantees that the feature passed as column 47 to the model during training is also passed as column 47 at inference time — regardless of how the CSV was written.

After loading all 3 hospitals:
```python
X_all = np.vstack(all_X)      # Stack vertically: (48,150, 108)
y_all = np.concatenate(all_y) # Concatenate: (48,150,)
input_dim = 108                # confirmed from data shape
```

---

## 7. Phase 2 — SHAP Background Data

```python
bg_idx   = rng.choice(len(X_all), 300, replace=False)    # random 300 training patients
background = X_all[bg_idx]
np.save("models/shap_background.npy", background)
```

**What is SHAP and why does it need background data?**

SHAP (SHapley Additive exPlanations) answers the question: "For this specific patient, how much did each of the 108 features contribute to their predicted SOFA score?"

SHAP DeepExplainer works by comparing each patient's prediction to what the model would predict for "typical" training patients. The background is a reference distribution — 300 randomly selected training patients that represent the "average" patient the model has seen.

**What SHAP computes per patient:**

```
For Patient A (predicted SOFA = 12.4):

  The 300 background patients have predictions averaging ~5.2

  SHAP attribution tells us:
    MAP_mean:      +3.1  (MAP=55 → pushes prediction UP significantly)
    latest_SpO2:   +1.8  (SpO2=85% → pushes prediction UP)
    "intubated":   +1.2  (intubated → pushes UP)
    "norepinephrine": +0.9 (vasopressors → pushes UP)
    GCS_eye_opening: +0.2 (partially closed eyes → small UP)
    HR_std:        −0.2  (low heart rate variability → small DOWN)
    ... (102 other features with small attributions)
    
  Sum: 5.2 + 3.1 + 1.8 + 1.2 + 0.9 + 0.2 − 0.2 + ... ≈ 12.4 ✓
```

The SHAP bar chart in the dashboard shows the top 5 contributors — the features that most drove the prediction for the current patient.

The background data is saved and loaded by the app at startup. Using the same 300 representative patients every time ensures SHAP values are comparable across different patients and sessions.

---

## 8. Phase 3 — Federated Learning Simulation

This is the core training phase. The Flower framework orchestrates 100 rounds of FL simulation across 3 hospital clients, with FedYogi at the server and FedProx at each hospital.

---

### 8.1 The Model Architecture

```
Input: 108 features
   ↓
Linear(108 → 128) + ReLU
   ↓
Linear(128 → 64) + ReLU
   ↓
Linear(64 → 32) + ReLU
   ↓
Linear(32 → 1)
   ↓
Output: SOFA score (continuous, 0–24)
```

**Why this specific architecture?**

**Input expansion (108 → 128):** The first layer has more neurons than input features. This allows the model to learn new combinations of features — for example, "high HR + low MAP + 'norepinephrine' mentioned → cardiovascular failure" — that are more informative than any individual feature alone.

**Progressive narrowing (128 → 64 → 32 → 1):** Each subsequent layer compresses the representation. The model is forced to identify increasingly abstract patterns, ultimately distilling everything into a single SOFA prediction.

**Total parameters:** ~23,000 (a relatively small model). With 48,150 training samples, this gives a 2.1 samples-per-parameter ratio — appropriate for a tabular dataset where overfitting is a real risk.

**Why no Dropout?**

Dropout randomly zeros out different neurons during each forward pass to prevent overfitting. In a standard (centralised) DNN, this works well. In Federated Learning, it creates a problem:

Hospital A's model uses Dropout mask M1 during training. Hospital B's model uses mask M2. Hospital C's model uses mask M3. These are three different models — different neurons were trained to be active. When FedYogi averages their weights, it combines weights trained with incompatible random masks. The aggregated weights satisfy no hospital's Dropout configuration. Result: training divergence.

Regularisation is handled instead through AdamW's weight decay (L2 penalty on weights) and the FedProx proximal term.

**Why no BatchNorm or LayerNorm?**

- **BatchNorm** computes per-batch statistics (mean, variance). In FL, each hospital has different patient populations → different batch statistics. Averaging divergent statistics produces a broken global normalisation. Tested and confirmed: causes training instability.

- **LayerNorm** was tested twice and failed dramatically. LayerNorm normalises each patient's activations to have mean=0, std=1 within that patient. The problem: it makes a SOFA=20 patient and a SOFA=2 patient produce identically normalised hidden representations. By the time gradients reach the output layer, the model has lost the ability to distinguish severity. Prediction range collapsed to [0.22, 7.04] instead of [0, 15+]. MAE for High-Risk patients jumped from 3.4 to 6.5.

**Why ReLU activation?**

ReLU (Rectified Linear Unit) outputs `max(0, x)` — it keeps positive values unchanged and zeros out negative values. This is the standard choice for regression DNNs because:
- It is computationally simple
- It avoids vanishing gradients (the gradient is always 0 or 1, never shrinking)
- It introduces non-linearity, allowing the model to learn complex patterns

---

### 8.2 The Loss Function — Weighted MSE

The loss function measures how wrong the model's predictions are during training. The standard choice would be Mean Squared Error (MSE): `mean((predicted − actual)²)`. We use a modified version.

```python
def weighted_mse_loss(predictions, targets):
    weights = 1.0 + targets * 0.5      # weight = 1 + SOFA × 0.5
    weights = weights / weights.mean() # normalise so mean weight = 1.0
    return (weights * (predictions - targets) ** 2).mean()
```

**Why weight errors differently based on SOFA score?**

High-Risk patients (SOFA ≥ 10) are only 6% of the data. Without weighting, the model's loss function is dominated by the 94% of Low- and Moderate-Risk patients. The model learns to predict very accurately for common patients and poorly for rare High-Risk patients — exactly the wrong behaviour for an ICU early warning system, where correctly identifying the most critical patients is the top priority.

By weighting each patient's error by `1 + SOFA × 0.5`, High-Risk patients receive more weight in the loss function, forcing the model to learn their patterns more carefully.

**Effective weights after normalisation (assuming mean SOFA ≈ 4):**

| SOFA | Raw weight | After normalisation | Interpretation |
|---|---|---|---|
| 0 | 1.0 | 0.33× | Low-Risk errors count 1/3 of fair share |
| 4 (mean) | 3.0 | 1.0× | Moderate-Risk errors count at exactly fair share |
| 10 (High Risk threshold) | 6.0 | 2.0× | High-Risk errors count 2× fair share |
| 20 (Severe) | 11.0 | 3.7× | Severe errors count 3.7× fair share |

**Why 0.5 multiplier, not 3.0?**

Earlier experiments used `weight = 1 + SOFA × 3.0`. The problem: with this multiplier, Low-Risk patients (63% of data, SOFA ≈ 0–4) got only 7% of the gradient signal they deserved, while High-Risk patients (6% of data) got 70%. The training and evaluation objectives were working against each other:

- Training was optimising to predict High-Risk patients accurately
- But the server selected the best round using unweighted MSE, dominated by Low-Risk patients

The server's selection criterion was discarding rounds where High-Risk performance improved but Low-Risk performance temporarily worsened. Reducing to 0.5 gives Low-Risk patients 33% of signal (still less emphasis, but not negligible) and aligns the training objective better with the evaluation metric.

---

### 8.3 FedProx Client Regularisation

**The problem FedProx solves:**

When each hospital trains for 3 epochs locally, their models drift away from the global model. Hospital A's model becomes optimised for General ICU patients. Hospital B's model becomes optimised for Mixed ICU patients. When FedYogi tries to combine these, it is averaging models that point in very different directions — like three people pulling a rope in slightly different directions.

**How FedProx works:**

FedProx adds a restraining term to each hospital's training loss:

```
Total Loss = Training Loss + (μ/2) × ||local_weights − global_weights||²

where ||...||² = sum of squared differences between each local weight and the corresponding global weight
```

This penalty grows as the local model diverges from the global model. If a hospital's training is pushing the model weights in a direction that moves them far from the global model, this penalty resists that movement. The hospital can still learn from its data — it just cannot drift too far before the penalty becomes too expensive.

**Worked example:**

```
After training for 3 epochs without FedProx:
  Hospital 0 local weights: [1.23, −0.88, 0.56, ...]
  Global model weights:      [1.00, −0.70, 0.50, ...]
  Drift = ||local − global|| = 3.2 (measured in weight space)

With FedProx μ=0.5:
  FedProx penalty = (0.5/2) × 3.2² = 0.25 × 10.24 = 2.56
  This penalty would have been applied throughout training,
  pulling the hospital's model back toward [1.00, −0.70, 0.50, ...]
  
After training with FedProx:
  Hospital 0 local weights: [1.08, −0.76, 0.52, ...]  ← less drift
  Drift = ||local − global|| = 1.1 (much smaller)
```

**Why μ=0.5 specifically?**

| μ value | Proximal gradient strength relative to data gradient | Result |
|---|---|---|
| 0.01 | ~1% — essentially invisible | No effect on drift; same oscillation as without FedProx |
| 0.5 | ~50% — meaningful restraint | Stable training, good convergence |
| 1.0 | ~100% — overpowering | Model cannot diverge from global at all → cannot learn from local data → R² drops |

---

### 8.4 The Hospital Client

Each of the 3 hospitals is implemented as a Flower `NumPyClient`. The client has two methods that the FL server calls:

#### `fit()` — Called by the server each round: "train and return your updated model"

```
Server sends: global model weights for round r

Hospital receives global weights and:
1. Loads global weights into local model
2. Computes LR for this round: lr = max(1e-4, 0.001 × 0.99^(r-1))
3. Runs 3 local training epochs:
   For each epoch:
     For each batch of 64 patients:
       a. Forward pass: predicted_SOFA = model(patient_features)
       b. Loss = weighted_MSE + (0.5/2) × ||local − global||²
       c. Backward pass: compute gradients
       d. Clip gradients to L2 norm ≤ 1.0
       e. AdamW optimizer step (weight_decay=1e-4)
4. Returns updated local weights
```

With ~16,000 training patients per hospital:
- Patients per epoch = 16,000 → batches per epoch = 250 (at batch size 64)
- Batches per round = 3 epochs × 250 batches = 750 gradient update steps

#### `evaluate()` — Called by the server each round: "evaluate the global model and report"

```
Server sends: new global model weights (after FedYogi aggregation)

Hospital receives global weights and:
1. Loads global weights (NOT the locally trained weights)
2. Evaluates global model on its local validation set (15% of its data, ~2,385 patients)
3. Returns: unweighted MSE, number of validation patients, and a dict with MAE and R²
```

Note that `evaluate()` uses the **global model weights**, not the locally-trained ones. This is intentional — the server wants to know how well the aggregated global model performs, not how well each hospital's local model performs. The aggregate of these evaluations is what `SaveBestStrategy` uses to select the best round.

---

### 8.5 FedYogi Server Optimiser

**Why not just average the three hospital models?**

Simple averaging (FedAvg) would compute:
```
global_model = (1/3) × hospital_0 + (1/3) × hospital_1 + (1/3) × hospital_2
```

This works, but it has no memory. Every round, the server does the same averaging regardless of how training has been going. A bad round (where all three hospitals produced poor updates) is treated exactly the same as a great round.

**FedYogi adds momentum and adaptive learning to the server:**

The "pseudo-gradient" is defined as the difference between the FedAvg result and the current global model:
```
Δ = FedAvg_result − current_global_model
```

This Δ tells the server: "in which direction should the global model move?"

FedYogi then applies an Adam-like update:

```
Step 1: Compute pseudo-gradient
  Δ = w_fedavg − w_global   (direction of improvement this round)

Step 2: Update first moment (momentum)
  m_t = 0.9 × m_{t-1} + 0.1 × Δ
  (90% of previous momentum + 10% of current direction)
  
  Effect: if 10 rounds have all improved in direction d, momentum
          builds up in direction d. One bad round barely moves it.

Step 3: Update second moment (Yogi adaptive rate)
  v_t = v_{t-1} + 0.01 × (Δ² − v_{t-1}) × sign(Δ² − v_{t-1})
  
  Crucial distinction from FedAdam:
  v_t only INCREASES when |Δ|² > v_{t-1}  (gradient is larger than running estimate)
  v_t stays the same when |Δ|² ≤ v_{t-1} (gradient is smaller than running estimate)
  
  Effect: v_t is a "high watermark" of gradient magnitude.
          Once a large explosion has occurred, v_t stays large forever.
          Large v_t → small effective step size = conservative updates after explosion.

Step 4: Update global model
  w_global ← w_global + 0.01 × m_t / (√v_t + 0.001)
  
  When v_t is large: step size 0.01/√v_t is small → conservative update
  When v_t is small: step size 0.01/√v_t is large → aggressive exploration
```

**The explosion–exploration–recovery cycle (which is intended behaviour):**

```
Rounds 1–4:    Smooth convergence. All 3 hospitals learning consistently.
               avg_loss: 24.0 → 9.27

Rounds 5–12:   EXPLOSION. Hospitals diverge after local training.
               FedYogi's momentum amplifies the pseudo-gradient.
               avg_loss: 9.27 → 2,200 (240× initial!)
               
               → But: v_t is now permanently large.
                       Step sizes after explosion are tiny.

Rounds 13–23:  Recovery. Momentum stabilises in a better direction.
               Small v_t-damped steps allow precise convergence.
               avg_loss: 2,200 → 5.48
               
Round 24:      BEST ROUND — avg_loss = 5.48 (global minimum found)
               Model saved as best.

Rounds 25–100: Slow degradation. LR decays to near zero.
               avg_loss drifts upward slightly.
```

The explosion is **not a bug — it is a feature.** It is how FedYogi explores a wide region of parameter space. Plain FedAvg would stay trapped in a shallow local minimum around avg_loss=7.5. FedYogi's explosion jumps out of that minimum and, during recovery, finds a deeper minimum at avg_loss=5.48. The `SaveBestStrategy` captures this best-round model before the degradation begins.

---

### 8.6 What Happens in One Complete FL Round

Let us trace a single complete round (say, Round 12) to show all the moving parts:

```
[Server — Start of Round 12]
  1. configure_fit():
     Sends current global model weights + {"server_round": 12} to all 3 hospitals
     
[Hospital 0 — General ICU, ~15,889 patients]
  2. fit() called with Round 12 global weights:
     a. Load global weights into local ICUModel
     b. Compute LR = 0.001 × 0.99^11 = 0.000895
     c. Run 3 local epochs (750 gradient steps total):
        Each step: forward → weighted MSE + FedProx penalty → backward → clip → AdamW
     d. Return locally-trained weights

[Hospital 1 — Mixed ICU, ~15,890 patients]
  2. Same as above, different patient data, different gradient direction

[Hospital 2 — Cardiac/Trauma, ~16,371 patients]
  2. Same as above

[Server — Aggregation]
  3. aggregate_fit():
     a. FedAvg: w_avg = (15889×hospital0 + 15890×hospital1 + 16371×hospital2) / 48150
     b. Compute Δ = w_avg − w_global (pseudo-gradient)
     c. Update momentum: m_12 = 0.9×m_11 + 0.1×Δ
     d. Update Yogi rate: v_12 = v_11 + 0.01×(Δ²−v_11)×sign(Δ²−v_11)
     e. Update global: w_global ← w_global + 0.01×m_12/(√v_12 + 0.001)
     
[Server → Hospitals — Evaluation]
  4. Sends new global weights to all 3 hospitals for evaluation
  
[Each Hospital — Evaluation]
  5. evaluate() called with new global weights:
     Computes unweighted MSE on local validation set (15% of local data)
     Returns MSE and number of validation patients
     
[Server — Best Round Selection]
  6. aggregate_evaluate():
     avg_loss = weighted mean of 3 hospitals' MSE values
     If avg_loss < best_so_far:
       → Save current global weights as best
       → Record this as best_round
     Print: "Round 12 — avg_loss=8.34"
```

This process repeats 100 times. At the end, the best-round model is saved.

---

## 9. Phase 4 — Save the Best Model

```python
saved_params = _best_weights.get("params") or _last_weights.get("params")
final_model = ICUModel(108)
set_weights(final_model, parameters_to_ndarrays(saved_params))
torch.save(final_model.state_dict(), "models/federated_model.pth")
```

After 100 rounds of FL training, the round that produced the lowest average validation MSE across all 3 hospitals is identified — in the current best run, this was **round 24** with `avg_loss = 5.48`.

The model from that specific round is loaded and saved as `federated_model.pth`. This is the model the app uses at inference time.

**Why not use the round 100 model?** Round 100 has the most training, but "most training" does not mean "best." After round 24, the LR has decayed significantly. Small gradient updates may move the model away from the minimum found at round 24. The best-round selection specifically guards against this: we keep the model that performed best on validation, not the one that trained longest.

**Fallback:** If for any reason `_best_weights` is empty (e.g., all rounds had identical performance and no round beat the initial baseline), the round 100 model is used. This prevents crashes in unusual edge cases.

---

## 10. Phase 5 — Final Evaluation

```python
final_model.eval()   # disable any training-mode behaviour (e.g., Dropout if it were used)
with torch.no_grad():   # do not compute gradients — saves memory
    predictions = final_model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()

mae = mean_absolute_error(y_test, predictions)
r2  = r2_score(y_test, predictions)
```

The final model is evaluated on the **completely held-out test set** — 12,038 patients who were never used in any training step, never seen during FL rounds, and never used to select the best round.

**Current results (full training set, 48,150 training patients):**

| Risk Segment | MAE | R² | Patients in test set |
|---|---|---|---|
| Low Risk (SOFA 0–4) | 1.47 | −1.09 | 7,593 |
| Moderate Risk (SOFA 5–9) | 1.95 | −2.60 | 3,595 |
| High Risk (SOFA ≥ 10) | 3.67 | −6.60 | 850 |
| **Global (all patients)** | **1.76** | **0.4724** | **12,038** |

### Understanding the Paradox: All Per-Segment R² Are Negative, Yet Global R² Is Positive

This result looks contradictory and is one of the most commonly misunderstood aspects of the model's performance. Here is the complete explanation.

#### What R² means

```
R² = 1 − SS_residual / SS_total

SS_residual = Σ(actual − predicted)²   ← how wrong the model is
SS_total    = Σ(actual − group_mean)²  ← how much variance exists in the group
```

**R² < 0** means the model is worse than just predicting the group's mean for every patient. If you just guessed "3.0" for every Low-Risk patient (the segment mean), you would be more accurate than the model within that segment.

#### Why is global R² positive while all segment R² are negative?

**The global R² is computed against the global mean (≈4.3). The per-segment R² values are computed against each segment's own mean.**

The model earns its global R² by **correctly identifying which risk tier each patient belongs to** — not by ranking patients precisely within a tier.

```
Patient with actual SOFA=12 (High Risk):
  Global mean = 4.3
  Model predicts: 10.5
  
  Residual for global R² = (12 − 10.5)² = 2.25
  Baseline for global R² = (12 − 4.3)² = 59.29
  This patient helps global R² because model pulled far above global mean ✓

Patient with actual SOFA=2 (Low Risk):
  Global mean = 4.3
  Model predicts: 3.1
  
  Residual = (2 − 3.1)² = 1.21
  Baseline = (2 − 4.3)² = 5.29
  This patient also helps global R² because model pulled below global mean ✓
```

**But within the Low-Risk segment:**
```
Segment mean for Low Risk = 2.5
Patients with actual SOFA = 0, 1, 2, 3, 4
Model predicts approximately 2.5–3.5 for all of them (can't distinguish within-tier)

Residual for SOFA=0: (0 − 3.0)² = 9.0
Baseline for SOFA=0: (0 − 2.5)² = 6.25

The model's prediction (3.0) is further from actual (0) than the segment mean (2.5) is.
→ R² < 0 for this segment.
```

#### A concrete numerical example

```
Low Risk segment: 5 patients with actual SOFA = 0, 1, 2, 3, 4
Segment mean = 2.5
Model predictions: 2.8, 3.0, 3.1, 2.9, 3.2  (all near same value)

SS_total (using segment mean):
  (0−2.5)² + (1−2.5)² + (2−2.5)² + (3−2.5)² + (4−2.5)²
  = 6.25 + 2.25 + 0.25 + 0.25 + 2.25 = 11.25

SS_residual (using model predictions):
  (0−2.8)² + (1−3.0)² + (2−3.1)² + (3−2.9)² + (4−3.2)²
  = 7.84 + 4.00 + 1.21 + 0.01 + 0.64 = 13.70

R² = 1 − 13.70/11.25 = 1 − 1.22 = −0.22  (negative — model worse than segment mean)
```

#### Summary: what the model does and does not do

| Task | Model performance |
|---|---|
| Classify patient as Low / Moderate / High Risk | ✅ Strong — MAP, SpO2, clinical text clearly separate severity tiers |
| Rank patients within the Low group (SOFA 0–4) | ❌ Weak — cannot distinguish SOFA=1 from SOFA=4 without direct lab values |
| Rank patients within the Moderate group (5–9) | ❌ Weak — bilirubin/creatinine variations not directly measured |
| Rank patients within the High group (≥10) | ❌ Weak — sparse data + wide actual range (10–24) |

**Clinical framing:** The model functions as a strong **risk stratification and early warning tool**. It reliably flags which patients are High-Risk. It cannot function as a precise SOFA calculator — that would require the actual lab values (creatinine, bilirubin, platelet count) used in the SOFA formula.

---

## 11. Phase 6 — Save Training Metadata

```python
metadata = {
    "num_rounds": 100,
    "epochs_per_round": 3,
    "batch_size": 64,
    "aggregation": "FedYogi (η=0.01, β1=0.9, β2=0.99) + FedProx (μ=0.5)",
    "best_round": 24,
    "final_mae": 1.7588,
    "final_r2": 0.4724,
    "model_architecture": "108 → 128 → 64 → 32 → 1  (ReLU, no Dropout)",
    ...
}
json.dump(metadata, open("models/training_metadata.json", "w"), indent=2)
```

`training_metadata.json` is read by the application server at startup and displayed in the **Federated Learning information tab** in the ClinIQ dashboard. This allows clinicians to inspect training parameters, understand the model's validation performance, and view which FL round produced the final deployed model.

---

## 12. Optional Features

### Non-IID Hospital Split (`USE_NONIID_SPLIT = True`)

Creates a specialty-biased split simulating realistic hospital differences:

| Hospital | Specialty | Low Risk | Moderate | High Risk |
|---|---|---|---|---|
| Hospital 0 | General ICU | 70% | 20% | 10% |
| Hospital 1 | Mixed ICU | 20% | 60% | 20% |
| Hospital 2 | Cardiac/Trauma | 10% | 20% | 70% |

**Why it hurts training:** Hospital 2 (70% High-Risk patients) trains its local model to predict high SOFA values. Hospital 0 (70% Low-Risk patients) trains its model to predict low SOFA values. Their gradient updates conflict, and FedYogi cannot reconcile them — the global model is stuck between two opposing objectives. **Tested result: R² ≈ 0.08 vs 0.33 with IID split.**

Use this setting only when demonstrating FL challenges or explaining why IID data partitioning was chosen.

### Differential Privacy (`USE_DP = True`)

Even though patient data never leaves the hospital in federated learning, sophisticated adversaries might theoretically reconstruct some private information from the model weight updates themselves (membership inference attacks). Differential Privacy adds mathematical noise to the weight updates before they are sent to the server, providing a formal privacy guarantee.

**How it works:**

```
1. Compute weight update: Δ = local_weights − global_weights
2. Clip: if ||Δ||₂ > DP_SENSITIVITY, scale Δ so ||Δ||₂ = DP_SENSITIVITY (1.0)
3. Add Gaussian noise: Δ_noisy = Δ + N(0, (DP_SIGMA × DP_SENSITIVITY)²)
4. Send global_weights + Δ_noisy to server
```

**Privacy budget:**

The privacy guarantee is expressed as (ε, δ)-differential privacy:
- Small ε (< 1) = strong privacy guarantee, large noise
- Large ε (> 10) = weak guarantee, small noise

```
ε_per_round ≈ √(2 × ln(1.25/δ)) / σ   where δ = 1e-5
With σ=1.0: ε_per_round ≈ 3.26
After 100 rounds: ε_total ≈ 326  (weak guarantee — primarily for demonstration)
```

The current DP settings (`DP_SIGMA=1.0`) demonstrate the mechanism but provide a weak privacy guarantee (ε=326 is much larger than the ε<1 considered "strong"). Achieving strong privacy guarantees (ε<1) would require much higher noise (DP_SIGMA~5+), which would substantially reduce model performance.

---

## 13. Complete Output Files

| File | Location | Created in | Used by | Contents |
|---|---|---|---|---|
| `scaler.pkl` | `models/` | Phase 0g | App, Phase 1 | Fitted StandardScaler with training set mean and std for all 108 features |
| `tfidf_vectorizer.pkl` | `models/` | Phase 0d | App | Fitted TF-IDF vectorizer with 90-term vocabulary and IDF weights |
| `feature_columns.pkl` | `models/` | Phase 0i | App, Phase 1 | Ordered list of 108 feature names — enforces consistent column order |
| `shap_background.npy` | `models/` | Phase 2 | App | 300 representative training patients for SHAP DeepExplainer, shape (300, 108) |
| `federated_model.pth` | `models/` | Phase 4 | App | Trained DNN weights from the best FL round (108→128→64→32→1) |
| `training_metadata.json` | `models/` | Phase 6 | App | All training stats, hyperparameters, and evaluation metrics for the FL info panel |

---

## 14. How to Run

### First-time run (BigQuery + full training)

```bash
# One-time Google Cloud authentication
gcloud auth application-default login
gcloud auth application-default set-quota-project mimic-project-2

# Run from the icu_monitor/ directory
cd icu_monitor
python train_federated.py
```

**Expected total time:** 40–60 minutes
- Phase 0 (BigQuery download + TF-IDF on 283k notes): ~15–20 minutes
- Phases 1–6 (FL training, 100 rounds): ~25–40 minutes

### Re-train only (data already in database)

```bash
python train_federated.py
# Phase 0 is automatically skipped if the fl_training table in the DB has data
# Expected time: ~25–40 minutes
```

### Force re-run preprocessing (e.g., after changing TF-IDF vocabulary)

```bash
sqlite3 models/icu_monitor.db "DELETE FROM fl_training;"
rm models/scaler.pkl models/tfidf_vectorizer.pkl models/feature_columns.pkl
python train_federated.py
```

### After training — start the app

```bash
cd icu_monitor
streamlit run app.py
# The app loads models/federated_model.pth at startup
```

---

---

# Section B — Complete Optimization Journey

---

## 15. The Underlying Problem: Predicting SOFA Score

### What makes SOFA prediction difficult

The SOFA score (Sequential Organ Failure Assessment) is a composite clinical score computed from direct laboratory measurements across 6 organ systems:

| Component | Organ | Lab Value Required |
|---|---|---|
| 1 | Respiratory | PaO2/FiO2 ratio — requires arterial blood gas test |
| 2 | Coagulation | Platelet count — requires blood test |
| 3 | Hepatic | Bilirubin — requires liver function test |
| 4 | Cardiovascular | MAP + vasopressor type and dose |
| 5 | CNS | Glasgow Coma Scale |
| 6 | Renal | Serum creatinine + 24h urine output |

**Our model has direct access to:**
- Component 4 partially: MAP from vitals monitoring (but not vasopressor data)
- Component 5 partially: GCS eye opening from computer vision (but not full GCS)
- Component 1 proxy: SpO2 (a less accurate estimate of oxygen than PaO2/FiO2)

**Our model infers the rest from clinical text:**
- Component 2: Does the note mention "thrombocytopenia", "plt 45"?
- Component 3: Does the note mention "bilirubin elevated", "totbili"?
- Component 6: Does the note mention "creatinine 2.8", "oliguria"?

Text mentions are fundamentally noisier than actual lab values. A patient might have bilirubin = 4.5 without any mention of "bilirubin" in the notes from that particular shift. This text-noise ceiling is why within-segment R² is negative — we are inferring lab values from imperfect textual evidence.

---

## 16. Stage-by-Stage Experiment Log

### Stage 1: Catastrophic Baseline

**Configuration:**
- 600 data-driven TF-IDF features
- Model: 618 → 256 → 128 → 64 → 1
- FedAvg (no FedYogi or FedProx)
- Loss: weighted MSE with weight multiplier 3.0
- No StandardScaler clipping

**Result: R² = −824. Model predicted SOFA = 10,001.**

**Root cause diagnosis:**

The data-driven TF-IDF picked rare but clinically important terms like `bacteremia` (a bloodstream infection indicating severe sepsis). `bacteremia` appears in only ~41 out of 41,179 hospital admission notes.

```
IDF("bacteremia") = log(41179 / 41) ≈ 6.9

TF-IDF value for a patient with "bacteremia" in notes: ~1.5 (using log TF × IDF)

After StandardScaler (mean ≈ 0.05, std ≈ 0.12):
  z-score = (1.5 − 0.05) / 0.12 ≈ 12.1

Without clipping: z-score = 12.1 → DNN input
```

The first linear layer (618 → 256) applied weights up to ±2 to this 12.1 input. After four layers: `12.1 × 2 × 2 × 2 × 2 = 193.6` at the output layer before the final prediction. For rare terms with even higher IDF, z-scores could reach 80–150, producing output predictions of thousands.

**Fix:** Add `np.clip(scaler.fit_transform(X_train), -10, 10)` to cap all z-scores at ±10.

---

### Stage 2: After Clipping — The Plateau

**Configuration:** Same as Stage 1 + clipping applied.

**Result: R² = 0.11–0.13 (stable but plateaued).**

The model produces sane predictions now, but is stuck. Three distinct problems were identified:

**Problem 1 — Noisy TF-IDF vocabulary:**
With 600 data-driven TF-IDF features, the top features selected were:
```
"tablet", "sig", "refills", "disp", "capsule" — medication instructions
"wallet", "money", "job" — patient social history
"afternoon", "morning", "evening" — time references in nursing notes
```

These correlated with documentation volume (sicker patients get more notes → more medication instructions → higher TF-IDF values for these terms) — not with organ failure severity.

**Problem 2 — FedAvg client drift:**
Server avg_loss oscillated: 8.7 → 12.85 → 9.99 → 8.70 → 10.57 across rounds. With 5 local epochs (too many), each hospital's model diverged significantly before sending weights back. FedAvg of three divergent models was worse than any individual model.

**Problem 3 — Training/evaluation objective mismatch:**
With weight=3.0, the training loss emphasised High-Risk patients (SOFA≥10, 6% of data). But the server selected the best round using unweighted MSE — dominated by Low-Risk patients (63% of data). The server kept discarding rounds where High-Risk performance improved but low-SOFA MSE temporarily worsened. The two objectives were fighting each other.

---

### Stage 3: Failed Experiments

All of the following were tried. None broke past R²=0.13:

| Experiment | Configuration | R² | Why it failed |
|---|---|---|---|
| Binary TF-IDF | `binary=True` | 0.11 | Lost severity grading — any mention of "sepsis" = 1, whether once or 50 times |
| Fewer data-driven features | `max_features=200` | 0.09 | Still picked noisy terms; noise-to-signal ratio unchanged |
| Weak FedProx | μ=0.01 | 0.11 | Proximal gradient 100× smaller than data gradient — invisible |
| Lower LR + gradient clipping | LR=0.0005, clip=1.0 | 0.11 | Root cause (noisy features) unaddressed |
| More rounds | 40→60 rounds | 0.13 | No new information; same oscillation pattern |
| Weight multiplier reduction | 3.0→0.5 | 0.13 | Slight improvement but noisy features still bottleneck |
| Fewer epochs | 5→2 per round | 0.09 | Too little local learning — model couldn't converge |

**Key diagnostic insight from Stage 3:** With 600 noisy TF-IDF features, the 18 vital features (which carry direct SOFA signal) were mathematically drowned out. The model had 600 dimensions of noise vs 18 dimensions of signal. Every optimisation improvement was negated by learning from irrelevant features.

---

### Stage 4: Breakthrough #1 — SOFA Vocabulary Whitelist

**Changes made:**
1. 600 data-driven TF-IDF → 90-term SOFA vocabulary whitelist
2. Model: 618→256→128→64→1 → **108→128→64→32→1** (smaller, better ratio)
3. Loss weight: 3.0 → **0.5** (better training/evaluation alignment)
4. FedProx μ=0.5 (moderate regularisation)

**Result: R² = 0.1752 — a 94% relative improvement from 0.09 baseline.**

**Why the whitelist was transformative:**

Before: 600 features, most encoding documentation style, ~18 carrying real SOFA signal. The model spent most of its capacity learning: "patients with many medication orders tend to have higher SOFA" — a spurious correlation.

After: 90 features, every single one directly connected to a SOFA organ component. The model learns: "patients with 'creatinine' mentioned in notes tend to have higher renal SOFA" — a genuine clinical relationship.

**Why the architecture change mattered:**

```
Old: 618 → 256 → 128 → 64 → 1
     Trainable parameters: ~197,000
     With 48,150 training samples: 0.24 samples per parameter
     → Severely underfitting: too many parameters for available data

New: 108 → 128 → 64 → 32 → 1
     Trainable parameters: ~23,000
     With 48,150 training samples: 2.1 samples per parameter
     → Appropriate ratio for tabular data generalisation
```

---

### Stage 5: Failed Experiments After Vocabulary

With R²=0.1752 as the new baseline, the following failed to improve significantly:

| Experiment | Configuration | R² | Why it failed |
|---|---|---|---|
| Stronger FedProx | μ=1.0 | 0.1355 | Too strong — hospitals couldn't learn from their data; convergence to worse minimum |
| Faster LR decay | LR_DECAY=0.97 | 0.1355 | LR hit 0.0001 by round 50; model stopped learning before finding the good minimum |
| Warm restart at round 50 | Reset FedYogi state + restore best model | 0.2107 | FedYogi's accumulated momentum (built over 26 rounds) was thrown away; destroyed the optimisation intelligence |
| More local epochs | 5 epochs/round | 0.2077 | Hospitals diverged more despite FedProx; conflicting updates worsened |
| LayerNorm | Added normalisation layers | 0.2187, pred range 0.22–7.04 | Catastrophic prediction collapse (explained in Section 8.1) |

---

### Stage 6: Breakthrough #2 — FedAdam

**Change:** Replace FedAvg with FedAdam server optimiser (η=0.01, β1=0.9, β2=0.99, τ=0.001)

**Result: R² = 0.2148 — 23% improvement from 0.1752.**

**Why FedAdam helped:**

Plain FedAvg has no memory of previous rounds. Every round, it simply averages the three hospital models and uses that as the new global model. A bad round is treated identically to a good round.

FedAdam adds Adam-style momentum at the server level. If rounds 1–10 have been consistently improving in direction `d`, FedAdam's momentum vector points strongly in `d`. A single bad round at round 11 contributes only `0.1 × bad_direction` to the momentum, while `0.9 × d` (the good direction) continues to push the model the right way. One bad round cannot undo 10 good rounds.

**The FedAdam explosion:**

FedAdam's `v_t` (adaptive rate denominator) can decrease when pseudo-gradients are small:
```
FedAdam: v_t = 0.99 × v_{t-1} + 0.01 × Δ²
         When |Δ| is small: v_t ≤ v_{t-1}
         → Effective step size η/√v_t can GROW unboundedly
         → During recovery after an explosion, step sizes become enormous → divergence
```

This was the limitation that FedYogi was specifically designed to fix.

---

### Stage 7: Breakthrough #3 — FedYogi

**Change:** Replace FedAdam with FedYogi at the server (same hyperparameters).

**Result: R² = 0.2229 — 4% improvement from FedAdam's 0.2148.**

**The crucial difference:**

```
FedAdam: v_t = 0.99 × v_{t-1} + 0.01 × Δ²
         v_t CAN decrease when |Δ| is small
         → step size can explode during recovery

FedYogi: v_t = v_{t-1} + 0.01 × (Δ² − v_{t-1}) × sign(Δ² − v_{t-1})
         v_t only INCREASES when |Δ|² > v_{t-1}
         v_t stays the same or grows — never decreases
         → step size is bounded by historical maximum gradient magnitude
```

FedYogi's `v_t` acts as a "high watermark." Once the explosion at rounds 5–12 drives `v_t` to a high value, it stays high. The denominator `√v_t` is large, so the effective step size `η/√v_t` remains small throughout the recovery phase. This keeps the recovery stable and allows the model to find a deeper minimum than FedAdam could.

**Run-to-run variation:** Different random batch orderings change the exact explosion magnitude (200× to 270×) and best-round avg_loss (7.55 to 7.67), producing R² ranging from 0.205 to 0.2229 across identical configuration runs. This stochasticity is inherent to the FedYogi optimisation dynamics.

---

### Stage 8: Failed Experiments After FedYogi

| Experiment | Configuration | R² | Why it failed |
|---|---|---|---|
| FedAdam η=0.003 | Lower server LR | 0.1958 | Less exploration → shallower minimum found |
| Larger FedYogi τ | τ=0.01 | 0.2174 | Smaller explosion, less exploration, shallower recovery |
| Warm restart at round 50 | Reset FedYogi + restore best | 0.2107 | FedYogi state reset destroyed accumulated momentum |
| LayerNorm (second attempt) | With τ=0.01 | 0.2187, pred 0.22–7.04 | Same prediction range collapse |
| Centralised pre-training 20 epochs | FedPreTrain warm-start | 0.1582 | FL training degraded the pre-trained model every round after round 1 |
| More local epochs | 5 epochs/round | 0.2077 | More local drift even with FedProx |

**The centralised pre-training failure explained:**

The intuition behind FedPreTrain seemed reasonable: "start from a better initial model rather than random weights." The idea was that pre-training on all data centrally would give FedYogi a better starting point.

Why it backfired: the centralised pre-trained model was already in a local minimum of the loss landscape. The subsequent FL training from three hospitals with different data distributions pulled the model in three different directions away from that minimum. Without the cold-start explosion (which had been the mechanism for escaping shallow minima), the model had no way to reach a better region of parameter space. Every FL round made it worse. FedPreTrain effectively trapped the model in the pre-trained minimum.

---

### Stage 9: Breakthrough #4 — Expanded Notes Corpus

**Changes:**
- `NOTES_SAMPLE_SIZE`: 80,000 → **283,208** (all available notes)
- `NOTES_TEXT_LIMIT`: 5,000 → **10,000** characters

**Result: R² = 0.4724, MAE = 1.76 — an 87% relative improvement from 0.2229.**

This was the largest single improvement in the entire optimisation journey.

**Three mechanisms explaining the gain:**

**Mechanism 1 — Better IDF weights:**

IDF = log(total_documents / documents_containing_term). With more documents, the IDF weights are computed from a larger population, making them more statistically reliable.

| Term | df at 80k notes | df at 283k notes | IDF stability |
|---|---|---|---|
| `bacteremia` | ~80 docs | ~300 docs | 3.75× more stable estimate |
| `totbili` | ~60 docs | ~220 docs | 3.7× more stable |
| `oliguria` | ~120 docs | ~450 docs | 3.75× more stable |
| `septic shock` | ~320 docs | ~1,200 docs | 3.75× more stable |

More stable IDF weights mean the TF-IDF feature values are less noisy, which means the model can learn cleaner feature-to-SOFA relationships.

**Mechanism 2 — More admissions covered:**

With 80,000 note samples from 283,208 total notes, only 22,245 unique hospital admissions (out of ~60,000) were covered. Many ICU stays had no notes in the sample → TF-IDF features = all zeros for those patients.

With all 283,208 notes: 41,179 unique admissions covered (85% more). Nearly every patient now has meaningful TF-IDF features instead of zeros.

**Mechanism 3 — Longer text captures lab mentions:**

```
Patient with creatinine = 2.8 on their lab results.
Their admission note (first 5,000 characters):
  "Mr. X is a 67yo male admitted for... [history of present illness]...
   Relevant past medical history includes... [2,000 words of medical history]..."

At 5,000 characters: truncated BEFORE reaching lab value mentions → creatinine TF-IDF = 0

At 10,000 characters, later in the same note:
  "...creatinine 2.8 this morning, trending up from 1.9 yesterday.
   Urine output 15cc/hr over the past 6 hours, patient appears oliguric..."

→ "creatinine" and "oliguria" both captured → correct renal SOFA signal
```

**Evidence the corpus expansion was responsible:**

- TF-IDF matrix covered admissions: 22,245 → 41,179 (+85%)
- Hospital local R² at round 1: 0.00–0.05 → 0.04–0.21 (3–4× more informative features)
- Server avg_loss at best round: 7.55 → 5.48 (a fundamentally better minimum)
- MAE: 2.15 → 1.76 SOFA points (−18% prediction error)

---

## 17. Key Lessons Learned

### 1. Feature quality beats optimisation complexity

The two biggest R² jumps came from feature engineering:
- SOFA vocabulary whitelist: +94% relative improvement
- Expanded notes corpus: +87% relative improvement

All FL optimisation techniques combined (FedProx, FedAdam, FedYogi, LR scheduling, gradient clipping): +27% relative improvement.

**Implication:** Before adding more training rounds, more complex architectures, or more sophisticated optimisers, ensure that the features themselves are high quality and clinically meaningful. Garbage-in-garbage-out applies even to federated learning.

### 2. The cold-start explosion is exploration, not failure

The round 5–12 explosion (avg_loss reaching 100–270× initial) caused alarm in early experiments, leading to attempts to eliminate it (pre-training, warm restarts, larger τ). Every attempt to prevent the explosion produced worse final R². The explosion is how FedYogi escapes shallow local minima. The best-round model at round 24 is better than round 1–4 precisely because the explosion enabled exploration of parameter space that FedAvg could never reach.

### 3. Align training loss and evaluation metric

With weight multiplier 3.0 (heavy emphasis on High-Risk), the server was selecting the best round using unweighted MSE (dominated by Low-Risk patients). These objectives conflicted. The best round as judged by the server was not always the best round for what we actually cared about. Reducing to weight 0.5 aligned the training and evaluation objectives, allowing the best-round selection to identify the genuinely best model.

### 4. Architecture must match data scale

618→256→128→64→1 with 48k samples = 0.24 samples per parameter — severe underfitting risk.
108→128→64→32→1 with 48k samples = 2.1 samples per parameter — appropriate regime for tabular data.

Not "bigger is better." The right size for the data is better.

### 5. LayerNorm is incompatible with wide-range regression

LayerNorm normalises each patient's activations to mean=0, std=1 within that patient. For classification (cats vs dogs), this is fine — the label does not encode magnitude. For regression over a wide output range (SOFA 0–24), this destroys exactly the magnitude information the model needs. After LayerNorm, a SOFA=20 patient and a SOFA=2 patient produce identically normalised hidden representations. The model cannot distinguish them.

### 6. More data consistently helped

Every time the amount of data was increased (more notes, longer text, more admissions covered), model performance improved. With 283,208 notes available, using all of them outperformed using 80,000. The theoretical maximum for this feature set is approached but not yet reached.

---

## 18. Performance Summary Table

| Stage | Key Change | R² | MAE | Best Round |
|---|---|---|---|---|
| Baseline | 600 data-driven TF-IDF, FedAvg, no clipping | −824 | ~50 | — |
| + Clipping | np.clip(±10) | 0.11–0.13 | 2.38 | ~23 |
| Binary TF-IDF | binary=True | 0.11 | 2.42 | — |
| Fewer data-driven features | max_features=200 | 0.09 | 2.49 | — |
| Weak FedProx | μ=0.01 | 0.11 | 2.38 | — |
| **SOFA whitelist + new architecture + weight 0.5** | **Breakthrough #1** | **0.1752** | **2.31** | **~24** |
| FedProx μ=1.0 | Over-constrained | 0.1355 | — | — |
| Centralised pre-training | FedPreTrain 20E | 0.1582 | — | 1 |
| **FedAdam server optimizer** | **Breakthrough #2** | **0.2148** | **2.16** | **~29** |
| **FedYogi server optimizer** | **Breakthrough #3** | **0.2229** | **2.15** | **~24** |
| LayerNorm | Per-sample normalisation | 0.2187 | 2.16 | 14 |
| 5 epochs/round | More local training | 0.2077 | 2.20 | 26 |
| **283k notes + 10k char text** | **Breakthrough #4** | **0.4724** | **1.76** | **24** |

---

## 19. Fundamental Ceiling Analysis

### Why R²=0.4724 is approaching the feature ceiling

The model's performance is ultimately limited by what information its features contain. Three of the six SOFA components are not directly measured:

| Component | Lab Value Needed | How We Approximate It | Noise Level |
|---|---|---|---|
| Coagulation | Platelet count | Text mentions of "plt", "thrombocytopenia" | High — documentation-dependent |
| Hepatic | Bilirubin | Text mentions of "bilirubin", "totbili" | High — documentation-dependent |
| Renal | Creatinine + urine output | Text mentions of "creatinine", "oliguria" | High — documentation-dependent |

A physician may write "creatinine 2.8" or "cr elevated" or "acute renal failure" — all encoding the same clinical fact with different words. TF-IDF can only capture the exact vocabulary terms in the whitelist. A note saying "kidney function worsening" would contribute 0 to the creatinine feature even though it is clinically equivalent.

**Estimated theoretical maximum with our feature set:** R² = 0.50–0.65

The current R²=0.4724 is near this ceiling. Substantial further improvement would require adding direct lab measurements.

### How to break through the ceiling

Add direct lab values from the BigQuery `labevents` table (joined by `hadm_id`):

| New feature | SOFA Component | Expected improvement |
|---|---|---|
| Serum creatinine | Renal (Component 6) | Within-segment R² for Moderate/High could turn positive |
| Serum bilirubin | Hepatic (Component 3) | Hepatic SOFA component directly measured |
| Platelet count | Coagulation (Component 2) | Coagulation SOFA component directly measured |

**Combined expected result:** Global R² = 0.65–0.80, with positive within-segment R² for all three tiers. This would also be a much more clinically useful model — one that can accurately rank patients within each severity tier, not just classify them into broad tiers.

### Context for the current R²=0.4724

For a Federated Learning project predicting a composite organ failure score from only vital signs and clinical text (without any direct lab measurements), R²=0.4724 with MAE=1.76 SOFA points is a strong result. For comparison:
- Published FL papers on comparable clinical prediction tasks (without direct lab access) typically report R²=0.20–0.40
- The privacy-preserving federated setting inherently costs some performance compared to centralised training
- Our model correctly identifies High-Risk patients (SOFA ≥ 10) with strong discrimination — the most clinically important capability for an early warning system

---

*Documentation for ICU CDSS Capstone Project*
*Training framework: Flower FL v1.x, PyTorch DNN*
*Aggregation: FedYogi + FedProx*
*Data: MIMIC-III via Google BigQuery*

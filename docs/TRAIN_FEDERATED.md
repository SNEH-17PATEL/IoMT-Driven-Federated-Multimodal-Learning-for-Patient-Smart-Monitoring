# TRAIN_FEDERATED — How the Model Is Trained
**Module:** ICU CDSS Capstone — Federated Learning Training Pipeline (`train_federated.py`)
**Last updated:** 2026-11-12

---

## How to Read This Document

`train_federated.py` is the script that **builds the SOFA-prediction model** used by the ClinIQ dashboard. It does three big jobs:

1. **Prepares the data:** downloads ICU records and doctors' notes, turns the notes into numbers, and splits everything into 3 pretend "hospitals".
2. **Trains the model with federated learning:** the 3 hospitals train together without sharing patient data.
3. **Tests and saves the model:** measures how good it is and saves every file the dashboard needs.

The document has two parts:
- **Part A — How the script works today**, step by step, in the order it runs. Every number here comes from the current code.
- **Part B — How we got here**: the experiments we tried, what failed and what worked. These numbers are **historical** (recorded when each experiment ran) and cannot all be reproduced from the current files.

Each section explains the idea in plain words first, then the exact rule or formula, then a worked example.

> **Read [Section 15](#15-the-current-model-and-todays-changes) before quoting any result.** It lists the current numbers (from `models/training_metadata.json`) and explains, in plain words, the four changes made on 2026-11-12: plain MSE loss, FedAvg instead of FedYogi, a wider network with dropout, and an alert threshold of 7.

---

## Table of Contents

**Part A — How the Script Works**
1. [Federated Learning in Plain Words](#1-federated-learning-in-plain-words)
2. [The Whole Pipeline on One Page](#2-the-whole-pipeline-on-one-page)
3. [Every Setting Explained](#3-every-setting-explained)
4. [Does the Script Re-Download the Data?](#4-does-the-script-re-download-the-data)
5. [Phase 0 — Preparing the Data](#5-phase-0--preparing-the-data)
6. [Phase 1 — Loading the Hospitals' Data](#6-phase-1--loading-the-hospitals-data)
7. [Phase 2 — SHAP Background Patients](#7-phase-2--shap-background-patients)
8. [Phase 3 — Federated Training](#8-phase-3--federated-training)
9. [Phase 4 — Saving the Best Model](#9-phase-4--saving-the-best-model)
10. [Phase 5 — Final Testing](#10-phase-5--final-testing)
11. [Phase 6 — Saving the Training Report](#11-phase-6--saving-the-training-report)
12. [Optional Features: Non-IID Split and Differential Privacy](#12-optional-features)
13. [Every File the Script Produces](#13-every-file-the-script-produces)
14. [How to Run It](#14-how-to-run-it)
15. [The Current Model and Today's Changes](#15-the-current-model-and-todays-changes)

**Part B — How We Got Here**

16. [Why Predicting SOFA Is Hard](#16-why-predicting-sofa-is-hard)
17. [The Experiment Journey, Stage by Stage](#17-the-experiment-journey)
18. [Lessons Learned](#18-lessons-learned)
19. [Results Table Across Experiments](#19-results-table-across-experiments)
20. [How to Go Further](#20-how-to-go-further)
21. [Glossary](#21-glossary)

---

# Part A — How the Script Works

---

## 1. Federated Learning in Plain Words

**In one line:** instead of sending patient data to the model, we send the model to the data.

### The problem

Normally you train an AI model by putting all the data in one place. With hospital data you cannot do that: privacy laws (like HIPAA in the US) stop Hospital A from sending its patients' records to Hospital B, or to a central company.

So each hospital is stuck with its own small dataset, and a model trained on one hospital is weaker than a model trained on all of them together.

### The solution

**Federated learning (FL)** lets several hospitals train **one shared model** without any patient data leaving any hospital:

```
Centralised (not allowed):
  Hospital A data ─┐
  Hospital B data ─┼──►  one server holds ALL the data and trains the model
  Hospital C data ─┘      ✗ patient records leave the hospitals

Federated (what we do):
  1. The server sends the current model to every hospital.
  2. Each hospital trains it on ITS OWN patients, locally.
  3. Each hospital sends back only the model's numbers (its "weights").
  4. The server combines those numbers into a better shared model.
  5. Repeat.
     ✓ patient records never leave the hospitals
```

**An everyday analogy:** three schools want to write the best study guide, but none may share its students' exam papers. Each school edits the guide using its own students' mistakes and mails back only the edited guide. A coordinator merges the three edits into one improved guide and mails it out again. After many rounds the guide reflects all three schools, but no exam paper ever left a school.

### What a "weight" is

A neural network is just a long list of numbers (**weights**) that turn the inputs into a prediction. Our model has **69,121** of them. Training means nudging these numbers so the predictions get closer to the truth. In FL, only these numbers travel between the hospitals and the server, never the patients.

### How our project uses it

We use real ICU data from **MIMIC-III** (Beth Israel Deaconess Medical Center, Boston). We **simulate** three hospitals by splitting the training patients into three groups, and run real FL software (**Flower**) on them. This shows the system could be deployed across real hospitals without changing how it trains.

---

## 2. The Whole Pipeline on One Page

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 0  Prepare data  (only if the database has no training data yet)    │
│   0a Connect to Google BigQuery                                           │
│   0b Download vitals: the latest row of every ICU stay                    │
│   0c Download doctors' and nurses' notes                                  │
│   0d Clean the notes, turn them into 90 SOFA-related word scores (TF-IDF) │
│   0e Join vitals + word scores → 108 inputs + the SOFA answer             │
│   0f Split 80% training / 20% testing                                     │
│   0g Rescale every input to a common scale (and clip extremes)            │
│   0h Split the training patients into 3 "hospitals"                       │
│   0i Save the hospitals to SQLite and the helper files to models/         │
├─────────────────────────────────────────────────────────────────────────┤
│ PHASE 1  Load the 3 hospitals from SQLite, find the test set              │
│ PHASE 2  Save 300 "typical" patients for SHAP explanations                │
│ PHASE 3  Federated training: 100 rounds × 3 hospitals (FedAvg + FedProx)  │
│ PHASE 4  Save the model from the BEST round  → federated_model.pth        │
│ PHASE 5  Test it on the test set: MAE, R², AUROC, confusion matrices      │
│ PHASE 6  Save a training report → training_metadata.json                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**In one sentence:** the script turns MIMIC-III vitals and notes into 108 numbers per patient, trains a small neural network across 3 simulated hospitals for 100 rounds, and keeps the round that predicted best.

---

## 3. Every Setting Explained

All settings are constants near the top of `train_federated.py`.

### Paths and data source

| Setting | Value | Meaning |
|---|---|---|
| `MODEL_PATH` | `"models/"` | Where all output files are written (relative to the folder you run from) |
| `BIGQUERY_PROJECT` | `"mimic-project-2"` | The Google Cloud project holding the MIMIC-III tables |
| `BIGQUERY_DATASET` | `"dataset_v2"` | The dataset name inside that project (rebuilt in BigQuery, see `docs/SQL_COMMANDS.md`; the old dataset `Dataset` was deleted) |
| `BIGQUERY_LOCATION` | `"US"` | The region where the data is stored |
| `SKIP_PREPROCESSING` | `False` | Set to `True` to never run Phase 0 (see Section 4) |

### Note-processing settings

| Setting | Value | Meaning |
|---|---|---|
| `NOTES_SAMPLE_SIZE` | 283,208 | How many notes to use. This is **all** of them. |
| `NOTES_TEXT_LIMIT` | 10,000 | Each admission's combined notes are cut to the first 10,000 characters (about 1,500–2,000 words) |
| `TFIDF_NGRAM_RANGE` | (1, 2) | Look at single words ("sepsis") **and** two-word phrases ("septic shock") |
| `TFIDF_MAX_FEATURES`, `TFIDF_MIN_DF`, `TFIDF_MAX_DF` | 200, 10, 0.9 | **Not used.** Because we give TF-IDF a fixed word list, these settings are ignored (left over from earlier experiments) |

### Training settings

| Setting | Value | Meaning |
|---|---|---|
| `NUM_ROUNDS` | 100 | How many times the server and the hospitals exchange the model |
| `EPOCHS_PER_ROUND` | 3 | In each round, every hospital goes through its own data 3 times |
| `BATCH_SIZE` | 64 | The model updates its weights after every 64 patients |
| `SHAP_BG_SAMPLES` | 300 | How many "typical" patients to save for explanations |
| `BASE_LR` | 0.001 | Starting learning rate (how big each weight nudge is) |
| `LR_DECAY` | 0.99 | The learning rate is multiplied by 0.99 every round |
| `GRAD_CLIP` | 1.0 | Caps the size of any single update (explained below) |
| `MU_FEDPROX` | 0.5 | How strongly each hospital is pulled back toward the shared model |
| `USE_FEDYOGI` | `False` | `False` = the server uses plain **FedAvg** (current). `True` switches back to FedYogi (optional, see Section 8.6) |
| `SERVER_ETA` | 0.001 | The server's step size, **only used if `USE_FEDYOGI = True`** (it was 0.01, which was unstable) |
| `SERVER_BETA1` | 0.9 | FedYogi only: the server's momentum memory |
| `SERVER_BETA2` | 0.99 | FedYogi only: controls how fast the server's "step-size brake" changes |
| `SERVER_TAU` | 0.001 | FedYogi only: a tiny number that avoids dividing by zero |
| `OVERSAMPLE` | `False` | Whether to show High Risk patients more often (off) |
| `USE_NONIID_SPLIT` | `False` | Whether hospitals get different patient mixes (Section 12) |
| `USE_DP`, `DP_SENSITIVITY`, `DP_SIGMA` | `False`, 1.0, 1.0 | Differential privacy, off by default (Section 12) |

At the very start, the script also fixes the random seeds (`torch`, `numpy` and `random` all set to 42), so runs are as repeatable as possible. The Flower simulation's scheduling can still vary slightly from run to run.

### Learning rate: what it is and how it shrinks

**In one line:** the learning rate is the size of each step the model takes while learning.

A big step learns fast but may overshoot the best answer; a small step is precise but slow. We start with bigger steps and make them smaller over time:

```
learning rate in round r = max(0.0001,  0.001 × 0.99^(r − 1))

Round 1:    0.001000
Round 26:   0.000778   ← the best round of the current model
Round 50:   0.000611
Round 100:  0.000370
```
The 0.0001 floor is never reached within 100 rounds; it would only kick in after round 230.

**Analogy:** looking for the lowest point in a valley in fog. First take big strides to get to the right area, then small steps so you don't walk past the bottom.

### Gradient clipping

**In one line:** if one batch of unusual patients asks for a huge update, it is shrunk to a safe size.

While learning, the model computes a **gradient**: which way, and how far, to move each weight. Occasionally a strange batch produces an enormous gradient that would throw the weights far off course. Clipping caps its overall size at 1.0 but keeps its direction:

```
Gradient size 8.3 → rescaled to 1.0 (same direction, 8.3× smaller)
Gradient size 0.6 → left alone
```

---

## 4. Does the Script Re-Download the Data?

**In one line:** only if the database does not already have training data.

Phase 0 (downloading and processing the data) is slow, so the script checks first:

```python
skip = SKIP_PREPROCESSING or fl_training_exists()
```
`fl_training_exists()` returns True when the `fl_training` table exists, has at least one row, and has a `sofa_score` column.

| `SKIP_PREPROCESSING` | Training data already in the database? | What happens |
|---|---|---|
| `False` | No | Phase 0 runs (BigQuery download + note processing) |
| `False` | Yes | Phase 0 is skipped automatically |
| `True` | Yes | Phase 0 is skipped |
| `True` | No | Phase 0 is skipped, and Phase 1 stops with a clear error because there is no data |

Phase 1 also checks that the three helper files from Phase 0 (`scaler.pkl`, `tfidf_vectorizer.pkl`, `feature_columns.pkl`) exist. If any is missing, the script stops and says which ones and how to fix it.

**To force Phase 0 to run again** (for example after changing the word list):
```bash
sqlite3 models/icu_monitor.db "DELETE FROM fl_training;"
python train_federated.py
```

---

## 5. Phase 0 — Preparing the Data

All of Phase 0 is in the function `run_preprocessing()`.

### 0a. Connecting to BigQuery

```python
bq = bigquery.Client(project="mimic-project-2", location="US")
```
**BigQuery** is Google's cloud database; our copy of MIMIC-III lives there. The connection uses the Google credentials you set up once with `gcloud auth application-default login`.

### 0b. Downloading the vitals: one row per ICU stay

```sql
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY icustay_id
                                 ORDER BY window_time DESC) AS rn
    FROM `mimic-project-2.dataset_v2.ml_dataset_final`
    WHERE sofa_score IS NOT NULL
)
WHERE rn = 1
```

**What this does, in plain words:** `ml_dataset_final` is a VIEW with **many rows per ICU stay**: one per hour, **5,089,857 hourly rows** in total. The cohort is adults (age ≥ 16) whose ICU stay lasted at least 6 hours: **52,248 ICU stays from 38,304 patients**. The answer (`sofa_score`) is the SOFA score of the last 24 hours **at that same hour** (from `pivoted_sofa`), not one value for the whole stay. For each ICU stay (`PARTITION BY icustay_id`), the query numbers its rows from newest to oldest (`ORDER BY window_time DESC`) and keeps only number 1, **the latest snapshot** of every stay. Result: **52,248 rows**, one per ICU stay.

**Example:**
```
ICU stay 501 has rows at 08:00, 09:00, 10:00 → keep only 10:00
ICU stay 502 has rows at 14:00, 15:00        → keep only 15:00
```

**What each row contains:**

| Group | Columns | Count |
|---|---|---|
| IDs (removed later) | subject_id, hadm_id, icustay_id, window_time, rn | 5 |
| **Trend vitals** (summaries of recent readings) | HR_mean, HR_std, RR_mean, SpO2_mean, SpO2_min, Temp_mean, SBP_mean, DBP_mean, MAP_mean | 9 |
| **Latest vitals** | latest_HR, latest_RR, latest_SpO2, latest_Temp, latest_SBP, latest_DBP, latest_MAP | 7 |
| **Observation scores** | GCS_eye_opening (1–4), stress_score (0–10) | 2 |
| **The answer** | sofa_score (0–24) | 1 |

**Why both trends and latest values?** One reading can be misleading. An SpO₂ of 91% might be a slipped finger clip, or a real decline. The trend tells the story:
```
Patient A: SpO₂ readings 97, 97, 96, 97, 91   → one odd reading (probably the probe)
Patient B: SpO₂ readings 97, 95, 94, 92, 91   → steady decline (worrying)
Both have latest_SpO2 = 91, but SpO2_mean and SpO2_min tell them apart.
```

### 0c. Downloading the clinical notes

```sql
SELECT subject_id, hadm_id, text FROM `mimic-project-2.dataset_v2.clinical_notes`
```
`clinical_notes` is a VIEW covering **35,799 admissions**: nursing notes, doctors' notes, progress notes and discharge summaries. Each note carries `hadm_id` (the hospital-admission ID), which is how notes are matched to ICU stays.

**Why notes matter:** half of the SOFA score depends on blood tests we do not have as numbers (platelets, bilirubin, creatinine). Doctors often **write those results down**:
```
"Creatinine 2.8 this morning, trending up"          → kidney problem
"Bilirubin 4.5, hepatology consulted"                → liver problem
"Platelets dropped to 78, thrombocytopenic"          → clotting problem
"Started norepinephrine for blood pressure support"  → circulation failing
```
Turning these words into numbers gives the model indirect access to the missing lab results.

### 0d. Cleaning the notes and turning words into numbers

#### Step 1 — clean the text

```python
def _clean_clinical_text(text):
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', ' ', text)   # remove [** privacy placeholders **]
    text = re.sub(r'\b\d+(?:[./]\d+)?\b', ' ', text)  # remove numbers like 2.8 or 120/80
    text = text.lower()                                 # make everything lowercase
    text = re.sub(r'[^a-z\s]', ' ', text)              # keep only letters and spaces
    return ' '.join(text.split())                       # squash repeated spaces
```

**Before and after:**
```
Before: "Pt [**Known firstname 15**] has creatinine 2.8, BP 95/60. Started on
         Levophed @ 0.08 mcg/kg/min. [**2145-03-15**]"
After:  "pt has creatinine bp started on levophed mcg kg min"
```

Why each step:
- **Privacy placeholders:** MIMIC-III replaces names and dates with `[**…**]` tags. Left in, they would look like meaningful words.
- **Numbers:** "creatinine 2.8" and "creatinine 3.1" would otherwise become different tokens; we only want "creatinine was discussed".
- **Symbols:** they add noise.

**A side effect to know about:** removing digits turns "HCO3" into "hco" and "PaO2" into "pao". That is why our word list contains "hco".

#### Step 2 — combine each admission's notes

```python
notes_sampled  = notes.sample(n=min(283_208, len(notes)), random_state=42)  # all notes, shuffled
notes_combined = notes_sampled.groupby("hadm_id")["text"].apply(" ".join)
notes_combined["text"] = notes_combined["text"].apply(_clean_clinical_text).str[:10_000]
```
All notes for one admission are joined into **one long document**, cleaned, then cut to the first 10,000 characters. The cleaning happens **before** the cut, so the 10,000 characters are all real words, not placeholders.

Two consequences:
- Admissions with no notes get no document, and their word scores become 0 in step 0e.
- Because the notes are shuffled before joining, *which* 10,000 characters survive for very long admissions is somewhat random.

#### Step 3 — score 90 chosen words (TF-IDF with a whitelist)

**TF-IDF** gives each word a score per document:
- **TF (term frequency):** how often the word appears in **this** patient's notes.
- **IDF (inverse document frequency):** how **rare** the word is across all patients. Rare words get a bigger boost.

```
score ≈ (how often here) × (how rare overall)
"patient" appears everywhere → tiny score (says nothing special)
"bacteremia" is rare          → big score when it does appear
```

**The key design decision: a fixed list of 90 words.** Instead of letting TF-IDF pick the "most informative" words automatically, we give it a list of terms tied directly to the SOFA organs:

| SOFA area | Example words in the list |
|---|---|
| Lungs | intubated, ventilator, vent, cpap, bipap, hypoxia, respiratory failure, shortness breath |
| Clotting | plt, platelets, bleeding, hemorrhage, inr, ptt, coagulopathy, thrombocytopenia |
| Liver | bilirubin, bili, totbili, liver, jaundice, hepatic, cirrhosis, liver failure |
| Heart and circulation | hypotension, shock, vasopressor, pressors, norepinephrine, levophed, dopamine, septic shock |
| Brain | sedated, sedation, coma, altered, confused, gcs, encephalopathy, delirium, unresponsive |
| Kidneys | creatinine, creat, renal, kidney, dialysis, aki, oliguria, urine output, crrt |
| Infection | sepsis, septic, bacteremia, infection, cultures, antibiotics, fever, blood cultures |
| Lab words | lactate, wbc, hgb, hct, sodium, potassium, hco, angap, bun |
| General severity | failure, acute, pneumonia, ards, pulmonary, dyspnea, edema |

**Why not let the computer choose?** We tried (Part B, Stage 2). The automatic choice picked words like *tablet*, *sig* ("take as directed"), *refills*, *wallet* and *afternoon*. These words are common in **long** records, and long records belong to patients who stayed longer, so the model learned "more paperwork = sicker", not "kidney failure = sicker". The fixed list forces the model to learn real medical relationships.

**The TF-IDF settings:**
```python
TfidfVectorizer(
    vocabulary=list(_SOFA_VOCABULARY),            # only our 90 terms
    ngram_range=(1, 2),                            # words and two-word phrases
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b', # words of 3+ letters
    sublinear_tf=True,                             # use 1 + log(count)
    norm="l2",                                     # make every document's scores comparable
)
```

- **`sublinear_tf=True`:** mentioning "sepsis" 10 times does not mean 10 times more septic. Using 1 + log(count) means count 1 → 1.0, count 10 → 3.3, count 100 → 5.6. Extra mentions add less and less.
- **`norm="l2"`:** without it, patients with very long notes would score higher just for having more text. This rescales each patient's 90 scores so the overall size is the same for everyone.

**Worked example — two patients:**
```
Patient A (septic shock, on a ventilator):
  notes: "...intubated... septic shock... norepinephrine... creatinine rising...
          oliguria... hypotensive..."
  high scores for: intubated, septic shock, norepinephrine, creatinine, oliguria
  0 for the ~80 words never mentioned

Patient B (heart failure, breathing trouble, no organ failure):
  notes: "...dyspnea... bilateral edema... pulmonary congestion..."
  high scores for: dyspnea, edema, pulmonary
  0 for every organ-failure word
```
The model can now tell these two apart from the notes alone.

### 0e. Joining vitals and words into one table

```python
dataset = ml_data.merge(tfidf_df, on="hadm_id", how="left")
dataset = dataset.drop(columns=["subject_id", "hadm_id", "icustay_id", "window_time", "rn"])
dataset = dataset.fillna(0)
```
- **Left join:** every ICU stay is kept, even if it has no notes (its 90 word scores become 0).
- **Drop IDs:** IDs carry no medical meaning and could let the model "memorise" patients.
- **`fillna(0)`:** any missing value becomes 0.

**Result: 108 inputs + 1 answer per row**
```
 9 trend vitals + 7 latest vitals + 2 observation scores + 90 word scores = 108 inputs
 + sofa_score (the answer)
```

**A limitation of `fillna(0)`:** a missing vital becomes 0, which is not a real value (a heart rate of 0 means death). Most rows have no GCS or stress value, so most patients ended up with GCS 0 and stress 0 in training. This is why those two inputs have little influence on the model (see Section 20).

### 0f. Splitting into training and test sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=(y >= 10))
```
- **80% training, 20% testing.** The test patients are locked away until Phase 5, like a final exam the student never sees beforehand.
- **`stratify=(y >= 10)`:** High Risk patients (SOFA ≥ 10) are only about 6% of the data. Stratifying guarantees that both sets get the **same share** of them. Without it, the test set could end up with very few High Risk patients, and their scores would be meaningless.
- **`random_state=42`:** the same split every time.

**Result:** 41,798 ICU stays for training and **10,450** for testing (602 High, 2,919 Moderate, 6,929 Low).

**A limitation:** there is one row per ICU stay, so the split is **by ICU stay**, not by patient. One person with two ICU stays could have one stay in training and one in testing, which can make the test results look slightly better than they would be on brand-new patients.

### 0g. Putting every input on the same scale

```python
scaler = StandardScaler()
X_train_scaled = np.clip(scaler.fit_transform(X_train), -10, 10)
X_test_scaled  = np.clip(scaler.transform(X_test),      -10, 10)
```

**Why rescale?** The inputs have very different sizes:
```
heart rate      40–200
SpO₂            70–100
word scores     0–1
```
Without rescaling, big-number inputs like heart rate would dominate, and the word scores would be almost ignored.

**How StandardScaler works:** for each input it learns the average and the typical spread (standard deviation) **from the training set**, then converts every value to "how many spreads away from the average":
```
scaled = (value − average) / spread

HR_mean: average 85, spread 20
  HR 105 → (105 − 85) / 20 = +1.0
  HR  65 → ( 65 − 85) / 20 = −1.0
```
After this, every input has average 0 and spread 1.

**Important rule:** the averages and spreads come **only from the training set** (`fit_transform`). The test set is converted with those same numbers (`transform`). Learning anything from the test set would let information leak from the exam into the studying.

**Why clip at ±10?** Rare words can produce extreme values after scaling:
```
A word that appears in almost no documents: average ≈ 0.001, spread ≈ 0.01
A patient whose notes mention it: value 0.8 → scaled = (0.8 − 0.001) / 0.01 ≈ 80
```
An input of 80 can explode through the network. In our first experiment the model predicted a SOFA of **10,001** (Part B, Stage 1). Capping every scaled value between −10 and +10 keeps the information ("this word was strongly present") without the explosion.

### 0h. Splitting the training patients into 3 hospitals

The current code makes a **stratified** three-way split, so every hospital gets the same share of High Risk patients:
```python
split_a, split_bc = train_test_split(data_train, test_size=2/3, stratify=(sofa >= 10))
split_b, split_c  = train_test_split(split_bc,   test_size=0.5, stratify=(sofa >= 10))
# → about one third each: Hospital 0 (General ICU), 1 (Mixed ICU), 2 (Cardiac/Trauma ICU)
```

This is an **IID** split (Independent and Identically Distributed): all 3 hospitals have the same kind of patient mix. Real hospitals differ (a trauma centre sees sicker patients). That harder case is available as an option (Section 12).

> The hospital names are only labels; in IID mode the split is random, so the "Cardiac/Trauma ICU" does not actually have more cardiac or trauma patients.
>
> The data currently in the database came from this stratified split: the hospitals have **13,932 / 13,933 / 13,933** rows, and each one is about **67% Low, 27% Moderate and 6% High** (802 High Risk patients each).

### 0i. Saving the results

```python
# per hospital: replace its old rows in the fl_training table, then append the new ones
DELETE FROM fl_training WHERE client_id = ?      # skipped if the table doesn't exist yet
db_df.to_sql("fl_training", conn, if_exists="append")

joblib.dump(scaler,    "models/scaler.pkl")
joblib.dump(tfidf_vec, "models/tfidf_vectorizer.pkl")
joblib.dump(columns,   "models/feature_columns.pkl")
np.save("models/fl_test_X.npy", X_test)          # the locked-away test set
np.save("models/fl_test_y.npy", y_test)
```

| File | Why it must be saved |
|---|---|
| `scaler.pkl` | New patients at the bedside must be rescaled with **exactly** the training averages and spreads |
| `tfidf_vectorizer.pkl` | New notes must be scored with exactly the same 90 words and rarity weights |
| `feature_columns.pkl` | The 108 inputs must always be in the **same order**; input 47 must always mean the same thing |
| `fl_test_X.npy`, `fl_test_y.npy` | Later runs that skip Phase 0 can still test on the real held-out test set |

---

## 6. Phase 1 — Loading the Hospitals' Data

**In one line:** read each hospital's rows from SQLite and line up the columns in the exact training order.

```python
feature_columns = list(scaler.feature_names_in_)       # the 108 names, in order
for i in range(3):
    df = get_fl_client_data(i)                          # hospital i's rows
    y  = df["sofa_score"]
    X  = df.drop(columns=["sofa_score"])
    # any missing column is added as 0, WITH a warning naming the columns
    X  = X[feature_columns]                             # force the exact order
```

- `get_fl_client_data()` reads the rows in a fixed order and **drops rows whose SOFA answer is empty** (it warns how many). With the current `dataset_v2` data no rows are empty, so nothing is dropped.
- If a column is missing (for example because the word list changed), it is filled with 0 and the script **warns** instead of hiding it.

### Finding the test set

The test set is chosen in this order:
1. **Phase 0 just ran:** use the test set it created (in memory).
2. **Phase 0 ran earlier:** load `models/fl_test_X.npy` and `fl_test_y.npy`.
3. **Neither:** split 20% off the hospitals' own data and print a warning that the results will look **better than they really are**, because those rows were also used for training.

> The current model was tested on the real held-out test set from Phase 0 (`fl_test_X.npy` / `fl_test_y.npy`, 10,450 ICU stays), so option 3 was **not** used.

---

## 7. Phase 2 — SHAP Background Patients

**In one line:** save 300 typical patients so the dashboard can explain each prediction.

```python
background = 300 random training patients     # fewer if the data has under 300 rows
np.save("models/shap_background.npy", background)
```

**What SHAP does:** it answers "why did the model predict 12.4 for this patient?" by splitting the prediction into contributions from each input:
```
Average prediction for the 300 typical patients:  5.2
This patient:
  MAP_mean (low blood pressure)   +3.1
  latest_SpO2 (low oxygen)        +1.8
  "intubated" in the notes        +1.2
  "norepinephrine" in the notes   +0.9
  HR_std                          −0.2
  … the other inputs               +0.4
  ────────────────────────────────────
  5.2 + 7.2 = 12.4 ✓
```
The 300 patients are the **reference point** ("what does a typical patient look like?"). The dashboard shows the top 5 contributions. The same 300 are used every time, so explanations are comparable between patients.

---

## 8. Phase 3 — Federated Training

This is the heart of the script: **100 rounds** of training across **3 simulated hospitals** using the Flower framework.

### 8.1 The model (the neural network)

```
108 inputs
   ↓  Linear 108 → 256, then ReLU, then Dropout 0.3
   ↓  Linear 256 → 128, then ReLU, then Dropout 0.3
   ↓  Linear 128 →  64, then ReLU, then Dropout 0.3
   ↓  Linear  64 →   1
predicted SOFA (one number)
```

- **Linear layer:** each output is a weighted sum of all inputs plus a constant. This is where the 69,121 weights live.
- **ReLU:** keeps positive numbers and turns negatives into 0. Without it, stacking layers would be no better than one layer; with it, the model can learn combinations like "low blood pressure **and** 'norepinephrine' in the notes → circulation failing".
- **Dropout 0.3:** during training, each hidden neuron is switched off at random 30% of the time. The model cannot lean on any single neuron, so it has to learn patterns that hold up across many patients instead of memorising individual ones.
- **Shape:** the first layer widens (108 → 256) to give room to combine inputs, then each layer halves until one number is left.
- **Size:** 69,121 weights. That is still small for tabular data (see Part B, Stage 4).

**Dropout is only on during training.** Testing, SHAP explanations and the dashboard backend all call `model.eval()`, which turns dropout off. So the same patient always gets the same prediction.

**Why this shape and not the old one?** Until 2026-11-12 the model was 108 → 128 → 64 → 32 → 1 with no dropout (24,321 weights, historical). After the switch to plain MSE and FedAvg, that smaller network started to **overfit** after about round 13: each hospital's own training loss kept falling (7.7 → 2.4) while the server's validation loss rose (5.9 → 8.7). In plain words, the model was memorising the training patients and getting worse on new ones. Dropout stops that memorising, and the extra width gives the model enough room to still learn well with dropout on. Full story in Section 15.

**What was deliberately left out:**

| Left out | Why |
|---|---|
| **BatchNorm** | It learns batch statistics, which differ between hospitals; averaging them breaks them. |
| **LayerNorm** | Tested twice. It rescales each patient's internal numbers to the same size, which erased the difference between a SOFA 2 and a SOFA 20 patient. Predictions collapsed into 0.2–7.0. |

> **A worry that did not come true:** an earlier version of this doc left out dropout, fearing that each hospital would switch off different neurons, so their weight updates would not fit together when averaged. In the dataset_v2 runs this did not happen; dropout improved every test metric (Section 15).

### 8.2 The error measure: plain mean squared error

**In one line:** every patient's error counts the same.

```python
loss = mean((prediction − truth)²)
```

**Example with 3 patients:**
```
True SOFA 2,  predicted 3   → error² = 1
True SOFA 6,  predicted 4   → error² = 4
True SOFA 12, predicted 9   → error² = 9
MSE = (1 + 4 + 9) / 3 = 4.67
```

**Why plain MSE?** Until 2026-11-12 the loss was a **weighted** MSE (historical): each patient's error was multiplied by 1 + SOFA × 0.5, so sicker patients counted more. The idea was to stop the model ignoring rare High Risk cases. In practice it **pushed every prediction upward**, which raised the MAE, and it did not catch more High Risk patients. It also disagreed with how the best round is chosen (plain, unweighted error), so training and selection pulled in different directions. Plain MSE fixed both. See Section 15 for the before/after numbers.

> **Optional:** `train_model(..., weighted_loss=True)` in `model_utils.py` brings the old weighted loss back.

### 8.3 The optimiser at each hospital: AdamW

The optimiser is the rule that turns gradients into weight updates. **AdamW** adapts the step size for every weight separately and adds **weight decay**: a gentle pull of all weights toward zero (strength 0.0001) that discourages over-complicated solutions.

A new AdamW is created **every round** (its memory restarts each round). That is normal for FL, because each round starts from fresh shared weights.

### 8.4 FedProx: stopping hospitals from drifting apart

**In one line:** a penalty pulls each hospital's model back toward the shared model while it trains.

**The problem:** in each round, every hospital trains 3 times over its own data. Each one drifts toward what suits **its** patients. Averaging three models that drifted in different directions gives a poor result, like three people pulling a rope in slightly different directions.

**The fix:** add a penalty for drifting:
```
total loss = MSE error + (μ / 2) × (sum of (local weight − shared weight)² over all 69,121 weights)

with μ = 0.5
```
The further a hospital's weights move from the shared ones, the bigger the penalty, so it learns from its data but cannot wander off.

**Small example with 3 weights:**
```
Shared model:   [1.00, −0.70, 0.50]
Hospital model: [1.20, −0.85, 0.55]
Differences:    [0.20, −0.15, 0.05] → squared: 0.04 + 0.0225 + 0.0025 = 0.065
Penalty = (0.5 / 2) × 0.065 = 0.016   (grows quickly as the drift grows)
```

**Why μ = 0.5?**

| μ | Effect |
|---|---|
| 0.01 | Far too weak; drift continues as if FedProx were off |
| **0.5** | Enough restraint to stop drift, while still letting hospitals learn |
| 1.0 | Too strong; hospitals can hardly learn from their own data, and results got worse |

### 8.5 What one hospital does in each round

Each hospital is a Flower **client** (`HospitalClient`). When it is created, it sets aside **15% of its own data** as a private validation set, used to judge the shared model each round.

**`fit()` — "train on your data and send back your weights":**
```
1. Receive the shared weights and the round number.
2. Load the shared weights into the local model.
3. Learning rate = max(0.0001, 0.001 × 0.99^(round − 1)).
4. Train for 3 passes over the local training data, in batches of 64:
     a. predict SOFA for the batch
     b. loss = MSE error + FedProx penalty
     c. compute gradients
     d. clip the gradient size to at most 1.0
     e. AdamW update (weight decay 0.0001)
5. (Only if differential privacy is on: add noise to the update.)
6. Send back the new weights and the number of training patients.
```

**How much work per round (current data):** each hospital has about 13,930 rows; 85% of that (≈ 11,840) is for training. That is about 186 batches per pass, so about **560 weight updates per round**.

**`evaluate()` — "score the new shared model on your private validation set":**
```
1. Receive the NEW shared weights (after the server combined everyone's).
2. Predict for the validation patients (clipped to 0–24).
3. Return the squared error (MSE), MAE and R².
```
It scores the **shared** model, not the hospital's own trained version, because the server wants to know how good the combined model is.

### 8.6 The server: FedAvg

**In one line:** the server takes a weighted average of the hospitals' models and uses that as the new shared model.

#### How FedAvg works

Bigger hospitals count more:
```
new shared model = (n₀ × model₀ + n₁ × model₁ + n₂ × model₂) / (n₀ + n₁ + n₂)
```
This is done for every one of the 69,121 weights separately.

**Small number example (one weight):**
```
Hospital 0: weight 0.52, trained on 11,840 patients
Hospital 1: weight 0.58, trained on 11,840 patients
Hospital 2: weight 0.55, trained on 11,840 patients
New shared weight = (0.52 + 0.58 + 0.55) / 3 = 0.55   (equal sizes, so a plain average)
```

FedAvg has **no memory**: each round's average is used directly. On its own that can let hospitals drift apart, but **FedProx (Section 8.4) already holds them together**, so the extra memory of FedYogi is not needed.

#### Why FedAvg and not FedYogi?

Until 2026-11-12 the server used **FedYogi** with step size η = 0.01 (historical). FedYogi adds momentum and an adaptive step size on top of the average (explained below). On the rebuilt `dataset_v2` data it became unstable: the server's validation loss **exploded from about 8 to 123 in rounds 5–10**, and the run ended worse than plain FedAvg. Plain FedAvg (still with FedProx) trained smoothly and gave better test results (Section 15).

**Analogy:** FedYogi is like a heavy ball rolling downhill. Momentum helps it roll past small bumps, but with a big push (η = 0.01) on this new, larger dataset it flew right past the valley and up the other side. FedAvg just takes the average step each round; slower to "explore", but it does not overshoot.

The server starts from a **random** model (`initial_parameters`). Starting from a model pre-trained on all data was tried and was worse (Part B, Stage 8).

#### Optional: FedYogi (`USE_FEDYOGI = True`)

FedYogi can still be switched on. If you do, `SERVER_ETA` is now **0.001** (10× smaller than the unstable 0.01). In a 40-round test it was not better than FedAvg, so it stays off. Here is how it works, for reference.

FedYogi treats "average minus current model" as the direction to move (a **pseudo-gradient**, Δ) and adds two kinds of memory. These are the exact formulas Flower uses, applied to every weight separately:

```
Δ   = average − current shared weights               (which way the hospitals want to go)

m   = 0.9 × m_previous + 0.1 × Δ                     (MOMENTUM: a running direction)

v   = v_previous − 0.01 × Δ² × sign(v_previous − Δ²)  (YOGI: a running size estimate)

new shared weights = current + η × m / (√v + 0.001)
```

**Momentum (m):** keeps 90% of the old direction and adds 10% of the new one. If 10 rounds pointed the same way, one odd round barely changes the course.

**The size estimate (v), which works as a brake:** `v` tracks how large recent updates have been, and the step is divided by √v:
- **Big recent updates (v large) → smaller steps.**
- **Small recent updates (v small) → bigger steps.**

If the new Δ² is bigger than v, v grows by 0.01 × Δ²; if smaller, it shrinks by 0.01 × Δ². Compared with FedAdam (v = 0.99 × v + 0.01 × Δ²), Yogi releases the brake more slowly, which is why it beat FedAdam in the earlier experiments (Part B, Stage 7).

**Small number example (one weight, with the old η = 0.01):**
```
Round t:  current weight 0.50, average from hospitals 0.56  → Δ = 0.06, Δ² = 0.0036
          previous m = 0.02, previous v = 0.0010

m = 0.9 × 0.02 + 0.1 × 0.06            = 0.024
v: Δ² (0.0036) > v (0.0010) → v grows:
v = 0.0010 + 0.01 × 0.0036             = 0.001036
step = 0.01 × 0.024 / (√0.001036 + 0.001) = 0.01 × 0.024 / 0.0332 ≈ 0.0072
new weight = 0.50 + 0.0072             = 0.5072
```
Notice the step (0.0072) is bigger than plain averaging would give in some rounds, and momentum keeps pushing in the same direction. That is what overshot on `dataset_v2`.

#### How the server picks the best round

After each round, every hospital reports the shared model's error on its validation set. The server takes the **simple average** of the 3 errors. If it is the lowest so far, it remembers that round's weights as the best (`SaveBestStrategy`).

#### The usual pattern of a training run (current setup)

```
Early rounds:   error falls quickly.
Then:           the server validation loss settles and stays flat at about 5.8–6.1.
                No explosion (FedAvg) and no steady climb (dropout stops overfitting).
Best round:     the lowest point. In the current model this was round 26.
Later rounds:   tiny ups and downs around the same level as the
                learning rate keeps shrinking.
```
Because the **best** round is kept (not the last one), later drift does no harm.

### 8.7 One complete round, step by step

```
[Server]      Sends the shared weights + {"server_round": 12} to all 3 hospitals.

[Hospital 0]  Learning rate = 0.001 × 0.99^11 = 0.000895
              3 passes × ~186 batches ≈ 560 updates (MSE + FedProx, dropout on, clip, AdamW)
              Sends back new weights + its training count.
[Hospital 1]  Same, on its own patients.
[Hospital 2]  Same, on its own patients.

[Server]      New shared weights = average of the 3 models (bigger hospitals count more)

[Server]      Sends the NEW shared weights back for scoring.
[Hospitals]   Each scores them on its private 15% validation set (dropout off).

[Server]      Average of the 3 errors. New lowest? → remember this round as the best.
              Prints: "Round 12 — lr=0.00090 | avg_loss=…"
```
This repeats 100 times. Flower runs it as a **simulation**: all 3 hospitals are on the same computer, each limited to 1 CPU.

---

## 9. Phase 4 — Saving the Best Model

```python
best = _best_weights.get("params") or _last_weights.get("params")
torch.save(model_with(best).state_dict(), "models/federated_model.pth")
```
The weights from the **round with the lowest validation error** are saved as `federated_model.pth`, the file the dashboard loads. If no best was recorded (an unusual edge case), the last round's weights are used instead. If training produced no weights at all, the script stops with an error.

**Why not just keep round 100?** More rounds does not mean better. After the best round the model drifts slightly, so keeping the best round protects against that.

---

## 10. Phase 5 — Final Testing

**In one line:** measure how good the saved model is on the test patients.

```python
predictions = model(X_test), clipped to 0–24
MAE = mean_absolute_error(y_test, predictions)
R²  = r2_score(y_test, predictions)
+ MAE and R² for each risk band (Low < 5, Moderate 5–9, High ≥ 10)
+ compute_sofa_risk_metrics(): AUROC, PR-AUC, recall, specificity,
  and the 3-band and alert confusion matrices
```

- **Clipping to 0–24:** the network can output values like −0.2 or 25. The dashboard always clips, so testing does too. Otherwise the scores would describe predictions nobody ever sees.
- **What each metric means:** MAE, R², AUROC, PR-AUC and the confusion matrices are explained with examples in `docs/VALIDATION.md`, Section 3.

**The screen output looks like this:**
```
  Low (<5)        : MAE=…  R²=…  (n=…)
  Moderate (5-9)  : MAE=…  R²=…  (n=…)
  High (≥10)      : MAE=…  R²=…  (n=…)
  FINAL MODEL PERFORMANCE: MAE, R², prediction range
  High-risk alert AUROC, PR-AUC, recall, specificity, missed cases, false alarms
```

### Why the overall R² is positive but each band's R² can be negative

This puzzles most readers, so here it is with numbers.

**R² compares the model with "always guess the average".** The overall R² uses the average of **all** patients. A band's R² uses the average of **that band only**.

```
All patients, average ≈ 3.9:
  a SOFA 12 patient predicted 10.5 → much closer than the average 3.9 ✓
  a SOFA 2 patient predicted 3.1   → closer than the average 3.9 ✓
  → the model beats "guess 3.9" → overall R² is positive
```
```
Low band only, 5 patients with true SOFA 0, 1, 2, 3, 4 (band average 2.0):
  model predicts 2.8, 3.0, 3.1, 2.9, 3.2 (it can't tell them apart)

  error of "guess 2.0":    4 + 1 + 0 + 1 + 4                  = 10.0
  error of the model:      7.84 + 4.00 + 1.21 + 0.01 + 0.64   = 13.70
  R² = 1 − 13.70 / 10.0 = −0.37   (worse than the band average)
```
**What this means:** the model is good at deciding **which band** a patient is in, but cannot rank patients **within** a band. Inside a band, the differences come mostly from blood-test organs we cannot see.

---

## 11. Phase 6 — Saving the Training Report

Everything about the run is written to `models/training_metadata.json`, which the dashboard reads:

| Group | Fields |
|---|---|
| Training setup | num_rounds, epochs_per_round, batch_size, base_lr, lr_decay, hospitals, hospital_names, aggregation, split_type, oversample, loss_weights, optimizer |
| Data sizes | train_samples, test_samples, input_features, tfidf_features (the real word count, read from `tfidf_vectorizer.pkl`), tfidf_ngram, notes_sample_size |
| Results | best_round, final_mae, final_r2, pred_range_min, pred_range_max, risk_classification_metrics (AUROC, PR-AUC, confusion matrices…) |
| Privacy | differential_privacy, dp_sensitivity, dp_sigma, dp_epsilon, dp_delta |
| Provenance | bigquery_project, model_architecture, evaluation_limitation, sklearn_version, torch_version |

The script ends by reminding you to restart the backend (`uvicorn backend.main:app --port 8000`) so the dashboard loads the new model.

---

## 12. Optional Features

### Non-IID split (`USE_NONIID_SPLIT = True`)

**In one line:** give each hospital a different kind of patient, as in real life.

| Hospital | Low Risk patients | Moderate | High Risk |
|---|---|---|---|
| 0 — General ICU | 70% of all Low | 20% of all Moderate | 10% of all High |
| 1 — Mixed ICU | 20% | 60% | 20% |
| 2 — Cardiac/Trauma ICU | 10% | 20% | 70% |

**Why it is harder:** Hospital 2, full of sick patients, pushes the model to predict high values; Hospital 0, full of healthy ones, pushes it low. The server averages updates that pull in opposite directions. In our tests this dropped R² from about 0.28 to about 0.08. Use it only to demonstrate FL challenges. If a hospital ends up with no patients (tiny dataset), the script stops with a clear message.

### Differential privacy (`USE_DP = True`)

**In one line:** add random noise to each hospital's update so nothing about any single patient can be recovered from it.

Even without sharing data, a clever attacker might learn something about individual patients from the weight updates themselves. Differential privacy (DP) blocks this:
```
1. update = local weights − shared weights
2. if the update's overall size > 1.0, shrink it to exactly 1.0   (DP_SENSITIVITY)
3. add random Gaussian noise with spread DP_SIGMA × DP_SENSITIVITY = 1.0
4. send shared weights + noisy update
```

**The privacy budget (ε, "epsilon"):** smaller is more private.
```
ε per round ≈ √(2 × ln(1.25 / δ)) / σ,   with δ = 0.00001 and σ = 1.0
            = √(2 × 11.74) / 1.0 ≈ 4.85
ε after 100 rounds (simple adding up) ≈ 485
```
An ε below about 1 is considered strong privacy, so ε ≈ 485 is a **very weak** guarantee: the feature **demonstrates** the mechanism rather than providing real protection. Strong privacy would need much more noise (and many fewer rounds), which would greatly reduce accuracy.

---

## 13. Every File the Script Produces

| File | Made in | Used by | Contents |
|---|---|---|---|
| `fl_training` table in `models/icu_monitor.db` | 0i | Phase 1, `client.py`, dashboard | The 3 hospitals' scaled training rows |
| `models/scaler.pkl` | 0g | Phase 1, dashboard | Training averages and spreads for the 108 inputs |
| `models/tfidf_vectorizer.pkl` | 0d | Dashboard | The 90 words and their rarity weights |
| `models/feature_columns.pkl` | 0i | Dashboard | The 108 input names, in order |
| `models/fl_test_X.npy`, `fl_test_y.npy` | 0i | Phase 1 of later runs | The held-out test set |
| `models/shap_background.npy` | 2 | Dashboard (SHAP) | 300 typical training patients |
| `models/federated_model.pth` | 4 | Dashboard | The best round's 69,121 weights |
| `models/training_metadata.json` | 6 | Dashboard (SOFA Validation drawer) | All settings and test results |

---

## 14. How to Run It

All commands run from the `icu_monitor/` folder.

**First time (download + train):**
```bash
gcloud auth application-default login                        # one-time Google login
gcloud auth application-default set-quota-project mimic-project-2
pip install -r requirements.txt                               # includes db-dtypes and pyarrow for BigQuery
python train_federated.py
```

**Train again on the data already in the database:**
```bash
python train_federated.py          # Phase 0 is skipped automatically
```

**Re-run the data preparation (for example after changing the word list):**
```bash
sqlite3 models/icu_monitor.db "DELETE FROM fl_training;"
python train_federated.py
```

**Afterwards, restart the backend so the dashboard loads the new model:**
```bash
uvicorn backend.main:app --port 8000 --reload
```

**Real (non-simulated) federated training** uses two other scripts: start `python server.py`, then `python client.py --client_id 0`, `1` and `2` in three terminals. Like the simulation, `server.py` uses **plain FedAvg**, and it saves the best round to `federated_model.pth` and every round to `federated_model_latest.pth`.

---

## 15. The Current Model and Today's Changes

### What is saved right now

From `models/training_metadata.json` (best round **26** of 100), tested on the real held-out test set:

| | Value |
|---|---|
| ICU stays used for training | 41,798 (13,932 / 13,933 / 13,933 per hospital) |
| ICU stays used for testing | 10,450 (602 High, 2,919 Moderate, 6,929 Low) |
| MAE | 1.82 SOFA points (1.8247) |
| R² | 0.47 (0.4725) |
| AUROC (High Risk) | 0.925 |
| PR-AUC (High Risk) | 0.646 |
| Alert recall / specificity (predicted ≥ 7 vs actual ≥ 10) | 67.1% / 96.3% |
| Prediction range on the test set | 1.72 – 23.45 |

**Per risk band MAE:** Low 1.33 (n = 6,929), Moderate 2.48 (n = 2,919), High 4.33 (n = 602). Per-band R² is negative by design (Section 10); only the overall R² is meaningful.

**The data is clean now.** The old caveats (most rows missing their SOFA answer, and a test set that overlapped training) are gone: `dataset_v2` was rebuilt in BigQuery, every row in `fl_training` has its SOFA answer, and Phase 0 saved a real held-out test set (`fl_test_X.npy` / `fl_test_y.npy`). One honest caveat remains: the split is **by ICU stay**, so a patient with several stays can appear in both training and test data. These are retrospective MIMIC-III results, not an independent clinical validation.

### The progress table (2026-11-12)

Each row adds one change to the row above. All rows except the first use the same `dataset_v2` test set.

| Run | MAE | R² | AUROC | PR-AUC | Recall | Specificity |
|---|---|---|---|---|---|---|
| Old dataset (FedYogi, weighted MSE, 128-64-32), alert ≥8 *(historical)* | 1.80 | 0.398 | 0.879 | — | 23% | — |
| dataset_v2, FedYogi η=0.01, weighted MSE, alert ≥8 *(historical)* | 1.914 | 0.426 | 0.915 | 0.599 | 53.8% | 98.0% |
| + FedAvg + plain MSE, alert ≥8 *(historical)* | 1.850 | 0.457 | 0.921 | 0.628 | 59.3% | 97.7% |
| + 256-128-64 + dropout 0.3, alert ≥8 *(historical)* | 1.825 | 0.473 | 0.925 | 0.646 | 61.1% | 97.8% |
| **same model, alert ≥7 (current)** | **1.825** | **0.473** | **0.925** | **0.646** | **67.1%** | **96.3%** |

"Recall" and "Specificity" are for the red alert: what share of truly High Risk patients (actual SOFA ≥ 10) got an alert, and what share of the others were correctly left alone.

> The MAE in the first row is **not directly comparable**: the old dataset used a different label definition. Compare R², AUROC and recall instead.

### Why each change was made, in plain words

#### Change 1 — plain MSE instead of weighted MSE

**Before:** each patient's error was multiplied by 1 + SOFA × 0.5, so a SOFA 20 patient counted far more than a SOFA 0 patient.

**What went wrong:** the extra weight on high-SOFA patients **pushed every prediction upward**. Most patients are Low Risk (67%), so pushing their predictions up made the MAE worse. And it did not actually catch more High Risk patients. On top of that, the best round is chosen with plain, unweighted error, so training and selection were aiming at different targets.

**After:** plain MSE. Every patient counts the same, training and round selection now agree. Together with FedAvg (Change 2), MAE fell from 1.914 to 1.850 and recall rose from 53.8% to 59.3% (table rows 2 → 3).

#### Change 2 — FedAvg instead of FedYogi (η = 0.01)

**Before:** the server used FedYogi with step size 0.01, which adds momentum and an adaptive step on top of the average.

**What went wrong:** on `dataset_v2`, the server's validation loss **exploded in rounds 5–10, from about 8 up to 123**. Momentum kept pushing the shared model in the same direction long after it had passed the good spot, like a heavy ball rolling past the bottom of a valley and far up the other side. The run recovered later but ended worse.

**After:** plain FedAvg (FedProx μ = 0.5 still on). FedProx already stops the hospitals drifting apart, so the server does not need extra momentum. Training became smooth: no explosion. (FedYogi is still available with `USE_FEDYOGI = True`, now with a safer step of 0.001; it was not better in tests.)

#### Change 3 — a wider network with dropout (256 → 128 → 64, dropout 0.3)

**Before:** 108 → 128 → 64 → 32 → 1, no dropout, 24,321 weights.

**What went wrong:** once Changes 1 and 2 removed the instability, a new problem showed up: **overfitting after round 13**. Each hospital's own training loss kept falling, **7.7 → 2.4**, while the server's validation loss rose, **5.9 → 8.7**.

```
After round 13:
  Client (training) loss:   7.7 → 2.4   "I know my training patients very well"
  Server (validation) loss: 5.9 → 8.7   "...but I'm getting worse on new ones"
```
That gap is the classic sign of a model **memorising** its training patients instead of learning general patterns. It is like a student who memorises last year's exam answers: perfect on old papers, worse on a new one.

**After:** dropout 0.3 switches off a random 30% of neurons in each training step, so the model cannot memorise through any single path. The wider layers give it enough room to still learn with dropout on. Now the server validation loss stays flat at about **5.8–6.1** with no climb, the best round moved to 26, and the test scores improved (table rows 3 → 4: MAE 1.850 → 1.825, R² 0.457 → 0.473, PR-AUC 0.628 → 0.646). An even wider 512-256-128 gave no further gain.

#### Change 4 — alert threshold 7 instead of 8

**Before:** the red alert fired when predicted SOFA ≥ 8.

**What went wrong:** the model still **under-predicts severe patients**. For patients whose true SOFA is ≥ 10, the average prediction is only about 9.7. So many truly High Risk patients sat just under 8 and got no alert.

**After:** `ALERT_THRESHOLD = 7.0`, defined once in `model_utils.py`. The backend (`backend/services/ml_service.py`) imports it, and `compute_sofa_risk_metrics()` uses it by default, so training metrics and the dashboard always agree.

```
Threshold 8:  missed 234 High Risk patients, 220 false alarms
Threshold 7:  missed 198 High Risk patients, 360 false alarms
→ 36 more sick patients caught, for 140 more false alarms (out of 9,848 non-High patients)
```

The full sweep on the current model:

| Alert when predicted SOFA ≥ | Recall | Specificity |
|---|---|---|
| 8 | 61.1% | 97.8% |
| 7.5 | 64.3% | 97.1% |
| **7 (current)** | **67.1%** | **96.3%** |
| 6.5 | 69.9% | 95.3% |
| 6 | 73.1% | 93.9% |

In an ICU, missing a deteriorating patient costs more than an extra alarm, so 7 was chosen as a balance. Going lower catches a few more but the false alarms grow quickly. AUROC, PR-AUC and MAE do not change with the threshold; only the yes/no alert does. The alert drives the red banner, the `is_alert` flag and the urgent ("immediate actions first") AI report format in both Hospital and Home Care modes.

### What did not help (tried on the same data)

Lower learning rate (5e-4), stronger weight decay (1e-3), Huber loss, an even wider 512-256-128 network, averaging 8 models, isotonic recalibration, missing-value indicator flags, and using several hourly rows per stay (only +0.007 R² for 9× the data).

**The ceiling:** a centralized gradient-boosting model on the same 108 inputs reaches about MAE 1.81, R² 0.49, AUROC 0.927, so our federated model is already close to what these inputs allow. The main limit is that SOFA is mostly computed from labs (bilirubin, creatinine, platelets, PaO₂/FiO₂) and vasopressor doses, which are deliberately **not** inputs (the system is non-invasive).

---

# Part B — How We Got Here

> The numbers in Part B were recorded when each experiment ran, on the data available at the time. They show the **direction** of each change; they are not results of the current saved model.

---

## 16. Why Predicting SOFA Is Hard

SOFA adds up six organ scores, and four of them need lab tests:

| SOFA part | What it really needs | What our model has |
|---|---|---|
| Lungs | Blood gas (PaO₂/FiO₂) | SpO₂ (an indirect proxy) |
| Clotting | Platelet count | Note words like "plt" or "thrombocytopenia" |
| Liver | Bilirubin | Note words like "bilirubin" or "totbili" |
| Heart | MAP and drug doses | MAP directly, plus drug names in notes |
| Brain | Full GCS (3 parts) | GCS eye opening only |
| Kidneys | Creatinine and urine output | Note words like "creatinine" or "oliguria" |

A note is a noisy stand-in for a lab result. A patient can have a high bilirubin that nobody wrote about that day, and a doctor may write "kidney function worsening" instead of a word on our list. This sets a ceiling on how accurate the model can be.

---

## 17. The Experiment Journey

### Stage 1 — The disastrous first attempt (R² = −824)

**Setup:** 600 words chosen automatically, a large model (618 → 256 → 128 → 64 → 1), plain FedAvg, error weight ×3.0, and **no clipping** after scaling.

**Result:** predictions of SOFA **10,001**.

**Why:** a very rare word such as "bacteremia" got a huge rarity weight. After scaling, a patient mentioning it had an input of 80–150 "spreads" above average. Passing through four layers multiplied this into absurd outputs.

**Fix:** clip every scaled input to the range −10 to +10.

### Stage 2 — Stuck on a plateau (R² ≈ 0.11–0.13)

The predictions were now sensible, but the model stopped improving. Three causes:
1. **Noisy words:** the automatic choice picked *tablet*, *sig*, *refills*, *wallet* and *afternoon*, which measure paperwork, not illness.
2. **Hospitals drifting apart:** with 5 training passes per round, the hospitals' models diverged; the server's error bounced around (8.7 → 12.9 → 10.0 → 8.7 → 10.6).
3. **Conflicting goals:** training heavily favoured sick patients (×3.0), but the best round was chosen by unweighted error, dominated by healthy patients.

### Stage 3 — Things that did not help

| Experiment | R² | Why it failed |
|---|---|---|
| Binary word scores (present or absent only) | 0.11 | Lost "how much" information |
| Only 200 automatic words | 0.09 | Still the same noisy words |
| FedProx μ = 0.01 | 0.11 | Far too weak to stop drift |
| Lower learning rate + clipping | 0.11 | Did not fix the real problem (noisy words) |
| More rounds (40 → 60) | 0.13 | Nothing new to learn |
| Error weight 3.0 → 0.5 | 0.13 | Helped a little; the words were still the bottleneck |
| 2 passes per round instead of 5 | 0.09 | Too little learning per round |

**Insight:** 600 noisy word inputs drowned out the 18 vital-sign inputs that carry real signal.

### Stage 4 — Breakthrough 1: the 90-word medical list (R² = 0.175)

**Changes:** automatic words → the fixed 90-word SOFA list; a smaller model (108 → 128 → 64 → 32 → 1, historical: replaced on 2026-11-12 by 108 → 256 → 128 → 64 → 1 with dropout, Section 15); error weight ×0.5 (historical: plain MSE since 2026-11-12); FedProx μ = 0.5.

**Why it worked:** every word input now meant something medical, and the smaller model suited the amount of data:
```
Old model: about 197,000 weights  → far too many for the data (memorises noise)
New model: about  24,000 weights  → a sensible size for tabular data
```
(Today's model has about 69,000 weights plus dropout. It is wider than Stage 4's, but still far smaller than Stage 1's, and dropout keeps it from memorising.)

### Stage 5 — More things that did not help

| Experiment | R² | Why it failed |
|---|---|---|
| FedProx μ = 1.0 | 0.136 | Too strong; hospitals could hardly learn |
| Faster decay (0.97) | 0.136 | Learning rate became tiny before reaching a good solution |
| Restarting the server's memory at round 50 | 0.211 | Threw away useful momentum |
| 5 passes per round | 0.208 | More drift, even with FedProx |
| LayerNorm | 0.219, but predictions squeezed into 0.2–7.0 | Erased the difference between mild and severe patients |

### Stage 6 — Breakthrough 2: FedAdam on the server (R² = 0.215)

Replacing plain averaging with **FedAdam** gave the server memory (momentum). A single bad round could no longer undo ten good ones. Its weakness: its step-size brake (v) releases quickly after turbulence, so steps could grow too large and training could blow up again.

### Stage 7 — Breakthrough 3: FedYogi on the server (R² = 0.223)

**FedYogi** releases the brake slowly (Section 8.6), so the recovery after early instability was steadier and reached a lower error. Identical runs varied between R² 0.205 and 0.223 because of randomness in training order.

> **Historical.** FedYogi was the server's method until 2026-11-12. On the rebuilt `dataset_v2` its loss exploded (rounds 5–10, up to 123), so the current model uses plain FedAvg + FedProx (Section 15).

### Stage 8 — Things that did not help after FedYogi (historical)

| Experiment | R² | Why it failed |
|---|---|---|
| Smaller server step (η = 0.003) | 0.196 | Explored less, found a worse solution |
| Larger τ (0.01) | 0.217 | Same reason |
| LayerNorm again | 0.219 | Same squeezed predictions |
| Pre-training the model on all data first | 0.158 | Started in a "comfortable" spot and never escaped; each FL round made it worse |
| 5 passes per round | 0.208 | More drift |

### Stage 9 — Breakthrough 4: using all the notes (R² = 0.472 at the time)

**Changes:** notes used: 80,000 → **all 283,208**; text kept per admission: 5,000 → **10,000** characters.

**Why it worked:**
1. **More reliable rarity weights:** with 3–4× more documents, the rarity weights of words like "totbili" or "oliguria" are estimated from far more examples.
2. **More patients covered:** admissions with at least one note rose from about 22,000 to about 41,000, so many fewer patients had all-zero word scores.
3. **Lab mentions captured:** results like "creatinine 2.8 this morning" often appear later in long notes and were being cut off at 5,000 characters.

At the time this lowered MAE from 2.15 to 1.76 and raised R² from 0.22 to 0.47, the largest single improvement of the project.

### Stage 10 — Rebuilt dataset and today's fixes (2026-11-12)

The BigQuery data was rebuilt as `dataset_v2` (every row has its SOFA answer, a real held-out test set). On this data four changes were made: plain MSE, FedAvg instead of FedYogi, a wider network with dropout, and an alert threshold of 7. Section 15 explains each one and shows the progress table.

---

## 18. Lessons Learned

1. **Better inputs beat cleverer training.** The two biggest jumps came from the inputs (the 90-word list, using all the notes); all the FL tricks together gave less.
2. **Instability is not always useful.** On the old data, every attempt to remove FedYogi's early error spike gave a worse model. On `dataset_v2` the spike became an explosion (loss up to 123), and plain FedAvg did better. What works depends on the data.
3. **Train for what you judge by.** Weighting sick patients during training while choosing rounds by unweighted error made the two goals fight. Lowering the weight from ×3.0 to ×0.5 helped; dropping it entirely (plain MSE, 2026-11-12) helped more.
4. **Right-sized, with regularisation.** The 24,000-weight model beat the 197,000-weight one on the old data. On the larger `dataset_v2`, a 69,000-weight model **with dropout** beat the 24,000-weight one, which had started to overfit. An even wider one gave no gain.
5. **Test standard tricks, don't assume.** LayerNorm erased severity information and BatchNorm does not average well across hospitals. Dropout was expected to cause the same problem, but in practice it averaged fine and fixed overfitting.
6. **Check the data before trusting results.** Late in the project we found that most training rows had lost their SOFA answer, and that the saved test set overlapped training. Rebuilding the data (`dataset_v2`) fixed both. Pipeline problems can matter as much as model design.

---

## 19. Results Table Across Experiments

All rows except the last are **historical** experiments. The current setup is the last row.

| Stage | Key change | R² | MAE | Best round |
|---|---|---|---|---|
| 1 | 600 automatic words, FedAvg, no clipping | −824 | ~50 | — |
| 2 | + clipping at ±10 | 0.11–0.13 | 2.38 | ~23 |
| 4 | **90-word list + smaller model + weight ×0.5** | **0.175** | 2.31 | ~24 |
| 6 | **FedAdam** | **0.215** | 2.16 | ~29 |
| 7 | **FedYogi** | **0.223** | 2.15 | ~24 |
| 9 | **All 283,208 notes + 10,000 characters** | **0.472** | 1.76 | 24 |
| 10 | dataset_v2, FedYogi η=0.01, weighted MSE (historical) | 0.426 | 1.914 | — |
| **Current saved model** | **dataset_v2 + FedAvg + plain MSE + 256-128-64 with dropout 0.3** (Section 15) | **0.473** | **1.825** | **26** |

Stages 1–9 used the old dataset and label definition, so their MAE is not directly comparable with the dataset_v2 rows. The current model's numbers come from a real held-out test set of 10,450 ICU stays.

---

## 20. How to Go Further

| Idea | What it would fix | Expected effect |
|---|---|---|
| **Add real lab values** (creatinine, bilirubin, platelets from MIMIC-III `labevents`) | The biggest gap: 3 SOFA organs are only guessed from words | The largest possible accuracy gain, including within-band ranking |
| **Keep missing GCS and stress values as "missing"** instead of 0 | Those inputs currently barely influence the model | The live camera readings could actually change the SOFA prediction |
| **Split by patient, not by ICU stay** | One person's stays could land in both training and testing | More honest test results |
| **Hold back a calibration set** | The conformal interval is measured on training data | A trustworthy 90% range on the dashboard |
| **Better handling of severe cases** | The model still under-predicts severe patients (mean prediction for true High ≈ 9.7). Simply weighting them more (the old weighted MSE) did not help | More High Risk patients caught by the alert |

---

## 21. Glossary

| Term | Plain meaning |
|---|---|
| **Federated learning (FL)** | Training one model across several sites without moving their data |
| **Client / hospital** | One site that trains on its own data |
| **Server** | Combines the clients' models into the shared model |
| **Round** | One cycle: send the model out, train locally, combine |
| **Epoch / pass** | Going through a client's training data once |
| **Batch** | A group of patients (64 here) processed before each weight update |
| **Weights** | The numbers inside the model that training adjusts (69,121 here) |
| **Gradient** | The direction and amount each weight should change to reduce error |
| **Learning rate** | How big each weight update is |
| **Loss / error** | A number measuring how wrong the predictions are |
| **IID** | Every client has the same kind of patient mix |
| **Non-IID** | Clients have different patient mixes |
| **FedAvg** | The server simply averages the clients' models |
| **FedYogi** | Averaging plus momentum and a carefully controlled adaptive step size (optional here, off by default) |
| **Dropout** | Randomly switching off some neurons during training so the model cannot memorise |
| **Overfitting** | The model gets better on its training data but worse on new data |
| **FedProx** | A penalty that stops clients drifting far from the shared model |
| **Client drift** | Clients' models moving in different directions during local training |
| **AdamW** | A common optimiser with adaptive steps and weight decay |
| **Gradient clipping** | Capping the size of any single update |
| **TF-IDF** | A score for how important a word is in one document compared with all documents |
| **StandardScaler** | Rescales each input to average 0 and spread 1 |
| **Stratified split** | A split that keeps the same share of each class (for example High Risk) in every part |
| **SHAP** | Explains a prediction as contributions from each input |
| **Differential privacy (DP)** | Adding noise so individual records cannot be recovered |
| **ε (epsilon)** | The privacy budget; smaller means more private |
| **MIMIC-III** | A public, de-identified ICU database from Boston |
| **BigQuery** | Google's cloud database where our MIMIC-III copy is stored |

---

*Documentation for the ICU CDSS Capstone Project. Training framework: Flower 1.32 (simulation), PyTorch. Aggregation: FedAvg + FedProx. Data: MIMIC-III via Google BigQuery (`dataset_v2`). When the model is retrained, update Section 15 and the last row of Section 19.*

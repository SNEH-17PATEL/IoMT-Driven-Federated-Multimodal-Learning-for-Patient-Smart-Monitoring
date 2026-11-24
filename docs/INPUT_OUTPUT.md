# Complete Input-Output Lifecycle — ClinIQ ICU CDSS
**Multimodal Intelligence System with Federated Learning for ICU Patient Monitoring**
**Status:** Fully implemented and verified
**Last updated:** 2026-11-24

---

## What This Document Covers

This document follows **one prediction** from the moment a new reading arrives, through every computation, to the final screen. Every step says:
- what goes in and where it comes from,
- exactly what is computed,
- what comes out (with real values),
- why the step exists and what would go wrong without it.

**All example numbers in this document are real.** They were produced by running the final trained model on the demo **Patient 3** (septic shock), reading 3 of 30. Where a number is only there to explain an idea (a made-up arithmetic example), the text says so.

> Related documents: model and training details are in `docs/TRAIN_FEDERATED.md` and `docs/MODEL_DISCUSSION.md`; the checks on the SOFA score and the AI report are in `docs/VALIDATION.md`; the whole-project overview is in `docs/Project_Context.md`.

---

## Table of Contents

1. [The Big Picture — What the System Does](#1-the-big-picture--what-the-system-does)
2. [System Inputs — Where the Data Comes From](#2-system-inputs--where-the-data-comes-from)
3. [Input Checks — The Safety Gate](#3-input-checks--the-safety-gate)
4. [Step 1 — Sliding Window (Temporal Buffer)](#4-step-1--sliding-window-temporal-buffer)
5. [Step 2 — Trend Features (9)](#5-step-2--trend-features-9)
6. [Step 3 — Latest Vital and Camera Features (9)](#6-step-3--latest-vital-and-camera-features-9)
7. [Step 4 — TF-IDF Text Encoding (90)](#7-step-4--tf-idf-text-encoding-90)
8. [Step 5 — Feature Fusion and Alignment (108)](#8-step-5--feature-fusion-and-alignment-108)
9. [Step 6 — StandardScaler Normalisation](#9-step-6--standardscaler-normalisation)
10. [Step 7 — Network Inference (108 → SOFA)](#10-step-7--network-inference-108--sofa)
11. [Step 8 — SOFA Post-processing, Risk Level and Alert](#11-step-8--sofa-post-processing-risk-level-and-alert)
12. [Step 9 — Runtime Safety Checks and the Conformal Range](#12-step-9--runtime-safety-checks-and-the-conformal-range)
13. [Step 10 — SHAP Explainability](#13-step-10--shap-explainability)
14. [Step 11 — Trend Text](#14-step-11--trend-text)
15. [Step 12 — The Response Sent to the Browser](#15-step-12--the-response-sent-to-the-browser)
16. [Step 13 — The AI Report: Prompt Construction](#16-step-13--the-ai-report-prompt-construction)
17. [Step 14 — The AI Report: Reliability Score](#17-step-14--the-ai-report-reliability-score)
18. [Step 15 — What the Clinician Sees](#18-step-15--what-the-clinician-sees)
19. [Complete Data Flow Diagram](#19-complete-data-flow-diagram)
20. [Artifacts Loaded at Startup](#20-artifacts-loaded-at-startup)
21. [Feature Count Summary](#21-feature-count-summary)
22. [Timing, Caching and Failure Behaviour](#22-timing-caching-and-failure-behaviour)

---

## 1. The Big Picture — What the System Does

A clinician is watching a patient's dashboard. **Every 90 seconds** a new reading arrives. Within seconds the system:

1. **Predicts** a SOFA score (0–24), a measure of how badly the patient's organs are failing.
2. **Classifies** the risk (Low / Moderate / High) and decides whether to raise the red **alert** (predicted SOFA ≥ 7).
3. **Checks itself:** three plausibility checks and a 90% uncertainty range.
4. **Explains** which factors pushed the score up or down (SHAP).
5. **Writes** a structured clinical report with a large language model (LLM), in Hospital or Home Care style.
6. **Scores the report's reliability** with nine checks against the real patient data.

The system is not replacing clinical judgment. It is a second opinion, based on patterns learned from 41,798 MIMIC-III ICU stays, with its reasoning shown.

**Running example used in every step (real values from the final model):**

```
Patient 3 — "Septic shock, vasopressors, intubated, multi-organ failure"
Reading 3 of 30 (the 30 recorded readings are replayed in a loop)

  HR:    124 bpm       (normal 60–100)
  RR:    30 br/min     (normal 12–20)
  SpO₂:  85 %          (normal 95–100)
  Temp:  39.2 °C       (normal 36.0–37.5)
  SBP:   84 mmHg       (normal 90–120)
  DBP:   49 mmHg       (normal 60–80)
  MAP:   61 mmHg       (normal 70–100)
  GCS Eye Opening: 2   (eyes open only to pain)
  Stress Score:    8.0 (high visible distress)

Clinical note: "72-year-old male with septic shock on norepinephrine
 vasopressor support. Intubated and mechanically ventilated at FiO2 0.8.
 SpO2 critically low despite ventilator. Creatinine critically elevated at
 3.2 mg/dL with oliguria less than 0.3 mL/kg/h. History of COPD and chronic
 kidney disease. Vasopressor requirement increasing. Broad-spectrum
 antibiotics initiated. Blood cultures pending. Hemodynamic instability
 worsening."

Result: predicted SOFA 8.2 → Moderate Risk, but the alert fires (8.2 ≥ 7).
```

---

## 2. System Inputs — Where the Data Comes From

The dashboard does **not** ask the clinician to type vitals. In the demo, the data comes from the database and, optionally, the webcam. There are five kinds of input:

### Input 1 — The next recorded reading (7 vital signs)

Each demo patient has **30 recorded readings** (from MIMIC-III) in the `sample_patients` table. Every 90 seconds the backend takes the next one (`row_idx`, wrapping around after 30).

| Input | Unit | What it measures | Normal range (dashboard) | Our example |
|---|---|---|---|---|
| HR | bpm | Heart rate | 60–100 | 124 |
| RR | br/min | Respiratory rate | 12–20 | 30 |
| SpO₂ | % | Blood oxygen saturation | 95–100 | 85 |
| Temp | °C | Body temperature | 36.0–37.5 | 39.2 |
| SBP | mmHg | Systolic blood pressure | 90–120 | 84 |
| DBP | mmHg | Diastolic blood pressure | 60–80 | 49 |
| MAP | mmHg | Mean arterial pressure | 70–100 | 61 |

Each vital gets a status badge (`low` / `normal` / `high`) by comparing it with these ranges. (The short trend summary that goes to the AI report uses 36.5 and 100 as the lower limits for temperature and SBP; both sets are as written in the code.)

**What the vitals cover in SOFA:** SpO₂ is a proxy for the respiratory score, MAP is the cardiovascular score, and GCS eye opening is part of the brain score. The other organs (clotting, liver, kidney) need lab values that are not inputs; they are captured only through words in the clinical note.

### Input 2 — The two camera / observation values

| Input | Scale | Meaning | Our example |
|---|---|---|---|
| GCS Eye Opening | 1–4 | 4 = eyes open on their own, 3 = to voice, 2 = to pain, 1 = no response | 2 |
| Stress Score | 0–10 | Visible distress or pain: LOW < 3.5, MODERATE 3.5–6.5, HIGH ≥ 6.5 | 8.0 |

Two sources:
- **Recorded values** from the sample row (the default).
- **Live webcam values**, when the clinician turns the **Camera monitor** on. The browser analyses the face (MediaPipe, entirely on the device) and sends only two numbers, `cv_gcs_eye` and `cv_stress`. When present they **replace** the recorded values in the model input. Details: `docs/CV_MONITOR.md`.

### Input 3 — The clinical note (free text)

Each sample row has a clinical note (shown above). Notes capture what vitals cannot: **why** things are abnormal ("SpO₂ critically low *despite* ventilator"), **lab values** that are not inputs ("creatinine critically elevated"), **treatments in place** ("on norepinephrine") and **history** ("COPD and chronic kidney disease"). The text becomes 90 numbers in Step 4.

### Input 4 — The vitals history (from the database)

The `patient_vitals` table in `models/icu_monitor.db` holds the **last 20 readings** for each patient with timestamps. It is how the system remembers the trajectory. (An older version used a CSV file and a "Reset Patient History" button; the database replaced both.)

### Input 5 — Pre-trained artifacts (loaded from disk once)

These are products of the federated training run, loaded at startup (full list in Section 20): `federated_model.pth`, `scaler.pkl`, `tfidf_vectorizer.pkl`, `feature_columns.pkl`, `shap_background.npy` and `training_metadata.json`. They do not change between predictions.

### The request that starts everything

The browser sends `POST /api/predict` with:
```json
{ "patient_id": 3, "row_idx": 2, "cv_gcs_eye": 2, "cv_stress": 8.0 }
```
The last two fields are optional (only when the camera monitor is on).

---

## 3. Input Checks — The Safety Gate

Before any computation, the API checks the request. A failed check stops the prediction with a clear error:

| Check | Rule | Error |
|---|---|---|
| Patient | `patient_id` must be 1, 2 or 3 | 400 "patient_id must be 1, 2, or 3" |
| Camera eye score | `cv_gcs_eye`, if given, must be 1, 2, 3 or 4 | 400 "cv_gcs_eye must be 1–4" |
| Camera stress | `cv_stress`, if given, must be between 0 and 10 | 400 "cv_stress must be 0–10" |
| Data present | The patient must have sample rows | 400 "No sample data… Run migrate_to_db.py first" |
| Anything else | Unexpected failure | 500 "Prediction failed — check server logs." (details go to the server log only) |

The vitals themselves come from the database, so they are not typed by a person. If a stored value is missing or not a number (NaN or infinity), `_safe_float()` replaces it with a safe default (for example SpO₂ 98, temperature 37, SBP 120, DBP 80, MAP 90) so one bad cell never crashes a prediction.

The report endpoint has its own checks (Section 16): the care mode must be `hospital` or `homecare`, a prediction must already exist for that patient (otherwise **409**), and the Groq key must be configured (otherwise **500**). The key is checked up front so the system fails before running any AI call, not after.

---

## 4. Step 1 — Sliding Window (Temporal Buffer)

**Purpose:** keep a 20-reading history of the vitals so trends can be computed.

**Why one reading is not enough.** SpO₂ = 91% could mean a steady 91% for hours (concerning but stable), a fall from 97% an hour ago (urgent), or a probe that slipped (false alarm). These need very different responses and look identical in a single reading. Twenty readings show the direction.

**The process:**
```
Input:  the current reading's 7 vitals  +  the patient_vitals table

A. If the patient has NO stored vitals yet (fresh start):
     seed the window with earlier readings from the sample data
     (starting from row 1, so row 0 — the current reading — is not duplicated),
     timestamped a minute and a half apart going backwards in time.
B. Append the current reading with the real clock time.
C. Keep only the newest 20 rows for this patient (older rows are deleted).
D. Read the window back: vitals_df, up to 20 rows × 7 vitals.
```

**Why 20?** The training data used 20-reading windows, so the model learned what "normal trend" looks like over exactly that span. A different window length would shift every trend feature and make the model's inputs inconsistent with its training.

**Example (Patient 3):** the window held 20 rows; its trend statistics are in the next step.

---

## 5. Step 2 — Trend Features (9)

**Purpose:** summarise the 20-reading trajectory into 9 numbers.

```
HR_mean    = mean of HR over the window
HR_std     = standard deviation of HR  (variability; 0 if only one reading)
RR_mean    = mean of RR
SpO2_mean  = mean of SpO₂
SpO2_min   = lowest SpO₂ in the window
Temp_mean  = mean of Temp
SBP_mean   = mean of SBP
DBP_mean   = mean of DBP
MAP_mean   = mean of MAP
```

**Real values for Patient 3:**

| Feature | Value |
|---|---|
| HR_mean | 123.3 bpm |
| HR_std | 2.98 bpm (a steady, sustained fast heart rate) |
| RR_mean | 29.45 br/min |
| SpO2_mean | 85.67 % |
| SpO2_min | 84.5 % |
| Temp_mean | 39.11 °C |
| SBP_mean | 84.3 mmHg |
| DBP_mean | 49.55 mmHg |
| MAP_mean | 61.3 mmHg |

**Why these 9 and not mean + std for all 7 vitals (14)?** Each was chosen for clinical value:
- **HR_mean and HR_std.** Heart-rate *variability* is a signal itself. Very low variability at a high rate can mean autonomic dysfunction in sepsis; very high variability can mean instability. (Made-up illustration: readings 120, 121, 119, 120, 122 have std ≈ 1.1; readings 90, 130, 85, 125, 95 have std ≈ 21 with a similar mean, a different physiological story.)
- **SpO2_mean and SpO2_min.** The mean shows the overall level; the minimum catches even a brief dangerous dip that a mean would hide. (Illustration: 95, 96, 94, 95, 70, 95, 96, 94 has mean ≈ 91.9 but min 70.)
- **The others are means only.** Their variability is a weaker predictor; adding it would raise the count from 9 to 16 for little gain.

---

## 6. Step 3 — Latest Vital and Camera Features (9)

**Purpose:** include the *current* values next to the trend summaries, because a trend summary alone loses the present state.

```
latest_HR       124.0       latest_DBP        49.0
latest_RR        30.0       latest_MAP        61.0
latest_SpO2      85.0       GCS_eye_opening    2.0
latest_Temp      39.2       stress_score       8.0
latest_SBP       84.0
```
(The 7 vitals of the current reading plus the 2 camera/observation values = 9.)

**Why keep both trends and latest values?** They answer different questions (made-up illustration): a patient whose HR rose 85 → 122 over the window has `HR_mean` ≈ 98 ("moderately high") but `latest_HR` = 122 ("acutely high"). A patient recovering from 130 → 96 has `HR_mean` ≈ 111 but `latest_HR` = 96 ("nearly normal"). Without latest values the model would not know the present state; without trends it would not know whether the present state is a spike or a sustained pattern.

**Why GCS and stress are in this group:** they are point-in-time observations (camera or nurse), not a series of vitals.

---

## 7. Step 4 — TF-IDF Text Encoding (90)

**Purpose:** turn the free-text note into 90 numbers the network can use.

### What TF-IDF is
A score for each word in a fixed list:
- **TF (term frequency):** how often the word appears in *this* note. We use `sublinear_tf`, so 1 mention scores 1.0, 5 mentions about 2.6 and 10 mentions about 3.3 (repeats count less and less).
- **IDF (inverse document frequency):** how *rare* the word is across all training notes. A word found everywhere ("acute") gets a small boost; a rare, specific word ("oliguria") gets a big one.

Score ≈ TF × IDF, so a word is high when it is frequent in this note **and** rare overall.

### Why a fixed list of 90 words
Letting TF-IDF pick the 600 "most variable" words chose paperwork words: "tablet", "sig" (take as directed), "refills", "disp", "capsule", "afternoon", "wallet". Sicker patients have more notes, so these words measured **how much was written**, not how ill the patient was. A model trained on them learns "more paperwork = sicker", which fails on new patients.

The fixed list has 90 terms, each tied to a SOFA organ or to overall severity:

| SOFA area | Example words in the list |
|---|---|
| Respiratory | intubated, ventilator, vent, cpap, bipap, hypoxia, respiratory failure, extubated |
| Coagulation | plt, platelets, bleeding, hemorrhage, inr, ptt, coagulopathy, thrombocytopenia |
| Hepatic | bilirubin, bili, totbili, liver, jaundice, hepatic, cirrhosis, liver failure |
| Cardiovascular | hypotension, shock, vasopressor, pressors, norepinephrine, levophed, dopamine, septic shock |
| Brain | sedated, sedation, coma, altered, confused, gcs, encephalopathy, delirium, unresponsive |
| Kidney | creatinine, creat, renal, kidney, dialysis, aki, oliguria, urine output, crrt |
| Infection | sepsis, septic, bacteremia, infection, cultures, antibiotics, fever, blood cultures |
| Lab words | lactate, wbc, hgb, hct, sodium, potassium, hco, angap, bun |
| General severity | failure, acute, pneumonia, ards, pulmonary, dyspnea, edema |

### The configuration
```python
TfidfVectorizer(
    vocabulary=list(_SOFA_VOCABULARY),             # the 90 words
    ngram_range=(1, 2),                            # single words and two-word phrases
    token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b', # words of 3+ letters
    sublinear_tf=True,
    norm="l2",
)
```
**L2 normalisation** divides each note's 90 scores by their overall length, so a 5,000-word note and a 500-word note are on the same scale. Without it, a long, well-documented stable patient could look sicker than a briefly documented critical one.

### Real result for Patient 3
Of the 90 words, **13 appear** in the note; the other 77 are 0.

| Word | TF-IDF | Why it fires |
|---|---|---|
| vasopressor | 0.619 | "norepinephrine vasopressor support", "vasopressor requirement" |
| oliguria | 0.320 | "oliguria less than 0.3 mL/kg/h" |
| norepinephrine | 0.304 | named drug |
| septic shock | 0.297 | two-word phrase |
| septic | 0.253 | |
| kidney | 0.230 | "chronic kidney disease" |
| blood cultures | 0.226 | "blood cultures pending" |
| shock | 0.210 | |
| ventilator | 0.190 | "despite ventilator" |
| cultures | 0.175 | |
| intubated | 0.151 | |
| creatinine | 0.130 | "creatinine critically elevated" |
| antibiotics | 0.122 | "broad-spectrum antibiotics" |

These 90 numbers tell the model that the cardiovascular, kidney, respiratory and infection areas are all mentioned strongly. A note with none of the 90 words gives all zeros, which is valid.

**At the bedside** the note is transformed with the saved `tfidf_vectorizer.pkl`; words outside the list are ignored and the vocabulary and IDF weights are exactly those used in training.

---

## 8. Step 5 — Feature Fusion and Alignment (108)

**Purpose:** join the three groups into one 108-number vector for the network.

| Group | Count | Content |
|---|---|---|
| Trend vitals | 9 | Step 2 |
| Latest vitals + camera | 9 | Step 3 |
| TF-IDF words | 90 | Step 4 |
| **Total** | **108** | The model input |

```python
input_df = pd.DataFrame([{**trend_features, **latest_features}])    # (1, 18)
final_df = pd.concat([input_df, tfidf_df], axis=1).fillna(0)        # (1, 108)
for col in feature_cols:                                            # add any missing column as 0
    if col not in final_df.columns:
        final_df[col] = 0
final_df = final_df[list(feature_cols)].astype(float)               # force the exact training order
```

**Why alignment matters.** The network's first layer has one weight column per position: position 0 is `HR_mean`, position 17 is `stress_score`, position 18 is the first TF-IDF word (`acute`) and position 107 the last (`wbc`). If columns were shuffled, the scaler would apply `HR_mean`'s statistics to another feature and the network would apply its learned weights to the wrong variable, giving a meaningless prediction. `feature_columns.pkl` stores the exact order used in training (the same as `scaler.feature_names_in_`).

(Made-up illustration of the failure: if position 0 held a TF-IDF value of 0.42, scaling it as a heart rate with mean 83.3 and spread 15.6 would give −5.3, a value the model never saw for heart rate.)

This is **early multimodal fusion**: vitals, camera values and text enter one flat vector, so the network can learn cross-modal patterns such as "high HR_mean *and* 'vasopressor' in the note → especially high SOFA".

---

## 9. Step 6 — StandardScaler Normalisation

**Purpose:** put all 108 numbers on a common scale so each is learned from fairly.

```python
final_scaled = np.clip(scaler.transform(final_df), -10, 10)
```
Each input becomes `(value − training mean) / training spread`, then is clipped to −10…+10.

**Why scale.** The raw inputs differ hugely in size (heart rate ≈ 100, TF-IDF ≈ 0.3). Without scaling the network's first layer would be dominated by the big numbers and almost ignore the words. After scaling, every input has average 0 and spread 1 across the training set.

**Real values for Patient 3** (training mean and spread come from `scaler.pkl`):

| Feature | Raw | Training mean | Training spread | Scaled (z) |
|---|---|---|---|---|
| HR_mean | 123.3 | 83.3 | 15.62 | **+2.56** |
| latest_HR | 124.0 | 82.2 | 17.61 | +2.37 |
| SpO2_mean | 85.67 | 91.08 | 21.42 | −0.25 |
| SpO2_min | 84.5 | 89.5 | 21.49 | −0.23 |
| latest_SpO2 | 85.0 | 88.67 | 25.45 | −0.14 |
| MAP_mean | 61.3 | 76.07 | 18.46 | −0.80 |
| latest_MAP | 61.0 | 72.56 | 24.54 | −0.47 |
| GCS_eye_opening | 2.0 | 3.61 | 1.01 | **−1.61** |
| stress_score | 8.0 | 1.29 | 2.34 | **+2.86** |

**Why some training spreads look large** (for example SpO₂ ≈ 21 and temperature ≈ 8): the training data stored missing readings as 0 (`fillna(0)`), which widens the spread of those inputs. The model learned with exactly these statistics, so inference uses the same ones.

**Why clip at ±10.** A very rare word can give an extreme scaled value (made-up illustration: raw 1.5, training mean 0.05, spread 0.12 → z ≈ 12). Unclipped, such a value could blow up through the network (an early experiment predicted a SOFA of 10,001). Clipping keeps "this word is strongly present" without the explosion.

**Critical rule:** the scaler is fitted on the **training set only** and never updated at inference. Using the training mean and spread keeps new patients on the same reference scale the model learned with.

---

## 10. Step 7 — Network Inference (108 → SOFA)

**Purpose:** run the 108 scaled numbers through the trained network.

### The architecture
```
Input:           108 numbers
Hidden layer 1:  256 neurons + ReLU + Dropout(0.3)
Hidden layer 2:  128 neurons + ReLU + Dropout(0.3)
Hidden layer 3:   64 neurons + ReLU + Dropout(0.3)
Output:            1 neuron  (no activation — a raw number)

Total trainable parameters: 69,121
```
**Dropout (0.3)** randomly switches off 30% of a layer's neurons during *training* so the network learns patterns that generalise instead of memorising patients. At prediction time it is **off**, so every neuron is used and the same patient always gets the same score.

### The forward pass
```python
model.eval()                  # prediction mode: dropout off
with torch.no_grad():         # no gradient tracking: faster, and no accidental accumulation
    X_tensor = torch.tensor(final_scaled, dtype=torch.float32)
    raw_pred = model(X_tensor).numpy().flatten()[0]
```

What each layer does, in words:
- **Layer 1 (108 → 256)** detects simple patterns: "low oxygen", "fast heart rate", "vasopressor mentioned".
- **Layer 2 (256 → 128)** combines them into syndromes: "septic respiratory failure", "kidney stress".
- **Layer 3 (128 → 64)** compresses them into severity signals.
- **Output (64 → 1)** is a weighted sum of the 64 signals: the predicted SOFA.

**Result for Patient 3: raw prediction ≈ 8.2.**

`model.eval()` matters because dropout would otherwise make predictions jump around. `torch.no_grad()` matters because gradients are only needed for training.

---

## 11. Step 8 — SOFA Post-processing, Risk Level and Alert

### Clipping
```python
sofa_score = float(np.clip(raw_pred, 0, 24))
```
The network can in principle output −0.3 or 25.8; clipping forces the valid SOFA range. (Illustration: −0.31 → 0.0, 25.8 → 24.0, 8.2 stays 8.2.) On the full test set the raw outputs ranged 1.78 to 23.07.

### Risk level (standard clinical bands)

| Predicted SOFA | Risk | Colour |
|---|---|---|
| below 5 | Low Risk | green |
| 5 to below 10 | Moderate Risk | amber |
| 10 or more | High Risk | red |

Patient 3: **8.2 → Moderate Risk (amber).**

### The alert — why threshold 7 and not 10
`is_alert = sofa_score >= ALERT_THRESHOLD`, where `ALERT_THRESHOLD = 7.0` is defined once in `model_utils.py`; the backend imports it, and the training metrics use the same value, so the dashboard and the reports always agree.

**Why not wait for 10?** The model **under-predicts severe patients**. For truly High Risk stays (true SOFA ≥ 10) the average true SOFA is about 13.1, but the average prediction is only about 9.5, because most of SOFA comes from lab values and drug doses that are not inputs. An alert at 10 would catch only 43.9% of High Risk patients.

**Held-out test set: 10,450 ICU stays, 602 truly High Risk:**

| Alert when predicted SOFA ≥ | Recall | Specificity | Missed | False alarms |
|---|---|---|---|---|
| 10 | 43.9% | 99.3% | 338 | 73 |
| 8 (earlier setting) | 59.6% | 97.9% | 243 | 207 |
| **7 (current)** | **67.8%** | **96.6%** | **194** | **335** |
| 6 | 73.1% | 94.1% | 162 | 578 |

At 7: precision 54.9%, AUROC 0.918, PR-AUC 0.647. Moving from 8 to 7 catches 49 more High Risk patients for 128 more false alarms; going below 7 adds false alarms quickly for small gains. **Even at 7, 194 of 602 High Risk stays (32%) get no alert**, so ClinIQ supports clinicians and does not replace them. Full discussion: `docs/VALIDATION.md`, Section 3.6.

Patient 3: 8.2 ≥ 7 → **the alert fires** (`is_alert = true`), although the band is Moderate. This is intentional: the band uses the standard clinical cut-offs, the alert is an early warning calibrated separately.

**What the alert controls:** the `is_alert` flag and the urgent "Immediate Actions first" format of the AI report. (The red/amber **banner** at the top of the page is driven by the risk *level*: High → red "CRITICAL ALERT", Moderate → amber "ELEVATED RISK", Low → none.)

### Saving the prediction
Every prediction is appended to the `prediction_history` table (timestamp, reading number, SOFA, risk, the 7 vitals, GCS eye, stress). The newest **50** rows per patient are kept; older ones are deleted. The History button on the dashboard shows them.

---

## 12. Step 9 — Runtime Safety Checks and the Conformal Range

**In one line:** after the prediction, the system asks "is this number even plausible?" and attaches an honest error bar. These checks **flag** a problem; they never change the predicted number. Full explanation: `docs/VALIDATION.md`, Sections 4–5.

### Check 1 — SOFA floor
Some vitals guarantee a minimum SOFA. The floor adds: MAP < 70 → +1, MAP < 65 → +1 more, SpO₂ < 94 → +1, SpO₂ < 90 → +1 more, GCS eye = 2 → +2 (or = 1 → +3).

Patient 3: MAP 61 (+2), SpO₂ 85 (+2), GCS eye 2 (+2) → **floor = 6**. The prediction 8.2 is above 6, so it is plausible. A prediction below the floor would be flagged as "may be underestimated".

### Check 2 — Dangerous vital combinations
- HR > 130 **and** SBP < 90 **and** SOFA < 4 → "implies shock state — SOFA expected ≥ 4".
- SpO₂ < 88 **and** MAP < 65 **and** SOFA < 6 → "implies multi-organ stress — SOFA expected ≥ 6".

Patient 3 (HR 124, SBP 84, SOFA 8.2): neither rule fires.

### Check 3 — Unexplained jumps
If SOFA moved by more than 4 points since the previous reading **and** no vital changed meaningfully (HR by 20, RR by 4, SpO₂ by 5, SBP by 20, MAP by 15), flag it. The first reading of a patient is never flagged. Patient 3: jump 0.0, not flagged.

### The conformal range (90% uncertainty)
A single number hides how sure the model is. The system shows a range:
```
range = prediction ± q̂,  kept within 0–24
q̂ = the 90th percentile of absolute errors over the 41,798 training stays = 3.28
```
Patient 3: 8.2 ± 3.28 → **5.0 to 11.5**. On the dashboard this is the "prediction interval" bar under the SOFA gauge. If the training data is unavailable, the code falls back to q̂ = 2.9. The confidence shown is `1 − q̂/12 = 0.727`. Because q̂ is measured on training data, real coverage on new patients is probably a little under 90% (stated in `docs/VALIDATION.md`).

---

## 13. Step 10 — SHAP Explainability

**Purpose:** show which inputs pushed this patient's score up or down, in plain language.

### Why it is needed
A SOFA of 8.2 says "deteriorating" but not *why*: lungs, kidneys, circulation, all three? Without SHAP the number is a black box.

### What SHAP computes
SHAP splits the prediction into one contribution per input, from game theory (fairly dividing a prize among players). The contributions are **additive and complete**:
```
sum of all 108 SHAP values = predicted SOFA − baseline
baseline = the model's average prediction over the 300 background patients = 3.85
Patient 3: 8.2 − 3.85 = +4.35 points to be explained
```
Positive = pushed the score up (worse); negative = pushed it down.

### The background set
`models/shap_background.npy` has shape (300, 108): 300 typical real training patients. SHAP compares this patient against them, which answers "how does this patient differ from typical real ICU patients?" A vector of zeros would describe an impossible patient (heart rate 0), so real patients are used. The same 300 are used every time, so explanations are comparable between patients.

### The computation
```python
explainer = shap.DeepExplainer(model, background_tensor)   # built once, cached
shap_vals = explainer.shap_values(X_tensor)                # per prediction (model in eval mode)
```
`DeepExplainer` attributes the output to each input by backpropagation (DeepLIFT, related to Integrated Gradients).

### From 108 values to something readable
1. Sort all 108 by absolute effect.
2. Keep clinically meaningful inputs by keyword (vitals, GCS, stress and words such as hypotension, respiratory, septic, failure, intubated, vasopressor, shock, fever, infection, oxygen, ventilat, cardiac, renal, hepatic). Short abbreviations like "hr" must match as a whole token so "hr" inside "epinephrine" is not taken as heart rate. (If fewer than 3 survive, the plain top 7 is used.)
3. Take the **top 7** for plain-English explanations and key risks; send the **top 5** to the dashboard.

**Real top 5 for Patient 3:**

| Feature | Original value | SHAP effect | Reading |
|---|---|---|---|
| GCS_eye_opening | 2 | **+0.94** | Eyes open only to pain → raises SOFA |
| "septic" (note word) | 0.25 | +0.91 | Note mentions sepsis → raises |
| "ventilator" (note word) | 0.19 | +0.69 | Note mentions ventilator → raises |
| HR_mean | 123.3 bpm | +0.62 | Sustained fast heart rate → raises |
| stress_score | 8 | **−0.45** | High stress → *lowers* the score (see below) |

**Plain-English explanations produced for the AI report (real):**
- "GCS Eye Opening = 2/4 → impaired consciousness, increasing risk"
- 'Clinical note contains "septic" → increasing risk'
- 'Clinical note contains "ventilator" → increasing risk'
- "HR mean = 123.3bpm → elevated heart rate, increasing risk"
- "Stress Score = 8/10 → high physiological stress, reducing risk"
- "Heart Rate = 124.0bpm → elevated heart rate, reducing risk"
- "RR mean = 29.4br/min → elevated breathing rate, increasing risk"

**Key risks (real):** Neurological deterioration, Sepsis / Infection, Abnormal heart rate, High physiological stress, Respiratory distress. (Words are matched to this list; for example "septic" gives "Sepsis / Infection".)

**Display:** the dashboard shows the **original** values (84.0%, 62 mmHg), never the scaled z-scores, and for note words it shows only the word, not the meaningless TF-IDF number.

### Two counter-intuitive results (and why they are not bugs)
SHAP explains what the **model learned**, not what medical theory says.
- **Stress *lowers* the score.** In MIMIC-III, stressed and responsive patients often had lower SOFA than unresponsive, sedated ones (a coma shows no stress). The model learned that pattern.
- **A high latest heart rate can show "reducing risk"** while the heart-rate *mean* shows "increasing": the two inputs overlap, and SHAP splits the credit between them in ways that can have opposite signs. Read the explanations as a set, not individually.

---

## 14. Step 11 — Trend Text

**Purpose:** classify each vital's 20-reading trend for the AI report. It is **not** a second input to the network.

```
For each of the 7 vitals:
  1. Fit a straight line through the 20 readings (numpy.polyfit); take the slope.
       slope >  +0.1 → "increasing"
       slope <  −0.1 → "decreasing"
       otherwise     → "stable"            (NaN values are ignored)
  2. Compare the window mean with the normal range → "low" / "normal" / "high".
  3. Write "[vital] → [status] & [direction]".
```
(The normal ranges used here: HR 60–100, RR 12–20, SpO₂ 95–100, temperature 36.5–37.5, SBP 100–120, DBP 60–80, MAP 70–100.)

**Real output for Patient 3:**
```
- HR → high & stable
- RR → high & stable
- SpO₂ → low & stable
- Temperature → high & stable
- SBP → low & stable
- DBP → low & stable
- MAP → low & stable
```
(For Patient 1 the same step gave "normal" for every vital and "decreasing" for HR, RR and the pressures.)

**Why a fitted line and not "first vs last"?** A comparison of two points is sensitive to one noisy reading. A line through all 20 points shows the average rate of change and is far more robust. The text goes into the AI prompt so the AI can say "SpO₂ is persistently low despite ventilation".

---

## 15. Step 12 — The Response Sent to the Browser

`POST /api/predict` returns this JSON (the prompt-building fields are removed before it leaves the server; the full copy is kept **server-side** for the report):

| Field | Meaning | Patient 3 |
|---|---|---|
| `patient_id`, `cur_idx`, `total_rows` | Which reading | 3, 2, 30 |
| `sofa_score` | Predicted SOFA | 8.2 |
| `risk_level`, `risk_color`, `risk_icon` | Band | Moderate Risk, amber |
| `is_alert`, `alert_threshold` | Alert flag and line | true, 7.0 |
| `conformal_lo`, `conformal_hi`, `conformal_q` | 90% range | 5.0, 11.5, 3.28 |
| `confidence` | `1 − q̂/12` | 0.727 |
| `vitals` | The 7 vitals, GCS eye, stress | as in Section 1 |
| `vitals_status` | low / normal / high per vital | HR high, SpO₂ low, MAP low … |
| `clinical_note` | The note text | shown in the dashboard |
| `shap_top5` | Top 5 drivers (value, effect, direction, relative size) | Section 13 |
| `key_risks`, `clinical_explanations` | Plain-English risks and sentences | Section 13 |
| `trend_text`, `vitals_history` | Trend summary and the 20-reading chart data | Section 14 |
| `_prev_sofa`, `_sofa_jump`, `_traj_flagged`, `_traj_msg` | Runtime check 3 | none / 0.0 / false |
| `_training_meta` | Contents of `training_metadata.json` | read by the SOFA validation drawer |

Removed before sending to the browser: `_vitals_block`, `_cv_block`, `_risk_block`, `_explanation_block` (prompt scaffolding).

---

## 16. Step 13 — The AI Report: Prompt Construction

The browser then asks for the report: `POST /api/report` with `{ "patient_id": 3, "care_mode": "hospital" }`. It sends **only** the patient ID and mode. The server uses its **own stored prediction** and never trusts data sent by the browser.

Errors: invalid patient or mode → 400; no prediction yet → **409**; missing Groq key → 500; any other failure → 500 with a generic message.

### When the report is requested
The frontend builds a `reportKey` from the patient, SOFA and every vital (HR, RR, SpO₂, SBP, DBP, MAP, temperature, GCS, stress). A new report is requested only when this key **changes** or the care mode is switched. Reports for older readings are discarded. After one mode's report is ready, the other mode is generated in the background so switching is instant. The backend cache key matches this key exactly.

### The prompt (what the AI receives)

**1. Urgency header (alert patients only).** If SOFA ≥ 7 the prompt starts with a clinical alert line, for example "⚠️ CLINICAL ALERT — Predicted SOFA 8.2 (alert threshold ≥ 7). This patient is actively deteriorating. Lead with immediate actions." (Home Care mode uses a plain "PATIENT ALERT" line.)

**2. Section instructions.** Four sections, written with the exact headings: Current Condition, Probable Cause, Risk Forecast, Immediate Actions. For an alert patient **Immediate Actions comes first**; otherwise it comes last. Each mode has its own rules:
- **Hospital mode:** bullet points; every drug has name, dose, route and target; every vital shows value and normal range; examples of good and bad bullets are included.
- **Home Care mode:** plain English for a home nurse and family; sub-headings "For the Nurse", "For the Family", "Home Remedies", "⚠️ Go to Hospital NOW if" (alert) ; both worsening and improving outcomes in the forecast; no unexplained jargon.

**3. SOFA and risk.** "SOFA Score: 8.2 / 24 (0–4 = Low Risk, 5–9 = Moderate Risk, ≥10 = High Risk …)", the risk level, and "Predicted SOFA uncertainty: ± 3.3 points (90% conformal interval)".

**4. Clinical note (verbatim)** from the patient chart.

**5. Latest vitals with normal ranges (real):**
```
HR: 124.0 bpm (normal: 60–100)
RR: 30.0 breaths/min (normal: 12–20)
SpO₂: 85.0% (normal: 95–100)
Temperature: 39.2 °C (normal: 36.5–37.5)
Blood Pressure: 84.0/49.0 mmHg (normal: ~120/80)
MAP: 61.0 mmHg (normal: 70–100)
```
The ranges are included so the AI does not have to guess which values are abnormal.

**6. Neurological and stress values (real):**
```
GCS Eye Opening: 2 (1=No response, 2=To pain, 3=To voice, 4=Spontaneous)
Stress Score: 8.0/10 (higher = more distress)
```

**7. SHAP key risks, SHAP explanations (Hospital mode) and the vital trends** from Sections 13–14.

**Why SHAP is in the prompt.** Without it the AI sees only raw numbers and the note (what any physician sees) and writes a generic septic-shock report. With it, the AI knows which inputs the predictive model found most important for *this* patient and can say, for example, that impaired consciousness, sepsis and ventilation drive the score.

**Closing instruction:** "Base all clinical reasoning strictly on the data above. Do not invent values or assume findings not present in the data."

### The AI call
| Setting | Writer | Judge |
|---|---|---|
| Model | `openai/gpt-oss-120b` (Groq) | the same model |
| Temperature | 0.15 (factual, consistent) | 0.0 |
| Max tokens | 6,000 | 1,200 |
| Limit | — | 12 seconds; sees the first 900 characters of the report |

Any `<think>…</think>` text is stripped from the reply, and the reply is split into the four sections (tolerating headings written with `#`, bold, numbers or dashes). All four keys always exist, empty if the AI omitted one.

*(An earlier version asked the AI the same question three times and compared the answers; that check was removed because agreement measures consistency, not correctness. See Section 17.)*

---

## 17. Step 14 — The AI Report: Reliability Score

**Purpose:** give a number for how much to trust the report, by checking it against the **real patient data**, not against itself. Full explanation with worked examples: `docs/VALIDATION.md`, Part 2.

### Nine methods

| # | Method | Weight | What it checks |
|---|---|---|---|
| 1 | Factual grounding | 13% | Every abnormal vital is mentioned (by related words) |
| 2 | Clinical hard rules | 13% | The standard treatment is mentioned (6 rules, for example SpO₂ < 90 → oxygen therapy) |
| 3 | SHAP–LLM coherence | 9% | The report discusses what actually drove the prediction |
| 4 | Response structure | 9% | All four sections present and enough length (≥ 200 words counts as full) |
| 5 | Severity calibration | 8% | Tone matches severity (SOFA ≥ 10 needs urgent wording; a low-risk patient should not get panic wording) |
| 6 | Contraindication safety | 5% | No dangerous drug for these vitals (3 rules: beta-blockers in shock, opioids/benzodiazepines in severe hypoxia, NSAIDs) |
| 7 | Numeric accuracy | 8% | Every quoted patient number is close to a real value (within 10% or 3 units) or a standard reference number |
| 8 | G-Eval (AI judge) | 13% | Overall quality: 4 aspects scored 1–5 |
| 9 | RAGAS faithfulness (AI judge) | 22% | Lists unsupported claims; each lowers the score |

Methods 1–7 are instant rule checks (65% of the weight); 8–9 use the judge (35%). If the judge is late or fails, methods 8 and 9 score a neutral 0.5 and the report is still shown. If the judge flags a dangerous recommendation, method 6 is forced to 0.

### Labels
| Score | Label |
|---|---|
| ≥ 0.75 | **High** reliability (green) |
| 0.55 – 0.75 | **Moderate** (amber) |
| < 0.55 | **Low** (red) |

A separate **caution** flag is set below 0.60. Every specific problem found (broken rules, dangerous drugs, wrong numbers, unsupported claims) is returned in `violations` for the clinician.

### The report response
```json
{
  "sections": { "IMMEDIATE ACTIONS": "…", "CURRENT CONDITION": "…",
                "PROBABLE CAUSE": "…", "RISK FORECAST": "…" },
  "raw_response": "full text",
  "reliability_score": 0.89, "reliability_label": "High", "reliability_color": "#28a745",
  "caution": false,
  "violations": [],
  "care_mode": "hospital",
  "judge_used": true,
  "error": null
}
```
(The score and label shown are an illustration of the format; each report gets its own.)

### Honest limits
The same model writes and grades the report, and the judge sees only 900 characters, so the seven rule checks are the independent part of the score. Keyword checks look for words, not meaning.

---

## 18. Step 15 — What the Clinician Sees

The dashboard is **one scrolling page** (no tabs), plus four floating buttons.

```
┌───────────────────────────────────────────────────────────┐
│ HEADER: patient · live status · countdown to next reading │
├───────────────────────────────────────────────────────────┤
│ ALERT BANNER (Moderate: amber "ELEVATED RISK";            │
│               High: red "CRITICAL ALERT"; Low: none)      │
├───────────────────────────────────────────────────────────┤
│ CURRENT STATUS                                            │
│  SOFA gauge: 8.2 / 24  [Moderate Risk]                    │
│    prediction interval 5.0 – 11.5, confidence 73%         │
│  Vital cards: HR · SpO₂ · RR · Temp · SBP · MAP · DBP ·   │
│    GCS eye · Stress   (colour-coded low / normal / high)  │
├───────────────────────────────────────────────────────────┤
│ VITAL TRENDS: 7 line charts (last 20 readings) with       │
│    normal-range bands                                     │
├───────────────────────────────────────────────────────────┤
│ AI CLINICAL REPORT                                        │
│  [Hospital | Home Care] toggle                            │
│  4 section cards (Immediate Actions first for alerts)     │
│  SHAP bars: top 5 drivers, original values                │
│  Reliability badge (High / Moderate / Low)                │
│  Disclaimer: decision support only                        │
└───────────────────────────────────────────────────────────┘
Floating buttons (right edge): History · Camera monitor ·
                               LLM validation · SOFA validation
```

- **History drawer:** table of recent predictions (time, SOFA, risk, vitals, GCS, stress).
- **Camera monitor:** live webcam stress and eye scores (Section 2).
- **LLM validation drawer:** the nine-method breakdown and every violation.
- **SOFA validation drawer ("Model Accuracy Report"):** reads every number **live** from `training_metadata.json` (the `_training_meta` field): AUROC, PR-AUC, MAE, R², alert recall, specificity, missed and false-alarm counts, the 3-band confusion matrix and its accuracy, and the sample counts. Also shows the three runtime checks and the conformal range. If the metadata is missing it shows "—". The final run's values: MAE 1.80, R² 0.48, AUROC 0.918, PR-AUC 0.647, alert recall 67.8%, specificity 96.6%, 10,450 test stays.
- **Dark and light themes** are available.

The SHAP bars, the vital cards and the gauge all show original clinical values, not model-internal numbers, so the clinician sees familiar units.

---

## 19. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  EVERY 90 SECONDS: browser → POST /api/predict                  │
│  { patient_id, row_idx, [cv_gcs_eye], [cv_stress] }             │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
                     ┌────────────────────┐
                     │ INPUT CHECKS       │  patient 1–3, GCS 1–4, stress 0–10
                     └─────────┬──────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │ READ SAMPLE ROW (sample_patients, looped)    │
        │  7 vitals · GCS · stress · clinical note     │
        │  live camera values override GCS / stress    │
        └───────┬───────────────────────────┬──────────┘
                ▼                           ▼
      ┌──────────────────┐        ┌───────────────────┐
      │ SLIDING WINDOW   │        │ TF-IDF            │
      │ append, keep 20  │        │ 90-word SOFA list │
      │ (patient_vitals) │        │ → 90 numbers      │
      └────────┬─────────┘        └─────────┬─────────┘
               ▼                            │
      ┌──────────────────┐                  │
      │ TREND FEATURES 9 │  + LATEST 9      │
      └────────┬─────────┘  (7 vitals+2 cam)│
               └─────────────┬──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ FUSION 9+9+90   │  column-aligned to feature_columns.pkl
                    │ = 108           │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ STANDARDSCALER  │  z-score, clip ±10
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ FEDERATED DNN   │  108→256→128→64→1, eval mode
                    └────────┬────────┘
                             ▼
              ┌───────────────────────────────┐
              │ SOFA = clip(raw, 0, 24)       │
              │ risk band · is_alert (≥ 7)    │
              │ conformal range ± 3.28        │
              │ 3 runtime checks              │
              │ save to prediction_history    │
              └───────┬───────────────┬───────┘
                      ▼               ▼
              ┌──────────────┐ ┌───────────────┐
              │ SHAP         │ │ TREND TEXT    │
              │ top 7 → 5    │ │ slope per     │
              │ explanations │ │ vital         │
              └──────┬───────┘ └──────┬────────┘
                     └───────┬────────┘
                             ▼
          server stores full result · browser gets cleaned JSON
                             │
        (frontend: reportKey changed or mode switched?)
                             ▼
                POST /api/report { patient_id, care_mode }
                             │
                             ▼
              ┌────────────────────────────┐
              │ 90 s cache? → return it    │
              └──────────────┬─────────────┘
                             ▼
              ┌────────────────────────────┐
              │ PROMPT (alert-first if ≥ 7)│
              │ Groq gpt-oss-120b, t=0.15  │
              └──────────────┬─────────────┘
                             ▼
              ┌────────────────────────────┐
              │ 7 RULE CHECKS (instant)    │
              │ + AI JUDGE (≤ 12 s)        │
              │ → 9-method reliability     │
              └──────────────┬─────────────┘
                             ▼
              ┌────────────────────────────┐
              │ DASHBOARD: gauge, vitals,  │
              │ trends, report, SHAP,      │
              │ reliability, drawers       │
              └────────────────────────────┘
```

---

## 20. Artifacts Loaded at Startup

Loaded once (on the first request) and cached for the life of the backend:

| Artifact | How loaded | What it provides |
|---|---|---|
| `models/federated_model.pth` | `torch.load(..., weights_only=True)` | The network weights (108→256→128→64→1); set to `eval()` |
| `models/scaler.pkl` | `joblib.load()` | Training mean and spread for the 108 inputs |
| `models/tfidf_vectorizer.pkl` | `joblib.load()` | The 90 words and their IDF weights |
| `models/feature_columns.pkl` | `joblib.load()` | The 108 input names in training order (falls back to `scaler.feature_names_in_`) |
| `models/shap_background.npy` | `np.load()` | 300 background patients; the SHAP explainer is built from it on first use |
| `models/training_metadata.json` | `json.load()` (each prediction) | Training settings and test results for the validation drawer |

**Computed once at first use:** the conformal q̂ (3.28), from the training rows in the database.

**Read and written on every prediction (database, not cached):**

| Table | Read / write | Purpose |
|---|---|---|
| `sample_patients` | Read | The recorded reading to replay |
| `patient_vitals` | Read + write | The 20-reading sliding window |
| `prediction_history` | Read + write | Log of the newest 50 predictions per patient |

The tables are created at backend start if missing (`init_db()`), so a fresh clone does not fail. A checkpoint trained with a different network size will not load; retrain instead.

---

## 21. Feature Count Summary

| Pipeline stage | Count | Format |
|---|---|---|
| Recorded reading — vital signs | 7 | Raw numbers (bpm, %, mmHg, °C) |
| Camera / observation values | 2 | GCS 1–4, stress 0–10 |
| Clinical note | 1 | Text |
| After the sliding window | 20 rows × 7 vitals | Temporal history |
| Trend features | 9 | Means, std, min over 20 readings |
| Latest features | 9 | 7 current vitals + GCS + stress |
| TF-IDF features | 90 | Word scores (13 non-zero for Patient 3) |
| **After fusion** | **108** | **One flat vector — the model input** |
| After StandardScaler | 108 | z-scores clipped to ±10 |
| Raw network output | 1 | Unrestricted number (8.2) |
| After clipping | 1 | SOFA in 0–24 |
| Conformal range | 2 | Lower and upper (5.0, 11.5) |
| SHAP values | 108 | Sum = prediction − baseline (4.35) |
| SHAP shown | 5 (7 used for text) | Top clinically meaningful drivers |
| Trend strings | 7 | One per vital (for the AI prompt only) |
| AI report | 1 (4 sections) | Hospital or Home Care |
| Reliability | 1 | Score 0–1 + label + violations |

---

## 22. Timing, Caching and Failure Behaviour

| Item | Behaviour |
|---|---|
| Refresh | A new prediction every **90 seconds**; the countdown never goes below 1, and resets only after data arrives |
| Overlapping requests | Prevented (one fetch at a time); a response is discarded if the patient changed while it was in flight |
| Resume | The reading position is kept in the browser's session storage, so a page refresh continues from the same reading |
| Report cache | **90 seconds**; key = care mode + patient + SOFA + every vital in the prompt, so a changed reading never gets a report quoting old numbers; expired entries are cleaned on every write |
| Identical simultaneous requests | One AI call is made; the second request waits and reuses it |
| Both care modes | After one report is ready, the other mode is prepared in the background |
| Cost | 2 AI calls per report (write + judge); about 4 per new reading with both modes |
| Judge late or failing | Methods 8 and 9 score 0.5; the report is still shown with `judge_used: false` |
| AI unavailable | The dashboard keeps showing the last report with a notice |
| Prediction failure | The dashboard keeps the last reading, shows "Refresh failed… Retrying in 90s." |
| Model checkpoint mismatch | The backend fails to start the model load; retrain so the weights match the 108→256→128→64→1 network |

---

*Documentation for the ClinIQ ICU CDSS capstone project — last updated 2026-11-24.*
*Pipeline: 7 vitals + 2 camera/observation values + clinical text → 108 features → SOFA → conformal range, 3 runtime checks, SHAP → AI report → 9-method reliability score.*
*Model: PyTorch network 108→256→128→64→1 (Dropout 0.3, 69,121 parameters), trained with Flower federated learning (FedAvg + FedProx, plain MSE); final run MAE 1.80, R² 0.48, AUROC 0.918.*
*All worked examples use real outputs for demo Patient 3 (reading 3 of 30). When the model is retrained, re-run a prediction and refresh Sections 7, 9, 12, 13 and 18.*

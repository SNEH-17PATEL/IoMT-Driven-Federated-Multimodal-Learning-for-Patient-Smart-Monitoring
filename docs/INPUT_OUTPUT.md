# Complete Input-Output Lifecycle — ICU CDSS
**Multimodal Intelligence System with Federated Learning for ICU Patient Monitoring**
**Status:** Fully implemented and verified

---

## What This Document Covers

This document traces the complete journey of a single prediction — from the moment a clinician enters data, through every computation, transformation, and model inference step, all the way to the final output displayed on the dashboard.

Every step includes:
- What the input is and where it comes from
- Exactly what computation is performed
- What the output looks like (with real example values)
- Why this step exists and what would go wrong without it

---

## Table of Contents

1. [The Big Picture — What the System Does](#1-the-big-picture--what-the-system-does)
2. [System Inputs — Five Modalities](#2-system-inputs--five-modalities)
3. [Input Validation — The Safety Gate](#3-input-validation--the-safety-gate)
4. [Step 1 — Sliding Window (Temporal Buffer)](#4-step-1--sliding-window-temporal-buffer)
5. [Step 2 — Trend Feature Engineering (9 features)](#5-step-2--trend-feature-engineering-9-features)
6. [Step 3 — Latest Vital and CV Features (9 features)](#6-step-3--latest-vital-and-cv-features-9-features)
7. [Step 4 — TF-IDF Text Encoding (90 features)](#7-step-4--tfidf-text-encoding-90-features)
8. [Step 5 — Feature Fusion and Alignment (108 features)](#8-step-5--feature-fusion-and-alignment-108-features)
9. [Step 6 — StandardScaler Normalisation](#9-step-6--standardscaler-normalisation)
10. [Step 7 — DNN Inference (108 → SOFA score)](#10-step-7--dnn-inference-108--sofa-score)
11. [Step 8 — SOFA Post-processing and Alert Classification](#11-step-8--sofa-post-processing-and-alert-classification)
12. [Step 9 — SHAP Explainability](#12-step-9--shap-explainability)
13. [Step 10 — Trend Analysis](#13-step-10--trend-analysis)
14. [Step 11 — LLM Prompt Construction](#14-step-11--llm-prompt-construction)
15. [Step 12 — LLM Self-Consistency Check](#15-step-12--llm-self-consistency-check)
16. [Step 13 — Output Display (Four Tabs)](#16-step-13--output-display-four-tabs)
17. [Complete Data Flow Diagram](#17-complete-data-flow-diagram)
18. [Artifacts Loaded at Startup](#18-artifacts-loaded-at-startup)
19. [Feature Count Summary](#19-feature-count-summary)

---

## 1. The Big Picture — What the System Does

Before tracing the technical details, it helps to understand the problem being solved.

A clinician is monitoring a patient in the ICU. They enter the patient's current vital signs, a brief clinical note, and two scores from the computer vision module. Within a few seconds, the system:

1. **Predicts** a SOFA score (0–24) — a clinically validated measure of organ failure severity
2. **Explains** which factors drove that score up or down, using SHAP attribution values
3. **Generates** a structured clinical report using a large language model (LLM)
4. **Verifies** the LLM report's reliability by running it three times and measuring consistency

The system is not replacing clinical judgment. It is giving the clinician a second opinion in real time — grounded in patterns learned from 48,150 MIMIC-III ICU patient records, with clear explanations of its reasoning.

**Running example throughout this document:**

To make every step concrete, we will trace a single patient through the entire pipeline:

```
Patient: 67-year-old male, septic shock
  HR:    122 bpm      (normal: 60–100)
  RR:    30 br/min    (normal: 12–20)
  SpO₂:  84%          (normal: 95–100)
  Temp:  39.1°C       (normal: 36.5–37.5)
  SBP:   85 mmHg      (normal: 100–120)
  DBP:   50 mmHg      (normal: 60–80)
  MAP:   62 mmHg      (normal: 70–100)
  GCS Eye Opening: 2  (eyes open only to pain)
  Stress Score: 8.0   (significant visible distress)
  
Clinical note: "67-year-old male. SpO2 declining despite high-flow O2 therapy.
               History of COPD and hypertension. Creatinine elevated.
               Urine output low. On vasopressors."
               
Expected SOFA: High (multiple organ systems failing)
```

---

## 2. System Inputs — Five Modalities

The system accepts **five independent categories** of input. Three come directly from the clinician at the time of each assessment. Two are pre-loaded from files saved on disk.

### Modality 1 — Real-time Vital Signs (7 numbers)

These are the standard bedside monitoring values entered for the current assessment:

| Input | Unit | What it measures | Normal range | Valid range | Our example |
|---|---|---|---|---|---|
| HR | bpm | Heart rate | 60–100 | 30–220 | 122.0 |
| RR | br/min | Respiratory rate | 12–20 | 5–60 | 30.0 |
| SpO₂ | % | Blood oxygen saturation | 95–100 | 50–100 | 84.0 |
| Temp | °C | Body temperature | 36.5–37.5 | 30–43 | 39.1 |
| SBP | mmHg | Systolic blood pressure | 100–120 | 40–250 | 85.0 |
| DBP | mmHg | Diastolic blood pressure | 60–80 | 20–150 | 50.0 |
| MAP | mmHg | Mean arterial pressure | 70–100 | 30–200 | 62.0 |

**What these vitals cover in the SOFA score:**
- SpO₂ readings → proxy for SOFA Component 1 (Respiratory)
- MAP readings → direct SOFA Component 4 (Cardiovascular)
- GCS Eye Opening (next section) → SOFA Component 5 (CNS)

The remaining three SOFA components (Coagulation, Hepatic, Renal) require lab values not available at the bedside — these are captured through the clinical notes text.

**The difference between "Normal Range" and "Valid Range":**

The valid range is the physiologically plausible boundary — values outside it indicate a data entry error or equipment malfunction (HR=10 bpm or SpO₂=110% cannot be real). The normal range is the clinical reference — values outside it are concerning but medically real. Both ranges are checked, but only the valid range causes the system to halt. Values outside the normal range trigger coloured indicators on the dashboard.

---

### Modality 2 — Computer Vision Features (2 numbers)

| Input | Scale | What it measures | Our example |
|---|---|---|---|
| GCS Eye Opening | 1–4 | Neurological consciousness level assessed from eye behaviour | 2 |
| Stress Score | 0–10 | Pain and agitation level assessed from facial expression | 8.0 |

**What these two numbers mean clinically:**

**GCS Eye Opening (E1–E4):**
| Score | Label | Meaning | Brain state implication |
|---|---|---|---|
| 4 | Spontaneous | Eyes open naturally without any stimulus | Brainstem arousal intact — patient alert |
| 3 | To Voice | Eyes open when spoken to | Partial arousal — patient responds to external stimulation |
| 2 | To Pain | Eyes open only to painful stimulus | Minimal arousal — only strongest stimuli penetrate |
| 1 | No Response | Eyes remain closed regardless | No functional eye-opening response |

Our patient scores E2 — eyes open only to pain. This indicates significant neurological impairment, which directly contributes to SOFA Component 5 (CNS).

**Stress Score (0.0–10.0):**
| Range | Label | Meaning |
|---|---|---|
| 0.0–3.4 | LOW | Patient appears calm and comfortable |
| 3.5–6.4 | MODERATE | Noticeable discomfort, needs monitoring |
| 6.5–10.0 | HIGH | Severe visible distress — likely acute pain or agitation |

Our patient scores 8.0 — HIGH stress. Combined with GCS E2, this paints a picture of a neurologically impaired but physiologically distressed patient.

**Why these are called "Computer Vision features":**

In the full system architecture, GCS Eye Opening and Stress Score are derived automatically from a camera feed using MediaPipe facial analysis — no nurse intervention needed. The camera runs continuously, the CV module computes these scores in real time, and they feed into predictions automatically. For this implementation, the clinician enters them manually. The rest of the pipeline treats them identically regardless of how they were obtained.

---

### Modality 3 — Clinical Notes (free text)

```
Example input (our running patient):
"67-year-old male. SpO2 declining despite high-flow O2 therapy.
 History of COPD and hypertension. Creatinine elevated. Urine output low.
 On vasopressors."
```

Clinical notes capture information that structured vital signs cannot:
- **Why** vitals are abnormal ("SpO2 declining DESPITE high-flow O2" — the resistance to treatment matters)
- **Lab values** not automatically monitored ("creatinine elevated" — this encodes renal SOFA component)
- **Interventions in place** ("on vasopressors" — this directly encodes cardiovascular SOFA component)
- **Patient history** ("history of COPD" — chronic disease context changes the interpretation of SpO2)

This free text is converted into 90 numerical TF-IDF features in Step 4.

---

### Modality 4 — Historical Vitals (from file)

The system reads `models/patient_vitals.csv`, which stores the last 20 readings of the 7 vital signs with timestamps. This file is how the system maintains a memory of the patient's trajectory.

```
Structure of patient_vitals.csv:
  timestamp           | HR  | RR | SpO2 | Temp | SBP | DBP | MAP
  2024-03-15 08:00:00 | 88  | 18 | 97   | 37.1 | 118 | 76  | 90
  2024-03-15 08:15:00 | 92  | 19 | 96   | 37.2 | 115 | 74  | 88
  2024-03-15 08:30:00 | 97  | 21 | 95   | 37.3 | 112 | 72  | 85
  ...
  2024-03-15 11:45:00 | 118 | 28 | 86   | 38.9 | 88  | 54  | 65
  (most recent reading — the patient is deteriorating)
```

**What "Reset Patient History" does:**

When the clinician presses "Reset Patient History," this file is replaced with 20 default stable readings. This **must** be pressed when switching to a new patient — otherwise the sliding window would mix the previous patient's vital history with the new patient's data, producing nonsensical trend statistics.

---

### Modality 5 — Pre-trained Model Artifacts (loaded from disk at startup)

These are not entered by the clinician. They are the products of the federated training run and are loaded once when the application starts:

| Artifact | What it contains | Why it is needed |
|---|---|---|
| `models/federated_model.pth` | Trained DNN weights (108→128→64→32→1) | The model that makes predictions |
| `models/scaler.pkl` | StandardScaler fitted on 48,150 training patients | Normalises new inputs to the same scale as training data |
| `models/tfidf_vectorizer.pkl` | TF-IDF with 90-term SOFA vocabulary whitelist | Converts clinical notes to numeric features |
| `models/feature_columns.pkl` | Ordered list of 108 column names | Ensures features are presented to the model in the correct order |
| `models/shap_background.npy` | 300 representative training patients | Reference distribution for SHAP feature attribution |

All 5 are loaded once and cached. They do not change between predictions — only the clinician-entered inputs change.

---

## 3. Input Validation — The Safety Gate

**Before any computation runs**, the system checks whether the entered vital signs are within physiologically plausible bounds.

```
For each of the 7 vital signs:
  Check: lo ≤ value ≤ hi   (using VALID_RANGES dictionary)

If ANY vital fails:
  → Show error message: "HR=10 bpm is outside the valid range [30, 220]"
  → Stop immediately — no prediction is made

If ALL vitals pass:
  → Continue to Step 1
```

**Why this check is essential:**

Without validation, a typo (HR=10 instead of HR=110) would propagate through the entire pipeline and produce a wildly wrong SOFA prediction. The model was trained on physiologically real data — if you feed it values outside that range, its behaviour is undefined.

```
Example of what happens without validation:
  Clinician accidentally enters SpO2 = 110 (valid SpO2 is 50–100)
  
  After StandardScaler: (110 - mean_spO2) / std_spO2
    ≈ (110 - 96) / 3 = +4.7 z-scores (already high)
  
  After clipping to ±10: 4.7 (not clipped)
  
  The DNN processes 4.7 for SpO2 — a value it has never seen in training
  → unpredictable, potentially wildly incorrect SOFA prediction
  → patient may receive wrong clinical advice

With validation: error message → prediction halted → clinician corrects the typo
```

Also checked: the Groq API key must be present (stored in the `.env` file). Without it, the LLM clinical report cannot be generated. This is checked upfront rather than failing after all the ML computation has already run.

---

## 4. Step 1 — Sliding Window (Temporal Buffer)

**Purpose:** Build and maintain a 20-reading history of vital signs to enable trend computation.

**Why a single reading is not enough:**

A single vital sign snapshot is ambiguous. SpO₂ = 91% could mean:
- The patient has been stably at 91% for 4 hours (concerning but stable)
- The patient was at 97% an hour ago and is rapidly deteriorating (urgent emergency)
- The probe slipped off the finger and gave one bad reading (false alarm)

These three scenarios have completely different clinical responses, but they are indistinguishable from a single reading. Twenty readings across time tell the trajectory — the direction of change — which is far more informative.

**The process:**

```
Input:
  Current reading: {HR: 122, RR: 30, SpO2: 84, Temp: 39.1, SBP: 85, DBP: 50, MAP: 62}
  Historical file: patient_vitals.csv (up to 20 previous rows)

Step A: Load patient_vitals.csv
  → DataFrame with up to 20 rows, 8 columns (timestamp + 7 vitals)

Step B: Append the current reading as a new row
  → DataFrame now has up to 21 rows

Step C: Keep only the last 20 rows
  new_df = df.tail(20)
  → If fewer than 20 readings exist (e.g., new patient with only 3 readings):
    use however many are available
  → If more than 20 exist: oldest reading is dropped

Step D: Save the updated DataFrame back to patient_vitals.csv

Output: vitals_df — DataFrame of shape (≤20, 8)
```

**A concrete timeline example — showing the window moving forward:**

```
Time 08:00: Patient admitted. Only 1 reading (SpO2=97, HR=85, ...)
  Window: [reading_1]  (1 reading)
  
Time 08:15: Second reading entered (SpO2=96, HR=89)
  Window: [reading_1, reading_2]  (2 readings)
  
...continuing until...
  
Time 11:45: 20th reading (SpO2=86, HR=118)
  Window: [reading_1, ..., reading_20]  (20 readings, FULL)
  
Time 12:00: 21st reading (SpO2=84, HR=122) — current assessment
  Window: [reading_2, ..., reading_21]  (20 readings, reading_1 dropped)
  → reading_1 from 08:00 (SpO2=97) is no longer in the window
  → The trend computation now covers the last 3 hours, not the entire stay
```

**Why exactly 20 readings?**

The training data in MIMIC-III was also computed using 20-reading windows. The model learned what "normal trend" looks like across 20 readings. If we used 10 readings at inference but the model was trained on 20-reading statistics, the z-scores for trend features would be systematically different, leading to incorrect predictions. Matching the inference window to the training window ensures consistency.

---

## 5. Step 2 — Trend Feature Engineering (9 features)

**Purpose:** Summarise the patient's 20-reading trajectory into 9 statistical measures that capture both the current level and the direction of change.

```
Input:  vitals_df — DataFrame with ≤20 rows and 7 vital sign columns

Process: Compute 9 statistics across the rows:

  HR_mean    = mean(HR values across all 20 readings)
  HR_std     = standard deviation of HR values (measure of variability)
  RR_mean    = mean(RR values)
  SpO2_mean  = mean(SpO2 values)
  SpO2_min   = minimum(SpO2 values across all 20 readings)
  Temp_mean  = mean(Temp values)
  SBP_mean   = mean(SBP values)
  DBP_mean   = mean(DBP values)
  MAP_mean   = mean(MAP values)

Output: trend_features — dictionary of 9 float values
```

**Worked example using our running patient:**

```
The 20 readings in the window (simplified):
  Reading 1 (08:00):  HR=85,  SpO2=97, MAP=90
  Reading 2 (08:15):  HR=89,  SpO2=96, MAP=88
  Reading 3 (08:30):  HR=92,  SpO2=95, MAP=86
  Reading 4 (08:45):  HR=96,  SpO2=94, MAP=85
  ...
  Reading 15 (10:30): HR=108, SpO2=89, MAP=74
  ...
  Reading 19 (11:30): HR=118, SpO2=87, MAP=67
  Reading 20 (11:45): HR=118, SpO2=86, MAP=65
  (Reading 21 = 12:00, HR=122, SpO2=84, MAP=62 — current, included)

After computation:
  HR_mean    = average of 85, 89, 92, ..., 118, 122 = 101.5 bpm
  HR_std     = std of those values = 14.2 bpm  (large spread = unstable)
  RR_mean    = 24.1 br/min  (was 18, now 30 — elevated throughout)
  SpO2_mean  = mean of 97, 96, 95, ..., 86, 84 = 88.5%
  SpO2_min   = 84%  (the nadir — worst single desaturation)
  Temp_mean  = 38.6°C  (progressively rising from 37.1)
  SBP_mean   = 92.3 mmHg  (declining from 118)
  DBP_mean   = 57.1 mmHg
  MAP_mean   = 68.8 mmHg  (declining from 90)
```

**Why exactly these 9 features and not all 7×2=14 (mean + std for each vital)?**

Each feature was chosen based on clinical significance:

**HR_mean and HR_std — two features for heart rate:**
- HR_mean: The average heart rate over the window (is the patient tachycardic overall?)
- HR_std: Heart rate variability (HRV) is itself a clinical indicator. Low HRV (HR very stable at a high level) can indicate autonomic nervous system dysfunction in sepsis. High HRV (HR jumping up and down) can indicate cardiac instability. Standard deviation captures this.

```
Example: 
  Patient A: HR readings = 120, 121, 119, 120, 122 → HR_std = 1.1 (very stable tachycardia)
  Patient B: HR readings = 90, 130, 85, 125, 95   → HR_std = 21.3 (highly variable)
  
  Both have similar HR_mean (~115), but Patient B's high variability
  suggests a different physiological process. The model can learn to distinguish them.
```

**SpO2_mean and SpO2_min — two features for oxygen saturation:**
- SpO2_mean: The overall level of oxygenation across the window
- SpO2_min: The worst single value — even a brief desaturation can cause tissue damage

```
Example:
  Patient A: SpO2 readings = 95, 96, 94, 95, 70, 95, 96, 94  → mean=91.9%, min=70%
  Patient B: SpO2 readings = 91, 92, 91, 92, 92, 91, 91, 92  → mean=91.5%, min=91%
  
  Both have similar SpO2_mean, but Patient A had a dangerous 70% desaturation.
  SpO2_min captures this episode; SpO2_mean alone would hide it.
```

**RR_mean, Temp_mean, SBP_mean, DBP_mean, MAP_mean — means only:**
The variability of these vitals is less clinically significant as a standalone predictor compared to their absolute levels. Adding their standard deviations would double the feature count (from 9 to 16) without proportional gain in predictive power.

---

## 6. Step 3 — Latest Vital and CV Features (9 features)

**Purpose:** Include the most recent absolute values alongside the trend statistics, because trend summaries alone lose information about the current state.

```
Input:
  7 vital sign floats entered by the clinician (the CURRENT reading)
  GCS_eye_opening integer (1–4)
  stress_score float (0–10)

Process: Package as a dictionary

Output: latest_features — dictionary with 9 key-value pairs
  {
    "latest_HR":       122.0,
    "latest_RR":       30.0,
    "latest_SpO2":     84.0,
    "latest_Temp":     39.1,
    "latest_SBP":      85.0,
    "latest_DBP":      50.0,
    "latest_MAP":      62.0,
    "GCS_eye_opening": 2.0,
    "stress_score":    8.0,
  }
```

**Why keep both trend features AND latest values — aren't they redundant?**

They contain different and complementary information:

```
Scenario A: Patient deteriorating rapidly in the last 3 readings
  HR readings across 20 readings: [85, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                                    96, 97, 98, 99, 100, 105, 110, 115, 118, 122]
  
  HR_mean = 97.7 bpm  ← pulled toward the stable early readings
  latest_HR = 122 bpm ← current state is already at dangerous level
  
  The mean says "moderately elevated." The latest says "acutely high."
  Both are true but emphasise different aspects. The model uses both.

Scenario B: Patient with previously high but now recovering HR
  HR readings: [130, 128, 125, 122, 119, 117, 115, 113, 111, 110,
                109, 108, 107, 106, 105, 103, 102, 100, 98, 96]
  
  HR_mean = 110.9 bpm  ← reflects the high starting state
  latest_HR = 96 bpm   ← now nearly normal (recovering)
  
  The mean would suggest dangerous tachycardia.
  The latest corrects this: the patient is improving.
```

Without latest values, the model would not know the current state. Without trend features, the model would not know whether the current state is a brief spike or a sustained pattern.

**Why GCS_eye_opening and stress_score appear here (not in trend features):**

GCS and stress score are not collected at every vital sign reading interval — they require active assessment (camera monitoring or nurse observation). They reflect the patient's current neurological and pain state, which is treated as a point-in-time measurement rather than a time series. The two values are placed in the "latest" group because they describe the current state at the time of this assessment.

---

## 7. Step 4 — TF-IDF Text Encoding (90 features)

**Purpose:** Convert the free-text clinical note into 90 numerical features that the DNN can process alongside the vital sign numbers.

### What is TF-IDF?

TF-IDF (Term Frequency – Inverse Document Frequency) is a classical text analysis method that converts a document into a bag-of-words vector where each dimension represents a vocabulary term and the value represents how important that term is in this specific document compared to all documents in the training corpus.

**Two components:**

**TF (Term Frequency):** How often does a term appear in THIS note?
```
Example: "creatinine" appears 2 times in a 500-word note
  TF = log(1 + 2) = 1.099  (using sublinear_tf=True — we use log count)
  
  Why log instead of raw count?
  "creatinine" mentioned 10 times doesn't mean the problem is 10× as severe
  as mentioning it once. The log scale compresses this:
    count=1  → log(2)   = 0.69
    count=5  → log(6)   = 1.79  (2.6× the single-mention weight, not 5×)
    count=10 → log(11)  = 2.40  (3.5× the single-mention weight, not 10×)
```

**IDF (Inverse Document Frequency):** How rare is this term across ALL training notes?
```
IDF = log(total_documents / documents_containing_this_term)

"creatinine" appears in 2,000 of 41,179 training admissions:
  IDF = log(41,179 / 2,000) = log(20.6) = 3.03
  Moderately rare → moderate importance weight

"acute" appears in 30,000 of 41,179 training admissions:
  IDF = log(41,179 / 30,000) = log(1.37) = 0.31
  Very common → low importance weight (appears everywhere, less discriminating)

"oliguria" (low urine output) appears in 300 of 41,179 admissions:
  IDF = log(41,179 / 300) = log(137) = 4.92
  Rare but clinically important → high importance weight
```

**Final TF-IDF weight = TF × IDF:**
A term gets a high weight when it is both frequently mentioned in this specific note (TF) AND rarely mentioned across most notes (IDF). This correctly identifies the clinically important, specific terms that distinguish this patient from the average.

### The SOFA Vocabulary Whitelist — Why Not Just Use All Words?

The most straightforward approach would be to let TF-IDF automatically pick the most variable terms across the training corpus. This is called data-driven TF-IDF.

**Why data-driven TF-IDF fails for this application:**

Clinical notes in MIMIC-III contain enormous amounts of administrative and medication dispensing language:
```
Top-ranked data-driven TF-IDF terms (when using 600 features):
  "tablet"     — appears in every medication order: "take 1 tablet twice daily"
  "sig"        — Latin "take as directed" — in every prescription
  "refills"    — "no refills" / "1 refill" — in every discharge prescription
  "disp"       — dispense count, in every pharmacy note
  "capsule"    — medication form, very common
  "afternoon"  — time references in nursing notes
  "wallet"     — patient personal effects documentation
```

These terms correlate with **documentation volume** (sicker patients get more notes, so more medication orders and more of these terms) but NOT with organ failure severity. A model trained on these features learns "patients with many notes are sicker" — a spurious correlation that fails when applied to new patients.

**The solution: an explicit 90-term SOFA vocabulary whitelist**

Every term in the vocabulary directly corresponds to one of the 6 SOFA organ components:

```
SOFA Component 1 — Respiratory (PaO2/FiO2 proxy):
  intubated, intubation, ventilator, vent, cpap, bipap,
  hypoxia, respiratory failure, respiratory distress,
  extubated, shortness breath

SOFA Component 2 — Coagulation (Platelet proxy):
  plt, platelets, bleed, bleeding, hemorrhage, inr, ptt,
  coagulopathy, thrombocytopenia

SOFA Component 3 — Hepatic (Bilirubin proxy):
  bilirubin, bili, totbili, liver, jaundice,
  hepatic, cirrhosis, liver failure

SOFA Component 4 — Cardiovascular (MAP + vasopressor proxy):
  hypotension, hypotensive, shock, vasopressor, pressors,
  norepinephrine, levophed, dopamine, epinephrine, dobutamine,
  septic shock

SOFA Component 5 — CNS (GCS proxy in text):
  sedated, sedation, coma, altered, confused, gcs,
  encephalopathy, delirium, unresponsive, altered mental

SOFA Component 6 — Renal (Creatinine + Urine Output proxy):
  creatinine, creat, renal, kidney, dialysis, aki,
  oliguria, urine, urine output, renal failure, acute kidney, crrt

General severity / infection:
  sepsis, septic, bacteremia, infection, cultures, antibiotics,
  fever, blood cultures, lactate, wbc, hgb, hct, sodium, potassium,
  hco, angap, bun, failure, acute, pneumonia, ards, pulmonary, dyspnea, edema
```

**Result:** Every one of the 90 features has a direct, explainable clinical meaning tied to a SOFA component or overall severity indicator.

### Worked Example — Our Running Patient

```
Clinical note (after loading the tfidf_vectorizer.pkl):
  "67-year-old male. SpO2 declining despite high-flow O2 therapy.
   History of COPD and hypertension. Creatinine elevated. Urine output low.
   On vasopressors."

After applying the 90-term vocabulary TF-IDF:

Feature          | TF-IDF value | Explanation
-----------------|--------------|------------------------------------------
"creatinine"     | 0.42         | Mentioned once → renal SOFA signal
"urine"          | 0.31         | "urine output low" → renal SOFA signal
"urine output"   | 0.38         | Bigram — more specific than "urine" alone
"vasopressor"    | 0.35         | "on vasopressors" → cardiovascular SOFA
"hypoxia"        | 0.28         | Implied by "SpO2 declining" context
(86 other terms) | 0.00         | Not mentioned in this note

Total non-zero entries: ~5 (most of the 90 terms are not present → 0.0)
```

The resulting 90-dimensional vector encodes which SOFA-relevant clinical concepts appear in this patient's note and how prominent they are.

### L2 Normalisation

```
norm="l2" in the TF-IDF configuration
```

After computing all 90 TF-IDF values, the vector is divided by its total length (L2 norm):
```
norm = √(0.42² + 0.31² + 0.38² + 0.35² + 0.28² + ...)
normalised_vector = original_vector / norm
```

This ensures that a patient whose note is 5,000 words long and a patient whose note is 500 words long are on the same scale, even though the longer note naturally has higher raw term frequencies. Without L2 normalisation, verbose documentation style would inflate TF-IDF values — a well-documented stable patient might appear more severe than a briefly-documented critical patient.

---

## 8. Step 5 — Feature Fusion and Alignment (108 features)

**Purpose:** Merge the three feature groups (trend vitals, latest vitals, TF-IDF text) into a single 108-dimensional input vector for the DNN.

### The Three Feature Groups

| Group | Count | What it represents |
|---|---|---|
| Trend vitals | 9 | Patient trajectory over last 20 readings (HR_mean, HR_std, SpO2_mean, SpO2_min, etc.) |
| Latest vitals + CV | 9 | Current state (latest HR, SpO2, MAP, GCS_eye_opening, stress_score) |
| TF-IDF text | 90 | Clinical note SOFA vocabulary encoding |
| **Total** | **108** | Complete feature vector for the DNN |

### The Fusion Process

```
Input:
  trend_features  = {"HR_mean": 101.5, "HR_std": 14.2, "RR_mean": 24.1,
                      "SpO2_mean": 88.5, "SpO2_min": 84.0, "Temp_mean": 38.6,
                      "SBP_mean": 92.3, "DBP_mean": 57.1, "MAP_mean": 68.8}
  
  latest_features = {"latest_HR": 122.0, "latest_RR": 30.0, "latest_SpO2": 84.0,
                      "latest_Temp": 39.1, "latest_SBP": 85.0, "latest_DBP": 50.0,
                      "latest_MAP": 62.0, "GCS_eye_opening": 2.0, "stress_score": 8.0}
  
  tfidf_df        = DataFrame(1, 90) — 90 TF-IDF values

Step A: Merge trend and latest into one DataFrame
  input_df = pd.DataFrame([{**trend_features, **latest_features}])
  Shape: (1, 18)  — 18 vital/CV features in one row

Step B: Horizontal concatenation with TF-IDF
  final_df = pd.concat([input_df, tfidf_df], axis=1).fillna(0)
  Shape: (1, 108)  — all 108 features

Step C: Column alignment to training order
  expected_cols = joblib.load("models/feature_columns.pkl")
  final_df = final_df.reindex(columns=expected_cols, fill_value=0)
  Shape: (1, 108) — columns in the EXACT order the model expects
```

### Why Column Alignment Is Critical

This step appears redundant — why would columns be in the wrong order? The answer lies in how pandas DataFrames are concatenated.

When two DataFrames are concatenated horizontally, the column order depends on which DataFrame was created first and how they were merged. In different Python versions, Pandas versions, or execution environments, this order can subtly differ.

The `feature_columns.pkl` file stores the exact column order that was present during training. The model's weight matrix at Layer 1 has a specific neuron for each of the 108 positions:
- Position 0 = HR_mean
- Position 17 = stress_score
- Position 18 = first TF-IDF term (e.g., "aki")
- ...
- Position 107 = last TF-IDF term

If the columns are shuffled, the StandardScaler applies HR_mean's statistics to a different feature. The model applies the weights it learned for HR_mean to a completely different clinical variable. The prediction becomes meaningless.

```
Example of what goes wrong without column alignment:

Correct order (position 0 = HR_mean, value = 101.5):
  First StandardScaler operation: (101.5 - 85.2) / 19.7 = +0.83 → correct z-score

Wrong order (position 0 = "creatinine" TF-IDF, value = 0.42):
  StandardScaler treats this as HR_mean: (0.42 - 85.2) / 19.7 = -4.31
  The model's HR_mean weight (learned for values around +0.83) gets applied to -4.31
  → catastrophically wrong computation → meaningless prediction
```

**This is called multimodal early fusion:** all three modalities (trend vitals, latest vitals, clinical text) are combined into one flat vector before entering the model. The DNN sees all 108 features simultaneously and can learn cross-modal relationships — for example, "high HR_mean AND 'vasopressor' mentioned in notes → especially high SOFA, because both confirm cardiovascular failure."

---

## 9. Step 6 — StandardScaler Normalisation

**Purpose:** Convert all 108 raw feature values to a common numeric scale so the DNN can learn equally from all features regardless of their natural units.

```
Input:
  final_df — DataFrame (1, 108) with raw clinical values
  scaler   — sklearn StandardScaler fitted on 48,150 training patients

Process:
  For each of the 108 features:
    scaled_value = (raw_value - mean_of_training_set) / std_of_training_set
  
  Then clip: scaled_value = max(-10, min(10, scaled_value))

Output: final_scaled — numpy array (1, 108) with z-scores clipped to [-10, 10]
```

### Why Normalisation Is Necessary

The 108 features have wildly different natural scales:

| Feature | Typical range | Natural magnitude |
|---|---|---|
| HR_mean | 60–150 bpm | ~100 |
| SpO2_min | 70–100% | ~90 |
| MAP_mean | 50–100 mmHg | ~75 |
| "creatinine" TF-IDF | 0.0–0.8 | ~0.3 |
| "septic" TF-IDF | 0.0–0.5 | ~0.2 |

The DNN's first layer computes a weighted sum: `h = W × x + b`. Without normalisation, this sum would be dominated by HR_mean (value ~100) and completely ignore TF-IDF features (values ~0.3). The weights for HR_mean would receive 300× larger gradients than the weights for TF-IDF features during training.

After StandardScaler, every feature has mean=0 and std=1 in the training set. Now all 108 features contribute on equal footing, and the model can learn from all modalities.

### Worked Example — Our Patient's 9 Vital Features After Scaling

```
Feature          | Raw value | Training mean | Training std | Z-score
-----------------|-----------|---------------|--------------|--------
HR_mean          | 101.5     | 85.2          | 19.7         | +0.83
HR_std           | 14.2      | 8.1           | 7.3          | +0.84
RR_mean          | 24.1      | 18.7          | 6.2          | +0.87
SpO2_mean        | 88.5      | 96.4          | 4.8          | −1.65
SpO2_min         | 84.0      | 93.2          | 7.1          | −1.30
Temp_mean        | 38.6      | 37.0          | 0.9          | +1.78
SBP_mean         | 92.3      | 119.4         | 24.3         | −1.12
DBP_mean         | 57.1      | 73.8          | 14.7         | −1.14
MAP_mean         | 68.8      | 88.7          | 17.2         | −1.16

After scaling:
  HR_mean = +0.83     (heart rate 0.83 standard deviations above training mean)
  SpO2_mean = −1.65   (oxygen saturation 1.65 std deviations below training mean — very low)
  Temp_mean = +1.78   (temperature significantly above normal — fever)
  MAP_mean = −1.16    (blood pressure below training mean — hypotension)
```

The model now sees our patient as someone with elevated HR and temperature, severely low oxygen, and significantly low blood pressure — all expressed in the same unit (standard deviations from training mean).

### Why Clipping at ±10 Is Essential

Some TF-IDF terms are very rare in the training corpus:

```
"bacteremia" (bloodstream infection) appears in only 0.1% of training admissions:
  IDF = log(41,179 / 41) ≈ 6.9
  
  For a patient with "bacteremia" prominently in their notes:
    TF-IDF raw value ≈ 1.5
    Training mean ≈ 0.05 (most notes don't contain it)
    Training std  ≈ 0.12
    Z-score = (1.5 - 0.05) / 0.12 ≈ 12.1 standard deviations above mean
    
  Without clipping: 12.1 propagates through the DNN
    First layer weight (say, 0.5) × 12.1 = 6.05 for this feature alone
    After 4 layers: could produce predictions of hundreds
    → "bacteremia" in the note would push SOFA prediction to 1,000+ (impossible)
  
  With clipping: 12.1 → clipped to 10.0
    The model still correctly knows "this term is very strongly present"
    But the value cannot become catastrophically large
```

**Critical rule:** The scaler is fitted ONLY on the training data (the 48,150 MIMIC-III patients). It is never updated at inference time. Using the training set's mean and std ensures that new patient data is interpreted relative to the same reference distribution the model was trained on.

---

## 10. Step 7 — DNN Inference (108 → SOFA score)

**Purpose:** Run the 108 normalised features through the trained neural network to produce a predicted SOFA score.

### The Architecture Recap

```
Input layer:    108 neurons  (one per scaled feature)
Hidden layer 1: 128 neurons  + ReLU
Hidden layer 2:  64 neurons  + ReLU
Hidden layer 3:  32 neurons  + ReLU
Output layer:     1 neuron   (no activation — raw regression output)
```

### The Forward Pass — Step by Step With Our Patient

```python
model.eval()           # disable training-mode behaviour
with torch.no_grad():  # do not track gradients — saves memory and computation
    X_tensor = torch.tensor(final_scaled, dtype=torch.float32)
    raw_pred = model(X_tensor).numpy().flatten()[0]
```

**What happens inside the model:**

```
Layer 1 computation (108 → 128):
  h1 = ReLU(W1 @ x + b1)
  
  W1 is a 128×108 weight matrix. For each of the 128 neurons, this computes
  a different weighted combination of the 108 input features.
  
  Example — one neuron in Layer 1:
    This neuron might have learned:
      Large weight on SpO2_mean (−0.71): "fires when SpO2 is very low"
      Large weight on "vasopressor" TF-IDF (+0.62): "fires when vasopressors mentioned"
      Large weight on MAP_mean (−0.55): "fires when MAP is low"
      
    z = (−0.71)×(−1.65) + 0.62×scaled_vasopressor + (−0.55)×(−1.16) + ...
      = 1.17 + 0.54 + 0.64 + (contributions from other features)
      = ~2.8 (before ReLU)
    
    After ReLU: max(0, 2.8) = 2.8  (neuron is ACTIVE — pattern found)
    
    This neuron is detecting something like "hypoxic cardiovascular failure"
    All 128 neurons detect different patterns simultaneously.

Layer 2 computation (128 → 64):
  h2 = ReLU(W2 @ h1 + b2)
  
  Takes the 128 pattern activations from Layer 1 and combines them into 64
  higher-level clinical syndrome representations.
  
  Example: a Layer 2 neuron might combine Layer 1's "hypoxic failure" detector
  with Layer 1's "renal stress" detector and Layer 1's "tachycardia" detector
  to produce a "multi-organ failure" syndrome signal.

Layer 3 computation (64 → 32):
  h3 = ReLU(W3 @ h2 + b3)
  
  Further compression — 32 final severity representations.

Output layer (32 → 1):
  y = W4 @ h3 + b4
  
  Linear combination of 32 severity signals → single SOFA prediction.
  No activation function — the output is an unrestricted real number.
  
Output:
  raw_pred = 8.37  (for our septic shock patient)
             (model predicts SOFA ≈ 8.4 — High Risk territory)
```

**Why `model.eval()` matters:**

In training mode, some model behaviours are active that should not be active at inference. Even though we have no Dropout or BatchNorm, calling `model.eval()` is a best practice that signals all components (including any that might exist in future versions) to behave deterministically.

**Why `torch.no_grad()` matters:**

During training, PyTorch tracks every computation to enable backpropagation (the gradient computation). At inference time, we don't need gradients — we just want the prediction. `no_grad()` tells PyTorch to skip this tracking, saving memory and computation time. For a model this small, the difference is modest, but it is a correctness guarantee: no accidental gradient accumulation across inference calls.

---

## 11. Step 8 — SOFA Post-processing and Alert Classification

**Purpose:** Convert the raw DNN output (which is an unrestricted float) into a clinically meaningful score, risk level, and alert banner.

### SOFA Clipping

```
Input:  raw_pred — float (can be any real number, e.g., 8.37, or −0.3, or 26.1)

Process:
  sofa_score = float(np.clip(raw_pred, 0, 24))

Output: sofa_score — float in [0.0, 24.0]
```

The model is a regression function trained to output values in the SOFA range, but there is no mathematical guarantee that it stays within [0, 24]. For most patients, the raw output falls within range. For unusual patients near the boundary, it might predict slightly negative values or values slightly above 24. Clipping enforces the physiologically valid range.

```
Examples:
  raw_pred =  8.37 → sofa_score =  8.37  (unchanged)
  raw_pred = −0.31 → sofa_score =  0.00  (clipped to minimum — "no organ failure")
  raw_pred = 25.8  → sofa_score = 24.00  (clipped to maximum — "worst possible")
```

### Risk Classification

```
if sofa_score < 5.0:   risk = "Low Risk"       → green indicator
if sofa_score < 10.0:  risk = "Moderate Risk"  → amber indicator
else:                  risk = "High Risk"       → red indicator

Our patient: sofa_score = 8.37 → "Moderate Risk" (amber)
```

Note: 8.37 falls in the Moderate Risk band (5–10), even though the alert threshold fires at 8.0. This is intentional — the risk classification uses the standard clinical SOFA bands, while the alert threshold is calibrated separately for optimal sensitivity.

### The Alert Banner — Why Threshold 8.0 and Not 10?

The SOFA score is divided into standard clinical bands (Low: <5, Moderate: 5–10, High: ≥10). Naively, one might expect the alert to fire at SOFA ≥ 10 — the clinical High Risk boundary.

**But our model systematically under-predicts high SOFA scores:**

```
Why the model under-predicts:
  Training data: SOFA 0–4 = 63% of patients
                 SOFA 5–9 = 30% of patients
                 SOFA ≥10 = 6% of patients
  
  The model sees many more low-SOFA examples during training.
  Even with weighted loss, the model's predictions are pulled toward the majority.
  
  Observed pattern:
    Patient with actual SOFA = 12 → model predicts ~9.5 (under by 2.5)
    Patient with actual SOFA = 15 → model predicts ~11.2 (under by 3.8)
```

If the alert threshold were set at 10, the model would fail to alert for a patient with actual SOFA=12 (predicted=9.5 → below threshold → no alert → dangerous miss).

**Calibration at threshold 8.0:**

```
Alert threshold = 8.0:
  True Positive Rate (recall) for High-Risk patients = 52.5%
  (52.5% of true SOFA≥10 patients get alerts — a meaningful improvement)
  False alarm rate on Low-Risk patients = 1.4%
  (only 1.4% of Low-Risk patients trigger unnecessary alerts — acceptable)

Alert threshold = 10 (clinical boundary):
  True Positive Rate = 28.4%  (catches barely a quarter of High-Risk patients)
  
Alert threshold = 8.0 catches 24.1 percentage points more true critical patients
at the cost of only 1.4 percentage points more false alarms on stable patients.
```

For our patient: sofa_score = 8.37 ≥ 8.0 → **RED ALERT BANNER fires**.

### Prediction History

Every prediction is appended to `models/prediction_history.csv`:
```
Columns: timestamp, sofa_score, risk_level, HR, RR, SpO2, Temp, SBP, DBP, MAP, GCS, stress
Max rows: 50 (older predictions are dropped when the file exceeds 50 rows)
```

This enables the clinician to see a table of all recent predictions and visually track SOFA trends over successive assessments.

---

## 12. Step 9 — SHAP Explainability

**Purpose:** Compute how much each of the 108 features contributed to the predicted SOFA score, and present the most important ones to the clinician in plain language.

### The Clinical Need for Explanation

A SOFA score of 8.37 tells the clinician "this patient is deteriorating." But it does not tell them: is it the lungs that are failing? The kidneys? The cardiovascular system? All three? This is exactly what SHAP answers.

Without SHAP, the prediction is a black box — the clinician must independently assess every system. With SHAP, the system highlights the most impactful clinical factors for this specific patient, guiding the clinical assessment.

### What SHAP DeepExplainer Computes

SHAP (SHapley Additive exPlanations) comes from cooperative game theory. The original problem was: if a team of players wins a prize together, how should the prize be fairly divided among them based on each player's contribution?

Transposed to ML: if the model's output deviates from the baseline prediction by +4.57 points, how much of that +4.57 is attributable to each of the 108 input features?

**The SHAP guarantee:** The attributions are additive and complete:
```
Σ(all 108 SHAP values) = predicted_SOFA − baseline_SOFA

Where baseline_SOFA is the model's average prediction across the 300 background patients.

Our example:
  baseline_SOFA ≈ 3.80  (average SOFA prediction for "typical" training patients)
  predicted_SOFA = 8.37
  
  Total to attribute = 8.37 − 3.80 = +4.57
  
  Sum of all 108 SHAP values must equal exactly +4.57 ✓
```

### The Background Dataset — Why 300 Patients?

```
models/shap_background.npy — shape (300, 108)
```

The 300 background patients represent the "average" or "baseline" patient the model has seen. SHAP DeepExplainer answers: "If we replaced THIS patient's value for feature j with values drawn from the background distribution, how much would the prediction change?"

With 300 background samples, SHAP marginalises over 300 possible "average" values for each feature. This gives a statistically robust attribution rather than depending on a single arbitrary reference value.

**Why training data for the background, not a "zero vector"?**

Using zeros as background would mean asking "how much does this feature matter relative to a patient with all-zero features?" — which corresponds to a physiologically impossible patient (HR=0, SpO2=0, etc.). Using real training patients means asking "how much does this feature matter relative to typical real ICU patients?" — which is the clinically relevant question.

### The Computation

```python
explainer = shap.DeepExplainer(model, background_tensor)  # loaded at startup
shap_vals = explainer.shap_values(X_tensor)                # computed per patient
```

**What DeepExplainer does internally:**

DeepExplainer uses a technique related to Integrated Gradients. It:
1. Runs the patient's features through the model (the prediction to explain)
2. For each feature j: gradually interpolates between the patient's value and each of the 300 background values
3. Computes the gradient of the output with respect to feature j at each interpolation step
4. Integrates these gradients to get the total attribution for feature j

This is mathematically rigorous: the attributions satisfy the Shapley axioms (efficiency, symmetry, linearity, nullity) which together guarantee a fair and complete attribution.

**Output — SHAP values for our patient:**

```
Feature              | Original value | SHAP value | Interpretation
---------------------|----------------|------------|------------------------
latest_SpO2          | 84.0%          | +0.82      | Low O2 → pushed SOFA up 0.82 pts
SpO2_mean            | 88.5%          | +0.71      | Sustained low O2 → pushed up
latest_MAP           | 62.0 mmHg      | +0.61      | Low BP → pushed SOFA up
MAP_mean             | 68.8 mmHg      | +0.58      | Sustained low BP → pushed up
"vasopressor" (TF-IDF)| 0.35          | +0.44      | Vasopressor mention → pushed up
"creatinine" (TF-IDF) | 0.42          | +0.27      | Creatinine mention → pushed up
"urine output" (TF-IDF)| 0.38         | +0.21      | Oliguria mention → pushed up
GCS_eye_opening      | 2.0            | +0.18      | Low GCS → pushed up
HR_mean              | 101.5 bpm      | +0.15      | Elevated HR → small upward push
stress_score         | 8.0            | −0.12      | High stress → slightly down (see below)
SpO2_min             | 84.0%          | +0.09      | Nadir captures severe desaturation
... (97 other features with small attributions)

Sum of all SHAP values ≈ +4.57 = 8.37 − 3.80 ✓
```

**The counterintuitive stress_score result:**

The stress score contributes −0.12 (slightly reducing the predicted SOFA). This seems wrong — shouldn't a distressed patient have a higher SOFA?

SHAP explains what the **MODEL learned from MIMIC-III data**, not what clinical theory predicts. In the training data, patients who were agitated and visibly stressed (high stress score) often had lower SOFA scores than unresponsive, sedated patients (who had lower stress scores because they showed no facial expression). Consciousness and distress response are signs of a functioning brain — a SOFA=20 patient in a coma has no stress response. The model learned this correlation from training data.

This is documented in the dashboard's "About SHAP" expander as a known counterintuitive pattern.

### Clinical Filtering and Explanation Generation

Not all 108 SHAP values are equally useful to display. Technical feature names like "HR_std" or "MAP_mean" require interpretation. The top 7 clinically meaningful features are selected and converted to plain language:

```
Filtering:
  - Sort all 108 features by absolute SHAP value (largest impact first)
  - Filter to features with recognisable clinical names (using CLINICAL_KEYWORDS list)
  - Take top 7 from the filtered list

Clinical explanation generation per feature:
  Feature: "latest_SpO2", original_value = 84.0, SHAP = +0.82
  → "SpO₂ = 84.0% → severe hypoxaemia, increasing risk"
  
  Feature: "creatinine" (TF-IDF), original_value = 0.42, SHAP = +0.27
  → 'Clinical note contains "creatinine" → increasing risk'
  (The TF-IDF weight 0.42 is meaningless to a clinician; only the term is shown)
```

These 7 explanation strings appear in the dashboard's Explainability tab alongside a bar chart showing SHAP magnitudes.

---

## 13. Step 10 — Trend Analysis

**Purpose:** Classify each vital sign's 20-reading trend as increasing, stable, or decreasing, and compare the mean to the normal range. This is used exclusively to enrich the LLM prompt — it is NOT a second input to the DNN.

```
For each vital sign:
  1. Fit a linear regression line to the 20 readings
     (find the slope: is the trend going up or down over time?)
  
  2. Classify slope:
     slope > +0.1 → "increasing"
     slope < -0.1 → "decreasing"
     otherwise   → "stable"
  
  3. Classify mean relative to normal range:
     mean < normal_low → "low"
     mean > normal_high → "high"
     otherwise → "normal"
  
  4. Combine: "[vital] → [range_status] & [direction]"

Output for our patient:
  "HR → high & increasing"
  "RR → high & increasing"
  "SpO₂ → low & decreasing"
  "Temp → high & increasing"
  "SBP → low & decreasing"
  "DBP → low & decreasing"
  "MAP → low & decreasing"
```

**Why linear regression for trend, and not just comparing first to last value?**

Comparing first to last is sensitive to outlier values. If the 20th reading happened to be slightly higher due to measurement noise, a comparison would say "increasing" even if the overall trend was flat or declining.

A linear regression fits a line through all 20 points. The slope of that line represents the average rate of change, which is more robust to individual noisy readings. For a clinically meaningful trend (like SpO2 declining from 97% to 84% over 20 readings), the regression slope would be clearly negative — unambiguously "decreasing."

This trend description goes directly into the LLM prompt in Step 11. The LLM can then state: "SpO₂ is trending downward despite high-flow oxygen therapy, suggesting worsening respiratory failure" — incorporating the temporal context that the DNN sees only as summary statistics.

---

## 14. Step 11 — LLM Prompt Construction

**Purpose:** Package all computed results into a structured natural-language prompt that gives the LLM everything it needs to generate a clinically useful assessment.

The LLM (accessed via Groq's API) receives a single text prompt containing 7 components:

### Component 1 — Risk-Adaptive Urgency Prefix (Conditional)

```
IF sofa_score ≥ ALERT_THRESHOLD (8.0):
  Add at the very beginning of the prompt:
  
  "⚠️ CLINICAL ALERT — Predicted SOFA 8.4 (alert threshold ≥ 8)
   This patient shows signs of significant organ dysfunction.
   Structure your response for IMMEDIATE clinical action."
   
IF sofa_score < 8.0:
  No prefix — the prompt begins with the standard assessment request.
```

**Why this matters:** The LLM's instructions tell it to structure its response differently for high-risk patients. For SOFA ≥ 8, the "Immediate Actions" section appears first in the output — the clinician sees what to do right now before reading any background. For lower-risk patients, the "Condition" section appears first to give context before recommendations.

### Component 2 — Section Structure Instructions

```
For SOFA ≥ 8 (urgent — our patient):
  "Provide your assessment in this EXACT format:
   ## IMMEDIATE ACTIONS
   ## CURRENT CONDITION
   ## LIKELY CAUSES
   ## RECOMMENDED MONITORING"
   (Actions first — urgency communicated immediately)

For SOFA < 8 (monitoring):
  "Provide your assessment in this EXACT format:
   ## CURRENT CONDITION
   ## LIKELY CAUSES
   ## RECOMMENDED MONITORING
   ## IMMEDIATE ACTIONS"
   (Context first, actions at the end)
```

### Component 3 — SOFA Score and Risk Level

```
"SOFA Score: 8.4 / 24   (higher = worse organ failure)
 Risk Level: Moderate Risk
 Alert: SOFA ≥ 8.0 threshold exceeded"
```

### Component 4 — Clinical Notes (verbatim)

```
"CLINICAL NOTES FROM CARE TEAM:
 67-year-old male. SpO2 declining despite high-flow O2 therapy.
 History of COPD and hypertension. Creatinine elevated. Urine output low.
 On vasopressors."
```

The LLM reads these notes the same way a physician would — understanding clinical context, patient history, and the significance of "despite high-flow O2 therapy" (the resistance to treatment matters).

### Component 5 — Latest Vital Signs With Normal Ranges

```
"CURRENT VITAL SIGNS:
  HR:   122 bpm    (normal: 60–100)
  RR:   30 br/min  (normal: 12–20)
  SpO₂: 84%        (normal: 95–100)
  Temp: 39.1°C     (normal: 36.5–37.5)
  SBP:  85 mmHg    (normal: 100–120)
  DBP:  50 mmHg    (normal: 60–80)
  MAP:  62 mmHg    (normal: 70–100)"
```

The normal ranges are included so the LLM can identify which vitals are abnormal without needing to infer from its training knowledge. This makes the assessment grounded in current clinical values.

### Component 6 — Neurological and Stress Indicators

```
"NEUROLOGICAL ASSESSMENT:
  GCS Eye Opening: 2/4 (1=No response, 2=To pain, 3=To voice, 4=Spontaneous)
  Stress Score: 8.0/10 (0=calm, 10=severe distress)"
```

### Component 7 — SHAP Outputs, Risk Factors, and Vital Trends

```
"KEY RISK FACTORS IDENTIFIED BY PREDICTIVE MODEL:
  - Low oxygen levels
  - Hypotension
  - Renal stress indicators

FEATURE-LEVEL EXPLANATIONS (from SHAP attribution analysis):
  - SpO₂ = 84.0% → severe hypoxaemia, increasing risk
  - MAP = 62.0 mmHg → blood pressure instability, increasing risk
  - Clinical note contains 'vasopressor' → increasing risk
  - Clinical note contains 'creatinine' → increasing risk
  - Clinical note contains 'urine output' → increasing risk
  - GCS Eye Opening = 2 → neurological compromise, increasing risk
  - SpO₂_mean = 88.5% (24h average) → sustained hypoxaemia, increasing risk

VITAL SIGN TRENDS (last 20 readings):
  - HR → high & increasing
  - RR → high & increasing
  - SpO₂ → low & decreasing
  - Temp → high & increasing
  - MAP → low & decreasing"
```

**Why include SHAP in the prompt?**

Without SHAP, the LLM only sees raw numbers and clinical notes — the same information any physician would have. It would generate a generic septic shock assessment.

With SHAP, the LLM knows WHICH features the predictive model found most important for THIS specific patient. The LLM can now say: "The predictive model identified SpO₂ and MAP as the dominant drivers — combined with vasopressor mentions, this suggests cardiovascular failure is the primary concern driving the SOFA score, with renal compromise as a secondary finding." This personalises the assessment to the specific pattern of organ dysfunction in this patient.

---

## 15. Step 12 — LLM Self-Consistency Check

**Purpose:** Measure how reliably the LLM responds to this patient's data by sending the identical prompt three times and measuring whether all three independent responses agree.

### Why Three Independent Calls?

Language models are probabilistic — given the same input, they generate somewhat different outputs each time due to temperature-controlled sampling. This variability is usually a feature (it prevents repetitive outputs), but in clinical decision support, variability raises a concern: is the LLM uncertain about what to recommend?

**The self-consistency principle** (Wang et al., 2022): if the correct answer is well-determined, multiple independent reasoning paths should all arrive at the same conclusion. If the LLM gives different recommendations for the same patient three times, either the clinical picture is genuinely ambiguous (defensible) or the LLM is unreliable for this patient type (concerning).

```
Why temperature = 0.2 (not 0 and not 1.0):
  temperature = 0.0: always picks the single highest-probability token
    → three responses would be nearly identical → artificial high consistency
    → does not reflect genuine robustness
  
  temperature = 0.2: small amount of randomness → genuine but not extreme variation
    → if all three agree at 0.2 temperature, the LLM is genuinely confident
    → appropriate for measuring clinical reliability
  
  temperature = 1.0: high randomness → responses differ even when the clinical
    picture is clear → artificially low consistency → misleading
```

### Three API Calls

```
For each of 3 iterations:
  response = Groq API call:
    model       = "openai/gpt-oss-120b"
    temperature = 0.2
    max_tokens  = 2,500
    messages    = [system_message, user_message_with_full_prompt]

Output:
  responses[0], responses[1], responses[2] — three complete clinical assessments
  main_response = responses[0]  (displayed as the primary report)
```

The three responses are not shown to the clinician one-by-one — only the consistency score and the first response are shown in the main interface. All three are accessible in an expandable section for review.

### The Three-Component Consistency Score

**The problem with simple text similarity:**

Two responses might say "Administer norepinephrine" vs "Start vasopressor support" — these are clinically identical recommendations, but share no common words. Simple text comparison would incorrectly score them as disagreeing.

**The solution: measure clinical concept agreement:**

```
Component 1 — TF-IDF Cosine Similarity (20% weight):
  Convert all 3 responses to TF-IDF vectors
  Compute pairwise cosine similarity between all pairs: (1,2), (1,3), (2,3)
  Average the 3 similarity scores
  
  Purpose: Captures overall phrasing and vocabulary overlap
  Limitation: misses synonym clinical recommendations (why this is only 20%)

Component 2 — Clinical Intervention Agreement (50% weight):
  For each of 7 intervention categories:
    Category: vasopressors
    Synonyms: ["vasopressor", "norepinephrine", "levophed", "dopamine",
               "epinephrine", "dobutamine", "pressors"]
    
    For each response: does ANY synonym appear? → 1 or 0
    Agreement score for this category = max(count_yes, count_no) / 3
    (if 3 agree "yes" or 3 agree "no" → 1.0; if 2 vs 1 → 0.67)
  
  Average agreement across all 7 categories
  Categories covered: vasopressors, antibiotics, fluids, oxygen therapy,
                      monitoring orders, lab orders, renal support
  
  This correctly scores "norepinephrine" and "vasopressor" as AGREEING
  because both match the "vasopressor" synonym list.

Component 3 — Clinical Condition Agreement (30% weight):
  Same majority-agreement formula applied to 8 clinical condition categories:
  septic shock, infection, ARDS, AKI, hypoxaemia, hypotension, tachycardia, urgency
  
  If all 3 responses identify "septic shock" (or its synonyms) → agreement = 1.0
  If only 2 of 3 mention it → agreement = 0.67
  If only 1 mentions it → agreement = 0.33

Combined score = 0.20 × TF-IDF + 0.50 × interventions + 0.30 × conditions
```

**Worked example — our septic shock patient:**

```
All 3 LLM responses for our high-risk patient:
  Response 1: "... immediately start norepinephrine for hemodynamic support,
               broad-spectrum antibiotics given suspected septic shock,
               urgent ICU monitoring..."
  
  Response 2: "... vasopressor support initiated (levophed), antibiotic coverage
               for presumed sepsis, continuous monitoring essential..."
  
  Response 3: "... initiate pressors for MAP <65, empirical antibiotics for septic shock,
               close monitoring of renal function and urine output..."

Component 1 — TF-IDF similarity:
  Responses use different words → sim(1,2)=0.55, sim(1,3)=0.52, sim(2,3)=0.57
  Average = 0.55

Component 2 — Intervention agreement:
  vasopressors:  all 3 mention (norepinephrine, levophed, pressors) → 1.0
  antibiotics:   all 3 mention → 1.0
  monitoring:    all 3 mention → 1.0
  fluids:        2 of 3 mention → 0.67
  oxygen:        1 of 3 explicitly mentions (implied but not stated) → 0.33
  Average = 0.80

Component 3 — Condition agreement:
  septic shock:  all 3 → 1.0
  infection:     all 3 → 1.0
  hypotension:   all 3 → 1.0
  AKI:           2 of 3 → 0.67
  Average = 0.92

Combined = 0.20 × 0.55 + 0.50 × 0.80 + 0.30 × 0.92
         = 0.11 + 0.40 + 0.28 = 0.79
         → "Moderate ⚠️" (just below the 0.80 "High" threshold)
```

### Reliability Labels

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.80 | High ✅ | All 3 responses agree on findings, diagnoses, and interventions. |
| 0.60–0.79 | Moderate ⚠️ | Responses agree on core findings with some variation in secondary recommendations. |
| < 0.60 | Low ❌ | Significant disagreement — clinical picture may be ambiguous or LLM is uncertain. |

**What each reliability level should tell the clinician:**

- **High:** The LLM's recommendations are robust. They appear regardless of which random sampling path the model took. Clinician can treat the report with reasonable confidence (while still applying their own judgment).
- **Moderate:** Core recommendations are consistent, but details vary. Pay attention to the main findings and be cautious about specific medication doses or secondary interventions.
- **Low:** Something about this patient's presentation is unusual or ambiguous. The LLM's recommendations should be treated with extra skepticism. Direct clinical assessment is especially important.

---

## 16. Step 13 — Output Display (Four Tabs)

### Tab 1 — Risk Assessment

**What it shows:**

```
┌─────────────────────────────────────────────────┐
│  ⚠️ ALERT: Predicted SOFA = 8.4 ≥ threshold    │
│  (red banner when SOFA ≥ 8.0)                   │
├────────────────────────────────────────────────-┤
│  SOFA Score: 8.4 / 24   [Moderate Risk] 🟡      │
├─────────────────────────────────────────────────┤
│  VITAL SIGNS TABLE:                             │
│  HR:   122 bpm   [↑ HIGH]  (normal: 60–100)    │
│  RR:   30 br/min [↑ HIGH]  (normal: 12–20)     │
│  SpO₂: 84%       [↓ LOW]   (normal: 95–100)    │
│  ...                                            │
├─────────────────────────────────────────────────┤
│  PREDICTION HISTORY TABLE (last 50 assessments)│
│  timestamp | SOFA | Risk | HR | SpO2 | MAP ...  │
├─────────────────────────────────────────────────┤
│  VITAL TRENDS CHART (last 20 readings):        │
│  [7 line charts, one per vital, with green      │
│   bands showing normal ranges]                  │
└─────────────────────────────────────────────────┘
```

The vital trends chart gives the clinician immediate visual context for whether the current values are a sudden change or the latest point in a sustained trend.

---

### Tab 2 — Explainability

**What it shows:**

```
┌─────────────────────────────────────────────────┐
│  SHAP FEATURE IMPORTANCE TABLE:                 │
│  Feature        | Value  | Impact | Direction   │
│  latest_SpO2    | 84.0%  | +0.82  | ↑ worsening│
│  SpO2_mean      | 88.5%  | +0.71  | ↑ worsening│
│  latest_MAP     | 62 mmHg| +0.61  | ↑ worsening│
│  vasopressor    | 0.35   | +0.44  | ↑ worsening│
│  creatinine     | 0.42   | +0.27  | ↑ worsening│
│  urine output   | 0.38   | +0.21  | ↑ worsening│
│  GCS_eye_opening| 2.0    | +0.18  | ↑ worsening│
├─────────────────────────────────────────────────┤
│  CLINICAL INTERPRETATIONS:                      │
│  • SpO₂ = 84.0% → severe hypoxaemia            │
│  • MAP = 62 mmHg → blood pressure instability  │
│  • Note contains "vasopressor" → risk signal   │
│  • Note contains "creatinine" → risk signal    │
│  ...                                            │
├─────────────────────────────────────────────────┤
│  KEY RISK FACTORS:                              │
│  ⚠️ Low oxygen levels  ⚠️ Hypotension          │
│  ⚠️ Renal stress indicators                    │
├─────────────────────────────────────────────────┤
│  [ℹ️ About SHAP — expander with disclaimer      │
│   explaining counterintuitive values]           │
└─────────────────────────────────────────────────┘
```

The SHAP table uses original clinical values (84.0%, 62 mmHg) not z-scores (−3.2, −1.16) — the clinician sees familiar numbers, not internal model representations.

---

### Tab 3 — AI Clinical Report

**What it shows:**

```
┌─────────────────────────────────────────────────┐
│  CONSISTENCY SCORE:  0.79   RELIABILITY: Moderate⚠️│
│  (coloured banner explaining what it means)    │
├─────────────────────────────────────────────────┤
│  PRIMARY CLINICAL ASSESSMENT (Response 1):      │
│                                                 │
│  ## IMMEDIATE ACTIONS                           │
│  • Initiate vasopressor support (MAP <65 mmHg)  │
│  • Begin broad-spectrum antibiotics for sepsis  │
│  • Urgent ICU-level monitoring                 │
│  • Obtain blood cultures before antibiotics    │
│                                                 │
│  ## CURRENT CONDITION                           │
│  Patient presents with septic shock...          │
│  [full 4-section LLM report]                   │
├─────────────────────────────────────────────────┤
│  [▶ View all 3 independent LLM responses]       │
│  (expandable — shows all 3 for comparison)     │
├─────────────────────────────────────────────────┤
│  ⚠️ CLINICAL DISCLAIMER:                        │
│  This is a decision-support tool. All clinical  │
│  decisions must be made by qualified healthcare │
│  professionals.                                 │
└─────────────────────────────────────────────────┘
```

**Why the 3-response expander exists:**

If the consistency score is Moderate or Low, the clinician may want to see where the LLM responses diverged. By expanding the section, they can compare all three responses side-by-side and identify which recommendations were consistent (trustworthy) and which varied (uncertain).

---

### Tab 4 — Federated Learning

**What it shows:**

```
┌─────────────────────────────────────────────────┐
│  FL ARCHITECTURE DIAGRAM:                       │
│  Hospital 0 ─┐                                  │
│  Hospital 1 ─┼──► Central Server (FedYogi) ──►  │
│  Hospital 2 ─┘   [Aggregation + Best Round]     │
│  ↑ Only model weights travel — never patient    │
│    data                                         │
├─────────────────────────────────────────────────┤
│  TRAINING CONFIGURATION:                        │
│  Aggregation: FedYogi (η=0.01, β1=0.9, β2=0.99)│
│  Client reg.: FedProx (μ=0.5)                  │
│  Rounds: 100  |  Epochs/round: 3               │
│  Patients: 48,150 across 3 hospitals            │
├─────────────────────────────────────────────────┤
│  PERFORMANCE METRICS:                           │
│  MAE = 1.76 SOFA points                        │
│  R² = 0.4724                                   │
│  Best round: 24 of 100                         │
├─────────────────────────────────────────────────┤
│  MODEL ARCHITECTURE:                            │
│  108 → 128 → 64 → 32 → 1                       │
│  ~23,000 parameters, ReLU activations          │
│  No Dropout (FL incompatible)                  │
│                                                 │
│  PRIVACY: Differential Privacy OFF (can enable) │
└─────────────────────────────────────────────────┘
```

This tab contextualises the entire prediction: the clinician can see exactly how the model was trained, what privacy guarantees are in place, and what the model's validated accuracy is — giving them informed context for how much weight to place on its predictions.

---

## 17. Complete Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                      CLINICIAN INPUTS (Sidebar)                    │
├─────────────────┬──────────────────┬────────────────────────────────┤
│  7 VITAL SIGNS  │  2 CV FEATURES   │     CLINICAL NOTES (text)      │
│  HR, RR, SpO2   │  GCS Eye (1–4)   │     Free-text nurse/doctor     │
│  Temp, SBP      │  Stress (0–10)   │     observations, history      │
│  DBP, MAP       │                  │                                │
└───────┬─────────┴─────────┬────────┴──────────────┬─────────────────┘
        │                   │                        │
        ▼ validate first    │                        ▼
  ┌─────────────┐           │              ┌────────────────────┐
  │ INPUT       │           │              │  TF-IDF TRANSFORM  │
  │ VALIDATION  │           │              │  90-term SOFA      │
  │ (halt if    │           │              │  vocabulary        │
  │  invalid)   │           │              │  → 90 float values │
  └──────┬──────┘           │              └──────────┬─────────┘
         │ valid            │                         │
         ▼                  ▼                         │
  ┌────────────────┐  ┌─────────────────┐             │
  │ SLIDING WINDOW │  │ LATEST FEATURES │             │
  │ Append reading │  │ (7 vitals +     │             │
  │ Keep last 20   │  │  GCS + stress   │             │
  │ Save to CSV    │  │  = 9 values)    │             │
  └───────┬────────┘  └───────┬─────────┘             │
          │                   │                        │
          ▼                   │                        │
  ┌────────────────┐          │                        │
  │ TREND FEATURES │          │                        │
  │ 9 statistics   │          │                        │
  │ (mean, std,    │          │                        │
  │  min across    │          │                        │
  │  20 readings)  │          │                        │
  └────────┬───────┘          │                        │
           └──────────────────┘                        │
                    │ 9 trend + 9 latest = 18 features  │
                    ▼                                   │
           ┌─────────────────┐                          │
           │ FEATURE FUSION  │◄─────────────────────────┘
           │ 18 + 90 = 108   │
           │ Column-aligned  │
           │ to training     │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ STANDARDSCALER  │
           │ z-score norm    │
           │ + clip(±10)     │
           │ (108 features)  │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ FEDERATED DNN   │
           │ 108→128→64      │
           │     →32→1       │
           │ (ReLU, no       │
           │  Dropout)       │
           └────────┬────────┘
                    │ raw float
                    ▼
           ┌─────────────────┐
           │ SOFA SCORE      │
           │ clip to [0,24]  │
           │ → Risk label    │
           │ → Alert banner  │
           │ → Save to CSV   │
           └───┬─────────┬───┘
               │         │
      ┌────────┘         └──────────┐
      ▼                             ▼
┌────────────┐              ┌───────────────────┐
│    SHAP    │              │  TREND ANALYSIS   │
│ DeepExpl.  │              │  Linear reg per   │
│ 300 bg     │              │  vital → trend    │
│ samples    │              │  direction string │
│ → 108      │              │  (LLM prompt only)│
│ attributions│             └──────────┬────────┘
└────┬───────┘                         │
     │ top 7 features                  │
     │ + clinical                      │
     │ explanations                    │
     └─────────────────┬───────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │  LLM PROMPT BUILD   │
              │  Risk prefix +      │
              │  SOFA + vitals +    │
              │  notes + SHAP +     │
              │  trends + structure │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  3× GROQ API CALLS  │
              │  gpt-oss-120b       │
              │  temperature=0.2    │
              │  max_tokens=2,500   │
              └──────────┬──────────┘
                         │ 3 responses
                         ▼
              ┌─────────────────────┐
              │ CONSISTENCY METRIC  │
              │ 20% TF-IDF sim      │
              │ 50% interventions   │
              │ 30% conditions      │
              │ → score + label     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌───────────────────────────────────────┐
              │           4-TAB DASHBOARD              │
              │  Tab 1: SOFA Gauge + Vital Trends     │
              │  Tab 2: SHAP Explainability           │
              │  Tab 3: AI Report + Consistency Score │
              │  Tab 4: Federated Learning Info       │
              └───────────────────────────────────────┘
```

---

## 18. Artifacts Loaded at Startup

All model artifacts are loaded once when the application starts, using `@st.cache_resource` to prevent reloading on every user interaction:

| Artifact | File size | How loaded | What it provides |
|---|---|---|---|
| `models/federated_model.pth` | ~0.3 MB | `torch.load()` | The DNN weights (108→128→64→32→1) — the core prediction model |
| `models/scaler.pkl` | ~8 KB | `joblib.load()` | Training set mean and std for all 108 features — normalises inference inputs |
| `models/tfidf_vectorizer.pkl` | ~2 MB | `joblib.load()` | 90-term SOFA vocabulary with IDF weights — converts clinical text to features |
| `models/feature_columns.pkl` | ~40 KB | `joblib.load()` | Ordered list of 108 column names — enforces consistent feature ordering |
| `models/shap_background.npy` | ~1.4 MB | `np.load()` | 300 training patients as reference for SHAP attribution |
| `models/training_metadata.json` | ~2 KB | `json.load()` | Training stats, hyperparameters, performance metrics for Tab 4 |

**Per-prediction file access (not cached — read/written each time):**

| Artifact | Read or Write | Purpose |
|---|---|---|
| `models/patient_vitals.csv` | Read + Write | The 20-reading sliding window (Step 1) |
| `models/prediction_history.csv` | Read + Write | Log of last 50 predictions (Tab 1 history table) |

---

## 19. Feature Count Summary

| Pipeline Stage | Feature count | Format |
|---|---|---|
| Clinician input — vital signs | 7 | Raw floats (bpm, %, mmHg, °C) |
| Clinician input — CV scores | 2 | Raw float/int (1–4, 0–10) |
| Clinician input — clinical note | 1 | Variable-length string |
| After sliding window | 7 vitals × 20 rows | DataFrame — temporal history |
| After trend feature engineering | 9 | Floats (means, std, min across 20 readings) |
| After latest vital packaging | 9 | Raw floats + GCS + stress score |
| After TF-IDF encoding | 90 | Float weights (SOFA vocabulary presence/frequency) |
| **After feature fusion (Step 5)** | **108** | **Single flat float vector — model input** |
| After StandardScaler | 108 | Z-scores clipped to ±10 — same scale as training |
| After DNN inference (Step 7) | 1 | Raw SOFA prediction (unrestricted float) |
| After SOFA clipping (Step 8) | 1 | SOFA score in [0.0, 24.0] |
| After SHAP computation (Step 9) | 108 | Attribution values (+/−, sum = prediction − baseline) |
| After SHAP filtering | 7 | Top clinically meaningful features displayed |
| After trend analysis (Step 10) | 7 | Trend direction strings (for LLM prompt only) |
| After LLM calls (Step 12) | 3 | Full clinical assessment text reports |
| After consistency scoring | 1 | Reliability score [0.0, 1.0] |
| **Dashboard output** | **4 tabs** | **SOFA gauge, SHAP chart, clinical report, FL info** |

---

*Documentation for ICU CDSS Capstone Project*
*Pipeline: 7 vitals + 2 CV scores + clinical text → 108 features → SOFA → SHAP + LLM*
*Model: PyTorch DNN trained via Flower Federated Learning (FedYogi + FedProx)*
*LLM: gpt-oss-120b via Groq API with 3-call self-consistency verification*

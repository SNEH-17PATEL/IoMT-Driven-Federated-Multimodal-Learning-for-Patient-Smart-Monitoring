# Complete Input-Output Lifecycle — ICU CDSS

**Multimodal Intelligence System with Federated Learning for ICU Patient Monitoring**

This document traces every input from the moment a clinician enters data, through every
transformation, through the model, through SHAP, and through the LLM — with exact
shapes, values, and reasoning at each step.

---

## Table of Contents

1. [System Inputs — Five Modalities](#1-system-inputs--five-modalities)
2. [Input Validation](#2-input-validation)
3. [Step 1 — Sliding Window (Temporal Buffer)](#3-step-1--sliding-window-temporal-buffer)
4. [Step 2 — Trend Feature Engineering (9 features)](#4-step-2--trend-feature-engineering-9-features)
5. [Step 3 — Latest Vital + CV Features (9 features)](#5-step-3--latest-vital--cv-features-9-features)
6. [Step 4 — TF-IDF Text Encoding (600 features)](#6-step-4--tf-idf-text-encoding-600-features)
7. [Step 5 — Feature Fusion and Alignment (618 features)](#7-step-5--feature-fusion-and-alignment-618-features)
8. [Step 6 — StandardScaler Normalisation](#8-step-6--standardscaler-normalisation)
9. [Step 7 — DNN Inference (618 → SOFA)](#9-step-7--dnn-inference-618--sofa)
10. [Step 8 — SOFA Post-processing and Alert Classification](#10-step-8--sofa-post-processing-and-alert-classification)
11. [Step 9 — SHAP Explainability](#11-step-9--shap-explainability)
12. [Step 10 — Trend Analysis (Parallel to SHAP)](#12-step-10--trend-analysis-parallel-to-shap)
13. [Step 11 — LLM Prompt Construction](#13-step-11--llm-prompt-construction)
14. [Step 12 — LLM Self-Consistency Check](#14-step-12--llm-self-consistency-check)
15. [Step 13 — Output Display (Four Tabs)](#15-step-13--output-display-four-tabs)
16. [Complete Data Flow Diagram](#16-complete-data-flow-diagram)
17. [Artifacts Loaded at Startup](#17-artifacts-loaded-at-startup)

---

## 1. System Inputs — Five Modalities

The system accepts **5 independent input modalities** from the clinician via the sidebar:

### Modality 1 — Real-time Vital Signs (7 numbers)

| Input Field   | Unit    | Normal Range  | Valid Range  | Example Value |
|---------------|---------|---------------|--------------|---------------|
| HR            | bpm     | 60–100        | 30–220       | 122.0         |
| RR            | br/min  | 12–20         | 5–60         | 30.0          |
| SpO₂          | %       | 95–100        | 50–100       | 84.0          |
| Temp          | °C      | 36.5–37.5     | 30–43        | 39.1          |
| SBP           | mmHg    | 100–120       | 40–250       | 85.0          |
| DBP           | mmHg    | 60–80         | 20–150       | 50.0          |
| MAP           | mmHg    | 70–100        | 30–200       | 62.0          |

These are the **latest** single-point readings entered by the clinician. They represent
the patient's current state right now.

### Modality 2 — Computer Vision Features (2 numbers)

| Input Field     | Scale   | Source / What it Represents              | Example |
|-----------------|---------|------------------------------------------|---------|
| GCS Eye Opening | 1–4     | Neurological score: 4=spontaneous,       | 2       |
|                 |         | 3=to voice, 2=to pain, 1=no response     |         |
| Stress Score    | 0–10    | Pain/agitation level assessed from       | 8       |
|                 |         | patient facial expression or demeanour   |         |

**Design note:** GCS Eye Opening and Stress Score are called "Computer Vision features"
because in the full system architecture they are meant to be derived from camera-based
CV models (facial expression analysis, pupil response). For the current implementation,
the clinician enters them manually; the pipeline treats them identically regardless of
how they were obtained.

### Modality 3 — Clinical Notes (free text)

```
Example input:
"67-year-old male. SpO2 declining despite high-flow O2 therapy.
 History of COPD and hypertension. Creatinine elevated. Urine output low."
```

- Type: Unstructured free text (any length)
- Source: Nurse/physician documentation, patient history, medication lists
- Preprocessing: Converted to 600 TF-IDF features (Step 4)

### Modality 4 — Historical Vitals (read from file)

The system reads `models/patient_vitals.csv`, which stores the last 20 readings of
the 7 vital signs with timestamps. This enables temporal feature computation.

- On first use: file is pre-populated with 20 default readings
- After each prediction: the new reading is appended and only the last 20 are kept

### Modality 5 — Loaded Model Artifacts (from training, not from clinician)

The following pre-trained artifacts are loaded at startup (once, cached):

| Artifact                         | What it contains                              |
|----------------------------------|-----------------------------------------------|
| `models/federated_model.pth`     | Trained DNN weights (618→256→128→64→1)        |
| `models/scaler.pkl`              | StandardScaler fitted on 618 training features|
| `models/tfidf_vectorizer.pkl`    | TF-IDF fitted on MIMIC-III clinical notes     |
| `models/feature_columns.pkl`     | Ordered list of 618 feature column names      |
| `models/shap_background.npy`     | 300 random background samples for SHAP        |

---

## 2. Input Validation

**Before any computation**, the system checks every vital is within its physiologically
plausible range (`VALID_RANGES`):

```
Input:   7 raw vital sign floats
Process: For each vital, check lo ≤ value ≤ hi
Output:  Pass (continue) OR error message + hard stop
```

If any vital is out of range (e.g., HR = 10 bpm or SpO₂ = 110%), the app halts with
an error message and no prediction is made. This prevents garbage-in-garbage-out
predictions.

Also checked: Groq API key must be present (required for the LLM module).

---

## 3. Step 1 — Sliding Window (Temporal Buffer)

**Purpose:** Build a 20-reading temporal context for trend computation.

```
Input:
  - Current reading: {HR: 122, RR: 30, SpO2: 84, Temp: 39.1,
                      SBP: 85, DBP: 50, MAP: 62}
  - Historical file: up to 20 previous readings

Process:
  1. Load patient_vitals.csv into a DataFrame (up to 20 rows)
  2. Append the new reading as row 21
  3. Keep only the LAST 20 rows (.tail(20))
  4. Save the updated DataFrame back to patient_vitals.csv

Output:
  - vitals_df: DataFrame of shape (≤20, 8)
    Columns: [time, HR, RR, SpO2, Temp, SBP, DBP, MAP]
```

**Why 20 readings?**
20 readings match the sliding window used during training. Each row in the
MIMIC-III training data was also derived from a 20-reading window. Using the
same window size at inference ensures the trend statistics are computed in the
same way the model was trained on them.

**Temporal reset:** The "Reset Patient History" button replaces the file with 20
default readings. This must be pressed when switching to a new patient so that
trend statistics are not contaminated with the previous patient's history.

---

## 4. Step 2 — Trend Feature Engineering (9 features)

**Purpose:** Summarise the patient's trajectory over the last 20 readings into
statistics that capture both current state and direction of change.

```
Input:  vitals_df — DataFrame of shape (≤20, 8)

Process: Compute 9 aggregate statistics across all rows:

  HR_mean   = mean of the HR column across all 20 readings
  HR_std    = standard deviation of HR (variability measure)
  RR_mean   = mean of RR
  SpO2_mean = mean of SpO2
  SpO2_min  = minimum SpO2 seen in the window (catches worst desaturation)
  Temp_mean = mean of Temperature
  SBP_mean  = mean of SBP
  DBP_mean  = mean of DBP
  MAP_mean  = mean of MAP

Output: trend_features — Python dict with 9 key-value pairs (floats)
  Example:
    {"HR_mean": 101.5, "HR_std": 14.2, "RR_mean": 24.1,
     "SpO2_mean": 88.5, "SpO2_min": 84.0, "Temp_mean": 38.6,
     "SBP_mean": 92.3, "DBP_mean": 57.1, "MAP_mean": 68.8}
```

**Why these 9 and not all 7×2=14?**
- HR has both mean AND std because heart rate variability (HRV) is itself a clinical
  signal — low HRV can indicate autonomic dysfunction in sepsis.
- SpO₂ has both mean AND min because even a brief desaturation nadir is dangerous
  (a patient averaging 93% but hitting 70% for 30 seconds is critically different
  from one stably at 93%).
- Other vitals use only mean because their variability is not independently prognostic
  in the same way.

---

## 5. Step 3 — Latest Vital + CV Features (9 features)

**Purpose:** Include the most recent absolute values alongside the trend statistics,
because the trend features alone lose the actual current values.

```
Input:
  - 7 vital sign floats entered by the clinician (current reading)
  - GCS_eye integer (1–4)
  - stress float (0–10)

Process: Package as a dictionary

Output: latest_features — Python dict with 9 key-value pairs
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

**Why keep latest values when trend features already include a mean?**
The mean is a summary of the window. If a patient deteriorated from stable to
critical in the last 3 readings, the mean would be pulled toward stable while
`latest_HR=122` clearly shows the current crisis. Both are informative in
different ways — together they give the model both current state and trajectory.

---

## 6. Step 4 — TF-IDF Text Encoding (600 features)

**Purpose:** Convert the free-text clinical note into a numeric vector that
the DNN can process alongside the vital sign numbers.

### What is TF-IDF?

TF-IDF (Term Frequency–Inverse Document Frequency) is a classical NLP method that
converts a text document into a bag-of-words numeric vector.

- **TF (Term Frequency):** How often does a word appear in THIS document?
  - "septic" appearing 3 times → higher TF than if it appeared once.
- **IDF (Inverse Document Frequency):** How rare is this word across ALL training notes?
  - "the" appears everywhere → low IDF → downweighted.
  - "vasopressor" is rare but medically specific → high IDF → upweighted.
- **TF-IDF weight = TF × IDF:** Words that are both frequent in this note AND rare
  across all notes get the highest weights.

### Configuration

- **Number of features:** 600 (top 600 most discriminative n-grams from MIMIC-III)
- **n-gram range:** (1, 2) — single words AND two-word phrases
  - Unigrams: "septic", "oxygen", "creatinine"
  - Bigrams: "septic shock", "acute kidney", "mechanical ventilation"
  - Bigrams capture clinical phrases that have different meaning from individual words
    ("failure" alone is ambiguous; "renal failure" is specific)
- **Fitted on:** MIMIC-III discharge summaries and nursing notes (~48,000 ICU stays)
- **Saved as:** `models/tfidf_vectorizer.pkl`

### Transformation

```
Input:  clinical_note (string)
        Example: "67-year-old male. SpO2 declining despite high-flow O2 therapy.
                  History of COPD and hypertension. Creatinine elevated. Urine output low."

Process:
  tfidf.transform([clinical_note])   # applies fitted vocabulary + IDF weights
  → sparse matrix of shape (1, 600)
  → convert to dense array and wrap in DataFrame

Output: tfidf_df — DataFrame of shape (1, 600)
  Columns: 600 n-gram strings (e.g., "septic", "acute kidney", "o2 therapy", ...)
  Values:  floats ≥ 0.0 (most are 0 — only present n-grams are non-zero)

  Example non-zero entries for this note:
    "creatinine":       0.42
    "o2 therapy":       0.38
    "copd":             0.35
    "urine output":     0.31
    "high flow":        0.28
    (all other 595 columns = 0.0)
```

**Why TF-IDF and not a transformer (BERT/BioBERT)?**
TF-IDF is fast (< 1 ms), deterministic, and produces a fixed-size 600-d vector that
concatenates cleanly with the tabular vital features. A transformer would produce
768-d or 1024-d contextual embeddings that require additional alignment architecture
(cross-modal attention, projection layers) and are much slower. For MIMIC-III SOFA
prediction, TF-IDF on clinical notes was shown in the literature to achieve comparable
predictive performance to BERT-based methods on this specific task.

---

## 7. Step 5 — Feature Fusion and Alignment (618 features)

**Purpose:** Merge the three feature groups into a single 618-dimensional input
vector that the DNN expects.

### Feature Groups

| Group             | Features | Columns                                         |
|-------------------|----------|-------------------------------------------------|
| Trend vitals      | 9        | HR_mean, HR_std, RR_mean, SpO2_mean, SpO2_min, |
|                   |          | Temp_mean, SBP_mean, DBP_mean, MAP_mean         |
| Latest vitals+CV  | 9        | latest_HR, latest_RR, latest_SpO2, latest_Temp,|
|                   |          | latest_SBP, latest_DBP, latest_MAP,             |
|                   |          | GCS_eye_opening, stress_score                   |
| TF-IDF text       | 600      | 600 clinical n-gram columns                     |
| **Total**         | **618**  |                                                 |

### Fusion Process

```
Input:
  trend_features  — dict of 9 floats
  latest_features — dict of 9 floats
  tfidf_df        — DataFrame (1, 600)

Step A: Merge trend and latest into one DataFrame
  input_df = pd.DataFrame([{**trend_features, **latest_features}])
  Shape: (1, 18)

Step B: Horizontal concatenation with TF-IDF
  final_df = pd.concat([input_df, tfidf_df], axis=1).fillna(0)
  Shape: (1, 618)

Step C: Column alignment to training order
  - Load the exact column order from feature_columns.pkl (saved during training)
  - Add any missing columns as 0 (handles edge case where the clinical note
    contains no n-grams from the trained vocabulary)
  - Reorder columns to match training exactly: final_df = final_df[expected_cols]
  Shape: (1, 618) — columns in the exact order the scaler and model expect

Output: final_df — DataFrame of shape (1, 618), dtype float64
```

**Why is column alignment critical?**
The StandardScaler learned a mean and variance for each position. Position 0 = HR_mean,
position 17 = stress_score, position 18 = tfidf["septic"], etc. If columns were in a
different order, the scaler would apply the HR_mean statistics to the wrong feature,
producing completely wrong normalised values and therefore wrong predictions.

**This is early fusion:** all three modalities are concatenated into a single flat vector
before entering the model. The model sees them simultaneously and can learn cross-modal
interactions (e.g., "high HR_mean AND 'septic' in the note → especially dangerous").

---

## 8. Step 6 — StandardScaler Normalisation

**Purpose:** Convert all 618 features to a common numeric scale so that no single
feature dominates the model's gradients due to its natural unit magnitude.

```
Input:  final_df — DataFrame (1, 618), raw clinical values
        scaler   — sklearn StandardScaler fitted on 48,000+ MIMIC-III training rows

Process:
  final_scaled = scaler.transform(final_df)

  For each feature j:
    scaled[j] = (value[j] - mean_j) / std_j

  Where mean_j and std_j were computed across all training patients during training.

Output: final_scaled — numpy array of shape (1, 618), dtype float64
        All values are now z-scores (mean≈0, std≈1 across training data)

  Examples:
    HR_mean = 101.5 bpm  → scaled ≈ +1.8 (1.8 std above training mean)
    SpO2_min = 84.0%     → scaled ≈ -3.2 (3.2 std below training mean — severe)
    MAP_mean = 68.8 mmHg → scaled ≈ -0.6
    "creatinine" TF-IDF  → scaled by its training distribution
```

**Why normalise?**
Without normalisation:
- HR_mean might range 60–150 bpm (range ≈ 90)
- SpO₂_min ranges 50–100% (range ≈ 50)
- "creatinine" TF-IDF score ranges 0–0.8 (range ≈ 0.8)

The DNN's first Linear layer computes a weighted sum of all 618 features.
Without normalisation, HR values (magnitude ~100) would dominate TF-IDF values
(magnitude ~0.3), making the model unable to learn from text features.
StandardScaler brings all features to the same scale (z-scores) so the model
can fairly learn from all modalities.

**The scaler is NEVER retrained at inference.** It was fitted once on all 48,000
training patients. The mean/std for every feature represents what "normal" looked like
in the MIMIC-III training cohort.

---

## 9. Step 7 — DNN Inference (618 → SOFA)

**Purpose:** Run the scaled 618-d feature vector through the trained neural network
to produce a predicted SOFA score.

### Architecture

```
Input layer:    618 neurons  (one per feature)
Hidden layer 1: 256 neurons  + ReLU activation
Hidden layer 2: 128 neurons  + ReLU activation
Hidden layer 3:  64 neurons  + ReLU activation
Output layer:     1 neuron   (linear — no activation)
```

### What each layer does

**Linear(618 → 256):**
Each of the 256 neurons computes a weighted sum of all 618 inputs:
  `h₁ = W₁ × x + b₁`   (W₁ is a 256×618 weight matrix)
This layer creates 256 latent representations that capture combinations of vital
signs and text features.

**ReLU:**
`output = max(0, h₁)` — sets negative values to zero.
Introduces non-linearity so the model can learn non-linear clinical relationships
(e.g., "SpO₂ below 88 AND HR above 120" is more dangerous than either alone).

**Linear(256 → 128) → ReLU → Linear(128 → 64) → ReLU:**
Each successive layer compresses the representation and learns increasingly abstract
clinical patterns.

**Linear(64 → 1):**
Final projection to a single output value — the predicted SOFA score.
No activation function — allows the output to be any real number (unrestricted range).

### Forward Pass

```
Input:
  X_tensor = torch.tensor(final_scaled, dtype=torch.float32)
  Shape: (1, 618)

Process:
  model.eval()          # disable training mode (no gradient tracking)
  with torch.no_grad(): # save memory, disable backprop
      raw_pred = model(X_tensor).numpy().flatten()[0]

  Internal computation:
    h1 = ReLU(W1 @ x + b1)   # shape: (1, 256)
    h2 = ReLU(W2 @ h1 + b2)  # shape: (1, 128)
    h3 = ReLU(W3 @ h2 + b3)  # shape: (1, 64)
    y  = W4 @ h3 + b4         # shape: (1, 1)

Output:
  raw_pred — single float (unrestricted, can be any real number)
  Example: 6.17 (for the septic shock patient)
```

### Why No Dropout?

Dropout was tested (p=0.3/0.2 → R²=0.057, p=0.1/0.05 → R²=-0.067) but caused
Federated Learning divergence. The reason: each hospital client trains with different
random Dropout masks, so Hospital 0 might drop neurons 5, 23, 187 while Hospital 1
drops neurons 12, 67, 201. Their gradient updates conflict because they are
effectively training different sub-networks. FedAvg averages these conflicting updates
and cannot converge. Regularisation is achieved instead with AdamW weight decay (1e-4).

### Training Details (for context)

The model was trained using Federated Learning across 3 simulated hospitals:
- **Loss function:** Weighted MSE — `loss = mean(w × (pred − target)²)`
  where `w = 1 + target × 3.0` (high-SOFA patients get 2–5× more gradient weight)
- **Optimizer:** AdamW (Adam with decoupled weight decay = 1e-4)
- **Aggregation:** FedAvg — server averages weight matrices from all 3 hospitals
- **Rounds:** 20 FL rounds, 10 local epochs per round
- **Final performance:** MAE = 2.05 SOFA points, R² = 0.25

---

## 10. Step 8 — SOFA Post-processing and Alert Classification

### SOFA Clipping

```
Input:  raw_pred — float (unrestricted, e.g., 6.17 or -0.3 or 26.1)

Process:
  sofa_score = float(np.clip(raw_pred, 0, 24))

  The model outputs a regression value; clipping enforces the physiological
  bounds of the SOFA scale (0 = no organ failure, 24 = maximum organ failure).

Output: sofa_score — float in [0.0, 24.0]
  Example: 6.17 → sofa_score = 6.17
           -0.3 → sofa_score = 0.0 (clipped to minimum)
           26.1 → sofa_score = 24.0 (clipped to maximum)
```

### Risk Classification

```
Input:  sofa_score — float in [0, 24]

Process:
  if sofa_score < 5:  risk = "Low Risk",      icon = "🟢"
  elif sofa_score < 10: risk = "Moderate Risk", icon = "🟡"
  else:               risk = "High Risk",      icon = "🔴"

Output: (risk_text, risk_icon, risk_color) — three strings
```

### Alert Banner

```
Input:  sofa_score, ALERT_THRESHOLD = 8.0

Process:
  if sofa_score ≥ 8.0:  → RED ALERT BANNER (high risk, urgent action)
  elif sofa_score ≥ 5.0: → YELLOW WARNING BANNER (moderate risk, monitor)
  else:                   → no banner

Output: Streamlit banner displayed at top of app
```

**Why threshold 8.0 instead of the clinical boundary of 10?**
The federated model systematically under-predicts high SOFA scores by ~2–3 points
(a patient with true SOFA=13 is often predicted as ~9–10). By lowering the alert
threshold from 10 to 8, the system catches more true high-risk patients:
- At threshold 10: High Risk recall = 28.4%
- At threshold 8.0: High Risk recall = 52.5% (+24.1 percentage points)
- False alarm rate on Low Risk patients: only 1.4% (negligible increase)

### Prediction History

Each prediction is appended to `models/prediction_history.csv` (up to last 50 rows):
- Timestamp, SOFA score, Risk label, all 7 vitals, GCS Eye, Stress Score

---

## 11. Step 9 — SHAP Explainability

**Purpose:** Explain WHY the model predicted this specific SOFA score by computing
how much each of the 618 features pushed the prediction up or down from the baseline.

### What is SHAP?

SHAP (SHapley Additive exPlanations) is a game-theoretic framework that attributes
the model's prediction to each individual input feature. It answers: "If I remove
this feature, how much would the predicted SOFA change?"

For a predicted SOFA of 6.17 with a baseline of 3.8 (model's average prediction):
- The sum of all 618 SHAP values = 6.17 - 3.8 = +2.37
- Each individual SHAP value = that feature's contribution to this +2.37 deviation.

### SHAP Background (300 samples)

```
Input:  models/shap_background.npy — numpy array of shape (300, 618)
        These are 300 randomly selected rows from the MIMIC-III training data,
        already StandardScaler-normalised (same scale as inference input).

Purpose: The background represents the "baseline patient" distribution.
         SHAP computes: "What would the model predict if I replaced this feature
         with a typical value from the background?"
         With 300 background samples, SHAP marginalises over 300 typical values
         for each feature — giving a robust, averaged attribution.
```

### SHAP DeepExplainer

```
Input:
  explainer — shap.DeepExplainer(model, bg_tensor)
              Created at startup using the 300 background samples as a PyTorch tensor.

  X_tensor  — shape (1, 618), the scaled inference input for this patient

Process:
  shap_raw = explainer.shap_values(X_tensor)

  DeepExplainer works by:
  1. Running the model forward with the patient's features (regular prediction)
  2. Running the model forward 300 times with background samples replacing each
     feature combination (marginalisation)
  3. Using gradient-based integration (similar to Integrated Gradients) to
     attribute the prediction difference to each feature

Output:
  shap_vals — numpy array of shape (618,)
  Each value is a signed float:
    Positive (+) → this feature INCREASED the predicted SOFA (worsening risk)
    Negative (−) → this feature DECREASED the predicted SOFA (protective/neutral)

  Example (for septic shock patient):
    latest_SpO2:    +0.82  (low SpO2 → pushed SOFA up by 0.82 points)
    latest_MAP:     +0.61  (low MAP → pushed SOFA up)
    SpO2_min:       +0.58  (severe desaturation nadir → pushed up)
    GCS_eye_opening: −0.31 (GCS=2 → pushed DOWN? see counterintuitive note below)
    "creatinine":   +0.27  (TF-IDF → mention of creatinine → pushed up)
    ...
    (all 618 values, summing to ≈ sofa_score − baseline)
```

**Important: SHAP uses SCALED values for computation but ORIGINAL values for display.**
The forward pass and gradient computation happen on `final_scaled` (z-scores).
But when displaying to the clinician, `final_df.iloc[0].values` (original clinical
values) are used so they see "SpO₂ = 84%" not "SpO₂ = -3.2 (z-score)".

### SHAP Feature Filtering

```
Input:  shap_vals (618,), expected_cols (618 feature names)

Process:
  1. Build a DataFrame with columns: [feature, original_value, impact, abs_impact]
  2. Sort by |impact| descending
  3. Filter to "clinically relevant" features using CLINICAL_KEYWORDS:
       - Vital/CV names: "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP", "GCS", "stress"
       - Clinical TF-IDF terms: "hypotension", "respiratory", "septic", "failure",
         "intubated", "vasopressor", "shock", "fever", "infection", "oxygen", etc.
  4. Take top 7 from filtered list (or top 7 overall if fewer than 3 pass filter)

Output:
  top_shap — DataFrame of shape (7, 4) with the 7 most impactful clinical features
  clinical_explanations — list of 7 human-readable strings
  key_risks — list of risk category labels (e.g., "Hypotension", "Sepsis / Infection")
```

### Clinical Explanation Generation

For each of the 7 top SHAP features, `interpret_shap()` generates a sentence:

```
Vital feature example:
  feature = "latest_SpO2", original_value = 84.0, impact = +0.82
  → "SpO₂ = 84.0% → severe hypoxaemia, increasing risk"

TF-IDF feature example:
  feature = "creatinine", original_value = 0.42, impact = +0.27
  → 'Clinical note contains "creatinine" → increasing risk'
  (Note: TF-IDF weight 0.42 is meaningless to clinicians; only the term is shown)
```

### RISK_MAP — Key Risk Categories

The SHAP feature names are also mapped to high-level risk labels using RISK_MAP:

```python
RISK_MAP = {
  "SpO2": "Low oxygen levels",
  "MAP":  "Hypotension",
  "GCS":  "Neurological deterioration",
  "septic": "Sepsis / Infection",
  ...
}
```

These labels appear as warning chips in the Explainability tab.

**Counterintuitive SHAP results:**
SHAP explains what the MODEL learned, not established clinical physiology. In MIMIC-III,
some correlations exist that seem backwards:
- A high Stress Score might appear as "reducing risk" — in the training data, agitated/
  responsive patients often had lower SOFA than unresponsive ones (consciousness implies
  less organ failure). SHAP correctly reflects this learned correlation.
- These cases are documented in the in-app "About SHAP" expander.

---

## 12. Step 10 — Trend Analysis (Parallel to SHAP)

**Purpose:** Classify whether each vital's 20-reading trend is increasing, stable,
or decreasing, and whether the mean is in the normal range.

```
Input:  vitals_df — DataFrame (20, 8)
        VITAL_CFG — list of (column, label, normal_low, normal_high) for 7 vitals

Process (for each vital):
  direction = get_trend(vitals_df[col].tolist())
    → fits a linear regression line to the 20 values
    → if slope > +0.1: "increasing"
    → if slope < -0.1: "decreasing"
    → else: "stable"

  status = classify_range(mean_value, lo, hi)
    → "low" / "normal" / "high"

Output:
  trend_lines — list of 7 strings like:
    "HR → high & increasing"
    "SpO₂ → low & decreasing"
    "MAP → low & stable"

  trend_text — newline-joined string passed to the LLM prompt
```

This information is NOT fed back into the DNN (the model already has trend features
built-in from Step 2). It is used exclusively to enrich the LLM prompt with
human-readable trend descriptions.

---

## 13. Step 11 — LLM Prompt Construction

**Purpose:** Package all computed information into a structured natural-language
prompt for the LLM to reason over.

### What goes into the prompt (7 components)

```
1. Risk-adaptive urgency prefix (conditional on SOFA ≥ ALERT_THRESHOLD)
   If SOFA ≥ 8.0:
     "⚠️ CLINICAL ALERT — Predicted SOFA 6.2 (alert threshold ≥ 8)
      This patient shows signs of significant deterioration. Structure your
      response for IMMEDIATE clinical action."

2. Section structure instructions (risk-adaptive order)
   If SOFA ≥ 8.0 → IMMEDIATE ACTIONS listed FIRST
   If SOFA < 8.0 → CURRENT CONDITION listed first, IMMEDIATE ACTIONS last
   (High-risk patients need to see "what to do NOW" before anything else)

3. SOFA score and risk level
   "SOFA Score: 6.2 / 24   (higher = worse organ failure)
    Risk Level: Moderate Risk"

4. Clinical notes (verbatim from clinician input)
   "SpO2 declining despite high-flow O2 therapy. History of COPD..."

5. Latest vital signs with normal ranges (formatted)
   "HR: 122 bpm (normal: 60–100)
    RR: 30 breaths/min (normal: 12–20)
    SpO₂: 84% (normal: 95–100)
    ..."

6. Neurological & stress indicators
   "GCS Eye Opening: 2 (1=No response, 2=To pain, 3=To voice, 4=Spontaneous)
    Stress Score: 8/10 (higher = more distress)"

7. SHAP outputs (key risk factors + feature-level explanations + trends)
   "Key Risk Factors:
    - Low oxygen levels
    - Hypotension
    - Sepsis / Infection

    Feature-Level Explanations (SHAP-derived):
    - SpO₂ = 84.0% → severe hypoxaemia, increasing risk
    - MAP = 62.0 mmHg → blood pressure instability, increasing risk
    - Clinical note contains 'creatinine' → increasing risk
    ...

    Vital Sign Trends (last 20 readings):
    - HR → high & increasing
    - SpO₂ → low & decreasing
    ..."
```

### Why include SHAP in the LLM prompt?

SHAP tells the LLM WHICH features the model found most important for this specific
patient. Without this, the LLM would only see raw numbers and generate generic sepsis
advice. With SHAP, the LLM knows: "the MODEL identified SpO₂ and MAP as the dominant
drivers — emphasise those in your reasoning." This grounds the LLM's narrative in the
model's actual attribution rather than general clinical knowledge.

---

## 14. Step 12 — LLM Self-Consistency Check

**Purpose:** Measure how reliably the LLM responds to this patient's data by sending
the SAME prompt 3 times and measuring agreement across the 3 independent responses.

### Three Independent LLM Calls

```
Input:  final_prompt (string) — the full structured clinical prompt
        GROQ_API_KEY — authentication for the Groq cloud API

Process:
  for each of 3 iterations:
    response_i = Groq API call with:
      model       = "openai/gpt-oss-120b"  (accessed via Groq API)
      temperature = 0.2    (low randomness → similar responses)
      max_tokens  = 1500   (enough for complete 4-section report)
      messages    = [system_message, user_message]

Output:
  responses — list of 3 strings (each is a full clinical assessment)
  main_response = responses[0]  (displayed as the primary report)
```

**Why temperature = 0.2 and not higher?**
Lower temperature = more deterministic sampling. At temperature=1.0, the LLM randomly
explores diverse phrasings even for identical prompts. At 0.2, it consistently selects
the highest-probability clinical reasoning path. For a clinical system where
reproducibility and accuracy are paramount, low temperature is appropriate.

**Why 3 calls?**
Self-consistency checking (from the 2022 Wang et al. paper "Self-Consistency Improves
Chain of Thought Reasoning") uses multiple independent completions to verify that
the model's reasoning is stable. In clinical AI, reliability across multiple runs is
a proxy for factual confidence: if the model always says "vasopressors + antibiotics"
for this patient, that is likely the correct answer. If it says different things each
time, the clinical picture may be genuinely ambiguous or the model may be hallucinating.

### Consistency Metric — Three-Component Score

The consistency score combines three independent measurements:

```
Input:  responses — list of 3 strings

Component 1: TF-IDF Cosine Similarity (20% weight)
  - Vectorise all 3 responses using TF-IDF (ngram 1-2, English stop words removed)
  - Compute pairwise cosine similarity matrix (3×3)
  - Average the 3 off-diagonal similarities
  - Captures: word-level phrasing overlap

Component 2: Clinical Intervention Agreement (50% weight)
  - For each of 7 intervention categories (vasopressors, antibiotics, fluid,
    oxygen, monitoring, labs, renal support) check which synonym terms appear in
    each of the 3 responses
  - For each category: count = number of responses mentioning it
  - Agreement score = max(count, 3-count) / 3
    Examples:
      All 3 mention vasopressors → max(3, 0)/3 = 1.0 (full agreement)
      0 mention vasopressors     → max(0, 3)/3 = 1.0 (all agree it's not needed)
      1 of 3 mentions it         → max(1, 2)/3 = 0.67 (partial disagreement)
  - Average across all 7 categories

Component 3: Clinical Condition Agreement (30% weight)
  - Same majority-agreement formula applied to 8 condition categories:
    septic shock, infection, ARDS, AKI, hypoxemia, hypotension, tachycardia, urgency
  - Captures: do all 3 responses identify the same pathological conditions?

Combined score = 0.20 × tfidf + 0.50 × intervention_agreement + 0.30 × condition_agreement
Output: consistency — float in [0.0, 1.0]
```

**Why this metric is better than pure TF-IDF:**
"Administer norepinephrine" vs "Start vasopressor support" — TF-IDF scores these as
completely different (no shared non-stop words) even though they mean the same
clinical action. The intervention agreement component checks: "does response mention
ANY of [vasopressor, norepinephrine, dopamine, epinephrine]?" — and thus correctly
scores these two phrasings as agreeing. This makes the metric capture clinical
consistency rather than linguistic similarity.

### Reliability Label

```
Input:  consistency — float [0, 1]

  consistency ≥ 0.80 → "High ✅"
    "All 3 LLM responses agree on clinical findings, diagnoses, and interventions."

  consistency ≥ 0.60 → "Moderate ⚠️"
    "Responses agree on core findings with some variation in secondary recommendations."

  consistency < 0.60 → "Low ❌"
    "Significant disagreement across responses on clinical findings or interventions."

Output: reliability label string + coloured banner
```

**Expected consistency scores by patient type:**

| Patient Type | Intervention Agreement | Condition Agreement | TF-IDF | Combined |
|---|---|---|---|---|
| Clear septic shock (HR=122, SpO₂=84, MAP=62) | ~0.90 | ~0.93 | ~0.60 | ~0.84 ✅ |
| Borderline moderate (SOFA 4–6) | ~0.75 | ~0.75 | ~0.55 | ~0.72 ⚠️ |
| Stable patient (HR=90, SpO₂=97, SBP=120) | ~0.90 | ~0.90 | ~0.65 | ~0.85 ✅ |

High-risk patients score high because all 3 LLMs agree on the critical interventions.
Stable patients also score high because all 3 agree nothing urgent is needed.
Borderline patients score moderate because reasonable clinicians could emphasise
different aspects of the clinical picture.

---

## 15. Step 13 — Output Display (Four Tabs)

### Tab 1 — Risk Assessment

```
Inputs: sofa_score, risk_text, vitals_df, vitals_input, NORMAL_RANGES

Displays:
  - SOFA gauge (large numeric display with colour coding)
  - Current vital signs table (with red/green normal range indicators)
  - Historical prediction table (last 50 runs from prediction_history.csv)
  - 7-panel vital sign trend chart (20 readings, green normal range bands)
```

### Tab 2 — Explainability

```
Inputs: top_shap (7 rows), clinical_explanations (7 strings), key_risks (list)

Displays:
  - SHAP feature importance table
    (feature name | original clinical value | SHAP impact ↑↓ | direction)
  - Clinical Interpretations (7 natural-language sentences)
  - Key Risk Factors (warning chips, e.g., "⚠️ Hypotension")
  - SHAP counterintuitive result disclaimer (expander)
```

### Tab 3 — AI Clinical Report

```
Inputs: consistency, main_response, responses (all 3), llm_ok

Displays:
  - Consistency Score metric card (e.g., "0.82")
  - Reliability label metric card (e.g., "High ✅")
  - Coloured reliability banner
  - Full text of main_response (Response 1) — the primary clinical assessment
  - "View all 3 independent LLM responses" expander (all 3 for comparison)
  - Clinical disclaimer (CDSS is a support tool, not a diagnosis)
```

### Tab 4 — Federated Learning

```
Inputs: training_meta (loaded from training_metadata.json)

Displays:
  - FL protocol ASCII diagram (server ↔ 3 hospital clients)
  - Training configuration table (rounds, epochs, batch size)
  - Performance metrics (MAE, R²)
  - Privacy configuration (DP enabled/disabled, ε, δ)
  - Model architecture description
```

---

## 16. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLINICIAN INPUTS (Sidebar)                   │
├──────────────────┬──────────────────┬───────────────────────────┤
│  7 VITAL SIGNS   │  2 CV FEATURES   │   CLINICAL NOTES (text)   │
│  HR, RR, SpO2,   │  GCS Eye (1–4)   │   Free-text nurse/doctor  │
│  Temp, SBP,      │  Stress (0–10)   │   observations            │
│  DBP, MAP        │                  │                           │
└──────┬───────────┴────────┬─────────┴──────────────┬────────────┘
       │                    │                         │
       ▼                    │                         ▼
┌─────────────────┐         │              ┌──────────────────────┐
│  SLIDING WINDOW │         │              │   TF-IDF TRANSFORM   │
│  Append to CSV  │         │              │  tfidf.transform()   │
│  Keep last 20   │         │              │  → 600 float values  │
└──────┬──────────┘         │              └──────────┬───────────┘
       │                    │                         │
       ▼                    ▼                         │
┌─────────────────┐  ┌─────────────────┐              │
│ TREND FEATURES  │  │ LATEST FEATURES │              │
│ (9 statistics)  │  │ (7+2 = 9 vals)  │              │
│ means, std,min  │  │ latest vitals   │              │
└──────┬──────────┘  └──────┬──────────┘              │
       │                    │                         │
       └──────────────┬─────┘                         │
                      ▼                               │
              ┌────────────────┐                      │
              │ FEATURE FUSION │ ◄────────────────────┘
              │ 9 + 9 + 600    │
              │ = 618 features │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │ STANDARD SCALER│
              │ z-score norm.  │
              │ (618 features) │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  FEDERATED DNN │
              │ 618→256→128    │
              │    →64→1       │
              │ (ReLU layers)  │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │  SOFA SCORE    │
              │ clip to [0,24] │
              │ → Risk label   │
              │ → Alert banner │
              └───┬────────┬───┘
                  │        │
        ┌─────────┘        └──────────┐
        ▼                             ▼
┌──────────────┐              ┌───────────────────┐
│    SHAP      │              │   TREND ANALYSIS  │
│ DeepExplainer│              │ get_trend() +     │
│ 300 bg samps │              │ classify_range()  │
│ → 618 attrs  │              │ → 7 trend strings │
└──────┬───────┘              └──────────┬────────┘
       │  top 7 clinical                 │
       │  explanations                   │
       │  + key risks                    │
       └──────────────┬──────────────────┘
                      │
                      ▼
              ┌────────────────────┐
              │  LLM PROMPT BUILD  │
              │ SOFA + risk text + │
              │ vitals + SHAP +    │
              │ trends + notes     │
              │ + risk-adaptive    │
              │ urgency prefix     │
              └────────┬───────────┘
                       │
                       ▼
              ┌────────────────────┐
              │  3× GROQ API CALL  │
              │  temperature=0.2   │
              │  max_tokens=1500   │
              └────────┬───────────┘
                       │ 3 responses
                       ▼
              ┌────────────────────┐
              │ CONSISTENCY METRIC │
              │  20% TF-IDF sim.   │
              │  50% interventions │
              │  30% conditions    │
              │ → score + label    │
              └────────┬───────────┘
                       │
                       ▼
              ┌────────────────────────────────────────┐
              │           4-TAB OUTPUT DISPLAY         │
              │  Tab 1: Risk Assessment + Trend Chart  │
              │  Tab 2: SHAP Explainability            │
              │  Tab 3: AI Clinical Report + Score     │
              │  Tab 4: Federated Learning Info        │
              └────────────────────────────────────────┘
```

---

## 17. Artifacts Loaded at Startup

These 5 files are loaded once at application startup and cached in memory for the
duration of the session (`@st.cache_resource` prevents reloading on every interaction):

| Artifact                      | Size    | Loaded by         | Purpose                              |
|-------------------------------|---------|-------------------|--------------------------------------|
| `models/federated_model.pth`  | ~1.5 MB | `torch.load()`    | DNN weights for inference            |
| `models/scaler.pkl`           | ~8 KB   | `joblib.load()`   | StandardScaler for 618 features      |
| `models/tfidf_vectorizer.pkl` | ~2 MB   | `joblib.load()`   | TF-IDF vocabulary + IDF weights      |
| `models/feature_columns.pkl`  | ~40 KB  | `joblib.load()`   | Ordered list of 618 column names     |
| `models/shap_background.npy`  | ~1.4 MB | `np.load()`       | 300×618 background tensor for SHAP   |

Additionally:
| Artifact                          | Loaded when      | Purpose                          |
|-----------------------------------|------------------|----------------------------------|
| `models/patient_vitals.csv`       | Each prediction  | 20-reading temporal window       |
| `models/prediction_history.csv`   | Each prediction  | Last 50 prediction logs          |
| `models/training_metadata.json`   | At startup       | FL training stats for Tab 4     |

---

## Summary — Feature Count at Each Stage

| Stage                  | Feature Count | Format                        |
|------------------------|---------------|-------------------------------|
| Vital signs (input)    | 7             | Raw floats (bpm, %, mmHg, °C) |
| CV features (input)    | 2             | Raw ints (1–4, 0–10)          |
| Clinical note (input)  | 1             | String (variable length)      |
| After sliding window   | 7 × 20 rows   | DataFrame (temporal buffer)   |
| After trend features   | 9             | Floats (mean, std, min)       |
| After latest features  | 9             | Raw floats + GCS + stress     |
| After TF-IDF           | 600           | Float weights                 |
| **After fusion**       | **618**       | Single flat float vector      |
| After StandardScaler   | 618           | Z-scores (mean≈0, std≈1)      |
| After DNN              | 1             | Raw SOFA prediction (float)   |
| After SHAP             | 618           | Attribution values (floats)   |
| After filtering        | 7             | Top clinical SHAP features    |
| After LLM              | 3             | Full text clinical reports    |
| After consistency      | 1             | Reliability score [0, 1]      |

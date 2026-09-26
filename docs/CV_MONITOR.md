# CV Monitor — Technical Documentation

**File:** `cv_monitor.py`  
**Module:** ICU Clinical Decision Support System — Computer Vision Subsystem  
**Purpose:** Real-time measurement of Stress Score and GCS Eye Score from a camera feed using facial analysis, with no physical contact with the patient.

---

## Table of Contents

1. [What is Stress Score?](#1-what-is-stress-score)
2. [What is GCS Eye Score?](#2-what-is-gcs-eye-score)
3. [Evolution of Methods — Stress Score](#3-evolution-of-methods--stress-score)
4. [Evolution of Methods — GCS Eye Score](#4-evolution-of-methods--gcs-eye-score)
5. [The Current Method: PSPI for Stress](#5-the-current-method-pspi-for-stress)
6. [The Current Method: Rolling Window for GCS](#6-the-current-method-rolling-window-for-gcs)
7. [Why These Methods Are Best](#7-why-these-methods-are-best)
8. [Complete Technical Pipeline](#8-complete-technical-pipeline)
9. [Output Scores and Interpretation](#9-output-scores-and-interpretation)
10. [System Constants and Calibration](#10-system-constants-and-calibration)
11. [Limitations and Clinical Disclaimer](#11-limitations-and-clinical-disclaimer)

---

## 1. What is Stress Score?

### Clinical Definition

The **Stress Score** is a quantitative measure of a patient's psychological and physiological distress level, expressed on a continuous scale from **0.0 to 10.0**.

| Score Range | Label | Clinical Meaning |
|---|---|---|
| 0.0 – 3.4 | **LOW** | Patient appears calm, relaxed, no visible distress |
| 3.5 – 6.4 | **MODERATE** | Noticeable discomfort or agitation, requires monitoring |
| 6.5 – 10.0 | **HIGH** | Severe distress, pain, or agitation — immediate attention required |

In the ICU context, elevated stress scores correlate with:
- Acute pain from procedures or injury
- Ventilator dyssynchrony or respiratory distress
- Agitation from delirium or medication effects
- Anxiety or psychological trauma

### Why Measuring Stress Non-Invasively Matters

Traditional pain and stress assessment in ICU patients relies on:
- Self-report scales (impossible for sedated or intubated patients)
- Physiological signals like heart rate (requires contact sensors)
- Behavioural observation (requires a nurse to be present continuously)

Computer vision stress measurement provides **continuous, non-contact, objective** assessment — the camera monitors the patient 24/7 without any physical attachment.

---

## 2. What is GCS Eye Score?

### Clinical Definition

The **Glasgow Coma Scale (GCS)** is the most widely used clinical tool in neuroscience for measuring a patient's level of consciousness. It was developed by Teasdale and Jennett (1974) and has three components: Eye Opening (E), Verbal Response (V), and Motor Response (M).

The **Eye Opening component** specifically measures brainstem arousal — whether and how the patient opens their eyes. It is scored **1 to 4**:

| Score | Code | Label | Clinical Meaning | Brainstem Status |
|---|---|---|---|---|
| 4 | **E4** | Spontaneous | Eyes open on their own without any stimulus | Arousal mechanisms intact |
| 3 | **E3** | To Voice | Eyes open only when spoken to or given a verbal command | Partial arousal |
| 2 | **E2** | To Pain | Eyes open only when painful pressure is applied | Minimal arousal |
| 1 | **E1** | No Response | Eyes remain completely closed regardless of any stimulus | No brainstem arousal |

### Clinical Significance

- **E4 + Normal verbal + Normal motor** = GCS 15 = fully conscious, alert patient
- **E1 + No verbal + No motor** = GCS 3 = deepest possible unconsciousness (coma)
- **Improvement from E1 → E2 → E3 → E4** = clinical recovery from neurological injury (e.g., after TBI or stroke)
- A single-point change in the eye score can signal significant neurological improvement or deterioration

In an ICU, the eye score is assessed routinely every 1–4 hours by nurses. Our system provides **continuous automated assessment** between nurse assessments.

---

## 3. Evolution of Methods — Stress Score

### Method V1: Geometric Facial Landmark Analysis (Discarded)

**How it worked:**  
The first implementation measured raw geometric distances from MediaPipe's 468 face landmarks:

```
Stress components:
  c_eye   = (ear_baseline - current_EAR) / (ear_baseline × 0.30)
  c_brow  = (brow_baseline - current_brow_dist) / (brow_baseline × 0.35)
  c_blink = (blink_rate - 8) / 22.0

Stress Score = 10 × (0.42 × c_brow + 0.36 × c_eye + 0.22 × c_blink)
```

**Eye Aspect Ratio (EAR)** was computed as:
```
EAR = (||p1–p5|| + ||p2–p4||) / (2 × ||p0–p3||)
```
Where p0–p5 are 6 landmark points around each eye. High EAR = wide open; low EAR = squinting.

**Inner-brow distance** measured how close the inner brow points were — furrowing brings them together.

**Why this was replaced:**

| Problem | Impact |
|---|---|
| Used only 2 muscle groups (brow, eye) | Missed nose wrinkle, lip raise, cheek tension — all key pain indicators |
| Required 5-second personal calibration | Scores stayed at 0.0 until calibration completed |
| No smile correction | A smiling face (same eye squint + cheek movement) read as moderate stress |
| Mouth tension metric was structurally broken | MediaPipe inner-lip landmarks (13, 14) give near-zero gap for any closed mouth, making it always show 100% tension regardless of expression |
| Blink rate unreliable in first 60 seconds | The 60-second sliding window had no data early in the session |

**Root cause of the fundamental failure:**  
The user's natural open-eye EAR was 0.325, but the hardcoded baseline was 0.28. This made `(0.28 - 0.325) / 0.14 = -0.32`, which clips to 0. **All three components clamped to zero — stress score was always 0.0 regardless of expression.**

---

### Method V2: PSPI from MediaPipe Blendshapes (Current — Best)

Replaced the geometric approach with the **Prkachin-Solomon Pain Intensity (PSPI)** index — a clinically validated formula from peer-reviewed pain research.

Full details in [Section 5](#5-the-current-method-pspi-for-stress).

---

## 4. Evolution of Methods — GCS Eye Score

### Method V1: Single-Frame EAR Threshold (Discarded)

**How it worked:**  
Applied fixed population-average EAR thresholds to the current smoothed EAR value:

```
if EAR ≥ 0.22 → E4 (Spontaneous)
if EAR ≥ 0.12 → E3 (To Voice)
if EAR ≥ 0.04 → E2 (To Pain)
else          → E1 (No Response)
```

**Why this was replaced:**

| Problem | Impact |
|---|---|
| Fixed thresholds for all people | A naturally narrow-eyed person (EAR 0.19 when fully open) reads as E3 even when completely awake |
| Single-frame snapshot | One blink in a frame gives E1 — clinically meaningless and unstable |
| No blendshape cross-check | EAR alone cannot distinguish "slightly squinting due to bright light" from "drowsy eyes" |
| Clinical GCS is a behavioral pattern, not a snapshot | True GCS assesses what the patient does over a sustained observation period |
| No confidence flag | Claimed E4 from the very first frame with no data |

---

### Method V2: Rolling Window + Personal Calibration (Current — Best)

Uses a **30-second behavioral window** and **personal eye calibration** guided by the eyeBlink blendshape.

Full details in [Section 6](#6-the-current-method-rolling-window-for-gcs).

---

## 5. The Current Method: PSPI for Stress

### 5.1 What is PSPI?

The **Prkachin-Solomon Pain Intensity (PSPI)** index is a validated clinical metric for automated pain and distress assessment from facial video. It is derived from the **Facial Action Coding System (FACS)** — a comprehensive taxonomy of facial muscle movements developed by Paul Ekman and Wallace Friesen.

**Original publication:** Prkachin & Solomon (2008), *Pain*, 137(2), 242–251.

The formula encodes the muscle groups that pain research has shown to be most reliably activated during distress and pain:

```
PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43

Range: 0 to 16
```

### 5.2 Action Units and Their Meaning

Each term in the formula corresponds to a specific facial Action Unit (AU):

| AU | Name | Muscle | Facial Appearance | Blendshape Used |
|---|---|---|---|---|
| **AU4** | Brow Lowerer | Corrugator supercilii, Depressor supercilii | Inner brows pulled down and together — the "pain frown" | `browDownLeft / browDownRight` |
| **AU6** | Cheek Raiser | Orbicularis oculi (orbital) | Cheeks lifted, creating wrinkles below the eyes | `cheekSquintLeft / cheekSquintRight` |
| **AU7** | Lid Tightener | Orbicularis oculi (palpebral) | Eyes narrowed, lids tightened — the stress squint | `eyeSquintLeft / eyeSquintRight` |
| **AU9** | Nose Wrinkler | Levator labii superioris alaeque nasi | Nose bridge wrinkles, nasal passage narrows | `noseSneerLeft / noseSneerRight` |
| **AU10** | Upper Lip Raiser | Levator labii superioris | Upper lip raised, grimace shape | `mouthUpperUpLeft / mouthUpperUpRight` |
| **AU43** | Eyes Closed | Orbicularis oculi (complete closure) | Eyes shut tight (not just blinking) | `eyeBlinkLeft / eyeBlinkRight > 0.55` |

### 5.3 Why `max(AU6, AU7)` and `max(AU9, AU10)`?

The PSPI formula uses `max()` for these pairs because:
- AU6 and AU7 are anatomically related orbital tension responses — whichever is stronger represents the true orbital involvement
- AU9 and AU10 both indicate grimacing — the dominant one captures the pain grimace correctly
- Taking the maximum prevents double-counting overlapping muscle activations

### 5.4 How MediaPipe Blendshapes Provide AU Estimates

MediaPipe's FaceLandmarker model outputs **52 blendshapes** — learned neural network activations trained on thousands of facial images and motion capture data. Each blendshape is a continuous score from 0.0 to 1.0 that directly estimates the activation strength of specific facial muscle groups.

Unlike raw geometric measurements (distances between landmarks), blendshapes are:
- **Trained, not computed** — the neural network has learned what AU4 looks like across thousands of people with different face shapes
- **Person-independent** — unlike geometric thresholds that fail for narrow-eyed or wide-browed individuals
- **More granular** — captures subtle muscle movements that geometry cannot detect

### 5.5 Smile Correction

A genuine smile causes AU6 (cheek squint) and AU10 (upper lip raise) — the same muscles that pain activates. Without correction, a laughing patient would get a false high PSPI score.

The correction uses the `mouthSmile` blendshape:

```
corrected_PSPI = raw_PSPI × (1 - mouthSmile)
```

- Full smile (mouthSmile = 1.0) → PSPI reduced to 0
- No smile (mouthSmile = 0.0) → PSPI unchanged
- Partial smile → proportional damping

This is critical for correct operation — pain and happiness cannot be confused.

### 5.6 Head Movement Agitation Component

Physical agitation (restlessness) is a validated stress indicator in ICU patients independent of facial expression. It is measured by tracking the nose tip position (landmark 1) across frames:

```python
motion_energy = mean(||nose[t] - nose[t-1]||) over last 30 frames
```

Normalized coordinates (0–1 range) are used, with `MOTION_SAT = 0.020` as the saturation level (2% of frame width per frame = maximum agitation).

Final stress combines facial and agitation signals:
```
frame_stress = 10 × (0.80 × (PSPI / 16) + 0.20 × min(motion / 0.020, 1.0))
```

80% comes from facial pain expression, 20% from physical agitation.

### 5.7 EMA Smoothing

Raw per-frame stress values are noisy. Exponential Moving Average (EMA) smoothing is applied:

```
EMA_new = 0.15 × frame_stress + 0.85 × EMA_prev
```

With α = 0.15:
- **Rise time:** ~7 frames / 0.2 seconds — fast enough to capture a genuine grimace
- **Decay:** gradual — the score doesn't spike from a single facial twitch
- **Advantage over rolling mean:** EMA weights recent frames more heavily, giving faster response to real stress events while staying smooth during sustained calm

### 5.8 Full Stress Score Formula

```
For each frame:
  1. Extract blendshapes from MediaPipe FaceLandmarker
  2. au4  = avg(browDownLeft, browDownRight) × 5
     au6  = avg(cheekSquintLeft, cheekSquintRight) × 5
     au7  = avg(eyeSquintLeft, eyeSquintRight) × 5
     au9  = avg(noseSneerLeft, noseSneerRight) × 5
     au10 = avg(mouthUpperUpLeft, mouthUpperUpRight) × 5
     au43 = 1.0 if avg(eyeBlinkLeft, eyeBlinkRight) > 0.55 else 0.0
  3. smile = avg(mouthSmileLeft, mouthSmileRight)
  4. PSPI = (au4 + max(au6,au7) + max(au9,au10) + au43) × (1 - smile)
  5. motion = mean nose displacement over 30 frames
  6. frame_stress = 10 × (0.80 × PSPI/16 + 0.20 × min(motion/0.020, 1.0))
  7. EMA = 0.15 × frame_stress + 0.85 × EMA_prev
  8. Stress Score = round(min(EMA, 10.0), 1)
```

---

## 6. The Current Method: Rolling Window for GCS

### 6.1 Why a Window Instead of a Snapshot

The clinical GCS definition is inherently behavioral and temporal:
- **E4** does not mean "eyes are open right now" — it means the patient's eyes are **predominantly and spontaneously open**
- **E3** means the patient **opens their eyes in response to your voice** — a sustained behavioural tendency
- **E2** means **brief eye opening in response to pain** — a flicker counts

A single-frame EAR snapshot captures none of this. A patient who blinks in that one frame gets E1. A patient whose eyes are half-open in that frame gets E3 even if they're fully awake.

The solution: assess the **distribution of eye states over 30 seconds**, which matches the clinical observation period.

### 6.2 Personal Calibration

Eye shape varies enormously between individuals:
- Person A (narrow natural eyes): open-eye EAR ≈ 0.19
- Person B (wide natural eyes): open-eye EAR ≈ 0.40

Fixed thresholds (e.g., open ≥ 0.22) call Person A's open eyes "partial" or "closed" — a systematic misclassification that would give a healthy awake patient an E3 score.

**Calibration process** (runs automatically in the first ~3 seconds):

1. For each frame where `eyeBlinkLeft / eyeBlinkRight < 0.30` (blendshape confirms eyes are clearly open):
   - Record the current EAR value
2. After collecting 90 such frames (≈ 3 seconds of clearly-open eyes):
   - Compute `baseline_EAR = median(collected EAR values)`
   - Validate: 0.15 ≤ baseline ≤ 0.50 (reject implausible values)
   - Set personalised thresholds:
     ```
     open_threshold    = baseline_EAR × 0.70
     partial_threshold = baseline_EAR × 0.40
     ```

The `×0.70` and `×0.40` factors mean:
- Eyes at 70% of their normal open aperture → classified as "open"
- Eyes at 40% of normal → classified as "partial" (drowsy/drooping)
- Below 40% → classified as "closed"

This works correctly for all patients regardless of natural eye shape.

**Why eyeBlink blendshape is used for calibration (not just EAR):**  
The blendshape is a trained neural network output that has learned to distinguish "this person's eyes are closed" from "this person just has naturally narrow eyes." It provides a second opinion that makes calibration samples far more reliable.

### 6.3 Eye State Per Frame

Each frame is classified into one of three states using dual cross-check:

```python
if eyeBlink_blendshape > 0.55:
    state = "closed"          # blendshape overrides EAR
elif EAR >= open_threshold:
    state = "open"
elif EAR >= partial_threshold:
    state = "partial"
else:
    state = "closed"
```

The blendshape override is critical: if the EAR says "open" but the neural network detects the eye is closing, the blendshape wins. This prevents misclassification of partial blinks as open eyes.

### 6.4 Rolling Window Assessment

Every frame's eye state is appended to a 30-second sliding window. When the window is full, the GCS score is computed:

```
open_fraction    = (frames where state == "open") / total_frames
any_open_fraction = (frames where state != "closed") / total_frames

if open_fraction    ≥ 0.70 → E4 (Spontaneous)
if any_open_fraction ≥ 0.30 → E3 (To Voice)
if longest_continuous_opening ≥ 0.10 s → E2 (To Pain)
else → E1 (No Response)
```

**Why 70% for E4:**  
A healthy awake person blinks approximately 12–15 times per minute. Each blink lasts ~150–400ms. At 30fps, a blink is 5–12 frames. In 30 seconds at 30fps (900 frames), 15 blinks × 10 frames = 150 frames closed = 83% open. This comfortably exceeds 70%, so normal blinks do not prevent an awake person from scoring E4.

**Why the flicker threshold for E2:**  
Clinical GCS E2 requires that the patient opens their eyes in response to pain — even briefly. The 0.1-second minimum ensures that a single-frame noise spike is not counted as a flicker. A genuine eye flicker at 30fps = at least 3 consecutive open frames, which represents 100ms.

### 6.5 Provisional Flag

The GCS score is marked as **provisional** until the rolling window contains at least 10 seconds of face video. Before that, there is insufficient behavioral data to make a reliable assessment.

Display: `Score: 4 / 4   (provisional — < 10s)`

After 10 seconds: `Score: 4 / 4` (no qualifier — assessment is final for the current window)

---

## 7. Why These Methods Are Best

### 7.1 PSPI vs Geometric EAR/Brow for Stress

| Criterion | Geometric V1 | PSPI V2 (Current) |
|---|---|---|
| Clinical validation | None | Peer-reviewed (Prkachin & Solomon, 2008) |
| Muscle groups covered | 2 (brow gap, eye width) | 6 (brow, orbital, lid, nose, lip, closure) |
| Person-independence | No — fails if EAR > baseline | Yes — absolute blendshape metric |
| Smile false-positive | Yes — smiling looks like stress | No — smile correction eliminates this |
| Agitation component | No | Yes — 20% from head movement |
| Response speed | ~1.5s (45-frame rolling mean) | ~0.2s (EMA α=0.15) |
| Works from first frame | No — 5-second calibration needed | Yes — PSPI is absolute |
| Source | Academic EAR papers (2016) | Clinical pain assessment literature (2008) |

### 7.2 Rolling Window GCS vs Snapshot GCS

| Criterion | Snapshot V1 | Rolling Window V2 (Current) |
|---|---|---|
| Matches clinical GCS definition | No — GCS is behavioral | Yes — window captures behaviour |
| Handles natural eye variation | No — misclassifies narrow eyes | Yes — personal calibration adapts |
| Blink robustness | No — blink → instant E1 | Yes — blinks don't affect open fraction |
| Confidence indicator | None | Provisional flag for first 10s |
| Blendshape cross-check | No | Yes — dual EAR + blendshape |
| Scientific basis | Simple threshold | GCS clinical criteria (Teasdale & Jennett, 1974) |

### 7.3 Why MediaPipe Blendshapes Over Raw Geometry

Raw geometry (measuring pixel distances between landmarks) assumes that the relationship between pixel distances and muscle activation is linear and consistent across all faces. It is not:

- A person with a wider face has a larger absolute brow distance even when furrowing hard
- A person with naturally narrow eyes has a low EAR even when fully awake
- Lighting changes affect apparent landmark positions

Blendshapes are **learned neural network activations** trained on thousands of face captures across diverse demographics, lighting conditions, and facial structures. They output the **muscle activation level directly**, not a raw distance that requires interpretation.

---

## 8. Complete Technical Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  WEBCAM CAPTURE (1280×720 native resolution)                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ raw frame (BGR)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MEDIAPIPE FACELM DETECTION (on native resolution)                     │
│  • 478 normalised 3D face landmarks (x,y,z ∈ [0,1])                  │
│  • 52 blendshape scores (each ∈ [0,1])                                │
│  • output_face_blendshapes=True required                               │
└────────┬──────────────────────────────────┬────────────────────────────┘
         │ landmarks                         │ blendshapes
         ▼                                   ▼
┌─────────────────────┐         ┌────────────────────────────────────────┐
│  EAR COMPUTATION    │         │  PSPI COMPUTATION                      │
│  6 eye landmarks ×  │         │  au4  = browDown × 5                  │
│  2 eyes → average   │         │  au6  = cheekSquint × 5               │
│  EAR value          │         │  au7  = eyeSquint × 5                 │
│  Smoothed: 20-frame │         │  au9  = noseSneer × 5                 │
│  rolling mean       │         │  au10 = mouthUpperUp × 5              │
└──────────┬──────────┘         │  au43 = eyeBlink > 0.55 ? 1 : 0       │
           │                    │  PSPI = au4 + max(au6,au7) +           │
           │                    │         max(au9,au10) + au43           │
           │                    │  × (1 - mouthSmile)  ← smile fix       │
           │                    └──────────────────┬─────────────────────┘
           │                                       │ PSPI score
           │                                       ▼
           │                    ┌────────────────────────────────────────┐
           │                    │  MOTION ENERGY                         │
           │                    │  nose tip tracked across 30 frames     │
           │                    │  mean ||nose[t] - nose[t-1]||          │
           │                    └──────────────────┬─────────────────────┘
           │                                       │ motion
           │                                       ▼
           │                    ┌────────────────────────────────────────┐
           │                    │  STRESS EMA                            │
           │                    │  frame_stress = 10×(0.8×PSPI/16       │
           │                    │                  + 0.2×motion/0.02)   │
           │                    │  EMA = 0.15×frame + 0.85×prev         │
           │                    │  → Stress Score 0.0–10.0              │
           │                    └────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  GCS PERSONAL CALIBRATION (first ~3 seconds)                           │
│  Collect EAR from frames where eyeBlink < 0.30                        │
│  After 90 frames: baseline = median(collected EARs)                   │
│  open_threshold    = baseline × 0.70                                  │
│  partial_threshold = baseline × 0.40                                  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ personal thresholds
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PER-FRAME EYE STATE                                                   │
│  if eyeBlink > 0.55 → "closed"   (blendshape overrides EAR)          │
│  elif EAR ≥ open_th → "open"                                          │
│  elif EAR ≥ part_th → "partial"                                       │
│  else → "closed"                                                       │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ eye_state per frame
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  30-SECOND ROLLING WINDOW                                              │
│  open_frac = frames("open") / total                                   │
│  any_frac  = frames(≠"closed") / total                                │
│                                                                        │
│  if open_frac ≥ 0.70 → E4 Spontaneous                                │
│  if any_frac  ≥ 0.30 → E3 To Voice                                    │
│  if longest_run × frame_s ≥ 0.10s → E2 To Pain                       │
│  else → E1 No Response                                                │
│                                                                        │
│  provisional = observed_seconds < 10.0                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Output Scores and Interpretation

### Stress Score

| Display | Meaning | What to look for clinically |
|---|---|---|
| `0.0` to `3.4` — **LOW** (green) | Calm, no visible distress | Normal sedation, resting state |
| `3.5` to `6.4` — **MODERATE** (amber) | Noticeable discomfort | Procedure discomfort, mild pain |
| `6.5` to `10.0` — **HIGH** (red) | Severe distress | Acute pain, agitation, respiratory distress |
| `Session peak: X.X` | Highest stress reached since monitor started | Useful even if patient has calmed down since |

**Component bars** show which PSPI components are driving the score:
- `Brow Furrow` — AU4, pain frown
- `Eye Tension` — AU6/7, orbital squint
- `Nose / Lip` — AU9/10, grimace

### GCS Eye Score

| Display | Clinical Interpretation |
|---|---|
| `E4 Spontaneous` | Patient is awake and alert — eyes open naturally |
| `E3 To Voice` | Patient opens eyes intermittently or when addressed |
| `E2 To Pain` | Only brief eye flickers observed in 30-second window |
| `E1 No Response` | Eyes remain continuously closed |
| `(provisional — < 10s)` | Less than 10 seconds of data — score may change |
| `[calibrating...]` in title | Personal eye baseline not yet established |
| `(personal)` after thresholds | Personalised to this patient's eye anatomy |
| `(default)` after thresholds | Using population averages — calibration incomplete |

---

## 10. System Constants and Calibration

| Constant | Value | Purpose |
|---|---|---|
| `_SMOOTH_WIN` | 20 frames | EAR rolling-mean window (~0.7s at 30fps) |
| `_EMA_ALPHA` | 0.15 | Stress EMA smoothing factor |
| `_PSPI_MAX` | 16.0 | Maximum raw PSPI score |
| `_BLINK_CLOSED` | 0.55 | eyeBlink blendshape threshold for "eye closed" |
| `_MOTION_SAT` | 0.020 | Nose displacement saturating agitation signal |
| `_FACIAL_W` | 0.80 | 80% facial PSPI, 20% agitation in stress score |
| `_GCS_WIN_S` | 30.0 s | Duration of the GCS rolling window |
| `_GCS_CALIB_N` | 90 frames | Open-eye frames needed for personal calibration |
| `_GCS_CALIB_BLINK` | 0.30 | eyeBlink max for a frame to count as "clearly open" |
| `_GCS_CALIB_EAR_MIN` | 0.15 | Minimum plausible open-eye EAR |
| `_GCS_CALIB_EAR_MAX` | 0.50 | Maximum plausible open-eye EAR |
| `_GCS_OPEN_DEFAULT` | 0.22 | Default open threshold before calibration |
| `_GCS_PART_DEFAULT` | 0.12 | Default partial threshold before calibration |
| `_GCS_OPEN_FRAC` | 0.70 | Minimum open fraction for E4 |
| `_GCS_ANY_FRAC` | 0.30 | Minimum any-open fraction for E3 |
| `_GCS_FLICKER_S` | 0.10 s | Minimum opening duration for E2 |
| `_GCS_MIN_OBS_S` | 10.0 s | Minimum observation for non-provisional GCS |

---

## 11. Limitations and Clinical Disclaimer

### What this system can do
- Provide continuous non-contact monitoring of stress and eye behaviour
- Detect changes in facial expression associated with pain and distress
- Approximate GCS eye behaviour from camera observation alone
- Track session peak stress for clinical review
- Adapt to individual patient anatomy via personal calibration

### What this system cannot do

**GCS limitations:**
- True GCS E3 requires verbal stimulus (calling the patient's name) — the camera cannot apply this
- True GCS E2 requires painful pressure (sternal rub, nail bed pressure) — the camera cannot apply this
- The system can only assess spontaneous eye opening (E4) reliably without stimuli
- E2 and E3 scores are inferences from observed eye behavior, not from applied stimuli

**Stress limitations:**
- Glasses, oxygen masks, or bandages covering the face will reduce accuracy
- Poor lighting significantly degrades MediaPipe detection
- PSPI was validated on populations without facial paralysis — neurological conditions affecting facial muscles may produce false readings
- The system measures distress expression, not internal pain — a stoic patient may have high pain but low expression

### Clinical Use Guidance

> **This system is a decision-support tool, not a diagnostic instrument.**  
> All scores should be interpreted by a qualified clinician alongside other clinical findings.  
> The GCS scores produced by this system are camera-based proxies and must not replace clinical GCS assessment.  
> Any score suggesting high distress (stress > 6.5 or GCS deterioration) should prompt immediate clinical review.

---

*Documentation generated for ICU CDSS Capstone Project*  
*Scoring engine: PSPI (Prkachin & Solomon, 2008) + GCS Rolling Window (Teasdale & Jennett, 1974)*

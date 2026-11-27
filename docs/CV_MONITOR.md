# CV Monitor — Technical Documentation
**Module:** ClinIQ ICU Clinical Decision Support System — Camera (Computer Vision) Monitor
**Status:** Fully implemented and verified
**Last updated:** 2026-11-27
**Purpose:** Measure two values from a webcam using facial analysis, with no contact with the patient: a **Stress Score (0–10)** and a **GCS Eye-Opening score (1–4)**.

> **What changed since the first version of this document.** The monitor used to be a separate Python program (`cv_monitor.py`, OpenCV). **That file no longer exists.** The monitor is now built into the dashboard and runs **entirely in the web browser**: React plus Google's MediaPipe FaceLandmarker compiled to WebAssembly. The scoring formulas are the same ones described here; they were ported one-to-one into JavaScript. Everything in this document describes the current browser implementation, in `frontend/src/components/cv/CVMonitorModal.jsx`.

---

## How to Read This Document

- **Sections 1–2** explain what the two scores mean clinically.
- **Section 3** explains the face-analysis technology (MediaPipe landmarks and blendshapes).
- **Sections 4–5** are the history: what was tried first and why it failed.
- **Sections 6–7** are the heart of the document: the **current stress method (PSPI)** and the **current eye method (30-second window with personal calibration)**, step by step with worked examples.
- **Section 8** explains why these methods were chosen.
- **Sections 9–11** cover how the monitor runs in the browser, what the screen shows, and how the two numbers reach the SOFA model (including an honest measurement of how much they change the result).
- **Sections 12–14** are the constants, the tests, and the limitations.
- **Section 15** is a glossary.

---

## Table of Contents

1. [What is Stress Score?](#1-what-is-stress-score)
2. [What is GCS Eye Score?](#2-what-is-gcs-eye-score)
3. [The Foundation: MediaPipe and Facial Blendshapes](#3-the-foundation-mediapipe-and-facial-blendshapes)
4. [Evolution of Methods — Stress Score](#4-evolution-of-methods--stress-score)
5. [Evolution of Methods — GCS Eye Score](#5-evolution-of-methods--gcs-eye-score)
6. [The Current Stress Method: PSPI](#6-the-current-stress-method-pspi)
7. [The Current GCS Method: Rolling Window with Personal Calibration](#7-the-current-gcs-method-rolling-window-with-personal-calibration)
8. [Why These Are the Best Methods Available](#8-why-these-are-the-best-methods-available)
9. [How It Runs in the Browser — Complete Pipeline](#9-how-it-runs-in-the-browser--complete-pipeline)
10. [What the Screen Shows](#10-what-the-screen-shows)
11. [How the Two Numbers Reach the SOFA Model](#11-how-the-two-numbers-reach-the-sofa-model)
12. [System Constants Explained](#12-system-constants-explained)
13. [Tests](#13-tests)
14. [Limitations and Clinical Disclaimer](#14-limitations-and-clinical-disclaimer)
15. [Glossary](#15-glossary)

---

## 1. What is Stress Score?

### The clinical problem it solves

An ICU patient on a ventilator cannot speak. A patient in delirium may not follow instructions. A heavily sedated patient cannot press a button. Yet all of them may be in significant pain, breathless or agitated, which worsens outcomes if nobody notices.

The Stress Score gives clinicians a continuous, non-contact measure of **visible distress**, updated about once a second from the patient's face.

### Definition and scale

The **Stress Score** is a number from **0.0 to 10.0**:

| Score | Label | Colour | Meaning | Typical ICU causes |
|---|---|---|---|---|
| 0.0 – 3.4 | **LOW** | Green | Calm, comfortable | Effective sedation, resting, comfortable position |
| 3.5 – 6.4 | **MODERATE** | Amber | Noticeable discomfort or agitation; keep watching | Procedures, mild under-treated pain, mild breathlessness |
| 6.5 – 10.0 | **HIGH** | Red | Severe distress, probable acute pain or agitation | Ventilator mismatch, untreated pain, delirium, acute deterioration |

These exact bands (LOW < 3.5 ≤ MODERATE < 6.5 ≤ HIGH) are used in **four places** that must stay in sync: the camera monitor, the dashboard's Stress vital card, the SHAP explanations in `ml_service.py`, and the AI-report rule "Stress ≥ 6.5 must mention pain management or sedation".

### What the score measures

It is driven by facial muscle groups that pain research has shown to activate involuntarily during distress, described by the **Facial Action Coding System (FACS)**, not by how the face "looks" to a person:
- **Brow furrow** (inner brows pulled down and together): the most reliable single sign of pain.
- **Eye squint and cheek raise:** tension around the eye.
- **Nose wrinkle:** the grimace component.
- **Upper-lip raise:** the lip part of a grimace.
- **Forced eye closure:** eyes shut during a pain episode.
- **Restlessness:** head movement, which shows agitation even when the face is held still.

### Why it adds something

Heart rate and blood pressure rise with pain but also with fever, low blood volume, anxiety or exertion, so they are not specific. The Stress Score looks directly at the facial expression of distress, a more direct signal.

---

## 2. What is GCS Eye Score?

### The Glasgow Coma Scale — background

The **Glasgow Coma Scale (GCS)**, developed by Teasdale and Jennett in 1974, is the most widely used measure of consciousness. It has three parts: **Eye opening (E, 1–4)**, **Verbal response (V, 1–5)** and **Motor response (M, 1–6)**, giving 3 (unresponsive) to 15 (fully conscious).

The camera measures only the **Eye-opening part**, because it is the only one visible without touching or speaking to the patient.

### The four levels

| Score | Label | What it means | What it says about the brain |
|---|---|---|---|
| **E4** | Spontaneous | Eyes open on their own | Brainstem arousal intact: the patient keeps themselves awake |
| **E3** | To Voice | Eyes open when spoken to | Partial arousal |
| **E2** | To Pain | Eyes open only to painful pressure | Minimal arousal |
| **E1** | No Response | Eyes stay closed whatever the stimulus | No functional arousal: deep coma or heavy sedation |

### Why one point matters

E4 versus E3 is the difference between "alert" and "semi-conscious, watch closely". E2 versus E1 can signal a neurological emergency. The camera adds **continuous** assessment between nurse checks, so a drop from E4 to E3 at 2 a.m. (between 12 a.m. and 4 a.m. checks) is noticed, and the next SOFA prediction can reflect it.

### An important limitation upfront

True E3 needs a voice stimulus and true E2 needs a painful stimulus. A camera can apply neither. It reliably sees **spontaneous** opening (E4) and **infers** E3 and E2 from eye-opening patterns. Treat E3/E2 as "this patient may need a formal nurse GCS assessment", not as a replacement for one (Section 14).

---

## 3. The Foundation: MediaPipe and Facial Blendshapes

### What MediaPipe does

**MediaPipe FaceLandmarker** is Google's real-time face model. For each video frame it returns two things:
1. **478 face landmarks:** the position of 478 points on the face (eye corners, nose tip, lip edges, brow points…), normalised to 0–1.
2. **52 blendshapes:** each a score from 0.0 to 1.0 for the strength of one facial-muscle action (for example `browDownLeft`, `eyeBlinkRight`, `mouthSmileLeft`).

**Where it runs.** The model runs **in the browser**, using WebAssembly code stored in the project at `frontend/public/mediapipe/wasm/` (six files, loaded from `/mediapipe/wasm`) and the npm package `@mediapipe/tasks-vision`. The model weights file (about **12 MB**, `face_landmarker.task`) is downloaded from Google's storage **the first time** the monitor is opened, so that first launch needs internet. The monitor asks for the GPU where available.

### Landmarks versus blendshapes

- **Landmarks** are positions: "the eye corner is at (0.34, 0.41)". To tell whether an eye is open you must measure distances and pick a threshold. That is the *geometric* approach.
- **Blendshapes** are interpretations by a neural network trained on many faces: "the brow is 45% furrowed". They are largely independent of face shape.

| Situation | Geometric approach | Blendshape approach |
|---|---|---|
| Naturally narrow eyes | Always looks "partly closed" even when wide open | The network has learned what this person's "open" looks like |
| Thick, low brows | Looks "slightly furrowed" even when relaxed | The blendshape reflects actual muscle tension |
| Different lighting | Shadows move landmarks, so distances change | Trained across lighting conditions, more robust |
| Facial hair, partly hidden face | Missing landmarks cause errors | The network fills in from context |

**Key idea:** blendshapes are person-independent measures of muscle activation, while geometric distances depend on each person's anatomy. The monitor uses blendshapes for stress and as a cross-check for eye state, and landmarks only for the eye-opening ratio (EAR) and head movement.

---

## 4. Evolution of Methods — Stress Score

### Version 1: geometric approach (rejected)

The first version computed stress from raw landmark distances:
```
c_eye   = (baseline_EAR − current_EAR) / (baseline_EAR × 0.30)
c_brow  = (baseline_brow − current_brow_dist) / (baseline_brow × 0.35)
c_blink = (blink_rate − 8) / 22.0
Stress  = 10 × (0.42·c_brow + 0.36·c_eye + 0.22·c_blink)
```
where **EAR (Eye Aspect Ratio)** is
```
EAR = (‖p1−p5‖ + ‖p2−p4‖) / (2 × ‖p0−p3‖)      p0…p5 = six landmarks around one eye
High EAR (~0.35) = wide open eye     Low EAR (~0.08) = nearly closed
```

**The fatal flaw: a baseline mismatch.** The formula needed each user's own resting EAR, but a default of 0.28 was hard-coded. The tester's natural open-eye EAR was 0.325. So `(0.28 − 0.325) / 0.14 = −0.32`, which was clipped to 0. The eye component stayed at 0 permanently, the same happened to the brow component, and the blink component had no data in the first 60 seconds. **The stress score was exactly 0.0 whatever the expression.**

| Other problem | Why it matters |
|---|---|
| Only 2 muscle groups | Pain research names 6; ignoring four loses most of the signal |
| No smile correction | A smile and a grimace both squint and furrow, so both scored as stress |
| Broken mouth-tension metric | Inner-lip landmarks 13/14 give near-zero gap for any closed mouth |
| 60-second blink window | Empty at the start, so wrong results early on |
| 5-second personal calibration | The score sat at 0.0 meanwhile, so users thought it was broken |

### Version 2: PSPI from blendshapes (current)

The geometric method was replaced by the clinically validated **Prkachin–Solomon Pain Intensity (PSPI)** index (Section 6).

---

## 5. Evolution of Methods — GCS Eye Score

### Version 1: single-frame EAR threshold (rejected)

The first version compared the current frame's EAR with fixed population thresholds:
```
EAR ≥ 0.22 → E4      EAR ≥ 0.12 → E3      EAR ≥ 0.04 → E2      else → E1
```
It failed for three reasons.

**1. Fixed thresholds ignore anatomy.** A fully awake person with naturally narrow eyes (open EAR 0.19) reads 0.19 < 0.22 → "E3, semi-conscious". A wide-eyed person (0.40) reads E4. No single threshold fits all eye shapes.

**2. A snapshot is not what GCS means.** GCS describes sustained behaviour. A frame during a blink (EAR ≈ 0.03) would read "E1 — no response" for an alert patient, while an unconscious patient with one eye slightly ajar (EAR ≈ 0.08) would read "E2".

**3. No confidence indicator.** It claimed E4 from the very first frame with no data at all.

### Version 2: rolling window with personal calibration (current)

Redesigned to match the clinical definition (Section 7): a **personal calibration** adapts thresholds to each patient's eyes; a **30-second window** assesses what the patient does over time; a **blendshape cross-check** catches blinks; and a **provisional flag** tells the clinician how much data stands behind the number.

---

## 6. The Current Stress Method: PSPI

### 6.1 What PSPI is

The **Prkachin–Solomon Pain Intensity (PSPI)** index is a peer-reviewed pain formula (Prkachin & Solomon, 2008, *Pain* 137(2)), built on the **Facial Action Coding System (FACS)**, Ekman and Friesen's catalogue of every possible facial movement. FACS defines **Action Units (AUs)**: the smallest separately controllable facial-muscle movements (AU4 = brow lowerer, AU43 = eye closure…). PSPI picks the AUs most reliably active in pain:

```
PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43        maximum possible value: 16
```

### 6.2 Each Action Unit

Each blendshape pair is averaged over the left and right side, then (except AU43) multiplied by 5 to give a 0–5 contribution.

| AU | Muscle action | What it looks like | Blendshapes used |
|---|---|---|---|
| **AU4** Brow lowerer | Inner brow pulled down and together | The "furrowed brow", a vertical crease between the eyebrows. The single most reliable pain sign, hard to fake and hard to suppress | `browDownLeft/Right` × 5 |
| **AU6** Cheek raiser | Cheeks lift, lower lid rises | Crow's-feet, skin creasing under the eye (also appears in genuine smiling) | `cheekSquintLeft/Right` × 5 |
| **AU7** Lid tightener | Eyelid tightens | Narrowed, squinting eyes | `eyeSquintLeft/Right` × 5 |
| **AU9** Nose wrinkler | Nostril wing lifts | The wince | `noseSneerLeft/Right` × 5 |
| **AU10** Upper-lip raiser | Upper lip lifts | Part of the grimace (also part of a smile) | `mouthUpperUpLeft/Right` × 5 |
| **AU43** Eye closure | Eyes shut firmly | A forced closure, not a quick blink | average `eyeBlink` > 0.55 → 1.0, else 0.0 |

`max(AU6, AU7)` and `max(AU9, AU10)` take the stronger of two anatomically overlapping muscles instead of adding them, to avoid counting one tension twice. AU43 is all-or-nothing because forced closure either happened or did not.

### 6.3 Worked example

A patient with moderate post-surgical pain; MediaPipe returns:
```
browDown      L 0.48  R 0.51        eyeSquint    L 0.41  R 0.38
cheekSquint   L 0.22  R 0.25        mouthUpperUp L 0.08  R 0.09
noseSneer     L 0.19  R 0.22        eyeBlink     L 0.12  R 0.10
mouthSmile    L 0.03  R 0.02
```
```
AU4  = 0.495 × 5 = 2.48                       AU9  = 0.205 × 5 = 1.03
AU6  = 0.235 × 5 = 1.18                       AU10 = 0.085 × 5 = 0.43
AU7  = 0.395 × 5 = 1.98                       AU43 = 0.11 < 0.55  → 0.0

PSPI (raw)  = 2.48 + max(1.18, 1.98) + max(1.03, 0.43) + 0.0 = 5.49   (of 16)
smile       = (0.03 + 0.02) / 2 = 0.025
PSPI        = 5.49 × (1 − 0.025) = 5.35
```

### 6.4 Smile correction — why it is essential

A genuine smile activates AU6 and AU10, the same muscles PSPI counts as pain. Without a correction, a happy patient could score 6+ out of 16. So the raw PSPI is multiplied by `(1 − mouthSmile)`:
```
mouthSmile 0.0 (no smile)   → PSPI unchanged
mouthSmile 0.5 (half smile) → PSPI halved
mouthSmile 1.0 (full smile) → PSPI = 0
```
(Example: AU6 = 2.0, AU10 = 1.5. With a big smile of 0.85 the score is 3.5 × 0.15 = 0.53, correctly near zero. With no smile, 0.02, it is 3.5 × 0.98 = 3.43, a genuine grimace.) The test suite checks that a full smile cancels the score.

### 6.5 Head-movement (agitation) component

Restlessness is a validated agitation sign: a patient may hold their face still but cannot stop moving. The monitor tracks the **nose tip** (landmark 1, the most stable and always-visible point, unaffected by expressions):
```
Keep the nose-tip position of the last 30 frames.
motion = average distance the nose moved between consecutive frames  (in 0–1 picture units)
agitation = min(motion / 0.020, 1.0)        0.020 = 2% of the picture width per frame = maximum
```
If the face is lost, the nose-position history is **cleared**, so a head that reappears somewhere else is not read as one huge jump that pins agitation at maximum for 30 frames.

### 6.6 Combining and smoothing

```
facial     = min(PSPI / 16, 1.0)
frame_stress = 10 × (0.80 × facial + 0.20 × agitation)          80% face, 20% movement
EMA        = 0.15 × frame_stress + 0.85 × EMA_previous           (the first frame seeds the EMA directly)
Stress Score = min(EMA, 10), rounded to 1 decimal place
```
The 80/20 split reflects that facial expression is the more specific, validated pain signal, while movement is a broader distress signal.

**EMA (exponential moving average)** smooths momentary noise: the current frame counts 15%, the history 85%. With α = 0.15:
```
Sustained pain, frame_stress jumps 2.0 → 8.0:           A one-frame twitch to 8.0, then back to 2.0:
  frame 1  EMA 2.90                                       frame 1  EMA 2.90
  frame 3  EMA 4.32                                       frame 2  EMA 2.77
  frame 7  EMA ≈ 6.0  (≈ 0.23 s at 30 fps)                frame 3  EMA 2.65
  frame 10 EMA ≈ 7.0                                      → barely moved, correctly ignored
```
So a real sustained change registers within about a third of a second, but a flicker does not. EMA also weights recent frames more than a plain rolling mean would, which suits a "what is happening now" signal.

### 6.7 Per-frame stress, step by step

```
Every new camera frame in which a face is found:
 1. Read the 52 blendshapes.
 2. Compute AU4, AU6, AU7, AU9, AU10, AU43 and the smile value.
 3. PSPI = (AU4 + max(AU6,AU7) + max(AU9,AU10) + AU43) × (1 − smile)      range 0–16
 4. motion = mean nose-tip movement over the last 30 frames
 5. frame_stress = 10 × (0.80 × PSPI/16 + 0.20 × min(motion/0.020, 1))     range 0–10
 6. EMA smoothing (α = 0.15), cap at 10, round to 1 decimal
 7. Update the on-screen number, band, colour and the three component bars
 8. Session peak: the highest rounded score since monitoring began, counted
    only after the first 90 face frames (so start-up noise cannot set the peak)
```

---

## 7. The Current GCS Method: Rolling Window with Personal Calibration

### 7.1 Why a window, not a snapshot

Clinical GCS describes sustained behaviour: E4 means the patient's *predominant* behaviour is to keep their eyes open unprompted. One video frame (33 ms) cannot show that, so the monitor watches **30 seconds** and judges the **distribution** of eye states.

### 7.2 Personal calibration

Eye shape varies a lot: an alert person with narrow eyes may have a resting open-eye EAR of 0.19, an alert wide-eyed person 0.42. With a fixed open threshold of 0.22 the first would be scored E3 while fully awake. So the monitor first learns **this** patient's open-eye EAR:

```
During the first frames of video:
  For each frame, compute EAR (average of both eyes).
  Keep the frame's EAR only if the eyeBlink blendshape is below 0.30 (eyes clearly open).

When 90 such frames are collected (about 3 seconds at 30 fps):
  baseline = the median of those EAR values       (median, so a stray blink cannot skew it)
  Valid only if 0.15 ≤ baseline ≤ 0.50            (outside = tracking error)
  open_threshold    = baseline × 0.70
  partial_threshold = baseline × 0.40
  If invalid, keep the defaults 0.22 and 0.12.
```
Example: baseline 0.19 → open ≥ 0.133, partial ≥ 0.076. Baseline 0.42 → open ≥ 0.294, partial ≥ 0.168. Both patients are now judged against their own eyes. The blendshape check makes sure the baseline really is an open-eye state and not a half-closed wide eye.

**Fallback for patients whose eyes stay closed (new).** A patient who really is E1 or E2 never produces 90 clearly-open frames, so calibration would never finish and the monitor could never score the very patients who matter most. After **300 face frames (about 10 seconds)** calibration ends anyway: it uses the personal baseline if enough open frames were found, otherwise the **default** thresholds 0.22 / 0.12. The screen then shows "(default)" instead of "(personal)".

When calibration finishes, the eye-state history logged so far (which used default thresholds) is **discarded** and the 30-second window starts fresh with the final thresholds.

### 7.3 Per-frame eye state

Each frame gets one of three states, with the blendshape given priority:
```
1. If average eyeBlink > 0.55                 → "closed"    (the blendshape overrides the EAR)
2. else if smoothed EAR ≥ open_threshold      → "open"
3. else if smoothed EAR ≥ partial_threshold   → "partial"    (drooping, drowsy)
4. else                                       → "closed"
```
The EAR used here is a **20-frame rolling mean** (about 0.67 s), to smooth frame-to-frame noise.

**Why the blendshape overrides the EAR:** in a fast blink the landmarks are imprecisely tracked, but the blendshape network recognises the blink reliably. Example with open ≥ 0.175, partial ≥ 0.100: a mid-blink frame with EAR 0.12 and eyeBlink 0.64 reads "closed" (the EAR alone would wrongly say "partial"); a drowsy frame with EAR 0.14 and eyeBlink 0.30 correctly reads "partial".

### 7.4 The 30-second window and the score

Every frame's state is added to a window that keeps the last **30 seconds** (older entries are dropped by timestamp). The score comes from three statistics:
```
open_fraction     = frames "open" / all frames
any_open_fraction = frames "open" or "partial" / all frames
longest_run       = longest unbroken stretch of non-"closed" frames

  open_fraction     ≥ 0.70                          → E4  Spontaneous
  else any_open_fraction ≥ 0.30                     → E3  To Voice
  else longest_run × seconds-per-frame ≥ 0.10 s     → E2  To Pain
  else                                              → E1  No Response
```
Seconds-per-frame is measured from the real timestamps (window length ÷ number of frames), so it adapts to the camera's actual speed. An empty window returns E4 "Spontaneous" until data arrives.

**Why 70% for E4.** A healthy awake person blinks about 12–15 times a minute, each blink lasting 150–400 ms (5–12 frames at 30 fps). Over 30 s that is about 900 frames, with about 150 closed, so about 83% are open, comfortably above 70%. A 90% cutoff would mark frequent blinkers E3; 50% would let drowsy, half-closed patients score E4.

**Why 0.10 s for E2.** A real flicker lasts at least 3–4 frames (about 100–130 ms). One noisy frame (33 ms) must not count, so a single-frame twitch is **not** a flicker (a test checks this).

### 7.5 The provisional flag

For the first **10 seconds** of observation the window has too little data, so the score is shown as **provisional**. After 10 seconds the qualifier disappears. Ten seconds is about 300 frames, enough for stable fractions even if half are blinks.

### 7.6 When the patient leaves the picture

If **no face is seen for 3 seconds**, the monitor starts a **fresh observation**: it empties the 30-second window and eye-smoothing queue, resets the stress smoothing, marks the score provisional again, and tells the dashboard to **stop using the last reading**. Otherwise the gap would count as "observed time", and the E2 test (run length × seconds-per-frame) could misfire on it.

---

## 8. Why These Are the Best Methods Available

### 8.1 Stress: PSPI versus the geometric method

| Criterion | Geometric V1 | PSPI V2 (current) |
|---|---|---|
| Clinical validation | None | Peer-reviewed (Prkachin & Solomon, 2008) |
| Muscle groups | 2 | 6 |
| Person independence | No (fails if natural EAR ≠ hard-coded baseline) | Yes |
| Smile false positive | Yes | No (smile multiplier) |
| Agitation (movement) | No | Yes (20%) |
| Response speed | ~1.5 s | ~0.2–0.3 s |
| Works from frame 1 | No (needs 5 s calibration, score stays 0.0) | Yes (PSPI needs no baseline) |
| Fatal-bug risk | Yes (permanent 0.0) | No |

### 8.2 GCS: rolling window versus snapshot

| Criterion | Snapshot V1 | Rolling window V2 (current) |
|---|---|---|
| Matches GCS definition | No | Yes (sustained behaviour) |
| Individual anatomy | No (fixed 0.22) | Yes (personal calibration) |
| Blink robustness | No (one blink → E1) | Yes |
| Confidence indicator | None | Provisional for 10 s |
| Blendshape cross-check | No | Yes |
| Partial vs closed | No | Three states |
| Patients who cannot open their eyes | Not handled | Handled (10-second calibration fallback) |

### 8.3 Why blendshapes beat raw geometry
Geometry fails on anatomical variation, on baseline dependency (a wrong default fails silently, as the stress V1 bug showed) and on lighting. Blendshapes were trained across many faces and lighting conditions, and PSPI needs no personal baseline at all.

---

## 9. How It Runs in the Browser — Complete Pipeline

### Where the code lives

| File | What it does |
|---|---|
| `frontend/src/components/cv/CVMonitorModal.jsx` (1,014 lines) | The whole monitor: model loading, camera, per-frame loop, scoring functions, the pop-up screen |
| `frontend/src/components/cv/CVLivePanel.jsx` | The **right-hand panel** beside the camera: live SOFA, the two camera inputs, the clinical note, the vitals and a timer (Section 10, "The live model panel") |
| `frontend/src/components/cv/CVMonitorFAB.jsx` | The round floating button that opens it |
| `frontend/src/pages/MonitorDashboard.jsx` | Holds the on/off state and the live values; passes them to the prediction loop, and passes the latest prediction, the countdown and the live values to the window so the panel can show them |
| `frontend/public/mediapipe/wasm/` | The six MediaPipe WebAssembly files |
| `frontend/package.json` | Dependency `@mediapipe/tasks-vision` |

The scoring functions `calcEAR`, `pspiFromBlendshapes`, `updateStress`, `eyeState`, `gcsFromWindow` and `stressLabel` are **exported** so they can be unit-tested.

### Opening and starting up
The monitor starts only when the pop-up opens (the purple eye button on the right edge of the monitor page). Startup runs in three stages, each shown on screen:

| Stage | What happens | Screen message |
|---|---|---|
| `loading-model` | Load MediaPipe; download the ~12 MB face model (first launch only) | "Loading AI Model — Downloading MediaPipe FaceLandmarker (~12 MB) — first launch only" |
| `loading-camera` | Ask the browser for the camera (ideal 1280×720, front-facing) | "Starting Camera — Please allow camera access when prompted" |
| `running` | The detection loop runs | The live display |

Settings: one face, video mode, GPU requested, minimum face-detection, face-presence and tracking confidence 0.50 each.

If something fails, the screen shows a message and a **Retry** button: *"Camera permission denied. Please allow camera access in your browser."* · *"No camera found. Please connect a webcam and try again."* · *"Could not download face detection model — internet connection required for first use."* · or the error text itself.

### The per-frame loop
```
Repeat for every NEW camera frame (the loop runs at the screen's refresh rate, 60–120 Hz,
but the camera gives ~30 frames/s, so a repeated frame is skipped: each real frame
counts once towards calibration and the 30-second window):

  1. Run FaceLandmarker on the frame (timestamps strictly increasing; a failed frame is
     skipped, never fatal).
  2. No face?  Clear the nose history. If no face for > 3 s: reset the observation
     (Section 7.6) and tell the dashboard to drop the last reading.
  3. Face found:
       a. Read blendshapes → blink value (average of both eyes).
       b. EAR for each eye → average → push into the 20-frame queue → smoothed EAR.
       c. If calibration is not done: collect open-eye EARs; finish after 90 samples or
          300 face frames (Section 7.2).
       d. Head movement → agitation.
       e. PSPI → stress (EMA) → rounded score, band, colour, session peak, 3 component bars.
       f. Eye state → add to 30-s window → GCS score and provisional flag.
       g. Send a reading to the dashboard (Section 11).
       h. Draw the eye outlines and key points on the video.
```

### The calibration screen
Until calibration finishes, the monitor shows a progress bar ("Calibrating personal eye baseline — keep eyes open and expression neutral") and no scores. Progress is the **slower** of two things: the first 90 face frames, and the eye baseline (or its 300-frame fallback). Two reasons: blinks do not count towards the baseline, and a patient with closed eyes needs the fallback.

### Closing
Closing the pop-up (Esc, the close button or the backdrop) stops the camera, closes the model, and sends `null` to the dashboard so predictions return to the recorded values. Switching patient also closes the monitor and drops the live values: the readings belong to whoever is in front of the camera.

### Privacy
All video analysis happens **in the browser on the user's own device**. No video or image is sent to any server. Only two small numbers (stress and GCS eye) go to the backend, as part of the next prediction. The only network traffic specific to the monitor is the one-time model download from Google.

---

## 10. What the Screen Shows

### Stress Score card (labelled "Stress Score (PSPI)")
| Display | Meaning | Suggested action |
|---|---|---|
| `0.0–3.4` **LOW** (green) | Calm, no visible distress | Routine monitoring |
| `3.5–6.4` **MODERATE** (amber) | Noticeable discomfort or agitation | Review pain relief and comfort measures |
| `6.5–10.0` **HIGH** (red) | Severe visible distress | Immediate clinical review for pain, breathlessness or delirium |
| **Session peak** | Highest score since monitoring began (after the first 90 face frames) | Useful even if the patient has since calmed |

Three **component bars** show what is driving the score (each 0–100%, green below 35%, amber 35–65%, red above 65%):
- **Brow Furrow:** AU4, the pain frown.
- **Eye Tension:** `max(AU6, AU7)`, orbital squint.
- **Nose / Lip:** `max(AU9, AU10)`, the grimace.

A HIGH score driven by Brow Furrow is more likely to be genuine pain than one driven only by Eye Tension (which bright light can cause).

### GCS Eye Score card (labelled "GCS Eye Score (Rolling 30s)")
A scale of four dots E1–E4 with the current one lit, the label and colour (**E4 green, E3 amber, E2 and E1 red**), and:

| Display | Meaning |
|---|---|
| `E4` Spontaneous | Eyes open unprompted: alert |
| `E3` To Voice | Eyes open intermittently: partly aroused |
| `E2` To Pain | Only brief flickers in the window: minimal arousal |
| `E1` No Response | Eyes closed throughout the window |
| amber line "Provisional — observing (< 10s window)" | Less than 10 s of data: the score may still change |
| `(personal)` after the thresholds | Calibrated to this patient's eyes: most accurate |
| `(default)` after the thresholds | Calibration used population defaults (for example a patient whose eyes stay closed): less accurate for non-average eyes |

Below the dots the card shows "Eye Aspect Ratio (EAR)" (the smoothed value, to 3 decimals) and "Open≥ … Partial≥ …" (the two thresholds in use) followed by "(personal)" in green or "(default)" in amber. When the face is lost after calibration, the video shows "No face detected — centre your face in the camera". The footer reads: "PSPI formula · 30s GCS window · Live readings replace dataset GCS Eye & Stress in the next SOFA prediction".

### The live model panel (right-hand side)
Added 2026-11-27. The camera window is now wide: the **camera picture and the two score cards on the left**, and a **vertical panel on the right** (`CVLivePanel`) whose job is to **prove that the camera values go into the model and to show the result**. On a narrow screen the panel wraps below the picture. It has five blocks, top to bottom:

| Block | What it shows |
|---|---|
| **Predicted SOFA (output)** | The score out of 24, the risk level, the 90% range and an ALERT tag when the alert fires. This is the latest prediction from the dashboard |
| **Next SOFA prediction (timer)** | The dashboard's 90-second countdown as `m:ss` with a progress bar, or "Predicting now" while a request is running. Under it, **one line says which values the next prediction will use** (see below) |
| **Input 1 and 2: camera features** | For *stress* and for *GCS eye*: **"Live from camera"** (what the camera measures now) next to **"Used in last SOFA"** (what the backend really received for the last prediction), with a badge |
| **Input 3: clinical note** | The note text for this reading (it becomes the 90 TF-IDF features), or "No clinical notes on record for this reading." |
| **Other inputs: vitals** | HR, RR, SpO₂, Temp, SBP, DBP and MAP, coloured normal or abnormal |

**The badge on each camera input**

| Badge | Meaning |
|---|---|
| **WAITING FOR CAMERA** | There is no live reading yet (still calibrating, no face, or the camera is closed) |
| **SENDS NEXT CYCLE** | The camera has a live value that differs from what the last SOFA used; it goes into the model when the timer ends |
| **IN MODEL ✓** | The live value equals the one the last SOFA used: the camera value **is** in the model result you are looking at |

**The timer line: database or camera?**
- **"At 0 → CAMERA values go into the model"** (purple) while the camera is calibrated and sees a face.
- **"At 0 → DATABASE values go into the model"** (grey) in every other case, including when the camera window is closed.

This is how the dashboard already behaved (Section 11); the panel only makes it visible. The values are read **at the moment the timer ends**, not averaged over the 90 seconds.

**A demo that shows the whole path:** (1) open the camera with no one in front of it: the badges say WAITING FOR CAMERA and the line says DATABASE; (2) face the camera until calibration ends: the line turns purple and the live values appear; (3) wait for the timer to reach zero: the SOFA updates, "Used in last SOFA" now equals the live values, and the badge reads **IN MODEL ✓**; (4) close the camera: the next prediction goes back to the database values. The effect on the SOFA is small (Section 11).

**One thing to know:** while the window is open, its dark backdrop covers the floating camera button, so the button cannot be clicked again. Close the window with Esc, the ✕, or by clicking outside it.

### The floating button
`CVMonitorFAB`: a purple eye icon at the bottom right of the monitor page, second from the bottom (History sits 28 px from the bottom edge, the camera button 88 px, LLM validation 148 px, SOFA validation 208 px). It blinks slowly; when the monitor is active it blinks faster, glows and shows a red live dot, and its label reads "CV Live — Click to Close" (otherwise "Live CV Monitor"). The four floating buttons from bottom to top are History, Camera monitor, LLM validation and SOFA validation.

---

## 11. How the Two Numbers Reach the SOFA Model

### Sending the readings
The monitor sends a reading `{ stress, gcs_eye }` to the dashboard **only when**:
- calibration is finished, **and**
- at least the first 90 face frames have passed, **and**
- at least 10 seconds of eye observation exist (the score is no longer provisional), **and**
- at least 1 second has passed since the last send, **and**
- the values actually changed since the last send.

This throttle matters: sending every frame (30+ per second) re-rendered the whole dashboard that often. Readings taken during calibration are never sent because they are unreliable. When the monitor closes, or the face is lost for 3 seconds, `null` is sent so a stale reading is never reused.

### Using them in the prediction
The dashboard holds the latest live reading and passes it to the 90-second prediction loop. The next `POST /api/predict` then contains the two extra fields `cv_gcs_eye` (must be 1–4) and `cv_stress` (must be 0–10); otherwise the API returns an error. In `ml_service.py`, when the live values are present they **replace** the recorded GCS and stress from the sample row for that prediction. They go in as two of the 108 inputs (`GCS_eye_opening` and `stress_score`), and also reach:
- the **vital cards** (GCS Eye and Stress),
- the **SOFA floor** check (GCS eye = 2 adds 2, = 1 adds 3 to the minimum SOFA),
- the **SHAP** explanations and key risks,
- the **AI report** prompt (the neurological block) and its rules (Stress ≥ 6.5 must mention pain relief or sedation; GCS eye = 1 must mention a neurological assessment),
- the **report cache and reportKey**, so a changed camera value triggers a fresh report.

The vital cards read the values with their own bands: GCS eye is flagged when ≤ 2; Stress follows LOW/MODERATE/HIGH as above.

### How much do the camera values actually change the prediction? (measured)
Honest answer: **only a little.** On 2026-11-24 the same reading of each demo patient was run through the final model with different live camera values (on a copy of the database):

| Patient (recorded GCS / stress) | Recorded | GCS 4, stress 0 | GCS 4, stress 8 | GCS 3, stress 2 | GCS 2, stress 8 | GCS 1, stress 10 |
|---|---|---|---|---|---|---|
| 1 Low Risk (4 / 2.0) | 2.6 | 2.8 | 2.3 | 3.2 | 3.3 | 3.4 |
| 2 Moderate (3 / 6.0) | 3.6 | 3.1 | 2.8 | 3.9 | 3.7 | 3.6 |
| 3 High Risk (2 / 9.0) | 8.1 | 7.8 | 7.7 | 9.2 | 8.3 | 7.7 |

- The predicted SOFA moves by roughly **0.3 to 1.4 points** across the whole range of camera values, and **no risk band or alert flag changed** in any of the 18 cases.
- The effect is **not always in the expected direction.** For Patient 3, "GCS 1, stress 10" gives a *lower* SOFA (7.7) than "GCS 4, stress 0" (7.8) or the recorded values (8.1). This reflects what the model learned from MIMIC-III (for example, high stress often accompanied *lower* SOFA because a distressed patient is awake), not a bug (see `docs/INPUT_OUTPUT.md`, Section 13).
- **Why:** in the training data these two inputs carry modest weight. Stress comes from nurse-recorded pain scores, and 71% of stays have a stress of 0. GCS eye is 4 in 84% of stays. The network therefore relies mostly on the vitals and the note words.

**Takeaway:** the camera monitor is a real, working live measurement and it is shown on the dashboard and used by the SOFA floor check and the AI report rules, but **it should not be presented as a major driver of the predicted SOFA.** Strongly weighting these inputs would need training data where they are better filled in (see Section 14).

---

## 12. System Constants Explained

Every value below is in `CVMonitorModal.jsx`.

| Constant | Value | Why this value |
|---|---|---|
| `SMOOTH_WIN` | 20 frames | EAR smoothing window (≈ 0.67 s at 30 fps): averages out frame noise without lagging real changes |
| `EMA_ALPHA` | 0.15 | Stress smoothing factor: reacts to sustained pain in about 0.3 s, ignores single-frame twitches |
| `PSPI_MAX` | 16.0 | Largest possible raw PSPI (5 + 5 + 5 + 1); the divisor in PSPI/16 |
| `BLINK_CLOSED` | 0.55 | eyeBlink level above which an eye counts as closing. Genuine blinks give 0.7–1.0; squints stay below 0.5 |
| `MOTION_SAT` | 0.020 | Nose movement per frame that counts as maximum agitation: 2% of picture width, about 25 pixels per frame on a 1280-pixel picture |
| `FACIAL_W` | 0.80 | Share of the stress score from the face (the rest, 20%, is head movement) |
| `GCS_WIN_S` | 30.0 s | The GCS window: long enough for sustained behaviour, short enough to follow deterioration |
| `GCS_CALIB_N` | 90 frames | Open-eye samples needed for the personal baseline (≈ 3 s at 30 fps) |
| `GCS_CALIB_BLINK` | 0.30 | A calibration frame counts only if eyeBlink is below this (clearly open; well below the 0.55 closure level) |
| `GCS_CALIB_EAR_MIN` / `MAX` | 0.15 / 0.50 | Plausible range for a personal baseline; outside it is treated as a tracking error |
| `GCS_OPEN_DEFAULT` | 0.22 | Population-average open threshold, used before or without calibration |
| `GCS_PART_DEFAULT` | 0.12 | Population-average "partial" threshold |
| `GCS_OPEN_FRAC` | 0.70 | Share of the window with open eyes needed for E4 (allows normal blinking) |
| `GCS_ANY_FRAC` | 0.30 | Share with open or partly open eyes needed for E3 |
| `GCS_FLICKER_S` | 0.10 s | Shortest continuous opening that counts as E2 (≥ 3 frames) |
| `GCS_MIN_OBS_S` | 10.0 s | Observation needed before the score stops being provisional |
| `CALIB_FACE_N` | 90 frames | Face frames before readings are trusted and the peak starts counting |
| `GCS_CALIB_MAX_FACE_N` | 300 frames (≈ 10 s) | Calibration fallback for patients whose eyes stay closed |
| `SEND_INTERVAL_MS` | 1000 ms | At most one reading per second sent to the dashboard |
| `FACE_LOST_MS` | 3000 ms | No face for this long → fresh observation and stop sending the old reading |
| Personal-threshold multipliers | 0.70 / 0.40 | Open = 70% and partial = 40% of the patient's own baseline EAR |
| Landmark indices | left eye 33,160,158,133,153,144 · right eye 362,385,387,263,373,380 · nose tip 1 · inner brows 107, 336 | The six points per eye used for EAR; the nose tip for movement; the brow points are drawn on the video |

---

## 13. Tests

The camera monitor has **94 automated unit tests** in `tests/cliniq/frontend/unit/components/cv/` (all passing, run with Vitest), plus API tests in the backend and **12 browser tests** that open the window for real. See `docs/CLINIQ_TESTING.md` for the full guide.

| File | Tests | What it checks |
|---|---|---|
| `cvScoring.test.js` | 16 | The scoring maths (below) |
| `CVMonitorModal.test.jsx` | 6 | The pop-up: hidden when closed; shown when open; the supplementary label; the footer states that live readings feed the next SOFA prediction; no score cards before the camera runs; **live readings are cleared when the monitor closes** |
| `CVMonitorModalFlow.test.jsx` | 35 | The **whole start-up and detection flow** with a fake camera, fake face detector and fake clock (below) |
| `CVLivePanel.test.jsx` | 33 | The right-hand panel (below) |
| `CVMonitorFAB.test.jsx` | 4 | The floating button |

**`cvScoring.test.js` in detail:**
- **PSPI (4):** a neutral face gives 0; the formula `AU4 + max(AU6,AU7) + max(AU9,AU10) + AU43` adds up correctly (example 7.5); the maximum is 16; a full smile cancels the score.
- **Stress (4):** the first frame seeds the EMA (`updateStress(16, 0, null)` gives 8); maximum pain plus agitation gives 10; smoothing uses α = 0.15 (gives 1.2); the bands LOW < 3.5 ≤ MODERATE < 6.5 ≤ HIGH (checked at 3.4, 3.5, 6.4, 6.5).
- **EAR (1):** height ÷ width for an open eye (0.3 for the test shape).
- **Eye state (2):** the blink blendshape overrides EAR; thresholds give open, partial and closed.
- **GCS window (5):** E4 at ≥ 70% open (blinks allowed); E3 at ≥ 30% open or partial; E2 for a flicker of at least 0.1 s; E1 when eyes stay closed; a single-frame twitch is not a flicker.

**`CVMonitorModalFlow.test.jsx` in detail** (added 2026-11-27). It drives the real detection loop with a scripted face detector, so no webcam and no 12 MB model are needed:
- **Start-up (9):** the loading messages in order; `NotAllowedError` and `NotFoundError` give the friendly messages; a download failure explains "internet connection required"; any other message passes through; **Retry** starts again.
- **Closing (6):** Esc, ✕ and the backdrop close it; unmounting stops the camera, closes the model and sends `null`; closing during start-up never starts the camera; a camera stream that arrives late is released.
- **Calibration (6):** progress is shown; "centre your face" with no face; **nothing is sent during calibration**; duplicate frames are skipped; a frame where detection throws does not stop the loop.
- **After calibration (8):** the stress and GCS cards appear; **nothing is sent until the 10-second window is full**, then `{stress, gcs_eye}` is; open eyes give E4 and closed eyes give E1 (and that is what is sent); a strained face raises stress; an unchanged reading is not resent; the default thresholds are used when the eyes never open enough to calibrate.
- **Face lost (3):** the warning appears; **after 3 seconds the reading is cleared (`null`)**; a reading that was never sent is never cleared.
- **Panel wiring (2):** the panel receives the prediction, the countdown and the live values.

**`CVLivePanel.test.jsx` in detail** (added): the predicted SOFA, risk, range and ALERT; the timer as `m:ss`, "Predicting now" and the progress bar; the **DATABASE / CAMERA line**; the three camera badges (WAITING FOR CAMERA, IN MODEL ✓, SENDS NEXT CYCLE), each feature decided on its own; the plain-words description of the live reading; the note text and its empty message; the seven vitals.

**API and client tests** (added): `test_api_error_paths.py` sends `cv_gcs_eye` of 0, 5, −1, 99 and `cv_stress` of −0.1, 10.1, 11, −5 to `/api/predict` and expects **400** for each; `test_ml_service_branches.py` proves that **live camera values replace the stored ones and change the model input** (the SOFA differs) and that the prediction is logged with them; `clientCameraFields.test.js` proves `runPredict` sends the camera fields **only when the camera has a value** (a stress of exactly 0 is sent).

**Browser tests** (`cv_live_panel.spec.js`, 12): Chromium is given a synthetic webcam and the model download is answered locally. They check that the window opens with the panel on its right, that the panel shows the predicted SOFA, both camera inputs, the note and the vitals, that **with no camera reading it says DATABASE values will be used**, that Esc and the backdrop close it, and that the floating camera button is covered while the window is open.

**What is still checked only by hand:** the **real** MediaPipe model and a real face. The unit tests use a fake detector, and the browser tests use a synthetic webcam (which never shows a face), so how well the real model reads a real face is judged by eye, not by an automated test.

---

## 14. Limitations and Clinical Disclaimer

### What this system can reliably do
- Continuous, non-contact, real-time monitoring of visible stress and eye behaviour.
- Detect sustained facial changes linked to pain and distress using a validated formula.
- Estimate GCS eye opening for patients with typical facial anatomy and good camera conditions.
- Adapt to an individual's eyes through personal calibration, and still score patients whose eyes stay closed (using defaults).
- Track the session-peak stress for review.
- Keep all video on the user's own device.

### What it cannot do

**The stimulus problem (GCS).** True E3 needs a spoken stimulus and true E2 a painful one; the camera can apply neither. **E4** is assessed reliably; **E3** is inferred from intermittent opening; **E2** from brief flickers and is the least certain. Treat camera E3/E2 as a prompt for a formal nurse GCS assessment.

**Stress limitations**

| Limitation | Impact | When |
|---|---|---|
| Face obstruction | Tracking reduced or impossible | Oxygen masks, bandages, ventilator tubing across the face |
| Poor lighting | Detection degrades | Dark rooms, strong backlight |
| Facial paralysis | PSPI does not reflect pain | Stroke, Bell's palsy |
| Stoic expression | Pain under-estimated | Some patients suppress expression |
| Validation population | PSPI was validated on neurologically normal adults | Children, developmental differences, severe neurological conditions |

**Angle.** FaceLandmarker needs a roughly frontal face; beyond about 45° of head turn, tracking becomes unreliable. Camera placement and patient position matter.

**Practical limits of this implementation**
- **First use needs internet** (about 12 MB model download) and a camera; the browser must allow camera access.
- **One face at a time** (the monitor is set to a single face).
- **The monitor reads the person in front of the camera.** In the demo that is the person at the laptop, not the recorded MIMIC-III patient; the live values then **replace** the recorded ones for that prediction. This demonstrates the live pipeline; it does not mean the camera is watching the demo patient.
- **Modest influence on the prediction** (Section 11): the live values change the predicted SOFA only a little, because the training data had few informative GCS and stress values.
- **The first ~10 seconds are provisional** and nothing is sent to the dashboard until calibration and 10 seconds of observation are done.
- **Browser-only:** there is no longer a standalone desktop version.

### Clinical use guidance

> **This system is a decision-support tool, not a diagnostic instrument.**
>
> All Stress and GCS Eye scores must be interpreted by a qualified clinician alongside other findings and direct patient assessment.
>
> The GCS scores are camera-based proxies from observed behaviour. They do not replace a clinical GCS assessment with verbal and pain stimuli by a nurse or physician.
>
> A high-distress score (Stress ≥ 6.5) or a GCS deterioration (E4 → E3 or lower) should prompt immediate clinical review and direct assessment.
>
> The output is one input to the SOFA prediction model, which is itself decision support requiring clinical judgment before any action.

---

## 15. Glossary

| Term | Plain meaning |
|---|---|
| **MediaPipe FaceLandmarker** | Google's face-analysis model; gives 478 landmarks and 52 blendshapes per frame |
| **WebAssembly (WASM)** | A format that lets the face model run at near-native speed inside a web page |
| **Landmark** | A tracked point on the face (eye corner, nose tip…) |
| **Blendshape** | A 0–1 score for the strength of one facial-muscle action |
| **EAR (Eye Aspect Ratio)** | Eye height ÷ eye width from six landmarks; high = open, low = closed |
| **Action Unit (AU)** | One facial-muscle movement in FACS (AU4 = brow lowerer…) |
| **FACS** | Facial Action Coding System: the catalogue of facial movements |
| **PSPI** | Prkachin–Solomon Pain Intensity: pain score from four AU groups, 0–16 |
| **EMA** | Exponential moving average: a smoothing that weights recent frames more |
| **GCS** | Glasgow Coma Scale of consciousness (the monitor measures only the Eye part, E1–E4) |
| **Calibration** | Learning a patient's own open-eye EAR so thresholds fit their eyes |
| **Provisional** | A score based on less than 10 seconds of data |
| **Session peak** | Highest stress score since monitoring began |
| **Median** | The middle value of a sorted list; not thrown off by a few outliers |

---

*Documentation for the ClinIQ ICU CDSS capstone — last updated 2026-11-27.*
*Stress scoring: PSPI formula (Prkachin & Solomon, 2008, Pain 137(2)).*
*GCS Eye scoring: Teasdale & Jennett (1974), adapted for continuous camera-based assessment.*
*Facial analysis: MediaPipe FaceLandmarker (Google) with 52 blendshapes, running in the browser.*
*Implementation: `frontend/src/components/cv/CVMonitorModal.jsx` and `CVLivePanel.jsx`; 94 Vitest tests, 12 browser tests and API tests; measured effect on SOFA: about 0.3–1.4 points, no band or alert change (Section 11).*

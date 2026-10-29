# CV Monitor — Technical Documentation
**Module:** ICU Clinical Decision Support System — Computer Vision Subsystem
**Status:** Fully implemented and verified
**Purpose:** Real-time measurement of Stress Score and GCS Eye Opening Score from a camera feed using facial analysis — no physical contact with the patient required.

---

## What This Module Does and Why It Exists

In an ICU, clinicians need to continuously monitor two things about every patient that are extremely difficult to measure automatically:

1. **How much pain or distress is the patient experiencing?**
2. **What is the patient's level of consciousness?**

Traditional methods have serious limitations:
- Pain assessment requires the patient to self-report ("rate your pain 1–10") — impossible for sedated, intubated, or unconscious patients
- Level of consciousness (GCS) is assessed by nurses every 1–4 hours — it misses rapid changes between assessments
- Both require a human to be physically present at the bedside

The CV Monitor module solves this with a **camera**. By continuously analysing the patient's face using MediaPipe's facial AI model, it computes two scores in real time:

- **Stress Score (0.0–10.0):** How distressed or in pain the patient appears, based on their facial muscle movements
- **GCS Eye Opening Score (1–4):** How spontaneously and frequently the patient opens their eyes, a key indicator of consciousness level

Both scores feed directly into the 108-feature vector that the SOFA prediction model uses, adding a computer-vision dimension that no other ICU monitoring system in the market currently provides.

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
9. [Complete Technical Pipeline](#9-complete-technical-pipeline)
10. [Output Scores and Dashboard Display](#10-output-scores-and-dashboard-display)
11. [System Constants Explained](#11-system-constants-explained)
12. [Limitations and Clinical Disclaimer](#12-limitations-and-clinical-disclaimer)

---

## 1. What is Stress Score?

### The clinical problem it solves

An ICU patient under mechanical ventilation cannot speak. A patient in delirium may not be able to follow instructions. A heavily sedated patient cannot press a button. Yet all of these patients may be experiencing significant pain, respiratory distress, or psychological agitation — conditions that worsen outcomes and increase SOFA scores if left unaddressed.

The Stress Score gives clinicians a continuous, objective, non-contact measurement of visible distress, updated every second from the patient's facial expression.

### Definition and scale

The **Stress Score** is a continuous number from **0.0 to 10.0**, representing the severity of observable physiological and psychological distress:

| Score Range | Label | Colour | Clinical Meaning | Typical Causes in ICU |
|---|---|---|---|---|
| 0.0 – 3.4 | LOW | Green | Patient appears calm, comfortable, relaxed | Effective sedation, resting state, comfortable positioning |
| 3.5 – 6.4 | MODERATE | Amber | Noticeable discomfort or agitation, needs monitoring | Procedural discomfort, mild inadequate analgesia, mild dyspnoea |
| 6.5 – 10.0 | HIGH | Red | Severe distress, probable acute pain or agitation | Ventilator dyssynchrony, untreated pain, ICU delirium, acute deterioration |

### What the score actually measures

The score is driven by specific facial muscle groups that pain and stress research has shown to activate involuntarily and consistently across people during distress — even when a patient is trying to appear calm. This is not based on subjective appearance; it is based on the activation of specific Action Units (AU) in the Facial Action Coding System (FACS), which were identified through decades of controlled pain research.

Key muscle groups that drive the score:
- **Brow furrow (inner brow pulled down and together):** The most reliable single indicator of pain
- **Eye squinting and cheek raising:** Orbital tension around the eye, activated by pain
- **Nose wrinkling:** The grimace component of pain expression
- **Upper lip raising:** The lip component of a pain grimace
- **Tight eye closure:** Eyes shut during a pain episode
- **Physical restlessness:** Head movement indicating agitation, not just facial expression

### Why this matters more than traditional monitoring

Traditional vital signs (heart rate, blood pressure) can be elevated by pain — but they are not specific. A high heart rate could mean pain, fever, hypovolaemia, anxiety, or normal exertion. The Stress Score is specifically targeted at facial expression of distress, providing a signal that is more direct and interpretable.

---

## 2. What is GCS Eye Score?

### The Glasgow Coma Scale — background

The **Glasgow Coma Scale (GCS)** was developed by Teasdale and Jennett in 1974 and is the most widely used tool in medicine for measuring a patient's level of consciousness. It is a composite score made up of three independent components:

- **Eye Opening (E):** Does the patient open their eyes? When? Scored 1–4.
- **Verbal Response (V):** Can the patient speak coherently? Scored 1–5.
- **Motor Response (M):** Can the patient follow commands and respond appropriately? Scored 1–6.

The full GCS ranges from 3 (completely unresponsive) to 15 (fully conscious).

The CV Monitor measures only the **Eye Opening component**, because it is the only one accessible from a camera without applying stimuli to the patient. Eyes can be observed continuously and non-invasively. Voice and motor responses require interaction.

### The four GCS Eye Opening levels

| Score | Label | What it means | What it tells us about the brain |
|---|---|---|---|
| **E4** | Spontaneous | Eyes open naturally, without any prompting | Arousal mechanisms in the brainstem are intact — the patient's brain is keeping itself awake |
| **E3** | To Voice | Eyes open only when someone speaks to the patient | Partial arousal — the brainstem can respond to external stimuli but cannot maintain arousal independently |
| **E2** | To Pain | Eyes open only when painful pressure is applied (in clinical practice: sternal rub, nail bed pressure) | Minimal arousal — only the strongest stimuli penetrate |
| **E1** | No Response | Eyes remain closed regardless of any stimulus | No functional brainstem arousal — deep coma or sedation |

### Why a one-point change in GCS eye score matters

In clinical practice, the difference between E3 and E4 is the difference between "patient is semi-conscious and needs close monitoring" and "patient is alert and oriented." The difference between E2 and E1 can signal brainstem herniation — a neurological emergency.

The camera provides **continuous automated assessment** between nurse assessments. If a patient deteriorates from E4 to E3 at 2:00 AM between the scheduled 12:00 AM and 4:00 AM nurse assessments, the system detects it and the SOFA score updates immediately — potentially triggering a clinical alert before the nurse's scheduled check.

### An important limitation to understand upfront

True clinical GCS E3 (eyes open to voice) requires a nurse to call the patient's name. True GCS E2 (eyes open to pain) requires physical pressure stimuli that a camera cannot apply. Our system can reliably assess **spontaneous eye opening (E4)** and makes reasonable inferences about E3 and E2 from observed eye behaviour patterns. This is explicitly acknowledged as a limitation in Section 12.

---

## 3. The Foundation: MediaPipe and Facial Blendshapes

Before explaining the algorithms, it is essential to understand the technology they are built on — because the choice of MediaPipe blendshapes over raw geometry is a core design decision that determines why the current methods work when earlier approaches failed.

### What MediaPipe does

**MediaPipe FaceLandmarker** is Google's real-time facial analysis AI model. When given a video frame, it outputs two types of data:

1. **478 face landmarks** — the 3D position (x, y, z) of 478 specific points on the face (corners of the eyes, tip of the nose, edges of the lips, points along the brow, etc.), all normalised to a [0, 1] coordinate space relative to the face bounding box.

2. **52 blendshapes** — learned neural network activations, each a continuous score from 0.0 to 1.0, representing the strength of a specific facial muscle action.

### Landmarks vs. blendshapes — the crucial difference

**Landmarks** give you raw positions: "the left eye corner is at position (0.34, 0.41)." To know whether the eye is open, you compute the distance between the upper and lower eyelid landmarks and compare it to some threshold. This is the geometric approach.

**Blendshapes** give you interpretations: "the left eye is 73% squinting" or "the brow is 45% furrowed downward." These are output by a neural network that was trained on thousands of people's faces in controlled expressions, and it has learned to output muscle activation levels directly — regardless of face shape, eye size, or lighting.

**Why blendshapes are better for this application:**

| Situation | Geometric approach | Blendshape approach |
|---|---|---|
| Person with naturally narrow eyes | Narrow eye gap → always reads as "partially closed" even when wide open | Neural network learned: this is what THIS person's "open" looks like → correctly reads as open |
| Person with thick brows | Brow landmarks close together even when relaxed → always reads as "slightly furrowed" | Neural network: brow furrow blendshape reflects actual muscle tension, not resting anatomy |
| Different lighting angles | Shadows change apparent landmark positions → wrong distances | Neural network trained across lighting conditions → more robust |
| Facial hair or partially obscured face | Missing landmarks cause errors | Neural network fills in from context |

The key insight: **blendshapes are person-independent absolute measures of muscle activation, while geometric distances are person-dependent relative measures of facial anatomy.**

---

## 4. Evolution of Methods — Stress Score

Understanding why the current method was chosen requires understanding what was tried first and why it failed.

### Version 1: Geometric Approach (Rejected)

The first implementation computed stress from raw geometric distances between face landmarks:

```
Stress components:
  c_eye   = (baseline_EAR − current_EAR) / (baseline_EAR × 0.30)
  c_brow  = (baseline_brow − current_brow_dist) / (baseline_brow × 0.35)
  c_blink = (blink_rate − 8) / 22.0

Stress Score = 10 × (0.42 × c_brow + 0.36 × c_eye + 0.22 × c_blink)
```

Where **Eye Aspect Ratio (EAR)** is:
```
EAR = (upper_lid_to_lower_lid_distance) / (inner_canthus_to_outer_canthus_distance)

More specifically:
  EAR = (||p1–p5|| + ||p2–p4||) / (2 × ||p0–p3||)
  where p0–p5 are 6 landmark points around each eye
  High EAR (e.g. 0.35) = wide open eye
  Low EAR (e.g. 0.08) = nearly closed eye
```

**The fatal flaw — calibration baseline mismatch:**

The formula required a personal baseline EAR for each user. The system had a hardcoded default baseline of 0.28. The person testing the system had a natural open-eye EAR of 0.325 (their eyes were naturally slightly wider than average).

When the formula computed `(0.28 − 0.325) / 0.14 = −0.32`, this negative value was clipped to 0. The eye component was permanently 0. The same happened to the brow component. The blink component had no data in the first 60 seconds. **Result: stress score was always exactly 0.0 regardless of any expression.**

**Additional problems beyond the baseline bug:**

| Problem | Why It Matters |
|---|---|
| Only 2 muscle groups measured (brow, eye) | Pain research identifies 6 distinct muscle groups — ignoring 4 of them misses most of the signal |
| No smile correction | A smiling patient and a pain-grimacing patient both squint and furrow → both get high stress scores |
| Mouth tension metric was broken | MediaPipe inner-lip landmarks 13 and 14 give near-zero gap for any closed mouth, regardless of tension |
| Blink rate used 60-second window | In the first 60 seconds, the window has no data → always returns wrong results |
| Required 5-second personal calibration | Score stayed at 0.0 during calibration, misleading users into thinking the system was not working |

---

### Version 2: PSPI from MediaPipe Blendshapes (Current Implementation)

The geometric approach was completely replaced with the **Prkachin-Solomon Pain Intensity (PSPI)** index — a clinically validated, peer-reviewed formula that measures pain intensity directly from facial muscle activations.

This is explained in full detail in Section 6.

---

## 5. Evolution of Methods — GCS Eye Score

### Version 1: Single-Frame EAR Threshold (Rejected)

The first GCS implementation applied fixed population-average EAR thresholds to the EAR value computed from the current frame:

```
if EAR ≥ 0.22 → GCS Eye = 4 (Spontaneous)
if EAR ≥ 0.12 → GCS Eye = 3 (To Voice)
if EAR ≥ 0.04 → GCS Eye = 2 (To Pain)
else           → GCS Eye = 1 (No Response)
```

**Why this fails fundamentally:**

**Problem 1 — Fixed thresholds ignore individual anatomy:**

```
Person A (naturally narrow eyes, fully awake and alert):
  Open-eye EAR = 0.19
  Fixed threshold for E4 = 0.22
  System reads: 0.19 < 0.22 → E3 (To Voice)
  Reality: Person A is fully awake, looking around, completely alert
  Result: Healthy awake person misclassified as semi-conscious
```

```
Person B (naturally wide eyes, same situation):
  Open-eye EAR = 0.40
  System reads: 0.40 ≥ 0.22 → E4 (Spontaneous) ✓
  Result: Correct, but only because this person's anatomy happened to fit
```

The same threshold that works for Person B misclassifies Person A. There is no single EAR threshold that works across all human eye shapes.

**Problem 2 — Snapshot misses clinical GCS definition:**

GCS is a behavioural assessment, not a measurement of the position of your eyelids at a single moment in time.

```
Scenario: Patient is fully awake but blinks during the exact frame being analysed
  EAR during blink ≈ 0.03  (lids nearly touching)
  System reads: 0.03 < 0.04 → E1 (No Response)
  Reality: Patient is conscious and alert, just happened to blink

Scenario: Patient is unconscious but one eye is slightly ajar
  EAR ≈ 0.08 (eyes not fully closed)
  System reads: 0.08 ≥ 0.04 → E2 (To Pain)
  Reality: Patient has no functional eye opening behaviour
```

Clinical GCS is defined over a **sustained observation period** with applied stimuli, not over a single moment. The snapshot approach violates this fundamental definition.

**Problem 3 — No confidence indicator:**

The system claimed E4 from the very first frame with zero data. A single blurry or poorly lit initial frame could produce a confident but completely wrong score.

---

### Version 2: Rolling Window with Personal Calibration (Current Implementation)

Completely redesigned to match the clinical definition of GCS. The key innovations are:
- **Personal calibration** adapts thresholds to each patient's individual eye anatomy
- **30-second behavioral window** assesses what the patient does over time, not at one instant
- **Blendshape cross-check** validates EAR-based classifications with neural network output
- **Provisional flag** communicates confidence level to clinicians

This is explained in full detail in Section 7.

---

## 6. The Current Stress Method: PSPI

### 6.1 What PSPI Is and Where It Comes From

The **Prkachin-Solomon Pain Intensity (PSPI)** index is a clinical pain assessment formula published in peer-reviewed pain research (Prkachin & Solomon, 2008, *Pain*, 137(2), 242–251). It was derived from the **Facial Action Coding System (FACS)** — a comprehensive, empirically-derived taxonomy of all possible human facial movements, created by psychologist Paul Ekman and Wallace Friesen after analysing thousands of hours of facial video.

FACS defines **Action Units (AUs)** — the smallest independently controllable facial muscle movements, each coded with a number. For example, AU1 is inner brow raise, AU4 is brow lowerer, AU43 is eye closure. PSPI identifies the six specific AUs that pain research has shown to be the most reliably activated during pain and distress.

**The formula:**
```
PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43

Maximum possible value: 16
```

Think of this as a checklist of the most pain-specific facial muscle actions — with a maximum score when all of them activate simultaneously.

---

### 6.2 Each Action Unit Explained

**AU4 — Brow Lowerer (most important pain indicator)**

The corrugator supercilii and depressor supercilii muscles pull the inner brow downward and inward, creating the characteristic V-shaped frown seen in pain. This is the single most reliable indicator of pain across all facial expression research. It is very difficult to fake and almost impossible to suppress when experiencing genuine acute pain.

```
What it looks like: The space between the eyebrows narrows and a vertical
                    crease forms at the centre of the forehead.
                    Think of the expression "furrowed brow."

MediaPipe blendshape: browDownLeft, browDownRight
Scaling: multiplied by 5 to bring the 0–1 blendshape range to a 0–5 contribution
```

---

**max(AU6, AU7) — Orbital Eye Tension**

Two separate muscles around the eye can both activate in pain, but they are anatomically related and often co-activate. The `max()` takes the stronger of the two:

- **AU6 — Cheek Raiser:** The orbital portion of orbicularis oculi lifts the cheeks, creating crow's feet wrinkles and raising the lower eyelid. Seen in genuine smiling AND in pain.
- **AU7 — Lid Tightener:** The palpebral portion of orbicularis oculi tightens the eyelid, creating a narrowed, squinting appearance characteristic of the stress squint.

```
What it looks like: Eyes narrowed and tightened, cheeks slightly lifted,
                    skin creasing below the eyes.

MediaPipe blendshapes: 
  AU6 → cheekSquintLeft, cheekSquintRight
  AU7 → eyeSquintLeft, eyeSquintRight

Why max() instead of sum():
  These muscles partially overlap in their anatomy.
  Taking max() prevents double-counting when both activate together.
  It picks the dominant orbital tension signal.
```

---

**max(AU9, AU10) — Nose-Lip Grimace**

The pain grimace around the nose and upper lip:

- **AU9 — Nose Wrinkler:** Levator labii superioris alaeque nasi pulls the nostril wing up and wrinkles the nose bridge — the characteristic "wincing" expression during sharp pain.
- **AU10 — Upper Lip Raiser:** Levator labii superioris raises the upper lip, contributing to the grimace shape. Note: this is the same muscle that activates in a genuine smile, which is why smile correction (see 6.4) is critical.

```
What it looks like: Nose wrinkled, upper lip slightly raised,
                    creating the "grimace" shape.

MediaPipe blendshapes:
  AU9  → noseSneerLeft, noseSneerRight
  AU10 → mouthUpperUpLeft, mouthUpperUpRight

Why max() again:
  Same reasoning as AU6/7 — anatomically adjacent muscles,
  dominant signal is more informative than sum.
```

---

**AU43 — Eye Closure (pain episode indicator)**

During acute intense pain, the eyes often shut tight — not a normal blink, but a sustained or forced closure. This is distinct from normal blinking (which lasts 150–400ms) and represents a pain episode.

```
What it looks like: Eyes shut firmly, not a quick blink.

MediaPipe blendshape: eyeBlinkLeft, eyeBlinkRight
Threshold: average eyeBlink > 0.55 → AU43 = 1.0 (eyes closing detected)
           otherwise → AU43 = 0.0

This is binary (0 or 1), not scaled, because forced eye closure
is a qualitative event — it either happened or it did not.
```

---

### 6.3 Putting the Formula Together: A Worked Example

**Scenario: A patient is experiencing moderate post-surgical pain**

MediaPipe outputs these blendshape values:
```
browDownLeft = 0.48,  browDownRight = 0.51
cheekSquintLeft = 0.22, cheekSquintRight = 0.25
eyeSquintLeft = 0.41, eyeSquintRight = 0.38
noseSneerLeft = 0.19, noseSneerRight = 0.22
mouthUpperUpLeft = 0.08, mouthUpperUpRight = 0.09
eyeBlinkLeft = 0.12, eyeBlinkRight = 0.10
mouthSmileLeft = 0.03, mouthSmileRight = 0.02
```

**Step 1: Compute AUs**
```
AU4  = avg(browDownLeft, browDownRight) × 5
     = avg(0.48, 0.51) × 5 = 0.495 × 5 = 2.48

AU6  = avg(cheekSquintLeft, cheekSquintRight) × 5
     = avg(0.22, 0.25) × 5 = 0.235 × 5 = 1.18

AU7  = avg(eyeSquintLeft, eyeSquintRight) × 5
     = avg(0.41, 0.38) × 5 = 0.395 × 5 = 1.98

AU9  = avg(noseSneerLeft, noseSneerRight) × 5
     = avg(0.19, 0.22) × 5 = 0.205 × 5 = 1.03

AU10 = avg(mouthUpperUpLeft, mouthUpperUpRight) × 5
     = avg(0.08, 0.09) × 5 = 0.085 × 5 = 0.43

AU43 = avg(eyeBlinkLeft, eyeBlinkRight) = avg(0.12, 0.10) = 0.11
     → 0.11 < 0.55 threshold → AU43 = 0.0  (eyes not closed)
```

**Step 2: Compute raw PSPI**
```
PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
     = 2.48 + max(1.18, 1.98) + max(1.03, 0.43) + 0.0
     = 2.48 + 1.98 + 1.03 + 0.0
     = 5.49  (out of 16)
```

**Step 3: Apply smile correction**
```
smile = avg(mouthSmileLeft, mouthSmileRight) = avg(0.03, 0.02) = 0.025
corrected_PSPI = 5.49 × (1 − 0.025) = 5.49 × 0.975 = 5.35
(almost no correction — patient is not smiling)
```

---

### 6.4 Smile Correction — Why It Is Critical

A genuine Duchenne smile activates AU6 (cheek squinting) and AU10 (upper lip raising) — the exact same muscles that PSPI uses as pain indicators. Without correction, a patient who is genuinely happy or laughing could score a PSPI of 6+ out of 16 — a falsely elevated stress reading.

The correction multiplies raw PSPI by `(1 − mouthSmile)`:

```
mouthSmile = 0.0  (no smile)   → PSPI unchanged
mouthSmile = 0.5  (half smile) → PSPI × 0.5  (halved)
mouthSmile = 1.0  (full smile) → PSPI × 0.0  (zeroed out)
```

**Example:**
```
Patient A: AU6 = 2.0, AU10 = 1.5, mouthSmile = 0.85
  Meaning: cheeks squinting + upper lip raised because of a big smile
  Raw PSPI (AU portion) = 2.0 + 1.5 = 3.5
  With smile correction: 3.5 × (1 − 0.85) = 3.5 × 0.15 = 0.53
  Result: near-zero stress score → correct (this person is happy, not in pain)

Patient B: AU6 = 2.0, AU10 = 1.5, mouthSmile = 0.02
  Meaning: same muscle activation but no smile → genuine pain grimace
  Raw PSPI (AU portion) = 3.5
  With smile correction: 3.5 × (1 − 0.02) = 3.5 × 0.98 = 3.43
  Result: meaningful stress score → correct (this person is in pain)
```

---

### 6.5 Head Movement Agitation Component

Facial expression is not the only indicator of distress. Physical restlessness — constant head movement, repositioning — is a validated behavioural indicator of agitation in ICU patients. A patient who is physically agitated may suppress their facial expression but cannot stop moving.

The agitation component is computed by tracking the movement of the nose tip (landmark 1) across frames:

```
For each consecutive pair of frames:
  displacement = Euclidean distance between nose tip positions
                 in normalised coordinates [0, 1]

Motion energy = mean displacement over the last 30 frames

Saturation level: 0.020 (2% of the frame width per frame)
  → Any motion above 0.020 per frame is considered maximum agitation
```

**Why the nose tip specifically?**
The nose tip is the most stable and accurately tracked facial landmark — it is in the centre of the face, always visible unless completely side-on, and is not affected by expression changes (it does not move like eyebrows or lips do during expressions).

**Final stress score combination:**
```
frame_stress = 10 × (0.80 × PSPI/16 + 0.20 × min(motion/0.020, 1.0))

80% of the signal comes from facial pain expression (PSPI)
20% comes from physical agitation (head movement)
```

The 80/20 weighting was chosen because facial expression is a more specific pain indicator (validated by PSPI research), while motion is a broader distress signal (includes discomfort from positioning, restlessness, etc.).

---

### 6.6 EMA Smoothing — Why It Is Used and How It Works

Raw per-frame stress values jump around. A single frame where the patient tightens their face for 33ms would produce a spike, then instantly return to zero. This is not clinically meaningful — it is noise.

**Exponential Moving Average (EMA)** smooths the signal:

```
EMA_new = 0.15 × frame_stress + 0.85 × EMA_prev

α = 0.15 (smoothing factor)
```

This formula weights the current frame's stress at 15% and the running history at 85%. The effect is a smooth signal that responds to real sustained changes but does not spike from momentary noise.

**Why α = 0.15 specifically?**

```
If a patient suddenly develops genuine acute pain (sustained activation):
  Their frame_stress jumps from 2.0 to 8.0

  Frame 1:  EMA = 0.15 × 8.0 + 0.85 × 2.0 = 1.20 + 1.70 = 2.90
  Frame 2:  EMA = 0.15 × 8.0 + 0.85 × 2.90 = 1.20 + 2.47 = 3.67
  Frame 3:  EMA = 0.15 × 8.0 + 0.85 × 3.67 = 1.20 + 3.12 = 4.32
  Frame 7:  EMA ≈ 6.0  (alert zone reached in ~0.23 seconds at 30fps)
  Frame 10: EMA ≈ 7.0  (fully responsive in ~0.33 seconds)

This is fast enough to detect a genuine pain episode almost instantly.
```

```
If a patient twitches their face for a single frame:
  frame_stress spikes to 8.0 for 1 frame, then returns to 2.0

  Frame 1 (spike):  EMA = 0.15 × 8.0 + 0.85 × 2.0 = 2.90
  Frame 2 (normal): EMA = 0.15 × 2.0 + 0.85 × 2.90 = 2.77
  Frame 3 (normal): EMA = 0.15 × 2.0 + 0.85 × 2.77 = 2.65

The spike barely moved the score and decayed within 3 frames.
Not clinically significant — correctly ignored.
```

**Advantage over a simple rolling mean:**
A 30-frame rolling mean treats a stress event 29 frames ago equally to the current frame. EMA weights recent frames more heavily — newer information about the patient's state counts more than older information. This makes the score more clinically relevant: what is happening *now* matters more than what happened 30 frames ago.

---

### 6.7 Complete Stress Score Computation — Step by Step

```
Every frame:

1. Feed frame to MediaPipe FaceLandmarker
   → Get 478 landmarks (3D positions)
   → Get 52 blendshapes (0.0–1.0 each)

2. Extract AU blendshapes and scale to 0–5:
   au4  = avg(browDownLeft, browDownRight) × 5
   au6  = avg(cheekSquintLeft, cheekSquintRight) × 5
   au7  = avg(eyeSquintLeft, eyeSquintRight) × 5
   au9  = avg(noseSneerLeft, noseSneerRight) × 5
   au10 = avg(mouthUpperUpLeft, mouthUpperUpRight) × 5
   au43 = 1.0 if avg(eyeBlinkLeft, eyeBlinkRight) > 0.55 else 0.0

3. Get smile correction factor:
   smile = avg(mouthSmileLeft, mouthSmileRight)

4. Compute smile-corrected PSPI:
   PSPI = (au4 + max(au6, au7) + max(au9, au10) + au43) × (1 − smile)
   Range: 0 to 16

5. Compute head movement agitation:
   Track nose tip (landmark 1) position
   motion = mean displacement of nose over last 30 frames

6. Combine into frame-level stress:
   frame_stress = 10 × (0.80 × PSPI/16 + 0.20 × min(motion/0.020, 1.0))
   Range: 0 to 10

7. Apply EMA smoothing:
   EMA = 0.15 × frame_stress + 0.85 × EMA_prev

8. Final output:
   Stress Score = round(min(EMA, 10.0), 1)
```

---

## 7. The Current GCS Method: Rolling Window with Personal Calibration

### 7.1 Why a Window, Not a Snapshot

The GCS clinical definition describes sustained behavioural patterns observed over time:

- **E4 (Spontaneous)** does not mean "eyes are open at this instant." It means the patient's **predominant, spontaneous behaviour** is to keep their eyes open without any prompting.
- **E3 (To Voice)** means the patient **tends to open their eyes** when spoken to — a pattern, not a single event.
- **E2 (To Pain)** means there is **some observable eye-opening response**, even if brief.

A single video frame captures a millisecond in time. It cannot capture any of these patterns. The solution is to observe eye behaviour over **30 seconds** — matching the clinical observation period — and assess the distribution of eye states across that window.

### 7.2 Personal Calibration — Adapting to Each Patient

Eye shape varies enormously between individuals. This is not a small difference:

```
Person A (naturally narrow eyes, fully alert and awake):
  Normal open-eye EAR ≈ 0.19
  
Person B (naturally wide eyes, same alertness level):
  Normal open-eye EAR ≈ 0.42
```

If we use a fixed threshold (e.g., open = EAR ≥ 0.22), then:
- Person A: 0.19 < 0.22 → E3 (misclassified as not fully open, despite being awake)
- Person B: 0.42 ≥ 0.22 → E4 (correctly classified)

Person A would consistently be underscored. Personal calibration fixes this by learning each patient's own anatomy before starting assessment.

**How calibration works:**

```
Phase: First ~3 seconds of face video

For each frame:
  1. Compute EAR (Eye Aspect Ratio from 6 eye landmarks per eye)
  2. Check eyeBlink blendshape: if eyeBlink < 0.30 → eyes are clearly open
     (blendshape provides a second opinion — neural network confirms eyes are open)
  3. If confirmed open: record this frame's EAR value

After collecting 90 such clearly-open frames:
  4. baseline_EAR = median of all collected EAR values
     (median used, not mean, because it is robust to the occasional blink)
  
  5. Validate the baseline is within physiological range:
     Must be between 0.15 and 0.50
     (outside this range → calibration failed → use population defaults)
  
  6. Set personalised thresholds:
     open_threshold    = baseline_EAR × 0.70
     partial_threshold = baseline_EAR × 0.40
```

**What the multipliers mean:**

```
Person A: baseline_EAR = 0.19
  open_threshold    = 0.19 × 0.70 = 0.133  (eyes at 70% normal aperture → "open")
  partial_threshold = 0.19 × 0.40 = 0.076  (eyes at 40% normal aperture → "partial/drowsy")

Person B: baseline_EAR = 0.42
  open_threshold    = 0.42 × 0.70 = 0.294  (eyes at 70% normal aperture → "open")
  partial_threshold = 0.42 × 0.40 = 0.168  (eyes at 40% normal aperture → "partial")

Now if both people close their eyes to 70% of their normal aperture,
both correctly score as "open" — the threshold adapted to their anatomy.
```

**Why the eyeBlink blendshape is used during calibration:**

The EAR alone cannot reliably distinguish "Person A with naturally narrow eyes that are wide open" from "Person B with wide eyes that are half-closed." The eyeBlink blendshape provides a neural network's assessment: "is this eye actually closing?" By only collecting calibration frames where the blendshape says the eye is clearly open, we ensure the baseline truly represents this person's open-eye state.

---

### 7.3 Per-Frame Eye State Classification

Once thresholds are calibrated, every frame is classified into one of three states using a dual-check:

```
1. Check blendshape first (highest priority):
   if avg(eyeBlinkLeft, eyeBlinkRight) > 0.55:
     state = "closed"   ← blendshape overrides EAR completely

2. If blendshape does not indicate closure, use EAR:
   elif EAR ≥ open_threshold:
     state = "open"
   elif EAR ≥ partial_threshold:
     state = "partial"   (drowsy, drooping — eyes present but not fully open)
   else:
     state = "closed"
```

**Why blendshape overrides EAR:**

During a blink, the EAR drops very rapidly (from ~0.30 to near 0 in 5–10 frames). However, EAR is computed from 6 landmark positions that may be imprecisely tracked during fast motion. The blendshape, being a neural network trained on blink motion, is more reliable for detecting the blink event itself. If the network says "eye is closing," that takes precedence over what the EAR geometry says.

**Concrete example:**

```
Patient with calibrated thresholds: open = 0.175, partial = 0.100

Frame A: EAR = 0.22, eyeBlink = 0.15
  Blendshape: 0.15 < 0.55 → not forcing override
  EAR: 0.22 ≥ 0.175 → state = "open" ✓

Frame B (mid-blink): EAR = 0.12, eyeBlink = 0.64
  Blendshape: 0.64 > 0.55 → state = "closed" (blink event)
  EAR would say: 0.12 ≥ 0.100 → "partial" (wrong — it's a blink)
  Blendshape override prevents this misclassification ✓

Frame C (drowsy patient): EAR = 0.14, eyeBlink = 0.30
  Blendshape: 0.30 < 0.55 → no override
  EAR: 0.14 ≥ 0.100 but < 0.175 → state = "partial" ✓
  (correctly identifies drooping but not closed eyes)
```

---

### 7.4 The 30-Second Rolling Window

Every frame's eye state is added to a sliding 30-second window. When enough data has accumulated, the GCS score is computed from the distribution of states in that window:

```
Window statistics computed:
  open_fraction     = count("open") / total_frames
  any_open_fraction = count("open" or "partial") / total_frames
  longest_run       = longest consecutive sequence of non-"closed" frames

GCS score determination:

  if open_fraction ≥ 0.70:
    → E4 (Spontaneous)
    Patient's eyes are open for at least 70% of the observation window
    
  elif any_open_fraction ≥ 0.30:
    → E3 (To Voice)
    Patient opens their eyes (even partially) for at least 30% of the window
    
  elif longest_run × frame_duration ≥ 0.10 seconds:
    → E2 (To Pain)
    There was at least one brief but genuine sustained opening (≥ 100ms, ≥ 3 frames)
    
  else:
    → E1 (No Response)
    Eyes remained essentially closed throughout the entire window
```

**Explaining the 70% threshold for E4:**

A healthy, fully awake person blinks approximately 12–15 times per minute. Each blink takes about 150–400ms. At 30 frames per second, a blink takes roughly 5–12 frames.

```
Calculation for a healthy awake person:
  30 seconds × 30fps = 900 total frames
  15 blinks × 10 frames per blink = 150 frames with closed eyes
  Open frames = 900 − 150 = 750
  Open fraction = 750/900 = 83.3%

83.3% comfortably exceeds the 70% threshold.
→ Normal blinking does not prevent an awake patient from scoring E4. ✓
```

If the threshold were 90%, some awake patients who blink frequently would score E3. If it were 50%, drowsy patients with eyes half-closed would score E4. The 70% value represents the optimal point that separates truly alert patients from drowsy ones while allowing for normal blinking.

**Explaining the 0.10-second (100ms) minimum for E2:**

Clinical GCS E2 requires the patient to open their eyes in response to pain — even briefly. However, a single video frame at 30fps represents only 33ms. A genuine flicker (a patient partially opening their eyes in response to stimulus) lasts at least 3–4 consecutive frames (100–130ms).

Without the 100ms minimum, a single noisy frame where the eye landmark positions momentarily fluctuated upward could be counted as an E2 flicker — a false positive from noise. The 100ms threshold ensures only genuine, sustained openings count.

---

### 7.5 Provisional Score Flag

For the first 10 seconds, there is insufficient behavioural data to make a reliable GCS assessment. The score is marked as **provisional**:

```
If observed_seconds < 10.0:
  Display: "E4 (provisional — < 10s data)"

After 10 seconds:
  Display: "E4"  (no qualifier — full confidence in the assessment)
```

**Why 10 seconds?**

Ten seconds at 30fps provides 300 frames. Even if half are blinks or transitions, 150 frames of data is sufficient to reliably compute the open_fraction and any_open_fraction statistics. Before 10 seconds, the window is too short to distinguish a patient who is truly awake from one who just happened to have their eyes open in the first few frames.

---

## 8. Why These Are the Best Methods Available

### 8.1 Stress Score: PSPI vs. Geometric EAR/Brow

| Criterion | Geometric V1 | PSPI V2 (Current) |
|---|---|---|
| **Clinical validation** | None — self-derived formula | Peer-reviewed publication (Prkachin & Solomon, 2008) |
| **Muscle groups covered** | 2 (brow gap, eye width) | 6 (brow furrow, cheek squint, lid tighten, nose wrinkle, lip raise, eye closure) |
| **Person independence** | No — fails if natural EAR ≠ hardcoded baseline | Yes — blendshapes are trained across diverse faces |
| **Smile false-positive** | Yes — smiling = stress | No — smile multiplier explicitly cancels this |
| **Agitation component** | No — expression only | Yes — 20% from head movement |
| **Response speed** | ~1.5 seconds (45-frame rolling mean) | ~0.2 seconds (EMA α=0.15) |
| **Works from frame 1** | No — 5-second calibration required, score stays 0.0 | Yes — PSPI is absolute, no calibration needed |
| **Fatal bug risk** | Yes — hardcoded baseline caused permanent 0.0 | No — blendshapes are relative to muscle activation, not anatomy |

### 8.2 GCS Eye Score: Rolling Window vs. Snapshot

| Criterion | Snapshot V1 | Rolling Window V2 (Current) |
|---|---|---|
| **Matches GCS clinical definition** | No — GCS is a behavioural pattern, not a pixel measurement | Yes — 30-second window assesses sustained behaviour |
| **Handles individual anatomy** | No — fixed 0.22 threshold misclassifies narrow-eyed patients | Yes — personal calibration adapts to each patient |
| **Blink robustness** | No — one blink in the assessed frame → E1 | Yes — single blinks are a tiny fraction of 900 frames |
| **Confidence indicator** | None — claimed E4 from first frame | Yes — provisional flag for first 10 seconds |
| **Blendshape cross-check** | No — EAR only | Yes — blendshape overrides EAR during blink detection |
| **Distinguishes partial from closed** | No — binary open/closed | Yes — three states: open, partial, closed |
| **Scientific basis** | Arbitrary threshold | GCS clinical criteria (Teasdale & Jennett, 1974) |

### 8.3 Why Blendshapes Beat Raw Geometry for Both Methods

**Geometric distances fail because of three unavoidable problems:**

1. **Anatomical variation:** A brow that is naturally low on the face produces smaller brow-to-eye distances even when not furrowed. A person with naturally small eyes has low EAR even when fully open. Fixed thresholds cannot account for this.

2. **Baseline dependency:** Geometric approaches need a personal baseline to detect changes. If that baseline is wrong (wrong default, poor calibration conditions), the entire system fails silently — it produces scores that look plausible but are based on wrong reference points.

3. **Lighting sensitivity:** Landmarks are detected from edge and texture patterns in the image. Changing lighting shifts where the model thinks the landmark is, even if the face has not moved.

**Blendshapes solve all three:**

1. **Trained across diverse faces:** The neural network has seen thousands of faces, narrow and wide, brows high and low. It has learned to output "brow is 70% furrowed" regardless of where the brow is in resting position — it assesses the muscle activation relative to that person's neutral state.

2. **No baseline required for stress:** PSPI is an absolute metric of muscle activation level. It does not need to know what your "normal" face looks like. The formula evaluates whether specific muscles are currently contracting.

3. **Trained across lighting conditions:** The model training included diverse lighting, and the blendshape output is more robust to illumination changes than raw distance computations.

---

## 9. Complete Technical Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│  WEBCAM INPUT (1280×720, ~30fps)                                       │
│  Each frame: BGR image array                                           │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ raw frame
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  MEDIAPIPE FACELM DETECTION                                            │
│  Input:  raw frame (BGR → RGB converted)                               │
│  Output: 478 normalised 3D landmarks (x, y, z ∈ [0, 1])              │
│          52 blendshape scores (each ∈ [0.0, 1.0])                     │
│  If no face detected: skip frame, keep previous EMA value             │
└───────────────┬────────────────────────────┬───────────────────────────┘
                │ landmarks                  │ blendshapes
                ▼                            ▼
┌───────────────────────┐    ┌──────────────────────────────────────────┐
│  EAR COMPUTATION      │    │  PSPI COMPUTATION                        │
│                       │    │                                          │
│  For each eye:        │    │  au4  = avg(browDown L+R) × 5            │
│  6 landmarks (p0–p5)  │    │  au6  = avg(cheekSquint L+R) × 5        │
│  EAR = (||p1–p5||     │    │  au7  = avg(eyeSquint L+R) × 5          │
│       + ||p2–p4||) /  │    │  au9  = avg(noseSneer L+R) × 5          │
│       (2×||p0–p3||)   │    │  au10 = avg(mouthUpperUp L+R) × 5       │
│  Final: avg(L, R)     │    │  au43 = eyeBlink avg > 0.55 ? 1 : 0     │
│  Smoothed: 20-frame   │    │                                          │
│  rolling mean         │    │  smile = avg(mouthSmile L+R)            │
│                       │    │                                          │
└──────────┬────────────┘    │  PSPI = (au4 + max(au6,au7)             │
           │                 │        + max(au9,au10) + au43)          │
           │                 │        × (1 − smile)                    │
           │                 └────────────────────┬─────────────────────┘
           │                                      │ PSPI (0–16)
           │                                      ▼
           │                 ┌──────────────────────────────────────────┐
           │                 │  HEAD MOVEMENT (AGITATION)               │
           │                 │  Track nose tip (landmark 1)            │
           │                 │  motion = mean ||nose[t]−nose[t−1]||    │
           │                 │          over last 30 frames            │
           │                 └────────────────────┬─────────────────────┘
           │                                      │ motion energy
           │                                      ▼
           │                 ┌──────────────────────────────────────────┐
           │                 │  STRESS SCORE (EMA)                      │
           │                 │  frame_s = 10×(0.80×PSPI/16             │
           │                 │             + 0.20×min(motion/0.02,1))  │
           │                 │  EMA = 0.15×frame_s + 0.85×EMA_prev    │
           │                 │  Output: Stress Score 0.0–10.0          │
           │                 └──────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────────┐
│  GCS PERSONAL CALIBRATION (first ~3 seconds)                          │
│                                                                        │
│  For each frame where eyeBlink_blendshape < 0.30:                    │
│    Record current EAR value                                           │
│                                                                        │
│  After 90 confirmed-open frames (~3 seconds):                         │
│    baseline_EAR = median(recorded values)                             │
│    Validate: 0.15 ≤ baseline_EAR ≤ 0.50                              │
│    open_threshold    = baseline × 0.70                                │
│    partial_threshold = baseline × 0.40                                │
│    (If validation fails: use defaults 0.22 / 0.12)                    │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │ personalised thresholds
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  PER-FRAME EYE STATE CLASSIFICATION                                   │
│                                                                        │
│  Priority 1 — blendshape override:                                    │
│    if eyeBlink > 0.55 → state = "closed"                              │
│                                                                        │
│  Priority 2 — personalised EAR thresholds:                            │
│    elif EAR ≥ open_threshold    → state = "open"                      │
│    elif EAR ≥ partial_threshold → state = "partial"                   │
│    else                         → state = "closed"                    │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │ frame eye state
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  30-SECOND ROLLING WINDOW ASSESSMENT                                  │
│                                                                        │
│  Statistics computed:                                                  │
│    open_frac     = count("open")   / total_frames                    │
│    any_frac      = count(≠"closed") / total_frames                   │
│    longest_run   = longest consecutive non-closed sequence            │
│                                                                        │
│  GCS Eye Score:                                                       │
│    open_frac   ≥ 0.70       → E4 Spontaneous                         │
│    any_frac    ≥ 0.30       → E3 To Voice                            │
│    longest_run × frame_s ≥ 0.10s → E2 To Pain                        │
│    otherwise               → E1 No Response                          │
│                                                                        │
│  Provisional flag:                                                     │
│    if observed_seconds < 10.0 → mark score as provisional            │
└────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    Stress Score + GCS Eye Score
                    → fed into 108-feature SOFA vector
```

---

## 10. Output Scores and Dashboard Display

### Stress Score Display

| Display | What it means | When to act clinically |
|---|---|---|
| `0.0–3.4` — **LOW** (green) | Patient appears calm, no visible distress | Routine monitoring |
| `3.5–6.4` — **MODERATE** (amber) | Noticeable discomfort or agitation | Review analgesia and comfort measures |
| `6.5–10.0` — **HIGH** (red) | Severe visible distress | Immediate clinical review for pain, dyspnoea, delirium |
| `Session peak: X.X` | Highest score recorded since monitoring began | Useful even if the patient has calmed — peak may have clinical significance |

**PSPI component bars also displayed:**
- **Brow Furrow bar:** How active AU4 is (the pain frown) — the most reliable single pain indicator
- **Eye Tension bar:** How active max(AU6, AU7) is (orbital squinting)
- **Nose/Lip bar:** How active max(AU9, AU10) is (pain grimace around nose and lip)

A HIGH stress score driven entirely by the Brow Furrow bar is more likely to represent genuine pain than one driven only by the Eye Tension bar (which can be caused by bright lighting).

### GCS Eye Score Display

| Display | Clinical Interpretation | What it suggests about consciousness |
|---|---|---|
| `E4 — Spontaneous` | Eyes open naturally, without prompting | Brainstem arousal intact — patient is alert |
| `E3 — To Voice` | Eyes open intermittently or when the patient is spoken to | Partial arousal — patient needs stimulation to maintain wakefulness |
| `E2 — To Pain` | Only brief eye flickers observed across 30-second window | Minimal arousal — patient is deeply sedated or significantly impaired |
| `E1 — No Response` | Eyes remain continuously closed throughout window | No observable eye opening — deep coma state or heavy sedation |
| `(provisional — < 10s)` | Score based on less than 10 seconds of data — may change | Interpret with caution until 10s elapsed |
| `[calibrating...]` in title | Personal eye baseline not yet established | Thresholds are population defaults until calibration completes |
| `(personal)` after thresholds | Thresholds adapted to this patient's anatomy | Most accurate state |
| `(default)` after thresholds | Calibration incomplete or failed — using population averages | Accuracy reduced for patients with non-average eye anatomy |

---

## 11. System Constants Explained

Every constant in the system was chosen for a specific clinical or engineering reason. This section explains each one.

| Constant | Value | Why This Value |
|---|---|---|
| `_SMOOTH_WIN` | 20 frames | EAR smoothing window for GCS computation. At 30fps, 20 frames = 0.67 seconds — long enough to average out frame-to-frame noise, short enough to not lag behind genuine state changes. |
| `_EMA_ALPHA` | 0.15 | Stress EMA smoothing factor. α = 0.15 gives ~0.2 seconds rise time (fast enough to detect acute pain) while not spiking from momentary expressions. Lower α = slower response; higher α = more noise. |
| `_PSPI_MAX` | 16.0 | The theoretical maximum raw PSPI score if all Action Units activate simultaneously (5+5+5+1=16). Used as the denominator in PSPI/16 normalisation. |
| `_BLINK_CLOSED` | 0.55 | eyeBlink blendshape threshold above which we classify the eye as closing/closed. 0.55 was chosen because genuine blinks produce blendshape values of 0.7–1.0, while partial squints stay below 0.5. The 0.55 threshold cleanly separates these ranges. |
| `_MOTION_SAT` | 0.020 | The agitation signal saturates (reaches maximum = 1.0) at 2% of frame width per frame displacement. At 30fps and 1280px width, this equals 0.02 × 1280 = 25.6 pixels per frame = approximately 768 pixels/second of nose movement — vigorous head restlessness. |
| `_FACIAL_W` | 0.80 | 80% of stress score from facial PSPI, 20% from motion. Facial expression is the primary validated pain indicator. Motion adds broader agitation signal. |
| `_GCS_WIN_S` | 30.0 seconds | Duration of the GCS rolling window. Matches the typical clinical observation period for GCS assessment. Long enough to observe sustained behaviour patterns, not so long that it cannot track rapid deterioration. |
| `_GCS_CALIB_N` | 90 frames | The number of confirmed open-eye frames required for personal calibration. At 30fps, 90 frames = 3 seconds. 90 frames provides enough EAR values for a statistically stable median estimate. |
| `_GCS_CALIB_BLINK` | 0.30 | eyeBlink blendshape maximum for a calibration frame to count as "clearly open." Conservative threshold (well below the 0.55 closure threshold) ensures only frames where the eye is unambiguously open contribute to the baseline. |
| `_GCS_CALIB_EAR_MIN` | 0.15 | Minimum physiologically plausible open-eye EAR across all human populations. Below 0.15 suggests a tracking error or impossible eye shape. |
| `_GCS_CALIB_EAR_MAX` | 0.50 | Maximum physiologically plausible open-eye EAR. Above 0.50 suggests a detection error (e.g., wide-open mouth being tracked as an eye). |
| `_GCS_OPEN_DEFAULT` | 0.22 | Population-average open-eye EAR threshold, used before personal calibration completes. Matches the value from published EAR literature for average human eye shape. |
| `_GCS_PART_DEFAULT` | 0.12 | Population-average partial-open threshold. At roughly 55% of the 0.22 open threshold — corresponds to drooping or nearly-closed eyes. |
| `_GCS_OPEN_FRAC` | 0.70 | Minimum fraction of the 30-second window where eyes must be "open" to score E4. 70% allows for normal blinking (~83% open for a healthy awake person) while excluding patients who are drowsy and only occasionally open their eyes. |
| `_GCS_ANY_FRAC` | 0.30 | Minimum fraction of the window where eyes are open OR partial to score E3. 30% means eyes were open in some form for 9 of 30 seconds — meaningful but not sustained. |
| `_GCS_FLICKER_S` | 0.10 seconds | Minimum duration of a continuous eye opening to count as E2. At 30fps, 0.10 seconds = 3 frames. Prevents single-frame noise from being counted as a genuine E2 flicker. |
| `_GCS_MIN_OBS_S` | 10.0 seconds | Minimum observation window before GCS score becomes non-provisional. 10 seconds = 300 frames = sufficient data for stable statistics. |

---

## 12. Limitations and Clinical Disclaimer

### What this system can reliably do

- Provide **continuous, non-contact, real-time monitoring** of visible stress and eye behaviour
- Detect sustained changes in facial expression associated with pain and distress using a clinically validated formula
- Produce GCS eye estimates that are accurate for patients with typical facial anatomy and good camera conditions
- Adapt to individual patient eye anatomy via personal calibration
- Track session peak stress for clinical review and escalation decisions

### What this system cannot do

**GCS limitations — the stimulus problem:**

True clinical GCS is defined by how the patient responds to stimuli:
- **E3** requires a verbal stimulus (calling the patient's name) and observing whether eyes open in response
- **E2** requires a painful pressure stimulus (sternal rub, nail bed pressure) and observing whether eyes open

The camera cannot apply these stimuli. Therefore:
- **E4** (spontaneous opening) is assessed reliably — the camera can observe eyes opening without any stimulus
- **E3** is inferred from intermittent eye opening patterns — this is a proxy, not a true stimulus-response test
- **E2** is inferred from brief flickers in the observation window — the most uncertain score

In clinical practice, E3 and E2 camera scores should be treated as indicators that the patient may benefit from a formal nurse GCS assessment, not as replacements for one.

**Stress score limitations:**

| Limitation | Impact | When it occurs |
|---|---|---|
| Face obstruction | Reduced or impossible tracking | Oxygen masks, bandages, ventilator tubing across the face |
| Poor lighting | MediaPipe detection fails or degrades | Dark rooms, strong backlighting from a window directly behind the patient |
| Facial paralysis | PSPI activations do not reflect actual pain | Stroke, Bell's palsy, any condition affecting facial nerve function |
| Stoic expression style | Under-estimates pain | Some patients suppress facial expression — particularly common in certain cultural contexts and among trained healthcare workers as patients |
| PSPI validation population | Research was done on neurologically normal adults | Paediatric patients, patients with developmental differences, or severe neurological conditions may not fit the validated formula's assumptions |

**Coverage angle limitation:**

MediaPipe FaceLandmarker requires a roughly frontal face view. If the patient's head is turned more than approximately 45° from the camera, landmark tracking becomes unreliable. Patient positioning and camera placement are important operational considerations.

---

### Clinical Use Guidance

> **This system is a decision-support tool, not a diagnostic instrument.**
>
> All Stress Scores and GCS Eye scores produced by this system should be interpreted by a qualified clinician alongside other clinical findings and direct patient assessment.
>
> The GCS scores are camera-based proxies derived from observed behaviour. They do not replace clinical GCS assessment involving verbal and pain stimuli administered by a nurse or physician.
>
> A score suggesting high distress (Stress Score > 6.5) or GCS deterioration (E4 → E3 or lower) should prompt immediate clinical review and direct patient assessment.
>
> This system's output is one input to the SOFA prediction model, which is itself a decision-support tool requiring clinical judgment before any action is taken.

---

*Documentation for ICU CDSS Capstone Project*
*Stress scoring: PSPI formula (Prkachin & Solomon, 2008, Pain 137(2))*
*GCS Eye scoring: Teasdale & Jennett (1974), adapted for continuous camera-based assessment*
*Facial analysis: MediaPipe FaceLandmarker (Google, 2023) with 52 blendshapes*

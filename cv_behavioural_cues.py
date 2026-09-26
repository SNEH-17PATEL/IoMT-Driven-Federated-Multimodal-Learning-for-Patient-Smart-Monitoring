"""
cv_behavioural_cues.py — Computer Vision behavioural cues (standalone module)
============================================================================

Estimates the two "Computer Vision" features used by the ICU model from a
camera / video feed of the patient's face:

    GCS_eye_opening   int   1–4   Glasgow Coma Scale eye response
    stress_score      float 0–10  pain / distress level from facial expression

The output dict uses exactly those keys, so it can later be dropped into the
`latest_features` dict of the inference pipeline (see docs/INPUT_OUTPUT.md).
This file is NOT wired into the Streamlit UI yet.

How the scores are derived
--------------------------
GCS Eye Opening (camera only — no speech or stimulus input)
    Each frame's eye state comes from the Eye Aspect Ratio (EAR, Soukupová &
    Čech 2016) on MediaPipe face-mesh landmarks, cross-checked with the
    model's `eyeBlink` blendshape:

        open     EAR ≥ 70 % of the patient's normal open-eye EAR
                 and the eyeBlink blendshape does not say "closed"
        partial  lids apart but drooping (EAR ≥ 40 % of normal) — drowsy eyes
        closed   otherwise

    Per-patient calibration: eye shape differs between people (a normal open
    eye can be EAR 0.19 for one patient and 0.40 for another), so fixed
    thresholds misjudge some patients. The first ~2 s of frames where the
    blendshape says the eyes are clearly open are used to learn that
    patient's normal open-eye EAR (median), and the thresholds are scaled
    from it. Until then, or if the eyes are never seen open, the population
    defaults (open ≥ 0.21, partial ≥ 0.12) are used. The learnt value is
    returned as `baseline_ear` and can be passed back in (`--baseline-ear`)
    to reuse it for the same patient in later sessions.

    Over a rolling 30 s window the eye behaviour is mapped to a GCS-style
    score (a behavioural proxy, since no voice/pain stimulus is applied):

        4  eyes fully open for most of the window (≥ 70 % of frames)
        3  eyes partly or intermittently open (open or partial ≥ 30 %)
        2  only brief flickers of opening (at least one ≥ 0.1 s episode)
        1  eyes closed throughout

    Blinks are closures inside open periods, so they barely move the open
    fraction. `gcs_assessment_complete` is False until at least 10 s of face
    video has been observed; until then the score is provisional.

Stress Score
    Based on the Prkachin–Solomon Pain Intensity (PSPI) metric, built from
    facial action units that are reliably associated with pain:

        PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43        (0–16)

    AU intensities are approximated from MediaPipe face blendshapes
    (browDown, cheekSquint, eyeSquint, noseSneer, mouthUpperUp, eyeBlink).
    Smiling uses the same eye-squint / upper-lip muscles, so the PSPI is
    scaled down by the `mouthSmile` blendshape (a broad smile → near 0).
    Restlessness/agitation (normalised head-movement energy) is blended in,
    then the result is scaled to 0–10 and smoothed with an EMA.

Dependencies
------------
    pip install mediapipe opencv-python numpy

MediaPipe also needs the FaceLandmarker model bundle (~3.7 MB):
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
Pass its path with --model, or run with --download-model once.

Usage
-----
    python cv_behavioural_cues.py --source 0                 # webcam
    python cv_behavioural_cues.py --source clip.mp4 --show   # video file
    python cv_behavioural_cues.py --self-test                # no camera/model

While running with --show, press q to quit.

This is a decision-support signal, not a validated clinical instrument; a
clinician should confirm GCS and pain assessments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join("models", "face_landmarker.task")

# MediaPipe 478-point face mesh indices, ordered p1..p6 for the EAR formula:
# p1/p4 = eye corners, (p2,p6) and (p3,p5) = upper/lower lid pairs.
LEFT_EYE_EAR_IDX = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_EAR_IDX = (33, 160, 158, 133, 153, 144)
NOSE_TIP_IDX = 1

EAR_OPEN_THRESHOLD = 0.21        # EAR above this → eyes fully open
EAR_PARTIAL_THRESHOLD = 0.12     # EAR above this → lids at least partly apart
BLINK_CLOSED_THRESHOLD = 0.55    # eyeBlink blendshape above this → closed

CALIB_SAMPLES = 60               # clearly-open frames used to learn baseline (~2 s)
CALIB_BLINK_MAX = 0.30           # only frames with eyeBlink below this are used
CALIB_OPEN_RATIO = 0.70          # open    ≥ 70 % of the patient's baseline EAR
CALIB_PARTIAL_RATIO = 0.40       # partial ≥ 40 % of the patient's baseline EAR
CALIB_EAR_MIN = 0.15             # plausible range for a normal open-eye EAR;
CALIB_EAR_MAX = 0.50             # outside it the defaults are kept

GCS_WINDOW_S = 30.0              # rolling window the GCS score is judged over
MIN_OBSERVATION_S = 10.0         # face video needed before the score is final
SPONTANEOUS_OPEN_FRACTION = 0.70 # ≥ this fraction fully open → 4
INTERMITTENT_FRACTION = 0.30     # ≥ this fraction open/partial → 3
MIN_FLICKER_S = 0.1              # shortest opening that counts as a flicker → 2

STRESS_EMA_ALPHA = 0.15          # smoothing for per-frame stress
MOTION_WINDOW_FRAMES = 30        # frames used for head-movement energy
MOTION_SATURATION = 0.02         # normalised movement treated as max agitation
FACIAL_WEIGHT = 0.8              # stress = 0.8 * facial pain + 0.2 * agitation
PSPI_MAX = 16.0

GCS_EYE_LABELS = {
    4: "Eyes open (spontaneous)",
    3: "Intermittent / partial opening",
    2: "Brief flickers only",
    1: "Eyes closed",
}


# ── Pure scoring functions (no MediaPipe needed) ─────────────────────────────

def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR for one eye. `points` is a (6, 2) array ordered p1..p6."""
    p1, p2, p3, p4, p5, p6 = points
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / horizontal)


def pspi_from_blendshapes(bs: dict[str, float]) -> float:
    """
    Approximate PSPI (0–16) from MediaPipe blendshape scores (each 0–1).
    Each AU is mapped to 0–5 like the FACS intensity scale; AU43 is 0/1.
    """
    def pair(name: str) -> float:
        return (bs.get(f"{name}Left", 0.0) + bs.get(f"{name}Right", 0.0)) / 2.0

    au4 = pair("browDown") * 5.0                        # brow lowerer
    au6 = pair("cheekSquint") * 5.0                     # cheek raiser
    au7 = pair("eyeSquint") * 5.0                       # lid tightener
    au9 = pair("noseSneer") * 5.0                       # nose wrinkler
    au10 = pair("mouthUpperUp") * 5.0                   # upper lip raiser
    au43 = 1.0 if pair("eyeBlink") > BLINK_CLOSED_THRESHOLD else 0.0  # eyes closed

    pspi = au4 + max(au6, au7) + max(au9, au10) + au43
    # A smile (AU12) also squints the eyes and lifts the upper lip, which PSPI
    # would read as pain; damp the score in proportion to smile strength.
    return float(pspi * (1.0 - pair("mouthSmile")))


def combine_stress(pspi: float, motion_energy: float) -> float:
    """Blend facial pain (PSPI) with head-movement agitation into 0–10."""
    facial = min(max(pspi / PSPI_MAX, 0.0), 1.0)
    agitation = min(max(motion_energy / MOTION_SATURATION, 0.0), 1.0)
    return 10.0 * (FACIAL_WEIGHT * facial + (1.0 - FACIAL_WEIGHT) * agitation)


def eye_state(ear: float, blink_score: float | None,
              open_threshold: float = EAR_OPEN_THRESHOLD,
              partial_threshold: float = EAR_PARTIAL_THRESHOLD) -> str:
    """Frame-level eye state: 'open', 'partial' or 'closed'."""
    if blink_score is not None and blink_score > BLINK_CLOSED_THRESHOLD:
        return "closed"
    if ear >= open_threshold:
        return "open"
    if ear >= partial_threshold:
        return "partial"
    return "closed"


# ── Temporal estimators ──────────────────────────────────────────────────────

@dataclass
class FrameCues:
    """Per-frame measurements extracted from one image."""
    timestamp_s: float
    face_present: bool
    ear: float = 0.0
    blink_score: float | None = None
    blendshapes: dict[str, float] = field(default_factory=dict)
    nose_xy: tuple[float, float] | None = None


class GCSEyeEstimator:
    """
    Scores eye opening from camera behaviour alone over a rolling window.
    Feed frames in time order with `update`; read the score with `result`.

    Pass `baseline_ear` (from `result()["baseline_ear"]` of an earlier session)
    to reuse a patient's calibration instead of re-learning it.
    """

    def __init__(self, window_s: float = GCS_WINDOW_S, baseline_ear: float | None = None):
        self.window_s = window_s
        # Raw measurements, so all frames are re-scored once calibration lands.
        self.history: deque[tuple[float, float, float | None]] = deque()
        self.calib_samples: list[float] = []
        self.baseline_ear: float | None = None
        self.calibration = "default"
        self.open_threshold = EAR_OPEN_THRESHOLD
        self.partial_threshold = EAR_PARTIAL_THRESHOLD
        self.last_state = "closed"
        if baseline_ear is not None:
            self._apply_baseline(baseline_ear, "saved")

    def _apply_baseline(self, baseline: float, source: str) -> bool:
        if not CALIB_EAR_MIN <= baseline <= CALIB_EAR_MAX:
            return False  # implausible (e.g. learnt while eyes were half-shut)
        self.baseline_ear = baseline
        self.calibration = source
        self.open_threshold = CALIB_OPEN_RATIO * baseline
        self.partial_threshold = CALIB_PARTIAL_RATIO * baseline
        return True

    def _calibrate(self, cues: FrameCues) -> None:
        """Learn this patient's open-eye EAR from frames that are clearly open."""
        if self.calibration != "default" or len(self.calib_samples) >= CALIB_SAMPLES:
            return
        if cues.blink_score is None or cues.blink_score >= CALIB_BLINK_MAX:
            return
        self.calib_samples.append(cues.ear)
        if len(self.calib_samples) == CALIB_SAMPLES:
            self._apply_baseline(float(np.median(self.calib_samples)), "calibrated")

    def _state(self, ear: float, blink: float | None) -> str:
        return eye_state(ear, blink, self.open_threshold, self.partial_threshold)

    def update(self, cues: FrameCues) -> None:
        if not cues.face_present:
            return  # face not visible: unknown, not "closed"
        self._calibrate(cues)
        self.last_state = self._state(cues.ear, cues.blink_score)
        self.history.append((cues.timestamp_s, cues.ear, cues.blink_score))
        while cues.timestamp_s - self.history[0][0] > self.window_s:
            self.history.popleft()

    def _observed_s(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return self.history[-1][0] - self.history[0][0]

    def _longest_opening_s(self, states: list[str]) -> float:
        """Longest run of open/partial frames, in seconds."""
        if len(states) < 2:
            return 0.0
        frame_s = self._observed_s() / (len(states) - 1)
        longest = run = 0
        for state in states:
            run = run + 1 if state != "closed" else 0
            longest = max(longest, run)
        return longest * frame_s

    def result(self) -> dict:
        states = [self._state(ear, blink) for _, ear, blink in self.history]
        n = len(states)
        open_frac = sum(s == "open" for s in states) / n if n else 0.0
        any_open_frac = sum(s != "closed" for s in states) / n if n else 0.0

        if open_frac >= SPONTANEOUS_OPEN_FRACTION:
            score = 4
        elif any_open_frac >= INTERMITTENT_FRACTION:
            score = 3
        elif self._longest_opening_s(states) >= MIN_FLICKER_S:
            score = 2
        else:
            score = 1

        return {
            "GCS_eye_opening": score,
            "gcs_eye_label": GCS_EYE_LABELS[score],
            "gcs_assessment_complete": self._observed_s() >= MIN_OBSERVATION_S,
            "eyes_open_fraction": round(open_frac, 2),
            "eyes_open_or_partial_fraction": round(any_open_frac, 2),
            "eye_calibration": self.calibration,
            "baseline_ear": round(self.baseline_ear, 3) if self.baseline_ear else None,
        }


class StressEstimator:
    """Smooths per-frame PSPI + head-movement agitation into a 0–10 score."""

    def __init__(self, alpha: float = STRESS_EMA_ALPHA,
                 motion_window: int = MOTION_WINDOW_FRAMES):
        self.alpha = alpha
        self.nose_track: deque[tuple[float, float]] = deque(maxlen=motion_window)
        self.ema: float | None = None
        self.peak = 0.0
        self.last_pspi = 0.0
        self.last_motion = 0.0

    def _motion_energy(self) -> float:
        """Mean frame-to-frame nose displacement (normalised image coords)."""
        if len(self.nose_track) < 2:
            return 0.0
        pts = np.asarray(self.nose_track)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())

    def update(self, cues: FrameCues) -> None:
        if not cues.face_present:
            return
        if cues.nose_xy is not None:
            self.nose_track.append(cues.nose_xy)
        self.last_pspi = pspi_from_blendshapes(cues.blendshapes)
        self.last_motion = self._motion_energy()
        frame_stress = combine_stress(self.last_pspi, self.last_motion)
        self.ema = frame_stress if self.ema is None else (
            self.alpha * frame_stress + (1 - self.alpha) * self.ema)
        self.peak = max(self.peak, self.ema)

    def result(self) -> dict:
        return {
            "stress_score": round(self.ema if self.ema is not None else 0.0, 1),
            "stress_peak": round(self.peak, 1),
            "pspi_last": round(self.last_pspi, 2),
            "motion_energy_last": round(self.last_motion, 4),
        }


# ── MediaPipe feature extraction ─────────────────────────────────────────────

def download_model(path: str = DEFAULT_MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading FaceLandmarker model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


class FaceCueExtractor:
    """Wraps MediaPipe FaceLandmarker (VIDEO mode) → FrameCues per frame."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at '{model_path}'. "
                f"Run with --download-model or fetch it from {MODEL_URL}")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def extract(self, frame_bgr: np.ndarray, timestamp_s: float) -> FrameCues:
        import cv2

        # VIDEO mode requires strictly increasing integer timestamps.
        ts_ms = max(int(timestamp_s * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(image, ts_ms)

        if not res.face_landmarks:
            return FrameCues(timestamp_s=timestamp_s, face_present=False)

        h, w = frame_bgr.shape[:2]
        lm = res.face_landmarks[0]
        # Pixel coords so EAR is not distorted by the frame's aspect ratio.
        pts = np.array([(p.x * w, p.y * h) for p in lm])
        ear = (eye_aspect_ratio(pts[list(LEFT_EYE_EAR_IDX)]) +
               eye_aspect_ratio(pts[list(RIGHT_EYE_EAR_IDX)])) / 2.0

        blendshapes = {}
        if res.face_blendshapes:
            blendshapes = {c.category_name: c.score for c in res.face_blendshapes[0]}
        blink = None
        if "eyeBlinkLeft" in blendshapes:
            blink = (blendshapes["eyeBlinkLeft"] + blendshapes["eyeBlinkRight"]) / 2.0

        nose = lm[NOSE_TIP_IDX]
        return FrameCues(
            timestamp_s=timestamp_s,
            face_present=True,
            ear=ear,
            blink_score=blink,
            blendshapes=blendshapes,
            nose_xy=(nose.x, nose.y),
        )

    def close(self) -> None:
        self._landmarker.close()


# ── High-level analyzer ──────────────────────────────────────────────────────

class BehaviouralCueAnalyzer:
    """
    One assessment session for one patient.

        analyzer = BehaviouralCueAnalyzer("models/face_landmarker.task")
        for each frame:  analyzer.process_frame(frame_bgr, t_seconds)
        features = analyzer.result()                 # GCS_eye_opening, stress_score
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 extractor: FaceCueExtractor | None = None,
                 baseline_ear: float | None = None):
        self.extractor = extractor or FaceCueExtractor(model_path)
        self.gcs = GCSEyeEstimator(baseline_ear=baseline_ear)
        self.stress = StressEstimator()
        self.frames_total = 0
        self.frames_with_face = 0

    def process_frame(self, frame_bgr: np.ndarray, timestamp_s: float) -> FrameCues:
        cues = self.extractor.extract(frame_bgr, timestamp_s)
        self.process_cues(cues)
        return cues

    def process_cues(self, cues: FrameCues) -> None:
        self.frames_total += 1
        if cues.face_present:
            self.frames_with_face += 1
        self.gcs.update(cues)
        self.stress.update(cues)

    def result(self) -> dict:
        coverage = self.frames_with_face / self.frames_total if self.frames_total else 0.0
        return {
            **self.gcs.result(),
            **self.stress.result(),
            "face_coverage": round(coverage, 2),
            "frames_analysed": self.frames_total,
        }

    def close(self) -> None:
        self.extractor.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _draw_overlay(frame, cues: FrameCues, gcs: GCSEyeEstimator, res: dict) -> None:
    import cv2
    eye = gcs.last_state if cues.face_present else "no face"
    lines = [
        f"GCS eye: {res['GCS_eye_opening']} ({res['gcs_eye_label']})"
        + ("" if res["gcs_assessment_complete"] else " *provisional"),
        f"Stress: {res['stress_score']:.1f}/10   PSPI: {res['pspi_last']:.1f}",
        f"Eyes: {eye}   EAR: {cues.ear:.2f}   open: {res['eyes_open_fraction']:.0%}",
        f"Calibration: {res['eye_calibration']}   baseline EAR: {res['baseline_ear']}",
        "q=quit",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 25 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 180), 2, cv2.LINE_AA)


def run_capture(source: str, model_path: str, show: bool, max_seconds: float | None,
                baseline_ear: float | None = None) -> dict:
    import cv2

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source '{source}'")
    is_file = not source.isdigit()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    analyzer = BehaviouralCueAnalyzer(model_path, baseline_ear=baseline_ear)
    start = time.monotonic()
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Video files use their own clock; live cameras use wall time.
            t = frame_idx / fps if is_file else time.monotonic() - start
            frame_idx += 1
            cues = analyzer.process_frame(frame, t)

            if max_seconds is not None and t >= max_seconds:
                break
            if show:
                _draw_overlay(frame, cues, analyzer.gcs, analyzer.result())
                cv2.imshow("Behavioural cues", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    finally:
        cap.release()
        analyzer.close()
        if show:
            cv2.destroyAllWindows()
    return analyzer.result()


def self_test() -> None:
    """Checks the scoring logic with synthetic cues (no camera or model needed)."""
    fps = 30

    def open_eyes(t):
        return FrameCues(t, True, ear=0.30, blink_score=0.05, nose_xy=(0.5, 0.5))

    def partial_eyes(t):
        return FrameCues(t, True, ear=0.16, blink_score=0.30, nose_xy=(0.5, 0.5))

    def closed_eyes(t):
        return FrameCues(t, True, ear=0.08, blink_score=0.90, nose_xy=(0.5, 0.5))

    def no_face(t):
        return FrameCues(t, False)

    def run(frames):
        est = GCSEyeEstimator()
        for i, make in enumerate(frames):
            est.update(make(i / fps))
        return est.result()

    # Eyes open with normal blinking (every 4 s) for 20 s → 4, complete.
    r = run(([open_eyes] * 115 + [closed_eyes] * 5) * 5)
    assert r["GCS_eye_opening"] == 4 and r["gcs_assessment_complete"], r
    # Drowsy: heavy-lidded half the time → 3.
    r = run(([partial_eyes] * 45 + [closed_eyes] * 45) * 5)
    assert r["GCS_eye_opening"] == 3, r
    # Mostly closed with a couple of short flickers → 2.
    r = run(([closed_eyes] * 140 + [open_eyes] * 10) * 3)
    assert r["GCS_eye_opening"] == 2, r
    # Closed throughout → 1, complete.
    r = run([closed_eyes] * 450)
    assert r["GCS_eye_opening"] == 1 and r["gcs_assessment_complete"], r
    # A single-frame landmark glitch is not a flicker.
    r = run([closed_eyes] * 200 + [open_eyes] + [closed_eyes] * 200)
    assert r["GCS_eye_opening"] == 1, r
    # Too little video → provisional.
    r = run([open_eyes] * 60)
    assert r["GCS_eye_opening"] == 4 and not r["gcs_assessment_complete"], r
    # Frames without a face are ignored, not counted as closed.
    r = run([open_eyes] * 300 + [no_face] * 600)
    assert r["GCS_eye_opening"] == 4, r
    # Only the last 30 s count: closed early, open for the latest 30 s → 4.
    r = run([closed_eyes] * 900 + [open_eyes] * 900)
    assert r["GCS_eye_opening"] == 4, r

    # Calibration: a patient with naturally narrow eyes (open EAR ≈ 0.19).
    # Fixed thresholds call them only "partial" → 3; calibrated → 4.
    def narrow_open(t):
        return FrameCues(t, True, ear=0.19, blink_score=0.10, nose_xy=(0.5, 0.5))

    fixed = GCSEyeEstimator()
    fixed.calibration = "off"  # blocks calibration → old fixed-threshold behaviour
    for i in range(600):
        fixed.update(narrow_open(i / fps))
    assert fixed.result()["GCS_eye_opening"] == 3, fixed.result()
    r = run([narrow_open] * 600)
    assert r["GCS_eye_opening"] == 4 and r["eye_calibration"] == "calibrated", r
    assert r["baseline_ear"] == 0.19, r

    # Wide-eyed patient (open EAR ≈ 0.40) who becomes drowsy (EAR 0.22).
    # Fixed thresholds would still say "open" → 4; calibrated → 3.
    def wide_open(t):
        return FrameCues(t, True, ear=0.40, blink_score=0.05, nose_xy=(0.5, 0.5))

    def drowsy(t):
        return FrameCues(t, True, ear=0.22, blink_score=0.35, nose_xy=(0.5, 0.5))

    r = run([wide_open] * 90 + [drowsy] * 810)
    assert r["GCS_eye_opening"] == 3 and r["baseline_ear"] == 0.40, r

    # Eyes never seen open → nothing to learn from → defaults are kept.
    r = run([closed_eyes] * 450)
    assert r["eye_calibration"] == "default" and r["baseline_ear"] is None, r

    # A saved baseline from an earlier session is reused straight away.
    saved = GCSEyeEstimator(baseline_ear=0.19)
    for i in range(300):
        saved.update(narrow_open(i / fps))
    r = saved.result()
    assert r["GCS_eye_opening"] == 4 and r["eye_calibration"] == "saved", r
    # Implausible saved baselines are ignored.
    assert GCSEyeEstimator(baseline_ear=0.05).calibration == "default"

    # Stress: relaxed face ~0, grimace with movement → high.
    relaxed = StressEstimator()
    for i in range(60):
        relaxed.update(FrameCues(i / fps, True, blendshapes={}, nose_xy=(0.5, 0.5)))
    assert relaxed.result()["stress_score"] < 0.5, relaxed.result()

    grimace_bs = {f"{n}{s}": 0.9 for n in ("browDown", "eyeSquint", "noseSneer", "eyeBlink")
                  for s in ("Left", "Right")}
    pained = StressEstimator()
    for i in range(90):
        jitter = 0.03 * math.sin(i)
        pained.update(FrameCues(i / fps, True, blendshapes=grimace_bs,
                                nose_xy=(0.5 + jitter, 0.5 - jitter)))
    assert pained.result()["stress_score"] > 7.0, pained.result()

    # A broad smile squints the eyes too, but must not read as pain. Values
    # are the real blendshapes MediaPipe gave for a smiling portrait.
    smile_bs = {"browDownLeft": 0.84, "browDownRight": 0.82, "eyeSquintLeft": 0.74,
                "eyeSquintRight": 0.66, "mouthUpperUpLeft": 0.76, "mouthUpperUpRight": 0.76,
                "eyeBlinkLeft": 0.27, "eyeBlinkRight": 0.26,
                "mouthSmileLeft": 0.96, "mouthSmileRight": 0.93}
    assert pspi_from_blendshapes(smile_bs) < 1.0, pspi_from_blendshapes(smile_bs)

    # EAR geometry: wide-open eye > nearly-closed eye.
    wide = np.array([[0, 0], [1, -1], [2, -1], [3, 0], [2, 1], [1, 1]], float)
    shut = np.array([[0, 0], [1, -0.1], [2, -0.1], [3, 0], [2, 0.1], [1, 0.1]], float)
    assert eye_aspect_ratio(wide) > EAR_OPEN_THRESHOLD > eye_aspect_ratio(shut)

    print("Self-test passed.")
    print("Example pained result:", json.dumps(pained.result()))


def main() -> None:
    ap = argparse.ArgumentParser(description="GCS eye + stress score from video (behavioural cues).")
    ap.add_argument("--source", default="0", help="camera index or video file path")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH, help="path to face_landmarker.task")
    ap.add_argument("--show", action="store_true", help="show live overlay window")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--baseline-ear", type=float, default=None,
                    help="reuse a patient's saved baseline_ear instead of re-calibrating")
    ap.add_argument("--download-model", action="store_true", help="download the model and exit")
    ap.add_argument("--self-test", action="store_true", help="run scoring-logic tests and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.download_model:
        download_model(args.model)
        return

    result = run_capture(args.source, args.model, args.show, args.max_seconds, args.baseline_ear)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

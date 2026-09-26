"""
cv_monitor.py  –  Real-time Stress Score & GCS Eye Score via Computer Vision
Standalone module for ICU CDSS capstone project.

Scoring engine (v2 — upgraded from geometric features to clinical metrics):

  STRESS SCORE (0–10)
      Uses the PSPI (Prkachin-Solomon Pain Intensity) — a peer-reviewed clinical
      formula built from facial Action Units (AUs) detected via MediaPipe
      blendshapes.  Covers 6 muscle groups simultaneously:
          PSPI = AU4 + max(AU6,AU7) + max(AU9,AU10) + AU43   (range 0–16)
      Smile-corrected: genuine smile dampens PSPI so laughter doesn't
      read as pain.  Head-movement agitation (nose tracking) blended in at 20 %.
      Smoothed with EMA (α=0.15) for responsive but stable output.
      No personal baseline needed — PSPI is an absolute metric.

  GCS EYE SCORE (E1–E4)
      30-second rolling behavioral window instead of a single-frame EAR snapshot.
      Personal eye calibration: learns each patient's own open-eye EAR from
      ~3 s of confirmed-open frames (cross-checked with eyeBlink blendshape).
      Maps the window's open fraction to the GCS E4/E3/E2/E1 criteria.

Install:
    pip install opencv-python mediapipe numpy

Run:
    python cv_monitor.py
    python cv_monitor.py --cam 1   # USB/external webcam
"""

import os
os.environ.setdefault("GLOG_minloglevel",    "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")

import sys
import time
import argparse
import pathlib
import urllib.request
from collections import deque

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
except ImportError:
    sys.exit("mediapipe not found.  Run:  pip install mediapipe")


# ── Model asset ───────────────────────────────────────────────────────────────
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = pathlib.Path(__file__).parent / "face_landmarker.task"


def _ensure_model() -> None:
    """Download the FaceLandmarker model on first run (~12 MB, once only)."""
    if not _MODEL_PATH.exists():
        print("[cv_monitor] Downloading face landmark model (~12 MB) — first run only…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[cv_monitor] Model ready.")


# ═══════════════════════════════════════════════════════════════════════════════
#  LANDMARK INDICES  (MediaPipe FaceMesh canonical 478-point model)
# ═══════════════════════════════════════════════════════════════════════════════
_L_EYE        = [33,  160, 158, 133, 153, 144]   # left  eye EAR points
_R_EYE        = [362, 385, 387, 263, 373, 380]   # right eye EAR points
_L_BROW_INNER = 107                               # left  inner brow (landmark overlay)
_R_BROW_INNER = 336                               # right inner brow
_NOSE_TIP     = 1                                 # nose tip — used for head-motion tracking


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
# EAR smoothing
_SMOOTH_WIN = 20          # rolling-mean window (frames) — ~0.7 s at 30 fps

# Stress / PSPI
_EMA_ALPHA   = 0.15       # exponential smoothing for stress EMA
_PSPI_MAX    = 16.0       # maximum raw PSPI score
_BLINK_CLOSED = 0.55      # eyeBlink blendshape above this → eye counted as closed
_MOTION_SAT  = 0.020      # nose displacement treated as maximum agitation
_FACIAL_W    = 0.80       # 80 % facial pain, 20 % head-motion agitation

# GCS rolling window + calibration
_GCS_WIN_S        = 30.0  # window length for GCS assessment
_GCS_CALIB_N      = 90    # clearly-open frames needed to learn baseline (~3 s)
_GCS_CALIB_BLINK  = 0.30  # eyeBlink above this → frame not "clearly open"
_GCS_CALIB_EAR_MIN = 0.15 # plausible range for a personal open-eye EAR
_GCS_CALIB_EAR_MAX = 0.50
_GCS_OPEN_DEFAULT  = 0.22 # fallback thresholds before calibration
_GCS_PART_DEFAULT  = 0.12
_GCS_OPEN_FRAC     = 0.70 # ≥ this fraction fully open  → E4
_GCS_ANY_FRAC      = 0.30 # ≥ this fraction open+partial → E3
_GCS_FLICKER_S     = 0.10 # longest continuous opening ≥ this → E2
_GCS_MIN_OBS_S     = 10.0 # face-video needed before GCS is considered final

# Calibration progress bar
_CALIB_FACE_N = 90        # face-frames shown in the 3-second progress bar

# BGR palette
_BG    = ( 28,  45,  68)
_BDR   = ( 65, 105, 150)
_WHITE = (230, 230, 230)
_MUTED = (130, 155, 175)
_GREEN = ( 80, 200,  80)
_AMBER = ( 40, 180, 255)
_RED   = ( 60,  80, 230)
_CYAN  = (200, 190,  60)


# ═══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _xy(lm, idx, w, h):
    p = lm[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)

def _pts(lm, idxs, w, h):
    return np.array([[lm[i].x * w, lm[i].y * h] for i in idxs], dtype=np.float32)

def _dist(a, b):
    return float(np.linalg.norm(a - b))

def _ear(lm, idxs, w, h):
    """Eye Aspect Ratio (Soukupová & Čech, 2016)."""
    p = _pts(lm, idxs, w, h)
    num = _dist(p[1], p[5]) + _dist(p[2], p[4])
    den = 2.0 * _dist(p[0], p[3]) + 1e-6
    return num / den


# ═══════════════════════════════════════════════════════════════════════════════
#  STRESS SCORING  —  PSPI + head movement + EMA
# ═══════════════════════════════════════════════════════════════════════════════
def _pspi_from_blendshapes(bs: dict) -> float:
    """
    Prkachin-Solomon Pain Intensity from MediaPipe blendshapes.
    Clinical formula: PSPI = AU4 + max(AU6,AU7) + max(AU9,AU10) + AU43
    Each AU is scaled 0–5 (matching FACS intensity).  Range 0–16.
    Smile-corrected: mouthSmile blendshape dampens the score to prevent
    happy eye-squinting from registering as pain.
    """
    def pair(name: str) -> float:
        return (bs.get(f"{name}Left", 0.0) + bs.get(f"{name}Right", 0.0)) / 2.0

    au4  = pair("browDown")     * 5.0   # brow lowerer      (furrowing)
    au6  = pair("cheekSquint")  * 5.0   # cheek raiser      (orbital squint)
    au7  = pair("eyeSquint")    * 5.0   # lid tightener     (eye narrowing)
    au9  = pair("noseSneer")    * 5.0   # nose wrinkler     (disgust / pain)
    au10 = pair("mouthUpperUp") * 5.0   # upper lip raiser  (grimace)
    au43 = 1.0 if pair("eyeBlink") > _BLINK_CLOSED else 0.0  # eyes shut

    pspi = au4 + max(au6, au7) + max(au9, au10) + au43
    return float(pspi * (1.0 - pair("mouthSmile")))   # smile correction


def _motion_energy(nose_track: deque) -> float:
    """Mean frame-to-frame nose displacement in normalised image coords."""
    if len(nose_track) < 2:
        return 0.0
    pts = np.array(nose_track, dtype=np.float32)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())


def _update_stress(pspi: float, motion: float, ema) -> tuple:
    """
    Blend PSPI (80 %) with agitation (20 %), scale to 0–10, apply EMA.
    Returns (display_score, new_ema).
    """
    facial    = min(pspi / _PSPI_MAX, 1.0)
    agitation = min(motion / _MOTION_SAT, 1.0)
    frame_s   = 10.0 * (_FACIAL_W * facial + (1 - _FACIAL_W) * agitation)
    new_ema   = frame_s if ema is None else (_EMA_ALPHA * frame_s + (1 - _EMA_ALPHA) * ema)
    return round(min(float(new_ema), 10.0), 1), new_ema


# ═══════════════════════════════════════════════════════════════════════════════
#  GCS EYE SCORING  —  rolling window + personal calibration
# ═══════════════════════════════════════════════════════════════════════════════
def _eye_state(ear: float, blink_score, open_th: float, partial_th: float) -> str:
    """
    Per-frame eye state using EAR cross-checked with eyeBlink blendshape.
    Returns 'open', 'partial', or 'closed'.
    """
    if blink_score is not None and blink_score > _BLINK_CLOSED:
        return "closed"
    if ear >= open_th:
        return "open"
    if ear >= partial_th:
        return "partial"
    return "closed"


def _gcs_from_window(history: deque) -> tuple:
    """
    GCS E1–E4 from a rolling deque of (timestamp_s, eye_state) pairs.
    Matches the clinical GCS behavioral criteria:
        E4 — eyes predominantly open  (≥ 70 % of window open)
        E3 — some opening, intermittent (≥ 30 % open or partial)
        E2 — brief flicker only  (longest opening ≥ 0.1 s)
        E1 — closed throughout
    Normal blinks inside open periods barely affect the open fraction.
    """
    if not history:
        return 4, "Spontaneous"

    states    = [s for _, s in history]
    n         = len(states)
    open_frac = sum(s == "open"   for s in states) / n
    any_frac  = sum(s != "closed" for s in states) / n

    if open_frac >= _GCS_OPEN_FRAC:
        return 4, "Spontaneous"
    if any_frac  >= _GCS_ANY_FRAC:
        return 3, "To Voice"

    # Longest continuous non-closed run ≥ 0.1 s → E2
    obs_s   = (history[-1][0] - history[0][0]) if len(history) >= 2 else 0.0
    frame_s = obs_s / (n - 1) if n > 1 else 1.0 / 30.0
    run = longest = 0
    for s in states:
        run     = run + 1 if s != "closed" else 0
        longest = max(longest, run)
    if longest * frame_s >= _GCS_FLICKER_S:
        return 2, "To Pain"

    return 1, "No Response"


# ═══════════════════════════════════════════════════════════════════════════════
#  LABEL / COLOUR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _stress_label(s: float):
    if s < 3.5: return "LOW",      _GREEN
    if s < 6.5: return "MODERATE", _AMBER
    return              "HIGH",     _RED

def _gcs_col(g: int):
    return (_RED, _RED, _AMBER, _GREEN)[g - 1]


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
_FONT = cv2.FONT_HERSHEY_SIMPLEX

def _t(img, txt, x, y, scale=0.5, col=_WHITE, thick=1):
    cv2.putText(img, str(txt), (x, y), _FONT, scale, col, thick, cv2.LINE_AA)

def _bar(img, x, y, w, h, frac, col, track=(40, 58, 78)):
    frac = float(np.clip(frac, 0, 1))
    cv2.rectangle(img, (x, y),               (x + w, y + h),             track, -1)
    cv2.rectangle(img, (x, y),               (x + int(w * frac), y + h), col,   -1)
    cv2.rectangle(img, (x, y),               (x + w, y + h),             _BDR,   1)


# ═══════════════════════════════════════════════════════════════════════════════
#  OVERLAY RENDERER
# ═══════════════════════════════════════════════════════════════════════════════
def _draw_overlay(frame, *, stress, s_lbl, s_col, comps,
                  gcs_n, gcs_lbl, gcs_col, ear_val,
                  gcs_thresholds, gcs_ready, stress_peak=0.0,
                  gcs_provisional=True, cal_pct=1.0, detected=True):
    """
    1280×800 HUD:
        Header   44 px  ·  Camera  516 px  ·  Panel  240 px
    comps  = (browDown, max(cheekSquint,eyeSquint), max(noseSneer,mouthUpperUp))
    gcs_thresholds = (open_th, partial_th) — personalised after calibration
    """
    H, W = frame.shape[:2]
    ph = 265          # expanded panel — fixes gauge/peak overlap and right-side clipping
    py = H - ph

    # ── backgrounds ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, py), (W, H), _BG, -1)
    cv2.line(frame, (0, py), (W, py), _BDR, 2)
    cv2.rectangle(frame, (0, 0), (W, 44), _BG, -1)
    cv2.line(frame, (0, 44), (W, 44), _BDR, 2)
    _t(frame, "ICU CDSS  |  Computer Vision Monitor  |  Stress Score & GCS Eye Score",
       14, 29, 0.60, (180, 210, 230))
    _t(frame, time.strftime("%H:%M:%S"), W - 108, 29, 0.55, _MUTED)

    # ── calibration overlay (first 3 s) ──────────────────────────────────────
    if cal_pct < 1.0:
        secs_left = max(0, int((1.0 - cal_pct) * 3 + 0.9))
        _t(frame, "CALIBRATING  —  Learning your personal eye baseline",
           24, py + 30, 0.72, _AMBER, 2)
        _t(frame, "Open your eyes normally and keep a neutral expression",
           24, py + 58, 0.56, _MUTED)
        _bar(frame, 24, py + 76, W - 48, 22, cal_pct, _AMBER)
        _t(frame, f"{int(cal_pct * 100)} %  complete   —   {secs_left}s remaining",
           24, py + 120, 0.60, _AMBER)
        if detected:
            _t(frame, f"Stress:  {stress:.1f} / 10   ({s_lbl})", 24, py + 188, 0.60, s_col)
        else:
            _t(frame, "No face detected — please move into camera view",
               24, py + 155, 0.54, _RED)
        return

    # ── no-face fallback ─────────────────────────────────────────────────────
    if not detected:
        _t(frame, "No face detected  —  centre your face in the camera frame",
           24, py + 60, 0.72, (110, 130, 155))
        return

    # ═════════════════════════════════════════════════════════════════════════
    #  LEFT HALF  —  STRESS SCORE  (PSPI-based)
    # ═════════════════════════════════════════════════════════════════════════
    lx = 24
    _t(frame, "STRESS  SCORE", lx, py + 24, 0.65, _MUTED)

    _t(frame, f"{stress:.1f}", lx, py + 90, 2.60, s_col, 3)
    _t(frame, "/ 10",          lx + 138, py + 82, 0.70, _MUTED)
    _t(frame, s_lbl,           lx, py + 116, 0.85, s_col, 2)

    _bar(frame, lx, py + 128, 340, 17, stress / 10.0, s_col)

    # session peak — clear 20px gap below gauge (gauge ends at py+145)
    peak_col = _RED if stress_peak >= 6.5 else _AMBER if stress_peak >= 3.5 else _GREEN
    _t(frame, f"Session peak:  {stress_peak:.1f}", lx, py + 162, 0.52, peak_col)

    # PSPI component bars (AU4 / AU6,7 / AU9,10)  — start at py+182
    _comp_labels = ["Brow Furrow ", "Eye Tension ", "Nose / Lip  "]
    for i, (lbl, val) in enumerate(zip(_comp_labels, comps)):
        cy  = py + 182 + i * 28
        col = _RED if val > 0.65 else _AMBER if val > 0.35 else _GREEN
        _t(frame, lbl,                         lx,       cy + 14, 0.55, _MUTED)
        _bar(frame, lx + 148, cy, 158, 16, val, col)
        _t(frame, f"{val * 100:.0f}%", lx + 314, cy + 14, 0.52, col)

    # ═════════════════════════════════════════════════════════════════════════
    #  RIGHT HALF  —  GCS EYE SCORE  (rolling window)
    # ═════════════════════════════════════════════════════════════════════════
    mid = W // 2
    cv2.line(frame, (mid, py + 12), (mid, H - 12), _BDR, 1)
    rx = mid + 24

    gcs_title = "GCS  EYE  SCORE  (CV proxy)"
    if not gcs_ready:
        gcs_title += "   [calibrating...]"
    _t(frame, gcs_title, rx, py + 24, 0.58, _MUTED)

    _t(frame, f"E{gcs_n}", rx,       py + 86, 2.60, gcs_col, 3)
    _t(frame, gcs_lbl,     rx + 105, py + 86, 0.80, gcs_col, 2)
    score_line = f"Score:  {gcs_n} / 4"
    if gcs_provisional:
        score_line += "   (provisional — < 10s)"
    _t(frame, score_line, rx + 105, py + 110, 0.50, gcs_col)

    dot_r, dot_gap = 20, 68
    for i in range(1, 5):
        cx_d   = rx + (i - 1) * dot_gap + 26
        cy_d   = py + 136
        active = (i == gcs_n)
        cv2.circle(frame, (cx_d, cy_d), dot_r, gcs_col if active else (48, 65, 82), -1)
        cv2.circle(frame, (cx_d, cy_d), dot_r, _BDR, 2)
        lbl_e = f"E{i}"
        thick = 2 if active else 1
        (tw, th), _ = cv2.getTextSize(lbl_e, _FONT, 0.50, thick)
        _t(frame, lbl_e, cx_d - tw // 2, cy_d + th // 2, 0.50,
           (10, 10, 10) if active else (80, 100, 122), thick)

    open_th, part_th = gcs_thresholds
    _t(frame, f"Eye Aspect Ratio (EAR):  {ear_val:.3f}",
       rx, py + 172, 0.54, _MUTED)
    _t(frame, f"Thresholds:  open≥{open_th:.2f}   partial≥{part_th:.2f}"
              + ("  (personal)" if gcs_ready else "  (default)"),
       rx, py + 196, 0.46, _MUTED)
    _t(frame, "E4:open≥70%   E3:any≥30%   E2:flicker≥0.1s   E1:closed",
       rx, py + 218, 0.44, (95, 118, 142))
    _t(frame, "* CV proxy  -  not a clinical diagnosis",
       rx, py + 248, 0.44, (85, 105, 128))


# ═══════════════════════════════════════════════════════════════════════════════
#  LANDMARK DRAWING  (eye contours + key dots)
# ═══════════════════════════════════════════════════════════════════════════════
def _draw_landmarks(frame, lm, W, H):
    for eye_idxs in (_L_EYE, _R_EYE):
        pts = _pts(lm, eye_idxs, W, H).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, _CYAN, 1, cv2.LINE_AA)
    for idx in set(_L_EYE + _R_EYE + [_L_BROW_INNER, _R_BROW_INNER]):
        cv2.circle(frame, (int(lm[idx].x * W), int(lm[idx].y * H)), 2, _CYAN, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def run(cam_index: int = 0) -> None:
    _ensure_model()

    # MediaPipe FaceLandmarker — blendshapes REQUIRED for PSPI + GCS calibration
    _base_opts = _mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
    _lm_opts   = _mp_vision.FaceLandmarkerOptions(
        base_options                  = _base_opts,
        running_mode                  = _mp_vision.RunningMode.VIDEO,
        num_faces                     = 1,
        output_face_blendshapes       = True,   # ← enables PSPI + GCS calib
        min_face_detection_confidence = 0.50,
        min_face_presence_confidence  = 0.50,
        min_tracking_confidence       = 0.50,
    )
    landmarker = _mp_vision.FaceLandmarker.create_from_options(_lm_opts)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[cv_monitor] ERROR: Cannot open camera (index {cam_index}).")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    _WIN, _DISP_W, _DISP_H = "ICU CDSS - CV Monitor", 1280, 800
    cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WIN, _DISP_W, _DISP_H)

    # ── state ─────────────────────────────────────────────────────────────────
    ear_q       = deque(maxlen=_SMOOTH_WIN)   # EAR rolling mean (for GCS)

    _stress_ema  = None                       # stress EMA accumulator
    _stress_peak = 0.0                        # highest stress reached this session
    _nose_track = deque(maxlen=30)            # nose positions for agitation

    _gcs_history    = deque()                 # (timestamp_s, eye_state)
    _gcs_calib_ear  = []                      # clearly-open EAR samples
    _gcs_calib_done = False
    _gcs_open_th    = _GCS_OPEN_DEFAULT       # personalised after calib
    _gcs_partial_th = _GCS_PART_DEFAULT

    _face_frames = 0                          # for 3-s progress bar

    # persisted display values
    stress_v    = 0.0
    comps_v     = (0.0, 0.0, 0.0)
    gcs_v       = 4
    gcs_lbl_v   = "Spontaneous"
    ear_display = 0.0

    print("[cv_monitor] Running — press  Q  or  ESC  to quit.")
    print(f"             Camera index: {cam_index}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int(time.time() * 1000)
        result   = landmarker.detect_for_video(mp_image, ts_ms)

        frame = cv2.resize(frame, (_DISP_W, _DISP_H))
        H, W  = frame.shape[:2]

        detected = bool(result.face_landmarks)

        if detected:
            lm = result.face_landmarks[0]
            _face_frames += 1

            # ── blendshapes ───────────────────────────────────────────────────
            bs = {}
            if result.face_blendshapes:
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]}

            # eyeBlink blendshape (average of left + right)
            blink_v = None
            if "eyeBlinkLeft" in bs:
                blink_v = (bs["eyeBlinkLeft"] + bs.get("eyeBlinkRight", 0.0)) / 2.0

            # ── EAR (for GCS) ─────────────────────────────────────────────────
            ear_v = (_ear(lm, _L_EYE, W, H) + _ear(lm, _R_EYE, W, H)) / 2.0
            ear_q.append(ear_v)
            ear_sm      = float(np.mean(ear_q))
            ear_display = ear_sm

            # ── GCS personal calibration ──────────────────────────────────────
            if not _gcs_calib_done:
                # only collect frames where eyeBlink says the eye is clearly open
                if blink_v is None or blink_v < _GCS_CALIB_BLINK:
                    _gcs_calib_ear.append(ear_v)
                if len(_gcs_calib_ear) >= _GCS_CALIB_N:
                    baseline = float(np.median(_gcs_calib_ear))
                    if _GCS_CALIB_EAR_MIN <= baseline <= _GCS_CALIB_EAR_MAX:
                        _gcs_open_th    = baseline * 0.70
                        _gcs_partial_th = baseline * 0.40
                        print(f"[cv_monitor] GCS calibrated: EAR baseline={baseline:.3f}"
                              f"  open≥{_gcs_open_th:.3f}  partial≥{_gcs_partial_th:.3f}")
                    _gcs_calib_done = True

            # ── head movement (agitation for stress) ──────────────────────────
            nose = lm[_NOSE_TIP]
            _nose_track.append((nose.x, nose.y))
            motion = _motion_energy(_nose_track)

            # ── PSPI stress + EMA ─────────────────────────────────────────────
            pspi     = _pspi_from_blendshapes(bs)
            stress_v, _stress_ema = _update_stress(pspi, motion, _stress_ema)
            _stress_peak = max(_stress_peak, stress_v)

            # ── PSPI component bars (each blendshape 0–1) ─────────────────────
            def pair(name: str) -> float:
                return (bs.get(f"{name}Left", 0.0) + bs.get(f"{name}Right", 0.0)) / 2.0

            comps_v = (
                pair("browDown"),                               # AU4  brow furrow
                max(pair("cheekSquint"), pair("eyeSquint")),   # AU6/7 eye tension
                max(pair("noseSneer"),   pair("mouthUpperUp")),# AU9/10 nose/lip
            )

            # ── GCS rolling window ────────────────────────────────────────────
            now_s  = time.time()
            eye_st = _eye_state(ear_sm, blink_v, _gcs_open_th, _gcs_partial_th)
            _gcs_history.append((now_s, eye_st))
            while _gcs_history and now_s - _gcs_history[0][0] > _GCS_WIN_S:
                _gcs_history.popleft()
            gcs_v, gcs_lbl_v = _gcs_from_window(_gcs_history)

            _draw_landmarks(frame, lm, W, H)

        # ── render HUD ────────────────────────────────────────────────────────
        cal_pct      = min(_face_frames / _CALIB_FACE_N, 1.0)
        s_lbl, s_col = _stress_label(stress_v)
        g_col        = _gcs_col(gcs_v)

        # GCS provisional: True until ≥ 10 s of continuous face observation
        _gcs_obs_s   = ((_gcs_history[-1][0] - _gcs_history[0][0])
                        if len(_gcs_history) >= 2 else 0.0)
        gcs_provisional = _gcs_obs_s < _GCS_MIN_OBS_S

        _draw_overlay(
            frame,
            stress          = stress_v,
            s_lbl           = s_lbl,
            s_col           = s_col,
            comps           = comps_v,
            gcs_n           = gcs_v,
            gcs_lbl         = gcs_lbl_v,
            gcs_col         = g_col,
            ear_val         = ear_display,
            gcs_thresholds  = (_gcs_open_th, _gcs_partial_th),
            gcs_ready       = _gcs_calib_done,
            stress_peak     = _stress_peak,
            gcs_provisional = gcs_provisional,
            cal_pct         = cal_pct,
            detected        = detected,
        )

        cv2.imshow(_WIN, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[cv_monitor] Session ended.")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ICU CDSS — Real-time Stress & GCS Eye Score via Computer Vision"
    )
    parser.add_argument("--cam", type=int, default=0,
                        help="Camera device index (default: 0 = built-in webcam)")
    args = parser.parse_args()
    run(args.cam)

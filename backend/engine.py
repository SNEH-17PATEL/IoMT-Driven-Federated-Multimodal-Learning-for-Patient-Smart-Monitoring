"""
Prediction engine for the ICU / Home-Care monitor.

All clinical logic here is the logic of the original Streamlit app (app.py), moved
into an importable module so any front end can use it: SOFA prediction, the 4 runtime
validation checks, SHAP explanations, the LLM clinical report and its 9-method
reliability score. The helper functions between the two markers below are copied
verbatim from app.py.
"""
import os
import re
import sys
import json
import time
import atexit
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
import shap
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_utils import ICUModel, get_trend, classify_range          # noqa: E402
from db import (                                                      # noqa: E402
    init_db, reset_patient,
    get_patient_vitals, append_patient_vitals,
    get_prediction_history, append_prediction,
    get_sample_patient_row, get_sample_patient_count,
    get_fl_client_data, fl_training_exists,
)

load_dotenv(ROOT / ".env")

MODEL_PATH      = str(ROOT / "models") + os.sep
ALERT_THRESHOLD = 8.0

NORMAL_RANGES = {
    "HR":   (60,  100),
    "RR":   (12,  20),
    "SpO2": (95,  100),
    "Temp": (36.5, 37.5),
    "SBP":  (100, 120),
    "DBP":  (60,  80),
    "MAP":  (70,  100),
}


def risk_label(sofa):
    if sofa < 5:
        return "Low Risk", "🟢", "#28a745"
    elif sofa < 10:
        return "Moderate Risk", "🟡", "#e6a817"
    return "High Risk", "🔴", "#dc3545"


# ═══ BEGIN: helpers copied verbatim from app.py ══════════════════════════════════

def _strip_thinking(text: str) -> str:
    """
    Remove chain-of-thought blocks from models like Qwen3 that emit <think>…</think>.

    Two cases handled:
    1. Properly closed: <think>…</think> → regex strips the whole block.
    2. Unclosed / truncated: <think> with no </think> (hit max_tokens during thinking)
       → everything from <think> onward is discarded because the actual response
         was never reached.  The caller detects an empty result and uses a fallback.
    """
    if "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    elif "<think>" in text:
        # Entire response is inside an unclosed thinking block — nothing useful remains
        text = text[: text.find("<think>")]
    return text.strip()


_FALLBACK_MSG = (
    "⚠️ The AI model ran out of tokens during its internal reasoning process "
    "and did not produce a clinical assessment for this reading. "
    "This can happen when the model's chain-of-thought exceeds the token limit. "
    "The next automatic reading (in ~30 s) will generate a fresh response."
)


def get_multiple_llm_responses(api_key, prompt, n=3):
    client = Groq(api_key=api_key)
    responses = []
    for _ in range(n):
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert ICU clinical decision support assistant. "
                        "Provide concise, structured, and clinically accurate reasoning."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        text = _strip_thinking(resp.choices[0].message.content)
        responses.append(text if text else _FALLBACK_MSG)
    return responses


# ── Clinical Decision Agreement ─────────────────────────────────────────────
_INTERVENTIONS = {
    "vasopressors":  ["vasopressor", "norepinephrine", "dopamine", "epinephrine", "vasopressin"],
    "antibiotics":   ["antibiotic", "antimicrobial", "empiric", "broad-spectrum"],
    "fluid":         ["fluid", "crystalloid", "bolus", "resuscitat"],
    "oxygen":        ["oxygen", "ventilat", "intubat", "high-flow", "fio2"],
    "monitoring":    ["monitor", "arterial line", "reassess", "continuous"],
    "labs":          ["culture", "lactate", "creatinine", "cbc", "labs"],
    "renal_support": ["dialysis", "crrt", "diuretic", "furosemide"],
}

_CONDITIONS = {
    "septic_shock": ["septic shock", "septicemia"],
    "infection":    ["sepsis", "infection", "bacteremia", "infectious"],
    "ards":         ["ards", "respiratory distress", "respiratory failure"],
    "aki":          ["acute kidney", "renal failure", "renal impairment", "oliguria"],
    "hypoxemia":    ["hypoxemia", "hypoxia"],
    "hypotension":  ["hypotension", "low blood pressure", "map"],
    "tachycardia":  ["tachycardia"],
    "urgency":      ["immediate", "urgent", "emergent", "critical"],
}


def _category_agreement(responses_lower, term_dict):
    n = len(responses_lower)
    scores = []
    for terms in term_dict.values():
        count = sum(any(t in resp for t in terms) for resp in responses_lower)
        scores.append(max(count, n - count) / n)
    return float(np.mean(scores))


def compute_consistency(responses):
    """
    Three-component reliability metric:
      20% TF-IDF cosine similarity      (word-level phrasing overlap)
      50% Intervention agreement        (do all 3 agree on which treatments?)
      30% Condition/diagnosis agreement (do all 3 identify the same pathologies?)
    """
    if len(responses) < 2:
        return 0.0
    n = len(responses)

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(responses)
    sim = cosine_similarity(vec)
    tfidf_score = (sim.sum() - n) / (n * (n - 1))

    responses_lower = [r.lower() for r in responses]
    intervention_score = _category_agreement(responses_lower, _INTERVENTIONS)
    condition_score    = _category_agreement(responses_lower, _CONDITIONS)

    combined = (
        0.20 * tfidf_score
        + 0.50 * intervention_score
        + 0.30 * condition_score
    )
    return float(np.clip(combined, 0.0, 1.0))


# =============================================================
# SOFA VALIDATION HELPERS
# =============================================================

def compute_sofa_floor(MAP_val, SpO2_val, GCS_eye_val):
    """
    Minimum SOFA score guaranteed by directly-measured vitals.
    Only covers SOFA components 1 (respiratory/SpO₂), 4 (cardiovascular/MAP),
    5 (CNS/GCS). Missing components (hepatic, coagulation, renal) mean the
    true SOFA floor is always at least this high.
    """
    floor = 0
    if MAP_val  < 70: floor += 1
    if MAP_val  < 65: floor += 1
    if SpO2_val < 94: floor += 1
    if SpO2_val < 90: floor += 1
    if GCS_eye_val == 2: floor += 2
    if GCS_eye_val == 1: floor += 3
    return floor


def vital_consistency_flags(HR_val, SBP_val, SpO2_val, MAP_val, sofa_val):
    """
    Detect combined vital-sign syndrome patterns that imply a SOFA minimum
    the individual-vital floor (compute_sofa_floor) would miss.
    Returns a list of warning strings; empty = no inconsistency detected.
    """
    flags = []
    if HR_val > 130 and SBP_val < 90 and sofa_val < 4:
        flags.append("HR > 130 + SBP < 90 implies shock state — SOFA expected ≥ 4")
    if SpO2_val < 88 and MAP_val < 65 and sofa_val < 6:
        flags.append("SpO₂ < 88% + MAP < 65 implies multi-organ stress — SOFA expected ≥ 6")
    return flags


def check_trajectory(patient_id_arg, cur_sofa, HR_val, RR_val, SpO2_val, SBP_val, MAP_val):
    """
    Returns (prev_sofa, jump, flagged, message).
    Reads the last 2 predictions from the database.
    Flags SOFA changes > 4 pts between successive readings when no vital sign
    changed significantly — indicates model instability or a data entry error.
    """
    try:
        hist = get_prediction_history(patient_id_arg, limit=2)
    except Exception:
        return None, 0.0, False, "History unavailable"
    if len(hist) < 2:
        return None, 0.0, False, "First reading"

    prev      = hist.iloc[-1]    # most recent prior prediction
    prev_sofa = float(prev.get("SOFA", cur_sofa))
    jump      = cur_sofa - prev_sofa

    sig = sum([
        abs(HR_val   - float(prev.get("HR",     HR_val)))   >= 20,
        abs(RR_val   - float(prev.get("RR",     RR_val)))   >= 4,
        abs(SpO2_val - float(prev.get("SpO₂",   SpO2_val))) >= 5,
        abs(SBP_val  - float(prev.get("SBP",    SBP_val)))  >= 20,
        abs(MAP_val  - float(prev.get("MAP",     MAP_val)))  >= 15,
    ])

    if abs(jump) > 4 and sig == 0:
        direction = "↑" if jump > 0 else "↓"
        return prev_sofa, jump, True, (
            f"SOFA {direction} {abs(jump):.1f} pts "
            f"({prev_sofa:.1f} → {cur_sofa:.1f}) with no significant vital sign change — "
            "verify input data"
        )
    return prev_sofa, jump, False, None


# =============================================================
# LLM OUTPUT VALIDATION HELPERS  (8 methods, 2 LLM calls total)
# =============================================================

# ── Method 1: Factual Grounding (B1) ─────────────────────────────────────────
def factual_grounding_score(response, vitals, sofa_val):
    """
    Returns 0.0–1.0: fraction of abnormal vitals acknowledged in response.
    Checks problem identification (diagnostic layer).
    Reference: FactEHR (NEJM AI 2025).
    """
    r      = response.lower()
    checks = []
    if vitals.get("SpO2", 100) < 90:
        checks.append(any(t in r for t in [
            "hypox", "oxygen", "o2", "spo2", "saturation",
            "fio2", "ventilat", "respiratory", "breathing",
        ]))
    if vitals.get("MAP", 80) < 65:
        checks.append(any(t in r for t in [
            "hypotension", "vasopressor", "fluid", "resuscitat",
            "pressure", "map", "pressor", "norepinephrine", "dopamine",
        ]))
    if vitals.get("HR", 80) > 100:
        checks.append(any(t in r for t in [
            "tachycardia", "heart rate", "hr", "pulse", "cardiac",
        ]))
    if vitals.get("RR", 16) > 20:
        checks.append(any(t in r for t in [
            "tachypnea", "respiratory", "breathing", "rr", "breath", "ventilat",
        ]))
    if sofa_val >= 10:
        checks.append(any(t in r for t in [
            "high", "severe", "critical", "emergent", "immediate", "urgent",
        ]))
    elif sofa_val >= 5:
        checks.append(any(t in r for t in [
            "moderate", "significant", "concerning", "monitor", "attention",
        ]))
    return sum(checks) / len(checks) if checks else 1.0


# ── Method 2: Clinical Hard Rules (C1) ───────────────────────────────────────
_LLM_HARD_RULES = [
    {
        "condition": lambda v, s: v.get("SpO2", 100) < 90,
        "required":  ["oxygen", "ventilat", "intubat", "fio2",
                      "high-flow", "supplemental", "o2", "respiratory support"],
        "rule":      "SpO₂ < 90% → must mention oxygen therapy",
    },
    {
        "condition": lambda v, s: v.get("MAP", 80) < 65,
        "required":  ["vasopressor", "norepinephrine", "epinephrine", "dopamine",
                      "phenylephrine", "fluid", "resuscitat", "hypotension", "pressor"],
        "rule":      "MAP < 65 mmHg → must mention vasopressors or fluid resuscitation",
    },
    {
        "condition": lambda v, s: v.get("HR", 80) > 130 and v.get("SBP", 120) < 90,
        "required":  ["shock", "fluid", "vasopressor", "resuscit", "hemodynamic", "pressor"],
        "rule":      "HR > 130 + SBP < 90 → must address shock state",
    },
    {
        "condition": lambda v, s: s >= 10,
        "required":  ["immediate", "urgent", "critical", "emergent", "priority", "severe"],
        "rule":      "SOFA ≥ 10 → must use urgency language",
    },
    {
        "condition": lambda v, s: v.get("stress", 0) > 7,
        "required":  ["pain", "sedation", "agitation", "comfort", "distress", "analgesia"],
        "rule":      "Stress Score > 7 → must mention pain management or sedation",
    },
    {
        "condition": lambda v, s: v.get("GCS_eye", 4) == 1,
        "required":  ["consciousness", "gcs", "neurological", "unresponsive", "coma", "glasgow"],
        "rule":      "GCS Eye = 1 → must mention neurological assessment",
    },
]


def llm_clinical_rules_score(response, vitals, sofa_val):
    """
    Returns (compliance 0.0–1.0, list_of_violated_rules).
    Protocol-based check from Surviving Sepsis Campaign + ACLS guidelines.
    Strongest fix for 'all 3 wrong': catches consensus hallucinations.
    """
    r     = response.lower()
    fired = [rule for rule in _LLM_HARD_RULES if rule["condition"](vitals, sofa_val)]
    if not fired:
        return 1.0, []
    viols = [rule["rule"] for rule in fired
             if not any(t in r for t in rule["required"])]
    return 1.0 - (len(viols) / len(fired)), viols


# ── Method 3: SHAP-LLM Coherence (B3) ───────────────────────────────────────
_SHAP_TERMS = {
    "latest_SpO2":     ["spo2", "oxygen", "hypox", "saturation", "respiratory", "o2"],
    "SpO2_mean":       ["spo2", "oxygen", "hypox", "saturation"],
    "SpO2_min":        ["spo2", "oxygen", "hypox"],
    "latest_MAP":      ["map", "blood pressure", "hypotension", "vasopressor", "pressor"],
    "MAP_mean":        ["map", "blood pressure", "hypotension"],
    "latest_HR":       ["heart rate", "hr", "tachycardia", "pulse", "cardiac"],
    "latest_RR":       ["respiratory", "breathing", "tachypnea", "rr", "breath"],
    "GCS_eye_opening": ["gcs", "consciousness", "neurological", "glasgow"],
    "stress_score":    ["stress", "pain", "agitation", "distress", "comfort"],
}


def shap_coherence_score(response, top_shap_df):
    """
    Returns 0.0–1.0: fraction of high-impact SHAP features addressed in response.
    Measures alignment between the model's prediction drivers and LLM reasoning.
    Returns 0.5 (neutral) when SHAP is unavailable.
    """
    if top_shap_df is None or top_shap_df.empty:
        return 0.5
    r      = response.lower()
    scores = []
    for _, row in top_shap_df.iterrows():
        feat   = row["feature"]
        impact = row["impact"]
        if abs(impact) < 0.1:
            continue
        terms = _SHAP_TERMS.get(feat, [feat.lower().replace("_", " ")])
        scores.append(float(any(t in r for t in terms)))
    return sum(scores) / len(scores) if scores else 0.5


# ── Method 4: Response Structure ─────────────────────────────────────────────
def response_structure_score(response):
    """
    Returns 0.0–1.0: 70% section presence + 30% length substantiveness.
    Required: CURRENT CONDITION, PROBABLE CAUSE, RISK FORECAST, IMMEDIATE ACTIONS.
    """
    REQUIRED = [
        "current condition",
        "probable cause",
        "risk forecast",
        "immediate action",
    ]
    r      = response.lower()
    found  = sum(1 for sec in REQUIRED if sec in r)
    length = min(1.0, len(response.split()) / 200)
    return (found / len(REQUIRED)) * 0.70 + length * 0.30


# ── Method 5: Severity Calibration (bidirectional) ───────────────────────────
def severity_calibration_score(response, sofa_val):
    """
    Returns 0.0–1.0. Bidirectional:
    - High SOFA (≥10) without urgency language → fail
    - Low SOFA (<5) with panic language only → overcalibrated → partial fail
    """
    r       = response.lower()
    urgency = any(t in r for t in [
        "immediate", "urgent", "critical", "emergent", "emergenc",
        "severe", "life-threatening", "danger",
    ])
    calm    = any(t in r for t in [
        "stable", "monitor", "reassess", "routine", "continue",
        "improve", "recovering", "adequate",
    ])
    if sofa_val >= 10:
        return 1.0 if urgency else 0.2
    elif sofa_val >= 5:
        return 1.0
    else:
        return 0.4 if (urgency and not calm) else 1.0


# ── Method 6: Contraindication Safety Check (C2) ─────────────────────────────
_CONTRA_RULES = [
    {
        "condition": lambda v: v.get("MAP", 80) < 65 and v.get("HR", 80) > 100,
        "forbidden": ["beta-blocker", "metoprolol", "atenolol", "carvedilol",
                      "propranolol", "labetalol"],
        "flag": "Beta-blockers contraindicated in cardiogenic shock (MAP<65 + HR>100)",
    },
    {
        "condition": lambda v: v.get("SpO2", 100) < 90,
        "forbidden": ["morphine", "opioid", "benzodiazepine", "midazolam", "lorazepam"],
        "flag": "Unprotected opioids/benzodiazepines contraindicated in severe hypoxia (SpO₂<90%)",
    },
    {
        "condition": lambda v: True,
        "forbidden": ["nsaid", "ibuprofen", "diclofenac"],
        "flag": "NSAIDs contraindicated in ICU patients (renal/GI risk)",
    },
]


def contraindication_score(response, vitals):
    """Returns (1.0 safe / 0.0 dangerous, list_of_flags)."""
    r     = response.lower()
    flags = [rule["flag"] for rule in _CONTRA_RULES
             if rule["condition"](vitals) and any(t in r for t in rule["forbidden"])]
    return (0.0 if flags else 1.0), flags


# ── Method 9: Numeric Hallucination Check ────────────────────────────────────
# Regex patterns adapted from teammate's llm_safety.py (test/ folder).
_NUM_LABELED = re.compile(
    r"\b(?:sofa(?:\s+score)?|heart\s*rate|hr|respiratory\s*rate|rr|"
    r"spo2|oxygen\s*saturation|temperature|temp|sbp|dbp|map|"
    r"gcs(?:\s+eye)?|stress\s*score)\b[^\d+-]{0,24}([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_NUM_UNIT = re.compile(
    r"([+-]?\d+(?:\.\d+)?)\s*"
    r"(?:mmhg|bpm|br/min|breaths?\s*(?:/|per)\s*min|"
    r"°\s?[cf]|celsius|fahrenheit|%|percent)(?![a-z])",
    re.IGNORECASE,
)
_NUM_BARE = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])")


def _decimal_set(text):
    out = set()
    for m in _NUM_BARE.finditer(text):
        try:
            out.add(Decimal(m.group()).normalize())
        except InvalidOperation:
            pass
    return out


def numeric_hallucination_score(response, vitals, sofa_val):
    """
    Returns (score 0.0–1.0, list_of_unsupported_numbers).

    Extracts clinical measurements (labeled or unit-tagged) from the response
    and checks them against the actual input vitals.

    Tolerances:
      - ±10% of the actual value  (handles natural clinical rounding)
      - ±3 units absolute slack   (handles e.g. "BP ~75" when MAP=72)

    Clinical reference thresholds (MAP 65, SpO₂ 90, HR 100, etc.) are
    always allowed because they appear naturally in any ICU assessment text
    as protocol references, not as patient-specific hallucinated values.

    Method: adapted from teammate's llm_safety.py unsupported measurement check.
    """
    actual = {
        float(vitals.get("HR",      80)),
        float(vitals.get("RR",      16)),
        float(vitals.get("SpO2",   100)),
        float(vitals.get("Temp",  37.0)),
        float(vitals.get("SBP",   120)),
        float(vitals.get("DBP",    80)),
        float(vitals.get("MAP",    80)),
        float(vitals.get("GCS_eye", 4)),
        float(vitals.get("stress",  0)),
        float(sofa_val),
    }

    # Standard ICU clinical reference thresholds and protocol values.
    # These appear naturally in clinical text as guidelines, targets, and
    # protocol references — NOT as hallucinated patient-specific values.
    # Examples: "MAP > 65 mmHg", "30 mL/kg fluid bolus", "30 breaths/min tachypnea"
    _CLINICAL_REFS = {
        # Common ICU threshold numbers (Surviving Sepsis Campaign, ACLS, etc.)
        60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 92.0, 95.0, 100.0,
        120.0, 130.0, 140.0, 150.0, 160.0,
        # Respiratory reference values (RR thresholds, FiO₂ targets)
        20.0, 25.0, 30.0, 35.0, 40.0,
        # Temperature reference values (°C)
        36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 40.0,
        # Small clinical values (GCS components, SOFA tiers, O₂ flow rates)
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0,
        # Common ICU protocol round numbers (fluid doses, drug rates, targets)
        0.5, 1.5, 2.0, 2.5, 50.0, 24.0,
    }

    def _is_allowed(val_str):
        try:
            v = float(Decimal(val_str).normalize())
        except InvalidOperation:
            return True
        if v in _CLINICAL_REFS:
            return True
        for a in actual:
            tol = max(abs(a) * 0.10, 3.0)   # 10% or ±3 units
            if abs(v - a) <= tol:
                return True
        return False

    measured = []
    for pattern in (_NUM_LABELED, _NUM_UNIT):
        for m in pattern.finditer(response):
            measured.append(m.group(1))

    if not measured:
        return 1.0, []

    seen = set()
    unsupported = []
    for val in measured:
        if val not in seen:
            seen.add(val)
            if not _is_allowed(val):
                unsupported.append(val)

    score = max(0.0, 1.0 - len(unsupported) * 0.25)
    return score, unsupported


# ── Methods 7+8: Judge LLM — G-Eval (F1) + RAGAS Faithfulness (F2) ───────────
def call_judge_llm(api_key, main_response, vitals, sofa_val):
    """
    Second LLM call (lightweight llama-3.1-8b-instant).
    Combines G-Eval clinical scoring (F1) + RAGAS faithfulness (F2)
    in one structured JSON prompt.
    Returns parsed dict or None on failure — callers must handle None gracefully.
    """
    client = Groq(api_key=api_key)
    v_str  = (
        f"SpO₂={vitals.get('SpO2','?')}%, MAP={vitals.get('MAP','?')} mmHg, "
        f"HR={vitals.get('HR','?')} bpm, RR={vitals.get('RR','?')} br/min, "
        f"SBP={vitals.get('SBP','?')} mmHg, SOFA={sofa_val:.1f}/24, "
        f"Stress={vitals.get('stress','?')}/10, GCS Eye={vitals.get('GCS_eye','?')}/4"
    )
    prompt = f"""You are a clinical AI quality evaluator for an ICU decision support system.

PATIENT VITALS (ground truth input to the AI):
{v_str}

AI CLINICAL RESPONSE TO EVALUATE:
{main_response[:1500]}

Output ONLY a valid JSON object — no markdown, no explanation, no extra text:
{{
  "factual_accuracy": <integer 1-5>,
  "clinical_appropriateness": <integer 1-5>,
  "urgency_calibration": <integer 1-5>,
  "completeness": <integer 1-5>,
  "hypoxemia_addressed": <true or false>,
  "hypotension_addressed": <true or false>,
  "tachycardia_addressed": <true or false>,
  "unsupported_claims": [<list any statements contradicted by the vitals, or empty list []>],
  "dangerous_recommendation_detected": <true or false>,
  "dangerous_details": "<one-line description or empty string>"
}}

Scoring guide (be strict, not generous):
- factual_accuracy 1-5: 5=all vitals correctly interpreted, 1=major errors about patient values
- clinical_appropriateness 1-5: 5=textbook ICU management, 1=inappropriate or dangerous interventions
- urgency_calibration 1-5: 5=urgency language matches SOFA exactly, 1=completely wrong urgency
- completeness 1-5: 5=all 4 required sections present and detailed, 1=missing or superficial"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        text = _strip_thinking(resp.choices[0].message.content.strip())
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(text[s:e])
    except Exception:
        pass
    return None


# ── Combined 8-Method Reliability Score ──────────────────────────────────────
def compute_reliability_new(response, vitals, sofa_val, top_shap_df, judge_result):
    """
    9-component LLM output reliability score.
    Returns (combined_score 0-1, breakdown_dict, violations_list).

    Weights (sum = 1.00):
      7 deterministic:    0.65 total
        Factual Grounding    0.13
        Clinical Hard Rules  0.13
        SHAP Coherence       0.09
        Response Structure   0.09
        Severity Calibration 0.08
        Contraindication     0.05
        Numeric Accuracy     0.08  ← teammate's unsupported-measurement check
      2 judge LLM:         0.35 total
        G-Eval Score         0.13
        RAGAS Faithfulness   0.22
    """
    fg               = factual_grounding_score(response, vitals, sofa_val)
    chr_s, chr_viols = llm_clinical_rules_score(response, vitals, sofa_val)
    shap             = shap_coherence_score(response, top_shap_df)
    struct           = response_structure_score(response)
    sev              = severity_calibration_score(response, sofa_val)
    contra_s, contra_flags = contraindication_score(response, vitals)
    num_s, num_flags = numeric_hallucination_score(response, vitals, sofa_val)

    unsup = []
    if judge_result:
        geval_raw = (
            int(judge_result.get("factual_accuracy",         3)) +
            int(judge_result.get("clinical_appropriateness",  3)) +
            int(judge_result.get("urgency_calibration",       3)) +
            int(judge_result.get("completeness",              3))
        ) / 4.0
        geval = (geval_raw - 1.0) / 4.0   # 1–5 → 0.0–1.0
        unsup = judge_result.get("unsupported_claims", [])
        ragas = max(0.0, 1.0 - len(unsup) * 0.30)
        if judge_result.get("dangerous_recommendation_detected", False):
            contra_s = 0.0
            detail   = judge_result.get("dangerous_details", "")
            if detail:
                contra_flags = list(contra_flags) + [f"AI Judge: {detail}"]
    else:
        geval, ragas = 0.5, 0.5

    combined = float(np.clip(
        0.13 * fg    + 0.13 * chr_s + 0.09 * shap +
        0.09 * struct + 0.08 * sev  + 0.05 * contra_s +
        0.08 * num_s +
        0.13 * geval + 0.22 * ragas,
        0.0, 1.0,
    ))

    breakdown = {
        "Factual Grounding":       round(fg, 2),
        "Clinical Hard Rules":     round(chr_s, 2),
        "SHAP Coherence":          round(shap, 2),
        "Response Structure":      round(struct, 2),
        "Severity Calibration":    round(sev, 2),
        "Contraindication Check":  round(contra_s, 2),
        "Numeric Accuracy":        round(num_s, 2),
        "G-Eval Clinical Quality": round(geval, 2),
        "RAGAS Faithfulness":      round(ragas, 2),
    }
    all_viols = list(chr_viols) + list(contra_flags)
    for n in num_flags[:3]:
        all_viols.append(f"Hallucinated number in response: {n}")
    for c in unsup[:2]:
        all_viols.append(f"Unsupported claim: {c}")

    return combined, breakdown, all_viols


# Maps vital/CV feature names → (display label, unit string)
VITAL_UNITS = {
    "HR_mean":        ("HR mean",           "bpm"),
    "HR_std":         ("HR variability",     "bpm"),
    "RR_mean":        ("RR mean",            "br/min"),
    "SpO2_mean":      ("SpO₂ mean",          "%"),
    "SpO2_min":       ("SpO₂ minimum",       "%"),
    "Temp_mean":      ("Temperature mean",   "°C"),
    "SBP_mean":       ("SBP mean",           "mmHg"),
    "DBP_mean":       ("DBP mean",           "mmHg"),
    "MAP_mean":       ("MAP mean",           "mmHg"),
    "latest_HR":      ("Heart Rate",         "bpm"),
    "latest_RR":      ("Respiratory Rate",   "br/min"),
    "latest_SpO2":    ("SpO₂",              "%"),
    "latest_Temp":    ("Temperature",        "°C"),
    "latest_SBP":     ("Systolic BP",        "mmHg"),
    "latest_DBP":     ("Diastolic BP",       "mmHg"),
    "latest_MAP":     ("Mean Arterial P.",   "mmHg"),
    "GCS_eye_opening":("GCS Eye Opening",    "/4"),
    "stress_score":   ("Stress Score",       "/10"),
}


def interpret_shap(feature, orig_value, impact):
    direction = "increasing risk" if impact > 0 else "reducing risk"
    f = feature.lower()

    if feature in VITAL_UNITS:
        label, unit = VITAL_UNITS[feature]
        fmt = ".0f" if unit in ("/4", "/10") else ".1f"
        val_str = f"{orig_value:{fmt}}"
        if "spo2" in f:
            return f"{label} = {val_str}{unit} → low oxygen levels, {direction}"
        if "hr" in f:
            return f"{label} = {val_str}{unit} → abnormal heart rate, {direction}"
        if "rr" in f:
            return f"{label} = {val_str}{unit} → respiratory distress, {direction}"
        if any(x in f for x in ("sbp", "dbp", "map")):
            return f"{label} = {val_str}{unit} → blood pressure instability, {direction}"
        if "temp" in f:
            return f"{label} = {val_str}{unit} → possible infection / fever, {direction}"
        if "gcs" in f:
            return f"{label} = {val_str}{unit} → neurological deterioration, {direction}"
        if "stress" in f:
            return f"{label} = {val_str}{unit} → elevated physiological stress, {direction}"
        return f"{label} = {val_str}{unit} → {direction}"

    return f'Clinical note contains "{feature}" → {direction}'


RISK_MAP = {
    "SpO2":        "Low oxygen levels",
    "RR":          "Respiratory distress",
    "respiratory": "Respiratory distress",
    "failure":     "Organ failure",
    "septic":      "Sepsis / Infection",
    "intubated":   "Respiratory failure (intubated)",
    "vasopressor": "Haemodynamic instability",
    "SBP":         "Hypotension",
    "DBP":         "Hypotension",
    "MAP":         "Hypotension",
    "hypotension": "Hypotension",
    "HR":          "Abnormal heart rate",
    "mental":      "Altered mental status",
    "GCS":         "Neurological deterioration",
    "stress":      "High physiological stress",
}

def risk_factors_for(feature):
    """
    RISK_MAP labels that apply to one feature, matched on whole words.
    Plain substring matching mislabelled terms such as "norepinephrine"
    (contains "hr") as "Abnormal heart rate".
    """
    words = feature.lower().replace("_", " ")
    return [v for k, v in RISK_MAP.items() if re.search(r"\b" + re.escape(k.lower()) + r"\b", words)]


CLINICAL_KEYWORDS = [
    "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP", "GCS", "stress",
    "hypotension", "respiratory", "mental", "septic", "failure",
    "intubated", "vasopressor", "shock", "fever", "infection",
    "oxygen", "ventilat", "cardiac", "renal", "hepatic",
]

# ═══ END: helpers copied verbatim from app.py ════════════════════════════════════


_FALLBACK_SECTIONS = ("IMMEDIATE ACTIONS", "CURRENT CONDITION", "PROBABLE CAUSE", "RISK FORECAST")


def parse_llm_sections(text):
    """
    Split LLM text into {SECTION_NAME: content}. Handles numbered ("1. CURRENT CONDITION")
    and bare headers, with or without ** bold markers or trailing punctuation.
    """
    positions = []
    for sec in _FALLBACK_SECTIONS:
        m = re.search(
            rf'(?:^|(?<=\n))[ \t]*(?:\d+\.\s*)?\*{{0,2}}\s*{re.escape(sec)}\s*\*{{0,2}}\s*[—\-:–]?\s*',
            text, re.IGNORECASE
        )
        if m:
            positions.append((m.start(), sec, m.end()))
    positions.sort(key=lambda x: x[0])
    sections = {}
    for i, (_, name, end) in enumerate(positions):
        nxt = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections[name] = text[end:nxt].strip()
    return sections


def build_prompt(sofa_score, risk_text, clinical_note, HR, RR, SpO2, Temp, SBP, DBP, MAP,
                 GCS_eye, stress, key_risks, clinical_explanations, vitals_df):
    """Trend analysis + LLM prompt (same text as the original app)."""
    VITAL_CFG = [
        ("HR",   "HR",          60,   100),
        ("RR",   "RR",          12,   20),
        ("SpO2", "SpO₂",        95,   100),
        ("Temp", "Temperature", 36.5, 37.5),
        ("SBP",  "SBP",         100,  120),
        ("DBP",  "DBP",         60,   80),
        ("MAP",  "MAP",         70,   100),
    ]
    trend_lines = []
    for col, label, lo, hi in VITAL_CFG:
        direction = get_trend(vitals_df[col].tolist())
        status    = classify_range(vitals_df[col].mean(), lo, hi)
        trend_lines.append(f"{label} → {status} & {direction}")
    trend_text = "\n".join(f"- {t}" for t in trend_lines)

    vitals_block = (
        f"HR: {HR} bpm (normal: 60–100)\n"
        f"RR: {RR} breaths/min (normal: 12–20)\n"
        f"SpO₂: {SpO2}% (normal: 95–100)\n"
        f"Temperature: {Temp} °C (normal: 36.5–37.5)\n"
        f"Blood Pressure: {SBP}/{DBP} mmHg (normal: ~120/80)\n"
        f"MAP: {MAP} mmHg (normal: 70–100)"
    )
    cv_block = (
        f"GCS Eye Opening: {GCS_eye} "
        f"(1=No response, 2=To pain, 3=To voice, 4=Spontaneous)\n"
        f"Stress Score: {stress}/10 (higher = more distress)"
    )
    risk_block = (
        "\n".join(f"- {r}" for r in key_risks)
        if key_risks else "- No specific risk factors identified"
    )
    explanation_block = (
        "\n".join(f"- {e}" for e in clinical_explanations)
        if clinical_explanations else "- SHAP analysis not available"
    )

    if sofa_score >= ALERT_THRESHOLD:
        _urgency_prefix = (
            f"⚠️ CLINICAL ALERT — Predicted SOFA {sofa_score:.1f} "
            f"(alert threshold ≥ {ALERT_THRESHOLD:.0f})\n\n"
            "This patient is deteriorating. Prioritise immediate action.\n"
        )
        _section_instructions = (
            "1. IMMEDIATE ACTIONS — Organize by clinical domain "
            "(e.g., Hemodynamics, Respiratory, Renal, Infection, Monitoring). "
            "Mark the 2 most critical actions with ⚡. "
            "Write 2–4 bullet points (•) per domain. Be specific: drug names, doses, targets.\n"
            "2. CURRENT CONDITION — Write each problem as a bullet (•): "
            "• [Problem name]: [specific measured value] → [severity and trend]. "
            "Include 5–6 problems covering all abnormal findings.\n"
            "3. PROBABLE CAUSE — Write one clear explanatory paragraph (2–3 sentences) "
            "describing the primary mechanism. Then 3–4 bullet points (•) for contributing factors.\n"
            "4. RISK FORECAST — Write 3–4 bullet points (•). Each bullet: specific clinical "
            "trajectory with a realistic timeframe (e.g., 'within 2–4 h', 'within 12 h')."
        )
    else:
        _urgency_prefix = ""
        _section_instructions = (
            "1. CURRENT CONDITION — Write each problem as a bullet (•): "
            "• [Problem name]: [specific measured value] → [severity and trend]. "
            "Include 5–6 items covering all relevant findings.\n"
            "2. PROBABLE CAUSE — Write one clear explanatory paragraph (2–3 sentences) "
            "describing the primary mechanism. Then 3–4 bullet points (•) for contributing factors.\n"
            "3. RISK FORECAST — Write 3–4 bullet points (•). Each bullet: specific clinical "
            "trajectory with a realistic timeframe (e.g., 'within 12–24 h', 'within 48 h').\n"
            "4. IMMEDIATE ACTIONS — Organize by clinical domain. "
            "Mark the most critical action with ⚡. "
            "Write 2–3 bullet points (•) per domain. Be specific."
        )

    return f"""{_urgency_prefix}You are an ICU clinical decision support assistant.

Analyze the patient data below and write exactly 4 sections with these exact headings:

{_section_instructions}

Formatting rules (strict):
• Every item must be a bullet point starting with •  — no plain sentences without bullets inside sections.
• Do NOT use ** asterisks to create your own section subheadings — use plain text domain labels followed by a colon (e.g., "Hemodynamics:").
• Use **bold** only for drug names, critical values, and ⚡-marked priority actions.
• Be thorough and specific — include actual measured values, drug names, and clinical targets.
• Do NOT write a closing summary paragraph. End after the last bullet point.

---
SOFA Score: {sofa_score:.1f} / 24   (higher = worse organ failure)
Risk Level: {risk_text}

Clinical Notes:
{clinical_note}

Latest Vital Signs:
{vitals_block}

Neurological & Stress Indicators:
{cv_block}

Key Risk Factors (SHAP-identified):
{risk_block}

Feature-Level Explanations (SHAP-derived):
{explanation_block}

Vital Sign Trends (last 20 readings):
{trend_text}
---

Be specific and clinically precise. Base reasoning strictly on the data above.
Do NOT add a disclaimer, footnote, or asterisk note at the end — a clinical disclaimer is already displayed by the system."""


# ── Patients shown in each care setting ──────────────────────────────────────
# (title, one-line scenario). Patient 3 is ventilated, so it is ICU-only.
PATIENT_INFO = {
    1: {"icu":  ("Post-surgical recovery", "Appendectomy, day 2. Stable and improving."),
        "home": ("Post-discharge recovery", "Recovering at home after surgery. Caregiver visits daily.")},
    2: {"icu":  ("Pneumonia", "Community-acquired pneumonia on supplemental oxygen."),
        "home": ("Pneumonia recovery", "On home oxygen 4 L/min. Remote check-ins.")},
    3: {"icu":  ("Septic shock", "Intubated, on vasopressors, multi-organ failure."),
        "home": None},
    4: {"icu":  ("Gradual deterioration", "Ward patient with a urinary infection who slowly turns septic."),
        "home": ("Gradual deterioration", "Urinary infection at home that slowly turns septic. Watch the escalation.")},
}

VITAL_ORDER = ["HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"]

ESCALATION = {
    "icu": {
        "alert":  ("High-risk alert", ["Immediate clinical attention required",
                                       "Review the AI assessment and initiate appropriate protocols"]),
        "watch":  ("Close monitoring", ["Patient requires close monitoring",
                                        "Review the AI assessment and the trends"]),
        "stable": ("Stable", ["Continue routine monitoring"]),
    },
    "home": {
        "alert":  ("Escalate now", ["Alert the supervising clinician immediately",
                                    "Prepare transfer to hospital; call emergency services if the "
                                    "patient is unresponsive or oxygen stays low",
                                    "Caregiver stays with the patient and keeps the camera on"]),
        "watch":  ("Watch closely", ["Notify the caregiver and increase check-in frequency",
                                     "Request a tele-consult with the supervising clinician",
                                     f"Escalates automatically if predicted SOFA reaches {ALERT_THRESHOLD:g}"]),
        "stable": ("Stable", ["Continue routine home monitoring",
                              "Caregiver check-in as scheduled",
                              "No escalation needed"]),
    },
}


def read_cv_file(max_age_s=15):
    """Raw heartbeat written by cv_monitor.py (None if absent or stale)."""
    try:
        with open(os.path.join(MODEL_PATH, "cv_live.json")) as f:
            d = json.load(f)
        if time.time() - float(d["ts"]) > max_age_s:
            return None
        return {"stress": float(d["stress"]), "gcs": int(d["gcs"]), "face": bool(d.get("face", True))}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_cv_live(max_age_s=15):
    """Latest stress / GCS-eye scores — only while a face is actually in view."""
    d = read_cv_file(max_age_s)
    return d if d and d["face"] else None


class Monitor:
    """Holds the loaded model and the per-patient replay sessions."""

    def __init__(self):
        init_db()
        self.model, self.scaler, self.tfidf, self.feature_cols, self.background = self._load_artifacts()
        self.explainer = self._load_explainer()
        self.q_hat, self.q_calibrated = self._conformal()
        self.meta = self._load_meta()
        self.groq_key = os.getenv("GROQ_API_KEY", "")
        self.lock = threading.Lock()                 # torch / SHAP are not re-entrant
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.sessions = {}
        self.reports = {}

    # ── loading ──────────────────────────────────────────────────────────────
    def _load_artifacts(self):
        scaler = joblib.load(MODEL_PATH + "scaler.pkl")
        tfidf  = joblib.load(MODEL_PATH + "tfidf_vectorizer.pkl")
        fc_path = MODEL_PATH + "feature_columns.pkl"
        feature_cols = joblib.load(fc_path) if os.path.exists(fc_path) else list(scaler.feature_names_in_)
        model = ICUModel(len(feature_cols))
        model.load_state_dict(torch.load(MODEL_PATH + "federated_model.pth",
                                         map_location="cpu", weights_only=True))
        model.eval()
        bg_path = MODEL_PATH + "shap_background.npy"
        background = np.load(bg_path).astype(np.float32) if os.path.exists(bg_path) else None
        return model, scaler, tfidf, feature_cols, background

    def _load_explainer(self):
        if self.background is None:
            return None
        self.model.eval()
        return shap.DeepExplainer(self.model, torch.tensor(self.background, dtype=torch.float32))

    def _conformal(self):
        """90% split-conformal quantile from the FL calibration data (fallback 2.9 ≈ 1.6 × MAE)."""
        try:
            if not fl_training_exists():
                return 2.9, False
            errors = []
            for i in range(3):
                df = get_fl_client_data(i)
                if df.empty or "sofa_score" not in df.columns:
                    continue
                y_true = df["sofa_score"].values.astype(np.float32)
                X = df.drop(columns=["sofa_score"])
                for col in self.feature_cols:
                    if col not in X.columns:
                        X[col] = 0.0
                X = np.clip(X[list(self.feature_cols)].values.astype(np.float32), -10, 10)
                self.model.eval()
                with torch.no_grad():
                    preds = self.model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
                errors.extend(np.abs(preds - y_true).tolist())
            if len(errors) < 50:
                return 2.9, False
            return round(float(np.quantile(errors, 0.90)), 2), True
        except Exception:
            return 2.9, False

    def _load_meta(self):
        path = MODEL_PATH + "training_metadata.json"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {"num_rounds": 100, "epochs_per_round": 3, "hospitals": 3,
                "hospital_names": ["General ICU", "Mixed ICU", "Cardiac/Trauma ICU"],
                "aggregation": "FedYogi + FedProx", "differential_privacy": False}

    # ── public API ───────────────────────────────────────────────────────────
    def patients(self, mode):
        out = []
        for pid, info in PATIENT_INFO.items():
            entry = info.get(mode)
            if entry:
                out.append({"id": pid, "title": entry[0], "summary": entry[1],
                            "readings": get_sample_patient_count(pid)})
        return out

    def start(self, pid):
        """Begin a fresh monitoring session: clear the patient's window and log."""
        with self.lock:
            reset_patient(pid)
            self.sessions[pid] = {"row_idx": 0, "count": 0, "live_note": ""}
        return {"patient": pid, "total_rows": get_sample_patient_count(pid)}

    def set_live_note(self, pid, text):
        sess = self.sessions.setdefault(pid, {"row_idx": 0, "count": 0, "live_note": ""})
        sess["live_note"] = (text or "").strip()[:600]
        return {"live_note": sess["live_note"]}

    # ── live camera: the toggle launches cv_monitor.py itself ────────────────
    CAM_LOG = ROOT / "models" / "cv_monitor.log"

    def camera_start(self):
        proc = getattr(self, "cam_proc", None)
        if proc is not None and proc.poll() is None:
            return self.camera()
        try:
            (ROOT / "models" / "cv_live.json").unlink(missing_ok=True)   # never show a stale feed
        except OSError:
            pass
        log = open(self.CAM_LOG, "w", encoding="utf-8")
        self.cam_proc = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "cv_monitor.py")], cwd=str(ROOT),
            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"})
        self.cam_started = time.time()
        if not getattr(self, "_cam_atexit", False):
            atexit.register(self.camera_stop)
            self._cam_atexit = True
        return self.camera()

    def camera_stop(self):
        proc = getattr(self, "cam_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.cam_proc = None
        try:
            (ROOT / "models" / "cv_live.json").unlink(missing_ok=True)
        except OSError:
            pass
        return self.camera()

    def camera(self):
        proc = getattr(self, "cam_proc", None)
        running = proc is not None and proc.poll() is None
        raw = read_cv_file() if running else None
        d = raw if raw and raw["face"] else None
        error = None
        if proc is not None and not running:
            tail = ""
            try:
                tail = self.CAM_LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1]
            except (OSError, IndexError):
                pass
            error = tail or "the camera program closed"
            if "Cannot open camera" in error:
                error = "no camera could be opened (is another app using it?)"
        return {"running": running, "live": d is not None, "error": error,
                "no_face": bool(raw and not raw["face"]),
                "stress": d["stress"] if d else None, "gcs": d["gcs"] if d else None,
                "starting": running and raw is None,
                "waited": round(time.time() - getattr(self, "cam_started", time.time()), 1)}

    def report(self, rid):
        return self.reports.get(rid, {"status": "unknown"})

    def history(self, pid):
        df = get_prediction_history(pid, limit=50)
        return df.iloc[::-1].to_dict(orient="records")

    def model_info(self):
        m = self.meta
        return {"meta": m, "conformal_q": self.q_hat, "conformal_calibrated": self.q_calibrated,
                "n_features": len(self.feature_cols), "alert_threshold": ALERT_THRESHOLD,
                "llm_enabled": bool(self.groq_key) and not self.groq_key.startswith("your_")}

    def tick(self, pid, mode="icu", use_cam=False):
        with self.lock:
            return self._tick(pid, mode, use_cam)

    # ── one reading through the whole pipeline ───────────────────────────────
    def _tick(self, pid, mode, use_cam):
        sess = self.sessions.setdefault(pid, {"row_idx": 0, "count": 0, "live_note": ""})
        total_rows = get_sample_patient_count(pid)
        if total_rows == 0:
            raise RuntimeError("Database not initialised. Run `python migrate_to_db.py` first.")
        row_idx = sess["row_idx"]
        cur_idx = row_idx % total_rows
        cur = get_sample_patient_row(pid, cur_idx)

        HR, RR, SpO2, Temp = float(cur["HR"]), float(cur["RR"]), float(cur["SpO2"]), float(cur["Temp"])
        SBP, DBP, MAP = float(cur["SBP"]), float(cur["DBP"]), float(cur["MAP"])
        GCS_eye = int(cur["GCS_eye_opening"])
        stress  = int(cur["stress_score"])
        replay_note = str(cur["clinical_note"])

        # Seed the sliding window on a patient's first reading. The model expects a 20-reading
        # window, so the pre-history is the patient's FIRST reading repeated (a flat baseline,
        # 10 minutes apart). Seeding with the stream's own next rows would leak the patient's
        # future into the trend features of a deteriorating patient.
        if len(get_patient_vitals(pid, limit=1)) == 0:
            base = get_sample_patient_row(pid, 0)
            t0 = datetime.now()
            for si in range(20):
                append_patient_vitals(pid, {
                    "time": (t0 - timedelta(minutes=10 * (20 - si))).strftime("%Y-%m-%d %H:%M:%S"),
                    "HR": float(base["HR"]), "RR": float(base["RR"]), "SpO2": float(base["SpO2"]),
                    "Temp": float(base["Temp"]), "SBP": float(base["SBP"]),
                    "DBP": float(base["DBP"]), "MAP": float(base["MAP"]),
                })

        # Camera modality: replayed from the stream, or the live webcam when switched on
        cv_source = "replayed"
        if use_cam:
            cv = read_cv_live()
            if cv is not None:
                stress  = int(round(min(max(cv["stress"], 0.0), 10.0)))
                GCS_eye = int(min(max(cv["gcs"], 1), 4))
                cv_source = "live"
            else:
                cv_source = "no_feed"

        # Text modality: replayed note, plus any note typed live by the caregiver
        live_note = sess.get("live_note", "")
        clinical_note = replay_note + (f" [Live caregiver note] {live_note}" if live_note else "")

        # -- 1. sliding window --
        append_patient_vitals(pid, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "HR": HR, "RR": RR, "SpO2": SpO2, "Temp": Temp, "SBP": SBP, "DBP": DBP, "MAP": MAP,
        })
        vitals_df = get_patient_vitals(pid, limit=20)

        # -- 2. trend features --
        trend_features = {
            "HR_mean":   vitals_df["HR"].mean(),
            "HR_std":    vitals_df["HR"].std() if len(vitals_df) > 1 else 0.0,
            "RR_mean":   vitals_df["RR"].mean(),
            "SpO2_mean": vitals_df["SpO2"].mean(),
            "SpO2_min":  vitals_df["SpO2"].min(),
            "Temp_mean": vitals_df["Temp"].mean(),
            "SBP_mean":  vitals_df["SBP"].mean(),
            "DBP_mean":  vitals_df["DBP"].mean(),
            "MAP_mean":  vitals_df["MAP"].mean(),
        }
        # -- 3. latest features --
        latest_features = {
            "latest_HR": HR, "latest_RR": RR, "latest_SpO2": SpO2, "latest_Temp": Temp,
            "latest_SBP": SBP, "latest_DBP": DBP, "latest_MAP": MAP,
            "GCS_eye_opening": float(GCS_eye), "stress_score": float(stress),
        }
        # -- 4. TF-IDF of the note --
        tfidf_vec = self.tfidf.transform([clinical_note])
        tfidf_df  = pd.DataFrame(tfidf_vec.toarray(), columns=self.tfidf.get_feature_names_out())
        # -- 5. combine + align --
        input_df = pd.DataFrame([{**trend_features, **latest_features}])
        final_df = pd.concat([input_df, tfidf_df], axis=1).fillna(0)
        expected_cols = self.feature_cols
        for col in expected_cols:
            if col not in final_df.columns:
                final_df[col] = 0
        final_df = final_df[expected_cols].astype(float)
        # -- 6. scale (clip to [-10, 10], matches training) --
        final_scaled = np.clip(self.scaler.transform(final_df), -10, 10)
        # -- 7. predict --
        X_tensor = torch.tensor(final_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            raw_pred = self.model(X_tensor).numpy().flatten()[0]
        sofa_score = float(np.clip(raw_pred, 0, 24))
        risk_text, _, _ = risk_label(sofa_score)

        # -- validation checks --
        sofa_floor  = compute_sofa_floor(MAP, SpO2, GCS_eye)
        floor_ok    = sofa_score >= (sofa_floor - 1)
        vital_flags = vital_consistency_flags(HR, SBP, SpO2, MAP, sofa_score)
        prev_sofa, sofa_jump, traj_flagged, traj_msg = check_trajectory(
            pid, sofa_score, HR, RR, SpO2, SBP, MAP)
        conf_lo = round(max(0.0,  sofa_score - self.q_hat), 1)
        conf_hi = round(min(24.0, sofa_score + self.q_hat), 1)

        # -- save prediction history --
        append_prediction(pid, {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Reading":   f"{cur_idx + 1}/{total_rows}",
            "SOFA":      round(sofa_score, 1), "Risk": risk_text,
            "HR": HR, "RR": RR, "SpO₂": SpO2, "Temp (°C)": Temp, "SBP": SBP, "MAP": MAP,
            "GCS Eye": GCS_eye, "Stress": stress,
        })

        # -- SHAP --
        shap_vals, top_shap = None, None
        clinical_explanations, key_risks = [], []
        if self.explainer is not None:
            try:
                self.model.eval()
                shap_raw = self.explainer.shap_values(X_tensor)
                if isinstance(shap_raw, list):
                    shap_vals = shap_raw[0][0]
                else:
                    shap_vals = shap_raw[0] if shap_raw.ndim > 1 else shap_raw
                shap_vals = np.array(shap_vals).flatten()
            except Exception:
                shap_vals = None

        drivers, lowered = [], 0
        if shap_vals is not None:
            original_values = final_df.iloc[0].values
            shap_df = pd.DataFrame({"feature": expected_cols, "original_value": original_values,
                                    "impact": shap_vals})
            shap_df["abs_impact"] = shap_df["impact"].abs()
            shap_df = shap_df.sort_values("abs_impact", ascending=False)

            def is_clinical(f):
                return any(k.lower() in f.lower() for k in CLINICAL_KEYWORDS)

            filtered = shap_df[shap_df["feature"].apply(is_clinical)]
            top_shap = (filtered.head(7) if len(filtered) >= 3 else shap_df.head(7)).copy()

            pos = top_shap[top_shap["impact"] > 0]
            lowered = int((top_shap["impact"] <= 0).sum())
            for _, r in pos.iterrows():
                text = interpret_shap(r["feature"], r["original_value"], r["impact"])
                clinical_explanations.append(text)
                if r["feature"] in VITAL_UNITS:
                    label, unit = VITAL_UNITS[r["feature"]]
                    fmt = ".0f" if unit in ("/4", "/10") else ".1f"
                    drivers.append({"kind": "vital", "label": label,
                                    "value": f"{r['original_value']:{fmt}} {unit}",
                                    "impact": round(float(r["impact"]), 3), "text": text})
                else:
                    drivers.append({"kind": "note", "label": str(r["feature"]),
                                    "value": "mentioned in the note",
                                    "impact": round(float(r["impact"]), 3), "text": text})
            seen = set()
            for _, r in pos.iterrows():
                for v in risk_factors_for(r["feature"]):
                    if v not in seen:
                        key_risks.append(v)
                        seen.add(v)

        # -- LLM report (runs in the background so the screen never waits for it) --
        prompt = build_prompt(sofa_score, risk_text, clinical_note, HR, RR, SpO2, Temp, SBP, DBP,
                              MAP, GCS_eye, stress, key_risks, clinical_explanations, vitals_df)
        sess["count"] += 1
        rid = f"{pid}-{sess['count']}"
        self.reports[rid] = {"status": "pending"}
        if len(self.reports) > 80:                     # keep memory bounded
            for old in list(self.reports)[:20]:
                self.reports.pop(old, None)
        ctx = {"prompt": prompt, "sofa": sofa_score, "top_shap": top_shap,
               "vitals": {"SpO2": SpO2, "MAP": MAP, "HR": HR, "RR": RR, "SBP": SBP, "DBP": DBP,
                          "Temp": Temp, "stress": stress, "GCS_eye": GCS_eye}}
        self.pool.submit(self._report_job, rid, ctx)

        # -- tiers + escalation --
        tier = "alert" if sofa_score >= ALERT_THRESHOLD else "watch" if sofa_score >= 5 else "stable"
        esc_title, esc_actions = ESCALATION["home" if mode == "home" else "icu"][tier]

        # -- advance the stream for the next reading --
        sess["row_idx"] = (row_idx + 1) % total_rows

        # -- build the response --
        vit = {"HR": HR, "RR": RR, "SpO2": SpO2, "Temp": Temp, "SBP": SBP, "DBP": DBP, "MAP": MAP}
        hist = get_prediction_history(pid, limit=50)
        win20 = hist.tail(20)
        tiles = []
        for k in VITAL_ORDER:
            lo, hi = NORMAL_RANGES[k]
            v = vit[k]
            tiles.append({"key": k, "value": v, "lo": lo, "hi": hi,
                          "status": "low" if v < lo else "high" if v > hi else "ok",
                          "series": [float(x) for x in vitals_df[k].tolist()]})
        validation = [
            {"key": "interval", "title": "Prediction interval", "ok": True,
             "detail": f"90% interval {conf_lo} – {conf_hi} (±{self.q_hat} SOFA points)"
                       + ("" if self.q_calibrated else " · default width, FL calibration data not loaded")},
            {"key": "floor", "title": "Physiological floor", "ok": bool(floor_ok),
             "detail": f"Vitals imply SOFA ≥ {sofa_floor}; predicted {sofa_score:.1f}"},
            {"key": "consistency", "title": "Vital-sign consistency", "ok": not vital_flags,
             "detail": "; ".join(vital_flags) if vital_flags else "No conflicting vital-sign patterns"},
            {"key": "trajectory", "title": "Trajectory", "ok": not traj_flagged,
             "detail": (traj_msg if traj_flagged else
                        f"Previous {prev_sofa:.1f} → now {sofa_score:.1f}" if prev_sofa is not None
                        else (traj_msg or "First reading"))},
        ]
        return {
            "reading_id": rid, "patient": pid, "mode": mode,
            "reading_no": cur_idx + 1, "total_rows": total_rows,
            "wrapped": cur_idx + 1 == total_rows,
            "time": datetime.now().strftime("%H:%M:%S"),
            "sofa": round(sofa_score, 2), "risk": risk_text, "tier": tier,
            "severity_pct": int(sofa_score / 24 * 100),
            "alert_note": ("Clinical High Risk threshold is SOFA ≥ 10"
                           if ALERT_THRESHOLD <= sofa_score < 10 else ""),
            "window_n": int(len(vitals_df)),
            "interval": [conf_lo, conf_hi],
            "escalation": {"title": esc_title, "actions": esc_actions},
            "escalations": {m_: {"title": ESCALATION[m_][tier][0], "actions": ESCALATION[m_][tier][1]}
                            for m_ in ("icu", "home")},
            "vitals": tiles,
            "gcs": {"value": GCS_eye, "series": [int(x) for x in win20["GCS Eye"].tolist()]},
            "stress": {"value": stress, "series": [float(x) for x in win20["Stress"].tolist()]},
            "cv_source": cv_source,
            "note": clinical_note, "replay_note": replay_note, "live_note": live_note,
            "validation": validation,
            "validation_passed": sum(1 for v in validation if v["ok"]),
            "explain": {"drivers": drivers, "lowered": lowered,
                        "interpretations": clinical_explanations, "key_risks": key_risks,
                        "available": shap_vals is not None},
            "sofa_series": [float(x) for x in hist["SOFA"].tolist()],
            "report": {"id": rid, "status": "pending"},
        }

    # ── background LLM report + reliability ──────────────────────────────────
    def _report_job(self, rid, ctx):
        rep = self.reports.get(rid)
        if rep is None:
            return
        key = self.groq_key
        if not key or key.startswith("your_"):
            rep.update(status="unavailable", reason="no GROQ_API_KEY is set in the .env file")
            return
        try:
            responses = get_multiple_llm_responses(key, ctx["prompt"], n=1)
        except Exception as e:
            msg = str(e)
            if "401" in msg or "invalid_api_key" in msg.lower():
                reason = "the Groq API key in .env was rejected (invalid key)"
            elif "429" in msg:
                reason = "the Groq rate limit was reached — it will retry on the next reading"
            else:
                reason = f"the Groq service could not be reached ({type(e).__name__})"
            rep.update(status="unavailable", reason=reason)
            return
        try:
            main = responses[0]
            judge = call_judge_llm(key, main, {k: ctx["vitals"][k] for k in
                                               ("SpO2", "MAP", "HR", "RR", "SBP", "stress", "GCS_eye")},
                                   ctx["sofa"])
            score, breakdown, viols = compute_reliability_new(
                main, ctx["vitals"], ctx["sofa"], ctx["top_shap"], judge)
            rep.update(status="ok", text=main, sections=parse_llm_sections(main),
                       reliability=round(float(score), 3), breakdown=breakdown,
                       violations=viols, judge_ok=judge is not None)
        except Exception as e:
            rep.update(status="unavailable", reason=f"the report could not be scored ({type(e).__name__})")

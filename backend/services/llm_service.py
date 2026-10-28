import os
import re
import sys
import json
import time
import threading
from decimal import Decimal, InvalidOperation

import numpy as np

# ─── Simple in-process TTL cache ──────────────────────────────────────────────
_REPORT_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL  = 90  # seconds — skip re-calling LLM if same patient+mode+sofa

def _cache_key(predict_result: dict, care_mode: str) -> str:
    pid    = predict_result.get("patient_id", 0)
    sofa   = round(predict_result.get("sofa_score", 0), 1)
    vitals = predict_result.get("vitals", {})
    hr     = round(vitals.get("HR",   0))
    spo2   = round(vitals.get("SpO2", 0))
    sbp    = round(vitals.get("SBP",  0))
    rr     = round(vitals.get("RR",   0))
    return f"{care_mode}|{pid}|{sofa}|{hr}|{spo2}|{sbp}|{rr}"

def _cache_get(key: str):
    with _CACHE_LOCK:
        entry = _REPORT_CACHE.get(key)
        if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
            return entry["data"]
        if entry:
            del _REPORT_CACHE[key]
    return None

def _cache_set(key: str, data: dict):
    with _CACHE_LOCK:
        _REPORT_CACHE[key] = {"ts": time.time(), "data": data}

_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT         = os.path.dirname(os.path.dirname(_SERVICES_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from groq import Groq

# ──────────────────────────────────────────────
# SYSTEM MESSAGES
# ──────────────────────────────────────────────

_HOSPITAL_SYSTEM_MSG = (
    "You are a senior ICU attending physician generating a structured clinical decision "
    "support note for a critically ill patient. You follow SCCM/Surviving Sepsis Campaign "
    "guidelines.\n\n"
    "Non-negotiable rules:\n"
    "- Every vital sign cited includes the patient's measured value and the normal range.\n"
    "- Every drug recommendation includes: name, dose, route, and monitoring target.\n"
    "- Every risk forecast item includes a specific timeframe (e.g., 'within 6–12 h').\n"
    "- Write every section completely — never truncate, never end mid-sentence.\n"
    "- Base all reasoning strictly on the provided patient data — never invent values."
)

_HOMECARE_SYSTEM_MSG = (
    "You are a clinical decision support assistant generating a home care guidance report "
    "for a patient being monitored at home by a home care nurse and non-medical family members.\n\n"
    "Your audience: ONE home care nurse (clinical training, no prescribing authority, limited "
    "equipment) AND 2–3 family members with no medical background who may be anxious.\n\n"
    "Non-negotiable rules:\n"
    "- Write in clear, plain English. When a medical term is unavoidable, immediately explain "
    "it in simple words in brackets — e.g., 'tachycardia [fast heart rate]'.\n"
    "- Recommend ONLY actions doable at home: positioning, oral medications already at home, "
    "first-aid measures, comfort techniques, and safe home remedies.\n"
    "- Include safe home remedies where clinically appropriate: warm fluids, turmeric, ginger, "
    "steam inhalation, honey, cool compress, leg elevation, deep breathing exercises, etc.\n"
    "- Never mention IV lines, hospital equipment, or procedures requiring a physician.\n"
    "- Always include specific, observable warning signs that mean the patient must go to hospital.\n"
    "- Clearly state when home care alone is sufficient — not every case needs hospital.\n"
    "- Write every section completely — never truncate, never end mid-sentence.\n"
    "- Be compassionate and calm in tone — family members may be frightened.\n"
    "- Base all reasoning strictly on the provided patient data — never invent values."
)

_FALLBACK_MSG = (
    "⚠️ The AI model ran out of tokens during its internal reasoning process "
    "and did not produce a clinical assessment for this reading. "
    "The next automatic reading (in ~30 s) will generate a fresh response."
)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    elif "<think>" in text:
        text = text[: text.find("<think>")]
    return text.strip()


def _parse_llm_sections(text: str) -> dict:
    _NAMES = [
        "IMMEDIATE ACTIONS", "CURRENT CONDITION",
        "PROBABLE CAUSE",    "RISK FORECAST",
    ]
    positions = []
    for sec in _NAMES:
        m = re.search(
            rf'(?:^|(?<=\n))[ \t]*#*[ \t]*(?:\d+\.\s*)?\*{{0,2}}\s*{re.escape(sec)}\s*\*{{0,2}}\s*[—\-:–]?\s*',
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

# ──────────────────────────────────────────────
# PROMPT BUILDERS
# ──────────────────────────────────────────────

def _build_prompt(predict_result: dict, care_mode: str) -> tuple[str, str]:
    """Returns (final_prompt, system_msg)."""
    sofa        = predict_result["sofa_score"]
    risk_text   = predict_result["risk_level"]
    conf_q      = predict_result["conformal_q"]
    clinical_note  = predict_result["clinical_note"]
    vitals_block   = predict_result["_vitals_block"]
    cv_block       = predict_result["_cv_block"]
    risk_block     = predict_result["_risk_block"]
    explanation_block = predict_result["_explanation_block"]
    trend_text     = predict_result["trend_text"]
    alert_threshold = predict_result["alert_threshold"]

    if care_mode == "hospital":
        if sofa >= alert_threshold:
            urgency_prefix = (
                f"⚠️ CLINICAL ALERT — Predicted SOFA {sofa:.1f} "
                f"(alert threshold ≥ {alert_threshold:.0f}). "
                "This patient is actively deteriorating. Lead with immediate actions.\n\n"
            )
            section_instructions = """\
1. IMMEDIATE ACTIONS — Organize under clinical domain subheadings (e.g., Respiratory:, Hemodynamics:, Infection:, Renal:, Monitoring:).
   Mark the 2 most time-critical actions with ⚡ at the START of the bullet.
   Write 2–4 bullets (•) per domain. Each bullet must include: drug name + dose + route + specific target OR procedure + indication + monitoring parameter.
   GOOD example: • ⚡ **Norepinephrine** 0.05–0.3 mcg/kg/min IV — titrate to **MAP ≥65 mmHg**; escalate every 15 min if no response.
   BAD example: • Start vasopressors if blood pressure low.

2. CURRENT CONDITION — 6–8 bullets (•), one per organ system or abnormal finding.
   Format each bullet: • **[System/Problem]:** [measured value] ([normal range]) → [clinical interpretation] — [trajectory/trend].
   GOOD example: • **Respiratory:** SpO₂ 91% on 4 L/min O₂ (normal ≥95%) → mild-to-moderate hypoxemia — oxygen demand increasing, high-flow therapy likely needed.
   BAD example: • Low oxygen levels.

3. PROBABLE CAUSE — One explanatory paragraph (3–4 sentences) naming the primary mechanism and its physiological chain. Then 4–5 bullets (•) for specific contributing factors, each citing a measured value.
   GOOD contributing factor: • Persistent tachycardia (HR 108 bpm) and fever (Temp 38.4 °C) indicate ongoing systemic inflammatory response — consistent with early sepsis physiology.

4. RISK FORECAST — 4–5 bullets (•). Each bullet must state: specific deterioration event + specific threshold + specific timeframe.
   GOOD example: • Progression to septic shock (MAP <65 mmHg) within **12–24 h** if IV fluid resuscitation and vasopressor support are not initiated.
   BAD example: • Patient may get worse."""
        else:
            urgency_prefix = ""
            section_instructions = """\
1. CURRENT CONDITION — 6–8 bullets (•), one per organ system or abnormal finding.
   Format each bullet: • **[System/Problem]:** [measured value] ([normal range]) → [clinical interpretation] — [trajectory/trend].
   GOOD example: • **Cardiovascular:** HR 95 bpm (normal 60–100) at upper limit — tachycardia likely compensatory; monitor for escalation.

2. PROBABLE CAUSE — One explanatory paragraph (3–4 sentences) naming the primary mechanism and its physiological chain. Then 4–5 bullets (•) for specific contributing factors, each citing a measured value.

3. RISK FORECAST — 4–5 bullets (•). Each bullet must state: specific deterioration event + specific threshold + specific timeframe.
   GOOD example: • Renal function decline (creatinine >2.0 mg/dL) within **48 h** if hemodynamic instability and reduced perfusion pressure persist.

4. IMMEDIATE ACTIONS — Organize under clinical domain subheadings (e.g., Respiratory:, Hemodynamics:, Infection:, Monitoring:).
   Mark the most time-critical action with ⚡. Write 2–3 bullets (•) per domain.
   Each bullet must include: drug name + dose + route + target OR procedure + indication + monitoring parameter."""

        final_prompt = f"""{urgency_prefix}Generate a structured ICU clinical decision support report with exactly 4 sections using these exact headings (write each heading exactly as shown, on its own line):

{section_instructions}

━━━ STRICT FORMATTING RULES ━━━
• Every item inside a section must be a bullet starting with the • character (Unicode bullet, not * or -)
• Domain subheadings (Respiratory:, Hemodynamics:, etc.) go on their own line without a bullet — they act as sub-section labels
• Write each of the 4 section headings EXACTLY as shown above — no # symbols, no markdown, no numbering prefix
• Use **bold** ONLY for: drug names, critical numeric values, ⚡-marked actions, and organ system labels in CURRENT CONDITION bullets
• Do NOT use # markdown headings or ** to create your own section headings — use only the 4 headings listed above
• Do NOT write a closing summary, footnote, or disclaimer — end after the last bullet
• Write every section completely — do not stop mid-sentence or truncate

━━━ PATIENT DATA ━━━
SOFA Score: {sofa:.1f} / 24   (0–6 = low risk, 7–9 = moderate, ≥10 = high, ≥15 = critical)
Risk Level: {risk_text}
Predicted SOFA uncertainty: ± {conf_q:.1f} points (90% conformal interval)

Clinical Notes (from patient chart):
{clinical_note}

Latest Vital Signs:
{vitals_block}

Neurological & Stress Indicators:
{cv_block}

Key Risk Factors (SHAP-identified — highest model contributors):
{risk_block}

Feature-Level Explanations (SHAP-derived):
{explanation_block}

Vital Sign Trends (last 20 readings — use to identify trajectories):
{trend_text}
━━━━━━━━━━━━━━━━━━━━

Base all clinical reasoning strictly on the data above. Do not invent values or assume findings not present in the data."""
        return final_prompt, _HOSPITAL_SYSTEM_MSG

    else:  # homecare
        if sofa >= alert_threshold:
            urgency_prefix = (
                f"⚠️ PATIENT ALERT — SOFA score {sofa:.1f} indicates the patient is "
                f"deteriorating and needs immediate attention at home.\n\n"
            )
            section_instructions = """\
1. IMMEDIATE ACTIONS — Organize under these plain sub-headings on separate lines (no bullet before sub-headings):
   "For the Nurse:", "For the Family:", "Home Remedies:", "⚠️ Go to Hospital NOW if:"
   Mark the single most urgent action with ⚡.
   Every bullet is a simple, specific, step-by-step action doable at home without hospital equipment.
   Include relevant safe home remedies: positioning, warm fluids with honey/ginger/turmeric, steam inhalation, cool compress, leg elevation, gentle breathing exercises, etc.
   The "⚠️ Go to Hospital NOW if:" sub-section must list 4–5 specific, observable danger signs a family member can recognise.
   GOOD nurse example: • ⚡ Sit the patient upright at 45 degrees — place 2–3 firm pillows behind their back and shoulders.
   GOOD home remedy: • Prepare warm water with half a teaspoon of turmeric and a teaspoon of honey — give small sips every 15 minutes.
   GOOD hospital sign: • Lips, fingernails, or fingertips turning blue or grey — go to hospital immediately.

2. CURRENT CONDITION — 5–7 bullets in plain English, one per finding.
   Format: • [What it means in simple words]: [value] (healthy range: [normal]) — [one plain sentence explaining what this means right now].
   GOOD example: • Breathing rate: 24 breaths per minute (healthy: 12–20) — breathing much faster than normal, which means the body is working hard to get enough oxygen.
   BAD example: • Tachypnea: RR 24 breaths/min → moderate respiratory distress.

3. PROBABLE CAUSE — One paragraph (3–4 sentences) in plain, compassionate English explaining what is likely happening. Use simple analogies. Then 3–4 bullets listing specific signs from the data, in plain language.

4. RISK FORECAST — 4–5 bullets covering BOTH possible outcomes: worsening (needs hospital) AND improvement (stays at home).
   Each bullet: specific observable change + what it means + realistic timeframe.
   GOOD worsening: • If breathing rate stays above 25 per minute and lips look pale or bluish in the next 2–3 hours — take them to hospital immediately.
   GOOD improvement: • If fever drops below 38°C and breathing slows to under 20 per minute within 4–6 hours of rest — safely monitor at home."""
        else:
            urgency_prefix = ""
            section_instructions = """\
1. CURRENT CONDITION — 5–7 bullets in plain English, one per finding.
   Format: • [What it means in simple words]: [value] (healthy range: [normal]) — [one plain sentence].
   GOOD example: • Heart rate: 95 beats per minute (healthy: 60–100) — at the upper end of normal, worth watching but not alarming right now.
   BAD example: • HR 95 bpm (normal 60–100) — compensatory tachycardia.

2. PROBABLE CAUSE — One paragraph (3–4 sentences) in plain, compassionate English. Use simple analogies. Then 3–4 bullets listing specific patient signs supporting this, in plain language.

3. RISK FORECAST — 4–5 bullets covering BOTH possible outcomes: worsening AND improvement.
   Each bullet: specific observable change + what it means + realistic timeframe.
   GOOD improvement: • If the patient rests and fever reduces to below 38°C within 4–6 hours — they are likely stable at home; continue monitoring every hour.
   GOOD worsening: • If the patient becomes confused or breathing becomes very fast (above 25 per minute) — call emergency services immediately.

4. IMMEDIATE ACTIONS — Organize under plain sub-headings: "For the Nurse:", "For the Family:", "Home Remedies:".
   Mark the most important action with ⚡.
   Include relevant home remedies (warm fluids, turmeric, ginger, steam, cool compress, elevation, etc.).
   GOOD home remedy: • Give warm ginger tea with honey — ginger helps reduce nausea and has mild anti-inflammatory effects."""

        final_prompt = f"""{urgency_prefix}Generate a structured home care guidance report for a patient being monitored at home. Write exactly 4 sections using these exact headings (write each heading exactly as shown, on its own line):

{section_instructions}

━━━ STRICT FORMATTING RULES ━━━
• Every item inside a section must be a bullet starting with the • character (Unicode bullet, not * or -)
• Sub-headings like "For the Nurse:", "For the Family:", "Home Remedies:", "⚠️ Go to Hospital NOW if:" go on their own line WITHOUT a bullet
• Write each of the 4 section headings EXACTLY as shown above — no # symbols, no markdown, no numbering prefix
• Use **bold** ONLY for: the most critical actions, key numbers the family must watch, and ⚡-marked urgent actions
• Do NOT use # markdown headings or ** to create your own section headings — use only the 4 headings listed above
• Write in plain, calm English throughout — avoid clinical abbreviations and unexplained jargon
• Do NOT write a closing summary, footnote, or disclaimer — end after the last bullet
• Write every section completely — do not stop mid-sentence or truncate

━━━ PATIENT DATA ━━━
SOFA Score: {sofa:.1f} / 24   (higher = worse; above 6 = concerning, above 10 = serious)
Current Risk Level: {risk_text}
Monitoring Uncertainty: predictions accurate to within ± {conf_q:.1f} SOFA points

Clinical Notes (from patient records):
{clinical_note}

Current Vital Signs:
{vitals_block}

Neurological & Stress Indicators:
{cv_block}

Key Risk Factors identified by the monitoring system:
{risk_block}

Vital Sign Trends (last 20 readings — use to identify whether things are getting better or worse):
{trend_text}
━━━━━━━━━━━━━━━━━━━━

Base all guidance strictly on the data above. Do not invent values or assume findings not in the data. Remember: your audience is a home care nurse and non-medical family members."""
        return final_prompt, _HOMECARE_SYSTEM_MSG

# ──────────────────────────────────────────────
# VALIDATION HELPERS  (7 deterministic methods)
# ──────────────────────────────────────────────

def factual_grounding_score(response, vitals, sofa_val):
    r = response.lower()
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
    r     = response.lower()
    fired = [rule for rule in _LLM_HARD_RULES if rule["condition"](vitals, sofa_val)]
    if not fired:
        return 1.0, []
    viols = [rule["rule"] for rule in fired
             if not any(t in r for t in rule["required"])]
    return 1.0 - (len(viols) / len(fired)), viols


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


def shap_coherence_score(response, shap_top5: list) -> float:
    if not shap_top5:
        return 0.5
    r = response.lower()
    scores = []
    for item in shap_top5:
        feat   = item["feature"]
        impact = item["impact"]
        if abs(impact) < 0.1:
            continue
        terms = _SHAP_TERMS.get(feat, [feat.lower().replace("_", " ")])
        scores.append(float(any(t in r for t in terms)))
    return sum(scores) / len(scores) if scores else 0.5


def response_structure_score(response):
    REQUIRED = ["current condition", "probable cause", "risk forecast", "immediate action"]
    r      = response.lower()
    found  = sum(1 for sec in REQUIRED if sec in r)
    length = min(1.0, len(response.split()) / 200)
    return (found / len(REQUIRED)) * 0.70 + length * 0.30


def severity_calibration_score(response, sofa_val):
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
    r     = response.lower()
    flags = [rule["flag"] for rule in _CONTRA_RULES
             if rule["condition"](vitals) and any(t in r for t in rule["forbidden"])]
    return (0.0 if flags else 1.0), flags


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

_CLINICAL_REFS = {
    60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 92.0, 95.0, 100.0,
    120.0, 130.0, 140.0, 150.0, 160.0,
    20.0, 25.0, 30.0, 35.0, 40.0,
    36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 40.0,
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0,
    0.5, 1.5, 2.0, 2.5, 50.0, 24.0,
}


def numeric_hallucination_score(response, vitals, sofa_val):
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

    def _is_allowed(val_str):
        try:
            v = float(Decimal(val_str).normalize())
        except InvalidOperation:
            return True
        if v in _CLINICAL_REFS:
            return True
        for a in actual:
            if abs(v - a) <= max(abs(a) * 0.10, 3.0):
                return True
        return False

    measured = []
    for pattern in (_NUM_LABELED, _NUM_UNIT):
        for m in pattern.finditer(response):
            measured.append(m.group(1))

    if not measured:
        return 1.0, []

    seen, unsupported = set(), []
    for val in measured:
        if val not in seen:
            seen.add(val)
            if not _is_allowed(val):
                unsupported.append(val)

    return max(0.0, 1.0 - len(unsupported) * 0.25), unsupported


def call_judge_llm(api_key: str, main_response: str, vitals: dict, sofa_val: float, timeout: float = 5.0):
    """Run a quick LLM judge with a hard wall-clock timeout to avoid blocking the response."""
    result_box = [None]

    def _run():
        try:
            client = Groq(api_key=api_key)
            v_str = (
                f"SpO2={vitals.get('SpO2','?')}%, MAP={vitals.get('MAP','?')} mmHg, "
                f"HR={vitals.get('HR','?')} bpm, RR={vitals.get('RR','?')} br/min, "
                f"SOFA={sofa_val:.1f}/24"
            )
            prompt = (
                f"Patient vitals: {v_str}\n\n"
                f"AI response (first 900 chars):\n{main_response[:900]}\n\n"
                "Output ONLY valid JSON, no markdown:\n"
                '{"factual_accuracy":1-5,"clinical_appropriateness":1-5,'
                '"urgency_calibration":1-5,"completeness":1-5,'
                '"unsupported_claims":[],"dangerous_recommendation_detected":false,'
                '"dangerous_details":""}'
            )
            resp = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            text = _strip_thinking(resp.choices[0].message.content.strip())
            s, e = text.find("{"), text.rfind("}") + 1
            if s >= 0 and e > s:
                result_box[0] = json.loads(text[s:e])
        except Exception as _e:
            import sys as _sys
            print(f"[ClinIQ] Judge LLM call failed: {_e}", file=_sys.stderr)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result_box[0]


def compute_reliability(response, vitals, sofa_val, shap_top5, judge_result):
    fg               = factual_grounding_score(response, vitals, sofa_val)
    chr_s, chr_viols = llm_clinical_rules_score(response, vitals, sofa_val)
    shap_s           = shap_coherence_score(response, shap_top5)
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
        geval = (geval_raw - 1.0) / 4.0
        unsup = judge_result.get("unsupported_claims", [])
        ragas = max(0.0, 1.0 - len(unsup) * 0.30)
        if judge_result.get("dangerous_recommendation_detected", False):
            contra_s = 0.0
            detail = judge_result.get("dangerous_details", "")
            if detail:
                contra_flags = list(contra_flags) + [f"AI Judge: {detail}"]
    else:
        geval, ragas = 0.5, 0.5

    combined = float(np.clip(
        0.13 * fg    + 0.13 * chr_s + 0.09 * shap_s +
        0.09 * struct + 0.08 * sev  + 0.05 * contra_s +
        0.08 * num_s +
        0.13 * geval + 0.22 * ragas,
        0.0, 1.0,
    ))

    all_viols = list(chr_viols) + list(contra_flags)
    for n in num_flags[:3]:
        all_viols.append(f"Hallucinated number in response: {n}")
    for c in unsup[:2]:
        all_viols.append(f"Unsupported claim: {c}")

    return combined, all_viols

# ──────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────

def _reliability_label(score: float) -> tuple[str, str]:
    if score >= 0.75:
        return "High", "#28a745"
    elif score >= 0.55:
        return "Moderate", "#e6a817"
    return "Low", "#dc3545"


def generate_report(predict_result: dict, care_mode: str, api_key: str) -> dict:
    """
    Builds prompt from predict_result, calls Groq, validates, returns report dict.
    care_mode: "hospital" or "homecare"
    """
    # ── TTL cache check ───────────────────────────────────────────────────────
    cache_key = _cache_key(predict_result, care_mode)
    cached = _cache_get(cache_key)
    if cached:
        return cached

    final_prompt, system_msg = _build_prompt(predict_result, care_mode)

    vitals_for_validation = {
        **predict_result["vitals"],
        "GCS_eye": predict_result["vitals"]["GCS_eye"],
        "stress":  predict_result["vitals"]["stress"],
    }
    sofa = predict_result["sofa_score"]

    # ── Main LLM call ─────────────────────────────────────────────────────────
    main_response = None
    llm_ok        = False
    try:
        client = Groq(api_key=api_key)
        resp   = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": final_prompt},
            ],
            temperature=0.15,
            max_tokens=3500,
        )
        text = _strip_thinking(resp.choices[0].message.content)
        main_response = text if text else _FALLBACK_MSG
        llm_ok = bool(text)
    except Exception as e:
        main_response = f"⚠️ LLM unavailable: {e}"

    if not llm_ok:
        return {
            "sections": None,
            "raw_response": main_response,
            "reliability_score": None,
            "reliability_label": None,
            "reliability_color": None,
            "caution": False,
            "care_mode": care_mode,
            "error": main_response,
        }

    # ── Judge LLM call (capped at 12s so it never hangs the response) ──────────
    judge_result = None
    judge_used   = False
    if api_key:
        judge_result = call_judge_llm(api_key, main_response, vitals_for_validation, sofa, timeout=12.0)
        judge_used   = judge_result is not None
        if not judge_used:
            import sys as _sys
            print("[ClinIQ] Judge LLM timed out or failed; geval/ragas defaulting to 0.5", file=_sys.stderr)

    # ── 9-method reliability ──
    rel_score, violations = compute_reliability(
        main_response, vitals_for_validation, sofa,
        predict_result.get("shap_top5", []),
        judge_result,
    )

    rel_label, rel_color = _reliability_label(rel_score)

    # ── Parse into 4 sections ──
    sections = _parse_llm_sections(main_response)
    # Ensure all 4 keys exist even if parsing missed one
    for key in ["CURRENT CONDITION", "PROBABLE CAUSE", "RISK FORECAST", "IMMEDIATE ACTIONS"]:
        if key not in sections:
            sections[key] = ""

    result = {
        "sections":          sections,
        "raw_response":      main_response,
        "reliability_score": round(rel_score, 3),
        "reliability_label": rel_label,
        "reliability_color": rel_color,
        "caution":           rel_score < 0.60,
        "violations":        violations,
        "care_mode":         care_mode,
        "judge_used":        judge_used,
        "error":             None,
    }
    _cache_set(cache_key, result)
    return result

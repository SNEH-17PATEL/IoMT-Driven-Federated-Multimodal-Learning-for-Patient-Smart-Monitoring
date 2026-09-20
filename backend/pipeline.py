"""
ICU Clinical Decision Support System — computation pipeline.

This module is a straight port of the pipeline that used to live inline in
app.py (Streamlit). Every constant, formula and code path is unchanged —
only the presentation layer (st.markdown/st.plotly_chart/...) was removed
so the exact same computation can be served as JSON from FastAPI.
"""

import os
import re
import json
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import joblib
import shap
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from model_utils import ICUModel, get_trend, classify_range

load_dotenv()

# =============================================================
# CONSTANTS  (identical to app.py)
# =============================================================
MODEL_PATH      = "models/"
DATA_PATH       = "data/"
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

VALID_RANGES = {
    "HR":   (30,  220),
    "RR":   (5,   60),
    "SpO2": (50,  100),
    "Temp": (30,  43),
    "SBP":  (40,  250),
    "DBP":  (20,  150),
    "MAP":  (30,  200),
}

PATIENTS = {
    1: {
        "name":        "Patient 1 — Low Risk",
        "short":       "Low Risk",
        "file":        "sample_patients/sample_patient01_data.csv",
        "vitals_file": "patient_vitals/patient_vitals_01.csv",
        "hist_file":   "prediction_history/prediction_history_01.csv",
        "icon":        "🟢",
        "description": "Post-surgical recovery — stable, improving trend",
        "color":       "#28a745",
        "sofa_range":  "0 – 4",
        "risk_label":  "LOW RISK",
        "long_desc":   "Post-surgical recovery. Alert and oriented. No active infection.",
    },
    2: {
        "name":        "Patient 2 — Moderate Risk",
        "short":       "Moderate Risk",
        "file":        "sample_patients/sample_patient02_data.csv",
        "vitals_file": "patient_vitals/patient_vitals_02.csv",
        "hist_file":   "prediction_history/prediction_history_02.csv",
        "icon":        "🟡",
        "description": "Community-acquired pneumonia — on supplemental O₂",
        "color":       "#e6a817",
        "sofa_range":  "5 – 9",
        "risk_label":  "MODERATE RISK",
        "long_desc":   "Community-acquired pneumonia. On supplemental O₂ 4L/min. Elevated WBC.",
    },
    3: {
        "name":        "Patient 3 — High Risk",
        "short":       "High Risk",
        "file":        "sample_patients/sample_patient03_data.csv",
        "vitals_file": "patient_vitals/patient_vitals_03.csv",
        "hist_file":   "prediction_history/prediction_history_03.csv",
        "icon":        "🔴",
        "description": "Septic shock — vasopressors, intubated, multi-organ failure",
        "color":       "#dc3545",
        "sofa_range":  "≥ 10",
        "risk_label":  "HIGH RISK",
        "long_desc":   "Septic shock. Intubated & ventilated. Vasopressors. Multi-organ failure.",
    },
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# =============================================================
# ARTIFACT LOADING (loaded once at import time, mirrors
# st.cache_resource behaviour)
# =============================================================
_lock = threading.Lock()


def _load_artifacts():
    scaler = joblib.load(MODEL_PATH + "scaler.pkl")
    tfidf  = joblib.load(MODEL_PATH + "tfidf_vectorizer.pkl")

    fc_path = MODEL_PATH + "feature_columns.pkl"
    if os.path.exists(fc_path):
        feature_cols = joblib.load(fc_path)
    else:
        feature_cols = list(scaler.feature_names_in_)

    input_dim = len(feature_cols)
    model = ICUModel(input_dim)
    model.load_state_dict(
        torch.load(MODEL_PATH + "federated_model.pth",
                   map_location="cpu", weights_only=True)
    )
    model.eval()

    bg_path = MODEL_PATH + "shap_background.npy"
    background = np.load(bg_path).astype(np.float32) if os.path.exists(bg_path) else None

    return model, scaler, tfidf, feature_cols, background


def _load_shap_explainer(model, background):
    if background is None:
        return None
    bg_tensor = torch.tensor(background, dtype=torch.float32)
    model.eval()
    return shap.DeepExplainer(model, bg_tensor)


def load_training_metadata():
    path = MODEL_PATH + "training_metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "num_rounds": 100, "epochs_per_round": 3, "batch_size": 64,
        "hospitals": 3,
        "hospital_names": ["General ICU", "Mixed ICU", "Cardiac/Trauma ICU"],
        "aggregation": "FedYogi (η=0.01, β1=0.9, β2=0.99) + FedProx (μ=0.5)",
        "split_type": "IID",
        "train_samples": 48150, "test_samples": 12038,
        "input_features": 108,
        "model_architecture": "108 → 128 → 64 → 32 → 1  (ReLU, no Dropout, FedProx)",
        "best_round": 24,
        "final_mae": 1.9608, "final_r2": 0.3357,
        "pred_range_min": -0.26, "pred_range_max": 13.91,
        "differential_privacy": False,
        "dp_sensitivity": None, "dp_sigma": None,
        "dp_epsilon": None, "dp_delta": None,
    }


model, scaler, tfidf, feature_cols, background_data = _load_artifacts()
explainer = _load_shap_explainer(model, background_data)
training_meta = load_training_metadata()

# =============================================================
# IN-MEMORY SESSION STATE — one browser session's worth (this
# app is a single-user clinical demo, exactly like the original
# single-session Streamlit app)
# =============================================================
row_indices = {1: 0, 2: 0, 3: 0}


def reset_patient(patient_id: int):
    with _lock:
        row_indices[patient_id] = 0


# =============================================================
# HELPERS  (identical logic to app.py)
# =============================================================

def risk_label(sofa):
    if sofa < 5:
        return "Low Risk", "🟢", "#28a745"
    elif sofa < 10:
        return "Moderate Risk", "🟡", "#e6a817"
    return "High Risk", "🔴", "#dc3545"


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    elif "<think>" in text:
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

CLINICAL_KEYWORDS = [
    "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP", "GCS", "stress",
    "hypotension", "respiratory", "mental", "septic", "failure",
    "intubated", "vasopressor", "shock", "fever", "infection",
    "oxygen", "ventilat", "cardiac", "renal", "hepatic",
]

VITAL_CFG = [
    ("HR",   "HR",          60,   100),
    ("RR",   "RR",          12,   20),
    ("SpO2", "SpO₂",        95,   100),
    ("Temp", "Temperature", 36.5, 37.5),
    ("SBP",  "SBP",         100,  120),
    ("DBP",  "DBP",         60,   80),
    ("MAP",  "MAP",         70,   100),
]

_SEC_NAMES = ["IMMEDIATE ACTIONS", "CURRENT CONDITION", "PROBABLE CAUSE", "RISK FORECAST"]


def _parse_llm_sections(text):
    positions = []
    for sec in _SEC_NAMES:
        m = re.search(
            rf'\b\d+\.\s*\*{{0,2}}\s*{re.escape(sec)}\s*\*{{0,2}}\s*[—\-:–]?\s*',
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


def _resp_valid(r):
    return bool(r) and r != _FALLBACK_MSG and len(r) > 80


# =============================================================
# MAIN PIPELINE — one "reading" (mirrors the body of app.py that
# ran on every Streamlit rerun)
# =============================================================

def run_reading_pipeline(patient_id: int) -> dict:
    if patient_id not in PATIENTS:
        raise ValueError(f"Unknown patient_id {patient_id}")
    if not GROQ_API_KEY:
        return {"error": "groq_key_missing",
                "message": "Groq API key required. Add GROQ_API_KEY=your_key to the .env "
                           "file in the project root and restart the backend."}

    patient_cfg = PATIENTS[patient_id]

    with _lock:
        row_idx = row_indices[patient_id]

    patient_csv = DATA_PATH + patient_cfg["file"]
    patient_df  = pd.read_csv(patient_csv)
    total_rows  = len(patient_df)
    cur_idx     = row_idx % total_rows
    current_row = patient_df.iloc[cur_idx]
    cycle_num   = row_idx // total_rows + 1

    HR            = float(current_row["HR"])
    RR            = float(current_row["RR"])
    SpO2          = float(current_row["SpO2"])
    Temp          = float(current_row["Temp"])
    SBP           = float(current_row["SBP"])
    DBP           = float(current_row["DBP"])
    MAP           = float(current_row["MAP"])
    GCS_eye       = int(current_row["GCS_eye_opening"])
    stress        = int(current_row["stress_score"])
    clinical_note = str(current_row["clinical_note"])

    VITALS_FILE  = MODEL_PATH + patient_cfg["vitals_file"]
    HISTORY_FILE = MODEL_PATH + patient_cfg["hist_file"]

    if not os.path.exists(VITALS_FILE):
        os.makedirs(os.path.dirname(VITALS_FILE), exist_ok=True)
        _seed = patient_df[["HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"]].head(20).copy()
        _ts = [
            f"2026-08-29 {(8 + i // 6):02d}:{(i % 6) * 10:02d}:00"
            for i in range(len(_seed))
        ]
        _seed.insert(0, "time", _ts)
        _seed.to_csv(VITALS_FILE, index=False)

    # -- 1. SLIDING WINDOW --
    vitals_df = pd.read_csv(VITALS_FILE)
    new_row = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "HR": HR, "RR": RR, "SpO2": SpO2, "Temp": Temp,
        "SBP": SBP, "DBP": DBP, "MAP": MAP
    }])
    vitals_df = pd.concat([vitals_df, new_row], ignore_index=True).tail(20)
    vitals_df.to_csv(VITALS_FILE, index=False)

    # -- 2. TREND FEATURES --
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

    # -- 3. LATEST FEATURES --
    latest_features = {
        "latest_HR":       HR,
        "latest_RR":       RR,
        "latest_SpO2":     SpO2,
        "latest_Temp":     Temp,
        "latest_SBP":      SBP,
        "latest_DBP":      DBP,
        "latest_MAP":      MAP,
        "GCS_eye_opening": float(GCS_eye),
        "stress_score":    float(stress),
    }

    # -- 4. TF-IDF --
    tfidf_vec = tfidf.transform([clinical_note])
    tfidf_df  = pd.DataFrame(tfidf_vec.toarray(), columns=tfidf.get_feature_names_out())

    # -- 5. COMBINE + ALIGN --
    input_df = pd.DataFrame([{**trend_features, **latest_features}])
    final_df = pd.concat([input_df, tfidf_df], axis=1).fillna(0)

    expected_cols = feature_cols
    for col in expected_cols:
        if col not in final_df.columns:
            final_df[col] = 0
    final_df = final_df[expected_cols].astype(float)

    # -- 6. SCALE --
    final_scaled = np.clip(scaler.transform(final_df), -10, 10)

    # -- 7. PREDICT --
    X_tensor = torch.tensor(final_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        raw_pred = model(X_tensor).numpy().flatten()[0]

    sofa_score = float(np.clip(raw_pred, 0, 24))
    risk_text, risk_icon, risk_color = risk_label(sofa_score)

    # -- 8. ALERT --
    alert = None
    if sofa_score >= ALERT_THRESHOLD:
        extra = " (Clinical High Risk threshold is SOFA≥10)" if sofa_score < 10 else ""
        alert = {"level": "error",
                  "message": f"HIGH RISK PATIENT DETECTED — Immediate clinical attention required.{extra} "
                             f"Review AI assessment below and initiate appropriate protocols."}
    elif sofa_score >= 5:
        alert = {"level": "warning",
                  "message": "MODERATE RISK — Patient requires close monitoring. Review assessment below."}

    # -- 8b. SAVE HISTORY --
    _hist_row = pd.DataFrame([{
        "Timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Reading":     f"{cur_idx + 1}/{total_rows}",
        "SOFA":        round(sofa_score, 1),
        "Risk":        risk_text,
        "HR":          HR,
        "RR":          RR,
        "SpO₂":        SpO2,
        "Temp (°C)":   Temp,
        "SBP":         SBP,
        "MAP":         MAP,
        "GCS Eye":     GCS_eye,
        "Stress":      stress,
    }])
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    if os.path.exists(HISTORY_FILE):
        _existing = pd.read_csv(HISTORY_FILE)
        _hist_df  = pd.concat([_existing, _hist_row], ignore_index=True).tail(50)
    else:
        _hist_df = _hist_row
    _hist_df.to_csv(HISTORY_FILE, index=False)

    # -- 9. SHAP --
    shap_vals = None
    top_shap_records = []
    clinical_explanations = []
    key_risks = []

    if explainer is not None:
        try:
            model.eval()
            shap_raw = explainer.shap_values(X_tensor)
            if isinstance(shap_raw, list):
                shap_vals = shap_raw[0][0]
            else:
                shap_vals = shap_raw[0] if shap_raw.ndim > 1 else shap_raw
            shap_vals = np.array(shap_vals).flatten()
        except Exception:
            shap_vals = None

    if shap_vals is not None:
        original_values = final_df.iloc[0].values

        shap_df = pd.DataFrame({
            "feature":        expected_cols,
            "original_value": original_values,
            "impact":         shap_vals
        })
        shap_df["abs_impact"] = shap_df["impact"].abs()
        shap_df = shap_df.sort_values("abs_impact", ascending=False)

        def is_clinical(f):
            return any(k.lower() in f.lower() for k in CLINICAL_KEYWORDS)

        filtered = shap_df[shap_df["feature"].apply(is_clinical)]
        top_shap = (filtered.head(7) if len(filtered) >= 3 else shap_df.head(7)).copy()

        clinical_explanations = [
            interpret_shap(r["feature"], r["original_value"], r["impact"])
            for _, r in top_shap.iterrows()
        ]

        max_imp = float(top_shap["abs_impact"].max()) if not top_shap.empty else 1.0
        for _, r in top_shap.iterrows():
            feat, orig, imp, abimp = r["feature"], r["original_value"], r["impact"], r["abs_impact"]
            if feat in VITAL_UNITS:
                lbl, unit = VITAL_UNITS[feat]
                fmt = ".0f" if unit in ("/4", "/10") else ".1f"
                val_str = f"{orig:{fmt}} {unit}"
            else:
                lbl, val_str = feat, ("detected in notes" if orig > 0 else "absent in notes")
            top_shap_records.append({
                "feature": feat,
                "label": lbl,
                "value": val_str,
                "impact": float(imp),
                "abs_impact": float(abimp),
                "bar_pct": round(abimp / max_imp * 100, 1) if max_imp else 0,
                "increases_risk": bool(imp > 0),
            })

        seen = set()
        for _, r in top_shap.iterrows():
            for k, v in RISK_MAP.items():
                if k.lower() in r["feature"].lower() and v not in seen:
                    key_risks.append(v)
                    seen.add(v)

    # -- 10. TREND ANALYSIS --
    trend_lines = []
    for col, label, lo, hi in VITAL_CFG:
        direction = get_trend(vitals_df[col].tolist())
        status    = classify_range(vitals_df[col].mean(), lo, hi)
        trend_lines.append({"vital": label, "range": status, "direction": direction})

    # -- 11. BUILD LLM PROMPT --
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
    trend_text = "\n".join(
        f"- {t['vital']} → {t['range']} & {t['direction']}" for t in trend_lines
    )

    if sofa_score >= ALERT_THRESHOLD:
        _urgency_prefix = (
            f"⚠️ CLINICAL ALERT — Predicted SOFA {sofa_score:.1f} "
            f"(alert threshold ≥ {ALERT_THRESHOLD:.0f})\n\n"
            "This patient shows signs of significant deterioration. "
            "Structure your response for IMMEDIATE clinical action.\n"
        )
        _section_instructions = (
            "1. IMMEDIATE ACTIONS — List the 3 most critical interventions "
            "needed in the NEXT 30 MINUTES (be specific: drug names, doses, procedures)\n"
            "2. CURRENT CONDITION — What is happening with this patient right now\n"
            "3. PROBABLE CAUSE — Why is this deterioration occurring\n"
            "4. RISK FORECAST — What may happen in the next 1–2 hours if untreated"
        )
    else:
        _urgency_prefix = ""
        _section_instructions = (
            "1. CURRENT CONDITION — What is happening with this patient right now\n"
            "2. PROBABLE CAUSE — Why is this deterioration occurring\n"
            "3. RISK FORECAST — What may happen in the next 2–4 hours if untreated\n"
            "4. IMMEDIATE ACTIONS — Specific interventions required now"
        )

    final_prompt = f"""{_urgency_prefix}You are an ICU clinical decision support assistant.

Analyze the patient data below and provide a structured response with exactly 4 sections:

{_section_instructions}

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

Be specific and clinically precise. Base reasoning strictly on the data above."""

    # -- 12. LLM SELF-CONSISTENCY --
    try:
        responses     = get_multiple_llm_responses(GROQ_API_KEY, final_prompt, n=3)
        main_response = responses[0]
        consistency   = compute_consistency(responses)
    except Exception as e:
        responses     = []
        main_response = f"⚠️ LLM unavailable: {e}"
        consistency   = 0.0

    sections = _parse_llm_sections(main_response) if main_response else {}
    resp_valid = [_resp_valid(r) for r in responses] if responses else [False, False, False]
    while len(resp_valid) < 3:
        resp_valid.append(False)

    # -- advance state for next call --
    with _lock:
        row_indices[patient_id] = (row_idx + 1) % total_rows

    history_tail = []
    if os.path.exists(HISTORY_FILE):
        _ph = pd.read_csv(HISTORY_FILE)
        history_tail = _ph.iloc[::-1].to_dict(orient="records")

    vitals_window = vitals_df.to_dict(orient="records")

    gcs_labels = {4: "Spontaneous", 3: "To Voice", 2: "To Pain", 1: "No Response"}

    return {
        "patient": {**patient_cfg, "id": patient_id},
        "reading_index": cur_idx + 1,
        "total_rows": total_rows,
        "cycle_num": cycle_num,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "vitals": {
            "HR": HR, "RR": RR, "SpO2": SpO2, "Temp": Temp,
            "SBP": SBP, "DBP": DBP, "MAP": MAP,
            "GCS_eye": GCS_eye, "GCS_label": gcs_labels.get(GCS_eye, "?"),
            "stress": stress,
        },
        "normal_ranges": NORMAL_RANGES,
        "clinical_note": clinical_note,
        "vitals_window": vitals_window,
        "sofa_score": round(sofa_score, 2),
        "risk_text": risk_text,
        "risk_icon": risk_icon,
        "risk_color": risk_color,
        "alert": alert,
        "alert_threshold": ALERT_THRESHOLD,
        "shap": top_shap_records,
        "clinical_explanations": clinical_explanations,
        "key_risks": key_risks,
        "trend_lines": trend_lines,
        "llm": {
            "responses": responses,
            "response_valid": resp_valid,
            "main_response": main_response,
            "sections": sections,
            "consistency": round(consistency, 4),
        },
        "history": history_tail,
        "model_meta": training_meta,
    }

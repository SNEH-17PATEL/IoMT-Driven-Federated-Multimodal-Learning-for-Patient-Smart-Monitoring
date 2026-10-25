import os
import sys
import json
import functools
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import joblib
import shap

# Make icu_monitor/ importable (db.py, model_utils.py live there)
_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))   # backend/services/
_BACKEND_DIR  = os.path.dirname(_SERVICES_DIR)               # backend/
_ROOT         = os.path.dirname(_BACKEND_DIR)                 # icu_monitor/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model_utils import ICUModel, get_trend, classify_range
from db import (
    get_sample_patient_count, get_sample_patient_row,
    get_patient_vitals, append_patient_vitals,
    append_prediction, get_prediction_history,
    fl_training_exists, get_fl_client_data,
)

MODEL_PATH = os.path.join(_ROOT, "models") + os.sep

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

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

VITAL_UNITS = {
    "HR_mean":         ("HR mean",           "bpm"),
    "HR_std":          ("HR variability",     "bpm"),
    "RR_mean":         ("RR mean",            "br/min"),
    "SpO2_mean":       ("SpO₂ mean",          "%"),
    "SpO2_min":        ("SpO₂ minimum",       "%"),
    "Temp_mean":       ("Temperature mean",   "°C"),
    "SBP_mean":        ("SBP mean",           "mmHg"),
    "DBP_mean":        ("DBP mean",           "mmHg"),
    "MAP_mean":        ("MAP mean",           "mmHg"),
    "latest_HR":       ("Heart Rate",         "bpm"),
    "latest_RR":       ("Respiratory Rate",   "br/min"),
    "latest_SpO2":     ("SpO₂",              "%"),
    "latest_Temp":     ("Temperature",        "°C"),
    "latest_SBP":      ("Systolic BP",        "mmHg"),
    "latest_DBP":      ("Diastolic BP",       "mmHg"),
    "latest_MAP":      ("Mean Arterial P.",   "mmHg"),
    "GCS_eye_opening": ("GCS Eye Opening",    "/4"),
    "stress_score":    ("Stress Score",       "/10"),
}

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

# ──────────────────────────────────────────────
# CACHED LOADERS  (lru_cache replaces st.cache_resource)
# ──────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def load_artifacts():
    scaler = joblib.load(MODEL_PATH + "scaler.pkl")
    tfidf  = joblib.load(MODEL_PATH + "tfidf_vectorizer.pkl")

    fc_path = MODEL_PATH + "feature_columns.pkl"
    feature_cols = (
        joblib.load(fc_path) if os.path.exists(fc_path)
        else list(scaler.feature_names_in_)
    )

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


@functools.lru_cache(maxsize=1)
def _load_shap_explainer():
    model, _, _, _, background = load_artifacts()
    if background is None:
        return None
    bg_tensor = torch.tensor(background, dtype=torch.float32)
    model.eval()
    return shap.DeepExplainer(model, bg_tensor)


@functools.lru_cache(maxsize=1)
def _compute_conformal_q_hat():
    model, _, _, feature_cols, _ = load_artifacts()
    try:
        if not fl_training_exists():
            return 2.9
        errors = []
        for i in range(3):
            df = get_fl_client_data(i)
            if df.empty or "sofa_score" not in df.columns:
                continue
            y_true = df["sofa_score"].values.astype(np.float32)
            X = df.drop(columns=["sofa_score"])
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[list(feature_cols)].values.astype(np.float32)
            X = np.clip(X, -10, 10)
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
            errors.extend(np.abs(preds - y_true).tolist())
        if len(errors) < 50:
            return 2.9
        return round(float(np.quantile(errors, 0.90)), 2)
    except Exception:
        return 2.9


@functools.lru_cache(maxsize=1)
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
        "final_mae": 1.7588, "final_r2": 0.4724,
    }

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def risk_label(sofa):
    if sofa < 5:
        return "Low Risk", "🟢", "#28a745"
    elif sofa < 10:
        return "Moderate Risk", "🟡", "#e6a817"
    return "High Risk", "🔴", "#dc3545"


def _vital_status(value, key):
    lo, hi = NORMAL_RANGES[key]
    if value < lo:
        return "low"
    if value > hi:
        return "high"
    return "normal"


def compute_sofa_floor(MAP_val, SpO2_val, GCS_eye_val):
    floor = 0
    if MAP_val  < 70: floor += 1
    if MAP_val  < 65: floor += 1
    if SpO2_val < 94: floor += 1
    if SpO2_val < 90: floor += 1
    if GCS_eye_val == 2: floor += 2
    if GCS_eye_val == 1: floor += 3
    return floor


def vital_consistency_flags(HR_val, SBP_val, SpO2_val, MAP_val, sofa_val):
    flags = []
    if HR_val > 130 and SBP_val < 90 and sofa_val < 4:
        flags.append("HR > 130 + SBP < 90 implies shock state — SOFA expected ≥ 4")
    if SpO2_val < 88 and MAP_val < 65 and sofa_val < 6:
        flags.append("SpO₂ < 88% + MAP < 65 implies multi-organ stress — SOFA expected ≥ 6")
    return flags


def check_trajectory(patient_id, cur_sofa, HR_val, RR_val, SpO2_val, SBP_val, MAP_val):
    try:
        hist = get_prediction_history(patient_id, limit=2)
    except Exception:
        return None, 0.0, False, "History unavailable"
    if len(hist) < 2:
        return None, 0.0, False, "First reading"

    prev      = hist.iloc[-1]
    prev_sofa = float(prev.get("SOFA", cur_sofa))
    jump      = cur_sofa - prev_sofa

    sig = sum([
        abs(HR_val   - float(prev.get("HR",    HR_val)))   >= 20,
        abs(RR_val   - float(prev.get("RR",    RR_val)))   >= 4,
        abs(SpO2_val - float(prev.get("SpO₂",  SpO2_val))) >= 5,
        abs(SBP_val  - float(prev.get("SBP",   SBP_val)))  >= 20,
        abs(MAP_val  - float(prev.get("MAP",    MAP_val)))  >= 15,
    ])

    if abs(jump) > 4 and sig == 0:
        direction = "↑" if jump > 0 else "↓"
        return prev_sofa, jump, True, (
            f"SOFA {direction} {abs(jump):.1f} pts "
            f"({prev_sofa:.1f} → {cur_sofa:.1f}) with no significant vital sign change"
        )
    return prev_sofa, jump, False, None


def _interpret_shap(feature, orig_value, impact):
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


# ──────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────

def run_prediction(patient_id: int, row_idx: int) -> dict:
    """
    Reads sample patient row from DB, runs full SOFA pipeline,
    saves vitals + prediction to DB, returns complete result dict.
    """
    model, scaler, tfidf, feature_cols, _ = load_artifacts()
    explainer      = _load_shap_explainer()
    conformal_q    = _compute_conformal_q_hat()

    # ── 1. READ SIMULATION ROW ──
    total_rows = get_sample_patient_count(patient_id)
    if total_rows == 0:
        raise ValueError(f"No sample data for patient {patient_id}. Run migrate_to_db.py first.")

    cur_idx     = row_idx % total_rows
    current_row = get_sample_patient_row(patient_id, cur_idx)

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

    # ── 2. SEED VITALS IF EMPTY ──
    if len(get_patient_vitals(patient_id, limit=1)) == 0:
        for _si in range(min(20, total_rows)):
            _sr = get_sample_patient_row(patient_id, _si)
            if _sr is not None:
                append_patient_vitals(patient_id, {
                    "time": f"2026-08-29 {(8 + _si // 6):02d}:{(_si % 6) * 10:02d}:00",
                    "HR": float(_sr["HR"]),   "RR":  float(_sr["RR"]),
                    "SpO2": float(_sr["SpO2"]), "Temp": float(_sr["Temp"]),
                    "SBP": float(_sr["SBP"]),  "DBP":  float(_sr["DBP"]),
                    "MAP": float(_sr["MAP"]),
                })

    # ── 3. APPEND CURRENT VITALS ──
    append_patient_vitals(patient_id, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "HR": HR, "RR": RR, "SpO2": SpO2, "Temp": Temp,
        "SBP": SBP, "DBP": DBP, "MAP": MAP,
    })
    vitals_df = get_patient_vitals(patient_id, limit=20)

    # ── 4. TREND FEATURES ──
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

    # ── 5. LATEST FEATURES ──
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

    # ── 6. TF-IDF ──
    tfidf_vec = tfidf.transform([clinical_note])
    tfidf_df  = pd.DataFrame(tfidf_vec.toarray(), columns=tfidf.get_feature_names_out())

    # ── 7. COMBINE + ALIGN ──
    input_df  = pd.DataFrame([{**trend_features, **latest_features}])
    final_df  = pd.concat([input_df, tfidf_df], axis=1).fillna(0)
    for col in feature_cols:
        if col not in final_df.columns:
            final_df[col] = 0
    final_df = final_df[list(feature_cols)].astype(float)

    # ── 8. SCALE + PREDICT ──
    final_scaled = np.clip(scaler.transform(final_df), -10, 10)
    X_tensor     = torch.tensor(final_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        raw_pred = model(X_tensor).numpy().flatten()[0]

    sofa_score            = float(np.clip(raw_pred, 0, 24))
    risk_text, risk_icon, risk_color = risk_label(sofa_score)
    conf_lo               = round(max(0.0,  sofa_score - conformal_q), 1)
    conf_hi               = round(min(24.0, sofa_score + conformal_q), 1)

    # ── 9. VALIDATION ──
    _sofa_floor  = compute_sofa_floor(MAP, SpO2, GCS_eye)
    _vital_flags = vital_consistency_flags(HR, SBP, SpO2, MAP, sofa_score)
    _prev_sofa, _sofa_jump, _traj_flagged, _traj_msg = check_trajectory(
        patient_id, sofa_score, HR, RR, SpO2, SBP, MAP
    )

    # ── 10. SAVE PREDICTION ──
    append_prediction(patient_id, {
        "Timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Reading":    f"{cur_idx + 1}/{total_rows}",
        "SOFA":       round(sofa_score, 1),
        "Risk":       risk_text,
        "HR":         HR,
        "RR":         RR,
        "SpO₂":       SpO2,
        "Temp (°C)":  Temp,
        "SBP":        SBP,
        "DBP":        DBP,
        "MAP":        MAP,
        "GCS Eye":    GCS_eye,
        "Stress":     stress,
    })

    # ── 11. SHAP ──
    shap_vals             = None
    top_shap_list         = []
    clinical_explanations = []
    key_risks             = []

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
            "feature":        list(feature_cols),
            "original_value": original_values,
            "impact":         shap_vals,
        })
        shap_df["abs_impact"] = shap_df["impact"].abs()
        shap_df = shap_df.sort_values("abs_impact", ascending=False)

        def _is_clinical(f):
            return any(k.lower() in f.lower() for k in CLINICAL_KEYWORDS)

        filtered  = shap_df[shap_df["feature"].apply(_is_clinical)]
        top_shap_df = (filtered.head(7) if len(filtered) >= 3 else shap_df.head(7)).copy()

        clinical_explanations = [
            _interpret_shap(r["feature"], r["original_value"], r["impact"])
            for _, r in top_shap_df.iterrows()
        ]

        seen = set()
        for _, r in top_shap_df.iterrows():
            for k, v in RISK_MAP.items():
                if k.lower() in r["feature"].lower() and v not in seen:
                    key_risks.append(v)
                    seen.add(v)

        # Build top-5 SHAP list for frontend
        max_abs = top_shap_df["abs_impact"].max() if len(top_shap_df) > 0 else 1.0
        for _, r in top_shap_df.head(5).iterrows():
            feat   = r["feature"]
            label, unit = VITAL_UNITS.get(feat, (feat, ""))
            top_shap_list.append({
                "feature":    feat,
                "label":      label,
                "value":      round(float(r["original_value"]), 2),
                "unit":       unit,
                "impact":     round(float(r["impact"]), 3),
                "abs_impact": round(float(r["abs_impact"]), 3),
                "rel_impact": round(float(r["abs_impact"]) / max_abs, 3),
                "direction":  "up" if r["impact"] > 0 else "down",
            })

    # ── 12. TREND TEXT ──
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

    # ── 13. VITALS FOR CHARTS ──
    vitals_for_charts = vitals_df[[
        "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"
    ]].copy()
    vitals_for_charts.index = range(len(vitals_for_charts))
    vitals_for_charts_list = vitals_for_charts.to_dict(orient="records")

    # ── 14. BUILD VITALS BLOCK (for LLM, passed through in result) ──
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

    return {
        "patient_id":    patient_id,
        "row_idx":       row_idx,
        "cur_idx":       cur_idx,
        "total_rows":    total_rows,
        "sofa_score":    round(sofa_score, 1),
        "risk_level":    risk_text,
        "risk_color":    risk_color,
        "risk_icon":     risk_icon,
        "is_alert":      sofa_score >= ALERT_THRESHOLD,
        "alert_threshold": ALERT_THRESHOLD,
        "conformal_lo":  conf_lo,
        "conformal_hi":  conf_hi,
        "conformal_q":   conformal_q,
        "vitals": {
            "HR": HR, "RR": RR, "SpO2": SpO2, "Temp": Temp,
            "SBP": SBP, "DBP": DBP, "MAP": MAP,
            "GCS_eye": GCS_eye, "stress": stress,
        },
        "vitals_status": {
            "HR":   _vital_status(HR,   "HR"),
            "RR":   _vital_status(RR,   "RR"),
            "SpO2": _vital_status(SpO2, "SpO2"),
            "Temp": _vital_status(Temp, "Temp"),
            "SBP":  _vital_status(SBP,  "SBP"),
            "DBP":  _vital_status(DBP,  "DBP"),
            "MAP":  _vital_status(MAP,  "MAP"),
        },
        "clinical_note":        clinical_note,
        "shap_top5":            top_shap_list,
        "key_risks":            key_risks,
        "clinical_explanations": clinical_explanations,
        "trend_text":           trend_text,
        "vitals_history":       vitals_for_charts_list,
        # Blocks passed to LLM in Phase 3
        "_vitals_block":      vitals_block,
        "_cv_block":          cv_block,
        "_risk_block":        risk_block,
        "_explanation_block": explanation_block,
        # Validation fields consumed by SOFAValidationDrawer (Layer 2)
        "_prev_sofa":         _prev_sofa,
        "_sofa_jump":         round(_sofa_jump, 1) if _sofa_jump is not None else None,
        "_traj_flagged":      _traj_flagged,
        "_traj_msg":          _traj_msg,
    }

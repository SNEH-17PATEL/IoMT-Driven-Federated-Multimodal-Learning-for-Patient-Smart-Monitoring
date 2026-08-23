"""
ICU Clinical Decision Support System
Multimodal Intelligence System with Federated Learning
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from model_utils import ICUModel, get_trend, classify_range

load_dotenv()

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="ICU Risk Monitor",
    page_icon="🏥",
    layout="wide"
)

# =============================================================
# CONSTANTS
# =============================================================
MODEL_PATH    = "models/"
VITALS_FILE   = MODEL_PATH + "patient_vitals.csv"
HISTORY_FILE  = MODEL_PATH + "prediction_history.csv"

# Alert fires at predicted SOFA ≥ ALERT_THRESHOLD (not ≥ 10).
# The federated model systematically under-predicts high SOFA by ~2-3 pts —
# a patient with true SOFA=13 is often predicted as ~9.
# At threshold ≥ 8: High Risk recall jumps from 28.4% → 52.5%,
# false alarm rate on Low Risk patients stays low at 1.4%.
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

DEFAULT_VITALS = [
    ["2026-03-15 09:00:00", 88, 18, 97, 37.1, 120, 80, 93],
    ["2026-03-15 09:10:00", 90, 19, 97, 37.2, 118, 78, 91],
    ["2026-03-15 09:20:00", 91, 20, 96, 37.2, 117, 77, 90],
    ["2026-03-15 09:30:00", 92, 20, 96, 37.3, 115, 76, 89],
    ["2026-03-15 09:40:00", 93, 21, 96, 37.3, 114, 75, 88],
    ["2026-03-15 09:50:00", 95, 21, 95, 37.4, 112, 74, 87],
    ["2026-03-15 10:00:00", 96, 22, 95, 37.4, 110, 73, 86],
    ["2026-03-15 10:10:00", 97, 22, 94, 37.5, 108, 72, 84],
    ["2026-03-15 10:20:00", 98, 23, 94, 37.5, 106, 70, 82],
    ["2026-03-15 10:30:00", 99, 23, 93, 37.6, 105, 69, 81],
    ["2026-03-15 10:40:00", 100, 24, 93, 37.6, 104, 68, 80],
    ["2026-03-15 10:50:00", 101, 24, 92, 37.7, 102, 67, 79],
    ["2026-03-15 11:00:00", 102, 25, 92, 37.7, 100, 66, 77],
    ["2026-03-15 11:10:00", 103, 25, 91, 37.8,  99, 65, 76],
    ["2026-03-15 11:20:00", 104, 26, 91, 37.8,  98, 64, 75],
    ["2026-03-15 11:30:00", 105, 26, 90, 37.9,  96, 63, 74],
    ["2026-03-15 11:40:00", 106, 27, 90, 37.9,  95, 62, 73],
    ["2026-03-15 11:50:00", 107, 27, 89, 38.0,  94, 61, 72],
    ["2026-03-15 12:00:00", 108, 28, 89, 38.0,  92, 60, 71],
    ["2026-03-15 12:10:00", 109, 28, 88, 38.1,  90, 59, 69],
]

# =============================================================
# LOAD ARTIFACTS (cached — runs only once)
# =============================================================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODEL_PATH + "scaler.pkl")
    tfidf  = joblib.load(MODEL_PATH + "tfidf_vectorizer.pkl")

    # Load feature column order from the dedicated artifact saved during training.
    # This is identical to scaler.feature_names_in_ but using a separate file
    # decouples feature ordering from the scaler and provides a fallback if a
    # future sklearn version changes how feature_names_in_ is serialised.
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


@st.cache_resource
def load_shap_explainer(_model, _background):
    if _background is None:
        return None
    bg_tensor = torch.tensor(_background, dtype=torch.float32)
    _model.eval()
    return shap.DeepExplainer(_model, bg_tensor)


model, scaler, tfidf, feature_cols, background_data = load_artifacts()
explainer = load_shap_explainer(model, background_data)


@st.cache_resource
def load_training_metadata():
    """Load FL training metadata saved by train_federated.py."""
    path = MODEL_PATH + "training_metadata.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fallback: hardcoded defaults for current trained model
    return {
        "num_rounds": 20, "epochs_per_round": 10, "batch_size": 64,
        "hospitals": 3,
        "hospital_names": ["General ICU", "Mixed ICU", "Cardiac/Trauma ICU"],
        "aggregation": "FedAvg", "split_type": "IID",
        "train_samples": 38520, "test_samples": 9631,
        "input_features": 618,
        "model_architecture": "618 → 256 → 128 → 64 → 1  (ReLU)",
        "best_round": 19,
        "final_mae": 2.071, "final_r2": 0.222,
        "pred_range_min": 0.60, "pred_range_max": 20.25,
        "differential_privacy": False,
        "dp_sensitivity": None, "dp_sigma": None,
        "dp_epsilon": None, "dp_delta": None,
    }

training_meta = load_training_metadata()

# =============================================================
# INITIALISE VITALS FILE
# =============================================================
if not os.path.exists(VITALS_FILE):
    pd.DataFrame(
        DEFAULT_VITALS,
        columns=["time", "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"]
    ).to_csv(VITALS_FILE, index=False)

# =============================================================
# SIDEBAR — INPUTS
# =============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

st.sidebar.title("🏥 ICU Monitor")
st.sidebar.caption("Multimodal Patient Monitoring")
st.sidebar.divider()

if not GROQ_API_KEY:
    GROQ_API_KEY = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Paste your Groq API key, or add GROQ_API_KEY to the .env file"
    )
    st.sidebar.divider()

st.sidebar.subheader("📋 Clinical Notes")
clinical_note = st.sidebar.text_area(
    "Patient notes (history, medications, observations)",
    placeholder="e.g. 67-year-old male. History of COPD and hypertension. Currently on vasopressors. Oxygen requirement increasing...",
    height=130,
    label_visibility="collapsed"
)
if not clinical_note.strip():
    clinical_note = "No clinical notes provided"

st.sidebar.subheader("💓 Vital Signs")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    HR   = st.number_input("HR (bpm)",    min_value=30.0, max_value=220.0, value=90.0,  step=1.0)
    RR   = st.number_input("RR (br/min)", min_value=5.0,  max_value=60.0,  value=18.0,  step=1.0)
    SpO2 = st.number_input("SpO₂ (%)",   min_value=50.0, max_value=100.0, value=97.0,  step=0.5)
    Temp = st.number_input("Temp (°C)",  min_value=30.0, max_value=43.0,  value=37.0,  step=0.1)
with col_b:
    SBP  = st.number_input("SBP (mmHg)", min_value=40.0,  max_value=250.0, value=120.0, step=1.0)
    DBP  = st.number_input("DBP (mmHg)", min_value=20.0,  max_value=150.0, value=80.0,  step=1.0)
    MAP  = st.number_input("MAP (mmHg)", min_value=30.0,  max_value=200.0, value=90.0,  step=1.0)

st.sidebar.subheader("👁️ Computer Vision Features")
GCS_eye = st.sidebar.selectbox(
    "GCS Eye Opening",
    options=[4, 3, 2, 1],
    format_func=lambda x: {
        4: "4 — Spontaneous",
        3: "3 — To voice",
        2: "2 — To pain",
        1: "1 — No response"
    }[x]
)
stress = st.sidebar.slider(
    "Stress Score (0 = calm, 10 = severe distress)",
    min_value=0, max_value=10, value=3
)
st.sidebar.divider()
run = st.sidebar.button("🚀 Run Prediction", type="primary", use_container_width=True)

# ---- RESET PATIENT HISTORY ----
if st.sidebar.button(
    "🔄 Reset Patient History",
    type="secondary",
    use_container_width=True,
    help="Clears the 20-reading sliding window. Use when switching to a new patient."
):
    pd.DataFrame(
        DEFAULT_VITALS,
        columns=["time", "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"]
    ).to_csv(VITALS_FILE, index=False)
    st.sidebar.success("✅ Patient history reset to defaults.")
    st.rerun()

# ---- MODEL INFORMATION ----
with st.sidebar.expander("ℹ️ Model Information"):
    _m = training_meta
    st.caption(f"**Algorithm:** Federated DNN  •  FedAvg")
    st.caption(
        f"**Accuracy:** MAE = {_m['final_mae']:.3f} SOFA pts  "
        f"|  R² = {_m['final_r2']:.3f}"
    )
    st.caption(
        f"**Training:** {_m['train_samples']:,} ICU samples  "
        f"•  {_m['num_rounds']} FL rounds  "
        f"•  {_m['hospitals']} hospitals"
    )
    dp_status = "Enabled" if _m.get("differential_privacy") else "Disabled"
    st.caption(f"**Privacy:** {dp_status}  •  Split: {_m.get('split_type','IID')}")

# =============================================================
# HEADER
# =============================================================
st.title("🏥 ICU Clinical Decision Support System")
st.caption(
    "Multimodal Intelligence System — Federated Learning | SHAP Explainability | LLM Self-Consistency"
)

if not run:
    st.info(
        "Enter patient data in the sidebar and click **Run Prediction** to generate a clinical assessment.",
        icon="ℹ️"
    )
    st.stop()

# =============================================================
# INPUT VALIDATION
# =============================================================
vitals_input = {
    "HR": HR, "RR": RR, "SpO2": SpO2,
    "Temp": Temp, "SBP": SBP, "DBP": DBP, "MAP": MAP
}
invalid = []
for vital, value in vitals_input.items():
    lo, hi = VALID_RANGES[vital]
    if not (lo <= value <= hi):
        invalid.append(f"{vital}: {value} (valid range: {lo}–{hi})")

if invalid:
    st.error("⛔ Invalid vital sign values detected:")
    for msg in invalid:
        st.write(f"  • {msg}")
    st.stop()

if not GROQ_API_KEY:
    st.error("Groq API key is required. Enter it in the sidebar or set GROQ_API_KEY in the .env file.")
    st.stop()

# =============================================================
# HELPER FUNCTIONS
# =============================================================
def risk_label(sofa):
    if sofa < 5:
        return "Low Risk", "🟢", "#28a745"
    elif sofa < 10:
        return "Moderate Risk", "🟡", "#e6a817"
    return "High Risk", "🔴", "#dc3545"


def build_trend_chart(df):
    """
    Subplot chart: each vital gets its own panel with its own y-axis scale.
    Previous single-axis design made Temperature (±2°C range) invisible
    against HR and SBP (±100 range). Each panel also shows the green normal
    range band so clinicians can immediately see abnormal values.

    Layout: 2 rows × 4 cols
      Row 1: Heart Rate | Respiratory Rate | SpO₂ | Temperature
      Row 2: Systolic BP | Diastolic BP | MAP | (empty)
    """
    # (column, display name, unit, hex color, normal_low, normal_high)
    VITAL_PANELS = [
        ("HR",   "Heart Rate",       "bpm",   "#e74c3c", 60,   100),
        ("RR",   "Respiratory Rate", "br/min","#3498db", 12,   20),
        ("SpO2", "SpO₂",            "%",     "#2ecc71", 95,   100),
        ("Temp", "Temperature",      "°C",    "#f39c12", 36.5, 37.5),
        ("SBP",  "Systolic BP",      "mmHg",  "#9b59b6", 100,  120),
        ("DBP",  "Diastolic BP",     "mmHg",  "#1abc9c", 60,   80),
        ("MAP",  "MAP",              "mmHg",  "#e67e22", 70,   100),
    ]

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"{v[1]} ({v[2]})" for v in VITAL_PANELS],
        vertical_spacing=0.20,
        horizontal_spacing=0.07,
    )

    x = list(range(1, len(df) + 1))

    for i, (col, name, unit, color, lo, hi) in enumerate(VITAL_PANELS):
        row = (i // 4) + 1
        c   = (i %  4) + 1

        # Green shaded band = normal clinical range
        fig.add_hrect(
            y0=lo, y1=hi,
            fillcolor="rgba(80,200,120,0.15)",
            line_width=0,
            row=row, col=c
        )

        # Trend line with hover showing actual value + unit
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col].values.tolist(),
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=4),
                name=name,
                showlegend=False,
                hovertemplate=f"{name}: %{{y:.1f}} {unit}<extra></extra>"
            ),
            row=row, col=c
        )

    fig.update_layout(
        title_text=(
            "Vital Signs — Last 20 Readings  "
            "<span style='color:green;font-size:12px'>■ green band = normal range</span>"
        ),
        title_font_size=13,
        height=440,
        margin=dict(l=0, r=0, t=65, b=5),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(title_text="Reading →", title_font_size=9)

    return fig


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
            temperature=0.7,
            max_tokens=800
        )
        responses.append(resp.choices[0].message.content)
    return responses


def compute_consistency(responses):
    if len(responses) < 2:
        return 0.0
    vec = TfidfVectorizer(stop_words="english").fit_transform(responses)
    sim = cosine_similarity(vec)
    n = len(responses)
    score = (sim.sum() - n) / (n * (n - 1))
    return float(np.clip(score, 0.0, 1.0))


# Maps vital/CV feature names → (display label, unit string)
# Used to show original clinical values (not z-scores) in SHAP explanations.
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
    """
    Convert a SHAP feature + its original clinical value into a human-readable
    clinical explanation.

    Vital/CV features: show original value with unit (e.g., "SpO₂ mean = 88.0%").
    TF-IDF text features: show the clinical term (no numeric value — TF-IDF
      weights are meaningless to clinicians).
    """
    direction = "increasing risk" if impact > 0 else "reducing risk"
    f = feature.lower()

    if feature in VITAL_UNITS:
        label, unit = VITAL_UNITS[feature]
        fmt = ".0f" if unit in ("/4", "/10") else ".1f"
        val_str = f"{orig_value:{fmt}}"
        if "spo2" in f:
            return f"{label} = {val_str}{unit} → low oxygen levels, {direction}"
        if "rr" in f:
            return f"{label} = {val_str}{unit} → respiratory distress, {direction}"
        if "hr" in f:
            return f"{label} = {val_str}{unit} → abnormal heart rate pattern, {direction}"
        if any(x in f for x in ("sbp", "dbp", "map")):
            return f"{label} = {val_str}{unit} → blood pressure instability, {direction}"
        if "temp" in f:
            return f"{label} = {val_str}{unit} → possible infection / fever, {direction}"
        if "gcs" in f:
            return f"{label} = {val_str}{unit} → neurological deterioration, {direction}"
        if "stress" in f:
            return f"{label} = {val_str}{unit} → elevated physiological stress, {direction}"
        return f"{label} = {val_str}{unit} → {direction}"

    # TF-IDF clinical text feature — show the term, not a z-score value
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

# Keywords used to filter SHAP features to clinically relevant ones.
# Vital/CV names + important clinical TF-IDF terms that may appear in top SHAP.
CLINICAL_KEYWORDS = [
    # Vital and CV feature names
    "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP", "GCS", "stress",
    # High-value clinical TF-IDF terms
    "hypotension", "respiratory", "mental", "septic", "failure",
    "intubated", "vasopressor", "shock", "fever", "infection",
    "oxygen", "ventilat", "cardiac", "renal", "hepatic",
]

# =============================================================
# PIPELINE
# =============================================================

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
tfidf_vec   = tfidf.transform([clinical_note])
tfidf_df    = pd.DataFrame(tfidf_vec.toarray(), columns=tfidf.get_feature_names_out())

# -- 5. COMBINE + ALIGN --
input_df = pd.DataFrame([{**trend_features, **latest_features}])
final_df = pd.concat([input_df, tfidf_df], axis=1).fillna(0)

expected_cols = feature_cols
for col in expected_cols:
    if col not in final_df.columns:
        final_df[col] = 0
final_df = final_df[expected_cols].astype(float)

# -- 6. SCALE --
final_scaled = scaler.transform(final_df)

# -- 7. PREDICT --
X_tensor = torch.tensor(final_scaled, dtype=torch.float32)
model.eval()
with torch.no_grad():
    raw_pred = model(X_tensor).numpy().flatten()[0]

# Model outputs SOFA directly (0-24 range) — no scaling needed.
# train_federated.py also trains on raw SOFA to keep this consistent.
sofa_score = float(np.clip(raw_pred, 0, 24))
risk_text, risk_icon, risk_color = risk_label(sofa_score)

# -- 8. ALERT BANNER --
# Alert fires at ALERT_THRESHOLD (8.0) instead of the clinical SOFA ≥ 10 boundary.
# Rationale: the model under-predicts high SOFA by ~2-3 points. Lowering the
# alert threshold compensates for this, doubling High Risk recall (28%→52%) while
# keeping the false alarm rate on stable patients low (1.4%).
if sofa_score >= ALERT_THRESHOLD:
    extra = " (Clinical High Risk threshold is SOFA≥10)" if sofa_score < 10 else ""
    st.error(
        f"⚠️ **HIGH RISK PATIENT DETECTED** — Immediate clinical attention required."
        f"{extra} Review AI assessment below and initiate appropriate protocols."
    )
elif sofa_score >= 5:
    st.warning(
        "⚠️ **MODERATE RISK** — Patient requires close monitoring. Review assessment below."
    )

# -- 8b. SAVE PREDICTION HISTORY --
_hist_row = pd.DataFrame([{
    "Timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
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
if os.path.exists(HISTORY_FILE):
    _existing = pd.read_csv(HISTORY_FILE)
    _hist_df  = pd.concat([_existing, _hist_row], ignore_index=True).tail(50)
else:
    _hist_df = _hist_row
_hist_df.to_csv(HISTORY_FILE, index=False)

# -- 9. SHAP --
shap_vals = None
top_shap  = None
clinical_explanations = []
key_risks = []

if explainer is not None:
    with st.spinner("Computing SHAP feature importance..."):
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
    # Use ORIGINAL (unscaled) feature values for display so clinicians see
    # meaningful numbers (e.g. SpO₂=88.0% not z-score=-1.8).
    # final_df contains the pre-scaling values; final_scaled is the z-score version.
    original_values = final_df.iloc[0].values  # shape (618,)

    shap_df = pd.DataFrame({
        "feature":        expected_cols,
        "original_value": original_values,   # clinical values (bpm, %, °C, etc.)
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

    seen = set()
    for _, r in top_shap.iterrows():
        for k, v in RISK_MAP.items():
            if k.lower() in r["feature"].lower() and v not in seen:
                key_risks.append(v)
                seen.add(v)

# -- 10. TREND ANALYSIS --
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

# Build urgency header and instructions specific to risk level
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
with st.spinner("Generating AI clinical assessment (3 independent responses for reliability check)..."):
    try:
        responses       = get_multiple_llm_responses(GROQ_API_KEY, final_prompt, n=3)
        main_response   = responses[0]
        consistency     = compute_consistency(responses)
        llm_ok          = True
    except Exception as e:
        responses     = []
        main_response = f"⚠️ LLM unavailable: {e}"
        consistency   = 0.0
        llm_ok        = False

# =============================================================
# DISPLAY — THREE TABS
# =============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Assessment",
    "🔍 Explainability",
    "🧠 AI Clinical Report",
    "🔒 Federated Learning"
])

# ---- TAB 1: RISK ASSESSMENT ----
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("SOFA Score", f"{sofa_score:.1f} / 24")
    with c2:
        st.metric("Risk Level", f"{risk_icon} {risk_text}")
    with c3:
        st.metric("Severity", f"{int((sofa_score / 24) * 100)}%")
    with c4:
        st.metric("Readings in Window", f"{len(vitals_df)} / 20")

    st.progress(min(sofa_score / 24, 1.0))

    # Model accuracy context — lets the clinician calibrate trust in the prediction
    st.caption(
        f"📊 **Model accuracy on held-out ICU data** — "
        f"MAE: {training_meta['final_mae']:.3f} SOFA pts  |  "
        f"R²: {training_meta['final_r2']:.3f}  |  "
        f"Trained on {training_meta['train_samples']:,} samples via "
        f"{training_meta['num_rounds']} FL rounds across {training_meta['hospitals']} hospitals"
    )

    # High Risk Advisory — shown only when alert fires
    if sofa_score >= ALERT_THRESHOLD:
        with st.expander("🔴 High Risk Clinical Advisory — read before acting", expanded=True):
            st.markdown(f"""
**Prediction uncertainty for this risk level:**
The federated model has a typical error of **±3.9 SOFA points** for High Risk patients.
A predicted SOFA of **{sofa_score:.1f}** could reflect a true SOFA of **{max(0, sofa_score-4.0):.0f}–{min(24, sofa_score+4.0):.0f}**.
The alert fires at predicted SOFA ≥ {ALERT_THRESHOLD:.0f} to compensate for systematic under-prediction.

**Why the alert threshold is ≥ {ALERT_THRESHOLD:.0f} (not ≥ 10):**
The model under-predicts severe cases by ~2–3 SOFA points due to limited high-risk training data
(only 6% of ICU stays have SOFA ≥ 10). Lowering the threshold doubles high-risk recall
(28% → 52%) while keeping false alarms on stable patients below 1.5%.

**Key clinical guidance:**
- Use the AI report (Tab 3) as supporting context — not as a diagnostic tool
- Prioritise the immediate actions listed by the LLM
- Reassess within 30 minutes and re-run prediction to track SOFA trajectory
""")

    st.divider()

    st.plotly_chart(build_trend_chart(vitals_df), use_container_width=True)
    st.divider()

    st.subheader("Trend Summary")
    col_left, col_right = st.columns(2)
    for i, line in enumerate(trend_lines):
        (col_left if i < 4 else col_right).text(f"  {line}")

    # ---- PREDICTION HISTORY ----
    if os.path.exists(HISTORY_FILE):
        _ph = pd.read_csv(HISTORY_FILE)
        if len(_ph) > 1:
            with st.expander(f"📋 Prediction History  ({len(_ph)} readings, latest first)"):
                st.caption(
                    "Each row = one 'Run Prediction' click. "
                    f"Alert threshold: SOFA ≥ {ALERT_THRESHOLD:.0f}. "
                    "Keeps last 50 predictions."
                )
                # Show most recent first, colour Risk column
                _ph_display = _ph.iloc[::-1].reset_index(drop=True)
                st.dataframe(_ph_display, use_container_width=True, hide_index=True)

# ---- TAB 2: EXPLAINABILITY ----
with tab2:
    if top_shap is not None:
        st.subheader("Top Contributing Features (SHAP)")

        # Build a display table with original clinical values (not z-scores)
        display_rows = []
        for _, r in top_shap.iterrows():
            feat = r["feature"]
            orig = r["original_value"]
            if feat in VITAL_UNITS:
                label, unit = VITAL_UNITS[feat]
                fmt = ".0f" if unit in ("/4", "/10") else ".1f"
                val_display = f"{orig:{fmt}}{unit}"
            else:
                # TF-IDF feature — show presence/absence (value 0 = absent)
                val_display = "detected" if orig > 0 else "absent"
            display_rows.append({
                "Feature":     label if feat in VITAL_UNITS else feat,
                "Value":       val_display,
                "SHAP Impact": round(r["abs_impact"], 4),
                "Direction":   "↑ Increases risk" if r["impact"] > 0 else "↓ Reduces risk"
            })

        st.dataframe(
            pd.DataFrame(display_rows),
            use_container_width=True,
            hide_index=True
        )
        st.divider()

        st.subheader("Clinical Interpretations")
        for exp in clinical_explanations:
            st.write(f"• {exp}")
        st.divider()

        st.subheader("Key Risk Factors")
        if key_risks:
            cols = st.columns(min(len(key_risks), 3))
            for i, risk in enumerate(key_risks):
                cols[i % 3].warning(f"⚠️ {risk}")
        else:
            st.info("No specific clinical risk factors identified from model features.")

        st.divider()

        # SHAP counterintuitive result disclaimer
        with st.expander("ℹ️ About SHAP — understanding counterintuitive results"):
            st.markdown("""
**What SHAP values represent:**
SHAP (SHapley Additive exPlanations) quantifies how much each feature
*shifted* this patient's predicted SOFA away from the model's baseline.
↑ means the feature pushed the prediction toward a higher (worse) SOFA score;
↓ means it pushed it lower (toward better).

**Why some results may seem counterintuitive:**
SHAP explains what the *model* learned — not established clinical logic.
MIMIC-III training data contains correlations that can differ from clinical intuition:

- A high **Stress Score** might appear as "reducing risk" because, in the training
  data, responsive/agitated patients often had lower SOFA than unresponsive ones
  (consciousness implies less organ failure).
- Clinical notes containing certain terms might be associated with lower SOFA in
  the training cohort for reasons unrelated to the term itself (e.g., documentation
  patterns, patient selection bias).

**How to use SHAP correctly:**
- Use the feature importance ranking to understand *which signals* drove this prediction.
- Do not interpret individual SHAP directions as clinical ground truth.
- The LLM report in Tab 3 integrates all information including trends and clinical notes
  to provide holistic reasoning beyond what SHAP alone shows.
""")

    else:
        st.info(
            "SHAP background data not found. Run `python train_federated.py` once to "
            "generate SHAP background samples, then restart the app.",
            icon="ℹ️"
        )

# ---- TAB 3: AI CLINICAL REPORT ----
with tab3:
    # Consistency score header
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.subheader("AI Clinical Assessment")
    with c2:
        st.metric("Consistency Score", f"{consistency:.2f}")
    with c3:
        if consistency >= 0.85:
            st.metric("Reliability", "High ✅")
        elif consistency >= 0.65:
            st.metric("Reliability", "Moderate ⚠️")
        else:
            st.metric("Reliability", "Low ❌")

    if consistency >= 0.85:
        st.success("✅ **High Reliability** — All 3 LLM responses are consistent with each other.")
    elif consistency >= 0.65:
        st.warning("⚠️ **Moderate Reliability** — Some variation across responses. Review with care.")
    elif llm_ok:
        st.error("❌ **Low Reliability** — Significant variation across responses. Use clinical judgment.")

    st.divider()
    st.write(main_response)

    if responses:
        with st.expander("🔎 View all 3 independent LLM responses"):
            for i, resp in enumerate(responses):
                st.markdown(f"**Response {i + 1}:**")
                st.write(resp)
                if i < len(responses) - 1:
                    st.divider()

    st.divider()
    st.info(
        "⚠️ **Clinical Disclaimer** — This system is a decision support tool only. "
        "It does not diagnose disease or replace the clinical judgment of qualified healthcare professionals. "
        "All AI-generated outputs must be reviewed by a licensed clinician before any clinical action is taken."
    )

# ---- TAB 4: FEDERATED LEARNING INFO ----
with tab4:
    m = training_meta  # shorthand

    st.subheader("Federated Learning — How This Model Was Trained")

    # FL process diagram
    st.markdown("""
```
┌──────────────────────────────────────────────────────────────────┐
│                  Federated Learning Protocol                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│           ┌──────────────────────────────────┐                   │
│           │         Global FL Server          │                   │
│           │    (Flower · FedAvg aggregation)  │                   │
│           └───────────┬──────────────────────┘                   │
│                       │  (1) Share global weights                 │
│           ┌───────────┼──────────────────────┐                   │
│           ↓           ↓                      ↓                   │
│    ┌─────────────┐ ┌────────────┐ ┌──────────────────┐           │
│    │ Hospital 0  │ │ Hospital 1 │ │   Hospital 2      │           │
│    │ General ICU │ │ Mixed ICU  │ │ Cardiac/Trauma    │           │
│    │ ~15,889 pts │ │ ~15,890 pts│ │   ~16,372 pts     │           │
│    │             │ │            │ │                   │           │
│    │  (2) Train locally on PRIVATE patient data       │           │
│    └──────┬──────┘ └─────┬──────┘ └─────────┬─────────┘           │
│           │              │                  │                    │
│           └──────────────┴──────────────────┘                   │
│                       │  (3) Send ONLY model weights             │
│                       │      (zero patient data shared)          │
│           ┌───────────↓──────────────────────┐                   │
│           │    FedAvg: average all weights    │                   │
│           │    → improved global model        │                   │
│           └──────────────────────────────────┘                   │
│                                                                    │
│                  Repeat for 20 rounds                             │
└──────────────────────────────────────────────────────────────────┘
```
""")

    st.divider()

    # Training configuration metrics
    st.subheader("Training Configuration")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("FL Rounds",      str(m["num_rounds"]))
    col2.metric("Hospitals",      str(m["hospitals"]))
    col3.metric("Epochs / Round", str(m["epochs_per_round"]))
    col4.metric("Aggregation",    m["aggregation"])

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Training Samples", f"{m['train_samples']:,}")
    col6.metric("Test Samples",     f"{m['test_samples']:,}")
    col7.metric("Best Round",       str(m["best_round"]))
    col8.metric("Split Type",       m["split_type"])

    st.divider()

    # Model performance
    st.subheader("Global Model Performance")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("MAE",  f"{m['final_mae']:.3f} SOFA pts")
    pc2.metric("R²",   f"{m['final_r2']:.3f}")
    pc3.metric("Pred Min", f"{m['pred_range_min']:.1f}")
    pc4.metric("Pred Max", f"{m['pred_range_max']:.1f}")

    st.divider()

    # Model architecture
    st.subheader("Model Architecture  (PyTorch DNN)")
    st.code(
        f"Input ({m['input_features']} features: trend vitals + latest vitals + CV + TF-IDF)\n"
        "  ↓\n"
        "Linear(618, 256) → ReLU\n"
        "  ↓\n"
        "Linear(256, 128) → ReLU\n"
        "  ↓\n"
        "Linear(128,  64) → ReLU\n"
        "  ↓\n"
        "Linear( 64,   1)\n"
        "  ↓\n"
        "SOFA Score (0–24)"
    )

    st.divider()

    # Differential Privacy status
    st.subheader("Differential Privacy")
    if m.get("differential_privacy"):
        st.success(
            f"✅ **Differential Privacy ENABLED**  "
            f"(σ={m['dp_sigma']}, S={m['dp_sensitivity']}, "
            f"ε≈{m['dp_epsilon']}, δ={m['dp_delta']})"
        )
        st.markdown("""
**How it works in this system:**
1. Each hospital computes its model update = `local_weights − global_weights`
2. The update L2 norm is clipped to a max sensitivity (`S`) to bound any single patient's influence
3. Gaussian noise `N(0, (σ·S)²)` is added to every weight before sending to the server
4. The server aggregates the noisy weights — it cannot recover any individual patient's contribution

**Privacy guarantee:** This provides approximate **(ε, δ)-Differential Privacy** per client per round.
A lower σ gives stronger privacy (larger ε means weaker privacy).
""")
    else:
        st.info(
            "ℹ️ **Differential Privacy: Disabled** — the model was trained with plain FedAvg "
            "(no noise added). To enable DP, set `USE_DP = True` in `train_federated.py` "
            "and retrain."
        )
        st.markdown("""
**What DP would add:**
- Clips model updates to max L2 norm (sensitivity clipping)
- Adds calibrated Gaussian noise before transmitting weights to server
- Provides a formal (ε, δ)-DP guarantee per hospital per round
- Trade-off: stronger privacy → more noise → slightly lower model accuracy
""")

    st.divider()

    # Feature breakdown
    st.subheader("Feature Vector Breakdown  (618 total)")
    feat_data = {
        "Modality":        ["Trend Vitals",    "Latest Vitals",  "Computer Vision", "Clinical NLP (TF-IDF)"],
        "Features":        [9,                  7,                2,                 600],
        "Examples":        [
            "HR_mean, HR_std, SpO₂_min, MAP_mean …",
            "latest_HR, latest_SpO₂, latest_MAP …",
            "GCS Eye Opening (1–4), Stress Score (0–10)",
            "hypotension, respiratory, intubated … (bigrams)"
        ]
    }
    st.dataframe(feat_data, use_container_width=True, hide_index=True)

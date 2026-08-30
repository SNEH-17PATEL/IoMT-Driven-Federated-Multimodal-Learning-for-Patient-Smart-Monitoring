"""
ICU Clinical Decision Support System
Multimodal Intelligence System with Federated Learning — Continuous Monitoring
"""

import os
import re
import json
import time
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
# GLOBAL CSS — ICU monitoring aesthetic
# =============================================================
st.markdown("""
<style>
/* ── Animations ── */
@keyframes livePulse {
    0%   { box-shadow: 0 0 0 0 rgba(255,51,51,0.7); }
    70%  { box-shadow: 0 0 0 10px rgba(255,51,51,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,51,51,0); }
}
@keyframes blink {
    0%,100% { opacity:1; }
    50%      { opacity:0.35; }
}
@keyframes slideIn {
    from { opacity:0; transform:translateY(-8px); }
    to   { opacity:1; transform:translateY(0); }
}

/* ── LIVE badge ── */
.live-dot {
    display:inline-block; width:11px; height:11px;
    background:#ff3333; border-radius:50%;
    animation: livePulse 1.6s infinite;
    vertical-align:middle; margin-right:6px;
}
.live-badge {
    background:linear-gradient(90deg,#ff3333,#cc0000);
    color:white; font-size:12px; font-weight:800;
    padding:3px 10px; border-radius:20px;
    letter-spacing:1.5px; vertical-align:middle; margin-right:8px;
    animation: blink 2s infinite;
}

/* ── Monitoring banner ── */
.monitor-banner {
    background:linear-gradient(135deg,#0f2027 0%,#203a43 50%,#2c5364 100%);
    color:white; padding:14px 22px; border-radius:12px;
    border-left:6px solid #00d2ff;
    animation:slideIn 0.3s ease; margin-bottom:4px;
}
.monitor-banner b { color:#00d2ff; }

/* ── Patient selector cards ── */
.pcard {
    border-radius:16px; padding:22px 20px; margin:6px 0;
    animation:slideIn 0.4s ease; transition:transform 0.15s;
}
.pcard:hover { transform:translateY(-2px); }
.pcard-low  { border:3px solid #28a745; background:linear-gradient(140deg,#e8fce8,#c8f7d0); }
.pcard-mod  { border:3px solid #f0a500; background:linear-gradient(140deg,#fffbf0,#fff0b3); }
.pcard-high { border:3px solid #dc3545; background:linear-gradient(140deg,#fff5f5,#ffd0d0); }
.pcard h3   { margin:0 0 6px 0; font-size:20px; }
.pcard p    { margin:4px 0; color:#444; font-size:13px; }
.pcard .tag { display:inline-block; border-radius:20px; padding:2px 10px;
              font-size:11px; font-weight:700; letter-spacing:0.5px; }
.tag-low    { background:#d4edda; color:#155724; }
.tag-mod    { background:#fff3cd; color:#856404; }
.tag-high   { background:#f8d7da; color:#721c24; }

/* ── Vital sign cards ── */
.vcard {
    border-radius:12px; padding:14px 10px;
    text-align:center; min-height:120px;
    display:flex; flex-direction:column;
    justify-content:space-between; margin:3px;
    transition: transform 0.1s;
}
.vcard:hover { transform:scale(1.02); }
.v-ok   { background:linear-gradient(145deg,#e8fce8,#d4f8d4); border:2px solid #4caf50; }
.v-bad  { background:linear-gradient(145deg,#fee8e8,#fdd0d0); border:2px solid #f44336; }
.v-warn { background:linear-gradient(145deg,#fff8e1,#ffe8a0); border:2px solid #ff9800; }
.vnum   { font-size:32px; font-weight:900; color:#1a1a2e; line-height:1; }
.vlabel { font-size:10px; font-weight:700; color:#555;
          text-transform:uppercase; letter-spacing:0.6px; }
.vunit  { font-size:10px; color:#888; }
.vstatus-ok  { font-size:11px; font-weight:700; color:#2e7d32; }
.vstatus-bad { font-size:11px; font-weight:700; color:#c62828; }
.vrange { font-size:9px; color:#999; }

/* ── SOFA gauge box ── */
.sofa-gauge {
    border-radius:20px; padding:28px 20px; text-align:center;
    box-shadow:0 6px 24px rgba(0,0,0,0.12);
    animation:slideIn 0.4s ease;
}
.sofa-num { font-size:80px; font-weight:900; line-height:1; }
.sofa-denom { font-size:22px; font-weight:600; opacity:0.7; }
.sofa-risk  { font-size:18px; font-weight:800; margin-top:6px; letter-spacing:0.3px; }
.sofa-sev   { font-size:12px; opacity:0.75; margin-top:2px; }

/* ── Countdown / next reading bar ── */
.cdbar {
    background:linear-gradient(90deg,#0f2027,#2c5364);
    color:white; border-radius:12px; padding:12px 20px;
    margin-top:16px; border:1.5px solid #00d2ff;
    display:flex; align-items:center; gap:12px; font-size:13px;
}

/* ── Alert banner override ── */
div[data-testid="stAlert"] > div {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================
# CONSTANTS
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

# =============================================================
# PATIENT CONFIGURATION
# =============================================================
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
    },
}

# =============================================================
# LOAD ARTIFACTS (cached — runs only once)
# =============================================================
@st.cache_resource
def load_artifacts():
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
# SESSION STATE
# =============================================================
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None
if "row_indices" not in st.session_state:
    st.session_state.row_indices = {1: 0, 2: 0, 3: 0}

# =============================================================
# GROQ API KEY
# =============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# =============================================================
# SIDEBAR
# =============================================================
st.sidebar.title("🏥 ICU Monitor")
st.sidebar.caption("Continuous Patient Monitoring")
st.sidebar.divider()

if st.session_state.selected_patient is not None:
    pcfg_side = PATIENTS[st.session_state.selected_patient]
    st.sidebar.info(
        f"📡 **Monitoring:**\n{pcfg_side['icon']} {pcfg_side['name']}"
    )
    if st.sidebar.button("⏹️ Stop Monitoring", use_container_width=True, type="secondary"):
        st.session_state.selected_patient = None
        st.rerun()
    st.sidebar.divider()

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
    "Multimodal Intelligence System — Continuous Monitoring | "
    "Federated Learning | SHAP Explainability | LLM Self-Consistency"
)

# =============================================================
# PATIENT SELECTOR  (shown when no patient is being monitored)
# =============================================================
if st.session_state.selected_patient is None:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 24px 0;">
        <div style="font-size:38px; font-weight:900; color:#1a1a2e; letter-spacing:-1px;">
            🏥 Select Patient to Monitor
        </div>
        <div style="font-size:15px; color:#555; margin-top:8px;">
            AI reads vitals every <strong>30 seconds</strong> — generating SOFA predictions,
            SHAP explainability & clinical AI reports automatically.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _CARD_CFG = [
        (1, "pcard-low",  "tag-low",  "LOW RISK",      "🟢",
         "0 – 4", "Post-surgical recovery. Alert and oriented. No active infection."),
        (2, "pcard-mod",  "tag-mod",  "MODERATE RISK", "🟡",
         "5 – 9", "Community-acquired pneumonia. On supplemental O₂ 4L/min. Elevated WBC."),
        (3, "pcard-high", "tag-high", "HIGH RISK",      "🔴",
         "≥ 10",  "Septic shock. Intubated & ventilated. Vasopressors. Multi-organ failure."),
    ]

    col1, col2, col3 = st.columns(3)
    _cols = [col1, col2, col3]
    for pid, card_cls, tag_cls, risk_label_txt, icon, sofa_range, desc in _CARD_CFG:
        with _cols[pid - 1]:
            pcfg = PATIENTS[pid]
            st.markdown(f"""
            <div class="pcard {card_cls}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:28px;">{icon}</span>
                    <span class="tag {tag_cls}">{risk_label_txt}</span>
                </div>
                <h3 style="margin:10px 0 4px 0;">Patient {pid}</h3>
                <p style="font-size:14px;font-weight:600;color:#333;">{pcfg['description'].split(' — ')[0]}</p>
                <p style="font-size:12px;color:#666;margin-top:4px;">{desc}</p>
                <div style="margin-top:12px;padding:8px;background:rgba(255,255,255,0.6);
                            border-radius:8px;font-size:12px;">
                    <b>Expected SOFA:</b> {sofa_range} &nbsp;|&nbsp;
                    <b>30 readings</b> × 10-min intervals
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(
                f"▶  Start Monitoring Patient {pid}",
                key=f"sel_{pid}",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.selected_patient = pid
                st.session_state.row_indices[pid] = 0
                st.rerun()
    st.stop()

# =============================================================
# GROQ KEY CHECK
# =============================================================
if not GROQ_API_KEY:
    st.error(
        "⛔ Groq API key required. Add `GROQ_API_KEY=your_key` to the `.env` file "
        "in the `icu_monitor/` directory and restart the app."
    )
    st.stop()

# =============================================================
# READ CURRENT ROW FROM PATIENT CSV
# =============================================================
patient_id  = st.session_state.selected_patient
patient_cfg = PATIENTS[patient_id]
row_idx     = st.session_state.row_indices[patient_id]

patient_csv = DATA_PATH + patient_cfg["file"]
patient_df  = pd.read_csv(patient_csv)
total_rows  = len(patient_df)
cur_idx     = row_idx % total_rows
current_row = patient_df.iloc[cur_idx]

# Extract all inputs from the CSV row
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

# Per-patient file paths (keeps each patient's history separate)
VITALS_FILE  = MODEL_PATH + patient_cfg["vitals_file"]
HISTORY_FILE = MODEL_PATH + patient_cfg["hist_file"]

# Seed the patient's vitals file from CSV data if it doesn't exist yet
if not os.path.exists(VITALS_FILE):
    os.makedirs(os.path.dirname(VITALS_FILE), exist_ok=True)
    _seed = patient_df[["HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"]].head(20).copy()
    _ts = [
        f"2026-08-29 {(8 + i // 6):02d}:{(i % 6) * 10:02d}:00"
        for i in range(len(_seed))
    ]
    _seed.insert(0, "time", _ts)
    _seed.to_csv(VITALS_FILE, index=False)

# =============================================================
# MONITORING STATUS BAR
# =============================================================
cycle_num = row_idx // total_rows + 1
_now = datetime.now().strftime("%H:%M:%S")
_risk_colors = {1: "#28a745", 2: "#f0a500", 3: "#dc3545"}
_accent = _risk_colors[patient_id]

st.markdown(f"""
<div class="monitor-banner">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
            <span class="live-dot"></span>
            <span class="live-badge">LIVE</span>
            <span style="font-size:18px; font-weight:800; letter-spacing:0.2px;">
                {patient_cfg['icon']} {patient_cfg['name']}
            </span>
            <span style="font-size:12px; color:#99b8cc; padding:2px 10px;
                         background:rgba(255,255,255,0.08); border-radius:20px;">
                {patient_cfg['description']}
            </span>
        </div>
        <div style="display:flex; gap:24px; font-size:12px; color:#ccc;">
            <div style="text-align:center;">
                <div style="color:#7fb3c8; font-size:10px; text-transform:uppercase; letter-spacing:0.8px;">Reading</div>
                <div style="color:white; font-weight:800; font-size:20px; line-height:1.1;">{cur_idx + 1}<span style="font-size:13px;color:#7fb3c8;">/{total_rows}</span></div>
            </div>
            <div style="text-align:center;">
                <div style="color:#7fb3c8; font-size:10px; text-transform:uppercase; letter-spacing:0.8px;">Cycle</div>
                <div style="color:white; font-weight:800; font-size:20px; line-height:1.1;">#{cycle_num}</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#7fb3c8; font-size:10px; text-transform:uppercase; letter-spacing:0.8px;">Time</div>
                <div style="color:#00d2ff; font-weight:800; font-size:20px; font-family:monospace; line-height:1.1;">{_now}</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TOP COUNTDOWN PLACEHOLDER — filled by the loop at the bottom of the page
_top_cd  = st.empty()   # countdown bar HTML
_top_bar = st.empty()   # progress bar
# Show an initial "ready" state while the page loads
_top_cd.markdown(f"""
<div class="cdbar" style="margin-top:4px; margin-bottom:2px;">
    <span class="live-dot"></span>
    <span style="font-weight:700; letter-spacing:0.5px;">CONTINUOUS MONITORING</span>
    <span style="color:#4a7a8a;">|</span>
    <span>{patient_cfg['icon']} {patient_cfg['name']}</span>
    <span style="color:#4a7a8a;">|</span>
    <span>Reading <b style="color:#00d2ff;">{cur_idx + 1}/{total_rows}</b></span>
    <span style="color:#4a7a8a;">|</span>
    <span style="color:#aaa; font-style:italic;">⚙ Processing data…</span>
</div>
""", unsafe_allow_html=True)
_top_bar.progress(0.0)

# =============================================================
# LIVE VITALS DISPLAY — ICU monitor style
# =============================================================
st.markdown("<div style='font-size:17px;font-weight:700;margin:14px 0 8px 0;'>📊 Latest Vitals — Current Reading</div>",
            unsafe_allow_html=True)

def _vcard(label, value, fmt, unit, lo, hi):
    """Render a colored vital sign card as HTML."""
    val_fmt = f"{value:{fmt}}"
    if lo <= value <= hi:
        cls = "v-ok";  s_cls = "vstatus-ok";  s_txt = "✓ Normal"
        rng = f"Normal: {lo}–{hi} {unit}"
    elif value < lo:
        cls = "v-bad"; s_cls = "vstatus-bad"; s_txt = "⚠ Below Normal"
        rng = f"Normal: {lo}–{hi} {unit}"
    else:
        cls = "v-bad"; s_cls = "vstatus-bad"; s_txt = "⚠ Above Normal"
        rng = f"Normal: {lo}–{hi} {unit}"
    return f"""
    <div class="vcard {cls}">
        <div class="vlabel">{label}</div>
        <div class="vnum">{val_fmt}</div>
        <div class="vunit">{unit}</div>
        <div class="{s_cls}">{s_txt}</div>
        <div class="vrange">{rng}</div>
    </div>"""

gcs_labels = {4: "Spontaneous", 3: "To Voice", 2: "To Pain", 1: "No Response"}
_lo_hi = NORMAL_RANGES

row1_html = "".join([
    _vcard("❤️ Heart Rate",      HR,   ".0f", "bpm",    *_lo_hi["HR"]),
    _vcard("🫁 Resp. Rate",       RR,   ".0f", "br/min", *_lo_hi["RR"]),
    _vcard("💧 SpO₂",            SpO2, ".1f", "%",      _lo_hi["SpO2"][0], _lo_hi["SpO2"][1]),
    _vcard("🌡️ Temperature",     Temp, ".1f", "°C",     *_lo_hi["Temp"]),
])
row2_html = "".join([
    _vcard("🩸 Systolic BP",     SBP,  ".0f", "mmHg",   *_lo_hi["SBP"]),
    _vcard("🩸 Diastolic BP",    DBP,  ".0f", "mmHg",   *_lo_hi["DBP"]),
    _vcard("📉 Mean Art. Press.", MAP,  ".0f", "mmHg",   *_lo_hi["MAP"]),
    f"""<div class="vcard v-ok">
        <div class="vlabel">🧠 GCS Eye Opening</div>
        <div class="vnum" style="font-size:22px;">{GCS_eye}<span style="font-size:14px;font-weight:500;">/4</span></div>
        <div class="vunit">{gcs_labels.get(GCS_eye,'?')}</div>
        <div class="vstatus-ok">Neurological</div>
        <div class="vrange">Stress: {stress}/10</div>
    </div>""",
])

st.markdown(
    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">{row1_html}</div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:6px;">{row2_html}</div>',
    unsafe_allow_html=True
)

with st.expander("📋 Clinical Notes", expanded=False):
    st.write(clinical_note)
    st.caption(f"Stress Score: {stress}/10")

st.divider()

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
    Layout: 2 rows × 4 cols
      Row 1: Heart Rate | Respiratory Rate | SpO₂ | Temperature
      Row 2: Systolic BP | Diastolic BP | MAP | (empty)
    """
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

        fig.add_hrect(
            y0=lo, y1=hi,
            fillcolor="rgba(80,200,120,0.15)",
            line_width=0,
            row=row, col=c
        )

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

CLINICAL_KEYWORDS = [
    "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP", "GCS", "stress",
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
final_scaled = scaler.transform(final_df)

# -- 7. PREDICT --
X_tensor = torch.tensor(final_scaled, dtype=torch.float32)
model.eval()
with torch.no_grad():
    raw_pred = model(X_tensor).numpy().flatten()[0]

sofa_score = float(np.clip(raw_pred, 0, 24))
risk_text, risk_icon, risk_color = risk_label(sofa_score)

# -- 8. ALERT BANNER --
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
        responses     = get_multiple_llm_responses(GROQ_API_KEY, final_prompt, n=3)
        main_response = responses[0]
        consistency   = compute_consistency(responses)
        llm_ok        = True
    except Exception as e:
        responses     = []
        main_response = f"⚠️ LLM unavailable: {e}"
        consistency   = 0.0
        llm_ok        = False

# =============================================================
# DISPLAY — FOUR TABS
# =============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Assessment",
    "🔍 Explainability",
    "🧠 AI Clinical Report",
    "🔒 Federated Learning"
])

# ---- TAB 1: RISK ASSESSMENT ----
with tab1:
    # ── Colour palette for current SOFA ──
    _sbg  = "#e8fce8" if sofa_score < 5 else "#fffbf0" if sofa_score < 10 else "#fff0f0"
    _sbrd = "#28a745" if sofa_score < 5 else "#f0a500" if sofa_score < 10 else "#dc3545"
    _stxt = "#1b5e20" if sofa_score < 5 else "#7a4100" if sofa_score < 10 else "#b71c1c"
    _m    = training_meta

    # ── Two-column header: SOFA gauge  |  Severity bar + model stats ──
    col_gauge, col_right = st.columns([2, 3])

    with col_gauge:
        st.markdown(f"""
        <div class="sofa-gauge" style="background:linear-gradient(135deg,{_sbg},white);
             border:4px solid {_sbrd}; height:220px; justify-content:center;">
            <div style="font-size:11px;font-weight:700;color:{_sbrd};letter-spacing:2.5px;
                        text-transform:uppercase;">Predicted SOFA Score</div>
            <div class="sofa-num" style="color:{_stxt};font-size:88px;">{sofa_score:.1f}
                <span style="font-size:24px;font-weight:600;color:{_stxt};opacity:0.7;">/24</span>
            </div>
            <div style="font-size:17px;font-weight:800;color:{_sbrd};margin-top:2px;">
                {risk_icon} {risk_text}
            </div>
            <div style="font-size:12px;color:{_stxt};opacity:0.75;margin-top:6px;">
                Severity: {int((sofa_score/24)*100)}%
                &nbsp;·&nbsp; Window: {len(vitals_df)}/20 readings
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # ── SOFA Severity Scale (SVG) ──
        _pct = sofa_score / 24        # 0.0–1.0
        _low_w  = 4/24 * 300          # 0–4 green  → 50 px
        _mod_w  = 5/24 * 300          # 5–9 amber  → 62.5 px
        _hi_w   = 300 - _low_w - _mod_w
        _marker = _pct * 300          # marker x-position
        st.markdown(f"""
        <div style="background:#f8fafc;border-radius:14px;padding:18px 20px;
                    border:1.5px solid #e2e8f0;margin-bottom:10px;">
            <div style="font-size:11px;font-weight:700;color:#64748b;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:10px;">
                📊 SOFA Severity Scale — Current Reading
            </div>
            <svg width="100%" viewBox="0 0 300 52" xmlns="http://www.w3.org/2000/svg">
                <!-- Zone bars -->
                <rect x="0"   y="14" width="{_low_w:.1f}" height="22" fill="#28a745" rx="5"/>
                <rect x="{_low_w:.1f}" y="14" width="{_mod_w:.1f}" height="22" fill="#f0a500"/>
                <rect x="{_low_w+_mod_w:.1f}" y="14" width="{_hi_w:.1f}" height="22" fill="#dc3545" rx="5"/>
                <!-- Marker needle -->
                <polygon points="{_marker:.1f},8 {_marker-5:.1f},14 {_marker+5:.1f},14"
                         fill="white" stroke="#1a1a2e" stroke-width="1.2"/>
                <rect x="{_marker-2.5:.1f}" y="12" width="5" height="26" fill="white"
                      rx="2.5" stroke="#1a1a2e" stroke-width="1"/>
                <!-- Zone labels -->
                <text x="4"   y="50" font-size="9" fill="#28a745" font-weight="600">LOW (0–4)</text>
                <text x="{_low_w+4:.1f}" y="50" font-size="9" fill="#e07800" font-weight="600">MOD (5–9)</text>
                <text x="{_low_w+_mod_w+4:.1f}" y="50" font-size="9" fill="#dc3545" font-weight="600">HIGH (≥10)</text>
                <text x="295" y="50" font-size="9" fill="#888" text-anchor="end">24</text>
                <!-- Current SOFA label -->
                <text x="{min(max(_marker, 20), 280):.1f}" y="8" font-size="9" fill="{_sbrd}"
                      text-anchor="middle" font-weight="700">▼ {sofa_score:.1f}</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)

        # ── Model accuracy mini-tiles (2×2 grid) ──
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;">
            <div style="background:linear-gradient(135deg,#e8f4fd,#d0eaf8);border-radius:10px;
                        padding:12px 8px;text-align:center;border:1px solid #90cdf4;">
                <div style="font-size:9px;font-weight:700;color:#2b6cb0;letter-spacing:0.7px;
                            text-transform:uppercase;">Model MAE</div>
                <div style="font-size:22px;font-weight:900;color:#1a365d;line-height:1.1;">
                    {_m['final_mae']:.2f}</div>
                <div style="font-size:9px;color:#4a90d9;">SOFA pts</div>
            </div>
            <div style="background:linear-gradient(135deg,#f0fff4,#c6f6d5);border-radius:10px;
                        padding:12px 8px;text-align:center;border:1px solid #9ae6b4;">
                <div style="font-size:9px;font-weight:700;color:#276749;letter-spacing:0.7px;
                            text-transform:uppercase;">R² Score</div>
                <div style="font-size:22px;font-weight:900;color:#1a4731;line-height:1.1;">
                    {_m['final_r2']:.3f}</div>
                <div style="font-size:9px;color:#38a169;">variance</div>
            </div>
            <div style="background:linear-gradient(135deg,#fffaf0,#feebc8);border-radius:10px;
                        padding:12px 8px;text-align:center;border:1px solid #f6ad55;">
                <div style="font-size:9px;font-weight:700;color:#7b341e;letter-spacing:0.7px;
                            text-transform:uppercase;">Trained on</div>
                <div style="font-size:22px;font-weight:900;color:#7b341e;line-height:1.1;">
                    {_m['train_samples']//1000}K</div>
                <div style="font-size:9px;color:#c05621;">patients</div>
            </div>
            <div style="background:linear-gradient(135deg,#faf5ff,#e9d8fd);border-radius:10px;
                        padding:12px 8px;text-align:center;border:1px solid #d6bcfa;">
                <div style="font-size:9px;font-weight:700;color:#553c9a;letter-spacing:0.7px;
                            text-transform:uppercase;">FL Rounds</div>
                <div style="font-size:22px;font-weight:900;color:#44337a;line-height:1.1;">
                    {_m['num_rounds']}</div>
                <div style="font-size:9px;color:#805ad5;">{_m['hospitals']} hospitals</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── High Risk Advisory ──
    if sofa_score >= ALERT_THRESHOLD:
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        with st.expander("🔴 High Risk Clinical Advisory — read before acting", expanded=True):
            st.markdown(f"""
**Prediction uncertainty:** The federated model has a typical error of **±3.9 SOFA points**
for High Risk patients. A predicted SOFA of **{sofa_score:.1f}** could reflect a true SOFA of
**{max(0, sofa_score-4.0):.0f}–{min(24, sofa_score+4.0):.0f}**.

**Alert threshold = {ALERT_THRESHOLD:.0f} (not 10):** Model under-predicts severe cases by ~2–3 SOFA
points (only 6% of training data is High Risk). Threshold = 8 doubles recall (28% → 52%),
false alarm rate on stable patients = 1.4%.

**Clinical guidance:** AI report (Tab 3) → supporting context only.
System re-assesses every 30 seconds — watch the SOFA trajectory over readings.
""")

    st.divider()

    # ── Vital Sign Trend Charts ──
    st.plotly_chart(build_trend_chart(vitals_df), use_container_width=True)
    st.divider()

    # ── Trend Summary ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#1a1a2e;margin-bottom:10px;'>📈 Vital Sign Trends</div>",
                unsafe_allow_html=True)
    _dir_style = {
        "increasing": ("🔺", "#dc3545", "#fff0f0"),
        "decreasing": ("🔻", "#28a745", "#f0fff4"),
        "stable":     ("➡", "#0066cc", "#f0f4ff"),
    }
    _rng_style = {
        "high":   ("HIGH",   "#dc3545", "#fff0f0"),
        "low":    ("LOW",    "#ff9800", "#fff8e1"),
        "normal": ("NORMAL", "#28a745", "#f0fff4"),
    }

    def _trend_badge(line):
        try:
            vital, rest = line.split(" → ")
            rng, dirn = rest.split(" & ")
        except Exception:
            return f"<span style='font-size:13px;'>{line}</span>"
        d_icon, d_col, _ = _dir_style.get(dirn.strip(), ("•", "#555", "#eee"))
        r_lbl, r_col, r_bg = _rng_style.get(rng.strip(), (rng.upper(), "#555", "#eee"))
        return (
            f"<span style='font-size:13px;font-weight:700;color:#1a1a2e;min-width:60px;"
            f"display:inline-block;'>{vital}</span>"
            f"<span style='margin:0 6px;color:#bbb;'>→</span>"
            f"<span style='background:{r_bg};color:{r_col};border:1.5px solid {r_col};"
            f"border-radius:5px;padding:2px 9px;font-size:11px;font-weight:800;"
            f"margin-right:5px;letter-spacing:0.3px;'>{r_lbl}</span>"
            f"<span style='background:#f0f4ff;border-radius:5px;padding:2px 9px;"
            f"font-size:11px;font-weight:700;color:{d_col};border:1px solid #c8d8f8;'>"
            f"{d_icon} {dirn.strip()}</span>"
        )

    _trend_html = ""
    for i, line in enumerate(trend_lines):
        _trend_html += f"<div style='padding:6px 10px;background:{'#fafafa' if i%2==0 else 'white'};" \
                       f"border-radius:6px;margin:3px 0;'>{_trend_badge(line)}</div>"

    tl, tr = st.columns(2)
    with tl:
        for i, line in enumerate(trend_lines[:4]):
            st.markdown(f"<div style='padding:6px 10px;background:{'#fafafa' if i%2==0 else 'white'};"
                        f"border-radius:6px;margin:3px 0;border-left:3px solid #e2e8f0;'>"
                        f"{_trend_badge(line)}</div>", unsafe_allow_html=True)
    with tr:
        for i, line in enumerate(trend_lines[4:]):
            st.markdown(f"<div style='padding:6px 10px;background:{'#fafafa' if i%2==0 else 'white'};"
                        f"border-radius:6px;margin:3px 0;border-left:3px solid #e2e8f0;'>"
                        f"{_trend_badge(line)}</div>", unsafe_allow_html=True)

    # ── Prediction History (colour-coded by risk) ──
    if os.path.exists(HISTORY_FILE):
        _ph = pd.read_csv(HISTORY_FILE)
        if len(_ph) > 1:
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            with st.expander(f"📋 Auto-Reading History  ({len(_ph)} readings, newest first)"):
                st.caption(
                    f"Each row = one automatic 30-second reading for "
                    f"{patient_cfg['icon']} {patient_cfg['name']}. "
                    f"Alert threshold: SOFA ≥ {ALERT_THRESHOLD:.0f}. Keeps last 50."
                )
                _ph_display = _ph.iloc[::-1].reset_index(drop=True)

                def _color_risk_row(row):
                    risk_val = str(row.get("Risk", ""))
                    if "High" in risk_val:
                        bg = "background-color: #fff0f0"
                    elif "Moderate" in risk_val:
                        bg = "background-color: #fffbf0"
                    else:
                        bg = "background-color: #f0fff4"
                    return [bg] * len(row)

                _styled = _ph_display.style.apply(_color_risk_row, axis=1)
                st.dataframe(_styled, use_container_width=True, hide_index=True)

# ---- TAB 2: EXPLAINABILITY ----
with tab2:
    if top_shap is not None:

        # ── SHAP feature impact icon map ──
        _FEAT_ICONS = {
            "HR": "❤️", "RR": "🫁", "SpO2": "💧", "Temp": "🌡️",
            "SBP": "🩸", "DBP": "🩸", "MAP": "📉", "GCS": "🧠",
            "stress": "😰", "HR_mean": "❤️", "HR_std": "❤️",
            "SpO2_mean": "💧", "SpO2_min": "💧", "Temp_mean": "🌡️",
            "SBP_mean": "🩸", "DBP_mean": "🩸", "MAP_mean": "📉",
        }
        def _feat_icon(feat):
            for k, v in _FEAT_ICONS.items():
                if k.lower() in feat.lower():
                    return v
            return "🔬"  # TF-IDF term

        # ─────────────────────────────────────────────────────────────
        # Section 1: SHAP Impact Bars
        # ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:18px;font-weight:800;color:#1a1a2e;margin-bottom:6px;">
            🔬 SHAP Feature Impact Analysis
        </div>
        <div style="font-size:12px;color:#666;margin-bottom:14px;">
            How much each feature <em>shifted</em> this patient's predicted SOFA away from baseline.
            Wider bar = stronger influence. Colour = direction.
        </div>
        """, unsafe_allow_html=True)

        _max_imp = top_shap["abs_impact"].max() if not top_shap.empty else 1.0

        for _, _r in top_shap.iterrows():
            _feat  = _r["feature"]
            _orig  = _r["original_value"]
            _imp   = _r["impact"]
            _abimp = _r["abs_impact"]

            if _feat in VITAL_UNITS:
                _lbl, _unit = VITAL_UNITS[_feat]
                _fmt = ".0f" if _unit in ("/4", "/10") else ".1f"
                _val = f"{_orig:{_fmt}} {_unit}"
                _ico = _feat_icon(_feat)
            else:
                _lbl = _feat
                _val = "detected in notes" if _orig > 0 else "absent in notes"
                _ico = "🔬"

            _inc   = _imp > 0
            _bcol  = "#dc3545" if _inc else "#28a745"
            _dcol  = "#c62828" if _inc else "#2e7d32"
            _rbg   = "linear-gradient(90deg,#fff5f5,#fff9f9)" if _inc else "linear-gradient(90deg,#f5fff8,#f8fff9)"
            _dico  = "↑" if _inc else "↓"
            _dtxt  = "Increases Risk" if _inc else "Reduces Risk"
            _bw    = int(_abimp / _max_imp * 100)

            st.markdown(f"""
            <div style="background:{_rbg};border-left:5px solid {_bcol};border-radius:0 10px 10px 0;
                        padding:12px 16px;margin:5px 0;display:flex;align-items:center;gap:14px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                <div style="font-size:22px;line-height:1;">{_ico}</div>
                <div style="min-width:150px;flex-shrink:0;">
                    <div style="font-size:13px;font-weight:700;color:#1a1a2e;">{_lbl}</div>
                    <div style="font-size:11px;color:#777;margin-top:2px;">{_val}</div>
                </div>
                <div style="flex:1;background:#e8e8e8;border-radius:6px;height:14px;overflow:hidden;min-width:80px;">
                    <div style="width:{_bw}%;background:{_bcol};height:100%;border-radius:6px;"></div>
                </div>
                <div style="min-width:110px;text-align:right;flex-shrink:0;">
                    <div style="font-size:15px;font-weight:900;color:{_dcol};">{_abimp:.4f}</div>
                    <div style="font-size:11px;font-weight:700;color:{_dcol};">{_dico} {_dtxt}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # Section 2: Clinical Interpretations (two-column cards)
        # ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:18px;font-weight:800;color:#1a1a2e;margin-bottom:10px;">
            💡 Clinical Interpretations
        </div>
        """, unsafe_allow_html=True)

        _ci_left, _ci_right = st.columns(2)
        for _i, _exp in enumerate(clinical_explanations):
            _is_inc = "increasing risk" in _exp
            _ci_bg  = "linear-gradient(135deg,#fff5f5,#fff9f9)" if _is_inc else "linear-gradient(135deg,#f5fff8,#f8fff9)"
            _ci_brd = "#dc3545" if _is_inc else "#28a745"
            _ci_ico = "↑" if _is_inc else "↓"
            _ci_col = "#c62828" if _is_inc else "#2e7d32"
            _html = (
                f"<div style='background:{_ci_bg};border:1.5px solid {_ci_brd};"
                f"border-radius:8px;padding:10px 14px;margin:5px 0;"
                f"display:flex;align-items:flex-start;gap:10px;'>"
                f"<span style='font-size:18px;font-weight:900;color:{_ci_col};"
                f"line-height:1.2;flex-shrink:0;'>{_ci_ico}</span>"
                f"<span style='font-size:12px;color:#1a1a2e;line-height:1.5;'>{_exp}</span>"
                f"</div>"
            )
            (_ci_left if _i % 2 == 0 else _ci_right).markdown(_html, unsafe_allow_html=True)

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # Section 3: Key Risk Factors (icon chips)
        # ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:18px;font-weight:800;color:#1a1a2e;margin-bottom:10px;">
            ⚠️ Key Risk Factors
        </div>
        """, unsafe_allow_html=True)

        if key_risks:
            _RISK_ICONS = {
                "Low oxygen levels":              ("💧", "#1565c0", "#e3f2fd", "#90caf9"),
                "Respiratory distress":           ("🫁", "#7b1fa2", "#f3e5f5", "#ce93d8"),
                "Organ failure":                  ("🚨", "#b71c1c", "#ffebee", "#ef9a9a"),
                "Sepsis / Infection":             ("🦠", "#e65100", "#fff3e0", "#ffcc80"),
                "Respiratory failure (intubated)":("😮‍💨", "#880e4f", "#fce4ec", "#f48fb1"),
                "Haemodynamic instability":       ("💔", "#b71c1c", "#ffebee", "#ef9a9a"),
                "Hypotension":                    ("📉", "#4a148c", "#ede7f6", "#b39ddb"),
                "Abnormal heart rate":            ("❤️", "#c62828", "#ffebee", "#ef9a9a"),
                "Altered mental status":          ("🧠", "#1a237e", "#e8eaf6", "#9fa8da"),
                "Neurological deterioration":     ("🧠", "#1a237e", "#e8eaf6", "#9fa8da"),
                "High physiological stress":      ("😰", "#e65100", "#fff3e0", "#ffcc80"),
            }

            _chips_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:4px 0;">'
            for _risk in key_risks:
                _rico, _tcol, _rbg2, _rbrd = _RISK_ICONS.get(
                    _risk, ("⚠️", "#b71c1c", "#ffebee", "#ef9a9a")
                )
                _chips_html += (
                    f"<div style='background:{_rbg2};border:2px solid {_rbrd};"
                    f"border-radius:10px;padding:10px 18px;"
                    f"display:inline-flex;align-items:center;gap:10px;"
                    f"box-shadow:0 2px 8px rgba(0,0,0,0.08);'>"
                    f"<span style='font-size:20px;'>{_rico}</span>"
                    f"<div>"
                    f"<div style='font-size:13px;font-weight:800;color:{_tcol};'>{_risk}</div>"
                    f"<div style='font-size:10px;color:#888;'>SHAP-identified risk factor</div>"
                    f"</div></div>"
                )
            _chips_html += "</div>"
            st.markdown(_chips_html, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='padding:12px;background:#f0fff4;border-radius:8px;"
                "border:1px solid #9ae6b4;color:#276749;font-size:13px;'>"
                "✅ No specific clinical risk factors flagged by the model for this reading.</div>",
                unsafe_allow_html=True
            )

        st.divider()

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

    # ── Section colours for parsed LLM output ──
    _SEC_CFG = {
        "CURRENT CONDITION":  ("📋", "#1565c0", "#e8f4fd", "#90caf9",
                               "Patient status right now"),
        "PROBABLE CAUSE":     ("🔍", "#bf360c", "#fff8f5", "#ffab91",
                               "Why this is happening"),
        "RISK FORECAST":      ("🔮", "#6a1b9a", "#f5f0ff", "#ce93d8",
                               "Predicted trajectory if untreated"),
        "IMMEDIATE ACTIONS":  ("🚨", "#b71c1c", "#fff5f5", "#ef9a9a",
                               "Critical interventions — next 30 minutes"),
    }

    # ── Reliability palette ──
    _rel_col = "#28a745" if consistency >= 0.80 else "#f0a500" if consistency >= 0.60 else "#dc3545"
    _rel_bg  = "#e8fce8" if consistency >= 0.80 else "#fffbf0" if consistency >= 0.60 else "#fff0f0"
    _rel_brd = "#9ae6b4" if consistency >= 0.80 else "#fbd38d" if consistency >= 0.60 else "#feb2b2"
    _rel_ico = "✅" if consistency >= 0.80 else "⚠️" if consistency >= 0.60 else "❌"
    _rel_lbl = "High Reliability" if consistency >= 0.80 else \
               "Moderate Reliability" if consistency >= 0.60 else "Low Reliability"
    _rel_msg = (
        "All 3 AI responses agree on clinical findings, diagnoses, and recommended interventions."
        if consistency >= 0.80 else
        "Responses agree on core findings with some variation in secondary recommendations. Review with care."
        if consistency >= 0.60 else
        "Significant disagreement across responses on clinical findings or interventions. Use clinical judgment."
    )

    # ── Per-response validity indicators ──
    def _resp_valid(r):
        return r and r != _FALLBACK_MSG and len(r) > 80

    _rvalid = [_resp_valid(r) for r in responses] if responses else [False, False, False]
    while len(_rvalid) < 3:
        _rvalid.append(False)

    _r_icons_html = "".join(
        f"<div style='text-align:center;background:white;border-radius:8px;padding:6px 10px;"
        f"border:2px solid {'#28a745' if v else '#dc3545'};min-width:52px;'>"
        f"<div style='font-size:16px;font-weight:900;color:{'#28a745' if v else '#dc3545'};'>"
        f"{'✓' if v else '✗'}</div>"
        f"<div style='font-size:9px;color:#888;margin-top:2px;'>R{n+1}</div>"
        f"</div>"
        for n, v in enumerate(_rvalid)
    )

    # ── Reliability banner ──
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{_rel_bg},white);
                border:2px solid {_rel_brd};border-radius:14px;
                padding:18px 22px;margin-bottom:14px;
                box-shadow:0 3px 12px rgba(0,0,0,0.07);">
        <div style="display:flex;justify-content:space-between;align-items:center;
                    flex-wrap:wrap;gap:16px;">
            <div>
                <div style="font-size:11px;font-weight:700;color:{_rel_col};letter-spacing:1.5px;
                            text-transform:uppercase;margin-bottom:4px;">
                    🧠 AI Clinical Assessment
                </div>
                <div style="font-size:22px;font-weight:900;color:{_rel_col};margin-bottom:6px;">
                    {_rel_ico} {_rel_lbl}
                </div>
                <div style="font-size:12px;color:#444;max-width:440px;line-height:1.5;">
                    {_rel_msg}
                </div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:10px;">
                <div style="text-align:center;">
                    <div style="font-size:10px;color:#888;text-transform:uppercase;
                                letter-spacing:0.6px;">Consistency Score</div>
                    <div style="font-size:52px;font-weight:900;color:{_rel_col};
                                line-height:1;">{consistency:.2f}</div>
                    <div style="font-size:10px;color:#888;">across 3 independent responses</div>
                </div>
                <div style="display:flex;gap:6px;">{_r_icons_html}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Parse and render the main response as section cards ──
    def _parse_llm_sections(text):
        """Split LLM text into {SECTION_NAME: content} dict."""
        _NAMES = [
            "IMMEDIATE ACTIONS", "CURRENT CONDITION",
            "PROBABLE CAUSE",    "RISK FORECAST",
        ]
        positions = []
        for sec in _NAMES:
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

    _sections = _parse_llm_sections(main_response)

    if _sections:
        for _sname, _scontent in _sections.items():
            _sico, _stcol, _ssbg, _ssbrd, _ssub = _SEC_CFG.get(
                _sname, ("📄", "#555", "#f8f8f8", "#ccc", "")
            )
            # Section header as styled div
            st.markdown(f"""
            <div style="background:{_ssbg};border-left:6px solid {_stcol};
                        border-radius:0 10px 10px 0;padding:12px 16px 6px;margin-top:14px;
                        box-shadow:0 2px 6px rgba(0,0,0,0.05);">
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:20px;">{_sico}</span>
                    <div>
                        <div style="font-size:13px;font-weight:800;color:{_stcol};
                                    letter-spacing:0.8px;text-transform:uppercase;">{_sname}</div>
                        <div style="font-size:10px;color:{_stcol};opacity:0.7;">{_ssub}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Section content — rendered as normal markdown so bullets/bold work
            with st.container():
                st.markdown(
                    f"<div style='background:{_ssbg};border-left:6px solid {_stcol};"
                    f"border-radius:0;padding:2px 16px 14px 16px;margin-bottom:4px;'>",
                    unsafe_allow_html=True
                )
                st.markdown(_scontent)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Fallback: couldn't parse sections — render as plain markdown
        st.markdown(main_response)

    # ── All 3 responses expander ──
    if responses:
        _valid_count = sum(1 for r in responses if _resp_valid(r))
        with st.expander(
            f"🔎 View all 3 independent LLM responses "
            f"— {_valid_count}/3 produced valid output"
        ):
            for _i, _resp in enumerate(responses):
                _is_valid = _resp_valid(_resp)
                _card_bg  = "#f8fafc" if _is_valid else "#fff8f0"
                _card_brd = "#e2e8f0" if _is_valid else "#f6ad55"
                _hdr_ico  = f"✓ Response {_i+1}" if _is_valid else f"⚠ Response {_i+1} — incomplete"
                _hdr_col  = "#1a1a2e" if _is_valid else "#c05621"

                st.markdown(f"""
                <div style="background:{_card_bg};border:1.5px solid {_card_brd};
                            border-radius:10px;padding:14px 18px;margin:10px 0;">
                    <div style="font-size:13px;font-weight:800;color:{_hdr_col};
                                margin-bottom:10px;border-bottom:1px solid {_card_brd};
                                padding-bottom:8px;">{_hdr_ico}</div>
                </div>
                """, unsafe_allow_html=True)
                if _is_valid:
                    st.markdown(_resp)
                else:
                    st.warning(_resp)

    # ── Clinical Disclaimer ──
    st.divider()
    st.markdown("""
    <div style="background:#e8f4fd;border:1.5px solid #90caf9;border-left:5px solid #1565c0;
                border-radius:0 10px 10px 0;padding:14px 18px;">
        <div style="font-size:13px;font-weight:700;color:#1565c0;margin-bottom:4px;">
            ⚕️ Clinical Disclaimer
        </div>
        <div style="font-size:12px;color:#1a1a2e;line-height:1.6;">
            This system is a <strong>decision support tool only</strong>. It does not diagnose
            disease or replace the clinical judgment of qualified healthcare professionals.
            All AI-generated outputs must be reviewed by a licensed clinician before any
            clinical action is taken.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- TAB 4: FEDERATED LEARNING INFO ----
with tab4:
    m = training_meta

    # ── Tab header banner ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
                border-radius:14px;padding:20px 24px;margin-bottom:20px;">
        <div style="font-size:22px;font-weight:900;color:white;margin-bottom:4px;">
            🔒 Federated Learning — How This Model Was Trained
        </div>
        <div style="font-size:13px;color:#7fb3c8;line-height:1.5;">
            Privacy-preserving collaborative AI across 3 hospital ICUs.
            Patient data <strong style="color:#00d2ff;">never leaves</strong> each hospital —
            only model weights are shared.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FL Protocol Diagram (HTML, replacing ASCII art) ──
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0f2027,#1a2f3a,#1e3a4a);'
        'border-radius:16px;padding:28px 24px;color:white;border:1.5px solid #2a4a5a;">'

        '<div style="display:flex;justify-content:center;margin-bottom:12px;">'
        '<div style="background:rgba(0,210,255,0.12);border:2px solid #00d2ff;'
        'border-radius:12px;padding:14px 32px;text-align:center;'
        'box-shadow:0 0 20px rgba(0,210,255,0.15);">'
        '<div style="font-size:20px;margin-bottom:4px;">🖥</div>'
        '<div style="font-size:13px;font-weight:800;color:#00d2ff;letter-spacing:0.5px;">Global FL Server</div>'
        '<div style="font-size:11px;color:#7fb3c8;margin-top:2px;">Flower Framework · FedAvg Aggregation</div>'
        '</div></div>'

        '<div style="text-align:center;color:#00d2ff;font-size:12px;margin:8px 0;font-weight:600;">'
        '① Share global weights ↓↓↓</div>'

        '<div style="display:flex;justify-content:center;gap:14px;margin:8px 0;">'

        '<div style="background:rgba(40,167,69,0.12);border:1.5px solid #4caf50;'
        'border-radius:10px;padding:12px 16px;text-align:center;flex:1;max-width:200px;">'
        '<div style="font-size:16px;margin-bottom:4px;">🏥</div>'
        '<div style="font-size:12px;font-weight:700;color:#81c784;">Hospital 0</div>'
        '<div style="font-size:10px;color:#aaa;">General ICU</div>'
        '<div style="font-size:11px;color:#4caf50;font-weight:600;margin-top:4px;">~15,889 patients</div>'
        '<div style="font-size:10px;color:#666;margin-top:6px;background:rgba(0,0,0,0.3);'
        'border-radius:4px;padding:4px;">🔒 PRIVATE data</div></div>'

        '<div style="background:rgba(240,165,0,0.12);border:1.5px solid #f0a500;'
        'border-radius:10px;padding:12px 16px;text-align:center;flex:1;max-width:200px;">'
        '<div style="font-size:16px;margin-bottom:4px;">🏥</div>'
        '<div style="font-size:12px;font-weight:700;color:#ffd54f;">Hospital 1</div>'
        '<div style="font-size:10px;color:#aaa;">Mixed ICU</div>'
        '<div style="font-size:11px;color:#f0a500;font-weight:600;margin-top:4px;">~15,890 patients</div>'
        '<div style="font-size:10px;color:#666;margin-top:6px;background:rgba(0,0,0,0.3);'
        'border-radius:4px;padding:4px;">🔒 PRIVATE data</div></div>'

        '<div style="background:rgba(220,53,69,0.12);border:1.5px solid #dc3545;'
        'border-radius:10px;padding:12px 16px;text-align:center;flex:1;max-width:200px;">'
        '<div style="font-size:16px;margin-bottom:4px;">🏥</div>'
        '<div style="font-size:12px;font-weight:700;color:#ef9a9a;">Hospital 2</div>'
        '<div style="font-size:10px;color:#aaa;">Cardiac/Trauma ICU</div>'
        '<div style="font-size:11px;color:#dc3545;font-weight:600;margin-top:4px;">~16,372 patients</div>'
        '<div style="font-size:10px;color:#666;margin-top:6px;background:rgba(0,0,0,0.3);'
        'border-radius:4px;padding:4px;">🔒 PRIVATE data</div></div>'

        '</div>'

        '<div style="text-align:center;color:#7fb3c8;font-size:11px;margin:10px 0;line-height:1.7;">'
        '② Train locally on private data (no external access)<br>'
        '<span style="color:#00d2ff;font-weight:600;">'
        '③ Send ONLY model weights — zero patient records shared ↑↑↑</span></div>'

        '<div style="display:flex;justify-content:center;margin:8px 0;">'
        '<div style="background:rgba(106,27,154,0.2);border:2px solid #9c27b0;'
        'border-radius:12px;padding:12px 36px;text-align:center;">'
        '<div style="font-size:13px;font-weight:800;color:#ce93d8;">④ FedAvg: Average all hospital weights</div>'
        '<div style="font-size:11px;color:#7fb3c8;margin-top:4px;">→ Improved global model · Repeat for 20 rounds</div>'
        '</div></div>'

        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Training Configuration (coloured stat tiles) ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#1a1a2e;margin-bottom:12px;'>⚙️ Training Configuration</div>",
                unsafe_allow_html=True)

    _cfg_tiles = [
        ("FL Rounds",         str(m["num_rounds"]),         "#1565c0","#e3f2fd","#90caf9", "rounds of federation"),
        ("Hospitals",          str(m["hospitals"]),           "#6a1b9a","#f3e5f5","#ce93d8", "ICU sites"),
        ("Epochs / Round",     str(m["epochs_per_round"]),   "#2e7d32","#e8f5e9","#a5d6a7", "local training epochs"),
        ("Aggregation",        m["aggregation"],              "#e65100","#fff3e0","#ffcc80", "weight averaging method"),
        ("Training Samples",   f"{m['train_samples']:,}",    "#00695c","#e0f2f1","#80cbc4", "real ICU patients"),
        ("Test Samples",       f"{m['test_samples']:,}",      "#558b2f","#f1f8e9","#c5e1a5", "held-out patients"),
        ("Best Round",         str(m["best_round"]),          "#f57f17","#fffde7","#fff176", "lowest eval loss"),
        ("Split Type",         m["split_type"],               "#4527a0","#ede7f6","#b39ddb", "data distribution"),
    ]

    _t1, _t2, _t3, _t4 = st.columns(4)
    _t5, _t6, _t7, _t8 = st.columns(4)
    for _cols, _tiles in [([_t1,_t2,_t3,_t4], _cfg_tiles[:4]),
                           ([_t5,_t6,_t7,_t8], _cfg_tiles[4:])]:
        for _col, (_lbl, _val, _tc, _bg, _brd, _sub) in zip(_cols, _tiles):
            with _col:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{_bg},white);border:1.5px solid {_brd};
                            border-radius:10px;padding:14px 12px;text-align:center;
                            box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:8px;">
                    <div style="font-size:9px;font-weight:700;color:{_tc};text-transform:uppercase;
                                letter-spacing:0.6px;">{_lbl}</div>
                    <div style="font-size:26px;font-weight:900;color:{_tc};line-height:1.1;
                                margin:4px 0;">{_val}</div>
                    <div style="font-size:9px;color:#888;">{_sub}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ── Global Model Performance ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#1a1a2e;margin-bottom:12px;'>📊 Global Model Performance</div>",
                unsafe_allow_html=True)

    _perf_tiles = [
        ("Mean Abs. Error",     f"{m['final_mae']:.3f}", "SOFA points", "#1565c0","#e3f2fd","#90caf9",
         "Average prediction error on 9,631 held-out ICU patients"),
        ("R² Score",            f"{m['final_r2']:.3f}",  "variance",   "#2e7d32","#e8f5e9","#a5d6a7",
         "Proportion of variance explained by the model"),
        ("Pred Range Min",      f"{m['pred_range_min']:.1f}", "SOFA", "#00695c","#e0f2f1","#80cbc4",
         "Lowest predicted SOFA across test set"),
        ("Pred Range Max",      f"{m['pred_range_max']:.1f}", "SOFA", "#e65100","#fff3e0","#ffcc80",
         "Highest predicted SOFA across test set"),
    ]

    _p1, _p2, _p3, _p4 = st.columns(4)
    for _col, (_lbl, _val, _unit, _tc, _bg, _brd, _desc) in zip([_p1,_p2,_p3,_p4], _perf_tiles):
        with _col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{_bg},white);border:2px solid {_brd};
                        border-radius:12px;padding:18px 12px;text-align:center;
                        box-shadow:0 3px 10px rgba(0,0,0,0.07);">
                <div style="font-size:9px;font-weight:700;color:{_tc};text-transform:uppercase;
                            letter-spacing:0.7px;margin-bottom:4px;">{_lbl}</div>
                <div style="font-size:34px;font-weight:900;color:{_tc};line-height:1;">{_val}</div>
                <div style="font-size:10px;color:#888;">{_unit}</div>
                <div style="font-size:9px;color:#aaa;margin-top:6px;line-height:1.4;">{_desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Model Architecture ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#1a1a2e;margin-bottom:12px;'>🧠 Model Architecture (PyTorch DNN)</div>",
                unsafe_allow_html=True)

    _layers = [
        ("INPUT",  "618 features",  "Trend vitals + Latest vitals + CV + TF-IDF", "#1565c0","#e3f2fd"),
        ("Linear", "618 → 256",     "Fully connected · ReLU activation",          "#2e7d32","#e8f5e9"),
        ("Linear", "256 → 128",     "Fully connected · ReLU activation",          "#2e7d32","#e8f5e9"),
        ("Linear", "128 →  64",     "Fully connected · ReLU activation",          "#2e7d32","#e8f5e9"),
        ("Linear", " 64 →   1",     "Output layer · No activation (regression)",  "#6a1b9a","#f3e5f5"),
        ("OUTPUT", "SOFA (0–24)",   "Predicted SOFA score · clip(0, 24)",         "#e65100","#fff3e0"),
    ]

    _arch_html = ""
    for _i, (_ltype, _ldim, _ldesc, _tc, _bg) in enumerate(_layers):
        _arrow = "<div style='text-align:center;font-size:18px;color:#888;margin:2px 0;'>↓</div>" if _i < len(_layers)-1 else ""
        _arch_html += f"""
        <div style="background:{_bg};border:1.5px solid {_tc};border-radius:8px;
                    padding:10px 16px;display:flex;align-items:center;gap:12px;">
            <div style="background:{_tc};color:white;border-radius:5px;padding:3px 8px;
                        font-size:10px;font-weight:800;letter-spacing:0.5px;
                        white-space:nowrap;">{_ltype}</div>
            <div style="font-size:14px;font-weight:700;color:{_tc};min-width:80px;">{_ldim}</div>
            <div style="font-size:11px;color:#555;">{_ldesc}</div>
        </div>{_arrow}"""

    _la, _lb = st.columns([3, 2])
    with _la:
        st.markdown(_arch_html, unsafe_allow_html=True)
    with _lb:
        st.markdown(f"""
        <div style="background:#f8fafc;border:1.5px solid #e2e8f0;border-radius:12px;
                    padding:16px;font-size:12px;color:#444;line-height:1.8;height:100%;">
            <div style="font-size:13px;font-weight:700;color:#1a1a2e;margin-bottom:8px;">
                🔧 Training Details
            </div>
            <b>Optimizer:</b> AdamW (weight_decay=1e-4)<br>
            <b>Loss:</b> Weighted MSE — high-SOFA patients get up to 4.7× gradient weight<br>
            <b>No Dropout</b> — causes FL divergence; regularised by AdamW instead<br>
            <b>Parameters:</b> 199,681 total trainable weights<br>
            <b>FL Framework:</b> Flower (flwr) with FedAvg aggregation<br>
            <b>Best round:</b> {m['best_round']} of {m['num_rounds']}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Differential Privacy ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#1a1a2e;margin-bottom:12px;'>🔐 Differential Privacy</div>",
                unsafe_allow_html=True)

    if m.get("differential_privacy"):
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#e8f5e9,white);border:2px solid #4caf50;
                    border-left:6px solid #2e7d32;border-radius:0 12px 12px 0;padding:16px 20px;">
            <div style="font-size:14px;font-weight:800;color:#2e7d32;margin-bottom:8px;">
                ✅ Differential Privacy ENABLED
            </div>
            <div style="display:flex;gap:20px;flex-wrap:wrap;font-size:12px;color:#444;">
                <span><b>σ (noise multiplier):</b> {m['dp_sigma']}</span>
                <span><b>S (sensitivity):</b> {m['dp_sensitivity']}</span>
                <span><b>ε (privacy budget):</b> ≈{m['dp_epsilon']}</span>
                <span><b>δ:</b> {m['dp_delta']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        _dp_steps = [
            ("①", "Compute update", "local_weights − global_weights", "#1565c0"),
            ("②", "Clip L2 norm",   "Bounds any patient's max influence (sensitivity S)", "#e65100"),
            ("③", "Add noise",      "Gaussian N(0, (σ·S)²) to every weight parameter", "#6a1b9a"),
            ("④", "Transmit",       "Server receives noisy update — cannot trace individuals", "#2e7d32"),
        ]
        st.markdown("""
        <div style="background:linear-gradient(135deg,#fff8f0,white);border:1.5px solid #ffcc80;
                    border-left:5px solid #f0a500;border-radius:0 12px 12px 0;
                    padding:14px 18px;margin-bottom:12px;">
            <div style="font-size:13px;font-weight:700;color:#e65100;">
                ⚙️ Differential Privacy: Disabled in current build
            </div>
            <div style="font-size:11px;color:#666;margin-top:4px;">
                Model trained with plain FedAvg (no noise). Enable: set
                <code>USE_DP = True</code> in train_federated.py and retrain.
            </div>
        </div>
        """, unsafe_allow_html=True)

        _dp_html = '<div style="display:flex;flex-direction:column;gap:8px;">'
        for _step, _title, _desc, _tc in _dp_steps:
            _dp_html += (
                f"<div style='display:flex;align-items:flex-start;gap:12px;background:#f8fafc;"
                f"border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;'>"
                f"<div style='background:{_tc};color:white;border-radius:50%;width:24px;height:24px;"
                f"display:flex;align-items:center;justify-content:center;font-size:11px;"
                f"font-weight:800;flex-shrink:0;'>{_step}</div>"
                f"<div><div style='font-size:12px;font-weight:700;color:{_tc};'>{_title}</div>"
                f"<div style='font-size:11px;color:#666;margin-top:2px;'>{_desc}</div></div>"
                f"</div>"
            )
        _dp_html += "</div>"
        st.markdown(
            "<div style='font-size:12px;font-weight:700;color:#1a1a2e;margin-bottom:8px;'>What DP would add:</div>",
            unsafe_allow_html=True
        )
        st.markdown(_dp_html, unsafe_allow_html=True)

    st.divider()

    # ── Feature Vector Breakdown (visual bars) ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#1a1a2e;margin-bottom:12px;'>🔢 Feature Vector Breakdown — 618 total</div>",
                unsafe_allow_html=True)

    _feat_rows = [
        ("📈 Trend Vitals",         9,   618, "#1565c0","#e3f2fd",
         "HR_mean, HR_std, RR_mean, SpO₂_mean, SpO₂_min, Temp_mean, SBP_mean, DBP_mean, MAP_mean"),
        ("📊 Latest Vitals",         7,   618, "#2e7d32","#e8f5e9",
         "latest_HR, latest_RR, latest_SpO₂, latest_Temp, latest_SBP, latest_DBP, latest_MAP"),
        ("👁️ Computer Vision",       2,   618, "#6a1b9a","#f3e5f5",
         "GCS Eye Opening (1–4 scale), Stress Score (0–10)"),
        ("📝 Clinical NLP (TF-IDF)", 600, 618, "#e65100","#fff3e0",
         "600 clinical bigrams: hypotension, respiratory, intubated, vasopressor … (MIMIC-III vocab)"),
    ]

    for _flabel, _fcount, _ftotal, _ftc, _fbg, _fex in _feat_rows:
        _fw = _fcount / _ftotal * 100
        st.markdown(f"""
        <div style="background:{_fbg};border-left:5px solid {_ftc};border-radius:0 10px 10px 0;
                    padding:12px 16px;margin:6px 0;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
            <div style="display:flex;justify-content:space-between;align-items:center;
                        margin-bottom:6px;">
                <div style="font-size:13px;font-weight:700;color:{_ftc};">{_flabel}</div>
                <div style="font-size:14px;font-weight:900;color:{_ftc};">
                    {_fcount} <span style="font-size:10px;color:#888;">/ 618 features ({_fw:.1f}%)</span>
                </div>
            </div>
            <div style="background:#ddd;border-radius:6px;height:10px;margin-bottom:6px;overflow:hidden;">
                <div style="width:{_fw:.0f}%;background:{_ftc};height:100%;border-radius:6px;"></div>
            </div>
            <div style="font-size:10px;color:#666;">{_fex}</div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================
# CONTINUOUS MONITORING — 30-SECOND COUNTDOWN (writes to TOP slots)
# =============================================================
st.divider()

for _rem in range(30, 0, -1):
    _pct = (30 - _rem) / 30
    # Update the placeholder that was rendered right after the LIVE banner
    _top_cd.markdown(f"""
    <div class="cdbar" style="margin-top:4px; margin-bottom:2px;">
        <span class="live-dot"></span>
        <span style="font-weight:700; letter-spacing:0.5px;">CONTINUOUS MONITORING</span>
        <span style="color:#4a7a8a;">|</span>
        <span>{patient_cfg['icon']} {patient_cfg['name']}</span>
        <span style="color:#4a7a8a;">|</span>
        <span>Reading <b style="color:#00d2ff;">{cur_idx + 1}/{total_rows}</b></span>
        <span style="color:#4a7a8a;">|</span>
        <span>Next reading in <b style="color:#00d2ff; font-family:monospace;">{_rem:02d}s</b></span>
    </div>
    """, unsafe_allow_html=True)
    _top_bar.progress(_pct)
    time.sleep(1)

_top_cd.markdown(f"""
<div class="cdbar" style="margin-top:4px; margin-bottom:2px; border-color:#28a745;">
    <span style="font-size:15px;">✅</span>
    <span style="font-weight:700;">Fetching next reading…</span>
    <span style="color:#4a7a8a;">|</span>
    <span>{patient_cfg['icon']} {patient_cfg['name']}</span>
</div>
""", unsafe_allow_html=True)
_top_bar.progress(1.0)

# Advance to next row and trigger rerun
st.session_state.row_indices[patient_id] = (row_idx + 1) % total_rows
st.rerun()

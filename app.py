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
from decimal import Decimal, InvalidOperation
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from model_utils import ICUModel, get_trend, classify_range
from db import (
    init_db,
    get_patient_vitals, append_patient_vitals,
    get_prediction_history, append_prediction,
    get_sample_patient_row, get_sample_patient_count,
    get_fl_client_data, fl_training_exists,
)

load_dotenv()
init_db()   # create tables if this is a fresh run

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
    animation:slideIn 0.4s ease; transition:transform 0.15s, box-shadow 0.15s;
}
.pcard:hover { transform:translateY(-3px); box-shadow:0 8px 30px rgba(0,0,0,0.4); }
.pcard-low  { border:2.5px solid #28a745; background:linear-gradient(140deg,rgba(40,167,69,0.18),rgba(40,167,69,0.06)); }
.pcard-mod  { border:2.5px solid #f0a500; background:linear-gradient(140deg,rgba(240,165,0,0.18),rgba(240,165,0,0.06)); }
.pcard-high { border:2.5px solid #dc3545; background:linear-gradient(140deg,rgba(220,53,69,0.18),rgba(220,53,69,0.06)); }
.pcard h3   { margin:0 0 6px 0; font-size:20px; color:#e8f4ff; }
.pcard p    { margin:4px 0; color:#b8d0e0; font-size:13px; }
.pcard .tag { display:inline-block; border-radius:20px; padding:3px 12px;
              font-size:11px; font-weight:800; letter-spacing:0.8px; }
.tag-low    { background:rgba(40,167,69,0.25);  color:#5fda80; border:1.5px solid #28a745; }
.tag-mod    { background:rgba(240,165,0,0.25);  color:#ffc93c; border:1.5px solid #f0a500; }
.tag-high   { background:rgba(220,53,69,0.25);  color:#ff7b7b; border:1.5px solid #dc3545; }

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

/* ── Remove sidebar completely ─────────────────────────────────────── */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="collapsedControl"] {
    display: none !important;
    width: 0 !important;
}
/* Main content expands to fill full width */
.main .block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"] > .main { margin-left: 0 !important; }

/* ── Dark theme base for app.py ────────────────────────────────────── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background: #0a1628 !important;
    color: #d0e0ec !important;
}

/* ── Text visibility fixes ─────────────────────────────────────────── */
div[data-testid="stMarkdownContainer"] p { color: #d0e0ec !important; }
div[data-testid="stMarkdownContainer"] strong,
div[data-testid="stMarkdownContainer"] b { color: #e8f4ff !important; }
label[data-testid="stWidgetLabel"] p { color: #7fb3c8 !important; }

/* ── Expander dark style ───────────────────────────────────────────── */
details {
    background: rgba(12,26,46,0.9) !important;
    border: 1px solid #1e3a50 !important;
    border-radius: 10px !important;
}
details summary { color: #7fb3c8 !important; font-weight: 600 !important; }
details summary p, details summary span { color: #7fb3c8 !important; }
div[data-testid="stExpanderDetails"] p,
div[data-testid="stExpanderDetails"] li { color: #c8dced !important; }
div[data-testid="stExpanderDetails"] b,
div[data-testid="stExpanderDetails"] strong { color: #e8f4ff !important; }

/* ── Caption text ──────────────────────────────────────────────────── */
[data-testid="stCaption"] p,
small { color: #8ab0c8 !important; font-size: 11px !important; }

/* ── Primary button — force dark text on cyan background ───────────── */
/* NOTE: Streamlit renders button text inside stMarkdownContainer > p,    */
/* so we must target the <p> specifically to override the global p rule.  */
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-primary"] p,
button[data-testid="baseButton-primary"] span,
button[data-testid="baseButton-primary"] div,
button[data-testid="baseButton-primary"] div[data-testid="stMarkdownContainer"] p {
    color: #0a1628 !important;
    font-weight: 800 !important;
}
button[data-testid="baseButton-primary"]:hover,
button[data-testid="baseButton-primary"]:hover p {
    color: #0a1628 !important;
    filter: brightness(1.08);
}

/* ── Secondary button — ensure readable text ───────────────────────── */
button[data-testid="baseButton-secondary"] {
    color: #c8dced !important;
    border-color: #2a4a6a !important;
}
button[data-testid="baseButton-secondary"] p,
button[data-testid="baseButton-secondary"] div[data-testid="stMarkdownContainer"] p {
    color: #c8dced !important;
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
        "icon":        "🟢",
        "description": "Post-surgical recovery — stable, improving trend",
        "color":       "#28a745",
    },
    2: {
        "name":        "Patient 2 — Moderate Risk",
        "short":       "Moderate Risk",
        "icon":        "🟡",
        "description": "Community-acquired pneumonia — on supplemental O₂",
        "color":       "#e6a817",
    },
    3: {
        "name":        "Patient 3 — High Risk",
        "short":       "High Risk",
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
def compute_conformal_q_hat(_model, _feature_cols):
    """
    Compute 90% conformal prediction quantile (q_hat) from FL calibration data.

    Method: inductive conformal prediction (split conformal).
      1. Load FL client data from fl_training table (pre-scaled, same pipeline as training).
      2. Compute nonconformity scores: |predicted − true SOFA|.
      3. q_hat = 90th percentile of those scores.
    Runtime:  interval = [pred − q_hat,  pred + q_hat]
    Guarantee: P(true SOFA ∈ interval) ≥ 90% on this data distribution.
    Reference: JAMIA Open 2025 — 90.4% empirical coverage on MIMIC-III at 90% target.
    """
    try:
        if not fl_training_exists():
            return 2.9  # DB not yet populated — fall back to ≈1.6 × MAE
        errors = []
        for i in range(3):
            df = get_fl_client_data(i)          # reads from fl_training table
            if df.empty or "sofa_score" not in df.columns:
                continue
            y_true = df["sofa_score"].values.astype(np.float32)
            X = df.drop(columns=["sofa_score"])
            for col in _feature_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[list(_feature_cols)].values.astype(np.float32)
            X = np.clip(X, -10, 10)
            _model.eval()
            with torch.no_grad():
                preds = _model(
                    torch.tensor(X, dtype=torch.float32)
                ).numpy().flatten()
            errors.extend(np.abs(preds - y_true).tolist())
        if len(errors) < 50:
            return 2.9
        return round(float(np.quantile(errors, 0.90)), 2)
    except Exception:
        return 2.9


conformal_q_hat = compute_conformal_q_hat(model, feature_cols)


@st.cache_resource
def load_training_metadata():
    """Load FL training metadata saved by train_federated.py."""
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
        "pred_range_min": 0.03, "pred_range_max": 13.87,
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
# _sel_slot is created on EVERY run. In patient-selector runs it is filled.
# In monitoring runs nothing is written into it → Streamlit auto-clears it,
# guaranteeing the patient cards are fully removed before monitoring renders.
_sel_slot = st.empty()

if st.session_state.selected_patient is None:
    with _sel_slot.container():
        st.markdown("""
        <div style="text-align:center; padding:10px 0 28px 0;">
            <div style="font-size:36px; font-weight:900; color:#e8f4ff; letter-spacing:-0.5px;">
                🏥 Select Patient to Monitor
            </div>
            <div style="font-size:15px; color:#b8d0e0; margin-top:10px; max-width:640px; margin-left:auto; margin-right:auto;">
                AI reads vitals every <strong style="color:#007BB5;">30 seconds</strong> — generating SOFA predictions,
                SHAP explainability &amp; clinical AI reports automatically.
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
        _clicked_patient = None
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
                    <p style="font-size:14px;font-weight:600;color:#c8dced;">{pcfg['description'].split(' — ')[0]}</p>
                    <p style="font-size:12px;color:#8ab8cc;margin-top:4px;">{desc}</p>
                    <div style="margin-top:14px;padding:9px 12px;background:rgba(0,0,0,0.35);
                                border:1px solid rgba(200,220,237,0.15);
                                border-radius:8px;font-size:12px;color:#c8dced;">
                        <b style="color:#7fb3c8;">Expected SOFA:</b>&nbsp;{sofa_range}&nbsp;&nbsp;|&nbsp;&nbsp;
                        <b style="color:#7fb3c8;">30 readings</b>&nbsp;× 10-min intervals
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(
                    f"▶  Start Monitoring Patient {pid}",
                    key=f"sel_{pid}",
                    use_container_width=True,
                    type="primary",
                ):
                    _clicked_patient = pid

        if _clicked_patient is not None:
            st.session_state.selected_patient = _clicked_patient
            st.session_state.row_indices[_clicked_patient] = 0
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
# READ CURRENT ROW FROM DATABASE (sample_patients table)
# =============================================================
patient_id  = st.session_state.selected_patient
patient_cfg = PATIENTS[patient_id]
row_idx     = st.session_state.row_indices[patient_id]

# -- Read current simulation row from database --
total_rows  = get_sample_patient_count(patient_id)
if total_rows == 0:
    st.error(
        "⚠️ Database not initialised. Run `python migrate_to_db.py` from `icu_monitor/` first."
    )
    st.stop()
cur_idx     = row_idx % total_rows
current_row = get_sample_patient_row(patient_id, cur_idx)

# Extract all inputs from the database row
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

# Seed patient vitals in DB if empty (first run for this patient)
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

# ── Stop Monitoring button — 3 columns to match patient selector's 3 columns ──
# IMPORTANT: must use st.columns(3) here so Streamlit widget reconciliation
# correctly maps old patient-card columns (3) to new columns (3), preventing
# the old patient cards from bleeding into the monitoring view layout.
_, _, _stop_btn_col = st.columns([5, 3, 2])
with _stop_btn_col:
    if st.button("⏹ Stop", key="stop_main_inline", use_container_width=True, type="secondary"):
        st.session_state.selected_patient = None
        st.rerun()

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
    for ann in fig.layout.annotations:
        ann.font.color = "#b8d0e0"

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
            "<span style='color:#5fda80;font-size:12px'>■ green band = normal range</span>"
        ),
        title_font_color="#e8f4ff",
        title_font_size=13,
        height=440,
        margin=dict(l=0, r=0, t=65, b=5),
        plot_bgcolor="rgba(10,22,40,0.95)",
        paper_bgcolor="rgba(10,22,40,0.0)",
        font=dict(color="#b8d0e0"),
    )
    fig.update_xaxes(
        title_text="Reading →", title_font_size=9,
        gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)",
        title_font_color="#8ab8cc", tickfont_color="#8ab8cc",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)",
        tickfont_color="#8ab8cc",
    )

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

CLINICAL_KEYWORDS = [
    "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP", "GCS", "stress",
    "hypotension", "respiratory", "mental", "septic", "failure",
    "intubated", "vasopressor", "shock", "fever", "infection",
    "oxygen", "ventilat", "cardiac", "renal", "hepatic",
]

# =============================================================
# PIPELINE
# =============================================================

# -- 1. SLIDING WINDOW (database) --
append_patient_vitals(patient_id, {
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "HR": HR, "RR": RR, "SpO2": SpO2, "Temp": Temp,
    "SBP": SBP, "DBP": DBP, "MAP": MAP,
})
vitals_df = get_patient_vitals(patient_id, limit=20)

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
# Clip to [-10, 10] — matches training (widened from -5 to preserve extremes).
final_scaled = np.clip(scaler.transform(final_df), -10, 10)

# -- 7. PREDICT --
X_tensor = torch.tensor(final_scaled, dtype=torch.float32)
model.eval()
with torch.no_grad():
    raw_pred = model(X_tensor).numpy().flatten()[0]

sofa_score = float(np.clip(raw_pred, 0, 24))
risk_text, risk_icon, risk_color = risk_label(sofa_score)

# ── SOFA Validation Checks (run before display) ──
_sofa_floor  = compute_sofa_floor(MAP, SpO2, GCS_eye)
_floor_ok    = sofa_score >= (_sofa_floor - 1)   # 1-pt tolerance for proxy imprecision
_vital_flags = vital_consistency_flags(HR, SBP, SpO2, MAP, sofa_score)
_prev_sofa, _sofa_jump, _traj_flagged, _traj_msg = check_trajectory(
    patient_id, sofa_score, HR, RR, SpO2, SBP, MAP
)
_conf_lo = round(max(0.0,  sofa_score - conformal_q_hat), 1)
_conf_hi = round(min(24.0, sofa_score + conformal_q_hat), 1)

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

# -- 8b. SAVE PREDICTION HISTORY (database) --
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
    "MAP":        MAP,
    "GCS Eye":    GCS_eye,
    "Stress":     stress,
})

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

final_prompt = f"""{_urgency_prefix}You are an ICU clinical decision support assistant.

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

# -- 12. LLM ASSESSMENT (1 main call) --
with st.spinner("Generating AI clinical assessment..."):
    try:
        responses     = get_multiple_llm_responses(GROQ_API_KEY, final_prompt, n=1)
        main_response = responses[0]
        llm_ok        = True
    except Exception as e:
        responses     = []
        main_response = f"⚠️ LLM unavailable: {e}"
        llm_ok        = False

# -- 12b. JUDGE LLM CALL (2nd call — G-Eval F1 + RAGAS Faithfulness F2) --
_judge_result = None
if llm_ok and GROQ_API_KEY:
    with st.spinner("Running AI quality validation (G-Eval + RAGAS faithfulness check)..."):
        _judge_result = call_judge_llm(
            GROQ_API_KEY, main_response,
            {"SpO2": SpO2, "MAP": MAP, "HR": HR, "RR": RR,
             "SBP": SBP, "stress": stress, "GCS_eye": GCS_eye},
            sofa_score,
        )

# -- 12c. COMPUTE 9-METHOD RELIABILITY --
_vitals_llm = {"SpO2": SpO2, "MAP": MAP, "HR": HR, "RR": RR,
               "SBP": SBP, "DBP": DBP, "Temp": Temp,
               "stress": stress, "GCS_eye": GCS_eye}
if llm_ok:
    consistency, _rel_breakdown, _rel_violations = compute_reliability_new(
        main_response, _vitals_llm, sofa_score, top_shap, _judge_result
    )
else:
    consistency, _rel_breakdown, _rel_violations = 0.0, {}, []

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
        <div style="background:rgba(14,31,54,0.95);border-radius:14px;padding:18px 20px;
                    border:1.5px solid #1e3a50;margin-bottom:10px;">
            <div style="font-size:11px;font-weight:700;color:#7fb3c8;letter-spacing:1px;
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

    # ── Held-out Risk Classification Metrics (AUROC / PR-AUC) ──
    _risk_metrics = _m.get("risk_classification_metrics")
    with st.expander("📈 Held-out Risk Classification Metrics (AUROC · PR-AUC · Confusion Matrix)"):
        if not _risk_metrics:
            st.info(
                "Metrics not yet generated. Run `python train_federated.py` to compute "
                "AUROC, PR-AUC, and confusion matrices on the held-out test set."
            )
        else:
            st.caption(
                _m.get("evaluation_limitation",
                       "Row-level split — patient IDs unavailable. Initial results only.")
            )
            _bands = _risk_metrics["risk_bands"]
            _alert = _risk_metrics["high_risk_alert"]

            # AUROC + PR-AUC headline tiles
            _r1, _r2, _r3, _r4 = st.columns(4)
            for _col, _lbl, _val, _tc, _bg, _brd in [
                (_r1, "AUROC",        _alert["auroc"],   "#64b5f6","rgba(21,101,192,0.2)","#1565c0"),
                (_r2, "PR-AUC",       _alert["pr_auc"],  "#ce93d8","rgba(106,27,154,0.2)","#6a1b9a"),
                (_r3, "Sensitivity",  _alert["sensitivity"], "#5fda80","rgba(46,125,50,0.2)","#2e7d32"),
                (_r4, "Specificity",  _alert["specificity"], "#ff8a65","rgba(230,81,0,0.2)","#e65100"),
            ]:
                with _col:
                    _disp = "N/A" if _val is None else f"{_val:.3f}"
                    st.markdown(f"""
<div style="background:{_bg};border:1.5px solid {_brd};border-radius:10px;
            padding:12px 8px;text-align:center;margin-bottom:8px;">
    <div style="font-size:9px;font-weight:700;color:{_tc};text-transform:uppercase;
                letter-spacing:0.7px;">{_lbl}</div>
    <div style="font-size:24px;font-weight:900;color:{_tc};line-height:1.2;">{_disp}</div>
</div>""", unsafe_allow_html=True)

            # False negative / positive counts
            st.markdown(
                f"**Alert threshold:** predicted SOFA ≥ {_alert['predicted_sofa_threshold']:.0f} &nbsp;·&nbsp; "
                f"**True high-risk:** actual SOFA ≥ {_alert['actual_high_risk_threshold']:.0f} &nbsp;·&nbsp; "
                f"Missed high-risk cases (FN): **{_alert['false_negative']:,}** &nbsp;·&nbsp; "
                f"False alerts (FP): **{_alert['false_positive']:,}**"
            )

            # Risk-band confusion matrix
            _bl = _bands["labels"]
            st.markdown("**Risk-band confusion matrix** — rows = actual class, columns = predicted class")
            st.dataframe(
                pd.DataFrame(_bands["confusion_matrix"], index=_bl, columns=_bl),
                use_container_width=True,
            )

            # Per-class metrics
            st.markdown("**Per-class metrics**")
            st.dataframe(
                pd.DataFrame.from_dict(_bands["per_class"], orient="index").round(3),
                use_container_width=True,
            )

            # Alert confusion matrix
            st.markdown("**High-risk alert confusion matrix**")
            st.dataframe(
                pd.DataFrame(
                    _alert["confusion_matrix"],
                    index=["Actual not high-risk", "Actual high-risk"],
                    columns=["Predicted no alert", "Predicted alert"],
                ),
                use_container_width=True,
            )

    # ── SOFA Prediction Validation Panel ──
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # Row 1: Conformal Prediction interval
    _conf_detail = (
        f"Predicted <b>{sofa_score:.1f}</b> &nbsp;·&nbsp; "
        f"90% coverage interval: "
        f"<b style='color:#5fda80;'>[{_conf_lo} – {_conf_hi}]</b>"
        f"<span style='color:#7fb3c8;font-size:11px;'> &nbsp;(±{conformal_q_hat} SOFA pts guaranteed)</span>"
    )

    # Row 2: Physiological lower-bound
    if _sofa_floor == 0:
        _floor_detail = (
            f"All measured vitals within normal ranges — no organ failure implied. "
            f"<span style='color:#5fda80;'>Floor = 0, prediction <b>{sofa_score:.1f}</b> consistent ✓</span>"
        )
        _floor_icon = "✅"
    elif _floor_ok:
        _floor_detail = (
            f"Vitals imply minimum SOFA ≥ <b>{_sofa_floor}</b> &nbsp;·&nbsp; "
            f"Predicted <b>{sofa_score:.1f}</b> &nbsp;"
            f"<span style='color:#5fda80;'>is physiologically consistent ✓</span>"
        )
        _floor_icon = "✅"
    else:
        _floor_detail = (
            f"Vitals imply minimum SOFA ≥ <b>{_sofa_floor}</b> &nbsp;·&nbsp; "
            f"Predicted <b>{sofa_score:.1f}</b> &nbsp;"
            f"<span style='color:#f0a500;'>may be underestimated ⚠ — consider clinical review</span>"
        )
        _floor_icon = "⚠️"

    # Row 3: Vital sign consistency
    if not _vital_flags:
        _vc_detail = "<span style='color:#5fda80;'>No combined vital-sign inconsistency detected ✓</span>"
        _vc_icon   = "✅"
    else:
        _vc_detail = "<br>".join(
            f"<span style='color:#f0a500;'>⚠ {f}</span>" for f in _vital_flags
        )
        _vc_icon = "⚠️"

    # Row 4: Trajectory coherence
    if _prev_sofa is None:
        _traj_detail = "<span style='color:#7fb3c8;'>Establishing baseline — first reading for this session</span>"
        _traj_icon   = "ℹ️"
    elif _traj_flagged:
        _traj_detail = f"<span style='color:#f0a500;'>⚠ {_traj_msg}</span>"
        _traj_icon   = "⚠️"
    else:
        _jump_str  = (f"({'+' if _sofa_jump >= 0 else ''}{_sofa_jump:.1f})"
                      if abs(_sofa_jump) > 0.05 else "(stable)")
        _traj_detail = (
            f"Prev: <b>{_prev_sofa:.1f}</b> → Now: <b>{sofa_score:.1f}</b> "
            f"<span style='color:#7fb3c8;'>{_jump_str}</span> &nbsp;·&nbsp; "
            f"<span style='color:#5fda80;'>Trajectory coherent ✓</span>"
        )
        _traj_icon = "✅"

    _val_rows = [
        ("📏", "Conformal Prediction",    _conf_detail,  "✅"),
        ("🧬", "Physiological Floor",     _floor_detail, _floor_icon),
        ("⚡", "Vital Sign Consistency",  _vc_detail,    _vc_icon),
        ("🔁", "Trajectory Coherence",    _traj_detail,  _traj_icon),
    ]
    _val_rows_html = ""
    for _vico, _vname, _vdetail, _vstatus in _val_rows:
        _is_warn   = "⚠️" in _vstatus
        _row_bord  = "#f0a500" if _is_warn else "#1e3a50"
        _row_bg    = "rgba(240,165,0,0.06)" if _is_warn else "rgba(14,28,48,0.7)"
        _val_rows_html += f"""
<div style="display:flex;align-items:flex-start;gap:12px;padding:10px 14px;
            border-left:3px solid {_row_bord};border-radius:7px;
            background:{_row_bg};margin-bottom:6px;">
    <div style="font-size:16px;min-width:22px;padding-top:1px;">{_vico}</div>
    <div style="flex:1;min-width:0;">
        <div style="font-size:10px;font-weight:800;color:#7fb3c8;letter-spacing:1.2px;
                    text-transform:uppercase;margin-bottom:4px;">{_vname}</div>
        <div style="font-size:12px;color:#c8dced;line-height:1.6;">{_vdetail}</div>
    </div>
    <div style="font-size:16px;flex-shrink:0;padding-top:2px;">{_vstatus}</div>
</div>"""

    st.markdown(f"""
<div style="background:rgba(6,14,28,0.97);border:1.5px solid #1e3a50;
            border-radius:12px;padding:16px 18px;margin-top:4px;">
    <div style="font-size:11px;font-weight:800;color:#00d2ff;letter-spacing:1.8px;
                text-transform:uppercase;margin-bottom:12px;border-bottom:1px solid #1e3a50;
                padding-bottom:8px;">
        🔬 SOFA Prediction Validation
    </div>
    {_val_rows_html}
    <div style="font-size:10px;color:#3a5a7a;margin-top:10px;padding-top:8px;
                border-top:1px solid #1a2e48;line-height:1.6;">
        Conformal calibration: FL training data (3 hospitals) &nbsp;·&nbsp;
        Physiological floor uses MAP, SpO₂, GCS Eye (SOFA components 1, 4, 5) &nbsp;·&nbsp;
        Trajectory threshold: SOFA Δ &gt; 4 with no vital change
    </div>
</div>""", unsafe_allow_html=True)

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
    st.markdown("<div style='font-size:18px;font-weight:800;color:#e8f4ff;margin-bottom:10px;border-left:3px solid #7fb3c8;padding-left:10px;'>📈 Vital Sign Trends</div>",
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
            f"<span style='font-size:13px;font-weight:700;color:#d0e0ec;min-width:60px;"
            f"display:inline-block;'>{vital}</span>"
            f"<span style='margin:0 6px;color:#4a6a8a;'>→</span>"
            f"<span style='background:{r_bg};color:{r_col};border:1.5px solid {r_col};"
            f"border-radius:5px;padding:2px 9px;font-size:11px;font-weight:800;"
            f"margin-right:5px;letter-spacing:0.3px;'>{r_lbl}</span>"
            f"<span style='background:rgba(20,40,65,0.9);border-radius:5px;padding:2px 9px;"
            f"font-size:11px;font-weight:700;color:{d_col};border:1px solid #2a4a6a;'>"
            f"{d_icon} {dirn.strip()}</span>"
        )

    _trend_html = ""
    for i, line in enumerate(trend_lines):
        _trend_html += f"<div style='padding:6px 10px;background:{'#fafafa' if i%2==0 else 'white'};" \
                       f"border-radius:6px;margin:3px 0;'>{_trend_badge(line)}</div>"

    tl, tr = st.columns(2)
    with tl:
        for i, line in enumerate(trend_lines[:4]):
            bg = "rgba(14,28,48,0.9)" if i % 2 == 0 else "rgba(10,20,38,0.9)"
            st.markdown(f"<div style='padding:8px 12px;background:{bg};"
                        f"border-radius:8px;margin:4px 0;border-left:3px solid #2a4a6a;'>"
                        f"{_trend_badge(line)}</div>", unsafe_allow_html=True)
    with tr:
        for i, line in enumerate(trend_lines[4:]):
            bg = "rgba(14,28,48,0.9)" if i % 2 == 0 else "rgba(10,20,38,0.9)"
            st.markdown(f"<div style='padding:8px 12px;background:{bg};"
                        f"border-radius:8px;margin:4px 0;border-left:3px solid #2a4a6a;'>"
                        f"{_trend_badge(line)}</div>", unsafe_allow_html=True)

    # ── Prediction History (colour-coded by risk) ──
    _ph = get_prediction_history(patient_id, limit=50)
    if len(_ph) > 1:
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            with st.expander(f"📋 Auto-Reading History  ({len(_ph)} readings, newest first)"):
                st.markdown(
                    f'<div style="font-size:11px;color:#6a9ab0;padding:4px 0 8px 0;">'
                    f'Each row = one 30-second reading for {patient_cfg["icon"]} '
                    f'{patient_cfg["name"]}. Alert threshold: SOFA ≥ {ALERT_THRESHOLD:.0f}. '
                    f'Keeps last 50.</div>',
                    unsafe_allow_html=True,
                )
                _ph_display = _ph.iloc[::-1].reset_index(drop=True)
                # Build dark HTML table (st.dataframe renders with white background)
                _cols_show = ["Timestamp", "Reading", "SOFA", "Risk", "HR", "RR",
                              "SpO₂", "Temp (°C)", "SBP", "MAP", "GCS Eye", "Stress"]
                _hdr = "".join(
                    f'<th style="padding:9px 12px;color:#7fb3c8;font-weight:700;'
                    f'font-size:11px;text-align:left;white-space:nowrap;">{c}</th>'
                    for c in _cols_show if c in _ph_display.columns
                )
                _rows_html = ""
                for _, _r in _ph_display.iterrows():
                    risk_v = str(_r.get("Risk", ""))
                    if "High" in risk_v:
                        row_bg = "rgba(220,53,69,0.12)"
                    elif "Moderate" in risk_v:
                        row_bg = "rgba(240,165,0,0.10)"
                    else:
                        row_bg = "rgba(40,167,69,0.07)"
                    _cells = "".join(
                        f'<td style="padding:7px 12px;font-size:11px;color:#c8dced;'
                        f'border-bottom:1px solid #1a3050;white-space:nowrap;">'
                        f'{_r[c]}</td>'
                        for c in _cols_show if c in _ph_display.columns
                    )
                    _rows_html += (
                        f'<tr style="background:{row_bg};">{_cells}</tr>'
                    )
                st.markdown(f"""
<div style="overflow-x:auto;border:1.5px solid #1e3a50;border-radius:10px;
            background:rgba(8,16,32,0.95);max-height:320px;overflow-y:auto;">
  <table style="width:100%;border-collapse:collapse;font-size:11px;">
    <thead>
      <tr style="background:rgba(0,210,255,0.08);border-bottom:2px solid #1e3a50;
                 position:sticky;top:0;">
        {_hdr}
      </tr>
    </thead>
    <tbody>{_rows_html}</tbody>
  </table>
</div>""", unsafe_allow_html=True)

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
        <div style="font-size:18px;font-weight:800;color:#e8f4ff;margin-bottom:6px;
                    border-left:3px solid #00d2ff;padding-left:10px;">
            🔬 SHAP Feature Impact Analysis
        </div>
        <div style="font-size:12px;color:#8ab8cc;margin-bottom:14px;">
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
            _dcol  = "#ff7b7b" if _inc else "#5fda80"
            _rbg   = "linear-gradient(90deg,rgba(220,53,69,0.14),rgba(220,53,69,0.04))" if _inc else "linear-gradient(90deg,rgba(40,167,69,0.14),rgba(40,167,69,0.04))"
            _dico  = "↑" if _inc else "↓"
            _dtxt  = "Increases Risk" if _inc else "Reduces Risk"
            _bw    = int(_abimp / _max_imp * 100)

            st.markdown(f"""
            <div style="background:{_rbg};border-left:5px solid {_bcol};border-radius:0 10px 10px 0;
                        padding:12px 16px;margin:5px 0;display:flex;align-items:center;gap:14px;
                        box-shadow:0 2px 6px rgba(0,0,0,0.25);">
                <div style="font-size:22px;line-height:1;">{_ico}</div>
                <div style="min-width:150px;flex-shrink:0;">
                    <div style="font-size:13px;font-weight:700;color:#e8f4ff;">{_lbl}</div>
                    <div style="font-size:11px;color:#8ab8cc;margin-top:2px;">{_val}</div>
                </div>
                <div style="flex:1;background:rgba(255,255,255,0.12);border-radius:6px;height:14px;overflow:hidden;min-width:80px;">
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
        <div style="font-size:18px;font-weight:800;color:#e8f4ff;margin-bottom:10px;
                    border-left:3px solid #00d2ff;padding-left:10px;">
            💡 Clinical Interpretations
        </div>
        """, unsafe_allow_html=True)

        _ci_left, _ci_right = st.columns(2)
        for _i, _exp in enumerate(clinical_explanations):
            _is_inc = "increasing risk" in _exp
            _ci_bg  = "linear-gradient(135deg,rgba(220,53,69,0.14),rgba(220,53,69,0.04))" if _is_inc else "linear-gradient(135deg,rgba(40,167,69,0.14),rgba(40,167,69,0.04))"
            _ci_brd = "#dc3545" if _is_inc else "#28a745"
            _ci_ico = "↑" if _is_inc else "↓"
            _ci_col = "#ff7b7b" if _is_inc else "#5fda80"
            _html = (
                f"<div style='background:{_ci_bg};border:1.5px solid {_ci_brd};"
                f"border-radius:8px;padding:10px 14px;margin:5px 0;"
                f"display:flex;align-items:flex-start;gap:10px;"
                f"box-shadow:0 2px 6px rgba(0,0,0,0.2);'>"
                f"<span style='font-size:18px;font-weight:900;color:{_ci_col};"
                f"line-height:1.2;flex-shrink:0;'>{_ci_ico}</span>"
                f"<span style='font-size:12px;color:#c8dced;line-height:1.5;'>{_exp}</span>"
                f"</div>"
            )
            (_ci_left if _i % 2 == 0 else _ci_right).markdown(_html, unsafe_allow_html=True)

        st.divider()

        # ─────────────────────────────────────────────────────────────
        # Section 3: Key Risk Factors (icon chips)
        # ─────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:18px;font-weight:800;color:#e8f4ff;margin-bottom:10px;
                    border-left:3px solid #00d2ff;padding-left:10px;">
            ⚠️ Key Risk Factors
        </div>
        """, unsafe_allow_html=True)

        if key_risks:
            _RISK_ICONS = {
                "Low oxygen levels":              ("💧", "#64b5f6", "rgba(21,101,192,0.2)",  "#1565c0"),
                "Respiratory distress":           ("🫁", "#ce93d8", "rgba(123,31,162,0.2)",  "#7b1fa2"),
                "Organ failure":                  ("🚨", "#ff7b7b", "rgba(183,28,28,0.2)",   "#b71c1c"),
                "Sepsis / Infection":             ("🦠", "#ffb74d", "rgba(230,81,0,0.2)",    "#e65100"),
                "Respiratory failure (intubated)":("😮‍💨", "#f48fb1", "rgba(136,14,79,0.2)",  "#880e4f"),
                "Haemodynamic instability":       ("💔", "#ff7b7b", "rgba(183,28,28,0.2)",   "#b71c1c"),
                "Hypotension":                    ("📉", "#b39ddb", "rgba(74,20,140,0.2)",   "#4a148c"),
                "Abnormal heart rate":            ("❤️", "#ef9a9a", "rgba(198,40,40,0.2)",   "#c62828"),
                "Altered mental status":          ("🧠", "#9fa8da", "rgba(26,35,126,0.2)",   "#1a237e"),
                "Neurological deterioration":     ("🧠", "#9fa8da", "rgba(26,35,126,0.2)",   "#1a237e"),
                "High physiological stress":      ("😰", "#ffb74d", "rgba(230,81,0,0.2)",    "#e65100"),
            }

            _chips_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:4px 0;">'
            for _risk in key_risks:
                _rico, _tcol, _rbg2, _rbrd = _RISK_ICONS.get(
                    _risk, ("⚠️", "#ff7b7b", "rgba(183,28,28,0.2)", "#b71c1c")
                )
                _chips_html += (
                    f"<div style='background:{_rbg2};border:2px solid {_rbrd};"
                    f"border-radius:10px;padding:10px 18px;"
                    f"display:inline-flex;align-items:center;gap:10px;"
                    f"box-shadow:0 2px 8px rgba(0,0,0,0.3);'>"
                    f"<span style='font-size:20px;'>{_rico}</span>"
                    f"<div>"
                    f"<div style='font-size:13px;font-weight:800;color:{_tcol};'>{_risk}</div>"
                    f"<div style='font-size:10px;color:#8ab8cc;'>SHAP-identified risk factor</div>"
                    f"</div></div>"
                )
            _chips_html += "</div>"
            st.markdown(_chips_html, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='padding:12px;background:rgba(40,167,69,0.12);border-radius:8px;"
                "border:1px solid #28a745;color:#5fda80;font-size:13px;'>"
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
    # icon, text_color (light, on dark bg), bg (dark semi-transparent), border (vivid), subtitle
    _SEC_CFG = {
        "CURRENT CONDITION":  ("📋", "#64b5f6", "rgba(21,101,192,0.18)",  "#1565c0",
                               "Patient status right now"),
        "PROBABLE CAUSE":     ("🔍", "#ff8a65", "rgba(191,54,12,0.18)",   "#bf360c",
                               "Why this is happening"),
        "RISK FORECAST":      ("🔮", "#ce93d8", "rgba(106,27,154,0.18)",  "#6a1b9a",
                               "Predicted trajectory if untreated"),
        "IMMEDIATE ACTIONS":  ("🚨", "#ef9a9a", "rgba(183,28,28,0.18)",   "#b71c1c",
                               "Critical interventions — next 30 minutes"),
    }

    # ── Reliability palette ──
    _rel_col = "#5fda80" if consistency >= 0.80 else "#ffc93c" if consistency >= 0.60 else "#ff7b7b"
    _rel_bg  = ("rgba(40,167,69,0.15)"  if consistency >= 0.80 else
                "rgba(240,165,0,0.15)"  if consistency >= 0.60 else
                "rgba(220,53,69,0.15)")
    _rel_brd = "#28a745" if consistency >= 0.80 else "#f0a500" if consistency >= 0.60 else "#dc3545"
    _rel_ico = "✅" if consistency >= 0.80 else "⚠️" if consistency >= 0.60 else "❌"
    _rel_lbl = ("High Reliability"     if consistency >= 0.80 else
                "Moderate Reliability" if consistency >= 0.60 else "Low Reliability")
    _rel_msg = (
        "Response passes all 8 validation checks — factually grounded, clinically sound, and coherent."
        if consistency >= 0.80 else
        "Response passes most checks with minor gaps — review highlighted violations before acting."
        if consistency >= 0.60 else
        "Significant validation failures detected — apply full clinical judgment before acting."
    )

    def _resp_valid(r):
        return r and r != _FALLBACK_MSG and len(r) > 80

    # ── 8-Method Reliability Banner ──
    # Top summary bar
    st.markdown(f"""
    <div style="background:{_rel_bg};border:2px solid {_rel_brd};border-radius:14px;
                padding:16px 20px;margin-bottom:12px;box-shadow:0 3px 16px rgba(0,0,0,0.35);">
        <div style="display:flex;justify-content:space-between;align-items:center;
                    flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:10px;font-weight:700;color:{_rel_col};letter-spacing:1.5px;
                            text-transform:uppercase;margin-bottom:4px;">
                    🧠 AI Clinical Assessment — 9-Method Validation
                </div>
                <div style="font-size:20px;font-weight:900;color:{_rel_col};margin-bottom:5px;">
                    {_rel_ico} {_rel_lbl}
                </div>
                <div style="font-size:12px;color:#b8d0e0;max-width:460px;line-height:1.5;">
                    {_rel_msg}
                </div>
                <div style="font-size:10px;color:#4a6a8a;margin-top:6px;">
                    6 deterministic checks + G-Eval (LLM judge) + RAGAS Faithfulness &nbsp;·&nbsp; 2 LLM calls total
                </div>
            </div>
            <div style="text-align:center;min-width:90px;">
                <div style="font-size:10px;color:#7fb3c8;text-transform:uppercase;
                            letter-spacing:0.6px;margin-bottom:2px;">Reliability</div>
                <div style="font-size:52px;font-weight:900;color:{_rel_col};line-height:1;">
                    {consistency:.2f}</div>
                <div style="font-size:10px;color:#7fb3c8;">out of 1.00</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 8-component breakdown grid
    if _rel_breakdown:
        _METHOD_META = {
            "Factual Grounding":       ("📋", "Vital abnormalities acknowledged?",        0.13),
            "Clinical Hard Rules":     ("⚖️", "ICU protocol requirements met?",           0.13),
            "SHAP Coherence":          ("🔬", "Addresses model's top predictors?",         0.09),
            "Response Structure":      ("📑", "All 4 required sections present?",          0.09),
            "Severity Calibration":    ("🎚️", "Urgency matches SOFA level?",               0.08),
            "Contraindication Check":  ("🚫", "No dangerous recommendations?",             0.05),
            "Numeric Accuracy":        ("🔢", "No hallucinated vital-sign numbers?",       0.08),
            "G-Eval Clinical Quality": ("🤖", "LLM judge: clinical quality score",         0.13),
            "RAGAS Faithfulness":      ("🔍", "LLM judge: statements vs. vitals",          0.22),
        }
        _bd_rows = ""
        for _mname, _mscore in _rel_breakdown.items():
            _mico, _mdesc, _mwt = _METHOD_META.get(_mname, ("•", _mname, 0.0))
            _is_ok   = _mscore >= 0.70
            _is_warn = 0.40 <= _mscore < 0.70
            _mcol  = "#5fda80" if _is_ok else "#ffc93c" if _is_warn else "#ff7b7b"
            _mbg   = ("rgba(40,167,69,0.08)"  if _is_ok else
                      "rgba(240,165,0,0.08)"  if _is_warn else
                      "rgba(220,53,69,0.08)")
            _mbrd  = "#1e3a50" if _is_ok else "#f0a500" if _is_warn else "#dc3545"
            _stato = "✅" if _is_ok else "⚠️" if _is_warn else "❌"
            _bar_w = int(_mscore * 120)
            _bd_rows += f"""
<div style="display:flex;align-items:center;gap:14px;padding:13px 18px;
            background:{_mbg};border-left:4px solid {_mbrd};border-radius:8px;
            margin-bottom:7px;">
    <div style="font-size:20px;min-width:26px;">{_mico}</div>
    <div style="flex:1;min-width:0;">
        <div style="font-size:13px;font-weight:700;color:#e0eeff;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                    margin-bottom:3px;">{_mname}</div>
        <div style="font-size:11px;color:#5a7a9a;">{_mdesc}
            &nbsp;·&nbsp; <span style="color:#3a5a7a;">weight {int(_mwt*100)}%</span>
        </div>
    </div>
    <div style="background:#0a1628;border-radius:5px;width:120px;height:10px;
                overflow:hidden;flex-shrink:0;">
        <div style="background:{_mcol};width:{_bar_w}px;height:10px;border-radius:5px;"></div>
    </div>
    <div style="font-size:15px;font-weight:900;color:{_mcol};min-width:40px;text-align:right;">
        {_mscore:.2f}</div>
    <div style="font-size:18px;flex-shrink:0;">{_stato}</div>
</div>"""

        st.markdown(f"""
<div style="background:rgba(6,14,28,0.97);border:1.5px solid #1e3a50;
            border-radius:14px;padding:18px 20px;margin-bottom:14px;">
    <div style="font-size:11px;font-weight:800;color:#00d2ff;letter-spacing:1.8px;
                text-transform:uppercase;margin-bottom:12px;border-bottom:1px solid #1e3a50;
                padding-bottom:9px;">📊 Validation Breakdown — 9 Methods</div>
    {_bd_rows}
</div>""", unsafe_allow_html=True)

    # Violation list (if any)
    if _rel_violations:
        _viol_html = "".join(
            f"<div style='padding:10px 14px;background:rgba(220,53,69,0.08);"
            f"border-left:4px solid #dc3545;border-radius:6px;margin-bottom:6px;"
            f"font-size:13px;color:#ff9090;'>"
            f"⚠ {v}</div>"
            for v in _rel_violations
        )
        st.markdown(f"""
<div style="background:rgba(6,14,28,0.97);border:1.5px solid #dc3545;
            border-radius:12px;padding:14px 16px;margin-bottom:12px;">
    <div style="font-size:10px;font-weight:800;color:#ff7b7b;letter-spacing:1.5px;
                text-transform:uppercase;margin-bottom:10px;">⚠ Validation Violations</div>
    {_viol_html}
</div>""", unsafe_allow_html=True)

    # ── Parse and render the main response as section cards ──
    def _parse_llm_sections(text):
        """
        Split LLM text into {SECTION_NAME: content} dict.
        Handles both numbered ("1. CURRENT CONDITION") and bare ("CURRENT CONDITION")
        section headers, with or without ** bold markers or trailing punctuation.
        """
        _NAMES = [
            "IMMEDIATE ACTIONS", "CURRENT CONDITION",
            "PROBABLE CAUSE",    "RISK FORECAST",
        ]
        positions = []
        for sec in _NAMES:
            # Optional number + optional ** + section name + optional ** + optional trailing char
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

    # ── Section content renderer — clean text + bullets ──────────────────────
    def _render_section_html(content, section_name):
        """
        Render LLM section content as clean, readable text with bullet points.
        Uses HTML escaping to prevent medical symbols (<65, >90%) from breaking layout.
        Section headers stay styled; content inside is clean text — readable by
        both clinical staff and patients.
        """
        import html as _html_mod
        sn    = section_name.upper()
        lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
        parts = []

        def _safe(t):
            """HTML-escape raw text, then apply safe inline styling."""
            t = _html_mod.escape(t)
            t = re.sub(r'\*\*(.*?)\*\*',
                       r'<strong style="color:#e8f4ff;">\1</strong>', t)
            t = t.replace('→', '<span style="color:#5a8aaa;font-weight:600;"> → </span>')
            t = t.replace('↑', '<span style="color:#ff9090;font-weight:700;">↑</span>')
            t = t.replace('↓', '<span style="color:#7ec8e3;font-weight:700;">↓</span>')
            t = re.sub(r'\s+—\s+→', ' → ', t)  # clean up "— →" artifact
            return t

        for line in lines:
            is_bullet   = line.startswith(('•', '-', '·'))
            is_priority = '⚡' in line
            is_domain   = (
                not is_bullet
                and line.endswith(':')
                and len(line) < 55
                and re.match(r'^[A-Z][A-Za-z ,&/]+:$', line)
            )

            # ── Domain subheader: "Respiratory:", "Hemodynamics:", etc. ─────
            if is_domain:
                label = _html_mod.escape(line.rstrip(':'))
                parts.append(
                    f'<div style="margin:18px 0 10px 0;padding:7px 16px;'
                    f'background:rgba(0,210,255,0.07);border-left:3px solid #00d2ff;'
                    f'border-radius:0 6px 6px 0;">'
                    f'<span style="font-size:12px;font-weight:800;color:#00d2ff;'
                    f'letter-spacing:1.8px;text-transform:uppercase;">{label}</span>'
                    f'</div>'
                )

            # ── ⚡ Priority action ───────────────────────────────────────────
            elif is_priority:
                text = re.sub(r'^[•·\-\s]*⚡\s*', '', line)
                parts.append(
                    f'<div style="background:rgba(255,200,30,0.10);border-left:4px solid #ffc93c;'
                    f'border-radius:7px;padding:13px 18px;margin:10px 0;">'
                    f'<div style="font-size:11px;font-weight:800;color:#ffc93c;'
                    f'letter-spacing:1.2px;text-transform:uppercase;margin-bottom:6px;">⚡ Priority Action</div>'
                    f'<div style="font-size:15px;color:#ffe5a0;font-weight:600;line-height:1.6;">{_safe(text)}</div>'
                    f'</div>'
                )

            # ── Bullet points (explicit •) ────────────────────────────────────
            elif is_bullet:
                text = re.sub(r'^[•·\-]\s*', '', line)
                dot_col = '#f0a500' if 'RISK' in sn else '#5fda80' if 'CURRENT' in sn else '#00d2ff'
                parts.append(
                    f'<div style="display:flex;align-items:flex-start;gap:14px;'
                    f'padding:11px 6px;border-bottom:1px solid rgba(30,58,80,0.25);">'
                    f'<span style="color:{dot_col};font-size:20px;flex-shrink:0;'
                    f'line-height:1.0;margin-top:2px;">•</span>'
                    f'<span style="font-size:15px;color:#d0e8f8;line-height:1.65;">{_safe(text)}</span>'
                    f'</div>'
                )

            # ── Plain line — treat as a bullet to ensure consistent formatting ──
            else:
                # Non-bullet lines are rendered as bullet rows (the LLM sometimes
                # omits the leading • even when instructed to use it)
                dot_col = '#f0a500' if 'RISK' in sn else '#5fda80' if 'CURRENT' in sn else '#00d2ff'
                parts.append(
                    f'<div style="display:flex;align-items:flex-start;gap:14px;'
                    f'padding:11px 6px;border-bottom:1px solid rgba(30,58,80,0.25);">'
                    f'<span style="color:{dot_col};font-size:20px;flex-shrink:0;'
                    f'line-height:1.0;margin-top:2px;">•</span>'
                    f'<span style="font-size:15px;color:#d0e8f8;line-height:1.65;">{_safe(line)}</span>'
                    f'</div>'
                )

        return '\n'.join(parts)

    # ── Parse and render the main response as enhanced section cards ────────
    _sections = _parse_llm_sections(main_response)

    if _sections:
        for _sname, _scontent in _sections.items():
            _sico, _stcol, _ssbg, _ssbrd, _ssub = _SEC_CFG.get(
                _sname, ("📄", "#b8d0e0", "rgba(255,255,255,0.05)", "#4a6a8a", "")
            )
            # Section header
            st.markdown(f"""
            <div style="background:{_ssbg};border-left:6px solid {_ssbrd};
                        border-radius:0 10px 10px 0;padding:12px 16px;margin-top:18px;
                        box-shadow:0 2px 10px rgba(0,0,0,0.3);">
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:22px;">{_sico}</span>
                    <div>
                        <div style="font-size:13px;font-weight:800;color:{_stcol};
                                    letter-spacing:1px;text-transform:uppercase;">{_sname}</div>
                        <div style="font-size:10px;color:{_stcol};opacity:0.75;margin-top:2px;">{_ssub}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Clean LLM artifacts then render as enhanced HTML
            _clean = re.sub(r'\*{1,2}\s*$', '', _scontent.strip())
            _clean = re.sub(r'^\s*\*{1,2}\s*', '', _clean)
            _clean = re.sub(r'<br\s*/?>', '\n', _clean)
            _clean = re.sub(
                r'\n?\*?\*?(?:clinical\s+)?disclaimer[\s:*].*$',
                '', _clean, flags=re.IGNORECASE | re.DOTALL
            ).strip()

            _rendered = _render_section_html(_clean, _sname)
            st.markdown(
                f"""<div style="background:rgba(10,22,42,0.85);border:1.5px solid #1a3050;
                    border-radius:0 0 12px 12px;padding:18px 20px 14px 20px;
                    margin-bottom:6px;">
                    {_rendered}
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    else:
        _fb = re.sub(r'<br\s*/?>', '\n', main_response)
        st.markdown(_fb)

    # ── Judge evaluation detail expander ──
    if _judge_result:
        with st.expander("🤖 G-Eval + RAGAS Judge Details — expand to see AI evaluator output"):
            _jcols = st.columns(4)
            _jdims = [
                ("Factual Accuracy",        _judge_result.get("factual_accuracy", "—")),
                ("Clinical Appropriateness", _judge_result.get("clinical_appropriateness", "—")),
                ("Urgency Calibration",      _judge_result.get("urgency_calibration", "—")),
                ("Completeness",             _judge_result.get("completeness", "—")),
            ]
            for _jcol, (_jlbl, _jval) in zip(_jcols, _jdims):
                with _jcol:
                    _jcol_c = "#5fda80" if int(_jval or 3) >= 4 else "#ffc93c" if int(_jval or 3) == 3 else "#ff7b7b"
                    st.markdown(f"""
<div style="background:rgba(14,28,48,0.95);border:1.5px solid #1e3a50;border-radius:10px;
            padding:12px 8px;text-align:center;">
    <div style="font-size:9px;font-weight:700;color:#7fb3c8;text-transform:uppercase;
                letter-spacing:0.5px;">{_jlbl}</div>
    <div style="font-size:30px;font-weight:900;color:{_jcol_c};line-height:1.2;">{_jval}<span style="font-size:14px;color:#4a6a8a;">/5</span></div>
</div>""", unsafe_allow_html=True)

            _unsup = _judge_result.get("unsupported_claims", [])
            if _unsup:
                st.markdown("**Statements contradicted by patient vitals:**")
                for _uc in _unsup:
                    st.markdown(f"- ⚠ {_uc}")

            _danger = _judge_result.get("dangerous_details", "")
            if _danger:
                st.error(f"🚫 Dangerous recommendation detected: {_danger}")

            _judge_addrs = []
            if _judge_result.get("hypoxemia_addressed"):  _judge_addrs.append("Hypoxemia ✓")
            if _judge_result.get("hypotension_addressed"): _judge_addrs.append("Hypotension ✓")
            if _judge_result.get("tachycardia_addressed"): _judge_addrs.append("Tachycardia ✓")
            if _judge_addrs:
                st.caption(f"Semantically addressed: {' · '.join(_judge_addrs)}")

    # ── Clinical Disclaimer ──
    st.divider()
    st.markdown("""
    <div style="background:rgba(21,101,192,0.12);border:1.5px solid #1565c0;
                border-left:5px solid #64b5f6;border-radius:0 10px 10px 0;padding:14px 18px;">
        <div style="font-size:13px;font-weight:700;color:#64b5f6;margin-bottom:6px;">
            ⚕️ Clinical Disclaimer
        </div>
        <div style="font-size:12px;color:#c8dced;line-height:1.6;">
            This system is a <strong style="color:#e8f4ff;">decision support tool only</strong>.
            It does not diagnose disease or replace the clinical judgment of qualified healthcare
            professionals. All AI-generated outputs must be reviewed by a licensed clinician
            before any clinical action is taken.
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
        '<div style="font-size:11px;color:#7fb3c8;margin-top:4px;">→ Improved global model · Repeat for 100 rounds</div>'
        '</div></div>'

        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ── Training Configuration (coloured stat tiles) ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#e8f4ff;margin-bottom:12px;"
                "border-left:3px solid #007BB5;padding-left:10px;'>⚙️ Training Configuration</div>",
                unsafe_allow_html=True)

    # (label, value, text_color, bg, border, subtitle)
    _cfg_tiles = [
        ("FL Rounds",       str(m["num_rounds"]),       "#64b5f6","rgba(21,101,192,0.2)","#1565c0","rounds of federation"),
        ("Hospitals",        str(m["hospitals"]),         "#ce93d8","rgba(106,27,154,0.2)","#6a1b9a","ICU sites"),
        ("Epochs / Round",   str(m["epochs_per_round"]), "#5fda80","rgba(46,125,50,0.2)", "#2e7d32","local training epochs"),
        ("Aggregation",      m["aggregation"],            "#ff8a65","rgba(230,81,0,0.2)",  "#e65100","weight averaging method"),
        ("Training Samples", f"{m['train_samples']:,}",  "#4db6ac","rgba(0,105,92,0.2)",  "#00695c","real ICU patients"),
        ("Test Samples",     f"{m['test_samples']:,}",    "#aed581","rgba(85,139,47,0.2)", "#558b2f","held-out patients"),
        ("Best Round",       str(m["best_round"]),        "#ffb74d","rgba(245,127,23,0.2)","#f57f17","lowest eval loss"),
        ("Split Type",       m["split_type"],             "#b39ddb","rgba(69,39,160,0.2)", "#4527a0","data distribution"),
    ]

    _t1, _t2, _t3, _t4 = st.columns(4)
    _t5, _t6, _t7, _t8 = st.columns(4)
    for _cols, _tiles in [([_t1,_t2,_t3,_t4], _cfg_tiles[:4]),
                           ([_t5,_t6,_t7,_t8], _cfg_tiles[4:])]:
        for _col, (_lbl, _val, _tc, _bg, _brd, _sub) in zip(_cols, _tiles):
            with _col:
                st.markdown(f"""
                <div style="background:{_bg};border:1.5px solid {_brd};
                            border-radius:10px;padding:14px 12px;text-align:center;
                            box-shadow:0 2px 10px rgba(0,0,0,0.3);margin-bottom:8px;">
                    <div style="font-size:9px;font-weight:700;color:{_tc};text-transform:uppercase;
                                letter-spacing:0.6px;">{_lbl}</div>
                    <div style="font-size:26px;font-weight:900;color:{_tc};line-height:1.1;
                                margin:4px 0;">{_val}</div>
                    <div style="font-size:9px;color:#8ab8cc;">{_sub}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ── Global Model Performance ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#e8f4ff;margin-bottom:12px;"
                "border-left:3px solid #007BB5;padding-left:10px;'>📊 Global Model Performance</div>",
                unsafe_allow_html=True)

    _perf_tiles = [
        ("Mean Abs. Error",  f"{m['final_mae']:.3f}", "SOFA points", "#64b5f6","rgba(21,101,192,0.2)","#1565c0",
         "Average prediction error on held-out ICU patients"),
        ("R² Score",         f"{m['final_r2']:.3f}",  "variance",    "#5fda80","rgba(46,125,50,0.2)","#2e7d32",
         "Proportion of variance explained by the model"),
        ("Pred Range Min",   f"{m['pred_range_min']:.1f}", "SOFA",   "#4db6ac","rgba(0,105,92,0.2)","#00695c",
         "Lowest predicted SOFA across test set"),
        ("Pred Range Max",   f"{m['pred_range_max']:.1f}", "SOFA",   "#ff8a65","rgba(230,81,0,0.2)","#e65100",
         "Highest predicted SOFA across test set"),
    ]

    _p1, _p2, _p3, _p4 = st.columns(4)
    for _col, (_lbl, _val, _unit, _tc, _bg, _brd, _desc) in zip([_p1,_p2,_p3,_p4], _perf_tiles):
        with _col:
            st.markdown(f"""
            <div style="background:{_bg};border:2px solid {_brd};
                        border-radius:12px;padding:18px 12px;text-align:center;
                        box-shadow:0 3px 12px rgba(0,0,0,0.3);">
                <div style="font-size:9px;font-weight:700;color:{_tc};text-transform:uppercase;
                            letter-spacing:0.7px;margin-bottom:4px;">{_lbl}</div>
                <div style="font-size:34px;font-weight:900;color:{_tc};line-height:1;">{_val}</div>
                <div style="font-size:10px;color:#8ab8cc;">{_unit}</div>
                <div style="font-size:9px;color:#6a8aaa;margin-top:6px;line-height:1.4;">{_desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Model Architecture ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#e8f4ff;margin-bottom:12px;"
                "border-left:3px solid #007BB5;padding-left:10px;'>🧠 Model Architecture (PyTorch DNN)</div>",
                unsafe_allow_html=True)

    _layers = [
        ("INPUT",  "108 features", "18 vitals (trends + latest + GCS) + 90 SOFA-vocab TF-IDF", "#64b5f6","rgba(21,101,192,0.2)"),
        ("Linear", "108 → 128",   "Fully connected · ReLU activation",                         "#5fda80","rgba(46,125,50,0.2)"),
        ("Linear", "128 →  64",   "Fully connected · ReLU activation",                         "#5fda80","rgba(46,125,50,0.2)"),
        ("Linear", " 64 →  32",   "Fully connected · ReLU activation",                         "#5fda80","rgba(46,125,50,0.2)"),
        ("Linear", " 32 →   1",   "Output layer · No activation (regression)",                 "#ce93d8","rgba(106,27,154,0.2)"),
        ("OUTPUT", "SOFA (0–24)", "Predicted SOFA score · clip(0, 24)",                        "#ff8a65","rgba(230,81,0,0.2)"),
    ]

    _arch_html = ""
    for _i, (_ltype, _ldim, _ldesc, _tc, _bg) in enumerate(_layers):
        _arrow = "<div style='text-align:center;font-size:18px;color:#4a6a8a;margin:2px 0;'>↓</div>" if _i < len(_layers)-1 else ""
        _arch_html += f"""
        <div style="background:{_bg};border:1.5px solid {_tc};border-radius:8px;
                    padding:10px 16px;display:flex;align-items:center;gap:12px;">
            <div style="background:{_tc};color:#0a1628;border-radius:5px;padding:3px 8px;
                        font-size:10px;font-weight:800;letter-spacing:0.5px;
                        white-space:nowrap;">{_ltype}</div>
            <div style="font-size:14px;font-weight:700;color:{_tc};min-width:80px;">{_ldim}</div>
            <div style="font-size:11px;color:#9ab8cc;">{_ldesc}</div>
        </div>{_arrow}"""

    _la, _lb = st.columns([3, 2])
    with _la:
        st.markdown(_arch_html, unsafe_allow_html=True)
    with _lb:
        st.markdown(f"""
        <div style="background:rgba(14,28,48,0.95);border:1.5px solid #1e3a50;border-radius:12px;
                    padding:16px;font-size:12px;color:#c8dced;line-height:1.8;height:100%;">
            <div style="font-size:13px;font-weight:700;color:#b8d0e0;margin-bottom:8px;">
                🔧 Training Details
            </div>
            <b style="color:#7fb3c8;">Optimizer:</b> AdamW (weight_decay=1e-4)<br>
            <b style="color:#7fb3c8;">Loss:</b> Weighted MSE — high-SOFA patients get up to 4.7× gradient weight<br>
            <b style="color:#7fb3c8;">No Dropout</b> — causes FL divergence; regularised by AdamW instead<br>
            <b style="color:#7fb3c8;">Parameters:</b> ~23,000 total trainable weights<br>
            <b style="color:#7fb3c8;">FL Framework:</b> Flower (flwr) · FedYogi + FedProx<br>
            <b style="color:#7fb3c8;">Best round:</b> {m['best_round']} of {m['num_rounds']}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Differential Privacy ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#e8f4ff;margin-bottom:12px;"
                "border-left:3px solid #007BB5;padding-left:10px;'>🔐 Differential Privacy</div>",
                unsafe_allow_html=True)

    if m.get("differential_privacy"):
        st.markdown(f"""
        <div style="background:rgba(46,125,50,0.15);border:2px solid #4caf50;
                    border-left:6px solid #2e7d32;border-radius:0 12px 12px 0;padding:16px 20px;">
            <div style="font-size:14px;font-weight:800;color:#5fda80;margin-bottom:8px;">
                ✅ Differential Privacy ENABLED
            </div>
            <div style="display:flex;gap:20px;flex-wrap:wrap;font-size:12px;color:#c8dced;">
                <span><b style="color:#7fb3c8;">σ (noise multiplier):</b> {m['dp_sigma']}</span>
                <span><b style="color:#7fb3c8;">S (sensitivity):</b> {m['dp_sensitivity']}</span>
                <span><b style="color:#7fb3c8;">ε (privacy budget):</b> ≈{m['dp_epsilon']}</span>
                <span><b style="color:#7fb3c8;">δ:</b> {m['dp_delta']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        _dp_steps = [
            ("①", "Compute update", "local_weights − global_weights", "#64b5f6"),
            ("②", "Clip L2 norm",   "Bounds any patient's max influence (sensitivity S)", "#ff8a65"),
            ("③", "Add noise",      "Gaussian N(0, (σ·S)²) to every weight parameter", "#ce93d8"),
            ("④", "Transmit",       "Server receives noisy update — cannot trace individuals", "#5fda80"),
        ]
        st.markdown("""
        <div style="background:rgba(240,165,0,0.12);border:1.5px solid #f0a500;
                    border-left:5px solid #f0a500;border-radius:0 12px 12px 0;
                    padding:14px 18px;margin-bottom:12px;">
            <div style="font-size:13px;font-weight:700;color:#ffc93c;">
                ⚙️ Differential Privacy: Disabled in current build
            </div>
            <div style="font-size:11px;color:#b8d0e0;margin-top:4px;">
                Model trained with plain FedAvg (no noise). Enable: set
                <code style="background:rgba(0,0,0,0.4);padding:1px 5px;border-radius:3px;">USE_DP = True</code>
                in train_federated.py and retrain.
            </div>
        </div>
        """, unsafe_allow_html=True)

        _dp_html = '<div style="display:flex;flex-direction:column;gap:8px;">'
        for _step, _title, _desc, _tc in _dp_steps:
            _dp_html += (
                f"<div style='display:flex;align-items:flex-start;gap:12px;"
                f"background:rgba(8,16,32,0.95);border:1px solid #1e3a50;"
                f"border-radius:8px;padding:10px 14px;'>"
                f"<div style='background:{_tc};color:#0a1628;border-radius:50%;width:24px;height:24px;"
                f"display:flex;align-items:center;justify-content:center;font-size:11px;"
                f"font-weight:800;flex-shrink:0;'>{_step}</div>"
                f"<div><div style='font-size:12px;font-weight:700;color:{_tc};'>{_title}</div>"
                f"<div style='font-size:11px;color:#8ab8cc;margin-top:2px;'>{_desc}</div></div>"
                f"</div>"
            )
        _dp_html += "</div>"
        st.markdown(
            "<div style='font-size:12px;font-weight:700;color:#b8d0e0;margin-bottom:8px;'>What DP would add:</div>",
            unsafe_allow_html=True
        )
        st.markdown(_dp_html, unsafe_allow_html=True)

    st.divider()

    # ── Feature Vector Breakdown (visual bars) ──
    st.markdown("<div style='font-size:18px;font-weight:800;color:#e8f4ff;margin-bottom:12px;"
                "border-left:3px solid #007BB5;padding-left:10px;'>🔢 Feature Vector Breakdown — 108 total</div>",
                unsafe_allow_html=True)

    _feat_rows = [
        ("📈 Trend Vitals",         9,  108, "#64b5f6","rgba(21,101,192,0.18)","#1565c0",
         "HR_mean, HR_std, RR_mean, SpO₂_mean, SpO₂_min, Temp_mean, SBP_mean, DBP_mean, MAP_mean"),
        ("📊 Latest Vitals",         7,  108, "#5fda80","rgba(46,125,50,0.18)","#2e7d32",
         "latest_HR, latest_RR, latest_SpO₂, latest_Temp, latest_SBP, latest_DBP, latest_MAP"),
        ("👁️ Computer Vision",       2,  108, "#ce93d8","rgba(106,27,154,0.18)","#6a1b9a",
         "GCS Eye Opening (1–4 scale), Stress Score (0–10)"),
        ("📝 Clinical NLP (TF-IDF)", 90, 108, "#ff8a65","rgba(230,81,0,0.18)","#e65100",
         "90 SOFA-vocabulary whitelist terms: creatinine, bilirubin, vasopressor, intubated, sepsis … directly mapped to 6 SOFA organ components"),
    ]

    for _flabel, _fcount, _ftotal, _ftc, _fbg, _fbrd, _fex in _feat_rows:
        _fw = _fcount / _ftotal * 100
        st.markdown(f"""
        <div style="background:{_fbg};border-left:5px solid {_fbrd};border-radius:0 10px 10px 0;
                    padding:12px 16px;margin:6px 0;box-shadow:0 2px 8px rgba(0,0,0,0.3);">
            <div style="display:flex;justify-content:space-between;align-items:center;
                        margin-bottom:6px;">
                <div style="font-size:13px;font-weight:700;color:{_ftc};">{_flabel}</div>
                <div style="font-size:14px;font-weight:900;color:{_ftc};">
                    {_fcount} <span style="font-size:10px;color:#8ab8cc;">/ 108 features ({_fw:.1f}%)</span>
                </div>
            </div>
            <div style="background:rgba(255,255,255,0.1);border-radius:6px;height:10px;margin-bottom:6px;overflow:hidden;">
                <div style="width:{_fw:.0f}%;background:{_fbrd};height:100%;border-radius:6px;"></div>
            </div>
            <div style="font-size:10px;color:#8ab8cc;">{_fex}</div>
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

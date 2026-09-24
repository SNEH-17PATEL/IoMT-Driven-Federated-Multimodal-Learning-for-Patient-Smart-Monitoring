"""
Federated Learning Simulation Dashboard
=========================================
Standalone Streamlit app — run separately from app.py:

    cd icu_monitor
    streamlit run fl_dashboard.py

Simulates what happens when you run server.py + 3 × client.py in separate
terminals, but shown live and visually in a browser.

Each FL round has 6 phases:
  ① Distribute  — server sends global weights → all hospitals
  ② Train H0    — Hospital 0 trains locally (3 epochs, FedProx optional)
  ③ Train H1    — Hospital 1 trains locally
  ④ Train H2    — Hospital 2 trains locally
  ⑤ Aggregate   — server blends all three weight sets (FedAvg or FedYogi)
  ⑥ Complete    — metrics logged, best model tracked
"""

import os
import time
import threading
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split

from model_utils import ICUModel, train_model, evaluate_model, get_weights, set_weights

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FL Simulation — ICU CDSS",
    page_icon="🔬",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
# ── Anti-flash: inject dark background BEFORE any Streamlit content ──────────
# Without this, Streamlit shows a white flash every time the page auto-reruns.
st.markdown(
    "<style>"
    "html,body{background:#0a1628!important;background-color:#0a1628!important;"
    "color:#d0e0ec!important;}"
    "</style>",
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* ══ BASE — force dark on every rerun ══════════════════════════════════ */
html, body { background: #0a1628 !important; color: #d0e0ec !important; }
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background: #0a1628 !important;
    color: #d0e0ec !important;
}
.stApp > header { background: #0a1628 !important; }
section[data-testid="stSidebar"] { background: #0d1e30 !important; }
[data-testid="block-container"] { padding-top: 1rem; }

/* ── Prevent white flash from Streamlit's progress / spinner ─────────── */
[data-testid="stSpinner"] > div,
[data-testid="stStatusWidget"],
.element-container { background: transparent !important; }

/* ── iframe (components.html) — seamless, no border, no hover tooltip ── */
iframe { border: none !important; outline: none !important;
         background: #0a1628 !important; }
[data-testid="stCustomComponentV1"] {
    margin: 0 !important; padding: 0 !important; background: #0a1628 !important;
}
/* Hide the "st.iframe" browser tooltip that appears on hover */
[data-testid="stCustomComponentV1"] iframe[title] {
    pointer-events: auto;
}
[data-testid="stCustomComponentV1"]::after { display: none !important; }
[data-testid="stIFrame"] { background: #0a1628 !important; }

/* ── All generic Streamlit text — always light ───────────────────────── */
div[data-testid="stMarkdownContainer"] { color: #d0e0ec !important; }
div[data-testid="stMarkdownContainer"] p   { color: #d0e0ec !important; }
div[data-testid="stMarkdownContainer"] li  { color: #d0e0ec !important; }
div[data-testid="stMarkdownContainer"] span { color: inherit; }
div[data-testid="stMarkdownContainer"] strong,
div[data-testid="stMarkdownContainer"] b   { color: #e8f4ff !important; }
div[data-testid="stMarkdownContainer"] code {
    background: rgba(0,210,255,0.1) !important;
    color: #7fb3c8 !important; border-radius: 4px; padding: 2px 6px;
}

/* ── Widget labels ───────────────────────────────────────────────────── */
label[data-testid="stWidgetLabel"] p,
.stSlider  [data-testid="stWidgetLabel"] p,
.stSelectbox [data-testid="stWidgetLabel"] p { color: #7fb3c8 !important; }

/* ── Expanders ───────────────────────────────────────────────────────── */
details {
    background: rgba(12,26,46,0.9) !important;
    border-radius: 10px !important; border: 1px solid #1e3a50 !important;
}
/* Header / summary row */
details summary {
    background: rgba(0,210,255,0.06) !important;
    border-radius: 10px !important; padding: 10px 14px !important;
    color: #7fb3c8 !important; font-weight: 700 !important;
    font-size: 13px !important; letter-spacing: 0.2px !important;
}
details summary:hover { color: #00d2ff !important; background: rgba(0,210,255,0.1) !important; }
details summary p,
details summary span,
details summary div { color: #7fb3c8 !important; }
/* Expander body */
div[data-testid="stExpanderDetails"] {
    padding: 12px 16px !important;
}
div[data-testid="stExpanderDetails"] p,
div[data-testid="stExpanderDetails"] li  { color: #c8dced !important; }
div[data-testid="stExpanderDetails"] strong,
div[data-testid="stExpanderDetails"] b   { color: #e8f4ff !important; }
div[data-testid="stExpanderDetails"] code {
    background: rgba(0,210,255,0.12) !important;
    color: #7fb3c8 !important; border-radius: 4px; padding: 2px 6px;
}

/* ── Dividers ────────────────────────────────────────────────────────── */
hr { margin: 8px 0 !important; border-color: #1e3a50 !important;
     border-top: 1px solid #1e3a50 !important; }

/* ── Vertical gap reduction ──────────────────────────────────────────── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] { gap: 0.4rem; }

/* ── Progress bar text ───────────────────────────────────────────────── */
[data-testid="stProgressBarMessage"] { color: #a8c8d8 !important; font-size: 12px !important; }
[data-testid="stCaption"] p { color: #8ab0c8 !important; font-size: 11px !important; }
small { color: #8ab0c8 !important; }

/* ── Force dark on any remaining white containers ────────────────────── */
/* Selectbox, text input, number input */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] { background: rgba(10,22,40,0.95) !important; }
[data-baseweb="select"] * { color: #d0e0ec !important; }
/* Selectbox dropdown list */
[data-baseweb="popover"] { background: #0e1f36 !important; }
[data-baseweb="popover"] li { color: #d0e0ec !important; }
[data-baseweb="popover"] li:hover { background: rgba(0,210,255,0.1) !important; }
/* Dataframe container — even if we replace with HTML, belt+suspenders */
[data-testid="stDataFrameResizable"],
[data-testid="stDataFrame"] { background: #0a1628 !important; }
iframe.stDataFrame { background: #0a1628 !important; }

/* ── Alert/error ─────────────────────────────────────────────────────── */
div[data-testid="stAlert"] > div { border-radius: 10px !important; }

/* ── Primary button — dark text on cyan background ───────────────────── */
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
button[data-testid="baseButton-secondary"] {
    color: #c8dced !important;
    border-color: #2a4a6a !important;
}
button[data-testid="baseButton-secondary"] p,
button[data-testid="baseButton-secondary"] div[data-testid="stMarkdownContainer"] p {
    color: #c8dced !important;
}

/* ══ KEYFRAME ANIMATIONS ════════════════════════════════════════════════ */
@keyframes slideIn {
    from { opacity:0; transform:translateY(-6px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes blink {
    0%,100% { opacity:1; } 50% { opacity:0.35; }
}

/* ══ REUSABLE COMPONENT CLASSES ═════════════════════════════════════════ */
.fl-header {
    background: linear-gradient(135deg,#0f2027 0%,#203a43 50%,#2c5364 100%);
    border-left: 6px solid #00d2ff; border-radius: 14px;
    padding: 18px 26px; margin-bottom: 18px; animation: slideIn 0.3s ease;
}
.metric-tile {
    background: rgba(12,26,46,0.95); border: 1.5px solid #1e3a50;
    border-radius: 12px; padding: 14px 12px; text-align: center;
    transition: transform 0.1s;
}
.metric-tile:hover { transform: scale(1.02); }
.algo-banner {
    background: rgba(0,210,255,0.07); border: 1px solid #1e3a50;
    border-left: 4px solid #00d2ff; border-radius: 8px;
    padding: 8px 16px; margin-bottom: 12px; font-size: 12px; color: #7fb3c8;
}
.running-pill {
    display:inline-block; background:rgba(0,210,255,0.15);
    border:1.5px solid #00d2ff; color:#00d2ff; border-radius:20px;
    padding:4px 18px; font-size:12px; font-weight:700; letter-spacing:1px;
    animation: blink 1.8s infinite;
}
/* Section titles — !important ensures Streamlit cascade doesn't override */
.section-title {
    font-size: 16px !important; font-weight: 800 !important;
    color: #e8f4ff !important; margin: 0 0 12px 0 !important;
    border-left: 3px solid #3a6a8a; padding-left: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Paths & display constants ─────────────────────────────────────────────────
MODEL_PATH   = "models/"
DATA_PATH    = "data/fl_training/"
HOSP_NAMES   = ["General ICU", "Mixed ICU", "Cardiac/Trauma ICU"]
HOSP_COLORS  = ["#28a745", "#f0a500", "#dc3545"]
HOSP_RGBA    = ["40,167,69", "240,165,0", "220,53,69"]
HOSP_DIM     = ["#1a3525",   "#3a2a10",   "#3a1515"]


# ── Shared singleton state (survives Streamlit reruns) ────────────────────────

def _blank_state() -> dict:
    return {
        "status":         "idle",
        "current_round":  0,
        "total_rounds":   10,
        "algorithm":      "FedYogi + FedProx",
        "current_phase":  "idle",
        "phase_label":    "Configure settings above and press ▶ Start",
        "client_losses":      [[], [], []],
        "global_losses":      [],
        "client_mae_history": [[], [], []],   # per-round local MAE per hospital
        "global_mae_history": [],             # per-round global MAE on combined val set
        "global_r2_history":  [],             # per-round global R² on combined val set
        "client_metrics":     [{}, {}, {}],
        "weight_stats":       [],
        "pre_agg_flat":   None,
        "post_agg_flat":  None,
        "layer1_weights": None,
        "best_round":     0,
        "best_loss":      float("inf"),
        "round_log":      [],
        "error":          None,
    }


@st.cache_resource
def _shared():
    return {
        "state": _blank_state(),
        "lock":  threading.Lock(),
        "stop":  threading.Event(),
    }


_S     = _shared()
_STATE = _S["state"]
_LOCK  = _S["lock"]
_STOP  = _S["stop"]


def _reset_state(num_rounds: int, algorithm: str):
    with _LOCK:
        _STATE.clear()
        _STATE.update(_blank_state())
        _STATE["total_rounds"] = num_rounds
        _STATE["algorithm"]    = algorithm


def _read_state() -> dict:
    with _LOCK:
        return {
            "status":         _STATE["status"],
            "current_round":  _STATE["current_round"],
            "total_rounds":   _STATE["total_rounds"],
            "algorithm":      _STATE["algorithm"],
            "current_phase":  _STATE["current_phase"],
            "phase_label":    _STATE["phase_label"],
            "client_losses":      [list(x) for x in _STATE["client_losses"]],
            "global_losses":      list(_STATE["global_losses"]),
            "client_mae_history": [list(x) for x in _STATE["client_mae_history"]],
            "global_mae_history": list(_STATE["global_mae_history"]),
            "global_r2_history":  list(_STATE["global_r2_history"]),
            "client_metrics":     [dict(m) for m in _STATE["client_metrics"]],
            "weight_stats":   list(_STATE["weight_stats"]),
            "pre_agg_flat":   _STATE["pre_agg_flat"],
            "post_agg_flat":  _STATE["post_agg_flat"],
            "layer1_weights": _STATE["layer1_weights"],
            "best_round":     _STATE["best_round"],
            "best_loss":      _STATE["best_loss"],
            "round_log":      list(_STATE["round_log"]),
            "error":          _STATE["error"],
        }


# ── Aggregation algorithms ────────────────────────────────────────────────────

def _fedavg(client_weights: list, client_samples: list) -> list:
    total  = sum(client_samples)
    result = []
    for li in range(len(client_weights[0])):
        layer = sum(
            client_weights[ci][li] * (client_samples[ci] / total)
            for ci in range(len(client_weights))
        )
        result.append(layer.astype(np.float32))
    return result


def _fedyogi(client_weights, client_samples, global_weights,
              m_t, v_t, eta=0.01, beta1=0.9, beta2=0.99, tau=1e-3):
    avg   = _fedavg(client_weights, client_samples)
    delta = [a - g for a, g in zip(avg, global_weights)]
    if m_t is None:
        m_t = [np.zeros_like(d) for d in delta]
        v_t = [np.full_like(d, tau ** 2, dtype=np.float32) for d in delta]
    new_m, new_v, new_w = [], [], []
    for d, m, v, g in zip(delta, m_t, v_t, global_weights):
        m_new = beta1 * m + (1 - beta1) * d
        d_sq  = d ** 2
        v_new = v + (1 - beta2) * np.sign(d_sq - v) * d_sq
        v_new = np.maximum(v_new, tau ** 2)
        w_new = (g + eta * m_new / (np.sqrt(v_new) + tau)).astype(np.float32)
        new_m.append(m_new.astype(np.float32))
        new_v.append(v_new.astype(np.float32))
        new_w.append(w_new)
    return new_w, new_m, new_v


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_client_data():
    scaler       = joblib.load(MODEL_PATH + "scaler.pkl")
    feature_cols = list(scaler.feature_names_in_)
    input_dim    = len(feature_cols)
    clients = []
    for cid in range(3):
        df = pd.read_csv(DATA_PATH + f"client_{cid}.csv")
        y  = df["sofa_score"].values.astype(np.float32)
        X  = df.drop(columns=["sofa_score"])
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_cols].values.astype(np.float32)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        clients.append({
            "X_train": X_tr,  "y_train": y_tr,
            "X_val":   X_val, "y_val":   y_val,
            "n_train": int(len(X_tr)),
            "n_total": int(len(X)),
            "sofa_min":  float(y.min()),
            "sofa_max":  float(y.max()),
            "sofa_mean": float(y.mean()),
        })
    return clients, input_dim


# ── FL background thread ──────────────────────────────────────────────────────

def fl_thread(num_rounds: int, speed: float, algorithm: str,
              clients: list, input_dim: int):
    use_yogi  = "FedYogi" in algorithm
    use_prox  = "FedProx" in algorithm
    mu        = 0.5 if use_prox else 0.0
    eta, b1, b2, tau = 0.01, 0.9, 0.99, 1e-3
    m_t = v_t = None

    init_model     = ICUModel(input_dim)
    global_weights = get_weights(init_model)

    # Pre-build combined validation set for global evaluation each round
    combined_X_val = np.concatenate([c["X_val"] for c in clients])
    combined_y_val = np.concatenate([c["y_val"] for c in clients])

    with _LOCK:
        _STATE["global_weights"] = global_weights

    for rnd in range(1, num_rounds + 1):
        if _STOP.is_set():
            break

        # ── Phase ① Distribute ────────────────────────────────────────────
        with _LOCK:
            _STATE["current_round"] = rnd
            _STATE["current_phase"] = "distributing"
            _STATE["phase_label"]   = (
                f"Round {rnd}/{num_rounds} — Server distributing global weights → hospitals"
            )
        time.sleep(speed)
        if _STOP.is_set():
            break

        # ── Phases ② ③ ④  Train each client ──────────────────────────────
        c_weights, c_losses, c_samples, c_stats = [], [], [], []
        c_maes = []   # per-client MAE this round (for dual-axis chart)

        for cid in range(3):
            if _STOP.is_set():
                break
            with _LOCK:
                _STATE["current_phase"] = f"training_{cid}"
                _STATE["phase_label"]   = (
                    f"Round {rnd}/{num_rounds} — "
                    f"Training Hospital {cid} ({HOSP_NAMES[cid]})…"
                )
            c     = clients[cid]
            model = ICUModel(input_dim)
            set_weights(model, global_weights)
            lr    = max(1e-4, 0.001 * (0.99 ** (rnd - 1)))
            loss  = train_model(
                model, c["X_train"], c["y_train"],
                epochs=3, lr=lr, batch_size=64, grad_clip=1.0,
                oversample=False,
                global_params=global_weights if use_prox else None,
                mu=mu,
            )
            local_w = get_weights(model)
            metrics = evaluate_model(model, c["X_val"], c["y_val"])
            lf      = np.concatenate([w.flatten() for w in local_w])
            gf      = np.concatenate([w.flatten() for w in global_weights])
            c_stats.append({
                "n_params":   int(lf.size),
                "mean":       float(np.mean(lf)),
                "std":        float(np.std(lf)),
                "delta_mean": float(np.mean(np.abs(lf - gf))),
            })
            c_weights.append(local_w)
            c_losses.append(float(loss))
            c_samples.append(c["n_train"])
            c_maes.append(float(metrics["mae"]))
            with _LOCK:
                _STATE["client_metrics"][cid] = {
                    "mae":  round(float(metrics["mae"]), 4),
                    "r2":   round(float(metrics["r2"]),  4),
                    "loss": round(float(loss),            4),
                }
            time.sleep(speed)

        if _STOP.is_set():
            break

        # ── Phase ⑤ Aggregate ─────────────────────────────────────────────
        with _LOCK:
            _STATE["current_phase"] = "aggregating"
            _STATE["phase_label"]   = (
                f"Round {rnd}/{num_rounds} — "
                f"Aggregating weights at server ({algorithm})…"
            )
        pre_flat = np.concatenate([w.flatten() for w in global_weights])
        if use_yogi:
            global_weights, m_t, v_t = _fedyogi(
                c_weights, c_samples, global_weights, m_t, v_t, eta, b1, b2, tau
            )
        else:
            global_weights = _fedavg(c_weights, c_samples)
        post_flat = np.concatenate([w.flatten() for w in global_weights])
        layer1_w  = global_weights[0].copy()   # (128, 108)
        total_n   = sum(c_samples)
        g_loss    = float(sum(l * n / total_n for l, n in zip(c_losses, c_samples)))

        # Global MAE + R² on combined validation set (mirrors app_test.py approach)
        _eval_model = ICUModel(input_dim)
        set_weights(_eval_model, global_weights)
        _global_eval = evaluate_model(_eval_model, combined_X_val, combined_y_val)
        g_mae = float(_global_eval["mae"])
        g_r2  = float(_global_eval["r2"])
        time.sleep(speed)
        if _STOP.is_set():
            break

        # ── Phase ⑥ Round complete ────────────────────────────────────────
        is_best = g_loss < _STATE["best_loss"]
        with _LOCK:
            _STATE["current_phase"] = "round_complete"
            _STATE["phase_label"]   = (
                f"Round {rnd}/{num_rounds} complete — "
                f"global loss: {g_loss:.4f}"
                + ("  ✓ New best!" if is_best else "")
            )
            for cid in range(3):
                _STATE["client_losses"][cid].append(c_losses[cid])
                _STATE["client_mae_history"][cid].append(c_maes[cid])
            _STATE["global_losses"].append(g_loss)
            _STATE["global_mae_history"].append(g_mae)
            _STATE["global_r2_history"].append(g_r2)
            _STATE["weight_stats"].append(c_stats)
            _STATE["pre_agg_flat"]   = pre_flat
            _STATE["post_agg_flat"]  = post_flat
            _STATE["layer1_weights"] = layer1_w
            if is_best:
                _STATE["best_round"] = rnd
                _STATE["best_loss"]  = g_loss
            _STATE["round_log"].append({
                "Round":   rnd,
                "H0 Loss": round(c_losses[0], 4),
                "H1 Loss": round(c_losses[1], 4),
                "H2 Loss": round(c_losses[2], 4),
                "Global":  round(g_loss, 4),
                "Best":    "✓ Best" if is_best else "",
            })
        time.sleep(speed * 0.5)

    with _LOCK:
        if _STOP.is_set():
            _STATE["status"]       = "stopped"
            _STATE["current_phase"] = "stopped"
            _STATE["phase_label"]  = "Simulation stopped by user."
        else:
            br = _STATE["best_round"]
            bl = _STATE["best_loss"]
            _STATE["status"]       = "done"
            _STATE["current_phase"] = "done"
            _STATE["phase_label"]  = (
                f"Simulation complete!  Best round: {br}  (loss: {bl:.4f})"
            )


# ── Chart builders ────────────────────────────────────────────────────────────

def _training_chart(client_mae_history, global_mae_history, global_r2_history):
    """
    Dual-axis chart inspired by app_test.py Watch AI Learn section:
      Left Y-axis  — MAE per round (local per hospital + global, lower = better)
      Right Y-axis — Global R² per round (higher = better)
    """
    rounds = list(range(1, len(global_mae_history) + 1))
    if not rounds:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    short_names = ["H0 · General ICU", "H1 · Mixed ICU", "H2 · Cardiac ICU"]
    for i in range(3):
        if len(client_mae_history[i]) == len(rounds):
            fig.add_trace(go.Scatter(
                x=rounds, y=client_mae_history[i],
                name=short_names[i], mode="lines+markers",
                line=dict(color=HOSP_COLORS[i], width=1.5, dash="dot"),
                marker=dict(size=4), opacity=0.75,
            ), secondary_y=False)

    # Global MAE — bold, prominent
    fig.add_trace(go.Scatter(
        x=rounds, y=global_mae_history,
        name="Global MAE", mode="lines+markers",
        line=dict(color="#ff6b85", width=3),
        marker=dict(size=9, symbol="diamond", color="#ff6b85",
                    line=dict(color="white", width=1.5)),
    ), secondary_y=False)

    # Global R² — secondary axis, cyan
    fig.add_trace(go.Scatter(
        x=rounds, y=global_r2_history,
        name="Global R²", mode="lines+markers",
        line=dict(color="#00d2ff", width=2.5),
        marker=dict(size=7, symbol="circle", color="#00d2ff",
                    line=dict(color="white", width=1)),
    ), secondary_y=True)

    dtick = 1 if len(rounds) <= 10 else 2
    fig.update_layout(
        title=dict(
            text="Training Progress — MAE & R² per Round",
            font=dict(color="white", size=13),
        ),
        paper_bgcolor="#0d1a24", plot_bgcolor="#0d1a24",
        font=dict(color="#ccc"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)", bordercolor="#1e3a50",
            font=dict(size=10), x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        margin=dict(l=10, r=55, t=42, b=38),
        height=320,
        xaxis=dict(
            title="Round", gridcolor="#132030", color="#7fb3c8",
            tickmode="linear", dtick=dtick,
            range=[0.5, len(rounds) + 0.5], tick0=1,
        ),
    )
    fig.update_yaxes(
        title_text="MAE  (↓ lower = better)", secondary_y=False,
        gridcolor="#132030", color="#ff6b85", title_font=dict(color="#ff6b85"),
    )

    # R² range: always include 0 as a reference, with padding above and below
    r2_min = min(global_r2_history) if global_r2_history else -0.5
    r2_max = max(global_r2_history) if global_r2_history else 0.5
    r2_lo  = min(-0.1, r2_min - abs(r2_min) * 0.1)
    r2_hi  = max(0.5,  r2_max + 0.05)

    fig.update_yaxes(
        title_text="R²  (↑ higher = better)", secondary_y=True,
        gridcolor="rgba(0,210,255,0.08)", color="#00d2ff",
        title_font=dict(color="#00d2ff"),
        range=[r2_lo, r2_hi],
        zeroline=True, zerolinecolor="rgba(0,210,255,0.35)", zerolinewidth=1.5,
    )

    # Annotation: explain negative R² in early rounds
    if global_r2_history and min(global_r2_history) < 0:
        fig.add_annotation(
            text="R² < 0 = model still warming up",
            x=0.5, y=1.06, xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="rgba(0,210,255,0.55)", size=10),
            align="center",
        )
    return fig


def _hist_chart(pre_flat, post_flat, rnd):
    if pre_flat is None or post_flat is None:
        return None
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pre_flat), size=min(5000, len(pre_flat)), replace=False)
    fig = go.Figure()

    # "After aggregation" — solid orange fill (rendered first, behind)
    fig.add_trace(go.Histogram(
        x=post_flat[idx], name="After aggregation",
        opacity=0.80,
        marker=dict(color="#f0a500", line=dict(color="#c08000", width=0.5)),
        nbinsx=60,
    ))

    # "Before aggregation" — OUTLINE ONLY (no fill) in cyan.
    # Using transparent fill + thick cyan border ensures the "before" distribution
    # is ALWAYS visible regardless of how much it overlaps with "after".
    fig.add_trace(go.Histogram(
        x=pre_flat[idx], name="Before aggregation",
        marker=dict(
            color="rgba(0,0,0,0)",        # fully transparent fill
            line=dict(color="#00d2ff", width=2.5),   # bright cyan outline
        ),
        nbinsx=60,
    ))

    fig.update_layout(
        barmode="overlay",
        title=f"Weight Distribution — Round {rnd}  "
              f"<span style='font-size:11px;color:#7fb3c8;'>"
              f"(orange = after agg · cyan outline = before agg)</span>",
        title_font=dict(color="white", size=13),
        paper_bgcolor="#0d1a24", plot_bgcolor="#0d1a24",
        font=dict(color="#ccc"),
        legend=dict(bgcolor="rgba(0,0,0,0.45)", bordercolor="#1e3a50",
                    orientation="h", x=0.5, y=-0.18, xanchor="center",
                    font=dict(size=11)),
        margin=dict(l=0, r=0, t=42, b=52),
        height=300,
        xaxis=dict(title="Weight value", gridcolor="#132030", color="#7fb3c8"),
        yaxis=dict(title="Count",        gridcolor="#132030", color="#7fb3c8"),
    )
    return fig


def _heatmap_chart(layer1_weights):
    if layer1_weights is None:
        return None
    W = layer1_weights.T[:40, :64]
    fig = go.Figure(data=go.Heatmap(
        z=W, colorscale="RdBu", zmid=0,
        colorbar=dict(
            title="Weight", tickfont=dict(color="#ccc"),
            title_font=dict(color="#ccc"), thickness=12,
        ),
        hovertemplate="Input %{y} → Neuron %{x}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Layer 1 Weights — inputs 0–39 × neurons 0–63",
        title_font=dict(color="white", size=12),
        paper_bgcolor="#0d1a24", plot_bgcolor="#0d1a24",
        font=dict(color="#ccc"),
        margin=dict(l=0, r=0, t=38, b=0),
        height=290,
        xaxis=dict(title="Neuron (0–63)",        color="#7fb3c8"),
        yaxis=dict(title="Input feature (0–39)", color="#7fb3c8"),
    )
    return fig


# ── Network diagram — complete self-contained HTML doc for components.html ────
# NOTE: Must use components.html(), NOT st.markdown().
# st.markdown processes multi-line indented HTML as Markdown code blocks,
# which causes the raw HTML to appear as text. components.html() renders
# inside an iframe — bypassing all Markdown processing and sanitization.

def _network_diagram(phase, rnd, total, algorithm, client_metrics,
                      best_round, best_loss):
    """Returns a full HTML document for use with st.components.v1.html()."""

    # ── Phase badge ───────────────────────────────────────────────────────
    badge_cfg = {
        "idle":          ("#7fb3c8", "⊙",  "Idle — configure and press ▶ Start"),
        "distributing":  ("#00d2ff", "①",  f"Round {rnd}/{total} — Server distributing global weights → hospitals"),
        "training_0":    ("#28a745", "②",  f"Round {rnd}/{total} — Training Hospital 0 ({HOSP_NAMES[0]})"),
        "training_1":    ("#f0a500", "③",  f"Round {rnd}/{total} — Training Hospital 1 ({HOSP_NAMES[1]})"),
        "training_2":    ("#dc3545", "④",  f"Round {rnd}/{total} — Training Hospital 2 ({HOSP_NAMES[2]})"),
        "aggregating":   ("#9c27b0", "⑤",  f"Round {rnd}/{total} — Hospitals → Server  [{algorithm}]"),
        "round_complete":("#00c853", "⑥",  f"Round {rnd}/{total} complete"),
        "done":          ("#ffd700", "✓",  f"Simulation complete — Best: Rnd {best_round}  (loss: {best_loss:.4f})"),
        "stopped":       ("#ff9800", "⏹", "Simulation stopped by user"),
        "error":         ("#f44336", "✗",  "Error during simulation"),
    }
    bc, bicon, btxt = badge_cfg.get(phase, ("#ccc", "?", phase))

    # ── Connection helpers ────────────────────────────────────────────────
    show_down = (phase == "distributing")
    show_up   = (phase == "aggregating")
    dot_col   = "#00d2ff" if (show_down or show_up) else "#1e3050"

    def _conn(i):
        delay = i * 180
        if show_down:
            # Animated dots — larger, glowing, flowing DOWN (distributing)
            inner = "".join(
                f'<div style="width:13px;height:13px;border-radius:50%;'
                f'background:{dot_col};'
                f'box-shadow:0 0 8px {dot_col},0 0 16px {dot_col}55;'
                f'margin:3px auto;'
                f'animation:dotDown 1.0s ease-in-out infinite;'
                f'animation-delay:{delay + d * 260}ms;"></div>'
                for d in range(4)
            )
        elif show_up:
            # Animated dots — larger, glowing, flowing UP (aggregating)
            inner = "".join(
                f'<div style="width:13px;height:13px;border-radius:50%;'
                f'background:{dot_col};'
                f'box-shadow:0 0 8px {dot_col},0 0 16px {dot_col}55;'
                f'margin:3px auto;'
                f'animation:dotUp 1.0s ease-in-out infinite;'
                f'animation-delay:{delay + d * 260}ms;"></div>'
                for d in range(4)
            )
        else:
            # Static dashed line — visible connection, no data flowing
            inner = (
                f'<div style="width:2px;height:62px;margin:0 auto;'
                f'background:repeating-linear-gradient(to bottom,'
                f'#3a6080 0px,#3a6080 7px,transparent 7px,transparent 14px);'
                f'border-radius:1px;opacity:0.55;"></div>'
            )
        return (f'<div style="display:flex;flex-direction:column;align-items:center;'
                f'justify-content:center;height:68px;overflow:hidden;">{inner}</div>')

    # ── Hospital node helpers ─────────────────────────────────────────────
    def _hbg(i):
        return f"rgba({HOSP_RGBA[i]},0.2)" if phase == f"training_{i}" else "rgba(8,16,26,0.9)"

    def _hborder(i):
        return f"2.5px solid {HOSP_COLORS[i]}" if phase == f"training_{i}" else f"1.5px solid {HOSP_DIM[i]}"

    def _hfooter(i):
        m = client_metrics[i]
        c = HOSP_COLORS[i]
        if phase == f"training_{i}":
            return f'<div style="font-size:11px;font-weight:700;color:{c};margin-top:8px;">&#9881; Training&#8230;</div>'
        if m:
            return (f'<div style="font-size:10px;color:#7fb3c8;margin-top:8px;">'
                    f'MAE <b style="color:#ddd;">{m.get("mae","—")}</b>'
                    f' &nbsp; R&#178; <b style="color:#ddd;">{m.get("r2","—")}</b></div>')
        return '<div style="font-size:10px;color:#3a5a70;margin-top:8px;">awaiting training</div>'

    # ── Server node ───────────────────────────────────────────────────────
    srv_on     = phase in ("distributing", "aggregating", "round_complete", "done")
    srv_border = f"2px solid {'#00d2ff' if srv_on else '#2a4a5a'}"
    srv_bg     = "rgba(0,210,255,0.12)" if srv_on else "rgba(8,16,26,0.9)"
    srv_col    = "#00d2ff" if srv_on else "#4a7a8a"
    best_info  = f" &nbsp;&#183;&nbsp; Best: Rnd {best_round}" if best_round > 0 else ""

    # ── Build full HTML document ──────────────────────────────────────────
    conn0, conn1, conn2 = _conn(0), _conn(1), _conn(2)
    h0bg, h1bg, h2bg   = _hbg(0), _hbg(1), _hbg(2)
    h0bd, h1bd, h2bd   = _hborder(0), _hborder(1), _hborder(2)
    h0ft, h1ft, h2ft   = _hfooter(0), _hfooter(1), _hfooter(2)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ background: #0a1628; }}
body {{ background: #0a1628; font-family: 'Segoe UI', Arial, sans-serif;
       padding: 2px 0 0 0; overflow: hidden; }}
@keyframes dotDown {{
  0%   {{ transform: translateY(-14px); opacity: 0; }}
  15%  {{ opacity: 1; }}
  85%  {{ opacity: 1; }}
  100% {{ transform: translateY(62px);  opacity: 0; }}
}}
@keyframes dotUp {{
  0%   {{ transform: translateY(62px);  opacity: 0; }}
  15%  {{ opacity: 1; }}
  85%  {{ opacity: 1; }}
  100% {{ transform: translateY(-14px); opacity: 0; }}
}}
.card {{
  background: linear-gradient(145deg, #0a1628, #0e2033, #122840);
  border: 1.5px solid #1e3a50;
  border-radius: 18px;
  padding: 18px 18px 16px 18px;
}}
.badge {{
  text-align: center;
  margin-bottom: 16px;
}}
.badge span {{
  background: rgba(0,0,0,0.55);
  border: 2px solid {bc};
  border-radius: 22px;
  padding: 7px 22px;
  font-size: 13px;
  font-weight: 700;
  color: {bc};
  display: inline-block;
  letter-spacing: 0.3px;
}}
.server-wrap {{ display: flex; justify-content: center; margin-bottom: 0; }}
.server {{
  background: {srv_bg};
  border: {srv_border};
  border-radius: 14px;
  padding: 12px 26px;
  text-align: center;
  min-width: 260px;
  max-width: 340px;
}}
.server-title {{ font-size: 15px; font-weight: 800; color: {srv_col}; }}
.server-sub {{ font-size: 10px; color: #7fb3c8; margin-top: 4px; }}
.server-rnd {{ font-size: 9px; color: #4a7a8a; margin-top: 3px; }}
.connections {{
  display: flex;
  justify-content: space-around;
  padding: 0 50px;
}}
.hospitals {{
  display: flex;
  gap: 10px;
}}
.hosp {{
  flex: 1;
  border-radius: 12px;
  padding: 12px 13px;
  min-height: 95px;
}}
.hosp-title {{ font-size: 13px; font-weight: 800; }}
.hosp-sub {{ font-size: 10px; color: #9ab8cc; margin-top: 3px; }}
.hosp-cnt {{ font-size: 9px; color: #6a8aaa; margin-top: 2px; }}
</style>
</head>
<body>
<div class="card">
  <div class="badge"><span>{bicon}&nbsp;&nbsp;{btxt}</span></div>
  <div class="server-wrap">
    <div class="server">
      <div class="server-title">&#128187;&nbsp; Global FL Server</div>
      <div class="server-sub">{algorithm}</div>
      <div class="server-rnd">Round {rnd} / {total}{best_info}</div>
    </div>
  </div>
  <div class="connections">
    <div>{conn0}</div>
    <div>{conn1}</div>
    <div>{conn2}</div>
  </div>
  <div class="hospitals">
    <div class="hosp" style="background:{h0bg};border:{h0bd};">
      <div class="hosp-title" style="color:{HOSP_COLORS[0]};">&#127973; Hospital 0</div>
      <div class="hosp-sub">{HOSP_NAMES[0]}</div>
      <div class="hosp-cnt">~15,889 patients</div>
      {h0ft}
    </div>
    <div class="hosp" style="background:{h1bg};border:{h1bd};">
      <div class="hosp-title" style="color:{HOSP_COLORS[1]};">&#127973; Hospital 1</div>
      <div class="hosp-sub">{HOSP_NAMES[1]}</div>
      <div class="hosp-cnt">~15,890 patients</div>
      {h1ft}
    </div>
    <div class="hosp" style="background:{h2bg};border:{h2bd};">
      <div class="hosp-title" style="color:{HOSP_COLORS[2]};">&#127973; Hospital 2</div>
      <div class="hosp-sub">{HOSP_NAMES[2]}</div>
      <div class="hosp-cnt">~16,372 patients</div>
      {h2ft}
    </div>
  </div>
</div>
</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fl-header">
  <div style="display:flex;justify-content:space-between;align-items:center;
              flex-wrap:wrap;gap:8px;">
    <div>
      <div style="font-size:24px;font-weight:900;color:white;letter-spacing:-0.5px;">
        🔬 Federated Learning Simulation Dashboard
      </div>
      <div style="font-size:12px;color:#7fb3c8;margin-top:5px;">
        ICU CDSS &nbsp;·&nbsp; Real FL Training Visualizer &nbsp;·&nbsp;
        MIMIC-III Data &nbsp;·&nbsp; PyTorch DNN &nbsp;·&nbsp;
        FedAvg / FedYogi / FedProx
      </div>
    </div>
    <div style="font-size:11px;color:#4a7a8a;text-align:right;line-height:1.7;">
      Simulates
      <span style="color:#7fb3c8;font-weight:700;">server.py + 3 × client.py</span><br>
      visually — no terminals needed
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Guard: required files ─────────────────────────────────────────────────────
_missing = [
    f"`{DATA_PATH}client_{i}.csv`"
    for i in range(3)
    if not os.path.exists(DATA_PATH + f"client_{i}.csv")
]
if not os.path.exists(MODEL_PATH + "scaler.pkl"):
    _missing.append(f"`{MODEL_PATH}scaler.pkl`")
if _missing:
    st.error(
        "⛔ Missing files: " + ", ".join(_missing) +
        ". Run `python train_federated.py` once then restart.",
        icon="⛔",
    )
    st.stop()

clients, input_dim = load_client_data()

# ── Read shared state ─────────────────────────────────────────────────────────
S          = _read_state()
is_running = (S["status"] == "running")

# ── Control panel ─────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:15px;font-weight:700;color:#7fb3c8;'
    'border-left:4px solid #00d2ff;padding-left:12px;margin-bottom:10px;">'
    '⚙️&nbsp; Simulation Controls</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns([2, 2, 3, 3])
with c1:
    num_rounds_ui = st.slider(
        "Rounds", 5, 15, 10, 1,
        disabled=is_running, help="Number of FL rounds (5–15)",
    )
with c2:
    speed_ui = st.slider(
        "Speed (s / phase)", 0.5, 5.0, 1.5, 0.5,
        disabled=is_running, help="Pause between phases — lower = faster",
    )
with c3:
    algorithm_ui = st.selectbox(
        "Aggregation Algorithm",
        ["FedAvg", "FedYogi", "FedAvg + FedProx", "FedYogi + FedProx"],
        index=3,
        disabled=is_running,
        help="FedYogi + FedProx = your actual train_federated.py setup",
    )
with c4:
    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        start = st.button("▶ Start", type="primary",
                          use_container_width=True, disabled=is_running)
    with b2:
        stop  = st.button("⏹ Stop", use_container_width=True, disabled=not is_running)
    with b3:
        reset = st.button("↺ Reset", use_container_width=True, disabled=is_running)

# Algorithm indicator
if algorithm_ui == "FedYogi + FedProx" and not is_running:
    st.markdown("""
    <div class="algo-banner">
      ★ <b style="color:#00d2ff;">FedYogi + FedProx</b> — this is the exact algorithm
      used in your <code>train_federated.py</code> run that achieved
      <b style="color:#00d2ff;">R² = 0.3357</b>
    </div>
    """, unsafe_allow_html=True)

# ── Button handlers ────────────────────────────────────────────────────────────
if start and not is_running:
    _STOP.clear()
    _reset_state(num_rounds_ui, algorithm_ui)
    with _LOCK:
        _STATE["status"] = "running"
    threading.Thread(
        target=fl_thread,
        args=(num_rounds_ui, speed_ui, algorithm_ui, clients, input_dim),
        daemon=True,
    ).start()
    st.rerun()

if stop and is_running:
    _STOP.set()
    st.rerun()

if reset and not is_running:
    _reset_state(10, "FedYogi + FedProx")
    st.rerun()

# ── Idle state: show the "How it works" guide FIRST, above everything ─────────
# (must appear before the hospital summary so it's visible without scrolling)
if S["status"] == "idle":
    st.markdown("""
<div style="padding:16px 0 20px 0;">
  <div style="font-size:15px;font-weight:700;color:#e8f4ff;
              margin-bottom:14px;border-left:4px solid #00d2ff;padding-left:12px;">
    How federated learning works — 6 phases per round
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px;">
    <div style="background:rgba(0,210,255,0.07);border:1px solid #1e3a50;
                border-top:3px solid #00d2ff;border-radius:12px;padding:16px;">
      <div style="font-size:22px;font-weight:900;color:#00d2ff;margin-bottom:8px;">① ②③④</div>
      <div style="font-size:13px;font-weight:700;color:#e8f4ff;margin-bottom:6px;">Distribute + Train</div>
      <div style="font-size:12px;color:#b8d0e0;line-height:1.7;">
        Server sends current global model to all 3 hospitals.
        Each hospital trains locally for 3 epochs on its own private patient data.
      </div>
    </div>
    <div style="background:rgba(156,39,176,0.07);border:1px solid #2a1a3a;
                border-top:3px solid #9c27b0;border-radius:12px;padding:16px;">
      <div style="font-size:22px;font-weight:900;color:#9c27b0;margin-bottom:8px;">⑤</div>
      <div style="font-size:13px;font-weight:700;color:#e8f4ff;margin-bottom:6px;">Aggregate</div>
      <div style="font-size:12px;color:#b8d0e0;line-height:1.7;">
        Hospitals send only model weights — no patient data — to the server.
        Server blends them using FedAvg or FedYogi.
      </div>
    </div>
    <div style="background:rgba(0,200,83,0.07);border:1px solid #1a3a20;
                border-top:3px solid #00c853;border-radius:12px;padding:16px;">
      <div style="font-size:22px;font-weight:900;color:#00c853;margin-bottom:8px;">⑥</div>
      <div style="font-size:13px;font-weight:700;color:#e8f4ff;margin-bottom:6px;">Track + Repeat</div>
      <div style="font-size:12px;color:#b8d0e0;line-height:1.7;">
        Global loss logged. Best round saved. New global model shared
        to all hospitals — next round begins immediately.
      </div>
    </div>
  </div>
  <div style="text-align:center;font-size:12px;color:#7fb3c8;padding:6px 0;">
    Uses your existing
    <span style="color:#7fb3c8;background:rgba(0,210,255,0.08);
                 padding:2px 8px;border-radius:4px;font-family:monospace;font-size:11px;">
      data/fl_training/client_0/1/2.csv
    </span>
    <span style="color:#7fb3c8;">— no BigQuery or internet connection required</span>
  </div>
</div>
    """, unsafe_allow_html=True)

# ── Hospital dataset summary (always shown, always starts collapsed) ───────────
with st.expander(
    "🏥 Hospital Dataset Summary — click to expand",
    expanded=False,
):
    st.markdown(
        '<div style="font-size:12px;color:#a8c8d8;padding:4px 0 10px 0;">'
        'Each hospital\'s private dataset used for local training. '
        'Patient data <b style="color:#e8f4ff;">never</b> leaves the hospital '
        '— only model weights are shared.</div>',
        unsafe_allow_html=True,
    )
    dc0, dc1, dc2 = st.columns(3)
    for col, cid in zip([dc0, dc1, dc2], range(3)):
        c = clients[cid]
        with col:
            st.markdown(f"""
            <div style="background:rgba({HOSP_RGBA[cid]},0.08);
                        border:1.5px solid {HOSP_COLORS[cid]};
                        border-radius:12px;padding:16px;">
              <div style="font-size:13px;font-weight:800;color:{HOSP_COLORS[cid]};
                          margin-bottom:10px;">
                🏥 Hospital {cid} &nbsp;—&nbsp; {HOSP_NAMES[cid]}
              </div>
              <div style="display:grid;grid-template-columns:1fr auto;gap:5px 14px;
                          font-size:12px;">
                <span style="color:#a8c8d8;">Training samples</span>
                <span style="color:#e8f4ff;font-weight:700;text-align:right;">{c['n_train']:,}</span>
                <span style="color:#a8c8d8;">Total samples</span>
                <span style="color:#e8f4ff;font-weight:700;text-align:right;">{c['n_total']:,}</span>
                <span style="color:#a8c8d8;">SOFA range</span>
                <span style="color:#e8f4ff;font-weight:700;text-align:right;">{c['sofa_min']:.1f} – {c['sofa_max']:.1f}</span>
                <span style="color:#a8c8d8;">SOFA mean</span>
                <span style="color:#e8f4ff;font-weight:700;text-align:right;">{c['sofa_mean']:.2f}</span>
                <span style="color:#a8c8d8;">Features / sample</span>
                <span style="color:#e8f4ff;font-weight:700;text-align:right;">{input_dim}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("ℹ️ Why do hospital weights differ after local training?"):
        st.markdown("""
Each hospital trains on its own patient population — different case mixes, SOFA
distributions, and clinical note vocabularies. These differences cause each
hospital's model to **drift** from the global model, known as **client drift**.

**FedProx** (μ = 0.5) limits drift by adding a regularisation term to training loss:
> `total_loss = task_loss + (μ/2) × ‖w_local − w_global‖²`

After aggregation, the global model blends all three hospitals' knowledge —
without any patient records leaving their respective institutions.
        """)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SIMULATION VIEW  (only shown when simulation is running / done / stopped)
# ═════════════════════════════════════════════════════════════════════════════

if S["status"] != "idle":
    # ── Running badge ──────────────────────────────────────────────────────────
    if is_running:
        st.markdown(
            '<div style="text-align:center;margin-bottom:12px;">'
            '<span class="running-pill">● &nbsp;SIMULATION RUNNING</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Row 1: Network diagram (left) + Live metrics (right) ───────────────
    net_col, metric_col = st.columns([3, 2])

    with net_col:
        bl = S["best_loss"]
        # Use components.html — renders in an iframe, bypassing Streamlit's
        # Markdown processor that turns multi-line indented HTML into code blocks.
        components.html(
            _network_diagram(
                S["current_phase"],
                S["current_round"],
                S["total_rounds"],
                S["algorithm"],
                S["client_metrics"],
                S["best_round"],
                bl if bl < 1e9 else 0.0,
            ),
            height=350,
            scrolling=False,
        )
        with st.expander("ℹ️ How to read this diagram"):
            st.markdown("""
**Phase ① Distribute:** Animated dots flow **downward** — server sends the global
model weights to every hospital.

**Phases ②③④ Training:** The active hospital node lights up — it trains
locally on its private patient data for 3 epochs. No data leaves the hospital.

**Phase ⑤ Aggregate:** Animated dots flow **upward** — hospitals send their
updated weights back to the server, which blends them using the selected algorithm.

**Phase ⑥ Complete:** The best round is tracked. A new round then begins.
            """)

    with metric_col:
        rnd   = S["current_round"]
        total = S["total_rounds"]
        pct   = rnd / total if total > 0 else 0.0
        sc    = {"running":"#00d2ff","done":"#ffd700",
                 "stopped":"#ff9800","error":"#f44336"}.get(S["status"],"#7fb3c8")
        bl    = S["best_loss"]
        bl_d  = f"{bl:.3f}" if bl < 1e9 else "—"
        br    = S["best_round"]

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">
          <div class="metric-tile">
            <div style="font-size:9px;color:#7fb3c8;text-transform:uppercase;letter-spacing:0.8px;">Round</div>
            <div style="font-size:34px;font-weight:900;color:#00d2ff;line-height:1.1;margin-top:4px;">
              {rnd}<span style="font-size:15px;color:#3a5a70;">/{total}</span>
            </div>
          </div>
          <div class="metric-tile">
            <div style="font-size:9px;color:#7fb3c8;text-transform:uppercase;letter-spacing:0.8px;">Status</div>
            <div style="font-size:14px;font-weight:800;color:{sc};margin-top:8px;">{S['status'].upper()}</div>
          </div>
          <div class="metric-tile">
            <div style="font-size:9px;color:#7fb3c8;text-transform:uppercase;letter-spacing:0.8px;">Best Round</div>
            <div style="font-size:34px;font-weight:900;color:#00c853;line-height:1.1;margin-top:4px;">
              {br if br > 0 else "—"}
            </div>
          </div>
          <div class="metric-tile">
            <div style="font-size:9px;color:#7fb3c8;text-transform:uppercase;letter-spacing:0.8px;">Best Loss</div>
            <div style="font-size:26px;font-weight:900;color:#ffd700;line-height:1.1;margin-top:6px;">
              {bl_d}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(pct, text=f"Round {rnd} / {total}  —  {int(pct*100)}% complete")

        fig_train = _training_chart(
            S["client_mae_history"],
            S["global_mae_history"],
            S["global_r2_history"],
        )
        if fig_train:
            st.plotly_chart(fig_train, use_container_width=True)
        else:
            st.markdown(
                '<div style="height:220px;display:flex;align-items:center;'
                'justify-content:center;color:#2a4a5a;font-size:12px;'
                'background:rgba(10,22,40,0.6);border-radius:10px;'
                'border:1px solid #1e3a50;">MAE &amp; R² curves appear after Round 1</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Row 2: Weight statistics ───────────────────────────────────────────
    if S["weight_stats"]:
        latest = S["weight_stats"][-1]
        st.markdown(
            f'<div style="font-size:16px;font-weight:800;color:#e8f4ff;margin:0 0 12px 0;'
            f'border-left:3px solid #7fb3c8;padding-left:10px;">'
            f'🔢 Weight Statistics — Round {S["current_round"]}</div>',
            unsafe_allow_html=True,
        )
        wc0, wc1, wc2 = st.columns(3)
        for col, cid, stats in zip([wc0, wc1, wc2], range(3), latest):
            m   = S["client_metrics"][cid]
            mae = m.get("mae", "—")
            r2  = m.get("r2",  "—")
            with col:
                st.markdown(f"""
                <div style="background:rgba({HOSP_RGBA[cid]},0.07);
                            border:1.5px solid {HOSP_COLORS[cid]};
                            border-radius:12px;padding:16px;">
                  <div style="font-size:12px;font-weight:800;color:{HOSP_COLORS[cid]};
                              margin-bottom:10px;">
                    🏥 Hospital {cid} — {HOSP_NAMES[cid]}
                  </div>
                  <div style="display:grid;grid-template-columns:1fr auto;
                              gap:6px 14px;font-size:12px;">
                    <span style="color:#a8c8d8;">Parameters</span>
                    <span style="color:#e8f4ff;font-weight:700;text-align:right;">{stats['n_params']:,}</span>
                    <span style="color:#a8c8d8;">Weight mean</span>
                    <span style="color:#e8f4ff;font-weight:700;text-align:right;">{stats['mean']:.5f}</span>
                    <span style="color:#a8c8d8;">Weight std</span>
                    <span style="color:#e8f4ff;font-weight:700;text-align:right;">{stats['std']:.5f}</span>
                    <span style="color:#a8c8d8;">Δ from global</span>
                    <span style="color:{HOSP_COLORS[cid]};font-weight:800;text-align:right;">{stats['delta_mean']:.5f}</span>
                    <span style="color:#a8c8d8;">Local MAE</span>
                    <span style="color:#e8f4ff;font-weight:700;text-align:right;">{mae}</span>
                    <span style="color:#a8c8d8;">Local R²</span>
                    <span style="color:#e8f4ff;font-weight:700;text-align:right;">{r2}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("ℹ️ What does Δ from global mean?"):
            st.markdown("""
**Δ from global** = mean absolute difference between this hospital's local weight
values and the current global model's weight values.

A **larger Δ** means the hospital's model drifted further from global consensus —
it over-specialised to its own patients. **FedProx** (μ = 0.5) penalises large
drift, keeping Δ values smaller across all clients.

After aggregation the global model is broadcast back to all hospitals,
resetting Δ toward zero before the next round of local training.
            """)

        st.divider()

    # ── Row 3: Histogram + Heatmap ─────────────────────────────────────────
    hc, vc = st.columns(2)

    with hc:
        st.markdown(
            '<div style="font-size:16px;font-weight:800;color:#e8f4ff;margin:0 0 10px 0;'
            'border-left:3px solid #7fb3c8;padding-left:10px;">'
            '📊 Weight Distribution — Before vs After Aggregation</div>',
            unsafe_allow_html=True,
        )
        fig_hist = _hist_chart(S["pre_agg_flat"], S["post_agg_flat"], S["current_round"])
        if fig_hist:
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.markdown(
                '<div style="height:220px;display:flex;align-items:center;'
                'justify-content:center;color:#2a4a5a;font-size:12px;'
                'background:rgba(10,22,40,0.6);border-radius:10px;'
                'border:1px solid #1e3a50;">'
                'Histogram appears after Round 1 aggregation</div>',
                unsafe_allow_html=True,
            )
        with st.expander("ℹ️ What does this histogram show?"):
            st.markdown("""
**🟦 Cyan (blue outline bars)** = global weight distribution **BEFORE** this round's aggregation

**🟧 Orange/gold bars** = global weight distribution **AFTER** aggregation

Aggregation blends all three hospitals' weight distributions into a consensus.
Over many rounds the distribution **narrows and stabilises** — the model is converging.
A wide or shifting distribution may indicate client drift or an overly high learning rate.
            """)

    with vc:
        st.markdown(
            '<div style="font-size:16px;font-weight:800;color:#e8f4ff;margin:0 0 10px 0;'
            'border-left:3px solid #7fb3c8;padding-left:10px;">'
            '🔥 Layer 1 Weight Heatmap (108 → 128)</div>',
            unsafe_allow_html=True,
        )
        fig_heat = _heatmap_chart(S["layer1_weights"])
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.markdown(
                '<div style="height:220px;display:flex;align-items:center;'
                'justify-content:center;color:#2a4a5a;font-size:12px;'
                'background:rgba(10,22,40,0.6);border-radius:10px;'
                'border:1px solid #1e3a50;">'
                'Heatmap appears after Round 1 completes</div>',
                unsafe_allow_html=True,
            )
        with st.expander("ℹ️ What does this heatmap show?"):
            st.markdown("""
Each cell = one weight connecting an **input feature** (row) to a **hidden neuron**
(column) in Layer 1 (108 inputs → 128 neurons).

- **Red** → positive weight: feature activates this neuron
- **Blue** → negative weight: feature suppresses this neuron
- **White** → near-zero: feature has little influence on this neuron

As FL progresses, **clusters of red/blue emerge** — neurons specialise to respond
to specific clinical features (low SpO₂, high RR, certain TF-IDF terms).
            """)

    st.divider()

    # ── Round history table — custom dark HTML (replaces white st.dataframe) ──
    if S["round_log"]:
        n_done = len(S["round_log"])
        st.markdown(
            f'<div style="font-size:16px;font-weight:800;color:#e8f4ff;margin:0 0 10px 0;'
            f'border-left:3px solid #7fb3c8;padding-left:10px;">'
            f'📋 Round History — {n_done} round(s) completed</div>',
            unsafe_allow_html=True,
        )

        # Build rows HTML
        rows_html = ""
        for entry in S["round_log"]:
            is_best   = entry.get("Best") == "✓ Best"
            row_bg    = "rgba(0,200,83,0.10)" if is_best else "rgba(10,22,40,0.7)"
            best_cell = (
                '<td style="text-align:center;color:#00e676;font-weight:800;'
                'letter-spacing:0.3px;">✓ Best</td>'
                if is_best else
                '<td style="text-align:center;color:#2a4a5a;">—</td>'
            )
            rows_html += (
                f'<tr style="background:{row_bg};border-bottom:1px solid #1a3050;">'
                f'<td style="color:#00c853;font-weight:800;padding:10px 16px;">'
                f'{int(entry["Round"])}</td>'
                f'<td style="color:#4caf50;padding:10px 16px;text-align:right;">'
                f'{entry["H0 Loss"]:.4f}</td>'
                f'<td style="color:#f0a500;padding:10px 16px;text-align:right;">'
                f'{entry["H1 Loss"]:.4f}</td>'
                f'<td style="color:#ef5350;padding:10px 16px;text-align:right;">'
                f'{entry["H2 Loss"]:.4f}</td>'
                f'<td style="color:#00d2ff;font-weight:700;padding:10px 16px;text-align:right;">'
                f'{entry["Global"]:.4f}</td>'
                f'{best_cell}'
                f'</tr>'
            )

        st.markdown(f"""
<div style="border:1.5px solid #1e3a50;border-radius:12px;overflow:hidden;
            background:rgba(8,16,32,0.9);">
  <table style="width:100%;border-collapse:collapse;font-size:13px;font-family:'Segoe UI',sans-serif;">
    <thead>
      <tr style="background:rgba(0,210,255,0.10);border-bottom:2px solid #1e3a50;">
        <th style="color:#7fb3c8;font-weight:700;padding:12px 16px;text-align:left;
                   letter-spacing:0.3px;">Round</th>
        <th style="color:#4caf50;font-weight:700;padding:12px 16px;text-align:right;">H0 Loss</th>
        <th style="color:#f0a500;font-weight:700;padding:12px 16px;text-align:right;">H1 Loss</th>
        <th style="color:#ef5350;font-weight:700;padding:12px 16px;text-align:right;">H2 Loss</th>
        <th style="color:#00d2ff;font-weight:700;padding:12px 16px;text-align:right;">Global</th>
        <th style="color:#7fb3c8;font-weight:700;padding:12px 16px;text-align:center;">Best</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:11px;color:#4a7a8a;padding:8px 2px 0 2px;">'
            '✓ Best = lowest global loss achieved so far &nbsp;·&nbsp; '
            'H0/H1/H2 = per-hospital validation loss &nbsp;·&nbsp; '
            'Global = weighted average &nbsp;·&nbsp; Updates live during simulation.</div>',
            unsafe_allow_html=True,
        )

# ── Completion summary card ────────────────────────────────────────────────────
if S["status"] == "done":
    br  = S["best_round"]
    bl  = S["best_loss"]
    gm  = S["global_mae_history"]
    gr  = S["global_r2_history"]
    final_mae = f"{gm[-1]:.4f}" if gm else "—"
    final_r2  = f"{gr[-1]:.4f}" if gr else "—"
    best_mae  = f"{min(gm):.4f}" if gm else "—"
    best_r2   = f"{max(gr):.4f}" if gr else "—"
    # Use components.html — multi-line indented HTML in st.markdown renders as code blocks
    _summary_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
html,body {{ background:#0a1628; padding:8px; font-family:'Segoe UI',Arial,sans-serif; }}
.card {{ background:linear-gradient(135deg,rgba(0,200,83,0.08),rgba(0,210,255,0.04)); border:2px solid #00c853; border-radius:18px; padding:22px 20px; text-align:center; }}
.title {{ font-size:22px; font-weight:900; color:#00c853; margin-bottom:5px; letter-spacing:-0.5px; }}
.sub {{ font-size:12px; color:#5a8a9a; margin-bottom:20px; }}
.tiles {{ display:flex; justify-content:center; gap:14px; flex-wrap:wrap; }}
.tile {{ background:rgba(8,16,32,0.92); border-radius:12px; padding:16px 22px; min-width:108px; }}
.lbl {{ font-size:9px; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }}
.num {{ font-size:36px; font-weight:900; line-height:1; }}
.unit {{ font-size:9px; margin-top:4px; }}
</style></head><body>
<div class="card">
  <div class="title">&#9989;&nbsp; Simulation Complete</div>
  <div class="sub">{S["total_rounds"]} rounds finished &nbsp;&middot;&nbsp; Algorithm: {S["algorithm"]}</div>
  <div class="tiles">
    <div class="tile" style="border:1px solid #1a4030;border-top:3px solid #00c853;">
      <div class="lbl" style="color:#4a8a6a;">Best Round</div>
      <div class="num" style="color:#00c853;">{br}</div>
      <div class="unit" style="color:#3a6050;">of {S["total_rounds"]} rounds</div>
    </div>
    <div class="tile" style="border:1px solid #2a2010;border-top:3px solid #ffd700;">
      <div class="lbl" style="color:#8a7a30;">Best Loss</div>
      <div class="num" style="color:#ffd700;">{bl:.4f}</div>
      <div class="unit" style="color:#5a4a20;">weighted MSE</div>
    </div>
    <div class="tile" style="border:1px solid #2a1020;border-top:3px solid #ff6b85;">
      <div class="lbl" style="color:#8a3a50;">Best MAE</div>
      <div class="num" style="color:#ff6b85;">{best_mae}</div>
      <div class="unit" style="color:#5a2030;">SOFA pts</div>
    </div>
    <div class="tile" style="border:1px solid #0a2535;border-top:3px solid #00d2ff;">
      <div class="lbl" style="color:#2a6a8a;">Best R&#178;</div>
      <div class="num" style="color:#00d2ff;">{best_r2}</div>
      <div class="unit" style="color:#1a4a6a;">variance explained</div>
    </div>
  </div>
</div>
</body></html>"""
    components.html(_summary_html, height=255, scrolling=False)

# ── Auto-refresh while simulation is running ──────────────────────────────────
if is_running:
    time.sleep(1)
    st.rerun()

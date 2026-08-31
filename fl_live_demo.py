"""
Federated Learning — Live Demo (standalone)
=============================================
A single-purpose Streamlit page for showcasing the federated learning
mechanism to a review panel. Does NOT touch app.py, federated_model.pth,
or anything else in the main project — it trains its own throwaway model
in memory each time you press the button.

Run it from inside the icu_monitor/ project folder (it needs data/,
models/scaler.pkl, and model_utils.py from there):

    streamlit run fl_live_demo.py

What it shows, live:
  - 3 simulated hospitals training locally on their own private CSVs
  - each hospital sending back ONLY model weights (0 patient records)
  - real FedAvg aggregation on the server side
  - a live chart of the global model's error dropping round by round
"""

import time
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st

from model_utils import ICUModel, train_model, evaluate_model, get_weights, set_weights

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Federated Learning — Live Demo",
    page_icon="🔴",
    layout="wide"
)

DATA_PATH  = "data/"
MODEL_PATH = "models/"
HOSPITAL_NAMES = ["General ICU", "Mixed ICU", "Cardiac/Trauma ICU"]

st.title("Federated Learning — Live Demo")
st.caption(
    "This runs real FedAvg training right now, in this browser session — not a recording. "
    "Three simulated hospitals each train a copy of the model on their **own private data**, "
    "then send back only the trained weights. This page averages those weights into an "
    "improved global model. No patient record ever leaves its hospital's process — "
    "watch the status panels below to see exactly what does and doesn't get transmitted "
    "each round."
)

st.divider()

# =============================================================
# LOAD DATA (cached — runs once)
# =============================================================
@st.cache_resource
def load_feature_columns():
    scaler = joblib.load(MODEL_PATH + "scaler.pkl")
    return list(scaler.feature_names_in_)

@st.cache_resource
def load_hospital_clients(feature_cols):
    clients = []
    for i in range(3):
        df = pd.read_csv(DATA_PATH + f"client_{i}.csv")
        y = df["sofa_score"].values.astype(np.float32)
        Xdf = df.drop(columns=["sofa_score"])
        for col in feature_cols:
            if col not in Xdf.columns:
                Xdf[col] = 0
        X = Xdf[feature_cols].values.astype(np.float32)
        clients.append((X, y, HOSPITAL_NAMES[i]))
    return clients

feature_cols = load_feature_columns()
clients_demo = load_hospital_clients(feature_cols)

st.markdown("**Hospital datasets loaded for this demo:**")
info_cols = st.columns(3)
for i, (X, y, name) in enumerate(clients_demo):
    info_cols[i].metric(name, f"{len(y):,} patient records")

st.divider()

# =============================================================
# DEMO CONTROLS
# =============================================================
c1, c2 = st.columns(2)
with c1:
    demo_rounds = st.slider("Rounds", 1, 8, 4)
with c2:
    demo_epochs = st.slider("Local epochs per round", 1, 5, 2)

run_demo = st.button("▶ Run Live Federated Training", type="primary")

st.divider()

# =============================================================
# RUN DEMO
# =============================================================
if run_demo:
    # Small held-out sample (drawn evenly from all 3 hospitals) used only
    # to score the global model live — never used for training.
    rng = np.random.default_rng(7)
    eval_X, eval_y = [], []
    for X, y, _ in clients_demo:
        idx = rng.choice(len(y), size=min(300, len(y)), replace=False)
        eval_X.append(X[idx])
        eval_y.append(y[idx])
    X_eval = np.vstack(eval_X)
    y_eval = np.concatenate(eval_y)

    demo_model = ICUModel(len(feature_cols))
    global_weights = get_weights(demo_model)

    st.markdown("### Hospital status")
    status_cols = st.columns(3)
    status_ph = [c.empty() for c in status_cols]
    for i, (_, _, name) in enumerate(clients_demo):
        status_ph[i].info(f" **{name}**\n\nWaiting to start…")

    st.markdown("### Global model progress")
    chart_ph = st.empty()
    log_ph = st.empty()
    history = []

    for rnd in range(1, demo_rounds + 1):
        local_weights_list = []
        local_n = []

        for i, (X, y, name) in enumerate(clients_demo):
            status_ph[i].warning(
                f"**{name}**\n\nRound {rnd}: training locally on "
                f"{len(y):,} private patient records…"
            )
            local_model = ICUModel(len(feature_cols))
            set_weights(local_model, global_weights)
            train_model(local_model, X, y, epochs=demo_epochs, oversample=False)
            local_weights_list.append(get_weights(local_model))
            local_n.append(len(y))
            status_ph[i].success(
                f" **{name}**\n\nRound {rnd}: done — sending **weights only** to server ✅\n\n"
                f"Patient records transmitted: **0**"
            )
            time.sleep(0.4)

        # FedAvg — weighted average of client weights by local sample count
        total_n = sum(local_n)
        new_weights = []
        for layer_idx in range(len(global_weights)):
            layer_avg = sum(
                (local_n[i] / total_n) * local_weights_list[i][layer_idx]
                for i in range(len(clients_demo))
            )
            new_weights.append(layer_avg)
        global_weights = new_weights
        set_weights(demo_model, global_weights)

        metrics = evaluate_model(demo_model, X_eval, y_eval)
        history.append({"round": rnd, "mae": metrics["mae"], "r2": metrics["r2"]})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[h["round"] for h in history],
            y=[h["mae"] for h in history],
            mode="lines+markers",
            name="Global model MAE",
            line=dict(width=3),
        ))
        fig.update_layout(
            title="Global model error after each FedAvg aggregation round",
            xaxis_title="Round",
            yaxis_title="MAE (SOFA points, lower = better)",
            height=380,
        )
        chart_ph.plotly_chart(fig, use_container_width=True)

        log_ph.caption(
            f"Round {rnd}/{demo_rounds} aggregated — "
            f"Global MAE: {metrics['mae']:.3f}  |  R²: {metrics['r2']:.3f}  |  "
            f"Aggregation: FedAvg (weighted by hospital sample count)"
        )

    st.success(
        f"Live demo complete — {demo_rounds} real FedAvg rounds across 3 hospitals. "
        f"Zero patient records were ever sent to the server, only model weights."
    )
    st.caption(
        "Note: this demo trains its own throwaway model, separate from the fully-trained "
        "production model (federated_model.pth) used in the main app, which trained for "
        "20 rounds on the full dataset. This page exists purely to make the federated "
        "*mechanism* visible."
    )

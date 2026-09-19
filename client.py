"""
Flower Federated Learning — Hospital Client
=============================================
Run one instance per hospital (in separate terminals) AFTER server.py is started:

    python client.py --client_id 0
    python client.py --client_id 1
    python client.py --client_id 2

Each client:
  - Loads its own private dataset from data/fl_training/client_<id>.csv
  - Trains the model locally (3 epochs per round, matching train_federated.py)
  - Sends only model weights to the server — NO patient data leaves
  - Receives updated global weights after each round

All hyperparameters below mirror train_federated.py for consistency.
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import flwr as fl
from sklearn.model_selection import train_test_split

from model_utils import (
    ICUModel, train_model, evaluate_model, get_weights, set_weights,
    apply_dp_to_update, estimate_privacy_budget,
)

# =============================================================
# ARGUMENTS
# =============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--client_id", type=int, required=True,
                    help="Hospital ID: 0, 1, or 2")
args = parser.parse_args()
CLIENT_ID = args.client_id

DATA_PATH    = "data/fl_training/"   # matches train_federated.py DATA_PATH
MODEL_PATH   = "models/"
SERVER_ADDR  = "127.0.0.1:8080"

# ── Training hyperparameters — keep in sync with train_federated.py ──
EPOCHS       = 3      # epochs per FL round (sweet spot: 3 with FedYogi)
BATCH_SIZE   = 64
BASE_LR      = 0.001  # starting LR; decays per round (see HospitalClient.fit)
LR_DECAY     = 0.99   # per-round LR multiplier; mirrors train_federated.py
GRAD_CLIP    = 1.0    # gradient L2 norm clip; prevents single-batch spikes
MU_FEDPROX   = 0.5    # FedProx proximal term weight; prevents client drift

# ── Differential Privacy ──
# Must match USE_DP / DP_SENSITIVITY / DP_SIGMA in train_federated.py.
USE_DP          = False
DP_SENSITIVITY  = 1.0
DP_SIGMA        = 1.0

# =============================================================
# LOAD + PREPARE DATA
# =============================================================
print(f"[Hospital {CLIENT_ID}] Loading private dataset...")

scaler  = joblib.load(MODEL_PATH + "scaler.pkl")
feature_columns = list(scaler.feature_names_in_)

df = pd.read_csv(DATA_PATH + f"client_{CLIENT_ID}.csv")

y = df["sofa_score"].values.astype(np.float32)   # raw SOFA (0-24)
X = df.drop(columns=["sofa_score"])

for col in feature_columns:
    if col not in X.columns:
        X[col] = 0
X = X[feature_columns].values.astype(np.float32)
y = y.astype(np.float32)

# Local train / validation split (no data sent to server)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)

input_dim = X_train.shape[1]
model = ICUModel(input_dim)

dp_info = f"DP ENABLED (σ={DP_SIGMA}, S={DP_SENSITIVITY})" if USE_DP else "DP disabled"
print(f"[Hospital {CLIENT_ID}] Ready — "
      f"{len(X_train)} train | {len(X_val)} val | {dp_info}")

# =============================================================
# FLOWER CLIENT
# =============================================================
class HospitalClient(fl.client.NumPyClient):

    def get_parameters(self, config):  # noqa: ARG002
        return get_weights(model)

    def fit(self, parameters, config):
        # Save global weights BEFORE local training (needed for DP + FedProx)
        global_weights = [w.copy() for w in parameters]
        set_weights(model, parameters)

        # Compute per-round decaying LR (mirrors train_federated.py)
        server_round = config.get("server_round", 1)
        lr = max(1e-4, BASE_LR * (LR_DECAY ** (server_round - 1)))

        loss = train_model(
            model, X_train, y_train,
            epochs=EPOCHS, lr=lr,
            batch_size=BATCH_SIZE, grad_clip=GRAD_CLIP,
            global_params=parameters, mu=MU_FEDPROX,   # FedProx
        )
        metrics = evaluate_model(model, X_val, y_val)

        dp_tag = ""
        if USE_DP:
            local_weights   = get_weights(model)
            weights_to_send = apply_dp_to_update(
                local_weights, global_weights, DP_SENSITIVITY, DP_SIGMA
            )
            dp_tag = " | DP ✓"
        else:
            weights_to_send = get_weights(model)

        print(
            f"[Hospital {CLIENT_ID}] Round {server_round} — "
            f"lr={lr:.5f} | loss={loss:.4f} | "
            f"MAE={metrics['mae']:.3f} SOFA | "
            f"R²={metrics['r2']:.3f}{dp_tag}"
        )
        return weights_to_send, len(X_train), {}

    def evaluate(self, parameters, config):  # noqa: ARG002
        set_weights(model, parameters)
        metrics = evaluate_model(model, X_val, y_val)
        print(
            f"[Hospital {CLIENT_ID}] Eval — "
            f"MAE={metrics['mae']:.3f} SOFA | "
            f"R²={metrics['r2']:.3f}"
        )
        return metrics["mse"], len(X_val), {
            "mae": metrics["mae"],
            "r2":  metrics["r2"]
        }


# =============================================================
# CONNECT TO SERVER
# =============================================================
print(f"[Hospital {CLIENT_ID}] Connecting to server at {SERVER_ADDR}...")

fl.client.start_numpy_client(
    server_address=SERVER_ADDR,
    client=HospitalClient()
)

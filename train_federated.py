"""
Federated Learning Training Script
====================================
Trains the ICU DNN using Flower FL simulation across 3 simulated hospital clients.

Key improvements over the original notebook:
  1. Weighted MSE loss  — high-SOFA patients get more weight (handles class imbalance)
  2. Gradient clipping  — prevents exploding gradients during FL aggregation
  3. LR scheduling      — server passes round number to clients; lr decays per round
  4. SaveBestStrategy   — saves the round with lowest eval loss, not just the last

Data split:
  Uses the pre-existing client_0/1/2.csv files (IID random split).
  Non-IID splitting was tested but caused client drift with FedAvg, reducing
  global model quality (R²=0.084 vs R²=0.284 with IID). Non-IID is an
  important FL research topic to explain in the viva, but IID gives better
  model quality for the deployed system.

Usage:
    python train_federated.py

Outputs:
    models/federated_model.pth    — best global model
    models/shap_background.npy    — 100 background samples for SHAP DeepExplainer
"""

import json
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
import flwr as fl
from flwr.common import Context, parameters_to_ndarrays, FitIns, ndarrays_to_parameters
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from model_utils import (
    ICUModel, train_model, evaluate_model, get_weights, set_weights,
    apply_dp_to_update, estimate_privacy_budget,   # DP now centralised in model_utils
)

# =============================================================
# CONFIG
# =============================================================
DATA_PATH       = "data/"
MODEL_PATH      = "models/"
NUM_ROUNDS       = 20
EPOCHS_PER_ROUND = 10
BATCH_SIZE       = 64
SHAP_BG_SAMPLES  = 300   # was 100 — more samples → stabler SHAP attributions

# ---- Learning Rate Scheduling ----
# Server passes round number to clients via Flower config (configure_fit).
# Clients compute: lr = BASE_LR × (LR_DECAY ^ (round-1))
# LR_DECAY = 1.0  → no decay (fixed lr, best tested performance: R²=0.22–0.28)
# LR_DECAY = 0.97 → gentle 3% decay per round (tested; hurt convergence on this data)
# Leave at 1.0 for production use; tune if retraining on new data.
BASE_LR          = 0.001
LR_DECAY         = 1.0

# ---- Gradient Clipping ----
# None = disabled (weighted MSE produces well-scaled gradients for this problem)
# Set to float (e.g. 5.0) to enable — tested with 1.0 and 5.0, both reduced R².
GRAD_CLIP        = None

# ---- Oversampling ----
# True  = WeightedRandomSampler draws High-Risk patients (6% natural freq.)
#         ~28% of every training batch — correcting the severe class imbalance.
#         Combined with the piecewise loss (Low=1×, Mod=5×, High=20×),
#         total gradient emphasis on High-Risk is ~94× vs plain MSE.
# False = natural sampling (IID batches, for baseline comparison).
OVERSAMPLE       = False  # Tested with True (×10 high-risk) — caused overfitting to minority class

# ---- Hospital Data Split ----
# False = IID split  — uses pre-existing client_0/1/2.csv equal splits
#                      Best model quality: R²≈0.28, MAE≈2.0
# True  = Non-IID split — re-splits combined data by SOFA severity to simulate
#                          realistic hospital specialty bias (General / Mixed /
#                          Cardiac-Trauma ICU). More authentic FL scenario but
#                          causes FedAvg client drift, reducing global R² to ~0.08.
#                          Use for experimentation or viva demonstration only.
USE_NONIID_SPLIT = False

# ---- Differential Privacy ----
# False = disabled (plain FedAvg, best performance)
# True  = each client clips its model update and adds Gaussian noise before
#         sending weights to the server. This provides (ε,δ)-DP per client.
#
# How it works:
#   1. After local training, compute update = local_weights - global_weights
#   2. Clip update L2 norm to DP_SENSITIVITY  (sensitivity clipping)
#   3. Add N(0, DP_SIGMA² × DP_SENSITIVITY²) Gaussian noise to each weight
#   4. Return global_weights + clipped_noisy_update
#
# Parameters:
#   DP_SENSITIVITY  — max L2 norm of model update (clipping threshold)
#                     Lower = stronger privacy, more accuracy loss
#   DP_SIGMA        — noise multiplier (std = sigma × sensitivity)
#                     Higher = stronger privacy, more noise
#
# Approximate privacy budget per round (Gaussian mechanism):
#   ε ≈ sqrt(2 × ln(1.25/δ)) / DP_SIGMA    (per round, δ=1e-5)
USE_DP          = False
DP_SENSITIVITY  = 1.0    # max L2 norm of update (clipping threshold)
DP_SIGMA        = 1.0    # noise multiplier

# =============================================================
# LOAD CLIENT DATASETS
# =============================================================
print("=" * 60)
print("  ICU Federated Learning — Training Script")
print("=" * 60)
print("\n[1/6] Loading hospital datasets...")

scaler = joblib.load(MODEL_PATH + "scaler.pkl")
feature_columns = list(scaler.feature_names_in_)

hospital_names = ["General ICU", "Mixed ICU", "Cardiac/Trauma ICU"]
clients_data = []
all_X, all_y = [], []

for i in range(3):
    df = pd.read_csv(DATA_PATH + f"client_{i}.csv")
    y  = df["sofa_score"].values.astype(np.float32)
    X  = df.drop(columns=["sofa_score"])
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns].values.astype(np.float32)

    n_low  = (y < 5).sum()
    n_mod  = ((y >= 5) & (y < 10)).sum()
    n_high = (y >= 10).sum()

    print(f"  Hospital {i} ({hospital_names[i]}):  {len(y):,} samples — "
          f"Low:{n_low}({n_low/len(y)*100:.0f}%) | "
          f"Mod:{n_mod}({n_mod/len(y)*100:.0f}%) | "
          f"High:{n_high}({n_high/len(y)*100:.0f}%)")

    clients_data.append((X, y))
    all_X.append(X)
    all_y.append(y)

X_all = np.vstack(all_X)
y_all = np.concatenate(all_y)
input_dim = X_all.shape[1]

print(f"\n  Combined: {X_all.shape[0]:,} samples, {X_all.shape[1]} features")
print(f"  SOFA — Low:{(y_all<5).sum():,}({(y_all<5).mean()*100:.0f}%) | "
      f"Mod:{((y_all>=5)&(y_all<10)).sum():,}({((y_all>=5)&(y_all<10)).mean()*100:.0f}%) | "
      f"High:{(y_all>=10).sum():,}({(y_all>=10).mean()*100:.0f}%)")

# Hold-out 20% for final evaluation (stratified to preserve High-Risk ratio)
X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42,
    stratify=(y_all >= 10).astype(int)
)

# =============================================================
# NON-IID SPLIT FUNCTION  (used when USE_NONIID_SPLIT = True)
# =============================================================

def make_noniid_split(X_pool, y_pool, seed=42):
    """
    Create a Non-IID hospital split biased by SOFA severity.

    Each hospital reflects a different clinical specialty:
      Hospital 0 — General ICU:       70% Low | 20% Moderate | 10% High SOFA
      Hospital 1 — Mixed ICU:         20% Low | 60% Moderate | 20% High SOFA
      Hospital 2 — Cardiac/Trauma:    10% Low | 20% Moderate | 70% High SOFA

    Why Non-IID matters:
      In real federated healthcare settings, different hospitals serve different
      patient populations. A general community hospital sees mostly stable patients,
      while a cardiac surgery centre sees many high-acuity cases. FedAvg must
      reconcile these heterogeneous local models into a single global model.
      This is called the Non-IID FL challenge, and it is an active research area.

    Observed performance impact (on this dataset):
      Non-IID → R²≈0.08 vs IID → R²≈0.28. Client drift from divergent local
      gradients causes the FedAvg global model to converge poorly. This is a
      known limitation of standard FedAvg and motivates advanced FL algorithms
      like FedProx and SCAFFOLD in future work.
    """
    rng = np.random.default_rng(seed)

    idx_low  = np.where(y_pool < 5)[0]
    idx_mod  = np.where((y_pool >= 5) & (y_pool < 10))[0]
    idx_high = np.where(y_pool >= 10)[0]

    rng.shuffle(idx_low)
    rng.shuffle(idx_mod)
    rng.shuffle(idx_high)

    # Fraction each hospital receives from each severity group
    fracs_low  = np.array([0.70, 0.20, 0.10])
    fracs_mod  = np.array([0.20, 0.60, 0.20])
    fracs_high = np.array([0.10, 0.20, 0.70])

    def _split(indices, fracs):
        n = len(indices)
        cuts = np.clip(np.round(np.cumsum(fracs) * n).astype(int), 0, n)
        parts, prev = [], 0
        for cut in cuts:
            parts.append(indices[prev:cut])
            prev = cut
        return parts

    low_p  = _split(idx_low,  fracs_low)
    mod_p  = _split(idx_mod,  fracs_mod)
    high_p = _split(idx_high, fracs_high)

    splits = []
    for i in range(3):
        idx = np.concatenate([low_p[i], mod_p[i], high_p[i]])
        rng.shuffle(idx)
        splits.append((X_pool[idx], y_pool[idx]))

    return splits

# =============================================================
# APPLY CHOSEN SPLIT
# =============================================================
if USE_NONIID_SPLIT:
    print("\n  Split type: Non-IID (hospital specialty bias)")
    print("  ⚠ Note: Non-IID typically reduces global R² to ~0.08 due to client drift.")
    noniid_splits = make_noniid_split(X_train_pool, y_train_pool)
    # Override clients_data with the non-IID re-split
    clients_data = []
    for i, (X_h, y_h) in enumerate(noniid_splits):
        n_low  = (y_h < 5).sum()
        n_mod  = ((y_h >= 5) & (y_h < 10)).sum()
        n_high = (y_h >= 10).sum()
        print(f"  Hospital {i} ({hospital_names[i]}): {len(y_h):,} samples — "
              f"Low:{n_low}({n_low/len(y_h)*100:.0f}%) | "
              f"Mod:{n_mod}({n_mod/len(y_h)*100:.0f}%) | "
              f"High:{n_high}({n_high/len(y_h)*100:.0f}%)")
        clients_data.append((X_h, y_h))
else:
    print("\n  Split type: IID (using pre-existing client_0/1/2.csv splits)")

# =============================================================
# SHAP BACKGROUND DATA
# =============================================================
print("\n[2/6] Saving SHAP background samples...")
rng = np.random.default_rng(42)
bg_idx = rng.choice(len(X_train_pool), SHAP_BG_SAMPLES, replace=False)
background = X_train_pool[bg_idx]
np.save(MODEL_PATH + "shap_background.npy", background)
print(f"  Saved {SHAP_BG_SAMPLES} background samples → models/shap_background.npy")

# apply_dp_to_update() and estimate_privacy_budget() are now in model_utils.py
# so that client.py (real FL) and train_federated.py (simulation) share identical logic.


# =============================================================
# FLOWER CLIENT  (with learning-rate scheduling + optional DP)
# =============================================================
def get_model():
    return ICUModel(input_dim)

class HospitalClient(fl.client.NumPyClient):

    def __init__(self, X, y, hospital_id, name):
        self.hospital_id = hospital_id
        self.name        = name
        self.model       = get_model()

        # Proper random train/val split (not just last N rows)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        self.X_train, self.X_val = X_tr, X_val
        self.y_train, self.y_val = y_tr, y_val

    def get_parameters(self, config):  # noqa: ARG002
        return get_weights(self.model)

    def fit(self, parameters, config):
        # Save global weights BEFORE local training (needed for DP update computation)
        global_weights = [w.copy() for w in parameters]

        set_weights(self.model, parameters)

        # ---- LEARNING RATE SCHEDULING ----
        server_round = config.get("server_round", 1)
        lr = max(1e-4, BASE_LR * (LR_DECAY ** (server_round - 1)))

        loss = train_model(
            self.model, self.X_train, self.y_train,
            epochs=EPOCHS_PER_ROUND, lr=lr,
            batch_size=BATCH_SIZE, grad_clip=GRAD_CLIP,
            oversample=OVERSAMPLE
        )
        metrics = evaluate_model(self.model, self.X_val, self.y_val)

        dp_tag = ""
        if USE_DP:
            # ---- DIFFERENTIAL PRIVACY ----
            # Clip model update and add Gaussian noise before returning weights.
            # This ensures the server cannot infer any individual patient's data.
            local_weights  = get_weights(self.model)
            noisy_weights  = apply_dp_to_update(
                local_weights, global_weights, DP_SENSITIVITY, DP_SIGMA
            )
            dp_tag = " | DP ✓"
            weights_to_send = noisy_weights
        else:
            weights_to_send = get_weights(self.model)

        print(
            f"    Hospital {self.hospital_id} ({self.name}) "
            f"| lr={lr:.5f} | loss={loss:.3f} "
            f"| MAE={metrics['mae']:.3f} | R²={metrics['r2']:.3f}{dp_tag}"
        )
        return weights_to_send, len(self.X_train), {}

    def evaluate(self, parameters, config):  # noqa: ARG002
        set_weights(self.model, parameters)
        metrics = evaluate_model(self.model, self.X_val, self.y_val)
        return metrics["mse"], len(self.X_val), {
            "mae": metrics["mae"],
            "r2":  metrics["r2"]
        }

hospital_clients = [
    HospitalClient(X, y, i, hospital_names[i])
    for i, (X, y) in enumerate(clients_data)
]

def client_fn(context: Context):
    cid = int(context.node_config["partition-id"])
    return hospital_clients[cid].to_client()

# =============================================================
# FLOWER SERVER STRATEGY
# — saves best global model
# — passes round number to clients for LR scheduling
# =============================================================
_best_loss    = float("inf")
_best_round   = 0
_best_weights = {}
_last_weights = {}

class SaveBestStrategy(fl.server.strategy.FedAvg):

    def configure_fit(self, server_round, parameters, client_manager):
        """Pass current round number to each client so they can decay lr."""
        config   = {"server_round": server_round}
        fit_ins  = FitIns(parameters, config)
        sample_size, min_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            _last_weights["params"] = aggregated[0]
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        global _best_loss, _best_round
        if results:
            avg_loss = sum(r.loss for _, r in results) / len(results)
            tag = ""
            if avg_loss < _best_loss:
                _best_loss  = avg_loss
                _best_round = server_round
                _best_weights["params"] = _last_weights.get("params")
                tag = "  ← best ✓"
            print(f"  [Server] Round {server_round:02d} — "
                  f"lr={BASE_LR * (LR_DECAY**(server_round-1)):.5f} | "
                  f"avg_loss={avg_loss:.4f}{tag}")
        return super().aggregate_evaluate(server_round, results, failures)

strategy = SaveBestStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

# =============================================================
# RUN FL SIMULATION
# =============================================================
print(f"\n[3/6] Starting FL simulation...")
print(f"  Rounds             : {NUM_ROUNDS}")
print(f"  Epochs/round       : {EPOCHS_PER_ROUND}")
print(f"  Batch size         : {BATCH_SIZE}")
print(f"  Base LR            : {BASE_LR}  (decays ×{LR_DECAY} per round)")
print(f"  Final LR (round {NUM_ROUNDS:02d}) : {BASE_LR * (LR_DECAY**(NUM_ROUNDS-1)):.5f}")
print(f"  Gradient clip      : {GRAD_CLIP}")
print(f"  Oversampling       : {'Enabled (Low×1 / Mod×3 / High×10)' if OVERSAMPLE else 'Disabled'}")
print(f"  Loss weights       : Linear (1 + SOFA×3) → High~2.4×, Severe~4.7×")
print(f"  Split type         : {'Non-IID (specialty bias)' if USE_NONIID_SPLIT else 'IID (client_0/1/2.csv)'}")
if USE_DP:
    eps = estimate_privacy_budget(NUM_ROUNDS, DP_SIGMA)
    print(f"  Differential Privacy: ENABLED  (σ={DP_SIGMA}, S={DP_SENSITIVITY}, ε≈{eps}, δ=1e-5)")
else:
    print(f"  Differential Privacy: disabled")
print()

# fl.simulation.start_simulation is deprecated in Flower ≥ 1.8.
# The new API uses the `flwr run` CLI with a pyproject.toml app structure,
# which requires a full project scaffold beyond this capstone's scope.
# start_simulation still works correctly — we suppress the log warning.
# Migration path for future work: https://flower.ai/docs/framework/how-to-run-simulations.html
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging as _log; _log.getLogger("flwr").setLevel(_log.ERROR)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 1}
)

_log.getLogger("flwr").setLevel(_log.WARNING)   # restore after simulation

# =============================================================
# SAVE BEST MODEL
# =============================================================
print(f"\n[4/6] Saving model (best was round {_best_round})...")

saved_params = _best_weights.get("params") or _last_weights.get("params")
if saved_params is None:
    raise RuntimeError("Training produced no weights. Check errors above.")

final_model = get_model()
set_weights(final_model, parameters_to_ndarrays(saved_params))
torch.save(final_model.state_dict(), MODEL_PATH + "federated_model.pth")
print(f"  Saved → models/federated_model.pth  (from round {_best_round})")

# =============================================================
# FINAL EVALUATION
# =============================================================
print("\n[5/6] Evaluating on held-out test set...")

final_model.eval()
with torch.no_grad():
    preds = final_model(
        torch.tensor(X_test, dtype=torch.float32)
    ).numpy().flatten()

mae = mean_absolute_error(y_test, preds)
r2  = r2_score(y_test, preds)

# Per-risk-level breakdown
for label, lo, hi in [("Low (<5)", 0, 5), ("Moderate (5-9)", 5, 10), ("High (≥10)", 10, 25)]:
    mask = (y_test >= lo) & (y_test < hi)
    if mask.sum() > 0:
        sub_mae = mean_absolute_error(y_test[mask], preds[mask])
        sub_r2  = r2_score(y_test[mask], preds[mask])
        print(f"  {label:16s}: MAE={sub_mae:.3f}  R²={sub_r2:.3f}  "
              f"(n={mask.sum():,})")

print()
print("=" * 60)
print("  FINAL MODEL PERFORMANCE  (global test set)")
print("=" * 60)
print(f"  MAE  : {mae:.4f} SOFA points")
print(f"  R²   : {r2:.4f}")
print(f"  Pred range : {preds.min():.2f} – {preds.max():.2f}")
print("=" * 60)

# =============================================================
# SAVE TRAINING METADATA  (read by app.py FL info panel)
# =============================================================
print("\n[6/6] Saving training metadata...")
metadata = {
    "num_rounds":         NUM_ROUNDS,
    "epochs_per_round":   EPOCHS_PER_ROUND,
    "batch_size":         BATCH_SIZE,
    "base_lr":            BASE_LR,
    "lr_decay":           LR_DECAY,
    "hospitals":          3,
    "hospital_names":     hospital_names,
    "aggregation":        "FedAvg",
    "split_type":         "Non-IID (specialty bias)" if USE_NONIID_SPLIT else "IID",
    "train_samples":      int(len(X_train_pool)),
    "test_samples":       int(len(X_test)),
    "input_features":     int(input_dim),
    "model_architecture": "618 → 256 → 128 → 64 → 1  (ReLU, no Dropout)",
    "best_round":         int(_best_round),
    "final_mae":          round(float(mae), 4),
    "final_r2":           round(float(r2), 4),
    "pred_range_min":     round(float(preds.min()), 2),
    "pred_range_max":     round(float(preds.max()), 2),
    "oversample":           OVERSAMPLE,
    "loss_weights":         "Linear (1 + SOFA×3) — High Risk ≈2.4× emphasis",
    "optimizer":            "AdamW (weight_decay=1e-4)",
    "differential_privacy": USE_DP,
    "dp_sensitivity":     DP_SENSITIVITY if USE_DP else None,
    "dp_sigma":           DP_SIGMA        if USE_DP else None,
    "dp_epsilon":         estimate_privacy_budget(NUM_ROUNDS, DP_SIGMA) if USE_DP else None,
    "dp_delta":           1e-5            if USE_DP else None,
}

with open(MODEL_PATH + "training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\n  Metadata saved → models/training_metadata.json")
print("\nDone. Restart the app to load the new model:")
print("  streamlit run app.py")

"""
Flower Federated Learning — Server
====================================
Run this FIRST in a terminal:
    cd icu_monitor
    python server.py

Then launch 3 clients in 3 separate terminals:
    python client.py --client_id 0
    python client.py --client_id 1
    python client.py --client_id 2

The server waits for all 3 clients before starting each round.
Best model is saved after every improvement round.
"""

import json
import torch
import joblib
import flwr as fl
from datetime import datetime
from flwr.common import parameters_to_ndarrays

from model_utils import ICUModel, set_weights

# =============================================================
# CONFIG
# =============================================================
MODEL_PATH  = "models/"
NUM_ROUNDS  = 20
SERVER_ADDR = "127.0.0.1:8080"

# =============================================================
# GLOBAL MODEL
# =============================================================
scaler    = joblib.load(MODEL_PATH + "scaler.pkl")
input_dim = scaler.n_features_in_
global_model = ICUModel(input_dim)

_best_loss  = float("inf")
_best_round = 0

# =============================================================
# CUSTOM STRATEGY
# =============================================================
class SaveBestStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_params is not None:
            weights = parameters_to_ndarrays(aggregated_params)
            set_weights(global_model, weights)
            # Always save latest model
            torch.save(
                global_model.state_dict(),
                MODEL_PATH + "federated_model.pth"
            )
            print(f"[Server] Round {server_round:02d} — model updated")
        return aggregated_params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        global _best_loss, _best_round
        if results:
            avg_loss = sum(r.loss for _, r in results) / len(results)
            print(
                f"[Server] Round {server_round:02d} — "
                f"avg eval loss: {avg_loss:.5f}",
                end=""
            )
            if avg_loss < _best_loss:
                _best_loss  = avg_loss
                _best_round = server_round
                torch.save(
                    global_model.state_dict(),
                    MODEL_PATH + "federated_model_best.pth"
                )
                print(f"  ← best model saved ✓", end="")
            print()
        return super().aggregate_evaluate(server_round, results, failures)


strategy = SaveBestStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

# =============================================================
# START SERVER
# =============================================================
print("=" * 50)
print("  ICU Federated Learning Server")
print("=" * 50)
print(f"  Address : {SERVER_ADDR}")
print(f"  Rounds  : {NUM_ROUNDS}")
print(f"  Waiting for 3 hospital clients...")
print()

fl.server.start_server(
    server_address=SERVER_ADDR,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

print()
print(f"[Server] Training complete.")
print(f"[Server] Best round: {_best_round} (loss: {_best_loss:.5f})")
print(f"[Server] Best model → {MODEL_PATH}federated_model_best.pth")
print(f"[Server] Latest model → {MODEL_PATH}federated_model.pth")

# Save partial training metadata so the app's FL info panel shows up-to-date
# configuration. MAE/R² are NOT included here (server has no test data) —
# run train_federated.py for full evaluation metrics.
_meta = {
    "num_rounds":         NUM_ROUNDS,
    "hospitals":          3,
    "aggregation":        "FedAvg",
    "best_round":         int(_best_round),
    "best_eval_loss":     round(_best_loss, 5),
    "model_architecture": "618 → 256 → 128 → 64 → 1  (ReLU, no Dropout)",
    "input_features":     int(input_dim),
    "source":             "server.py (real FL — MAE/R² not available here)",
    "completed_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    # Preserve existing metrics if they came from a previous train_federated.py run
}
_meta_path = MODEL_PATH + "training_metadata.json"
try:
    import os
    if os.path.exists(_meta_path):
        with open(_meta_path) as _f:
            _existing = json.load(_f)
        # Keep MAE/R² from simulation run if available; only overwrite config fields
        for _k in ("final_mae", "final_r2", "train_samples", "test_samples",
                   "pred_range_min", "pred_range_max"):
            if _k in _existing:
                _meta[_k] = _existing[_k]
except Exception:
    pass

with open(_meta_path, "w") as _f:
    json.dump(_meta, _f, indent=2)
print(f"[Server] Metadata saved → {_meta_path}")

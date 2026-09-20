"""
Live Federated Learning demo for the UI.

The real hospital datasets (data/fl_training/client_0/1/2.csv, ~190 MB each,
derived from MIMIC-III under a data-use agreement) are intentionally excluded
from this repository — see data/fl_training/README.md. That means the actual
100-round FedYogi+FedProx run behind models/federated_model.pth cannot be
replayed live in a browser here.

This module instead runs a REAL, from-scratch federated-averaging session
(same model architecture, same model_utils.train_model / evaluate_model
building blocks, genuine per-hospital training + weighted weight-averaging
each round) against synthetic-but-structured per-hospital data, purely so
the UI can demonstrate *how FedAvg-style FL works* end-to-end. It never
reads or writes the production model files.
"""

import asyncio
import time

import numpy as np

from model_utils import (
    ICUModel, train_model, evaluate_model, get_weights, set_weights,
)
from backend.pipeline import feature_cols, training_meta

INPUT_DIM = len(feature_cols)

HOSPITAL_NAMES = training_meta.get(
    "hospital_names", ["General ICU", "Mixed ICU", "Cardiac/Trauma ICU"]
)

# Feature indices we bias the synthetic SOFA target on, so training loss
# genuinely decreases round over round (a real, if illustrative, regression
# signal rather than pure noise).
_BIAS_FEATURES = {
    "latest_HR":     2.6,
    "latest_SpO2":  -3.0,
    "latest_RR":     1.8,
    "stress_score":  1.4,
    "GCS_eye_opening": -2.2,
    "SBP_mean":     -1.1,
}
_BIAS_IDX = {
    feature_cols.index(name): w
    for name, w in _BIAS_FEATURES.items() if name in feature_cols
}


def _make_hospital_data(n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, INPUT_DIM)).astype(np.float32)

    y = np.full(n_samples, 4.0, dtype=np.float32)
    for idx, weight in _BIAS_IDX.items():
        y += weight * X[:, idx]
    # Small per-hospital shift so the three sites look like distinct
    # populations (e.g. a cardiac/trauma ICU skewing sicker).
    y += rng.normal(0.0, 2.5, size=n_samples).astype(np.float32)
    y = np.clip(y, 0, 24).astype(np.float32)

    split = int(n_samples * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


def _fedavg(weight_list, sample_counts):
    total = sum(sample_counts)
    avg = []
    for layer_idx in range(len(weight_list[0])):
        stacked = sum(
            w[layer_idx] * (n / total)
            for w, n in zip(weight_list, sample_counts)
        )
        avg.append(stacked)
    return avg


class FLDemoRunner:
    """One live-demo session, driven round-by-round so progress can be
    streamed to a websocket as it happens."""

    def __init__(self, rounds: int = 15, epochs: int = 1, hospital_size: int = 500):
        self.rounds = max(1, min(rounds, 60))
        self.epochs = max(1, min(epochs, 5))
        self.hospital_size = max(100, min(hospital_size, 2000))
        self.cancelled = False

        self.hospitals = []
        for i, name in enumerate(HOSPITAL_NAMES):
            X_tr, y_tr, X_val, y_val = _make_hospital_data(self.hospital_size, seed=100 + i)
            self.hospitals.append({
                "id": i, "name": name,
                "X_train": X_tr, "y_train": y_tr,
                "X_val": X_val, "y_val": y_val,
            })

        # Combined held-out set to report a single "global" metric, mirroring
        # the server-side aggregate_evaluate step in server.py / train_federated.py.
        self._global_X_val = np.concatenate([h["X_val"] for h in self.hospitals])
        self._global_y_val = np.concatenate([h["y_val"] for h in self.hospitals])

        self.global_model = ICUModel(INPUT_DIM)
        self.global_weights = get_weights(self.global_model)

    def init_payload(self):
        return {
            "type": "init",
            "rounds": self.rounds,
            "epochs_per_round": self.epochs,
            "input_dim": INPUT_DIM,
            "architecture": f"{INPUT_DIM} → 128 → 64 → 32 → 1  (ReLU)",
            "hospitals": [
                {
                    "id": h["id"], "name": h["name"],
                    "n_train": len(h["X_train"]), "n_val": len(h["X_val"]),
                }
                for h in self.hospitals
            ],
            "note": (
                "Synthetic demo data — the production model (see 'Federated "
                "Learning Info' tab) was trained on real de-identified MIMIC-III "
                "ICU records across 3 hospitals; those raw files are too large "
                "and access-restricted to ship in this repo, so this tab "
                "reproduces the same FedAvg training mechanics live, on "
                "generated data, purely for demonstration."
            ),
        }

    async def run(self, send):
        await send(self.init_payload())
        loop = asyncio.get_event_loop()
        best_mae = float("inf")
        best_round = 0

        for rnd in range(1, self.rounds + 1):
            if self.cancelled:
                break
            await send({"type": "round_start", "round": rnd})

            local_weights = []
            sample_counts = []

            for h in self.hospitals:
                if self.cancelled:
                    break
                local_model = ICUModel(INPUT_DIM)
                set_weights(local_model, self.global_weights)

                t0 = time.time()
                loss = await loop.run_in_executor(
                    None, train_model, local_model, h["X_train"], h["y_train"],
                    self.epochs, 0.01, 64, 1.0, False, self.global_weights, 0.3,
                )
                metrics = await loop.run_in_executor(
                    None, evaluate_model, local_model, h["X_val"], h["y_val"]
                )
                elapsed = round(time.time() - t0, 2)

                local_weights.append(get_weights(local_model))
                sample_counts.append(len(h["X_train"]))

                await send({
                    "type": "hospital_update",
                    "round": rnd,
                    "hospital": {
                        "id": h["id"], "name": h["name"],
                        "loss": round(float(loss), 4),
                        "mae": round(float(metrics["mae"]), 4),
                        "r2": round(float(metrics["r2"]), 4),
                        "seconds": elapsed,
                    },
                })
                # Paced deliberately slower than the actual (near-instant) compute
                # time so the round-by-round training is visible/legible in the UI.
                await asyncio.sleep(0.45)

            if self.cancelled:
                break

            # FedAvg aggregation — weighted average of all hospital weights.
            self.global_weights = _fedavg(local_weights, sample_counts)
            set_weights(self.global_model, self.global_weights)
            global_metrics = await loop.run_in_executor(
                None, evaluate_model, self.global_model,
                self._global_X_val, self._global_y_val,
            )

            is_best = global_metrics["mae"] < best_mae
            if is_best:
                best_mae, best_round = global_metrics["mae"], rnd

            await send({
                "type": "aggregated",
                "round": rnd,
                "global": {
                    "mse": round(float(global_metrics["mse"]), 4),
                    "mae": round(float(global_metrics["mae"]), 4),
                    "r2": round(float(global_metrics["r2"]), 4),
                },
                "is_best": is_best,
            })
            await asyncio.sleep(0.5)

        if not self.cancelled:
            await send({
                "type": "done",
                "best_round": best_round,
                "best_mae": round(float(best_mae), 4),
                "total_rounds": self.rounds,
            })
        else:
            await send({"type": "cancelled"})

    def cancel(self):
        self.cancelled = True

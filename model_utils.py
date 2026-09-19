import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import mean_absolute_error, r2_score


# =============================================================
# MODEL ARCHITECTURE
# =============================================================

class ICUModel(nn.Module):
    """
    Fully connected DNN for SOFA score prediction.

    Input:  ~100 features — 9 trend vitals + 7 latest vitals + 2 CV + ~80 SOFA-vocab TF-IDF
    Output: raw SOFA score in the 0–24 range (NOT normalised; no ×24 needed).

    Architecture:
        Linear(input_dim → 128) → ReLU
        Linear(128 → 64)  → ReLU
        Linear( 64 → 32)  → ReLU
        Linear( 32 →  1)

    No LayerNorm: tested but collapsed prediction range to 0.22–7.04.
        LayerNorm normalises each sample's activations to mean=0, std=1 within
        the sample itself — this makes a SOFA=20 patient's activations look the
        same scale as a SOFA=2 patient, destroying the wide-range regression
        signal. High-SOFA predictions were capped at ~7, giving MAE=6.5 for
        the ≥10 segment. Removed.

    No Dropout — each hospital trains with different random masks, producing
    divergent gradient directions that FedAvg cannot reconcile. Regularisation
    is handled instead by AdamW weight_decay=1e-4 + FedProx.
    Saved weights: models/federated_model.pth
    """
    def __init__(self, input_dim):
        super(ICUModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================
# DATASET
# =============================================================

class ICUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================
# WEIGHTED MSE LOSS
# =============================================================

def weighted_mse_loss(preds, targets):
    """
    Linear weighted MSE: mild emphasis on high-SOFA patients.

    Formula: weight = 1 + target × 0.5
      SOFA =  0  → weight ≈ 0.33  (after batch-mean normalisation)
      SOFA =  4  → weight ≈ 1.0   (approx. mean SOFA in training data)
      SOFA = 10  → weight ≈ 2.0   (High Risk: 2× more important)
      SOFA = 20  → weight ≈ 3.7   (Very severe: 3.7× more important)

    Why 0.5 multiplier (reduced from 3.0):
      With multiplier=3.0, low-SOFA patients (63% of data) receive only ~7% of
      fair gradient signal after batch-mean normalisation.  The model barely
      trains on the majority class, yet the server evaluation metric (unweighted
      MSE) is dominated by low-SOFA patients.  Training and evaluation objectives
      work against each other → oscillating server loss → poor model selection.
      Reducing to 0.5 gives low-SOFA patients ≥33% of fair gradient signal
      while still providing meaningful emphasis on high-SOFA cases.

    Batch-mean normalisation keeps the loss magnitude comparable to plain MSE
    so the learning rate needs no retuning.
    """
    weights = 1.0 + targets * 0.5
    weights = weights / weights.mean()
    return (weights * (preds - targets) ** 2).mean()


def get_sample_weights(y):
    """
    Per-sample weights for PyTorch WeightedRandomSampler.
    Available for experimentation; currently not used (OVERSAMPLE=False)
    because aggressive oversampling of the 6% high-risk minority caused
    overfitting and degraded overall performance.
    """
    w = np.ones(len(y), dtype=np.float32)
    w[(y >= 5) & (y < 10)] = 2.0
    w[y >= 10]              = 5.0
    return w


# =============================================================
# TRAINING
# =============================================================

def train_model(model, X, y, epochs=10, lr=0.001, batch_size=64,
                grad_clip=None, oversample=True, global_params=None, mu=0.0):
    """
    Mini-batch training with AdamW, linear weighted MSE, optional oversampling,
    and optional FedProx proximal regularisation.

    global_params / mu (FedProx):
        When global_params (list of numpy arrays) and mu > 0 are provided,
        adds a proximal term  (μ/2) × ||w_local − w_global||²  to every
        mini-batch loss.  This penalises each hospital's model for drifting
        too far from the global model during local training, directly fixing
        the FedAvg oscillation caused by client drift.

    AdamW (vs plain Adam):
        Adam with decoupled weight decay regularises the weights more
        effectively on tabular data, reducing overfitting to the majority
        low-SOFA class.
    """
    model.train()
    dataset = ICUDataset(X, y)

    if oversample:
        sample_w = get_sample_weights(y)
        sampler  = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.float32),
            num_samples=len(sample_w),
            replacement=True
        )
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # FedProx: snapshot global weights as tensors once before training starts
    global_tensors = None
    if global_params is not None and mu > 0.0:
        global_tensors = [torch.tensor(w, dtype=torch.float32) for w in global_params]

    final_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = weighted_mse_loss(preds, y_batch)

            if global_tensors is not None:
                prox = sum(
                    ((p - gp) ** 2).sum()
                    for p, gp in zip(model.parameters(), global_tensors)
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader)

    return final_loss


# =============================================================
# EVALUATION
# =============================================================

def evaluate_model(model, X, y):
    """Returns MSE, MAE, R² on raw SOFA scale (0-24)."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_tensor).numpy().flatten()

    mse = float(nn.MSELoss()(
        torch.tensor(preds).view(-1, 1),
        torch.tensor(y, dtype=torch.float32).view(-1, 1)
    ).item())
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    return {"mse": mse, "mae": mae, "r2": r2}


# =============================================================
# FEDERATED LEARNING WEIGHT UTILITIES
# =============================================================

def get_weights(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_weights(model, weights):
    state_dict = model.state_dict()
    new_state_dict = {
        key: torch.tensor(val)
        for key, val in zip(state_dict.keys(), weights)
    }
    model.load_state_dict(new_state_dict)


# =============================================================
# TREND UTILITIES (shared between app and training)
# =============================================================

def get_trend(series):
    """Returns 'increasing', 'decreasing', or 'stable' based on linear slope."""
    if len(series) < 2:
        return "stable"
    slope = np.polyfit(range(len(series)), series, 1)[0]
    if slope > 0.1:
        return "increasing"
    elif slope < -0.1:
        return "decreasing"
    return "stable"


def classify_range(value, low, high):
    """Returns 'low', 'normal', or 'high' based on clinical thresholds."""
    if value < low:
        return "low"
    elif value > high:
        return "high"
    return "normal"


# =============================================================
# DIFFERENTIAL PRIVACY UTILITIES
# Centralised here so both train_federated.py (simulation)
# and client.py (real FL) share identical DP logic.
# =============================================================

def apply_dp_to_update(local_weights, global_weights, sensitivity, sigma):
    """
    Apply client-side Differential Privacy to a model update.

    Steps:
      1. Compute update  =  local_weights − global_weights
      2. Clip update L2 norm to `sensitivity`  (bounds any single patient's
         worst-case influence on the transmitted weights)
      3. Add i.i.d. Gaussian noise  N(0, (sigma × sensitivity)²)  to every
         weight parameter
      4. Return  global_weights + clipped_noisy_update

    The server receives only the noisy update and cannot infer any individual
    patient's data contribution, providing (ε, δ)-DP per client per round.
    """
    update = [lw - gw for lw, gw in zip(local_weights, global_weights)]

    flat  = np.concatenate([u.flatten() for u in update])
    l2    = np.linalg.norm(flat)
    if l2 > sensitivity:
        scale  = sensitivity / l2
        update = [u * scale for u in update]

    noise_std = sigma * sensitivity
    return [
        (gw + u + np.random.normal(0, noise_std, u.shape)).astype(np.float32)
        for gw, u in zip(global_weights, update)
    ]


def estimate_privacy_budget(num_rounds, sigma, delta=1e-5):
    """
    Approximate total (ε, δ)-DP budget via simple Gaussian-mechanism composition.

    Per-round:  ε_round ≈ sqrt(2 × ln(1.25/δ)) / sigma
    Total (simple composition):  ε_total = ε_round × num_rounds

    Conservative upper bound — advanced composition (RDP / moments accountant)
    gives tighter bounds but is more complex.
    """
    if sigma <= 0:
        return float("inf")
    return round(math.sqrt(2 * math.log(1.25 / delta)) / sigma * num_rounds, 3)

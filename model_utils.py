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

    Input:  618 features — 9 trend vitals + 7 latest vitals + 2 CV + 600 TF-IDF
    Output: raw SOFA score in the 0–24 range (NOT normalised; no ×24 needed).

    Architecture:
        Linear(618 → 256) → ReLU
        Linear(256 → 128) → ReLU
        Linear(128 → 64)  → ReLU
        Linear( 64 →  1)

    No Dropout — tested p=0.3/0.2 and p=0.1/0.05; both caused FL training
    divergence (R²=0.057 and -0.067 vs 0.235 without Dropout). The root cause:
    each hospital trains with different random Dropout masks, producing divergent
    gradient directions that FedAvg cannot reconcile. This is a known limitation
    of Dropout + FedAvg. Regularisation is handled instead by AdamW weight_decay.
    Trained using AdamW (weight_decay=1e-4) + linear weighted MSE.
    Saved weights: models/federated_model.pth
    """
    def __init__(self, input_dim):
        super(ICUModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
    Linear weighted MSE: high-SOFA patients get more gradient weight.

    Formula: weight = 1 + target × 3
      SOFA =  0  → weight ≈ 0.08  (after batch-mean normalisation)
      SOFA =  4  → weight ≈ 1.0   (approx. mean SOFA in training data)
      SOFA = 10  → weight ≈ 2.4   (High Risk: 2.4× more important)
      SOFA = 20  → weight ≈ 4.7   (Very severe: 4.7× more important)

    Why linear and not piecewise or exponential:
      Piecewise (Low=1/Mod=5/High=20) combined with oversampling was tested
      but caused the model to abandon low-risk accuracy for high-risk patients,
      worsening overall R² from 0.22 to -0.82. Linear weighting provides a
      smooth, stable gradient that the FedAvg aggregation handles well.

    Batch-mean normalisation keeps the loss magnitude comparable to plain MSE
    so the learning rate needs no retuning.
    """
    weights = 1.0 + targets * 3.0
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
                grad_clip=None, oversample=True):
    """
    Mini-batch training with AdamW, piecewise weighted MSE, and optional
    oversampling of minority (high-SOFA) patients.

    oversample=True (default):
        Uses WeightedRandomSampler so that high-risk patients appear ~28%
        of every batch instead of their natural 6%. Combined with the
        piecewise loss weights, this gives High-Risk predictions ~94× more
        gradient signal than plain MSE training.

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

    final_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = weighted_mse_loss(preds, y_batch)
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

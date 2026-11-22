import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


# =============================================================
# CLINICAL-NOTE CLEANER  (used by Phase 0 AND saved inside tfidf_vectorizer.pkl)
# =============================================================
NOTE_TEXT_LIMIT = 30_000   # characters of raw note text read per admission


def clean_clinical_text(text):
    """
    Clean a MIMIC-III clinical note before TF-IDF.
    Removes de-identification placeholders ([**Name**]), numbers and punctuation,
    lowercases, and reads at most NOTE_TEXT_LIMIT raw characters.
    Module-level so the fitted vectorizer can be unpickled by the backend, which
    then cleans a live clinical note exactly the way the training notes were cleaned.
    """
    import re
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[\*\*[^\]]*\*\*\]", " ", text[:NOTE_TEXT_LIMIT])
    text = re.sub(r"\b\d+(?:[./]\d+)?\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(text.split())


# =============================================================
# ALERT THRESHOLD  (shared by training evaluation and the backend)
# =============================================================
# The dashboard raises a clinical alert when predicted SOFA ≥ ALERT_THRESHOLD.
# It is lower than the High-risk band (actual SOFA ≥ 10) because the model
# under-predicts severe patients (mean prediction for true High ≈ 9.7).
# Lowered 8.0 → 7.0: in an ICU, a missed deteriorating patient costs more than
# a false alarm.  Saved model (256-128-64 + dropout), held-out test set,
# 602 true High-risk patients out of 10,450:
#     threshold 8.0 : recall 61.1%  specificity 97.8%  missed 234  false alarms 220
#     threshold 7.0 : recall 67.1%  specificity 96.3%  missed 198  false alarms 360
# AUROC / PR-AUC / MAE do not depend on this value.
ALERT_THRESHOLD = 7.0


# =============================================================
# MODEL ARCHITECTURE
# =============================================================

class ICUModel(nn.Module):
    """
    Fully connected DNN for SOFA score prediction.

    Input:  108 features — 9 trend vitals + 7 latest vitals + 2 CV + 90 SOFA-vocab TF-IDF
    Output: raw SOFA score in the 0–24 range (NOT normalised; no ×24 needed).

    Architecture (default hidden=(256, 128, 64), dropout=0.3):
        Linear(input_dim → 256) → ReLU → Dropout(0.3)
        Linear(256 → 128)       → ReLU → Dropout(0.3)
        Linear(128 →  64)       → ReLU → Dropout(0.3)
        Linear( 64 →   1)
    ≈ 69,121 trainable parameters for input_dim=108.

    Why wider + Dropout (changed from 128→64→32, no Dropout):
        With plain MSE + FedAvg the old network peaked at round ~13 and then
        overfit — each hospital's training loss kept falling (7.7 → 2.4) while
        the server validation loss rose (5.9 → 8.7).  Dropout stops that
        memorisation, and the extra width gives the model room to use it.
        Federated simulation on the dataset_v2 training set, same test set:
            128-64-32, no dropout  : MAE 1.854  R² 0.458  AUROC 0.921  PR-AUC 0.632
            256-128-64, dropout 0.3: MAE 1.831  R² 0.472  AUROC 0.925  PR-AUC 0.645
        (confirmed on two random seeds; 512-256-128 gave no further gain).
        An earlier note warned that per-hospital dropout masks could pull FedAvg
        in different directions — that did not happen in these runs.

    Dropout is only active in model.train(); evaluate_model(), the backend and
    SHAP all call model.eval(), so predictions are deterministic.

    No LayerNorm: tested earlier but collapsed prediction range to 0.22–7.04
        (it rescales every sample to the same scale, which hides how severe a
        high-SOFA patient is).

    Saved weights: models/federated_model.pth — a checkpoint trained with a
    different hidden size will not load into this class (retrain instead).
    """
    def __init__(self, input_dim, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

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
                grad_clip=None, oversample=False, global_params=None, mu=0.0,
                weighted_loss=False):
    """
    Mini-batch training with AdamW, plain MSE (default), optional oversampling,
    and optional FedProx proximal regularisation.

    weighted_loss=False (default): plain MSE.  The linear-weighted loss
        (weighted_mse_loss, 1 + SOFA×0.5) pushed every prediction upward and
        raised MAE without catching more high-risk patients.  Tested on the
        dataset_v2 training set (40-round FL simulation, same test set):
            weighted MSE : MAE 1.976  R² 0.423  recall@10 0.457
            plain MSE    : MAE 1.871  R² 0.450  recall@10 0.465
        Pass weighted_loss=True to bring the weighted loss back.

    oversample=False (default): WeightedRandomSampler is disabled. Aggressive
        oversampling of the 6% high-risk minority caused overfitting and degraded
        overall performance.  Combined with weighted_loss=True it would also
        double-count the high-SOFA emphasis.

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
    if len(X) == 0:
        return 0.0

    model.train()
    try:
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
                if weighted_loss:
                    loss = weighted_mse_loss(preds, y_batch)
                else:
                    loss = torch.mean((preds - y_batch) ** 2)

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
    finally:
        model.eval()


# =============================================================
# EVALUATION
# =============================================================

def evaluate_model(model, X, y):
    """Returns MSE, MAE, R² on raw SOFA scale (0-24)."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_tensor).numpy().flatten()

    # Clip to valid SOFA range — raw model output can go negative or above 24.
    preds = np.clip(preds, 0.0, 24.0)

    mse = float(nn.MSELoss()(
        torch.tensor(preds).view(-1, 1),
        torch.tensor(y, dtype=torch.float32).view(-1, 1)
    ).item())
    mae = mean_absolute_error(y, preds)
    # r2_score is undefined when y is constant (zero variance); return 0.0 in that case.
    r2 = float(r2_score(y, preds)) if len(y) >= 2 and float(np.std(y)) > 0.0 else 0.0

    return {"mse": mse, "mae": mae, "r2": r2}


def compute_sofa_risk_metrics(y_true, y_score, alert_threshold=ALERT_THRESHOLD):
    """
    Risk-band classification and high-risk alert metrics on held-out test data.

    Risk bands:  Low SOFA < 5 / Moderate 5–9 / High ≥ 10
    Alert logic: predicted SOFA ≥ alert_threshold → alert fired
                 actual SOFA ≥ 10                 → true high-risk case

    Returns a nested dict saved to training_metadata.json and displayed in
    the app's Tab 1 "Held-out Risk Classification Metrics" expander.

    AUROC and PR-AUC use the continuous predicted SOFA score, not the
    thresholded alert label — this gives a threshold-free discrimination
    metric.  Both are set to None when the test set contains only one class.

    Method adapted from: teammate's model_utils.py (test/ folder).
    """
    y_true  = np.asarray(y_true,  dtype=np.float64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_true.size != y_score.size:
        raise ValueError("y_true and y_score must have the same non-zero length")
    if not np.isfinite(y_true).all() or not np.isfinite(y_score).all():
        raise ValueError("y_true and y_score must contain only finite values")

    y_score = np.clip(y_score, 0, 24)

    # ── Risk-band evaluation (3-class) ──
    labels       = ["Low (<5)", "Moderate (5-<10)", "High (>=10)"]
    y_true_band  = np.digitize(y_true,  [5, 10])
    y_pred_band  = np.digitize(y_score, [5, 10])
    matrix       = confusion_matrix(y_true_band, y_pred_band, labels=[0, 1, 2])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_band, y_pred_band, labels=[0, 1, 2], zero_division=0
    )

    per_class = {}
    total     = int(matrix.sum())
    for idx, label in enumerate(labels):
        tp = int(matrix[idx, idx])
        fn = int(matrix[idx, :].sum() - tp)
        fp = int(matrix[:, idx].sum() - tp)
        tn = total - tp - fn - fp
        sp_denom = tn + fp
        per_class[label] = {
            "precision":   float(precision[idx]),
            "recall":      float(recall[idx]),
            "sensitivity": float(recall[idx]),
            "specificity": float(tn / sp_denom) if sp_denom else 0.0,
            "f1":          float(f1[idx]),
        }

    # ── High-risk alert evaluation (binary) ──
    actual_high    = (y_true  >= 10).astype(int)
    predicted_high = (y_score >= alert_threshold).astype(int)
    alert_mat      = confusion_matrix(actual_high, predicted_high, labels=[0, 1])
    tn, fp, fn, tp = (
        int(alert_mat[0, 0]), int(alert_mat[0, 1]),
        int(alert_mat[1, 0]), int(alert_mat[1, 1]),
    )
    a_prec = tp / (tp + fp) if tp + fp else 0.0
    a_rec  = tp / (tp + fn) if tp + fn else 0.0
    a_spec = tn / (tn + fp) if tn + fp else 0.0
    a_f1   = (2 * a_prec * a_rec / (a_prec + a_rec)) if a_prec + a_rec else 0.0

    if np.unique(actual_high).size == 2:
        alert_auroc  = float(roc_auc_score(actual_high, y_score))
        alert_pr_auc = float(average_precision_score(actual_high, y_score))
    else:
        alert_auroc  = None
        alert_pr_auc = None

    return {
        "risk_bands": {
            "labels":           labels,
            "confusion_matrix": matrix.tolist(),
            "per_class":        per_class,
            "macro": {
                "precision":   float(np.mean(precision)),
                "recall":      float(np.mean(recall)),
                "sensitivity": float(np.mean(recall)),
                "specificity": float(np.mean([m["specificity"] for m in per_class.values()])),
                "f1":          float(np.mean(f1)),
            },
        },
        "high_risk_alert": {
            "predicted_sofa_threshold":  float(alert_threshold),
            "actual_high_risk_threshold": 10.0,
            "confusion_matrix": alert_mat.tolist(),
            "true_negative":    tn,
            "false_positive":   fp,
            "false_negative":   fn,
            "true_positive":    tp,
            "precision":        float(a_prec),
            "recall":           float(a_rec),
            "sensitivity":      float(a_rec),
            "specificity":      float(a_spec),
            "f1":               float(a_f1),
            "auroc":            alert_auroc,
            "pr_auc":           alert_pr_auc,
        },
    }


# =============================================================
# FEDERATED LEARNING WEIGHT UTILITIES
# =============================================================

def get_weights(model):
    # .copy() ensures the returned arrays are independent snapshots — training the
    # model afterwards will not mutate them (numpy shares memory with CPU tensors).
    return [val.cpu().numpy().copy() for val in model.state_dict().values()]


def set_weights(model, weights):
    state_dict = model.state_dict()
    if len(weights) != len(state_dict):
        raise ValueError(
            f"set_weights: expected {len(state_dict)} weight arrays, got {len(weights)}"
        )
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
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]   # drop NaN — polyfit returns NaN slope otherwise
    if len(arr) < 2:
        return "stable"
    slope = np.polyfit(range(len(arr)), arr, 1)[0]
    if slope > 0.1:
        return "increasing"
    elif slope < -0.1:
        return "decreasing"
    return "stable"


def classify_range(value, low, high):
    """Returns 'low', 'normal', or 'high' based on clinical thresholds."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "normal"
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

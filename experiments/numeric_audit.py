"""
Numeric-feature audit: how much do preprocessing and the evaluation split matter?

Uses the raw BigQuery export of the 18 numeric features (data_values_exp.csv, which has
subject / stay IDs). It does NOT include clinical notes, so it cannot rebuild the full
108-feature model; it isolates the numeric half so each fix can be measured on its own.

Everything except the one thing under test is held fixed: same network as the project
(model_utils.ICUModel), same optimiser, same weighted loss, same seeds.

    python experiments/numeric_audit.py <path to data_values_exp.csv>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model_utils import ICUModel                                   # noqa: E402

torch.set_num_threads(4)
NUM = ["HR_mean", "HR_std", "RR_mean", "SpO2_mean", "SpO2_min", "Temp_mean", "SBP_mean", "DBP_mean",
       "MAP_mean", "latest_HR", "latest_RR", "latest_SpO2", "latest_Temp", "latest_SBP", "latest_DBP",
       "latest_MAP", "GCS_eye_opening", "stress_score"]
# physiologically possible ranges (same as the app's VALID_RANGES); Temp handled separately
RANGE = {"HR": (30, 220), "RR": (5, 60), "SpO2": (50, 100), "SBP": (40, 250), "DBP": (20, 150),
         "MAP": (30, 200)}


# ── preprocessing ────────────────────────────────────────────────────────────
def prep_current(df):
    """What the deployed model was trained on: raw values, missing -> 0, mixed units, outliers kept."""
    return df[NUM].fillna(0.0).to_numpy(np.float32), None


def _temp_c(v):
    """Mixed °C/°F column -> °C; anything implausible -> NaN."""
    v = v.copy()
    f = (v >= 86) & (v <= 110)
    v[f] = (v[f] - 32.0) * 5.0 / 9.0
    v[(v < 30) | (v > 43)] = np.nan
    return v


def prep_clean_fit(df):
    """Fit imputation values on the TRAIN rows only."""
    X = _clean_raw(df)
    med = X.median()
    miss_cols = [c for c in NUM if X[c].isna().mean() > 0.02]
    return {"med": med, "miss_cols": miss_cols}


def _clean_raw(df):
    X = df[NUM].copy()
    for c in NUM:
        base = c.replace("latest_", "").replace("_mean", "").replace("_min", "").replace("_std", "")
        if base in RANGE:
            lo, hi = RANGE[base]
            X.loc[(X[c] < lo) | (X[c] > hi), c] = np.nan
    X["Temp_mean"] = _temp_c(X["Temp_mean"]); X["latest_Temp"] = _temp_c(X["latest_Temp"])
    X.loc[(X["HR_std"] < 0) | (X["HR_std"] > 60), "HR_std"] = np.nan
    return X


def prep_clean_apply(df, fit):
    X = _clean_raw(df)
    ind = {f"{c}__missing": X[c].isna().astype(np.float32) for c in fit["miss_cols"]}
    # a "latest" reading is one sample from the same window, so its window mean is the best fallback
    for c in ("SpO2", "Temp", "SBP", "DBP", "MAP", "HR", "RR"):
        X[f"latest_{c}"] = X[f"latest_{c}"].fillna(X[f"{c}_mean"])
    X = X.fillna(fit["med"])
    out = pd.concat([X, pd.DataFrame(ind, index=X.index)], axis=1)
    return out.to_numpy(np.float32)


# ── training / metrics ───────────────────────────────────────────────────────
def fit_mlp(Xtr, ytr, Xva, yva, k_weight=0.5, epochs=40, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = ICUModel(Xtr.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(Xva)
    best, best_state = 1e9, None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            xb, yb = Xt[idx], yt[idx]
            w = 1.0 + yb * k_weight
            w = w / w.mean()
            loss = (w * (m(xb) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            mae = float(np.abs(m(Xv).numpy().ravel() - yva).mean())
        if mae < best:
            best, best_state = mae, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    return m


def predict(m, X):
    with torch.no_grad():
        return np.clip(m(torch.tensor(X)).numpy().ravel(), 0, 24)


def metrics(y, p):
    hi, alert = y >= 10, p >= 8
    return {
        "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p),
        "AUROC>=10": roc_auc_score(hi, p), "AUROC>=8": roc_auc_score(y >= 8, p),
        "recall>=10 @pred>=8": float((alert & hi).sum() / max(hi.sum(), 1)),
        "precision @pred>=8": float((alert & hi).sum() / max(alert.sum(), 1)),
        "MAE (SOFA>=10)": float(np.abs(p[hi] - y[hi]).mean()),
        "mean pred (SOFA>=10)": float(p[hi].mean()),
    }


def split_val(tr_idx, groups, frac=0.1, seed=0):
    g = np.unique(groups[tr_idx]); rng = np.random.default_rng(seed)
    val_g = set(rng.choice(g, int(len(g) * frac), replace=False))
    m = np.array([x in val_g for x in groups[tr_idx]])
    return tr_idx[~m], tr_idx[m]


def run_cv(df, y, groups, pipeline, split="grouped", k_weight=0.5, model="mlp"):
    folds = (GroupKFold(3).split(df, y, groups) if split == "grouped"
             else KFold(3, shuffle=True, random_state=0).split(df))
    rows = []
    for fi, (tr, te) in enumerate(folds):
        tr2, va = split_val(tr, groups)
        if pipeline == "current":
            Xtr, _ = prep_current(df.iloc[tr2]); Xva, _ = prep_current(df.iloc[va]); Xte, _ = prep_current(df.iloc[te])
        else:
            fit = prep_clean_fit(df.iloc[tr2])
            Xtr, Xva, Xte = (prep_clean_apply(df.iloc[i], fit) for i in (tr2, va, te))
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte = (np.clip(sc.transform(a), -10, 10).astype(np.float32) for a in (Xtr, Xva, Xte))
        if model == "mlp":
            p = predict(fit_mlp(Xtr, y[tr2], Xva, y[va], k_weight, seed=fi), Xte)
        else:
            gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=fi).fit(Xtr, y[tr2])
            p = np.clip(gb.predict(Xte), 0, 24)
        rows.append(metrics(y[te], p))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main(path):
    df = pd.read_csv(path)
    y = df["sofa_score"].to_numpy(np.float32)
    groups = df["subject_id"].to_numpy()
    print(f"{len(df):,} rows · {df.icustay_id.nunique():,} stays · {df.subject_id.nunique():,} patients\n")
    plan = [
        ("A. current preprocessing, ROW split (how it is evaluated today)", dict(pipeline="current", split="row")),
        ("B. current preprocessing, PATIENT split", dict(pipeline="current", split="grouped")),
        ("C. clean preprocessing,   PATIENT split", dict(pipeline="clean", split="grouped")),
        ("D. C + loss weight 0 (plain MSE)", dict(pipeline="clean", split="grouped", k_weight=0.0)),
        ("E. C + loss weight 1.5 (heavier on severe)", dict(pipeline="clean", split="grouped", k_weight=1.5)),
        ("F. C with gradient boosting (reference ceiling)", dict(pipeline="clean", split="grouped", model="gb")),
    ]
    out = {}
    for name, kw in plan:
        t = time.time(); r = run_cv(df, y, groups, **kw); out[name] = r
        print(f"{name}\n   " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()) + f"   [{time.time()-t:.0f}s]\n", flush=True)
    (ROOT / "experiments" / "numeric_audit_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "data_values_exp.csv"))

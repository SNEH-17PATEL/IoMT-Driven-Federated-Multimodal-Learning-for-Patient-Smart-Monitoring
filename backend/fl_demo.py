"""
Live federated-learning demo for the panel.

This is REAL federated training, not an animation: three simulated organisations each hold a private slice
of real MIMIC-derived data (18 numeric features, SOFA target), train the same network locally, and only
send weight updates to a server that averages them. The data are split by patient, never by row.

What it deliberately is NOT: the deployed model (that also uses note text and ran 100 rounds of
FedYogi + FedProx). This demo uses vitals + camera-proxy features only, because that is the data we hold with
patient identifiers, and it runs in seconds so it can be shown live.

Data file: data_values_exp.csv (raw BigQuery export, MIMIC licence: not stored in the repo).
Looked up in the folder above the project, or the path in the FL_DEMO_DATA environment variable.
"""
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score

from model_utils import ICUModel

ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
_sys.path.insert(0, str(ROOT))
DATA_PATH = Path(os.environ.get("FL_DEMO_DATA", ROOT.parent / "data_values_exp.csv"))
torch.set_num_threads(2)           # tiny network: more threads do not help

NUM = ["HR_mean", "HR_std", "RR_mean", "SpO2_mean", "SpO2_min", "Temp_mean", "SBP_mean", "DBP_mean", "MAP_mean",
       "latest_HR", "latest_RR", "latest_SpO2", "latest_Temp", "latest_SBP", "latest_DBP", "latest_MAP",
       "GCS_eye_opening", "stress_score"]
RANGE = {"HR": (30, 220), "RR": (5, 60), "SpO2": (50, 100), "SBP": (40, 250), "DBP": (20, 150), "MAP": (30, 200)}
SITES = [
    {"name": "Home-care provider", "short": "Home care", "blurb": "mostly stable patients", "affinity": -1.6},
    {"name": "Regional hospital", "short": "Regional", "blurb": "a mix of patients", "affinity": 0.0},
    {"name": "Tertiary ICU", "short": "Tertiary ICU", "blurb": "mostly very sick patients", "affinity": 1.6},
]
ALERT = 8.0
K_WEIGHT = 0.5            # same severity weighting as the project's loss: weight = 1 + 0.5 * SOFA


# ── data ─────────────────────────────────────────────────────────────────────
def _temp_c(v):
    v = v.copy()
    f = (v >= 86) & (v <= 110)
    v[f] = (v[f] - 32.0) * 5.0 / 9.0                     # the column mixes degrees C and F
    v[(v < 30) | (v > 43)] = np.nan
    return v


def _clean(df):
    X = df[NUM].copy()
    for c in NUM:
        base = c.replace("latest_", "").replace("_mean", "").replace("_min", "").replace("_std", "")
        if base in RANGE:
            lo, hi = RANGE[base]
            X.loc[(X[c] < lo) | (X[c] > hi), c] = np.nan
    X["Temp_mean"], X["latest_Temp"] = _temp_c(X["Temp_mean"]), _temp_c(X["latest_Temp"])
    X.loc[(X["HR_std"] < 0) | (X["HR_std"] > 60), "HR_std"] = np.nan
    return X


_DATA = {}


def load():
    """Clean, split by patient, standardise. Cached after the first call."""
    if _DATA:
        return _DATA
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Demo data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    rng = np.random.default_rng(0)
    pats = df["subject_id"].unique()
    test_p = set(rng.choice(pats, int(len(pats) * 0.2), replace=False))
    is_test = df["subject_id"].isin(test_p).to_numpy()
    X = _clean(df)
    miss_cols = [c for c in NUM if X.loc[~is_test, c].isna().mean() > 0.02]
    ind = pd.DataFrame({f"{c}__m": X[c].isna().astype(np.float32) for c in miss_cols})
    for c in ("SpO2", "Temp", "SBP", "DBP", "MAP", "HR", "RR"):
        X[f"latest_{c}"] = X[f"latest_{c}"].fillna(X[f"{c}_mean"])
    X = X.fillna(X[~is_test].median())
    M = pd.concat([X, ind], axis=1).to_numpy(np.float32)
    mu, sd = M[~is_test].mean(0), M[~is_test].std(0) + 1e-6
    M = np.clip((M - mu) / sd, -10, 10).astype(np.float32)
    y = df["sofa_score"].to_numpy(np.float32)
    _DATA.update(X=M, y=y, subj=df["subject_id"].to_numpy(), is_test=is_test, n_feat=M.shape[1])
    return _DATA


def partition(mode, seed=1):
    """Assign TRAIN patients to 3 organisations. 'even' = random; 'uneven' = severity-skewed (realistic)."""
    d = load()
    tr = np.where(~d["is_test"])[0]
    subj, y = d["subj"][tr], d["y"][tr]
    ug = np.unique(subj)
    rng = np.random.default_rng(seed)
    if mode == "even":
        site_of = dict(zip(ug, rng.integers(0, 3, len(ug))))
    else:
        sev = pd.Series(y).groupby(subj).max()
        z = ((sev - sev.mean()) / sev.std()).to_numpy()
        aff = np.array([s["affinity"] for s in SITES])
        logits = np.outer(z, aff)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        site_of = {g: rng.choice(3, p=pi) for g, pi in zip(sev.index, p)}
    sid = np.array([site_of[s] for s in subj])
    return [tr[sid == k] for k in range(3)]


def site_stats(idx_list):
    d = load()
    out = []
    for k, idx in enumerate(idx_list):
        yy = d["y"][idx]
        out.append({"name": SITES[k]["name"], "short": SITES[k]["short"], "blurb": SITES[k]["blurb"],
                    "rows": int(len(idx)), "patients": int(len(np.unique(d["subj"][idx]))),
                    "mean_sofa": float(yy.mean()), "pct_high": float((yy >= ALERT).mean()),
                    "raw_bytes": int(len(idx) * d["n_feat"] * 4)})
    return out


# ── model, training, evaluation ──────────────────────────────────────────────
def new_model(seed=0):
    torch.manual_seed(seed)
    return ICUModel(load()["n_feat"])


def flat(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def set_flat(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel()
        p.data.copy_(v[i:i + n].view_as(p))
        i += n


def train_local(model, idx, epochs, lr=5e-3, mu=0.0, global_vec=None, seed=0):
    d = load()
    X = torch.tensor(d["X"][idx]); y = torch.tensor(d["y"][idx]).unsqueeze(1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    loss_v = 0.0
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), 2048):
            b = perm[i:i + 2048]
            w = 1.0 + y[b] * K_WEIGHT
            w = w / w.mean()
            loss = (w * (model(X[b]) - y[b]) ** 2).mean()
            if mu > 0 and global_vec is not None:                       # FedProx: stay close to the global model
                prox, off = 0.0, 0
                for prm in model.parameters():
                    n = prm.numel()
                    prox = prox + ((prm - global_vec[off:off + n].view_as(prm)) ** 2).sum()
                    off += n
                loss = loss + (mu / 2) * prox
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        loss_v = tot / len(X)
    return loss_v


def evaluate(model):
    d = load()
    te = np.where(d["is_test"])[0]
    model.eval()
    with torch.no_grad():
        p = np.clip(model(torch.tensor(d["X"][te])).numpy().ravel(), 0, 24)
    y = d["y"][te]
    band = lambda m: float(np.abs(p[m] - y[m]).mean())
    return {"mae": float(mean_absolute_error(y, p)), "auroc": float(roc_auc_score(y >= ALERT, p)),
            "recall_alert": float(((p >= ALERT) & (y >= ALERT)).sum() / max((y >= ALERT).sum(), 1)),
            # who is the model wrong about? error on mild / moderate / severe patients separately
            "mae_mild": band(y < 5), "mae_moderate": band((y >= 5) & (y < 10)), "mae_severe": band(y >= 10)}


def sample_idx(n, k=40):
    return np.linspace(0, n - 1, k).astype(int)


# ── job runner (live, streams events to the UI) ──────────────────────────────
class Job:
    def __init__(self, cfg):
        self.id = uuid.uuid4().hex[:8]
        self.cfg = cfg
        self.events = []
        self.state = "running"
        self.cancel = False
        self.error = None

    def emit(self, **ev):
        self.events.append(ev)

    def pause(self, secs):
        t = time.time()
        while time.time() - t < secs and not self.cancel:
            time.sleep(0.05)

    def run(self):
        try:
            self._run()
            self.state = "cancelled" if self.cancel else "done"
        except Exception as e:                                   # surface to the page
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"

    def _run(self):
        c = self.cfg
        rounds, epochs, mu = c["rounds"], c["epochs"], (c["mu"] if c["algo"] == "fedprox" else 0.0)
        delay = {"slow": 0.9, "normal": 0.45, "fast": 0.05}[c["speed"]]
        sigma = {"off": 0.0, "low": 0.1, "high": 0.5}[c["privacy"]]
        parts = partition(c["split"])
        stats = site_stats(parts)
        gm = new_model(0)
        n_params = int(flat(gm).numel())
        self.emit(type="setup", sites=stats, n_params=n_params, upload_bytes=n_params * 4,
                  config=c, test_rows=int(load()["is_test"].sum()))
        # reference lines (each site alone, and all data pooled) train in the background while the rounds play
        budget = rounds * epochs

        def baselines():
            local = []
            for k in range(3):
                m = new_model(0)
                train_local(m, parts[k], budget, seed=k)
                local.append(evaluate(m))
            pooled = new_model(0)
            train_local(pooled, np.concatenate(parts), budget, seed=9)
            self.emit(type="baselines", local=local, central=evaluate(pooled))

        bt = threading.Thread(target=baselines, daemon=True)
        bt.start()
        init = evaluate(gm)
        self.emit(type="round_end", round=0, metrics=init, sample=flat(gm)[sample_idx(n_params)].tolist())
        rng = np.random.default_rng(5)
        for r in range(1, rounds + 1):
            if self.cancel:
                return
            gv = flat(gm).clone()
            self.emit(type="phase", round=r, phase="broadcast"); self.pause(delay)
            self.emit(type="phase", round=r, phase="local")
            updates, sizes = [], []
            for k in range(3):
                m = new_model(0); set_flat(m, gv)
                loss = train_local(m, parts[k], epochs, lr=max(1e-3, 5e-3 * 0.85 ** (r - 1)), mu=mu, global_vec=gv,
                                   seed=r * 10 + k)
                upd = flat(m) - gv
                norm = float(upd.norm())
                if sigma > 0:                                    # privacy: clip the update, add Gaussian noise
                    upd = upd * min(1.0, 1.0 / (norm + 1e-9))
                    upd = upd + torch.tensor(rng.normal(0, sigma / np.sqrt(n_params), n_params), dtype=torch.float32)
                updates.append(upd); sizes.append(len(parts[k]))
                self.emit(type="site", round=r, site=k, loss=float(loss), update_norm=norm,
                          sample=(gv + upd)[sample_idx(n_params)].tolist())
            self.pause(delay)
            self.emit(type="phase", round=r, phase="upload"); self.pause(delay)
            self.emit(type="phase", round=r, phase="aggregate"); self.pause(delay)
            w = np.array(sizes, dtype=np.float64); w /= w.sum()
            agg = sum(float(wi) * u for wi, u in zip(w, updates))
            set_flat(gm, gv + agg)
            self.emit(type="round_end", round=r, metrics=evaluate(gm), sample=flat(gm)[sample_idx(n_params)].tolist(),
                      weights=[float(x) for x in w])
            self.pause(delay * 0.5)
        bt.join()


_JOBS = {}
_LOCK = threading.Lock()


def start(cfg):
    with _LOCK:
        for j in _JOBS.values():
            j.cancel = True
        job = Job(cfg)
        _JOBS.clear()
        _JOBS[job.id] = job
        threading.Thread(target=job.run, daemon=True).start()
        return job.id


def events(job_id, since=0):
    j = _JOBS.get(job_id)
    if j is None:
        return {"state": "unknown", "events": [], "next": since}
    return {"state": j.state, "error": j.error, "events": j.events[since:], "next": len(j.events)}


def stop(job_id):
    j = _JOBS.get(job_id)
    if j:
        j.cancel = True
    return {"ok": True}


def available():
    return DATA_PATH.exists()

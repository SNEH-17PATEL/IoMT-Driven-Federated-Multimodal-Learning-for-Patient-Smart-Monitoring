"""
Training-only tuning harness (no Flower / Ray needed).

Re-implements the same federated loop as train_federated.py -- 3 IID hospitals,
FedAvg aggregation weighted by sample count, FedProx local training, AdamW, LR decay --
using model_utils.train_model, so a config that wins here transfers directly to
train_federated.py.  It never touches the data or the test set; the test set is the
fixed held-out fl_test_X.npy / fl_test_y.npy.

Model selection uses the clients' own validation split (as the real script does),
NOT the test set; the test set is scored once per config at the selected round.

Usage:
    python tune_training.py                 # run the default grid
    python tune_training.py --rounds 60 --quick
    python tune_training.py --synthetic     # smoke-test the harness on fake data
"""
import argparse, itertools, json, time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from model_utils import ICUModel, train_model, evaluate_model, get_weights, set_weights

MODEL_PATH = "models/"


def load_clients(synthetic=False, data_dir=None):
    if synthetic:
        rng = np.random.default_rng(0)
        w = rng.normal(size=108)
        def mk(n):
            X = rng.normal(size=(n, 108)).astype(np.float32)
            y = np.clip(4 + X @ w * 0.3 + rng.normal(size=n), 0, 24).astype(np.float32)
            return X, y
        clients = [mk(3000) for _ in range(3)]
        return clients, mk(2000)
    import joblib, sqlite3, pandas as pd
    mp = (data_dir.rstrip("/\\") + "/models/") if data_dir else MODEL_PATH
    cols = list(joblib.load(mp + "scaler.pkl").feature_names_in_)
    con = sqlite3.connect(f"file:{mp}icu_monitor.db?mode=ro", uri=True)
    clients = []
    for i in range(3):
        df = pd.read_sql("SELECT * FROM fl_training WHERE client_id = ?", con, params=(i,))
        clients.append((df[cols].values.astype(np.float32),
                        df["sofa_score"].values.astype(np.float32)))
    con.close()
    return clients, (np.load(mp + "fl_test_X.npy"), np.load(mp + "fl_test_y.npy"))


def run(cfg, clients, test, seed=42, return_model=False):
    torch.manual_seed(seed); np.random.seed(seed)
    splits = []
    for X, y in clients:
        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.15, random_state=42)
        splits.append((Xt, yt, Xv, yv))
    input_dim = clients[0][0].shape[1]
    gmodel = ICUModel(input_dim, hidden=cfg["hidden"], dropout=cfg["dropout"])
    gw = get_weights(gmodel)
    best_val, best_w, best_round = float("inf"), gw, 0
    work = ICUModel(input_dim, hidden=cfg["hidden"], dropout=cfg["dropout"])
    for rnd in range(1, cfg["rounds"] + 1):
        lr = max(1e-4, cfg["lr"] * cfg["lr_decay"] ** (rnd - 1))
        updates, sizes = [], []
        for Xt, yt, _, _ in splits:
            set_weights(work, gw)
            train_model(work, Xt, yt, epochs=cfg["epochs"], lr=lr, batch_size=cfg["batch"],
                        grad_clip=1.0, global_params=gw, mu=cfg["mu"],
                        weighted_loss=cfg["weighted"])
            updates.append(get_weights(work)); sizes.append(len(Xt))
        tot = sum(sizes)
        gw = [sum(u[k] * s for u, s in zip(updates, sizes)) / tot for k in range(len(gw))]
        set_weights(gmodel, gw)
        val = float(np.mean([evaluate_model(gmodel, Xv, yv)["mse"] for _, _, Xv, yv in splits]))
        if val < best_val:
            best_val, best_w, best_round = val, [w.copy() for w in gw], rnd
    set_weights(gmodel, best_w)
    gmodel.eval()
    with torch.no_grad():
        p = np.clip(gmodel(torch.tensor(test[0])).numpy().ravel(), 0, 24)
    result = dict(best_round=best_round, val_mse=best_val,
                  r2=float(r2_score(test[1], p)), mae=float(mean_absolute_error(test[1], p)))
    return (result, gmodel) if return_model else result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-dir", default=None, help="folder containing models/ with icu_monitor.db + test npy")
    a = ap.parse_args()
    clients, test = load_clients(a.synthetic, a.data_dir)
    print(f"clients: {[len(c[1]) for c in clients]}  test: {len(test[1])}")

    base = dict(hidden=(256, 128, 64), dropout=0.3, lr=1e-3, lr_decay=0.99, epochs=3,
                batch=64, mu=0.5, weighted=False, rounds=a.rounds)
    grid = {
        "epochs": [3, 5],
        "mu": [0.5, 0.05, 0.0],
        "lr": [1e-3, 3e-3],
        "batch": [64, 128],
        "hidden": [(256, 128, 64), (512, 256, 128)],
        "dropout": [0.3, 0.1],
    }
    # one-factor-at-a-time sweep from the current config, then combine the winners
    results = []
    t0 = time.time()
    r = run(base, clients, test); results.append(("baseline", base, r)); print("baseline", r)
    winners = {}
    for key, vals in grid.items():
        if a.quick and key in ("batch", "hidden"):
            continue
        for v in vals:
            if v == base[key]:
                continue
            cfg = {**base, key: v}
            r = run(cfg, clients, test); results.append((f"{key}={v}", cfg, r))
            print(f"{key}={v}", r)
            if r["val_mse"] < results[0][2]["val_mse"] and (key not in winners or r["val_mse"] < winners[key][1]):
                winners[key] = (v, r["val_mse"])
    if winners:
        cfg = {**base, **{k: v for k, (v, _) in winners.items()}}
        r = run(cfg, clients, test); results.append(("combined", cfg, r)); print("combined", cfg, r)
    print(f"\n{'config':28s} {'round':>5s} {'val_mse':>8s} {'MAE':>7s} {'R2':>7s}")
    for name, _, r in results:
        print(f"{name:28s} {r['best_round']:5d} {r['val_mse']:8.3f} {r['mae']:7.3f} {r['r2']:7.4f}")
    with open("tune_results.json", "w") as f:
        json.dump([dict(name=n, cfg={k: list(v) if isinstance(v, tuple) else v for k, v in c.items()}, **r)
                   for n, c, r in results], f, indent=2)
    print(f"done in {time.time()-t0:.0f}s -> tune_results.json")


if __name__ == "__main__":
    main()

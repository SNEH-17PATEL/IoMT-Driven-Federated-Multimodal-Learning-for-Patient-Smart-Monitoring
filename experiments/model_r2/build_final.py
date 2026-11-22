"""Train the final federated model (same algorithm as train_federated.py) and write all artifacts to scratch/final/."""
import sys, os, json, time, shutil
import numpy as np, torch, joblib
REPO = r"<REPO>"
SCR = r"<WORKDIR>"
WORK = SCR + r"\work"; OUT = SCR + r"\final"
sys.path.insert(0, REPO)
import tune_training as t
from model_utils import compute_sofa_risk_metrics, estimate_privacy_budget
from sklearn.metrics import r2_score, mean_absolute_error

os.makedirs(OUT, exist_ok=True)
clients, (Xte, yte) = t.load_clients(False, WORK)
cfg = dict(hidden=(256, 128, 64), dropout=0.3, lr=1e-3, lr_decay=0.99, epochs=3, batch=64, mu=0.5, weighted=False, rounds=100)
t0 = time.time()
res, model = t.run(cfg, clients, (Xte, yte), return_model=True)
print("TRAINED", res, round(time.time() - t0), "s", flush=True)

torch.save(model.state_dict(), OUT + r"\federated_model.pth")
model.eval()
with torch.no_grad():
    preds = np.clip(model(torch.tensor(Xte)).numpy().ravel(), 0, 24)
mae = mean_absolute_error(yte, preds); r2 = r2_score(yte, preds)
risk = compute_sofa_risk_metrics(yte, preds)
print("R2", round(r2, 4), "MAE", round(mae, 4), "alert", {k: risk["high_risk_alert"][k] for k in ("auroc", "pr_auc", "recall", "specificity")}, flush=True)

X_all = np.vstack([c[0] for c in clients])
rng = np.random.default_rng(42)
bg = X_all[rng.choice(len(X_all), 300, replace=False)]
np.save(OUT + r"\shap_background.npy", bg)

for f in ("scaler.pkl", "tfidf_vectorizer.pkl", "feature_columns.pkl", "fl_test_X.npy", "fl_test_y.npy"):
    shutil.copy(WORK + "\\models\\" + f, OUT + "\\" + f)

tf = joblib.load(OUT + r"\tfidf_vectorizer.pkl")
d_in = X_all.shape[1]
meta = {
    "num_rounds": cfg["rounds"], "epochs_per_round": cfg["epochs"], "batch_size": cfg["batch"],
    "base_lr": cfg["lr"], "lr_decay": cfg["lr_decay"], "hospitals": 3,
    "hospital_names": ["General ICU", "Mixed ICU", "Cardiac/Trauma ICU"],
    "aggregation": "FedAvg + FedProx (\u03bc=0.5)", "split_type": "IID",
    "train_samples": int(X_all.shape[0]), "test_samples": int(len(yte)),
    "input_features": int(d_in),
    "model_architecture": f"{d_in} \u2192 256 \u2192 128 \u2192 64 \u2192 1  (ReLU, Dropout 0.3, FedProx)",
    "best_round": int(res["best_round"]), "final_mae": round(float(mae), 4), "final_r2": round(float(r2), 4),
    "pred_range_min": round(float(preds.min()), 2), "pred_range_max": round(float(preds.max()), 2),
    "oversample": False, "loss_weights": "None \u2014 plain MSE", "optimizer": "AdamW (weight_decay=1e-4)",
    "differential_privacy": False, "dp_sensitivity": None, "dp_sigma": None, "dp_epsilon": None, "dp_delta": None,
    "bigquery_project": "mimic-project-2",
    "tfidf_features": int(len(tf.get_feature_names_out())), "tfidf_ngram": [1, 2],
    "note_vocabulary": "200 words chosen from training notes only (candidate pool 3,000 uni/bigrams, XGBoost gain ranking), English stop-words removed",
    "risk_classification_metrics": risk,
    "training_run": ("Federated loop run with tune_training.py (identical algorithm to train_federated.py: FedAvg, FedProx, "
                     "3 IID hospitals, AdamW, LR decay, best round by client validation MSE) because Flower/Ray cannot run on Windows."),
    "evaluation_limitation": ("One row per ICU stay; train/test split 80/20 GROUPED BY PATIENT (no patient on both sides). "
                              "Retrospective MIMIC-III results, not independent clinical validation."),
    "sklearn_version": __import__("sklearn").__version__, "torch_version": torch.__version__,
}
with open(OUT + r"\training_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print("BUILD DONE", os.listdir(OUT), flush=True)

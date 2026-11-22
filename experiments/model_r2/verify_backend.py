import sys, os
REPO = r"<REPO>"
os.chdir(REPO); sys.path.insert(0, REPO)
import numpy as np, pandas as pd, torch, shap
from backend.services import ml_service as ml
model, scaler, tfidf, feature_cols, bg = ml.load_artifacts()
print("input_dim", len(feature_cols), "model first layer", tuple(model.net[0].weight.shape), "bg", bg.shape)
note = ("Patient intubated on ventilator, sedated, on levophed for septic shock, "
        "acute kidney injury with rising creatinine, hypotension, acidosis, acute distress.")
tv = tfidf.transform([note]); cols = list(tfidf.get_feature_names_out())
nz = [(cols[i], round(float(tv[0, i]), 3)) for i in tv[0].nonzero()[1]]
print("note words picked up:", nz)
vit = dict(HR_mean=110, HR_std=12, RR_mean=26, SpO2_mean=90, SpO2_min=84, Temp_mean=38.4, SBP_mean=92, DBP_mean=50,
           MAP_mean=62, latest_HR=118, latest_RR=28, latest_SpO2=88, latest_Temp=38.6, latest_SBP=88, latest_DBP=48,
           latest_MAP=58, GCS_eye_opening=2.0, stress_score=6.0)
df = pd.concat([pd.DataFrame([vit]), pd.DataFrame(tv.toarray(), columns=cols)], axis=1).fillna(0)
for c in feature_cols:
    if c not in df.columns: df[c] = 0
df = df[list(feature_cols)].astype(float)
X = torch.tensor(np.clip(scaler.transform(df), -10, 10), dtype=torch.float32)
model.eval()
with torch.no_grad(): print("predicted SOFA:", round(float(model(X).item()), 2))
sv = shap.DeepExplainer(model, torch.tensor(bg)).shap_values(X)
sv = np.array(sv[0] if isinstance(sv, list) else sv).reshape(-1)
top = sorted(zip(feature_cols, sv), key=lambda t: -abs(t[1]))[:8]
print("top SHAP features:", [(f, round(float(v), 2)) for f, v in top])
print("shap sum + base ~ pred check ok" if np.isfinite(sv).all() else "SHAP NaN")

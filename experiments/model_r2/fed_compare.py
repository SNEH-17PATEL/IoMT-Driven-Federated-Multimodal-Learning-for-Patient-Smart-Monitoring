"""Federated comparison on the SAME patient-grouped split.  usage: python fed_compare.py new|old [rounds]"""
import sys, json, time
import numpy as np, pandas as pd, torch
REPO = r"<REPO>"
SCR = r"<WORKDIR>"
sys.path.insert(0, REPO)
import tune_training as t
torch.set_num_threads(4)
mode = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 40
if mode == "new":
    clients, test = t.load_clients(False, SCR + r"\work")
else:
    import joblib
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from model_utils import clean_clinical_text
    B = SCR + r"\handoff\handoff_full"
    d = pd.read_parquet(B + r"\bigquery_tables\ml_dataset_final_last_row_per_stay.parquet").sort_values("icustay_id").reset_index(drop=True)
    n = pd.read_parquet(B + r"\bigquery_tables\clinical_notes.parquet")
    n = n.groupby("hadm_id")["text"].apply(lambda x: " ".join(v for v in x if isinstance(v, str))).rename("note_text")
    d = d.merge(n, on="hadm_id", how="left"); d["note_text"] = d["note_text"].fillna("")
    tr, te = next(GroupShuffleSplit(1, test_size=.2, random_state=42).split(d, groups=d.subject_id))
    wl = list(joblib.load(B + r"\models\tfidf_vectorizer.pkl").get_feature_names_out())
    v = TfidfVectorizer(vocabulary=wl, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",
                        sublinear_tf=True, norm="l2", preprocessor=clean_clinical_text)
    T = v.fit(d.note_text.iloc[tr]).transform(d.note_text).toarray()
    num = [c for c in d.columns if c not in ("subject_id", "hadm_id", "icustay_id", "window_time", "sofa_score", "rn", "note_text")]
    X = np.hstack([d[num].fillna(0).values, T]); y = d.sofa_score.values.astype(np.float32)
    sc = StandardScaler().fit(X[tr])
    Xs = np.clip(sc.transform(X), -10, 10).astype(np.float32)
    strat = (y[tr] >= 10).astype(int)
    a, bc = train_test_split(np.arange(len(tr)), test_size=2/3, random_state=42, stratify=strat)
    b, c = train_test_split(bc, test_size=.5, random_state=42, stratify=strat[bc])
    clients = [(Xs[tr][i], y[tr][i]) for i in (a, b, c)]
    test = (Xs[te], y[te])
print(mode, [len(c[1]) for c in clients], len(test[1]), clients[0][0].shape[1], flush=True)
base = dict(hidden=(256, 128, 64), dropout=0.3, lr=1e-3, lr_decay=0.99, epochs=3, batch=64, mu=0.5, weighted=False, rounds=rounds)
t0 = time.time(); r = t.run(base, clients, test)
print("RESULT", mode, json.dumps(r), round(time.time() - t0), "s", flush=True)

# experiments/model_r2 — scripts behind `docs/MODEL_IMPROVEMENT.md`

Every number in the report comes from one of these scripts. They are **reference/experiment code**, not part of the
product. Each has `<WORKDIR>` / `<REPO>` placeholders for local paths — replace them with your own folders
(`<WORKDIR>` is a folder that contains `handoff/handoff_full/` = the unzipped hand-off package; the MIMIC-III data is
restricted and is **not** in git).

| Script | What it measures | Report section |
|---|---|---|
| `ceil2.py`, `ceil3.py` | Centralised ceiling on the 108 original features: XGBoost (several settings, √ target) and a centralised MLP | 4 |
| `feat1.py` | Feature-set comparison (vitals only, 90 whitelist, 3,000 learned terms, + note length) on a patient-grouped split | 5 |
| `feat2.py` | Regex-extracted lab values from notes (no gain) | 5 |
| `xgb10.py` | XGBoost ceiling with a 10,000-term vocabulary | 5 |
| `mlp1.py`, `mlp2.py`, `mlp3.py` | Can an MLP use the richer notes? raw terms, SVD sizes, dropout, 10k vocabulary (SVD was later rejected for SHAP reasons) | 6, 7 |
| `words1.py` | **Adopted idea:** top-N data-chosen words (XGBoost-ranked), regularisation variants, vs the 90 whitelist | 8 |
| `run_phase0.py` | Runs only Phase 0 of `train_federated.py` (parquet source, patient-grouped split, word selection) in a scratch copy | 8 |
| `fed_compare.py` | Federated 40-round run, new vs old features on the identical split | 9 |
| `fed_cfg.py` | Federated runs with overridden hyper-parameters (epochs, μ, lr, batch, rounds, network width) | 9 |
| `build_final.py` | Trains the final model (100 rounds, best round kept) and writes every artifact incl. metadata | 10 |
| `verify_backend.py` | Loads the artifacts through `backend.services.ml_service`, predicts for a synthetic severe patient, prints top SHAP features | 12 |

The federated harness these scripts call is `tune_training.py` in the repo root (same algorithm as
`train_federated.py`, no Flower/Ray needed). Run scripts from the repo root, e.g. `python experiments/model_r2/fed_compare.py new 40`
(set `PYTHONIOENCODING=utf-8` on Windows).

# Data Folder

This folder should contain three hospital client datasets used for Federated Learning training.

## Expected Files

| File | Size | Rows | Columns |
|---|---|---|---|
| `client_0.csv` | ~191 MB | 15,889 | 619 |
| `client_1.csv` | ~191 MB | 15,890 | 619 |
| `client_2.csv` | ~197 MB | 16,372 | 619 |

These files are **not included in the repository** because:
1. Each file is ~191–197 MB (exceeds GitHub's 100 MB file limit)
2. They are derived from the MIMIC-III clinical database (PhysioNet data use agreement applies)

## Column Structure

Each CSV contains **618 feature columns** (already StandardScaler-normalised) + `sofa_score`:

```
HR_mean, HR_std, RR_mean, SpO2_mean, SpO2_min, Temp_mean,
SBP_mean, DBP_mean, MAP_mean,           ← 9 trend features
latest_HR, latest_RR, latest_SpO2, latest_Temp,
latest_SBP, latest_DBP, latest_MAP,     ← 7 latest vitals
GCS_eye_opening, stress_score,           ← 2 CV features
[600 TF-IDF columns],                    ← clinical note features
sofa_score                               ← target label (0–24, raw)
```

## How to Obtain

### Option 1 — Run the original training notebook (requires BigQuery access)

1. Get MIMIC-III access at [https://physionet.org/content/mimiciii](https://physionet.org/content/mimiciii)
2. Load the dataset into Google BigQuery
3. Run the 12 SQL preprocessing queries (documented in `sql_queries.pdf` in the root capstone folder)
4. Run `notebooks/federated_learning.ipynb` in Google Colab with BigQuery auth
5. The notebook produces and saves `client_0/1/2.csv`

### Option 2 — Request from project author

Contact the project owner to obtain the pre-processed hospital simulation datasets.

## SOFA Distribution

All 3 clients have similar IID distributions:

| Risk Level | SOFA Range | % of data |
|---|---|---|
| Low Risk | 0 – 4 | 63.3% |
| Moderate Risk | 5 – 9 | 30.6% |
| High Risk | ≥ 10 | 6.0% |

**Note:** The data files are already StandardScaler-normalised.
Do NOT re-apply the scaler from `models/scaler.pkl` to these files.
The scaler is only applied at inference time (in `app.py`) to new user inputs.

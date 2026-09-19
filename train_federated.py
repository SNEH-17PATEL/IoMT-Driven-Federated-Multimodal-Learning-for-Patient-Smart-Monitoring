"""
Federated Learning Training Script  —  Unified Edition
=======================================================
Combines Phase 0 (data preprocessing from BigQuery) with Phase 1–6 (FL training).

Phase 0 — Preprocessing  (skipped automatically if client CSVs already exist,
                           or set SKIP_PREPROCESSING = True to force-skip)
  • Queries BigQuery: ml_dataset_final (last row per ICU stay) + clinical_notes
  • TF-IDF on clinical notes  (600 bigram features, 80K note sample)
  • Merges vitals + TF-IDF, drops IDs, fills NaN → fillna(0)
  • Train/test split  (80 / 20, stratified by High-Risk flag)
  • StandardScaler fit on X_train
  • Splits into 3 hospital client CSVs (IID 33/33/34)
  • Saves: scaler.pkl, tfidf_vectorizer.pkl, feature_columns.pkl, client_0/1/2.csv

Phase 1-6 — FL Training  (always runs)
  • Loads client CSVs (already StandardScaler-normalised)
  • SHAP background samples (300 rows from combined training data)
  • Flower FL simulation: 20 rounds, all-3-clients, SaveBestStrategy
      - Weighted MSE loss  (1 + SOFA×3  →  High-Risk gets ~2.4× gradient weight)
      - AdamW optimizer    (lr=0.001, weight_decay=1e-4)
      - LR scheduling via server_round config (LR_DECAY=1.0 = no decay by default)
      - Optional: Non-IID split, Differential Privacy, Oversampling
  • Saves: federated_model.pth, shap_background.npy, training_metadata.json

BigQuery prerequisites (Phase 0 only):
  1. pip install google-cloud-bigquery google-cloud-bigquery-storage
  2. gcloud auth application-default login
  3. Project mimic-project-2 must be accessible to your account

Usage:
    python train_federated.py          # preprocess from BigQuery + train FL
    python train_federated.py          # with SKIP_PREPROCESSING=True → train only
"""

import json
import warnings
import numpy as np
import pandas as pd
import torch
import joblib
import flwr as fl
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters, FitIns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from model_utils import (
    ICUModel, train_model, evaluate_model, get_weights, set_weights,
    apply_dp_to_update, estimate_privacy_budget,
)

# =============================================================
# ── CONFIG ────────────────────────────────────────────────────
# =============================================================

# ── Paths ──
DATA_PATH  = "data/fl_training/"
MODEL_PATH = "models/"

# ── BigQuery (Phase 0) ──
BIGQUERY_PROJECT  = "mimic-project-2"
BIGQUERY_DATASET  = "Dataset"   # actual dataset name in BigQuery console
BIGQUERY_LOCATION = "US"        # Data location shown in BigQuery → Dataset → Details

# Set True to skip BigQuery + TF-IDF entirely and go straight to FL training.
# Auto-set to True if all three client CSVs already exist.
SKIP_PREPROCESSING = False

# ── TF-IDF settings (Phase 0) ──
NOTES_SAMPLE_SIZE = 283_208  # increased from 80k → 200k: better IDF for rare SOFA terms
NOTES_TEXT_LIMIT  = 10_000   # increased from 5k → 10k: more lab value mentions per patient
TFIDF_MAX_FEATURES = 200
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MAX_DF       = 0.9
TFIDF_MIN_DF       = 10

# ── FL Training ──
NUM_ROUNDS       = 100   # 100 rounds — FedYogi consistently finds best at round 24-29
EPOCHS_PER_ROUND = 3     # sweet spot: 3E with FedYogi gave best R²=0.2229
BATCH_SIZE       = 64
SHAP_BG_SAMPLES  = 300

# ── Learning Rate Scheduling ──
BASE_LR   = 0.001   # restored — FedProx proximal term controls drift instead of LR
LR_DECAY  = 0.99   # slow decay: LR → 0.00074 at round 29 (higher than 0.00056 with 0.98)

# ── Gradient Clipping ──
GRAD_CLIP = 1.0    # clips gradient L2 norm — prevents single-batch loss spikes

# ── FedProx proximal regularisation ──
# Adds (μ/2) × ||w_local − w_global||² to each hospital's loss.
# Prevents client drift that causes FedAvg server loss to oscillate.
# μ=0.5: proximal gradient is ~50% of data gradient for typical drift magnitudes.
# Reverted from μ=1.0 which was too strong — model couldn't explore parameter space.
MU_FEDPROX = 0.5

# ── Server-side optimiser (FedAdam) ──
# FedYogi applies Yogi adaptive optimiser at the SERVER after FedAvg aggregation.
# Yogi differs from Adam in its v_t update:
#   FedAdam: v_t = β2·v_{t-1} + (1-β2)·Δ²          ← can grow unboundedly
#   FedYogi: v_t = v_{t-1} + (1-β2)·(Δ²-v_{t-1})·sign(Δ²-v_{t-1})
# Yogi only increases v_t when the current pseudo-gradient Δ² > running estimate.
# This prevents the step size 1/√v_t from exploding when a single bad round
# produces a large pseudo-gradient — the very failure mode that caused the
# 82× loss explosion seen with FedAdam η=0.01 (rounds 7-13 in a prior run).
# Same η=0.01 as the best FedAdam run → same exploration power, no explosion.
SERVER_ETA    = 0.01   # server-side step size — same as best FedAdam run (R²=0.2148)
SERVER_BETA1  = 0.9    # momentum decay (standard Adam)
SERVER_BETA2  = 0.99   # adaptive rate decay (standard Adam)
SERVER_TAU    = 1e-3   # stability constant

# ── Oversampling ──
OVERSAMPLE = False  # WeightedRandomSampler for High-Risk minority

# ── Hospital Data Split (affects FL training only) ──
USE_NONIID_SPLIT = False   # True = specialty-biased; reduces R² ~0.08 vs ~0.28 IID

# ── Differential Privacy ──
USE_DP         = False
DP_SENSITIVITY = 1.0
DP_SIGMA       = 1.0

# =============================================================
# ── HELPERS ───────────────────────────────────────────────────
# =============================================================
hospital_names = ["General ICU", "Mixed ICU", "Cardiac/Trauma ICU"]
_client_paths  = [DATA_PATH + f"client_{i}.csv" for i in range(3)]

import os
_csvs_exist = all(os.path.exists(p) for p in _client_paths)

# =============================================================
# ── PHASE 0: PREPROCESSING  (BigQuery → TF-IDF → Scale) ──────
# =============================================================
def run_preprocessing():
    """
    Download data from BigQuery, run TF-IDF, scale, split into 3 client CSVs
    and save all sklearn artifacts so train_federated.py can be re-run offline.
    """
    print("=" * 60)
    print("  PHASE 0 — Data Preprocessing from BigQuery")
    print("=" * 60)

    # ── 0a. BigQuery auth + data load ────────────────────────
    try:
        from google.cloud import bigquery
    except ImportError:
        raise ImportError(
            "google-cloud-bigquery not installed.\n"
            "Run: pip install google-cloud-bigquery google-cloud-bigquery-storage"
        )

    print(f"\n[0a] Connecting to BigQuery project: {BIGQUERY_PROJECT}  (location: {BIGQUERY_LOCATION})")
    bq = bigquery.Client(project=BIGQUERY_PROJECT, location=BIGQUERY_LOCATION)

    print("[0b] Querying ml_dataset_final (last row per ICU stay)...")
    query_vitals = f"""
    SELECT *
    FROM (
        SELECT *,
               ROW_NUMBER() OVER(
                   PARTITION BY icustay_id
                   ORDER BY window_time DESC
               ) AS rn
        FROM `{BIGQUERY_PROJECT}.{BIGQUERY_DATASET}.ml_dataset_final`
        WHERE sofa_score IS NOT NULL
    )
    WHERE rn = 1
    """
    ml_data = bq.query(query_vitals).to_dataframe()
    print(f"  Vitals dataset: {ml_data.shape}  (rows × cols)")

    print("[0c] Querying clinical_notes...")
    query_notes = f"""
    SELECT subject_id, hadm_id, text
    FROM `{BIGQUERY_PROJECT}.{BIGQUERY_DATASET}.clinical_notes`
    """
    notes = bq.query(query_notes).to_dataframe()
    print(f"  Notes dataset:  {notes.shape}")

    # ── 0d. TF-IDF on clinical notes ─────────────────────────
    print(f"\n[0d] Building TF-IDF  (sample {NOTES_SAMPLE_SIZE:,} notes, "
          f"{TFIDF_MAX_FEATURES} features, ngram {TFIDF_NGRAM_RANGE})...")

    import re
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

    # ── Text cleaning: removes MIMIC artifacts + numbers before TF-IDF ──
    def _clean_clinical_text(text):
        """
        Clean MIMIC-III clinical notes before TF-IDF.
        Removes:
          1. De-identification placeholders:  [**Name**], [**Date**], [**Hospital1**] etc.
          2. Pure numbers and numeric tokens:  120, 2.5, 120/80, 98.6
          3. Punctuation and special characters
        Leaves only lowercase alphabetic words for TF-IDF to process.
        """
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\[\*\*[^\]]*\*\*\]', ' ', text)   # [**...**] patterns
        text = re.sub(r'\b\d+(?:[./]\d+)?\b', ' ', text)  # standalone numbers
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)              # keep only letters + spaces
        return ' '.join(text.split())

    # ── Extended stop words (replaces the whitelist approach) ──
    # Rather than restricting to a fixed vocabulary (which reduces features
    # and loses data-driven signal), we use aggressive stop words to remove
    # noise while letting TF-IDF pick the top 600 most informative clean terms.
    # The _clean_clinical_text() function already removes numbers and MIMIC
    # de-identification artifacts before TF-IDF runs.
    # NOTE: _MEDICAL_VOCABULARY is kept here as a reference/documentation
    # of the clinical terms we expect to appear in the vocabulary, but it
    # is NOT passed as the vocabulary= parameter — TF-IDF is data-driven.
    _MEDICAL_VOCABULARY_REF = sorted(set([
        # ── Respiratory (SOFA component 1: PaO2/FiO2) ──
        "airway", "aspiration", "asthma", "atelectasis", "auscultation",
        "bipap", "breath", "breathing", "bronchial", "copd", "cpap",
        "crackles", "diminished", "dyspnea", "effusion", "extubated",
        "extubation", "fio", "hypoxia", "hypercapnia", "intubated", "intubation",
        "lungs", "nasal", "oxygen", "pleural", "pneumonia", "pna",
        "pulmonary", "respiratory", "secretions", "sob", "sputum",
        "stridor", "tachypnea", "ventilation", "ventilator", "wheezes",
        "breath sounds", "clear auscultation", "lungs clear",
        "mechanical ventilation", "positive pressure", "pulmonary edema",
        "respiratory distress", "respiratory failure", "respiratory rate",
        "shortness breath",

        # ── Cardiovascular (SOFA component 4: MAP/vasopressors) ──
        "afib", "arrhythmia", "arterial", "artery", "aorta", "aortic",
        "bradycardia", "cabg", "cad", "cardiac", "cardiac arrest",
        "cardiothoracic", "cardiovascular", "cardiogenic", "carotid", "chf",
        "coronary", "cva", "dobutamine", "dopamine",
        "ejection", "epinephrine", "femoral", "fibrillation",
        "hemodynamic", "hemodynamically", "htn", "hypotension", "hypertension",
        "ischemia", "lad", "lopressor", "mitral", "murmur", "myocardial",
        "norepinephrine", "pacemaker", "pericardial", "pressors",
        "rca", "regurgitation", "resuscitation", "shock",
        "stent", "stenosis", "systolic",
        "tachycardia", "vascular", "vasopressor", "vasopressors",
        "ventricular", "valve",
        "aortic valve", "artery bypass", "artery disease",
        "atrial fibrillation", "bypass graft", "cardiac arrest",
        "cardiac catheterization", "cardiac output", "cardiogenic shock",
        "coronary artery", "failure acute", "heart failure", "heart rate",
        "hemodynamically stable", "rate rhythm", "ventricular tachycardia",

        # ── Renal (SOFA component 6: creatinine/urine output) ──
        "angap", "bun", "creat", "creatinine", "dialysis", "hco",
        "oliguria", "phos", "phosphate", "renal", "urine", "urean",
        "angap blood", "creat hco", "hco angap",
        "acute kidney", "kidney injury", "renal failure",
        "urine output", "urean creat",

        # ── Hepatic (SOFA component 3: bilirubin) ──
        "anicteric", "ascites", "alt", "ast", "bilirubin", "cirrhosis",
        "coagulopathy", "hepatic", "hepatitis", "hepatorenal",
        "icterus", "jaundice", "liver", "transaminase",
        "end stage", "liver failure", "portal hypertension",

        # ── CNS (SOFA component 5: GCS) ──
        "agitation", "altered", "aphasia", "coma", "confused",
        "encephalopathy", "glasgow", "lethargy", "neuro",
        "obtunded", "pupils", "sedated", "sedation", "seizure",
        "alert oriented", "altered mental", "mental status", "neck supple",

        # ── Coagulation (SOFA component 2: platelets) ──
        "anticoagulation", "bleed", "bleeding", "coagulation",
        "coumadin", "hematoma", "hemorrhage", "heparin",
        "inr", "plt", "thrombosis", "transfusion",
        "blood ptt", "fresh frozen", "inr blood",
        "plt blood", "ptt inr", "rdw plt",

        # ── Infection / Sepsis ──
        "antibiotics", "bacteremia", "candida", "cellulitis",
        "cultures", "culture", "endocarditis", "fever", "fevers",
        "fungemia", "infection", "infectious", "meningitis",
        "mrsa", "sepsis", "septic", "septicemia",
        "vanco", "vancomycin",
        "blood cultures", "gram negative", "gram positive",

        # ── ICU laboratory values ──
        "albumin", "arterial", "bicarbonate", "bnp", "calcium",
        "chloride", "fibrinogen", "glucose", "hct", "hematocrit",
        "hgb", "lactate", "magnesium", "mch", "mchc", "mcv",
        "phosphate", "platelets", "potassium", "procalcitonin",
        "rbc", "rdw", "sodium", "troponin", "wbc",
        "arterial blood", "blood glucose", "blood gas",
        "glucose urean", "hct mcv", "hgb hct",
        "mch mchc", "mchc rdw", "mcv mch",
        "platelet count", "potassium chloride",
        "rbc hgb", "sodium potassium", "wbc rbc",

        # ── ICU medications ──
        "amiodarone", "aspirin", "ativan", "bolus", "coumadin",
        "dilaudid", "dobutamine", "dopamine", "drip",
        "epinephrine", "fentanyl", "foley", "gtt",
        "heparin", "insulin", "ivf", "lasix", "lopressor",
        "metoprolol", "morphine", "norepinephrine",
        "propofol", "rocuronium", "steroids", "tylenol",
        "vanco", "vasopressin",

        # ── Clinical symptoms / signs ──
        "abdominal", "afebrile", "bilateral", "bilaterally",
        "chills", "cough", "diarrhea", "distended", "distress",
        "dyspnea", "edema", "hemoptysis", "nausea",
        "nontender", "pain", "sob", "syncope",
        "tender", "tenderness", "vomiting", "weakness", "worsening",
        "abdominal pain", "bowel sounds", "chest pain",
        "nausea vomiting", "pain control",

        # ── Diagnoses / medical history ──
        "anemia", "aneurysm", "cancer",
        "diabetes", "hyperlipidemia", "hypertension",
        "infarction", "ischemia", "stroke",
        "atrial fibrillation", "chronic pain",
        "diabetes mellitus", "heart failure",
        "myocardial infarction", "renal failure",

        # ── Anatomy (clinically relevant) ──
        "abd", "abdomen", "cardiac", "carotid",
        "coronary", "femoral", "hepatic", "lungs",
        "mitral", "pleural", "pulmonary", "renal",
        "respiratory", "ventricular",
        "abd soft", "abdomen soft", "intensive care",

        # ── Procedures ──
        "bypass", "cabg", "catheter", "catheterization", "cath",
        "chest tube", "dialysis", "drainage", "extubation",
        "foley", "intubation", "paracentesis", "repair",
        "replacement", "resection", "stent", "surgery",
        "surgical", "thoracentesis", "transplant",
        "artery bypass", "bypass graft", "cardiac catheterization",
        "central line", "fluid resuscitation",
        "invasive procedure", "surgical invasive",

        # ── Vital sign clinical references ──
        "bradycardia", "hypotensive", "pulse", "pulses",
        "sat", "sats", "saturation", "tachycardia", "temperature",

        # ── Other important ICU terms ──
        "blood", "crrt", "icu", "micu", "nad", "resp", "stable",
        "blood pressure", "blood wbc",
    ]))

    # ── Sample, group by admission, clean, limit ─────────────
    notes_sampled = notes.sample(n=min(NOTES_SAMPLE_SIZE, len(notes)), random_state=42)
    notes_combined = (
        notes_sampled
        .groupby("hadm_id")["text"]
        .apply(lambda x: " ".join(x))
        .reset_index()
    )
    # Clean BEFORE applying the character limit so all 3000 chars are
    # meaningful clinical content, not MIMIC de-identification artifacts.
    notes_combined["text"] = (
        notes_combined["text"]
        .fillna("")
        .apply(_clean_clinical_text)
        .str[:NOTES_TEXT_LIMIT]
    )

    # ── SOFA-component vocabulary whitelist ──────────────────────
    # Rather than data-driven selection (which picks statistically frequent
    # terms like "tablet", "sig", "refills" that correlate with documentation
    # style rather than patient severity), we use an explicit whitelist of
    # terms that directly encode SOFA organ-component information from notes.
    # sklearn ignores max_features/min_df/max_df/stop_words when vocabulary= is set.
    _SOFA_VOCABULARY = sorted({
        # Component 1: Respiratory (PaO2/FiO2 proxy from notes)
        "intubated", "intubation", "ventilator", "vent", "cpap", "bipap",
        "hypoxia", "respiratory", "extubated", "extubation",
        "respiratory failure", "respiratory distress",
        "shortness breath",
        # Component 2: Coagulation (Platelet proxy from notes)
        "plt", "platelets", "bleed", "bleeding", "hemorrhage",
        "inr", "ptt", "coagulopathy", "thrombocytopenia",
        "plt blood", "ptt inr",
        # Component 3: Hepatic (Bilirubin proxy — notes use "totbili"/"bili")
        "bilirubin", "bili", "totbili", "liver", "jaundice",
        "hepatic", "cirrhosis", "liver failure",
        # Component 4: Cardiovascular (MAP / vasopressors from notes)
        "hypotension", "hypotensive", "shock", "vasopressor", "pressors",
        "norepinephrine", "levophed", "dopamine", "epinephrine", "dobutamine",
        "septic shock",
        # Component 5: CNS (GCS proxy from notes)
        "sedated", "sedation", "coma", "altered", "confused", "gcs",
        "encephalopathy", "delirium", "unresponsive",
        "altered mental",
        # Component 6: Renal (Creatinine / urine output from notes)
        "creatinine", "creat", "renal", "kidney", "dialysis", "aki",
        "oliguria", "urine", "urean", "crrt",
        "urine output", "renal failure", "acute kidney",
        # Sepsis / infection (primary driver of multi-organ SOFA elevation)
        "sepsis", "septic", "bacteremia", "infection", "cultures",
        "antibiotics", "fever",
        "blood cultures",
        # Key lab values that appear verbally in ICU notes
        "lactate", "wbc", "hgb", "hct", "sodium", "potassium",
        "hco", "angap", "bun",
        # General severity / failure language
        "failure", "acute", "pneumonia", "ards",
        "pulmonary", "dyspnea", "edema",
    })

    # ── Legacy stop-word block (kept for reference, no longer active) ─────
    # _ADMIN_STOP was used with data-driven TF-IDF to filter noise.
    # With vocabulary=_SOFA_VOCABULARY, sklearn ignores stop_words entirely.
    _ADMIN_STOP = {
        # MIMIC-III de-identification artifact words
        "hospital", "hospital1", "hospital2", "hospital3", "hospital4",
        "name", "lastname", "firstname", "namepattern1", "namepattern2",
        "name3", "name4", "name8", "stitle",
        # Administrative / chart-navigation
        "admission", "admitted", "discharge", "discharged",
        "transfer", "transferred", "rehab",
        "date", "birth", "age", "sex", "gender",
        "floor", "room", "bed", "unit", "service", "team",
        "note", "report", "documentation", "recorded", "having", "known",
        "shift", "morning", "night", "overnight", "evening", "daily", "weekly",
        # People / social
        "patient", "family", "daughter", "son", "wife", "husband",
        "mother", "father", "brother", "sister", "friend",
        # Generic temporal
        "ago", "prior", "previous", "recent", "currently", "initial",
        "following", "subsequent", "overall", "approximately",
        # Chart-completion filler
        "noted", "ordered", "placed", "obtained", "performed", "given",
        "called", "states", "denies", "reports",
        "cont", "continue", "continued", "continues",
        "action", "response", "evaluation",
        # Generic qualifiers
        "good", "poor", "old", "new", "large", "small",
        "right", "left", "upper", "lower",
        "including", "total", "multiple", "also", "however",
        "since", "although", "within", "without", "due", "via",
        # Common chart narrative words
        "plan", "assessment", "case", "course", "brief", "general",
        "male", "man", "woman", "female", "year", "years",
        "day", "days", "week", "weeks", "month", "months",
        "history", "past", "present", "social", "complaint",
        "allergies", "medications", "meds", "labs", "results",
        "exam", "examination", "physical", "pertinent",
        "changed", "changes", "improved", "improving",
        "high", "low", "normal", "abnormal", "positive", "negative",
        # Generic ability/state words (not clinically discriminative)
        "able", "unable", "active", "activity", "adequate", "appropriate",
        "awake", "aware", "benign", "comfortable", "complicated",
        "concern", "condition", "consistent", "controlled",
        "current", "decreased", "deep", "did", "does",
        "equal", "free", "frequent", "general", "goal",
        "improved", "improvement", "intact", "later", "likely",
        "minimal", "moderate", "minimal", "minor", "multiple",
        "non", "notable", "occasional", "open", "oral",
        "outside", "possible", "present", "prior", "quit",
        "recent", "recently", "resolved", "rest", "showed",
        "significant", "slightly", "stopped", "subsequently",
        "today", "tolerated", "tolerating", "unable", "unchanged",
        "unknown", "use", "went", "white", "work", "yesterday",
        # Anatomical locations too generic to be discriminative
        "air", "area", "arm", "arrival", "base", "baseline", "bases",
        "bilat", "cervical", "chair", "code", "face", "facial",
        "fall", "foot", "groin", "head", "hip", "leg",
        "line", "lobe", "midline", "mouth", "movement", "movements",
        "site", "size", "skin", "sounds", "spine",
        "stage", "wall", "weight",
        # Common clinical narrative verbs (not discriminative)
        "admit", "appears", "began", "care", "cell",
        "checks", "commands", "consult", "consulted",
        "denied", "died", "diet", "drain",
        "evidence", "feeling", "felt", "flow",
        "follow", "followed", "function", "grade",
        "life", "light", "loss", "management",
        "monitor", "monitoring", "pending", "place", "placement",
        "precautions", "presented", "presents", "received",
        "remained", "remains", "removal", "removed",
        "reported", "required", "requiring", "revealed",
        "review", "seen", "sensation", "sent", "setting",
        "signs", "sleep", "soft", "speech",
        "started", "status", "strength", "study",
        "supple", "support", "symptoms", "syndrome",
        "systems", "taken", "taking", "therapy", "thought",
        "time", "times", "tobacco", "treated", "treatment",
        "tube", "type", "underwent", "units", "valuables",
        "vessel", "vital", "vital signs", "vitals",
        # Administrative chart terms
        "attending", "attending chief", "drugs attending",
        "procedure illness", "review systems", "sig tablet",
        "tablet sig", "warm perfused",
        # Non-clinical / personal items that slip through text cleaning
        "wallet", "money", "job", "wires", "afternoon", "morning",
        "evening", "contact", "valuables", "belongings",
    }
    _all_stop_words = list(ENGLISH_STOP_WORDS) + list(_ADMIN_STOP)
    # _all_stop_words is no longer passed to TfidfVectorizer (vocabulary= mode
    # ignores stop_words). Kept for potential future use.

    tfidf_vec = TfidfVectorizer(
        vocabulary=list(_SOFA_VOCABULARY),                 # SOFA-specific whitelist only
        ngram_range=TFIDF_NGRAM_RANGE,                     # (1,2) — unigrams + bigrams
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]{2,}\b',   # alphabetic only, min 3 chars
        sublinear_tf=True,    # log(1+tf) — grades severity without extreme values
        norm="l2",            # L2-normalise — bounds each document to unit sphere
    )
    # Note: max_features / min_df / max_df / stop_words are ignored by sklearn
    # when vocabulary= is supplied. All SOFA-vocabulary terms are always kept.
    tfidf_matrix = tfidf_vec.fit_transform(notes_combined["text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_vec.get_feature_names_out()
    )
    tfidf_df["hadm_id"] = notes_combined["hadm_id"].values
    actual_vocab = tfidf_vec.get_feature_names_out()
    print(f"  TF-IDF vocabulary: {len(_SOFA_VOCABULARY)} SOFA-specific terms (whitelist)")
    print(f"  TF-IDF features selected: {len(actual_vocab)}")
    print(f"  TF-IDF matrix shape: {tfidf_df.shape}")
    print(f"  Sample features: {list(actual_vocab[:15])}")

    # ── 0e. Merge vitals + TF-IDF, clean ─────────────────────
    print("\n[0e] Merging datasets and cleaning...")
    dataset = ml_data.merge(tfidf_df, on="hadm_id", how="left")
    print(f"  Merged shape: {dataset.shape}")

    drop_cols = [c for c in ["subject_id", "hadm_id", "icustay_id", "window_time", "rn"]
                 if c in dataset.columns]
    dataset = dataset.drop(columns=drop_cols)
    dataset = dataset.fillna(0)
    print(f"  After drop+fillna: {dataset.shape}")

    # ── 0f. Train / test split ────────────────────────────────
    print("\n[0f] Train/test split (80/20, stratified by High-Risk)...")
    y = dataset["sofa_score"].values.astype(np.float32)
    X = dataset.drop(columns=["sofa_score"])

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=(y >= 10).astype(int)
    )
    print(f"  Train: {X_train_df.shape}  |  Test: {X_test_df.shape}")

    # ── 0g. StandardScaler on X_train ────────────────────────
    print("\n[0g] Fitting StandardScaler on X_train...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Clip to [-10, 10] — wider than the previous (-5,5) to preserve gradient
    # signal for high/low SOFA extremes (critical patients).
    # Clip(-5,5) caused oscillating server loss because it removed information
    # for genuinely extreme vitals (SpO2_min, HR_std for critical patients).
    # (-10,10) keeps useful outlier signal while still preventing catastrophic
    # predictions from any remaining rare TF-IDF features.
    X_train_scaled = np.clip(scaler.fit_transform(X_train_df), -10, 10)
    X_test_scaled  = np.clip(scaler.transform(X_test_df),      -10, 10)

    X_train_df = pd.DataFrame(X_train_scaled,
                               index=X_train_df.index,
                               columns=X.columns)
    X_test_np  = X_test_scaled.astype(np.float32)

    # ── 0h. Split into 3 hospital client datasets (IID) ──────
    print("\n[0h] Splitting training data into 3 hospital client CSVs (IID)...")
    from sklearn.utils import shuffle as sk_shuffle

    data_train = X_train_df.copy()
    data_train["sofa_score"] = y_train
    data_train = sk_shuffle(data_train, random_state=42)

    n = len(data_train)
    splits = [
        data_train.iloc[:int(n * 0.33)],
        data_train.iloc[int(n * 0.33):int(n * 0.66)],
        data_train.iloc[int(n * 0.66):],
    ]

    os.makedirs(DATA_PATH, exist_ok=True)
    for i, split in enumerate(splits):
        y_h = split["sofa_score"]
        X_h = split.drop(columns=["sofa_score"])
        save_df = X_h.copy()
        save_df["sofa_score"] = y_h.values
        save_df.to_csv(_client_paths[i], index=False)
        n_low  = (y_h < 5).sum()
        n_mod  = ((y_h >= 5) & (y_h < 10)).sum()
        n_high = (y_h >= 10).sum()
        print(f"  Hospital {i} ({hospital_names[i]}): {len(y_h):,} samples — "
              f"Low:{n_low}({n_low/len(y_h)*100:.0f}%) | "
              f"Mod:{n_mod}({n_mod/len(y_h)*100:.0f}%) | "
              f"High:{n_high}({n_high/len(y_h)*100:.0f}%)")

    # ── 0i. Save sklearn artifacts ────────────────────────────
    print("\n[0i] Saving sklearn artifacts...")
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(scaler,    MODEL_PATH + "scaler.pkl")
    joblib.dump(tfidf_vec, MODEL_PATH + "tfidf_vectorizer.pkl")
    joblib.dump(X_train_df.columns.tolist(), MODEL_PATH + "feature_columns.pkl")
    print("  Saved: scaler.pkl, tfidf_vectorizer.pkl, feature_columns.pkl")
    print("  Saved: client_0.csv, client_1.csv, client_2.csv")

    return X_test_np, y_test


# =============================================================
# ── PHASE 1–6: FL TRAINING ───────────────────────────────────
# =============================================================
print("=" * 60)
print("  ICU Federated Learning — Training Script  (Unified)")
print("=" * 60)

# ── Determine whether preprocessing is needed ────────────────
skip = SKIP_PREPROCESSING or _csvs_exist
if not skip:
    X_test_np, y_test_global = run_preprocessing()
    _csvs_exist = True   # they now exist
    skip = True

print("\n[1/6] Loading hospital datasets from client CSVs...")

scaler_obj     = joblib.load(MODEL_PATH + "scaler.pkl")
feature_columns = list(scaler_obj.feature_names_in_)

clients_data = []
all_X, all_y = [], []

for i in range(3):
    df = pd.read_csv(_client_paths[i])
    y  = df["sofa_score"].values.astype(np.float32)
    X  = df.drop(columns=["sofa_score"])

    # Align columns to scaler's expected order
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_columns].values.astype(np.float32)

    n_low  = (y < 5).sum()
    n_mod  = ((y >= 5) & (y < 10)).sum()
    n_high = (y >= 10).sum()
    print(f"  Hospital {i} ({hospital_names[i]}): {len(y):,} samples — "
          f"Low:{n_low}({n_low/len(y)*100:.0f}%) | "
          f"Mod:{n_mod}({n_mod/len(y)*100:.0f}%) | "
          f"High:{n_high}({n_high/len(y)*100:.0f}%)")

    clients_data.append((X, y))
    all_X.append(X)
    all_y.append(y)

X_all = np.vstack(all_X)
y_all = np.concatenate(all_y)
input_dim = X_all.shape[1]

print(f"\n  Combined: {X_all.shape[0]:,} samples, {X_all.shape[1]} features")

# Build held-out test set:
# - If preprocessing just ran: use the real held-out X_test from Phase 0
# - If skipped: reconstruct a 20% split from the combined client data
try:
    X_test_np    # set by run_preprocessing() if Phase 0 ran
    y_test_np = y_test_global
    print(f"  Using held-out test set from Phase 0: {X_test_np.shape}")
except NameError:
    # Preprocessing was skipped — reconstruct test set from client data
    X_train_pool, X_test_np, y_train_pool, y_test_np = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42,
        stratify=(y_all >= 10).astype(int)
    )
    print(f"  Reconstructed test set: {X_test_np.shape}")

# =============================================================
# NON-IID SPLIT  (only if USE_NONIID_SPLIT = True)
# =============================================================
def make_noniid_split(X_pool, y_pool, seed=42):
    """
    Specialty-biased hospital split (General / Mixed / Cardiac-Trauma).
    Tested: R²≈0.08 vs IID R²≈0.28. Documented in MODEL_DISCUSSION.md.
    """
    rng = np.random.default_rng(seed)
    idx_low  = np.where(y_pool < 5)[0]
    idx_mod  = np.where((y_pool >= 5) & (y_pool < 10))[0]
    idx_high = np.where(y_pool >= 10)[0]
    rng.shuffle(idx_low); rng.shuffle(idx_mod); rng.shuffle(idx_high)

    fracs_low  = np.array([0.70, 0.20, 0.10])
    fracs_mod  = np.array([0.20, 0.60, 0.20])
    fracs_high = np.array([0.10, 0.20, 0.70])

    def _split(indices, fracs):
        n = len(indices)
        cuts = np.clip(np.round(np.cumsum(fracs) * n).astype(int), 0, n)
        parts, prev = [], 0
        for cut in cuts:
            parts.append(indices[prev:cut]); prev = cut
        return parts

    low_p  = _split(idx_low,  fracs_low)
    mod_p  = _split(idx_mod,  fracs_mod)
    high_p = _split(idx_high, fracs_high)

    splits = []
    for i in range(3):
        idx = np.concatenate([low_p[i], mod_p[i], high_p[i]])
        rng.shuffle(idx)
        splits.append((X_pool[idx], y_pool[idx]))
    return splits

if USE_NONIID_SPLIT:
    print("\n  Split type: Non-IID (hospital specialty bias)")
    print("  ⚠ Non-IID reduces global R² to ~0.08 — use for research/demo only.")
    try:
        X_train_pool
    except NameError:
        X_train_pool = X_all
        y_train_pool = y_all
    noniid_splits = make_noniid_split(X_train_pool, y_train_pool)
    clients_data = []
    for i, (X_h, y_h) in enumerate(noniid_splits):
        n_low  = (y_h < 5).sum()
        n_mod  = ((y_h >= 5) & (y_h < 10)).sum()
        n_high = (y_h >= 10).sum()
        print(f"  Hospital {i} ({hospital_names[i]}): {len(y_h):,} — "
              f"Low:{n_low}({n_low/len(y_h)*100:.0f}%) | "
              f"Mod:{n_mod}({n_mod/len(y_h)*100:.0f}%) | "
              f"High:{n_high}({n_high/len(y_h)*100:.0f}%)")
        clients_data.append((X_h, y_h))
else:
    print("\n  Split type: IID (client_0/1/2.csv)")

# =============================================================
# [2/6] SHAP BACKGROUND
# =============================================================
print("\n[2/6] Saving SHAP background samples...")
try:
    X_bg_pool = X_train_pool
except NameError:
    X_bg_pool = X_all
rng = np.random.default_rng(42)
bg_idx    = rng.choice(len(X_bg_pool), SHAP_BG_SAMPLES, replace=False)
background = X_bg_pool[bg_idx]
np.save(MODEL_PATH + "shap_background.npy", background)
print(f"  Saved {SHAP_BG_SAMPLES} background samples → models/shap_background.npy")

# =============================================================
# [3/6] FLOWER CLIENT  (weighted MSE + AdamW + LR scheduling + optional DP)
# =============================================================
def get_model():
    return ICUModel(input_dim)

class HospitalClient(fl.client.NumPyClient):

    def __init__(self, X, y, hospital_id, name):
        self.hospital_id = hospital_id
        self.name        = name
        self.model       = get_model()

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        self.X_train, self.X_val = X_tr, X_val
        self.y_train, self.y_val = y_tr, y_val

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        global_weights = [w.copy() for w in parameters]
        set_weights(self.model, parameters)

        server_round = config.get("server_round", 1)
        lr = max(1e-4, BASE_LR * (LR_DECAY ** (server_round - 1)))

        loss = train_model(
            self.model, self.X_train, self.y_train,
            epochs=EPOCHS_PER_ROUND, lr=lr,
            batch_size=BATCH_SIZE, grad_clip=GRAD_CLIP,
            oversample=OVERSAMPLE,
            global_params=parameters,   # FedProx: penalise drift from global model
            mu=MU_FEDPROX,
        )
        metrics = evaluate_model(self.model, self.X_val, self.y_val)

        dp_tag = ""
        if USE_DP:
            local_weights = get_weights(self.model)
            noisy_weights = apply_dp_to_update(
                local_weights, global_weights, DP_SENSITIVITY, DP_SIGMA
            )
            dp_tag = " | DP ✓"
            weights_to_send = noisy_weights
        else:
            weights_to_send = get_weights(self.model)

        print(
            f"    Hospital {self.hospital_id} ({self.name}) "
            f"| lr={lr:.5f} | loss={loss:.3f} "
            f"| MAE={metrics['mae']:.3f} | R²={metrics['r2']:.3f}{dp_tag}"
        )
        return weights_to_send, len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        metrics = evaluate_model(self.model, self.X_val, self.y_val)
        return metrics["mse"], len(self.X_val), {
            "mae": metrics["mae"],
            "r2":  metrics["r2"]
        }

hospital_clients = [
    HospitalClient(X, y, i, hospital_names[i])
    for i, (X, y) in enumerate(clients_data)
]

def client_fn(context: Context):
    cid = int(context.node_config["partition-id"])
    return hospital_clients[cid].to_client()

# =============================================================
# FLOWER SERVER STRATEGY  (SaveBestStrategy + FedAdam)
# =============================================================
_best_loss    = float("inf")
_best_round   = 0
_best_weights = {}
_last_weights = {}

# Initial parameters: random model weights for FedYogi bootstrap.
# FedPreTrain (centralized warm-start) was tested but hurt performance:
# it trapped the model near the centralized local minimum, removing the
# cold-start exploration that allows FedYogi to find deeper global minima.
_init_model     = get_model()
_initial_params = ndarrays_to_parameters(get_weights(_init_model))

class SaveBestStrategy(fl.server.strategy.FedYogi):

    def configure_fit(self, server_round, parameters, client_manager):
        config  = {"server_round": server_round}
        fit_ins = FitIns(parameters, config)
        sample_size, min_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_clients
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            _last_weights["params"] = aggregated[0]
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        global _best_loss, _best_round
        if results:
            avg_loss = sum(r.loss for _, r in results) / len(results)
            tag = ""
            if avg_loss < _best_loss:
                _best_loss  = avg_loss
                _best_round = server_round
                _best_weights["params"] = _last_weights.get("params")
                tag = "  ← best ✓"
            print(f"  [Server] Round {server_round:02d} — "
                  f"lr={BASE_LR * (LR_DECAY**(server_round-1)):.5f} | "
                  f"avg_loss={avg_loss:.4f}{tag}")
        return super().aggregate_evaluate(server_round, results, failures)

strategy = SaveBestStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    initial_parameters=_initial_params,  # required by FedAdam
    eta=SERVER_ETA,          # server-side Adam learning rate
    eta_l=BASE_LR,           # client-side LR (informational — we compute it ourselves)
    beta_1=SERVER_BETA1,
    beta_2=SERVER_BETA2,
    tau=SERVER_TAU,
)

# =============================================================
# RUN FL SIMULATION
# =============================================================
print(f"\n[3/6] Starting FL simulation...")
print(f"  Rounds             : {NUM_ROUNDS}")
print(f"  Epochs/round       : {EPOCHS_PER_ROUND}")
print(f"  Batch size         : {BATCH_SIZE}")
print(f"  Base LR            : {BASE_LR}  (decays ×{LR_DECAY} per round)")
print(f"  Gradient clip      : {GRAD_CLIP}")
print(f"  FedProx μ          : {MU_FEDPROX}  (client-side proximal term)")
print(f"  Server optimizer   : FedYogi  (η={SERVER_ETA}, β1={SERVER_BETA1}, β2={SERVER_BETA2}, τ={SERVER_TAU})")
print(f"  Oversampling       : {'Enabled' if OVERSAMPLE else 'Disabled'}")
print(f"  Loss weights       : Linear (1 + SOFA×0.5)  →  High~2.0×, Severe~3.7×")
print(f"  Split type         : {'Non-IID' if USE_NONIID_SPLIT else 'IID'}")
if USE_DP:
    eps = estimate_privacy_budget(NUM_ROUNDS, DP_SIGMA)
    print(f"  Differential Privacy: ENABLED  (σ={DP_SIGMA}, S={DP_SENSITIVITY}, ε≈{eps}, δ=1e-5)")
else:
    print(f"  Differential Privacy: disabled")
print()

warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging as _log; _log.getLogger("flwr").setLevel(_log.ERROR)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 1}
)

_log.getLogger("flwr").setLevel(_log.WARNING)

# =============================================================
# [4/6] SAVE BEST MODEL
# =============================================================
print(f"\n[4/6] Saving model (best was round {_best_round})...")

saved_params = _best_weights.get("params") or _last_weights.get("params")
if saved_params is None:
    raise RuntimeError("Training produced no weights. Check errors above.")

final_model = get_model()
set_weights(final_model, parameters_to_ndarrays(saved_params))
torch.save(final_model.state_dict(), MODEL_PATH + "federated_model.pth")
print(f"  Saved → models/federated_model.pth  (from round {_best_round})")

# =============================================================
# [5/6] FINAL EVALUATION
# =============================================================
print("\n[5/6] Evaluating on held-out test set...")

final_model.eval()
with torch.no_grad():
    preds = final_model(
        torch.tensor(X_test_np, dtype=torch.float32)
    ).numpy().flatten()

mae = mean_absolute_error(y_test_np, preds)
r2  = r2_score(y_test_np, preds)

for label, lo, hi in [("Low (<5)", 0, 5), ("Moderate (5-9)", 5, 10), ("High (≥10)", 10, 25)]:
    mask = (y_test_np >= lo) & (y_test_np < hi)
    if mask.sum() > 0:
        sub_mae = mean_absolute_error(y_test_np[mask], preds[mask])
        sub_r2  = r2_score(y_test_np[mask], preds[mask])
        print(f"  {label:16s}: MAE={sub_mae:.3f}  R²={sub_r2:.3f}  (n={mask.sum():,})")

print()
print("=" * 60)
print("  FINAL MODEL PERFORMANCE  (global test set)")
print("=" * 60)
print(f"  MAE  : {mae:.4f} SOFA points")
print(f"  R²   : {r2:.4f}")
print(f"  Pred range : {preds.min():.2f} – {preds.max():.2f}")
print("=" * 60)

# =============================================================
# [6/6] SAVE TRAINING METADATA
# =============================================================
print("\n[6/6] Saving training metadata...")
metadata = {
    "num_rounds":         NUM_ROUNDS,
    "epochs_per_round":   EPOCHS_PER_ROUND,
    "batch_size":         BATCH_SIZE,
    "base_lr":            BASE_LR,
    "lr_decay":           LR_DECAY,
    "hospitals":          3,
    "hospital_names":     hospital_names,
    "aggregation":        f"FedYogi (η={SERVER_ETA}, β1={SERVER_BETA1}, β2={SERVER_BETA2}) + FedProx (μ={MU_FEDPROX})",
    "split_type":         "Non-IID (specialty bias)" if USE_NONIID_SPLIT else "IID",
    "train_samples":      int(X_all.shape[0]),
    "test_samples":       int(len(y_test_np)),
    "input_features":     int(input_dim),
    "model_architecture": f"{input_dim} → 128 → 64 → 32 → 1  (ReLU, no Dropout, FedProx)",
    "best_round":         int(_best_round),
    "final_mae":          round(float(mae), 4),
    "final_r2":           round(float(r2), 4),
    "pred_range_min":     round(float(preds.min()), 2),
    "pred_range_max":     round(float(preds.max()), 2),
    "oversample":           OVERSAMPLE,
    "loss_weights":         "Linear (1 + SOFA×0.5) — mild high-SOFA emphasis (~2× at SOFA=10)",
    "optimizer":            "AdamW (weight_decay=1e-4)",
    "differential_privacy": USE_DP,
    "dp_sensitivity":     DP_SENSITIVITY if USE_DP else None,
    "dp_sigma":           DP_SIGMA        if USE_DP else None,
    "dp_epsilon":         estimate_privacy_budget(NUM_ROUNDS, DP_SIGMA) if USE_DP else None,
    "dp_delta":           1e-5            if USE_DP else None,
    # Preprocessing metadata
    "bigquery_project":   BIGQUERY_PROJECT,
    "tfidf_features":     TFIDF_MAX_FEATURES,
    "tfidf_ngram":        list(TFIDF_NGRAM_RANGE),
    "notes_sample_size":  NOTES_SAMPLE_SIZE,
}

with open(MODEL_PATH + "training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  Metadata saved → models/training_metadata.json")
print("\nDone. Restart the app to load the new model:")
print("  streamlit run app.py")

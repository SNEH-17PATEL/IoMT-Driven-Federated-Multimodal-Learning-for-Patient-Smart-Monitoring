"""
Builds the replayed demo stream for the monitor.

Each patient has 30 readings, 10 minutes apart. For every reading the stream holds
vitals + camera scores (GCS eye, stress) + a clinical note. This script:

  * rewrites the clinical notes of patients 1-3 so the note EVOLVES with the vitals
    (timestamped nursing observations) instead of repeating one fixed paragraph.
    The numeric vitals of patients 1-3 are never touched.
  * writes patient 4, a scripted "gradual deterioration" scenario (stable -> unwell ->
    septic) so escalation can be demonstrated across all three modalities.

Run from the project root, then refresh the database:
    python backend/make_demo_data.py
    python migrate_to_db.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP_DIR = os.path.join(ROOT, "data", "sample_patients")
KEEP_LAST = 3          # a note shows the base history + the 3 most recent observations

# ── Evolving observations: reading index -> observation text ────────────────────
# Wording deliberately uses terms the model's text vocabulary knows (e.g. "fever",
# "hypotension", "vasopressor", "septic shock") only where they are clinically true
# for that moment, so the text modality moves the prediction for the right reasons.
BASE = {
    1: "62-year-old male, post-appendectomy day 2.",
    2: "58-year-old female with community-acquired pneumonia on supplemental oxygen 4L/min.",
    3: ("72-year-old male with septic shock on norepinephrine vasopressor support. "
        "Intubated and mechanically ventilated at FiO2 0.8. "
        "History of COPD and chronic kidney disease."),
    4: ("68-year-old female with a recent urinary tract infection on oral antibiotics. "
        "History of type 2 diabetes."),
}

OBS = {
    1: {
        0:  "Recovery progressing well. Alert and oriented. Tolerating oral fluids. "
            "Mild surgical site pain managed with oral analgesics. No fever. No respiratory distress.",
        5:  "Sat up in the chair with assistance. Pain 3/10, no extra analgesia needed. Lungs clear.",
        10: "Walked the corridor with physiotherapy. Tolerating a light diet. Breathing comfortably.",
        15: "Wound clean and dry, no redness. Appetite improving. Mood good.",
        20: "Walking independently. Voiding without difficulty. Ready for discharge planning review.",
        25: "Vitals steady and improving. Discharge teaching completed with family.",
    },
    2: {
        0:  "Creatinine mildly elevated at 1.4 mg/dL. WBC 14.2. On IV ceftriaxone and azithromycin. "
            "Moderate respiratory distress. Monitoring for response to antibiotics.",
        4:  "Productive cough, speaking in short sentences. Fever persists.",
        8:  "Antibiotics continuing. Oxygen saturation holding at 92 to 93 percent on 4L/min. "
            "Less breathless than earlier.",
        12: "Fever settling. Cough looser, eating small meals.",
        16: "Oxygen weaned to 3L/min with saturation steady. WBC trending down.",
        20: "Temperature settling toward normal. Sitting out of bed, appetite returning.",
        25: "Creatinine back to 1.1 mg/dL. Plan to step oxygen down to nasal cannula 2L/min.",
    },
    3: {
        0:  "SpO2 critically low despite ventilator. Creatinine critically elevated at 3.2 mg/dL "
            "with oliguria less than 0.3 mL/kg/h. Vasopressor requirement increasing. "
            "Broad-spectrum antibiotics initiated. Blood cultures pending. "
            "Hemodynamic instability worsening.",
        3:  "Norepinephrine increased, second vasopressor being considered. Sedated, not following commands.",
        6:  "Lactate remains elevated. Fluid bolus given with minimal response in blood pressure.",
        9:  "Ventilator settings adjusted for low oxygenation. Urine output remains very low.",
        12: "Slight improvement after fluid and sedation adjustment. Remains on vasopressor support.",
        15: "Sudden drop in blood pressure and oxygen saturation. Vasopressor dose raised, ventilator alarms.",
        18: "Partially recovered after vasopressor titration. Still hemodynamically unstable.",
        21: "Renal function worsening, dialysis under consideration. Broad-spectrum antibiotics continuing.",
        24: "Marginal stabilization. Sedated, pupils sluggish. Family updated on critical condition.",
        27: "Another episode of hypotension and desaturation. Vasopressor requirement increasing again.",
    },
    4: {
        0:  "Resting comfortably, alert and oriented. Eating and drinking normally. No complaints.",
        4:  "Feels a little tired but otherwise well. Taking oral antibiotics as prescribed.",
        8:  "Reports feeling warm and chilled. Appetite reduced. Mild fever developing.",
        11: "Fever persisting. Drinking less and passing very little urine today.",
        14: "Feeling weak and lightheaded when sitting up. Heart racing. Breathing faster than usual.",
        17: "Increasing weakness and dizziness. Skin flushed and warm. Infection progressing, "
            "concern for sepsis. Lactate elevated.",
        20: "Hypotension with low blood pressure readings. Drowsy and confused. "
            "Shortness of breath at rest. Septic shock developing, lactate rising.",
        23: "Altered mental status, difficult to rouse. Acute kidney injury with oliguria, "
            "creatinine rising. Respiratory distress with hypoxia. Septic shock, vasopressor "
            "required. Bilirubin rising.",
        26: "Unresponsive to voice. Septic shock with refractory hypotension on norepinephrine "
            "vasopressor. Respiratory failure, intubation needed. Renal failure. Urgent hospital "
            "transfer required. Multi-organ failure with liver failure and coagulopathy.",
    },
}


def note_for(patient: int, idx: int, clock: str) -> str:
    """Base history + the most recent observations up to reading `idx`."""
    seen = [(i, t) for i, t in sorted(OBS[patient].items()) if i <= idx][-KEEP_LAST:]
    # timestamp each observation with the clock time of the reading it was made at
    parts = [BASE[patient]]
    for i, t in seen:
        parts.append(f"[{_clock_at(i, clock)}] {t}")
    return " ".join(parts)


def _clock_at(i: int, start_clock: str) -> str:
    h, m = map(int, start_clock.split(":"))
    tot = h * 60 + m + 10 * i
    return f"{tot // 60:02d}:{tot % 60:02d}"


# idx, HR, RR, SpO2, Temp, SBP, DBP
P4_KEYFRAMES = [
    (0,  84, 16, 97.0, 37.3, 124, 78),
    (7,  86, 17, 96.5, 37.5, 122, 77),
    (11, 100, 21, 94.5, 38.3, 110, 68),
    (15, 118, 26, 91.5, 38.9, 96, 58),
    (19, 134, 31, 87.0, 39.3, 82, 48),
    (23, 146, 36, 81.0, 39.7, 72, 40),
    (26, 154, 40, 76.0, 39.9, 66, 35),
    (29, 160, 44, 72.0, 40.1, 60, 30),
]


def patient4_vitals(n: int = 30) -> pd.DataFrame:
    """Scripted deterioration: stable -> early infection -> sepsis -> shock."""
    rng = np.random.default_rng(4)
    key = P4_KEYFRAMES
    xs = [k[0] for k in key]
    cols = list(zip(*[k[1:] for k in key]))
    rows = []
    for i in range(n):
        hr, rr, sp, tp, sb, db = (float(np.interp(i, xs, c)) for c in cols)
        hr += rng.normal(0, 1.5); rr += rng.normal(0, 0.6); sp += rng.normal(0, 0.3)
        tp += rng.normal(0, 0.05); sb += rng.normal(0, 1.5); db += rng.normal(0, 1.0)
        sbp, dbp = int(round(sb)), int(round(db))
        gcs = 4 if i < 17 else 3 if i < 22 else 2 if i < 26 else 1
        stress = 2 if i < 8 else min(10, 2 + (i - 8) // 2 + (1 if i > 22 else 0))
        rows.append({
            "HR": int(round(hr)), "RR": int(round(rr)), "SpO2": round(sp * 2) / 2,
            "Temp": round(tp, 1), "SBP": sbp, "DBP": dbp,
            "MAP": int(round((sbp + 2 * dbp) / 3)),
            "GCS_eye_opening": gcs, "stress_score": stress,
        })
    return pd.DataFrame(rows)


def main() -> None:
    for pid in (1, 2, 3):
        path = os.path.join(SP_DIR, f"sample_patient0{pid}_data.csv")
        df = pd.read_csv(path)
        start = df["timestamp"].iloc[0][11:16]
        df["clinical_note"] = [note_for(pid, i, start) for i in range(len(df))]
        df.to_csv(path, index=False)
        print(f"patient {pid}: {len(df)} rows, {df['clinical_note'].nunique()} distinct notes")

    df4 = patient4_vitals()
    ref = pd.read_csv(os.path.join(SP_DIR, "sample_patient01_data.csv"))
    df4.insert(0, "timestamp", ref["timestamp"].values[: len(df4)])
    df4["clinical_note"] = [note_for(4, i, "08:00") for i in range(len(df4))]
    df4.to_csv(os.path.join(SP_DIR, "sample_patient04_data.csv"), index=False)
    print(f"patient 4: {len(df4)} rows, {df4['clinical_note'].nunique()} distinct notes")


if __name__ == "__main__":
    main()

"""
ICU CDSS — SQLite Database Layer
=================================
Single source of truth for all operational data.

Tables
------
patient_vitals      — sliding-window vitals per patient (replaces patient_vitals_01/02/03.csv)
prediction_history  — SOFA prediction log per patient  (replaces prediction_history_01/02/03.csv)
sample_patients     — simulation rows per patient      (replaces sample_patient01/02/03_data.csv)
fl_training         — pre-scaled FL client datasets    (replaces client_0/1/2.csv)

Database file: models/icu_monitor.db
"""

import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "models", "icu_monitor.db")

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    """Return a new SQLite connection with WAL mode for better concurrency."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they do not already exist."""
    conn = get_conn()
    cur  = conn.cursor()

    # ── Patient vitals sliding window ─────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_vitals (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            patient INTEGER NOT NULL,
            time    TEXT    NOT NULL,
            HR      REAL,
            RR      REAL,
            SpO2    REAL,
            Temp    REAL,
            SBP     REAL,
            DBP     REAL,
            MAP     REAL
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_pv_patient ON patient_vitals(patient)"
    )

    # ── Prediction history ────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient     INTEGER NOT NULL,
            Timestamp   TEXT,
            Reading     TEXT,
            SOFA        REAL,
            Risk        TEXT,
            HR          REAL,
            RR          REAL,
            SpO2        REAL,
            Temp        REAL,
            SBP         REAL,
            MAP         REAL,
            GCS_Eye     INTEGER,
            Stress      REAL
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ph_patient ON prediction_history(patient)"
    )

    # ── Sample patient simulation data ────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sample_patients (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            patient       INTEGER NOT NULL,
            row_idx       INTEGER NOT NULL,
            timestamp     TEXT,
            HR            REAL,
            RR            REAL,
            SpO2          REAL,
            Temp          REAL,
            SBP           REAL,
            DBP           REAL,
            MAP           REAL,
            GCS_eye_opening  INTEGER,
            stress_score  REAL,
            clinical_note TEXT
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_sp_patient_row "
        "ON sample_patients(patient, row_idx)"
    )

    # fl_training is created dynamically by migrate_to_db.py with the full
    # 109-column schema (pandas to_sql). init_db() does not pre-create it
    # so that the correct column set is always used on first import.

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# patient_vitals helpers
# ---------------------------------------------------------------------------

def get_patient_vitals(patient: int, limit: int = 20) -> pd.DataFrame:
    """Return the last `limit` rows for a patient as a DataFrame."""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT time, HR, RR, SpO2, Temp, SBP, DBP, MAP "
        "FROM patient_vitals "
        "WHERE patient = ? "
        "ORDER BY id DESC LIMIT ?",
        conn, params=(patient, limit)
    )
    conn.close()
    # Return in ascending time order (oldest first, matching original CSV behaviour)
    return df.iloc[::-1].reset_index(drop=True)


def append_patient_vitals(patient: int, row: dict) -> None:
    """Insert one vitals row and keep only the most recent 20 per patient."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO patient_vitals (patient, time, HR, RR, SpO2, Temp, SBP, DBP, MAP) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (patient, row["time"], row["HR"], row["RR"], row["SpO2"],
         row["Temp"], row["SBP"], row["DBP"], row["MAP"])
    )
    # Keep sliding window to 20 rows
    cur.execute(
        "DELETE FROM patient_vitals WHERE patient = ? AND id NOT IN ("
        "  SELECT id FROM patient_vitals WHERE patient = ? "
        "  ORDER BY id DESC LIMIT 20"
        ")",
        (patient, patient)
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# prediction_history helpers
# ---------------------------------------------------------------------------

def get_prediction_history(patient: int, limit: int = 50) -> pd.DataFrame:
    """Return the last `limit` predictions for a patient."""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT Timestamp, Reading, SOFA, Risk, HR, RR, SpO2 as 'SpO₂', "
        "       Temp as 'Temp (°C)', SBP, MAP, GCS_Eye as 'GCS Eye', Stress "
        "FROM prediction_history "
        "WHERE patient = ? "
        "ORDER BY id DESC LIMIT ?",
        conn, params=(patient, limit)
    )
    conn.close()
    return df.iloc[::-1].reset_index(drop=True)


def append_prediction(patient: int, row: dict) -> None:
    """Insert one prediction row and keep only the most recent 50 per patient."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO prediction_history "
        "(patient, Timestamp, Reading, SOFA, Risk, HR, RR, SpO2, Temp, "
        " SBP, MAP, GCS_Eye, Stress) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (patient,
         row.get("Timestamp"), row.get("Reading"),
         row.get("SOFA"),      row.get("Risk"),
         row.get("HR"),        row.get("RR"),
         row.get("SpO₂"),      row.get("Temp (°C)"),
         row.get("SBP"),       row.get("MAP"),
         row.get("GCS Eye"),   row.get("Stress"))
    )
    # Keep only last 50 per patient
    cur.execute(
        "DELETE FROM prediction_history WHERE patient = ? AND id NOT IN ("
        "  SELECT id FROM prediction_history WHERE patient = ? "
        "  ORDER BY id DESC LIMIT 50"
        ")",
        (patient, patient)
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# sample_patients helpers
# ---------------------------------------------------------------------------

def get_sample_patient_row(patient: int, row_idx: int) -> pd.Series | None:
    """Return one simulation row by patient + row index."""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT timestamp, HR, RR, SpO2, Temp, SBP, DBP, MAP, "
        "       GCS_eye_opening, stress_score, clinical_note "
        "FROM sample_patients "
        "WHERE patient = ? AND row_idx = ? LIMIT 1",
        conn, params=(patient, row_idx)
    )
    conn.close()
    return df.iloc[0] if len(df) else None


def get_sample_patient_count(patient: int) -> int:
    """Return total number of simulation rows for a patient."""
    conn = get_conn()
    cur  = conn.cursor()
    n = cur.execute(
        "SELECT COUNT(*) FROM sample_patients WHERE patient = ?", (patient,)
    ).fetchone()[0]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# fl_training helpers
# ---------------------------------------------------------------------------

def get_fl_client_data(client_id: int) -> pd.DataFrame:
    """Return all rows for a given FL client as a DataFrame."""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM fl_training WHERE client_id = ?",
        conn, params=(client_id,)
    )
    conn.close()
    # Drop DB-only columns that the ML code doesn't expect
    for col in ["id", "client_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    # Drop rows with missing labels — these indicate a schema-mismatch import
    if "sofa_score" in df.columns:
        before = len(df)
        df = df.dropna(subset=["sofa_score"])
        dropped = before - len(df)
        if dropped > 0:
            import warnings
            warnings.warn(
                f"fl_training client {client_id}: dropped {dropped} rows with NULL "
                "sofa_score — re-run migrate_to_db.py to fix the database.",
                stacklevel=2,
            )
    # Fill any remaining NaN in feature columns with 0
    df = df.fillna(0)
    return df


def fl_training_exists() -> bool:
    """Return True if fl_training table has any rows."""
    conn = get_conn()
    cur  = conn.cursor()
    n = cur.execute("SELECT COUNT(*) FROM fl_training").fetchone()[0]
    conn.close()
    return n > 0

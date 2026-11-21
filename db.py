"""
ICU CDSS — SQLite Database Layer
=================================
Single source of truth for all operational data.

Tables
------
patient_vitals      — sliding-window vitals per patient
prediction_history  — SOFA prediction log per patient
sample_patients     — simulation rows per patient
fl_training         — pre-scaled FL client datasets (created by migrate_to_db.py)

Database file: models/icu_monitor.db
"""
from __future__ import annotations

import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "models", "icu_monitor.db")

# Sliding-window sizes — single source of truth; backend/api limits must match.
_VITALS_WINDOW  = 20   # rows kept per patient in patient_vitals
_HISTORY_WINDOW = 50   # rows kept per patient in prediction_history

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    """Return a new per-call SQLite connection with WAL journal mode."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables and indexes if they do not already exist. Safe to re-run."""
    conn = get_conn()
    cur  = conn.cursor()

    # ── Patient vitals sliding window ─────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS patient_vitals (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            patient    INTEGER NOT NULL,
            time       TEXT    NOT NULL,
            HR         REAL,
            RR         REAL,
            SpO2       REAL,
            Temp       REAL,
            SBP        REAL,
            DBP        REAL,
            MAP        REAL,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    # Composite index serves both WHERE patient=? and ORDER BY id DESC
    cur.execute("DROP INDEX IF EXISTS idx_pv_patient")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_pv_patient ON patient_vitals(patient, id)"
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
            DBP         REAL,
            MAP         REAL,
            GCS_Eye     INTEGER,
            Stress      REAL,
            created_at  TEXT,
            updated_at  TEXT
        )
    """)
    # Migrate existing DBs that may not have DBP column yet
    try:
        cur.execute("ALTER TABLE prediction_history ADD COLUMN DBP REAL")
    except Exception:
        pass  # column already exists
    cur.execute("DROP INDEX IF EXISTS idx_ph_patient")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ph_patient ON prediction_history(patient, id)"
    )

    # ── Sample patient simulation data ────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sample_patients (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            patient          INTEGER NOT NULL,
            row_idx          INTEGER NOT NULL,
            timestamp        TEXT,
            HR               REAL,
            RR               REAL,
            SpO2             REAL,
            Temp             REAL,
            SBP              REAL,
            DBP              REAL,
            MAP              REAL,
            GCS_eye_opening  INTEGER,
            stress_score     REAL,
            clinical_note    TEXT,
            created_at       TEXT,
            updated_at       TEXT
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_sp_patient_row "
        "ON sample_patients(patient, row_idx)"
    )

    # fl_training is created dynamically by migrate_to_db.py with the full
    # 109-column schema (pandas to_sql). init_db() does not pre-create it
    # so that the correct column set is always used on first import.

    # ── Triggers: auto-stamp created_at + updated_at on INSERT; ──────────
    # ── auto-update updated_at on UPDATE of data columns only.   ──────────
    # AFTER UPDATE OF <data cols> prevents infinite recursion — the trigger
    # won't re-fire when it writes updated_at itself.
    # trg_ph_updated_at had DBP missing from its column list; drop and recreate it.
    cur.execute("DROP TRIGGER IF EXISTS trg_ph_updated_at")

    cur.executescript("""
        CREATE TRIGGER IF NOT EXISTS trg_pv_created_at
        AFTER INSERT ON patient_vitals
        BEGIN
            UPDATE patient_vitals
            SET created_at = datetime('now','localtime'),
                updated_at = datetime('now','localtime')
            WHERE id = NEW.id;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_pv_updated_at
        AFTER UPDATE OF patient, time, HR, RR, SpO2, Temp, SBP, DBP, MAP
        ON patient_vitals
        BEGIN
            UPDATE patient_vitals
            SET updated_at = datetime('now','localtime')
            WHERE id = NEW.id;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_ph_created_at
        AFTER INSERT ON prediction_history
        BEGIN
            UPDATE prediction_history
            SET created_at = datetime('now','localtime'),
                updated_at = datetime('now','localtime')
            WHERE id = NEW.id;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_ph_updated_at
        AFTER UPDATE OF patient, Timestamp, Reading, SOFA, Risk, HR, RR, SpO2, Temp, SBP, DBP, MAP, GCS_Eye, Stress
        ON prediction_history
        BEGIN
            UPDATE prediction_history
            SET updated_at = datetime('now','localtime')
            WHERE id = NEW.id;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_sp_created_at
        AFTER INSERT ON sample_patients
        BEGIN
            UPDATE sample_patients
            SET created_at = datetime('now','localtime'),
                updated_at = datetime('now','localtime')
            WHERE id = NEW.id;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_sp_updated_at
        AFTER UPDATE OF patient, row_idx, timestamp, HR, RR, SpO2, Temp, SBP, DBP, MAP, GCS_eye_opening, stress_score, clinical_note
        ON sample_patients
        BEGIN
            UPDATE sample_patients
            SET updated_at = datetime('now','localtime')
            WHERE id = NEW.id;
        END;

    """)

    # fl_training is created by migrate_to_db.py via pandas to_sql, which does NOT
    # add created_at/updated_at columns. Installing a trigger that references those
    # columns would break every subsequent insert, so no fl_training trigger is created.
    # Drop the stale trigger if it was installed by an older version.
    cur.execute("DROP TRIGGER IF EXISTS trg_ft_created_at")

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
    """Insert one vitals row and keep only the most recent _VITALS_WINDOW per patient."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO patient_vitals (patient, time, HR, RR, SpO2, Temp, SBP, DBP, MAP) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (patient, row["time"], row["HR"], row["RR"], row["SpO2"],
         row["Temp"], row["SBP"], row["DBP"], row["MAP"])
    )
    cur.execute(
        "DELETE FROM patient_vitals WHERE patient = ? AND id NOT IN ("
        "  SELECT id FROM patient_vitals WHERE patient = ? "
        f" ORDER BY id DESC LIMIT {_VITALS_WINDOW}"
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
        "       Temp as 'Temp (°C)', SBP, DBP, MAP, GCS_Eye as 'GCS Eye', Stress "
        "FROM prediction_history "
        "WHERE patient = ? "
        "ORDER BY id DESC LIMIT ?",
        conn, params=(patient, limit)
    )
    conn.close()
    return df.iloc[::-1].reset_index(drop=True)


def append_prediction(patient: int, row: dict) -> None:
    """Insert one prediction row and keep only the most recent _HISTORY_WINDOW per patient."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO prediction_history "
        "(patient, Timestamp, Reading, SOFA, Risk, HR, RR, SpO2, Temp, "
        " SBP, DBP, MAP, GCS_Eye, Stress) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (patient,
         row.get("Timestamp"), row.get("Reading"),
         row.get("SOFA"),      row.get("Risk"),
         row.get("HR"),        row.get("RR"),
         row.get("SpO₂"),      row.get("Temp (°C)"),
         row.get("SBP"),       row.get("DBP"),
         row.get("MAP"),
         row.get("GCS Eye"),   row.get("Stress"))
    )
    cur.execute(
        "DELETE FROM prediction_history WHERE patient = ? AND id NOT IN ("
        "  SELECT id FROM prediction_history WHERE patient = ? "
        f" ORDER BY id DESC LIMIT {_HISTORY_WINDOW}"
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
    # ORDER BY rowid gives a stable physical order so train_test_split(random_state=42)
    # produces the same split every time, regardless of SQLite page layout.
    df = pd.read_sql(
        "SELECT * FROM fl_training WHERE client_id = ? ORDER BY rowid",
        conn, params=(client_id,)
    )
    conn.close()
    # Drop DB-only / metadata columns that the ML code doesn't expect
    for col in ["id", "client_id", "created_at", "updated_at"]:
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
    """Return True if fl_training has the full schema (sofa_score column) and at least one row."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        row = cur.execute("SELECT 1 FROM fl_training LIMIT 1").fetchone()
        if row is None:
            conn.close()
            return False
        cols = [r[1] for r in cur.execute("PRAGMA table_info(fl_training)").fetchall()]
        conn.close()
        return "sofa_score" in cols
    except sqlite3.OperationalError:
        conn.close()
        return False

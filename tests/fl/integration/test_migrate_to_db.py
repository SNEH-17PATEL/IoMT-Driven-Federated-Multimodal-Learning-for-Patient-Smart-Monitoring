"""
Phase 9 — Migration Integration  (test_migrate_to_db.py)
=========================================================
Tests migrate_to_db.migrate() with a temp SQLite DB and stub CSV files.

Strategy:
  • monkeypatch.chdir(tmp_path) makes migrate()'s relative CSV paths
    resolve inside the temp directory, so no real CSV files are touched.
  • Both db.DB_PATH and migrate_to_db.DB_PATH are redirected to a temp
    file so the production icu_monitor.db is never read or written.
  • Stub CSV files are created in the expected sub-directories of tmp_path
    before each call to migrate().

17 tests:
  9A  Happy-path presence checks   —  5 tests
  9B  Idempotency                  —  1 test
  9C  Column renames               —  3 tests
  9D  FL client_id assignment      —  2 tests
  9E  Edge cases (missing / empty) —  2 tests
  9F  Structural checks            —  4 tests
"""
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

# ── sys.path fallback ─────────────────────────────────────────────────────────
_INTEG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_INTEG_DIR, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import db as db_mod              # noqa: E402
import migrate_to_db as mig      # noqa: E402


# ── CSV stub builders ─────────────────────────────────────────────────────────

def _vitals_df():
    return pd.DataFrame({
        "time": ["2024-01-01 00:00", "2024-01-01 00:05"],
        "HR": [75.0, 78.0], "RR": [16.0, 17.0],
        "SpO2": [98.0, 97.0], "Temp": [37.0, 37.1],
        "SBP": [120.0, 122.0], "DBP": [80.0, 81.0], "MAP": [93.0, 95.0],
    })


def _hist_df():
    """Prediction history with the raw special-character column names."""
    return pd.DataFrame({
        "Timestamp": ["2024-01-01 00:00"],
        "Reading":   ["Alert"],
        "SOFA":      [3.0],
        "Risk":      ["Low"],
        "HR":        [75.0], "RR":  [16.0],
        "SpO₂":      [98.0],   # → SpO2
        "Temp (°C)": [37.0],   # → Temp
        "SBP":       [120.0],  "MAP": [93.0],
        "GCS Eye":   [4],      # → GCS_Eye
        "Stress":    [0.2],
    })


def _sp_df(n=3):
    return pd.DataFrame({
        "timestamp":       [f"2024-01-01 00:{i:02d}" for i in range(n)],
        "HR":              [75.0] * n, "RR": [16.0] * n,
        "SpO2":            [98.0] * n, "Temp": [37.0] * n,
        "SBP":             [120.0] * n, "DBP": [80.0] * n, "MAP": [93.0] * n,
        "GCS_eye_opening": [4] * n,
        "stress_score":    [0.2] * n,
        "clinical_note":   ["ok"] * n,
    })


def _fl_df(n=3):
    return pd.DataFrame({
        "feature1":   [float(i) for i in range(n)],
        "feature2":   [float(i * 2) for i in range(n)],
        "sofa_score": [3.0, 7.0, 12.0][:n],
    })


def _create_all_csvs(base: Path):
    """Populate base/ with the full stub CSV tree that migrate() expects."""
    # patient_vitals
    pv = base / "models" / "patient_vitals"
    pv.mkdir(parents=True, exist_ok=True)
    for fname in ("patient_vitals_01.csv", "patient_vitals_02.csv",
                  "patient_vitals_03.csv"):
        _vitals_df().to_csv(pv / fname, index=False)

    # prediction_history  (raw cols — migrate() renames them)
    ph = base / "models" / "prediction_history"
    ph.mkdir(parents=True, exist_ok=True)
    for fname in ("prediction_history_01.csv", "prediction_history_02.csv",
                  "prediction_history_03.csv"):
        _hist_df().to_csv(ph / fname, index=False)

    # sample_patients
    sp = base / "data" / "sample_patients"
    sp.mkdir(parents=True, exist_ok=True)
    for fname in ("sample_patient01_data.csv", "sample_patient02_data.csv",
                  "sample_patient03_data.csv"):
        _sp_df(n=3).to_csv(sp / fname, index=False)

    # fl_training
    fl = base / "data" / "fl_training"
    fl.mkdir(parents=True, exist_ok=True)
    for fname in ("client_0.csv", "client_1.csv", "client_2.csv"):
        _fl_df(n=3).to_csv(fl / fname, index=False)


# ── Core fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def patched_env(tmp_path, monkeypatch):
    """Redirect DB paths and chdir to tmp_path.  No CSV files are created here."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    monkeypatch.setattr(mig,    "DB_PATH", db_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path, db_path


@pytest.fixture
def migrated_db(patched_env):
    """Full stub CSVs + one migrate() call. Returns (tmp_path, db_path)."""
    tmp_path, db_path = patched_env
    _create_all_csvs(tmp_path)
    mig.migrate()
    return tmp_path, db_path


@pytest.fixture
def conn(migrated_db):
    """SQLite connection to the migrated temp DB, closed after the test."""
    _, db_path = migrated_db
    c = sqlite3.connect(db_path)
    yield c
    c.close()


# ─────────────────────────────────────────────────────────────────────────────
# 9A — Happy-path presence checks
# ─────────────────────────────────────────────────────────────────────────────

def test_migrate_completes_without_error(patched_env):
    """migrate() runs to completion without raising an exception."""
    tmp_path, _ = patched_env
    _create_all_csvs(tmp_path)
    mig.migrate()   # must not raise


def test_patient_vitals_has_rows(conn):
    """patient_vitals table is non-empty after migration."""
    n = conn.execute("SELECT COUNT(*) FROM patient_vitals").fetchone()[0]
    assert n > 0


def test_prediction_history_has_rows(conn):
    """prediction_history table is non-empty after migration."""
    n = conn.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]
    assert n > 0


def test_sample_patients_has_rows(conn):
    """sample_patients table is non-empty after migration."""
    n = conn.execute("SELECT COUNT(*) FROM sample_patients").fetchone()[0]
    assert n > 0


def test_fl_training_has_rows_for_all_three_clients(conn):
    """fl_training contains rows for client_id 0, 1, and 2."""
    rows = conn.execute(
        "SELECT DISTINCT client_id FROM fl_training ORDER BY client_id"
    ).fetchall()
    client_ids = [r[0] for r in rows]
    assert client_ids == [0, 1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# 9B — Idempotency
# ─────────────────────────────────────────────────────────────────────────────

def test_migrate_twice_leaves_row_counts_unchanged(patched_env):
    """Calling migrate() a second time produces identical row counts."""
    tmp_path, db_path = patched_env
    _create_all_csvs(tmp_path)

    mig.migrate()
    c1 = sqlite3.connect(db_path)
    counts_first = {
        tbl: c1.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        for tbl in ("patient_vitals", "prediction_history",
                    "sample_patients", "fl_training")
    }
    c1.close()

    mig.migrate()
    c2 = sqlite3.connect(db_path)
    counts_second = {
        tbl: c2.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        for tbl in ("patient_vitals", "prediction_history",
                    "sample_patients", "fl_training")
    }
    c2.close()

    assert counts_first == counts_second


# ─────────────────────────────────────────────────────────────────────────────
# 9C — Column renames (special-character CSV headers → clean DB column names)
# ─────────────────────────────────────────────────────────────────────────────

def test_spo2_column_renamed_from_special_char(conn):
    """SpO₂ in the CSV is stored in the DB column SpO2 (not NULL)."""
    val = conn.execute(
        "SELECT SpO2 FROM prediction_history WHERE patient=1 LIMIT 1"
    ).fetchone()
    assert val is not None and val[0] is not None


def test_temp_column_renamed_from_special_char(conn):
    """Temp (°C) in the CSV is stored in the DB column Temp (not NULL)."""
    val = conn.execute(
        "SELECT Temp FROM prediction_history WHERE patient=1 LIMIT 1"
    ).fetchone()
    assert val is not None and val[0] is not None


def test_gcs_eye_column_renamed_from_space(conn):
    """GCS Eye in the CSV is stored in the DB column GCS_Eye (not NULL)."""
    val = conn.execute(
        "SELECT GCS_Eye FROM prediction_history WHERE patient=1 LIMIT 1"
    ).fetchone()
    assert val is not None and val[0] is not None


# ─────────────────────────────────────────────────────────────────────────────
# 9D — FL client_id assignment
# ─────────────────────────────────────────────────────────────────────────────

def test_fl_training_client_id_column_has_values_0_1_2(conn):
    """fl_training.client_id contains exactly the values {0, 1, 2}."""
    ids = {r[0] for r in conn.execute("SELECT DISTINCT client_id FROM fl_training")}
    assert ids == {0, 1, 2}


def test_fl_training_client0_rows_assigned_client_id_0(conn):
    """Rows imported from client_0.csv carry client_id=0."""
    n = conn.execute(
        "SELECT COUNT(*) FROM fl_training WHERE client_id=0"
    ).fetchone()[0]
    assert n > 0


# ─────────────────────────────────────────────────────────────────────────────
# 9E — Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_csv_skipped_without_crash(patched_env):
    """migrate() completes without error when some CSV files are absent."""
    tmp_path, _ = patched_env
    # Create only patient_vitals_01.csv — 02 and 03 are intentionally missing
    pv = tmp_path / "models" / "patient_vitals"
    pv.mkdir(parents=True, exist_ok=True)
    _vitals_df().to_csv(pv / "patient_vitals_01.csv", index=False)
    # All other CSV directories are also absent — migrate() must still complete
    mig.migrate()   # must not raise


def test_empty_csv_skipped_without_crash(patched_env):
    """migrate() completes without error when a CSV file exists but is empty."""
    tmp_path, _ = patched_env
    # patient_vitals_01.csv exists but has only the header (0 data rows)
    pv = tmp_path / "models" / "patient_vitals"
    pv.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=_vitals_df().columns).to_csv(
        pv / "patient_vitals_01.csv", index=False
    )
    mig.migrate()   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# 9F — Structural checks
# ─────────────────────────────────────────────────────────────────────────────

def test_patient_vitals_patient_column_has_values_1_2_3(conn):
    """patient_vitals.patient contains rows for patients 1, 2, and 3."""
    ids = {r[0] for r in conn.execute("SELECT DISTINCT patient FROM patient_vitals")}
    assert ids == {1, 2, 3}


def test_fl_training_index_idx_ft_client_exists(conn):
    """The idx_ft_client index is present in fl_training after migration."""
    row = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='index' AND name='idx_ft_client'"
    ).fetchone()
    assert row is not None


def test_sample_patients_row_idx_starts_at_zero(conn):
    """sample_patients.row_idx begins at 0 for each patient."""
    min_idx = conn.execute(
        "SELECT MIN(row_idx) FROM sample_patients WHERE patient=1"
    ).fetchone()[0]
    assert min_idx == 0


def test_db_file_size_reported_without_crash(migrated_db):
    """os.path.getsize(DB_PATH) runs without error at the end of migrate()."""
    _, db_path = migrated_db
    size = os.path.getsize(db_path)
    assert size > 0

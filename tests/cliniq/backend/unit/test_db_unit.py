"""
Unit tests for db.py — uses a patched DB_PATH pointing to a temp SQLite file
so tests never touch the production icu_monitor.db.
"""
import os
import sys
import sqlite3
import tempfile
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_UNIT_DIR, "../../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """
    Redirect db.DB_PATH to a fresh temp file for every test.
    Returns the path string so tests can open it directly if needed.
    """
    import db as db_mod
    db_path = str(tmp_path / "test_icu.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    # Run schema init against the patched path
    db_mod.init_db()
    return db_path


# ── Schema initialisation ────────────────────────────────────────────────────

def test_init_db_creates_patient_vitals(tmp_db):
    conn = sqlite3.connect(tmp_db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "patient_vitals" in tables


def test_init_db_creates_prediction_history(tmp_db):
    conn = sqlite3.connect(tmp_db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "prediction_history" in tables


def test_init_db_creates_sample_patients(tmp_db):
    conn = sqlite3.connect(tmp_db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "sample_patients" in tables


def test_init_db_is_idempotent(tmp_db):
    import db as db_mod
    # Should not raise on second call
    db_mod.init_db()
    db_mod.init_db()


# ── patient_vitals CRUD ──────────────────────────────────────────────────────

def _vitals_row(time="2026-01-01 12:00:00"):
    return {"time": time, "HR": 72.0, "RR": 16.0, "SpO2": 98.0,
            "Temp": 37.0, "SBP": 110.0, "DBP": 70.0, "MAP": 83.0}


def test_append_patient_vitals_inserts(tmp_db):
    import db as db_mod
    db_mod.append_patient_vitals(1, _vitals_row())
    df = db_mod.get_patient_vitals(1, limit=20)
    assert len(df) == 1


def test_get_patient_vitals_ascending_order(tmp_db):
    import db as db_mod
    for i in range(3):
        db_mod.append_patient_vitals(1, _vitals_row(time=f"2026-01-01 {10+i}:00:00"))
    df = db_mod.get_patient_vitals(1, limit=20)
    assert list(df["time"]) == sorted(df["time"].tolist())


def test_patient_vitals_sliding_window_keeps_20(tmp_db):
    import db as db_mod
    for i in range(21):
        db_mod.append_patient_vitals(1, _vitals_row(time=f"2026-01-01 {i:02d}:00:00"))
    df = db_mod.get_patient_vitals(1, limit=20)
    assert len(df) == 20


def test_patient_vitals_cross_patient_isolation(tmp_db):
    import db as db_mod
    db_mod.append_patient_vitals(1, _vitals_row())
    df2 = db_mod.get_patient_vitals(2, limit=20)
    assert len(df2) == 0


def test_patient_vitals_limit_respected(tmp_db):
    import db as db_mod
    for i in range(10):
        db_mod.append_patient_vitals(1, _vitals_row(time=f"2026-01-01 {i:02d}:00:00"))
    df = db_mod.get_patient_vitals(1, limit=5)
    assert len(df) <= 5


# ── prediction_history CRUD ──────────────────────────────────────────────────

def _pred_row(ts="2026-01-01 12:00:00"):
    return {
        "Timestamp":  ts,
        "Reading":    "R1",
        "SOFA":       6.0,
        "Risk":       "Moderate Risk",
        "HR":         80.0, "RR": 18.0, "SpO₂": 97.0, "Temp (°C)": 37.0,
        "SBP":       115.0, "DBP": 72.0, "MAP": 86.0,
        "GCS Eye":    4, "Stress": 3.0,
    }


def test_append_prediction_inserts(tmp_db):
    import db as db_mod
    db_mod.append_prediction(1, _pred_row())
    df = db_mod.get_prediction_history(1, limit=50)
    assert len(df) == 1


def test_prediction_history_limit_respected(tmp_db):
    import db as db_mod
    for i in range(10):
        db_mod.append_prediction(1, _pred_row(ts=f"2026-01-01 {i:02d}:00:00"))
    df = db_mod.get_prediction_history(1, limit=3)
    assert len(df) <= 3


def test_prediction_sliding_window_keeps_50(tmp_db):
    import db as db_mod
    for i in range(51):
        db_mod.append_prediction(1, _pred_row(ts=f"2026-01-01 {i // 24:02d}:{i % 24:02d}:00"))
    df = db_mod.get_prediction_history(1, limit=50)
    assert len(df) == 50


def test_prediction_history_cross_patient_isolation(tmp_db):
    import db as db_mod
    db_mod.append_prediction(1, _pred_row())
    df2 = db_mod.get_prediction_history(2, limit=50)
    assert len(df2) == 0


def test_prediction_history_sofa_values(tmp_db):
    import db as db_mod
    db_mod.append_prediction(1, _pred_row())
    df = db_mod.get_prediction_history(1, limit=50)
    assert df.iloc[0]["SOFA"] == pytest.approx(6.0)


# ── sample_patients helpers ──────────────────────────────────────────────────

def test_get_sample_patient_count_empty(tmp_db):
    import db as db_mod
    count = db_mod.get_sample_patient_count(1)
    # Empty temp DB has 0 rows
    assert count == 0


def test_get_sample_patient_row_missing(tmp_db):
    import db as db_mod
    row = db_mod.get_sample_patient_row(1, 999)
    assert row is None


# ── Connection quality ────────────────────────────────────────────────────────

def test_get_conn_wal_mode(tmp_db):
    import db as db_mod
    conn = db_mod.get_conn()
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode == "wal"

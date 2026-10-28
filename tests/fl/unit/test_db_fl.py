"""
Phase 1 — Database FL Helpers  (test_db_fl.py)
===============================================
Tests for fl_training_exists() and get_fl_client_data() in db.py,
plus the fl_training table schema and get_conn() extended behaviour.

All tests use the tmp_db fixture from conftest.py:
  - db.DB_PATH is monkeypatched to a fresh temp SQLite file per test
  - init_db() is called so all non-dynamic tables exist
  - fl_training is created manually via pandas.to_sql where needed

21 tests across 4 groups:
  1A  fl_training_exists()          — 4 tests
  1B  get_fl_client_data()          — 11 tests
  1C  fl_training schema/triggers   — 4 tests
  1D  get_conn() extended           — 2 tests
"""
import os
import sys
import sqlite3
import warnings

import pandas as pd
import pytest

# ── sys.path (fallback for direct pytest invocation outside icu_monitor/) ─────
_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))   # tests/fl/unit/
_FL_DIR   = os.path.dirname(_UNIT_DIR)                   # tests/fl/
_TESTS    = os.path.dirname(_FL_DIR)                     # tests/
_ROOT     = os.path.dirname(_TESTS)                      # icu_monitor/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rows(client_id, n=5, *, with_null_sofa=False, with_nan_feature=False,
               with_timestamps=False):
    """Return a list of row dicts for fl_training."""
    rows = []
    for i in range(n):
        row = {
            "client_id":  client_id,
            "feature1":   float(i),
            "feature2":   float(i * 2),
            "sofa_score": None if (with_null_sofa and i == 0) else float(i % 10),
        }
        if with_nan_feature and i == 1:
            row["feature1"] = float("nan")
        if with_timestamps:
            row["created_at"] = None
            row["updated_at"] = None
        rows.append(row)
    return rows


def _create_fl_table(db_path, rows, if_exists="replace"):
    """Create / extend fl_training via pandas.to_sql (mirrors migrate_to_db.py)."""
    conn = sqlite3.connect(db_path)
    pd.DataFrame(rows).to_sql("fl_training", conn, if_exists=if_exists, index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ft_client ON fl_training(client_id)"
    )
    conn.commit()
    conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# 1A — fl_training_exists()
# ═════════════════════════════════════════════════════════════════════════════

def test_fl_training_exists_raises_when_table_absent(tmp_db):
    """fl_training is not created by init_db() → OperationalError on first call."""
    import db as db_mod
    # init_db() was already called by tmp_db fixture; fl_training still absent
    with pytest.raises(Exception):
        db_mod.fl_training_exists()


def test_fl_training_exists_false_when_empty(tmp_db):
    """Empty fl_training table → fl_training_exists() returns False."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=1))
    # Delete the only row to leave an empty table
    conn = sqlite3.connect(tmp_db)
    conn.execute("DELETE FROM fl_training")
    conn.commit()
    conn.close()
    assert db_mod.fl_training_exists() is False


def test_fl_training_exists_true_after_one_row(tmp_db):
    """One row inserted → fl_training_exists() returns True."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=1))
    assert db_mod.fl_training_exists() is True


def test_fl_training_exists_true_for_any_client(tmp_db):
    """Rows with client_id=2 only → fl_training_exists() still returns True."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(client_id=2, n=3))
    assert db_mod.fl_training_exists() is True


# ═════════════════════════════════════════════════════════════════════════════
# 1B — get_fl_client_data(client_id)
# ═════════════════════════════════════════════════════════════════════════════

def test_get_fl_client_data_returns_dataframe(tmp_db):
    """get_fl_client_data returns a pandas DataFrame."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=5))
    result = db_mod.get_fl_client_data(0)
    assert isinstance(result, pd.DataFrame)


def test_get_fl_client_data_drops_id_column(tmp_db):
    """If an id column is present in fl_training, it is dropped from the result."""
    import db as db_mod
    rows = [{"id": i, "client_id": 0, "feature1": float(i), "sofa_score": float(i)}
            for i in range(3)]
    _create_fl_table(tmp_db, rows)
    result = db_mod.get_fl_client_data(0)
    assert "id" not in result.columns


def test_get_fl_client_data_drops_client_id_column(tmp_db):
    """client_id column is always dropped from the returned DataFrame."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=5))
    result = db_mod.get_fl_client_data(0)
    assert "client_id" not in result.columns


def test_get_fl_client_data_filters_by_client_id(tmp_db):
    """Only rows for the requested client_id are returned."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=10))
    _create_fl_table(tmp_db, _make_rows(1, n=5), if_exists="append")
    result = db_mod.get_fl_client_data(0)
    assert len(result) == 10


def test_get_fl_client_data_empty_for_unknown_client(tmp_db):
    """Requesting an unknown client_id returns an empty DataFrame."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=5))
    result = db_mod.get_fl_client_data(99)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_get_fl_client_data_drops_null_sofa_rows(tmp_db):
    """Rows with NULL sofa_score are silently dropped."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=5, with_null_sofa=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = db_mod.get_fl_client_data(0)
    # 5 rows inserted; row 0 had NULL sofa_score → 4 remain
    assert len(result) == 4


def test_get_fl_client_data_warns_on_null_sofa(tmp_db):
    """UserWarning is emitted when NULL sofa_score rows are dropped."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=5, with_null_sofa=True))
    with pytest.warns(UserWarning, match="sofa_score"):
        db_mod.get_fl_client_data(0)


def test_get_fl_client_data_fills_nan_features_with_zero(tmp_db):
    """NaN values in feature columns are replaced with 0 via fillna(0)."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=5, with_nan_feature=True))
    result = db_mod.get_fl_client_data(0)
    assert result["feature1"].isna().sum() == 0
    assert (result["feature1"] == 0.0).any()


def test_get_fl_client_data_sofa_score_present(tmp_db):
    """sofa_score column survives the column-drop step."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=5))
    result = db_mod.get_fl_client_data(0)
    assert "sofa_score" in result.columns


def test_get_fl_client_data_correct_row_count(tmp_db):
    """Row count equals the number of inserted rows when no NULL sofa_score."""
    import db as db_mod
    n = 12
    _create_fl_table(tmp_db, _make_rows(0, n=n))
    result = db_mod.get_fl_client_data(0)
    assert len(result) == n


def test_get_fl_client_data_cross_client_isolation(tmp_db):
    """Two clients return independent DataFrames with their own row counts."""
    import db as db_mod
    _create_fl_table(tmp_db, _make_rows(0, n=8))
    _create_fl_table(tmp_db, _make_rows(1, n=6), if_exists="append")
    r0 = db_mod.get_fl_client_data(0)
    r1 = db_mod.get_fl_client_data(1)
    assert len(r0) == 8
    assert len(r1) == 6


# ═════════════════════════════════════════════════════════════════════════════
# 1C — fl_training table schema & triggers
# ═════════════════════════════════════════════════════════════════════════════

def test_fl_training_table_created_via_pandas_to_sql(tmp_db):
    """pandas.to_sql creates the fl_training table (dynamic schema creation)."""
    _create_fl_table(tmp_db, _make_rows(0, n=3))
    conn = sqlite3.connect(tmp_db)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "fl_training" in tables


def test_fl_training_client_id_index_created(tmp_db):
    """idx_ft_client index is created on fl_training(client_id) after first insert."""
    _create_fl_table(tmp_db, _make_rows(0, n=3))
    conn = sqlite3.connect(tmp_db)
    indexes = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()}
    conn.close()
    assert "idx_ft_client" in indexes


def test_fl_training_created_at_trigger_fires(tmp_db):
    """trg_ft_created_at trigger auto-sets created_at on INSERT into fl_training."""
    import db as db_mod
    # Create table with timestamp columns so the trigger can write them
    _create_fl_table(tmp_db, _make_rows(0, n=2, with_timestamps=True))
    # Reinstall trigger now that the table exists
    db_mod.init_db()
    # Insert a new row via raw SQL and verify trigger sets created_at
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        "INSERT INTO fl_training "
        "(client_id, feature1, feature2, sofa_score, created_at, updated_at) "
        "VALUES (0, 99.0, 198.0, 7.0, NULL, NULL)"
    )
    conn.commit()
    row = conn.execute(
        "SELECT created_at FROM fl_training WHERE feature1 = 99.0"
    ).fetchone()
    conn.close()
    assert row is not None, "Row not found after INSERT"
    assert row[0] is not None, "created_at should be set by trigger"


def test_init_db_skips_fl_trigger_when_table_absent(tmp_db):
    """init_db() completes without error and does NOT create fl_training trigger
    when the fl_training table has not yet been created."""
    import db as db_mod
    # tmp_db fixture already ran init_db() once; fl_training still absent
    db_mod.init_db()  # must not raise
    conn = sqlite3.connect(tmp_db)
    triggers = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()}
    conn.close()
    assert "trg_ft_created_at" not in triggers


# ═════════════════════════════════════════════════════════════════════════════
# 1D — get_conn() extended
# ═════════════════════════════════════════════════════════════════════════════

def test_get_conn_foreign_keys_enabled(tmp_db):
    """get_conn() sets PRAGMA foreign_keys=ON (returns 1)."""
    import db as db_mod
    conn = db_mod.get_conn()
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    conn.close()
    assert fk == 1


def test_get_conn_concurrent_wal_writes(tmp_db):
    """Two connections under WAL mode can write and read concurrently without corruption."""
    import db as db_mod
    conn1 = db_mod.get_conn()
    conn2 = db_mod.get_conn()

    # Write from conn1
    conn1.execute(
        "INSERT INTO patient_vitals "
        "(patient, time, HR, RR, SpO2, Temp, SBP, DBP, MAP) "
        "VALUES (1, '2026-01-01 00:00:00', 70, 16, 98, 37, 110, 70, 83)"
    )
    conn1.commit()

    # Read from conn2 — committed data should be visible
    count = conn2.execute(
        "SELECT COUNT(*) FROM patient_vitals WHERE patient = 1"
    ).fetchone()[0]
    conn1.close()
    conn2.close()
    assert count == 1

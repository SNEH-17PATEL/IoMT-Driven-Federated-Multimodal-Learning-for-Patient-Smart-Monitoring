"""
Shared fixtures for all FL test phases (tests/fl/).
"""
import os
import sys
import sqlite3

import numpy as np
import pandas as pd
import pytest

# ── sys.path: ensure icu_monitor/ root is importable ─────────────────────────
_FL_DIR = os.path.dirname(os.path.abspath(__file__))   # tests/fl/
_TESTS  = os.path.dirname(_FL_DIR)                      # tests/
_ROOT   = os.path.dirname(_TESTS)                       # icu_monitor/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── tmp_db ────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect db.DB_PATH to a fresh temp SQLite file for every test.

    Returns the path string so tests can open the file directly.
    Calls init_db() after patching so all non-dynamic tables exist.
    """
    import db as db_mod
    db_path = str(tmp_path / "test_fl.db")
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    db_mod.init_db()
    return db_path


# ── synthetic_client_data ─────────────────────────────────────────────────────

@pytest.fixture
def synthetic_client_data():
    """Return (X, y) with realistic SOFA distribution: 70% Low, 24% Mod, 6% High.

    X: shape (200, 10), float32
    y: shape (200,),    float32, range 0–24
    """
    rng = np.random.default_rng(42)
    n = 200
    input_dim = 10
    X = rng.standard_normal((n, input_dim)).astype(np.float32)

    n_low  = int(n * 0.70)
    n_mod  = int(n * 0.24)
    n_high = n - n_low - n_mod

    y = np.concatenate([
        rng.uniform(0, 5,  n_low),
        rng.uniform(5, 10, n_mod),
        rng.uniform(10, 24, n_high),
    ]).astype(np.float32)
    rng.shuffle(y)
    return X, y


# ── mini_model ────────────────────────────────────────────────────────────────

@pytest.fixture
def mini_model():
    """Return ICUModel(input_dim=10) — tiny network for speed."""
    from model_utils import ICUModel
    return ICUModel(10)


# ── mini_weights ──────────────────────────────────────────────────────────────

@pytest.fixture
def mini_weights(mini_model):
    """Return get_weights(mini_model) — list of numpy arrays."""
    from model_utils import get_weights
    return get_weights(mini_model)


# ── fl_training_rows ──────────────────────────────────────────────────────────

@pytest.fixture
def fl_training_rows():
    """Factory: insert n synthetic rows into fl_training in tmp_db.

    Usage::
        def test_foo(tmp_db, fl_training_rows):
            fl_training_rows(tmp_db, client_id=0, n=50)
    """
    def _insert(db_path, client_id, n=50, with_null_sofa=False,
                with_nan_feature=False):
        rows = []
        for i in range(n):
            row = {
                "client_id":   client_id,
                "feature1":    float(i),
                "feature2":    float(i * 2),
                "sofa_score":  (
                    None if (with_null_sofa and i == 0)
                    else float(i % 15)
                ),
            }
            if with_nan_feature and i == 1:
                row["feature1"] = float("nan")
            rows.append(row)
        conn = sqlite3.connect(db_path)
        df = pd.DataFrame(rows)
        existing = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fl_training'"
        ).fetchone()
        if_exists = "append" if existing else "replace"
        df.to_sql("fl_training", conn, if_exists=if_exists, index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ft_client ON fl_training(client_id)"
        )
        conn.commit()
        conn.close()

    return _insert

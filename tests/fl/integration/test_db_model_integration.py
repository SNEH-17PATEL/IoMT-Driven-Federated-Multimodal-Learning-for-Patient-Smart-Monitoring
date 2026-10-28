"""
Phase 7 — Database ↔ Model Integration  (test_db_model_integration.py)
=======================================================================
Verifies the pipeline from db.get_fl_client_data() through model_utils
functions using the tmp_db and fl_training_rows fixtures.

The tmp_db fixture redirects db.DB_PATH to a fresh temp SQLite file and
calls init_db() — so every test works on an isolated, empty database and
the production icu_monitor.db is never touched.

The fl_training_rows factory inserts synthetic rows (feature1, feature2,
sofa_score) for a given client_id.  After get_fl_client_data(), id and
client_id are dropped, leaving only feature columns + sofa_score.

10 tests:
  7A  DB → ICUDataset              —  2 tests
  7B  DB → Training / Evaluation  —  4 tests
  7C  Weight round-trip            —  1 test
  7D  Full pipeline utilities      —  3 tests
"""
import os
import sys

import numpy as np
import numpy.testing as npt
import pytest

# ── sys.path fallback ─────────────────────────────────────────────────────────
_INTEG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_INTEG_DIR, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import db as db_mod                      # noqa: E402
from model_utils import (                # noqa: E402
    ICUDataset, ICUModel,
    train_model, evaluate_model,
    get_weights, set_weights,
    compute_sofa_risk_metrics,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(db_path, client_id):
    """Return (X: float32, y: float32) from the temp DB for client_id."""
    df = db_mod.get_fl_client_data(client_id)
    X  = df.drop(columns=["sofa_score"]).values.astype(np.float32)
    y  = df["sofa_score"].values.astype(np.float32)
    return X, y


def _trained_model(X, y, epochs=2):
    """Return a freshly trained ICUModel on the given data."""
    model = ICUModel(X.shape[1])
    train_model(model, X, y, epochs=epochs, lr=0.01, oversample=False)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 7A — DB → ICUDataset
# ═════════════════════════════════════════════════════════════════════════════

def test_db_to_dataset_length(tmp_db, fl_training_rows):
    """get_fl_client_data(0) → ICUDataset has the same row count as inserted."""
    fl_training_rows(tmp_db, client_id=0, n=50)
    X, y = _load(tmp_db, 0)
    dataset = ICUDataset(X, y)
    assert len(dataset) == 50


def test_db_to_dataset_x_values_match(tmp_db, fl_training_rows):
    """ICUDataset.X.numpy() matches the raw numpy array from get_fl_client_data."""
    fl_training_rows(tmp_db, client_id=0, n=30)
    X, y = _load(tmp_db, 0)
    dataset = ICUDataset(X, y)
    npt.assert_array_almost_equal(dataset.X.numpy(), X)


# ═════════════════════════════════════════════════════════════════════════════
# 7B — DB → Training / Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def test_db_to_train_model_completes(tmp_db, fl_training_rows):
    """get_fl_client_data → train_model completes without error."""
    fl_training_rows(tmp_db, client_id=0, n=50)
    X, y = _load(tmp_db, 0)
    loss = train_model(ICUModel(X.shape[1]), X, y, epochs=2, lr=0.01, oversample=False)
    assert isinstance(loss, float)


def test_db_to_evaluate_model_returns_finite_metrics(tmp_db, fl_training_rows):
    """get_fl_client_data → evaluate_model returns finite mse, mae, r2."""
    fl_training_rows(tmp_db, client_id=0, n=50)
    X, y = _load(tmp_db, 0)
    model = _trained_model(X, y)
    metrics = evaluate_model(model, X, y)
    assert np.isfinite(metrics["mse"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["r2"])


def test_different_clients_return_different_data(tmp_db, fl_training_rows):
    """get_fl_client_data isolates rows per client — client 0 ≠ client 1 shape."""
    fl_training_rows(tmp_db, client_id=0, n=50)
    fl_training_rows(tmp_db, client_id=1, n=30)
    df0 = db_mod.get_fl_client_data(0)
    df1 = db_mod.get_fl_client_data(1)
    assert len(df0) != len(df1)


def test_trained_model_weights_differ_from_untrained(tmp_db, fl_training_rows):
    """A model trained on DB data has at least one weight array that differs
    from a fresh (randomly-initialised) model of the same architecture."""
    fl_training_rows(tmp_db, client_id=0, n=50)
    X, y = _load(tmp_db, 0)

    input_dim = X.shape[1]
    untrained = get_weights(ICUModel(input_dim))
    model = _trained_model(X, y, epochs=3)
    trained = get_weights(model)

    any_different = any(
        not np.allclose(w_t, w_u) for w_t, w_u in zip(trained, untrained)
    )
    assert any_different, "Training did not update any weights"


# ═════════════════════════════════════════════════════════════════════════════
# 7C — Weight round-trip
# ═════════════════════════════════════════════════════════════════════════════

def test_weights_round_trip_on_db_trained_model(tmp_db, fl_training_rows):
    """get_weights → set_weights → get_weights produces bit-identical arrays."""
    fl_training_rows(tmp_db, client_id=0, n=50)
    X, y = _load(tmp_db, 0)

    model = _trained_model(X, y)
    original = get_weights(model)

    fresh = ICUModel(X.shape[1])
    set_weights(fresh, original)
    recovered = get_weights(fresh)

    for orig_w, rec_w in zip(original, recovered):
        npt.assert_array_equal(orig_w, rec_w)


# ═════════════════════════════════════════════════════════════════════════════
# 7D — Full pipeline utilities
# ═════════════════════════════════════════════════════════════════════════════

def test_compute_sofa_risk_metrics_with_db_data(tmp_db, fl_training_rows):
    """compute_sofa_risk_metrics() works when y_true / y_score come from DB data."""
    import torch
    # n=60 guarantees SOFA values 0-14 (i%15), ensuring multi-class presence
    fl_training_rows(tmp_db, client_id=0, n=60)
    X, y = _load(tmp_db, 0)

    model = _trained_model(X, y, epochs=3)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

    metrics = compute_sofa_risk_metrics(y, preds)
    assert "risk_bands"     in metrics
    assert "high_risk_alert" in metrics


def test_nan_features_filled_with_zero(tmp_db, fl_training_rows):
    """NaN in a feature column from the DB is filled with 0 by get_fl_client_data."""
    fl_training_rows(tmp_db, client_id=0, n=20, with_nan_feature=True)
    df = db_mod.get_fl_client_data(0)
    # fillna(0) must have eliminated every NaN — including the injected one
    assert not df.isnull().any().any()
    # The specific injected row had feature2=2.0 (i=1) and feature1 was NaN
    nan_row = df[df["feature2"] == 2.0]
    assert len(nan_row) == 1
    assert nan_row["feature1"].iloc[0] == pytest.approx(0.0)


def test_train_test_split_correct_sizes(tmp_db, fl_training_rows):
    """get_fl_client_data + train_test_split → train/val sizes sum to total rows."""
    from sklearn.model_selection import train_test_split

    fl_training_rows(tmp_db, client_id=0, n=50)
    X, y = _load(tmp_db, 0)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    assert len(X_tr) + len(X_val) == len(X)
    assert len(y_tr) + len(y_val) == len(y)
    # 20% of 50 = 10
    assert len(X_val) == 10

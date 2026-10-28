"""
Phase 6 — Train Federated Helpers  (test_train_federated_fns.py)
=================================================================
Tests make_noniid_split, HospitalClient (train variant),
SaveBestStrategy (FedYogi variant), and config constants from
train_federated.py.

Import strategy (train_federated.py has heavy module-level side-effects):
  • fl.simulation.start_simulation → side_effect that populates
    _best_weights/_last_weights with real Parameters so the post-training
    code (model save, evaluation, metadata write) can complete without error.
  • np.save → guarded for shap_background.npy to avoid overwriting artefact.
  • torch.save → MagicMock to prevent writing federated_model.pth.
  • builtins.open (write mode) → StringIO for training_metadata.json so no
    invalid JSON (best_eval_loss=inf) can reach disk.
  All guards are restored immediately after the import.

30 tests across 4 groups:
  6A  make_noniid_split()           —  8 tests
  6B  HospitalClient (train)        —  7 tests
  6C  SaveBestStrategy (FedYogi)    —  5 tests
  6D  Config constants              — 10 tests
"""
import builtins
import os
import sys
from io import StringIO
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pytest

# ── sys.path fallback ─────────────────────────────────────────────────────────
_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_UNIT_DIR, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Pre-import mocks ──────────────────────────────────────────────────────────
# train_federated.py runs an entire FL simulation, saves model weights, runs
# evaluation, and writes JSON — all at module level.  The four guards below
# keep the import fast and side-effect-free on disk.

import flwr as fl          # noqa: E402
import torch               # noqa: E402

# 1) start_simulation → populate _best_weights / _last_weights with real Parameters
#    so the post-training code can complete without raising RuntimeError.
def _mock_start_simulation(*args, **kwargs):
    from flwr.common import ndarrays_to_parameters
    from model_utils import ICUModel, get_weights
    trf = sys.modules.get("train_federated")
    if trf and hasattr(trf, "_best_weights"):
        fake_model  = ICUModel(trf.input_dim)
        fake_params = ndarrays_to_parameters(get_weights(fake_model))
        trf._best_weights["params"] = fake_params
        trf._last_weights["params"] = fake_params

fl.simulation.start_simulation = _mock_start_simulation

# 2) np.save → guard shap_background.npy (don't overwrite the artefact)
_real_np_save = np.save
def _guarded_np_save(file, arr, *a, **kw):
    if "shap_background" in str(file):
        return
    return _real_np_save(file, arr, *a, **kw)
np.save = _guarded_np_save

# 3) torch.save → no-op (don't overwrite federated_model.pth)
_real_torch_save = torch.save
torch.save = MagicMock()

# 4) builtins.open (write) → redirect training_metadata.json to StringIO sink
_real_open = builtins.open
def _guarded_open(path, mode="r", *a, **kw):
    if "w" in str(mode) and "training_metadata" in str(path):
        return StringIO()
    return _real_open(path, mode, *a, **kw)
builtins.open = _guarded_open

import train_federated as trf  # noqa: E402 — must come after all guards above

# Restore originals immediately so other test files are unaffected
np.save        = _real_np_save
torch.save     = _real_torch_save
builtins.open  = _real_open


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_pool(n=300, seed=0):
    """Return (X, y) with SOFA distribution suitable for make_noniid_split tests.

    Uses 50/25/25 Low/Mod/High so Hospital 2 can easily reach ≥50% High-SOFA
    after the non-IID split assigns it 70% of the High-SOFA stratum.
    """
    rng    = np.random.default_rng(seed)
    X      = rng.standard_normal((n, 5)).astype(np.float32)
    n_low  = int(n * 0.50)           # 150
    n_mod  = int(n * 0.25)           # 75
    n_high = n - n_low - n_mod       # 75
    y = np.concatenate([
        rng.uniform(0,  5,  n_low),
        rng.uniform(5,  10, n_mod),
        rng.uniform(10, 24, n_high),
    ]).astype(np.float32)
    rng.shuffle(y)
    return X, y


def _make_client(n=120):
    """Return a HospitalClient (train variant) with synthetic data."""
    rng = np.random.default_rng(7)
    X   = rng.standard_normal((n, trf.input_dim)).astype(np.float32)
    y   = rng.uniform(0, 24, n).astype(np.float32)
    return trf.HospitalClient(X, y, hospital_id=0, name="Test ICU")


def _mock_metrics():
    return {"mse": 0.5, "mae": 0.4, "r2": 0.6}


# ── Autouse: reset train_federated module globals before every test ─────────

@pytest.fixture(autouse=True)
def _reset_trf_globals(monkeypatch):
    """Isolate module-level tracking state between tests."""
    monkeypatch.setattr(trf, "_best_loss",  float("inf"))
    monkeypatch.setattr(trf, "_best_round", 0)
    monkeypatch.setattr(trf, "_best_weights", {})
    monkeypatch.setattr(trf, "_last_weights", {})


# ═════════════════════════════════════════════════════════════════════════════
# 6A — make_noniid_split()
# ═════════════════════════════════════════════════════════════════════════════

def test_noniid_split_returns_exactly_3_tuples():
    """make_noniid_split returns a list of exactly 3 (X, y) tuples."""
    X, y = _synthetic_pool()
    splits = trf.make_noniid_split(X, y, seed=42)
    assert isinstance(splits, list) and len(splits) == 3


def test_noniid_split_each_element_is_xy_pair():
    """Every element of the returned list is a (ndarray, ndarray) tuple."""
    X, y = _synthetic_pool()
    splits = trf.make_noniid_split(X, y, seed=42)
    for X_h, y_h in splits:
        assert isinstance(X_h, np.ndarray)
        assert isinstance(y_h, np.ndarray)


def test_noniid_split_total_samples_preserved():
    """Sum of samples across all 3 hospitals equals the input total."""
    X, y = _synthetic_pool(n=300)
    splits = trf.make_noniid_split(X, y, seed=42)
    assert sum(len(y_h) for _, y_h in splits) == 300


def test_noniid_split_hospital0_low_sofa_majority():
    """Hospital 0 (General ICU) receives ≥50% Low-SOFA samples (<5)."""
    X, y = _synthetic_pool(n=300)
    splits = trf.make_noniid_split(X, y, seed=42)
    _, y0 = splits[0]
    assert (y0 < 5).mean() >= 0.50


def test_noniid_split_hospital2_high_sofa_majority():
    """Hospital 2 (Cardiac/Trauma ICU) receives ≥50% High-SOFA samples (≥10)."""
    X, y = _synthetic_pool(n=300)
    splits = trf.make_noniid_split(X, y, seed=42)
    _, y2 = splits[2]
    assert (y2 >= 10).mean() >= 0.50


def test_noniid_split_is_deterministic():
    """Same seed produces bit-identical splits on repeated calls."""
    X, y = _synthetic_pool()
    s1 = trf.make_noniid_split(X, y, seed=7)
    s2 = trf.make_noniid_split(X, y, seed=7)
    for (X1, y1), (X2, y2) in zip(s1, s2):
        npt.assert_array_equal(X1, X2)
        npt.assert_array_equal(y1, y2)


def test_noniid_split_different_seeds_produce_different_results():
    """Different seeds produce at least one differing split."""
    X, y = _synthetic_pool()
    sa = trf.make_noniid_split(X, y, seed=1)
    sb = trf.make_noniid_split(X, y, seed=2)
    all_equal = all(
        np.array_equal(ya, yb) for (_, ya), (_, yb) in zip(sa, sb)
    )
    assert not all_equal


def test_noniid_split_handles_empty_high_sofa_class():
    """Split completes without error when no High-SOFA samples exist."""
    rng = np.random.default_rng(99)
    n   = 120
    X   = rng.standard_normal((n, 3)).astype(np.float32)
    y   = rng.uniform(0, 5, n).astype(np.float32)   # all Low-SOFA
    splits = trf.make_noniid_split(X, y, seed=42)
    assert sum(len(y_h) for _, y_h in splits) == n


# ═════════════════════════════════════════════════════════════════════════════
# 6B — HospitalClient (train variant)
# ═════════════════════════════════════════════════════════════════════════════

def test_hospital_client_get_parameters_returns_model_weights():
    """get_parameters({}) returns the weights of self.model as numpy arrays."""
    from model_utils import get_weights
    c = _make_client()
    params = c.get_parameters({})
    assert isinstance(params, list)
    assert len(params) == len(get_weights(c.model))
    assert all(isinstance(w, np.ndarray) for w in params)


def test_hospital_client_fit_calls_set_weights_before_train(monkeypatch):
    """fit() calls set_weights(self.model, parameters) before train_model."""
    order = []
    monkeypatch.setattr(trf, "set_weights",
                        lambda m, p: order.append("set_weights"))
    monkeypatch.setattr(trf, "train_model",
                        lambda *a, **kw: order.append("train") or 0.5)
    monkeypatch.setattr(trf, "evaluate_model",
                        MagicMock(return_value=_mock_metrics()))
    c = _make_client()
    c.fit(c.get_parameters({}), {"server_round": 1})
    assert order[0] == "set_weights"
    assert "train" in order


def test_hospital_client_fit_passes_server_round_to_lr_formula(monkeypatch):
    """fit() computes lr = max(1e-4, BASE_LR * LR_DECAY**(round-1))."""
    captured = {}
    def _fake_train(m, X, y, *, lr, **kw):
        captured["lr"] = lr
        return 0.5
    monkeypatch.setattr(trf, "set_weights",    MagicMock())
    monkeypatch.setattr(trf, "train_model",    _fake_train)
    monkeypatch.setattr(trf, "evaluate_model", MagicMock(return_value=_mock_metrics()))
    c = _make_client()
    c.fit(c.get_parameters({}), {"server_round": 10})
    expected = max(1e-4, trf.BASE_LR * (trf.LR_DECAY ** 9))
    assert captured["lr"] == pytest.approx(expected, rel=1e-6)


def test_hospital_client_fit_dp_enabled_calls_apply_dp(monkeypatch):
    """When USE_DP=True, apply_dp_to_update is called exactly once."""
    monkeypatch.setattr(trf, "set_weights",    MagicMock())
    monkeypatch.setattr(trf, "train_model",    MagicMock(return_value=0.5))
    monkeypatch.setattr(trf, "evaluate_model", MagicMock(return_value=_mock_metrics()))
    fake_w = [np.zeros(1)]
    monkeypatch.setattr(trf, "get_weights",    MagicMock(return_value=fake_w))
    mock_dp = MagicMock(return_value=fake_w)
    monkeypatch.setattr(trf, "apply_dp_to_update", mock_dp)
    monkeypatch.setattr(trf, "USE_DP", True)
    c = _make_client()
    c.fit(c.get_parameters({}), {"server_round": 1})
    mock_dp.assert_called_once()


def test_hospital_client_fit_returns_correct_layer_count(monkeypatch):
    """fit() first return value has the same number of layers as the model."""
    from model_utils import get_weights
    monkeypatch.setattr(trf, "train_model",    MagicMock(return_value=0.5))
    monkeypatch.setattr(trf, "evaluate_model", MagicMock(return_value=_mock_metrics()))
    c = _make_client()
    weights, n, extras = c.fit(c.get_parameters({}), {"server_round": 1})
    assert isinstance(weights, list)
    assert len(weights) == len(get_weights(c.model))
    assert extras == {}


def test_hospital_client_evaluate_calls_set_weights_before_eval(monkeypatch):
    """evaluate() applies received parameters before calling evaluate_model."""
    order = []
    monkeypatch.setattr(trf, "set_weights",
                        lambda m, p: order.append("set_weights"))
    monkeypatch.setattr(trf, "evaluate_model",
                        lambda *a: order.append("evaluate") or _mock_metrics())
    c = _make_client()
    c.evaluate(c.get_parameters({}), {})
    assert order == ["set_weights", "evaluate"]


def test_hospital_client_evaluate_returns_correct_structure(monkeypatch):
    """evaluate() returns (mse: float, n_val: int, dict with mae + r2)."""
    monkeypatch.setattr(trf, "evaluate_model",
                        MagicMock(return_value={"mse": 1.5, "mae": 0.8, "r2": 0.3}))
    c = _make_client()
    result = c.evaluate(c.get_parameters({}), {})
    assert isinstance(result, tuple) and len(result) == 3
    mse, n_val, metrics = result
    assert mse == pytest.approx(1.5)
    assert isinstance(n_val, int) and n_val > 0
    assert "mae" in metrics and "r2" in metrics


# ═════════════════════════════════════════════════════════════════════════════
# 6C — SaveBestStrategy (FedYogi variant)
# ═════════════════════════════════════════════════════════════════════════════

def test_strategy_configure_fit_adds_server_round(monkeypatch):
    """configure_fit() embeds server_round in every FitIns config dict."""
    mock_cm = MagicMock()
    mock_cm.num_available.return_value = 3
    mock_clients = [MagicMock() for _ in range(3)]
    mock_cm.sample.return_value = mock_clients
    monkeypatch.setattr(trf.strategy, "num_fit_clients",
                        MagicMock(return_value=(3, 3)))
    instructions = trf.strategy.configure_fit(7, MagicMock(), mock_cm)
    assert len(instructions) == 3
    for _, fit_ins in instructions:
        assert fit_ins.config.get("server_round") == 7


def test_strategy_aggregate_fit_stores_params_in_last_weights(monkeypatch):
    """aggregate_fit() stores the aggregated result in _last_weights['params']."""
    fake_params = MagicMock()
    monkeypatch.setattr(fl.server.strategy.FedYogi, "aggregate_fit",
                        MagicMock(return_value=(fake_params, {})))
    trf.strategy.aggregate_fit(1, [], [])
    assert trf._last_weights.get("params") is fake_params


def test_strategy_aggregate_evaluate_updates_best_loss(monkeypatch):
    """aggregate_evaluate() updates _best_loss when avg_loss is a new minimum."""
    monkeypatch.setattr(fl.server.strategy.FedYogi, "aggregate_evaluate",
                        MagicMock(return_value=(None, {})))
    results = [(MagicMock(), MagicMock(loss=0.4))]
    trf.strategy.aggregate_evaluate(1, results, [])
    assert trf._best_loss == pytest.approx(0.4)


def test_strategy_aggregate_evaluate_copies_last_to_best_on_new_best(monkeypatch):
    """On a new best, _best_weights['params'] is set from _last_weights['params']."""
    monkeypatch.setattr(fl.server.strategy.FedYogi, "aggregate_evaluate",
                        MagicMock(return_value=(None, {})))
    fake_params = MagicMock()
    trf._last_weights["params"] = fake_params     # pre-populate (autouse gave us {})
    results = [(MagicMock(), MagicMock(loss=0.2))]
    trf.strategy.aggregate_evaluate(1, results, [])
    assert trf._best_weights.get("params") is fake_params


def test_strategy_aggregate_evaluate_calls_super(monkeypatch):
    """aggregate_evaluate() delegates to the parent FedYogi.aggregate_evaluate."""
    mock_parent = MagicMock(return_value=(None, {}))
    monkeypatch.setattr(fl.server.strategy.FedYogi, "aggregate_evaluate", mock_parent)
    results = [(MagicMock(), MagicMock(loss=0.5))]
    trf.strategy.aggregate_evaluate(1, results, [])
    mock_parent.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# 6D — Config constants
# ═════════════════════════════════════════════════════════════════════════════

def test_num_rounds_is_100():
    assert trf.NUM_ROUNDS == 100


def test_epochs_per_round_is_3():
    assert trf.EPOCHS_PER_ROUND == 3


def test_batch_size_is_64():
    assert trf.BATCH_SIZE == 64


def test_mu_fedprox_is_0_5():
    assert trf.MU_FEDPROX == pytest.approx(0.5)


def test_server_eta_is_0_01():
    assert trf.SERVER_ETA == pytest.approx(0.01)


def test_server_beta1_is_0_9():
    assert trf.SERVER_BETA1 == pytest.approx(0.9)


def test_server_beta2_is_0_99():
    assert trf.SERVER_BETA2 == pytest.approx(0.99)


def test_use_dp_default_is_false():
    assert trf.USE_DP is False


def test_oversample_default_is_false():
    assert trf.OVERSAMPLE is False


def test_lr_decay_is_0_99():
    assert trf.LR_DECAY == pytest.approx(0.99)

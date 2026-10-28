"""
Phase 4 — Client Logic  (test_client_logic.py)
===============================================
Tests for HospitalClient from client.py with mocked Flower gRPC and
real (but tiny) numpy data produced by the module-level setup.

Strategy:
  • sys.argv is set to include --client_id 0 BEFORE importing client.py
    so argparse.parse_args() succeeds without needing a real CLI invocation.
  • flwr.client.start_numpy_client is patched to a no-op before the import
    so the module-level fl.client.start_numpy_client(...) call doesn't block.
  • Everything else (scaler.pkl, DB, ICUModel) runs normally.
  • Individual tests use monkeypatch to replace train_model / evaluate_model /
    apply_dp_to_update / get_weights / set_weights in the client module namespace.

19 tests across 3 groups:
  4A  get_parameters()             —  2 tests
  4B  fit()                        — 12 tests
  4C  evaluate()                   —  3 tests
  4D  Config constants             —  5 tests (no group header needed — sequential)
"""
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── sys.path fallback ─────────────────────────────────────────────────────────
_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_UNIT_DIR, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Fix sys.argv so argparse succeeds without a real CLI invocation ───────────
# client.py calls parser.parse_args() at import time; pytest's argv would
# cause argparse to error with "required argument --client_id missing".
sys.argv = ["client.py", "--client_id", "0"]

# ── Patch flwr.client.start_numpy_client to prevent actual server connection ──
# client.py calls fl.client.start_numpy_client(...) at module level.
# We replace it with a no-op before importing so the import completes instantly.
import flwr as fl                                    # noqa: E402
fl.client.start_numpy_client = MagicMock()           # no-op; real class kept for NumPyClient

import client as cli                                 # noqa: E402  — must come after patches


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_metrics():
    return {"mse": 0.5, "mae": 0.4, "r2": 0.7}


def _hospital_client():
    """Return a fresh HospitalClient instance."""
    return cli.HospitalClient()


def _real_params():
    """Return the current global-model weights as a list of numpy arrays."""
    from model_utils import get_weights
    return get_weights(cli.model)


# ═════════════════════════════════════════════════════════════════════════════
# 4A — get_parameters()
# ═════════════════════════════════════════════════════════════════════════════

def test_get_parameters_returns_list_of_arrays():
    """get_parameters({}) returns a list of numpy arrays."""
    client = _hospital_client()
    params = client.get_parameters({})
    assert isinstance(params, list)
    assert all(isinstance(w, np.ndarray) for w in params)


def test_get_parameters_length_matches_model():
    """get_parameters({}) list length matches the model's layer count."""
    from model_utils import get_weights
    client = _hospital_client()
    expected = len(get_weights(cli.model))
    assert len(client.get_parameters({})) == expected


# ═════════════════════════════════════════════════════════════════════════════
# 4B — fit()
# ═════════════════════════════════════════════════════════════════════════════

def test_fit_returns_three_element_tuple(monkeypatch):
    """fit() returns a 3-tuple (weights, n_samples, extra_dict)."""
    monkeypatch.setattr(cli, "train_model", MagicMock(return_value=0.5))
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    client = _hospital_client()
    result = client.fit(_real_params(), {"server_round": 1})
    assert isinstance(result, tuple) and len(result) == 3


def test_fit_n_samples_equals_x_train_len(monkeypatch):
    """fit() second return value equals len(X_train)."""
    monkeypatch.setattr(cli, "train_model", MagicMock(return_value=0.5))
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    client = _hospital_client()
    _, n_samples, _ = client.fit(_real_params(), {"server_round": 1})
    assert n_samples == len(cli.X_train)


def test_fit_calls_train_model_once(monkeypatch):
    """fit() invokes train_model exactly once per call."""
    mock_train = MagicMock(return_value=0.5)
    monkeypatch.setattr(cli, "train_model", mock_train)
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    client = _hospital_client()
    client.fit(_real_params(), {"server_round": 1})
    mock_train.assert_called_once()


def test_fit_calls_evaluate_model_once(monkeypatch):
    """fit() invokes evaluate_model exactly once per call."""
    monkeypatch.setattr(cli, "train_model", MagicMock(return_value=0.5))
    mock_eval = MagicMock(return_value=_make_mock_metrics())
    monkeypatch.setattr(cli, "evaluate_model", mock_eval)
    client = _hospital_client()
    client.fit(_real_params(), {"server_round": 1})
    mock_eval.assert_called_once()


def test_fit_lr_round_1(monkeypatch):
    """Round 1 → lr = max(1e-4, BASE_LR * LR_DECAY**0) == BASE_LR."""
    mock_train = MagicMock(return_value=0.5)
    monkeypatch.setattr(cli, "train_model", mock_train)
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    client = _hospital_client()
    client.fit(_real_params(), {"server_round": 1})
    _, kw = mock_train.call_args
    expected = max(1e-4, cli.BASE_LR * (cli.LR_DECAY ** 0))
    assert kw["lr"] == pytest.approx(expected, rel=1e-6)


def test_fit_lr_round_50(monkeypatch):
    """Round 50 → lr = max(1e-4, BASE_LR * LR_DECAY**49)."""
    mock_train = MagicMock(return_value=0.5)
    monkeypatch.setattr(cli, "train_model", mock_train)
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    client = _hospital_client()
    client.fit(_real_params(), {"server_round": 50})
    _, kw = mock_train.call_args
    expected = max(1e-4, cli.BASE_LR * (cli.LR_DECAY ** 49))
    assert kw["lr"] == pytest.approx(expected, rel=1e-6)


def test_fit_lr_clamps_to_floor(monkeypatch):
    """Very high round number → lr is clamped to the 1e-4 floor."""
    mock_train = MagicMock(return_value=0.5)
    monkeypatch.setattr(cli, "train_model", mock_train)
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    client = _hospital_client()
    client.fit(_real_params(), {"server_round": 1_000})
    _, kw = mock_train.call_args
    assert kw["lr"] == pytest.approx(1e-4, rel=1e-6)


def test_fit_dp_disabled_no_dp_call(monkeypatch):
    """When USE_DP=False, apply_dp_to_update is never called."""
    monkeypatch.setattr(cli, "train_model", MagicMock(return_value=0.5))
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    mock_dp = MagicMock()
    monkeypatch.setattr(cli, "apply_dp_to_update", mock_dp)
    monkeypatch.setattr(cli, "USE_DP", False)
    client = _hospital_client()
    client.fit(_real_params(), {"server_round": 1})
    mock_dp.assert_not_called()


def test_fit_dp_enabled_calls_dp(monkeypatch):
    """When USE_DP=True, apply_dp_to_update is called exactly once."""
    monkeypatch.setattr(cli, "train_model", MagicMock(return_value=0.5))
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    # get_weights is called in the DP branch to retrieve local weights
    fake_local = _real_params()
    monkeypatch.setattr(cli, "get_weights", MagicMock(return_value=fake_local))
    mock_dp = MagicMock(return_value=fake_local)
    monkeypatch.setattr(cli, "apply_dp_to_update", mock_dp)
    monkeypatch.setattr(cli, "USE_DP", True)
    client = _hospital_client()
    client.fit(_real_params(), {"server_round": 1})
    mock_dp.assert_called_once()


def test_fit_extra_dict_is_empty(monkeypatch):
    """fit() third return value is always an empty dict."""
    monkeypatch.setattr(cli, "train_model", MagicMock(return_value=0.5))
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    client = _hospital_client()
    _, _, extras = client.fit(_real_params(), {"server_round": 1})
    assert extras == {}


def test_fit_weights_list_is_returned(monkeypatch):
    """fit() first return value is a non-empty list of numpy arrays."""
    monkeypatch.setattr(cli, "train_model", MagicMock(return_value=0.5))
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    client = _hospital_client()
    weights, _, _ = client.fit(_real_params(), {"server_round": 1})
    assert isinstance(weights, list) and len(weights) > 0
    assert all(isinstance(w, np.ndarray) for w in weights)


def test_fit_default_server_round_is_1(monkeypatch):
    """When server_round is absent from config, lr behaves as if round=1."""
    mock_train = MagicMock(return_value=0.5)
    monkeypatch.setattr(cli, "train_model", mock_train)
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    client = _hospital_client()
    client.fit(_real_params(), {})          # no server_round key
    _, kw = mock_train.call_args
    expected = max(1e-4, cli.BASE_LR * (cli.LR_DECAY ** 0))
    assert kw["lr"] == pytest.approx(expected, rel=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# 4C — evaluate()
# ═════════════════════════════════════════════════════════════════════════════

def test_evaluate_returns_triple(monkeypatch):
    """evaluate() returns (mse: float, n_val: int, dict with mae+r2)."""
    monkeypatch.setattr(cli, "evaluate_model", MagicMock(return_value=_make_mock_metrics()))
    client = _hospital_client()
    result = client.evaluate(_real_params(), {})
    assert isinstance(result, tuple) and len(result) == 3
    mse, n_val, metrics = result
    assert isinstance(mse, float)
    assert isinstance(n_val, int)
    assert "mae" in metrics and "r2" in metrics


def test_evaluate_calls_set_weights_before_evaluate_model(monkeypatch):
    """evaluate() calls set_weights before evaluate_model (correct order)."""
    call_order = []

    def _fake_set_weights(model, params):
        call_order.append("set_weights")

    def _fake_eval(model, X, y):
        call_order.append("evaluate_model")
        return _make_mock_metrics()

    monkeypatch.setattr(cli, "set_weights", _fake_set_weights)
    monkeypatch.setattr(cli, "evaluate_model", _fake_eval)
    client = _hospital_client()
    client.evaluate(_real_params(), {})
    assert call_order == ["set_weights", "evaluate_model"]


def test_evaluate_returned_mse_matches_metrics(monkeypatch):
    """evaluate() first return value is exactly metrics['mse']."""
    expected_mse = 2.71828
    monkeypatch.setattr(cli, "evaluate_model",
                        MagicMock(return_value={"mse": expected_mse, "mae": 0.4, "r2": 0.7}))
    client = _hospital_client()
    mse, _, _ = client.evaluate(_real_params(), {})
    assert mse == pytest.approx(expected_mse)


# ═════════════════════════════════════════════════════════════════════════════
# 4D — Config constants
# ═════════════════════════════════════════════════════════════════════════════

def test_mu_fedprox_is_0_5():
    assert cli.MU_FEDPROX == pytest.approx(0.5)


def test_use_dp_default_is_false():
    assert cli.USE_DP is False


def test_epochs_is_3():
    assert cli.EPOCHS == 3


def test_batch_size_is_64():
    assert cli.BATCH_SIZE == 64


def test_server_addr_is_localhost_8080():
    assert cli.SERVER_ADDR == "127.0.0.1:8080"

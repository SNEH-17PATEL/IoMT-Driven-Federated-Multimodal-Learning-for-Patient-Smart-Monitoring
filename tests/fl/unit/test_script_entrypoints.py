"""
Entry-point and edge-case tests for client.py, server.py and model_utils.py.

client.py and server.py do their work at import time (they are scripts, not
libraries), so these tests run them with `runpy` and mock the two calls that would
open a network connection (`fl.client.start_numpy_client`, `fl.server.start_server`).

Safety:
  • server.py WRITES models/training_metadata.json — it is always run from a
    temporary working directory with its own copy of the model folder.
  • client.py only READS (scaler + database), and is run from the project root so
    its relative "models/" path resolves.
"""
import json
import os
import runpy
import shutil
import sys
import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_UNIT_DIR, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import flwr as fl                      # noqa: E402
import db as db_mod                    # noqa: E402
from model_utils import ICUModel, classify_range, train_model   # noqa: E402

CLIENT_PY = os.path.join(_ROOT, "client.py")
SERVER_PY = os.path.join(_ROOT, "server.py")


# ── model_utils edge cases ────────────────────────────────────────────────────

def test_train_model_with_no_data_returns_zero_loss():
    model = ICUModel(10)
    before = [p.detach().clone() for p in model.parameters()]
    loss = train_model(model, np.empty((0, 10), dtype=np.float32), np.empty((0,), dtype=np.float32))
    assert loss == 0.0
    # nothing to learn from, so the weights must be untouched
    assert all((a == b).all() for a, b in zip(before, model.parameters()))


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_classify_range_treats_missing_value_as_normal(bad):
    assert classify_range(bad, 60, 100) == "normal"


def test_classify_range_basic_bands():
    assert classify_range(50, 60, 100) == "low"
    assert classify_range(80, 60, 100) == "normal"
    assert classify_range(120, 60, 100) == "high"


# ── server.py ─────────────────────────────────────────────────────────────────

@pytest.fixture
def server_sandbox(tmp_path, monkeypatch):
    """A throw-away copy of models/ so server.py can write its metadata safely."""
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy(os.path.join(_ROOT, "models", "scaler.pkl"), models / "scaler.pkl")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(fl.server, "start_server", MagicMock())
    return models


def _run_server():
    return runpy.run_path(SERVER_PY, run_name="server_under_test")


def test_server_fit_config_passes_round_number(server_sandbox):
    ns = _run_server()
    assert ns["_fit_config"](1) == {"server_round": 1}
    assert ns["_fit_config"](42) == {"server_round": 42}


def test_server_starts_flower_with_100_rounds_and_3_clients(server_sandbox):
    ns = _run_server()
    kwargs = fl.server.start_server.call_args.kwargs
    assert kwargs["config"].num_rounds == 100
    assert kwargs["server_address"] == "127.0.0.1:8080"
    strat = kwargs["strategy"]
    assert strat.min_fit_clients == 3 and strat.min_available_clients == 3
    assert ns["NUM_ROUNDS"] == 100


def test_server_writes_metadata_into_the_sandbox_only(server_sandbox):
    _run_server()
    meta = json.loads((server_sandbox / "training_metadata.json").read_text())
    assert meta["num_rounds"] == 100
    assert meta["aggregation"] == "FedAvg"
    assert meta["hospitals"] == 3


def test_server_keeps_metrics_from_an_earlier_training_run(server_sandbox):
    (server_sandbox / "training_metadata.json").write_text(
        json.dumps({"final_mae": 1.8006, "final_r2": 0.4806, "split_type": "IID", "ignored": 1}))
    _run_server()
    meta = json.loads((server_sandbox / "training_metadata.json").read_text())
    assert meta["final_mae"] == 1.8006
    assert meta["final_r2"] == 0.4806
    assert meta["split_type"] == "IID"
    assert "ignored" not in meta          # only the known metric keys are carried over


def test_server_survives_a_corrupt_existing_metadata_file(server_sandbox):
    (server_sandbox / "training_metadata.json").write_text("{ this is not json")
    _run_server()                           # must not raise
    meta = json.loads((server_sandbox / "training_metadata.json").read_text())
    assert meta["num_rounds"] == 100        # rewritten cleanly


# ── client.py ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client_env(monkeypatch):
    monkeypatch.chdir(_ROOT)
    monkeypatch.setattr(fl.client, "start_numpy_client", MagicMock())


def _run_client(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["client.py", *argv])
    return runpy.run_path(CLIENT_PY, run_name="client_under_test")


@pytest.mark.parametrize("bad_id", ["3", "5", "-1"])
def test_client_rejects_hospital_id_outside_0_to_2(client_env, monkeypatch, bad_id):
    with pytest.raises(SystemExit) as exc:
        _run_client(monkeypatch, "--client_id", bad_id)
    assert "must be 0, 1, or 2" in str(exc.value)


def test_client_requires_the_client_id_argument(client_env, monkeypatch):
    with pytest.raises(SystemExit):
        _run_client(monkeypatch)


@pytest.mark.parametrize("cid", ["0", "1", "2"])
def test_client_connects_to_the_local_server_for_each_hospital(client_env, monkeypatch, cid):
    ns = _run_client(monkeypatch, "--client_id", cid)
    assert ns["CLIENT_ID"] == int(cid)
    call = fl.client.start_numpy_client.call_args
    assert call.kwargs["server_address"] == "127.0.0.1:8080"


def test_client_zero_fills_feature_columns_missing_from_the_database(client_env, monkeypatch):
    # A table that only has 3 of the model's 108 features plus the label.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "HR_mean": rng.normal(size=60), "RR_mean": rng.normal(size=60),
        "SpO2_mean": rng.normal(size=60),
        "sofa_score": rng.uniform(0, 20, 60),
    })
    monkeypatch.setattr(db_mod, "get_fl_client_data", lambda cid: df.copy())
    with pytest.warns(UserWarning, match="missing from fl_training"):
        ns = _run_client(monkeypatch, "--client_id", "0")
    # every one of the model's input columns is still present after zero-filling
    assert ns["X"].shape[1] == len(ns["feature_columns"]) == 108
    # the columns that were missing are all zeros
    missing_idx = [i for i, c in enumerate(ns["feature_columns"]) if c not in df.columns]
    assert missing_idx and np.all(ns["X"][:, missing_idx] == 0)


# ── setup.py ──────────────────────────────────────────────────────────────────

def test_setup_prints_the_real_dashboard_address():
    """The final lines of setup.py tell the user where to open the dashboard. Vite's default
    port is 5173, so printing any other port sends people to a page that does not exist."""
    src = open(os.path.join(_ROOT, "setup.py"), encoding="utf-8").read()
    assert "http://localhost:5173" in src
    assert "http://localhost:5174" not in src


def test_setup_address_matches_the_vite_and_backend_settings():
    """setup.py, the Playwright config and the backend's allowed origins must agree on 5173."""
    pw = open(os.path.join(_ROOT, "frontend", "playwright.config.js"), encoding="utf-8").read()
    main = open(os.path.join(_ROOT, "backend", "main.py"), encoding="utf-8").read()
    assert "localhost:5173" in pw
    assert "http://localhost:5173" in main

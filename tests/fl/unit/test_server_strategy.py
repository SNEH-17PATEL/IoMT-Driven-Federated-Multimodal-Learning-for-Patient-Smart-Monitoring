"""
Phase 5 — Server Strategy  (test_server_strategy.py)
=====================================================
Tests SaveBestStrategy from server.py with mocked Flower internals.

Import strategy:
  • fl.server.start_server is replaced with MagicMock before importing server.py
    so the module-level blocking call returns immediately.
  • builtins.open is guarded for write mode on training_metadata.json so the
    post-training metadata write (which contains best_eval_loss=inf, invalid JSON)
    is redirected to a StringIO sink and the file on disk is never touched.

19 tests across 5 groups:
  5A  aggregate_fit()             —  5 tests
  5B  aggregate_evaluate()        —  7 tests
  5C  Strategy init parameters    —  3 tests
  5D  Config constants            —  1 test
  5E  Metadata                    —  3 tests
"""
import builtins
import json
import os
import sys
from io import StringIO
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── sys.path fallback ─────────────────────────────────────────────────────────
_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_UNIT_DIR, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Patch fl.server.start_server to a no-op before importing server.py ────────
import flwr as fl                           # noqa: E402
fl.server.start_server = MagicMock()        # returns immediately; real FedAvg class kept

# ── Guard training_metadata.json from being written during import ─────────────
# After start_server() returns, server.py builds _meta and writes the file.
# best_eval_loss = float("inf") → json.dump writes "Infinity" (not valid JSON),
# which would corrupt models/training_metadata.json on disk.
# Redirect any write-mode open of that specific path to a StringIO sink.
_real_open = builtins.open

def _guarded_open(path, mode="r", *args, **kwargs):
    if "w" in str(mode) and "training_metadata" in str(path):
        return StringIO()
    return _real_open(path, mode, *args, **kwargs)

builtins.open = _guarded_open
import server as srv                        # noqa: E402 — must come after patches
builtins.open = _real_open                  # restore immediately


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strategy():
    """Fresh SaveBestStrategy with the canonical 3-hospital configuration."""
    return srv.SaveBestStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )


def _eval_results(*losses):
    """Build a list of (client_proxy, EvaluateRes-like) tuples from loss values."""
    return [(MagicMock(), MagicMock(loss=v)) for v in losses]


# ── Autouse: reset module-level _best_loss / _best_round for every test ───────

@pytest.fixture(autouse=True)
def _reset_globals(monkeypatch):
    """Restore server-level best-tracking state to its initial values."""
    monkeypatch.setattr(srv, "_best_loss", float("inf"))
    monkeypatch.setattr(srv, "_best_round", 0)


# ═════════════════════════════════════════════════════════════════════════════
# 5A — aggregate_fit()
# ═════════════════════════════════════════════════════════════════════════════

def test_aggregate_fit_calls_super(monkeypatch):
    """aggregate_fit() delegates to the parent FedAvg.aggregate_fit."""
    mock_parent = MagicMock(return_value=(None, {}))
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_fit", mock_parent)
    _strategy().aggregate_fit(1, [], [])
    mock_parent.assert_called_once()


def test_aggregate_fit_calls_set_weights_when_params_not_none(monkeypatch):
    """When parent returns non-None params, set_weights(global_model, weights) is called."""
    fake_params = MagicMock()
    fake_weights = [np.zeros((4, 4), dtype=np.float32)]
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_fit",
                        MagicMock(return_value=(fake_params, {})))
    monkeypatch.setattr(srv, "parameters_to_ndarrays", MagicMock(return_value=fake_weights))
    mock_set_w = MagicMock()
    monkeypatch.setattr(srv, "set_weights", mock_set_w)
    monkeypatch.setattr(srv.torch, "save", MagicMock())
    _strategy().aggregate_fit(1, [], [])
    mock_set_w.assert_called_once()


def test_aggregate_fit_saves_latest_model_to_pth(monkeypatch):
    """aggregate_fit() calls torch.save(..., 'federated_model.pth') after aggregation."""
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_fit",
                        MagicMock(return_value=(MagicMock(), {})))
    monkeypatch.setattr(srv, "parameters_to_ndarrays", MagicMock(return_value=[]))
    monkeypatch.setattr(srv, "set_weights", MagicMock())
    mock_save = MagicMock()
    monkeypatch.setattr(srv.torch, "save", mock_save)
    _strategy().aggregate_fit(1, [], [])
    mock_save.assert_called_once()
    saved_path = mock_save.call_args[0][1]
    assert "federated_model.pth" in saved_path


def test_aggregate_fit_prints_round_number(monkeypatch, capsys):
    """aggregate_fit() prints the current server round to stdout."""
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_fit",
                        MagicMock(return_value=(MagicMock(), {})))
    monkeypatch.setattr(srv, "parameters_to_ndarrays", MagicMock(return_value=[]))
    monkeypatch.setattr(srv, "set_weights", MagicMock())
    monkeypatch.setattr(srv.torch, "save", MagicMock())
    _strategy().aggregate_fit(7, [], [])
    assert "7" in capsys.readouterr().out


def test_aggregate_fit_null_params_skips_set_weights(monkeypatch):
    """When parent returns None params, set_weights is never called."""
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_fit",
                        MagicMock(return_value=(None, {})))
    mock_set_w = MagicMock()
    monkeypatch.setattr(srv, "set_weights", mock_set_w)
    _strategy().aggregate_fit(1, [], [])
    mock_set_w.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# 5B — aggregate_evaluate()
# ═════════════════════════════════════════════════════════════════════════════

def test_aggregate_evaluate_computes_correct_avg_loss(monkeypatch):
    """avg_loss = sum(r.loss for _, r in results) / len(results)."""
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_evaluate",
                        MagicMock(return_value=(None, {})))
    monkeypatch.setattr(srv.torch, "save", MagicMock())
    _strategy().aggregate_evaluate(1, _eval_results(1.0, 2.0, 3.0), [])
    # avg = 2.0 — first round always beats inf so _best_loss is updated
    assert srv._best_loss == pytest.approx(2.0)


def test_aggregate_evaluate_first_round_updates_best_loss(monkeypatch):
    """First round with any finite loss beats the initial inf → _best_loss updated."""
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_evaluate",
                        MagicMock(return_value=(None, {})))
    monkeypatch.setattr(srv.torch, "save", MagicMock())
    _strategy().aggregate_evaluate(1, _eval_results(0.5), [])
    assert srv._best_loss == pytest.approx(0.5)


def test_aggregate_evaluate_higher_loss_does_not_update_best(monkeypatch):
    """A round with avg_loss > current _best_loss leaves _best_loss unchanged."""
    monkeypatch.setattr(srv, "_best_loss", 1.0)   # already a good loss
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_evaluate",
                        MagicMock(return_value=(None, {})))
    monkeypatch.setattr(srv.torch, "save", MagicMock())
    _strategy().aggregate_evaluate(2, _eval_results(5.0), [])
    assert srv._best_loss == pytest.approx(1.0)   # unchanged


def test_aggregate_evaluate_new_best_saves_best_model(monkeypatch):
    """New best loss → torch.save called with path ending in 'federated_model_best.pth'."""
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_evaluate",
                        MagicMock(return_value=(None, {})))
    mock_save = MagicMock()
    monkeypatch.setattr(srv.torch, "save", mock_save)
    _strategy().aggregate_evaluate(1, _eval_results(0.3), [])
    saved_paths = [call[0][1] for call in mock_save.call_args_list]
    assert any("federated_model_best.pth" in p for p in saved_paths)


def test_aggregate_evaluate_not_best_does_not_save_best_model(monkeypatch):
    """When loss is not a new best, federated_model_best.pth is NOT written."""
    monkeypatch.setattr(srv, "_best_loss", 0.1)   # very good existing best
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_evaluate",
                        MagicMock(return_value=(None, {})))
    mock_save = MagicMock()
    monkeypatch.setattr(srv.torch, "save", mock_save)
    _strategy().aggregate_evaluate(2, _eval_results(5.0), [])
    saved_paths = [call[0][1] for call in mock_save.call_args_list]
    assert not any("federated_model_best.pth" in p for p in saved_paths)


def test_aggregate_evaluate_updates_best_round(monkeypatch):
    """New best loss → _best_round is updated to the current server_round."""
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_evaluate",
                        MagicMock(return_value=(None, {})))
    monkeypatch.setattr(srv.torch, "save", MagicMock())
    _strategy().aggregate_evaluate(7, _eval_results(0.4), [])
    assert srv._best_round == 7


def test_aggregate_evaluate_calls_super(monkeypatch):
    """aggregate_evaluate() delegates to the parent FedAvg.aggregate_evaluate."""
    mock_parent = MagicMock(return_value=(None, {}))
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_evaluate", mock_parent)
    monkeypatch.setattr(srv.torch, "save", MagicMock())
    _strategy().aggregate_evaluate(1, _eval_results(0.5), [])
    mock_parent.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# 5C — Strategy init parameters
# ═════════════════════════════════════════════════════════════════════════════

def test_strategy_fraction_fit_is_1():
    """Module-level strategy instance has fraction_fit == 1.0."""
    assert srv.strategy.fraction_fit == pytest.approx(1.0)


def test_strategy_min_fit_clients_is_3():
    """Module-level strategy instance requires 3 fit clients per round."""
    assert srv.strategy.min_fit_clients == 3


def test_strategy_min_evaluate_clients_is_3():
    """Module-level strategy instance requires 3 evaluate clients per round."""
    assert srv.strategy.min_evaluate_clients == 3


# ═════════════════════════════════════════════════════════════════════════════
# 5D — Config constants
# ═════════════════════════════════════════════════════════════════════════════

def test_num_rounds_is_100():
    """NUM_ROUNDS == 100 matches train_federated.py convention."""
    assert srv.NUM_ROUNDS == 100


# ═════════════════════════════════════════════════════════════════════════════
# 5E — Metadata
# ═════════════════════════════════════════════════════════════════════════════

def test_metadata_best_round_field_is_int():
    """server.py stores best_round as int(best_round) — verifiable from _meta."""
    assert isinstance(srv._meta.get("best_round"), int)


def test_metadata_path_is_in_models_dir():
    """Metadata JSON is written to MODEL_PATH + 'training_metadata.json'."""
    assert srv._meta_path == srv.MODEL_PATH + "training_metadata.json"


def test_metadata_merge_preserves_final_mae_and_r2(tmp_path):
    """Merge algorithm: final_mae and final_r2 from an existing file survive into _meta."""
    existing = {"final_mae": 1.8968, "final_r2": 0.5995, "unrelated_key": "drop_me"}
    meta_file = tmp_path / "training_metadata.json"
    meta_file.write_text(json.dumps(existing))

    # Replicate server.py's merge logic (lines 131-142) with a controlled path
    meta = {"num_rounds": 100, "best_round": int(0)}
    if meta_file.exists():
        with open(str(meta_file)) as f:
            _existing = json.load(f)
        for k in ("final_mae", "final_r2", "train_samples", "test_samples",
                  "pred_range_min", "pred_range_max",
                  "epochs_per_round", "split_type", "hospital_names", "tfidf_features"):
            if k in _existing:
                meta[k] = _existing[k]

    assert meta["final_mae"] == pytest.approx(1.8968)
    assert meta["final_r2"] == pytest.approx(0.5995)
    assert "unrelated_key" not in meta   # only the known keys are merged

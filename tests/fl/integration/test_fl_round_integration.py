"""
Phase 8 — FL Round Integration  (test_fl_round_integration.py)
==============================================================
Simulates a full FL round (distribute → train → aggregate) without Flower
gRPC by calling fl_dashboard helpers (_fedavg, _fedyogi, fl_thread) with
synthetic data and an isolated module state.

All tests are marked @pytest.mark.slow because each runs real torch training.

Import strategy:
  fl_dashboard.py uses @st.cache_resource at module level.  When collected
  after test_fl_dashboard_fns.py (unit tests), fl_dashboard is already in
  sys.modules.  When run alone, we mock streamlit first.  The conditional
  guard handles both cases without double-importing.

14 tests across 4 groups:
  8A  Manual training + aggregation (no fl_thread)  —  5 tests
  8B  fl_thread state tracking                       —  7 tests
  8C  FedProx vs FedAvg                              —  1 test
  8D  Histogram / heatmap data shapes                —  1 test
"""
import os
import sys
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── sys.path fallback ─────────────────────────────────────────────────────────
_INTEG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_INTEG_DIR, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Import fl_dashboard with mocked Streamlit ─────────────────────────────────
# If the unit tests already imported fl_dashboard (mocked streamlit in place),
# we reuse the cached module.  Otherwise, set up the mock now.
if "fl_dashboard" not in sys.modules:
    _st_mock = MagicMock()
    _st_mock.cache_resource = lambda f: f
    _st_mock.tabs.side_effect = (
        lambda tab_list: [MagicMock() for _ in tab_list]
    )
    _st_mock.columns.side_effect = (
        lambda spec, **kw: [MagicMock() for _ in range(spec)]
        if isinstance(spec, int) else [MagicMock() for _ in spec]
    )
    sys.modules["streamlit"]               = _st_mock
    sys.modules["streamlit.components"]    = MagicMock()
    sys.modules["streamlit.components.v1"] = MagicMock()
    sys.modules.setdefault("joblib", MagicMock())

import fl_dashboard as fld                  # noqa: E402 — after mock setup

from model_utils import (                   # noqa: E402
    ICUModel, train_model, get_weights, set_weights,
)


# ── Constants ─────────────────────────────────────────────────────────────────

INPUT_DIM = 10      # small network for speed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_clients(n_train=100, n_val=25, seed=0):
    """3 synthetic hospital clients with random SOFA targets."""
    rng = np.random.default_rng(seed)
    clients = []
    for _ in range(3):
        X = rng.standard_normal((n_train + n_val, INPUT_DIM)).astype(np.float32)
        y = rng.uniform(0, 24, n_train + n_val).astype(np.float32)
        clients.append({
            "X_train": X[:n_train], "y_train": y[:n_train],
            "X_val":   X[n_train:], "y_val":   y[n_train:],
            "n_train": n_train,
        })
    return clients


def _make_clients_structured(n_train=150, n_val=30, seed=1):
    """Clients with a clear linear signal for reliable convergence tests."""
    rng = np.random.default_rng(seed)
    clients = []
    for _ in range(3):
        X = rng.standard_normal((n_train + n_val, INPUT_DIM)).astype(np.float32)
        # y has a strong linear dependency on X[:,0] → model can learn it
        y = np.clip(X[:, 0] * 10 + 12, 0, 24).astype(np.float32)
        clients.append({
            "X_train": X[:n_train], "y_train": y[:n_train],
            "X_val":   X[n_train:], "y_val":   y[n_train:],
            "n_train": n_train,
        })
    return clients


# ── Autouse: reset fl_dashboard shared state before every test ────────────────

@pytest.fixture(autouse=True)
def _reset_fl_state():
    """Isolate _STATE and _STOP between tests."""
    fld._reset_state(1, "FedAvg")
    fld._STOP.clear()
    yield
    fld._STOP.clear()


# ═════════════════════════════════════════════════════════════════════════════
# 8A — Manual training + aggregation (without fl_thread)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_three_clients_produce_three_weight_lists():
    """Each of the 3 hospitals trains independently and returns a weight list."""
    global_w = get_weights(ICUModel(INPUT_DIM))
    clients  = _make_clients()

    c_weights = []
    for c in clients:
        model = ICUModel(INPUT_DIM)
        set_weights(model, global_w)
        train_model(model, c["X_train"], c["y_train"],
                    epochs=2, lr=0.001, oversample=False)
        c_weights.append(get_weights(model))

    assert len(c_weights) == 3
    for w in c_weights:
        assert isinstance(w, list)
        assert all(isinstance(a, np.ndarray) for a in w)


@pytest.mark.slow
def test_fedavg_aggregated_shapes_match_global():
    """`_fedavg` returns one array per layer with the same shape as the global model."""
    global_w  = get_weights(ICUModel(INPUT_DIM))
    clients   = _make_clients()
    c_weights = []
    c_samples = []
    for c in clients:
        model = ICUModel(INPUT_DIM)
        set_weights(model, global_w)
        train_model(model, c["X_train"], c["y_train"],
                    epochs=1, lr=0.001, oversample=False)
        c_weights.append(get_weights(model))
        c_samples.append(c["n_train"])

    agg = fld._fedavg(c_weights, c_samples)
    assert len(agg) == len(global_w)
    for orig, agg_layer in zip(global_w, agg):
        assert orig.shape == agg_layer.shape


@pytest.mark.slow
def test_post_fedavg_model_forward_pass_is_valid():
    """After FedAvg aggregation, setting the global model and running a forward
    pass produces a finite tensor of shape (n, 1)."""
    import torch

    global_w  = get_weights(ICUModel(INPUT_DIM))
    clients   = _make_clients()
    c_weights = []
    c_samples = []
    for c in clients:
        model = ICUModel(INPUT_DIM)
        set_weights(model, global_w)
        train_model(model, c["X_train"], c["y_train"],
                    epochs=1, lr=0.001, oversample=False)
        c_weights.append(get_weights(model))
        c_samples.append(c["n_train"])

    agg_w = fld._fedavg(c_weights, c_samples)
    eval_model = ICUModel(INPUT_DIM)
    set_weights(eval_model, agg_w)
    eval_model.eval()

    X_test = torch.randn(5, INPUT_DIM)
    with torch.no_grad():
        out = eval_model(X_test)

    assert out.shape == (5, 1)
    assert torch.isfinite(out).all()


@pytest.mark.slow
def test_fedyogi_variance_floored_at_tau_squared():
    """`_fedyogi` guarantees v_t ≥ tau² for numerical stability."""
    tau      = 1e-3
    global_w = get_weights(ICUModel(INPUT_DIM))
    clients  = _make_clients()
    c_weights, c_samples = [], []
    for c in clients:
        model = ICUModel(INPUT_DIM)
        set_weights(model, global_w)
        train_model(model, c["X_train"], c["y_train"],
                    epochs=1, lr=0.001, oversample=False)
        c_weights.append(get_weights(model))
        c_samples.append(c["n_train"])

    new_w, new_m, new_v = fld._fedyogi(
        c_weights, c_samples, global_w, None, None
    )
    for v in new_v:
        assert (v >= tau ** 2 - 1e-10).all(), "v_t violated tau² floor"
    for w in new_w:
        assert np.isfinite(w).all(), "FedYogi produced non-finite weights"


@pytest.mark.slow
def test_round2_global_loss_lower_than_round1():
    """Global training loss decreases from round 1 to round 2 (FedAvg on
    structured data with a learnable signal)."""
    clients = _make_clients_structured()

    def _one_round(global_w):
        c_weights, c_losses, c_samples = [], [], []
        for c in clients:
            model = ICUModel(INPUT_DIM)
            set_weights(model, global_w)
            loss = train_model(model, c["X_train"], c["y_train"],
                               epochs=8, lr=0.005, oversample=False)
            c_weights.append(get_weights(model))
            c_losses.append(loss)
            c_samples.append(c["n_train"])
        agg_w = fld._fedavg(c_weights, c_samples)
        total = sum(c_samples)
        g_loss = sum(l * n / total for l, n in zip(c_losses, c_samples))
        return agg_w, g_loss

    init_w = get_weights(ICUModel(INPUT_DIM))
    w1, loss1 = _one_round(init_w)
    _, loss2   = _one_round(w1)

    assert loss2 < loss1, (
        f"Expected round-2 loss ({loss2:.4f}) < round-1 loss ({loss1:.4f})"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 8B — fl_thread state tracking
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_fl_thread_status_done_after_completion():
    """fl_thread sets status='done' and current_phase='done' when it finishes."""
    clients = _make_clients()
    fld._STATE["status"] = "running"
    fld.fl_thread(1, 0.0, "FedAvg", clients, INPUT_DIM)
    assert fld._STATE["status"]       == "done"
    assert fld._STATE["current_phase"] == "done"


@pytest.mark.slow
def test_state_current_round_is_1_after_one_round():
    """`current_round` is 1 in _read_state() after a single-round simulation."""
    clients = _make_clients()
    fld.fl_thread(1, 0.0, "FedAvg", clients, INPUT_DIM)
    assert fld._read_state()["current_round"] == 1


@pytest.mark.slow
def test_state_global_losses_length_equals_rounds():
    """`global_losses` has exactly 1 entry after a 1-round simulation."""
    clients = _make_clients()
    fld.fl_thread(1, 0.0, "FedAvg", clients, INPUT_DIM)
    assert len(fld._read_state()["global_losses"]) == 1


@pytest.mark.slow
def test_state_client_losses_one_entry_per_client():
    """Each of the 3 client loss lists has 1 entry after a 1-round simulation."""
    clients = _make_clients()
    fld.fl_thread(1, 0.0, "FedAvg", clients, INPUT_DIM)
    state = fld._read_state()
    assert len(state["client_losses"]) == 3
    for cid_losses in state["client_losses"]:
        assert len(cid_losses) == 1


@pytest.mark.slow
def test_stop_event_before_round_yields_stopped():
    """`_STOP.set()` before calling fl_thread produces status='stopped'."""
    clients = _make_clients()
    fld._STOP.set()
    fld.fl_thread(1, 0.0, "FedAvg", clients, INPUT_DIM)
    assert fld._STATE["status"] == "stopped"


@pytest.mark.slow
def test_best_round_is_1_after_single_round():
    """`best_round` is 1 — the only round is trivially the best."""
    clients = _make_clients()
    fld.fl_thread(1, 0.0, "FedAvg", clients, INPUT_DIM)
    assert fld._read_state()["best_round"] == 1


@pytest.mark.slow
def test_round_log_has_one_entry_with_correct_keys():
    """`round_log` has 1 entry containing Round, H0/H1/H2 Loss, and Global keys."""
    clients = _make_clients()
    fld.fl_thread(1, 0.0, "FedAvg", clients, INPUT_DIM)
    state = fld._read_state()
    assert len(state["round_log"]) == 1
    entry = state["round_log"][0]
    assert entry["Round"] == 1
    for key in ("H0 Loss", "H1 Loss", "H2 Loss", "Global"):
        assert key in entry


# ═════════════════════════════════════════════════════════════════════════════
# 8C — FedProx vs FedAvg
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_fedprox_produces_smaller_weight_delta_than_fedavg():
    """With mu=0.5, FedProx keeps client weights closer to the global model than
    plain FedAvg (mu=0), measurable as a smaller average L1 weight delta."""
    clients  = _make_clients(n_train=100, seed=42)
    global_w = get_weights(ICUModel(INPUT_DIM))
    global_flat = np.concatenate([w.flatten() for w in global_w])

    def _mean_delta(use_prox):
        mu = 0.5 if use_prox else 0.0
        deltas = []
        for c in clients:
            model = ICUModel(INPUT_DIM)
            set_weights(model, global_w)
            train_model(
                model, c["X_train"], c["y_train"],
                epochs=10, lr=0.01, oversample=False,
                global_params=global_w if use_prox else None,
                mu=mu,
            )
            local_flat = np.concatenate([w.flatten() for w in get_weights(model)])
            deltas.append(float(np.mean(np.abs(local_flat - global_flat))))
        return float(np.mean(deltas))

    delta_fedavg  = _mean_delta(use_prox=False)
    delta_fedprox = _mean_delta(use_prox=True)

    assert delta_fedprox < delta_fedavg, (
        f"FedProx delta ({delta_fedprox:.6f}) ≥ FedAvg delta ({delta_fedavg:.6f})"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 8D — Histogram / heatmap data shapes
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_pre_post_agg_flat_and_layer1_weights_shapes():
    """After 1 round of fl_thread:
      • pre_agg_flat and post_agg_flat are 1-D arrays of equal length.
      • layer1_weights has shape (128, INPUT_DIM) — the first layer matrix.
    """
    clients = _make_clients()
    fld.fl_thread(1, 0.0, "FedAvg", clients, INPUT_DIM)
    state = fld._read_state()

    pre  = state["pre_agg_flat"]
    post = state["post_agg_flat"]
    l1   = state["layer1_weights"]

    assert pre.ndim  == 1
    assert post.ndim == 1
    assert len(pre)  == len(post)
    assert l1.shape  == (128, INPUT_DIM)

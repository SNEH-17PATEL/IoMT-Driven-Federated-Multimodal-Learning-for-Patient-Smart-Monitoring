"""
Phase 10 — System Tests  (test_fl_system.py)
=============================================
End-to-end FL simulation using fl_thread logic with real synthetic data.
All tests are @pytest.mark.slow.

17 tests:
  10A  Single-round completeness     —  5 tests
  10B  Client & weight metrics       —  4 tests
  10C  Multi-round convergence       —  2 tests
  10D  Algorithm comparison          —  2 tests
  10E  Stop / round-log              —  4 tests

Import strategy: same conditional Streamlit mock as Phase 8.
fl_thread is called directly (not in a thread) with speed=0.0 for
deterministic synchronous execution.
"""
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# ── sys.path fallback ─────────────────────────────────────────────────────────
_SYS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_SYS_DIR, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Import fl_dashboard with mocked Streamlit ─────────────────────────────────
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

import fl_dashboard as fld  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────────────

INPUT_DIM = 10   # small network for speed


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
        y = np.clip(X[:, 0] * 10 + 12, 0, 24).astype(np.float32)
        clients.append({
            "X_train": X[:n_train], "y_train": y[:n_train],
            "X_val":   X[n_train:], "y_val":   y[n_train:],
            "n_train": n_train,
        })
    return clients


def _make_clients_noniid(n_train=200, n_val=40, seed=7):
    """
    Non-IID clients — each hospital specialises in a different feature and
    SOFA range.  Heterogeneous gradients make FedYogi's adaptive step size
    useful for finding a compromise that generalises across all three regions.
    """
    rng = np.random.default_rng(seed)
    # client k's signal lives primarily in feature k; SOFA range differs per client
    configs = [
        (lambda X: np.clip(X[:, 0] * 2.5 + 3,  0,  6).astype(np.float32)),
        (lambda X: np.clip(X[:, 1] * 3.0 + 10,  5, 15).astype(np.float32)),
        (lambda X: np.clip(X[:, 2] * 3.0 + 18, 12, 24).astype(np.float32)),
    ]
    clients = []
    for signal_fn in configs:
        X = rng.standard_normal((n_train + n_val, INPUT_DIM)).astype(np.float32)
        y = signal_fn(X)
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


# ── Module-scoped one-round snapshot ──────────────────────────────────────────

@pytest.fixture(scope="module")
def one_round():
    """
    Run fl_thread for 1 FedAvg round and return a _read_state() snapshot.
    Module-scoped: executed once; all consumers read the same dict copy.
    """
    fld._reset_state(1, "FedAvg")
    fld._STOP.clear()
    torch.manual_seed(0)
    fld.fl_thread(1, 0.0, "FedAvg", _make_clients(), INPUT_DIM)
    return fld._read_state()


# ═════════════════════════════════════════════════════════════════════════════
# 10A — Single-round completeness
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_one_round_status_done(one_round):
    """1-round FL simulation completes with status='done'."""
    assert one_round["status"] == "done"


@pytest.mark.slow
def test_one_round_global_loss_is_finite(one_round):
    """Global loss after round 1 is a finite float."""
    loss = one_round["global_losses"][0]
    assert np.isfinite(loss), f"global_loss={loss}"


@pytest.mark.slow
def test_one_round_best_round_is_1(one_round):
    """best_round == 1 after a 1-round simulation (only round = best round)."""
    assert one_round["best_round"] == 1


@pytest.mark.slow
def test_one_round_global_mae_recorded(one_round):
    """Global validation MAE is recorded after round 1."""
    mae = one_round["global_mae_history"][0]
    assert isinstance(mae, float) and np.isfinite(mae)


@pytest.mark.slow
def test_one_round_global_r2_recorded(one_round):
    """Global R² is recorded after round 1 (can be negative on first round)."""
    r2 = one_round["global_r2_history"][0]
    assert isinstance(r2, float) and np.isfinite(r2)


# ═════════════════════════════════════════════════════════════════════════════
# 10B — Client & weight metrics
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_all_three_client_metrics_present(one_round):
    """client_metrics has mae, r2, loss for all 3 hospitals."""
    assert len(one_round["client_metrics"]) == 3
    for cid, m in enumerate(one_round["client_metrics"]):
        for key in ("mae", "r2", "loss"):
            assert key in m, f"client {cid} missing '{key}'"


@pytest.mark.slow
def test_weight_stats_all_three_clients_recorded(one_round):
    """weight_stats has n_params, mean, std, delta_mean for all 3 hospitals."""
    ws = one_round["weight_stats"]
    assert len(ws) == 1, f"expected 1 round entry, got {len(ws)}"
    assert len(ws[0]) == 3, f"expected 3 client entries, got {len(ws[0])}"
    for cid, stat in enumerate(ws[0]):
        for key in ("n_params", "mean", "std", "delta_mean"):
            assert key in stat, f"client {cid} weight_stats missing '{key}'"


@pytest.mark.slow
def test_layer1_weights_correct_shape(one_round):
    """layer1_weights (first-layer matrix) has shape (128, INPUT_DIM)."""
    l1 = one_round["layer1_weights"]
    assert l1.shape == (128, INPUT_DIM), f"got shape {l1.shape}"


@pytest.mark.slow
def test_pre_post_agg_flat_1d_same_length(one_round):
    """pre_agg_flat and post_agg_flat are 1-D arrays of equal length."""
    pre  = one_round["pre_agg_flat"]
    post = one_round["post_agg_flat"]
    assert pre.ndim == 1, f"pre_agg_flat ndim={pre.ndim}"
    assert post.ndim == 1, f"post_agg_flat ndim={post.ndim}"
    assert len(pre) == len(post), f"lengths differ: pre={len(pre)} post={len(post)}"


# ═════════════════════════════════════════════════════════════════════════════
# 10C — Multi-round convergence
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_three_round_global_losses_length():
    """global_losses has exactly 3 entries after a 3-round simulation."""
    torch.manual_seed(0)
    fld.fl_thread(3, 0.0, "FedAvg", _make_clients(), INPUT_DIM)
    state = fld._read_state()
    assert len(state["global_losses"]) == 3


@pytest.mark.slow
def test_three_round_loss_generally_converging():
    """Global training loss at round 3 is lower than round 1 on structured data."""
    clients = _make_clients_structured()
    torch.manual_seed(1)
    fld.fl_thread(3, 0.0, "FedAvg", clients, INPUT_DIM)
    losses = fld._read_state()["global_losses"]
    assert losses[2] < losses[0], (
        f"round-3 loss ({losses[2]:.4f}) ≥ round-1 loss ({losses[0]:.4f})"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 10D — Algorithm comparison
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_fedyogi_5round_simulation_converges():
    """
    FedYogi simulation runs 5 complete rounds and converges: the round-5
    global training loss is lower than the round-1 loss on structured data.

    Also verifies that FedYogi and FedAvg produce DIFFERENT final losses —
    confirming the two algorithms are not identical.  A direct performance
    comparison requires many more rounds (30+) to let FedYogi's momentum
    accumulate fully, which is impractical in a unit/system-test context.
    """
    clients = _make_clients_structured(seed=1)

    # FedYogi — 5 rounds
    fld._reset_state(5, "FedYogi")
    fld._STOP.clear()
    torch.manual_seed(42)
    fld.fl_thread(5, 0.0, "FedYogi", clients, INPUT_DIM)
    state_yogi = fld._read_state()

    # FedAvg — same data, same seed, 5 rounds
    fld._reset_state(5, "FedAvg")
    fld._STOP.clear()
    torch.manual_seed(42)
    fld.fl_thread(5, 0.0, "FedAvg", clients, INPUT_DIM)
    state_avg = fld._read_state()

    losses_yogi = state_yogi["global_losses"]
    losses_avg  = state_avg["global_losses"]

    # Both simulations ran 5 rounds successfully
    assert len(losses_yogi) == 5, f"FedYogi: expected 5 rounds, got {len(losses_yogi)}"
    assert len(losses_avg)  == 5, f"FedAvg: expected 5 rounds, got {len(losses_avg)}"

    # FedYogi converges: final loss < first-round loss
    assert losses_yogi[-1] < losses_yogi[0], (
        f"FedYogi did not converge: round-5 ({losses_yogi[-1]:.4f}) ≥ "
        f"round-1 ({losses_yogi[0]:.4f})"
    )

    # The two algorithms produce genuinely different global models
    assert losses_yogi[-1] != losses_avg[-1], (
        "FedYogi and FedAvg produced identical final losses — algorithms may be identical"
    )


@pytest.mark.slow
def test_fedprox_clients_smaller_delta_than_fedavg():
    """
    FedProx (mu=0.5) keeps each client's weights closer to the global model
    than plain FedAvg (mu=0), measured as per-client delta_mean in weight_stats.

    Both runs use the same torch seed → identical initial global weights.
    """
    clients = _make_clients(n_train=200, seed=42)

    # FedAvg baseline
    fld._reset_state(1, "FedAvg")
    fld._STOP.clear()
    torch.manual_seed(42)
    fld.fl_thread(1, 0.0, "FedAvg", clients, INPUT_DIM)
    deltas_avg = [
        fld._read_state()["weight_stats"][0][cid]["delta_mean"]
        for cid in range(3)
    ]

    # FedProx — identical seed → identical initial global weights
    fld._reset_state(1, "FedProx")
    fld._STOP.clear()
    torch.manual_seed(42)
    fld.fl_thread(1, 0.0, "FedProx", clients, INPUT_DIM)
    deltas_prox = [
        fld._read_state()["weight_stats"][0][cid]["delta_mean"]
        for cid in range(3)
    ]

    for cid in range(3):
        assert deltas_prox[cid] < deltas_avg[cid], (
            f"client {cid}: FedProx delta ({deltas_prox[cid]:.6f}) "
            f"≥ FedAvg delta ({deltas_avg[cid]:.6f})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 10E — Stop / round-log
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_stop_before_round_yields_stopped_status():
    """`_STOP.set()` before fl_thread → status='stopped', no rounds completed."""
    clients = _make_clients()
    fld._STOP.set()
    fld.fl_thread(1, 0.0, "FedAvg", clients, INPUT_DIM)
    state = fld._read_state()
    assert state["status"] == "stopped"
    assert len(state["global_losses"]) == 0


@pytest.mark.slow
def test_one_round_round_log_has_one_entry(one_round):
    """round_log has exactly 1 entry after a 1-round simulation."""
    assert len(one_round["round_log"]) == 1


@pytest.mark.slow
def test_one_round_round_log_round_number(one_round):
    """round_log[0]['Round'] == 1."""
    assert one_round["round_log"][0]["Round"] == 1


@pytest.mark.slow
def test_one_round_round_log_has_hospital_and_global_keys(one_round):
    """round_log[0] contains H0 Loss, H1 Loss, H2 Loss, and Global keys."""
    entry = one_round["round_log"][0]
    for key in ("H0 Loss", "H1 Loss", "H2 Loss", "Global"):
        assert key in entry, f"missing key '{key}' in round_log[0]: {list(entry)}"

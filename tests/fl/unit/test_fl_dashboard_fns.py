"""
Phase 3 — FL Dashboard Pure Functions  (test_fl_dashboard_fns.py)
=================================================================
Tests every extractable pure function from fl_dashboard.py WITHOUT
launching Streamlit.  Strategy: mock streamlit and @st.cache_resource
at module level BEFORE importing fl_dashboard so the module-level
st.* calls are silently absorbed.

50 tests across 6 groups:
  3A  State management             — 11 tests
  3B  _fedavg()                    —  6 tests
  3C  _fedyogi()                   —  8 tests
  3D  load_training_metadata()     —  7 tests
  3E  Chart builders               —  9 tests
  3F  _network_diagram()           —  9 tests
"""
import json
import math
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import numpy.testing as npt
import pytest

# ── sys.path fallback ─────────────────────────────────────────────────────────
_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.normpath(os.path.join(_UNIT_DIR, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Mock streamlit BEFORE importing fl_dashboard ──────────────────────────────
# Module-level st.set_page_config / st.markdown / @st.cache_resource calls
# must be absorbed silently so the module imports without a running Streamlit
# server.  cache_resource is mocked as a no-op decorator so _shared() returns
# a plain dict with real threading.Lock / Event.
#
# st.tabs / st.columns return proper-length lists so that tuple-unpacking at
# module level (e.g. tab_sim, tab_info = st.tabs([...])) doesn't raise
# ValueError.
_st_mock = MagicMock()
_st_mock.cache_resource = lambda f: f          # no-op: return function unchanged
_st_mock.tabs.side_effect = (
    lambda tab_list: [MagicMock() for _ in tab_list]
)
_st_mock.columns.side_effect = (
    lambda spec, **kw: [MagicMock() for _ in range(spec)]
    if isinstance(spec, int)
    else [MagicMock() for _ in spec]
)
sys.modules["streamlit"]              = _st_mock
sys.modules["streamlit.components"]   = MagicMock()
sys.modules["streamlit.components.v1"] = MagicMock()
sys.modules.setdefault("joblib", MagicMock())  # load_client_data uses joblib

import fl_dashboard as fld                     # noqa: E402 — must come after mocks


# ── Autouse fixture: reset shared state before every test ─────────────────────

@pytest.fixture(autouse=True)
def _reset():
    """Restore _STATE to _blank_state() before and after every test."""
    fld._reset_state(10, "FedYogi + FedProx")
    yield
    fld._reset_state(10, "FedYogi + FedProx")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simple_weights(n_layers=2, shape=(4, 4)):
    """Return a list of n_layers float32 numpy arrays of `shape`."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal(shape).astype(np.float32) for _ in range(n_layers)]


# ═════════════════════════════════════════════════════════════════════════════
# 3A — State Management
# ═════════════════════════════════════════════════════════════════════════════

def test_blank_state_status_is_idle():
    assert fld._blank_state()["status"] == "idle"


def test_blank_state_current_round_is_zero():
    assert fld._blank_state()["current_round"] == 0


def test_blank_state_total_rounds_is_ten():
    assert fld._blank_state()["total_rounds"] == 10


def test_blank_state_client_losses_three_empty_lists():
    assert fld._blank_state()["client_losses"] == [[], [], []]


def test_blank_state_best_loss_is_inf():
    assert fld._blank_state()["best_loss"] == float("inf")


def test_blank_state_has_twenty_keys():
    assert len(fld._blank_state()) == 20


def test_reset_state_sets_total_rounds():
    fld._reset_state(25, "FedAvg")
    assert fld._STATE["total_rounds"] == 25


def test_reset_state_sets_algorithm():
    fld._reset_state(10, "FedAvg")
    assert fld._STATE["algorithm"] == "FedAvg"


def test_reset_state_clears_global_losses():
    # Manually populate global_losses, then reset
    fld._STATE["global_losses"] = [1.0, 0.9, 0.8]
    fld._reset_state(10, "FedAvg")
    assert fld._STATE["global_losses"] == []


def test_read_state_returns_copy_not_reference():
    """Mutating the returned dict does not alter _STATE."""
    snap = fld._read_state()
    snap["status"] = "running"
    assert fld._STATE["status"] == "idle"


def test_read_state_client_losses_is_deep_copy():
    """client_losses lists in the snapshot are independent from _STATE."""
    snap = fld._read_state()
    snap["client_losses"][0].append(99.9)
    assert fld._STATE["client_losses"][0] == []


# ═════════════════════════════════════════════════════════════════════════════
# 3B — _fedavg()
# ═════════════════════════════════════════════════════════════════════════════

def test_fedavg_equal_samples_gives_arithmetic_mean():
    """Equal sample counts → simple arithmetic mean per layer."""
    w0 = [np.array([[1.0, 2.0]], dtype=np.float32)]
    w1 = [np.array([[3.0, 4.0]], dtype=np.float32)]
    result = fld._fedavg([w0, w1], [100, 100])
    npt.assert_allclose(result[0], np.array([[2.0, 3.0]], dtype=np.float32))


def test_fedavg_unequal_samples_weighted():
    """Different sample counts produce a weighted average."""
    w0 = [np.array([0.0], dtype=np.float32)]
    w1 = [np.array([4.0], dtype=np.float32)]
    # client 0: 25 samples, client 1: 75 samples → 0*0.25 + 4*0.75 = 3.0
    result = fld._fedavg([w0, w1], [25, 75])
    npt.assert_allclose(result[0], np.array([3.0], dtype=np.float32), atol=1e-5)


def test_fedavg_single_client_unchanged():
    """Single client returns that client's weights unchanged."""
    w = _simple_weights(n_layers=2, shape=(3, 3))
    result = fld._fedavg([w], [100])
    for r, orig in zip(result, w):
        npt.assert_allclose(r, orig, atol=1e-6)


def test_fedavg_output_shape_matches_input():
    """Output list length and per-layer shapes match the inputs."""
    w0 = _simple_weights(n_layers=3, shape=(5, 7))
    w1 = _simple_weights(n_layers=3, shape=(5, 7))
    result = fld._fedavg([w0, w1], [50, 50])
    assert len(result) == 3
    for r, orig in zip(result, w0):
        assert r.shape == orig.shape


def test_fedavg_output_dtype_float32():
    """All output arrays have dtype float32."""
    w0 = _simple_weights(); w1 = _simple_weights()
    result = fld._fedavg([w0, w1], [50, 50])
    assert all(r.dtype == np.float32 for r in result)


def test_fedavg_result_bounded_by_inputs():
    """Weighted average never extrapolates beyond min/max of client weights."""
    w0 = [np.array([1.0, 5.0], dtype=np.float32)]
    w1 = [np.array([3.0, 7.0], dtype=np.float32)]
    for s0, s1 in [(10, 90), (50, 50), (90, 10)]:
        result = fld._fedavg([w0, w1], [s0, s1])
        assert np.all(result[0] >= 1.0) and np.all(result[0] <= 7.0)


# ═════════════════════════════════════════════════════════════════════════════
# 3C — _fedyogi()
# ═════════════════════════════════════════════════════════════════════════════

def _yogi_inputs(n=2):
    """Return (client_weights, client_samples, global_weights)."""
    rng = np.random.default_rng(1)
    global_w = [rng.standard_normal((4, 4)).astype(np.float32) for _ in range(n)]
    # Perturb clients slightly from global
    client_w = [
        [g + rng.standard_normal(g.shape).astype(np.float32) * 0.1 for g in global_w]
        for _ in range(3)
    ]
    return client_w, [100, 100, 100], global_w


def test_fedyogi_returns_triple():
    """Returns a (new_weights, new_m, new_v) 3-tuple."""
    cw, cs, gw = _yogi_inputs()
    result = fld._fedyogi(cw, cs, gw, None, None)
    assert isinstance(result, tuple) and len(result) == 3


def test_fedyogi_output_shape_matches_global():
    """new_weights has the same shape as global_weights per layer."""
    cw, cs, gw = _yogi_inputs()
    new_w, _, _ = fld._fedyogi(cw, cs, gw, None, None)
    for nw, g in zip(new_w, gw):
        assert nw.shape == g.shape


def test_fedyogi_cold_start_none_mt_vt():
    """m_t=None, v_t=None initialises momentum and variance from scratch."""
    cw, cs, gw = _yogi_inputs()
    new_w, new_m, new_v = fld._fedyogi(cw, cs, gw, None, None)
    # Cold start: m_t was zeros → new_m = (1-beta1)*delta ≠ None
    assert new_m is not None
    assert new_v is not None
    assert all(isinstance(m, np.ndarray) for m in new_m)


def test_fedyogi_vt_never_below_tau_squared():
    """v_t values are always ≥ tau² (numerical stability guard)."""
    cw, cs, gw = _yogi_inputs()
    tau = 1e-3
    _, _, new_v = fld._fedyogi(cw, cs, gw, None, None, tau=tau)
    for v in new_v:
        assert np.all(v >= tau ** 2), "v_t fell below tau²"


def test_fedyogi_new_m_shape_matches_delta():
    """new_m has the same shape as the delta (avg − global) per layer."""
    cw, cs, gw = _yogi_inputs()
    _, new_m, _ = fld._fedyogi(cw, cs, gw, None, None)
    for m, g in zip(new_m, gw):
        assert m.shape == g.shape


def test_fedyogi_zero_delta_no_weight_change():
    """When client weights equal global weights, output weights are unchanged."""
    global_w  = _simple_weights(n_layers=2, shape=(3, 3))
    client_w  = [list(global_w), list(global_w), list(global_w)]
    new_w, _, _ = fld._fedyogi(client_w, [100, 100, 100], global_w, None, None)
    for nw, g in zip(new_w, global_w):
        npt.assert_allclose(nw, g, atol=1e-5)


def test_fedyogi_output_dtype_float32():
    """All new_weights arrays have dtype float32."""
    cw, cs, gw = _yogi_inputs()
    new_w, _, _ = fld._fedyogi(cw, cs, gw, None, None)
    assert all(w.dtype == np.float32 for w in new_w)


def test_fedyogi_warm_start_uses_previous_mt_vt():
    """Passing previous m_t/v_t yields different output than cold start."""
    cw, cs, gw = _yogi_inputs()
    _, m1, v1  = fld._fedyogi(cw, cs, gw, None, None)
    # Warm start: m1/v1 are not zero — output should differ from cold-start
    cold_w, _, _ = fld._fedyogi(cw, cs, gw, None, None)
    warm_w, _, _ = fld._fedyogi(cw, cs, gw, m1,   v1)
    diffs = [np.abs(c - w).max() for c, w in zip(cold_w, warm_w)]
    assert any(d > 1e-7 for d in diffs), "warm/cold start produced identical results"


# ═════════════════════════════════════════════════════════════════════════════
# 3D — load_training_metadata()
# ═════════════════════════════════════════════════════════════════════════════

def test_load_metadata_returns_dict(monkeypatch):
    """load_training_metadata always returns a dict."""
    monkeypatch.setattr(fld.os.path, "exists", lambda p: False)
    assert isinstance(fld.load_training_metadata(), dict)


def test_load_metadata_fallback_num_rounds(monkeypatch):
    """Fallback (no file): num_rounds == 100."""
    monkeypatch.setattr(fld.os.path, "exists", lambda p: False)
    assert fld.load_training_metadata()["num_rounds"] == 100


def test_load_metadata_fallback_final_mae(monkeypatch):
    """Fallback: final_mae == 1.8968."""
    monkeypatch.setattr(fld.os.path, "exists", lambda p: False)
    assert fld.load_training_metadata()["final_mae"] == pytest.approx(1.8968)


def test_load_metadata_fallback_final_r2(monkeypatch):
    """Fallback: final_r2 == 0.5995."""
    monkeypatch.setattr(fld.os.path, "exists", lambda p: False)
    assert fld.load_training_metadata()["final_r2"] == pytest.approx(0.5995)


def test_load_metadata_fallback_hospitals(monkeypatch):
    """Fallback: hospitals == 3."""
    monkeypatch.setattr(fld.os.path, "exists", lambda p: False)
    assert fld.load_training_metadata()["hospitals"] == 3


def test_load_metadata_reads_from_file(monkeypatch):
    """When JSON file exists, values are loaded from it."""
    data = {"num_rounds": 50, "final_mae": 1.2, "final_r2": 0.75, "hospitals": 3}
    monkeypatch.setattr(fld.os.path, "exists", lambda p: True)
    with patch("builtins.open", mock_open(read_data=json.dumps(data))):
        result = fld.load_training_metadata()
    assert result["num_rounds"] == 50


def test_load_metadata_file_overrides_final_r2(monkeypatch):
    """Custom final_r2 in JSON file is returned (not the fallback value)."""
    data = {"num_rounds": 100, "final_mae": 1.8, "final_r2": 0.99, "hospitals": 3}
    monkeypatch.setattr(fld.os.path, "exists", lambda p: True)
    with patch("builtins.open", mock_open(read_data=json.dumps(data))):
        result = fld.load_training_metadata()
    assert result["final_r2"] == pytest.approx(0.99)


# ═════════════════════════════════════════════════════════════════════════════
# 3E — Chart builders
# ═════════════════════════════════════════════════════════════════════════════

import plotly.graph_objects as go   # noqa: E402


def test_training_chart_empty_returns_none():
    """_training_chart with empty histories → None."""
    assert fld._training_chart([], [], []) is None


def test_training_chart_with_data_returns_figure():
    """_training_chart with 3 rounds → returns a Plotly Figure."""
    mae_h = [[2.1, 1.9, 1.7], [2.0, 1.8, 1.6], [2.2, 2.0, 1.8]]
    g_mae = [2.1, 1.9, 1.7]
    g_r2  = [-0.3, 0.2, 0.5]
    fig   = fld._training_chart(mae_h, g_mae, g_r2)
    assert isinstance(fig, go.Figure)


def test_training_chart_has_secondary_y_axis():
    """_training_chart figure uses a secondary y-axis for R²."""
    mae_h = [[1.5, 1.3], [1.6, 1.4], [1.7, 1.5]]
    g_mae = [1.5, 1.3]
    g_r2  = [0.3, 0.5]
    fig   = fld._training_chart(mae_h, g_mae, g_r2)
    # Secondary y-axis is present when specs include it
    assert fig.layout.yaxis2 is not None


def test_hist_chart_none_inputs_returns_none():
    """_hist_chart(None, None, 1) → None."""
    assert fld._hist_chart(None, None, 1) is None


def test_hist_chart_with_data_returns_figure():
    """_hist_chart with real pre/post flat arrays → Plotly Figure."""
    rng  = np.random.default_rng(0)
    pre  = rng.standard_normal(1000).astype(np.float32)
    post = rng.standard_normal(1000).astype(np.float32)
    fig  = fld._hist_chart(pre, post, 3)
    assert isinstance(fig, go.Figure)


def test_hist_chart_barmode_overlay():
    """_hist_chart figure uses barmode='overlay'."""
    rng  = np.random.default_rng(0)
    pre  = rng.standard_normal(500).astype(np.float32)
    post = rng.standard_normal(500).astype(np.float32)
    fig  = fld._hist_chart(pre, post, 1)
    assert fig.layout.barmode == "overlay"


def test_heatmap_chart_none_returns_none():
    """_heatmap_chart(None) → None."""
    assert fld._heatmap_chart(None) is None


def test_heatmap_chart_with_weights_returns_figure():
    """_heatmap_chart with a weight matrix → Plotly Figure."""
    rng     = np.random.default_rng(0)
    weights = rng.standard_normal((128, 108)).astype(np.float32)
    fig     = fld._heatmap_chart(weights)
    assert isinstance(fig, go.Figure)


def test_heatmap_chart_slice_shape():
    """Heatmap data slice is weights.T[:40, :64] — verify shape from heatmap trace."""
    rng     = np.random.default_rng(0)
    weights = rng.standard_normal((128, 108)).astype(np.float32)  # (input→128)
    fig     = fld._heatmap_chart(weights)
    # weights.T = (108,128) → [:40,:64] = (40,64)
    heatmap_data = fig.data[0].z
    assert np.array(heatmap_data).shape == (40, 64)


# ═════════════════════════════════════════════════════════════════════════════
# 3F — _network_diagram()
# ═════════════════════════════════════════════════════════════════════════════

def _diagram(phase="idle", rnd=1, total=10, algorithm="FedYogi",
             client_metrics=None, best_round=0, best_loss=float("inf")):
    if client_metrics is None:
        client_metrics = [{}, {}, {}]
    return fld._network_diagram(phase, rnd, total, algorithm,
                                 client_metrics, best_round, best_loss)


def test_network_diagram_returns_nonempty_string():
    html = _diagram()
    assert isinstance(html, str) and len(html) > 0


def test_network_diagram_contains_doctype():
    assert "<!DOCTYPE html>" in _diagram()


def test_network_diagram_distributing_has_dotdown():
    """Phase 'distributing' → dots flow down (dotDown animation)."""
    assert "dotDown" in _diagram(phase="distributing")


def test_network_diagram_aggregating_has_dotup():
    """Phase 'aggregating' → dots flow up (dotUp animation)."""
    assert "dotUp" in _diagram(phase="aggregating")


def test_network_diagram_idle_no_animation():
    """Phase 'idle' → static dashed line; animation property not applied to any element."""
    html = _diagram(phase="idle")
    # @keyframes dotDown/@keyframes dotUp definitions appear in <style> unconditionally;
    # only check that the property is not applied to any element (animation:dotDown / animation:dotUp).
    assert "animation:dotDown" not in html
    assert "animation:dotUp"   not in html


def test_network_diagram_done_contains_best_round():
    """Phase 'done' with best_round=7 → HTML contains '7'."""
    html = _diagram(phase="done", best_round=7, best_loss=1.23)
    assert "7" in html


def test_network_diagram_hospital_names_present():
    """All three hospital names appear in the HTML."""
    html = _diagram()
    for name in fld.HOSP_NAMES:
        assert name in html, f"Hospital name '{name}' missing from diagram"


def test_network_diagram_algorithm_name_present():
    """Algorithm name appears in the HTML (shown in server-sub div)."""
    html = _diagram(algorithm="FedAvg")
    assert "FedAvg" in html


def test_network_diagram_inf_best_loss_no_crash():
    """best_loss=inf does not crash; function returns a non-empty string."""
    html = _diagram(phase="done", best_round=0, best_loss=float("inf"))
    assert isinstance(html, str) and len(html) > 0

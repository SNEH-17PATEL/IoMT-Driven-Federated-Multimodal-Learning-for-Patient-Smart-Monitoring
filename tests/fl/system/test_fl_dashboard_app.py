"""
Page-level tests for fl_dashboard.py, driven by Streamlit's own AppTest runner.

The page is run for real (every st.* call executes) but against a SANDBOX:
  • db.DB_PATH            -> a temp SQLite file holding small synthetic fl_training rows
  • fl_dashboard.MODEL_PATH -> a temp folder with a copy of scaler.pkl (+ optional metadata)
so the real database and models/ folder are never read or written.

Simulation state lives in module globals that survive Streamlit reruns, so tests set it
directly (_STATE) to put the page into the running / stopped / done / error views without
waiting for real training.
"""
import json
import os
import shutil
import sqlite3
import sys
import threading
import time

import joblib
import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import db as db_mod                            # noqa: E402

# AppTest is imported lazily by the `real_streamlit` fixture below (see its docstring).
AppTest = None

SCRIPT = os.path.join(_ROOT, "fl_dashboard.py")
REAL_MODELS = os.path.join(_ROOT, "models")

pytestmark = pytest.mark.slow


def _is_streamlit_name(name):
    return name == "streamlit" or name.startswith("streamlit.")


@pytest.fixture(scope="module", autouse=True)
def real_streamlit():
    """Give this module the REAL streamlit.

    Three other test files (test_fl_dashboard_fns, test_fl_round_integration, test_fl_system)
    replace sys.modules["streamlit"] with a MagicMock when pytest collects them, and never put
    the real one back. AppTest and the page itself need the real library, so swap it in for the
    duration of this module and restore exactly what was there before, so those files are
    unaffected."""
    global AppTest
    saved = {k: v for k, v in sys.modules.items() if _is_streamlit_name(k)}
    if not hasattr(sys.modules.get("streamlit"), "__path__"):      # a mock, not the package
        for k in saved:
            sys.modules.pop(k)
    from streamlit.testing.v1 import AppTest as _AppTest
    AppTest = _AppTest
    yield
    for k in [k for k in sys.modules if _is_streamlit_name(k)]:
        sys.modules.pop(k)
    sys.modules.update(saved)


# ── Sandbox ───────────────────────────────────────────────────────────────────

def _rows(n_per_client=60, seed=0):
    rng = np.random.default_rng(seed)
    names = list(joblib.load(os.path.join(REAL_MODELS, "scaler.pkl")).feature_names_in_)
    frames = []
    for cid in range(3):
        df = pd.DataFrame(rng.standard_normal((n_per_client, len(names))).astype(np.float32), columns=names)
        df["sofa_score"] = rng.uniform(0, 20, n_per_client)
        df.insert(0, "client_id", cid)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy(os.path.join(REAL_MODELS, "scaler.pkl"), models / "scaler.pkl")
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "dash.db"))
    db_mod.init_db()
    conn = sqlite3.connect(db_mod.DB_PATH)
    _rows().to_sql("fl_training", conn, index=False, if_exists="append")
    conn.close()
    return tmp_path



def _patch_model_path(monkeypatch, sandbox):
    """AppTest re-executes the file, so patch os.path.exists/model path via the module the
    script imports from. fl_dashboard computes MODEL_PATH from __file__, so redirect the two
    functions that use it instead: joblib.load and os.path.exists for the models folder."""
    real_exists = os.path.exists
    real_load = joblib.load
    models_dir = str(sandbox / "models")
    real_models = os.path.join(_ROOT, "models")

    def redirect(p):
        p = str(p)
        return p.replace(real_models, models_dir) if p.startswith(real_models) else p

    monkeypatch.setattr(os.path, "exists", lambda p: real_exists(redirect(p)))
    monkeypatch.setattr(joblib, "load", lambda p, *a, **k: real_load(redirect(p), *a, **k))
    import builtins
    real_open = builtins.open

    def guarded_open(p, *a, **k):
        return real_open(redirect(p) if isinstance(p, (str, os.PathLike)) else p, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded_open)


@pytest.fixture
def dash_cache(monkeypatch):
    """The page keeps its simulation state in an @st.cache_resource singleton. AppTest runs the
    script in its own namespace, so the only reliable way for a test to reach that state is to
    replace st.cache_resource with a memoiser whose storage the test owns. A fresh dict per
    test also keeps tests independent of each other."""
    import streamlit as st
    store = {}

    def cache_resource(fn=None, **_kw):
        def deco(f):
            def wrapper(*a, **k):
                if f.__name__ not in store:
                    store[f.__name__] = f(*a, **k)
                return store[f.__name__]
            wrapper.__name__ = f.__name__
            return wrapper
        return deco(fn) if callable(fn) else deco

    monkeypatch.setattr(st, "cache_resource", cache_resource)
    return store


@pytest.fixture
def at(sandbox, monkeypatch, dash_cache):
    _patch_model_path(monkeypatch, sandbox)
    return AppTest.from_file(SCRIPT, default_timeout=60)


def _texts(at):
    out = []
    for group in (at.markdown, at.error, at.info, at.warning, at.success, at.caption):
        out += [str(e.value) for e in group]
    return "\n".join(out)


def _shared(dash_cache):
    return dash_cache["_shared"]            # {"state": dict, "lock": Lock, "stop": Event}


def _set_state(dash_cache, **values):
    sh = _shared(dash_cache)
    with sh["lock"]:
        sh["state"].update(values)


def _stop_page_refresh(monkeypatch):
    """A running page ends with time.sleep(1); st.rerun() — an endless refresh loop.
    Make sleep raise, but ONLY when called from fl_dashboard.py, so AppTest's own internal
    waiting is unaffected. Everything above that line has already been drawn by then."""
    real_sleep = time.sleep

    def sleep(seconds):
        if sys._getframe(1).f_code.co_filename.endswith("fl_dashboard.py"):
            raise RuntimeError("stop-autorefresh")
        return real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", sleep)


# ── Guard screens ─────────────────────────────────────────────────────────────

def test_page_stops_with_a_clear_message_when_training_data_is_missing(tmp_path, monkeypatch, dash_cache):
    (tmp_path / "models").mkdir()                     # no scaler.pkl, and no fl_training table
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "empty.db"))
    db_mod.init_db()
    _patch_model_path(monkeypatch, tmp_path)
    at = AppTest.from_file(SCRIPT, default_timeout=60).run()
    assert not at.exception
    msg = " ".join(e.value for e in at.error)
    assert "Missing data" in msg
    assert "fl_training" in msg and "scaler.pkl" in msg
    assert len(at.button) == 0                        # st.stop() hid the controls


def test_page_reports_a_data_loading_failure(sandbox, monkeypatch, dash_cache):
    _patch_model_path(monkeypatch, sandbox)
    monkeypatch.setattr(db_mod, "get_fl_client_data",
                        lambda cid: (_ for _ in ()).throw(RuntimeError("db unreadable")))
    at = AppTest.from_file(SCRIPT, default_timeout=60).run()
    assert not at.exception
    msg = " ".join(e.value for e in at.error)
    assert "Failed to load training data" in msg and "db unreadable" in msg


# ── Idle screen ───────────────────────────────────────────────────────────────

def test_idle_page_shows_controls_and_how_it_works(at):
    at.run()
    assert not at.exception
    assert [s.label for s in at.slider] == ["Rounds", "Speed (s / phase)"]
    assert at.slider[0].value == 10 and (at.slider[0].min, at.slider[0].max) == (5, 15)
    assert at.selectbox[0].options == ["FedAvg", "FedYogi", "FedAvg + FedProx", "FedYogi + FedProx"]
    assert at.selectbox[0].value == "FedYogi + FedProx"
    assert [b.label for b in at.button] == ["▶ Start", "⏹ Stop", "↺ Reset"]
    assert at.button[0].disabled is False and at.button[1].disabled is True
    assert "How federated learning works" in _texts(at)


def test_default_algorithm_shows_the_algorithm_banner(at):
    at.run()
    assert 'class="algo-banner"' in _texts(at)


def test_other_algorithms_hide_the_banner(at):
    at.run()
    at.selectbox[0].set_value("FedAvg").run()
    assert 'class="algo-banner"' not in _texts(at)


def test_model_info_tab_shows_fallback_numbers_without_metadata_file(at):
    at.run()
    t = _texts(at)
    assert "Training Configuration" in t and "Global Model Performance" in t
    assert "Differential Privacy" in t


def test_model_info_tab_reads_the_metadata_file_when_present(sandbox, monkeypatch, dash_cache):
    _patch_model_path(monkeypatch, sandbox)
    (sandbox / "models" / "training_metadata.json").write_text(json.dumps({
        "num_rounds": 77, "hospitals": 3, "epochs_per_round": 3, "aggregation": "FedAvg + FedProx (μ=0.5)",
        "train_samples": 1234, "test_samples": 321, "best_round": 9, "split_type": "IID",
        "final_mae": 1.5, "final_r2": 0.5, "pred_range_min": 0.1, "pred_range_max": 20.0,
        "differential_privacy": False,
    }))
    at = AppTest.from_file(SCRIPT, default_timeout=60).run()
    t = _texts(at)
    assert "77" in t and "1,234" in t and "321" in t
    assert "1.500" in t and "0.500" in t


def test_model_info_tab_shows_dp_block_when_enabled(sandbox, monkeypatch, dash_cache):
    _patch_model_path(monkeypatch, sandbox)
    (sandbox / "models" / "training_metadata.json").write_text(json.dumps({
        "differential_privacy": True, "dp_sigma": 1.5, "dp_sensitivity": 2.0,
        "dp_epsilon": 4.2, "dp_delta": 1e-5,
    }))
    at = AppTest.from_file(SCRIPT, default_timeout=60).run()
    t = _texts(at)
    assert "Differential Privacy ENABLED" in t and "1.5" in t and "4.2" in t


def test_dataset_summary_lists_all_three_hospitals(at):
    at.run()
    t = _texts(at)
    for name in ("General ICU", "Mixed ICU", "Cardiac/Trauma ICU"):
        assert name in t
    assert "Training samples" in t


# ── Buttons ───────────────────────────────────────────────────────────────────

def test_start_launches_a_background_run_with_the_chosen_settings(at, monkeypatch, dash_cache):
    started = {}
    RealThread = threading.Thread

    class _SpyThread(RealThread):
        """Records the FL thread instead of running it; every other thread behaves normally."""
        def __init__(self, *a, **kw):
            self._is_fl = getattr(kw.get("target"), "__name__", "") == "fl_thread"
            if self._is_fl:
                started["args"] = kw["args"]
            super().__init__(*a, **kw)

        def start(self):
            if self._is_fl:
                started["started"] = True
                return
            super().start()

    monkeypatch.setattr(threading, "Thread", _SpyThread)
    _stop_page_refresh(monkeypatch)                    # the page is 'running' after Start
    at.run()
    at.slider[0].set_value(6)
    at.slider[1].set_value(2.5)
    at.selectbox[0].set_value("FedAvg")
    at.button[0].click().run()
    assert started.get("started") is True
    rounds, speed, algorithm = started["args"][:3]
    assert (rounds, speed, algorithm) == (6, 2.5, "FedAvg")
    st_ = _shared(dash_cache)["state"]
    assert st_["status"] == "running" and st_["total_rounds"] == 6 and st_["algorithm"] == "FedAvg"


def test_reset_button_returns_the_page_to_idle(at, dash_cache):
    at.run()
    _set_state(dash_cache, status="done", **_history())
    at.run()
    assert "Round History" in _texts(at)
    next(b for b in at.button if b.label == "↺ Reset").click().run()
    assert not at.exception
    assert "How federated learning works" in _texts(at)
    assert _shared(dash_cache)["state"]["status"] == "idle"


# ── Result views (state is injected, so no real training is needed) ───────────

def _history(n=3):
    rng = np.random.default_rng(5)
    return {
        "current_round": n, "total_rounds": 10,
        "client_losses": [[1.0 - 0.1 * i for i in range(n)] for _ in range(3)],
        "global_losses": [1.0 - 0.1 * i for i in range(n)],
        "client_mae_history": [[2.0 - 0.1 * i for i in range(n)] for _ in range(3)],
        "global_mae_history": [2.0 - 0.1 * i for i in range(n)],
        "global_r2_history": [0.1 * i for i in range(n)],
        "client_metrics": [{"mae": 1.9, "r2": 0.3, "loss": 0.8}] * 3,
        "weight_stats": [[{"n_params": 69121, "mean": 0.01, "std": 0.2, "delta_mean": 0.05}] * 3
                         for _ in range(n)],
        "pre_agg_flat": rng.standard_normal(500), "post_agg_flat": rng.standard_normal(500),
        "layer1_weights": rng.standard_normal((256, 108)),
        "best_round": 2, "best_loss": 0.8,
        "round_log": [{"Round": i + 1, "H0 Loss": 1.0, "H1 Loss": 0.9, "H2 Loss": 0.8,
                       "Global": 0.9, "Best": "✓ Best" if i == 1 else ""} for i in range(n)],
    }


@pytest.mark.parametrize("status", ["done", "stopped", "error"])
def test_finished_views_render_without_errors(at, dash_cache, status):
    at.run()
    _set_state(dash_cache, status=status, algorithm="FedAvg + FedProx",
               current_phase=status, phase_label=f"sim {status}", error=None, **_history())
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    t = _texts(at)
    assert "Round History" in t and "Weight Statistics" in t
    assert status.upper() in t                        # the status tile shows DONE / STOPPED / ERROR
    assert "3 round(s) completed" in t


def test_best_round_is_marked_in_the_history_table(at, dash_cache):
    at.run()
    _set_state(dash_cache, status="done", **_history())
    at.run()
    assert "✓ Best" in _texts(at)


def test_view_with_no_completed_round_shows_the_placeholder_chart(at, dash_cache):
    at.run()
    _set_state(dash_cache, status="stopped", current_round=1, total_rounds=10)
    at.run()
    assert not at.exception
    assert "appear after Round 1" in _texts(at)


def test_running_view_disables_controls_and_enables_stop(at, dash_cache, monkeypatch):
    at.run()
    _set_state(dash_cache, status="running", current_phase="training_1",
               phase_label="Round 2/10 — Training Hospital 1", **_history(2))
    _stop_page_refresh(monkeypatch)
    at.run()
    # the auto-refresh branch ran (our patched sleep is what ended it)
    assert any("stop-autorefresh" in str(e.value) for e in at.exception)
    assert all(sl.disabled for sl in at.slider)
    assert at.selectbox[0].disabled is True
    by_label = {b.label: b for b in at.button}
    assert by_label["▶ Start"].disabled is True
    assert by_label["⏹ Stop"].disabled is False
    assert by_label["↺ Reset"].disabled is True
    assert "SIMULATION RUNNING" in _texts(at)


def test_clicking_stop_sets_the_stop_flag(at, dash_cache, monkeypatch):
    at.run()
    _set_state(dash_cache, status="running", current_phase="aggregating",
               phase_label="Aggregating", **_history(2))
    _stop_page_refresh(monkeypatch)
    at.run()
    stop_flag = _shared(dash_cache)["stop"]
    stop_flag.clear()
    next(b for b in at.button if b.label == "⏹ Stop").click().run()
    assert stop_flag.is_set()

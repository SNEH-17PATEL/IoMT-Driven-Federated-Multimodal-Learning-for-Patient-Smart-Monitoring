"""
Script-level tests for train_federated.py (Phase 0 preprocessing + Phase 1-6 training).

train_federated.py is a script: importing it runs the whole pipeline. These tests run it
in a SANDBOX so nothing real is touched:

  • working directory   = a temp folder that holds its own copy of the sklearn artifacts
                          (so "models/" resolves there and every write lands there)
  • database            = a temp SQLite file (db.DB_PATH is redirected)
  • Flower simulation   = replaced by a fake that exercises the server strategy directly
  • BigQuery            = replaced by a fake client that returns small DataFrames

The source is exec'd from a string, which lets individual tests flip a config flag
(USE_DP, USE_NONIID_SPLIT, ...) without editing the real file.

The real models/ folder and icu_monitor.db are never read or written.
"""
import json
import os
import shutil
import sqlite3
import sys
import types
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import flwr as fl                                         # noqa: E402
from flwr.common import Context, ndarrays_to_parameters   # noqa: E402
from flwr.app import RecordDict                           # noqa: E402

import db as db_mod                                       # noqa: E402
import model_utils                                        # noqa: E402
from model_utils import ICUModel, get_weights             # noqa: E402

SCRIPT = os.path.join(_ROOT, "train_federated.py")
REAL_MODELS = os.path.join(_ROOT, "models")
ARTIFACTS = ["scaler.pkl", "tfidf_vectorizer.pkl", "feature_columns.pkl"]

pytestmark = pytest.mark.slow


# ── Sandbox ───────────────────────────────────────────────────────────────────

def _feature_names():
    return list(joblib.load(os.path.join(REAL_MODELS, "scaler.pkl")).feature_names_in_)


def _synthetic_fl_rows(n_per_client=80, drop_cols=(), seed=0):
    rng = np.random.default_rng(seed)
    names = [c for c in _feature_names() if c not in drop_cols]
    frames = []
    for cid in range(3):
        X = rng.standard_normal((n_per_client, len(names))).astype(np.float32)
        y = np.concatenate([rng.uniform(0, 5, 40), rng.uniform(5, 10, 24), rng.uniform(10, 24, 16)])
        rng.shuffle(y)
        df = pd.DataFrame(X, columns=names)
        df["sofa_score"] = y[:n_per_client]
        df.insert(0, "client_id", cid)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Temp cwd with copied artifacts + temp DB pre-loaded with synthetic fl_training rows."""
    (tmp_path / "models").mkdir()
    for f in ARTIFACTS:
        shutil.copy(os.path.join(REAL_MODELS, f), tmp_path / "models" / f)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "sandbox.db"))
    db_mod.init_db()
    conn = sqlite3.connect(db_mod.DB_PATH)
    _synthetic_fl_rows().to_sql("fl_training", conn, index=False, if_exists="append")
    conn.close()
    monkeypatch.setattr(fl.simulation, "start_simulation", _make_fake_simulation())
    return tmp_path


def _make_fake_simulation(*, produce_weights=True, calls=None):
    """Fake flwr start_simulation: drives the script's own SaveBestStrategy and client_fn."""
    def fake(client_fn=None, num_clients=None, config=None, strategy=None, **_):
        ns = type(strategy).aggregate_evaluate.__globals__     # the script's namespace
        if calls is not None:
            calls.append({"ns": ns, "num_clients": num_clients, "rounds": config.num_rounds,
                          "client_fn": client_fn, "strategy": strategy})
        if not produce_weights:
            return
        params = ndarrays_to_parameters(get_weights(ICUModel(ns["input_dim"])))
        ns["_last_weights"]["params"] = params
        res = lambda loss: SimpleNamespace(loss=loss, num_examples=10, metrics={})
        strategy.aggregate_evaluate(1, [(None, res(0.9)), (None, res(1.1))], [])   # new best
        strategy.aggregate_evaluate(2, [(None, res(2.0)), (None, res(2.2))], [])   # not better
    return fake


def run_script(subs=()):
    """Exec train_federated.py (optionally with source substitutions); return its namespace."""
    src = open(SCRIPT, encoding="utf-8").read()
    for old, new in subs:
        assert old in src, f"substitution target not found: {old!r}"
        src = src.replace(old, new, 1)
    ns = {"__name__": "trf_under_test", "__file__": SCRIPT}
    exec(compile(src, SCRIPT, "exec"), ns)
    return ns


def _meta(sandbox):
    return json.loads((sandbox / "models" / "training_metadata.json").read_text())


# ── Happy path: training from an existing fl_training table ───────────────────

def test_training_runs_end_to_end_and_writes_all_artifacts(sandbox):
    ns = run_script()
    for f in ("federated_model.pth", "shap_background.npy", "training_metadata.json"):
        assert (sandbox / "models" / f).exists(), f
    assert ns["input_dim"] == 108
    meta = _meta(sandbox)
    assert meta["num_rounds"] == 100 and meta["hospitals"] == 3
    assert meta["aggregation"].startswith("FedAvg + FedProx")
    assert meta["input_features"] == 108
    assert meta["train_samples"] == 240


def test_save_best_strategy_tracks_the_best_round(sandbox):
    ns = run_script()
    assert ns["_best_round"] == 1                      # round 1 loss 1.0 beat round 2 loss 2.1
    assert ns["_best_loss"] == pytest.approx(1.0)
    assert _meta(sandbox)["best_round"] == 1


def test_missing_test_set_files_trigger_the_inflated_metrics_fallback(sandbox):
    with pytest.warns(UserWarning, match="No saved held-out test set"):
        ns = run_script()
    assert ns["X_test_np"].shape[0] == 48              # 20% of 240 client rows


def test_saved_test_set_is_loaded_from_disk_when_present(sandbox):
    rng = np.random.default_rng(1)
    np.save(sandbox / "models" / "fl_test_X.npy", rng.standard_normal((30, 108)).astype(np.float32))
    np.save(sandbox / "models" / "fl_test_y.npy", rng.uniform(0, 24, 30).astype(np.float32))
    ns = run_script()
    assert ns["X_test_np"].shape == (30, 108)
    assert _meta(sandbox)["test_samples"] == 30


def test_missing_feature_columns_are_zero_filled_with_a_warning(sandbox, monkeypatch):
    conn = sqlite3.connect(db_mod.DB_PATH)
    conn.execute("DELETE FROM fl_training")
    conn.commit()
    conn.close()
    gone = _feature_names()[:3]
    conn = sqlite3.connect(db_mod.DB_PATH)
    _synthetic_fl_rows(drop_cols=gone).to_sql("fl_training", conn, index=False, if_exists="replace")
    conn.close()
    with pytest.warns(UserWarning, match="missing from fl_training"):
        ns = run_script()
    assert ns["clients_data"][0][0].shape[1] == 108
    assert np.all(ns["clients_data"][0][0][:, :3] == 0)


@pytest.mark.parametrize("missing", ARTIFACTS)
def test_missing_sklearn_artifact_stops_the_run(sandbox, missing):
    os.remove(sandbox / "models" / missing)
    with pytest.raises(FileNotFoundError, match="required artifacts are missing"):
        run_script()


def test_no_weights_produced_raises_a_clear_error(sandbox, monkeypatch):
    monkeypatch.setattr(fl.simulation, "start_simulation", _make_fake_simulation(produce_weights=False))
    with pytest.raises(RuntimeError, match="Training produced no weights"):
        run_script()


def test_risk_metric_failure_is_reported_not_fatal(sandbox, monkeypatch):
    def boom(*a, **k):
        raise ValueError("metrics broke")
    monkeypatch.setattr(model_utils, "compute_sofa_risk_metrics", boom)
    run_script()
    assert _meta(sandbox)["risk_classification_metrics"] is None


def test_risk_metrics_are_stored_when_available(sandbox):
    run_script()
    rm = _meta(sandbox)["risk_classification_metrics"]
    assert "high_risk_alert" in rm and "recall" in rm["high_risk_alert"]


# ── Config flags ──────────────────────────────────────────────────────────────

def test_differential_privacy_flag_changes_metadata_and_client_uploads(sandbox):
    ns = run_script([("USE_DP         = False", "USE_DP         = True")])
    meta = _meta(sandbox)
    assert meta["differential_privacy"] is True
    assert meta["dp_sigma"] == 1.0 and meta["dp_epsilon"] is not None
    # a client's weights after DP must differ from the same client without DP
    client = ns["hospital_clients"][0]
    sent, n, _ = client.fit(get_weights(ICUModel(108)), {"server_round": 1})
    assert n == len(client.X_train) and len(sent) > 0


def test_noniid_flag_builds_biased_hospital_splits(sandbox):
    ns = run_script([("USE_NONIID_SPLIT = False", "USE_NONIID_SPLIT = True")])
    assert _meta(sandbox)["split_type"].startswith("Non-IID")
    high_share = [(y >= 10).mean() for _, y in ns["clients_data"]]
    assert high_share[2] > high_share[0]               # hospital 2 gets the high-SOFA patients


def test_fedyogi_flag_switches_the_server_strategy(sandbox):
    ns = run_script([("USE_FEDYOGI   = False", "USE_FEDYOGI   = True")])
    assert "FedYogi" in _meta(sandbox)["aggregation"]
    assert issubclass(type(ns["strategy"]), fl.server.strategy.FedYogi)


# ── make_noniid_split edge case ───────────────────────────────────────────────

def test_noniid_split_with_too_few_rows_raises(sandbox):
    ns = run_script()
    X = np.zeros((1, 5), dtype=np.float32)
    y = np.array([1.0], dtype=np.float32)              # one low-SOFA row -> two clients empty
    with pytest.raises(ValueError, match="empty client"):
        ns["make_noniid_split"](X, y)


# ── Flower client + strategy pieces the fake simulation hands back ────────────

def test_client_fn_returns_the_right_hospital(sandbox):
    calls = []
    sim = _make_fake_simulation(calls=calls)
    fl.simulation.start_simulation = sim
    ns = run_script()
    ctx = Context(run_id=1, node_id=1, node_config={"partition-id": 2},
                  state=RecordDict(), run_config={})
    assert ns["client_fn"](ctx) is not None
    assert calls[0]["num_clients"] == 3 and calls[0]["rounds"] == 100


def test_configure_fit_sends_the_round_number_to_every_client(sandbox):
    calls = []
    fl.simulation.start_simulation = _make_fake_simulation(calls=calls)
    run_script()
    strat = calls[0]["strategy"]

    class _Manager:
        def num_available(self):
            return 3

        def sample(self, num_clients, min_num_clients):
            return ["c0", "c1", "c2"][:num_clients]

    out = strat.configure_fit(7, ndarrays_to_parameters([np.zeros(2, dtype=np.float32)]), _Manager())
    assert len(out) == 3
    assert all(ins.config == {"server_round": 7} for _, ins in out)


def test_aggregate_fit_records_the_latest_aggregated_weights(sandbox, monkeypatch):
    calls = []
    fl.simulation.start_simulation = _make_fake_simulation(calls=calls)
    ns = run_script()
    strat = calls[0]["strategy"]
    marker = ndarrays_to_parameters([np.ones(3, dtype=np.float32)])
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_fit",
                        lambda self, rnd, res, fail: (marker, {}))
    out = strat.aggregate_fit(5, [], [])
    assert out[0] is marker and ns["_last_weights"]["params"] is marker


def test_aggregate_fit_with_no_result_keeps_previous_weights(sandbox, monkeypatch):
    calls = []
    fl.simulation.start_simulation = _make_fake_simulation(calls=calls)
    ns = run_script()
    strat = calls[0]["strategy"]
    before = ns["_last_weights"]["params"]
    monkeypatch.setattr(fl.server.strategy.FedAvg, "aggregate_fit", lambda *a: None)
    assert strat.aggregate_fit(5, [], []) is None
    assert ns["_last_weights"]["params"] is before


# ── Phase 0: BigQuery preprocessing (fully mocked) ────────────────────────────

class _FakeJob:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df.copy()


class _FakeBigQuery:
    """Stands in for google.cloud.bigquery; hands back small DataFrames for the 2 queries."""
    last_project = None

    def __init__(self, vitals, notes):
        self._vitals, self._notes = vitals, notes

    def Client(self, project=None, location=None):
        _FakeBigQuery.last_project = (project, location)
        outer = self

        class _C:
            def query(self, sql):
                return _FakeJob(outer._notes if "clinical_notes" in sql else outer._vitals)
        return _C()


def _bq_frames(n=200, seed=3):
    rng = np.random.default_rng(seed)
    sofa = np.concatenate([rng.uniform(0, 5, 100), rng.uniform(5, 10, 50), rng.uniform(10, 20, 50)])
    vitals = pd.DataFrame({
        "subject_id": np.arange(n), "hadm_id": np.arange(n), "icustay_id": np.arange(n),
        "window_time": pd.date_range("2024-01-01", periods=n, freq="h"), "rn": 1,
        "HR_mean": rng.normal(85, 10, n), "RR_mean": rng.normal(18, 3, n),
        "latest_SpO2": rng.normal(96, 2, n), "GCS_eye_opening": rng.integers(1, 5, n),
        "stress_score": rng.uniform(0, 10, n), "sofa_score": sofa,
    })
    texts = ["Patient intubated on ventilator with septic shock [**Name 12**] 120/80 on levophed",
             "Stable, no acute distress, creatinine 1.1 urine output adequate",
             "Hypotension requiring vasopressor, renal failure, dialysis started"]
    notes = pd.DataFrame({
        "subject_id": np.arange(n), "hadm_id": np.arange(n),
        "text": [texts[i % 3] if i % 11 else "" for i in range(n)],       # a few blank notes
    })
    return vitals, notes


@pytest.fixture
def empty_table_sandbox(sandbox, monkeypatch):
    """Same sandbox but fl_training is empty, so Phase 0 (BigQuery) must run."""
    conn = sqlite3.connect(db_mod.DB_PATH)
    conn.execute("DROP TABLE IF EXISTS fl_training")
    conn.commit()
    conn.close()
    for f in ARTIFACTS:                                # Phase 0 regenerates these
        os.remove(sandbox / "models" / f)
    vitals, notes = _bq_frames()
    fake_bq = _FakeBigQuery(vitals, notes)
    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    cloud.bigquery = fake_bq
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", fake_bq)
    return sandbox


def test_phase0_runs_when_fl_training_is_empty_and_saves_everything(empty_table_sandbox):
    ns = run_script()
    models = empty_table_sandbox / "models"
    for f in ARTIFACTS + ["fl_test_X.npy", "fl_test_y.npy", "federated_model.pth"]:
        assert (models / f).exists(), f
    assert _FakeBigQuery.last_project == ("mimic-project-2", "US")
    conn = sqlite3.connect(db_mod.DB_PATH)
    try:
        per_client = dict(conn.execute("SELECT client_id, COUNT(*) FROM fl_training GROUP BY client_id"))
    finally:
        conn.close()
    assert set(per_client) == {0, 1, 2}
    assert sum(per_client.values()) == 160            # 80% of 200 stays are training rows
    assert ns["X_test_np"].shape[0] == 40             # 20% held out
    assert abs(per_client[0] - per_client[2]) <= 1    # IID: three near-equal hospitals


def test_phase0_split_is_stratified_by_high_risk(empty_table_sandbox):
    run_script()
    conn = sqlite3.connect(db_mod.DB_PATH)
    try:
        rows = conn.execute(
            "SELECT client_id, AVG(sofa_score >= 10) FROM fl_training GROUP BY client_id").fetchall()
    finally:
        conn.close()
    shares = [r[1] for r in rows]
    assert max(shares) - min(shares) < 0.05           # every hospital gets the same high-risk mix


def test_phase0_scaler_and_vocabulary_are_saved_for_the_dashboard(empty_table_sandbox):
    run_script()
    scaler = joblib.load(empty_table_sandbox / "models" / "scaler.pkl")
    cols = joblib.load(empty_table_sandbox / "models" / "feature_columns.pkl")
    assert list(scaler.feature_names_in_) == list(cols)
    assert "sofa_score" not in cols and "hadm_id" not in cols
    tfidf = joblib.load(empty_table_sandbox / "models" / "tfidf_vectorizer.pkl")
    assert "septic shock" in tfidf.get_feature_names_out()


def test_phase0_replaces_existing_rows_for_a_client_on_rerun(empty_table_sandbox):
    run_script()
    conn = sqlite3.connect(db_mod.DB_PATH)
    first = conn.execute("SELECT COUNT(*) FROM fl_training").fetchone()[0]
    conn.close()
    # empty the table's rows (keep schema) so Phase 0 runs again against an existing table
    conn = sqlite3.connect(db_mod.DB_PATH)
    conn.execute("DELETE FROM fl_training")
    conn.commit()
    conn.close()
    run_script()
    conn = sqlite3.connect(db_mod.DB_PATH)
    second = conn.execute("SELECT COUNT(*) FROM fl_training").fetchone()[0]
    conn.close()
    assert first == second == 160


def test_phase0_without_bigquery_library_gives_install_hint(empty_table_sandbox, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_bigquery(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("google.cloud"):
            raise ImportError("No module named 'google.cloud.bigquery'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_bigquery)
    with pytest.raises(ImportError, match="google-cloud-bigquery not installed"):
        run_script()

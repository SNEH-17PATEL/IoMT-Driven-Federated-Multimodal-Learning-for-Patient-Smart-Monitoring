"""
Branch-coverage unit tests for backend/services/ml_service.py.

Covers the code paths the existing tests do not reach:
  • _safe_float with bad input
  • the conformal-interval calculation and all of its fall-back values
  • the SHAP explainer loader and the different shapes a SHAP result can have
  • risk labels, consistency flags, trajectory checks
  • every branch of the SHAP -> plain-English sentence builder
  • run_prediction end to end against a TEMPORARY database
    (empty database, vitals seeding, camera override, wrap-around, missing columns)

The real icu_monitor.db is never read or written: db.DB_PATH is redirected to a
temp file for every test that touches the database.
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import backend.services.ml_service as ml


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Fresh, empty SQLite database with the real schema."""
    import db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "ml_branch_test.db"))
    db.init_db()
    return db


def _insert_sample_rows(db, patient=1, n=25, note="septic shock, intubated, on vasopressor"):
    conn = db.get_conn()
    for i in range(n):
        conn.execute(
            "INSERT INTO sample_patients (patient,row_idx,timestamp,HR,RR,SpO2,Temp,SBP,DBP,MAP,"
            "GCS_eye_opening,stress_score,clinical_note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (patient, i, f"2024-01-01 00:{i:02d}", 90 + i, 18, 96, 37.2, 118, 76, 90, 4, 0.5, note),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def db_with_samples(tmp_db):
    _insert_sample_rows(tmp_db)
    return tmp_db


class _FakeExplainer:
    """Stands in for shap.DeepExplainer; returns whatever shape the test gives it."""
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def shap_values(self, X):
        if self.error:
            raise self.error
        return self.value


class _ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(len(x), 1)


# ── _safe_float ───────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("bad", [None, "abc", float("nan"), float("inf"), float("-inf"), [1]])
def test_safe_float_returns_default_for_bad_input(bad):
    assert ml._safe_float(bad, default=7.5) == 7.5


@pytest.mark.unit
def test_safe_float_converts_numeric_strings():
    assert ml._safe_float("3.5") == 3.5
    assert ml._safe_float(2) == 2.0


# ── risk_label / _vital_status ────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("sofa,text", [
    (0.0, "Low Risk"), (4.99, "Low Risk"),
    (5.0, "Moderate Risk"), (9.99, "Moderate Risk"),
    (10.0, "High Risk"), (24.0, "High Risk"),
])
def test_risk_label_band_edges(sofa, text):
    assert ml.risk_label(sofa)[0] == text


@pytest.mark.unit
def test_vital_status_low_normal_high():
    lo, hi = ml.NORMAL_RANGES["HR"]
    assert ml._vital_status(lo - 1, "HR") == "low"
    assert ml._vital_status((lo + hi) / 2, "HR") == "normal"
    assert ml._vital_status(hi + 1, "HR") == "high"


# ── compute_sofa_floor ────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("MAP,SpO2,gcs,expected", [
    (90, 98, 4, 0),
    (69, 98, 4, 1),
    (60, 98, 4, 2),
    (90, 93, 4, 1),
    (90, 85, 4, 2),
    (90, 98, 2, 2),
    (90, 98, 1, 3),
    (60, 85, 1, 7),
])
def test_sofa_floor_values(MAP, SpO2, gcs, expected):
    assert ml.compute_sofa_floor(MAP, SpO2, gcs) == expected


# ── vital_consistency_flags ───────────────────────────────────────────────────

@pytest.mark.unit
def test_consistency_flags_none_for_healthy():
    assert ml.vital_consistency_flags(80, 120, 98, 90, 2.0) == []


@pytest.mark.unit
def test_consistency_flag_shock_state():
    flags = ml.vital_consistency_flags(140, 80, 98, 90, 2.0)
    assert len(flags) == 1 and "shock" in flags[0]


@pytest.mark.unit
def test_consistency_flag_multi_organ_stress():
    flags = ml.vital_consistency_flags(80, 120, 85, 60, 3.0)
    assert len(flags) == 1 and "multi-organ" in flags[0]


@pytest.mark.unit
def test_consistency_flags_both_can_fire():
    assert len(ml.vital_consistency_flags(140, 80, 85, 60, 1.0)) == 2


@pytest.mark.unit
def test_consistency_flags_quiet_when_sofa_already_high():
    assert ml.vital_consistency_flags(140, 80, 85, 60, 9.0) == []


# ── check_trajectory ──────────────────────────────────────────────────────────

def _hist(rows):
    return pd.DataFrame(rows)


def _row(sofa, ts, hr=80, rr=16, spo2=97, sbp=120, map_=90):
    return {"SOFA": sofa, "Timestamp": ts, "HR": hr, "RR": rr, "SpO₂": spo2, "SBP": sbp, "MAP": map_}


@pytest.mark.unit
def test_trajectory_history_unavailable(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(ml, "get_prediction_history", boom)
    assert ml.check_trajectory(1, 5.0, 80, 16, 97, 120, 90) == (None, 0.0, False, "History unavailable")


@pytest.mark.unit
def test_trajectory_first_reading_when_less_than_two_rows(monkeypatch):
    monkeypatch.setattr(ml, "get_prediction_history", lambda *a, **k: _hist([_row(3, "t1")]))
    assert ml.check_trajectory(1, 5.0, 80, 16, 97, 120, 90)[3] == "First reading"


@pytest.mark.unit
def test_trajectory_duplicate_rows_count_as_first_reading(monkeypatch):
    monkeypatch.setattr(ml, "get_prediction_history",
                        lambda *a, **k: _hist([_row(3, "t1"), _row(3, "t1")]))
    assert ml.check_trajectory(1, 5.0, 80, 16, 97, 120, 90) == (None, 0.0, False, "First reading")


@pytest.mark.unit
def test_trajectory_big_rise_without_vital_change_is_flagged(monkeypatch):
    monkeypatch.setattr(ml, "get_prediction_history",
                        lambda *a, **k: _hist([_row(9, "t2"), _row(3, "t1")]))
    prev, jump, flagged, msg = ml.check_trajectory(1, 10.0, 80, 16, 97, 120, 90)
    assert prev == 3.0 and jump == 7.0 and flagged is True
    assert "↑" in msg


@pytest.mark.unit
def test_trajectory_big_drop_without_vital_change_is_flagged(monkeypatch):
    monkeypatch.setattr(ml, "get_prediction_history",
                        lambda *a, **k: _hist([_row(2, "t2"), _row(9, "t1")]))
    _, jump, flagged, msg = ml.check_trajectory(1, 3.0, 80, 16, 97, 120, 90)
    assert jump == -6.0 and flagged is True
    assert "↓" in msg


@pytest.mark.unit
def test_trajectory_big_jump_with_vital_change_is_not_flagged(monkeypatch):
    monkeypatch.setattr(ml, "get_prediction_history",
                        lambda *a, **k: _hist([_row(9, "t2"), _row(3, "t1", hr=80)]))
    # current HR differs by 40 from the previous 80 -> a real vital change explains the jump
    _, jump, flagged, msg = ml.check_trajectory(1, 10.0, 120, 16, 97, 120, 90)
    assert flagged is False and msg is None and jump == 7.0


@pytest.mark.unit
def test_trajectory_small_change_is_not_flagged(monkeypatch):
    monkeypatch.setattr(ml, "get_prediction_history",
                        lambda *a, **k: _hist([_row(4, "t2"), _row(3, "t1")]))
    assert ml.check_trajectory(1, 4.5, 80, 16, 97, 120, 90)[2] is False


# ── _interpret_shap ───────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("feature,value,expected", [
    ("latest_SpO2", 90, "low oxygen saturation"),
    ("latest_SpO2", 98, "oxygen saturation within range"),
    ("latest_HR", 120, "elevated heart rate"),
    ("latest_HR", 50, "low heart rate"),
    ("latest_HR", 80, "heart rate within range"),
    ("latest_RR", 30, "elevated breathing rate"),
    ("latest_RR", 8, "low breathing rate"),
    ("latest_RR", 16, "breathing rate within range"),
    ("latest_SBP", 150, "elevated blood pressure"),
    ("latest_MAP", 55, "low blood pressure"),
    ("latest_DBP", 80, "blood pressure within range"),
    ("latest_Temp", 38.6, "fever"),
    ("latest_Temp", 35.0, "hypothermia"),
    ("latest_Temp", 37.0, "normal temperature"),
    ("GCS_eye_opening", 1, "impaired consciousness"),
    ("GCS_eye_opening", 4, "intact eye opening"),
    ("stress_score", 8.0, "high physiological stress"),
    ("stress_score", 4.0, "elevated physiological stress"),
    ("stress_score", 1.0, "low physiological stress"),
])
def test_interpret_shap_descriptions(feature, value, expected):
    assert expected in ml._interpret_shap(feature, value, 0.5)


@pytest.mark.unit
def test_interpret_shap_direction_words():
    assert "increasing risk" in ml._interpret_shap("latest_HR", 80, 0.5)
    assert "reducing risk" in ml._interpret_shap("latest_HR", 80, -0.5)


@pytest.mark.unit
def test_interpret_shap_note_word():
    out = ml._interpret_shap("septic", 1.0, 0.3)
    assert out == 'Clinical note contains "septic" → increasing risk'


@pytest.mark.unit
def test_interpret_shap_unmatched_vital_uses_generic_sentence(monkeypatch):
    monkeypatch.setitem(ml.VITAL_UNITS, "zzz", ("Mystery", "u"))
    assert ml._interpret_shap("zzz", 2.0, 0.5) == "Mystery = 2.0u → increasing risk"


# ── Cached loaders ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_shap_explainer_is_none_without_background(monkeypatch):
    real = ml.load_artifacts()
    monkeypatch.setattr(ml, "load_artifacts", lambda: (*real[:4], None))
    assert ml._load_shap_explainer.__wrapped__() is None


@pytest.mark.unit
def test_training_metadata_falls_back_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ml, "MODEL_PATH", str(tmp_path) + os.sep)
    meta = ml.load_training_metadata.__wrapped__()
    assert meta["num_rounds"] == 100
    assert meta["input_features"] == 108


@pytest.mark.unit
def test_training_metadata_reads_file_when_present(monkeypatch, tmp_path):
    (tmp_path / "training_metadata.json").write_text(json.dumps({"final_mae": 9.99}))
    monkeypatch.setattr(ml, "MODEL_PATH", str(tmp_path) + os.sep)
    assert ml.load_training_metadata.__wrapped__() == {"final_mae": 9.99}


# ── Conformal interval ────────────────────────────────────────────────────────

def _patch_conformal(monkeypatch, frames, cols=("a", "b"), exists=True):
    monkeypatch.setattr(ml, "load_artifacts", lambda: (_ZeroModel(), None, None, list(cols), None))
    monkeypatch.setattr(ml, "fl_training_exists", lambda: exists)
    monkeypatch.setattr(ml, "get_fl_client_data", lambda i: frames[i])


def _frame(n=100, **extra):
    d = {"a": np.zeros(n), "b": np.zeros(n), "sofa_score": np.arange(n, dtype=float)}
    d.update(extra)
    return pd.DataFrame(d)


@pytest.mark.unit
def test_conformal_falls_back_when_table_missing(monkeypatch):
    _patch_conformal(monkeypatch, [_frame()] * 3, exists=False)
    assert ml._compute_conformal_q_hat.__wrapped__() == 2.9


@pytest.mark.unit
def test_conformal_falls_back_with_too_few_samples(monkeypatch):
    _patch_conformal(monkeypatch, [_frame(10)] * 3)   # 30 errors < 50
    assert ml._compute_conformal_q_hat.__wrapped__() == 2.9


@pytest.mark.unit
def test_conformal_falls_back_on_any_error(monkeypatch):
    _patch_conformal(monkeypatch, [])   # frames[i] -> IndexError inside the try
    assert ml._compute_conformal_q_hat.__wrapped__() == 2.9


@pytest.mark.unit
def test_conformal_is_90th_percentile_of_absolute_errors(monkeypatch):
    # model predicts 0 everywhere, so |error| == sofa_score; 3 identical clients of 0..99
    _patch_conformal(monkeypatch, [_frame()] * 3)
    expected = round(float(np.quantile(np.tile(np.arange(100.0), 3), 0.90)), 2)
    assert ml._compute_conformal_q_hat.__wrapped__() == expected


@pytest.mark.unit
def test_conformal_skips_empty_and_unlabelled_clients(monkeypatch):
    frames = [pd.DataFrame(), pd.DataFrame({"a": [1.0], "b": [1.0]}), _frame()]
    _patch_conformal(monkeypatch, frames)
    # only client 2 contributes: 100 errors
    assert ml._compute_conformal_q_hat.__wrapped__() == round(float(np.quantile(np.arange(100.0), 0.90)), 2)


@pytest.mark.unit
def test_conformal_zero_fills_missing_feature_columns(monkeypatch):
    # model expects columns a, b, c but the data has no "c" -> must not crash
    _patch_conformal(monkeypatch, [_frame()] * 3, cols=("a", "b", "c"))
    assert ml._compute_conformal_q_hat.__wrapped__() > 2.9


# ── run_prediction against a temporary database ───────────────────────────────

@pytest.mark.unit
def test_run_prediction_raises_when_patient_has_no_sample_rows(tmp_db):
    with pytest.raises(ValueError, match="No sample data for patient 1"):
        ml.run_prediction(1, 0)


@pytest.mark.unit
def test_run_prediction_seeds_the_vitals_window_when_empty(db_with_samples):
    assert len(db_with_samples.get_patient_vitals(1, limit=50)) == 0
    ml.run_prediction(1, 0)
    # 19 seeded history rows (indices 1..19) + the current reading
    assert len(db_with_samples.get_patient_vitals(1, limit=50)) == 20


@pytest.mark.unit
def test_run_prediction_does_not_reseed_when_vitals_exist(db_with_samples):
    ml.run_prediction(1, 0)
    ml.run_prediction(1, 1)
    # window is capped at 20; a second seed would not change that, so check the history table instead
    assert len(db_with_samples.get_prediction_history(1, limit=50)) == 2


@pytest.mark.unit
def test_run_prediction_wraps_row_index(db_with_samples):
    out = ml.run_prediction(1, 27)          # 27 % 25 == 2
    assert out["cur_idx"] == 2 and out["total_rows"] == 25 and out["row_idx"] == 27


@pytest.mark.unit
def test_run_prediction_uses_dataset_camera_values_by_default(db_with_samples):
    out = ml.run_prediction(1, 0)
    assert out["vitals"]["GCS_eye"] == 4
    assert out["vitals"]["stress"] == 0.5


@pytest.mark.unit
def test_run_prediction_live_camera_values_override_the_dataset(db_with_samples):
    out = ml.run_prediction(1, 0, cv_gcs_eye=1, cv_stress=9.0)
    assert out["vitals"]["GCS_eye"] == 1
    assert out["vitals"]["stress"] == 9.0
    assert "GCS Eye Opening: 1" in out["_cv_block"]


@pytest.mark.unit
def test_run_prediction_camera_value_changes_the_model_input(db_with_samples):
    base = ml.run_prediction(1, 0)["sofa_score"]
    live = ml.run_prediction(1, 0, cv_gcs_eye=1, cv_stress=10.0)["sofa_score"]
    assert base != live


@pytest.mark.unit
def test_run_prediction_logs_the_prediction_with_camera_inputs(db_with_samples):
    ml.run_prediction(1, 0, cv_gcs_eye=2, cv_stress=7.0)
    hist = db_with_samples.get_prediction_history(1, limit=5)
    assert len(hist) == 1
    assert hist.iloc[0]["Stress"] == 7.0


@pytest.mark.unit
def test_run_prediction_result_shape(db_with_samples):
    out = ml.run_prediction(1, 0)
    assert 0.0 <= out["sofa_score"] <= 24.0
    assert out["conformal_lo"] <= out["sofa_score"] <= out["conformal_hi"]
    assert out["is_alert"] == (out["sofa_score"] >= out["alert_threshold"])
    assert len(out["shap_top5"]) == 5
    assert set(out["vitals_status"]) == {"HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"}
    assert len(out["vitals_history"]) == 20
    assert out["clinical_note"].startswith("septic")


@pytest.mark.unit
def test_run_prediction_second_call_reports_trajectory_fields(db_with_samples):
    ml.run_prediction(1, 0)
    out = ml.run_prediction(1, 1)
    assert "_traj_flagged" in out and "_traj_msg" in out


# ── SHAP result shapes ────────────────────────────────────────────────────────

_IMPACT = np.arange(108, dtype=float) / 10.0


@pytest.mark.unit
@pytest.mark.parametrize("shape_name,value", [
    ("list", [_IMPACT.reshape(1, -1)]),
    ("2d-array", _IMPACT.reshape(1, -1)),
    ("1d-array", _IMPACT),
])
def test_run_prediction_accepts_every_shap_output_shape(db_with_samples, monkeypatch, shape_name, value):
    monkeypatch.setattr(ml, "_load_shap_explainer", lambda: _FakeExplainer(value))
    out = ml.run_prediction(1, 0)
    assert len(out["shap_top5"]) == 5, shape_name
    assert out["clinical_explanations"], shape_name


@pytest.mark.unit
def test_run_prediction_survives_a_shap_failure(db_with_samples, monkeypatch):
    monkeypatch.setattr(ml, "_load_shap_explainer",
                        lambda: _FakeExplainer(error=RuntimeError("shap exploded")))
    out = ml.run_prediction(1, 0)
    assert out["shap_top5"] == []
    assert out["key_risks"] == []
    assert out["sofa_score"] >= 0.0


@pytest.mark.unit
def test_run_prediction_works_without_any_explainer(db_with_samples, monkeypatch):
    monkeypatch.setattr(ml, "_load_shap_explainer", lambda: None)
    out = ml.run_prediction(1, 0)
    assert out["shap_top5"] == []
    assert "SHAP analysis not available" in out["_explanation_block"]


# ── Feature columns missing from the assembled input ──────────────────────────

class _IdentityScaler:
    def transform(self, df):
        return np.asarray(df, dtype=float)


@pytest.mark.unit
def test_run_prediction_zero_fills_feature_columns_it_cannot_build(db_with_samples, monkeypatch):
    from model_utils import ICUModel
    _, _, tfidf, cols, _ = ml.load_artifacts()
    cols = list(cols) + ["column_the_pipeline_never_makes"]
    model = ICUModel(len(cols))
    monkeypatch.setattr(ml, "load_artifacts", lambda: (model, _IdentityScaler(), tfidf, cols, None))
    monkeypatch.setattr(ml, "_load_shap_explainer", lambda: None)
    monkeypatch.setattr(ml, "_compute_conformal_q_hat", lambda: 3.0)
    out = ml.run_prediction(1, 0)
    assert 0.0 <= out["sofa_score"] <= 24.0
    assert out["conformal_q"] == 3.0


# ── sys.path bootstrap lines (re-executed by reloading) ───────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("module_name", [
    "backend.main",
    "backend.api.history",
    "backend.api.vitals",
    "backend.services.llm_service",
    "backend.services.ml_service",
])
def test_module_adds_project_root_to_sys_path_when_missing(module_name):
    """Each backend module inserts the project root into sys.path if it is not there,
    so `from db import ...` works however the app is launched."""
    # ml_service.py lives in <root>/backend/services/, so go up three levels
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ml.__file__))))
    saved = list(sys.path)
    try:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != root]
        assert root not in sys.path
        importlib.reload(importlib.import_module(module_name))
        assert root in sys.path
    finally:
        sys.path[:] = saved

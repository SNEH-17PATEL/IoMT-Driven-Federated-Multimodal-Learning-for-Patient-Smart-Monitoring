"""
Unit tests for pure helper functions in ml_service.py.
ML model loading is NOT triggered here — only isolated utility functions are tested.
"""
import os
import sys
import re
import importlib
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_UNIT_DIR, "../../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Import only the pure constants/functions we need ─────────────────────────
# We import specific names to avoid triggering lru_cache'd model loads at
# module level (those are only called lazily via run_prediction).

from backend.services.ml_service import (
    _is_clinical,
    _VITAL_ABBR,
    VITAL_UNITS,
    CLINICAL_KEYWORDS,
    NORMAL_RANGES,
    ALERT_THRESHOLD,
    RISK_MAP,
    load_training_metadata,
)


# ── _is_clinical — VITAL_UNITS path ──────────────────────────────────────────

@pytest.mark.unit
def test_is_clinical_vital_units_hr_mean():
    assert _is_clinical("HR_mean") is True


@pytest.mark.unit
def test_is_clinical_vital_units_latest_spo2():
    assert _is_clinical("latest_SpO2") is True


@pytest.mark.unit
def test_is_clinical_vital_units_gcs_eye():
    assert _is_clinical("GCS_eye_opening") is True


@pytest.mark.unit
def test_is_clinical_vital_units_stress_score():
    assert _is_clinical("stress_score") is True


# ── _is_clinical — CLINICAL_KEYWORDS path ─────────────────────────────────────

@pytest.mark.unit
def test_is_clinical_keyword_septic():
    assert _is_clinical("septic_shock_flag") is True


@pytest.mark.unit
def test_is_clinical_keyword_respiratory():
    assert _is_clinical("respiratory_distress") is True


@pytest.mark.unit
def test_is_clinical_keyword_vasopressor():
    assert _is_clinical("vasopressor_use") is True


# ── _is_clinical — Non-clinical features ─────────────────────────────────────

@pytest.mark.unit
def test_is_clinical_weight_false():
    assert _is_clinical("weight") is False


@pytest.mark.unit
def test_is_clinical_age_false():
    assert _is_clinical("age") is False


@pytest.mark.unit
def test_is_clinical_glucose_false():
    # 'glucose' is not in CLINICAL_KEYWORDS and not in VITAL_UNITS
    assert _is_clinical("glucose") is False


# ── _is_clinical — Boundary matching guards (C3 fix) ─────────────────────────

@pytest.mark.unit
def test_is_clinical_epinephrine_does_not_match_hr():
    # 'hr' is a substring of 'epinephrine' — must NOT trigger "Abnormal heart rate"
    result = _is_clinical("epinephrine")
    # epinephrine is not in VITAL_UNITS and boundary check must fail for 'hr'
    assert result is False


@pytest.mark.unit
def test_is_clinical_norepinephrine_does_not_match_hr():
    assert _is_clinical("norepinephrine") is False


@pytest.mark.unit
def test_is_clinical_thrombocytopenia_does_not_match_hr():
    assert _is_clinical("thrombocytopenia") is False


@pytest.mark.unit
def test_is_clinical_cirrhosis_does_not_match_rr():
    assert _is_clinical("cirrhosis") is False


@pytest.mark.unit
def test_is_clinical_crrt_does_not_match_rr():
    assert _is_clinical("crrt") is False


@pytest.mark.unit
def test_is_clinical_hemorrhage_does_not_match_rr():
    assert _is_clinical("hemorrhage") is False


@pytest.mark.unit
def test_is_clinical_latest_hr_matches_hr():
    # "latest_hr" — the underscore is a non-letter boundary so hr SHOULD match
    assert _is_clinical("latest_hr") is True


@pytest.mark.unit
def test_is_clinical_trend_rr_matches_rr():
    # "trend_rr" — underscore before 'rr' is a non-letter boundary
    assert _is_clinical("trend_rr") is True


# ── _VITAL_ABBR completeness ──────────────────────────────────────────────────

@pytest.mark.unit
def test_vital_abbr_contains_hr():
    assert "hr" in _VITAL_ABBR


@pytest.mark.unit
def test_vital_abbr_contains_rr():
    assert "rr" in _VITAL_ABBR


@pytest.mark.unit
def test_vital_abbr_contains_map():
    assert "map" in _VITAL_ABBR


@pytest.mark.unit
def test_vital_abbr_contains_sbp():
    assert "sbp" in _VITAL_ABBR


@pytest.mark.unit
def test_vital_abbr_contains_dbp():
    assert "dbp" in _VITAL_ABBR


@pytest.mark.unit
def test_vital_abbr_contains_gcs():
    assert "gcs" in _VITAL_ABBR


@pytest.mark.unit
def test_vital_abbr_contains_spo2():
    assert "spo2" in _VITAL_ABBR


@pytest.mark.unit
def test_vital_abbr_has_no_extras():
    # Exactly these 7 abbreviations — no more, no less
    expected = {"hr", "rr", "map", "sbp", "dbp", "gcs", "spo2"}
    assert _VITAL_ABBR == expected


# ── NORMAL_RANGES completeness ────────────────────────────────────────────────

@pytest.mark.unit
def test_normal_ranges_has_7_vitals():
    expected = {"HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"}
    assert expected.issubset(NORMAL_RANGES.keys())


@pytest.mark.unit
def test_normal_ranges_hr_bounds():
    lo, hi = NORMAL_RANGES["HR"]
    assert lo == 60 and hi == 100


@pytest.mark.unit
def test_normal_ranges_spo2_bounds():
    lo, hi = NORMAL_RANGES["SpO2"]
    assert lo == 95 and hi == 100


# ── ALERT_THRESHOLD ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_alert_threshold_is_8():
    assert ALERT_THRESHOLD == 8.0


# ── load_training_metadata ────────────────────────────────────────────────────

@pytest.mark.unit
def test_load_training_metadata_returns_dict():
    meta = load_training_metadata()
    assert isinstance(meta, dict)


@pytest.mark.unit
def test_load_training_metadata_final_mae():
    meta = load_training_metadata()
    assert meta["final_mae"] == pytest.approx(1.8968, rel=1e-3)


@pytest.mark.unit
def test_load_training_metadata_final_r2():
    meta = load_training_metadata()
    assert meta["final_r2"] == pytest.approx(0.5995, rel=1e-3)


@pytest.mark.unit
def test_load_training_metadata_test_samples():
    meta = load_training_metadata()
    assert meta["test_samples"] == 2559


@pytest.mark.unit
def test_load_training_metadata_train_samples():
    meta = load_training_metadata()
    assert meta["train_samples"] == 12795


@pytest.mark.unit
def test_load_training_metadata_has_best_round():
    meta = load_training_metadata()
    assert "best_round" in meta
    assert isinstance(meta["best_round"], int)


# ── Seed timestamp fix (C7) ───────────────────────────────────────────────────

@pytest.mark.unit
def test_seed_timestamps_no_hardcoded_date():
    """Seed rows must use relative timestamps, not the old hard-coded 2026-08-29."""
    import backend.services.ml_service as ml
    import inspect
    src = inspect.getsource(ml)
    assert "2026-08-29" not in src, "Hard-coded seed date '2026-08-29' found in ml_service.py"


@pytest.mark.unit
def test_seed_timestamps_use_timedelta():
    """Seed rows must be generated with timedelta-based offsets."""
    import backend.services.ml_service as ml
    import inspect
    src = inspect.getsource(ml)
    assert "timedelta" in src, "Seed timestamps should use timedelta"


@pytest.mark.unit
def test_seed_loop_starts_from_index_1():
    """Seed loop must start from index 1 to avoid duplicating row 0."""
    import backend.services.ml_service as ml
    import inspect
    src = inspect.getsource(ml)
    # Should see range(1, min(... not range(0, min(
    assert "range(1, min(" in src, "Seed loop should start from 1, not 0"


# ── RISK_MAP keys ─────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_risk_map_has_gcs_key():
    assert "GCS" in RISK_MAP


@pytest.mark.unit
def test_risk_map_has_hr_key():
    assert "HR" in RISK_MAP


@pytest.mark.unit
def test_risk_map_values_are_strings():
    for k, v in RISK_MAP.items():
        assert isinstance(v, str), f"RISK_MAP[{k!r}] value must be str, got {type(v)}"

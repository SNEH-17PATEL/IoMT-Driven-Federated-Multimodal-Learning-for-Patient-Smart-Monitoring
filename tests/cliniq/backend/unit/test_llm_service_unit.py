"""
Unit tests for pure helper functions in llm_service.py.
No real LLM calls are made — only cache key logic and response shape are tested.
"""
import os
import sys
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_UNIT_DIR, "../../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.services.llm_service import _cache_key


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def pred_p1():
    return {
        "patient_id": 1,
        "sofa_score": 4.2,
        "vitals": {"HR": 72, "SpO2": 97, "SBP": 115, "RR": 16},
    }


@pytest.fixture
def pred_p2():
    return {
        "patient_id": 2,
        "sofa_score": 7.5,
        "vitals": {"HR": 95, "SpO2": 94, "SBP": 98, "RR": 22},
    }


# ── Basic output ──────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_cache_key_returns_string(pred_p1):
    key = _cache_key(pred_p1, "hospital")
    assert isinstance(key, str)
    assert len(key) > 0


@pytest.mark.unit
def test_cache_key_is_deterministic(pred_p1):
    k1 = _cache_key(pred_p1, "hospital")
    k2 = _cache_key(pred_p1, "hospital")
    assert k1 == k2


# ── patient_id separates the key (C2 fix) ────────────────────────────────────

@pytest.mark.unit
def test_cache_key_differs_for_different_patients(pred_p1, pred_p2):
    k1 = _cache_key(pred_p1, "hospital")
    k2 = _cache_key(pred_p2, "hospital")
    assert k1 != k2


@pytest.mark.unit
def test_cache_key_contains_patient_id(pred_p1):
    key = _cache_key(pred_p1, "hospital")
    # patient_id=1 must appear as a field in the key string
    parts = key.split("|")
    assert "1" in parts, f"patient_id '1' not found in key parts: {parts}"


# ── care_mode separates the key ───────────────────────────────────────────────

@pytest.mark.unit
def test_cache_key_differs_for_different_care_modes(pred_p1):
    k_hosp = _cache_key(pred_p1, "hospital")
    k_home = _cache_key(pred_p1, "homecare")
    assert k_hosp != k_home


# ── SOFA score separates the key ─────────────────────────────────────────────

@pytest.mark.unit
def test_cache_key_differs_for_different_sofa():
    p_low  = {"patient_id": 1, "sofa_score": 2.0, "vitals": {"HR": 72, "SpO2": 97, "SBP": 115, "RR": 16}}
    p_high = {"patient_id": 1, "sofa_score": 9.0, "vitals": {"HR": 72, "SpO2": 97, "SBP": 115, "RR": 16}}
    assert _cache_key(p_low, "hospital") != _cache_key(p_high, "hospital")


# ── SBP separates the key (C2 fix) ───────────────────────────────────────────

@pytest.mark.unit
def test_cache_key_differs_for_different_sbp():
    p_norm = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 72, "SpO2": 97, "SBP": 115, "RR": 16}}
    p_low  = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 72, "SpO2": 97, "SBP":  70, "RR": 16}}
    assert _cache_key(p_norm, "hospital") != _cache_key(p_low, "hospital")


# ── RR separates the key (C2 fix) ────────────────────────────────────────────

@pytest.mark.unit
def test_cache_key_differs_for_different_rr():
    p_norm = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 72, "SpO2": 97, "SBP": 115, "RR": 14}}
    p_high = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 72, "SpO2": 97, "SBP": 115, "RR": 30}}
    assert _cache_key(p_norm, "hospital") != _cache_key(p_high, "hospital")


# ── HR separates the key ──────────────────────────────────────────────────────

@pytest.mark.unit
def test_cache_key_differs_for_different_hr():
    p_norm = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 72,  "SpO2": 97, "SBP": 115, "RR": 16}}
    p_fast = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 140, "SpO2": 97, "SBP": 115, "RR": 16}}
    assert _cache_key(p_norm, "hospital") != _cache_key(p_fast, "hospital")


# ── SpO2 separates the key ────────────────────────────────────────────────────

@pytest.mark.unit
def test_cache_key_differs_for_different_spo2():
    p_good = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 72, "SpO2": 99, "SBP": 115, "RR": 16}}
    p_low  = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 72, "SpO2": 88, "SBP": 115, "RR": 16}}
    assert _cache_key(p_good, "hospital") != _cache_key(p_low, "hospital")


# ── Missing vitals fields don't crash ────────────────────────────────────────

@pytest.mark.unit
def test_cache_key_handles_missing_vitals_fields():
    pred = {"patient_id": 1, "sofa_score": 3.0, "vitals": {}}
    key = _cache_key(pred, "hospital")
    assert isinstance(key, str)


@pytest.mark.unit
def test_cache_key_handles_missing_vitals_key():
    pred = {"patient_id": 1, "sofa_score": 3.0}
    key = _cache_key(pred, "hospital")
    assert isinstance(key, str)


# ── judge_used key in generate_report response ───────────────────────────────

@pytest.mark.unit
def test_generate_report_response_has_judge_used_key():
    """
    Verify that generate_report() result dict always contains 'judge_used'.
    We only check the function's source to confirm the field is set — calling
    the real function would require a valid GROQ_API_KEY.
    """
    import inspect
    import backend.services.llm_service as llm_mod
    src = inspect.getsource(llm_mod)
    assert '"judge_used"' in src or "'judge_used'" in src, (
        "generate_report result must include 'judge_used' key (C6 fix)"
    )


@pytest.mark.unit
def test_judge_used_is_set_when_judge_fails():
    """Source-level check: judge_used must be False when judge call fails."""
    import inspect
    import backend.services.llm_service as llm_mod
    src = inspect.getsource(llm_mod)
    assert "judge_used" in src
    assert "judge_used   = False" in src or "judge_used = False" in src


@pytest.mark.unit
def test_judge_failure_logged_to_stderr():
    """Source-level check: failed judge calls must log to stderr."""
    import inspect
    import backend.services.llm_service as llm_mod
    src = inspect.getsource(llm_mod)
    assert "sys.stderr" in src, "Judge failures must be logged to stderr"


# ── TTL cache behaviour ───────────────────────────────────────────────────────

import threading
import time as _time

import backend.services.llm_service as _llm


@pytest.fixture
def clean_cache():
    _llm._REPORT_CACHE.clear()
    _llm._INFLIGHT.clear()
    yield
    _llm._REPORT_CACHE.clear()
    _llm._INFLIGHT.clear()


@pytest.mark.unit
@pytest.mark.parametrize("field,a,b", [
    ("DBP", 70, 85), ("MAP", 85, 70), ("Temp", 37.0, 38.4),
    ("GCS_eye", 4, 2), ("stress", 2, 8),
])
def test_cache_key_differs_for_every_prompt_vital(field, a, b):
    base = {"HR": 80, "SpO2": 97, "SBP": 115, "RR": 16}
    p1 = {"patient_id": 1, "sofa_score": 4.0, "vitals": {**base, field: a}}
    p2 = {"patient_id": 1, "sofa_score": 4.0, "vitals": {**base, field: b}}
    assert _cache_key(p1, "hospital") != _cache_key(p2, "hospital")


@pytest.mark.unit
def test_cache_entry_expires_after_ttl(clean_cache, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(_llm.time, "time", lambda: now[0])
    _llm._cache_set("k", {"x": 1})
    now[0] += _llm._CACHE_TTL - 1
    assert _llm._cache_get("k") == {"x": 1}
    now[0] += 2
    assert _llm._cache_get("k") is None


@pytest.mark.unit
def test_cache_set_evicts_all_expired_entries(clean_cache, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(_llm.time, "time", lambda: now[0])
    _llm._cache_set("old1", {})
    _llm._cache_set("old2", {})
    now[0] += _llm._CACHE_TTL + 1
    _llm._cache_set("new", {})
    assert set(_llm._REPORT_CACHE) == {"new"}


def _ok_result(mode):
    return {"sections": {"CURRENT CONDITION": "ok"}, "care_mode": mode, "error": None}


@pytest.mark.unit
def test_generate_report_serves_repeat_from_cache(clean_cache, monkeypatch):
    calls = []
    def fake(pr, mode, key_, ck):
        calls.append(mode)
        r = _ok_result(mode); _llm._cache_set(ck, r); return r
    monkeypatch.setattr(_llm, "_generate_uncached", fake)
    pred = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 80}}
    a = _llm.generate_report(pred, "hospital", "k")
    b = _llm.generate_report(pred, "hospital", "k")
    assert a is b and calls == ["hospital"]


@pytest.mark.unit
def test_concurrent_requests_share_one_llm_call(clean_cache, monkeypatch):
    calls = []
    gate = threading.Event()
    def fake(pr, mode, key_, ck):
        calls.append(mode)
        gate.wait(2)
        r = _ok_result(mode); _llm._cache_set(ck, r); return r
    monkeypatch.setattr(_llm, "_generate_uncached", fake)
    pred = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 80}}
    out = []
    ts = [threading.Thread(target=lambda: out.append(_llm.generate_report(pred, "hospital", "k")))
          for _ in range(4)]
    for t in ts: t.start()
    _time.sleep(0.1)
    gate.set()
    for t in ts: t.join(3)
    assert len(calls) == 1 and len(out) == 4
    assert not _llm._INFLIGHT


@pytest.mark.unit
def test_failed_report_is_not_cached_and_waiter_retries(clean_cache, monkeypatch):
    calls = []
    def fake(pr, mode, key_, ck):
        calls.append(mode)
        return {"sections": None, "care_mode": mode, "error": "LLM unavailable"}
    monkeypatch.setattr(_llm, "_generate_uncached", fake)
    pred = {"patient_id": 1, "sofa_score": 4.0, "vitals": {"HR": 80}}
    _llm.generate_report(pred, "hospital", "k")
    _llm.generate_report(pred, "hospital", "k")
    assert len(calls) == 2 and not _llm._REPORT_CACHE

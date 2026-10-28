"""Tests for POST /api/report."""
import pytest

REQUIRED_KEYS     = {"sections", "reliability_score", "reliability_label", "reliability_color"}
REQUIRED_SECTIONS = {"CURRENT CONDITION", "PROBABLE CAUSE", "RISK FORECAST", "IMMEDIATE ACTIONS"}


def _report(client, predict_result, patient_id=1, care_mode="hospital"):
    return client.post("/api/report", json={
        "patient_id":     patient_id,
        "care_mode":      care_mode,
        "predict_result": predict_result,
    })


# ── Happy path ────────────────────────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.slow
def test_report_hospital_mode_returns_200(client, predict_p1):
    r = _report(client, predict_p1)
    assert r.status_code == 200, r.text


@pytest.mark.api
@pytest.mark.slow
def test_report_required_keys_present(client, predict_p1):
    body = _report(client, predict_p1).json()
    assert REQUIRED_KEYS.issubset(body.keys()), f"Missing keys: {REQUIRED_KEYS - set(body.keys())}"


@pytest.mark.api
@pytest.mark.slow
def test_report_all_sections_present(client, predict_p1):
    sections = _report(client, predict_p1).json()["sections"]
    assert REQUIRED_SECTIONS.issubset(set(sections.keys()))


@pytest.mark.api
@pytest.mark.slow
def test_report_sections_have_content(client, predict_p1):
    sections = _report(client, predict_p1).json()["sections"]
    for key in REQUIRED_SECTIONS:
        assert isinstance(sections[key], str), f"Section {key!r} is not a string"
        assert len(sections[key]) > 20, f"Section {key!r} suspiciously short"


@pytest.mark.api
@pytest.mark.slow
def test_report_reliability_score_in_range(client, predict_p1):
    score = _report(client, predict_p1).json()["reliability_score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.api
@pytest.mark.slow
def test_report_reliability_label_valid(client, predict_p1):
    label = _report(client, predict_p1).json()["reliability_label"]
    assert label in {"High", "Moderate", "Low"}


@pytest.mark.api
@pytest.mark.slow
def test_report_homecare_mode_returns_200(client, predict_p1):
    r = _report(client, predict_p1, care_mode="homecare")
    assert r.status_code == 200


@pytest.mark.api
@pytest.mark.slow
def test_report_homecare_all_sections_present(client, predict_p1):
    sections = _report(client, predict_p1, care_mode="homecare").json()["sections"]
    assert REQUIRED_SECTIONS.issubset(set(sections.keys()))


@pytest.mark.api
@pytest.mark.slow
def test_report_judge_used_key_present(client, predict_p1):
    """C6 fix: judge_used must always be in the response."""
    body = _report(client, predict_p1).json()
    assert "judge_used" in body, "judge_used key missing from report response"


@pytest.mark.api
@pytest.mark.slow
def test_report_judge_used_is_bool(client, predict_p1):
    body = _report(client, predict_p1).json()
    assert isinstance(body["judge_used"], bool)


# ── Validation errors — no LLM call needed ────────────────────────────────────

@pytest.mark.api
def test_report_invalid_care_mode_returns_400(client, predict_p1):
    r = _report(client, predict_p1, care_mode="icu_mode")
    assert r.status_code == 400


@pytest.mark.api
def test_report_missing_care_mode_returns_422(client, predict_p1):
    r = client.post("/api/report", json={
        "patient_id": 1, "predict_result": predict_p1
    })
    assert r.status_code == 422


@pytest.mark.api
def test_report_invalid_patient_id_returns_400(client, predict_p1):
    r = _report(client, predict_p1, patient_id=4)
    assert r.status_code == 400


@pytest.mark.api
def test_report_patient_zero_returns_400(client, predict_p1):
    r = _report(client, predict_p1, patient_id=0)
    assert r.status_code == 400


@pytest.mark.api
def test_report_predict_result_patient_id_mismatch_returns_400(client, predict_p1):
    """C5 fix: predict_result.patient_id must match patient_id in body."""
    r = _report(client, predict_p1, patient_id=2)  # p1 data but pid=2
    assert r.status_code == 400


@pytest.mark.api
def test_report_missing_vitals_returns_400(client, predict_p1):
    """C5 fix: predict_result.vitals must be a dict."""
    bad = {**predict_p1, "vitals": "not_a_dict"}
    r = _report(client, bad)
    assert r.status_code == 400


@pytest.mark.api
def test_report_missing_sofa_score_returns_400(client, predict_p1):
    """C5 fix: predict_result.sofa_score is required."""
    bad = {k: v for k, v in predict_p1.items() if k != "sofa_score"}
    r = _report(client, bad)
    assert r.status_code == 400


@pytest.mark.api
def test_report_empty_body_returns_422(client):
    assert client.post("/api/report", json={}).status_code == 422

import pytest

REQUIRED_KEYS     = {"sections", "reliability_score", "reliability_label", "reliability_color"}
REQUIRED_SECTIONS = {"CURRENT CONDITION", "PROBABLE CAUSE", "RISK FORECAST", "IMMEDIATE ACTIONS"}


@pytest.mark.slow
def test_report_hospital_mode_shape(client, predict_p1):
    r = client.post("/api/report", json={
        "patient_id":     predict_p1["patient_id"],
        "care_mode":      "hospital",
        "predict_result": predict_p1,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert REQUIRED_KEYS.issubset(body.keys())
    assert REQUIRED_SECTIONS.issubset(set(body["sections"].keys()))


@pytest.mark.slow
def test_report_sections_have_content(client, predict_p1):
    r = client.post("/api/report", json={
        "patient_id":     predict_p1["patient_id"],
        "care_mode":      "hospital",
        "predict_result": predict_p1,
    })
    sections = r.json()["sections"]
    for key in REQUIRED_SECTIONS:
        assert isinstance(sections[key], str), f"Section {key} is not a string"
        assert len(sections[key]) > 20, f"Section {key} suspiciously short"


@pytest.mark.slow
def test_report_reliability_score_in_range(client, predict_p1):
    r = client.post("/api/report", json={
        "patient_id":     predict_p1["patient_id"],
        "care_mode":      "hospital",
        "predict_result": predict_p1,
    })
    score = r.json()["reliability_score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.slow
def test_report_homecare_mode(client, predict_p1):
    r = client.post("/api/report", json={
        "patient_id":     predict_p1["patient_id"],
        "care_mode":      "homecare",
        "predict_result": predict_p1,
    })
    assert r.status_code == 200
    assert REQUIRED_SECTIONS.issubset(set(r.json()["sections"].keys()))


# ── Validation errors (no LLM call needed) ────────────────────────────────────

def test_report_invalid_care_mode_returns_400(client, predict_p1):
    r = client.post("/api/report", json={
        "patient_id":     predict_p1["patient_id"],
        "care_mode":      "icu_mode",
        "predict_result": predict_p1,
    })
    assert r.status_code == 400


def test_report_missing_body_returns_422(client):
    assert client.post("/api/report", json={}).status_code == 422

"""
Error-path tests for the FastAPI routes.

The happy paths are covered in the other test_api_*.py files. These tests reach
the branches that only run when something goes wrong: bad camera values, an
unexpected exception inside a service, a missing API key.

Nothing here calls the real database writer, the model, or Groq — the functions
the routes call are replaced with small fakes using monkeypatch.
"""
import pytest

import backend.api.history as history_api
import backend.api.predict as predict_api
import backend.api.report as report_api
import backend.api.vitals as vitals_api


@pytest.fixture(autouse=True)
def _restore_last_predict():
    """/api/predict stores its result in a module-level dict that /api/report reads.
    Several tests below put fake results in it, so save and restore it around every
    test — otherwise later report tests would read these fakes instead of real data."""
    saved = dict(predict_api._LAST_PREDICT)
    yield
    predict_api._LAST_PREDICT.clear()
    predict_api._LAST_PREDICT.update(saved)


# ── POST /api/predict ─────────────────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("bad_gcs", [0, 5, -1, 99])
def test_predict_rejects_gcs_eye_outside_1_to_4(client, bad_gcs):
    r = client.post("/api/predict", json={"patient_id": 1, "row_idx": 0, "cv_gcs_eye": bad_gcs})
    assert r.status_code == 400
    assert "cv_gcs_eye" in r.json()["detail"]


@pytest.mark.api
@pytest.mark.parametrize("bad_stress", [-0.1, 10.1, 11, -5])
def test_predict_rejects_stress_outside_0_to_10(client, bad_stress):
    r = client.post("/api/predict", json={"patient_id": 1, "row_idx": 0, "cv_stress": bad_stress})
    assert r.status_code == 400
    assert "cv_stress" in r.json()["detail"]


@pytest.mark.api
def test_predict_rejects_unknown_patient(client):
    r = client.post("/api/predict", json={"patient_id": 9, "row_idx": 0})
    assert r.status_code == 400
    assert "patient_id" in r.json()["detail"]


@pytest.mark.api
def test_predict_service_value_error_becomes_400(client, monkeypatch):
    def boom(*a, **k):
        raise ValueError("No sample data for patient 1")
    monkeypatch.setattr(predict_api, "run_prediction", boom)
    r = client.post("/api/predict", json={"patient_id": 1, "row_idx": 0})
    assert r.status_code == 400
    assert "No sample data" in r.json()["detail"]


@pytest.mark.api
def test_predict_unexpected_error_becomes_500_without_leaking_details(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(predict_api, "run_prediction", boom)
    r = client.post("/api/predict", json={"patient_id": 1, "row_idx": 0})
    assert r.status_code == 500
    assert "secret internal detail" not in r.text
    assert "Prediction failed" in r.json()["detail"]


@pytest.mark.api
def test_predict_passes_camera_values_to_service(client, monkeypatch):
    seen = {}

    def fake(patient_id, row_idx, cv_gcs_eye=None, cv_stress=None):
        seen.update(patient_id=patient_id, row_idx=row_idx, gcs=cv_gcs_eye, stress=cv_stress)
        return {"sofa_score": 3.0, "_vitals_block": "x", "_cv_block": "x",
                "_risk_block": "x", "_explanation_block": "x"}

    monkeypatch.setattr(predict_api, "run_prediction", fake)
    r = client.post("/api/predict",
                    json={"patient_id": 2, "row_idx": 4, "cv_gcs_eye": 3, "cv_stress": 6.5})
    assert r.status_code == 200
    assert seen == {"patient_id": 2, "row_idx": 4, "gcs": 3, "stress": 6.5}


@pytest.mark.api
def test_predict_response_strips_llm_prompt_fields(client, monkeypatch):
    monkeypatch.setattr(predict_api, "run_prediction", lambda *a, **k: {
        "sofa_score": 3.0, "_vitals_block": "v", "_cv_block": "c",
        "_risk_block": "r", "_explanation_block": "e", "keep": 1,
    })
    body = client.post("/api/predict", json={"patient_id": 1, "row_idx": 0}).json()
    assert body == {"sofa_score": 3.0, "keep": 1}


@pytest.mark.api
def test_predict_stores_full_result_for_report(client, monkeypatch):
    monkeypatch.setattr(predict_api, "run_prediction", lambda *a, **k: {
        "sofa_score": 3.0, "_vitals_block": "stored-for-report",
    })
    client.post("/api/predict", json={"patient_id": 3, "row_idx": 0})
    assert predict_api._LAST_PREDICT[3]["_vitals_block"] == "stored-for-report"


# ── GET /api/history and /api/vitals ──────────────────────────────────────────

@pytest.mark.api
def test_history_database_error_becomes_500(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk gone")
    monkeypatch.setattr(history_api, "get_prediction_history", boom)
    r = client.get("/api/history/1")
    assert r.status_code == 500
    assert "disk gone" not in r.text
    assert "History query failed" in r.json()["detail"]


@pytest.mark.api
def test_vitals_database_error_becomes_500(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk gone")
    monkeypatch.setattr(vitals_api, "get_patient_vitals", boom)
    r = client.get("/api/vitals/1")
    assert r.status_code == 500
    assert "disk gone" not in r.text
    assert "Vitals query failed" in r.json()["detail"]


# ── POST /api/report ──────────────────────────────────────────────────────────

@pytest.fixture
def stored_prediction(monkeypatch):
    """Put a fake prediction in the server-side store so /api/report gets past the 409 check."""
    monkeypatch.setitem(predict_api._LAST_PREDICT, 1, {"sofa_score": 3.0, "vitals": {}})


@pytest.mark.api
def test_report_without_api_key_returns_500(client, monkeypatch, stored_prediction):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = client.post("/api/report", json={"patient_id": 1, "care_mode": "hospital"})
    assert r.status_code == 500
    assert "GROQ_API_KEY" in r.json()["detail"]


@pytest.mark.api
def test_report_generation_failure_becomes_500(client, monkeypatch, stored_prediction):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    def boom(*a, **k):
        raise RuntimeError("groq exploded")
    monkeypatch.setattr(report_api, "generate_report", boom)
    r = client.post("/api/report", json={"patient_id": 1, "care_mode": "hospital"})
    assert r.status_code == 500
    assert "groq exploded" not in r.text
    assert "Report generation failed" in r.json()["detail"]


@pytest.mark.api
def test_report_success_returns_service_result(client, monkeypatch, stored_prediction):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(report_api, "generate_report",
                        lambda pr, mode, key: {"care_mode": mode, "ok": True})
    r = client.post("/api/report", json={"patient_id": 1, "care_mode": "homecare"})
    assert r.status_code == 200
    assert r.json() == {"care_mode": "homecare", "ok": True}


@pytest.mark.api
def test_report_passes_the_servers_own_prediction_not_client_data(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setitem(predict_api._LAST_PREDICT, 2, {"marker": "server-copy"})
    captured = {}

    def fake(pr, mode, key):
        captured["pr"] = pr
        return {"ok": True}

    monkeypatch.setattr(report_api, "generate_report", fake)
    client.post("/api/report", json={"patient_id": 2, "care_mode": "hospital",
                                     "predict_result": {"marker": "client-copy"}})
    assert captured["pr"] == {"marker": "server-copy"}

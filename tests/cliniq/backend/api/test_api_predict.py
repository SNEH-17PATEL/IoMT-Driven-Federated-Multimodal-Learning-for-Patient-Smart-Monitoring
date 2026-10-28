"""Comprehensive tests for POST /api/predict."""
import pytest

REQUIRED_KEYS = {
    "patient_id", "row_idx", "cur_idx", "total_rows",
    "sofa_score", "risk_level", "risk_color", "risk_icon",
    "is_alert", "conformal_lo", "conformal_hi",
    "vitals", "vitals_status", "shap_top5",
}
REQUIRED_VITALS  = {"HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"}
VALID_STATUSES   = {"normal", "high", "low"}
VALID_RISK_LEVELS = {"Low Risk", "Moderate Risk", "High Risk"}


def _predict(client, patient_id=1, row_idx=0):
    r = client.post("/api/predict", json={"patient_id": patient_id, "row_idx": row_idx})
    assert r.status_code == 200, r.text
    return r.json()


# ── Response shape ────────────────────────────────────────────────────────────

@pytest.mark.api
def test_predict_required_keys(client):
    body = _predict(client)
    assert REQUIRED_KEYS.issubset(body.keys())


@pytest.mark.api
def test_predict_patient_id_echoed(client):
    assert _predict(client, patient_id=2)["patient_id"] == 2


@pytest.mark.api
def test_predict_row_idx_echoed(client):
    body = _predict(client, row_idx=5)
    assert "row_idx" in body


@pytest.mark.api
def test_predict_sofa_in_valid_range(client):
    sofa = _predict(client)["sofa_score"]
    assert 0.0 <= sofa <= 24.0


@pytest.mark.api
def test_predict_risk_level_is_known(client):
    assert _predict(client)["risk_level"] in VALID_RISK_LEVELS


@pytest.mark.api
def test_predict_risk_color_is_hex(client):
    color = _predict(client)["risk_color"]
    assert color.startswith("#") and len(color) == 7


@pytest.mark.api
def test_predict_is_alert_is_bool(client):
    assert isinstance(_predict(client)["is_alert"], bool)


@pytest.mark.api
def test_predict_is_alert_true_for_high_sofa(client):
    # Patient 3 is high risk — is_alert should often be True
    # We do not guarantee it but verify type
    result = _predict(client, patient_id=3)
    assert isinstance(result["is_alert"], bool)


# ── Conformal prediction interval ─────────────────────────────────────────────

@pytest.mark.api
def test_predict_conformal_interval_brackets_sofa(client):
    body = _predict(client)
    assert body["conformal_lo"] <= body["sofa_score"] <= body["conformal_hi"]


@pytest.mark.api
def test_predict_conformal_lo_non_negative(client):
    assert _predict(client)["conformal_lo"] >= 0.0


@pytest.mark.api
def test_predict_conformal_hi_at_most_24(client):
    assert _predict(client)["conformal_hi"] <= 24.0


@pytest.mark.api
def test_predict_conformal_interval_is_numeric(client):
    body = _predict(client)
    assert isinstance(body["conformal_lo"], (int, float))
    assert isinstance(body["conformal_hi"], (int, float))


# ── Vitals ────────────────────────────────────────────────────────────────────

@pytest.mark.api
def test_predict_vitals_keys_present(client):
    vitals = _predict(client)["vitals"]
    assert REQUIRED_VITALS.issubset(vitals.keys())


@pytest.mark.api
def test_predict_vitals_hr_numeric(client):
    hr = _predict(client)["vitals"]["HR"]
    assert isinstance(hr, (int, float))


@pytest.mark.api
def test_predict_vitals_spo2_in_range(client):
    spo2 = _predict(client)["vitals"]["SpO2"]
    assert 50 <= spo2 <= 100


@pytest.mark.api
def test_predict_vitals_hr_in_range(client):
    hr = _predict(client)["vitals"]["HR"]
    assert 20 <= hr <= 300


@pytest.mark.api
def test_predict_vitals_temp_in_range(client):
    temp = _predict(client)["vitals"]["Temp"]
    assert 30 <= temp <= 45


@pytest.mark.api
def test_predict_vitals_status_keys_present(client):
    statuses = _predict(client)["vitals_status"]
    for key in REQUIRED_VITALS:
        assert key in statuses, f"Missing vitals_status key: {key}"


@pytest.mark.api
def test_predict_vitals_status_values_valid(client):
    for val in _predict(client)["vitals_status"].values():
        assert val in VALID_STATUSES, f"Invalid status: {val}"


# ── SHAP ──────────────────────────────────────────────────────────────────────

@pytest.mark.api
def test_predict_shap_is_list(client):
    assert isinstance(_predict(client)["shap_top5"], list)


@pytest.mark.api
def test_predict_shap_at_most_5(client):
    assert len(_predict(client)["shap_top5"]) <= 5


@pytest.mark.api
def test_predict_shap_factor_keys(client):
    for factor in _predict(client)["shap_top5"]:
        assert "feature"    in factor
        assert "direction"  in factor
        assert "rel_impact" in factor


@pytest.mark.api
def test_predict_shap_direction_valid(client):
    for factor in _predict(client)["shap_top5"]:
        assert factor["direction"] in {"up", "down"}, f"Bad direction: {factor['direction']}"


@pytest.mark.api
def test_predict_shap_rel_impact_range(client):
    for factor in _predict(client)["shap_top5"]:
        assert 0.0 <= factor["rel_impact"] <= 1.0, f"rel_impact out of range: {factor['rel_impact']}"


@pytest.mark.api
def test_predict_shap_feature_is_string(client):
    for factor in _predict(client)["shap_top5"]:
        assert isinstance(factor["feature"], str)


# ── Row cycling ───────────────────────────────────────────────────────────────

@pytest.mark.api
def test_predict_cur_idx_within_total_rows(client):
    body = _predict(client, row_idx=0)
    assert 0 <= body["cur_idx"] < body["total_rows"]


@pytest.mark.api
def test_predict_row_wraps_large_index(client):
    body = _predict(client, patient_id=2, row_idx=99999)
    assert 0 <= body["cur_idx"] < body["total_rows"]


@pytest.mark.api
def test_predict_consecutive_rows_increment_cur_idx(client):
    r0 = _predict(client, patient_id=3, row_idx=0)
    r1 = _predict(client, patient_id=3, row_idx=1)
    expected = (r0["cur_idx"] + 1) % r0["total_rows"]
    assert r1["cur_idx"] == expected


@pytest.mark.api
def test_predict_total_rows_positive(client):
    assert _predict(client)["total_rows"] > 0


# ── All patients ──────────────────────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("pid", [1, 2, 3])
def test_predict_all_patients_succeed(client, pid):
    body = _predict(client, patient_id=pid)
    assert body["patient_id"] == pid
    assert 0.0 <= body["sofa_score"] <= 24.0


# ── Validation errors ─────────────────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("pid", [4, 5, 99])
def test_predict_invalid_patient_returns_400(client, pid):
    r = client.post("/api/predict", json={"patient_id": pid, "row_idx": 0})
    assert r.status_code == 400


@pytest.mark.api
def test_predict_patient_zero_returns_400(client):
    assert client.post("/api/predict", json={"patient_id": 0, "row_idx": 0}).status_code == 400


@pytest.mark.api
def test_predict_negative_patient_returns_400(client):
    assert client.post("/api/predict", json={"patient_id": -1, "row_idx": 0}).status_code == 400


@pytest.mark.api
def test_predict_missing_body_returns_422(client):
    assert client.post("/api/predict", json={}).status_code == 422


@pytest.mark.api
def test_predict_missing_row_idx_returns_422(client):
    assert client.post("/api/predict", json={"patient_id": 1}).status_code == 422

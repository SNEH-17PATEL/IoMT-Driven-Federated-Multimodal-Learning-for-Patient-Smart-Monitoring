REQUIRED_KEYS = {
    "patient_id", "row_idx", "cur_idx", "total_rows",
    "sofa_score", "risk_level", "risk_color", "risk_icon",
    "is_alert", "conformal_lo", "conformal_hi",
    "vitals", "vitals_status", "shap_top5",
}
REQUIRED_VITALS = {"HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"}
VALID_STATUSES  = {"normal", "high", "low"}
VALID_RISK_LEVELS = {"Low Risk", "Moderate Risk", "High Risk"}


def _predict(client, patient_id=1, row_idx=0):
    r = client.post("/api/predict", json={"patient_id": patient_id, "row_idx": row_idx})
    assert r.status_code == 200, r.text
    return r.json()


# ── Response shape ────────────────────────────────────────────────────────────

def test_predict_required_keys(client):
    body = _predict(client)
    assert REQUIRED_KEYS.issubset(body.keys())


def test_predict_patient_id_echoed(client):
    body = _predict(client, patient_id=2)
    assert body["patient_id"] == 2


def test_predict_sofa_in_valid_range(client):
    sofa = _predict(client)["sofa_score"]
    assert 0.0 <= sofa <= 24.0


def test_predict_risk_level_is_known(client):
    assert _predict(client)["risk_level"] in VALID_RISK_LEVELS


def test_predict_risk_color_is_hex(client):
    color = _predict(client)["risk_color"]
    assert color.startswith("#") and len(color) == 7


def test_predict_conformal_interval_brackets_sofa(client):
    body = _predict(client)
    assert body["conformal_lo"] <= body["sofa_score"] <= body["conformal_hi"]


def test_predict_conformal_lo_non_negative(client):
    assert _predict(client)["conformal_lo"] >= 0.0


def test_predict_conformal_hi_at_most_24(client):
    assert _predict(client)["conformal_hi"] <= 24.0


# ── Vitals ────────────────────────────────────────────────────────────────────

def test_predict_vitals_keys(client):
    vitals = _predict(client)["vitals"]
    assert REQUIRED_VITALS.issubset(vitals.keys())


def test_predict_vitals_status_keys_and_values(client):
    statuses = _predict(client)["vitals_status"]
    for key in ["HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"]:
        assert key in statuses
        assert statuses[key] in VALID_STATUSES


# ── SHAP ──────────────────────────────────────────────────────────────────────

def test_predict_shap_is_list(client):
    shap = _predict(client)["shap_top5"]
    assert isinstance(shap, list)


def test_predict_shap_at_most_5(client):
    assert len(_predict(client)["shap_top5"]) <= 5


def test_predict_shap_factor_structure(client):
    for factor in _predict(client)["shap_top5"]:
        assert "feature"    in factor
        assert "direction"  in factor
        assert "rel_impact" in factor
        assert factor["direction"] in {"up", "down"}
        assert 0.0 <= factor["rel_impact"] <= 1.0


# ── Row cycling ───────────────────────────────────────────────────────────────

def test_predict_cur_idx_within_total_rows(client):
    body = _predict(client, row_idx=0)
    assert 0 <= body["cur_idx"] < body["total_rows"]


def test_predict_row_wraps_large_index(client):
    body = _predict(client, patient_id=2, row_idx=99999)
    assert body["status_code"] if "status_code" in body else True  # just checks no exception
    assert 0 <= body["cur_idx"] < body["total_rows"]


def test_predict_consecutive_rows_increment_cur_idx(client):
    r0 = _predict(client, patient_id=3, row_idx=0)
    r1 = _predict(client, patient_id=3, row_idx=1)
    assert r1["cur_idx"] == (r0["cur_idx"] + 1) % r0["total_rows"]


# ── All patients ──────────────────────────────────────────────────────────────

def test_predict_all_three_patients(client):
    for pid in [1, 2, 3]:
        body = _predict(client, patient_id=pid)
        assert body["patient_id"] == pid
        assert 0.0 <= body["sofa_score"] <= 24.0


# ── Validation errors ─────────────────────────────────────────────────────────

def test_predict_invalid_patient_returns_400(client):
    r = client.post("/api/predict", json={"patient_id": 4, "row_idx": 0})
    assert r.status_code == 400


def test_predict_patient_zero_returns_400(client):
    r = client.post("/api/predict", json={"patient_id": 0, "row_idx": 0})
    assert r.status_code == 400


def test_predict_negative_patient_returns_400(client):
    r = client.post("/api/predict", json={"patient_id": -1, "row_idx": 0})
    assert r.status_code == 400


def test_predict_missing_body_returns_422(client):
    r = client.post("/api/predict", json={})
    assert r.status_code == 422

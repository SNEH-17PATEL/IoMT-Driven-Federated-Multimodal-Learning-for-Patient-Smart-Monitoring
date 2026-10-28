"""Tests for GET /api/patients and GET /api/patients/{id}."""
import pytest

REQUIRED_KEYS = {"id", "name", "short", "icon", "description", "color", "risk_level", "sofa_range"}


# ── List endpoint ─────────────────────────────────────────────────────────────

@pytest.mark.api
def test_list_patients_returns_200(client):
    assert client.get("/api/patients").status_code == 200


@pytest.mark.api
def test_list_patients_returns_three(client):
    assert len(client.get("/api/patients").json()) == 3


@pytest.mark.api
def test_list_patients_required_fields(client):
    for p in client.get("/api/patients").json():
        assert REQUIRED_KEYS.issubset(p.keys()), f"Missing keys in {p}"


@pytest.mark.api
def test_list_patients_ids_are_1_2_3(client):
    ids = {p["id"] for p in client.get("/api/patients").json()}
    assert ids == {1, 2, 3}


@pytest.mark.api
def test_list_patients_colors_are_hex(client):
    for p in client.get("/api/patients").json():
        assert p["color"].startswith("#") and len(p["color"]) == 7, \
            f"Patient {p['id']} color not valid hex: {p['color']}"


@pytest.mark.api
def test_list_patients_risk_levels_valid(client):
    valid = {"Low Risk", "Moderate Risk", "High Risk"}
    for p in client.get("/api/patients").json():
        assert p["risk_level"] in valid, f"Invalid risk_level: {p['risk_level']}"


@pytest.mark.api
def test_list_patients_names_nonempty(client):
    for p in client.get("/api/patients").json():
        assert len(p["name"]) > 0


@pytest.mark.api
def test_list_patients_icons_nonempty(client):
    for p in client.get("/api/patients").json():
        assert len(p["icon"]) > 0


@pytest.mark.api
def test_list_patients_sofa_range_nonempty(client):
    for p in client.get("/api/patients").json():
        assert len(p["sofa_range"]) > 0


# ── Per-patient risk level mapping ────────────────────────────────────────────

@pytest.mark.api
def test_patient_1_is_low_risk(client):
    assert client.get("/api/patients/1").json()["risk_level"] == "Low Risk"


@pytest.mark.api
def test_patient_2_is_moderate_risk(client):
    assert client.get("/api/patients/2").json()["risk_level"] == "Moderate Risk"


@pytest.mark.api
def test_patient_3_is_high_risk(client):
    assert client.get("/api/patients/3").json()["risk_level"] == "High Risk"


# ── Single patient ────────────────────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("pid", [1, 2, 3])
def test_get_patient_returns_200(client, pid):
    assert client.get(f"/api/patients/{pid}").status_code == 200


@pytest.mark.api
@pytest.mark.parametrize("pid", [1, 2, 3])
def test_get_patient_id_echoed(client, pid):
    assert client.get(f"/api/patients/{pid}").json()["id"] == pid


@pytest.mark.api
@pytest.mark.parametrize("pid", [1, 2, 3])
def test_get_patient_required_fields(client, pid):
    body = client.get(f"/api/patients/{pid}").json()
    assert REQUIRED_KEYS.issubset(body.keys())


# ── Not found cases ───────────────────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("pid", [4, 5, 99, 0, -1])
def test_get_patient_unknown_returns_404(client, pid):
    assert client.get(f"/api/patients/{pid}").status_code == 404

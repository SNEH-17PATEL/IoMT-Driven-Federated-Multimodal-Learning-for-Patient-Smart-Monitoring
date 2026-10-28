"""Tests for GET /api/vitals/{patient_id}."""
import pytest

REQUIRED_KEYS = {"time", "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"}


@pytest.mark.api
def test_vitals_returns_200(client):
    assert client.get("/api/vitals/1").status_code == 200


@pytest.mark.api
def test_vitals_returns_list(client):
    assert isinstance(client.get("/api/vitals/1").json(), list)


@pytest.mark.api
def test_vitals_default_at_most_20(client):
    assert len(client.get("/api/vitals/1").json()) <= 20


@pytest.mark.api
def test_vitals_custom_limit_respected(client):
    assert len(client.get("/api/vitals/1?limit=5").json()) <= 5


@pytest.mark.api
def test_vitals_custom_limit_1(client):
    assert len(client.get("/api/vitals/1?limit=1").json()) <= 1


@pytest.mark.api
def test_vitals_columns_present(client):
    data = client.get("/api/vitals/1").json()
    if data:
        assert REQUIRED_KEYS.issubset(data[0].keys())


@pytest.mark.api
def test_vitals_time_is_string(client):
    data = client.get("/api/vitals/1").json()
    for row in data:
        assert isinstance(row["time"], str), f"time not str: {row['time']}"


@pytest.mark.api
def test_vitals_hr_in_plausible_range(client):
    for row in client.get("/api/vitals/1").json():
        assert 20 <= row["HR"] <= 300, f"Implausible HR: {row['HR']}"


@pytest.mark.api
def test_vitals_spo2_in_plausible_range(client):
    for row in client.get("/api/vitals/1").json():
        assert 50 <= row["SpO2"] <= 100, f"Implausible SpO2: {row['SpO2']}"


@pytest.mark.api
def test_vitals_sbp_in_plausible_range(client):
    for row in client.get("/api/vitals/1").json():
        assert 30 <= row["SBP"] <= 250, f"Implausible SBP: {row['SBP']}"


@pytest.mark.api
def test_vitals_rr_in_plausible_range(client):
    for row in client.get("/api/vitals/1").json():
        assert 4 <= row["RR"] <= 60, f"Implausible RR: {row['RR']}"


@pytest.mark.api
@pytest.mark.parametrize("pid", [1, 2, 3])
def test_vitals_all_patients_succeed(client, pid):
    assert client.get(f"/api/vitals/{pid}").status_code == 200


# ── Validation errors ─────────────────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("pid", [4, 99])
def test_vitals_invalid_patient_returns_400(client, pid):
    assert client.get(f"/api/vitals/{pid}").status_code == 400


@pytest.mark.api
def test_vitals_patient_zero_returns_400(client):
    assert client.get("/api/vitals/0").status_code == 400


@pytest.mark.api
def test_vitals_limit_too_large_returns_400(client):
    assert client.get("/api/vitals/1?limit=21").status_code == 400


@pytest.mark.api
def test_vitals_limit_zero_returns_400(client):
    assert client.get("/api/vitals/1?limit=0").status_code == 400


@pytest.mark.api
def test_vitals_limit_negative_returns_400(client):
    assert client.get("/api/vitals/1?limit=-1").status_code == 400

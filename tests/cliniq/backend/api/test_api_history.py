"""Tests for GET /api/history/{patient_id}."""
import pytest

REQUIRED_KEYS = {"Timestamp", "SOFA", "Risk"}


@pytest.mark.api
def test_history_returns_200(client):
    assert client.get("/api/history/1").status_code == 200


@pytest.mark.api
def test_history_returns_list(client):
    assert isinstance(client.get("/api/history/1").json(), list)


@pytest.mark.api
def test_history_default_limit_at_most_5(client):
    assert len(client.get("/api/history/1").json()) <= 5


@pytest.mark.api
def test_history_custom_limit_respected(client):
    assert len(client.get("/api/history/1?limit=2").json()) <= 2


@pytest.mark.api
def test_history_columns_present(client):
    data = client.get("/api/history/1").json()
    if data:
        assert REQUIRED_KEYS.issubset(data[0].keys()), f"Missing keys: {REQUIRED_KEYS - set(data[0].keys())}"


@pytest.mark.api
def test_history_sofa_values_are_numeric(client):
    for row in client.get("/api/history/1").json():
        assert isinstance(row["SOFA"], (int, float))


@pytest.mark.api
def test_history_sofa_in_valid_range(client):
    for row in client.get("/api/history/1").json():
        assert 0 <= row["SOFA"] <= 24, f"SOFA out of range: {row['SOFA']}"


@pytest.mark.api
def test_history_risk_values_nonempty(client):
    for row in client.get("/api/history/1").json():
        assert row["Risk"], "Risk field must not be empty"


@pytest.mark.api
@pytest.mark.parametrize("pid", [1, 2, 3])
def test_history_all_patients_succeed(client, pid):
    assert client.get(f"/api/history/{pid}").status_code == 200


# ── Validation errors ─────────────────────────────────────────────────────────

@pytest.mark.api
@pytest.mark.parametrize("pid", [4, 99])
def test_history_invalid_patient_returns_400(client, pid):
    assert client.get(f"/api/history/{pid}").status_code == 400


@pytest.mark.api
def test_history_patient_zero_returns_400(client):
    assert client.get("/api/history/0").status_code == 400


@pytest.mark.api
def test_history_limit_zero_returns_400(client):
    assert client.get("/api/history/1?limit=0").status_code == 400


@pytest.mark.api
def test_history_limit_too_large_returns_400(client):
    assert client.get("/api/history/1?limit=51").status_code == 400


@pytest.mark.api
def test_history_limit_negative_returns_400(client):
    assert client.get("/api/history/1?limit=-1").status_code == 400

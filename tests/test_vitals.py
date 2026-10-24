REQUIRED_KEYS = {"time", "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP"}


def test_vitals_returns_list(client):
    r = client.get("/api/vitals/1")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_vitals_default_at_most_20(client):
    data = client.get("/api/vitals/1").json()
    assert len(data) <= 20


def test_vitals_custom_limit_respected(client):
    data = client.get("/api/vitals/1?limit=5").json()
    assert len(data) <= 5


def test_vitals_columns_present(client):
    data = client.get("/api/vitals/1").json()
    if data:
        assert REQUIRED_KEYS.issubset(data[0].keys())


def test_vitals_hr_in_plausible_range(client):
    data = client.get("/api/vitals/1").json()
    for row in data:
        assert 20 <= row["HR"] <= 300, f"Implausible HR: {row['HR']}"


def test_vitals_spo2_in_plausible_range(client):
    data = client.get("/api/vitals/1").json()
    for row in data:
        assert 50 <= row["SpO2"] <= 100, f"Implausible SpO2: {row['SpO2']}"


def test_vitals_all_patients(client):
    for pid in [1, 2, 3]:
        assert client.get(f"/api/vitals/{pid}").status_code == 200


def test_vitals_invalid_patient_returns_400(client):
    assert client.get("/api/vitals/99").status_code == 400


def test_vitals_patient_zero_returns_400(client):
    assert client.get("/api/vitals/0").status_code == 400


def test_vitals_limit_too_large_returns_400(client):
    # Max is 20 per vitals.py validation
    assert client.get("/api/vitals/1?limit=21").status_code == 400

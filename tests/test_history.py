REQUIRED_KEYS = {"Timestamp", "SOFA", "Risk"}


def test_history_returns_list(client):
    r = client.get("/api/history/1")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_history_default_limit_at_most_5(client):
    data = client.get("/api/history/1").json()
    assert len(data) <= 5


def test_history_custom_limit_respected(client):
    data = client.get("/api/history/1?limit=2").json()
    assert len(data) <= 2


def test_history_columns_present(client):
    data = client.get("/api/history/1").json()
    if data:  # Only assert columns if there are rows (DB may be empty on fresh run)
        assert REQUIRED_KEYS.issubset(data[0].keys())


def test_history_sofa_values_are_numeric(client):
    data = client.get("/api/history/1").json()
    for row in data:
        assert isinstance(row["SOFA"], (int, float))
        assert 0 <= row["SOFA"] <= 24


def test_history_all_patients(client):
    for pid in [1, 2, 3]:
        assert client.get(f"/api/history/{pid}").status_code == 200


def test_history_invalid_patient_returns_400(client):
    assert client.get("/api/history/99").status_code == 400


def test_history_patient_zero_returns_400(client):
    assert client.get("/api/history/0").status_code == 400


def test_history_limit_zero_returns_400(client):
    assert client.get("/api/history/1?limit=0").status_code == 400


def test_history_limit_too_large_returns_400(client):
    # Max is 50 per history.py validation
    assert client.get("/api/history/1?limit=51").status_code == 400

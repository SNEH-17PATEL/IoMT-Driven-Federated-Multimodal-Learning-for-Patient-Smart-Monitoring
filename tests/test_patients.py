REQUIRED_KEYS = {"id", "name", "short", "icon", "description", "color", "risk_level", "sofa_range"}


def test_list_patients_returns_three(client):
    r = client.get("/api/patients")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_list_patients_fields(client):
    for p in client.get("/api/patients").json():
        assert REQUIRED_KEYS.issubset(p.keys())


def test_list_patients_ids_are_1_2_3(client):
    ids = {p["id"] for p in client.get("/api/patients").json()}
    assert ids == {1, 2, 3}


def test_get_patient_valid_all_three(client):
    for pid in [1, 2, 3]:
        r = client.get(f"/api/patients/{pid}")
        assert r.status_code == 200
        assert r.json()["id"] == pid


def test_get_patient_color_is_hex(client):
    for pid in [1, 2, 3]:
        color = client.get(f"/api/patients/{pid}").json()["color"]
        assert color.startswith("#"), f"Expected hex color, got: {color}"
        assert len(color) == 7


def test_get_patient_risk_levels(client):
    expected = {
        1: "Low Risk",
        2: "Moderate Risk",
        3: "High Risk",
    }
    for pid, expected_level in expected.items():
        level = client.get(f"/api/patients/{pid}").json()["risk_level"]
        assert level == expected_level


def test_get_patient_unknown_returns_404(client):
    assert client.get("/api/patients/99").status_code == 404


def test_get_patient_zero_returns_404(client):
    assert client.get("/api/patients/0").status_code == 404

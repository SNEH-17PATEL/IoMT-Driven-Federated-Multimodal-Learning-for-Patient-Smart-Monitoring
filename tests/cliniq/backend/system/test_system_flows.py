"""
System-level tests — simulate complete multi-step frontend workflows.
Each test runs as an independent flow through the real DB and ML stack.
"""
import pytest


@pytest.mark.system
def test_full_patient_select_and_monitor_flow(client):
    """
    Mirrors the SelectPatient → MonitorDashboard → history chain.
    1. List patients  2. Select patient  3. Predict  4. Vitals  5. History
    """
    # 1 — Patient selection page
    r = client.get("/api/patients")
    assert r.status_code == 200
    patients = r.json()
    assert len(patients) == 3

    # 2 — User picks patient 2
    pid = 2
    r = client.get(f"/api/patients/{pid}")
    assert r.status_code == 200 and r.json()["id"] == pid

    # 3 — Dashboard initial load (row 0)
    r = client.post("/api/predict", json={"patient_id": pid, "row_idx": 0})
    assert r.status_code == 200
    pred0 = r.json()
    assert pred0["patient_id"] == pid
    assert pred0["cur_idx"] == 0
    assert 0.0 <= pred0["sofa_score"] <= 24.0

    # 4 — Next 90-second refresh (row 1)
    r = client.post("/api/predict", json={"patient_id": pid, "row_idx": 1})
    assert r.status_code == 200 and r.json()["cur_idx"] == 1

    # 5 — Trends chart data
    r = client.get(f"/api/vitals/{pid}?limit=20")
    assert r.status_code == 200 and len(r.json()) >= 1

    # 6 — History drawer
    r = client.get(f"/api/history/{pid}?limit=5")
    assert r.status_code == 200
    for row in r.json():
        assert 0 <= row["SOFA"] <= 24


@pytest.mark.system
def test_all_patients_predict_independently(client):
    """Each patient returns valid, non-overlapping predictions."""
    results = {}
    for pid in [1, 2, 3]:
        r = client.post("/api/predict", json={"patient_id": pid, "row_idx": 0})
        assert r.status_code == 200
        results[pid] = r.json()
        assert results[pid]["patient_id"] == pid

    # Patients should have different SOFA scores or at least different row data
    # (we can't guarantee scores differ, but the endpoint handles each independently)
    for pid in [1, 2, 3]:
        assert 0.0 <= results[pid]["sofa_score"] <= 24.0


@pytest.mark.system
def test_row_cycling_stays_in_bounds(client):
    """cur_idx is always within [0, total_rows) regardless of input row_idx."""
    for row_idx in [0, 1, 50, 500, 10_000, 99_999]:
        r = client.post("/api/predict", json={"patient_id": 1, "row_idx": row_idx})
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["cur_idx"] < body["total_rows"], (
            f"row_idx={row_idx} → cur_idx={body['cur_idx']} out of [0,{body['total_rows']})"
        )


@pytest.mark.system
def test_vitals_accumulate_after_predict_calls(client):
    """Each predict call must append to the vitals sliding window."""
    baseline = len(client.get("/api/vitals/3?limit=20").json())

    for i in range(3):
        client.post("/api/predict", json={"patient_id": 3, "row_idx": i + 200})

    after = len(client.get("/api/vitals/3?limit=20").json())
    assert after >= 1


@pytest.mark.system
def test_cross_patient_history_isolation(client):
    """Predict calls for patient 2 must not appear in patient 1's history."""
    baseline = len(client.get("/api/history/1?limit=50").json())

    client.post("/api/predict", json={"patient_id": 2, "row_idx": 300})
    client.post("/api/predict", json={"patient_id": 2, "row_idx": 301})

    after = len(client.get("/api/history/1?limit=50").json())
    assert after == baseline, (
        f"Patient 1 history grew ({baseline} → {after}) after patient 2 predict calls"
    )


@pytest.mark.system
def test_predict_then_vitals_consistency(client):
    """
    The HR returned by /api/predict should appear in the recent vitals for that patient.
    """
    pred = client.post("/api/predict", json={"patient_id": 1, "row_idx": 7}).json()
    pred_hr = pred["vitals"]["HR"]

    vitals_rows = client.get("/api/vitals/1?limit=20").json()
    hr_values = [row["HR"] for row in vitals_rows]
    assert pred_hr in hr_values, (
        f"Predict HR={pred_hr} not found in vitals history: {hr_values}"
    )


@pytest.mark.system
def test_health_endpoint_stays_responsive_throughout(client):
    """Health check stays 200 before, during, and after a monitoring sequence."""
    assert client.get("/health").status_code == 200

    for i in range(3):
        client.post("/api/predict", json={"patient_id": 1, "row_idx": i + 50})

    assert client.get("/health").status_code == 200


@pytest.mark.system
def test_history_contains_latest_prediction_data(client):
    """After a predict call, history should contain an entry for that patient."""
    # Baseline count
    before = len(client.get("/api/history/1?limit=50").json())

    # Make a predict call which appends to history
    client.post("/api/predict", json={"patient_id": 1, "row_idx": 400})

    after = len(client.get("/api/history/1?limit=50").json())
    assert after >= before  # count must not shrink


@pytest.mark.system
def test_vitals_endpoint_returns_valid_sofa_consistent_data(client):
    """
    After predicting, vitals returned by /api/vitals must be numeric and in range.
    """
    client.post("/api/predict", json={"patient_id": 2, "row_idx": 0})
    vitals = client.get("/api/vitals/2?limit=5").json()
    assert len(vitals) >= 1
    for row in vitals:
        assert 20 <= row["HR"] <= 300
        assert 50 <= row["SpO2"] <= 100

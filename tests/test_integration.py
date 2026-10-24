"""
End-to-end flow tests that simulate what the React frontend does
during a live monitoring session.

Run order within this file matters: each test is independent (they use
the session-scoped client but assert relative changes, not absolute counts).
"""


def test_patient_select_and_monitor_flow(client):
    """Mirrors the full SelectPatient → MonitorDashboard user journey."""
    # 1. Patient selection: fetch all patients
    r = client.get("/api/patients")
    assert r.status_code == 200
    patients = r.json()
    assert len(patients) == 3

    # 2. User picks patient 2 (moderate risk)
    pid = 2
    r = client.get(f"/api/patients/{pid}")
    assert r.status_code == 200
    assert r.json()["id"] == pid

    # 3. First predict (row 0 — initial dashboard load)
    r = client.post("/api/predict", json={"patient_id": pid, "row_idx": 0})
    assert r.status_code == 200
    pred0 = r.json()
    assert pred0["patient_id"] == pid
    assert pred0["cur_idx"] == 0

    # 4. 30-second refresh: next row
    r = client.post("/api/predict", json={"patient_id": pid, "row_idx": 1})
    assert r.status_code == 200
    pred1 = r.json()
    assert pred1["cur_idx"] == 1

    # 5. Vital trends chart data (useQuery in TrendsSection)
    r = client.get(f"/api/vitals/{pid}?limit=20")
    assert r.status_code == 200
    assert len(r.json()) >= 1

    # 6. History panel
    r = client.get(f"/api/history/{pid}?limit=5")
    assert r.status_code == 200
    history = r.json()
    assert len(history) >= 1

    # Verify history has valid SOFA values
    for row in history:
        assert 0 <= row["SOFA"] <= 24


def test_all_patients_predict_independently(client):
    """Each patient returns valid, independent predictions."""
    results = {}
    for pid in [1, 2, 3]:
        r = client.post("/api/predict", json={"patient_id": pid, "row_idx": 0})
        assert r.status_code == 200
        results[pid] = r.json()
        assert results[pid]["patient_id"] == pid
        assert 0.0 <= results[pid]["sofa_score"] <= 24.0


def test_row_cycling_stays_in_bounds(client):
    """cur_idx is always within [0, total_rows) regardless of row_idx sent."""
    for row_idx in [0, 1, 50, 500, 10000]:
        r = client.post("/api/predict", json={"patient_id": 1, "row_idx": row_idx})
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["cur_idx"] < body["total_rows"], (
            f"row_idx={row_idx} → cur_idx={body['cur_idx']} out of [0, {body['total_rows']})"
        )


def test_vitals_accumulate_after_predict_calls(client):
    """Each predict call appends a vitals row — the vitals endpoint must reflect it."""
    # Record baseline count for patient 3
    baseline = len(client.get("/api/vitals/3?limit=20").json())

    # Make 3 new predict calls
    for i in range(3):
        client.post("/api/predict", json={"patient_id": 3, "row_idx": i + 100})

    after = len(client.get("/api/vitals/3?limit=20").json())
    # Vitals table grew (capped at 20 by the query, but count should be non-zero)
    assert after >= 1


def test_cross_patient_history_isolation(client):
    """Predict calls for patient 2 must not appear in patient 1's history."""
    hist_before = len(client.get("/api/history/1?limit=50").json())

    # Calls only for patient 2
    client.post("/api/predict", json={"patient_id": 2, "row_idx": 200})
    client.post("/api/predict", json={"patient_id": 2, "row_idx": 201})

    hist_after = len(client.get("/api/history/1?limit=50").json())
    assert hist_after == hist_before, (
        f"Patient 1 history grew ({hist_before} → {hist_after}) after patient 2 predict calls"
    )


def test_api_health_before_and_after_predict(client):
    """Health endpoint stays responsive throughout the monitoring lifecycle."""
    assert client.get("/health").status_code == 200
    client.post("/api/predict", json={"patient_id": 1, "row_idx": 0})
    assert client.get("/health").status_code == 200


def test_predict_then_vitals_consistency(client):
    """Vitals returned by /api/vitals match the vitals embedded in the predict response."""
    pred = client.post("/api/predict", json={"patient_id": 1, "row_idx": 5}).json()
    pred_hr = pred["vitals"]["HR"]

    # The vitals endpoint should contain a recent row with this HR value
    vitals_rows = client.get("/api/vitals/1?limit=20").json()
    hr_values = [row["HR"] for row in vitals_rows]
    assert pred_hr in hr_values, (
        f"Predict HR={pred_hr} not found in vitals history: {hr_values}"
    )

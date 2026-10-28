import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure icu_monitor/ root is on sys.path so db.py, model_utils.py are importable
_TEST_DIR  = os.path.dirname(os.path.abspath(__file__))   # tests/cliniq/backend/
_CLINIQ    = os.path.dirname(_TEST_DIR)                    # tests/cliniq/
_TESTS     = os.path.dirname(_CLINIQ)                     # tests/
_ROOT      = os.path.dirname(_TESTS)                      # icu_monitor/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.main import app


@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient — loads ML models once for the full test suite."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def predict_p1(client):
    """Cached predict result for patient 1, reused by report and system tests."""
    r = client.post("/api/predict", json={"patient_id": 1, "row_idx": 0})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="session")
def predict_p2(client):
    """Cached predict result for patient 2."""
    r = client.post("/api/predict", json={"patient_id": 2, "row_idx": 0})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="session")
def predict_p3(client):
    """Cached predict result for patient 3."""
    r = client.post("/api/predict", json={"patient_id": 3, "row_idx": 0})
    assert r.status_code == 200, r.text
    return r.json()

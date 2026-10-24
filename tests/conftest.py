import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure icu_monitor/ root is importable (db.py, model_utils.py live there)
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_TEST_DIR)
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
    """Cached predict result for patient 1, reused by report tests to avoid redundant ML calls."""
    r = client.post("/api/predict", json={"patient_id": 1, "row_idx": 0})
    assert r.status_code == 200, r.text
    return r.json()

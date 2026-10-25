import json
import os
import sys

from fastapi import APIRouter, HTTPException

# Make icu_monitor/ importable (db.py lives there)
_API_DIR     = os.path.dirname(os.path.abspath(__file__))   # backend/api/
_BACKEND_DIR = os.path.dirname(_API_DIR)                    # backend/
_ROOT        = os.path.dirname(_BACKEND_DIR)                # icu_monitor/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from db import get_prediction_history  # noqa: E402

router = APIRouter()

_VALID_PATIENTS = {1, 2, 3}


@router.get("/api/history/{patient_id}")
def get_history(patient_id: int, limit: int = 5):
    if patient_id not in _VALID_PATIENTS:
        raise HTTPException(status_code=400, detail="patient_id must be 1, 2, or 3")
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 50")
    try:
        df = get_prediction_history(patient_id, limit=limit)
        # pandas .to_json() converts NaN → null; parse back so FastAPI can serialize
        return json.loads(df.to_json(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History query failed: {str(e)}")

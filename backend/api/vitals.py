import os
import sys

from fastapi import APIRouter, HTTPException

# Make icu_monitor/ importable (db.py lives there)
_API_DIR     = os.path.dirname(os.path.abspath(__file__))   # backend/api/
_BACKEND_DIR = os.path.dirname(_API_DIR)                    # backend/
_ROOT        = os.path.dirname(_BACKEND_DIR)                # icu_monitor/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from db import get_patient_vitals  # noqa: E402

router = APIRouter()

_VALID_PATIENTS = {1, 2, 3}


@router.get("/api/vitals/{patient_id}")
def get_vitals(patient_id: int, limit: int = 20):
    if patient_id not in _VALID_PATIENTS:
        raise HTTPException(status_code=400, detail="patient_id must be 1, 2, or 3")
    if limit < 1 or limit > 20:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 20")
    try:
        df = get_patient_vitals(patient_id, limit=limit)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vitals query failed: {str(e)}")

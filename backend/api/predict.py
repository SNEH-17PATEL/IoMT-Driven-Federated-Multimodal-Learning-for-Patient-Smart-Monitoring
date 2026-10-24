from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.ml_service import run_prediction

router = APIRouter()


class PredictRequest(BaseModel):
    patient_id: int
    row_idx: int


@router.post("/api/predict")
def predict(body: PredictRequest):
    if body.patient_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="patient_id must be 1, 2, or 3")
    try:
        result = run_prediction(body.patient_id, body.row_idx)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

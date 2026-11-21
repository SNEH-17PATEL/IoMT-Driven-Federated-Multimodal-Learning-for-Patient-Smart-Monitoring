from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.ml_service import run_prediction

router = APIRouter()

# Server-side store of the most recent predict result per patient.
# /api/report reads from here so it never trusts client-supplied data.
_LAST_PREDICT: dict[int, dict] = {}


class PredictRequest(BaseModel):
    patient_id: int
    row_idx: int
    cv_gcs_eye: int | None = None    # live GCS eye-opening from CV monitor (1–4)
    cv_stress: float | None = None   # live physiological stress score from CV monitor (0–10)


@router.post("/api/predict")
def predict(body: PredictRequest):
    if body.patient_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="patient_id must be 1, 2, or 3")
    if body.cv_gcs_eye is not None and body.cv_gcs_eye not in (1, 2, 3, 4):
        raise HTTPException(status_code=400, detail="cv_gcs_eye must be 1–4")
    if body.cv_stress is not None and not (0.0 <= body.cv_stress <= 10.0):
        raise HTTPException(status_code=400, detail="cv_stress must be 0–10")
    try:
        result = run_prediction(
            body.patient_id, body.row_idx,
            cv_gcs_eye=body.cv_gcs_eye,
            cv_stress=body.cv_stress,
        )
        # Store the full result (including LLM prompt blocks) for /api/report
        _LAST_PREDICT[body.patient_id] = result
        # Return a clean response: strip prompt scaffold fields the browser never needs
        _PROMPT_FIELDS = {'_vitals_block', '_cv_block', '_risk_block', '_explanation_block'}
        return {k: v for k, v in result.items() if k not in _PROMPT_FIELDS}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import sys as _sys
        print(f"[ClinIQ] Prediction error: {e}", file=_sys.stderr)
        raise HTTPException(status_code=500, detail="Prediction failed — check server logs.")

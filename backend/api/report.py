import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.llm_service import generate_report

router = APIRouter()


class ReportRequest(BaseModel):
    patient_id: int
    care_mode: str        # "hospital" or "homecare"
    predict_result: dict  # full result dict from /api/predict


@router.post("/api/report")
def report(body: ReportRequest):
    if body.patient_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="patient_id must be 1, 2, or 3")

    if body.care_mode not in ("hospital", "homecare"):
        raise HTTPException(status_code=400, detail="care_mode must be 'hospital' or 'homecare'")

    # Ensure the predict_result actually belongs to the claimed patient to prevent
    # cross-patient report generation and prompt injection via fabricated payloads.
    pr = body.predict_result
    if pr.get("patient_id") != body.patient_id:
        raise HTTPException(status_code=400, detail="predict_result.patient_id does not match patient_id")
    if not isinstance(pr.get("vitals"), dict):
        raise HTTPException(status_code=400, detail="predict_result.vitals must be a dict")
    if not isinstance(pr.get("sofa_score"), (int, float)):
        raise HTTPException(status_code=400, detail="predict_result.sofa_score is required")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        result = generate_report(pr, body.care_mode, api_key)
        return result
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field in predict_result: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

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
    if body.care_mode not in ("hospital", "homecare"):
        raise HTTPException(status_code=400, detail="care_mode must be 'hospital' or 'homecare'")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        result = generate_report(body.predict_result, body.care_mode, api_key)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

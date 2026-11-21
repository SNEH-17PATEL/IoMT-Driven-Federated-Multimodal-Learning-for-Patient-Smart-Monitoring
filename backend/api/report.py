import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.llm_service import generate_report
from backend.api.predict import _LAST_PREDICT

router = APIRouter()


class ReportRequest(BaseModel):
    patient_id: int
    care_mode: str   # "hospital" or "homecare"


@router.post("/api/report")
def report(body: ReportRequest):
    if body.patient_id not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="patient_id must be 1, 2, or 3")

    if body.care_mode not in ("hospital", "homecare"):
        raise HTTPException(status_code=400, detail="care_mode must be 'hospital' or 'homecare'")

    # Always use the server's own last prediction — never trust client-supplied data.
    pr = _LAST_PREDICT.get(body.patient_id)
    if pr is None:
        raise HTTPException(
            status_code=409,
            detail="No prediction has been run yet for this patient. Call /api/predict first.",
        )

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        result = generate_report(pr, body.care_mode, api_key)
        return result
    except Exception as e:
        import sys as _sys
        print(f"[ClinIQ] Report error: {e}", file=_sys.stderr)
        raise HTTPException(status_code=500, detail="Report generation failed — check server logs.")

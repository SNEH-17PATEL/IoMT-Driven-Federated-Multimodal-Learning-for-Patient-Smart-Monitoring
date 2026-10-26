from fastapi import APIRouter, HTTPException

router = APIRouter()

PATIENTS = {
    1: {
        "id":          1,
        "name":        "Patient 1 — Low Risk",
        "short":       "Low Risk",
        "icon":        "🟢",
        "description": "Post-surgical recovery — stable, improving trend",
        "color":       "#28a745",
        "risk_level":  "Low Risk",
        "sofa_range":  "0 – 4",
    },
    2: {
        "id":          2,
        "name":        "Patient 2 — Moderate Risk",
        "short":       "Moderate Risk",
        "icon":        "🟡",
        "description": "Community-acquired pneumonia — on supplemental O₂",
        "color":       "#e6a817",
        "risk_level":  "Moderate Risk",
        "sofa_range":  "5 – 9",
    },
    3: {
        "id":          3,
        "name":        "Patient 3 — High Risk",
        "short":       "High Risk",
        "icon":        "🔴",
        "description": "Septic shock — vasopressors, intubated, multi-organ failure",
        "color":       "#dc3545",
        "risk_level":  "High Risk",
        "sofa_range":  "10 – 24",
    },
}


@router.get("/api/patients")
def list_patients():
    return list(PATIENTS.values())


@router.get("/api/patients/{patient_id}")
def get_patient(patient_id: int):
    if patient_id not in PATIENTS:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return PATIENTS[patient_id]

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import patients, predict, report, history, vitals

load_dotenv()

app = FastAPI(title="ClinIQ API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router)
app.include_router(predict.router)
app.include_router(report.router)
app.include_router(history.router)
app.include_router(vitals.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "ClinIQ"}

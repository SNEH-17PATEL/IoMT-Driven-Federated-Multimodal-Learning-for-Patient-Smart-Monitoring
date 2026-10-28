import os
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import patients, predict, report, history, vitals

load_dotenv()

# Ensure repo root (icu_monitor/) is on the path so db.py is importable
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_BACKEND_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from db import init_db  # noqa: E402 — import after path fix


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the SQLite schema on startup so a fresh clone doesn't 500."""
    init_db()
    yield


app = FastAPI(title="ClinIQ API", version="1.0", lifespan=lifespan)

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

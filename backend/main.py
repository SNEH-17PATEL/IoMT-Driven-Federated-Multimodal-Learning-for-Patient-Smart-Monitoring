"""
FastAPI backend for the ICU Clinical Decision Support System.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000

This is a presentation-layer replacement for the Streamlit app.py — it does
not change any model, data, or scoring logic (see backend/pipeline.py, which
is a direct port of app.py's computation pipeline).
"""

import asyncio
import json
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend import pipeline
from backend.fl_demo import FLDemoRunner

app = FastAPI(title="ICU Clinical Decision Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/patients")
def get_patients():
    return [
        {**cfg, "id": pid}
        for pid, cfg in pipeline.PATIENTS.items()
    ]


@app.get("/api/model-info")
def get_model_info():
    return pipeline.training_meta


@app.post("/api/monitor/{patient_id}/start")
def start_monitoring(patient_id: int):
    if patient_id not in pipeline.PATIENTS:
        return {"error": "unknown_patient"}
    pipeline.reset_patient(patient_id)
    return {"status": "started", "patient_id": patient_id}


@app.get("/api/monitor/{patient_id}/reading")
def get_reading(patient_id: int):
    if patient_id not in pipeline.PATIENTS:
        return {"error": "unknown_patient"}
    try:
        return pipeline.run_reading_pipeline(patient_id)
    except Exception as e:
        traceback.print_exc()
        return {"error": "pipeline_failed", "message": str(e)}


# =============================================================
# LIVE FEDERATED LEARNING DEMO
# =============================================================

@app.websocket("/ws/fl-demo")
async def fl_demo_socket(ws: WebSocket):
    await ws.accept()
    runner = None
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            action = msg.get("action")
            if action == "start":
                if runner is not None:
                    runner.cancel()
                runner = FLDemoRunner(
                    rounds=int(msg.get("rounds", 15)),
                    epochs=int(msg.get("epochs", 1)),
                    hospital_size=int(msg.get("hospital_size", 500)),
                )

                async def send(payload):
                    await ws.send_text(json.dumps(payload))

                asyncio.create_task(runner.run(send))
            elif action == "cancel":
                if runner is not None:
                    runner.cancel()
    except WebSocketDisconnect:
        if runner is not None:
            runner.cancel()

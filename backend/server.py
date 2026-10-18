"""
Web server for the monitor.

    python run.py            (or)    python -m uvicorn backend.server:app --port 8000

The model loads in a background thread so the page can open straight away and show
a "warming up" state (the first load imports torch + SHAP and takes ~20-30 s).
"""
import json
import threading
import traceback
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="CareWatch — multimodal patient monitor")

_state = {"monitor": None, "error": None}


def _load():
    try:
        from backend.engine import Monitor
        _state["monitor"] = Monitor()
    except Exception as e:                       # surfaced to the page, not swallowed
        traceback.print_exc()
        _state["error"] = f"{type(e).__name__}: {e}"


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load, daemon=True).start()


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def reply(obj):
    return JSONResponse(json.loads(json.dumps(obj, default=_np)))


def mon():
    if _state["error"]:
        raise HTTPException(500, _state["error"])
    if _state["monitor"] is None:
        raise HTTPException(503, "model is still loading")
    return _state["monitor"]


class Start(BaseModel):
    patient: int


class Tick(BaseModel):
    patient: int
    mode: str = "icu"
    use_cam: bool = False


class FLStart(BaseModel):
    rounds: int = 12
    epochs: int = 2
    algo: str = "fedavg"          # fedavg | fedprox
    split: str = "uneven"         # uneven | even
    privacy: str = "off"          # off | low | high
    speed: str = "normal"         # slow | normal | fast


class Note(BaseModel):
    patient: int
    text: str = ""


@app.get("/api/health")
def health():
    return {"ready": _state["monitor"] is not None, "error": _state["error"]}


@app.get("/api/patients")
def patients(mode: str = "icu"):
    return reply(mon().patients("home" if mode == "home" else "icu"))


@app.post("/api/start")
def start(body: Start):
    return reply(mon().start(body.patient))


@app.post("/api/tick")
def tick(body: Tick):
    try:
        return reply(mon().tick(body.patient, "home" if body.mode == "home" else "icu", body.use_cam))
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@app.post("/api/note")
def note(body: Note):
    return reply(mon().set_live_note(body.patient, body.text))


@app.get("/api/report/{rid}")
def report(rid: str):
    return reply(mon().report(rid))


@app.get("/api/history/{patient}")
def history(patient: int):
    return reply(mon().history(patient))


@app.get("/api/camera")
def camera():
    return reply(mon().camera())


@app.post("/api/camera/start")
def camera_start():
    return reply(mon().camera_start())


@app.post("/api/camera/stop")
def camera_stop():
    return reply(mon().camera_stop())


@app.get("/api/fl/available")
def fl_available():
    from backend import fl_demo
    return {"available": fl_demo.available(), "path": str(fl_demo.DATA_PATH)}


@app.post("/api/fl/start")
def fl_start(body: FLStart):
    from backend import fl_demo
    if not fl_demo.available():
        raise HTTPException(404, f"Demo data not found at {fl_demo.DATA_PATH}")
    cfg = {"rounds": max(3, min(30, body.rounds)), "epochs": max(1, min(4, body.epochs)),
           "algo": body.algo if body.algo in ("fedavg", "fedprox") else "fedavg", "mu": 0.1,
           "split": body.split if body.split in ("uneven", "even") else "uneven",
           "privacy": body.privacy if body.privacy in ("off", "low", "high") else "off",
           "speed": body.speed if body.speed in ("slow", "normal", "fast") else "normal"}
    return {"job": fl_demo.start(cfg)}


@app.get("/api/fl/events")
def fl_events(job: str, since: int = 0):
    from backend import fl_demo
    return reply(fl_demo.events(job, since))


@app.post("/api/fl/stop")
def fl_stop(job: str):
    from backend import fl_demo
    return fl_demo.stop(job)


@app.get("/api/model")
def model_info():
    return reply(mon().model_info())


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")

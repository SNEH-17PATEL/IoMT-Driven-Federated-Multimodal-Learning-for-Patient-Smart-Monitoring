"""One-command launcher:  python run.py   ->  opens http://localhost:8000"""
import threading
import time
import webbrowser

import uvicorn

PORT = 8000


def _open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("backend.server:app", host="127.0.0.1", port=PORT, log_level="warning")

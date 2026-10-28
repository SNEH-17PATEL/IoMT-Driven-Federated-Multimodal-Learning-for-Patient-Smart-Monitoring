"""
setup.py  —  ClinIQ ICU CDSS  —  Cross-platform project setup
==============================================================
Works on macOS, Linux, and Windows.  No extra tools required.

Usage
-----
    python setup.py              # recommended: creates venv, full setup
    python setup.py --no-venv    # skip venv, install into system Python
    python setup.py --help

What it does
------------
  1. Verifies Python >= 3.12
  2. Checks Node.js >= 18 and npm  (warns if missing; backend still works)
  3. Creates a virtual environment in venv/  (unless --no-venv)
  4. Installs Python dependencies  (requirements.txt)
  5. Installs frontend dependencies  (frontend/package.json)
  6. Initialises the SQLite database  (migrate_to_db.py)
  7. Creates .env from .env.example  (if .env doesn't already exist)
  8. Prints a next-steps summary
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── Terminal colour support ───────────────────────────────────────────────────
# On Windows 10+ we can enable ANSI escape codes in the console.
# On older Windows or dumb terminals we fall back to plain ASCII.

_ANSI = False

if platform.system() == "Windows":
    try:
        import ctypes
        # Enable ENABLE_VIRTUAL_TERMINAL_PROCESSING (flag 0x0004) on stdout
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        mode   = ctypes.c_ulong()
        ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        _ANSI = True
    except Exception:
        _ANSI = False
else:
    _ANSI = sys.stdout.isatty()   # disable ANSI when piped to a file


def _c(code: str, text: str) -> str:
    """Wrap text in an ANSI colour code if the terminal supports it."""
    return f"\033[{code}m{text}\033[0m" if _ANSI else text


# ── Logging helpers ───────────────────────────────────────────────────────────

def ok(msg: str)  -> None: print(_c("32", f"  [OK]  {msg}"))
def warn(msg: str)-> None: print(_c("33", f"  [!!]  {msg}"))
def err(msg: str) -> None: print(_c("31", f"  [XX]  {msg}"))
def info(msg: str)-> None: print(_c("36", f"  -->   {msg}"))
def head(msg: str)-> None: print(_c("1;34", f"\n{msg}"))
def rule()        -> None: print(_c("1;34", "=" * 62))


# ── Subprocess helper ─────────────────────────────────────────────────────────

def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Run a command list; exit with the command's code on failure."""
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        err(f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")
        sys.exit(result.returncode)
    return result


# ── Step 1 — Python version ───────────────────────────────────────────────────

def check_python() -> None:
    head("Step 1/7  —  Python version")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 12):
        err(f"Python 3.12 or later is required.  You have {major}.{minor}.")
        err("Download the latest Python from:  https://www.python.org/downloads/")
        sys.exit(1)
    ok(f"Python {major}.{minor}  —  OK")


# ── Step 2 — Node.js / npm ────────────────────────────────────────────────────

def check_node() -> tuple:
    """Return (node_ok: bool, npm_path: str | None).  Warns but never exits."""
    head("Step 2/7  —  Node.js and npm")

    # Locate the node executable
    node = shutil.which("node")
    if not node:
        warn("Node.js not found on PATH.")
        warn("The Python backend, FL dashboard, and tests still work without it.")
        warn("To enable the React frontend, install Node.js from:")
        warn("  https://nodejs.org/  (choose the LTS version, 18 or later)")
        return False, None

    # Check the version number
    result = subprocess.run([node, "--version"], capture_output=True, text=True)
    ver_str = result.stdout.strip().lstrip("v")
    try:
        major = int(ver_str.split(".")[0])
    except ValueError:
        major = 0

    if major < 18:
        warn(f"Node.js 18+ is recommended.  You have v{ver_str}.")
        warn("Download a newer version from  https://nodejs.org/")
        node_ok = False
    else:
        ok(f"Node.js v{ver_str}  —  OK")
        node_ok = True

    # Locate npm  — on Windows it is npm.cmd, not npm
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        warn("npm not found.  Frontend install will be skipped.")
        return node_ok, None

    result = subprocess.run([npm, "--version"], capture_output=True, text=True)
    ok(f"npm v{result.stdout.strip()}  —  OK")
    return node_ok, npm


# ── Step 3 — Virtual environment ──────────────────────────────────────────────

def resolve_python(use_venv: bool) -> Path:
    head("Step 3/7  —  Virtual environment")

    if not use_venv:
        info("--no-venv flag set: packages will be installed into the system Python.")
        ok(f"Using  {sys.executable}")
        return Path(sys.executable)

    venv_dir = Path("venv")

    if venv_dir.exists():
        ok("venv/ already exists — reusing it")
    else:
        info("Creating venv/ ...")
        run([sys.executable, "-m", "venv", "venv"])
        ok("venv/ created")

    # The Python binary lives in different places on Windows vs Unix
    if platform.system() == "Windows":
        python = venv_dir / "Scripts" / "python.exe"
    else:
        python = venv_dir / "bin" / "python"

    if not python.exists():
        err(f"Could not find Python inside the new venv at: {python}")
        err("Try deleting venv/ and running setup.py again.")
        sys.exit(1)

    ok(f"Virtual environment Python: {python}")
    return python


# ── Step 4 — Python dependencies ──────────────────────────────────────────────

def install_python_deps(python: Path) -> None:
    head("Step 4/7  —  Python dependencies")

    req = Path("requirements.txt")
    if not req.exists():
        err("requirements.txt not found.")
        err("Make sure you are running setup.py from the project root folder")
        err("(the same folder that contains requirements.txt and migrate_to_db.py).")
        sys.exit(1)

    info("Upgrading pip ...")
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "-q"])

    info("Installing packages from requirements.txt  (this may take a minute) ...")
    run([str(python), "-m", "pip", "install", "-r", "requirements.txt", "-q"])
    ok("All Python packages installed")


# ── Step 5 — Frontend dependencies ───────────────────────────────────────────

def install_frontend_deps(npm: "str | None") -> None:
    head("Step 5/7  —  Frontend dependencies  (React + Vite)")

    frontend = Path("frontend")
    if not frontend.exists():
        warn("frontend/ directory not found — skipping")
        return

    if npm is None:
        warn("npm not available — skipping frontend install")
        warn("Install Node.js (https://nodejs.org/) and re-run setup.py to enable the React frontend.")
        return

    info("Running  npm install  inside frontend/  (this may take a minute) ...")
    run([npm, "install"], cwd=str(frontend))
    ok("Frontend packages installed")


# ── Step 6 — SQLite database ──────────────────────────────────────────────────

def setup_database(python: Path) -> None:
    head("Step 6/7  —  SQLite database")

    db = Path("models") / "icu_monitor.db"
    migrate = Path("migrate_to_db.py")

    if not migrate.exists():
        warn("migrate_to_db.py not found — skipping database setup")
        return

    if db.exists():
        size_mb = db.stat().st_size / (1024 * 1024)
        ok(f"Database already exists  ({size_mb:.1f} MB)  —  skipping migration")
        info("To force a fresh migration: delete models/icu_monitor.db and re-run setup.py")
        return

    info("Running migrate_to_db.py  (importing MIMIC-III data into SQLite) ...")
    run([str(python), "migrate_to_db.py"])
    ok("Database initialised")


# ── Step 7 — .env file ────────────────────────────────────────────────────────

def setup_env() -> None:
    head("Step 7/7  —  Environment file  (.env)")

    env     = Path(".env")
    example = Path(".env.example")

    if env.exists():
        ok(".env already exists — keeping it as-is")
        return

    if not example.exists():
        warn(".env.example not found.")
        warn("Create a file called .env in this folder and add:")
        warn("  GROQ_API_KEY=gsk_your_key_here")
        warn("Get a free key at  https://console.groq.com")
        return

    shutil.copy(example, env)
    ok(".env created from .env.example")
    warn("ACTION REQUIRED: open .env and set your GROQ_API_KEY")
    warn("  Get a free key at  https://console.groq.com")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(use_venv: bool, npm: "str | None") -> None:
    rule()
    print(_c("1;32", "  Setup complete!"))
    rule()

    # Activation instructions
    if use_venv:
        if platform.system() == "Windows":
            activate = r"venv\Scripts\activate"
        else:
            activate = "source venv/bin/activate"
        print(f"\n  Activate the virtual environment before running anything:")
        print(f"    {_c('32', activate)}")

    print(f"\n  {_c('1', '1. Start the FastAPI backend')}  (Terminal 1)")
    print(f"     {_c('32', 'uvicorn backend.main:app --port 8000 --reload')}")
    print(f"     API docs  →  {_c('36', 'http://localhost:8000/docs')}")

    if npm:
        print(f"\n  {_c('1', '2. Start the React frontend')}  (Terminal 2)")
        print(f"     {_c('32', 'cd frontend && npm run dev')}")
        print(f"     Dashboard  →  {_c('36', 'http://localhost:5174')}")
    else:
        print(f"\n  {_c('33', '2. React frontend: install Node.js first, then re-run setup.py')}")

    print(f"\n  {_c('1', '3. FL Simulation Dashboard')}")
    print(f"     {_c('32', 'streamlit run fl_dashboard.py')}")
    print(f"     Opens at  →  {_c('36', 'http://localhost:8501')}")

    print(f"\n  {_c('1', '4. Real Flower Federated Learning')}  (4 terminals)")
    print(f"     {_c('32', 'python server.py')}")
    print(f"     {_c('32', 'python client.py --client_id 0')}")
    print(f"     {_c('32', 'python client.py --client_id 1')}")
    print(f"     {_c('32', 'python client.py --client_id 2')}")

    print(f"\n  {_c('1', '5. Run all Python tests')}")
    print(f"     {_c('32', 'python -m pytest tests/ -v')}")
    if npm:
        print(f"\n  {_c('1', '6. Run all frontend tests')}")
        print(f"     {_c('32', 'cd frontend && npx vitest run')}")

    print(f"\n  {_c('1', '7. SQLite database inspection')}")
    print(f"     {_c('32', 'sqlite3 models/icu_monitor.db')}")
    print()
    rule()
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClinIQ ICU CDSS — cross-platform project setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help=(
            "Skip creating a virtual environment and install packages "
            "directly into the system Python."
        ),
    )
    args = parser.parse_args()
    use_venv = not args.no_venv

    # ── Header ────────────────────────────────────────────────────────────────
    rule()
    print(_c("1;34", "  ClinIQ  —  ICU Clinical Decision Support System"))
    print(_c("1;34", "  Project Setup"))
    rule()
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  Python   : {sys.version.split()[0]}  at  {sys.executable}")
    print(f"  Mode     : {'virtual environment (recommended)' if use_venv else 'system Python (--no-venv)'}")
    rule()

    # ── Guard: must run from the project root ─────────────────────────────────
    if not Path("requirements.txt").exists():
        err("requirements.txt not found in the current directory.")
        err("Please run setup.py from the project root folder:")
        err("  cd <project-folder>")
        err("  python setup.py")
        sys.exit(1)

    # ── Run all steps ─────────────────────────────────────────────────────────
    check_python()
    _node_ok, npm = check_node()
    python = resolve_python(use_venv)
    install_python_deps(python)
    install_frontend_deps(npm)
    setup_database(python)
    setup_env()
    print_summary(use_venv, npm)


if __name__ == "__main__":
    main()

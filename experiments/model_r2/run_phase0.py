"""Run ONLY Phase 0 of train_federated.py in a scratch copy of the project."""
import os, shutil, sys, time
REPO = r"<REPO>"
SCR = r"<WORKDIR>"
WORK = os.path.join(SCR, "work")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(os.path.join(WORK, "models"))
for f in ("train_federated.py", "model_utils.py", "db.py"):
    shutil.copy(os.path.join(REPO, f), WORK)
os.chdir(WORK)
sys.path.insert(0, WORK)
src = open("train_federated.py", encoding="utf-8").read()
src = src.replace("PARQUET_DIR = None",
                  'PARQUET_DIR = r"%s"' % os.path.join(SCR, "handoff", "handoff_full", "bigquery_tables"), 1)
head = src.split("# =============================================================\n# ── PHASE 1–6: FL TRAINING")[0]
head = head.replace("_db_has_data = fl_training_exists()", "_db_has_data = False")
ns = {"__name__": "phase0"}
exec(compile(head, "train_federated_phase0", "exec"), ns)
t0 = time.time()
Xte, yte = ns["run_preprocessing"]()
print("PHASE0 DONE", Xte.shape, yte.shape, round(time.time() - t0), "s", flush=True)

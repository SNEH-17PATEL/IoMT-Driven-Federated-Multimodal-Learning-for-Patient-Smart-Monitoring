"""usage: python fed_cfg.py name key=val [key=val ...]   (federated run on the NEW features, 40 rounds)"""
import sys, json, time, ast
import torch
REPO = r"<REPO>"
SCR = r"<WORKDIR>"
sys.path.insert(0, REPO)
import tune_training as t
torch.set_num_threads(2)
name = sys.argv[1]
base = dict(hidden=(256, 128, 64), dropout=0.3, lr=1e-3, lr_decay=0.99, epochs=3, batch=64, mu=0.5, weighted=False, rounds=40)
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1); base[k] = ast.literal_eval(v)
clients, test = t.load_clients(False, SCR + r"\work")
t0 = time.time(); r = t.run(base, clients, test)
print("RESULT", name, json.dumps(r), {k: v for k, v in base.items()}, round(time.time() - t0), "s", flush=True)

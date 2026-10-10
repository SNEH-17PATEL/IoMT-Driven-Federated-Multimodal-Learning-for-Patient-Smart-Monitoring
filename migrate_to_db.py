"""
One-time CSV → SQLite migration script
=======================================
Run once from icu_monitor/:

    python migrate_to_db.py

Imports all CSV data into models/icu_monitor.db.
Original CSV files are NOT deleted — they remain as backup.
"""

import os
import sqlite3
import pandas as pd
from db import DB_PATH, get_conn, init_db

def migrate():
    print("=" * 56)
    print("  ICU CDSS — CSV → SQLite Migration")
    print("=" * 56)
    print(f"  Database: {DB_PATH}")
    print()

    # Ensure all tables exist
    init_db()
    print("[1/4] Tables initialised ✓")

    conn = get_conn()

    # ── 1. Patient vitals ──────────────────────────────────────────────────
    print()
    print("[2/4] Importing patient_vitals ...")
    vitals_dir = "models/patient_vitals"
    for patient_id, fname in [(1, "patient_vitals_01.csv"),
                               (2, "patient_vitals_02.csv"),
                               (3, "patient_vitals_03.csv")]:
        path = os.path.join(vitals_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠ {fname} not found — skipping")
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"  ⚠ {fname} is empty — skipping")
            continue
        # Check if already imported
        cur = conn.cursor()
        existing = cur.execute(
            "SELECT COUNT(*) FROM patient_vitals WHERE patient = ?", (patient_id,)
        ).fetchone()[0]
        if existing > 0:
            print(f"  patient {patient_id}: already has {existing} rows — skipping")
            continue
        df.insert(0, "patient", patient_id)
        df.to_sql("patient_vitals", conn, if_exists="append", index=False)
        print(f"  patient {patient_id}: {len(df)} rows imported ✓")
    conn.commit()

    # ── 2. Prediction history ──────────────────────────────────────────────
    print()
    print("[3/4] Importing prediction_history ...")
    hist_dir = "models/prediction_history"
    col_map  = {
        "SpO₂": "SpO2", "Temp (°C)": "Temp",
        "GCS Eye": "GCS_Eye"
    }
    for patient_id, fname in [(1, "prediction_history_01.csv"),
                               (2, "prediction_history_02.csv"),
                               (3, "prediction_history_03.csv")]:
        path = os.path.join(hist_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠ {fname} not found — skipping")
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"  ⚠ {fname} is empty — skipping")
            continue
        cur = conn.cursor()
        existing = cur.execute(
            "SELECT COUNT(*) FROM prediction_history WHERE patient = ?", (patient_id,)
        ).fetchone()[0]
        if existing > 0:
            print(f"  patient {patient_id}: already has {existing} rows — skipping")
            continue
        df = df.rename(columns=col_map)
        df.insert(0, "patient", patient_id)
        df.to_sql("prediction_history", conn, if_exists="append", index=False)
        print(f"  patient {patient_id}: {len(df)} rows imported ✓")
    conn.commit()

    # ── 3. Sample patient simulation data ──────────────────────────────────
    print()
    print("[4a/4] Importing sample_patients ...")
    sp_dir = "data/sample_patients"
    for patient_id, fname in [(1, "sample_patient01_data.csv"),
                               (2, "sample_patient02_data.csv"),
                               (3, "sample_patient03_data.csv")]:
        path = os.path.join(sp_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠ {fname} not found — skipping")
            continue
        df = pd.read_csv(path)
        cur = conn.cursor()
        existing = cur.execute(
            "SELECT COUNT(*) FROM sample_patients WHERE patient = ?", (patient_id,)
        ).fetchone()[0]
        if existing > 0:
            print(f"  patient {patient_id}: already has {existing} rows — skipping")
            continue
        df.insert(0, "row_idx", range(len(df)))
        df.insert(0, "patient", patient_id)
        df.to_sql("sample_patients", conn, if_exists="append", index=False)
        print(f"  patient {patient_id}: {len(df)} rows imported ✓")
    conn.commit()

    # ── 4. FL training data ────────────────────────────────────────────────
    print()
    print("[4b/4] Importing fl_training (this may take a moment — ~48k rows) ...")
    fl_dir = "data/fl_training"
    # Check whether the fl_training table already has the full schema
    cur = conn.cursor()
    has_full_schema = False
    try:
        cols = [r[1] for r in cur.execute("PRAGMA table_info(fl_training)").fetchall()]
        has_full_schema = len(cols) > 5   # minimal placeholder has ≤3 cols
    except Exception:
        pass

    for client_id, fname in [(0, "client_0.csv"),
                              (1, "client_1.csv"),
                              (2, "client_2.csv")]:
        path = os.path.join(fl_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠ {fname} not found — skipping")
            continue
        df = pd.read_csv(path)
        # Check for existing rows
        try:
            existing = cur.execute(
                "SELECT COUNT(*) FROM fl_training WHERE client_id = ?", (client_id,)
            ).fetchone()[0]
        except Exception:
            existing = 0
        if existing > 0:
            print(f"  client {client_id}: already has {existing:,} rows — skipping")
            continue
        # Build DataFrame with client_id as first column (avoid fragmentation warning)
        df = pd.concat([pd.DataFrame({"client_id": [client_id] * len(df)}), df], axis=1)
        # First import: replace drops the minimal placeholder and creates full schema
        if_exists = "replace" if (client_id == 0 and not has_full_schema) else "append"
        df.to_sql("fl_training", conn, if_exists=if_exists, index=False)
        has_full_schema = True
        print(f"  client {client_id}: {len(df):,} rows imported ✓")
    conn.commit()
    # Rebuild index after replace/append
    if has_full_schema:   # table only exists if at least one client CSV was imported
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ft_client ON fl_training(client_id)")
    conn.close()

    # ── Summary ────────────────────────────────────────────────────────────
    conn2 = get_conn()
    cur   = conn2.cursor()
    print()
    print("=" * 56)
    print("  Migration complete — row counts")
    print("=" * 56)
    for tbl in ["patient_vitals", "prediction_history", "sample_patients", "fl_training"]:
        try:
            n = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        except Exception:
            n = 0   # fl_training absent when the FL client CSVs are not available
        print(f"  {tbl:<25} {n:>8,} rows")
    db_size = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"\n  Database size: {db_size:.1f} MB")
    print(f"  Location:      {DB_PATH}")
    print()
    print("  Original CSV files are untouched — delete them only")
    print("  after verifying the app works correctly from the DB.")
    conn2.close()


if __name__ == "__main__":
    migrate()

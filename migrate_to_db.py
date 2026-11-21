"""
One-time CSV → SQLite migration script
=======================================
Run once from icu_monitor/:

    python migrate_to_db.py

Imports all CSV data into models/icu_monitor.db.
Original CSV files are NOT deleted — they remain as backup.

Note: data/fl_training/client_*.csv are large (~190 MB each) and are gitignored.
If they are missing, fl_training will be skipped. Run train_federated.py (which
re-exports from BigQuery / MIMIC-III) to regenerate them, then re-run this script.
"""

import os
import pandas as pd
from db import DB_PATH, get_conn, init_db


# ── Required columns per table (checked before importing) ─────────────────────

_REQUIRED_SP_COLS = {
    "HR", "RR", "SpO2", "Temp", "SBP", "DBP", "MAP",
    "GCS_eye_opening", "stress_score",
}


def _check_cwd() -> None:
    """Abort loudly if we are not in the icu_monitor/ directory.

    CSV paths are relative to the repo root (models/, data/) while DB_PATH is
    absolute. Running from any other directory silently succeeds with empty
    tables because every 'not found — skipping' is swallowed.
    """
    import sys
    markers = ["db.py", "migrate_to_db.py"]
    missing = [m for m in markers if not os.path.exists(m)]
    if missing:
        print("⛔ ERROR: migrate_to_db.py must be run from the icu_monitor/ directory.")
        print(f"   Missing files in cwd: {', '.join(missing)}")
        print(f"   Current directory:    {os.getcwd()}")
        print()
        print("   Fix:")
        print("     cd /path/to/icu_monitor")
        print("     python migrate_to_db.py")
        sys.exit(1)


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop stray pandas index columns (Unnamed: 0, Unnamed: 0.1, …)."""
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def migrate():
    _check_cwd()

    print("=" * 56)
    print("  ICU CDSS — CSV → SQLite Migration")
    print("=" * 56)
    print(f"  Database: {DB_PATH}")
    print()

    # Ensure all tables exist
    init_db()
    print("[1/5] Tables initialised ✓")

    conn = get_conn()

    # ── 1. Patient vitals ──────────────────────────────────────────────────
    print()
    print("[2/5] Importing patient_vitals ...")
    vitals_dir = "models/patient_vitals"
    for patient_id, fname in [(1, "patient_vitals_01.csv"),
                               (2, "patient_vitals_02.csv"),
                               (3, "patient_vitals_03.csv")]:
        path = os.path.join(vitals_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠ {fname} not found — skipping")
            continue
        df = _drop_unnamed(pd.read_csv(path))
        if df.empty:
            print(f"  ⚠ {fname} is empty — skipping")
            continue
        cur = conn.cursor()
        existing = cur.execute(
            "SELECT COUNT(*) FROM patient_vitals WHERE patient = ?", (patient_id,)
        ).fetchone()[0]
        if existing > 0:
            print(f"  patient {patient_id}: already has {existing} rows — skipping")
            continue
        df.insert(0, "patient", patient_id)
        df.to_sql("patient_vitals", conn, if_exists="append", index=False, chunksize=500)
        print(f"  patient {patient_id}: {len(df)} rows imported ✓")
    conn.commit()

    # ── 2. Prediction history ──────────────────────────────────────────────
    print()
    print("[3/5] Importing prediction_history ...")
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
        df = _drop_unnamed(pd.read_csv(path))
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
        df.to_sql("prediction_history", conn, if_exists="append", index=False, chunksize=500)
        print(f"  patient {patient_id}: {len(df)} rows imported ✓")
    conn.commit()

    # ── 3. Sample patient simulation data ──────────────────────────────────
    print()
    print("[4/5] Importing sample_patients ...")
    sp_dir = "data/sample_patients"
    for patient_id, fname in [(1, "sample_patient01_data.csv"),
                               (2, "sample_patient02_data.csv"),
                               (3, "sample_patient03_data.csv")]:
        path = os.path.join(sp_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠ {fname} not found — skipping")
            continue
        df = _drop_unnamed(pd.read_csv(path))
        # Validate required columns are present
        missing_cols = _REQUIRED_SP_COLS - set(df.columns)
        if missing_cols:
            print(f"  ⚠ {fname} missing required columns: {sorted(missing_cols)} — skipping")
            continue
        cur = conn.cursor()
        existing = cur.execute(
            "SELECT COUNT(*) FROM sample_patients WHERE patient = ?", (patient_id,)
        ).fetchone()[0]
        if existing > 0:
            print(f"  patient {patient_id}: already has {existing} rows — skipping")
            continue
        df.insert(0, "row_idx", range(len(df)))
        df.insert(0, "patient", patient_id)
        df.to_sql("sample_patients", conn, if_exists="append", index=False, chunksize=500)
        print(f"  patient {patient_id}: {len(df)} rows imported ✓")
    conn.commit()

    # ── 4. FL training data ────────────────────────────────────────────────
    print()
    print("[5/5] Importing fl_training (large files — may take a moment) ...")
    print("      (client_*.csv are gitignored; if missing, run train_federated.py first)")
    fl_dir = "data/fl_training"

    # Determine whether the existing fl_training table already has the full schema.
    # Uses sofa_score presence rather than a column-count heuristic.
    cur = conn.cursor()
    first_import_done = False
    try:
        cols = [r[1] for r in cur.execute("PRAGMA table_info(fl_training)").fetchall()]
        first_import_done = "sofa_score" in cols
    except Exception:
        pass

    for client_id, fname in [(0, "client_0.csv"),
                              (1, "client_1.csv"),
                              (2, "client_2.csv")]:
        path = os.path.join(fl_dir, fname)
        if not os.path.exists(path):
            print(f"  ⚠ {fname} not found — skipping")
            print(f"     (this is a large gitignored CSV; see the note above)")
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
        # First CSV that needs importing: replace the placeholder (or absent) table
        # so the full ~110-column schema is set. Decoupled from client_id so it works
        # even when client_0.csv is missing and client_1.csv is the first file found.
        if_exists = "append" if first_import_done else "replace"
        df.to_sql("fl_training", conn, if_exists=if_exists, index=False, chunksize=1000)
        first_import_done = True
        print(f"  client {client_id}: {len(df):,} rows imported ✓")
    conn.commit()

    # Rebuild index after any write to fl_training
    if cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fl_training'"
    ).fetchone():
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ft_client ON fl_training(client_id)"
        )
    conn.close()

    # ── Summary ────────────────────────────────────────────────────────────
    conn2 = get_conn()
    cur2  = conn2.cursor()
    print()
    print("=" * 56)
    print("  Migration complete — row counts")
    print("=" * 56)
    row_counts: dict[str, int] = {}
    for tbl in ["patient_vitals", "prediction_history", "sample_patients", "fl_training"]:
        exists = cur2.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
        ).fetchone()
        n = cur2.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0] if exists else 0
        row_counts[tbl] = n
        print(f"  {tbl:<25} {n:>8,} rows")
    db_size = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"\n  Database size: {db_size:.1f} MB")
    print(f"  Location:      {DB_PATH}")
    conn2.close()

    # Warn if any expected table ended up empty (partial migration)
    empty = [t for t in ["patient_vitals", "sample_patients"] if row_counts.get(t, 0) == 0]
    if empty:
        print()
        print("  ⚠ WARNING: the following tables are empty — migration may be incomplete:")
        for t in empty:
            print(f"    • {t}")
        print("  Re-check that the source CSV directories exist and re-run.")
    else:
        print()
        print("  Original CSV files are untouched — delete them only")
        print("  after verifying the app works correctly from the DB.")


if __name__ == "__main__":
    migrate()

"""Create role/db, extensions, schema, and load SQL dump CSVs."""
from __future__ import annotations

import csv
import os
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

HOST = os.getenv("PGHOST", "127.0.0.1")
PORT = os.getenv("PGPORT", "5432")
DB = os.getenv("PGDATABASE", "super_pass")
USER = os.getenv("PGUSER", "vivek")
PASSWORD = os.getenv("PGPASSWORD", "bruno@9588")
ZIP_PATH = ROOT / "SQL_dump-20260813T141733Z-1-001.zip"
EXTRACT_DIR = ROOT / "SQL_dump_extracted"


def connect_super():
    # Installer password matched vivek's intended password in probe
    for pw in (PASSWORD, "bruno@9588", "postgres"):
        try:
            conn = psycopg2.connect(
                host=HOST,
                port=PORT,
                dbname="postgres",
                user="postgres",
                password=pw,
                connect_timeout=8,
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            print(f"superuser connected with password len={len(pw)}")
            return conn
        except Exception:
            continue
    raise RuntimeError("Could not connect as postgres superuser")


def ensure_role_and_db(conn) -> None:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (USER,))
    if not cur.fetchone():
        cur.execute(f"CREATE ROLE {USER} LOGIN PASSWORD %s", (PASSWORD,))
        print(f"created role {USER}")
    else:
        cur.execute(f"ALTER ROLE {USER} WITH LOGIN PASSWORD %s", (PASSWORD,))
        print(f"updated password for {USER}")

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB,))
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {DB} OWNER {USER}")
        print(f"created database {DB}")
    else:
        cur.execute(f"ALTER DATABASE {DB} OWNER TO {USER}")
        print(f"database {DB} already exists; owner set to {USER}")

    cur.execute(f"GRANT ALL PRIVILEGES ON DATABASE {DB} TO {USER}")


def connect_app():
    conn = psycopg2.connect(
        host=HOST,
        port=PORT,
        dbname=DB,
        user=USER,
        password=PASSWORD,
        connect_timeout=8,
    )
    conn.autocommit = True
    return conn


def ensure_extensions(conn) -> None:
    cur = conn.cursor()
    for ext in ("vector", "pg_trgm"):
        try:
            cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
            print(f"extension OK: {ext}")
        except Exception as e:
            print(f"extension FAIL {ext}: {e}")
            if ext == "vector":
                print(
                    "WARNING: pgvector not installed. Schema uses vector columns. "
                    "Will try to create tables without vector columns if needed."
                )


def extract_zip() -> Path:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"Missing dump zip: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(EXTRACT_DIR)
    dump = EXTRACT_DIR / "SQL_dump"
    if not dump.is_dir():
        # maybe nested differently
        candidates = list(EXTRACT_DIR.rglob("inventory_backup_*.csv"))
        if candidates:
            return candidates[0].parent
        raise FileNotFoundError("SQL_dump folder not found after extract")
    print(f"extracted to {dump}")
    return dump


def run_sql_file(conn, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    cur = conn.cursor()
    try:
        cur.execute(sql)
        print(f"applied schema: {path.name}")
    except Exception as e:
        print(f"schema apply error ({path.name}): {e}")
        # Try without vector columns if vector missing
        if "vector" in str(e).lower() or "type" in str(e).lower():
            print("Retrying schema with vector types replaced by text placeholders...")
            patched = sql
            patched = patched.replace("vector(1536)", "double precision[]")
            patched = patched.replace("vector(768)", "double precision[]")
            try:
                cur.execute(patched)
                print(f"applied patched schema: {path.name}")
            except Exception as e2:
                print(f"patched schema still failed: {e2}")
                raise


def table_count(conn, table: str) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])


def load_csv(conn, table: str, csv_path: Path) -> None:
    if table_count(conn, table) > 0:
        print(f"{table} already has data ({table_count(conn, table)} rows) — skip load")
        return
    print(f"Loading {csv_path.name} into {table} (this may take several minutes)...")
    cur = conn.cursor()
    # Use COPY FROM STDIN via copy_expert for speed
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline()
        cols = [c.strip().strip('"') for c in header.strip().split(",")]
        col_list = ", ".join(f'"{c}"' if c.lower() != c else c for c in cols)
        # rebuild stream with header for COPY HEADER
        f.seek(0)
        sql = f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, HEADER true)"
        try:
            cur.copy_expert(sql, f)
            print(f"loaded {table}: {table_count(conn, table)} rows")
        except Exception as e:
            print(f"COPY failed ({e}); trying psycopg2 fallback without explicit columns...")
            conn.rollback()
            conn.autocommit = True
            f.seek(0)
            sql2 = f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)"
            cur.copy_expert(sql2, f)
            print(f"loaded {table}: {table_count(conn, table)} rows")


def grant_schema_privs(super_conn) -> None:
    # connect to app db as superuser for grants
    conn = psycopg2.connect(
        host=HOST, port=PORT, dbname=DB, user="postgres", password=PASSWORD, connect_timeout=8
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"GRANT USAGE ON SCHEMA soap TO {USER}")
    cur.execute(f"GRANT USAGE ON SCHEMA kb TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA soap TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA kb TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA soap TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA kb TO {USER}")
    cur.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA soap GRANT ALL ON TABLES TO {USER}")
    cur.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA kb GRANT ALL ON TABLES TO {USER}")
    print("granted schema privileges to", USER)
    conn.close()


def main() -> int:
    print("=== 1) Ensure role + database ===")
    super_conn = connect_super()
    ensure_role_and_db(super_conn)
    super_conn.close()

    print("=== 2) Connect as app user ===")
    # App user may not own extensions yet — use superuser for schema setup
    admin = psycopg2.connect(
        host=HOST, port=PORT, dbname=DB, user="postgres", password=PASSWORD, connect_timeout=8
    )
    admin.autocommit = True

    print("=== 3) Extensions ===")
    ensure_extensions(admin)

    print("=== 4) Extract dump ===")
    dump_dir = extract_zip()

    print("=== 5) Apply schemas ===")
    schema_files = sorted(dump_dir.glob("*schema*.sql"))
    if not schema_files:
        schema_files = list(dump_dir.glob("*.sql"))
    for sf in schema_files:
        run_sql_file(admin, sf)

    print("=== 6) Load CSVs ===")
    inv = next(dump_dir.glob("inventory_backup_*.csv"), None)
    svc = next(dump_dir.glob("service_master_backup_*.csv"), None)
    if not inv or not svc:
        print("CSV files missing in", dump_dir)
        return 1
    # Load as superuser then grant
    load_csv(admin, "soap.inventory", inv)
    load_csv(admin, "soap.service_master", svc)

    print("=== 7) Grants ===")
    # create schemas may already exist
    cur = admin.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS soap")
    cur.execute("CREATE SCHEMA IF NOT EXISTS kb")
    admin.close()
    grant_schema_privs(None)

    print("=== 8) Verify as app user ===")
    app = connect_app()
    for tbl in ("soap.inventory", "soap.service_master"):
        print(f"{tbl}: {table_count(app, tbl)} rows")
    cur = app.cursor()
    cur.execute(
        "SELECT location_id, COUNT(*) FROM soap.inventory "
        "GROUP BY location_id ORDER BY COUNT(*) DESC LIMIT 5"
    )
    print("top locations:", cur.fetchall())
    app.close()

    # kb_ner_db check
    sys.path.insert(0, str(ROOT))
    from kb_ner_db import get_pg_conn

    c = get_pg_conn(reuse=False)
    cur = c.cursor()
    cur.execute("SELECT COUNT(*) FROM soap.inventory")
    print("kb_ner_db inventory count:", cur.fetchone()[0])
    print("SUCCESS — DB connected and loaded")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("FATAL:", type(e).__name__, e)
        raise

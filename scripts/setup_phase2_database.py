"""
One-shot PostgreSQL setup for Phase 1 grounding + Phase 2 knowledge atoms.

Creates/updates:
  - Database role + super_pass database (when postgres superuser is reachable)
  - Extensions: vector, pg_trgm, fuzzystrmatch
  - soap.inventory, soap.service_master, kb.vitals_registry
  - kb.assertion_types, kb.attributes_schema (Phase 2 instruction manual)
  - Search indexes via kb_ner_db helpers
  - Clinic catalog from SQL_dump CSVs when present, else demo seed rows

After success, writes/updates .env with PG* vars and CLINIC_ID=1 (Master Doc default).

Usage (repo root):
  python scripts/setup_phase2_database.py
  python scripts/setup_phase2_database.py --demo-only   # skip CSV load, use demo seed
  python scripts/setup_phase2_database.py --force-demo  # load demo even if CSV exists
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("Install psycopg2-binary: pip install psycopg2-binary")
    raise SystemExit(2)

HOST = os.getenv("PGHOST", "127.0.0.1")
PORT = os.getenv("PGPORT", "5432")
DB = os.getenv("PGDATABASE", "super_pass")
USER = os.getenv("PGUSER", "vivek")
PASSWORD = os.getenv("PGPASSWORD") or "bruno@9588"

SCHEMA_SQL = ROOT / "scripts" / "local_soap_schema.sql"
PHASE2_SEED_SQL = ROOT / "scripts" / "seed_phase2_kb_tables.sql"
DEMO_SEED_SQL = ROOT / "scripts" / "seed_demo_clinic_data.sql"
ZIP_PATH = ROOT / "SQL_dump-20260813T141733Z-1-001.zip"
EXTRACT_DIR = ROOT / "SQL_dump_extracted"
ENV_EXAMPLE = ROOT / ".env.deploy.example"
ENV_PATH = ROOT / ".env"


def _connect(dbname: str, user: str, password: str):
    return psycopg2.connect(
        host=HOST,
        port=PORT,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=10,
    )


def probe_superuser() -> tuple[str, str] | None:
    env_pw = os.getenv("PGPASSWORD") or ""
    candidates: list[tuple[str, str]] = []
    for pw in (env_pw, PASSWORD, "bruno@9588", "postgres"):
        if pw and ("postgres", pw) not in candidates:
            candidates.append(("postgres", pw))
    for user, pw in candidates:
        try:
            conn = _connect("postgres", user, pw)
            conn.close()
            return user, pw
        except Exception:
            continue
    return None


def probe_app_user(password: str) -> bool:
    try:
        conn = _connect(DB, USER, password)
        conn.close()
        return True
    except Exception:
        return False


def ensure_role_and_database(super_pw: str) -> None:
    conn = _connect("postgres", "postgres", super_pw)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (USER,))
    if not cur.fetchone():
        cur.execute(f"CREATE ROLE {USER} LOGIN PASSWORD %s", (PASSWORD,))
        print(f"created role {USER}")
    else:
        cur.execute(f"ALTER ROLE {USER} WITH LOGIN PASSWORD %s", (PASSWORD,))
        print(f"updated password for role {USER}")

    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB,))
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {DB} OWNER {USER}")
        print(f"created database {DB}")
    else:
        cur.execute(f"ALTER DATABASE {DB} OWNER TO {USER}")
        print(f"database {DB} exists; owner set to {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON DATABASE {DB} TO {USER}")
    conn.close()


def admin_conn(super_pw: str):
    conn = _connect(DB, "postgres", super_pw)
    conn.autocommit = True
    return conn


def run_sql_file(conn, path: Path, *, label: str | None = None) -> None:
    sql = path.read_text(encoding="utf-8")
    cur = conn.cursor()
    try:
        cur.execute(sql)
        print(f"applied: {label or path.name}")
    except Exception as e:
        err = str(e).lower()
        if "vector" in err and path == SCHEMA_SQL:
            print("retrying schema without vector columns (install pgvector for full vector search)...")
            patched = sql.replace("vector(1536)", "text").replace("vector(768)", "text")
            cur.execute(patched)
            print(f"applied patched schema: {path.name}")
        else:
            raise


def ensure_extensions(conn) -> dict[str, bool]:
    cur = conn.cursor()
    status: dict[str, bool] = {}
    for ext in ("vector", "pg_trgm", "fuzzystrmatch"):
        try:
            cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
            status[ext] = True
            print(f"extension OK: {ext}")
        except Exception as e:
            status[ext] = False
            print(f"extension WARN {ext}: {e}")
    return status


def table_count(conn, table: str) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])


def find_dump_dir() -> Path | None:
    direct = EXTRACT_DIR / "SQL_dump"
    if direct.is_dir():
        return direct
    if ZIP_PATH.is_file():
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(EXTRACT_DIR)
        if direct.is_dir():
            return direct
        candidates = list(EXTRACT_DIR.rglob("inventory_backup_*.csv"))
        if candidates:
            return candidates[0].parent
    nested = list(ROOT.glob("**/inventory_backup_*.csv"))
    if nested:
        return nested[0].parent
    return None


def load_csv_if_empty(conn, table: str, csv_path: Path) -> int:
    if table_count(conn, table) > 0:
        n = table_count(conn, table)
        print(f"{table}: already has {n} rows — skip CSV load")
        return n
    print(f"loading {csv_path.name} -> {table} (may take a few minutes)...")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        cur = conn.cursor()
        cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)", f)
    n = table_count(conn, table)
    print(f"loaded {table}: {n} rows")
    return n


def null_placeholder_embeddings(conn) -> None:
    cur = conn.cursor()
    for table in ("soap.inventory", "soap.service_master"):
        for col in ("vector_embedding", "internal_description_vector", "vector_embedding_vetbert"):
            try:
                cur.execute(
                    f"""
                    UPDATE {table} SET {col} = NULL
                    WHERE {col} IS NOT NULL AND btrim({col}::text) = ''
                    """
                )
            except Exception:
                conn.rollback()
                conn.autocommit = True


def grant_app_privileges(conn) -> None:
    cur = conn.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS soap")
    cur.execute("CREATE SCHEMA IF NOT EXISTS kb")
    cur.execute(f"GRANT USAGE ON SCHEMA soap TO {USER}")
    cur.execute(f"GRANT USAGE ON SCHEMA kb TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA soap TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA kb TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA soap TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA kb TO {USER}")
    cur.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA soap GRANT ALL ON TABLES TO {USER}")
    cur.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA kb GRANT ALL ON TABLES TO {USER}")
    print(f"granted schema privileges to {USER}")


def ensure_env_file() -> None:
    lines: list[str] = []
    if ENV_PATH.is_file():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    elif ENV_EXAMPLE.is_file():
        lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    def upsert(key: str, value: str) -> None:
        nonlocal lines
        prefix = f"{key}="
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                lines[i] = f"{prefix}{value}"
                return
        lines.append(f"{prefix}{value}")

    upsert("PGHOST", HOST)
    upsert("PGPORT", str(PORT))
    upsert("PGDATABASE", DB)
    upsert("PGUSER", USER)
    if PASSWORD:
        upsert("PGPASSWORD", PASSWORD)
    upsert("CLINIC_ID", "1")

    ENV_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {ENV_PATH} (CLINIC_ID=1, PGDATABASE={DB})")


def verify_phase2(conn) -> bool:
    ok = True
    checks = [
        ("kb.assertion_types", 1),
        ("kb.attributes_schema", 1),
        ("soap.inventory", 1),
        ("soap.service_master", 1),
        ("kb.vitals_registry", 1),
    ]
    cur = conn.cursor()
    for table, min_rows in checks:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            n = int(cur.fetchone()[0])
            status = "OK" if n >= min_rows else "LOW"
            print(f"  verify {table}: {n} rows [{status}]")
            if n < min_rows:
                ok = False
        except Exception as e:
            print(f"  verify {table}: MISSING — {e}")
            ok = False

    cur.execute(
        "SELECT location_id, COUNT(*) FROM soap.inventory "
        "GROUP BY location_id ORDER BY COUNT(*) DESC LIMIT 5"
    )
    print("  top inventory location_ids:", cur.fetchall())
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup PostgreSQL for Phase 2")
    parser.add_argument("--demo-only", action="store_true", help="Skip CSV load; use demo seed only")
    parser.add_argument("--force-demo", action="store_true", help="Load demo seed even when CSV data exists")
    args = parser.parse_args()

    print("=== Phase 2 database setup ===")
    print(f"target: {USER}@{HOST}:{PORT}/{DB}")

    super = probe_superuser()
    if super:
        _, super_pw = super
        print("postgres superuser: OK")
        ensure_role_and_database(super_pw)
        conn = admin_conn(super_pw)
    elif probe_app_user(PASSWORD):
        print("app user connects; skipping role/database creation")
        conn = _connect(DB, USER, PASSWORD)
        conn.autocommit = True
        super_pw = None
    else:
        print(
            "FATAL: Cannot connect to PostgreSQL.\n"
            "  1. Install PostgreSQL 15+ and start the service\n"
            "  2. Set PGPASSWORD in .env (or use default bruno@9588 for role vivek)\n"
            "  3. Re-run: python scripts/setup_phase2_database.py"
        )
        return 1

    print("\n=== Extensions ===")
    ext_status = ensure_extensions(conn)

    print("\n=== Schema (soap + kb base tables) ===")
    run_sql_file(conn, SCHEMA_SQL)

    print("\n=== Phase 2 instruction tables ===")
    run_sql_file(conn, PHASE2_SEED_SQL)

    loaded_from_csv = False
    if not args.demo_only:
        dump_dir = find_dump_dir()
        if dump_dir:
            inv = next(dump_dir.glob("inventory_backup_*.csv"), None)
            svc = next(dump_dir.glob("service_master_backup_*.csv"), None)
            if inv and svc:
                print(f"\n=== Clinic catalog from {dump_dir} ===")
                load_csv_if_empty(conn, "soap.inventory", inv)
                load_csv_if_empty(conn, "soap.service_master", svc)
                null_placeholder_embeddings(conn)
                loaded_from_csv = table_count(conn, "soap.inventory") > 10
            else:
                print("CSV files not found in dump dir — will use demo seed")
        else:
            print("\nNo SQL_dump zip/extract found — will use demo clinic seed")
            print(f"  (Place {ZIP_PATH.name} in repo root for full clinic catalog)")

    inv_n = table_count(conn, "soap.inventory")
    svc_n = table_count(conn, "soap.service_master")
    if args.demo_only or args.force_demo or inv_n == 0 or svc_n == 0:
        print("\n=== Demo clinic seed (location_id=8, CLINIC_ID=1) ===")
        run_sql_file(conn, DEMO_SEED_SQL)

    print("\n=== Vitals registry + search indexes ===")
    from kb_ner_db import (
        ensure_fuzzystrmatch,
        ensure_kb_search_indexes,
        ensure_soft_gate_indexes,
        ensure_vector_extension,
        ensure_vitals_registry_table,
        get_pg_conn,
        seed_vitals_registry,
        vector_extension_available,
    )

    seed_vitals_registry(conn, embed=False)
    ensure_vitals_registry_table(conn)
    ensure_fuzzystrmatch(conn)
    if ext_status.get("vector"):
        ensure_vector_extension(conn)
    ensure_kb_search_indexes(conn)
    ensure_soft_gate_indexes(conn)

    print("\n=== Grants ===")
    grant_app_privileges(conn)
    conn.close()

    print("\n=== .env ===")
    ensure_env_file()
    load_dotenv(ENV_PATH, override=True)

    print("\n=== Verify (app user) ===")
    app = get_pg_conn(reuse=False)
    ok = verify_phase2(app)
    print("vector_extension_available:", vector_extension_available(app))

    from kb_ner_local_search import search_local_inventory_topk

    hits = search_local_inventory_topk(
        app, "Bravecto", entity_kind="Product", clinic_id=1, top_k=3, logger=None
    )
    print("local search bravecto hits:", len(hits or []))
    if hits:
        h = hits[0]
        print("  top:", h.get("display_name") or h.get("item_name"), "score=", h.get("match_score"))

    # Phase 2 schema loaders
    from SOAP_notes_billing_phase2_kb_atoms import get_all_attributes_schema, get_assertion_types

    at = get_assertion_types(conn=app)
    asc = get_all_attributes_schema(conn=app)
    print(f"Phase 2 KB schema: {len(at)} assertion types, {len(asc)} attribute kinds")

    inv_final = table_count(app, "soap.inventory")
    if ok and at and asc:
        print("\nSUCCESS — PostgreSQL is ready for Phase 1 grounding + Phase 2")
        if inv_final < 50 and not loaded_from_csv:
            print("NOTE: Small catalog. Add SQL_dump zip to repo root for full clinic inventory.")
        return 0

    print("\nPARTIAL — check warnings above")
    return 1 if not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())

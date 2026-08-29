"""Finish DB setup: compatible schema + CSV load + verify kb_ner_db."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
import psycopg2

HOST = os.getenv("PGHOST", "127.0.0.1")
PORT = os.getenv("PGPORT", "5432")
DB = os.getenv("PGDATABASE", "super_pass")
USER = os.getenv("PGUSER", "vivek")
PASSWORD = os.getenv("PGPASSWORD", "bruno@9588")
DUMP = ROOT / "SQL_dump_extracted" / "SQL_dump"
SCHEMA = ROOT / "scripts" / "local_soap_schema.sql"


def admin_conn():
    conn = psycopg2.connect(
        host=HOST, port=PORT, dbname=DB, user="postgres", password=PASSWORD, connect_timeout=10
    )
    conn.autocommit = True
    return conn


def app_conn():
    conn = psycopg2.connect(
        host=HOST, port=PORT, dbname=DB, user=USER, password=PASSWORD, connect_timeout=10
    )
    conn.autocommit = True
    return conn


def main() -> int:
    print("Connecting as postgres to", DB)
    admin = admin_conn()
    cur = admin.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    print("pg_trgm OK")

    # Drop incomplete tables if prior failed attempt left junk
    cur.execute("DROP TABLE IF EXISTS soap.inventory CASCADE")
    cur.execute("DROP TABLE IF EXISTS soap.service_master CASCADE")
    cur.execute("DROP TABLE IF EXISTS kb.vitals_registry CASCADE")

    sql = SCHEMA.read_text(encoding="utf-8")
    cur.execute(sql)
    print("schema applied:", SCHEMA.name)

    inv = DUMP / "inventory_backup_20260221_192846.csv"
    svc = DUMP / "service_master_backup_20260221_192846.csv"
    if not inv.is_file() or not svc.is_file():
        print("CSV missing — extract the zip first")
        return 1

    print("Loading inventory CSV (large — please wait)...")
    with inv.open("r", encoding="utf-8", newline="") as f:
        cur.copy_expert(
            "COPY soap.inventory FROM STDIN WITH (FORMAT csv, HEADER true)", f
        )
    cur.execute("SELECT COUNT(*) FROM soap.inventory")
    print("soap.inventory rows:", cur.fetchone()[0])

    print("Loading service_master CSV...")
    with svc.open("r", encoding="utf-8", newline="") as f:
        cur.copy_expert(
            "COPY soap.service_master FROM STDIN WITH (FORMAT csv, HEADER true)", f
        )
    cur.execute("SELECT COUNT(*) FROM soap.service_master")
    print("soap.service_master rows:", cur.fetchone()[0])

    # Null out placeholder embedding text so vector SQL branches skip
    cur.execute(
        """
        UPDATE soap.inventory SET
          vector_embedding = NULLIF(TRIM(vector_embedding), ''),
          internal_description_vector = NULLIF(TRIM(internal_description_vector), ''),
          vector_embedding_vetbert = NULLIF(TRIM(vector_embedding_vetbert), '')
        """
    )
    cur.execute(
        """
        UPDATE soap.service_master SET
          vector_embedding = NULLIF(TRIM(vector_embedding), ''),
          internal_description_vector = NULLIF(TRIM(internal_description_vector), ''),
          vector_embedding_vetbert = NULLIF(TRIM(vector_embedding_vetbert), '')
        """
    )

    cur.execute(f"GRANT USAGE ON SCHEMA soap TO {USER}")
    cur.execute(f"GRANT USAGE ON SCHEMA kb TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA soap TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA kb TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA soap TO {USER}")
    cur.execute(f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA kb TO {USER}")
    print("grants OK")

    cur.execute(
        "SELECT location_id, COUNT(*) c FROM soap.inventory "
        "GROUP BY location_id ORDER BY c DESC LIMIT 8"
    )
    locs = cur.fetchall()
    print("top location_ids:", locs)
    admin.close()

    print("Verifying as app user", USER)
    app = app_conn()
    cur = app.cursor()
    cur.execute("SELECT COUNT(*) FROM soap.inventory")
    inv_n = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM soap.service_master")
    svc_n = cur.fetchone()[0]
    print(f"app user sees inventory={inv_n} services={svc_n}")
    app.close()

    sys.path.insert(0, str(ROOT))
    from kb_ner_db import get_pg_conn

    c = get_pg_conn(reuse=False)
    cur = c.cursor()
    cur.execute("SELECT COUNT(*) FROM soap.inventory")
    print("kb_ner_db OK inventory=", cur.fetchone()[0])
    print("SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

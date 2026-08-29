"""One-off DB connectivity check for VetAI."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

try:
    import psycopg2
except ImportError:
    print("NEED_PSYCOPG2: pip install psycopg2-binary")
    sys.exit(2)

host = os.getenv("PGHOST", "127.0.0.1")
port = os.getenv("PGPORT", "5432")
dbname = os.getenv("PGDATABASE", "vetinstant")
user = os.getenv("PGUSER", "vivek")
password = os.getenv("PGPASSWORD")
clinic = os.getenv("CLINIC_ID", "1")

print(f"Connecting {user}@{host}:{port}/{dbname} ...")

try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=8,
    )
except Exception as e:
    print(f"CONNECT_FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT current_database(), current_user")
print("CONNECTED:", cur.fetchone())

cur.execute(
    "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm') ORDER BY 1"
)
print("extensions:", [r[0] for r in cur.fetchall()])

for tbl in ("soap.inventory", "soap.service_master", "kb.vitals_registry"):
    try:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        print(f"{tbl}: {cur.fetchone()[0]} rows")
    except Exception as e:
        conn.rollback()
        print(f"{tbl}: MISSING — {str(e).splitlines()[0][:140]}")

try:
    cur.execute(
        "SELECT location_id, COUNT(*) AS c FROM soap.inventory "
        "GROUP BY location_id ORDER BY c DESC LIMIT 8"
    )
    rows = cur.fetchall()
    print("top location_ids:", rows)
    print("CLINIC_ID env:", clinic)
except Exception as e:
    conn.rollback()
    print("location_query:", str(e).splitlines()[0][:140])

# Also verify kb_ner_db path
try:
    sys.path.insert(0, str(ROOT))
    from kb_ner_db import get_pg_conn

    c2 = get_pg_conn(reuse=False)
    cur2 = c2.cursor()
    cur2.execute("SELECT COUNT(*) FROM soap.inventory")
    print("kb_ner_db.get_pg_conn OK, inventory=", cur2.fetchone()[0])
except Exception as e:
    print("kb_ner_db check:", type(e).__name__, e)

conn.close()
print("DONE")

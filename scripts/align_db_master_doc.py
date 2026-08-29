"""Seed Phase 2 KB tables + set CLINIC_ID per Master Doc; smoke-test local search."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)
import psycopg2

HOST = os.getenv("PGHOST", "127.0.0.1")
PORT = os.getenv("PGPORT", "5432")
DB = os.getenv("PGDATABASE", "super_pass")
PASSWORD = os.getenv("PGPASSWORD", "bruno@9588")
USER = os.getenv("PGUSER", "vivek")
SEED = ROOT / "scripts" / "seed_phase2_kb_tables.sql"


def main() -> int:
    admin = psycopg2.connect(
        host=HOST, port=PORT, dbname=DB, user="postgres", password=PASSWORD
    )
    admin.autocommit = True
    cur = admin.cursor()
    cur.execute(SEED.read_text(encoding="utf-8"))
    cur.execute(f"GRANT USAGE ON SCHEMA kb TO {USER}")
    cur.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA kb TO {USER}")
    cur.execute("SELECT COUNT(*) FROM kb.assertion_types")
    print("assertion_types:", cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM kb.attributes_schema")
    print("attributes_schema:", cur.fetchone()[0])
    admin.close()

    # Master Doc default CLINIC_ID=1; code maps clinic_id 1 -> location_id 8
    env_path = ROOT / ".env"
    text = env_path.read_text(encoding="utf-8")
    if "CLINIC_ID=8" in text:
        env_path.write_text(text.replace("CLINIC_ID=8", "CLINIC_ID=1"), encoding="utf-8")
        print("Set CLINIC_ID=1 (Master Doc default; local search maps 1 -> location 8)")
    load_dotenv(env_path, override=True)

    from kb_ner_db import get_pg_conn, ensure_fuzzystrmatch, vector_extension_available
    from kb_ner_local_search import search_local_inventory_topk

    conn = get_pg_conn(reuse=False)
    print("fuzzystrmatch:", ensure_fuzzystrmatch(conn))
    print("vector_available:", vector_extension_available(conn))
    hits = search_local_inventory_topk(
        conn,
        "Bravecto",
        entity_kind="Product",
        clinic_id=1,
        top_k=5,
        logger=None,
    )
    print("bravecto_hits:", len(hits or []))
    if hits:
        print("top:", hits[0].get("display_name") or hits[0].get("item_name"), "score", hits[0].get("match_score"))
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

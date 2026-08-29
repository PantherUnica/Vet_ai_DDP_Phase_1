"""Check PG extensions and try to enable vector/fuzzystrmatch/pg_trgm."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
import psycopg2

PASSWORD = os.getenv("PGPASSWORD", "bruno@9588")
HOST = os.getenv("PGHOST", "127.0.0.1")
PORT = os.getenv("PGPORT", "5432")
DB = os.getenv("PGDATABASE", "super_pass")


def main() -> None:
    conn = psycopg2.connect(
        host=HOST, port=PORT, dbname=DB, user="postgres", password=PASSWORD
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT version()")
    print("version:", cur.fetchone()[0][:80])
    cur.execute("SELECT extname FROM pg_extension ORDER BY 1")
    print("installed:", [r[0] for r in cur.fetchall()])
    cur.execute(
        "SELECT name, default_version, installed_version "
        "FROM pg_available_extensions "
        "WHERE name IN ('vector','fuzzystrmatch','pg_trgm') ORDER BY 1"
    )
    print("available:", cur.fetchall())
    for ext in ("pg_trgm", "fuzzystrmatch", "vector"):
        try:
            cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
            print(f"CREATE {ext}: OK")
        except Exception as e:
            print(f"CREATE {ext}: FAIL — {e}")
    conn.close()


if __name__ == "__main__":
    main()

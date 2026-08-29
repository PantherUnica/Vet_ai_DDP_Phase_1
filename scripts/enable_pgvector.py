"""Enable pgvector and align soap.* embedding columns with Master Doc types."""
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


def col_type(cur, schema: str, table: str, col: str) -> str | None:
    cur.execute(
        """
        SELECT data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema=%s AND table_name=%s AND column_name=%s
        """,
        (schema, table, col),
    )
    row = cur.fetchone()
    if not row:
        return None
    return f"{row[0]}/{row[1]}"


def ensure_vector_col(cur, table: str, col: str, dims: int) -> None:
    """Convert text embedding column to vector(dims) when needed."""
    t = col_type(cur, "soap", table, col)
    print(f"  {table}.{col}: {t}")
    if t is None:
        cur.execute(
            f"ALTER TABLE soap.{table} ADD COLUMN {col} vector({dims})"
        )
        print(f"  + added soap.{table}.{col} vector({dims})")
        return
    if t.endswith("/vector"):
        print(f"  = already vector")
        return
    # text / character varying workaround from local load
    cur.execute(
        f"""
        ALTER TABLE soap.{table}
          ALTER COLUMN {col} TYPE vector({dims})
          USING CASE
            WHEN {col} IS NULL OR btrim({col}::text) = '' THEN NULL
            ELSE {col}::vector
          END
        """
    )
    print(f"  -> converted to vector({dims})")


def main() -> int:
    conn = psycopg2.connect(
        host=HOST, port=PORT, dbname=DB, user="postgres", password=PASSWORD
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(
        "SELECT extname, extversion FROM pg_extension WHERE extname='vector'"
    )
    print("extension:", cur.fetchone())

    for table, col, dims in [
        ("inventory", "vector_embedding", 1536),
        ("inventory", "vector_embedding_vetbert", 768),
        ("service_master", "vector_embedding", 1536),
        ("service_master", "vector_embedding_vetbert", 768),
    ]:
        try:
            ensure_vector_col(cur, table, col, dims)
        except Exception as e:
            print(f"  !! {table}.{col}: {e}")
            conn.rollback()
            conn.autocommit = True

    # Smoke: type exists + distance operator
    cur.execute("SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector")
    print("distance_ok:", cur.fetchone()[0])

    cur.execute(
        "SELECT name FROM pg_available_extensions WHERE name='vector'"
    )
    print("available:", cur.fetchall())
    conn.close()

    # App-level check (clear process cache by fresh import)
    from kb_ner_db import get_pg_conn, ensure_vector_extension, vector_extension_available
    import kb_ner_db as dbmod

    dbmod._VECTOR_EXT_CACHE = None
    c2 = get_pg_conn(reuse=False)
    print("ensure_vector_extension:", ensure_vector_extension(c2))
    print("vector_extension_available:", vector_extension_available(c2))
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

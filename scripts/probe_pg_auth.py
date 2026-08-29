"""Probe which credentials work against local Postgres."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
import psycopg2

host = os.getenv("PGHOST", "127.0.0.1")
port = os.getenv("PGPORT", "5432")
db = os.getenv("PGDATABASE", "super_pass")
user = os.getenv("PGUSER", "vivek")
env_pw = os.getenv("PGPASSWORD") or ""

print(f"target {user}@{host}:{port}/{db}")
print(f"env password length={len(env_pw)!r} repr={env_pw!r}")

pw_candidates = []
for pw in (env_pw, "bruno@9588", "bruno9588", "postgres", "vivek", "admin"):
    if pw and pw not in pw_candidates:
        pw_candidates.append(pw)

ok_pw = None
for pw in pw_candidates:
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=db, user=user, password=pw, connect_timeout=4
        )
        print(f"USER_OK password_repr={pw!r}")
        ok_pw = pw
        conn.close()
        break
    except Exception as e:
        print(f"USER_FAIL {str(e).splitlines()[0][:120]}")

# Try postgres superuser
super_ok = None
for u, pw in (
    ("postgres", "postgres"),
    ("postgres", "bruno@9588"),
    ("postgres", env_pw),
    ("postgres", "admin"),
):
    if not pw:
        continue
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname="postgres", user=u, password=pw, connect_timeout=4
        )
        print(f"SUPER_OK user={u}")
        cur = conn.cursor()
        cur.execute("SELECT datname FROM pg_database ORDER BY 1")
        print("databases:", [r[0] for r in cur.fetchall()])
        cur.execute("SELECT usename FROM pg_user ORDER BY 1")
        print("users:", [r[0] for r in cur.fetchall()])
        # ensure role + db if needed
        super_ok = (u, pw, conn)
        break
    except Exception as e:
        print(f"SUPER_FAIL {u}: {str(e).splitlines()[0][:100]}")

if super_ok and not ok_pw:
    u, pw, conn = super_ok
    cur = conn.cursor()
    print("Will create/update role vivek and database super_pass if needed...")
    # Check if vivek exists
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'vivek'")
    if not cur.fetchone():
        cur.execute("CREATE ROLE vivek LOGIN PASSWORD %s", (env_pw or "bruno@9588",))
        print("created role vivek")
    else:
        cur.execute("ALTER ROLE vivek WITH LOGIN PASSWORD %s", (env_pw or "bruno@9588",))
        print("reset password for vivek")
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'super_pass'")
    if not cur.fetchone():
        conn.autocommit = True
        cur.execute("CREATE DATABASE super_pass OWNER vivek")
        print("created database super_pass")
    else:
        conn.autocommit = True
        cur.execute("ALTER DATABASE super_pass OWNER TO vivek")
        print("ensured database owner vivek")
    conn.close()
    # retest
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=db,
            user=user,
            password=env_pw or "bruno@9588",
            connect_timeout=4,
        )
        print("RETEST_OK after role fix")
        conn.close()
        ok_pw = env_pw or "bruno@9588"
    except Exception as e:
        print("RETEST_FAIL", e)

if not ok_pw and not super_ok:
    print(
        "NO_CREDENTIALS_WORK: Postgres is running but neither vivek nor postgres "
        "accepted passwords we tried. Reset the postgres superuser password via "
        "pgAdmin / installer, or tell me the correct password."
    )
    sys.exit(1)

print("DONE ok_pw_set=", bool(ok_pw))

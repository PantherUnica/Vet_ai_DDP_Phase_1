import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)
import os
from kb_ner_db import get_pg_conn

clinic = int(os.getenv("CLINIC_ID", "8"))
conn = get_pg_conn(reuse=False)
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM soap.inventory WHERE location_id=%s", (clinic,))
print("CLINIC_ID", clinic, "inventory_at_location", cur.fetchone()[0])
cur.execute(
    """
    SELECT stock_id, item_name, trade_name
    FROM soap.inventory
    WHERE location_id=%s
      AND (item_name ILIKE %s OR trade_name ILIKE %s)
    LIMIT 5
    """,
    (clinic, "%bravecto%", "%bravecto%"),
)
print("bravecto_matches", cur.fetchall())
cur.execute("SELECT COUNT(*) FROM soap.service_master")
print("services", cur.fetchone()[0])
print("DB_READY")

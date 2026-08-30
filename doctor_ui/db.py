"""SQLite persistence for doctor consultations."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def get_data_dir() -> Path:
    """Persistent data directory (override with VETAI_DATA_DIR for deploy volumes)."""
    raw = (os.getenv("VETAI_DATA_DIR") or "").strip()
    path = Path(raw) if raw else _DEFAULT_DATA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_db_path() -> Path:
    return get_data_dir() / "consultations.db"


# Back-compat for imports that read DB_PATH at module load
DB_PATH = get_db_path()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consultations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    doctor_name TEXT,
    pet_name TEXT,
    consultation_language TEXT DEFAULT 'multi',
    input_mode TEXT NOT NULL,
    step1_raw_text TEXT,
    audio_path TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    output_dir TEXT,
    soap_json TEXT,
    error_message TEXT
);
"""


def _connect() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def create_consultation(
    *,
    doctor_name: str = "",
    pet_name: str = "",
    consultation_language: str = "multi",
    input_mode: str = "typed",
    step1_raw_text: str = "",
    audio_path: Optional[str] = None,
    status: str = "draft",
    output_dir: Optional[str] = None,
) -> int:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO consultations
            (created_at, doctor_name, pet_name, consultation_language, input_mode,
             step1_raw_text, audio_path, status, output_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                doctor_name,
                pet_name,
                consultation_language,
                input_mode,
                step1_raw_text,
                audio_path,
                status,
                output_dir,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_consultation(consultation_id: int, **fields: Any) -> None:
    if not fields:
        return
    init_db()
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [consultation_id]
    with _connect() as conn:
        conn.execute(f"UPDATE consultations SET {cols} WHERE id = ?", vals)
        conn.commit()


def get_consultation(consultation_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM consultations WHERE id = ?", (consultation_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("soap_json"):
        try:
            d["soap_json_parsed"] = json.loads(d["soap_json"])
        except json.JSONDecodeError:
            d["soap_json_parsed"] = None
    return d


def list_consultations(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM consultations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def save_soap_result(
    consultation_id: int,
    soap_json: Dict[str, Any],
    output_dir: str,
    status: str = "complete",
) -> None:
    update_consultation(
        consultation_id,
        soap_json=json.dumps(soap_json, ensure_ascii=False),
        output_dir=output_dir,
        status=status,
    )

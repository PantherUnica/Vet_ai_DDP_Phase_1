"""Async pipeline wrapper for the doctor UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doctor_ui import db  # noqa: E402
from doctor_ui.languages import stored_to_deepgram  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "runs"
logger = logging.getLogger("doctor_ui")

_SOAP_PRIMARY_KEYS = (
    "Subjective",
    "Objective",
    "Assessment",
    "Plan",
    "Conclusion",
    "KeyIssues",
)


def _postgres_reachable(host: str = "127.0.0.1", port: int = 5432, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _prepare_doctor_ui_pipeline_env() -> None:
    """Skip inventory grounding / Phase 2 when local Postgres is down (common for UI testing)."""
    host = os.getenv("PGHOST", "127.0.0.1")
    try:
        port = int(os.getenv("PGPORT", "5432"))
    except ValueError:
        port = 5432
    if _postgres_reachable(host, port):
        return
    os.environ["SKIP_BILLING_PIPELINE"] = "true"
    os.environ["SKIP_PHASE2"] = "true"
    logger.warning(
        "Postgres not reachable at %s:%s — skipping inventory grounding and Phase 2 "
        "(SOAP generation still runs).",
        host,
        port,
    )


def _soap_looks_failed(soap_note: Any, soap_json: Dict[str, Any]) -> Optional[str]:
    """Return an error message if the pipeline did not produce a real SOAP note."""
    if isinstance(soap_note, str):
        text = soap_note.strip()
        if text.lower().startswith("error generating soap"):
            return text
        if text.lower().startswith("error:"):
            return text
    raw = soap_json.get("raw")
    if isinstance(raw, str) and raw.strip():
        low = raw.strip().lower()
        if low.startswith("error generating soap") or low.startswith("error:"):
            return raw.strip()
    if not soap_json:
        return "Pipeline returned an empty SOAP note."
    if "raw" in soap_json and len(soap_json) == 1:
        return f"SOAP note was not structured JSON: {str(soap_json.get('raw'))[:300]}"
    if not any(soap_json.get(k) for k in _SOAP_PRIMARY_KEYS):
        return "SOAP note missing Subjective/Objective/Assessment/Plan fields."
    return None


def _extract_soap_json(result: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    soap_note = result.get("soap_note")
    if isinstance(soap_note, dict):
        return soap_note
    if isinstance(soap_note, str):
        soap_note = soap_note.strip()
        if soap_note.startswith("{"):
            try:
                return json.loads(soap_note)
            except json.JSONDecodeError:
                pass
    # Try latest soap_note_*.json on disk
    json_files = sorted(output_dir.glob("soap_note_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if json_files:
        try:
            return json.loads(json_files[0].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"raw": soap_note} if soap_note else {}


async def transcribe_audio_file(
    audio_path: str,
    language: str = "multi",
) -> str:
    from asr_providers import transcribe

    result = transcribe(
        audio_path,
        language=stored_to_deepgram(language),
        logger=logging.getLogger("doctor_ui"),
    )
    return result.text


async def run_pipeline_for_consultation(
    consultation_id: int,
    step1_text: str,
    *,
    source: str = "typed",
    audio_path: Optional[str] = None,
    consultation_language: str = "multi",
) -> Dict[str, Any]:
    _prepare_doctor_ui_pipeline_env()
    from SOAP_notes_phase1_experiment import generate_soap_note_from_transcript_async

    output_dir = RUNS_DIR / str(consultation_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    db.update_consultation(
        consultation_id,
        step1_raw_text=step1_text,
        input_mode=source,
        consultation_language=consultation_language,
        status="processing",
        output_dir=str(output_dir),
    )

    try:
        result = await generate_soap_note_from_transcript_async(
            step1_text,
            str(output_dir),
            source=source,
            audio_path=audio_path,
            consultation_language=consultation_language,
        )
        soap_json = _extract_soap_json(result, output_dir)
        # Prefer formatter output if present in soap_json
        try:
            from soap_section_formatter import format_soap_dict

            if soap_json:
                soap_json = format_soap_dict(soap_json)
        except Exception:
            pass
        fail_msg = _soap_looks_failed(result.get("soap_note"), soap_json)
        if fail_msg:
            db.update_consultation(
                consultation_id,
                status="error",
                error_message=fail_msg,
                soap_json=json.dumps(soap_json, ensure_ascii=False),
                output_dir=str(output_dir),
            )
            raise RuntimeError(fail_msg)
        db.save_soap_result(consultation_id, soap_json, str(output_dir), status="complete")
        return {
            "result": result,
            "soap_json": soap_json,
            "output_dir": str(output_dir),
            "pipeline_flags": result.get("pipeline_flags") or {},
        }
    except Exception as exc:
        # Avoid overwriting a more specific error already saved above
        rec = db.get_consultation(consultation_id)
        if not rec or rec.get("status") != "error":
            db.update_consultation(
                consultation_id,
                status="error",
                error_message=str(exc),
            )
        raise


def run_pipeline_sync(
    consultation_id: int,
    step1_text: str,
    **kwargs,
) -> Dict[str, Any]:
    return asyncio.run(run_pipeline_for_consultation(consultation_id, step1_text, **kwargs))

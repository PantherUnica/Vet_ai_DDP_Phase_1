"""Async pipeline wrapper for the doctor UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doctor_ui import db  # noqa: E402
from doctor_ui.languages import stored_to_deepgram  # noqa: E402
from doctor_ui.pipeline_logging import init_pipeline_logging, log_pipeline_banner  # noqa: E402


def get_runs_dir() -> Path:
    """Persistent runs directory (override with VETAI_RUNS_DIR for deploy volumes)."""
    raw = (os.getenv("VETAI_RUNS_DIR") or "").strip()
    path = Path(raw) if raw else Path(__file__).resolve().parent / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


RUNS_DIR = get_runs_dir()
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


def _apply_latency_env_defaults() -> None:
    """Quality-safe parallel defaults for Doctor UI (no model downgrades)."""
    os.environ.setdefault("PHASE2_BLOCKING", "false")
    os.environ.setdefault("PHASE2_START_WITH_SOAP_READY", "true")
    os.environ.setdefault("TARGET_60S", "true")
    os.environ.setdefault("CHUNK_PARALLEL_ENABLED", "true")
    os.environ.setdefault("FAST_TRANSCRIPTION", "true")
    os.environ.setdefault("ASR_PREP_WAV", "true")


def _prepare_doctor_ui_pipeline_env() -> None:
    """
    Toggle inventory grounding / Phase 2 from Postgres reachability.
    IMPORTANT: clear SKIP_* when DB is back — sticky true permanently skips Master Doc RAG.
    """
    _apply_latency_env_defaults()
    host = os.getenv("PGHOST", "127.0.0.1")
    try:
        port = int(os.getenv("PGPORT", "5432"))
    except ValueError:
        port = 5432
    if _postgres_reachable(host, port):
        if os.getenv("SKIP_BILLING_PIPELINE", "").lower() in ("1", "true", "yes"):
            os.environ.pop("SKIP_BILLING_PIPELINE", None)
            logger.info("Postgres reachable — cleared SKIP_BILLING_PIPELINE (grounding enabled)")
        if os.getenv("SKIP_PHASE2", "").lower() in ("1", "true", "yes"):
            os.environ.pop("SKIP_PHASE2", None)
            logger.info("Postgres reachable — cleared SKIP_PHASE2")
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
    json_files = sorted(output_dir.glob("soap_note_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if json_files:
        try:
            return json.loads(json_files[0].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"raw": soap_note} if soap_note else {}


def _merge_timing_with_asr_ui(
    pipeline_timing: Dict[str, Any],
    step1_asr_ui_ms: Optional[int],
) -> Dict[str, Any]:
    if not pipeline_timing or not step1_asr_ui_ms:
        return pipeline_timing or {}
    out = dict(pipeline_timing)
    out["step1_asr_ui_ms"] = step1_asr_ui_ms
    pipeline_ms = out.get("total_ms") or 0
    out["user_perceived_total_ms"] = pipeline_ms + step1_asr_ui_ms
    meta = dict(out.get("metadata") or {})
    meta["step1_asr_ui_ms"] = step1_asr_ui_ms
    out["metadata"] = meta
    return out


async def transcribe_audio_file(
    audio_path: str,
    language: str = "multi",
) -> Tuple[str, Dict[str, Any]]:
    """Transcribe audio; return (text, metadata dict with latency_ms and asr_profile)."""
    from asr_providers import transcribe

    result = transcribe(
        audio_path,
        language=stored_to_deepgram(language),
        logger=logging.getLogger("doctor_ui"),
    )
    meta = {
        "latency_ms": result.latency_ms,
        "provider": result.provider,
        "model": result.model,
        **(result.metadata or {}),
    }
    return result.text, meta


async def run_pipeline_for_consultation(
    consultation_id: int,
    step1_text: str,
    *,
    source: str = "typed",
    audio_path: Optional[str] = None,
    consultation_language: str = "multi",
    step1_asr_ui_ms: Optional[int] = None,
) -> Dict[str, Any]:
    _prepare_doctor_ui_pipeline_env()
    from SOAP_notes_phase1_experiment import generate_soap_note_from_transcript_async

    output_dir = get_runs_dir() / str(consultation_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_pipeline_logging(str(output_dir))
    log_pipeline_banner(
        consultation_id=consultation_id,
        source=source,
        output_dir=str(output_dir),
        step="START",
    )

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
            step1_asr_ui_ms=step1_asr_ui_ms,
        )
        soap_json = _extract_soap_json(result, output_dir)
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
        timing = _merge_timing_with_asr_ui(
            result.get("pipeline_timing") or {},
            step1_asr_ui_ms,
        )
        log_pipeline_banner(
            consultation_id=consultation_id,
            source=source,
            output_dir=str(output_dir),
            step="COMPLETE",
        )
        logger.info(
            "Pipeline timing total_ms=%s user_perceived_total_ms=%s",
            timing.get("total_ms"),
            timing.get("user_perceived_total_ms"),
        )
        return {
            "result": result,
            "soap_json": soap_json,
            "output_dir": str(output_dir),
            "pipeline_flags": result.get("pipeline_flags") or {},
            "pipeline_timing": timing,
        }
    except Exception as exc:
        logger.exception("PIPELINE FAILED | consultation=%s | %s", consultation_id, exc)
        rec = db.get_consultation(consultation_id)
        if not rec or rec.get("status") != "error":
            db.update_consultation(
                consultation_id,
                status="error",
                error_message=str(exc),
            )
        raise


async def run_pipeline_from_audio(
    consultation_id: int,
    audio_path: str,
    *,
    consultation_language: str = "multi",
    doctor_name: str = "",
    pet_name: str = "",
) -> Dict[str, Any]:
    """
    Single-click voice path: ASR inside the pipeline (STEP1 in timing JSON).
    Brain NER and all downstream stages unchanged.
    """
    _prepare_doctor_ui_pipeline_env()
    from SOAP_notes_phase1_experiment import generate_soap_note_from_audio_async

    output_dir = get_runs_dir() / str(consultation_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_pipeline_logging(str(output_dir))
    log_pipeline_banner(
        consultation_id=consultation_id,
        source="voice",
        output_dir=str(output_dir),
        step="START",
    )

    db.update_consultation(
        consultation_id,
        doctor_name=doctor_name,
        pet_name=pet_name,
        consultation_language=consultation_language,
        input_mode="voice",
        audio_path=audio_path,
        status="processing",
        output_dir=str(output_dir),
    )

    try:
        result = await generate_soap_note_from_audio_async(
            audio_path,
            str(output_dir),
            source="voice",
            consultation_language=consultation_language,
            asr_language=stored_to_deepgram(consultation_language),
        )
        soap_json = _extract_soap_json(result, output_dir)
        try:
            from soap_section_formatter import format_soap_dict

            if soap_json:
                soap_json = format_soap_dict(soap_json)
        except Exception:
            pass
        transcript = (result.get("cleaned_text") or result.get("raw") or "")
        if not transcript and output_dir.joinpath("step1_raw_transcription.txt").is_file():
            transcript = output_dir.joinpath("step1_raw_transcription.txt").read_text(encoding="utf-8")
        fail_msg = _soap_looks_failed(result.get("soap_note"), soap_json)
        if fail_msg:
            db.update_consultation(
                consultation_id,
                status="error",
                error_message=fail_msg,
                step1_raw_text=transcript[:50000] if transcript else None,
                soap_json=json.dumps(soap_json, ensure_ascii=False),
                output_dir=str(output_dir),
            )
            raise RuntimeError(fail_msg)
        db.update_consultation(
            consultation_id,
            step1_raw_text=transcript[:50000] if transcript else None,
        )
        db.save_soap_result(consultation_id, soap_json, str(output_dir), status="complete")
        timing = result.get("pipeline_timing") or {}
        log_pipeline_banner(
            consultation_id=consultation_id,
            source="voice",
            output_dir=str(output_dir),
            step="COMPLETE",
        )
        logger.info("Pipeline timing total_ms=%s", timing.get("total_ms"))
        return {
            "result": result,
            "soap_json": soap_json,
            "output_dir": str(output_dir),
            "pipeline_flags": result.get("pipeline_flags") or {},
            "pipeline_timing": result.get("pipeline_timing") or {},
        }
    except Exception as exc:
        logger.exception("PIPELINE FAILED | consultation=%s | %s", consultation_id, exc)
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

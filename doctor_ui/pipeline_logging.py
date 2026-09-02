"""Console (and optional file) logging for Doctor UI / Streamlit pipeline runs."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Loggers that participate in the end-to-end consultation pipeline.
PIPELINE_LOGGER_NAMES = (
    "soap_generator",
    "doctor_ui",
    "asr_providers",
    "kb_phase2_integration",
    "kb_ner_super_pass",
    "kb_anchor_span",
)

# Keep third-party HTTP noise out of DEBUG console output.
_QUIET_LOGGER_NAMES = (
    "httpx",
    "httpcore",
    "urllib3",
    "openai",
    "anthropic",
)

_CONSOLE_HANDLER_ATTR = "_vetai_pipeline_console_handler"
_CONFIGURED = False


def _log_level_from_env() -> int:
    raw = (os.getenv("VETAI_PIPELINE_LOG_LEVEL") or "DEBUG").strip().upper()
    return getattr(logging, raw, logging.DEBUG)


def _attach_console_handler(logger: logging.Logger, level: int, formatter: logging.Formatter) -> None:
    for handler in logger.handlers:
        if getattr(handler, _CONSOLE_HANDLER_ATTR, False):
            handler.setLevel(level)
            return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    setattr(handler, _CONSOLE_HANDLER_ATTR, True)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def configure_pipeline_console_logging(level: Optional[int] = None) -> None:
    """
    Idempotent: attach DEBUG (or env) console handlers for pipeline loggers.
    Safe to call from Streamlit app startup and before each pipeline run.
    """
    global _CONFIGURED
    level = _log_level_from_env() if level is None else level
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in _QUIET_LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in PIPELINE_LOGGER_NAMES:
        _attach_console_handler(logging.getLogger(name), level, formatter)
    _CONFIGURED = True


def init_pipeline_logging(output_dir: str, *, level: Optional[int] = None) -> Path:
    """
    Console logging for all pipeline modules + per-run file log under output_dir.
    Returns path to the run log file (soap_generator).
    """
    level = _log_level_from_env() if level is None else level
    configure_pipeline_console_logging(level=level)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from SOAP_notes_phase1_experiment import setup_logging

        setup_logging(str(out), console_level=level, file_level=logging.DEBUG)
    except Exception as exc:
        log = logging.getLogger("doctor_ui")
        log.warning("Could not initialize file logging via setup_logging: %s", exc)
        return out / "pipeline.log"

    # setup_logging logs the file path on the soap_generator logger
    log = logging.getLogger("doctor_ui")
    log.debug("Pipeline file logging directory: %s", out)
    return out


def log_pipeline_banner(
    *,
    consultation_id: int,
    source: str,
    output_dir: str,
    step: str = "START",
) -> None:
    log = logging.getLogger("doctor_ui")
    log.info("=" * 72)
    log.info(
        "PIPELINE %s | consultation=%s | source=%s | output=%s",
        step,
        consultation_id,
        source,
        output_dir,
    )
    log.info("=" * 72)

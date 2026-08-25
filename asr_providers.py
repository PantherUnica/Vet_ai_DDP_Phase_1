"""
Pluggable ASR providers for Step 1 transcription.

Configure via environment (or pass explicitly):
  ASR_PROVIDER=deepgram|fireworks
  ASR_MODEL=<provider-specific model id>
  ASR_LANGUAGE=<optional, e.g. en, hi, multi — Deepgram defaults to multi>
"""

from __future__ import annotations

import importlib
import logging
import mimetypes
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

DEFAULT_MODELS: Dict[str, str] = {
    "deepgram": "nova-3",
    "fireworks": "whisper-v3-turbo",
}

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"


@dataclass
class TranscriptionResult:
    text: str
    provider: str
    model: str
    latency_ms: int
    audio_path: str = ""
    segments: Optional[list] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text or "")

    @property
    def word_count(self) -> int:
        return len((self.text or "").split())


def resolve_provider(provider: Optional[str] = None) -> str:
    return (provider or os.getenv("ASR_PROVIDER") or "deepgram").strip().lower()


def resolve_model(provider: str, model: Optional[str] = None) -> str:
    explicit = (model or os.getenv("ASR_MODEL") or "").strip()
    if explicit:
        return explicit
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["deepgram"])


def resolve_asr_language(provider: Optional[str] = None, language: Optional[str] = None) -> Optional[str]:
    """Resolve ASR language. Deepgram defaults to 'multi' for code-switching audio."""
    explicit = (language or os.getenv("ASR_LANGUAGE") or "").strip()
    if explicit:
        return explicit
    p = resolve_provider(provider)
    if p == "deepgram":
        return "multi"
    return None


def asr_run_label(provider: Optional[str] = None, model: Optional[str] = None) -> str:
    p = resolve_provider(provider)
    m = resolve_model(p, model)
    safe_model = m.replace("/", "_").replace(" ", "_")
    return f"{p}_{safe_model}"


def _read_key_from_file(path: str, prefixes: tuple, valid_prefix: Optional[str] = None) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for prefix in prefixes:
                    if line.startswith(prefix):
                        key = line.split("=", 1)[1].strip().strip("\"'")
                        if valid_prefix is None or key.startswith(valid_prefix):
                            return key
                if valid_prefix and line.startswith(valid_prefix):
                    return line
    except OSError:
        pass
    return None


def _resolve_key_file(name: str) -> str:
    here = Path(__file__).resolve().parent
    for base in (here, here.parent):
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return str(here / name)


def load_deepgram_api_key() -> str:
    api_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
    if api_key:
        return api_key

    for filename in ("deepgram_api.txt", "API_Key.txt", ".env"):
        api_key = _read_key_from_file(
            _resolve_key_file(filename),
            ("DEEPGRAM_API_KEY=", "deepgram_api=", "deepgram_API="),
        )
        if api_key:
            return api_key

    raise RuntimeError(
        "Deepgram API key not found. Set DEEPGRAM_API_KEY env or add to deepgram_api.txt / API_Key.txt "
        "(format: DEEPGRAM_API_KEY=your_key)."
    )


def _mime_type_for_path(audio_path: str) -> str:
    ext = Path(audio_path).suffix.lower()
    mime_map = {
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    if ext in mime_map:
        return mime_map[ext]
    guessed, _ = mimetypes.guess_type(audio_path)
    return guessed or "audio/wav"


def transcribe_deepgram(
    audio_path: str,
    *,
    model: str,
    language: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> TranscriptionResult:
    log = logger or logging.getLogger("asr_providers")
    api_key = load_deepgram_api_key()
    lang = resolve_asr_language("deepgram", language)

    params: Dict[str, Any] = {
        "model": model,
        "smart_format": "false",
        "punctuate": "true",
    }
    if lang:
        params["language"] = lang

    file_size = os.path.getsize(audio_path)
    timeout = max(60, file_size // (1024 * 1024) * 10)
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": _mime_type_for_path(audio_path),
    }

    # Transient DNS / connection blips are common on flaky networks — retry a few times.
    max_attempts = int(os.getenv("ASR_HTTP_RETRIES", "4"))
    backoff_sec = (2, 4, 8, 12)
    started = time.perf_counter()
    last_err: Optional[Exception] = None
    response = None

    for attempt in range(max_attempts):
        try:
            with open(audio_path, "rb") as audio_file:
                response = requests.post(
                    DEEPGRAM_LISTEN_URL,
                    headers=headers,
                    params=params,
                    data=audio_file,
                    timeout=timeout,
                )
            last_err = None
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_err = exc
            if attempt >= max_attempts - 1:
                break
            wait = backoff_sec[attempt] if attempt < len(backoff_sec) else 12
            log.warning(
                "Deepgram connection failed (attempt %d/%d): %s; retrying in %ds",
                attempt + 1,
                max_attempts,
                exc,
                wait,
            )
            time.sleep(wait)

    latency_ms = int((time.perf_counter() - started) * 1000)
    if last_err is not None or response is None:
        raise RuntimeError(
            f"Deepgram connection failed after {max_attempts} attempts "
            f"(DNS/network). Check internet / VPN / DNS. Last error: {last_err}"
        ) from last_err

    if response.status_code != 200:
        body = (response.text or "")[:500]
        raise RuntimeError(f"Deepgram transcription failed ({response.status_code}): {body}")

    payload = response.json()
    try:
        text = (
            payload["results"]["channels"][0]["alternatives"][0]["transcript"] or ""
        ).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Deepgram returned unexpected response shape: {exc}") from exc

    if not text:
        raise RuntimeError("Deepgram returned empty transcription.")

    log.info("Deepgram ASR complete: model=%s latency_ms=%d chars=%d", model, latency_ms, len(text))
    return TranscriptionResult(
        text=text,
        provider="deepgram",
        model=model,
        latency_ms=latency_ms,
        audio_path=audio_path,
        metadata={"language": lang, "response_keys": list(payload.keys())},
    )


def transcribe_fireworks(
    audio_path: str,
    *,
    model: str,
    logger: Optional[logging.Logger] = None,
) -> TranscriptionResult:
    """Delegate to existing Fireworks transcription (lazy import avoids circular deps)."""
    log = logger or logging.getLogger("asr_providers")
    started = time.perf_counter()
    mod = importlib.import_module("SOAP_notes_phase1_experiment")
    text = mod.transcribe_audio(audio_path, model=model)
    latency_ms = int((time.perf_counter() - started) * 1000)
    if not text:
        raise RuntimeError("Fireworks returned empty transcription.")
    log.info("Fireworks ASR complete: model=%s latency_ms=%d chars=%d", model, latency_ms, len(text))
    return TranscriptionResult(
        text=text,
        provider="fireworks",
        model=model,
        latency_ms=latency_ms,
        audio_path=audio_path,
    )


PROVIDERS: Dict[str, Callable[..., TranscriptionResult]] = {
    "deepgram": transcribe_deepgram,
    "fireworks": transcribe_fireworks,
}


def transcribe(
    audio_path: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    language: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> TranscriptionResult:
    p = resolve_provider(provider)
    m = resolve_model(p, model)
    fn = PROVIDERS.get(p)
    if fn is None:
        supported = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown ASR provider '{p}'. Supported: {supported}")

    log = logger or logging.getLogger("asr_providers")
    log.info("ASR transcribe: provider=%s model=%s file=%s", p, m, audio_path)

    if p == "deepgram":
        return fn(audio_path, model=m, language=language, logger=log)
    return fn(audio_path, model=m, logger=log)

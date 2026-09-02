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
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

DEFAULT_MODELS: Dict[str, str] = {
    "deepgram": "nova-3",
    "fireworks": "whisper-v3-turbo",
}

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"
ASR_TARGET_SAMPLE_RATE = int(os.getenv("ASR_TARGET_SAMPLE_RATE", "16000"))


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


def profile_audio_file(audio_path: str) -> Dict[str, Any]:
    """Collect duration, size, and format for ASR latency diagnosis."""
    path = Path(audio_path)
    profile: Dict[str, Any] = {
        "audio_path": str(path),
        "file_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.is_file() else None,
        "format": path.suffix.lower().lstrip(".") or "unknown",
        "duration_sec": None,
        "sample_rate_hz": None,
        "channels": None,
    }
    if not path.is_file():
        return profile
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            profile["sample_rate_hz"] = rate
            profile["channels"] = wf.getnchannels()
            if rate:
                profile["duration_sec"] = round(frames / float(rate), 2)
    except Exception:
        try:
            import soundfile as sf  # type: ignore

            info = sf.info(str(path))
            profile["duration_sec"] = round(float(info.duration), 2)
            profile["sample_rate_hz"] = info.samplerate
            profile["channels"] = info.channels
        except Exception:
            pass
    return profile


def _enrich_asr_metadata(
    audio_path: str,
    base: Optional[Dict[str, Any]] = None,
    *,
    http_attempts: int = 1,
) -> Dict[str, Any]:
    meta = dict(base or {})
    meta["asr_profile"] = profile_audio_file(audio_path)
    meta["http_attempts"] = http_attempts
    return meta


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


def _is_optimal_wav(audio_path: str, sample_rate: int = ASR_TARGET_SAMPLE_RATE) -> bool:
    try:
        import wave

        with wave.open(audio_path, "rb") as wf:
            return wf.getframerate() == sample_rate and wf.getnchannels() == 1
    except Exception:
        return False


def _ffmpeg_convert_to_wav(input_path: str, output_path: str, sample_rate: int = ASR_TARGET_SAMPLE_RATE) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav",
        "-loglevel", "error", "-threads", "0",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _resample_wav_stdlib(input_path: str, output_path: str, target_rate: int = ASR_TARGET_SAMPLE_RATE) -> None:
    """16-bit PCM WAV resample/downmix without ffmpeg (48kHz mono -> 16kHz mono, etc.)."""
    import struct
    import wave

    with wave.open(input_path, "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError(f"unsupported sample width: {sampwidth}")
        raw = wf.readframes(wf.getnframes())

    samples = list(struct.unpack(f"<{len(raw) // 2}h", raw))
    if channels > 1:
        mono: List[int] = []
        for i in range(0, len(samples), channels):
            chunk = samples[i : i + channels]
            mono.append(int(sum(chunk) / len(chunk)))
        samples = mono

    if rate == target_rate:
        out_samples = samples
    else:
        if rate % target_rate != 0:
            raise ValueError(f"cannot resample {rate}Hz to {target_rate}Hz without ffmpeg")
        step = rate // target_rate
        out_samples = [
            int(sum(samples[i : i + step]) / step)
            for i in range(0, len(samples) - step + 1, step)
        ]

    with wave.open(output_path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(target_rate)
        out.writeframes(struct.pack(f"<{len(out_samples)}h", *out_samples))


def prepare_audio_for_asr(
    audio_path: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Convert to 16 kHz mono WAV when beneficial (smaller upload for Deepgram).
    Returns (path_to_transcribe, prep_metadata). Caller must delete temp_path if set.
    """
    log = logger or logging.getLogger("asr_providers")
    meta: Dict[str, Any] = {"prepared": False, "original_path": audio_path}
    if not _env_bool("ASR_PREP_WAV", True):
        meta["skipped"] = "ASR_PREP_WAV=false"
        return audio_path, meta

    path = Path(audio_path)
    if not path.is_file():
        return audio_path, meta

    orig_profile = profile_audio_file(audio_path)
    meta["original_profile"] = orig_profile

    if path.suffix.lower() == ".wav" and _is_optimal_wav(audio_path):
        meta["skipped"] = "already_16khz_mono_wav"
        return audio_path, meta

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="vetai_asr_prep_")
    tmp.close()
    try:
        try:
            _ffmpeg_convert_to_wav(audio_path, tmp.name)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            if path.suffix.lower() == ".wav":
                try:
                    _resample_wav_stdlib(audio_path, tmp.name)
                except ValueError:
                    meta["skipped"] = "wav_resample_not_supported"
                    return audio_path, meta
            else:
                raise
        conv_profile = profile_audio_file(tmp.name)
        meta.update({
            "prepared": True,
            "temp_path": tmp.name,
            "converted_profile": conv_profile,
            "size_reduction_mb": round(
                (orig_profile.get("file_mb") or 0) - (conv_profile.get("file_mb") or 0), 3
            ),
        })
        log.info(
            "ASR prep: %s (%.2f MB) -> 16kHz mono WAV (%.2f MB)",
            path.name,
            orig_profile.get("file_mb") or 0,
            conv_profile.get("file_mb") or 0,
        )
        return tmp.name, meta
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        log.warning("ASR WAV prep failed (%s); using original file", exc)
        meta["prep_error"] = str(exc)
        return audio_path, meta


def _split_audio_chunks(
    audio_path: str,
    *,
    chunk_seconds: int,
    overlap_seconds: float,
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[float, float, str]]:
    """Split audio into time-based chunks. Uses ffmpeg, or stdlib wave for 16-bit PCM WAV."""
    log = logger or logging.getLogger("asr_providers")
    duration = profile_audio_file(audio_path).get("duration_sec")
    if not duration or duration <= chunk_seconds:
        return [(0.0, float(duration or 0), audio_path)]

    try:
        return _split_audio_chunks_ffmpeg(
            audio_path, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds,
            duration=float(duration), logger=log,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        if Path(audio_path).suffix.lower() == ".wav":
            return _split_audio_chunks_wav_stdlib(
                audio_path, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds,
                duration=float(duration), logger=log,
            )
        raise


def _split_audio_chunks_ffmpeg(
    audio_path: str,
    *,
    chunk_seconds: int,
    overlap_seconds: float,
    duration: float,
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[float, float, str]]:
    log = logger or logging.getLogger("asr_providers")
    chunks: List[Tuple[float, float, str]] = []
    tmp_dir = tempfile.mkdtemp(prefix="vetai_asr_chunks_")
    start = 0.0
    idx = 0
    step = max(1.0, chunk_seconds - overlap_seconds)

    while start < duration:
        end = min(start + chunk_seconds, duration)
        chunk_path = os.path.join(tmp_dir, f"chunk_{idx:03d}.wav")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", audio_path,
            "-t", str(end - start),
            "-ar", str(ASR_TARGET_SAMPLE_RATE),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            "-loglevel", "error",
            chunk_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        chunks.append((start, end, chunk_path))
        if end >= duration:
            break
        start += step
        idx += 1

    if log:
        log.info("ASR parallel (ffmpeg): split %.1fs audio into %d chunk(s)", duration, len(chunks))
    return chunks


def _split_audio_chunks_wav_stdlib(
    audio_path: str,
    *,
    chunk_seconds: int,
    overlap_seconds: float,
    duration: float,
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[float, float, str]]:
    """Split 16-bit PCM WAV by sample offsets (no ffmpeg)."""
    import struct
    import wave

    log = logger or logging.getLogger("asr_providers")
    with wave.open(audio_path, "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if sampwidth != 2:
            raise ValueError("stdlib split requires 16-bit PCM WAV")
        raw = wf.readframes(wf.getnframes())

    samples = list(struct.unpack(f"<{len(raw) // 2}h", raw))
    if channels > 1:
        samples = [
            int(sum(samples[i : i + channels]) / channels)
            for i in range(0, len(samples), channels)
        ]

    tmp_dir = tempfile.mkdtemp(prefix="vetai_asr_chunks_")
    chunks: List[Tuple[float, float, str]] = []
    start_sec = 0.0
    idx = 0
    step_sec = max(1.0, chunk_seconds - overlap_seconds)

    while start_sec < duration:
        end_sec = min(start_sec + chunk_seconds, duration)
        s0 = int(start_sec * rate)
        s1 = int(end_sec * rate)
        chunk_samples = samples[s0:s1]
        chunk_path = os.path.join(tmp_dir, f"chunk_{idx:03d}.wav")
        with wave.open(chunk_path, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(struct.pack(f"<{len(chunk_samples)}h", *chunk_samples))
        chunks.append((start_sec, end_sec, chunk_path))
        if end_sec >= duration:
            break
        start_sec += step_sec
        idx += 1

    if log:
        log.info("ASR parallel (stdlib): split %.1fs audio into %d chunk(s)", duration, len(chunks))
    return chunks


def _normalize_merge_token(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _merge_chunk_transcripts(parts: List[str], *, overlap_seconds: float) -> str:
    """Stitch chunk transcripts; trim duplicated prefix from overlap regions."""
    if not parts:
        return ""
    merged = (parts[0] or "").strip()
    if not merged:
        merged = ""
    # Overlap trim: drop leading words from next chunk if they repeat tail of merged text
    overlap_words = max(3, int(overlap_seconds * 2.5))
    for part in parts[1:]:
        nxt = (part or "").strip()
        if not nxt:
            continue
        if not merged:
            merged = nxt
            continue
        tail_tokens = _normalize_merge_token(merged).split()[-overlap_words:]
        head_tokens = _normalize_merge_token(nxt).split()[:overlap_words]
        trim = 0
        for k in range(min(len(tail_tokens), len(head_tokens)), 0, -1):
            if tail_tokens[-k:] == head_tokens[:k]:
                trim = k
                break
        if trim:
            nxt_words = nxt.split()
            nxt = " ".join(nxt_words[trim:]) if trim < len(nxt_words) else ""
        if nxt:
            merged = f"{merged} {nxt}".strip()
    return merged.strip()


def transcribe_deepgram_parallel(
    audio_path: str,
    *,
    model: str,
    language: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> TranscriptionResult:
    """Parallel batch ASR: split long audio, transcribe chunks concurrently, merge."""
    log = logger or logging.getLogger("asr_providers")
    chunk_minutes = float(os.getenv("ASR_CHUNK_MINUTES", "3"))
    overlap_sec = float(os.getenv("ASR_CHUNK_OVERLAP_SEC", "2"))
    max_workers = int(os.getenv("ASR_PARALLEL_MAX_WORKERS", "4"))
    chunk_seconds = max(60, int(chunk_minutes * 60))

    started = time.perf_counter()
    chunk_specs = _split_audio_chunks(
        audio_path,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_sec,
        logger=log,
    )
    chunk_dir = os.path.dirname(chunk_specs[0][2]) if chunk_specs else None
    texts: List[Optional[str]] = [None] * len(chunk_specs)
    total_http_attempts = 0

    def _transcribe_idx(idx: int, spec: Tuple[float, float, str]) -> Tuple[int, TranscriptionResult]:
        _start, _end, cpath = spec
        tr = transcribe_deepgram(cpath, model=model, language=language, logger=log)
        return idx, tr

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_transcribe_idx, i, spec) for i, spec in enumerate(chunk_specs)]
        for fut in as_completed(futures):
            idx, tr = fut.result()
            texts[idx] = tr.text
            total_http_attempts = max(total_http_attempts, (tr.metadata or {}).get("http_attempts") or 1)

    # Cleanup chunk temp dir
    if chunk_dir and chunk_dir != os.path.dirname(os.path.abspath(audio_path)):
        try:
            for _s, _e, cpath in chunk_specs:
                if os.path.isfile(cpath):
                    os.unlink(cpath)
            os.rmdir(chunk_dir)
        except OSError:
            pass

    merged_text = _merge_chunk_transcripts([t or "" for t in texts], overlap_seconds=overlap_sec)
    if not merged_text:
        raise RuntimeError("Parallel ASR produced empty merged transcript.")

    latency_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "Deepgram parallel ASR complete: model=%s chunks=%d latency_ms=%d chars=%d",
        model, len(chunk_specs), latency_ms, len(merged_text),
    )
    return TranscriptionResult(
        text=merged_text,
        provider="deepgram",
        model=model,
        latency_ms=latency_ms,
        audio_path=audio_path,
        metadata=_enrich_asr_metadata(
            audio_path,
            {
                "language": resolve_asr_language("deepgram", language),
                "parallel_chunks": len(chunk_specs),
                "chunk_minutes": chunk_minutes,
                "chunk_overlap_sec": overlap_sec,
            },
            http_attempts=total_http_attempts,
        ),
    )


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
    http_attempts = 0

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
            http_attempts = attempt + 1
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
        metadata=_enrich_asr_metadata(
            audio_path,
            {"language": lang, "response_keys": list(payload.keys())},
            http_attempts=http_attempts or 1,
        ),
    )


def transcribe_fireworks(
    audio_path: str,
    *,
    model: str,
    logger: Optional[logging.Logger] = None,
) -> TranscriptionResult:
    """Delegate to existing Fireworks transcription (lazy import avoids circular deps)."""
    log = logger or logging.getLogger("asr_providers")
    fw_key = (os.getenv("FIREWORKS_API_KEY") or os.getenv("fireworks_API") or "").strip()
    if fw_key and not os.getenv("FIREWORKS_API_KEY"):
        os.environ["FIREWORKS_API_KEY"] = fw_key
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
        metadata=_enrich_asr_metadata(audio_path),
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

    prep_path, prep_meta = prepare_audio_for_asr(audio_path, logger=log)
    use_path = prep_path
    temp_prep = prep_meta.get("temp_path")

    try:
        parallel_min_sec = float(os.getenv("ASR_PARALLEL_MIN_DURATION_SEC", "300"))
        duration_sec = (prep_meta.get("converted_profile") or prep_meta.get("original_profile") or {}).get("duration_sec")
        if duration_sec is None:
            duration_sec = profile_audio_file(use_path).get("duration_sec")

        use_parallel = (
            p == "deepgram"
            and _env_bool("ASR_PARALLEL_CHUNKS", False)
            and duration_sec is not None
            and float(duration_sec) >= parallel_min_sec
        )

        if use_parallel:
            result = transcribe_deepgram_parallel(use_path, model=m, language=language, logger=log)
        elif p == "deepgram":
            result = fn(use_path, model=m, language=language, logger=log)
        else:
            result = fn(use_path, model=m, logger=log)

        result.audio_path = audio_path
        result.metadata = dict(result.metadata or {})
        result.metadata["asr_prep"] = prep_meta
        if use_parallel:
            result.metadata["asr_parallel"] = True
        return result
    finally:
        if temp_prep and os.path.isfile(temp_prep):
            try:
                os.unlink(temp_prep)
            except OSError:
                pass

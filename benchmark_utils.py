"""
Shared helpers for ASR benchmark batch runs and scoring.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from asr_providers import TranscriptionResult, asr_run_label, resolve_model, resolve_provider

BENCHMARK_ROOT = Path(__file__).resolve().parent / "benchmark_runs"
REFERENCES_GEMINI_DIR = BENCHMARK_ROOT / "references" / "gemini"
REFERENCES_FULL_DIR = BENCHMARK_ROOT / "references" / "full"
REPORTS_DIR = BENCHMARK_ROOT / "reports"
CLINICAL_GOLD_DIR = BENCHMARK_ROOT / "clinical_gold"


def audio_slug(audio_path: str | Path) -> str:
    stem = Path(audio_path).stem.lower()
    slug = re.sub(r"[^\w\-]+", "_", stem)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "audio"


def make_run_id(when: Optional[datetime] = None) -> str:
    dt = when or datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def make_run_dir(
    audio_path: str | Path,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    run_id: Optional[str] = None,
    output_root: Optional[str | Path] = None,
) -> Path:
    """Legacy timestamped run dir — prefer make_latest_run_dir for flat layout."""
    root = Path(output_root) if output_root else BENCHMARK_ROOT
    label = asr_run_label(provider, model)
    slug = audio_slug(audio_path)
    rid = run_id or make_run_id()
    run_dir = root / label / slug / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_latest_run_dir(
    audio_path: str | Path,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    output_root: Optional[str | Path] = None,
    archive_existing: bool = True,
) -> Path:
    """Flat run dir: benchmark_runs/{provider_model}/{slug}/ (one active run per audio)."""
    root = Path(output_root) if output_root else BENCHMARK_ROOT
    label = asr_run_label(provider, model)
    slug = audio_slug(audio_path)
    run_dir = root / label / slug
    if archive_existing and run_dir.is_dir():
        has_artifacts = (
            (run_dir / "step1_raw_transcription.txt").exists()
            or (run_dir / "manifest.json").exists()
            or bool(list(run_dir.glob("soap_note_*.json")))
        )
        if has_artifacts:
            archive_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def archive_run_dir(run_dir: Path) -> Path:
    """Move current slug dir contents into archive/{timestamp}/ before overwrite."""
    archive_root = run_dir / "archive" / make_run_id()
    archive_root.mkdir(parents=True, exist_ok=True)
    for item in run_dir.iterdir():
        if item.name == "archive":
            continue
        dest = archive_root / item.name
        if dest.exists():
            continue
        item.rename(dest)
    return archive_root


def full_reference_dir(slug: str, references_root: Optional[Path] = None) -> Path:
    root = references_root or REFERENCES_FULL_DIR
    return root / slug


def find_latest_glob(run_dir: Path, pattern: str) -> Optional[Path]:
    """Return newest file matching glob pattern under run_dir (by mtime)."""
    matches = sorted(run_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def gemini_reference_path(audio_path: str | Path, references_root: Optional[Path] = None) -> Path:
    root = references_root or REFERENCES_GEMINI_DIR
    return root / f"{audio_slug(audio_path)}.txt"


def save_step1_artifacts(
    output_dir: str | Path,
    result: TranscriptionResult,
    *,
    audio_path: Optional[str] = None,
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    step1_path = out / "step1_raw_transcription.txt"
    step1_path.write_text(result.text, encoding="utf-8")

    metadata = {
        "provider": result.provider,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "audio_path": audio_path or result.audio_path,
        "char_count": result.char_count,
        "word_count": result.word_count,
        **result.metadata,
    }
    meta_path = out / "step1_asr_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "step1_raw_transcription": str(step1_path),
        "step1_asr_metadata": str(meta_path),
    }


def write_manifest(
    output_dir: str | Path,
    manifest: Dict[str, Any],
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_manifest(
    *,
    audio_file: str,
    run_id: str,
    mode: str,
    status: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    latency_ms: Optional[int] = None,
    artifacts: Optional[Dict[str, str]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    p = resolve_provider(provider)
    m = resolve_model(p, model)
    slug = audio_slug(audio_file)
    return {
        "audio_file": str(audio_file),
        "audio_slug": slug,
        "asr_provider": p,
        "asr_model": m,
        "asr_run_label": asr_run_label(p, m),
        "run_id": run_id,
        "mode": mode,
        "status": status,
        "latency_ms": latency_ms,
        "artifacts": artifacts or {},
        "gemini_reference_path": str(gemini_reference_path(audio_file)),
        "error": error,
    }


def _slug_dir_has_step1(slug_dir: Path) -> bool:
    step1 = slug_dir / "step1_raw_transcription.txt"
    if not step1.is_file():
        return False
    text = step1.read_text(encoding="utf-8").strip()
    if not text:
        return False
    if text.lower() in {"the dog has a limp and needs xray", "the dog has a lump and needs x-ray"}:
        return False
    return True


def find_latest_run_dir(hypothesis_root: Path, slug: str) -> Optional[Path]:
    """Return run folder for audio_slug — flat layout first, then legacy timestamp subdirs."""
    slug_dir = hypothesis_root / slug
    if not slug_dir.is_dir():
        return None

    # Flat layout: step1 lives directly under slug_dir
    if _slug_dir_has_step1(slug_dir):
        manifest_path = slug_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "error":
                    pass  # fall through to nested search
                else:
                    return slug_dir
            except (json.JSONDecodeError, OSError):
                return slug_dir
        else:
            return slug_dir

    def _run_sort_key(p: Path) -> tuple:
        name = p.name
        # Timestamp runs: (0, name) so reverse sort keeps newest first
        if re.fullmatch(r"\d{8}_\d{6}", name):
            return (0, name)
        # Everything else (e.g. *_test001) ranks lower
        return (1, name)

    run_dirs = sorted(
        (p for p in slug_dir.iterdir() if p.is_dir() and p.name != "archive"),
        key=_run_sort_key,
    )
    # Prefer timestamp runs newest-first, then others newest-first
    timestamp_runs = [p for p in run_dirs if _run_sort_key(p)[0] == 0]
    other_runs = [p for p in run_dirs if _run_sort_key(p)[0] == 1]
    ordered = list(reversed(timestamp_runs)) + list(reversed(other_runs))

    for run_dir in ordered:
        manifest_path = run_dir / "manifest.json"
        step1_path = run_dir / "step1_raw_transcription.txt"
        if not step1_path.is_file():
            continue
        text = step1_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Skip obvious mock fixture text
        if text.lower() in {"the dog has a limp and needs xray", "the dog has a lump and needs x-ray"}:
            continue
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "success":
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        return run_dir
    return None


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_punct: bool = False,
    normalize_numbers: bool = False,
) -> str:
    t = (text or "").strip()
    if lowercase:
        t = t.lower()
    if strip_punct:
        t = re.sub(r"[^\w\s]", " ", t)
    if normalize_numbers:
        t = re.sub(r"\d+", "0", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

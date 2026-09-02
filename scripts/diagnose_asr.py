#!/usr/bin/env python3
"""
Profile STEP1 ASR latency for one or more audio files.

Example:
  python scripts/diagnose_asr.py path/to/audio.wav
  python scripts/diagnose_asr.py --input-dir input_audio_examples
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr_providers import profile_audio_file, resolve_model, resolve_provider, transcribe  # noqa: E402

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".webm"}


def _discover_audio(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def profile_one(audio_path: Path, *, provider: str | None, model: str | None, language: str | None, dry_run: bool) -> Dict[str, Any]:
    p = resolve_provider(provider)
    m = resolve_model(p, model)
    prof = profile_audio_file(str(audio_path))
    row: Dict[str, Any] = {
        "audio_file": str(audio_path),
        "provider": p,
        "model": m,
        **prof,
        "latency_ms": None,
        "char_count": None,
        "latency_per_audio_minute_ms": None,
        "http_attempts": None,
    }
    if dry_run:
        return row
    result = transcribe(str(audio_path), provider=p, model=m, language=language)
    row["latency_ms"] = result.latency_ms
    row["char_count"] = result.char_count
    row["http_attempts"] = (result.metadata or {}).get("http_attempts")
    dur = prof.get("duration_sec")
    if dur and dur > 0:
        row["latency_per_audio_minute_ms"] = int(result.latency_ms / (dur / 60.0))
    return row


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose ASR latency vs audio duration.")
    parser.add_argument("audio", nargs="*", help="Audio file path(s)")
    parser.add_argument("--input-dir", default=None, help="Directory of audio files")
    parser.add_argument("--asr-provider", default=None)
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--asr-language", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Profile files only, no API calls")
    parser.add_argument("--parallel", action="store_true", help="Enable ASR_PARALLEL_CHUNKS for this run")
    parser.add_argument("--output", default=None, help="Write JSON report path")
    args = parser.parse_args(argv)

    if args.parallel:
        os.environ["ASR_PARALLEL_CHUNKS"] = "true"

    files: List[Path] = [Path(a) for a in args.audio]
    if args.input_dir:
        files.extend(_discover_audio(Path(args.input_dir)))
    if not files:
        print("No audio files specified.", file=sys.stderr)
        return 1

    rows = [
        profile_one(f, provider=args.asr_provider, model=args.asr_model, language=args.asr_language, dry_run=args.dry_run)
        for f in files
    ]
    print(f"{'File':<30} {'Dur(s)':>8} {'MB':>8} {'Latency':>12} {'ms/min':>10} {'Chars':>8}")
    print("-" * 80)
    for r in rows:
        lat = r.get("latency_ms")
        lat_s = f"{lat/1000:.1f}s" if lat is not None else "dry-run"
        print(
            f"{Path(r['audio_file']).name:<30} "
            f"{r.get('duration_sec') or 0:>8.1f} "
            f"{r.get('file_mb') or 0:>8.2f} "
            f"{lat_s:>12} "
            f"{r.get('latency_per_audio_minute_ms') or '-':>10} "
            f"{r.get('char_count') or '-':>8}"
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

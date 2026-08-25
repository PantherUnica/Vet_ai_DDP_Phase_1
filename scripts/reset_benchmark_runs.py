#!/usr/bin/env python3
"""
Reset benchmark_runs for Test_* audio corpus.

Wipes old Nova-3 / Gemini / hypotheses / reports and recreates empty slug folders.
Keeps references/gemini/PROMPT.txt.

Usage:
  python scripts/reset_benchmark_runs.py              # dry-run (print plan)
  python scripts/reset_benchmark_runs.py --execute    # actually delete
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark_utils import (  # noqa: E402
    BENCHMARK_ROOT,
    REFERENCES_FULL_DIR,
    REFERENCES_GEMINI_DIR,
    audio_slug,
    full_reference_dir,
    gemini_reference_path,
)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".webm"}
INPUT_DIR = ROOT / "input_audio_examples"


def discover_audio(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def wipe_path(path: Path, execute: bool) -> None:
    if not path.exists():
        return
    if execute:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"  deleted: {path}")
    else:
        print(f"  would delete: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset benchmark_runs for Test_* corpus.")
    parser.add_argument("--execute", action="store_true", help="Actually delete files (default: dry-run)")
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    args = parser.parse_args()
    execute = args.execute
    input_dir = Path(args.input_dir).resolve()
    audio_files = discover_audio(input_dir)

    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"\n=== Reset benchmark_runs [{mode}] ===\n")

    # Wipe provider run roots (any deepgram_* / fireworks_* under benchmark_runs)
    if BENCHMARK_ROOT.is_dir():
        for item in BENCHMARK_ROOT.iterdir():
            if item.is_dir() and (
                item.name.startswith("deepgram_")
                or item.name.startswith("fireworks_")
            ):
                wipe_path(item, execute)

    # hypotheses/
    wipe_path(BENCHMARK_ROOT / "hypotheses", execute)

    # Old reports
    reports = BENCHMARK_ROOT / "reports"
    if reports.is_dir():
        for item in reports.iterdir():
            if item.name.startswith("asr_benchmark") or item.name.startswith("pipeline_benchmark"):
                wipe_path(item, execute)
            if item.name.startswith("validation_"):
                wipe_path(item, execute)

    # Gemini refs except PROMPT.txt
    if REFERENCES_GEMINI_DIR.is_dir():
        for item in REFERENCES_GEMINI_DIR.iterdir():
            if item.name != "PROMPT.txt":
                wipe_path(item, execute)

    # Full gold packs (recreate soap_testing starter separately via pipeline setup)
    wipe_path(REFERENCES_FULL_DIR, execute)

    REFERENCES_GEMINI_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCES_FULL_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    # Recreate empty slug folders under deepgram_nova-3
    provider_root = BENCHMARK_ROOT / "deepgram_nova-3"
    provider_root.mkdir(parents=True, exist_ok=True)

    print("\n--- Checklist (slug -> paths) ---\n")
    for audio in audio_files:
        slug = audio_slug(audio)
        slug_dir = provider_root / slug
        if execute:
            slug_dir.mkdir(parents=True, exist_ok=True)
        gemini_ref = gemini_reference_path(audio)
        full_gold = full_reference_dir(slug)
        print(f"{audio.name}")
        print(f"  slug:         {slug}")
        print(f"  nova-3 out:   {slug_dir}/")
        print(f"  gemini ref:   {gemini_ref}")
        print(f"  full gold:    {full_gold}/")
        print()

    if not execute:
        print("Dry-run only. Re-run with --execute to apply.\n")
    else:
        print(f"Reset complete. {len(audio_files)} slug folder(s) ready under {provider_root}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

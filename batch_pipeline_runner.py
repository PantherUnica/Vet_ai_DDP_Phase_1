#!/usr/bin/env python3
"""
Batch-run ASR or full SOAP pipeline over input audio files.

Examples:
  python batch_pipeline_runner.py --input-dir input_audio_examples --mode asr-only
  python batch_pipeline_runner.py --mode full --asr-provider deepgram --asr-model nova-3
  python batch_pipeline_runner.py --dry-run --exclude "*-1.wav"
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable, List, Optional

from asr_providers import asr_run_label, resolve_model, resolve_provider, transcribe
from benchmark_utils import (
    BENCHMARK_ROOT,
    REFERENCES_GEMINI_DIR,
    audio_slug,
    build_manifest,
    full_reference_dir,
    gemini_reference_path,
    make_latest_run_dir,
    make_run_id,
    save_step1_artifacts,
    write_manifest,
)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".webm"}


def discover_audio_files(input_dir: Path) -> List[Path]:
    files = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
    return files


def filter_files(
    files: Iterable[Path],
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
) -> List[Path]:
    selected: List[Path] = []
    for path in files:
        name = path.name
        if include_patterns and not any(fnmatch.fnmatch(name, p) for p in include_patterns):
            continue
        if exclude_patterns and any(fnmatch.fnmatch(name, p) for p in exclude_patterns):
            continue
        selected.append(path)
    return selected


def apply_asr_env(provider: Optional[str], model: Optional[str], language: Optional[str] = None) -> None:
    if provider:
        os.environ["ASR_PROVIDER"] = provider
    if model:
        os.environ["ASR_MODEL"] = model
    if language:
        os.environ["ASR_LANGUAGE"] = language


def print_run_summary(
    audio_path: Path,
    run_dir: Path,
    provider: str,
    model: str,
    status: str,
) -> None:
    slug = audio_slug(audio_path)
    ref_path = gemini_reference_path(audio_path)
    icon = "[OK]" if status == "success" else "[ERR]" if status == "error" else "[--]"
    print(f"\n{icon} {audio_path.name}")
    print(f"   slug:       {slug}")
    print(f"   asr:        {provider} / {model}")
    print(f"   output:     {run_dir}")
    print(f"   step1:      {run_dir / 'step1_raw_transcription.txt'}")
    print(f"   gemini ref: {ref_path}")
    print(f"   full gold:  {full_reference_dir(slug)}")


def run_asr_only(
    audio_path: Path,
    run_dir: Path,
    provider: str,
    model: str,
) -> dict:
    result = transcribe(str(audio_path), provider=provider, model=model)
    artifacts = save_step1_artifacts(run_dir, result, audio_path=str(audio_path))
    manifest = build_manifest(
        audio_file=str(audio_path),
        run_id=audio_slug(audio_path),
        mode="asr-only",
        status="success",
        provider=provider,
        model=model,
        latency_ms=result.latency_ms,
        artifacts={
            "step1_raw_transcription": "step1_raw_transcription.txt",
            "step1_asr_metadata": "step1_asr_metadata.json",
        },
    )
    write_manifest(run_dir, manifest)
    return manifest


def run_full_pipeline(audio_path: Path, run_dir: Path, provider: str, model: str) -> dict:
    from SOAP_notes_phase1_experiment import generate_soap_note_from_audio_async

    asyncio.run(generate_soap_note_from_audio_async(str(audio_path), output_dir=str(run_dir)))

    meta_path = run_dir / "step1_asr_metadata.json"
    latency_ms = None
    if meta_path.is_file():
        import json

        try:
            latency_ms = json.loads(meta_path.read_text(encoding="utf-8")).get("latency_ms")
        except (json.JSONDecodeError, OSError):
            pass

    manifest = build_manifest(
        audio_file=str(audio_path),
        run_id=audio_slug(audio_path),
        mode="full",
        status="success",
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        artifacts={
            "step1_raw_transcription": "step1_raw_transcription.txt",
            "step1_asr_metadata": "step1_asr_metadata.json",
        },
    )
    write_manifest(run_dir, manifest)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch ASR / pipeline runner for benchmark corpus.")
    parser.add_argument(
        "--input-dir",
        default="input_audio_examples",
        help="Folder containing audio files (default: input_audio_examples)",
    )
    parser.add_argument(
        "--output-root",
        default=str(BENCHMARK_ROOT),
        help="Root output directory (default: benchmark_runs/)",
    )
    parser.add_argument(
        "--mode",
        choices=("asr-only", "full"),
        default="asr-only",
        help="asr-only: Step 1 only; full: complete SOAP pipeline",
    )
    parser.add_argument("--asr-provider", default=None, help="Override ASR_PROVIDER env")
    parser.add_argument("--asr-model", default=None, help="Override ASR_MODEL env")
    parser.add_argument("--asr-language", default=None, help="Override ASR_LANGUAGE env (e.g. multi, en, hi, kn)")
    parser.add_argument("--include", action="append", default=None, help="Glob include filter (repeatable)")
    parser.add_argument("--exclude", action="append", default=None, help="Glob exclude filter (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without API calls")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    apply_asr_env(args.asr_provider, args.asr_model, args.asr_language)
    provider = resolve_provider()
    model = resolve_model(provider)
    label = asr_run_label(provider, model)

    files = filter_files(discover_audio_files(input_dir), args.include, args.exclude)
    if not files:
        print(f"No audio files found in {input_dir}", file=sys.stderr)
        return 1

    REFERENCES_GEMINI_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Batch runner: {len(files)} file(s) | mode={args.mode} | asr={provider}/{model}")
    print(f"Output root: {Path(args.output_root) / label}")

    results = []
    for audio_path in files:
        run_id = make_run_id()
        slug = audio_slug(audio_path)

        if args.dry_run:
            label = asr_run_label(provider, model)
            run_dir = Path(args.output_root) / label / slug
            print_run_summary(audio_path, run_dir, provider, model, "dry-run")
            results.append({"audio": str(audio_path), "status": "dry-run", "run_dir": str(run_dir)})
            continue

        run_dir = make_latest_run_dir(
            audio_path,
            provider=provider,
            model=model,
            output_root=args.output_root,
        )

        try:
            if args.mode == "asr-only":
                manifest = run_asr_only(audio_path, run_dir, provider, model)
            else:
                manifest = run_full_pipeline(audio_path, run_dir, provider, model)
            # Flat layout uses slug as run_id
            manifest["run_id"] = slug
            write_manifest(run_dir, manifest)
            print_run_summary(audio_path, run_dir, provider, model, "success")
            results.append({"audio": str(audio_path), "status": "success", "run_dir": str(run_dir), "manifest": manifest})
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            manifest = build_manifest(
                audio_file=str(audio_path),
                run_id=slug,
                mode=args.mode,
                status="error",
                provider=provider,
                model=model,
                error=err,
            )
            write_manifest(run_dir, manifest)
            print_run_summary(audio_path, run_dir, provider, model, "error")
            print(f"   error:      {err}", file=sys.stderr)
            traceback.print_exc()
            results.append({"audio": str(audio_path), "status": "error", "run_dir": str(run_dir), "error": err})

    ok = sum(1 for r in results if r["status"] == "success")
    fail = sum(1 for r in results if r["status"] == "error")
    dry = sum(1 for r in results if r["status"] == "dry-run")
    print(f"\nDone: {ok} succeeded, {fail} failed, {dry} dry-run (of {len(results)} total)")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

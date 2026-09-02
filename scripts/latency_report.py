#!/usr/bin/env python3
"""
Summarize pipeline timing JSON files across one or many run directories.

Examples:
  python scripts/latency_report.py --run-dir doctor_ui/runs/13
  python scripts/latency_report.py --runs-root doctor_ui/runs
  python scripts/latency_report.py --runs-root benchmark_runs/deepgram_nova-3 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_timing import format_duration_ms, load_pipeline_timing  # noqa: E402


STAGE_KEYS = [
    "total_ms",
    "user_perceived_total_ms",
    "pipeline_only_ms",
    "step1_asr_ui_ms",
    "step1_transcription_ms",
    "step2_total_ms",
    "step2_super_pass_ms",
    "step2_brain_ner_ms",
    "step2_cer_ms",
    "grounding_ms",
    "step3_soap_ms",
    "step4_injection_ms",
    "phase1_total_ms",
    "phase2_total_ms",
    "phase2_step1_atom_extraction_ms",
    "phase2_step2_post_process_ms",
    "phase2_step3_dashboard_ms",
]


def _discover_run_dirs(runs_root: Path) -> List[Path]:
    if not runs_root.is_dir():
        return []
    dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if any((d / "pipeline_timing_latest.json").is_file() for d in dirs):
        return sorted(dirs, key=lambda p: p.name)
    # Nested benchmark layout: runs_root/<slug>/...
    nested: List[Path] = []
    for slug_dir in dirs:
        if slug_dir.name == "archive":
            continue
        latest = slug_dir / "pipeline_timing_latest.json"
        if latest.is_file():
            nested.append(slug_dir)
            continue
        for sub in slug_dir.iterdir():
            if sub.is_dir() and (sub / "pipeline_timing_latest.json").is_file():
                nested.append(sub)
    return sorted(nested, key=lambda p: str(p))


def _flatten_report(run_dir: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    stages = report.get("stages") or {}
    phase2 = stages.get("phase2") or {}
    row: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "source": report.get("source"),
        "generated_at": report.get("generated_at"),
        "total_ms": report.get("total_ms"),
        "user_perceived_total_ms": report.get("user_perceived_total_ms"),
        "pipeline_only_ms": report.get("pipeline_only_ms"),
        "step1_asr_ui_ms": report.get("step1_asr_ui_ms"),
        "critical_path_ms": report.get("critical_path_ms"),
        "step1_transcription_ms": stages.get("step1_transcription_ms"),
        "step2_total_ms": stages.get("step2_total_ms"),
        "step2_super_pass_ms": stages.get("step2_super_pass_ms"),
        "step2_brain_ner_ms": stages.get("step2_brain_ner_ms"),
        "step2_cer_ms": stages.get("step2_cer_ms"),
        "grounding_ms": stages.get("grounding_ms"),
        "step3_soap_ms": stages.get("step3_soap_ms"),
        "step4_injection_ms": stages.get("step4_injection_ms"),
        "phase1_total_ms": stages.get("phase1_total_ms"),
        "phase2_total_ms": stages.get("phase2_total_ms"),
        "phase2_step1_atom_extraction_ms": phase2.get("step1_atom_extraction_ms"),
        "phase2_step2_post_process_ms": phase2.get("step2_post_process_ms"),
        "phase2_step3_dashboard_ms": phase2.get("step3_dashboard_ms"),
    }
    meta = report.get("metadata") or {}
    row["entity_count"] = meta.get("entity_count")
    row["transcript_chars"] = meta.get("transcript_chars")
    return row


def _mean(vals: List[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def _p50(vals: List[float]) -> Optional[float]:
    return statistics.median(vals) if vals else None


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No timing reports found.")
        return

    print(f"Runs with timing data: {len(rows)}\n")
    print(f"{'Run':<40} {'Total':>10} {'ASR+Pipe':>10} {'STEP1':>10} {'STEP2':>10} {'GROUND':>10} {'STEP3':>10} {'PHASE2':>10}")
    print("-" * 110)
    for row in rows:
        name = Path(row["run_dir"]).name
        perceived = row.get("user_perceived_total_ms") or row.get("total_ms")
        print(
            f"{name:<40} "
            f"{format_duration_ms(row.get('total_ms')):>10} "
            f"{format_duration_ms(perceived):>10} "
            f"{format_duration_ms(row.get('step1_transcription_ms') or row.get('step1_asr_ui_ms')):>10} "
            f"{format_duration_ms(row.get('step2_total_ms')):>10} "
            f"{format_duration_ms(row.get('grounding_ms')):>10} "
            f"{format_duration_ms(row.get('step3_soap_ms')):>10} "
            f"{format_duration_ms(row.get('phase2_total_ms')):>10}"
        )

    print("\nAggregates (ms):")
    for key in STAGE_KEYS:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if not vals:
            continue
        print(
            f"  {key}: mean={_mean(vals):.0f}  p50={_p50(vals):.0f}  max={max(vals):.0f}  n={len(vals)}"
        )


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_dir",
        "source",
        "generated_at",
        "total_ms",
        "user_perceived_total_ms",
        "pipeline_only_ms",
        "step1_asr_ui_ms",
        "critical_path_ms",
        *STAGE_KEYS[1:],
        "entity_count",
        "transcript_chars",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize pipeline timing JSON reports.")
    parser.add_argument("--run-dir", default=None, help="Single run directory")
    parser.add_argument("--runs-root", default=None, help="Root containing multiple run directories")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args(argv)

    run_dirs: List[Path] = []
    if args.run_dir:
        run_dirs = [Path(args.run_dir).resolve()]
    elif args.runs_root:
        run_dirs = _discover_run_dirs(Path(args.runs_root).resolve())
    else:
        parser.error("Provide --run-dir or --runs-root")

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        report = load_pipeline_timing(run_dir)
        if report:
            rows.append(_flatten_report(run_dir, report))

    print_summary(rows)
    if args.csv:
        write_csv(rows, Path(args.csv))
        print(f"\nCSV written: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

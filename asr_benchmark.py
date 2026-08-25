#!/usr/bin/env python3
"""
Score Step-1 ASR transcripts against Gemini reference files (WER / CER).

Example:
  python asr_benchmark.py \\
    --hypothesis-root benchmark_runs/deepgram_nova-3 \\
    --reference-dir benchmark_runs/references/gemini \\
    --output benchmark_runs/reports/asr_benchmark_20260820
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark_utils import (
    BENCHMARK_ROOT,
    find_latest_run_dir,
    normalize_text,
)

try:
    import jiwer
except ImportError:
    jiwer = None  # type: ignore


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_latency(run_dir: Path) -> Optional[int]:
    meta = run_dir / "step1_asr_metadata.json"
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("latency_ms")
    except (json.JSONDecodeError, OSError):
        return None


def score_pair(reference: str, hypothesis: str) -> Dict[str, Any]:
    if jiwer is None:
        raise RuntimeError("jiwer is not installed. Run: pip install jiwer")

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    if not ref_norm:
        return {
            "wer": None,
            "cer": None,
            "substitutions": None,
            "deletions": None,
            "insertions": None,
            "hits": None,
            "reference_words": 0,
            "hypothesis_words": len(hyp_norm.split()) if hyp_norm else 0,
            "note": "empty reference after normalization",
        }

    wer = float(jiwer.wer(ref_norm, hyp_norm))
    cer = float(jiwer.cer(ref_norm, hyp_norm))
    measures = jiwer.process_words(ref_norm, hyp_norm)
    return {
        "wer": wer,
        "cer": cer,
        "substitutions": int(measures.substitutions),
        "deletions": int(measures.deletions),
        "insertions": int(measures.insertions),
        "hits": int(measures.hits),
        "reference_words": len(ref_norm.split()),
        "hypothesis_words": len(hyp_norm.split()) if hyp_norm else 0,
    }


def discover_slugs(hypothesis_root: Path) -> List[str]:
    if not hypothesis_root.is_dir():
        return []
    return sorted(p.name for p in hypothesis_root.iterdir() if p.is_dir())


def run_benchmark(
    hypothesis_root: Path,
    reference_dir: Path,
    *,
    run_id: Optional[str] = None,
    strip_punct: bool = False,
    normalize_numbers: bool = False,
) -> Dict[str, Any]:
    slugs = discover_slugs(hypothesis_root)
    rows: List[Dict[str, Any]] = []
    missing_references: List[str] = []
    missing_hypotheses: List[str] = []

    for slug in slugs:
        ref_path = reference_dir / f"{slug}.txt"
        if run_id:
            hyp_dir = hypothesis_root / slug / run_id
        else:
            hyp_dir = find_latest_run_dir(hypothesis_root, slug)

        if hyp_dir is None or not (hyp_dir / "step1_raw_transcription.txt").is_file():
            missing_hypotheses.append(slug)
            continue
        if not ref_path.is_file():
            missing_references.append(slug)
            continue

        reference = _load_text(ref_path)
        hypothesis = _load_text(hyp_dir / "step1_raw_transcription.txt")

        if strip_punct or normalize_numbers:
            reference = normalize_text(
                reference,
                strip_punct=strip_punct,
                normalize_numbers=normalize_numbers,
            )
            hypothesis = normalize_text(
                hypothesis,
                strip_punct=strip_punct,
                normalize_numbers=normalize_numbers,
            )

        metrics = score_pair(reference, hypothesis)
        rows.append(
            {
                "audio_slug": slug,
                "reference_path": str(ref_path),
                "hypothesis_path": str(hyp_dir / "step1_raw_transcription.txt"),
                "run_dir": str(hyp_dir),
                "latency_ms": _load_latency(hyp_dir),
                **metrics,
            }
        )

    wers = [r["wer"] for r in rows if r.get("wer") is not None]
    cers = [r["cer"] for r in rows if r.get("cer") is not None]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]

    summary = {
        "files_scored": len(rows),
        "mean_wer": statistics.mean(wers) if wers else None,
        "median_wer": statistics.median(wers) if wers else None,
        "mean_cer": statistics.mean(cers) if cers else None,
        "median_cer": statistics.median(cers) if cers else None,
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis_root": str(hypothesis_root),
        "reference_dir": str(reference_dir),
        "run_id_filter": run_id,
        "summary": summary,
        "rows": rows,
        "missing_references": missing_references,
        "missing_hypotheses": missing_hypotheses,
    }


def write_csv(report: Dict[str, Any], path: Path) -> None:
    fieldnames = [
        "audio_slug",
        "wer",
        "cer",
        "substitutions",
        "deletions",
        "insertions",
        "hits",
        "reference_words",
        "hypothesis_words",
        "latency_ms",
        "reference_path",
        "hypothesis_path",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in report.get("rows", []):
            writer.writerow(row)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ASR Step-1 benchmark vs Gemini references.")
    parser.add_argument(
        "--hypothesis-root",
        required=True,
        help="Root folder for ASR runs, e.g. benchmark_runs/deepgram_nova-3",
    )
    parser.add_argument(
        "--reference-dir",
        default=str(BENCHMARK_ROOT / "references" / "gemini"),
        help="Folder with {audio_slug}.txt Gemini reference transcripts",
    )
    parser.add_argument("--run-id", default=None, help="Pin a specific run_id instead of latest per slug")
    parser.add_argument("--output", default=None, help="Output path prefix (writes .json and .csv)")
    parser.add_argument("--normalize-punct", action="store_true", help="Strip punctuation before scoring")
    parser.add_argument("--normalize-numbers", action="store_true", help="Normalize digits before scoring")
    args = parser.parse_args(argv)

    if jiwer is None:
        print("Error: jiwer not installed. Run: pip install jiwer", file=sys.stderr)
        return 1

    hypothesis_root = Path(args.hypothesis_root).resolve()
    reference_dir = Path(args.reference_dir).resolve()

    if not hypothesis_root.is_dir():
        print(f"Hypothesis root not found: {hypothesis_root}", file=sys.stderr)
        return 1

    report = run_benchmark(
        hypothesis_root,
        reference_dir,
        run_id=args.run_id,
        strip_punct=args.normalize_punct,
        normalize_numbers=args.normalize_numbers,
    )

    out_prefix = Path(
        args.output
        or (BENCHMARK_ROOT / "reports" / f"asr_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json") if out_prefix.suffix else Path(str(out_prefix) + ".json")
    csv_path = json_path.with_suffix(".csv")

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(report, csv_path)

    s = report["summary"]
    print(f"Scored: {s['files_scored']} file(s)")
    if s["mean_wer"] is not None:
        print(f"Mean WER: {s['mean_wer']:.4f} | Mean CER: {s['mean_cer']:.4f}")
    if report["missing_references"]:
        print(f"Missing references ({len(report['missing_references'])}): {', '.join(report['missing_references'][:5])}...")
    if report["missing_hypotheses"]:
        print(f"Missing hypotheses ({len(report['missing_hypotheses'])}): {', '.join(report['missing_hypotheses'][:5])}...")
    print(f"Report: {json_path}")
    print(f"CSV:    {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

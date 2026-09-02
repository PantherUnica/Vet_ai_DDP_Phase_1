#!/usr/bin/env python3
"""
Compare single-pass vs parallel-chunk ASR on long audio (latency + WER vs reference).

Example:
  python scripts/benchmark_parallel_asr.py --audio input_audio_examples/Test_2_Long_audio.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr_benchmark import score_pair  # noqa: E402
from asr_providers import transcribe  # noqa: E402
from benchmark_utils import audio_slug  # noqa: E402


def _run_once(audio: Path, *, parallel: bool, language: Optional[str]) -> Dict[str, Any]:
    os.environ["ASR_PARALLEL_CHUNKS"] = "true" if parallel else "false"
    t0 = time.perf_counter()
    result = transcribe(str(audio), language=language)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    row: Dict[str, Any] = {
        "mode": "parallel" if parallel else "single",
        "latency_ms": result.latency_ms,
        "wall_ms": wall_ms,
        "char_count": result.char_count,
        "metadata": result.metadata,
    }
    ref = ROOT / "benchmark_runs" / "references" / "gemini" / f"{audio_slug(audio)}.txt"
    if ref.is_file():
        scores = score_pair(ref.read_text(encoding="utf-8"), result.text)
        row["wer"] = scores.get("wer")
        row["cer"] = scores.get("cer")
    return row


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark parallel vs single ASR.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--asr-language", default="multi")
    parser.add_argument("--output", default="reports/asr_parallel_benchmark.json")
    args = parser.parse_args(argv)

    audio = Path(args.audio)
    if not audio.is_file():
        print(f"Audio not found: {audio}", file=sys.stderr)
        return 1

    print(f"Benchmarking {audio.name} (single then parallel)...")
    single = _run_once(audio, parallel=False, language=args.asr_language)
    print(f"  single:   {single['latency_ms']}ms  WER={single.get('wer')}  CER={single.get('cer')}")
    parallel = _run_once(audio, parallel=True, language=args.asr_language)
    print(f"  parallel: {parallel['latency_ms']}ms  WER={parallel.get('wer')}  CER={parallel.get('cer')}")

    speedup = None
    if single.get("latency_ms") and parallel.get("latency_ms"):
        speedup = single["latency_ms"] / max(parallel["latency_ms"], 1)

    quality_ok = True
    if single.get("wer") is not None and parallel.get("wer") is not None:
        quality_ok = parallel["wer"] <= single["wer"]
    if single.get("cer") is not None and parallel.get("cer") is not None:
        quality_ok = quality_ok and parallel["cer"] <= single["cer"]

    report = {
        "audio": str(audio),
        "single": single,
        "parallel": parallel,
        "speedup_x": speedup,
        "quality_ok_vs_single": quality_ok,
        "recommendation": (
            "Enable ASR_PARALLEL_CHUNKS=true in .env"
            if quality_ok and speedup and speedup > 1.15
            else "Keep ASR_PARALLEL_CHUNKS=false (quality or speedup insufficient)"
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{report['recommendation']}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

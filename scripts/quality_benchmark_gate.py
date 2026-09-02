#!/usr/bin/env python3
"""
Quality gate: compare ASR + pipeline benchmark reports to a baseline.
Ship only when mean WER/CER/entity F1/SOAP fact recall are >= baseline.

Example:
  python scripts/quality_benchmark_gate.py \\
    --asr-hypothesis benchmark_runs/deepgram_nova-3 \\
    --asr-baseline benchmark_runs/reports/asr_baseline.json \\
    --pipeline-hypothesis benchmark_runs/deepgram_nova-3 \\
    --pipeline-baseline benchmark_runs/reports/pipeline_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr_benchmark import run_benchmark  # noqa: E402
from pipeline_benchmark import run_pipeline_benchmark  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_gte(name: str, current: Optional[float], baseline: Optional[float], failures: list) -> None:
    if current is None or baseline is None:
        return
    if current < baseline:
        failures.append(f"{name}: {current:.4f} < baseline {baseline:.4f}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Quality benchmark gate vs baseline.")
    parser.add_argument("--asr-hypothesis", default=None, help="Run dir root for ASR")
    parser.add_argument("--asr-baseline", default=None, help="Baseline asr_benchmark.json")
    parser.add_argument("--asr-reference", default=str(ROOT / "benchmark_runs" / "references" / "gemini"))
    parser.add_argument("--pipeline-hypothesis", default=None)
    parser.add_argument("--pipeline-baseline", default=None)
    parser.add_argument("--pipeline-reference", default=str(ROOT / "benchmark_runs" / "references" / "full"))
    parser.add_argument("--output", default=None, help="Write gate result JSON")
    args = parser.parse_args(argv)

    failures: list = []
    report: Dict[str, Any] = {"passed": True, "checks": []}

    if args.asr_hypothesis and args.asr_baseline:
        asr_current = run_benchmark(
            Path(args.asr_hypothesis),
            Path(args.asr_reference),
        )
        asr_base = _load_json(Path(args.asr_baseline))
        cur_s = asr_current.get("summary") or {}
        base_s = asr_base.get("summary") or {}
        # Lower WER/CER is better — current must be <= baseline (inverted gte)
        for key in ("mean_wer", "mean_cer"):
            c = cur_s.get(key)
            b = base_s.get(key)
            if c is not None and b is not None and c > b:
                failures.append(f"ASR {key}: {c:.4f} worse than baseline {b:.4f}")
        report["checks"].append({"asr_current": cur_s, "asr_baseline": base_s})

    if args.pipeline_hypothesis and args.pipeline_baseline:
        pipe_current = run_pipeline_benchmark(
            Path(args.pipeline_hypothesis),
            Path(args.pipeline_reference),
        )
        pipe_base = _load_json(Path(args.pipeline_baseline))
        cur_s = pipe_current.get("summary") or {}
        base_s = pipe_base.get("summary") or {}
        _check_gte("entity F1", cur_s.get("mean_entity_f1"), base_s.get("mean_entity_f1"), failures)
        _check_gte(
            "SOAP fact recall",
            cur_s.get("mean_soap_fact_recall"),
            base_s.get("mean_soap_fact_recall"),
            failures,
        )
        report["checks"].append({"pipeline_current": cur_s, "pipeline_baseline": base_s})

    report["passed"] = len(failures) == 0
    report["failures"] = failures

    if failures:
        print("QUALITY GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("QUALITY GATE PASSED")

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

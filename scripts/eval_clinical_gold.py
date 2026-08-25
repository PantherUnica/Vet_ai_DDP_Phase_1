#!/usr/bin/env python3
"""Evaluate ASR/SOAP quality against clinical_gold rows (sample WER + field overlap)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from asr_benchmark import score_pair  # noqa: E402
from benchmark_utils import CLINICAL_GOLD_DIR, normalize_text  # noqa: E402

SOAP_KEYS = [
    "subjective", "objective", "assessment", "plan", "conclusion",
    "key_issues", "abnormal_findings", "customer_instructions", "reminders",
]


def _field_overlap(gold: str, hyp: str) -> float:
    g = normalize_text(gold or "")
    h = normalize_text(hyp or "")
    if not g:
        return 1.0
    g_words = set(g.split())
    h_words = set(h.split())
    if not g_words:
        return 0.0
    return len(g_words & h_words) / len(g_words)


def eval_row(row: Dict[str, Any], hypothesis_transcript: str, hypothesis_soap: Dict[str, str]) -> Dict[str, Any]:
    asr = score_pair(row.get("transcript", ""), hypothesis_transcript)
    soap_scores = {}
    for key in SOAP_KEYS:
        gold_val = row.get(key, "") or ""
        hyp_val = hypothesis_soap.get(key) or hypothesis_soap.get(key.title()) or ""
        soap_scores[f"{key}_overlap"] = _field_overlap(str(gold_val), str(hyp_val))
    return {"asr_wer": asr.get("wer"), "asr_cer": asr.get("cer"), **soap_scores}


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval clinical gold vs hypothesis")
    parser.add_argument("--gold-row", default="row_001.json")
    parser.add_argument("--hypothesis-transcript", required=True)
    parser.add_argument("--hypothesis-soap-json", default=None)
    args = parser.parse_args()

    gold_path = CLINICAL_GOLD_DIR / "rows" / args.gold_row
    if not gold_path.is_file():
        print(f"Run import_clinical_notes.py first. Missing: {gold_path}", file=sys.stderr)
        return 1

    row = json.loads(gold_path.read_text(encoding="utf-8"))
    hyp_text = Path(args.hypothesis_transcript).read_text(encoding="utf-8")
    hyp_soap = {}
    if args.hypothesis_soap_json:
        hyp_soap = json.loads(Path(args.hypothesis_soap_json).read_text(encoding="utf-8"))

    report = eval_row(row, hyp_text, hyp_soap)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

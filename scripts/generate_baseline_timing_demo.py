#!/usr/bin/env python3
"""Generate illustrative baseline timing JSON for typed, voice, and CLI audio entry paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_timing import save_pipeline_timing  # noqa: E402

BASELINES = {
    "baseline_typed": {
        "source": "typed",
        "total_ms": 164000,
        "critical_path_ms": 164000,
        "stages": {
            "step1_transcription_ms": 0,
            "step2_total_ms": 48000,
            "step2_super_pass_ms": 22000,
            "step2_brain_ner_ms": 18000,
            "step2_cer_ms": None,
            "step2_cer_skipped": True,
            "step2_batch_intent_ms": None,
            "grounding_ms": 55000,
            "step3_soap_ms": 32000,
            "step4_injection_ms": 0,
            "step4_injection_skipped": True,
            "phase1_total_ms": 135000,
            "phase2_total_ms": 29000,
            "phase2": {
                "step1_atom_extraction_ms": 23000,
                "step2_post_process_ms": 2500,
                "step3_dashboard_ms": 3500,
                "early_subjective_objective_ms": None,
            },
            "soap_pipeline_ms": 32000,
            "billing_pipeline_ms": 55000,
            "parallel_overlap_ms": 32000,
        },
        "metadata": {"transcript_chars": 2352, "entity_count": 6, "source": "typed"},
    },
    "baseline_voice": {
        "source": "voice",
        "total_ms": 192400,
        "critical_path_ms": 185000,
        "stages": {
            "step1_transcription_ms": 25000,
            "step2_total_ms": 48000,
            "step2_super_pass_ms": 22000,
            "step2_brain_ner_ms": 18000,
            "step2_cer_ms": None,
            "step2_cer_skipped": True,
            "grounding_ms": 55000,
            "step3_soap_ms": 32000,
            "step4_injection_ms": 0,
            "step4_injection_skipped": True,
            "phase1_total_ms": 160000,
            "phase2_total_ms": 28400,
            "phase2": {
                "step1_atom_extraction_ms": 22000,
                "step2_post_process_ms": 2100,
                "step3_dashboard_ms": 4300,
            },
            "soap_pipeline_ms": 32000,
            "billing_pipeline_ms": 55000,
            "parallel_overlap_ms": 32000,
        },
        "metadata": {"transcript_chars": 2352, "entity_count": 6, "source": "voice"},
    },
    "baseline_cli_audio": {
        "source": "audio",
        "total_ms": 7134,
        "critical_path_ms": 7134,
        "stages": {
            "step1_transcription_ms": 7134,
            "step2_total_ms": None,
            "grounding_ms": None,
            "step3_soap_ms": None,
            "phase2_total_ms": None,
            "phase2": {},
        },
        "metadata": {"note": "ASR-only short audio benchmark reference"},
    },
}


def main() -> int:
    out_root = ROOT / "doctor_ui" / "runs" / "baseline_timing"
    out_root.mkdir(parents=True, exist_ok=True)
    for name, report in BASELINES.items():
        run_dir = out_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        report = dict(report)
        report.setdefault(
            "parallel_note",
            "grounding and step3_soap overlap after STEP2; critical_path uses max(soap, grounding) not their sum",
        )
        report.setdefault("generated_at", "2026-09-01T00:00:00")
        save_pipeline_timing(run_dir, report, timestamp="20260901_baseline")
        print(f"Wrote {run_dir / 'pipeline_timing_latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

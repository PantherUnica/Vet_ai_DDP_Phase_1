"""
Pipeline stage timing for VetAI SOAP workflow.

Records wall-clock durations per stage, handles parallel branches, and saves
pipeline_timing_{timestamp}.json + pipeline_timing_latest.json per run.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _ms(seconds: Optional[float]) -> Optional[int]:
    if seconds is None:
        return None
    return max(0, int(round(seconds * 1000)))


def _delta_ms(start: Optional[float], end: Optional[float]) -> Optional[int]:
    if start is None or end is None:
        return None
    return _ms(end - start)


def format_duration_ms(ms: Optional[int]) -> str:
    """Human-readable duration for UI display."""
    if ms is None:
        return "n/a"
    if ms < 1000:
        return f"{ms}ms"
    sec = ms / 1000.0
    if sec < 60:
        return f"{sec:.1f}s"
    minutes = int(sec // 60)
    rem = sec % 60
    return f"{minutes}m {rem:.1f}s"


class PipelineTimer:
    """Lightweight perf_counter-based stage timer."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._marks: Dict[str, float] = {}
        self._durations_ms: Dict[str, int] = {}
        self._flags: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}

    def mark(self, name: str) -> None:
        self._marks[name] = time.perf_counter()

    def set_duration_ms(self, name: str, ms: Optional[int]) -> None:
        if ms is not None:
            self._durations_ms[name] = max(0, int(ms))

    def set_flag(self, name: str, value: Any) -> None:
        self._flags[name] = value

    def set_metadata(self, **kwargs: Any) -> None:
        self._metadata.update(kwargs)

    def elapsed_ms(self, start_mark: str, end_mark: str) -> Optional[int]:
        return _delta_ms(self._marks.get(start_mark), self._marks.get(end_mark))

    def duration_since_start_ms(self, end_mark: str) -> Optional[int]:
        return _delta_ms(self._start, self._marks.get(end_mark))

    def merge_phase2(self, phase2_timing: Dict[str, Any]) -> None:
        if phase2_timing:
            self._metadata["phase2_timing"] = phase2_timing

    def build_report(
        self,
        *,
        source: str = "audio",
        step1_asr_ms: Optional[int] = None,
        step1_asr_ui_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        m = self._marks
        d = self._durations_ms
        f = self._flags

        step1_ms = step1_asr_ms if step1_asr_ms is not None else self.elapsed_ms("pipeline_start", "transcription_done")
        if step1_ms is None and m.get("transcription_done") is not None:
            step1_ms = self.duration_since_start_ms("transcription_done")

        super_pass_ms = self.elapsed_ms("step2_super_pass_start", "step2_super_pass_done")
        brain_ner_ms = self.elapsed_ms("step2_brain_ner_start", "step2_brain_ner_done")
        cer_ms = d.get("step2_cer")
        cer_skipped = f.get("step2_cer_skipped", cer_ms is None and f.get("step2_cer_attempted") is not True)
        batch_intent_ms = d.get("step2_batch_intent")

        # Whole STEP2 block (super-pass path end)
        step2_total_ms = self.elapsed_ms("transcription_done", "superpass_done")

        grounding_ms = None
        if m.get("billing_start") and m.get("grounding_done"):
            grounding_ms = _delta_ms(m["billing_start"], m["grounding_done"])
        elif m.get("superpass_done") and m.get("grounding_done"):
            grounding_ms = _delta_ms(m["superpass_done"], m["grounding_done"])

        soap_ms = self.elapsed_ms("superpass_done", "soap_done")
        injection_ms = self.elapsed_ms("injection_start", "injection_done")
        if injection_ms is None and f.get("step4_injection_skipped"):
            injection_ms = 0

        phase2_meta = self._metadata.get("phase2_timing") or {}
        phase2_total_ms = phase2_meta.get("phase2_total_ms")
        if phase2_total_ms is None:
            phase2_total_ms = self.elapsed_ms("phase2_start", "phase2_done")

        end_mark = m.get("pipeline_end", time.perf_counter())
        total_ms = _delta_ms(self._start, end_mark if isinstance(end_mark, float) else time.perf_counter())

        # Parallel: SOAP and billing (CER+grounding) overlap after superpass_done
        soap_pipeline_ms = soap_ms
        billing_pipeline_ms = grounding_ms
        parallel_overlap_ms = None
        if soap_pipeline_ms is not None and billing_pipeline_ms is not None:
            parallel_overlap_ms = min(soap_pipeline_ms, billing_pipeline_ms)

        # Critical path through Phase 1 (before Phase 2)
        phase1_parts: List[int] = []
        if step1_ms is not None:
            phase1_parts.append(step1_ms)
        if step2_total_ms is not None:
            phase1_parts.append(step2_total_ms)
        post_step2 = max(soap_pipeline_ms or 0, billing_pipeline_ms or 0)
        if post_step2:
            phase1_parts.append(post_step2)
        if injection_ms is not None:
            phase1_parts.append(injection_ms)
        phase1_total_ms = sum(phase1_parts) if phase1_parts else None

        critical_path_ms = phase1_total_ms
        if critical_path_ms is not None and phase2_total_ms is not None:
            critical_path_ms = critical_path_ms + phase2_total_ms

        phase2_block = {
            "step1_atom_extraction_ms": phase2_meta.get("step1_atom_extraction_ms"),
            "step2_post_process_ms": phase2_meta.get("step2_post_process_ms"),
            "step3_dashboard_ms": phase2_meta.get("step3_dashboard_ms"),
            "early_subjective_objective_ms": phase2_meta.get("early_subjective_objective_ms"),
        }

        # UI ASR (Doctor UI transcribe-before-generate) vs in-pipeline STEP1
        pipeline_only_ms = total_ms
        user_perceived_ms = None
        if step1_asr_ui_ms is not None and step1_asr_ui_ms > 0:
            user_perceived_ms = (pipeline_only_ms or 0) + step1_asr_ui_ms
        elif step1_ms is not None and step1_ms > 0 and source in ("audio", "voice"):
            user_perceived_ms = pipeline_only_ms

        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "total_ms": total_ms,
            "pipeline_only_ms": pipeline_only_ms,
            "user_perceived_total_ms": user_perceived_ms,
            "step1_asr_ui_ms": step1_asr_ui_ms if step1_asr_ui_ms else None,
            "critical_path_ms": critical_path_ms,
            "parallel_note": (
                "grounding and step3_soap overlap after STEP2; "
                "critical_path uses max(soap, grounding) not their sum"
            ),
            "stages": {
                "step1_transcription_ms": step1_ms,
                "step2_total_ms": step2_total_ms,
                "step2_super_pass_ms": super_pass_ms,
                "step2_brain_ner_ms": brain_ner_ms,
                "step2_cer_ms": cer_ms,
                "step2_cer_skipped": bool(cer_skipped),
                "step2_batch_intent_ms": batch_intent_ms,
                "grounding_ms": grounding_ms,
                "step3_soap_ms": soap_ms,
                "step4_injection_ms": injection_ms,
                "step4_injection_skipped": bool(f.get("step4_injection_skipped")),
                "phase1_total_ms": phase1_total_ms,
                "phase2_total_ms": phase2_total_ms,
                "phase2": phase2_block,
                "soap_pipeline_ms": soap_pipeline_ms,
                "billing_pipeline_ms": billing_pipeline_ms,
                "parallel_overlap_ms": parallel_overlap_ms,
            },
            "metadata": dict(self._metadata),
        }


def save_pipeline_timing(
    output_dir: str | Path,
    report: Dict[str, Any],
    *,
    timestamp: str,
) -> Tuple[Path, Path]:
    """Write timestamped and latest timing JSON files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts_path = out / f"pipeline_timing_{timestamp}.json"
    latest_path = out / "pipeline_timing_latest.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    ts_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return ts_path, latest_path


def load_pipeline_timing(output_dir: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    if not output_dir:
        return None
    p = Path(output_dir)
    latest = p / "pipeline_timing_latest.json"
    if latest.is_file():
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    files = sorted(p.glob("pipeline_timing_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# Display labels for Doctor UI (order matters)
TIMING_DISPLAY_ROWS: List[Tuple[str, str, Optional[str]]] = [
    ("total_ms", "Total", None),
    ("step1_transcription_ms", "STEP1 — Transcription", None),
    ("step2_total_ms", "STEP2 — Clean + NER", None),
    ("step2_super_pass_ms", "Super-Pass", "step2_total_ms"),
    ("step2_brain_ner_ms", "Brain NER", "step2_total_ms"),
    ("step2_cer_ms", "CER", "step2_total_ms"),
    ("step2_batch_intent_ms", "Batch Intent", "step2_total_ms"),
    ("grounding_ms", "GROUNDING — Inventory link", None),
    ("step3_soap_ms", "STEP3 — SOAP note", None),
    ("step4_injection_ms", "STEP4 — Injection", None),
    ("phase1_total_ms", "Phase 1 total", None),
    ("phase2_total_ms", "PHASE2 — Knowledge atoms", None),
]


def timing_rows_for_display(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten timing report into UI rows with indentation."""
    stages = report.get("stages") or {}
    phase2 = stages.get("phase2") or {}
    rows: List[Dict[str, Any]] = []

    def _add(key: str, label: str, indent: int = 0, ms: Optional[int] = None) -> None:
        value = ms if ms is not None else stages.get(key)
        if key == "step2_cer_ms" and stages.get("step2_cer_skipped"):
            rows.append({"label": label, "value_ms": None, "display": "skipped", "indent": indent})
            return
        if key == "step4_injection_ms" and stages.get("step4_injection_skipped"):
            rows.append({"label": label, "value_ms": 0, "display": "skipped", "indent": indent})
            return
        if value is None and key not in ("total_ms",):
            return
        rows.append({
            "label": label,
            "value_ms": value,
            "display": format_duration_ms(value) if value is not None else "n/a",
            "indent": indent,
        })

    _add("total_ms", "Total (pipeline only)", ms=report.get("total_ms"))
    ui_asr = report.get("step1_asr_ui_ms")
    if ui_asr:
        rows.append({
            "label": "STEP1 — ASR (Doctor UI, before pipeline)",
            "value_ms": ui_asr,
            "display": format_duration_ms(ui_asr),
            "indent": 0,
        })
    perceived = report.get("user_perceived_total_ms")
    if perceived is not None and perceived != report.get("total_ms"):
        rows.append({
            "label": "Total (ASR + pipeline)",
            "value_ms": perceived,
            "display": format_duration_ms(perceived),
            "indent": 0,
        })
    _add("step1_transcription_ms", "STEP1 — Transcription (in pipeline)")
    _add("step2_total_ms", "STEP2 — Clean + NER")
    _add("step2_super_pass_ms", "Super-Pass", indent=1)
    _add("step2_brain_ner_ms", "Brain NER", indent=1)
    _add("step2_cer_ms", "CER", indent=1)
    _add("step2_batch_intent_ms", "Batch Intent", indent=1)
    _add("grounding_ms", "GROUNDING — Inventory link")
    _add("step3_soap_ms", "STEP3 — SOAP note")
    _add("step4_injection_ms", "STEP4 — Injection")
    _add("phase1_total_ms", "Phase 1 total")
    _add("phase2_total_ms", "PHASE2 — Knowledge atoms")
    if phase2.get("step1_atom_extraction_ms") is not None:
        rows.append({
            "label": "Step1 — Atom extraction (LLM)",
            "value_ms": phase2["step1_atom_extraction_ms"],
            "display": format_duration_ms(phase2["step1_atom_extraction_ms"]),
            "indent": 1,
        })
    if phase2.get("step2_post_process_ms") is not None:
        rows.append({
            "label": "Step2 — Dedupe + manifest stitch",
            "value_ms": phase2["step2_post_process_ms"],
            "display": format_duration_ms(phase2["step2_post_process_ms"]),
            "indent": 1,
        })
    if phase2.get("step3_dashboard_ms") is not None:
        rows.append({
            "label": "Step3 — Verification dashboard",
            "value_ms": phase2["step3_dashboard_ms"],
            "display": format_duration_ms(phase2["step3_dashboard_ms"]),
            "indent": 1,
        })
    early = phase2.get("early_subjective_objective_ms")
    if early is not None:
        rows.append({
            "label": "Early Subjective/Objective extraction",
            "value_ms": early,
            "display": format_duration_ms(early),
            "indent": 1,
        })
    return rows

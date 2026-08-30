"""
Pipeline health / problem flags for VetAI SOAP workflow (Master Doc steps).

Produces a structured report the UI can show, e.g.:
  STEP1 — unclear / short transcript
  GROUNDING — billable entities left unlinked
  STEP3 — Plan empty
  PHASE2 — schema or atoms missing
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PipelineFlag:
    step: str  # STEP1 | STEP2 | GROUNDING | STEP3 | PHASE2 | SYSTEM
    severity: str  # info | warning | error
    code: str
    message: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineHealthReport:
    overall: str  # healthy | degraded | failing
    generated_at: str
    flags: List[PipelineFlag] = field(default_factory=list)
    stage_summary: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "generated_at": self.generated_at,
            "stage_summary": self.stage_summary,
            "flags": [f.to_dict() for f in self.flags],
            "counts": {
                "error": sum(1 for f in self.flags if f.severity == "error"),
                "warning": sum(1 for f in self.flags if f.severity == "warning"),
                "info": sum(1 for f in self.flags if f.severity == "info"),
            },
        }


_UNCLEAR_PATTERNS = re.compile(
    r"\[(?:unclear|inaudible|unintelligible)\]|\b(?:inaudible|unintelligible)\b|\?\?\?",
    re.I,
)
_GARBLED_TOKEN = re.compile(r"\b[a-z]{10,}\b", re.I)  # long nonsense-ish tokens


def _latest(path: Path, pattern: str) -> Optional[Path]:
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_json(path: Optional[Path]) -> Any:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def build_pipeline_flags(
    output_dir: str | Path,
    *,
    soap_json: Optional[Dict[str, Any]] = None,
    entity_manifest: Optional[List[Dict[str, Any]]] = None,
    knowledge_atoms: Optional[Any] = None,
    extra_flags: Optional[List[PipelineFlag]] = None,
) -> PipelineHealthReport:
    out = Path(output_dir)
    flags: List[PipelineFlag] = list(extra_flags or [])
    stages: Dict[str, str] = {
        "STEP1": "ok",
        "STEP2": "ok",
        "GROUNDING": "ok",
        "STEP3": "ok",
        "PHASE2": "ok",
    }

    # --- STEP1 ASR / raw transcript ---
    asr_meta = _load_json(_latest(out, "step1_asr_metadata.json")) or {}
    raw = _load_text(_latest(out, "step1_raw_transcription.txt")) or _load_text(
        _latest(out, "raw_transcription_*.txt")
    )
    if not raw:
        # glob fallback
        raw = _load_text(_latest(out, "raw_transcription_*.txt"))
    cleaned = _load_text(_latest(out, "cleaned_transcript_*.txt"))

    words = len(raw.split()) if raw else 0
    if words and words < 25:
        flags.append(
            PipelineFlag(
                "STEP1",
                "warning",
                "SHORT_TRANSCRIPT",
                "STEP1 transcript is very short — clinical content may be incomplete.",
                f"word_count={words}",
            )
        )
        stages["STEP1"] = "degraded"
    if not raw.strip():
        flags.append(
            PipelineFlag(
                "STEP1",
                "error",
                "EMPTY_TRANSCRIPT",
                "STEP1 produced an empty transcript.",
            )
        )
        stages["STEP1"] = "failing"

    unclear_hits = _UNCLEAR_PATTERNS.findall(raw or "") + _UNCLEAR_PATTERNS.findall(cleaned or "")
    if unclear_hits:
        flags.append(
            PipelineFlag(
                "STEP1",
                "warning",
                "UNCLEAR_AUDIO_MARKERS",
                "STEP1 has unclear/inaudible markers — review those spans before trusting the note.",
                f"markers={unclear_hits[:8]}",
            )
        )
        stages["STEP1"] = "degraded" if stages["STEP1"] == "ok" else stages["STEP1"]

    # Heuristic: repeated garbled tokens in cleaned text
    if cleaned:
        garbled = [t for t in _GARBLED_TOKEN.findall(cleaned) if t.lower() not in ("approximately", "prescription", "vaccination", "temperature")]
        # only flag if several unique long tokens look suspicious (consonant-heavy)
        suspicious = []
        for t in garbled:
            vowels = sum(1 for c in t.lower() if c in "aeiou")
            if vowels <= max(1, len(t) // 5) and len(t) >= 12:
                suspicious.append(t)
        if len(set(suspicious)) >= 2:
            flags.append(
                PipelineFlag(
                    "STEP1",
                    "warning",
                    "SUSPECT_ASR_TOKENS",
                    "STEP1 cleaned transcript has tokens that look ASR-garbled — verify before billing.",
                    f"examples={list(dict.fromkeys(suspicious))[:5]}",
                )
            )
            stages["STEP1"] = "degraded" if stages["STEP1"] == "ok" else stages["STEP1"]

    if asr_meta.get("error"):
        flags.append(
            PipelineFlag(
                "STEP1",
                "error",
                "ASR_ERROR",
                f"STEP1 ASR reported an error: {asr_meta.get('error')}",
            )
        )
        stages["STEP1"] = "failing"

    # --- STEP2 Super-Pass / cleaned ---
    if raw and not (cleaned or "").strip():
        flags.append(
            PipelineFlag(
                "STEP2",
                "error",
                "CLEANING_FAILED",
                "STEP2 cleaned transcript missing — Super-Pass / cleaning may have failed.",
            )
        )
        stages["STEP2"] = "failing"

    brain = _load_json(_latest(out, "brain_ner_output_*.json"))
    brain_ents: List[Any] = []
    if isinstance(brain, list):
        brain_ents = brain
    elif isinstance(brain, dict):
        raw_ents = brain.get("entities") or brain.get("spans") or []
        if isinstance(raw_ents, list):
            brain_ents = raw_ents
    if brain is not None and len(brain_ents) == 0:
        flags.append(
            PipelineFlag(
                "STEP2",
                "warning",
                "NO_NER_ENTITIES",
                "STEP2 Brain NER returned no entities — grounding/RAG may be empty.",
            )
        )
        stages["STEP2"] = "degraded"

    # --- GROUNDING / RAG ---
    manifest = entity_manifest
    if manifest is None:
        manifest = _load_json(_latest(out, "entity_manifest_*.json")) or []
    if not isinstance(manifest, list):
        manifest = []

    grounding = _load_json(_latest(out, "grounding_layer_output_*.json"))
    billable_kinds = {
        "medicine",
        "medication",
        "drug",
        "product",
        "vaccine",
        "procedure",
        "diagnostic",
        "service",
        "nutrition",
        "preventive",
        "parasitecontrol",
    }
    # Methods that are intentional note-only (not a RAG failure)
    note_only_methods = {
        "non_billable_preserved",
        "vital_sign_structured",
        "note_only_below_threshold",
        "skip_signalment",
        "skip_identity",
        "skip_vitals",
    }
    linked = 0
    need_sku = 0  # true billing gaps (unlinked / judge rejected without ID)
    with_suggestions = 0
    dual_reject = 0
    for e in manifest:
        if not isinstance(e, dict):
            continue
        kind = str(e.get("kind") or "").lower().replace(" ", "")
        method = str(e.get("match_method") or "")
        if isinstance(e.get("final_binding"), dict):
            fb = e["final_binding"]
            method = method or str(fb.get("match_method") or "")
        if method in note_only_methods:
            continue
        if kind not in billable_kinds and e.get("route") != "dual_sync":
            continue
        sid = e.get("local_stock_id") or e.get("local_service_id")
        if isinstance(e.get("final_binding"), dict):
            fb = e["final_binding"]
            sid = sid or fb.get("local_stock_id") or fb.get("local_service_id")
        sug = []
        attrs = e.get("attributes") or {}
        if isinstance(attrs, dict):
            sug = attrs.get("suggestions") or []
        if sid:
            linked += 1
        elif method in ("unlinked", "dual_sync_reject", "dual_sync_judge_rejected", "") or not sid:
            need_sku += 1
            if sug:
                with_suggestions += 1
        if method in ("dual_sync_reject", "dual_sync_judge_rejected") or e.get("dual_sync_rejected"):
            dual_reject += 1

    billable = linked + need_sku
    if billable == 0 and manifest:
        flags.append(
            PipelineFlag(
                "GROUNDING",
                "info",
                "NO_BILLABLE_ENTITIES",
                "Grounding found no billable medicine/service entities (note-only run may be OK).",
            )
        )
    elif billable > 0:
        ratio = need_sku / max(billable, 1)
        if linked > 0:
            flags.append(
                PipelineFlag(
                    "GROUNDING",
                    "info",
                    "RAG_LINKS_OK",
                    f"RAG linked {linked}/{billable} billable entities to clinic inventory/services.",
                )
            )
        if ratio >= 0.6 and linked == 0:
            flags.append(
                PipelineFlag(
                    "GROUNDING",
                    "warning",
                    "HIGH_UNLINKED_RATE",
                    f"RAG left {need_sku}/{billable} billable entities without a SKU — vet should pick from suggestions.",
                    f"unlinked_ratio={ratio:.2f}; with_suggestions={with_suggestions}",
                )
            )
            stages["GROUNDING"] = "degraded"
        elif ratio >= 0.35:
            flags.append(
                PipelineFlag(
                    "GROUNDING",
                    "info" if with_suggestions >= max(1, need_sku // 2) else "warning",
                    "PARTIAL_UNLINKED",
                    f"RAG linked {linked}/{billable}; {need_sku} need SKU selection ({with_suggestions} have suggestions).",
                )
            )
            if with_suggestions < max(1, need_sku // 2):
                stages["GROUNDING"] = "degraded"
        if dual_reject:
            flags.append(
                PipelineFlag(
                    "GROUNDING",
                    "info",
                    "DUAL_SYNC_REJECTS",
                    f"{dual_reject} Dual-Sync candidate(s) rejected (safety: prevented wrong SKU / global-only hallucination).",
                )
            )

    if not manifest and (cleaned or raw):
        if len(brain_ents) > 0:
            # Brain NER ran but grounding never wrote a manifest — Master Doc quality regression
            flags.append(
                PipelineFlag(
                    "GROUNDING",
                    "error",
                    "GROUNDING_PIPELINE_SKIPPED",
                    f"Brain NER found {len(brain_ents)} entities but entity_manifest is empty — "
                    "local RAG/inventory linking did not run (SKIP_BILLING_PIPELINE, missing grounding "
                    "LLM client, or Postgres skip). Quality below Master Doc dual_sync path.",
                )
            )
            stages["GROUNDING"] = "failing"
        else:
            flags.append(
                PipelineFlag(
                    "GROUNDING",
                    "warning",
                    "EMPTY_MANIFEST",
                    "Entity manifest is empty — RAG / inventory linking did not produce entities.",
                )
            )
            stages["GROUNDING"] = "degraded"

    # --- STEP3 SOAP ---
    soap = soap_json
    if soap is None:
        soap = _load_json(_latest(out, "soap_note_*.json")) or {}
    if not isinstance(soap, dict):
        soap = {}

    for key, label in (
        ("Plan", "Plan"),
        ("KeyIssues", "Key Issues"),
        ("AbnormalFindings", "Abnormal Findings"),
        ("CustomerInstructions", "Customer Instructions"),
    ):
        val = (soap.get(key) or "").strip() if isinstance(soap.get(key), str) else ""
        if not val:
            flags.append(
                PipelineFlag(
                    "STEP3",
                    "warning",
                    f"EMPTY_{key.upper()}",
                    f"STEP3 SOAP section '{label}' is empty.",
                )
            )
            stages["STEP3"] = "degraded"

    plan = soap.get("Plan") or ""
    if isinstance(plan, str) and plan.strip() and "\n" not in plan and re.search(r"\d+\.\s+\S.+\d+\.\s+", plan):
        flags.append(
            PipelineFlag(
                "STEP3",
                "info",
                "PLAN_INLINE_NUMBERING",
                "Plan used inline numbering; formatter should expand to one item per line.",
            )
        )

    if not soap or not any(soap.get(k) for k in ("Subjective", "Objective", "Assessment", "Plan")):
        flags.append(
            PipelineFlag(
                "STEP3",
                "error",
                "SOAP_INCOMPLETE",
                "STEP3 SOAP note missing core Subjective/Objective/Assessment/Plan content.",
            )
        )
        stages["STEP3"] = "failing"

    # --- PHASE2 ---
    atoms_payload = knowledge_atoms
    if atoms_payload is None:
        atoms_payload = _load_json(_latest(out, "knowledge_atoms_*.json"))
    atoms_list: List[Any] = []
    if isinstance(atoms_payload, dict):
        atoms_list = atoms_payload.get("knowledge_atoms") or []
        if atoms_payload.get("error"):
            flags.append(
                PipelineFlag(
                    "PHASE2",
                    "error",
                    "PHASE2_ERROR",
                    f"Phase 2 failed: {atoms_payload.get('error')}",
                )
            )
            stages["PHASE2"] = "failing"
    elif isinstance(atoms_payload, list):
        atoms_list = atoms_payload

    dash = _load_json(_latest(out, "verification_dashboard_*.json")) or {}
    unlinked_mod = []
    if isinstance(dash, dict):
        unlinked_mod = dash.get("module_4_unlinked_entities") or []
    if isinstance(unlinked_mod, list) and unlinked_mod:
        with_sug = sum(1 for u in unlinked_mod if isinstance(u, dict) and (u.get("suggestions") or []))
        linked_meds = len((dash.get("medications_prescribed") or [])) + len(
            (dash.get("medications_administered") or [])
        )
        linked_svc = len(dash.get("procedures_services") or []) + len(dash.get("diagnostics") or [])
        if linked_meds + linked_svc > 0:
            flags.append(
                PipelineFlag(
                    "PHASE2",
                    "info",
                    "PHASE2_DASHBOARD_PARTIAL",
                    f"Phase 2 dashboard has {linked_meds + linked_svc} linked billing row(s); "
                    f"{len(unlinked_mod)} still need SKU pick ({with_sug} with suggestions).",
                )
            )
            if with_sug == 0 and len(unlinked_mod) >= 3:
                stages["PHASE2"] = "degraded"
        else:
            sev = "info" if with_sug >= max(1, len(unlinked_mod) // 2) else "warning"
            flags.append(
                PipelineFlag(
                    "PHASE2",
                    sev,
                    "BILLING_ACTION_REQUIRED",
                    f"Phase 2: {len(unlinked_mod)} item(s) need clinic SKU selection "
                    f"({with_sug} already have suggestions for the vet).",
                    ", ".join(
                        str((u or {}).get("item_name") or "")
                        for u in unlinked_mod[:5]
                        if isinstance(u, dict)
                    ),
                )
            )
            if sev == "warning":
                stages["PHASE2"] = "degraded"
            else:
                # Actionable with suggestions = Phase 2 working as designed
                stages["PHASE2"] = "ok" if stages["PHASE2"] == "ok" else stages["PHASE2"]

    if isinstance(atoms_list, list) and atoms_list:
        flags.append(
            PipelineFlag(
                "PHASE2",
                "info",
                "PHASE2_ATOMS_OK",
                f"Phase 2 extracted {len(atoms_list)} knowledge atom(s).",
            )
        )
    elif not atoms_list and stages["STEP3"] != "failing":
        if stages["GROUNDING"] == "failing":
            flags.append(
                PipelineFlag(
                    "PHASE2",
                    "warning",
                    "PHASE2_SKIPPED_NO_MANIFEST",
                    "Phase 2 skipped because entity_manifest was empty (depends on grounding).",
                )
            )
            stages["PHASE2"] = "degraded"
        elif not dash:
            flags.append(
                PipelineFlag(
                    "PHASE2",
                    "info",
                    "PHASE2_NO_ATOMS_YET",
                    "Phase 2 knowledge atoms not found in run folder (may still be writing or skipped).",
                )
            )

    # overall
    if any(v == "failing" for v in stages.values()) or any(f.severity == "error" for f in flags):
        overall = "failing"
    elif any(v == "degraded" for v in stages.values()) or any(f.severity == "warning" for f in flags):
        overall = "degraded"
    else:
        overall = "healthy"
        flags.append(
            PipelineFlag(
                "SYSTEM",
                "info",
                "WORKFLOW_HEALTHY",
                "No major STEP1–Phase2 issues flagged for this run.",
            )
        )

    return PipelineHealthReport(
        overall=overall,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        flags=flags,
        stage_summary=stages,
    )


def save_pipeline_flags(
    output_dir: str | Path,
    report: PipelineHealthReport,
    *,
    timestamp: Optional[str] = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"pipeline_flags_{ts}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    # Also write a stable latest pointer for the UI
    latest = out / "pipeline_flags_latest.json"
    latest.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_and_save_pipeline_flags(
    output_dir: str | Path,
    *,
    timestamp: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    report = build_pipeline_flags(output_dir, **kwargs)
    save_pipeline_flags(output_dir, report, timestamp=timestamp)
    return report.to_dict()

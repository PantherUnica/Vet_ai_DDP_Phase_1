"""Primary / secondary diagnosis helpers for SOAP generation and display."""

from __future__ import annotations

from typing import Any, Dict

DIAGNOSIS_SECTION_GUIDANCE = """
DIFFERENTIAL DIAGNOSIS — two sub-fields under this heading in the UI:

PRIMARY DIAGNOSIS (mandatory, may be MULTIPLE):
- Root/main condition(s) driving this consultation (the real problem(s) causing the visit).
- When the vet says "X secondary to Y", Y belongs here (not X).
- **FORMAT (mandatory):** Numbered list, one diagnosis per line:
  1. Atopic dermatitis (Dermatology-Atopic Dermatitis)
  2. Hip dysplasia (Orthopedic-Hip Dysplasia)
- Each line: Plain clinical name (System-Condition)
- If only one condition, use a single numbered line.
- If nothing is clearly supported: 1. Unknown-Not specified (Unknown-Not specified)

SECONDARY DIAGNOSIS (optional, may be MULTIPLE):
- Contributing or complicating condition(s) supporting the primary problem(s).
- Use when the vet states "secondary to", "associated with", comorbidity, or overlay infection on chronic issue.
- CRITICAL: "X secondary to Y" → Y in PrimaryDiagnosis, X in SecondaryDiagnosis.
- **FORMAT:** Same numbered-list format as Primary (or empty string "" if none).
- Do NOT invent secondary diagnoses.
- Source only from Assessment, Plan, conversation, or Brain NER.
"""


def migrate_legacy_diagnosis(soap: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy DifferentialDiagnosis to PrimaryDiagnosis when needed."""
    out = dict(soap or {})
    primary = (out.get("PrimaryDiagnosis") or "").strip()
    if not primary:
        legacy = (out.get("DifferentialDiagnosis") or "").strip()
        if legacy:
            out["PrimaryDiagnosis"] = legacy
    if "SecondaryDiagnosis" not in out or out["SecondaryDiagnosis"] is None:
        out["SecondaryDiagnosis"] = ""
    return out


def diagnosis_text_for_storage(soap: Dict[str, Any]) -> str:
    """Combine primary/secondary for DB columns that store a single diagnosis string."""
    soap = migrate_legacy_diagnosis(soap)
    primary = (soap.get("PrimaryDiagnosis") or "").strip()
    secondary = (soap.get("SecondaryDiagnosis") or "").strip()
    if primary and secondary:
        return f"{primary}; Secondary: {secondary}"
    return primary or secondary

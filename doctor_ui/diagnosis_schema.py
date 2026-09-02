"""Primary / secondary diagnosis helpers for SOAP generation and display."""

from __future__ import annotations

from typing import Any, Dict

DIAGNOSIS_SECTION_GUIDANCE = """
PRIMARY DIAGNOSIS (mandatory):
- The main underlying condition driving this consultation (the real/root problem).
- When the vet says "X secondary to Y", Y belongs here (not X).
- Format: Plain clinical name (System-Condition)
- Example: Atopic dermatitis (Dermatology-Atopic Dermatitis)
- If only one condition is discussed, put it here and leave SecondaryDiagnosis empty.
- If nothing is clearly supported, use: Unknown-Not specified (Unknown-Not specified)

SECONDARY DIAGNOSIS (optional):
- A contributing or complicating condition that supports or explains the primary problem.
- Use when the vet states "secondary to", "associated with", comorbidity, or an overlying infection on top of a chronic issue.
- CRITICAL: If the vet says "X secondary to Y", then Y is PrimaryDiagnosis (underlying/root) and X is SecondaryDiagnosis (overlay/complication).
- Same format as primary: Plain clinical name (System-Condition)
- Use empty string "" if no secondary condition is explicitly stated.
- Do NOT invent a secondary diagnosis.
- Source only from Assessment, Plan, conversation, or Brain NER — do not add new conditions.
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

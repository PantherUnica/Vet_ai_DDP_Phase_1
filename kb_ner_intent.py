"""
Clinical Intent: pass-through + grounding for the soap_notes_phase_1 pipeline.

This module lives in the same project (soap_notes_phase_1). The default run uses:
- Batch intent from Super-Pass when search_term is present (no per-entity call).
- When batch intent is missing, per-entity intent runs here: pass-through (mention -> query)
  plus ground_clinical_terms() at the call site for ostectomy->osteotomy etc.

No LLM call in this module by default, so the pipeline runs without errors or external API.
"""

import os
from typing import Any, Dict, List, Optional

# Model name used when a client is requested for intent (e.g. run_pipeline_to_intent).
# Same project default as Super-Pass so get_client_for_model works if needed later.
CLINICAL_INTENT_MODEL = os.getenv("CLINICAL_INTENT_MODEL", "gpt-4.1-nano").strip()

# Kind -> family for batch intent and category guard (matches Super-Pass).
# PRODUCT | PROCEDURE | CLINICAL | OTHER
FAMILY_MAP: Dict[str, str] = {
    "Drug": "PRODUCT",
    "Supplement": "PRODUCT",
    "Nutrition": "PRODUCT",
    "Vaccine": "PRODUCT",
    "Substance": "PRODUCT",
    "Device": "PRODUCT",
    "Procedure": "PROCEDURE",
    "DiagnosticTest": "PROCEDURE",
    "LabTest": "PROCEDURE",
    "Service": "PROCEDURE",
    "Condition": "CLINICAL",
    "Finding": "CLINICAL",
    "Observation": "CLINICAL",
    "Symptom": "CLINICAL",
    "VitalSign": "OTHER",
    "ReasonForVisit": "OTHER",
    "Anatomy": "OTHER",
    "Organism": "OTHER",
    "Species": "OTHER",
    "Breed": "OTHER",
    "Toxin": "OTHER",
    "Other": "OTHER",
}


def should_trigger_intent_resolution(normalized_name: str, canonical_kind: str) -> bool:
    """
    Whether to run per-entity intent when batch intent (search_term) is missing.
    True for kinds that typically need ASR correction or LCD mapping; intent then
    uses pass-through + ground_clinical_terms at call site.
    """
    if not (normalized_name or "").strip():
        return False
    trigger_kinds = {
        "Procedure", "DiagnosticTest", "LabTest", "Service",
        "Condition", "Finding", "Observation", "Symptom", "ReasonForVisit",
        "Drug", "Nutrition", "Vaccine", "Device",
    }
    return canonical_kind in trigger_kinds


def resolve_clinical_intent(
    mention: str,
    anchors: List[Dict[str, Any]],
    ner_kind: str,
    client: Optional[Any] = None,
    logger: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve clinical intent for one entity. Default: pass-through (mention -> query).
    Caller must run ground_clinical_terms(intent_result) after this for ostectomy->osteotomy etc.
    Returns None only if the mention should be dropped as non-clinical; otherwise returns
    a dict with at least "query", "reasoning", "category".
    """
    mention = (mention or "").strip()
    if not mention:
        return None
    # Pass-through: preserve observation level; grounding corrections applied by caller.
    return {
        "query": mention,
        "reasoning": "pass-through (default intent)",
        "category": ner_kind,
    }

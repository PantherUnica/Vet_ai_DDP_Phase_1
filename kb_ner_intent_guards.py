"""
Clinical Specificity Guard: Medical grounding corrections and multi-intent helpers.

- CLINICAL_GROUNDING_CORRECTIONS: Fix common ASR/LLM spelling/suffix errors (e.g. ostectomy -> osteotomy).
- ground_clinical_terms(intent_result): Apply corrections to intent result (query, reasoning).
  Clinical Register (Lexical Tier): Do not normalize clinical observations into colloquialisms.
  We never replace with blocked colloquial display terms (e.g. "walking funny"); see BLOCKED_COLLOQUIAL.
- parse_multi_intent_search_term(search_term): Split delimited search_term for downstream billing.
"""

import re
from typing import Dict, Any, List, Optional

# Full-phrase replacements (order matters: longer phrases first).
# When key is found in query (case-insensitive), the entire query is replaced by the value.
CLINICAL_GROUNDING_FULL: Dict[str, str] = {
    "femoral head and neck ostectomy": "Femoral head and neck osteotomy",
    "fho ostectomy": "Femoral head and neck osteotomy",
    "spirocoxin": "Spirocoxib",
    "antoloposteral": "Anteroposterior and Lateral views",
}

# Substring (word) replacements: only the matched token is replaced so the rest of the query is preserved.
# E.g. "FHO ostectomy" -> "FHO osteotomy" (ostectomy = removal of bone; osteotomy = cutting; FHO is osteotomy).
CLINICAL_GROUNDING_SUBSTRING: Dict[str, str] = {
    "ostectomy": "osteotomy",
}

# Clinical Register: Never replace with these colloquial display terms (preserve vet vocabulary).
# E.g. "walking problem" (Clinical) must not become "walking funny" (Colloquial).
BLOCKED_COLLOQUIAL: frozenset = frozenset({
    "walking funny",
})

# Combined view for external use (full-phrase only; substring handled separately in code).
CLINICAL_GROUNDING_CORRECTIONS: Dict[str, str] = {**CLINICAL_GROUNDING_FULL, **CLINICAL_GROUNDING_SUBSTRING}

# Delimiter for multi-intent search_term (e.g. "Anteroposterior view; Lateral view").
MULTI_INTENT_DELIMITER = ";"


def ground_clinical_terms(intent_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Apply medical grounding corrections to an intent result in place.
    Expects intent_result to have at least "query" (search term); optionally "reasoning".
    Corrects query and appends a note to reasoning when a fix is applied.
    Full-phrase matches replace the entire query; substring matches replace only the word (e.g. ostectomy -> osteotomy).

    Clinical Register (Lexical Tier): Do not normalize clinical observations into colloquialisms.
    If a correction would set the query to a blocked colloquial term (e.g. "walking funny"), it is skipped.
    """
    if intent_result is None:
        return None
    query = intent_result.get("query") or ""
    if not isinstance(query, str) or not query.strip():
        return intent_result
    search_lower = query.lower()
    applied: Optional[str] = None
    # Full-phrase replacements first (longer phrases before shorter).
    for error, correction in CLINICAL_GROUNDING_FULL.items():
        if error in search_lower:
            if correction.lower().strip() in BLOCKED_COLLOQUIAL:
                continue  # Preserve clinical register; do not replace with colloquialism
            intent_result["query"] = correction
            applied = f"{error} -> {correction}"
            break
    if applied is None:
        for error, correction in CLINICAL_GROUNDING_SUBSTRING.items():
            if error in search_lower:
                new_query = re.sub(
                    re.escape(error), correction, query, flags=re.IGNORECASE, count=1
                )
                if new_query.lower().strip() in BLOCKED_COLLOQUIAL:
                    continue  # Preserve clinical register
                intent_result["query"] = new_query
                applied = f"{error} -> {correction}"
                break
    if applied:
        if "reasoning" not in intent_result:
            intent_result["reasoning"] = ""
        r = (intent_result.get("reasoning") or "").strip()
        intent_result["reasoning"] = f"{r} (Grounding fix applied: {applied})".strip()
    return intent_result


def parse_multi_intent_search_term(search_term: Optional[str]) -> List[str]:
    """
    Split a search_term that may contain multiple intents (e.g. "Anteroposterior view; Lateral view")
    into a list of terms for downstream billing or multiple KB searches.
    Returns a list of non-empty stripped strings; if search_term is empty or None, returns [].
    """
    if not search_term or not isinstance(search_term, str):
        return []
    parts = [p.strip() for p in search_term.split(MULTI_INTENT_DELIMITER) if p.strip()]
    return parts if parts else [search_term.strip()] if search_term.strip() else []

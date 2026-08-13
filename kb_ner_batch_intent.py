"""
Batch Intent: resolve search_term and family for all entities in one LLM call (Clean-then-Intent).

Used after Super-Pass (cleaning + NER) to assign search_term and family per entity so grounding
uses canonical/ASR-corrected terms (e.g. "ultralining test" -> "Ortolani test") without per-entity
intent calls. Category-Locked Guard and LCD/specificity rules apply.

Model: gpt-4.1-mini (BATCH_INTENT_MODEL); fallback gpt-4o-mini on 404.

Signature: run_batch_intent(cleaned_transcription, initial_entities, client=None, logger=logger)
Returns: list of entities (same order as input) with search_term and family set.

---
NER flow by step (what happens for each NER stage):
  1) Super-Pass (kb_ner_super_pass)
     Input: raw transcript.
     Output: cleaned transcript + entities (span_text, normalized_name, kind; NO search_term/family).
     Model: gpt-4.1-mini (SUPER_PASS_MODEL). Single call or chunked streaming.
  2) Batch Intent (this module)
     Input: cleaned transcript + entities from Super-Pass.
     Output: same entities with search_term and family added; ground_clinical_terms() applied.
     Model: gpt-4.1-mini (BATCH_INTENT_MODEL).
  3) Step 2.3 / Grounding (kb_ner_linker.run_step_2_3_normalization → kb_ner_parallel)
     Input: cleaned transcript, raw transcript, entities (with search_term/family if from Batch Intent).
     Per entity: if search_term present → use it (no per-entity intent); else → resolve_clinical_intent (kb_ner_intent).
     Then: route (skip_vitals / dual_sync / global_direct) → local/global search → Judge → manifest.
  4) Per-entity intent (kb_ner_intent.resolve_clinical_intent)
     Only when search_term is missing (e.g. legacy NER path or chunk without batch intent).
     Model: CLINICAL_INTENT_MODEL.
  5) Legacy extraction (kb_ner_extraction), when NOT using Super-Pass
     extract_entities_from_cleaned_transcript / ner_extract_entities; then Step 2.3 runs without
     pre_extracted_entities and may call per-entity intent for each.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Model: gpt-4.1-nano by default (no fallback).
BATCH_INTENT_MODEL = os.getenv("BATCH_INTENT_MODEL", "gpt-4.1-nano").strip()

VALID_FAMILIES = frozenset({"PRODUCT", "PROCEDURE", "CLINICAL", "OTHER"})

# Max characters of context snippet per entity (so prompt stays bounded).
BATCH_INTENT_CONTEXT_WINDOW = int(os.getenv("BATCH_INTENT_CONTEXT_WINDOW", "120").strip())


def _get_entity_context(cleaned: str, ent: Dict[str, Any], window: int = BATCH_INTENT_CONTEXT_WINDOW) -> str:
    """
    Get a short snippet of transcript around this entity so Batch Intent can disambiguate
    (e.g. "walking" in "walking on sandy surfaces" vs "walking problem").
    Uses start_char/end_char if present, else finds span_text in cleaned.
    """
    if not cleaned or window <= 0:
        return ""
    span = (ent.get("span_text") or "").strip()
    if not span:
        return ""
    start = ent.get("start_char")
    end = ent.get("end_char")
    if start is not None and end is not None and isinstance(start, (int, float)) and isinstance(end, (int, float)):
        start, end = int(start), int(end)
        lo = max(0, start - window)
        hi = min(len(cleaned), end + window)
        return cleaned[lo:hi].strip()
    idx = cleaned.find(span)
    if idx < 0:
        return ""
    lo = max(0, idx - window)
    hi = min(len(cleaned), idx + len(span) + window)
    return cleaned[lo:hi].strip()


def run_batch_intent(
    cleaned_transcription: str,
    initial_entities: List[Dict[str, Any]],
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Resolve search_term and family for each entity from the cleaned transcript in one batch.

    Args:
        cleaned_transcription: Full cleaned transcript (source of truth; no modification).
        initial_entities: List of entity dicts with at least span_text, kind, normalized_name.
        client: Optional OpenAI-compatible client. If None, get_client_for_model(BATCH_INTENT_MODEL) is used.
        logger: Optional logger.

    Returns:
        List of entity dicts (same length and order as initial_entities), each with
        search_term and family added/updated. Preserves all original keys.
    """
    if not initial_entities:
        return []

    from kb_ner_intent import FAMILY_MAP
    from kb_ner_intent_guards import ground_clinical_terms
    from kb_ner_routing import canonicalize_kind

    # Pass-through: no transcript or empty entities
    cleaned = (cleaned_transcription or "").strip()
    out: List[Dict[str, Any]] = []
    for ent in initial_entities:
        ent = dict(ent)
        span = (ent.get("span_text") or "").strip()
        norm = (ent.get("normalized_name") or span or "").strip()
        kind = (ent.get("kind") or "Other").strip()
        can_kind = canonicalize_kind(kind)
        family = FAMILY_MAP.get(can_kind, "OTHER")
        if family not in VALID_FAMILIES:
            family = "OTHER"
        search_term = norm or span
        intent_result = {"query": search_term, "reasoning": "pass-through", "category": can_kind}
        intent_result = ground_clinical_terms(intent_result)
        if intent_result and intent_result.get("query"):
            search_term = intent_result["query"].strip()
        ent["search_term"] = search_term
        ent["family"] = family
        out.append(ent)

    if not cleaned:
        if logger:
            logger.info("  Batch intent: no cleaned transcript; using pass-through (search_term=normalized_name, family from kind)")
        return out

    # Resolve client if not provided
    if client is None:
        try:
            from kb_ner_clients import get_client_for_model
            result = get_client_for_model(BATCH_INTENT_MODEL, logger=logger)
            if isinstance(result, tuple):
                client, _ = result
            else:
                client = result
        except Exception as e:
            if logger:
                logger.warning("  Batch intent: could not get client for %s: %s; using pass-through", BATCH_INTENT_MODEL, e)
            return out

    # Build entity list for prompt: span_text, kind, and context snippet so the model can disambiguate.
    entity_list = []
    for ent in initial_entities:
        span = (ent.get("span_text") or "").strip()
        kind = (ent.get("kind") or "Other").strip()
        if span:
            ctx = _get_entity_context(cleaned, ent, BATCH_INTENT_CONTEXT_WINDOW)
            item = {"span_text": span, "kind": kind}
            if ctx:
                item["context"] = ctx[:200].strip()  # cap per entity so prompt stays bounded
            entity_list.append(item)

    if not entity_list:
        return out

    system_prompt = """You are a veterinary clinical terminology resolver. For each entity mention you must output: (1) the correct word/phrase (search_term) for KB search, and (2) family for category accuracy so Knowledge Atoms can route correctly (e.g. PRODUCT for medications).

Given:
- NER output: span_text and kind for each mention
- The sentence/context where the mention appears (context snippet)

Do the following:

1. ASR correction: Fix misheard words (e.g. "ultralining test" -> "Ortolani test", "noble angle" -> "Norberg angle", "Spirocoxin" -> "Spirocoxib").

2. Linguistic / canonical term: Using the context, determine the correct clinical term. If already correct and unambiguous, copy it. If ambiguous, choose the appropriate term from context. Do NOT over-normalize.

3. Family (Category Accuracy): Output family so downstream steps (e.g. Knowledge Atoms) can use it. When the mention is a medication or product (e.g. Contraway, brand names, drugs), set family PRODUCT so Knowledge Atoms can look for dosage and confirm Medication. Otherwise set family from kind: PRODUCT, PROCEDURE, CLINICAL, or OTHER.

4. Clinical Register (Lexical Tier): Maintain the clinical register of the speaker. Do NOT normalize clinical observations into colloquialisms.
- If the input is "walking problem" (Clinical), do NOT output "walking funny" (Colloquial).
- If the input is "not willing to wake", use the Lowest Common Denominator (e.g. "Reluctance to move") rather than a diagnostic jump (e.g. "Altered consciousness").
This preserves the professional tone of the note and ensures display_name matches the vet's actual vocabulary.

Output a JSON array with one object per entity in the SAME ORDER as the input list. Each object must have:
- span_text: exact same as input (copy)
- kind: exact same as input (copy)
- search_term: the correct word/phrase for KB search (ASR-corrected, linguistically resolved, clinical register preserved)
- family: exactly one of PRODUCT, PROCEDURE, CLINICAL, OTHER (use PRODUCT for medications/products like Contraway)

Output ONLY a valid JSON array, no markdown or explanation. Preserve order and length of the input list."""

    user_prompt = f"""Cleaned transcript (excerpt, first 8000 chars):\n{cleaned[:8000]}\n\nEntity list (for each, output search_term and family; maintain clinical register):\n{json.dumps(entity_list, ensure_ascii=False)}"""

    model_to_use = BATCH_INTENT_MODEL
    response_text = None
    try:
        resp = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        if getattr(resp, "choices", None) and len(resp.choices) > 0:
            msg = getattr(resp.choices[0], "message", None)
            if msg is not None:
                response_text = (getattr(msg, "content", None) or "").strip()
    except Exception as e:
        if logger:
            logger.warning("  Batch intent LLM failed (%s); using pass-through: %s", model_to_use, e)
        return out

    if not response_text:
        return out

    # Parse JSON array
    try:
        # Strip markdown code block if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)
        parsed = json.loads(response_text)
        if not isinstance(parsed, list):
            parsed = [parsed]
    except json.JSONDecodeError as e:
        if logger:
            logger.warning("  Batch intent: JSON parse failed (%s); using pass-through", e)
        return out

    # Build map (span_text, kind) -> (search_term, family); use first occurrence
    intent_map: Dict[tuple, tuple] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        st = (item.get("span_text") or "").strip()
        k = (item.get("kind") or "Other").strip()
        q = (item.get("search_term") or st or "").strip()
        # Family: use LLM output for category accuracy (e.g. Contraway -> PRODUCT); fallback to kind-based
        fam = (item.get("family") or "").strip().upper()
        if fam not in VALID_FAMILIES:
            fam = FAMILY_MAP.get(canonicalize_kind(k), "OTHER")
        if fam not in VALID_FAMILIES:
            fam = "OTHER"
        if (st, k) not in intent_map:
            intent_map[(st, k)] = (q or st, fam)

    # Apply to entities in order; preserve original keys and apply ground_clinical_terms to search_term
    out = []
    for ent in initial_entities:
        ent = dict(ent)
        span = (ent.get("span_text") or "").strip()
        kind = (ent.get("kind") or "Other").strip()
        norm = (ent.get("normalized_name") or span or "").strip()
        can_kind = canonicalize_kind(kind)
        default_family = FAMILY_MAP.get(can_kind, "OTHER")
        if default_family not in VALID_FAMILIES:
            default_family = "OTHER"

        search_term = norm or span
        family = default_family
        key = (span, kind)
        if key in intent_map:
            q, fam = intent_map[key]
            if (q or "").strip():
                search_term = (q or search_term).strip()
            # family is derived from kind (set in intent_map from FAMILY_MAP)
            if fam in VALID_FAMILIES:
                family = fam

        intent_result = {"query": search_term, "reasoning": "Batch intent (single call)", "category": can_kind}
        intent_result = ground_clinical_terms(intent_result)
        if intent_result and intent_result.get("query"):
            search_term = intent_result["query"].strip()

        ent["search_term"] = search_term
        ent["family"] = family
        out.append(ent)

    if logger:
        logger.info("  ✅ Batch intent complete: %s entities (model=%s)", len(out), model_to_use)
        # Optional: log Shadow Intent output per entity (set BATCH_INTENT_DEBUG=1 to see what Batch Intent returned)
        if os.getenv("BATCH_INTENT_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            for ent in out:
                st = (ent.get("span_text") or "").strip()
                k = (ent.get("kind") or "Other").strip()
                q = (ent.get("search_term") or "").strip()
                if st:
                    logger.info("  [Batch Intent] span=%r kind=%r → search_term=%r", st, k, q)
    return out

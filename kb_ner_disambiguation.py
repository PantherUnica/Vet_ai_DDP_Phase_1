"""
LLM judge and disambiguation utilities.

This module handles:
- LLM judge for local match disambiguation
- Generic mention detection
- ASR correction (legacy and candidate-aware)
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import json
import threading
import time
from concurrent.futures import Future

from kb_ner_clients import get_openai_client


# ==============================================================================
# Option C batching (micro-batcher)
# ==============================================================================
#
# Goal: eliminate sequential judge latency by coalescing multiple Option C requests
# into a single LLM call. This is especially helpful when processing entities in
# parallel (kb_ner_parallel) or when many items fall through deterministic gates.
#
# This is a micro-batcher: it waits a short window (default 50ms) to collect
# requests, then flushes one batch call.
#
# Safety: if batch fails, each request is completed with None (conservative).
_BATCH_JUDGE_LOCK = threading.Lock()
_BATCH_JUDGE_PENDING: list[dict] = []
_BATCH_JUDGE_TIMER: Optional[threading.Timer] = None

# Judge model: configurable via LLM_JUDGE_MODEL (default gpt-4.1-nano).
# Rule: GPT/OpenAI models → OpenAI API only; Fireworks models → Fireworks API only.
# We always use the client from get_client_for_model(LLM_JUDGE_MODEL). If that returns None
# (e.g. no OPENAI_API_KEY for gpt-4.1-nano), we do not substitute the pipeline's client—
# we fail with a clear message: set the appropriate API key or set LLM_JUDGE_MODEL to a model
# on the provider you have (e.g. a Fireworks model if you only have FIREWORKS_API_KEY).
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "gpt-4.1-nano").strip()


def _detect_judge_model_from_client(client: Any) -> str:
    # Use configured Judge model so batch path uses correct client
    return LLM_JUDGE_MODEL


# ==============================================================================
# Kind-to-Category Mapping (Semantic Bridge)
# ==============================================================================
#
# Maps Phase 2 entity kinds (Medicine, Procedure, etc.) to database categories
# (Pharmacy, Diagnostics, etc.) to enable role-based filtering.
#
# This is a many-to-one mapping: multiple linguistic terms map to the same
# logical database bucket.

_KIND_TO_CATEGORY_MAP = {
    # Medicine/Drug kinds -> Pharmacy category
    "Medicine": ["Pharmacy", "Drug", "Medication"],
    "Drug": ["Pharmacy", "Drug", "Medication"],
    "Medication": ["Pharmacy", "Drug", "Medication"],
    "Substance": ["Pharmacy", "Drug", "Medication"],
    "Nutrition": ["Pharmacy", "Nutrition", "Supplements"],
    "Vaccine": ["Pharmacy", "Vaccines"],
    
    # Procedure/Service kinds -> Services category
    "Procedure": ["Services", "Procedures", "Treatment"],
    "Service": ["Services", "Procedures", "Treatment"],
    "Treatment": ["Services", "Procedures", "Treatment"],
    
    # Diagnostic kinds -> Diagnostics category only (Lab and Radiology are subcategories, not domains)
    "DiagnosticTest": ["Diagnostics"],
    "LabTest": ["Diagnostics"],
    "Diagnostic": ["Diagnostics"],
    
    # Device kinds -> Devices category
    "Device": ["Devices", "Equipment"],
}


def _normalize_category(category: Optional[str]) -> Optional[str]:
    """Normalize category string for comparison."""
    if not category:
        return None
    return category.strip().lower()


def _kind_matches_category(entity_kind: str, candidate_category: Optional[str]) -> Optional[bool]:
    """
    Check if entity kind matches candidate category using semantic bridge.
    
    Returns:
        True: Kind matches category (e.g., Medicine -> Pharmacy)
        False: Kind conflicts with category (e.g., Medicine -> Diagnostics)
        None: Cannot determine (category not in map, or kind not mapped)
    """
    if not candidate_category:
        return None
    
    normalized_category = _normalize_category(candidate_category)
    if not normalized_category:
        return None
    
    # Get expected categories for this kind
    expected_categories = _KIND_TO_CATEGORY_MAP.get(entity_kind, [])
    if not expected_categories:
        return None  # Kind not mapped
    
    # Check if candidate category matches any expected category
    normalized_expected = [_normalize_category(cat) for cat in expected_categories]
    if normalized_category in normalized_expected:
        return True
    
    # Check for explicit conflicts (common mismatches)
    conflict_patterns = {
        "Medicine": ["diagnostics", "lab", "reagent", "test"],
        "Drug": ["diagnostics", "lab", "reagent", "test"],
        "Medication": ["diagnostics", "lab", "reagent", "test"],
        "Procedure": ["pharmacy", "drug", "medication"],
        "DiagnosticTest": ["pharmacy", "drug", "medication"],
        "LabTest": ["pharmacy", "drug", "medication"],
    }
    
    conflicts = conflict_patterns.get(entity_kind, [])
    for conflict in conflicts:
        if conflict in normalized_category:
            return False
    
    return None  # Ambiguous - let judge decide


def _apply_category_penalty(
    candidates: List[Dict[str, Any]],
    entity_kind: str,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Category-based boosting/penalties are DISABLED.
    
    Returns candidates unchanged - no category-based score adjustments.
    Category matching is informational only and does not affect match scores.
    
    Args:
        candidates: List of candidate dicts
        entity_kind: Entity kind (unused, kept for signature compatibility)
        logger: Optional logger (unused)
        
    Returns:
        List of candidates unchanged (no score modifications)
    """
    # Return candidates as-is - no category-based boosting or penalties
    return candidates


def _build_batch_judge_prompt(batch_items: list[dict]) -> str:
    """
    Build a prosecutorial prompt to judge multiple mentions at once.
    
    This prompt acts as a "prosecutor" - it looks for reasons to REJECT candidates,
    not reasons to ACCEPT them. This prevents hallucinations by prioritizing
    safety over automation.
    
    Each item must contain:
      - req_id, original_mention, search_term_used, entity_kind, context_sentence
      - assessment_context (optional)
      - candidates: list of dicts (already limited / scored, with category info)
    """
    lines: list[str] = []
    lines.append("You are the PROSECUTOR JUDGE for a veterinary entity-linking system.")
    lines.append("Your role is to REJECT candidates that do not match, not to find matches.")
    lines.append("Return JSON only. No prose.")
    lines.append("")
    lines.append("You will be given multiple ITEMS. For each ITEM, select ONE of:")
    lines.append('- a local candidate number as a STRING: "1", "2", ... (ONLY if certain)')
    lines.append('- "NONE_GENERIC" (mention is generic vs specific candidates)')
    lines.append('- "NONE" (no appropriate candidate - DEFAULT if uncertain)')
    lines.append("")
    lines.append('Return JSON with shape: {"decisions":[{"id":"...","choice":"..."}]}')
    lines.append("")
    lines.append("CRITICAL REJECTION RULES (Prosecutor Mode):")
    lines.append("0. CATEGORY INCOMPATIBILITY (Cross-Kind Check): You MUST REJECT any match where ENTITY_KIND is a finding (Symptom, Diagnosis, ReasonForVisit) but the CANDIDATE is a tangible product (Medication, Lab Kit, Consumable, Pharmacy item). A diagnosis or reason-for-visit like 'pus', 'yeast', or 'yeast growth' is a state of being or a finding, not a product you can put in a bag. Example: ENTITY_KIND=Diagnosis, ORIGINAL_MENTION='bus in ear' (ASR for pus), Candidate='Lasix 16 mg' → REJECT (NONE). Example: ENTITY_KIND=ReasonForVisit, ORIGINAL_MENTION='yeast growth', Candidate='AST KIT' (lab reagent) → REJECT (NONE).")
    lines.append("")
    lines.append("1. ROLE-TO-CATEGORY ALIGNMENT: The ENTITY_KIND represents the 'Role' the Vet mentioned (e.g., 'Medicine' = treating the patient).")
    lines.append("   The candidate CATEGORY represents what the item actually is (e.g., 'Diagnostics' = lab reagent).")
    lines.append("   QUESTION: Does the Role mentioned by the Vet align with the Category of the candidate?")
    lines.append("   - If Vet is TREATING the patient (Medicine/Nutrition), REJECT Diagnostic Reagents.")
    lines.append("   - If Vet is TESTING the patient (DiagnosticTest), REJECT Pharmacy items.")
    lines.append("   Example: ENTITY_KIND=Medicine, Candidate='M-18 CF LYSE' (Category: Diagnostics) → REJECT (NONE)")
    lines.append("   Reason: Vet is treating, but candidate is a lab reagent (wrong role).")
    lines.append("")
    lines.append("2. ASSESSMENT-AWARE FILTER: Use ASSESSMENT_CONTEXT to identify clinically absurd matches.")
    lines.append("   The patient is diagnosed with [ASSESSMENT_CONTEXT]. Does the candidate make clinical sense?")
    lines.append("   - Assessment='Anorexia/Infection' + Candidate='Lab Reagent' → REJECT (NONE)")
    lines.append("   - Assessment='Anorexia/Infection' + Candidate='Nutrition Supplement' → CONSIDER (if name matches)")
    lines.append("   Rule: If the Vet is treating the patient, reject Diagnostic Reagents even if name similarity is high.")
    lines.append("")
    lines.append("2.5. SYMPTOM / PHYSICAL FINDING SUPPRESSOR (CRITICAL - Safety):")
    lines.append("   If the span is a Symptom or Physical Finding (e.g. pus, yeast, yeast growth, discharge, shaking, swelling, growth in ear), you MUST REJECT any match to Medications or Lab Kits/Reagents unless the CONTEXT explicitly states the vet is prescribing that product or ordering that test.")
    lines.append("   Symptoms and findings are descriptive; they must NOT be grounded to billable SKUs. Example: 'bus inside the left ear' (ASR for 'pus') + Candidate='Lasix 16 mg' → REJECT (NONE). Example: 'yeast growth' + Candidate='AST KIT' (lab reagent) → REJECT (NONE). When in doubt for symptom/finding mentions, choose NONE.")
    lines.append("")
    lines.append("3. GENERIC MENTIONS: If mention is generic ('tablet', 'injection') but candidates are specific, REJECT (NONE_GENERIC).")
    lines.append("5. WHEN IN DOUBT: Choose NONE. Do not guess or 'help' by selecting a candidate.")
    lines.append("")
    lines.append("ACCEPTANCE CRITERIA (Only accept if ALL are true):")
    lines.append("- Candidate name matches mention (exact or strong synonym)")
    lines.append("- ENTITY_KIND matches candidate CATEGORY (Medicine→Pharmacy, Procedure→Services)")
    lines.append("- Assessment context supports the match (if provided)")
    lines.append("- Mention is specific enough (not generic)")
    lines.append("")
    lines.append("EXPANSION BRIDGE (phonetic / ASR correction):")
    lines.append("BRAIN_HINTS and QUERY_EXPANSIONS are the clinical engine's best-guess alternatives for the mention (e.g. 'Expense exotic pump' → hints like 'Easotic', 'ear drops'; query_expansions like 'Easotic', 'Easotic 10ml').")
    lines.append("If a LOCAL candidate's name matches a term in QUERY_EXPANSIONS, that indicates a high-confidence phonetic correction. You should FAVOR this match even if ORIGINAL_MENTION looks mangled.")
    lines.append("If a candidate matches a QUERY_EXPANSION term, treat it as strong evidence toward ACCEPTANCE (do not reject solely due to lexical mismatch with ORIGINAL_MENTION).")
    lines.append("")
    lines.append("4. FORM-FACTOR & ROUTE-TO-FORM ALIGNMENT (CRITICAL):")
    lines.append("Match the ROUTE and FORM of the vet's mention to the candidate's formulation. Use administration cues and unit cues from ORIGINAL_MENTION and CONTEXT.")
    lines.append("")
    lines.append("Liquid indicators (ORAL): If the mention includes ml, cc, syrup, suspension, oral solution, drops, liquid → you MUST prioritize candidates that are liquid formulations (syrup, suspension, drops, solution). REJECT solid (tablet/capsule) candidates even if they have a higher match score.")
    lines.append("Solid indicators: If the mention includes mg (without ml), tablet, tab, capsule, pill, cap, bolus → you MUST prioritize solid candidates. REJECT oral liquids if the vet clearly indicated solid form.")
    lines.append("Injectable alignment: If the mention includes inject, injection, IM, IV, SC, Sub-Q, vial, ampoule → you MUST prioritize injectable candidates. REJECT oral syrups/suspensions and topical formulations.")
    lines.append("Topical/external alignment: If the mention includes apply, topical, cream, ointment, spray, pump, drops (in ear/eye context) → you MUST prioritize external/topical formulations. REJECT systemic oral or injectable when the vet clearly indicated topical route.")
    lines.append("")
    lines.append("CONFLICT RESOLUTION: If the vet mentions a liquid unit (e.g. '3 ml') or the word 'syrup'/'suspension', and the top-scoring candidate is a tablet while a liquid candidate exists in the list, REJECT the tablet and SELECT the liquid candidate. Same logic in reverse for solid vs liquid, and for injectable vs oral vs topical. The candidate's physical form and route MUST align with the veterinarian's intended delivery.")
    lines.append("")
    lines.append("=== ITEMS ===")

    for item in batch_items:
        req_id = item["req_id"]
        orig = item.get("original_mention", "")
        st = item.get("search_term_used", "")
        ek = item.get("entity_kind", "")
        ctx = item.get("context_sentence", "")
        assess = (item.get("assessment_context") or "").strip()
        cands = item.get("candidates") or []

        cand_lines: list[str] = []
        for i, cand in enumerate(cands[:8], 1):
            display_name = cand.get("display_name", "") or cand.get("preferred_name", "")
            stock_id = cand.get("stock_id")
            service_id = cand.get("service_id")
            category = cand.get("category", "Unknown")
            description = cand.get("description", "") or cand.get("definition", "")  # Local description or KB definition
            trigram_score = float(cand.get("trigram_score", 0.0) or 0.0)
            phonetic_score = float(cand.get("phonetic_score", 0.0) or 0.0)
            vector_score = float(cand.get("vector_score", 0.0) or 0.0)
            match_score = float(cand.get("match_score", 0.0) or 0.0)
            original_score = cand.get("original_match_score")
            score_info = f"match={match_score:.3f}"
            if original_score and abs(original_score - match_score) > 0.01:
                score_info += f" (original={original_score:.3f})"
            score_info += f", trigram={trigram_score:.3f}, phonetic={phonetic_score:.1f}, vector={vector_score:.3f}"
            
            # Include category prominently for prosecutor evaluation
            category_info = f"Category: {category}"
            desc_info = f"\n   Description: {description}" if description else ""
            if stock_id:
                cand_lines.append(f'{i}. {display_name} (stock_id: {stock_id}, {category_info}, {score_info}){desc_info}')
            elif service_id:
                cand_lines.append(f'{i}. {display_name} (service_id: {service_id}, {category_info}, {score_info}){desc_info}')
            else:
                cand_lines.append(f"{i}. {display_name} ({category_info}, {score_info}){desc_info}")

        hints_list = item.get("hints") or []
        hints_str = ", ".join(str(h) for h in hints_list[:5])
        qe_list = item.get("query_expansion") or []
        qe_str = ", ".join(str(x) for x in qe_list[:5])
        lines.append("")
        lines.append(f"ITEM_ID: {req_id}")
        lines.append(f'ORIGINAL_MENTION: "{orig}"')
        lines.append(f'SEARCH_TERM_USED: "{st}"')
        lines.append(f"ENTITY_KIND: {ek}")
        lines.append(f"BRAIN_HINTS: [{hints_str}]")
        lines.append(f"QUERY_EXPANSIONS: [{qe_str}]")
        lines.append(f'CONTEXT: "{ctx}"')
        if assess:
            lines.append(f'ASSESSMENT_CONTEXT: "{assess}"')
        else:
            lines.append('ASSESSMENT_CONTEXT: ""')
        lines.append("LOCAL_CANDIDATES:")
        lines.extend(cand_lines if cand_lines else ["(none)"])
        lines.append("")
        lines.append("EVALUATION (for each candidate):")
        lines.append("0. CATEGORY INCOMPATIBILITY: Is ENTITY_KIND a finding (Symptom, Diagnosis, ReasonForVisit) and candidate a product (Medication, Lab Kit, Consumable)? If yes → REJECT (NONE).")
        lines.append("1. ROLE CHECK: Does the ENTITY_KIND (Role) align with candidate CATEGORY?")
        lines.append("   - Medicine/Nutrition Role + Diagnostics Category → REJECT")
        lines.append("   - DiagnosticTest Role + Pharmacy Category → REJECT")
        lines.append("2. ASSESSMENT CHECK: Given ASSESSMENT_CONTEXT, does this candidate make clinical sense?")
        lines.append("   - If Vet is treating (Medicine) but candidate is a lab reagent → REJECT")
        lines.append("   - If Vet is testing (DiagnosticTest) but candidate is a medication → REJECT")
        lines.append("2.5. SYMPTOM/FINDING CHECK: Is ORIGINAL_MENTION a symptom or physical finding (pus, yeast, discharge, growth, shaking, swelling)? If yes and candidate is Medication or Lab Reagent → REJECT (NONE).")
        lines.append("3. FORM-FACTOR / ROUTE CHECK: Does the candidate's form match the vet's mention?")
        lines.append("   - Mention has syrup/ml/drops/suspension but candidate is tablet/capsule → REJECT; choose a liquid candidate if present.")
        lines.append("   - Mention has tablet/tab/mg (solid) but candidate is syrup/suspension → REJECT; choose a solid candidate if present.")
        lines.append("   - Mention has inject/vial/IM/IV but candidate is oral syrup → REJECT; choose an injectable candidate if present.")
        lines.append("   - Mention has apply/spray/pump/cream/drops (topical) but candidate is oral/injectable → REJECT; choose a topical candidate if present.")
        lines.append("4. GENERIC CHECK: Is the mention generic? If yes, consider NONE_GENERIC.")
        lines.append("5. CERTAINTY CHECK: Only select a candidate number if you are CERTAIN it matches.")
        lines.append("   - If Role doesn't align with Category → REJECT (NONE)")
        lines.append("   - If Assessment contradicts candidate role → REJECT (NONE)")
        lines.append("   - If form/route does not align (see step 3 above) → REJECT or pick the aligned candidate.")

    lines.append("")
    lines.append("=== OUTPUT ===")
    lines.append('Return ONLY JSON: {"decisions":[{"id":"ITEM_ID","choice":"1|2|...|NONE|NONE_GENERIC"}, ...]}')
    lines.append("Remember: When in doubt, choose NONE. Do not guess.")
    return "\n".join(lines)


def _flush_batch_judge():
    global _BATCH_JUDGE_TIMER
    with _BATCH_JUDGE_LOCK:
        batch = _BATCH_JUDGE_PENDING[:]
        _BATCH_JUDGE_PENDING.clear()
        _BATCH_JUDGE_TIMER = None

    if not batch:
        return

    # All requests in a batch share the same client (by design in caller)
    client = batch[0]["client"]
    logger = batch[0].get("logger")

    # Safety: if any item has a different client, we conservatively complete them with None.
    for it in batch:
        if it.get("client") is not client:
            try:
                it["future"].set_result(None)
            except Exception:
                pass
    batch = [it for it in batch if it.get("client") is client]
    if not batch:
        return

    prompt = _build_batch_judge_prompt(batch)
    model = _detect_judge_model_from_client(client)
    # Judge model may be OpenAI (e.g. gpt-4.1-mini); use correct client for that model
    judge_client = client
    try:
        from kb_ner_clients import get_client_for_model
        resolved_client, _ = get_client_for_model(model)
        if resolved_client:
            judge_client = resolved_client
    except Exception:
        pass

    raw_text = ""
    try:
        resp = judge_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return JSON only. No prose."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        raw_text = (resp.choices[0].message.content or "").strip()
        
        # Extract JSON from response (handle cases where LLM adds prose)
        obj = {}
        if raw_text:
            # Try direct parse first
            try:
                obj = json.loads(raw_text)
            except json.JSONDecodeError:
                # Try to extract JSON object from response (may have prose before/after)
                json_start = raw_text.find("{")
                json_end = raw_text.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    try:
                        extracted_json = raw_text[json_start:json_end + 1]
                        obj = json.loads(extracted_json)
                    except json.JSONDecodeError:
                        # Last resort: try to repair common issues
                        try:
                            import re
                            repaired = re.sub(r",\s*([}\]])", r"\1", raw_text[json_start:json_end + 1])
                            obj = json.loads(repaired)
                        except Exception:
                            if logger:
                                logger.debug(f"  ⚠️  Could not parse batch judge JSON. Raw text: {raw_text[:500]}")
                            obj = {}
        
        decisions = obj.get("decisions") if isinstance(obj, dict) else None
        if not isinstance(decisions, list):
            decisions = []
        decision_map: dict[str, str] = {}
        for d in decisions:
            if not isinstance(d, dict):
                continue
            did = str(d.get("id") or "").strip()
            choice = str(d.get("choice") or "").strip().upper()
            if did:
                decision_map[did] = choice

        # Fulfill futures
        for it in batch:
            req_id = it["req_id"]
            choice = (decision_map.get(req_id) or "").upper()
            candidates = it.get("candidates") or []
            selected = None
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(candidates):
                    selected = candidates[idx - 1]
            # NONE / NONE_GENERIC => None (conservative)
            try:
                it["future"].set_result(selected)
            except Exception:
                pass

    except Exception as e:
        if logger:
            logger.warning(f"  ⚠️  Batch LLM judge failed; falling back to None for batch (err={e})")
            if raw_text:
                logger.debug(f"  📝 Batch LLM judge raw response (first 800 chars): {raw_text[:800]}")
            else:
                logger.debug(f"  📝 Batch LLM judge returned empty response")
        for it in batch:
            try:
                it["future"].set_result(None)
            except Exception:
                pass


def run_single_batch_llm_judge(
    *,
    batch_items: List[Dict[str, Any]],
    client: Any,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Execute ONE explicit Option-C batch judge call for all unresolved entities.

    Returns:
      Dict[req_id -> selected_candidate_or_None]
    """
    if not batch_items or not client:
        return {}

    normalized_items: List[Dict[str, Any]] = []
    for it in batch_items:
        req_id = str(it.get("req_id") or "").strip()
        if not req_id:
            continue
        penalized_candidates = _apply_category_penalty(
            list(it.get("candidates") or []),
            str(it.get("entity_kind") or ""),
            logger,
        )
        normalized_items.append(
            {
                "req_id": req_id,
                "original_mention": it.get("original_mention") or "",
                "search_term_used": it.get("search_term_used") or "",
                "candidates": penalized_candidates,
                "entity_kind": it.get("entity_kind") or "",
                "context_sentence": it.get("context_sentence") or "",
                "assessment_context": it.get("assessment_context"),
                "hints": it.get("hints") or [],
                "query_expansion": it.get("query_expansion") or [],
            }
        )

    if not normalized_items:
        return {}

    prompt = _build_batch_judge_prompt(normalized_items)
    model = _detect_judge_model_from_client(client)
    judge_client = None
    try:
        from kb_ner_clients import get_client_for_model

        resolved_client, _ = get_client_for_model(model, logger=logger)
        judge_client = resolved_client
    except Exception as e:
        if logger:
            logger.warning("  ⚖️ Batch judge: could not resolve client for model %s: %s", model, e)

    if not judge_client:
        if logger:
            logger.error(
                "  ⚖️ LLM Judge skipped: model '%s' requires the matching API. "
                "Set OPENAI_API_KEY for GPT models or set LLM_JUDGE_MODEL to a Fireworks model (e.g. SUPER_PASS_MODEL) to use Fireworks API.",
                model,
            )
        return {}

    if logger:
        logger.info("  ⚖️ Single-shot batch judge: %s unresolved entities (one LLM call)", len(normalized_items))

    raw_text = ""
    decision_map: Dict[str, str] = {}
    messages = [
        {"role": "system", "content": "Return JSON only. No prose."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = judge_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
        )
        raw_text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        if logger:
            logger.warning("  ⚠️ Batch LLM judge failed; falling back to None for batch (err=%s)", e)

    if raw_text:
        obj: Dict[str, Any] = {}
        try:
            obj = json.loads(raw_text)
        except json.JSONDecodeError:
            json_start = raw_text.find("{")
            json_end = raw_text.rfind("}")
            if json_start >= 0 and json_end > json_start:
                try:
                    obj = json.loads(raw_text[json_start : json_end + 1])
                except Exception:
                    obj = {}
        decisions = obj.get("decisions") if isinstance(obj, dict) else None
        if isinstance(decisions, list):
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                did = str(d.get("id") or "").strip()
                choice = str(d.get("choice") or "").strip().upper()
                if did:
                    decision_map[did] = choice

    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for it in normalized_items:
        req_id = it["req_id"]
        choice = (decision_map.get(req_id) or "").upper()
        selected = None
        cands = it.get("candidates") or []
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(cands):
                selected = cands[idx - 1]
        out[req_id] = selected
    return out


def _submit_to_batch_judge(
    *,
    req_id: str,
    original_mention: str,
    search_term_used: str,
    candidates: List[Dict[str, Any]],
    entity_kind: str,
    context_sentence: str,
    assessment_context: Optional[str],
    client: Any,
    logger: Optional[logging.Logger],
    batch_window_ms: int = 50,
    hints: Optional[List[str]] = None,
    query_expansion: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Submit a judge request to the micro-batcher and block until decision returns.
    
    Note: Category-based boosting/penalties are disabled - candidates are passed through unchanged.
    """
    # Category-based boosting disabled - pass candidates through unchanged
    penalized_candidates = _apply_category_penalty(candidates, entity_kind, logger)
    
    fut: Future = Future()
    item = {
        "req_id": req_id,
        "original_mention": original_mention,
        "search_term_used": search_term_used,
        "candidates": penalized_candidates,  # Use penalized candidates
        "entity_kind": entity_kind,
        "context_sentence": context_sentence,
        "assessment_context": assessment_context,
        "hints": (hints or [])[:5],
        "query_expansion": (query_expansion or [])[:5],
        "client": client,
        "logger": logger,
        "future": fut,
    }
    with _BATCH_JUDGE_LOCK:
        _BATCH_JUDGE_PENDING.append(item)
        global _BATCH_JUDGE_TIMER
        if _BATCH_JUDGE_TIMER is None:
            _BATCH_JUDGE_TIMER = threading.Timer(batch_window_ms / 1000.0, _flush_batch_judge)
            _BATCH_JUDGE_TIMER.daemon = True
            _BATCH_JUDGE_TIMER.start()
    try:
        # Block until batch flush completes.
        return fut.result(timeout=30)
    except Exception:
        return None


def is_generic_mention(mention: str, entity_kind: str) -> bool:
    """
    Classify if a mention is GENERIC (broad class/form with no identifying anchor) or SPECIFIC.
    
    GENERIC if it is only a broad class/form with no identifying anchor:
    - "examination", "test", "injection", "capsules", "tablets", "medicine", "vaccine", etc.
    
    SPECIFIC if it includes ANY identifying anchor:
    - A brand/trade token (even if spelled differently due to ASR/phonetics)
    - A specific procedure/test name ("anal gland expression", etc.)
    - Strength/pack markers ("500mg", "60S", "1 ml", etc.)
    - A distinctive multi-word phrase beyond generic form words
    
    Args:
        mention: The original mention from transcript
        entity_kind: Entity kind (Drug, Procedure, etc.)
    
    Returns:
        True if generic, False if specific
    """
    mention_lower = mention.lower().strip()
    
    # Remove [unclear] tag for analysis
    mention_lower = mention_lower.replace("[unclear]", "").strip()
    
    # Generic patterns (broad class/form words only)
    generic_patterns = [
        r'^(examination|exam|test|injection|shot|vaccine|medicine|medication|drug|tablet|tablets|capsule|capsules|pill|pills|liquid|drops|cream|ointment|spray|powder|syrup|solution|suspension|inhaler|patch|suppository)$',
        r'^(blood\s+test|urine\s+test|ultrasound|scan|imaging)$',
        r'^(surgery|procedure|treatment|therapy|consultation|checkup|visit)$',
    ]
    
    # Check if mention matches generic patterns exactly
    for pattern in generic_patterns:
        if re.match(pattern, mention_lower):
            return True
    
    # Check if mention is ONLY generic words (no anchors)
    generic_words = {
        'examination', 'exam', 'test', 'injection', 'shot', 'vaccine', 'medicine', 
        'medication', 'drug', 'tablet', 'tablets', 'capsule', 'capsules', 'pill', 
        'pills', 'liquid', 'drops', 'cream', 'ointment', 'spray', 'powder', 
        'syrup', 'solution', 'suspension', 'inhaler', 'patch', 'suppository',
        'blood', 'urine', 'ultrasound', 'scan', 'imaging', 'surgery',
        'procedure', 'treatment', 'therapy', 'consultation', 'checkup', 'visit'
    }
    
    words = mention_lower.split()
    if len(words) == 1 and words[0] in generic_words:
        return True
    
    # Check if all words are generic (no anchors)
    if all(word in generic_words for word in words):
        return True
    
    # SPECIFIC indicators (presence of any makes it specific)
    specific_indicators = [
        # Brand/trade name patterns (common veterinary brands)
        r'\b(cortex|coatex|nutrish|nutrich|simparica|simparico|bravecto|nexgard|heartgard|frontline|advantage|revolution|sentinel|interceptor|trifexis|comfortis|capstar|seresto|k9|advantix|vanguard|nobivac|felocell|purevax|rabvac|rabies|dhpp|dhlpp|fvr|fvrcp|felv|fiv|bordetella|lepto|lyme|anaplasma|ehrlichia|giardia|panleuk|calici|herpes|chlamydia|pneumonia|distemper|parvo|adenovirus|hepatitis|corona|parainfluenza|bordetella|kennel\s+cough)\b',
        # Strength/pack markers
        r'\b\d+\s*(mg|ml|g|kg|lb|oz|tablets?|capsules?|pills?|doses?|vials?|bottles?|pack|packet|box|s|count)\b',
        r'\b\d+[sm]\b',  # e.g., "60S", "500mg"
        # Specific procedure names
        r'\b(anal\s+gland|gland\s+expression|dental\s+cleaning|teeth\s+cleaning|spay|neuter|castration|ovariohysterectomy|declaw|microchip|ear\s+cleaning|nail\s+trim|grooming|bath|vaccination|deworming|flea\s+treatment|tick\s+treatment|heartworm\s+test|fecal\s+exam|urinalysis|blood\s+work|biopsy|surgery|suture|stitch|wound\s+care|bandage|splint|cast)\b',
        # Multi-word phrases (likely specific)
        r'\b\w+\s+\w+\s+\w+',  # 3+ words usually indicates specificity
    ]
    
    for pattern in specific_indicators:
        if re.search(pattern, mention_lower):
            return False  # Has specific anchor
    
    # If mention has 2+ words and doesn't match generic patterns, likely specific
    if len(words) >= 2:
        return False
    
    # Default: if unclear, assume generic (conservative)
    return True


def refine_term_with_llm_candidate_aware(
    text: str,
    category: str,
    candidates: List[Dict[str, Any]],
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    FIX D: Candidate-aware ASR correction.
    
    Instead of open-ended correction, asks LLM if the mention is a noisy version
    of one of the candidates. This prevents wild corrections like "General Physical Fitness → Generalized Seizure Disorder".
    
    Args:
        text: The potentially misrecognized term
        category: Entity kind
        candidates: List of candidate matches (top 5)
        client: OpenAI-compatible client
        logger: Logger instance
    
    Returns:
        Corrected text if it matches a candidate, or original text if none match
    """
    if not client or not candidates:
        return text
    
    # Build candidate list for LLM
    candidate_names = [c.get("display_name", "") for c in candidates[:5]]
    
    prompt = f"""ROLE: Veterinary ASR Error Corrector (Candidate-Aware).

TASK: Is the mention "{text}" a noisy/ASR-error version of one of these candidates?

CANDIDATES:
{chr(10).join(f"{i+1}. {name}" for i, name in enumerate(candidate_names))}

INSTRUCTIONS:
1. Check if "{text}" sounds like or is a phonetic error for any candidate
2. Examples:
   - "animal plant expression" → "ANAL SAC EXPRESSION GROOMING" (phonetic match)
   - "cortex" → "COATEX BLISTER CAPSULE" (phonetic match)
   - "nutrish" → "NUTRICH TABLET 60S" (phonetic match)
3. If the mention matches a candidate phonetically, return that candidate's name
4. If NONE of the candidates match, return "NONE"

OUTPUT: Return ONLY the candidate name (if match) or "NONE" (if no match). No explanations."""
    
    try:
        # Use configured Judge model (LLM_JUDGE_MODEL)
        model = LLM_JUDGE_MODEL
        try:
            from kb_ner_clients import get_client_for_model
            judge_client, _ = get_client_for_model(model)
            if judge_client:
                client = judge_client
        except Exception:
            pass
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a veterinary ASR error corrector. Return only a candidate name or NONE."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=50,
        )
        
        corrected = response.choices[0].message.content.strip().strip('"').strip("'")
        
        # Check if correction matches any candidate
        corrected_lower = corrected.lower()
        for cand in candidates:
            cand_name = cand.get("display_name", "").lower()
            if corrected_lower in cand_name or cand_name in corrected_lower:
                if logger:
                    logger.info(f"  ✅ Candidate-aware ASR: '{text}' → '{cand.get('display_name')}'")
                return cand.get("display_name", text)
        
        # If LLM said "NONE" or correction doesn't match, return original
        if "none" in corrected_lower:
            if logger:
                logger.debug(f"  ℹ️  Candidate-aware ASR: '{text}' matches no candidates - keeping original")
            return text
        
        # If correction doesn't match any candidate, return original (prevent wild corrections)
        if logger:
            logger.warning(f"  ⚠️  Candidate-aware ASR: '{corrected}' doesn't match any candidate - keeping original '{text}'")
        return text
        
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠️  Candidate-aware ASR correction failed: {e}")
        return text


def refine_term_with_llm(
    text: str,
    category: str,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    model: Optional[str] = None,
    prevent_hallucination: bool = True,
    context: Optional[str] = None,
) -> str:
    """
    PERMANENT FIX: Context-Aware ASR Correction.
    
    Uses actual transcript context (if provided) or wraps the term in a 'Clinical Anchor Sentence' 
    to force the LLM to see the context (Reason vs Diet vs Anatomy).
    
    This handles cases where semantic distance fails (e.g., "Animal Plant Expression" 
    vs "Anal Gland Expression") because vector embeddings match by meaning, not sound.
    
    Args:
        text: The potentially misrecognized term
        category: Entity kind (Drug, Procedure, Finding, Reason, Diet, etc.)
        client: OpenAI-compatible client
        logger: Logger instance
        model: Model to use for correction (auto-detected if None)
        prevent_hallucination: Whether to prevent hallucinated corrections
        context: Optional actual transcript context around the term (preferred over simulated sentence)
    
    Returns:
        Corrected text string, or original text if no correction needed/found
    """
    if not client:
        if logger:
            logger.debug("No LLM client available for ASR correction")
        return text
    
    # Auto-detect model based on client base_url
    if model is None:
        base_url = ''
        try:
            if hasattr(client, 'base_url'):
                base_url = str(client.base_url) if client.base_url else ''
            elif hasattr(client, '_client') and hasattr(client._client, 'base_url'):
                base_url = str(client._client.base_url) if client._client.base_url else ''
            elif hasattr(client, '_base_url'):
                base_url = str(client._base_url) if client._base_url else ''
        except Exception:
            pass
        
        if not base_url:
            base_url = os.getenv("FIREWORKS_BASE_URL", "")
        
        model = LLM_JUDGE_MODEL
        try:
            from kb_ner_clients import get_client_for_model
            judge_client, _ = get_client_for_model(model)
            if judge_client:
                client = judge_client
        except Exception:
            pass
        if logger:
            logger.debug(f"Using judge model: {model}")
    
    # Define Clinical Anchors (Category-specific sentence templates)
    anchors = {
        "Reason": f"The patient presented for {text}.",
        "Reason_for_Visit": f"The patient presented for {text}.",
        "ReasonForVisit": f"The patient presented for {text}.",
        "Diet": f"The patient is currently eating {text}.",
        "Nutrition": f"The patient is currently eating {text}.",
        "Drug": f"I administered {text} to the patient.",
        "Substance": f"I administered {text} to the patient.",
        "Procedure": f"We performed a {text} on the patient.",
        "Service": f"We performed a {text} on the patient.",
        "Finding": f"On examination, I found {text}.",
        "Anatomy": f"The problem is located at the {text}.",
        "Location": f"The lesion is on the {text}.",
        "Condition": f"The patient has {text}.",
        "Observation": f"I observed {text} during the examination.",
        "Device": f"We used {text} during the procedure.",
        "Vaccine": f"We administered {text} to the patient.",
        "DiagnosticTest": f"We performed a {text} on the patient.",
        "LabTest": f"We performed a {text} on the patient.",
        "Other": f"The clinical term is {text}.",
    }
    
    anchor_sentence = anchors.get(category, f"The clinical term is {text}.")
    
    # Use transcript summary as additional context if provided
    # This provides broader context without overwhelming the prompt
    if context and context.strip():
        # Context is a summary or excerpt of the clean transcript
        # Combine anchor sentence with transcript summary for best accuracy
        context_sentence = f"{anchor_sentence}\n\nTRANSCRIPT CONTEXT: {context[:300]}"  # Limit to 300 chars to keep prompt focused
        if logger:
            logger.debug(f"  📝 Using anchor sentence + transcript summary for ASR correction (summary length: {len(context)} chars)")
    else:
        # Fallback: Use anchor sentence only
        context_sentence = anchor_sentence
        if logger:
            logger.debug(f"  📝 Using clinical anchor sentence only for ASR correction (category: {category})")
    
    try:
        prompt = f"""ROLE: Veterinary ASR Error Corrector.

TASK: The Speech-to-Text system made a phonetic error. 
You must correct the term "{text}" using the anchor sentence and transcript context below.

ANCHOR SENTENCE: "{anchor_sentence}"
TRANSCRIPT CONTEXT: "{context if context and context.strip() else 'N/A - use anchor sentence only'}"
TARGET CATEGORY: {category}

INSTRUCTIONS:
1. The anchor sentence shows the category-specific context (e.g., "presented for" = ReasonForVisit).
2. The transcript context provides broader conversation context.
3. What veterinary term sounds like "{text}" phonetically and fits BOTH the anchor sentence AND transcript context?
4. Examples:
   - "animal plant expression" in anchor "presented for..." + context "came for animal plant expression" → "anal gland expression"
   - "royal cannon" in anchor "eating..." + context "feed royal cannon GI diet" → "Royal Canin"
   - "cortex" in anchor "administered..." + context "give cortex capsules" → "Coatex"
5. Use BOTH the anchor sentence (category context) AND transcript context (conversation context) to understand what was meant.
6. Return ONLY the corrected clinical term (max 1 phrase). Do not return the full sentence. Do not add explanations."""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a veterinary medical terminology specialist. Correct ASR transcription errors to proper medical terms. Output only the corrected term, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=50,
        )
        
        corrected_term = response.choices[0].message.content.strip()
        corrected_term = corrected_term.strip('"').strip("'").strip().rstrip(".")
        
        # PREVENT HALLUCINATION: Validate correction is semantically related
        if prevent_hallucination and corrected_term.lower() != text.lower():
            clinical_categories = ['Drug', 'Procedure', 'Finding', 'Substance', 'Reason', 'ReasonForVisit']
            if category in clinical_categories:
                if logger:
                    logger.debug(f"  ✅ Trusting LLM correction for clinical category '{category}': '{text}' -> '{corrected_term}'")
                return corrected_term
            else:
                # For non-clinical categories, use stricter validation
                original_words = set(text.lower().split())
                corrected_words = set(corrected_term.lower().split())
                
                if not original_words.intersection(corrected_words):
                    char_similarity = SequenceMatcher(None, text.lower(), corrected_term.lower()).ratio()
                    
                    if char_similarity < 0.3:
                        if logger:
                            logger.warning(f"  ⚠️  ASR correction rejected (hallucination): '{text}' -> '{corrected_term}' (no word overlap, char similarity: {char_similarity:.2f})")
                        return text
                
                common_clinical_words = {"normal", "lymph", "nodes", "gland", "expression", "plant", "animal"}
                if any(word in original_words for word in common_clinical_words):
                    if not any(word in corrected_term.lower() for word in original_words if word in common_clinical_words):
                        if logger:
                            logger.warning(f"  ⚠️  ASR correction rejected (context mismatch): '{text}' -> '{corrected_term}' (lost context words)")
                        return text
        
        # Logging the "Magic" - Context Anchor Fix
        if logger:
            if corrected_term.lower() != text.lower():
                logger.info(f"🧠 Context Anchor Fix: '{text}' -> '{corrected_term}' (Context: {category}, Anchor: '{simulated_sentence}')")
            else:
                logger.debug(f"ASR Correction: '{text}' unchanged (no correction needed)")
        
        return corrected_term
        
    except Exception as e:
        if logger:
            logger.warning(f"ASR correction failed for '{text}': {e}")
        return text


# NOTE: is_generic_mention is already defined at the top of this file (line 19).
# The duplicate definition that was here has been removed to avoid confusion.


def _effective_auto_bind_threshold(entity_kind: str, auto_bind_threshold: Optional[float] = None) -> float:
    """Use lower threshold for Medication kinds (0.88) for higher auto-link rate; default 0.92 otherwise."""
    base = 0.92 if auto_bind_threshold is None else float(auto_bind_threshold)
    if entity_kind in ("Drug", "Medicine", "Medication"):
        try:
            return float(os.getenv("MEDICATION_AUTO_BIND_THRESHOLD", "0.88"))
        except Exception:
            return 0.88
    return base


def apply_decision_flow(
    mention: str,
    search_term_used: str,
    local_candidates: List[Dict[str, Any]],
    entity_kind: str,
    context_sentence: str,
    assessment_context: Optional[str] = None,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    global_candidates: Optional[List[Dict[str, Any]]] = None,
    auto_bind_threshold: Optional[float] = None,
    # NEW: Contextual information for clinical decision-making
    species: Optional[str] = None,
    breed: Optional[str] = None,
    suspected_condition: Optional[str] = None,
    section: Optional[str] = None,
    # Hints and query_expansion passed to Judge prompt (BRAIN_HINTS / QUERY_EXPANSIONS for phonetic bridge)
    hints: Optional[List[str]] = None,
    query_expansion: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Final decision flow: Option A (3/3 agreement) → Option B (deterministic gates) → Option C (LLM judge).
    
    Args:
        mention: Original mention from transcript (may include [unclear])
        search_term_used: What the retrieval system searched with (may be ASR-corrected)
        local_candidates: List of local candidates with match scores
        entity_kind: Entity kind (Drug, Procedure, etc.)
        context_sentence: Short transcript context
        client: OpenAI-compatible client
        logger: Optional logger
        global_candidates: Optional global KB candidates (for context only)
    
    Returns:
        Selected candidate dict, or None if rejected
    """
    if not local_candidates:
        if logger:
            logger.debug(f"  ⚠️  Decision Flow: No local candidates for '{mention}' - cannot bind billable item")
        return None
    
    # Step 0: Classify the mention
    is_generic = is_generic_mention(mention, entity_kind)
    if logger:
        logger.debug(f"  🔍 Decision Flow: '{mention}' classified as {'GENERIC' if is_generic else 'SPECIFIC'}")
    
    # Extract scores from candidates
    # Each candidate should have: trigram_score, phonetic_score, vector_score, match_score
    candidates_with_scores = []
    for cand in local_candidates[:10]:  # Top 10
        trigram = cand.get("trigram_score", 0.0)
        phonetic = cand.get("phonetic_score", 0.0)
        vector = cand.get("vector_score", 0.0)
        match_score = cand.get("match_score", 0.0)
        
        # If match_score exists but individual scores don't, use match_score as proxy
        if match_score > 0 and trigram == 0 and phonetic == 0 and vector == 0:
            # Assume match_score is from trigram (most common)
            trigram = match_score
        
        candidates_with_scores.append({
            **cand,
            "trigram_score": float(trigram) if trigram else 0.0,
            "phonetic_score": float(phonetic) if phonetic else 0.0,
            "vector_score": float(vector) if vector else 0.0,
            "match_score": float(match_score) if match_score else 0.0,
        })
    
    if not candidates_with_scores:
        return None
    
    # Check if vector matching is enabled (any candidate has non-zero vector_score)
    vector_enabled = any(c.get("vector_score", 0.0) > 0.0 for c in candidates_with_scores)
    
    # LOG INDIVIDUAL SCORES FOR TOP CANDIDATES
    if logger:
        logger.info(f"   📊 MATCHING SCORES BREAKDOWN (Top 5 candidates):")
        logger.info(f"      Vector matching: {'✅ ENABLED' if vector_enabled else '❌ DISABLED'}")
        for i, cand in enumerate(candidates_with_scores[:5], 1):
            display_name = cand.get("display_name", "N/A")
            stock_id = cand.get("stock_id") or cand.get("service_id", "N/A")
            trigram = cand.get("trigram_score", 0.0)
            phonetic = cand.get("phonetic_score", 0.0)
            vector = cand.get("vector_score", 0.0)
            match = cand.get("match_score", 0.0)
            
            # Determine which method contributed to match_score
            contributing_methods = []
            if match == trigram and trigram > 0:
                contributing_methods.append("TRIGRAM")
            if match == phonetic and phonetic > 0:
                contributing_methods.append("PHONETIC")
            if match == vector and vector > 0:
                contributing_methods.append("VECTOR")
            if not contributing_methods:
                contributing_methods.append("MAX")
            
            method_str = " + ".join(contributing_methods) if contributing_methods else "N/A"
            
            logger.info(f"      {i}. {display_name} (ID: {stock_id})")
            logger.info(f"         Trigram: {trigram:.3f} | Phonetic: {phonetic:.1f} | Vector: {vector:.3f} | Match: {match:.3f} ({method_str})")
    
    # Find top candidates by each method (needed for Option A and B2)
    top_trigram = max(candidates_with_scores, key=lambda x: x.get("trigram_score", 0.0))
    top_phonetic = max(candidates_with_scores, key=lambda x: (x.get("phonetic_score", 0.0), x.get("match_score", 0.0)))
    top_trigram_id = top_trigram.get("stock_id") or top_trigram.get("service_id")
    top_phonetic_id = top_phonetic.get("stock_id") or top_phonetic.get("service_id")
    
    top_vector = None
    top_vector_id = None
    if vector_enabled:
        top_vector = max(candidates_with_scores, key=lambda x: x.get("vector_score", 0.0))
        top_vector_id = top_vector.get("stock_id") or top_vector.get("service_id")
    
    # LOG TOP CANDIDATES BY EACH METHOD
    if logger:
        logger.info(f"   🎯 TOP CANDIDATES BY METHOD:")
        logger.info(f"      Top TRIGRAM: {top_trigram.get('display_name', 'N/A')} (ID: {top_trigram_id}, score: {top_trigram.get('trigram_score', 0.0):.3f})")
        logger.info(f"      Top PHONETIC: {top_phonetic.get('display_name', 'N/A')} (ID: {top_phonetic_id}, score: {top_phonetic.get('phonetic_score', 0.0):.1f})")
        if vector_enabled and top_vector:
            logger.info(f"      Top VECTOR: {top_vector.get('display_name', 'N/A')} (ID: {top_vector_id}, score: {top_vector.get('vector_score', 0.0):.3f})")
        else:
            logger.info(f"      Top VECTOR: N/A (vector matching disabled)")
    
    # Option A: 3/3 agreement (or 2/2 if vector not enabled)
    # Hard floor: even if methods agree, never auto-accept low-signal matches.
    # This prevents cases like "chicken broth" auto-binding to an unrelated SKU just because all weak methods picked the same wrong ID.
    option_a_min_score = 0.70
    try:
        option_a_min_score = float(os.getenv("KB_OPTION_A_MIN_SCORE", "0.70"))
    except Exception:
        option_a_min_score = 0.70

    if not is_generic:
        if vector_enabled:
            # 3/3 agreement
            if top_trigram_id and top_trigram_id == top_phonetic_id == top_vector_id:
                trigram_s = float(top_trigram.get("trigram_score", 0.0) or 0.0)
                vector_s = float((top_vector or {}).get("vector_score", 0.0) or 0.0)
                # Require at least one "signal" score (trigram or vector) above the hard floor.
                # Phonetic is often a coarse score; do not use it as the primary auto-accept floor.
                if max(trigram_s, vector_s) < option_a_min_score:
                    if logger:
                        logger.info(
                            f"   ❌ OPTION A (3/3 AGREEMENT): FAILED (hard floor) "
                            f"(max(trigram={trigram_s:.3f}, vector={vector_s:.3f}) < {option_a_min_score:.2f})"
                        )
                    # fall through to deterministic gates / LLM judge
                else:
                    if logger:
                        logger.info(f"   ✅ OPTION A (3/3 AGREEMENT): Auto-accept")
                        logger.info(f"      All three methods agree on: '{top_trigram.get('display_name')}'")
                        logger.info(f"      Trigram: {top_trigram.get('trigram_score', 0.0):.3f}, Phonetic: {top_phonetic.get('phonetic_score', 0.0):.1f}, Vector: {top_vector.get('vector_score', 0.0):.3f}")
                    return top_trigram
            else:
                if logger:
                    logger.info(f"   ❌ OPTION A (3/3 AGREEMENT): FAILED")
                    logger.info(f"      Trigram ID: {top_trigram_id}, Phonetic ID: {top_phonetic_id}, Vector ID: {top_vector_id}")
                    logger.info(f"      IDs do not match - cannot auto-accept")
        else:
            # 2/2 agreement (trigram + phonetic)
            if top_trigram_id and top_trigram_id == top_phonetic_id:
                trigram_s = float(top_trigram.get("trigram_score", 0.0) or 0.0)
                if trigram_s < option_a_min_score:
                    if logger:
                        logger.info(
                            f"   ❌ OPTION A (2/2 AGREEMENT): FAILED (hard floor) "
                            f"(trigram={trigram_s:.3f} < {option_a_min_score:.2f})"
                        )
                    # fall through to deterministic gates / LLM judge
                else:
                    if logger:
                        logger.info(f"   ✅ OPTION A (2/2 AGREEMENT): Auto-accept")
                        logger.info(f"      Trigram and Phonetic agree on: '{top_trigram.get('display_name')}'")
                        logger.info(f"      Trigram: {top_trigram.get('trigram_score', 0.0):.3f}, Phonetic: {top_phonetic.get('phonetic_score', 0.0):.1f}")
                    return top_trigram
            else:
                if logger:
                    logger.info(f"   ❌ OPTION A (2/2 AGREEMENT): FAILED")
                    logger.info(f"      Trigram ID: {top_trigram_id}, Phonetic ID: {top_phonetic_id}")
                    logger.info(f"      IDs do not match - cannot auto-accept")
    
    # Option B: Deterministic gates
    top_candidate = candidates_with_scores[0]  # Already sorted by match_score
    top_score = top_candidate.get("match_score", 0.0)
    
    second_candidate = candidates_with_scores[1] if len(candidates_with_scores) > 1 else None
    second_score = second_candidate.get("match_score", 0.0) if second_candidate else 0.0
    
    threshold_b1 = _effective_auto_bind_threshold(entity_kind, auto_bind_threshold)
    
    # Embedding-based domain relevance (scalable: no keyword lists)
    # Candidate or session affinity to domain (e.g. orthopedic) => use 0.80 threshold; LLM Judge remains final arbiter
    is_domain_relevant_phonetic = False
    top_phonetic_score = top_candidate.get("phonetic_score", 0.0)
    top_trigram_score = top_candidate.get("trigram_score", 0.0)
    candidate_name = (top_candidate.get("display_name", "") or top_candidate.get("preferred_name", "") or "")
    
    if candidate_name and client:
        try:
            from kb_domain_affinity import is_domain_relevant_for_phonetic_threshold
            is_domain_relevant_phonetic = is_domain_relevant_for_phonetic_threshold(
                candidate_name=candidate_name,
                suspected_condition=suspected_condition,
                domain_key="orthopedic",
                client=client,
                logger=logger,
                candidate_threshold=0.80,  # Orthopedic gate 0.80 so "ultralining" → Ortolani auto-confirms
                session_threshold=0.75,
                candidate_threshold_with_session=0.80,
            )
        except Exception as e:
            if logger:
                logger.debug(f"Domain affinity check skipped: {e}")
    
    # Lower threshold for domain-relevant phonetic match (0.80); LLM Judge still used for verification.
    if is_domain_relevant_phonetic and top_phonetic_score > 0.15 and top_trigram_score < 0.3:
        # Domain-relevant phonetic match: use 0.80 so Ortolani auto-links (e.g. ultralining → Ortolani)
        threshold_b1 = min(threshold_b1, 0.80)
        if logger:
            logger.info(f"   🎯 Domain-relevant phonetic match detected: '{top_candidate.get('display_name')}' (phonetic: {top_phonetic_score:.3f}, trigram: {top_trigram_score:.3f})")
            logger.info(f"      Lowering threshold from {_effective_auto_bind_threshold(entity_kind, auto_bind_threshold):.2f} to {threshold_b1:.2f}")
    
    if logger:
        logger.info(f"   🔍 OPTION B (DETERMINISTIC GATES):")
        logger.info(f"      Top candidate: '{top_candidate.get('display_name')}' (score: {top_score:.3f})")
        if second_candidate:
            logger.info(f"      Second candidate: '{second_candidate.get('display_name')}' (score: {second_score:.3f})")
            logger.info(f"      Margin: {top_score - second_score:.3f}")
    
    # Rule B1: Score auto-accept (>= threshold; Medication kinds use 0.88; orthopedic domain-relevant use 0.80)
    # Domain-relevant phonetic matches use 0.80 so Ortolani auto-links (e.g. ultralining → Ortolani)
    if not is_generic and top_score >= threshold_b1:
        if logger:
            logger.info(f"   ✅ OPTION B1 (score >= {threshold_b1:.2f}): Auto-accept")
            logger.info(f"      '{top_candidate.get('display_name')}' (score: {top_score:.3f} >= {threshold_b1:.2f})")
        return top_candidate
    else:
        if logger:
            if is_generic:
                logger.info(f"   ❌ OPTION B1: SKIPPED (mention is generic)")
            else:
                logger.info(f"   ❌ OPTION B1: FAILED (score {top_score:.3f} < {threshold_b1:.2f})")
    
    # Rule B2: 2-of-3 agreement + margin (must involve trigram or phonetic)
    if not is_generic:
        top_id = top_candidate.get("stock_id") or top_candidate.get("service_id")
        
        # Check for 2-of-3 agreement
        agreements = []
        if top_trigram_id == top_id:
            agreements.append("trigram")
        if top_phonetic_id == top_id:
            agreements.append("phonetic")
        if vector_enabled and top_vector_id == top_id:
            agreements.append("vector")
        
        # Must have 2+ agreements and include trigram or phonetic
        if len(agreements) >= 2 and ("trigram" in agreements or "phonetic" in agreements):
            margin = top_score - second_score
            if margin >= 0.08:
                # Conservative "ASR brand confusion" gate:
                # If TRIGRAM + VECTOR agree and the top candidate is dominant, allow a slightly lower
                # hard floor. This captures stable near-miss brand spellings (e.g., cortex↔coatex)
                # without re-introducing brittle ASR rewrite rules.
                brand_gate_trigram_min = 0.62
                brand_gate_vector_min = 0.30
                brand_gate_margin_min = 0.15
                try:
                    brand_gate_trigram_min = float(os.getenv("KB_BRAND_GATE_TRIGRAM_MIN", "0.62"))
                    brand_gate_vector_min = float(os.getenv("KB_BRAND_GATE_VECTOR_MIN", "0.30"))
                    brand_gate_margin_min = float(os.getenv("KB_BRAND_GATE_MARGIN_MIN", "0.15"))
                except Exception:
                    pass

                def _form_overlap(a: str, b: str) -> bool:
                    a_l = (a or "").lower()
                    b_l = (b or "").lower()
                    forms = ["capsule", "capsules", "tablet", "tablets", "tab", "tabs", "syrup", "inj", "injection", "drops"]
                    return any(f in a_l for f in forms) and any(f in b_l for f in forms) and any(
                        (("capsule" in a_l or "capsules" in a_l) and ("capsule" in b_l or "capsules" in b_l))
                        or (("tablet" in a_l or "tablets" in a_l or "tab" in a_l) and ("tablet" in b_l or "tablets" in b_l or "tab" in b_l))
                        or (("syrup" in a_l) and ("syrup" in b_l))
                        or (("inj" in a_l or "injection" in a_l) and ("inj" in b_l or "injection" in b_l))
                        or (("drops" in a_l) and ("drops" in b_l))
                        for _ in [0]
                    )

                # Hard floor (same spirit as Option A): never auto-accept based on phonetic alone.
                # Require at least one "signal" score (trigram or vector) above the floor.
                # This prevents phonetic-similarity from auto-binding to the wrong SKU.
                trigram_s = float(top_candidate.get("trigram_score", 0.0) or 0.0)
                vector_s = float(top_candidate.get("vector_score", 0.0) or 0.0)
                signal_s = max(trigram_s, vector_s)
                if (
                    "trigram" in agreements
                    and "vector" in agreements
                    and trigram_s >= brand_gate_trigram_min
                    and vector_s >= brand_gate_vector_min
                    and margin >= brand_gate_margin_min
                    and _form_overlap(mention, top_candidate.get("display_name", "") or "")
                    and entity_kind in ("Drug", "Medicine", "Nutrition", "Supplement", "Vaccine")
                ):
                    if logger:
                        logger.info(
                            "   ✅ BRAND-GATE (trigram+vector dominance): Auto-accept despite hard floor "
                            f"(trigram={trigram_s:.3f}>= {brand_gate_trigram_min:.2f}, "
                            f"vector={vector_s:.3f}>= {brand_gate_vector_min:.2f}, "
                            f"margin={margin:.3f}>= {brand_gate_margin_min:.2f})"
                        )
                    return top_candidate
                if signal_s < option_a_min_score:
                    if logger:
                        logger.info(
                            f"   ❌ OPTION B2: FAILED (hard floor) "
                            f"(max(trigram={trigram_s:.3f}, vector={vector_s:.3f}) < {option_a_min_score:.2f})"
                        )
                else:
                    if logger:
                        logger.info(f"   ✅ OPTION B2 (2-of-3 agreement + margin): Auto-accept")
                        logger.info(
                            f"      '{top_candidate.get('display_name')}' "
                            f"(agreements: {', '.join(agreements)}, margin: {margin:.3f} >= 0.08, score: {top_score:.3f} >= {option_a_min_score:.2f})"
                        )
                    return top_candidate
            else:
                if logger:
                    logger.info(f"   ❌ OPTION B2: FAILED (margin {margin:.3f} < 0.08)")
        else:
            if logger:
                logger.info(f"   ❌ OPTION B2: FAILED (agreements: {', '.join(agreements) if agreements else 'none'}, need 2+ with trigram/phonetic)")
    
    # Rule B3: Dominance + specific mention + margin (same threshold as B1)
    if not is_generic and top_score >= threshold_b1 and (top_score - second_score) >= 0.08:
        if logger:
            logger.info(f"   ✅ OPTION B3 (dominance + margin): Auto-accept")
            logger.info(f"      '{top_candidate.get('display_name')}' (score: {top_score:.3f} >= {threshold_b1:.2f}, margin: {top_score - second_score:.3f} >= 0.08)")
        return top_candidate
    else:
        if logger:
            if is_generic:
                logger.info(f"   ❌ OPTION B3: SKIPPED (mention is generic)")
            elif top_score < threshold_b1:
                logger.info(f"   ❌ OPTION B3: FAILED (score {top_score:.3f} < {threshold_b1:.2f})")
            else:
                logger.info(f"   ❌ OPTION B3: FAILED (margin {top_score - second_score:.3f} < 0.08)")
    
    # Predictive grounding fallback: skip LLM Judge when both trigram and vector are high (saves 3–5s per call).
    # Adaptive: default 0.75/0.75 for latency; set KB_PREDICTIVE_STRICT=true or KB_PREDICTIVE_TRIGRAM_MIN=0.8 for stricter.
    try:
        strict = os.getenv("KB_PREDICTIVE_STRICT", "false").strip().lower() in ("1", "true", "yes")
        default_min = 0.8 if strict else 0.75
        predictive_trigram_min = float(os.getenv("KB_PREDICTIVE_TRIGRAM_MIN", str(default_min)))
        predictive_vector_min = float(os.getenv("KB_PREDICTIVE_VECTOR_MIN", str(default_min)))
    except Exception:
        predictive_trigram_min = 0.75
        predictive_vector_min = 0.75
    top_trigram_s = float(top_candidate.get("trigram_score", 0.0) or 0.0)
    top_vector_s = float(top_candidate.get("vector_score", 0.0) or 0.0)

    # Known medical abbreviations / short terms: bypass judge when vector is high (e.g. FHO, physiotherapy, X-ray).
    _KNOWN_ABBREV_OR_SHORT = frozenset(
        {"fho", "physiotherapy", "physio", "xray", "x-ray", "x ray", "bhrt", "cbc", "ua", "dhm", "ecg", "usg", "mri", "ct", "iv", "im", "po", "sid", "bid", "tid", "qid"}
    )
    mention_norm = (mention or "").strip().lower().replace("-", " ").replace("  ", " ")
    mention_tokens = set(mention_norm.split())
    is_known_abbrev_or_short = (
        mention_norm in _KNOWN_ABBREV_OR_SHORT
        or (len(mention_norm) <= 20 and mention_tokens & _KNOWN_ABBREV_OR_SHORT)
        or (len(mention_norm) <= 4 and top_vector_s >= 0.75)
    )
    if not is_generic and is_known_abbrev_or_short and top_vector_s >= 0.75:
        if logger:
            logger.info(
                f"   ✅ PREDICTIVE FALLBACK (known abbrev/short + vector={top_vector_s:.3f}>=0.75): "
                f"Skip LLM Judge, accept '{top_candidate.get('display_name')}'"
            )
        return top_candidate

    if not is_generic and top_trigram_s >= predictive_trigram_min and top_vector_s >= predictive_vector_min:
        if logger:
            logger.info(
                f"   ✅ PREDICTIVE FALLBACK (trigram={top_trigram_s:.3f}>= {predictive_trigram_min}, vector={top_vector_s:.3f}>= {predictive_vector_min}): "
                f"Skip LLM Judge, accept '{top_candidate.get('display_name')}'"
            )
        return top_candidate

    # Option C: Call LLM judge only if A + B + predictive fallback fail
    if logger:
        logger.info(f"   🧠 OPTION C (LLM JUDGE): All deterministic gates failed - calling LLM judge")
    if is_generic:
        # Check if all candidates are specific/branded
        all_specific = all(
            not is_generic_mention(c.get("display_name", ""), entity_kind)
            for c in candidates_with_scores[:5]  # Check top 5
        )
        if all_specific:
            if logger:
                logger.info(f"  🚫 Option C (pre-judge): Generic mention '{mention}' with all specific candidates → NONE_GENERIC")
            return None
    
    # Call LLM judge with new prompt
    return disambiguate_local_match_v2(
        original_mention=mention,
        search_term_used=search_term_used,
        candidates=candidates_with_scores,
        entity_kind=entity_kind,
        context_sentence=context_sentence,
        assessment_context=assessment_context,
        client=client,
        logger=logger,
        global_candidates=global_candidates,
        species=species,
        breed=breed,
        suspected_condition=suspected_condition,
        section=section,
        hints=(hints or [])[:5],
        query_expansion=(query_expansion or [])[:5],
    )


def apply_decision_flow_deterministic_only(
    *,
    mention: str,
    candidates: List[Dict[str, Any]],
    entity_kind: str,
    logger: Optional[logging.Logger] = None,
    auto_bind_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Deterministic-only version of apply_decision_flow: Option A + Option B gates only.
    - No LLM judge (Option C) => **zero network latency**
    - Works for global KB candidates too (uses concept_id as ID when stock/service IDs are absent)

    Expected candidate fields (best-effort; missing fields default to 0):
      - trigram_score, phonetic_score, vector_score, match_score
      - display_name/preferred_name, stock_id/service_id/concept_id
    """
    if not candidates:
        return None

    def _cand_id(c: Dict[str, Any]) -> Any:
        return c.get("stock_id") or c.get("service_id") or c.get("concept_id") or c.get("kb_concept_id")

    # Step 0: generic vs specific
    is_generic = is_generic_mention(mention, entity_kind)

    # Normalize score fields
    candidates_with_scores: List[Dict[str, Any]] = []
    for cand in candidates[:10]:
        trigram = float(cand.get("trigram_score", 0.0) or 0.0)
        phonetic = float(cand.get("phonetic_score", 0.0) or 0.0)
        vector = float(cand.get("vector_score", 0.0) or 0.0)
        match_score = float(cand.get("match_score", 0.0) or 0.0)
        # Some callers provide hybrid_score only
        if match_score <= 0.0:
            match_score = float(cand.get("hybrid_score", 0.0) or 0.0)
        candidates_with_scores.append(
            {
                **cand,
                "trigram_score": trigram,
                "phonetic_score": phonetic,
                "vector_score": vector,
                "match_score": match_score,
            }
        )

    if not candidates_with_scores:
        return None

    # Sort by match_score descending
    candidates_with_scores.sort(key=lambda x: float(x.get("match_score", 0.0) or 0.0), reverse=True)

    vector_enabled = any(float(c.get("vector_score", 0.0) or 0.0) > 0.0 for c in candidates_with_scores)

    # Find top candidates by each method
    top_trigram = max(candidates_with_scores, key=lambda x: float(x.get("trigram_score", 0.0) or 0.0))
    top_phonetic = max(
        candidates_with_scores,
        key=lambda x: (float(x.get("phonetic_score", 0.0) or 0.0), float(x.get("match_score", 0.0) or 0.0)),
    )
    top_vector = max(candidates_with_scores, key=lambda x: float(x.get("vector_score", 0.0) or 0.0)) if vector_enabled else None

    top_trigram_id = _cand_id(top_trigram)
    top_phonetic_id = _cand_id(top_phonetic)
    top_vector_id = _cand_id(top_vector) if top_vector else None

    # Option A hard floor
    option_a_min_score = 0.70
    try:
        option_a_min_score = float(os.getenv("KB_OPTION_A_MIN_SCORE", "0.70"))
    except Exception:
        option_a_min_score = 0.70

    # OPTION A: agreement (3/3 if vector enabled, else 2/2)
    if not is_generic:
        if vector_enabled and top_trigram_id and top_trigram_id == top_phonetic_id == top_vector_id:
            trigram_s = float(top_trigram.get("trigram_score", 0.0) or 0.0)
            vector_s = float((top_vector or {}).get("vector_score", 0.0) or 0.0)
            if max(trigram_s, vector_s) >= option_a_min_score:
                return top_trigram
        if (not vector_enabled) and top_trigram_id and top_trigram_id == top_phonetic_id:
            trigram_s = float(top_trigram.get("trigram_score", 0.0) or 0.0)
            if trigram_s >= option_a_min_score:
                return top_trigram

    # OPTION B: deterministic gates
    top_candidate = candidates_with_scores[0]
    top_score = float(top_candidate.get("match_score", 0.0) or 0.0)
    second_candidate = candidates_with_scores[1] if len(candidates_with_scores) > 1 else None
    second_score = float(second_candidate.get("match_score", 0.0) or 0.0) if second_candidate else 0.0

    threshold_b = _effective_auto_bind_threshold(entity_kind, auto_bind_threshold)
    # B1: high confidence score (Medication kinds use lower threshold for higher auto-link rate)
    if (not is_generic) and top_score >= threshold_b:
        return top_candidate

    # B2: 2-of-3 agreement + margin + hard floor (same as Option A)
    if not is_generic:
        top_id = _cand_id(top_candidate)
        agreements: List[str] = []
        if top_trigram_id == top_id:
            agreements.append("trigram")
        if top_phonetic_id == top_id:
            agreements.append("phonetic")
        if vector_enabled and top_vector_id == top_id:
            agreements.append("vector")
        if len(agreements) >= 2 and ("trigram" in agreements or "phonetic" in agreements):
            margin = top_score - second_score
            trigram_s = float(top_candidate.get("trigram_score", 0.0) or 0.0)
            vector_s = float(top_candidate.get("vector_score", 0.0) or 0.0)
            # Conservative "brand confusion" escape hatch (same as apply_decision_flow):
            # allow trigram+vector dominance + form overlap with a slightly lower floor.
            brand_gate_trigram_min = 0.62
            brand_gate_vector_min = 0.30
            brand_gate_margin_min = 0.15
            try:
                brand_gate_trigram_min = float(os.getenv("KB_BRAND_GATE_TRIGRAM_MIN", "0.62"))
                brand_gate_vector_min = float(os.getenv("KB_BRAND_GATE_VECTOR_MIN", "0.30"))
                brand_gate_margin_min = float(os.getenv("KB_BRAND_GATE_MARGIN_MIN", "0.15"))
            except Exception:
                pass

            def _form_overlap(a: str, b: str) -> bool:
                a_l = (a or "").lower()
                b_l = (b or "").lower()
                forms = ["capsule", "capsules", "tablet", "tablets", "tab", "tabs", "syrup", "inj", "injection", "drops"]
                return any(f in a_l for f in forms) and any(f in b_l for f in forms) and any(
                    (("capsule" in a_l or "capsules" in a_l) and ("capsule" in b_l or "capsules" in b_l))
                    or (("tablet" in a_l or "tablets" in a_l or "tab" in a_l) and ("tablet" in b_l or "tablets" in b_l or "tab" in b_l))
                    or (("syrup" in a_l) and ("syrup" in b_l))
                    or (("inj" in a_l or "injection" in a_l) and ("inj" in b_l or "injection" in b_l))
                    or (("drops" in a_l) and ("drops" in b_l))
                    for _ in [0]
                )

            if (
                "trigram" in agreements
                and "vector" in agreements
                and trigram_s >= brand_gate_trigram_min
                and vector_s >= brand_gate_vector_min
                and margin >= brand_gate_margin_min
                and _form_overlap(mention, top_candidate.get("display_name", "") or "")
                and entity_kind in ("Drug", "Medicine", "Nutrition", "Supplement", "Vaccine")
            ):
                return top_candidate

            if margin >= 0.08 and max(trigram_s, vector_s) >= option_a_min_score:
                return top_candidate

    # B3: dominance + margin (same threshold as B1)
    if (not is_generic) and top_score >= threshold_b and (top_score - second_score) >= 0.08:
        return top_candidate

    # Reject (would have required LLM judge)
    if logger:
        logger.debug("  🧯 Deterministic-only decision flow: no safe auto-accept; returning None (no LLM)")
    return None


def disambiguate_local_match_v2(
    original_mention: str,
    search_term_used: str,
    candidates: List[Dict[str, Any]],
    entity_kind: str,
    context_sentence: str,
    assessment_context: Optional[str] = None,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    global_candidates: Optional[List[Dict[str, Any]]] = None,
    # NEW: Contextual information for clinical decision-making
    species: Optional[str] = None,
    breed: Optional[str] = None,
    suspected_condition: Optional[str] = None,
    section: Optional[str] = None,
    hints: Optional[List[str]] = None,
    query_expansion: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    LLM Judge (Option C) with new prompt format.
    
    Args:
        original_mention: Exactly what appeared in transcript/span_text (may include [unclear])
        search_term_used: What the retrieval system searched with (may be ASR-corrected)
        candidates: List of candidates with match scores
        entity_kind: Entity kind
        context_sentence: Short transcript context
        client: OpenAI-compatible client
        logger: Optional logger
        global_candidates: Optional global KB candidates (for context only)
    
    Returns:
        Selected candidate dict, or None if rejected
    """
    if not client:
        return None
    
    if not candidates:
        if logger:
            logger.debug(f"  ⚠️  LLM Judge: No local candidates for '{original_mention}' - cannot bind billable item")
        return None
    
    if logger:
        logger.debug(f"  🧠 LLM Judge (Option C): Disambiguating '{original_mention}' with {len(candidates)} candidates")
    
    # Build candidate list with method scores and descriptions/definitions
    candidate_list = []
    for i, cand in enumerate(candidates[:10], 1):  # Limit to top 10
        display_name = cand.get("display_name", "")
        stock_id = cand.get("stock_id")
        service_id = cand.get("service_id")
        description = cand.get("description", "") or cand.get("definition", "")  # Local description or KB definition
        trigram_score = cand.get("trigram_score", 0.0)
        phonetic_score = cand.get("phonetic_score", 0.0)
        vector_score = cand.get("vector_score", 0.0)
        match_score = cand.get("match_score", 0.0)
        
        score_info = f"trigram={trigram_score:.3f}, phonetic={phonetic_score:.1f}, vector={vector_score:.3f}, match={match_score:.3f}"
        desc_info = f"\n   Description: {description}" if description else ""
        
        if stock_id:
            candidate_list.append(f"{i}. {display_name} (stock_id: {stock_id}, {score_info}){desc_info}")
        elif service_id:
            candidate_list.append(f"{i}. {display_name} (service_id: {service_id}, {score_info}){desc_info}")
        else:
            candidate_list.append(f"{i}. {display_name} ({score_info}){desc_info}")
    
    # Build global candidates context (for explanation only, not binding) - include definitions and domain_key
    global_context = ""
    detected_domain = None
    if global_candidates:
        global_list = []
        for i, gc in enumerate(global_candidates[:15], 1):
            gc_name = gc.get('preferred_name', 'Unknown')
            gc_kind = gc.get('kind', 'Unknown')
            gc_domain = gc.get('domain_key', '')  # Extract domain_key from KB concept
            gc_def = gc.get('definition', '')
            def_info = f" - {gc_def}" if gc_def else ""
            domain_info = f" [domain: {gc_domain}]" if gc_domain else ""
            global_list.append(f"{i}. {gc_name} (KB concept, kind: {gc_kind}{domain_info}){def_info}")
            # Detect domain from global candidates (most common domain among top candidates)
            if gc_domain and not detected_domain:
                detected_domain = gc_domain
        global_context = f"\n{chr(10).join(global_list)}"
    
    # Also try to detect domain from suspected_condition or context_sentence
    if not detected_domain and suspected_condition:
        try:
            from kb_ner_super_pass import detect_domain
            # Use suspected_condition as domain signal (e.g., "hip dysplasia" → "orthopedic")
            detected_domain = detect_domain(suspected_condition)
            if detected_domain == 'general':
                detected_domain = None
        except Exception:
            pass
    
    # Build contextual information section (include detected domain)
    context_info_parts = []
    if species:
        context_info_parts.append(f"Species: {species}")
    if breed:
        context_info_parts.append(f"Breed: {breed}")
    if suspected_condition:
        context_info_parts.append(f"Suspected Condition: {suspected_condition}")
    if section:
        context_info_parts.append(f"SOAP Section: {section}")
    if detected_domain:
        context_info_parts.append(f"Clinical Domain: {detected_domain}")
    context_info = "\n".join(context_info_parts) if context_info_parts else "Not available"
    
    judge_prompt = f"""You are the LLM JUDGE for a veterinary entity-linking system.
Your job is to act as a STRICT PLAUSIBILITY GATE for BILLING ACCURACY.

You will receive:
- ORIGINAL_MENTION: exactly what appeared in transcript/span_text (may include [unclear])
- SEARCH_TERM_USED: what the retrieval system searched with (may be ASR-corrected if [unclear])
- ENTITY_KIND: Drug / Procedure / Service / Vaccine / DiagnosticTest / LabTest / Nutrition / Device / Substance
- CONTEXT: short transcript context
- ASSESSMENT_CONTEXT: brief patient assessment/diagnosis context (may be empty)
- CLINICAL_CONTEXT: species, breed, suspected condition, SOAP section, clinical domain (for clinical decision-making)
- LOCAL CANDIDATES: the ONLY selectable options (for billing accuracy)
- GLOBAL CANDIDATES: hints only (NOT selectable) - may include domain_key metadata

Match scores come from 3 methods and are hints only:
- trigram_score: 0.0–1.0 lexical similarity
- phonetic_score: 0.0 or 0.8 (metaphone match; pronunciation/ASR brand variant signal)
- vector_score: 0.0–1.0 semantic similarity (may be 0 if embeddings not used)
- match_score = max(trigram_score, phonetic_score, vector_score)

CRITICAL OUTPUT RULE:
Return ONLY ONE of:
- a LOCAL candidate number (e.g., 1, 2, 3...)
- NONE_GENERIC
- NONE
No other text.

========================
INPUT
ORIGINAL_MENTION: "{original_mention}"
SEARCH_TERM_USED: "{search_term_used}"
ENTITY_KIND: {entity_kind}
CONTEXT: "{context_sentence}"
ASSESSMENT_CONTEXT: "{(assessment_context or '').strip()}"
CLINICAL_CONTEXT:
{context_info}

LOCAL CANDIDATES (ONLY SELECT FROM THESE):
{chr(10).join(candidate_list)}

GLOBAL CANDIDATES (HINTS ONLY, NOT SELECTABLE):
{global_context}
========================

DECISION RULES (FOLLOW IN ORDER)

1) Decide if the mention is GENERIC or SPECIFIC (use ORIGINAL_MENTION primarily; SEARCH_TERM_USED helps if [unclear]).

GENERIC if it is only a broad class/form with no identifying anchor:
- "examination", "test", "injection", "capsules", "tablets", "medicine", "vaccine", etc.

SPECIFIC if it includes ANY identifying anchor:
- A brand/trade token (even if spelled differently due to ASR/phonetics)
- A specific procedure/test name ("anal gland expression", etc.)
- Strength/pack markers ("500mg", "60S", "1 ml", etc.)
- A distinctive multi-word phrase beyond generic form words

IMPORTANT: Brand tokens can be phonetic/ASR variants.
If phonetic_score == 0.8 for a candidate, treat it as strong evidence of SAME-BRAND pronunciation match,
even if trigram_score is low.

2) GENERIC vs SPECIFIC safety rule (billing protection)

If the mention is GENERIC and the plausible LOCAL candidates are SPECIFIC named/branded items,
you MUST return NONE_GENERIC.

Only return a candidate number when the mention is SPECIFIC enough to justify billing that specific item/service.

3) Plausibility filters (local-only)

A candidate is selectable ONLY if:
- KIND COMPATIBILITY: the candidate matches ENTITY_KIND (do not map drug→procedure or procedure→drug)
- ANCHOR SUPPORT:
  - If mention has a brand/procedure anchor, candidate must match that anchor
    (exactly OR as a close phonetic/ASR variant indicated by phonetic_score==0.8 or obvious spelling variant)
  - Form words alone (tablet/capsule/injection) are NOT enough
- CONTEXT SUPPORT: context does not contradict the candidate type/intent
- CLINICAL SENSE (use ASSESSMENT_CONTEXT and CLINICAL_CONTEXT when provided):
  - If candidate looks like a lab reagent / non-clinical supply but ENTITY_KIND is drug/procedure and assessment suggests active disease treatment,
    reject it.
  - If assessment suggests infection/respiratory distress and candidate is an unrelated chemical/reagent, reject it.
  - DOMAIN MATCHING (Clinical Plausibility Gate): Use CLINICAL_DOMAIN from CLINICAL_CONTEXT:
    * If CLINICAL_DOMAIN is detected (e.g., "orthopedic", "cardiology", "oncology") and a GLOBAL candidate has domain_key matching that domain,
      that candidate is 10x more likely to be correct than a candidate from a different domain.
    * Example: If domain is "orthopedic" and you see "ultralining" vs "Ortolani" (orthopedic) vs "ultrasound-guided" (general),
      prefer "Ortolani" because it matches the clinical domain, even if scores are similar.
    * This prevents wrong-domain matches (e.g., "noble angle" → "nebula" [ophthalmology] when domain is "orthopedic").
  - CONTEXTUAL WEIGHTING: Use CLINICAL_CONTEXT to resolve phonetic conflicts:
    * If mention is phonetically similar to multiple candidates (e.g., "ultralining" vs "Ortolani" vs "Ultra-Lining"),
      and CLINICAL_CONTEXT shows: Labrador + Hip Joint + Positive Test + SOAP Section: Objective + Clinical Domain: orthopedic,
      then "Ortolani" is the correct match (orthopedic test for hip dysplasia).
    * If phonetic_score==0.8 for a candidate AND it fits the clinical context (species, breed, condition, section, domain),
      approve it even if trigram_score is lower.
- NO STRONG CONTRADICTIONS: if mention includes strength/pack and candidate conflicts clearly, reject

4) How to use method scores (scores are evidence, not authority)

- Prefer candidates with strong lexical/phonetic anchors:
  - trigram high AND/OR phonetic_score==0.8 is usually reliable for SKU/brand matching
- Be conservative with "vector-only" matches:
  - If a candidate is mainly supported by vector_score while trigram and phonetic are weak,
    select it ONLY if the mention/context provide clear anchors.
  - Otherwise, reject (NONE or NONE_GENERIC depending on Rule 2).

5) Selection / tie-break (deterministic)

- If no candidate passes plausibility → return NONE (or NONE_GENERIC if Rule 2 triggered)
- If exactly one passes → return its number
- If multiple pass:
  - pick the one that matches the most anchors (brand + form + strength/pack)
  - if still tied, pick the higher match_score
  - if still tied/uncertain → return NONE (conservative)

YOUR SELECTION (ONE TOKEN ONLY):
""".strip()
    
    try:
        # Batch Option C (micro-batcher). Falls back to None on failure.
        # This coalesces multiple judge calls into one request, reducing latency.
        req_id = f"req_{abs(hash((original_mention, search_term_used, entity_kind, context_sentence))) % 10_000_000}"
        selected = _submit_to_batch_judge(
            req_id=req_id,
            original_mention=original_mention,
            search_term_used=search_term_used,
            candidates=candidates[:10],
            entity_kind=entity_kind,
            context_sentence=context_sentence,
            assessment_context=assessment_context,
            client=client,
            logger=logger,
            hints=(hints or [])[:5],
            query_expansion=(query_expansion or [])[:5],
        )
        if selected:
            if logger:
                logger.info(f"  ✅ LLM Judge SELECTED: '{original_mention}' → '{selected.get('display_name')}' (batched)")
            return selected
        # Batched judge returned NONE/NONE_GENERIC or failed => conservative reject
        if logger:
            logger.info(f"  🚫 LLM Judge REJECTED: '{original_mention}' (batched NONE/NONE_GENERIC)")
        return None
        
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠️  LLM Judge failed: {e}")
        return None


def disambiguate_local_match(
    mention: str,
    candidates: List[Dict[str, Any]],
    context_sentence: str,
    entity_kind: str,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    global_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    LLM Judge as 'Plausibility Gate' with DUAL-SYNC support.
    
    This function prevents over-specification by allowing the LLM to reject matches
    when a generic mention (e.g., "examination") is matched to a specific service
    (e.g., "Fecal examination") that the doctor didn't actually mention.
    
    CRITICAL: Judge can ONLY select from LOCAL candidates (billing accuracy).
    Global candidates are for context/explanation only, not for direct binding.
    
    Args:
        mention: The original mention from transcript (e.g., "examination")
        candidates: List of candidate matches from local search
        context_sentence: Surrounding context from transcript
        entity_kind: Entity kind (e.g., "Procedure", "Drug")
        client: OpenAI-compatible client
        logger: Optional logger
        global_candidates: Optional global KB candidates (for context only)
    
    Returns:
        Selected LOCAL candidate dict, or None if judge rejects all candidates
    """
    if not client:
        return None
    
    # CRITICAL: If no local candidates, judge cannot bind (no billing item)
    if not candidates:
        if logger:
            logger.debug(f"  ⚠️  LLM Judge: No local candidates for '{mention}' - cannot bind billable item")
        return None
    
    if logger:
        logger.debug(f"  🧠 LLM Judge: Disambiguating '{mention}' with {len(candidates)} candidates")
    
    # Build candidate list for judge - include descriptions
    candidate_list = []
    for i, cand in enumerate(candidates[:10], 1):  # Limit to top 10
        display_name = cand.get("display_name", "")
        stock_id = cand.get("stock_id")
        service_id = cand.get("service_id")
        description = cand.get("description", "") or cand.get("definition", "")  # Local description or KB definition
        score = cand.get("match_score", 0)
        desc_info = f"\n   Description: {description}" if description else ""
        
        if stock_id:
            candidate_list.append(f"{i}. {display_name} (stock_id: {stock_id}, score: {score:.3f}){desc_info}")
        elif service_id:
            candidate_list.append(f"{i}. {display_name} (service_id: {service_id}, score: {score:.3f}){desc_info}")
        else:
            candidate_list.append(f"{i}. {display_name} (score: {score:.3f}){desc_info}")
    
    # Build global candidates context (for explanation only, not binding) - include definitions
    global_context = ""
    if global_candidates:
        global_list = []
        for i, gc in enumerate(global_candidates[:15], 1):
            gc_name = gc.get('preferred_name', 'Unknown')
            gc_kind = gc.get('kind', 'Unknown')
            gc_def = gc.get('definition', '')
            def_info = f" - {gc_def}" if gc_def else ""
            global_list.append(f"{i}. {gc_name} (KB concept, kind: {gc_kind}){def_info}")
        global_context = f"\n\nGLOBAL KB CONTEXT (for reference only - DO NOT bind these directly):\n{chr(10).join(global_list)}\n\nNOTE: Global candidates are for context/explanation. You can ONLY select from LOCAL candidates above for billing."
    
    judge_prompt = f"""You are a veterinary clinical entity linker acting as a Plausibility Gate.

MENTION: "{mention}"
ENTITY KIND: {entity_kind}
CONTEXT: "{context_sentence}"

LOCAL INVENTORY/SERVICES CANDIDATES (YOU CAN ONLY SELECT FROM THESE):
{chr(10).join(candidate_list)}
{global_context}
CRITICAL RULE - GENERIC vs SPECIFIC:
- If the mention is GENERIC (e.g., "examination", "capsules", "tablets") 
  and the candidates are SPECIFIC (e.g., "Fecal examination", "Cortex capsules", "Nutrish tablets 60S"),
  you MUST select "NONE_GENERIC" to reject the match.

CATEGORY INCOMPATIBILITY (Cross-Kind Check):
- You MUST REJECT (select NONE) when ENTITY_KIND is a finding (Symptom, Diagnosis, ReasonForVisit) and the candidate is a tangible product (Medication, Lab Kit, Consumable). A diagnosis or reason-for-visit like "pus" or "yeast" is a state of being, not a product you can put in a bag. Example: Diagnosis + Lasix → NONE; ReasonForVisit "yeast growth" + AST KIT → NONE.

SYMPTOM / PHYSICAL FINDING SUPPRESSOR (Safety):
- If the mention is a symptom or physical finding (e.g. pus, yeast, yeast growth, discharge, shaking, swelling, growth in ear), you MUST select "NONE" for any candidate that is a Medication or Lab Reagent unless the CONTEXT explicitly says the vet is prescribing that product or ordering that test. Symptoms and findings must NOT be grounded to billable SKUs (e.g. "bus inside the left ear" [ASR for pus] → NONE; "yeast growth" + lab reagent candidate → NONE).

FORM-FACTOR & ROUTE ALIGNMENT:
- Match the vet's mention form/route to the candidate. Syrup/ml/drops/suspension → prefer liquid candidates; reject tablet if a liquid candidate exists. Tablet/tab/mg → prefer solid; reject syrup. Inject/vial/IM/IV → prefer injectable; reject oral. Apply/spray/pump/cream/drops (topical) → prefer topical; reject oral/injectable. If the top candidate has the wrong form/route, select the candidate that matches (even if lower score).

- Only select a LOCAL candidate ID if:
  1. The candidate is a DIRECT medical match (exact or clinically certain synonym)
  2. The mention contains enough specificity to justify the candidate
  3. The candidate's form/route matches the mention (syrup→liquid, tablet→solid, inject→injectable, apply→topical)
  4. The context supports the specific match
  5. The candidate is from the LOCAL list above (for billing accuracy)

EXAMPLES:
- Mention: "examination" → Candidate: "Fecal examination" → Select: NONE_GENERIC (too specific - mention is generic)
- Mention: "fecal examination" → Candidate: "Fecal examination" → Select: candidate ID (exact match)
- Mention: "capsules" → Candidate: "Cortex capsules" → Select: NONE_GENERIC (too specific - mention is generic)
- Mention: "Cortex capsules" → Candidate: "COATEX BLISTER CAPSULE" → Select: candidate ID (brand + form match - APPROVE)
- Mention: "Coatex" → Candidate: "COATEX BLISTER CAPSULE" → Select: candidate ID (brand match - APPROVE)
- Mention: "nutrish tablets" → Candidate: "NUTRICH TABLET 60S" → Select: candidate ID (brand match - APPROVE)

IMPORTANT: 
- If the mention contains a BRAND NAME (e.g., "Cortex", "Coatex", "Nutrish") or a SPECIFIC PROCEDURE NAME (e.g., "anal gland expression"), it is SPECIFIC enough to match.
- Only reject if the mention is truly generic (e.g., just "examination", "capsules", "tablets" without a brand or specific name).
- You MUST select from LOCAL candidates only (for billing accuracy). Global candidates are for context only.

Return ONLY:
- A LOCAL candidate number (1-10) if you select a match
- "NONE_GENERIC" if the mention is too generic for the specific candidates
- "NONE" if no LOCAL candidate is appropriate

Your selection:"""
    
    try:
        # Use configured Judge model (LLM_JUDGE_MODEL)
        model = LLM_JUDGE_MODEL
        try:
            from kb_ner_clients import get_client_for_model
            judge_client, _ = get_client_for_model(model)
            if judge_client:
                client = judge_client
        except Exception:
            pass
        judge_resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a veterinary clinical entity linker. Return only a number or NONE_GENERIC or NONE."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0,
            max_tokens=10,
        )
        judge_text = judge_resp.choices[0].message.content.strip().upper()
        
        if "NONE_GENERIC" in judge_text or "NONE" in judge_text:
            if logger:
                logger.info(f"  🚫 LLM Judge REJECTED: '{mention}' is too generic for specific candidates")
            return None
        
        # Try to extract number
        numbers = re.findall(r'\d+', judge_text)
        if numbers:
            judge_choice = int(numbers[0])
            if 1 <= judge_choice <= len(candidates):
                selected = candidates[judge_choice - 1]
                if logger:
                    logger.info(f"  ✅ LLM Judge SELECTED: '{mention}' → '{selected.get('display_name')}' (candidate #{judge_choice})")
                return selected
        
        if logger:
            logger.warning(f"  ⚠️  LLM Judge returned unclear response: {judge_text}")
        return None
        
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠️  LLM Judge failed: {e}")
        return None

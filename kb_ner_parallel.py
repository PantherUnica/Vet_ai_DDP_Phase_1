"""
Parallel Entity Processing Module

This module provides async functions to process entities in parallel,
significantly reducing latency when processing multiple entities.

Grounding: query_expansion uses the same grounding process as hints (reference: soap_notes_phase_1).
Both are passed into local/global search for suggestion boost, into batch need_global (9-tuple),
into Judge batch items (BRAIN_HINTS / QUERY_EXPANSIONS), and preserved on entity result.
"""

# CRITICAL: Import asyncio FIRST before any other imports that might cause circular dependencies
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# CRITICAL: Avoid importing kb_ner_linker at module load time (circular import risk).
# Import the underlying modules directly instead.
from kb_ner_routing import (
    sanitize_asr_errors,
    canonicalize_kind,
    classify_entity_route,
    classify_procedure_role,
    extract_context_window,
    is_clinical_status_constant,
    is_services_only_kind,
    DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD,
)
from kb_ner_db import acquire_pg_conn, release_pg_conn, pg_conn_ctx, insert_ai_draft_entity, search_vitals_registry_topk
from kb_ner_local_search import search_local_inventory_topk, search_local_services_topk
from kb_ner_global_search import (
    map_ner_kind_to_kb_kind_filter,
    REASON_ALLOWED_KB_KINDS,
)
LOCAL_ONLY = os.getenv("LOCAL_ONLY", "true").lower() in ("1", "true", "yes")
from kb_ner_disambiguation import (
    refine_term_with_llm_candidate_aware,
    apply_decision_flow,
    apply_decision_flow_deterministic_only,
    run_single_batch_llm_judge,
)


def _hint_item_to_str(h: Any) -> str:
    """Normalize a hint item (string or dict with 'hint' key) to a single stripped string."""
    if h is None:
        return ""
    if isinstance(h, str):
        return (h or "").strip()
    if isinstance(h, dict):
        return str((h.get("hint") or "")).strip()
    return str(h).strip()


def _normalize_hints(hints: Any, max_items: int = 5) -> List[str]:
    """Return a list of hint strings from entity hints (accepts list of strings or list of dicts with 'hint')."""
    if not hints or not isinstance(hints, list):
        return []
    out = []
    for item in hints:
        s = _hint_item_to_str(item)
        if s:
            out.append(s)
    return out[:max_items]


def _get_highest_probability_hint(entity: Dict[str, Any]) -> Optional[str]:
    """
    Extract the hint with the highest probability from entity's hints and hint_probabilities.
    Returns the hint text (string) with highest probability, or None if no hints/probabilities available.
    """
    hints = entity.get("hints")
    hint_probabilities = entity.get("hint_probabilities", {}) or {}

    if not hints or not isinstance(hints, list):
        return None

    best_hint: Optional[str] = None
    best_prob: float = 0.0

    for hint_item in hints:
        hint_text = _hint_item_to_str(hint_item)
        if isinstance(hint_item, dict):
            try:
                prob = float(hint_item.get("probability") or 0.0)
            except (TypeError, ValueError):
                prob = 0.0
        else:
            try:
                prob = float(hint_probabilities.get(hint_text, 0.0))
            except (TypeError, ValueError):
                prob = 0.0

        if hint_text and prob > best_prob:
            best_prob = prob
            best_hint = hint_text

    if best_hint:
        return best_hint

    # Fallback: if we never found a probability but have at least one hint, use the first
    first = hints[0]
    return _hint_item_to_str(first) or None


async def _run_parallel_llm_judges_for_dual_route(
    entities_needing_judge: List[Tuple[int, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]],
    client: Optional[Any],
    logger: Optional[logging.Logger],
    auto_bind_threshold: float,
    llm_judge_threshold: float,
    raw_transcript: Optional[str] = None,
) -> Dict[int, Optional[Dict[str, Any]]]:
    """
    Run LLM judges for dual_route entities with strict two-phase flow:
    1) deterministic gates for every entity
    2) one single-shot Option-C batch call for unresolved entities

    Args:
        entities_needing_judge: List of (entity_idx, entity_dict, local_candidates, global_candidates) tuples
        client: OpenAI-compatible client
        logger: Optional logger
        auto_bind_threshold: Auto-bind threshold
        llm_judge_threshold: LLM judge threshold
        raw_transcript: Optional raw transcript for domain detection (passed through to apply_decision_flow)

    Returns:
        Dict mapping entity_idx -> selected candidate (or None if rejected)
    """
    if not entities_needing_judge:
        return {}

    judge_results: Dict[int, Optional[Dict[str, Any]]] = {}
    unresolved_batch_items: List[Dict[str, Any]] = []
    req_to_entity_idx: Dict[str, int] = {}

    # Phase 1: deterministic-only pass (no network)
    for entity_idx, entity, local_cands, global_cands in entities_needing_judge:
        try:
            span_text = entity.get("span_text", "")
            normalized_name = entity.get("normalized_name", span_text)
            canonical_kind = entity.get("canonical_kind") or entity.get("kind", "Other")
            search_term = (entity.get("search_term") or "").strip() or normalized_name
            context_sentence = entity.get("context_sentence", "")

            # Make sure local ranking uses boosted score consistently
            local_sorted = sorted(
                local_cands or [],
                key=lambda x: float(x.get("final_score", x.get("match_score", 0)) or 0),
                reverse=True,
            )
            local_for_decision = [
                {
                    **c,
                    "match_score": float(c.get("final_score", c.get("match_score", 0)) or 0),
                }
                for c in local_sorted
            ]

            deterministic_selected = apply_decision_flow_deterministic_only(
                mention=span_text,
                candidates=local_for_decision,
                entity_kind=canonical_kind,
                logger=logger,
                auto_bind_threshold=auto_bind_threshold,
            )
            if deterministic_selected is not None:
                judge_results[entity_idx] = deterministic_selected
                continue

            req_id = f"dual_{entity_idx}_{abs(hash((span_text, search_term, canonical_kind))) % 1_000_000}"
            req_to_entity_idx[req_id] = entity_idx
            _hints = _normalize_hints(entity.get("hints"), max_items=5)
            _qe = entity.get("query_expansion")
            _qe = [str(x).strip() for x in _qe if str(x).strip()][:5] if isinstance(_qe, list) else []
            unresolved_batch_items.append(
                {
                    "req_id": req_id,
                    "original_mention": span_text,
                    "search_term_used": search_term,
                    "candidates": local_for_decision[:10],
                    "entity_kind": canonical_kind,
                    "context_sentence": context_sentence,
                    "assessment_context": entity.get("assessment_context"),
                    "hints": _hints,
                    "query_expansion": _qe,
                    # Keep global context available for future prompt expansion
                    "global_candidates": (global_cands or [])[:10],
                }
            )
        except Exception as e:
            if logger:
                logger.warning(
                    "  ⚠️ Deterministic pre-pass failed for entity %s '%s': %s",
                    entity_idx, entity.get("span_text", ""), e,
                )
            judge_results[entity_idx] = None

    # Phase 2: one explicit Option-C batch call for unresolved entities
    if unresolved_batch_items and client:
        try:
            batch_out = await asyncio.to_thread(
                run_single_batch_llm_judge,
                batch_items=unresolved_batch_items,
                client=client,
                logger=logger,
            )
            for req_id, entity_idx in req_to_entity_idx.items():
                judge_results[entity_idx] = batch_out.get(req_id)
        except Exception as e:
            if logger:
                logger.warning("  ⚠️ Single-shot batch judge failed: %s", e)
            for req_id, entity_idx in req_to_entity_idx.items():
                judge_results[entity_idx] = None
    else:
        for req_id, entity_idx in req_to_entity_idx.items():
            judge_results[entity_idx] = None

    if logger:
        successful = sum(1 for v in judge_results.values() if v is not None)
        unresolved = len(req_to_entity_idx)
        logger.info(
            "  ✅ Dual-route judge complete: %s/%s linked (deterministic + single-shot batch unresolved=%s)",
            successful, len(judge_results), unresolved,
        )

    return judge_results


async def batch_search_and_judge_all_billable_entities(
    billable_entities: List[Dict[str, Any]],
    cleaned_transcript: str,
    raw_transcript: Optional[str],
    client: Optional[Any],
    clinic_id: Optional[int],
    logger: Optional[logging.Logger],
    embedding_cache: Optional[Dict[str, List[float]]] = None,
    auto_bind_threshold: float = 0.92,
    llm_judge_threshold: float = 0.55,
) -> Dict[int, Optional[Dict[str, Any]]]:
    """
    Batch search and judge for ALL billable entities in ONE LLM call.
    
    This function:
    1. Runs local inventory + local services searches for all billable entities in parallel (LOCAL-ONLY mode)
    2. Collects all candidates with scores
    3. Passes everything to ONE batch LLM judge call
    4. Returns decisions for all entities
    
    Args:
        billable_entities: List of dicts with keys: entity_idx, entity (dict)
        cleaned_transcript: Cleaned transcript for context
        raw_transcript: Raw transcript for domain detection
        client: OpenAI client
        clinic_id: Clinic ID
        logger: Logger
        embedding_cache: Pre-computed embeddings
        auto_bind_threshold: Threshold for auto-binding
        llm_judge_threshold: Threshold for LLM judge
        
    Returns:
        Dict mapping entity_idx -> judge result (candidate dict or None)
    """
    if not billable_entities or not client:
        return {}
    
    if logger:
        logger.info(f"  🚀 BATCH SEARCH + JUDGE: Processing {len(billable_entities)} billable entities in one LLM call")
    
    # Step 1: Run all searches in parallel for all entities
    search_tasks = []
    
    for item in billable_entities:
        entity_idx = item.get("entity_idx")
        entity = item.get("entity", {})
        span_text = entity.get("span_text", "")
        normalized_name = entity.get("normalized_name", span_text)
        search_term = (entity.get("search_term") or "").strip() or normalized_name
        
        if not span_text:
            continue
        
        canonical_kind = canonicalize_kind(entity.get("kind") or entity.get("kb_kind") or "Other")
        route = classify_entity_route(canonical_kind, entity=entity, logger=logger)
        
        if route != "dual_sync":
            continue  # Only process dual_sync entities here
        
        # Extract entity parameters
        entity_suggestion_prob = entity.get("suggestion_probability")
        entity_hints = _normalize_hints(entity.get("hints"), max_items=3)
        entity_hint_probs = entity.get("hint_probabilities")
        entity_inventory_category = entity.get("inventory_category")
        entity_service_category = entity.get("service_category")
        entity_service_type = (entity.get("service_type") or "").strip()
        
        # Domain filter
        _local_domain_filter = None
        if raw_transcript:
            try:
                from kb_ner_super_pass import detect_domain
                d = detect_domain(raw_transcript)
                if d and str(d).strip().lower() != "general":
                    _local_domain_filter = d.strip()
            except Exception:
                pass
        
        entity_query_expansion = entity.get("query_expansion")
        if isinstance(entity_query_expansion, list):
            entity_query_expansion = [str(x).strip() for x in entity_query_expansion if str(x).strip()][:3]
        else:
            entity_query_expansion = []
        search_tasks.append({
            "entity_idx": entity_idx,
            "entity": entity,
            "span_text": span_text,
            "search_term": search_term,
            "canonical_kind": canonical_kind,
            "entity_suggestion_prob": entity_suggestion_prob,
            "entity_hints": entity_hints,
            "entity_hint_probs": entity_hint_probs,
            "entity_query_expansion": entity_query_expansion,
            "entity_inventory_category": entity_inventory_category,
            "entity_service_category": entity_service_category,
            "entity_service_type": entity_service_type,
            "domain_filter": _local_domain_filter,
        })
    
    # Execute all searches in parallel (using connection pool)
    search_results = {}
    with pg_conn_ctx(logger=logger) as conn:
        for task in search_tasks:
            entity_idx = task["entity_idx"]
            search_term = task["search_term"]
            canonical_kind = task["canonical_kind"]
            
            try:
                # Pharmacy-Free Zone: Diagnosis and ReasonForVisit search Services ONLY (never Pharmacy/Inventory).
                services_only = is_services_only_kind(canonical_kind)
                if services_only:
                    local_inv = []
                else:
                    local_inv = search_local_inventory_topk(
                        conn, search_term, canonical_kind, clinic_id, logger,
                        0.50, 5, client,
                        embedding_cache=embedding_cache,
                        precomputed_embedding=embedding_cache.get(search_term) if embedding_cache else None,
                        domain_filter=task["domain_filter"],
                        suggestion_probability=task["entity_suggestion_prob"],
                        search_term=search_term,
                        hints=task["entity_hints"],
                        hint_probabilities=task["entity_hint_probs"],
                        query_expansion=task.get("entity_query_expansion") or None,
                        category_hints=task["entity_inventory_category"],
                    ) or []
                
                # Local services search
                local_svc = search_local_services_topk(
                    conn, search_term, canonical_kind, clinic_id, logger,
                    0.50, 5, client,
                    embedding_cache=embedding_cache,
                    precomputed_embedding=embedding_cache.get(search_term) if embedding_cache else None,
                    domain_filter=task["domain_filter"],
                    suggestion_probability=task["entity_suggestion_prob"],
                    search_term=search_term,
                    hints=task["entity_hints"],
                    hint_probabilities=task["entity_hint_probs"],
                    query_expansion=task.get("entity_query_expansion") or None,
                    category_hints=task["entity_service_category"],
                    service_type=task["entity_service_type"] or None,
                ) or []
                
                # Merge local results (LOCAL-ONLY mode: no global search)
                local_candidates = sorted(
                    local_inv + local_svc,
                    key=lambda x: float(x.get("final_score", x.get("match_score", 0)) or 0),
                    reverse=True
                )[:10]  # Top 10 local
                
                search_results[entity_idx] = {
                    "local_candidates": local_candidates,
                    "entity": task["entity"],
                    "span_text": task["span_text"],
                    "search_term": search_term,
                    "canonical_kind": canonical_kind,
                }
            except Exception as e:
                if logger:
                    logger.warning(f"  ⚠️ Search failed for entity {entity_idx}: {e}")
                search_results[entity_idx] = {
                    "local_candidates": [],
                    "entity": task["entity"],
                    "span_text": task["span_text"],
                    "search_term": search_term,
                    "canonical_kind": canonical_kind,
                }
    
    # Step 2: Apply deterministic gates (auto-bind high-confidence matches)
    auto_bound_results = {}
    entities_needing_judge = []
    
    for entity_idx, results in search_results.items():
        entity = results["entity"]
        span_text = results["span_text"]
        local_cands = results["local_candidates"]
        
        # Check auto-bind threshold
        best_local = local_cands[0] if local_cands else None
        # 0.95 certainty wall: Diagnosis/ReasonForVisit below threshold must not be grounded (note-only).
        threshold_diag_rfv = float(os.getenv("DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD", str(DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD)))
        if canonical_kind in ("Diagnosis", "ReasonForVisit"):
            best_score = float(best_local.get("match_score", 0) or 0) if best_local else 0.0
            if best_score < threshold_diag_rfv:
                # Do not auto-bind; do not send to judge for linking — entity stays note-only (handled later in flow).
                auto_bound_results[entity_idx] = None  # Signal: note-only, no link
                continue
        
        if best_local and best_local.get("match_score", 0) >= auto_bind_threshold:
            auto_bound_results[entity_idx] = best_local
            continue
        
        # Need judge
        entities_needing_judge.append({
            "entity_idx": entity_idx,
            "results": results,
        })
    
    if logger:
        logger.info(f"  📊 Batch search complete: {len(auto_bound_results)} auto-bound, {len(entities_needing_judge)} need judge")
    
    # Step 3: Single batch LLM judge for all entities needing judgment
    if not entities_needing_judge:
        return auto_bound_results
    
    # Build batch judge items
    batch_judge_items = []
    for item in entities_needing_judge:
        entity_idx = item["entity_idx"]
        results = item["results"]
        entity = results["entity"]
        span_text = results["span_text"]
        search_term = results["search_term"]
        local_cands = results["local_candidates"]
        canonical_kind = results["canonical_kind"]
        
        # Extract context
        context_start = max(0, cleaned_transcript.find(span_text) - 50)
        context_end = min(len(cleaned_transcript), cleaned_transcript.find(span_text) + len(span_text) + 50)
        context_sentence = cleaned_transcript[context_start:context_end]
        
        # LOCAL-ONLY mode: only local candidates (no global)
        all_candidates = local_cands[:10]
        
        batch_judge_items.append({
            "req_id": str(entity_idx),
            "original_mention": span_text,
            "search_term_used": search_term,
            "candidates": all_candidates,
            "entity_kind": canonical_kind,
            "context_sentence": context_sentence,
            "assessment_context": entity.get("assessment_context"),
        })
    
    # Single batch judge call
    judge_results = run_single_batch_llm_judge(
        batch_items=batch_judge_items,
        client=client,
        logger=logger,
    )
    
    # Combine auto-bound and judge results
    final_results = auto_bound_results.copy()
    for entity_idx, judge_result in judge_results.items():
        final_results[int(entity_idx)] = judge_result
    
    if logger:
        logger.info(f"  ✅ Batch judge complete: {len(final_results)} decisions for {len(billable_entities)} entities")
    
    return final_results


def _preserve_entity_metadata(entity_result: Dict[str, Any], source_entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preserve metadata from source entity (entity_id, hints, hint_probabilities, search_term, domain) in entity_result.
    This ensures Brain NER metadata flows through to final manifest and anchor integrity is maintained.
    """
    if not isinstance(entity_result, dict) or not isinstance(source_entity, dict):
        return entity_result
    
    # CRITICAL: Preserve entity_id - anchor integrity depends on this
    # Never drop/overwrite entity_id. If merging entities, keep stable "primary" id.
    if "entity_id" not in entity_result and source_entity.get("entity_id"):
        entity_result["entity_id"] = source_entity.get("entity_id")
    elif source_entity.get("entity_id") and not entity_result.get("entity_id"):
        # If result lost entity_id but source has it, restore it
        entity_result["entity_id"] = source_entity.get("entity_id")
    
    # Preserve hints if not already present (normalize to list of strings for downstream)
    if "hints" not in entity_result and source_entity.get("hints"):
        entity_result["hints"] = _normalize_hints(source_entity.get("hints"), max_items=10)[:3]
    
    # Preserve hint_probabilities if not already present
    if "hint_probabilities" not in entity_result and source_entity.get("hint_probabilities"):
        entity_result["hint_probabilities"] = source_entity.get("hint_probabilities")
    
    # Preserve search_term if not already present
    if "search_term" not in entity_result and source_entity.get("search_term"):
        entity_result["search_term"] = source_entity.get("search_term")
    
    # Preserve domain if not already present
    if "domain" not in entity_result and source_entity.get("domain"):
        entity_result["domain"] = source_entity.get("domain")

    if "query_expansion" not in entity_result and source_entity.get("query_expansion"):
        qe = source_entity.get("query_expansion")
        entity_result["query_expansion"] = (qe[:3] if isinstance(qe, list) else []) or []

    return entity_result


_DOSAGE_FORM_TOKENS = {
    "capsule",
    "capsules",
    "tablet",
    "tablets",
    "tab",
    "tabs",
    "syrup",
    "solution",
    "suspension",
    "injection",
    "inj",
    "ointment",
    "cream",
    "spray",
    "drop",
    "drops",
    "powder",
    "shampoo",
    "chew",
    "chews",
}


def _norm_tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _contains_ci(a: str, b: str) -> bool:
    return (b or "").lower() in (a or "").lower()


def _token_overlap_ratio(a: str, b: str) -> float:
    sa = set(_norm_tokens(a))
    sb = set(_norm_tokens(b))
    if not sa:
        return 0.0
    return len(sa & sb) / max(len(sa), 1)


def _pick_safe_kb_context(
    *,
    global_candidates: List[Dict[str, Any]],
    local_display_name: str,
    canonical_kind: str,
    search_term: Optional[str] = None,
    hints: Optional[List[str]] = None,
    hint_probabilities: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Choose a SAFE KB context concept to attach alongside a local billing match.

    Safety rules:
    - Prefer generic dosage-form KB concepts (capsule/tablet/etc.) if present.
    - Otherwise only attach if there is a strong lexical token overlap between KB preferred_name and local display name.
    - Never attach a KB concept that could create a "ghost drug" identity (e.g., Cortex -> Kredex).
    """
    if not global_candidates:
        return None

    def _norm_name(v: Optional[str]) -> str:
        return (v or "").strip().lower()

    # 0) Highest-priority specificity anchor: exact match to search_term / top-probability hint.
    # This prevents generic KB context (e.g., "test") from replacing specific intent (e.g., "Ortolani test").
    preferred_targets: List[str] = []
    st = (search_term or "").strip()
    if st:
        preferred_targets.append(st)
    top_hint = None
    try:
        top_hint = _get_highest_probability_hint(
            {"hints": hints or [], "hint_probabilities": hint_probabilities or {}}
        )
    except Exception:
        top_hint = None
    if top_hint:
        preferred_targets.append(top_hint)
    preferred_targets = [p for p in preferred_targets if p]

    if preferred_targets:
        target_set = {_norm_name(t) for t in preferred_targets}
        exact_matches = [
            cand
            for cand in global_candidates
            if _norm_name(cand.get("preferred_name")) in target_set
            or _norm_name(cand.get("display_name")) in target_set
        ]
        if exact_matches:
            exact_matches.sort(
                key=lambda c: float(c.get("hybrid_score", c.get("match_score", 0.0)) or 0.0),
                reverse=True,
            )
            return exact_matches[0]

    # 1) Prefer dosage-form context (safe, non-identity)
    for cand in global_candidates:
        pn = (cand.get("preferred_name") or "").strip()
        toks = set(_norm_tokens(pn))
        if toks & _DOSAGE_FORM_TOKENS:
            return cand

    # 2) Otherwise require strong token overlap (identity-level match)
    local_toks = {t for t in _norm_tokens(local_display_name) if len(t) >= 4 and t not in _DOSAGE_FORM_TOKENS}
    if not local_toks:
        return None
    generic_clinical_tokens = {"test", "exam", "scan", "panel", "workup", "checkup"}
    if all(t in generic_clinical_tokens for t in local_toks):
        # Avoid attaching generic concept context when local name is itself generic.
        return None

    for cand in global_candidates:
        pn = (cand.get("preferred_name") or "").strip()
        kb_toks = {t for t in _norm_tokens(pn) if len(t) >= 4 and t not in _DOSAGE_FORM_TOKENS}
        if not kb_toks:
            continue
        # Require at least one substantial shared token
        if local_toks & kb_toks:
            return cand

    return None


def calculate_display_name(
    entity: Dict[str, Any],
    best_local: Optional[Dict[str, Any]] = None,
    best_global: Optional[Dict[str, Any]] = None,  # kept for signature parity
    normalized_name: Optional[str] = None,
) -> str:
    """
    Local copy of display_name hierarchy (kept here to avoid circular imports):
    Local Truth (invoice name) > ASR-corrected name > original span text.
    """
    span_text = (entity.get("span_text") or "").strip()
    normalized = (normalized_name or entity.get("normalized_name") or "").strip()

    if entity.get("local_stock_id") or entity.get("local_service_id"):
        if best_local and (best_local.get("stock_id") or best_local.get("service_id")):
            dn = (best_local.get("display_name") or "").strip()
            if dn:
                return dn
        dn = (entity.get("kb_preferred_name") or "").strip()
        if dn:
            return dn

    if normalized and normalized.lower() != span_text.lower():
        return normalized

    return span_text


async def process_single_entity_async(
    entity: Dict[str, Any],
    idx: int,
    total: int,
    cleaned_transcript: str,
    raw_transcript: Optional[str],
    conn,  # Database connection (each entity gets its own)
    client: Optional[Any],
    clinic_id: Optional[int],
    visit_id: Optional[str],
    auto_bind_threshold: float,
    llm_judge_threshold: float,
    logger: Optional[logging.Logger],
    embedding_cache: Optional[Dict[str, List[float]]] = None,
    role_classification_future: Optional[Any] = None,  # asyncio.Future from early classify_procedure_role (streaming)
    all_entities: Optional[List[Dict[str, Any]]] = None,  # For Clinical Intent context anchors (other span_texts)
    precomputed_global: Optional[Dict[int, List[Dict[str, Any]]]] = None,  # entity_idx -> global KB candidates (batch vector search)
    grounding_collector: Optional[List] = None,  # When set, append grounding record per entity (candidates, Judge, final binding)
    precomputed_judge_results: Optional[Dict[int, Any]] = None,  # entity_idx -> precomputed LLM judge result (from parallel judge calls)
    precomputed_batch_search_judge_result: Optional[Dict[str, Any]] = None,  # Precomputed result from batch_search_and_judge_all_billable_entities (candidate dict or None)
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Process a single entity asynchronously.
    
    Returns:
        Tuple of (entity_result, presenting_request_entity)
        - entity_result: The processed entity manifest entry (or None if skipped)
        - presenting_request_entity: PresentingRequest entity dict if created, else None
    """
    try:
        if logger:
            logger.info(f"🔍 [ASYNC] PROCESSING ENTITY {idx}/{total}: '{entity.get('span_text', 'N/A')}'")
            logger.info(f"   Kind: {entity.get('kind', 'N/A')}, KB Kind: {entity.get('kb_kind', 'N/A')}, Source: {entity.get('source', 'N/A')}")
        
        span_text = entity.get("span_text", "")
        if not span_text:
            return None, None
        
        # ASR correction
        if "[unclear]" in span_text:
            normalized_name = span_text.replace("[unclear]", "").strip()
        else:
            normalized_name = span_text
        
        asr_corrected_name = sanitize_asr_errors(normalized_name)
        if asr_corrected_name != normalized_name:
            if logger:
                logger.info(f"  💡 ASR Correction (pre-routing): '{normalized_name}' → '{asr_corrected_name}'")
            normalized_name = asr_corrected_name
            if "anal gland" in normalized_name.lower() or ("expression" in normalized_name.lower() and "gland" in normalized_name.lower()):
                if entity.get("kind", "Other") == "Other":
                    if logger:
                        logger.info(f"  🔄 Re-classifying '{span_text}' from 'Other' to 'Procedure' based on ASR correction")
                    entity["kind"] = "Procedure"
                    entity["kb_kind"] = "Procedure"
        
        # Canonicalize kind (Brain NER precedence): trust entity.kind first, then legacy kb_kind.
        kb_kind_raw = entity.get("kind") or entity.get("kb_kind") or "Other"
        canonical_kind = canonicalize_kind(kb_kind_raw)
        entity_kind = entity.get("kind", "Other")
        intent_kind = entity.get("intent_kind")
        entity["canonical_kind"] = canonical_kind

        display_name = span_text  # Default

        # Role classification (if procedure-like and not already classified by super-pass)
        is_procedure_like_span = canonical_kind in ["Procedure", "Service", "Diagnostic", "Diagnosis", "ReasonForVisit"]
        role_classification = None
        billing_eligible = False
        presenting_request_entity = None
        
        # Check if roles were already provided by super-pass
        if entity.get("roles") and len(entity.get("roles", [])) > 0:
            # Super-pass already classified roles - use them
            roles = entity.get("roles", [])
            if "PresentingRequest" in roles:
                # Create PresentingRequest entity for separate processing
                presenting_request_entity = {
                    "span_text": span_text,
                    "normalized_name": normalized_name,
                    "kind": "ReasonForVisit",
                    "kb_kind": "ReasonForVisit",
                    "intent_kind": "ReasonForVisit",
                    "role_classification": {"roles": {"presenting_request": {"value": True, "confidence": 0.7}}},
                    "original_entity": entity,
                }
            billing_eligible = entity.get("is_actionable", True)
        elif is_procedure_like_span:
            # Use pre-started role classification from streaming (Grounding Prep parallelized) or run now
            if role_classification_future is not None:
                try:
                    role_classification = await role_classification_future
                except Exception as e:
                    if logger:
                        logger.debug(f"  ⚠️ Role classification future failed: {e}; running inline")
                    role_classification = None
            else:
                role_classification = None
            if role_classification is None:
                full_transcript = raw_transcript or cleaned_transcript
                context_window = extract_context_window(span_text, full_transcript, window_lines=2, window_chars=500)
                speaker = None
                if entity.get("source") == "cleaned":
                    if "Veterinarian:" in context_window or "vet:" in context_window.lower():
                        speaker = "Veterinarian"
                    elif "Pet Parent:" in context_window or "owner:" in context_window.lower():
                        speaker = "Pet Parent"
                role_classification = await asyncio.to_thread(
                    classify_procedure_role,
                    span_text=span_text,
                    context_window=context_window,
                    speaker=speaker,
                    client=client,
                    logger=logger,
                )
            
            if role_classification and isinstance(role_classification, dict):
                billing_eligible = role_classification.get("billing_eligible", False)
                roles = role_classification.get("roles", {})
                presenting_request = roles.get("presenting_request", {})
                if presenting_request.get("value", False) and presenting_request.get("confidence", 0.0) >= 0.5:
                    presenting_request_entity = {
                        "span_text": span_text,
                        "normalized_name": normalized_name,
                        "kind": "ReasonForVisit",
                        "kb_kind": "ReasonForVisit",
                        "intent_kind": "ReasonForVisit",
                        "role_classification": role_classification,
                        "original_entity": entity,
                    }
        
        # Route classification
        route = classify_entity_route(canonical_kind, entity=entity, logger=logger)

        # Grounding gate: grounding_recommended (billable/clinical?). Generic symptom/chief-complaint phrases retained as-is for clinical integrity (no grounding).
        GENERIC_SYMPTOM_PHRASES = frozenset([
            "walking problem", "difficulty walking", "difficulty in walking", "very lazy", "not eating well",
            "lazy", "not playful", "walking difficulty", "having trouble walking", "lameness", "limping",
        ])
        if route in ("dual_sync", "global_direct"):
            grounding_recommended = entity.get("grounding_recommended")
            span_lower = (span_text or "").strip().lower()
            if canonical_kind in ("ReasonForVisit", "Symptom") and span_lower in GENERIC_SYMPTOM_PHRASES:
                route = "skip_other"
                if logger:
                    logger.info(f"   ⏭️  Skip KB grounding: generic symptom/chief-complaint retained as-is for '{span_text}'")
            elif isinstance(grounding_recommended, bool) and not grounding_recommended:
                route = "skip_other"
                if logger:
                    logger.info(f"   ⏭️  Skip KB grounding: grounding_recommended=false for '{span_text}'")

        if logger:
            logger.info(f"   📍 Route Classification: {route}")
            logger.info(f"   📝 Normalized name: '{normalized_name}' (from span_text: '{span_text}')")
        
        # Context anchors for Intent Interceptor (other entity span_texts from same note)
        context_anchors: List[str] = []
        if all_entities and isinstance(all_entities, list):
            current_span = (span_text or "").strip().lower()
            for e in all_entities:
                if not isinstance(e, dict):
                    continue
                s = (e.get("span_text") or "").strip()
                if s and s.lower() != current_span and s not in context_anchors:
                    context_anchors.append(s)
                    if len(context_anchors) >= 20:
                        break

        # Precomputed global KB results from batch vector search (when available)
        precomputed_global_results = (precomputed_global or {}).get(idx - 1) if precomputed_global is not None else None
        
        # Precomputed LLM judge result from parallel judge calls (when available).
        # IMPORTANT: None is a valid precomputed value (explicit judge rejection).
        has_precomputed_judge_result = (
            precomputed_judge_results is not None and (idx - 1) in precomputed_judge_results
        )
        precomputed_judge_result = (
            precomputed_judge_results.get(idx - 1) if has_precomputed_judge_result else None
        )

        # Process based on route (this is where most I/O happens - can be parallelized)
        entity_result = await process_entity_by_route_async(
            entity=entity,
            span_text=span_text,
            normalized_name=normalized_name,
            canonical_kind=canonical_kind,
            route=route,
            role_classification=role_classification,
            billing_eligible=billing_eligible,
            conn=None,  # DB work is done inside thread-local connections (psycopg2 is not thread-safe)
            client=client,
            clinic_id=clinic_id,
            visit_id=visit_id,
            auto_bind_threshold=auto_bind_threshold,
            llm_judge_threshold=llm_judge_threshold,
            logger=logger,
            embedding_cache=embedding_cache,
            raw_transcript=raw_transcript,  # NEW: Pass raw transcript for domain-aware fallback search
            context_anchors=context_anchors,  # For Clinical Intent circuit breaker
            precomputed_global_results=precomputed_global_results,
            grounding_collector=grounding_collector,
            precomputed_judge_result=precomputed_judge_result,
            has_precomputed_judge_result=has_precomputed_judge_result,
            precomputed_batch_search_judge_result=precomputed_batch_search_judge_result,  # Pass through batch result
        )
        
        # Preserve entity metadata (hints, hint_probabilities, search_term, domain) in final result
        if entity_result:
            entity_result = _preserve_entity_metadata(entity_result, entity)
        return entity_result, presenting_request_entity
        
    except Exception as e:
        if logger:
            logger.error(f"  ❌ Error processing entity {idx} '{entity.get('span_text', 'N/A')}': {e}")
            import traceback
            logger.debug(traceback.format_exc())
        return None, None
    finally:
        pass


def _boost_search_term_to_rank1(
    candidates: List[Dict[str, Any]],
    search_term: Optional[str],
) -> List[Dict[str, Any]]:
    """When Brain provided high-certainty search_term (canonical form), move the matching candidate to rank 1 for the LLM judge. Returns a new list."""
    if not candidates or not (search_term or "").strip():
        return list(candidates)
    st = (search_term or "").strip().lower()
    if not st:
        return list(candidates)
    for i, c in enumerate(candidates):
        name = (c.get("preferred_name") or c.get("display_name") or "").strip().lower()
        if name == st or (st in name) or (name in st):
            if i == 0:
                return list(candidates)
            out = list(candidates)
            out.insert(0, out.pop(i))
            return out
    return list(candidates)


def _candidate_exact_match_search_term(candidate: Dict[str, Any], search_term: Optional[str]) -> bool:
    """True if candidate's preferred_name or display_name exactly matches search_term (case-insensitive, strip). Used for high-certainty auto-link."""
    if not (search_term or "").strip():
        return False
    st = (search_term or "").strip().lower()
    name = (candidate.get("preferred_name") or candidate.get("display_name") or "").strip().lower()
    return name == st


def _is_overly_generic_candidate_name(name: Optional[str]) -> bool:
    """True for very generic candidate labels that should not beat specific clinical terms."""
    toks = [t for t in _norm_tokens(name or "") if t]
    if not toks:
        return True
    generic = {
        "test",
        "exam",
        "examination",
        "scan",
        "panel",
        "assessment",
        "evaluation",
        "procedure",
        "service",
    }
    return all(t in generic for t in toks)


def _has_specific_anchor_text(text: Optional[str]) -> bool:
    """Heuristic: text has a specific anchor beyond generic clinical form words."""
    toks = [t for t in _norm_tokens(text or "") if t]
    if not toks:
        return False
    generic = {
        "test",
        "tests",
        "exam",
        "examination",
        "scan",
        "panel",
        "assessment",
        "evaluation",
        "procedure",
        "service",
        "xray",
        "x",
        "ray",
        "blood",
        "urine",
        "imaging",
    }
    if len(toks) >= 3 and any(t not in generic for t in toks):
        return True
    return any((len(t) >= 4 and t not in generic) for t in toks)


def _entity_certainty_float(entity: Dict[str, Any]) -> Optional[float]:
    """Parse entity certainty; returns None if missing or invalid."""
    v = entity.get("certainty")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _inject_hint_search_term_matches(
    judge_candidates: List[Dict[str, Any]],
    global_results: List[Dict[str, Any]],
    entity: Dict[str, Any],
    kind: Optional[str] = None,
    max_candidates: int = 10,
) -> List[Dict[str, Any]]:
    """Prepend candidates that match entity search_term or hints (by name), so judge always sees Brain-suggested terms. No ranking preference."""
    st = (entity.get("search_term") or "").strip().lower()
    hints = _normalize_hints(entity.get("hints"), max_items=20)
    want_names = set()
    if st:
        want_names.add(st)
    for h in hints:
        want_names.add(h.lower())
    if not want_names or not global_results:
        return list(judge_candidates)[:max_candidates]
    # Collect all candidates that match search_term or any hint (by preferred_name)
    compulsory = []
    seen_ids = set()
    for c in global_results:
        name = (c.get("preferred_name") or c.get("display_name") or "").strip().lower()
        if name and name in want_names:
            cid = c.get("concept_id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                compulsory.append(c)
    if not compulsory:
        return list(judge_candidates)[:max_candidates]
    # Prepend compulsory, then add rest (no duplicates), cap at max_candidates
    rest = [c for c in judge_candidates if c.get("concept_id") not in seen_ids]
    for c in rest:
        seen_ids.add(c.get("concept_id"))
    out = compulsory + rest[: max(0, max_candidates - len(compulsory))]
    return out[:max_candidates]


def _build_grounding_record(
    span_text: str,
    normalized_name: str,
    kind: str,
    route: str,
    local_candidates: Optional[List[Dict[str, Any]]],
    global_candidates: Optional[List[Dict[str, Any]]],
    selected: Optional[Dict[str, Any]],
    final_binding: Dict[str, Any],
    reranking_applied: bool = False,
    judge_justification: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one grounding-layer record for logging/debug: candidates, rank, LLM selection, reason, final binding."""
    candidates = []
    for rank, c in enumerate((local_candidates or [])[:15], 1):
        candidates.append({
            "rank": rank,
            "source": "local",
            "preferred_name": c.get("display_name") or c.get("preferred_name") or "",
            "concept_id": c.get("concept_id"),
            "kind": c.get("kind"),
            "domain_key": c.get("domain_key"),
            "definition": (c.get("definition") or "")[:300] if c.get("definition") else None,
            "score": float(c.get("match_score", 0) or 0),
        })
    for rank, c in enumerate((global_candidates or [])[:15], 1):
        candidates.append({
            "rank": rank,
            "source": "global",
            "preferred_name": c.get("preferred_name") or "",
            "concept_id": c.get("concept_id"),
            "kind": c.get("kind"),
            "domain_key": c.get("domain_key"),
            "definition": (c.get("definition") or "")[:300] if c.get("definition") else None,
            "score": float(c.get("hybrid_score", c.get("match_score", 0)) or 0),
        })
    llm_judge = None
    if selected:
        llm_judge = {
            "selected_display_name": selected.get("display_name") or selected.get("preferred_name"),
            "selected_concept_id": selected.get("concept_id"),
            "selected_kind": selected.get("kind"),
            "justification": judge_justification,
        }
    # Never expose "0" as display_name: use span_text or normalized_name when missing/invalid
    _dn = (final_binding.get("display_name") or "").strip()
    if not _dn or _dn == "0":
        _dn = (normalized_name or span_text or "").strip() or span_text
    return {
        "span_text": span_text,
        "normalized_name": normalized_name,
        "kind": kind,
        "route": route,
        "candidates": candidates,
        "reranking_applied": reranking_applied,
        "llm_judge": llm_judge,
        "final_binding": {
            "display_name": _dn,
            "kb_concept_id": final_binding.get("kb_concept_id"),
            "local_service_id": final_binding.get("local_service_id"),
            "local_stock_id": final_binding.get("local_stock_id"),
            "match_method": final_binding.get("match_method"),
            "similarity_score": final_binding.get("similarity_score"),
        },
    }


async def process_entity_by_route_async(
    entity: Dict[str, Any],
    span_text: str,
    normalized_name: str,
    canonical_kind: str,
    route: str,
    role_classification: Optional[Dict],
    billing_eligible: bool,
    conn,
    client: Optional[Any],
    clinic_id: Optional[int],
    visit_id: Optional[str],
    auto_bind_threshold: float,
    llm_judge_threshold: float,
    logger: Optional[logging.Logger],
    embedding_cache: Optional[Dict[str, List[float]]] = None,
    raw_transcript: Optional[str] = None,  # For domain-aware fallback search
    context_anchors: Optional[List[str]] = None,  # For Intent Interceptor (other entity span_texts from same note)
    precomputed_global_results: Optional[List[Dict[str, Any]]] = None,  # From batch vector search (avoids per-entity global lookup)
    grounding_collector: Optional[List] = None,  # When set, append one grounding record per entity (candidates, Judge, final binding)
    precomputed_judge_result: Optional[Any] = None,  # Precomputed LLM judge result (from parallel judge calls) - when provided, skip judge call
    has_precomputed_judge_result: bool = False,  # Distinguish missing value vs explicit precomputed rejection (None)
    precomputed_batch_search_judge_result: Optional[Dict[str, Any]] = None,  # Precomputed result from batch_search_and_judge_all_billable_entities
) -> Optional[Dict[str, Any]]:
    """
    Process an entity based on its route classification.
    Step 2.3 flow: Extracted Entities → Intent Interceptor (if enabled) → Search Global TopK / Local → LLM Judge.
    This function handles the I/O-bound operations (database queries, LLM calls) asynchronously.
    """
    display_name = span_text  # Default
    selected: Optional[Dict[str, Any]] = None
    suggestions: List[Dict[str, Any]] = []

    # Domain gating for local inventory/services (same as linker)
    _local_domain_filter = None
    if raw_transcript:
        try:
            from kb_ner_super_pass import detect_domain
            d = detect_domain(raw_transcript)
            if d and str(d).strip().lower() != "general":
                _local_domain_filter = d.strip()
        except Exception:
            pass

    # -------------------------
    # Thread-local DB helpers
    # psycopg2 connections/cursors are not thread-safe; never share them across asyncio.to_thread calls.
    # -------------------------
    # When NER gives Family/Other but span is tick/flea/preventive, use ParasiteControl for local search so Bravecto etc. are found
    effective_local_kind = canonical_kind
    if canonical_kind in ("Family", "Other") and entity:
        span_lower = ((entity.get("span_text") or "").strip()).lower()
        if span_lower and any(c in span_lower for c in ("tick", "flea", "flee", "bravecto", "simparica", "parasite", "deworm", "flea control", "tick control")):
            effective_local_kind = "ParasiteControl"
    
    # Extract suggestion boost parameters from entity for local search
    entity_suggestion_prob = entity.get("suggestion_probability") if isinstance(entity, dict) else None
    entity_search_term = (entity.get("search_term") or "").strip() if isinstance(entity, dict) else None
    entity_hints = _normalize_hints(entity.get("hints") if isinstance(entity, dict) else None, max_items=3)
    entity_hint_probs = entity.get("hint_probabilities") if isinstance(entity, dict) else None
    if not isinstance(entity_hint_probs, dict):
        entity_hint_probs = None
    entity_service_type = (entity.get("service_type") or "").strip() if isinstance(entity, dict) else ""
    # For dual_sync + Diagnosis: use inventory_category for local inventory, service_category for local services
    def _norm_cat_list(val: Any) -> Optional[List[str]]:
        if val is None:
            return None
        if isinstance(val, str):
            return [val.strip()] if val.strip() else None
        if isinstance(val, list):
            out = [str(c).strip() for c in val if str(c).strip()]
            return out if out else None
        return None

    # Strict mode: no fallback to legacy "category" field.
    entity_inventory_category = _norm_cat_list(entity.get("inventory_category"))
    entity_service_category = _norm_cat_list(entity.get("service_category"))

    def _with_pooled_conn(callable_fn, *args, **kwargs):
        with pg_conn_ctx(logger=logger) as c:
            return callable_fn(c, *args, **kwargs)

    entity_query_expansion = (entity.get("query_expansion") or [])[:3] if isinstance(entity.get("query_expansion"), list) else []

    def _local_inventory(term: str):
        return _with_pooled_conn(
            search_local_inventory_topk,
            term,
            effective_local_kind,
            clinic_id,
            logger,
            0.50,
            5,
            client,
            embedding_cache=embedding_cache,
            precomputed_embedding=embedding_cache.get(term) if embedding_cache else None,
            domain_filter=_local_domain_filter,
            suggestion_probability=entity_suggestion_prob,
            search_term=entity_search_term,
            hints=entity_hints,
            hint_probabilities=entity_hint_probs,
            query_expansion=entity_query_expansion or None,
            category_hints=entity_inventory_category,
        )

    def _local_services(term: str):
        return _with_pooled_conn(
            search_local_services_topk,
            term,
            effective_local_kind,
            clinic_id,
            logger,
            0.50,
            5,
            client,
            embedding_cache=embedding_cache,
            precomputed_embedding=embedding_cache.get(term) if embedding_cache else None,
            domain_filter=_local_domain_filter,
            suggestion_probability=entity_suggestion_prob,
            search_term=entity_search_term,
            hints=entity_hints,
            hint_probabilities=entity_hint_probs,
            query_expansion=entity_query_expansion or None,
            category_hints=entity_service_category,
            service_type=entity_service_type or None,
        )

    def _global_search(term: str):
        # Local-only: no global KB search
        def _call(conn_local):
            return []
        return _with_pooled_conn(_call)

    def _extract_entity_domains() -> List[str]:
        raw = entity.get("domain")
        out: List[str] = []
        if isinstance(raw, str):
            d = raw.strip().lower()
            if d and d != "general":
                out.append(d)
        elif isinstance(raw, list):
            for v in raw:
                d = str(v or "").strip().lower()
                if d and d != "general" and d not in out:
                    out.append(d)
        return out

    def _lookup_domain_consultation_candidates(domain_key: str) -> List[Dict[str, Any]]:
        """
        Targeted local fallback search:
        service category Consultation + domain_key match.
        """
        q = f"{domain_key} consultation".strip()
        pre_vec = None
        if embedding_cache:
            pre_vec = embedding_cache.get(q) or embedding_cache.get("consultation")
        return _with_pooled_conn(
            search_local_services_topk,
            q,
            canonical_kind,
            clinic_id,
            logger,
            0.30,
            5,
            client,
            embedding_cache=embedding_cache,
            precomputed_embedding=pre_vec,
            domain_filter=domain_key,
            suggestion_probability=entity_suggestion_prob,
            search_term=q,
            hints=[q, "consultation"],
            hint_probabilities=None,
            category_hints=["Consultation"],
            service_type="medical",
        )

    async def _build_domain_consultation_fallback(
        attrs: Dict[str, Any],
        current_display_name: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Plan-B fallback:
        For high-stakes unlinked entities, draft a domain consultation local service.
        """
        high_stakes = {"Procedure", "Diagnostic", "DiagnosticTest", "Surgery"}
        if canonical_kind not in high_stakes:
            return None, None
        if not clinic_id:
            return None, None
        domains = _extract_entity_domains()
        if not domains:
            return None, None
        chosen: Optional[Dict[str, Any]] = None
        chosen_domain: Optional[str] = None
        for dk in domains:
            try:
                cands = await asyncio.to_thread(_lookup_domain_consultation_candidates, dk)
            except Exception:
                cands = []
            if not cands:
                continue
            # Keep consultation-only candidates (defensive; SQL already hard-gates category).
            consultation_cands = [
                c for c in cands
                if "consult" in str(c.get("category") or "").strip().lower()
            ]
            top = (consultation_cands or cands)[0]
            if top.get("service_id"):
                chosen = top
                chosen_domain = dk
                break
        if not chosen or not chosen.get("service_id"):
            return None, None
        local_display = (chosen.get("display_name") or chosen.get("preferred_name") or current_display_name).strip()
        out_attrs = dict(attrs or {})
        source_mention = (search_term or span_text or normalized_name or "").strip()
        out_attrs["fallback_consultation"] = {
            "domain_key": chosen_domain,
            "service_id": chosen.get("service_id"),
            "service_name": local_display,
            "source_mention": source_mention,
            "dedupe_key": f"domain_consultation:{chosen_domain}:{chosen.get('service_id')}",
        }
        # Preserve NER normalized_name for Phase 2/dashboard matching (atom "Cefpodoxime syrup" → manifest); display_name = candidate for billing.
        _ner_norm = (entity.get("normalized_name") or normalized_name or "").strip() or local_display
        out = {
            "span_text": span_text,
            "normalized_name": _ner_norm,
            "display_name": local_display,
            "kind": canonical_kind,
            "kb_concept_id": chosen.get("concept_id"),
            "kb_preferred_name": local_display,
            "kb_kind": chosen.get("kind") or canonical_kind,
            "match_method": "domain_consultation_fallback",
            "similarity_score": float(chosen.get("match_score", 0) or 0),
            "local_stock_id": None,
            "local_service_id": chosen.get("service_id"),
            "assertion_id": entity.get("assertion_id", "CONF"),
            "attributes": out_attrs,
        }
        if logger:
            logger.info(
                "  🧩 Fallback SKU: '%s' -> '%s' (service_id=%s, domain=%s)",
                span_text,
                local_display,
                chosen.get("service_id"),
                chosen_domain,
            )
        return out, chosen

    def _vitals_registry(term: str):
        # NOTE: Default is trigram+phonetic only (no embedding API call) for speed.
        # If you want vector querying too, pass `client=...` and let embed_text run.
        return _with_pooled_conn(
            search_vitals_registry_topk,
            term,
            3,
            None,  # client=None to avoid extra embedding calls
            logger,
            None,  # embedding
        )
    
    # Route 1: Skip Vitals / Signalment / Identity / Non-billable (no local/global search; preserve verbatim)
    if route in ("skip_vitals", "global_vitals", "skip_signalment", "skip_identity", "skip_other", "skip_non_billable"):
        if route == "global_vitals":
            # Safe global canonicalization against kb.vitals_registry (no billing/local matching).
            attrs = entity.get("attributes", {}) or {}
            try:
                vitals_hits = await asyncio.to_thread(_vitals_registry, normalized_name or span_text)
            except Exception:
                vitals_hits = []

            best = vitals_hits[0] if vitals_hits else None
            metric_name = (best.get("metric_name") if best else None) or normalized_name or span_text
            vital_id = best.get("vital_id") if best else None

            # Keep display_name canonical for downstream UIs; preserve source_text in attributes.
            display_name = metric_name
            out = {
                "span_text": span_text,
                "normalized_name": metric_name,
                "display_name": display_name,
                "kind": "VitalSign",
                "kb_concept_id": None,  # vitals_registry is not kb.concepts
                "kb_preferred_name": metric_name,
                "kb_kind": "VitalSign",
                "match_method": "global_vitals_registry",
                "similarity_score": float(best.get("match_score", 0.0)) if best else None,
                "assertion_id": entity.get("assertion_id", "CONF"),
                "attributes": {
                    **attrs,
                    "source_text": span_text,
                    "vital_id": vital_id,
                    "vital_metric_name": metric_name,
                    "vitals_trigram_score": float(best.get("trigram_score", 0.0)) if best else 0.0,
                    "vitals_phonetic_score": float(best.get("phonetic_score", 0.0)) if best else 0.0,
                },
            }
            # CRITICAL: Preserve entity_id for anchor integrity
            if entity.get("entity_id"):
                out["entity_id"] = entity.get("entity_id")
            return out

        if route == "skip_signalment":
            # Preserve demographic/signalment blobs verbatim; never KB-link.
            display_name = span_text
            out = {
                "span_text": span_text,
                "normalized_name": normalized_name,
                "display_name": display_name,
                "kind": "Signalment",
                "kb_concept_id": None,
                "kb_preferred_name": None,
                "kb_kind": None,
                "match_method": "signalment_preserved",
                "similarity_score": None,
                "assertion_id": entity.get("assertion_id", "CONF"),
                "attributes": {
                    **(entity.get("attributes") or {}),
                    "observation_type": "Signalment",
                    "source_text": span_text,
                },
            }
            # CRITICAL: Preserve entity_id for anchor integrity
            if entity.get("entity_id"):
                out["entity_id"] = entity.get("entity_id")
            return out

        if route == "skip_identity":
            # Preserve pet/owner/doctor names verbatim; never KB-link.
            display_name = span_text
            out = {
                "span_text": span_text,
                "normalized_name": normalized_name,
                "display_name": display_name,
                "kind": "Identity",
                "kb_concept_id": None,
                "kb_preferred_name": None,
                "kb_kind": None,
                "match_method": "identity_preserved",
                "similarity_score": None,
                "assertion_id": entity.get("assertion_id", "CONF"),
                "attributes": {
                    **(entity.get("attributes") or {}),
                    "observation_type": "Identity",
                    "source_text": span_text,
                },
            }
            # CRITICAL: Preserve entity_id for anchor integrity
            if entity.get("entity_id"):
                out["entity_id"] = entity.get("entity_id")
            return out

        if route == "skip_non_billable":
            # Hard-skip: Anatomy, Symptom; never hit DB; preserve in SOAP as text only (latency + billing accuracy).
            display_name = span_text
            out = {
                "span_text": span_text,
                "normalized_name": normalized_name,
                "display_name": display_name,
                "kind": canonical_kind,
                "kb_concept_id": None,
                "kb_preferred_name": None,
                "kb_kind": None,
                "match_method": "non_billable_preserved",
                "similarity_score": None,
                "assertion_id": entity.get("assertion_id", "CONF"),
                "attributes": entity.get("attributes", {}) or {},
            }
            if entity.get("entity_id"):
                out["entity_id"] = entity.get("entity_id")
            return out

        if route == "skip_other":
            # Preserve "Other" / unrecognized kinds verbatim; only explicit NER kinds get KB grounding.
            display_name = span_text
            out = {
                "span_text": span_text,
                "normalized_name": normalized_name,
                "display_name": display_name,
                "kind": canonical_kind,
                "kb_concept_id": None,
                "kb_preferred_name": None,
                "kb_kind": None,
                "match_method": "other_preserved" if canonical_kind == "Other" else "preserved",
                "similarity_score": None,
                "assertion_id": entity.get("assertion_id", "CONF"),
                "attributes": entity.get("attributes", {}) or {},
            }
            # CRITICAL: Preserve entity_id for anchor integrity
            if entity.get("entity_id"):
                out["entity_id"] = entity.get("entity_id")
            return out

        attrs = entity.get("attributes", {})
        observation_type = attrs.get("observation_type")
        
        if is_clinical_status_constant(span_text):
            if logger:
                logger.info(f"  🛡️  Clinical Status Constant '{span_text}' - preserving verbatim")
            display_name = span_text
            out = {
                "span_text": span_text,
                "normalized_name": normalized_name,
                "display_name": display_name,
                "kind": "Status_Constant",
                "kb_concept_id": None,
                "kb_preferred_name": normalized_name,
                "kb_kind": None,
                "match_method": "status_constant_preserved",
                "is_verified": True,
                "similarity_score": 1.0,
                "assertion_id": entity.get("assertion_id", "CONF"),
                "attributes": {
                    "observation_type": "StatusConstant",
                    "source_text": span_text,
                },
            }
            # CRITICAL: Preserve entity_id for anchor integrity
            if entity.get("entity_id"):
                out["entity_id"] = entity.get("entity_id")
            return out
        else:
            # Vital sign - structure it
            display_name = normalized_name if normalized_name.lower() != span_text.lower() else span_text
            return {
                "span_text": span_text,
                "normalized_name": normalized_name,
                "display_name": display_name,
                "kind": canonical_kind,
                "kb_concept_id": None,
                "kb_preferred_name": None,
                "kb_kind": "VitalSign",
                "match_method": "vital_sign_structured",
                "similarity_score": None,
                "assertion_id": entity.get("assertion_id", "CONF"),
                "attributes": attrs,
            }
    
    # Route 2: Dual-Sync (Billable Items)
    elif route == "dual_sync":
        # OPTIMIZATION: If we have a precomputed batch search+judge result, use it directly
        # This avoids redundant search and judge calls (all billable entities processed in one batch)
        if precomputed_batch_search_judge_result is not None:
            if logger:
                logger.info(f"  ✅ Using precomputed batch search+judge result for '{span_text}'")
            
            # The batch result is the selected candidate (or None if rejected / note-only)
            selected_candidate = precomputed_batch_search_judge_result
            # 0.95 certainty wall: Diagnosis/ReasonForVisit — if Judge selected but score < 0.95, force note-only.
            if selected_candidate and canonical_kind in ("Diagnosis", "ReasonForVisit"):
                thresh_rfv = float(os.getenv("DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD", str(DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD)))
                if float(selected_candidate.get("match_score", 0) or 0) < thresh_rfv:
                    selected_candidate = None  # Force note-only (unlinked)
            if selected_candidate:
                # Build entity result from selected candidate
                # Pharmacy-Free Zone: Diagnosis/ReasonForVisit never get local_stock_id (services only).
                local_display = (selected_candidate.get("display_name") or selected_candidate.get("preferred_name") or normalized_name).strip()
                stock_id = None if is_services_only_kind(canonical_kind) else selected_candidate.get("stock_id")
                service_id = selected_candidate.get("service_id")
                match_score = float(selected_candidate.get("match_score", 0) or 0)
                # Preserve NER normalized_name for Phase 2/dashboard (atom→manifest match); display_name = candidate for billing.
                _ner_norm = (entity.get("normalized_name") or normalized_name or "").strip() or local_display
                out = {
                    "span_text": span_text,
                    "normalized_name": _ner_norm,
                    "display_name": local_display,
                    "kind": canonical_kind,
                    "kb_concept_id": selected_candidate.get("concept_id"),
                    "kb_preferred_name": local_display,
                    "kb_kind": selected_candidate.get("kind") or canonical_kind,
                    "match_method": "batch_search_judge",
                    "similarity_score": match_score,
                    "local_stock_id": stock_id if billing_eligible else None,
                    "local_service_id": service_id if billing_eligible else None,
                    "assertion_id": entity.get("assertion_id", "CONF"),
                    "attributes": entity.get("attributes", {}) or {},
                }
                if entity.get("entity_id"):
                    out["entity_id"] = entity.get("entity_id")
                if grounding_collector is not None:
                    grounding_collector.append(_build_grounding_record(
                        span_text, normalized_name, canonical_kind, route,
                        [selected_candidate], [], selected_candidate, out,
                        reranking_applied=False, judge_justification="Batch judge",
                    ))
                return out
            else:
                # Batch judge rejected or note-only (e.g. Diagnosis/ReasonForVisit below 0.95) - mark as unlinked
                display_name = normalized_name if normalized_name.lower() != span_text.lower() else span_text
                out = {
                    "span_text": span_text,
                    "normalized_name": normalized_name,
                    "display_name": display_name,
                    "kind": canonical_kind,
                    "kb_concept_id": None,
                    "kb_preferred_name": None,
                    "kb_kind": None,
                    "match_method": "batch_judge_rejected",
                    "similarity_score": None,
                    "local_stock_id": None,
                    "local_service_id": None,
                    "assertion_id": entity.get("assertion_id", "CONF"),
                    "attributes": entity.get("attributes", {}) or {},
                }
                if entity.get("entity_id"):
                    out["entity_id"] = entity.get("entity_id")
                return out
        
        # DEFAULT PATH: Use batch intent (search_term) from Super-Pass when present.
        # FALLBACK: Per-entity intent only when search_term is missing (e.g. legacy NER or chunk without batch intent).
        batch_search_term = (entity.get("search_term") or "").strip()
        search_term = batch_search_term if batch_search_term else normalized_name
        if batch_search_term:
            if logger:
                logger.info(f"  📦 Batch intent (default): using search_term '{search_term}'")
        else:
            # Fallback: Intent Interceptor (Step 2.3) when batch intent not available
            try:
                from kb_ner_intent import resolve_clinical_intent, should_trigger_intent_resolution
                from kb_ner_intent_guards import ground_clinical_terms
                if should_trigger_intent_resolution(normalized_name, canonical_kind):
                    intent_result = await asyncio.to_thread(
                        resolve_clinical_intent,
                        normalized_name,
                        context_anchors or [],
                        canonical_kind,
                        client,
                        logger,
                    )
                    if intent_result is not None:
                        intent_result = ground_clinical_terms(intent_result)
                    if intent_result is None:
                        if logger:
                            logger.info(f"  🚫 Intent Interceptor: '{span_text}' dropped (non-clinical)")
                        display_name = normalized_name if normalized_name.lower() != span_text.lower() else span_text
                        return {
                            "span_text": span_text,
                            "normalized_name": normalized_name,
                            "display_name": display_name,
                            "kind": canonical_kind,
                            "kb_concept_id": None,
                            "kb_preferred_name": None,
                            "kb_kind": canonical_kind,
                            "match_method": "intent_skip_not_clinical",
                            "similarity_score": None,
                            "assertion_id": entity.get("assertion_id", "CONF"),
                            "attributes": entity.get("attributes", {}) or {},
                        }
                    if intent_result.get("query"):
                        search_term = intent_result["query"]
                        if logger and search_term != normalized_name:
                            logger.info(f"  ✅ Intent Interceptor: '{normalized_name}' → search term '{search_term}'")
            except Exception as e:
                if logger:
                    logger.debug("  Intent Interceptor failed (using original term): %s", e)

        # LOCAL-ONLY MODE:
        # Pharmacy-Free Zone: Diagnosis and ReasonForVisit search Services ONLY (never Pharmacy/Inventory).
        # Other kinds search both inventory and services.
        services_only = is_services_only_kind(canonical_kind)
        if not clinic_id:
            if logger:
                logger.debug("  ⚠️  No clinic_id provided - skipping local inventory and local services")
            inventory_results = []
            services_results = []
            global_results = []
        else:
            if logger:
                logger.info("  🔍 Starting LOCAL-ONLY search (%s)", "services only (Diagnosis/ReasonForVisit)" if services_only else "local inventory + local services")
                logger.info(f"     Search term: '{search_term}' (ASR-corrected: {search_term != span_text})")
            if services_only:
                inventory_results = []
                services_results = (await asyncio.to_thread(_local_services, search_term)) or []
            else:
                inventory_task = asyncio.to_thread(_local_inventory, search_term)
                services_task = asyncio.to_thread(_local_services, search_term)
                inventory_results, services_results = await asyncio.gather(inventory_task, services_task)
                inventory_results = inventory_results or []
                services_results = services_results or []
            global_results = []
            # Query expansion: for Diagnosis/ReasonForVisit only run services; else run both
            qe_terms = (entity.get("query_expansion") or [])[:3]
            if qe_terms and clinic_id:
                for q in qe_terms:
                    q = (q or "").strip()
                    if not q:
                        continue
                    try:
                        if services_only:
                            inv_q = []
                            svc_q = (await asyncio.to_thread(_local_services, q)) or []
                        else:
                            inv_q, svc_q = await asyncio.gather(
                                asyncio.to_thread(_local_inventory, q),
                                asyncio.to_thread(_local_services, q),
                            )
                            inv_q = inv_q or []
                            svc_q = svc_q or []
                    except Exception:
                        inv_q, svc_q = [], []
                    seen_inv = {(c.get("stock_id"), c.get("service_id")) for c in inventory_results + services_results}
                    for c in inv_q + svc_q:
                        key = (c.get("stock_id"), c.get("service_id"))
                        if key not in seen_inv and (key[0] or key[1]):
                            seen_inv.add(key)
                            if c.get("stock_id"):
                                inventory_results.append(c)
                            else:
                                services_results.append(c)
        # Merge local inventory + services into one list sorted by final_score (includes domain + suggestion boost)
        # Both inventory and services results already have final_score calculated with boosts
        local_results = sorted(
            inventory_results + services_results,
            key=lambda x: float(x.get("final_score", x.get("match_score", 0)) or 0),
            reverse=True,
        )
        
        # Decision flow (Local-first for billing integrity).
        # If local candidates exist, we NEVER bind a billable entity to a global-only KB concept.
        best_local = local_results[0] if local_results else None
        best_global = global_results[0] if global_results else None
        reranked_this_entity = False  # Set True when optional reranker runs (for grounding record)

        # 0.95 certainty wall: Diagnosis/ReasonForVisit below threshold → note-only (no billing link).
        threshold_diag_rfv = float(os.getenv("DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD", str(DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD)))
        if canonical_kind in ("Diagnosis", "ReasonForVisit"):
            best_score = float(best_local.get("final_score", best_local.get("match_score", 0)) or 0) if best_local else 0.0
            if best_score < threshold_diag_rfv:
                display_name = normalized_name if (normalized_name or "").strip() else span_text
                out = {
                    "span_text": span_text,
                    "normalized_name": normalized_name,
                    "display_name": display_name,
                    "kind": canonical_kind,
                    "kb_concept_id": None,
                    "kb_preferred_name": None,
                    "kb_kind": canonical_kind,
                    "match_method": "note_only_below_threshold",
                    "similarity_score": None,
                    "local_stock_id": None,
                    "local_service_id": None,
                    "assertion_id": entity.get("assertion_id", "CONF"),
                    "attributes": entity.get("attributes", {}) or {},
                }
                if entity.get("entity_id"):
                    out["entity_id"] = entity.get("entity_id")
                if grounding_collector is not None:
                    grounding_collector.append(_build_grounding_record(
                        span_text, normalized_name, canonical_kind, route,
                        local_results, global_results or [], None, out,
                        reranking_applied=reranked_this_entity, judge_justification="0.95_certainty_wall",
                    ))
                return out

        # Determine display_name using hierarchy of truth
        display_name = calculate_display_name(
            entity={"span_text": span_text},
            best_local=best_local,
            best_global=best_global,
            normalized_name=normalized_name,
        )

        # Context sentence (helps judge; short window)
        context_sentence = ""
        try:
            source_for_context = raw_transcript or ""
            context_sentence = extract_context_window(source_for_context, span_text, window=120)
        except Exception:
            context_sentence = ""

        # A) Strong local match → auto-bind locally (and never global-only).
        # Low-certainty (<0.7): disable auto-link; force full multi-signal audit and LLM Judge (certainty as weight, not gate).
        cert = _entity_certainty_float(entity)
        low_certainty_skip_autobind = cert is not None and cert < float(os.getenv("CERTAINTY_LOW_THRESHOLD", "0.7"))
        local_score_for_bind = float(best_local.get("final_score", best_local.get("match_score", 0)) or 0) if best_local else 0.0
        if best_local and local_score_for_bind >= auto_bind_threshold and not low_certainty_skip_autobind:
            # High-stakes safeguard: do not force local-first if global is materially stronger.
            high_stakes_kinds = {"Procedure", "Medication", "Diagnostic", "DiagnosticTest"}
            should_skip_local_first = False
            if canonical_kind in high_stakes_kinds and global_results:
                best_global = global_results[0]
                global_score = float(best_global.get("hybrid_score", best_global.get("match_score", 0)) or 0)
                try:
                    margin = float(os.getenv("HIGH_STAKES_LOCAL_FIRST_MARGIN", "0.10"))
                except Exception:
                    margin = 0.10
                if global_score >= max(float(llm_judge_threshold or 0.55), local_score_for_bind + margin):
                    should_skip_local_first = True
                    if logger:
                        logger.info(
                            "  ⚖️ High-stakes arbitration: skip local-first for '%s' (local=%.3f, global=%.3f, margin=%.2f)",
                            span_text, local_score_for_bind, global_score, margin,
                        )
            if should_skip_local_first:
                pass
            else:
                local_display = (best_local.get("display_name") or best_local.get("preferred_name") or normalized_name).strip()
                kb_ctx = _pick_safe_kb_context(
                    global_candidates=global_results or [],
                    local_display_name=local_display,
                    canonical_kind=canonical_kind,
                    search_term=search_term,
                    hints=entity.get("hints") if isinstance(entity, dict) else None,
                    hint_probabilities=entity.get("hint_probabilities") if isinstance(entity, dict) else None,
                )
                best_inv = inventory_results[0] if inventory_results else None
                best_svc = services_results[0] if services_results else None
                thresh = float(auto_bind_threshold or 0)
                # Pharmacy-Free Zone: Diagnosis/ReasonForVisit never get local_stock_id (services only).
                stock_id = None if is_services_only_kind(canonical_kind) else (best_local.get("stock_id") or (best_inv.get("stock_id") if best_inv and float(best_inv.get("match_score", 0) or 0) >= thresh else None))
                service_id = best_local.get("service_id") or (best_svc.get("service_id") if best_svc and float(best_svc.get("match_score", 0) or 0) >= thresh else None)
                # Preserve NER normalized_name for Phase 2/dashboard (atom→manifest match); display_name = candidate for billing.
                _ner_norm = (entity.get("normalized_name") or normalized_name or "").strip() or local_display
                out = {
                    "span_text": span_text,
                    "normalized_name": _ner_norm,
                    "display_name": local_display,
                    "kind": canonical_kind,
                    # Attach SAFE KB context when available (never override local display truth)
                    "kb_concept_id": kb_ctx.get("concept_id") if kb_ctx else None,
                    "kb_preferred_name": local_display,
                    "kb_kind": kb_ctx.get("kind") if kb_ctx else canonical_kind,
                    "match_method": "dual_sync_auto_bind_local",
                    "is_verified": True,
                    "similarity_score": float(best_local.get("match_score", 0) or 0),
                    "local_stock_id": stock_id,
                    "local_service_id": service_id,
                    "assertion_id": entity.get("assertion_id", "CONF"),
                    "attributes": entity.get("attributes", {}),
                }
                if grounding_collector is not None:
                    grounding_collector.append(_build_grounding_record(
                        span_text, normalized_name, canonical_kind, route,
                        local_results, global_results or [], best_local, out,
                        reranking_applied=reranked_this_entity, judge_justification=None,
                    ))
                return out

        # B) Local candidates exist → apply full decision flow (deterministic gates → judge).
        if local_results:
            # IMPORTANT: rank locals by final_score so domain + suggestion boost are actually used.
            local_sorted = sorted(
                local_results,
                key=lambda x: float(x.get("final_score", x.get("match_score", 0)) or 0),
                reverse=True,
            )
            # Feed boosted score into decision flow as match_score.
            local_sorted = [
                {
                    **c,
                    "match_score": float(c.get("final_score", c.get("match_score", 0)) or 0),
                }
                for c in local_sorted
            ]
            # Certainty boost: if Brain gave high certainty that search_term is correct intent, put that candidate at rank 1 for the judge
            try:
                cert_val = entity.get("certainty")
                if cert_val is not None and float(cert_val) >= float(os.getenv("CERTAINTY_BOOST_RANK1_THRESHOLD", "0.90")):
                    st = (search_term or "").strip()
                    if st:
                        local_sorted = _boost_search_term_to_rank1(local_sorted, st)
                        if global_results:
                            global_results = _boost_search_term_to_rank1(global_results, st)
                        if logger:
                            logger.info("  📌 Certainty boost: search_term '%s' moved to rank 1 for judge (certainty=%.2f)", st[:40], float(cert_val))
            except (TypeError, ValueError):
                pass

            # High-Certainty Fast Track (Signal Convergence): if KB has exact match for search_term, auto-link and skip judge
            try:
                cert_val = _entity_certainty_float(entity)
                if cert_val is not None and cert_val >= float(os.getenv("CERTAINTY_BOOST_RANK1_THRESHOLD", "0.90")):
                    st = (search_term or "").strip()
                    if st:
                        for cand in local_sorted:
                            if _candidate_exact_match_search_term(cand, st):
                                local_display = (cand.get("display_name") or cand.get("preferred_name") or normalized_name).strip()
                                kb_ctx = _pick_safe_kb_context(
                                    global_candidates=global_results or [],
                                    local_display_name=local_display,
                                    canonical_kind=canonical_kind,
                                    search_term=search_term,
                                    hints=entity.get("hints") if isinstance(entity, dict) else None,
                                    hint_probabilities=entity.get("hint_probabilities") if isinstance(entity, dict) else None,
                                )
                                # Pharmacy-Free Zone: Diagnosis/ReasonForVisit never get local_stock_id (services only).
                                stock_id = None if is_services_only_kind(canonical_kind) else cand.get("stock_id")
                                service_id = cand.get("service_id")
                                _ner_norm = (entity.get("normalized_name") or normalized_name or "").strip() or local_display
                                out = {
                                    "span_text": span_text,
                                    "normalized_name": _ner_norm,
                                    "display_name": local_display,
                                    "kind": canonical_kind,
                                    "kb_concept_id": kb_ctx.get("concept_id") if kb_ctx else None,
                                    "kb_preferred_name": local_display,
                                    "kb_kind": kb_ctx.get("kind") if kb_ctx else canonical_kind,
                                    "match_method": "dual_sync_high_certainty_auto_link_local",
                                    "is_verified": True,
                                    "similarity_score": float(cand.get("match_score", 0) or 0),
                                    "local_stock_id": stock_id,
                                    "local_service_id": service_id,
                                    "assertion_id": entity.get("assertion_id", "CONF"),
                                    "attributes": entity.get("attributes", {}),
                                }
                                if grounding_collector is not None:
                                    grounding_collector.append(_build_grounding_record(
                                        span_text, normalized_name, canonical_kind, route,
                                        local_sorted, global_results or [], cand, out,
                                        reranking_applied=reranked_this_entity, judge_justification="high_certainty_auto_link",
                                    ))
                                if logger:
                                    logger.info("  ✅ High-certainty auto-link (local): '%s' → '%s' (exact match, certainty=%.2f)", span_text, local_display, cert_val)
                                return out
                        for cand in (global_results or []):
                            if _candidate_exact_match_search_term(cand, st):
                                out = {
                                    "span_text": span_text,
                                    "normalized_name": normalized_name,
                                    "display_name": cand.get("preferred_name", display_name),
                                    "kind": canonical_kind,
                                    "kb_concept_id": cand.get("concept_id"),
                                    "kb_preferred_name": cand.get("preferred_name", normalized_name),
                                    "kb_kind": cand.get("kind"),
                                    "match_method": "dual_sync_high_certainty_auto_link_global",
                                    "similarity_score": float(cand.get("hybrid_score", cand.get("match_score", 0)) or 0),
                                    "local_stock_id": None,
                                    "local_service_id": None,
                                    "assertion_id": entity.get("assertion_id", "CONF"),
                                    "attributes": entity.get("attributes", {}),
                                }
                                if grounding_collector is not None:
                                    grounding_collector.append(_build_grounding_record(
                                        span_text, normalized_name, canonical_kind, route,
                                        local_sorted, global_results or [], cand, out,
                                        reranking_applied=reranked_this_entity, judge_justification="high_certainty_auto_link",
                                    ))
                                if logger:
                                    logger.info("  ✅ High-certainty auto-link (global): '%s' → '%s' (exact match, certainty=%.2f)", span_text, cand.get("preferred_name"), cert_val)
                                return out
            except (TypeError, ValueError):
                pass

            # Extract contextual information from entity
            entity_section = entity.get("section")
            # Note: species/breed/condition would need to be extracted from all_entities
            # For now, we pass what's available in the entity dict
            
            # OPTIMIZATION: Build suggestions in parallel with LLM Judge decision
            # This eliminates serial wait time - suggestions are ready when judge finishes
            async def _build_suggestions_async():
                """Build suggestions array while judge is running"""
                suggestions = []
                # First, add local candidates (billing items)
                if local_results and len(local_results) > 0:
                    for cand in local_results[:5]:
                        cand_name = cand.get("display_name") or cand.get("preferred_name") or ""
                        suggestion = {
                            "name": cand_name,
                            "match_score": float(cand.get("match_score", 0) or 0),
                            "recommendation": "MEDIUM",
                        }
                        if cand.get("service_id"):
                            suggestion["service_id"] = cand.get("service_id")
                        if cand.get("stock_id"):
                            suggestion["inventory_id"] = cand.get("stock_id")
                            suggestion["stock_id"] = cand.get("stock_id")
                        if cand.get("concept_id"):
                            suggestion["kb_concept_id"] = cand.get("concept_id")
                        suggestions.append(suggestion)
                
                # Also add global KB candidates (clinical context) - fetch definitions in parallel
                if global_results and len(global_results) > 0:
                    for cand in global_results[:15]:  # Top 15 global candidates (Ortolani at rank 12)
                        # Skip if already in suggestions (avoid duplicates)
                        cand_name = cand.get("preferred_name") or ""
                        if any(s.get("name", "").lower() == cand_name.lower() for s in suggestions):
                            continue
                        cand_score = float(cand.get("hybrid_score", cand.get("match_score", 0)) or 0)
                        
                        suggestion = {
                            "name": cand_name,
                            "match_score": cand_score,
                            "recommendation": "MEDIUM",
                            "kb_concept_id": cand.get("concept_id"),
                        }
                        # Include KB definition if available
                        if cand.get("definition"):
                            suggestion["definition"] = cand.get("definition")
                        # If definition missing but we have concept_id, fetch it asynchronously
                        elif cand.get("concept_id") and conn and not LOCAL_ONLY:
                            try:
                                # Use thread pool for DB query (non-blocking); skipped when LOCAL_ONLY (no kb.concepts)
                                def _fetch_definition():
                                    try:
                                        with conn.cursor() as cur:
                                            cur.execute("SELECT definition FROM kb.concepts WHERE concept_id = %s", (cand.get("concept_id"),))
                                            row = cur.fetchone()
                                            return row[0] if row and row[0] else None
                                    except Exception:
                                        return None
                                definition = await asyncio.to_thread(_fetch_definition)
                                if definition:
                                    suggestion["definition"] = definition
                            except Exception:
                                pass  # Ignore errors fetching definition
                        suggestions.append(suggestion)
                
                return suggestions
            
            # Use precomputed judge result if available (from parallel judge calls), otherwise run judge decision and suggestion building in parallel.
            # Use search_term (ASR-corrected, e.g. Spirocoxib) so Judge can accept local firocoxib match.
            # TIERED VERIFICATION: Only run LLM judge for high-stakes kinds (Procedure, Medication, Diagnostic)
            # Skip judge for low-stakes kinds (ReasonForVisit, Reminder, Diet, ParasiteControl)
            HIGH_STAKES_JUDGE_KINDS = {"Procedure", "Medication", "Diagnostic"}
            should_run_judge = canonical_kind in HIGH_STAKES_JUDGE_KINDS
            
            if has_precomputed_judge_result:
                # Use precomputed judge result (from parallel judge calls).
                # Note: explicit rejection is represented as None and must NOT trigger a second judge call.
                selected = precomputed_judge_result
                suggestions = await _build_suggestions_async()
                if logger:
                    if selected is None:
                        logger.info(f"  ⚖️ Using precomputed judge rejection for '{span_text}' (skip duplicate judge)")
                    else:
                        logger.info(f"  ⚖️ Using precomputed judge result for '{span_text}'")
            elif should_run_judge:
                # Run judge decision and suggestion building in parallel (only for high-stakes kinds)
                _hints = _normalize_hints(entity.get("hints") if isinstance(entity, dict) else None, max_items=5)
                _qe = (entity.get("query_expansion") or [])[:5] if isinstance(entity.get("query_expansion"), list) else []
                judge_task = asyncio.to_thread(
                    apply_decision_flow,
                    mention=span_text,
                    search_term_used=search_term,
                    local_candidates=local_sorted,
                    entity_kind=canonical_kind,
                    context_sentence=context_sentence,
                    assessment_context=(entity.get("assessment_context") if isinstance(entity, dict) else None),
                    client=client,
                    logger=logger,
                    global_candidates=global_results or [],
                    auto_bind_threshold=auto_bind_threshold,
                    species=entity.get("species"),
                    breed=entity.get("breed"),
                    suspected_condition=entity.get("suspected_condition"),
                    section=entity_section,
                    hints=_hints,
                    query_expansion=_qe,
                )
                suggestions_task = _build_suggestions_async()
                # Wait for both to complete
                selected, suggestions = await asyncio.gather(judge_task, suggestions_task)
            else:
                # TIERED VERIFICATION: Skip LLM judge for low-stakes kinds (ReasonForVisit, Reminder, Diet, ParasiteControl)
                # Use deterministic selection: best local match if available, else best global match
                # NOTE: Boosts are already included in scores:
                # - Local candidates: domain + suggestion boost applied in search functions, stored in final_score (fallback to match_score)
                # - Global candidates: domain + suggestion boost applied in batch global search, stored in match_score and hybrid_score
                suggestions = await _build_suggestions_async()
                if local_sorted:
                    # Prefer local match for billing integrity
                    # Check final_score first (includes domain boost), then match_score
                    best_local = local_sorted[0]
                    local_score = float(best_local.get("final_score", best_local.get("match_score", 0)) or 0)
                    if local_score >= float(llm_judge_threshold or 0.55):
                        selected = best_local
                        if logger:
                            logger.info(f"  ⏭️ Skipped LLM judge for low-stakes kind '{canonical_kind}' '{span_text}' → selected best local match: '{best_local.get('display_name') or best_local.get('preferred_name')}' (score: {local_score:.3f}, includes domain + suggestion boost)")
                    else:
                        # Safety: never auto-fallback ParasiteControl to global-only when local is weak.
                        # This prevents clinically absurd substitutions like "tick and flea control" -> "car seat belt".
                        if canonical_kind == "ParasiteControl":
                            if logger:
                                logger.info(
                                    "  🛡️ Category guard: ParasiteControl local match below threshold (%.3f); skipping global auto-fallback for '%s'",
                                    local_score, span_text,
                                )
                            selected = None
                            # keep suggestions only; downstream will preserve as unlinked if judge is skipped
                            pass
                        else:
                            # Fall back to global if local match is too weak
                            if global_results:
                                best_global = global_results[0]
                                # Global candidates: hybrid_score includes domain + suggestion boost, match_score also includes boosts
                                global_score = float(best_global.get("hybrid_score", best_global.get("match_score", 0)) or 0)
                                if global_score >= float(llm_judge_threshold or 0.55):
                                    selected = best_global
                                    if logger:
                                        logger.info(f"  ⏭️ Skipped LLM judge for low-stakes kind '{canonical_kind}' '{span_text}' → selected best global match: '{best_global.get('preferred_name')}' (score: {global_score:.3f}, includes domain + suggestion boost)")
                elif global_results:
                    # No local candidates, use best global
                    best_global = global_results[0]
                    # Global candidates: hybrid_score includes domain + suggestion boost, match_score also includes boosts
                    global_score = float(best_global.get("hybrid_score", best_global.get("match_score", 0)) or 0)
                    if global_score >= float(llm_judge_threshold or 0.55):
                        selected = best_global
                        if logger:
                            logger.info(f"  ⏭️ Skipped LLM judge for low-stakes kind '{canonical_kind}' '{span_text}' → selected best global match: '{best_global.get('preferred_name')}' (score: {global_score:.3f}, includes domain + suggestion boost)")
                if not selected and logger:
                    logger.debug(f"  ⏭️ Skipped LLM judge for low-stakes kind '{canonical_kind}' '{span_text}' → no match above threshold")

            # FHO fix: when mention is "FHO" (or Procedure with FHO in span), bind to best local candidate that is clearly FHO (SURGERY-FHO, Femoral Head) even if judge returned NONE_GENERIC
            if not selected and local_sorted:
                span_upper = (span_text or "").strip().upper()
                is_fho_mention = span_upper == "FHO" or (canonical_kind == "Procedure" and "FHO" in span_upper)
                if is_fho_mention:
                    fho_candidates = [
                        c for c in local_sorted
                        if ("FHO" in ((c.get("preferred_name") or c.get("display_name") or "").upper())
                         or "FEMORAL HEAD" in ((c.get("preferred_name") or c.get("display_name") or "").upper()))
                    ]
                    if fho_candidates:
                        best_fho = max(fho_candidates, key=lambda c: float(c.get("match_score", 0) or 0))
                        selected = best_fho
                        if logger:
                            logger.info("  ✅ FHO override: binding '%s' to local candidate '%s' (judge had rejected as generic)", span_text, best_fho.get("preferred_name") or best_fho.get("display_name"))

            if selected:
                # 0.95 certainty wall: Diagnosis/ReasonForVisit — if Judge selected but score < 0.95, force note-only.
                sel_score = float(selected.get("match_score", 0) or 0)
                if canonical_kind in ("Diagnosis", "ReasonForVisit") and sel_score < threshold_diag_rfv:
                    selected = None  # Fall through to unlinked (note-only)
                    if logger:
                        logger.info(f"  📋 0.95 wall: Diagnosis/ReasonForVisit '{span_text}' score {sel_score:.3f} < {threshold_diag_rfv} → note-only (no link)")
                else:
                    pass  # proceed to build out from selected
            if selected:
                local_display = (selected.get("display_name") or selected.get("preferred_name") or normalized_name).strip()
                kb_ctx = _pick_safe_kb_context(
                    global_candidates=global_results or [],
                    local_display_name=local_display,
                    canonical_kind=canonical_kind,
                    search_term=search_term,
                    hints=entity.get("hints") if isinstance(entity, dict) else None,
                    hint_probabilities=entity.get("hint_probabilities") if isinstance(entity, dict) else None,
                )
                # Attach both inventory and service ID when both sources have a match (kind does not gate).
                # Pharmacy-Free Zone: Diagnosis/ReasonForVisit never get local_stock_id (services only).
                best_inv = inventory_results[0] if inventory_results else None
                best_svc = services_results[0] if services_results else None
                thresh = float(llm_judge_threshold or 0.55)
                stock_id = None if is_services_only_kind(canonical_kind) else (selected.get("stock_id") or (best_inv.get("stock_id") if best_inv and float(best_inv.get("match_score", 0) or 0) >= thresh else None))
                service_id = selected.get("service_id") or (best_svc.get("service_id") if best_svc and float(best_svc.get("match_score", 0) or 0) >= thresh else None)
                _ner_norm = (entity.get("normalized_name") or normalized_name or "").strip() or local_display
                out = {
                    "span_text": span_text,
                    "normalized_name": _ner_norm,
                    "display_name": local_display,
                    "kind": canonical_kind,
                    "kb_concept_id": kb_ctx.get("concept_id") if kb_ctx else None,
                    "kb_preferred_name": local_display,
                    "kb_kind": kb_ctx.get("kind") if kb_ctx else canonical_kind,
                    "match_method": "dual_sync_local",
                    "is_verified": True,
                    "similarity_score": float(selected.get("match_score", 0) or 0),
                    "local_stock_id": stock_id,
                    "local_service_id": service_id,
                    "assertion_id": entity.get("assertion_id", "CONF"),
                    "attributes": entity.get("attributes", {}),
                }
                if grounding_collector is not None:
                    grounding_collector.append(_build_grounding_record(
                        span_text, normalized_name, canonical_kind, route,
                        local_sorted, global_results or [], selected, out,
                        reranking_applied=reranked_this_entity, judge_justification=None,
                    ))
                return out

            # Local candidates existed but decision flow rejected them → preserve unlinked (no ghost drugs)
            if logger:
                logger.warning(
                    f"  🚫 Dual-Sync: local candidates rejected for '{span_text}'. "
                    f"Preserving as unlinked (preventing global-only hallucination)."
                )
            
            # Suggestions already built in parallel with judge (above) - use them. When local exists, attach
            # primary_local_suggestion so UI can show suggestions with IDs even when judge rejected.
            # NOTE: When LLM judge explicitly rejects a match, we do NOT replace display_name with highest probability hint.
            # The judge's rejection is intentional - we preserve the original normalized_name/span_text.
            display_name = normalized_name if normalized_name.lower() != span_text.lower() else span_text
            attrs = entity.get("attributes", {}) or {}
            if suggestions:
                attrs["suggestions"] = suggestions
                if logger:
                    logger.info(f"  ✅ Captured {len(suggestions)} suggestions for '{span_text}': {[s.get('name') for s in suggestions]}")
            top_local = local_sorted[0] if local_sorted else None
            if top_local:
                attrs["primary_local_suggestion"] = {
                    "name": top_local.get("display_name") or top_local.get("preferred_name") or "",
                    "stock_id": top_local.get("stock_id"),
                    "service_id": top_local.get("service_id"),
                }
            out = {
                "span_text": span_text,
                "normalized_name": normalized_name,
                "display_name": display_name,
                "kind": canonical_kind,
                "kb_concept_id": None,
                "kb_preferred_name": None,
                "kb_kind": canonical_kind,
                "match_method": "dual_sync_judge_rejected",
                "similarity_score": None,
                # Judge rejected -> entity must remain truly unlinked for billing/injection.
                "local_stock_id": None,
                "local_service_id": None,
                "assertion_id": entity.get("assertion_id", "CONF"),
                "attributes": attrs,
            }
            fallback_out, fallback_selected = await _build_domain_consultation_fallback(attrs, display_name)
            if fallback_out is not None:
                if grounding_collector is not None:
                    grounding_collector.append(_build_grounding_record(
                        span_text, normalized_name, canonical_kind, route,
                        local_sorted, global_results or [], fallback_selected, fallback_out,
                        reranking_applied=reranked_this_entity, judge_justification="domain_consultation_fallback",
                    ))
                return fallback_out
            if grounding_collector is not None:
                grounding_collector.append(_build_grounding_record(
                    span_text, normalized_name, canonical_kind, route,
                    local_sorted, global_results or [], None, out,
                    reranking_applied=reranked_this_entity, judge_justification=None,
                ))
            return out

        # C) No local candidates → allow global binding (clinical-only fallback).
        # ACTION family: For Procedure/Diagnostic when Brain provided search_term or hints (Diagnostic hint),
        # use a lower threshold so intent-corrected matches (e.g. ultralining → Ortolani Test) link reliably.
        effective_global_threshold = llm_judge_threshold
        if canonical_kind in ("Procedure", "DiagnosticTest", "Service"):
            has_hint = bool((entity.get("search_term") or "").strip() or (entity.get("hints") or []))
            if has_hint:
                try:
                    effective_global_threshold = float(os.getenv("PROCEDURE_DIAGNOSTIC_HINT_GLOBAL_THRESHOLD", "0.50"))
                except Exception:
                    effective_global_threshold = 0.50
                if logger:
                    logger.info(f"  📌 Procedure/Diagnostic with intent hint: using global threshold {effective_global_threshold} for '{span_text}'")

        # C) Global-only: run LLM Judge with top global candidates (definitions + domain).
        # Compulsorily add candidates that match entity search_term or hints (by name) so judge always sees Brain-suggested terms (e.g. Ortolani, lameness).
        judge_candidates = (global_results or [])[:8]
        judge_candidates = _inject_hint_search_term_matches(
            judge_candidates, global_results or [], entity, canonical_kind, max_candidates=10
        )
        # Specificity guardrail: if mention/search_term/hints are specific, drop purely generic candidates
        # when at least one specific candidate exists.
        specific_anchor_present = _has_specific_anchor_text(search_term or span_text) or any(
            _has_specific_anchor_text(h) for h in (entity.get("hints") or [])
        )
        if specific_anchor_present and judge_candidates:
            specific_only = [
                c
                for c in judge_candidates
                if not _is_overly_generic_candidate_name(c.get("preferred_name") or c.get("display_name"))
            ]
            if specific_only:
                judge_candidates = specific_only
                if logger:
                    logger.info("  🛡️ Specificity guard: filtered generic global candidates for '%s'", span_text)
        cert_global = _entity_certainty_float(entity)
        try:
            if cert_global is not None and cert_global >= float(os.getenv("CERTAINTY_BOOST_RANK1_THRESHOLD", "0.90")):
                st = (search_term or "").strip()
                if st and judge_candidates:
                    judge_candidates = _boost_search_term_to_rank1(judge_candidates, st)
                    if logger:
                        logger.info("  📌 Certainty boost (global-only): search_term '%s' moved to rank 1 for judge (certainty=%.2f)", st[:40], cert_global)
        except (TypeError, ValueError):
            pass
        # High-Certainty Fast Track: if first candidate exactly matches search_term, auto-link and skip judge
        if judge_candidates and cert_global is not None and cert_global >= float(os.getenv("CERTAINTY_BOOST_RANK1_THRESHOLD", "0.90")):
            st = (search_term or "").strip()
            if st and _candidate_exact_match_search_term(judge_candidates[0], st):
                selected = judge_candidates[0]
                out = {
                    "span_text": span_text,
                    "normalized_name": normalized_name,
                    "display_name": selected.get("preferred_name", display_name),
                    "kind": canonical_kind,
                    "kb_concept_id": selected.get("concept_id"),
                    "kb_preferred_name": selected.get("preferred_name", normalized_name),
                    "kb_kind": selected.get("kind"),
                    "match_method": "dual_sync_global_high_certainty_auto_link",
                    "similarity_score": float(selected.get("hybrid_score", 0) or 0),
                    "local_stock_id": None,
                    "local_service_id": None,
                    "assertion_id": entity.get("assertion_id", "CONF"),
                    "attributes": entity.get("attributes", {}),
                }
                if grounding_collector is not None:
                    grounding_collector.append(_build_grounding_record(
                        span_text, normalized_name, canonical_kind, route,
                        local_results or [], global_results or [], selected, out,
                        reranking_applied=reranked_this_entity, judge_justification="high_certainty_auto_link",
                    ))
                if logger:
                    logger.info("  ✅ High-certainty auto-link (global-only): '%s' → '%s' (exact match, certainty=%.2f)", span_text, selected.get("preferred_name"), cert_global)
                return out
        # Specificity guardrail (global-only): when Brain provided search_term/hints and we have an exact-name
        # candidate that is specific (not generic "test/exam"), prefer it directly before LLM Judge.
        # This prevents cases like "ultralining test" being downgraded to generic "test".
        if judge_candidates and canonical_kind in ("Procedure", "Diagnostic", "DiagnosticTest", "Service"):
            exact_targets = set()
            st = (search_term or "").strip().lower()
            if st:
                exact_targets.add(st)
            for h in (entity.get("hints") or []):
                hs = (h or "").strip().lower()
                if hs:
                    exact_targets.add(hs)
            if exact_targets:
                exact_specific = []
                for cand in judge_candidates:
                    name = (cand.get("preferred_name") or cand.get("display_name") or "").strip()
                    if not name:
                        continue
                    n_l = name.lower()
                    if n_l in exact_targets and (not _is_overly_generic_candidate_name(name)):
                        exact_specific.append(cand)
                if exact_specific:
                    exact_specific.sort(
                        key=lambda c: float(c.get("hybrid_score", c.get("match_score", 0)) or 0),
                        reverse=True,
                    )
                    top_exact = exact_specific[0]
                    top_exact_score = float(top_exact.get("hybrid_score", top_exact.get("match_score", 0)) or 0)
                    exact_min = float(os.getenv("GLOBAL_EXACT_HINT_MIN_SCORE", "0.75"))
                    if top_exact_score >= exact_min:
                        out = {
                            "span_text": span_text,
                            "normalized_name": normalized_name,
                            "display_name": top_exact.get("preferred_name", display_name),
                            "kind": canonical_kind,
                            "kb_concept_id": top_exact.get("concept_id"),
                            "kb_preferred_name": top_exact.get("preferred_name", normalized_name),
                            "kb_kind": top_exact.get("kind"),
                            "match_method": "dual_sync_global_exact_hint_auto_link",
                            "similarity_score": top_exact_score,
                            "local_stock_id": None,
                            "local_service_id": None,
                            "assertion_id": entity.get("assertion_id", "CONF"),
                            "attributes": entity.get("attributes", {}),
                        }
                        if grounding_collector is not None:
                            grounding_collector.append(_build_grounding_record(
                                span_text, normalized_name, canonical_kind, route,
                                local_results or [], global_results or [], top_exact, out,
                                reranking_applied=reranked_this_entity, judge_justification="exact_hint_specificity_guard",
                            ))
                        if logger:
                            logger.info(
                                "  ✅ Specificity guard: global exact hint auto-link '%s' → '%s' (score=%.3f, skip judge)",
                                span_text, top_exact.get("preferred_name"), top_exact_score,
                            )
                return out
        # Run judge when: (a) best score meets threshold, or (b) low certainty — always audit via judge (no auto-link by score alone).
        low_cert_force_judge = cert_global is not None and cert_global < float(os.getenv("CERTAINTY_LOW_THRESHOLD", "0.7"))
        score_ok = best_global and float(best_global.get("hybrid_score", 0) or 0) >= effective_global_threshold
        judge_min_floor = float(os.getenv("GLOBAL_JUDGE_MIN_CANDIDATE_SCORE", "0.35"))
        top_judge_score = max(
            (float(c.get("hybrid_score", c.get("match_score", 0)) or 0) for c in (judge_candidates or [])),
            default=0.0,
        )
        if judge_candidates and top_judge_score < judge_min_floor:
            if logger:
                logger.info(
                    "  🚫 Fallback guardrail: skip global judge for '%s' (top_score=%.3f < floor=%.2f)",
                    span_text, top_judge_score, judge_min_floor,
                )
            judge_candidates = []
        if judge_candidates and (score_ok or low_cert_force_judge):
            # Log Brain suggestions (search_term, hints) and all candidates with definitions for debugging (e.g. Ortolani vs insonate)
            if logger:
                logger.info(
                    "  📋 Brain suggestions for '%s': search_term=%r, hints=%s",
                    span_text,
                    (entity.get("search_term") or "").strip() or None,
                    entity.get("hints") or [],
                )
                for i, cand in enumerate(judge_candidates, 1):
                    defn = (cand.get("definition") or "")[:150]
                    logger.info(
                        "  📋 Judge candidate #%s: %s (id=%s, kind=%s, domain=%s, score=%.3f) defn=%r",
                        i, cand.get("preferred_name", "Unknown"), cand.get("concept_id"), cand.get("kind"),
                        cand.get("domain_key") or "N/A", float(cand.get("hybrid_score", 0) or 0), (defn + "...") if len((cand.get("definition") or "")) > 150 else defn or "",
                    )
            _detected_domain = None
            if raw_transcript:
                try:
                    from kb_ner_domain import detect_domain
                    _detected_domain = detect_domain(raw_transcript)
                    if _detected_domain == "general":
                        _detected_domain = None
                except Exception:
                    pass
            judge_return_justification = os.getenv("LLM_JUDGE_RETURN_JUSTIFICATION", "").strip().lower() in ("1", "true", "yes")
            # Judge gets word, definition, context, domain — no scores or ranking preference
            _context_for_judge = (context_sentence or "")[:300] if context_sentence else ""
            judge_prompt = f"""You are a veterinary clinical entity linker. Select the BEST match for the mention below. Do not use scores or ranking — use only the word, definition, context, and domain.

MENTION: "{span_text}"
ENTITY KIND: {canonical_kind}
{"CLINICAL DOMAIN: " + _detected_domain if _detected_domain else ""}
{"CONTEXT: " + _context_for_judge if _context_for_judge else ""}

CANDIDATES (number, name, definition, domain only):
"""
            for i, cand in enumerate(judge_candidates, 1):
                domain_info = cand.get("domain_key") or "N/A"
                defn = (cand.get("definition") or "")[:200]
                defn_line = f" Definition: {defn}..." if defn else " (no definition)"
                judge_prompt += f"{i}. {cand.get('preferred_name', 'Unknown')} |{defn_line} | Domain: {domain_info}\n"
            if _detected_domain:
                judge_prompt += f"""
DOMAIN MATCHING (Clinical Plausibility Gate):
- The conversation domain is "{_detected_domain}".
- If a candidate's domain matches "{_detected_domain}", it is much more likely correct than a candidate from a different domain.
- Example: domain "orthopedic" → prefer "Ortolani test" over "otoendoscopic exam" (dermatology/ear).
"""
            if low_cert_force_judge:
                judge_prompt += """
LOW-CERTAINTY AUDIT (extraction layer uncertain about this intent):
- Prioritize character-level similarity (Trigram/Phonetic) from the raw mention over the LLM's suggested search_term.
- Prefer candidates that match the spoken form (e.g. "noble" → Norberg) when domain supports it; do not over-trust the suggested search_term.
"""
            if judge_return_justification:
                judge_prompt += f"\nReturn exactly two lines: Line 1 = candidate number (1-{len(judge_candidates)}) or 0. Line 2 = JUSTIFICATION: <one short sentence explaining why>."
            else:
                judge_prompt += f"\nReturn ONLY the candidate number (1-{len(judge_candidates)}) or 0 if none match. No explanation."
            try:
                from kb_ner_clients import get_client_for_model
                judge_model = os.getenv("LLM_JUDGE_MODEL", "gpt-4.1-nano").strip()
                judge_client, _ = get_client_for_model(judge_model, logger)
                if judge_client:
                    judge_resp = await asyncio.to_thread(
                        lambda: judge_client.chat.completions.create(
                            model=judge_model,
                            messages=[
                                {"role": "system", "content": "You are a veterinary clinical entity linker. Return only a number." + (" Optionally add a second line: JUSTIFICATION: <sentence>." if judge_return_justification else "")},
                                {"role": "user", "content": judge_prompt},
                            ],
                            temperature=0.0,
                            max_tokens=80 if judge_return_justification else 10,
                        )
                    )
                    judge_text = (judge_resp.choices[0].message.content or "").strip()
                    judge_justification = None
                    if judge_return_justification and "\n" in judge_text:
                        first_line, rest = judge_text.split("\n", 1)
                        judge_text = first_line.strip()
                        if "JUSTIFICATION:" in rest or "justification:" in rest.lower():
                            judge_justification = rest.split(":", 1)[-1].strip() if ":" in rest else rest.strip()
                    judge_choice = int(judge_text.split()[0]) if judge_text.split() else 0
                    if logger and judge_justification:
                        logger.info("  ⚖️ LLM Judge (global-only) justification for '%s': %s", span_text, judge_justification)
                    if 1 <= judge_choice <= len(judge_candidates):
                        selected = judge_candidates[judge_choice - 1]
                        if logger:
                            logger.info(f"  ⚖️ LLM Judge (global-only) selected #{judge_choice} for '{span_text}' → '{selected.get('preferred_name')}'")
                        out = {
                            "span_text": span_text,
                            "normalized_name": normalized_name,
                            "display_name": selected.get("preferred_name", display_name),
                            "kind": canonical_kind,
                            "kb_concept_id": selected.get("concept_id"),
                            "kb_preferred_name": selected.get("preferred_name", normalized_name),
                            "kb_kind": selected.get("kind"),
                            "match_method": "dual_sync_global",
                            "similarity_score": float(selected.get("hybrid_score", 0) or 0),
                            "local_stock_id": None,
                            "local_service_id": None,
                            "assertion_id": entity.get("assertion_id", "CONF"),
                            "attributes": entity.get("attributes", {}),
                        }
                        if grounding_collector is not None:
                            grounding_collector.append(_build_grounding_record(
                                span_text, normalized_name, canonical_kind, route,
                                local_results or [], global_results or [], selected, out,
                                reranking_applied=reranked_this_entity, judge_justification=judge_justification,
                            ))
                        return out
                    if logger and judge_choice != 0:
                        logger.debug(f"  ⚖️ LLM Judge (global-only) returned {judge_choice} for '{span_text}' — no bind")
            except Exception as e:
                if logger:
                    logger.debug("  ⚖️ LLM Judge (global-only) failed for '%s': %s", span_text, e)
            # Judge returned 0 or failed: fall through to unlinked (do not bind by score alone)

        # D) Unlinked - capture suggestions from candidates
        suggestions = []
        # Collect suggestions from local_results if available
        if local_results and len(local_results) > 0:
            for cand in local_results[:5]:
                suggestion = {
                    "name": cand.get("display_name") or cand.get("preferred_name") or "",
                    "match_score": float(cand.get("match_score", 0) or 0),
                    "recommendation": "MEDIUM",
                }
                if cand.get("service_id"):
                    suggestion["service_id"] = cand.get("service_id")
                if cand.get("stock_id"):
                    suggestion["inventory_id"] = cand.get("stock_id")
                    suggestion["stock_id"] = cand.get("stock_id")
                if cand.get("concept_id"):
                    suggestion["kb_concept_id"] = cand.get("concept_id")
                suggestions.append(suggestion)
        
        # Also add global KB candidates (clinical context) - these may have definitions
        if global_results and len(global_results) > 0:
            for cand in global_results[:15]:  # Top 15 global candidates (Ortolani at rank 12)
                # Skip if already in suggestions (avoid duplicates)
                cand_name = cand.get("preferred_name") or ""
                if any(s.get("name", "").lower() == cand_name.lower() for s in suggestions):
                    continue
                
                suggestion = {
                    "name": cand_name,
                    "match_score": float(cand.get("hybrid_score", cand.get("match_score", 0)) or 0),
                    "recommendation": "MEDIUM",
                    "kb_concept_id": cand.get("concept_id"),
                }
                # Include KB definition if available (helps vet verify)
                if cand.get("definition"):
                    suggestion["definition"] = cand.get("definition")
                suggestions.append(suggestion)
        
        if suggestions and logger:
            logger.info(f"  ✅ Captured {len(suggestions)} suggestions for '{span_text}': {[s.get('name') for s in suggestions]}")
        
        display_name = normalized_name if normalized_name.lower() != span_text.lower() else span_text
        # For ungrounded entities, use highest probability hint from Brain NER if available
        highest_hint = _get_highest_probability_hint(entity)
        if highest_hint:
            display_name = highest_hint
            if logger:
                logger.info(f"  💡 Ungrounded entity '{span_text}': using highest probability hint '{highest_hint}' as display_name")
        attrs = entity.get("attributes", {}) or {}
        if suggestions:
            attrs["suggestions"] = suggestions
        
        out = {
            "span_text": span_text,
            "normalized_name": normalized_name,
            "display_name": display_name,
            "kind": canonical_kind,
            "kb_concept_id": None,
            "kb_preferred_name": None,
            "kb_kind": canonical_kind,
            "match_method": "unlinked",
            "similarity_score": None,
            "assertion_id": entity.get("assertion_id", "CONF"),
            "attributes": attrs,
        }
        fallback_out, fallback_selected = await _build_domain_consultation_fallback(attrs, display_name)
        if fallback_out is not None:
            if grounding_collector is not None:
                grounding_collector.append(_build_grounding_record(
                    span_text, normalized_name, canonical_kind, route,
                    local_results or [], global_results or [], fallback_selected, fallback_out,
                    reranking_applied=reranked_this_entity, judge_justification="domain_consultation_fallback",
                ))
            return fallback_out
        if grounding_collector is not None:
            grounding_collector.append(_build_grounding_record(
                span_text, normalized_name, canonical_kind, route,
                local_results or [], global_results or [], None, out,
                reranking_applied=reranked_this_entity, judge_justification=None,
            ))
        return out
    
    # Route 3: Global-Direct (Clinical Observations)
    elif route == "global_direct":
        # DEFAULT PATH: Use batch intent (search_term) from Super-Pass when present.
        # FALLBACK: Per-entity intent only when search_term is missing.
        batch_search_term = (entity.get("search_term") or "").strip()
        search_term = batch_search_term if batch_search_term else normalized_name
        if batch_search_term:
            if logger:
                logger.info(f"  📦 Batch intent (default): using search_term '{search_term}'")
        else:
            # Fallback: Intent Interceptor (Step 2.3) when batch intent not available
            try:
                from kb_ner_intent import resolve_clinical_intent, should_trigger_intent_resolution
                from kb_ner_intent_guards import ground_clinical_terms
                if should_trigger_intent_resolution(normalized_name, canonical_kind):
                    intent_result = await asyncio.to_thread(
                        resolve_clinical_intent,
                        normalized_name,
                        context_anchors or [],
                        canonical_kind,
                        client,
                        logger,
                    )
                    if intent_result is not None:
                        intent_result = ground_clinical_terms(intent_result)
                    if intent_result is None:
                        if logger:
                            logger.info(f"  🚫 Intent Interceptor: '{span_text}' dropped (non-clinical)")
                        display_name = normalized_name if normalized_name.lower() != span_text.lower() else span_text
                        return {
                            "span_text": span_text,
                            "normalized_name": normalized_name,
                            "display_name": display_name,
                            "kind": canonical_kind,
                            "kb_concept_id": None,
                            "kb_preferred_name": None,
                            "kb_kind": canonical_kind,
                            "match_method": "intent_skip_not_clinical",
                            "similarity_score": None,
                            "assertion_id": entity.get("assertion_id", "CONF"),
                            "attributes": entity.get("attributes", {}) or {},
                        }
                    if intent_result.get("query"):
                        search_term = intent_result["query"]
                        if logger and search_term != normalized_name:
                            logger.info(f"  ✅ Intent Interceptor: '{normalized_name}' → search term '{search_term}'")
            except Exception as e:
                if logger:
                    logger.debug("  Intent Interceptor failed (using original term): %s", e)

        if logger:
            logger.info(f"  🌐 Route 3 (Global-Direct): Skipping local search, going directly to Global KB for '{span_text}' (kind: {canonical_kind})")

        if precomputed_global_results is not None:
            global_results = precomputed_global_results
        else:
            global_results = await asyncio.to_thread(_global_search, search_term)
        # IMPORTANT: global_results already include trigram/phonetic/vector scores (no extra DB calls needed here).
        best_global = global_results[0] if global_results else None

        # High-stakes diagnostics: do not auto-accept when search_term differs significantly from kb_preferred_name.
        # Prevents false positives like "ultralining test" → insonate (0.78); forces Judge or unlink.
        def _diagnostic_lexical_mismatch(st: str, kb_name: str, kind: str) -> bool:
            if (kind or "").strip().lower() not in ("procedure", "diagnostictest", "diagnostic"):
                return False
            if not (st or "").strip() or not (kb_name or "").strip():
                return False
            sa = {t for t in re.split(r"[^a-z0-9]+", (st or "").lower()) if t and len(t) > 1}
            sb = {t for t in re.split(r"[^a-z0-9]+", (kb_name or "").lower()) if t and len(t) > 1}
            if not sa:
                return True
            overlap = len(sa & sb) / max(len(sa), 1)
            threshold = float(os.getenv("DIAGNOSTIC_LEXICAL_MISMATCH_THRESHOLD", "0.35"))
            return overlap < threshold

        # Signal Convergence (Consensus Bypass): when Top 1 from original span grounding matches any of Top 3 from intent hints (same KB ID), bypass judge
        signal_convergence_threshold = float(os.getenv("SIGNAL_CONVERGENCE_THRESHOLD", "0.80"))
        if best_global and global_results and len(global_results) > 0:
            top1 = global_results[0]
            top1_id = top1.get("concept_id")
            top1_score = float(top1.get("hybrid_score") or top1.get("match_score") or 0)
            # For diagnostics, skip bypass when search_term vs kb_preferred_name differ (avoid e.g. ultralining → insonate)
            if _diagnostic_lexical_mismatch(search_term or span_text, top1.get("preferred_name") or "", canonical_kind):
                if logger:
                    logger.info(f"  ⚠️ Diagnostic lexical mismatch: search_term vs kb_preferred_name → require Judge (skip signal convergence)")
                top1 = None  # will fall through; do not use top1 for bypass
            elif top1_score > signal_convergence_threshold:
                from_orig = top1.get("_from_original", False)
                from_hint = top1.get("_from_hint", False)
                consensus = from_orig and from_hint
                hint_top3_ids = set()
                if not consensus:
                    from_hint_cands = [c for c in global_results if c.get("_from_hint")]
                    from_hint_cands.sort(key=lambda x: -float(x.get("match_score") or 0))
                    hint_top3_ids = {c.get("concept_id") for c in from_hint_cands[:3] if c.get("concept_id")}
                if consensus or (top1_id and top1_id in hint_top3_ids):
                    if logger:
                        logger.info(f"  ✅ Signal Convergence: '{span_text}' → '{top1.get('preferred_name')}' (score={top1_score:.3f}, consensus={consensus}) — skip judge")
                    candidate_name = (top1.get("preferred_name") or "").strip()
                    candidate_kind = (top1.get("kind") or "").strip()
                    display_name = (normalized_name or span_text).strip()
                    if candidate_name and (candidate_name.lower() == display_name.lower() or (candidate_name or "").lower() in (display_name or "").lower() or (display_name or "").lower() in (candidate_name or "").lower()):
                        display_name = candidate_name
                    _global_score = float(top1.get("final_score") or top1.get("match_score") or top1.get("hybrid_score", 0) or 0)
                    out = {
                        "span_text": span_text,
                        "normalized_name": normalized_name,
                        "display_name": display_name,
                        "kind": canonical_kind,
                        "kb_concept_id": top1_id,
                        "kb_preferred_name": candidate_name or normalized_name,
                        "kb_kind": candidate_kind or None,
                        "match_method": "signal_convergence" if consensus else "signal_convergence_hint_overlap",
                        "similarity_score": _global_score,
                        "assertion_id": entity.get("assertion_id", "CONF"),
                        "attributes": entity.get("attributes", {}),
                        "hints": _normalize_hints(entity.get("hints"), max_items=3) or None,
                        "search_term": (entity.get("search_term") or "").strip() or None,
                    }
                    # Preserve hint_probabilities if present
                    if entity.get("hint_probabilities"):
                        out["hint_probabilities"] = entity.get("hint_probabilities")
                    if grounding_collector is not None:
                        grounding_collector.append(_build_grounding_record(
                            span_text, normalized_name, canonical_kind, route,
                            [], global_results or [], top1, out,
                            reranking_applied=False, judge_justification=None,
                        ))
                    return out

        def _unsafe_species_hint(candidate_name: str, query: str) -> bool:
            """
            Reject obvious cross-species KB concepts when the query doesn't mention them.
            This prevents clinical hallucinations like '...cattle' injected into canine notes.
            """
            cand = (candidate_name or "").lower()
            q = (query or "").lower()
            species_terms = ["cattle", "bovine", "equine", "horse", "porcine", "swine", "ovine", "caprine", "goat", "sheep"]
            for term in species_terms:
                if term in cand and term not in q:
                    return True
            return False

        def _is_preventative_intent(span: str) -> bool:
            s = (span or "").lower()
            if "control" in s or "prevention" in s or "prevent" in s or "protection" in s:
                if ("tick" in s) or ("flea" in s) or ("parasite" in s) or ("deworm" in s) or ("worm" in s):
                    return True
            return False

        # Apply deterministic decision-flow gates (Option A + B) on global candidates.
        # This is CPU-only (no LLM), so it does not add pipeline latency.
        selected_global = None
        try:
            selected_global = apply_decision_flow_deterministic_only(
                mention=span_text,
                candidates=global_results or [],
                entity_kind=canonical_kind,
                logger=logger,
                auto_bind_threshold=auto_bind_threshold,
            )
        except Exception:
            selected_global = None

        # Use the selected candidate if it exists, otherwise fall back to best_global for backward compatibility.
        if selected_global:
            best_global = selected_global

        # Unified score: global search (two-stage) returns final_score/match_score; disambiguation may use hybrid_score.
        # Using only hybrid_score caused 0 to be used and broke threshold + anchor (e.g. noble angle → Norberg stored 0, so no anchors).
        _global_score = float(
            best_global.get("final_score")
            or best_global.get("match_score")
            or best_global.get("hybrid_score", 0)
            or 0
        ) if best_global else 0.0
        if best_global and _global_score >= float(llm_judge_threshold):
            candidate_name = (best_global.get("preferred_name") or "").strip()
            candidate_kind = (best_global.get("kind") or "").strip()
            query = normalized_name or span_text

            # High-stakes diagnostics: force Judge when search_term differs significantly from kb_preferred_name (avoid e.g. ultralining → insonate).
            if _diagnostic_lexical_mismatch(search_term or span_text, candidate_name, canonical_kind):
                if logger:
                    logger.info(f"  ⚠️ Diagnostic lexical mismatch: search_term vs kb_preferred_name → skip auto-accept (require Judge or unlink)")
                best_global = None
            # Hard safety gates: never let global_direct "over-fit" generic intents to unsafe KB kinds.
            elif _unsafe_species_hint(candidate_name, query):
                best_global = None
            elif _is_preventative_intent(query) and candidate_kind in {"Condition", "Disease"}:
                # e.g., "tick and flea control" must not bind to a specific disease/condition concept.
                best_global = None
            elif canonical_kind.lower() in {"exercise", "treatment", "therapy", "supplement", "nutrition", "parasite", "vaccination"} and candidate_kind == "Anatomy":
                best_global = None
            else:
                # Lexical sanity: if there's almost no token overlap, treat as unsafe embedding neighbor.
                overlap = _token_overlap_ratio(query, candidate_name)
                if overlap < 0.25 and not _contains_ci(query, candidate_name) and not _contains_ci(candidate_name, query):
                    best_global = None

        if best_global:
            candidate_name = (best_global.get("preferred_name") or "").strip()
            candidate_kind = (best_global.get("kind") or "").strip()

            # CRITICAL: For Global-Direct, do NOT default display_name to KB preferred_name.
            # Using KB names directly can cause catastrophic constraint-injection hallucinations.
            # Default display_name is the ASR-corrected normalized span; only use KB name if it’s a close lexical match.
            display_name = (normalized_name or span_text).strip()
            if candidate_name and (
                candidate_name.lower() == display_name.lower()
                or _contains_ci(display_name, candidate_name)
                or _contains_ci(candidate_name, display_name)
            ):
                display_name = candidate_name

            out = {
                "span_text": span_text,
                "normalized_name": normalized_name,
                "display_name": display_name,
                "kind": canonical_kind,
                "kb_concept_id": best_global.get("concept_id"),
                "kb_preferred_name": candidate_name or normalized_name,
                "kb_kind": candidate_kind or None,
                "match_method": "global_direct",
                "similarity_score": _global_score,
                "assertion_id": entity.get("assertion_id", "CONF"),
                "attributes": entity.get("attributes", {}),
            }
            if grounding_collector is not None:
                grounding_collector.append(_build_grounding_record(
                    span_text, normalized_name, canonical_kind, route,
                    [], global_results or [], best_global, out,
                    reranking_applied=False, judge_justification=None,
                ))
            return out

        # Unlinked (safe default)
        display_name = normalized_name if normalized_name.lower() != span_text.lower() else span_text
        # For ungrounded entities, use highest probability hint from Brain NER if available
        highest_hint = _get_highest_probability_hint(entity)
        if highest_hint:
            display_name = highest_hint
            if logger:
                logger.info(f"  💡 Ungrounded entity '{span_text}': using highest probability hint '{highest_hint}' as display_name")
        out = {
            "span_text": span_text,
            "normalized_name": normalized_name,
            "display_name": display_name,
            "kind": canonical_kind,
            "kb_concept_id": None,
            "kb_preferred_name": None,
            "kb_kind": canonical_kind,
            "match_method": "unlinked",
            "similarity_score": None,
            "assertion_id": entity.get("assertion_id", "CONF"),
            "attributes": entity.get("attributes", {}),
        }
        if grounding_collector is not None:
            grounding_collector.append(_build_grounding_record(
                span_text, normalized_name, canonical_kind, route,
                [], global_results or [], None, out,
                reranking_applied=False, judge_justification=None,
            ))
        return out
    
    # Default: Unlinked
    else:
        display_name = normalized_name if normalized_name.lower() != span_text.lower() else span_text
        # For ungrounded entities, use highest probability hint from Brain NER if available
        highest_hint = _get_highest_probability_hint(entity)
        if highest_hint:
            display_name = highest_hint
            if logger:
                logger.info(f"  💡 Ungrounded entity '{span_text}': using highest probability hint '{highest_hint}' as display_name")
        return {
            "span_text": span_text,
            "normalized_name": normalized_name,
            "display_name": display_name,
            "kind": canonical_kind,
            "kb_concept_id": None,
            "kb_preferred_name": None,
            "kb_kind": canonical_kind,
            "match_method": "unlinked",
            "similarity_score": None,
            "assertion_id": entity.get("assertion_id", "CONF"),
            "attributes": entity.get("attributes", {}),
        }


async def process_entities_parallel(
    all_entities: List[Dict[str, Any]],
    cleaned_transcript: str,
    raw_transcript: Optional[str],
    client: Optional[Any],
    clinic_id: Optional[int],
    visit_id: Optional[str],
    auto_bind_threshold: float,
    llm_judge_threshold: float,
    logger: Optional[logging.Logger],
    output_dir: Optional[str] = None,  # When set, write grounding output into this folder
    tracking_id: Optional[str] = None,  # When set, used in grounding output filename
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process all entities in parallel.
    
    Returns:
        Tuple of (entity_manifest, presenting_request_entities)
    """
    if not all_entities:
        return [], []
    
    if logger:
        logger.info("=" * 80)
        logger.info("⚡ PARALLEL ENTITY PROCESSING")
        logger.info("=" * 80)
        logger.info(f"Processing {len(all_entities)} entities in parallel")

    # ---------------------------------------------------------------------
    # Build lightweight assessment/diagnosis context (deterministic).
    # This is injected into Option C (LLM judge) prompts to improve plausibility
    # decisions (e.g., reject lab reagents as medicines when treating infection).
    # ---------------------------------------------------------------------
    def _build_assessment_context(entities: List[Dict[str, Any]]) -> str:
        terms: list[str] = []
        for e in entities or []:
            if not isinstance(e, dict):
                continue
            k = (e.get("kb_kind") or e.get("kind") or "").strip()
            if k in {"Condition", "Disease", "Diagnosis", "Symptom", "Finding"}:
                nm = (e.get("normalized_name") or e.get("span_text") or "").strip()
                if nm and nm.lower() not in {"condition", "diagnosis", "disease", "symptom", "finding"}:
                    terms.append(nm)
        seen = set()
        out = []
        for t in terms:
            tl = t.lower()
            if tl in seen:
                continue
            seen.add(tl)
            out.append(t)
        return ", ".join(out[:10])

    assessment_context = _build_assessment_context(all_entities)
    if assessment_context:
        for e in all_entities:
            if isinstance(e, dict):
                e.setdefault("assessment_context", assessment_context)

    # OPTIMIZATION: Batch search + judge for all billable entities in ONE LLM call
    # This replaces per-entity search + judge calls with a single batch operation
    batch_search_judge_results: Dict[int, Optional[Dict[str, Any]]] = {}
    if client and len(all_entities) > 0:
        try:
            # Collect billable entities
            billable_entities_list = []
            for idx, entity in enumerate(all_entities):
                if not isinstance(entity, dict):
                    continue
                span_text = entity.get("span_text", "")
                if not span_text:
                    continue
                canonical_kind = canonicalize_kind(entity.get("kind") or entity.get("kb_kind") or "Other")
                route = classify_entity_route(canonical_kind, entity=entity, logger=logger)
                if route == "dual_sync":  # Only billable entities
                    billable_entities_list.append({
                        "entity_idx": idx,
                        "entity": entity,
                    })
            
            if billable_entities_list:
                # Batch embeddings first (needed for batch search)
                embedding_cache: Optional[Dict[str, List[float]]] = None
                try:
                    from kb_ner_embeddings import embed_texts
                    unique_texts: List[str] = []
                    seen_texts: set = set()
                    for item in billable_entities_list:
                        entity = item["entity"]
                        t = (entity.get("span_text") or "").strip()
                        if t and t not in seen_texts:
                            seen_texts.add(t)
                            unique_texts.append(t)
                        search_term = (entity.get("search_term") or "").strip()
                        if search_term and search_term != t and search_term not in seen_texts:
                            seen_texts.add(search_term)
                            unique_texts.append(search_term)
                        for h in (entity.get("hints") or [])[:5]:
                            hs = str(h or "").strip()
                            if hs and hs not in seen_texts:
                                seen_texts.add(hs)
                                unique_texts.append(hs)
                    if unique_texts:
                        embeddings = embed_texts(unique_texts, client=client, logger=logger)
                        if embeddings and len(embeddings) == len(unique_texts):
                            embedding_cache = {text: vec for text, vec in zip(unique_texts, embeddings) if vec}
                except Exception as e:
                    if logger:
                        logger.debug(f"  ⚠️  Batch embedding pre-pass failed: {e}")
                
                # Run batch search + judge
                batch_search_judge_results = await batch_search_and_judge_all_billable_entities(
                    billable_entities=billable_entities_list,
                    cleaned_transcript=cleaned_transcript,
                    raw_transcript=raw_transcript,
                    client=client,
                    clinic_id=clinic_id,
                    logger=logger,
                    embedding_cache=embedding_cache,
                    auto_bind_threshold=auto_bind_threshold,
                    llm_judge_threshold=llm_judge_threshold,
                )
                if logger:
                    logger.info(f"  ✅ Batch search+judge: {len(batch_search_judge_results)} results for {len(billable_entities_list)} billable entities")
        except Exception as e:
            if logger:
                logger.warning(f"  ⚠️  Batch search+judge failed: {e} (falling back to per-entity processing)")
            batch_search_judge_results = {}

    # Batch embeddings once for entities that need them (avoids 20+ sequential embedding calls; ~8s -> ~0.5s)
    # CRITICAL: Only create embeddings for billable entities (dual_sync) that need LOCAL search (inventory/services)
    # Global search is disabled (GLOBAL_DIRECT_KINDS = []), so no embeddings needed for global-only entities
    # Skip non-billable/skipped entities - they don't need embeddings
    embedding_cache: Optional[Dict[str, List[float]]] = None
    if client and len(all_entities) > 0:
        try:
            from kb_ner_embeddings import embed_texts
            from kb_ner_routing import DUAL_SYNC_BILLABLE_KINDS, canonicalize_kind, classify_entity_route
            unique_texts: List[str] = []
            seen_texts: set = set()
            for e in all_entities:
                if not isinstance(e, dict):
                    continue
                # Check route to determine if entity needs embeddings
                kind = e.get("kind") or ""
                canonical_kind = canonicalize_kind(kind)
                route = classify_entity_route(canonical_kind, entity=e, logger=logger)
                # Only create embeddings for dual_sync entities (billable - need LOCAL search)
                # Global_direct is disabled (GLOBAL_DIRECT_KINDS = []), so no global-only entities exist
                if route != "dual_sync":
                    continue  # Skip non-billable/skipped entities - they don't need embeddings
                # Batch embed span_text (used by local inventory/service search)
                t = (e.get("span_text") or "").strip()
                if t and t not in seen_texts:
                    seen_texts.add(t)
                    unique_texts.append(t)
                # Also batch embed search_term if different from span_text (ASR-corrected terms)
                search_term = (e.get("search_term") or "").strip()
                if search_term and search_term != t and search_term not in seen_texts:
                    seen_texts.add(search_term)
                    unique_texts.append(search_term)
                for h in (e.get("hints") or [])[:5]:
                    hs = str(h or "").strip()
                    if hs and hs not in seen_texts:
                        seen_texts.add(hs)
                        unique_texts.append(hs)
            if unique_texts:
                embeddings = embed_texts(unique_texts, client=client, logger=logger)
                if embeddings and len(embeddings) == len(unique_texts):
                    embedding_cache = {text: vec for text, vec in zip(unique_texts, embeddings) if vec}
                    if logger and embedding_cache:
                        billable_count = len([e for e in all_entities if isinstance(e, dict) and classify_entity_route(canonicalize_kind(e.get('kind') or ''), entity=e, logger=logger) == "dual_sync"])
                        logger.info(f"  📊 Batch embeddings: {len(embedding_cache)} unique terms for {billable_count} billable entities (dual_sync only - local search)")
        except Exception as e:
            if logger:
                logger.debug(f"  ⚠️  Batch embedding pre-pass failed: {e} (local search will use trigram/phonetic only)")
    
    # OPTIMIZATION: Speculative KB pre-fetching for domain-related concepts
    # When domain is detected (e.g., "orthopedic"), pre-fetch common KB concepts to cache definitions
    # This eliminates DB lookups during suggestion building (e.g., "Ortolani" definition ready immediately)
    domain_kb_cache: Optional[Dict[str, Dict[str, Any]]] = None
    if raw_transcript:
        try:
            from kb_ner_super_pass import detect_domain, DOMAIN_PRIMERS
            domain = detect_domain(raw_transcript)
            likely_terms = DOMAIN_PRIMERS.get(domain, [])
            
            if likely_terms and len(likely_terms) > 0 and not LOCAL_ONLY:
                # Pre-fetch KB concepts for domain terms (speculative cache); skipped when LOCAL_ONLY
                def _prefetch_domain_kb():
                    """Pre-fetch KB concept definitions for domain terms"""
                    cache = {}
                    try:
                        with pg_conn_ctx(logger=logger) as conn:
                            with conn.cursor() as cur:
                                # Query KB for domain-related concepts (top 10 most common)
                                for term in likely_terms[:10]:  # Limit to top 10 to avoid overhead
                                    try:
                                        cur.execute("""
                                            SELECT concept_id, preferred_name, definition, kind
                                            FROM kb.concepts
                                            WHERE LOWER(preferred_name) = LOWER(%s)
                                            LIMIT 1
                                        """, (term,))
                                        row = cur.fetchone()
                                        if row:
                                            cache[term.lower()] = {
                                                "concept_id": row[0],
                                                "preferred_name": row[1],
                                                "definition": row[2],
                                                "kind": row[3],
                                            }
                                    except Exception:
                                        continue
                    except Exception:
                        pass
                    return cache
                
                # Pre-fetch in background (non-blocking)
                domain_kb_cache = await asyncio.to_thread(_prefetch_domain_kb)
                if domain_kb_cache and logger:
                    logger.info(f"  🎯 Speculative KB cache: {len(domain_kb_cache)} domain concepts pre-fetched for '{domain}'")
        except Exception as e:
            if logger:
                logger.debug(f"  ⚠️  Domain KB pre-fetch failed: {e} (non-critical)")

    # Batch vector search: one embedding request + one DB round trip for all entities needing global KB
    # Set KB_USE_BATCH_VECTOR_SEARCH=false to fall back to per-entity global search (e.g. for debugging).
    use_batch_vector = os.getenv("KB_USE_BATCH_VECTOR_SEARCH", "true").strip().lower() in ("1", "true", "yes")
    global_prefetch_by_entity_idx: Dict[int, List[Dict[str, Any]]] = {}
    if use_batch_vector and client and len(all_entities) > 0:
        try:
            # 9-tuple (entity_idx, search_term, kind_filter, original_span, hints, domain, suggestion_prob, hint_probs, query_expansion_list)
            need_global: List[Tuple] = []
            for idx0, ent in enumerate(all_entities):
                if not isinstance(ent, dict):
                    continue
                span_text = (ent.get("span_text") or "").strip()
                if not span_text:
                    continue
                normalized_name = span_text.replace("[unclear]", "").strip() if "[unclear]" in span_text else span_text
                normalized_name = sanitize_asr_errors(normalized_name)
                kb_kind_raw = ent.get("kb_kind") or ent.get("kind", "Other")
                canonical_kind = canonicalize_kind(kb_kind_raw)
                route = classify_entity_route(canonical_kind, entity=ent, logger=logger)
                # Local-only production mode: prefetch global only for explicit global_direct routes.
                if route != "global_direct":
                    continue
                search_term = (ent.get("search_term") or "").strip() or normalized_name
                if canonical_kind == "ReasonForVisit":
                    kind_filter = list(REASON_ALLOWED_KB_KINDS)
                else:
                    kind_filter = map_ner_kind_to_kb_kind_filter(canonical_kind) or []
                hints_list = _normalize_hints(ent.get("hints"), max_items=3)
                domain_from_brain = ent.get("domain")
                if isinstance(domain_from_brain, str) and (domain_from_brain or "").strip().lower() not in ("", "general"):
                    domain_arg = [domain_from_brain.strip().lower()]
                elif isinstance(domain_from_brain, list) and domain_from_brain:
                    domain_arg = [(d or "").strip().lower() for d in domain_from_brain if (d or "").strip().lower() and (d or "").strip().lower() != "general"]
                else:
                    domain_arg = None
                suggestion_prob = ent.get("suggestion_probability")
                hint_probs = ent.get("hint_probabilities") if isinstance(ent.get("hint_probabilities"), dict) else {}
                qe = ent.get("query_expansion")
                query_expansion_list = [str(x).strip() for x in qe if str(x).strip()][:3] if isinstance(qe, list) else []
                if span_text or search_term:
                    need_global.append((idx0, search_term, kind_filter, span_text, hints_list, domain_arg, suggestion_prob, hint_probs, query_expansion_list))
                else:
                    need_global.append((idx0, search_term, kind_filter, None, None, domain_arg, suggestion_prob, hint_probs, query_expansion_list))
            if need_global and not LOCAL_ONLY:
                def _run_batch_global() -> Dict[int, List[Dict[str, Any]]]:
                    with pg_conn_ctx(logger=logger) as c:
                        from kb_ner_global_search import run_batch_global_vector_search
                        return run_batch_global_vector_search(
                            c, need_global, client, logger=logger, raw_transcript=raw_transcript,
                            anchor_concept_ids_by_entity=None,
                        )
                global_prefetch_by_entity_idx = await asyncio.to_thread(_run_batch_global)
                if logger and global_prefetch_by_entity_idx:
                    n_terms = len(set(st for _, st, _ in need_global))
                    logger.info(f"  📊 Batch vector search: {len(need_global)} entities, {n_terms} unique terms → 1 embed + 1 DB round trip")
        except Exception as e:
            if logger:
                logger.debug(f"  ⚠️  Batch global vector search failed: {e} (will fall back to per-entity search)")

    # Create database connections for each entity (or use connection pool)
    # For now, we'll create a connection per entity (can be optimized with connection pool)
    entity_manifest = []
    presenting_request_entities = []
    # Grounding layer output: one record per NER entity (candidates, rank, Judge, reason, final binding)
    grounding_records: List[Dict[str, Any]] = []
    # Prefer explicit output_dir from caller (SOAP output folder); fall back to env for backward compatibility.
    effective_out_dir = (str(output_dir).strip() if output_dir else "") or os.getenv("GROUNDING_LAYER_OUTPUT_DIR", "").strip()
    emit_grounding_output = effective_out_dir != ""
    
    # Bounded concurrency for robustness on long transcripts:
    # - prevents spawning thousands of tasks / DB ops at once
    # - keeps pool + LLM usage stable
    try:
        max_workers = int(os.getenv("KB_MAX_PARALLEL_ENTITIES", "12"))
    except Exception:
        max_workers = 12
    if max_workers < 1:
        max_workers = 1

    q: asyncio.Queue = asyncio.Queue()
    results_by_idx: Dict[int, Any] = {}

    async def worker():
        while True:
            item = await q.get()
            if item is None:
                q.task_done()
                return
            idx0, ent = item
            try:
                # Check if this entity has a precomputed batch search+judge result
                precomputed_batch_result = batch_search_judge_results.get(idx0) if batch_search_judge_results else None
                
                res = await process_single_entity_async(
                    entity=ent,
                    idx=idx0 + 1,
                    total=len(all_entities),
                    cleaned_transcript=cleaned_transcript,
                    raw_transcript=raw_transcript,
                    conn=None,
                    client=client,
                    clinic_id=clinic_id,
                    visit_id=visit_id,
                    auto_bind_threshold=auto_bind_threshold,
                    llm_judge_threshold=llm_judge_threshold,
                    logger=logger,
                    embedding_cache=embedding_cache,
                    all_entities=all_entities,
                    precomputed_global=global_prefetch_by_entity_idx,
                    grounding_collector=grounding_records if emit_grounding_output else None,
                    precomputed_batch_search_judge_result=precomputed_batch_result,  # Pass batch search+judge result
                )
                results_by_idx[idx0] = res
            except Exception as e:
                results_by_idx[idx0] = e
            finally:
                q.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(min(max_workers, len(all_entities)))]
    for idx0, ent in enumerate(all_entities):
        await q.put((idx0, ent))
    for _ in workers:
        await q.put(None)
    await q.join()
    for w in workers:
        await w

    # Preserve ordering of results
    results = [results_by_idx[i] for i in range(len(all_entities))]
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            if logger:
                logger.error(f"  ❌ Entity processing failed: {result}")
                import traceback
                logger.debug(traceback.format_exc())
            continue
        
        if result is None:
            continue
        
        entity_result, presenting_request_entity = result
        
        if entity_result:
            entity_manifest.append(entity_result)
        
        if presenting_request_entity:
            presenting_request_entities.append(presenting_request_entity)

    # Dedup safeguard for domain-consultation fallback:
    # only one draft SKU per domain/service per visit; group extra mentions into remarks.
    fallback_groups: Dict[Tuple[str, str], List[int]] = {}
    for i, ent in enumerate(entity_manifest):
        if not isinstance(ent, dict):
            continue
        if (ent.get("match_method") or "").strip() != "domain_consultation_fallback":
            continue
        attrs = ent.get("attributes") or {}
        fb = attrs.get("fallback_consultation") if isinstance(attrs, dict) else None
        if not isinstance(fb, dict):
            continue
        dk = (fb.get("domain_key") or "").strip().lower()
        sid = str(fb.get("service_id") or ent.get("local_service_id") or "").strip()
        if not dk or not sid:
            continue
        key = (dk, sid)
        fallback_groups.setdefault(key, []).append(i)

    deduped_entities = 0
    for (dk, sid), idxs in fallback_groups.items():
        if len(idxs) <= 1:
            # Keep a single mention list even for one entity.
            i0 = idxs[0]
            e0 = entity_manifest[i0]
            attrs0 = e0.get("attributes") or {}
            fb0 = attrs0.get("fallback_consultation") if isinstance(attrs0, dict) else None
            if isinstance(fb0, dict):
                mention = (fb0.get("source_mention") or e0.get("span_text") or "").strip()
                if mention:
                    fb0["grouped_mentions"] = [mention]
                    fb0["remarks"] = f"Drafted as domain consultation for: {mention}"
            continue

        primary_idx = idxs[0]
        primary = entity_manifest[primary_idx]
        p_attrs = primary.get("attributes") or {}
        p_fb = p_attrs.get("fallback_consultation") if isinstance(p_attrs, dict) else None
        mentions: List[str] = []
        for ix in idxs:
            e = entity_manifest[ix]
            ea = e.get("attributes") or {}
            efb = ea.get("fallback_consultation") if isinstance(ea, dict) else None
            m = ""
            if isinstance(efb, dict):
                m = (efb.get("source_mention") or "").strip()
            if not m:
                m = (e.get("span_text") or "").strip()
            if m and m not in mentions:
                mentions.append(m)

        if isinstance(p_fb, dict):
            p_fb["grouped_mentions"] = mentions
            p_fb["remarks"] = "Drafted as domain consultation for: " + ", ".join(mentions)

        for ix in idxs[1:]:
            e = entity_manifest[ix]
            e["local_service_id"] = None
            e["local_stock_id"] = None
            e["match_method"] = "domain_consultation_fallback_grouped"
            ea = e.get("attributes") or {}
            ea["grouped_under_domain_consultation"] = {
                "domain_key": dk,
                "service_id": sid,
                "primary_entity_id": primary.get("entity_id"),
            }
            e["attributes"] = ea
            deduped_entities += 1

    if logger and deduped_entities:
        logger.info("🛡️ Domain consultation dedupe: grouped %d fallback entities under primary domain SKUs", deduped_entities)
    
    if logger:
        logger.info(f"✅ Parallel processing complete: {len(entity_manifest)} entities in manifest, {len(presenting_request_entities)} PresentingRequest entities")
    
    # Write grounding layer output when enabled (candidates, scores, definitions, rank, Judge, reason, final binding)
    if emit_grounding_output and grounding_records:
        out_dir = effective_out_dir
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
                file_id = (tracking_id or "").strip() or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(out_dir, f"grounding_layer_output_{file_id}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(grounding_records, f, indent=2, ensure_ascii=False)
                if logger:
                    logger.info(f"  📄 Grounding layer output: {len(grounding_records)} records → {path}")
            except Exception as e:
                if logger:
                    logger.warning(f"  ⚠️ Could not write grounding layer output: {e}")
    
    return entity_manifest, presenting_request_entities

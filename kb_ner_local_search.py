"""
Local inventory and service search utilities.

This module handles:
- Local inventory search (drugs, nutrition)
- Local service search (procedures, services)
- Search term normalization
- SKU resolution with deterministic tie-breaking
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional

from kb_ner_db import ensure_fuzzystrmatch, vector_extension_available, invalidate_vector_extension_cache

# Soft Gate for local: domain as boost (not hard filter). When True, do not filter by domain in SQL; apply boost in Python.
USE_SOFT_GATE_LOCAL = os.getenv("SOFT_GATE_LOCAL", "true").lower() in ("1", "true", "yes")
try:
    _sg_lw = float(os.getenv("SOFT_GATE_LOCAL_BASE_WEIGHT", "0.8"))
    _sg_lb = float(os.getenv("SOFT_GATE_LOCAL_DOMAIN_BOOST", "0.2"))
except Exception:
    _sg_lw, _sg_lb = 0.8, 0.2
SOFT_GATE_LOCAL_BASE_WEIGHT = max(0.0, min(1.0, _sg_lw))
SOFT_GATE_LOCAL_DOMAIN_BOOST = max(0.0, min(1.0, _sg_lb))
from kb_ner_routing import canonicalize_kind

# Import embedding functions for vector matching
try:
    from kb_ner_embeddings import embed_text, to_pgvector_literal
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    embed_text = None
    to_pgvector_literal = None

# Category soft-gate for local matching (inventory/services).
# When USE_CATEGORY_HARD_GATE is True (production), soft gate is redundant; kept for non-gated or fallback.
USE_CATEGORY_SOFT_GATE_LOCAL = os.getenv("CATEGORY_SOFT_GATE_LOCAL", "true").lower() in ("1", "true", "yes")
try:
    _cat_boost = float(os.getenv("CATEGORY_SOFT_GATE_BOOST", "0.12"))
    _cat_penalty = float(os.getenv("CATEGORY_SOFT_GATE_MISMATCH_PENALTY", "0.35"))
except Exception:
    _cat_boost, _cat_penalty = 0.12, 0.35
CATEGORY_SOFT_GATE_BOOST = max(0.0, min(1.0, _cat_boost))
CATEGORY_SOFT_GATE_MISMATCH_PENALTY = max(0.0, min(1.0, _cat_penalty))

# Production: hard category gating. When True, search is restricted to bucket group (SQL WHERE category IN (...)).
# Prevents cross-category hallucinations (e.g. "flea control" never sees "Accessories & Toys").
USE_CATEGORY_HARD_GATE_LOCAL = os.getenv("CATEGORY_HARD_GATE_LOCAL", "true").lower() in ("1", "true", "yes")
LOCAL_TRGM_RECALL_THRESHOLD = float(os.getenv("LOCAL_TRGM_RECALL_THRESHOLD", "0.30"))
LOCAL_VECTOR_ON_DEMAND_EMBED = os.getenv("LOCAL_VECTOR_ON_DEMAND_EMBED", "true").lower() in ("1", "true", "yes")

# Bucket groups (category aliasing): Brain NER category or kind -> list of DB category values (normalized lower).
# SQL normalizes DB category as LOWER(TRIM(REPLACE(category, '&', 'and'))); bucket values must match that form.
# When adding new DB category literals (e.g. "Post-operative Care"), add their normalized form to the right bucket.
# Used for hard gate: search only within these categories. Overlap avoids misclassification (e.g. medicated shampoo).
# Inventory (soap.inventory) categories: Deworming, Flea & Tick Treatment, Other Parasite Treatment, Vaccines,
# Medication, Fluid Therapy, Diet, Nutrition & Supplements, OTC Products, Grooming & Hygiene Care, General Consumables,
# Medical Supplies, Surgical Supplies, Lab Consumables, Lab Supplies, Cleaning Supplies, Mortuary, Accessories & Toys, Pet Supplies.
_CATEGORY_BUCKET_GROUPS: Dict[str, List[str]] = {
    # Inventory (soap.inventory) — canonical list; DB category normalized same way (& -> and, lower)
    "deworming": ["deworming"],
    "flea and tick treatment": ["flea and tick treatment"],
    "other parasite treatment": ["other parasite treatment"],
    "vaccines": ["vaccines"],
    "medication": ["medication"],
    "fluid therapy": ["fluid therapy"],
    "diet": ["diet"],
    "nutrition and supplements": ["nutrition and supplements"],
    "otc products": ["otc products"],
    "grooming and hygiene care": ["grooming and hygiene care"],
    "general consumables": ["general consumables"],
    "medical supplies": ["medical supplies"],
    "surgical supplies": ["surgical supplies"],
    "lab consumables": ["lab consumables"],
    "lab supplies": ["lab supplies"],
    "cleaning supplies": ["cleaning supplies"],
    "mortuary": ["mortuary"],
    "accessories and toys": ["accessories and toys"],
    "pet supplies": ["pet supplies"],
    # Aliases for kind/context expansion (Brain NER uses these; they map to one or more inventory categories)
    "preventive and parasite control": ["deworming", "flea and tick treatment", "other parasite treatment"],
    "procedure": ["medical supplies", "surgical supplies", "lab supplies", "lab consumables", "general consumables"],
    "diagnostic": ["lab supplies", "lab consumables", "medical supplies"],
    # Service (soap.service_master) — canonical list; DB category normalized same way (& -> and, lower)
    "consultation": ["consultation"],
    "general care": ["general care"],
    "hospitalisation": ["hospitalisation", "hospitalization"],
    "procedure service": ["procedure"],  # disambiguate from inventory "procedure"
    "surgery": ["surgery"],
    "preventive care": ["preventive care"],
    "rehabilitation": ["rehabilitation", "rehabilitation and physiotherapy", "physiotherapy and rehabilitation", "physiotherapy"],
    "physiotherapy": ["rehabilitation", "rehabilitation and physiotherapy", "physiotherapy and rehabilitation", "physiotherapy"],
    "post-operative": ["post-operative", "post operative", "post-operative care", "post operative care"],
    "counselling": ["counselling", "counseling"],
    "diet planning": ["diet planning"],
    "speciality services": ["speciality services", "specialty services"],
    "lab": ["lab"],
    "radiology": ["radiology"],
    "boarding": ["boarding"],
    "hygiene and grooming": ["hygiene and grooming", "hygiene & grooming", "grooming"],
    "training": ["training"],
    "behavior": ["behavior", "behaviour"],
    "other non-medical": ["other non-medical", "other non medical", "other non-medical services"],
}
# Kind -> bucket groups when Brain does not provide category (fallback).
# Inventory kinds (Medication, Diet, ParasiteControl) use inventory buckets; Procedure/Diagnostic/Preventive use service buckets when searching service_master.
# Dual-sync kinds that have kind-level category hints for local search. ReasonForVisit/Diagnosis included;
# kinds not listed (e.g. Diagnosis with no entity hints) get no category filter (search all categories).
_KIND_TO_LOCAL_CATEGORY_HINTS = {
    "Preventive": ["vaccines", "preventive care"],  # inventory (vaccines) + service (preventive care)
    "ParasiteControl": ["preventive and parasite control"],  # expands to deworming, flea and tick treatment, other parasite treatment
    "Medication": ["medication", "nutrition and supplements", "otc products", "fluid therapy", "vaccines"],
    "Diet": ["diet", "nutrition and supplements", "diet planning"],
    "Procedure": [
        "procedure service", "surgery", "rehabilitation", "post-operative", "consultation", "general care",
        "speciality services",
    ],
    "Diagnostic": ["lab", "radiology"],
    "ReasonForVisit": [],  # can be anything; no kind-level restriction; entity category hints used when present; else search all
    "Diagnosis": [],  # no kind-level restriction; entity inventory_category/service_category used when present; else search all
}

_MEDICAL_SERVICE_CATEGORY_HINTS = [
    "consultation",
    "general care",
    "hospitalisation",
    "procedure service",
    "surgery",
    "preventive care",
    "rehabilitation",
    "post-operative",
    "counselling",
    "diet planning",
    "speciality services",
    "lab",
    "radiology",
]

_NON_MEDICAL_SERVICE_CATEGORY_HINTS = [
    "boarding",
    "hygiene and grooming",
    "training",
    "behavior",
    "other non-medical",
]


def _jaro_winkler_similarity(a: str, b: str, prefix_scale: float = 0.1, max_prefix: int = 4) -> float:
    """
    Pure-Python Jaro-Winkler similarity in [0,1].
    Used for reranking so prefix-preserving ASR errors are scored higher than edit-distance alone.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    a = a.lower().strip()
    b = b.lower().strip()
    if not a or not b:
        return 0.0

    la, lb = len(a), len(b)
    match_distance = max(la, lb) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    a_matches = [False] * la
    b_matches = [False] * lb
    matches = 0
    transpositions = 0

    for i in range(la):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, lb)
        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(la):
        if not a_matches[i]:
            continue
        while k < lb and not b_matches[k]:
            k += 1
        if k < lb and a[i] != b[k]:
            transpositions += 1
        k += 1

    transpositions /= 2.0
    m = float(matches)
    jaro = (m / la + m / lb + (m - transpositions) / m) / 3.0

    prefix = 0
    max_pref = min(max_prefix, la, lb)
    while prefix < max_pref and a[prefix] == b[prefix]:
        prefix += 1
    return min(1.0, jaro + (prefix * prefix_scale * (1.0 - jaro)))


def _best_jw_score(query: str, *candidate_texts: Optional[str]) -> float:
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    best = 0.0
    for t in candidate_texts:
        cand = (t or "").strip().lower()
        if not cand:
            continue
        best = max(best, _jaro_winkler_similarity(q, cand))
    return best


def _normalize_category_value(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_effective_category_hints(
    entity_kind: str,
    category_hints: Optional[List[str]],
    service_type: Optional[str] = None,
    is_service_search: bool = False,
) -> List[str]:
    hints: List[str] = []
    for c in (category_hints or []):
        n = _normalize_category_value(c)
        if n and n not in hints:
            hints.append(n)

    st = _normalize_category_value(service_type)
    service_type_keys = []
    if is_service_search and st in {"medical", "non medical", "non-medical"}:
        service_type_keys = (
            [_normalize_category_value(c) for c in _MEDICAL_SERVICE_CATEGORY_HINTS]
            if st == "medical"
            else [_normalize_category_value(c) for c in _NON_MEDICAL_SERVICE_CATEGORY_HINTS]
        )

    # If Brain already gave explicit category hints, keep them strict (do not broaden with kind fallback).
    if hints:
        # Enforce service_type hard partition if provided.
        if is_service_search and service_type_keys:
            hints = [h for h in hints if h in service_type_keys]
            if not hints:
                hints = list(service_type_keys)
        # "General" on service side should mean clinical medical services only.
        if is_service_search and "general" in hints:
            hints = [h for h in hints if h != "general"]
            for c in _MEDICAL_SERVICE_CATEGORY_HINTS:
                n = _normalize_category_value(c)
                if n and n not in hints:
                    hints.append(n)
        return hints

    # No explicit category hints: use service_type policy first for service search.
    if is_service_search and st in {"medical", "non medical", "non-medical"}:
        src = _MEDICAL_SERVICE_CATEGORY_HINTS if st == "medical" else _NON_MEDICAL_SERVICE_CATEGORY_HINTS
        for c in src:
            n = _normalize_category_value(c)
            if n and n not in hints:
                hints.append(n)
        return hints

    # Default fallback by kind (already excludes non-medical for Procedure above).
    fallback = _KIND_TO_LOCAL_CATEGORY_HINTS.get(canonicalize_kind(entity_kind), [])
    for c in fallback:
        n = _normalize_category_value(c)
        if n and n not in hints:
            hints.append(n)
    return hints


def _get_hard_gate_categories(
    entity_kind: str,
    category_hints: Optional[List[str]],
    service_type: Optional[str] = None,
    is_service_search: bool = False,
) -> List[str]:
    """
    Returns list of normalized (lower) DB category values for hard-gate SQL: LOWER(TRIM(category)) IN (...).
    Expands Brain category + kind fallback via bucket groups (aliasing) so e.g. Medication -> [medication, nutrition & supplements].
    Returns empty list when no hints/fallback -> no category filter (search all).
    """
    hints = _build_effective_category_hints(
        entity_kind,
        category_hints,
        service_type=service_type,
        is_service_search=is_service_search,
    )
    out: set = set()
    for h in hints:
        out.update(_CATEGORY_BUCKET_GROUPS.get(h, [h]))
    return sorted(out)


def _category_match_soft_score(
    entity_kind: str,
    candidate_category: Any,
    category_hints: Optional[List[str]],
) -> tuple[float, Optional[bool], List[str]]:
    """
    Returns (category_score_adjustment, match_state, effective_hints)
    - match_state: True (match), False (mismatch), None (not enough info)
    """
    effective_hints = _build_effective_category_hints(entity_kind, category_hints)
    cand = _normalize_category_value(candidate_category)
    if not cand or not effective_hints:
        return 0.0, None, effective_hints
    for h in effective_hints:
        if h == cand or h in cand or cand in h:
            return CATEGORY_SOFT_GATE_BOOST, True, effective_hints
    return -CATEGORY_SOFT_GATE_MISMATCH_PENALTY, False, effective_hints


def normalize_search_term_for_local(span_text: str, entity_kind: str) -> List[str]:
    """
    Normalize search term for better local matching.
    Returns a list of normalized terms to try.
    
    Args:
        span_text: Original search text
        entity_kind: Entity kind
        
    Returns:
        List of normalized search terms
    """
    terms = [span_text.lower().strip()]
    base = span_text.lower().strip()
    
    # Remove common drug suffixes
    suffixes = ["tablets", "tablet", "capsules", "capsule", "mg", "ml", "g", "kg", "s", "60s", "30s"]
    for suffix in suffixes:
        if base.endswith(" " + suffix) or base.endswith(suffix):
            # Remove suffix and any trailing numbers
            cleaned = base.rsplit(" " + suffix, 1)[0].strip()
            # Also try without trailing numbers
            cleaned = re.sub(r'\s+\d+$', '', cleaned).strip()
            if cleaned and cleaned != base:
                terms.append(cleaned)
    
    # Handle synonyms
    if "anal gland" in base:
        terms.append(base.replace("anal gland", "anal sac"))
    if "gland" in base and "anal" in base:
        terms.append(base.replace("gland", "sac"))
    
    # Handle common phonetic/ASR errors
    # "cortex" -> "coatex" (common ASR error)
    if "cortex" in base:
        terms.append(base.replace("cortex", "coatex"))
    # Also try just the base word if it's a single word
    if len(base.split()) == 1 and len(base) <= 10:
        # For short single words, try common variations
        if base == "cortex":
            terms.append("coatex")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    return unique_terms


def _build_search_terms_with_hints(
    base_terms: List[str],
    explicit_search_term: Optional[str],
    hints: Optional[List[str]],
    hint_probabilities: Optional[Dict[str, float]],
    max_hints: int = 5,
) -> List[str]:
    """
    Extend base search terms with explicit search_term (if not already present) and hints
    so that hints are grounded: we retrieve candidates that match hint text, not only the main span.
    Hints are added in descending order of hint_probability (highest confidence first), up to max_hints.
    """
    seen = {(t or "").strip().lower() for t in base_terms if (t or "").strip()}
    out = list(base_terms)
    if explicit_search_term and (explicit_search_term or "").strip():
        st = (explicit_search_term or "").strip()
        if st.lower() not in seen:
            out.append(st)
            seen.add(st.lower())
    if hints:
        hint_probs = hint_probabilities or {}
        sorted_hints = sorted(
            [h for h in hints if (h or "").strip()],
            key=lambda h: float(hint_probs.get((h or "").strip(), 0.0)),
            reverse=True,
        )
        for h in sorted_hints[:max_hints]:
            hnorm = (h or "").strip()
            if not hnorm or hnorm.lower() in seen:
                continue
            seen.add(hnorm.lower())
            out.append(hnorm)
    return out


from kb_ner_db import _column_exists


def resolve_default_sku(
    candidates: List[Dict[str, Any]],
    span_text: str,
    entity_kind: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """
    Deterministic resolver for selecting the best SKU from multiple candidates.
    
    Applies tie-break rules:
    1. Prefer exact brand token match
    2. Prefer matching form (capsule vs tablet)
    3. Prefer clinic's default pack mapping (e.g., "60S" for Nutrish)
    4. Prefer higher match score
    
    Args:
        candidates: List of candidate matches (already sorted by score)
        span_text: Original mention text
        entity_kind: Entity kind
        logger: Optional logger
        
    Returns:
        Best candidate based on deterministic rules
    """
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Extract brand and form from span_text
    span_lower = span_text.lower()
    
    # Common brand names to look for
    brand_keywords = ["coatex", "cortex", "nutrish", "nutrich", "maropitant", "simparica"]
    form_keywords = ["capsule", "capsules", "tablet", "tablets", "mg", "ml"]
    
    # Default pack size mappings (clinic preferences)
    default_pack_sizes = {
        "nutrish": "60",  # Prefer 60S for Nutrish
        "nutrich": "60",  # Prefer 60S for Nutrich
        "coatex": None,   # No default preference
        "cortex": None,   # No default preference
    }
    
    detected_brand = None
    detected_form = None
    detected_number = None
    
    for brand in brand_keywords:
        if brand in span_lower:
            detected_brand = brand
            break
    
    for form in form_keywords:
        if form in span_lower:
            detected_form = form
            break
    
    # Extract numbers (e.g., "60s", "30", "10mg")
    numbers = re.findall(r'\d+', span_lower)
    if numbers:
        detected_number = numbers[0]  # Take first number found
    
    # Get default pack size for brand
    default_pack = None
    if detected_brand:
        default_pack = default_pack_sizes.get(detected_brand.lower())
    
    # Score each candidate
    def score_candidate(cand):
        score = cand.get("match_score", 0)
        display_lower = cand.get("display_name", "").lower()
        
        # Boost for exact brand match
        if detected_brand and detected_brand in display_lower:
            score += 0.2
        
        # Boost for form match
        if detected_form:
            if detected_form in display_lower:
                score += 0.15
            # Handle plural/singular
            form_singular = detected_form.rstrip('s')
            if form_singular in display_lower:
                score += 0.15
        
        # Boost for number match (e.g., "60s" in "NUTRICH TABLET 60S")
        if detected_number:
            if detected_number in display_lower:
                score += 0.1
            # Also check for "60S" vs "60s"
            if detected_number.upper() in display_lower.upper():
                score += 0.1
        
        # MAJOR BOOST: Prefer default pack size if no number specified in mention
        if not detected_number and default_pack:
            # Check if display name contains the default pack size
            if default_pack in display_lower or f"{default_pack}s" in display_lower or f"{default_pack}S" in display_lower:
                score += 0.3  # Strong preference for default pack
            else:
                # Penalize non-default packs
                score -= 0.1
        
        # Prefer items with "BLISTER" if mention has "capsule"
        if "capsule" in span_lower and "blister" in display_lower:
            score += 0.05
        
        # Prefer items with "TABLET" if mention has "tablet"
        if "tablet" in span_lower and "tablet" in display_lower:
            score += 0.05
        
        return score
    
    # Score all candidates and pick best
    scored_candidates = [(score_candidate(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_score, best_candidate = scored_candidates[0]
    
    if logger and len(candidates) > 1:
        logger.debug(f"  🎯 Resolver: Selected '{best_candidate.get('display_name')}' from {len(candidates)} candidates (resolved score: {best_score:.3f})")
    
    return best_candidate


def search_local_inventory_topk(
    conn,
    span_text: str,
    entity_kind: str,
    clinic_id: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    threshold: float = 0.50,
    top_k: int = 20,
    client: Optional[Any] = None,
    embedding_cache: Optional[Dict[str, List[float]]] = None,
    precomputed_embedding: Optional[List[float]] = None,
    domain_filter: Optional[str] = None,
    suggestion_probability: Optional[float] = None,
    search_term: Optional[str] = None,
    hints: Optional[List[str]] = None,
    hint_probabilities: Optional[Dict[str, float]] = None,
    query_expansion: Optional[List[str]] = None,
    category_hints: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    LOCAL GUARD v2: Returns Top-K candidates instead of single match.
    
    This enables the disambiguation and resolver pipeline.
    
    CRITICAL: Uses canonical kind for consistent routing (fixes "Drug/Substance" → "Drug").
    
    Args:
        conn: Database connection
        span_text: Entity text from transcript
        entity_kind: Entity kind from NER
        clinic_id: Clinic ID for clinic-scoped search
        logger: Optional logger
        threshold: Similarity threshold
        top_k: Number of candidates to return
        
    Returns:
        List of candidate matches, sorted by score (best first)
    """
    # Pharmacy-Free Zone: Diagnosis and ReasonForVisit must NEVER search Pharmacy (defense in depth).
    kind = (entity_kind or "").strip()
    if kind in ("Diagnosis", "ReasonForVisit"):
        if logger:
            logger.debug("  ⚠️  Pharmacy-Free Zone: skipping inventory search for kind=%s", kind)
        return []
    if not clinic_id:
        if logger:
            logger.debug("  ⚠️  No clinic_id provided - skipping local inventory search")
        return []
    
    # No kind gate: clinic inventory is small; search all and let score + Judge decide (avoids missing links for any kind).
    try:
        # Normalize search terms for better matching; then add explicit search_term and hints so hints are grounded
        search_terms = normalize_search_term_for_local(span_text, entity_kind)
        search_terms = _build_search_terms_with_hints(
            search_terms, search_term, hints, hint_probabilities, max_hints=5
        )
        
        all_candidates = []
        location_id = 8 if clinic_id == 1 else clinic_id if clinic_id else None
        
        # Use precomputed embedding or cache to avoid per-entity API calls (batch embeddings at caller)
        # CRITICAL: Never call embed_text() here - all embeddings must be batched before calling this function
        use_vector = False
        vec_literal = None
        embedding = None
        if precomputed_embedding and to_pgvector_literal:
            embedding = precomputed_embedding
        elif embedding_cache is not None and span_text in embedding_cache and to_pgvector_literal:
            embedding = embedding_cache.get(span_text)
        # Fallback: on-demand embedding when cache miss (keeps vector retrieval enabled even if pre-batch misses some terms).
        if (not embedding) and LOCAL_VECTOR_ON_DEMAND_EMBED and embed_text and to_pgvector_literal:
            try:
                embedding = embed_text(span_text, client=client, logger=logger)
                if embedding and embedding_cache is not None:
                    embedding_cache[span_text] = embedding
            except Exception:
                embedding = None
        # If embedding is not found, fall back to trigram + phonetic matching only
        if not embedding and logger:
            logger.debug(f"  📊 No embedding found for '{span_text}' in cache - using trigram + phonetic matching only")
        if embedding and to_pgvector_literal:
            try:
                vec_literal = to_pgvector_literal(embedding)
                # Master Doc: use vector when pgvector is installed; else trigram + phonetic only.
                use_vector = bool(
                    vec_literal
                    and vec_literal.startswith("[")
                    and vector_extension_available(logger=logger)
                )
                if logger and use_vector:
                    logger.debug(f"  📊 Vector embedding used for '{span_text}' (cached or precomputed)")
            except Exception:
                use_vector = False
        
        # Domain: soft gate (boost) or hard filter. When USE_SOFT_GATE_LOCAL, do not filter in SQL; apply boost in Python.
        domain_filter_clause = ""
        domain_filter_params: tuple = ()
        if domain_filter and str(domain_filter).strip():
            if not USE_SOFT_GATE_LOCAL:
                domain_filter_clause = " AND (LOWER(TRIM(COALESCE(domain_key, ''))) IN (%s, 'general', '') OR TRIM(COALESCE(domain_key, '')) = '') "
                domain_filter_params = (domain_filter.strip().lower(),)

        # Hard category gate (production): restrict search to bucket group only; prevents cross-category hallucinations.
        category_filter_clause = ""
        category_filter_params: tuple = ()
        if USE_CATEGORY_HARD_GATE_LOCAL:
            allowed_categories = _get_hard_gate_categories(entity_kind, category_hints)
            if allowed_categories:
                placeholders = ", ".join(["%s"] * len(allowed_categories))
                # Normalize DB category same as Python: & -> and, so "Rehabilitation & Physiotherapy" matches "rehabilitation and physiotherapy"
                category_filter_clause = f" AND (LOWER(TRIM(REPLACE(COALESCE(category, ''), '&', 'and'))) IN ({placeholders})) "
                category_filter_params = tuple(allowed_categories)

        # Build SQL query with or without vector matching
        if use_vector and vec_literal:
            # SQL with vector matching (trigram + phonetic + vector)
            sql_exact = """
                WITH q AS (
                    SELECT
                        lower(%s) AS q_full,
                        split_part(lower(%s), ' ', 1) AS q_first,
                        metaphone(lower(%s), 10) AS q_mfull,
                        metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst,
                        dmetaphone(lower(%s)) AS q_dmfull,
                        dmetaphone_alt(lower(%s)) AS q_dmfull_alt,
                        dmetaphone(split_part(lower(%s), ' ', 1)) AS q_dmfirst,
                        dmetaphone_alt(split_part(lower(%s), ' ', 1)) AS q_dmfirst_alt
                ),
                scored AS (
                SELECT 
                    stock_id,
                    item_name,
                    trade_name,
                    category,
                    COALESCE(trade_name, item_name) AS display_name,
                    COALESCE(NULLIF(internal_description, ''), NULLIF(brief_description, ''), '') AS description,
                    GREATEST(
                        similarity(lower(item_name), q.q_full),
                        similarity(lower(COALESCE(trade_name, '')), q.q_full)
                    ) AS trigram_score,
                    GREATEST(
                        -- Phonetic similarity (metaphone edit-distance), scaled to max 0.8
                        GREATEST(
                            0.0,
                            0.8 * (
                                1.0 - (
                                    levenshtein(
                                        metaphone(split_part(lower(item_name), ' ', 1), 10),
                                        q.q_mfull
                                    )::float
                                    / GREATEST(
                                        length(metaphone(split_part(lower(item_name), ' ', 1), 10)),
                                        length(q.q_mfull),
                                        1
                                    )
                                )
                            )
                        ),
                        GREATEST(
                            0.0,
                            0.8 * (
                                1.0 - (
                                    levenshtein(
                                        metaphone(split_part(lower(COALESCE(trade_name, '')), ' ', 1), 10),
                                        q.q_mfull
                                    )::float
                                    / GREATEST(
                                        length(metaphone(split_part(lower(COALESCE(trade_name, '')), ' ', 1), 10)),
                                        length(q.q_mfull),
                                        1
                                    )
                                )
                            )
                        ),
                        GREATEST(
                            0.0,
                            0.8 * (
                                1.0 - (
                                    levenshtein(
                                        metaphone(lower(item_name), 10),
                                        q.q_mfull
                                    )::float
                                    / GREATEST(
                                        length(metaphone(lower(item_name), 10)),
                                        length(q.q_mfull),
                                        1
                                    )
                                )
                            )
                        ),
                        GREATEST(
                            0.0,
                            0.8 * (
                                1.0 - (
                                    levenshtein(
                                        metaphone(lower(COALESCE(trade_name, '')), 10),
                                        q.q_mfull
                                    )::float
                                    / GREATEST(
                                        length(metaphone(lower(COALESCE(trade_name, '')), 10)),
                                        length(q.q_mfull),
                                        1
                                    )
                                )
                            )
                        )
                    ) AS phonetic_score,
                    CASE 
                        WHEN vector_embedding IS NOT NULL 
                        THEN (1.0 - LEAST(vector_embedding <=> %s::vector, 1.0))
                        ELSE 0.0
                    END AS vector_score,
                    COALESCE(domain_key, '') AS domain_key
                FROM soap.inventory
                CROSS JOIN q
                WHERE (location_id = %s OR %s IS NULL)
                  AND (
                      lower(item_name) %% q.q_full
                      OR lower(COALESCE(trade_name, '')) %% q.q_full
                      OR similarity(lower(item_name), q.q_full) >= %s
                      OR similarity(lower(COALESCE(trade_name, '')), q.q_full) >= %s
                      OR dmetaphone(lower(item_name)) IN (q.q_dmfull, q.q_dmfull_alt)
                      OR dmetaphone(split_part(lower(item_name), ' ', 1)) IN (q.q_dmfirst, q.q_dmfirst_alt)
                      OR dmetaphone(lower(COALESCE(trade_name, ''))) IN (q.q_dmfull, q.q_dmfull_alt)
                      OR dmetaphone(split_part(lower(COALESCE(trade_name, '')), ' ', 1)) IN (q.q_dmfirst, q.q_dmfirst_alt)
                      OR (vector_embedding IS NOT NULL AND vector_embedding <=> %s::vector < 0.5)
                  )
                  {domain_filter_clause}{category_filter_clause}
                )
                SELECT * FROM scored
                ORDER BY GREATEST(trigram_score, phonetic_score, vector_score) DESC
                LIMIT %s;
            """
            sql_exact = sql_exact.replace("{domain_filter_clause}", domain_filter_clause).replace("{category_filter_clause}", category_filter_clause)
        else:
            # SQL without vector matching (trigram + phonetic only - original)
            sql_exact = """
                WITH q AS (
                    SELECT
                        lower(%s) AS q_full,
                        split_part(lower(%s), ' ', 1) AS q_first,
                        metaphone(lower(%s), 10) AS q_mfull,
                        metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst,
                        dmetaphone(lower(%s)) AS q_dmfull,
                        dmetaphone_alt(lower(%s)) AS q_dmfull_alt,
                        dmetaphone(split_part(lower(%s), ' ', 1)) AS q_dmfirst,
                        dmetaphone_alt(split_part(lower(%s), ' ', 1)) AS q_dmfirst_alt
                ),
                scored AS (
                SELECT 
                    stock_id,
                    item_name,
                    trade_name,
                    category,
                    COALESCE(trade_name, item_name) AS display_name,
                    COALESCE(NULLIF(internal_description, ''), NULLIF(brief_description, ''), '') AS description,
                    GREATEST(
                        similarity(lower(item_name), q.q_full),
                        similarity(lower(COALESCE(trade_name, '')), q.q_full)
                    ) AS trigram_score,
                    GREATEST(
                        GREATEST(
                            0.0,
                            0.8 * (
                                1.0 - (
                                    levenshtein(
                                        metaphone(split_part(lower(item_name), ' ', 1), 10),
                                        q.q_mfull
                                    )::float
                                    / GREATEST(
                                        length(metaphone(split_part(lower(item_name), ' ', 1), 10)),
                                        length(q.q_mfull),
                                        1
                                    )
                                )
                            )
                        ),
                        GREATEST(
                            0.0,
                            0.8 * (
                                1.0 - (
                                    levenshtein(
                                        metaphone(split_part(lower(COALESCE(trade_name, '')), ' ', 1), 10),
                                        q.q_mfull
                                    )::float
                                    / GREATEST(
                                        length(metaphone(split_part(lower(COALESCE(trade_name, '')), ' ', 1), 10)),
                                        length(q.q_mfull),
                                        1
                                    )
                                )
                            )
                        ),
                        GREATEST(
                            0.0,
                            0.8 * (
                                1.0 - (
                                    levenshtein(
                                        metaphone(lower(item_name), 10),
                                        q.q_mfull
                                    )::float
                                    / GREATEST(
                                        length(metaphone(lower(item_name), 10)),
                                        length(q.q_mfull),
                                        1
                                    )
                                )
                            )
                        ),
                        GREATEST(
                            0.0,
                            0.8 * (
                                1.0 - (
                                    levenshtein(
                                        metaphone(lower(COALESCE(trade_name, '')), 10),
                                        q.q_mfull
                                    )::float
                                    / GREATEST(
                                        length(metaphone(lower(COALESCE(trade_name, '')), 10)),
                                        length(q.q_mfull),
                                        1
                                    )
                                )
                            )
                        )
                    ) AS phonetic_score,
                    0.0 AS vector_score,
                    COALESCE(domain_key, '') AS domain_key
                FROM soap.inventory
                CROSS JOIN q
                WHERE (location_id = %s OR %s IS NULL)
                  AND (
                      lower(item_name) %% q.q_full
                      OR lower(COALESCE(trade_name, '')) %% q.q_full
                      OR similarity(lower(item_name), q.q_full) >= %s
                      OR similarity(lower(COALESCE(trade_name, '')), q.q_full) >= %s
                      OR dmetaphone(lower(item_name)) IN (q.q_dmfull, q.q_dmfull_alt)
                      OR dmetaphone(split_part(lower(item_name), ' ', 1)) IN (q.q_dmfirst, q.q_dmfirst_alt)
                      OR dmetaphone(lower(COALESCE(trade_name, ''))) IN (q.q_dmfull, q.q_dmfull_alt)
                      OR dmetaphone(split_part(lower(COALESCE(trade_name, '')), ' ', 1)) IN (q.q_dmfirst, q.q_dmfirst_alt)
                  )
                  {domain_filter_clause}{category_filter_clause}
                )
                SELECT * FROM scored
                ORDER BY GREATEST(trigram_score, phonetic_score) DESC
                LIMIT %s;
            """
            sql_exact = sql_exact.replace("{domain_filter_clause}", domain_filter_clause).replace("{category_filter_clause}", category_filter_clause)
        
        with conn.cursor() as cur:
            try:
                ensure_fuzzystrmatch(conn, logger)
            except Exception:
                pass
            
            # FIX: Track best candidate per stock_id (not just seen/not seen)
            # This allows later search terms to replace earlier matches if they have a better score
            best_candidates = {}  # stock_id -> candidate dict with best match_score
            
            for search_term in search_terms:
                # Prepare parameters based on whether vector matching is used
                if use_vector and vec_literal:
                    params = (
                        search_term, search_term, search_term, search_term,  # metaphone keys
                        search_term, search_term, search_term, search_term,  # double-metaphone keys
                        vec_literal,  # SELECT vector_score
                        location_id, location_id,  # WHERE location_id
                        LOCAL_TRGM_RECALL_THRESHOLD, LOCAL_TRGM_RECALL_THRESHOLD,  # similarity floor
                        vec_literal,  # WHERE vector
                    ) + domain_filter_params + category_filter_params + (top_k,)
                else:
                    params = (
                        search_term, search_term, search_term, search_term,  # metaphone keys
                        search_term, search_term, search_term, search_term,  # double-metaphone keys
                        location_id, location_id,  # WHERE location_id
                        LOCAL_TRGM_RECALL_THRESHOLD, LOCAL_TRGM_RECALL_THRESHOLD,  # similarity floor
                    ) + domain_filter_params + category_filter_params + (top_k,)
                
                cur.execute(sql_exact, params)
                rows = cur.fetchall()
                
                for row in rows:
                    # Unpack row (10 cols with domain_key, or 9/8 legacy)
                    domain_key_val = ""
                    if len(row) >= 10:
                        stock_id, item_name, trade_name, category, display_name, description, trigram_score, phonetic_score, vector_score, domain_key_val = row[:10]
                    elif len(row) == 9:
                        stock_id, item_name, trade_name, category, display_name, description, trigram_score, phonetic_score, vector_score = row
                    elif len(row) == 8:
                        stock_id, item_name, trade_name, category, display_name, description, trigram_score, phonetic_score = row
                        vector_score = 0.0
                    else:
                        if logger:
                            logger.debug(f"  ⚠️  Unexpected inventory row shape (len={len(row)}); skipping")
                        continue
                    domain_key_val = (domain_key_val or "").strip() if isinstance(domain_key_val, str) else ""
                    
                    trigram_score = float(trigram_score) if trigram_score else 0.0
                    phonetic_score = float(phonetic_score) if phonetic_score else 0.0
                    vector_score = float(vector_score) if vector_score else 0.0
                    
                    # Rerank with prefix-aware similarity: Jaro-Winkler is robust to ASR suffix noise.
                    jaro_winkler_score = _best_jw_score(search_term, display_name, item_name, trade_name)
                    # Keep phonetic_score for observability, but use JW for the lexical reranking layer.
                    match_score = max(trigram_score, jaro_winkler_score, vector_score)
                    
                    # Only process if score meets threshold
                    if match_score < 0.30:
                        continue
                    
                    # FIX: Keep best match score across all search terms
                    # If this stock_id hasn't been seen, or this match has a better score, use it
                    if stock_id not in best_candidates:
                        # First time seeing this stock_id - add it
                        best_candidates[stock_id] = {
                            "stock_id": stock_id,
                            "item_name": item_name,
                            "trade_name": trade_name,
                            "category": category,  # Include category for Kind-to-Category matching
                            "display_name": display_name,
                            "description": description or "",  # Include description for LLM Judge
                            "match_score": match_score,
                            "trigram_score": trigram_score,  # Include individual scores for decision flow
                            "phonetic_score": phonetic_score,
                            "jaro_winkler_score": jaro_winkler_score,
                            "vector_score": vector_score,
                            "domain_key": domain_key_val,  # For soft gate (domain boost)
                            "match_source": "local_inventory",
                        }
                    elif match_score > best_candidates[stock_id]["match_score"]:
                        # Replace with better score
                        old_score = best_candidates[stock_id]["match_score"]
                        if logger:
                            logger.debug(f"  🔄 Replaced candidate for stock_id {stock_id}: score {old_score:.3f} → {match_score:.3f} (search_term: '{search_term}')")
                        best_candidates[stock_id] = {
                            "stock_id": stock_id,
                            "item_name": item_name,
                            "trade_name": trade_name,
                            "category": category,
                            "display_name": display_name,
                            "description": description or "",
                            "match_score": match_score,
                            "trigram_score": trigram_score,
                            "phonetic_score": phonetic_score,
                            "jaro_winkler_score": jaro_winkler_score,
                            "vector_score": vector_score,
                            "domain_key": domain_key_val,
                            "match_source": "local_inventory",
                        }
                    else:
                        # Keep existing candidate but update individual scores if this search term has better scores
                        existing = best_candidates[stock_id]
                        if trigram_score > existing.get("trigram_score", 0.0):
                            existing["trigram_score"] = trigram_score
                        if phonetic_score > existing.get("phonetic_score", 0.0):
                            existing["phonetic_score"] = phonetic_score
                        if jaro_winkler_score > existing.get("jaro_winkler_score", 0.0):
                            existing["jaro_winkler_score"] = jaro_winkler_score
                        if vector_score > existing.get("vector_score", 0.0):
                            existing["vector_score"] = vector_score
            
            # Convert dictionary to list and sort by score (or final_score when soft gate applied)
            all_candidates = list(best_candidates.values())
            
            # Apply domain boost and suggestion boost to all candidates
            d_lower = (domain_filter or "").strip().lower() if domain_filter else None
            for c in all_candidates:
                # Domain boost
                domain_boost_val = 0.0
                if d_lower and USE_SOFT_GATE_LOCAL:
                    cand_d = (c.get("domain_key") or "").strip().lower() or ""
                    domain_boost_val = SOFT_GATE_LOCAL_DOMAIN_BOOST if cand_d == d_lower else 0.0
                    c["domain_boost"] = domain_boost_val
                # Category soft gate (local only): reward likely category matches, penalize mismatches
                category_boost_val = 0.0
                if USE_CATEGORY_SOFT_GATE_LOCAL:
                    category_boost_val, category_match_state, effective_cat_hints = _category_match_soft_score(
                        entity_kind,
                        c.get("category"),
                        category_hints,
                    )
                    c["category_boost"] = category_boost_val
                    c["category_match"] = category_match_state
                    if effective_cat_hints:
                        c["category_hints_used"] = effective_cat_hints
                
                # Suggestion boost: apply to candidates that match search_term or hints, weighted by hint probabilities
                suggestion_boost_val = 0.0
                if suggestion_probability is not None:
                    try:
                        from kb_ner_global_search import calculate_suggestion_boost
                        pname = (c.get("display_name") or c.get("preferred_name") or "").strip()
                        pname_lower = pname.lower() if pname else ""
                        
                        # Build suggestion terms list: search_term first, then hints
                        suggestion_terms = []
                        if search_term and search_term.strip():
                            suggestion_terms.append(search_term.strip().lower())
                        if hints:
                            for h in hints:
                                if h and str(h).strip():
                                    suggestion_terms.append(str(h).strip().lower())
                        
                        if suggestion_terms and pname_lower:
                            # Check if candidate name matches search_term first
                            search_term_lower = suggestion_terms[0] if suggestion_terms else ""
                            if search_term_lower and (search_term_lower in pname_lower or pname_lower in search_term_lower):
                                suggestion_boost_val = calculate_suggestion_boost(suggestion_probability)
                            else:
                                # Check hints and use hint probability for weighted boost
                                best_hint_prob = 0.0
                                matched_hint = None
                                hint_probs_dict = hint_probabilities or {}
                                for hint_term in suggestion_terms[1:]:  # Skip search_term (first element)
                                    if hint_term in pname_lower or pname_lower in hint_term:
                                        hint_prob = hint_probs_dict.get(hint_term, 0.0)
                                        if hint_prob > best_hint_prob:
                                            best_hint_prob = hint_prob
                                            matched_hint = hint_term

                                if matched_hint:
                                    # Weight the boost by hint probability: higher hint prob = higher boost
                                    base_boost = calculate_suggestion_boost(suggestion_probability)
                                    suggestion_boost_val = base_boost * best_hint_prob
                                    if logger:
                                        logger.debug(f"Local inventory: candidate '{pname}' matched hint '{matched_hint}' (prob={best_hint_prob:.2f}) -> weighted boost={suggestion_boost_val:.3f}")
                                else:
                                    # Check query_expansion terms (phonetic/ASR correction): fixed-weight boost
                                    expansion_terms = (query_expansion or [])[:3]
                                    for exp_term in expansion_terms:
                                        if not exp_term or not isinstance(exp_term, str):
                                            continue
                                        exp_lower = exp_term.strip().lower()
                                        if exp_lower in pname_lower or pname_lower in exp_lower:
                                            base_boost = calculate_suggestion_boost(suggestion_probability)
                                            qe_weight = float(os.getenv("QUERY_EXPANSION_BOOST_WEIGHT", "0.8"))
                                            suggestion_boost_val = base_boost * qe_weight
                                            if logger:
                                                logger.debug(f"Local inventory: candidate matched query_expansion '{exp_term}' -> boost={suggestion_boost_val:.3f}")
                                            break
                    except Exception as e:
                        if logger:
                            logger.debug(f"Local inventory: suggestion boost calculation failed: {e}")
                
                c["suggestion_boost"] = suggestion_boost_val
                
                # Calculate final_score: base match_score (with weight if soft gate) + domain boost + suggestion boost
                if d_lower and USE_SOFT_GATE_LOCAL:
                    c["final_score"] = (c["match_score"] * SOFT_GATE_LOCAL_BASE_WEIGHT) + domain_boost_val + category_boost_val + suggestion_boost_val
                else:
                    c["final_score"] = c["match_score"] + category_boost_val + suggestion_boost_val
            
            # Sort by final_score (includes domain + suggestion boost)
            all_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
            return all_candidates[:top_k]
                
    except Exception as e:
        msg = str(e).lower()
        if 'type "vector" does not exist' in msg or ("vector" in msg and "does not exist" in msg):
            invalidate_vector_extension_cache()
        try:
            conn.rollback()
        except Exception:
            pass
        if logger:
            logger.warning(f"  ⚠️  Local inventory search failed: {e}")
        return []


def search_local_inventory(
    conn,
    span_text: str,
    entity_kind: str,
    clinic_id: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    threshold: float = 0.50,
) -> Optional[Dict[str, Any]]:
    """
    LOCAL GUARD (Step A): Fast search in clinic inventory for Drug entities.
    
    This is the "Billing Route" - ensures we only link to items the clinic actually sells.
    Prevents "Ghost Drugs" (KB concepts not in clinic inventory).
    
    Now uses Top-K approach internally and applies deterministic resolver.
    
    Args:
        conn: Database connection
        span_text: Entity text from transcript (e.g., "Cortex Capsules")
        entity_kind: Entity kind from NER (e.g., "Drug", "Procedure")
        clinic_id: Clinic ID for clinic-scoped search
        logger: Optional logger
        threshold: Similarity threshold (default: 0.50)
        
    Returns:
        Dict with stock_id, item_name, match_score, or None if no match
    """
    # Use Top-K search internally
    candidates = search_local_inventory_topk(
        conn, span_text, entity_kind, clinic_id, logger, threshold, top_k=20
    )
    
    if not candidates:
        return None
    
    # Apply deterministic resolver to select best SKU
    best_match = resolve_default_sku(candidates, span_text, entity_kind, logger)
    
    if best_match and best_match.get("match_score", 0) >= threshold:
        if logger:
            logger.info(f"  ✅ Local Guard: '{span_text}' → '{best_match['display_name']}' (stock_id: {best_match['stock_id']}, score: {best_match['match_score']:.3f})")
        return best_match
    
    # If best match is below threshold, return None (don't use low confidence)
    if logger:
        logger.debug(f"  ⚠️  Local Guard: Best match for '{span_text}' below threshold {threshold} (score: {best_match.get('match_score', 0):.3f})")
    return None


def search_local_services_topk(
    conn,
    span_text: str,
    entity_kind: str,
    clinic_id: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    threshold: float = 0.50,
    top_k: int = 20,
    client: Optional[Any] = None,
    embedding_cache: Optional[Dict[str, List[float]]] = None,
    precomputed_embedding: Optional[List[float]] = None,
    domain_filter: Optional[str] = None,
    suggestion_probability: Optional[float] = None,
    search_term: Optional[str] = None,
    hints: Optional[List[str]] = None,
    hint_probabilities: Optional[Dict[str, float]] = None,
    query_expansion: Optional[List[str]] = None,
    category_hints: Optional[List[str]] = None,
    service_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    LOCAL GUARD v2: Returns Top-K service candidates instead of single match.
    
    This enables the disambiguation and resolver pipeline.
    
    CRITICAL: Uses canonical kind for consistent routing (fixes "Procedure/Service" → "Procedure").
    
    Args:
        conn: Database connection
        span_text: Entity text from transcript
        entity_kind: Entity kind from NER
        clinic_id: Clinic ID for clinic-scoped search
        logger: Optional logger
        threshold: Similarity threshold
        top_k: Number of candidates to return
        
    Returns:
        List of candidate matches, sorted by score (best first)
    """
    if not clinic_id:
        if logger:
            logger.debug("  ⚠️  No clinic_id provided - skipping local service search")
        return []
    
    # No kind gate: clinic service_master is small; search all and let score + Judge decide (avoids missing links for any kind).
    try:
        # Normalize search terms for better matching; then add explicit search_term and hints so hints are grounded
        search_terms = normalize_search_term_for_local(span_text, entity_kind)
        search_terms = _build_search_terms_with_hints(
            search_terms, search_term, hints, hint_probabilities, max_hints=5
        )
        
        all_candidates = []
        
        # Use precomputed embedding or cache to avoid per-entity API calls (batch embeddings at caller)
        # CRITICAL: Never call embed_text() here - all embeddings must be batched before calling this function
        use_vector = False
        vec_literal = None
        embedding = None
        if precomputed_embedding and to_pgvector_literal:
            embedding = precomputed_embedding
        elif embedding_cache is not None and span_text in embedding_cache and to_pgvector_literal:
            embedding = embedding_cache.get(span_text)
        # Fallback: on-demand embedding when cache miss (keeps vector retrieval enabled even if pre-batch misses some terms).
        if (not embedding) and LOCAL_VECTOR_ON_DEMAND_EMBED and embed_text and to_pgvector_literal:
            try:
                embedding = embed_text(span_text, client=client, logger=logger)
                if embedding and embedding_cache is not None:
                    embedding_cache[span_text] = embedding
            except Exception:
                embedding = None
        # If embedding is not found, fall back to trigram + phonetic matching only
        if not embedding and logger:
            logger.debug(f"  📊 No embedding found for '{span_text}' in cache - using trigram + phonetic matching only")
        if embedding and to_pgvector_literal:
            try:
                vec_literal = to_pgvector_literal(embedding)
                if vec_literal and vec_literal.startswith("["):
                    use_vector = bool(vector_extension_available(logger=logger))
                    if logger and use_vector:
                        logger.debug(f"  📊 Vector embedding used for service '{span_text}' (cached or precomputed)")
            except Exception:
                use_vector = False

        domain_filter_clause = ""
        domain_filter_params_svc: tuple = ()
        if domain_filter and str(domain_filter).strip():
            if not USE_SOFT_GATE_LOCAL:
                domain_filter_clause = " AND (LOWER(TRIM(COALESCE(domain_key, ''))) IN (%s, 'general', '') OR TRIM(COALESCE(domain_key, '')) = '') "
                domain_filter_params_svc = (domain_filter.strip().lower(),)

        # Hard category gate for services (same bucket-group logic as inventory)
        category_filter_clause_svc = ""
        category_filter_params_svc: tuple = ()
        if USE_CATEGORY_HARD_GATE_LOCAL:
            allowed_categories_svc = _get_hard_gate_categories(
                entity_kind,
                category_hints,
                service_type=service_type,
                is_service_search=True,
            )
            if allowed_categories_svc:
                placeholders_svc = ", ".join(["%s"] * len(allowed_categories_svc))
                # Normalize DB category same as Python: & -> and, so "Rehabilitation & Physiotherapy" matches
                category_filter_clause_svc = f" AND (LOWER(TRIM(REPLACE(COALESCE(category, ''), '&', 'and'))) IN ({placeholders_svc})) "
                category_filter_params_svc = tuple(allowed_categories_svc)

        if use_vector and vec_literal:
            # SQL with vector matching (trigram + phonetic + vector)
            sql = """
                WITH q AS (
                    SELECT
                        lower(%s) AS q_full,
                        split_part(lower(%s), ' ', 1) AS q_first,
                        metaphone(lower(%s), 10) AS q_mfull,
                        metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst,
                        dmetaphone(lower(%s)) AS q_dmfull,
                        dmetaphone_alt(lower(%s)) AS q_dmfull_alt,
                        dmetaphone(split_part(lower(%s), ' ', 1)) AS q_dmfirst,
                        dmetaphone_alt(split_part(lower(%s), ' ', 1)) AS q_dmfirst_alt
                ),
                scored AS (
                    SELECT 
                        service_id,
                        procedure_name,
                        category,
                        procedure_name AS display_name,
                        COALESCE(NULLIF(internal_description, ''), NULLIF(remarks, ''), '') AS description,
                        similarity(lower(procedure_name), q.q_full) AS trigram_score,
                        GREATEST(
                            0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(procedure_name), ' ', 1), 10), q.q_mfirst)::float
                                         / GREATEST(length(metaphone(split_part(lower(procedure_name), ' ', 1), 10)), length(q.q_mfirst), 1))),
                            0.8 * (1.0 - (levenshtein(metaphone(lower(procedure_name), 10), q.q_mfull)::float
                                         / GREATEST(length(metaphone(lower(procedure_name), 10)), length(q.q_mfull), 1)))
                        ) AS phonetic_score,
                        CASE 
                            WHEN vector_embedding IS NOT NULL 
                            THEN (1.0 - LEAST(vector_embedding <=> %s::vector, 1.0))
                            ELSE 0.0
                        END AS vector_score,
                        COALESCE(domain_key, '') AS domain_key
                    FROM soap.service_master
                    CROSS JOIN q
                    WHERE (
                        lower(procedure_name) %% q.q_full
                        OR similarity(lower(procedure_name), q.q_full) >= %s
                        OR dmetaphone(split_part(lower(procedure_name), ' ', 1)) IN (q.q_dmfirst, q.q_dmfirst_alt)
                        OR dmetaphone(lower(procedure_name)) IN (q.q_dmfull, q.q_dmfull_alt)
                        OR (vector_embedding IS NOT NULL AND vector_embedding <=> %s::vector < 0.5)
                    )
                    {domain_filter_clause_svc}{category_filter_clause_svc}
                )
                SELECT
                    service_id,
                    procedure_name,
                    category,
                    display_name,
                    description,
                    trigram_score,
                    phonetic_score,
                    vector_score,
                    domain_key,
                    GREATEST(trigram_score, phonetic_score, vector_score) AS match_score
                FROM scored
                ORDER BY match_score DESC
                LIMIT %s;
            """
            sql = sql.replace("{domain_filter_clause_svc}", domain_filter_clause).replace("{category_filter_clause_svc}", category_filter_clause_svc)
        else:
            # SQL without vector matching (trigram + phonetic only)
            sql = """
                WITH q AS (
                    SELECT
                        lower(%s) AS q_full,
                        split_part(lower(%s), ' ', 1) AS q_first,
                        metaphone(lower(%s), 10) AS q_mfull,
                        metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst,
                        dmetaphone(lower(%s)) AS q_dmfull,
                        dmetaphone_alt(lower(%s)) AS q_dmfull_alt,
                        dmetaphone(split_part(lower(%s), ' ', 1)) AS q_dmfirst,
                        dmetaphone_alt(split_part(lower(%s), ' ', 1)) AS q_dmfirst_alt
                ),
                scored AS (
                    SELECT 
                        service_id,
                        procedure_name,
                        category,
                        procedure_name AS display_name,
                        COALESCE(NULLIF(internal_description, ''), NULLIF(remarks, ''), '') AS description,
                        similarity(lower(procedure_name), q.q_full) AS trigram_score,
                        GREATEST(
                            0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(procedure_name), ' ', 1), 10), q.q_mfirst)::float
                                         / GREATEST(length(metaphone(split_part(lower(procedure_name), ' ', 1), 10)), length(q.q_mfirst), 1))),
                            0.8 * (1.0 - (levenshtein(metaphone(lower(procedure_name), 10), q.q_mfull)::float
                                         / GREATEST(length(metaphone(lower(procedure_name), 10)), length(q.q_mfull), 1)))
                        ) AS phonetic_score,
                        0.0 AS vector_score,
                        COALESCE(domain_key, '') AS domain_key
                    FROM soap.service_master
                    CROSS JOIN q
                    WHERE (
                        lower(procedure_name) %% q.q_full
                        OR similarity(lower(procedure_name), q.q_full) >= %s
                        OR dmetaphone(split_part(lower(procedure_name), ' ', 1)) IN (q.q_dmfirst, q.q_dmfirst_alt)
                        OR dmetaphone(lower(procedure_name)) IN (q.q_dmfull, q.q_dmfull_alt)
                    )
                    {domain_filter_clause_svc}{category_filter_clause_svc}
                )
                SELECT
                    service_id,
                    procedure_name,
                    category,
                    display_name,
                    description,
                    trigram_score,
                    phonetic_score,
                    vector_score,
                    domain_key,
                    GREATEST(trigram_score, phonetic_score) AS match_score
                FROM scored
                ORDER BY match_score DESC
                LIMIT %s;
            """
            sql = sql.replace("{domain_filter_clause_svc}", domain_filter_clause).replace("{category_filter_clause_svc}", category_filter_clause_svc)
        
        with conn.cursor() as cur:
            try:
                ensure_fuzzystrmatch(conn, logger)
            except Exception:
                pass
            
            seen_service_ids = set()
            for search_term in search_terms:
                # Prepare parameters based on whether vector matching is used
                if use_vector and vec_literal:
                    params = (
                        search_term, search_term, search_term, search_term,  # metaphone keys
                        search_term, search_term, search_term, search_term,  # double-metaphone keys
                        vec_literal,  # SELECT vector_score (1)
                        LOCAL_TRGM_RECALL_THRESHOLD,  # similarity floor
                        vec_literal,  # WHERE vector (1)
                    ) + domain_filter_params_svc + category_filter_params_svc + (top_k,)
                else:
                    params = (
                        search_term, search_term, search_term, search_term,  # metaphone keys
                        search_term, search_term, search_term, search_term,  # double-metaphone keys
                        LOCAL_TRGM_RECALL_THRESHOLD,  # similarity floor
                    ) + domain_filter_params_svc + category_filter_params_svc + (top_k,)
                
                cur.execute(sql, params)
                rows = cur.fetchall()
                
                for row in rows:
                    # Unpack row (10 cols with domain_key, or 9 legacy)
                    domain_key_svc = ""
                    if len(row) >= 10:
                        service_id, procedure_name, category, display_name, description, trigram_score, phonetic_score, vector_score, domain_key_svc, match_score = row[:10]
                    else:
                        service_id, procedure_name, category, display_name, description, trigram_score, phonetic_score, vector_score, match_score = row[:9]
                    domain_key_svc = (domain_key_svc or "").strip() if isinstance(domain_key_svc, str) else ""
                    
                    trigram_score = float(trigram_score) if trigram_score else 0.0
                    phonetic_score = float(phonetic_score) if phonetic_score else 0.0
                    vector_score = float(vector_score) if vector_score else 0.0
                    jaro_winkler_score = _best_jw_score(search_term, display_name, procedure_name)
                    match_score = max(trigram_score, jaro_winkler_score, vector_score)
                    
                    if match_score >= 0.30:  # Include all candidates above 30%
                        if service_id not in seen_service_ids:
                            seen_service_ids.add(service_id)
                            all_candidates.append({
                                "service_id": service_id,
                                "service_name": procedure_name,
                                "category": category,
                                "display_name": display_name,
                                "description": description or "",
                                "match_score": match_score,
                                "trigram_score": trigram_score,
                                "phonetic_score": phonetic_score,
                                "jaro_winkler_score": jaro_winkler_score,
                                "vector_score": vector_score,
                                "domain_key": domain_key_svc,
                                "match_source": "local_service",
                            })
        
        # Apply domain boost and suggestion boost to all candidates
        d_lower = (domain_filter or "").strip().lower() if domain_filter else None
        for c in all_candidates:
            # Domain boost
            domain_boost_val = 0.0
            if d_lower and USE_SOFT_GATE_LOCAL:
                cand_d = (c.get("domain_key") or "").strip().lower() or ""
                domain_boost_val = SOFT_GATE_LOCAL_DOMAIN_BOOST if cand_d == d_lower else 0.0
                c["domain_boost"] = domain_boost_val
            # Category soft gate for services as well (kept soft to avoid over-filtering)
            category_boost_val = 0.0
            if USE_CATEGORY_SOFT_GATE_LOCAL:
                category_boost_val, category_match_state, effective_cat_hints = _category_match_soft_score(
                    entity_kind,
                    c.get("category"),
                    category_hints,
                )
                c["category_boost"] = category_boost_val
                c["category_match"] = category_match_state
                if effective_cat_hints:
                    c["category_hints_used"] = effective_cat_hints
            
            # Suggestion boost: apply to candidates that match search_term or hints, weighted by hint probabilities
            suggestion_boost_val = 0.0
            if suggestion_probability is not None:
                try:
                    from kb_ner_global_search import calculate_suggestion_boost
                    pname = (c.get("display_name") or c.get("preferred_name") or "").strip()
                    pname_lower = pname.lower() if pname else ""
                    
                    # Build suggestion terms list: search_term first, then hints
                    suggestion_terms = []
                    if search_term and search_term.strip():
                        suggestion_terms.append(search_term.strip().lower())
                    if hints:
                        for h in hints:
                            if h and str(h).strip():
                                suggestion_terms.append(str(h).strip().lower())
                    
                    if suggestion_terms and pname_lower:
                        # Check if candidate name matches search_term first
                        search_term_lower = suggestion_terms[0] if suggestion_terms else ""
                        if search_term_lower and (search_term_lower in pname_lower or pname_lower in search_term_lower):
                            suggestion_boost_val = calculate_suggestion_boost(suggestion_probability)
                        else:
                            # Check hints and use hint probability for weighted boost
                            best_hint_prob = 0.0
                            matched_hint = None
                            hint_probs_dict = hint_probabilities or {}
                            for hint_term in suggestion_terms[1:]:  # Skip search_term (first element)
                                if hint_term in pname_lower or pname_lower in hint_term:
                                    hint_prob = hint_probs_dict.get(hint_term, 0.0)
                                    if hint_prob > best_hint_prob:
                                        best_hint_prob = hint_prob
                                        matched_hint = hint_term
                            
                            if matched_hint:
                                # Weight the boost by hint probability: higher hint prob = higher boost
                                base_boost = calculate_suggestion_boost(suggestion_probability)
                                suggestion_boost_val = base_boost * best_hint_prob
                                if logger:
                                    logger.debug(f"Local services: candidate '{pname}' matched hint '{matched_hint}' (prob={best_hint_prob:.2f}) → weighted boost={suggestion_boost_val:.3f}")
                            else:
                                # Check query_expansion terms (phonetic/ASR correction): fixed-weight boost
                                expansion_terms = (query_expansion or [])[:3]
                                for exp_term in expansion_terms:
                                    if not exp_term or not isinstance(exp_term, str):
                                        continue
                                    exp_lower = exp_term.strip().lower()
                                    if exp_lower in pname_lower or pname_lower in exp_lower:
                                        base_boost = calculate_suggestion_boost(suggestion_probability)
                                        qe_weight = float(os.getenv("QUERY_EXPANSION_BOOST_WEIGHT", "0.8"))
                                        suggestion_boost_val = base_boost * qe_weight
                                        if logger:
                                            logger.debug(f"Local services: candidate matched query_expansion '{exp_term}' -> boost={suggestion_boost_val:.3f}")
                                        break
                except Exception as e:
                    if logger:
                        logger.debug(f"Local services: suggestion boost calculation failed: {e}")
            
            c["suggestion_boost"] = suggestion_boost_val
            
            # Calculate final_score: base match_score (with weight if soft gate) + domain boost + suggestion boost
            if d_lower and USE_SOFT_GATE_LOCAL:
                c["final_score"] = (c["match_score"] * SOFT_GATE_LOCAL_BASE_WEIGHT) + domain_boost_val + category_boost_val + suggestion_boost_val
            else:
                c["final_score"] = c["match_score"] + category_boost_val + suggestion_boost_val
        
        # Sort by final_score (includes domain + suggestion boost)
        all_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return all_candidates[:top_k]
                
    except Exception as e:
        msg = str(e).lower()
        if 'type "vector" does not exist' in msg or ("vector" in msg and "does not exist" in msg):
            invalidate_vector_extension_cache()
        try:
            conn.rollback()
        except Exception:
            pass
        if logger:
            logger.warning(f"  ⚠️  Local service search failed: {e}")
        return []


def search_local_services(
    conn,
    span_text: str,
    entity_kind: str,
    clinic_id: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    threshold: float = 0.50,
) -> Optional[Dict[str, Any]]:
    """
    LOCAL GUARD (Step A): Fast search in clinic service_master for Procedure/Service entities.
    
    This is the "Billing Route" - ensures we only link to services the clinic actually offers.
    Prevents hallucinations like "Oral Examination" when clinic only offers "General Consultation".
    
    Now uses Top-K approach internally.
    
    Args:
        conn: Database connection
        span_text: Entity text from transcript (e.g., "examination", "anal gland expression")
        entity_kind: Entity kind from NER (e.g., "Procedure", "Service")
        clinic_id: Clinic ID for clinic-scoped search
        logger: Optional logger
        threshold: Similarity threshold (default: 0.50)
        
    Returns:
        Dict with service_id, service_name, match_score, or None if no match
    """
    # Use Top-K search internally
    candidates = search_local_services_topk(
        conn, span_text, entity_kind, clinic_id, logger, threshold, top_k=20
    )
    
    if not candidates:
        return None
    
    # For services, just pick the best match (no resolver needed)
    best_match = candidates[0]
    
    if best_match and best_match.get("match_score", 0) >= threshold:
        if logger:
            logger.info(f"  ✅ Local Guard: '{span_text}' → '{best_match['display_name']}' (service_id: {best_match['service_id']}, score: {best_match['match_score']:.3f})")
        return best_match
    
    # If best match is below threshold, return None (don't use low confidence)
    if logger:
        logger.debug(f"  ⚠️  Local Guard: Best match for '{span_text}' below threshold {threshold} (score: {best_match.get('match_score', 0):.3f})")
    return None


def search_local_topk_union(
    text: str, 
    entity_kind: str, 
    clinic_id: int, 
    conn,
    topk: int = 8,
    logger: Optional[logging.Logger] = None,
    domain_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Combine inventory + services search results.
    
    Args:
        text: Search text
        entity_kind: Entity kind
        clinic_id: Clinic ID
        conn: Database connection (required)
        topk: Number of results to return
        logger: Optional logger
        
    Returns:
        Combined list of local candidates
    """
    if not conn:
        if logger:
            logger.warning("  ⚠️  No database connection provided for local search")
        return []
    
    candidates = []
    
    # No kind gate: clinic inventory and service_master are small; search both and let score + Judge decide.
    try:
        inv_candidates = search_local_inventory_topk(
            conn, text, entity_kind, clinic_id, logger=logger, top_k=topk,
            domain_filter=domain_filter
        )
        if inv_candidates:
            candidates.extend(inv_candidates)
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠️  Local inventory search failed: {e}")
    try:
        srv_candidates = search_local_services_topk(
            conn, text, entity_kind, clinic_id, logger=logger, top_k=topk,
            domain_filter=domain_filter
        )
        if srv_candidates:
            candidates.extend(srv_candidates)
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠️  Local services search failed: {e}")
    
    # Sort by final_score (includes domain + suggestion boost) or match_score, and return top-k
    candidates.sort(key=lambda x: float(x.get("final_score", x.get("match_score", 0)) or 0), reverse=True)
    return candidates[:topk]

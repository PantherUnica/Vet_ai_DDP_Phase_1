"""
Global KB search utilities.

This module handles:
- Exact concept lookup
- Trigram similarity search
- RAG-based search
- Embedding-based search (OpenAI vector + lexical)
- Hybrid search (combining methods)
- Learned alias management

Tri-retrieval policy (required): We always run the full tri-retrieval process:
  1. Phonetic – pg_trgm similarity + metaphone (fuzzystrmatch) for ASR variants (e.g. ultralining → Ortolani).
  2. Fuzzy/trigram – trigram similarity for typo and spelling variants.
  3. Vector – OpenAI embedding search for semantic matches.
Do NOT gate RAG/semantic behind "primary path fails" or cap top-K in a way that skips any of these three.
Candidates from all three are merged and reranked (multi-signal: tri + pho + openai weights; optional domain/relation).
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

try:
    import psycopg2
    from psycopg2 import Error as Psycopg2Error
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    Psycopg2Error = Exception
    PSYCOPG2_AVAILABLE = False

from kb_ner_db import get_pg_conn, ensure_pg_trgm, ensure_fuzzystrmatch
from kb_ner_embeddings import embed_text, to_pgvector_literal
from kb_ner_routing import canonicalize_kind


def calculate_suggestion_boost(suggestion_probability: Optional[float]) -> float:
    """
    Calculate suggestion boost score based on suggestion_probability.
    
    Boost rules:
    - suggestion_probability >= 0.90: boost = 0.25
    - suggestion_probability >= 0.80 and < 0.90: boost = 0.15
    - Otherwise: boost = 0.0
    
    Args:
        suggestion_probability: Probability score (0.0 to 1.0) from Brain NER
        
    Returns:
        Boost score to add to grounding match scores
    """
    if suggestion_probability is None:
        return 0.0
    prob = float(suggestion_probability)
    if prob >= 0.90:
        return 0.25
    elif prob >= 0.80:
        return 0.15
    else:
        return 0.0

# Dual-embedding (OpenAI-only in current flow)
KB_USE_DUAL_EMBEDDING = os.getenv("KB_USE_DUAL_EMBEDDING", "true").lower() in ("1", "true", "yes")
try:
    _w_openai = float(os.getenv("KB_DUAL_EMBEDDING_OPENAI_WEIGHT", "0.4"))
except Exception:
    _w_openai = 0.4
KB_DUAL_EMBEDDING_OPENAI_WEIGHT = max(0.0, min(1.0, _w_openai))

# Local-only mode: no global KB (kb.concepts) or global routing; all grounding is local (inventory + service_master).
LOCAL_ONLY = os.getenv("LOCAL_ONLY", "true").lower() in ("1", "true", "yes")

# Constants for dual-sync routing
INTENT_KINDS = {"Reason", "ReasonForVisit", "Reason_for_Visit"}
REASON_ALLOWED_KB_KINDS = {"Condition", "Finding", "Procedure", "Service", "Anatomy"}

# Soft Gate: domain as boost (not hard filter) for resilient clinical grounding
# final_score = base_score * SOFT_GATE_BASE_WEIGHT + (SOFT_GATE_DOMAIN_BOOST if domain match else 0)
try:
    _sg_base = float(os.getenv("SOFT_GATE_BASE_WEIGHT", "0.8"))
    _sg_boost = float(os.getenv("SOFT_GATE_DOMAIN_BOOST", "0.2"))
    _sg_thresh = float(os.getenv("SOFT_GATE_THRESHOLD", "0.35"))
except Exception:
    _sg_base, _sg_boost, _sg_thresh = 0.8, 0.2, 0.35
SOFT_GATE_BASE_WEIGHT = max(0.0, min(1.0, _sg_base))
SOFT_GATE_DOMAIN_BOOST = max(0.0, min(1.0, _sg_boost))
SOFT_GATE_THRESHOLD = max(0.0, min(1.0, _sg_thresh))
SOFT_GATE_ENABLED = os.getenv("SOFT_GATE_ENABLED", "true").lower() in ("1", "true", "yes")

# Global search: OpenAI vector + lexical (tri/pho); soft-gate (base*0.8 + domain_boost*0.2)
# Number of candidates to fetch with OpenAI
GLOBAL_TWO_STAGE_FETCH_K = int(os.getenv("GLOBAL_TWO_STAGE_FETCH_K", "50"))
# Ensemble fetch: when domain is set, Stage 1 = semantic bucket + phonetic/trigram bucket (no hardcoded pairs)
GLOBAL_TWO_STAGE_ENSEMBLE_FETCH = os.getenv("GLOBAL_TWO_STAGE_ENSEMBLE_FETCH", "true").lower() in ("1", "true", "yes")
GLOBAL_TWO_STAGE_SEMANTIC_BUCKET_SIZE = int(os.getenv("GLOBAL_TWO_STAGE_SEMANTIC_BUCKET_SIZE", "25"))
GLOBAL_TWO_STAGE_PHONETIC_BUCKET_SIZE = int(os.getenv("GLOBAL_TWO_STAGE_PHONETIC_BUCKET_SIZE", "25"))
# Trigram similarity threshold for phonetic bucket.
# Specialty (orthopedic, cardiology, etc.): 0.1 restores Ortolani for "ultralining test"; domain reranker sinks noise.
# General / no domain: 0.2 to minimize noise and latency (fewer candidates into Stage 2).
GLOBAL_TWO_STAGE_PHONETIC_SIM_THRESHOLD = float(os.getenv("GLOBAL_TWO_STAGE_PHONETIC_SIM_THRESHOLD", "0.1"))
GLOBAL_TWO_STAGE_PHONETIC_SIM_THRESHOLD_GENERAL = float(os.getenv("GLOBAL_TWO_STAGE_PHONETIC_SIM_THRESHOLD_GENERAL", "0.2"))
# Conditional multi-pass: when True and specialty, run phonetic at GENERAL first; if 0 rows, escalate to SPECIALTY threshold (0.1).
GLOBAL_TWO_STAGE_PHONETIC_ESCALATION = os.getenv("GLOBAL_TWO_STAGE_PHONETIC_ESCALATION", "true").lower() in ("1", "true", "yes")
# Multi-signal ranker: tri + pho + openai + domain + relation; generic penalty for "test/exam" noise
GLOBAL_TWO_STAGE_MULTI_SIGNAL_RANKER = os.getenv("GLOBAL_TWO_STAGE_MULTI_SIGNAL_RANKER", "true").lower() in ("1", "true", "yes")
try:
    _w_tri = float(os.getenv("GLOBAL_TWO_STAGE_TRIGRAM_WEIGHT", "0.10"))
    _w_pho = float(os.getenv("GLOBAL_TWO_STAGE_PHONETIC_WEIGHT", "0.20"))
    _w_oa = float(os.getenv("GLOBAL_TWO_STAGE_OPENAI_WEIGHT", "0.55"))
    _w_rel = float(os.getenv("GLOBAL_TWO_STAGE_RELATION_WEIGHT", "0.15"))
    _w_dom = float(os.getenv("GLOBAL_TWO_STAGE_DOMAIN_WEIGHT", "0.20"))
except Exception:
    _w_tri, _w_pho, _w_oa, _w_rel, _w_dom = 0.10, 0.20, 0.55, 0.15, 0.20
GLOBAL_TWO_STAGE_TRIGRAM_WEIGHT = max(0.0, min(1.0, _w_tri))
GLOBAL_TWO_STAGE_PHONETIC_WEIGHT = max(0.0, min(1.0, _w_pho))
GLOBAL_TWO_STAGE_OPENAI_WEIGHT = max(0.0, min(1.0, _w_oa))
GLOBAL_TWO_STAGE_RELATION_WEIGHT = max(0.0, min(1.0, _w_rel))
GLOBAL_TWO_STAGE_DOMAIN_WEIGHT = max(0.0, min(1.0, _w_dom))
# Backward compat: LEXICAL = TRI+PHO
GLOBAL_TWO_STAGE_LEXICAL_WEIGHT = GLOBAL_TWO_STAGE_TRIGRAM_WEIGHT + GLOBAL_TWO_STAGE_PHONETIC_WEIGHT
# Weight redistributed from removed vector reranker to openai/tri/pho (keeps total ~1.0)
REDISTRIBUTED_VECTOR_WEIGHT = 0.35
# Generic procedure penalty: reduce unfair advantage of "test/exam/assessment" candidates (anti-noise, not ASR mapping)
try:
    _gproc = float(os.getenv("GLOBAL_GENERIC_PROC_PENALTY", "0.12"))
except Exception:
    _gproc = 0.08
GLOBAL_GENERIC_PROC_PENALTY = max(0.0, min(0.3, _gproc))
GENERIC_PROC_TOKENS = frozenset(
    os.getenv("GLOBAL_GENERIC_PROC_TOKENS", "test,tests,exam,examination,assessment,scoring,evaluation,check").lower().replace(" ", "").split(",")
)
# Only apply generic penalty to Procedure/DiagnosticTest/Service (avoid suppressing e.g. "joint assessment" in other kinds)
GENERIC_PENALTY_KINDS = frozenset(
    os.getenv("GLOBAL_GENERIC_PENALTY_KINDS", "Procedure,DiagnosticTest,Service").replace(" ", "").split(",")
)
# Cross-domain friction: soft penalty when candidate is in a different specialty (Negative Domain Signaling)
try:
    _friction = float(os.getenv("GLOBAL_TWO_STAGE_DOMAIN_FRICTION", "0.1"))
except Exception:
    _friction = 0.1
GLOBAL_TWO_STAGE_DOMAIN_FRICTION = max(0.0, min(0.5, _friction))
# Domains that get neither boost nor penalty (generic / multi-system)
GLOBAL_TWO_STAGE_DOMAIN_NEUTRAL = frozenset(os.getenv("GLOBAL_TWO_STAGE_DOMAIN_NEUTRAL", "general,common").lower().replace(" ", "").split(","))
# Cross-domain penalty for soft gate: when candidate has domain_key set and it does NOT match detected_domain (and not neutral), apply penalty (global fix; no keyword hardcoding).
try:
    _sg_friction = float(os.getenv("SOFT_GATE_DOMAIN_FRICTION", "0.15"))
except Exception:
    _sg_friction = 0.15
SOFT_GATE_DOMAIN_FRICTION = max(0.0, min(0.5, _sg_friction))
# Context-oriented retrieval: embed Q1=mention + Q2=context (snippet + domain + anchor names), union candidates, same kind/domain gating
GLOBAL_TWO_STAGE_CONTEXT_QUERY = os.getenv("GLOBAL_TWO_STAGE_CONTEXT_QUERY", "true").lower() in ("1", "true", "yes")
GLOBAL_TWO_STAGE_CONTEXT_SNIPPET_TOKENS = int(os.getenv("GLOBAL_TWO_STAGE_CONTEXT_SNIPPET_TOKENS", "20"))
# Debug: log per-candidate tri, pho, openai, domain, generic_penalty, relation_boost, final_score
GLOBAL_DEBUG_RERANK = os.getenv("GLOBAL_DEBUG_RERANK", "false").lower() in ("1", "true", "yes")


def _normalize_mention_for_phonetic(mention: str, max_len: int = 200) -> str:
    """Normalize mention for phonetic/trigram SQL: strip, lower, ASCII-safe. Avoids encoding/param issues."""
    if not mention:
        return ""
    s = mention.strip().lower()[:max_len]
    try:
        return s.encode("ascii", "replace").decode("ascii")
    except Exception:
        return s


def _kb_has_domain_keys(conn) -> bool:
    """Return True if kb.concepts has domain_keys column (text[] for multi-domain)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = 'kb' AND table_name = 'concepts' AND column_name = 'domain_keys'"
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def _kb_has_concept_card_embedding(conn) -> bool:
    """Return True if kb.concepts has concept_card_embedding (name + definition + kind + domain)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = 'kb' AND table_name = 'concepts' AND column_name = 'concept_card_embedding'"
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def _extract_context_snippet(mention: str, raw_transcript: Optional[str], window_tokens: int = 20) -> str:
    """Return ±window_tokens around first occurrence of mention in raw_transcript, or empty."""
    if not raw_transcript or not mention or not mention.strip():
        return ""
    mention_clean = mention.strip()
    text = raw_transcript
    try:
        idx = text.lower().find(mention_clean.lower())
    except Exception:
        return ""
    if idx < 0:
        return ""
    tokens = text.split()
    # Approximate char offset to token index
    before = text[:idx]
    start_token = max(0, len(before.split()) - window_tokens)
    end_token = min(len(tokens), len(before.split()) + len(mention_clean.split()) + window_tokens)
    snippet = " ".join(tokens[start_token:end_token])
    return snippet[:500] if snippet else ""


def _get_anchor_preferred_names(conn, anchor_concept_ids: Optional[List[int]], logger: Optional[logging.Logger] = None) -> List[str]:
    """Return preferred_name for each anchor concept_id (for Q2 context string)."""
    if not anchor_concept_ids:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT preferred_name FROM kb.concepts WHERE concept_id = ANY(%s) AND preferred_name IS NOT NULL",
                (list(anchor_concept_ids)[:20],),
            )
            rows = cur.fetchall()
        return [r[0].strip() for r in rows if r and r[0] and r[0].strip()]
    except Exception as e:
        if logger:
            logger.debug("Anchor names fetch failed: %s", e)
        return []


def _build_context_query(
    mention: str,
    raw_transcript: Optional[str],
    detected_domain: Optional[str],
    anchor_concept_ids: Optional[List[int]],
    conn,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Build Q2: mention + domain + anchor preferred names + ±N tokens around mention. Same kind/domain gating applies."""
    parts = [mention.strip()]
    if detected_domain and detected_domain.strip().lower() not in ("", "general"):
        parts.append(detected_domain.strip().lower())
    if anchor_concept_ids and conn:
        names = _get_anchor_preferred_names(conn, anchor_concept_ids, logger=logger)
        for n in names[:10]:
            if n and n not in parts:
                parts.append(n)
    snippet = _extract_context_snippet(
        mention, raw_transcript, window_tokens=GLOBAL_TWO_STAGE_CONTEXT_SNIPPET_TOKENS
    )
    if snippet and snippet.strip():
        parts.append(snippet.strip()[:300])
    return " ".join(p for p in parts if p)


def _search_global_two_stage(
    conn,
    text: str,
    kind_filter: List[str],
    topk: int,
    client: Any,
    detected_domain: Optional[str],
    logger: Optional[logging.Logger],
    embedding_cache: Optional[dict],
    ner_kind: Optional[str] = None,
    neighbor_cids: Optional[set] = None,
    raw_transcript: Optional[str] = None,
    anchor_concept_ids: Optional[List[int]] = None,
    embedding_q1: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Two-stage grounding: OpenAI fetch (wide net) + multi-signal rerank (tri/pho/openai/domain/relation).
    1. Fetch: Q1=mention, optionally Q2=context (snippet+domain+anchors); same kind/domain gating; union+dedupe.
    2. Rerank: multi-signal (trigram + phonetic + OpenAI + domain + relation).
    When embedding_q1 is provided, skip the per-call embed and use it (batched grounding).
    """
    if not text or not text.strip():
        return []
    mention = text.strip()
    fetch_k = max(topk, min(GLOBAL_TWO_STAGE_FETCH_K, 100))

    # Stage 1: OpenAI embedding Q1 (mention only) — use precomputed when provided (batch path)
    query_q1 = mention
    if embedding_q1 is not None and len(embedding_q1) > 0:
        emb_q1 = embedding_q1
    elif embedding_cache is not None and query_q1 in embedding_cache:
        emb_q1 = embedding_cache.get(query_q1)
    else:
        emb_q1 = embed_text(query_q1, client=client, logger=logger)
        if embedding_cache is not None and emb_q1:
            embedding_cache[query_q1] = emb_q1
    if not emb_q1:
        if logger:
            logger.warning("Two-stage: OpenAI embedding failed for '%s'", mention[:50])
        return []
    try:
        vec_literal = to_pgvector_literal(emb_q1)
    except Exception as e:
        if logger:
            logger.warning("Two-stage: vec_literal failed for '%s': %s", mention[:50], e)
        return []
    if not vec_literal or not isinstance(vec_literal, str):
        return []

    # Optional Q2 context query: mention + domain + anchor names + snippet (±N tokens)
    use_context_query = (
        GLOBAL_TWO_STAGE_CONTEXT_QUERY
        and (raw_transcript or (detected_domain and (detected_domain or "").strip().lower() not in ("", "general")) or anchor_concept_ids)
    )
    vec_literal_q2 = None
    use_card_embedding = False
    if use_context_query and client:
        q2_text = _build_context_query(mention, raw_transcript, detected_domain, anchor_concept_ids, conn, logger=logger)
        if q2_text and q2_text.strip() != mention:
            cache_key = "q2:" + q2_text[:200]
            if embedding_cache is not None and cache_key in embedding_cache:
                emb_q2 = embedding_cache.get(cache_key)
            else:
                emb_q2 = embed_text(q2_text, client=client, logger=logger)
                if embedding_cache is not None and emb_q2:
                    embedding_cache[cache_key] = emb_q2
            if emb_q2:
                try:
                    vec_literal_q2 = to_pgvector_literal(emb_q2)
                except Exception:
                    vec_literal_q2 = None
            if vec_literal_q2 and _kb_has_concept_card_embedding(conn):
                use_card_embedding = True
            if logger and vec_literal_q2:
                logger.debug("Two-stage: Q2 context query len=%s (card_emb=%s)", len(q2_text), use_card_embedding)

    # Stage 1: Fetch candidates. When domain is set and ensemble is on, use two buckets (semantic + phonetic).
    kind_filter_list = list(kind_filter) if kind_filter else [
        "Drug", "Procedure", "Finding", "Condition", "Anatomy", "Observation"
    ]
    domain_norm = (detected_domain or "").strip().lower()
    use_domain_filter = bool(domain_norm and domain_norm not in ("", "general"))
    has_domain_keys = use_domain_filter and _kb_has_domain_keys(conn)
    use_ensemble = use_domain_filter and GLOBAL_TWO_STAGE_ENSEMBLE_FETCH
    if use_domain_filter and logger:
        logger.debug(
            "Two-stage: domain-constrained fetch (domain_key=%s%s)",
            domain_norm, ", ensemble=semantic+phonetic" if use_ensemble else "",
        )

    if use_domain_filter:
        if has_domain_keys:
            domain_clause = " AND ((c.domain_keys IS NOT NULL AND %s = ANY(c.domain_keys)) OR LOWER(TRIM(COALESCE(c.domain_key, ''))) = %s) "
        else:
            domain_clause = " AND LOWER(TRIM(COALESCE(c.domain_key, ''))) = %s "
    else:
        domain_clause = ""

    rows: List[Tuple[Any, ...]] = []
    if use_ensemble:
        # Ensemble fetch: Bucket A (semantic, top N by OpenAI) + Bucket B (phonetic, top M by trigram similarity).
        # Merged pool for rerank; no hardcoded golden pairs.
        semantic_n = min(GLOBAL_TWO_STAGE_SEMANTIC_BUCKET_SIZE, 50)
        phonetic_n = min(GLOBAL_TWO_STAGE_PHONETIC_BUCKET_SIZE, 50)
        # Conditional floor: specialty vs general. If escalation enabled and specialty, use GENERAL first; escalate to 0.1 only if phonetic returns 0 rows.
        is_specialty = use_domain_filter and domain_norm and domain_norm not in GLOBAL_TWO_STAGE_DOMAIN_NEUTRAL
        if is_specialty and GLOBAL_TWO_STAGE_PHONETIC_ESCALATION:
            sim_thresh = max(0.1, min(0.9, GLOBAL_TWO_STAGE_PHONETIC_SIM_THRESHOLD_GENERAL))  # Pass 1: clean threshold
        elif is_specialty:
            sim_thresh = max(0.1, min(0.9, GLOBAL_TWO_STAGE_PHONETIC_SIM_THRESHOLD))
        else:
            sim_thresh = max(0.1, min(0.9, GLOBAL_TWO_STAGE_PHONETIC_SIM_THRESHOLD_GENERAL))
        if logger:
            logger.debug(
                "Two-stage: ensemble kind_filter=%s sim_thresh=%.3f mention_repr=%s",
                kind_filter_list, sim_thresh, repr(mention),
            )

        # Bucket A: semantic (same as single-stream, limit semantic_n)
        sql_semantic = f"""
        WITH all_candidates AS (
            SELECT c.concept_id, c.preferred_name, c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                (c.embedding <=> %s::vector) AS openai_dist
            FROM kb.concepts c
            WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED'){domain_clause}
            UNION ALL
            SELECT ca.concept_id, ca.alias_text AS preferred_name, c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                (ca.embedding <=> %s::vector) AS openai_dist
            FROM kb.concept_aliases ca
            INNER JOIN kb.concepts c ON c.concept_id = ca.concept_id
            WHERE ca.embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED'){domain_clause}
            UNION ALL
            SELECT la.kb_concept_id AS concept_id, la.alias_text AS preferred_name, c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                (la.embedding <=> %s::vector) AS openai_dist
            FROM kb.learned_aliases la
            INNER JOIN kb.concepts c ON c.concept_id = la.kb_concept_id
            WHERE la.embedding IS NOT NULL AND (la.is_validated = TRUE OR la.frequency_count > 0)
              AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED'){domain_clause}
        )
        SELECT concept_id, preferred_name, kind, definition, venom_code, snomed_code,
               embedding, domain_key, openai_dist
        FROM all_candidates
        ORDER BY openai_dist
        LIMIT %s
        """
        if use_domain_filter:
            if has_domain_keys:
                params_sem = (
                    vec_literal, kind_filter_list, domain_norm, domain_norm,
                    vec_literal, kind_filter_list, domain_norm, domain_norm,
                    vec_literal, kind_filter_list, domain_norm, domain_norm,
                    semantic_n,
                )
            else:
                params_sem = (
                    vec_literal, kind_filter_list, domain_norm,
                    vec_literal, kind_filter_list, domain_norm,
                    vec_literal, kind_filter_list, domain_norm,
                    semantic_n,
                )
        else:
            params_sem = (vec_literal, kind_filter_list, vec_literal, kind_filter_list, vec_literal, kind_filter_list, semantic_n)
        try:
            ensure_pg_trgm(conn, logger=logger)
            with conn.cursor() as cur:
                cur.execute(sql_semantic, params_sem)
                rows_semantic = cur.fetchall()
        except Exception as e:
            if logger:
                logger.warning("Two-stage: semantic bucket failed for '%s': %s", mention[:50], e)
            rows_semantic = []

        # Q2 context fetch: same kind/domain gating; union + dedupe (keep min openai_dist per concept)
        rows_semantic_q2: List[Tuple[Any, ...]] = []
        if vec_literal_q2:
            if use_card_embedding:
                sql_q2_card = f"""
                SELECT c.concept_id, c.preferred_name, c.kind,
                    COALESCE(c.definition, '') AS definition,
                    COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                    c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                    (c.concept_card_embedding <=> %s::vector) AS openai_dist
                FROM kb.concepts c
                WHERE c.concept_card_embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED'){domain_clause}
                ORDER BY openai_dist
                LIMIT %s
                """
                if use_domain_filter:
                    if has_domain_keys:
                        params_q2 = (vec_literal_q2, kind_filter_list, domain_norm, domain_norm, semantic_n)
                    else:
                        params_q2 = (vec_literal_q2, kind_filter_list, domain_norm, semantic_n)
                else:
                    params_q2 = (vec_literal_q2, kind_filter_list, semantic_n)
                try:
                    with conn.cursor() as cur2:
                        cur2.execute(sql_q2_card, params_q2)
                        rows_semantic_q2 = list(cur2.fetchall())
                except Exception as e2:
                    if logger:
                        logger.debug("Two-stage: Q2 card fetch failed: %s", e2)
            else:
                params_sem_q2 = (
                    (vec_literal_q2, kind_filter_list, domain_norm, domain_norm, vec_literal_q2, kind_filter_list, domain_norm, domain_norm, vec_literal_q2, kind_filter_list, domain_norm, domain_norm, semantic_n)
                    if use_domain_filter and has_domain_keys
                    else (vec_literal_q2, kind_filter_list, domain_norm, vec_literal_q2, kind_filter_list, domain_norm, vec_literal_q2, kind_filter_list, domain_norm, semantic_n)
                    if use_domain_filter
                    else (vec_literal_q2, kind_filter_list, vec_literal_q2, kind_filter_list, vec_literal_q2, kind_filter_list, semantic_n)
                )
                try:
                    with conn.cursor() as cur2:
                        cur2.execute(sql_semantic, params_sem_q2)
                        rows_semantic_q2 = list(cur2.fetchall())
                except Exception as e2:
                    if logger:
                        logger.debug("Two-stage: Q2 semantic fetch failed: %s", e2)
            # Union + dedupe by (concept_id, preferred_name), keep row with min(openai_dist)
            if rows_semantic_q2:
                by_key: Dict[Tuple[Any, Any], Tuple[Any, ...]] = {}
                for r in rows_semantic:
                    if len(r) >= 9:
                        by_key[(r[0], r[1])] = r
                for r in rows_semantic_q2:
                    if len(r) < 9:
                        continue
                    key = (r[0], r[1])
                    existing = by_key.get(key)
                    dist_new = float(r[8]) if r[8] is not None else 1.0
                    if existing is None or dist_new < float(existing[8] if existing[8] is not None else 1.0):
                        by_key[key] = r
                rows_semantic = list(by_key.values())
                if logger:
                    logger.debug("Two-stage: Q1+Q2 union semantic pool=%s", len(rows_semantic))

        # Bucket B: phonetic/trigram (concepts + aliases, similarity >= threshold, limit phonetic_n)
        # Use real OpenAI distance so phonetic rows don't get oa=1.0 and saturate the score (was 0.0 AS openai_dist).
        mention_ph = _normalize_mention_for_phonetic(mention)
        rows_phonetic = []
        sql_phonetic = f"""
        WITH phonetic_candidates AS (
            SELECT c.concept_id, c.preferred_name, c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                similarity(lower(c.preferred_name), lower(%s)) AS sim,
                COALESCE((c.embedding <=> %s::vector), 1.0) AS openai_dist
            FROM kb.concepts c
            WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s){domain_clause}
              AND (c.status IS NULL OR c.status != 'REJECTED')
              AND similarity(lower(c.preferred_name), lower(%s)) >= {sim_thresh}
            UNION ALL
            SELECT ca.concept_id, ca.alias_text AS preferred_name, c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                similarity(lower(ca.alias_text), lower(%s)) AS sim,
                COALESCE((ca.embedding <=> %s::vector), 1.0) AS openai_dist
            FROM kb.concept_aliases ca
            INNER JOIN kb.concepts c ON c.concept_id = ca.concept_id
            WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s){domain_clause}
              AND (c.status IS NULL OR c.status != 'REJECTED')
              AND similarity(lower(ca.alias_text), lower(%s)) >= {sim_thresh}
        )
        SELECT concept_id, preferred_name, kind, definition, venom_code, snomed_code,
               embedding, domain_key, openai_dist
        FROM (
            SELECT * FROM phonetic_candidates ORDER BY sim DESC LIMIT %s
        ) sub
        """
        # Params: concepts (mention_ph, vec_literal, kind, [domain], mention_ph) + aliases (mention_ph, vec_literal, kind, [domain], mention_ph) + limit
        if use_domain_filter:
            if has_domain_keys:
                params_ph = (mention_ph, vec_literal, kind_filter_list, domain_norm, domain_norm, mention_ph, mention_ph, vec_literal, kind_filter_list, domain_norm, domain_norm, mention_ph, phonetic_n)
            else:
                params_ph = (mention_ph, vec_literal, kind_filter_list, domain_norm, mention_ph, mention_ph, vec_literal, kind_filter_list, domain_norm, mention_ph, phonetic_n)
        else:
            params_ph = (mention_ph, vec_literal, kind_filter_list, mention_ph, mention_ph, vec_literal, kind_filter_list, mention_ph, phonetic_n)
        try:
            with conn.cursor() as cur:
                cur.execute(sql_phonetic, params_ph)
                rows_phonetic = cur.fetchall()
        except Exception as e:
            if logger:
                logger.debug("Two-stage: phonetic bucket failed for '%s' (pg_trgm?): %s", mention[:50], e)
            rows_phonetic = []

        # Conditional multi-pass: if specialty and escalation and Pass 1 (0.2) returned 0 phonetic rows, run Pass 2 at 0.1 (sniper within specialty).
        if is_specialty and GLOBAL_TWO_STAGE_PHONETIC_ESCALATION and len(rows_phonetic) == 0:
            sim_thresh_esc = max(0.1, min(0.9, GLOBAL_TWO_STAGE_PHONETIC_SIM_THRESHOLD))
            sql_phonetic_esc = f"""
        WITH phonetic_candidates AS (
            SELECT c.concept_id, c.preferred_name, c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                similarity(lower(c.preferred_name), lower(%s)) AS sim,
                COALESCE((c.embedding <=> %s::vector), 1.0) AS openai_dist
            FROM kb.concepts c
            WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s){domain_clause}
              AND (c.status IS NULL OR c.status != 'REJECTED')
              AND similarity(lower(c.preferred_name), lower(%s)) >= {sim_thresh_esc}
            UNION ALL
            SELECT ca.concept_id, ca.alias_text AS preferred_name, c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                similarity(lower(ca.alias_text), lower(%s)) AS sim,
                COALESCE((ca.embedding <=> %s::vector), 1.0) AS openai_dist
            FROM kb.concept_aliases ca
            INNER JOIN kb.concepts c ON c.concept_id = ca.concept_id
            WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s){domain_clause}
              AND (c.status IS NULL OR c.status != 'REJECTED')
              AND similarity(lower(ca.alias_text), lower(%s)) >= {sim_thresh_esc}
        )
        SELECT concept_id, preferred_name, kind, definition, venom_code, snomed_code,
               embedding, domain_key, openai_dist
        FROM (
            SELECT * FROM phonetic_candidates ORDER BY sim DESC LIMIT %s
        ) sub
            """
            if use_domain_filter:
                if has_domain_keys:
                    params_ph_esc = (mention_ph, vec_literal, kind_filter_list, domain_norm, domain_norm, mention_ph, mention_ph, vec_literal, kind_filter_list, domain_norm, domain_norm, mention_ph, phonetic_n)
                else:
                    params_ph_esc = (mention_ph, vec_literal, kind_filter_list, domain_norm, mention_ph, mention_ph, vec_literal, kind_filter_list, domain_norm, mention_ph, phonetic_n)
            else:
                params_ph_esc = (mention_ph, vec_literal, kind_filter_list, mention_ph, mention_ph, vec_literal, kind_filter_list, mention_ph, phonetic_n)
            try:
                with conn.cursor() as cur:
                    cur.execute(sql_phonetic_esc, params_ph_esc)
                    rows_phonetic = cur.fetchall()
                if logger and rows_phonetic:
                    logger.debug("Two-stage: phonetic escalation (0.2->0.1) returned %s rows for '%s'", len(rows_phonetic), mention[:50])
            except Exception as e_esc:
                if logger:
                    logger.debug("Two-stage: phonetic escalation failed for '%s': %s", mention[:50], e_esc)

        # Metaphone fallback: when trigram phonetic still 0 and specialty, use sound-skeleton (fuzzystrmatch).
        # Handles ASR variants (e.g. "ultralining" → ARTLN vs "Ortolani" → ARTLN) without hardcoded pairs.
        has_fuzzystrmatch = False
        if is_specialty and len(rows_phonetic) == 0:
            try:
                ensure_fuzzystrmatch(conn, logger=logger)
                with conn.cursor() as cur:
                    cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'fuzzystrmatch');")
                    row = cur.fetchone()
                    has_fuzzystrmatch = bool(row and row[0])
            except Exception:
                pass
            if has_fuzzystrmatch:
                sql_metaphone = f"""
                WITH q AS (
                    SELECT metaphone(lower(%s), 10) AS q_mfull,
                           metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst
                ),
                metaphone_candidates AS (
                    SELECT c.concept_id, c.preferred_name, c.kind,
                        COALESCE(c.definition, '') AS definition,
                        COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                        c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                        metaphone(lower(c.preferred_name), 10) AS m_full,
                        metaphone(split_part(lower(c.preferred_name), ' ', 1), 10) AS m_first,
                        COALESCE((c.embedding <=> %s::vector), 1.0) AS openai_dist
                    FROM kb.concepts c
                    CROSS JOIN q
                    WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s){domain_clause}
                      AND (c.status IS NULL OR c.status != 'REJECTED')
                      AND (metaphone(lower(c.preferred_name), 10) = q.q_mfull
                           OR metaphone(split_part(lower(c.preferred_name), ' ', 1), 10) = q.q_mfirst
                           OR (levenshtein(metaphone(lower(c.preferred_name), 10), q.q_mfull)::float
                               / GREATEST(length(metaphone(lower(c.preferred_name), 10)), length(q.q_mfull), 1) < 0.65)
                           OR (levenshtein(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10), q.q_mfirst)::float
                               / GREATEST(length(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10)), length(q.q_mfirst), 1) < 0.65))
                    UNION ALL
                    SELECT ca.concept_id, ca.alias_text AS preferred_name, c.kind,
                        COALESCE(c.definition, '') AS definition,
                        COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                        c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                        metaphone(lower(ca.alias_text), 10) AS m_full,
                        metaphone(split_part(lower(ca.alias_text), ' ', 1), 10) AS m_first,
                        COALESCE((ca.embedding <=> %s::vector), 1.0) AS openai_dist
                    FROM kb.concept_aliases ca
                    INNER JOIN kb.concepts c ON c.concept_id = ca.concept_id
                    CROSS JOIN q
                    WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s){domain_clause}
                      AND (c.status IS NULL OR c.status != 'REJECTED')
                      AND (metaphone(lower(ca.alias_text), 10) = q.q_mfull
                           OR metaphone(split_part(lower(ca.alias_text), ' ', 1), 10) = q.q_mfirst
                           OR (levenshtein(metaphone(lower(ca.alias_text), 10), q.q_mfull)::float
                               / GREATEST(length(metaphone(lower(ca.alias_text), 10)), length(q.q_mfull), 1) < 0.65)
                           OR (levenshtein(metaphone(split_part(lower(ca.alias_text), ' ', 1), 10), q.q_mfirst)::float
                               / GREATEST(length(metaphone(split_part(lower(ca.alias_text), ' ', 1), 10)), length(q.q_mfirst), 1) < 0.65))
                )
                SELECT concept_id, preferred_name, kind, definition, venom_code, snomed_code,
                       embedding, domain_key, openai_dist
                FROM (
                    SELECT DISTINCT ON (concept_id, preferred_name)
                        concept_id, preferred_name, kind, definition, venom_code, snomed_code, embedding, domain_key, openai_dist
                    FROM metaphone_candidates
                    ORDER BY concept_id, preferred_name
                    LIMIT %s
                ) sub
                """
                if use_domain_filter:
                    if has_domain_keys:
                        params_m = (mention_ph, mention_ph, vec_literal, kind_filter_list, domain_norm, domain_norm, vec_literal, kind_filter_list, domain_norm, domain_norm, phonetic_n)
                    else:
                        params_m = (mention_ph, mention_ph, vec_literal, kind_filter_list, domain_norm, vec_literal, kind_filter_list, domain_norm, phonetic_n)
                else:
                    params_m = (mention_ph, mention_ph, vec_literal, kind_filter_list, vec_literal, kind_filter_list, phonetic_n)
                try:
                    with conn.cursor() as cur:
                        cur.execute(sql_metaphone, params_m)
                        rows_phonetic = cur.fetchall()
                    if logger and rows_phonetic:
                        logger.debug("Two-stage: metaphone fallback returned %s rows for '%s'", len(rows_phonetic), mention[:50])
                except Exception as e_m:
                    if logger:
                        logger.debug("Two-stage: metaphone fallback failed for '%s': %s", mention[:50], e_m)

        # Merge: semantic first, then phonetic; dedupe by (concept_id, preferred_name)
        seen = set()
        for r in rows_semantic:
            key = (r[0], r[1])
            if key not in seen:
                rows.append(r)
                seen.add(key)
        for r in rows_phonetic:
            key = (r[0], r[1])
            if key not in seen:
                rows.append(r)
                seen.add(key)
        if logger and (rows_semantic or rows_phonetic):
            logger.debug(
                "Two-stage: ensemble fetch semantic=%s phonetic=%s merged=%s for '%s'",
                len(rows_semantic), len(rows_phonetic), len(rows), mention[:50],
            )
    else:
        # Single-stream: top fetch_k by OpenAI distance only
        sql_fetch = f"""
        WITH all_candidates AS (
            SELECT c.concept_id, c.preferred_name, c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                (c.embedding <=> %s::vector) AS openai_dist
            FROM kb.concepts c
            WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED'){domain_clause}
            UNION ALL
            SELECT ca.concept_id, ca.alias_text AS preferred_name, c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                (ca.embedding <=> %s::vector) AS openai_dist
            FROM kb.concept_aliases ca
            INNER JOIN kb.concepts c ON c.concept_id = ca.concept_id
            WHERE ca.embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED'){domain_clause}
            UNION ALL
            SELECT la.kb_concept_id AS concept_id, la.alias_text AS preferred_name, c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                (la.embedding <=> %s::vector) AS openai_dist
            FROM kb.learned_aliases la
            INNER JOIN kb.concepts c ON c.concept_id = la.kb_concept_id
            WHERE la.embedding IS NOT NULL AND (la.is_validated = TRUE OR la.frequency_count > 0)
              AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED'){domain_clause}
        )
        SELECT concept_id, preferred_name, kind, definition, venom_code, snomed_code,
               embedding, domain_key, openai_dist
        FROM all_candidates
        ORDER BY openai_dist
        LIMIT %s
        """
        if use_domain_filter:
            if has_domain_keys:
                params_fetch = (
                    vec_literal, kind_filter_list, domain_norm, domain_norm,
                    vec_literal, kind_filter_list, domain_norm, domain_norm,
                    vec_literal, kind_filter_list, domain_norm, domain_norm,
                    fetch_k,
                )
            else:
                params_fetch = (
                    vec_literal, kind_filter_list, domain_norm,
                    vec_literal, kind_filter_list, domain_norm,
                    vec_literal, kind_filter_list, domain_norm,
                    fetch_k,
                )
        else:
            params_fetch = (vec_literal, kind_filter_list, vec_literal, kind_filter_list, vec_literal, kind_filter_list, fetch_k)
        try:
            with conn.cursor() as cur:
                cur.execute(sql_fetch, params_fetch)
                rows = list(cur.fetchall())
            # Q2 context fetch (single-stream): same kind/domain; union + dedupe
            if vec_literal_q2:
                if use_card_embedding:
                    sql_q2_card = f"""
                    SELECT c.concept_id, c.preferred_name, c.kind,
                        COALESCE(c.definition, '') AS definition,
                        COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                        c.embedding, COALESCE(c.domain_key, '') AS domain_key,
                        (c.concept_card_embedding <=> %s::vector) AS openai_dist
                    FROM kb.concepts c
                    WHERE c.concept_card_embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED'){domain_clause}
                    ORDER BY openai_dist
                    LIMIT %s
                    """
                    if use_domain_filter:
                        params_q2 = (vec_literal_q2, kind_filter_list, domain_norm, domain_norm, fetch_k) if has_domain_keys else (vec_literal_q2, kind_filter_list, domain_norm, fetch_k)
                    else:
                        params_q2 = (vec_literal_q2, kind_filter_list, fetch_k)
                    with conn.cursor() as c2:
                        c2.execute(sql_q2_card, params_q2)
                        rows_q2 = list(c2.fetchall())
                else:
                    params_fetch_q2 = (
                        (vec_literal_q2, kind_filter_list, domain_norm, domain_norm, vec_literal_q2, kind_filter_list, domain_norm, domain_norm, vec_literal_q2, kind_filter_list, domain_norm, domain_norm, fetch_k)
                        if use_domain_filter and has_domain_keys
                        else (vec_literal_q2, kind_filter_list, domain_norm, vec_literal_q2, kind_filter_list, domain_norm, vec_literal_q2, kind_filter_list, domain_norm, fetch_k)
                        if use_domain_filter
                        else (vec_literal_q2, kind_filter_list, vec_literal_q2, kind_filter_list, vec_literal_q2, kind_filter_list, fetch_k)
                    )
                    with conn.cursor() as c2:
                        c2.execute(sql_fetch, params_fetch_q2)
                        rows_q2 = list(c2.fetchall())
                by_key_s: Dict[Tuple[Any, Any], Tuple[Any, ...]] = {}
                for r in rows:
                    if len(r) >= 9:
                        by_key_s[(r[0], r[1])] = r
                for r in rows_q2:
                    if len(r) < 9:
                        continue
                    key = (r[0], r[1])
                    existing = by_key_s.get(key)
                    dist_new = float(r[8]) if r[8] is not None else 1.0
                    if existing is None or dist_new < float(existing[8] if existing[8] is not None else 1.0):
                        by_key_s[key] = r
                rows = list(by_key_s.values())
                if logger:
                    logger.debug("Two-stage: single-stream Q1+Q2 union pool=%s", len(rows))
        except Exception as e:
            if logger:
                logger.warning("Two-stage: fetch query failed for '%s': %s", mention[:50], e)
            return []

    if not rows:
        if logger and use_domain_filter:
            logger.debug(
                "Two-stage: fetch returned 0 rows for '%s' (domain=%s)",
                mention[:50], domain_norm,
            )
        return []

    # Debug: openai_dist distribution (once per query) to catch saturation / wrong transform
    if logger and rows:
        dists = []
        n_null = 0
        n_zero = 0
        for r in rows:
            if len(r) < 9:
                continue
            d = r[8]
            if d is None:
                n_null += 1
            else:
                try:
                    fd = float(d)
                    dists.append(fd)
                    if fd == 0:
                        n_zero += 1
                except (TypeError, ValueError):
                    n_null += 1
        if dists:
            min_d, max_d = min(dists), max(dists)
            avg_d = sum(dists) / len(dists)
            logger.info(
                "Two-stage: openai_dist min=%.4f max=%.4f avg=%.4f | null=%s zero=%s n=%s",
                min_d, max_d, avg_d, n_null, n_zero, len(rows),
            )
            # Sanity: if >30%% of candidates have oa >= 0.99, log warning (saturation)
            oa_scores = [max(0.0, 1.0 - float(x)) for x in dists]
            n_saturated = sum(1 for s in oa_scores if s >= 0.99)
            if len(oa_scores) and (n_saturated / len(oa_scores)) > 0.30:
                logger.warning(
                    "OpenAI score saturated: %.0f%%%% of candidates have oa>=0.99; check dist transform or phonetic bucket openai_dist.",
                    100.0 * n_saturated / len(oa_scores),
                )
        elif n_null or n_zero:
            logger.info("Two-stage: openai_dist all null/zero or missing | null=%s zero=%s n=%s", n_null, n_zero, len(rows))

        # Raw openai_dist for top 5 + Ortolani (id=45424) so we can pick the right transform
        logger.info(
            "Two-stage: pgvector operator=<=> (cosine distance); vectors=check schema (OpenAI text-embedding-3-small often unit-normalized, cosine dist in [0,2])"
        )
        ORTOLANI_CONCEPT_ID = 45424
        seen_top5 = set()
        for i, r in enumerate(rows[:5]):
            if len(r) < 9:
                continue
            cid, pname = r[0], (r[1] or "")[:40]
            raw_d = r[8]
            dist_val = float(raw_d) if raw_d is not None else None
            seen_top5.add(cid)
            logger.info("  openai_dist top5[%s] id=%s name=%s raw_dist=%s", i + 1, cid, pname, dist_val)
        ortolani_row = None
        for r in rows:
            if len(r) >= 9 and r[0] == ORTOLANI_CONCEPT_ID:
                ortolani_row = r
                break
        if ortolani_row is not None:
            raw_d = ortolani_row[8]
            dist_val = float(raw_d) if raw_d is not None else None
            if ortolani_row[0] not in seen_top5:
                logger.info("  openai_dist Ortolani id=%s name=%s raw_dist=%s (not in top 5)", ortolani_row[0], (ortolani_row[1] or "")[:40], dist_val)
            else:
                logger.info("  openai_dist Ortolani id=%s name=%s raw_dist=%s (in top 5)", ortolani_row[0], (ortolani_row[1] or "")[:40], dist_val)

    # Stage 2: Rerank with OpenAI + optional multi-signal (tri/pho)
    detected_domain_norm = (detected_domain or "").strip().lower()

    # Multi-signal: batch compute tri and pho separately (weighted sum, not max) so phonetic can lift Ortolani
    tri_map: Dict[str, float] = {}
    pho_map: Dict[str, float] = {}
    if GLOBAL_TWO_STAGE_MULTI_SIGNAL_RANKER and rows:
        preferred_names = [r[1] for r in rows]
        try:
            ensure_pg_trgm(conn, logger=logger)
            ensure_fuzzystrmatch(conn, logger=logger)
            with conn.cursor() as cur:
                if preferred_names:
                    placeholders = ", ".join(["(%s)"] * len(preferred_names))
                    cur.execute(
                        """
                        WITH q AS (
                            SELECT lower(%s) AS q_full,
                                metaphone(lower(%s), 10) AS q_mfull,
                                metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst
                        ),
                        names(pname) AS (VALUES """
                        + placeholders
                        + """ )
                        SELECT v.pname,
                            similarity(lower(v.pname), q.q_full) AS tri,
                            GREATEST(
                                0.8 * (1.0 - levenshtein(metaphone(split_part(lower(v.pname), ' ', 1), 10), q.q_mfirst)::float
                                    / GREATEST(length(metaphone(split_part(lower(v.pname), ' ', 1), 10)), length(q.q_mfirst), 1)),
                                0.8 * (1.0 - levenshtein(metaphone(lower(v.pname), 10), q.q_mfull)::float
                                    / GREATEST(length(metaphone(lower(v.pname), 10)), length(q.q_mfull), 1))
                            ) AS pho
                        FROM names v
                        CROSS JOIN q
                        """,
                        (mention, mention, mention) + tuple(preferred_names),
                    )
                    for (pname, tri, pho) in cur.fetchall():
                        if pname is not None:
                            tri_map[pname] = float(tri) if tri is not None else 0.0
                            pho_map[pname] = float(pho) if pho is not None else 0.0
        except Exception as e:
            if logger:
                logger.debug("Two-stage: batch lexical (trigram+phonetic) for multi-signal failed: %s", e)
            try:
                with conn.cursor() as cur:
                    if preferred_names:
                        placeholders = ", ".join(["(%s)"] * len(preferred_names))
                        cur.execute(
                            "SELECT v.pname, similarity(lower(v.pname), lower(%s)) FROM (VALUES " + placeholders + ") AS v(pname)",
                            (mention,) + tuple(preferred_names),
                        )
                        for (pname, sim) in cur.fetchall():
                            if pname is not None and sim is not None:
                                tri_map[pname] = float(sim)
                                pho_map[pname] = 0.0
            except Exception:
                pass

    use_multi_signal = bool(GLOBAL_TWO_STAGE_MULTI_SIGNAL_RANKER and (tri_map or pho_map))
    reranked = []
    for row in rows:
        (concept_id, preferred_name, kind, definition, venom_code, snomed_code,
         _emb, domain_key, openai_dist) = row[:9]
        openai_score = max(0.0, 1.0 - float(openai_dist))
        aux_vector_sim = 0.0
        cand_domain = (domain_key or "").strip().lower() or ""
        # If domain_key is null/empty, try to infer domain from concept name using keyword matching
        # This helps when KB concepts haven't been backfilled with domain_key yet (e.g., "Ortolani" should be orthopedic)
        if not cand_domain and preferred_name:
            try:
                from kb_ner_domain import DOMAIN_KEYWORDS
                pname_lower = preferred_name.lower()
                for domain, keywords in DOMAIN_KEYWORDS.items():
                    if any(kw in pname_lower for kw in keywords):
                        cand_domain = domain
                        break
            except Exception:
                pass
        if cand_domain == detected_domain_norm:
            domain_score = GLOBAL_TWO_STAGE_DOMAIN_WEIGHT
        elif cand_domain and cand_domain not in GLOBAL_TWO_STAGE_DOMAIN_NEUTRAL and detected_domain_norm:
            domain_score = -GLOBAL_TWO_STAGE_DOMAIN_FRICTION
        else:
            domain_score = 0.0
        tri = float(tri_map.get(preferred_name, 0.0))
        pho = float(pho_map.get(preferred_name, 0.0))
        cand_kind = (kind or "").strip()
        cand_tokens = set((preferred_name or "").lower().split())
        is_generic_proc = (
            cand_kind in GENERIC_PENALTY_KINDS
            and len(cand_tokens & GENERIC_PROC_TOKENS) > 0
        )
        generic_penalty = -GLOBAL_GENERIC_PROC_PENALTY if is_generic_proc else 0.0
        is_neighbor = (concept_id in neighbor_cids) if neighbor_cids else False
        relation_boost = GLOBAL_TWO_STAGE_RELATION_WEIGHT if is_neighbor else 0.0
        # Multi-signal weights: openai + tri + pho (redistributed vector weight folded into these)
        if use_multi_signal:
            vb_weight = 0.0
            oa_weight = GLOBAL_TWO_STAGE_OPENAI_WEIGHT + (REDISTRIBUTED_VECTOR_WEIGHT * 0.43)
            tri_weight = GLOBAL_TWO_STAGE_TRIGRAM_WEIGHT + (REDISTRIBUTED_VECTOR_WEIGHT * 0.14)
            pho_weight = GLOBAL_TWO_STAGE_PHONETIC_WEIGHT + (REDISTRIBUTED_VECTOR_WEIGHT * 0.43)
            final_score = (
                (tri * tri_weight)
                + (pho * pho_weight)
                + (aux_vector_sim * vb_weight)
                + (openai_score * oa_weight)
                + domain_score
                + generic_penalty
                + relation_boost
            )
        else:
            final_score = (openai_score * SOFT_GATE_BASE_WEIGHT) + domain_score + generic_penalty
        r = {
            "concept_id": concept_id,
            "preferred_name": preferred_name,
            "kind": kind,
            "definition": definition or None,
            "venom_code": venom_code or None,
            "snomed_code": snomed_code or None,
            "domain_key": (domain_key or "").strip() or None,
            "match_score": final_score,
            "final_score": final_score,
            "match_source": "ensemble_multi_signal" if use_multi_signal else "openai_fetch",
            "openai_score": openai_score,
            "trigram_score": tri,
            "phonetic_score": pho,
            "domain_score": domain_score,
            "generic_penalty": generic_penalty,
            "relation_boost": relation_boost,
            "is_neighbor": is_neighbor,
        }
        if GLOBAL_DEBUG_RERANK and logger:
            logger.debug(
                "  [rerank] %s | tri=%.3f pho=%.3f oa=%.3f dom=%.3f gpen=%.3f rel=%.3f → final=%.3f",
                (preferred_name or "")[:35], tri, pho, openai_score, domain_score, generic_penalty, relation_boost, final_score,
            )
        reranked.append(r)
    reranked.sort(key=lambda x: float(x.get("final_score", 0)), reverse=True)

    # Tie-breaker (safety valve): only when top candidates are within 0.05 and vector is not decisive
    if use_multi_signal and len(reranked) >= 2:
        top_score = float(reranked[0].get("final_score", 0))
        second_score = float(reranked[1].get("final_score", 0))
        margin = top_score - second_score
        if margin < 0.05:
            all_names = { (r.get("preferred_name") or "").strip() for r in reranked }
            # NER anchor: only give specificity bonus when candidate kind matches NER intent
            preferred_kinds_for_ner = {
                "Measurement": ["Observation"],
                "Procedure": ["Procedure"],
                "DiagnosticTest": ["Procedure", "Observation"],
            }
            preferred_kinds = preferred_kinds_for_ner.get(ner_kind or "", [])
            for r in reranked:
                adj = 0.0
                name = (r.get("preferred_name") or "").strip()
                # Subset penalty: this name is a proper substring of another in the pool
                for other in all_names:
                    if other != name and name in other:
                        adj -= 0.1
                        break
                # Specificity bonus: 2–3 word concepts whose kind matches NER (avoid rewarding long phonetic noise)
                wc = name.count(" ") + 1
                if 2 <= wc <= 3 and (r.get("kind") in preferred_kinds):
                    adj += 0.05
                # Long-name penalty: 4+ word candidates in tie-breaker zone often phonetic false positives
                if wc >= 4:
                    adj -= 0.08
                r["final_score"] = float(r.get("final_score", 0)) + adj
            reranked.sort(key=lambda x: float(x.get("final_score", 0)), reverse=True)

    # Dedupe by concept_id: keep best surface per concept so "Norberg angle" / "Norberg" don't occupy multiple slots
    seen_cid = set()
    deduped = []
    for r in reranked:
        cid = r.get("concept_id")
        if cid is not None and cid not in seen_cid:
            deduped.append(r)
            seen_cid.add(cid)
    if logger:
        top = deduped[0] if deduped else {}
        mode = "multi-signal" if use_multi_signal else "openai"
        logger.info(
            "Two-stage: fetch %s → %s rerank → top final %.3f (domain=%s)%s",
            len(rows), mode, top.get("final_score", 0), detected_domain or "",
            f" [deduped {len(reranked)}→{len(deduped)}]" if len(deduped) < len(reranked) else "",
        )
    return deduped[:topk]


def kb_lookup_concepts_by_embedding_batch(
    conn,
    mention_texts: List[str],
    mention_embeddings: List[List[float]],
    *,
    kind_filter: Optional[List[str]] = None,
    top_k_per_mention: int = 3,
    max_distance: float = 0.20,
    logger: Optional[logging.Logger] = None,
    include_aliases: bool = True,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Batch vector lookup in ONE round trip using UNNEST + LATERAL.

    This is the core building block for “millisecond” global KB linking when you
    already have embeddings for all mentions.

    Returns:
        Dict: mention_index -> list of candidate dicts (best-first)
    """
    if not mention_texts or not mention_embeddings:
        return {}
    if len(mention_texts) != len(mention_embeddings):
        raise ValueError("mention_texts and mention_embeddings must be same length")

    # Convert embeddings to pgvector literals and pass as TEXT[] to avoid needing a psycopg2 vector adapter.
    vec_texts: List[str] = []
    for emb in mention_embeddings:
        try:
            vec_texts.append(to_pgvector_literal(emb))
        except Exception:
            vec_texts.append("")

    # Drop empty vectors
    idxs: List[int] = []
    texts: List[str] = []
    vecs: List[str] = []
    for i, (t, v) in enumerate(zip(mention_texts, vec_texts)):
        if t and v:
            idxs.append(i)
            texts.append(t)
            vecs.append(v)

    if not idxs:
        return {}

    kind_filter_list = kind_filter or ["Drug", "Procedure", "Finding", "Condition", "Anatomy", "Observation"]
    ord_to_original_idx = {ord_i: original_idx for ord_i, original_idx in enumerate(idxs, start=1)}
    limit_k = int(top_k_per_mention)
    max_d = float(max_distance)
    use_aliases = include_aliases and os.environ.get("BATCH_INCLUDE_ALIASES", "true").strip().lower() in ("1", "true", "yes")

    sql_concepts_only = f"""
    WITH mentions AS (
        SELECT m.ord::int AS mention_idx, m.mention_text, m.vec_text::vector AS mention_vec
        FROM unnest(%s::text[], %s::text[]) WITH ORDINALITY AS m(mention_text, vec_text, ord)
    )
    SELECT m.mention_idx, m.mention_text, kb.concept_id, kb.preferred_name, kb.kind, kb.distance, kb.definition, kb.domain_key
    FROM mentions m
    CROSS JOIN LATERAL (
        SELECT c.concept_id, c.preferred_name, c.kind, (c.embedding <=> m.mention_vec) AS distance,
               COALESCE(c.definition, '') AS definition, COALESCE(c.domain_key, '') AS domain_key
        FROM kb.concepts c
        WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED')
        ORDER BY c.embedding <=> m.mention_vec
        LIMIT {limit_k}
    ) kb
    WHERE kb.distance < %s
    ORDER BY m.mention_idx, kb.distance ASC;
    """
    sql_with_aliases = f"""
    WITH mentions AS (
        SELECT m.ord::int AS mention_idx, m.mention_text, m.vec_text::vector AS mention_vec
        FROM unnest(%s::text[], %s::text[]) WITH ORDINALITY AS m(mention_text, vec_text, ord)
    )
    SELECT m.mention_idx, m.mention_text, kb.concept_id, kb.preferred_name, kb.kind, kb.distance, kb.definition, kb.domain_key
    FROM mentions m
    CROSS JOIN LATERAL (
        SELECT * FROM (
            SELECT DISTINCT ON (concept_id) concept_id, preferred_name, kind, definition, domain_key, distance
            FROM (
                SELECT c.concept_id, c.preferred_name, c.kind, COALESCE(c.definition,'') AS definition, COALESCE(c.domain_key,'') AS domain_key,
                       (c.embedding <=> m.mention_vec) AS distance
                FROM kb.concepts c
                WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED')
                UNION ALL
                SELECT c.concept_id, ca.alias_text AS preferred_name, c.kind, COALESCE(c.definition,''), COALESCE(c.domain_key,''),
                       (ca.embedding <=> m.mention_vec)
                FROM kb.concept_aliases ca INNER JOIN kb.concepts c ON c.concept_id = ca.concept_id
                WHERE ca.embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED')
                UNION ALL
                SELECT la.kb_concept_id, la.alias_text AS preferred_name, c.kind, COALESCE(c.definition,''), COALESCE(c.domain_key,''),
                       (la.embedding <=> m.mention_vec)
                FROM kb.learned_aliases la INNER JOIN kb.concepts c ON c.concept_id = la.kb_concept_id
                WHERE la.embedding IS NOT NULL AND (la.is_validated = TRUE OR la.frequency_count > 0) AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED')
            ) u
            ORDER BY concept_id, distance ASC
        ) dedup
        ORDER BY distance ASC
        LIMIT {limit_k}
    ) kb
    WHERE kb.distance < %s
    ORDER BY m.mention_idx, kb.distance ASC;
    """
    sql = sql_with_aliases if use_aliases else sql_concepts_only
    params = (texts, vecs, kind_filter_list, kind_filter_list, kind_filter_list, max_d) if use_aliases else (texts, vecs, kind_filter_list, max_d)

    out: Dict[int, List[Dict[str, Any]]] = {i: [] for i in idxs}
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("SET LOCAL hnsw.ef_search = %s", (64,))
            except Exception:
                pass
            cur.execute(sql, params)
            rows = cur.fetchall()

        for row in rows:
            mention_idx = row[0]
            concept_id, preferred_name, kind, distance = row[2], row[3], row[4], row[5]
            definition = row[6] if len(row) > 6 else ""
            domain_key = (str(row[7]).strip() or None) if len(row) > 7 else None
            original_idx = ord_to_original_idx.get(int(mention_idx))
            if original_idx is None:
                continue
            dist_f = float(distance) if distance is not None else None
            out[original_idx].append(
                {
                    "concept_id": concept_id,
                    "preferred_name": preferred_name,
                    "kind": kind,
                    "definition": definition,
                    "domain_key": domain_key,
                    "distance": dist_f,
                    "match_score": (1.0 - min(dist_f, 1.0)) if dist_f is not None else None,
                    "match_source": "embedding_batch",
                }
            )
        return out
    except Exception as e:
        if use_aliases and logger:
            logger.warning(f"Batch embedding lookup with aliases failed ({e}), retrying concepts-only.")
        if use_aliases:
            try:
                with conn.cursor() as cur:
                    try:
                        cur.execute("SET LOCAL hnsw.ef_search = %s", (64,))
                    except Exception:
                        pass
                    cur.execute(sql_concepts_only, (texts, vecs, kind_filter_list, max_d))
                    rows = cur.fetchall()
                for row in rows:
                    mention_idx = row[0]
                    concept_id, preferred_name, kind, distance = row[2], row[3], row[4], row[5]
                    definition = row[6] if len(row) > 6 else ""
                    domain_key = (str(row[7]).strip() or None) if len(row) > 7 else None
                    original_idx = ord_to_original_idx.get(int(mention_idx))
                    if original_idx is None:
                        continue
                    dist_f = float(distance) if distance is not None else None
                    out[original_idx].append(
                        {
                            "concept_id": concept_id,
                            "preferred_name": preferred_name,
                            "kind": kind,
                            "definition": definition,
                            "domain_key": domain_key,
                            "distance": dist_f,
                            "match_score": (1.0 - min(dist_f, 1.0)) if dist_f is not None else None,
                            "match_source": "embedding_batch",
                        }
                    )
                return out
            except Exception as e2:
                if logger:
                    logger.warning(f"Batch embedding lookup (concepts-only) failed: {e2}")
                return {}
        if logger:
            logger.warning(f"Batch embedding lookup failed: {e}")
        return {}


def _batch_trigram_phonetic_scores(
    conn,
    mention: str,
    preferred_names: List[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute trigram and phonetic scores for one mention vs many candidate names (parity with single-path two-stage). Returns (tri_map, pho_map) keyed by preferred_name."""
    tri_map: Dict[str, float] = {}
    pho_map: Dict[str, float] = {}
    if not mention or not preferred_names:
        return tri_map, pho_map
    mention_ph = _normalize_mention_for_phonetic(mention)
    if not mention_ph:
        return tri_map, pho_map
    try:
        ensure_pg_trgm(conn, logger=logger)
        ensure_fuzzystrmatch(conn, logger=logger)
        with conn.cursor() as cur:
            placeholders = ", ".join(["(%s)"] * len(preferred_names))
            cur.execute(
                """
                WITH q AS (
                    SELECT lower(%s) AS q_full,
                        metaphone(lower(%s), 10) AS q_mfull,
                        metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst
                ),
                names(pname) AS (VALUES """
                + placeholders
                + """ )
                SELECT v.pname,
                    similarity(lower(v.pname), q.q_full) AS tri,
                    GREATEST(
                        0.8 * (1.0 - levenshtein(metaphone(split_part(lower(v.pname), ' ', 1), 10), q.q_mfirst)::float
                            / GREATEST(length(metaphone(split_part(lower(v.pname), ' ', 1), 10)), length(q.q_mfirst), 1)),
                        0.8 * (1.0 - levenshtein(metaphone(lower(v.pname), 10), q.q_mfull)::float
                            / GREATEST(length(metaphone(lower(v.pname), 10)), length(q.q_mfull), 1))
                    ) AS pho
                FROM names v
                CROSS JOIN q
                """,
                (mention_ph, mention_ph, mention_ph) + tuple(preferred_names),
            )
            for (pname, tri, pho) in cur.fetchall():
                if pname is not None:
                    tri_map[pname] = float(tri) if tri is not None else 0.0
                    pho_map[pname] = float(pho) if pho is not None else 0.0
    except Exception as e:
        if logger:
            logger.debug("Batch tri/pho failed: %s", e)
        try:
            with conn.cursor() as cur:
                placeholders = ", ".join(["(%s)"] * len(preferred_names))
                cur.execute(
                    "SELECT v.pname, similarity(lower(v.pname), lower(%s)) FROM (VALUES " + placeholders + ") AS v(pname)",
                    (mention_ph,) + tuple(preferred_names),
                )
                for (pname, sim) in cur.fetchall():
                    if pname is not None and sim is not None:
                        tri_map[pname] = float(sim)
                        pho_map[pname] = 0.0
        except Exception:
            pass
    return tri_map, pho_map


def run_batch_global_vector_search(
    conn,
    items: List[Tuple[Any, ...]],
    client: Optional[Any],
    logger: Optional[logging.Logger] = None,
    *,
    top_k_per_mention: int = 8,
    max_distance: float = 0.5,
    raw_transcript: Optional[str] = None,
    anchor_concept_ids_by_entity: Optional[Dict[int, List[int]]] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Single embedding request + single DB round trip for all entities that need global KB search.
    Reduces grounding latency from N lookups to 1 batch embed + 1 batch vector search.

    Dual-signal (Top-3 Intent Boost): When item is 5-tuple (entity_idx, search_term, kind_filter, original_span, hints),
    we embed original_span + search_term + hints, run vector search for each, merge results per entity, and apply
    consensus boost: if the same concept_id appears from both the original span and an intent hint, match_score is
    overridden to 1.0 (Gold Standard) so the LLM Judge can auto-link.

    Args:
        conn: DB connection
        items: List of (entity_idx, search_term, kind_filter_list) or (entity_idx, search_term, kind_filter_list, original_span, hints)
            or 6-tuple (..., domain) with domain from Brain (str or list of str).
        client: OpenAI client (for batch embed)
        logger: Optional logger
        top_k_per_mention: Max candidates per term
        max_distance: Max vector distance (e.g. 0.5)
        raw_transcript: Optional transcript (for two-stage Q2 and fallbacks)
        anchor_concept_ids_by_entity: Optional per-entity anchor concept IDs (for neighbor + two-stage)

    Returns:
        Dict[entity_idx, list of candidate dicts] (match_score, preferred_name, concept_id, kind, consensus_boost, etc.)
    """
    if LOCAL_ONLY:
        return {}
    if not items or not client:
        return {}
    try:
        from kb_ner_embeddings import embed_texts
    except ImportError:
        if logger:
            logger.warning("run_batch_global_vector_search: embed_texts not available")
        return {}

    # Normalize items to 5-tuple (entity_idx, search_term, kind_filter, original_span, hints); extract per-entity Brain domain, suggestion_probability, and hint_probabilities
    normalized_items: List[Tuple[int, str, List[str], Optional[str], Optional[List[str]]]] = []
    entity_domain_by_idx: Dict[int, List[str]] = {}
    entity_suggestion_probability_by_idx: Dict[int, float] = {}  # Store suggestion_probability per entity
    entity_suggestion_terms_by_idx: Dict[int, List[str]] = {}  # Store search_term + hints per entity (for suggestion boost matching)
    entity_hint_probabilities_by_idx: Dict[int, Dict[str, float]] = {}  # Store hint_probabilities dict per entity (hint_text -> probability)
    entity_query_expansion_terms_by_idx: Dict[int, List[str]] = {}  # Store query_expansion list per entity (up to 3 terms)
    for it in items:
        if len(it) >= 5:
            normalized_items.append((int(it[0]), str(it[1]).strip(), it[2] or [], (it[3] or "").strip() or None, it[4] if isinstance(it[4], list) else []))
        else:
            normalized_items.append((int(it[0]), str(it[1]).strip(), it[2] or [], None, None))
        # Per-entity domain from Brain (6th element): str or list of str → list of lowercase non-empty
        if len(it) >= 6 and it[5] is not None:
            raw_d = it[5]
            if isinstance(raw_d, str):
                dom_list = [raw_d]
            elif isinstance(raw_d, list):
                dom_list = list(raw_d)
            else:
                dom_list = []
            normalized_domains = [
                (d or "").strip().lower() for d in dom_list
                if (d or "").strip().lower() and (d or "").strip().lower() != "general"
            ]
            if normalized_domains:
                entity_domain_by_idx[int(it[0])] = normalized_domains
        # Per-entity suggestion_probability from Brain NER (7th element): float → store for boost calculation
        if len(it) >= 7 and it[6] is not None:
            try:
                prob = float(it[6])
                if 0.0 <= prob <= 1.0:
                    entity_suggestion_probability_by_idx[int(it[0])] = prob
            except (ValueError, TypeError):
                pass
        # Per-entity hint_probabilities from Brain NER (8th element): dict {hint_text: probability} → store for weighted hint matching
        if len(it) >= 8 and it[7] is not None:
            try:
                hint_probs = it[7]
                if isinstance(hint_probs, dict):
                    # Validate and normalize hint probabilities
                    validated_probs = {}
                    for hint_text, prob_val in hint_probs.items():
                        if hint_text and isinstance(hint_text, str):
                            try:
                                prob_float = float(prob_val)
                                if 0.0 <= prob_float <= 1.0:
                                    validated_probs[hint_text.strip().lower()] = prob_float
                            except (ValueError, TypeError):
                                pass
                    if validated_probs:
                        entity_hint_probabilities_by_idx[int(it[0])] = validated_probs
            except (ValueError, TypeError, AttributeError):
                pass
        # Store suggestion terms (search_term + hints) for matching candidates
        if len(it) >= 5:
            entity_idx = int(it[0])
            search_term = str(it[1]).strip() if it[1] else ""
            hints = it[4] if isinstance(it[4], list) else []
            suggestion_terms = []
            if search_term:
                suggestion_terms.append(search_term.lower())
            if hints:
                for hint in hints[:3]:  # Up to 3 hints
                    if hint and isinstance(hint, str):
                        suggestion_terms.append(hint.strip().lower())
            if suggestion_terms:
                entity_suggestion_terms_by_idx[entity_idx] = suggestion_terms
        # Per-entity query_expansion (9th element): list or comma-separated string, cap at 3
        if len(it) >= 9 and it[8] is not None:
            qe_raw = it[8]
            if isinstance(qe_raw, list):
                qe_list = [str(x).strip().lower() for x in qe_raw if str(x).strip()][:3]
            elif isinstance(qe_raw, str) and (qe_raw or "").strip():
                qe_list = [x.strip().lower() for x in qe_raw.split(",") if x.strip()][:3]
            else:
                qe_list = []
            if qe_list:
                entity_query_expansion_terms_by_idx[int(it[0])] = qe_list

    # Batch embedding mode: one request per entity (3+2 hints combined into single string) to cut ~60s to ~15s.
    # Set BATCH_EMBED_ONE_PER_ENTITY=true to group original_span + search_term + up to 3 hints into one string per entity.
    batch_embed_one_per_entity = os.environ.get("BATCH_EMBED_ONE_PER_ENTITY", "true").strip().lower() in ("1", "true", "yes")

    # Build flat list of (position, entity_idx, term, is_original) for dual-signal; or single term per entity for legacy
    flat: List[Tuple[int, int, str, bool]] = []  # (position, entity_idx, term, is_original)
    pos = 0
    for entity_idx, search_term, kind_filter, original_span, hints in normalized_items:
        if not search_term and not original_span:
            continue
        if batch_embed_one_per_entity and (original_span or search_term):
            # One combined term per entity: "original | search_term | hint1 | hint2 | hint3" → single embed per entity
            parts = []
            if original_span and original_span.strip():
                parts.append(original_span.strip())
            if search_term and search_term.strip() and search_term.strip() not in {p.strip().lower() for p in parts}:
                parts.append(search_term.strip())
            for h in (hints or [])[:3]:
                if h and str(h).strip() and str(h).strip().lower() not in {p.strip().lower() for p in parts}:
                    parts.append(str(h).strip())
            combined = " | ".join(parts) if parts else (search_term or original_span or "").strip()
            if combined:
                flat.append((pos, entity_idx, combined, True))
                pos += 1
            continue
        if original_span is not None and (hints is not None or search_term):
            # Dual-signal: original + primary + up to 3 hints (unique per entity)
            seen = set()
            if original_span and original_span not in seen:
                flat.append((pos, entity_idx, original_span, True))
                seen.add(original_span)
                pos += 1
            if search_term and search_term not in seen:
                flat.append((pos, entity_idx, search_term, False))
                seen.add(search_term)
                pos += 1
            for h in (hints or [])[:3]:
                if h and str(h).strip() and str(h).strip() not in seen:
                    flat.append((pos, entity_idx, str(h).strip(), False))
                    seen.add(str(h).strip())
                    pos += 1
        else:
            if search_term:
                flat.append((pos, entity_idx, search_term, False))
                pos += 1
    if not flat:
        return {}

    unique_terms = [t for (_, _, t, _) in flat]
    # Union of all kind filters
    kind_union = set()
    for _, _, kf, _, _ in normalized_items:
        if kf:
            kind_union.update(kf)
    kind_filter_list = list(kind_union) if kind_union else [
        "Drug", "Procedure", "Service", "Finding", "Condition", "Anatomy", "Observation", "Vaccine", "Nutrition"
    ]

    # Parity with single path: when domain is detected (transcript or per-entity Brain domain), fetch more candidates
    effective_top_k = top_k_per_mention
    _any_domain = bool(entity_domain_by_idx and any(entity_domain_by_idx.values()))
    if not _any_domain and raw_transcript:
        try:
            from kb_ner_domain import detect_domain
            _detected = detect_domain(raw_transcript)
            if _detected and (_detected or "").strip().lower() not in ("", "general"):
                _any_domain = True
        except Exception:
            pass
    if _any_domain:
        try:
            _batch_domain_k = int(os.environ.get("BATCH_DOMAIN_FETCH_K", "24"))
            effective_top_k = max(top_k_per_mention, min(50, _batch_domain_k))
        except Exception:
            effective_top_k = max(top_k_per_mention, min(30, top_k_per_mention * 3))
        if logger:
            logger.debug("Batch global: domain (Brain or transcript) → effective_top_k=%s", effective_top_k)

    embeddings = embed_texts(unique_terms, client=client, logger=logger)
    if not embeddings or len(embeddings) != len(unique_terms):
        if logger:
            logger.warning("Batch embed failed or length mismatch")
        return {}
    valid = [(i, t, e) for i, (t, e) in enumerate(zip(unique_terms, embeddings)) if t and e]
    if not valid:
        return {}
    unique_terms = [t for _, t, _ in valid]
    embeddings = [e for _, _, e in valid]
    # batch_result[j] corresponds to valid[j] = (orig_idx, term, emb); orig_idx is index in flat
    # So for each batch position j we get (entity_idx, is_original) from flat[valid[j][0]]
    valid_idx_to_entity_origin: Dict[int, List[Tuple[int, bool]]] = {}
    # For Multi-Query 3+2: entity -> [(valid_idx, pos, is_orig)] sorted by pos (original first, then hints)
    entity_to_ordered_terms: Dict[int, List[Tuple[int, int, bool]]] = {}
    for j in range(len(valid)):
        orig_idx = valid[j][0]
        if 0 <= orig_idx < len(flat):
            pos, entity_idx, _term, is_orig = flat[orig_idx]
            valid_idx_to_entity_origin.setdefault(j, []).append((entity_idx, is_orig))
            entity_to_ordered_terms.setdefault(entity_idx, []).append((j, pos, is_orig))
    for eidx in entity_to_ordered_terms:
        entity_to_ordered_terms[eidx].sort(key=lambda x: x[1])  # by pos

    batch_result = kb_lookup_concepts_by_embedding_batch(
        conn,
        unique_terms,
        embeddings,
        kind_filter=kind_filter_list,
        top_k_per_mention=effective_top_k,
        max_distance=max_distance,
        logger=logger,
    )
    # Multi-Query 3+2: Original top 3 + each hint top 2, then suggestion boost (+0.15) when concept in both.
    # When false, use merged batch (single ranking); when true, hints get dedicated lanes so ASR noise cannot drown clinical intent.
    use_3_2_pool = os.environ.get("MULTI_QUERY_3_2_POOL", "true").strip().lower() in ("1", "true", "yes")
    suggestion_boost = float(os.environ.get("SUGGESTION_BOOST_ADD", "0.15").strip() or "0.15")
    try:
        suggestion_boost = max(0.0, min(1.0, suggestion_boost))
    except Exception:
        suggestion_boost = 0.15

    # Map back to entity_idx; merge when dual-signal and apply consensus / suggestion boost
    out: Dict[int, List[Dict[str, Any]]] = {}
    for entity_idx, search_term, kind_filter, original_span, hints in normalized_items:
        kf_set = set(kind_filter or [])
        kind_ok = lambda c: (not kf_set) or (c.get("kind") in kf_set)

        if use_3_2_pool and entity_idx in entity_to_ordered_terms:
            # 3+2: Original top 3, each hint top 2; then suggestion boost if concept in both.
            ordered = entity_to_ordered_terms[entity_idx]
            original_top3: List[Dict[str, Any]] = []
            hint_tops: List[Dict[str, Any]] = []  # up to 2 per hint term, flattened
            for term_rank, (valid_idx, _pos, is_orig) in enumerate(ordered):
                cands = [c for c in (batch_result.get(valid_idx) or []) if c.get("concept_id") is not None and kind_ok(c)]
                cands_sorted = sorted(cands, key=lambda x: -float(x.get("match_score") or 0))
                if term_rank == 0:
                    original_top3 = cands_sorted[:3]
                else:
                    hint_tops.extend(cands_sorted[:2])
            original_top3_ids = {c.get("concept_id") for c in original_top3 if c.get("concept_id")}
            hint_top_ids = {c.get("concept_id") for c in hint_tops if c.get("concept_id")}
            # Pool: merge by concept_id (keep best score)
            by_cid: Dict[int, Dict[str, Any]] = {}
            for c in original_top3 + hint_tops:
                cid = c.get("concept_id")
                if cid is None:
                    continue
                score = float(c.get("match_score") or 0)
                from_orig = cid in original_top3_ids
                from_hint = cid in hint_top_ids
                if cid not in by_cid:
                    by_cid[cid] = dict(c)
                    by_cid[cid]["_from_original"] = from_orig
                    by_cid[cid]["_from_hint"] = from_hint
                else:
                    by_cid[cid]["_from_original"] = by_cid[cid].get("_from_original") or from_orig
                    by_cid[cid]["_from_hint"] = by_cid[cid].get("_from_hint") or from_hint
                    if score > float(by_cid[cid].get("match_score") or 0):
                        by_cid[cid]["match_score"] = score
            # Suggestion boost: +0.15 (or env) for any concept that came from a hint (not only when in both)
            for cid, c in by_cid.items():
                if c.get("_from_hint"):
                    c["suggestion_boost"] = True
                    new_score = min(1.0, float(c.get("match_score") or 0) + suggestion_boost)
                    c["match_score"] = new_score
            result_list = [c for c in by_cid.values()]
            for c in result_list:
                if c.get("hybrid_score") is None:
                    c["hybrid_score"] = float(c.get("match_score") or 0)
            result_list.sort(key=lambda x: -float(x.get("match_score") or 0))
            out[entity_idx] = result_list[: max(top_k_per_mention, 10)]
            if logger and (original_top3 or hint_tops):
                logger.debug(
                    "Batch global (3+2): entity_idx=%s original_top3=%s hint_cands=%s pool=%s",
                    entity_idx, len(original_top3), len(hint_tops), len(result_list),
                )
        else:
            # Merged batch (legacy): single merge by concept, consensus -> 1.0
            by_concept: Dict[int, Dict[str, Any]] = {}
            for valid_idx, cands in batch_result.items():
                for (eidx, is_orig) in valid_idx_to_entity_origin.get(valid_idx, []):
                    if eidx != entity_idx:
                        continue
                    for c in cands:
                        cid = c.get("concept_id")
                        if cid is None:
                            continue
                        if kf_set and c.get("kind") not in kf_set:
                            continue
                        score = float(c.get("match_score") or 0)
                        existing = by_concept.get(cid)
                        if existing is None:
                            by_concept[cid] = dict(c)
                            by_concept[cid]["_from_original"] = is_orig
                            by_concept[cid]["_from_hint"] = not is_orig
                        else:
                            existing["_from_original"] = existing.get("_from_original") or is_orig
                            existing["_from_hint"] = existing.get("_from_hint") or (not is_orig)
                            if score > float(existing.get("match_score") or 0):
                                existing["match_score"] = score
                                existing["distance"] = c.get("distance")
            for cid, c in by_concept.items():
                from_orig = c.get("_from_original", False)
                from_hint = c.get("_from_hint", False)
                if from_orig and from_hint:
                    c["match_score"] = 1.0
                    c["consensus_boost"] = True
            result_list = [c for c in by_concept.values()]
            for c in result_list:
                if c.get("hybrid_score") is None:
                    c["hybrid_score"] = float(c.get("match_score") or 0)
            result_list.sort(key=lambda x: -float(x.get("match_score") or 0))
            out[entity_idx] = result_list[: max(top_k_per_mention, 10)]

    # Exact-match injection: when Brain provides search_term (e.g. "Ortolani test"), prepend KB exact matches
    # so they rank first instead of being buried by embedding-only batch (fixes Ortolani / Norberg angle).
    # Also try partial matches: if search_term="ortolani test", also try exact match on "ortolani" (handles cases where KB has "Ortolani" without "test").
    for entity_idx, search_term, kind_filter, _original_span, _hints in normalized_items:
        if not (search_term or "").strip():
            continue
        search_term_clean = (search_term or "").strip()
        exact_list = kb_lookup_concept_exact(conn, search_term_clean)
        # If no exact match, try partial: if search_term has multiple words, try exact match on each word
        # Example: "ortolani test" → try "ortolani" and "test" separately
        if not exact_list and " " in search_term_clean:
            words = search_term_clean.split()
            # Try longest word first (more specific), then shorter words
            words_sorted = sorted(words, key=len, reverse=True)
            for word in words_sorted:
                if len(word) < 3:  # Skip very short words like "test", "a", "an"
                    continue
                exact_list = kb_lookup_concept_exact(conn, word)
                if exact_list:
                    if logger:
                        logger.debug("Batch global: partial exact-match found for entity_idx=%s search_term=%r → word=%r", entity_idx, search_term_clean[:50], word)
                    break
        if not exact_list:
            continue
        kf_set = set(kind_filter or [])
        kind_ok = lambda c: (not kf_set) or (c.get("kind") in kf_set)
        current = out.get(entity_idx) or []
        by_cid = {c.get("concept_id"): dict(c) for c in current if c.get("concept_id") is not None}
        for c in exact_list:
            if not kind_ok(c):
                continue
            cid = c.get("concept_id")
            c = dict(c)
            # Exact match gets perfect score (1.0) to rank first
            c["match_score"] = 1.0
            c["hybrid_score"] = 1.0
            c["final_score"] = 1.0  # Also set final_score to ensure it ranks first
            c["openai_score"] = 1.0  # Set all component scores to 1.0 for exact matches
            c["trigram_score"] = 1.0
            c["phonetic_score"] = 1.0
            c["_from_exact_match"] = True
            c["_from_original"] = False
            c["_from_hint"] = True
            by_cid[cid] = c
        merged = list(by_cid.values())
        merged.sort(key=lambda x: -float(x.get("final_score", x.get("match_score") or 0)))
        out[entity_idx] = merged[: max(top_k_per_mention, 10)]
        if logger and exact_list:
            logger.debug("Batch global: exact-match injected for entity_idx=%s search_term=%r → %d matches", entity_idx, search_term_clean[:50], len(exact_list))

    # Trigram retrieval (parity with single path): fetch candidates from KB by trigram for search_term and hint terms.
    # Single path runs kb_lookup_concept_by_trgm(conn, text) as retrieval; batch previously only used trigram for rerank.
    # Run trigram search on search_term and on each hint (same terms we use for vector), merge into pool by concept_id.
    # Also try partial word matches: if search_term="ortolani test", also search for "ortolani" to find "Ortolani" concept.
    trigram_retrieval_topk = min(5, max(2, top_k_per_mention))
    for entity_idx, search_term, kind_filter, original_span, hints in normalized_items:
        terms_to_trgm: List[str] = []
        if (search_term or "").strip():
            terms_to_trgm.append((search_term or "").strip())
            # Also try partial matches: if search_term has multiple words, try each word separately
            # Example: "ortolani test" → also try "ortolani" to find "Ortolani" concept
            if " " in (search_term or "").strip():
                words = (search_term or "").strip().split()
                for word in words:
                    if len(word) >= 4:  # Only use words >= 4 chars (avoid "test", "the", etc.)
                        if word not in terms_to_trgm:
                            terms_to_trgm.append(word)
        if (original_span or "").strip() and (original_span or "").strip() not in terms_to_trgm:
            terms_to_trgm.append((original_span or "").strip())
            # Also try partial matches for original_span
            if " " in (original_span or "").strip():
                words = (original_span or "").strip().split()
                for word in words:
                    if len(word) >= 4 and word not in terms_to_trgm:
                        terms_to_trgm.append(word)
        for h in (hints or [])[:3]:
            if h and (str(h).strip() not in terms_to_trgm):
                terms_to_trgm.append(str(h).strip())
        if not terms_to_trgm:
            continue
        kf_set = set(kind_filter or [])
        kind_ok = lambda c: (not kf_set) or (c.get("kind") in kf_set)
        current = out.get(entity_idx) or []
        by_cid = {c.get("concept_id"): dict(c) for c in current if c.get("concept_id") is not None}
        for term in terms_to_trgm:
            trgm_list = kb_lookup_concept_by_trgm(
                conn, term, top_k=trigram_retrieval_topk, kind_filter=kind_filter or None, logger=logger
            )
            for c in trgm_list:
                if not kind_ok(c):
                    continue
                cid = c.get("concept_id")
                if cid is None:
                    continue
                score = float(c.get("similarity_score") or c.get("hybrid_score") or 0)
                c = dict(c)
                c["match_source"] = c.get("match_source") or "pg_trgm"
                c["_from_trigram_retrieval"] = True
                # Boost score if this is a partial word match (e.g., "ortolani" matching "Ortolani" from "ortolani test")
                if term.lower() in (c.get("preferred_name") or "").lower() or (c.get("preferred_name") or "").lower() in term.lower():
                    score = max(score, 0.85)  # Boost partial matches to at least 0.85
                if cid not in by_cid or score > float(by_cid[cid].get("match_score") or 0):
                    c["match_score"] = score
                    c["hybrid_score"] = c.get("hybrid_score") if c.get("hybrid_score") is not None else score
                    by_cid[cid] = c
        merged = list(by_cid.values())
        merged.sort(key=lambda x: -float(x.get("match_score") or 0))
        out[entity_idx] = merged[: max(top_k_per_mention, 10)]

    # Transcript domain (fallback when entity has no Brain domain)
    _transcript_domain: Optional[str] = None
    if raw_transcript:
        try:
            from kb_ner_domain import detect_domain
            _transcript_domain = detect_domain(raw_transcript)
            if (_transcript_domain or "").strip().lower() in ("", "general"):
                _transcript_domain = None
        except Exception:
            pass

    # Parity with single path: neighbor anchoring, trigram/phonetic, multi-signal rerank, generic penalty
    detected_domain_for_parity: Optional[str] = None
    detected_domains_set_parity: Optional[set] = None
    if raw_transcript:
        try:
            from kb_ner_domain import detect_domain
            detected_domain_for_parity = detect_domain(raw_transcript)
            detected_domains_list = detect_domain(raw_transcript, return_multiple=True)
            detected_domains_set_parity = set(
                (d or "").strip().lower()
                for d in (detected_domains_list or [])
                if (d or "").strip().lower() and (d or "").strip().lower() != "general"
            )
            if not detected_domains_set_parity and detected_domain_for_parity and (detected_domain_for_parity or "").strip().lower() not in ("", "general"):
                detected_domains_set_parity = {(detected_domain_for_parity or "").strip().lower()}
        except Exception:
            pass
    # Ensure PostgreSQL extensions once at batch level (reduces log spam from repeated checks)
    try:
        ensure_pg_trgm(conn, logger=logger)
        ensure_fuzzystrmatch(conn, logger=logger)
    except Exception:
        pass  # Will be checked again in _batch_trigram_phonetic_scores if needed
    
    entity_to_mention: Dict[int, str] = {}
    entity_to_original_span: Dict[int, str] = {}  # Track original_span per entity for proper scoring
    entity_to_search_term: Dict[int, str] = {}  # Track search_term per entity for proper scoring
    entity_to_kind_filter: Dict[int, List[str]] = {}
    for entity_idx, search_term, kind_filter, original_span, _ in normalized_items:
        entity_to_mention[entity_idx] = (original_span or search_term or "").strip() or ""
        entity_to_original_span[entity_idx] = (original_span or "").strip() if original_span else ""
        entity_to_search_term[entity_idx] = (search_term or "").strip() if search_term else ""
        entity_to_kind_filter[entity_idx] = list(kind_filter or [])

    use_multi_signal_batch = GLOBAL_TWO_STAGE_MULTI_SIGNAL_RANKER
    for entity_idx in list(out.keys()):
        candidates = out[entity_idx]
        mention = entity_to_mention.get(entity_idx) or ""
        kf = entity_to_kind_filter.get(entity_idx) or []
        # Per-entity domain: Brain first, else transcript
        entity_domains = entity_domain_by_idx.get(entity_idx) or []
        _entity_domain = (entity_domains[0] if entity_domains else None) or (detected_domain_for_parity or "").strip() or None
        detected_domain_norm = (_entity_domain or "").strip().lower()
        anchor_ids = (anchor_concept_ids_by_entity or {}).get(entity_idx)
        if anchor_ids and candidates:
            neighbors = _fetch_neighbor_candidates(
                conn, anchor_ids, kf or list(kind_filter_list), _entity_domain or detected_domain_for_parity,
                GLOBAL_NEIGHBOR_BOOST, logger
            )
            if neighbors:
                candidates = _merge_neighbor_candidates(
                    candidates, conn, anchor_ids, kf or list(kind_filter_list),
                    _entity_domain or detected_domain_for_parity, logger, neighbor_list=neighbors
                )
                out[entity_idx] = candidates
        preferred_names = list(dict.fromkeys((c.get("preferred_name") or "").strip() for c in candidates if (c.get("preferred_name") or "").strip()))
        
        # CRITICAL FIX: Score candidates against the term that retrieved them, not a single mention
        # Candidates from original_span search should be scored against original_span
        # Candidates from search_term search should be scored against search_term
        original_span_for_entity = entity_to_original_span.get(entity_idx) or ""
        search_term_for_entity = entity_to_search_term.get(entity_idx) or ""
        
        # Build separate scoring maps for original_span vs search_term candidates
        tri_map_orig: Dict[str, float] = {}
        pho_map_orig: Dict[str, float] = {}
        tri_map_search: Dict[str, float] = {}
        pho_map_search: Dict[str, float] = {}
        
        # Helper function to get partial term (longest word >= 4 chars)
        def _get_partial_term(term: str) -> str:
            if " " in term:
                words = term.split()
                longest_word = max(words, key=len)
                if len(longest_word) >= 4:
                    return longest_word
            return term
        
        # Score against original_span if it exists (call once with full term, partial handled in function)
        if original_span_for_entity:
            tri_map_orig_full, pho_map_orig_full = _batch_trigram_phonetic_scores(conn, original_span_for_entity, preferred_names, logger)
            # Also try partial match if different from full
            orig_partial = _get_partial_term(original_span_for_entity)
            if orig_partial != original_span_for_entity:
                tri_map_orig_partial, pho_map_orig_partial = _batch_trigram_phonetic_scores(conn, orig_partial, preferred_names, logger)
                for pname in preferred_names:
                    tri_map_orig[pname] = max(tri_map_orig_full.get(pname, 0.0), tri_map_orig_partial.get(pname, 0.0))
                    pho_map_orig[pname] = max(pho_map_orig_full.get(pname, 0.0), pho_map_orig_partial.get(pname, 0.0))
            else:
                tri_map_orig = tri_map_orig_full
                pho_map_orig = pho_map_orig_full
        
        # Score against search_term if it exists and is different from original_span
        if search_term_for_entity and search_term_for_entity.lower() != (original_span_for_entity or "").lower():
            tri_map_search_full, pho_map_search_full = _batch_trigram_phonetic_scores(conn, search_term_for_entity, preferred_names, logger)
            # Also try partial match if different from full
            search_partial = _get_partial_term(search_term_for_entity)
            if search_partial != search_term_for_entity:
                tri_map_search_partial, pho_map_search_partial = _batch_trigram_phonetic_scores(conn, search_partial, preferred_names, logger)
                for pname in preferred_names:
                    tri_map_search[pname] = max(tri_map_search_full.get(pname, 0.0), tri_map_search_partial.get(pname, 0.0))
                    pho_map_search[pname] = max(pho_map_search_full.get(pname, 0.0), pho_map_search_partial.get(pname, 0.0))
            else:
                tri_map_search = tri_map_search_full
                pho_map_search = pho_map_search_full
        
        # Fallback: if no original_span or search_term, use mention (backward compatibility)
        tri_map_fallback: Dict[str, float] = {}
        pho_map_fallback: Dict[str, float] = {}
        if not original_span_for_entity and not search_term_for_entity:
            tri_map_full, pho_map_full = _batch_trigram_phonetic_scores(conn, mention, preferred_names, logger)
            mention_partial = _get_partial_term(mention)
            if mention_partial != mention:
                tri_map_partial, pho_map_partial = _batch_trigram_phonetic_scores(conn, mention_partial, preferred_names, logger)
                for pname in preferred_names:
                    tri_map_fallback[pname] = max(tri_map_full.get(pname, 0.0), tri_map_partial.get(pname, 0.0))
                    pho_map_fallback[pname] = max(pho_map_full.get(pname, 0.0), pho_map_partial.get(pname, 0.0))
            else:
                tri_map_fallback = tri_map_full
                pho_map_fallback = pho_map_full
        neighbor_cids = set(anchor_ids or [])
        for c in candidates:
            # Preserve exact matches: if this candidate came from exact match, keep score at 1.0 and skip recalculation
            if c.get("_from_exact_match") and float(c.get("final_score", 0.0)) >= 1.0:
                # Exact match already has perfect score, don't recalculate
                continue
            pname = (c.get("preferred_name") or "").strip()
            # Use the appropriate scoring map based on which search retrieved this candidate
            from_original = c.get("_from_original", False)
            if from_original and original_span_for_entity:
                # Candidate came from original_span search → score against original_span
                tri = float(tri_map_orig.get(pname, 0.0))
                pho = float(pho_map_orig.get(pname, 0.0))
            elif not from_original and search_term_for_entity:
                # Candidate came from search_term/hint search → score against search_term
                tri = float(tri_map_search.get(pname, 0.0))
                pho = float(pho_map_search.get(pname, 0.0))
            elif original_span_for_entity:
                # If we only have original_span (no search_term), use original_span scoring for all
                tri = float(tri_map_orig.get(pname, 0.0))
                pho = float(pho_map_orig.get(pname, 0.0))
            elif search_term_for_entity:
                # If we only have search_term (no original_span), use search_term scoring for all
                tri = float(tri_map_search.get(pname, 0.0))
                pho = float(pho_map_search.get(pname, 0.0))
            else:
                # Fallback: use mention-based scoring (backward compatibility)
                tri = float(tri_map_fallback.get(pname, 0.0))
                pho = float(pho_map_fallback.get(pname, 0.0))
            c["trigram_score"] = tri
            c["phonetic_score"] = pho
            openai_score = float(c.get("openai_score", 0.0))
            aux_vector_score = 0.0
            # Check if we have any trigram/phonetic scores available (from any of the scoring maps)
            has_tri_pho_scores = bool(
                (tri_map_orig and len(tri_map_orig) > 0) or
                (tri_map_search and len(tri_map_search) > 0) or
                (tri_map_fallback and len(tri_map_fallback) > 0) or
                (pho_map_orig and len(pho_map_orig) > 0) or
                (pho_map_search and len(pho_map_search) > 0) or
                (pho_map_fallback and len(pho_map_fallback) > 0)
            )
            if use_multi_signal_batch and has_tri_pho_scores:
                vb_weight = 0.0
                oa_weight = GLOBAL_TWO_STAGE_OPENAI_WEIGHT + (REDISTRIBUTED_VECTOR_WEIGHT * 0.43)
                tri_weight = GLOBAL_TWO_STAGE_TRIGRAM_WEIGHT + (REDISTRIBUTED_VECTOR_WEIGHT * 0.14)
                pho_weight = GLOBAL_TWO_STAGE_PHONETIC_WEIGHT + (REDISTRIBUTED_VECTOR_WEIGHT * 0.43)
            cand_domain = (c.get("domain_key") or "").strip().lower() or ""
            if cand_domain == detected_domain_norm and detected_domain_norm:
                domain_score = GLOBAL_TWO_STAGE_DOMAIN_WEIGHT
            elif cand_domain and cand_domain not in GLOBAL_TWO_STAGE_DOMAIN_NEUTRAL and detected_domain_norm:
                domain_score = -GLOBAL_TWO_STAGE_DOMAIN_FRICTION
            else:
                domain_score = 0.0
            cand_kind = (c.get("kind") or "").strip()
            cand_tokens = set((pname or "").lower().split())
            is_generic_proc = (
                cand_kind in GENERIC_PENALTY_KINDS
                and len(cand_tokens & GENERIC_PROC_TOKENS) > 0
            )
            generic_penalty = -GLOBAL_GENERIC_PROC_PENALTY if is_generic_proc else 0.0
            is_neighbor = (c.get("concept_id") in neighbor_cids) or c.get("is_neighbor")
            relation_boost = GLOBAL_TWO_STAGE_RELATION_WEIGHT if is_neighbor else 0.0
            # Suggestion boost: apply to candidates that match search_term or hints, weighted by hint probabilities
            suggestion_prob = entity_suggestion_probability_by_idx.get(entity_idx)
            suggestion_boost = 0.0
            hint_match_source = None  # Track which hint matched for logging
            hint_match_prob = None

            if suggestion_prob is not None:
                # Check if this candidate matches any suggestion term (search_term or hints)
                suggestion_terms = entity_suggestion_terms_by_idx.get(entity_idx, [])
                hint_probs_dict = entity_hint_probabilities_by_idx.get(entity_idx, {})

                if suggestion_terms:
                    pname_lower = pname.lower()
                    # Check if candidate name matches search_term first
                    search_term = suggestion_terms[0] if suggestion_terms else ""
                    if search_term and (search_term in pname_lower or pname_lower in search_term):
                        suggestion_boost = calculate_suggestion_boost(suggestion_prob)
                        hint_match_source = "search_term"
                    else:
                        # Check hints and use hint probability for weighted boost
                        best_hint_prob = 0.0
                        matched_hint = None
                        for hint_term in suggestion_terms[1:]:  # Skip search_term (first element)
                            if hint_term in pname_lower or pname_lower in hint_term:
                                # Get probability for this specific hint
                                hint_prob = hint_probs_dict.get(hint_term, 0.0)
                                if hint_prob > best_hint_prob:
                                    best_hint_prob = hint_prob
                                    matched_hint = hint_term

                        if matched_hint:
                            # Weight the boost by hint probability: higher hint prob = higher boost
                            # Base boost from suggestion_prob, scaled by hint probability
                            base_boost = calculate_suggestion_boost(suggestion_prob)
                            # Scale boost by hint probability (hint_prob 0.9 → 90% of base boost)
                            suggestion_boost = base_boost * best_hint_prob
                            hint_match_source = f"hint:{matched_hint}"
                            hint_match_prob = best_hint_prob
                            if logger:
                                logger.debug(f"Batch global: entity_idx={entity_idx} candidate '{pname}' matched hint '{matched_hint}' (prob={best_hint_prob:.2f}) → weighted boost={suggestion_boost:.3f}")
                        else:
                            # Check query_expansion terms (phonetic/ASR correction): fixed-weight boost
                            expansion_terms = entity_query_expansion_terms_by_idx.get(entity_idx, [])
                            for exp_term in expansion_terms[:3]:
                                if exp_term and (exp_term in pname_lower or pname_lower in exp_term):
                                    base_boost = calculate_suggestion_boost(suggestion_prob)
                                    qe_weight = float(os.getenv("QUERY_EXPANSION_BOOST_WEIGHT", "0.8"))
                                    suggestion_boost = base_boost * qe_weight
                                    hint_match_source = f"query_expansion:{exp_term}"
                                    break
                final_score = (
                    (tri * tri_weight)
                    + (pho * pho_weight)
                    + (aux_vector_score * vb_weight)
                    + (openai_score * oa_weight)
                    + domain_score
                    + generic_penalty
                    + relation_boost
                    + suggestion_boost
                )
            else:
                cand_domain = (c.get("domain_key") or "").strip().lower() or ""
                # If domain_key is null/empty, try to infer domain from concept name using keyword matching
                if not cand_domain and pname:
                    try:
                        from kb_ner_domain import DOMAIN_KEYWORDS
                        pname_lower = pname.lower()
                        for domain, keywords in DOMAIN_KEYWORDS.items():
                            if any(kw in pname_lower for kw in keywords):
                                cand_domain = domain
                                break
                    except Exception:
                        pass
                if cand_domain == detected_domain_norm and detected_domain_norm:
                    domain_score = GLOBAL_TWO_STAGE_DOMAIN_WEIGHT
                elif cand_domain and cand_domain not in GLOBAL_TWO_STAGE_DOMAIN_NEUTRAL and detected_domain_norm:
                    domain_score = -GLOBAL_TWO_STAGE_DOMAIN_FRICTION
                else:
                    domain_score = 0.0
                cand_kind = (c.get("kind") or "").strip()
                cand_tokens = set((pname or "").lower().split())
                is_generic_proc = (
                    cand_kind in GENERIC_PENALTY_KINDS
                    and len(cand_tokens & GENERIC_PROC_TOKENS) > 0
                )
                generic_penalty = -GLOBAL_GENERIC_PROC_PENALTY if is_generic_proc else 0.0
                is_neighbor = (c.get("concept_id") in neighbor_cids) or c.get("is_neighbor")
                relation_boost = GLOBAL_TWO_STAGE_RELATION_WEIGHT if is_neighbor else 0.0
                # Suggestion boost: apply to candidates that match search_term or hints, weighted by hint probabilities
                suggestion_prob = entity_suggestion_probability_by_idx.get(entity_idx)
                suggestion_boost = 0.0
                hint_match_source = None  # Track which hint matched for logging
                hint_match_prob = None
                
                if suggestion_prob is not None:
                    # Check if this candidate matches any suggestion term (search_term or hints)
                    suggestion_terms = entity_suggestion_terms_by_idx.get(entity_idx, [])
                    hint_probs_dict = entity_hint_probabilities_by_idx.get(entity_idx, {})
                    
                    if suggestion_terms:
                        pname_lower = pname.lower()
                        # Check if candidate name matches search_term first
                        search_term = suggestion_terms[0] if suggestion_terms else ""
                        if search_term and (search_term in pname_lower or pname_lower in search_term):
                            suggestion_boost = calculate_suggestion_boost(suggestion_prob)
                            hint_match_source = "search_term"
                        else:
                            # Check hints and use hint probability for weighted boost
                            best_hint_prob = 0.0
                            matched_hint = None
                            for hint_term in suggestion_terms[1:]:  # Skip search_term (first element)
                                if hint_term in pname_lower or pname_lower in hint_term:
                                    # Get probability for this specific hint
                                    hint_prob = hint_probs_dict.get(hint_term, 0.0)
                                    if hint_prob > best_hint_prob:
                                        best_hint_prob = hint_prob
                                        matched_hint = hint_term
                            
                            if matched_hint:
                                # Weight the boost by hint probability: higher hint prob = higher boost
                                # Base boost from suggestion_prob, scaled by hint probability
                                base_boost = calculate_suggestion_boost(suggestion_prob)
                                # Scale boost by hint probability (hint_prob 0.9 → 90% of base boost)
                                suggestion_boost = base_boost * best_hint_prob
                                hint_match_source = f"hint:{matched_hint}"
                                hint_match_prob = best_hint_prob
                                if logger:
                                    logger.debug(f"Batch global: entity_idx={entity_idx} candidate '{pname}' matched hint '{matched_hint}' (prob={best_hint_prob:.2f}) → weighted boost={suggestion_boost:.3f}")
                            else:
                                # Check query_expansion terms (phonetic/ASR correction): fixed-weight boost
                                expansion_terms = entity_query_expansion_terms_by_idx.get(entity_idx, [])
                                for exp_term in expansion_terms[:3]:
                                    if exp_term and (exp_term in pname_lower or pname_lower in exp_term):
                                        base_boost = calculate_suggestion_boost(suggestion_prob)
                                        qe_weight = float(os.getenv("QUERY_EXPANSION_BOOST_WEIGHT", "0.8"))
                                        suggestion_boost = base_boost * qe_weight
                                        hint_match_source = f"query_expansion:{exp_term}"
                                        break
                final_score = (openai_score * SOFT_GATE_BASE_WEIGHT) + domain_score + generic_penalty + (relation_boost if is_neighbor else 0.0) + suggestion_boost
            c["final_score"] = final_score
            c["match_score"] = final_score
        out[entity_idx] = sorted(candidates, key=lambda x: -float(x.get("final_score", x.get("match_score") or 0)))

    # Apply domain soft-gate per entity (use Brain domain per entity when present, else transcript domain)
    if (raw_transcript or entity_domain_by_idx) and out:
        try:
            from kb_ner_domain import detect_domain
            _tx_domain = detect_domain(raw_transcript) if raw_transcript else None
            _tx_domains_list = detect_domain(raw_transcript, return_multiple=True) if raw_transcript else []
            transcript_domains_set = set(
                (d or "").strip().lower()
                for d in (_tx_domains_list or [])
                if (d or "").strip().lower() and (d or "").strip().lower() != "general"
            )
            if not transcript_domains_set and _tx_domain and (_tx_domain or "").strip().lower() not in ("", "general"):
                transcript_domains_set = {(_tx_domain or "").strip().lower()}
            for entity_idx in list(out.keys()):
                entity_domains = entity_domain_by_idx.get(entity_idx) or []
                detected_domains_set = set(entity_domains) if entity_domains else transcript_domains_set
                if not detected_domains_set:
                    continue
                candidates = out[entity_idx]
                for c in candidates:
                    base = float(c.get("match_score") or c.get("hybrid_score") or 0.0)
                    cand_domain = (c.get("domain_key") or "").strip().lower() or ""
                    if cand_domain in detected_domains_set:
                        domain_boost, domain_penalty = SOFT_GATE_DOMAIN_BOOST, 0.0
                    elif cand_domain and cand_domain not in GLOBAL_TWO_STAGE_DOMAIN_NEUTRAL:
                        domain_boost, domain_penalty = 0.0, SOFT_GATE_DOMAIN_FRICTION
                    else:
                        domain_boost, domain_penalty = 0.0, 0.0
                    c["domain_boost"] = domain_boost
                    c["domain_penalty"] = domain_penalty
                    final = (base * SOFT_GATE_BASE_WEIGHT) + domain_boost - domain_penalty
                    c["final_score"] = max(0.0, final)
                    if c.get("hybrid_score") is None:
                        c["hybrid_score"] = c["final_score"]
                out[entity_idx] = sorted(candidates, key=lambda x: -float(x.get("final_score", x.get("match_score") or 0)))
            if logger and (entity_domain_by_idx or transcript_domains_set):
                logger.info("Batch global: applied domain soft-gate (per-entity Brain domain when present)")
        except Exception as e:
            if logger:
                logger.debug("Batch global: domain soft-gate skipped: %s", e)

    # Domain-aware phonetic boost (parity with single path): use per-entity Brain domain when present, else transcript
    # OPTIMIZED: Batch all domain embeddings and all candidate name embeddings to avoid sequential calls
    if (raw_transcript or entity_domain_by_idx) and out and client:
        try:
            from kb_domain_affinity import get_domain_embedding, cosine_similarity, DOMAIN_ANCHORS
            # Domains to consider: union of per-entity Brain domains + transcript
            all_domains = set(d for doms in entity_domain_by_idx.values() for d in doms) | (detected_domains_set_parity or set())
            domain_embeddings_ba: Dict[str, List[float]] = {}
            
            # OPTIMIZATION: Batch embed all uncached domains at once
            from kb_domain_affinity import _domain_embedding_cache, _domain_cache_lock
            domains_to_embed = []
            for d in all_domains:
                # Check cache first (get_domain_embedding checks cache internally, but we need to batch uncached ones)
                with _domain_cache_lock:
                    if d in _domain_embedding_cache:
                        domain_embeddings_ba[d] = _domain_embedding_cache[d]
                    elif d in DOMAIN_ANCHORS:
                        domains_to_embed.append((d, DOMAIN_ANCHORS[d]))
            
            # Batch embed all uncached domains in one call
            if domains_to_embed and client:
                try:
                    from kb_ner_embeddings import embed_texts
                    domain_descriptions = [desc for _, desc in domains_to_embed]
                    domain_embs_batch = embed_texts(domain_descriptions, client=client, logger=logger)
                    if domain_embs_batch and len(domain_embs_batch) == len(domains_to_embed):
                        for (d, _), emb in zip(domains_to_embed, domain_embs_batch):
                            if emb:
                                domain_embeddings_ba[d] = emb
                                # Cache it for future use
                                with _domain_cache_lock:
                                    _domain_embedding_cache[d] = emb
                except Exception as e:
                    if logger:
                        logger.debug("Batch domain embedding failed, falling back to individual calls: %s", e)
                    # Fallback: individual calls (slower but works)
                    for d, _ in domains_to_embed:
                        emb = get_domain_embedding(d, client=client, logger=logger)
                        if emb:
                            domain_embeddings_ba[d] = emb
            
            if domain_embeddings_ba:
                # OPTIMIZATION: Collect ALL candidate names from ALL entities first, then batch embed once
                all_candidate_names: Dict[int, List[str]] = {}  # entity_idx -> list of candidate names
                try:
                    max_boost = int(os.getenv("KB_DOMAIN_BOOST_MAX_CANDIDATES", "150"))
                except Exception:
                    max_boost = 150
                
                for entity_idx in list(out.keys()):
                    domains_for_entity = entity_domain_by_idx.get(entity_idx) or list(detected_domains_set_parity or [])
                    if not domains_for_entity:
                        continue
                    entity_domain_embs = {k: v for k, v in domain_embeddings_ba.items() if k in domains_for_entity}
                    if not entity_domain_embs:
                        continue
                    candidates = out[entity_idx]
                    unique_names = list(dict.fromkeys((c.get("preferred_name") or "").strip() for c in candidates if (c.get("preferred_name") or "").strip()))
                    if max_boost > 0 and len(unique_names) > max_boost:
                        unique_names = unique_names[:max_boost]
                    if unique_names:
                        all_candidate_names[entity_idx] = unique_names
                
                # Batch embed ALL candidate names from ALL entities in ONE call
                all_names_flat: List[str] = []
                entity_idx_to_name_indices: Dict[int, Tuple[int, int]] = {}  # entity_idx -> (start_idx, end_idx)
                start_idx = 0
                for entity_idx, names in all_candidate_names.items():
                    all_names_flat.extend(names)
                    end_idx = start_idx + len(names)
                    entity_idx_to_name_indices[entity_idx] = (start_idx, end_idx)
                    start_idx = end_idx
                
                # Single batch embedding call for all candidate names
                all_name_embeddings: Dict[str, List[float]] = {}
                if all_names_flat:
                    try:
                        from kb_ner_embeddings import embed_texts
                        batch_emb = embed_texts(all_names_flat, client=client, logger=logger)
                        if batch_emb and len(batch_emb) == len(all_names_flat):
                            for name, vec in zip(all_names_flat, batch_emb):
                                if vec:
                                    all_name_embeddings[name] = vec
                    except Exception as e:
                        if logger:
                            logger.debug("Batch candidate name embedding failed: %s", e)
                
                # Now process each entity using the pre-computed embeddings
                for entity_idx in list(out.keys()):
                    # Per-entity domain: Brain first, else transcript
                    domains_for_entity = entity_domain_by_idx.get(entity_idx) or list(detected_domains_set_parity or [])
                    if not domains_for_entity:
                        continue
                    entity_domain_embs = {k: v for k, v in domain_embeddings_ba.items() if k in domains_for_entity}
                    if not entity_domain_embs:
                        continue
                    candidates = out[entity_idx]
                    # Get pre-computed embeddings for this entity's candidates
                    name_to_emb: Dict[str, List[float]] = {}
                    if entity_idx in all_candidate_names:
                        entity_names = all_candidate_names[entity_idx]
                        for name in entity_names:
                            if name in all_name_embeddings:
                                name_to_emb[name] = all_name_embeddings[name]
                    for cand in candidates:
                        phonetic_score = cand.get("phonetic_score", 0) or 0
                        trigram = cand.get("trigram_score", 0) or 0
                        vector = cand.get("openai_score", 0) or 0
                        candidate_name = (cand.get("preferred_name", "") or "").strip()
                        cand_emb = name_to_emb.get(candidate_name) if candidate_name else None
                        is_domain_relevant = False
                        if cand_emb and entity_domain_embs:
                            for _d, d_emb in entity_domain_embs.items():
                                if cosine_similarity(cand_emb, d_emb) >= 0.85:
                                    is_domain_relevant = True
                                    break
                        base_signal = max(trigram, phonetic_score, vector)
                        if is_domain_relevant and base_signal > 0.15:
                            if trigram < 0.1 and 0.15 <= phonetic_score <= 0.4:
                                boosted_phonetic = max(0.80, phonetic_score * 3.5) if phonetic_score < 0.25 else max(0.95, phonetic_score * 2.5)
                            elif 0.15 <= phonetic_score <= 0.4:
                                boosted_phonetic = min(1.0, phonetic_score * 2.0)
                            else:
                                boosted_phonetic = min(1.0, phonetic_score * 1.5)
                            original_match = cand.get("match_score", 0) or 0
                            boosted_match = max(trigram, boosted_phonetic, vector)
                            if boosted_match > original_match:
                                cand["phonetic_score"] = boosted_phonetic
                                cand["match_score"] = boosted_match
                                cand["final_score"] = boosted_match
                                cand["match_source"] = (cand.get("match_source") or "batch") + "_domain_boosted"
                                if logger:
                                    logger.info("Batch global: domain-aware phonetic boost applied for entity_idx=%s candidate=%s", entity_idx, candidate_name)
                    for entity_idx in list(out.keys()):
                        out[entity_idx] = sorted(out[entity_idx], key=lambda x: -float(x.get("final_score", x.get("match_score") or 0)))
            if logger and domain_embeddings_ba:
                logger.info("Batch global: domain-aware phonetic boost pass completed (domains=%s)", list(domain_embeddings_ba.keys()))
        except Exception as e:
            if logger:
                logger.debug("Batch global: domain-aware phonetic boost skipped: %s", e)

    # Semantic/RAG fallback policy:
    # - Gate expensive fallbacks to unresolved HIGH-STAKES entities only (default enabled).
    # - Keep fallbacks batched (embed all weak mentions once; embed all RAG queries once).
    semantic_fallback_entity_ids: set = set()
    HIGH_STAKES_FALLBACK_KINDS = {
        "procedure", "service", "treatment",
        "medication", "medicine", "drug", "substance", "vaccine",
        "diagnostic", "diagnostictest", "labtest",
    }

    def _is_high_stakes_kind_filter(kf: Optional[List[str]]) -> bool:
        if not kf:
            return False
        lowered = {str(x).strip().lower() for x in (kf or []) if str(x).strip()}
        return bool(lowered & HIGH_STAKES_FALLBACK_KINDS)

    fallback_high_stakes_only = os.getenv("BATCH_FALLBACK_HIGH_STAKES_ONLY", "true").strip().lower() in ("1", "true", "yes")

    try:
        semantic_thresh = float(os.getenv("BATCH_SEMANTIC_FALLBACK_THRESHOLD", "0.40"))
    except Exception:
        semantic_thresh = 0.45

    weak_items: List[Tuple[int, str, Optional[List[str]]]] = []
    for entity_idx in list(out.keys()):
        candidates = out[entity_idx]
        best = max((float(c.get("final_score", c.get("match_score") or 0)) for c in candidates), default=0.0)
        if best >= semantic_thresh:
            continue
        mention = entity_to_mention.get(entity_idx) or ""
        if not mention:
            continue
        kf = entity_to_kind_filter.get(entity_idx)
        if fallback_high_stakes_only and not _is_high_stakes_kind_filter(kf):
            continue
        weak_items.append((entity_idx, mention, kf))

    if weak_items and client:
        try:
            from kb_ner_embeddings import embed_texts
            weak_mentions = [m for _, m, _ in weak_items]
            weak_embs = embed_texts(weak_mentions, client=client, logger=logger)
            if not weak_embs or len(weak_embs) != len(weak_mentions):
                weak_embs = [None] * len(weak_mentions)
        except Exception as e:
            if logger:
                logger.debug("Batch semantic fallback embed_texts failed: %s", e)
            weak_embs = [None] * len(weak_items)
    else:
        weak_embs = [None] * len(weak_items)

    for i, (entity_idx, mention, kf) in enumerate(weak_items):
        emb = weak_embs[i] if i < len(weak_embs) else None
        try:
            extra = kb_lookup_concept_by_embedding(
                conn,
                mention,
                top_k=50,
                client=client,
                logger=logger,
                kind_filter=kf,
                embedding=emb,
            )
        except Exception:
            extra = []
        if not extra:
            continue
        semantic_fallback_entity_ids.add(entity_idx)
        candidates = out.get(entity_idx) or []
        best = max((float(c.get("final_score", c.get("match_score") or 0)) for c in candidates), default=0.0)
        if logger:
            logger.info(
                "Batch global: semantic fallback for entity_idx=%s (mention=%r, best=%.2f < %.2f, added %s candidates)",
                entity_idx, mention[:50], best, semantic_thresh, len(extra)
            )
        by_cid = {c.get("concept_id"): c for c in candidates if c.get("concept_id") is not None}
        for c in extra:
            cid = c.get("concept_id")
            if cid is None or cid in by_cid:
                continue
            by_cid[cid] = dict(c)
            by_cid[cid]["match_source"] = (c.get("match_source") or "embedding") + "_semantic_fallback"
        out[entity_idx] = sorted(by_cid.values(), key=lambda x: -float(x.get("final_score", x.get("match_score") or 0)))[: max(top_k_per_mention, 10)]

    # RAG fallback: for very weak entities OR semantic-fallback entities, but (by default) only high-stakes kinds.
    try:
        rag_thresh = float(os.getenv("BATCH_RAG_FALLBACK_THRESHOLD", "0.25"))
    except Exception:
        rag_thresh = 0.30

    rag_items: List[Tuple[int, str, Optional[List[str]], float]] = []
    for entity_idx in list(out.keys()):
        candidates = out[entity_idx]
        best = max((float(c.get("final_score", c.get("match_score") or 0)) for c in candidates), default=0.0)
        # Guardrail: do not always run RAG just because semantic fallback ran once.
        # RAG should run only when still weak after semantic augmentation.
        run_rag = best < rag_thresh or (entity_idx in semantic_fallback_entity_ids and best < semantic_thresh)
        if not run_rag:
            continue
        mention = entity_to_mention.get(entity_idx) or ""
        if not mention:
            continue
        kf = entity_to_kind_filter.get(entity_idx)
        if fallback_high_stakes_only and not _is_high_stakes_kind_filter(kf):
            continue
        rag_items.append((entity_idx, mention, kf, best))

    # Batch-embed RAG queries once to avoid per-entity single-input embed calls
    if rag_items and client:
        try:
            from kb_ner_embeddings import embed_texts
            rag_queries = [(f"{mention} {raw_transcript}" if raw_transcript else mention) for _, mention, _, _ in rag_items]
            rag_embs = embed_texts(rag_queries, client=client, logger=logger)
            if not rag_embs or len(rag_embs) != len(rag_queries):
                rag_embs = [None] * len(rag_items)
        except Exception as e:
            if logger:
                logger.debug("Batch RAG fallback embed_texts failed: %s", e)
            rag_embs = [None] * len(rag_items)
    else:
        rag_embs = [None] * len(rag_items)

    for i, (entity_idx, mention, kf, best) in enumerate(rag_items):
        emb = rag_embs[i] if i < len(rag_embs) else None
        try:
            extra = kb_lookup_concept_by_embedding(
                conn,
                mention,
                top_k=20,
                context=raw_transcript,
                client=client,
                logger=logger,
                use_rag=True,
                kind_filter=kf,
                embedding=emb,
            )
        except Exception:
            extra = []
        if not extra:
            continue
        if logger:
            logger.info(
                "Batch global: RAG fallback for entity_idx=%s (mention=%r, best=%.2f, added %s candidates)%s",
                entity_idx, mention[:50], best, len(extra),
                " (semantic_fallback_entity)" if entity_idx in semantic_fallback_entity_ids else "",
            )
        candidates = out.get(entity_idx) or []
        by_cid = {c.get("concept_id"): c for c in candidates if c.get("concept_id") is not None}
        for c in extra:
            cid = c.get("concept_id")
            if cid is None or cid in by_cid:
                continue
            by_cid[cid] = dict(c)
            by_cid[cid]["match_source"] = (c.get("match_source") or "rag") + "_rag_fallback"
        out[entity_idx] = sorted(by_cid.values(), key=lambda x: -float(x.get("final_score", x.get("match_score") or 0)))[: max(top_k_per_mention, 10)]

    return out


def kb_lookup_concept_exact(
    conn,
    normalized_name: str,
) -> List[Dict[str, Any]]:
    """
    First-pass concept lookup by exact case-insensitive match on kb.concepts / kb.concept_aliases.
    When LOCAL_ONLY is true (default), returns [] and does not query kb.concepts.
    """
    if LOCAL_ONLY:
        return []
    sql = """
        SELECT DISTINCT 
            c.concept_id, 
            c.preferred_name, 
            c.kind, 
            COALESCE(c.venom_id, '') AS venom_code,
            COALESCE(c.snomed_id, '') AS snomed_code,
            COALESCE(c.domain_key, '') AS domain_key,
            'name_or_alias' AS source
        FROM kb.concepts c
        LEFT JOIN kb.concept_aliases a
          ON a.concept_id = c.concept_id
        WHERE lower(c.preferred_name) = lower(%s)
           OR lower(a.alias_text) = lower(%s)
        LIMIT 10;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (normalized_name, normalized_name))
            rows = cur.fetchall()

        results = []
        for concept_id, preferred_name, kind, venom_code, snomed_code, domain_key, source in rows:
            results.append(
                {
                    "concept_id": concept_id,
                    "preferred_name": preferred_name,
                    "kind": kind,
                    "venom_code": venom_code if venom_code else None,
                    "snomed_code": snomed_code if snomed_code else None,
                    "domain_key": (domain_key or "").strip() or None,
                    "match_source": source,
                }
            )
        return results
    except Exception as e:
        logging.warning(f"Error in exact lookup: {e}")
        return []


def kb_lookup_concept_by_trgm(
    conn,
    query_text: str,
    top_k: int = 15,
    kind_filter: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    """Trigram search on kb.concepts. When LOCAL_ONLY is true (default), returns []."""
    if LOCAL_ONLY:
        return []
    # Normalize kind_filter to SQL clause
    kind_clause = ""
    params: List[Any] = [query_text, query_text, query_text, query_text]
    if kind_filter:
        if isinstance(kind_filter, list):
            kind_clause = " AND c.kind = ANY(%s)"
            params.append(kind_filter)
        else:
            kind_clause = " AND c.kind = %s"
            params.append(kind_filter)

    sql = f"""
        WITH candidates AS (
            SELECT
                c.concept_id,
                c.preferred_name,
                c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.domain_key, '') AS domain_key,
                GREATEST(similarity(lower(c.preferred_name), lower(%s)),
                         similarity(lower(COALESCE(a.alias_text,'')), lower(%s))) AS sim
            FROM kb.concepts c
            LEFT JOIN kb.concept_aliases a ON a.concept_id = c.concept_id
            WHERE (
                similarity(lower(c.preferred_name), lower(%s)) >= {threshold}
                OR similarity(lower(COALESCE(a.alias_text,'')), lower(%s)) >= {threshold}
            )
            {kind_clause}
        )
        SELECT concept_id, preferred_name, kind, definition, domain_key, sim
        FROM candidates
        ORDER BY sim DESC
        LIMIT {top_k};
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            {
                "concept_id": r[0],
                "preferred_name": r[1],
                "kind": r[2],
                "definition": r[3] if len(r) > 3 and r[3] else None,
                "domain_key": (r[4] if len(r) > 4 and r[4] else "").strip() or None,
                "match_source": "pg_trgm",
                "similarity_score": float(r[5]) if len(r) > 5 else (float(r[4]) if len(r) > 4 else 0.0),
            }
            for r in rows
            if len(r) >= 5  # Ensure we have at least 5 columns (concept_id, preferred_name, kind, definition, domain_key, sim)
        ]
    except Exception as e:
        if logger:
            logger.debug(f"pg_trgm lookup failed: {e}")
        return []


def kb_best_similarity_score(
    conn,
    text: str,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Quick trigram similarity check against KB (concepts + aliases).
    Returns the best similarity score in [0, 1], or 0.0 if no match.
    Used by post-extraction KB whitelist filter to discard noise (e.g. "lazy" with score < 0.2).
    """
    if not text or not text.strip():
        return 0.0
    try:
        candidates = kb_lookup_concept_by_trgm(
            conn, text.strip(), top_k=1, kind_filter=None, logger=logger, threshold=0.01
        )
        if not candidates:
            return 0.0
        return float(candidates[0].get("similarity_score", 0.0))
    except Exception as e:
        if logger:
            logger.debug("kb_best_similarity_score failed: %s", e)
        return 0.0


def filter_entities_by_kb_whitelist(
    entities: List[Dict[str, Any]],
    conn,
    threshold: float = 0.2,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Post-extraction filter (dictionary gating): keep only entities whose span_text
    has trigram similarity >= threshold against the clinical KB. Discards generic
    verbs, common adjectives (e.g. "lazy"), and conversational filler that do not
    map to KB concepts.
    """
    if not entities or threshold <= 0.0:
        return entities
    kept = []
    for entity in entities:
        span = (entity.get("span_text") or entity.get("normalized_name") or "").strip()
        if not span:
            kept.append(entity)
            continue
        score = kb_best_similarity_score(conn, span, logger=logger)
        if score >= threshold:
            kept.append(entity)
        elif logger:
            logger.debug("KB whitelist filter: dropped '%s' (score=%.3f < %.2f)", span, score, threshold)
    if logger and len(kept) < len(entities):
        logger.info(
            "KB whitelist filter: kept %s / %s entities (threshold=%.2f)",
            len(kept), len(entities), threshold,
        )
    return kept


def filter_atoms_by_kb_whitelist(
    atoms: List[Dict[str, Any]],
    conn,
    threshold: float = 0.2,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Post-extraction filter for knowledge atoms. When LOCAL_ONLY is true (default),
    returns atoms unchanged (no kb.concepts lookup).
    """
    if LOCAL_ONLY:
        return atoms
    if not atoms or threshold <= 0.0:
        return atoms
    kept = []
    for atom in atoms:
        concept = (atom.get("concept") or atom.get("span_text") or "").strip()
        if not concept:
            kept.append(atom)
            continue
        score = kb_best_similarity_score(conn, concept, logger=logger)
        if score >= threshold:
            kept.append(atom)
        elif logger:
            logger.debug("KB whitelist filter (atoms): dropped '%s' (score=%.3f < %.2f)", concept, score, threshold)
    if logger and len(kept) < len(atoms):
        logger.info(
            "KB whitelist filter (atoms): kept %s / %s (threshold=%.2f)",
            len(kept), len(atoms), threshold,
        )
    return kept


def kb_lookup_concept_by_rag(
    conn,
    normalized_name: str,
    top_k: int = 15,
    context: Optional[str] = None,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    embedding: Optional[List[float]] = None,
    embedding_cache: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """RAG retrieval from kb.kb_text_embeddings → kb.concepts. When LOCAL_ONLY is true (default), returns []."""
    if LOCAL_ONLY:
        return []
    # Build context-aware query if context is provided
    if context:
        query = f"{normalized_name} {context}"
        if logger:
            logger.debug(f"Using RAG retrieval for '{normalized_name}' with context: {context[:100]}...")
    else:
        query = normalized_name
    
    if embedding:
        emb = embedding
    elif embedding_cache is not None and query in embedding_cache:
        emb = embedding_cache.get(query)
    else:
        emb = embed_text(query, client=client, logger=logger)
        if embedding_cache is not None and emb:
            embedding_cache[query] = emb
    if not emb:
        return []

    try:
        vec_literal = to_pgvector_literal(emb)
    except (ValueError, TypeError) as e:
        if logger:
            logger.error(f"Failed to convert embedding to pgvector literal: {e}")
        return []

    # Step 1: Query RAG chunks (kb.kb_text_embeddings)
    rag_top_k = top_k * 3  # Get 3x chunks to ensure we have enough concepts
    
    ef_search = 64  # Optimize for accuracy
    
    sql_rag = f"""
        SET LOCAL hnsw.ef_search = {ef_search};
        SELECT kb_text_id, concept_id, text_chunk,
               (embedding <-> %s::vector) AS distance
        FROM kb.kb_text_embeddings
        WHERE embedding IS NOT NULL
        ORDER BY embedding <-> %s::vector
        LIMIT {rag_top_k};
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql_rag, (vec_literal, vec_literal))
            rag_chunks = cur.fetchall()
        
        # Step 2: Extract concept_ids from relevant chunks
        concept_ids = []
        seen_concept_ids = set()
        chunk_distances = {}
        
        for kb_text_id, concept_id, text_chunk, distance in rag_chunks:
            if concept_id and concept_id not in seen_concept_ids:
                concept_ids.append(concept_id)
                seen_concept_ids.add(concept_id)
                chunk_distances[concept_id] = float(distance)
                if len(concept_ids) >= top_k:
                    break
        
        if not concept_ids:
            if logger:
                logger.debug(f"No concepts found in RAG chunks for '{normalized_name}'")
            return []
        
        # Step 3: Get concept details from kb.concepts
        placeholders = ','.join(['%s'] * len(concept_ids))
        sql_concepts = f"""
            SELECT 
                concept_id, 
                preferred_name, 
                kind,
                COALESCE(venom_id, '') AS venom_code,
                COALESCE(snomed_id, '') AS snomed_code
            FROM kb.concepts
            WHERE concept_id IN ({placeholders});
        """
        
        with conn.cursor() as cur:
            cur.execute(sql_concepts, concept_ids)
            concept_rows = cur.fetchall()
        
        # Step 4: Build results with RAG distance scores
        results = []
        for concept_id, preferred_name, kind, venom_code, snomed_code in concept_rows:
            results.append(
                {
                    "concept_id": concept_id,
                    "preferred_name": preferred_name,
                    "kind": kind,
                    "venom_code": venom_code if venom_code else None,
                    "snomed_code": snomed_code if snomed_code else None,
                    "match_source": "rag",
                    "similarity_score": 1.0 - chunk_distances.get(concept_id, 1.0),  # Convert distance to similarity
                }
            )
        
        return results
    except Exception as e:
        if logger:
            logger.debug(f"RAG lookup failed: {e}")
        return []


def _run_single_source_query(
    cur,
    query_sql: str,
    params: tuple,
    source_name: str,
    normalized_name: str,
    logger: Optional[logging.Logger],
) -> list:
    """Run one single-source query (no UNION). Returns list of rows or [] on any error.
    Ensures no single query can cause a global failure (e.g. psycopg2 IndexError)."""
    try:
        cur.execute(query_sql, params)
        return cur.fetchall()
    except Exception as e:
        if logger:
            logger.debug(f"   Single-source query '{source_name}' failed for '{normalized_name}': {type(e).__name__}")
        return []


def _merge_dedupe_topk(rows_list: list, top_k: int) -> list:
    """Merge rows from multiple queries, dedupe by concept_id (keep best match_score), sort, return top_k.
    Each row is a tuple with at least (concept_id, ..., match_score). match_score is at index 9 (0-based)."""
    if not rows_list:
        return []
    by_id = {}
    for row in rows_list:
        if len(row) < 10:
            continue
        cid = row[0]
        match_score = float(row[9]) if row[9] is not None else 0.0
        if cid not in by_id or (float(by_id[cid][9]) if by_id[cid][9] is not None else 0.0) < match_score:
            by_id[cid] = row
    merged = sorted(by_id.values(), key=lambda r: (float(r[9]) if r[9] is not None else 0.0), reverse=True)
    return merged[:top_k]


def kb_lookup_concept_by_embedding(
    conn,
    normalized_name: str,
    top_k: int = 15,
    context: Optional[str] = None,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    use_rag: bool = False,
    kind_filter: Optional[Any] = None,
    embedding: Optional[List[float]] = None,
    embedding_cache: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """Concept lookup by embedding. When LOCAL_ONLY is true (default), returns []."""
    if LOCAL_ONLY:
        return []
    # Use RAG retrieval only if explicitly requested
    if use_rag:
        return kb_lookup_concept_by_rag(
            conn, normalized_name, top_k, context, client, logger, embedding=embedding,
            embedding_cache=embedding_cache,
        )
    
    # Build context-aware query with domain anchor injection (scalable fix)
    # If raw_transcript is provided, detect domain and inject as anchor (e.g., "Orthopedic: noble angle")
    # This forces the search to look in the correct clinical neighborhood without hardcoding rules
    domain_anchor = None
    if context and isinstance(context, str) and len(context) > 100:
        # If context is full transcript, detect domain
        try:
            from kb_ner_domain import detect_domain
            detected_domain = detect_domain(context)
            if detected_domain and detected_domain != 'general':
                domain_anchor = detected_domain
        except Exception:
            pass
    
    # Query for embedding: use mention (optionally with context). Domain used only for soft-gate boost post-search.
    if context and not domain_anchor:
        query = f"{normalized_name} {context}"
        if logger:
            logger.debug(f"Using context-aware embedding for '{normalized_name}' with context: {context[:100]}...")
    else:
        query = normalized_name
    if domain_anchor and logger:
        logger.debug(f"Domain anchor for soft-gate only (embedding uses mention): '%s' (domain: %s)", normalized_name[:40], domain_anchor)

        if embedding:
            emb = embedding
        elif embedding_cache is not None and query in embedding_cache:
            emb = embedding_cache.get(query)
        else:
            emb = embed_text(query, client=client, logger=logger)
            if embedding_cache is not None and emb:
                embedding_cache[query] = emb

    if not emb:
        if logger:
            logger.warning(f"Failed to generate/get embedding for '{normalized_name}'")
        return []

    # Validate embedding
    if not isinstance(emb, (list, tuple)) or len(emb) == 0:
        if logger:
            logger.warning(f"Invalid embedding format for '{normalized_name}': {type(emb)}")
        return []

    try:
        vec_literal = to_pgvector_literal(emb)
        # Validate vec_literal is not None or empty
        if not vec_literal or not isinstance(vec_literal, str) or len(vec_literal.strip()) == 0:
            if logger:
                logger.warning(f"Invalid vec_literal for '{normalized_name}': {vec_literal}")
            return []
    except (ValueError, TypeError) as e:
        if logger:
            logger.error(f"Failed to convert embedding to pgvector literal for '{normalized_name}': {e}")
        return []

    # Clean the Kind Filter
    if not kind_filter:
        kind_filter_list = ['Drug', 'Procedure', 'Finding', 'Condition', 'Anatomy', 'Observation']
    elif isinstance(kind_filter, str):
        kind_filter_list = [kind_filter]
    elif isinstance(kind_filter, list):
        kind_filter_list = kind_filter
    else:
        kind_filter_list = ['Drug', 'Procedure', 'Finding', 'Condition', 'Anatomy', 'Observation']
    
    # Check if pg_trgm and fuzzystrmatch extensions are available
    has_pg_trgm = False
    has_fuzzystrmatch = False
    try:
        with conn.cursor() as cur_check:
            cur_check.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
            has_pg_trgm = cur_check.fetchone()[0]
            cur_check.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'fuzzystrmatch')")
            has_fuzzystrmatch = cur_check.fetchone()[0]
    except Exception:
        pass
    
    # Ensure extensions are enabled
    if has_pg_trgm is False:
        ensure_pg_trgm(conn, logger=logger)
        has_pg_trgm = True
    if has_fuzzystrmatch is False:
        ensure_fuzzystrmatch(conn, logger=logger)
        has_fuzzystrmatch = True
    
    ef_search = 64
    use_separate_queries_merge = False
    single_source_queries = []

    if has_pg_trgm:
        # Three-method search: TRIGRAM + PHONETIC + VECTOR (matching local search logic)
        # Use a robust query structure that avoids UNION ALL complexity
        # This prevents psycopg2 IndexError when parsing PostgreSQL error responses
        if has_fuzzystrmatch:
                # Full three-method search with phonetic matching
                # UNION ALL structure with concepts, aliases, and learned_aliases
                # Use a robust query structure that handles psycopg2 error parsing better
                # SET LOCAL must be executed separately (cannot be in same statement as WITH CTE)
                query_sql = """
            WITH q AS (
                SELECT
                    lower(%s) AS q_full,
                    split_part(lower(%s), ' ', 1) AS q_first,
                    metaphone(lower(%s), 10) AS q_mfull,
                    metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst
            ),
            all_candidates AS (
                -- Concepts (always available)
                SELECT 
                    c.concept_id,
                    c.preferred_name, 
                    c.kind,
                    COALESCE(c.definition, '') AS definition,
                    COALESCE(c.venom_id, '') AS venom_code,
                    COALESCE(c.snomed_id, '') AS snomed_code,
                    c.embedding,
                    'concept' AS source
                FROM kb.concepts c
                WHERE c.embedding IS NOT NULL
                  AND c.kind = ANY(%s)
                  AND (c.status IS NULL OR c.status != 'REJECTED')
                
                UNION ALL
                
                -- Concept aliases
                SELECT 
                    ca.concept_id,
                    ca.alias_text AS preferred_name,
                    c.kind,
                    COALESCE(c.definition, '') AS definition,
                    COALESCE(c.venom_id, '') AS venom_code,
                    COALESCE(c.snomed_id, '') AS snomed_code,
                    ca.embedding,
                    'alias' AS source
                FROM kb.concept_aliases ca
                INNER JOIN kb.concepts c ON c.concept_id = ca.concept_id
                WHERE ca.embedding IS NOT NULL
                  AND c.kind = ANY(%s)
                  AND (c.status IS NULL OR c.status != 'REJECTED')
                
                UNION ALL
                
                -- Learned aliases
                SELECT 
                    la.kb_concept_id AS concept_id,
                    la.alias_text AS preferred_name,
                    c.kind,
                    COALESCE(c.definition, '') AS definition,
                    COALESCE(c.venom_id, '') AS venom_code,
                    COALESCE(c.snomed_id, '') AS snomed_code,
                    la.embedding,
                    'learned' AS source
                FROM kb.learned_aliases la
                INNER JOIN kb.concepts c ON c.concept_id = la.kb_concept_id
                WHERE la.embedding IS NOT NULL 
                  AND (la.is_validated = TRUE OR la.frequency_count > 0)
                  AND c.kind = ANY(%s)
                  AND (c.status IS NULL OR c.status != 'REJECTED')
            ),
            ranked AS (
                SELECT DISTINCT ON (concept_id)
                    concept_id,
                    preferred_name,
                    kind,
                    definition,
                    venom_code,
                    snomed_code,
                    -- Individual scores (matching local search)
                    similarity(lower(preferred_name), q.q_full) AS trigram_score,
                    GREATEST(
                        0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(preferred_name), ' ', 1), 10), q.q_mfirst)::float
                                     / GREATEST(length(metaphone(split_part(lower(preferred_name), ' ', 1), 10)), length(q.q_mfirst), 1))),
                        0.8 * (1.0 - (levenshtein(metaphone(lower(preferred_name), 10), q.q_mfull)::float
                                     / GREATEST(length(metaphone(lower(preferred_name), 10)), length(q.q_mfull), 1)))
                    ) AS phonetic_score,
                    CASE 
                        WHEN embedding IS NOT NULL 
                        THEN (1.0 - LEAST(embedding <=> %s::vector, 1.0))
                        ELSE 0.0
                    END AS vector_score,
                    -- Match score: max(trigram, phonetic, vector) - matching local search logic
                    GREATEST(
                        similarity(lower(preferred_name), q.q_full),
                        GREATEST(
                            0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(preferred_name), ' ', 1), 10), q.q_mfirst)::float
                                         / GREATEST(length(metaphone(split_part(lower(preferred_name), ' ', 1), 10)), length(q.q_mfirst), 1))),
                            0.8 * (1.0 - (levenshtein(metaphone(lower(preferred_name), 10), q.q_mfull)::float
                                         / GREATEST(length(metaphone(lower(preferred_name), 10)), length(q.q_mfull), 1)))
                        ),
                        CASE 
                            WHEN embedding IS NOT NULL 
                            THEN (1.0 - LEAST(embedding <=> %s::vector, 1.0))
                            ELSE 0.0
                        END
                    ) AS match_score,
                    embedding <=> %s::vector AS distance,
                    source AS match_source
                FROM all_candidates
                CROSS JOIN q
                WHERE (
                    -- Filter: must match at least one method (trigram, phonetic, or vector)
                    similarity(lower(preferred_name), q.q_full) >= 0.30
                    OR metaphone(split_part(lower(preferred_name), ' ', 1), 10) = q.q_mfirst
                    OR metaphone(lower(preferred_name), 10) = q.q_mfull
                    OR (
                        levenshtein(metaphone(split_part(lower(preferred_name), ' ', 1), 10), q.q_mfirst)::float
                        / GREATEST(length(metaphone(split_part(lower(preferred_name), ' ', 1), 10)), length(q.q_mfirst), 1) < 0.65
                        OR levenshtein(metaphone(lower(preferred_name), 10), q.q_mfull)::float
                        / GREATEST(length(metaphone(lower(preferred_name), 10)), length(q.q_mfull), 1) < 0.65
                    )
                    OR (embedding IS NOT NULL AND embedding <=> %s::vector < 0.5)
                )
                ORDER BY concept_id, match_score DESC
            )
            SELECT * FROM ranked
            ORDER BY match_score DESC
            LIMIT %s
            """
                # Params order: WITH q (4), kind filters (3), vector literals (4), LIMIT (1)
                # Total: 12 params
                # Validate vec_literal before using it
                if not vec_literal or not isinstance(vec_literal, str):
                    if logger:
                        logger.error(f"Invalid vec_literal for '{normalized_name}': {vec_literal} (type: {type(vec_literal)})")
                    return []
                params = (
                normalized_name, normalized_name, normalized_name, normalized_name,  # q CTE (4)
                kind_filter_list, kind_filter_list, kind_filter_list,  # all_candidates kind filters (3)
                vec_literal,  # vector_score
                vec_literal,  # match_score vector
                vec_literal,  # distance
                vec_literal,  # WHERE vector
                top_k,  # LIMIT
                )

                # Fallback path: if the single UNION fails (e.g. IndexError), run three separate
                # single-source queries and merge in Python. Use UNION first to avoid added latency.
                use_separate_queries_merge = False  # Try single UNION first; use 3-query merge only on failure
                _sql_concepts = """
            WITH q AS (
                SELECT lower(%s) AS q_full, split_part(lower(%s), ' ', 1) AS q_first,
                metaphone(lower(%s), 10) AS q_mfull, metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst
            )
            SELECT c.concept_id, c.preferred_name, c.kind, COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                COALESCE(c.domain_key, '') AS domain_key,
                similarity(lower(c.preferred_name), q.q_full) AS trigram_score,
                GREATEST(0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10), q.q_mfirst)::float / GREATEST(length(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10)), length(q.q_mfirst), 1))),
                    0.8 * (1.0 - (levenshtein(metaphone(lower(c.preferred_name), 10), q.q_mfull)::float / GREATEST(length(metaphone(lower(c.preferred_name), 10)), length(q.q_mfull), 1)))) AS phonetic_score,
                (1.0 - LEAST(c.embedding <=> %s::vector, 1.0)) AS vector_score,
                GREATEST(similarity(lower(c.preferred_name), q.q_full), GREATEST(0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10), q.q_mfirst)::float / GREATEST(length(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10)), length(q.q_mfirst), 1))), 0.8 * (1.0 - (levenshtein(metaphone(lower(c.preferred_name), 10), q.q_mfull)::float / GREATEST(length(metaphone(lower(c.preferred_name), 10)), length(q.q_mfull), 1)))), (1.0 - LEAST(c.embedding <=> %s::vector, 1.0)))) AS match_score,
                c.embedding <=> %s::vector AS distance, 'concept' AS match_source
            FROM kb.concepts c CROSS JOIN q
            WHERE c.embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED')
              AND (similarity(lower(c.preferred_name), q.q_full) >= 0.30 OR metaphone(split_part(lower(c.preferred_name), ' ', 1), 10) = q.q_mfirst OR metaphone(lower(c.preferred_name), 10) = q.q_mfull
                   OR (levenshtein(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10), q.q_mfirst)::float / GREATEST(length(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10)), length(q.q_mfirst), 1) < 0.65
                       OR levenshtein(metaphone(lower(c.preferred_name), 10), q.q_mfull)::float / GREATEST(length(metaphone(lower(c.preferred_name), 10)), length(q.q_mfull), 1) < 0.65)
                   OR (c.embedding IS NOT NULL AND c.embedding <=> %s::vector < 0.5))
            ORDER BY match_score DESC LIMIT %s
            """
                _params_concepts = (normalized_name, normalized_name, normalized_name, normalized_name, vec_literal, vec_literal, vec_literal, kind_filter_list, vec_literal, top_k)
                _sql_aliases = """
            WITH q AS (
                SELECT lower(%s) AS q_full, split_part(lower(%s), ' ', 1) AS q_first,
                metaphone(lower(%s), 10) AS q_mfull, metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst
            )
            SELECT ca.concept_id, ca.alias_text AS preferred_name, c.kind, COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                COALESCE(c.domain_key, '') AS domain_key,
                similarity(lower(ca.alias_text), q.q_full) AS trigram_score,
                GREATEST(0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(ca.alias_text), ' ', 1), 10), q.q_mfirst)::float / GREATEST(length(metaphone(split_part(lower(ca.alias_text), ' ', 1), 10)), length(q.q_mfirst), 1))),
                    0.8 * (1.0 - (levenshtein(metaphone(lower(ca.alias_text), 10), q.q_mfull)::float / GREATEST(length(metaphone(lower(ca.alias_text), 10)), length(q.q_mfull), 1)))) AS phonetic_score,
                (1.0 - LEAST(ca.embedding <=> %s::vector, 1.0)) AS vector_score,
                GREATEST(similarity(lower(ca.alias_text), q.q_full), GREATEST(0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(ca.alias_text), ' ', 1), 10), q.q_mfirst)::float / GREATEST(length(metaphone(split_part(lower(ca.alias_text), ' ', 1), 10)), length(q.q_mfirst), 1))), 0.8 * (1.0 - (levenshtein(metaphone(lower(ca.alias_text), 10), q.q_mfull)::float / GREATEST(length(metaphone(lower(ca.alias_text), 10)), length(q.q_mfull), 1)))), (1.0 - LEAST(ca.embedding <=> %s::vector, 1.0)))) AS match_score,
                ca.embedding <=> %s::vector AS distance, 'alias' AS match_source
            FROM kb.concept_aliases ca INNER JOIN kb.concepts c ON c.concept_id = ca.concept_id CROSS JOIN q
            WHERE ca.embedding IS NOT NULL AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED')
              AND (similarity(lower(ca.alias_text), q.q_full) >= 0.30 OR metaphone(split_part(lower(ca.alias_text), ' ', 1), 10) = q.q_mfirst OR metaphone(lower(ca.alias_text), 10) = q.q_mfull
                   OR (levenshtein(metaphone(split_part(lower(ca.alias_text), ' ', 1), 10), q.q_mfirst)::float / GREATEST(length(metaphone(split_part(lower(ca.alias_text), ' ', 1), 10)), length(q.q_mfirst), 1) < 0.65
                       OR levenshtein(metaphone(lower(ca.alias_text), 10), q.q_mfull)::float / GREATEST(length(metaphone(lower(ca.alias_text), 10)), length(q.q_mfull), 1) < 0.65)
                   OR (ca.embedding IS NOT NULL AND ca.embedding <=> %s::vector < 0.5))
            ORDER BY match_score DESC LIMIT %s
            """
                _params_aliases = (normalized_name, normalized_name, normalized_name, normalized_name, vec_literal, vec_literal, vec_literal, kind_filter_list, vec_literal, top_k)
                _sql_learned = """
            WITH q AS (
                SELECT lower(%s) AS q_full, split_part(lower(%s), ' ', 1) AS q_first,
                metaphone(lower(%s), 10) AS q_mfull, metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst
            )
            SELECT la.kb_concept_id AS concept_id, la.alias_text AS preferred_name, c.kind, COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
                COALESCE(c.domain_key, '') AS domain_key,
                similarity(lower(la.alias_text), q.q_full) AS trigram_score,
                GREATEST(0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(la.alias_text), ' ', 1), 10), q.q_mfirst)::float / GREATEST(length(metaphone(split_part(lower(la.alias_text), ' ', 1), 10)), length(q.q_mfirst), 1))),
                    0.8 * (1.0 - (levenshtein(metaphone(lower(la.alias_text), 10), q.q_mfull)::float / GREATEST(length(metaphone(lower(la.alias_text), 10)), length(q.q_mfull), 1)))) AS phonetic_score,
                (1.0 - LEAST(la.embedding <=> %s::vector, 1.0)) AS vector_score,
                GREATEST(similarity(lower(la.alias_text), q.q_full), GREATEST(0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(la.alias_text), ' ', 1), 10), q.q_mfirst)::float / GREATEST(length(metaphone(split_part(lower(la.alias_text), ' ', 1), 10)), length(q.q_mfirst), 1))), 0.8 * (1.0 - (levenshtein(metaphone(lower(la.alias_text), 10), q.q_mfull)::float / GREATEST(length(metaphone(lower(la.alias_text), 10)), length(q.q_mfull), 1)))), (1.0 - LEAST(la.embedding <=> %s::vector, 1.0)))) AS match_score,
                la.embedding <=> %s::vector AS distance, 'learned' AS match_source
            FROM kb.learned_aliases la INNER JOIN kb.concepts c ON c.concept_id = la.kb_concept_id CROSS JOIN q
            WHERE la.embedding IS NOT NULL AND (la.is_validated = TRUE OR la.frequency_count > 0) AND c.kind = ANY(%s) AND (c.status IS NULL OR c.status != 'REJECTED')
              AND (similarity(lower(la.alias_text), q.q_full) >= 0.30 OR metaphone(split_part(lower(la.alias_text), ' ', 1), 10) = q.q_mfirst OR metaphone(lower(la.alias_text), 10) = q.q_mfull
                   OR (levenshtein(metaphone(split_part(lower(la.alias_text), ' ', 1), 10), q.q_mfirst)::float / GREATEST(length(metaphone(split_part(lower(la.alias_text), ' ', 1), 10)), length(q.q_mfirst), 1) < 0.65
                       OR levenshtein(metaphone(lower(la.alias_text), 10), q.q_mfull)::float / GREATEST(length(metaphone(lower(la.alias_text), 10)), length(q.q_mfull), 1) < 0.65)
                   OR (la.embedding IS NOT NULL AND la.embedding <=> %s::vector < 0.5))
            ORDER BY match_score DESC LIMIT %s
            """
                _params_learned = (normalized_name, normalized_name, normalized_name, normalized_name, vec_literal, vec_literal, vec_literal, kind_filter_list, vec_literal, top_k)
                single_source_queries = [
                    (_sql_concepts, _params_concepts, 'concepts'),
                    (_sql_aliases, _params_aliases, 'concept_aliases'),
                    (_sql_learned, _params_learned, 'learned_aliases'),
                ]
        else:
            use_separate_queries_merge = False
            single_source_queries = []
            # Two-method search (trigram + vector, no phonetic)
            # SET LOCAL must be executed separately (cannot be in same statement as WITH CTE)
            query_sql = """
            WITH all_candidates AS (
                SELECT 
                    c.concept_id,
                    c.preferred_name, 
                    c.kind,
                    COALESCE(c.definition, '') AS definition,
                    COALESCE(c.venom_id, '') AS venom_code,
                    COALESCE(c.snomed_id, '') AS snomed_code,
                    COALESCE(c.domain_key, '') AS domain_key,
                    c.embedding,
                    'concept' AS source
                FROM kb.concepts c
                WHERE c.embedding IS NOT NULL
                  AND c.kind = ANY(%s)
                  AND (c.status IS NULL OR c.status != 'REJECTED')
                
                UNION ALL
                
                SELECT 
                    ca.concept_id,
                    ca.alias_text AS preferred_name,  -- Use alias_text for searching
                    c.kind,
                    COALESCE(c.definition, '') AS definition,
                    COALESCE(c.venom_id, '') AS venom_code,
                    COALESCE(c.snomed_id, '') AS snomed_code,
                    COALESCE(c.domain_key, '') AS domain_key,
                    ca.embedding,
                    'alias' AS source
                FROM kb.concept_aliases ca
                JOIN kb.concepts c ON c.concept_id = ca.concept_id
                WHERE ca.embedding IS NOT NULL
                  AND c.kind = ANY(%s)
                  AND (c.status IS NULL OR c.status != 'REJECTED')
                
                UNION ALL
                
                SELECT 
                    la.kb_concept_id AS concept_id,
                    la.alias_text AS preferred_name,  -- Use alias_text for searching
                    c.kind,
                    COALESCE(c.definition, '') AS definition,
                    COALESCE(c.venom_id, '') AS venom_code,
                    COALESCE(c.snomed_id, '') AS snomed_code,
                    COALESCE(c.domain_key, '') AS domain_key,
                    la.embedding,
                    'learned' AS source
                FROM kb.learned_aliases la
                JOIN kb.concepts c ON c.concept_id = la.kb_concept_id
                WHERE la.embedding IS NOT NULL 
                  AND (la.is_validated = TRUE OR la.frequency_count > 0)
                  AND c.kind = ANY(%s)
                  AND (c.status IS NULL OR c.status != 'REJECTED')
            ),
            ranked AS (
                SELECT DISTINCT ON (concept_id)
                    concept_id,
                    preferred_name,
                    kind,
                    definition,
                    venom_code,
                    snomed_code,
                    domain_key,
                    -- Individual scores (trigram + vector, no phonetic)
                    similarity(lower(preferred_name), lower(%s)) AS trigram_score,
                    0.0 AS phonetic_score,
                    CASE 
                        WHEN embedding IS NOT NULL 
                        THEN (1.0 - LEAST(embedding <=> %s::vector, 1.0))
                        ELSE 0.0
                    END AS vector_score,
                    -- Match score: max(trigram, vector)
                    GREATEST(
                        similarity(lower(preferred_name), lower(%s)),
                        CASE 
                            WHEN embedding IS NOT NULL 
                            THEN (1.0 - LEAST(embedding <=> %s::vector, 1.0))
                            ELSE 0.0
                        END
                    ) AS match_score,
                    embedding <=> %s::vector AS distance,
                    source AS match_source
                FROM all_candidates
                WHERE (
                    similarity(lower(preferred_name), lower(%s)) >= 0.30
                    OR (embedding IS NOT NULL AND embedding <=> %s::vector < 0.5)
                )
                ORDER BY concept_id, match_score DESC
            )
            SELECT * FROM ranked
            ORDER BY match_score DESC
            LIMIT %s
            """
            # Parameters: kind_filter (x3), normalized_name (x4), vec_literal (x3), top_k
            # Note: ef_search is now set separately
            params = (kind_filter_list, kind_filter_list, kind_filter_list, 
                      normalized_name,  # trigram_score
                      vec_literal,  # vector_score
                      normalized_name,  # match_score trigram
                      vec_literal,  # match_score vector
                      vec_literal,  # distance
                      normalized_name,  # WHERE trigram
                      vec_literal,  # WHERE vector
                      top_k)
    else:
        # Vector-only search
        # SET LOCAL must be executed separately (cannot be in same statement as WITH CTE)
        query_sql = """
        WITH all_candidates AS (
            SELECT 
                c.concept_id, 
                c.preferred_name, 
                c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code,
                COALESCE(c.snomed_id, '') AS snomed_code,
                COALESCE(c.domain_key, '') AS domain_key,
                c.embedding,
                'concept' AS source
            FROM kb.concepts c
            WHERE c.embedding IS NOT NULL
              AND c.kind = ANY(%s)
              AND (c.status IS NULL OR c.status != 'REJECTED')
            
            UNION ALL
            
            SELECT 
                ca.concept_id,
                COALESCE(c.preferred_name, ca.alias_text) AS preferred_name,
                c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code,
                COALESCE(c.snomed_id, '') AS snomed_code,
                COALESCE(c.domain_key, '') AS domain_key,
                ca.embedding,
                'alias' AS source
            FROM kb.concept_aliases ca
            JOIN kb.concepts c ON c.concept_id = ca.concept_id
            WHERE ca.embedding IS NOT NULL
              AND c.kind = ANY(%s)
              AND (c.status IS NULL OR c.status != 'REJECTED')
            
            UNION ALL
            
            SELECT 
                la.kb_concept_id AS concept_id,
                COALESCE(c.preferred_name, la.alias_text) AS preferred_name,
                c.kind,
                COALESCE(c.definition, '') AS definition,
                COALESCE(c.venom_id, '') AS venom_code,
                COALESCE(c.snomed_id, '') AS snomed_code,
                COALESCE(c.domain_key, '') AS domain_key,
                la.embedding,
                'learned' AS source
            FROM kb.learned_aliases la
            JOIN kb.concepts c ON c.concept_id = la.kb_concept_id
            WHERE la.embedding IS NOT NULL
              AND (la.is_validated = TRUE OR la.frequency_count > 0)
              AND c.kind = ANY(%s)
              AND (c.status IS NULL OR c.status != 'REJECTED')
        )
        SELECT 
            concept_id,
            preferred_name,
            kind,
            definition,
            venom_code,
            snomed_code,
            domain_key,
            (1.0 - LEAST(embedding <=> %s::vector, 1.0)) AS hybrid_score,
            embedding <=> %s::vector AS distance,
            source AS match_source
        FROM all_candidates
        ORDER BY hybrid_score DESC
        LIMIT %s
        """
        # Parameters: kind_filter (x3), vec_literal (x2), top_k
        # Note: ef_search is now set separately
        params = (kind_filter_list, kind_filter_list, kind_filter_list, vec_literal, vec_literal, top_k)

    try:
        rows = []  # Initialize rows to empty list
        main_query_failed = False
        main_query_error = None
        
        # Ensure we're in a transaction for SET LOCAL to work
        # SET LOCAL requires a transaction, so we need to start one if autocommit is enabled
        autocommit_was_enabled = getattr(conn, 'autocommit', False)
        if autocommit_was_enabled:
            conn.autocommit = False
        
        try:
            # Use a savepoint to allow rollback on error without affecting outer transaction
            with conn.cursor() as cur:
                try:
                    # Create a savepoint for this query execution
                    try:
                        cur.execute("SAVEPOINT query_execution")
                    except Exception:
                        pass  # Savepoints may not be available in all transaction modes

                    # SET LOCAL must be executed separately before the main query
                    try:
                        cur.execute("SET LOCAL hnsw.ef_search = %s", (ef_search,))
                    except Exception as set_local_error:
                        if logger:
                            logger.debug(f"   SET LOCAL failed for '{normalized_name}' (non-critical): {set_local_error}")

                    # Alternate path: run three separate single-source queries and merge in Python.
                    # No single UNION execute() — eliminates psycopg2 IndexError as a global failure point.
                    if use_separate_queries_merge and single_source_queries:
                        all_rows = []
                        for sql, params, source_name in single_source_queries:
                            part = _run_single_source_query(cur, sql, params, source_name, normalized_name, logger)
                            all_rows.extend(part)
                        rows = _merge_dedupe_topk(all_rows, top_k)
                        if logger:
                            logger.debug(f"   Separate-queries merge returned {len(rows)} rows for '{normalized_name}'")
                    else:
                        # Validate params before execution - CRITICAL for production
                        import re
                        placeholder_count = len(re.findall(r'%s', query_sql))
                        if placeholder_count != len(params):
                            if logger:
                                logger.error(f"❌ CRITICAL: Parameter count mismatch for '{normalized_name}':")
                                logger.error(f"   Query has {placeholder_count} placeholders but params tuple has {len(params)} elements")
                                logger.error(f"   This will cause query execution to fail. Using fallback query instead.")
                            main_query_failed = True
                            main_query_error = ValueError(f"Parameter count mismatch: {placeholder_count} placeholders vs {len(params)} params")
                        else:
                            # Validate vec_literal format - ensure it's a valid PostgreSQL vector literal
                            if vec_literal:
                                if not isinstance(vec_literal, str):
                                    if logger:
                                        logger.error(f"❌ CRITICAL: vec_literal is not a string for '{normalized_name}': {type(vec_literal)}")
                                    main_query_failed = True
                                    main_query_error = ValueError(f"vec_literal must be a string, got {type(vec_literal)}")
                                elif not vec_literal.strip().startswith('[') or not vec_literal.strip().endswith(']'):
                                    if logger:
                                        logger.error(f"❌ CRITICAL: vec_literal has invalid format for '{normalized_name}': {vec_literal[:50]}...")
                                    main_query_failed = True
                                    main_query_error = ValueError(f"vec_literal must be a valid PostgreSQL vector literal (starts with [ and ends with ])")

                        if not main_query_failed:
                            if logger:
                                logger.debug(f"   Executing main query for '{normalized_name}' with kind_filter: {kind_filter_list}")
                            logger.debug(f"   Params count: {len(params)}, vec_literal type: {type(vec_literal)}, vec_literal length: {len(str(vec_literal)) if vec_literal else 0}")
                            logger.debug(f"   Query placeholder count: {placeholder_count}")
                        
                        # Pre-validate query structure by checking if tables exist
                        # This helps prevent errors that psycopg2 can't parse
                        try:
                            cur.execute("""
                                SELECT EXISTS (
                                    SELECT 1 FROM information_schema.tables 
                                    WHERE table_schema = 'kb' AND table_name = 'concept_aliases'
                                ) AS has_aliases,
                                EXISTS (
                                    SELECT 1 FROM information_schema.tables 
                                    WHERE table_schema = 'kb' AND table_name = 'learned_aliases'
                                ) AS has_learned
                            """)
                            table_check = cur.fetchone()
                            has_aliases_table = table_check[0] if table_check else False
                            has_learned_table = table_check[1] if table_check else False
                            
                            if logger:
                                logger.debug(f"   Table check: concept_aliases={has_aliases_table}, learned_aliases={has_learned_table}")
                        except Exception as check_error:
                            # Table check failed - assume tables exist and proceed
                            if logger:
                                logger.debug(f"   Table existence check failed (non-critical): {check_error}")
                            has_aliases_table = True
                            has_learned_table = True
                        
                        # Execute query with proper error handling
                        # Use a more robust execution method that handles psycopg2 IndexError
                        # The IndexError occurs when psycopg2 tries to parse PostgreSQL error responses
                        # We wrap execution in a way that's more compatible with psycopg2's error handling
                        rows = None
                        try:
                            # Execute query using a method that's more robust to psycopg2 parsing issues
                            # First, try to execute a simple validation query to ensure connection is healthy
                            try:
                                cur.execute("SELECT 1")
                                cur.fetchone()
                            except Exception:
                                # Connection issue - skip main query
                                if logger:
                                    logger.debug(f"   Connection validation failed for '{normalized_name}'")
                                main_query_failed = True
                                main_query_error = Exception("Connection validation failed")
                                rows = None
                            else:
                                # Connection is healthy - execute main query
                                # Use execute with explicit error handling
                                # The IndexError occurs when psycopg2 tries to parse PostgreSQL error responses
                                # We need to catch errors before psycopg2 tries to parse them
                                try:
                                    # Execute query - use execute with proper error handling
                                    # Wrap execution to catch PostgreSQL errors before psycopg2 parsing
                                    cur.execute(query_sql, params)
                                    rows = cur.fetchall()
                                    if logger:
                                        logger.debug(f"   Main query returned {len(rows)} rows")
                                    # Main query succeeded - skip fallback
                                except (IndexError, TypeError, AttributeError) as parse_error:
                                    # These errors occur when psycopg2 tries to parse PostgreSQL error responses
                                    # This is a known psycopg2 issue - the actual PostgreSQL error is lost
                                    # We need to handle this gracefully and use fallback
                                    if logger:
                                        logger.debug(f"   psycopg2 parsing error for '{normalized_name}' (PostgreSQL error not parseable): {parse_error}")
                                    # Don't re-raise - let the outer exception handler catch it
                                    raise
                        except IndexError as idx_error:
                            # IndexError from psycopg2 internals - this is the known issue
                            # It happens when psycopg2 tries to parse a PostgreSQL error response
                            # Rollback to savepoint to clean up transaction state
                            try:
                                cur.execute("ROLLBACK TO SAVEPOINT query_execution")
                            except Exception:
                                pass
                            
                            main_query_failed = True
                            main_query_error = idx_error
                            
                            # Try to get the actual PostgreSQL error by checking connection state
                            # Unfortunately, psycopg2 has already failed to parse it, so we can't get the original error
                            # But we can log that this is a known psycopg2 parsing issue
                            if logger:
                                logger.debug(f"   IndexError during query execution for '{normalized_name}' - psycopg2 error parsing issue, using fallback")
                            
                            rows = None
                        except Psycopg2Error as pg_error:
                            # Actual PostgreSQL error - we can get details
                            try:
                                cur.execute("ROLLBACK TO SAVEPOINT query_execution")
                            except Exception:
                                pass
                            
                            main_query_failed = True
                            main_query_error = pg_error
                            
                            error_code = getattr(pg_error, 'pgcode', None)
                            error_msg = str(pg_error)
                            if logger:
                                logger.debug(f"   PostgreSQL error for '{normalized_name}': {error_msg} (code: {error_code})")
                            
                            rows = None
                        except Exception as exec_error:
                            # Any other unexpected error
                            try:
                                cur.execute("ROLLBACK TO SAVEPOINT query_execution")
                            except Exception:
                                pass
                            
                            main_query_failed = True
                            main_query_error = exec_error
                            if logger:
                                logger.debug(f"   Query execution error for '{normalized_name}' (using fallback): {type(exec_error).__name__}: {exec_error}")
                            rows = None
                        
                        # Release savepoint if query succeeded
                        if rows is not None:
                            try:
                                cur.execute("RELEASE SAVEPOINT query_execution")
                            except Exception:
                                pass
                        elif main_query_failed:
                            # Query failed - savepoint already rolled back
                            pass
                except Exception as cursor_error:
                    # Catch any errors during cursor operations
                    main_query_failed = True
                    main_query_error = cursor_error
                    if logger:
                        logger.debug(f"   Cursor operation error for '{normalized_name}' (using fallback): {cursor_error}")
                
                # If main query failed, execute fallback (must be inside cursor context)
                sql_fallback = None
                params_fallback = None
                if main_query_failed:
                    if logger:
                        logger.debug(f"   Main query failed ({type(main_query_error).__name__}), executing fallback")
                        if not isinstance(main_query_error, IndexError):
                            logger.warning(f"Universal pool query failed for '{normalized_name}': {main_query_error}")
                        import traceback
                        logger.debug(f"   Traceback: {traceback.format_exc()}")

                    # Prefer 3-query merge fallback (concepts + aliases + learned) — no extra latency when UNION succeeds
                    if single_source_queries:
                        all_rows = []
                        for sql, params, source_name in single_source_queries:
                            part = _run_single_source_query(cur, sql, params, source_name, normalized_name, logger)
                            all_rows.extend(part)
                        rows = _merge_dedupe_topk(all_rows, top_k)
                        if logger:
                            logger.debug(f"   Fallback: separate-queries merge returned {len(rows)} rows for '{normalized_name}'")
                    # Else fallback to concepts-only SQL
                    elif has_pg_trgm and has_fuzzystrmatch:
                        # Fallback with three-method search (INCLUDES aggressive phonetic matching)
                        # SET LOCAL will be executed separately
                        sql_fallback = """
                    WITH q AS (
                        SELECT
                            lower(%s) AS q_full,
                            split_part(lower(%s), ' ', 1) AS q_first,
                            metaphone(lower(%s), 10) AS q_mfull,
                            metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst
                    )
                    SELECT 
                        c.concept_id,
                        c.preferred_name,
                        c.kind,
                        COALESCE(c.definition, '') AS definition,
                        COALESCE(c.venom_id, '') AS venom_code,
                        COALESCE(c.snomed_id, '') AS snomed_code,
                        COALESCE(c.domain_key, '') AS domain_key,
                        similarity(lower(c.preferred_name), q.q_full) AS trigram_score,
                        GREATEST(
                            0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10), q.q_mfirst)::float
                                         / GREATEST(length(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10)), length(q.q_mfirst), 1))),
                            0.8 * (1.0 - (levenshtein(metaphone(lower(c.preferred_name), 10), q.q_mfull)::float
                                         / GREATEST(length(metaphone(lower(c.preferred_name), 10)), length(q.q_mfull), 1)))
                        ) AS phonetic_score,
                        (1.0 - LEAST(c.embedding <=> %s::vector, 1.0)) AS vector_score,
                        GREATEST(
                            similarity(lower(c.preferred_name), q.q_full),
                            GREATEST(
                                0.8 * (1.0 - (levenshtein(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10), q.q_mfirst)::float
                                             / GREATEST(length(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10)), length(q.q_mfirst), 1))),
                                0.8 * (1.0 - (levenshtein(metaphone(lower(c.preferred_name), 10), q.q_mfull)::float
                                             / GREATEST(length(metaphone(lower(c.preferred_name), 10)), length(q.q_mfull), 1)))
                            ),
                            (1.0 - LEAST(c.embedding <=> %s::vector, 1.0))
                        ) AS match_score,
                        c.embedding <=> %s::vector AS distance,
                        'concept' AS match_source
                    FROM kb.concepts c
                    CROSS JOIN q
                    WHERE c.embedding IS NOT NULL
                      AND c.kind = ANY(%s)
                      AND (c.status IS NULL OR c.status != 'REJECTED')
                      AND (
                          similarity(lower(c.preferred_name), q.q_full) >= 0.30
                          OR metaphone(split_part(lower(c.preferred_name), ' ', 1), 10) = q.q_mfirst
                          OR metaphone(lower(c.preferred_name), 10) = q.q_mfull
                          OR (
                              levenshtein(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10), q.q_mfirst)::float
                              / GREATEST(length(metaphone(split_part(lower(c.preferred_name), ' ', 1), 10)), length(q.q_mfirst), 1) < 0.65
                              OR levenshtein(metaphone(lower(c.preferred_name), 10), q.q_mfull)::float
                              / GREATEST(length(metaphone(lower(c.preferred_name), 10)), length(q.q_mfull), 1) < 0.65
                          )
                          OR (c.embedding IS NOT NULL AND c.embedding <=> %s::vector < 0.5)
                      )
                    ORDER BY match_score DESC
                    LIMIT %s
                    """
                        # Parameters: normalized_name (x4 for q CTE), vec_literal (x3), kind_filter_list, vec_literal (WHERE), top_k
                        params_fallback = (normalized_name, normalized_name, normalized_name, normalized_name,
                                         vec_literal, vec_literal, vec_literal,
                                         kind_filter_list, vec_literal, top_k)
                    elif has_pg_trgm:
                        # Fallback with trigram + vector (no phonetic)
                        # SET LOCAL will be executed separately
                        sql_fallback = """
                    SELECT 
                        concept_id,
                        preferred_name,
                        kind,
                        COALESCE(definition, '') AS definition,
                        COALESCE(venom_id, '') AS venom_code,
                        COALESCE(snomed_id, '') AS snomed_code,
                        COALESCE(domain_key, '') AS domain_key,
                        similarity(lower(preferred_name), lower(%s)) AS trigram_score,
                        0.0 AS phonetic_score,
                        (1.0 - LEAST(embedding <=> %s::vector, 1.0)) AS vector_score,
                        GREATEST(
                            similarity(lower(preferred_name), lower(%s)),
                            (1.0 - LEAST(embedding <=> %s::vector, 1.0))
                        ) AS match_score,
                        embedding <=> %s::vector AS distance,
                        'concept' AS match_source
                    FROM kb.concepts
                    WHERE embedding IS NOT NULL
                      AND kind = ANY(%s)
                      AND (status IS NULL OR status != 'REJECTED')
                    ORDER BY match_score DESC
                    LIMIT %s
                    """
                        # Parameters: normalized_name (x2), vec_literal (x3), kind_filter_list, top_k
                        # Note: ef_search is now set separately
                        params_fallback = (normalized_name, vec_literal, normalized_name, vec_literal, 
                                         vec_literal, kind_filter_list, top_k)
                    else:
                        # Vector-only fallback (no trigram, no phonetic)
                        # SET LOCAL will be executed separately
                        sql_fallback = """
                    SELECT 
                        concept_id, 
                        preferred_name, 
                        kind,
                        COALESCE(definition, '') AS definition,
                        COALESCE(venom_id, '') AS venom_code,
                        COALESCE(snomed_id, '') AS snomed_code,
                        COALESCE(domain_key, '') AS domain_key,
                        0.0 AS trigram_score,
                        0.0 AS phonetic_score,
                        (1.0 - LEAST(embedding <=> %s::vector, 1.0)) AS vector_score,
                        (1.0 - LEAST(embedding <=> %s::vector, 1.0)) AS match_score,
                        embedding <=> %s::vector AS distance,
                        'concept' AS match_source
                    FROM kb.concepts
                    WHERE embedding IS NOT NULL
                      AND kind = ANY(%s)
                      AND (status IS NULL OR status != 'REJECTED')
                    ORDER BY match_score DESC
                    LIMIT %s
                    """
                        # Parameters: vec_literal (x3), kind_filter_list, top_k
                        # Note: ef_search is now set separately
                        params_fallback = (vec_literal, vec_literal, vec_literal, kind_filter_list, top_k)
                    
                    # Execute fallback query (common for all fallback types)
                    if main_query_failed and sql_fallback is not None and params_fallback is not None:
                        # SET LOCAL must be executed separately for fallback queries too
                        try:
                            try:
                                cur.execute("SET LOCAL hnsw.ef_search = %s", (ef_search,))
                            except Exception:
                                pass  # SET LOCAL may fail in some configs - non-critical
                            if logger:
                                logger.debug(f"   Executing fallback query for '{normalized_name}'")
                            # Validate fallback params too
                            import re
                            fallback_placeholder_count = len(re.findall(r'%s', sql_fallback))
                            if fallback_placeholder_count != len(params_fallback):
                                if logger:
                                    logger.error(f"❌ CRITICAL: Fallback query parameter mismatch!")
                                    logger.error(f"   Fallback query has {fallback_placeholder_count} placeholders but params_fallback has {len(params_fallback)} elements")
                                # This is a code bug - should not happen
                                raise ValueError(f"Fallback parameter count mismatch: {fallback_placeholder_count} vs {len(params_fallback)}")
                            cur.execute(sql_fallback, params_fallback)
                            rows = cur.fetchall()
                            if logger:
                                logger.debug(f"   Fallback query returned {len(rows)} rows")
                        except Exception as fallback_error:
                            # Catch ALL exceptions in fallback - if this fails, we return empty results
                            if logger:
                                # Only log as warning since main query already failed - this is expected behavior
                                logger.warning(f"   Fallback query also failed for '{normalized_name}': {type(fallback_error).__name__}")
                            # Set rows to empty list - don't re-raise
                            rows = []
                    elif main_query_failed:
                        # No fallback available - return empty results
                        if logger:
                            logger.warning(f"   No fallback query available (pg_trgm={has_pg_trgm}, fuzzystrmatch={has_fuzzystrmatch})")
                        rows = []
        except Exception as outer_error:
            # Catch any unexpected errors in the entire query execution block
            if logger:
                logger.error(f"❌ Unexpected error in query execution block for '{normalized_name}': {outer_error}")
                logger.error(f"   Error type: {type(outer_error).__name__}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
            # Return empty results on any unexpected error
            rows = []
        finally:
            # Restore autocommit state if we changed it
            if autocommit_was_enabled:
                try:
                    conn.autocommit = True
                except Exception:
                    pass  # Ignore errors when restoring autocommit
            
            # Check rows length (rows is now set from either main query or fallback)
            if logger:
                logger.debug(f"   📊 Final rows count before processing: {len(rows)} (type: {type(rows)})")
                if len(rows) == 0:
                    logger.warning(f"   ⚠️  Query executed but returned 0 rows for '{normalized_name}' (kind_filter: {kind_filter_list})")
                    # Debug: Check if concept exists
                    try:
                        with conn.cursor() as debug_cur:
                            debug_cur.execute("""
                                SELECT concept_id, preferred_name, kind 
                                FROM kb.concepts 
                                WHERE lower(preferred_name) LIKE %s 
                                  AND kind = ANY(%s)
                                LIMIT 5;
                            """, (f'%{normalized_name.lower()}%', kind_filter_list))
                            debug_results = debug_cur.fetchall()
                            if debug_results:
                                logger.warning(f"   🔍 DEBUG: Found {len(debug_results)} concepts matching pattern:")
                                for r in debug_results:
                                    logger.warning(f"      - '{r[1]}' (ID: {r[0]}, Kind: {r[2]})")
                    except Exception as debug_e:
                        logger.debug(f"   Debug query failed: {debug_e}")
                        import traceback
                        logger.debug(f"   Traceback: {traceback.format_exc()}")

        # Process results
        results = []
        if logger:
            logger.info(f"   📊 Starting to process {len(rows)} rows from query for '{normalized_name}'")
        for i, row in enumerate(rows):
            if logger and i == 0:
                logger.debug(f"   First row format: {len(row)} fields, preview: {str(row)[:150]}")
            try:
                # Handle different row formats based on whether phonetic matching is enabled
                # New format with individual scores + domain_key: concept_id, preferred_name, kind, definition, venom_code, snomed_code, 
                # domain_key, trigram_score, phonetic_score, vector_score, match_score, distance, match_source (13 fields)
                if len(row) >= 13:
                    concept_id, preferred_name, kind, definition, venom_code, snomed_code, domain_key, \
                    trigram_score, phonetic_score, vector_score, match_score, distance, match_source = row[:13]
                    hybrid_score = match_score  # Use match_score as hybrid_score for backward compatibility
                elif len(row) >= 12:
                    # Fallback: old format without domain_key (12 fields)
                    concept_id, preferred_name, kind, definition, venom_code, snomed_code, \
                    trigram_score, phonetic_score, vector_score, match_score, distance, match_source = row[:12]
                    domain_key = None  # No domain_key in old format
                    hybrid_score = match_score  # Use match_score as hybrid_score for backward compatibility
                elif len(row) >= 9:
                    # Old format: concept_id, preferred_name, kind, definition, venom_code, snomed_code, hybrid_score, distance, match_source
                    concept_id, preferred_name, kind, definition, venom_code, snomed_code, hybrid_score, distance, match_source = row[:9]
                    trigram_score = None
                    phonetic_score = None
                    vector_score = None
                    match_score = hybrid_score
                elif len(row) >= 8:
                    concept_id, preferred_name, kind, definition, venom_code, snomed_code, hybrid_score, match_source = row[:8]
                    distance = hybrid_score
                    trigram_score = None
                    phonetic_score = None
                    vector_score = None
                    match_score = hybrid_score
                elif len(row) >= 7:
                    concept_id, preferred_name, kind, venom_code, snomed_code, distance, match_source = row[:7]
                    definition = None
                    hybrid_score = None
                    trigram_score = None
                    phonetic_score = None
                    vector_score = None
                    match_score = None
                else:
                    if logger:
                        logger.warning(f"   ⚠️  Unexpected row format: {len(row)} fields (expected 12, 9, 8, or 7). Row: {row[:3] if len(row) >= 3 else row}")
                    continue
                
                result_dict = {
                    "concept_id": concept_id,
                    "preferred_name": preferred_name,
                    "kind": kind,
                    "definition": definition if definition else None,
                    "venom_code": venom_code if venom_code else None,
                    "snomed_code": snomed_code if snomed_code else None,
                    "domain_key": domain_key if domain_key else None,  # Include domain_key for clinical plausibility gate
                    "match_source": f"embedding_{match_source}",
                    "similarity_score": float(distance) if distance is not None else 0.0,
                }
                # Add individual scores if available (matching local search)
                if trigram_score is not None:
                    result_dict["trigram_score"] = float(trigram_score)
                if phonetic_score is not None:
                    result_dict["phonetic_score"] = float(phonetic_score)
                if vector_score is not None:
                    result_dict["vector_score"] = float(vector_score)
                # Use match_score as hybrid_score for backward compatibility
                if match_score is not None:
                    result_dict["hybrid_score"] = float(match_score)
                    result_dict["match_score"] = float(match_score)  # Also include match_score
                elif hybrid_score is not None:
                    result_dict["hybrid_score"] = float(hybrid_score)
                results.append(result_dict)
            except (ValueError, IndexError, TypeError) as e:
                if logger:
                    logger.warning(f"   ⚠️  Skipping malformed row: {e}. Row length: {len(row) if row else 0}, Row preview: {str(row)[:100] if row else 'None'}")
                    import traceback
                    logger.debug(f"   Traceback: {traceback.format_exc()}")
                continue
        
        if logger:
            if len(results) == 0 and len(rows) > 0:
                logger.error(f"   ❌ CRITICAL: Query returned {len(rows)} rows but all were filtered out during processing!")
                logger.error(f"   This indicates a row format mismatch. First row preview: {str(rows[0])[:200] if rows else 'No rows'}")
                logger.error(f"   First row length: {len(rows[0]) if rows else 0}, First row type: {type(rows[0]) if rows else 'None'}")
                if rows and len(rows[0]) > 0:
                    logger.error(f"   First row first element: {rows[0][0]}, type: {type(rows[0][0])}")
            elif len(results) > 0:
                logger.info(f"   ✅ Successfully processed {len(results)} results from {len(rows)} rows")
        
        if logger and len(results) > 0:
            best_match = results[0]
            logger.debug(f"   ✅ Best match: '{normalized_name}' → '{best_match['preferred_name']}' (score: {best_match.get('hybrid_score', best_match['similarity_score']):.4f})")
        elif logger and len(results) == 0:
            logger.warning(f"   ⚠️  Embedding search returned 0 results for '{normalized_name}' (kind_filter: {kind_filter_list})")
        
        if logger:
            logger.debug(f"   📤 Returning {len(results)} results from kb_lookup_concept_by_embedding for '{normalized_name}'")
        
        return results
    except Exception as e:
        if logger:
            logger.error(f"❌ Embedding search error for '{normalized_name}': {e}")
            import traceback
            logger.debug(f"   Traceback: {traceback.format_exc()}")
        return []


def kb_lookup_concept_hybrid_topk(
    conn,
    text: str,
    kind_filter: Optional[List[str]] = None,
    topk: int = 8,
    client=None,
    logger: Optional[logging.Logger] = None,
    embedding_cache: Optional[dict] = None,
    context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid KB search using both trigram and embedding (OPTIMIZED: parallel execution).
    When LOCAL_ONLY is true (default), returns [] and does not query kb.concepts.
    """
    if LOCAL_ONLY:
        return []
    # Try exact match first
    exact_matches = kb_lookup_concept_exact(conn, text)
    if exact_matches:
        return exact_matches[:topk]

    # OPTIMIZATION: Run trigram and embedding searches in parallel
    # CRITICAL: Pass context so kb_lookup_concept_by_embedding can set domain_anchor
    trigram_matches = []
    embedding_matches = []

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_trigram = executor.submit(
                kb_lookup_concept_by_trgm,
                conn, text, topk, kind_filter, logger
            )
            fut_embedding = executor.submit(
                kb_lookup_concept_by_embedding,
                conn, text, topk, context, client, logger, False, kind_filter, None,
                embedding_cache,
            )

            trigram_matches = fut_trigram.result() or []
            embedding_matches = fut_embedding.result() or []
    except Exception as e:
        if logger:
            logger.warning(f"Parallel search failed, falling back to sequential: {e}")
        trigram_matches = kb_lookup_concept_by_trgm(
            conn, text, top_k=topk, kind_filter=kind_filter, logger=logger
        )
        embedding_matches = kb_lookup_concept_by_embedding(
            conn, text, top_k=topk, context=context, kind_filter=kind_filter, client=client, logger=logger,
            embedding_cache=embedding_cache,
        )
    
    # Combine and deduplicate (preserve SQL hybrid_score when available)
    seen_ids = {}
    combined = []
    
    # Process trigram matches first (they have similarity_score)
    for match in trigram_matches:
        concept_id = match.get("concept_id")
        if concept_id:
            if concept_id not in seen_ids:
                seen_ids[concept_id] = match
                combined.append(match)
            else:
                # Keep the one with better score
                existing = seen_ids[concept_id]
                existing_score = existing.get("similarity_score", existing.get("hybrid_score", 0))
                new_score = match.get("similarity_score", 0)
                if new_score > existing_score:
                    seen_ids[concept_id] = match
                    combined[combined.index(existing)] = match
    
    # Process embedding matches (they have hybrid_score from SQL)
    for match in embedding_matches:
        concept_id = match.get("concept_id")
        if concept_id:
            if concept_id not in seen_ids:
                seen_ids[concept_id] = match
                combined.append(match)
            else:
                # Keep the one with better score (prefer SQL hybrid_score)
                existing = seen_ids[concept_id]
                existing_score = existing.get("hybrid_score", existing.get("similarity_score", 0))
                new_score = match.get("hybrid_score", match.get("similarity_score", 0))
                if new_score > existing_score:
                    seen_ids[concept_id] = match
                    combined[combined.index(existing)] = match
    
    # Sort by hybrid_score (from SQL) if available, otherwise similarity_score
    combined.sort(key=lambda x: x.get("hybrid_score", x.get("similarity_score", 0)), reverse=True)
    
    return combined[:topk]


def map_ner_kind_to_kb_kind_filter(ner_kind: str) -> Optional[List[str]]:
    """
    Map NER kind to KB kind filter.
    Returns None to search all kinds.
    
    Args:
        ner_kind: NER-extracted kind
        
    Returns:
        List of KB kinds to filter by, or None for all kinds
    """
    # 11-kind production schema + legacy: map NER kind to KB schema kinds for global search
    mapping = {
        # Production 11 kinds
        "Medication": ["Drug", "Substance"],
        "Procedure": ["Procedure", "Service", "DiagnosticTest", "Observation"],
        "Diagnostic": ["Procedure", "Service", "DiagnosticTest", "Observation"],
        "Reminder": ["Procedure", "Service", "Observation"],
        "Symptom": ["Symptom", "Finding", "Condition"],
        "Diagnosis": ["Condition", "Finding", "Procedure", "Service"],
        "Anatomy": ["Anatomy", "Observation"],
        "Diet": ["Nutrition", "Condition"],
        "ParasiteControl": ["Drug", "Substance"],
        "VitalSign": ["Observation"],
        "ReasonForVisit": ["Condition", "Finding", "Procedure", "Service"],
        # Legacy aliases
        "Drug": ["Drug", "Substance"],
        "DiagnosticTest": ["Procedure", "Service", "DiagnosticTest", "Observation"],
        "LabTest": ["Procedure", "Service", "DiagnosticTest", "Observation"],
        "Finding": ["Finding", "Condition"],
        "Condition": ["Condition", "Finding", "Procedure", "Service"],
        "Measurement": ["Observation", "Anatomy"],
        "Organism": ["Organism"],
        "Toxin": ["Toxin"],
        "Nutrition": ["Nutrition", "Condition"],
    }
    return mapping.get(ner_kind)


# Relational anchoring: boost for neighbor concepts (e.g. Norberg Angle -> Ortolani in same exam)
GLOBAL_NEIGHBOR_BOOST = float(os.getenv("GLOBAL_NEIGHBOR_BOOST", "0.4"))


def _fetch_neighbor_candidates(
    conn,
    anchor_concept_ids: List[int],
    kind_filter: List[str],
    detected_domain: Optional[str],
    boost: float,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch concepts that are clinical neighbors of anchor concepts (from kb.concept_relations).
    Used for relational anchoring: when the transcript already linked e.g. Norberg Angle,
    add Ortolani (and other neighbors) to the pool for other mentions with a boost.
    Returns list of candidate dicts with match_score=boost, match_source='relational_anchor'.
    """
    if not anchor_concept_ids or not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'kb' AND table_name = 'concept_relations'
                );
                """
            )
            row = cur.fetchone()
            if not (row and row[0]):
                return []
    except Exception:
        return []
    kind_list = list(kind_filter) if kind_filter else ["Procedure", "Service", "Observation", "Condition", "Finding"]
    domain_norm = (detected_domain or "").strip().lower()
    use_domain = bool(domain_norm and domain_norm not in ("", "general"))
    placeholders = ",".join(["%s"] * len(anchor_concept_ids))
    domain_clause = " AND LOWER(TRIM(COALESCE(c.domain_key, ''))) = %s " if use_domain else ""
    params: tuple = (tuple(anchor_concept_ids), kind_list)
    if use_domain:
        params = (tuple(anchor_concept_ids), kind_list, domain_norm)
    sql = f"""
    SELECT c.concept_id, c.preferred_name, c.kind,
        COALESCE(c.definition, '') AS definition,
        COALESCE(c.venom_id, '') AS venom_code, COALESCE(c.snomed_id, '') AS snomed_code,
        COALESCE(c.domain_key, '') AS domain_key
    FROM kb.concept_relations r
    INNER JOIN kb.concepts c ON c.concept_id = r.related_concept_id
    WHERE r.concept_id IN ({placeholders})
      AND c.kind = ANY(%s)
      AND (c.status IS NULL OR c.status != 'REJECTED')
    """ + domain_clause + """
    LIMIT 20
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    except Exception as e:
        if logger:
            logger.debug("Relational anchor fetch failed: %s", e)
        return []
    results = []
    for row in rows:
        (concept_id, preferred_name, kind, definition, venom_code, snomed_code, domain_key) = row[:7]
        results.append({
            "concept_id": concept_id,
            "preferred_name": preferred_name or "",
            "kind": kind or "",
            "definition": definition or None,
            "venom_code": venom_code or None,
            "snomed_code": snomed_code or None,
            "domain_key": (domain_key or "").strip() or None,
            "match_score": boost,
            "final_score": boost,
            "match_source": "relational_anchor",
            "is_neighbor": True,
            "neighbor_boost": boost,
        })
    if logger and results:
        logger.debug("Relational anchor: added %s neighbors for anchors %s", len(results), anchor_concept_ids[:5])
    return results


def _merge_neighbor_candidates(
    candidates: List[Dict[str, Any]],
    conn,
    anchor_concept_ids: Optional[List[int]],
    kind_filter: List[str],
    detected_domain: Optional[str],
    logger: Optional[logging.Logger] = None,
    neighbor_list: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Merge neighbor candidates (from anchor concepts) into the pool. Tag only: set is_neighbor, relation_boost.
    Do not add relation_boost to final_score here (rerank/two_stage already did for existing; for new neighbors add once)."""
    if not anchor_concept_ids or not candidates:
        return candidates
    if logger:
        logger.info(
            "Relational anchoring: merging neighbors for anchor_concept_ids=%s (mention pool size=%s)",
            anchor_concept_ids[:10], len(candidates),
        )
    neighbors = neighbor_list
    if neighbors is None:
        neighbors = _fetch_neighbor_candidates(
            conn, anchor_concept_ids, kind_filter, detected_domain, GLOBAL_NEIGHBOR_BOOST, logger
        )
    if not neighbors:
        return candidates
    if logger:
        logger.info("Relational anchoring: added %s neighbor concepts to pool", len(neighbors))
    w_rel = GLOBAL_TWO_STAGE_RELATION_WEIGHT
    by_cid = {c["concept_id"]: c for c in candidates}
    for n in neighbors:
        cid = n.get("concept_id")
        if cid is None:
            continue
        n["is_neighbor"] = True
        n["relation_boost"] = w_rel
        if cid not in by_cid:
            # New neighbor: not in rerank loop, so add relation once here (seed + w_rel).
            seed = float(n.get("final_score") or n.get("match_score", 0))
            n["final_score"] = seed + w_rel
            n["match_score"] = n["final_score"]
            by_cid[cid] = n
        else:
            # Existing candidate: rerank already added relation_boost to final_score; only tag.
            existing = by_cid[cid]
            existing["is_neighbor"] = True
            existing["relation_boost"] = w_rel
            # Do not mutate final_score (avoids double-counting).
    merged = list(by_cid.values())
    merged.sort(key=lambda x: float(x.get("final_score") or x.get("match_score", 0)), reverse=True)
    return merged


def search_global_topk(
    text: str,
    ner_kind: str,
    conn,
    client,
    topk: int = 8,
    logger: Optional[logging.Logger] = None,
    embedding_cache: Optional[dict] = None,
    raw_transcript: Optional[str] = None,
    anchor_concept_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Global KB search with proper intent handling and domain-aware phonetic fallback.
    When LOCAL_ONLY is true (default), returns [] and does not query kb.concepts.
    """
    if LOCAL_ONLY:
        return []
    # Canonicalize kind
    canonical_kind = canonicalize_kind(ner_kind)
    
    # Handle intent kinds (Reason) - allow multiple KB kinds
    if canonical_kind in INTENT_KINDS:
        kind_filter = list(REASON_ALLOWED_KB_KINDS)
    else:
        # Map NER kind to KB kind filter
        kind_filter = map_ner_kind_to_kb_kind_filter(canonical_kind)
    
    # Detect domain(s) once for this search (single for compat; multi for soft gate so we don't penalize valid domains)
    detected_domain = None
    detected_domains_set = set()
    if raw_transcript:
        try:
            from kb_ner_domain import detect_domain
            detected_domain = detect_domain(raw_transcript)
            detected_domains_list = detect_domain(raw_transcript, return_multiple=True)
            detected_domains_set = set(
                (d or "").strip().lower()
                for d in (detected_domains_list or [])
                if (d or "").strip().lower() and (d or "").strip().lower() != "general"
            )
            if not detected_domains_set and detected_domain and detected_domain.strip().lower() != "general":
                detected_domains_set = {detected_domain.strip().lower()}
        except Exception:
            pass
    
    # Use hybrid search (trigram + embedding)
    # When domain context is available, fetch more candidates to allow phonetic matches
    # to be boosted and included in final results
    effective_topk = topk
    if detected_domain and detected_domain != 'general':
        # Fetch a moderate number of extra candidates for domain-aware boosting.
        base = topk * 3
        min_candidates = 30  # Reduced from 60 to lower embedding latency
        max_candidates = 40  # Reduced from 150 to lower embedding latency
        effective_topk = max(min_candidates, min(base, max_candidates))
        if logger:
            logger.debug(
                f"   Domain '{detected_domain}' detected, fetching {effective_topk} candidates for phonetic/domain boosting"
            )
    try:
        candidates = kb_lookup_concept_hybrid_topk(
            conn, text, kind_filter=kind_filter, topk=effective_topk,
            client=client, logger=logger, embedding_cache=embedding_cache,
            context=raw_transcript,
        )

        # ------------------------------------------------------------------
        # SEMANTIC BRIDGE FALLBACK (embedding-only) when hybrid is weak
        # ------------------------------------------------------------------
        best_score = 0.0
        if candidates:
            for c in candidates:
                s = float(
                    c.get("match_score")
                    or c.get("hybrid_score")
                    or c.get("similarity_score", 0.0)
                    or 0.0
                )
                if s > best_score:
                    best_score = s
        fallback_min = 0.45
        try:
            fallback_min = float(os.getenv("KB_GLOBAL_EMBED_FALLBACK_THRESHOLD", "0.45"))
        except Exception:
            fallback_min = 0.45

        # Track whether we used semantic fallback (hybrid was weak). When True, we also run RAG
        # so definition-based matches (e.g. "Norberg angle" from text) can compete with
        # high-scoring but wrong concept matches (e.g. "nebula" for "noble angle").
        semantic_fallback_used = False
        if (not candidates) or best_score < fallback_min:
            if logger:
                logger.debug(
                    f"  🌉 Semantic fallback: hybrid best_score={best_score:.3f} < {fallback_min:.2f} for '{text}' – running embedding-only search"
                )
            try:
                embed_only = kb_lookup_concept_by_embedding(
                    conn,
                    text,
                    top_k=max(topk * 5, 50),
                    context=raw_transcript,
                    client=client,
                    logger=logger,
                    use_rag=False,
                    kind_filter=kind_filter,
                    embedding_cache=embedding_cache,
                )
                if embed_only:
                    candidates = embed_only
                    semantic_fallback_used = True
                    if logger:
                        logger.debug(
                            f"  🌉 Semantic fallback succeeded for '{text}': {len(candidates)} candidates from embedding-only search"
                        )
            except Exception as e_fallback:
                if logger:
                    logger.warning(
                        f"  ⚠️ Semantic fallback (embedding-only) failed for '{text}': {e_fallback}"
                    )

        # ------------------------------------------------------------------
        # RAG-BASED DEFINITION FALLBACK (kb.kb_text_embeddings → kb.concepts)
        # Trigger when: (1) no/weak candidates (best_score < rag_min), OR
        # (2) we used semantic fallback — then the top candidate may be wrong (e.g. "nebula"
        # for "noble angle"); RAG pulls concepts from definitions so Norberg angle can surface.
        # ------------------------------------------------------------------
        rag_min = 0.30
        try:
            rag_min = float(os.getenv("KB_GLOBAL_RAG_FALLBACK_THRESHOLD", "0.30"))
        except Exception:
            rag_min = 0.30

        # Recompute best_score on the (possibly updated) candidate set
        best_score_rag = 0.0
        if candidates:
            for c in candidates:
                s = float(
                    c.get("match_score")
                    or c.get("hybrid_score")
                    or c.get("similarity_score", 0.0)
                    or 0.0
                )
                if s > best_score_rag:
                    best_score_rag = s

        # Invoke RAG when: no candidates, OR best score is very low (< rag_min), OR
        # semantic fallback was used (even if best_score is high, the top candidate may be wrong).
        # RAG acts as a Clinical Intelligence Layer: it prevents clinically impossible links
        # (e.g., "nebula" for "noble angle" in an orthopedic note) and bridges severe ASR hallucinations
        # (e.g., "ultralining" → Ortolani) by searching KB definitions with full transcript context.
        run_rag = raw_transcript and client and (
            (not candidates)
            or best_score_rag < rag_min
            or semantic_fallback_used
        )
        if run_rag:
            if logger:
                logger.debug(
                    f"  📚 RAG fallback: querying kb_text_embeddings for '{text}'"
                    f" (best_score={best_score_rag:.3f}, semantic_fallback_used={semantic_fallback_used})"
                )
            try:
                rag_candidates = kb_lookup_concept_by_embedding(
                    conn,
                    text,
                    top_k=max(topk * 3, 30),
                    context=raw_transcript,
                    client=client,
                    logger=logger,
                    use_rag=True,
                    kind_filter=kind_filter,
                    embedding_cache=embedding_cache,
                )
                if rag_candidates:
                    # Merge RAG candidates into pool by concept_id (keep best score per concept),
                    # so definition-based matches (e.g. Norberg) compete with direct-embedding matches.
                    prior_count = len(candidates)
                    by_cid: Dict[Any, Dict[str, Any]] = {}
                    for c in candidates:
                        cid = c.get("concept_id")
                        if cid is not None:
                            s = float(
                                c.get("match_score")
                                or c.get("hybrid_score")
                                or c.get("similarity_score", 0.0)
                                or 0.0
                            )
                            if cid not in by_cid or (by_cid[cid].get("match_score") or 0) < s:
                                by_cid[cid] = {**c, "match_score": s}
                    for c in rag_candidates:
                        cid = c.get("concept_id")
                        if cid is None:
                            continue
                        s = float(c.get("similarity_score", 0.0) or 0.0)
                        existing = by_cid.get(cid)
                        if existing is None or (existing.get("match_score") or 0) < s:
                            by_cid[cid] = {
                                **c,
                                "match_score": s,
                                "match_source": c.get("match_source", "rag"),
                            }
                    candidates = sorted(
                        by_cid.values(),
                        key=lambda x: float(x.get("match_score") or 0.0),
                        reverse=True,
                    )
                    if logger:
                        logger.debug(
                            f"  📚 RAG fallback succeeded for '{text}': merged {len(rag_candidates)} RAG + {prior_count} prior → {len(candidates)} unique candidates"
                        )
            except Exception as e_rag:
                if logger:
                    logger.debug(
                        f"  ⚠️ RAG-based fallback failed for '{text}': {e_rag} (non-critical)"
                    )

        # SOFT GATE: Domain as boost (not hard filter) for resilient clinical grounding.
        # Multi-domain: boost when candidate matches ANY detected domain; penalize only when candidate is outside all detected domains.
        # So orthopedic + infectious transcript: both Ortolani and Parvo candidates get boosted; only unrelated domains penalized.
        if SOFT_GATE_ENABLED and detected_domains_set and candidates:
            # Safety net: drop candidates with base score below threshold (noise)
            if SOFT_GATE_THRESHOLD > 0:
                base_score_key = "match_score"
                candidates = [c for c in candidates if float(c.get(base_score_key) or c.get("hybrid_score") or c.get("similarity_score") or 0.0) > SOFT_GATE_THRESHOLD]
            for c in candidates:
                base = float(c.get("match_score") or c.get("hybrid_score") or c.get("similarity_score") or 0.0)
                cand_domain = (c.get("domain_key") or "").strip().lower() or ""
                # Boost when candidate matches any detected domain; penalize only when candidate has a domain outside the set
                if cand_domain in detected_domains_set:
                    domain_boost = SOFT_GATE_DOMAIN_BOOST
                    domain_penalty = 0.0
                elif cand_domain and cand_domain not in GLOBAL_TWO_STAGE_DOMAIN_NEUTRAL:
                    domain_boost = 0.0
                    domain_penalty = SOFT_GATE_DOMAIN_FRICTION
                else:
                    domain_boost = 0.0
                    domain_penalty = 0.0
                c["domain_boost"] = domain_boost
                c["domain_penalty"] = domain_penalty
                # Note: suggestion_boost is not available in search_global_topk (per-entity function)
                # It's only applied in run_batch_global_vector_search where we have entity_idx mapping
                final = (base * SOFT_GATE_BASE_WEIGHT) + domain_boost - domain_penalty
                c["final_score"] = max(0.0, final)
            candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
            if logger:
                top_fs = candidates[0].get("final_score", 0.0) if candidates else 0.0
                logger.debug(
                    "   Soft gate (multi-domain %s): boost match / penalize outside (top final_score=%.3f)",
                    sorted(detected_domains_set), top_fs,
                )

        # DOMAIN-AWARE PHONETIC BOOST: Boost phonetic scores when domain context matches
        # This makes phonetic matching scalable without hardcoding specific mappings
        # The SQL WHERE clause already includes aggressive phonetic matching (levenshtein < 0.65)
        # Here we boost scores post-query when domain context validates the match
        # Skip when results are already from domain-anchored global (e.g. "orthopedic: noble angle"),
        # so no need to call OpenAI to re-embed domain + candidate names for affinity.
        if raw_transcript and candidates:
            try:
                from kb_ner_domain import detect_domain
                # Detect ALL domains (not just first match) to handle multi-problem cases
                detected_domains = detect_domain(raw_transcript, return_multiple=True)
                
                # Filter out "general" domain
                detected_domains = [d for d in detected_domains if d != "general"]
                
                # Skip OpenAI embedding when all candidates are from domain-anchored global (saves API calls)
                from_domain_global = all(
                    (c.get("match_source") or "").strip().startswith("embedding_")
                    for c in candidates
                )
                if from_domain_global and detected_domains and client:
                    if logger:
                        logger.debug("  Skipping embedding-based domain boost (results from domain-anchored global; no OpenAI embed)")
                    # Apply soft-gate (multi-domain: boost only when cand in detected_domains_set)
                    if detected_domains_set:
                        for c in candidates:
                            base = float(c.get("match_score", 0.0))
                            cand_domain = (c.get("domain_key") or "").strip().lower() or ""
                            if cand_domain in detected_domains_set:
                                c["final_score"] = (base * SOFT_GATE_BASE_WEIGHT) + SOFT_GATE_DOMAIN_BOOST
                            elif cand_domain and cand_domain not in GLOBAL_TWO_STAGE_DOMAIN_NEUTRAL:
                                c["final_score"] = max(0.0, (base * SOFT_GATE_BASE_WEIGHT) - SOFT_GATE_DOMAIN_FRICTION)
                            else:
                                c["final_score"] = base * SOFT_GATE_BASE_WEIGHT
                            c["match_score"] = c["final_score"]
                        candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
                elif detected_domains and client:
                    if logger:
                        logger.debug(f"  🎯 Domains detected: {detected_domains} - applying embedding-based domain boosting")
                    
                    # Vector-to-domain affinity (scalable: no keyword lists)
                    from kb_domain_affinity import get_domain_embedding, cosine_similarity
                    from kb_ner_embeddings import embed_texts
                    domain_embeddings: Dict[str, List[float]] = {}
                    for d in detected_domains:
                        emb = get_domain_embedding(d, client=client, logger=logger)
                        if emb:
                            domain_embeddings[d] = emb
                    if not domain_embeddings:
                        if logger:
                            logger.debug("  ⚠️ No domain embeddings available - skipping domain boost")
                    else:
                        unique_names = list(
                            dict.fromkeys(
                                (c.get("preferred_name") or "").strip()
                                for c in candidates
                                if (c.get("preferred_name") or "").strip()
                            )
                        )
                        # Limit how many candidate names we embed for domain boosting to control cost.
                        try:
                            max_boost = int(os.getenv("KB_DOMAIN_BOOST_MAX_CANDIDATES", "150"))
                        except Exception:
                            max_boost = 150
                        if max_boost > 0 and len(unique_names) > max_boost:
                            unique_names = unique_names[:max_boost]
                        name_to_emb: Dict[str, List[float]] = {}
                        if unique_names:
                            try:
                                batch = embed_texts(unique_names, client=client, logger=logger)
                                for name, vec in zip(unique_names, batch):
                                    if vec:
                                        name_to_emb[name] = vec
                            except Exception as emb_err:
                                if logger:
                                    logger.debug(f"  ⚠️ Batch embed for domain boost failed: {emb_err}")
                        # Only boost candidates that are domain-relevant by embedding affinity (>= 0.85)
                        for cand in candidates:
                            phonetic_score = cand.get("phonetic_score", 0)
                            trigram = cand.get("trigram_score", 0)
                            vector = cand.get("vector_score", 0)
                            candidate_name = (cand.get("preferred_name", "") or "").strip()
                            cand_emb = name_to_emb.get(candidate_name) if candidate_name else None
                            is_domain_relevant = False
                            if cand_emb and domain_embeddings:
                                for _d, d_emb in domain_embeddings.items():
                                    if cosine_similarity(cand_emb, d_emb) >= 0.85:
                                        is_domain_relevant = True
                                        break
                        
                        # Only boost if candidate is domain-relevant and has some signal (trigram/phonetic/vector)
                        base_signal = max(trigram, phonetic_score, vector)
                        if is_domain_relevant and base_signal > 0.15:
                                # For domain-relevant phonetic-only matches (low trigram < 0.1, phonetic 0.15-0.4),
                                # apply strong boost to ensure they rank in top results
                                if trigram < 0.1 and 0.15 <= phonetic_score <= 0.4:
                                    if phonetic_score < 0.25:
                                        boosted_phonetic = max(0.80, phonetic_score * 3.5)  # Boost to at least 0.80 for auto-link
                                    else:
                                        boosted_phonetic = max(0.95, phonetic_score * 2.5)
                                elif 0.15 <= phonetic_score <= 0.4:
                                    boosted_phonetic = min(1.0, phonetic_score * 2.0)
                                else:
                                    boosted_phonetic = min(1.0, phonetic_score * 1.5)
                                original_match = cand.get("match_score", 0)
                                boosted_match = max(trigram, boosted_phonetic, vector)
                                if boosted_match > original_match:
                                    cand["phonetic_score"] = boosted_phonetic
                                    cand["match_score"] = boosted_match
                                    cand["match_source"] = f"{cand.get('match_source', 'hybrid')}_domain_boosted"
                                    if logger:
                                        logger.debug(f"  🎯 Domain boost: '{cand.get('preferred_name', 'Unknown')}' phonetic {phonetic_score:.3f} → {boosted_phonetic:.3f}, match {original_match:.3f} → {boosted_match:.3f}")
                        # Re-sort by match_score after boosting
                        candidates.sort(key=lambda x: x.get("match_score", 0), reverse=True)
                        # Re-apply soft gate (multi-domain: boost if in detected_domains_set, penalize if outside)
                        if SOFT_GATE_ENABLED and detected_domains_set:
                            for c in candidates:
                                base = float(c.get("match_score", 0.0))
                                cand_domain = (c.get("domain_key") or "").strip().lower() or ""
                                if cand_domain in detected_domains_set:
                                    domain_boost, domain_penalty = SOFT_GATE_DOMAIN_BOOST, 0.0
                                elif cand_domain and cand_domain not in GLOBAL_TWO_STAGE_DOMAIN_NEUTRAL:
                                    domain_boost, domain_penalty = 0.0, SOFT_GATE_DOMAIN_FRICTION
                                else:
                                    domain_boost, domain_penalty = 0.0, 0.0
                                c["domain_boost"] = domain_boost
                                c["domain_penalty"] = domain_penalty
                                c["final_score"] = max(0.0, (base * SOFT_GATE_BASE_WEIGHT) + domain_boost - domain_penalty)
                            candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
            except Exception as e:
                if logger:
                    logger.debug(f"  ⚠️  Domain-aware phonetic boost failed: {e} (non-critical)")

        # Relational anchoring: merge neighbor candidates from anchors (same transcript)
        if anchor_concept_ids:
            candidates = _merge_neighbor_candidates(
                candidates, conn, anchor_concept_ids, kind_filter, detected_domain, logger
            )
        # Return topk results (ordered by final_score when soft gate applied, else match_score)
        return candidates[:topk] if candidates else []
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠️  Global KB search failed: {e}")
        return []


def kb_lookup_learned_alias(
    conn,
    alias_text: str,
    kb_kind: Optional[str] = None,
    top_k: int = 15,
    logger: Optional[logging.Logger] = None,
    prefer_validated: bool = True,
) -> List[Dict[str, Any]]:
    """
    Layer-0 self-healing alias lookup: exact match against kb.learned_aliases.
    
    HIGH-BAR PROTOCOL: Prioritizes validated aliases and requires minimum frequency for unvalidated ones.
    
    Args:
        conn: Database connection
        alias_text: Alias text to lookup
        kb_kind: Optional KB kind filter
        top_k: Number of results
        logger: Optional logger
        prefer_validated: Whether to prefer validated aliases
        
    Returns:
        List of concept matches from learned aliases
    """
    try:
        with conn.cursor() as cur:
            if prefer_validated:
                cur.execute("""
                    SELECT 
                        la.kb_concept_id, 
                        c.preferred_name, 
                        c.kind, 
                        la.confidence_score, 
                        la.frequency_count,
                        la.is_validated
                    FROM kb.learned_aliases la
                    JOIN kb.concepts c ON c.concept_id = la.kb_concept_id
                    WHERE lower(la.alias_text) = lower(%s)
                      AND (
                          la.is_validated = TRUE 
                          OR (la.is_validated = FALSE AND la.frequency_count >= 3)
                      )
                    ORDER BY la.is_validated DESC, la.confidence_score DESC, la.frequency_count DESC
                """, (alias_text,))
            else:
                cur.execute("""
                    SELECT 
                        la.kb_concept_id, 
                        c.preferred_name, 
                        c.kind, 
                        la.confidence_score, 
                        la.frequency_count,
                        la.is_validated
                    FROM kb.learned_aliases la
                    JOIN kb.concepts c ON c.concept_id = la.kb_concept_id
                    WHERE lower(la.alias_text) = lower(%s)
                      AND la.frequency_count > 0
                    ORDER BY la.is_validated DESC, la.confidence_score DESC, la.frequency_count DESC
                """, (alias_text,))
            rows = cur.fetchall()
        results = []
        for concept_id, preferred_name, kind, confidence, frequency_count, is_validated in rows:
            if kb_kind and kind != kb_kind:
                continue
            results.append({
                "concept_id": concept_id,
                "preferred_name": preferred_name,
                "kind": kind,
                "match_source": "learned_alias_exact",
                "confidence": float(confidence) if confidence is not None else None,
                "frequency_count": int(frequency_count) if frequency_count is not None else 0,
                "is_validated": bool(is_validated) if is_validated is not None else False,
            })
        return results[:top_k]
    except Exception as e:
        if logger:
            logger.debug(f"learned_alias lookup failed: {e}")
        return []


def should_learn_alias(
    conn,
    alias_text: str,
    concept_id: int,
    confidence: float,
    min_confidence_threshold: float = 0.95,
    min_frequency_for_activation: int = 3,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str, Optional[int]]:
    """
    HIGH-BAR PROTOCOL: Three Safety Gates to prevent data poisoning.
    
    Gate 1: Confidence Threshold (> 0.95)
    Gate 2: Provisional Flag (Quarantine)
    Gate 3: Frequency Consensus
    
    Args:
        conn: Database connection
        alias_text: Alias text
        concept_id: Concept ID
        confidence: Confidence score
        min_confidence_threshold: Minimum confidence threshold
        min_frequency_for_activation: Minimum frequency for activation
        logger: Optional logger
        
    Returns:
        Tuple of (should_learn, reason, current_frequency)
    """
    # Gate 1: Confidence Threshold
    if confidence < min_confidence_threshold:
        return False, f"Confidence {confidence:.3f} below threshold {min_confidence_threshold}", None
    
    # Check if alias already exists
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT frequency_count, is_validated, confidence_score
                FROM kb.learned_aliases
                WHERE alias_text = %s AND kb_concept_id = %s;
            """, (alias_text, concept_id))
            existing = cur.fetchone()
            
            if existing:
                current_freq, is_validated, existing_conf = existing
                if existing_conf and existing_conf >= confidence:
                    return False, f"Existing mapping has higher confidence ({existing_conf:.3f} >= {confidence:.3f})", current_freq
                
                return True, f"Updating existing alias (freq: {current_freq}, validated: {is_validated})", current_freq
            else:
                # New alias - check for conflicts
                cur.execute("""
                    SELECT COUNT(DISTINCT kb_concept_id)
                    FROM kb.learned_aliases
                    WHERE alias_text = %s;
                """, (alias_text,))
                conflict_count = cur.fetchone()[0]
                
                if conflict_count > 0:
                    return False, f"Alias '{alias_text}' already maps to {conflict_count} different concept(s) - conflict detected", None
                
                return True, f"New alias meets high-bar criteria (confidence: {confidence:.3f})", None
                
    except Exception as e:
        if logger:
            logger.warning(f"Error checking alias status: {e}")
        return False, f"Error checking alias status: {e}", None


def upsert_learned_alias(
    conn,
    alias_text: str,
    concept_id: int,
    kb_kind: Optional[str],
    confidence: float,
    source: str = "llm_judge",
    embedding: Optional[List[float]] = None,
    logger: Optional[logging.Logger] = None,
    min_confidence_threshold: float = 0.95,
    enforce_high_bar: bool = True,
) -> Tuple[bool, str]:
    """
    Write-through cache for self-healing: once a typo is confidently resolved,
    store it so future runs are 0ms DB hits (no embeddings / no LLM).
    
    HIGH-BAR PROTOCOL: Only saves aliases that pass strict safety gates.
    
    Args:
        conn: Database connection
        alias_text: Alias text
        concept_id: Concept ID
        kb_kind: KB kind
        confidence: Confidence score
        source: Source of the alias
        embedding: Optional embedding
        logger: Optional logger
        min_confidence_threshold: Minimum confidence threshold
        enforce_high_bar: Whether to enforce high-bar protocol
        
    Returns:
        Tuple of (saved, reason)
    """
    # HIGH-BAR PROTOCOL: Check if alias should be learned
    if enforce_high_bar:
        should_learn, reason, current_freq = should_learn_alias(
            conn, alias_text, concept_id, confidence, 
            min_confidence_threshold=min_confidence_threshold,
            logger=logger
        )
        
        if not should_learn:
            if logger:
                logger.info(f"⚠️  HIGH-BAR REJECTED: '{alias_text}' -> {concept_id} (confidence: {confidence:.3f}) - {reason}")
            return False, reason
    
    try:
        # Generate embedding if not provided
        if embedding is None:
            embedding = embed_text(alias_text, logger=logger)
        
        embedding_literal = to_pgvector_literal(embedding) if embedding else None
        
        with conn.cursor() as cur:
            # Check if table has embedding column
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_schema = 'kb' 
                    AND table_name = 'learned_aliases'
                    AND column_name = 'embedding'
                );
            """)
            has_embedding_col = cur.fetchone()[0]
            
            if has_embedding_col:
                if embedding_literal:
                    cur.execute("""
                        INSERT INTO kb.learned_aliases 
                            (alias_text, kb_concept_id, embedding, confidence_score, source, frequency_count, last_used_at, is_validated)
                        VALUES 
                            (%s, %s, %s::vector, %s, %s, 1, NOW(), FALSE)
                        ON CONFLICT (alias_text, kb_concept_id) 
                        DO UPDATE SET
                            embedding = COALESCE(EXCLUDED.embedding, kb.learned_aliases.embedding),
                            confidence_score = GREATEST(
                                COALESCE(kb.learned_aliases.confidence_score, 0), 
                                COALESCE(EXCLUDED.confidence_score, 0)
                            ),
                            frequency_count = kb.learned_aliases.frequency_count + 1,
                            last_used_at = NOW();
                    """, (alias_text, concept_id, embedding_literal, confidence, source))
                else:
                    cur.execute("""
                        INSERT INTO kb.learned_aliases 
                            (alias_text, kb_concept_id, confidence_score, source, frequency_count, last_used_at, is_validated)
                        VALUES 
                            (%s, %s, %s, %s, 1, NOW(), FALSE)
                        ON CONFLICT (alias_text, kb_concept_id) 
                        DO UPDATE SET
                            confidence_score = GREATEST(
                                COALESCE(kb.learned_aliases.confidence_score, 0), 
                                COALESCE(EXCLUDED.confidence_score, 0)
                            ),
                            frequency_count = kb.learned_aliases.frequency_count + 1,
                            last_used_at = NOW();
                    """, (alias_text, concept_id, confidence, source))
                
                # Get updated frequency
                cur.execute("""
                    SELECT frequency_count, is_validated
                    FROM kb.learned_aliases
                    WHERE alias_text = %s AND kb_concept_id = %s;
                """, (alias_text, concept_id))
                updated = cur.fetchone()
                updated_freq = updated[0] if updated else 1
                is_validated = updated[1] if updated else False
                
                if logger:
                    val_status = "✅ VALIDATED" if is_validated else "⚠️  UNVALIDATED (quarantine)"
                    logger.info(f"🎓 Learned alias: '{alias_text}' -> {concept_id} (confidence: {confidence:.3f}, freq: {updated_freq}, {val_status})")
                
                if updated_freq >= 3 and not is_validated:
                    if logger:
                        logger.info(f"💡 Alias '{alias_text}' has been seen {updated_freq} times - candidate for manual validation")
        conn.commit()
        return True, "Saved successfully"
    except Exception as e:
        if logger:
            logger.debug(f"Failed to upsert learned alias '{alias_text}' -> {concept_id}: {e}")
        return False, str(e)

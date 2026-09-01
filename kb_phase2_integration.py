"""
Phase 2 Knowledge Atom Extraction - Optimized Integration for Phase 1 Pipeline

This module provides a lightweight async wrapper for Phase 2 that:
- Runs after constraint injection, using the FINAL SOAP note (accuracy)
- Uses Phase 1's cost tracking module (no duplication)
- Uses Phase 1's connection pool (no new connections)
- Accepts SOAP note text + entity manifest directly (no file I/O)
- Default mode is MINIMAL extraction (assertion + attributes). Optional heavy steps are OFF by default.

Author: VetInstant P.A.W.S Team
Version: 1.0 (Phase 1 Integration)
"""

import os
import json
import uuid
import asyncio
import logging
import time
import threading
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Import Phase 2 core functions (reuse existing code)
try:
    from SOAP_notes_billing_phase2_kb_atoms import (
        parse_soap_sections,
        build_knowledge_atom_prompt,
        build_phase2_prompt_with_grounding,
        enrich_atoms_with_manifest_bindings,
        _sha256,
        ensure_phase2_session_tables,
        ensure_session,
        is_section_dirty,
        mark_section_dirty,
        upsert_section_atoms,
        load_cached_atoms,
        intent_filter,
        Phase2KnowledgeAtomMatcher,
        get_assertion_types,
        get_all_attributes_schema,
        _build_verification_dashboard,
    )
    from kb_ner_db import pg_conn_ctx
    from kb_ner_clients import get_client_for_model
    PHASE2_AVAILABLE = True
except ImportError as e:
    PHASE2_AVAILABLE = False
    import_error = str(e)


# ==============================================================================
# PHASE 2 CONFIGURATION (aligned with Phase 1)
# ==============================================================================

# Default Phase 2 model (can be overridden via env)
os.environ.setdefault("PHASE2_MODEL", "gpt-4.1-mini")
os.environ.setdefault("PHASE2_MODEL_PROVIDER", "openai")

PHASE2_MODEL = os.getenv("PHASE2_MODEL", "gpt-4.1-mini")
PHASE2_MODEL_PROVIDER = os.getenv("PHASE2_MODEL_PROVIDER", "openai")

# Phase 2 enable flag (can be disabled for testing)
ENABLE_PHASE2 = os.getenv("ENABLE_PHASE2", "true").lower() in ("1", "true", "yes")

# Default behavior: minimal, single-call extraction (lowest latency).
PHASE2_MODE = os.getenv("PHASE2_MODE", "minimal").lower()  # "minimal" or "full"
PHASE2_PARALLEL_SECTIONS = os.getenv("PHASE2_PARALLEL_SECTIONS", "false").lower() in ("1", "true", "yes")
PHASE2_ENABLE_SESSION_CACHE = os.getenv("PHASE2_ENABLE_SESSION_CACHE", "false").lower() in ("1", "true", "yes")
PHASE2_ENABLE_BILLING_MATCHING_DEFAULT = os.getenv("PHASE2_ENABLE_BILLING_MATCHING", "false").lower() in ("1", "true", "yes")
PHASE2_DRY_RUN = os.getenv("PHASE2_DRY_RUN", "false").lower() in ("1", "true", "yes")
# Phase 2 output: floor high enough to avoid truncation (JSON parse errors).
# NOTE: Truncation causes "Phase 2: JSON parse error" and retry; 3200 floor reduces that.
PHASE2_MAX_TOKENS = int(os.getenv("PHASE2_MAX_TOKENS", "3200"))
PHASE2_TEMPERATURE = float(os.getenv("PHASE2_TEMPERATURE", "0.0"))
# For long runs, ceiling high so the first call rarely truncates.
PHASE2_MAX_TOKENS_LONG = int(os.getenv("PHASE2_MAX_TOKENS_LONG", "10000"))
# Parallel batches: dynamic max_tokens = f(entities in batch). Cap and floor below.
PHASE2_MAX_TOKENS_PARALLEL = int(os.getenv("PHASE2_MAX_TOKENS_PARALLEL", "8000"))   # cap
PHASE2_MAX_TOKENS_PARALLEL_MIN = int(os.getenv("PHASE2_MAX_TOKENS_PARALLEL_MIN", "2000"))  # floor for small batches
PHASE2_TOKENS_PER_ENTITY_ESTIMATE = int(os.getenv("PHASE2_TOKENS_PER_ENTITY_ESTIMATE", "500"))  # ~tokens per entity in JSON (generous to avoid truncation)
# Max entities per batch: smaller batches = smaller JSON, less truncation (critical for gpt-4.1-nano).
# Default 5 → reduces "salvaged JSON by extracting first object" truncation (was 4; 5 keeps output under max_tokens).
PHASE2_MAX_ENTITIES_PER_BATCH = int(os.getenv("PHASE2_MAX_ENTITIES_PER_BATCH", "5"))
# Lower thresholds to catch more cases as "long" (prevent truncation)
PHASE2_LONG_SOAP_CHARS = int(os.getenv("PHASE2_LONG_SOAP_CHARS", "4000"))  # Was 5000
PHASE2_LONG_MANIFEST_ENTITIES = int(os.getenv("PHASE2_LONG_MANIFEST_ENTITIES", "10"))  # Was 15: use LONG max_tokens when batch has >=10 entities
PHASE2_SCHEMA_CACHE_ENABLE = os.getenv("PHASE2_SCHEMA_CACHE_ENABLE", "true").lower() in ("1", "true", "yes")
PHASE2_SCHEMA_CACHE_TTL_SEC = int(os.getenv("PHASE2_SCHEMA_CACHE_TTL_SEC", "3600"))  # 1h

# Robustness: auto-escalate Phase 2 model only when JSON parsing fails (rare).
PHASE2_ESCALATE_ON_JSON_FAIL = os.getenv("PHASE2_ESCALATE_ON_JSON_FAIL", "true").lower() in ("1", "true", "yes")
# Clinical core only: send only Subjective, Objective, Assessment, Plan to Phase 2 (~25s -> ~12s).
PHASE2_CLINICAL_CORE_ONLY = os.getenv("PHASE2_CLINICAL_CORE_ONLY", "true").lower() in ("1", "true", "yes")
PHASE2_ESCALATE_MODEL = os.getenv("PHASE2_ESCALATE_MODEL", "gpt-4.1-mini")
# Parallel atom extraction: when entity count exceeds this, split into parallel batches (saves ~15s for 23 entities).
PHASE2_PARALLEL_ATOM_ENTITY_THRESHOLD = int(os.getenv("PHASE2_PARALLEL_ATOM_ENTITY_THRESHOLD", "8"))  # Lower threshold: more parallelization
PHASE2_PARALLEL_ATOM_MAX_BATCHES = int(os.getenv("PHASE2_PARALLEL_ATOM_MAX_BATCHES", "5"))  # Increased from 4 to 5 for higher concurrency

_SCHEMA_CACHE_LOCK = threading.Lock()
_SCHEMA_CACHE: dict = {
    "assertion_types": None,
    "attributes_schema": None,
    "cached_at": None,
}


def _set_phase2_timing(timing_ref: Optional[Dict[str, Any]], **kwargs: Any) -> None:
    if timing_ref is not None:
        timing_ref.update(kwargs)


def _elapsed_ms_since(start: float) -> int:
    return max(0, int(round((time.perf_counter() - start) * 1000)))


def _estimate_phase2_max_tokens(soap_chars: int, entity_count: int) -> int:
    """
    Continuous estimator for Phase 2 max_tokens_first.

    - Starts from a small base budget
    - Adds tokens based on SOAP length and number of entities
    - Clamped between PHASE2_MAX_TOKENS (floor) and PHASE2_MAX_TOKENS_LONG (ceiling)

    This replaces the old binary short/long step and makes the first call size
    adapt smoothly to note complexity, reducing truncation/retry risk without
    hardcoding specific thresholds beyond the global caps.
    """
    # Base budget for minimal notes
    base = 800  # prompt + small JSON
    # Scale with SOAP length (very approximate; we mainly care about "small vs huge")
    per_1k_chars = 300
    soap_extra = (soap_chars // 1000) * per_1k_chars
    # Scale with entity count using the existing per-entity estimate
    entities_extra = entity_count * PHASE2_TOKENS_PER_ENTITY_ESTIMATE
    est = base + soap_extra + entities_extra
    floor = PHASE2_MAX_TOKENS
    ceil = PHASE2_MAX_TOKENS_LONG
    return max(floor, min(est, ceil))


# ==============================================================================
# Phase 2 Structured Output Schema (provider-enforced JSON)
# ==============================================================================

# Keep schema permissive on nested objects (attributes/codes) because the exact keys depend on DB schema.
# Goal: enforce valid JSON + required fields so we don't accept malformed responses.
PHASE2_OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    # OpenAI structured outputs (strict) require closed schemas:
    # additionalProperties MUST be explicitly provided and set to false.
    "additionalProperties": False,
    "properties": {
        "knowledge_atoms": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "concept": {"type": "string"},
                    "kind": {"type": "string"},
                    "assertion_id": {"type": "string"},
                    # NOTE: codes (venom/snomed/loinc) are NOT extracted by LLM - they're enriched
                    # post-extraction via _enrich_atoms_with_codes_and_fulfillment_impl() in Phase 2.
                    # Removing from strict schema to avoid OpenAI validation errors.
                    # CRITICAL: attributes_schema is dynamic (varies by kind), so strict schemas
                    # cannot represent arbitrary attribute keys. We model this as KV pairs and
                    # postprocess back into the legacy dict shape.
                    "attributes_kv": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "relationship": {"type": "string"},
                                "value": {
                                    "type": ["string", "number", "boolean", "null"],
                                },
                            },
                            "required": ["relationship", "value"],
                        },
                    },
                    "intent_context": {"type": ["string", "null"]},
                    "source_text": {"type": ["string", "null"]},
                    "section": {"type": ["string", "null"]},
                },
                "required": ["concept", "kind", "assertion_id", "attributes_kv", "intent_context", "source_text", "section"],
            },
        },
        "extraction_summary": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
        "metadata": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
        "error": {"type": ["string", "null"]},
        "parse_error": {"type": ["string", "null"]},
        "raw": {"type": ["string", "null"]},
    },
    "required": [
        "knowledge_atoms",
        "extraction_summary",
        "metadata",
        "error",
        "parse_error",
        "raw",
    ],
}


def _load_phase2_schema_cached(logger: Optional[logging.Logger] = None) -> tuple[list, dict, bool]:
    """
    Load (assertion_types, attributes_schema) with an in-memory cache so we don't hit DB every run.

    Returns: (assertion_types, attributes_schema, cache_hit)
    """
    if not PHASE2_SCHEMA_CACHE_ENABLE:
        with pg_conn_ctx() as conn:
            return (
                get_assertion_types(conn=conn, logger=logger),
                get_all_attributes_schema(conn=conn, logger=logger),
                False,
            )

    now = time.time()
    with _SCHEMA_CACHE_LOCK:
        cached_at = _SCHEMA_CACHE.get("cached_at")
        if (
            cached_at is not None
            and (now - float(cached_at)) < float(PHASE2_SCHEMA_CACHE_TTL_SEC)
            and isinstance(_SCHEMA_CACHE.get("assertion_types"), list)
            and isinstance(_SCHEMA_CACHE.get("attributes_schema"), dict)
        ):
            return _SCHEMA_CACHE["assertion_types"], _SCHEMA_CACHE["attributes_schema"], True

    # Cache miss / expired: reload outside the lock (avoid blocking other work)
    try:
        with pg_conn_ctx() as conn:
            assertion_types = get_assertion_types(conn=conn, logger=logger)
            attributes_schema = get_all_attributes_schema(conn=conn, logger=logger)
    except Exception as e:
        # If DB is temporarily unavailable but cache exists, serve stale.
        if logger:
            logger.warning(f"⚠️ Phase 2 schema load failed; attempting stale cache: {e}")
        with _SCHEMA_CACHE_LOCK:
            if isinstance(_SCHEMA_CACHE.get("assertion_types"), list) and isinstance(_SCHEMA_CACHE.get("attributes_schema"), dict):
                return _SCHEMA_CACHE["assertion_types"], _SCHEMA_CACHE["attributes_schema"], True
        raise

    # Only cache if we got something meaningful
    with _SCHEMA_CACHE_LOCK:
        _SCHEMA_CACHE["assertion_types"] = assertion_types
        _SCHEMA_CACHE["attributes_schema"] = attributes_schema
        _SCHEMA_CACHE["cached_at"] = now
    return assertion_types, attributes_schema, False


def _call_phase2_llm_json(
    client,
    model: str,
    prompt: str,
    provider_name: str,
    max_tokens_first: int,
    temperature: float,
    logger: Optional[logging.Logger],
) -> Dict[str, Any]:
    """
    Call the model and return parsed JSON dict.
    Retries once with higher max_tokens on truncation.

    Why the model can return empty or non-JSON (even with strict JSON / json_schema):
    - For OpenAI gpt-* we use response_format=json_schema (strict); OpenAI enforces it.
      Empty content is then usually: rate limit, content filter, timeout, or refusal.
    - For non-OpenAI (e.g. PHASE2_MODEL=Fireworks Llama) we fall back to json_object;
      that provider may not guarantee JSON or may return empty under load. Using
      gpt-4.1-mini/nano (default) avoids this; if you switched to Llama for Phase 2,
      empty/non-JSON can appear more often.
    """
    last_raw = ""
    last_err = None

    def _extract_first_json_object(text: str) -> str:
        """
        Best-effort extraction of the first complete JSON object from a string.
        Handles nested braces and ignores braces inside strings.
        Also handles unterminated strings by closing them if needed.
        """
        if not isinstance(text, str):
            return ""
        s = text.strip()
        if not s:
            return ""
        start = s.find("{")
        if start < 0:
            return ""
        in_str = False
        esc = False
        depth = 0
        last_valid_pos = -1
        
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
                    elif depth > 0:
                        last_valid_pos = i
        
        # If we reached the end without closing, try to salvage:
        # If we're in a string, close it and try to close braces
        if in_str:
            # Close the string
            s = s[:len(s)] + '"'
            # Try to close remaining braces
            while depth > 0:
                s += "}"
                depth -= 1
            return s[start:]
        
        # If we have unmatched braces but not in a string, try to close them
        if depth > 0:
            result = s[start:]
            while depth > 0:
                result += "}"
                depth -= 1
            return result
        
        return ""

    def _repair_json_simple(json_str: str) -> str:
        """
        Minimal JSON repair for common issues:
        - remove trailing commas before } or ]
        - close unterminated strings
        - remove incomplete key-value pairs at the end
        """
        try:
            import re
            # Remove trailing commas before } or ]
            repaired = re.sub(r",\s*([}\]])", r"\1", json_str)
            
            # If we're in the middle of a string value (unterminated), try to close it
            # Look for patterns like: "key": "value that was cut off
            # This handles truncation mid-string
            if repaired.count('"') % 2 != 0:
                # Odd number of quotes = unterminated string
                # Find the last quote and see if we're in a value
                last_quote = repaired.rfind('"')
                if last_quote > 0:
                    # Check if this looks like a value (has : before it)
                    before_quote = repaired[:last_quote].rstrip()
                    if before_quote.endswith(':') or before_quote.endswith('":'):
                        # Close the string
                        repaired = repaired[:last_quote + 1] + '"'
            
            # Remove incomplete key-value pairs at the end (e.g., "key": "value that was)
            # Look for patterns like: ,"key": "incomplete
            repaired = re.sub(r',\s*"[^"]*":\s*"[^"]*$', '', repaired)
            repaired = re.sub(r',\s*"[^"]*":\s*[^,}\]]+$', '', repaired)
            
            return repaired
        except Exception:
            return json_str

    def _parse_or_salvage(raw: str) -> Optional[Dict[str, Any]]:
        """Parse raw JSON; on failure try extract/repair. Returns dict or None. Empty response returns None (not 0 atoms)."""
        nonlocal last_err
        if not isinstance(raw, str) or not raw.strip():
            last_err = ValueError("Empty model response")
            return None
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as e:
            last_err = e
            if logger and ("Unterminated string" in str(e) or "Expecting" in str(e)):
                logger.warning(f"  ⚠️ Phase 2: JSON parse error (likely truncation): {str(e)[:200]}")
            try:
                extracted = _extract_first_json_object(raw or "")
                if extracted:
                    try:
                        obj2 = json.loads(extracted)
                        if isinstance(obj2, dict):
                            if logger:
                                logger.warning("  ⚠️ Phase 2: salvaged JSON by extracting first object from response text")
                            return obj2
                    except json.JSONDecodeError:
                        repaired = _repair_json_simple(extracted)
                        try:
                            obj3 = json.loads(repaired)
                            if isinstance(obj3, dict):
                                if logger:
                                    logger.warning("  ⚠️ Phase 2: salvaged JSON after minimal repair")
                                return obj3
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
        except Exception as e:
            last_err = e
        return None

    max_tokens_options = [max_tokens_first]
    if provider_name == "openai" and isinstance(model, str) and ("gpt-4.1-nano" in model or "gpt-4.1-mini" in model):
        retry_mt = min(int(max_tokens_first * 2.5), 8000)
        if retry_mt > max_tokens_first:
            max_tokens_options.append(retry_mt)

    for i, mt in enumerate(max_tokens_options, start=1):
        if logger and i > 1:
            logger.warning(f"  ⚠️ Phase 2: retrying once with higher max_tokens={mt} (previous JSON parse failed)")

        response_format = {"type": "json_object"}
        # Strongest guarantee for OpenAI: strict json_schema structured outputs.
        try:
            if provider_name == "openai" and isinstance(model, str) and model.startswith("gpt-"):
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Phase2KnowledgeAtoms",
                        "schema": PHASE2_OUTPUT_JSON_SCHEMA,
                        "strict": True,
                    },
                }
        except Exception:
            response_format = {"type": "json_object"}

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return JSON only. No prose, no markdown, no <think>."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=mt,
            response_format=response_format,
        )

        raw = ""
        try:
            choice = resp.choices[0] if resp and resp.choices else None
            raw = choice.message.content if choice else ""
            if not raw and choice and logger:
                fr = getattr(choice, "finish_reason", None)
                logger.warning(
                    "  ⚠️ Phase 2: model returned empty content (finish_reason=%s). "
                    "Common causes: content_filter, length/truncation, rate_limit; or non-OpenAI model (e.g. Fireworks) may not enforce JSON.",
                    fr,
                )
        except Exception:
            raw = ""
        last_raw = raw or ""

        result = _parse_or_salvage(raw)
        if result is not None:
            return result
        # Log retry/salvage hint for truncation
        if logger and i < len(max_tokens_options):
            logger.info(f"  🔄 Will retry with higher max_tokens={max_tokens_options[i]}")
        elif logger:
            logger.warning("  ⚠️ Phase 2: All retry attempts exhausted, attempting JSON salvage")

    # Failed all attempts
    return {
        "error": "Invalid JSON from Phase 2 model",
        "raw": (last_raw[:8000] if isinstance(last_raw, str) else str(last_raw)[:8000]),
        "knowledge_atoms": [],
        "parse_error": str(last_err),
    }


def _looks_like_json_parse_failure(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return True
    err = (obj.get("error") or "").lower()
    if "invalid json from phase 2 model" in err:
        return True
    # Safety: treat any error + empty payload as failure.
    if err and not obj.get("knowledge_atoms"):
        return True
    return False


def _phase2_kind_candidates_from_manifest(
    entity_manifest: List[Dict[str, Any]],
    *,
    schema_kinds: Optional[set[str]] = None,
) -> set[str]:
    """
    Production-safe mapping from Phase 1 manifest kinds to Phase 2 schema kinds.

    Why this exists:
    - Phase 1 kinds should be a stable enum (via `kb_ner_routing.canonicalize_kind()`), but we cannot assume
      every upstream path normalized perfectly.
    - Phase 2 schema kinds are DB-backed (`kb.attributes_schema.source_kind`) and can vary by deployment.

    Strategy:
    1) Canonicalize Phase 1 kinds (best-effort) to reduce drift (Diagnosis/Disease/Treatment/etc.).
    2) Map directly if the normalized kind matches a normalized DB schema kind.
    3) Otherwise apply a small synonym bridge (Drug→Medicine, Nutrition→Diet, FollowUp→Follow-up, etc.).
    4) If still unknown: do not break production. Warn only if PHASE2_WARN_UNMAPPED_KINDS=true, or fail
       only if PHASE2_STRICT_KIND_MAP=true.
    """
    # Best-effort canonicalizer (avoid hard dependency / circular import)
    try:
        from kb_ner_routing import canonicalize_kind as _canon_kind  # type: ignore
    except Exception:
        _canon_kind = None  # type: ignore

    # Synonym bridge (Phase 1 canonical → Phase 2 schema kind)
    # NOTE: In this deployment Phase 2 schema includes Diet + ReasonForVisit (not Nutrition/Reason).
    PHASE1_TO_PHASE2_KIND: Dict[str, str] = {
        # Billable / dual_sync
        "Drug": "Medicine",
        "Nutrition": "Diet",
        "Procedure": "Procedure",
        "Device": "Device",
        "Vaccine": "Medicine",  # vaccine behaves drug-like in schema
        "Service": "Procedure",
        "LabTest": "LabTest",
        "DiagnosticTest": "DiagnosticTest",
        "Condition": "Condition",
        "ReasonForVisit": "ReasonForVisit",
        # Clinical / global_direct
        "Finding": "Finding",
        "Observation": "Finding",  # schema may not have Observation
        "Anatomy": "Anatomy",
        "Organism": "Organism",
        "Symptom": "Symptom",
        # Admin / reminders
        "Reminder": "Reminder",
        "FollowUp": "Follow-up",
        # Common drift variants
        "Diagnosis": "Condition",
        "Disease": "Condition",
        "Treatment": "Procedure",
        "Therapy": "Procedure",
        # Phase 2-native variants / synonyms we might see
        "VitalSign": "VitalSign",
        "Reason": "ReasonForVisit",
        "Medicine": "Medicine",
        "Medication": "Medicine",
        "Substance": "Medicine",
        "Diagnostic": "DiagnosticTest",
    }

    def _norm_kind(k: str) -> str:
        if not k:
            return ""
        k2 = str(k).strip()
        k2 = k2.replace("-", "").replace("_", "").replace(" ", "")
        return k2.lower()

    canon_lookup: Dict[str, str] = {_norm_kind(k): k for k in PHASE1_TO_PHASE2_KIND.keys()}

    # Normalized lookup of actual DB schema kinds (recommended)
    schema_norm_lookup: Dict[str, str] = {}
    if schema_kinds:
        for sk in schema_kinds:
            if not sk:
                continue
            schema_norm_lookup[_norm_kind(sk)] = sk

    def _map_to_schema(kind_in: str) -> Optional[str]:
        if not kind_in:
            return None
        k_raw = str(kind_in).strip()
        if not k_raw:
            return None

        # Canonicalize best-effort (reduces Diagnosis/Disease/Treatment drift)
        k_canon = None
        try:
            if _canon_kind is not None:
                k_canon = _canon_kind(k_raw)
        except Exception:
            k_canon = None

        candidates = [k_raw]
        if isinstance(k_canon, str) and k_canon and k_canon not in candidates:
            candidates.append(k_canon)

        # (1) Direct normalized match to schema kinds
        if schema_norm_lookup:
            for cand in candidates:
                n = _norm_kind(cand)
                if n in schema_norm_lookup:
                    return schema_norm_lookup[n]

        # (2) Synonym bridge then re-check schema presence
        for cand in candidates:
            key = canon_lookup.get(_norm_kind(cand))
            if not key:
                continue
            mapped = PHASE1_TO_PHASE2_KIND.get(key)
            if not mapped:
                continue
            if schema_norm_lookup:
                mn = _norm_kind(mapped)
                if mn in schema_norm_lookup:
                    return schema_norm_lookup[mn]
                # If mapped kind doesn't exist in schema, don't force it.
                return None
            return mapped

        return None

    out: set[str] = set()
    unmapped: set[str] = set()
    for e in entity_manifest or []:
        if not isinstance(e, dict):
            continue
        k = (e.get("kind") or e.get("kb_kind") or "").strip()
        if not k:
            continue
        mapped = _map_to_schema(k)
        if mapped:
            out.add(mapped)
        else:
            unmapped.add(k)

    if unmapped:
        strict = os.getenv("PHASE2_STRICT_KIND_MAP", "false").lower() in ("1", "true", "yes")
        warn = os.getenv("PHASE2_WARN_UNMAPPED_KINDS", "false").lower() in ("1", "true", "yes")
        msg = f"Phase 2 kind-map: unmapped manifest kinds encountered: {sorted(unmapped)[:20]}"
        if strict:
            raise ValueError(msg)
        if warn:
            try:
                logging.getLogger(__name__).warning("⚠️ " + msg)
            except Exception:
                pass

    # Always include high-yield kinds (only if they exist in schema_kinds when provided)
    always = {"Medicine", "Procedure", "ReasonForVisit"}
    if schema_norm_lookup:
        for a in always:
            n = _norm_kind(a)
            if n in schema_norm_lookup:
                out.add(schema_norm_lookup[n])
    else:
        out.update(always)

    return out


def _compact_manifest_for_prompt(entity_manifest: List[Dict[str, Any]], *, max_entities: int = 250) -> str:
    """
    Reduce manifest prompt size: keep only fields useful for grounding/anchoring.
    """
    compact: List[Dict[str, Any]] = []
    for e in (entity_manifest or [])[:max_entities]:
        if not isinstance(e, dict):
            continue
        compact.append({
            "entity_id": e.get("entity_id") or e.get("id"),
            "span_text": e.get("span_text"),
            "normalized_name": e.get("normalized_name"),
            "display_name": e.get("display_name"),
            "kb_preferred_name": e.get("kb_preferred_name"),
            "kind": e.get("kind"),
            "kb_kind": e.get("kb_kind"),
            "kb_concept_id": e.get("kb_concept_id") or e.get("concept_id"),
            "local_stock_id": e.get("local_stock_id"),
            "local_service_id": e.get("local_service_id"),
            "match_method": e.get("match_method"),
        })
    try:
        return json.dumps(compact, ensure_ascii=False)
    except Exception:
        return "[]"


def _compact_attributes_schema(
    full_schema: Dict[str, List[Dict[str, Any]]],
    *,
    only_kinds: set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reduce schema prompt size: only include relevant kinds, and drop verbose use_case text.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for kind, rows in (full_schema or {}).items():
        if kind not in only_kinds:
            continue
        kept: List[Dict[str, Any]] = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            kept.append({
                "relationship": r.get("relationship"),
                "target_attribute": r.get("target_attribute"),
                "use_case": None,  # drop examples for prompt size
                "is_required": bool(r.get("is_required")),
            })
        out[kind] = kept
    return out


def _deduplicate_knowledge_atoms(atoms_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ID-first de-duplication for Phase 2 atoms.
    Priority:
    1) local_service_id / local_stock_id (source-of-truth; merge regardless of text variation)
    2) explicit dedup_key (when model provides a stable clinical normalized form)
    3) same meta-kind + high concept similarity fallback
    Section priority for representative atom: Plan > Assessment > Subjective > Objective.
    """
    if not atoms_list:
        return atoms_list
    _SECTION_PRIORITY = {"Plan": 0, "Assessment": 1, "Subjective": 2, "Objective": 3}

    def _norm_text(text: str) -> str:
        if not text:
            return ""
        t = re.sub(r"[^\w\s]", " ", str(text).lower()).strip()
        t = re.sub(r"\s+", " ", t)
        return t

    def _norm_levenshtein_ratio(a: str, b: str) -> float:
        a = _norm_text(a)
        b = _norm_text(b)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        n, m = len(a), len(b)
        prev = list(range(m + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i] + [0] * m
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            prev = cur
        dist = prev[m]
        return max(0.0, 1.0 - (float(dist) / float(max(n, m, 1))))

    def _meta_kind(kind: str) -> str:
        k = (kind or "").strip().lower()
        if k in {"medicine", "drug", "medication", "substance", "supplement", "nutrition", "vaccine", "preventive", "parasitecontrol"}:
            return "PHARMACOLOGICAL"
        if k in {"diagnostic", "labtest", "diagnostictest", "imaging"}:
            return "DIAGNOSTIC"
        if k in {"procedure", "service", "treatment"}:
            return "INTERVENTIONAL"
        if k in {"vitalsign", "vital"}:
            return "VITAL"
        if k in {"reminder", "followup", "follow-up"}:
            return "REMINDER"
        if k in {"reasonforvisit", "reason"}:
            return "RFV"
        return k or "OTHER"

    def _pri(a: Dict[str, Any]) -> Tuple[int, int]:
        sec = (a.get("section") or "").strip()
        has_id = bool(
            a.get("local_service_id")
            or a.get("local_stock_id")
            or (isinstance(a.get("codes"), dict) and (a["codes"].get("local_service_id") or a["codes"].get("local_stock_id")))
        )
        return (_SECTION_PRIORITY.get(sec, 99), 0 if has_id else 1)

    def _merge_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = min(group, key=_pri).copy()
        merged_sources: List[str] = []
        if best.get("source_text"):
            merged_sources.append(str(best.get("source_text")).strip())
        for other in group:
            if other is best:
                continue
            for id_key in ("local_service_id", "local_stock_id", "kb_concept_id"):
                if other.get(id_key) and not best.get(id_key):
                    best[id_key] = other[id_key]
            codes = best.get("codes") if isinstance(best.get("codes"), dict) else {}
            other_codes = other.get("codes") if isinstance(other.get("codes"), dict) else {}
            for id_key in ("local_service_id", "local_stock_id"):
                if other.get(id_key) and not codes.get(id_key):
                    codes[id_key] = other[id_key]
                if other_codes.get(id_key) and not codes.get(id_key):
                    codes[id_key] = other_codes.get(id_key)
            best["codes"] = codes
            src = (other.get("source_text") or "").strip()
            if src and src not in merged_sources:
                merged_sources.append(src)
            # If this atom has stronger grounding, prefer its kind as metadata.
            if (other.get("local_service_id") or other.get("local_stock_id")) and not (best.get("local_service_id") or best.get("local_stock_id")):
                best["kind"] = other.get("kind") or best.get("kind")
        if merged_sources:
            best["source_text"] = " | ".join(merged_sources)
        return best

    groups_by_id: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    leftovers: List[Dict[str, Any]] = []
    for atom in atoms_list:
        if not isinstance(atom, dict):
            continue
        codes = atom.get("codes") if isinstance(atom.get("codes"), dict) else {}
        sid = atom.get("local_service_id") or codes.get("local_service_id")
        stk = atom.get("local_stock_id") or codes.get("local_stock_id")
        if sid:
            groups_by_id.setdefault(("svc", str(sid)), []).append(atom)
            continue
        if stk:
            groups_by_id.setdefault(("stk", str(stk)), []).append(atom)
            continue
        leftovers.append(atom)

    out: List[Dict[str, Any]] = []
    for group in groups_by_id.values():
        out.append(_merge_group(group))

    groups_by_dedup_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    no_key: List[Dict[str, Any]] = []
    for atom in leftovers:
        dkey = _norm_text(atom.get("dedup_key") or "")
        if dkey:
            mk = _meta_kind(atom.get("kind") or "")
            groups_by_dedup_key.setdefault((mk, dkey), []).append(atom)
        else:
            no_key.append(atom)
    for group in groups_by_dedup_key.values():
        out.append(_merge_group(group))

    # Final fallback: same meta-kind + high similarity
    clusters: List[Dict[str, Any]] = []
    for atom in no_key:
        concept = atom.get("concept") or atom.get("normalized_name") or atom.get("source_text") or ""
        mk = _meta_kind(atom.get("kind") or "")
        placed = False
        for c in clusters:
            if c["meta_kind"] != mk:
                continue
            if _norm_levenshtein_ratio(concept, c["rep_concept"]) >= 0.90:
                c["items"].append(atom)
                placed = True
                break
        if not placed:
            clusters.append({"meta_kind": mk, "rep_concept": concept, "items": [atom]})
    for c in clusters:
        out.append(_merge_group(c["items"]))

    return out


def _apply_knowledge_atom_constraints(
    atoms_list: List[Dict[str, Any]],
    entity_manifest: List[Dict[str, Any]],
    soap_note_text: str,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Constraint injection (disabled): atoms are returned unchanged.
    Previously dropped ReasonForVisit/Diagnosis unlinked+not in Plan, unlinked atoms not in Plan (except vitals/reminders), and customer-instruction-only items; that filtering has been removed.
    """
    return atoms_list if atoms_list else []


def _enrich_atoms_with_manifest_ids(
    atoms_list: List[Dict[str, Any]],
    entity_manifest: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Enrich knowledge atoms with inventory/service IDs from entity manifest.
    Matches atoms to manifest entries by concept name and kind with improved fuzzy matching.
    """
    if not entity_manifest:
        return atoms_list
    
    # Build multiple lookup strategies
    manifest_by_exact: Dict[tuple, Dict[str, Any]] = {}  # (name_lower, kind) -> entity
    manifest_by_name: Dict[str, List[Dict[str, Any]]] = {}  # name_lower -> [entities]
    manifest_by_kind: Dict[str, List[Dict[str, Any]]] = {}  # kind -> [entities]
    manifest_by_entity_id: Dict[str, Dict[str, Any]] = {}  # explicit manifest entity id -> entity
    manifest_by_service_id: Dict[str, Dict[str, Any]] = {}  # local_service_id -> entity
    manifest_by_stock_id: Dict[str, Dict[str, Any]] = {}  # local_stock_id -> entity
    # ID-based linking: (normalized_name_lower, kind) -> entity for entries that have local_stock_id or local_service_id.
    # Use this as primary key for dashboard linking so "Cefpodoxime syrup" (atom) matches manifest normalized_name "Cefpodoxime Syrup" and gets that entry's local_stock_id, avoiding display_name string mismatch orphans.
    manifest_by_normalized_name_id: Dict[tuple, Dict[str, Any]] = {}  # (norm_name_lower, kind) -> entity (only ID-bearing)

    for entity in entity_manifest:
        if not isinstance(entity, dict):
            continue
        
        # Index by normalized_name, span_text, display_name, kb_preferred_name (all four for exact and name-based lookups).
        # - normalized_name may be a canonical/candidate name (e.g., "COATEX BLISTER CAPSULE")
        # - span_text is what the user/ASR said (e.g., "Cortex capsule")
        # - display_name / kb_preferred_name are grounded inventory/service names (e.g., "EASOTIC 10ML")
        norm_name = (entity.get("normalized_name") or "").strip()
        span_name = (entity.get("span_text") or "").strip()
        display_name = (entity.get("display_name") or "").strip()
        kb_preferred_name = (entity.get("kb_preferred_name") or "").strip()
        names_to_index = [n for n in (norm_name, span_name, display_name, kb_preferred_name) if isinstance(n, str) and n.strip()]
        # CRITICAL: for local ID stitching we must prefer the Phase 1 entity kind (Procedure/Nutrition/Drug/etc.)
        # NOT the KB kind (e.g., "Substance") which is often different and breaks kind compatibility.
        kind = (entity.get("kind") or entity.get("kb_kind") or "").strip()
        kb_kind = (entity.get("kb_kind") or "").strip()
        ent_id = entity.get("entity_id") or entity.get("id")
        if ent_id is not None and str(ent_id).strip():
            manifest_by_entity_id[str(ent_id).strip()] = entity
        if entity.get("local_service_id") is not None:
            manifest_by_service_id[str(entity.get("local_service_id"))] = entity
        if entity.get("local_stock_id") is not None:
            manifest_by_stock_id[str(entity.get("local_stock_id"))] = entity
        # Primary key for ID-based dashboard linking: index by normalized_name only, for ID-bearing entries
        if (entity.get("local_stock_id") is not None or entity.get("local_service_id") is not None) and norm_name:
            norm_lower = norm_name.lower()
            if kind and norm_lower:
                manifest_by_normalized_name_id[(norm_lower, kind)] = entity
        
        for nm in names_to_index:
            nm_lower = nm.lower()
            if not nm_lower:
                continue
            # Exact match lookup
            if kind:
                manifest_by_exact[(nm_lower, kind)] = entity
            # Name-based lookup (for fuzzy matching)
            if nm_lower not in manifest_by_name:
                manifest_by_name[nm_lower] = []
            manifest_by_name[nm_lower].append(entity)
        # Kind-based lookup
        if kind:
            if kind not in manifest_by_kind:
                manifest_by_kind[kind] = []
            manifest_by_kind[kind].append(entity)
    
    def normalize_text(text: str) -> str:
        """Normalize text for matching (lowercase, remove extra spaces/punctuation)"""
        if not text:
            return ""
        import re
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text
    
    def kind_compatible(atom_kind: str, manifest_kind: str) -> bool:
        """Check if kinds are compatible for matching"""
        kind_map = {
            # Drug-like (Phase 1 often uses Drug; Phase 2 often uses Medicine)
            'Medicine': ['Drug', 'Medicine', 'Medication', 'Substance', 'Supplement', 'Vaccine', 'Nutrition', 'Preventive', 'ParasiteControl'],
            'Drug': ['Drug', 'Medicine', 'Medication', 'Substance', 'Supplement', 'Vaccine', 'Nutrition', 'Preventive', 'ParasiteControl'],
            'Medication': ['Drug', 'Medicine', 'Medication', 'Substance', 'Supplement', 'Vaccine', 'Nutrition', 'Preventive', 'ParasiteControl'],
            'Substance': ['Drug', 'Medicine', 'Medication', 'Substance', 'Supplement', 'Vaccine', 'Nutrition', 'Preventive', 'ParasiteControl'],
            'Supplement': ['Drug', 'Medicine', 'Medication', 'Substance', 'Supplement', 'Vaccine', 'Nutrition', 'Preventive', 'ParasiteControl'],
            'Vaccine': ['Drug', 'Medicine', 'Medication', 'Substance', 'Supplement', 'Vaccine', 'Preventive', 'ParasiteControl'],
            'Nutrition': ['Drug', 'Medicine', 'Medication', 'Substance', 'Supplement', 'Nutrition'],
            'Preventive': ['Drug', 'Medicine', 'Medication', 'Substance', 'Supplement', 'Vaccine', 'Preventive', 'ParasiteControl'],
            'ParasiteControl': ['Drug', 'Medicine', 'Medication', 'Substance', 'Supplement', 'Vaccine', 'Preventive', 'ParasiteControl'],
            # Procedure-like
            'Procedure': ['Procedure', 'Service', 'Treatment', 'Diagnostic'],
            'Service': ['Procedure', 'Service', 'Treatment', 'Diagnostic'],
            'Treatment': ['Procedure', 'Service', 'Treatment', 'Diagnostic'],
            # Diagnostics (often modeled as Procedure in masters)
            'Diagnostic': ['Diagnostic', 'LabTest', 'DiagnosticTest', 'Procedure', 'Service'],
            'LabTest': ['Diagnostic', 'LabTest', 'DiagnosticTest', 'Procedure', 'Service'],
            'DiagnosticTest': ['Diagnostic', 'LabTest', 'DiagnosticTest', 'Procedure', 'Service'],
        }
        if atom_kind == manifest_kind:
            return True
        compatible = kind_map.get(atom_kind, [])
        return manifest_kind in compatible

    def _norm_levenshtein_ratio(a: str, b: str) -> float:
        """
        Normalized similarity ratio in [0,1] using Levenshtein distance.
        Small, dependency-free implementation for stitching fallback only.
        """
        a = normalize_text(a)
        b = normalize_text(b)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        # DP edit distance (O(n*m)), strings are short here.
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return 0.0
        prev = list(range(m + 1))
        for i, ca in enumerate(a, start=1):
            cur = [i] + [0] * m
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                cur[j] = min(
                    prev[j] + 1,      # deletion
                    cur[j - 1] + 1,   # insertion
                    prev[j - 1] + cost,  # substitution
                )
            prev = cur
        dist = prev[m]
        return max(0.0, 1.0 - (float(dist) / float(max(n, m, 1))))

    def _form_overlap(a: str, b: str) -> bool:
        a_l = (a or "").lower()
        b_l = (b or "").lower()
        forms = ["capsule", "capsules", "tablet", "tablets", "tab", "tabs", "syrup", "inj", "injection", "drops"]
        # Require at least one form word present on both sides to avoid accidental stitches.
        return any(f in a_l for f in forms) and any(f in b_l for f in forms)
    
    enriched = []
    for atom in atoms_list:
        if not isinstance(atom, dict):
            enriched.append(atom)
            continue
        
        atom_copy = atom.copy()
        concept = (atom.get("concept") or "").strip()
        concept_lower = concept.lower()
        concept_normalized = normalize_text(concept)
        kind = (atom.get("kind") or "").strip()
        
        match = None

        # Strategy 0a: explicit referenced_entity_id from prompt/model output
        ref_ent_id = atom.get("referenced_entity_id") or atom.get("entity_id")
        if ref_ent_id is not None:
            match = manifest_by_entity_id.get(str(ref_ent_id).strip())

        # Strategy 0b: ID-dominant inheritance from already-stitched local ids
        if not match:
            codes = atom.get("codes") if isinstance(atom.get("codes"), dict) else {}
            sid = atom.get("local_service_id") or codes.get("local_service_id")
            stk = atom.get("local_stock_id") or codes.get("local_stock_id")
            if sid is not None:
                match = manifest_by_service_id.get(str(sid))
            if not match and stk is not None:
                match = manifest_by_stock_id.get(str(stk))
        
        # Strategy 1: Exact match (name + kind)
        if not match and kind:
            match = manifest_by_exact.get((concept_lower, kind))
        
        # Strategy 1b: ID-based linking by manifest normalized_name (primary key for dashboard)
        # Reduces orphans when atom concept matches Brain NER normalized_name but manifest display_name differs (e.g. "Cefpodoxime syrup" vs "CefPET Dry Syrup").
        if not match and kind and concept_lower:
            match = manifest_by_normalized_name_id.get((concept_lower, kind))
            if not match:
                for (norm_lower, m_kind), ent in manifest_by_normalized_name_id.items():
                    if norm_lower == concept_lower and kind_compatible(kind, m_kind):
                        match = ent
                        break
        
        # Strategy 2: Exact name match, compatible kind
        if not match:
            candidates = manifest_by_name.get(concept_lower, [])
            for candidate in candidates:
                m_kind = (candidate.get("kb_kind") or candidate.get("kind") or "").strip()
                if kind_compatible(kind, m_kind):
                    match = candidate
                    break
        
        # Strategy 3: Fuzzy name match (substring or contains)
        if not match:
            for m_name, candidates in manifest_by_name.items():
                if not m_name:
                    continue
                # Check if concepts overlap significantly
                if (concept_normalized in normalize_text(m_name) or 
                    normalize_text(m_name) in concept_normalized or
                    len(set(concept_normalized.split()) & set(normalize_text(m_name).split())) >= 2):
                    for candidate in candidates:
                        m_kind = (candidate.get("kb_kind") or candidate.get("kind") or "").strip()
                        if kind_compatible(kind, m_kind):
                            match = candidate
                            break
                    if match:
                        break
        
        # Strategy 3b: Distinctive-token match for ID-bearing manifest (e.g. "Easotic ear drops" -> "EASOTIC 10ML" via token "easotic")
        if not match and concept_normalized and kind:
            _stop = {"the", "and", "for", "with", "from", "ear", "eye", "drops", "tablet", "capsule", "syrup", "injection", "mg", "ml", "tab", "caps"}
            concept_words = [w for w in concept_normalized.split() if len(w) >= 4 and w not in _stop]
            if not concept_words and concept_normalized.split():
                concept_words = [w for w in concept_normalized.split() if len(w) >= 3 and w not in _stop]
            concept_tokens = set(concept_words) if concept_words else set()
            for ent in entity_manifest or []:
                if not isinstance(ent, dict):
                    continue
                if not (ent.get("local_stock_id") or ent.get("local_service_id")):
                    continue
                m_kind = (ent.get("kind") or ent.get("kb_kind") or "").strip()
                if m_kind and (not kind_compatible(kind, m_kind)):
                    continue
                m_names = [
                    (ent.get("normalized_name") or "").strip(),
                    (ent.get("span_text") or "").strip(),
                    (ent.get("display_name") or "").strip(),
                    (ent.get("kb_preferred_name") or "").strip(),
                ]
                manifest_tokens = set()
                for mn in m_names:
                    if not mn:
                        continue
                    manifest_tokens.update(w for w in normalize_text(mn).split() if len(w) >= 3 and w not in _stop)
                if concept_tokens and manifest_tokens and (concept_tokens & manifest_tokens):
                    match = ent
                    break
        
        # Strategy 4: Kind-only match by NAME only (never assign ID by kind alone)
        # Previously we fell back to "first candidate with ID" which caused all Procedure atoms
        # (physiotherapy, tick/flea, weight diet, X-ray) to get the same service_id (e.g. XRAY).
        if not match and kind:
            candidates = manifest_by_kind.get(kind, [])
            candidates_with_ids = [c for c in candidates if c.get("local_stock_id") or c.get("local_service_id")]
            if candidates_with_ids:
                for candidate in candidates_with_ids:
                    m_name = normalize_text(candidate.get("normalized_name") or candidate.get("span_text") or "")
                    if m_name and (concept_normalized in m_name or m_name in concept_normalized):
                        match = candidate
                        break
                # Do NOT fall back to candidates_with_ids[0] — prevents over-attribution to one service.

        # Strategy 5: Fuzzy name stitch (safe): when atom has no IDs yet, try best local-ID manifest candidate
        # within compatible kinds using a conservative normalized Levenshtein ratio + form overlap.
        if not match and concept_normalized and kind:
            scored: list[tuple[float, Dict[str, Any]]] = []
            for ent in entity_manifest or []:
                if not isinstance(ent, dict):
                    continue
                if not (ent.get("local_stock_id") or ent.get("local_service_id")):
                    continue
                m_kind = (ent.get("kind") or ent.get("kb_kind") or "").strip()
                if m_kind and (not kind_compatible(kind, m_kind)):
                    continue
                m_name = (ent.get("normalized_name") or ent.get("span_text") or "").strip()
                if not m_name:
                    continue
                if not _form_overlap(concept, m_name):
                    continue
                s = _norm_levenshtein_ratio(concept, m_name)
                scored.append((s, ent))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored:
                best_s, best_ent = scored[0]
                second_s = scored[1][0] if len(scored) > 1 else 0.0
                # Conservative acceptance: high similarity + clear margin
                if best_s >= 0.72 and (best_s - second_s) >= 0.08:
                    match = best_ent
        
        if match:
            # Enrich with IDs from manifest
            if match.get("local_stock_id"):
                atom_copy["local_stock_id"] = match["local_stock_id"]
                # Also mirror into codes for downstream consumers that look there.
                if isinstance(atom_copy.get("codes"), dict):
                    atom_copy["codes"].setdefault("local_stock_id", match["local_stock_id"])
            if match.get("local_service_id"):
                atom_copy["local_service_id"] = match["local_service_id"]
                if isinstance(atom_copy.get("codes"), dict):
                    atom_copy["codes"].setdefault("local_service_id", match["local_service_id"])
            if match.get("kb_concept_id"):
                atom_copy["kb_concept_id"] = match["kb_concept_id"]
            if match.get("kb_kind"):
                atom_copy["kb_kind"] = match["kb_kind"]
            # Grounding is source-of-truth: when stitched, inherit manifest kind metadata.
            if match.get("kind"):
                atom_copy["kind"] = match["kind"]
            if match.get("entity_id") or match.get("id"):
                atom_copy["referenced_entity_id"] = match.get("entity_id") or match.get("id")
            if match.get("match_method"):
                atom_copy["match_method"] = match["match_method"]
            # Assertion and attributes from consolidated Brain NER (manifest as source of truth)
            if match.get("assertion_id"):
                atom_copy["assertion_id"] = match["assertion_id"]
            match_attrs_manifest = match.get("attributes") or {}
            if isinstance(match_attrs_manifest, dict) and match_attrs_manifest:
                attrs = atom_copy.get("attributes") or {}
                if not isinstance(attrs, dict):
                    attrs = {}
                for k, v in match_attrs_manifest.items():
                    if v is not None and v != "" and (k not in attrs or not attrs.get(k)):
                        attrs[k] = v
                atom_copy["attributes"] = attrs
            
            # Preserve suggestions from manifest attributes (for unlinked entities)
            match_attrs = match.get("attributes", {}) or {}
            if isinstance(match_attrs, dict) and match_attrs.get("suggestions"):
                # Merge suggestions into atom attributes
                atom_attrs = atom_copy.get("attributes", {}) or {}
                if not isinstance(atom_attrs, dict):
                    atom_attrs = {}
                atom_attrs["suggestions"] = match_attrs["suggestions"]
                atom_copy["attributes"] = atom_attrs
        else:
            # No exact match found - try fuzzy matching for suggestions only
            # This handles cases where Phase 2 extracts "X-ray" but Phase 1 had "X-ray imaging"
            atom_attrs = atom_copy.get("attributes", {}) or {}
            if not isinstance(atom_attrs, dict):
                atom_attrs = {}
            
            # Try to find suggestions by fuzzy name matching (even if no ID match)
            best_suggestion_match = None
            best_similarity = 0.0
            
            for entity in entity_manifest:
                if not isinstance(entity, dict):
                    continue
                # Only consider unlinked entities (they have suggestions)
                if entity.get("local_stock_id") or entity.get("local_service_id"):
                    continue  # Skip linked entities
                
                entity_attrs = entity.get("attributes", {}) or {}
                if not isinstance(entity_attrs, dict) or not entity_attrs.get("suggestions"):
                    continue  # Skip entities without suggestions
                
                # Try to match by concept name similarity
                entity_names = [
                    entity.get("span_text", "").lower(),
                    entity.get("normalized_name", "").lower(),
                    entity.get("display_name", "").lower(),
                ]
                entity_names = [n for n in entity_names if n]
                
                for entity_name in entity_names:
                    if not entity_name:
                        continue
                    # Check if concepts overlap significantly
                    similarity = _norm_levenshtein_ratio(concept_normalized, normalize_text(entity_name))
                    if similarity > best_similarity and similarity >= 0.5:  # Threshold for fuzzy match
                        best_similarity = similarity
                        best_suggestion_match = entity
            
            # If we found a fuzzy match with suggestions, preserve them
            if best_suggestion_match:
                match_attrs = best_suggestion_match.get("attributes", {}) or {}
                if isinstance(match_attrs, dict) and match_attrs.get("suggestions"):
                    atom_attrs["suggestions"] = match_attrs["suggestions"]
                    atom_copy["attributes"] = atom_attrs
        
        enriched.append(atom_copy)
    
    return enriched


def _postprocess_phase2_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic cleanup for Phase 2 output:
    - Ensure `knowledge_atoms` is a list[dict]
    - Ensure required keys exist (best-effort defaults)
    - Recompute `extraction_summary` so counts never contradict the array length
    """
    if not isinstance(obj, dict):
        return {"knowledge_atoms": [], "error": "phase2_output_not_dict"}

    atoms = obj.get("knowledge_atoms")
    if not isinstance(atoms, list):
        atoms = []

    cleaned_atoms: List[Dict[str, Any]] = []
    for a in atoms:
        if not isinstance(a, dict):
            continue
        a.setdefault("concept", a.get("concept") or "")
        a.setdefault("kind", a.get("kind") or "")
        a.setdefault("assertion_id", a.get("assertion_id") or "CONF")
        # Accept either legacy dict attributes OR schema-safe KV pairs (attributes_kv)
        attrs: Dict[str, Any] = a.get("attributes") if isinstance(a.get("attributes"), dict) else {}
        if not attrs:
            kv = a.get("attributes_kv")
            if isinstance(kv, list):
                out: Dict[str, Any] = {}
                for item in kv:
                    if not isinstance(item, dict):
                        continue
                    k = (item.get("relationship") or "").strip()
                    if not k:
                        continue
                    out[k] = item.get("value")
                attrs = out
        a["attributes"] = attrs
        # Drop KV form to keep downstream stable
        if "attributes_kv" in a:
            try:
                del a["attributes_kv"]
            except Exception:
                pass
        # Initialize codes as empty dict if missing (will be enriched post-extraction)
        if "codes" not in a or not isinstance(a.get("codes"), dict):
            a["codes"] = {}
        a.setdefault("intent_context", a.get("intent_context") or a.get("intent_type") or "")
        a.setdefault("source_text", a.get("source_text") or "")
        a.setdefault("section", a.get("section") or "")
        cleaned_atoms.append(a)

    # CRITICAL: Filter out non-billing kinds (Identity, Signalment, Anatomy, Symptom, etc.)
    # Knowledge atoms and verification dashboard are ONLY for billing items
    # Based on Phase 2 prompt: Only extract Procedures, Services, Medications, Diagnostics, Vitals, Reminders, ReasonForVisit
    BILLING_ONLY_KINDS = {
        "Procedure", "Service", "Treatment",
        "Medicine", "Drug", "Medication", "Substance", "Vaccine", "Supplement", "Nutrition",
        "LabTest", "DiagnosticTest", "Diagnostic", "Imaging",
        "VitalSign", "Vital",
        "Reminder", "FollowUp", "Follow-up",
        "ReasonForVisit", "Reason",
        "ParasiteControl", "Preventive",
    }
    # Non-billing kinds to exclude (Identity, Signalment, Anatomy, Symptom, etc.)
    NON_BILLING_KINDS = {
        "Identity", "PatientName", "OwnerName", "Owner",
        "Signalment", "Species", "Breed", "Sex", "Age", "Weight",
        "Anatomy", "BodySite", "BodySystem",
        "Symptom", "Finding", "Observation", "Condition", "Disease", "Diagnosis",
        "Other",  # Explicitly exclude "Other" kind
    }
    
    filtered_atoms = []
    excluded_count = 0
    for atom in cleaned_atoms:
        kind = (atom.get("kind") or "").strip()
        # Normalize kind aliases
        kind_alias = {
            "Medication": "Medicine",
            "Medications": "Medicine",
            "Drugs": "Drug",
            "Substance": "Medicine",
        }
        kind_normalized = kind_alias.get(kind, kind)
        
        # Exclude non-billing kinds
        if kind_normalized in NON_BILLING_KINDS or kind in NON_BILLING_KINDS:
            excluded_count += 1
            continue
        
        # Include billing kinds (explicit check for safety)
        if kind_normalized in BILLING_ONLY_KINDS or kind in BILLING_ONLY_KINDS:
            filtered_atoms.append(atom)
        # Also include if kind is empty/unknown but has billing intent (safety net)
        elif not kind and atom.get("intent_context") in ("Performed", "Ordered", "Administered", "Prescribed"):
            filtered_atoms.append(atom)
        else:
            # Unknown kind - exclude by default (strict filtering)
            excluded_count += 1
    
    # Keep this function pure (no logger arg in signature).
    # Exclusion count is reflected in extraction_summary totals downstream.
    cleaned_atoms = filtered_atoms
    
    # Post-extraction filter (dictionary gating): drop atoms whose concept has KB similarity < threshold
    if cleaned_atoms and os.getenv("KB_POST_EXTRACT_WHITELIST_FILTER", "true").lower() in ("1", "true", "yes"):
        try:
            from kb_ner_db import pg_conn_ctx
            from kb_ner_global_search import filter_atoms_by_kb_whitelist
            with pg_conn_ctx() as conn:
                thresh = max(0.0, min(1.0, float(os.getenv("KB_POST_EXTRACT_WHITELIST_THRESHOLD", "0.2"))))
                cleaned_atoms = filter_atoms_by_kb_whitelist(cleaned_atoms, conn, threshold=thresh)
        except Exception:
            pass  # Keep all atoms if filter unavailable

    from collections import Counter
    assertion_counts = Counter((a.get("assertion_id") or "") for a in cleaned_atoms)
    kind_counts = Counter((a.get("kind") or "") for a in cleaned_atoms)

    obj["knowledge_atoms"] = cleaned_atoms
    obj["extraction_summary"] = {
        "total_atoms": len(cleaned_atoms),
        "confirmed": int(assertion_counts.get("CONF", 0)),
        "negated": int(assertion_counts.get("NEG", 0)),
        "by_kind": {k: int(v) for k, v in kind_counts.items() if k},
    }
    # Ensure metadata is always a dict (schema allows null, but we need a dict)
    if not isinstance(obj.get("metadata"), dict):
        obj["metadata"] = {}
    obj["metadata"]["postprocess_applied"] = True
    return obj


# ==============================================================================
# CLINICAL CORE (Subjective, Objective, Assessment, Plan only)
# ==============================================================================

def _extract_clinical_core_soap(soap_note_text: str) -> str:
    """
    Return only Subjective, Objective, Assessment, Plan (exclude Conclusion, Key Issues, Customer Instructions).
    Reduces Phase 2 prompt size and extraction time (~25s -> ~12s).
    """
    if not soap_note_text or not soap_note_text.strip():
        return soap_note_text or ""
    text = (soap_note_text or "").strip()
    core_sections = []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("Subjective", "Objective", "Assessment", "Plan"):
                val = obj.get(key) or obj.get(key.lower(), "")
                if isinstance(val, str) and val.strip():
                    core_sections.append(f"{key}:\n{val.strip()}")
            if core_sections:
                return "\n\n".join(core_sections)
    except (json.JSONDecodeError, TypeError):
        pass
    if PHASE2_AVAILABLE:
        try:
            sections = parse_soap_sections(text)
            for key in ("Subjective", "Objective", "Assessment", "Plan"):
                val = (sections.get(key) or "").strip()
                if val:
                    core_sections.append(f"{key}:\n{val}")
            if core_sections:
                return "\n\n".join(core_sections)
        except Exception:
            pass
    return text


# ==============================================================================
# EARLY PHASE 2 (Signalment/Subjective on chunk 1 complete)
# ==============================================================================

async def extract_knowledge_atoms_early_async(
    chunk_text: str,
    entity_manifest_subset: List[Dict[str, Any]],
    visit_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], bool]:
    """
    Pre-Phase 2: extract Knowledge Atoms for Subjective/Objective from chunk 1 text.
    Called when Super-Pass chunk 1 completes so extraction overlaps with chunk 2 and SOAP (~8–10s save).
    Full Phase 2 later extracts only Assessment+Plan and merges with these early atoms.
    """
    if not chunk_text or not chunk_text.strip():
        return {"knowledge_atoms": [], "early_sections": ["Subjective", "Objective"]}, True
    if not PHASE2_AVAILABLE or not ENABLE_PHASE2:
        return {"knowledge_atoms": [], "early_sections": ["Subjective", "Objective"]}, True
    try:
        assertion_types, attributes_schema, _ = _load_phase2_schema_cached(logger=logger)
        base_prompt = build_knowledge_atom_prompt(
            assertion_types=assertion_types or [],
            attributes_schema=attributes_schema or {},
        )
        manifest_json = _compact_manifest_for_prompt(entity_manifest_subset or [], max_entities=100)
        prompt_with_manifest = build_phase2_prompt_with_grounding(
            base_prompt=base_prompt,
            session_id=visit_id,
            section_name="EarlyChunk",
            entity_manifest_json=manifest_json,
        )
        early_instruction = (
            "\n\n--- EARLY EXTRACTION (Subjective/Objective only) ---\n"
            "Extract knowledge atoms ONLY for Subjective and Objective sections from the transcript excerpt below.\n"
            "Do NOT extract Assessment or Plan atoms. Return JSON with knowledge_atoms array; set section to \"Subjective\" or \"Objective\" for each atom.\n\n"
            "--- TRANSCRIPT EXCERPT ---\n"
        )
        prompt_full = prompt_with_manifest + early_instruction + chunk_text.strip() + "\n\nExtract all Knowledge Atoms from the excerpt above (Subjective/Objective only)."
        client, provider = get_client_for_model(PHASE2_MODEL, logger=logger)
        if client is None:
            if logger:
                logger.warning("  ⚠️ Early Phase 2: client init failed, skipping")
            return {"knowledge_atoms": [], "early_sections": ["Subjective", "Objective"]}, True
        provider_name = provider or PHASE2_MODEL_PROVIDER
        max_tok = min(PHASE2_MAX_TOKENS_PARALLEL, max(PHASE2_MAX_TOKENS_PARALLEL_MIN, 1500 + len(entity_manifest_subset or []) * PHASE2_TOKENS_PER_ENTITY_ESTIMATE))
        raw = _call_phase2_llm_json(
            client=client,
            model=PHASE2_MODEL,
            prompt=prompt_full,
            provider_name=provider_name,
            max_tokens_first=max_tok,
            temperature=PHASE2_TEMPERATURE,
            logger=logger,
        )
        parsed = _postprocess_phase2_json(raw) or {}
        atoms = parsed.get("knowledge_atoms") or []
        early_atoms = [a for a in atoms if isinstance(a, dict) and (a.get("section") or "").strip() in ("Subjective", "Objective", "Signalment", "")]
        if logger:
            logger.info(f"  📋 Early Phase 2 (chunk 1): {len(chunk_text)} chars → {len(early_atoms)} atoms (Subjective/Objective)")
        return {"knowledge_atoms": early_atoms, "early_sections": ["Subjective", "Objective"], "metadata": {"early_chunk": True}}, True
    except Exception as e:
        if logger:
            logger.debug(f"Early Phase 2 failed: {e}")
        return {"knowledge_atoms": [], "early_sections": ["Subjective", "Objective"], "error": str(e)}, False


# ==============================================================================
# OPTIMIZED ASYNC PHASE 2 EXTRACTION
# ==============================================================================

async def extract_knowledge_atoms_async(
    soap_note_text: str,
    entity_manifest: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    visit_id: Optional[str] = None,
    clinic_id: Optional[int] = None,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    force_all_sections: bool = False,
    enable_billing_matching: bool = False,
    early_atoms_task_ref: Optional[Dict[str, Any]] = None,
    run_timestamp: Optional[str] = None,
    timing_ref: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool]:
    """
    Optimized async Phase 2 Knowledge Atom extraction.
    
    This function (default PHASE2_MODE=minimal):
    - Uses Phase 1's connection pool (no new connections)
    - Accepts SOAP note text + manifest directly (no file I/O)
    - Runs a SINGLE structured extraction call (lowest latency, lowest cost)
    Optional:
    - Parallel per-section extraction + session cache (OFF by default)
    - Billing matching (OFF by default; Phase 3 responsibility)
    
    Args:
        soap_note_text: Final SOAP note text (after constraint injection)
        entity_manifest: Entity manifest from Phase 1 grounding
        session_id: Session ID for caching (defaults to visit_id or new UUID)
        visit_id: Visit ID (used as session_id if provided)
        clinic_id: Clinic ID for billing matching
        output_dir: Output directory for saving results
        logger: Logger instance (uses Phase 1 logger if not provided)
        force_all_sections: Force re-extraction even if cached
        enable_billing_matching: Enable SKU matching (requires clinic_id)
        run_timestamp: Optional run timestamp (YYYYMMDD_HHMMSS) so Phase 2 output filenames match Phase 1 run.
        
    Returns:
        Tuple of (knowledge_atoms_dict, success_flag)
    """
    if not PHASE2_AVAILABLE:
        if logger:
            logger.warning(f"⚠️ Phase 2 not available: {import_error}")
        return {"error": "Phase 2 module not available", "knowledge_atoms": []}, False
    
    if not ENABLE_PHASE2:
        if logger:
            logger.info("⏭️ Phase 2 disabled (ENABLE_PHASE2=false)")
        return {"knowledge_atoms": []}, True
    
    if not soap_note_text or not soap_note_text.strip():
        if logger:
            logger.warning("⚠️ Phase 2: Empty SOAP note, skipping extraction")
        return {"knowledge_atoms": []}, True
    
    session_id = session_id or visit_id or str(uuid.uuid4())
    start_time = asyncio.get_event_loop().time()

    # Pre-Phase 2 merge: await early Subjective/Objective atoms if started on chunk 1
    early_atoms: List[Dict[str, Any]] = []
    if early_atoms_task_ref and early_atoms_task_ref.get("task") is not None:
        try:
            t_early = time.perf_counter()
            early_result, _ = await early_atoms_task_ref["task"]
            _set_phase2_timing(timing_ref, early_subjective_objective_ms=_elapsed_ms_since(t_early))
            early_atoms = (early_result or {}).get("knowledge_atoms") or []
            if logger and early_atoms:
                logger.info(f"  📋 Pre-Phase 2: merging {len(early_atoms)} early atoms (Subjective/Objective)")
        except Exception as e:
            if logger:
                logger.debug(f"  Early Phase 2 task failed or not ready: {e}")
            early_atoms = []
    
    if logger:
        logger.info("=" * 60)
        logger.info("🧬 PHASE 2: KNOWLEDGE ATOM EXTRACTION")
        logger.info("=" * 60)
        logger.info(f"  Model: {PHASE2_MODEL} ({PHASE2_MODEL_PROVIDER})")
        logger.info(f"  Session ID: {session_id}")
        logger.info(f"  Entity manifest: {len(entity_manifest)} entities")
        logger.info(f"  SOAP note length: {len(soap_note_text)} chars")
    
    try:
        # Manifest JSON for prompt injection
        entity_manifest_json = _compact_manifest_for_prompt(entity_manifest)
        
        # Load KB schema (cached in-memory once per process by default)
        assertion_types = []
        attributes_schema = {}
        schema_cache_hit = False
        try:
            assertion_types, attributes_schema, schema_cache_hit = _load_phase2_schema_cached(logger=logger)
            if logger:
                logger.info(
                    f"  ✅ Loaded KB schema: {len(assertion_types)} assertion types, {len(attributes_schema)} kinds "
                    f"(cache_hit={schema_cache_hit})"
                )
        except Exception as e:
            if logger:
                logger.warning(f"  ⚠️ Could not load KB schema: {e}")
        
        # Use full schema (no manifest-based kind filtering)
        relevant_kinds = set((attributes_schema or {}).keys())
        compact_schema = _compact_attributes_schema(attributes_schema, only_kinds=relevant_kinds)

        # Build base prompt (with compact schema; builder omits use_case examples when None)
        base_prompt = build_knowledge_atom_prompt(
            assertion_types=assertion_types,
            attributes_schema=compact_schema
        )

        # Initialize client using the Phase 1 client factory (env vars, fireworks_api.txt, etc.)
        client, provider = get_client_for_model(PHASE2_MODEL, logger=logger)
        # Use the real provider for logs/metadata (don't trust env string).
        provider_name = provider or PHASE2_MODEL_PROVIDER
        if logger:
            logger.info(f"  Provider resolved: {provider_name}")
        if client is None:
            return {"error": f"Phase 2 client init failed for provider={provider_name}", "knowledge_atoms": []}, False

        if PHASE2_DRY_RUN:
            # Offline-safe: build prompt and return metadata without calling any LLM.
            prompt_template = build_phase2_prompt_with_grounding(
                base_prompt=base_prompt,
                session_id=session_id,
                section_name="FullNote",
                entity_manifest_json=entity_manifest_json,
            )
            return ({
                "knowledge_atoms": [],
                "dry_run": True,
                "metadata": {
                    "session_id": session_id,
                    "visit_id": visit_id,
                    "clinic_id": clinic_id,
                    "mode": PHASE2_MODE,
                    "model_provider": provider_name,
                    "model_name": PHASE2_MODEL,
                    "attempted_models": [PHASE2_MODEL],
                    "kb_schema_loaded": len(assertion_types) > 0,
                    "schema_cache_hit": schema_cache_hit,
                    "soap_chars": len(soap_note_text),
                    "manifest_entities": len(entity_manifest or []),
                    "prompt_chars": len(prompt_template) + len(soap_note_text),
                }
            }, True)

        # Clinical core only: Subjective, Objective, Assessment, Plan (exclude Conclusion, Key Issues, Customer Instructions)
        soap_for_prompt = _extract_clinical_core_soap(soap_note_text) if PHASE2_CLINICAL_CORE_ONLY else (soap_note_text or "")
        # Pre-Phase 2: when we have early Subjective/Objective atoms, run only on Assessment+Plan to save time
        if early_atoms:
            try:
                sections = parse_soap_sections(soap_for_prompt or soap_note_text or "")
                a, p = (sections.get("Assessment") or "").strip(), (sections.get("Plan") or "").strip()
                if a or p:
                    soap_for_prompt = "Assessment:\n" + a + "\n\nPlan:\n" + p
                    if logger:
                        logger.info(f"  📋 Phase 2 Assessment+Plan only (early atoms merged): {len(soap_for_prompt)} chars")
            except Exception:
                pass
        if PHASE2_CLINICAL_CORE_ONLY and not early_atoms and logger and len(soap_for_prompt) < len(soap_note_text or ""):
            logger.info(f"  📋 Phase 2 clinical core only: {len(soap_for_prompt)} chars (full SOAP {len(soap_note_text or '')} chars)")

        # MINIMAL MODE (default): single call or parallel batches when entity count > threshold
        n_entities = len(entity_manifest or [])
        use_parallel_atoms = (
            PHASE2_MODE == "minimal"
            and not PHASE2_PARALLEL_SECTIONS
            and n_entities > PHASE2_PARALLEL_ATOM_ENTITY_THRESHOLD
        )
        if PHASE2_MODE == "minimal" and not PHASE2_PARALLEL_SECTIONS:
            if use_parallel_atoms:
                # Parallel atom extraction: split manifest into batches, run LLM calls in parallel (~15s saving for 23 entities).
                import math
                # Smaller fixed-size batches: cap entities per batch to reduce JSON size and truncation risk.
                num_batches = min(
                    PHASE2_PARALLEL_ATOM_MAX_BATCHES,
                    max(2, math.ceil(n_entities / PHASE2_MAX_ENTITIES_PER_BATCH)),
                )
                batch_size = math.ceil(n_entities / num_batches)
                batches = [
                    (entity_manifest or [])[i : i + batch_size]
                    for i in range(0, n_entities, batch_size)
                ]
                if logger:
                    logger.info(
                        f"  ⚡ Phase 2 parallel atom extraction: {n_entities} entities → {num_batches} batches (~{batch_size} each)"
                    )
                    logger.info(
                        f"  🧠 Phase 2 parallel: dynamic max_tokens per batch "
                        f"(min={PHASE2_MAX_TOKENS_PARALLEL_MIN}, cap={PHASE2_MAX_TOKENS_PARALLEL}, ~{PHASE2_TOKENS_PER_ENTITY_ESTIMATE} tokens/entity)"
                    )
                soap_body = soap_for_prompt or soap_note_text or ""

                # OPTIMIZATION: Use dedicated thread pool executor for Phase 2 batches
                # Default asyncio thread pool has limited threads, causing sequential execution
                # Creating dedicated executor ensures true parallelism
                phase2_executor = ThreadPoolExecutor(max_workers=num_batches, thread_name_prefix="phase2_batch")
                
                async def _call_one_batch(batch_manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
                    manifest_json_b = _compact_manifest_for_prompt(batch_manifest)
                    prompt_t = build_phase2_prompt_with_grounding(
                        base_prompt=base_prompt,
                        session_id=session_id,
                        section_name="FullNote",
                        entity_manifest_json=manifest_json_b,
                    )
                    early_only = "\n\nCRITICAL: Subjective and Objective were already extracted. Extract ONLY Assessment and Plan atoms." if early_atoms else ""
                    prompt_b = (
                        prompt_t
                        + "\n\n--- SOAP NOTE ---\n"
                        + soap_body
                        + early_only
                        + "\n\nExtract all Knowledge Atoms from the SOAP note above."
                    )
                    # Dynamic max_tokens per batch: scale with entities in this batch (smaller batches → lower tokens).
                    n_in_batch = len(batch_manifest)
                    if num_batches > 1:
                        base_tokens = 1500  # schema + structure overhead
                        max_tok = min(
                            PHASE2_MAX_TOKENS_PARALLEL,
                            max(PHASE2_MAX_TOKENS_PARALLEL_MIN, base_tokens + n_in_batch * PHASE2_TOKENS_PER_ENTITY_ESTIMATE),
                        )
                    else:
                        is_long_b = (len(soap_body) >= PHASE2_LONG_SOAP_CHARS) or (n_in_batch >= PHASE2_LONG_MANIFEST_ENTITIES)
                        max_tok = PHASE2_MAX_TOKENS_LONG if is_long_b else PHASE2_MAX_TOKENS
                    if isinstance(PHASE2_MODEL, str) and PHASE2_MODEL.startswith("accounts/fireworks/"):
                        max_tok = min(max_tok, 4096)
                    def _call_wrapper():
                        return _call_phase2_llm_json(
                            client=client,
                            model=PHASE2_MODEL,
                            prompt=prompt_b,
                            provider_name=provider_name,
                            max_tokens_first=max_tok,
                            temperature=PHASE2_TEMPERATURE,
                            logger=logger,
                        )
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(phase2_executor, _call_wrapper)

                try:
                    t_llm = time.perf_counter()
                    batch_results = await asyncio.gather(*[_call_one_batch(b) for b in batches])
                    _set_phase2_timing(timing_ref, step1_atom_extraction_ms=_elapsed_ms_since(t_llm))
                finally:
                    # Cleanup executor
                    phase2_executor.shutdown(wait=False)
                all_atoms = []
                for raw_res in batch_results:
                    parsed = _postprocess_phase2_json(raw_res) or {}
                    all_atoms.extend(parsed.get("knowledge_atoms", []) or [])
                if early_atoms:
                    all_atoms = early_atoms + [a for a in all_atoms if (a.get("section") or "").strip() not in ("Subjective", "Objective", "Signalment")]
                t_post = time.perf_counter()
                all_atoms = _deduplicate_knowledge_atoms(all_atoms)
                # Constraint injection: Plan gate, unlinked-only-if-in-plan, drop customer-instruction-only
                soap_for_constraint = soap_for_prompt or soap_note_text or ""
                all_atoms = _apply_knowledge_atom_constraints(all_atoms, entity_manifest, soap_for_constraint, logger=logger)
                knowledge_atoms = {"knowledge_atoms": all_atoms}
                if all_atoms and entity_manifest:
                    all_atoms = _enrich_atoms_with_manifest_ids(all_atoms, entity_manifest)
                    knowledge_atoms["knowledge_atoms"] = all_atoms
                _set_phase2_timing(timing_ref, step2_post_process_ms=_elapsed_ms_since(t_post))
                atoms_list = all_atoms
                # Phase 2 of knowledge atoms step: build verification dashboard from consolidated manifest + filtered atoms
                if atoms_list:
                    t_dash = time.perf_counter()
                    with pg_conn_ctx() as conn:
                        verification_dashboard = _build_verification_dashboard(
                            atoms_list, conn=conn, logger=logger, entity_manifest=entity_manifest
                        )
                    _set_phase2_timing(timing_ref, step3_dashboard_ms=_elapsed_ms_since(t_dash))
                    knowledge_atoms["verification_dashboard"] = verification_dashboard
                    if logger:
                        logger.info(f"  📊 Verification dashboard built: {sum(len(v) for v in verification_dashboard.values())} items across 7 modules")
                if not isinstance(knowledge_atoms.get("metadata"), dict):
                    knowledge_atoms["metadata"] = {}
                knowledge_atoms["metadata"].update({
                    "session_id": session_id,
                    "visit_id": visit_id,
                    "clinic_id": clinic_id,
                    "mode": "minimal_parallel",
                    "model_provider": provider_name,
                    "model_name": PHASE2_MODEL,
                    "attempted_models": [PHASE2_MODEL],
                    "kb_schema_loaded": len(assertion_types) > 0,
                    "parallel_batches": len(batches),
                })
            else:
                # Single-call path (entity count <= threshold or non-minimal)
                prompt_template = build_phase2_prompt_with_grounding(
                    base_prompt=base_prompt,
                    session_id=session_id,
                    section_name="FullNote",
                    entity_manifest_json=entity_manifest_json,
                )
                if logger:
                    logger.info("  ⚡ Phase 2 minimal mode: single-call extraction (assertion + attributes)")
                early_only = "\n\nCRITICAL: Subjective and Objective were already extracted. Extract ONLY Assessment and Plan atoms." if early_atoms else ""
                prompt = (
                    prompt_template
                    + "\n\n--- SOAP NOTE ---\n"
                    + (soap_for_prompt or soap_note_text or "")
                    + early_only
                    + "\n\nExtract all Knowledge Atoms from the SOAP note above."
                )
                soap_len = len(soap_for_prompt or soap_note_text or "")
                entity_count = len(entity_manifest or [])
                max_tokens = _estimate_phase2_max_tokens(soap_len, entity_count)
                if isinstance(PHASE2_MODEL, str) and PHASE2_MODEL.startswith("accounts/fireworks/"):
                    max_tokens = min(max_tokens, 4096)
                if logger:
                    logger.info(
                        f"  🧠 Phase 2 dynamic max_tokens={max_tokens} "
                        f"(soap_chars={soap_len}, entities={entity_count}, "
                        f"floor={PHASE2_MAX_TOKENS}, ceil={PHASE2_MAX_TOKENS_LONG})"
                    )
                t_llm = time.perf_counter()
                knowledge_atoms = await asyncio.to_thread(
                    _call_phase2_llm_json,
                    client=client,
                    model=PHASE2_MODEL,
                    prompt=prompt,
                    provider_name=provider_name,
                    max_tokens_first=max_tokens,
                    temperature=PHASE2_TEMPERATURE,
                    logger=logger,
                )
                _set_phase2_timing(timing_ref, step1_atom_extraction_ms=_elapsed_ms_since(t_llm))

                attempted_models = [PHASE2_MODEL]

                if PHASE2_ESCALATE_ON_JSON_FAIL and _looks_like_json_parse_failure(knowledge_atoms):
                    escalate_model = (PHASE2_ESCALATE_MODEL or "").strip()
                    if escalate_model and escalate_model != PHASE2_MODEL:
                        esc_client, esc_provider = get_client_for_model(escalate_model, logger=logger)
                        if esc_client is not None:
                            esc_provider_name = esc_provider or provider_name
                            if logger:
                                logger.warning(
                                    f"  ⚠️ Phase 2 JSON parse failed on '{PHASE2_MODEL}'. "
                                    f"Escalating once to '{escalate_model}'..."
                                )
                            t_esc = time.perf_counter()
                            knowledge_atoms = await asyncio.to_thread(
                                _call_phase2_llm_json,
                                client=esc_client,
                                model=escalate_model,
                                prompt=prompt,
                                provider_name=esc_provider_name,
                                max_tokens_first=max_tokens,
                                temperature=PHASE2_TEMPERATURE,
                                logger=logger,
                            )
                            prev = timing_ref.get("step1_atom_extraction_ms", 0) if timing_ref else 0
                            _set_phase2_timing(
                                timing_ref,
                                step1_atom_extraction_ms=prev + _elapsed_ms_since(t_esc),
                            )
                            attempted_models.append(escalate_model)

                knowledge_atoms = _postprocess_phase2_json(knowledge_atoms) or {}

                atoms_list = knowledge_atoms.get("knowledge_atoms", []) or []
                if early_atoms:
                    atoms_list = early_atoms + [a for a in atoms_list if (a.get("section") or "").strip() not in ("Subjective", "Objective", "Signalment")]
                t_post = time.perf_counter()
                atoms_list = _deduplicate_knowledge_atoms(atoms_list)
                soap_for_constraint = soap_for_prompt or soap_note_text or ""
                atoms_list = _apply_knowledge_atom_constraints(atoms_list, entity_manifest, soap_for_constraint, logger=logger)
                knowledge_atoms["knowledge_atoms"] = atoms_list
                if atoms_list and entity_manifest:
                    atoms_list = _enrich_atoms_with_manifest_ids(atoms_list, entity_manifest)
                    knowledge_atoms["knowledge_atoms"] = atoms_list
                _set_phase2_timing(timing_ref, step2_post_process_ms=_elapsed_ms_since(t_post))

                # Phase 2 of knowledge atoms step: verification dashboard from consolidated manifest + filtered atoms
                if atoms_list:
                    t_dash = time.perf_counter()
                    with pg_conn_ctx() as conn:
                        verification_dashboard = _build_verification_dashboard(
                            atoms_list, conn=conn, logger=logger, entity_manifest=entity_manifest
                        )
                    _set_phase2_timing(timing_ref, step3_dashboard_ms=_elapsed_ms_since(t_dash))
                    knowledge_atoms["verification_dashboard"] = verification_dashboard
                    if logger:
                        logger.info(f"  📊 Verification dashboard built: {sum(len(v) for v in verification_dashboard.values())} items across 7 modules")

                if not isinstance(knowledge_atoms.get("metadata"), dict):
                    knowledge_atoms["metadata"] = {}
                knowledge_atoms["metadata"].update({
                    "session_id": session_id,
                    "visit_id": visit_id,
                    "clinic_id": clinic_id,
                    "mode": "minimal",
                    "model_provider": provider_name,
                    "model_name": attempted_models[-1] if attempted_models else PHASE2_MODEL,
                    "attempted_models": attempted_models,
                    "kb_schema_loaded": len(assertion_types) > 0,
                })

        else:
            # Safety: if someone enables full/parallel flags, we still run minimal extraction
            # rather than failing the pipeline. (You can re-enable the heavier path later.)
            if logger:
                logger.warning(
                    "  ⚠️ Phase 2 full/parallel/session-cache flags are enabled, but integration currently "
                    "runs minimal single-call mode for stability + low latency."
                )
            prompt_template = build_phase2_prompt_with_grounding(
                base_prompt=base_prompt,
                session_id=session_id,
                section_name="FullNote",
                entity_manifest_json=entity_manifest_json,
            )
            early_only = "\n\nCRITICAL: Subjective and Objective were already extracted. Extract ONLY Assessment and Plan atoms." if early_atoms else ""
            prompt = (
                prompt_template
                + "\n\n--- SOAP NOTE ---\n"
                + (soap_for_prompt or soap_note_text or "")
                + early_only
                + "\n\nExtract all Knowledge Atoms from the SOAP note above."
            )
            soap_len = len(soap_for_prompt or soap_note_text or "")
            entity_count = len(entity_manifest or [])
            max_tokens = _estimate_phase2_max_tokens(soap_len, entity_count)
            if isinstance(PHASE2_MODEL, str) and PHASE2_MODEL.startswith("accounts/fireworks/"):
                max_tokens = min(max_tokens, 4096)
            if logger:
                logger.info(
                    f"  🧠 Phase 2 dynamic max_tokens={max_tokens} "
                    f"(soap_chars={soap_len}, entities={entity_count}, "
                    f"floor={PHASE2_MAX_TOKENS}, ceil={PHASE2_MAX_TOKENS_LONG})"
                )
            t_llm = time.perf_counter()
            knowledge_atoms = await asyncio.to_thread(
                _call_phase2_llm_json,
                client=client,
                model=PHASE2_MODEL,
                prompt=prompt,
                provider_name=provider_name,
                max_tokens_first=max_tokens,
                temperature=PHASE2_TEMPERATURE,
                logger=logger,
            )
            _set_phase2_timing(timing_ref, step1_atom_extraction_ms=_elapsed_ms_since(t_llm))
            knowledge_atoms = _postprocess_phase2_json(knowledge_atoms)
            
            # Enrich atoms with inventory/service IDs from entity manifest
            atoms_list = knowledge_atoms.get("knowledge_atoms", []) or []
            if early_atoms:
                atoms_list = early_atoms + [a for a in atoms_list if (a.get("section") or "").strip() not in ("Subjective", "Objective", "Signalment")]
            t_post = time.perf_counter()
            atoms_list = _deduplicate_knowledge_atoms(atoms_list)
            # Constraint injection: Plan gate, unlinked-only-if-in-plan, drop customer-instruction-only
            soap_for_constraint = soap_for_prompt or soap_note_text or ""
            atoms_list = _apply_knowledge_atom_constraints(atoms_list, entity_manifest, soap_for_constraint, logger=logger)
            knowledge_atoms["knowledge_atoms"] = atoms_list
            if atoms_list and entity_manifest:
                atoms_list = _enrich_atoms_with_manifest_ids(atoms_list, entity_manifest)
                knowledge_atoms["knowledge_atoms"] = atoms_list
            _set_phase2_timing(timing_ref, step2_post_process_ms=_elapsed_ms_since(t_post))

            # Phase 2 of knowledge atoms step: build verification dashboard from consolidated manifest + filtered atoms
            if atoms_list:
                t_dash = time.perf_counter()
                # Use Phase 1's connection pool for master table queries
                with pg_conn_ctx() as conn:
                    verification_dashboard = _build_verification_dashboard(
                        atoms_list, conn=conn, logger=logger, entity_manifest=entity_manifest
                    )
                _set_phase2_timing(timing_ref, step3_dashboard_ms=_elapsed_ms_since(t_dash))
                knowledge_atoms["verification_dashboard"] = verification_dashboard
                if logger:
                    logger.info(f"  📊 Verification dashboard built: {sum(len(v) for v in verification_dashboard.values())} items across 7 modules")
        
        # Save output if output_dir provided (use run_timestamp so Phase 2 files match Phase 1 run)
        if output_dir:
            try:
                from datetime import datetime
                file_ts = (run_timestamp or "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = Path(output_dir) / f"knowledge_atoms_{file_ts}.json"
                output_json = json.dumps(knowledge_atoms, indent=2, ensure_ascii=False)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(output_json)
                if logger:
                    logger.info(f"  ✅ Knowledge atoms saved to: {output_file}")
                
                # Also save verification dashboard as separate file
                if isinstance(knowledge_atoms, dict) and "verification_dashboard" in knowledge_atoms:
                    dashboard_file = Path(output_dir) / f"verification_dashboard_{file_ts}.json"
                    dashboard_json = json.dumps(knowledge_atoms["verification_dashboard"], indent=2, ensure_ascii=False)
                    with open(dashboard_file, 'w', encoding='utf-8') as f:
                        f.write(dashboard_json)
                    if logger:
                        logger.info(f"  ✅ Verification dashboard saved to: {dashboard_file}")
            except Exception as e:
                if logger:
                    logger.warning(f"  ⚠️ Could not save knowledge atoms: {e}")
        
        elapsed = asyncio.get_event_loop().time() - start_time
        _set_phase2_timing(timing_ref, phase2_total_ms=int(round(elapsed * 1000)))
        atoms_len = len(knowledge_atoms.get("knowledge_atoms", []) or []) if isinstance(knowledge_atoms, dict) else 0
        is_failure = _looks_like_json_parse_failure(knowledge_atoms) if isinstance(knowledge_atoms, dict) else True
        if logger:
            if is_failure:
                logger.warning(
                    f"  ❌ Phase 2 failed: invalid/unparseable JSON (atoms={atoms_len}) in {elapsed:.2f}s "
                    f"(parse_error={knowledge_atoms.get('parse_error') if isinstance(knowledge_atoms, dict) else 'n/a'})"
                )
            else:
                logger.info(f"  ✅ Phase 2 complete: {atoms_len} atoms extracted in {elapsed:.2f}s")

        return knowledge_atoms, (not is_failure)
        
    except Exception as e:
        if logger:
            logger.error(f"  ❌ Phase 2 extraction failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        return {"error": str(e), "knowledge_atoms": []}, False

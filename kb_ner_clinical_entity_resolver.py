"""
Clinical Entity Resolver (CER).

Runs between Brain NER and Grounding. The LLM performs two phases in one call:
- Phase 1 (consolidation): Merge synonyms, resolve kind ambiguity, drop filler.
- Phase 2 (billing-only): Remove Reminder, Diagnosis, customer-instruction nature, and any kind
  not in [ReasonForVisit, Medication, Procedure, Diagnostic, Diet, Preventive, ParasiteControl].
  The LLM is instructed to output only this Phase-2 result as the final CER_NER output.

The final CER output is taken for local grounding only. A Python post-filter is also applied
as a safety net to enforce billing-only (same exclusions).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

# Default model for CER. Override with CER_MODEL env if needed.
CER_MODEL = (os.getenv("CER_MODEL") or os.getenv("SUPER_PASS_MODEL") or "gpt-4.1-mini").strip()

# Kinds that are billing-relevant and kept by CER. Diagnosis and Reminder are excluded per requirement.
CER_BILLABLE_KINDS = [
    "ReasonForVisit",
    "Medication",
    "Procedure",
    "Diagnostic",
    "Diet",
    "Preventive",
    "ParasiteControl",
]

# Patterns that suggest customer-instruction-only (not a billable product/service name).
_CER_INSTRUCTION_PATTERNS = re.compile(
    r"\b(bring\s+him|bring\s+her|come\s+back|follow\s+up|give\s+once|give\s+twice|feed\s+twice|"
    r"administer\s+at\s+home|apply\s+at\s+home|call\s+us|schedule\s+an?\s+appointment|"
    r"monitor\s+for|watch\s+for|return\s+if|recheck|re-check)\b",
    re.I,
)


def _is_customer_instruction_nature(ent: Dict[str, Any]) -> bool:
    """True if the entity looks like a customer instruction (not a billable item name)."""
    span = (ent.get("span_text") or "").strip()
    norm = (ent.get("normalized_name") or "").strip() or span
    text = (span + " " + norm).strip()
    return bool(_CER_INSTRUCTION_PATTERNS.search(text))


def _filter_cer_to_billing_only(
    entities: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Remove entities that are customer instructions, reminders, diagnosis, or otherwise not billing-relevant.
    Keep only kinds in CER_BILLABLE_KINDS and drop instruction-nature spans.
    """
    billable_set = {k.lower() for k in CER_BILLABLE_KINDS}
    out = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        kind = (e.get("kind") or "Other").strip()
        kind_lower = kind.lower()
        # Drop: Reminder, Diagnosis, and any kind not in billing set
        if kind_lower == "reminder":
            if logger:
                logger.debug("  CER filter: removed Reminder '%s'", (e.get("span_text") or e.get("normalized_name"))[:50])
            continue
        if kind_lower == "diagnosis":
            if logger:
                logger.debug("  CER filter: removed Diagnosis '%s'", (e.get("span_text") or e.get("normalized_name"))[:50])
            continue
        if kind_lower not in billable_set:
            if logger:
                logger.debug("  CER filter: removed non-billing kind '%s' for '%s'", kind, (e.get("span_text") or "")[:40])
            continue
        # Drop: customer-instruction nature (e.g. "bring him back", "feed twice daily")
        if _is_customer_instruction_nature(e):
            if logger:
                logger.debug("  CER filter: removed customer-instruction '%s'", (e.get("span_text") or "")[:50])
            continue
        out.append(e)
    return out


_CER_SYSTEM = """You are a clinical entity resolver for veterinary SOAP notes. You perform TWO PHASES in one response. The final output is the CER_NER output and will be used for local grounding only.

═════════════════════════════════════════
PHASE 1 — CONSOLIDATION (internal)
═════════════════════════════════════════
1. MERGE SYNONYMS: Group entities that refer to the same real-world concept into ONE entity. Examples:
   - "Oreo", "5-year-old male Labrador", "Patient", "dog" → one Identity/Signalment entity.
   - "FHO" and "Femoral Head and Neck Ostectomy" → one procedure entity (prefer expanded form).
   - "X-ray" and "Radiograph" → one diagnostic entity.
2. RESOLVE KIND AMBIGUITY: Pick the most specific/billing-relevant kind when ambiguous (e.g. Medication over Other, Procedure over Anatomy when clearly a procedure).
3. DROP FILLER: Remove purely conversational or duplicate mentions. Preserve schema: span_text, normalized_name, kind, hints, domain, correctness_probability, suggestion_probability, start_char, end_char, attributes, etc.

═════════════════════════════════════════
PHASE 2 — BILLING-ONLY FILTER (same LLM, before output)
═════════════════════════════════════════
From the consolidated list from Phase 1, REMOVE all of the following so they do NOT appear in your final output:
- REMINDER: Every entity with kind "Reminder" (e.g. follow-up, schedule appointment, bring back).
- DIAGNOSIS: Every entity with kind "Diagnosis" (e.g. hip dysplasia, arthritis as diagnosis — do not include in output).
- CUSTOMER INSTRUCTIONS: Any entity that is in the nature of customer instructions (e.g. "bring him back", "feed twice daily", "give once daily", "administer at home", "schedule an appointment", "monitor for", "recheck", "follow up"). These are instructions to the owner, not billable items.
- NON-BILLING KINDS: Any entity whose kind is NOT one of: ReasonForVisit, Medication, Procedure, Diagnostic, Diet, Preventive, ParasiteControl. So REMOVE: Anatomy, Symptom, VitalSign, Other, Signalment, Identity, Reminder, Diagnosis, and any other kind not in the billing set above.

KEEP only entities that are billing-relevant: kind must be one of ReasonForVisit, Medication, Procedure, Diagnostic, Diet, Preventive, ParasiteControl; and the entity must NOT be customer-instruction nature.

═════════════════════════════════════════
FINAL OUTPUT (CER_NER — for local grounding)
═════════════════════════════════════════
Return a JSON object with a single key "unique_actionable_items" whose value is the array of entities AFTER Phase 2. That is, only the billing-relevant, consolidated entities. This list will be taken for local grounding. Each object must have at least: span_text, normalized_name, kind. Preserve other fields (hints, domain, correctness_probability, suggestion_probability, start_char, end_char, attributes, inventory_category, service_category, etc.) from the canonical representative."""

_CER_USER_TEMPLATE = """Apply both phases in one response.

PHASE 1: Consolidate the input entities (merge synonyms, resolve kind ambiguity, drop filler).
PHASE 2: From that consolidated list, remove Reminder, Diagnosis, customer-instruction nature, and any kind not in [ReasonForVisit, Medication, Procedure, Diagnostic, Diet, Preventive, ParasiteControl]. Keep only billing-relevant entities.

Your final output must be the result of Phase 2 only — the CER_NER output that will be used for local grounding. Return ONLY valid JSON with key "unique_actionable_items" (array of entity objects).

INPUT ENTITIES (JSON array):
{entities_json}

Return a single JSON object: {{ "unique_actionable_items": [ ... ] }}"""


def _compact_entity_for_cer(ent: Dict[str, Any]) -> Dict[str, Any]:
    """Keep fields needed for CER; drop very long or redundant fields to save tokens."""
    out = {}
    for k, v in ent.items():
        if k in ("span_text", "normalized_name", "kind", "hints", "domain", "attributes",
                 "correctness_probability", "suggestion_probability", "start_char", "end_char",
                 "entity_id", "category", "inventory_category", "service_category", "supporting_text"):
            if v is not None and v != "" and v != [] and v != {}:
                out[k] = v
    return out


def _safe_get_delta_text(chunk: Any) -> str:
    """Extract streamed delta content from OpenAI-compatible streaming chunks."""
    try:
        choices = getattr(chunk, "choices", None)
        if choices and len(choices) > 0:
            c0 = choices[0]
            delta = getattr(c0, "delta", None)
            if delta is not None:
                return getattr(delta, "content", "") or ""
            msg = getattr(c0, "message", None)
            if msg is not None:
                return getattr(msg, "content", "") or ""
    except Exception:
        return ""
    return ""


def _repair_cer_streaming_json(content: str) -> str:
    """
    Repair common LLM/streaming JSON issues: trailing commas before } or ],
    and optionally close unclosed brackets when truncated.
    """
    if not content:
        return content
    # Remove trailing commas before } or ] (common in LLM output)
    content = re.sub(r",\s*([}\]])", r"\1", content)
    # If still likely truncated, try closing brackets (same idea as kb_ner_super_pass)
    open_sq = content.count("[") - content.count("]")
    open_br = content.count("{") - content.count("}")
    if open_sq > 0 or open_br > 0:
        content = content.strip() + "]" * open_sq + "}" * open_br
    return content


def _run_cer_streaming(
    messages: List[Dict[str, Any]],
    client: Any,
    model: str,
    logger: Optional[logging.Logger] = None,
    max_tokens: int = 8000,
) -> Optional[List[Dict[str, Any]]]:
    """
    Run CER with stream=True (for Fireworks when max_tokens > 4096).
    Returns unique_actionable_items list or None on failure.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
            stream=True,
        )
        full_text = ""
        for chunk in resp:
            delta = _safe_get_delta_text(chunk)
            if delta:
                full_text += delta
        content = (full_text or "").strip()
        if not content:
            return None
        if content.startswith("```"):
            for start in ("```json\n", "```\n"):
                if content.startswith(start):
                    content = content[len(start):]
                    break
            if content.endswith("```"):
                content = content[:-3].strip()
        # Try parse; on failure try repair (trailing commas, truncation)
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            content = _repair_cer_streaming_json(content)
            try:
                obj = json.loads(content)
            except json.JSONDecodeError as e:
                if logger:
                    logger.warning("  ⚠️ CER streaming JSON repair failed: %s", e)
                return None
        items = obj.get("unique_actionable_items")
        if isinstance(items, list):
            return items
        return None
    except Exception as e:
        if logger:
            logger.warning("  ⚠️ CER streaming failed: %s", e)
        return None


def run_clinical_entity_resolver(
    brain_ner_entities: List[Dict[str, Any]],
    client: Any,
    model: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    timing_out: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run Clinical Entity Resolver on Brain NER output.
    Returns consolidated list of unique, actionable entities for grounding.

    Args:
        brain_ner_entities: List of entity dicts from Brain NER (skeleton_list parsed).
        client: OpenAI-compatible client for chat completion.
        model: Model name (default CER_MODEL env or gpt-4.1-nano).
        logger: Optional logger.

    Returns:
        List of entity dicts (unique_actionable_items). On failure, returns input list unchanged.
    """
    if not brain_ner_entities:
        return []
    model = (model or CER_MODEL).strip() or CER_MODEL
    compact = [_compact_entity_for_cer(e) for e in brain_ner_entities if isinstance(e, dict) and (e.get("span_text") or e.get("normalized_name"))]
    if not compact:
        return list(brain_ner_entities)
    try:
        entities_json = json.dumps(compact, ensure_ascii=False, indent=0)
    except (TypeError, ValueError):
        entities_json = json.dumps([{"span_text": (e.get("span_text") or e.get("normalized_name") or ""), "kind": e.get("kind", "Other")} for e in compact], ensure_ascii=False)
    user_msg = _CER_USER_TEMPLATE.format(entities_json=entities_json)
    messages = [
        {"role": "system", "content": _CER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    # OpenAI (e.g. gpt-4.1-mini): non-streaming with max_tokens=8000. No streaming path.
    # Fireworks: non-streaming limited to 4096; use streaming when we need > 4096.
    cer_max_tokens = 8000
    is_fireworks = "fireworks" in (model or "").lower() or "accounts/" in (model or "")
    if is_fireworks:
        cer_max_tokens = min(cer_max_tokens, 4096)
    use_streaming = is_fireworks  # Only Fireworks uses streaming (OpenAI doesn't need it)
    if use_streaming:
        if logger:
            logger.info("  CER: Fireworks with large output → auto-switching to streaming mode")
        t0 = time.perf_counter()
        items = _run_cer_streaming(messages, client, model, logger=logger, max_tokens=8000)
        cer_latency_s = time.perf_counter() - t0
        if timing_out is not None:
            timing_out["latency_ms"] = int(round(cer_latency_s * 1000))
            timing_out["attempted"] = True
        if logger:
            logger.info(f"  ⏱️ CER LLM call (streaming): {cer_latency_s:.1f}s ({len(compact)} entities)")
        if items is not None:
            # Same merge + filter as non-streaming path below
            by_key: Dict[str, Dict[str, Any]] = {}
            for e in brain_ner_entities:
                if not isinstance(e, dict):
                    continue
                span = (e.get("span_text") or "").strip()
                norm = (e.get("normalized_name") or "").strip() or span
                key = (norm.lower(), (e.get("kind") or "Other").strip().lower())
                if key not in by_key:
                    by_key[key] = dict(e)
            out = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                span = (item.get("span_text") or "").strip()
                norm = (item.get("normalized_name") or "").strip() or span
                kind = (item.get("kind") or "Other").strip()
                key = (norm.lower(), kind.lower())
                base = by_key.get(key)
                if base is not None:
                    merged = dict(base)
                    for k, v in item.items():
                        if v is not None and v != "" and v != [] and v != {}:
                            merged[k] = v
                    out.append(merged)
                else:
                    out.append(dict(item))
            before_filter = len(out)
            out = _filter_cer_to_billing_only(out, logger=logger)
            if logger:
                logger.info(
                    f"  ✅ CER: {len(brain_ner_entities)} → {before_filter} consolidated → {len(out)} billing-only (removed Reminder/Diagnosis/customer-instruction/non-billing)"
                )
            return out
        # Streaming failed or empty → fall back to non-streaming with 4096 cap
    try:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=cer_max_tokens,
        )
        cer_latency_s = time.perf_counter() - t0
        if timing_out is not None:
            timing_out["latency_ms"] = int(round(cer_latency_s * 1000))
            timing_out["attempted"] = True
        if logger:
            logger.info(f"  ⏱️ CER LLM call: {cer_latency_s:.1f}s ({len(compact)} entities)")
        first = (resp.choices or [None])[0] if getattr(resp, "choices", None) else None
        if first is None:
            content = ""
        elif hasattr(first, "message"):
            msg = first.message
            content = (getattr(msg, "content", None) or "").strip()
        elif isinstance(first, dict):
            content = ((first.get("message") or {}).get("content") or "").strip()
        else:
            content = ""
        content = content.strip() if isinstance(content, str) else ""
        if content.startswith("```"):
            for start in ("```json\n", "```\n"):
                if content.startswith(start):
                    content = content[len(start):]
                    break
            if content.endswith("```"):
                content = content[:-3].strip()
        obj = json.loads(content)
        items = obj.get("unique_actionable_items")
        if not isinstance(items, list):
            if logger:
                logger.warning("CER did not return unique_actionable_items array, using full Brain NER list")
            return list(brain_ner_entities)
        # Preserve full entity shape: match back to original by span_text/normalized_name and fill missing fields
        by_key: Dict[str, Dict[str, Any]] = {}
        for e in brain_ner_entities:
            if not isinstance(e, dict):
                continue
            span = (e.get("span_text") or "").strip()
            norm = (e.get("normalized_name") or "").strip() or span
            key = (norm.lower(), (e.get("kind") or "Other").strip().lower())
            if key not in by_key:
                by_key[key] = dict(e)
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            span = (item.get("span_text") or "").strip()
            norm = (item.get("normalized_name") or "").strip() or span
            kind = (item.get("kind") or "Other").strip()
            key = (norm.lower(), kind.lower())
            base = by_key.get(key)
            if base is not None:
                merged = dict(base)
                for k, v in item.items():
                    if v is not None and v != "" and v != [] and v != {}:
                        merged[k] = v
                out.append(merged)
            else:
                out.append(dict(item))
        # Remove customer instructions, reminders, diagnosis, and non-billing kinds (only keep billing-relevant entities)
        before_filter = len(out)
        out = _filter_cer_to_billing_only(out, logger=logger)
        if logger:
            logger.info(
                f"  ✅ CER: {len(brain_ner_entities)} → {before_filter} consolidated → {len(out)} billing-only (removed Reminder/Diagnosis/customer-instruction/non-billing)"
            )
        return out
    except json.JSONDecodeError as e:
        if logger:
            logger.warning(f"  ⚠️ CER JSON parse failed: {e}, using full Brain NER list")
        return list(brain_ner_entities)
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠️ CER failed: {e}, using full Brain NER list")
        return list(brain_ner_entities)


async def run_clinical_entity_resolver_async(
    brain_ner_entities: List[Dict[str, Any]],
    client: Any,
    model: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    timing_out: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Async wrapper: run CER in a thread to avoid blocking the event loop."""
    import asyncio
    return await asyncio.to_thread(
        run_clinical_entity_resolver,
        brain_ner_entities,
        client,
        model,
        logger,
        timing_out,
    )

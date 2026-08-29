"""
Post-grounding helpers: ensure unlinked entities keep clinic SKU suggestions
for Phase 2 verification dashboard (RAG → billing handoff).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def enrich_unlinked_manifest_suggestions(
    entity_manifest: List[Dict[str, Any]],
    grounding_records: Optional[List[Dict[str, Any]]] = None,
    logger: Optional[Any] = None,
) -> int:
    """
    Copy grounding-layer candidates into attributes.suggestions when the entity
    stayed unlinked and suggestions are missing. Returns number of entities enriched.
    """
    if not isinstance(entity_manifest, list) or not entity_manifest:
        return 0

    by_span: Dict[str, Dict[str, Any]] = {}
    if isinstance(grounding_records, list):
        for row in grounding_records:
            if not isinstance(row, dict):
                continue
            key = (row.get("span_text") or "").strip().lower()
            if key:
                by_span[key] = row

    enriched = 0
    for ent in entity_manifest:
        if not isinstance(ent, dict):
            continue
        if ent.get("local_stock_id") or ent.get("local_service_id"):
            continue
        method = str(ent.get("match_method") or "")
        if method in ("non_billable_preserved", "vital_sign_structured", "skip_signalment", "skip_identity"):
            continue

        attrs = ent.get("attributes")
        if not isinstance(attrs, dict):
            attrs = {}
            ent["attributes"] = attrs
        existing = attrs.get("suggestions") or []
        if isinstance(existing, list) and existing:
            continue

        row = by_span.get((ent.get("span_text") or "").strip().lower())
        candidates = (row or {}).get("candidates") or []
        if not candidates:
            continue

        suggestions: List[Dict[str, Any]] = []
        for cand in candidates[:5]:
            if not isinstance(cand, dict):
                continue
            name = cand.get("preferred_name") or cand.get("display_name") or cand.get("name") or ""
            if not name:
                continue
            sug: Dict[str, Any] = {
                "name": name,
                "match_score": float(cand.get("score") or cand.get("match_score") or 0),
                "recommendation": "MEDIUM",
            }
            if cand.get("service_id"):
                sug["service_id"] = cand.get("service_id")
            if cand.get("stock_id") or cand.get("inventory_id") or cand.get("concept_id"):
                sid = cand.get("stock_id") or cand.get("inventory_id") or cand.get("concept_id")
                # concept_id in grounding local candidates is often the stock/service id field misuse; keep if numeric
                if cand.get("stock_id") or cand.get("inventory_id"):
                    sug["inventory_id"] = cand.get("stock_id") or cand.get("inventory_id")
                    sug["stock_id"] = sug["inventory_id"]
            suggestions.append(sug)

        if suggestions:
            attrs["suggestions"] = suggestions
            enriched += 1

    if logger and enriched:
        logger.info(
            "✅ Enriched %d unlinked entities with grounding candidate suggestions for Phase 2",
            enriched,
        )
    return enriched

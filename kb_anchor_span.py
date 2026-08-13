"""
Anchor-Span Architecture for Bi-Directional SOAP ↔ Billing Sync

Provides deterministic mapping between SOAP note text and Entity Manifest via
invisible anchor IDs (E1, E2, ...). Enables:
- Upstream (Billing → SOAP): Replace term by manifest_ref without guessing
- Downstream (SOAP → Billing): Track edits by anchor_id and re-run search
- Frontend: Render [[E1|Term]] as <span data-entity-id="E1">Term</span>
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Tag format: [[anchor_id|display_text]] (pipe = normalized) or [[anchor_id:display_text]] (colon = SOAP gen output)
# Both are supported; injection normalizes colon -> pipe with canonical term from manifest.
ANCHOR_TAG_PATTERN = re.compile(r"\[\[(E\d+)(?:\||:)(.*?)\]\]", re.DOTALL)


def ensure_anchor_ids_on_manifest(entity_manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assign short, unique anchor_id (E1, E2, ...) to each entity in the manifest.
    Idempotent: if anchor_id already set, only fills missing ones.

    Anchor order is by transcript position (start_char) so that E1 = first entity
    in transcript order (typically patient name/signalment), E2 = second, etc.
    This keeps SOAP anchor usage consistent for signalment (E1/E2) across runs.

    Call this before constraint injection so replacement logic can emit [[E1|term]].
    """
    manifest = entity_manifest or []
    # Sort by transcript order so E1, E2, ... follow first appearance (signalment first)
    def _order_key(e: Dict[str, Any]) -> tuple:
        start = int(e.get("start_char") or 0)
        eid = (e.get("entity_id") or "").strip()
        span = (e.get("span_text") or "").strip().lower()
        return (start, eid, span)
    sorted_manifest = sorted(
        (e for e in manifest if isinstance(e, dict)),
        key=_order_key,
    )
    for i, entity in enumerate(sorted_manifest):
        if not entity.get("anchor_id"):
            eid = (entity.get("entity_id") or "").strip()
            # Preserve Brain/parallel-path entity_id (E1, E2, ...) so transcript tags match manifest
            if eid and len(eid) > 1 and eid[0] == "E" and eid[1:].isdigit():
                entity["anchor_id"] = eid
            else:
                entity["anchor_id"] = f"E{i + 1}"
    return entity_manifest


def parse_anchored_soap(text: str) -> List[Dict[str, Any]]:
    """
    Parse SOAP text containing [[E1|Term]] tags into a list of segments for the frontend.
    Frontend can render entity_span segments as <span data-entity-id="E1">Term</span>.

    Returns:
        List of segments: {"type": "text"|"entity_span", "content": str, "manifest_ref": str?}
    """
    if not text or not text.strip():
        return [{"type": "text", "content": text or ""}]

    segments: List[Dict[str, Any]] = []
    last_end = 0

    for m in ANCHOR_TAG_PATTERN.finditer(text):
        anchor_id = m.group(1)
        display_text = m.group(2)
        start, end = m.span()

        if start > last_end:
            segments.append({
                "type": "text",
                "content": text[last_end:start],
            })
        segments.append({
            "type": "entity_span",
            "content": display_text,
            "manifest_ref": anchor_id,
        })
        last_end = end

    if last_end < len(text):
        segments.append({
            "type": "text",
            "content": text[last_end:],
        })

    if not segments:
        return [{"type": "text", "content": text}]
    return segments


def soap_anchored_to_display_text(text: str) -> str:
    """
    Strip anchor tags for plain-text display (e.g. email, print).
    [[E1|Drontal Plus]] → Drontal Plus
    """
    if not text:
        return text
    return ANCHOR_TAG_PATTERN.sub(r"\2", text)


def soap_display_to_anchored_segments_by_section(soap_json: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse a SOAP JSON (section -> content) into per-section anchored segments.
    Useful when SOAP is stored as JSON with Subjective, Objective, Plan, etc.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key, value in (soap_json or {}).items():
        if isinstance(value, str):
            out[key] = parse_anchored_soap(value)
        else:
            out[key] = [{"type": "text", "content": str(value or "")}]
    return out


def normalize_anchor_tags_colon_to_pipe(
    soap_text: str,
    entity_manifest: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Normalize SOAP-gen anchor tags from [[Eid:display_text]] to [[Eid|canonical]] using manifest.
    Only replaces tags that use colon (SOAP gen output); pipe tags are left unchanged.
    Builds anchor_id -> canonical term from manifest (kb_preferred_name or display_name).
    """
    if not soap_text or not entity_manifest:
        return soap_text
    # Pattern for colon form only (SOAP gen writes [[E1:term]])
    colon_pattern = re.compile(r"\[\[(E\d+):(.*?)\]\]", re.DOTALL)
    aid_to_canonical: Dict[str, str] = {}
    for e in entity_manifest:
        if not isinstance(e, dict):
            continue
        aid = e.get("anchor_id")
        if not aid:
            continue
        canonical = (
            (e.get("kb_preferred_name") or e.get("display_name") or e.get("normalized_name") or "").strip()
        )
        if canonical:
            aid_to_canonical[aid] = canonical

    def repl(m: re.Match) -> str:
        aid, display = m.group(1), m.group(2)
        canonical = aid_to_canonical.get(aid)
        if canonical is not None:
            return f"[[{aid}|{canonical}]]"
        if logger:
            logger.debug("Anchor %s not in manifest, keeping display: %s", aid, display[:50])
        return f"[[{aid}|{display}]]"

    out = colon_pattern.sub(repl, soap_text)
    return out

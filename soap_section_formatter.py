"""
Normalize SOAP list sections into clean numbered lines.

Target shape (one item per line):
  1. First item
  2. Second item
  3. Third item

Follow-up content is consolidated into a single Reminders section
(FollowUpInstructions / FollowUpReminders are merged away to cut token cost).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Sections that must render as numbered lists
NUMBERED_SECTIONS = (
    "Plan",
    "KeyIssues",
    "AbnormalFindings",
    "CustomerInstructions",
    "Reminders",
    "DifferentialDiagnosis",
)

# Legacy keys merged into Reminders (not emitted in new SOAP schema)
_LEGACY_FOLLOWUP_KEYS = ("FollowUpInstructions", "FollowUpReminders")


def _split_items(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []
    raw = str(text).strip()

    if "\n" in raw:
        parts = re.split(r"\n+", raw)
    elif re.search(r"(?:^|\s)\d+[.)]\s+\S", raw):
        parts = re.split(r"(?:(?<=\D)|^)\s*\d+[.)]\s+", raw)
    elif ";" in raw:
        parts = raw.split(";")
    else:
        sentence_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", raw)
        if len(sentence_parts) >= 2:
            parts = sentence_parts
        else:
            comma_parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(comma_parts) >= 3 and all(len(p) < 120 for p in comma_parts):
                parts = comma_parts
            else:
                parts = [raw]

    items: List[str] = []
    for p in parts:
        s = (p or "").strip()
        if not s:
            continue
        s = re.sub(r"^\s*(?:\d+[.)]\s*|[-•–—]\s*)", "", s).strip()
        if re.match(
            r"^(?:follow[-\s]?up\s+(?:instructions?|reminders?)|customer\s+instructions?|reminders?)\s*:?\s*$",
            s,
            flags=re.I,
        ):
            continue
        if s:
            items.append(s)
    return items


def format_numbered_list(text: str) -> str:
    """Return newline-separated `1. …\\n2. …` (empty string if no content)."""
    items = _split_items(text)
    if not items:
        return ""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def _dedupe_items(items: List[str]) -> List[str]:
    """Drop exact/near-duplicate reminder lines (token-cost + UX)."""
    cleaned: List[str] = []
    for item in items:
        s = item.strip()
        if not s:
            continue
        s = re.sub(r"^(?:reminder to |remind (?:the owner to |to ))", "", s, flags=re.I).strip()
        cleaned.append(s)

    out: List[str] = []
    for item in cleaned:
        key_tokens = set(re.findall(r"[a-z0-9]+", item.lower()))
        key_tokens -= {"the", "a", "an", "to", "for", "of", "and", "or", "if", "any", "in"}
        drop = False
        replace_idx = None
        for i, prev in enumerate(out):
            prev_tokens = set(re.findall(r"[a-z0-9]+", prev.lower()))
            prev_tokens -= {"the", "a", "an", "to", "for", "of", "and", "or", "if", "any", "in"}
            if not key_tokens or not prev_tokens:
                continue
            overlap = len(key_tokens & prev_tokens) / max(1, min(len(key_tokens), len(prev_tokens)))
            if overlap >= 0.7:
                # Keep the longer, more specific line
                if len(item) > len(prev):
                    replace_idx = i
                else:
                    drop = True
                break
        if drop:
            continue
        if replace_idx is not None:
            out[replace_idx] = item
        else:
            out.append(item)
    return out


def merge_reminders_fields(soap: Dict[str, Any]) -> str:
    """
    Combine Reminders + legacy FollowUp* into one numbered list.
    Preference order preserved; near-duplicates dropped.
    """
    chunks: List[str] = []
    for key in ("Reminders",) + _LEGACY_FOLLOWUP_KEYS:
        val = soap.get(key)
        if isinstance(val, str) and val.strip():
            chunks.extend(_split_items(val))
    return format_numbered_list("\n".join(_dedupe_items(chunks)))


def format_soap_dict(soap: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize list-like fields; collapse follow-ups into Reminders only."""
    if not isinstance(soap, dict):
        return soap
    out = dict(soap)

    # Merge follow-up siblings into Reminders first
    merged = merge_reminders_fields(out)
    out["Reminders"] = merged
    for key in _LEGACY_FOLLOWUP_KEYS:
        out.pop(key, None)

    for key in NUMBERED_SECTIONS:
        if key == "Reminders":
            continue  # already merged+numbered
        if key in out and isinstance(out[key], str) and out[key].strip():
            out[key] = format_numbered_list(out[key])

    return out


def format_soap_note_text(soap_note: str, logger: Optional[Any] = None) -> str:
    """Format a SOAP note string (JSON object preferred)."""
    if not soap_note or not str(soap_note).strip():
        return soap_note
    text = str(soap_note).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            formatted = format_soap_dict(obj)
            return json.dumps(formatted, indent=2, ensure_ascii=False)
    except Exception:
        pass
    if logger:
        try:
            logger.debug("SOAP formatter: note was not JSON; left unchanged")
        except Exception:
            pass
    return soap_note

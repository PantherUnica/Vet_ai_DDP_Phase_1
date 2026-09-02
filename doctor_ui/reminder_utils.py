"""Shared reminder filtering and Phase 2 intent helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

INTENT_NOT_RECOGNISED = "Intent not recognised"

KNOWN_INTENT_CONTEXTS = frozenset({
    "Performed",
    "Prescribed",
    "Administered",
    "Ordered",
    "Scheduled",
    "Measured",
    "Declined",
    "Presented",
    "Recommended",
    "Future",
    "Reminder",
    INTENT_NOT_RECOGNISED,
})

MEDICINE_INTENTS = frozenset({"Prescribed", "Administered", "Ordered"})
RECOMMENDATION_INTENT = "Recommended"

# Actionable: scheduling / due-date cues
_ACTIONABLE_PATTERNS = re.compile(
    r"\b("
    r"recheck|re-check|follow[- ]?up|followup|return visit|callback|call back|"
    r"schedule|book|appointment|vaccin|booster|surgery|suture removal|"
    r"recall|next visit|in \d+ (day|week|month)|after \d+ (day|week|month)|"
    r"next month|next week|tomorrow|asap|due date|lab (test|work)|x-?ray|"
    r"ultrasound|cbc|blood test|reculture|re-culture|physio|hydrotherapy"
    r")\b",
    re.IGNORECASE,
)

# Non-actionable: passive / conditional-only monitoring
_NON_ACTIONABLE_PATTERNS = re.compile(
    r"\b("
    r"if (symptoms )?worsen|if needed|as needed|watch|monitor|observe|"
    r"keep an eye|contact (us )?if|call (us )?if|return if|unless|"
    r"should (symptoms|signs)|appetite|water intake|activity level"
    r")\b",
    re.IGNORECASE,
)


def normalise_intent(intent: Any) -> str:
    """Return stripped intent_context or empty string."""
    return str(intent or "").strip()


def is_intent_recognised(intent: Any) -> bool:
    text = normalise_intent(intent)
    return bool(text) and text in KNOWN_INTENT_CONTEXTS


def intent_display(intent: Any) -> str:
    text = normalise_intent(intent)
    if not text:
        return INTENT_NOT_RECOGNISED
    return text


def _parse_numbered_lines(text: str) -> List[str]:
    """Split numbered list text into individual items."""
    raw = (text or "").strip()
    if not raw:
        return []
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if line:
            lines.append(line)
    if not lines and raw:
        lines = [raw]
    return lines


def _item_text(item: Dict[str, Any]) -> str:
    parts = [
        item.get("item_name"),
        item.get("name"),
        item.get("remarks"),
        item.get("due_date"),
    ]
    return " ".join(str(p) for p in parts if p).strip()


def is_actionable_reminder_text(text: str) -> bool:
    """True when reminder line needs scheduling or has explicit timing."""
    line = (text or "").strip()
    if not line:
        return False
    if _ACTIONABLE_PATTERNS.search(line):
        # Conditional-only lines that also mention scheduling stay actionable
        if _NON_ACTIONABLE_PATTERNS.search(line) and not re.search(
            r"\b(recheck|schedule|book|appointment|vaccin|in \d+|next (week|month|visit))\b",
            line,
            re.IGNORECASE,
        ):
            return False
        return True
    if _NON_ACTIONABLE_PATTERNS.search(line):
        return False
    return False


def is_actionable_reminder_item(item: Dict[str, Any]) -> bool:
    """Filter structured Phase 2 reminder rows to actionable-only."""
    intent = normalise_intent(item.get("intent_context"))
    due = (item.get("due_date") or "").strip()
    if intent in ("Scheduled", "Future"):
        return True
    if due and due.upper() not in ("ASAP", ""):
        return True
    item_id = item.get("item_id")
    if item_id and str(item_id).replace("-", "").isdigit():
        return True
    return is_actionable_reminder_text(_item_text(item))


def filter_actionable_reminders_text(text: str) -> Tuple[str, int]:
    """Filter Phase 1 Reminders prose; return (filtered_text, hidden_count)."""
    items = _parse_numbered_lines(text)
    if not items:
        return (text or "").strip(), 0
    kept = [it for it in items if is_actionable_reminder_text(it)]
    hidden = len(items) - len(kept)
    if not kept:
        return "", hidden
    numbered = "\n".join(f"{i + 1}. {it}" for i, it in enumerate(kept))
    return numbered, hidden


def filter_actionable_reminder_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter Phase 2 reminder list to actionable-only."""
    return [it for it in (items or []) if is_actionable_reminder_item(it)]


def merge_reminder_sources(
    primary: List[Dict[str, Any]],
    extra: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge reminder lists deduping by normalised item_name."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in list(primary or []) + list(extra or []):
        name = " ".join((item.get("item_name") or item.get("name") or "").lower().split())
        if name and name in seen:
            continue
        if name:
            seen.add(name)
        out.append(item)
    return out


def _self_check() -> None:
    assert is_actionable_reminder_text("Recheck skin in 2-3 days")
    assert is_actionable_reminder_text("Schedule vaccination booster next month")
    assert not is_actionable_reminder_text("Return if symptoms worsen")
    assert not is_actionable_reminder_text("Watch appetite and water intake")

    filtered, hidden = filter_actionable_reminders_text(
        "1. Recheck in 3 days\n2. Return if worsening\n3. Vaccine booster next month"
    )
    assert "Recheck" in filtered and "Vaccine" in filtered
    assert hidden == 1

    assert is_actionable_reminder_item({"intent_context": "Scheduled", "item_name": "Recheck"})
    assert not is_actionable_reminder_item({"intent_context": "Reminder", "item_name": "Watch appetite"})
    assert not is_intent_recognised("")
    assert intent_display("") == INTENT_NOT_RECOGNISED


if __name__ == "__main__":
    _self_check()
    print("reminder_utils self-check ok")

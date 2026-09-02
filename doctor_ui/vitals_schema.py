"""Canonical vitals field definitions (from vitals.xlsx) for SOAP extraction and UI display."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VitalsFieldDef:
    label: str
    db_key: str
    data_type: str
    unit: str
    allowed_values: str


VITALS_FIELD_DEFS: List[VitalsFieldDef] = [
    VitalsFieldDef("Body Weight", "body_weight", "Decimal", "kg", "User-entered number"),
    VitalsFieldDef("Body Temperature", "body_temperature", "Decimal", "°F", "User-entered number"),
    VitalsFieldDef("Heart Rate", "heart_rate", "Integer", "bpm", "User-entered number"),
    VitalsFieldDef("Respiratory Rate", "respiratory_rate", "Integer", "brpm", "User-entered number"),
    VitalsFieldDef(
        "Capillary Refill Time",
        "capillary_refill_time",
        "Enum",
        "seconds",
        "Normal (<2s) / Prolonged (≥2s) / Very prolonged (≥3s) / Not assessed",
    ),
    VitalsFieldDef(
        "Mucous Membrane Color",
        "mucous_membrane_color",
        "Enum",
        "—",
        "Pink / Pale / White / Cyanotic / Injected / Icteric / Muddy / Not assessed",
    ),
    VitalsFieldDef(
        "Body Condition Score",
        "body_condition_score",
        "Integer enum",
        "Scale",
        "1–9 recommended: 1, 2, 3, 4, 5, 6, 7, 8, 9 / Not assessed",
    ),
    VitalsFieldDef(
        "Pain Score",
        "pain_score",
        "Enum",
        "scale",
        "None / Mild / Moderate / Severe",
    ),
    VitalsFieldDef(
        "Mentation",
        "mentation",
        "Enum",
        "Status",
        "BAR / QAR / Depressed / Obtunded / Stupor / Coma / Agitated / Not assessed",
    ),
    VitalsFieldDef(
        "Hydration Status",
        "hydration_status",
        "Enum",
        "level",
        "Normal / Mild / Moderate / Severe / Overhydrated",
    ),
    VitalsFieldDef(
        "Pulse Quality",
        "pulse_quality",
        "Enum",
        "Quality",
        "Strong / Normal / Weak / Thready / Bounding / Irregular / Not assessed",
    ),
]

VITALS_DB_KEYS: List[str] = [f.db_key for f in VITALS_FIELD_DEFS]

_VITALS_KEY_TO_DEF: Dict[str, VitalsFieldDef] = {f.db_key: f for f in VITALS_FIELD_DEFS}


def vitals_json_schema_properties() -> Dict[str, Any]:
    """JSON-schema property map for SOAP Vitals object."""
    return {key: {"type": "string"} for key in VITALS_DB_KEYS}


def build_vitals_prompt_block() -> str:
    """Prompt text listing all vitals fields for LLM extraction."""
    lines = [
        "VITALS TEMPLATE (use these db_key names in the Vitals JSON object):",
    ]
    for field in VITALS_FIELD_DEFS:
        unit_part = f" ({field.unit})" if field.unit and field.unit != "—" else ""
        lines.append(
            f"- {field.label} [{field.db_key}]{unit_part}: {field.data_type}. "
            f"Allowed: {field.allowed_values}"
        )
    return "\n".join(lines)


def _is_empty_vital_value(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return True
    return text.lower().replace("_", " ") in {"not assessed", "n/a", "na", "none", "unknown"}


def normalize_vitals(raw: Any) -> Dict[str, str]:
    """
    Normalize Vitals to a dict of db_key -> value string (only filled fields).
    Accepts dict, JSON string, legacy plain text, or empty values.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out: Dict[str, str] = {}
        for key, value in raw.items():
            if key not in _VITALS_KEY_TO_DEF:
                continue
            text = str(value).strip() if value is not None else ""
            if text and not _is_empty_vital_value(text):
                out[key] = text
        return out
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return normalize_vitals(parsed)
            except json.JSONDecodeError:
                pass
        # Legacy plain-text: "Body Weight: 12.5 kg" lines
        out = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            label_part, _, value_part = line.partition(":")
            value_part = value_part.strip()
            if not value_part:
                continue
            label_key = label_part.strip().lower().replace(" ", "_")
            for field in VITALS_FIELD_DEFS:
                if (
                    field.label.lower() == label_part.strip().lower()
                    or field.db_key == label_key
                ):
                    out[field.db_key] = value_part
                    break
        return out
    return {}


def vitals_rows_for_display(vitals: Any) -> List[Dict[str, str]]:
    """Rows for UI table: Field, Value, Unit (only filled vitals, template order)."""
    normalized = normalize_vitals(vitals)
    rows: List[Dict[str, str]] = []
    for field in VITALS_FIELD_DEFS:
        value = normalized.get(field.db_key)
        if not value:
            continue
        unit = field.unit if field.unit != "—" else ""
        rows.append({"Field": field.label, "Value": value, "Unit": unit})
    return rows

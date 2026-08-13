"""
Reusable helpers for local inventory/service internal_description and embedding texts.

This module mirrors the templates and embedding inputs used in
`backfill_soap_domain_and_vetbert.py`, but exposes them as pure functions that
can be imported from other modules (e.g. local search, audits, tools).

Key ideas:
  - INTERNAL DESCRIPTION is the canonical structured text stored in the DB.
  - OPENAI EMBEDDING uses the same text as internal_description.
  - VETBERT EMBEDDING uses "domain: {body}" where body is the same template.

INVENTORY TEMPLATE (only non-empty parts, in this order):
  Product: [trade_name] [trade_name].
  Generic: [item_name] [item_name].
  Category: [category] / [sub_category].
  Specialty: [domain_key].
  Use Case: [brief_description].
  Ingredients: [major_active_ingredients].

SERVICE TEMPLATE (only non-empty parts, in this order):
  Procedure: [procedure_name] [procedure_name].
  Variant: [variant_name] [variant_name].
  Category: [type] > [category] > [sub_category].
  Specialty: [domain_key].
  Technical: [modality] [sample_type].
"""

from typing import Any, Dict
import re


def _clean_text_for_embedding(text: str) -> str:
    """Light cleanup: collapse whitespace, strip trailing punctuation gaps."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    while " . " in t:
        t = t.replace(" . ", " ")
    if t.endswith(" ."):
        t = t[:-2]
    return t.strip()


def build_inventory_internal_description(row: Dict[str, Any]) -> str:
    """
    Build the canonical internal_description for an inventory row.

    Expected keys in `row` (missing keys are treated as empty strings):
      - trade_name
      - item_name
      - category
      - sub_category
      - domain_key
      - brief_description
      - major_active_ingredients
    """
    trade = (row.get("trade_name") or "").strip()
    generic = (row.get("item_name") or "").strip()
    cat = (row.get("category") or "").strip()
    sub = (row.get("sub_category") or "").strip()
    domain = (row.get("domain_key") or "").strip()
    raw_use = (row.get("brief_description") or "").strip()
    use_case = _clean_text_for_embedding(raw_use)
    ingredients = (row.get("major_active_ingredients") or "").strip()

    sections = []
    if trade:
        sections.append(f"Product: {trade} {trade}.")
    if generic:
        sections.append(f"Generic: {generic} {generic}.")
    if cat:
        cat_display = f"{cat} / {sub}".strip(" /") if sub else cat
        sections.append(f"Category: {cat_display}.")
    if domain:
        sections.append(f"Specialty: {domain}.")
    if use_case:
        sections.append(f"Use Case: {use_case}.")
    if ingredients:
        sections.append(f"Ingredients: {ingredients}.")
    return " ".join(sections) if sections else (trade or generic or "")


def build_inventory_openai_text(row: Dict[str, Any]) -> str:
    """
    Text for OpenAI embeddings for inventory.

    By design this is exactly the same as the internal_description template, so
    callers can simply do:

        body = build_inventory_openai_text(row)
        vec = embed(body)
    """
    return build_inventory_internal_description(row)


def build_inventory_vetbert_text(row: Dict[str, Any]) -> str:
    """
    Text for VetBERT embeddings for inventory.

    Format:
      "{domain_key}: {inventory_internal_description}"
    """
    domain = (row.get("domain_key") or "general").strip()
    body = build_inventory_internal_description(row)
    return f"{domain}: {body}" if body else f"{domain}:"


def build_service_internal_description(row: Dict[str, Any]) -> str:
    """
    Build the canonical internal_description for a service_master row.

    Expected keys in `row`:
      - procedure_name
      - variant_name
      - type
      - category
      - sub_category
      - domain_key
      - modality
      - sample_type
    """
    procedure = (row.get("procedure_name") or "").strip()
    variant = (row.get("variant_name") or "").strip()
    type_ = (row.get("type") or "").strip()
    cat = (row.get("category") or "").strip()
    sub = (row.get("sub_category") or "").strip()
    domain = (row.get("domain_key") or "").strip()
    modality = (row.get("modality") or "").strip()
    sample_type = (row.get("sample_type") or "").strip()

    sections = []
    if procedure:
        sections.append(f"Procedure: {procedure} {procedure}.")
    if variant:
        sections.append(f"Variant: {variant} {variant}.")
    if type_ or cat or sub:
        cat_parts = [p for p in [type_, cat, sub] if p]
        sections.append(f"Category: {' > '.join(cat_parts)}.")
    if domain:
        sections.append(f"Specialty: {domain}.")
    tech_parts = [p for p in [modality, sample_type] if p]
    if tech_parts:
        sections.append(f"Technical: {' '.join(tech_parts)}.")
    return " ".join(sections) if sections else (procedure or "")


def build_service_openai_text(row: Dict[str, Any]) -> str:
    """
    Text for OpenAI embeddings for services.

    Same as the structured internal_description template.
    """
    return build_service_internal_description(row)


def build_service_vetbert_text(row: Dict[str, Any]) -> str:
    """
    Text for VetBERT embeddings for services.

    Format:
      "{domain_key}: {service_internal_description}"
    """
    domain = (row.get("domain_key") or "general").strip()
    body = build_service_internal_description(row)
    return f"{domain}: {body}" if body else f"{domain}:"


__all__ = [
    "build_inventory_internal_description",
    "build_inventory_openai_text",
    "build_inventory_vetbert_text",
    "build_service_internal_description",
    "build_service_openai_text",
    "build_service_vetbert_text",
]


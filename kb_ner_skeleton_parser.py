"""
Skeleton Parser for Brain NER Compressed Format

Converts compressed "skeleton" format to full JSON entities.
Format: id|span_text|normalized_name|kind|domains|inv_cats|svc_cats|corr_prob|sugg_prob|hints|is_new|context|query_expansion

Where:
- domains: comma-separated list (e.g., "orthopedic,dermatology")
- inv_cats: comma-separated list (e.g., "Medication,Pet Supplies")
- svc_cats: comma-separated list (e.g., "Consultation")
- hints: comma-separated, optionally with probabilities (e.g., "hint1:0.9,hint2:0.8" or just "hint1,hint2")
- is_new: 0 or 1
- context: optional context sentence
- query_expansion: (13th field) up to 3 comma-separated likely brand/product names when term is phonetic/garbled

Example:
e1|Spirocoxin|Spirocoxin|Medication|orthopedic|Medication|Consultation|0.95|0.95|NSAID:0.9,anti-inflammatory:0.8|0|Given for pain
"""

import re
from typing import List, Dict, Any, Union


def _normalize_skeleton_lines(skeleton_input: Union[str, List[str]]) -> List[str]:
    """
    Normalize skeleton input into a list of line strings.

    Supports:
    - Single string with newline/semicolon-delimited entities
    - List of strings where each item is one entity line
    """
    if isinstance(skeleton_input, list):
        return [str(line).strip() for line in skeleton_input if str(line).strip()]
    if isinstance(skeleton_input, str):
        return [line.strip() for line in re.split(r'[\n;]+', skeleton_input.strip()) if line.strip()]
    return []


def parse_skeleton_entities(skeleton_input: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Parse compressed skeleton format into full JSON entities.
    
    Format: id|span_text|normalized_name|kind|domains|inv_cats|svc_cats|corr_prob|sugg_prob|hints|is_new|context|query_expansion
    
    Args:
        skeleton_input: Compressed format string OR list of skeleton lines.
    
    Returns:
        List of entity dictionaries matching Brain NER JSON schema
    """
    entities = []
    
    lines = _normalize_skeleton_lines(skeleton_input)
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue
        
        # Split by pipe. Be tolerant: some model outputs omit trailing fields.
        parts = line.split('|')
        if len(parts) < 4:  # Must at least have id|span|normalized|kind
            continue
        if len(parts) < 13:
            parts = parts + ([""] * (13 - len(parts)))
        
        try:
            entity_id = parts[0].strip()
            span_text = parts[1].strip()
            normalized_name = parts[2].strip() or span_text
            kind = parts[3].strip()
            
            # Parse arrays (comma-separated)
            # Handle empty arrays: if field is "[]" or empty, return empty list
            def parse_array_field(field_str: str) -> List[str]:
                field_str = field_str.strip()
                if not field_str or field_str == '[]':
                    return []
                # Split by comma and filter out empty strings and "[]" literals
                items = [item.strip() for item in field_str.split(',') if item.strip() and item.strip() != '[]']
                return items

            def _drop_numeric_categories(items: List[str]) -> List[str]:
                """Keep only valid category/kind strings; drop numeric-only tokens (e.g. leaked '0.95')."""
                out = []
                for s in items:
                    if not s:
                        continue
                    try:
                        float(s)
                        continue  # skip numeric
                    except (ValueError, TypeError):
                        out.append(s)
                return out

            domains = parse_array_field(parts[4])
            inv_cats = _drop_numeric_categories(parse_array_field(parts[5]))
            svc_cats = _drop_numeric_categories(parse_array_field(parts[6]))
            
            # Parse probabilities
            try:
                corr_prob = float(parts[7].strip()) if parts[7].strip() else 0.95
            except Exception:
                corr_prob = 0.95
            try:
                sugg_prob = float(parts[8].strip()) if parts[8].strip() else 0.95
            except Exception:
                sugg_prob = 0.95
            
            def _is_placeholder_hint(s: str) -> bool:
                t = (s or "").strip().lower()
                return t in {"0", "none", "null", "n/a", "na", "-", "--"}

            # Parse hints (comma-separated, optionally with probabilities)
            hints_raw = parts[9].strip() if len(parts) > 9 else ""
            hints = []
            if hints_raw:
                for hint_part in hints_raw.split(','):
                    hint_part = hint_part.strip()
                    if not hint_part or _is_placeholder_hint(hint_part):
                        continue
                    if ':' in hint_part:
                        # Format: hint:probability
                        hint_text, prob_str = hint_part.rsplit(':', 1)
                        if _is_placeholder_hint(hint_text):
                            continue
                        try:
                            prob = float(prob_str.strip())
                            hints.append({"hint": hint_text.strip(), "probability": prob})
                        except ValueError:
                            hints.append(hint_text.strip())
                    else:
                        hints.append(hint_part)
            
            # Parse is_new (0 or 1)
            is_new = parts[10].strip() == '1' if len(parts) > 10 else False
            
            # Parse context (optional, 11th field)
            context = parts[11].strip() if len(parts) > 11 else ""
            
            # Parse query_expansion (13th field): up to 3 comma-separated likely brand/product names when term is phonetic/garbled
            query_expansion_raw = parts[12].strip() if len(parts) > 12 else ""
            query_expansion: List[str] = []
            if query_expansion_raw:
                for x in query_expansion_raw.split(","):
                    x = x.strip()
                    if not x or _is_placeholder_hint(x):
                        continue
                    query_expansion.append(x)
            query_expansion = query_expansion[:3]  # Cap at 3
            
            entity = {
                "id": entity_id,
                "span_text": span_text,
                "normalized_name": normalized_name,
                "kind": kind,
                "domain": domains,
                "inventory_category": inv_cats,
                "service_category": svc_cats,
                "correctness_probability": corr_prob,
                "suggestion_probability": sugg_prob,
                # Hard rule: placeholder/noisy hints are dropped at parser boundary; keep [] if none.
                "hints": hints,
                "is_new": is_new,
                "query_expansion": query_expansion,
            }
            
            if context:
                entity["context_sentence"] = context
            
            entities.append(entity)
            
        except (ValueError, IndexError) as e:
            # Skip malformed lines
            continue
    
    return entities


def format_skeleton_example() -> str:
    """Return example skeleton format for prompt."""
    return """e1|Spirocoxin|Spirocoxin|Medication|orthopedic|Medication|Consultation|0.95|0.95|NSAID:0.9,anti-inflammatory:0.8|0|Given for pain
e2|hip displasia|hip dysplasia|Diagnosis|orthopedic|||0.75|0.95|hip dysplasia:0.95,CHD:0.7|0|X-ray shows hip displasia
e3|walking problem|lameness|Symptom|orthopedic|||0.90|0.85|lameness:0.9,abnormal gait:0.85|0|Owner reports walking problem"""

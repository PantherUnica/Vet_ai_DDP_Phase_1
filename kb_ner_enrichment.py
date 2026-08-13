"""
Entity enrichment utilities.

This module handles:
- Attribute schema retrieval
- Slang to clinical term translation
"""

import json
import logging
from typing import List, Dict, Any, Optional

from kb_ner_db import get_pg_conn, pg_conn_ctx
from kb_ner_clients import get_openai_client

def get_attributes_for_kind(
    source_kind: str,
    conn=None,
    logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Get all valid attributes for a given source kind from kb.attributes_schema.
    
    Args:
        source_kind: The kind to get attributes for (e.g., "Medicine", "Procedure", "Diagnostic")
        conn: Optional database connection (will create if not provided)
        logger: Optional logger
        
    Returns:
        List of dicts with keys: relationship, target_attribute, use_case, is_required
    """
    if conn is None:
        try:
            with pg_conn_ctx(logger=logger) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT relationship, target_attribute, use_case, is_required
                        FROM kb.attributes_schema
                        WHERE source_kind = %s
                        ORDER BY is_required DESC, relationship;
                    """, (source_kind,))
                    rows = cur.fetchall()
                    
                    results = []
                    for row in rows:
                        results.append({
                            "relationship": row[0],
                            "target_attribute": row[1],
                            "use_case": row[2],
                            "is_required": row[3],
                        })
                    return results
        except Exception as e:
            if logger:
                logger.error(f"Could not connect to database: {e}")
            return []
    else:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT relationship, target_attribute, use_case, is_required
                    FROM kb.attributes_schema
                    WHERE source_kind = %s
                    ORDER BY is_required DESC, relationship;
                """, (source_kind,))
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        "relationship": row[0],
                        "target_attribute": row[1],
                        "use_case": row[2],
                        "is_required": row[3],
                    })
                return results
        except Exception as e:
            if logger:
                logger.warning(f"Error fetching attributes for kind '{source_kind}': {e}")
            return []


# Alias for backward compatibility
get_attributes_schema = get_attributes_for_kind


def translate_slang_to_clinical_term(
    slang_term: str,
    entity_kind: str,
    context: Optional[str] = None,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """
    Lazy Two-Pass Strategy - Rescue Pass: Translate slang/layman terms to clinical terms.
    
    This is triggered only when vector search fails (score < 0.45) to avoid latency penalty
    for terms that vector search can already handle.
    
    Universal Architecture: Works for ALL categories (Drugs, Procedures, Anatomy, Findings, etc.)
    - "Safai" (Hindi) → "Dental Scaling" (Procedure)
    - "Not doing potty" → "Constipation" (Condition)
    - "Big belly" → "Ascites" (Finding)
    - "Crocin" → "Paracetamol" (Drug)
    
    Args:
        slang_term: Slang/layman term to translate
        entity_kind: Entity kind
        context: Optional context
        client: OpenAI client
        logger: Optional logger
        
    Returns:
        Clinical term if translation succeeds, None otherwise
    """
    if not slang_term or not slang_term.strip():
        return None
    
    # Don't translate if it looks like a clinical term already
    medical_keywords = ["disease", "syndrome", "disorder", "condition", "diagnosis", "test", "procedure", "scaling", "prophylaxis"]
    if any(keyword in slang_term.lower() for keyword in medical_keywords):
        if logger:
            logger.debug(f"  ℹ️  Skipping translation for '{slang_term}' - appears to be clinical term already")
        return None
    
    if not client:
        client = get_openai_client()
        if not client:
            if logger:
                logger.warning("  ⚠️  Cannot translate slang term - OpenAI client not available")
            return None
    
    try:
        prompt = f"""You are a veterinary clinical terminology expert.

TASK: Translate a layman/slang/local language term to the standard clinical term.

LAYMAN TERM: "{slang_term}"
ENTITY KIND: {entity_kind}
CONTEXT: {context[:200] if context else "Not provided"}

INSTRUCTIONS:
1. If the term is already a clinical term (e.g., "Constipation", "Vomiting"), return it unchanged.
2. If it's slang/layman/local language (e.g., "not doing potty", "Safai", "Big belly", "Crocin"), return the standard clinical term.
3. Return ONLY the clinical term, no explanation, no JSON, just the term.
4. If you're uncertain, return the most likely clinical term.

EXAMPLES:
- "not doing potty" → "Constipation"
- "Safai" (Hindi for cleaning) → "Dental Scaling" or "Dental Prophylaxis"
- "Big belly" → "Ascites" or "Abdominal Distension"
- "Crocin" (human brand) → "Paracetamol" or "Acetaminophen"
- "Water in the stomach" → "Ascites" or "Abdominal Effusion"
- "Constipation" → "Constipation" (already clinical)

CLINICAL TERM:"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a veterinary clinical terminology expert. Return only the clinical term, no explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=50,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        clinical_term = content.strip() if content else None
        
        # Clean up: remove quotes, JSON markers, etc.
        if clinical_term:
            clinical_term = clinical_term.strip('"\'`').strip()
            # Remove JSON markers if present
            if clinical_term.startswith("{"):
                try:
                    parsed = json.loads(clinical_term)
                    clinical_term = parsed.get("clinical_term") or parsed.get("term") or clinical_term
                except:
                    pass
        
        if clinical_term and clinical_term.lower() != slang_term.lower():
            if logger:
                latency = getattr(timer, 'seconds', 0.0)
                logger.info(f"  🔄 Rescue Pass: Translated '{slang_term}' → '{clinical_term}' (latency: {latency:.2f}s)")
            return clinical_term
        else:
            if logger:
                logger.debug(f"  ℹ️  Translation returned same term or empty for '{slang_term}'")
            return None
            
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠️  Failed to translate slang term '{slang_term}': {e}")
        return None

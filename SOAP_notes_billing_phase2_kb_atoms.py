"""
Veterinary Knowledge Atom Extractor - Phase 2

This module extracts structured Knowledge Atoms from veterinary SOAP notes
using kb.assertion_types and kb.attributes_schema as the "Instruction Manual."

Features:
- Processes veterinary SOAP notes to extract clinical intents
- Uses kb.assertion_types to tag clinical status (CONF, NEG, SUSP, HIST, HYPO, RECUR)
- Uses kb.attributes_schema to structure relationships (dose→drug, route→medication, etc.)
- Outputs structured JSON Knowledge Atoms for downstream use
- Supports multiple LLM providers including Llama models via Fireworks
- Comprehensive error handling and logging
- Configurable model selection

Architecture:
Phase 1: Voice Transcript → Natural Language SOAP Note (Text Generation)
Phase 2: SOAP Note + KB Schema → Knowledge Atoms (Clinical Intent Extraction) ← THIS MODULE
Knowledge Atom Structure:
{
  "concept": "Meloxicam",
  "kind": "Medicine",
  "assertion_id": "CONF",
  "attributes": {
    "dose": "5mg",
    "route": "PO"
  },
  "intent_type": "Administered"
}

Configuration:
- Set MODEL_PROVIDER and MODEL_NAME at the top of the file.
- API keys are loaded from API_Key.txt in the same directory.

Input Files:
1. SOAP Note (Required): Text file containing the complete SOAP note.

Author: VetInstant P.A.W.S Team
Version: 3.0 (Knowledge Atoms Architecture)
"""
from __future__ import annotations

import os
import logging
import time
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import threading

# ==============================================================================
# PHASE 2 NOTES
# ==============================================================================
# The intended "core job" of Phase 2 is:
#   - Extract structured Knowledge Atoms:
#       concept + kind + assertion_id + attributes (+ intent_context/section/source_text)
# using kb.assertion_types and kb.attributes_schema as the instruction manual.

# Import KB schema helpers
try:
    # Prefer light imports to avoid pulling in the full Phase 1 linker.
    # Phase 2 only needs DB access + (optional) KB lookups for enrichment.
    from kb_ner_db import get_pg_conn, pg_conn_ctx, close_owned_pg_conn  # lightweight DB connector + context manager
    from kb_ner_global_search import kb_lookup_concept_exact, kb_lookup_concept_by_embedding
    from kb_ner_enrichment import get_attributes_for_kind
    KB_SCHEMA_AVAILABLE = True
except ImportError:
    KB_SCHEMA_AVAILABLE = False
    logging.warning("kb_ner_linker not available. KB schema features will be disabled.")
    # Fallback stubs
    def get_pg_conn(*args, **kwargs):
        return None
    def pg_conn_ctx(*args, **kwargs):
        from contextlib import nullcontext
        return nullcontext()
    def close_owned_pg_conn(conn):
        pass

# ==============================================================================
# KB SCHEMA LOADERS (Phase 2)
# ==============================================================================

_VITALS_REGISTRY_EXCERPT_LOCK = threading.Lock()
_VITALS_REGISTRY_EXCERPT_CACHE: Optional[str] = None


def get_vitals_registry_excerpt(
    conn=None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Pull a compact vitals taxonomy anchor from kb.vitals_registry for prompt injection.

    Why:
    - Phase 1 is identity/grounding; Phase 2 is role/kind + attribute extraction.
    - This anchor reduces kind drift (Finding vs VitalSign) and improves unit/field precision.

    Behavior:
    - Cached once per process (no per-run DB overhead).
    - Bounded size (caps rows + synonyms per row).
    - Returns "" on any DB error (safe fallback to static instructions).
    """
    global _VITALS_REGISTRY_EXCERPT_CACHE

    # Toggle (default ON)
    if os.getenv("PHASE2_VITALS_REGISTRY_INJECT", "true").strip().lower() not in ("1", "true", "yes"):
        return ""

    with _VITALS_REGISTRY_EXCERPT_LOCK:
        if _VITALS_REGISTRY_EXCERPT_CACHE is not None:
            return _VITALS_REGISTRY_EXCERPT_CACHE

        should_close = False
        if conn is None:
            try:
                # reuse=False: do not touch the shared grounding connection
                conn = get_pg_conn(reuse=False)
                should_close = True
            except Exception as e:
                if logger:
                    logger.debug(f"Vitals registry excerpt: DB connect failed: {e}")
                _VITALS_REGISTRY_EXCERPT_CACHE = ""
                return ""

        try:
            max_rows = int(os.getenv("PHASE2_VITALS_REGISTRY_MAX_ROWS", "30"))
        except Exception:
            max_rows = 30
        try:
            max_syn = int(os.getenv("PHASE2_VITALS_REGISTRY_MAX_SYNONYMS", "5"))
        except Exception:
            max_syn = 5

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT metric_name, expected_unit, synonyms
                    FROM kb.vitals_registry
                    ORDER BY metric_name
                    LIMIT %s;
                    """,
                    (max_rows,),
                )
                rows = cur.fetchall()

            lines: List[str] = []
            for metric_name, expected_unit, synonyms in rows:
                mn = (metric_name or "").strip()
                if not mn:
                    continue
                unit = (expected_unit or "").strip()
                syn_list = synonyms if isinstance(synonyms, list) else []
                syn_list = [str(s).strip() for s in syn_list if s and str(s).strip()]
                syn_list = syn_list[:max_syn]

                syn_text = f" | synonyms: {', '.join(syn_list)}" if syn_list else ""
                unit_text = f" [{unit}]" if unit else ""
                lines.append(f"- {mn}{unit_text}{syn_text}")

            excerpt = "\n".join(lines).strip()
            _VITALS_REGISTRY_EXCERPT_CACHE = excerpt
            return excerpt
        except Exception as e:
            if logger:
                logger.debug(f"Vitals registry excerpt: query failed: {e}")
            _VITALS_REGISTRY_EXCERPT_CACHE = ""
            return ""
        finally:
            if should_close and conn:
                try:
                    close_owned_pg_conn(conn)
                except Exception:
                    pass


def get_assertion_types(conn=None, logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """
    Load kb.assertion_types as a list of dicts.
    This is Phase 2's instruction manual for CONF/NEG/SUSP/HIST/HYPO/RECUR.
    """
    should_close = False
    if conn is None:
        try:
            conn = get_pg_conn(reuse=False)
            should_close = True
        except Exception as e:
            if logger:
                logger.warning(f"Could not connect to DB for assertion_types: {e}")
            return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT assertion_id, label, description, billing_impact
                FROM kb.assertion_types
                ORDER BY assertion_id;
            """)
            rows = cur.fetchall()
        return [
            {
                "assertion_id": r[0],
                "label": r[1],
                "description": r[2],
                "billing_impact": r[3],
            }
            for r in rows
        ]
    except Exception as e:
        if logger:
            logger.warning(f"Error loading kb.assertion_types: {e}")
        return []
    finally:
        if should_close and conn:
            try:
                close_owned_pg_conn(conn)
            except Exception:
                pass


def get_all_attributes_schema(conn=None, logger: Optional[logging.Logger] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load kb.attributes_schema into a dict: {source_kind: [relationship rows...]}.
    This is Phase 2's relationship grammar for attribute extraction.
    """
    should_close = False
    if conn is None:
        try:
            conn = get_pg_conn(reuse=False)
            should_close = True
        except Exception as e:
            if logger:
                logger.warning(f"Could not connect to DB for attributes_schema: {e}")
            return {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_kind, relationship, target_attribute, use_case, is_required
                FROM kb.attributes_schema
                ORDER BY source_kind, is_required DESC, relationship;
            """)
            rows = cur.fetchall()
        out: Dict[str, List[Dict[str, Any]]] = {}
        for source_kind, relationship, target_attribute, use_case, is_required in rows:
            out.setdefault(source_kind, []).append({
                "relationship": relationship,
                "target_attribute": target_attribute,
                "use_case": use_case,
                "is_required": is_required,
            })
        return out
    except Exception as e:
        if logger:
            logger.warning(f"Error loading kb.attributes_schema: {e}")
        return {}
    finally:
        if should_close and conn:
            try:
                close_owned_pg_conn(conn)
            except Exception:
                pass

# ==============================================================================
# MASTER TABLE QUERIES (Service Master & Inventory Master)
# ==============================================================================

def get_service_master_record(
    service_id: str,
    conn=None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """
    Query Service Master by service_id to get canonical name and metadata.
    
    This validates the ID exists and pulls the canonical procedure_name from the master.
    Used by Verification Dashboard to ensure correct routing and pre-populate fields.
    
    Args:
        service_id: Service ID from Phase 1 grounding
        conn: Database connection (uses get_pg_conn() if None)
        logger: Logger instance
        
    Returns:
        Dict with service_id, procedure_name, and other metadata, or None if not found
    """
    if not service_id:
        return None
    
    should_close = False
    if conn is None:
        try:
            conn = get_pg_conn(reuse=False)
            should_close = True
        except Exception as e:
            if logger:
                logger.warning(f"Could not connect to DB for service_master lookup: {e}")
            return None
    
    try:
        with conn.cursor() as cur:
            # Query service_master for canonical name and metadata
            cur.execute("""
                SELECT 
                    service_id,
                    procedure_name,
                    COALESCE(remarks, '') AS remarks,
                    COALESCE(category, '') AS category
                FROM soap.service_master
                WHERE service_id = %s
                LIMIT 1;
            """, (service_id,))
            row = cur.fetchone()
            
            if row:
                return {
                    "service_id": row[0],
                    "procedure_name": row[1],
                    "remarks": row[2],
                    "category": row[3],
                }
            return None
    except Exception as e:
        if logger:
            logger.warning(f"Error querying service_master for service_id={service_id}: {e}")
        return None
    finally:
        if should_close and conn:
            try:
                close_owned_pg_conn(conn)
            except Exception:
                pass


def lookup_service_id_by_name(
    name: str,
    conn=None,
    logger: Optional[logging.Logger] = None,
) -> Optional[int]:
    """
    Look up service_id from soap.service_master by procedure_name (ILIKE).
    Used to link known preventive/products (e.g. Bravecto) when they exist in the clinic.
    """
    if not (name or "").strip():
        return None
    should_close = False
    if conn is None:
        try:
            conn = get_pg_conn(reuse=False)
            should_close = True
        except Exception as e:
            if logger:
                logger.warning(f"Could not connect to DB for service lookup by name: {e}")
            return None
    try:
        with conn.cursor() as cur:
            # Prefer filtering by status when column exists; fallback to name-only (schema-safe)
            pattern = "%" + (name or "").strip() + "%"
            try:
                cur.execute("""
                    SELECT service_id FROM soap.service_master
                    WHERE procedure_name ILIKE %s AND (status IS NULL OR status != 'INACTIVE')
                    LIMIT 1;
                """, (pattern,))
            except Exception as col_err:
                if "status" in str(col_err) or "column" in str(col_err).lower():
                    cur.execute("""
                        SELECT service_id FROM soap.service_master
                        WHERE procedure_name ILIKE %s
                        LIMIT 1;
                    """, (pattern,))
                else:
                    raise
            row = cur.fetchone()
            return int(row[0]) if row else None
    except Exception as e:
        if logger:
            logger.debug("lookup_service_id_by_name failed: %s", e)
        return None
    finally:
        if should_close and conn:
            try:
                close_owned_pg_conn(conn)
            except Exception:
                pass


# Known preventive/flea-tick product names: link from inventory when present; do not label as "Generic vaccine" in diagnostics
_KNOWN_PREVENTIVE_PRODUCT_NAMES = frozenset(
    n.strip().lower() for n in os.environ.get(
        "KNOWN_PREVENTIVE_PRODUCTS",
        "bravecto,simparica,nexgard,seresto,frontline,advantage,revolution,credelio"
    ).split(",") if n.strip()
)


def get_inventory_master_record(
    stock_id: str,
    conn=None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """
    Query Inventory Master by stock_id to get canonical name, form, and metadata.
    
    This validates the ID exists and pulls:
    - Canonical item_name and trade_name
    - Form (Tablet, Liquid, Injection, etc.) - determines if we prompt for "Quantity" (tabs) or "Volume" (ml)
    - Other metadata needed for medication fulfillment
    
    Used by Verification Dashboard to ensure correct routing and pre-populate fields.
    
    Args:
        stock_id: Inventory ID (stock_id) from Phase 1 grounding
        conn: Database connection (uses get_pg_conn() if None)
        logger: Logger instance
        
    Returns:
        Dict with stock_id, item_name, trade_name, form, and other metadata, or None if not found
    """
    if not stock_id:
        return None
    
    should_close = False
    if conn is None:
        try:
            conn = get_pg_conn(reuse=False)
            should_close = True
        except Exception as e:
            if logger:
                logger.warning(f"Could not connect to DB for inventory lookup: {e}")
            return None
    
    try:
        with conn.cursor() as cur:
            # Query inventory for canonical name, form, and metadata
            # NOTE: soap.inventory schema varies across deployments; select only columns that exist
            # in our current DB (see information_schema.columns). In this DB:
            # - there's no `form` column
            # - `dosage_type` approximates form (tablet/capsule/liquid/etc.)
            # - `sales_uom`/`administered_uom` approximates unit (tabs/ml/etc.)
            # - `internal_description`/`brief_description` approximates remarks
            cur.execute("""
                SELECT 
                    stock_id,
                    item_name,
                    COALESCE(trade_name, '') AS trade_name,
                    COALESCE(dosage_type, '') AS form,
                    COALESCE(NULLIF(sales_uom, ''), NULLIF(administered_uom, ''), '') AS unit,
                    COALESCE(NULLIF(internal_description, ''), NULLIF(brief_description, ''), '') AS remarks,
                    COALESCE(batch_number, '') AS batch_number,
                    COALESCE(sales_uom, '') AS sales_uom,
                    COALESCE(administered_uom, '') AS administered_uom
                FROM soap.inventory
                WHERE stock_id = %s
                LIMIT 1;
            """, (stock_id,))
            row = cur.fetchone()
            
            if row:
                return {
                    "stock_id": row[0],
                    "item_name": row[1],
                    "trade_name": row[2],
                    "form": row[3],  # Tablet, Liquid, Injection, etc.
                    "unit": row[4],  # tabs, ml, etc.
                    "remarks": row[5],
                    "batch_number": row[6],
                    "sales_uom": row[7],
                    "administered_uom": row[8],
                }
            return None
    except Exception as e:
        if logger:
            logger.warning(f"Error querying inventory for stock_id={stock_id}: {e}")
        return None
    finally:
        if should_close and conn:
            try:
                close_owned_pg_conn(conn)
            except Exception:
                pass


def is_retail_or_diet_product(inventory_record: Optional[Dict[str, Any]]) -> bool:
    """
    Return True if the inventory record is Diet & Nutrition or Non-Medical (retail).
    Used to route items to "Other Products & Retail" instead of Pharmacy (Prescribed/Administered).
    """
    if not inventory_record or not isinstance(inventory_record, dict):
        return False
    text = " ".join([
        str(inventory_record.get("item_name") or ""),
        str(inventory_record.get("trade_name") or ""),
        str(inventory_record.get("remarks") or ""),
    ]).lower()
    # Diet & Nutrition: weight management, prescription diets, satiety, metabolic
    # Non-Medical: behavior (chew toys, puzzles), hygiene (shampoo, brushes)
    retail_keywords = (
        "diet", "satiety", "metabolic", "weight management", "weight loss", "obesity",
        "nutrition", "prescription diet", "rc ", "hills ", "royal canin",
        "chew", "toy", "puzzle", "behavior",
        "shampoo", "brush", "hygiene", "grooming",
        "retail", "non-medical",
    )
    return any(kw in text for kw in retail_keywords)


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# --- Configuration ---
# Set the model provider and model name:
# MODEL_PROVIDER: 'openai', 'claude', 'mistral', or 'fireworks'
# MODEL_NAME: the model string for the selected provider (default from PHASE2_MODEL env for 60s target)
MODEL_PROVIDER = os.getenv("PHASE2_MODEL_PROVIDER", "openai").strip().lower() or "openai"
MODEL_NAME = os.getenv("PHASE2_MODEL", "gpt-4.1-mini").strip()

# Model recommendations by speed (fastest to slowest):
# OpenAI: "gpt-4o-mini" (fastest), "gpt-4o", "gpt-4-turbo"
# Claude: "claude-3-haiku-20240307" (fastest), "claude-3-sonnet-20240229", "claude-3-opus-20240229"
# Mistral: "mistral-small-latest" (fastest), "mistral-medium-latest", "mistral-large-latest"
# Fireworks: "accounts/fireworks/models/gpt-oss-120b" (120B model), "accounts/fireworks/models/llama-v3p1-8b-instruct" (fastest), "accounts/fireworks/models/llama-v3p3-8b-instruct", "accounts/fireworks/models/llama-v3p3-70b-instruct"

# API Key file configuration
FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
API_KEY_FILE = os.path.join(FOLDER_PATH, "API_Key.txt")
FIREWORKS_API_FILE = os.path.join(FOLDER_PATH, "fireworks_api.txt")


# ==============================================================================
# FILE PATHS
# ==============================================================================
# --- Configuration ---
# Set the full paths to your input and output files here.
# Use 'r' before the string to handle backslashes in Windows paths, e.g., r"C:\Users\..."

# New: Path to the SOAP notes directory (dynamic selection)
SOAP_NOTES_DIR = r"/Users/vivek/VETINSTANT/wip/New folder/P.A.W.S/SOAP notes - voice to text/OP/soap_note_experiment/output/soap_notes"

# Required: Path to the directory where output files will be saved.
OUTPUT_DIR = r"/Users/vivek/VETINSTANT/wip/New folder/P.A.W.S/SOAP notes - voice to text/OP/soap_note_experiment/output/knowledge_atoms"


# ==============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# ==============================================================================

# Performance optimizations implemented:
# 1. Reduced max_tokens from 4000 to 1800 (most reports are 800-1500 tokens)
# 2. Reduced request timeout from 120s to 60s
# 3. Reduced max retries from 3 to 2
# 4. Configurable model selection for easy switching
# Target: Reduce generation time from 10-11s to 3-5s

OPTIMIZED_CONFIG = {
    "max_tokens": 4000,        # Keep original for complete responses
    "temperature": 0.1,        # Keep low for consistency
    "request_timeout": 45,     # Reduced from 60s for faster failure detection
    "max_retries": 1,          # Reduced from 2 for faster failure handling
}


# ==============================================================================
# CONFIGURATION AND CONSTANTS
# ==============================================================================

class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    MISTRAL = "mistral"
    FIREWORKS = "fireworks"


def get_model_provider_enum(provider_string: str) -> ModelProvider:
    """Convert string model provider to enum."""
    provider_map = {
        'openai': ModelProvider.OPENAI,
        'claude': ModelProvider.CLAUDE,
        'mistral': ModelProvider.MISTRAL,
        'fireworks': ModelProvider.FIREWORKS
    }
    provider_lower = provider_string.lower()
    if provider_lower not in provider_map:
        raise ValueError(f"Unsupported model provider: {provider_string}. Supported: {list(provider_map.keys())}")
    return provider_map[provider_lower]


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


@dataclass
class Config:
    """Configuration settings for Phase 2 knowledge atom extraction."""
    # Required fields
    input_soap_note_path: str
    output_dir: str
    api_key_file: str
    model_provider: ModelProvider
    model_name: str
    
    # Optional fields
    request_timeout: int = OPTIMIZED_CONFIG["request_timeout"]
    max_retries: int = OPTIMIZED_CONFIG["max_retries"]

# Default configuration - uses configurable MODEL_PROVIDER and MODEL_NAME variables
DEFAULT_CONFIG = Config(
    # Required fields
    input_soap_note_path=SOAP_NOTES_DIR,
    output_dir=OUTPUT_DIR,
    api_key_file=API_KEY_FILE,
    model_provider=get_model_provider_enum(MODEL_PROVIDER),
    model_name=MODEL_NAME,
)


# ==============================================================================
# KNOWLEDGE ATOM EXTRACTION PROMPT TEMPLATE
# ==============================================================================

def build_knowledge_atom_prompt(assertion_types: List[Dict], attributes_schema: Dict[str, List[Dict]]) -> str:
    """
    Build the Knowledge Atom extraction prompt with KB schema context.
    
    Args:
        assertion_types: List of assertion type definitions from kb.assertion_types
        attributes_schema: Dictionary mapping source_kind to list of attribute relationships
        
    Returns:
        Complete prompt string for Knowledge Atom extraction
    """
    
    # Format assertion types for prompt
    assertion_types_text = "\n".join([
        f"- {at['assertion_id']}: {at['label']} - {at['description']} ({at['billing_impact']})"
        for at in assertion_types
    ])
    
    # Format attributes schema for prompt
    attributes_schema_text = ""
    for kind, attrs in sorted(attributes_schema.items()):
        attributes_schema_text += f"\n{kind}:\n"
        for attr in attrs:
            required_marker = " [REQUIRED]" if attr.get('is_required') else ""
            attributes_schema_text += f"  - {attr['relationship']} → {attr['target_attribute']}{required_marker}\n"
            if attr.get('use_case'):
                attributes_schema_text += f"    Example: {attr['use_case']}\n"
    
    # DB-backed vitals anchor (compact) — safe to be empty if DB is unavailable.
    vitals_anchor = ""
    try:
        vitals_anchor = get_vitals_registry_excerpt(conn=None, logger=None) or ""
    except Exception:
        vitals_anchor = ""

    vitals_anchor_block = ""
    if vitals_anchor:
        vitals_anchor_block = f"""
**DB-backed Vitals Taxonomy Anchor (from kb.vitals_registry):**
If a mention matches any metric name OR synonym below, you MUST set kind="VitalSign"
and extract the appropriate value/unit/qualitative_flag.

{vitals_anchor}
"""

    prompt = f"""
Phase 2: Knowledge Atom Extraction - Constraint-Based Logic Engine
Role: You are a Clinical Intelligence System that extracts structured Knowledge Atoms from veterinary SOAP notes for the Verification Dashboard (billing and clinical order list).

Your job is to transform natural language SOAP notes into structured clinical intents (Knowledge Atoms) that drive billing accuracy and compliance. Phase 2 is a TRANSACTIONAL layer, not a summary layer.

================================================================================
STRICT SCOPE GATE (Actionable Only - Verification Dashboard)
================================================================================
ONLY extract entities that represent:
- Billable services (Procedures, Services)
- Prescribed or administered products (Medicines, Drugs, Vaccines, Consumables)
- Diagnostic tests / labs (Diagnostics, LabTest, imaging)
- Vitals (measured values with unit)
- Scheduled follow-ups / reminders (Reminders, Follow-ups)
- Reason for Visit (chief complaint - anchor for the visit)

Do NOT extract for this module:
- Clinical findings or narrative observations (e.g. "walking funny", "whining", "lazy") — these stay in the SOAP text only.
- Suspected conditions or differentials as standalone narrative (unless tied to a diagnostic order or procedure).
- Signalment (breed, age, sex, weight) — already in Header.
- Patient or owner identity (names) — already in Header.

EXPLICIT FILTERING: Do not extract generic verbs, common adjectives (e.g. lazy, fine, okay), or conversational filler. Noisy atoms are filtered downstream. Restrict output to actionable kinds above to reduce dashboard clutter and latency.

================================================================================
SOAP SECTION AWARENESS (Plan vs Vitals vs Reminders)
================================================================================
**PLAN SECTION**: Recognize and extract items that appear in the **Plan** section of the SOAP note. Medications, procedures, diagnostics, diet, supplements, and treatments listed in the Plan are primary candidates for knowledge atoms. Set section="Plan" for atoms extracted from the Plan. The downstream constraint layer uses Plan presence to align atoms with what the vet documented.

**VITALS**: Vitals are NOT in the Plan section. They appear in Vitals, Objective, or signalment. Extract vitals from those sections and set section="Vitals" or "Objective" accordingly. Do not require Plan for vitals — they are considered separately.

**REMINDERS AND FOLLOW-UPS**: These appear in Reminders, CustomerInstructions, or elsewhere in the SOAP note (e.g. "Recheck in 2 weeks", "Schedule X-ray"). Extract them and set section="Reminders" or the section where they appear. They are considered separately from Plan items.

CRITICAL: You must use the KB Schema tables provided below as your "Instruction Manual" for extraction.

This is a CONSTRAINT-BASED LOGIC ENGINE, not a simple keyword search. By forcing you to look for specific attributes for every Kind, you are anchored more deeply in the context, making it much harder to "forget" or ignore mangled terms.

================================================================================
ASSERTION TYPES (Clinical Status)
================================================================================
Every clinical mention must be tagged with one of these assertion types:

{assertion_types_text}

Usage Rules:
- CONF: Use when action is happening or planned now (mark active)
- NEG: Use when explicitly negated ("No vomiting", "Declined vaccine") (mark inactive)
- SUSP: Use for rule-out or differential diagnoses (diagnostics focus)
- HIST: Use for past history ("Was on Prednisone last month") (mark inactive)
- HYPO: Use for hypothetical/future options discussed (mark inactive)
- RECUR: Use for ongoing maintenance or chronic protocols (recurring care)

================================================================================
ATTRIBUTES SCHEMA (Relationship Grammar)
================================================================================
Each entity kind has specific attributes that MUST be extracted when present:

{attributes_schema_text}

CRITICAL RELATIONSHIP RULES:
1. Medicine MUST have: dose, route (both required)
2. Medicine MAY have: frequency, duration, refills, batch_id
3. Procedure MUST have: at_site (body_part) if location is mentioned
4. Diagnostic MUST have: specimen_type if specimen is mentioned
5. Vitals MUST have: value, unit (both required)
6. Reminder MUST have: due_on, action_item (both required)

================================================================================
EXTRACTION PROCESS - REASON FOR VISIT IS THE MASTER ANCHOR
================================================================================

Step 0: EXTRACT REASON FOR VISIT FIRST (HIGHEST PRIORITY)
The Reason for Visit (RFV) is the "Master Anchor" that defines the Clinical Gravity of the session.
Extract it BEFORE all other entities - it creates Semantic Bias that helps identify related procedures, drugs, and findings.

Look for phrases: "came for", "presented for", "brought in for", "chief complaint is", "reason for visit"

For RFV entities, you MUST populate these attributes:
- chief_complaint (MANDATORY - cannot be null): The primary issue (e.g., "scoots", "vomiting", chief complaint as stated)
  * Preserve original text even if nonsensical - this anchors the visit
- urgency (REQUIRED): Routine, Urgent, or Emergency (infer from context if not explicit)
- duration: How long the issue has been present (if mentioned)
- previous_history: Whether new or recurring issue (if mentioned)

The RFV creates Semantic Bias: If RFV is "Scooting", expect rear-end procedures. If RFV is "Skin itching", 
prioritize Anatomy (Skin, Ears) and Drugs (Apoquel, Cytopoint).

Step 1: Identify Actionable Entities Only (After RFV)
Extract ONLY entities that impact billing or clinical orders (see STRICT SCOPE GATE above):
- Reason for Visit (extracted in Step 0 - acts as anchor)
- Medicines (administered or prescribed)
- Procedures (performed or recommended) - **CRITICAL: Use kind="Procedure" for treatments/therapies**
- Diagnostics (X-rays, lab tests, imaging) - **CRITICAL: Use kind="Diagnostic" or "LabTest" for X-rays/labs**
- Vitals (measured - requires numerical extraction)
- Preventive care (vaccinations, deworming, flea/tick treatment)
- Consumables (used during visit)
- Reminders/Follow-ups (scheduled)

Do NOT extract for this module: Findings (e.g. "walking funny", "whining"), Conditions as narrative only, Signalment, or Identity. These remain in the SOAP text and do not appear in the Verification Dashboard.

**CRITICAL KIND CLASSIFICATION RULES:**

1. **Diagnostics MUST use kind="Diagnostic" or "LabTest"**:
   - X-rays, radiographs, imaging studies → kind="Diagnostic" (NOT "Procedure")
   - Lab tests, blood work, urinalysis → kind="LabTest" or "Diagnostic"
   - Ultrasound, CT, MRI → kind="Diagnostic"
   - Examples: "X-ray" → kind="Diagnostic", "CBC" → kind="LabTest", "Norberg angle measurement" → kind="Diagnostic"

2. **Procedures/Services use kind="Procedure"**:
   - Treatments/therapies → kind="Procedure"
   - Examples: "Physiotherapy" → kind="Procedure", "Hydrotherapy" → kind="Procedure", "Surgery" → kind="Procedure"
   - **DO NOT classify X-rays/labs as Procedures** - they are Diagnostics

Step 2: Assign Assertion Status (Clinical Status)
For each entity, determine its clinical status using the assertion types:
- If explicitly mentioned as happening: CONF (mark active)
- If explicitly negated ("No", "Declined", "Not given"): NEG (mark inactive)
- If mentioned as past history ("Has been on X", "History of Y"): HIST (mark inactive, preserve in medical record)
- If mentioned as future option: HYPO (mark inactive)
- If mentioned as ongoing/recurring: RECUR (recurring care)
- If mentioned as suspected/rule-out/differential/considered/possible: SUSP (diagnostics focus)

SUSP TRIGGERS (CRITICAL):
- Words/phrases like: "possible", "consider", "considered", "rule out", "r/o", "suspect", "suspicious for",
  "differential", "ddx", "cannot exclude"
- Example: "Possible patellar luxation considered but not confirmed" → extract a Condition/Diagnosis atom with assertion_id="SUSP"

CRITICAL: assertion_id is the primary clinical status tag:
- CONF, SUSP, RECUR → Active clinical intent
- NEG, HIST, HYPO → Not active for current plan

Past Medical History (PMH) Handling:
- When doctor says "Patient has a history of heart murmurs", extract Condition: "Heart Murmur" with assertion_id: "HIST"
- This preserves history while marking it inactive for the current plan

Step 3: Extract Attributes (Constraint-Based - Structural Guardrails)
For each entity, you MUST attempt to extract attributes according to the attributes_schema.
Attributes are STRUCTURAL GUARDRAILS - they force you to look deeper in the context, improving identification accuracy.

MANDATORY ATTRIBUTE MAPPING BY KIND:

Drug / Medicine:
- Identifying: brand_name, generic_name, strength
- Action: dose (REQUIRED if mentioned), route (REQUIRED if mentioned), frequency, duration, quantity, intent_type

Procedure:
- Identifying: procedure_name, technique
- Action: site (Anatomy - REQUIRED if location mentioned), status (Performed/Planned), intent_type

Finding / Exam:
- Descriptive: location (Anatomy - REQUIRED), symmetry (L/R/Bilateral)
- Contextual: observation (REQUIRED), measurement (for Vitals), severity

Condition / Disease:
- Descriptive: severity, chronicity (Acute/Chronic)
- Contextual: status (Confirmed/Suspected), onset_date, body_system

Reason for Visit (RFV):
- chief_complaint (MANDATORY - cannot be null)
- urgency (REQUIRED: Routine/Urgent/Emergency)
- duration, previous_history

LabTest / Diagnostic:
- Identifying: test_name, panel_components
- Action: sample_type (REQUIRED if mentioned), status (Performed/Ordered - CRITICAL), priority (Stat/Routine), site (for imaging)

VitalSign / Vitals (Numerical Extraction - High Precision Required):
- Identifying: metric_name (Temp, Weight, HR, RR, CRT, BCS, etc.)
- Numerical: value (REQUIRED - numeric result, e.g., 38.5, 22.4)
- Units: unit (REQUIRED - kg, lbs, Celsius, Fahrenheit, bpm, etc.) - Prevents dangerous unit conversion errors
- Qualitative: qualitative_flag (Normal, High, Low, "WNL") - Anchors Assessment section
- Contextual: measurement_type (Rectal, Axillary, etc.), reference_range (if mentioned)

**VETERINARY VITALS CHECKLIST (MUST be kind="VitalSign", not Finding/Observation):**
- Core TPR: Temperature, Pulse/Heart Rate, Respiration Rate
- Perfusion: CRT/Capillary Refill Time, Mucous Membrane Color/Moisture, Pulse Quality
- Physical condition: Weight, BCS (Body Condition Score), MCS (Muscle Condition Score)
- Auscultation status: Heart auscultation (murmur/arrhythmia/normal), Lung auscultation (crackles/wheezes/normal)
- Neuro/pain: Mentation (BAR/QAR/Dull/Obtunded), Pain Score
- Advanced/triage/anesthesia: Blood Pressure (systolic/diastolic/MAP), SpO2/Oxygen saturation, ETCO2, Blood Glucose

**Mapping rule**: If the SOAP note says "mucous membranes normal/pink/moist", "CRT 1.5 sec", "BAR", "pain 2/4",
"heart sounds normal/no murmur", or "lungs clear", extract as VitalSign with:
- metric_name: one of the checklist names (e.g., "Mucous Membranes", "CRT", "Mentation", "Pain Score")
- value: the measurement or categorical value (e.g., "1.5", "BAR", "2/4", "Normal")
- unit: include when applicable (sec, bpm, kg, %, mmHg, mg/dL)

{vitals_anchor_block}

DiagnosticTest / Radiology (Imaging - Highly Structured):
- Identifying: test_name, modality (X-Ray, Ultrasound, CT, MRI - REQUIRED)
- Anatomical: site (REQUIRED - Thorax, Abdomen, Left Stifle, etc.) - Essential for legal medical record compliance
- Technical: views (REQUIRED - 2-view, 3-view, LAT/VD, etc.)
- Clinical: interpretation (Narrative finding, e.g., "Enlarged heart") - Anchors Assessment section and links to Finding atoms

CRITICAL ATTRIBUTE RULE:
- IDENTIFICATION IS MANDATORY: You must identify the concept (e.g., "Meloxicam" as a Drug) even if attributes are missing
- ATTRIBUTES ARE OPPORTUNISTIC: Look for attributes in surrounding context, but do NOT fail to extract the entity if they're missing
- Create Partial Atoms: If a doctor says "I gave Meloxicam" without dose, extract attributes_kv with null values, e.g.:
  {{"kind": "Drug", "concept": "Meloxicam", "attributes_kv": [{{"relationship":"dose","value":null}}, {{"relationship":"route","value":null}}]}}
- Downstream systems handle missing attributes via: (1) Direct extraction from context, (2) Clinic defaults, (3) Draft state requiring confirmation

WHY THIS PREVENTS FORGOTTEN ACTIONS:
By forcing you to look for dose, route, and frequency for every drug, you are effectively told:
"This is a medication, not a random word. Look at the surrounding words for administration details."
This significantly improves identification accuracy, even if the attributes themselves end up being null.

Step 4: Determine Intent Context (Fulfillment Status) - CRITICAL FOR PHASE 3
Classify the intent_context to determine Fulfillment Requirements:

**STRICT INTENT CONTEXT TAXONOMY (MANDATORY):**

**CRITICAL CLASSIFICATION RULES:**

1. **Procedures, Services, and Treatments** → **MUST use intent_context="Performed"**
   - These are clinical services/treatments done DURING the clinic visit
   - Examples: "Physiotherapy", "Hydrotherapy", "Swimming", "Sit-stand-step exercise", "Wound dressing", "Surgery"
   - **NEVER use "Prescribed" for Procedures/Services** - you "Prescribe" medicine, you "Perform" services
   - If mentioned in Plan section as future action → use intent_context="Scheduled" (goes to reminders)

2. **Diagnostics (X-rays, Lab Tests, Imaging)** → **MUST use kind="Diagnostic" or "LabTest"**
   - **CRITICAL**: X-rays, radiographs, imaging studies → kind="Diagnostic" (NOT "Procedure")
   - Lab tests, blood work, urinalysis → kind="LabTest" or "Diagnostic"
   - Ultrasound, CT, MRI → kind="Diagnostic"
   - Intent context depends on status:
     - If done during visit → intent_context="Performed"
     - If ordered for future → intent_context="Ordered"

3. **Medications (Medicine, Drug, Supplement, Vaccine)** → **MUST use intent_context="Prescribed" OR "Administered"**
   - **"Prescribed"**: Medicine ordered for owner to give at home
     - Examples: "Prescribed Spirocoxin", "Prescribed ContraVale Forte", "Give Omega 3/6 at home"
   - **"Administered"**: Medicine given DURING the clinic visit
     - Examples: "Gave 5mg Meloxicam", "Administered vaccine", "Injected today"
   - **NEVER use "Performed" for Medications** - medications are "Prescribed" or "Administered", not "Performed"

**INTENT CONTEXT DEFINITIONS:**

1. **"Performed"**: 
   - **MANDATORY for Procedures/Services/Treatments** done during visit
   - **ALSO for Diagnostics** done during visit
   - Examples: "Physiotherapy performed", "X-ray performed", "Surgery completed"
   - Physical verification via QR code scan

2. **"Prescribed"**: 
   - **RESERVED EXCLUSIVELY for Medicines, Drugs, Supplements, Nutrition**
   - Medicine ordered for owner to administer/complete at home
   - Examples: "Prescribed Amoxicillin for 7 days", "Prescribed Spirocoxin", "Prescribed Obesity Diet"
   - **NEVER use for Procedures** - Procedures are "Performed", not "Prescribed"

3. **"Administered"**: 
   - **RESERVED EXCLUSIVELY for Medicines, Drugs, Supplements, Vaccines**
   - Medicine/Vaccine given DURING the clinic visit
   - Examples: "Gave 5mg Meloxicam", "Administered vaccine", "Injected today"
   - Physical verification via QR code scan

4. **"Ordered"**: 
   - **For Diagnostics** ordered but not yet completed
   - Examples: "Ordered CBC for next week", "Ordered X-ray for follow-up"
   - Creates requisition/worklist entry

5. **"Scheduled"**: 
   - Reminder/follow-up scheduled for future
   - Examples: "Recheck in 2 weeks", "Vaccination due next month"
   - Creates reminder entry

6. **"Measured"**: 
   - Vital sign taken during visit
   - Examples: "Temperature 102.5°F", "Heart rate 140 bpm"
   - Clinical record entry

7. **"Declined"**: 
   - Explicitly NOT done (assertion_id should be NEG)
   - Examples: "Owner declined vaccine", "Not given today"
   - None - skip entirely

**CRITICAL RULE**: 
- Procedures/Services → intent_context="Performed" (NEVER "Prescribed")
- Medications → intent_context="Prescribed" OR "Administered" (NEVER "Performed")
- Diagnostics → kind="Diagnostic" AND intent_context="Performed" OR "Ordered" 

**SPECIALIZED FULFILLMENT LOGIC:**

1. **Vitals (Kind: VitalSign)**
   - Updates clinical chart directly
   - Updates clinical chart directly with value, unit, and qualitative_flag
   - Example: "Temperature 102.5°F"

2. **Lab Tests (Kind: LabTest)**
   - **Status: "Ordered"** → Downstream systems handle requisitions
   - **Status: "Performed"** → Downstream systems handle results
   - Example: "I've ordered a CBC" → Requisition created
   - Example: "CBC showed anemia" → Assumes Performed, creates downstream line + Result Atom

3. **Radiology (Kind: DiagnosticTest)**
   - Requires: modality, site, views (all REQUIRED)
   - Downstream systems handle scheduling
   - The "Interpretation" attribute links to Finding atoms (e.g., "Enlarged heart" finding anchored to "Thoracic Radiograph")
   - Example: "2-view X-ray of the thorax" → Imaging worklist entry + downstream line with correct Variant ID

**ANCHORING FORGOTTEN DIAGNOSTICS:**
By requiring these attributes, you are forced to capture Action Items even when mentioned in passing:
- Trigger: "Let's do some bloods and an x-ray of that leg."
- Schema Requirement: LabTest requires Status (Ordered), DiagnosticTest requires Site (Leg)
- Extraction: You must look for "Leg" mention and "Status"
- Automation: capture intent_type and required attributes accurately

CRITICAL ACCURACY RULES:
0. ONE CONCEPT PER CLINICAL ACTION (Defense-in-depth): Each extracted concept (e.g., each Procedure or Diagnostic) must represent one distinct clinical action; do not group unrelated plan items under a single concept name. This ensures the highest granularity for downstream enrichment and verification.

1. RELATIONSHIP LOCKING: When multiple attributes are mentioned together, lock them to the correct entity.
   Example: "Administered 5mg Meloxicam and 10ml Saline"
   - Create TWO separate atoms:
     - Atom 1: concept="Meloxicam", attributes={{"dose": "5mg", "route": "PO"}}
     - Atom 2: concept="Saline", attributes={{"dose": "10ml", "route": "IV"}}
   - DO NOT mix: "5mg" belongs to Meloxicam, "10ml" belongs to Saline

2. NEGATION DETECTION: Always check for negations.
   Example: "Owner declined Rabies vaccine"
   - assertion_id: "NEG"
   - This marks the atom as negated

3. ATTRIBUTE VALIDATION: Only extract attributes that are valid for the entity kind.
   - Medicine cannot have "specimen_type"
   - Diagnostic cannot have "dose" or "route"
   - Use the attributes_schema to validate

4. REQUIRED ATTRIBUTES: If a required attribute is missing, mark it as null but still create the atom.
   - Medicine without dose: Still create atom, but dose=null
   - Vitals without unit: Still create atom, but unit=null

================================================================================
OUTPUT FORMAT
================================================================================

Return JSON ONLY in this exact format:

{{
  "knowledge_atoms": [
    {{
      "concept": "Anal Gland Expression",
      "kind": "Reason",
      "assertion_id": "CONF",
      "attributes_kv": [
        {{"relationship":"chief_complaint","value":"anal gland expression"}},
        {{"relationship":"urgency","value":"Routine"}},
        {{"relationship":"duration","value":null}},
        {{"relationship":"previous_history","value":null}}
      ],
      "intent_context": "Presented",
      "source_text": "The animal came for anal gland expression",
      "section": "Subjective"
    {{
      "concept": "Meloxicam",
      "kind": "Medicine",
      "assertion_id": "CONF",
      "attributes_kv": [
        {{"relationship":"dose","value":"5mg"}},
        {{"relationship":"route","value":"PO"}},
        {{"relationship":"frequency","value":"Once daily"}}
      ],
      "intent_context": "Administered",
      "source_text": "Administered 5mg Meloxicam PO once daily",
      "section": "Plan"
    }},
    {{
      "concept": "Rabies Vaccine",
      "kind": "Vaccine",
      "assertion_id": "NEG",
      "attributes_kv": [],
      "intent_context": "Declined",
      "source_text": "Owner declined Rabies vaccine",
      "section": "Plan"
    }},
    {{
      "concept": "CBC",
      "kind": "Diagnostic",
      "assertion_id": "CONF",
      "attributes_kv": [
        {{"relationship":"specimen_type","value":"Blood"}},
        {{"relationship":"priority","value":"Routine"}}
      ],
      "intent_context": "Ordered",
      "source_text": "Ordered CBC with blood sample",
      "section": "Plan"
    }},
    {{
      "concept": "Temperature",
      "kind": "VitalSign",
      "assertion_id": "CONF",
      "attributes_kv": [
        {{"relationship":"metric_name","value":"Temperature"}},
        {{"relationship":"value","value":"102.5"}},
        {{"relationship":"unit","value":"Fahrenheit"}},
        {{"relationship":"qualitative_flag","value":"High"}},
        {{"relationship":"measurement_type","value":"Rectal"}},
        {{"relationship":"reference_range","value":null}}
      ],
      "intent_context": "Measured",
      "source_text": "Temperature 102.5°F (rectal)",
      "section": "Objective"
    }},
    {{
      "concept": "CBC",
      "kind": "LabTest",
      "assertion_id": "CONF",
      "attributes_kv": [
        {{"relationship":"test_name","value":"CBC"}},
        {{"relationship":"sample_type","value":"Whole blood"}},
        {{"relationship":"status","value":"Ordered"}},
        {{"relationship":"priority","value":"Routine"}}
      ],
      "intent_context": "Ordered",
      "source_text": "I've ordered a CBC with whole blood sample",
      "section": "Plan"
    }},
    {{
      "concept": "Thoracic Radiograph",
      "kind": "DiagnosticTest",
      "assertion_id": "CONF",
      "attributes_kv": [
        {{"relationship":"modality","value":"X-Ray"}},
        {{"relationship":"site","value":"Thorax"}},
        {{"relationship":"views","value":"2-view"}},
        {{"relationship":"interpretation","value":"Enlarged heart, clear lungs"}}
      ],
      "intent_context": "Performed",
      "source_text": "2-view X-ray of the thorax showed enlarged heart",
      "section": "Objective"
    }},
    {{
      "concept": "Follow-up Recheck",
      "kind": "Reminder",
      "assertion_id": "CONF",
      "attributes_kv": [
        {{"relationship":"due_on","value":"In 2 weeks"}},
        {{"relationship":"action_item","value":"Suture removal"}}
      ],
      "intent_context": "Scheduled",
      "source_text": "Recheck in 2 weeks for suture removal",
      "section": "Plan"
    }}
  ],
  "extraction_summary": {{
    "total_atoms": 9,
    "confirmed": 8,
    "negated": 1,
    "by_kind": {{
      "Reason": 1,
      "Medicine": 1,
      "Vaccine": 1,
      "LabTest": 1,
      "DiagnosticTest": 1,
      "VitalSign": 1,
      "Reminder": 1
    }}
  }}
}}

IMPORTANT OUTPUT RULES:
1. Every atom MUST have: concept, kind, assertion_id, attributes_kv (can be empty list), intent_context, source_text, section
2. assertion_id MUST be one of: CONF, NEG, SUSP, HIST, HYPO, RECUR
   - assertion_id is the primary clinical status tag
3. attributes_kv MUST only contain relationships defined in attributes_schema for that kind
4. If an attribute is mentioned but not in schema, DO NOT include it
5. If a required attribute is missing, include it with value=null - create Partial Atom, don't drop entity
6. source_text should be the exact phrase from SOAP note that led to this atom
7. section should be: Subjective, Objective, Assessment, or Plan
8. REASON FOR VISIT: Must be extracted FIRST, chief_complaint is MANDATORY (cannot be null)

**CRITICAL KIND AND INTENT_CONTEXT VALIDATION:**

1. **Procedures/Services**:
   - kind MUST be: "Procedure", "Service", or "Treatment"
   - intent_context MUST be: "Performed" (if done during visit) OR "Scheduled" (if future)
   - **NEVER use intent_context="Prescribed" for Procedures** - Procedures are "Performed", not "Prescribed"
   - Examples: "Physiotherapy" → kind="Procedure", intent_context="Performed"
   - Examples: "Hydrotherapy" → kind="Procedure", intent_context="Performed"

2. **Diagnostics**:
   - kind MUST be: "Diagnostic", "LabTest", or "DiagnosticTest"
   - **NEVER use kind="Procedure" for X-rays, labs, or imaging**
   - intent_context MUST be: "Performed" (if done) OR "Ordered" (if future)
   - Examples: "X-ray" → kind="Diagnostic", intent_context="Ordered" or "Performed"
   - Examples: "CBC" → kind="LabTest", intent_context="Ordered" or "Performed"

3. **Medications**:
   - kind MUST be: "Medicine", "Drug", "Supplement", "Vaccine", or "Nutrition"
   - intent_context MUST be: "Prescribed" (for home) OR "Administered" (at clinic)
   - **NEVER use intent_context="Performed" for Medications** - Medications are "Prescribed" or "Administered"
   - Examples: "Spirocoxin" → kind="Medicine", intent_context="Prescribed"
   - Examples: "ContraVale Forte" → kind="Medicine", intent_context="Prescribed"

CRITICAL BILLING RULE:
- DO NOT guess prices. Focus on clinical intent extraction only.
- assertion_id determines clinical status
- HIST atoms are preserved in medical record but not billed (past medical history)

SPECIALIZED DATA FLOW:
- **Vitals (VitalSign)**: Extracted as Findings → Updated in Patient Charts (NOT billed)
- **Lab Tests (LabTest)**: Extracted → Created as Requisitions (Ordered) or Charges (Performed)
- **Radiology (DiagnosticTest)**: Extracted → Scheduled in Imaging Worklist AND Billed

NUMERICAL & RELATIONAL EXTRACTION REQUIREMENTS:
- **Vitals**: MUST extract value (numeric) and unit (REQUIRED) - prevents dangerous unit conversion errors
- **Lab Tests**: MUST extract status (Ordered vs. Performed) - determines Requisition vs. Charge
- **Radiology**: MUST extract modality, site, and views (all REQUIRED) - affects Variant ID and legal compliance

By defining these specialized attributes, diagnostics are not just notes in the text, but are deterministic triggers for downstream workflows.

================================================================================
INPUT
================================================================================
"""
    return prompt


# ==============================================================================
# KB + INVENTORY BRIDGING HELPERS (CODE -> VECTOR -> NUDGE)
# ==============================================================================

def _to_pgvector_literal(vec: Any) -> Optional[str]:
    """
    Convert embedding to pgvector literal text. Accepts:
    - list[float]
    - string like "[0.1,0.2,...]"
    """
    if vec is None:
        return None
    if isinstance(vec, str):
        s = vec.strip()
        if s.startswith("[") and s.endswith("]"):
            return s
        return None
    if isinstance(vec, (list, tuple)) and vec:
        try:
            return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"
        except Exception:
            return None
    return None


def _parse_pgvector_to_list(vec: Any) -> Optional[List[float]]:
    """
    Parse pgvector value into list[float]. Accepts:
    - list/tuple already
    - string like "[1,2,3]"
    """
    if vec is None:
        return None
    if isinstance(vec, (list, tuple)):
        try:
            return [float(x) for x in vec]
        except Exception:
            return None
    if isinstance(vec, str):
        s = vec.strip()
        if not (s.startswith("[") and s.endswith("]")):
            return None
        inner = s[1:-1].strip()
        if not inner:
            return []
        try:
            return [float(x) for x in inner.split(",")]
        except Exception:
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", inner)
            try:
                return [float(x) for x in nums]
            except Exception:
                return None
    return None


def fetch_kb_concept_embedding(
    conn,
    concept_id: Optional[int] = None,
    preferred_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[List[float]]:
    """
    Fetch the canonical 1536-dim KB concept embedding from `kb.concepts.embedding`.
    When LOCAL_ONLY is true (default), returns None without querying kb.concepts.
    """
    if os.getenv("LOCAL_ONLY", "true").lower() in ("1", "true", "yes"):
        return None
    if not conn:
        return None

    try:
        with conn.cursor() as cur:
            if concept_id is not None:
                cur.execute(
                    "SELECT embedding FROM kb.concepts WHERE concept_id = %s AND embedding IS NOT NULL LIMIT 1;",
                    (concept_id,),
                )
            elif preferred_name:
                cur.execute(
                    "SELECT embedding FROM kb.concepts WHERE lower(preferred_name) = lower(%s) AND embedding IS NOT NULL LIMIT 1;",
                    (preferred_name,),
                )
            else:
                return None

            row = cur.fetchone()
            if not row:
                return None
            return _parse_pgvector_to_list(row[0])
    except Exception as e:
        if logger:
            logger.debug(f"Could not fetch KB embedding: {e}")
        return None


# ==============================================================================
# PHASE 2 INTEGRATION HELPERS (minimal stubs for kb_phase2_integration.py)
# ==============================================================================

def build_phase2_prompt_with_grounding(
    base_prompt: str,
    session_id: Optional[str] = None,
    section_name: str = "FullNote",
    entity_manifest_json: Optional[str] = None,
) -> str:
    """
    Add entity manifest context to the base prompt for grounding.
    Minimal implementation: just appends manifest JSON if provided.
    """
    if not entity_manifest_json or entity_manifest_json.strip() == "[]":
        return base_prompt
    
    manifest_block = f"""
[ENTITY_MANIFEST_CONTEXT]
The following entities were extracted and linked from the transcript:
{entity_manifest_json}

Grounding-aware extraction rules (MANDATORY):
1) Treat manifest IDs as source-of-truth for identity persistence.
2) If an extracted atom corresponds to a manifest entity, include:
   - referenced_entity_id: manifest entity_id
3) For the same clinical action phrased differently, emit a stable:
   - dedup_key (e.g., "norberg_angle", "thoracic_xray")
4) If the LLM kind conflicts with a matched manifest entity that has local_service_id/local_stock_id,
   keep the atom but inherit manifest identity/IDs and prefer manifest kind metadata.
5) Use both raw mention (span_text) and normalized_name in the manifest to resolve ASR variants.
"""
    return base_prompt + manifest_block


def parse_soap_sections(soap_note_text: str) -> Dict[str, str]:
    """
    Parse SOAP note into sections (Subjective, Objective, Assessment, Plan).
    Minimal implementation for compatibility.
    """
    sections = {
        "Subjective": "",
        "Objective": "",
        "Assessment": "",
        "Plan": "",
    }
    
    # Simple regex-based parsing
    import re
    patterns = {
        "Subjective": r"(?i)^\s*(?:S|SUBJECTIVE)[:\s]+\s*(.*?)(?=\n\s*(?:O|OBJECTIVE|A|ASSESSMENT|P|PLAN)[:\s]|$)",
        "Objective": r"(?i)^\s*(?:O|OBJECTIVE)[:\s]+\s*(.*?)(?=\n\s*(?:A|ASSESSMENT|P|PLAN)[:\s]|$)",
        "Assessment": r"(?i)^\s*(?:A|ASSESSMENT)[:\s]+\s*(.*?)(?=\n\s*(?:P|PLAN)[:\s]|$)",
        "Plan": r"(?i)^\s*(?:P|PLAN)[:\s]+\s*(.*?)$",
    }
    
    for section, pattern in patterns.items():
        match = re.search(pattern, soap_note_text, re.DOTALL | re.MULTILINE)
        if match:
            sections[section] = match.group(1).strip()
    
    return sections


def enrich_atoms_with_manifest_bindings(
    atoms_list: List[Dict[str, Any]],
    entity_manifest_json: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Enrich knowledge atoms with manifest binding info (binding_level, binding_track).
    Minimal implementation: returns atoms as-is (no enrichment needed in minimal mode).
    """
    return atoms_list


def _sha256(text: str) -> str:
    """Simple SHA256 hash function."""
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _normalize_frequency(frequency_raw: str) -> str:
    """Normalize frequency to Morning/Afternoon/Night/Custom."""
    if not frequency_raw:
        return "Custom"
    freq_lower = frequency_raw.lower()
    # Check for multiple times (e.g., "Morning/Night" or "Morning and Night")
    if "/" in freq_lower or " and " in freq_lower or "&" in freq_lower:
        return "Custom"
    elif "morning" in freq_lower and ("night" in freq_lower or "evening" in freq_lower):
        return "Custom"
    elif "morning" in freq_lower:
        return "Morning"
    elif "afternoon" in freq_lower:
        return "Afternoon"
    elif "night" in freq_lower or "evening" in freq_lower:
        return "Night"
    else:
        return "Custom"


def _normalize_instructions(instructions_raw: str) -> str:
    """Normalize instructions to After meal/Before meal/With Meal or custom."""
    if not instructions_raw:
        return ""
    inst_lower = instructions_raw.lower()
    if "after meal" in inst_lower or "after food" in inst_lower:
        return "After meal"
    elif "before meal" in inst_lower or "before food" in inst_lower:
        return "Before meal"
    elif "with meal" in inst_lower or "with food" in inst_lower:
        return "With Meal"
    else:
        return instructions_raw


# Certainty hierarchy for clinical history roll-up: highest wins
_ASSERTION_CERTAINTY_ORDER = ("CONF", "SUSP", "HIST", "RECUR", "NEG")
_SECTION_ORDER = ("Subjective", "Objective", "Assessment", "Plan", "Signalment", "")


def summarize_clinical_history(
    raw_items: List[Dict[str, Any]],
    pet_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Clinical Resolution & Deduplication Layer for Module 5 (Clinical History).

    - Identity/noise filtering: drop Reason kind and concepts matching pet name (e.g. "Oreo", "examination of Oreo").
    - Status roll-up: when same condition appears with different assertion_id, keep highest certainty (CONF > SUSP > HIST > NEG).
    - Cross-sectional deduplication: group by kb_concept_id or normalized concept; merge source_text into chronological_summary.
    - Result: clean Active Problem List (e.g. "Hip Dysplasia (Confirmed) — Based on posture, Ortolani test.").
    """
    if not raw_items:
        return []
    pet = (pet_name or "").strip().lower()
    # 1. Noise filter: drop Reason kind and pet-name-only / administrative phrases
    filtered: List[Dict[str, Any]] = []
    for item in raw_items:
        kind = (item.get("kind") or "").strip()
        concept = (item.get("condition_finding") or item.get("concept") or "").strip()
        concept_lower = concept.lower()
        if kind == "Reason":
            continue
        if pet and (
            concept_lower == pet
            or concept_lower == f"examination of {pet}"
            or concept_lower == f"exam of {pet}"
            or concept_lower.startswith(f"{pet} ")  # "Oreo presented for..."
        ):
            continue
        filtered.append(item)
    # 2. Group by kb_concept_id (entity manifest anchor) or normalized concept name
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for item in filtered:
        kb_id = item.get("kb_concept_id") or item.get("concept_id")
        concept = (item.get("condition_finding") or item.get("concept") or "").strip()
        norm_name = re.sub(r"\s+", " ", concept).lower().strip() if concept else ""
        key = (kb_id, norm_name) if kb_id is not None else (None, norm_name or f"_blank_{id(item)}")
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    # 3. Roll up: pick best assertion_id, merge source texts in section order
    result: List[Dict[str, Any]] = []
    for (_kb_id, _norm), items in groups.items():
        if not items:
            continue
        # Resolve assertion to highest certainty in group (CONF > SUSP > HIST > RECUR > NEG)
        def _certainty_index(a: str) -> int:
            a = (a or "").strip().upper()
            for i, level in enumerate(_ASSERTION_CERTAINTY_ORDER):
                if level == a:
                    return i
            return len(_ASSERTION_CERTAINTY_ORDER)
        best_idx = _certainty_index(items[0].get("assertion_id") or "")
        best = items[0]
        for it in items[1:]:
            idx = _certainty_index(it.get("assertion_id") or "")
            if idx < best_idx:
                best_idx = idx
                best = it
        best_assertion = (best.get("assertion_id") or "").strip().upper() or "CONF"
        concept = best.get("condition_finding") or best.get("concept") or ""
        kind = best.get("kind") or ""
        confirm_target = best.get("confirm_and_add_to") or "Active Problems"
        status = "UNLINKED" if best_assertion == "SUSP" else ("NEGATED" if best_assertion == "NEG" else "CONFIRMED")
        # Chronological summary: concatenate source_text from all mentions (by section order)
        by_section: Dict[str, List[str]] = {}
        for it in items:
            sec = (it.get("section") or "").strip()
            src = (it.get("source_text") or "").strip()
            if src:
                by_section.setdefault(sec, []).append(src)
        parts = []
        for sec in _SECTION_ORDER:
            if sec in by_section and by_section[sec]:
                for s in by_section[sec]:
                    if s and s not in parts:
                        parts.append(s)
        for sec, texts in by_section.items():
            if sec not in _SECTION_ORDER:
                for s in texts:
                    if s and s not in parts:
                        parts.append(s)
        chronological_summary = " | ".join(parts) if parts else ""
        result.append({
            "condition_finding": concept.strip(),
            "kind": kind,
            "assertion_id": best_assertion,
            "confirm_and_add_to": confirm_target,
            "status": status,
            "chronological_summary": chronological_summary,
            "kb_concept_id": _kb_id,
        })
    return result


# ---------------------------------------------------------------------------
# Category-First Streamlined Dashboard (Human-in-the-Loop)
# Headers group by functional role; status (CONFIRMED/DRAFTED/ACTION_REQUIRED) within each.
# Threshold-based: >90% CONFIRMED, 60-90% DRAFTED (single-click verify), <60% or multi-variant ACTION_REQUIRED.
# ---------------------------------------------------------------------------
STREAMLINED_HEADERS = (
    "signalment_vitals",            # Signalment & Vitals: baseline data
    "clinical_assessment",          # Clinical Assessment: symptoms, conditions, findings
    "pharmacy_administered",        # Medications: Administered (in-clinic, today's bill)
    "pharmacy_prescribed",          # Medications: Prescribed (home care, prescription label)
    "diagnostics_labs_imaging",     # Diagnostics (Labs & Imaging): Done-At, Sample Type
    "clinical_procedures_services", # Clinical Procedures & Services: Estimated Duration, Service Charge
    "other_products_retail",        # Other Products & Retail: Diet & Nutrition, Non-Medical (SKU, Quantity, Unit Price)
    "preventive",                   # Preventive: vaccines, wellness
    "patient_reminders",            # Patient Reminders: owner instructions (discharge summary)
    "clinical_follow_ups",         # Clinical Follow-ups: operational tasks (scheduler/CRM)
)
# Schema-driven: required fields for HITL verification (UI cannot Confirm without these)
PHARMACY_ADMINISTERED_REQUIRED_FIELDS = ("dosage", "route")
PHARMACY_PRESCRIBED_REQUIRED_FIELDS = ("frequency", "duration")
CONFIRMED_THRESHOLD = 0.90   # Auto-confirm above this
DRAFTED_THRESHOLD = 0.60     # Drafted state (single-click verify) between this and CONFIRMED_THRESHOLD


def map_kind_to_dashboard_head(kind: str, intent_context: str = "", section: str = "") -> str:
    """
    Map entity/atom kind (and optional intent/section) to streamlined dashboard header.
    Returns one of: signalment_vitals, clinical_assessment, pharmacy_administered,
    pharmacy_prescribed, diagnostics_labs_imaging, clinical_procedures_services,
    other_products_retail, preventive, patient_reminders, clinical_follow_ups.
    """
    kind = (kind or "").strip().lower()
    intent = (intent_context or "").strip().lower()
    sec = (section or "").strip().lower()
    # Reminder-type: split into Patient Reminders (informational) vs Clinical Follow-ups (actionable)
    if intent in ("scheduled", "future") or "follow" in kind:
        return "clinical_follow_ups"
    if intent in ("recommended", "reminder") or "reminder" in kind:
        return "patient_reminders"
    # Signalment, VitalSign, Measurement → signalment_vitals
    if kind in ("signalment", "vitalsign", "vital sign", "measurement", "vital"):
        return "signalment_vitals"
    # Condition, Finding, Observation (clinical) → clinical_assessment
    if kind in ("condition", "finding", "observation", "reason", "disease", "diagnosis"):
        return "clinical_assessment"
    # Vaccine, Preventive → preventive (unless intent is Reminder → action_plan already)
    if kind in ("vaccine", "preventive", "preventive care"):
        return "preventive"
    # Medication, Drug, Supplement, Nutrition → pharmacy by intent (retail routing done in builder via inventory category)
    if kind in ("medicine", "drug", "medication", "substance", "vaccine", "supplement", "nutrition"):
        return "pharmacy_administered" if intent == "administered" else "pharmacy_prescribed"
    # Diagnostics (Labs & Imaging) vs Clinical Procedures & Services: split by schema
    if kind in ("diagnostictest", "diagnostic", "labtest", "imaging"):
        return "diagnostics_labs_imaging"
    if kind in ("procedure", "service", "treatment"):
        return "clinical_procedures_services"
    # Default: put in clinical_assessment if it looks clinical, else clinical_procedures_services
    if kind in ("anatomy", "organism", "toxin"):
        return "clinical_assessment"
    return "clinical_procedures_services"


def generate_streamlined_dashboard(
    dashboard: Dict[str, Any],
    entity_manifest: Optional[List[Dict[str, Any]]] = None,
    signalment: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build Category-First streamlined dashboard from the module-based dashboard.
    Every entity appears under one of: signalment_vitals, clinical_assessment,
    pharmacy_administered, pharmacy_prescribed, diagnostics_labs_imaging,
    clinical_procedures_services, other_products_retail, preventive,
    patient_reminders, clinical_follow_ups.
    Pharmacy is split: Administered (in-clinic, today's bill) vs Prescribed (home care, prescription label).
    Reminders are split: Patient Reminders vs Clinical Follow-ups.
    Status: CONFIRMED, DRAFTED, or ACTION_REQUIRED. required_fields drive HITL validation per header.
    """
    out = {h: [] for h in STREAMLINED_HEADERS}
    manifest = entity_manifest or []

    # ---- Signalment (baseline) → signalment_vitals ----
    if signalment and isinstance(signalment, dict):
        parts = []
        for key in ("age", "breed", "gender", "weight", "species", "pet_name", "patient_name"):
            val = signalment.get(key) or signalment.get(key.replace("_", " ").title())
            if val:
                parts.append(str(val).strip())
        if parts:
            out["signalment_vitals"].append({
                "name": " | ".join(parts),
                "original_mention": "",
                "status": "CONFIRMED",
                "linked_id": None,
                "suggestions": [],
                "label": "Signalment",
            })

    def _score_for_concept(concept_name: str) -> Optional[float]:
        if not concept_name:
            return None
        cn = (concept_name or "").strip().lower()
        for e in manifest:
            if not isinstance(e, dict):
                continue
            for key in ("display_name", "normalized_name", "span_text"):
                val = (e.get(key) or "").strip().lower()
                if val and (cn == val or cn in val or val in cn):
                    s = e.get("similarity_score")
                    if s is not None:
                        return float(s)
                    return None
        return None

    def _status(has_linked_id: bool, score: Optional[float], suggestions_count: int) -> str:
        if not has_linked_id:
            return "ACTION_REQUIRED"
        if suggestions_count > 1:
            return "ACTION_REQUIRED"
        # Linked but no score from manifest → treat as DRAFTED (single-click verify)
        if score is None:
            return "DRAFTED"
        if score >= CONFIRMED_THRESHOLD:
            return "CONFIRMED"
        if score >= DRAFTED_THRESHOLD:
            return "DRAFTED"
        return "ACTION_REQUIRED"

    # ---- Vitals → signalment_vitals ----
    for v in dashboard.get("vitals", []) or []:
        out["signalment_vitals"].append({
            "name": v.get("vital_name") or v.get("value") or "",
            "original_mention": "",
            "status": "CONFIRMED" if v.get("value") else "ACTION_REQUIRED",
            "linked_id": v.get("vital_id"),
            "suggestions": [],
            "value": v.get("value"),
            "label": "Vitals",
        })

    # ---- Procedures → clinical_procedures_services (Estimated Duration, Service Charge) ----
    for item in dashboard.get("procedures_services", []) or []:
        name = item.get("item_name") or ""
        sid = item.get("service_id")
        score = _score_for_concept(name)
        out["clinical_procedures_services"].append({
            "name": name,
            "original_mention": name,
            "status": _status(bool(sid), score, 0),
            "linked_id": sid,
            "suggestions": [],
            "remarks": item.get("remarks"),
        })
    # ---- Diagnostics (Labs & Imaging) → diagnostics_labs_imaging (Done-At, Sample Type) ----
    for item in dashboard.get("diagnostics", []) or []:
        name = item.get("test_name") or item.get("lab_radiology") or ""
        sid = item.get("service_id")
        score = _score_for_concept(name)
        out["diagnostics_labs_imaging"].append({
            "name": name,
            "original_mention": name,
            "status": _status(bool(sid), score, 0),
            "linked_id": sid,
            "suggestions": [],
            "remarks": item.get("remarks"),
            "done": item.get("done"),
            "sample_collected": item.get("sample_collected"),
        })
    # ---- Other Products & Retail (Diet & Nutrition, Non-Medical inventory) ----
    for item in dashboard.get("other_products_retail", []) or []:
        name = item.get("item_name") or ""
        inv_id = item.get("inventory_id")
        score = _score_for_concept(name)
        out["other_products_retail"].append({
            "name": name,
            "original_mention": name,
            "status": _status(bool(inv_id), score, len(item.get("suggestions") or [])),
            "linked_id": inv_id,
            "suggestions": item.get("suggestions") or [],
            "quantity": item.get("quantity") or "",
            "unit_price": item.get("unit_price"),
            "remarks": item.get("remarks"),
        })

    # ---- Medications: Administered (in-clinic, today's bill) ----
    for item in dashboard.get("medications_administered", []) or []:
        name = item.get("item_name") or ""
        inv_id = item.get("inventory_id")
        score = _score_for_concept(name)
        out["pharmacy_administered"].append({
            "name": name,
            "original_mention": name,
            "status": _status(bool(inv_id), score, 0),
            "linked_id": inv_id,
            "suggestions": [],
            "dosage": item.get("dosage") or "",
            "route": item.get("route") or "",
            "remarks": item.get("remarks"),
            "required_fields": list(PHARMACY_ADMINISTERED_REQUIRED_FIELDS),
            "label": "Administered (In-Clinic)",
        })
    # ---- Medications: Prescribed (home care, prescription label / discharge) ----
    for item in dashboard.get("medications_prescribed", []) or []:
        name = item.get("item_name") or ""
        inv_id = item.get("inventory_id")
        score = _score_for_concept(name)
        out["pharmacy_prescribed"].append({
            "name": name,
            "original_mention": name,
            "status": _status(bool(inv_id), score, 0),
            "linked_id": inv_id,
            "suggestions": [],
            "dosage": item.get("dosage") or "",
            "frequency": item.get("frequency") or "",
            "duration": item.get("duration") or "",
            "quantity": item.get("quantity") or "",
            "instructions": item.get("instructions") or "",
            "remarks": item.get("remarks"),
            "required_fields": list(PHARMACY_PRESCRIBED_REQUIRED_FIELDS),
            "label": "Prescribed (Home Care)",
        })

    # ---- Reminders → split into Patient Reminders (instructions) vs Clinical Follow-ups (operational) ----
    for item in dashboard.get("reminders_follow_ups", []) or []:
        name = item.get("item_name") or ""
        item_id = item.get("item_id")
        due_date = (item.get("due_date") or "").strip()
        intent = (item.get("intent_context") or "").strip()
        kind = (item.get("kind") or "").strip()
        # Operational = actionable (booking/scheduler): Scheduled, Future, or has service_id / specific timing
        has_booking_id = item_id and str(item_id).replace("-", "").isdigit()
        has_specific_timing = due_date and due_date.upper() not in ("ASAP", "")
        is_operational = (
            intent in ("Scheduled", "Future")
            or has_booking_id
            or has_specific_timing
        )
        # Build scheduling_data for operational follow-ups (Book Now / Add to Pending)
        scheduling_data = None
        if is_operational:
            try:
                sid = int(item_id) if (item_id and str(item_id).replace("-", "").isdigit()) else None
            except (TypeError, ValueError):
                sid = None
            scheduling_data = {
                "service_id": sid,
                "temporal_offset": due_date if due_date else None,
                "recommended_date": None,  # Can be filled by Super-Pass/LLM relative to current date
                "priority": "Routine",
            }
            if item_id and sid is None:
                scheduling_data["item_id"] = item_id  # Placeholder e.g. [PROCEDURE-XXX] for UI
        entry = {
            "name": name,
            "original_mention": name,
            "status": "CONFIRMED" if has_booking_id else ("DRAFTED" if is_operational else "ACTION_REQUIRED"),
            "linked_id": item_id,
            "suggestions": [],
            "due_date": due_date or None,
            "remarks": item.get("remarks"),
            "is_operational": is_operational,
            "scheduling_data": scheduling_data,
            "label": "Clinical Follow-up" if is_operational else "Patient Instruction",
        }
        if is_operational:
            out["clinical_follow_ups"].append(entry)
        else:
            out["patient_reminders"].append(entry)

    # ---- Clinical History (Module 5) → clinical_assessment ----
    for item in dashboard.get("module_5_clinical_history", []) or []:
        name = item.get("condition_finding") or ""
        status = (item.get("status") or "").strip()
        if status == "NEGATED":
            ui_status = "CONFIRMED"  # Negated is confirmed as "not applied"
        elif status == "UNLINKED":
            ui_status = "ACTION_REQUIRED"
        else:
            ui_status = "CONFIRMED" if item.get("kb_concept_id") else "ACTION_REQUIRED"
        out["clinical_assessment"].append({
            "name": name,
            "original_mention": name,
            "status": ui_status,
            "linked_id": item.get("kb_concept_id"),
            "suggestions": [],
            "confirm_and_add_to": item.get("confirm_and_add_to"),
        })

    # ---- Unlinked (Module 4) → route by kind to correct header; add schema fields for pharmacy ----
    for item in dashboard.get("module_4_unlinked_entities", []) or []:
        name = item.get("item_name") or ""
        suggestions = item.get("suggestions", []) or []
        kind_raw = item.get("kind", "")
        intent_ctx = (item.get("intent_context") or "").strip()
        if not kind_raw and "(" in str(item.get("intent", "")):
            intent_str = str(item.get("intent", ""))
            if ")" in intent_str:
                kind_raw = intent_str.split("(")[-1].replace(")", "").strip()
        head = map_kind_to_dashboard_head(kind_raw or "Procedure", intent_ctx, "")
        entry = {
            "name": name,
            "original_mention": name,
            "status": "ACTION_REQUIRED",
            "linked_id": item.get("service_id") or item.get("inventory_id"),
            "suggestions": suggestions,
            "remarks": item.get("remarks"),
            "reason": item.get("reason"),
        }
        # Schema-driven: add required_fields and medication attributes for pharmacy headers (Frequency/Duration even when unlinked)
        if head == "pharmacy_administered":
            entry["dosage"] = item.get("dosage") or ""
            entry["route"] = item.get("route") or ""
            entry["required_fields"] = list(PHARMACY_ADMINISTERED_REQUIRED_FIELDS)
            entry["label"] = "Administered (In-Clinic)"
        elif head == "pharmacy_prescribed":
            entry["dosage"] = item.get("dosage") or ""
            entry["frequency"] = item.get("frequency") or ""
            entry["duration"] = item.get("duration") or ""
            entry["quantity"] = item.get("quantity") or ""
            entry["instructions"] = item.get("instructions") or ""
            entry["required_fields"] = list(PHARMACY_PRESCRIBED_REQUIRED_FIELDS)
            entry["label"] = "Prescribed (Home Care)"
        elif head == "other_products_retail":
            entry["quantity"] = item.get("quantity") or ""
            entry["unit_price"] = item.get("unit_price")
        out[head].append(entry)

    return out


def _build_verification_dashboard(
    atoms_list: List[Dict[str, Any]],
    conn=None,
    logger: Optional[logging.Logger] = None,
    entity_manifest: Optional[List[Dict[str, Any]]] = None,  # NEW: For suggestions lookup
    pet_name: Optional[str] = None,
    signalment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build verification dashboard JSON with strict ID-gate routing logic.
    
    This function interfaces with both Service Master and Inventory Master to:
    1. Validate IDs exist (high-integrity check)
    2. Pull canonical names from masters (not from Phase 2 extraction)
    3. Get metadata (Form for medications, category for services)
    4. Ensure correct routing based on master source
    
    STRICT DATA MAPPING:
    - Procedures/Services: Service Master → service_id
    - Medications (Admin/Presc): Inventory Master → inventory_id (stock_id)
    - Diagnostics: Service Master → service_id
    - Reminders: Both Masters → service_id or inventory_id
    - Unlinked Entities: None (Phase 1 grounding failed) → Null (Action Required)
    
    INTENT GATE: intent_context "Recommended", "Future", or "Reminder" → Reminders ONLY (never billing).
    Billing (Diagnostics/Procedures/Medications) only when intent is Performed, Ordered, Administered, or Prescribed.
    OUTPUT FORMAT:
    - Procedures/Services: Item name (from service_master), Service ID, status, Remarks
    - If Administered: Item name (from inventory), Inventory ID, status, administered/prescribed, Dosage, route, Remarks

    UX: The "remarks" field on procedures_services and diagnostics holds clinical context (e.g. "Advise physiotherapy
    including swimming", "to measure the Norberg angle"). The UI MUST display remarks so the vet knows why a procedure
    or diagnostic was ordered (e.g. why the X-ray was ordered).
    - If Prescribed: Item name (from inventory), Inventory ID, status, administered/prescribed, Fulfillment, Quantity, dosage, duration, frequency, Instructions, remarks
    - If Diagnostics: Lab/Radiology, Test Name (from service_master), Date of test, Done (In-house/External), Sample Collected
    - Reminders: Item Name (from master), Item ID, Due Date, Remarks
    - Vitals: Vital name, Value
    - Module 4 (Unlinked): Item Name, Intent, Status, Remarks
    - Module 5 (Clinical History): Condition/Finding, Confirm and add to, Status
    
    Args:
        atoms_list: List of knowledge atoms from Phase 2.a
        conn: Database connection (uses get_pg_conn() if None)
        logger: Logger instance
        entity_manifest: Optional entity manifest for suggestions lookup
        pet_name: Optional pet/patient name for clinical history noise filter (e.g. drop "Oreo", "examination of Oreo")
        signalment: Optional dict with pet_name/patient_name; used to derive pet_name if pet_name not set
    """
    dashboard = {
        "procedures_services": [],
        "medications_administered": [],
        "medications_prescribed": [],
        "diagnostics": [],
        "other_products_retail": [],
        "reminders_follow_ups": [],
        "vitals": [],
        "module_4_unlinked_entities": [],
        "module_5_clinical_history": [],
    }
    
    # CRITICAL: Filter out non-billing kinds BEFORE processing
    # Verification dashboard is ONLY for billing items (same restrictions as knowledge atoms)
    BILLING_ONLY_KINDS = {
        "Procedure", "Service", "Treatment",
        "Medicine", "Drug", "Medication", "Substance", "Vaccine", "Supplement", "Nutrition",
        "LabTest", "DiagnosticTest", "Diagnostic", "Imaging",
        "VitalSign", "Vital",
        "Reminder", "FollowUp", "Follow-up",
        "ReasonForVisit", "Reason",
        "ParasiteControl", "Preventive",
    }
    NON_BILLING_KINDS = {
        "Identity", "PatientName", "OwnerName", "Owner",
        "Signalment", "Species", "Breed", "Sex", "Age", "Weight",
        "Anatomy", "BodySite", "BodySystem",
        "Symptom", "Finding", "Observation", "Condition", "Disease", "Diagnosis",
        "Other",  # Explicitly exclude "Other" kind
    }
    
    filtered_atoms_list = []
    excluded_count = 0
    for atom in atoms_list or []:
        if not isinstance(atom, dict):
            continue
        
        kind = (atom.get("kind") or "").strip()
        # Normalize kind aliases
        kind_alias = {
            "Medication": "Medicine",
            "Substance": "Medicine",
            "Medications": "Medicine",
            "Drugs": "Drug",
        }
        kind_normalized = kind_alias.get(kind, kind)
        
        # Exclude non-billing kinds
        if kind_normalized in NON_BILLING_KINDS or kind in NON_BILLING_KINDS:
            excluded_count += 1
            continue
        
        # Include billing kinds
        if kind_normalized in BILLING_ONLY_KINDS or kind in BILLING_ONLY_KINDS:
            filtered_atoms_list.append(atom)
        # Also include if kind is empty/unknown but has billing intent (safety net)
        elif not kind and atom.get("intent_context") in ("Performed", "Ordered", "Administered", "Prescribed"):
            filtered_atoms_list.append(atom)
        else:
            # Unknown kind - exclude by default (strict filtering)
            excluded_count += 1
    
    if excluded_count > 0 and logger:
        logger.info(f"  🔍 Verification dashboard: Filtered {excluded_count} non-billing atoms (Identity/Signalment/Anatomy/Symptom/Other)")
    
    # Process filtered atoms only
    for atom in filtered_atoms_list:
        kind = (atom.get("kind") or "").strip()
        # Phase 2 LLMs sometimes emit near-synonym kinds (e.g., Medication/Substance vs Medicine/Drug).
        # Canonicalize here so routing is stable and we don't mis-route linked items into Module 4.
        kind_alias = {
            "Medication": "Medicine",
            "Substance": "Medicine",
            "Medications": "Medicine",
            "Drugs": "Drug",
        }
        kind = kind_alias.get(kind, kind)
        assertion_id = (atom.get("assertion_id") or "").strip()
        intent_context = (atom.get("intent_context") or atom.get("intent_type") or "").strip()
        section = (atom.get("section") or "").strip()
        concept = (atom.get("concept") or "").strip()
        attributes = atom.get("attributes", {}) or {}
        
        # Get IDs from enriched atom (Phase 1 grounding provides immutable source metadata)
        # Some upstream components may mirror IDs into `codes`, so treat that as a fallback.
        codes = atom.get("codes", {}) if isinstance(atom.get("codes"), dict) else {}
        local_service_id = atom.get("local_service_id") or codes.get("local_service_id")
        local_stock_id = atom.get("local_stock_id") or codes.get("local_stock_id")
        
        # CRITICAL: Linking and status use local_stock_id / local_service_id as PRIMARY KEY (not string matching).
        # Atoms get these IDs from _enrich_atoms_with_manifest_ids (kb_phase2_integration), which matches by manifest normalized_name + kind so Plan terms (e.g. "Cefpodoxime syrup") link to the correct SKU even when display_name differs (e.g. "CefPET Dry Syrup").
        # - local_service_id (from service_master) → Module 1 (Services) or Module 3 (Diagnostics)
        # - local_stock_id (from inventory_master) → Module 2 (Medications)
        has_service_id = bool(local_service_id)
        has_stock_id = bool(local_stock_id)
        has_local_id = has_service_id or has_stock_id
        
        # RULE 1: Vitals (always goes to vitals module)
        # Extract vitals from explicit VitalSign kind OR from a conservative keyword-based fallback.
        # This protects against Phase 2 kind drift where models emit Finding/Observation for vitals
        # (e.g., "Mucous membranes normal" should be a VitalSign).
        is_vital = kind == "VitalSign" or "vital" in kind.lower()
        vital_keywords = (
            "temperature", "temp",
            "heart rate", "hr", "pulse", "pulse quality",
            "respiration", "respiratory rate", "rr", "breaths per minute", "bpm",
            "crt", "capillary refill",
            "mucous membrane", "mucous membranes", "mm",
            "weight", "bcs", "body condition", "mcs", "muscle condition",
            "mentation", "bar", "qar", "obtunded", "stuporous", "dull", "depressed",
            "pain score", "pain",
            "hydration", "skin turgor",
            "blood pressure", "bp", "systolic", "diastolic", "map",
            "spo2", "oxygen saturation", "etco2", "end tidal", "glucose", "blood glucose",
            "auscultation", "heart sounds", "murmur", "arrhythmia", "lung sounds", "crackles", "wheezes", "clear lungs",
        )
        if (not is_vital) and section == "Objective" and kind in ("Finding", "Observation", "Condition"):
            probe = " ".join([
                concept or "",
                str(attributes.get("metric_name") or ""),
                str(attributes.get("vital_name") or ""),
                str(attributes.get("location") or ""),
                str(attributes.get("observation") or ""),
            ]).lower()
            if any(kw in probe for kw in vital_keywords):
                is_vital = True
        
        # Also check for weight/BCS in Condition attributes (e.g., "Obesity" condition may have weight/BCS attributes)
        if not is_vital and kind == "Condition":
            # Check if condition has vital-related attributes
            weight = attributes.get("weight") or attributes.get("has_weight") or ""
            bcs = attributes.get("bcs") or attributes.get("body_condition_score") or attributes.get("has_bcs") or ""
            if weight or bcs:
                # Extract as vitals
                if weight:
                    vital_item = {
                        "vital_name": "Weight",
                        "value": weight,
                    }
                    dashboard["vitals"].append(vital_item)
                if bcs:
                    vital_item = {
                        "vital_name": "Body Condition Score (BCS)",
                        "value": bcs,
                    }
                    dashboard["vitals"].append(vital_item)
                # Continue processing condition (may also go to Module 5)
                # Don't continue here - let it also route to Module 5 if needed
        
        if is_vital:
            # Prefer explicit numeric value; fall back to observation/qualitative_flag.
            raw_val = (
                attributes.get("value")
                or attributes.get("has_value")
                or attributes.get("numeric_value")
                or attributes.get("measurement")
                or ""
            )
            if raw_val in (None, "None"):
                raw_val = ""
            if not raw_val:
                raw_val = attributes.get("qualitative_flag") or attributes.get("observation") or ""
            unit = attributes.get("unit") or attributes.get("has_unit") or ""
            val_str = str(raw_val).strip() if raw_val is not None else ""
            unit_str = str(unit).strip() if unit is not None else ""
            value_out = f"{val_str} {unit_str}".strip() if (val_str and unit_str) else val_str
            vital_item = {
                "vital_name": attributes.get("metric_name") or attributes.get("vital_name") or attributes.get("location") or concept or "",
                "value": value_out,
            }
            dashboard["vitals"].append(vital_item)
            continue
        
        # INTENT GATE: "Recommended", "Future", "Reminder" → Reminders ONLY (never billing)
        # Stops Plan-section mentions (e.g. "X-ray" with intent "Ordered" when LLM meant future) from hitting diagnostics
        _REMINDER_ONLY_INTENTS = ("Recommended", "Future", "Reminder")
        _BILLING_INTENTS_SERVICES = ("Performed", "Ordered")
        _BILLING_INTENTS_MEDICATIONS = ("Administered", "Prescribed")
        intent_norm = (intent_context or "").strip()
        if intent_norm in _REMINDER_ONLY_INTENTS:
            if (has_service_id or has_stock_id) and kind in (
                "LabTest", "DiagnosticTest", "Diagnostic", "Procedure", "Service", "Treatment",
                "Medicine", "Drug", "Medication", "Substance", "Vaccine", "Supplement", "Nutrition",
            ):
                reminder_item = {
                    "item_name": concept or "",
                    "item_id": local_service_id or local_stock_id or f"[{kind.upper()}-XXX]" if kind else "[UNKNOWN-XXX]",
                    "due_date": attributes.get("due_on") or attributes.get("has_due_on") or attributes.get("duration") or "ASAP",
                    "remarks": attributes.get("action_item") or attributes.get("remarks") or atom.get("source_text", "") or "",
                    "intent_context": intent_norm,
                    "kind": kind or "",
                }
                dashboard["reminders_follow_ups"].append(reminder_item)
                if logger:
                    logger.debug(f"  Intent gate: {intent_norm} → Reminders (not billing): '{concept or kind}'")
                continue
        
        # RULE 2: Diagnostics (Service ID from service_master → Module 3)
        # Billing only when intent is Performed or Ordered (not Recommended/Future/Reminder).
        # Service IDs that are always diagnostics (e.g. X-ray 1185): route to diagnostics even if kind is Procedure/Imaging.
        _DIAGNOSTIC_SERVICE_IDS = {1185}  # X-ray / imaging services that must populate diagnostics module
        is_diagnostic_kind = kind in ("LabTest", "DiagnosticTest", "Diagnostic")
        try:
            is_diagnostic_service_id = local_service_id is not None and int(local_service_id) in _DIAGNOSTIC_SERVICE_IDS
        except (TypeError, ValueError):
            is_diagnostic_service_id = False
        if has_service_id and (is_diagnostic_kind or is_diagnostic_service_id) and assertion_id == "CONF" and intent_context in _BILLING_INTENTS_SERVICES:
            # Query Service Master to validate ID and get canonical name
            service_record = get_service_master_record(local_service_id, conn=conn, logger=logger)
            if service_record:
                # Use canonical name from master (not from Phase 2 extraction)
                canonical_name = service_record.get("procedure_name") or concept or ""
                # Determine Lab vs Radiology from master category or concept
                category_from_master = service_record.get("category", "").lower()
                is_radiology = (
                    any(x in (canonical_name or "").lower() for x in ["x-ray", "xray", "radiology", "imaging", "ultrasound", "ct", "mri", "radiograph"])
                    or "radiology" in category_from_master
                    or "imaging" in category_from_master
                )
                category = "Radiology" if is_radiology else "LAB"
                
                diagnostic_item = {
                    "service_id": local_service_id or None,  # Include service_id for billing
                    "lab_radiology": category or "LAB",
                    "test_name": canonical_name or "",
                    "date_of_test": attributes.get("due_on") or attributes.get("has_due_on") or "TBD",
                    "done": "In-house" if intent_context == "Performed" else "External",
                    "sample_collected": "Yes" if intent_context == "Performed" else "No",
                    "remarks": attributes.get("remarks") or service_record.get("remarks") or atom.get("source_text", "") or "",
                }
                dashboard["diagnostics"].append(diagnostic_item)
            else:
                # ID not found in Service Master - route to Unlinked (safety gate)
                if logger:
                    logger.warning(f"Service ID {local_service_id} not found in service_master for diagnostic - routing to unlinked")
                unlinked_item = {
                    "item_name": concept or "",
                    "kind": kind or "DiagnosticTest",
                    "intent_context": intent_context or "",
                    "intent": f"{intent_context or assertion_id or 'Unknown'} ({kind or 'Unknown'})",
                    "status": "ACTION REQUIRED",
                    "reason": "INVALID_SERVICE_ID",
                    "remarks": f"Diagnostic Service ID {local_service_id} not found in Service Master. Please verify or select correct service.",
                    "suggestions": [],
                }
                dashboard["module_4_unlinked_entities"].append(unlinked_item)
            continue
        
        # RULE 3: Reminders & Follow-ups (Future/Hypothetical/Recommended/Reminder intents)
        # Intent Gate: Recommended, Future, Reminder → Reminders ONLY (never billing)
        # HYPO or Scheduled also → Reminders even if item has IDs
        if (
            assertion_id == "HYPO"
            or intent_context == "Scheduled"
            or intent_context in ("Recommended", "Future", "Reminder")
            or (section == "Plan" and assertion_id == "CONF" and intent_context == "Scheduled" and kind not in ("LabTest", "DiagnosticTest", "Diagnostic"))
        ):
            reminder_item = {
                "item_name": concept or "",
                "item_id": local_service_id or local_stock_id or f"[{kind.upper()}-XXX]" if kind else "[UNKNOWN-XXX]",
                "due_date": attributes.get("due_on") or attributes.get("has_due_on") or attributes.get("duration") or "ASAP",
                "remarks": attributes.get("action_item") or attributes.get("remarks") or atom.get("source_text", "") or "",
                "intent_context": intent_context or "Reminder",
                "kind": kind or "",
            }
            dashboard["reminders_follow_ups"].append(reminder_item)
            continue
        
        # LOCUS-BASED ROUTING: At-home / instructed activities (swimming, walking on sand, low-impact exercise)
        # → Reminders, not Procedure billing or Module 4 Unlinked. Reduces "Missing Local ID" for home activities.
        _AT_HOME_ACTIVITY_CUES = (
            "swimming", "walk on sand", "walking on sand", "low-impact", "low impact",
            "physiotherapy", "physical therapy", "exercise", "home exercise", "at home",
            "owner to", "pet owner to", "instructed", "customer instruction",
        )
        if kind in ("Procedure", "Service", "Treatment") and not has_local_id:
            probe = " ".join([(concept or ""), (atom.get("source_text") or ""), str(attributes.get("action_item") or ""), str(attributes.get("remarks") or "")]).lower()
            if any(cue in probe for cue in _AT_HOME_ACTIVITY_CUES):
                reminder_item = {
                    "item_name": concept or "",
                    "item_id": f"[{kind.upper()}-XXX]",
                    "due_date": attributes.get("due_on") or attributes.get("has_due_on") or attributes.get("duration") or "ASAP",
                    "remarks": attributes.get("action_item") or attributes.get("remarks") or atom.get("source_text", "") or "",
                    "intent_context": intent_context or "Reminder",
                    "kind": kind or "",
                }
                dashboard["reminders_follow_ups"].append(reminder_item)
                if logger:
                    logger.debug(f"  Locus gate: at-home activity → Reminders (not billing): '{concept or kind}'")
                continue
        
        # RULE 3: Module 5 - Clinical History (Conditions, Findings, Reasons, Differential Diagnoses, Negated items - medical record only)
        # CRITICAL: NEG, HIST, SUSP items go to History even if they have IDs (ID-Gate filter)
        # Example: "We will NOT use Meloxicam" → has ID from Phase 1, but NEG assertion → goes to History, not billing
        if kind in ("Condition", "Finding", "Reason") or assertion_id in ("HIST", "SUSP", "NEG"):
            confirm_target = "Active Problems"
            if kind == "Condition":
                confirm_target = "Diagnosis List"
            elif kind == "Finding" or assertion_id == "SUSP":
                confirm_target = "Objective Findings"
            elif assertion_id == "NEG":
                # Negated items go to history (e.g., "declined vaccine", "not using medication")
                confirm_target = "Medical History"
            kb_concept_id = atom.get("kb_concept_id") or atom.get("concept_id") or (codes.get("kb_concept_id") if isinstance(codes, dict) else None) or (codes.get("concept_id") if isinstance(codes, dict) else None)
            history_item = {
                "condition_finding": concept or "",
                "kind": kind or "",
                "assertion_id": assertion_id or "",
                "confirm_and_add_to": confirm_target or "Active Problems",
                "status": "UNLINKED" if assertion_id == "SUSP" else ("NEGATED" if assertion_id == "NEG" else "CONFIRMED"),
                "source_text": (atom.get("source_text") or "").strip(),
                "section": section or "",
                "kb_concept_id": kb_concept_id,
            }
            dashboard["module_5_clinical_history"].append(history_item)
            continue
        
        # RULE 4: Module 4 - Unlinked Entities (Missing local IDs)
        # CRITICAL: Module 4 acts as a "Shadow" of billing modules - includes ALL enrichment fields
        # This allows vet to both link SKU AND finalize dosage/instructions in one place
        if not has_local_id:
            # Only clinical intents go to unlinked (not vitals, not history-only items)
            if kind in ("Medicine", "Drug", "Medication", "Substance", "Vaccine", "Supplement", "Procedure", "Service", "Treatment", "LabTest", "DiagnosticTest", "Diagnostic", "Nutrition"):
                # Determine status based on assertion
                status = "ACTION REQUIRED"
                reason = "MISSING_LOCAL_ID"
                if assertion_id == "HYPO":
                    status = "HYPOTHETICAL"
                    reason = "NONE_GENERIC"
                
                # Extract suggestions from atom attributes (Phase 2 preserves Phase 1 attributes)
                suggestions = []
                atom_attrs = atom.get("attributes", {}) or {}
                if isinstance(atom_attrs, dict):
                    suggestions = atom_attrs.get("suggestions", []) or []
                
                # Fallback: If no suggestions in atom, try to find from entity_manifest by fuzzy matching
                if not suggestions and entity_manifest:
                    concept_lower = (concept or "").lower()
                    for entity in entity_manifest:
                        if not isinstance(entity, dict):
                            continue
                        # Only consider unlinked entities (they have suggestions)
                        if entity.get("local_stock_id") or entity.get("local_service_id"):
                            continue
                        
                        entity_attrs = entity.get("attributes", {}) or {}
                        if not isinstance(entity_attrs, dict) or not entity_attrs.get("suggestions"):
                            continue
                        
                        # Try to match by concept name (fuzzy)
                        entity_names = [
                            entity.get("span_text", "").lower(),
                            entity.get("normalized_name", "").lower(),
                            entity.get("display_name", "").lower(),
                        ]
                        entity_names = [n for n in entity_names if n]
                        
                        # Check if concepts overlap (substring match or word overlap)
                        matched = False
                        for entity_name in entity_names:
                            if not entity_name:
                                continue
                            # Substring match or significant word overlap
                            if (concept_lower in entity_name or entity_name in concept_lower or
                                len(set(concept_lower.split()) & set(entity_name.split())) >= 1):
                                suggestions = entity_attrs.get("suggestions", []) or []
                                matched = True
                                break
                        
                        if matched:
                            break
                
                # Apply contextual recommendations to suggestions
                # Check if Assessment section contains conditions that match suggestions
                if suggestions:
                    # Get all atoms to find suspected conditions
                    suspected_conditions = []
                    for other_atom in atoms_list or []:
                        if isinstance(other_atom, dict):
                            other_kind = (other_atom.get("kind") or "").strip()
                            other_section = (other_atom.get("section") or "").strip()
                            if other_kind in ("Condition", "Disease", "Diagnosis") and other_section == "Assessment":
                                cond_name = (other_atom.get("concept") or "").strip().lower()
                                if cond_name:
                                    suspected_conditions.append(cond_name)
                    
                    # Boost suggestions that match clinical context
                    for sug in suggestions:
                        sug_name_lower = (sug.get("name") or "").lower()
                        # Example: If suspected condition is "Hip Dysplasia", boost "HIP X RAY"
                        for cond in suspected_conditions:
                            if "hip" in cond and "hip" in sug_name_lower:
                                sug["recommendation"] = "HIGH"
                                sug["match_score"] = min(1.0, sug.get("match_score", 0) * 1.2)
                                break
                            elif "dental" in cond and "dental" in sug_name_lower:
                                sug["recommendation"] = "HIGH"
                                sug["match_score"] = min(1.0, sug.get("match_score", 0) * 1.2)
                                break
                    
                    # Sort suggestions: HIGH recommendation first, then by match_score descending
                    suggestions.sort(key=lambda x: (
                        0 if x.get("recommendation") == "HIGH" else (1 if x.get("recommendation") == "MEDIUM" else 2),
                        -x.get("match_score", 0)
                    ))
                    # Cap at top 5 for verification dashboard (HITL: vet selects from best candidates)
                    suggestions = suggestions[:5]
                
                # Determine reason based on whether we have suggestions
                if suggestions:
                    reason = "AMBIGUOUS_MATCH" if len(suggestions) > 1 else "MISSING_LOCAL_ID"
                    remarks = f"Generic {kind.lower()} detected. Select a specific billing variant from the suggestions below." if kind else "Item detected. Select a specific billing variant from the suggestions below."
                else:
                    # Use "Generic vaccine" only for actual Vaccine kind. Flea/tick/deworming are NOT vaccines.
                    if kind == "Vaccine":
                        remarks = "Generic vaccine detected. Please select the specific clinic brand/SKU for fulfillment."
                    elif kind in ("ParasiteControl", "Deworming", "Preventive") or (concept and (concept or "").strip().lower() in _KNOWN_PREVENTIVE_PRODUCT_NAMES):
                        remarks = "Preventive product (tick/flea/deworming). Please select the specific clinic product for fulfillment."
                    elif kind in ("Medicine", "Drug", "Supplement", "Nutrition"):
                        remarks = "Generic product detected. Please select the specific clinic brand/SKU for fulfillment."
                    elif kind:
                        remarks = f"Mentioned as {kind.lower()}; no matching inventory/service SKU found. Vet must select specific SKU for billing."
                    else:
                        remarks = "Item detected but no matching SKU found. Vet must select specific SKU for billing."
                
                # Build unlinked item with structure matching billing modules
                unlinked_item = {
                    "item_name": concept or "",
                    "kind": kind or "",
                    "intent_context": intent_context or "",
                    "intent": f"{intent_context or assertion_id or 'Unknown'} ({kind or 'Unknown'})",
                    "status": status or "ACTION REQUIRED",
                    "reason": reason or "MISSING_LOCAL_ID",
                    "remarks": remarks,
                    "suggestions": suggestions,  # NEW: Include suggestions array
                }
                
                # ENRICHMENT: Add billing module fields based on kind (shadow structure)
                if kind in ("Medicine", "Drug", "Medication", "Substance", "Vaccine", "Supplement", "Nutrition"):
                    # Medication structure (matches Module 2)
                    unlinked_item.update({
                        "inventory_id": None,  # Will be filled when vet selects SKU
                        "administered_prescribed": "Prescribed" if intent_context == "Prescribed" else ("Administered" if intent_context == "Administered" else ""),
                        "fulfillment": "Internal",  # Default
                        "quantity": attributes.get("quantity") or attributes.get("has_total_quantity") or "",
                        "dosage": attributes.get("dose") or attributes.get("has_dose") or "",
                        "duration": attributes.get("duration") or attributes.get("has_duration") or "",
                        "frequency": _normalize_frequency(attributes.get("frequency") or attributes.get("has_frequency") or ""),
                        "instructions": _normalize_instructions(attributes.get("instructions") or attributes.get("remarks") or attributes.get("has_remarks") or ""),
                        "route": attributes.get("route") or attributes.get("has_route") or "",
                    })
                elif kind in ("Procedure", "Service", "Treatment"):
                    # Service structure (matches Module 1)
                    unlinked_item.update({
                        "service_id": None,  # Will be filled when vet selects SKU
                    })
                elif kind in ("LabTest", "DiagnosticTest", "Diagnostic"):
                    # Diagnostic structure (matches Module 3)
                    # Determine Lab vs Radiology from concept
                    is_radiology = any(x in (concept or "").lower() for x in ["x-ray", "xray", "radiology", "imaging", "ultrasound", "ct", "mri", "radiograph"])
                    unlinked_item.update({
                        "service_id": None,  # Will be filled when vet selects SKU
                        "lab_radiology": "Radiology" if is_radiology else "LAB",
                        "test_name": concept or "",
                        "date_of_test": attributes.get("due_on") or attributes.get("has_due_on") or "TBD",
                        "done": "In-house" if intent_context == "Performed" else "External",
                        "sample_collected": "Yes" if intent_context == "Performed" else "No",
                    })
                
                dashboard["module_4_unlinked_entities"].append(unlinked_item)
                continue
        
        # RULE 5: Module 1 - Professional Services (Service ID from service_master → Module 1)
        # CRITICAL: Use ID source (has_service_id) as primary signal - Service IDs come from service_master
        # This ensures medications (which have stock_id) never appear in Procedures
        if has_service_id and kind in ("Procedure", "Service", "Treatment") and assertion_id == "CONF" and intent_context == "Performed" and section == "Plan":
            # Query Service Master to validate ID and get canonical name
            service_record = get_service_master_record(local_service_id, conn=conn, logger=logger)
            canonical_name = concept or ""
            if service_record:
                canonical_name = service_record.get("procedure_name") or concept or ""
            
            # remarks: clinical context for UX (e.g. "to measure Norberg angle") — UI should display so vet knows why ordered
            service_item = {
                "item_name": canonical_name or "",
                "service_id": local_service_id or None,
                "status": "DRAFT",  # System default: all items start as DRAFT until vet sign-off
                "remarks": attributes.get("technique") or attributes.get("procedure_name") or (service_record.get("remarks") if service_record else "") or atom.get("source_text", "") or "",
            }
            dashboard["procedures_services"].append(service_item)
            continue
        
        # RULE 6: Medications - Administered (Stock ID from inventory_master → Module 2a)
        # CRITICAL: Use ID source (has_stock_id) as primary signal - Inventory IDs come from inventory_master
        # Intent override: Administered = given in clinic today → "Medications Administered" sub-module
        if has_stock_id and kind in ("Medicine", "Drug", "Medication", "Substance", "Vaccine", "Supplement", "Nutrition") and assertion_id == "CONF" and intent_context == "Administered":
            inventory_record = get_inventory_master_record(local_stock_id, conn=conn, logger=logger)
            if inventory_record:
                canonical_name = inventory_record.get("trade_name") or inventory_record.get("item_name") or concept
                if is_retail_or_diet_product(inventory_record):
                    retail_item = {
                        "item_name": canonical_name or "",
                        "inventory_id": local_stock_id or None,
                        "status": "DRAFT",
                        "quantity": attributes.get("quantity") or attributes.get("has_total_quantity") or "",
                        "unit_price": None,
                        "remarks": attributes.get("remarks") or inventory_record.get("remarks") or atom.get("source_text", "") or "",
                        "suggestions": [],
                    }
                    dashboard["other_products_retail"].append(retail_item)
                else:
                    medication_item = {
                        "item_name": canonical_name or "",
                        "inventory_id": local_stock_id or None,
                        "status": "DRAFT",
                        "administered_prescribed": "Administered",
                        "dosage": attributes.get("dose") or attributes.get("has_dose") or "",
                        "route": attributes.get("route") or attributes.get("has_route") or "",
                        "remarks": attributes.get("remarks") or inventory_record.get("remarks") or atom.get("source_text", "") or "",
                    }
                    dashboard["medications_administered"].append(medication_item)
            else:
                # ID not found in Inventory Master - route to Unlinked (safety gate)
                if logger:
                    logger.warning(f"Inventory ID {local_stock_id} not found in inventory - routing to unlinked")
                unlinked_item = {
                    "item_name": concept,
                    "kind": kind or "Medicine",
                    "intent_context": intent_context or "",
                    "intent": f"{intent_context or assertion_id} ({kind})",
                    "status": "ACTION REQUIRED",
                    "reason": "INVALID_INVENTORY_ID",
                    "remarks": f"Inventory ID {local_stock_id} not found in Inventory Master. Please verify or select correct SKU.",
                    "suggestions": [],
                }
                dashboard["module_4_unlinked_entities"].append(unlinked_item)
            continue

        # RULE 7: Medications - Prescribed (Stock ID from inventory_master → Module 2b)
        # CRITICAL: Use ID source (has_stock_id) as primary signal - Inventory IDs come from inventory_master
        # Intent override: Prescribed = for owner to take home → "Medications Prescribed" sub-module
        # Diet & Nutrition / Non-Medical inventory → Other Products & Retail (not Pharmacy)
        if has_stock_id and kind in ("Medicine", "Drug", "Medication", "Substance", "Vaccine", "Supplement", "Nutrition") and assertion_id == "CONF" and intent_context == "Prescribed":
            inventory_record = get_inventory_master_record(local_stock_id, conn=conn, logger=logger)
            if not inventory_record:
                # ID not found in Inventory Master - route to Unlinked (safety gate)
                if logger:
                    logger.warning(f"Inventory ID {local_stock_id} not found in inventory - routing to unlinked")
                unlinked_item = {
                    "item_name": concept,
                    "kind": kind or "Medicine",
                    "intent_context": intent_context or "",
                    "intent": f"{intent_context or assertion_id} ({kind})",
                    "status": "ACTION REQUIRED",
                    "reason": "INVALID_INVENTORY_ID",
                    "remarks": f"Inventory ID {local_stock_id} not found in Inventory Master. Please verify or select correct SKU.",
                    "suggestions": [],
                }
                dashboard["module_4_unlinked_entities"].append(unlinked_item)
                continue

            canonical_name = inventory_record.get("trade_name") or inventory_record.get("item_name") or concept
            if is_retail_or_diet_product(inventory_record):
                retail_item = {
                    "item_name": canonical_name,
                    "inventory_id": local_stock_id,
                    "status": "DRAFT",
                    "quantity": attributes.get("quantity") or attributes.get("has_total_quantity") or "",
                    "unit_price": None,
                    "remarks": inventory_record.get("remarks") or atom.get("source_text", ""),
                    "suggestions": [],
                }
                dashboard["other_products_retail"].append(retail_item)
                continue

            form = inventory_record.get("form", "").lower()
            unit = inventory_record.get("unit", "").lower()
            quantity_hint = ""
            if "tablet" in form or "tab" in form or "capsule" in form or "cap" in form:
                quantity_hint = "[Vet to Input: Number of tablets/capsules]"
            elif "liquid" in form or "ml" in unit or "l" in unit:
                quantity_hint = "[Vet to Input: Volume in ml]"
            else:
                quantity_hint = "[Vet to Input: Quantity]"
            frequency = _normalize_frequency(attributes.get("frequency") or attributes.get("has_frequency") or "")
            instructions = _normalize_instructions(attributes.get("instructions") or attributes.get("remarks") or attributes.get("has_remarks") or "")
            medication_item = {
                "item_name": canonical_name,
                "inventory_id": local_stock_id,
                "status": "DRAFT",
                "administered_prescribed": "Prescribed",
                "fulfillment": "Internal",
                "quantity": attributes.get("quantity") or attributes.get("has_total_quantity") or quantity_hint,
                "dosage": attributes.get("dose") or attributes.get("has_dose") or "",
                "duration": attributes.get("duration") or attributes.get("has_duration") or "",
                "frequency": frequency,
                "instructions": instructions,
                "remarks": inventory_record.get("remarks") or atom.get("source_text", ""),
            }
            dashboard["medications_prescribed"].append(medication_item)
            continue
        
        
        # FALLBACK: If item has local_id but doesn't match above rules, check if it's a scheduled service (goes to reminders)
        # This handles items with IDs that are scheduled for future (not performed today)
        if has_local_id and section == "Plan" and assertion_id == "CONF" and intent_context != "Performed":
            # Scheduled service/medication goes to reminders (uses ID to pre-populate scheduler)
            # Query appropriate master to get canonical name
            canonical_name = concept
            if has_service_id:
                service_record = get_service_master_record(local_service_id, conn=conn, logger=logger)
                if service_record:
                    canonical_name = service_record.get("procedure_name") or concept
            elif has_stock_id:
                inventory_record = get_inventory_master_record(local_stock_id, conn=conn, logger=logger)
                if inventory_record:
                    canonical_name = inventory_record.get("trade_name") or inventory_record.get("item_name") or concept
            
            reminder_item = {
                "item_name": canonical_name or "",  # From appropriate Master
                "item_id": local_service_id or local_stock_id or f"[{kind.upper()}-XXX]" if kind else "[UNKNOWN-XXX]",  # ID source determines which master to query
                "due_date": "ASAP",
                "remarks": f"Scheduled {kind.lower()} from Plan section" if kind else "Scheduled item from Plan section",
                "intent_context": "Scheduled",
                "kind": kind or "",
            }
            dashboard["reminders_follow_ups"].append(reminder_item)
            continue

    # De-duplicate procedures_services by service_id (one line per service per visit for clean invoice)
    procedures = dashboard.get("procedures_services", [])
    if procedures:
        by_sid: Dict[Any, List[Dict[str, Any]]] = {}
        for item in procedures:
            sid = item.get("service_id")
            if sid not in by_sid:
                by_sid[sid] = []
            by_sid[sid].append(item)
        merged_procedures = []
        for _sid, items in by_sid.items():
            first = items[0]
            if len(items) > 1:
                remarks = " | ".join((i.get("remarks") or "").strip() for i in items if (i.get("remarks") or "").strip())
                if not remarks:
                    remarks = first.get("remarks") or ""
                merged_procedures.append({**first, "remarks": remarks})
            else:
                merged_procedures.append(first)
        dashboard["procedures_services"] = merged_procedures

    # De-duplicate diagnostics by service_id (same as procedures)
    diagnostics = dashboard.get("diagnostics", [])
    if diagnostics:
        by_sid_d: Dict[Any, List[Dict[str, Any]]] = {}
        for item in diagnostics:
            sid = item.get("service_id")
            if sid not in by_sid_d:
                by_sid_d[sid] = []
            by_sid_d[sid].append(item)
        merged_diagnostics = []
        for _sid, items in by_sid_d.items():
            first = items[0]
            if len(items) > 1:
                remarks = " | ".join((i.get("remarks") or "").strip() for i in items if (i.get("remarks") or "").strip())
                if not remarks:
                    remarks = first.get("remarks") or ""
                merged_diagnostics.append({**first, "remarks": remarks})
            else:
                merged_diagnostics.append(first)
        dashboard["diagnostics"] = merged_diagnostics

    # ID-first de-dup for medications by inventory_id (Administered / Prescribed separately)
    for mod_name in ("medications_administered", "medications_prescribed"):
        meds = dashboard.get(mod_name, [])
        if not meds:
            continue
        by_iid: Dict[Any, List[Dict[str, Any]]] = {}
        for item in meds:
            iid = item.get("inventory_id")
            if iid not in by_iid:
                by_iid[iid] = []
            by_iid[iid].append(item)
        merged_meds: List[Dict[str, Any]] = []
        for _iid, items in by_iid.items():
            first = items[0]
            if len(items) > 1:
                remarks = " | ".join((i.get("remarks") or "").strip() for i in items if (i.get("remarks") or "").strip())
                merged = {**first}
                if remarks:
                    merged["remarks"] = remarks
                # Preserve strongest non-empty medication fields from duplicates.
                for field in ("dosage", "route", "duration", "frequency", "instructions", "quantity"):
                    if not merged.get(field):
                        for i in items[1:]:
                            if i.get(field):
                                merged[field] = i.get(field)
                                break
                merged_meds.append(merged)
            else:
                merged_meds.append(first)
        dashboard[mod_name] = merged_meds

    # Ensure diagnostics module is populated from entity manifest when Phase 2 did not emit an atom (e.g. X-ray with local_service_id 1185)
    _DIAGNOSTIC_SERVICE_IDS_MANIFEST = {1185}  # X-ray / imaging services that must appear in diagnostics
    existing_diag_sids = {int(d.get("service_id")) for d in dashboard.get("diagnostics", []) if d.get("service_id") is not None}
    for e in (entity_manifest or []):
        if not isinstance(e, dict):
            continue
        try:
            sid = e.get("local_service_id")
            if sid is None:
                continue
            sid_int = int(sid)
        except (TypeError, ValueError):
            continue
        if sid_int not in _DIAGNOSTIC_SERVICE_IDS_MANIFEST or sid_int in existing_diag_sids:
            continue
        service_record = get_service_master_record(sid, conn=conn, logger=logger)
        if not service_record:
            continue
        canonical_name = service_record.get("procedure_name") or (e.get("kb_preferred_name") or e.get("display_name") or e.get("span_text") or "")
        category_from_master = (service_record.get("category") or "").lower()
        is_radiology = (
            any(x in (canonical_name or "").lower() for x in ["x-ray", "xray", "radiology", "imaging", "ultrasound", "ct", "mri", "radiograph"])
            or "radiology" in category_from_master
            or "imaging" in category_from_master
        )
        category = "Radiology" if is_radiology else "LAB"
        diagnostic_item = {
            "service_id": sid,
            "lab_radiology": category or "LAB",
            "test_name": canonical_name or "",
            "date_of_test": "TBD",
            "done": "In-house",
            "sample_collected": "No",
            "remarks": "",
        }
        dashboard["diagnostics"].append(diagnostic_item)
        existing_diag_sids.add(sid_int)
        if logger:
            logger.info("  Injected diagnostic from manifest: service_id=%s (%s)", sid, canonical_name or "")

    # Clinical Resolution & Deduplication: summarize Module 5 (clinical history) before render
    raw_history = dashboard.get("module_5_clinical_history", [])
    if raw_history:
        effective_pet_name = pet_name
        if not effective_pet_name and signalment and isinstance(signalment, dict):
            effective_pet_name = (
                signalment.get("pet_name")
                or signalment.get("patient_name")
                or signalment.get("Patient Name")
                or signalment.get("patient")
                or ""
            )
        dashboard["module_5_clinical_history"] = summarize_clinical_history(
            raw_history, pet_name=effective_pet_name, logger=logger
        )

    # Category-First Streamlined Dashboard: group by functional role (signalment_vitals, clinical_assessment, pharmacy, diagnostics_procedures, preventive, action_plan) with status CONFIRMED/DRAFTED/ACTION_REQUIRED
    streamlined = generate_streamlined_dashboard(
        dashboard, entity_manifest=entity_manifest, signalment=signalment, logger=logger
    )

    # Dedupe module_4_unlinked_entities by (item_name normalized, kind) so vet sees each actionable item once
    m4 = dashboard.get("module_4_unlinked_entities", [])
    _seen_m4 = set()
    _deduped_m4 = []
    _KNOWN_TYPOS_MODULE4 = {"spircoxin": "Spirocoxin", "spirocoxin": "Spirocoxin"}  # Phase 2 / ASR spelling fixes
    for _item in m4:
        name = str(_item.get("item_name") or "").strip()
        name_lower = name.lower()
        if name_lower in _KNOWN_TYPOS_MODULE4:
            _item = {**_item, "item_name": _KNOWN_TYPOS_MODULE4[name_lower]}
            name = _item["item_name"]
        _key = (name_lower, str(_item.get("kind") or "").strip().lower())
        if _key in _seen_m4:
            continue
        _seen_m4.add(_key)
        _deduped_m4.append(_item)
    dashboard["module_4_unlinked_entities"] = _deduped_m4

    # Populate diagnostics from unlinked Module 4 items that are diagnostic/imaging so the vet sees them in one place.
    # Known preventive products (e.g. Bravecto) are NOT diagnostics; try to link from service_master by name when present.
    _DIAG_LIKE_KINDS = ("LabTest", "DiagnosticTest", "Diagnostic")
    _IMAGING_CUES = ("x-ray", "xray", "radiology", "imaging", "ultrasound", "norberg", "radiograph", "ct", "mri", "lab test")
    for item in _deduped_m4:
        kind = (item.get("kind") or "").strip()
        name_lower = (item.get("item_name") or "").strip().lower()
        is_diag_kind = kind in _DIAG_LIKE_KINDS
        is_imaging_like = any(cue in name_lower for cue in _IMAGING_CUES)
        # Skip known preventive/flea-tick products (e.g. Bravecto): they belong in inventory/preventive, not diagnostics
        if name_lower in _KNOWN_PREVENTIVE_PRODUCT_NAMES or any(p in name_lower for p in _KNOWN_PREVENTIVE_PRODUCT_NAMES):
            # Try to link from service_master so it shows as linked when clinic has it
            resolved_sid = lookup_service_id_by_name(item.get("item_name") or "", conn=conn, logger=logger)
            if resolved_sid:
                dashboard["procedures_services"].append({
                    "item_name": item.get("item_name") or "",
                    "service_id": resolved_sid,
                    "status": "DRAFT",
                    "remarks": (item.get("remarks") or "").strip() or "Preventive product (e.g. tick/flea) – linked from clinic inventory.",
                })
            else:
                dashboard["reminders_follow_ups"].append({
                    "item_name": item.get("item_name") or "",
                    "item_id": "[PREVENTIVE-XXX]",
                    "due_date": "ASAP",
                    "remarks": (item.get("remarks") or "").strip() or "Recommend preventive product; add to inventory if not already linked.",
                    "intent_context": "Recommended",
                    "kind": "Preventive",
                })
            continue
        if not (is_diag_kind or is_imaging_like):
            continue
        # Avoid duplicate: only add if not already in diagnostics (by test_name)
        existing_names = {str(d.get("test_name") or "").strip().lower() for d in dashboard.get("diagnostics", [])}
        if (item.get("item_name") or "").strip().lower() in existing_names:
            continue
        is_radiology = any(x in name_lower for x in ["x-ray", "xray", "radiology", "imaging", "ultrasound", "ct", "mri", "radiograph", "norberg"])
        dashboard["diagnostics"].append({
            "service_id": item.get("service_id"),
            "lab_radiology": "Radiology" if is_radiology else "LAB",
            "test_name": item.get("item_name") or "",
            "date_of_test": item.get("date_of_test", "TBD"),
            "done": item.get("done", "External"),
            "sample_collected": item.get("sample_collected", "No"),
            "remarks": (item.get("remarks") or "") + " [Unlinked – ACTION REQUIRED: select service for billing]",
        })

    # Ensure all modules are present (even if empty) and return with strict format (legacy + streamlined)
    return {
        "procedures_services": dashboard.get("procedures_services", []),
        "medications_administered": dashboard.get("medications_administered", []),
        "medications_prescribed": dashboard.get("medications_prescribed", []),
        "diagnostics": dashboard.get("diagnostics", []),
        "other_products_retail": dashboard.get("other_products_retail", []),
        "reminders_follow_ups": dashboard.get("reminders_follow_ups", []),
        "vitals": dashboard.get("vitals", []),
        "module_4_unlinked_entities": dashboard.get("module_4_unlinked_entities", []),
        "module_5_clinical_history": dashboard.get("module_5_clinical_history", []),
        "streamlined_dashboard": streamlined,
    }


def build_phase2_prompt_with_grounding(
    base_prompt: str,
    session_id: Optional[str] = None,
    section_name: str = "FullNote",
    entity_manifest_json: Optional[str] = None,
) -> str:
    """
    Add entity manifest context to the base prompt for grounding.
    Minimal implementation: append manifest JSON if provided.
    """
    if not entity_manifest_json or entity_manifest_json.strip() == "[]":
        return base_prompt

    manifest_block = f"""
[ENTITY_MANIFEST_CONTEXT]
The following entities were extracted and linked from the transcript:
{entity_manifest_json}

Grounding-aware extraction rules (MANDATORY):
1) Treat manifest IDs as source-of-truth for identity persistence.
2) If an extracted atom corresponds to a manifest entity, include:
   - referenced_entity_id: manifest entity_id
3) For the same clinical action phrased differently, emit a stable:
   - dedup_key (e.g., "norberg_angle", "thoracic_xray")
4) If the LLM kind conflicts with a matched manifest entity that has local_service_id/local_stock_id,
   keep the atom but inherit manifest identity/IDs and prefer manifest kind metadata.
5) Use both raw mention (span_text) and normalized_name in the manifest to resolve ASR variants.
"""
    return base_prompt + manifest_block


def parse_soap_sections(soap_note_text: str) -> Dict[str, str]:
    """
    Parse SOAP note into sections (Subjective, Objective, Assessment, Plan).
    Minimal regex-based implementation for compatibility.
    """
    sections = {
        "Subjective": "",
        "Objective": "",
        "Assessment": "",
        "Plan": "",
    }

    patterns = {
        "Subjective": r"(?i)^\s*(?:S|SUBJECTIVE)[:\s]+\s*(.*?)(?=\n\s*(?:O|OBJECTIVE|A|ASSESSMENT|P|PLAN)[:\s]|$)",
        "Objective": r"(?i)^\s*(?:O|OBJECTIVE)[:\s]+\s*(.*?)(?=\n\s*(?:A|ASSESSMENT|P|PLAN)[:\s]|$)",
        "Assessment": r"(?i)^\s*(?:A|ASSESSMENT)[:\s]+\s*(.*?)(?=\n\s*(?:P|PLAN)[:\s]|$)",
        "Plan": r"(?i)^\s*(?:P|PLAN)[:\s]+\s*(.*?)$",
    }

    for section, pattern in patterns.items():
        match = re.search(pattern, soap_note_text or "", re.DOTALL | re.MULTILINE)
        if match:
            sections[section] = match.group(1).strip()

    return sections


def enrich_atoms_with_manifest_bindings(
    atoms_list: List[Dict[str, Any]],
    entity_manifest_json: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Enrich knowledge atoms with manifest binding info (binding_level, binding_track).
    Minimal implementation: returns atoms as-is (no enrichment needed in minimal mode).
    """
    return atoms_list

def _sha256(text: str) -> str:
    """Simple SHA256 hash function."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()




# Session cache stubs (not used in minimal mode, but imported by kb_phase2_integration.py)
def ensure_phase2_session_tables(conn=None, logger: Optional[logging.Logger] = None) -> bool:
    """Stub: session cache not used in minimal mode."""
    return False


def ensure_session(session_id: str, conn=None, logger: Optional[logging.Logger] = None) -> Optional[str]:
    """Stub: session cache not used in minimal mode."""
    return session_id


def is_section_dirty(session_id: str, section_name: str, conn=None, logger: Optional[logging.Logger] = None) -> bool:
    """Stub: session cache not used in minimal mode."""
    return False


def mark_section_dirty(session_id: str, section_name: str, conn=None, logger: Optional[logging.Logger] = None) -> None:
    """Stub: session cache not used in minimal mode."""
    pass


def upsert_section_atoms(session_id: str, section_name: str, atoms: List[Dict[str, Any]], conn=None, logger: Optional[logging.Logger] = None) -> None:
    """Stub: session cache not used in minimal mode."""
    pass


def load_cached_atoms(session_id: str, section_name: str, conn=None, logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """Stub: session cache not used in minimal mode."""
    return []


# Phase 3 stubs (removed, but imported by kb_phase2_integration.py)
def intent_filter(atoms: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Stub: Phase 3 intent filtering removed."""
    return {
        "billable_candidates": atoms or [],
        "prescriptions": [],
        "clinical_log_only": [],
        "dropped_hist": [],
    }


class Phase2KnowledgeAtomMatcher:
    """Stub: Phase 3 matcher removed."""
    pass

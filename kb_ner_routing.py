"""
Entity routing and classification utilities.

This module handles:
- Entity route classification (skip_vitals, dual_sync, global_direct)
- Kind canonicalization
- Entity bucket determination
- Procedure role classification
- Diet subtype classification
- Context extraction
"""

import re
import json
import logging
from typing import Optional, Dict, Any, List

from kb_ner_clients import get_openai_client, get_client_for_model

# Production-ready 12-kind NER schema (single source of truth for NER + intent NER).
# Family mapping: PRODUCT → Medication, Preventive, ParasiteControl, Diet; ACTION → Procedure, Diagnostic, Reminder;
# CLINICAL → Symptom, Diagnosis, Anatomy; OTHER → VitalSign, ReasonForVisit.
PRODUCTION_NER_KINDS = [
    "ReasonForVisit",  # Primary trigger/chief complaint
    "Medication",      # Drugs; MANDATORY attribute: status [Administered | Prescribed]
    "Procedure",       # Clinical actions, surgeries, maneuvers (e.g. Ortolani)
    "Diagnostic",      # Ordered tests, imaging, lab panels
    "VitalSign",       # Metrics (Weight, Temp, HR); value and unit in intent field
    "Reminder",        # Follow-up appointments, re-checks
    "Symptom",         # Clinical signs, owner reports; attributes can include is_negated
    "Diagnosis",       # Suspected or confirmed conditions
    "Anatomy",         # Body sites (e.g. hip joint, left stifle)
    "Diet",            # Prescription or specialized food
    "Preventive",      # Vaccines, heartworm prevention, wellness preventive products
    "ParasiteControl", # Parasite-specific preventatives (Bravecto, flea/tick treatment, deworming)
]

# Local grounding kinds (legacy route name: dual_sync; behavior is local-only in executor).
# User-approved local set:
# ReasonForVisit, Medication, Procedure, Diagnostic, Diet (Subtype A), Preventive, ParasiteControl, Diagnosis.
DUAL_SYNC_BILLABLE_KINDS = [
    "ReasonForVisit",
    "Medication",
    "Procedure",
    "Diagnostic",
    "Diet",
    "Preventive",
    "ParasiteControl",
    "Diagnosis",
]

# Hard-skip kinds: never run local or global search; preserve in SOAP as text only (zero DB/embedding calls).
# Billing accuracy: do not try to "bill" for anatomy, signalment, or symptoms.
HARD_SKIP_KINDS = ["Anatomy", "Symptom"]

# Service-only kinds: Diagnosis and ReasonForVisit must NEVER search Pharmacy/Inventory (Pharmacy-Free Zone).
# They may only match Services/Procedures (e.g. lab test, treatment). Prevents "pus"→Lasix, "yeast"→AST KIT.
SERVICE_ONLY_KINDS = ["Diagnosis", "ReasonForVisit"]

# Minimum match score for Diagnosis/ReasonForVisit to allow any grounding (0.95 certainty wall).
# Below this, entity is preserved as note-only (no local_stock_id / local_service_id).
DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD = 0.95

def is_services_only_kind(kind: str) -> bool:
    """True if this kind must search Services only (no Pharmacy/Inventory)."""
    k = (kind or "").strip()
    return k in SERVICE_ONLY_KINDS

# Global-direct disabled for production local-only grounding mode.
GLOBAL_DIRECT_KINDS = []

# VitalSign is routed via is_fast_lane_vital() / skip_vitals / global_vitals (not in GLOBAL_DIRECT list)
SKIP_VITALS_KINDS = []  # Determined dynamically by is_fast_lane_vital() and is_clinical_status_constant()

# Single source of truth: 12 kinds for prompts and validation. Routing uses DUAL_SYNC_BILLABLE_KINDS + GLOBAL_DIRECT_KINDS + VitalSign.
ALL_NER_KINDS = sorted(set(PRODUCTION_NER_KINDS))

# Export for backward compatibility
dual_sync_billable_kinds = DUAL_SYNC_BILLABLE_KINDS
global_direct_kinds = GLOBAL_DIRECT_KINDS
skip_vitals_kinds = SKIP_VITALS_KINDS


def is_signalment_like_span(span_text: str) -> bool:
    """
    Detect demographic/signalment blobs (age/sex/breed/weight) that must NEVER be KB-linked.

    Prevents high-risk hallucinations like:
      "5 year old male Labrador" → "h. node" (hilar node)
    """
    if not span_text:
        return False
    s = span_text.lower().strip()

    has_age = bool(re.search(r"\b\d+(\.\d+)?\s*(year|yr|yrs|month|mo|mos)\b", s)) or "year old" in s or "yr old" in s
    has_sex = bool(re.search(r"\b(male|female|neutered|spayed|intact)\b", s))
    has_weight = bool(re.search(r"\b\d+(\.\d+)?\s*(kg|kgs|kilogram|lb|lbs|pound|pounds)\b", s))

    # Small, conservative breed token list for detecting combined blobs (not for mapping)
    breed_tokens = [
        "labrador",
        "golden retriever",
        "german shepherd",
        "beagle",
        "pug",
        "rottweiler",
        "shih tzu",
        "shih-tzu",
        "indie",
        "indian pariah",
    ]
    has_breed = any(bt in s for bt in breed_tokens)

    return bool((has_age and has_sex) or (has_age and has_breed) or (has_sex and has_breed) or (has_weight and (has_age or has_sex)))


def sanitize_asr_errors(mention: str) -> str:
    """
    Step A: Phonetic/Keyword Sanitization for ASR errors.
    
    Replaces common ASR errors with correct terms before routing.
    This prevents "mucosmoburn" from being searched in KB.
    
    Args:
        mention: Original mention text (may contain ASR errors)
        
    Returns:
        Sanitized mention text
    """
    # ASR error corrections (case-insensitive)
    asr_corrections = {
        "mucosmoburn": "mucous membranes",
        "mucosmoburn": "mucous membranes",  # Common ASR error
        "animal plant expression": "anal gland expression",  # Note: This would then move to Lane A for billing
        # Ortho common ASR errors (kept minimal; semantic resolver handles most cases)
        "noberg angle": "norberg angle",
        "peterlular luxation": "patellar luxation",
    }
    
    clean_mention = mention.lower()
    for error, correction in asr_corrections.items():
        if error in clean_mention:
            # Replace the error with correction (preserve case of surrounding text)
            mention = mention.replace(error, correction)
            mention = mention.replace(error.capitalize(), correction.capitalize())
            mention = mention.replace(error.upper(), correction.upper())
    
    return mention


def is_fast_lane_vital(mention: str, kind: str) -> bool:
    """
    Step B: The Bypass Logic for Fast Lane (Lane C).
    
    Hard-coded bypass list to prevent hallucinations like "Normoglycemia" or "Seizure Disorder".
    These entities bypass the Linker, LLM Judge, and ASR Correction logic entirely.
    
    Args:
        mention: Entity mention text
        kind: Entity kind from NER
        
    Returns:
        True if entity should bypass KB linking (Fast Lane)
    """
    # 1. Check if NER already called it a Vital_Sign
    if kind == "VitalSign" or kind == "Vital_Sign":
        return True
    
    # 2. Sanitize ASR errors first
    clean_mention = sanitize_asr_errors(mention).lower()
    
    # 3. Hard-coded vital keywords (comprehensive list)
    # IMPORTANT: avoid ultra-broad tokens (e.g. "t") that would match almost any text.
    VITAL_PHRASES = [
        # Thermal & Mass
        "temp", "temperature", "rectal temp",
        "weight", "mass",
        # Cardiovascular & Respiratory
        "heart rate", "pulse", "beats per minute",
        "respiration", "respiratory rate", "breathing rate", "breaths per minute",
        # Oral & Perfusion (The "Hallucination" Shield)
        "mucous membrane", "mucous membranes", "mucosa", "oral mucosa", "gums",
        "capillary refill", "capillary refill time", "refill time",
        # Hydration & Condition Scores
        "hydration", "hydration status", "skin tent", "dehydrated", "well hydrated", "hydration level",
        "body condition", "body condition score",
        "pain score", "pain level",
    ]
    
    # Phrase match (safe)
    if any(phrase in clean_mention for phrase in VITAL_PHRASES):
        return True

    # Short-token match with word boundaries (prevents false positives like "medications" matching "t")
    # NOTE: "t" is ONLY considered temperature shorthand when it appears as a standalone token with a numeric value.
    short_token_pattern = r"\b(wt|hr|rr|bpm|rpm|mm|crt|bcs|kg|kgs|lbs)\b"
    if re.search(short_token_pattern, clean_mention, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(t)\b\s*[:=]?\s*\d", clean_mention, flags=re.IGNORECASE):
        return True
    
    return False


def is_clinical_status_constant(mention: str) -> bool:
    """
    Checks if a mention describes a general clinical status 
    that should bypass KB linking to prevent hallucinations.
    
    These are "Status Constants" that should be preserved verbatim,
    never medicalized or linked to KB concepts.
    
    Args:
        mention: Entity mention text
        
    Returns:
        True if entity is a clinical status constant (should bypass KB)
    """
    # Status phrases (case-insensitive, with word boundaries for acronyms)
    STATUS_PHRASES = [
        # Physical Status
        "physically fit", "healthy", "active", "alert", "bright",
        "bar", "bright alert responsive",
        # Normalcy Tags
        "otherwise normal", "everything is normal", "wnl", "within normal limits",
        "no abnormalities", "normal",
        # Negative Findings
        "nsf", "no significant findings", "nothing noted", "non-remarkable",
        "unremarkable", "no findings",
    ]
    
    clean_mention = mention.lower().strip()
    
    # Check for direct phrase matches
    if any(phrase in clean_mention for phrase in STATUS_PHRASES):
        return True
    
    # Check for acronyms with word boundaries (to avoid partial matches)
    acronyms = ["bar", "wnl", "nsf"]
    for acronym in acronyms:
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(acronym) + r'\b'
        if re.search(pattern, clean_mention, re.IGNORECASE):
            return True
    
    return False


# Cache for canonicalize_kind to avoid repeated lookups
_canonicalize_kind_cache: Dict[str, str] = {}

def canonicalize_kind(ner_kind: str) -> str:
    """
    DETERMINISTIC Canonical Kind Normalizer.
    
    Maps all NER kind variations to stable canonical forms.
    This MUST run before routing, local search, global search, and truth injector.
    
    CRITICAL: This fixes the brittleness from kind taxonomy mismatches:
    - "Drug/Substance" → "Drug"
    - "Reason_for_Visit" → "ReasonForVisit"
    - "Vital Sign" → "VitalSign"
    
    Args:
        ner_kind: NER-extracted kind (may be compound like "Drug/Substance")
        
    Returns:
        Canonicalized kind string (stable enum)
    """
    if not ner_kind:
        return "Other"
    
    # Check cache first
    if ner_kind in _canonicalize_kind_cache:
        return _canonicalize_kind_cache[ner_kind]
    
    # Normalize: lowercase, replace hyphens with underscores, collapse spaces
    k = ner_kind.strip().lower()
    k = k.replace("-", "_")
    k = k.replace("__", "_")
    k = re.sub(r"\s+", " ", k)  # Collapse multiple spaces
    
    # Direct canonical mapping: 11 production kinds + legacy aliases + pipeline family
    CANON = {
        # VitalSign
        "vitalsign": "VitalSign",
        "vital_sign": "VitalSign",
        "vital sign": "VitalSign",
        "vitals": "VitalSign",
        # ReasonForVisit
        "reasonforvisit": "ReasonForVisit",
        "reason_for_visit": "ReasonForVisit",
        "reason for visit": "ReasonForVisit",
        "reason": "ReasonForVisit",
        # Medication (11-kind production)
        "medication": "Medication",
        "drug": "Medication",
        "medicine": "Medication",
        "substance": "Medication",
        "drug/substance": "Medication",
        "vaccine": "Preventive",
        "vaccination": "Preventive",
        "heartworm": "Preventive",
        "heartworm prevention": "Preventive",
        "preventive": "Preventive",
        "preventative": "Preventive",
        # Procedure
        "procedure": "Procedure",
        "service": "Procedure",
        "treatment": "Procedure",
        "therapy": "Procedure",
        "physiotherapy": "Procedure",
        "physical_therapy": "Procedure",
        "physical therapy": "Procedure",
        # Diagnostic (11-kind production)
        "diagnostic": "Diagnostic",
        "diagnostictest": "Diagnostic",
        "labtest": "Diagnostic",
        # Reminder
        "reminder": "Reminder",
        "followup": "Reminder",
        "follow_up": "Reminder",
        # Symptom
        "symptom": "Symptom",
        "finding": "Symptom",
        "observation": "Symptom",
        # Diagnosis (11-kind production; suspected/confirmed conditions)
        "diagnosis": "Diagnosis",
        "condition": "Diagnosis",
        "disease": "Diagnosis",
        "illness": "Diagnosis",
        # Anatomy
        "anatomy": "Anatomy",
        # Diet
        "diet": "Diet",
        "nutrition": "Diet",
        # Preventive
        "preventive": "Preventive",
        "preventative": "Preventive",
        # ParasiteControl
        "parasitecontrol": "ParasiteControl",
        "parasite_control": "ParasiteControl",
        # Other / legacy
        "device": "Procedure",
        "other": "Other",
        # Pipeline family → schema kind for routing
        "product": "Medication",
        "clinical": "Diagnosis",
        "action": "Procedure",
        "exercise_restriction": "Other",
    }
    
    # Direct lookup first
    if k in CANON:
        result = CANON[k]
        # Cache the result
        _canonicalize_kind_cache[ner_kind] = result
        return result
    
    # Token-based fallback (handles compound kinds like "Drug/Substance", "Drug / Substance", etc.)
    tokens = re.split(r"[\/_\s]+", k)
    tokens = [t.strip() for t in tokens if t.strip()]
    
    # Check for vital signs
    if "vital" in tokens and "sign" in tokens:
        result = "VitalSign"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    
    # Check for medication/drug
    if any(t in ["drug", "medicine", "medication", "substance"] for t in tokens):
        result = "Medication"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    # Check for procedure/service (incl. physiotherapy, physical therapy)
    if any(t in ["procedure", "service", "treatment", "therapy", "physiotherapy", "physio"] for t in tokens):
        result = "Procedure"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    # Check for diagnostic/test
    if any(t in ["diagnostic", "diagnostictest", "labtest", "test"] for t in tokens):
        result = "Diagnostic"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    # Check for diagnosis/disease (suspected/confirmed conditions)
    if any(t in ["diagnosis", "disease", "illness", "condition"] for t in tokens):
        result = "Diagnosis"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    # Check for reason/visit
    if "reason" in tokens and "visit" in tokens:
        result = "ReasonForVisit"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    if "reason" in tokens:
        result = "ReasonForVisit"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    # Check for diet/nutrition
    if any(t in ["diet", "nutrition"] for t in tokens):
        result = "Diet"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    # Check for reminder/follow-up
    if any(t in ["reminder", "followup", "follow_up"] for t in tokens):
        result = "Reminder"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    # Check for preventive (vaccines, heartworm prevention)
    if any(t in ["vaccine", "vaccination", "heartworm", "preventive", "preventative"] for t in tokens):
        result = "Preventive"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    # Check for parasite control
    if any(t in ["parasite", "parasitecontrol"] for t in tokens):
        result = "ParasiteControl"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    
    # Return first token capitalized if we have tokens, otherwise "Other"
    if tokens:
        result = tokens[0].capitalize() if tokens[0] else "Other"
        _canonicalize_kind_cache[ner_kind] = result
        return result
    
    result = "Other"
    _canonicalize_kind_cache[ner_kind] = result
    return result


def classify_entity_route(entity_kind: str, entity: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None) -> str:
    """
    DETERMINISTIC Router with Attribute-Based + Lexical Cue Routing.
    
    Routing is driven by the KIND ARRAY (schema kinds): DUAL_SYNC_BILLABLE_KINDS, GLOBAL_DIRECT_KINDS,
    skip_vitals, etc. It is NOT based on free-form labels like "product" or "procedure". The combined
    pass may emit pipeline "family" (OTHER, CLINICAL, PRODUCT, PROCEDURE) in the kind field; those
    are mapped in canonicalize_kind() to schema kinds (Drug, Condition, Procedure, Other) so they
    participate in the same array-based routing.
    
    Routes entities to the appropriate search strategy based on:
    1. KB kind (from NER kb_kind field, or canonicalized from kind / family)
    2. Attributes (deterministic structure-based routing)
    3. Lexical cues (procedure keywords like "expression", "exam")
    
    CRITICAL: Uses kb_kind from NER (deterministic KB schema kind), else canonicalize kind.
    Attribute-first routing (deterministic, no LLM). Lexical overrides for procedure keywords.
    
    Args:
        entity_kind: The NER-extracted kind (may be kb_kind, schema kind, or pipeline family)
        entity: Optional entity dict with attributes, span_text, kb_kind, family
        
    Returns:
        "skip_vitals", "global_vitals", "skip_signalment", "skip_identity", "skip_non_billable", "skip_other", "dual_sync", or "global_direct"
    """
    # Prefer schema kind: kb_kind first, then canonicalize kind (maps family PRODUCT/CLINICAL/PROCEDURE → Drug/Condition/Procedure)
    if entity and entity.get("kb_kind"):
        kind = canonicalize_kind(entity.get("kb_kind"))
    else:
        kind = canonicalize_kind(entity_kind)
    
    # Get attributes and text for attribute-based routing
    attrs = (entity.get("attributes") or {}) if entity else {}
    text = ((entity.get("span_text") or "").lower()) if entity else ""

    # Step -1: Generic "meta" category mentions should never be KB-linked (avoid nonsense bindings).
    # Examples: "medications", "treatment", "therapy" (non-specific umbrella terms).
    if entity:
        raw_span_lower = ((entity.get("span_text") or "")).strip().lower()
        GENERIC_META_SPANS = {
            "medication", "medications", "medicine", "medicines", "meds",
            "treatment", "treatments",
            "therapy", "therapies",
        }
        if raw_span_lower in GENERIC_META_SPANS:
            return "skip_vitals"

    # Step 0: Signalment/demographic blobs should never be KB-linked.
    # This is a hard safety gate to prevent breed→anatomy collisions.
    if entity:
        raw_span = (entity.get("span_text") or "").strip()
        if raw_span and is_signalment_like_span(raw_span):
            return "skip_signalment"
    
    # CRITICAL FIX: Check billable kinds FIRST before Fast Lane checks
    # This prevents Drugs (like "Cortex capsule", "Nutrish tablet") from being incorrectly routed to skip_vitals
    # Drug-like: Check for dosage, frequency, route, form, duration
    if any(k in attrs for k in ("dosage", "dose", "frequency", "route", "form", "duration", "refills")):
        # Strict local-only mode: only route to local for approved kinds.
        if kind in DUAL_SYNC_BILLABLE_KINDS:
            return "dual_sync"
    
    # Route 1: Fast Lane (Bypass Lane) - Comprehensive keyword/regex library
    # CRITICAL: These must NEVER be sent to KB Linker to prevent hallucinations like "Normoglycemia" or "Seizure Disorder"
    # This is the "Permanent Fix" - zero hallucinations by physically blocking the KB search path
    # BUT: Only apply Fast Lane checks if the kind is NOT a billable kind (Drug, Procedure, etc.)
    
    # Step A: Check if NER already called it a VitalSign
    if kind == "VitalSign" or kind == "Vital_Sign":
        # Guard: NER sometimes mislabels product/supplement names as VitalSign.
        # If it looks like a branded/product-like span, route to dual_sync so it can be preserved/linked safely.
        if entity:
            raw_span = (entity.get("span_text") or "").strip()

            def _looks_product_like(s: str) -> bool:
                if not s:
                    return False
                s_stripped = s.strip()
                s_lower = s_stripped.lower()
                # dosage / units / forms
                if re.search(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|%)\b", s_lower):
                    return True
                if any(w in s_lower for w in ("tablet", "tab", "capsule", "cap", "syrup", "suspension", "inj", "injection", "chew", "chewable", "spot-on", "spot on", "spray", "drops", "shampoo")):
                    return True
                # common supplement markers
                if any(w in s_lower for w in ("omega", "fatty acid", "supplement", "force", "joint", "vitamin", "multivit")):
                    return True
                # brand-ish capitalization (2+ capitalized tokens) and not a normal vital phrase
                toks = [t for t in re.split(r"\s+", s_stripped) if t]
                cap_toks = [t for t in toks if len(t) >= 3 and t[0].isupper()]
                if len(cap_toks) >= 2 and not any(v in s_lower for v in ("temp", "temperature", "pulse", "resp", "respiratory", "heart rate", "hr", "rr", "bcs", "weight", "kg", "lbs")):
                    return True
                return False

            if _looks_product_like(raw_span):
                if logger:
                    logger.warning(f"  ⚠️  VitalSign mislabel guard: '{raw_span}' looks product-like → routing to dual_sync")
                return "dual_sync"

        return "skip_vitals"
        # NEW: With kb.vitals_registry present, we can canonicalize vitals via a safe global lookup
        # (no local billing search; no binding to diseases/procedures).
        return "global_vitals"
    
    # Diet: Most free-text food items should be treated as instructions (Bucket 4),
    # not KB-linked billables. We keep "prescription diet" items billable via explicit cues.
    if entity:
        span_text = (entity.get("span_text") or "").strip()
        if kind == "Diet":
            subtype = classify_diet_subtype(span_text)
            # B/C/D are instruction-like diets/advice (non-billable) in our taxonomy
            if subtype in ("B", "C", "D"):
                return "skip_vitals"

    # Step B: Check for vital keywords (comprehensive list with ASR error handling)
    # CRITICAL: Only check if kind is NOT a billable kind (prevents Drugs from being misrouted)
    span_text = (entity.get("span_text") or "").lower() if entity else text
    if kind not in DUAL_SYNC_BILLABLE_KINDS and is_fast_lane_vital(span_text, kind):
        return "global_vitals"  # Fast Lane - canonicalize as vital, don't KB-link
    
    # Step C: Check for clinical status constants (prevents "Seizure Disorder" hallucinations)
    # CRITICAL: Only check if kind is NOT a billable kind
    if kind not in DUAL_SYNC_BILLABLE_KINDS and is_clinical_status_constant(span_text):
        return "skip_vitals"  # Fast Lane - preserve verbatim, don't KB-link
    
    # Step D: Check for vital attributes (value_num, unit, qualitative, etc.)
    if any(k in attrs for k in ("vital_name", "value_num", "unit", "qualitative", "temperature", "pulse", "resp_rate", "respiratory_rate", "heart_rate", "blood_pressure", "measurement_type")):
        return "global_vitals"
    
    # Step E: General Health Statements (legacy check for backward compatibility)
    if attrs.get("observation_type") == "GeneralHealth":
        return "skip_vitals"  # Treat as structured observation, not KB-linked
    
    # Route 2: Lexical Overrides REMOVED - No keyword checks
    # GOLDEN RULE: "If a mention can be billed, it must be searched in Local."
    # ReasonForVisit and Condition are already in dual_sync_billable_kinds (Route 3 below)
    # The LLM Judge will determine if it's billable based on candidate matches, not keyword lists
    # This solves the problem for ALL 2,000+ procedures, not just keyword-matched ones
    
    # Route 3: Canonical Kind Routing
    # Dual-Sync (Billable Items) - MUST run local + global in parallel
    # CRITICAL: Condition and ReasonForVisit are in Dual-Sync to "revenue-proof" the system
    # Vets often name the condition when they mean the service (e.g., "He's here for a Dental")
    # By searching Local Inventory in parallel, we catch billable services even if NER misclassifies
    # Use module-level constant (exported for parallel processing)
    
    if kind in DUAL_SYNC_BILLABLE_KINDS:
        return "dual_sync"
    
    # CRITICAL: Hard-skip non-billable kinds - never hit DB (saves latency; billing accuracy)
    # Symptom and Anatomy stay in SOAP as text only; no local/global search.
    if kind in HARD_SKIP_KINDS:
        return "skip_non_billable"
    
    # Global-direct disabled in local-only production mode.
    if kind in GLOBAL_DIRECT_KINDS:
        return "global_direct"
    
    # Strict mode: all non-approved kinds are skipped.
    if kind == "Other":
        return "skip_other"

    # Unrecognized kind (not in DUAL_SYNC or GLOBAL_DIRECT): preserve verbatim, do not KB-link.
    return "skip_other"


def determine_entity_bucket(
    canonical_kind: str,
    route: str,
    diet_subtype: Optional[str] = None,
) -> int:
    """
    Determine the bucket (toolchain) for an entity based on Axis A.
    
    Bucket 1: Billable ID-binding (procedures/services, dispensed drugs, vaccines, billable tests)
    Bucket 2: Clinical truth ID-binding (symptoms, findings, conditions, anatomy)
    Bucket 3: Vitals/observations (do NOT KB-link, structure into vitals schema)
    Bucket 4: Instructions (diet/advice/home care, structure into instruction schema)
    
    Args:
        canonical_kind: Canonicalized entity kind
        route: Routing decision (skip_vitals, dual_sync, global_direct)
        diet_subtype: Diet subtype if applicable (A/B/C/D)
        
    Returns:
        Bucket number (1, 2, 3, or 4)
    """
    # Bucket 4: Instructions (diet subtypes B/C/D) - CHECK FIRST before route check
    # These use skip_vitals route but should be bucket 4, not bucket 3
    if canonical_kind == "Diet" and diet_subtype in ["B", "C", "D"]:
        return 4
    
    # Bucket 3: Vitals/observations + signalment + identity + non-billable kinds (do NOT KB-link)
    if route in ("skip_vitals", "global_vitals", "skip_signalment", "skip_identity", "skip_non_billable"):
        return 3
    
    # Bucket 1: Billable ID-binding (11-kind production schema)
    if canonical_kind in DUAL_SYNC_BILLABLE_KINDS:
        # Diet Subtype A is billable (product-like); B/C/D go to bucket 4 above
        if canonical_kind == "Diet" and diet_subtype not in (None, "A"):
            return 4
        return 1
    
    # Bucket 2: Clinical truth ID-binding (Symptom, Diagnosis, Anatomy)
    if canonical_kind in GLOBAL_DIRECT_KINDS:
        return 2
    
    # Bucket 4: Default for instructions/advice
    if canonical_kind == "Reminder":
        return 4
    if canonical_kind == "Diet":
        return 4  # Default to instructions if subtype not specified
    
    # Default: Bucket 2 (clinical truth)
    return 2


def extract_context_window(
    span_text: str,
    transcript: str,
    window_lines: int = 2,
    window_chars: int = 500,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Extract context window around a span in the transcript.
    
    Args:
        span_text: The span to find
        transcript: Full transcript text
        window_lines: Number of lines before/after to include
        window_chars: Maximum characters to include (fallback)
        logger: Optional logger
        
    Returns:
        Context window string
    """
    if not span_text or not transcript:
        return transcript[:window_chars] if transcript else ""
    
    # Find span position (case-insensitive)
    span_lower = span_text.lower()
    transcript_lower = transcript.lower()
    span_pos = transcript_lower.find(span_lower)
    
    if span_pos == -1:
        # Span not found - return beginning of transcript
        return transcript[:window_chars]
    
    # Extract context around span
    lines = transcript.split('\n')
    span_line_idx = None
    
    # Find which line contains the span
    char_count = 0
    for i, line in enumerate(lines):
        if span_pos >= char_count and span_pos < char_count + len(line):
            span_line_idx = i
            break
        char_count += len(line) + 1  # +1 for newline
    
    if span_line_idx is None:
        # Fallback: use character-based window
        start = max(0, span_pos - window_chars // 2)
        end = min(len(transcript), span_pos + len(span_text) + window_chars // 2)
        return transcript[start:end]
    
    # Extract line-based window
    start_line = max(0, span_line_idx - window_lines)
    end_line = min(len(lines), span_line_idx + window_lines + 1)
    context_lines = lines[start_line:end_line]
    
    return '\n'.join(context_lines)


def classify_procedure_role(
    span_text: str,
    context_window: str,
    speaker: Optional[str] = None,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Evidence-based role classification for procedure-like spans.
    
    Classifies a span into roles:
    - PresentingRequest: "came for", "here for", "wants"
    - PlannedProcedure: "we will do", "let's do", "plan to"
    - PerformedProcedure: "we did", "performed", "completed" (high bar: 0.85+ confidence)
    
    Args:
        span_text: The procedure span to classify
        context_window: Context around the span
        speaker: Optional speaker (Veterinarian/Pet Parent)
        client: OpenAI client
        logger: Optional logger
        
    Returns:
        Dict with span, roles (presenting_request, planned_procedure, performed_procedure),
        and billing_eligible flag
    """
    if not client:
        return {
            "span": span_text,
            "roles": {
                "presenting_request": {"value": False, "confidence": 0.0, "evidence": []},
                "planned_procedure": {"value": False, "confidence": 0.0, "evidence": []},
                "performed_procedure": {"value": False, "confidence": 0.0, "evidence": []}
            },
            "billing_eligible": False
        }
    
    prompt = f"""Classify the role(s) of this clinical span in the transcript.

Span: "{span_text}"
Context: "{context_window}"
Speaker: {speaker or "Unknown"}

Determine which role(s) apply:
1. PresentingRequest: Patient/pet parent came for this (e.g., "came for", "here for", "wants")
2. PlannedProcedure: Explicitly planned/ordered but not yet done (e.g., "we will do", "let's do", "plan to")
3. PerformedProcedure: Actually performed (e.g., "we did", "performed", "completed", "expressed")

CRITICAL RULES:
- PerformedProcedure requires HIGH CONFIDENCE (0.85+) and explicit evidence
- PresentingRequest requires MODERATE CONFIDENCE (0.5+) and evidence
- PlannedProcedure requires MODERATE CONFIDENCE (0.6-0.7+) and evidence
- A span can have MULTIPLE roles (e.g., both PresentingRequest and PerformedProcedure)
- billing_eligible is TRUE only if PerformedProcedure confidence >= 0.85

Return JSON:
{{
  "span": "{span_text}",
  "roles": {{
    "presenting_request": {{"value": bool, "confidence": float, "evidence": [strings]}},
    "planned_procedure": {{"value": bool, "confidence": float, "evidence": [strings]}},
    "performed_procedure": {{"value": bool, "confidence": float, "evidence": [strings]}}
  }},
  "billing_eligible": bool
}}"""

    # Single model: gpt-4.1-nano (no fallback chain)
    model_name = "gpt-4.1-nano"
    last_error = None
    resp = None
    try:
        model_client, provider = get_client_for_model(model_name, logger)
        if not model_client:
            if logger:
                logger.debug(f"  ⚠️  No client available for model '{model_name}' (provider: {provider})")
            last_error = RuntimeError(f"No client for {model_name}")
        else:
            resp = model_client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a veterinary clinical role classifier. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=400,
            )
            raw = resp.choices[0].message.content or "{}"
    except Exception as e:
        last_error = e
        if logger:
            logger.debug(f"  ⚠️  Role classification model '{model_name}' failed: {e}")

    if last_error:
        if logger:
            logger.warning(f"  ⚠️  All role classification models failed. Last error: {last_error}. Using defaults.")
        return {
            "span": span_text,
            "roles": {
                "presenting_request": {"value": False, "confidence": 0.0, "evidence": []},
                "planned_procedure": {"value": False, "confidence": 0.0, "evidence": []},
                "performed_procedure": {"value": False, "confidence": 0.0, "evidence": []}
            },
            "billing_eligible": False
        }
    
    # Parse response if we got one
    if resp:
        try:
            raw = resp.choices[0].message.content or "{}"
        except:
            raw = "{}"
        
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or prose
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                data = json.loads(json_match.group())
            else:
                if logger:
                    logger.warning(f"Failed to parse role classification JSON: {raw[:200]}")
                data = {}
        
        # Validate and normalize response
        span = data.get("span", span_text)
        roles = data.get("roles", {})
        
        presenting_request = roles.get("presenting_request", {})
        planned_procedure = roles.get("planned_procedure", {})
        performed_procedure = roles.get("performed_procedure", {})
        
        # Ensure all fields exist with defaults
        result = {
            "span": span,
            "roles": {
                "presenting_request": {
                    "value": bool(presenting_request.get("value", False)),
                    "confidence": float(presenting_request.get("confidence", 0.0)),
                    "evidence": list(presenting_request.get("evidence", []))
                },
                "planned_procedure": {
                    "value": bool(planned_procedure.get("value", False)),
                    "confidence": float(planned_procedure.get("confidence", 0.0)),
                    "evidence": list(planned_procedure.get("evidence", []))
                },
                "performed_procedure": {
                    "value": bool(performed_procedure.get("value", False)),
                    "confidence": float(performed_procedure.get("confidence", 0.0)),
                    "evidence": list(performed_procedure.get("evidence", []))
                }
            },
            "billing_eligible": bool(data.get("billing_eligible", False))
        }
        
        # Enforce strict threshold: billing_eligible only if performed_procedure confidence >= 0.85
        if result["roles"]["performed_procedure"]["confidence"] < 0.85:
            result["billing_eligible"] = False
        
        return result
    
    # If we get here, all models failed and we already returned defaults above
    # This should never be reached, but add as safety
    return {
        "span": span_text,
        "roles": {
            "presenting_request": {"value": False, "confidence": 0.0, "evidence": []},
            "planned_procedure": {"value": False, "confidence": 0.0, "evidence": []},
            "performed_procedure": {"value": False, "confidence": 0.0, "evidence": []}
        },
        "billing_eligible": False
    }


def batch_classify_procedure_roles(
    entities: List[Dict[str, Any]],
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Batch version of classify_procedure_role - processes multiple entities in one LLM call.
    
    Args:
        entities: List of dicts with keys: span_text, context_window, speaker (optional), entity_id (optional)
        client: OpenAI client
        logger: Optional logger
        
    Returns:
        Dict mapping entity_id (or span_text if no id) -> role classification result
    """
    if not entities or not client:
        # Return defaults for all entities
        results = {}
        for ent in entities:
            entity_id = ent.get("entity_id") or ent.get("span_text", "")
            results[entity_id] = {
                "span": ent.get("span_text", ""),
                "roles": {
                    "presenting_request": {"value": False, "confidence": 0.0, "evidence": []},
                    "planned_procedure": {"value": False, "confidence": 0.0, "evidence": []},
                    "performed_procedure": {"value": False, "confidence": 0.0, "evidence": []}
                },
                "billing_eligible": False
            }
        return results
    
    # Build batch prompt
    items = []
    for i, ent in enumerate(entities):
        entity_id = ent.get("entity_id") or f"entity_{i+1}"
        span_text = ent.get("span_text", "")
        context_window = ent.get("context_window", "")
        speaker = ent.get("speaker", "Unknown")
        items.append({
            "id": entity_id,
            "span": span_text,
            "context": context_window,
            "speaker": speaker
        })
    
    prompt = f"""Classify the role(s) of these clinical spans in the transcript. Process ALL {len(items)} entities in one response.

CRITICAL RULES:
- PerformedProcedure requires HIGH CONFIDENCE (0.85+) and explicit evidence
- PresentingRequest requires MODERATE CONFIDENCE (0.5+) and evidence
- PlannedProcedure requires MODERATE CONFIDENCE (0.6-0.7+) and evidence
- A span can have MULTIPLE roles (e.g., both PresentingRequest and PerformedProcedure)
- billing_eligible is TRUE only if PerformedProcedure confidence >= 0.85

Return JSON array with one object per entity:
{{
  "classifications": [
    {{
      "id": "entity_id",
      "span": "span_text",
      "roles": {{
        "presenting_request": {{"value": bool, "confidence": float, "evidence": [strings]}},
        "planned_procedure": {{"value": bool, "confidence": float, "evidence": [strings]}},
        "performed_procedure": {{"value": bool, "confidence": float, "evidence": [strings]}}
      }},
      "billing_eligible": bool
    }}
  ]
}}

Entities to classify:
"""
    
    for item in items:
        prompt += f"""
Entity ID: {item['id']}
Span: "{item['span']}"
Context: "{item['context'][:300]}"
Speaker: {item['speaker']}
---
"""
    
    model_name = "gpt-4.1-nano"
    results = {}
    
    # Initialize defaults for all entities
    for ent in entities:
        entity_id = ent.get("entity_id") or ent.get("span_text", "")
        results[entity_id] = {
            "span": ent.get("span_text", ""),
            "roles": {
                "presenting_request": {"value": False, "confidence": 0.0, "evidence": []},
                "planned_procedure": {"value": False, "confidence": 0.0, "evidence": []},
                "performed_procedure": {"value": False, "confidence": 0.0, "evidence": []}
            },
            "billing_eligible": False
        }
    
    try:
        model_client, provider = get_client_for_model(model_name, logger)
        if not model_client:
            if logger:
                logger.debug(f"  ⚠️  No client available for batch role classification model '{model_name}'")
            return results
        
        resp = model_client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a veterinary clinical role classifier. Return only valid JSON with a 'classifications' array."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=min(4000, 400 * len(entities)),  # Scale tokens with entity count
        )
        raw = resp.choices[0].message.content or "{}"
        if logger and resp.usage:
            logger.info(f"  ⚖️ Batch role classification: {len(entities)} entities in one LLM call ({resp.usage.total_tokens} tokens)")
        
        try:
            data = json.loads(raw)
            classifications = data.get("classifications", [])
            
            for cls in classifications:
                if not isinstance(cls, dict):
                    continue
                entity_id = str(cls.get("id", ""))
                if not entity_id:
                    continue
                
                roles = cls.get("roles", {})
                presenting_request = roles.get("presenting_request", {})
                planned_procedure = roles.get("planned_procedure", {})
                performed_procedure = roles.get("performed_procedure", {})
                
                billing_eligible = bool(cls.get("billing_eligible", False))
                # Enforce strict threshold
                if performed_procedure.get("confidence", 0.0) < 0.85:
                    billing_eligible = False
                
                if entity_id in results:
                    results[entity_id] = {
                        "span": cls.get("span", results[entity_id]["span"]),
                        "roles": {
                            "presenting_request": {
                                "value": bool(presenting_request.get("value", False)),
                                "confidence": float(presenting_request.get("confidence", 0.0)),
                                "evidence": list(presenting_request.get("evidence", []))
                            },
                            "planned_procedure": {
                                "value": bool(planned_procedure.get("value", False)),
                                "confidence": float(planned_procedure.get("confidence", 0.0)),
                                "evidence": list(planned_procedure.get("evidence", []))
                            },
                            "performed_procedure": {
                                "value": bool(performed_procedure.get("value", False)),
                                "confidence": float(performed_procedure.get("confidence", 0.0)),
                                "evidence": list(performed_procedure.get("evidence", []))
                            }
                        },
                        "billing_eligible": billing_eligible
                    }
        except json.JSONDecodeError as e:
            if logger:
                logger.warning(f"  ⚠️ Failed to parse batch role classification JSON: {e}")
                logger.debug(f"  Raw response (first 500 chars): {raw[:500]}")
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠️ Batch role classification failed: {e}")
    
    return results


def classify_diet_subtype(
    span_text: str,
    context_window: Optional[str] = None,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Classify a diet mention into one of 4 subtypes (deterministic, no rewriting).
    
    Subtype A - Brand/Product-like:
    - Brand tokens (TitleCase brand words, common pet-food brands)
    - SKU-ish patterns ("1.5kg", "2kg bag", "starter 20kg", "tin/can/pouch")
    - Exact product name in inventory
    
    Subtype B - Prescription diet class:
    - "GI diet", "renal diet", "urinary diet", "hypoallergenic", "hydrolyzed", "low fat", "recovery"
    
    Subtype C - Home-cooked/ingredient diet:
    - "chicken and rice", "boiled chicken", "curd", "pumpkin", "add fiber"
    
    Subtype D - Feeding instruction only:
    - "small frequent meals", "avoid treats", "mix with water", "feed twice a day"
    
    Args:
        span_text: The diet mention span
        context_window: Optional context around the span
        client: OpenAI client (optional, for LLM fallback)
        logger: Logger instance
        
    Returns:
        One of: "A" (Brand/Product), "B" (Prescription class), "C" (Home-cooked), "D" (Instruction)
    """
    if not span_text:
        return "D"  # Default to instruction
    
    span_lower = span_text.lower().strip()
    
    # Rule-based classification (deterministic, fast)
    
    # Subtype A: Brand/Product-like patterns
    # Common pet food brands (case-insensitive check)
    # IMPORTANT: avoid ultra-short substrings (e.g., "nd") that collide with normal words like "and".
    brand_phrases = [
        "royal canin",
        "hill's", "hills",
        "nutrish", "nutrich",
        "farmina",
        "whiskas", "pedigree", "purina", "iams", "eukanuba", "wellness", "orijen", "acana",
        "taste of the wild", "blue buffalo", "science diet",
        "prescription diet", "prescription",
        "coatex", "cortex", "virbac", "vetri", "vetriscience",
    ]
    # Short brand tokens must use word boundaries
    brand_token_regexes = [
        r"\brc\b",          # Royal Canin shorthand
        r"\bnd\b",          # N&D shorthand (must not match "and")
        r"\bn&d\b",
    ]
    
    # SKU patterns
    import re
    sku_patterns = [
        r"\d+\.?\d*\s*(kg|g|lb|lbs|oz|pound|pounds)",  # Weight
        r"\d+\s*(bag|pack|box|can|tin|pouch|pouches|bottle)",  # Package type
        r"(starter|puppy|kitten|adult|senior|mature)\s+\d+",  # Life stage + size
        r"\d+\s*(tablet|tablets|capsule|capsules|chew|chews)",  # Count + form
    ]
    
    # Check for brand patterns
    is_brand_like = any(brand in span_lower for brand in brand_phrases) or any(
        re.search(rx, span_lower, flags=re.IGNORECASE) for rx in brand_token_regexes
    )
    
    # Check for SKU patterns
    has_sku_pattern = any(re.search(pattern, span_text, re.IGNORECASE) for pattern in sku_patterns)
    
    # Check for TitleCase brand words (heuristic)
    words = span_text.split()
    titlecase_words = [w for w in words if w and w[0].isupper() and len(w) > 2]
    has_titlecase_brand = len(titlecase_words) >= 1 and any(len(w) >= 4 for w in titlecase_words)
    
    if is_brand_like or has_sku_pattern or has_titlecase_brand:
        return "A"  # Brand/Product-like
    
    # Subtype B: Prescription diet class patterns
    prescription_classes = [
        "gi diet", "renal diet", "urinary diet", "hypoallergenic", "hydrolyzed",
        "low fat", "low-fat", "low sodium", "low-sodium", "recovery diet",
        "weight management", "weight control", "diabetic diet", "cardiac diet",
        "gastrointestinal", "kidney diet", "liver diet", "pancreatic diet"
    ]
    
    if any(cls in span_lower for cls in prescription_classes):
        return "B"  # Prescription diet class
    
    # Subtype C: Home-cooked/ingredient patterns
    ingredient_patterns = [
        "chicken and rice", "boiled chicken", "chicken rice", "rice and chicken",
        "curd", "yogurt", "pumpkin", "sweet potato", "carrot", "green beans",
        # Common Indian home-cooked diet mentions
        "rice and sambar", "rice & sambar", "rice with sambar", "sambar",
        "rice and dal", "rice & dal", "dal", "khichdi", "khichadi",
        "chapati", "roti", "idli", "dosa",
        "home cooked food", "home-cooked food", "home cooked", "home-cooked",
        "add fiber", "add", "mix with", "home cooked", "home-cooked", "homemade"
    ]
    
    if any(ing in span_lower for ing in ingredient_patterns):
        return "C"  # Home-cooked/ingredient
    
    # Subtype D: Feeding instruction patterns (default)
    instruction_patterns = [
        "small frequent meals", "frequent meals", "avoid treats", "no treats",
        "mix with water", "feed twice", "feed once", "feed three times",
        "gradual transition", "transition", "wean", "introduce slowly"
    ]
    
    if any(inst in span_lower for inst in instruction_patterns):
        return "D"  # Feeding instruction
    
    # Fallback: Use LLM for ambiguous cases (only if client available)
    if client and context_window:
        try:
            diet_classification_prompt = f"""Classify this diet mention into exactly ONE subtype:

**Diet Mention:** "{span_text}"
**Context:** {context_window[:200]}

**Subtypes:**
- A: Brand/Product-like (e.g., "Royal Canin GI", "Nutrish", "Hill's 1.5kg")
- B: Prescription diet class (e.g., "GI diet", "renal diet", "low fat diet")
- C: Home-cooked/ingredient diet (e.g., "chicken and rice", "boiled chicken", "add pumpkin")
- D: Feeding instruction only (e.g., "small frequent meals", "avoid treats", "feed twice a day")

Return ONLY the letter (A, B, C, or D), no explanation."""

            resp = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are a diet classification assistant. Return ONLY a single letter: A, B, C, or D."},
                    {"role": "user", "content": diet_classification_prompt}
                ],
                temperature=0.0,
                max_tokens=5,
            )
            result = resp.choices[0].message.content.strip() if resp.choices else "D"
            if result.upper() in ["A", "B", "C", "D"]:
                if logger:
                    logger.debug(f"  🍽️  Diet subtype classification (LLM): '{span_text}' → {result.upper()}")
                return result.upper()
        except Exception as e:
            if logger:
                logger.debug(f"  ⚠️  Diet subtype LLM classification failed: {e}")
    
    # Default fallback
    return "D"


def batch_classify_diet_subtypes(
    entities: List[Dict[str, Any]],
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """
    Batch version of classify_diet_subtype - processes multiple diet entities in one LLM call.
    
    Args:
        entities: List of dicts with keys: span_text, context_window (optional), entity_id (optional)
        client: OpenAI client (optional, only used for ambiguous cases)
        logger: Optional logger
        
    Returns:
        Dict mapping entity_id (or span_text if no id) -> subtype ("A", "B", "C", or "D")
    """
    if not entities:
        return {}
    
    results = {}
    ambiguous_entities = []
    
    # First pass: rule-based classification for all entities
    for ent in entities:
        entity_id = ent.get("entity_id") or ent.get("span_text", "")
        span_text = ent.get("span_text", "")
        
        if not span_text:
            results[entity_id] = "D"
            continue
        
        # Use rule-based classification first (same logic as single-entity version)
        span_lower = span_text.lower().strip()
        
        # Subtype A: Brand/Product-like patterns
        brand_phrases = [
            "royal canin", "hill's", "hills", "nutrish", "nutrich", "farmina",
            "whiskas", "pedigree", "purina", "iams", "eukanuba", "wellness", "orijen", "acana",
            "taste of the wild", "blue buffalo", "science diet",
            "prescription diet", "prescription",
            "coatex", "cortex", "virbac", "vetri", "vetriscience",
        ]
        brand_token_regexes = [
            r"\brc\b", r"\bnd\b", r"\bn&d\b",
        ]
        
        import re
        sku_patterns = [
            r"\d+\.?\d*\s*(kg|g|lb|lbs|oz|pound|pounds)",
            r"\d+\s*(bag|pack|box|can|tin|pouch|pouches|bottle)",
            r"(starter|puppy|kitten|adult|senior|mature)\s+\d+",
            r"\d+\s*(tablet|tablets|capsule|capsules|chew|chews)",
        ]
        
        is_brand_like = any(brand in span_lower for brand in brand_phrases) or any(
            re.search(rx, span_lower, flags=re.IGNORECASE) for rx in brand_token_regexes
        )
        has_sku_pattern = any(re.search(pattern, span_text, re.IGNORECASE) for pattern in sku_patterns)
        
        words = span_text.split()
        titlecase_words = [w for w in words if w and w[0].isupper() and len(w) > 2]
        has_titlecase_brand = len(titlecase_words) >= 1 and any(len(w) >= 4 for w in titlecase_words)
        
        if is_brand_like or has_sku_pattern or has_titlecase_brand:
            results[entity_id] = "A"
            continue
        
        # Subtype B: Prescription diet class patterns
        prescription_classes = [
            "gi diet", "renal diet", "urinary diet", "hypoallergenic", "hydrolyzed",
            "low fat", "low-fat", "low sodium", "low-sodium", "recovery diet",
            "weight management", "weight control", "diabetic diet", "cardiac diet",
            "gastrointestinal", "kidney diet", "liver diet", "pancreatic diet"
        ]
        
        if any(cls in span_lower for cls in prescription_classes):
            results[entity_id] = "B"
            continue
        
        # Subtype C: Home-cooked/ingredient patterns
        ingredient_patterns = [
            "chicken and rice", "boiled chicken", "chicken rice", "rice and chicken",
            "curd", "yogurt", "pumpkin", "sweet potato", "carrot", "green beans",
            "rice and sambar", "rice & sambar", "rice with sambar", "sambar",
            "rice and dal", "rice & dal", "dal", "khichdi", "khichadi",
            "chapati", "roti", "idli", "dosa",
            "home cooked food", "home-cooked food", "home cooked", "home-cooked",
            "add fiber", "add", "mix with", "home cooked", "home-cooked", "homemade"
        ]
        
        if any(ing in span_lower for ing in ingredient_patterns):
            results[entity_id] = "C"
            continue
        
        # Subtype D: Feeding instruction patterns
        instruction_patterns = [
            "small frequent meals", "frequent meals", "avoid treats", "no treats",
            "mix with water", "feed twice", "feed once", "feed three times",
            "gradual transition", "transition", "wean", "introduce slowly"
        ]
        
        if any(inst in span_lower for inst in instruction_patterns):
            results[entity_id] = "D"
            continue
        
        # Ambiguous - needs LLM fallback
        ambiguous_entities.append({
            "entity_id": entity_id,
            "span_text": span_text,
            "context_window": ent.get("context_window", "")
        })
        results[entity_id] = "D"  # Default until LLM resolves
    
    # Batch LLM call for ambiguous entities only
    if ambiguous_entities and client:
        try:
            prompt = f"""Classify these diet mentions into exactly ONE subtype each (A, B, C, or D).

**Subtypes:**
- A: Brand/Product-like (e.g., "Royal Canin GI", "Nutrish", "Hill's 1.5kg")
- B: Prescription diet class (e.g., "GI diet", "renal diet", "low fat diet")
- C: Home-cooked/ingredient diet (e.g., "chicken and rice", "boiled chicken", "add pumpkin")
- D: Feeding instruction only (e.g., "small frequent meals", "avoid treats", "feed twice a day")

Return JSON array with one object per entity:
{{
  "classifications": [
    {{"id": "entity_id", "subtype": "A|B|C|D"}}
  ]
}}

Diet mentions to classify:
"""
            
            for item in ambiguous_entities:
                context_preview = item["context_window"][:200] if item["context_window"] else "No context"
                prompt += f"""
Entity ID: {item['entity_id']}
Diet Mention: "{item['span_text']}"
Context: {context_preview}
---
"""
            
            resp = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are a diet classification assistant. Return only valid JSON with a 'classifications' array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=min(1000, 50 * len(ambiguous_entities)),
            )
            
            raw = resp.choices[0].message.content or "{}"
            
            if logger:
                logger.info(f"  🍽️ Batch diet subtype classification: {len(ambiguous_entities)} ambiguous entities in one LLM call")
            
            try:
                data = json.loads(raw)
                classifications = data.get("classifications", [])
                
                for cls in classifications:
                    if not isinstance(cls, dict):
                        continue
                    entity_id = str(cls.get("id", ""))
                    subtype = str(cls.get("subtype", "D")).upper()
                    
                    if entity_id in results and subtype in ["A", "B", "C", "D"]:
                        results[entity_id] = subtype
            except json.JSONDecodeError as e:
                if logger:
                    logger.warning(f"  ⚠️ Failed to parse batch diet subtype JSON: {e}")
        except Exception as e:
            if logger:
                logger.warning(f"  ⚠️ Batch diet subtype classification failed: {e}")
    
    return results


def has_product_context_evidence(
    span_text: str,
    context_window: str,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Check if transcript context implies a product context (not just mention).
    
    Evidence patterns:
    - "start Nutrish", "feed Nutrish", "switch to Nutrish", "continue Nutrish"
    - NOT: "Nutrish... uh..." (ambiguous)
    
    Args:
        span_text: The diet mention
        context_window: Context around the mention
        logger: Optional logger
        
    Returns:
        True if context suggests product usage, False if ambiguous
    """
    if not context_window:
        return False
    
    context_lower = context_window.lower()
    span_lower = span_text.lower()
    
    # Product action verbs
    product_verbs = [
        "start", "feed", "switch", "continue", "give", "use", "prescribe",
        "recommend", "suggest", "try", "change to", "transition to"
    ]
    
    # Check if span appears near product verbs
    for verb in product_verbs:
        # Look for patterns like "verb + span" or "verb + to + span"
        patterns = [
            f"{verb} {span_lower}",
            f"{verb} to {span_lower}",
            f"{verb} the {span_lower}",
            f"{span_lower} {verb}",  # Less common but possible
        ]
        if any(pattern in context_lower for pattern in patterns):
            if logger:
                logger.debug(f"  ✅ Product context evidence found: '{verb}' + '{span_text}'")
            return True
    
    # Check for explicit product mentions with determiners
    product_determiners = ["the ", "this ", "that ", "a ", "an "]
    for det in product_determiners:
        if f"{det}{span_lower}" in context_lower:
            return True
    
    # If no clear evidence, return False (ambiguous)
    if logger:
        logger.debug(f"  ⚠️  No clear product context evidence for '{span_text}'")
    return False


def extract_roles_from_classification(
    role_classification: Optional[Dict[str, Any]],
    canonical_kind: str,
    billing_eligible: bool = False,
) -> List[str]:
    """
    Extract roles list from role classification (Axis B).
    
    Roles:
    - PresentingRequest (Reason for visit)
    - Performed (actually performed)
    - Planned/Ordered (explicitly planned/ordered)
    - Prescribed (for drugs - prescribed/dispensed, sent home)
    - Administered (for drugs - given at clinic)
    - Recommended (for drugs - recommended but not given)
    
    Args:
        role_classification: Role classification result from classify_procedure_role
        canonical_kind: Canonicalized entity kind
        billing_eligible: Whether entity is billing-eligible
        
    Returns:
        List of role strings
    """
    roles = []
    
    if not role_classification:
        # Default roles based on kind and billing eligibility
        if canonical_kind in ["Procedure", "Service"]:
            if billing_eligible:
                roles.append("Performed")
            else:
                roles.append("Planned")
        elif canonical_kind == "Drug":
            if billing_eligible:
                roles.append("Prescribed")  # Prescribed includes dispensed (sent home)
            else:
                roles.append("Recommended")
        return roles
    
    role_data = role_classification.get("roles", {})
    
    # PresentingRequest
    presenting_request = role_data.get("presenting_request", {})
    if presenting_request.get("value", False) and presenting_request.get("confidence", 0.0) >= 0.5:
        roles.append("PresentingRequest")
    
    # Performed (for procedures)
    if canonical_kind in ["Procedure", "Service"]:
        performed = role_data.get("performed_procedure", {})
        if performed.get("value", False) and performed.get("confidence", 0.0) >= 0.85:
            roles.append("Performed")
        elif performed.get("value", False):
            roles.append("Planned")  # Lower confidence = planned, not performed
    
    # Planned
    planned = role_data.get("planned_procedure", {})
    if planned.get("value", False) and planned.get("confidence", 0.0) >= 0.6:
        roles.append("Planned")
    
    # For medication, infer from billing_eligible
    if canonical_kind == "Medication":
        if billing_eligible:
            roles.append("Prescribed")  # Prescribed includes dispensed (sent home)
        else:
            roles.append("Recommended")
    
    return roles if roles else ["Unknown"]

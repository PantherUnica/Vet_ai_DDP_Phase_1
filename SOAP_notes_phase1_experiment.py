#!/opt/homebrew/bin/python3.11
"""
Veterinary SOAP Note Generator - Phase 1

This module generates SOAP notes from transcribed veterinary conversations
using various LLM providers (OpenAI, Claude, Mistral).

Features:
- Processes transcribed veterinary conversations
- Generates structured SOAP notes
- Supports multiple LLM providers
- Comprehensive error handling and logging
- Optional inputs for enhanced SOAP note generation
- Configurable model selection

Configuration:
- Set MODEL_PROVIDER and MODEL_NAME at the top of the file
- API keys are loaded from API_Key.txt in the same directory

API_Key.txt format:
MISTRAL_API_KEY=your_mistral_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
(Each key is optional, but required for the provider you select.)

Optional Input Files:
1. Pre-appointment summary: Text file containing owner's initial complaint/concerns
   Example content: "Dog limping for 3 days, seems to be in pain when walking..."

2. Protocols template: Text file with treatment protocols/checklists
   Example content: 
   "Orthopedic Examination Protocol:
   1. Visual gait assessment: [ ]
   2. Palpation of affected limb: [ ]
   3. Range of motion testing: [ ]"

3. Vitals template: Text file with vital signs template
   Example content:
   "Patient Vitals:
   Temperature: ___°F
   Heart Rate: ___ bpm
   Respiratory Rate: ___ rpm
   Weight: ___ lbs"

Author: VetInstant P.A.W.S Team
Version: 1.0
"""

import os
import sys
import requests
import logging
import traceback
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import json
import numpy as np
try:
    import soundfile as sf  # type: ignore
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None  # type: ignore
    SOUNDFILE_AVAILABLE = False
    logging.warning("soundfile not available. Some audio formats may require pydub/librosa/ffmpeg fallbacks.")
import tempfile

# Default Super-Pass model for cleaned transcript + Brain NER:
# Fireworks Llama v3p3 70B. Override with SUPER_PASS_MODEL if needed.
os.environ.setdefault("SUPER_PASS_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct")

# Long transcript safety helpers (summary blocks + prompt-safe excerpts)
try:
    from long_transcript_utils import extract_summary_block, build_prompt_safe_transcript
except Exception:
    extract_summary_block = None
    build_prompt_safe_transcript = None

# Import SQL embeddings functionality
# Note: sys and os are already imported at the top of the file (lines 49-50)
# sql_embeddings.py is now in the same directory as this file
try:
    from sql_embeddings import get_embedding, setup_openai, connect_to_postgres
except ImportError:
    # Use logging.warning() instead of logger.warning() since logger isn't defined yet
    logging.warning("sql_embeddings module not found. Database embedding functionality will be disabled.")
    get_embedding = None
    setup_openai = None
    connect_to_postgres = None


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# --- Configuration ---
# Set the model provider and model name:
# ==============================================================================
# MODEL CONFIGURATION: All LLM steps use gpt-4.1-nano except ASR (Whisper v3 turbo).
# ==============================================================================
# Override with env vars (SUPER_PASS_MODEL, SOAP_MODEL, BATCH_INTENT_MODEL, LLM_JUDGE_MODEL, PHASE2_MODEL) if needed.

# Step 3 (SOAP Generation): gpt-4.1-mini
os.environ.setdefault("SOAP_MODEL", "gpt-4.1-mini")
if os.getenv("SOAP_GENERATOR_MODEL"):
    os.environ["SOAP_MODEL"] = os.getenv("SOAP_GENERATOR_MODEL", "").strip()
os.environ.setdefault("SOAP_MODEL_PROVIDER", "openai")
os.environ.setdefault("SOAP_STRUCTURED_OUTPUT", "true")

# TARGET_60S / LOW_LATENCY_MODE: enable parallel chunks + early SOAP only.
TARGET_60S = os.getenv("TARGET_60S", "true").lower() in ("1", "true", "yes")
if (
    os.getenv("LOW_LATENCY_MODE", "").lower() in ("1", "true", "yes")
    or os.getenv("TARGET_LATENCY_SEC", "") == "60"
    or TARGET_60S
):
    if TARGET_60S:
        os.environ.setdefault("FORCE_PARALLEL_SUPER_PASS_CHUNKS", "true")
        os.environ.setdefault("EARLY_START_SOAP", "true")

MODEL_PROVIDER = os.getenv("SOAP_MODEL_PROVIDER", "openai")
MODEL_NAME = os.getenv("SOAP_MODEL", "gpt-4.1-mini")  # Step 3: SOAP Generation

# Step 2 (Cleaning/NER): aligned with SUPER_PASS_MODEL default (Fireworks Llama v3p3 70B)
STEP_2_CLEANING_MODEL = 'accounts/fireworks/models/llama-v3p3-70b-instruct'
STEP_2_3_NORMALIZER_MODEL = 'accounts/fireworks/models/llama-v3p3-70b-instruct'
STEP_2_5_NER_MODEL = 'accounts/fireworks/models/llama-v3p3-70b-instruct'

# Step 2 (Super-Pass): Combined Cleaning + NER in a single call
SUPER_PASS_MODEL = os.getenv("SUPER_PASS_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct")

# Chunk-Parallel Factory: Enable parallel chunk processing for sub-60s latency
# Set CHUNK_PARALLEL_ENABLED=true to enable chunk-parallel processing
# Chunk count/size is adaptive to transcript length (see kb_ner_chunk_parallel: CHUNK_SINGLE_THRESHOLD,
# CHUNK_TARGET_SIZE, CHUNK_MIN_PARALLEL, CHUNK_MAX_PARALLEL, CHUNK_MIN_SIZE).
CHUNK_PARALLEL_ENABLED = os.getenv("CHUNK_PARALLEL_ENABLED", "true").lower() in ("1", "true", "yes")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "4000"))  # Default: 4000 chars (adaptive logic may override per run)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))  # Default: 500 chars overlap
# Skip CER for short transcripts to avoid ~60s CER call; use Brain NER entities directly for grounding.
# Set to 0 to always run CER. Example: 5000 = skip CER when transcript has ≤5000 chars (restores ~20s for short).
CER_SKIP_UNDER_CHARS = int(os.getenv("CER_SKIP_UNDER_CHARS", "5000"))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


# Centralized grounding thresholds (used across streaming + non-streaming paths).
GROUNDING_AUTO_BIND_THRESHOLD = _env_float("GROUNDING_AUTO_BIND_THRESHOLD", 0.92)
GROUNDING_LLM_JUDGE_THRESHOLD = _env_float("GROUNDING_LLM_JUDGE_THRESHOLD", 0.55)

# ==============================================================================
# FEATURE FLAGS
# ==============================================================================

ENABLE_STEP_35_REFINEMENT = False  # Turn SOAP refinement on/off (Step 3.5) - DISABLED: Redundant with Step 2.3

# Super-Pass: Combined Cleaning + NER (eliminates 2.5s network round-trip)
# Default: ENABLED (set USE_SUPER_PASS=false to disable and use separate cleaning + NER)
USE_SUPER_PASS = os.getenv("USE_SUPER_PASS", "true").lower() == "true"

# Fast Transcription Mode: Optimized for speed (3-5x faster)
# When enabled: Skips audio preprocessing for WAV files, uses connection pooling, minimal API config
# When disabled: Full preprocessing with robust error handling and timestamps
# Default: ENABLED for production speed (set FAST_TRANSCRIPTION=false to disable)
FAST_TRANSCRIPTION = os.getenv("FAST_TRANSCRIPTION", "true").lower() == "true"
# DEFAULT: False - Step 3.5 is now redundant with Step 2.3
# This step corrects ASR errors, links entities to KB concepts, and produces Entity Manifest for Phase 2
# When True: Runs Step 3.5 automatically after Phase 1, producing refined SOAP + Entity Manifest JSON
# When False: Skips Step 3.5 (recommended - Step 2.3 already does the linking)
# 
# RECOMMENDED: Use conditional execution based on Step 2.5 confidence scores:
#   - Skip if all entities have high confidence (>0.85) and no Plausibility Gate corrections
#   - Run if low confidence (<0.60) or Plausibility Gate made corrections
# 
# RATIONALE: Step 3.5 adds significant latency without value when Step 2.5 already did the work.
# Focus Phase 1 on speed, use Phase 2 for deep attribute extraction.

# API Key file configuration (similar to OP_summary.py)
FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_FOLDER_PATH = os.path.dirname(FOLDER_PATH)  # soap_notes_phase_1
EXPERIMENT_FOLDER_PATH = os.path.dirname(PARENT_FOLDER_PATH)  # soap_note_experiment (API_Key.txt often here)
# Ensure phase_1 is on sys.path when run from parent (so kb_ner_* resolve)
if FOLDER_PATH not in sys.path:
    sys.path.insert(0, FOLDER_PATH)
# Resolve API_Key.txt: check experiment folder first, then parent (phase_1), then current (20feb20206)
def _resolve_api_key_file() -> str:
    for base in (EXPERIMENT_FOLDER_PATH, PARENT_FOLDER_PATH, FOLDER_PATH):
        path = os.path.join(base, "API_Key.txt")
        if os.path.exists(path):
            return path
    return os.path.join(FOLDER_PATH, "API_Key.txt")
API_KEY_FILE = _resolve_api_key_file()

# Lexical Harvester has been removed from the pipeline

# ==============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# ==============================================================================

# Performance optimizations implemented:
# 1. Increased max_tokens to 6000 to ensure complete SOAP notes with all sections
# 2. Request timeout set to 180s to handle longer responses with max_tokens=6000
# 3. Reduced max retries from 3 to 2
# 4. Parallel file reading for optional inputs
# 5. Configurable model selection for easy switching

# Sub-60s: lower SOAP max_tokens (output ~1k tokens; 2500 cap reduces latency)
OPTIMIZED_CONFIG = {
    "max_tokens": 2500 if TARGET_60S else 6000,
    "temperature": 0.1,        # Keep low for consistency
    "request_timeout": 180,    # Increased to 180s to handle longer responses with max_tokens=6000
    "max_retries": 2,          # Reduced from 3 - fewer retries for speed
}




# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class EmptyTranscriptionError(RuntimeError):
    """Custom exception for empty transcription errors."""
    pass


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


@dataclass
class Config:
    """Configuration settings for the SOAP note generator."""
    # Required fields (no default values) - must come first
    input_transcription_path: str
    output_dir: str
    api_key_file: str
    model_provider: ModelProvider
    model_name: str
    
    # Optional fields (with default values) - must come after required fields
    pre_appointment_summary_path: str = None
    protocols_template_path: str = None
    vitals_template_path: str = None
    request_timeout: int = OPTIMIZED_CONFIG["request_timeout"]
    max_retries: int = OPTIMIZED_CONFIG["max_retries"]


# Default configuration - uses configurable MODEL_PROVIDER and MODEL_NAME variables
# Note: input_transcription_path is now dynamically set when using generate_soap_note_from_audio()
DEFAULT_CONFIG = Config(
    # Required fields (input_transcription_path will be set dynamically)
    input_transcription_path="",  # Will be set to temp file when using audio input
    output_dir="/Users/vivek/VETINSTANT/wip/New folder/P.A.W.S/SOAP notes - voice to text/OP/soap_note_experiment/output/soap_notes",
    api_key_file=API_KEY_FILE,
    model_provider=get_model_provider_enum(MODEL_PROVIDER),  # Uses configurable MODEL_PROVIDER variable
    model_name=MODEL_NAME,  # Uses configurable MODEL_NAME variable
    # Optional fields - set paths here if available, otherwise leave as None
    pre_appointment_summary_path=None,  # "/Users/vivek/VETINSTANT/wip/New folder/P.A.W.S/SOAP notes - voice to text/OP/path/to/pre_appointment_summary.txt"
    protocols_template_path=None,       # "/Users/vivek/VETINSTANT/wip/New folder/P.A.W.S/SOAP notes - voice to text/OP/path/to/protocols_template.txt"
    vitals_template_path=None          # "/Users/vivek/VETINSTANT/wip/New folder/P.A.W.S/SOAP notes - voice to text/OP/path/to/vitals_template.txt"
)


# ==============================================================================
# PHASE 1: TRANSCRIPTION CLEANING PROMPT
# ==============================================================================

TRANSCRIPTION_CLEANING_PROMPT = """
Phase 0:
You are a veterinary clinical scribe specializing in cleaning raw ASR transcripts for clarity while strictly preserving clinical specificity.

**SYSTEM ROLE: Veterinary Clinical Scribe**
**TASK: Clean the raw ASR transcript for clarity while strictly preserving clinical specificity.**

**RAW TRANSCRIPTION:**
{conversation}
{optional_inputs}

**OPTIONAL CONTEXT (Phase 0):**
The {optional_inputs} section above might include:

- ClinicalContext:
  - Species, breed, age, sex
  - Known diagnoses / active problems
  - Current medications
  - Previous visit summary (for follow-up cases)
  - Reason for today's visit (short)

**STRICT RULE - ClinicalContext Usage:**
You may use ClinicalContext (if provided) ONLY to:
- Translate non-English phrases accurately
- Improve speaker attribution (e.g., if context helps identify who is speaking)
- Resolve purely formatting-level ambiguity (punctuation, sentence boundaries)

**STRICT BAN - No Clinical Corrections in Phase 0:**
- Do NOT correct or replace clinical terms (procedures, anatomy, diagnoses, findings, drug names) using context in Phase 0
- Do NOT choose between phonetically similar medical terms (e.g., do NOT change "animal plant expression" to "anal gland expression")
- Do NOT correct obvious drug/tick-product/vaccine names (e.g., do NOT change "Cortex" to "Coatex" or "mucosmoburn" to "mucous membranes")
- **CRITICAL: If a clinically important phrase is unclear/garbled, preserve it verbatim (do NOT add [unclear] tag)**
  - Examples: "animal plant expression" → "animal plant expression" (preserve verbatim, Phase 2.3 will handle ASR correction)
  - Examples: "mucosmoburn" → "mucosmoburn" (preserve verbatim, Phase 2.3 will handle ASR correction)
  - Phase 2.3 (Grounding Layer) will automatically detect and correct garbled terms using ASR correction
  - **IMPORTANT: Remove any [unclear] tags from the input transcript - they are not needed**
- Do not add new symptoms or diagnoses based only on context if they are not in the RAW TRANSCRIPTION

**Why This Rule Exists:**
Phase 0 is for verbatim preservation only. Clinical term correction and grounding happens in Phase 2.3 (Grounding Layer) using Local Inventory, Global KB, and LLM Judge. If Phase 0 "corrects" terms, it creates drift that makes Phase 2.3 look like it's hallucinating when it's actually inheriting Phase 0's corrections.

**CRITICAL RULES - VERBATIM CLINICAL PRESERVATION:**

1. **ZERO-SUMMARIZATION CONSTRAINT**
   - NEVER shorten a specific procedure or clinical instruction.
   - If the vet says "Give ten milligrams of Maropitant once a day for five days," output EXACTLY that string.
   - DO NOT convert to shorthand like "Maropitant 10mg SID x5d" - this destroys raw data needed for Phase 2 (Attribute Extraction).
   - Example: "anal gland expression" must remain "anal gland expression" - NOT shortened to "expression" or "examination".

2. **NUMERIC AND UNIT LOCKING**
   - Every number and clinical unit (mg, ml, kg, %, tablets, capsules, BID, TID, SID) must be treated as a PROTECTED TOKEN.
   - Only fix acoustic confusion (e.g., "ten and G" → "10mg") while leaving surrounding sentence structure intact.
   - Preserve the link between numbers and units - do not separate them.
   - Example: "two capsules" must remain "two capsules" - do not change to "2 caps" or "capsules".

3. **STRUCTURAL INTEGRITY**
   - Preserve conversational "glue" that helps understand relationships (e.g., "Give 2 capsules" must keep the verb "Give").
   - Do not remove context words that link medications to dosages.
   - Example: "dispensed Cortex capsules along with Nutrish tablets" must preserve the full phrase, not just "Cortex, Nutrish".

4. **CLINICAL DETAIL LOCKDOWN**
   - NO SUMMARIZATION: Never shorten a specific procedure. If the vet says 'anal gland expression' or 'orthopedic examination,' do NOT clean it to 'expression' or 'examination.'
   - Preserve all specific anatomical locations (e.g., 'right cranial,' 'distal,' 'anal gland'), procedure types, and brand names verbatim.
   - If a noun is preceded by an anatomical descriptor (e.g., 'Cardiac' examination, 'Dental' scaling, 'Anal' expression), the descriptor and the noun must be treated as a single, inseparable clinical unit.
   - Example: "anal gland expression" → MUST remain "anal gland expression" (not "expression" or "gland expression").

5. **TERMINOLOGY LOCKDOWN**
   - Preserve all specific anatomical locations (e.g., 'right cranial,' 'distal,' 'anal gland'), procedure types, and brand names verbatim.
   - Do not change one medical term into another (e.g., do not turn "scabies" into "allergy" or "plant" into "anal").
   - Brand names must be preserved exactly as spoken (e.g., "Cortex", "Coatex", "Nutrish", "Maropitant").

6. **ASR NOISE ONLY**
   - ONLY remove filler words (um, uh, like, hmm), repetitive stutters, and obvious background noise.
   - If a phrase sounds like a medical term but is slightly garbled, keep it as-is for the Grounding Layer to fix.
   - Do NOT remove phrases that might be the reason for visit, a diagnosis, a symptom, a procedure, or a treatment, even if the wording is odd or partially wrong.
   - If such a phrase is unclear or misrecognized, leave it in verbatim (do NOT add [unclear] tag - Phase 2.3 will handle ASR correction).

Step 1: Clean-up the transcript
Purpose: The transcript is a raw automated conversation between a pet parent and a veterinarian. This has to be cleaned up by identifying repetitions, any unnatural sounds or background noises which are literally translated as well as translating words in others languages to meaningful English.

Actions:
a. Analyze the Transcript:
   - Analyze the transcript for grammatical errors, fillers, and repeated phrases.
   - Some words may be misrecognized by ASR. Your job in this step is NOT to guess the correct medical term.
   - **MANDATORY: If a phrase seems clinically important but unclear/garbled, preserve it verbatim (do NOT add [unclear] tag - Phase 2.3 will handle ASR correction)**
     - Examples: "animal plant expression" → "animal plant expression [unclear]"
     - Examples: "mucosmoburn" → "mucosmoburn [unclear]"
     - Examples: "cortex capsule" → "cortex capsule [unclear]" (if garbled)
     - The [unclear] tag is a flag for Phase 2.3 to apply ASR correction/normalization
   - Do not change one medical term into another (e.g., do not turn "scabies" into "allergy" or "plant" into "anal" or "animal plant expression" into "anal gland expression") in this step. That will be handled later by Phase 2.3 (Grounding Layer).
   - Do not correct drug names, procedure names, or anatomical terms even if they seem wrong (e.g., do not change "Cortex" to "Coatex", "mucosmoburn" to "mucous membranes", or "animal plant" to "anal gland").
   - CRITICAL: Preserve all numerical values, dosages, medications, and clinical instructions exactly as spoken. Do not summarize the text. Do not convert numbers to shorthand.

b. Language inconsistencies:
   - Analyze the transcript for any non English (code-mixed) words or phrases. Some phrases may be in a local language (such as Telugu, Hindi, or Tamil) within an otherwise English conversation. Or it can be in a pure local language.

c. Accurately translate any non-English (code-mixed) words or phrases or sentences or entire conversation into natural, professional clear, contextually appropriate English.
   - Ensure all non-English segments are translated without changing meaning, timeline, or polarity (positive vs negative finding).

d. Remove ONLY:
   - Repeated words (only if truly redundant, not if part of clinical instruction)
   - Filler sounds ("uh", "hmm", "um", "like")
   - Obvious background chatter and system noise
   - DO NOT remove any clinical content, even if it seems redundant

e. Preserve Clinical Content:
   - Do not remove phrases that might be the reason for visit, a diagnosis, a symptom, a procedure, or a treatment, even if the wording is odd or partially wrong.
   - **MANDATORY: If such a phrase is unclear or misrecognized, preserve it verbatim and append [unclear] tag**
     - Examples: "animal plant expression" → "animal plant expression [unclear]"
     - Examples: "mucosmoburn" → "mucosmoburn [unclear]"
     - This signals Phase 2.3 that ASR correction/normalization is needed

f. Grammar and Spelling:
   - Correct only obvious grammar or spelling issues that do not affect clinical meaning (e.g., "teh" → "the").
   - Do NOT change clinical terminology, even if it seems misspelled or phonetically similar to another term.
   - Examples of what NOT to change (but DO tag with [unclear] if garbled):
     * "animal plant expression" → "animal plant expression [unclear]" (preserve + tag)
     * "Cortex" → "Cortex [unclear]" (if garbled, preserve + tag)
     * "mucosmoburn" → "mucosmoburn [unclear]" (preserve + tag)
     * "Nutrish" → "Nutrish [unclear]" (if garbled, preserve + tag)
   - These corrections will be handled by Phase 2.3 (Grounding Layer) using Local Inventory, Global KB, and LLM Judge.
   - The [unclear] tag helps Phase 2.3 identify which terms need normalization/ASR correction

g. STRICT PRESERVATION CHECKLIST:
   - ✓ All numerical values preserved (10, 2, 5, etc.)
   - ✓ All units preserved (mg, ml, kg, %, tablets, capsules)
   - ✓ All dosages preserved with context ("ten milligrams", "two capsules", "once a day")
   - ✓ All anatomical descriptors preserved ("anal gland", "right cranial", "dental")
   - ✓ All procedure names preserved ("anal gland expression", "orthopedic examination")
   - ✓ All brand names preserved ("Cortex", "Coatex", "Nutrish")
   - ✓ All routes of administration preserved (BID, TID, SID, orally, subcutaneously)

Rewrite the conversation as a clear, concise conversational transcript, preserving ALL clinical content exactly as spoken, including all specifics, numbers, units, and anatomical/procedure details. 
Step 2: Speaker Attribution:
Purpose: The cleaned up transcript in step 1, has to be made into a properly attributed vet and pet parent conversation. Identify and clearly attribute each sentence or statement to the correct speaker (e.g., Veterinarian: or Pet Parent) and if mentioned to the correct pet.
Actions:
a. Segment the Transcript by Statements:
• Divide the cleaned transcript into individual sentences or utterances based on context and punctuation.
b. Identify the Speaker for Each Statement:
• Use explicit cues from the transcript (e.g., Thank you, doctor → Pet Parent; Let's examine Buddy → Veterinarian).
• When speaker is ambiguous, infer from context (e.g., clinical advice or diagnosis is usually the Veterinarian; descriptions of symptoms or history are usually the Pet Parent).
c. Attribute Each Statement Clearly:
• Prefix each statement with the correct speaker label, e.g.:
o Veterinarian:
o Pet Parent:
• Optionally, reference the pet by name if there's more than one animal or it adds clarity (e.g., regarding Buddy).
d. Maintain Conversational Flow:
• Present the dialogue in the order it occurred, ensuring a clear back-and-forth between speakers.
e. Correct and Clarify Attributions:
• If speaker switches mid-paragraph, split into separate lines.
• If a statement refers to both the pet and the parent, clarify (e.g., Pet Parent (regarding Buddy): ...).
f. Final Proofread:
• Ensure each statement is attributed, there are no ambiguous lines, and all sentences are clear and professional.
Step 3: Verification of Cleaned Transcript
Input:
• RAW TRANSCRIPTION: {conversation}
Purpose:
After completing Steps 1 and 2, verify that your cleaned and attributed transcript is faithful and complete before it is used for SOAP.

Critical Rules - VERBATIM CLINICAL PRESERVATION:
• Do not introduce any new symptoms, diagnoses, exposures, medications, doses, timelines, or pets that are not present in the raw transcription.
• Do not upgrade uncertainty (e.g., maybe, I think, it looks like) into definite statements.
• Do not remove explicit negatives (e.g., no vomiting, not eating) or change their meaning.
• Do NOT summarize or shorten clinical instructions - preserve them verbatim.
• Do NOT remove anatomical descriptors from procedures (e.g., "anal gland" from "anal gland expression").
• Do NOT convert dosages to shorthand (e.g., "ten milligrams" must remain "ten milligrams", not "10mg").

Actions:
a. Alignment / Faithfulness Check
• For every sentence in your cleaned transcript, verify that you can find the supporting words or phrases in the RAW transcription (same meaning, even if rephrased).
• If a sentence is not clearly supported by the RAW transcription, mark it as UNSUPPORTED and remove or rewrite it so it only reflects what was actually said.
• CRITICAL: Verify that all specific procedures, anatomical locations, and dosages are preserved exactly as spoken.

b. Missing Clinical Content Check
• Scan the RAW transcription to ensure no clinically relevant information was lost in your cleaned version:
  o Symptoms and duration
  o Appetite, water intake, urination/defecation
  o Previous treatments, drugs, doses, allergies
  o Vet's assessments, test names, plan, follow-up.
  o Specific procedures with anatomical descriptors (e.g., "anal gland expression", not just "expression")
  o Complete dosage instructions (e.g., "ten milligrams once a day for five days", not shortened)
• If something important is missing in your cleaned transcript, add it back only by paraphrasing the original words, maintaining the same speaker.

c. Negation and Certainty Check
• Verify that no / not / never / nothing / negative statements remain negative in your cleaned transcript.
• Ensure expressions like maybe, possibly, I think, looks like remain qualified; do not convert them into firm diagnoses.

d. Translation Check (for code-mixed / non-English)
• Confirm all non-English segments are translated into natural, professional English without changing meaning, timeline, or polarity (positive vs negative finding).
• If the original is unclear or partially audible, mark as [unclear] or [inaudible] instead of guessing.

e. Clinical Specificity Check (NEW - CRITICAL)
• Verify that all specific procedures are preserved with their anatomical descriptors:
  - "anal gland expression" → MUST remain "anal gland expression" (not "expression" or "gland expression")
  - "orthopedic examination" → MUST remain "orthopedic examination" (not "examination")
  - "dental scaling" → MUST remain "dental scaling" (not "scaling")
  - "animal plant expression" → MUST remain "animal plant expression" (do NOT correct to "anal gland expression" - Phase 2.3 will handle this)
• Verify that all dosages are preserved in full:
  - "ten milligrams" → MUST remain "ten milligrams" (not "10mg" or "mg")
  - "two capsules" → MUST remain "two capsules" (not "2 caps" or "capsules")
  - "once a day for five days" → MUST remain "once a day for five days" (not "SID x5d")
• Verify that all brand names are preserved exactly as spoken:
  - "Cortex" → MUST remain "Cortex" (do NOT change to "Coatex" - Phase 2.3 Grounding Layer will handle this)
  - "Nutrish" → MUST remain "Nutrish" (do NOT change to "Nutrich" - Phase 2.3 will handle this)
  - "mucosmoburn" → MUST remain "mucosmoburn" (do NOT change to "mucous membranes" - Phase 2.3 will handle this)
• **CRITICAL: If a clinical term is garbled or unclear, preserve it verbatim and mark as [unclear] tag**
  - Examples: "animal plant expression" → "animal plant expression [unclear]"
  - Examples: "mucosmoburn" → "mucosmoburn [unclear]"
  - Examples: "cortex capsule" → "cortex capsule [unclear]" (if garbled)
  - The [unclear] tag is a flag for Phase 2.3 (Grounding Layer) to apply ASR correction/normalization
  - Phase 2.3 will use Local Inventory, Global KB, and LLM Judge to correct these tagged terms

f. Speaker Attribution Consistency
• Re-check that every line in your cleaned transcript has the correct speaker (Veterinarian / Pet Parent).
• History and concerns → usually Pet Parent; explanations, diagnosis, plan → usually Veterinarian.
• Fix any misattributions you detect.

g. Final Checklist
Before finalizing the transcript, ask:
• Is every line traceable to the raw transcription?
• Have I avoided adding any new clinical facts or stronger interpretations?
• Are all important clinical details from the raw conversation present somewhere in the cleaned transcript?
• Have I preserved all specific procedures with their anatomical descriptors?
• Have I preserved all dosages in their full spoken form (not converted to shorthand)?
• Have I preserved all brand names exactly as spoken?
Step 4: Final Output
The final output has to be a clear transcription conversation between the vet and pet parent, which can be used for downstream processing for the SOAP notes.
**OUTPUT FORMAT:**
Provide ONLY the cleaned and attributed conversation. Do not include any explanations, headers, or additional text. Just the conversation with proper speaker attribution.

The output should be in the format:
Veterinarian: [cleaned and attributed text]
Pet Parent: [cleaned and attributed text]
[Continue with the full conversation...]

**End of Phase 0**
"""

# Anchor Mapping (UUID Anchor) instruction block - inserted when ANCHOR_MAPPING_SOAP=true and manifest has anchor_id.
# SOAP generator must wrap entity mentions as [[anchor_id:display_text]] so linking is deterministic (no brittle string search).
# Examples use generic placeholders (EX, EY) to avoid the model copying hardcoded IDs that may not exist in the manifest.
ANCHOR_MAPPING_SOAP_INSTRUCTION = """
**ANCHOR MAPPING (UUID ANCHOR) PROTOCOL - MANDATORY:**
- Each entity in GROUNDING_MANIFEST_JSON has an **anchor_id** (e.g. E1, E2, E3). Use the exact ID provided **only from that manifest**.
- **MANDATORY: Use ONLY anchor_ids that appear in the GROUNDING_MANIFEST_JSON above. Do not use IDs from previous examples or invent new numbers.** The only valid anchor_ids for this note are exactly those listed in that JSON (e.g. if the manifest has E1, E2, E3, E4, E5, then you may only use E1–E5; never use E6, E15, or any other ID not in the manifest). If a concept is not in the manifest, do not wrap it in an anchor tag (use plain display text).
- **Anchor IDs follow transcript order:** The first entity in the manifest has the lowest ID (E1); the next has E2, and so on. Use the exact anchor_id from the manifest for each concept.
- **When you mention any medical entity that IS listed in the manifest, you MUST wrap it in anchor tags** in this format: [[anchor_id:display_text]]
- Examples (use the actual IDs from your manifest, not these placeholders):
  * If the manifest has an entity with anchor_id EX for "Ortolani test", write: "We performed an [[EX:Ortolani test]]."
  * If the manifest has anchor_id EY for a medication, write: "Prescribed [[EY:medication name]]."
  * For signalment: if the manifest assigns E1 to the patient name and E2 to age, write "[[E1:patient name]], a [[E2:age]]" using those exact IDs from the manifest.
- **CRITICAL:** Use the exact anchor_id from the manifest (E1, E2, ...). Do not invent IDs. Any medical term from the manifest that you mention MUST be wrapped as [[anchor_id:term]]. Terms NOT in the manifest must not use anchor tags.
- The display_text inside the tag should be the canonical or normalized term you would use in the note (e.g. "Ortolani test", "Spirocoxib", "hip dysplasia").

"""

SOAP_PROMPT_TEMPLATE = """
ROLE: Senior Veterinary Clinician
**INPUT 1 (CLEANED & REFINED TRANSCRIPT):**
{raw_transcript}
{optional_inputs}

[GROUNDING_MANIFEST_JSON]
{entity_manifest_json}
[/GROUNDING_MANIFEST_JSON]

{anchor_mapping_instruction}

**OPTIONAL CONTEXT (Phase 1):**
The {optional_inputs} section above may also include:

- PriorVisitSummary: A short structured summary of the last visit.
- ActiveProblemList: list of existing diagnoses / chronic issues.

**IMPORTANT CONTEXT NOTES:**
- This transcript has been cleaned (translated, de-noised, clinically refined) and may still include some ASR/phonetic errors; use the Grounding Manifest to resolve medical nouns
- Review the provided transcript carefully for medical information
- Extract only information that is directly supported by the conversation
- Focus on veterinary terminology and clinical details

**GROUNDING MANIFEST (KB-LINKED ENTITIES - NORMALIZED):**
- The GROUNDING_MANIFEST_JSON section above contains entities extracted from the transcript and normalized using KB concepts.
- Each entity has been validated by a Contextual Judge and includes:
  * concept_id: KB concept ID (if linked)
  * concept_preferred_name: **CANONICAL TERM** - You MUST use this in your SOAP note
  * concept_kind: KB kind (Condition, Drug, Symptom, etc.)
  * normalized_term: The normalized term (if different from surface_text)
  
- **CRITICAL: You MUST use the concept_preferred_name from GROUNDING_MANIFEST_JSON whenever you refer to the corresponding concept.**
  * Example: If transcript mentions "metacam" and GROUNDING_MANIFEST_JSON shows concept_preferred_name: "Meloxicam", you MUST write "Meloxicam" in the Plan section.
  * Example: If transcript mentions "Simparico" and GROUNDING_MANIFEST_JSON shows concept_preferred_name: "Simparica", you MUST write "Simparica" in the Assessment/Plan.
  * Do NOT use the surface_text from transcript if a concept_preferred_name is available.
  
- For Assessment / Plan / DifferentialDiagnosis: 
  * **ALWAYS use concept_preferred_name** if concept_id is present (entity was successfully linked and normalized).
  * For unlinked entities (no concept_id), use the surface_text or a vaguer description.
  
- You MUST NOT invent new diagnoses/conditions/drugs beyond what is in the conversation and/or GROUNDING_MANIFEST_JSON.
- If you think a detected concept is wrong or uncertain, mark it as uncertain in the Assessment rather than silently changing it.
- Signalment information (species, breed, sex, age) from GROUNDING_MANIFEST_JSON should be used to inform context but should not be added to SOAP sections unless explicitly mentioned in the conversation.

**ASSERTION HANDLING (CRITICAL - MUST SYNC WITH PHASE B):**
- Each entity in GROUNDING_MANIFEST_JSON includes an assertion_id field that indicates the clinical assertion type:
  * **CONF** (Confirmed): Current or planned issue/action. Include in SOAP note as confirmed/present.
  * **NEG** (Negated): Explicitly NOT present. DO NOT include in SOAP note, or explicitly state as "denied" or "not present".
  * **SUSP** (Suspected): Rule-out/differential. Include in Assessment as suspected/possible, not confirmed.
  * **HIST** (Historical): Past medical history. Include in Subjective/History section, NOT in current Assessment/Plan.
  * **HYPO** (Hypothetical): Future option/discussion only. Include in Plan as "discussed" or "may consider", NOT as current action.
  * **RECUR** (Recurring): Ongoing maintenance/preventive. Include in Plan as ongoing/recurring treatment.

- **CRITICAL RULES:**
  * If assertion_id is **NEG**: Do NOT write the entity as present. Either omit it or explicitly state "no [entity]" or "denies [entity]".
  * If assertion_id is **SUSP**: Write as "possible [entity]" or "rule out [entity]" in Assessment, NOT as confirmed diagnosis.
  * If assertion_id is **HIST**: Write in Subjective/History section only, NOT in current Assessment or Plan.
  * If assertion_id is **HYPO**: Write in Plan as "discussed [entity]" or "may consider [entity]", NOT as current prescription/action.
  * If assertion_id is **RECUR**: Write in Plan as "ongoing [entity]" or "continue [entity]", indicating maintenance/recurring nature.
  * If assertion_id is **CONF**: Write normally as confirmed/present entity.

- **Examples:**
  * Entity: "vomiting", assertion_id: "NEG" → Write "No vomiting" or "Denies vomiting", NOT "Vomiting present"
  * Entity: "infection", assertion_id: "SUSP" → Write "Possible infection" or "Rule out infection", NOT "Infection confirmed"
  * Entity: "diabetes", assertion_id: "HIST" → Write in History section: "History of diabetes", NOT in current Assessment
  * Entity: "surgery", assertion_id: "HYPO" → Write in Plan: "Discussed surgery" or "May consider surgery if needed", NOT "Surgery scheduled"
  * Entity: "Meloxicam", assertion_id: "CONF" → Write normally: "Prescribed Meloxicam" or "Administered Meloxicam"

- **SYNC REQUIREMENT**: Your SOAP note MUST respect the assertion_id from GROUNDING_MANIFEST_JSON. Do NOT contradict Phase B's assertion classification.

**SECTION-SPECIFIC MANIFEST ENFORCEMENT:**

The GROUNDING_MANIFEST_JSON is the intent/anchor entity manifest (entities may have anchor_id e.g. E1, E2). In every section, consider the manifest as a checklist so no relevant entity is left out or missed. When an entity has an anchor_id, use the format [[anchor_id:display_text]] when you mention it in any section (Subjective, Objective, Assessment, Plan, etc.) so downstream replacement and injection can match reliably. Grounding may map E-IDs to KB concept_id; use concept_preferred_name when available.

**HALLUCINATION GUARD:** Only tag entities that appear in GROUNDING_MANIFEST_JSON with [[anchor_id:...]]. If a clinical fact is in the transcript but NOT in the manifest (e.g. exercise restriction, swimming advice), write it as plain text without an anchor tag. Do not invent an E-ID for concepts not listed in the manifest.

### MANDATORY ENTITY MAPPING RULES

You must integrate the [GROUNDING_MANIFEST_JSON] into the SOAP narrative based on these functional roles. Failure to include these specific entities is a clinical error:

1. **SUBJECTIVE (Presentation & History):**
   - **Role: `PresentingRequest`**
   - MUST be the anchor of the first paragraph. 
   - *Instruction:* "The patient presented for [display_name]." (e.g., ANAL SAC EXPRESSION GROOMING).
   - This is MANDATORY - the Subjective section MUST begin with the presenting complaint from the manifest.
   - **Manifest checklist:** Consider the manifest for Subjective: ensure every entity relevant to presentation or history (e.g. PresentingRequest, HIST, owner-reported symptoms) is reflected so nothing is left out.
   - **Anchor IDs:** When you mention any entity from the manifest here, use [[anchor_id:display_text]] (e.g. [[E1:Oreo]], [[E5:walking problem]]) so replacement and injection work in every section. If no anchor_id, use display_name or concept_preferred_name without a tag.

2. **PLAN (Actions & Orders):**
   - **PLAN MUST USE MANIFEST AS CHECKLIST (NON-NEGOTIABLE):** Use GROUNDING_MANIFEST_JSON as a checklist (not as the sole source of the Plan). When writing the Plan, ensure every entity whose role is Prescribed, Administered, Performed, Planned, or RECUR—or whose kind/context indicates plan-relevant action—is included: at least one bullet or one sentence for EACH. Add narrative and explanatory prose as needed; the checklist ensures nothing is left or missed out. You are NOT allowed to omit an entity because it seems "preventive", "secondary", or "less important".
   - **Anchor IDs:** When you mention any entity from the manifest in the Plan, use [[anchor_id:display_text]] (e.g. [[E8:Meloxicam]], [[E12:Bravecto]]) so replacement and injection work in every section. Use the exact anchor_id from the manifest. If no anchor_id, use display_name or concept_preferred_name without a tag.
   - **Roles: `Prescribed`, `Administered`, `Performed`, `Planned`** (and RECUR for ongoing/preventive).
   - *Administered:* Items given in the clinic (e.g., vaccines, injections).
   - *Performed:* Procedures done during the visit (e.g., Anal Gland Expression, Nail Trim).
   - *Prescribed:* Medications sent home (e.g., COATEX capsules).
   - *Planned:* Future follow-ups or tests.
   - Use the `display_name` (or concept_preferred_name) from the manifest for each entity.
   - **ENTITY-INJECTED PLAN (CRITICAL):** You MUST use the specific medication names and procedures identified in the manifest to populate the PLAN (and SUBJECTIVE where relevant). Do NOT use generic placeholders like "as prescribed" or "as discussed" if a specific product (e.g. Bravecto, Spirocoxin, Contraway) or procedure is present in the manifest. Name the product or procedure explicitly in the Plan so the note is clinically precise and the verification dashboard can align with the narrative.
   - **PARASITE CONTROL / PREVENTIVES (CRITICAL):** If the conversation mentions tick/flea/parasite control (or the vet recommends or prescribes such products), and the manifest contains any parasite-control or preventive product names (e.g. Bravecto, Simparica, topical products, dewormers), you MUST list them in the Plan. Do not omit them because they are "preventive" or "secondary." Example: "Tick and flea control: [product name from manifest] recommended/prescribed." Include every such product from the manifest that was discussed or prescribed.

3. **OBJECTIVE (Findings):**
   - **Role: `Finding` or `VitalSign`**
   - All physical observations (e.g., "mucous membranes," "lymph nodes") must be described here using their `display_name` from the manifest.
   - Do not use generic terms - use the exact `display_name` provided.
   - **Manifest checklist:** Consider the manifest for Objective: ensure every Finding and VitalSign entity is reflected so nothing is left out.
   - **Anchor IDs:** When you mention any entity from the manifest here, use [[anchor_id:display_text]] using only IDs present in GROUNDING_MANIFEST_JSON (e.g. if E3 is in the manifest for a test, write [[E3:test name]]). Do NOT use any E-ID that is not listed in the manifest. If no anchor_id, use display_name without a tag.

4. **ASSESSMENT (Conditions & Differentials):**
   - **Manifest checklist:** Consider the manifest for Assessment: ensure every condition/differential entity (e.g. DiagnosisSuspected, SUSP, CONF) is reflected so nothing is left out. Use concept_preferred_name when available.
   - **Anchor IDs:** When you mention any entity from the manifest here, use [[anchor_id:display_text]] using only IDs present in GROUNDING_MANIFEST_JSON (e.g. if the manifest has anchor_id EX for a condition, write [[EX:condition name]]). Do not invent or reuse IDs. If no anchor_id, use display_name or concept_preferred_name without a tag.

**CRITICAL INSTRUCTION:** Do not summarize. If the manifest contains a specific procedure like "ANAL SAC EXPRESSION GROOMING," you are forbidden from writing "The animal was seen for a routine visit." Use the exact terms provided in the manifest's `display_name` field.

Role:
You are a knowledgeable veterinary assistant responsible for converting a pre-appointment complaint summary and a cleaned, attributed vet-pet parent conversation into a professional SOAP note format.
Your task is to understand the transcript of a vet and a pet owner conversation and other relevant information from the provided materials and organize it into the SOAP notes (Subjective, Objective, Assessment, Plan) structure, ensuring accuracy and professionalism.
Utilize a structured, step by step approach thought process and the SOAP note components to ensure all relevant medical details are accurately documented. Consider the pre-appointment complaint summary, if available, as a context for the conversation. Also consider any treatment protocol if available as a template, if relevant. If both the conversation and the complaint summary lack medically relevant content, return a null SOAP note.
Global Rule: At no point may you introduce new clinical facts, diagnoses, exposures, medications, or timelines that are not explicitly present in the pre-appointment summary, protocols, vitals, conversation, or GROUNDING_MANIFEST_JSON. You must use the detected concepts from GROUNDING_MANIFEST_JSON to ground your Assessment, Plan, and DifferentialDiagnosis sections. If a concept in GROUNDING_MANIFEST_JSON has high confidence (>= 0.8) and a concept_id, use the canonical KB name. For low-confidence or unlinked concepts, use the surface_text or a vaguer description.

THE GOLDEN RULE (DATA FIDELITY): Maintain strict fidelity to all measurements, dosages, and routes mentioned in the transcript. Never summarize '5mg IV' into 'administered medication'. If the doctor said it, it must appear in the SOAP note exactly as spoken.

Step 1: Initial Content Verification
Purpose: Determine if the transcription contains medically relevant information.
Actions:
a. Analyze the conversation:
• Review the doctor-patient conversation for medical history, symptoms, assessments, diagnoses, treatments, or any other medically relevant content.
b. Decision Point:
• If medically relevant content is present in the source:
  ▪ Proceed to Step 2.
• If medically relevant content is absent in the source:
  ▪ Output a null SOAP note with [Null] in each section.
  ▪ Include a statement: No medically relevant information was provided in all the sections of the SOAP.
c. Exclusion:
• Do not consider the pre-appointment complaint summary for this step. Focus only on the transcription provided, to determine whether medical relevant content is available.
• Remove any content that is not directly relevant to veterinary care, such as personal anecdotes, emotional conversations, entertainment, or social plans. Only retain text that clearly relates to a pet's health, symptoms, medical history, diagnosis, treatment, or care. No need to give any explanation in each of the sections as to the nature of the conversation. Simply output that no medically relevant information is available.
Step 2: Use the Pre-Appointment Complaint Summary, if available:
Where the pre-appointment complaint summary is provided,
a. Integrate as Context:
• Use the pre-appointment complaint summary to provide essential background and context in the SOAP note.
b. Highlight Key Information:
• Pay special attention to initial symptoms, concerns, and observations noted by the pet owner before the appointment.
c. Ensure Continuity:
• Reference the pre-appointment information throughout the SOAP note where relevant, especially in the Subjective section.
d. PriorVisitSummary and ActiveProblemList:
• Where a PriorVisitSummary or ActiveProblemList is provided, use it only as background context for understanding the current visit. Do not copy new diagnoses or treatments directly into the SOAP unless they are mentioned or clearly referenced in the current conversation.
Step 3: Use the Treatment Protocol as a template, if available
a. Check whether there is a treatment protocol given as an input:
• If available, a treatment protocol would be given as an input.
• The treatment protocol can be a pre-filled completed treatment protocol or the protocol adherence has to be filled in from the conversation.
b. Compare with the conversation:
• Where the treatment protocol adherence has not been pre-filled in, check against each of the items in the treatment protocol and fill in against each of the questions in the treatment protocol the adherence aspect of it.
c. Use Treatment Protocol as a template reference:
• The protocol has to serve as a template within the SOAP notes and has to be appropriately referenced across all appropriate sections of the SOAP notes.
Purpose:
• The treatment protocol serves as a template and an adherence checklist for the veterinarian during the consultation.
• The protocol can be in the form of different type of questions and the answers can also be defined to be in a specific format.
• Ensure that the format is followed.
• Give an explanation for the adherence, where appropriate, along with an explanation of how the adherence has been derived from the conversation.
Step 4: Subjective Information Extraction
Purpose: Identify and extract subjective information provided by the pet owner from both the pre-appointment complaint summary and the conversation.
Actions:
a. From Pre-Appointment Complaint Summary:
• Extract the pet owner's reported symptoms, concerns, and observations.
• Note any relevant history or changes in behavior.
b. From Doctor-Patient Conversation:
• Identify additional subjective information shared during the consultation.
• Include any clarifications or new symptoms mentioned.
c. From Treatment protocol form:
• Extract the relevant treatment protocol questions and responses relevant to the subjective section.
d. Combine Information:
• Integrate data from the above sources into a cohesive Subjective section.
• Ensure all information is relevant to the current condition.
e. Include: Owner's observations, reported symptoms, duration of symptoms, changes in behavior or appetite.
f. Exclude: Casual remarks, unrelated anecdotes, non-medical conversations.
g. If no medically relevant information is available, output only no medically relevant information available.

**MANDATORY LEAD-IN RULE:**
The Subjective section MUST begin with the patient's presenting complaint or the reason for the visit including the patient's past medical history. This is the anchor of the Subjective section and must be the first paragraph.
Step 5: Objective Data Identification
Purpose: Identify and extract objective findings observed by the veterinarian during the examination.
Actions:
a. Extract from Conversation:
• Note physical examination findings, vital signs, diagnostic test results, and observable clinical signs.
• Use precise medical terminology.
b. Include Measurable Data:
• Record specific metrics such as temperature, heart rate, weight, etc., as provided.
• Include vital signs recorded separately and given as an input to the model.
c. Include the relevant treatment protocol questions and responses relevant to the Objective section.
d. Include any vitals template provided and match the results against the template.
e. Include: Physical examination findings, vital signs, laboratory results, imaging findings.
f. Exclude: Veterinarian's small talk, non-clinical observations.
g. If no medically relevant information is available, output only no medically relevant information available.
Step 6: Assessment Extraction
Purpose: Identify and extract the Assessment findings from the conversation.
Actions:
a. Analyze the Conversation:
• Analyze the veterinarian's discussions and explanations.
• Pay attention to the veterinarian's observations, hypotheses, or considerations regarding the pet's condition.
b. Identify Assessments:
• Extract assessments that are explicitly mentioned through the veterinarian's comments or discussions with the pet owner.
• Only include assessments that are directly supported by the veterinarian's statements. Do not add interpretations or diagnoses not mentioned in the conversation.
• Ensure that the assessment is directly supported by the information in the conversation.
c. Avoid Adding New Information:
• Do not introduce assessments that are not supported by the conversation.
• Do not make medical judgments beyond what is discussed.
• Do not imply or attribute fictional assessments.
d. Assessment Section:
• Based on a, b, and c above, summarize the assessment, ensuring it reflects the veterinarian's conclusions extracted from the conversation.
• There can be multiple key issues identified in the SOAP notes and for each of those key issues there could be a linked assessment. Bring those key issues and linked assessment separately, if available in the conversation.
e. Avoid adding medical judgements:
• Avoid adding medical judgments, even if they seem obvious or reasonable, unless explicitly mentioned. Follow section c with avoiding adding new information in the assessment step, vigorously.
f. If no medically relevant information is available, output only no medically relevant information available.
Step 7: Plan of Action
Purpose: Extract the recommended plan for diagnostics, treatment, prescriptions and follow-ups and owner instructions based on the conversation and summarize the recommended plan for diagnostics, treatment, and follow-up and owner instructions as discussed in the conversation.
Actions:
a. Document all Recommendations:
• Include treatments, medications, therapies, interventions, diagnostic tests as discussed in the conversation.
• Should include both administered treatments or medication done at the clinic level as well as all medicines prescribed.
• Should also include services or procedures performed.
• Should also include any diagnostic tests conducted during the consultation or scheduled.
• Should also include any follow-ups or reminders or any other aspects to be done in the future course of time.
b. Provide Owner Instructions:
• Extract all care instructions given to the pet owner.
c. Ensure Accuracy:
• Only include actions explicitly mentioned in the conversation.
d. Provide assessment and plan linkages, if relevant and available:
• Include and link the key assessments with the plan of actions. Categorize the plan of actions into categories, if relevant:
  ▪ Diagnostic: Actions to confirm or rule out a condition (e.g., blood tests, imaging).
  ▪ Therapeutic: Actions to treat or manage a condition (e.g., medication, surgery).
  ▪ Preventive: Actions to prevent the onset or progression of a condition (e.g., vaccinations, diet modifications).
  ▪ Monitoring: Actions to observe and track the progress of a condition (e.g., regular check-ups, symptom tracking).
e. If no medically relevant information is available, output only no medically relevant information available.
Step 8: Conclusion and Summary
Purpose:
Craft a cohesive, clinically structured summary that encapsulates the entire SOAP note. This section should allow a veterinarian to understand the full case—presenting complaint, findings, diagnoses, treatments, and follow-up—without needing to review the full SOAP documentation. It should serve as the first reference point for case recall and can be adapted as the basis of a professional summary for clients.
a. Summarize Key Case Elements:
• Begin with a concise restatement of the initial complaint and history provided by the owner, using only information already present in the Subjective section.
• Clearly summarize physical findings, diagnostics (e.g., cytology), and any abnormal results, using only information already present in the Objective section.
• Include a brief rationale behind the assessment or differential diagnosis, but only if this rationale is explicitly discussed in the conversation or pre-appointment summary. Do not invent or infer new reasoning.
• Summarize treatments administered and medications prescribed, focusing on therapeutic intent, using only items already present in the Plan and Protocols/Vitals sections.
• Incorporate owner instructions and follow-up plans (e.g., recheck appointments, monitoring guidance) using only items already present in the Plan, Customer Instructions, Reminders, and Key Issues.
• Any statement about overall status (e.g., stable, improving, deteriorating, recovering, guarded prognosis) must be a direct paraphrase of wording already used in the Assessment or in the veterinarian’s own statements. If such wording is not clearly present, omit that status label instead of guessing.
• Ensure every sentence in the Conclusion can be traced back to specific content in Subjective, Objective, Assessment, Plan, or other structured sections above.
b. Maintain Clinical Structure and Professional Tone:
• Use formal, concise medical language suitable for veterinary case documentation.
• Follow a logical flow: Presentation → Findings → Assessment/Diagnosis → Treatment → Owner Instructions → Follow-up.
• Ensure the summary is stand-alone, medically informative, and easily scannable by a veterinary professional.
• Since it is the vet who is going to pursue their own records, don't use tone and terminology saying the veterinarian chose to do this or that. Rather use wording which is appropriate for a clinical record.
c. If no medically relevant information is available, output only no medically relevant information available. No need to describe or explain the nature of the non-medical relevant information. Instead simply output the result that no medically relevant information was available.
Step 9: Final Differential Diagnosis Heading
Purpose:
Provide a structured Differential Diagnosis heading that summarizes the main condition or system involved, even if the veterinarian does not explicitly say it as a heading.
Rules:
• The DifferentialDiagnosis must not introduce any new condition that is not already mentioned or clearly implied in the Assessment / Plan / conversation.
• It is only a relabelling/structuring of existing information, not a new diagnosis.
If no explicit condition is mentioned, identify the body system from the primary symptoms described and Only use 'Unknown-Not specified if symptoms are so vague or mixed that no single system can be reasonably identified.
Actions:
a. Source for Differential Diagnosis:
• Look only at the conditions, problems, or body systems already mentioned in the Assessment and Plan sections and in the veterinarian’s own words in the conversation.
b. Select the Term:
• If a specific condition is named (e.g., otitis externa, cranial cruciate rupture), map it to System-Condition (e.g., Dermatology-Otitis Externa, Orthopedic-Cruciate Rupture).
• If only a system-level problem is mentioned (e.g., spinal problem, skin issues), use a broad label such as Neurology-Spinal Problem (unspecified) or Dermatology-Skin Problem (unspecified).
• If only symptoms are described and no clear condition/system can be identified without guessing, use: Unknown-Not specified.
c. Avoid Adding New Information:
• Do not invent a more specific condition than what is stated. (Example: if conversation only says back pain, do NOT write Neurology-Spinal Cord Injury.)
• Do not introduce any system or condition not supported by the conversation.
d. Differential Diagnosis Section:
• Output exactly one heading in the format System-Condition (e.g., Gastrointestinal-Gastroenteritis.
Step 10: Customer Instructions Extraction
Purpose: Extract all the customer instructions given by the veterinarian to the customer following the treatment.
Actions:
a. Document all Instructions:
• Include all customer instructions, recommendations given by the doctor, medical dosage instructions and advice, any specific care instructions, follow-up recommendations and next visit dates as discussed in the conversation.
b. Include any customer instructions from the treatment protocol, if relevant and available.
c. Ensure Accuracy:
• Only include actions explicitly mentioned in the conversation.
Step 11: Protocols Section
Purpose: Extract all the responses against the protocol template from the conversation. Even though they might be part of the other sections in the SOAP note, consolidate all the protocol template and responses in this step.
Actions:
a. Document all responses:
• Pre-fill the protocol template against the unfilled aspects in the template. This has to be done strictly from the conversation.
• The template to be filled in will also have a response format. Follow them strictly. If the template does not have a response format, fill it up in the appropriate response format.
b. Leave the template as it is, without making any changes, even if there are null fields after filling it up from the conversation.
c. Ensure Accuracy:
• Only include items explicitly mentioned in the conversation.
Step 12: Vitals Section
Purpose: Extract all the vital signs mentioned in the conversation mainly from the objective section of the SOAP notes. If a vitals template, including pre-filled template, is given, retain it as it is and fill up the null values if available from the conversation.
Actions:
a. Document all observations:
• Pre-fill the vital sign template, against the unfilled aspects in the template. This has to be done strictly from the conversation.
• The template has to be filled-in strictly in the given format.
b. Leave the template as it is, without making any changes, even if there are null fields after filling it up from the conversation.
c. Ensure Accuracy:
• Only include items explicitly mentioned in the conversation.
Step 13: Reminders and Follow-up Section
Purpose: Extract all the reminders, follow-ups as well any future course of actions to be done by the veterinarian or by the pet owner following the present consultation. Extract it and consolidate those aspects from the conversation.
Actions:
a. Document all reminders and follow-ups:
• Include all follow-up recommendations and next visit dates, future diagnostic tests to be conducted, future surgeries planned or any other future course of action along with remarks against those as discussed in the conversation.
• Follow-ups can be for the next scheduled physical visit or virtual visit or home visit or follow up for a diagnostic test to be done or a monitoring to be done.
• Reminders can be about diagnostics tests, medications as well as future course of action or any other reminders either for the patient or for the doctor.
b. Include any instructions from the treatment protocol, if relevant and available.
c. Ensure Accuracy:
• Only include actions explicitly mentioned in the conversation.
Step 14: Key Issues Section
Purpose: Extract all the primary key clinical issues that were discussed in the conversation. Extract it and consolidate those aspects from the conversation.
Actions:
a. Document all key issues:
• Include all the key issues that were discussed during the conversation with a remarks summary.
b. Only include actions explicitly mentioned in the conversation.
c. Ensure Accuracy.
d. The key issues have to be clinically relevant and independent on their own, rather than signalment or single symptoms of the primary issues. If there are secondary issues, mention it separately as secondary issues linked to the primary issue against each primary issue.
e. The remarks can be similar to the overall SOAP note conclusion, but should be specific to the key issues, against which the remarks are mentioned.
Step 15: Abnormal Findings
Purpose: Identify any and all abnormal findings in the conversation or filled up vitals/protocols.
Actions:
a. Document all Abnormal findings:
• Include all abnormal findings discussed in the conversation.
• Abnormal findings can relate with reference to the issue at hand, any diagnostic test results discussed and any other aspects.
b. Ensure Accuracy.
c. Only include items explicitly mentioned in the conversation.
Step 16: Verification Step
Review for Accuracy and Relevance
Purpose: Ensure no invented or inferred content has been added.
a. Review for Accuracy and Source Faithfulness
• Before finalizing the SOAP note, carefully review all extracted information against the pre-appointment summary and the conversation.
• Ask yourself:
  o Can I point to the exact sentence(s) in the source that support this line?
  o Have I accidentally strengthened, expanded, or interpreted anything beyond what was actually said?
b. Review for Medical Relevance
• Does each piece of information directly relate to the pet’s medical condition, history, physical findings, tests, or plan?
• Have I excluded all non-medical and irrelevant content (small talk, greetings, billing admin, etc.)?
c. Make Necessary Corrections
• If any information is not clearly supported by the source, remove it or weaken it (e.g., possible instead of a definitive diagnosis), or mark as Unknown/Not mentioned.
• If any clinically important detail from the source is missing, add it only by copying/rephrasing, not by guessing.
• Ensure that only pertinent medical information is included in each section.


Step 18: Final Review and Compliance Check
Purpose: Ensure the SOAP note is faithful, complete, and follows all guidelines.
a. Verify Source Alignment
• Confirm that every sentence in the SOAP note is directly derived from the pre-appointment complaint summary and/or the conversation.
• The model must not introduce new diagnoses, exposures, medications, or timelines that are not explicitly present in the source unless it is an allowed exception.
b. Check Terminology and Formatting
• Use correct medical terminology only when it clearly matches what was described (e.g., vomiting for threw up).
• Follow the specified output format precisely (section names, JSON/field structure, etc.).
• If there are words or sentences which are not meaningful or unclear or ambiguous due to transmission or transcription losses, handle it in the following way:
  ▪ Transcription Errors and Colloquialisms
    1. Single-Word Errors
       • If the meaning is obvious from context, correct the word in the final notes.
       • Example: The transcription says, liver muscle, but you're certain (based on context) the veterinarian said liver mass.
    2. Colloquial Expressions
       • Rephrase in Professional Terms.
       • When a pet owner uses colloquial or layman's language (e.g., He's going to the toilet well), you may restate it with standard veterinary terminology in the SOAP note (e.g., normal bowel movements or defecating normally).
d. Maintain Professionalism
• Ensure the document reflects professional veterinary standards: clear, neutral, concise, no speculative language presented as fact.
• If something is clinically important but uncertain, label it as unclear / unspecified / not mentioned rather than guessing.
Step 19: Final Instructions for Phase 1
Ensure the SOAP note is accurate, complete, and adheres to all guidelines.
Actions:
a. Verify All Sections:
• Confirm that all information is directly derived from the pre-appointment complaint summary, treatment protocol and the conversation.
b. Check Terminology and Formatting:
• Use correct medical terminology.
• Follow the specified output format precisely.
c. Maintain Professionalism:
• Ensure the document reflects professional veterinary standards.
Step 20: Output Format
Before generating the final SOAP note, ensure that no part of the transcript or summary contains personal, emotional, or social content unrelated to the health or care of the pet. If no veterinary content is found, return a JSON object with all sections as empty strings, but maintain the proper JSON structure.
**CRITICAL: You MUST always return a valid JSON object, even if no medically relevant content is found.**

**IMPORTANT OUTPUT INSTRUCTIONS:**
1. The main SOAP note output should be a JSON object with ONLY the SOAP sections (Subjective, Objective, Assessment, Plan, etc.).
2. **DO NOT include the Transcript field** - the transcript is saved separately and should not be part of the SOAP note JSON.
3. Format the output as follows:

**MAIN SOAP NOTE JSON (first):**
{{
  Subjective: Subjective content here,
  Objective: Objective content here,
  Assessment: Assessment content here,
  Plan: Plan content here,
  Conclusion: Conclusion content here,
  DifferentialDiagnosis: Differential Diagnosis content here,
  KeyIssues: Key Issues content here,
  AbnormalFindings: Abnormal Findings content here,
  CustomerInstructions: Customer Instructions content here,
  Protocols: Protocols content here,
  Vitals: Vitals content here,
  Reminders: Reminders content here
}}

**IMPORTANT:** Do NOT include the Transcript field in the JSON output. The transcript is saved separately and should not be part of the SOAP note.

Step 21: Final Reminders
a. Assessment:
• Avoid adding medical judgements:
• Avoid adding medical judgments, even if they seem obvious or reasonable, unless explicitly mentioned. Do not add new information or make medical judgments beyond the conversation.
• In the Conclusion, do not use overall status labels such as stable, gradual recovery, deteriorating, etc., unless they are a direct paraphrase of wording already used by the veterinarian in the Assessment or conversation (e.g., getting better compared to yesterday).
b. Differential Diagnosis:
• Extract it only from conditions/systems already present in the Assessment / Plan or veterinarian’s statements.
• Do not add new information or make medical judgments beyond the conversation.
• If nothing is clearly supported, use: Unknown-Not specified.
c. Avoid Hallucinations:
• Do not introduce information not present in the pre-appointment complaint summary or conversation.
d. Professional Tone:
• Use formal language and correct medical terminology.
e. Consistency:
• Ensure that all sections align with the extracted information.
f. Mandatory Fields:
• Conclusion and Differential Diagnosis are mandatory. Differential Diagnosis should be in the format System-Condition.
• For example: Dermatology-Atopic Dermatitis.
**End of Phase 1**
"""

# Fireworks Structured Outputs for SOAP generation
# JSON schema: https://docs.fireworks.ai/structured-responses/structured-response-formatting
# Grammar mode (alternative): https://docs.fireworks.ai/structured-responses/structured-output-grammar-based
# Per Fireworks docs: include JSON instruction in prompt AND response_format; if finish_reason=="length", increase max_tokens.
SOAP_NOTE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "Subjective": {"type": "string"},
        "Objective": {"type": "string"},
        "Assessment": {"type": "string"},
        "Plan": {"type": "string"},
        "Conclusion": {"type": "string"},
        "DifferentialDiagnosis": {"type": "string"},
        "KeyIssues": {"type": "string"},
        "AbnormalFindings": {"type": "string"},
        "CustomerInstructions": {"type": "string"},
        "Protocols": {"type": "string"},
        "Vitals": {"type": "string"},
        "Reminders": {"type": "string"},
    },
    "required": [
        "Subjective",
        "Objective",
        "Assessment",
        "Plan",
        "Conclusion",
        "DifferentialDiagnosis",
        "KeyIssues",
        "AbnormalFindings",
        "CustomerInstructions",
        "Protocols",
        "Vitals",
        "Reminders",
    ],
    "additionalProperties": False,
}

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Set up comprehensive logging configuration."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('soap_generator')
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    # File handler for detailed logs
    log_file = Path(output_dir) / f"soap_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger





# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_api_keys_from_env() -> Dict[str, str]:
    """Build API keys dict from environment variables (fallback when no file)."""
    aliases = {
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "OPENAI_KEY": "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY": "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY": "MISTRAL_API_KEY",
        "FIREWORKS_API_KEY": "FIREWORKS_API_KEY",
        "FIREWORKS_API": "FIREWORKS_API_KEY",
    }
    keys = {}
    for env_name, env_value in os.environ.items():
        norm = aliases.get((env_name or "").strip().replace("-", "_").upper())
        if not norm:
            continue
        v = (env_value or "").strip()
        if v:
            keys[norm] = v
    return keys


def load_api_keys(api_key_file: str) -> Dict[str, str]:
    """
    Load API keys from file. If file does not exist, fall back to environment variables.
    
    Args:
        api_key_file: Path to file containing API keys in KEY=VALUE format
        
    Returns:
        Dictionary of API keys
        
    Raises:
        ValueError: If neither file nor env provides any keys
    """
    aliases = {
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "OPENAI_KEY": "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY": "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY": "MISTRAL_API_KEY",
        "FIREWORKS_API_KEY": "FIREWORKS_API_KEY",
        "FIREWORKS_API": "FIREWORKS_API_KEY",
    }

    def _normalize(raw_key: str) -> str:
        return aliases.get((raw_key or "").strip().replace("-", "_").upper(), "")

    def _read_file(path: str) -> Dict[str, str]:
        file_keys: Dict[str, str] = {}
        if not path or not os.path.exists(path):
            return file_keys
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    norm = _normalize(key)
                    val = value.strip().strip("\"'")
                    if norm and val:
                        file_keys[norm] = val
                elif line.startswith("fw_"):
                    file_keys["FIREWORKS_API_KEY"] = line
                else:
                    logging.warning(f"Invalid format in API key file line {line_num}: {line}")
        return file_keys

    keys: Dict[str, str] = {}
    try:
        # Merge API_Key.txt + fireworks_api.txt regardless of where each key is stored.
        for path in (api_key_file, FIREWORKS_API_KEY_FILE):
            parsed = _read_file(path)
            for k, v in parsed.items():
                if v and (k not in keys or not (keys.get(k) or "").strip()):
                    keys[k] = v
    except Exception as e:
        raise ValueError(f"Error reading API key file(s): {e}")

    if keys:
        env_keys = load_api_keys_from_env()
        for k, v in env_keys.items():
            if v and (k not in keys or not (keys.get(k) or "").strip()):
                keys[k] = v
        return keys
    
    # Fallback: load from environment (allows running without API_Key.txt when env is set)
    keys = load_api_keys_from_env()
    if keys:
        logging.info(
            "API key file not found at %s; using API keys from environment variables.",
            api_key_file,
        )
        return keys
    raise FileNotFoundError(
        f"API key file not found: {api_key_file}. "
        "Create the file with KEY=VALUE lines (e.g. FIREWORKS_API_KEY=fw_xxx), or set FIREWORKS_API_KEY (and others) in the environment."
    )


def read_transcription(file_path: str) -> str:
    """
    Read and validate transcribed conversation file.
    
    Args:
        file_path: Path to transcription file
        
    Returns:
        Transcription content
        
    Raises:
        FileNotFoundError: If transcription file doesn't exist
        ValueError: If transcription file is empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcription file not found: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        raise IOError(f"Error reading file {file_path}: {e}")
    
    if not content:
        raise ValueError("Transcription file is empty")
    
    return content


def save_output(content: str, file_path: str, logger: logging.Logger) -> bool:
    """
    Save content to file with error handling.
    
    Args:
        content: Content to save
        file_path: Output file path
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # CRITICAL: Validate content is not empty
        if not content or not content.strip():
            logger.error(f"❌ Cannot save empty content to {file_path}")
            logger.error(f"   Content length: {len(content) if content else 0} characters")
            return False
        
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Output saved successfully: {file_path} ({len(content)} characters)")
        return True
    except Exception as e:
        logger.error(f"Error saving output to {file_path}: {e}")
        return False


# Lexical Harvester functions removed - no longer used in pipeline


def read_optional_file(file_path: str, file_description: str, logger: logging.Logger) -> str:
    """
    Read an optional input file safely.
    
    Args:
        file_path: Path to the optional file
        file_description: Description of the file for logging
        logger: Logger instance
        
    Returns:
        File content as string, or empty string if file doesn't exist or path is None
    """
    if not file_path:
        logger.info(f"{file_description} not provided (path is None)")
        return ""
    
    if not os.path.exists(file_path):
        logger.warning(f"{file_description} file not found: {file_path}")
        return ""
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        logger.info(f"{file_description} loaded successfully: {len(content)} characters")
        return content
    except Exception as e:
        logger.error(f"Error reading {file_description} file {file_path}: {e}")
        return ""


def read_multiple_files_parallel(file_configs: list, logger: logging.Logger) -> dict:
    """
    Read multiple optional files in parallel for faster I/O.
    
    Args:
        file_configs: List of tuples (file_path, file_description)
        logger: Logger instance
        
    Returns:
        Dictionary mapping descriptions to file contents
    """
    results = {}
    
    def read_single_file(config):
        file_path, description = config
        return description, read_optional_file(file_path, description, logger)
    
    # Use ThreadPoolExecutor for parallel file reading
    if ThreadPoolExecutor is None:
        # Fallback to sequential reading if ThreadPoolExecutor is not available
        for file_path, description in file_configs:
            results[description] = read_optional_file(file_path, description, logger)
    else:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(read_single_file, config) for config in file_configs]
            for future in futures:
                description, content = future.result()
                results[description] = content
    
    return results


def calculate_dynamic_max_tokens(conversation: str, min_tokens: int = 2000, max_tokens: Optional[int] = None) -> int:
    """
    Calculate dynamic max_tokens based on conversation length.
    
    Heuristic: SOAP notes are typically 2-3x the conversation length in tokens,
    with a buffer for all sections (Subjective, Objective, Assessment, Plan, etc.)
    
    Args:
        conversation: The cleaned conversation text
        min_tokens: Minimum tokens to ensure basic SOAP sections (default: 2000)
        max_tokens: Maximum tokens to prevent excessive costs (default: from OPTIMIZED_CONFIG, 2500 when TARGET_60S else 6000)
        
    Returns:
        Calculated max_tokens value
    """
    if max_tokens is None:
        max_tokens = OPTIMIZED_CONFIG["max_tokens"]
    # Estimate conversation tokens (rough: 1 token ≈ 4 characters)
    conversation_tokens = len(conversation) // 4
    
    # Calculate: 2.5x conversation tokens + 30% buffer for all SOAP sections
    calculated_tokens = int(conversation_tokens * 2.5 * 1.3)
    
    # Apply min/max bounds
    dynamic_tokens = max(min_tokens, min(calculated_tokens, max_tokens))
    
    return dynamic_tokens


def _prepare_source_transcript_for_soap_prompt(source_transcript: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Prevents context overflow on long transcripts.

    - If transcript contains a [LONG_TRANSCRIPT_SUMMARY] block (from chunked super-pass),
      the prompt will use that summary + prompt-safe excerpts.
    - Otherwise, if transcript exceeds SOAP_MAX_TRANSCRIPT_CHARS_IN_PROMPT, it will be truncated (head+tail).
    """
    t = source_transcript or ""
    try:
        max_chars = int(os.getenv("SOAP_MAX_TRANSCRIPT_CHARS_IN_PROMPT", "22000"))
    except Exception:
        max_chars = 22000

    # Prefer summary block if present
    if extract_summary_block and build_prompt_safe_transcript:
        summary, remainder = extract_summary_block(t)
        if summary:
            try:
                excerpt_max = int(os.getenv("SOAP_EXCERPT_MAX_CHARS", "12000"))
            except Exception:
                excerpt_max = 12000
            excerpts = build_prompt_safe_transcript(remainder, max_chars=min(excerpt_max, max_chars))
            if logger:
                logger.info(
                    f"🧠 Long transcript prompt mode: using rolling summary + excerpts "
                    f"(summary={len(summary)} chars, excerpts={len(excerpts)} chars)"
                )
            return (
                "LONG TRANSCRIPT SUMMARY (from chunked super-pass):\n"
                f"{summary}\n\n"
                "EXCERPTS FROM CLEANED TRANSCRIPT (truncated for prompt safety):\n"
                f"{excerpts}"
            ).strip()

    # Fallback: truncate very long transcripts
    if build_prompt_safe_transcript and len(t) > max_chars:
        if logger:
            logger.info(f"✂️  Transcript too long for prompt ({len(t)} chars) → truncating to ~{max_chars} chars")
        return build_prompt_safe_transcript(t, max_chars=max_chars)

    return t


def build_optional_inputs_section(pre_appointment: str, protocols: str, vitals: str) -> str:
    """
    Build a section containing all optional inputs.
    
    Args:
        pre_appointment: Pre-appointment summary content
        protocols: Protocols template content  
        vitals: Vitals template content
        
    Returns:
        Combined optional inputs section
    """
    sections = []
    
    if pre_appointment:
        sections.append(f"**PRE-APPOINTMENT CONSULTATION SUMMARY:**\n{pre_appointment}")
    
    if protocols:
        sections.append(f"**PROTOCOLS TEMPLATE:**\n{protocols}")
    
    if vitals:
        sections.append(f"**VITALS TEMPLATE:**\n{vitals}")
    
    if sections:
        return "\n\n".join(sections) + "\n"
    else:
        return ""


def build_soap_prompt_from_brain_ner(raw_transcript: str, optional_inputs: str, brain_ner_json: str) -> str:
    """
    Build SOAP prompt using Brain NER handoff only (no grounded-manifest injection/anchors).
    """
    return (
        "ROLE: Senior Veterinary Clinician\n"
        "**INPUT 1 (CLEANED & REFINED TRANSCRIPT):**\n"
        f"{raw_transcript}\n"
        f"{optional_inputs}\n"
        "[BRAIN_NER_JSON]\n"
        f"{brain_ner_json}\n"
        "[/BRAIN_NER_JSON]\n\n"
        "Use BRAIN_NER_JSON as a clinical checklist.\n"
        "- Prefer `search_term` / `normalized_name` from Brain NER for medical wording.\n"
        "- Use `suggestion_probability` and `correctness_probability` as confidence signals.\n"
        "- **Hints and query_expansion:** Each entity may include `hints` (clinical clues, e.g. brand or product terms) and `query_expansion` (likely brand/product names for phonetic or ASR-mangled mentions). When writing the SOAP note, treat these as the intended clinical meaning: prefer wording that aligns with `normalized_name` and with any non-empty hints/query_expansion (e.g. if span_text is \"exotic pump\" and query_expansion includes \"Easotic\", write \"Easotic\" or \"Easotic ear drops\" in the note, not the raw transcript phrase).\n"
        "- Respect `assertion_id` semantics (CONF/NEG/SUSP/HIST/HYPO/RECUR).\n"
        "- Keep SOAP clean text only: DO NOT output anchor tags or injected markup.\n"
        "- Do not invent facts beyond transcript + optional inputs + BRAIN_NER_JSON.\n\n"
        "Return ONLY valid JSON matching the SOAP schema.\n"
    )


def build_detected_concepts_json_from_manifest(entity_manifest: list, logger: Optional[logging.Logger] = None) -> str:
    """
    Build SOAP handoff payload from Brain NER entities.

    IMPORTANT:
    - This payload is intentionally Brain-NER-centric (not grounded-manifest-centric).
    - SOAP uses normalized_name/search_term with probability signals directly.
    """
    if not entity_manifest:
        return ""

    payload = {
        "brain_ner_entities": [],
    }

    for entity in entity_manifest:
        if not isinstance(entity, dict):
            continue
        qe = entity.get("query_expansion")
        if isinstance(qe, list):
            query_expansion = [str(x).strip() for x in qe if str(x).strip()][:3]
        elif isinstance(qe, str) and qe.strip():
            query_expansion = [x.strip() for x in qe.split(",") if x.strip()][:3]
        else:
            query_expansion = []
        entry = {
            "entity_id": entity.get("entity_id"),
            "span_text": entity.get("span_text", ""),
            "kind": entity.get("kind") or entity.get("kb_kind") or "Other",
            "normalized_name": entity.get("normalized_name", ""),
            "search_term": entity.get("search_term") or entity.get("normalized_name") or entity.get("span_text") or "",
            "correctness_probability": entity.get("correctness_probability"),
            "suggestion_probability": entity.get("suggestion_probability"),
            "hints": entity.get("hints") or [],
            "query_expansion": query_expansion,
            "assertion_id": entity.get("assertion_id", "CONF"),
            "inventory_category": entity.get("inventory_category") or [],
            "service_category": entity.get("service_category") or [],
            "attributes": entity.get("attributes") or {},
        }
        payload["brain_ner_entities"].append(entry)

    detected_concepts_json = json.dumps(payload, indent=2, ensure_ascii=False)
    if logger:
        logger.info("📊 Built Brain NER SOAP payload (%d entities)", len(payload["brain_ner_entities"]))
    return detected_concepts_json


def enforce_dual_sync_reject_hard_stop(entity_manifest: list, logger: Optional[logging.Logger] = None) -> None:
    """
    Hard-stop propagation rule:
    if match_method=dual_sync_judge_rejected, local IDs must be NULL everywhere in manifest payload.
    """
    if not isinstance(entity_manifest, list):
        return
    fixed = 0
    for e in entity_manifest:
        if not isinstance(e, dict):
            continue
        if (e.get("match_method") or "").strip().lower() != "dual_sync_judge_rejected":
            continue
        if e.get("local_stock_id") is not None or e.get("local_service_id") is not None:
            fixed += 1
        e["local_stock_id"] = None
        e["local_service_id"] = None
        codes = e.get("codes")
        if isinstance(codes, dict):
            codes.pop("local_stock_id", None)
            codes.pop("local_service_id", None)
    if fixed and logger:
        logger.info("🛡️ Hard-stop applied: cleared local IDs on %d dual_sync_judge_rejected entities", fixed)

def repair_json_simple(json_str: str) -> str:
    """
    Simple JSON repair for common issues (fallback if json-repair not available).
    Fixes trailing commas, missing quotes, and unclosed braces.
    """
    import re
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # Fix unclosed strings (add closing quote if missing)
    # This is a simple heuristic - may not catch all cases
    lines = json_str.split('\n')
    repaired_lines = []
    for line in lines:
        # Count unescaped quotes
        quote_count = len(re.findall(r'(?<!\\)"', line))
        if quote_count % 2 != 0 and not line.strip().endswith('"'):
            # Odd number of quotes - might be unclosed
            if ':' in line and not line.strip().endswith(','):
                line = line.rstrip() + '"'
        repaired_lines.append(line)
    
    return '\n'.join(repaired_lines)


# SOAP section keys (same order as schema) for plain-text parsing
_SOAP_SECTION_KEYS = [
    "Subjective", "Objective", "Assessment", "Plan", "Conclusion",
    "DifferentialDiagnosis", "KeyIssues", "AbnormalFindings", "CustomerInstructions",
    "Protocols", "Vitals", "Reminders",
]


def _parse_plain_text_soap_sections(raw: str, logger: Optional[logging.Logger] = None) -> Optional[dict]:
    """
    Parse plain-text SOAP (e.g. "Subjective: ...\\nObjective: ...") into a dict.
    Used when the SOAP model (e.g. Fireworks Llama) returns prose instead of JSON
    despite response_format=json_schema (provider may not enforce it strictly).
    """
    if not raw or not raw.strip():
        return None
    out = {k: "" for k in _SOAP_SECTION_KEYS}
    # Section headers: "Subjective:", "Objective:", etc. (case-insensitive)
    pattern = re.compile(
        r"^\s*(" + "|".join(re.escape(k) for k in _SOAP_SECTION_KEYS) + r")\s*:\s*",
        re.IGNORECASE | re.MULTILINE,
    )
    parts = pattern.split(raw.strip())
    if len(parts) < 2:
        # No section headers; treat whole text as Subjective
        out["Subjective"] = raw.strip()[:10000]
        return out
    # parts[0] is text before first header (often empty); then [header1, content1, header2, content2, ...]
    for i in range(1, len(parts) - 1, 2):
        key_raw = parts[i].strip()
        value = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Next section starts at next header; value is everything until then
        next_header = pattern.search(value) if value else None
        if next_header:
            value = value[: next_header.start()].strip()
        for k in _SOAP_SECTION_KEYS:
            if key_raw.lower() == k.lower():
                out[k] = value[:10000]
                break
    return out


def extract_soap_json(raw_response: str, logger: Optional[logging.Logger] = None) -> Optional[dict]:
    """
    Production-grade SOAP JSON extraction with header-based detection.
    
    Implements all 6 fixes:
    1. Header-based start detection (looks for "Subjective", "Objective", etc.)
    2. Aggressive reasoning text removal
    3. Stateful brace matching (handles nested objects and escaped quotes)
    4. JSON repair logic (with fallback if json-repair not available)
    5. Enhanced error logging
    6. Response format enforcement (handled in API call)
    
    Args:
        raw_response: Raw LLM response text
        logger: Optional logger instance
        
    Returns:
        Parsed JSON dict or None if extraction fails
    """
    if not raw_response or not raw_response.strip():
        if logger:
            logger.warning("⚠️  Empty response received")
        return None
    
    # Try to import json-repair (optional dependency)
    try:
        from json_repair import repair_json
        has_json_repair = True
    except ImportError:
        has_json_repair = False
        if logger:
            logger.debug("json-repair not available, using simple repair fallback")
    
    # 5. Enhanced Error Logging: Preview the raw response
    if logger:
        preview = raw_response[:300] if len(raw_response) > 300 else raw_response
        logger.debug(f"📝 Raw LLM Response Preview: {preview}...")
    
    # 1 & 2. Header-Based Start Detection & Reasoning Removal
    # Look for the first '{' that is followed by a known SOAP key to avoid garbage starts
    start_indicators = ['"Subjective"', '"Objective"', '"Assessment"', '"Plan"', 
                        '"subjective"', '"objective"', '"assessment"', '"plan"']
    start_index = -1
    
    # Attempt to find a valid JSON object start by looking for SOAP section headers
    for indicator in start_indicators:
        # Look for pattern: { "Subjective" or { "subjective" (with optional whitespace)
        pattern = r'\{\s*' + re.escape(indicator)
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if match:
            start_index = match.start()
            if logger:
                logger.debug(f"✅ Found valid JSON start at position {start_index} (indicator: {indicator})")
            break
    
    if start_index == -1:
        # Fallback: Look for any { followed by a quote (valid JSON property start)
        for i in range(len(raw_response) - 1):
            if raw_response[i] == '{':
                next_chars = raw_response[i+1:].lstrip()
                if next_chars.startswith('"') or next_chars.startswith("'"):
                    start_index = i
                    if logger:
                        logger.debug(f"✅ Found JSON start at position {start_index} (fallback: quote after brace)")
                    break
    
    if start_index == -1:
        # Last resort: Find the very first brace
        start_index = raw_response.find('{')
        if start_index != -1 and logger:
            logger.warning(f"⚠️  Using first brace at position {start_index} (no header indicators found)")
    
    if start_index == -1:
        # Model returned plain text (e.g. "Subjective: ... Objective: ...") instead of JSON.
        # Common when using Fireworks Llama: json_schema may not be enforced as strictly as OpenAI.
        plain_dict = _parse_plain_text_soap_sections(raw_response, logger=logger)
        if plain_dict and any(plain_dict.get(k) for k in plain_dict):
            if logger:
                logger.info("✅ Converted plain-text SOAP (no JSON) to JSON for downstream")
            return plain_dict
        if logger:
            logger.warning("⚠️ No JSON object start found in response; using default SOAP structure")
        default_soap_note = {
            "Subjective": "",
            "Objective": "",
            "Assessment": "",
            "Plan": "",
            "Conclusion": "",
            "DifferentialDiagnosis": "",
            "KeyIssues": "",
            "AbnormalFindings": "",
            "CustomerInstructions": "",
            "Protocols": "",
            "Vitals": "",
            "Reminders": ""
        }
        return default_soap_note
    
    # Remove any reasoning text before the start
    if start_index > 0:
        skipped_text = raw_response[:start_index].strip()
        if logger:
            logger.debug(f"📋 Skipped {start_index} chars of reasoning text: '{skipped_text[:100]}...'")
    
    # 3. Improved Brace Matching (Stateful Parser)
    # This handles nested objects and escaped quotes correctly
    content = raw_response[start_index:]
    stack = []
    end_index = -1
    in_string = False
    is_escaped = False
    
    for i, char in enumerate(content):
        if char == '\\' and not is_escaped:
            is_escaped = True
            continue
        
        if char == '"' and not is_escaped:
            in_string = not in_string
        
        if not in_string:
            if char == '{':
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:  # All braces matched
                        end_index = i + 1
                        break
        
        is_escaped = False
    
    if end_index == -1:
        if logger:
            logger.warning("⚠️  No matching closing brace found. Attempting repair on truncated string.")
        json_str = content  # Pass to repair logic anyway
    else:
        json_str = content[:end_index]
        if logger:
            logger.debug(f"✅ Extracted JSON string ({len(json_str)} chars)")
    
    # 4. JSON Repair Logic
    try:
        # Try standard loading first
        parsed = json.loads(json_str)
        if logger:
            logger.debug("✅ Standard JSON parse succeeded")
        return parsed
    except json.JSONDecodeError as e:
        if logger:
            logger.warning(f"⚠️  Standard JSON parse failed: {e}. Attempting repair...")
        
        try:
            if has_json_repair:
                # Use json-repair library if available
                repaired = repair_json(json_str, return_objects=True)
                if repaired:
                    if logger:
                        logger.info("✅ JSON successfully repaired using json-repair library")
                    return repaired
            else:
                # Use simple repair fallback
                repaired_str = repair_json_simple(json_str)
                try:
                    repaired = json.loads(repaired_str)
                    if logger:
                        logger.info("✅ JSON successfully repaired using simple repair")
                    return repaired
                except json.JSONDecodeError:
                    pass
        except Exception as repair_err:
            if logger:
                logger.error(f"❌ JSON repair failed: {repair_err}")
    
    # Last resort: Try to extract the largest valid JSON structure
    if logger:
        logger.warning("⚠️  All repair attempts failed. Trying to extract largest valid JSON structure...")
    
    # Try to find the largest JSON-like structure
    brace_start = json_str.find('{')
    if brace_start != -1:
        last_brace = json_str.rfind('}')
        if last_brace > brace_start:
            potential_json = json_str[brace_start:last_brace + 1]
            # Try one more repair pass
            try:
                if has_json_repair:
                    repaired = repair_json(potential_json, return_objects=True)
                    if repaired:
                        if logger:
                            logger.info("✅ Extracted JSON using last-resort method with json-repair")
                        return repaired
                else:
                    repaired_str = repair_json_simple(potential_json)
                    repaired = json.loads(repaired_str)
                    if logger:
                        logger.info("✅ Extracted JSON using last-resort method with simple repair")
                    return repaired
            except Exception:
                pass
    
    if logger:
        logger.error(f"❌ Could not extract valid JSON from response")
        logger.debug(f"   Extracted string preview: {json_str[:500]}...")
        logger.warning("⚠️  Creating default SOAP note structure as final fallback")
    
    # Final fallback: Create default SOAP note structure to prevent pipeline failure
    # This ensures the pipeline continues even if JSON extraction completely fails
    default_soap_note = {
        "Subjective": "",
        "Objective": "",
        "Assessment": "",
        "Plan": "",
        "Conclusion": "",
        "DifferentialDiagnosis": "",
        "KeyIssues": "",
        "AbnormalFindings": "",
        "CustomerInstructions": "",
        "Protocols": "",
        "Vitals": "",
        "Reminders": ""
    }
    
    if logger:
        logger.info("✅ Default SOAP note structure created as fallback")
    
    return default_soap_note


def extract_json_from_llm_response(response_text: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Legacy function for backward compatibility.
    Now delegates to extract_soap_json and converts dict to JSON string.
    
    Returns:
        Cleaned JSON string (or default SOAP structure if extraction fails)
    """
    # Use the new production-grade extraction
    parsed = extract_soap_json(response_text, logger=logger)
    
    if parsed:
        # Convert dict back to JSON string
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    
    # This should not happen as extract_soap_json now returns a default structure
    # But keep this fallback for safety
    if logger:
        logger.warning("⚠️  extract_soap_json returned None (unexpected) - creating default structure")
    
    # Create default structure as emergency fallback
    default_soap_note = {
        "Subjective": "",
        "Objective": "",
        "Assessment": "",
        "Plan": "",
        "Conclusion": "",
        "DifferentialDiagnosis": "",
        "KeyIssues": "",
        "AbnormalFindings": "",
        "CustomerInstructions": "",
        "Protocols": "",
        "Vitals": "",
        "Reminders": ""
    }
    
    return json.dumps(default_soap_note, indent=2, ensure_ascii=False)


def extract_soap_sections(soap_note: str) -> dict:
    """
    Extract DifferentialDiagnosis, Conclusion, and KeyIssues sections from the SOAP note.
    Returns a dict with keys 'DifferentialDiagnosis', 'Conclusion', and 'KeyIssues'.
    """
    sections = {}
    patterns = {
        "DifferentialDiagnosis": r"DifferentialDiagnosis:\s*(.*?)(?:\n[A-Z][a-zA-Z ]+:|\Z)",
        "Conclusion": r"Conclusion:\s*(.*?)(?:\n[A-Z][a-zA-Z ]+:|\Z)",
        "KeyIssues": r"Key Issues?:\s*(.*?)(?:\n[A-Z][a-zA-Z ]+:|\Z)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, soap_note, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()
        else:
            sections[key] = ""
    return sections


def store_soap_embeddings_in_database(soap_note: str, case_id: str = None) -> bool:
    """
    Store SOAP note embeddings in the clinical database.
    Extracts differential diagnosis, key issues, and conclusion sections and stores their embeddings.
    """
    try:
        # Check if sql_embeddings functionality is available
        if not setup_openai or not get_embedding or not connect_to_postgres:
            logger.error("SQL embeddings functionality not available. Please ensure sql_embeddings.py is accessible.")
            return False
            
        # Setup OpenAI client
        client = setup_openai()
        if not client:
            logger.error("Failed to setup OpenAI client for database embeddings")
            return False
        
        # Connect to database
        conn = connect_to_postgres()
        if not conn:
            logger.error("Failed to connect to database for embeddings")
            return False
        
        # Extract sections from SOAP note
        sections = extract_soap_sections(soap_note)
        
        # Prepare texts for batch embedding generation
        section_names = []
        section_texts = []
        for section_name, text in sections.items():
            if text and text.strip():
                section_names.append(section_name)
                section_texts.append(text)
        
        # Generate embeddings in batch (single API call)
        embeddings = []
        if section_texts:
            try:
                # Try to use batch embedding if available in sql_embeddings
                # Otherwise, fall back to individual calls
                if hasattr(client, 'embeddings') and hasattr(client.embeddings, 'create'):
                    # Use OpenAI client directly for batch
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=section_texts
                    )
                    embeddings = [item.embedding for item in response.data]
                else:
                    # Fallback: use get_embedding in batch if it supports it
                    # Otherwise, make individual calls (old behavior)
                    for text in section_texts:
                        embedding = get_embedding(client, text)
                        if embedding:
                            embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"⚠️  Batch embedding failed, falling back to individual calls: {e}")
                embeddings = []
                for text in section_texts:
                    embedding = get_embedding(client, text)
                    if embedding:
                        embeddings.append(embedding)
        
        # Store embeddings in database
        with conn.cursor() as cursor:
            # Map section names to database columns
            vector_column_map = {
                "DifferentialDiagnosis": "differential_diagnosis_vector",
                "Conclusion": "soap_summary_vector",  # Using soap_summary_vector for conclusion
                "KeyIssues": "key_issues_vector"
            }
            
            # Generate case_id if not provided
            if not case_id:
                case_id = f"SOAP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            for i, section_name in enumerate(section_names):
                if i < len(embeddings) and embeddings[i]:
                    embedding = embeddings[i]
                    vector_column = vector_column_map.get(section_name)
                    if vector_column:
                        # Convert embedding to PostgreSQL vector format
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                        
                        # Insert or update the embedding
                        cursor.execute(f"""
                            INSERT INTO clinical (case_id, differential_diagnosis, "SOAP_summary", key_issues, {vector_column})
                            VALUES (%s, %s, %s, %s, %s::vector)
                            ON CONFLICT (case_id) 
                            DO UPDATE SET 
                                differential_diagnosis = EXCLUDED.differential_diagnosis,
                                "SOAP_summary" = EXCLUDED."SOAP_summary",
                                key_issues = EXCLUDED.key_issues,
                                {vector_column} = EXCLUDED.{vector_column}
                        """, (case_id, 
                              sections.get("DifferentialDiagnosis", ""), 
                              sections.get("Conclusion", ""), 
                              sections.get("KeyIssues", ""),
                              embedding_str))
                        
                        logger.info(f"  ✅ Stored {section_name} embedding in database for case_id: {case_id}")
        
        conn.commit()
        conn.close()
        logger.info("✅ SOAP note embeddings stored in clinical database")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error storing SOAP embeddings in database: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def get_openai_embedding(text: str, api_key: str, model: str = "text-embedding-3-small") -> list:
    """
    Get embedding for the given text using OpenAI's embedding API.
    """
    start_time = time.time()
    
    # Use the already initialized logger
    logger = logging.getLogger('soap_generator')
    
    logger.info(f"🕐 Starting embedding generation at {datetime.now().strftime('%H:%M:%S')}")
    
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": text,
        "task": "search_document"
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    embedding = data["data"][0]["embedding"]
    total_time = time.time() - start_time
    logger.info(f"✅ Embedding generation completed (took {total_time:.2f}s)")
    logger.info(f"📊 Input length: {len(text)} characters")
    logger.info(f"📊 Embedding dimensions: {len(embedding)}")
    
    return embedding


def get_openai_embeddings_batch(texts: List[str], api_key: str, model: str = "text-embedding-3-small") -> List[list]:
    """
    Get embeddings for multiple texts in a single API call using OpenAI's embedding API.
    This is more efficient than calling get_openai_embedding multiple times.
    
    Args:
        texts: List of text strings to generate embeddings for
        api_key: OpenAI API key
        model: Embedding model to use (default: text-embedding-3-small)
    
    Returns:
        List of embeddings, one for each input text (in the same order)
    """
    if not texts:
        return []
    
    start_time = time.time()
    logger = logging.getLogger('soap_generator')
    
    logger.info(f"🕐 Starting batch embedding generation for {len(texts)} texts at {datetime.now().strftime('%H:%M:%S')}")
    
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": texts,  # Pass list of texts for batch processing
        "task": "search_document"
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=60)  # Longer timeout for batch
    response.raise_for_status()
    data = response.json()
    embeddings = [item["embedding"] for item in data["data"]]
    total_chars = sum(len(text) for text in texts)
    total_time = time.time() - start_time
    logger.info(f"✅ Batch embedding generation completed (took {total_time:.2f}s)")
    logger.info(f"📊 Generated {len(embeddings)} embeddings from {len(texts)} texts")
    logger.info(f"📊 Total input length: {total_chars} characters")
    logger.info(f"📊 Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
    
    return embeddings


# ==============================================================================
# LLM INTEGRATION
# ==============================================================================

class LLMProvider:
    """Base class for LLM providers."""
    
    def __init__(self, api_key: str, config: Config, logger: logging.Logger):
        self.api_key = api_key
        self.config = config
        self.logger = logger
    
    def generate_soap_note(self, conversation: str, pre_appointment: str = "", protocols: str = "", vitals: str = "", detected_concepts_json: str = "", raw_transcript: str = "", anchor_mapping_instruction: str = "") -> str:
        """Generate SOAP note from conversation. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def clean_transcription(self, raw_transcription: str) -> str:
        """Clean and attribute the raw transcription (Phase 1). Must be implemented by subclasses."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for SOAP note generation."""
    
    def __init__(self, api_key: str, config: Config, logger: logging.Logger):
        super().__init__(api_key, config, logger)
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def clean_transcription(self, raw_transcription: str) -> str:
        """Clean and attribute the raw transcription (Phase 1)."""
        start_time = time.time()
        self.logger.info(f"🕐 Starting transcription cleaning at {datetime.now().strftime('%H:%M:%S')}")
        
        # Optional inputs are not used in transcription cleaning (Phase 0)
        # They are only used in SOAP note generation (Phase 1)
        prompt = TRANSCRIPTION_CLEANING_PROMPT.format(
            conversation=raw_transcription,
            optional_inputs=""  # Empty for transcription cleaning
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a clinical documentation specialist specializing in cleaning veterinary conversation transcripts."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4000  # Allow more tokens for transcription cleaning
        }
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Cleaning transcription (attempt {attempt + 1}/{self.config.max_retries})")
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    cleaned_transcription = result['choices'][0]['message']['content'].strip()
                    # CRITICAL FIX: Remove [unclear] tags from cleaned transcript
                    cleaned_transcription = cleaned_transcription.replace("[unclear]", "").replace("[unclear ]", "").strip()
                    import re
                    cleaned_transcription = re.sub(r'\s+', ' ', cleaned_transcription)
                    total_time = time.time() - start_time
                    self.logger.info("Transcription cleaned successfully")
                    self.logger.info(f"✅ Transcription cleaning completed (took {total_time:.2f}s)")
                    self.logger.info(f"📊 Input length: {len(raw_transcription)} characters")
                    self.logger.info(f"📊 Output length: {len(cleaned_transcription)} characters")
                    
                    return cleaned_transcription
                else:
                    raise ValueError("No valid response from OpenAI API")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in transcription cleaning: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise RuntimeError("Failed to clean transcription after all retry attempts")
    
    def generate_soap_note(self, conversation: str, pre_appointment: str = "", protocols: str = "", vitals: str = "", detected_concepts_json: str = "", raw_transcript: str = "", anchor_mapping_instruction: str = "") -> str:
        """Generate SOAP note using OpenAI API."""
        start_time = time.time()
        self.logger.info(f"🕐 Starting SOAP note generation at {datetime.now().strftime('%H:%M:%S')}")
        
        source_transcript = raw_transcript or conversation
        prompt_transcript = _prepare_source_transcript_for_soap_prompt(source_transcript, logger=self.logger)

        # Build dynamic prompt sections
        prompt_sections = build_optional_inputs_section(pre_appointment, protocols, vitals)
        
        # Format Brain NER JSON for prompt (or empty string if not provided)
        if not detected_concepts_json:
            detected_concepts_json = "No Brain NER entities available."
        else:
            # PERF: Avoid redundant json.loads->json.dumps churn when detected_concepts_json is already a JSON string.
            # Only serialize if a dict/list was provided.
            try:
                if not isinstance(detected_concepts_json, str):
                    detected_concepts_json = json.dumps(detected_concepts_json, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Use as-is if anything unexpected happens
        
        prompt = build_soap_prompt_from_brain_ner(
            raw_transcript=prompt_transcript,
            optional_inputs=prompt_sections,
            brain_ner_json=detected_concepts_json,
        )
        
        # Calculate dynamic max_tokens based on prompt transcript length (not full raw transcript)
        dynamic_max_tokens = calculate_dynamic_max_tokens(prompt_transcript)
        self.logger.info(f"📊 Transcript tokens (prompt): ~{len(prompt_transcript) // 4}")
        self.logger.info(f"📊 Dynamic max_tokens: {dynamic_max_tokens} (min: 2000, max: 6000)")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # NOTE: Parallel SOAP drafting was intentionally removed to preserve the original single-call flow.
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional veterinary assistant specializing in SOAP note generation. "
                        "Return ONLY valid JSON matching the provided schema. "
                        "Do NOT include reasoning, thinking, markdown, or explanatory text. "
                        "The first non-whitespace character of your response MUST be '{'."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            # Deterministic + structure-friendly settings
            "temperature": 0.0,
            "max_tokens": dynamic_max_tokens
        }

        # Enforce strict JSON schema output when supported (OpenAI Structured Outputs).
        # This dramatically reduces JSON repair/extraction overhead and failure modes.
        soap_structured = os.getenv("SOAP_STRUCTURED_OUTPUT", "true").strip().lower() in ("1", "true", "yes")
        if soap_structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "SoapNote",
                    "schema": SOAP_NOTE_JSON_SCHEMA,
                    "strict": True,
                },
        }
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Sending request to OpenAI (attempt {attempt + 1}/{self.config.max_retries})")
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    raw_response = result['choices'][0]['message']['content'].strip()
                    soap_json_dict = None
                    try:
                        soap_json_dict = json.loads(raw_response)
                    except Exception:
                        soap_json_dict = extract_soap_json(raw_response, logger=self.logger)
                    if not isinstance(soap_json_dict, dict):
                        soap_json_dict = {}
                    for k in SOAP_NOTE_JSON_SCHEMA["required"]:
                        if k not in soap_json_dict or soap_json_dict[k] is None:
                            soap_json_dict[k] = ""
                    soap_note = json.dumps(soap_json_dict, indent=2, ensure_ascii=False)
                    total_time = time.time() - start_time
                    self.logger.info("SOAP note generated successfully")
                    self.logger.info(f"✅ SOAP note generation completed (took {total_time:.2f}s)")
                    self.logger.info(f"📊 Input length: {len(source_transcript)} characters")
                    self.logger.info(f"📊 Output length: {len(soap_note)} characters")
                    
                    return soap_note
                else:
                    raise ValueError("No valid response from OpenAI API")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in OpenAI request: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise RuntimeError("Failed to generate SOAP note after all retry attempts")


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider for SOAP note generation."""
    
    def __init__(self, api_key: str, config: Config, logger: logging.Logger):
        super().__init__(api_key, config, logger)
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def generate_soap_note(self, conversation: str, pre_appointment: str = "", protocols: str = "", vitals: str = "", detected_concepts_json: str = "", raw_transcript: str = "", anchor_mapping_instruction: str = "") -> str:
        """Generate SOAP note using Claude API."""
        source_transcript = raw_transcript or conversation
        prompt_transcript = _prepare_source_transcript_for_soap_prompt(source_transcript, logger=self.logger)

        # Build dynamic prompt sections
        prompt_sections = build_optional_inputs_section(pre_appointment, protocols, vitals)
        
        # Format Brain NER JSON for prompt (or empty string if not provided)
        if not detected_concepts_json:
            detected_concepts_json = "No Brain NER entities available."
        else:
            # PERF: Avoid redundant json.loads->json.dumps churn when detected_concepts_json is already a JSON string.
            # Only serialize if a dict/list was provided.
            try:
                if not isinstance(detected_concepts_json, str):
                    detected_concepts_json = json.dumps(detected_concepts_json, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Use as-is if anything unexpected happens
        
        prompt = build_soap_prompt_from_brain_ner(
            raw_transcript=prompt_transcript,
            optional_inputs=prompt_sections,
            brain_ner_json=detected_concepts_json,
        )
        
        # Calculate dynamic max_tokens based on transcript length
        dynamic_max_tokens = calculate_dynamic_max_tokens(prompt_transcript)
        self.logger.info(f"📊 Transcript tokens (prompt): ~{len(prompt_transcript) // 4}")
        self.logger.info(f"📊 Dynamic max_tokens: {dynamic_max_tokens} (min: 2000, max: 6000)")
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.config.model_name,
            "max_tokens": dynamic_max_tokens,
            "temperature": OPTIMIZED_CONFIG["temperature"],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Sending request to Claude (attempt {attempt + 1}/{self.config.max_retries})")
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                result = response.json()
                if 'content' in result and len(result['content']) > 0:
                    raw_response = result['content'][0]['text'].strip()
                    soap_note = extract_json_from_llm_response(raw_response, logger=self.logger)
                    self.logger.info("SOAP note generated successfully")
                    return soap_note
                else:
                    raise ValueError("No valid response from Claude API")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in Claude request: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise RuntimeError("Failed to generate SOAP note after all retry attempts")


class FireworksProvider(LLMProvider):
    """Fireworks AI provider for SOAP note generation (supports Qwen 3, Qwen 2.5, Llama models)."""
    
    def __init__(self, api_key: str, config: Config, logger: logging.Logger):
        super().__init__(api_key, config, logger)
        # CRITICAL: Use correct base URL without trailing slash to prevent 404 errors
        self.base_url = "https://api.fireworks.ai/inference/v1/chat/completions"
    
    def clean_transcription(self, raw_transcription: str) -> str:
        """Clean and attribute the raw transcription (Phase 1)."""
        start_time = time.time()
        self.logger.info(f"🕐 Starting transcription cleaning at {datetime.now().strftime('%H:%M:%S')}")
        
        prompt = TRANSCRIPTION_CLEANING_PROMPT.format(
            conversation=raw_transcription,
            optional_inputs=""  # Empty for transcription cleaning
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a clinical documentation specialist specializing in cleaning veterinary conversation transcripts."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4000  # Allow more tokens for transcription cleaning
        }
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Cleaning transcription (attempt {attempt + 1}/{self.config.max_retries})")
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    cleaned_transcription = result['choices'][0]['message']['content'].strip()
                    total_time = time.time() - start_time
                    self.logger.info("Transcription cleaned successfully")
                    self.logger.info(f"✅ Transcription cleaning completed (took {total_time:.2f}s)")
                    self.logger.info(f"📊 Input length: {len(raw_transcription)} characters")
                    self.logger.info(f"📊 Output length: {len(cleaned_transcription)} characters")
                    
                    return cleaned_transcription
                else:
                    raise ValueError("No valid response from Fireworks API")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in transcription cleaning: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise RuntimeError("Failed to clean transcription after all retry attempts")
    
    def generate_soap_note(self, conversation: str, pre_appointment: str = "", protocols: str = "", vitals: str = "", detected_concepts_json: str = "", raw_transcript: str = "", anchor_mapping_instruction: str = "") -> str:
        """Generate SOAP note using Fireworks API."""
        start_time = time.time()
        self.logger.info(f"🕐 Starting SOAP note generation at {datetime.now().strftime('%H:%M:%S')}")
        
        source_transcript = raw_transcript or conversation
        prompt_transcript = _prepare_source_transcript_for_soap_prompt(source_transcript, logger=self.logger)

        # Build dynamic prompt sections
        prompt_sections = build_optional_inputs_section(pre_appointment, protocols, vitals)
        
        # Format Brain NER JSON for prompt (or empty string if not provided)
        if not detected_concepts_json:
            detected_concepts_json = "No Brain NER entities available."
        else:
            # PERF: Avoid redundant json.loads->json.dumps churn when detected_concepts_json is already a JSON string.
            # Only serialize if a dict/list was provided.
            try:
                if not isinstance(detected_concepts_json, str):
                    detected_concepts_json = json.dumps(detected_concepts_json, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Use as-is if anything unexpected happens
        
        prompt = build_soap_prompt_from_brain_ner(
            raw_transcript=prompt_transcript,
            optional_inputs=prompt_sections,
            brain_ner_json=detected_concepts_json,
        )
        
        # Calculate dynamic max_tokens based on transcript length
        dynamic_max_tokens = calculate_dynamic_max_tokens(prompt_transcript)
        # Fireworks constraint: some models require stream=true if max_tokens > 4096.
        # Keep non-streaming requests within 4096 for Fireworks models.
        #
        # IMPORTANT: DeepSeek can be verbose with long-form SOAP prompts. If max_tokens is too low,
        # the response can hit the cap and get truncated mid-JSON, causing parse/repair failures.
        # So we enforce a higher floor for Fireworks SOAP generation (still capped at 4096).
        dynamic_max_tokens = max(dynamic_max_tokens, 3200)
        if dynamic_max_tokens > 4096:
            if self.logger:
                self.logger.info(f"⚠️ Fireworks non-streaming max_tokens capped: {dynamic_max_tokens} → 4096")
            dynamic_max_tokens = 4096
        self.logger.info(f"📊 Transcript tokens (prompt): ~{len(prompt_transcript) // 4}")
        self.logger.info(f"📊 Dynamic max_tokens: {dynamic_max_tokens} (min: 2000, max: 6000)")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional veterinary assistant specializing in SOAP note generation. "
                        "Return ONLY valid JSON matching the provided schema. "
                        "Do NOT include reasoning, thinking, markdown, or explanatory text. "
                        "Be concise: keep each section brief and avoid unnecessary repetition. "
                        "The first non-whitespace character of your response MUST be '{'."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                },
                # JSON anchor: nudges the model to begin with '{' (helps avoid any preamble text)
                {"role": "assistant", "content": "{"},
            ],
            # Deterministic + structure-friendly settings
            "temperature": 0.0,
            "max_tokens": dynamic_max_tokens
        }
        # Some OpenAI-compatible APIs accept `seed`; others may reject it. Only send if explicitly set.
        try:
            _seed = os.getenv("FIREWORKS_SEED")
            if _seed is not None and str(_seed).strip() != "":
                payload["seed"] = int(_seed)
        except Exception:
            pass
        
        # Fireworks Structured Outputs: json_schema enforces output format during generation.
        # Docs: https://docs.fireworks.ai/structured-responses/structured-response-formatting
        # We also instruct JSON in the prompt (system + SOAP_PROMPT_TEMPLATE) per Fireworks recommendation.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "SoapNote",
                "schema": SOAP_NOTE_JSON_SCHEMA,
            },
        }
        # If finish_reason=="length", response may be truncated; dynamic_max_tokens (with floor/cap) helps avoid that.
        if self.logger:
            self.logger.debug(f"📤 Request payload includes response_format: {payload.get('response_format')}")

        # Simple non-streaming request (like old code)
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Sending request to Fireworks (attempt {attempt + 1}/{self.config.max_retries})")
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                result = response.json()
                if 'choices' not in result or len(result['choices']) == 0:
                    raise ValueError("No valid response from Fireworks API")
                content = result['choices'][0].get('message', {}).get('content', '')
                if not content:
                    # Log the full response for debugging
                    self.logger.error(f"❌ CRITICAL: API response has no content")
                    self.logger.error(f"   Response structure: {list(result.keys())}")
                    self.logger.error(f"   Choices structure: {result.get('choices', [{}])[0] if result.get('choices') else 'No choices'}")
                    raise ValueError("API response contains no content - empty SOAP note")
                
                raw_response = content.strip()
                
                # Log raw response preview for debugging
                if self.logger:
                    preview = raw_response[:300] if len(raw_response) > 300 else raw_response
                    self.logger.debug(f"📝 Raw response preview: {preview}...")
                
                # We request JSON (response_format=json_schema), but Fireworks/Llama may still
                # return plain text (e.g. "Subjective: ... Objective: ...") — provider does not
                # enforce format as strictly as OpenAI. extract_soap_json handles both JSON and
                # plain-text section headers and always returns a dict so downstream gets JSON.
                soap_json_dict = None
                try:
                    soap_json_dict = json.loads(raw_response)
                except Exception:
                    soap_json_dict = extract_soap_json(raw_response, logger=self.logger)
                
                if not soap_json_dict:
                    # This should not happen as extract_soap_json now returns a default structure
                    # But keep this check for safety
                    self.logger.error(f"❌ CRITICAL: extract_soap_json returned None (unexpected)")
                    self.logger.error(f"   Original content length: {len(content)}")
                    self.logger.error(f"   Raw response preview: {raw_response[:500]}")
                    self.logger.warning("⚠️  Creating default SOAP note structure as emergency fallback")
                    
                    # Emergency fallback: Create default structure
                    soap_json_dict = {
                        "Subjective": "",
                        "Objective": "",
                        "Assessment": "",
                        "Plan": "",
                        "Conclusion": "",
                        "DifferentialDiagnosis": "",
                        "KeyIssues": "",
                        "AbnormalFindings": "",
                        "CustomerInstructions": "",
                        "Protocols": "",
                        "Vitals": "",
                        "Reminders": ""
                    }
                else:
                    # Normalize to required schema keys (fill missing with empty strings)
                    for k in SOAP_NOTE_JSON_SCHEMA["required"]:
                        if k not in soap_json_dict or soap_json_dict[k] is None:
                            soap_json_dict[k] = ""
                
                # Convert dict back to JSON string for consistency with existing code
                soap_note = json.dumps(soap_json_dict, indent=2, ensure_ascii=False)
                
                if self.logger:
                    self.logger.debug("✅ Successfully extracted and validated SOAP JSON")
                    # Verify required sections exist
                    required_sections = ["Subjective", "Objective", "Assessment", "Plan"]
                    missing_sections = [s for s in required_sections if s not in soap_json_dict]
                    if missing_sections:
                        self.logger.warning(f"⚠️  Missing SOAP sections: {missing_sections}")
                    else:
                        self.logger.debug("✅ All required SOAP sections present")
                
                # CRITICAL: Validate that extracted text is valid JSON
                try:
                    json.loads(soap_note)
                    if self.logger:
                        self.logger.debug("✅ Extracted text is valid JSON")
                except json.JSONDecodeError as e:
                    # If extraction failed, try more aggressive extraction
                    if self.logger:
                        self.logger.warning(f"⚠️  Extracted text is not valid JSON: {e}")
                        self.logger.warning(f"   Extracted text preview: {soap_note[:200]}...")
                    
                    # Try to find JSON object more aggressively
                    brace_start = soap_note.find('{')
                    if brace_start != -1:
                        # Find matching closing brace
                        brace_count = 0
                        brace_end = -1
                        for i in range(brace_start, len(soap_note)):
                            if soap_note[i] == '{':
                                brace_count += 1
                            elif soap_note[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    brace_end = i
                                    break
                        
                        if brace_end != -1:
                            potential_json = soap_note[brace_start:brace_end + 1].strip()
                            try:
                                json.loads(potential_json)
                                soap_note = potential_json
                                if self.logger:
                                    self.logger.info("✅ Successfully extracted valid JSON using aggressive extraction")
                            except json.JSONDecodeError:
                                if self.logger:
                                    self.logger.error(f"❌ Aggressive extraction also failed: {e}")
                                raise ValueError(f"Could not extract valid JSON from response: {e}")
                    else:
                        raise ValueError(f"Could not find JSON object in response: {e}")
                
                total_time = time.time() - start_time
                
                self.logger.info("SOAP note generated successfully")
                self.logger.info(f"✅ SOAP note generation completed (took {total_time:.2f}s)")
                self.logger.info(f"📊 Input length: {len(source_transcript)} characters")
                self.logger.info(f"📊 Output length: {len(soap_note)} characters")
                return soap_note
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in SOAP note generation: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise RuntimeError("Failed to generate SOAP note after all retry attempts")


class MistralProvider(LLMProvider):
    """Mistral AI provider for SOAP note generation."""
    
    def __init__(self, api_key: str, config: Config, logger: logging.Logger):
        super().__init__(api_key, config, logger)
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
    
    def generate_soap_note(self, conversation: str, pre_appointment: str = "", protocols: str = "", vitals: str = "", detected_concepts_json: str = "", raw_transcript: str = "", anchor_mapping_instruction: str = "") -> str:
        """Generate SOAP note using Mistral API."""
        source_transcript = raw_transcript or conversation
        prompt_transcript = _prepare_source_transcript_for_soap_prompt(source_transcript, logger=self.logger)

        # Build dynamic prompt sections
        prompt_sections = build_optional_inputs_section(pre_appointment, protocols, vitals)
        
        # Format Brain NER JSON for prompt (or empty string if not provided)
        if not detected_concepts_json:
            detected_concepts_json = "No Brain NER entities available."
        else:
            # PERF: Avoid redundant json.loads->json.dumps churn when detected_concepts_json is already a JSON string.
            # Only serialize if a dict/list was provided.
            try:
                if not isinstance(detected_concepts_json, str):
                    detected_concepts_json = json.dumps(detected_concepts_json, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Use as-is if anything unexpected happens
        
        prompt = build_soap_prompt_from_brain_ner(
            raw_transcript=prompt_transcript,
            optional_inputs=prompt_sections,
            brain_ner_json=detected_concepts_json,
        )
        
        # Calculate dynamic max_tokens based on transcript length
        dynamic_max_tokens = calculate_dynamic_max_tokens(prompt_transcript)
        self.logger.info(f"📊 Transcript tokens (prompt): ~{len(prompt_transcript) // 4}")
        self.logger.info(f"📊 Dynamic max_tokens: {dynamic_max_tokens} (min: 2000, max: 6000)")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional veterinary assistant specializing in SOAP note generation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": OPTIMIZED_CONFIG["temperature"],
            "max_tokens": dynamic_max_tokens
        }
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Sending request to Mistral (attempt {attempt + 1}/{self.config.max_retries})")
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    raw_response = result['choices'][0]['message']['content'].strip()
                    soap_note = extract_json_from_llm_response(raw_response, logger=self.logger)
                    self.logger.info("SOAP note generated successfully")
                    return soap_note
                else:
                    raise ValueError("No valid response from Mistral API")
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error in Mistral request: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        raise RuntimeError("Failed to generate SOAP note after all retry attempts")


# ==============================================================================
# SOAP NOTE GENERATOR
# ==============================================================================

class SOAPNoteGenerator:
    """Main class for generating SOAP notes from transcribed conversations."""
    
    def __init__(self, config: Config):
        self.config = config
        # Use the already initialized logger instead of creating a new one
        self.logger = logging.getLogger('soap_generator')
        self.api_keys = self._load_api_keys()
        self.llm_provider = self._create_llm_provider()
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from file (experiment/parent/current dir) or from environment."""
        try:
            return load_api_keys(API_KEY_FILE)
        except Exception as e:
            self.logger.error(f"Failed to load API keys: {e}")
            raise
    
    def _create_llm_provider(self) -> LLMProvider:
        """Create appropriate LLM provider based on configuration."""
        # Use config.model_provider (not global MODEL_PROVIDER) to respect per-step provider selection
        provider_enum = self.config.model_provider
        
        if provider_enum == ModelProvider.OPENAI:
            api_key = (self.api_keys.get('OPENAI_API_KEY') or '').strip() or (os.getenv('OPENAI_API_KEY') or '').strip()
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found. Add OPENAI_API_KEY=... to API_Key.txt or set OPENAI_API_KEY in the environment."
                )
            return OpenAIProvider(api_key, self.config, self.logger)
        elif provider_enum == ModelProvider.CLAUDE:
            api_key = (self.api_keys.get('ANTHROPIC_API_KEY') or '').strip() or (os.getenv('ANTHROPIC_API_KEY') or '').strip()
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found. Add to API_Key.txt or set ANTHROPIC_API_KEY in the environment."
                )
            return ClaudeProvider(api_key, self.config, self.logger)
        elif provider_enum == ModelProvider.MISTRAL:
            api_key = (self.api_keys.get('MISTRAL_API_KEY') or '').strip() or (os.getenv('MISTRAL_API_KEY') or '').strip()
            if not api_key:
                raise ValueError(
                    "MISTRAL_API_KEY not found. Add to API_Key.txt or set MISTRAL_API_KEY in the environment."
                )
            return MistralProvider(api_key, self.config, self.logger)
        elif provider_enum == ModelProvider.FIREWORKS:
            # Try to load from fireworks_api.txt file first
            api_key = None
            try:
                api_key = load_fireworks_api_key()
            except Exception as e:
                self.logger.debug(f"Could not load Fireworks API key from fireworks_api.txt: {e}")
            
            # Fallback to API_Key.txt if not found
            if not api_key:
                api_key = self.api_keys.get('FIREWORKS_API_KEY', '') or self.api_keys.get('fireworks_API', '')
            
            if not api_key:
                raise ValueError("Fireworks API key not found. Please add it to fireworks_api.txt (format: fireworks_API=fw_xxx) or API_Key.txt (format: FIREWORKS_API_KEY=fw_xxx)")
            
            return FireworksProvider(api_key, self.config, self.logger)
        else:
            raise ValueError(f"Unknown model_provider: {provider_enum}. Supported: ModelProvider.OPENAI, ModelProvider.CLAUDE, ModelProvider.MISTRAL, ModelProvider.FIREWORKS")
    
    def process_transcription(self, timestamp: str = None, detected_concepts_json: str = "", raw_transcript_override: str = "", anchor_mapping_instruction: str = "") -> Tuple[str, bool]:
        """
        Process transcription file and generate SOAP note.
        
        Args:
            timestamp: Optional timestamp string to include in filename (format: YYYYMMDD_HHMMSS)
                      If None, uses plain "soap_note.txt" for backward compatibility
            detected_concepts_json: Optional JSON string of detected concepts from NER extraction
            raw_transcript_override: Optional raw transcript text to use in the SOAP prompt
            anchor_mapping_instruction: When set (e.g. ANCHOR_MAPPING_SOAP_INSTRUCTION), SOAP gen must wrap entity mentions as [[anchor_id:display_text]]
        
        Returns:
            Tuple of (soap_note_content, success_flag)
        """
        try:
            # Read transcription
            self.logger.info("Reading transcription file...")
            conversation = read_transcription(self.config.input_transcription_path)
            raw_transcript = raw_transcript_override or conversation
            self.logger.info(f"Transcription loaded: {len(conversation)} characters")
            # Removed full transcription content logging to avoid duplicate output
            # self.logger.info(f"Transcription content: {conversation}")
            
            # Read optional input files in parallel for speed
            self.logger.info("Loading optional input files in parallel...")
            file_configs = [
                (self.config.pre_appointment_summary_path, "Pre-appointment summary"),
                (self.config.protocols_template_path, "Protocols template"),
                (self.config.vitals_template_path, "Vitals template")
            ]
            
            file_contents = read_multiple_files_parallel(file_configs, self.logger)
            pre_appointment = file_contents["Pre-appointment summary"]
            protocols = file_contents["Protocols template"]
            vitals = file_contents["Vitals template"]
            
            # Generate SOAP note
            self.logger.info("Generating SOAP note...")
            self.logger.info(f"Conversation length being sent to LLM: {len(conversation)} characters")
            self.logger.info(f"Estimated conversation tokens: {len(conversation) // 4} tokens")
            
            soap_note = self.llm_provider.generate_soap_note(
                conversation=conversation,
                raw_transcript=raw_transcript,
                pre_appointment=pre_appointment,
                protocols=protocols,
                vitals=vitals,
                detected_concepts_json=detected_concepts_json,
                anchor_mapping_instruction=anchor_mapping_instruction
            )
            
            # CRITICAL: Validate SOAP note is not empty
            if not soap_note or not soap_note.strip():
                self.logger.error(f"❌ CRITICAL: SOAP note generation returned empty content")
                self.logger.error(f"   This indicates the LLM API returned no content or parsing failed")
                return "", False
            
            # Save SOAP note with timestamp if provided, otherwise use plain filename
            if timestamp:
                output_file = Path(self.config.output_dir) / f"soap_note_{timestamp}.txt"
            else:
                output_file = Path(self.config.output_dir) / "soap_note.txt"
            
            if save_output(soap_note, str(output_file), self.logger):
                self.logger.info("SOAP note generation completed successfully")
                # SOAP note is structured JSON: also save as .json when timestamped
                if timestamp and soap_note.strip().startswith("{"):
                    soap_json_file = Path(self.config.output_dir) / f"soap_note_{timestamp}.json"
                    try:
                        with open(soap_json_file, "w", encoding="utf-8") as f:
                            f.write(soap_note.strip())
                        self.logger.info(f"SOAP note (JSON) saved to: {soap_json_file}")
                    except Exception as json_err:
                        self.logger.warning(f"Could not save SOAP note as JSON: {json_err}")
                return soap_note, True
            else:
                self.logger.error("Failed to save SOAP note")
                return soap_note, False
                
        except Exception as e:
            self.logger.error(f"Error processing transcription: {e}")
            self.logger.debug(traceback.format_exc())
            return f"Error generating SOAP note: {e}", False


# ==============================================================================
# AUDIO TRANSCRIPTION UTILITIES
# ==============================================================================
# Transcription: Fireworks Whisper v3 Turbo (from reference implementation - use as is).
FIREWORKS_MODEL_NAME = "whisper-v3-turbo"
FIREWORKS_SAMPLE_RATE = 16000
# Resolve fireworks_api.txt: experiment folder first, then parent, then current (same order as API_Key.txt)
def _resolve_fireworks_api_key_file() -> str:
    for base in (EXPERIMENT_FOLDER_PATH, PARENT_FOLDER_PATH, FOLDER_PATH):
        path = os.path.join(base, "fireworks_api.txt")
        if os.path.exists(path):
            return path
    return os.path.join(FOLDER_PATH, "fireworks_api.txt")
FIREWORKS_API_KEY_FILE = _resolve_fireworks_api_key_file()

# Fireworks audio transcription endpoint (reference implementation: regional direct)
FIREWORKS_AUDIO_TRANSCRIPTION_URL = "https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions"
# Streaming ASR (WebSocket)
FIREWORKS_AUDIO_STREAMING_WS_URL = "wss://audio-streaming.api.fireworks.ai/v1/audio/transcriptions/streaming"

def check_audio_dependencies():
    """Check if required audio processing libraries are available."""
    logger = logging.getLogger('soap_generator')
    missing_libs = []
    
    # Check soundfile (usually comes with numpy)
    try:
        import soundfile
        logger.info("✅ soundfile available")
    except ImportError:
        missing_libs.append("soundfile")
        logger.warning("❌ soundfile not available")
    
    # Check pydub (good for MP3 handling)
    try:
        import pydub
        logger.info("✅ pydub available")
    except ImportError:
        missing_libs.append("pydub")
        logger.warning("❌ pydub not available")
    
    # Check librosa (good for audio processing)
    try:
        import librosa
        logger.info("✅ librosa available")
    except ImportError:
        missing_libs.append("librosa")
        logger.warning("❌ librosa not available")
    
    # Check ffmpeg-python (fallback for problematic files)
    try:
        import ffmpeg
        logger.info("✅ ffmpeg-python available")
    except ImportError:
        missing_libs.append("ffmpeg-python")
        logger.warning("❌ ffmpeg-python not available")
    
    if missing_libs:
        logger.warning(f"⚠️  Missing audio libraries: {', '.join(missing_libs)}")
        logger.info("For better audio file support, install missing libraries:")
        logger.info("pip install pydub librosa ffmpeg-python")
        logger.info("Note: ffmpeg-python requires ffmpeg to be installed on your system")
    
    return len(missing_libs) == 0

def convert_audio_to_wav(input_file, output_file=None):
    """
    Convert audio file to WAV format using available libraries.
    This can help with problematic MP3, WebM, and other compressed files.
    """
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.wav"
    
    print(f"Converting {input_file} to WAV format...")
    
    # Try pydub first (good for MP3, WebM, and other formats)
    try:
        from pydub import AudioSegment
        print(f"  Attempting conversion with pydub...")
        
        # Load audio with pydub (supports WebM, MP3, and many other formats)
        audio = AudioSegment.from_file(input_file)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set frame rate to target sample rate
        audio = audio.set_frame_rate(16000)
        
        # Export to WAV
        audio.export(output_file, format="wav")
        print(f"✅ Converted successfully: {output_file}")
        return output_file
        
    except ImportError:
        print("  pydub not available")
    except Exception as e:
        print(f"  pydub conversion failed: {e}")
    
    # Try ffmpeg if available (excellent for WebM and other formats)
    try:
        import ffmpeg
        print(f"  Attempting conversion with ffmpeg...")
        
        stream = ffmpeg.input(input_file)
        stream = ffmpeg.output(stream, output_file, acodec='pcm_s16le', ac=1, ar=16000)
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        print(f"✅ Converted successfully: {output_file}")
        return output_file
        
    except ImportError:
        print("  ffmpeg-python not available")
    except Exception as e:
        print(f"  ffmpeg conversion failed: {e}")
    
    # Try librosa as last resort (good for many formats)
    try:
        import librosa
        print(f"  Attempting conversion with librosa...")
        
        # Load audio with librosa
        audio_data, sample_rate = librosa.load(input_file, sr=16000, mono=True)
        
        # Save as WAV using soundfile
        import soundfile as sf
        sf.write(output_file, audio_data, 16000, subtype='PCM_16')
        
        print(f"✅ Converted successfully: {output_file}")
        return output_file
        
    except ImportError:
        print("  librosa not available")
    except Exception as e:
        print(f"  librosa conversion failed: {e}")
    
    raise RuntimeError("Could not convert audio file. Install pydub, ffmpeg-python, or librosa for conversion support.")

def load_fireworks_api_key():
    """
    Load Fireworks API key from (in order): env FIREWORKS_API_KEY, fireworks_api.txt, or API_Key.txt.
    Handles formats: fireworks_API=fw_xxx, FIREWORKS_API_KEY=fw_xxx, or standalone fw_xxx line.
    """
    # 1) Environment variable
    api_key = (os.getenv("FIREWORKS_API_KEY") or "").strip()
    if api_key and api_key.startswith("fw_"):
        return api_key

    def _read_key_from_file(path: str):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("fireworks_API=") or line.startswith("FIREWORKS_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"\'')
                        if key.startswith("fw_"):
                            return key
                    if line.startswith("fw_"):
                        return line
        except Exception:
            pass
        return None

    # 2) fireworks_api.txt (experiment folder, then parent, then current)
    api_key = _read_key_from_file(FIREWORKS_API_KEY_FILE)
    if api_key:
        return api_key

    # 3) Fallback: API_Key.txt (same locations as rest of app)
    api_key = _read_key_from_file(API_KEY_FILE)
    if api_key:
                return api_key

    raise RuntimeError(
        "Fireworks API key not found. Set FIREWORKS_API_KEY env, or add to fireworks_api.txt or API_Key.txt "
        "(format: FIREWORKS_API_KEY=fw_xxx or fireworks_API=fw_xxx). Key must start with fw_."
    )


def fast_convert_to_wav(input_path, output_path, sample_rate=16000):
    """Ultra-fast audio conversion using optimized ffmpeg settings (from reference implementation)."""
    import subprocess
    if input_path.lower().endswith('.wav'):
        try:
            if sf is not None:
                info = sf.info(input_path)
                if info.samplerate == sample_rate and info.channels == 1:
                    return input_path
        except Exception:
            pass
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav",
            "-loglevel", "error", "-threads", "0",
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", str(sample_rate), "-ac", "1", "-f", "wav", output_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to convert audio file: {e}")


def fast_convert_to_wav_optimized(input_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """Optimized audio conversion for large files with minimal processing (from reference implementation)."""
    import subprocess
    if input_path.lower().endswith('.wav'):
        try:
            if sf is not None:
                info = sf.info(input_path)
                if info.samplerate == sample_rate and info.channels == 1:
                    return input_path
        except Exception:
            pass
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav",
            "-loglevel", "error", "-threads", "0",
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", str(sample_rate), "-ac", "1", "-f", "wav", output_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to convert audio file: {e}")


def validate_audio_file_has_content(file_path: str) -> Tuple[bool, str]:
    """Validate that audio file is not empty or corrupted. Returns (is_valid, error_message)."""
    try:
        sz = os.path.getsize(file_path)
        if sz == 0:
            return (False, "Audio file is empty (0 bytes)")
        if sz < 1024:
            return (False, f"Audio file is too small ({sz} bytes) - likely corrupted or empty")
        with open(file_path, 'rb') as f:
            header = f.read(100)
        if len(header) < 12:
            return (False, "Audio file is corrupted - cannot read file header")
        if header[4:8] == b'ftyp':
            pass
        elif header[:4] == b'RIFF' and len(header) >= 12 and header[8:12] == b'WAVE':
            pass
        elif header[:3] == b'ID3' or header[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
            pass
        elif header[:4] == b'OggS' or header[:4] == b'fLaC' or header[:4] == b'\x1a\x45\xdf\xa3':
            pass
        elif len(header) >= 2 and (header[0] == 0xFF and (header[1] & 0xF0) == 0xF0):
            pass
        else:
            return (False, "Audio file format is invalid or corrupted - file header does not match known audio formats")
        if header[4:8] == b'ftyp' and sz < 8192:
            return (False, f"Audio file appears corrupted - file size ({sz} bytes) is too small for a valid recording")
        return (True, "")
    except Exception as e:
        return (False, str(e))


def detect_audio_format_from_content_simple(file_path: str) -> Optional[str]:
    """Detect audio format from file content (magic bytes). Returns extension e.g. '.m4a', '.mp3' or None."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(20)
        if len(header) >= 12 and header[4:8] == b'ftyp':
            if b'M4A' in header[8:20] or b'm4a' in header[8:20]:
                return '.m4a'
            if b'mp41' in header[8:20] or b'isom' in header[8:20]:
                return '.m4a'
        if len(header) >= 12 and header[:4] == b'RIFF' and header[8:12] == b'WAVE':
            return '.wav'
        if len(header) >= 3 and header[:3] == b'ID3':
            return '.mp3'
        if len(header) >= 2 and header[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
            return '.mp3'
        if len(header) >= 4 and header[:4] == b'OggS':
            return '.ogg'
        if len(header) >= 4 and header[:4] == b'fLaC':
            return '.flac'
        if len(header) >= 4 and header[:4] == b'\x1a\x45\xdf\xa3':
            return '.webm'
        if len(header) >= 2 and (header[0] == 0xFF and (header[1] & 0xF0) == 0xF0):
            return '.aac'
    except Exception:
        pass
    return None


def write_wav(fname, audio, samplerate):
    """Write audio data to WAV file with PCM_16 encoding."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio)) * 0.95
    if sf is not None:
        sf.write(fname, audio, samplerate, subtype="PCM_16")
    else:
        # Minimal fallback WAV writer (PCM16 mono) if soundfile isn't available
        import wave, struct
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(fname, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(samplerate))
            wf.writeframes(struct.pack("<" + "h" * len(pcm), *pcm))

# Retry config for Fireworks transcription (502/503 are transient)
TRANSCRIBE_RETRY_STATUSES = (502, 503)
TRANSCRIBE_RETRY_ATTEMPTS = 3
TRANSCRIBE_RETRY_BACKOFF_SEC = (2, 4, 8)


def transcribe_audio(audio_file_path, output_dir=None, timestamp: str = None):
    """
    Canonical Fireworks one-shot transcription function.
    Replaces redundant transcribe_audio_fireworks_* variants.
    Retries on 502/503 with backoff; raises a short user-facing message (no HTML dump).
    """
    logger = logging.getLogger('soap_generator')
    logger.info("🎤 Starting ultra-fast audio transcription...")
    logger.info(f"📁 Audio file: {audio_file_path}")
    api_key = load_fireworks_api_key()
    file_ext = Path(audio_file_path).suffix.lower()
    if not file_ext:
        file_ext = detect_audio_format_from_content_simple(audio_file_path) or '.wav'
    if file_ext == '.x-m4a':
        file_ext = '.m4a'
    mime_type_map = {'.wav': 'audio/wav', '.m4a': 'audio/mp4', '.mp4': 'audio/mp4', '.mp3': 'audio/mpeg', '.aac': 'audio/aac', '.ogg': 'audio/ogg', '.flac': 'audio/flac', '.webm': 'audio/webm'}
    mime_type = mime_type_map.get(file_ext, 'audio/wav')
    upload_filename = f"audio{file_ext}" if file_ext else "audio.wav"
    url = FIREWORKS_AUDIO_TRANSCRIPTION_URL
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": FIREWORKS_MODEL_NAME, "temperature": "0"}
    file_size = os.path.getsize(audio_file_path)
    if file_size == 0:
        raise RuntimeError("Audio file is empty (0 bytes)")
    timeout = max(60, file_size // (1024 * 1024) * 10)
    last_response = None
    for attempt in range(TRANSCRIBE_RETRY_ATTEMPTS):
        with open(audio_file_path, "rb") as audio_file:
            files = {"file": (upload_filename, audio_file, mime_type)}
            last_response = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)
        if last_response.status_code == 200:
            result = last_response.json()
            text = result.get("text", "").strip() or result.get("transcription", "").strip()
            if not text and isinstance(result, dict):
                for key in result:
                    if 'text' in key.lower() or 'transcript' in key.lower():
                        text = str(result[key]).strip()
                        break
            if not text:
                raise EmptyTranscriptionError("API returned empty transcription. The audio file may be corrupted, contain no speech, or be in an unsupported format.")
            return text
        if last_response.status_code in TRANSCRIBE_RETRY_STATUSES and attempt < TRANSCRIBE_RETRY_ATTEMPTS - 1:
            backoff = TRANSCRIBE_RETRY_BACKOFF_SEC[attempt] if attempt < len(TRANSCRIBE_RETRY_BACKOFF_SEC) else 8
            logger.warning("Fireworks transcription returned %s (attempt %d/%d); retrying in %ds ...", last_response.status_code, attempt + 1, TRANSCRIBE_RETRY_ATTEMPTS, backoff)
            time.sleep(backoff)
            continue
        break
    status = last_response.status_code
    body = (last_response.text or "").strip()
    if status in (502, 503) or "502" in body[:200] or "503" in body[:200]:
        raise RuntimeError(
            f"Fireworks transcription service temporarily unavailable ({status} Bad Gateway). "
            "Please try again in a few minutes. If it persists, check status.fireworks.ai or try again later."
        )
    if len(body) > 500 or "<!DOCTYPE" in body or "<html" in body.lower():
        raise RuntimeError(f"Fireworks AI Error: {status} - Service returned an error page. Check your API key and network, or try again later.")
    raise RuntimeError(f"Fireworks AI Error: {status} - {body[:500]}")


def transcribe_audio_fireworks_streaming(
    audio_file_path,
    output_dir=None,
    timestamp: str = None,
    on_transcript_chunk=None,
    logger: Optional[logging.Logger] = None,
):
    """
    Transcribe audio via Fireworks streaming WebSocket. As each segment arrives,
    the accumulated transcript is updated and on_transcript_chunk(accumulated_text, segments)
    is called if provided (for custom processing logic).

    Requires: websocket-client. If not available, falls back to one-shot transcription
    and calls on_transcript_chunk once with the full text.

    Returns: full raw transcript string.
    """
    log = logger or logging.getLogger("soap_generator")
    api_key = load_fireworks_api_key()

    # Ensure 16 kHz mono PCM WAV for streaming API
    wav_path = audio_file_path
    if Path(audio_file_path).suffix.lower() not in (".wav",):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            fast_convert_to_wav_optimized(audio_file_path, wav_path, sample_rate=FIREWORKS_SAMPLE_RATE)
        except Exception as e:
            if wav_path != audio_file_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            log.warning("Streaming: convert to WAV failed (%s), falling back to one-shot", e)
            raw = transcribe_audio(audio_file_path)
            if on_transcript_chunk and raw:
                on_transcript_chunk(raw, [])
            return raw

    try:
        if not sf or SOUNDFILE_AVAILABLE is False:
            raise ImportError("soundfile required for streaming")
        data, sr = sf.read(wav_path, dtype="int16")
        if sr != FIREWORKS_SAMPLE_RATE:
            raise ValueError(f"Streaming expects {FIREWORKS_SAMPLE_RATE} Hz, got {sr}")
        if data.ndim > 1:
            data = data.mean(axis=1).astype("int16")
        pcm_bytes = data.tobytes()
    except Exception as e:
        log.warning("Streaming: read WAV failed (%s), falling back to one-shot", e)
        raw = transcribe_audio(audio_file_path)
        if on_transcript_chunk and raw:
            on_transcript_chunk(raw, [])
        return raw
    finally:
        if wav_path != audio_file_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception:
                pass

    try:
        import urllib.parse
        import websocket
    except ImportError:
        log.warning("websocket-client not installed; run pip install websocket-client. Falling back to one-shot.")
        raw = transcribe_audio(audio_file_path)
        if on_transcript_chunk and raw:
            on_transcript_chunk(raw, [])
        return raw

    segment_state = {}

    def _accumulated_text():
        # Rebuild text from segment state in id order (numeric sort)
        order = sorted(segment_state.keys(), key=lambda x: (int(x) if str(x).isdigit() else 0, x))
        return " ".join(segment_state.get(k, "") for k in order).strip()

    def on_message(ws, message):
        nonlocal segment_state
        try:
            msg = json.loads(message)
        except Exception:
            return
        if msg.get("checkpoint_id") == "final":
            return
        segments = msg.get("segments") or []
        for s in segments:
            if not isinstance(s, dict):
                continue
            sid = s.get("id")
            text = (s.get("text") or "").strip()
            if sid is not None:
                segment_state[str(sid)] = text
        acc = _accumulated_text()
        if on_transcript_chunk and segments:
            on_transcript_chunk(acc, list(segment_state.items()))

    def on_error(ws, error):
        log.debug("Streaming ASR WebSocket error: %s", error)

    url = FIREWORKS_AUDIO_STREAMING_WS_URL
    params = urllib.parse.urlencode({"language": "en"})
    full_url = f"{url}?{params}"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Send audio in 50 ms chunks (1600 samples = 3200 bytes at 16-bit)
    chunk_samples = int(0.05 * FIREWORKS_SAMPLE_RATE)
    chunk_bytes = chunk_samples * 2

    result_holder = {"text": ""}
    import threading
    ws_obj = None

    def send_audio():
        for i in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[i : i + chunk_bytes]
            if not chunk:
                continue
            try:
                ws_ref = getattr(send_audio, "_ws", None)
                if ws_ref and getattr(ws_ref, "connected", True):
                    ws_ref.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                log.debug("Streaming send: %s", e)
            time.sleep(0.05)
        try:
            ws_ref = getattr(send_audio, "_ws", None)
            if ws_ref and getattr(ws_ref, "connected", True):
                ws_ref.send(json.dumps({"checkpoint_id": "final"}), opcode=websocket.ABNF.OPCODE_TEXT)
        except Exception as e:
            log.debug("Streaming final checkpoint: %s", e)

    def on_open(ws):
        nonlocal ws_obj
        ws_obj = ws
        setattr(send_audio, "_ws", ws)

    def on_close(ws_app, close_status_code, close_msg):
        result_holder["text"] = _accumulated_text()

    ws_app = websocket.WebSocketApp(
        full_url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=lambda: ws_app.run_forever(), daemon=True)
    ws_thread.start()

    for _ in range(100):
        if ws_obj is not None:
            break
        time.sleep(0.05)
    if ws_obj is None:
        log.warning("Streaming: WebSocket did not connect in time, falling back to one-shot")
        raw = transcribe_audio(audio_file_path)
        if on_transcript_chunk and raw:
            on_transcript_chunk(raw, [])
        return raw

    setattr(send_audio, "_ws", ws_obj)
    send_thread = threading.Thread(target=send_audio, daemon=True)
    send_thread.start()

    send_thread.join(timeout=300)
    ws_thread.join(timeout=60)
    result_holder["text"] = _accumulated_text()

    return (result_holder.get("text") or "").strip() or transcribe_audio(audio_file_path)

# ==============================================================================
# MAIN PIPELINE FUNCTION
# ==============================================================================

def run_transcription_cleaning(raw_transcription: str, output_dir: str = None) -> str:
    """
    Run Step 2 transcription cleaning using the Clinical Refinement Layer model.
    Automatically detects if model is OpenAI (gpt-*) and uses OpenAI provider.
    """
    logger = logging.getLogger('soap_generator')
    
    # Detect provider based on model name
    # If model starts with "gpt-", it's an OpenAI model and must use OpenAI provider
    cleaning_model = STEP_2_CLEANING_MODEL
    if cleaning_model.startswith("gpt-"):
        # Force OpenAI provider for OpenAI models
        cleaning_provider = ModelProvider.OPENAI
        logger.info(f"🔍 Detected OpenAI model '{cleaning_model}' - using OpenAI provider")
    else:
        # Use default provider for other models (Fireworks, etc.)
        cleaning_provider = DEFAULT_CONFIG.model_provider
        logger.info(f"🔍 Using default provider '{cleaning_provider}' for model '{cleaning_model}'")
    
    config = Config(
        input_transcription_path="",
        output_dir=output_dir or DEFAULT_CONFIG.output_dir,
        api_key_file=DEFAULT_CONFIG.api_key_file,
        model_provider=cleaning_provider,  # Use detected/forced provider
        model_name=cleaning_model,
        pre_appointment_summary_path=DEFAULT_CONFIG.pre_appointment_summary_path,
        protocols_template_path=DEFAULT_CONFIG.protocols_template_path,
        vitals_template_path=DEFAULT_CONFIG.vitals_template_path,
        request_timeout=DEFAULT_CONFIG.request_timeout,
        max_retries=DEFAULT_CONFIG.max_retries
    )
    generator = SOAPNoteGenerator(config)
    return generator.llm_provider.clean_transcription(raw_transcription)


def _structured_transcript_fallback(cleaned_transcription: str):
    """Build minimal structured transcript from cleaned text when SOAPTranscriptLoader returns [].
    Splits on newlines; parses 'V: ', 'O: ', 'U: ' prefixes. When the whole text is one long line,
    also splits on mid-string speaker tags ( V: , O: , U: ) so multiple turns are produced."""
    if not cleaned_transcription or not cleaned_transcription.strip():
        return []
    text = cleaned_transcription.strip()
    segments = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("V: "):
            segments.append({"speaker": "V", "text": line[3:].strip()})
        elif line.startswith("O: "):
            segments.append({"speaker": "O", "text": line[3:].strip()})
        elif line.startswith("U: "):
            segments.append({"speaker": "U", "text": line[3:].strip()})
        else:
            segments.append({"speaker": "U", "text": line})
    # If we got only one segment and it's a long single line, try splitting on mid-string V:/O:/U:
    if len(segments) == 1 and len(text) > 400 and "\n" not in text:
        parts = re.split(r"(?:\s+)([VOU]):\s*", text, flags=re.IGNORECASE)
        if len(parts) >= 3:
            segments = []
            first = (parts[0] or "").strip()
            if first:
                if first.upper().startswith("V: "):
                    segments.append({"speaker": "V", "text": first[3:].strip()})
                elif first.upper().startswith("O: "):
                    segments.append({"speaker": "O", "text": first[3:].strip()})
                elif first.upper().startswith("U: "):
                    segments.append({"speaker": "U", "text": first[3:].strip()})
                else:
                    segments.append({"speaker": "U", "text": first})
            for i in range(1, len(parts) - 1, 2):
                speaker = (parts[i] or "U").upper()
                if speaker not in ("V", "O", "U"):
                    speaker = "U"
                content = (parts[i + 1] or "").strip()
                if content:
                    segments.append({"speaker": speaker, "text": content})
    return segments


def generate_soap_with_grounding(
    raw_transcript: str,
    entity_manifest: list,
    output_dir: str,
    timestamp: str
) -> Tuple[str, bool, Any]:
    """
    Generate SOAP note using raw transcript + Brain NER entities (no anchor injection).
    Returns: (soap_note, success, generator) - generator is needed for same-session modification
    """
    logger = logging.getLogger('soap_generator')
    detected_concepts_json = build_detected_concepts_json_from_manifest(entity_manifest, logger=logger)

    temp_txt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    temp_txt.write(raw_transcript.encode("utf-8"))
    temp_txt.close()

    try:
        config = Config(
            input_transcription_path=temp_txt.name,
            output_dir=output_dir or DEFAULT_CONFIG.output_dir,
            api_key_file=DEFAULT_CONFIG.api_key_file,
            model_provider=DEFAULT_CONFIG.model_provider,
            model_name=DEFAULT_CONFIG.model_name,
            pre_appointment_summary_path=DEFAULT_CONFIG.pre_appointment_summary_path,
            protocols_template_path=DEFAULT_CONFIG.protocols_template_path,
            vitals_template_path=DEFAULT_CONFIG.vitals_template_path,
            request_timeout=DEFAULT_CONFIG.request_timeout,
            max_retries=DEFAULT_CONFIG.max_retries
        )
        generator = SOAPNoteGenerator(config)
        soap_note, success = generator.process_transcription(
            timestamp=timestamp,
            detected_concepts_json=detected_concepts_json,
            raw_transcript_override=raw_transcript,
            anchor_mapping_instruction=""
        )
        return soap_note, success, generator
    finally:
        os.unlink(temp_txt.name)


def apply_manifest_corrections_to_soap_json(
    soap_note_text: str,
    entity_manifest: list,
    logger=None,
    use_anchor_tags: bool = False,
) -> str:
    """
    DETERMINISTIC TRUTH INJECTOR: Fast, reliable string replacement (no LLM call).
    
    This function replaces verified entities in the SOAP note with their correct terms
    from the KB or Local Inventory. It preserves unlinked entities (does NOT remove them),
    ensuring no information erosion.
    
    When use_anchor_tags=True (Anchor-Span mode): replaces with [[anchor_id|correct_term]]
    so the frontend can render <span data-entity-id="anchor_id">correct_term</span> for
    bi-directional sync (Billing <-> SOAP). Requires entity_manifest entries to have
    anchor_id set (e.g. via ensure_anchor_ids_on_manifest).
    
    Key Features:
    - Only replaces verified entities (KB or Local matches)
    - Preserves unlinked entities for vet review (no information loss)
    - Deterministic string replacement (sub-millisecond latency)
    - CRITICAL TYPE GATE: Prevents hallucinations by rejecting matches where:
      - Original entity kind (from NER) doesn't match KB concept kind
      - Example: "Cortex capsules" (Drug) matched to "capsule" (Anatomy) -> REJECT
    
    Only acts on verified entities (KB or Local) with correct terms.

    When SOAP was generated with Anchor Mapping, it may contain [[Eid:display_text]].
    We first normalize those to [[Eid|canonical]] using the manifest, then run span_text
    replacement for any remaining unanchored mentions.
    """
    if not soap_note_text or not entity_manifest:
        return soap_note_text

    # Anchor-mapped SOAP: normalize [[Eid:...]] -> [[Eid|canonical]] so we don't rely on brittle string match
    if re.search(r"\[\[E\d+:", soap_note_text):
        try:
            from kb_anchor_span import normalize_anchor_tags_colon_to_pipe
            soap_note_text = normalize_anchor_tags_colon_to_pipe(
                soap_note_text, entity_manifest, logger=logger
            )
            if logger:
                logger.info("🔗 Truth Injector: Normalized anchor tags (colon -> pipe with canonical)")
        except Exception as e:
            if logger:
                logger.warning("⚠️ Truth Injector: normalize_anchor_tags_colon_to_pipe failed: %s", e)

    # Build replacement pairs: only use verified entities (KB or Local)
    # FIX A: Remove fallback replacement - only inject verified truth
    pairs = []
    skipped_reasons = {
        "no_span_text": 0,
        "not_verified": 0,
        "no_correct_term": 0,
        "terms_match": 0,
        "type_gate_rejected": 0,
    }
    
    if logger:
        logger.info(f"🔍 Truth Injector: Processing {len(entity_manifest)} entities from manifest")
    
    for e in entity_manifest:
        raw_mention = (e.get("span_text") or "").strip()
        if not raw_mention:
            skipped_reasons["no_span_text"] += 1
            continue
        
        # FIX C: Treat local matches as verified for injection
        is_local_verified = bool(e.get("local_stock_id") or e.get("local_service_id"))
        if (e.get("match_method") or "").strip().lower() == "dual_sync_judge_rejected":
            is_local_verified = False
        is_kb_verified = bool(e.get("kb_concept_id"))
        
        # Only inject if we have verified truth (KB or Local)
        if not (is_kb_verified or is_local_verified):
            skipped_reasons["not_verified"] += 1
            if logger:
                logger.debug(f"  ⏭️  Skipping '{raw_mention}': not verified (kb_id={e.get('kb_concept_id')}, local_id={e.get('local_stock_id') or e.get('local_service_id')})")
            continue  # Skip unverified entities - don't inject guesses
        
        # Get the correct term to inject
        if is_kb_verified:
            correct_term = (e.get("kb_preferred_name") or e.get("normalized_name") or "").strip()
            verification_type = "KB"
            concept_id = e.get("kb_concept_id")
        else:  # is_local_verified
            correct_term = (e.get("kb_preferred_name") or e.get("display_name") or e.get("normalized_name") or "").strip()
            verification_type = "Local"
            concept_id = e.get("local_stock_id") or e.get("local_service_id")
        
        if not correct_term:
            skipped_reasons["no_correct_term"] += 1
            if logger:
                logger.debug(f"  ⏭️  Skipping '{raw_mention}': no correct_term found")
            continue
        
        if raw_mention.lower() == correct_term.lower():
            skipped_reasons["terms_match"] += 1
            if logger:
                logger.debug(f"  ⏭️  Skipping '{raw_mention}': already matches '{correct_term}'")
            continue  # Skip if no correction needed
        
        # FIX B: Repair Type Gate - use original NER kind, not KB kind
        entity_kind = e.get('kind')  # Original NER kind (e.g., "Drug")
        kb_kind = e.get('kb_kind')  # KB concept kind (e.g., "Anatomy")
        
        # CRITICAL TYPE GATE: Prevent hallucinations by rejecting matches where kind mismatches
        # Example: "Cortex capsules" (Drug) matched to "capsule" (Anatomy) -> REJECT
        # Example: "hip joint" (BodySite/Anatomy) matched to "Hindlimb partial paralysis" (Finding) -> REJECT
        if entity_kind and kb_kind and is_kb_verified:
            # BodySite/Anatomy must NOT match to Finding (e.g. hip joint -> Hindlimb partial paralysis)
            entity_kind_norm = (entity_kind or "").strip().lower()
            if entity_kind_norm in ("anatomy", "bodysite", "body site") and (kb_kind or "").strip() == "Finding":
                kind_compatible = False
            else:
                # Allow compatible kind matches (e.g., Drug -> Substance, Procedure -> Service)
                kind_compatible = (
                    entity_kind == kb_kind or
                    (entity_kind == "Drug" and kb_kind in ["Substance", "Nutrition", "Vaccine", "Toxin"]) or
                    (entity_kind == "Procedure" and kb_kind in ["Service", "DiagnosticTest", "Vaccine", "Device"]) or
                    (entity_kind == "Finding" and kb_kind in ["Condition", "Observation", "Organism", "Anatomy"]) or
                    (entity_kind == "Reason" and kb_kind in ["Condition", "Finding", "Observation", "Organism", "Toxin", "Procedure", "Service", "Anatomy"]) or
                    (entity_kind == "ReasonForVisit" and kb_kind in ["Procedure", "Service", "DiagnosticTest"]) or
                    (entity_kind_norm in ("anatomy", "bodysite", "body site") and (kb_kind or "").strip() in ("Anatomy", "Observation"))
                )
            
            if not kind_compatible:
                skipped_reasons["type_gate_rejected"] += 1
                if logger:
                    logger.warning(f"🚫 Truth Injector REJECTED: '{raw_mention}' (kind: {entity_kind}) matched to '{correct_term}' (kind: {kb_kind}) - Type Gate violation")
                continue  # Skip this replacement - it's a hallucination
        
        # Only replace if the terms are different (avoid unnecessary replacements)
        anchor_id = e.get("anchor_id") if use_anchor_tags else None
        if use_anchor_tags and anchor_id:
            replacement_text = f"[[{anchor_id}|{correct_term}]]"
        else:
            replacement_text = correct_term
        pairs.append((raw_mention, replacement_text))
        # SOAP wording variants: when canonical term is Ortolani test, also replace common SOAP/ASR misspellings
        if correct_term and "ortolani" in correct_term.lower():
            for variant in ("Ultrasoning test", "ultrasoning test", "ultralining test"):
                if variant != raw_mention and not any(p[0] == variant for p in pairs):
                    pairs.append((variant, replacement_text))
                    if logger:
                        logger.info(f"✨ Truth Injector: variant '{variant}' -> '{replacement_text}' (Ortolani)")
        if logger:
            logger.info(f"✨ Truth Injector: '{raw_mention}' -> '{replacement_text}' (verified {verification_type} term, id: {concept_id}, kind: {kb_kind or entity_kind})")
    
    if logger:
        logger.info(f"📊 Truth Injector Summary: {len(pairs)} pairs created, {sum(skipped_reasons.values())} entities skipped")
        logger.info(f"   Skipped: {skipped_reasons}")
    
    # FIX A: REMOVED fallback replacement - no longer injecting unverified normalized names
    # This prevents "normalized but not verified" replacements that cause drift

    if not pairs:
        if logger:
            logger.warning(f"⚠️ Truth Injector: No replacement pairs found (checked {len(entity_manifest)} entities)")
        return soap_note_text

    if logger:
        logger.info(f"🔧 Truth Injector: Processing {len(pairs)} replacement pairs")

    def replace_in_str(s: str) -> str:
        out = s
        replacements_made = 0
        for span, replacement in pairs:
            # whole-word match, case-insensitive
            # FIX: Use single backslash for word boundary in raw f-string
            pattern = rf"\b{re.escape(span)}\b"
            before = out
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
            if before != out:
                replacements_made += 1
                if logger:
                    logger.debug(f"  ✅ Replaced '{span}' -> '{replacement}'")
        if logger and replacements_made == 0:
            logger.warning(f"  ⚠️ No replacements made in text (checked {len(pairs)} pairs)")
        return out

    # Try JSON parsing so we only replace inside values, not keys/structure
    try:
        obj = json.loads(soap_note_text)
        if isinstance(obj, dict):
            if logger:
                logger.debug(f"  📄 SOAP note is JSON format, replacing in string values")
            for k, v in list(obj.items()):
                if isinstance(v, str):
                    obj[k] = replace_in_str(v)
            result = json.dumps(obj, indent=2, ensure_ascii=False)
            if logger:
                logger.info(f"✅ Truth Injector: JSON replacement completed")
            return result
    except Exception as e:
        if logger:
            logger.debug(f"  📄 SOAP note is plain text format (JSON parse failed: {e})")

    # Fallback: plain text replacement
    result = replace_in_str(soap_note_text)
    if logger:
        logger.info(f"✅ Truth Injector: Plain text replacement completed")
    return result

def generate_soap_note_from_audio(audio_file_path=None, output_dir=None):
    """
    Transcribe an audio file and generate a SOAP note (sync wrapper).
    Runs the default pipeline only: async pipeline (chunk-parallel or sequential Super-Pass + local streaming grounding).
    If no audio_file_path is provided, uses the first audio file found in the input folder.
    Returns the SOAP note as a string.
    """
    result = asyncio.run(generate_soap_note_from_audio_async(audio_file_path, output_dir))
    return (result or {}).get("soap_note") or ""


async def generate_soap_note_from_audio_async(audio_file_path=None, output_dir=None):
    """
    Async pipeline: default path only (chunk-parallel or sequential Super-Pass + local streaming grounding).
    Returns dict with soap_note, manifest, cleaned_text, etc.
    """
    pipeline_start_time = time.time()
    stage_perf_start = time.perf_counter()
    stage_marks: Dict[str, float] = {}
    logger = logging.getLogger('soap_generator')
    logger.info("🎤 Async Audio to SOAP Note Generator")
    logger.info("=" * 50)
    logger.info(f"🕐 Pipeline started at {datetime.now().strftime('%H:%M:%S')}")

    # Check audio dependencies
    logger.info("Checking audio processing libraries...")
    check_audio_dependencies()
    logger.info("")

    # If no audio file path provided, use the input folder
    if audio_file_path is None:
        input_folder = "/Users/vivek/VETINSTANT/wip/New folder/P.A.W.S/SOAP notes - voice to text/OP/soap_note_experiment/output/input_audio"
        if not os.path.exists(input_folder):
            os.makedirs(input_folder, exist_ok=True)
            raise RuntimeError(f"Input folder created: {input_folder}. Please place an audio file in this folder and run again.")

        # Find audio files in the input folder
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.webm']
        audio_files = []
        for file in os.listdir(input_folder):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(file)

        if not audio_files:
            raise RuntimeError(f"No audio files found in {input_folder}. Supported formats: {', '.join(audio_extensions)}")

        # Use the first audio file found
        audio_file_path = os.path.join(input_folder, audio_files[0])
        logger.info(f"Using audio file: {audio_files[0]}")

        if len(audio_files) > 1:
            logger.info(f"Multiple audio files found. Using: {audio_files[0]}")
            logger.info(f"Other files: {', '.join(audio_files[1:])}")

    # Validate audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Check file size
    file_size = os.path.getsize(audio_file_path)
    if file_size == 0:
        raise RuntimeError(f"Audio file is empty: {audio_file_path}")

    output_dir_path = Path(output_dir or DEFAULT_CONFIG.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    executor = ThreadPoolExecutor(max_workers=5)
    loop = asyncio.get_event_loop()

    if os.getenv("LOW_LATENCY_MODE", "").lower() in ("1", "true", "yes") or os.getenv("TARGET_LATENCY_SEC", "") == "60":
        logger.info("⚡ LOW_LATENCY_MODE: using gpt-4.1-nano for Super-Pass, gpt-4.1-mini for SOAP, Phase 2 (target ~60s)")
    if TARGET_60S:
        logger.info("⚡ TARGET_60S: nano + FORCE_PARALLEL_SUPER_PASS_CHUNKS + EARLY_START_SOAP + SOAP max_tokens=2500")

    # STEP 1: Transcribe audio (sequential)
    TRANSCRIPTION_STREAMING = os.getenv("TRANSCRIPTION_STREAMING", "false").strip().lower() in ("1", "true", "yes")

    if TRANSCRIPTION_STREAMING:
        logger.info("🎤 STEP 1: STREAMING TRANSCRIPTION")
        raw_transcription = await loop.run_in_executor(
            executor,
            lambda: transcribe_audio_fireworks_streaming(
                audio_file_path,
                output_dir=output_dir_path,
                timestamp=timestamp,
                on_transcript_chunk=None,  # No harvester callback
                logger=logger,
            ),
        )
    else:
        logger.info(f"🎤 STEP 1: AUDIO TRANSCRIPTION (Async pipeline)")
        raw_transcription = await loop.run_in_executor(
            executor,
            transcribe_audio,
            audio_file_path,
            output_dir_path,
            timestamp,
        )

    if not raw_transcription:
        raise RuntimeError("No transcription generated from audio.")

    # Save raw transcript
    raw_transcript_file = output_dir_path / f"raw_transcription_{timestamp}.txt"
    save_output(raw_transcription, str(raw_transcript_file), logger)
    stage_marks["transcription_done"] = time.perf_counter()

    # --------------------------------------------------------------------------
    # STREAMING SUPER-PASS GROUNDING needs these BEFORE super-pass starts
    # --------------------------------------------------------------------------
    # FIX: Get clinic_id and visit_id for local inventory search (used by grounding)
    clinic_id = None
    visit_id = None
    try:
        clinic_id_env = os.getenv('CLINIC_ID')
        if clinic_id_env:
            try:
                clinic_id = int(clinic_id_env)
                logger.info(f"🏥 Using clinic_id from environment: {clinic_id}")
            except ValueError:
                logger.warning(f"⚠️  Invalid CLINIC_ID environment variable: {clinic_id_env}")

        visit_id_env = os.getenv('VISIT_ID')
        if visit_id_env:
            visit_id = visit_id_env
            logger.info(f"📋 Using visit_id from environment: {visit_id}")
        else:
            import uuid
            visit_id = str(uuid.uuid4())
            logger.info(f"📋 Generated visit_id: {visit_id}")

        if not clinic_id:
            clinic_id = 1
            logger.info(f"🏥 Using default clinic_id=1 (inventory and procedures location)")
            logger.info("   To use a different clinic_id, set CLINIC_ID environment variable")
    except Exception as e:
        logger.warning(f"⚠️  Could not get clinic_id/visit_id: {e}")
        if not clinic_id:
            clinic_id = 1
            logger.info(f"🏥 Using fallback default clinic_id=1")

    # Create a Fireworks OpenAI-compatible client for grounding (Super-Pass, Brain NER, etc.).
    # GPT models must use the OpenAI API; Fireworks models must use the Fireworks API. Steps that have
    # their own model (e.g. LLM_JUDGE_MODEL) resolve the correct client via get_client_for_model(model).
    # If Judge is set to gpt-4.1-nano, OPENAI_API_KEY is required; if you only have Fireworks, set
    # LLM_JUDGE_MODEL to a Fireworks model (e.g. SUPER_PASS_MODEL).
    fireworks_client = None
    try:
        from kb_ner_clients import get_openai_client
        fw_key = load_fireworks_api_key()
        fireworks_client = get_openai_client(api_key=fw_key, base_url="https://api.fireworks.ai/inference/v1")
    except Exception as e:
        logger.warning(f"⚠️ Could not create Fireworks client for grounding: {e}")
        fireworks_client = None

    # --------------------------------------------------------------------------
    # PERF: Pre-warm Postgres connection pool so the first streamed entity grounding
    # doesn't pay the cold-start cost mid-stream.
    # --------------------------------------------------------------------------
    try:
        from kb_ner_db import init_pg_pool
        init_pg_pool(logger=logger)
    except Exception as e:
        logger.debug(f"PG pool warmup skipped/failed: {e}")

    # STEP 2: Default path only - Parallel Voice + Brain with Shadow Grounding
    USE_SUPER_PASS = True  # Always use Super-Pass
    initial_entities = []  # Will be populated by shadow grounding path
    used_streaming_grounding = False
    early_phase2_task_ref: Dict[str, Any] = {"task": None}
    streaming_entity_manifest = []
    streaming_grounding_tasks = []
    streaming_dispatched = set()
    soap_early_future_ref: Dict[str, Any] = {"future": None}
    billing_pipeline_task = None  # Async CER + grounding; runs in parallel with SOAP

    # Default path: combined_path_used is always True (only path)
    combined_path_used = False  # Will be set to True in default path
    if USE_SUPER_PASS:
        logger.info("⚡ STEP 2: DEFAULT PATH (Super-Pass non-streaming — single combined call, unified prompt)")
        logger.info("   - Step 2a: Unified prompt (cleaning + minimal NER). Step 2b: Brain NER (enrich with hints/probabilities)")
        logger.info("   - Local-only grounding for billable routes + parallel LLM judges where applicable")
        
        try:
            from kb_ner_super_pass import (
                super_pass_cleaning_and_ner,
                run_brain_call,
                fuzzy_sync_anchor_injection,
                _build_entities_by_kind,
            )
            from kb_ner_clients import get_client_for_model
            
            # Get client for super-pass (configured at top of file: SUPER_PASS_MODEL)
            client_result = get_client_for_model(SUPER_PASS_MODEL)
            if isinstance(client_result, tuple):
                super_pass_client, _ = client_result
            else:
                super_pass_client = client_result
            
            if not super_pass_client:
                raise RuntimeError("Super-Pass client not available. Cannot run pipeline (default path only).")
            else:
                super_pass_stream_task = None
                # Single path only: Super-Pass non-streaming (one combined call, unified prompt). No Voice/Brain split, no streaming.
                shadow_grounding_used = False  # So batch global + dispatch run below
                combined_path_used = True
                logger.info("⚡ STEP 2: SUPER-PASS (unified prompt) → BRAIN NER (enrich with hints/probabilities)")

                # Use streaming-style grounding (shadow grounding is always enabled)
                if combined_path_used and fireworks_client:
                    used_streaming_grounding = True
                    logger.info("⚡ Batch-parallel grounding: Brain NER (non-streaming) → local-only routing/search → parallel LLM judges → dispatch all entities")
                    # Import here so _dispatch_entity closure has process_single_entity_async in its enclosing scope.
                    try:
                        from kb_ner_parallel import process_single_entity_async as _process_single_entity_async
                        if logger:
                            logger.info(f"✅ Imported process_single_entity_async for streaming grounding")
                    except Exception as e:
                        if logger:
                            logger.error(f"❌ CRITICAL: Streaming grounding unavailable ({e}); entities will not be grounded during stream.")
                            import traceback
                            logger.debug(traceback.format_exc())
                        _process_single_entity_async = None
                    try:
                        max_parallel = int(os.getenv("KB_MAX_PARALLEL_ENTITIES", "12"))
                    except Exception:
                        max_parallel = 12
                    if max_parallel < 1:
                        max_parallel = 1
                    sem = asyncio.Semaphore(max_parallel)

                    streaming_entity_manifest = []
                    streaming_grounding_tasks = []
                    streaming_dispatched = set()
                    # Grounding layer output: same folder and tracking ID as SOAP outputs
                    streaming_grounding_records = []
                    # Shared embedding cache so duplicate span_texts (e.g. "physiotherapy") are embedded once
                    streaming_embedding_cache = {}
                    # Batch vector search: precomputed global KB results (entity_idx -> candidates) for combined path
                    streaming_batch_global_prefetch = {}
                    # Parallel LLM judge results: precomputed judge results (entity_idx -> selected candidate) for dual_route entities
                    streaming_precomputed_judge_results = {}

                    def _ent_key(ent: dict) -> tuple:
                        span = (ent.get("span_text") or "").strip().lower()
                        kind = (ent.get("kind") or "Other").strip().lower()
                        return (kind, span)

                    # Grounding Prep parallelization: fire procedure_role_classification as soon as entity streams
                    procedure_like_kinds = {"Procedure", "Service", "Condition", "ReasonForVisit"}

                    async def _dispatch_entity(ent_obj: dict, entity_index: Optional[int] = None):
                        # Convert to pipeline-friendly entity dict (same as non-streaming conversion).
                        # Preserve search_term and family from Super-Pass so batch intent (default path) is used in grounding.
                        hints = ent_obj.get("hints")
                        if isinstance(hints, list):
                            hints = [str(x).strip() for x in hints if str(x).strip()][:3]
                        else:
                            hints = []
                        brain_kind = (ent_obj.get("kind") or "Other")
                        legacy_kb_kind = (ent_obj.get("kb_kind") or "").strip()
                        if logger and legacy_kb_kind and legacy_kb_kind.lower() != str(brain_kind).strip().lower():
                            logger.info(
                                "  🔁 Kind precedence: using Brain NER kind '%s' over legacy kb_kind '%s' for '%s'",
                                brain_kind,
                                legacy_kb_kind,
                                ent_obj.get("span_text", ""),
                            )

                        ent = {
                            "span_text": ent_obj.get("span_text", ""),
                            "normalized_name": ent_obj.get("normalized_name", ent_obj.get("span_text", "")),
                            "kind": brain_kind,
                            "kb_kind": brain_kind,
                            "roles": ent_obj.get("roles", []),
                            "attributes": ent_obj.get("attributes", {}),
                            "assertion_id": ent_obj.get("assertion_id", "CONF"),
                            "supporting_text": ent_obj.get("supporting_text", ""),
                            "start_char": ent_obj.get("start_char", 0),
                            "end_char": ent_obj.get("end_char", 0),
                            "is_actionable": ent_obj.get("is_actionable", True),
                            "search_term": (ent_obj.get("search_term") or "").strip() or None,
                            "hints": hints,
                            "family": (ent_obj.get("family") or "").strip() or None,
                        }
                        k = _ent_key(ent)
                        if not k[1]:  # empty span
                            return
                        if k in streaming_dispatched:
                            return
                        streaming_dispatched.add(k)

                        # Fire procedure_role_classification immediately for procedure-like entities (saves ~7s)
                        role_task = None
                        if not (ent.get("roles") and len(ent.get("roles", [])) > 0):
                            try:
                                from kb_ner_routing import canonicalize_kind, classify_procedure_role
                                canonical_kind = canonicalize_kind(ent.get("kb_kind") or ent.get("kind", "Other"))
                                if canonical_kind in procedure_like_kinds:
                                    attrs_ctx = (ent.get("attributes") or {}) if isinstance(ent.get("attributes"), dict) else {}
                                    stream_ctx = (attrs_ctx.get("_stream_context") or "").strip()
                                    context_text = stream_ctx or raw_transcription
                                    # Context window: slice around span or first 800 chars
                                    span = ent.get("span_text", "")
                                    if span and context_text and span in context_text:
                                        pos = context_text.find(span)
                                        ctx_start = max(0, pos - 400)
                                        ctx_end = min(len(context_text), pos + len(span) + 400)
                                        context_window = context_text[ctx_start:ctx_end]
                                    else:
                                        context_window = (context_text or "")[:800]
                                    role_task = asyncio.create_task(
                                        asyncio.to_thread(
                                            classify_procedure_role,
                                            span_text=span,
                                            context_window=context_window,
                                            speaker=None,
                                            client=fireworks_client,
                                            logger=logger,
                                        )
                                    )
                            except Exception:
                                role_task = None

                        async def _ground_one(role_future=None, eidx: Optional[int] = None):
                            if _process_single_entity_async is None:
                                if logger:
                                    logger.error(f"❌ CRITICAL: _process_single_entity_async is None - cannot ground entity '{ent.get('span_text','')}'")
                                    logger.error(f"   This means grounding tasks will be created but will do nothing. Check import of process_single_entity_async.")
                                return
                            async with sem:
                                entity_span = ent.get('span_text', 'Unknown')
                                try:
                                    attrs_ctx = (ent.get("attributes") or {}) if isinstance(ent.get("attributes"), dict) else {}
                                    stream_ctx = (attrs_ctx.get("_stream_context") or "").strip()
                                    context_text = stream_ctx or raw_transcription
                                    if logger:
                                        logger.debug(f"   🔍 Grounding entity {eidx + 1 if eidx is not None else '?'}: '{entity_span}'")
                                    res, presenting = await _process_single_entity_async(
                                        entity=ent,
                                        idx=(eidx + 1) if eidx is not None else len(streaming_dispatched),
                                        total=len(initial_entities) if (eidx is not None and initial_entities) else 0,
                                        cleaned_transcript=context_text,
                                        raw_transcript=context_text,
                                        conn=None,
                                        client=fireworks_client,
                                        clinic_id=clinic_id,
                                        visit_id=visit_id,
                                        auto_bind_threshold=GROUNDING_AUTO_BIND_THRESHOLD,
                                        llm_judge_threshold=GROUNDING_LLM_JUDGE_THRESHOLD,
                                        logger=logger,
                                        embedding_cache=streaming_embedding_cache,
                                        role_classification_future=role_future,
                                        precomputed_global=streaming_batch_global_prefetch if (eidx is not None and streaming_batch_global_prefetch) else None,
                                        grounding_collector=streaming_grounding_records,
                                        precomputed_judge_results=streaming_precomputed_judge_results if (eidx is not None and streaming_precomputed_judge_results) else None,
                                    )
                                    if res:
                                        # Never leave display_name as "0"; use normalized_name or span_text
                                        _dn = (res.get("display_name") or "").strip()
                                        if not _dn or _dn == "0":
                                            res = dict(res)
                                            res["display_name"] = (res.get("normalized_name") or res.get("span_text") or "").strip() or res.get("span_text", "")
                                        streaming_entity_manifest.append(res)
                                        if logger:
                                            logger.debug(f"   ✅ Grounded entity '{entity_span}' → added to manifest (total: {len(streaming_entity_manifest)})")
                                    else:
                                        if logger:
                                            logger.warning(f"   ⚠️ Grounding returned None/empty for entity '{entity_span}'")
                                except Exception as e:
                                    if logger:
                                        logger.error(f"❌ CRITICAL: Streaming grounding failed for '{entity_span}': {e}")
                                        import traceback
                                        logger.debug(traceback.format_exc())

                        # CRITICAL: Only create task if _process_single_entity_async is available
                        if _process_single_entity_async is None:
                            if logger:
                                logger.error(f"❌ CRITICAL: Cannot create grounding task for '{ent.get('span_text','')}' - _process_single_entity_async is None")
                        else:
                            task = asyncio.create_task(_ground_one(role_task, entity_index))
                            streaming_grounding_tasks.append(task)
                            if logger:
                                logger.debug(f"   📝 Created grounding task for entity '{ent.get('span_text','')}' (total tasks: {len(streaming_grounding_tasks)})")

                    if combined_path_used:
                        # Check if chunk-parallel processing is enabled
                        if CHUNK_PARALLEL_ENABLED and len(raw_transcription) > CHUNK_SIZE:
                            # Chunk-Parallel Factory: Process chunks in parallel for sub-60s latency
                            if logger:
                                logger.info("🚀 CHUNK-PARALLEL MODE: Processing raw transcript in parallel chunks")
                            try:
                                from kb_ner_chunk_parallel import process_chunks_parallel
                                cleaned_transcription, initial_entities, _entities_by_kind = await process_chunks_parallel(
                                    raw_transcription,
                                    model=SUPER_PASS_MODEL,
                                    client=super_pass_client,
                                    logger=logger,
                                    chunk_size=CHUNK_SIZE,
                                    overlap_size=CHUNK_OVERLAP,
                                )
                                if not cleaned_transcription:
                                    cleaned_transcription = raw_transcription.strip()
                                if logger:
                                    logger.info(f"✅ Chunk-parallel complete: cleaned={len(cleaned_transcription)} chars, {len(initial_entities)} entities")
                            except Exception as e:
                                if logger:
                                    logger.error(f"❌ Chunk-parallel processing failed: {e}")
                                raise
                        else:
                            # Sequential processing (when chunk-parallel disabled or transcript too short)
                            # Step 2a: Super-Pass unified prompt (upstream) — cleaning + minimal NER (id, span_text, kind, attributes). No hints/probabilities.
                            cleaned_transcript, super_pass_entities, _entities_by_kind = await super_pass_cleaning_and_ner(
                                raw_transcription, model=SUPER_PASS_MODEL, client=super_pass_client, logger=logger,
                            )
                            cleaned_transcription = cleaned_transcript or raw_transcription.strip()
                            # Step 2b: Brain NER (downstream) — enrich pre-extracted entities with hints, correctness/suggestion_probability, domain.
                            pre_extracted = [
                                {"id": "e" + str(i + 1), "span_text": e.get("span_text", ""), "kind": e.get("kind", "Other"), "attributes": e.get("attributes", {})}
                                for i, e in enumerate(super_pass_entities or [])
                            ]
                            pre_extracted_json = json.dumps(pre_extracted, ensure_ascii=False)
                            entity_manifest, _terms_not_grounded = await asyncio.to_thread(
                                run_brain_call, cleaned_transcription, pre_extracted_json, SUPER_PASS_MODEL, super_pass_client, logger,
                            )
                            initial_entities = list(entity_manifest or [])
                        # Save Brain NER output (pre-grounding) for inspection
                        brain_ner_file = output_dir_path / f"brain_ner_output_{timestamp}.json"
                        try:
                            with open(brain_ner_file, "w", encoding="utf-8") as f:
                                json.dump(initial_entities, f, indent=2, ensure_ascii=False)
                            if logger:
                                if initial_entities:
                                    logger.info(f"   📄 Brain NER output saved to: {brain_ner_file} ({len(initial_entities)} entities)")
                                else:
                                    logger.warning(f"   ⚠️ Brain NER output saved (EMPTY - 0 entities) to: {brain_ner_file}")
                        except Exception as e:
                            if logger:
                                logger.warning(f"   ⚠️ Could not save Brain NER output: {e}")
                        if logger:
                            logger.info(f"   ✅ Super-Pass (unified) + Brain NER (enriched): cleaned={len(cleaned_transcription)} chars, {len(initial_entities)} entities with hints/probabilities")
                        # Billing pipeline (CER + grounding) runs async, independent of SOAP. Only SOAP receives consolidated Brain NER (initial_entities).
                        async def _run_billing_pipeline_async():
                            """CER → batch prep → dispatch grounding. Runs in parallel with SOAP; result collected in streaming_entity_manifest."""
                            stage_marks["billing_start"] = time.perf_counter()
                            entities_for_grounding = initial_entities
                            enable_cer = os.getenv("ENABLE_CER", "true").strip().lower() in ("1", "true", "yes")
                            skip_cer_for_short = CER_SKIP_UNDER_CHARS > 0 and len(raw_transcription) <= CER_SKIP_UNDER_CHARS
                            if skip_cer_for_short and logger:
                                logger.info(
                                    f"  ⏩ CER skipped for short transcript ({len(raw_transcription)} chars ≤ CER_SKIP_UNDER_CHARS={CER_SKIP_UNDER_CHARS}); using Brain NER entities for grounding"
                                )
                            if enable_cer and (initial_entities or []) and not skip_cer_for_short:
                                try:
                                    from kb_ner_clinical_entity_resolver import (
                                        run_clinical_entity_resolver_async,
                                        CER_MODEL as CER_MODEL_DEFAULT,
                                    )
                                    from kb_ner_clients import get_client_for_model
                                    cer_model_name = (os.getenv("CER_MODEL", CER_MODEL_DEFAULT) or CER_MODEL_DEFAULT).strip()
                                    cer_client = None
                                    try:
                                        cer_client_result = get_client_for_model(cer_model_name, logger=logger)
                                        cer_client = cer_client_result[0] if isinstance(cer_client_result, tuple) else cer_client_result
                                    except Exception:
                                        cer_client = None
                                    if cer_client is None:
                                        cer_client = super_pass_client or fireworks_client
                                    if cer_client:
                                        entities_for_grounding = await run_clinical_entity_resolver_async(
                                            initial_entities, cer_client, model=cer_model_name, logger=logger
                                        )
                                        if not entities_for_grounding:
                                            entities_for_grounding = initial_entities
                                    else:
                                        entities_for_grounding = initial_entities
                                except Exception as e:
                                    if logger:
                                        logger.warning(f"   ⚠️ CER failed: {e}, using full Brain NER list")
                                    entities_for_grounding = initial_entities
                            else:
                                entities_for_grounding = initial_entities
                            if not shadow_grounding_used:
                                use_batch_vector = os.getenv("KB_USE_BATCH_VECTOR_SEARCH", "true").strip().lower() in ("1", "true", "yes")
                                if use_batch_vector and (entities_for_grounding or []) and fireworks_client:
                                    try:
                                        from kb_ner_routing import sanitize_asr_errors, canonicalize_kind, classify_entity_route
                                        from kb_ner_global_search import (
                                            run_batch_global_vector_search,
                                            map_ner_kind_to_kb_kind_filter,
                                            REASON_ALLOWED_KB_KINDS,
                                        )
                                        from kb_ner_db import acquire_pg_conn, release_pg_conn
                                        from kb_ner_embeddings import embed_texts
                                        need_global = []
                                        for idx0, ent_obj in enumerate(entities_for_grounding or []):
                                            if not isinstance(ent_obj, dict):
                                                continue
                                            span_text = (ent_obj.get("span_text") or "").strip()
                                            if not span_text:
                                                continue
                                            normalized_name = span_text.replace("[unclear]", "").strip() if "[unclear]" in span_text else span_text
                                            normalized_name = sanitize_asr_errors(normalized_name)
                                            kb_kind_raw = ent_obj.get("kb_kind") or ent_obj.get("kind", "Other")
                                            canonical_kind = canonicalize_kind(kb_kind_raw)
                                            route = classify_entity_route(canonical_kind, entity=ent_obj, logger=logger)
                                            if route != "global_direct":
                                                continue
                                            search_term = (ent_obj.get("search_term") or "").strip() or normalized_name
                                            if canonical_kind == "ReasonForVisit":
                                                kind_filter = list(REASON_ALLOWED_KB_KINDS)
                                            else:
                                                kind_filter = map_ner_kind_to_kb_kind_filter(canonical_kind) or []
                                            hints = ent_obj.get("hints")
                                            domain_from_brain = ent_obj.get("domain")
                                            suggestion_prob = ent_obj.get("suggestion_probability")
                                            hint_probabilities = ent_obj.get("hint_probabilities")
                                            if isinstance(domain_from_brain, str) and (domain_from_brain or "").strip().lower() not in ("", "general"):
                                                domain_arg = [domain_from_brain.strip().lower()]
                                            elif isinstance(domain_from_brain, list) and domain_from_brain:
                                                domain_arg = [(d or "").strip().lower() for d in domain_from_brain if (d or "").strip().lower() and (d or "").strip().lower() != "general"]
                                            else:
                                                domain_arg = None
                                            if isinstance(hints, list) and (span_text or search_term):
                                                    need_global.append((idx0, search_term, kind_filter, span_text, hints[:3], domain_arg, suggestion_prob, hint_probabilities))
                                            else:
                                                    need_global.append((idx0, search_term, kind_filter, None, None, domain_arg, suggestion_prob, hint_probabilities))
                                            # Default path only: no global KB (LOCAL_ONLY); skip run_batch_global_vector_search
                                            unique_texts = list(dict.fromkeys((e.get("span_text") or "").strip() for e in (entities_for_grounding or []) if (e.get("span_text") or "").strip()))
                                            if unique_texts:
                                                try:
                                                    _embeddings = embed_texts(unique_texts, client=fireworks_client, logger=logger)
                                                    if _embeddings and len(_embeddings) == len(unique_texts):
                                                        for _t, _v in zip(unique_texts, _embeddings):
                                                            if _v:
                                                                streaming_embedding_cache[_t] = _v
                                                        if logger:
                                                            logger.info(f"  📊 Batch embeddings: {len(unique_texts)} unique terms (billing pipeline)")
                                                except Exception as _e:
                                                    if logger:
                                                        logger.debug(f"  ⚠️  Batch embed pre-fill failed: {_e}")
                                    except Exception as e:
                                        if logger:
                                            logger.debug(f"  ⚠️  Batch vector search / batch embed failed: {e}")
                                    for idx0, ent in enumerate(entities_for_grounding or []):
                                        await _dispatch_entity(ent, idx0)
                                    if streaming_grounding_tasks:
                                        try:
                                            await asyncio.wait_for(
                                                asyncio.gather(*streaming_grounding_tasks, return_exceptions=True),
                                                timeout=60.0,
                                            )
                                        except asyncio.TimeoutError:
                                            if logger:
                                                pending = [t for t in streaming_grounding_tasks if not t.done()]
                                                logger.warning(f"  ⚠️ Billing pipeline: timeout waiting for {len(pending)} grounding tasks")
                            stage_marks["grounding_done"] = time.perf_counter()
                            if logger:
                                logger.info(f"  ✅ Billing pipeline complete: {len(streaming_entity_manifest)} entities in manifest")

                        billing_pipeline_task = asyncio.create_task(_run_billing_pipeline_async()) if not shadow_grounding_used else None
                        if not shadow_grounding_used:
                            super_pass_stream_task = None
                
                # Single path: cleaned_transcription and initial_entities set by Super-Pass non-streaming above
                # No streaming path and no Voice/Brain split - this is the only execution path
                
                # CRITICAL: Save Brain NER output unconditionally (before any conditional checks)
                # This ensures brain_ner_output is saved even if cleaned_transcription is falsy
                try:
                    brain_ner_file_unconditional = output_dir_path / f"brain_ner_output_{timestamp}.json"
                    with open(brain_ner_file_unconditional, "w", encoding="utf-8") as f:
                        json.dump(initial_entities, f, indent=2, ensure_ascii=False)
                    if logger:
                        if initial_entities:
                            logger.info(f"   📄 Brain NER output saved (unconditional): {brain_ner_file_unconditional} ({len(initial_entities)} entities)")
                        else:
                            logger.warning(f"   ⚠️ Brain NER output saved (EMPTY - 0 entities) unconditional: {brain_ner_file_unconditional}")
                except Exception as e:
                    if logger:
                        logger.warning(f"   ⚠️ Could not save Brain NER output (unconditional): {e}")
                
                if not cleaned_transcription:
                    raise RuntimeError("Super-Pass returned no cleaned transcript. Cannot continue (default path only).")
                else:
                    # Default path: cleaned_transcription set in shadow consumer
                    n_ent = len(initial_entities)
                    if used_streaming_grounding and n_ent == 0:
                        logger.info(f"✅ Shadow grounding: {len(cleaned_transcription)} chars cleaned, entities streamed to grounding (not aggregated)")
                    else:
                        logger.info(f"✅ Shadow grounding complete: {len(cleaned_transcription)} chars cleaned, {n_ent} entities extracted")
                        # Save cleaned transcript
                        cleaned_transcript_file = output_dir_path / f"cleaned_transcript_{timestamp}.txt"
                        save_output(cleaned_transcription, str(cleaned_transcript_file), logger)
                        logger.info(f"✅ Cleaned transcript saved: {cleaned_transcript_file}")
                        logger.info(f"📊 Input length: {len(raw_transcription)} characters")
                        logger.info(f"📊 Output length: {len(cleaned_transcription)} characters")
                    
                    # CRITICAL: Save Brain NER output unconditionally (after all paths have populated initial_entities)
                    # This ensures brain_ner_output is saved regardless of which execution path was taken
                    brain_ner_file = output_dir_path / f"brain_ner_output_{timestamp}.json"
                    try:
                        with open(brain_ner_file, "w", encoding="utf-8") as f:
                            json.dump(initial_entities, f, indent=2, ensure_ascii=False)
                        if logger:
                            if initial_entities:
                                logger.info(f"   📄 Brain NER output saved to: {brain_ner_file} ({len(initial_entities)} entities)")
                            else:
                                logger.warning(f"   ⚠️ Brain NER output saved (EMPTY - 0 entities) to: {brain_ner_file}")
                    except Exception as e:
                        if logger:
                            logger.warning(f"   ⚠️ Could not save Brain NER output: {e}")
                    # Save NER output by kind (12 kinds) alongside cleaned transcript so NER output is visible
                    try:
                        entities_by_kind = _build_entities_by_kind(initial_entities)
                        ner_by_kind_file = output_dir_path / f"cleaned_transcript_{timestamp}_entities_by_kind.json"
                        with open(ner_by_kind_file, "w", encoding="utf-8") as f:
                            json.dump(entities_by_kind, f, indent=2, ensure_ascii=False)
                        if logger:
                            logger.info(f"   📄 NER output by kind saved to: {ner_by_kind_file}")
                    except Exception as e:
                        if logger:
                            logger.warning(f"   ⚠️ Could not save entities_by_kind: {e}")
        except Exception as e:
            import traceback
            if logger:
                logger.error(f"❌ Super-pass failed: {e}")
                logger.debug(traceback.format_exc())
            raise RuntimeError(f"Super-pass failed (default path only): {e}") from e

    # Standalone cleaning path removed: pipeline always uses Super-Pass (cleaning + NER in one call).

    # STEP 2.3 & STEP 3: Launch grounding + SOAP generation in parallel (both use cleaned transcript)
    stage_marks["superpass_done"] = time.perf_counter()
    # In streaming mode, grounding started during super-pass streaming (or after Batch Intent if Clean-then-Intent).
    if used_streaming_grounding:
        # Ensure streaming super-pass finished so we have final (cleaned_transcription, initial_entities).
        try:
            if "super_pass_stream_task" in locals() and super_pass_stream_task is not None:
                await super_pass_stream_task
        except Exception as e:
            if logger:
                logger.error("❌ Super-pass streaming task failed (default path only).", exc_info=True)
            raise RuntimeError(f"Super-pass streaming failed: {e}") from e

        # Default path: streaming grounding only
        if used_streaming_grounding:
            logger.info("⚡ STEP 2.3: BATCH-PARALLEL GROUNDING (Brain NER non-streaming → local-only routing/search → parallel judges)")
            logger.info("📋 STEP 3: SOAP NOTE GENERATION (uses cleaned transcript)")
        else:
            if USE_SUPER_PASS and initial_entities:
                logger.info("⚡ STEP 2.3 (Grounding) + 📋 STEP 3 (SOAP Generation) in parallel")
                logger.info(f"   Using cleaned transcript from super-pass ({len(initial_entities)} entities pre-extracted)")
            else:
                logger.info("⚡ STEP 2.3 (Grounding) + 📋 STEP 3 (SOAP Generation) in parallel")
                logger.info("   Both tasks use the cleaned transcript as primary source")

        # Default path only: grounding is via streaming (process_single_entity_async). No run_step_2_3_normalization.
        grounding_task = None

        # Task B: SOAP Note Generation (Brain NER entities only; no anchor mapping)
        # Cancel early-start SOAP (2k prefix) if still running; we use full transcript SOAP as final.
        if soap_early_future_ref.get("future") is not None and not soap_early_future_ref["future"].done():
            soap_early_future_ref["future"].cancel()
            logger.info("⚡ Early-start SOAP cancelled (using full transcript SOAP)")
        manifest_for_soap_streaming = [dict(e) for e in (initial_entities or []) if isinstance(e, dict)]
        logger.info("📋 STEP 3: SOAP NOTE GENERATION (uses cleaned transcript)")
        logger.info("   SOAP pipeline started (independent of billing)")
        soap_task = loop.run_in_executor(
            executor,
            generate_soap_with_grounding,
            cleaned_transcription,  # Use cleaned transcript, not raw
            manifest_for_soap_streaming,  # Brain NER entities only
            output_dir_path,
            timestamp
        )
    # Await SOAP first (SOAP pipeline latency = superpass_done → soap_done)
    soap_result = await soap_task
    stage_marks["soap_done"] = time.perf_counter()
    if isinstance(soap_result, tuple) and len(soap_result) == 3:
        soap_note, success, generator = soap_result
    else:
        # Backward compatibility: if function returns 2-tuple
        soap_note, success = soap_result
        generator = None
    
    # SOAP is independent from the moment consolidated Brain NER was sent; no "resolve manifest" for SOAP.
    # Await billing pipeline (CER + grounding) only to obtain the entity manifest for Phase 2 and saving.
    if billing_pipeline_task is not None:
        try:
            await asyncio.wait_for(billing_pipeline_task, timeout=90.0)
            if logger:
                logger.info("✅ Billing pipeline (CER + grounding) completed")
        except asyncio.TimeoutError:
            if logger:
                logger.warning("⚠️ Billing pipeline timed out (90s); using partial manifest")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Billing pipeline error: {e}")
    
    # Resolve billing entity manifest (for Phase 2 and saving). SOAP does not depend on this.
    refined_transcript = cleaned_transcription
    entity_manifest = []
    # When billing ran async, manifest is in streaming_entity_manifest; no need to await streaming_grounding_tasks again
    if billing_pipeline_task is not None:
        entity_manifest = streaming_entity_manifest or []
        try:
            entity_manifest.sort(key=lambda e: (int(e.get("start_char") or 0), (e.get("span_text") or "").lower()))
        except Exception:
            pass
        if logger and entity_manifest:
            logger.info(f"✅ Billing pipeline: {len(entity_manifest)} entities in manifest")
        grounding_path = output_dir_path / f"grounding_layer_output_{timestamp}.json"
        try:
            with open(grounding_path, "w", encoding="utf-8") as f:
                json.dump(streaming_grounding_records if streaming_grounding_records else [], f, indent=2, ensure_ascii=False)
            if logger:
                if streaming_grounding_records:
                    logger.info(f"  📄 Grounding layer output: {len(streaming_grounding_records)} records → {grounding_path}")
                else:
                    logger.warning(f"  ⚠️ Grounding layer output saved (EMPTY) to: {grounding_path}")
        except Exception as e:
            if logger:
                logger.warning(f"  ⚠️ Could not write grounding layer output: {e}")
    # CRITICAL: Check if streaming grounding is being used OR if streaming_grounding_tasks exist (legacy path: no billing_pipeline_task)
    elif used_streaming_grounding or streaming_grounding_tasks:
        # Wait for any in-flight streaming grounding tasks to complete (should already be mostly done).
        if streaming_grounding_tasks:
            if logger:
                logger.info(f"⏳ Waiting for {len(streaming_grounding_tasks)} streaming grounding tasks to complete...")
            try:
                # Add timeout to prevent indefinite hanging (60 seconds max)
                results = await asyncio.wait_for(
                    asyncio.gather(*streaming_grounding_tasks, return_exceptions=True),
                    timeout=60.0
                )
                exceptions = [r for r in results if isinstance(r, Exception)]
                successful = [r for r in results if not isinstance(r, Exception) and r is not None]
                if exceptions and logger:
                    logger.error(f"❌ CRITICAL: {len(exceptions)}/{len(streaming_grounding_tasks)} streaming grounding tasks raised exceptions:")
                    for i, exc in enumerate(exceptions[:10]):  # Show first 10 exceptions
                        logger.error(f"   Exception {i+1}: {str(exc)[:200]}")
                if logger:
                    logger.info(
                        f"✅ All {len(streaming_grounding_tasks)} streaming grounding tasks completed: "
                        f"{len(successful)} successful, {len(exceptions)} exceptions, "
                        f"{len(streaming_entity_manifest)} entities in manifest"
                    )
            except asyncio.TimeoutError:
                pending = [t for t in streaming_grounding_tasks if not t.done()]
                if logger:
                    logger.warning(
                        f"⚠️ Timeout waiting for streaming grounding tasks (60s). "
                        f"{len(pending)}/{len(streaming_grounding_tasks)} tasks still pending. Proceeding with completed results."
                    )
                for task in pending:
                    task.cancel()
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Error waiting for streaming grounding tasks: {e}. Proceeding with completed results.")
                    import traceback
                    logger.debug(traceback.format_exc())

        # Use the manifest built incrementally during streaming.
        entity_manifest = streaming_entity_manifest or []
        try:
            entity_manifest.sort(key=lambda e: (int(e.get("start_char") or 0), (e.get("span_text") or "").lower()))
        except Exception:
            pass

        if not entity_manifest and initial_entities:
            if logger:
                logger.error(
                    f"❌ CRITICAL: Brain NER extracted {len(initial_entities)} entities but grounding returned empty manifest. "
                    f"Grounding failed - entities were not properly grounded."
                )
                logger.error("   This will cause downstream issues. Check grounding tasks for errors.")
        elif entity_manifest:
            if logger:
                logger.info(f"✅ Streaming grounding complete: {len(entity_manifest)} entities in manifest")
        else:
            if logger:
                logger.warning(
                    f"⚠️ No entities in manifest (Brain NER extracted {len(initial_entities)} entities, "
                    f"grounding returned {len(streaming_entity_manifest)} entities)"
                )

        # Always save grounding layer output (even if empty) for debugging and inspection.
        grounding_path = output_dir_path / f"grounding_layer_output_{timestamp}.json"
        try:
            with open(grounding_path, "w", encoding="utf-8") as f:
                json.dump(streaming_grounding_records if streaming_grounding_records else [], f, indent=2, ensure_ascii=False)
            if logger:
                if streaming_grounding_records:
                    logger.info(f"  📄 Grounding layer output: {len(streaming_grounding_records)} records → {grounding_path}")
                else:
                    logger.warning(f"  ⚠️ Grounding layer output saved (EMPTY - 0 records) to: {grounding_path}")
                    logger.warning("     This indicates grounding records were not collected. Check grounding_collector flow.")
        except Exception as e:
            if logger:
                logger.warning(f"  ⚠️ Could not write grounding layer output: {e}")
    else:
        # Await grounding task with error handling (only if grounding_task was created).
        if grounding_task is not None:
            try:
                refined_transcript, entity_manifest = await grounding_task
                if entity_manifest is None:
                    logger.warning("⚠️ Grounding task returned None for entity_manifest - setting to empty list")
                    entity_manifest = []
                elif not isinstance(entity_manifest, list):
                    logger.warning(f"⚠️ Grounding task returned invalid entity_manifest type: {type(entity_manifest)} - setting to empty list")
                    entity_manifest = []
                else:
                    logger.info(f"✅ Grounding task completed: {len(entity_manifest)} entities in manifest")
            except Exception as e:
                logger.error(f"❌ CRITICAL: Grounding task failed with error: {e}")
                logger.error("   This means entity extraction and linking did not complete")
                import traceback
                logger.error(traceback.format_exc())
                refined_transcript = cleaned_transcription
                entity_manifest = []
        else:
            # No grounding task was created.
            refined_transcript = cleaned_transcription
            entity_manifest = []
            if logger:
                logger.info("⚡ Skipping non-streaming grounding task: no grounding task created")

        # Note: Grounding layer output for non-streaming path is saved by run_step_2_3_normalization internally.
        non_streaming_grounding_path = output_dir_path / f"grounding_layer_output_{timestamp}.json"
        if non_streaming_grounding_path.exists():
            if logger:
                logger.info(f"  📄 Grounding layer output saved by run_step_2_3_normalization: {non_streaming_grounding_path}")
        else:
            if logger:
                logger.warning(f"  ⚠️ Grounding layer output file not found: {non_streaming_grounding_path}")
                logger.warning("     This may indicate grounding records were not collected in non-streaming path.")

    # CRITICAL: Ensure the original SOAP note is saved even if generation had issues
    # Check if file was already saved by process_transcription
    original_file = output_dir_path / f"soap_note_{timestamp}.txt"
    
    # If SOAP note is empty but file exists, try to read it
    if (not soap_note or not soap_note.strip()) and original_file.exists():
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                soap_note = f.read().strip()
            if soap_note:
                logger.info(f"✅ Read SOAP note from existing file: {original_file}")
                success = True
        except Exception as e:
            logger.warning(f"⚠️ Could not read SOAP note from file: {e}")
    
    # Save or verify SOAP note
    if soap_note and soap_note.strip():
        if not original_file.exists() or original_file.stat().st_size == 0:
            # File wasn't saved or is empty, save it now
            if save_output(soap_note, str(original_file), logger):
                logger.info(f"✅ Original SOAP note saved to: {original_file}")
            else:
                logger.error(f"❌ Failed to save original SOAP note to: {original_file}")
        else:
            logger.info(f"✅ SOAP note already saved to: {original_file}")
    else:
        logger.warning(f"⚠️ SOAP note is empty or None - cannot save")
        if original_file.exists():
            logger.info(f"   File exists at {original_file} but content is empty")

    # SOAP note is structured JSON: also save as .json for downstream consumption
    if soap_note and soap_note.strip().startswith("{"):
        soap_json_file = output_dir_path / f"soap_note_{timestamp}.json"
        try:
            with open(soap_json_file, "w", encoding="utf-8") as f:
                f.write(soap_note.strip())
            if logger:
                logger.info(f"✅ SOAP note (JSON) saved to: {soap_json_file}")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Could not save SOAP note as JSON: {e}")

    # Hard-stop: rejected dual-sync matches must never keep local IDs.
    enforce_dual_sync_reject_hard_stop(entity_manifest, logger=logger)

    # Sanitize display_name: never leave "0" (use normalized_name or span_text)
    for e in entity_manifest or []:
        if isinstance(e, dict):
            _dn = (e.get("display_name") or "").strip()
            if not _dn or _dn == "0":
                e["display_name"] = (e.get("normalized_name") or e.get("span_text") or "").strip() or e.get("span_text", "")

    # Always save entity manifest (even if empty) for debugging
    entity_manifest_file = output_dir_path / f"entity_manifest_{timestamp}.json"
    try:
        with open(entity_manifest_file, 'w', encoding='utf-8') as f:
            json.dump(entity_manifest if entity_manifest else [], f, indent=2, ensure_ascii=False)
        if entity_manifest:
            logger.info(f"✅ Entity manifest saved to: {entity_manifest_file} ({len(entity_manifest)} entities)")
        else:
            logger.warning(f"⚠️ Entity manifest is empty - saved empty manifest to: {entity_manifest_file}")
            logger.warning(f"   This indicates grounding/entity extraction failed or found no entities")
    except Exception as e:
        logger.error(f"❌ Failed to save entity manifest: {e}")
    if billing_pipeline_task is None:
        stage_marks["grounding_done"] = time.perf_counter()
    
    # PHASE 2 + CONSTRAINT INJECTION:
    # Pre-Extraction: Start Phase 2 as soon as entity_manifest + SOAP are ready (don't wait for injection).
    # When PHASE2_START_WITH_SOAP_READY=true (default), Phase 2 runs in parallel with constraint injection
    # using pre-injection SOAP note (~2-3s earlier start). Set false to use post-injection SOAP for accuracy.
    phase2_task = None
    knowledge_atoms_result = None
    stage_marks["phase2_start"] = None
    stage_marks["phase2_done"] = None
    phase2_start_with_soap_ready = os.getenv("PHASE2_START_WITH_SOAP_READY", "true").lower() in ("1", "true", "yes")

    if entity_manifest:
        # Start Phase 2 as soon as manifest + SOAP are ready (parallel with injection) when enabled
        if phase2_start_with_soap_ready and soap_note and soap_note.strip():
            try:
                from kb_phase2_integration import extract_knowledge_atoms_async
                phase2_enable_billing = os.getenv("PHASE2_ENABLE_BILLING_MATCHING", "false").lower() in ("1", "true", "yes")
                phase2_task = asyncio.create_task(
                    extract_knowledge_atoms_async(
                        soap_note_text=soap_note,  # Pre-injection SOAP (Phase 2 runs in parallel with injection)
                        entity_manifest=entity_manifest,
                        session_id=visit_id,
                        visit_id=visit_id,
                        clinic_id=clinic_id,
                        output_dir=output_dir_path,
                        logger=logger,
                        force_all_sections=False,
                        enable_billing_matching=phase2_enable_billing,
                        early_atoms_task_ref=early_phase2_task_ref,
                        run_timestamp=timestamp,
                    )
                )
                logger.info("🧬 Phase 2 Knowledge Atom extraction started (pre-extraction: parallel with constraint injection)")
                stage_marks["phase2_start"] = time.perf_counter()
            except ImportError as e:
                logger.debug(f"Phase 2 integration not available: {e}")
            except Exception as e:
                logger.warning(f"⚠️ Could not start Phase 2 (pre-extraction): {e}")

        # Anchor-Span: assign anchor_id (E1, E2, ...) for bi-directional SOAP <-> Billing sync
        use_anchor_tags = os.getenv("ANCHOR_SPAN_OUTPUT", "false").lower() in ("true", "1", "yes")
        if use_anchor_tags:
            try:
                from kb_anchor_span import ensure_anchor_ids_on_manifest
                ensure_anchor_ids_on_manifest(entity_manifest)
                logger.info("🔗 Anchor-Span: anchor_id assigned on manifest for bi-directional sync")
                # Re-save manifest so saved file includes anchor_id for frontend/debugging
                try:
                    entity_manifest_file = output_dir_path / f"entity_manifest_{timestamp}.json"
                    with open(entity_manifest_file, "w", encoding="utf-8") as f:
                        json.dump(entity_manifest, f, indent=2, ensure_ascii=False)
                    logger.info(f"🔗 Anchor-Span: Manifest re-saved with anchor_id to {entity_manifest_file}")
                except Exception as save_err:
                    logger.warning(f"⚠️ Anchor-Span: Could not re-save manifest with anchor_id: {save_err}")
            except Exception as e:
                logger.warning(f"⚠️ Anchor-Span: ensure_anchor_ids_on_manifest failed: {e} - continuing without anchor tags")
                use_anchor_tags = False

        # POST-SOAP INJECTION DISABLED:
        # Per production flow, SOAP consumes Brain NER + grounded manifest directly.
        # Do not run constraint/truth injector after SOAP generation.
        logger.info("⏭️ Skipping post-SOAP injection; using Brain NER/grounded manifest output directly.")

        # Start Phase 2 only if not already started (pre-extraction above); otherwise use post-injection SOAP
        if phase2_task is None and not phase2_start_with_soap_ready and soap_note and soap_note.strip():
            try:
                from kb_phase2_integration import extract_knowledge_atoms_async
                phase2_enable_billing = os.getenv("PHASE2_ENABLE_BILLING_MATCHING", "false").lower() in ("1", "true", "yes")
                phase2_task = asyncio.create_task(
                    extract_knowledge_atoms_async(
                        soap_note_text=soap_note,  # FINAL SOAP note (post injection)
                        entity_manifest=entity_manifest,
                        session_id=visit_id,
                        visit_id=visit_id,
                        clinic_id=clinic_id,
                        output_dir=output_dir_path,
                        logger=logger,
                        force_all_sections=False,
                        enable_billing_matching=phase2_enable_billing,
                        early_atoms_task_ref=early_phase2_task_ref,
                        run_timestamp=timestamp,
                    )
                )
                logger.info("🧬 Phase 2 Knowledge Atom extraction started (post-injection, using final SOAP note)")
                stage_marks["phase2_start"] = time.perf_counter()
            except ImportError as e:
                logger.debug(f"Phase 2 integration not available: {e}")
            except Exception as e:
                logger.warning(f"⚠️ Could not start Phase 2: {e}")
    
    # Phase 2 execution mode:
    # - Default non-blocking async: do not hold SOAP completion on downstream dashboard/atom extraction.
    # - Set PHASE2_BLOCKING=true to preserve legacy blocking behavior.
    phase2_blocking = os.getenv("PHASE2_BLOCKING", "false").lower() in ("1", "true", "yes")
    if phase2_task is not None and phase2_blocking:
        try:
            knowledge_atoms_result, phase2_success = await phase2_task
            stage_marks["phase2_done"] = time.perf_counter()
            if phase2_success:
                knowledge_atoms = knowledge_atoms_result.get('knowledge_atoms', [])
                logger.info(f"✅ Phase 2 complete: {len(knowledge_atoms)} knowledge atoms extracted")
            else:
                logger.warning(f"⚠️ Phase 2 extraction failed: {knowledge_atoms_result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.warning(f"⚠️ Phase 2 task failed: {e}")
            knowledge_atoms_result = None
    elif phase2_task is not None:
        def _phase2_done_callback(task: asyncio.Task) -> None:
            try:
                # Check if task was cancelled before accessing result
                if task.cancelled():
                    logger.debug("⚠️ Phase 2 task was cancelled (background task may have been interrupted)")
                    return
                
                result, ok = task.result()
                if ok:
                    atoms = (result or {}).get("knowledge_atoms", [])
                    logger.info("✅ Phase 2 complete (background): %d knowledge atoms extracted", len(atoms))
                else:
                    logger.warning(
                        "⚠️ Phase 2 extraction failed (background): %s",
                        (result or {}).get("error", "Unknown error"),
                    )
            except asyncio.CancelledError:
                logger.debug("⚠️ Phase 2 task was cancelled (background task interrupted)")
            except Exception as _e:
                logger.warning("⚠️ Phase 2 task failed (background): %s", _e)

        phase2_task.add_done_callback(_phase2_done_callback)
        logger.info("🧬 Phase 2 running in background (non-blocking); SOAP pipeline will not wait.")
        
        # Store phase2_task reference so we can wait for it before function returns
        # This prevents cancellation when asyncio.run() closes the event loop

    # Treat as success whenever we have usable SOAP content (avoids false "reported failure" when save/Phase 2 failed but content exists)
    if soap_note and soap_note.strip():
        success = True
    if not success:
        logger.error("❌ SOAP note generation reported failure, but content may still be available")
        if not soap_note or not soap_note.strip():
            raise RuntimeError("SOAP note generation failed in async pipeline - no content generated.")

    # Structured transcript (optional)
    try:
        # FIX: voice_tenor_analysis is in parent directory, not current directory
        voice_tenor_path = Path(__file__).parent.parent / "voice_tenor_analysis"
        if voice_tenor_path.exists() and str(voice_tenor_path) not in sys.path:
            sys.path.insert(0, str(voice_tenor_path))

        from soap_transcript_loader import SOAPTranscriptLoader

        # Use the segments file from THIS run if available; avoid accidentally loading a stale file.
        segments_file_path = output_dir_path / f"transcription_segments_{timestamp}.json"
        segments_file = str(segments_file_path) if segments_file_path.exists() else None

        loader = SOAPTranscriptLoader()
        structured_segments = loader.parse_cleaned_transcript(
            cleaned_transcription,
            transcription_segments_file=segments_file
        )
        # Fallback when loader returns [] (e.g. no segments file or parse failed): build from cleaned text
        if not structured_segments and cleaned_transcription:
            structured_segments = _structured_transcript_fallback(cleaned_transcription)
            if logger and structured_segments:
                logger.info("📄 Structured transcript: fallback from cleaned text (%d turns)", len(structured_segments))

        structured_transcript_file = output_dir_path / f"structured_transcript_{timestamp}.json"
        with open(structured_transcript_file, 'w', encoding='utf-8') as f:
            json.dump(structured_segments, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Structured transcript saved to: {structured_transcript_file}")
    except ImportError as e:
        # SOAPTranscriptLoader is optional (lives in voice_tenor_analysis or external package). Fallback builds structured transcript from cleaned text.
        logger.info("SOAPTranscriptLoader not available (%s); using built-in structured transcript fallback.", e)
        structured_segments = _structured_transcript_fallback(cleaned_transcription) if cleaned_transcription else []
        if structured_segments:
            structured_transcript_file = output_dir_path / f"structured_transcript_{timestamp}.json"
            with open(structured_transcript_file, 'w', encoding='utf-8') as f:
                json.dump(structured_segments, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Structured transcript (fallback) saved to: {structured_transcript_file}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create structured transcript: {e}")
        structured_segments = _structured_transcript_fallback(cleaned_transcription) if cleaned_transcription else []
        if structured_segments:
            try:
                structured_transcript_file = output_dir_path / f"structured_transcript_{timestamp}.json"
                with open(structured_transcript_file, 'w', encoding='utf-8') as f:
                    json.dump(structured_segments, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ Structured transcript (fallback) saved to: {structured_transcript_file}")
            except Exception:
                pass

    total_time = time.time() - pipeline_start_time
    logger.info(f"✅ Async pipeline completed in {total_time:.2f}s")
    # Stage-wise wall-clock contributions for latency audits.
    try:
        p0 = stage_perf_start
        t_trans = stage_marks.get("transcription_done")
        t_s2 = stage_marks.get("superpass_done")
        t_soap = stage_marks.get("soap_done")
        t_ground = stage_marks.get("grounding_done")
        t_billing_start = stage_marks.get("billing_start")
        t_p2s = stage_marks.get("phase2_start")
        t_p2e = stage_marks.get("phase2_done")
        p_end = time.perf_counter()

        def _d(a, b):
            if a is None or b is None:
                return None
            return max(0.0, float(b - a))

        d_trans = _d(p0, t_trans)
        d_super = _d(t_trans, t_s2)
        d_soap_pipeline = _d(t_s2, t_soap)
        d_billing_pipeline = _d(t_billing_start, t_ground) if t_billing_start is not None else _d(t_s2, t_ground)
        d_ground_tail = _d(t_soap, t_ground)
        d_phase2 = _d(t_p2s, t_p2e)
        d_total = _d(p0, p_end)

        logger.info(
            "⏱️ Stage Wall-Clock (s): transcription=%s, superpass=%s, soap_pipeline=%s, billing_pipeline=%s, grounding_tail_after_soap=%s, phase2=%s, total=%s",
            f"{d_trans:.2f}" if d_trans is not None else "n/a",
            f"{d_super:.2f}" if d_super is not None else "n/a",
            f"{d_soap_pipeline:.2f}" if d_soap_pipeline is not None else "n/a",
            f"{d_billing_pipeline:.2f}" if d_billing_pipeline is not None else "n/a",
            f"{d_ground_tail:.2f}" if d_ground_tail is not None else "n/a",
            f"{d_phase2:.2f}" if d_phase2 is not None else "n/a",
            f"{d_total:.2f}" if d_total is not None else "n/a",
        )
    except Exception:
        pass

    # Wait for Phase 2 background task to complete before returning
    # This prevents cancellation when asyncio.run() closes the event loop
    if phase2_task is not None and not phase2_blocking:
        try:
            # Wait for Phase 2 with a timeout (max 5 minutes) to avoid hanging forever
            # If it times out, we still return (Phase 2 will continue in background if possible)
            await asyncio.wait_for(phase2_task, timeout=300.0)
            if logger:
                logger.info("✅ Phase 2 background task completed before pipeline exit")
        except asyncio.TimeoutError:
            if logger:
                logger.warning("⚠️ Phase 2 background task timed out after 5 minutes; continuing anyway")
        except asyncio.CancelledError:
            if logger:
                logger.debug("⚠️ Phase 2 background task was cancelled during wait")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error waiting for Phase 2 task: {e}")

    return {
        "soap_note": soap_note,
        "manifest": entity_manifest,
        "cleaned_text": cleaned_transcription,
        "knowledge_atoms": knowledge_atoms_result  # Phase 2 results (if available)
    }

# ==============================================================================
# OPTIONAL: CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    # Initialize logging immediately when script starts
    logger = setup_logging(DEFAULT_CONFIG.output_dir)
    logger.info("🚀 Script started - logging initialized")
    logger.info(f"🕐 Script execution started at {datetime.now().strftime('%H:%M:%S')}")
    
    parser = argparse.ArgumentParser(description="Generate SOAP note from audio file.")
    parser.add_argument("audio_file", nargs='?', help="Path to audio file (wav/mp3/etc). If not provided, uses the first audio file in the input folder.")
    parser.add_argument("--output", help="Output directory for SOAP note", default=None)
    parser.add_argument("--convert", action="store_true", help="Convert audio file to WAV format before processing")
    parser.add_argument("--check-deps", action="store_true", help="Check audio processing dependencies")
    parser.add_argument("--sync-pipeline", action="store_true", help="Run legacy sequential pipeline (async is default)")
    args = parser.parse_args()
    
    logger.info("📋 Arguments parsed successfully")
    
    try:
        if args.check_deps:
            logger.info("🔍 Audio Processing Dependencies Check")
            logger.info("=" * 50)
            check_audio_dependencies()
            logger.info("✅ Dependencies check completed")
            sys.exit(0)
        
        # Handle audio conversion if requested
        audio_file = args.audio_file
        if args.convert and audio_file:
            logger.info(f"🔄 Audio conversion requested for: {audio_file}")
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                sys.exit(1)
            
            try:
                converted_file = convert_audio_to_wav(audio_file)
                logger.info(f"✅ Using converted file: {converted_file}")
                audio_file = converted_file
            except Exception as e:
                logger.error(f"❌ Error converting audio file: {e}")
                sys.exit(1)
        
        logger.info("🎤 Starting SOAP note generation from audio...")
        if args.sync_pipeline:
            note = generate_soap_note_from_audio(audio_file, output_dir=args.output)
        else:
            result = asyncio.run(generate_soap_note_from_audio_async(audio_file, output_dir=args.output))
            note = result["soap_note"]
        
        # --- SOAP Note Modification step (COMMENTED OUT to avoid duplicate output) ---
        # logger.info("📝 SOAP Note Modification step")
        # print("\n--- SOAP Note Modification ---")
        # user_input = input("Enter modification instructions (or press Enter to skip): ")
        # current_soap_note = note
        # if user_input.strip():
        #     logger.info("🔄 Applying modifications to SOAP note...")
        #     # Create a temporary SOAPNoteGenerator to handle modifications
        #     temp_config = Config(
        #         input_transcription_path="",  # Not needed for modifications
        #         output_dir=args.output or DEFAULT_CONFIG.output_dir,
        #         api_key_file=DEFAULT_CONFIG.api_key_file,
        #         model_provider=DEFAULT_CONFIG.model_provider,
        #         model_name=DEFAULT_CONFIG.model_name,
        #         pre_appointment_summary_path=DEFAULT_CONFIG.pre_appointment_summary_path,
        #         protocols_template_path=DEFAULT_CONFIG.protocols_template_path,
        #         vitals_template_path=DEFAULT_CONFIG.vitals_template_path,
        #         request_timeout=DEFAULT_CONFIG.request_timeout,
        #         max_retries=DEFAULT_CONFIG.max_retries
        #     )
        #     temp_generator = SOAPNoteGenerator(temp_config)
        #     
        #     # Use the LLM to apply modifications
        #     mod_prompt = f"""
        # You are a clinical documentation assistant. Here is a SOAP note:
        #
        # {current_soap_note}
        #
        # Here are the requested modifications:
        # {user_input}
        #
        # Please return the revised SOAP note, making only the requested changes and keeping all other content unchanged.
        # """
        #     revised_soap_note = temp_generator.llm_provider.generate_soap_note(
        #         conversation=mod_prompt,
        #         pre_appointment="",
        #         protocols="",
        #         vitals=""
        #     )
        #     current_soap_note = revised_soap_note
        #     logger.info("✅ SOAP note modifications applied successfully")
        #     print("\nFinal SOAP Note after modification:\n")
        #     print(current_soap_note)
        # else:
        #     logger.info("⏭️ No modifications requested")
        #     print("\nNo modifications made. Final SOAP Note:\n")
        #     print(current_soap_note)
        # --- End single modification step ---
    except Exception as e:
        err_str = str(e)
        is_fireworks_5xx = "502" in err_str or "503" in err_str or "temporarily unavailable" in err_str.lower() or "Bad Gateway" in err_str
        logger.error(f"❌ Critical error: {e}")
        logger.error("💡 Troubleshooting tips:")
        n = 1
        if is_fireworks_5xx:
            logger.error("1. Fireworks transcription service may be temporarily down (502/503). Wait a few minutes and try again.")
            logger.error("2. Check status.fireworks.ai or your network/firewall.")
            n = 3
        logger.error("%d. Run with --check-deps to verify audio libraries", n)
        logger.error("%d. Try --convert to convert problematic audio files to WAV", n + 1)
        logger.error("%d. Install missing libraries: pip install pydub librosa ffmpeg-python", n + 2)
        logger.error("%d. WebM files are now supported and will be automatically converted", n + 3)
        logger.error("%d. For WebM files, ensure ffmpeg is installed on your system", n + 4)
        print(f"Error: {e}")
        print("\n💡 Troubleshooting tips:")
        n = 1
        if is_fireworks_5xx:
            print("1. Fireworks transcription service may be temporarily down (502/503). Wait a few minutes and try again.")
            print("2. Check status.fireworks.ai or your network/firewall.")
            n = 3
        print(f"{n}. Run with --check-deps to verify audio libraries")
        print(f"{n + 1}. Try --convert to convert problematic audio files to WAV")
        print(f"{n + 2}. Install missing libraries: pip install pydub librosa ffmpeg-python")
        print(f"{n + 3}. WebM files are now supported and will be automatically converted")
        print(f"{n + 4}. For WebM files, ensure ffmpeg is installed on your system")
        sys.exit(1)

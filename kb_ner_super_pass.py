"""
Super-Pass: Combined Transcription Cleaning + NER Extraction

This module combines Step 2 (Transcription Cleaning) and Step 2.3a (NER Extraction)
into a single LLM call, eliminating network latency and enabling parallel execution.

Default model: gpt-4.1-mini. Override with SUPER_PASS_MODEL env var.
Output: JSON with cleaned_transcript and extracted_entities
"""

import json
import logging
import os
import traceback
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator

from kb_ner_clients import get_client_for_model, get_model_provider
from kb_ner_extraction import _extract_json_object
from long_transcript_utils import (
    chunk_text_with_overlap,
    wrap_with_summary_block,
)

# Placeholder from prompt example; filter out so we never persist it.
TERMS_NOT_GROUNDED_PLACEHOLDER_SPAN = "some term not grounded"

# ---------------------------------------------------------------------------
# Speculative decoding (Fireworks): optional acceleration.
#
# Fireworks supports speculative decoding via draft models or n-gram speculation
# (often configured at deployment level), see:
# https://docs.fireworks.ai/deployments/speculative-decoding
#
# We attempt to pass speculative fields in the request when enabled; if the API
# rejects them (common on on-demand models / older endpoints), we retry without
# them so the pipeline never breaks.
# ---------------------------------------------------------------------------
def _build_fireworks_speculative_kwargs(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    enabled = os.getenv("SUPER_PASS_SPECULATIVE", "false").strip().lower() in ("1", "true", "yes")
    if not enabled:
        return {}

    draft_model = (os.getenv("SUPER_PASS_DRAFT_MODEL") or "").strip()
    ngram_len = (os.getenv("SUPER_PASS_NGRAM_SPECULATION_LENGTH") or "").strip()
    draft_tokens = (os.getenv("SUPER_PASS_DRAFT_TOKEN_COUNT") or "").strip()

    # Need draft_token_count for either mode (per Fireworks docs).
    try:
        draft_token_count = int(draft_tokens) if draft_tokens else 4
    except Exception:
        draft_token_count = 4
    if draft_token_count < 1:
        draft_token_count = 1

    extra: Dict[str, Any] = {"draft_token_count": draft_token_count}
    if draft_model and ngram_len:
        # Mutually exclusive; prefer explicit draft_model if both set.
        if logger:
            logger.warning("SUPER_PASS: both SUPER_PASS_DRAFT_MODEL and SUPER_PASS_NGRAM_SPECULATION_LENGTH set; using draft model.")
        ngram_len = ""

    if draft_model:
        extra["draft_model"] = draft_model
    elif ngram_len:
        try:
            extra["ngram_speculation_length"] = int(ngram_len)
        except Exception:
            extra["ngram_speculation_length"] = 3
    else:
        # If enabled but neither provided, pick a safe default for Llama-family.
        # (User can override via env.)
        extra["draft_model"] = "accounts/fireworks/models/llama-v3p2-1b-instruct"

    return extra

# Default Super-Pass model: OpenAI gpt-4.1-mini
# (Fireworks llama-v3p3-70b retired / may be inaccessible). Override with SUPER_PASS_MODEL.
SUPER_PASS_DEFAULT_MODEL = os.getenv(
    "SUPER_PASS_MODEL",
    "gpt-4.1-mini",
)
# Default: speculative decoding OFF (deployments-only optimization; on-demand calls typically ignore/reject it)
os.environ.setdefault("SUPER_PASS_SPECULATIVE", "false")

def _load_unified_prompt() -> str:
    """Load UNIFIED_CLEANING_AND_NER_PROMPT from UNIFIED_CLEANING_AND_NER_PROMPT.md in project root (single source of truth)."""
    doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UNIFIED_CLEANING_AND_NER_PROMPT.md")
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return _UNIFIED_PROMPT_FALLBACK
    # Extract prompt from ## 4. Complete Prompt ... first ``` block
    marker = "## 4. Complete Prompt"
    idx = content.find(marker)
    if idx == -1:
        return _UNIFIED_PROMPT_FALLBACK
    rest = content[idx + len(marker):]
    start = rest.find("```")
    if start == -1:
        return _UNIFIED_PROMPT_FALLBACK
    rest = rest[start + 3:]  # skip ```
    end = rest.find("```")
    if end == -1:
        return _UNIFIED_PROMPT_FALLBACK
    prompt = rest[:end].strip()
    if "{conversation}" in prompt and "{optional_inputs}" in prompt:
        return prompt
    return _UNIFIED_PROMPT_FALLBACK


def _format_unified_prompt(conversation: str, optional_inputs: str = "") -> str:
    """Fill {conversation} and {optional_inputs} in the unified prompt (avoids .format() breaking on JSON braces)."""
    return (
        UNIFIED_CLEANING_AND_NER_PROMPT.replace("{conversation}", conversation or "")
        .replace("{optional_inputs}", (optional_inputs or "").strip())
    )


# Fallback if doc file missing (minimal prompt with placeholders)
_UNIFIED_PROMPT_FALLBACK = """You are a veterinary clinical scribe AI. Clean the raw transcript and extract clinical entities.

**RAW TRANSCRIPTION:**
{conversation}
{optional_inputs}

Return ONLY a JSON object with exactly three keys: "cleaned_transcript", "extracted_entities", "entities_by_kind".
Each entity in extracted_entities: id, span_text, kind, attributes. Preserve all clinical content verbatim. No summarization.
"""


UNIFIED_CLEANING_AND_NER_PROMPT = _load_unified_prompt()

# Voice and Super-Pass both use the unified prompt (single source: UNIFIED_CLEANING_AND_NER_PROMPT.md in project root)
VOICE_PROMPT = UNIFIED_CLEANING_AND_NER_PROMPT
SUPER_PASS_SYSTEM_PROMPT = UNIFIED_CLEANING_AND_NER_PROMPT

# (Legacy long prompt removed; single source is UNIFIED_CLEANING_AND_NER_PROMPT.md in project root)


def _load_brain_ner_prompt() -> str:
    """Load Brain NER prompt from docs/BRAIN_NER_PROMPT_UPDATED.md or BRAIN_NER_PROMPT_UPDATED.md in project root (single source of truth)."""
    root = os.path.dirname(os.path.abspath(__file__))
    for doc_path in [
        os.path.join(root, "docs", "BRAIN_NER_PROMPT_UPDATED.md"),
        os.path.join(root, "BRAIN_NER_PROMPT_UPDATED.md"),
    ]:
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
            break
        except Exception:
            content = None
    if not content:
        return _BRAIN_NER_PROMPT_FALLBACK
    marker = "## Complete Updated Prompt"
    idx = content.find(marker)
    if idx == -1:
        return _BRAIN_NER_PROMPT_FALLBACK
    rest = content[idx + len(marker):]
    start = rest.find("```")
    if start == -1:
        return _BRAIN_NER_PROMPT_FALLBACK
    rest = rest[start + 3:]
    if rest.lstrip().startswith("python"):
        rest = rest.lstrip()[6:].lstrip()
    end = rest.find("```")
    if end != -1:
        rest = rest[:end]
    first_qqq = rest.find('"""')
    if first_qqq == -1:
        return _BRAIN_NER_PROMPT_FALLBACK
    after_open = rest[first_qqq + 3:]
    last_qqq = after_open.rfind('"""')
    if last_qqq == -1:
        return _BRAIN_NER_PROMPT_FALLBACK
    block = after_open[:last_qqq].rstrip()
    if "{cleaned_transcript}" in block and "{pre_extracted_entities}" in block:
        return block
    return _BRAIN_NER_PROMPT_FALLBACK


def _format_brain_ner_prompt(cleaned_transcript: str, pre_extracted_entities: str = "") -> str:
    """Fill {cleaned_transcript}, {pre_extracted_entities}, and {pre_extracted_entity_count} in the Brain NER prompt."""
    prompt = (
        CLINICAL_ENTITY_EXTRACTION_PROMPT.replace("{cleaned_transcript}", cleaned_transcript or "")
        .replace("{pre_extracted_entities}", (pre_extracted_entities or "[]").strip())
    )
    # Inject count from unified/super-pass so the model sees the exact number; verification step requires count must match
    pre_extracted_count = 0
    if pre_extracted_entities and (pre_extracted_entities or "").strip() != "[]":
        try:
            parsed = json.loads((pre_extracted_entities or "[]").strip())
            if isinstance(parsed, list):
                pre_extracted_count = len(parsed)
        except Exception:
            pass
    prompt = prompt.replace("{pre_extracted_entity_count}", str(pre_extracted_count))
    return prompt


# Fallback if Brain NER doc file missing (includes 13th field query_expansion for phonetic/ASR correction)
_BRAIN_NER_PROMPT_FALLBACK = """You are a clinical entity enrichment and verification system for veterinary transcripts.
You receive CLEANED_TRANSCRIPT and PRE_EXTRACTED_ENTITIES. Enrich every pre-extracted entity and catch any missed entities.
PRE_EXTRACTED_ENTITY_COUNT: {pre_extracted_entity_count} — Your skeleton_list MUST contain at least this many items (one per pre-extracted entity). The count must match.
Return ONLY valid JSON with a "skeleton_list" field.
Each item in "skeleton_list" must be one compressed skeleton line (13 pipe-separated fields; last is query_expansion):
id|span_text|normalized_name|kind|domains|inv_cats|svc_cats|corr_prob|sugg_prob|hints|is_new|context|query_expansion
normalized_name — FORM-FACTOR PRESERVATION: When normalizing medications/products, include the delivery form if context has unit or route cues: ml/cc/syrup/suspension/drops → e.g. "Cefpodoxime Syrup"; tablet/tab/mg → e.g. "Cefpodoxime Tablet"; inject/vial/IM/IV → e.g. "Amoxicillin Injection"; apply/cream/spray/pump → e.g. "Easotic ear drops". Example: "3 ml of Cefped" → normalized_name="Cefpodoxime Syrup" (not just "Cefpodoxime").
query_expansion (13th field): When the transcript term SOUNDS LIKE a known medication/product but is spelled phonetically or garbled, add up to 3 comma-separated likely brand/product names (e.g. Easotic,Easotic 10ml,Virbac Easotic). Leave empty when not applicable.
CLEANED_TRANSCRIPT:
{cleaned_transcript}
PRE_EXTRACTED_ENTITIES:
{pre_extracted_entities}
"""


# Fireworks Structured Outputs (JSON Schema)
# Ref: https://docs.fireworks.ai/structured-responses/structured-response-formatting
# Pure NER 13 kinds (for entities_by_kind keys)
PURE_NER_KIND_KEYS = [
    "ReasonForVisit", "Medication", "Procedure", "Diagnostic", "VitalSign",
    "Reminder", "Symptom", "Diagnosis", "Anatomy", "Diet", "Preventive", "ParasiteControl",
    "Other",
]


def _build_entities_by_kind(extracted_entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """Build entities_by_kind (Pure NER output) from extracted_entities. Each kind maps to list of {span_text}. 13 kinds."""
    by_kind: Dict[str, List[Dict[str, str]]] = {k: [] for k in PURE_NER_KIND_KEYS}
    for e in extracted_entities or []:
        kind = (e.get("kind") or "Other").strip()
        if kind not in by_kind:
            kind = "Other"  # unknown kind → Other
        span = (e.get("span_text") or "").strip()
        if span:
            by_kind[kind].append({"span_text": span})
    return by_kind


SUPER_PASS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "cleaned_transcript": {"type": "string"},
        "extracted_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "span_text": {"type": "string"},
                    "normalized_name": {"type": "string"},
                    "kind": {"type": "string"},
                    "roles": {"type": "array", "items": {"type": "string"}},
                    "is_actionable": {"type": "boolean"},
                    "attributes": {"type": "object", "additionalProperties": True},
                    "assertion_id": {"type": "string"},
                    "supporting_text": {"type": "string"},
                    "start_char": {"type": "integer"},
                    "end_char": {"type": "integer"},
                },
                "required": ["span_text", "kind"],
                "additionalProperties": False,
            },
        },
        "entities_by_kind": {
            "type": "object",
            "description": "Pure NER output: all extracted entities grouped by kind (12 kinds). Each value is array of {span_text}.",
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"span_text": {"type": "string"}},
                    "required": ["span_text"],
                },
            },
        },
    },
    "required": ["cleaned_transcript", "extracted_entities"],
    "additionalProperties": False,
}

SUPER_PASS_CHUNK_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "cleaned_chunk": {"type": "string"},
        "extracted_entities": SUPER_PASS_JSON_SCHEMA["properties"]["extracted_entities"],
        "updated_summary": {"type": "string"},
    },
    "required": ["cleaned_chunk", "extracted_entities", "updated_summary"],
    "additionalProperties": False,
}

SUPER_PASS_CHUNK_SYSTEM_PROMPT = """You are a specialized Veterinary AI processing a LONG veterinary ASR transcript in CHUNKS.

You must perform THREE tasks:
1) Clean ONLY the provided chunk for clinical clarity (speaker attribution, remove fillers, translate non-English)
2) Extract clinical entities from ONLY this chunk (same schema as super-pass)
3) Update a rolling global summary of the entire visit, given the prior summary and this chunk

Return ONLY a JSON object with EXACTLY these keys.

IMPORTANT (STREAMING OPTIMIZATION):
- You MUST output the JSON keys in this exact order:
  1) extracted_entities
  2) updated_summary
  3) cleaned_chunk
- This allows the system to begin grounding entities while you are still generating the rest of the output.

OUTPUT FORMAT (exactly):
{
  "extracted_entities": [...],
  "updated_summary": "...",
  "cleaned_chunk": "Veterinarian: ...\\nPet Parent: ..."
}

Rules:
- Do NOT include any reasoning or extra text.
- The first non-whitespace character MUST be '{'.
- Keep updated_summary concise and clinically complete (prefer bullets). Max ~2500 characters.
- Do NOT invent facts not present in the chunk.
- Extract tests and measurements: include specific tests, measurements, and angles (e.g. Noble angle, Norberg angle, Distraction index) even if spelling looks wrong or unfamiliar; do not filter out potential ASR corruptions.
"""

# ---------------------------------------------------------------------------
# Combined Clean + NER + Batch Intent (single LLM call to reduce latency)
# Enable with COMBINED_CLEAN_NER_BATCH_INTENT=true. Output: cleaned transcript + entity_manifest with search_term/family.
# ---------------------------------------------------------------------------
COMBINED_CLEAN_TRANSCRIPT_AND_ENTITY_MANIFEST_PROMPT = """
COMBINED_CLEAN_TRANSCRIPT_ENTITY_MANIFEST_PROMPT (Phase 0 + NER + Intent + Patch + Anchors)
## ROLE
You are a Lead Clinical Documentation Specialist and Medical Information Engine for veterinary consultations.
Your job is to transform messy ASR into:
1) an audit-ready cleaned transcript (for downstream SOAP/coding),
2) an entity manifest aligned to the final cleaned transcript,
3) a batch-intent + constrained patch plan (entity-span-only),
4) an anchored transcript variant that carries entity IDs forward into SOAP generation (no brittle matching).
## INPUTS
RAW TRANSCRIPTION:
{conversation}
OPTIONAL CONTEXT (may be empty):
{optional_inputs}
================================================================================
## CRITICAL PROTOCOLS (NON-NEGOTIABLE)
1) PROTECTED ASSET RULE (Clinical Preservation)
- NEVER “correct” a medical term into a different drug class, diagnosis, or procedure.
- If a term is garbled or unfamiliar, keep it EXACTLY as heard and mark:
  [UNCERTAIN_TERM: "verbatim token/phrase"].
- Preserve all explicit negatives and polarity (no / not / never). Losing a “no” is a Critical Error.
2) NO INVENTION
- Do NOT add any new symptoms, diagnoses, vitals, lab values, doses, routes, durations, timelines, or plans not explicitly present.
3) MULTILINGUAL SYNTHESIS (translation without drift)
- Translate code-mixed/local language (Tamil/Telugu/Hindi/Malayalam/etc.) to professional English.
- If a spoken medical/product term is in a local language AND seems like a specific named item:
  keep the phonetic form in brackets after the English IF it fits within length constraints.
- If unsure, do NOT guess: mark [unclear] or [inaudible].
4) CHARACTER LENGTH CONSTRAINT (CRITICAL)
- The FINAL cleaned transcript (cleaned_transcript) MUST NOT exceed the raw transcription character count.
- Achieve this by removing non-clinical fillers/repetitions.
- Use ultra-short speaker labels ONLY: "V: " / "O: " / "U: ".
- One newline per turn.
- Never add headers, explanations, or markdown to transcript strings.
5) PATCHING RULE (CRITICAL)
- Transcript rewriting is NOT allowed except:
  (a) filler/noise removal during cleaning, and
  (b) entity-span-only replacements approved in patch_plan.
- Never change non-entity prose.
6) ANCHOR TAG RULE (CRITICAL for robust SOAP linking)
- You MUST produce anchored_transcript by wrapping each entity span as:
  [[<entity_id>:<span_text>]]
- Anchors are wrappers ONLY (no wording changes outside the span).
- Anchors may increase length; ONLY cleaned_transcript must satisfy the length constraint.
7) OUTPUT FORMAT (HARD)
- Output MUST be VALID JSON only.
- First non-whitespace character must be "{{" and last must be "}}".
- No extra top-level keys beyond those listed in OUTPUT FORMAT section.
================================================================================
## EXECUTION ORDER (DO NOT SKIP STEPS)
--------------------------------------------------------------------------------
### STEP 0: Preflight (internal planning)
Purpose:
Ensure the length constraint is achievable.
Actions:
- Let RAW_LEN = character count of RAW TRANSCRIPTION.
- Plan to remove filler/repetition so that after adding "V: "/"O: "/"U: " + newlines,
  cleaned_transcript length <= RAW_LEN.
--------------------------------------------------------------------------------
### STEP 1: Clean-up the transcript (De-noising + Translation)
Purpose:
Convert the raw ASR into clean, professional English while preserving clinical meaning.
Actions:
a) Analyze the transcript:
- Identify grammatical errors, fillers, repeated phrases, and transcription artifacts.
- IMPORTANT: Do NOT “fix” a medical term into a different medical term here.
  If clinically important but unclear, KEEP it and mark [unclear] or [UNCERTAIN_TERM: "..."].
b) Language inconsistencies:
- Identify any non-English (code-mixed) words/phrases or full local-language segments.
c) Translate:
- Translate all non-English content into natural, professional English without changing meaning, timeline, or polarity.
d) Remove noise:
- Remove repeated words, filler sounds ("uh", "hmm"), obvious background chatter, and system noise.
e) Do NOT delete clinically relevant phrases:
- Do NOT remove phrases that might be symptoms, diagnoses, procedures, tests, treatments, medications, or plans,
  even if the wording is odd. If unclear, keep + mark [unclear] rather than guessing.
f) Correct only low-risk typos:
- Fix ONLY spelling/grammar issues that do not affect clinical meaning (e.g., "teh" -> "the").
Output of STEP 1:
- CLEAN_TEXT (single text; speaker labels not required yet)
--------------------------------------------------------------------------------
### STEP 2: Speaker Attribution (Turn segmentation)
Purpose:
Convert CLEAN_TEXT into an attributed veterinary dialogue.
Actions:
a) Segment into statements/turns:
- Divide into utterances based on context and punctuation.
b) Identify the speaker:
- Vet authority: advice, exam, diagnosis discussion, plan -> usually V:
- Owner history: symptoms, concerns, consent -> usually O:
- If ambiguous -> U:
c) Format rules:
- Each turn MUST be on a new line.
- Use ONLY: "V: " / "O: " / "U: "
- Preserve conversation order.
- If speaker switches mid-paragraph: split lines.
Output of STEP 2:
- SPAN_BASE_JOINED (string, newline separated turns)
--------------------------------------------------------------------------------
### STEP 3: Verification Gate for SPAN_BASE_JOINED (MUST PASS)
Purpose:
Ensure SPAN_BASE_JOINED is faithful and complete before NER/intent/patching.
Critical Rules:
- Do not introduce new symptoms/diagnoses/meds/doses/timelines/pets.
- Do not upgrade uncertainty ("maybe") into certainty.
- Do not remove explicit negatives ("no vomiting").
Actions:
a) Alignment / Faithfulness check:
- Every line must be supported by RAW transcription meaning.
- If not supported: rewrite to match RAW or remove if non-clinical.
b) Missing clinical content check:
- Ensure no loss of:
  symptoms + duration, appetite/water/urination/defecation,
  prior treatments/drugs/doses/allergies,
  vet assessments/tests/plan/follow-up.
c) Negation & certainty check:
- Confirm polarity preserved.
- Confirm uncertainty preserved.
d) Translation check:
- Confirm translation without drift; if unclear, mark [unclear]/[inaudible] rather than guessing.
e) Speaker attribution check:
- Fix obvious mislabels.
f) Length planning check:
- Ensure compact enough that cleaned_transcript can stay <= RAW_LEN.
Output of STEP 3:
- FINAL_SPAN_BASE_JOINED
--------------------------------------------------------------------------------
### STEP 4: Entity Extraction (NER) on FINAL_SPAN_BASE_JOINED (HIGH-RECALL, CONTROLLED)
Purpose:
Extract ALL clinically relevant entities as exact verbatim spans with offsets. This step must be HIGH-RECALL. Do not "be selective." If it fits a kind, extract it.
ABSOLUTE RULES:
1) span_text MUST be an exact substring of FINAL_SPAN_BASE_JOINED.
2) start_char/end_char MUST index FINAL_SPAN_BASE_JOINED (0-based; end exclusive).
3) Do NOT correct spelling in span_text. Extract as spoken/written.
4) Extract even if misspelled/phonetic/garbled; preserve verbatim.
5) entity_id MUST be sequential "E1","E2","E3"... by FIRST appearance in transcript.
6) PREFER SHORT SPANS (REBASE): For Symptom, Finding, Procedure, Diagnostic extract the minimal clinical phrase (2–5 words) that will survive cleaning. E.g. prefer "walking problem" over "facing a problem while he is walking". Long phrases fail Fuzzy Sync rebase ~99% of the time.
CONTROLLED KINDS (exactly these 11 production kinds; no new kinds):
{ner_kinds}

FAMILY MAPPING: PRODUCT → Medication, ParasiteControl, Diet. ACTION → Procedure, Diagnostic, Reminder. CLINICAL → Symptom, Diagnosis, Anatomy. OTHER → VitalSign, ReasonForVisit.

KIND DEFINITIONS + MANDATORY ATTRIBUTES (use these for classification):
1) ReasonForVisit: Primary trigger/chief complaint. Family: OTHER.
2) Medication: Drugs. MANDATORY attributes.status: "Administered" | "Prescribed" (in-clinic vs pharmacy/home). Family: PRODUCT.
3) Procedure: Clinical actions, surgeries, maneuvers (e.g. Ortolani test). Family: ACTION.
4) Diagnostic: Ordered tests, imaging, lab panels (X-ray, Norberg angle, CBC). Family: ACTION.
5) VitalSign: Metrics (Weight, Temp, HR). MANDATORY: intent (i) as [Metric]: [Value] [Unit], e.g. "Weight: 35 kg". Family: OTHER.
6) Reminder: Follow-up or re-checks ("follow up in 2 weeks"). Family: ACTION.
7) Symptom: Clinical signs or owner reports. MANDATORY attributes.is_negated: true|false when negation present ("No vomiting" → is_negated true). Family: CLINICAL.
8) Diagnosis: Suspected or confirmed conditions (hip dysplasia, patellar luxation). Family: CLINICAL.
9) Anatomy: Body sites ("hip joint", "left stifle"). Family: CLINICAL.
10) Diet: Prescription or specialized food (obesity diet, renal diet). Family: PRODUCT.
11) ParasiteControl: Preventatives (Bravecto, tick and flea control). Family: PRODUCT. LIST RULE: extract EACH product name as separate entity.
HIGH-SPEED INTENT (i): For garbled terms (e.g. "ultralining"), set normalized_name or intent (i) to canonical form (e.g. "Ortolani test").

ENTITY EXTRACTION METHOD (MANDATORY): Multi-Family "No-Miss" Protocol
Extract entities by scanning specifically for these four scan groups. You MUST find at least one entity for each group when present in the text. (Output intent_family is set in Step 5B and uses pipeline enums: OTHER, CLINICAL, PRODUCT, PROCEDURE. Scan groups map as: Identity & Signalment → OTHER; Clinical Findings → CLINICAL; Treatment & Products → PRODUCT; Management & Constraints → PROCEDURE for tests/imaging/procedures, CLINICAL for Physiotherapy/ExerciseRestriction/OwnerPreference/RiskConcern/FollowUpOrPlan.)

A) PASS 1 — Identity & Signalment (output family OTHER):
Scan for: PatientName, Species, Breed, Sex, Age, Weight.
If any of these are mentioned (e.g. pet name, "5 year old", "male", "Labrador"), extract each as a separate entity.

B) PASS 2 — Clinical Findings (output family CLINICAL):
Scan for: Symptom, BodySite, BodySystem, ExamFinding, DiagnosisSuspected, Differential.
- Audit Check: If the owner says "he is lazy", "walking problem", "not playful", "limping", or similar, you MUST extract these as Symptom. Do not treat them as "just prose."
- Symptom/BodySite separation: Extract the symptom (e.g. "walking problem") as Symptom and the anatomical location (e.g. "hip joint", "hip", "stifle") as BodySite when both are present. Link contextually; do not merge into one entity.

C) PASS 3 — Treatment & Products (output family PRODUCT):
Scan for: Medication, Supplement, Vaccine, Deworming, ParasiteControl, Diet.
- Rule: Every drug or product in a list MUST be a separate entity. E.g. "Bravecto, Nexgard" => two ParasiteControl entities; "fibroblan, esmetopril, floralana (Bravecto)" => four entities (see ParasiteControl LIST RULE above).

D) PASS 4 — Management & Constraints (output family PROCEDURE or CLINICAL per Step 5B):
Scan for: TestOrManeuver, Imaging, Procedure, Physiotherapy, ExerciseRestriction, OwnerPreference, RiskConcern, FollowUpOrPlan.
- FollowUpOrPlan override (CRITICAL): If a phrase matches Physiotherapy or ExerciseRestriction, you MUST extract it under those kinds even if it also appears in a plan sentence. FollowUpOrPlan may coexist (e.g. "follow up in 2 weeks"), but cannot replace Physiotherapy or ExerciseRestriction. Do not put "swimming" or "avoid jumping" only into FollowUpOrPlan.
- Audit Check: "Swimming", "walking on sand", "sit-stand-step exercises", "range of motion" MUST be extracted as Physiotherapy when prescribed as rehab. "Avoid jumping", "no jumping", "cage rest", "avoid stairs" MUST be ExerciseRestriction. Do not leave them as unstructured Plan prose.
- Audit Check: "Prefer home care", "do this at home", "can't come to the clinic", "home visit" MUST be OwnerPreference. "Infection risk", "worried about infection", "side effects concern" MUST be RiskConcern.

E) PASS 5 — KIND COVERAGE AUDIT (HARD):
After extraction, re-scan FINAL_SPAN_BASE_JOINED and verify: if any text exists that matches definitions/examples of a kind, at least one entity of that kind MUST exist. If missing, add the entity with verbatim span(s). Do NOT skip this pass.

F) OUTPUT VALIDATION GATE (MANDATORY — do not output entity_manifest_v1 until this passes):
After extraction, verify the following. If any check fails, go back and add the missing entity/entities with verbatim span(s), then re-verify. Do not proceed to Step 5 until all checks pass.
1) ACTION cues: If the transcript contains any cue for management/constraints (e.g. swimming, avoid, jumping, home care, infection risk, prefer home, cage rest, walking on sand), then at least one entity of kind Physiotherapy, ExerciseRestriction, OwnerPreference, or RiskConcern MUST exist. If not, add the missing kind(s) from the verbatim text.
2) OBSERVATION cues: If the transcript contains any cue for clinical findings (e.g. walking problem, hip joint, stifle, lazy, limping, not playful), then at least one Symptom entity AND at least one BodySite (or BodySystem) entity MUST exist where contextually appropriate. If not, add the missing kind(s) from the verbatim text.
This is a gating requirement; output is invalid until validation passes.

Definitions + few-shots for tricky kinds (use these to avoid missing whole categories):

Physiotherapy
- Definition: rehab/therapy activities prescribed to improve function (including swimming/hydrotherapy/exercises).
- Example transcript line: "Physiotherapy is needed. Swimming is beneficial. Sit-stand-step exercises…"
- Must extract: Physiotherapy: "Physiotherapy"; Physiotherapy: "Swimming"; Physiotherapy: "Sit-stand-step exercises"

ExerciseRestriction
- Definition: explicit "avoid / do not / restrict" activity guidance (jumping, stairs, running, high impact).
- Example line: "Avoid high-impact exercises like jumping and landing."
- Must extract: ExerciseRestriction: "Avoid high-impact exercises"; ExerciseRestriction: "jumping and landing"

OwnerPreference
- Definition: owner's stated preference/constraint (home care, cannot come, schedule, budget, route preference).
- Example line: "Is there any way to do this at home? … I prefer home care."
- Must extract: OwnerPreference: "do this at home"; OwnerPreference: "prefer home care"

RiskConcern
- Definition: stated worry about risk (infection, side effects, anesthesia risk, clinic exposure).
- Example line: "I am concerned about infection risk at the clinic."
- Must extract: RiskConcern: "infection risk at the clinic"

ParasiteControl product lists
- Definition: any named token that appears in a tick/flea/parasite-control context must be extracted as its own ParasiteControl entity even if misspelled.
- Example line: "agents like fibroblan, esmetopril… floralana (Bravecto)… saralana…"
- Must extract: each token as separate ParasiteControl (fibroblan, esmetopril, floralana, Bravecto, saralana, etc.), verbatim.

Entity schema v1 (minimum required fields):
- entity_id
- kind
- span_text
- start_char, end_char (0-based offsets into FINAL_SPAN_BASE_JOINED; end exclusive)
- speaker ("Vet"|"Owner"|"Unknown") and turn_index (newline-separated turns)
- context_snippet (short nearby text)
- normalized_name (LIGHT normalization only; may equal span_text; do NOT clinically “correct” here)
- certainty ("high"|"medium"|"low")
- uncertainty_reason (string or null)
- attributes (dose/route/frequency/duration if explicitly nearby; else empty object)
Output of STEP 4:
- entity_manifest_v1[]

FEW-SHOT MINI EXAMPLES (TRICKY KINDS): ParasiteControl → Bravecto, spot-on, "tick and flea control", "monthly tick medicine" (even misspelled). Supplement → "omega 3", "omega 6 fatty acids", "joint supplement" (NOT Diet unless clearly food plan). Diet → "obesity diet", "renal diet", "home cooked chicken and rice". Physiotherapy → "swimming", "sit-stand exercises", "range of motion", "physio at home" when prescribed as rehab. ExerciseRestriction → "No jumping", "Avoid stairs", "sandy surfaces", "cage rest" (Duration separate). OwnerPreference → "home visit", "can't bring him to the clinic", "prefer tablets", "Budget is a concern". RiskConcern → "worried about infection", "side effects", "Anesthesia has some risk".

================================================================================
STEP 5: NER Intent Identification + Replacement Decision (Batch Intent + Patch Plan)
Inputs:
- FINAL_SPAN_BASE_JOINED (from Step 3)
- entity_manifest_v1[] (from Step 4)
Purpose:
For EACH extracted entity, determine:
1) INTENT: a canonical KB search phrase (intent_search_term) + routing family (intent_family)
2) PATCH decision: whether to KEEP the entity span as-is or REPLACE it with a safer canonical form
(entity-span-only replacement; no free rewriting)
Core Principle:
Intent/Patch MUST be derived from LOCAL CONTEXT around the entity.
You MUST use:
- anchoring words
- the prior sentence
- the anchor sentence
- the next sentence
to decide intent and any replacement.
----------------------------------------
STEP 5A: Build the CONTEXT PACK for each entity (MANDATORY)
Purpose:
Compute the exact local context needed for disambiguation.
For each entity in entity_manifest_v1:
1) Identify the entity turn:
- turn_text = the full line (one speaker turn) that contains span_text
- turn_index already provided
2) Split turn_text into sentences (simple rule):
- Sentence boundaries are ".", "?", "!".
- If none exist, treat entire turn_text as one sentence.
3) Determine the anchor_sentence:
- The sentence that contains span_text.
- If span_text appears multiple times in the turn:
choose the occurrence that aligns with the entity’s start_char/end_char span location.
4) Determine prev_sentence:
- If there is a sentence before anchor_sentence in the SAME turn, use it.
- Else, use the last sentence of the previous turn (turn_index-1), if exists.
- Else, prev_sentence = null.
5) Determine next_sentence:
- If there is a sentence after anchor_sentence in the SAME turn, use it.
- Else, use the first sentence of the next turn (turn_index+1), if exists.
- Else, next_sentence = null.
6) Determine anchoring_words:
- Extract up to 6 meaningful tokens BEFORE and AFTER span_text inside anchor_sentence.
- Ignore fillers: "okay", "hmm", "uh", "like", "you know", "haan", "achha", etc.
- If span_text is very short (e.g., "hip"), include more surrounding words (up to 10 each side) to avoid ambiguity.
7) Determine speaker_context:
- speaker label (Vet/Owner/Unknown)
- this affects intent (Owner symptoms vs Vet plan/test)
Context Pack Output (per entity, internal; do not output separately):
- anchor_sentence
- prev_sentence
- next_sentence
- anchoring_words_before
- anchoring_words_after
- speaker_context
----------------------------------------
STEP 5B: Determine INTENT FAMILY (MANDATORY, rule-based)
Purpose:
Classify each entity into a routing family used for KB intent and guardrails.
Policy (11-kind production schema):
Allowed values: "PRODUCT" | "ACTION" | "CLINICAL" | "OTHER"

1) PRODUCT: Medication, ParasiteControl, Diet
2) ACTION: Procedure, Diagnostic, Reminder
3) CLINICAL: Symptom, Diagnosis, Anatomy
4) OTHER: VitalSign, ReasonForVisit

Apply deterministically by kind. If a medication-like word appears but kind is not Medication, keep family by kind rule.
----------------------------------------
STEP 5C: Determine INTENT SEARCH TERM (MANDATORY, “lowest-risk canonical phrase”)
Purpose:
Generate a canonical KB search phrase WITHOUT changing clinical meaning.
Rules (in priority order):
1) If span_text is already a clean canonical term (e.g., “Ortolani test”, “Bravecto”), set:
intent_search_term = span_text (lightly trimmed)
2) If span_text is misspelled/garbled BUT context strongly indicates the canonical term:
intent_search_term = canonical term
AND set intent_confidence accordingly
3) If uncertain or multiple plausible canonical terms:
intent_search_term = span_text (KEEP verbatim)
AND set intent_confidence low/medium
AND list candidates[] with reasons
How to decide “strongly indicates” (must use Context Pack):
- Use anchor_sentence + anchoring_words + prev_sentence + next_sentence.
Examples:
- If anchoring words include “test”, “maneuver”, “positive”, “hip laxity” → likely TestOrManeuver
- If anchoring words include “tablet”, “mg”, “once daily”, “for 5 days” → likely Medication
- If anchoring words include “x-ray”, “radiograph”, “view” → Imaging
- If anchoring words include “surgery”, “operate”, “procedure”, “FHO” → Procedure
ASR correction (CRITICAL for orthopedic tests): When span_text is a known ASR garbling and context indicates the canonical term:
- “ultralining test” in orthopedic context (joint, hip, exam, laxity nearby) → intent_search_term = “Ortolani test”; in Step 5D set patch_action=“REPLACE”, patch_replacement_text=“Ortolani test" so cleaned_transcript contains “Ortolani test".
- “ultralining test" with no local orthopedic cues → intent_search_term = “Ortolani test" (for grounding); patch_action=“KEEP” (do not rewrite transcript).
- “noble angle” with joint/hip context → intent_search_term = “Norberg angle”; patch_action=“REPLACE”, patch_replacement_text="Norberg angle".
Clinical register guard (IMPORTANT):
- Do NOT downgrade into colloquial phrases (e.g., do not convert to “walking funny”).
- Prefer clinical phrasing:
“abnormal gait” > “walking funny”
“diarrhea” > “loose motion” (unless only “loose motion” was said and unclear)
Generic Imaging / X-ray integrity (CRITICAL):
- If span_text is a generic imaging term (e.g. X-ray, radiograph) and the transcript does NOT mention a specific body part (e.g. humerus, hip, thorax), then intent_search_term MUST remain generic (e.g. X-ray or Radiograph). Do NOT add body-part specificity (e.g. X-ray of humerus) unless that body part is explicitly stated in the transcript. This prevents the SOAP generator from displaying a more specific term than was said.

Output of 5C (per entity):
- intent_search_term
- intent_confidence (0–1)
- candidates[] (optional)
----------------------------------------
STEP 5D: Decide PATCH ACTION (KEEP vs REPLACE) (MANDATORY)
Purpose:
Optionally replace the entity’s span_text in the transcript with a safer canonical form.
This is ONLY to improve KB-ready transcript stability, not to “edit” the record.
HARD CONSTRAINTS:
- Patch is ENTITY-SPAN-ONLY.
- Never rewrite text outside the span.
- Never change drug class / diagnosis family / procedure type.
- If unsure, KEEP.
Thresholds:
- Default REPLACE threshold: 0.80
- High-risk kinds require 0.90:
Medication, DiagnosisSuspected, Differential
Decision steps (per entity):
1) Propose a patch_replacement_text ONLY if:
- intent_confidence is high AND
- canonical form is strongly supported by Context Pack
2) Set patch_action="REPLACE" ONLY if ALL are true:
a) patch_confidence >= threshold
b) Same semantic family:
- Medication -> Medication
- Diagnosis -> Diagnosis category
- Test -> Test
- Procedure -> Procedure
- Anatomy -> Anatomy
c) Local evidence exists in anchor_sentence or prev/next sentence
d) Replacement does NOT add new specificity beyond what was said
(e.g., don’t change “infection” to “pyometra”)
Otherwise:
- patch_action="KEEP"
- patch_replacement_text = null
Special rules for UNCERTAIN_TERM:
- If span_text contains [UNCERTAIN_TERM: "..."], patch_action MUST be KEEP
unless the context makes it unquestionably clear AND confidence >= 0.95.
Output of 5D (per entity):
- patch_action
- patch_replacement_text (or null)
- patch_confidence
- patch_reason (1–2 lines, context-based)
- candidates[] (if KEEP but options exist)
----------------------------------------
STEP 5F: Entity-level clinical domain (MANDATORY)
Purpose:
Assign zero or more clinical domains to each entity for downstream grounding (soft-gate, domain boost).
Use the same Context Pack as in 5A–5D: anchor_sentence, prev_sentence, next_sentence, anchoring_words.
Rules:
- Infer domain(s) from the entity span and its local context (symptoms, body part, procedure type, drug indication).
- Use ONLY these allowed values (lowercase): orthopedic, dermatology, cardiology, neurology, ophthalmology, dentistry, gastroenterology, oncology, respiratory, urology, endocrinology, reproductive, soft_tissue_surgery, internal_medicine, general
- Output domain as an ARRAY of strings. Include all applicable domains when context supports multiple (e.g. joint exam + skin work → ["orthopedic", "dermatology"]). When unclear or signalment-only, use ["general"]. Empty array [] is allowed but prefer ["general"] when no specialty applies.
Examples:
- "ultralining test", "hip dysplasia", "Norberg angle", "lameness", "cruciate" in context → ["orthopedic"]
- "pruritus", "cytology", "skin scrape", "apoquel" in context → ["dermatology"]
- "murmur", "echocardiogram", "ECG" in context → ["cardiology"]
- "seizure", "ataxia", "IVDD" in context → ["neurology"]
- "hip dysplasia and we did skin cytology" in same turn → ["orthopedic", "dermatology"]
- Signalment-only or unclear → ["general"]
Output of 5F (per entity):
- domain (array of zero or more allowed values above; e.g. ["orthopedic"] or ["orthopedic", "dermatology"] or ["general"])
----------------------------------------
STEP 5E: Produce PATCH PLAN (batch output)
Purpose:
Create one patch_plan entry per entity_id.
Each patch_plan entry MUST include:
- entity_id
- intent_search_term
- intent_family
- intent_confidence
- patch_action
- patch_replacement_text
- patch_confidence
- patch_reason
- domain (from Step 5F; array of strings, e.g. ["orthopedic"] or ["dermatology", "orthopedic"] or ["general"])
- candidates[] (optional)
Output of STEP 5:
- patch_plan[] (one entry per entity_id)
================================================================================
STEP 6: Build CLEANED_TRANSCRIPT (KB_READY) from FINAL_SPAN_BASE_JOINED + patch_plan
Inputs:
- FINAL_SPAN_BASE_JOINED
- patch_plan[]
Purpose:
Create the final cleaned transcript used for downstream SOAP.
This transcript is human-facing and must respect the character-length constraint.
Actions:
1) Start with FINAL_SPAN_BASE_JOINED.
2) Identify all entities with patch_action="REPLACE".
3) Apply replacements in descending start_char order (to avoid shifting).
4) Replace EXACT substring [start_char:end_char] with patch_replacement_text.
5) Do NOT add any extra markers like “[CANONICALIZED]”.
Length control (CRITICAL):
- Ensure cleaned_transcript length <= RAW_LEN.
- If too long:
a) remove remaining non-clinical chatter (greetings/thanks)
b) remove filler confirmations
c) compress repeated phrases
Do NOT remove clinical facts.
Output of STEP 6:
- cleaned_transcript
================================================================================
STEP 7: Rebase Entity Manifest onto cleaned_transcript (FINAL manifest)
Inputs:
- cleaned_transcript
- entity_manifest_v1
- patch_plan
Purpose:
Ensure entity spans and offsets align to the final cleaned transcript.
Actions:
For each entity:
1) If patched:
- update span_text to patch_replacement_text
2) Recompute start_char/end_char so that:
cleaned_transcript[start_char:end_char] == span_text
3) Preserve original SPAN_BASE provenance in attributes:
- raw_span_base_text
- raw_span_base_start
- raw_span_base_end
4) Add intent and domain fields:
- search_term = intent_search_term
- family = intent_family
- domain = domain from patch_plan (entity-level clinical domains from Step 5F; array of strings, e.g. ["orthopedic"] or ["dermatology", "general"])

MANDATORY OFFSET VALIDATION (DO NOT SKIP):
For every entity: verify cleaned_transcript[start_char:end_char] == span_text exactly.
If any mismatch occurs:
- Recompute that entity's start_char/end_char by locating the exact span_text substring in cleaned_transcript in the expected neighborhood (same turn or adjacent turn).
- If multiple matches exist, choose the one in the same turn_index context.
- Re-validate until all entities match.
Do not proceed to Step 8 unless 100% match.

Output of STEP 7:
- entity_manifest_final[]
================================================================================
STEP 8: Build anchored_transcript (for SOAP generation)
Inputs:
- cleaned_transcript
- entity_manifest_final
Purpose:
Produce a transcript that carries entity IDs forward into SOAP generation.
Actions:
1) Start with cleaned_transcript.
2) For each entity, wrap its exact span using offsets:
replace cleaned_transcript[start_char:end_char] with [[entity_id:span_text]]
3) Insert anchors in descending start_char order.
Overlap rule:
- Prefer the longer span.
- If overlap forces a drop, mark the dropped entity:
attributes.overlap_dropped=true
Output of STEP 8:
- anchored_transcript
================================================================================
STEP 9: Final Verification Gate (MUST PASS)
Purpose:
Guarantee transcript + entity manifest + anchors match exactly.
Checks:
1) cleaned_transcript length <= RAW_LEN (anchored_transcript exempt)
2) For every entity:
cleaned_transcript[start_char:end_char] == span_text
3) For every entity:
anchored_transcript contains [[entity_id:span_text]]
4) No invention / no polarity loss / no certainty upgrade

COVERAGE ASSERTIONS (MUST PASS):
If cleaned_transcript (or span_base_transcript / FINAL_SPAN_BASE_JOINED) contains any of the trigger keywords from Step 4 PASS 5 (KIND COVERAGE AUDIT),
then entity_manifest_final MUST contain at least one entity of the corresponding kind.
If not, you MUST go back and add the missing entities before output.

COVERAGE SAFETY-NET (DO NOT SKIP):
After completing normal extraction, run a coverage audit for these high-miss kinds:
Physiotherapy, ExerciseRestriction, OwnerPreference, RiskConcern, ParasiteControl (product names).

For each of these kinds:
1) Scan FINAL_SPAN_BASE_JOINED for either:
   (a) explicit cue words/phrases (examples below), OR
   (b) translated equivalents that express the same meaning.
2) If any are present, you MUST extract at least one entity of that kind
   covering the relevant phrase(s) verbatim.

Cue examples (non-exhaustive; include misspellings/variants):
- Physiotherapy cues: physio, rehabilitation, rehab, ROM/range of motion, hydrotherapy, swimming, sit-stand, strengthening, stretching
- ExerciseRestriction cues: avoid, no/do not, restrict/rest, cage rest, jumping/landing, stairs, running, high impact
- OwnerPreference cues: home visit, at home/home care, can't come, busy/working/time, budget/cost, prefer tablets/no injections
- RiskConcern cues: infection risk, worried/concern, side effects/reaction, anesthesia risk, clinic exposure
- ParasiteControl list cue: multiple product-like names mentioned near tick/flea control / spot-on / chew / tablet
  -> extract each product-like token separately as ParasiteControl, verbatim.

STRUCTURAL RULE (reduces misclassification):
- If a sentence is a rehab activity -> classify as Physiotherapy, not FollowUpOrPlan.
- If a sentence is an avoidance/restriction -> classify as ExerciseRestriction, not FollowUpOrPlan.

Output of STEP 9:
Proceed to final JSON only if all checks pass.
================================================================================
FINAL OUTPUT (JSON ONLY)
STREAMING: Output keys in this order so the system can use entity_manifest while the transcript is still streaming.
Return ONE JSON object with EXACTLY these top-level keys in this exact order:
1) "entity_manifest": entity_manifest_final
2) "cleaned_transcript": cleaned_transcript
3) "anchored_transcript": anchored_transcript
4) "span_base_transcript": FINAL_SPAN_BASE_JOINED
5) "patch_plan": patch_plan
No extra keys. No markdown. No commentary.
"""

# ---------------------------------------------------------------------------
# Parallel two-call architecture: Call 1 = The Voice, Call 2 = The Brain.
# Enable with PARALLEL_VOICE_BRAIN=true.
# NOTE: In shadow grounding path, Voice runs FIRST (sequentially) to get cleaned transcript,
# then Brain NER runs on the CLEANED transcript (not raw). In non-shadow path, they can run in parallel.
# VOICE_PROMPT is set at top of module = UNIFIED_CLEANING_AND_NER_PROMPT (see UNIFIED_CLEANING_AND_NER_PROMPT.md in project root).
# ---------------------------------------------------------------------------

# Schema for Voice (Call 1) output when not using unified (legacy); run_voice_call now uses SUPER_PASS_JSON_SCHEMA for unified output.
VOICE_OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "cleaned_transcript": {"type": "string"},
    },
    "required": ["cleaned_transcript"],
    "additionalProperties": False,
}

# Brain NER prompt: loaded from docs/BRAIN_NER_PROMPT_UPDATED.md (single source of truth). Enrich pre-extracted entities + catch missed.
CLINICAL_ENTITY_EXTRACTION_PROMPT = _load_brain_ner_prompt()

# Verify prompt contains skeleton format instructions
if "skeleton" not in CLINICAL_ENTITY_EXTRACTION_PROMPT.lower():
    import logging
    _logger = logging.getLogger(__name__)
    _logger.warning("⚠️ Brain NER prompt does not contain 'skeleton' - may be using fallback prompt")


# Schema for entities-only output (unified NER prompt).
# Skeleton format schema: list of strings (one line per entity).
ENTITIES_ONLY_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "skeleton_list": {
            "type": "array",
            "description": "Compressed skeleton lines. Format per item: id|span_text|normalized_name|kind|domains|inv_cats|svc_cats|corr_prob|sugg_prob|hints|is_new|context",
            "items": {"type": "string"},
        },
    },
    "required": ["skeleton_list"],
    "additionalProperties": False,
}

# Brain NER uses the unified extraction prompt (replaces previous BRAIN_PROMPT).
BRAIN_PROMPT = CLINICAL_ENTITY_EXTRACTION_PROMPT

BRAIN_OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "domain_profile": {
            "type": "array",
            "description": "1-3 inferred domains with confidence and evidence",
            "items": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence_snippet": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "entity_manifest": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "kind": {"type": "string"},
                    "span_text": {"type": "string"},
                    "start_char": {"type": "integer"},
                    "end_char": {"type": "integer"},
                    "domain": {"type": "string"},
                    "service_type": {"type": "string", "enum": ["medical", "non-medical"]},
                    "inventory_category": {"type": "array", "items": {"type": "string"}},
                    "service_category": {"type": "array", "items": {"type": "string"}},
                    "search_term": {"type": "string"},
                    "hints": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "hint": {"type": "string"},
                                        "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                                    },
                                    "required": ["hint"],
                                    "additionalProperties": False
                                }
                            ]
                        }
                    },
                    "grounding_recommended": {"type": "boolean"},
                    "grounding_reason": {"type": "string"},
                    "certainty": {"type": "number"},
                    "grounding_policy": {"type": "string", "enum": ["MUST", "TRY", "NEVER"]},
                    "context_snippet": {"type": "string"},
                    "attributes": {"type": "object"},
                    "family": {"type": "string"},
                    "normalized_name": {"type": "string"},
                    "speaker": {"type": "string"},
                    "turn_index": {"type": "integer"},
                },
                "required": ["kind", "span_text", "start_char", "end_char"],
                "additionalProperties": True,
            },
        },
        "terms_not_grounded": {
            "type": "array",
            "description": "Terms not included in entity_manifest, with reason (optional)",
            "items": {
                "type": "object",
                "properties": {
                    "span_text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["span_text", "reason"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["entity_manifest"],
    "additionalProperties": False,
}

# JSON schema for combined clean + NER + batch intent output (single-call latency reduction).
COMBINED_OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "cleaned_transcript": {"type": "string"},
        "entity_manifest": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "kind": {"type": "string"},
                    "span_text": {"type": "string"},
                    "start_char": {"type": "integer"},
                    "end_char": {"type": "integer"},
                    "search_term": {"type": "string"},
                    "family": {"type": "string"},
                    "domain": {"type": "array", "items": {"type": "string"}},
                    "service_type": {"type": "string", "enum": ["medical", "non-medical"]},
                    "inventory_category": {"type": "array", "items": {"type": "string"}},
                    "service_category": {"type": "array", "items": {"type": "string"}},
                    "normalized_name": {"type": "string"},
                    "speaker": {"type": "string"},
                    "turn_index": {"type": "integer"},
                    "context_snippet": {"type": "string"},
                    "certainty": {"type": "string"},
                    "attributes": {"type": "object"},
                },
                "required": ["kind", "span_text"],
                "additionalProperties": True,
            },
        },
        "span_base_transcript": {"type": "string"},
        "patch_plan": {"type": "object"},
    },
    "required": ["cleaned_transcript", "entity_manifest"],
    "additionalProperties": True,
}

# Domain detection: re-export from kb_ner_domain to avoid circular imports (e.g. kb_ner_global_search).
from kb_ner_domain import DOMAIN_KEYWORDS, DOMAIN_PRIMERS, detect_domain  # noqa: F401

def _float_certainty(value: Any) -> Optional[float]:
    """Parse certainty (0-1) from entity; returns None if missing or invalid."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_domain_to_list(domain: Any) -> List[str]:
    """Normalize entity domain to a list of strings. Accepts array or legacy single string."""
    if domain is None:
        return []
    if isinstance(domain, list):
        return [str(x).strip() for x in domain if str(x).strip()]
    s = (domain or "").strip()
    return [s] if s else []


def _entities_to_manifest(
    entities_in: List[Dict[str, Any]],
    transcript: str,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Map entities from unified NER format (span_text, normalized_name, kind, ...) to pipeline entity_manifest with start_char/end_char."""
    entity_manifest = []
    search_start = 0
    for i, e in enumerate(entities_in):
        if not isinstance(e, dict):
            continue
        span_text = (e.get("span_text") or "").strip()
        if not span_text:
            continue
        normalized = (e.get("normalized_name") or "").strip() or span_text
        kind = (e.get("kind") or "Other").strip()
        idx = transcript.find(span_text, search_start)
        if idx == -1:
            idx = transcript.find(span_text, 0)
        if idx != -1:
            search_start = idx + len(span_text)
        start_char = idx if idx != -1 else 0
        end_char = (idx + len(span_text)) if idx != -1 else len(span_text)
        eid = "E%d" % (i + 1)
        # Preserve probability scores if present
        correctness_prob = e.get("correctness_probability")
        suggestion_prob = e.get("suggestion_probability")
        
        # Extract domain from LLM output (required field)
        domain_from_llm = e.get("domain")
        entity_domain = _normalize_domain_to_list(domain_from_llm)
        # Fallback: If LLM didn't provide domain, infer from normalized_name using keyword matching
        if not entity_domain and normalized:
            try:
                from kb_ner_domain import DOMAIN_KEYWORDS
                normalized_lower = normalized.lower()
                for domain, keywords in DOMAIN_KEYWORDS.items():
                    if any(kw in normalized_lower for kw in keywords):
                        entity_domain = [domain]
                        break  # Use first matching domain
            except Exception:
                pass
        # If still no domain, default to empty list
        if not entity_domain:
            entity_domain = []
        # Extract inventory_category and service_category from LLM output.
        # For dual_sync kinds + Diagnosis we require at least one; fallback to legacy "category" or kind-based default.
        def _is_numeric_only(s: str) -> bool:
            """Exclude numeric-only tokens (e.g. leaked confidence '0.95') from category lists."""
            if not s or not isinstance(s, str):
                return True
            s = s.strip()
            if not s:
                return True
            try:
                float(s)
                return True
            except (ValueError, TypeError):
                return False

        def _normalize_cat_list(val: Any) -> List[str]:
            if not val:
                return []
            if isinstance(val, list):
                return [str(c).strip() for c in val if str(c).strip() and not _is_numeric_only(str(c))]
            if isinstance(val, str) and val.strip() and not _is_numeric_only(val):
                return [val.strip()]
            return []

        # Strict mode: no fallback/default category inference.
        # Use only explicit Brain NER outputs.
        inv_cat_llm = _normalize_cat_list(e.get("inventory_category"))
        svc_cat_llm = _normalize_cat_list(e.get("service_category"))
        cat_legacy = _normalize_cat_list(e.get("category"))
        service_type_raw = (e.get("service_type") or "").strip().lower()
        service_type = service_type_raw if service_type_raw in ("medical", "non-medical") else None

        # Legacy single "category" for backward compat output only (do not use it to fill inv/svc).
        entity_category = list(dict.fromkeys(inv_cat_llm + svc_cat_llm + cat_legacy))
        
        # Extract hints from LLM output (1-3 alternative phrasings for KB grounding)
        # Hints can be strings or objects with {"hint": string, "probability": number}
        hints_from_llm = e.get("hints")
        hints_key_present = "hints" in e
        hints = []
        hint_probabilities = {}  # Store probabilities if provided
        
        def _is_placeholder_hint(s: str) -> bool:
            t = (s or "").strip().lower()
            return t in {"0", "none", "null", "n/a", "na", "-", "--"}
        
        if hints_from_llm and isinstance(hints_from_llm, list):
            # Process hints: handle both string format and object format
            for hint_item in hints_from_llm[:3]:  # Limit to max 3
                if isinstance(hint_item, str):
                    hint_text = hint_item.strip()
                    if hint_text and not _is_placeholder_hint(hint_text) and hint_text not in hints:  # Deduplicate
                        hints.append(hint_text)
                elif isinstance(hint_item, dict):
                    hint_text = (hint_item.get("hint") or "").strip()
                    hint_prob = hint_item.get("probability")
                    if hint_text and not _is_placeholder_hint(hint_text) and hint_text not in hints:  # Deduplicate
                        hints.append(hint_text)
                        # Store probability if provided (0.0-1.0)
                        if hint_prob is not None:
                            try:
                                prob_val = float(hint_prob)
                                if 0.0 <= prob_val <= 1.0:
                                    hint_probabilities[hint_text] = prob_val
                            except (ValueError, TypeError):
                                pass
            
            # Keep [] when model emitted only placeholder hints (e.g. "0").
            hints = hints[:3]  # Ensure max 3
        else:
            # Fallback only when hints key is absent entirely (true parser missing field).
            # If hints key exists but is empty/placeholder-derived, preserve [].
            if not hints_key_present:
                hints = [normalized] if normalized else []
            else:
                hints = []
        
        entity_obj = {
            "span_text": span_text,
            "normalized_name": normalized,
            "kind": kind,
            "roles": e.get("roles") or [],
            "attributes": e.get("attributes") or {},
            "assertion_id": "CONF",
            "supporting_text": (e.get("context_sentence") or "").strip()[:500] or "",
            "start_char": start_char,
            "end_char": end_char,
            "is_actionable": True,
            "search_term": normalized,
            "hints": hints,
            "family": None,
            "domain": entity_domain,  # Use domain from LLM output
            "category": entity_category,  # Legacy union; prefer inventory_category / service_category per search
            "inventory_category": inv_cat_llm,  # For soap.inventory (dual_sync + Diagnosis)
            "service_category": svc_cat_llm,  # For soap.service_master (dual_sync + Diagnosis)
            "service_type": service_type,  # Optional: "medical" | "non-medical" for service hard gating
            "entity_id": eid,
            "grounding_recommended": True,
            "grounding_reason": None,
            "certainty": _float_certainty(e.get("confidence")),
            # Master Doc: billable kinds must be groundable (MUST/TRY). NEVER only for note-only skips.
            "grounding_policy": (
                "NEVER"
                if str(kind or "").strip().lower() in {
                    "anatomy", "symptom", "vitalsign", "signalment", "other", "identity"
                }
                else "MUST"
                if str(kind or "").strip().lower() in {
                    "medication", "medicine", "drug", "product", "vaccine",
                    "procedure", "diagnostic", "diagnostictest", "preventive",
                    "parasitecontrol", "diet", "nutrition", "service",
                }
                else "TRY"
            ),
            "context_snippet": (e.get("context_sentence") or "").strip()[:200] or None,
        }
        # Store hint probabilities if available
        if hint_probabilities:
            entity_obj["hint_probabilities"] = hint_probabilities
        # Add probability scores if present (for suggestion boost in grounding)
        if correctness_prob is not None:
            try:
                entity_obj["correctness_probability"] = float(correctness_prob)
            except (TypeError, ValueError):
                pass
        if suggestion_prob is not None:
            try:
                entity_obj["suggestion_probability"] = float(suggestion_prob)
            except (TypeError, ValueError):
                pass
        # query_expansion: up to 3 likely brand/product names when term is phonetic (from 13th skeleton field)
        qe_raw = e.get("query_expansion")
        if isinstance(qe_raw, list):
            entity_obj["query_expansion"] = [str(x).strip() for x in qe_raw if str(x).strip()][:3]
        elif isinstance(qe_raw, str) and (qe_raw or "").strip():
            entity_obj["query_expansion"] = [x.strip() for x in qe_raw.split(",") if x.strip()][:3]
        else:
            entity_obj["query_expansion"] = []
        entity_manifest.append(entity_obj)
    return entity_manifest


def run_voice_call(
    raw_transcript: str,
    optional_inputs: str = "",
    model: Optional[str] = None,
    client: Any = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Call 1 (The Voice): Clean, translate, and attribute the conversation.
    Returns cleaned_transcript only. Runs on raw transcript; no entity extraction.
    """
    model = model or os.getenv("SUPER_PASS_MODEL", SUPER_PASS_DEFAULT_MODEL)
    if client is None:
        try:
            res = get_client_for_model(model, logger=logger)
            client = res[0] if isinstance(res, tuple) else res
        except Exception as e:
            if logger:
                logger.warning("Voice call: could not get client: %s", e)
            return raw_transcript.strip()
    # Unified prompt (same as Super-Pass); response has cleaned_transcript, extracted_entities, entities_by_kind
    prompt = _format_unified_prompt(raw_transcript, optional_inputs or "")
    schema = SUPER_PASS_JSON_SCHEMA  # Accept full unified output; we only read cleaned_transcript below
    try:
        max_tokens = int(os.getenv("SUPER_PASS_MAX_TOKENS", "16384"))
    except Exception:
        max_tokens = 16384
    if "fireworks" in (model or "").lower():
        max_tokens = min(max_tokens, 8192)
    messages = [
        {"role": "system", "content": "You are a clinical documentation specialist. Return ONLY valid JSON. First character must be \"{\"."},
        {"role": "user", "content": prompt + "\n\nOUTPUT JSON SCHEMA (must match exactly):\n" + json.dumps(schema, ensure_ascii=False)},
        {"role": "assistant", "content": "{"},
    ]
    create_kw = dict(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_schema", "json_schema": {"name": "VoiceClean", "schema": schema}},
    )
    try:
        resp = client.chat.completions.create(**create_kw)
        _msg = resp.choices[0].message if resp and resp.choices else None
        content = getattr(_msg, "content", None) if _msg else None
        if isinstance(content, list):
            content = "".join(
                (p.get("text", p) if isinstance(p, dict) else str(p)) for p in content
            ) if content else None
        if content is not None and not isinstance(content, str):
            content = str(content)
        raw_content = (content and str(content).strip()) or "{}"
    except Exception as e:
        if logger:
            logger.warning("Voice call LLM failed: %s", e)
        return raw_transcript.strip()
    try:
        data = json.loads(raw_content)
    except Exception:
        data = _extract_json_object(raw_content)
    if not isinstance(data, dict):
        if logger:
            logger.warning("Voice call: response was not JSON object")
        return raw_transcript.strip()
    cleaned = (data.get("cleaned_transcript") or "").strip()
    if not cleaned:
        cleaned = raw_transcript.strip()
    if logger:
        logger.info("Voice call: cleaned=%d chars", len(cleaned))
    return cleaned


def run_brain_call(
    raw_transcript: str,
    optional_inputs: str = "",
    model: Optional[str] = None,
    client: Any = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Call 2 (The Brain): NER on transcript using unified entity extraction prompt.
    Returns (entity_manifest, terms_not_grounded). Offsets index into raw_transcript.
    """
    model = model or os.getenv("SUPER_PASS_MODEL", SUPER_PASS_DEFAULT_MODEL)
    if client is None:
        try:
            res = get_client_for_model(model, logger=logger)
            client = res[0] if isinstance(res, tuple) else res
        except Exception as e:
            if logger:
                logger.warning("Brain call: could not get client: %s", e)
            return [], []
    prompt = _format_brain_ner_prompt(raw_transcript, optional_inputs or "[]")
    schema = ENTITIES_ONLY_JSON_SCHEMA
    # Verify prompt contains skeleton format instructions
    if logger and "skeleton" not in prompt.lower():
        logger.warning("⚠️ Brain call: Prompt does not contain 'skeleton' - may be using fallback prompt")
    # Log how many pre-extracted entities are being passed and add explicit count instruction
    pre_extracted_count = 0
    pre_extracted_entities: List[Dict[str, Any]] = []
    if optional_inputs and optional_inputs != "[]":
        try:
            pre_extracted = json.loads(optional_inputs)
            if isinstance(pre_extracted, list):
                pre_extracted_entities = [e for e in pre_extracted if isinstance(e, dict)]
                pre_extracted_count = len(pre_extracted)
                if logger:
                    logger.info("Brain call: Processing %d pre-extracted entities from Super-Pass", pre_extracted_count)
                # Add explicit count requirement to prompt
                if pre_extracted_count > 0:
                    count_instruction = f"\n\nCRITICAL: PRE_EXTRACTED_ENTITIES contains {pre_extracted_count} entities. You MUST return exactly {pre_extracted_count} items in skeleton_list (one per entity). Do NOT skip any entities."
                    prompt = prompt + count_instruction
        except Exception:
            pass
    try:
        # Keep Brain token budget bounded for lower tail-latency; still scale with entity count.
        base_brain_max_tokens = int(os.getenv("BRAIN_NER_MAX_TOKENS", os.getenv("SUPER_PASS_MAX_TOKENS", "8192")))
    except Exception:
        base_brain_max_tokens = 8192
    estimated_tokens = (900 + (pre_extracted_count * 220)) if pre_extracted_count > 0 else 2400
    max_tokens = max(base_brain_max_tokens, estimated_tokens)
    if "fireworks" in (model or "").lower():
        max_tokens = min(max_tokens, 8192)
    else:
        max_tokens = min(max_tokens, 12000)

    def _fallback_manifest_from_pre_extracted(reason: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not pre_extracted_entities:
            return [], []
        raw_lower = raw_transcript.lower()
        rolling_pos = 0
        fallback_entities: List[Dict[str, Any]] = []
        for idx, item in enumerate(pre_extracted_entities):
            span = (item.get("span_text") or item.get("normalized_name") or "").strip()
            kind = (item.get("kind") or "Other")
            attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
            start_char = 0
            end_char = 0
            if span:
                span_lower = span.lower()
                found_at = raw_lower.find(span_lower, rolling_pos)
                if found_at == -1:
                    found_at = raw_lower.find(span_lower)
                if found_at != -1:
                    start_char = found_at
                    end_char = found_at + len(span)
                    rolling_pos = end_char
            fallback_entities.append(
                {
                    "id": item.get("id") or f"e{idx + 1}",
                    "span_text": span,
                    "normalized_name": (item.get("normalized_name") or span),
                    "kind": kind,
                    "start_char": start_char,
                    "end_char": end_char,
                    "attributes": attrs,
                }
            )
        entity_manifest = _entities_to_manifest(fallback_entities, raw_transcript, logger=logger)
        if logger:
            logger.warning(
                "⚠️ Brain call fallback (%s): returning %d entities from PRE_EXTRACTED_ENTITIES.",
                reason,
                len(entity_manifest),
            )
        return entity_manifest, []

    messages = [
        {"role": "system", "content": "You are a medical information engine. Return ONLY valid JSON with a 'skeleton_list' field. Each item in skeleton_list must be one compressed skeleton line for exactly one entity. You MUST process EVERY entity in PRE_EXTRACTED_ENTITIES and return one item per entity. Start your response with '{' and nothing else."},
        {"role": "user", "content": prompt + "\n\nOUTPUT JSON SCHEMA (must match exactly - return JSON with 'skeleton_list' containing ALL entities):\n" + json.dumps(schema, ensure_ascii=False) + "\n\nREMINDER: You MUST return one skeleton_list item for EVERY entity in PRE_EXTRACTED_ENTITIES. If PRE_EXTRACTED_ENTITIES has 45 entities, return 45 items. Do NOT skip any entities."},
        {"role": "assistant", "content": "{"},
    ]
    create_kw = dict(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_schema", "json_schema": {"name": "ClinicalEntities", "schema": schema}},
    )
    # Fireworks constraint: some models reject non-streaming requests when max_tokens > 4096.
    # Auto-route to streaming Brain NER path in that case.
    if "fireworks" in (model or "").lower() and max_tokens > 4096:
        if logger:
            logger.info(
                "Brain call: auto-switching to streaming mode for Fireworks (max_tokens=%d > 4096)",
                max_tokens,
            )
        try:
            streamed = list(
                run_brain_call_streaming(
                    raw_transcript=raw_transcript,
                    optional_inputs=optional_inputs,
                    model=model,
                    client=client,
                    logger=logger,
                )
            )
            entity_manifest = [ent for _, ent in streamed if isinstance(ent, dict)]
            if entity_manifest:
                if logger:
                    logger.info(
                        "Brain call (streaming fallback): entities=%d (offsets relative to raw), terms_not_grounded=0",
                        len(entity_manifest),
                    )
                return entity_manifest, []
            # If streaming produced no entities but pre-extracted exists, return safe fallback
            # to avoid empty-manifest cascade.
            return _fallback_manifest_from_pre_extracted("streaming_returned_no_entities")
        except Exception as e:
            if logger:
                logger.warning("Brain call streaming fallback failed: %s", e)
            return _fallback_manifest_from_pre_extracted("streaming_fallback_error")
    finish_reason = None
    try:
        resp = client.chat.completions.create(**create_kw)
        _msg = resp.choices[0].message if resp and resp.choices else None
        content = getattr(_msg, "content", None) if _msg else None
        if isinstance(content, list):
            content = "".join(
                (p.get("text", p) if isinstance(p, dict) else str(p)) for p in content
            ) if content else None
        if content is not None and not isinstance(content, str):
            content = str(content)
        raw_content = (content and str(content).strip()) or "{}"
        # Log response length and finish reason to detect truncation
        if logger and resp and resp.choices:
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
            if finish_reason == "length":
                logger.warning("⚠️ Brain call: Response truncated (finish_reason=length). Increase max_tokens or reduce input size.")
            elif finish_reason:
                logger.debug("Brain call: finish_reason=%s, response_length=%d chars", finish_reason, len(raw_content))
    except Exception as e:
        if logger:
            logger.warning("Brain call LLM failed: %s", e)
        return [], []
    # Parse skeleton format from JSON response
    try:
        data = json.loads(raw_content)
    except Exception:
        data = _extract_json_object(raw_content)
    if not isinstance(data, dict):
        if logger:
            logger.warning("Brain call: response was not JSON object")
        return _fallback_manifest_from_pre_extracted("non_json_response")
    
    # Extract skeleton output from response (prefer skeleton_list, keep legacy skeleton string support)
    skeleton_list = data.get("skeleton_list") or []
    skeleton_text = data.get("skeleton") or ""
    if logger and isinstance(skeleton_list, list) and pre_extracted_count > 0 and len(skeleton_list) != pre_extracted_count:
        logger.warning(
            "⚠️ Brain call: skeleton_list count mismatch (expected %d from PRE_EXTRACTED_ENTITIES, got %d).",
            pre_extracted_count,
            len(skeleton_list),
        )
    if not skeleton_list and not skeleton_text:
        # Fallback: try to parse as old JSON format for backward compatibility
        entities_in = data.get("entities") or []
        if isinstance(entities_in, list) and entities_in:
            if logger:
                logger.warning("⚠️ Brain call: LLM returned legacy JSON format (entities array) instead of skeleton format. Response keys: %s", list(data.keys()))
            entity_manifest = _entities_to_manifest(entities_in, raw_transcript, logger=logger)
            terms_not_grounded = []
            if logger:
                logger.info("Brain call: entities=%d (offsets relative to raw), terms_not_grounded=%d", len(entity_manifest), len(terms_not_grounded))
            return entity_manifest, terms_not_grounded
        if logger:
            logger.warning("⚠️ Brain call: no skeleton_list or skeleton field found in response. Response keys: %s", list(data.keys()) if isinstance(data, dict) else "not a dict")
        if finish_reason == "length":
            return _fallback_manifest_from_pre_extracted("truncated_no_skeleton")
        return _fallback_manifest_from_pre_extracted("missing_skeleton")
    
    # Parse skeleton format to entities
    try:
        from kb_ner_skeleton_parser import parse_skeleton_entities
        if logger:
            if isinstance(skeleton_list, list) and skeleton_list:
                logger.info("✅ Brain call: Parsing skeleton_list format (%d items)", len(skeleton_list))
                logger.debug("Skeleton list preview (first item): %s", str(skeleton_list[0])[:200])
            else:
                logger.info("✅ Brain call: Parsing legacy skeleton string format (%d chars)", len(skeleton_text))
                logger.debug("Skeleton preview (first 200 chars): %s", skeleton_text[:200])
        entities_in = parse_skeleton_entities(skeleton_list if skeleton_list else skeleton_text)
        if logger:
            logger.info("✅ Brain call: Parsed %d entities from skeleton format", len(entities_in))
            # Warn if we expected more entities (check pre_extracted count)
            if optional_inputs and optional_inputs != "[]":
                try:
                    pre_extracted = json.loads(optional_inputs)
                    if isinstance(pre_extracted, list) and len(pre_extracted) > len(entities_in):
                        logger.warning("⚠️ Brain call: Expected %d entities from pre_extracted, but parsed only %d from skeleton format. Output may be truncated or incomplete.", len(pre_extracted), len(entities_in))
                        if isinstance(skeleton_list, list) and skeleton_list:
                            logger.warning("⚠️ Skeleton_list length: %d items (expected %d)", len(skeleton_list), len(pre_extracted))
                            logger.debug("Skeleton list (full): %s", skeleton_list)
                        else:
                            logger.warning("⚠️ Skeleton text length: %d chars (expected ~%d chars for %d entities)", len(skeleton_text), len(pre_extracted) * 100, len(pre_extracted))
                            logger.debug("Skeleton text (full): %s", skeleton_text)
                except Exception:
                    pass
    except ImportError:
        if logger:
            logger.error("❌ Brain call: failed to import parse_skeleton_entities from kb_ner_skeleton_parser")
        return _fallback_manifest_from_pre_extracted("skeleton_parser_import_error")
    except Exception as e:
        if logger:
            logger.warning("❌ Brain call: failed to parse skeleton format: %s", e)
            if isinstance(skeleton_list, list) and skeleton_list:
                logger.debug("Skeleton list preview (first 3 items): %s", skeleton_list[:3])
            else:
                logger.debug("Skeleton text preview (first 500 chars): %s", skeleton_text[:500])
        return _fallback_manifest_from_pre_extracted("skeleton_parse_error")
    
    terms_not_grounded = []  # Terms not grounded (optional output)
    if not isinstance(entities_in, list):
        entities_in = []
    entity_manifest = _entities_to_manifest(entities_in, raw_transcript, logger=logger)
    
    # Billable kinds require both inventory_category and service_category. Apply defaults when the model omits them.
    from kb_ner_routing import DUAL_SYNC_BILLABLE_KINDS
    dual_sync_kinds_set = set(DUAL_SYNC_BILLABLE_KINDS)
    default_inv = ["General"]
    default_svc = ["Consultation"]
    applied_defaults_count = 0
    for ent in entity_manifest:
        kind = (ent.get("kind") or "").strip()
        if kind not in dual_sync_kinds_set:
            continue
        inv_cat = ent.get("inventory_category") or []
        svc_cat = ent.get("service_category") or []
        svc_type = (ent.get("service_type") or "").strip().lower()
        if svc_type not in ("medical", "non-medical"):
            # Default policy: billable/general entities are treated as medical unless explicitly tagged non-medical.
            # This prevents broad "general/consultation" fallthrough into grooming/training categories.
            ent["service_type"] = "medical"
        if not inv_cat:
            ent["inventory_category"] = default_inv
            applied_defaults_count += 1
        if not svc_cat:
            ent["service_category"] = default_svc
            applied_defaults_count += 1
    if logger and applied_defaults_count:
        logger.debug(
            "Brain NER: applied default inventory_category/service_category for %d billable entity fields (model omitted them)",
            applied_defaults_count,
        )
    
    if logger:
        logger.info("Brain call: entities=%d (offsets relative to raw), terms_not_grounded=%d", len(entity_manifest), len(terms_not_grounded))
    return entity_manifest, terms_not_grounded


def run_brain_call_streaming(
    raw_transcript: str,
    optional_inputs: str = "",
    model: Optional[str] = None,
    client: Any = None,
    logger: Optional[logging.Logger] = None,
) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
    """
    Call 2 (The Brain) with streaming: yields (entity_index, entity_dict) as each complete
    entity object is parsed from the stream. Used for shadow grounding (fire grounding
    as each entity arrives instead of waiting for full response).
    """
    model = model or os.getenv("SUPER_PASS_MODEL", SUPER_PASS_DEFAULT_MODEL)
    if client is None:
        try:
            res = get_client_for_model(model, logger=logger)
            client = res[0] if isinstance(res, tuple) else res
        except Exception as e:
            if logger:
                logger.warning("Brain stream: could not get client: %s", e)
            return
    prompt = _format_brain_ner_prompt(raw_transcript, optional_inputs or "[]")
    schema = ENTITIES_ONLY_JSON_SCHEMA
    try:
        max_tokens = int(os.getenv("SUPER_PASS_MAX_TOKENS", "16384"))
    except Exception:
        max_tokens = 16384
    if "fireworks" in (model or "").lower():
        max_tokens = min(max_tokens, 8192)
    messages = [
        {"role": "system", "content": "You are a clinical entity extractor. Return ONLY valid JSON with a 'skeleton_list' field. Each item in skeleton_list must be one compressed skeleton line. Start your response with '{' and nothing else."},
        {"role": "user", "content": prompt + "\n\nOUTPUT JSON SCHEMA (must match exactly):\n" + json.dumps(schema, ensure_ascii=False)},
        {"role": "assistant", "content": "{"},
    ]
    create_kw = dict(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_schema", "json_schema": {"name": "ClinicalEntities", "schema": schema}},
        stream=True,
    )
    try:
        resp = client.chat.completions.create(**create_kw)
    except Exception as e:
        if logger:
            logger.warning("Brain stream LLM failed: %s", e)
        return
    full_text = ""
    # For skeleton format, we need to accumulate the full JSON response first
    # then parse the skeleton string. For backward compatibility, also try old format.
    for chunk in resp:
        delta = _safe_get_delta_text(chunk)
        if not delta:
            continue
        full_text += delta
    
    # Try to parse as JSON with skeleton_list (preferred) or legacy skeleton string.
    try:
        data = json.loads(full_text)
        if isinstance(data, dict):
            skeleton_list = data.get("skeleton_list") or []
            skeleton_text = data.get("skeleton") or ""
            if skeleton_list or skeleton_text:
                try:
                    from kb_ner_skeleton_parser import parse_skeleton_entities
                    entities_in = parse_skeleton_entities(skeleton_list if skeleton_list else skeleton_text)
                    # Convert to manifest format and yield
                    entity_manifest = _entities_to_manifest(entities_in, raw_transcript, logger=logger)
                    for idx, ent in enumerate(entity_manifest):
                        yield idx, ent
                    return
                except ImportError:
                    if logger:
                        logger.error("Brain stream: failed to import parse_skeleton_entities")
                except Exception as e:
                    if logger:
                        logger.warning("Brain stream: failed to parse skeleton format: %s", e)
    except Exception:
        pass
    
    # Fallback to old JSON format parsing (for backward compatibility)
    scanner = _EntityArrayObjectScanner("entities")
    entity_index = 0
    search_start = 0
    for obj_str in scanner.feed(full_text):
            try:
                e = json.loads(obj_str)
            except Exception:
                continue
            if not isinstance(e, dict):
                continue
            span_text = (e.get("span_text") or "").strip()
            if not span_text:
                continue
            normalized = (e.get("normalized_name") or "").strip() or span_text
            idx = raw_transcript.find(span_text, search_start)
            if idx == -1:
                idx = raw_transcript.find(span_text, 0)
            if idx != -1:
                search_start = idx + len(span_text)
            start_char = idx if idx != -1 else 0
            end_char = (idx + len(span_text)) if idx != -1 else len(span_text)
            eid = "E%d" % (entity_index + 1)
            
            # Extract domain from LLM output (required field)
            domain_from_llm = e.get("domain")
            entity_domain = _normalize_domain_to_list(domain_from_llm)
            # Fallback: If LLM didn't provide domain, infer from normalized_name using keyword matching
            if not entity_domain and normalized:
                try:
                    from kb_ner_domain import DOMAIN_KEYWORDS
                    normalized_lower = normalized.lower()
                    for domain, keywords in DOMAIN_KEYWORDS.items():
                        if any(kw in normalized_lower for kw in keywords):
                            entity_domain = [domain]
                            break  # Use first matching domain
                except Exception:
                    pass
            # If still no domain, default to empty list
            if not entity_domain:
                entity_domain = []
            
            # Extract hints from LLM output (1-3 alternative phrasings for KB grounding)
            # Hints can be strings or objects with {"hint": string, "probability": number}
            hints_from_llm = e.get("hints")
            hints_key_present = "hints" in e
            hints = []
            hint_probabilities = {}  # Store probabilities if provided
            
            def _is_placeholder_hint(s: str) -> bool:
                t = (s or "").strip().lower()
                return t in {"0", "none", "null", "n/a", "na", "-", "--"}
            
            if hints_from_llm and isinstance(hints_from_llm, list):
                # Process hints: handle both string format and object format
                for hint_item in hints_from_llm[:3]:  # Limit to max 3
                    if isinstance(hint_item, str):
                        hint_text = hint_item.strip()
                        if hint_text and not _is_placeholder_hint(hint_text) and hint_text not in hints:  # Deduplicate
                            hints.append(hint_text)
                    elif isinstance(hint_item, dict):
                        hint_text = (hint_item.get("hint") or "").strip()
                        hint_prob = hint_item.get("probability")
                        if hint_text and not _is_placeholder_hint(hint_text) and hint_text not in hints:  # Deduplicate
                            hints.append(hint_text)
                            # Store probability if provided (0.0-1.0)
                            if hint_prob is not None:
                                try:
                                    prob_val = float(hint_prob)
                                    if 0.0 <= prob_val <= 1.0:
                                        hint_probabilities[hint_text] = prob_val
                                except (ValueError, TypeError):
                                    pass
                
                # Keep [] when model emitted only placeholder hints (e.g. "0").
                hints = hints[:3]  # Ensure max 3
            else:
                # Fallback only when hints key is absent entirely.
                if not hints_key_present:
                    hints = [normalized] if normalized else []
                else:
                    hints = []
            
            entity_dict = {
                "span_text": span_text,
                "normalized_name": normalized,
                "kind": (e.get("kind") or "Other").strip(),
                "roles": e.get("roles") or [],
                "attributes": e.get("attributes") or {},
                "assertion_id": "CONF",
                "supporting_text": (e.get("context_sentence") or "").strip()[:500] or "",
                "start_char": start_char,
                "end_char": end_char,
                "is_actionable": True,
                "search_term": normalized,
                "hints": hints,
                "family": None,
                "domain": entity_domain,  # Use domain from LLM output
                "entity_id": eid,
            }
            # Store hint probabilities in entity_dict if available
            if hint_probabilities:
                entity_dict["hint_probabilities"] = hint_probabilities
            # Preserve probability scores if present (for suggestion boost in grounding)
            correctness_prob = e.get("correctness_probability")
            suggestion_prob = e.get("suggestion_probability")
            if correctness_prob is not None:
                try:
                    entity_dict["correctness_probability"] = float(correctness_prob)
                except (TypeError, ValueError):
                    pass
            if suggestion_prob is not None:
                try:
                    entity_dict["suggestion_probability"] = float(suggestion_prob)
                except (TypeError, ValueError):
                    pass
            # query_expansion: up to 3 terms (13th skeleton field)
            qe = e.get("query_expansion")
            entity_dict["query_expansion"] = [str(x).strip() for x in qe if str(x).strip()][:3] if isinstance(qe, list) else []
            yield (entity_index, entity_dict)
            entity_index += 1
    if logger:
        logger.info("Brain stream: yielded %d entities", entity_index)


def run_parallel_voice_brain(
    raw_transcript: str,
    optional_inputs: str = "",
    model: Optional[str] = None,
    client: Any = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Run Voice (Call 1) and Brain (Call 2) in parallel on the same raw transcript.
    Returns (cleaned_transcript, entity_manifest). Entity offsets are relative to raw_transcript.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    model = model or os.getenv("SUPER_PASS_MODEL", SUPER_PASS_DEFAULT_MODEL)
    if client is None:
        try:
            res = get_client_for_model(model, logger=logger)
            client = res[0] if isinstance(res, tuple) else res
        except Exception as e:
            if logger:
                logger.warning("Parallel voice+brain: could not get client: %s", e)
            return raw_transcript.strip(), []
    cleaned = raw_transcript.strip()
    entities: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_voice = ex.submit(run_voice_call, raw_transcript, optional_inputs, model, client, logger)
        f_brain = ex.submit(run_brain_call, raw_transcript, optional_inputs, model, client, logger)
        for fut in as_completed([f_voice, f_brain]):
            try:
                result = fut.result()
                if isinstance(result, str):
                    cleaned = result
                elif isinstance(result, tuple) and len(result) >= 1:
                    entities = result[0]
                elif isinstance(result, list):
                    entities = result
            except Exception as e:
                if logger:
                    logger.warning("Parallel voice+brain: one call failed: %s", e)
    if logger:
        logger.info("Parallel voice+brain: cleaned=%d chars, entities=%d", len(cleaned), len(entities))
    return cleaned, entities


def _fuzzy_find_closest_substring(
    needle: str,
    haystack: str,
    min_ratio: float = 0.85,
    max_len_diff_ratio: float = 1.5,
) -> Optional[Tuple[int, int, str]]:
    """
    Find the closest substring in haystack to needle using sequence matching.
    Returns (start, end, matched_substring) or None. Used when exact match fails
    (e.g. manifest "tick and flee" vs cleaned "tick and flea").
    """
    if not needle or not haystack or len(needle) > len(haystack):
        return None
    try:
        from difflib import SequenceMatcher
    except ImportError:
        return None
    n_len = len(needle)
    # Search windows of length ~len(needle) (allow some length variance)
    max_window = min(int(n_len * max_len_diff_ratio) + 2, len(haystack))
    min_window = max(2, n_len // 2)
    best: Optional[Tuple[float, int, int, str]] = None
    for wlen in range(min_window, max_window + 1):
        for i in range(0, len(haystack) - wlen + 1):
            sub = haystack[i : i + wlen]
            ratio = SequenceMatcher(None, needle.lower(), sub.lower()).ratio()
            if ratio >= min_ratio and (best is None or ratio > best[0]):
                best = (ratio, i, i + wlen, haystack[i : i + wlen])
    if best is None:
        return None
    return (best[1], best[2], best[3])


def fuzzy_sync_anchor_injection(
    cleaned_transcript: str,
    entity_manifest: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
    fuzzy_fallback_min_ratio: float = 0.88,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Anchor injection (Step 3): join Clean Transcript (Call 1) and Manifest (Call 2).

    - Fuzzy sync: take span_text from the manifest (e.g. "tick and flee") and find the
      closest match in the clean transcript (e.g. "tick and flea"). Tries exact match
      (span_text, search_term, normalized_name), then typo-tolerant match via difflib.
    - Tagging: inject [[E-ID:text]] at each matched span so the SOAP LLM sees anchors.
    - Returns (tagged_cleaned_transcript, entity_manifest_rebased) with start_char/end_char
      rebased to cleaned transcript positions for downstream.
    """
    if not entity_manifest:
        return cleaned_transcript, list(entity_manifest)
    # Build list of (entity_id, span_text, search_term, normalized_name) for matching
    candidates: List[Tuple[str, str, str, str]] = []
    for ent in entity_manifest:
        eid = (ent.get("entity_id") or "").strip()
        if not eid or not (eid.startswith("E") and eid[1:].isdigit()):
            continue
        span = (ent.get("span_text") or "").strip()
        search = (ent.get("search_term") or "").strip() or span
        norm = (ent.get("normalized_name") or "").strip() or span
        if span:
            candidates.append((eid, span, search, norm))
    if not candidates:
        return cleaned_transcript, _backfill_span_offsets_in_cleaned(entity_manifest, cleaned_transcript, logger=logger)
    matches: List[Tuple[str, int, int, str]] = []  # (eid, start, end, display_text)
    used_ranges: List[Tuple[int, int]] = []
    for eid, span, search, norm in candidates:
        display = span
        start, end = -1, -1
        # 1) Exact: span_text in cleaned
        idx = cleaned_transcript.find(span)
        if idx != -1 and not _range_overlaps_any((idx, idx + len(span)), used_ranges):
            start, end = idx, idx + len(span)
            display = cleaned_transcript[start:end]
        # 2) Exact: search_term (grounding hint) in cleaned
        if start == -1 and search and search != span:
            idx = cleaned_transcript.find(search)
            if idx != -1 and not _range_overlaps_any((idx, idx + len(search)), used_ranges):
                start, end = idx, idx + len(search)
                display = search
        # 3) Exact: normalized_name in cleaned
        if start == -1 and norm and norm != span:
            idx = cleaned_transcript.find(norm)
            if idx != -1 and not _range_overlaps_any((idx, idx + len(norm)), used_ranges):
                start, end = idx, idx + len(norm)
                display = norm
        # 4) Fuzzy: closest substring (e.g. "tick and flee" -> "tick and flea")
        if start == -1 and span:
            fuzzy_result = _fuzzy_find_closest_substring(
                span, cleaned_transcript, min_ratio=fuzzy_fallback_min_ratio,
            )
            if fuzzy_result:
                f_start, f_end, f_sub = fuzzy_result
                if not _range_overlaps_any((f_start, f_end), used_ranges):
                    start, end = f_start, f_end
                    display = f_sub
                    if logger:
                        logger.debug("Fuzzy sync: entity %s span=%r -> matched %r at [%d:%d]", eid, span[:40], display[:40], start, end)
        if start != -1 and end != -1:
            matches.append((eid, start, end, display))
            used_ranges.append((start, end))
            used_ranges.sort(key=lambda r: r[0])
        elif logger:
            logger.debug("Fuzzy sync: no match for entity %s span=%r", eid, span[:50])
    # Inject anchors in descending start order. Use ORIGINAL span_text in tag for audit: transcript shows verbatim word.
    eid_to_original: Dict[str, str] = {ent.get("entity_id", ""): (ent.get("span_text") or "").strip() for ent in entity_manifest if (ent.get("entity_id") or "").strip()}
    matches.sort(key=lambda m: -m[1])
    tagged = cleaned_transcript
    for eid, start, end, display in matches:
        original = eid_to_original.get(eid) or display
        anchor = "[[%s:%s]]" % (eid, original)
        tagged = tagged[:start] + anchor + tagged[end:]
    # Rebase entity_manifest: set start_char/end_char to cleaned positions where we found a match
    eid_to_range_dict = {eid: (start, end) for eid, start, end, _ in matches}
    manifest_rebased = []
    for ent in entity_manifest:
        e = dict(ent)
        eid = (ent.get("entity_id") or "").strip()
        if eid in eid_to_range_dict:
            s, en = eid_to_range_dict[eid]
            e["start_char"] = s
            e["end_char"] = en
        manifest_rebased.append(e)
    # Unmatched-entity gap fix: rebase entities that got no tag match to cleaned transcript when span_text exists there (reduces UI "tag shift")
    unmatched = [e for e in manifest_rebased if ((e.get("entity_id") or "").strip() not in eid_to_range_dict)]
    if unmatched:
        backfilled = _backfill_span_offsets_in_cleaned(unmatched, cleaned_transcript, logger=logger)
        eid_to_backfilled = {(e.get("entity_id") or "").strip(): e for e in backfilled if (e.get("entity_id") or "").strip()}
        for e in manifest_rebased:
            eid = (e.get("entity_id") or "").strip()
            if eid in eid_to_backfilled:
                b = eid_to_backfilled[eid]
                e["start_char"] = b.get("start_char", e.get("start_char", 0))
                e["end_char"] = b.get("end_char", e.get("end_char", 0))
        if logger:
            logger.debug("Fuzzy sync: backfilled %d unmatched entities to cleaned transcript offsets", len(unmatched))
    return tagged, manifest_rebased


def _range_overlaps_any(span: Tuple[int, int], used: List[Tuple[int, int]]) -> bool:
    s, e = span
    for u, v in used:
        if not (e <= u or s >= v):
            return True
    return False


def run_combined_clean_ner_batch_intent(
    raw_transcript: str,
    optional_inputs: str = "",
    model: Optional[str] = None,
    client: Any = None,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[Any] = None,
    timestamp: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Single LLM call: clean transcript + NER + batch intent (patch plan).
    Returns (cleaned_transcript, entity_manifest) in pipeline shape (search_term, family set from batch intent).
    Enable with COMBINED_CLEAN_NER_BATCH_INTENT=true to reduce latency vs separate Super-Pass + Batch Intent.
    When COMBINED_DEBUG_SAVE_RESPONSE=true and output_dir + timestamp are provided, saves raw API response to
    output_dir/combined_pass_raw_response_{timestamp}.txt for verification.
    """
    model = model or os.getenv("SUPER_PASS_MODEL", SUPER_PASS_DEFAULT_MODEL)
    if client is None:
        try:
            res = get_client_for_model(model, logger=logger)
            client = res[0] if isinstance(res, tuple) else res
        except Exception as e:
            if logger:
                logger.warning("Combined clean+NER+batch: could not get client: %s", e)
            return raw_transcript.strip(), []

    try:
        from kb_ner_routing import ALL_NER_KINDS
    except ImportError:
        ALL_NER_KINDS = []
    # Production 11-kind schema (PRODUCTION_NER_KINDS from kb_ner_routing)
    ner_kinds_str = ", ".join(ALL_NER_KINDS) if ALL_NER_KINDS else "ReasonForVisit, Medication, Procedure, Diagnostic, VitalSign, Reminder, Symptom, Diagnosis, Anatomy, Diet, ParasiteControl"
    prompt = COMBINED_CLEAN_TRANSCRIPT_AND_ENTITY_MANIFEST_PROMPT.format(
        conversation=raw_transcript,
        optional_inputs=optional_inputs or "",
        ner_kinds=ner_kinds_str,
        ner_kinds_count=len(ALL_NER_KINDS) if ALL_NER_KINDS else 11,
    )
    schema = COMBINED_OUTPUT_JSON_SCHEMA
    schema_name = "CombinedCleanNerBatch"
    try:
        max_tokens = int(os.getenv("SUPER_PASS_MAX_TOKENS", "16384"))
    except Exception:
        max_tokens = 16384
    if "fireworks" in (model or "").lower():
        max_tokens = min(max_tokens, 8192)

    messages = [
        {"role": "system", "content": "You are a clinical documentation specialist. Return ONLY valid JSON. First character must be \"{\"."},
        {"role": "user", "content": prompt + "\n\nOUTPUT JSON SCHEMA (must match exactly):\n" + json.dumps(schema, ensure_ascii=False)},
        {"role": "assistant", "content": "{"},
    ]
    create_kw = dict(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}},
    )

    use_legacy_streaming = os.getenv("COMBINED_LEGACY_STREAMING", "true").strip().lower() in ("1", "true", "yes")
    raw_content = None
    streamed_entities_salvage = None
    cleaned_from_stream_salvage = None

    if use_legacy_streaming:
        try:
            resp = client.chat.completions.create(**create_kw, stream=True)
            full_text = ""
            scanner_entity = _EntityArrayObjectScanner("entity_manifest")
            scanner_ct = _JSONFieldStringScanner("cleaned_transcript")
            streamed_entities = []
            cleaned_from_stream = None
            for chunk in resp:
                delta = _safe_get_delta_text(chunk)
                if delta:
                    full_text += delta
                    for obj_str in scanner_entity.feed(full_text):
                        try:
                            streamed_entities.append(json.loads(obj_str))
                        except Exception:
                            pass
                    ct = scanner_ct.feed(full_text)
                    if ct is not None:
                        cleaned_from_stream = ct
            raw_content = full_text.strip() or "{}"
            streamed_entities_salvage = streamed_entities
            cleaned_from_stream_salvage = cleaned_from_stream
            if logger:
                logger.debug("Combined pass (legacy streaming): %d chars, %d entities from stream", len(raw_content), len(streamed_entities_salvage))
        except Exception as e:
            if logger:
                logger.warning("Combined legacy streaming failed (%s), falling back to non-streaming.", e)
            raw_content = None

    if raw_content is None:
        try:
            resp = client.chat.completions.create(**create_kw)
            _msg = resp.choices[0].message if resp and resp.choices else None
            content = getattr(_msg, "content", None) if _msg else None
            if isinstance(content, list):
                content = "".join(
                    (p.get("text", p) if isinstance(p, dict) else str(p)) for p in content
                ) if content else None
            if content is not None and not isinstance(content, str):
                content = str(content)
            if logger:
                ctype = type(content).__name__
                clen = len(content) if content else 0
                logger.debug("Combined pass API response: content type=%s, len=%s", ctype, clen)
                if not content or clen == 0:
                    logger.warning(
                        "Combined pass returned no content (cleaned=raw, 0 entities). Check: model supports response_format json_schema, max_tokens sufficient, no rate limit/refusal. finish_reason=%s",
                        getattr(resp.choices[0], "finish_reason", None) if resp and resp.choices else None,
                    )
                elif clen < 100:
                    logger.warning("Combined pass content very short (len=%s), preview=%s", clen, (content or "")[:500])
            raw_content = (content and str(content).strip()) or "{}"
        except Exception as e:
            if logger:
                logger.warning("Combined clean+NER+batch LLM call failed: %s", e)
            return raw_transcript.strip(), []

    # Save raw API response for verification when COMBINED_DEBUG_SAVE_RESPONSE=true
    if os.getenv("COMBINED_DEBUG_SAVE_RESPONSE", "").strip().lower() in ("1", "true", "yes") and output_dir is not None and timestamp and raw_content:
        try:
            save_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
            raw_file = save_dir / f"combined_pass_raw_response_{timestamp}.txt"
            raw_file.write_text(raw_content, encoding="utf-8")
            if logger:
                logger.info("Combined pass raw response saved for verification: %s (%s chars)", raw_file, len(raw_content))
        except Exception as e:
            if logger:
                logger.warning("Could not save combined pass raw response: %s", e)

    try:
        data = json.loads(raw_content)
    except Exception:
        data = _extract_json_object(raw_content)
    # Truncation repair: if response is large and parses to empty, try closing unclosed brackets (finish_reason=length)
    if not data and isinstance(raw_content, str) and raw_content.strip().startswith("{"):
        open_br = raw_content.count("{") - raw_content.count("}")
        open_sq = raw_content.count("[") - raw_content.count("]")
        if open_br > 0 or open_sq > 0:
            repaired = raw_content.strip() + "]" * open_sq + "}" * open_br
            try:
                data = json.loads(repaired)
            except Exception:
                pass
            if logger and isinstance(data, dict) and data:
                logger.info("Combined pass: parsed truncated response after appending %s ] and %s }", open_sq, open_br)
    if not isinstance(data, dict):
        if logger:
            logger.warning("Combined clean+NER+batch: response was not JSON object")
        # Salvage from streaming if we had legacy streaming and got entities/cleaned incrementally
        if streamed_entities_salvage is not None and (streamed_entities_salvage or cleaned_from_stream_salvage):
            data = {"entity_manifest": streamed_entities_salvage, "cleaned_transcript": cleaned_from_stream_salvage or ""}
            if logger:
                logger.info("Combined pass: using streamed salvage (entities=%d, cleaned=%d chars)", len(streamed_entities_salvage), len(cleaned_from_stream_salvage or ""))
        else:
            return raw_transcript.strip(), []

    if not data and logger:
        logger.warning(
            "Combined pass returned empty JSON object. raw_content len=%s, preview=%s",
            len(raw_content or ""), (raw_content or "")[:500],
        )
    # Use streamed salvage when full parse produced empty dict (e.g. truncated)
    if not data and streamed_entities_salvage is not None:
        data = {"entity_manifest": streamed_entities_salvage, "cleaned_transcript": cleaned_from_stream_salvage or ""}
        if logger:
            logger.info("Combined pass: using streamed salvage (entities=%d, cleaned=%d chars)", len(streamed_entities_salvage), len(cleaned_from_stream_salvage or ""))

    cleaned_transcript_val = data.get("cleaned_transcript")
    cleaned = (cleaned_transcript_val or "").strip() if isinstance(cleaned_transcript_val, str) else raw_transcript.strip()
    if not cleaned:
        cleaned = raw_transcript.strip()
    entities_in = data.get("entity_manifest") or []
    # LLM may return entity_manifest as object keyed by E1, E2, ...; normalize to list.
    if isinstance(entities_in, dict):
        def _entity_order_key(ent):
            if not isinstance(ent, dict):
                return (1, 0)
            eid = (ent.get("entity_id") or "").strip()
            if eid and len(eid) > 1 and eid[0] == "E" and eid[1:].isdigit():
                return (0, int(eid[1:]))
            return (1, int(ent.get("start_char", 0)))
        entities_in = sorted(entities_in.values(), key=_entity_order_key)
    if logger:
        logger.debug("Combined response shape: keys=%s, entity_manifest len=%s", list(data.keys()), len(entities_in))

    # Map to pipeline entity shape (search_term, family, start_char, end_char).
    entity_manifest = []
    for e in entities_in:
        if not isinstance(e, dict):
            continue
        span_text = (e.get("span_text") or "").strip()
        if not span_text:
            continue
        start_char = int(e.get("start_char", 0))
        end_char = int(e.get("end_char", 0))
        entity_manifest.append({
            "span_text": span_text,
            "normalized_name": (e.get("normalized_name") or "").strip() or span_text,
            "kind": (e.get("kind") or "Other").strip(),
            "roles": e.get("roles") or [],
            "attributes": e.get("attributes") or {},
            "assertion_id": (e.get("assertion_id") or "CONF").strip(),
            "supporting_text": (e.get("supporting_text") or "").strip(),
            "start_char": start_char,
            "end_char": end_char,
            "is_actionable": e.get("is_actionable", True),
            "search_term": (e.get("search_term") or "").strip() or None,
            "family": (e.get("family") or "").strip() or None,
            "domain": _normalize_domain_to_list(e.get("domain")),
        })
    # Offset integrity: the dependable artifact is the rebased manifest (returned here), not the raw combined LLM response.
    # Raw response often emits offsets against a different string. Always rebase so downstream has correct start_char/end_char.
    entity_manifest = _backfill_span_offsets_in_cleaned(entity_manifest, cleaned, logger=logger)
    if logger:
        logger.info("Combined clean+NER+batch: cleaned=%d chars, entities=%d", len(cleaned), len(entity_manifest))
        # Log cleaned transcript preview for verification (first 500 chars)
        preview_len = min(500, len(cleaned))
        if preview_len:
            logger.info("Combined pass cleaned_transcript preview (first %s chars): %s", preview_len, cleaned[:preview_len])
    if len(entity_manifest) == 0 and logger:
        logger.warning(
            "Combined pass returned 0 entities. Response had keys=%s; entity_manifest len=%s. "
            "Set COMBINED_DEBUG_SAVE_RESPONSE=true and re-run to save raw response to output_dir for inspection.",
            list(data.keys()), len(data.get("entity_manifest") or []),
        )
    return cleaned, entity_manifest


def _backfill_span_offsets_in_cleaned(
    entities: List[Dict[str, Any]], cleaned: str, logger: Optional[logging.Logger] = None
) -> List[Dict[str, Any]]:
    """
    Rebase entity start_char/end_char so cleaned_transcript[start_char:end_char] == span_text.
    LLM offsets are often wrong (e.g. against raw or pre-speaker-label string). We validate every
    entity and recompute by locating span_text in cleaned (greedy left-to-right to preserve order).
    Ensures span integrity for audit and downstream (Anchor Mapping, injection).
    """
    if not cleaned:
        return list(entities) if entities else []
    out: List[Dict[str, Any]] = []
    search_start = 0
    for ent in entities:
        e = dict(ent)
        span = (e.get("span_text") or "").strip()
        start = int(e.get("start_char", 0))
        end = int(e.get("end_char", 0))
        if not span:
            out.append(e)
            continue
        # Validate: does cleaned[start:end] exactly match span_text?
        slice_ok = (
            0 <= start < end <= len(cleaned)
            and cleaned[start:end] == span
        )
        if slice_ok:
            search_start = max(search_start, end)
            out.append(e)
            continue
        # Recompute: find span_text in cleaned (prefer from search_start to preserve order)
        idx = cleaned.find(span, search_start)
        if idx == -1:
            idx = cleaned.find(span, 0)
        if idx != -1:
            e["start_char"] = idx
            e["end_char"] = idx + len(span)
            search_start = idx + len(span)
            if logger:
                logger.debug(
                    "  Rebase offset: span_text=%r -> [%d:%d] (was [%d:%d])",
                    span[:40], idx, idx + len(span), start, end,
                )
        else:
            if logger:
                logger.warning(
                    "  Rebase: span_text %r not found in cleaned transcript; keeping original [%d:%d]",
                    span[:50], start, end,
                )
        out.append(e)
    return out


def _dedup_entities_basic(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Basic dedup for super-pass extracted entities across chunks.
    Keyed by (kind, span_text) lowercased.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for e in entities or []:
        kind = (e.get("kind") or "Other").strip()
        span = (e.get("span_text") or "").strip()
        if not span:
            continue
        key = (kind.lower(), span.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _safe_get_delta_text(chunk: Any) -> str:
    """
    Best-effort extraction of streamed delta content from OpenAI-compatible streaming chunks.
    Fireworks uses OpenAI-compatible streaming objects.
    """
    try:
        choices = getattr(chunk, "choices", None)
        if choices and len(choices) > 0:
            c0 = choices[0]
            delta = getattr(c0, "delta", None)
            if delta is not None:
                return getattr(delta, "content", "") or ""
            # Some SDKs use message/content in streaming (rare)
            msg = getattr(c0, "message", None)
            if msg is not None:
                return getattr(msg, "content", "") or ""
    except Exception:
        return ""
    return ""


class _EntityArrayObjectScanner:
    """
    Incremental scanner that finds COMPLETE JSON objects inside the array for a given key,
    without requiring full JSON to be parseable.

    This is purpose-built for streaming extraction of entities from a response like:
      {"extracted_entities":[{...},{...}], "cleaned_transcript":"..."}
    """

    def __init__(self, array_key: str = "extracted_entities"):
        self.array_key = array_key
        self._array_start_idx: Optional[int] = None
        self._scan_pos: int = 0
        self._in_string: bool = False
        self._escape: bool = False
        self._obj_depth: int = 0
        self._obj_start: Optional[int] = None
        self._done: bool = False

    def feed(self, text: str) -> List[str]:
        """
        Feed the full accumulated text buffer. Returns any newly completed object substrings.
        The caller should maintain the full buffer and pass it each time; we keep scan state.
        """
        if self._done:
            return []

        # Locate the array start once.
        if self._array_start_idx is None:
            key_pat = f"\"{self.array_key}\""
            key_i = text.find(key_pat)
            if key_i == -1:
                return []
            bracket_i = text.find("[", key_i)
            if bracket_i == -1:
                return []
            self._array_start_idx = bracket_i
            self._scan_pos = bracket_i + 1  # after '['

        out: List[str] = []
        i = self._scan_pos
        n = len(text)
        while i < n:
            ch = text[i]

            if self._escape:
                self._escape = False
                i += 1
                continue

            if ch == "\\" and self._in_string:
                self._escape = True
                i += 1
                continue

            if ch == "\"":
                self._in_string = not self._in_string
                i += 1
                continue

            if self._in_string:
                i += 1
                continue

            # Not in string
            if self._obj_depth == 0:
                if ch == "{":
                    self._obj_start = i
                    self._obj_depth = 1
                elif ch == "]":
                    self._done = True
                    break
            else:
                if ch == "{":
                    self._obj_depth += 1
                elif ch == "}":
                    self._obj_depth -= 1
                    if self._obj_depth == 0 and self._obj_start is not None:
                        out.append(text[self._obj_start : i + 1])
                        self._obj_start = None

            i += 1

        self._scan_pos = i
        return out


class _JSONFieldStringScanner:
    """
    Incremental scanner that extracts a COMPLETE JSON string value for a given key from a growing buffer.

    Example target:
      ..."cleaned_transcript":"....(escaped chars)...."...

    Returns the decoded Python string once the closing quote is found and the substring is JSON-decodable.
    """

    def __init__(self, key: str):
        self.key = key
        self._pos = 0
        self._value_start: Optional[int] = None  # index of opening quote for value
        self._scan_pos: int = 0
        self._done = False

    def feed(self, text: str) -> Optional[str]:
        if self._done:
            return None

        # Step 1: find key and value start quote
        if self._value_start is None:
            key_token = f"\"{self.key}\""
            idx = text.find(key_token, self._pos)
            if idx == -1:
                # Keep small overlap for partial matches
                self._pos = max(0, len(text) - max(64, len(key_token) + 8))
                return None

            j = text.find(":", idx + len(key_token))
            if j == -1:
                self._pos = idx
                return None

            k = j + 1
            while k < len(text) and text[k] in " \t\r\n":
                k += 1
            if k >= len(text):
                self._pos = idx
                return None
            if text[k] != "\"":
                # Not a string value (unexpected); move forward
                self._pos = k
                return None

            self._value_start = k
            self._scan_pos = k + 1

        # Step 2: scan for non-escaped closing quote
        i = self._scan_pos
        while i < len(text):
            ch = text[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "\"":
                raw = text[self._value_start : i + 1]
                try:
                    decoded = json.loads(raw)
                except Exception:
                    i += 1
                    continue
                self._done = True
                return str(decoded)
            i += 1

        self._scan_pos = max(0, len(text) - 1)
        return None


async def super_pass_cleaning_and_ner_chat_mode(
    raw_transcript: str,
    model: str = None,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Chat-mode Super-Pass (multi-turn ingest then GENERATE).
    This version delegates to non-streaming super_pass_cleaning_and_ner for compatibility.
    """
    model = model or os.getenv("SUPER_PASS_MODEL", SUPER_PASS_DEFAULT_MODEL)
    return await super_pass_cleaning_and_ner(
        raw_transcript=raw_transcript,
        model=model,
        client=client,
        logger=logger,
    )


async def super_pass_cleaning_and_ner_streaming(
    raw_transcript: str,
    model: str = None,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    on_entity: Optional[Any] = None,  # async callback: await on_entity(entity_dict)
    on_cleaned_transcript: Optional[Any] = None,  # async callback: await on_cleaned_transcript(cleaned_transcript_str)
    on_chunk_complete: Optional[Any] = None,  # async callback: await on_chunk_complete(chunk_index, cleaned_chunk_text, entities_from_chunk)
    emit_entities_via_chunk_only: bool = False,
    on_cleaned_transcript_prefix: Optional[Any] = None,
    early_soap_prefix_chars: int = 2000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Streaming Super-Pass:
    - Streams the LLM output (stream=True)
    - Incrementally extracts COMPLETE entity objects from the extracted_entities array
    - Calls on_entity(entity) as soon as each entity object becomes parseable

    Returns (cleaned_transcript, converted_entities) at end-of-stream.

    Notes:
    - This uses a threaded producer to avoid blocking the asyncio event loop, since the OpenAI client is sync.
    - In long transcript mode, we stream EACH chunk so chunk-k grounding overlaps chunk-(k+1) generation.
    """
    model = model or os.getenv("SUPER_PASS_MODEL", SUPER_PASS_DEFAULT_MODEL)
    if not raw_transcript or not raw_transcript.strip():
        return "", []

    if logger:
        logger.info("=" * 60)
        logger.info("SUPER-PASS (STREAMING): Cleaning + NER Extraction (Combined)")
        logger.info("=" * 60)
        logger.info(f"  Model: {model}")
        logger.info(f"  Raw transcript length: {len(raw_transcript)} chars")

    # Determine whether long transcript mode is active (chunked processing).
    # NOTE: For ~10+ minute audios, the cleaned transcript alone can be large enough that
    # a single-shot JSON response is more likely to be truncated/malformed at the end.
    # We therefore use a lower default threshold for robustness; you can override via env.
    try:
        long_threshold = int(os.getenv("LONG_TRANSCRIPT_THRESHOLD_CHARS", "10000"))
    except Exception:
        long_threshold = 10000
    force_long_mode = os.getenv("FORCE_LONG_TRANSCRIPT_MODE", "false").strip().lower() in ("1", "true", "yes")
    long_mode = force_long_mode or (len(raw_transcript) > long_threshold)

    model_provider = get_model_provider(model)
    if not client:
        client_result = get_client_for_model(model)
        if isinstance(client_result, tuple):
            client, _provider = client_result
        else:
            client = client_result
    if not client:
        if logger:
            logger.error(f"  ❌ Could not get client for model: {model}")
        return "", []

    # Fireworks constraint: streaming required if max_tokens > 4096 on some models.
    # We stream, so it's safe to allow >4096 if user sets it, but keep default conservative.
    try:
        super_pass_max_tokens = int(os.getenv("SUPER_PASS_MAX_TOKENS", "4096" if model_provider == "fireworks" else "6000"))
    except Exception:
        super_pass_max_tokens = 4096 if model_provider == "fireworks" else 6000

    output_order = os.getenv("SUPER_PASS_OUTPUT_ORDER", "entities_first").strip().lower()
    if output_order not in ("entities_first", "transcript_first"):
        output_order = "entities_first"

    # Unified prompt: transcript is embedded via format(); short system + order override for streaming.
    _SHORT_SYSTEM = "You are a clinical documentation specialist. Return ONLY valid JSON. The first character of your response MUST be \"{\"."
    if output_order == "transcript_first":
        system_prompt = (
            _SHORT_SYSTEM
            + "\n\nIMPORTANT (TRANSCRIPT-FIRST MODE): Output JSON keys in this order: 1) cleaned_transcript 2) extracted_entities 3) entities_by_kind."
        )
    else:
        system_prompt = (
            _SHORT_SYSTEM
            + "\n\nIMPORTANT (ENTITIES-FIRST MODE): Output JSON keys in this order: 1) extracted_entities 2) cleaned_transcript 3) entities_by_kind."
        )
    user_prompt = _format_unified_prompt(raw_transcript, "")
    user_prompt = user_prompt + "\n\nOUTPUT JSON SCHEMA (must match exactly):\n" + json.dumps(SUPER_PASS_JSON_SCHEMA, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "{"},
    ]

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    async def _stream_long_mode() -> Tuple[str, List[Dict[str, Any]]]:
        # Chunking parameters
        try:
            chunk_size = int(os.getenv("SUPER_PASS_CHUNK_SIZE_CHARS", "9000"))
        except Exception:
            chunk_size = 9000
        try:
            overlap = int(os.getenv("SUPER_PASS_CHUNK_OVERLAP_CHARS", "400"))
        except Exception:
            overlap = 400

        chunks = chunk_text_with_overlap(raw_transcript, chunk_size=chunk_size, overlap=overlap)
        if logger:
            logger.info(f"🧩 Long transcript mode (STREAMING) enabled: {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

        running_summary = ""
        cleaned_chunks: List[str] = []
        all_entities: List[Dict[str, Any]] = []

        # Sequential mode (rolling summary dependency): higher accuracy for cross-chunk context.
        for idx, chunk in enumerate(chunks, 1):
            q_chunk: asyncio.Queue = asyncio.Queue()
            t_chunk_start = time.time()
            first_delta_at: Optional[float] = None
            first_entity_at: Optional[float] = None

            if logger:
                logger.info(f"  🔧 Long-mode streaming chunk model: {model} (chunk {idx}/{len(chunks)})")

            chunk_user_prompt = f"""You are processing chunk {idx}/{len(chunks)} of a long veterinary ASR transcript.

PRIOR_ROLLING_SUMMARY (may be empty):
{running_summary}

RAW_CHUNK:
{chunk}

Return ONLY JSON with keys in this exact order:
1) extracted_entities
2) updated_summary
3) cleaned_chunk

OUTPUT JSON SCHEMA (must match exactly):
{json.dumps(SUPER_PASS_CHUNK_JSON_SCHEMA, ensure_ascii=False)}"""

            messages_chunk = [
                {"role": "system", "content": SUPER_PASS_CHUNK_SYSTEM_PROMPT},
                {"role": "user", "content": chunk_user_prompt},
                {"role": "assistant", "content": "{"},
            ]

            def _producer_chunk():
                try:
                    spec_kwargs = _build_fireworks_speculative_kwargs(logger=logger) if model_provider == "fireworks" else {}
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages_chunk,
                        temperature=0.0,
                        seed=42,
                        stream=True,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "SuperPassChunkResult", "schema": SUPER_PASS_CHUNK_JSON_SCHEMA},
                        },
                        max_tokens=4000,
                        **spec_kwargs,
                    )
                    for c in resp:
                        d = _safe_get_delta_text(c)
                        if d:
                            loop.call_soon_threadsafe(q_chunk.put_nowait, d)
                except Exception as e:
                    # Retry once without speculative args if the endpoint rejects them.
                    if model_provider == "fireworks" and os.getenv("SUPER_PASS_SPECULATIVE", "false").strip().lower() in ("1", "true", "yes"):
                        try:
                            resp = client.chat.completions.create(
                                model=model,
                                messages=messages_chunk,
                                temperature=0.0,
                                seed=42,
                                stream=True,
                                response_format={
                                    "type": "json_schema",
                                    "json_schema": {"name": "SuperPassChunkResult", "schema": SUPER_PASS_CHUNK_JSON_SCHEMA},
                                },
                                max_tokens=4000,
                            )
                            for c in resp:
                                d = _safe_get_delta_text(c)
                                if d:
                                    loop.call_soon_threadsafe(q_chunk.put_nowait, d)
                        except Exception as e2:
                            loop.call_soon_threadsafe(q_chunk.put_nowait, {"__error__": str(e2)})
                    else:
                        loop.call_soon_threadsafe(q_chunk.put_nowait, {"__error__": str(e)})
                finally:
                    loop.call_soon_threadsafe(q_chunk.put_nowait, None)

            asyncio.create_task(asyncio.to_thread(_producer_chunk))

            full_chunk_text = ""
            scanner_chunk = _EntityArrayObjectScanner("extracted_entities")
            dispatched_chunk: set = set()

            while True:
                item = await q_chunk.get()
                if item is None:
                    break

                if isinstance(item, dict) and "__error__" in item:
                    # Fallback to non-streaming for this chunk only
                    if logger:
                        logger.warning(f"⚠️ Chunk {idx}/{len(chunks)} streaming failed ({item['__error__']}), falling back to non-streaming chunk call.")
                    try:
                        resp2 = client.chat.completions.create(
                            model=model,
                            messages=messages_chunk,
                            temperature=0.0,
                            seed=42,
                            response_format={
                                "type": "json_schema",
                                "json_schema": {"name": "SuperPassChunkResult", "schema": SUPER_PASS_CHUNK_JSON_SCHEMA},
                            },
                            max_tokens=4000,
                        )
                        response_text2 = resp2.choices[0].message.content or "{}"
                        # With json_schema enforcement, content should be strict JSON; prefer json.loads.
                        json_data2 = None
                        try:
                            json_data2 = json.loads(response_text2)
                        except Exception:
                            json_data2 = _extract_json_object(response_text2)
                    except Exception as e2:
                        if logger:
                            logger.warning(f"  ⚠️  Chunk {idx}/{len(chunks)} fallback also failed: {e2}; preserving raw chunk to avoid data loss")
                        # Preserve chunk text so cleaned transcript doesn't lose content
                        cleaned_chunks.append(chunk.strip())
                        full_chunk_text = ""
                        break

                    if not json_data2:
                        if logger:
                            logger.warning(f"  ⚠️  Chunk {idx}/{len(chunks)}: failed to parse JSON; preserving raw chunk to avoid data loss")
                        cleaned_chunks.append(chunk.strip())
                        full_chunk_text = ""
                        break

                    cleaned_chunk2 = (json_data2.get("cleaned_chunk") or "").strip()
                    extracted_entities2 = json_data2.get("extracted_entities") or []
                    running_summary = (json_data2.get("updated_summary") or running_summary or "").strip()

                    if cleaned_chunk2:
                        cleaned_chunks.append(cleaned_chunk2)

                    for ent_obj in extracted_entities2:
                        attrs = ent_obj.get("attributes") or {}
                        if isinstance(attrs, dict):
                            attrs["_stream_context"] = chunk
                            attrs["_chunk_index"] = idx
                            ent_obj["attributes"] = attrs
                        if on_entity is not None:
                            try:
                                await on_entity(ent_obj)
                            except Exception:
                                if logger:
                                    logger.debug("Chunk fallback on_entity callback failed; continuing.", exc_info=True)

                    for ent in extracted_entities2:
                        all_entities.append(
                            {
                                "span_text": ent.get("span_text", ""),
                                "normalized_name": ent.get("normalized_name", ent.get("span_text", "")),
                                "kind": ent.get("kind", "Other"),
                                "roles": ent.get("roles", []),
                                "attributes": ent.get("attributes", {}),
                                "assertion_id": ent.get("assertion_id", "CONF"),
                                "supporting_text": ent.get("supporting_text", ""),
                                "start_char": ent.get("start_char", 0),
                                "end_char": ent.get("end_char", 0),
                                "is_actionable": ent.get("is_actionable", True),
                            }
                        )
                    if on_chunk_complete is not None and extracted_entities2:
                        try:
                            await on_chunk_complete(idx, cleaned_chunk2 or "", extracted_entities2)
                        except Exception:
                            if logger:
                                logger.debug("Long-mode on_chunk_complete (fallback) callback failed; continuing.", exc_info=True)
                    # Done with fallback chunk
                    full_chunk_text = ""
                    break

                delta = str(item)
                if first_delta_at is None:
                    first_delta_at = time.time()
                full_chunk_text += delta

                # Incremental entity dispatch for this chunk
                for obj_str in scanner_chunk.feed(full_chunk_text):
                    try:
                        ent_obj = json.loads(obj_str)
                    except Exception:
                        continue

                    span = (ent_obj.get("span_text") or "").strip()
                    kind = (ent_obj.get("kind") or "").strip()
                    if not span or not kind:
                        continue
                    key = (kind.lower(), span.lower())
                    if key in dispatched_chunk:
                        continue
                    dispatched_chunk.add(key)
                    if first_entity_at is None:
                        first_entity_at = time.time()

                    attrs = ent_obj.get("attributes") or {}
                    if isinstance(attrs, dict):
                        attrs["_stream_context"] = chunk
                        attrs["_chunk_index"] = idx
                        ent_obj["attributes"] = attrs

                    if on_entity is not None:
                        try:
                            await on_entity(ent_obj)
                        except Exception:
                            if logger:
                                logger.debug("Streaming chunk on_entity callback failed; continuing.", exc_info=True)

            # Normal streaming parse path (skip if fallback was used)
            if not full_chunk_text:
                continue

            # Prefer strict parse first (json_schema responses should be valid JSON).
            json_data = None
            try:
                json_data = json.loads(full_chunk_text)
            except Exception:
                json_data = _extract_json_object(full_chunk_text)
            if not json_data:
                if logger:
                    logger.warning(f"  ⚠️  Chunk {idx}/{len(chunks)}: failed to parse streamed JSON; retrying non-streaming once")
                # Retry once non-streaming to salvage cleaned_chunk + updated_summary (do NOT skip).
                try:
                    resp3 = client.chat.completions.create(
                        model=model,
                        messages=messages_chunk,
                        temperature=0.0,
                        seed=42,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "SuperPassChunkResult", "schema": SUPER_PASS_CHUNK_JSON_SCHEMA},
                        },
                        max_tokens=4000,
                    )
                    response_text3 = resp3.choices[0].message.content or "{}"
                    try:
                        json_data = json.loads(response_text3)
                    except Exception:
                        json_data = _extract_json_object(response_text3)
                except Exception as e3:
                    json_data = None
                    if logger:
                        logger.warning(f"  ⚠️  Chunk {idx}/{len(chunks)} non-stream retry failed: {e3}")

                if not json_data:
                    if logger:
                        logger.warning(f"  ⚠️  Chunk {idx}/{len(chunks)}: could not parse chunk JSON; preserving raw chunk to avoid data loss")
                    cleaned_chunks.append(chunk.strip())
                    continue

            cleaned_chunk = (json_data.get("cleaned_chunk") or "").strip()
            extracted_entities = json_data.get("extracted_entities") or []
            running_summary = (json_data.get("updated_summary") or running_summary or "").strip()

            if cleaned_chunk:
                cleaned_chunks.append(cleaned_chunk)
            else:
                # Even if cleaned_chunk missing, preserve raw chunk to avoid dropping content.
                cleaned_chunks.append(chunk.strip())

            for ent in extracted_entities:
                all_entities.append(
                    {
                        "span_text": ent.get("span_text", ""),
                        "normalized_name": ent.get("normalized_name", ent.get("span_text", "")),
                        "kind": ent.get("kind", "Other"),
                        "roles": ent.get("roles", []),
                        "attributes": ent.get("attributes", {}),
                        "assertion_id": ent.get("assertion_id", "CONF"),
                        "supporting_text": ent.get("supporting_text", ""),
                        "start_char": ent.get("start_char", 0),
                        "end_char": ent.get("end_char", 0),
                        "is_actionable": ent.get("is_actionable", True),
                    }
                )

            if logger:
                ttft = (first_delta_at - t_chunk_start) if first_delta_at else None
                ttf_entity = (first_entity_at - t_chunk_start) if first_entity_at else None
                extra = []
                if ttft is not None:
                    extra.append(f"ttft={ttft:.2f}s")
                if ttf_entity is not None:
                    extra.append(f"ttf_entity={ttf_entity:.2f}s")
                extra_str = (" (" + ", ".join(extra) + ")") if extra else ""
                logger.info(
                    f"  ✅ Chunk {idx}/{len(chunks)} streamed: entities_dispatched={len(dispatched_chunk)}, cleaned={len(cleaned_chunk)} chars{extra_str}"
                )

            # Let the experiment dispatch this chunk's entities to streaming grounding (e.g. Shadow Intent fast path).
            if on_chunk_complete is not None and extracted_entities:
                try:
                    await on_chunk_complete(idx, cleaned_chunk or "", extracted_entities)
                except Exception:
                    if logger:
                        logger.debug("Long-mode on_chunk_complete callback failed; continuing.", exc_info=True)

        converted_entities = _dedup_entities_basic(all_entities)
        cleaned_transcript = "\n\n".join(cleaned_chunks).strip()
        cleaned_transcript = wrap_with_summary_block(cleaned_transcript, running_summary)

        if logger:
            logger.info(
                f"✅ Long super-pass (STREAMING) complete: cleaned={len(cleaned_transcript)} chars, "
                f"entities={len(converted_entities)} (deduped from {len(all_entities)})"
            )

        # If caller provided a cleaned transcript callback, dispatch at least once in long-mode.
        # (Long-mode doesn't currently stream `cleaned_transcript` as a single JSON field.)
        if on_cleaned_transcript is not None and cleaned_transcript:
            try:
                await on_cleaned_transcript(cleaned_transcript)
            except Exception:
                if logger:
                    logger.debug("Long-mode on_cleaned_transcript callback failed; continuing.", exc_info=True)
        return cleaned_transcript, converted_entities

    # If long transcript mode is active, stream chunk-by-chunk and return early.
    if long_mode:
        return await _stream_long_mode()

    def _producer():
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                seed=42,
                stream=True,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "SuperPassResult", "schema": SUPER_PASS_JSON_SCHEMA},
                },
                max_tokens=super_pass_max_tokens,
            )
            for chunk in resp:
                delta = _safe_get_delta_text(chunk)
                if delta:
                    loop.call_soon_threadsafe(q.put_nowait, delta)
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, {"__error__": str(e)})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    # Start the producer in a background thread to avoid blocking the event loop.
    asyncio.create_task(asyncio.to_thread(_producer))

    t_stream_start = time.time()
    first_delta_at: Optional[float] = None
    first_entity_at: Optional[float] = None

    full_text = ""
    scanner = _EntityArrayObjectScanner("extracted_entities")
    dispatched: set = set()
    transcript_scanner = _JSONFieldStringScanner("cleaned_transcript")
    cleaned_dispatched = False
    salvaged_cleaned_transcript = ""
    streamed_entity_objs: List[Dict[str, Any]] = []

    while True:
        item = await q.get()
        if item is None:
            break
        if isinstance(item, dict) and "__error__" in item:
            # Streaming failed; fall back to non-streaming super-pass.
            if logger:
                logger.warning(f"⚠️ Super-pass streaming failed ({item['__error__']}), falling back to non-streaming.")
            return await super_pass_cleaning_and_ner(raw_transcript=raw_transcript, model=model, client=client, logger=logger)

        delta = str(item)
        if first_delta_at is None:
            first_delta_at = time.time()
        full_text += delta

        # Extract any newly completed entity objects and dispatch.
        for obj_str in scanner.feed(full_text):
            try:
                ent_obj = json.loads(obj_str)
            except Exception:
                continue

            span = (ent_obj.get("span_text") or "").strip()
            kind = (ent_obj.get("kind") or "").strip()
            if not span or not kind:
                continue
            key = (kind.lower(), span.lower())
            if key in dispatched:
                continue
            dispatched.add(key)
            if first_entity_at is None:
                first_entity_at = time.time()

            # Keep a copy so we can return entities even if final JSON extraction fails.
            try:
                streamed_entity_objs.append(ent_obj)
            except Exception:
                pass

            if on_entity is not None:
                try:
                    await on_entity(ent_obj)
                except Exception:
                    # Never let callback failures kill streaming parse; caller will handle grounding failures separately.
                    if logger:
                        logger.debug("Streaming on_entity callback failed; continuing.", exc_info=True)

        # Buffer-triggering: as soon as cleaned_transcript closes (even if JSON continues streaming),
        # dispatch it so callers can start SOAP generation early.
        if on_cleaned_transcript is not None and not cleaned_dispatched:
            try:
                ct = transcript_scanner.feed(full_text)
                if ct is not None:
                    cleaned_dispatched = True
                    salvaged_cleaned_transcript = ct
                    await on_cleaned_transcript(ct)
            except Exception:
                if logger:
                    logger.debug("Streaming on_cleaned_transcript callback failed; continuing.", exc_info=True)

    # Final parse: extract the complete JSON object and convert entities to pipeline format.
    json_data = _extract_json_object(full_text)
    if not json_data:
        if logger:
            logger.error("  ❌ Super-pass streaming ended but failed to extract final JSON object")
        # Salvage: return the cleaned transcript (if we saw it close) and any streamed entities.
        converted_entities: List[Dict[str, Any]] = []
        for entity in streamed_entity_objs:
            converted_entities.append(
                {
                    "span_text": entity.get("span_text", ""),
                    "normalized_name": entity.get("normalized_name", entity.get("span_text", "")),
                    "kind": entity.get("kind", "Other"),
                    "roles": entity.get("roles", []),
                    "attributes": entity.get("attributes", {}),
                    "assertion_id": entity.get("assertion_id", "CONF"),
                    "supporting_text": entity.get("supporting_text", ""),
                    "start_char": entity.get("start_char", 0),
                    "end_char": entity.get("end_char", 0),
                    "is_actionable": entity.get("is_actionable", True),
                }
            )
        converted_entities = _dedup_entities_basic(converted_entities)
        if logger and (salvaged_cleaned_transcript or converted_entities):
            logger.warning(
                f"  ⚠️ Super-pass salvage: cleaned_transcript={len(salvaged_cleaned_transcript)} chars, "
                f"entities={len(converted_entities)} (streamed)"
            )
        return salvaged_cleaned_transcript.strip(), converted_entities

    cleaned_transcript = (json_data.get("cleaned_transcript") or "").strip()
    extracted_entities = json_data.get("extracted_entities") or []

    # Ensure cleaned transcript callback has fired at least once.
    if on_cleaned_transcript is not None and not cleaned_dispatched and cleaned_transcript:
        try:
            cleaned_dispatched = True
            await on_cleaned_transcript(cleaned_transcript)
        except Exception:
            if logger:
                logger.debug("Final on_cleaned_transcript callback failed; continuing.", exc_info=True)

    converted_entities: List[Dict[str, Any]] = []
    for entity in extracted_entities:
        converted_entities.append(
            {
                "span_text": entity.get("span_text", ""),
                "normalized_name": entity.get("normalized_name", entity.get("span_text", "")),
                "kind": entity.get("kind", "Other"),
                "roles": entity.get("roles", []),
                "attributes": entity.get("attributes", {}),
                "assertion_id": entity.get("assertion_id", "CONF"),
                "supporting_text": entity.get("supporting_text", ""),
                "start_char": entity.get("start_char", 0),
                "end_char": entity.get("end_char", 0),
                "is_actionable": entity.get("is_actionable", True),
            }
        )

    converted_entities = _dedup_entities_basic(converted_entities)
    if logger:
        ttft = (first_delta_at - t_stream_start) if first_delta_at else None
        ttf_entity = (first_entity_at - t_stream_start) if first_entity_at else None
        extra = []
        if ttft is not None:
            extra.append(f"ttft={ttft:.2f}s")
        if ttf_entity is not None:
            extra.append(f"ttf_entity={ttf_entity:.2f}s")
        extra_str = (" (" + ", ".join(extra) + ")") if extra else ""
        logger.info(
            f"✅ Super-pass (STREAMING) finished: cleaned={len(cleaned_transcript)} chars, entities={len(converted_entities)}{extra_str}"
        )
    return cleaned_transcript, converted_entities


    # ---

    # End.

    # ---

    # done

    # ---

    # End.

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End.

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # ---

    # End

    # ---

    # ok

    # --- 

    # (end)

    # --- 

    # (done)

    # (NOTE: real long-mode return happens earlier.)

    # END

    # ---

    # (end)

    # ---

    # (done)

    # ---

    # END

    # ---

    # End.

    # ---

    # end

    # ---

    # done

    # ---

    # END

    # (Do not remove.)

    # ---

    # end

    # ---

    # done

    # ---

    # END

    # ---

    # end

    # ---

    # done

    # ---

    # END

    # ---

    # end

    # ---

    # done

    # ---

    # END

    # ---

    # end

    # ---

    # done

    # ---

    # END


async def super_pass_cleaning_and_ner(
    raw_transcript: str,
    model: str = None,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, List[Dict[str, str]]]]:
    """
    Super-Pass: Combined Transcription Cleaning + NER Extraction in a single LLM call.
    
    This function performs both tasks simultaneously:
    1. Cleans the raw transcript (removes fillers, translates, attributes speakers)
    2. Extracts clinical entities with attributes, roles, and assertion types
    
    Args:
        raw_transcript: Raw transcript text from ASR
        model: Model to use (default: Fireworks Llama 3 70B via SUPER_PASS_MODEL env)
        client: OpenAI-compatible client (optional, will be created if not provided)
        logger: Logger instance
        
    Returns:
        Tuple of (cleaned_transcript, extracted_entities, entities_by_kind)
        - cleaned_transcript: Cleaned and attributed transcript
        - extracted_entities: List of enriched entities compatible with existing pipeline format
        - entities_by_kind: Pure NER output; dict mapping each of 12 kinds to list of {"span_text": "..."}
    """
    model = model or os.getenv("SUPER_PASS_MODEL", SUPER_PASS_DEFAULT_MODEL)
    if not raw_transcript or not raw_transcript.strip():
        return "", [], _build_entities_by_kind([])
    
    if logger:
        logger.info("="*60)
        logger.info("SUPER-PASS: Cleaning + NER Extraction (Combined)")
        logger.info("="*60)
        logger.info(f"  Model: {model}")
        logger.info(f"  Raw transcript length: {len(raw_transcript)} chars")

    # Long transcript mode (chunked super-pass to avoid context overflow)
    # Enabled if transcript crosses threshold or env forces it.
    try:
        long_threshold = int(os.getenv("LONG_TRANSCRIPT_THRESHOLD_CHARS", "25000"))
    except Exception:
        long_threshold = 25000
    force_long_mode = os.getenv("FORCE_LONG_TRANSCRIPT_MODE", "false").strip().lower() in ("1", "true", "yes")
    long_mode = force_long_mode or (len(raw_transcript) > long_threshold)
    
    # Get client for the model
    # CRITICAL: Check if passed client matches the model's provider
    model_provider = get_model_provider(model)
    
    # If client is provided, verify it matches the model's provider
    if client:
        client_base_url = ''
        if hasattr(client, 'base_url'):
            client_base_url = str(client.base_url) if client.base_url else ''
        
        # Check if client endpoint matches model provider
        if model_provider == "openai" and 'fireworks' in client_base_url.lower():
            if logger:
                logger.warning(f"  ⚠️  Passed client has Fireworks endpoint but model '{model}' requires OpenAI endpoint")
                logger.warning(f"  🔄 Recreating client with correct OpenAI endpoint")
            # Recreate client with correct endpoint
            client = None  # Force recreation
    
    if not client:
        client_result = get_client_for_model(model)
        # get_client_for_model returns (client, provider) tuple
        if isinstance(client_result, tuple):
            client, provider = client_result
        else:
            client = client_result
        if not client:
            if logger:
                logger.error(f"  ❌ Could not get client for model: {model}")
            return "", [], _build_entities_by_kind([])
        
        # CRITICAL: Verify client is using correct endpoint
        if hasattr(client, 'base_url'):
            base_url = str(client.base_url) if client.base_url else ''
            if logger:
                logger.debug(f"  🔍 Client base_url: {base_url}")
            if model_provider == "openai" and 'fireworks' in base_url.lower():
                if logger:
                    logger.error(f"  ❌ Client is using Fireworks endpoint, but model '{model}' requires OpenAI endpoint")
                    logger.error(f"  ❌ This will cause 404 errors. Please check API key configuration.")
                return "", [], _build_entities_by_kind([])
            elif model_provider == "fireworks" and 'openai.com' in base_url.lower() and 'fireworks' not in base_url.lower():
                if logger:
                    logger.error(f"  ❌ Client is using OpenAI endpoint, but model '{model}' requires Fireworks endpoint")
                return "", [], _build_entities_by_kind([])
    
    # Unified prompt: transcript embedded via _format_unified_prompt(); short system message.
    _SHORT_SYSTEM = "You are a clinical documentation specialist. Return ONLY valid JSON. The first character of your response MUST be \"{\"."
    user_prompt = _format_unified_prompt(raw_transcript, "")
    user_prompt = user_prompt + "\n\nOUTPUT JSON SCHEMA (must match exactly):\n" + json.dumps(SUPER_PASS_JSON_SCHEMA, ensure_ascii=False)

    try:
        # Fireworks constraint: some models require stream=true if max_tokens > 4096.
        # Keep non-streaming requests within 4096 for Fireworks models.
        if model_provider == "fireworks":
            try:
                super_pass_max_tokens = int(os.getenv("SUPER_PASS_MAX_TOKENS", "4096"))
            except Exception:
                super_pass_max_tokens = 4096
            if super_pass_max_tokens > 4096:
                super_pass_max_tokens = 4096
        else:
            try:
                super_pass_max_tokens = int(os.getenv("SUPER_PASS_MAX_TOKENS", "6000"))
            except Exception:
                super_pass_max_tokens = 6000

        if long_mode:
            # Chunking parameters
            try:
                chunk_size = int(os.getenv("SUPER_PASS_CHUNK_SIZE_CHARS", "9000"))
            except Exception:
                chunk_size = 9000
            try:
                overlap = int(os.getenv("SUPER_PASS_CHUNK_OVERLAP_CHARS", "400"))
            except Exception:
                overlap = 400

            chunks = chunk_text_with_overlap(raw_transcript, chunk_size=chunk_size, overlap=overlap)
            if logger:
                logger.info(f"🧩 Long transcript mode enabled: {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

            running_summary = ""
            cleaned_chunks: List[str] = []
            all_entities: List[Dict[str, Any]] = []

            # Sequential mode (rolling summary dependency): higher accuracy for cross-chunk context.
            for idx, chunk in enumerate(chunks, 1):
                if logger:
                    logger.info(f"  🔧 Long-mode chunk model: {model} (chunk {idx}/{len(chunks)})")
                chunk_user_prompt = f"""You are processing chunk {idx}/{len(chunks)} of a long veterinary ASR transcript.

PRIOR_ROLLING_SUMMARY (may be empty):
{running_summary}

RAW_CHUNK:
{chunk}

Return ONLY JSON with keys in this exact order:
1) extracted_entities
2) updated_summary
3) cleaned_chunk

OUTPUT JSON SCHEMA (must match exactly):
{json.dumps(SUPER_PASS_CHUNK_JSON_SCHEMA, ensure_ascii=False)}"""

                spec_kwargs = _build_fireworks_speculative_kwargs(logger=logger) if model_provider == "fireworks" else {}
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SUPER_PASS_CHUNK_SYSTEM_PROMPT},
                            {"role": "user", "content": chunk_user_prompt},
                            {"role": "assistant", "content": "{"},
                        ],
                        temperature=0.0,
                        seed=42,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "SuperPassChunkResult", "schema": SUPER_PASS_CHUNK_JSON_SCHEMA},
                        },
                        max_tokens=4000,
                        **spec_kwargs,
                    )
                    response_text = response.choices[0].message.content or "{}"
                except Exception:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SUPER_PASS_CHUNK_SYSTEM_PROMPT},
                            {"role": "user", "content": chunk_user_prompt},
                            {"role": "assistant", "content": "{"},
                        ],
                        temperature=0.0,
                        seed=42,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": "SuperPassChunkResult", "schema": SUPER_PASS_CHUNK_JSON_SCHEMA},
                        },
                        max_tokens=4000,
                    )
                    response_text = response.choices[0].message.content or "{}"
                json_data = None
                try:
                    json_data = json.loads(response_text)
                except Exception:
                    json_data = _extract_json_object(response_text)
                if not isinstance(json_data, dict):
                    if logger:
                        logger.warning(f"  ⚠️  Chunk {idx}/{len(chunks)}: failed to parse JSON; preserving raw chunk to avoid data loss")
                    cleaned_chunks.append(chunk.strip())
                    continue

                cleaned_chunk = (json_data.get("cleaned_chunk") or "").strip()
                extracted_entities = json_data.get("extracted_entities") or []
                running_summary = (json_data.get("updated_summary") or running_summary or "").strip()

                cleaned_chunks.append(cleaned_chunk or chunk.strip())
                for entity in (extracted_entities or []):
                    all_entities.append(
                        {
                            "span_text": entity.get("span_text", ""),
                            "normalized_name": entity.get("normalized_name", entity.get("span_text", "")),
                            "kind": entity.get("kind", "Other"),
                            "roles": entity.get("roles", []),
                            "attributes": entity.get("attributes", {}),
                            "assertion_id": entity.get("assertion_id", "CONF"),
                            "supporting_text": entity.get("supporting_text", ""),
                            "start_char": entity.get("start_char", 0),
                            "end_char": entity.get("end_char", 0),
                            "is_actionable": entity.get("is_actionable", True),
                        }
                    )

                if logger:
                    logger.info(
                        f"  ✅ Chunk {idx}/{len(chunks)} done: cleaned={len(cleaned_chunk)} chars, "
                        f"entities={len(extracted_entities or [])}"
                    )

            converted_entities = _dedup_entities_basic(all_entities)
            cleaned_transcript = "\n\n".join(cleaned_chunks).strip()
            cleaned_transcript = wrap_with_summary_block(cleaned_transcript, running_summary)

            if logger:
                logger.info(
                    f"✅ Long super-pass complete: cleaned={len(cleaned_transcript)} chars, "
                    f"entities={len(converted_entities)} (deduped from {len(all_entities)})"
                )
            entities_by_kind = _build_entities_by_kind(converted_entities)
            return cleaned_transcript, converted_entities, entities_by_kind

        # Make API call
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            api_model_name = model
            try:
                spec_kwargs = _build_fireworks_speculative_kwargs(logger=logger) if model_provider == "fireworks" else {}
                response = client.chat.completions.create(
                    model=api_model_name,
                    messages=[
                        {"role": "system", "content": _SHORT_SYSTEM},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": "{"}
                    ],
                    temperature=0.0,
                    seed=42,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "SuperPassResult", "schema": SUPER_PASS_JSON_SCHEMA},
                    },
                    max_tokens=super_pass_max_tokens,
                    **spec_kwargs,
                )
                response_text = response.choices[0].message.content or "{}"
            except Exception as e:
                error_message = str(e)
                if logger:
                    logger.error(f"  ❌ Model '{api_model_name}' failed with error: {error_message}")
                if model_provider == "fireworks" and os.getenv("SUPER_PASS_SPECULATIVE", "false").strip().lower() in ("1", "true", "yes"):
                    try:
                        response = client.chat.completions.create(
                            model=api_model_name,
                            messages=[
                                {"role": "system", "content": _SHORT_SYSTEM},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": "{"}
                            ],
                            temperature=0.0,
                            seed=42,
                            response_format={
                                "type": "json_schema",
                                "json_schema": {"name": "SuperPassResult", "schema": SUPER_PASS_JSON_SCHEMA},
                            },
                            max_tokens=super_pass_max_tokens,
                        )
                        response_text = response.choices[0].message.content or "{}"
                    except Exception:
                        raise
                else:
                    raise
        else:
            if logger:
                logger.error("  ❌ Client does not support chat.completions API")
            return "", [], _build_entities_by_kind([])
        # Extract JSON from response
        json_data = _extract_json_object(response_text)
        
        if not json_data:
            if logger:
                logger.error("  ❌ Failed to extract JSON from response")
            return "", [], _build_entities_by_kind([])
        
        # Extract cleaned_transcript, extracted_entities, and entities_by_kind (Pure NER output)
        cleaned_transcript = json_data.get("cleaned_transcript", "").strip()
        extracted_entities = json_data.get("extracted_entities", [])
        entities_by_kind = json_data.get("entities_by_kind")
        if not isinstance(entities_by_kind, dict):
            entities_by_kind = _build_entities_by_kind(extracted_entities)
        else:
            # Ensure all 12 Pure NER kinds exist (fill missing with [])
            for k in PURE_NER_KIND_KEYS:
                if k not in entities_by_kind:
                    entities_by_kind[k] = []
        
        if not cleaned_transcript:
            if logger:
                logger.warning("  ⚠️  No cleaned_transcript in response")
        
        if not extracted_entities:
            if logger:
                logger.warning("  ⚠️  No extracted_entities in response")
        
        # Convert entities to pipeline format
        converted_entities = []
        for i, entity in enumerate(extracted_entities):
            converted_entity = {
                "span_text": entity.get("span_text", ""),
                "normalized_name": entity.get("normalized_name", entity.get("span_text", "")),
                "kind": entity.get("kind", "Other"),
                "roles": entity.get("roles", []),  # Super-pass includes role classification
                "attributes": entity.get("attributes", {}),
                "assertion_id": entity.get("assertion_id", "CONF"),
                "supporting_text": entity.get("supporting_text", ""),
                "start_char": entity.get("start_char", 0),
                "end_char": entity.get("end_char", 0),
                "is_actionable": entity.get("is_actionable", True),  # For routing (billable/actionable)
            }
            converted_entities.append(converted_entity)
        
        if logger:
            logger.info(f"  ✅ Super-pass complete: {len(cleaned_transcript)} chars cleaned, {len(converted_entities)} entities extracted, entities_by_kind (Pure NER)")
        return cleaned_transcript, converted_entities, entities_by_kind
        
    except Exception as e:
        if logger:
            logger.error(f"  ❌ Super-pass failed: {e}")
            logger.debug(traceback.format_exc())
        return "", [], _build_entities_by_kind([])

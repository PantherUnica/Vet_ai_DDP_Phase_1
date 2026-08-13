"""
NER extraction from transcripts and SOAP sections.

This module handles:
- Entity extraction from SOAP sections
- Entity extraction from cleaned transcripts
- Robust JSON parsing with repair retry
- NER prompt definitions
"""

import os
import json
import re
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

from kb_ner_clients import get_openai_client

# Unified NER prompt (replaces all previous NER/Brain NER prompts for entity extraction).
# Same as CLINICAL_ENTITY_EXTRACTION_PROMPT in kb_ner_super_pass.py; kept here to avoid circular import.
UNIFIED_NER_PROMPT = """You extract clinical entity mentions from a veterinary pet–vet transcript.

CRITICAL: Return ONLY valid JSON. Do NOT include any reasoning, thinking, or explanatory text.
Do NOT use <think> tags, <reasoning> tags, markdown, or code fences.
The first non-whitespace character MUST be '{'.
Return ONLY: {"entities": [ ... ]}

INPUT:
You will receive a single field:
- TRANSCRIPT: free-form, noisy, may contain ASR errors and filler words ("like", "you know", "uh").

TASK:
Extract a flat list of ALL clinical entity mentions from the ENTIRE transcript.
Do NOT output SOAP sections, summaries, plans, or prose.

ENTITY KINDS (EXACTLY one of these 11):
ReasonForVisit, Medication, Procedure, Diagnostic, VitalSign, Reminder, Symptom, Diagnosis, Anatomy, Diet, ParasiteControl

OUTPUT JSON SCHEMA:
{
  "entities": [
    {
      "span_text": string,
      "normalized_name": string,
      "kind": string,
      "roles": [string],
      "context_sentence": string,
      "confidence": number
    }
  ]
}

HARD CONSTRAINTS:
1) Return ONLY valid JSON with top-level key "entities".
2) Do not invent entities not in the transcript.
3) span_text MUST be an exact phrase copied from the transcript (contiguous substring).
4) normalized_name does NOT need to match span_text (normalization is allowed and required).
5) Each field max 200 characters.
6) Maximum entities = 60.

SPAN_TEXT RULES: span_text must be the SHORTEST possible exact substring that identifies the entity. Typical span length: 1–8 words.

NORMALIZED_NAME RULES: Remove fillers; you MAY reorder into a noun phrase. Do NOT add new clinical meaning.

EXTRACTION MUST BE INCLUSIVE (NO UNDER-EXTRACTION). Your job is to EXTRACT, not validate.

MULTI-PASS PROCESS (MANDATORY): PASS 1 — HARVEST all candidates; PASS 2 — CONVERT to entity objects; PASS 3 — COVERAGE AUDIT.

MINIMUM ENTITY COUNT CHECK: If the transcript contains multiple clinical topics and you output fewer than 15 entities, redo PASS 1–3.

The transcript will be provided in the next user message.
"""

# All NER prompts now use the unified prompt.
NER_SYSTEM_PROMPT = UNIFIED_NER_PROMPT
TRANSCRIPT_NER_SYSTEM_PROMPT_STRICT = UNIFIED_NER_PROMPT
TRANSCRIPT_NER_SYSTEM_PROMPT = UNIFIED_NER_PROMPT


def _call_mistral_chat_json(model_name: str, user_content: str, system_prompt: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Call Mistral's chat completions API directly and return parsed JSON object."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        # Try current directory first, then parent directory
        api_key_file = Path(__file__).with_name("API_Key.txt")
        if not api_key_file.exists():
            api_key_file = api_key_file.parent.parent / "API_Key.txt"
        
        if api_key_file.exists():
            try:
                for line in api_key_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("MISTRAL_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
            except Exception:
                api_key = None
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not found (set env var or add MISTRAL_API_KEY=... to API_Key.txt).")

    url = "https://api.mistral.ai/v1/chat/completions"
    if logger:
        logger.info(f"🌐 Routing to Mistral API: model={model_name} url={url}")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 2000,  # Increased for Phase A
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    # Better error handling with detailed logging
    if r.status_code != 200:
        error_detail = ""
        try:
            error_response = r.json()
            error_detail = f" Response: {error_response}"
            if logger:
                logger.error(f"❌ Mistral API error {r.status_code} for model '{model_name}': {error_detail}")
                logger.error(f"   Request URL: {url}")
                logger.error(f"   Request payload model: {payload.get('model')}")
                logger.error(f"   Request payload keys: {list(payload.keys())}")
                # Log the actual error message from Mistral
                if isinstance(error_response, dict):
                    error_msg = error_response.get('error', {})
                    if isinstance(error_msg, dict):
                        logger.error(f"   Mistral error message: {error_msg.get('message', 'N/A')}")
                        logger.error(f"   Mistral error type: {error_msg.get('type', 'N/A')}")
        except Exception as e:
            error_detail = f" Response text: {r.text[:500]}"
            if logger:
                logger.error(f"❌ Mistral API error {r.status_code} for model '{model_name}': {error_detail}")
                logger.error(f"   Exception parsing error: {e}")
        r.raise_for_status()
    
    result = r.json()
    content = ""
    try:
        choice0 = (result.get("choices") or [])[0]
        msg = choice0.get("message") or {}
        content = msg.get("content") or ""
    except Exception:
        content = ""

    data = _extract_json_object(content, retry_count=0) or {}
    if not data:
        if logger:
            preview = (content or "").strip().replace("\n", " ")[:220]
            logger.warning(f"Mistral API returned empty/unparseable output (preview='{preview}') - attempting parse-repair retry")
        
        repair_prompt = f"""The previous response was not valid JSON. Return ONLY valid JSON. No markdown. No code fences. Same schema.

Previous invalid response:
{preview}

Return ONLY valid JSON."""
        
        try:
            repair_payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "Return ONLY valid JSON. No markdown. No code fences. No explanations."},
                    {"role": "user", "content": repair_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 2000,
            }
            r2 = requests.post(url, headers=headers, json=repair_payload, timeout=60)
            r2.raise_for_status()
            result2 = r2.json()
            content2 = ""
            try:
                choice0 = (result2.get("choices") or [])[0]
                msg = choice0.get("message") or {}
                content2 = msg.get("content") or ""
            except Exception:
                content2 = ""
            
            data = _extract_json_object(content2, retry_count=1) or {}
            if data and logger:
                logger.info("✅ Parse-repair retry succeeded")
        except Exception as e:
            if logger:
                logger.warning(f"Parse-repair retry failed: {e}")
    
    return data


def _find_root_json_object_end(s: str) -> int:
    """
    Find the index of the closing '}' that matches the first '{' in s,
    respecting string boundaries (do not count brackets inside quoted strings).
    Returns -1 if no root object or unclosed.
    Used to extract the root JSON object when content may contain '}' inside string values.
    """
    if not s:
        return -1
    start = s.find("{")
    if start < 0:
        return -1
    stack: List[str] = []
    i = start
    n = len(s)
    in_string = False
    escape = False
    quote_char = None
    while i < n:
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            i += 1
            continue
        if in_string:
            if c == quote_char:
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            quote_char = c
            i += 1
            continue
        if c == "{":
            stack.append("}")
            i += 1
            continue
        if c == "[":
            stack.append("]")
            i += 1
            continue
        if c in "}]" and stack:
            stack.pop()
            if not stack:
                return i
        i += 1
    return -1


def _json_close_truncated(s: str) -> str:
    """
    Build a closing suffix for truncated JSON by tracking structural [ and {
    (ignoring brackets inside strings). Close in reverse order of opening.
    Used when the LLM response is cut off (e.g. finish_reason=length).
    """
    if not s or not s.strip():
        return ""
    stack: List[str] = []
    i = 0
    n = len(s)
    in_string = False
    escape = False
    quote_char = None
    while i < n:
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            i += 1
            continue
        if in_string:
            if c == quote_char:
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            quote_char = c
            i += 1
            continue
        if c == "{":
            stack.append("}")
            i += 1
            continue
        if c == "[":
            stack.append("]")
            i += 1
            continue
        if c in "}]" and stack:
            stack.pop()
        i += 1
    return "".join(reversed(stack))


def _extract_json_object(text: str, retry_count: int = 0) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from LLM output with parse-repair retry.
    Handles:
    - code fences ```json ... ```
    - leading/trailing prose
    - whitespace
    - truncation (tries to repair with stack-based ] and } closing)
    Returns {} if nothing parseable after retries.
    """
    if not isinstance(text, str):
        return {}
    s = text.strip()
    if not s:
        return {}
    
    # Remove <think>...</think> blocks (some models emit these)
    s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE).strip()
    # If it starts with <think> without a closing tag (truncated), try to drop everything up to first '{'
    if s.lower().startswith("<think>"):
        brace = s.find("{")
        if brace >= 0:
            s = s[brace:].strip()
        else:
            return {}
    
    # CRITICAL: Remove any reasoning text before JSON (Qwen models sometimes include thinking tokens)
    # Look for patterns like "Okay, let's tackle" or "First, I need" that indicate reasoning
    # Find the first '{' and remove everything before it if there's reasoning-like text
    reasoning_patterns = [
        r"^[^{]*?(?:okay|let's|first|i need|i should|let me|thinking|reasoning|tackle|understand|make sure|the user|the instructions)[^{]*?(\{)",
        r"^[^{]*?<think>[\s\S]*?(\{)",
        r"^[^{]*?<reasoning>[\s\S]*?(\{)",
        r"^[^{]*?<think>[\s\S]*?(\{)",
    ]
    for pattern in reasoning_patterns:
        match = re.search(pattern, s, re.IGNORECASE | re.DOTALL)
        if match:
            brace_pos = match.start(1)
            s = s[brace_pos:].strip()
            break
    
    # Also remove any text that looks like reasoning before the first '{'
    first_brace = s.find("{")
    if first_brace > 20:  # If there's text before the first brace, check if it's reasoning
        # Check if it looks like reasoning (contains common reasoning words)
        prefix = s[:first_brace].lower()
        reasoning_indicators = [
            "okay", "let's", "first", "i need", "i should", "let me", "thinking", "reasoning", 
            "tackle", "understand", "make sure", "the user wants", "the instructions say",
            "i'll", "i will", "i can", "i must", "i have", "i'm going", "here's", "here is"
        ]
        if any(indicator in prefix for indicator in reasoning_indicators):
            s = s[first_brace:].strip()
    
    # Remove code fences (CRITICAL: strip before parsing)
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    
    # Fast path: direct JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError as e:
        # Try to repair truncation: use stack-based closing (] and } in reverse order of opening)
        if retry_count < 2:
            suffix = _json_close_truncated(s)
            if suffix:
                try:
                    obj = json.loads(s + suffix)
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    pass
        
        # Substring: root JSON object from first { to its matching } (respect strings).
        # BUG FIX: rfind("}") is wrong when JSON contains "}" inside string values (e.g. transcript text).
        start = s.find("{")
        end = _find_root_json_object_end(s)
        if end < 0:
            end = s.rfind("}")  # fallback for malformed / no clear root
        if start >= 0 and end > start:
            candidate = s[start:end + 1]
            try:
                obj = json.loads(candidate)
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                if retry_count < 2:
                    suffix = _json_close_truncated(candidate)
                    if suffix:
                        try:
                            obj = json.loads(candidate + suffix)
                            return obj if isinstance(obj, dict) else {}
                        except Exception:
                            pass
                return {}
    
    return {}


def ner_extract_entities(
    text: str,
    section_type: str,
    species: Optional[str] = None,
    model: str = "gpt-4.1-nano",
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Run LLM NER on a SOAP section.

    Returns a list of entities:
      { span_text, normalized_name, kind, attributes }
    """
    if not text or not text.strip():
        return []
    
    if not client:
        client = get_openai_client()
        if not client:
            if logger:
                logger.error("OpenAI client not available for NER")
            return []
    
    if logger:
        logger.info(f"Running NER on {section_type} section ({len(text)} chars)")

    user_content = f"""Species: {species or "Not specified"}
Section Type: {section_type}
Text: {text}"""

    try:
        api_model_name = model
        if "-2025-04-14" in api_model_name:
            api_model_name = api_model_name.replace("-2025-04-14", "")
        resp = client.chat.completions.create(
            model=api_model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": NER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            if logger:
                logger.warning(f"Failed to parse NER JSON response: {e}")
            data = {}
    except Exception as e:
        if logger:
            logger.error(f"Error in NER extraction: {e}")
        return []

    ents = data.get("entities", [])
    if not isinstance(ents, list):
        ents = []

    cleaned = []
    for e in ents:
        if not isinstance(e, dict):
            continue
        span_text = e.get("span_text")
        # CRITICAL FIX: normalized_name MUST equal span_text (no semantic normalization at NER time)
        normalized_name = span_text  # Force normalized_name = span_text
        kind = e.get("kind") or "Other"
        kb_kind = e.get("kb_kind") or kind  # Use kb_kind if available
        intent_kind = e.get("intent_kind")
        assertion_id = e.get("assertion_id", "CONF")  # Default to CONF if not specified
        attributes = e.get("attributes") or {}
        if not isinstance(span_text, str) or not span_text.strip():
            continue
        if not isinstance(attributes, dict):
            attributes = {}
        
        # Validate assertion_id
        valid_assertions = {"CONF", "NEG", "SUSP", "HIST", "HYPO", "RECUR"}
        if assertion_id not in valid_assertions:
            assertion_id = "CONF"  # Default to CONF if invalid

        cleaned.append(
            {
                "span_text": span_text.strip(),
                "normalized_name": normalized_name.strip(),  # Always equals span_text
                "kind": kind,  # DEPRECATED - kept for backward compatibility
                "kb_kind": kb_kind,  # Use this for routing
                "intent_kind": intent_kind,  # What speaker meant
                "assertion_id": assertion_id,
                "attributes": attributes,
            }
        )
    
    if logger:
        logger.info(f"Extracted {len(cleaned)} entities from {section_type} section")
    
    return cleaned


def extract_entities_from_cleaned_transcript(
    cleaned_transcript: str,
    model: str = "gpt-4.1-nano",
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Extract entities and signalment from a cleaned, attributed transcript.
    
    This runs BEFORE SOAP note generation to provide concept grounding.
    
    Returns:
        {
            "signalment": {...},
            "reasons": [...],  # Reason for Visit entities
            "conditions": [...],
            "symptoms": [...],
            "drugs": [...],
            "tests": [...],
            "procedures": [...],
            "vitals": [...],
            "organisms": [...],
            "diet": [...],
            "household": [...],
            "other": [...]
        }
    """
    if not cleaned_transcript or not cleaned_transcript.strip():
        return {
            "signalment": {},
            "reasons": [],
            "conditions": [],
            "symptoms": [],
            "drugs": [],
            "tests": [],
            "procedures": [],
            "vitals": [],
            "organisms": [],
            "diet": [],
            "household": [],
            "other": []
        }
    
    if not client:
        client = get_openai_client()
        if not client:
            if logger:
                logger.error("OpenAI client not available for transcript NER")
            return {
                "signalment": {},
                "reasons": [],
                "conditions": [],
                "symptoms": [],
                "drugs": [],
                "tests": [],
                "procedures": [],
                "vitals": [],
                "organisms": [],
                "diet": [],
                "household": [],
                "other": []
            }
    
    if logger:
        logger.info(f"Running transcript-level NER extraction ({len(cleaned_transcript)} chars)")

    user_content = f"""Cleaned Transcript:
{cleaned_transcript}

Extract clinical entity mentions from the transcript above. Return ONLY a JSON object with "entities" array. No signalment, no attributes, no nested schemas."""

    try:
        if ("ministral" in model.lower() or "mistral" in model.lower()) and not model.startswith("accounts/fireworks/"):
            data = _call_mistral_chat_json(model, user_content, TRANSCRIPT_NER_SYSTEM_PROMPT_STRICT, logger)
        else:
            if logger:
                logger.info(f"🌐 Routing transcript NER to OpenAI-compatible endpoint (Fireworks/OpenAI): model={model}")
                # Try function calling for JSON format enforcement (more reliable than response_format)
                # Function calling forces structured output at API level, preventing reasoning tokens
                use_function_calling = False
                if "qwen" in model.lower() or "fireworks" in str(type(client)).lower():
                    try:
                        # Define the function schema for entity extraction
                        extract_entities_function = {
                            "type": "function",
                            "function": {
                                "name": "extract_entities",
                                "description": "Extract clinical entity mentions from veterinary transcript",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "entities": {
                                            "type": "array",
                                            "description": "List of extracted clinical entities",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "span_text": {"type": "string", "description": "Exact text as it appears in transcript"},
                                                    "normalized_name": {"type": "string", "description": "Normalized name (must equal span_text)"},
                                                    "kind": {"type": "string", "enum": ["ReasonForVisit", "Medication", "Procedure", "Diagnostic", "VitalSign", "Reminder", "Symptom", "Diagnosis", "Anatomy", "Diet", "ParasiteControl"]},
                                                    "roles": {"type": "array", "items": {"type": "string"}, "description": "Optional roles"},
                                                    "context_sentence": {"type": "string", "description": "Optional surrounding sentence"},
                                                    "confidence": {"type": "number", "description": "Confidence score 0.0-1.0"}
                                                },
                                                "required": ["span_text", "normalized_name", "kind"]
                                            }
                                        }
                                    },
                                    "required": ["entities"]
                                }
                            }
                        }
                        
                        request_params = {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": TRANSCRIPT_NER_SYSTEM_PROMPT_STRICT if ("qwen3-8b" in model.lower() or "ministral" in model.lower() or "mistral" in model.lower()) else TRANSCRIPT_NER_SYSTEM_PROMPT},
                                {"role": "user", "content": user_content},
                            ],
                            "tools": [extract_entities_function],
                            "tool_choice": {"type": "function", "function": {"name": "extract_entities"}},  # Force function call
                            "temperature": 0.0,
                            "seed": 42,
                            "max_tokens": 800,
                        }
                        use_function_calling = True
                        if logger:
                            logger.debug(f"  🔧 Attempting function calling for JSON format enforcement (prevents reasoning tokens)")
                    except Exception as e:
                        # Function calling not supported or failed, fall back to response_format
                        if logger:
                            logger.debug(f"  ⚠️  Function calling not available: {e} - falling back to response_format")
                        use_function_calling = False
                
                if not use_function_calling:
                    # CRITICAL FIX: Map internal model name to actual API model name
                    # Use model name directly - gpt-4.1-mini is a valid OpenAI model
                    # If model is "gpt-4.1-mini", use dated version "gpt-4.1-mini-2025-04-14"
                    api_model_name = model
                    # Remove date suffix if present (API doesn't accept it)
                    if "-2025-04-14" in api_model_name:
                        api_model_name = api_model_name.replace("-2025-04-14", "")
                    
                    # Fallback: Use response_format (OpenAI-compatible)
                    request_params = {
                        "model": api_model_name,  # Use model name directly (gpt-4.1-mini-2025-04-14)
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": TRANSCRIPT_NER_SYSTEM_PROMPT_STRICT if ("qwen3-8b" in model.lower() or "ministral" in model.lower() or "mistral" in model.lower()) else TRANSCRIPT_NER_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0.0,
                        "seed": 42,  # Mandatory for determinism - forces same expert path
                        "max_tokens": 800,
                    }
                    if "gpt-oss-120b" in model.lower():
                        request_params["reasoning_effort"] = "low"
                    # For Qwen models, try to suppress reasoning tokens if supported
                    if "qwen" in model.lower():
                        # Some Qwen models support this parameter to suppress reasoning
                        # If the API doesn't support it, it will be ignored
                        try:
                            request_params["stop_reasoning"] = True
                        except:
                            pass  # Parameter not supported, continue without it
                
                resp = client.chat.completions.create(**request_params)
                
                # Extract response based on whether function calling was used
                if use_function_calling:
                    # Function calling response: extract from tool_calls
                    if resp.choices and resp.choices[0].message.tool_calls:
                        tool_call = resp.choices[0].message.tool_calls[0]
                        if tool_call.function.name == "extract_entities":
                            raw = tool_call.function.arguments or "{}"
                        else:
                            raw = "{}"
                    else:
                        # Fallback if no tool_calls (shouldn't happen with tool_choice forced)
                        raw = resp.choices[0].message.content if resp.choices else ""
                        if logger:
                            logger.warning(f"  ⚠️  Function calling enabled but no tool_calls returned - using content instead")
                else:
                    # Standard response_format response
                    raw = resp.choices[0].message.content if resp.choices else ""
                
                if raw is None:
                    raw = ""
            data = _extract_json_object(raw, retry_count=0) or {}
            if not data:
                if logger:
                    preview = (raw or "").strip().replace("\n", " ")[:220]
                    logger.warning(f"Failed to parse transcript NER JSON response: empty/unparseable output (preview='{preview}') - attempting parse-repair retry")
                
                repair_prompt = f"""The previous response was not valid JSON. Return ONLY valid JSON. No markdown. No code fences. Same schema.

Previous invalid response:
{preview}

Return ONLY: {{"entities": [...]}}"""
                
                try:
                    # CRITICAL FIX: Map internal model name to actual API model name
                    try:
                        from kb_ner_clients import get_api_model_name
                        api_model_name = get_api_model_name(model)
                    except ImportError:
                        api_model_name = model  # Fallback if function not available
                    
                    repair_params = {
                        "model": api_model_name,  # Use actual API model name, not internal alias
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": "Return ONLY valid JSON. No markdown. No code fences. No explanations."},
                            {"role": "user", "content": repair_prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 800,
                    }
                    resp_repair = client.chat.completions.create(**repair_params)
                    raw_repair = resp_repair.choices[0].message.content if resp_repair.choices else ""
                    if raw_repair:
                        data = _extract_json_object(raw_repair, retry_count=1) or {}
                        if data and logger:
                            logger.info("✅ Parse-repair retry succeeded")
                except Exception as e:
                    if logger:
                        logger.warning(f"Parse-repair retry failed: {e}")

                if ("qwen3-8b" in model.lower() or "ministral" in model.lower() or "mistral" in model.lower()):
                    fallback_model = "accounts/fireworks/models/llama-v3p3-70b-instruct"
                    if logger:
                        logger.warning(f"Retrying transcript NER with fallback model: {fallback_model}")
                    try:
                        resp2 = client.chat.completions.create(
                            model=fallback_model,
                            response_format={"type": "json_object"},
                            messages=[
                                {"role": "system", "content": TRANSCRIPT_NER_SYSTEM_PROMPT_STRICT},
                                {"role": "user", "content": user_content},
                            ],
                            temperature=0.0,
                            max_tokens=800,
                        )
                        raw2 = resp2.choices[0].message.content if resp2.choices else ""
                        if raw2 is None:
                            raw2 = ""
                        data = _extract_json_object(raw2, retry_count=0) or {}
                    except Exception as e:
                        if logger:
                            logger.warning(f"Fallback transcript NER failed: {e}")
                if not data:
                    data = {}
    except Exception as e:
        if logger:
            logger.error(f"Error in transcript NER extraction: {e}")
        return {
            "signalment": {},
            "reasons": [],
            "conditions": [],
            "symptoms": [],
            "drugs": [],
            "tests": [],
            "procedures": [],
            "vitals": [],
            "organisms": [],
            "diet": [],
            "household": [],
            "other": []
        }

    # Convert minimal schema {"entities": [...]} to expected format if needed
    if "entities" in data and isinstance(data["entities"], list):
        entities = data["entities"]
        converted = {
            "signalment": {},
            "reasons": [],
            "conditions": [],
            "symptoms": [],
            "drugs": [],
            "tests": [],
            "procedures": [],
            "vitals": [],
            "organisms": [],
            "diet": [],
            "household": [],
            "other": []
        }
        
        # Comprehensive kind mapping
        kind_mapping = {
            "ReasonForVisit": "reasons",
            "Reason": "reasons",
            "Reason_for_Visit": "reasons",
            "Procedure": "procedures",
            "Service": "procedures",
            "Drug": "drugs",
            "Drug/Substance": "drugs",
            "Medication": "drugs",
            "Substance": "drugs",
            "VitalSign": "vitals",
            "Vital_Sign": "vitals",
            "Vitals": "vitals",
            "Condition": "conditions",
            "Diagnosis": "conditions",
            "Disease": "conditions",
            "Finding": "symptoms",
            "Symptom": "symptoms",
            "Observation": "symptoms",
            "Test": "tests",
            "DiagnosticTest": "tests",
            "LabTest": "tests",
            "Organism": "organisms",
            "Parasite": "organisms",
            "Diet": "diet",
            "Nutrition": "diet",
            "Food": "diet",
            "Household": "household",
            "Anatomy": "other",
            "BodyPart": "other",
            "Other": "other",
        }
        
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            
            span_text = entity.get("span_text") or entity.get("surface_text") or ""
            normalized_name = entity.get("normalized_name") or span_text
            kind = entity.get("kind") or "Other"
            roles = entity.get("roles") or []
            context_sentence = entity.get("context_sentence") or ""
            confidence = entity.get("confidence", 0.7)
            
            normalized_kind = kind.strip() if isinstance(kind, str) else "Other"
            category = kind_mapping.get(normalized_kind, "other")
            
            # Fallback: try canonicalizing the kind if direct lookup fails
            if category == "other" and normalized_kind:
                kind_lower = normalized_kind.lower()
                if "reason" in kind_lower or "visit" in kind_lower:
                    category = "reasons"
                elif "procedure" in kind_lower or "service" in kind_lower:
                    category = "procedures"
                elif "drug" in kind_lower or "medication" in kind_lower or "substance" in kind_lower:
                    category = "drugs"
                elif "vital" in kind_lower:
                    category = "vitals"
                elif "condition" in kind_lower or "disease" in kind_lower or "diagnosis" in kind_lower:
                    category = "conditions"
                elif "finding" in kind_lower or "symptom" in kind_lower or "observation" in kind_lower:
                    category = "symptoms"
                elif "test" in kind_lower or "lab" in kind_lower:
                    category = "tests"
                elif "organism" in kind_lower or "parasite" in kind_lower:
                    category = "organisms"
                elif "diet" in kind_lower or "nutrition" in kind_lower or "food" in kind_lower:
                    category = "diet"
                elif "household" in kind_lower:
                    category = "household"
            
            converted_entity = {
                "span_text": span_text,
                "surface_text": span_text,
                "normalized_name": normalized_name,
                "kind": kind,
                "intent_kind": kind,
                "kb_kind": kind,
                "assertion_id": entity.get("assertion_id", "CONF"),
                "attributes": entity.get("attributes", {}),
                "roles": roles,
                "context_sentence": context_sentence,
                "confidence": confidence,
                "search_query": entity.get("search_query", span_text),
            }
            
            converted[category].append(converted_entity)
        
        data = converted
    
    signalment = data.get("signalment", {})
    if not isinstance(signalment, dict):
        signalment = {}
    
    return {
        "signalment": signalment,
        "reasons": data.get("reasons", []),
        "conditions": data.get("conditions", []),
        "symptoms": data.get("symptoms", []),
        "drugs": data.get("drugs", []),
        "tests": data.get("tests", []),
        "procedures": data.get("procedures", []),
        "vitals": data.get("vitals", []),
        "organisms": data.get("organisms", []),
        "diet": data.get("diet", []),
        "household": data.get("household", []),
        "other": data.get("other", [])
    }

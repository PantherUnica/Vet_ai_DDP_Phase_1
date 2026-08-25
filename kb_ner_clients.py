"""
Client setup for OpenAI and Mistral APIs.

This module handles:
- OpenAI client initialization
- Embedding client resolution
- API key management
- Model-to-provider routing (deterministic client selection)
"""

import os
import logging
import requests
from typing import Optional, Tuple, Any, Dict, List
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai package not available. KB linking will be disabled.")

# Model Name Alias Mapping
# Maps internal model names to actual API model names
# Note: gpt-4.1-mini is a valid OpenAI model - no mapping needed
# If using dated version, use "gpt-4.1-mini-2025-04-14" directly
MODEL_NAME_ALIAS: Dict[str, str] = {
    # No mappings needed - gpt-4.1-mini is a valid OpenAI model name
    # If you need to use the dated version, specify "gpt-4.1-mini-2025-04-14" directly
}

# Model-to-Provider Configuration
# This determines which API endpoint/provider to use for each model
MODEL_PROVIDER_MAP: Dict[str, str] = {
    # Mistral API models
    "mistral-small-2506": "mistral",
    "mistral-medium-2508": "mistral",
    "mistral-large-2512": "mistral",
    "pixtral-12b-2409": "mistral",
    "pixtral-large-2409": "mistral",
    # OpenAI API models
    "gpt-4.1-mini": "openai",
    "gpt-4.1-nano": "openai",
    "gpt-4.1": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4o": "openai",
    "gpt-4": "openai",
    "gpt-3.5-turbo": "openai",
    # Fireworks API models (Llama, Qwen, DeepSeek, etc.)
    "accounts/fireworks/models/llama-v3-70b-instruct": "fireworks",
    "accounts/fireworks/models/llama-v3p3-70b-instruct": "fireworks",
    "accounts/fireworks/models/deepseek-v3p2": "fireworks",
    "accounts/fireworks/models/gpt-oss-20b": "fireworks",
    "accounts/fireworks/models/gpt-oss-120b": "fireworks",
    "accounts/fireworks/models/qwen3-8b": "fireworks",
    "whisper-v3-turbo": "fireworks",
    "nova-3": "deepgram",
    "nova-2": "deepgram",
}

# Phase-to-Model Configuration
# Centralized configuration for each phase in the SOAP note pipeline
PHASE_CONFIG: Dict[str, Dict[str, str]] = {
    "step_1_transcription": {
        "phase": "Audio Transcription",
        "model": "whisper-v3-turbo",
        "provider": "fireworks",  # Fireworks Whisper API
        "description": "Speech-to-text transcription from audio"
    },
    "step_2_cleaning": {
        "phase": "Transcription Cleaning",
        "model": "gpt-4.1-nano",
        "provider": "openai",
        "description": "Multi-language translation, de-noising, clinical refinement"
    },
    "step_2_3_phase_a": {
        "phase": "Phase A: Mention Extraction",
        "model": "gpt-4.1-nano",
        "provider": "openai",
        "description": "Fast entity mention extraction (spans + coarse kind)"
    },
    "step_2_3_phase_b": {
        "phase": "Phase B: Entity Enrichment",
        "model": "gpt-4.1-nano",
        "provider": "openai",
        "description": "Enrich entities with attributes, roles, and evidence"
    },
    "step_2_3_normalization": {
        "phase": "Entity Normalization & Linking",
        "model": "gpt-4.1-nano",
        "provider": "openai",
        "description": "ASR correction, KB linking, disambiguation"
    },
    "step_3_soap_generation": {
        "phase": "SOAP Note Generation",
        "model": "gpt-4.1-nano",
        "provider": "openai",
        "description": "Generate structured SOAP note from transcript + entities"
    }
}

def get_phase_config(phase_key: str) -> Dict[str, str]:
    """
    Get configuration for a specific phase.
    
    Args:
        phase_key: One of: "step_1_transcription", "step_2_cleaning", 
                   "step_2_3_phase_a", "step_2_3_phase_b", 
                   "step_2_3_normalization", "step_3_soap_generation"
    
    Returns:
        Dictionary with: phase, model, provider, description
    
    Example:
        config = get_phase_config("step_2_3_phase_a")
        # Returns: {
        #     "phase": "Phase A: Mention Extraction",
        #     "model": "mistral-small-2506",
        #     "provider": "mistral",
        #     "description": "Fast entity mention extraction..."
        # }
    """
    if phase_key == "step_1_transcription":
        return get_step1_transcription_config()
    return PHASE_CONFIG.get(phase_key, {})


def get_step1_transcription_config() -> Dict[str, str]:
    """Runtime Step-1 config from ASR_PROVIDER / ASR_MODEL env."""
    from asr_providers import resolve_model, resolve_provider

    provider = resolve_provider()
    model = resolve_model(provider)
    base = dict(PHASE_CONFIG.get("step_1_transcription", {}))
    base["provider"] = provider
    base["model"] = model
    base["description"] = (
        f"Speech-to-text via {provider} ({model}); override with ASR_PROVIDER / ASR_MODEL"
    )
    return base

def get_all_phase_configs() -> Dict[str, Dict[str, str]]:
    """
    Get all phase configurations.
    
    Returns:
        Dictionary mapping phase keys to their configurations
    """
    return PHASE_CONFIG.copy()


def get_api_model_name(model_name: str) -> str:
    """
    Get the actual API model name from an internal model name (handles aliases).
    
    Args:
        model_name: Internal model identifier (e.g., "gpt-4.1-mini")
    
    Returns:
        Actual API model name (e.g., "gpt-4.1-mini-2025-04-14" or original if no alias)
    """
    # Check if there's an alias mapping
    if model_name in MODEL_NAME_ALIAS:
        return MODEL_NAME_ALIAS[model_name]
    # No alias - return as-is (gpt-4.1-mini is a valid OpenAI model)
    return model_name


def get_model_provider(model_name: str) -> str:
    """
    Determine which API provider to use for a given model name.
    
    Args:
        model_name: The model identifier (e.g., "gpt-4.1-mini", "accounts/fireworks/models/qwen3-8b")
    
    Returns:
        Provider name: "openai", "mistral", or "fireworks"
    
    Logic:
        1. Exact match in MODEL_PROVIDER_MAP
        2. Pattern matching (mistral* → mistral, gpt-* → openai, accounts/fireworks/* → fireworks)
        3. Default: "openai" (fallback)
    """
    model_lower = model_name.lower()
    
    # 1. Exact match
    if model_name in MODEL_PROVIDER_MAP:
        return MODEL_PROVIDER_MAP[model_name]
    
    # 2. Pattern matching
    if "mistral" in model_lower or "ministral" in model_lower or "pixtral" in model_lower:
        return "mistral"
    elif model_lower.startswith("gpt-") or "gpt" in model_lower:
        return "openai"
    elif "fireworks" in model_lower or model_lower.startswith("accounts/"):
        return "fireworks"
    
    # 3. Default fallback
    return "openai"


_KEY_ALIASES: Dict[str, str] = {
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "OPENAI_KEY": "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY": "ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY": "MISTRAL_API_KEY",
    "FIREWORKS_API_KEY": "FIREWORKS_API_KEY",
    "FIREWORKS_API": "FIREWORKS_API_KEY",
}


def _normalize_key_name(key_name: str) -> Optional[str]:
    norm = (key_name or "").strip().replace("-", "_").upper()
    return _KEY_ALIASES.get(norm)


def _candidate_key_files() -> List[Path]:
    """
    Resolve API key files in priority order:
    experiment root -> phase root -> current module dir.
    """
    here = Path(__file__).resolve().parent
    phase_root = here.parent
    experiment_root = phase_root.parent
    files = [
        experiment_root / "API_Key.txt",
        phase_root / "API_Key.txt",
        here / "API_Key.txt",
        experiment_root / "fireworks_api.txt",
        phase_root / "fireworks_api.txt",
        here / "fireworks_api.txt",
    ]
    return files


def _parse_key_file(path: Path) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                raw_key, raw_value = line.split("=", 1)
                normalized = _normalize_key_name(raw_key)
                value = raw_value.strip().strip("\"'")
                if normalized and value:
                    parsed[normalized] = value
            elif line.startswith("fw_"):
                # Accept standalone Fireworks key line.
                parsed["FIREWORKS_API_KEY"] = line
    except Exception:
        return {}
    return parsed


def _collect_api_keys() -> Dict[str, str]:
    """
    Build normalized key map from files + environment with env precedence.
    """
    keys: Dict[str, str] = {}
    for path in _candidate_key_files():
        if not path.exists():
            continue
        file_keys = _parse_key_file(path)
        for key_name, key_value in file_keys.items():
            if key_value and not keys.get(key_name):
                keys[key_name] = key_value

    # Environment overrides files.
    for env_name, env_value in os.environ.items():
        normalized = _normalize_key_name(env_name)
        if normalized:
            value = (env_value or "").strip()
            if value:
                keys[normalized] = value
    return keys


def get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """
    Get OpenAI client. If api_key is provided, use it; otherwise try env var or API_Key.txt.
    If base_url is provided, use it (e.g., for Fireworks AI compatibility).
    
    Args:
        api_key: Optional API key (if not provided, tries OPENAI_API_KEY env var, then API_Key.txt)
        base_url: Optional base URL (e.g., for Fireworks AI compatibility)
    
    Returns:
        OpenAI client instance or None if not available
    """
    if not OPENAI_AVAILABLE:
        return None
    
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    
    if client_kwargs:
        return OpenAI(**client_kwargs)
    
    openai_key = (_collect_api_keys().get("OPENAI_API_KEY") or "").strip()
    
    if openai_key and not openai_key.startswith("fw_"):
        # CRITICAL: Explicitly set base_url to OpenAI endpoint to prevent Fireworks
        # If base_url was explicitly provided, use it; otherwise force OpenAI endpoint
        if base_url is not None:
            explicit_base_url = base_url
        else:
            # Force OpenAI endpoint - never use Fireworks for OpenAI models
            explicit_base_url = "https://api.openai.com/v1"
        return OpenAI(api_key=openai_key, base_url=explicit_base_url)
    
    # Final fallback: DO NOT create client without API key
    # CRITICAL: Never fallback to Fireworks or create client without OpenAI key
    # This prevents accidentally using Fireworks endpoint for OpenAI models
    return None  # Return None if no OpenAI key available


def _resolve_embedding_client(
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Resolve an OpenAI embedding client and its API key (if available).

    CRITICAL: Always uses OpenAI API for embeddings, never Fireworks.
    
    Args:
        logger: Optional logger for debugging
    
    Returns:
        Tuple of (embedding_client, openai_api_key)
    """
    embedding_client = None
    openai_api_key = None
    
    # Try to get OpenAI API key from normalized key map.
    openai_api_key = (_collect_api_keys().get("OPENAI_API_KEY") or "").strip()
    
    # Verify it's an OpenAI key (not Fireworks key)
    if openai_api_key and not openai_api_key.startswith("fw_"):
        embedding_client = OpenAI(api_key=openai_api_key, base_url=None)  # Explicitly use OpenAI endpoint
        if logger:
            logger.debug(f"✅ Created OpenAI embedding client (key starts with: {openai_api_key[:10]}...)")
    else:
        if logger and openai_api_key.startswith("fw_"):
            logger.warning("⚠️ OPENAI_API_KEY appears to be a Fireworks key; embeddings disabled.")
    
    # Final fallback: Use default OpenAI client (from environment)
    # CRITICAL: Only create if we have a valid OpenAI key
    if not embedding_client:
        openai_api_key = (_collect_api_keys().get("OPENAI_API_KEY") or "").strip()
        if openai_api_key and not openai_api_key.startswith("fw_"):
            # Create with explicit OpenAI endpoint
            embedding_client = OpenAI(api_key=openai_api_key, base_url="https://api.openai.com/v1")
        # If no valid OpenAI key, embedding_client remains None
    
    return embedding_client, openai_api_key


def get_client_for_model(
    model_name: str,
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[Any], str]:
    """
    Get the appropriate client for a given model name (deterministic routing).
    
    This function centralizes client selection based on model-to-provider mapping.
    Each phase can call this function to get the correct client for their model.
    
    Args:
        model_name: The model identifier (e.g., "gpt-4.1-mini", "accounts/fireworks/models/qwen3-8b")
        logger: Optional logger for debugging
    
    Returns:
        Tuple of (client, provider_name)
        - client: The initialized client (OpenAI client for OpenAI/Fireworks, None for Mistral)
        - provider_name: "openai", "mistral", or "fireworks"
    
    Example:
        client, provider = get_client_for_model("gpt-4.1-mini")
        # Returns OpenAI client with OpenAI endpoint
        
        client, provider = get_client_for_model("accounts/fireworks/models/qwen3-8b")
        # Returns None (Mistral uses direct HTTP calls, not OpenAI client)
    """
    provider = get_model_provider(model_name)
    
    if logger:
        logger.debug(f"🔍 Model '{model_name}' → Provider: {provider}")
    
    if provider == "mistral":
        # Mistral API uses direct HTTP calls (not OpenAI client)
        # Return None - the calling code should use _call_mistral_chat_json
        return None, "mistral"
    
    elif provider == "openai":
        # OpenAI models need OpenAI endpoint (not Fireworks)
        openai_key = (_collect_api_keys().get("OPENAI_API_KEY") or "").strip()
        if openai_key.startswith("fw_"):
            if logger:
                logger.warning("⚠️ OpenAI API key starts with 'fw_' (Fireworks key), skipping")
            openai_key = ""
        
        if openai_key and not openai_key.startswith("fw_"):
            # CRITICAL: Force OpenAI endpoint - never use Fireworks for OpenAI models
            # Create client directly with explicit OpenAI endpoint to prevent any Fireworks interference
            if not OPENAI_AVAILABLE:
                if logger:
                    logger.error(f"❌ OpenAI package not available for model '{model_name}'")
                return None, "openai"
            # Use the module-level OpenAI class
            from openai import OpenAI as OpenAIClient
            client = OpenAIClient(api_key=openai_key, base_url="https://api.openai.com/v1")
            if logger:
                logger.info(f"✅ Created OpenAI client for '{model_name}' (OpenAI endpoint, key: {openai_key[:10]}...)")
                if hasattr(client, 'base_url'):
                    logger.debug(f"   Client base_url: {client.base_url}")
            return client, "openai"
        else:
            # Fallback: Try to create OpenAI client without Fireworks endpoint
            if logger:
                logger.warning(f"⚠️  OpenAI API key not found or invalid for '{model_name}'. Attempting to create OpenAI client...")
            # Try to get OpenAI key from environment one more time
            env_key = (_collect_api_keys().get("OPENAI_API_KEY") or "").strip()
            if env_key and not env_key.startswith("fw_"):
                # CRITICAL: Explicitly set base_url to OpenAI endpoint
                if not OPENAI_AVAILABLE:
                    if logger:
                        logger.error(f"❌ OpenAI package not available for model '{model_name}'")
                    return None, "openai"
                # Use the module-level OpenAI class
                from openai import OpenAI as OpenAIClient
                client = OpenAIClient(api_key=env_key, base_url="https://api.openai.com/v1")
                if logger:
                    logger.info(f"✅ Created OpenAI client from environment for '{model_name}' (explicit OpenAI endpoint)")
                return client, "openai"
            else:
                # Last resort: Check if we have a Fireworks key and warn
                if logger:
                    if env_key and env_key.startswith("fw_"):
                        logger.error(f"❌ OPENAI_API_KEY is a Fireworks key (starts with 'fw_') for OpenAI model '{model_name}'")
                        logger.error(f"❌ Please set a valid OpenAI API key in OPENAI_API_KEY environment variable or API_Key.txt")
                    else:
                        logger.error(f"❌ No OpenAI API key found for model '{model_name}'")
                        logger.error(f"❌ Please set OPENAI_API_KEY environment variable or add it to API_Key.txt")
                return None, "openai"
    
    elif provider == "fireworks":
        # Fireworks models need Fireworks endpoint
        fireworks_key = (_collect_api_keys().get("FIREWORKS_API_KEY") or "").strip()
        
        if fireworks_key:
            # Fireworks uses OpenAI-compatible client with Fireworks endpoint
            from openai import OpenAI
            client = OpenAI(
                api_key=fireworks_key,
                base_url="https://api.fireworks.ai/inference/v1"
            )
            if logger:
                logger.debug(f"✅ Created Fireworks client for '{model_name}'")
            return client, "fireworks"
        else:
            if logger:
                logger.error(f"❌ Fireworks API key not found for '{model_name}'")
                logger.error(f"❌ Please set FIREWORKS_API_KEY environment variable or add it to API_Key.txt")
            return None, "fireworks"
    
    # Fallback: Unknown provider
    if logger:
        logger.warning(f"⚠️  Unknown provider for '{model_name}', returning None")
    return None, "openai"

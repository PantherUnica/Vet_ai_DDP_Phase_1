"""
Embedding generation utilities for KB NER linker.

This module handles:
- Single text embedding generation
- Batch embedding generation
- PostgreSQL vector literal conversion
"""

import time
import logging
import os
import threading
from collections import OrderedDict
from typing import List, Optional, Any, Tuple

from kb_ner_clients import _resolve_embedding_client, get_openai_client

_EMBED_CACHE_LOCK = threading.Lock()
_EMBED_CACHE: "OrderedDict[Tuple[str, str], List[float]]" = OrderedDict()


def _embed_cache_enabled() -> bool:
    return os.getenv("KB_EMBED_CACHE_ENABLE", "true").strip().lower() in ("1", "true", "yes")


def _embed_cache_max() -> int:
    try:
        v = int(os.getenv("KB_EMBED_CACHE_MAX", "2048"))
        return max(0, v)
    except Exception:
        return 2048


def _cache_get(key: Tuple[str, str]) -> Optional[List[float]]:
    if not _embed_cache_enabled():
        return None
    with _EMBED_CACHE_LOCK:
        v = _EMBED_CACHE.get(key)
        if v is None:
            return None
        # LRU bump
        _EMBED_CACHE.move_to_end(key, last=True)
        return v


def _cache_set(key: Tuple[str, str], vec: List[float]) -> None:
    if not _embed_cache_enabled():
        return
    maxn = _embed_cache_max()
    if maxn <= 0:
        return
    with _EMBED_CACHE_LOCK:
        _EMBED_CACHE[key] = vec
        _EMBED_CACHE.move_to_end(key, last=True)
        while len(_EMBED_CACHE) > maxn:
            _EMBED_CACHE.popitem(last=False)


def embed_text(
    text: str,
    model: str = "text-embedding-3-small",
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> List[float]:
    """
    Get a 1536-dim embedding vector for a piece of text.
    
    CRITICAL: Always uses OpenAI API for embeddings, even if a Fireworks client is passed.
    The client parameter is ignored - embeddings always use OpenAI API endpoint.
    
    Args:
        text: Text to embed
        model: Embedding model name (default: text-embedding-3-small)
        client: Ignored - always uses OpenAI API
        logger: Optional logger
    
    Returns:
        List of floats representing the embedding vector
    """
    text = text.strip()
    if not text:
        return []

    cache_key = (model, text)
    cached = _cache_get(cache_key)
    if cached is not None:
        return list(cached)

    # CRITICAL: Always create a dedicated OpenAI client for embeddings
    # Do NOT use the passed client (which may be a Fireworks client for LLM calls)
    embedding_client, openai_api_key = _resolve_embedding_client(logger=logger)
    
    if not embedding_client:
        if logger:
            logger.error("OpenAI client not available for embeddings")
        return []

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = embedding_client.embeddings.create(
                model=model,
                input=text,
            )
            embedding = resp.data[0].embedding
            try:
                _cache_set(cache_key, list(embedding))
            except Exception:
                pass

            return embedding
        except Exception as e:
            err_text = str(e).lower()
            retryable = any(
                token in err_text
                for token in [
                    "connection error",
                    "timeout",
                    "timed out",
                    "rate limit",
                    "server error",
                    "502",
                    "503",
                    "504",
                ]
            )
            if logger:
                logger.error(f"Error generating embedding: {e}")
                if hasattr(embedding_client, 'base_url'):
                    logger.debug(f"Embedding client base_url: {embedding_client.base_url}")
                logger.debug(f"Model requested: {model}")
            if attempt < max_retries - 1 and retryable:
                # Backoff and refresh client
                delay = 0.3 * (attempt + 1) * (attempt + 1)
                if logger:
                    logger.info(f"Retrying request to /embeddings in {delay:.3f} seconds")
                time.sleep(delay)
                # Recreate client (in case connection is stale)
                if openai_api_key and not openai_api_key.startswith("fw_"):
                    from openai import OpenAI
                    embedding_client = OpenAI(api_key=openai_api_key, base_url=None)
                else:
                    embedding_client = get_openai_client()
                continue
            return []


def embed_texts(
    texts: List[str],
    model: str = "text-embedding-3-small",
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> List[List[float]]:
    """
    Batch embedding: one API call for all texts (openai.embeddings.create(input=[list_of_texts])).
    Do not call embed_text() in a loop; use this to avoid 50+ sequential network calls.

    Returns embeddings in the same order as input texts. Cache hits are filled first;
    only cache misses are sent in a single batched request.

    CRITICAL: Always uses OpenAI API for embeddings, even if a Fireworks client is passed.
    The client parameter is ignored - embeddings always use OpenAI API endpoint.

    Args:
        texts: List of texts to embed (e.g. one per entity or list_of_5_terms)
        model: Embedding model name (default: text-embedding-3-small)
        client: Ignored - always uses OpenAI API
        logger: Optional logger

    Returns:
        List of embedding vectors (same order as input texts)
    """
    if not texts:
        return []

    # Pre-fill with empty embeddings for invalid/blank inputs
    embeddings_out: List[List[float]] = [[] for _ in texts]

    cleaned_texts: List[str] = []
    valid_indices: List[int] = []
    for i, t in enumerate(texts):
        t2 = t.strip() if isinstance(t, str) else ""
        if t2:
            cleaned_texts.append(t2)
            valid_indices.append(i)

    if not cleaned_texts:
        return embeddings_out

    # Cache check: fill hits immediately, batch-call only misses.
    miss_texts: List[str] = []
    miss_valid_indices: List[int] = []
    for orig_idx, t in zip(valid_indices, cleaned_texts):
        key = (model, t)
        cached = _cache_get(key)
        if cached is not None and cached:
            embeddings_out[orig_idx] = list(cached)
        else:
            miss_texts.append(t)
            miss_valid_indices.append(orig_idx)

    if not miss_texts:
        return embeddings_out

    embedding_client, openai_api_key = _resolve_embedding_client(logger=logger)
    if not embedding_client:
        if logger:
            logger.error("OpenAI client not available for batch embeddings")
        return embeddings_out

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = embedding_client.embeddings.create(
                model=model,
                input=miss_texts,
            )
            batch_embeddings = [None] * len(miss_texts)
            for item in resp.data:
                batch_embeddings[item.index] = item.embedding
            # Map back to original order + populate cache
            for orig_idx, text, embedding in zip(miss_valid_indices, miss_texts, batch_embeddings):
                vec = embedding or []
                embeddings_out[orig_idx] = vec
                if vec:
                    try:
                        _cache_set((model, text), list(vec))
                    except Exception:
                        pass

            return embeddings_out
        except Exception as e:
            err_text = str(e).lower()
            retryable = any(
                token in err_text
                for token in [
                    "connection error",
                    "timeout",
                    "timed out",
                    "rate limit",
                    "server error",
                    "502",
                    "503",
                    "504",
                ]
            )
            if logger:
                logger.error(f"Error generating batch embeddings: {e}")
                if hasattr(embedding_client, 'base_url'):
                    logger.debug(f"Embedding client base_url: {embedding_client.base_url}")
                logger.debug(f"Model requested: {model}")
            if attempt < max_retries - 1 and retryable:
                delay = 0.3 * (attempt + 1) * (attempt + 1)
                if logger:
                    logger.info(f"Retrying batch /embeddings in {delay:.3f} seconds")
                time.sleep(delay)
                if openai_api_key and not openai_api_key.startswith("fw_"):
                    from openai import OpenAI
                    embedding_client = OpenAI(api_key=openai_api_key, base_url=None)
                else:
                    embedding_client = get_openai_client()
                continue
            return embeddings_out


def to_pgvector_literal(vec: List[float]) -> str:
    """
    Convert Python list[float] to pgvector text literal: '[1,2,3,…]'.
    Ensures all elements are floats before formatting.
    
    Args:
        vec: List of floats representing the embedding vector
    
    Returns:
        PostgreSQL vector literal string
    
    Raises:
        ValueError: If vec is None, empty, or contains non-float values
    """
    if vec is None:
        raise ValueError("Cannot convert None to pgvector literal")
    if not vec or len(vec) == 0:
        raise ValueError("Cannot convert empty list to pgvector literal")
    
    # Convert all elements to float (handles strings, numpy types, etc.)
    # Add validation to catch malformed data
    float_vec = []
    for i, x in enumerate(vec):
        try:
            float_val = float(x)
            float_vec.append(float_val)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert element at index {i} to float: {x} (type: {type(x).__name__})") from e
    
    return "[" + ",".join(f"{x:.7f}" for x in float_vec) + "]"

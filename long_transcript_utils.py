"""
Utilities for handling long transcripts safely (chunking, summary blocks, prompt-safe excerpts).

Goal: prevent LLM context overflows on long audio (e.g., ~1 hour) while preserving key clinical content.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


SUMMARY_BEGIN = "[LONG_TRANSCRIPT_SUMMARY]"
SUMMARY_END = "[/LONG_TRANSCRIPT_SUMMARY]"


def chunk_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Chunk text by characters with overlap, attempting to break on whitespace.

    - chunk_size: size of each chunk in chars (approx)
    - overlap: chars of overlap between chunks (approx)
    """
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    if overlap < 0:
        overlap = 0

    t = text
    n = len(t)
    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        # Try to end on whitespace to avoid chopping words
        if end < n:
            ws = t.rfind(" ", start, end)
            if ws != -1 and ws > start + (chunk_size // 2):
                end = ws
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def wrap_with_summary_block(cleaned_transcript: str, summary: str) -> str:
    summary = (summary or "").strip()
    if not summary:
        return cleaned_transcript
    cleaned = cleaned_transcript or ""
    # Ensure summary block is at the top
    return f"{SUMMARY_BEGIN}\n{summary}\n{SUMMARY_END}\n\n{cleaned}".strip()


def extract_summary_block(text: str) -> Tuple[Optional[str], str]:
    """
    Returns (summary_or_None, text_without_summary_block).
    If multiple blocks exist, extracts the first.
    """
    if not text:
        return None, ""
    s = text
    begin = s.find(SUMMARY_BEGIN)
    end = s.find(SUMMARY_END)
    if begin == -1 or end == -1 or end < begin:
        return None, s
    summary = s[begin + len(SUMMARY_BEGIN) : end].strip()
    remainder = (s[:begin] + s[end + len(SUMMARY_END) :]).strip()
    return summary or None, remainder


def build_prompt_safe_transcript(
    transcript: str,
    *,
    max_chars: int,
    head_chars: int = 6000,
    tail_chars: int = 6000,
) -> str:
    """
    Build a prompt-safe representation of a transcript without any extra LLM calls.
    - If transcript <= max_chars, returns as-is
    - Else returns head+tail with a truncation marker
    """
    t = transcript or ""
    if len(t) <= max_chars:
        return t
    head = t[: max(0, head_chars)].strip()
    tail = t[-max(0, tail_chars) :].strip()
    omitted = max(0, len(t) - len(head) - len(tail))
    return (
        f"{head}\n\n"
        f"[... TRUNCATED {omitted} CHARS TO STAY WITHIN PROMPT LIMITS ...]\n\n"
        f"{tail}"
    ).strip()


def normalize_whitespace(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


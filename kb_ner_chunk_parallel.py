"""
Chunk-Parallel Factory for Sub-60s Pipeline

Implements raw-level chunking with overlap, parallel Super-Pass + Brain NER processing,
and intelligent merge/deduplication to achieve <55s latency while maintaining accuracy.

Chunking is adaptive to transcript length:
- Short transcript (<= CHUNK_SINGLE_THRESHOLD): 1 chunk, no parallel split.
- Longer transcript: number of chunks scales with length (min CHUNK_MIN_PARALLEL,
  max CHUNK_MAX_PARALLEL), and chunk_size is computed so chunks cover the text.
Env: CHUNK_SINGLE_THRESHOLD, CHUNK_TARGET_SIZE, CHUNK_MIN_PARALLEL, CHUNK_MAX_PARALLEL, CHUNK_MIN_SIZE.
"""

import asyncio
import json
import logging
import math
import os
import re
from typing import List, Dict, Any, Tuple, Optional, Set

from kb_ner_super_pass import super_pass_cleaning_and_ner, run_brain_call

# Adaptive chunking: derive number and size of chunks from transcript length
SINGLE_CHUNK_THRESHOLD = int(os.getenv("CHUNK_SINGLE_THRESHOLD", "4000"))  # Below this: 1 chunk (no split)
TARGET_CHUNK_SIZE = int(os.getenv("CHUNK_TARGET_SIZE", "4000"))  # Target chars per chunk for sizing
MIN_PARALLEL_CHUNKS = int(os.getenv("CHUNK_MIN_PARALLEL", "2"))  # When splitting, use at least this many
MAX_PARALLEL_CHUNKS = int(os.getenv("CHUNK_MAX_PARALLEL", "6"))  # Cap parallel workers
MIN_CHUNK_SIZE = int(os.getenv("CHUNK_MIN_SIZE", "2000"))  # Don't create chunks smaller than this
# Per-chunk timeout so one stuck chunk (e.g. LLM hang) doesn't block the whole pipeline
CHUNK_WORKER_TIMEOUT_SEC = float(os.getenv("CHUNK_WORKER_TIMEOUT_SEC", "180"))


def compute_adaptive_chunk_params(
    transcript_len: int,
    overlap_size: int,
    single_chunk_threshold: int = SINGLE_CHUNK_THRESHOLD,
    target_chunk_size: int = TARGET_CHUNK_SIZE,
    max_parallel: int = MAX_PARALLEL_CHUNKS,
    min_parallel: int = MIN_PARALLEL_CHUNKS,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> Tuple[int, int]:
    """
    Compute chunk_size and number of chunks from transcript length (adaptive/dynamic).
    - Short transcript → 1 chunk (no parallel split).
    - Longer transcript → more chunks, up to max_parallel; chunk_size shrinks as needed.
    Returns:
        (chunk_size, num_chunks) for use with chunk_raw_transcript_with_overlap.
    """
    if transcript_len <= single_chunk_threshold:
        return (transcript_len, 1)
    n = math.ceil(transcript_len / target_chunk_size)
    n = min(max_parallel, max(min_parallel, n))
    # chunk_size so that n chunks (with overlap) cover transcript_len
    chunk_size = (transcript_len + (n - 1) * overlap_size) / n
    if chunk_size < min_chunk_size and n > min_parallel:
        n = max(min_parallel, int((transcript_len + (n - 1) * overlap_size) / min_chunk_size))
        n = min(max_parallel, n)
        chunk_size = (transcript_len + (n - 1) * overlap_size) / n
    return (int(chunk_size), n)


def chunk_raw_transcript_with_overlap(
    raw_transcript: str,
    chunk_size: int = 4000,
    overlap_size: int = 500,
    min_chunk_size: int = 1000,
) -> List[Tuple[int, int, str]]:
    """
    Split raw transcript into chunks with overlap to prevent entity loss at boundaries.
    
    Args:
        raw_transcript: Raw transcript text
        chunk_size: Target size for each chunk (default: 4000 chars)
        overlap_size: Overlap between chunks (default: 500 chars)
        min_chunk_size: Minimum chunk size (smaller chunks merged with previous)
    
    Returns:
        List of (start_char, end_char, chunk_text) tuples
    """
    total_len = len(raw_transcript)
    if total_len <= chunk_size:
        return [(0, total_len, raw_transcript)]
    
    chunks = []
    start = 0
    
    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk_text = raw_transcript[start:end]
        
        # If this is not the last chunk and we're not at the end, extend to include overlap
        if end < total_len:
            # Extend chunk to include overlap region
            overlap_end = min(end + overlap_size, total_len)
            chunk_text = raw_transcript[start:overlap_end]
            chunks.append((start, overlap_end, chunk_text))
            # Next chunk starts at the overlap point (not at end)
            start = end - overlap_size
        else:
            chunks.append((start, end, chunk_text))
            break
    
    # Merge chunks that are too small
    merged_chunks = []
    for i, (start, end, text) in enumerate(chunks):
        if i == 0:
            merged_chunks.append((start, end, text))
        else:
            prev_start, prev_end, prev_text = merged_chunks[-1]
            if len(text) < min_chunk_size:
                # Merge tiny chunks (including a tiny tail chunk) into previous chunk to
                # avoid low-value extra LLM calls and latency spikes.
                merged_chunks[-1] = (prev_start, end, raw_transcript[prev_start:end])
            else:
                merged_chunks.append((start, end, text))
    
    return merged_chunks


def extract_signalment(raw_transcript: str) -> str:
    """
    Extract patient signalment (name, age, breed, sex) from raw transcript.
    Looks for patterns like "5 year old male Labrador", "Oreo", etc.
    
    Returns:
        Signalment string (e.g., "Patient: Oreo, 5yo male Labrador") or empty string
    """
    head = raw_transcript[:2000]
    signalment_parts: List[str] = []

    # Try to find a likely patient name (simple heuristic, early in transcript).
    name_match = re.search(r"\b([A-Z][a-z]{2,})\b", raw_transcript[:500])
    if name_match:
        potential_name = name_match.group(1)
        if potential_name.lower() not in {"the", "this", "that", "there", "here", "what", "when", "where", "who", "how"}:
            signalment_parts.append(potential_name)

    # Age (single numeric capture)
    age_match = re.search(r"\b(\d{1,2})\s*(?:year|yr|years|yrs?)\s*old\b", head, re.IGNORECASE)
    if age_match:
        signalment_parts.append(f"{age_match.group(1)}yo")

    # Sex
    sex_match = re.search(r"\b(male|female|m|f)\b", head, re.IGNORECASE)
    if sex_match:
        sex = sex_match.group(1).lower()
        signalment_parts.append("male" if sex in {"m", "male"} else "female")

    # Breed phrase after age/sex marker (kept conservative to avoid junk tokens).
    breed_match = re.search(
        r"\b\d{1,2}\s*(?:year|yr|years|yrs?)\s*old(?:\s*(?:male|female|m|f))?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\b",
        head,
    )
    if breed_match:
        breed = breed_match.group(1).strip()
        # Trim accidental directional/location spillover from ASR.
        tail_noise = {"right", "left", "side", "hip", "leg", "knee"}
        breed_tokens = [t for t in breed.split() if t.lower() not in tail_noise]
        breed = " ".join(breed_tokens).strip()
        if breed and breed.lower() not in {"dog", "cat", "pet"}:
            signalment_parts.append(breed)

    # Preserve order while deduplicating.
    deduped_parts: List[str] = []
    seen = set()
    for part in signalment_parts:
        key = part.lower().strip()
        if key and key not in seen:
            deduped_parts.append(part.strip())
            seen.add(key)

    if deduped_parts:
        return f"Patient: {', '.join(deduped_parts)}"
    return ""


def inject_signalment_header(chunk_text: str, signalment: str) -> str:
    """
    Inject signalment header at the beginning of chunk to maintain patient context.
    
    Args:
        chunk_text: Chunk text
        signalment: Signalment string (e.g., "Patient: Oreo, 5yo male Labrador")
    
    Returns:
        Chunk text with signalment header prepended
    """
    if not signalment:
        return chunk_text
    return f"{signalment}\n\n{chunk_text}"


async def process_chunk_worker(
    chunk_data: Tuple[int, int, str],
    signalment: str,
    model: str,
    client: Any,
    logger: Optional[logging.Logger],
    raw_transcript: str,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, List[Dict[str, str]]]]:
    """
    Process a single chunk: Super-Pass cleaning + NER, then Brain NER enrichment.
    
    OPTIMIZED FLOW: Super-Pass runs first, then Brain NER starts IMMEDIATELY when Super-Pass finishes
    (doesn't wait for other chunks). This maximizes parallelism across chunks.
    
    Args:
        chunk_data: (start_char, end_char, chunk_text) tuple
        signalment: Patient signalment header to inject
        model: Model name for LLM calls
        client: OpenAI-compatible client
        logger: Logger instance
        raw_transcript: Full raw transcript (for offset calculation)
    
    Returns:
        Tuple of (cleaned_segment, enriched_entities, entities_by_kind)
    """
    start_char, end_char, chunk_text = chunk_data
    
    chunk_with_header = inject_signalment_header(chunk_text, signalment)
    
    if logger:
        logger.info(f"  🔄 Processing chunk: chars {start_char}-{end_char} ({len(chunk_text)} chars) [PARALLEL START]")

    # Step A: Super-Pass (cleaning + NER) - wrap in thread to prevent blocking event loop
    # CRITICAL: super_pass_cleaning_and_ner is async but uses blocking client.chat.completions.create()
    # Wrapping in asyncio.to_thread ensures true parallelism across chunks (all 3 start simultaneously)
    def _run_super_pass_sync():
        """Synchronous wrapper: run async super_pass in a new event loop (in thread)."""
        return asyncio.run(
            super_pass_cleaning_and_ner(
                chunk_with_header,
                model=model,
                client=client,
                logger=logger,
            )
        )
    
    cleaned_segment_local, super_pass_entities, entities_by_kind_local = await asyncio.to_thread(_run_super_pass_sync)

    if logger:
        logger.info(f"  📋 Chunk {start_char}-{end_char}: Super-Pass complete ({len(super_pass_entities or [])} entities) → Brain NER starting")
    if not cleaned_segment_local:
        cleaned_segment_local = chunk_text.strip()

    # Step B: Brain NER enrichment - starts IMMEDIATELY after Super-Pass finishes (doesn't wait for other chunks)
    pre_extracted = [
        {
            "id": f"e{i+1}",
            "span_text": e.get("span_text", ""),
            "kind": e.get("kind", "Other"),
            "attributes": e.get("attributes", {}),
        }
        for i, e in enumerate(super_pass_entities or [])
    ]
    pre_extracted_json = json.dumps(pre_extracted, ensure_ascii=False)

    # Run Brain NER in thread (synchronous call) - this allows other chunks' Brain NER to run in parallel
    entity_manifest_local, _terms_not_grounded = await asyncio.to_thread(
        run_brain_call,
        cleaned_segment_local,
        pre_extracted_json,
        model,
        client,
        logger,
    )

    # Compute global offsets against raw chunk window for robust cross-chunk dedupe.
    raw_window = raw_transcript[start_char:end_char]
    raw_window_lower = raw_window.lower()
    rolling_pos = 0
    for ent in entity_manifest_local or []:
        span_text = (ent.get("span_text") or "").strip()
        span_lower = span_text.lower()
        local_idx = -1
        if span_lower:
            local_idx = raw_window_lower.find(span_lower, rolling_pos)
            if local_idx == -1:
                local_idx = raw_window_lower.find(span_lower)
        if local_idx != -1:
            global_start = start_char + local_idx
            global_end = global_start + len(span_text)
            rolling_pos = local_idx + len(span_text)
        else:
            local_start_fallback = int(ent.get("start_char") or 0)
            global_start = max(start_char, start_char + local_start_fallback)
            global_end = global_start + len(span_text)
        ent["start_char"] = global_start
        ent["end_char"] = global_end
        ent["_chunk_start"] = start_char
        ent["_chunk_end"] = end_char

    if logger:
        logger.info(f"  ✅ Chunk {start_char}-{end_char}: {len(entity_manifest_local or [])} enriched entities")

    return cleaned_segment_local, (entity_manifest_local or []), (entities_by_kind_local or {})


def merge_cleaned_segments(
    chunks: List[Tuple[int, int, str]],
    cleaned_segments: List[str],
    overlap_size: int = 500,
) -> str:
    """
    Merge cleaned segments by stitching at overlap midpoint to avoid double-texting.
    
    Args:
        chunks: List of (start_char, end_char, chunk_text) tuples
        cleaned_segments: List of cleaned segment texts (in same order as chunks)
        overlap_size: Size of overlap region
    
    Returns:
        Merged cleaned transcript
    """
    def _strip_header(text: str) -> str:
        return re.sub(r"^Patient:.*?\n\n+", "", text, count=1).strip()

    def _trim_prefix_overlap(existing: str, nxt: str, max_overlap: int = 1200) -> str:
        if not existing or not nxt:
            return nxt
        m = min(max_overlap, len(existing), len(nxt))
        for k in range(m, 30, -1):
            if existing[-k:] == nxt[:k]:
                return nxt[k:]
        return nxt

    ordered_segments = [_strip_header(seg) for seg in cleaned_segments if (seg or "").strip()]
    if not ordered_segments:
        return ""
    merged = ordered_segments[0]
    for seg in ordered_segments[1:]:
        merged += _trim_prefix_overlap(merged, seg, max_overlap=max(400, overlap_size * 2))
    return merged.strip()


# Terms that are recommendations/place context, not billable or actionable entities.
FILTER_SPAN_TEXT_LOWER: Set[str] = {
    "beach", "lawns", "sandy surfaces", "one fourth bowl", "neurological",
    "reflexes",  # test component, not anatomy
}
# Severity-only "diagnosis" to drop (merge into parent diagnosis elsewhere if needed).
FILTER_SEVERITY_ONLY: Set[str] = {"grade 3", "grade 2", "grade 1", "grade 4"}


def _normalized_name(ent: Dict[str, Any]) -> str:
    return ((ent.get("normalized_name") or ent.get("span_text") or "").strip().lower())


def _kb_identity(ent: Dict[str, Any]) -> Optional[Tuple[Optional[Any], Optional[Any], Optional[Any]]]:
    """Return (kb_concept_id, local_service_id, local_stock_id) if any is set (for semantic dedup)."""
    cid = ent.get("kb_concept_id")
    sid = ent.get("local_service_id")
    stock = ent.get("local_stock_id")
    if cid is not None or sid is not None or stock is not None:
        return (cid, sid, stock)
    return None


def _anatomy_or_target(ent: Dict[str, Any]) -> Optional[str]:
    """Return anatomy/body_part/target_site for procedure/diagnostic dual-check."""
    attrs = ent.get("attributes") or {}
    if not isinstance(attrs, dict):
        return None
    for key in ("anatomy", "body_part", "target_site", "body_site", "site"):
        v = attrs.get(key)
        if v and isinstance(v, str) and v.strip():
            return v.strip().lower()
    return None


def _intent_value(ent: Dict[str, Any]) -> Optional[str]:
    """Return Ordered vs History (or Past) for intent-based dedup guard."""
    attrs = ent.get("attributes") or {}
    if isinstance(attrs, dict):
        i = attrs.get("intent") or attrs.get("status") or attrs.get("temporal")
        if i and isinstance(i, str):
            return i.strip().lower()
    # Top-level if present
    i = ent.get("intent") or ent.get("status")
    if i and isinstance(i, str):
        return i.strip().lower()
    return None


def _fold_attributes_into(kept: Dict[str, Any], ent: Dict[str, Any]) -> None:
    """
    Merge attributes from ent into kept (attribute folding).
    Prefer non-empty values from ent when kept's value is missing or empty.
    """
    k_attrs = kept.get("attributes")
    if not isinstance(k_attrs, dict):
        k_attrs = {}
        kept["attributes"] = k_attrs
    e_attrs = ent.get("attributes") or {}
    if not isinstance(e_attrs, dict):
        return
    for key, val in e_attrs.items():
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        existing = k_attrs.get(key)
        if existing is None or (isinstance(existing, str) and not existing.strip()):
            k_attrs[key] = val
    # Optionally fold supporting_text if kept lacks it
    if not (kept.get("supporting_text") or "").strip() and (ent.get("supporting_text") or "").strip():
        kept["supporting_text"] = ent.get("supporting_text")


def _is_acronym_of(short: str, long: str) -> bool:
    """True if short is acronym/initialism of long (e.g. FHO vs femoral head and neck ostectomy)."""
    short = short.strip().lower().replace("-", " ").replace(".", "")
    long = long.strip().lower().replace("-", " ")
    if not short or not long or len(short) > 10:
        return False
    words = [w for w in re.split(r"[\s,]+", long) if w and len(w) > 1]
    if len(words) < len(short):
        return False
    # Match each letter of short to first letter of a word (in order, may skip words).
    j = 0
    for i in range(len(short)):
        while j < len(words) and words[j][:1] != short[i:i + 1]:
            j += 1
        if j >= len(words):
            return False
        j += 1
    return True


def _is_contained_or_contains(a: str, b: str) -> bool:
    """True if one string is substring of the other (for partial name match)."""
    a, b = a.strip().lower(), b.strip().lower()
    return a in b or b in a


def _same_signalment_concept(ent: Dict[str, Any], kept: Dict[str, Any]) -> bool:
    """True if both entities are signalment components (patient name, age, sex, breed)."""
    k1 = (ent.get("kind") or "").strip().lower()
    k2 = (kept.get("kind") or "").strip().lower()
    signalment_kinds = {"patient", "other", "anatomy", "vitalsign"}
    name1 = _normalized_name(ent)
    name2 = _normalized_name(kept)
    if k1 not in signalment_kinds or k2 not in signalment_kinds:
        return False
    # Same token (Oreo, 5yo, male, labrador, 5-year-old)
    if name1 == name2:
        return True
    # One is contained in the other (e.g. "oreo" in "oreo, 5yo, male, labrador")
    if _is_contained_or_contains(name1, name2):
        return True
    # Age variants
    if re.match(r"^\d+yo?$", name1) and re.match(r"^\d+yo?$", name2):
        return True
    if re.match(r"^\d+-year-old", name1) and re.match(r"^\d+yo?$", name2):
        return True
    return False


def _is_duplicate_of(
    ent: Dict[str, Any],
    kept: Dict[str, Any],
    offset_threshold: int,
) -> bool:
    """
    True if ent should be considered duplicate of kept.
    Uses KB_ID first, then intent guard, then anatomy guard for procedures, then string/offset logic.
    """
    kind_ent = (ent.get("kind") or "Other").strip().lower()
    kind_kept = (kept.get("kind") or "Other").strip().lower()
    start = int(ent.get("start_char") or 0)
    end = int(ent.get("end_char") or 0)
    ks = int(kept.get("start_char") or 0)
    ke = int(kept.get("end_char") or 0)
    has_valid = end > start >= 0
    kept_has_valid = ke > ks >= 0

    # A. Semantic synonym: same KB identity => merge regardless of string
    kid_ent = _kb_identity(ent)
    kid_kept = _kb_identity(kept)
    if kid_ent and kid_kept:
        if kid_ent[0] is not None and kid_kept[0] is not None and kid_ent[0] == kid_kept[0]:
            return True
        if kid_ent[1] is not None and kid_kept[1] is not None and kid_ent[1] == kid_kept[1]:
            return True
        if kid_ent[2] is not None and kid_kept[2] is not None and kid_ent[2] == kid_kept[2]:
            return True

    # B. Intent guard: never merge Ordered with History/Past
    i_ent = _intent_value(ent)
    i_kept = _intent_value(kept)
    ordered_like = {"ordered", "order", "planned", "prescribed", "to do"}
    history_like = {"history", "past", "previous", "had", "done", "performed"}
    if i_ent and i_kept:
        e_ordered = i_ent in ordered_like or "order" in i_ent
        k_ordered = i_kept in ordered_like or "order" in i_kept
        e_history = i_ent in history_like or "past" in i_ent or "history" in i_ent
        k_history = i_kept in history_like or "past" in i_kept or "history" in i_kept
        if (e_ordered and k_history) or (e_history and k_ordered):
            return False

    # C. Dual procedure risk: same procedure/diagnostic but different anatomy => do not merge
    if kind_ent in ("procedure", "diagnostic", "diagnosis") and kind_ent == kind_kept:
        a_ent = _anatomy_or_target(ent)
        a_kept = _anatomy_or_target(kept)
        if a_ent and a_kept and a_ent != a_kept:
            return False

    name_ent = _normalized_name(ent)
    name_kept = _normalized_name(kept)

    # Exact match + close offset
    if name_ent == name_kept and kind_ent == kind_kept:
        if has_valid and kept_has_valid:
            if abs(start - ks) <= offset_threshold and abs(end - ke) <= offset_threshold:
                return True
        else:
            return True

    # Same kind: one name contains the other (keep longer) and offsets close
    if kind_ent == kind_kept and _is_contained_or_contains(name_ent, name_kept):
        if has_valid and kept_has_valid and abs(start - ks) <= offset_threshold * 2:
            return True

    # Acronym expansion (e.g. FHO vs Femoral Head and Neck Ostectomy)
    if kind_ent == kind_kept:
        if _is_acronym_of(name_ent, name_kept) or _is_acronym_of(name_kept, name_ent):
            if has_valid and kept_has_valid and abs(start - ks) <= 200:
                return True
        # Diagnosis with severity: "hip dysplasia" vs "severe hip dysplasia, grade 3"
        if "diagnosis" in kind_ent:
            base_kept = re.sub(r",?\s*grade\s*\d+", "", name_kept).strip()
            base_ent = re.sub(r",?\s*grade\s*\d+", "", name_ent).strip()
            if base_ent == base_kept or _is_contained_or_contains(base_ent, base_kept):
                if has_valid and kept_has_valid and abs(start - ks) <= 150:
                    return True

    # Signalment components
    if _same_signalment_concept(ent, kept):
        if has_valid and kept_has_valid and abs(start - ks) <= offset_threshold * 3:
            return True
        if not has_valid or not kept_has_valid:
            return True

    return False


def filter_non_actionable_entities(
    entities: List[Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    Remove entities that are recommendations, quantities, or severity-only.
    """
    out: List[Dict[str, Any]] = []
    for e in entities:
        span = (e.get("span_text") or "").strip().lower()
        norm = (e.get("normalized_name") or "").strip().lower()
        kind = (e.get("kind") or "").strip().lower()
        if span in FILTER_SPAN_TEXT_LOWER or norm in FILTER_SPAN_TEXT_LOWER:
            if logger:
                logger.debug(f"  🚫 Filtered non-actionable: '{e.get('span_text')}' (kind={kind})")
            continue
        # Severity-only (grade 1–4): filter whether kind is Diagnosis or Diagnostic
        if span in FILTER_SEVERITY_ONLY or norm in FILTER_SEVERITY_ONLY:
            if logger:
                logger.debug(f"  🚫 Filtered severity-only: '{e.get('span_text')}' (kind={kind})")
            continue
        out.append(e)
    return out


# Order for final entity list (Verification Dashboard / billing readability).
BILLING_KIND_ORDER: List[str] = [
    "Medication", "Procedure", "Diagnostic", "Diagnosis", "Preventive", "ParasiteControl",
    "Diet", "VitalSign", "Symptom", "Anatomy", "Other", "Patient", "Reminder", "Identity",
    "Signalment",
]


def sort_entities_by_billing_kind(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort by Kind (Medication > Procedure > Diagnostic > ...) for Verification Dashboard."""
    order_map = {k.lower(): i for i, k in enumerate(BILLING_KIND_ORDER)}
    max_idx = len(BILLING_KIND_ORDER)

    def _key(e: Dict[str, Any]) -> Tuple[int, int]:
        kind = (e.get("kind") or "Other").strip().lower()
        idx = order_map.get(kind, max_idx)
        start = int(e.get("start_char") or 0)
        return (idx, start)

    return sorted(entities, key=_key)


def correct_entity_kinds(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fix known kind misclassifications (e.g. breed as Anatomy → Other).
    """
    # Common breed names that are sometimes misclassified as Anatomy
    breed_like = {"labrador", "retriever", "german shepherd", "golden retriever", "beagle", "poodle"}
    out: List[Dict[str, Any]] = []
    for e in entities:
        e = dict(e)
        norm = _normalized_name(e)
        kind = (e.get("kind") or "").strip().lower()
        if kind == "anatomy" and norm in breed_like:
            e["kind"] = "Other"
        out.append(e)
    return out


def deduplicate_entities(
    all_entities: List[List[Dict[str, Any]]],
    raw_transcript: str,
    offset_threshold: int = 80,
) -> List[Dict[str, Any]]:
    """
    Deduplicate entities across chunks: exact match, substring match,
    acronym expansion, severity-in-diagnosis, and signalment components.
    """
    flat_entities: List[Dict[str, Any]] = []
    for chunk_entities in all_entities:
        if isinstance(chunk_entities, list):
            flat_entities.extend([e for e in chunk_entities if isinstance(e, dict)])
    if not flat_entities:
        return []

    # Sort by (start_char, -length) so longer/more specific spans are considered after shorter ones.
    def _sort_key(ent: Dict[str, Any]) -> Tuple[int, int]:
        s = int(ent.get("start_char") or 0)
        span_len = len((ent.get("span_text") or ent.get("normalized_name") or ""))
        return (s, -span_len)

    flat_entities.sort(key=_sort_key)

    deduped: List[Dict[str, Any]] = []
    for ent in flat_entities:
        merged_into: Optional[Dict[str, Any]] = None
        for kept in deduped:
            if _is_duplicate_of(ent, kept, offset_threshold):
                _fold_attributes_into(kept, ent)
                merged_into = kept
                break
        if merged_into is None:
            deduped.append(ent)

    return deduped


async def process_chunks_parallel(
    raw_transcript: str,
    model: str,
    client: Any,
    logger: Optional[logging.Logger],
    chunk_size: int = 4000,
    overlap_size: int = 500,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, List[Dict[str, str]]]]:
    """
    Main chunk-parallel processing function.
    
    Args:
        raw_transcript: Raw transcript text
        model: Model name for LLM calls
        client: OpenAI-compatible client
        logger: Logger instance
        chunk_size: Target chunk size (default: 4000)
        overlap_size: Overlap between chunks (default: 500)
    
    Returns:
        Tuple of (merged_cleaned_transcript, deduplicated_enriched_entities, merged_entities_by_kind)
    """
    if logger:
        logger.info("=" * 60)
        logger.info("CHUNK-PARALLEL FACTORY: Processing")
        logger.info("=" * 60)
        logger.info(f"  Raw transcript length: {len(raw_transcript)} chars")
    
    # Step 1: Extract signalment
    signalment = extract_signalment(raw_transcript)
    if logger and signalment:
        logger.info(f"  📋 Extracted signalment: {signalment}")
    
    # Step 2: Adaptive chunking — derive chunk_size and num chunks from transcript length
    total_len = len(raw_transcript)
    adaptive_chunk_size, estimated_chunks = compute_adaptive_chunk_params(total_len, overlap_size)
    chunk_size_used = adaptive_chunk_size
    if logger:
        logger.info(
            f"  📐 Adaptive chunking: {total_len} chars → chunk_size={chunk_size_used}, ~{estimated_chunks} chunk(s)"
        )
    
    # Step 3: Chunk raw transcript with overlap
    chunks = chunk_raw_transcript_with_overlap(raw_transcript, chunk_size_used, overlap_size)
    if logger:
        logger.info(f"  ✂️ Split into {len(chunks)} chunks with {overlap_size}-char overlap")
        for i, (start, end, text) in enumerate(chunks):
            logger.info(f"     Chunk {i+1}: chars {start}-{end} ({len(text)} chars)")
    
    # Step 4: Launch parallel workers
    if logger:
        logger.info(f"  🚀 Launching {len(chunks)} parallel workers...")
    
    timeout_sec = CHUNK_WORKER_TIMEOUT_SEC

    async def _worker_with_timeout(chunk_data: Tuple[int, int, str]):
        try:
            return await asyncio.wait_for(
                process_chunk_worker(chunk_data, signalment, model, client, logger, raw_transcript),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            start_c, end_c, _ = chunk_data
            if logger:
                logger.error(
                    f"  ⏱️ Chunk {start_c}-{end_c}: TIMED OUT after {timeout_sec}s (Super-Pass or Brain NER stuck). "
                    "Returning empty for this chunk; other chunks will still be merged."
                )
            # Return empty so merge/dedup can proceed with other chunks
            return "", [], {}

    worker_tasks = [_worker_with_timeout(chunk_data) for chunk_data in chunks]
    results = await asyncio.gather(*worker_tasks, return_exceptions=True)

    # Check for errors (including TimeoutError already handled above; other exceptions still in results)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        if logger:
            logger.error(f"  ❌ {len(errors)} chunk(s) failed: {errors}")
        results = [r for r in results if not isinstance(r, Exception)]
    
    if not results:
        if logger:
            logger.error("  ❌ All chunks failed")
        return "", [], {}
    
    # Extract results
    cleaned_segments = [r[0] for r in results]
    enriched_entities_per_chunk = [r[1] for r in results]
    entities_by_kind_per_chunk = [r[2] for r in results]
    
    # Step 5: Merge cleaned segments
    merged_cleaned = merge_cleaned_segments(chunks, cleaned_segments, overlap_size)
    if logger:
        logger.info(f"  🔗 Merged cleaned transcript: {len(merged_cleaned)} chars")
    
    # Step 6: Deduplicate entities (KB_ID, acronym, signalment, severity, attribute folding)
    total_before_dedup = sum(len(e) for e in enriched_entities_per_chunk)
    deduplicated_entities = deduplicate_entities(enriched_entities_per_chunk, raw_transcript)
    if logger:
        logger.info(f"  🔍 Deduplicated entities: {total_before_dedup} → {len(deduplicated_entities)}")
    # Safety net: if reduction > 50%, warn (may indicate overly aggressive dedup)
    if logger and total_before_dedup > 0:
        reduction = (total_before_dedup - len(deduplicated_entities)) / total_before_dedup
        if reduction > 0.5:
            logger.warning(
                "  ⚠️ DEDUP SAFETY: entity count reduced by %.0f%% (%d → %d). "
                "Verify offset_threshold / intent/anatomy logic if this seems wrong.",
                reduction * 100, total_before_dedup, len(deduplicated_entities),
            )

    # Step 5b: Filter non-actionable entities (recommendations, quantities, severity-only)
    before_filter = len(deduplicated_entities)
    deduplicated_entities = filter_non_actionable_entities(deduplicated_entities, logger=logger)
    if logger and before_filter != len(deduplicated_entities):
        logger.info(f"  🚫 Filtered non-actionable: {before_filter} → {len(deduplicated_entities)}")

    # Step 5c: Correct kind misclassification (e.g. breed as Anatomy → Other)
    deduplicated_entities = correct_entity_kinds(deduplicated_entities)

    # Step 5d: Final sort for billing / Verification Dashboard (Medication > Procedure > Diagnostic > ...)
    deduplicated_entities = sort_entities_by_billing_kind(deduplicated_entities)

    # Step 7: Merge entities_by_kind (simple union)
    merged_entities_by_kind: Dict[str, List[Dict[str, str]]] = {}
    for ekb in entities_by_kind_per_chunk:
        if isinstance(ekb, dict):
            for kind, entities in ekb.items():
                if kind not in merged_entities_by_kind:
                    merged_entities_by_kind[kind] = []
                if isinstance(entities, list):
                    merged_entities_by_kind[kind].extend(entities)
    
    if logger:
        logger.info("  ✅ Chunk-parallel processing complete")
    
    return merged_cleaned, deduplicated_entities, merged_entities_by_kind

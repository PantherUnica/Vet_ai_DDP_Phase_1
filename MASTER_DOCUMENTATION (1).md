# SOAP Notes Phase 1 – Master Documentation

**Version**: 3.4  
**Last Updated**: 2026-02-25  
**Purpose**: Single source of truth for the SOAP Phase 1 pipeline: end-to-end flow, codebase, grounding (local-only, category-driven, medical/non-medical service type, domain fallback SKU), production safeguards (Golden Gate, symptom suppressor, Judge category incompatibility, ID-based dashboard), CER streaming, vector retrieval, Phase 2, and configuration.  
**Canonical prompt**: `BRAIN_NER_PROMPT_UPDATED.md` (project root) and `UNIFIED_CLEANING_AND_NER_PROMPT.md` (includes service_type). Historical and audit docs are in `legacy/docs/`.  
**Local grounding detail**: `docs/LOCAL_GROUNDING_SEARCH_AND_RETRIEVAL.md` (retrieval, scoring, hints, decision flow).  
**Grounded concepts & production safeguards**: `GROUNDED_CONCEPTS_QUERY_EXPANSION_DOCUMENTATION.md` (query expansion, hints, form-factor, symptom suppressor, Golden Gate for Diagnosis/ReasonForVisit, ID-based dashboard).

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Complete Pipeline Flow](#complete-pipeline-flow)
3. [Default Execution Path](#default-execution-path)
4. [Super-Pass (Cleaning + NER)](#super-pass-cleaning--ner)
5. [CER (Clinical Entity Resolver)](#cer-clinical-entity-resolver)
6. [Batch Intent & Families](#batch-intent--families)
7. [Grounding (Local-Only, Category-Driven)](#grounding-local-only-category-driven)
8. [Domain-Specific Fallback SKU](#domain-specific-fallback-sku)
9. [Routing Strategy](#routing-strategy)
10. [Phase 2: Knowledge Atoms & Dashboard](#phase-2-knowledge-atoms--dashboard)
11. [Indexes, Embeddings, and Fuzzy Matching](#indexes-embeddings-and-fuzzy-matching)
12. [Complete Codebase](#complete-codebase)
13. [Function Reference](#function-reference)
14. [Configuration & Environment](#configuration--environment)
15. [Database & Tables](#database--tables)
16. [Documentation & Legacy](#documentation--legacy)

---

## Executive Summary

The pipeline turns **raw audio** of veterinary consultations into **grounded SOAP notes** with entity linking to **local inventory** and **local services** only (no global KB in production).

**Flow (default):**
1. **Transcription** – Fireworks Whisper v3 Turbo (16 kHz).
2. **Super-Pass** – One LLM call: clean transcript + extract entities; outputs `inventory_category`, `service_category`, and **`service_type`** (medical | non-medical) per entity. Default model: **accounts/fireworks/models/llama-v3p3-70b-instruct**. Brain NER prompt in `docs/BRAIN_NER_PROMPT_UPDATED.md` and `UNIFIED_CLEANING_AND_NER_PROMPT.md`.
3. **CER (Clinical Entity Resolver)** – Consolidates entities to billing-only; default model **accounts/fireworks/models/llama-v3p3-70b-instruct**. On Fireworks, streaming path when output may exceed 4096 tokens; OpenAI non-streaming only. JSON repair for streamed output (trailing commas, truncation).
4. **Batch Intent** – One LLM call: assign `search_term` and `family` per entity (ASR correction, category). Category-Locked Guard: family must not contradict kind for PRODUCT/PROCEDURE.
5. **Grounding** – Per entity: route (skip_vitals, global_vitals, skip_signalment, skip_identity, **dual_sync**, global_direct). **Dual-sync (billable)**: search **both** `soap.inventory` and `soap.service_master` with **service_type** (medical vs non-medical) and category hints; **vector retrieval enabled** (batch + on-demand embed). **Domain fallback**: when high-stakes entities remain unlinked, draft one domain consultation SKU per (domain_key, service_id) per visit with grouped remarks; dedupe so multiple unlinked same-domain entities attach to one SKU. No global KB when `LOCAL_ONLY=true` (default).
6. **SOAP** – Generate note from cleaned transcript; entity-injected Plan (use manifest terms, no generic “as prescribed”).
7. **Injection** – Constraint + truth: replace span_text → display_name; optional Anchor-Span `[[E1|term]]`.
8. **Phase 2** – Knowledge atoms; default model **gpt-4.1-mini** (OpenAI). ID-first dedupe, locus routing (at-home → Reminders), verification dashboard.

**Design choices:**
- **Service type (medical / non-medical)**: Brain NER emits `service_type`; default for billable entities is **medical**. Service hard gate restricts to medical-only categories when entity is medical (Consultation, Surgery, Lab, Radiology, etc.); non-medical uses Boarding, Hygiene & Grooming, Training, Behavior, Other Non-Medical. **General** or missing category is treated as medical-only for services to prevent cross-category hallucinations (e.g. radiology term → grooming SKU).
- **Category-driven search**: Entity's `service_category` and `service_type` → search `soap.service_master`; `inventory_category` → search `soap.inventory`. Billable entities search **both** and merge; category buckets respect medical vs non-medical.
- **Domain**: Used as **soft** gate (boost/penalty only). No hard domain rejection.
- **Vector retrieval**: Enabled for local inventory and service_master. Batch embeddings include span_text, search_term, and hints; on-demand embed when cache miss (`LOCAL_VECTOR_ON_DEMAND_EMBED=true`).
- **Physiotherapy / Rehabilitation**: Kind "Physiotherapy" canonicalizes to "Procedure"; service category bucket includes "Rehabilitation & Physiotherapy", "Physiotherapy and Rehabilitation".
- **Local-only**: Global KB disabled by default (`LOCAL_ONLY=true`). All linking to clinic inventory and service_master.
- **Lexical safety gate**: Removed; judge-selected matches (e.g. "X-ray result" → "XRAY") are no longer rejected by token-overlap checks.
- **Production safeguards:** **Pharmacy-Free Zone** — Diagnosis and ReasonForVisit search Services only (never Pharmacy); **0.95 certainty wall** for those kinds (below 0.95 → note-only). **Judge:** Category Incompatibility (finding vs product → REJECT), Symptom/Physical Finding Suppressor. **Dashboard:** ID-based linking (local_stock_id / local_service_id as primary key; manifest_by_normalized_name_id reduces orphans). See `GROUNDED_CONCEPTS_QUERY_EXPANSION_DOCUMENTATION.md` §8.

**Entry:** `SOAP_notes_phase1_experiment.generate_soap_note_from_audio_async()`.

---

## Complete Pipeline Flow

```
Step 1: Audio → Raw transcript (Fireworks Whisper v3 Turbo, 16 kHz)
Step 2: Super-Pass → Cleaned transcript + extracted_entities (inventory_category, service_category, service_type per entity; default model Llama v3p3 70B)
Step 2a: CER (optional) → Resolve to billing entities (streaming when Fireworks + large output; JSON repair)
Step 2b: Batch Intent → search_term + family per entity
Step 2.3: Grounding → Entity manifest (dual_sync: local inventory + local services; service_type + category hints; vector retrieval; domain fallback SKU; merge; Judge if needed)
Step 3: SOAP generation → Note with entity-injected Plan
Step 4: Constraint + Truth injection → Grounded SOAP (optional Anchor-Span)
Step 5: Phase 2 → Knowledge atoms (default gpt-4.1-mini), dedupe, locus routing, verification dashboard
```

**Fallback:** `USE_SUPER_PASS=false` → separate cleaning then 2-Phase NER + grounding.

---

## Default Execution Path

1. **Startup:** Postgres pool init, `clinic_id` / `visit_id` from env (`CLINIC_ID`, `VISIT_ID`; default `clinic_id=1`).
2. **Step 1:** `transcribe_audio()` or `transcribe_audio_fireworks_streaming()` → raw transcript (Fireworks Whisper; retries on 502/503).
3. **Step 2 – Super-Pass (streaming default):** `super_pass_cleaning_and_ner_streaming()`; entities with `inventory_category`, `service_category`, `service_type`; default model Llama v3p3 70B; per chunk: batch embed → Batch Intent → dispatch to streaming grounding.
4. **Step 2.3 – Grounding:** `process_single_entity_async()` (streaming) or `run_step_2_3_normalization()` (non-streaming). Dual-sync: `search_local_inventory_topk(..., category_hints=...)` and `search_local_services_topk(..., category_hints=..., service_type=entity_service_type)` in parallel; vector retrieval (batch + on-demand embed); merge; auto-bind or LLM Judge; domain consultation fallback for high-stakes unlinked.
5. **Step 3 – SOAP:** `generate_soap_with_grounding()`; manifest with anchor_id (E1, E2, …) when `ANCHOR_MAPPING_SOAP=true`.
6. **Step 4 – Injection:** `apply_constraint_based_injection()` + `apply_manifest_corrections_to_soap_json()`.
7. **Phase 2:** `extract_knowledge_atoms_async()`; dedupe; locus routing; verification dashboard.

**Combined path:** `COMBINED_CLEAN_NER_BATCH_INTENT=true` → Clean + NER + Intent in one LLM call; then grounding with same local-only, category-driven search.

---

## Super-Pass (Cleaning + NER)

**File:** `kb_ner_super_pass.py`.  
**Prompt:** See `BRAIN_NER_PROMPT_UPDATED.md` (project root) and `UNIFIED_CLEANING_AND_NER_PROMPT.md`.

- **Cleaning:** Verbatim clinical preservation; no correction of drug/procedure names (grounding handles ASR).
- **Extraction:** Inclusive; tests/measurements/angles extracted even if misspelled (e.g. “Noble angle” → Batch Intent can correct to Norberg).
- **Output per entity:** `normalized_name`, `kind`, `domain`, **`inventory_category`**, **`service_category`**, **`service_type`** (medical | non-medical; default medical for billable), `correctness_probability`, `suggestion_probability`, `hints`.
- **Default model:** `accounts/fireworks/models/llama-v3p3-70b-instruct` (Fireworks).
- **Config:** `SUPER_PASS_MODEL`, `SUPER_PASS_STREAMING`, `LONG_TRANSCRIPT_THRESHOLD_CHARS`, `BATCH_EMBED_PER_CHUNK`.

---

## CER (Clinical Entity Resolver)

**File:** `kb_ner_clinical_entity_resolver.py`.

- **Purpose:** Resolve extracted entities into a consolidated set of billing-relevant actionable items (e.g. for Phase 2 or downstream billing).
- **Default model:** `accounts/fireworks/models/llama-v3p3-70b-instruct` (Fireworks). Can be set to OpenAI (e.g. `gpt-4.1-mini`) for non-streaming-only use.
- **Streaming (Fireworks):** When `max_tokens` may exceed provider limits (e.g. 4096), the CER uses a **streaming path**: accumulate streamed chunks, then parse JSON. **JSON repair** is applied before `json.loads()` to handle trailing commas and truncation so parsing does not fail on malformed streamed output.
- **OpenAI:** CER with OpenAI runs non-streaming only; no streaming path.
- **Config:** `CER_MODEL`, provider (Fireworks vs OpenAI).

---

## Batch Intent & Families

**File:** `kb_ner_batch_intent.py`.

- **Families:** PRODUCT | PROCEDURE | CLINICAL | OTHER. Kind → family in `kb_ner_intent.FAMILY_MAP`.
- **Output per entity:** `search_term`, `family`. Category-Locked Guard: if family disagrees with kind for PRODUCT/PROCEDURE, search_term reverted to span_text.
- **Config:** `BATCH_INTENT_MODEL`, `BATCH_INTENT_PER_CHUNK`, `BATCH_INTENT_BEFORE_GROUNDING`.

---

## Grounding (Local-Only, Category-Driven)

**Files:** `kb_ner_parallel.py`, `kb_ner_local_search.py`. (Default path is streaming; non-streaming grounding, if used, is wired from the experiment script.)

**Local-only:** When `LOCAL_ONLY=true` (default), no queries to `kb.concepts`; all linking is to `soap.inventory` and `soap.service_master`.

**Category-driven and service_type (Brain NER):**
- Entity's **service_category** and **service_type** (medical | non-medical) → search **services** table; **inventory_category** → search **inventory** table. **General** or missing category is treated as **medical-only** for services (prevents e.g. radiology terms matching grooming SKUs).
- **Billable (dual_sync):** Always search **both** inventory and services; pass `entity_inventory_category`, `entity_service_category`, and `entity_service_type` as `category_hints` / `service_type`; merge results by score; take top 20.
- **Domain:** Used only as **soft** gate (boost/penalty). No hard domain rejection; strict domain guard has been removed.
- **Vector retrieval:** Enabled. Batch embedding prefetch includes span_text, search_term, and hints; on cache miss, on-demand embed when `LOCAL_VECTOR_ON_DEMAND_EMBED=true`.
- **Lexical safety gate:** Removed; judge-selected matches are no longer rejected by post-judge token-overlap checks.

**Clinic ID:** Both inventory and services require `clinic_id`. If `clinic_id` is missing, neither local search runs (no default to 1).

**Pharmacy-Free Zone (Golden Gate):** For **Diagnosis** and **ReasonForVisit** (`SERVICE_ONLY_KINDS` in `kb_ner_routing.py`), local search runs **Services only** (no `search_local_inventory_topk`). **Every** grounding output path in `kb_ner_parallel.py` (batch judge, async auto-bind, Judge-selected, high-certainty auto-link local) sets `local_stock_id` = None for these kinds; only `local_service_id` when a service candidate matches. Prevents "pus"→Lasix, "yeast"→AST KIT type errors.

**0.95 certainty wall:** For Diagnosis and ReasonForVisit, if best match score < **0.95** (`DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD`, env override), the entity is **note-only** (no link; preserved in SOAP, not in billing manifest). Applied in batch and async paths before auto-bind and after Judge selection.

**Judge (kb_ner_disambiguation.py):** Uses **`get_client_for_model(LLM_JUDGE_MODEL)`** only (no cross-provider fallback). **Rule 0 — Category Incompatibility:** REJECT when ENTITY_KIND is Symptom, Diagnosis, or ReasonForVisit but the candidate is a tangible product (Medication, Lab Kit, Consumable). **Rule 2.5 — Symptom / Physical Finding Suppressor:** REJECT symptom/finding mentions (e.g. pus, yeast growth) matching Medications or Lab Reagents unless context explicitly says prescribing/ordering. Form-factor and route-to-form alignment rules remain.

**Dual-sync flow:**
1. Normalize categories and service_type: `entity_inventory_category`, `entity_service_category`, `entity_service_type` (default medical) from entity.
2. Run in parallel (only when `clinic_id` is set): for **non–service-only** kinds, `search_local_inventory_topk(..., category_hints=entity_inventory_category)` and `search_local_services_topk(...)`; for **Diagnosis/ReasonForVisit**, **services only** (no inventory).
3. Merge: `local_candidates = sorted(inv_list + svc_list, key=score, reverse=True)[:20]` (inv_list empty for service-only kinds).
4. For Diagnosis/ReasonForVisit: if best score < 0.95 → note-only; else auto-bind or Judge (only service_id can be set). For other kinds: auto-bind if best local score ≥ threshold; else decision flow → LLM Judge.

**Local search (kb_ner_local_search.py):**
- **Inventory categories (bucket groups):** Deworming, Flea & Tick Treatment, Vaccines, Medication, Diet, Nutrition & Supplements, etc. (see `_CATEGORY_BUCKET_GROUPS`).
- **Service categories:** Medical (Consultation, Surgery, Rehabilitation & Physiotherapy, Lab, Radiology, etc.) vs non-medical (Boarding, Hygiene & Grooming, Training, Behavior, Other Non-Medical). Hard gate uses **service_type** and category hints (see `_MEDICAL_SERVICE_CATEGORY_HINTS`, `_NON_MEDICAL_SERVICE_CATEGORY_HINTS`).
- **Matching:** Trigram + phonetic + vector (batch + on-demand embed); category hard gate restricts to allowed buckets per kind/hints and service_type.

**Kind canonicalization:** `kb_ner_routing.canonicalize_kind()` maps e.g. "Physiotherapy" → "Procedure" so dual_sync and service search run.

---

## Domain-Specific Fallback SKU

**File:** `kb_ner_parallel.py` (post–grounding step).

When **high-stakes** dual_sync entities (e.g. Procedure, Diagnostic, DiagnosticTest, Surgery) remain **unlinked** after normal grounding:

1. **Targeted category search:** Look up Consultation services that match the entity’s **domain_key** (e.g. orthopaedic, cardiology).
2. **One draft SKU per (domain_key, service_id) per visit:** Create a single domain consultation fallback record per domain/service combination; concatenate remarks from all unlinked entities in that group.
3. **Dedupe:** Multiple unlinked entities in the same domain are grouped under one domain consultation SKU: one primary result has `local_service_id` and `grouped_mentions` / `remarks`; others are marked `domain_consultation_fallback_grouped` with `local_service_id = None` and `grouped_under_domain_consultation` pointing to the primary.

This ensures high-stakes mentions still get a billable Consultation placeholder when no direct procedure/diagnostic match exists in the clinic catalog.

---

## Routing Strategy

**File:** `kb_ner_routing.py` → `classify_entity_route()`.

**Kind sets:**
- **DUAL_SYNC_BILLABLE_KINDS:** ReasonForVisit, Medication, Procedure, Diagnostic, Diet, Preventive, ParasiteControl, Diagnosis. These get local search + Judge when routed dual_sync.
- **SERVICE_ONLY_KINDS:** Diagnosis, ReasonForVisit. When dual_sync, they search **Services only** (Pharmacy-Free Zone); never inventory; threshold 0.95 for any link; below 0.95 → note-only.
- **HARD_SKIP_KINDS:** Anatomy, Symptom. Never local or global search; preserved as note-only (no DB/embedding calls).

**Routes (priority order):**
1. **skip_vitals** – Fast-lane VitalSign, generic meta spans.
2. **skip_signalment** – Demographics; never KB-linked.
3. **skip_identity** – Pet/owner/doctor names; never linked.
4. **global_vitals** – Vitals canonicalized against `kb.vitals_registry` only.
5. **dual_sync** – Billable kinds (see above). For **Diagnosis/ReasonForVisit**: local = **services only** + 0.95 certainty wall. For others: local = inventory + services (both); service_type + category hints from Brain NER; domain fallback for high-stakes unlinked.
6. **skip_non_billable** – Anatomy, Symptom (HARD_SKIP_KINDS).
7. **global_direct** – Disabled in production (`GLOBAL_DIRECT_KINDS = []`).

---

## Phase 2: Knowledge Atoms & Dashboard

**Files:** `kb_phase2_integration.py`, `SOAP_notes_billing_phase2_kb_atoms.py`.

- **Atoms:** Extracted from SOAP + manifest; section, kind, assertion_id, intent_context, concept; optional `referenced_entity_id`, `dedup_key` (grounding-aware prompt).
- **Dedupe:** ID-first: by local_service_id / local_stock_id first, then by dedup_key (if present), then same meta-kind + concept similarity ≥ 0.90; section priority Plan > Assessment > Subjective > Objective for representative atom; IDs and source_text merged from group.
- **Constraints:** Plan gate and unlinked/customer-instruction filtering are disabled; atoms are not dropped by those heuristics.
- **Stitching:** Manifest match by referenced_entity_id or by **local_stock_id / local_service_id** first (ID-based); then **manifest_by_normalized_name_id** (Strategy 1b: atom concept + kind → manifest normalized_name so "Cefpodoxime syrup" links to manifest entry with display_name "CefPET Dry Syrup"); then name + kind compatibility (Medicine/Supplement/Nutrition/Preventive/ParasiteControl etc.); when matched, atom inherits manifest kind and IDs.
- **ID-based dashboard linking:** Verification dashboard uses **local_stock_id / local_service_id as primary key** for linking/status (not display-name string matching). Reduces "orphans" when Plan atom text (e.g. "Cefpodoxime syrup") differs from manifest display_name (e.g. "CefPET Dry Syrup").
- **Locus routing:** At-home cues (physiotherapy, swimming, etc.) → reminders_follow_ups.
- **Dashboard:** Procedures, Medications, Diagnostics, Unlinked (module_4), Clinical History, Reminders, Vitals. Unlinked items get top-5 suggestions from local candidates.

**Default model:** `gpt-4.1-mini` (OpenAI). **Config:** `PHASE2_MODEL`, `PHASE2_MODEL_PROVIDER` (default `openai`), `PHASE2_ESCALATE_MODEL`, `PHASE2_START_WITH_SOAP_READY`, `PHASE2_ENABLE_BILLING_MATCHING`, `run_timestamp` passed from experiment for consistent output paths.

**ID-first behavior (v3.1):** Deduplication is ID-first: group by `local_service_id` or `local_stock_id` first (merge regardless of text); then by `dedup_key` (if LLM emits it); then same meta-kind + concept similarity ≥ 0.90. Stitching prefers `referenced_entity_id` and manifest IDs; when a manifest match is found, atom inherits manifest `kind`. Kind compatibility matrix includes Preventive/ParasiteControl with pharmacological kinds. Verification dashboard deduplicates medications by `inventory_id`. Plan gate and manifest-based kind filtering for the prompt are disabled; full schema is used.

---

## Indexes, Embeddings, and Fuzzy Matching

**Extensions (PostgreSQL):** `pg_trgm`, `fuzzystrmatch`, `vector`. Ensured in `kb_ner_db.py` via `ensure_pg_trgm()`, `ensure_fuzzystrmatch()`, `ensure_vector_extension()`.

### Index types and usage

| Type | Extension | Tables / columns | Index names (examples) | Purpose |
|------|-----------|------------------|------------------------|---------|
| **GIN (trigram)** | pg_trgm | `soap.inventory` (item_name, trade_name), `soap.service_master` (procedure_name), `kb.concepts` (preferred_name), `kb.concept_aliases` (alias_text), `kb.vitals_registry` (search_text) | `idx_soap_inventory_item_name_trgm`, `idx_soap_inventory_trade_name_trgm`, `idx_soap_service_master_procedure_name_trgm`, `idx_kb_concepts_preferred_name_trgm`, `idx_kb_concept_aliases_alias_text_trgm`, `idx_kb_vitals_registry_search_text_trgm` | Trigram similarity for fuzzy text match |
| **B-tree** | — | `kb.concepts` (kind, domain_key), `soap.inventory` (domain_key, category), `soap.service_master` (domain_key, category), `kb.vitals_registry` (metaphone_key) | `idx_kb_concepts_kind`, `idx_kb_concepts_domain_key`, `idx_soap_inventory_domain_key`, `idx_soap_service_master_domain_key`, `idx_soap_inventory_domain_category`, `idx_soap_service_master_domain_category`, `idx_kb_vitals_registry_metaphone_key` | Filters and composite lookups; **metaphone_key** is the only phonetic index (exact metaphone / phonetic filter for vitals) |
| **HNSW (vector)** | vector | `kb.concepts` (embedding, embedding_vetbert), `soap.inventory` (vector_embedding_vetbert), `soap.service_master` (vector_embedding_vetbert), `kb.vitals_registry` (embedding), `kb.learned_aliases` (embedding) | `idx_kb_concepts_embedding_vetbert_hnsw`, `idx_soap_inventory_vector_embedding_vetbert_hnsw`, `idx_soap_service_master_vector_embedding_vetbert_hnsw`, `idx_kb_vitals_registry_embedding_hnsw` | Approximate nearest-neighbor (cosine); local search may use `vector_embedding` (OpenAI) when present |

Local search uses **trigram + phonetic** always; **vector** is enabled via batch prefetch (span_text, search_term, hints) and on-demand embedding when cache misses (`LOCAL_VECTOR_ON_DEMAND_EMBED=true`), so vector scoring is used even when `vector_embedding` is not pre-populated on rows. Trigram GIN indexes are created by `ensure_trgm_gin_index()`; HNSW by `ensure_hnsw_index()`. See `ensure_kb_search_indexes()`, `ensure_soft_gate_indexes()`, `ensure_vitals_registry_table()` in `kb_ner_db.py`.

### Embeddings

| Use | Model | Dimensions | Source | Cache |
|-----|--------|------------|--------|--------|
| **Local inventory / service_master** | OpenAI `text-embedding-3-small` | 1536 | `kb_ner_embeddings.embed_text()`; column `vector_embedding` | In-memory LRU, key `(model, text)`; `KB_EMBED_CACHE_MAX` (default 2048) |
| **Optional VetBERT (local)** | havocy28/VetBERT | 768 | Backfill script; column `vector_embedding_vetbert` | — |
| **kb.concepts** | OpenAI (and optional VetBERT) | 1536 / 768 | Same; columns `embedding`, `embedding_vetbert` | Same cache for OpenAI |
| **kb.vitals_registry** | OpenAI | 1536 | Same; column `embedding` | Same |

Embedding client is always OpenAI for embeddings (see `kb_ner_embeddings.py` and `_resolve_embedding_client` in `kb_ner_clients.py`).

### Phonetic matching and phonetic indexes

**Extension:** `fuzzystrmatch` provides `metaphone`, `dmetaphone`, `dmetaphone_alt`, and `levenshtein()`.

**1) Local inventory & service_master (no phonetic index)**  
- **Matching (recall):** Double Metaphone computed **in SQL** on each row for full text and first token: `dmetaphone(...)` + `dmetaphone_alt(...)`; query computes primary/secondary keys similarly.  
- **Score (rerank):** Prefix-aware **Jaro-Winkler** is used in Python reranking (`jaro_winkler_score`) and combined with trigram/vector as `match_score = max(trigram_score, jaro_winkler_score, vector_score)`.  
- **Inclusion:** Row kept if trigram operator `%` matches, or trigram `similarity` passes floor (`LOCAL_TRGM_RECALL_THRESHOLD`, default 0.30), or Double-Metaphone primary/secondary keys match.  
- **Index:** No stored metaphone column or phonetic index on inventory/service_master; GIN trigram indexes on item_name, trade_name, procedure_name are used for the trigram part only.

**2) kb.vitals_registry (phonetic index)**  
- **Stored column:** `metaphone_key` = precomputed `metaphone(lower(search_text), 10)` (and backfilled in seed/upsert).  
- **Index:** **B-tree** on `metaphone_key`: `idx_kb_vitals_registry_metaphone_key` — used for exact metaphone match and efficient filters.  
- **Matching:** Query metaphone `q_mfull` / `q.q_mfirst`; score from `levenshtein(metaphone_key, q.q_mfull)` (and q_mfirst) normalized and scaled to max 0.8. If `metaphone_key` is null, fallback: `metaphone(lower(metric_name), 10)` in SQL.  
- **WHERE:** Row included if trigram match **or** metaphone_key = q_mfull **or** metaphone_key = q_mfirst **or** vector distance < 0.5.

**3) Global KB (kb.concepts / kb.concept_aliases) – no phonetic index**  
- **“Phonetic bucket” (Stage 1):** Actually **trigram** similarity: `similarity(lower(preferred_name), mention) >= sim_thresh` (default 0.2; **escalation** 0.1 for specialty when Pass 1 returns 0 rows). Uses GIN trigram indexes on preferred_name and alias_text.  
- **Metaphone fallback:** When **specialty** and trigram bucket returns 0 rows: `ensure_fuzzystrmatch` then pure metaphone matching — `metaphone(lower(preferred_name), 10) = q.q_mfull` or `= q.q_mfirst`, **or** normalized `levenshtein(metaphone(...), q) < 0.65` on concepts and concept_aliases. No stored metaphone column; all computed in SQL.  
- **Batch path:** `_batch_trigram_phonetic_scores()`: same metaphone + Levenshtein formula (0.8 * (1 - norm_lev)) for `phonetic_score` per candidate name.

**Summary of phonetic indexes**

| Table | Phonetic column | Index | Notes |
|-------|------------------|--------|--------|
| **kb.vitals_registry** | `metaphone_key` | B-tree `idx_kb_vitals_registry_metaphone_key` | Only stored phonetic key + index in the system |
| soap.inventory | — | — | Metaphone computed in SQL from item_name, trade_name |
| soap.service_master | — | — | Metaphone computed in SQL from procedure_name |
| kb.concepts | — | — | Metaphone fallback computed in SQL from preferred_name |
| kb.concept_aliases | — | — | Metaphone fallback computed in SQL from alias_text |

**Ensured:** `ensure_fuzzystrmatch(conn)` before local search, vitals registry search, and global metaphone fallback.

### Fuzzy matching rules (local search)

- **Trigram retrieval:** `%` operator is used as primary trigram retrieval signal (`lower(name) % query`) plus explicit similarity floor via `LOCAL_TRGM_RECALL_THRESHOLD` (default 0.30).
- **Phonetic retrieval:** Double-Metaphone primary/secondary key equality (`dmetaphone`, `dmetaphone_alt`) on full and first-token forms.
- **Reranking:** Jaro-Winkler (prefix-aware) is used for lexical reranking; `jaro_winkler_score` is logged per candidate.
- **Vector:** When `vector_embedding` is not null, `(1 - LEAST(embedding <=> query_vec, 1.0))` as vector_score; optional WHERE `embedding <=> query_vec < 0.5` to restrict to close neighbors.
- **match_score (per candidate):** `max(trigram_score, jaro_winkler_score, vector_score)`; then in Python candidates with `match_score < 0.30` are dropped.
- **Final ranking:** `final_score = (match_score * SOFT_GATE_LOCAL_BASE_WEIGHT) + domain_boost + category_boost + suggestion_boost` when domain soft gate is on (see `docs/LOCAL_GROUNDING_SEARCH_AND_RETRIEVAL.md`).
- **Default threshold** for “best match” in guard/validation: 0.50 (parameter `threshold` in `search_local_inventory_topk` / `search_local_services_topk`).

**Category:** SQL normalizes category as `LOWER(TRIM(REPLACE(COALESCE(category,''), '&', 'and')))`; bucket groups in `_CATEGORY_BUCKET_GROUPS` match that form. Hard gate restricts search to allowed buckets per kind + entity category hints.

**Clinic ID:** Local inventory and local services both require `clinic_id`. If `clinic_id` is missing, both searches are skipped (no default to 1). Location: `soap.inventory.location_id`; service_master may not have location (clinic_id still gates whether search runs).

---

## Complete Codebase

| File | Purpose |
|------|---------|
| **SOAP_notes_phase1_experiment.py** | Orchestration: audio → transcript → Super-Pass → Batch Intent → Grounding → SOAP → Injection → Phase 2; entry `generate_soap_note_from_audio_async()` |
| **kb_ner_super_pass.py** | Super-Pass: cleaning + NER; outputs entities with inventory_category, service_category, service_type; default model Llama v3p3 70B; streaming and chunked modes |
| **kb_ner_batch_intent.py** | Batch Intent: search_term + family per entity; Category-Locked Guard |
| **kb_ner_parallel.py** | Grounding (streaming): process_single_entity_async, process_entity_by_route_async; dual_sync = _local_inventory + _local_services (for non–service-only kinds) or _local_services only for Diagnosis/ReasonForVisit (Pharmacy-Free Zone); **every** output path (batch judge, auto-bind, Judge-selected, high-certainty auto-link local) sets local_stock_id = None for Diagnosis/ReasonForVisit; 0.95 certainty wall (note-only if score < 0.95); batch prefetch + on-demand embed; domain consultation fallback + dedupe for high-stakes unlinked |
| **kb_ner_local_search.py** | Local search: search_local_inventory_topk, search_local_services_topk; category bucket groups + service_type (medical/non-medical); trigram + phonetic + vector |
| **kb_ner_routing.py** | Routing: canonicalize_kind, classify_entity_route; DUAL_SYNC_BILLABLE_KINDS, SERVICE_ONLY_KINDS (Diagnosis, ReasonForVisit), HARD_SKIP_KINDS (Anatomy, Symptom), DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD (0.95), is_services_only_kind(); GLOBAL_DIRECT_KINDS=[] |
| **kb_ner_global_search.py** | Global KB search (stubbed when LOCAL_ONLY=true): search_global_topk, run_batch_global_vector_search, kb_lookup_* return [] or no-op |
| **kb_ner_disambiguation.py** | Decision flow and LLM Judge: apply_decision_flow, run_single_batch_llm_judge; Rule 0 Category Incompatibility (finding vs product → REJECT), Rule 2.5 Symptom/Physical Finding Suppressor; form-factor/route alignment; Judge uses get_client_for_model(LLM_JUDGE_MODEL) only (no cross-provider fallback) |
| **kb_ner_db.py** | DB: acquire_pg_conn, pg_conn_ctx, search_vitals_registry_topk, ensure_fuzzystrmatch |
| **kb_ner_embeddings.py** | Embeddings: embed_texts, to_pgvector_literal; cache |
| **kb_ner_skeleton_parser.py** | Parse Brain NER skeleton_list output into entity dicts (inventory_category, service_category, service_type) |
| **kb_ner_clinical_entity_resolver.py** | CER: resolve entities to billing items; Fireworks streaming + JSON repair when large output; default model Llama v3p3 70B |
| **kb_ner_constraint_injection.py** | Constraint + truth injection: enforce_sectional_truth_injection; manifest corrections |
| **kb_anchor_span.py** | Anchor IDs: ensure_anchor_ids_on_manifest (E1, E2, …) |
| **kb_anchor_resolve.py** | Re-grounding: resolve_anchor_update by anchor_id + new text |
| **kb_phase2_integration.py** | Phase 2: extract_knowledge_atoms_async, dedupe, Plan gate, enrich_atoms_with_manifest_ids |
| **SOAP_notes_billing_phase2_kb_atoms.py** | Phase 2: build_knowledge_atom_prompt, parse SOAP sections, verification dashboard, unlinked suggestions |
| **kb_ner_chunk_parallel.py** | Chunked long transcripts: chunking, per-chunk timeout, step-wise logging |
| **kb_ner_intent.py** | Intent families: FAMILY_MAP (kind → PRODUCT/PROCEDURE/CLINICAL/OTHER) |
| **kb_ner_intent_guards.py** | Intent guards: ground_clinical_terms, Category-Locked Guard |
| **kb_ner_domain.py** | Domain detection: detect_domain (for optional domain filter/boost) |
| **kb_ner_clients.py** | LLM/embedding clients: OpenAI, Fireworks |
| **kb_ner_enrichment.py** | Enrichment helpers (optional path) |
| **kb_ner_extraction.py** | Extraction helpers (optional path) |
| **kb_ner_lexical_harvester.py** | Lexical harvester (optional): load terms from DB for search |
| **kb_ner_2phase_integration.py** | 2-phase integration (optional when not using Super-Pass) |
| **kb_domain_affinity.py** | Domain affinity (optional) |
| **kb_ner_manifest_guardrail.py** | Manifest guardrail (optional) |

**Legacy (moved):** Backfills, audits, one-off scripts, and old pipeline code live under `legacy/python/`, `legacy/scripts/`, `legacy/docs/`. See `legacy/README.md`.

---

## Function Reference

| Function | File | Purpose |
|----------|------|---------|
| `generate_soap_note_from_audio_async` | SOAP_notes_phase1_experiment | Entry: audio → grounded SOAP + Phase 2 |
| `transcribe_audio` / `transcribe_audio_fireworks_streaming` | SOAP_notes_phase1_experiment | Step 1: audio → raw transcript (Fireworks Whisper; retries 502/503 with backoff) |
| `super_pass_cleaning_and_ner_streaming` / `super_pass_cleaning_and_ner` | kb_ner_super_pass | Step 2: clean + NER; inventory_category, service_category, service_type |
| `run_batch_intent` | kb_ner_batch_intent | Step 2b: search_term + family |
| `run_step_2_3_normalization` | (non-streaming path, if enabled) | Step 2.3 (non-streaming): full grounding; default is streaming via kb_ner_parallel |
| `process_single_entity_async` / `process_entity_by_route_async` | kb_ner_parallel | Step 2.3 (streaming): per-entity grounding; dual_sync = _local_inventory + _local_services |
| `classify_entity_route` | kb_ner_routing | Route: skip_vitals | global_vitals | skip_signalment | skip_identity | dual_sync | global_direct |
| `classify_procedure_role` | kb_ner_routing | Procedure role (Performed/Prescribed/etc.) |
| `search_local_inventory_topk` / `search_local_services_topk` | kb_ner_local_search | Local search with category_hints and service_type; trigram + phonetic + vector (batch + on-demand embed) |
| `apply_decision_flow` / LLM Judge | kb_ner_disambiguation | Option A/B/C; Judge uses search_term |
| `generate_soap_with_grounding` | SOAP_notes_phase1_experiment | Step 3: SOAP note |
| `apply_constraint_based_injection` | kb_ner_constraint_injection | Step 4: constraint injection |
| `apply_manifest_corrections_to_soap_json` | SOAP_notes_phase1_experiment | Step 4: truth injection; anchor tags |
| `ensure_anchor_ids_on_manifest` | kb_anchor_span | Assign E1, E2, … |
| `extract_knowledge_atoms_async` | kb_phase2_integration | Phase 2: atoms, dedupe, dashboard |
| `search_vitals_registry_topk` | kb_ner_db | global_vitals route |
| `embed_texts` | kb_ner_embeddings | Batch embeddings; cache |
| `canonicalize_kind` | kb_ner_routing | Normalize kind (e.g. Physiotherapy → Procedure) |

---

## Configuration & Environment

**Pipeline:**
- `USE_SUPER_PASS`, `SUPER_PASS_STREAMING`, `SUPER_PASS_MODEL` (default `accounts/fireworks/models/llama-v3p3-70b-instruct`)
- `CER_MODEL` (default Llama v3p3 70B on Fireworks); CER streaming + JSON repair when Fireworks and large output
- `BATCH_EMBED_PER_CHUNK`, `BATCH_INTENT_BEFORE_GROUNDING`, `BATCH_INTENT_PER_CHUNK`, `BATCH_INTENT_MODEL`
- `COMBINED_CLEAN_NER_BATCH_INTENT` (Unified Mega-Pass)
- `CLINIC_ID` (default 1), `VISIT_ID`
- `KB_MAX_PARALLEL_ENTITIES` (default 12)
- **`LOCAL_ONLY`** (default true): no global KB; local inventory + services only
- **`LOCAL_VECTOR_ON_DEMAND_EMBED`** (default true): on-demand embedding for local search when cache misses; vector retrieval enabled

**Grounding:**
- `auto_bind_threshold` (0.92), `llm_judge_threshold` (0.55)
- `LLM_JUDGE_MODEL` (Judge uses this model's provider only; no cross-provider fallback)
- `DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD` (default 0.95): Diagnosis/ReasonForVisit below this score → note-only (no link)

**SOAP / Phase 2:**
- `SOAP_MODEL`, `ANCHOR_MAPPING_SOAP`, `ANCHOR_SPAN_OUTPUT`
- `PHASE2_MODEL` (default `gpt-4.1-mini`), `PHASE2_MODEL_PROVIDER` (default `openai`), `PHASE2_ESCALATE_MODEL`, `PHASE2_START_WITH_SOAP_READY`, `PHASE2_ENABLE_BILLING_MATCHING`
- `CONSTRAINT_INJECTION_SINGLE_PASS`, `CONSTRAINT_INJECTION_MODE`, `CONSTRAINT_INJECTION_MODEL`

**Transcription:**
- `FIREWORKS_API_KEY`, `FIREWORKS_MODEL_NAME` (whisper-v3-turbo), `FIREWORKS_SAMPLE_RATE` (16000)
- **Resilience:** `transcribe_audio()` retries on 502/503 (up to 3 attempts, backoff 2s/4s/8s); on failure raises a short user-facing message (no HTML dump). Tips for 502/503 shown in exception handler.

**Chunked / long transcripts:**
- `LONG_TRANSCRIPT_THRESHOLD_CHARS`, per-chunk timeout (e.g. 180s) in kb_ner_chunk_parallel

---

## Database & Tables

**Extensions:** `vector` (pgvector), `pg_trgm`, `fuzzystrmatch`. See [Indexes, Embeddings, and Fuzzy Matching](#indexes-embeddings-and-fuzzy-matching) for index types and names.

**Local (clinic):**
- **soap.inventory** – Products (medication, vaccines, parasite control, diet, etc.); `location_id`, category, item_name, trade_name, stock_id; optional `vector_embedding` (1536, OpenAI), optional `vector_embedding_vetbert` (768); `domain_key` for soft gate. GIN trigram on item_name, trade_name; B-tree on domain_key, (domain_key, category); optional HNSW on vector_embedding_vetbert.
- **soap.service_master** – Services (consultation, surgery, rehabilitation & physiotherapy, lab, etc.); category, procedure_name, service_id; optional `vector_embedding` (1536), optional `vector_embedding_vetbert` (768); `domain_key`. GIN trigram on procedure_name; B-tree on domain_key, (domain_key, category); optional HNSW on vector_embedding_vetbert.

**Global (disabled when LOCAL_ONLY=true):**
- **kb.concepts** – concept_id, preferred_name, kind, definition, embedding (1536), embedding_vetbert (768), domain_key. Not queried when LOCAL_ONLY. Indexes: B-tree (kind, domain_key), GIN (preferred_name), HNSW (embedding_vetbert).
- **kb.vitals_registry** – Used for global_vitals route (vital metric names, synonyms, search_text, metaphone_key, embedding). Trigram + phonetic + vector search; indexes: GIN (search_text), B-tree (metaphone_key), HNSW (embedding).

**Setup:** Indexes are ensured by `kb_ner_db.py` (ensure_pg_trgm, ensure_fuzzystrmatch, ensure_vector_extension, ensure_trgm_gin_index, ensure_hnsw_index, ensure_kb_search_indexes, ensure_soft_gate_indexes, ensure_vitals_registry_table). Optional backfills for vector_embedding / VetBERT; see `legacy/README.md` and `legacy/python/backfill_soap_domain_and_vetbert.py`.

---

## Documentation & Legacy

| Location | Content |
|----------|---------|
| **UNIFIED_CLEANING_AND_NER_PROMPT.md** | **Required for pipeline.** Unified cleaning + NER prompt; loaded by `kb_ner_super_pass.py` from project root. Do not move to legacy. |
| **BRAIN_NER_PROMPT_UPDATED.md** (project root) | Canonical Brain NER prompt: entity kinds, inventory_category list, service_category list, skeleton format (13 fields), query_expansion (13th field), form-factor preservation in normalized_name |
| **LOCAL_GROUNDING_SEARCH_AND_RETRIEVAL.md** | Local grounding: retrieval (trigram, phonetic, vector), scoring (domain, category, suggestion/hints), decision flow (auto-bind, Option A/B/C). See also GROUNDED_CONCEPTS_QUERY_EXPANSION_DOCUMENTATION.md. |
| **GROUNDED_CONCEPTS_QUERY_EXPANSION_DOCUMENTATION.md** | Query expansion, hints, form-factor & route-to-form alignment, production safeguards (symptom suppressor, Golden Gate for Diagnosis/ReasonForVisit, ID-based dashboard), Judge rules, SOAP prompt (hints/query_expansion), summary table of files |
| **MASTER_DOCUMENTATION.md** | This file: flow, codebase, grounding (incl. Pharmacy-Free Zone, 0.95 wall), routing, indexes/embeddings/fuzzy, config |
| **README.md** | Project overview and run instructions |
| **legacy/docs/** | Audit reports, RCA docs, latency/quality reports, old pipeline flows, prompt comparisons, implementation plans (archived); copy of UNIFIED_CLEANING_AND_NER_PROMPT.md kept there for reference only |
| **legacy/python/** | Backfills, run_pipeline_to_intent, audit scripts, one-off scripts (check_local_inventory_spirocoxin, test_brain_ner_on_cleaned, debug_ontology_harvest, etc.) |
| **legacy/scripts/** | check_service_master_physiotherapy, check_local_inventory_bravecto, run_ortolani_rca_checks, kb_domain_auditor, etc. |
| **legacy/README.md** | List of legacy files and how to run them |

---

**End of Master Documentation (v3.4)**

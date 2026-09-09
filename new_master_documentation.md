# SOAP Notes Phase 1 – Master Documentation

**Version**: 4.0
**Last Updated**: 2026-09-08
**Purpose**: Single source of truth for the SOAP Phase 1 pipeline: end-to-end flow, codebase, grounding (local-only, category-driven, medical/non-medical service type, domain fallback SKU), production safeguards (Golden Gate, symptom suppressor, Judge category incompatibility, ID-based dashboard), CER streaming, chunk-parallel Super-Pass, vector retrieval, Conclusion-section SOAP generation, pipeline observability (timing + health flags), Phase 2, the Doctor UI application, benchmarking/evaluation tooling, deployment, and configuration.
**Supersedes**: `MASTER_DOCUMENTATION (1).md` (v3.4, 2026-02-25) in content. That file is kept in the repo for historical reference only — this file is the current source of truth and should be updated going forward.
**Canonical prompt**: `BRAIN_NER_PROMPT_UPDATED.md` (project root) and `UNIFIED_CLEANING_AND_NER_PROMPT.md` (includes service_type). Historical and audit docs are in `legacy/docs/`.
**Local grounding detail**: `docs/LOCAL_GROUNDING_SEARCH_AND_RETRIEVAL.md` (retrieval, scoring, hints, decision flow).
**Grounded concepts & production safeguards**: `GROUNDED_CONCEPTS_QUERY_EXPANSION_DOCUMENTATION.md` (query expansion, hints, form-factor, symptom suppressor, Golden Gate for Diagnosis/ReasonForVisit, ID-based dashboard).
**Doctor-facing app**: `docs/DOCTOR_UI.md` (quick start) and [Doctor UI (Streamlit App)](#doctor-ui-streamlit-app) below (full architecture).
**Benchmarking**: `docs/ASR_BENCHMARK.md`, `docs/PIPELINE_BENCHMARK.md`, and [Benchmarking & Evaluation](#benchmarking--evaluation) below.
**Deployment**: `DEPLOY.md` and [Deployment](#deployment) below.

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
10. [SOAP Generation (Sections, Formatting, Conclusion)](#soap-generation-sections-formatting-conclusion)
11. [Phase 2: Knowledge Atoms & Dashboard](#phase-2-knowledge-atoms--dashboard)
12. [Pipeline Observability (Timing & Health Flags)](#pipeline-observability-timing--health-flags)
13. [Indexes, Embeddings, and Fuzzy Matching](#indexes-embeddings-and-fuzzy-matching)
14. [Complete Codebase](#complete-codebase)
15. [Function Reference](#function-reference)
16. [Configuration & Environment](#configuration--environment)
17. [Database & Tables](#database--tables)
18. [Doctor UI (Streamlit App)](#doctor-ui-streamlit-app)
19. [Benchmarking & Evaluation](#benchmarking--evaluation)
20. [Deployment](#deployment)
21. [Documentation & Legacy](#documentation--legacy)

---



## Executive Summary

The pipeline turns **raw audio** (or typed text) of veterinary consultations into **grounded SOAP notes** with entity linking to **local inventory** and **local services** only (no global KB in production), plus a structured **Phase 2 knowledge-atom / billing dashboard**, surfaced to vets through a **Streamlit doctor-facing app** (`doctor_ui/`).

**Flow (default):**

1. **Transcription** – Pluggable ASR via `ASR_PROVIDER` / `ASR_MODEL` (default: Deepgram **nova-3**; Fireworks **whisper-v3-turbo** available). See `docs/ASR_BENCHMARK.md` and `asr_providers.py`.
2. **Super-Pass** – One LLM call (or, for long transcripts, a **chunk-parallel** set of calls — see `kb_ner_chunk_parallel.py`): clean transcript + extract entities; outputs `inventory_category`, `service_category`, and `service_type` (medical | non-medical) per entity. Default model: **accounts/fireworks/models/llama-v3p3-70b-instruct**. Brain NER prompt in `docs/BRAIN_NER_PROMPT_UPDATED.md` and `UNIFIED_CLEANING_AND_NER_PROMPT.md`.
3. **CER (Clinical Entity Resolver)** – Consolidates entities to billing-only; default model **accounts/fireworks/models/llama-v3p3-70b-instruct**. On Fireworks, streaming path when output may exceed 4096 tokens; OpenAI non-streaming only. JSON repair for streamed output (trailing commas, truncation). Can be skipped below a character threshold (`CER_SKIP_UNDER_CHARS`) or disabled (`ENABLE_CER`).
4. **Batch Intent** – One LLM call: assign `search_term` and `family` per entity (ASR correction, category). Category-Locked Guard: family must not contradict kind for PRODUCT/PROCEDURE.
5. **Grounding** – Per entity: route (skip_vitals, global_vitals, skip_signalment, skip_identity, **dual_sync**, global_direct). **Dual-sync (billable)**: search **both** `soap.inventory` and `soap.service_master` with **service_type** (medical vs non-medical) and category hints; **vector retrieval enabled** (batch + on-demand embed); domain soft-gating now driven by an **embedding-based domain affinity** model (`kb_domain_affinity.py`), not hardcoded word lists. **Domain fallback**: when high-stakes entities remain unlinked, draft one domain consultation SKU per (domain_key, service_id) per visit with grouped remarks; dedupe so multiple unlinked same-domain entities attach to one SKU. Entities that stay unlinked after grounding get their RAG candidate suggestions attached for dashboard display (`grounding_suggestion_enrichment.py`). No global KB when `LOCAL_ONLY=true` (default).
6. **SOAP** – Generate note from cleaned transcript; Plan/KeyIssues/AbnormalFindings/CustomerInstructions/Reminders are now **numbered lists** (one item per line, not paragraphs); Reminders is a single consolidated field; CustomerInstructions is scoped to **at-home care only** (rechecks/follow-ups go to Reminders); a new mandatory **Conclusion** section is a stand-alone 8–12 sentence (~150–300 word) clinical-prose executive summary that must weave Plan actions (named meds/procedures/diagnostics + follow-up timing) into prose rather than repeating the Plan list; entity-injected Plan (uses manifest terms, no generic "as prescribed").
7. **Injection** – Constraint + truth: replace span_text → display_name; optional Anchor-Span `[[E1|term]]`.
8. **Phase 2** – Knowledge atoms; default model **gpt-4.1-mini** (OpenAI). ID-first dedupe, locus routing (at-home → Reminders), Primary/Secondary diagnosis split, structured vitals schema, medicine-recommendation enrichment for issues without a matched medication atom, verification/billing dashboard.
9. **Observability** – Every run produces a `PipelineTimer` timing report and a `build_pipeline_flags()` health report (STEP1–PHASE2 flags, overall healthy/degraded/failing), both surfaced in the Doctor UI.
10. **Doctor UI** – Vets capture consultations (typed or voice) in a Streamlit app, generate the SOAP note + billing dashboard, review/edit line items, and browse history — see [Doctor UI (Streamlit App)](#doctor-ui-streamlit-app).

**Design choices:**

- **Service type (medical / non-medical)**: Brain NER emits `service_type`; default for billable entities is **medical**. Service hard gate restricts to medical-only categories when entity is medical (Consultation, Surgery, Lab, Radiology, etc.); non-medical uses Boarding, Hygiene & Grooming, Training, Behavior, Other Non-Medical. **General** or missing category is treated as medical-only for services to prevent cross-category hallucinations (e.g. radiology term → grooming SKU).
- **Category-driven search**: Entity's `service_category` and `service_type` → search `soap.service_master`; `inventory_category` → search `soap.inventory`. Billable entities search **both** and merge; category buckets respect medical vs non-medical.
- **Domain**: Used as a **soft** gate (boost/penalty only). The domain detector actually wired into the grounding hot path (`kb_ner_parallel.py`) is still `kb_ner_domain.detect_domain()` — fast, first-1000-chars keyword matching against `DOMAIN_KEYWORDS`/`DOMAIN_PRIMERS`, unchanged in behavior. A separate embedding-based module, `kb_domain_affinity.py` (`DOMAIN_ANCHORS`, `get_top_domain()`, cosine similarity), has existed since the initial commit and is used narrowly by the LLM Judge (`kb_ner_disambiguation.is_domain_relevant_for_phonetic_threshold()`) and by the (disabled-under-`LOCAL_ONLY`) global search — it is **not** the domain source for local inventory/service scoring. No hard domain rejection either way.
- **Vector retrieval**: Enabled for local inventory and service_master. Batch embeddings include span_text, search_term, and hints; on-demand embed when cache miss (`LOCAL_VECTOR_ON_DEMAND_EMBED=true`).
- **Auto-bind threshold**: `GROUNDING_AUTO_BIND_THRESHOLD` default is **0.92**. It was briefly lowered to 0.85 (commit "Improvement in RAG and Enriched SOAP") to auto-link more entities without an LLM Judge call, then reverted back to 0.92 (in code default, `.env`, and `.env.deploy.example`) after output quality at 0.85 proved unsatisfactory.
- **Physiotherapy / Rehabilitation**: Kind "Physiotherapy" canonicalizes to "Procedure"; service category bucket includes "Rehabilitation & Physiotherapy", "Physiotherapy and Rehabilitation".
- **Local-only**: Global KB disabled by default (`LOCAL_ONLY=true`). All linking to clinic inventory and service_master.
- **Lexical safety gate**: Removed; judge-selected matches (e.g. "X-ray result" → "XRAY") are no longer rejected by token-overlap checks.
- **Production safeguards:** **Pharmacy-Free Zone** — Diagnosis and ReasonForVisit search Services only (never Pharmacy); **0.95 certainty wall** for those kinds (below 0.95 → note-only). **Judge:** Category Incompatibility (finding vs product → REJECT), Symptom/Physical Finding Suppressor. **Dashboard:** ID-based linking (local_stock_id / local_service_id as primary key; manifest_by_normalized_name_id reduces orphans). See `GROUNDED_CONCEPTS_QUERY_EXPANSION_DOCUMENTATION.md` §8.
- **Diagnosis is now split**: Doctor UI / SOAP schema distinguish **PrimaryDiagnosis** (mandatory, root condition(s)) from **SecondaryDiagnosis** (optional, contributing/comorbid, "X secondary to Y"), replacing the old single `DifferentialDiagnosis` field (legacy values are migrated via `doctor_ui/diagnosis_schema.migrate_legacy_diagnosis()`).
- **Vitals are structured**: An 11-field canonical vitals template (`doctor_ui/vitals_schema.py`) drives both the SOAP extraction prompt and the UI's Vitals table.

**Entry:** `SOAP_notes_phase1_experiment.generate_soap_note_from_audio_async()` (voice path) / `generate_soap_note_from_transcript_async()` (typed-text path).

---



## Complete Pipeline Flow

```
Step 1: Audio → Raw transcript (ASR_PROVIDER / ASR_MODEL; default Deepgram nova-3)
Step 2: Super-Pass → Cleaned transcript + extracted_entities (inventory_category, service_category, service_type per entity; default model Llama v3p3 70B; chunk-parallel for long transcripts)
Step 2a: CER (optional) → Resolve to billing entities (streaming when Fireworks + large output; JSON repair; skippable under CER_SKIP_UNDER_CHARS)
Step 2b: Batch Intent → search_term + family per entity
Step 2.3: Grounding → Entity manifest (dual_sync: local inventory + local services; service_type + category hints; vector retrieval; embedding-based domain soft-gate; domain fallback SKU; unlinked-suggestion enrichment; merge; Judge if needed)
Step 3: SOAP generation → Note with numbered-list Plan/KeyIssues/AbnormalFindings/CustomerInstructions/Reminders + mandatory prose Conclusion; entity-injected Plan
Step 4: Constraint + Truth injection → Grounded SOAP (optional Anchor-Span)
Step 5: Phase 2 → Knowledge atoms (default gpt-4.1-mini), dedupe, locus routing, medicine-recommendation enrichment, verification/billing dashboard
Throughout: PipelineTimer records per-stage timing; build_pipeline_flags() records health flags — both written alongside run artifacts and shown in the Doctor UI.
```

**Fallback:** `USE_SUPER_PASS=false` → separate cleaning then 2-Phase NER + grounding.

---



## Default Execution Path

1. **Startup:** Postgres pool init, `clinic_id` / `visit_id` from env (`CLINIC_ID`, `VISIT_ID`; default `clinic_id=1`). When invoked from the Doctor UI, `doctor_ui/pipeline_runner.py` first probes Postgres reachability and sets `SKIP_BILLING_PIPELINE` / `SKIP_PHASE2` if it's unreachable, so SOAP generation still succeeds without inventory grounding.
2. **Step 1:** `asr_providers.transcribe()` (default Deepgram nova-3) or Fireworks streaming when `TRANSCRIPTION_STREAMING=true` + `ASR_PROVIDER=fireworks` → raw transcript; saves `step1_raw_transcription.txt` + `step1_asr_metadata.json`.
3. **Step 2 – Super-Pass (streaming default):** `super_pass_cleaning_and_ner_streaming()`; entities with `inventory_category`, `service_category`, `service_type`; default model Llama v3p3 70B; per chunk: batch embed → Batch Intent → dispatch to streaming grounding. Two distinct parallelization paths exist in `kb_ner_chunk_parallel.py`: (a) when `CHUNK_PARALLEL_ENABLED=true` and the raw transcript exceeds `CHUNK_SIZE`, `process_chunks_parallel()` splits it into overlapping chunks (`compute_adaptive_chunk_params`, `chunk_raw_transcript_with_overlap`), extracts signalment once and injects it into each chunk (`extract_signalment` / `inject_signalment_header`), processes chunks in parallel, then merges (`merge_cleaned_segments`, `deduplicate_entities`, `filter_non_actionable_entities`, `correct_entity_kinds`) — this runs after ASR completes; (b) `IncrementalAsrSuperPass` instead overlaps Super-Pass cleaning with ASR itself — while Deepgram/Fireworks is still streaming audio, it feeds stable transcript prefixes into Super-Pass early (`.feed()`), then `.finalize(raw_transcription)` reconciles the incremental results once ASR completes, so cleaning work happens concurrently with transcription rather than after it.
4. **Step 2.3 – Grounding:** `process_single_entity_async()` (streaming) or `run_step_2_3_normalization()` (non-streaming). Dual-sync: `search_local_inventory_topk(..., category_hints=...)` and `search_local_services_topk(..., category_hints=..., service_type=entity_service_type)` in parallel; vector retrieval (batch + on-demand embed); merge; auto-bind (threshold `GROUNDING_AUTO_BIND_THRESHOLD`, default 0.92) or LLM Judge; domain consultation fallback for high-stakes unlinked; `enrich_unlinked_manifest_suggestions()` attaches top RAG candidates to entities that stayed unlinked so the dashboard can still show suggestions.
5. **Step 3 – SOAP:** `generate_soap_with_grounding()`; manifest with anchor_id (E1, E2, …) when `ANCHOR_MAPPING_SOAP=true`; prompt includes `VITALS_SECTION_GUIDANCE` + structured vitals template, `DIAGNOSIS_SECTION_GUIDANCE` (Primary/Secondary), and `CONCLUSION_SECTION_GUIDANCE` (mandatory prose summary).
6. **Step 4 – Injection:** `apply_constraint_based_injection()` + `apply_manifest_corrections_to_soap_json()`.
7. **Phase 2:** `extract_knowledge_atoms_async()`; dedupe; locus routing; medicine-recommendation enrichment; verification/billing dashboard.
8. **Throughout:** a `PipelineTimer` instance is threaded through steps 2–7 (marks like `step2_super_pass_start/done`, `step2_brain_ner_start/done`, `step2_cer_skipped/attempted`, `injection_start/done`, `phase2_start`, plus Phase 2's own internal timings merged in via `merge_phase2()`); `build_pipeline_flags()` inspects the run's artifacts afterward to produce a health report.

**Combined path:** `COMBINED_CLEAN_NER_BATCH_INTENT=true` → Clean + NER + Intent in one LLM call; then grounding with same local-only, category-driven search.

---



## Super-Pass (Cleaning + NER)

**File:** `kb_ner_super_pass.py`.
**Prompt:** See `BRAIN_NER_PROMPT_UPDATED.md` (project root) and `UNIFIED_CLEANING_AND_NER_PROMPT.md`.

- **Cleaning:** Verbatim clinical preservation; no correction of drug/procedure names (grounding handles ASR).
- **Extraction:** Inclusive; tests/measurements/angles extracted even if misspelled (e.g. "Noble angle" → Batch Intent can correct to Norberg).
- **Output per entity:** `normalized_name`, `kind`, `domain`, `inventory_category`, `service_category`, `service_type` (medical | non-medical; default medical for billable), `correctness_probability`, `suggestion_probability`, `hints`.
- **Default model:** `accounts/fireworks/models/llama-v3p3-70b-instruct` (Fireworks).
- **Config:** `SUPER_PASS_MODEL`, `SUPER_PASS_STREAMING`, `LONG_TRANSCRIPT_THRESHOLD_CHARS`, `BATCH_EMBED_PER_CHUNK`.

### Chunk-parallel mode for long transcripts

**File:** `kb_ner_chunk_parallel.py`.

For long recordings, Super-Pass cleaning + NER can run as parallel chunk workers instead of one large call, via `process_chunks_parallel()` (present since the initial commit):

- `compute_adaptive_chunk_params()` – picks chunk size/overlap/parallelism from transcript length.
- `chunk_raw_transcript_with_overlap()` – splits with overlapping context windows so entities spanning a chunk boundary aren't lost.
- `extract_signalment()` / `inject_signalment_header()` – pulls patient signalment (species/breed/age/sex) once from the full transcript and injects it into every chunk's prompt so per-chunk cleaning has the same patient context.
- `merge_cleaned_segments()`, `deduplicate_entities()`, `filter_non_actionable_entities()`, `correct_entity_kinds()` – recombine chunk outputs into one cleaned transcript + entity list, removing cross-chunk duplicate entities (from the overlap) and non-actionable noise.
- **Config:** `CHUNK_PARALLEL_ENABLED`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_MIN_PARALLEL`, `CHUNK_MAX_PARALLEL`, `CHUNK_TARGET_SIZE`, `CHUNK_SINGLE_THRESHOLD`, `CHUNK_WORKER_TIMEOUT_SEC`.

**New in the "Phase 2" commit — `IncrementalAsrSuperPass`:** a separate mechanism in the same file that overlaps cleaning with transcription itself rather than chunking after the fact. While ASR is still streaming audio, `.feed(partial_text, threshold_chars)` is called from the ASR callback thread to run Super-Pass on stable transcript prefixes as they arrive; `.finalize(raw_transcription)` reconciles everything once ASR completes. Wired into `SOAP_notes_phase1_experiment.py` as `asr_overlap_helper`.

Related long-transcript safeguards in `long_transcript_utils.py`: chunking + `[LONG_TRANSCRIPT_SUMMARY]` wrap/extract helpers that keep the SOAP prompt from overflowing context on very long transcripts (`FORCE_LONG_TRANSCRIPT_MODE`, `SOAP_MAX_TRANSCRIPT_CHARS_IN_PROMPT`, `SOAP_EXCERPT_MAX_CHARS`).

---



## CER (Clinical Entity Resolver)

**File:** `kb_ner_clinical_entity_resolver.py`.

- **Purpose:** Resolve extracted entities into a consolidated set of billing-relevant actionable items (e.g. for Phase 2 or downstream billing).
- **Default model:** `accounts/fireworks/models/llama-v3p3-70b-instruct` (Fireworks). Can be set to OpenAI (e.g. `gpt-4.1-mini`) for non-streaming-only use.
- **Streaming (Fireworks):** When `max_tokens` may exceed provider limits (e.g. 4096), the CER uses a **streaming path**: accumulate streamed chunks, then parse JSON. **JSON repair** is applied before `json.loads()` to handle trailing commas and truncation so parsing does not fail on malformed streamed output.
- **OpenAI:** CER with OpenAI runs non-streaming only; no streaming path.
- **Skip path:** Below `CER_SKIP_UNDER_CHARS` characters (short transcripts), or when `ENABLE_CER=false`, CER is skipped entirely and Super-Pass entities pass straight to Batch Intent; `PipelineTimer` records this as `step2_cer_skipped` vs `step2_cer_attempted`.
- **Config:** `CER_MODEL`, `CER_SKIP_UNDER_CHARS`, `ENABLE_CER`, provider (Fireworks vs OpenAI).

---



## Batch Intent & Families

**File:** `kb_ner_batch_intent.py`.

- **Families:** PRODUCT | PROCEDURE | CLINICAL | OTHER. Kind → family in `kb_ner_intent.FAMILY_MAP`.
- **Output per entity:** `search_term`, `family`. Category-Locked Guard: if family disagrees with kind for PRODUCT/PROCEDURE, search_term reverted to span_text.
- **Config:** `BATCH_INTENT_MODEL`, `BATCH_INTENT_PER_CHUNK`, `BATCH_INTENT_BEFORE_GROUNDING`.

---



## Grounding (Local-Only, Category-Driven)

**Files:** `kb_ner_parallel.py`, `kb_ner_local_search.py`, `kb_domain_affinity.py`, `grounding_suggestion_enrichment.py`. (Default path is streaming; non-streaming grounding, if used, is wired from the experiment script.)

**Local-only:** When `LOCAL_ONLY=true` (default), no queries to `kb.concepts`; all linking is to `soap.inventory` and `soap.service_master`.

**Category-driven and service_type (Brain NER):**

- Entity's **service_category** and **service_type** (medical | non-medical) → search **services** table; **inventory_category** → search **inventory** table. **General** or missing category is treated as **medical-only** for services (prevents e.g. radiology terms matching grooming SKUs). `CATEGORY_HARD_GATE_RECALL_FALLBACK` controls whether the hard category gate falls back to a wider recall pass when it returns zero candidates.
- **Billable (dual_sync):** Always search **both** inventory and services; pass `entity_inventory_category`, `entity_service_category`, and `entity_service_type` as `category_hints` / `service_type`; merge results by score; take top 20.
- **Domain soft gate — two coexisting implementations, correcting an omission in the old doc:** The primary domain source for local inventory/service scoring is still `kb_ner_domain.detect_domain()` (keyword matching against `DOMAIN_KEYWORDS` per specialty, first 1000 chars of the transcript) — this has not changed and is what the old master doc's "domain soft gate" description was already describing, it just never named the file. Separately, `kb_domain_affinity.py` provides an embedding-based alternative (`DOMAIN_ANCHORS` per specialty, embedded once, cosine-similarity scoring via `get_top_domain()`); this module has existed since the initial commit but is **not** called from the main grounding path (`kb_ner_parallel.py` / `kb_ner_local_search.py`) — it is used only by the LLM Judge (`kb_ner_disambiguation.is_domain_relevant_for_phonetic_threshold()`) to adjust phonetic-threshold domain relevance, and by the global-KB search path (stubbed out under `LOCAL_ONLY=true`, the default). Either way, domain is only ever a **soft** gate (boost/penalty); no hard domain rejection. Anchor vectors for the embedding path are cached via `KB_DOMAIN_VECTORS_PATH`.
- **Vector retrieval:** Enabled. Batch embedding prefetch includes span_text, search_term, and hints; on cache miss, on-demand embed when `LOCAL_VECTOR_ON_DEMAND_EMBED=true`.
- **Lexical safety gate:** Removed; judge-selected matches are no longer rejected by post-judge token-overlap checks.
- **Unlinked-suggestion enrichment:** `grounding_suggestion_enrichment.enrich_unlinked_manifest_suggestions()` copies the top local RAG candidates onto `attributes.suggestions` for manifest entities that end up unlinked (rejected by Judge/threshold), so Phase 2's verification/billing dashboard can still surface "did you mean" SKU suggestions for unlinked items instead of a bare "unlinked" row. Called from `kb_ner_local_search.py`.

**Clinic ID:** Both inventory and services require `clinic_id`. If `clinic_id` is missing, neither local search runs (no default to 1).

**Pharmacy-Free Zone (Golden Gate):** For **Diagnosis** and **ReasonForVisit** (`SERVICE_ONLY_KINDS` in `kb_ner_routing.py`), local search runs **Services only** (no `search_local_inventory_topk`). **Every** grounding output path in `kb_ner_parallel.py` (batch judge, async auto-bind, Judge-selected, high-certainty auto-link local) sets `local_stock_id` = None for these kinds; only `local_service_id` when a service candidate matches. Prevents "pus"→Lasix, "yeast"→AST KIT type errors.

**0.95 certainty wall:** For Diagnosis and ReasonForVisit, if best match score < **0.95** (`DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD`, env override), the entity is **note-only** (no link; preserved in SOAP, not in billing manifest). Applied in batch and async paths before auto-bind and after Judge selection.

**Judge (kb_ner_disambiguation.py):** Uses `get_client_for_model(LLM_JUDGE_MODEL)` only (no cross-provider fallback). **Rule 0 — Category Incompatibility:** REJECT when ENTITY_KIND is Symptom, Diagnosis, or ReasonForVisit but the candidate is a tangible product (Medication, Lab Kit, Consumable). **Rule 2.5 — Symptom / Physical Finding Suppressor:** REJECT symptom/finding mentions (e.g. pus, yeast growth) matching Medications or Lab Reagents unless context explicitly says prescribing/ordering. Form-factor and route-to-form alignment rules remain.

**Dual-sync flow:**

1. Normalize categories and service_type: `entity_inventory_category`, `entity_service_category`, `entity_service_type` (default medical) from entity.
2. Run in parallel (only when `clinic_id` is set): for **non–service-only** kinds, `search_local_inventory_topk(..., category_hints=entity_inventory_category)` and `search_local_services_topk(...)`; for **Diagnosis/ReasonForVisit**, **services only** (no inventory).
3. Merge: `local_candidates = sorted(inv_list + svc_list, key=score, reverse=True)[:20]` (inv_list empty for service-only kinds).
4. For Diagnosis/ReasonForVisit: if best score < 0.95 → note-only; else auto-bind or Judge (only service_id can be set). For other kinds: auto-bind if best local score ≥ `GROUNDING_AUTO_BIND_THRESHOLD` (default **0.92**); else decision flow → LLM Judge.
5. Entities that remain unlinked after the above get `grounding_suggestion_enrichment.enrich_unlinked_manifest_suggestions()` applied so the dashboard has fallback suggestions to show.

**Local search (kb_ner_local_search.py):**

- **Inventory categories (bucket groups):** Deworming, Flea & Tick Treatment, Vaccines, Medication, Diet, Nutrition & Supplements, etc. (see `_CATEGORY_BUCKET_GROUPS`).
- **Service categories:** Medical (Consultation, Surgery, Rehabilitation & Physiotherapy, Lab, Radiology, etc.) vs non-medical (Boarding, Hygiene & Grooming, Training, Behavior, Other Non-Medical). Hard gate uses **service_type** and category hints (see `_MEDICAL_SERVICE_CATEGORY_HINTS`, `_NON_MEDICAL_SERVICE_CATEGORY_HINTS`).
- **Matching:** Trigram + phonetic + vector (batch + on-demand embed); category hard gate restricts to allowed buckets per kind/hints and service_type.

**Kind canonicalization:** `kb_ner_routing.canonicalize_kind()` maps e.g. "Physiotherapy" → "Procedure" so dual_sync and service search run.

---



## Domain-Specific Fallback SKU

**File:** `kb_ner_parallel.py` (post–grounding step).

When **high-stakes** dual_sync entities (e.g. Procedure, Diagnostic, DiagnosticTest, Surgery) remain **unlinked** after normal grounding:

1. **Targeted category search:** Look up Consultation services that match the entity's **domain_key** (e.g. orthopaedic, cardiology) — domain_key comes from `kb_ner_domain.detect_domain()` (keyword-based), the same detector used elsewhere in grounding (see [Grounding](#grounding-local-only-category-driven)).
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



## SOAP Generation (Sections, Formatting, Conclusion)

**File:** `SOAP_notes_phase1_experiment.py` (`build_soap_prompt_from_brain_ner` and related guidance constants).

The SOAP prompt/schema evolved significantly since v3.4:

- **Numbered-list formatting:** `Plan`, `KeyIssues`, `AbnormalFindings`, `CustomerInstructions`, and `Reminders` must each be rendered as a **numbered list, one item per line** — free-text paragraphs are no longer accepted for these fields.
- **Reminders consolidated:** A single `Reminders` field only; the prompt explicitly forbids emitting separate `FollowUpInstructions` / `FollowUpReminders` fields that existed in earlier iterations.
- **CustomerInstructions narrowed:** Scoped to **at-home care only** (e.g. medication administration, diet, activity restriction). Recheck/follow-up visit instructions now belong exclusively in `Reminders`, not `CustomerInstructions`.
- **Diagnosis section (`DIAGNOSIS_SECTION_GUIDANCE`):** Requires numbered `PrimaryDiagnosis` (mandatory, root condition, may be multiple) and optional `SecondaryDiagnosis` (contributing/comorbid conditions), with an explicit "X secondary to Y" disambiguation rule (Y = primary/root cause, X = the secondary/overlay condition). Mirrors `doctor_ui/diagnosis_schema.py`.
- **Vitals section (`VITALS_SECTION_GUIDANCE`) + vitals template:** Driven by the canonical 11-field vitals schema (`doctor_ui/vitals_schema.build_vitals_prompt_block()`) — see [Doctor UI](#doctor-ui-streamlit-app).
- **Conclusion section (`CONCLUSION_SECTION_GUIDANCE`) — new, mandatory:** A stand-alone executive-summary section in formal clinical prose (not a bullet list). Target length **~8–12 sentences (~150–300 words)**, explicitly required to be **longer and more complete than Subjective, Objective, Assessment, or Plan**. It must weave Plan actions (named medications/procedures/diagnostics, follow-up timing) into the prose rather than restating the numbered Plan list verbatim.
- **`_DEFAULT_SOAP_NOTE`:** A single shared default-SOAP-dict constant now backs the 4+ fallback/error paths that previously each hardcoded their own default SOAP JSON literal.

The SOAP JSON is still injected with grounded terms (span_text → display_name) in Step 4 (see `kb_ner_constraint_injection.py`, `kb_anchor_span.py`).

---



## Phase 2: Knowledge Atoms & Dashboard

**Files:** `kb_phase2_integration.py`, `SOAP_notes_billing_phase2_kb_atoms.py`.

- **Atoms:** Extracted from SOAP + manifest; section, kind, assertion_id, intent_context, concept; optional `referenced_entity_id`, `dedup_key` (grounding-aware prompt). Atoms whose intent can't be classified now get `intent_context = "Intent not recognised"` (previously left blank).
- **Clinical-core extraction now includes Reminders:** `_extract_clinical_core_soap()` pulls from Subjective/Objective/Assessment/Plan **and Reminders** (Reminders was previously excluded, along with Conclusion/KeyIssues/CustomerInstructions, which remain excluded from the atom-extraction core but are still rendered in the SOAP note itself).
- **Primary diagnosis surfaced to the dashboard:** `_primary_diagnosis_from_soap()` extracts the PrimaryDiagnosis text from the SOAP JSON and feeds it into `_build_verification_dashboard()` as `primary_diagnosis_text`.
- **Dedupe (ID-first, `_deduplicate_knowledge_atoms`, `kb_phase2_integration.py`):** By local_service_id / local_stock_id first (merge regardless of text, wins over any text-based grouping), then by dedup_key (if present), then same meta-kind (`PHARMACOLOGICAL` / `DIAGNOSTIC` / `INTERVENTIONAL` / `VITAL` / `REMINDER` / `RFV` / `OTHER`) + concept similarity ≥ 0.90 (Levenshtein-based fallback). Section priority Plan > Assessment > Subjective > Objective for representative atom; IDs and source_text merged from group.
- **Constraints:** Plan gate and unlinked/customer-instruction filtering are disabled; atoms are not dropped by those heuristics.
- **Stitching:** Manifest match by referenced_entity_id or by **local_stock_id / local_service_id** first (ID-based); then **manifest_by_normalized_name_id** (Strategy 1b: atom concept + kind → manifest normalized_name so "Cefpodoxime syrup" links to manifest entry with display_name "CefPET Dry Syrup"); then name + kind compatibility (Medicine/Supplement/Nutrition/Preventive/ParasiteControl etc.); when matched, atom inherits manifest kind and IDs.
- **Dashboard routing (reworked):** A Plan item with a bound ID but a non-`Performed` intent now routes to `reminders_follow_ups` **only** if its intent is Scheduled / Future / Recommended / Reminder; otherwise it routes to `module_4_unlinked_entities` with `status: "ACTION REQUIRED"`. The old standalone `generate_streamlined_dashboard()` call has been removed from the main build path.
- **Medicine-recommendation enrichment (new):** `SOAP_notes_billing_phase2_kb_atoms.py` adds `_search_inventory_top3()`, `_alt_from_candidate()`, `_collect_medicine_concepts()`, `_collect_clinical_issues()`, `_enrich_medicine_recommendations()`. For prescribed/administered medication atoms, it attaches the **top-3 clinic-inventory alternatives**; for diagnosed issues that have **no matching medicine atom at all**, it produces a new dashboard key **`issue_medicine_recommendations`** listing candidate medicines per issue so nothing falls through with zero suggestions.
- **ID-based dashboard linking:** Verification dashboard uses **local_stock_id / local_service_id as primary key** for linking/status (not display-name string matching). Reduces "orphans" when Plan atom text (e.g. "Cefpodoxime syrup") differs from manifest display_name (e.g. "CefPET Dry Syrup").
- **Locus routing:** At-home cues (physiotherapy, swimming, etc.) → reminders_follow_ups.
- **Dashboard:** Procedures, Medications, Diagnostics, Unlinked (module_4), Clinical History, Reminders, Vitals, and (new) Issue Medicine Recommendations. Unlinked items get top-5 suggestions from local candidates (backed by `grounding_suggestion_enrichment.py` for entities unlinked at the grounding stage).

**Default model:** `gpt-4.1-mini` (OpenAI). **Config (original five):** `PHASE2_MODEL`, `PHASE2_MODEL_PROVIDER` (default `openai`), `PHASE2_ESCALATE_MODEL`, `PHASE2_START_WITH_SOAP_READY`, `PHASE2_ENABLE_BILLING_MATCHING`, `run_timestamp` passed from experiment for consistent output paths.

**New Phase 2 config (performance/behavior tuning added since v3.4):** `PHASE2_ENABLE_SESSION_CACHE`, `PHASE2_SCHEMA_CACHE_ENABLE`, `PHASE2_SCHEMA_CACHE_TTL_SEC`, `PHASE2_CLINICAL_CORE_ONLY`, `PHASE2_DRY_RUN`, `PHASE2_BLOCKING`, `PHASE2_MODE`, `PHASE2_STRICT_KIND_MAP`, `PHASE2_WARN_UNMAPPED_KINDS`, `PHASE2_ESCALATE_ON_JSON_FAIL`, `PHASE2_MAX_TOKENS`, `PHASE2_MAX_TOKENS_LONG`, `PHASE2_MAX_TOKENS_PARALLEL`, `PHASE2_MAX_TOKENS_PARALLEL_MIN`, `PHASE2_TOKENS_PER_ENTITY_ESTIMATE`, `PHASE2_MAX_ENTITIES_PER_BATCH`, `PHASE2_PARALLEL_ATOM_ENTITY_THRESHOLD`, `PHASE2_PARALLEL_ATOM_MAX_BATCHES`, `PHASE2_PARALLEL_SECTIONS`, `PHASE2_LONG_MANIFEST_ENTITIES`, `PHASE2_LONG_SOAP_CHARS`, `PHASE2_VITALS_REGISTRY_INJECT`, `PHASE2_VITALS_REGISTRY_MAX_ROWS`, `PHASE2_VITALS_REGISTRY_MAX_SYNONYMS`, `PHASE2_TEMPERATURE`.

**ID-first behavior (v3.1, unchanged):** Deduplication is ID-first: group by `local_service_id` or `local_stock_id` first (merge regardless of text); then by `dedup_key` (if LLM emits it); then same meta-kind + concept similarity ≥ 0.90. Stitching prefers `referenced_entity_id` and manifest IDs; when a manifest match is found, atom inherits manifest `kind`. Kind compatibility matrix includes Preventive/ParasiteControl with pharmacological kinds. Verification dashboard deduplicates medications by `inventory_id`. Plan gate and manifest-based kind filtering for the prompt are disabled; full schema is used.

---



## Pipeline Observability (Timing & Health Flags)

**Files:** `pipeline_timing.py`, `pipeline_flags.py`.

New in this cycle: every pipeline run now produces both a machine-readable timing report and a health/problem-flag report, both consumed by the Doctor UI.

### PipelineTimer (`pipeline_timing.py`)

A small stateful timer class replacing the old ad-hoc `stage_marks` dict:

- `PipelineTimer.mark(name)` – records a named timestamp/checkpoint (e.g. `step2_super_pass_start`, `step2_super_pass_done`, `step2_brain_ner_start/done`, `injection_start/done`, `phase2_start`).
- `.set_duration_ms(name, ms)` – records an explicit duration for a named stage.
- `.set_flag(name, value)` – records a boolean/flag value alongside timings (e.g. whether CER was attempted or skipped).
- `.merge_phase2(phase2_timing)` – merges in the timing sub-report produced inside Phase 2 (`_set_phase2_timing()` / `_elapsed_ms_since()` in `kb_phase2_integration.py`: `early_subjective_objective_ms`, `step1_atom_extraction_ms`, `step2_post_process_ms`, `step3_dashboard_ms`, `phase2_total_ms`).
- `.build_report()` – produces the final JSON timing report written alongside run artifacts (`pipeline_timing_*.json`, plus a `_latest.json` pointer) and shown in the Doctor UI's "Pipeline timing" panel (`doctor_ui/components/soap_display.render_pipeline_timing()` / `load_pipeline_timing()`).

### Pipeline health flags (`pipeline_flags.py`)

`build_pipeline_flags(output_dir, soap_json=None, entity_manifest=None, knowledge_atoms=None, extra_flags=None) -> PipelineHealthReport` inspects a run's artifacts after the fact and produces a structured report:

- `PipelineFlag(step, severity, code, message, detail)` — `step` is one of `STEP1 | STEP2 | GROUNDING | STEP3 | PHASE2 | SYSTEM`; `severity` is `info | warning | error`.
- `PipelineHealthReport(overall, generated_at, flags, stage_summary)` — `overall` is `healthy | degraded | failing`, `stage_summary` maps each stage to `ok | degraded | failing`, and `.to_dict()` includes flag counts by severity.
- **Example flag codes:** `SHORT_TRANSCRIPT` / `EMPTY_TRANSCRIPT` / `UNCLEAR_AUDIO_MARKERS` / `SUSPECT_ASR_TOKENS` / `ASR_ERROR` (STEP1); `CLEANING_FAILED` / `NO_NER_ENTITIES` (STEP2); grounding-side flags for unlinked-entity rates and skipped grounding; SOAP-completeness flags for STEP3; schema/atom-missing flags for PHASE2.
- Written as `pipeline_flags_*.json` (+ `_latest.json`) and rendered in the Doctor UI's "Workflow health" panel (🟢/🟡/🔴) above the SOAP note.

---



## Indexes, Embeddings, and Fuzzy Matching

**Extensions (PostgreSQL):** `pg_trgm`, `fuzzystrmatch`, `vector`. Ensured in `kb_ner_db.py` via `ensure_pg_trgm()`, `ensure_fuzzystrmatch()`, `ensure_vector_extension()`.

### Index types and usage


| Type              | Extension | Tables / columns                                                                                                                                                                                                | Index names (examples)                                                                                                                                                                                                                               | Purpose                                                                                                                    |
| ----------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **GIN (trigram)** | pg_trgm   | `soap.inventory` (item_name, trade_name), `soap.service_master` (procedure_name), `kb.concepts` (preferred_name), `kb.concept_aliases` (alias_text), `kb.vitals_registry` (search_text)                         | `idx_soap_inventory_item_name_trgm`, `idx_soap_inventory_trade_name_trgm`, `idx_soap_service_master_procedure_name_trgm`, `idx_kb_concepts_preferred_name_trgm`, `idx_kb_concept_aliases_alias_text_trgm`, `idx_kb_vitals_registry_search_text_trgm` | Trigram similarity for fuzzy text match                                                                                    |
| **B-tree**        | —         | `kb.concepts` (kind, domain_key), `soap.inventory` (domain_key, category), `soap.service_master` (domain_key, category), `kb.vitals_registry` (metaphone_key)                                                   | `idx_kb_concepts_kind`, `idx_kb_concepts_domain_key`, `idx_soap_inventory_domain_key`, `idx_soap_service_master_domain_key`, `idx_soap_inventory_domain_category`, `idx_soap_service_master_domain_category`, `idx_kb_vitals_registry_metaphone_key` | Filters and composite lookups; **metaphone_key** is the only phonetic index (exact metaphone / phonetic filter for vitals) |
| **HNSW (vector)** | vector    | `kb.concepts` (embedding, embedding_vetbert), `soap.inventory` (vector_embedding_vetbert), `soap.service_master` (vector_embedding_vetbert), `kb.vitals_registry` (embedding), `kb.learned_aliases` (embedding) | `idx_kb_concepts_embedding_vetbert_hnsw`, `idx_soap_inventory_vector_embedding_vetbert_hnsw`, `idx_soap_service_master_vector_embedding_vetbert_hnsw`, `idx_kb_vitals_registry_embedding_hnsw`                                                       | Approximate nearest-neighbor (cosine); local search may use `vector_embedding` (OpenAI) when present                       |


Local search uses **trigram + phonetic** always; **vector** is enabled via batch prefetch (span_text, search_term, hints) and on-demand embedding when cache misses (`LOCAL_VECTOR_ON_DEMAND_EMBED=true`), so vector scoring is used even when `vector_embedding` is not pre-populated on rows. Trigram GIN indexes are created by `ensure_trgm_gin_index()`; HNSW by `ensure_hnsw_index()`. See `ensure_kb_search_indexes()`, `ensure_soft_gate_indexes()`, `ensure_vitals_registry_table()` in `kb_ner_db.py`.

### Embeddings


| Use                                  | Model                           | Dimensions | Source                                                      | Cache                                                                   |
| ------------------------------------ | -------------------------------- | ---------- | ----------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Local inventory / service_master** | OpenAI `text-embedding-3-small` | 1536       | `kb_ner_embeddings.embed_text()`; column `vector_embedding` | In-memory LRU, key `(model, text)`; `KB_EMBED_CACHE_MAX` (default 2048) |
| **Optional VetBERT (local)**         | havocy28/VetBERT                | 768        | Backfill script; column `vector_embedding_vetbert`          | —                                                                       |
| **kb.concepts**                      | OpenAI (and optional VetBERT)   | 1536 / 768 | Same; columns `embedding`, `embedding_vetbert`              | Same cache for OpenAI                                                   |
| **kb.vitals_registry**               | OpenAI                          | 1536       | Same; column `embedding`                                    | Same                                                                    |
| **Domain affinity anchors** (Judge/global search only, not main grounding) | OpenAI (same embedding client) | 1536 | `kb_domain_affinity.py`; `DOMAIN_ANCHORS` per specialty      | Cached to disk at `KB_DOMAIN_VECTORS_PATH`                              |


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

- **"Phonetic bucket" (Stage 1):** Actually **trigram** similarity: `similarity(lower(preferred_name), mention) >= sim_thresh` (default 0.2; **escalation** 0.1 for specialty when Pass 1 returns 0 rows). Uses GIN trigram indexes on preferred_name and alias_text.
- **Metaphone fallback:** When **specialty** and trigram bucket returns 0 rows: `ensure_fuzzystrmatch` then pure metaphone matching — `metaphone(lower(preferred_name), 10) = q.q_mfull` or `= q.q_mfirst`, **or** normalized `levenshtein(metaphone(...), q) < 0.65` on concepts and concept_aliases. No stored metaphone column; all computed in SQL.
- **Batch path:** `_batch_trigram_phonetic_scores()`: same metaphone + Levenshtein formula (0.8 * (1 - norm_lev)) for `phonetic_score` per candidate name.

**Summary of phonetic indexes**


| Table                  | Phonetic column | Index                                         | Notes                                                  |
| ---------------------- | --------------- | ---------------------------------------------- | -------------------------------------------------------- |
| **kb.vitals_registry** | `metaphone_key` | B-tree `idx_kb_vitals_registry_metaphone_key` | Only stored phonetic key + index in the system         |
| soap.inventory         | —               | —                                             | Metaphone computed in SQL from item_name, trade_name   |
| soap.service_master    | —               | —                                             | Metaphone computed in SQL from procedure_name          |
| kb.concepts            | —               | —                                             | Metaphone fallback computed in SQL from preferred_name |
| kb.concept_aliases     | —               | —                                             | Metaphone fallback computed in SQL from alias_text     |


**Ensured:** `ensure_fuzzystrmatch(conn)` before local search, vitals registry search, and global metaphone fallback.

### Fuzzy matching rules (local search)

- **Trigram retrieval:** `%` operator is used as primary trigram retrieval signal (`lower(name) % query`) plus explicit similarity floor via `LOCAL_TRGM_RECALL_THRESHOLD` (default 0.30).
- **Phonetic retrieval:** Double-Metaphone primary/secondary key equality (`dmetaphone`, `dmetaphone_alt`) on full and first-token forms.
- **Reranking:** Jaro-Winkler (prefix-aware) is used for lexical reranking; `jaro_winkler_score` is logged per candidate.
- **Vector:** When `vector_embedding` is not null, `(1 - LEAST(embedding <=> query_vec, 1.0))` as vector_score; optional WHERE `embedding <=> query_vec < 0.5` to restrict to close neighbors.
- **match_score (per candidate):** `max(trigram_score, jaro_winkler_score, vector_score)`; then in Python candidates with `match_score < 0.30` are dropped.
- **Final ranking:** `final_score = (match_score * SOFT_GATE_LOCAL_BASE_WEIGHT) + domain_boost + category_boost + suggestion_boost` when domain soft gate is on (domain_boost sourced from `kb_ner_domain.detect_domain()`, keyword-based; see `docs/LOCAL_GROUNDING_SEARCH_AND_RETRIEVAL.md`).
- **Default threshold** for "best match" in guard/validation: 0.50 (parameter `threshold` in `search_local_inventory_topk` / `search_local_services_topk`) — distinct from the auto-bind decision threshold `GROUNDING_AUTO_BIND_THRESHOLD` (0.92).

**Category:** SQL normalizes category as `LOWER(TRIM(REPLACE(COALESCE(category,''), '&', 'and')))`; bucket groups in `_CATEGORY_BUCKET_GROUPS` match that form. Hard gate restricts search to allowed buckets per kind + entity category hints, with `CATEGORY_HARD_GATE_RECALL_FALLBACK` controlling whether a zero-result hard-gated query falls back to a wider pass.

**Clinic ID:** Local inventory and local services both require `clinic_id`. If `clinic_id` is missing, both searches are skipped (no default to 1). Location: `soap.inventory.location_id`; service_master may not have location (clinic_id still gates whether search runs).

---



## Complete Codebase


| File                                      | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SOAP_notes_phase1_experiment.py**       | Orchestration: audio/text → transcript → Super-Pass → Batch Intent → Grounding → SOAP (incl. Conclusion) → Injection → Phase 2; entries `generate_soap_note_from_audio_async()` and `generate_soap_note_from_transcript_async()`; drives the `PipelineTimer`                                                                                                                                                                                                                                                                                     |
| **kb_ner_super_pass.py**                  | Super-Pass: cleaning + NER; outputs entities with inventory_category, service_category, service_type; default model Llama v3p3 70B; streaming and chunked modes                                                                                                                                                                                                                                                                                                                                                                                     |
| **kb_ner_chunk_parallel.py**               | Chunk-parallel Super-Pass for long transcripts: adaptive chunk sizing, overlap chunking, signalment extraction/injection, cross-chunk merge/dedupe/kind-correction (present since the initial commit); `IncrementalAsrSuperPass` (streams Super-Pass on stable transcript prefixes while ASR is still running) was added in the "Phase 2" commit                                                                                                                                                                                                                                                                                                                                                       |
| **kb_ner_batch_intent.py**                | Batch Intent: search_term + family per entity; Category-Locked Guard                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **kb_ner_parallel.py**                    | Grounding (streaming): process_single_entity_async, process_entity_by_route_async; dual_sync = _local_inventory + _local_services (for non–service-only kinds) or _local_services only for Diagnosis/ReasonForVisit (Pharmacy-Free Zone); **every** output path (batch judge, auto-bind, Judge-selected, high-certainty auto-link local) sets local_stock_id = None for Diagnosis/ReasonForVisit; 0.95 certainty wall (note-only if score < 0.95); auto-bind at GROUNDING_AUTO_BIND_THRESHOLD (0.92); batch prefetch + on-demand embed; domain consultation fallback + dedupe for high-stakes unlinked |
| **kb_ner_local_search.py**                | Local search: search_local_inventory_topk, search_local_services_topk; category bucket groups + service_type (medical/non-medical); trigram + phonetic + vector; invokes `grounding_suggestion_enrichment` for unlinked entities                                                                                                                                                                                                                                                                                                                   |
| **grounding_suggestion_enrichment.py**    | `enrich_unlinked_manifest_suggestions()` — attaches top RAG/local candidates to unlinked manifest entities so the dashboard can show suggestions instead of a bare unlinked status                                                                                                                                                                                                                                                                                                                                                                  |
| **kb_domain_affinity.py**                 | Embedding-based domain affinity: `DOMAIN_ANCHORS` per specialty, cosine-similarity scoring via `get_top_domain()`. Present since the initial commit; used narrowly by the LLM Judge (`is_domain_relevant_for_phonetic_threshold`) and global search — **not** the domain source for the main local-search grounding path (see `kb_ner_domain.py`)                                                                                                                                                                                                                                                                                                                                                          |
| **kb_ner_routing.py**                     | Routing: canonicalize_kind, classify_entity_route; DUAL_SYNC_BILLABLE_KINDS, SERVICE_ONLY_KINDS (Diagnosis, ReasonForVisit), HARD_SKIP_KINDS (Anatomy, Symptom), DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD (0.95), is_services_only_kind(); GLOBAL_DIRECT_KINDS=[]                                                                                                                                                                                                                                                                               |
| **kb_ner_global_search.py**               | Global KB search (stubbed when LOCAL_ONLY=true): search_global_topk, run_batch_global_vector_search, kb_lookup_* return [] or no-op                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **kb_ner_disambiguation.py**              | Decision flow and LLM Judge: apply_decision_flow, run_single_batch_llm_judge; Rule 0 Category Incompatibility (finding vs product → REJECT), Rule 2.5 Symptom/Physical Finding Suppressor; form-factor/route alignment; Judge uses get_client_for_model(LLM_JUDGE_MODEL) only (no cross-provider fallback)                                                                                                                                                                                                                                          |
| **kb_ner_db.py**                          | DB: acquire_pg_conn, pg_conn_ctx, search_vitals_registry_topk, ensure_fuzzystrmatch                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **kb_ner_embeddings.py**                  | Embeddings: embed_texts, to_pgvector_literal; cache                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **kb_ner_skeleton_parser.py**             | Parse Brain NER skeleton_list output into entity dicts (inventory_category, service_category, service_type)                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **kb_ner_clinical_entity_resolver.py**    | CER: resolve entities to billing items; Fireworks streaming + JSON repair when large output; default model Llama v3p3 70B; skip path via CER_SKIP_UNDER_CHARS / ENABLE_CER                                                                                                                                                                                                                                                                                                                                                                          |
| **kb_ner_constraint_injection.py**        | Constraint + truth injection: enforce_sectional_truth_injection; manifest corrections                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **kb_anchor_span.py**                     | Anchor IDs: ensure_anchor_ids_on_manifest (E1, E2, …)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **kb_anchor_resolve.py**                  | Re-grounding: resolve_anchor_update by anchor_id + new text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **pipeline_timing.py**                    | `PipelineTimer` — per-stage marks/durations/flags, merges Phase 2's internal timing, builds the final timing report                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **pipeline_flags.py**                     | `PipelineFlag`/`PipelineHealthReport` dataclasses, `build_pipeline_flags()` — post-hoc health/problem-flag report (STEP1–PHASE2, overall healthy/degraded/failing) from a run's artifacts                                                                                                                                                                                                                                                                                                                                                            |
| **kb_phase2_integration.py**              | Phase 2: extract_knowledge_atoms_async, dedupe, Plan gate, enrich_atoms_with_manifest_ids, per-stage Phase2 timing helpers (`_set_phase2_timing`, `_elapsed_ms_since`), `_primary_diagnosis_from_soap`                                                                                                                                                                                                                                                                                                                                              |
| **SOAP_notes_billing_phase2_kb_atoms.py** | Phase 2: build_knowledge_atom_prompt, parse SOAP sections, verification/billing dashboard, unlinked suggestions, medicine-recommendation enrichment (`_search_inventory_top3`, `_enrich_medicine_recommendations`, `issue_medicine_recommendations`)                                                                                                                                                                                                                                                                                                |
| **kb_ner_intent.py**                      | Intent families: FAMILY_MAP (kind → PRODUCT/PROCEDURE/CLINICAL/OTHER)                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **kb_ner_intent_guards.py**               | Intent guards: ground_clinical_terms, Category-Locked Guard                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **kb_ner_domain.py**                      | Domain detection actually used in the grounding hot path: keyword-based `detect_domain()` (`DOMAIN_KEYWORDS`/`DOMAIN_PRIMERS`, first 1000 chars of transcript), called from `kb_ner_parallel.py` for domain_key/domain_boost and the domain-fallback SKU                                                                                                                                                                                                                                                                                                                                                                                            |
| **kb_ner_clients.py**                     | LLM/embedding clients: OpenAI, Fireworks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **kb_ner_enrichment.py**                  | Enrichment helpers (optional path)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **kb_ner_extraction.py**                  | Extraction helpers (optional path)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **kb_ner_lexical_harvester.py**           | Lexical harvester (optional): load terms from DB for search                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **kb_ner_2phase_integration.py**          | 2-phase integration (optional when not using Super-Pass)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **local_embedding_workflow.py**           | Pure-function mirror of embedding / internal-description templates from the backfill script (`build_inventory_internal_description`, etc.), reused by search/audit tooling                                                                                                                                                                                                                                                                                                                                                                        |
| **long_transcript_utils.py**              | Chunking + `[LONG_TRANSCRIPT_SUMMARY]` wrap/extract helpers to keep the SOAP prompt from overflowing context on very long transcripts                                                                                                                                                                                                                                                                                                                                                                                                              |
| **ocr_test_inventory.py**                 | Standalone OCR/inventory-extraction test utility (Mistral/OpenAI/Claude OCR+LLM+embedding provider selection, hybrid search weights) — **not** part of the SOAP pipeline proper                                                                                                                                                                                                                                                                                                                                                                    |
| **asr_providers.py**                      | ASR provider abstraction (Deepgram/Fireworks); transcription + streaming + benchmarking/comparison support                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **asr_benchmark.py**                      | ASR benchmark scorer: WER/CER vs Gemini reference transcripts (see [Benchmarking & Evaluation](#benchmarking--evaluation))                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **pipeline_benchmark.py**                 | Stage-wise full-pipeline benchmark scorer (see [Benchmarking & Evaluation](#benchmarking--evaluation))                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **batch_pipeline_runner.py**              | Batch CLI: run ASR-only or full SOAP pipeline over a directory of audio files (see [Benchmarking & Evaluation](#benchmarking--evaluation))                                                                                                                                                                                                                                                                                                                                                                                                          |
| **benchmark_utils.py**                    | Shared benchmarking helpers: run-dir/manifest builders, slug/path resolution, text normalization for WER scoring                                                                                                                                                                                                                                                                                                                                                                                                                                    |


**Note on drift correction:** the previous master doc's "optional" file `kb_ner_manifest_guardrail.py` does not exist in the current repo (confirmed via file search) — it has been removed from this table rather than carried forward.

**Legacy (moved):** Backfills, audits, one-off scripts, and old pipeline code live under `legacy/python/`, `legacy/scripts/`, `legacy/docs/`. See `legacy/README.md`.

---



## Function Reference


| Function                                                                | File                             | Purpose                                                                                                  |
| ----------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `generate_soap_note_from_audio_async`                                   | SOAP_notes_phase1_experiment     | Entry (voice): audio → grounded SOAP + Phase 2                                                           |
| `generate_soap_note_from_transcript_async`                              | SOAP_notes_phase1_experiment     | Entry (typed/edited text): transcript → grounded SOAP + Phase 2                                          |
| `transcribe_audio` / `transcribe_audio_fireworks_streaming`             | SOAP_notes_phase1_experiment     | Step 1: audio → raw transcript (Fireworks Whisper; retries 502/503 with backoff)                         |
| `super_pass_cleaning_and_ner_streaming` / `super_pass_cleaning_and_ner` | kb_ner_super_pass                | Step 2: clean + NER; inventory_category, service_category, service_type                                  |
| `process_chunks_parallel`, `compute_adaptive_chunk_params`, `chunk_raw_transcript_with_overlap` | kb_ner_chunk_parallel   | Step 2 (long transcripts, post-ASR): chunk-parallel cleaning + NER                                          |
| `IncrementalAsrSuperPass.feed` / `.finalize`                            | kb_ner_chunk_parallel             | Step 2 (during ASR): overlaps Super-Pass cleaning with in-progress transcription                          |
| `run_batch_intent`                                                      | kb_ner_batch_intent              | Step 2b: search_term + family                                                                            |
| `run_step_2_3_normalization`                                            | (non-streaming path, if enabled) | Step 2.3 (non-streaming): full grounding; default is streaming via kb_ner_parallel                       |
| `process_single_entity_async` / `process_entity_by_route_async`         | kb_ner_parallel                  | Step 2.3 (streaming): per-entity grounding; dual_sync = _local_inventory + _local_services               |
| `classify_entity_route`                                                 | kb_ner_routing                   | Route: skip_vitals, skip_signalment, skip_identity, global_vitals, dual_sync, skip_non_billable, global_direct |
| `classify_procedure_role`                                               | kb_ner_routing                   | Procedure role (Performed/Prescribed/etc.)                                                               |
| `search_local_inventory_topk` / `search_local_services_topk`            | kb_ner_local_search              | Local search with category_hints and service_type; trigram + phonetic + vector (batch + on-demand embed) |
| `enrich_unlinked_manifest_suggestions`                                  | grounding_suggestion_enrichment  | Attaches RAG/local candidate suggestions to unlinked manifest entities                                    |
| `detect_domain`                                                         | kb_ner_domain                    | Keyword-based domain detection actually used for grounding's domain_key/domain_boost                     |
| `get_top_domain` / `is_domain_relevant_for_phonetic_threshold`          | kb_domain_affinity               | Embedding-based domain affinity; used only by the LLM Judge and global search, not local-search grounding |
| `apply_decision_flow` / LLM Judge                                       | kb_ner_disambiguation            | Option A/B/C; Judge uses search_term                                                                     |
| `generate_soap_with_grounding`                                          | SOAP_notes_phase1_experiment     | Step 3: SOAP note (numbered-list sections + mandatory Conclusion)                                         |
| `apply_constraint_based_injection`                                      | kb_ner_constraint_injection      | Step 4: constraint injection                                                                             |
| `apply_manifest_corrections_to_soap_json`                               | SOAP_notes_phase1_experiment     | Step 4: truth injection; anchor tags                                                                     |
| `ensure_anchor_ids_on_manifest`                                         | kb_anchor_span                   | Assign E1, E2, …                                                                                         |
| `extract_knowledge_atoms_async`                                         | kb_phase2_integration            | Phase 2: atoms, dedupe, dashboard                                                                         |
| `_primary_diagnosis_from_soap`                                          | kb_phase2_integration            | Extracts PrimaryDiagnosis text from SOAP JSON for the dashboard                                           |
| `_enrich_medicine_recommendations` / `_search_inventory_top3`           | SOAP_notes_billing_phase2_kb_atoms | Attaches inventory alternatives to medicine atoms / recommends medicines for unmatched issues            |
| `search_vitals_registry_topk`                                           | kb_ner_db                        | global_vitals route                                                                                      |
| `embed_texts`                                                           | kb_ner_embeddings                | Batch embeddings; cache                                                                                  |
| `canonicalize_kind`                                                     | kb_ner_routing                   | Normalize kind (e.g. Physiotherapy → Procedure)                                                          |
| `PipelineTimer.mark` / `.set_duration_ms` / `.set_flag` / `.merge_phase2` / `.build_report` | pipeline_timing | Per-stage timing instrumentation, threaded through the whole pipeline                                     |
| `build_pipeline_flags`                                                  | pipeline_flags                   | Post-hoc health/problem-flag report for a completed run                                                   |


---



## Configuration & Environment

**Pipeline:**

- `USE_SUPER_PASS`, `SUPER_PASS_STREAMING`, `SUPER_PASS_MODEL` (default `accounts/fireworks/models/llama-v3p3-70b-instruct`)
- `CER_MODEL` (default Llama v3p3 70B on Fireworks); CER streaming + JSON repair when Fireworks and large output; `CER_SKIP_UNDER_CHARS`, `ENABLE_CER`
- `BATCH_EMBED_PER_CHUNK`, `BATCH_INTENT_BEFORE_GROUNDING`, `BATCH_INTENT_PER_CHUNK`, `BATCH_INTENT_MODEL`
- `COMBINED_CLEAN_NER_BATCH_INTENT` (Unified Mega-Pass)
- `CLINIC_ID` (default 1), `VISIT_ID`
- `KB_MAX_PARALLEL_ENTITIES` (default 12)
- `LOCAL_ONLY` (default true): no global KB; local inventory + services only
- `LOCAL_VECTOR_ON_DEMAND_EMBED` (default true): on-demand embedding for local search when cache misses; vector retrieval enabled

**Chunk-parallel Super-Pass (long transcripts):**

- `CHUNK_PARALLEL_ENABLED`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNK_MIN_PARALLEL`, `CHUNK_MAX_PARALLEL`, `CHUNK_TARGET_SIZE`, `CHUNK_SINGLE_THRESHOLD`, `CHUNK_WORKER_TIMEOUT_SEC`
- `LONG_TRANSCRIPT_THRESHOLD_CHARS`, `FORCE_LONG_TRANSCRIPT_MODE`, `SOAP_MAX_TRANSCRIPT_CHARS_IN_PROMPT`, `SOAP_EXCERPT_MAX_CHARS`

**Grounding:**

- `auto_bind_threshold` parameter / **`GROUNDING_AUTO_BIND_THRESHOLD`** env override (default **0.92** — briefly lowered to 0.85, reverted after unsatisfactory output), `llm_judge_threshold` (0.55)
- `LLM_JUDGE_MODEL` (Judge uses this model's provider only; no cross-provider fallback)
- `DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD` (default 0.95): Diagnosis/ReasonForVisit below this score → note-only (no link)
- `CATEGORY_HARD_GATE_RECALL_FALLBACK`: whether a zero-result hard category gate falls back to a wider recall pass
- `KB_DOMAIN_VECTORS_PATH`: cache path for `kb_domain_affinity.py`'s embedded domain anchor vectors (Judge/global-search path only, not the main grounding domain detector)

**SOAP / Phase 2:**

- `SOAP_MODEL`, `ANCHOR_MAPPING_SOAP`, `ANCHOR_SPAN_OUTPUT`
- `PHASE2_MODEL` (default `gpt-4.1-mini`), `PHASE2_MODEL_PROVIDER` (default `openai`), `PHASE2_ESCALATE_MODEL`, `PHASE2_START_WITH_SOAP_READY`, `PHASE2_ENABLE_BILLING_MATCHING`
- **New Phase 2 tuning vars:** `PHASE2_ENABLE_SESSION_CACHE`, `PHASE2_SCHEMA_CACHE_ENABLE`, `PHASE2_SCHEMA_CACHE_TTL_SEC`, `PHASE2_CLINICAL_CORE_ONLY`, `PHASE2_DRY_RUN`, `PHASE2_BLOCKING`, `PHASE2_MODE`, `PHASE2_STRICT_KIND_MAP`, `PHASE2_WARN_UNMAPPED_KINDS`, `PHASE2_ESCALATE_ON_JSON_FAIL`, `PHASE2_MAX_TOKENS`, `PHASE2_MAX_TOKENS_LONG`, `PHASE2_MAX_TOKENS_PARALLEL`, `PHASE2_MAX_TOKENS_PARALLEL_MIN`, `PHASE2_TOKENS_PER_ENTITY_ESTIMATE`, `PHASE2_MAX_ENTITIES_PER_BATCH`, `PHASE2_PARALLEL_ATOM_ENTITY_THRESHOLD`, `PHASE2_PARALLEL_ATOM_MAX_BATCHES`, `PHASE2_PARALLEL_SECTIONS`, `PHASE2_LONG_MANIFEST_ENTITIES`, `PHASE2_LONG_SOAP_CHARS`, `PHASE2_VITALS_REGISTRY_INJECT`, `PHASE2_VITALS_REGISTRY_MAX_ROWS`, `PHASE2_VITALS_REGISTRY_MAX_SYNONYMS`, `PHASE2_TEMPERATURE`
- `CONSTRAINT_INJECTION_SINGLE_PASS`, `CONSTRAINT_INJECTION_MODE`, `CONSTRAINT_INJECTION_MODEL`

**Transcription:**

- `FIREWORKS_API_KEY`, `FIREWORKS_MODEL_NAME` (whisper-v3-turbo), `FIREWORKS_SAMPLE_RATE` (16000)
- `TRANSCRIPTION_STREAMING`, `ASR_PROVIDER`, `ASR_MODEL`, `ASR_LANGUAGE`, `DEEPGRAM_API_KEY`
- **Resilience:** `transcribe_audio()` retries on 502/503 (up to 3 attempts, backoff 2s/4s/8s); on failure raises a short user-facing message (no HTML dump). Tips for 502/503 shown in exception handler.

**Latency / performance:**

- `LOW_LATENCY_MODE`, `TARGET_LATENCY_SEC`, `TARGET_60S`, `FAST_TRANSCRIPTION`

**Pipeline control (Doctor UI / runtime):**

- `SKIP_PHASE2`, `SKIP_BILLING_PIPELINE` (auto-set by `doctor_ui/pipeline_runner.py` when Postgres is unreachable)
- `VETAI_DATA_DIR` (SQLite consultations DB location, default `doctor_ui/data`), `VETAI_RUNS_DIR` (per-run artifact location, default `doctor_ui/runs`), `VETAI_PIPELINE_LOG_LEVEL` (default DEBUG), `VETAI_PORT` (Doctor UI port, default 8501)

**Chunked / long transcripts:**

- `LONG_TRANSCRIPT_THRESHOLD_CHARS`, per-chunk timeout (e.g. 180s) in `kb_ner_chunk_parallel.py` (`CHUNK_WORKER_TIMEOUT_SEC`)

---



## Database & Tables

**Extensions:** `vector` (pgvector), `pg_trgm`, `fuzzystrmatch`. See [Indexes, Embeddings, and Fuzzy Matching](#indexes-embeddings-and-fuzzy-matching) for index types and names.

**Local (clinic):**

- **soap.inventory** – Products (medication, vaccines, parasite control, diet, etc.); `location_id`, category, item_name, trade_name, stock_id; optional `vector_embedding` (1536, OpenAI), optional `vector_embedding_vetbert` (768); `domain_key` for soft gate. GIN trigram on item_name, trade_name; B-tree on domain_key, (domain_key, category); optional HNSW on vector_embedding_vetbert.
- **soap.service_master** – Services (consultation, surgery, rehabilitation & physiotherapy, lab, etc.); category, procedure_name, service_id; optional `vector_embedding` (1536), optional `vector_embedding_vetbert` (768); `domain_key`. GIN trigram on procedure_name; B-tree on domain_key, (domain_key, category); optional HNSW on vector_embedding_vetbert.

**Global (disabled when LOCAL_ONLY=true):**

- **kb.concepts** – concept_id, preferred_name, kind, definition, embedding (1536), embedding_vetbert (768), domain_key. Not queried when LOCAL_ONLY. Indexes: B-tree (kind, domain_key), GIN (preferred_name), HNSW (embedding_vetbert).
- **kb.vitals_registry** – Used for global_vitals route (vital metric names, synonyms, search_text, metaphone_key, embedding). Trigram + phonetic + vector search; indexes: GIN (search_text), B-tree (metaphone_key), HNSW (embedding).

**Doctor UI (application-local):**

- **`doctor_ui/data/consultations.db`** (SQLite) – single `consultations` table; see [Doctor UI (Streamlit App)](#doctor-ui-streamlit-app) for schema.

**Setup:**

- Indexes are ensured by `kb_ner_db.py` (ensure_pg_trgm, ensure_fuzzystrmatch, ensure_vector_extension, ensure_trgm_gin_index, ensure_hnsw_index, ensure_kb_search_indexes, ensure_soft_gate_indexes, ensure_vitals_registry_table).
- **`scripts/setup_phase2_database.py`** (new) — one-shot Postgres bootstrap for a fresh clinic environment: creates the DB role/database, enables `vector`/`pg_trgm`/`fuzzystrmatch`, creates `soap.inventory`, `soap.service_master`, `kb.vitals_registry`, `kb.assertion_types`/`attributes_schema`, seeds demo clinic data from `scripts/seed_demo_clinic_data.sql`, and writes the resulting `PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER`/`CLINIC_ID` values into `.env`. This is the recommended path for standing up Phase 2/billing grounding from scratch.
- Optional backfills for vector_embedding / VetBERT; see `legacy/README.md` and `legacy/python/backfill_soap_domain_and_vetbert.py`.

---



## Doctor UI (Streamlit App)

**Folder:** `doctor_ui/`. **Quick start:** `docs/DOCTOR_UI.md`. Run with `python -m streamlit run doctor_ui/app.py` (opens at `http://localhost:8501`).

This is the doctor-facing application built on top of the pipeline documented above: vets capture a consultation (typed or voice), generate a SOAP note plus a billing/pharmacy dashboard, review and edit line items, and browse consultation history — all from one Streamlit app, without needing to touch the underlying Python pipeline directly.

### `app.py` — entry point and navigation

Sidebar-navigated across three views (`main()`):

- **New Consultation** (`page_new_consultation`) — collects consultation language (via `languages.language_labels()`) plus optional doctor/pet name, then offers two input tabs:
  - **Type conversation** — a free-text box; "Generate SOAP note" runs `pipeline_runner.run_pipeline_for_consultation()` → `generate_soap_note_from_transcript_async()`.
  - **Voice note** — file upload (wav/mp3/m4a/ogg/webm/flac) or in-browser mic capture (`st.audio_input`); either **"Transcribe only"** (`pipeline_runner.transcribe_audio_file()`, fills the text box for editing before generating) or a single-click **"Generate SOAP from voice"** (`pipeline_runner.run_pipeline_from_audio()` → `generate_soap_note_from_audio_async()`, i.e. ASR happens inside the pipeline as its own timed STEP1).
  - Both paths persist a `consultations` row first (status draft/processing/complete/error), then render results via `components.soap_display.render_soap_template()` and `components.billing_display.render_billing_forms()`.
- **Results** (`page_results`) — loads a consultation's flags/timing JSON and renders the SOAP template + billing dashboard for its `output_dir`.
- **History** (`page_history`) — lists past consultations from `db.list_consultations()` with an "Open" action that reloads one into Results.

### `pipeline_runner.py` — bridge to the core pipeline

- `get_runs_dir()` resolves `doctor_ui/runs/` (overridable via `VETAI_RUNS_DIR`).
- `_prepare_doctor_ui_pipeline_env()` probes Postgres (`127.0.0.1:5432` by default); if unreachable, it sets `SKIP_BILLING_PIPELINE`/`SKIP_PHASE2` so **SOAP generation still runs without inventory grounding or a billing dashboard**, rather than failing the whole consultation.
- `_apply_latency_env_defaults()` sets performance env flags (`PHASE2_BLOCKING=false`, `TARGET_60S=true`, `CHUNK_PARALLEL_ENABLED=true`, `FAST_TRANSCRIPTION=true`, etc.) so the UI defaults to a low-latency configuration.
- `run_pipeline_for_consultation()` (typed/edited-text path) and `run_pipeline_from_audio()` (single-click voice path) both extract/normalize the SOAP JSON (`_extract_soap_json`, with a fallback to reading `soap_note_*.json` from `output_dir`), detect failed generations (`_soap_looks_failed`), reformat via `soap_section_formatter.format_soap_dict`, persist to SQLite (`db.save_soap_result`), and merge in ASR UI latency (`_merge_timing_with_asr_ui`).

### `db.py` — local persistence

SQLite at `doctor_ui/data/consultations.db` (path overridable via `VETAI_DATA_DIR`). Single table `consultations`:

`id, created_at, doctor_name, pet_name, consultation_language, input_mode, step1_raw_text, audio_path, status (draft/processing/ready/complete/error), output_dir, soap_json, error_message`

CRUD helpers: `init_db`, `create_consultation`, `update_consultation` (dynamic column update), `get_consultation` (also JSON-parses `soap_json` into `soap_json_parsed`), `list_consultations`, `save_soap_result`.

### `diagnosis_schema.py` — Primary/Secondary diagnosis

Not a validator — prompt/display helpers backing the diagnosis split described in [SOAP Generation](#soap-generation-sections-formatting-conclusion): `DIAGNOSIS_SECTION_GUIDANCE` (prompt text), `migrate_legacy_diagnosis()` (back-fills `PrimaryDiagnosis` from the old single `DifferentialDiagnosis` field for older records), `diagnosis_text_for_storage()` (flattens both fields into one string for the SQLite column).

### `vitals_schema.py` — structured vitals template

Canonical vitals template (sourced from `vitals.xlsx`) via a `VitalsFieldDef` dataclass list: Body Weight (kg), Body Temperature (°F), Heart Rate (bpm), Respiratory Rate (brpm), Capillary Refill Time (enum), Mucous Membrane Color (enum), Body Condition Score (1–9), Pain Score (enum), Mentation (enum), Hydration Status (enum), Pulse Quality (enum). `build_vitals_prompt_block()` feeds the SOAP extraction prompt; `normalize_vitals()` accepts dict/JSON-string/legacy free-text and returns filled db_key→value pairs (filtering "not assessed"/"n/a"); `vitals_rows_for_display()` backs the UI's Field/Value/Unit table.

### `reminder_utils.py` — actionable reminder filtering

Classifies Phase 1 prose reminders and Phase 2 structured reminder atoms into **actionable** (needs scheduling: recheck, follow-up, vaccination, lab work, explicit time frames) vs **non-actionable/informational** (passive monitoring: "watch appetite", "if symptoms worsen"). Key functions: `is_actionable_reminder_text`/`is_actionable_reminder_item`, `filter_actionable_reminders_text`, `filter_actionable_reminder_items`, `merge_reminder_sources` (dedupes by item name), plus intent-recognition helpers (`normalise_intent`, `is_intent_recognised`, `intent_display`; `KNOWN_INTENT_CONTEXTS`: Performed/Prescribed/Administered/Ordered/Scheduled/Measured/Declined/Presented/Recommended/Future/Reminder). Powers both the SOAP note's Reminders section and the Results page's "Reminders & Follow-ups (Actionable)" billing panel, including a per-reminder "Schedule" date/time-picker UI stub (not yet backend-wired for notifications).

### `pipeline_logging.py` — run logging

`configure_pipeline_console_logging()` (runs at Streamlit startup) wires console logging for pipeline loggers (`soap_generator`, `doctor_ui`, `asr_providers`, `kb_phase2_integration`, `kb_ner_super_pass`, `kb_anchor_span`) and quiets noisy third-party loggers (httpx/httpcore/urllib3/openai/anthropic to WARNING). `init_pipeline_logging(output_dir)` additionally writes a per-consultation log file into `runs/{id}/`. `log_pipeline_banner()` writes START/COMPLETE banner lines. Level controlled by `VETAI_PIPELINE_LOG_LEVEL` (default DEBUG).

### `languages.py` — language options

`LANGUAGE_OPTIONS`: Auto (multi) → Deepgram `multi` (default), English → `en`, Hindi → `hi`, Kannada → `kn`, Hindi + English mix → stored `multi_hi_en` (Deepgram `multi`), Kannada + English mix → stored `multi_kn_en` (Deepgram `multi`). Helpers convert between UI label, stored DB value, and Deepgram ASR language code.

### `components/` — display layer

- **`soap_display.py`** — renders the Clinical Note (Subjective/Objective/Assessment/Plan/Conclusion/KeyIssues/AbnormalFindings/CustomerInstructions/Reminders), an "Additional fields" expander (Primary/Secondary Diagnosis, Protocols, Vitals table), a "Workflow health" panel (🟢/🟡/🔴 from `pipeline_flags_*.json`), a "Pipeline timing" panel (`render_pipeline_timing()`/`load_pipeline_timing()`), and a SOAP JSON download button.
- **`billing_display.py`** — the Phase 2 "Billing & Pharmacy" dashboard: loads `verification_dashboard_*.json` and `knowledge_atoms_*.json`, auto-fills billing rows (dose/route/frequency/stock_id/service_id) from knowledge atoms and inventory-suggestion match scores (`BillingContext`, `load_verification_dashboard`, `load_knowledge_atoms`, `auto_fill_item`, `_best_suggestion`), and renders expandable, editable sections per category (`_render_procedure_row`, `_render_inventory_row`, `_render_administered_row`, `_render_prescription_row`, `_render_reminder_row`, `_render_diagnostic_row`) for Procedure/Service, Inventory items, Administered meds, Prescriptions, Diagnostics (Labs/Imaging), Vitals, Clinical History, Reminders & Follow-ups, and issue-based Medicine Recommendations — each with status badges (AUTO-FILLED/ACTION_REQUIRED/DRAFT) and a dashboard JSON download button. Explains gracefully when Phase 2 was skipped (Postgres unreachable).

### Data & run artifacts

- **`doctor_ui/data/`** — `consultations.db` (SQLite; the file tracked as modified in git status for this repo).
- **`doctor_ui/runs/`** — per-consultation artifact directories (`runs/{consultation_id}/`, plus ad hoc test dirs like `diagnosis_verify/`, `baseline_timing/`), each holding the full pipeline trace: `raw_transcription_*.txt`, `step1_asr_metadata.json`, `cleaned_transcript_*.txt` (+ `_entities_by_kind.json`), `brain_ner_output_*.json`, `entity_manifest_*.json`, `grounding_layer_output_*.json`, `knowledge_atoms_*.json`, `structured_transcript_*.json`, `soap_note_*.json/.txt`, `verification_dashboard_*.json`, `pipeline_flags_*.json` (+ `_latest.json`), `pipeline_timing_*.json` (+ `_latest.json`), and occasional `soap_generator_*.log` run logs.
- **`requirements-ui.txt`** — key deps: `streamlit>=1.32`, `openpyxl` (vitals.xlsx), `jiwer` (WER/ASR eval), `python-dotenv`, `openai>=1.0.0`, `requests>=2.28.0`, `psycopg2-binary>=2.9.0` (Postgres/inventory grounding), `flashtext>=2.7`, `numpy`.

---



## Benchmarking & Evaluation

A self-contained benchmarking toolkit, separate from the core pipeline, for measuring ASR quality and full-pipeline quality/latency.

### ASR benchmark — `asr_benchmark.py`

Scores Step-1 ASR transcripts against Gemini-generated reference transcripts using WER/CER (via `jiwer`). `score_pair(reference, hypothesis)` returns wer, cer, substitution/deletion/insertion/hit counts, word counts. `run_benchmark(hypothesis_root, reference_dir, run_id=None, strip_punct=False, normalize_numbers=False)` discovers per-audio "slugs" under a hypothesis root, pairs each with `{slug}.txt` in the reference dir, and pulls `latency_ms` from `step1_asr_metadata.json`. CLI:

```powershell
python asr_benchmark.py --hypothesis-root benchmark_runs/deepgram_nova-3 --reference-dir benchmark_runs/references/gemini --output benchmark_runs/reports/asr_benchmark_<ts>
```

Flags: `--run-id`, `--normalize-punct`, `--normalize-numbers`. Writes `.json` (full report + summary: files_scored, mean/median WER & CER, mean latency) and `.csv` (per-file rows) to `benchmark_runs/reports/`.

### Full-pipeline benchmark — `pipeline_benchmark.py`

Stage-wise scorer for Steps 1–4 (ASR → cleaned English → entities/NER → SOAP facts), reusing `score_pair`. Adds `score_entities()` (precision/recall/F1 against a gold entity list) and `score_soap_facts()` (fraction of must-have facts found in generated SOAP text). Reads a "gold pack" per slug from `benchmark_runs/references/full/{slug}/` (`transcript.txt`, `cleaned_english.txt`, `entities.json`, `soap_facts.json`), falling back to the Gemini reference for Step 1. CLI:

```powershell
python pipeline_benchmark.py --hypothesis-root benchmark_runs/deepgram_nova-3 --reference-root benchmark_runs/references/full --gemini-fallback-dir benchmark_runs/references/gemini --output benchmark_runs/reports/pipeline_benchmark
```

Outputs `.json`/`.csv` with mean_step1/step2 WER/CER, mean_entity_f1, mean_soap_fact_recall.

### Batch runner — `batch_pipeline_runner.py`

Batch-runs ASR-only or the full SOAP pipeline over a directory of audio files (`--input-dir`, default `input_audio_examples`). Modes: `asr-only` (Step 1 only via `asr_providers.transcribe`) or `full` (invokes `generate_soap_note_from_audio_async`). Flags: `--asr-provider`, `--asr-model`, `--asr-language`, `--include`/`--exclude` (glob filters), `--dry-run`. Writes to `benchmark_runs/{provider}_{model}/{slug}/` (flat layout; re-runs auto-archive prior artifacts to `archive/{timestamp}/`) — `manifest.json`, `step1_raw_transcription.txt`, `step1_asr_metadata.json`, and (full mode) `cleaned_transcript_*.txt`, `entity_manifest_*.json`, `soap_note_*.json`.

### Shared helpers — `benchmark_utils.py`

`audio_slug()`, `make_run_id()`, `make_latest_run_dir()`/`make_run_dir()` (flat vs. legacy timestamped output paths), `archive_run_dir()`, `find_latest_run_dir()`/`find_latest_glob()`, `gemini_reference_path()`, `full_reference_dir()`, `save_step1_artifacts()`, `build_manifest()`/`write_manifest()`, `normalize_text()`. Path constants: `BENCHMARK_ROOT`, `REFERENCES_GEMINI_DIR`, `REFERENCES_FULL_DIR`, `REPORTS_DIR`, `CLINICAL_GOLD_DIR`.

### Supporting scripts and layout

- `scripts/benchmark_parallel_asr.py`, `scripts/compare_asr_models.py`, `scripts/diagnose_asr.py` — newer ASR comparison/diagnostic tooling (paired with `asr_providers.py`'s expanded benchmarking support).
- `scripts/latency_report.py`, `scripts/generate_baseline_timing_demo.py` — latency benchmarking/reporting built on `pipeline_timing.py`.
- `scripts/reset_benchmark_runs.py` — resets the benchmark corpus between runs.
- `benchmark_runs/` top level: `ASR_INVENTORY.json`, `clinical_gold/`, `deepgram_nova-3/` (per-slug run outputs), `references/{full,gemini}/` (gold data), `reports/` (generated `asr_benchmark.{json,csv}`, `pipeline_benchmark.{json,csv}`).
- `reports/asr_compare_summary.md` — example hand-written model-comparison report (Deepgram nova-3 vs nova-2 on a long-audio file: latency/WER/CER table, plus infra findings `ASR_PREP_WAV`, `ASR_PARALLEL_CHUNKS`).
- Docs `docs/ASR_BENCHMARK.md` and `docs/PIPELINE_BENCHMARK.md` document the end-to-end workflow (reset corpus → configure ASR via env `ASR_PROVIDER`/`ASR_MODEL`/`ASR_LANGUAGE`/`DEEPGRAM_API_KEY` → batch-run → manually create Gemini reference transcripts → score). Dependencies: `jiwer requests openpyxl`.

---



## Deployment

**Files:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `DEPLOY.md`, `.env.deploy.example`.

### Containerization

`Dockerfile` builds from `python:3.12-slim-bookworm`, installs `requirements-deploy.txt`, runs as non-root user `vetai`, exposes port 8501, and launches the Doctor UI (`streamlit run doctor_ui/app.py`). Includes a `HEALTHCHECK` hitting `/_stcore/health`. Bakes in `VETAI_DATA_DIR=/data`, `VETAI_RUNS_DIR=/runs`.

`docker-compose.yml` defines a single service, `doctor-ui` (no bundled Postgres/Redis — Postgres runs on the host and is reached via `host.docker.internal`). Publishes `${VETAI_PORT:-8501}:8501`, loads `.env` via `env_file`, sets `PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER` (password/API keys from `.env`), mounts two named volumes (`vetai_data` → `/data` for `consultations.db`, `vetai_runs` → `/runs` for pipeline artifacts), and has `restart: unless-stopped` + healthcheck.

`.dockerignore` excludes `__pycache__`, venvs, `.env*`, local data/runs dirs, `SQL_dump*`, audio files, `.git`, IDE folders, `agent-transcripts/`.

### Required environment (`.env.deploy.example`)

API keys: `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `FIREWORKS_API_KEY`. ASR: `ASR_LANGUAGE`. Model selectors: `SUPER_PASS_MODEL`, `CER_MODEL`, `SOAP_MODEL`, `PHASE2_MODEL`, `LLM_JUDGE_MODEL`, `BATCH_INTENT_MODEL`. Postgres: `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`, `CLINIC_ID`. Grounding tuning: `GROUNDING_AUTO_BIND_THRESHOLD`, `CATEGORY_HARD_GATE_RECALL_FALLBACK`. Port: `VETAI_PORT`.

### Deploy process (`DEPLOY.md`)

1. Copy `.env.deploy.example` → `.env` with real keys and `PGPASSWORD`; keep the host Postgres running (see `scripts/setup_phase2_database.py` for bootstrapping it).
2. `docker compose up -d --build`.
3. Open `http://localhost:8501`.

Ops commands: `docker compose logs -f doctor-ui`, `restart`, `down` (keeps volumes) vs `down -v` (destroys data — DEPLOY.md warns against this). Data persistence: SQLite consultation index + SOAP JSON in the `vetai_data` volume; full run artifacts in `vetai_runs` — both overridable via `VETAI_DATA_DIR`/`VETAI_RUNS_DIR`.

An alternate "Option C" in `DEPLOY.md` documents exposing the app publicly via a Cloudflare quick tunnel (`scripts/run_doctor_ui_lan.ps1` + `scripts/deploy_public_tunnel.ps1`), noting free tunnel URLs are ephemeral.

---



## Documentation & Legacy


| Location                                               | Content                                                                                                                                                                                                                                           |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **UNIFIED_CLEANING_AND_NER_PROMPT.md**                 | **Required for pipeline.** Unified cleaning + NER prompt; loaded by `kb_ner_super_pass.py` from project root. Do not move to legacy.                                                                                                              |
| **BRAIN_NER_PROMPT_UPDATED.md** (project root)         | Canonical Brain NER prompt: entity kinds, inventory_category list, service_category list, skeleton format (13 fields), query_expansion (13th field), form-factor preservation in normalized_name                                                  |
| **LOCAL_GROUNDING_SEARCH_AND_RETRIEVAL.md**            | Local grounding: retrieval (trigram, phonetic, vector), scoring (domain, category, suggestion/hints), decision flow (auto-bind, Option A/B/C). See also GROUNDED_CONCEPTS_QUERY_EXPANSION_DOCUMENTATION.md.                                       |
| **GROUNDED_CONCEPTS_QUERY_EXPANSION_DOCUMENTATION.md** | Query expansion, hints, form-factor & route-to-form alignment, production safeguards (symptom suppressor, Golden Gate for Diagnosis/ReasonForVisit, ID-based dashboard), Judge rules, SOAP prompt (hints/query_expansion), summary table of files |
| **new_master_documentation.md**                        | **This file.** Current source of truth: full pipeline (incl. chunk-parallel Super-Pass, 0.92 auto-bind threshold, Conclusion section, pipeline observability), Phase 2, Doctor UI, benchmarking, deployment, config                              |
| **MASTER_DOCUMENTATION (1).md**                        | Prior version (v3.4, 2026-02-25). Superseded in content by this file; retained for historical reference only — do not treat as current.                                                                                                          |
| **docs/DOCTOR_UI.md**                                  | Doctor UI quick-start: setup, run command, language dropdown, feature overview. See [Doctor UI (Streamlit App)](#doctor-ui-streamlit-app) here for the full architecture.                                                                        |
| **docs/ASR_BENCHMARK.md**                              | ASR benchmarking workflow (corpus reset, batch-run, Gemini reference creation, scoring) — see [Benchmarking & Evaluation](#benchmarking--evaluation).                                                                                             |
| **docs/PIPELINE_BENCHMARK.md**                         | Full-pipeline stage-wise benchmarking workflow — see [Benchmarking & Evaluation](#benchmarking--evaluation).                                                                                                                                      |
| **DEPLOY.md**                                          | Docker/compose deployment process — see [Deployment](#deployment).                                                                                                                                                                                 |
| **README.md**                                          | Project overview and run instructions                                                                                                                                                                                                             |
| **legacy/docs/**                                       | Audit reports, RCA docs, latency/quality reports, old pipeline flows, prompt comparisons, implementation plans (archived); copy of UNIFIED_CLEANING_AND_NER_PROMPT.md kept there for reference only                                               |
| **legacy/python/**                                     | Backfills, run_pipeline_to_intent, audit scripts, one-off scripts (check_local_inventory_spirocoxin, test_brain_ner_on_cleaned, debug_ontology_harvest, etc.)                                                                                     |
| **legacy/scripts/**                                    | check_service_master_physiotherapy, check_local_inventory_bravecto, run_ortolani_rca_checks, kb_domain_auditor, etc.                                                                                                                              |
| **legacy/README.md**                                   | List of legacy files and how to run them                                                                                                                                                                                                          |


---

**End of Master Documentation (v4.0)**

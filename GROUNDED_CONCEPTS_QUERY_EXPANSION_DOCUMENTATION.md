# Grounded Concepts: Query Expansion & Hints — Complete Documentation

This document describes the **query expansion** and **hints** features used to ground clinical entities (e.g. medications, procedures, diagnostics) to the knowledge base and clinic inventory, plus the **form-factor and route-to-form alignment** ("Golden Thread") that keeps extraction, Judge, and SOAP consistent. It covers: pipeline role of hints vs query expansion, all code changes by file, Brain NER updates (including normalized_name form-factor preservation), Judge integration (including form-factor/route rules and correct API usage), and the Phase 2 / Verification Dashboard fix.

---

## 1. Overview: Grounded Concepts, Hints, and Query Expansion

### 1.1 What “Grounded Concepts” Means

A **grounded concept** is a clinical mention from the transcript (e.g. “Expense exotic pump”) that has been:

1. **Normalized** by Brain NER (e.g. to “Easotic ear drops”).
2. **Matched** to a KB and/or clinic entity (inventory item or service) via local/global search.
3. **Decided** by the decision flow (auto-bind, fast-track, or LLM Judge) to produce a final link (e.g. `local_stock_id` 41358 for EASOTIC 10ML).
4. **Reflected** in the entity manifest and, after Phase 2, in knowledge atoms and the Verification Dashboard.

### 1.2 Hints vs Query Expansion — Purpose and Difference

| Aspect | **Hints** | **Query Expansion** |
|--------|-----------|----------------------|
| **Purpose** | Alternative phrasings for the **same** concept (synonyms, corrections, context). | **Phonetic / ASR correction**: likely **brand or product names** when the transcript term sounds like a known product but is garbled. |
| **When used** | Always available (1–3 per entity). Used for retrieval and scoring. | Only when the term “sounds like” a medication/product (e.g. “exotic pump” → Easotic; “cefeped syrup” → CefPET). |
| **Format** | `hint:probability` (e.g. `Easotic:0.9,ear drops:0.85`). | Comma-separated list of up to **3** likely brand/product names (e.g. `Easotic,Easotic 10ml,Virbac Easotic`). |
| **Retrieval** | Search uses `search_term` + hints (local and global). | **Extra searches**: separate local (and global) searches are run for each expansion term (capped at 3) to pull in SKUs that match the brand name but not the mangled span. |
| **Scoring** | Candidate that matches a hint gets a **weighted suggestion boost** (by hint probability). | Candidate that matches a query_expansion term gets a **fixed-weight boost** (`QUERY_EXPANSION_BOOST_WEIGHT`, default 0.8). |
| **Judge** | Passed as **BRAIN_HINTS** in the Batch Judge prompt. | Passed as **QUERY_EXPANSIONS**; if a candidate matches a QUERY_EXPANSION term, Judge is instructed to **FAVOR acceptance** (phonetic bridge). |
| **Cap** | 3 hints (parser/super_pass). | 3 terms (everywhere: parser, manifest, local/global, Judge). |

**Why both?**

- **Hints** improve recall and ranking for normal synonyms and context (“ear flush”, “otitis treatment”).
- **Query expansion** specifically addresses **ASR/pronunciation errors** where the written form is wrong but the intended product is identifiable (e.g. “Expense exotic pump” → Easotic). Without expansion, search on the mangled text often misses the correct SKU; with it, targeted searches on “Easotic”, “Easotic 10ml”, “Virbac Easotic” bring EASOTIC 10ML into the candidate set, and the Judge can accept it using the expansion bridge.

---

## 2. Pipeline Process: Where Hints and Query Expansion Are Used

High-level flow:

1. **Upstream NER** → entities with `span_text`, `kind`, etc.
2. **Brain NER (Super Pass)** → enriches each entity with `normalized_name`, `search_term`, `hints`, **`query_expansion`** (13th skeleton field), domains, categories, probabilities.
3. **Skeleton parser** → parses 13-field skeleton; caps hints and **query_expansion** at 3.
4. **Grounding (parallel)**  
   - **Local path**: for each entity, run local inventory + local services with `search_term`; then **for each query_expansion term** (up to 3), run additional local inventory + services and merge unique candidates. Pass `hints` and `query_expansion` into local search for **suggestion boost** (hint-weighted or query_expansion weight).  
   - **Global path** (batch): build `need_global` as **9-tuple** `(entity_idx, search_term, kind_filter, span_text, hints, domain, suggestion_prob, hint_probs, query_expansion_list)`. Batch global search uses hints and **query_expansion** for embedding (if combined) and for **suggestion boost** when scoring candidates.  
   - **Decision flow**: auto-bind / fast-track / LLM Judge. **No** compulsory Judge solely because query_expansion was used; same flow as hints. When Judge runs, it receives **BRAIN_HINTS** and **QUERY_EXPANSIONS** and applies the expansion-bridge rule.
5. **Entity manifest** → each entity gets `hints`, **`query_expansion`** (capped at 3), and final `local_stock_id` / `local_service_id` / `match_method`.
6. **Phase 2** → knowledge atoms are enriched with manifest IDs; **distinctive-token match** (Strategy 3b) links atoms like “Easotic ear drops” to manifest “EASOTIC 10ML” so the Verification Dashboard shows **Confirmed** instead of ACTION REQUIRED.

---

## 3. Code Changes by File

### 3.1 Brain NER Prompt & Skeleton (13th Field)

**File:** `BRAIN_NER_PROMPT_UPDATED.md` (project root)  
**File (prompt variable):** `kb_ner_super_pass.py` (CLINICAL_ENTITY_EXTRACTION_PROMPT)

**Changes:**

- Skeleton format extended to **13 pipe-separated fields**; the 13th is **`query_expansion`**.
- **Format:**  
  `id|span_text|normalized_name|kind|domains|inv_cats|svc_cats|corr_prob|sugg_prob|hints|is_new|context|query_expansion`
- **query_expansion** definition in prompt:
  - When the transcript term **sounds like** a known medication/product but is spelled phonetically or garbled (e.g. “exotic pump” for ear drops, “cefeped syrup” for cefepime), add **up to 3** comma-separated likely brand/product names (e.g. `Easotic,Easotic 10ml,Virbac Easotic`).
  - Used for extra KB search and scoring; leave empty when the term is already clear or not a medication/product.
- Example line added:  
  `e5|Expense exotic pump|Easotic ear drops|Medication|dermatology|Medication|Consultation|0.65|0.88|Easotic:0.9,ear drops:0.85,antifungal ear:0.75|0|Expense exotic pump applied to both ears.|Easotic,Easotic 10ml,Virbac Easotic`
- Optional-fields summary updated to include `query_expansion` (13th skeleton field, up to 3 terms).
- **NORMALIZED_NAME RULES — FORM-FACTOR PRESERVATION (see also §7.2):** The prompt instructs the model to **include the delivery form in `normalized_name`** when the transcript or context contains unit or route cues: liquid (ml, cc, syrup, suspension, drops) → e.g. "Cefpodoxime Syrup"; solid (mg, tablet, tab, capsule) → e.g. "Cefpodoxime Tablet"; injectable (inject, vial, IM, IV) → e.g. "Amoxicillin Injection"; topical (apply, cream, spray, pump, drops in ear/eye) → e.g. "Easotic ear drops". Example: "3 ml of Cefped" or "Cefpet syrup 3ml" → normalized_name "Cefpodoxime Syrup" or "Cefpodoxime Oral Suspension", not just "Cefpodoxime". This makes the extraction layer the source of truth for form so the Judge and SOAP stay aligned.

---

### 3.2 Skeleton Parser

**File:** `kb_ner_skeleton_parser.py`

**Changes:**

- **Format comment:**  
  `Format: id|span_text|normalized_name|kind|domains|inv_cats|svc_cats|corr_prob|sugg_prob|hints|is_new|context|query_expansion`
- **Parsing (13th field):**
  - `query_expansion_raw = parts[12].strip() if len(parts) > 12 else ""`
  - Split by comma; strip; skip placeholders (`0`, `none`, `null`, `n/a`, `-`, `--`).
  - **Cap:** `query_expansion = query_expansion[:3]` (same cap as hints).
- Entity dict now includes `"query_expansion": query_expansion`.
- If `parts` has fewer than 13 elements, pad with empty strings so `parts[12]` exists.

---

### 3.3 Brain NER Super Pass

**File:** `kb_ner_super_pass.py`

**Changes:**

- Skeleton format line in prompt: 13 fields including **query_expansion**.
- QUERY_EXPANSION note: “When the transcript term sounds like a known medication/product but is spelled phonetically (e.g. ‘exotic pump’), add up to 3 comma-separated likely brand names (e.g. Easotic,Easotic 10ml,Virbac Easotic). Used for retrieval only; Judge will validate. If none, leave empty.”
- Schema/description for compressed skeleton: “query_expansion=up to 3 comma-separated likely brand names when term is phonetic”.
- **Post-parse handling of `query_expansion`:**
  - If `e.get("query_expansion")` is a list: `entity_obj["query_expansion"] = [str(x).strip() for x in qe_raw if str(x).strip()][:3]`.
  - If string (e.g. from CSV): `entity_obj["query_expansion"] = [x.strip() for x in qe_raw.split(",") if x.strip()][:3]`.
  - Else: `entity_obj["query_expansion"] = []`.
- Same 3-term cap applied in the fallback/audit path: `entity_dict["query_expansion"] = [str(x).strip() for x in qe if str(x).strip()][:3] if isinstance(qe, list) else []`.

---

### 3.4 Grounding Parallel (Orchestration)

**File:** `kb_ner_parallel.py`

**Changes:**

1. **No compulsory Judge for query_expansion**  
   All conditions that previously required “and not used_query_expansion” were **removed**. So:
   - Auto-bind and high-certainty fast-track can run **even when** query_expansion was used for retrieval.
   - Query_expansion is treated like hints: extra retrieval and boost only; same decision flow.

2. **Local path (LOCAL_ONLY):**
   - After initial local inventory + services with `search_term`, **query expansion searches**: for each term in `entity.get("query_expansion")[:3]`, run `_local_inventory(q)` and `_local_services(q)`; merge new candidates (by `(stock_id, service_id)`) into `inventory_results` / `services_results`.
   - `used_query_expansion` is set to True when any expansion search is run (used only for logging; **no** longer used to force Judge).

3. **Passing hints and query_expansion into local search:**
   - When calling `search_local_inventory_topk` and `search_local_services_topk`, the task dict includes `entity_hints` and `entity_query_expansion` (each capped at 3).
   - These are passed as `hints=task["entity_hints"]`, `query_expansion=task.get("entity_query_expansion") or None`.

4. **Preserve query_expansion in entity result:**  
   After merging local result into entity, if `entity_result` does not already have `query_expansion`, copy from `source_entity.get("query_expansion")` (list, preserved as-is or sliced).

5. **Batch global: `need_global` tuple extended to 9 elements:**
   - From: `(idx0, search_term, kind_filter, span_text, hints, domain_arg)` (or similar).
   - To: `(idx0, search_term, kind_filter, span_text, (hints or [])[:3], domain_arg, suggestion_prob, hint_probs, query_expansion_list)`.
   - `query_expansion_list = [str(x).strip() for x in qe if str(x).strip()][:3] if isinstance(qe, list) else []`.

6. **Judge (apply_decision_flow):**
   - When building batch items for Judge (e.g. `unresolved_batch_items`), include `hints` and `query_expansion` from each entity (e.g. `entity.get("hints")`, `entity.get("query_expansion")`, normalized and capped).
   - When calling `apply_decision_flow` (single-entity Judge path), pass `hints=entity.get("hints")` and `query_expansion=entity.get("query_expansion")` (e.g. `_hints`, `_qe` capped at 5 and 3 respectively).

---

### 3.5 Local Search

**File:** `kb_ner_local_search.py`

**Changes:**

1. **`search_local_inventory_topk`** (and similarly **`search_local_services_topk`**):
   - New parameter: **`query_expansion: Optional[List[str]] = None`**.
   - In the suggestion-boost branch: if the candidate does **not** match `search_term` or any **hint**, then check **query_expansion** terms (e.g. `expansion_terms = query_expansion or []`); for each `exp_term` in `expansion_terms[:3]`, if the candidate name (e.g. `pname_lower`) contains or is contained in `exp_term`, apply:
     - `qe_weight = float(os.getenv("QUERY_EXPANSION_BOOST_WEIGHT", "0.8"))`
     - `suggestion_boost_val = base_boost * qe_weight`
   - Log as e.g. “candidate matched query_expansion ‘…’ → boost=…”.

2. **Config:**  
   `QUERY_EXPANSION_BOOST_WEIGHT` (default **0.8**).

---

### 3.6 Global Search (Batch)

**File:** `kb_ner_global_search.py`

**Changes:**

1. **Item tuple (from parallel):**  
   Items are now at least **9-tuples**: `(entity_idx, search_term, kind_filter, original_span, hints, domain, suggestion_prob, hint_probs, query_expansion_list)`.

2. **Parsing (9th element):**
   - `if len(it) >= 9 and it[8] is not None`: parse **query_expansion** (list or comma-separated string); normalize to list of strings (strip, lower), cap at 3.
   - Store in **`entity_query_expansion_terms_by_idx[entity_idx]`**.

3. **Suggestion boost (two scoring branches — dual-signal and non-dual-signal):**
   - After checking **hints** (and applying hint-probability-weighted boost), if no hint matched:
     - Get `expansion_terms = entity_query_expansion_terms_by_idx.get(entity_idx, [])`.
     - For each `exp_term`, if candidate name matches (substring or contains):
       - `qe_weight = float(os.getenv("QUERY_EXPANSION_BOOST_WEIGHT", "0.8"))`
       - `suggestion_boost = base_boost * qe_weight`
       - `hint_match_source = f"query_expansion:{exp_term}"`
       - break.
   - Same logic in both branches so that candidates matching a query_expansion term get a fixed-weight boost when they don’t match a hint.

---

### 3.7 Disambiguation (LLM Judge)

**File:** `kb_ner_disambiguation.py`

**Changes:**

1. **Batch Judge prompt (`_build_batch_judge_prompt`):**
   - New subsection **“EXPANSION BRIDGE (phonetic / ASR correction)”**:
     - “BRAIN_HINTS and QUERY_EXPANSIONS are the clinical engine’s best-guess alternatives for the mention (e.g. ‘Expense exotic pump’ → hints like ‘Easotic’, ‘ear drops’; query_expansions like ‘Easotic’, ‘Easotic 10ml’).”
     - “If a LOCAL candidate’s name matches a term in QUERY_EXPANSIONS, that indicates a high-confidence phonetic correction. You should FAVOR this match even if ORIGINAL_MENTION looks mangled.”
     - “If a candidate matches a QUERY_EXPANSION term, treat it as strong evidence toward ACCEPTANCE (do not reject solely due to lexical mismatch with ORIGINAL_MENTION).”
   - Per-item lines added:
     - `BRAIN_HINTS: [{hints_str}]`
     - `QUERY_EXPANSIONS: [{qe_str}]`
   - Batch items carry `hints` and `query_expansion` (from entity); up to 5 hints and 5 expansion terms shown in prompt (backend still caps at 3 for retrieval).

2. **`_submit_to_batch_judge` (or equivalent batch item builder):**  
   Each item includes `"hints": it.get("hints") or []`, `"query_expansion": it.get("query_expansion") or []`.

3. **`apply_decision_flow` (and single-entity Judge path):**
   - New optional parameters: **`hints`**, **`query_expansion`**.
   - Docstring: “hints and query_expansion are passed to the Judge prompt (BRAIN_HINTS / QUERY_EXPANSIONS) for phonetic bridge.”
   - When building the payload for the Judge (single or batch), include `"hints": (hints or [])[:5]`, `"query_expansion": (query_expansion or [])[:5]`.

4. **`disambiguate_local_match_v2` / callers:**  
   Signatures and call sites updated to pass `hints` and `query_expansion` through to the Judge.

5. **Judge client and endpoint (no cross-provider fallback):**  
   GPT/OpenAI models must use the OpenAI API; Fireworks models must use the Fireworks API. The Judge always uses the client from `get_client_for_model(LLM_JUDGE_MODEL)`. If that returns `None` (e.g. no OPENAI_API_KEY for `gpt-4.1-nano`), the Judge is skipped with a clear error message—no silent fallback to a Fireworks model. This avoids 404 "model not found" and keeps billing logic on the correct endpoint.

6. **Form-Factor & Route-to-Form Alignment (see §7):**  
   The Batch Judge prompt includes rule **"4. FORM-FACTOR & ROUTE-TO-FORM ALIGNMENT (CRITICAL)"** and per-item evaluation step **"3. FORM-FACTOR / ROUTE CHECK"**. The legacy single-entity Judge prompt also includes a FORM-FACTOR & ROUTE ALIGNMENT paragraph.

---

### 3.8 Phase 2 Integration (Manifest → Atoms & Dashboard)

**File:** `kb_phase2_integration.py`

**Changes:**

1. **`_compact_manifest_for_prompt`**
   - Each compact entity now includes **`display_name`** and **`kb_preferred_name`** (in addition to `span_text`, `normalized_name`, `entity_id`, `kind`, `kb_kind`, `kb_concept_id`, `local_stock_id`, `local_service_id`, `match_method`).
   - So the Phase 2 LLM sees the grounded inventory/service display names and can align Plan atoms to the correct manifest entry.

2. **`_enrich_atoms_with_manifest_ids`**
   - **Manifest indexing:**  
     Already indexed by `normalized_name`, `span_text`, **`display_name`**, **`kb_preferred_name`** (all four used in `names_to_index` for exact and name-based lookups). So “Easotic ear drops” (atom) can match a manifest row whose display name is “EASOTIC 10ML” when we use token overlap.
   - **Strategy 3b — Distinctive-token match for ID-bearing manifest:**
     - When the atom concept (e.g. “Easotic ear drops”) did **not** match any manifest entry by exact name, substring, or 2+ word overlap, and the manifest has entries with `local_stock_id` or `local_service_id`:
     - Build **concept_tokens**: words from normalized concept with length ≥ 4 (or ≥ 3 if none), excluding stop list: `the`, `and`, `for`, `with`, `from`, `ear`, `eye`, `drops`, `tablet`, `capsule`, `syrup`, `injection`, `mg`, `ml`, `tab`, `caps`.
     - For each manifest entry that has an ID, build token set from **normalized_name**, **span_text**, **display_name**, **kb_preferred_name**.
     - If **kind** is compatible and there is **at least one shared distinctive token** (overlap ≥ 1), treat that manifest entry as the match and copy its `local_stock_id` / `local_service_id` (and other metadata) onto the atom.
     - This links “Easotic ear drops” to the manifest row “EASOTIC 10ML” (shared token “easotic”) so the atom gets `local_stock_id` 41358 and the Verification Dashboard shows it as **Confirmed** instead of ACTION REQUIRED.

3. **Verification Dashboard**  
   The dashboard is built from the same enriched atoms and entity manifest. No separate code change in the dashboard builder was required; fixing the atom–manifest stitching (Strategy 3b + display names in compact manifest) fixes the “Easotic ear drops” showing as unlinked.

---

### 3.9 SOAP Note Prompt

**File:** `SOAP_notes_phase1_experiment.py` — `build_soap_prompt_from_brain_ner()`, `build_detected_concepts_json_from_manifest()`

**Payload sent to SOAP (BRAIN_NER_JSON):**

- Built by `build_detected_concepts_json_from_manifest(entity_manifest)`. Each entity in the payload includes: `entity_id`, `span_text`, `kind`, `normalized_name`, `search_term`, `correctness_probability`, `suggestion_probability`, **`hints`**, **`query_expansion`**, `assertion_id`, `inventory_category`, `service_category`, `attributes`.
- **hints:** List of clinical clues (e.g. brand or product terms) from Brain NER.
- **query_expansion:** Up to 3 likely brand/product names for phonetic or ASR-mangled mentions (e.g. "exotic pump" → ["Easotic", "Easotic 10ml", "Virbac Easotic"]).

**SOAP prompt instructions:**

- **"Use BRAIN_NER_JSON as a clinical checklist"** and **"Prefer `search_term` / `normalized_name` from Brain NER for medical wording."**
- **"Hints and query_expansion:"** The prompt instructs the SOAP model to treat `hints` and `query_expansion` as the intended clinical meaning: prefer wording that aligns with `normalized_name` and with any non-empty hints/query_expansion (e.g. if span_text is "exotic pump" and query_expansion includes "Easotic", write "Easotic" or "Easotic ear drops" in the note, not the raw transcript phrase).
- Use `suggestion_probability` and `correctness_probability` as confidence signals; respect `assertion_id`; no anchor markup; do not invent facts.
- **Form-factor:** No separate form-factor line; consistency comes from NER (form in `normalized_name`) and Judge (form-aligned candidate). SOAP uses normalized_name/hints/query_expansion as-is.

---

## 4. Configuration

| Variable | Default | Description |
|---------|---------|-------------|
| `QUERY_EXPANSION_BOOST_WEIGHT` | `0.8` | Weight applied to the suggestion boost when a candidate matches a **query_expansion** term (and does not match a hint). Same idea as a medium-confidence hint. |

---

## 5. End-to-End Example: “Expense exotic pump” → EASOTIC 10ML

1. **Brain NER**  
   Span “Expense exotic pump”; normalized_name “Easotic ear drops”; hints e.g. Easotic:0.9, ear drops:0.85; **query_expansion** “Easotic,Easotic 10ml,Virbac Easotic” (3 terms).

2. **Parser / Super pass**  
   Entity has `query_expansion = ["Easotic", "Easotic 10ml", "Virbac Easotic"]` (capped at 3).

3. **Local search**  
   - Search with `search_term` “Easotic ear drops” may or may not return EASOTIC 10ML.  
   - Additional searches for “Easotic”, “Easotic 10ml”, “Virbac Easotic” pull in EASOTIC 10ML.  
   - Candidate “EASOTIC 10ML” matches query_expansion term “Easotic” → suggestion boost with weight 0.8.

4. **Decision flow**  
   Auto-bind or Judge; if Judge, it sees BRAIN_HINTS and QUERY_EXPANSIONS and accepts the match (expansion bridge).

5. **Manifest**  
   Entity: span_text “Expense exotic pump”, normalized_name/display_name “EASOTIC 10ML”, `local_stock_id` 41358, `match_method` dual_sync_local, `query_expansion` [Easotic, Easotic 10ml, Virbac Easotic].

6. **Phase 2**  
   Plan atom concept “Easotic ear drops” or “EASOTIC 10ML” is stitched to the manifest entry (exact or Strategy 3b distinctive token “easotic”); atom gets `local_stock_id` 41358.

7. **Dashboard**  
   Medication appears under prescribed / pharmacy as **DRAFTED** with **linked_id** 41358 (Confirmed), not ACTION REQUIRED.

---

## 6. Hint Normalization (Robustness)

**File:** `kb_ner_parallel.py`

**Purpose:** Brain NER and CER can emit `hints` as a list of strings or as a list of dicts (e.g. `{"hint": "Easotic", "probability": 0.9}`). Downstream code must not assume one format.

**Changes:**

- **Helpers:** `_hint_item_to_str(h)` normalizes a single hint (string or dict with `"hint"` key) to a stripped string. `_normalize_hints(hints, max_items=5)` returns a list of such strings, capped.
- **Usage:** Applied when building Judge batch items, when creating local/global search tasks, when preserving entity metadata, and when building the final grounding output. This prevents `AttributeError: 'str' object has no attribute 'get'` and keeps behavior correct whether hints come from Brain NER (strings) or CER (dicts).

---

## 7. Form-Factor & Route-to-Form Alignment (Golden Thread)

To ensure **clinical safety and billing accuracy**, form-factor and route-of-delivery logic is enforced end-to-end. If Brain NER, the Judge, and the SOAP generator are not aligned, semantic drift can occur (e.g. Judge bills a syrup but SOAP tells the owner to give a tablet). The following keeps them in sync.

### 7.1 Principle

- **Extraction (Brain NER)** captures the delivery form in **normalized_name** when the transcript has unit/route cues (ml, syrup, tablet, inject, apply, etc.).
- **Judge** uses those cues to prefer candidates whose **physical form and route** match the vet’s mention (e.g. syrup → liquid candidate; tablet → solid; inject → injectable; apply → topical).
- **SOAP** consumes `normalized_name` and grounded manifest; once NER and Judge preserve form, SOAP text stays consistent.

### 7.2 Extraction Layer: Brain NER (normalized_name)

**Files:** `BRAIN_NER_PROMPT_UPDATED.md`, `kb_ner_super_pass.py` (fallback prompt)

**Changes:**

- **NORMALIZED_NAME RULES** in the Brain NER prompt now include **FORM-FACTOR PRESERVATION (CRITICAL)**:
  - When normalizing **medications or products**, always include the **delivery form** if the transcript or context contains unit or route cues.
  - **Liquid (oral):** ml, cc, syrup, suspension, oral solution, drops → e.g. `Cefpodoxime Syrup`, `Cefpodoxime Oral Suspension`, `Easotic ear drops`.
  - **Solid:** mg (without ml), tablet, tab, capsule, pill → e.g. `Cefpodoxime Tablet`, `PRODUCT_A 500mg Tablet`.
  - **Injectable:** inject, injection, IM, IV, vial, ampoule → e.g. `Amoxicillin Injection`.
  - **Topical/external:** apply, cream, ointment, spray, pump, drops (ear/eye context) → e.g. `Easotic ear drops`, `Antifungal cream`.
  - **Examples:**  
    - "3 ml of Cefped" or "Cefpet syrup 3ml" → normalized_name **"Cefpodoxime Syrup"** or **"Cefpodoxime Oral Suspension"** (not just "Cefpodoxime").  
    - "exotic pump" with context "apply once daily" → normalized_name **"Easotic ear drops"** (preserve drops/topical form).
- The **fallback** Brain NER prompt in `kb_ner_super_pass.py` (used when the .md file is missing) includes a one-line **normalized_name — FORM-FACTOR PRESERVATION** instruction so the same rule applies when the doc file is not loaded.

### 7.3 Disambiguation Layer: LLM Judge

**File:** `kb_ner_disambiguation.py`

**Changes:**

- **Batch Judge prompt (`_build_batch_judge_prompt`):**
  - New rule **"4. FORM-FACTOR & ROUTE-TO-FORM ALIGNMENT (CRITICAL)"**:
    - Match the **route and form** of the vet’s mention to the candidate’s formulation using administration and unit cues from ORIGINAL_MENTION and CONTEXT.
    - **Liquid (oral):** ml, cc, syrup, suspension, oral solution, drops, liquid → **prioritize liquid candidates**; **reject** solid (tablet/capsule) even if higher match score.
    - **Solid:** mg (without ml), tablet, tab, capsule, pill, cap, bolus → **prioritize solid**; reject oral liquids when the vet clearly indicated solid form.
    - **Injectable:** inject, injection, IM, IV, SC, Sub-Q, vial, ampoule → **prioritize injectables**; reject oral syrups/suspensions and topicals.
    - **Topical/external:** apply, topical, cream, ointment, spray, pump, drops (ear/eye) → **prioritize topical/external**; reject systemic oral or injectable when the vet clearly indicated topical route.
    - **CONFLICT RESOLUTION:** If the vet mentions a liquid unit (e.g. "3 ml") or "syrup"/"suspension", and the top-scoring candidate is a **tablet** while a **liquid** candidate exists in the list, **REJECT the tablet and SELECT the liquid candidate**. Same logic for solid vs liquid, and for injectable vs oral vs topical. The candidate’s physical form and route **must** align with the veterinarian’s intended delivery.
  - **Per-item EVALUATION** step **"3. FORM-FACTOR / ROUTE CHECK"**:
    - Mention has syrup/ml/drops/suspension but candidate is tablet/capsule → REJECT; choose a liquid candidate if present.
    - Mention has tablet/tab/mg (solid) but candidate is syrup/suspension → REJECT; choose a solid candidate if present.
    - Mention has inject/vial/IM/IV but candidate is oral syrup → REJECT; choose an injectable candidate if present.
    - Mention has apply/spray/pump/cream/drops (topical) but candidate is oral/injectable → REJECT; choose a topical candidate if present.
  - Top-level rule **"5. WHEN IN DOUBT"** (formerly "4.") unchanged.
- **Legacy single-entity Judge (`disambiguate_local_match`):**
  - New paragraph **FORM-FACTOR & ROUTE ALIGNMENT:** same logic (syrup/ml → liquid; tablet/tab → solid; inject/vial → injectable; apply/spray/pump → topical); instructs to select the candidate that matches form/route even if lower score. Selection criteria list updated to include "candidate's form/route matches the mention".

### 7.4 SOAP Generator

The SOAP generator receives consolidated Brain NER in **BRAIN_NER_JSON** (including **hints** and **query_expansion** per entity). The SOAP prompt explicitly instructs the model to treat hints and query_expansion as the intended clinical meaning and to prefer wording that aligns with them (e.g. write "Easotic ear drops" when query_expansion includes "Easotic", not the raw transcript phrase). It uses **normalized_name** and grounded manifest (display_name, kb_preferred_name) from Brain NER and the billing pipeline. No separate SOAP-prompt change was required for form-factor: once NER preserves form in normalized_name and the Judge selects the form-aligned candidate, the SOAP note reflects the correct product and form.

---

## 8. Production Safeguards (Symptom Suppressor & ID-Based Dashboard)

Two changes address production blockers: over-grounding of symptoms/findings to billable SKUs, and dashboard "orphans" when Plan text doesn’t match inventory display names.

### 8.1 Symptom / Physical Finding Suppressor (Judge)

**Problem:** ASR/transcript mentions like "bus inside the left ear" (pus) or "yeast growth" were grounded to Medications or Lab Kits (e.g. Lasix, AST KIT), creating clinical risk.

**Change:** The Judge prompt (batch and legacy) now includes an explicit **Symptom Suppressor** rule:

- **Batch Judge (`_build_batch_judge_prompt`):** New rule **"2.5. SYMPTOM / PHYSICAL FINDING SUPPRESSOR (CRITICAL - Safety)"**: If the span is a symptom or physical finding (e.g. pus, yeast, yeast growth, discharge, shaking, swelling), REJECT any match to Medications or Lab Kits/Reagents unless CONTEXT explicitly states the vet is prescribing or ordering that product/test. Examples: "bus inside the left ear" (ASR for pus) + Lasix → REJECT (NONE); "yeast growth" + AST KIT → REJECT (NONE).
- **Per-item EVALUATION:** New step **"2.5. SYMPTOM/FINDING CHECK"**: Is ORIGINAL_MENTION a symptom or physical finding? If yes and candidate is Medication or Lab Reagent → REJECT (NONE).
- **Legacy single-entity Judge:** New paragraph **"SYMPTOM / PHYSICAL FINDING SUPPRESSOR (Safety)"** with the same rule and examples.

**File:** `kb_ner_disambiguation.py`

### 8.2 ID-Based Dashboard Linking (Reduce Orphans)

**Problem:** "Cefpodoxime syrup" (Plan atom) appeared as ACTION REQUIRED (Unlinked) because the manifest used inventory display name "CefPET Dry Syrup"; string matching failed to link atom to manifest.

**Change:** Linking uses **local_stock_id / local_service_id as the primary key**, not display-name string matching.

- **`kb_phase2_integration.py` — `_enrich_atoms_with_manifest_ids`:**
  - New index **`manifest_by_normalized_name_id`**: (normalized_name_lower, kind) → entity, for manifest entries that have local_stock_id or local_service_id.
  - New **Strategy 1b:** Match atom to manifest by (atom concept_lower, kind) against this index (with kind_compatible fallback). So when the atom concept is "Cefpodoxime syrup" and a manifest entry has normalized_name "Cefpodoxime Syrup" and local_stock_id, the atom gets that local_stock_id even if display_name is "CefPET Dry Syrup".
- **`SOAP_notes_billing_phase2_kb_atoms.py` — `_build_verification_dashboard`:**
  - Comment clarified: linking and status are determined by **local_stock_id / local_service_id** (primary key); atoms receive these IDs from enrichment, which matches by manifest normalized_name + kind.

Result: Plan terms that match Brain NER normalized_name link to the correct SKU in the Verification Dashboard and show as linked (DRAFT/Confirmed), not ACTION REQUIRED.

### 8.3 Golden Gate: Diagnosis & ReasonForVisit (Service-Only + 0.95 Wall)

**Problem:** Diagnosis and ReasonForVisit were in the 8 "billable" kinds and could search Pharmacy/Inventory. ASR-mangled findings (e.g. "bus" for "pus", "yeast growth") were grounded to Medications or Lab Kits (Lasix, AST KIT), creating the "Lasix-for-pus" clinical risk.

**Change:** Diagnosis and ReasonForVisit are treated as **Service-Only** kinds with a **0.95 certainty wall**. They remain in dual_sync (so they can link to Services/Procedures) but never to Pharmacy.

| Entity Kind    | Allowed Search Tables   | Match Threshold | Result if &lt; Threshold   |
|----------------|-------------------------|-----------------|----------------------------|
| Medication     | Pharmacy + Services     | 0.80 (with Judge) | Unlinked Suggestion      |
| Procedure      | Services                | 0.85 (with Judge) | Unlinked Suggestion      |
| Diagnostic     | Services                | (with Judge)    | Unlinked Suggestion        |
| **Diagnosis**  | **Services ONLY**       | **0.95**        | **Note-Only (No Billing)** |
| **ReasonForVisit** | **Services ONLY**  | **0.95**        | **Note-Only (No Billing)** |
| Symptom        | None                    | N/A             | Note-Only (No Billing)    |

**Implementation:**

1. **Pharmacy-Free Zone (`kb_ner_routing.py`, `kb_ner_parallel.py`):**
   - **`SERVICE_ONLY_KINDS`** = `["Diagnosis", "ReasonForVisit"]`; **`is_services_only_kind(kind)`**.
   - For these kinds, local search runs **Services only** (no `search_local_inventory_topk`). Batch search and async dual_sync both skip inventory when `is_services_only_kind(canonical_kind)`.
   - **Every** assignment path in `kb_ner_parallel.py` sets **`local_stock_id` = None** for Diagnosis/ReasonForVisit (only `local_service_id` allowed): batch judge result, async auto-bind, async Judge-selected, and **high-certainty auto-link local** (`dual_sync_high_certainty_auto_link_local`). No code path can assign inventory to these kinds.

2. **0.95 Certainty Wall (`kb_ner_routing.py`, `kb_ner_parallel.py`):**
   - **`DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD`** = 0.95 (env override: `DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD`).
   - If kind is Diagnosis or ReasonForVisit and best match score &lt; 0.95 → **no link** (entity preserved as note-only, `match_method` = `note_only_below_threshold` or unlinked). Applied in batch path (before Judge and when merging Judge result) and in async dual_sync (before auto-bind and after Judge selection).

3. **Judge: Category Incompatibility Rule (`kb_ner_disambiguation.py`):**
   - **Rule 0 — CATEGORY INCOMPATIBILITY (Cross-Kind Check):** REJECT any match where ENTITY_KIND is **Symptom, Diagnosis, or ReasonForVisit** but the CANDIDATE is a **tangible product** (Medication, Lab Kit, Consumable, Pharmacy item). "A diagnosis or reason-for-visit like 'pus' or 'yeast' is a state of being, not a product you can put in a bag." Added to batch Judge prompt and per-item EVALUATION, and to legacy single-entity Judge.

**Files:** `kb_ner_routing.py`, `kb_ner_parallel.py`, `kb_ner_disambiguation.py`

### 8.4 Production Polish (Final 5%)

**1. Pharmacy-Free Zone — defense in depth (no "yeast" noise):**  
`kb_ner_local_search.search_local_inventory_topk()` now returns **[]** immediately when `entity_kind` is Diagnosis or ReasonForVisit. So even if any caller ever passed these kinds to inventory search, no pharmacy candidates are returned (no "INJECTION EFECTAL"–style noise for findings like "yeast growth").

**2. Manifest normalized_name preserved for dashboard (no "Cefpodoxime syrup" orphan):**  
When building the grounding output (auto-bind, Judge selected, batch judge), the manifest now keeps **NER `normalized_name`** (e.g. "Cefpodoxime Syrup") instead of overwriting it with the candidate **display_name** (e.g. "CefPET Dry Syrup"). So Phase 2 `_enrich_atoms_with_manifest_ids` can match the Plan atom concept "Cefpodoxime syrup" to the manifest entry via `manifest_by_normalized_name_id` and assign `local_stock_id` (e.g. 42000). The dashboard then shows the item as linked (ID as primary key). `display_name` remains the candidate/inventory name for billing and UI.

---

## 9. Summary Table of Files Touched

| File | Change summary |
|------|----------------|
| `BRAIN_NER_PROMPT_UPDATED.md` (project root) | 13th field `query_expansion`, definition, example line, optional-fields summary. **Form-factor:** NORMALIZED_NAME RULES extended with FORM-FACTOR PRESERVATION (liquid/solid/injectable/topical cues → include form in normalized_name). Prompt source of truth. |
| `kb_ner_super_pass.py` | Prompt 13-field format, query_expansion description; parse and cap query_expansion in entity output (list/string → list[:3]). **Form-factor:** Fallback prompt includes normalized_name FORM-FACTOR PRESERVATION line. |
| `kb_ner_skeleton_parser.py` | Parse 13th field; cap query_expansion at 3; add to entity dict. |
| `kb_ner_parallel.py` | Remove “and not used_query_expansion” from auto-bind/fast-track; query-expansion searches in local path; pass hints + query_expansion to local search and to batch global (9-tuple); preserve query_expansion on entity; pass hints + query_expansion to apply_decision_flow. **Hint robustness:** _hint_item_to_str, _normalize_hints; applied at Judge batch, local/global search, metadata preservation, and final output. |
| `kb_ner_local_search.py` | Add `query_expansion` param to top-k search; apply QUERY_EXPANSION_BOOST_WEIGHT when candidate matches expansion term. **Pharmacy-Free Zone (defense in depth):** `search_local_inventory_topk` returns [] when entity_kind is Diagnosis or ReasonForVisit. |
| `kb_ner_global_search.py` | Parse 9th element as query_expansion; store in entity_query_expansion_terms_by_idx; apply same boost in both scoring branches when candidate matches expansion term. |
| `kb_ner_routing.py` | **SERVICE_ONLY_KINDS** = ["Diagnosis", "ReasonForVisit"]; **DIAGNOSIS_REASONFORVISIT_GROUNDING_THRESHOLD** = 0.95; **is_services_only_kind(kind)**. |
| `kb_ner_parallel.py` | **Pharmacy-Free Zone:** Diagnosis/ReasonForVisit search Services only (batch + async); **0.95 wall:** below threshold → note-only (batch + async); **every** output path (batch judge, async auto-bind, Judge-selected, **high-certainty auto-link local**) sets `local_stock_id` = None for Diagnosis/ReasonForVisit. **Manifest:** Preserve NER `normalized_name` in grounding output (do not overwrite with candidate display_name) so Phase 2/dashboard can match atom "Cefpodoxime syrup" to manifest and show linked by ID. Hints/query_expansion, embedding cache, batch search+judge as before. |
| `kb_ner_disambiguation.py` | Batch Judge: **Rule 0 CATEGORY INCOMPATIBILITY** (finding Symptom/Diagnosis/ReasonForVisit + product candidate → REJECT); EXPANSION BRIDGE, BRAIN_HINTS/QUERY_EXPANSIONS; form-factor rules; **Symptom Suppressor** rule 2.5, per-item step 2.5, legacy paragraph. Judge endpoint: get_client_for_model only. |
| `kb_phase2_integration.py` | _compact_manifest_for_prompt: display_name, kb_preferred_name; _enrich_atoms_with_manifest_ids: manifest_by_normalized_name_id index (ID-bearing by normalized_name+kind), Strategy 1b (match atom by concept+kind → local_stock_id); Strategy 3b distinctive-token match. |
| `SOAP_notes_billing_phase2_kb_atoms.py` | _build_verification_dashboard: linking/status use local_stock_id/local_service_id as primary key (ID-based). |
| `SOAP_notes_phase1_experiment.py` | **SOAP payload:** build_detected_concepts_json_from_manifest() includes `hints` and `query_expansion` per entity. **SOAP prompt:** build_soap_prompt_from_brain_ner() instructs to use BRAIN_NER_JSON as checklist, prefer normalized_name/search_term, and to treat hints and query_expansion as intended clinical meaning (prefer wording that aligns with them, e.g. write "Easotic" not raw "exotic pump"). Form consistency from NER + Judge (see §3.9, §7.4). |

---

**Document version:** 1.4  
**Last updated:** 2026-02-25

# Full Pipeline Benchmark

Stage-wise accuracy scorer for Steps 1–4 of the SOAP pipeline (ASR → cleaning → entities → SOAP facts).

## Quick start

### 1. Run full pipeline batch

```powershell
python batch_pipeline_runner.py --input-dir input_audio_examples --mode full
```

### 2. Create a gold pack per audio slug

```
benchmark_runs/references/full/{audio_slug}/
  transcript.txt          # optional — falls back to references/gemini/{slug}.txt
  cleaned_english.txt     # gold Step 2 cleaned English
  entities.json           # gold entity list
  soap_facts.json           # must-have facts for SOAP recall
  soap_gold.json          # optional full SOAP JSON
```

**entities.json example:**

```json
{
  "entities": [
    {"name": "anti-rabies vaccine", "kind": "PRODUCT"},
    {"name": "bravecto", "kind": "PRODUCT"}
  ]
}
```

**soap_facts.json example:**

```json
{
  "facts": ["oreo", "labrador", "rabies vaccine", "five doses", "antibiotics"]
}
```

Starter pack for `soap_testing`:

```powershell
python scripts/create_soap_testing_gold_pack.py
```

### 3. Run scorer

```powershell
python pipeline_benchmark.py `
  --hypothesis-root benchmark_runs/deepgram_nova-3 `
  --reference-root benchmark_runs/references/full `
  --gemini-fallback-dir benchmark_runs/references/gemini `
  --output benchmark_runs/reports/pipeline_benchmark
```

Output: `pipeline_benchmark.json` + `pipeline_benchmark.csv`

---

## Metrics (plain language)

| Stage | Metric | Meaning |
|-------|--------|---------|
| Step 1 | WER / CER | How close Nova-3 transcript is to gold (words/characters wrong) |
| Step 2 | WER / CER | How close cleaned English is to your gold cleaned text |
| Entities | Precision / Recall / F1 | Did NER find the right medications, procedures, etc.? |
| SOAP | Fact recall % | What fraction of must-have facts appear in the generated SOAP note |

Slugs without a gold pack under `references/full/` are skipped (Step 1 still scores if Gemini ref exists via fallback).

---

## Files

| File | Role |
|------|------|
| [`pipeline_benchmark.py`](../pipeline_benchmark.py) | CLI scorer |
| [`asr_benchmark.py`](../asr_benchmark.py) | Shared WER/CER logic |
| [`benchmark_utils.py`](../benchmark_utils.py) | `find_latest_glob`, `full_reference_dir` |

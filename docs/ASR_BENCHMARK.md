# ASR Benchmark & Batch Pipeline

Batch-run Step-1 ASR (or the full SOAP pipeline) over `input_audio_examples/`, save outputs in a predictable layout, and score transcripts against Gemini reference files.

## Quick start

### 0. Reset benchmark folders (optional — fresh Test_* corpus)

```powershell
python scripts/reset_benchmark_runs.py              # dry-run checklist
python scripts/reset_benchmark_runs.py --execute    # wipe old runs, create test_* folders
```

### 1. Configure ASR (env or CLI — nothing hardcoded)

```powershell
# .env or shell
ASR_PROVIDER=deepgram
ASR_MODEL=nova-3
ASR_LANGUAGE=multi
DEEPGRAM_API_KEY=your_key
```

**Language:** Deepgram defaults to `language=multi` when `ASR_LANGUAGE` is unset (code-switching Hindi/Kannada + English). Override per run:

```powershell
python batch_pipeline_runner.py --asr-language en --mode asr-only
```

Output folders: `benchmark_runs/{ASR_PROVIDER}_{ASR_MODEL}/`.

### 2. Batch run

```powershell
# ASR-only (recommended first — no Postgres)
python batch_pipeline_runner.py --input-dir input_audio_examples --mode asr-only

# Full pipeline (needs Postgres + LLM keys)
python batch_pipeline_runner.py --input-dir input_audio_examples --mode full

# Preview paths without API calls
python batch_pipeline_runner.py --dry-run
```

### 3. Create Gemini reference transcripts

1. Upload **one audio file** to [Gemini](https://gemini.google.com).
2. Paste the prompt from [`benchmark_runs/references/gemini/PROMPT.txt`](../benchmark_runs/references/gemini/PROMPT.txt).
3. Save Gemini's plain-text reply as:

   `benchmark_runs/references/gemini/{audio_slug}.txt`

Example: `Test_1.wav` → `benchmark_runs/references/gemini/test_1.txt`

The batch runner prints slug and reference path after each run.

### 4. Score Step-1 accuracy

```powershell
python asr_benchmark.py `
  --hypothesis-root benchmark_runs/deepgram_nova-3 `
  --reference-dir benchmark_runs/references/gemini `
  --output benchmark_runs/reports/asr_benchmark
```

Reports: `.json` + `.csv` (WER, CER, latency per file).

---

## Output layout (flat — one folder per test)

For `Test_1.wav` with `ASR_PROVIDER=deepgram`, `ASR_MODEL=nova-3`:

```
benchmark_runs/deepgram_nova-3/test_1/
  manifest.json
  step1_raw_transcription.txt
  step1_asr_metadata.json          ← includes "language": "multi"
  cleaned_transcript_*.txt         ← full pipeline
  entity_manifest_*.json
  soap_note_*.json
  archive/                         ← previous run archived on re-run
```

Re-running overwrites the active folder; prior artifacts move to `archive/{timestamp}/`.

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ASR_PROVIDER` | `deepgram` | `deepgram` or `fireworks` |
| `ASR_MODEL` | provider default | Model id (`nova-3`, …) |
| `ASR_LANGUAGE` | `multi` (Deepgram) | Language for ASR (`en`, `hi`, `kn`, `multi`) |
| `DEEPGRAM_API_KEY` | — | Required when `ASR_PROVIDER=deepgram` |

---

## Files

| File | Role |
|------|------|
| [`asr_providers.py`](../asr_providers.py) | Pluggable ASR + `resolve_asr_language()` |
| [`benchmark_utils.py`](../benchmark_utils.py) | Slugs, flat paths, manifests |
| [`batch_pipeline_runner.py`](../batch_pipeline_runner.py) | Batch CLI |
| [`asr_benchmark.py`](../asr_benchmark.py) | Step-1 WER/CER scorer |
| [`pipeline_benchmark.py`](../pipeline_benchmark.py) | Full pipeline stage scorer |
| [`scripts/reset_benchmark_runs.py`](../scripts/reset_benchmark_runs.py) | Reset Test_* layout |

See also: [`PIPELINE_BENCHMARK.md`](PIPELINE_BENCHMARK.md), [`DOCTOR_UI.md`](DOCTOR_UI.md).

---

## Dependencies

```powershell
pip install jiwer requests openpyxl
```

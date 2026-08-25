# Doctor UI

Streamlit app for veterinarians to capture consultations (typed or voice), run the SOAP pipeline, and view the master clinical template.

## Setup

```powershell
pip install -r doctor_ui/requirements-ui.txt
```

Ensure `.env` has API keys (`DEEPGRAM_API_KEY`, Fireworks/OpenAI keys for full pipeline, Postgres for inventory grounding).

## Run

From repo root:

```powershell
python -m streamlit run doctor_ui/app.py
```

Opens in browser (default `http://localhost:8501`).

---

## Features

### Language dropdown

Select consultation language before input:

| Option | Deepgram ASR |
|--------|--------------|
| Auto (multi-language) | `multi` (default) |
| English | `en` |
| Hindi | `hi` |
| Kannada | `kn` |
| Hindi + English mix | `multi` |
| Kannada + English mix | `multi` |

Voice notes use the selected language for ASR. Typed notes store language for records; the pipeline still translates to English when needed.

### Input modes

1. **Type conversation** — paste or type doctor–owner dialogue in the conversation box.
2. **Voice note** — upload audio or use browser mic → Deepgram Nova-3 → fills the same conversation box (editable).

### Output

Master template fields:

- Subjective, Objective, Assessment, Plan, Conclusion
- Key Issues, Abnormal Findings, Customer Instructions, Reminders
- Additional: Differential Diagnosis, Protocols, Vitals

### History

All consultations saved in SQLite: `doctor_ui/data/consultations.db`

Pipeline artifacts: `doctor_ui/runs/{consultation_id}/`

---

## Flow

```
Language select → Type OR Voice → conversation text → Full pipeline → SOAP template on screen
```

---

## Files

| File | Role |
|------|------|
| [`doctor_ui/app.py`](../doctor_ui/app.py) | Streamlit entry |
| [`doctor_ui/db.py`](../doctor_ui/db.py) | SQLite CRUD |
| [`doctor_ui/pipeline_runner.py`](../doctor_ui/pipeline_runner.py) | Async pipeline wrapper |
| [`doctor_ui/languages.py`](../doctor_ui/languages.py) | Language options |

Pipeline entry: `generate_soap_note_from_transcript_async()` in [`SOAP_notes_phase1_experiment.py`](../SOAP_notes_phase1_experiment.py).

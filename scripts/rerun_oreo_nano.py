"""Re-run Oreo consult (#24 text) with current .env models; create DB row for Doctor UI."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

# Normalize alternate key names
if not (os.getenv("OPENAI_API_KEY") or "").strip():
    oa = (os.getenv("OPENAI_KEY") or "").strip()
    if oa:
        os.environ["OPENAI_API_KEY"] = oa

from doctor_ui import db
from doctor_ui.pipeline_runner import run_pipeline_for_consultation

TRANSCRIPT = ROOT / "doctor_ui" / "runs" / "24" / "step1_raw_transcription.txt"


async def main() -> None:
    text = TRANSCRIPT.read_text(encoding="utf-8")
    print("MODELS:")
    for k in (
        "SUPER_PASS_MODEL",
        "CER_MODEL",
        "SOAP_MODEL",
        "PHASE2_MODEL",
        "LLM_JUDGE_MODEL",
        "BATCH_INTENT_MODEL",
    ):
        print(f"  {k}={os.getenv(k)}")
    print(f"Transcript chars: {len(text)}")

    cid = db.create_consultation(
        doctor_name="Nano A/B",
        pet_name="Oreo",
        consultation_language="multi",
        input_mode="typed",
        step1_raw_text=text,
        status="ready",
    )
    print(f"Created consultation_id={cid}")
    print(f"Open Results in UI after run: consultation #{cid}")
    print("=" * 60)

    out = await run_pipeline_for_consultation(
        cid,
        text,
        source="typed",
        consultation_language="multi",
    )
    soap = out.get("soap_json") or {}
    flags = out.get("pipeline_flags") or {}
    timing = out.get("pipeline_timing") or {}
    print("=" * 60)
    print(f"DONE consultation={cid}")
    print(f"output_dir={out.get('output_dir')}")
    print(f"flags overall={flags.get('overall')} counts={flags.get('counts')}")
    print(f"timing total_ms={timing.get('total_ms')}")
    print(f"PrimaryDiagnosis={soap.get('PrimaryDiagnosis', '')[:120]}")
    print(f"KeyIssues={str(soap.get('KeyIssues', ''))[:160]}")
    print(f"\nBrowser: open http://localhost:8501 → Consultation history → #{cid}")


if __name__ == "__main__":
    asyncio.run(main())

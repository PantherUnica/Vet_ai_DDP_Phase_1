#!/usr/bin/env python3
"""
Doctor UI — type or record consultation, run SOAP pipeline, view master template.

Run from repo root:
  streamlit run doctor_ui/app.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env before pipeline imports (API keys, SUPER_PASS_MODEL, etc.)
_env_file = ROOT / ".env"
if _env_file.exists():
    try:
        import os

        from dotenv import load_dotenv

        # override=True so SUPER_PASS_MODEL / keys in .env win over stale process defaults
        load_dotenv(_env_file, override=True)
        # Normalize alternate key names used in this repo's .env
        if not (os.getenv("FIREWORKS_API_KEY") or "").strip():
            fw = (os.getenv("fireworks_API") or os.getenv("FIREWORKS_API") or "").strip()
            if fw:
                os.environ["FIREWORKS_API_KEY"] = fw
        if not (os.getenv("OPENAI_API_KEY") or "").strip():
            oa = (os.getenv("OPENAI_KEY") or "").strip()
            if oa:
                os.environ["OPENAI_API_KEY"] = oa
    except ImportError:
        pass

from doctor_ui import db  # noqa: E402
from doctor_ui.components.soap_display import (  # noqa: E402
    load_pipeline_flags,
    render_soap_template,
)
from doctor_ui.languages import (  # noqa: E402
    DEFAULT_LANGUAGE,
    label_to_stored,
    language_labels,
    stored_to_label,
)
from doctor_ui.pipeline_runner import (  # noqa: E402
    run_pipeline_for_consultation,
    transcribe_audio_file,
)

st.set_page_config(page_title="VetAI Doctor Notes", page_icon="🐾", layout="wide")
db.init_db()

PRIMARY_FIELDS = [
    "Subjective", "Objective", "Assessment", "Plan", "Conclusion",
    "KeyIssues", "AbnormalFindings", "CustomerInstructions", "Reminders",
]


def _ensure_session() -> None:
    if "consultation_id" not in st.session_state:
        st.session_state.consultation_id = None
    if "conversation_text" not in st.session_state:
        # Migrate legacy keys from the old dual text-area UI
        legacy = st.session_state.pop("step1_edit", "") or st.session_state.pop("step1_text", "")
        st.session_state.conversation_text = legacy
    if "language_label" not in st.session_state:
        st.session_state.language_label = language_labels()[0]
    if "last_input_mode" not in st.session_state:
        st.session_state.last_input_mode = "typed"
    if "last_audio_path" not in st.session_state:
        st.session_state.last_audio_path = None


def page_new_consultation() -> None:
    st.header("New Consultation")
    _ensure_session()

    lang_label = st.selectbox(
        "Consultation language",
        language_labels(),
        index=language_labels().index(st.session_state.language_label)
        if st.session_state.language_label in language_labels()
        else 0,
        help="Used for voice transcription (Deepgram). Typed notes are translated to English in the pipeline.",
    )
    st.session_state.language_label = lang_label
    consultation_language = label_to_stored(lang_label)

    col1, col2 = st.columns(2)
    with col1:
        doctor_name = st.text_input("Doctor name (optional)")
    with col2:
        pet_name = st.text_input("Pet name (optional)")

    tab_type, tab_voice = st.tabs(["Type conversation", "Voice note"])

    with tab_type:
        st.caption("Type or paste the doctor–owner dialogue in the conversation box below.")

    with tab_voice:
        uploaded = st.file_uploader(
            "Upload voice note",
            type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
        )
        if uploaded:
            tmp = Path(tempfile.gettempdir()) / f"vetai_upload_{uploaded.name}"
            tmp.write_bytes(uploaded.getvalue())
            if st.button("Transcribe upload", key="transcribe_upload"):
                with st.spinner("Transcribing with Deepgram Nova-3..."):
                    text = asyncio.run(
                        transcribe_audio_file(str(tmp), consultation_language)
                    )
                    st.session_state.conversation_text = text
                    st.session_state.last_input_mode = "voice"
                    st.session_state.last_audio_path = str(tmp)
                    st.rerun()
        st.caption("Or use browser mic (Streamlit audio_input):")
        mic_audio = st.audio_input("Record voice note")
        if mic_audio and st.button("Transcribe recording", key="transcribe_mic"):
            mic_path = Path(tempfile.gettempdir()) / "vetai_mic_recording.wav"
            mic_path.write_bytes(mic_audio.getvalue())
            with st.spinner("Transcribing recording..."):
                text = asyncio.run(
                    transcribe_audio_file(str(mic_path), consultation_language)
                )
                st.session_state.conversation_text = text
                st.session_state.last_input_mode = "voice"
                st.session_state.last_audio_path = str(mic_path)
                st.rerun()

    conversation = st.text_area(
        "Doctor–owner conversation",
        height=280,
        key="conversation_text",
        placeholder="Type what was discussed with the pet owner, or transcribe a voice note above...",
    )
    input_mode = st.session_state.last_input_mode
    audio_path_saved = st.session_state.last_audio_path

    col_save, col_run = st.columns(2)
    with col_save:
        if st.button("Save draft", use_container_width=True):
            if st.session_state.consultation_id:
                db.update_consultation(
                    st.session_state.consultation_id,
                    doctor_name=doctor_name,
                    pet_name=pet_name,
                    consultation_language=consultation_language,
                    step1_raw_text=conversation,
                    input_mode=input_mode,
                    status="draft",
                )
                st.success(f"Draft updated (#{st.session_state.consultation_id})")
            else:
                cid = db.create_consultation(
                    doctor_name=doctor_name,
                    pet_name=pet_name,
                    consultation_language=consultation_language,
                    input_mode=input_mode,
                    step1_raw_text=conversation,
                    audio_path=audio_path_saved,
                    status="draft",
                )
                st.session_state.consultation_id = cid
                st.success(f"Draft saved (#{cid})")

    with col_run:
        if st.button("Generate SOAP note", type="primary", use_container_width=True):
            if not conversation.strip():
                st.error("Enter or transcribe a conversation first.")
            else:
                if not st.session_state.consultation_id:
                    st.session_state.consultation_id = db.create_consultation(
                        doctor_name=doctor_name,
                        pet_name=pet_name,
                        consultation_language=consultation_language,
                        input_mode=input_mode,
                        step1_raw_text=conversation,
                        audio_path=audio_path_saved,
                        status="ready",
                    )
                else:
                    db.update_consultation(
                        st.session_state.consultation_id,
                        doctor_name=doctor_name,
                        pet_name=pet_name,
                        consultation_language=consultation_language,
                        step1_raw_text=conversation,
                        input_mode=input_mode,
                        status="ready",
                    )
                with st.spinner("Running full pipeline (may take a few minutes)..."):
                    try:
                        out = asyncio.run(
                            run_pipeline_for_consultation(
                                st.session_state.consultation_id,
                                conversation,
                                source=input_mode,
                                audio_path=audio_path_saved,
                                consultation_language=consultation_language,
                            )
                        )
                        st.session_state.soap_json = out["soap_json"]
                        st.session_state.pipeline_flags = out.get("pipeline_flags") or {}
                        st.session_state.view = "results"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Pipeline failed: {exc}")


def page_results() -> None:
    st.header("SOAP Results")
    cid = st.session_state.get("consultation_id")
    if not cid:
        st.info("No consultation selected. Create one under New Consultation.")
        return
    rec = db.get_consultation(cid)
    if not rec:
        st.error("Consultation not found.")
        return

    st.caption(
        f"#{cid} | {rec.get('input_mode')} | "
        f"{stored_to_label(rec.get('consultation_language') or DEFAULT_LANGUAGE)} | "
        f"status: {rec.get('status')}"
    )

    if rec.get("step1_raw_text"):
        with st.expander("Raw conversation"):
            st.write(rec["step1_raw_text"])

    soap_json = rec.get("soap_json_parsed") or st.session_state.get("soap_json")
    if soap_json:
        flags = (
            st.session_state.get("pipeline_flags")
            or load_pipeline_flags(rec.get("output_dir"))
        )
        render_soap_template(soap_json, flags_report=flags)
    elif rec.get("status") == "error":
        st.error(rec.get("error_message") or "Pipeline error")
    else:
        st.info("SOAP not generated yet.")


def page_history() -> None:
    st.header("Consultation history")
    rows = db.list_consultations()
    if not rows:
        st.info("No consultations yet.")
        return
    for r in rows:
        label = stored_to_label(r.get("consultation_language") or DEFAULT_LANGUAGE)
        with st.expander(
            f"#{r['id']} — {r.get('pet_name') or 'Pet'} — {r.get('status')} — {label}"
        ):
            st.write(f"Doctor: {r.get('doctor_name') or '—'}")
            st.write(f"Mode: {r.get('input_mode')} | Created: {r.get('created_at')}")
            if r.get("step1_raw_text"):
                st.text(r["step1_raw_text"][:500] + ("..." if len(r["step1_raw_text"]) > 500 else ""))
            if st.button("Open", key=f"open_{r['id']}"):
                st.session_state.consultation_id = r["id"]
                rec = db.get_consultation(r["id"])
                if rec and rec.get("soap_json_parsed"):
                    st.session_state.soap_json = rec["soap_json_parsed"]
                st.session_state.view = "results"
                st.rerun()


def main() -> None:
    _ensure_session()
    if "view" not in st.session_state:
        st.session_state.view = "new"

    st.sidebar.title("VetAI Doctor UI")
    nav = st.sidebar.radio(
        "Navigate",
        ["New Consultation", "Results", "History"],
        index={"new": 0, "results": 1, "history": 2}.get(st.session_state.view, 0),
    )
    if nav == "New Consultation":
        st.session_state.view = "new"
        page_new_consultation()
    elif nav == "Results":
        st.session_state.view = "results"
        page_results()
    else:
        st.session_state.view = "history"
        page_history()


if __name__ == "__main__":
    main()

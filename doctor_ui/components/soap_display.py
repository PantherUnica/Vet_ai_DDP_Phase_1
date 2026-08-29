"""SOAP master template display components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

PRIMARY_FIELDS = [
    ("Subjective", "Subjective"),
    ("Objective", "Objective"),
    ("Assessment", "Assessment"),
    ("Plan", "Plan"),
    ("Conclusion", "Conclusion"),
    ("Key Issues", "KeyIssues"),
    ("Abnormal Findings", "AbnormalFindings"),
    ("Customer Instructions", "CustomerInstructions"),
    ("Reminders / Follow-up", "Reminders"),
]

ADDITIONAL_FIELDS = [
    ("Differential Diagnosis", "DifferentialDiagnosis"),
    ("Protocols", "Protocols"),
    ("Vitals", "Vitals"),
]


def _render_multiline(value: str) -> None:
    """Show numbered / multi-line sections with preserved newlines."""
    text = (value or "").strip()
    if not text:
        return
    if "\n" in text:
        st.markdown(text.replace("\n", "  \n"))
    else:
        st.write(text)


def render_pipeline_flags(flags_report: Optional[Dict[str, Any]]) -> None:
    if not flags_report:
        return
    overall = (flags_report.get("overall") or "unknown").upper()
    counts = flags_report.get("counts") or {}
    stages = flags_report.get("stage_summary") or {}

    st.subheader("Workflow health")
    color = {"HEALTHY": "🟢", "DEGRADED": "🟡", "FAILING": "🔴"}.get(overall, "⚪")
    st.markdown(
        f"{color} **Overall: {overall}** — "
        f"{counts.get('error', 0)} error(s), {counts.get('warning', 0)} warning(s)"
    )
    if stages:
        st.caption(
            " · ".join(f"{k}: {v}" for k, v in stages.items())
        )

    flags: List[Dict[str, Any]] = flags_report.get("flags") or []
    if not flags:
        return

    for f in flags:
        sev = (f.get("severity") or "info").lower()
        step = f.get("step") or "?"
        msg = f.get("message") or ""
        detail = f.get("detail") or ""
        line = f"**[{step}]** {msg}"
        if detail:
            line += f"  \n_{detail}_"
        if sev == "error":
            st.error(line)
        elif sev == "warning":
            st.warning(line)
        else:
            st.info(line)


def load_pipeline_flags(output_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    if not output_dir:
        return None
    p = Path(output_dir)
    latest = p / "pipeline_flags_latest.json"
    if latest.exists():
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            return None
    files = sorted(p.glob("pipeline_flags_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def render_soap_template(
    soap_json: Dict[str, Any],
    *,
    flags_report: Optional[Dict[str, Any]] = None,
) -> None:
    if not soap_json:
        st.warning("No SOAP note generated yet.")
        return

    if flags_report:
        render_pipeline_flags(flags_report)
        st.divider()

    st.subheader("Clinical Note")
    for label, key in PRIMARY_FIELDS:
        value = soap_json.get(key) or soap_json.get(key.lower()) or ""
        if value:
            st.markdown(f"**{label}**")
            _render_multiline(str(value))

    with st.expander("Additional fields"):
        for label, key in ADDITIONAL_FIELDS:
            value = soap_json.get(key) or ""
            if value:
                st.markdown(f"**{label}**")
                _render_multiline(str(value))

    st.download_button(
        "Download SOAP JSON",
        data=json.dumps(soap_json, indent=2, ensure_ascii=False),
        file_name="soap_note.json",
        mime="application/json",
    )

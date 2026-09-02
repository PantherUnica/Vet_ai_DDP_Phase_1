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

def _resolve_diagnosis_fields(soap_json: Dict[str, Any]) -> Dict[str, str]:
    """Return primary/secondary with legacy DifferentialDiagnosis fallback."""
    try:
        from doctor_ui.diagnosis_schema import migrate_legacy_diagnosis
        data = migrate_legacy_diagnosis(soap_json)
    except ImportError:
        data = dict(soap_json or {})
        if not (data.get("PrimaryDiagnosis") or "").strip():
            legacy = (data.get("DifferentialDiagnosis") or "").strip()
            if legacy:
                data["PrimaryDiagnosis"] = legacy
    return {
        "PrimaryDiagnosis": (data.get("PrimaryDiagnosis") or "").strip(),
        "SecondaryDiagnosis": (data.get("SecondaryDiagnosis") or "").strip(),
    }


ADDITIONAL_FIELDS = [
    ("Protocols", "Protocols"),
    ("Vitals", "Vitals"),
]


def render_diagnosis_fields(soap_json: Dict[str, Any]) -> None:
    """Show differential diagnosis with primary and secondary numbered lists."""
    diag = _resolve_diagnosis_fields(soap_json)
    st.markdown("**Differential Diagnosis**")
    st.markdown("**Primary Diagnosis**")
    if diag["PrimaryDiagnosis"]:
        _render_multiline(diag["PrimaryDiagnosis"])
    else:
        st.caption("Not recorded")
    st.markdown("**Secondary Diagnosis**")
    if diag["SecondaryDiagnosis"]:
        _render_multiline(diag["SecondaryDiagnosis"])
    else:
        st.caption("None recorded")


def render_vitals_table(vitals_raw: Any) -> None:
    """Render structured vitals as a table; legacy string fallback."""
    try:
        from doctor_ui.vitals_schema import vitals_rows_for_display
    except ImportError:
        vitals_rows_for_display = None

    st.markdown("**Vitals**")
    if vitals_rows_for_display:
        rows = vitals_rows_for_display(vitals_raw)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            return
    if isinstance(vitals_raw, str) and vitals_raw.strip():
        _render_multiline(vitals_raw)
        return
    if isinstance(vitals_raw, dict) and vitals_raw:
        _render_multiline(json.dumps(vitals_raw, indent=2, ensure_ascii=False))
        return
    st.caption("No vitals recorded in this consultation")


def _render_multiline(value: str) -> None:
    """Show numbered / multi-line sections with preserved newlines."""
    text = (value or "").strip()
    if not text:
        return
    if "\n" in text:
        st.markdown(text.replace("\n", "  \n"))
    else:
        st.write(text)


def render_pipeline_timing(timing_report: Optional[Dict[str, Any]]) -> None:
    if not timing_report:
        return
    from pipeline_timing import format_duration_ms, timing_rows_for_display

    st.subheader("Pipeline timing")
    perceived = timing_report.get("user_perceived_total_ms")
    pipeline_ms = timing_report.get("total_ms")
    if perceived is not None and perceived != pipeline_ms:
        st.markdown(
            f"**Total (ASR + pipeline):** {format_duration_ms(perceived)}  \n"
            f"**Pipeline only:** {format_duration_ms(pipeline_ms)}"
        )
    elif pipeline_ms is not None:
        st.markdown(f"**Total:** {format_duration_ms(pipeline_ms)}")
    note = timing_report.get("parallel_note")
    if note:
        st.caption(note)

    asr_profile = (timing_report.get("metadata") or {}).get("asr_profile")
    if asr_profile:
        dur = asr_profile.get("duration_sec")
        mb = asr_profile.get("file_mb")
        fmt = asr_profile.get("format") or "?"
        if dur is not None:
            st.caption(f"Audio: {dur:.1f}s, {mb or '?'} MB, {fmt}")

    rows = timing_rows_for_display(timing_report)
    if not rows:
        return

    lines = []
    for row in rows:
        indent = "&nbsp;" * (4 * int(row.get("indent") or 0))
        label = row.get("label") or "?"
        display = row.get("display") or "n/a"
        if label.startswith("Total"):
            continue
        lines.append(f"{indent}**{label}:** {display}")

    if lines:
        st.markdown("  \n".join(lines), unsafe_allow_html=True)


def load_pipeline_timing(output_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    if not output_dir:
        return None
    try:
        from pipeline_timing import load_pipeline_timing as _load
        return _load(output_dir)
    except Exception:
        return None


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
    timing_report: Optional[Dict[str, Any]] = None,
) -> None:
    if not soap_json:
        st.warning("No SOAP note generated yet.")
        return

    if timing_report:
        render_pipeline_timing(timing_report)
        st.divider()

    if flags_report:
        render_pipeline_flags(flags_report)
        st.divider()

    st.subheader("Clinical Note")
    from doctor_ui.reminder_utils import filter_actionable_reminders_text

    for label, key in PRIMARY_FIELDS:
        value = soap_json.get(key) or soap_json.get(key.lower()) or ""
        if not value:
            continue
        if key == "Reminders":
            filtered, hidden = filter_actionable_reminders_text(str(value))
            if not filtered:
                continue
            st.markdown(f"**{label}**")
            _render_multiline(filtered)
            if hidden:
                st.caption(f"{hidden} informational reminder(s) hidden (passive monitoring / conditional only).")
            continue
        st.markdown(f"**{label}**")
        _render_multiline(str(value))

    with st.expander("Additional fields"):
        render_diagnosis_fields(soap_json)
        for label, key in ADDITIONAL_FIELDS:
            if key == "Vitals":
                render_vitals_table(soap_json.get(key))
                continue
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

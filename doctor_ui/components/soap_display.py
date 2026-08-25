"""SOAP master template display components."""

from __future__ import annotations

import json
from typing import Any, Dict

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
    ("Reminders", "Reminders"),
]

ADDITIONAL_FIELDS = [
    ("Differential Diagnosis", "DifferentialDiagnosis"),
    ("Protocols", "Protocols"),
    ("Vitals", "Vitals"),
]


def render_soap_template(soap_json: Dict[str, Any]) -> None:
    if not soap_json:
        st.warning("No SOAP note generated yet.")
        return

    st.subheader("Clinical Note")
    for label, key in PRIMARY_FIELDS:
        value = soap_json.get(key) or soap_json.get(key.lower()) or ""
        if value:
            st.markdown(f"**{label}**")
            st.write(value)

    with st.expander("Additional fields"):
        for label, key in ADDITIONAL_FIELDS:
            value = soap_json.get(key) or ""
            if value:
                st.markdown(f"**{label}**")
                st.write(value)

    st.download_button(
        "Download SOAP JSON",
        data=json.dumps(soap_json, indent=2, ensure_ascii=False),
        file_name="soap_note.json",
        mime="application/json",
    )

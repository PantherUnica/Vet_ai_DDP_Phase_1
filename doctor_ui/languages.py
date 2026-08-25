"""Supported consultation languages for doctor UI and Deepgram ASR."""

from __future__ import annotations

from typing import List, Tuple

# (UI label, stored value, deepgram language param)
LANGUAGE_OPTIONS: List[Tuple[str, str, str]] = [
    ("Auto (multi-language)", "multi", "multi"),
    ("English", "en", "en"),
    ("Hindi", "hi", "hi"),
    ("Kannada", "kn", "kn"),
    ("Hindi + English mix", "multi_hi_en", "multi"),
    ("Kannada + English mix", "multi_kn_en", "multi"),
]

DEFAULT_LANGUAGE = "multi"


def language_labels() -> List[str]:
    return [opt[0] for opt in LANGUAGE_OPTIONS]


def label_to_stored(label: str) -> str:
    for ui_label, stored, _ in LANGUAGE_OPTIONS:
        if ui_label == label:
            return stored
    return DEFAULT_LANGUAGE


def stored_to_deepgram(stored: str) -> str:
    for _, s, dg in LANGUAGE_OPTIONS:
        if s == stored:
            return dg
    if stored in ("en", "hi", "kn", "multi"):
        return stored
    return DEFAULT_LANGUAGE


def stored_to_label(stored: str) -> str:
    for ui_label, s, _ in LANGUAGE_OPTIONS:
        if s == stored:
            return ui_label
    return LANGUAGE_OPTIONS[0][0]

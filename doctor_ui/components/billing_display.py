"""Phase 2 verification dashboard — billing form display for Doctor UI."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

_PRESCRIBED_INTENTS = frozenset({"Prescribed", "prescribed"})
_ADMINISTERED_INTENTS = frozenset({"Administered", "administered"})
_PERFORMED_INTENTS = frozenset({"Performed", "performed"})
_EMPTY_FREQ = frozenset({"", "custom", "as directed"})


@dataclass
class BillingContext:
    atom_by_name: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    conversation_text: str = ""


def _norm_name(value: str) -> str:
    return " ".join((value or "").lower().split())


def load_verification_dashboard(output_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load latest verification_dashboard JSON from a consultation run folder."""
    if not output_dir:
        return None
    p = Path(output_dir)
    if not p.is_dir():
        return None

    latest = p / "verification_dashboard_latest.json"
    if latest.is_file():
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    files = sorted(
        p.glob("verification_dashboard_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for f in files:
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

    ka_files = sorted(
        p.glob("knowledge_atoms_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for f in ka_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            dash = data.get("verification_dashboard")
            if isinstance(dash, dict):
                return dash
        except (json.JSONDecodeError, OSError):
            continue
    return None


def load_knowledge_atoms(output_dir: Optional[str]) -> List[Dict[str, Any]]:
    """Load knowledge_atoms[] from the latest run artifact."""
    if not output_dir:
        return []
    p = Path(output_dir)
    if not p.is_dir():
        return []

    ka_files = sorted(
        p.glob("knowledge_atoms_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for f in ka_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            atoms = data.get("knowledge_atoms")
            if isinstance(atoms, list):
                return atoms
        except (json.JSONDecodeError, OSError):
            continue
    return []


def _build_billing_context(
    atoms: List[Dict[str, Any]],
    conversation_text: str = "",
) -> BillingContext:
    lookup: Dict[str, Dict[str, Any]] = {}
    for atom in atoms:
        concept = _norm_name(atom.get("concept") or "")
        if concept:
            lookup[concept] = atom
    return BillingContext(atom_by_name=lookup, conversation_text=conversation_text or "")


def phase2_was_skipped(flags_report: Optional[Dict[str, Any]]) -> bool:
    if not flags_report:
        return False
    for f in flags_report.get("flags") or []:
        step = (f.get("step") or "").upper()
        msg = (f.get("message") or "").lower()
        if step == "PHASE2" and ("skip" in msg or "unreachable" in msg):
            return True
    stages = flags_report.get("stage_summary") or {}
    if stages.get("PHASE2") == "skipped":
        return True
    return False


def _stock_id(item: Dict[str, Any]) -> Optional[int]:
    raw = item.get("inventory_id") or item.get("stock_id") or item.get("linked_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _service_id(item: Dict[str, Any]) -> Optional[int]:
    raw = item.get("service_id") or item.get("linked_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _find_atom(item: Dict[str, Any], ctx: BillingContext) -> Optional[Dict[str, Any]]:
    names = [
        item.get("item_name"),
        item.get("name"),
        item.get("original_mention"),
    ]
    for raw in names:
        key = _norm_name(str(raw or ""))
        if key and key in ctx.atom_by_name:
            return ctx.atom_by_name[key]
    for raw in names:
        key = _norm_name(str(raw or ""))
        if not key:
            continue
        for atom_key, atom in ctx.atom_by_name.items():
            if key in atom_key or atom_key in key:
                return atom
    return None


def _score_of(suggestion: Dict[str, Any]) -> float:
    try:
        return float(suggestion.get("match_score") or 0)
    except (TypeError, ValueError):
        return 0.0


def _best_suggestion(
    suggestions: List[Dict[str, Any]],
    atom: Optional[Dict[str, Any]],
    id_key: str,
) -> Optional[Dict[str, Any]]:
    attrs = (atom or {}).get("attributes") or {}
    primary = attrs.get("primary_local_suggestion") or {}
    if primary.get(id_key) or primary.get("stock_id") or primary.get("service_id"):
        pid = primary.get(id_key) or primary.get("stock_id") or primary.get("service_id")
        for s in suggestions:
            sid = s.get(id_key) or s.get("stock_id") or s.get("inventory_id") or s.get("service_id")
            if sid is not None and int(sid) == int(pid):
                return s
        return {
            "name": primary.get("name") or "",
            id_key: pid,
            "stock_id": primary.get("stock_id"),
            "service_id": primary.get("service_id"),
            "match_score": 1.0,
            "recommendation": "AUTO",
        }
    if not suggestions:
        return None
    return max(suggestions, key=_score_of)


def _infer_ongoing(source_text: str) -> str:
    low = (source_text or "").lower()
    if "ongoing" in low or "continues" in low or "current" in low:
        return "Ongoing"
    return ""


def _infer_route(name: str, dose: str) -> str:
    """ponytail: name-keyword heuristic only; upgrade path = atom extraction."""
    low = (name or "").lower()
    if "inj" in low or "injection" in low:
        return "IM"
    if "tab" in low or "tablet" in low or "capsule" in low:
        return "PO"
    if "syrup" in low or "suspension" in low:
        return "PO"
    return ""


def _merge_field(item: Dict[str, Any], field: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    current = str(item.get(field) or "").strip()
    if not current or current.lower() in _EMPTY_FREQ:
        item[field] = text


def auto_fill_item(item: Dict[str, Any], ctx: BillingContext) -> Tuple[Dict[str, Any], bool]:
    """
    Enrich a dashboard row from knowledge atoms + conversation.
    Returns (filled_item, auto_applied_best_match).
    """
    filled = deepcopy(item)
    atom = _find_atom(filled, ctx)
    auto_applied = False
    attrs = (atom or {}).get("attributes") or {}
    source_text = (atom or {}).get("source_text") or ""

    if atom:
        _merge_field(filled, "dosage", attrs.get("dose"))
        _merge_field(filled, "dose", attrs.get("dose"))
        _merge_field(filled, "route", attrs.get("route"))
        _merge_field(filled, "frequency", attrs.get("frequency"))
        _merge_field(filled, "duration", attrs.get("duration"))
        _merge_field(filled, "quantity", attrs.get("quantity"))
        _merge_field(filled, "instructions", attrs.get("instructions"))
        if not (filled.get("remarks") or "").strip():
            filled["remarks"] = source_text
        if not (filled.get("instructions") or "").strip():
            filled["instructions"] = source_text
        if not (filled.get("frequency") or "").strip() or str(filled.get("frequency")).lower() in _EMPTY_FREQ:
            ongoing = _infer_ongoing(source_text)
            if ongoing:
                filled["frequency"] = ongoing

        atom_suggestions = attrs.get("suggestions") or []
        if atom_suggestions and not filled.get("suggestions"):
            filled["suggestions"] = atom_suggestions

        if atom.get("local_stock_id") and not _stock_id(filled):
            filled["inventory_id"] = atom["local_stock_id"]
            filled["stock_id"] = atom["local_stock_id"]
            auto_applied = True
        if atom.get("local_service_id") and not _service_id(filled):
            filled["service_id"] = atom["local_service_id"]
            auto_applied = True

    if not _stock_id(filled) and filled.get("suggestions"):
        pick = _best_suggestion(filled["suggestions"], atom, "stock_id")
        if pick:
            sid = pick.get("stock_id") or pick.get("inventory_id")
            if sid is not None:
                filled["inventory_id"] = int(sid)
                filled["stock_id"] = int(sid)
                if not (filled.get("item_name") or filled.get("name")):
                    filled["item_name"] = pick.get("name") or filled.get("item_name")
                filled["_auto_pick_name"] = pick.get("name")
                filled["_auto_pick_score"] = _score_of(pick)
                auto_applied = True

    if not _service_id(filled) and filled.get("suggestions"):
        pick = _best_suggestion(filled["suggestions"], atom, "service_id")
        if pick and pick.get("service_id") is not None:
            filled["service_id"] = int(pick["service_id"])
            filled["_auto_pick_name"] = pick.get("name")
            filled["_auto_pick_score"] = _score_of(pick)
            auto_applied = True

    name = filled.get("item_name") or filled.get("name") or ""
    if not (filled.get("route") or "").strip():
        inferred = _infer_route(name, filled.get("dosage") or filled.get("dose") or "")
        if inferred:
            filled["route"] = inferred

    if source_text:
        filled["_source_text"] = source_text
    elif ctx.conversation_text and name:
        # ponytail: O(n) scan over conversation; fine for single consult UI
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        for line in ctx.conversation_text.splitlines():
            if pattern.search(line):
                filled.setdefault("_source_text", line.strip())
                break

    if auto_applied and (filled.get("status") or "").upper().replace(" ", "_") == "ACTION_REQUIRED":
        filled["status"] = "AUTO-FILLED"

    return filled, auto_applied


@lru_cache(maxsize=256)
def _enrich_inventory(stock_id: int) -> Optional[Dict[str, Any]]:
    try:
        from SOAP_notes_billing_phase2_kb_atoms import get_inventory_master_record
        return get_inventory_master_record(str(stock_id))
    except Exception:
        return None


@lru_cache(maxsize=256)
def _enrich_service(service_id: int) -> Optional[Dict[str, Any]]:
    try:
        from SOAP_notes_billing_phase2_kb_atoms import get_service_master_record
        return get_service_master_record(str(service_id))
    except Exception:
        return None


def _uom_for_item(stock_id: Optional[int], master: Optional[Dict[str, Any]]) -> str:
    if not master:
        return ""
    admin = (master.get("administered_uom") or "").strip()
    sales = (master.get("sales_uom") or "").strip()
    unit = (master.get("unit") or "").strip()
    return admin or sales or unit


def _missing_fields(item: Dict[str, Any], required: Tuple[str, ...]) -> List[str]:
    missing = []
    for fld in required:
        val = item.get(fld)
        if val is None or str(val).strip() == "" or str(val).strip().lower() in _EMPTY_FREQ:
            missing.append(fld)
    return missing


def _status_caption(item: Dict[str, Any], missing: List[str], auto_applied: bool) -> None:
    status = (item.get("status") or "DRAFT").upper().replace(" ", "_")
    if status == "AUTO-FILLED" or auto_applied:
        if missing:
            st.success(
                f"Auto-filled from consultation — still needs: {', '.join(missing)}"
            )
        else:
            st.success("Auto-filled from consultation — please verify before billing.")
        return
    if status == "ACTION_REQUIRED":
        if missing:
            st.warning(
                f"Could not fully auto-fill — missing: {', '.join(missing)}. "
                "Check conversation snippet below or pick a variant."
            )
        else:
            st.warning("Pick a variant or verify the auto-suggested values.")
    elif missing:
        st.info(f"Needs review: {', '.join(missing)}")
    else:
        st.caption(f"Status: {status}")


def _show_conversation_snippet(item: Dict[str, Any]) -> None:
    snippet = item.get("_source_text") or item.get("remarks") or ""
    if snippet:
        with st.expander("From conversation", expanded=False):
            st.write(snippet)


def _suggestion_labels(suggestions: List[Dict[str, Any]], id_key: str) -> List[str]:
    labels = []
    for s in suggestions or []:
        sid = s.get(id_key) or s.get("stock_id") or s.get("inventory_id") or s.get("service_id")
        name = s.get("name") or "?"
        rec = s.get("recommendation") or ""
        suffix = f" ({rec})" if rec else ""
        labels.append(f"{name} — {id_key}: {sid}{suffix}")
    return labels


def _split_unlinked(dashboard: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out = {
        "procedures": [],
        "inventory": [],
        "administered": [],
        "prescribed": [],
    }
    for item in dashboard.get("module_4_unlinked_entities") or []:
        intent = (item.get("intent_context") or item.get("administered_prescribed") or "").strip()
        kind = (item.get("kind") or "").strip().lower()
        if intent in _PERFORMED_INTENTS or kind == "procedure":
            out["procedures"].append(item)
        elif intent in _ADMINISTERED_INTENTS:
            out["administered"].append(item)
        elif intent in _PRESCRIBED_INTENTS:
            if kind in ("diet", "retail", "product") or "diet" in (item.get("item_name") or "").lower():
                out["inventory"].append(item)
            else:
                out["prescribed"].append(item)
        else:
            out["prescribed"].append(item)
    return out


def _render_id_banner(*, stock_id: Optional[int] = None, service_id: Optional[int] = None) -> None:
    parts = []
    if stock_id is not None:
        parts.append(f"**stock_id:** `{stock_id}`")
    if service_id is not None:
        parts.append(f"**service_id:** `{service_id}`")
    if parts:
        st.markdown(" · ".join(parts))


def _render_auto_pick_note(item: Dict[str, Any], id_key: str) -> None:
    if item.get("_auto_pick_name"):
        score = item.get("_auto_pick_score")
        score_txt = f" (match {score:.0%})" if isinstance(score, (int, float)) and score else ""
        st.caption(f"Best clinic match{score_txt}: {item['_auto_pick_name']}")


def _render_procedure_row(item: Dict[str, Any], key_prefix: str, ctx: BillingContext) -> None:
    item, auto_applied = auto_fill_item(item, ctx)
    sid = _service_id(item)
    suggestions = item.get("suggestions") or []
    name = item.get("item_name") or item.get("name") or ""

    master = _enrich_service(sid) if sid else None
    display_name = (master or {}).get("procedure_name") or item.get("_auto_pick_name") or name

    st.markdown(f"**{display_name or 'Procedure / Service'}**")
    _render_id_banner(service_id=sid)
    _render_auto_pick_note(item, "service_id")
    _status_caption(item, [], auto_applied)
    _show_conversation_snippet(item)

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Procedure / Service", value=display_name, key=f"{key_prefix}_name")
    with c2:
        variant_val = (master or {}).get("category") or item.get("_auto_pick_name") or ""
        if sid and suggestions:
            labels = _suggestion_labels(suggestions, "service_id")
            default = 0
            if item.get("_auto_pick_name"):
                for i, s in enumerate(suggestions):
                    if s.get("name") == item["_auto_pick_name"]:
                        default = i
                        break
            st.selectbox("Variant", options=labels, index=default, key=f"{key_prefix}_variant")
        else:
            st.text_input("Variant", value=variant_val, key=f"{key_prefix}_variant_txt")
    st.number_input("Billable qty", min_value=1, value=1, key=f"{key_prefix}_qty")


def _render_inventory_row(item: Dict[str, Any], key_prefix: str, ctx: BillingContext) -> None:
    item, auto_applied = auto_fill_item(item, ctx)
    stk = _stock_id(item)
    suggestions = item.get("suggestions") or []
    name = item.get("item_name") or item.get("name") or ""

    master = _enrich_inventory(stk) if stk else None
    display_name = (master or {}).get("item_name") or (master or {}).get("trade_name") or item.get("_auto_pick_name") or name
    batch = (master or {}).get("batch_number") or ""
    uom = _uom_for_item(stk, master)
    qty_default = 1
    try:
        qty_default = max(1, int(item.get("quantity") or 1))
    except (TypeError, ValueError):
        qty_default = 1

    st.markdown(f"**{display_name or 'Inventory item'}**")
    _render_id_banner(stock_id=stk)
    _render_auto_pick_note(item, "stock_id")
    _status_caption(item, _missing_fields(item, ("quantity",)), auto_applied)
    _show_conversation_snippet(item)

    if stk and suggestions:
        labels = _suggestion_labels(suggestions, "stock_id")
        default = 0
        if item.get("_auto_pick_name"):
            for i, s in enumerate(suggestions):
                if s.get("name") == item["_auto_pick_name"]:
                    default = i
                    break
        st.selectbox("Item", options=labels, index=default, key=f"{key_prefix}_pick")
    else:
        st.text_input("Item", value=display_name, key=f"{key_prefix}_item")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Batch / Lot no.", value=batch, key=f"{key_prefix}_batch")
    with c2:
        st.number_input("Billable qty", min_value=1, value=qty_default, key=f"{key_prefix}_qty")
    with c3:
        st.text_input("UOM", value=uom, key=f"{key_prefix}_uom")
    st.selectbox(
        "Link to procedure",
        options=["Not linked to a procedure"],
        key=f"{key_prefix}_proc_link",
    )


def _render_administered_row(item: Dict[str, Any], key_prefix: str, ctx: BillingContext) -> None:
    item, auto_applied = auto_fill_item(item, ctx)
    stk = _stock_id(item)
    suggestions = item.get("suggestions") or []
    name = item.get("item_name") or item.get("name") or ""

    master = _enrich_inventory(stk) if stk else None
    display_name = (master or {}).get("item_name") or (master or {}).get("trade_name") or item.get("_auto_pick_name") or name
    batch = (master or {}).get("batch_number") or ""
    uom = _uom_for_item(stk, master)
    dose = item.get("dosage") or item.get("dose") or ""
    route = item.get("route") or ""
    missing = _missing_fields(item, ("dosage", "route"))

    st.markdown(f"**{display_name or 'Administered item'}**")
    _render_id_banner(stock_id=stk)
    _render_auto_pick_note(item, "stock_id")
    _status_caption(item, missing, auto_applied)
    _show_conversation_snippet(item)

    if not stk and suggestions:
        labels = _suggestion_labels(suggestions, "stock_id")
        st.selectbox("Item (pick from suggestions)", options=labels, index=0, key=f"{key_prefix}_pick")
    else:
        st.text_input("Item", value=display_name, key=f"{key_prefix}_item")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Batch / Lot no.", value=batch, key=f"{key_prefix}_batch")
    with c2:
        st.number_input("Billable qty", min_value=1, value=1, key=f"{key_prefix}_qty")
    with c3:
        st.text_input("UOM", value=uom, key=f"{key_prefix}_uom")
    c4, c5 = st.columns(2)
    with c4:
        st.text_input("Dose", value=dose, key=f"{key_prefix}_dose")
    with c5:
        st.text_input("Route", value=route, key=f"{key_prefix}_route")
    st.selectbox(
        "Link to procedure",
        options=["Not linked to a procedure"],
        key=f"{key_prefix}_proc_link",
    )


def _render_prescription_row(item: Dict[str, Any], key_prefix: str, ctx: BillingContext) -> None:
    item, auto_applied = auto_fill_item(item, ctx)
    stk = _stock_id(item)
    suggestions = item.get("suggestions") or []
    name = item.get("item_name") or item.get("name") or ""

    master = _enrich_inventory(stk) if stk else None
    display_name = (master or {}).get("item_name") or (master or {}).get("trade_name") or item.get("_auto_pick_name") or name
    dose = item.get("dosage") or item.get("dose") or ""
    frequency = item.get("frequency") or ""
    route = item.get("route") or ""
    duration = item.get("duration") or ""
    instructions = item.get("instructions") or ""
    missing = _missing_fields(item, ("frequency", "duration"))

    st.markdown(f"**{display_name or 'Medication'}**")
    _render_id_banner(stock_id=stk)
    _render_auto_pick_note(item, "stock_id")
    _status_caption(item, missing, auto_applied)
    _show_conversation_snippet(item)

    if stk and suggestions:
        labels = _suggestion_labels(suggestions, "stock_id")
        default = 0
        if item.get("_auto_pick_name"):
            for i, s in enumerate(suggestions):
                if s.get("name") == item["_auto_pick_name"]:
                    default = i
                    break
        st.selectbox("Drug / Medication", options=labels, index=default, key=f"{key_prefix}_pick")
    else:
        st.text_input("Drug / Medication", value=display_name, key=f"{key_prefix}_drug")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Strength / Dose", value=dose, key=f"{key_prefix}_dose")
    with c2:
        st.text_input("Frequency", value=frequency, key=f"{key_prefix}_freq")
    c3, c4 = st.columns(2)
    with c3:
        st.text_input("Route", value=route, key=f"{key_prefix}_route")
    with c4:
        st.text_input("Duration (days)", value=duration, key=f"{key_prefix}_dur")
    st.text_input("Instructions", value=instructions, key=f"{key_prefix}_instr")


def render_billing_forms(
    dashboard: Optional[Dict[str, Any]],
    *,
    flags_report: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    conversation_text: str = "",
) -> None:
    """Render Phase 2 billing sections on the Results page."""
    st.subheader("Billing & Pharmacy (Phase 2)")

    if dashboard is None:
        if phase2_was_skipped(flags_report):
            st.info(
                "Phase 2 did not run — Postgres was unreachable or Phase 2 was skipped. "
                "Start Postgres and run a new consultation to see billing forms."
            )
        else:
            st.info(
                "No Phase 2 billing data for this consultation yet. "
                "Run the full pipeline with Postgres running."
            )
        return

    atoms = load_knowledge_atoms(output_dir)
    ctx = _build_billing_context(atoms, conversation_text)
    if atoms:
        st.caption(
            f"Auto-filling from {len(atoms)} knowledge atom(s) extracted from the consultation."
        )

    unlinked = _split_unlinked(dashboard)
    procedures = list(dashboard.get("procedures_services") or []) + unlinked["procedures"]
    inventory = list(dashboard.get("other_products_retail") or []) + unlinked["inventory"]
    administered = list(dashboard.get("medications_administered") or []) + unlinked["administered"]
    prescribed = list(dashboard.get("medications_prescribed") or []) + unlinked["prescribed"]

    stream = dashboard.get("streamlined_dashboard") or {}
    for row in stream.get("pharmacy_prescribed") or []:
        if (row.get("status") or "").upper().replace(" ", "_") == "ACTION_REQUIRED":
            if not any(
                (r.get("item_name") or r.get("name")) == row.get("name")
                for r in prescribed
            ):
                prescribed.append({
                    "item_name": row.get("name"),
                    "inventory_id": row.get("linked_id"),
                    "status": row.get("status"),
                    "dosage": row.get("dosage"),
                    "frequency": row.get("frequency"),
                    "duration": row.get("duration"),
                    "route": row.get("route"),
                    "instructions": row.get("instructions"),
                    "suggestions": row.get("suggestions") or [],
                    "remarks": row.get("remarks"),
                })

    total = len(procedures) + len(inventory) + len(administered) + len(prescribed)
    st.caption(f"{total} billing line(s) from Phase 2 verification dashboard.")

    with st.expander("Procedure / Service", expanded=bool(procedures)):
        if not procedures:
            st.caption("No procedures or services detected.")
        for i, item in enumerate(procedures):
            st.divider()
            _render_procedure_row(item, f"proc_{i}", ctx)

    with st.expander("Add inventory item", expanded=bool(inventory)):
        if not inventory:
            st.caption("No retail / inventory items detected.")
        for i, item in enumerate(inventory):
            st.divider()
            _render_inventory_row(item, f"inv_{i}", ctx)

    with st.expander("Add administered items", expanded=bool(administered)):
        if not administered:
            st.caption("No in-clinic administered medications detected.")
        for i, item in enumerate(administered):
            st.divider()
            _render_administered_row(item, f"adm_{i}", ctx)

    with st.expander("Add medication (Prescription)", expanded=bool(prescribed)):
        if not prescribed:
            st.caption("No prescriptions detected.")
        for i, item in enumerate(prescribed):
            st.divider()
            _render_prescription_row(item, f"rx_{i}", ctx)

    st.download_button(
        "Download verification dashboard JSON",
        data=json.dumps(dashboard, indent=2, ensure_ascii=False),
        file_name="verification_dashboard.json",
        mime="application/json",
        key="dl_verification_dashboard",
    )


def _self_check() -> None:
    ctx = _build_billing_context([
        {
            "concept": "Gabapentin",
            "source_text": "Owner reports ongoing use of gabapentin",
            "intent_context": "Prescribed",
            "attributes": {
                "primary_local_suggestion": {"name": "Vetina Gabapentin 100 mg", "stock_id": 42187},
                "suggestions": [{"name": "Vetina Gabapentin 100 mg", "stock_id": 42187, "match_score": 0.66}],
            },
        },
        {
            "concept": "Antacid tablet",
            "source_text": "half tablet of antacid medication today",
            "attributes": {"dose": "half tablet"},
            "local_stock_id": 41945,
        },
    ])
    rx, auto = auto_fill_item(
        {"item_name": "Gabapentin", "status": "ACTION REQUIRED", "frequency": "Custom", "suggestions": []},
        ctx,
    )
    assert auto and rx["stock_id"] == 42187 and rx["frequency"] == "Ongoing"

    adm, auto2 = auto_fill_item({"item_name": "Antacid tablet", "dosage": "", "route": ""}, ctx)
    assert adm["dosage"] == "half tablet" and adm["route"] == "PO"

    split = _split_unlinked({
        "module_4_unlinked_entities": [
            {"item_name": "G", "intent_context": "Prescribed", "kind": "Medicine"},
            {"item_name": "P", "intent_context": "Performed", "kind": "Procedure"},
        ],
    })
    assert len(split["prescribed"]) == 1 and len(split["procedures"]) == 1


if __name__ == "__main__":
    _self_check()
    print("billing_display self-check ok")

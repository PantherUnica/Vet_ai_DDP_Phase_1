#!/usr/bin/env python3
"""
Stage-wise full pipeline benchmark scorer (Steps 1–4).

Example:
  python pipeline_benchmark.py ^
    --hypothesis-root benchmark_runs/deepgram_nova-3 ^
    --reference-root benchmark_runs/references/full ^
    --gemini-fallback-dir benchmark_runs/references/gemini ^
    --output benchmark_runs/reports/pipeline_benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from asr_benchmark import score_pair
from benchmark_utils import (
    BENCHMARK_ROOT,
    find_latest_glob,
    find_latest_run_dir,
    full_reference_dir,
    normalize_text,
)

try:
    import jiwer
except ImportError:
    jiwer = None  # type: ignore


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def discover_slugs(hypothesis_root: Path) -> List[str]:
    if not hypothesis_root.is_dir():
        return []
    return sorted(
        p.name for p in hypothesis_root.iterdir()
        if p.is_dir() and p.name != "archive"
    )


def _entity_names(entity: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ("normalized_name", "span_text", "display_name", "search_term", "name"):
        val = entity.get(key)
        if val and isinstance(val, str):
            names.append(normalize_text(val))
    return [n for n in names if n]


def _load_hypothesis_entities(run_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = find_latest_glob(run_dir, "entity_manifest_*.json")
    if manifest_path:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "entities" in data:
            return data["entities"]
    brain_path = find_latest_glob(run_dir, "brain_ner_output_*.json")
    if brain_path:
        data = json.loads(brain_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "entities" in data:
            return data["entities"]
        if isinstance(data, list):
            return data
    return []


def _gold_entity_keys(entities: List[Dict[str, Any]]) -> List[str]:
    keys = []
    for ent in entities:
        name = ent.get("name") or ent.get("normalized_name") or ""
        if name:
            keys.append(normalize_text(str(name)))
    return keys


def _match_gold_entity(gold_key: str, hyp_names: List[str]) -> bool:
    if not gold_key:
        return False
    for hn in hyp_names:
        if not hn:
            continue
        if gold_key in hn or hn in gold_key:
            return True
    return False


def score_entities(gold_entities: List[Dict[str, Any]], hyp_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    gold_keys = _gold_entity_keys(gold_entities)
    if not gold_keys:
        return {"precision": None, "recall": None, "f1": None, "note": "empty gold entities"}

    hyp_names_flat: List[str] = []
    for ent in hyp_entities:
        hyp_names_flat.extend(_entity_names(ent))

    matched_gold: Set[str] = set()
    matched_hyp_indices: Set[int] = set()
    for gi, gk in enumerate(gold_keys):
        for hi, ent in enumerate(hyp_entities):
            names = _entity_names(ent)
            if _match_gold_entity(gk, names):
                matched_gold.add(gk)
                matched_hyp_indices.add(hi)
                break

    tp = len(matched_gold)
    recall = tp / len(gold_keys) if gold_keys else None
    precision = tp / len(hyp_entities) if hyp_entities else (0.0 if gold_keys else None)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_count": len(gold_keys),
        "hypothesis_count": len(hyp_entities),
        "true_positives": tp,
    }


def _soap_text_from_run(run_dir: Path) -> Optional[str]:
    json_path = find_latest_glob(run_dir, "soap_note_*.json")
    if json_path:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                parts = [str(v) for v in data.values() if v]
                return "\n".join(parts)
        except (json.JSONDecodeError, OSError):
            pass
    txt_path = find_latest_glob(run_dir, "soap_note_*.txt")
    if txt_path:
        return _load_text(txt_path)
    return None


def score_soap_facts(facts: List[str], soap_text: str) -> Dict[str, Any]:
    if not facts:
        return {"fact_recall": None, "facts_matched": 0, "facts_total": 0}
    norm_soap = normalize_text(soap_text or "")
    matched = 0
    missing: List[str] = []
    for fact in facts:
        nf = normalize_text(str(fact))
        if nf and nf in norm_soap:
            matched += 1
        else:
            missing.append(str(fact))
    total = len(facts)
    return {
        "fact_recall": matched / total if total else None,
        "facts_matched": matched,
        "facts_total": total,
        "missing_facts": missing,
    }


def run_pipeline_benchmark(
    hypothesis_root: Path,
    reference_root: Path,
    *,
    gemini_fallback_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    slugs = discover_slugs(hypothesis_root)
    rows: List[Dict[str, Any]] = []
    missing_references: List[str] = []
    missing_hypotheses: List[str] = []
    skipped_no_gold: List[str] = []

    for slug in slugs:
        hyp_dir = find_latest_run_dir(hypothesis_root, slug)
        if hyp_dir is None:
            missing_hypotheses.append(slug)
            continue

        ref_dir = reference_root / slug
        has_full_gold = ref_dir.is_dir() and any(ref_dir.iterdir()) if ref_dir.is_dir() else False
        if not has_full_gold:
            skipped_no_gold.append(slug)

        row: Dict[str, Any] = {
            "audio_slug": slug,
            "run_dir": str(hyp_dir),
            "stages_scored": [],
        }

        # Step 1
        step1_hyp = hyp_dir / "step1_raw_transcription.txt"
        step1_ref = ref_dir / "transcript.txt" if has_full_gold else None
        if step1_ref and not step1_ref.is_file() and gemini_fallback_dir:
            step1_ref = gemini_fallback_dir / f"{slug}.txt"
        if step1_hyp.is_file() and step1_ref and step1_ref.is_file():
            m1 = score_pair(_load_text(step1_ref), _load_text(step1_hyp))
            row["step1_wer"] = m1.get("wer")
            row["step1_cer"] = m1.get("cer")
            row["step1_reference"] = str(step1_ref)
            row["stages_scored"].append("step1")

        # Step 2
        if has_full_gold:
            cleaned_ref = ref_dir / "cleaned_english.txt"
            cleaned_hyp = find_latest_glob(hyp_dir, "cleaned_transcript_*.txt")
            if cleaned_ref.is_file() and cleaned_hyp:
                m2 = score_pair(_load_text(cleaned_ref), _load_text(cleaned_hyp))
                row["step2_wer"] = m2.get("wer")
                row["step2_cer"] = m2.get("cer")
                row["stages_scored"].append("step2")

            # Entities
            entities_ref = ref_dir / "entities.json"
            if entities_ref.is_file():
                gold_data = json.loads(entities_ref.read_text(encoding="utf-8"))
                gold_entities = gold_data.get("entities", gold_data if isinstance(gold_data, list) else [])
                hyp_entities = _load_hypothesis_entities(hyp_dir)
                if hyp_entities is not None:
                    em = score_entities(gold_entities, hyp_entities)
                    row["entity_precision"] = em.get("precision")
                    row["entity_recall"] = em.get("recall")
                    row["entity_f1"] = em.get("f1")
                    row["stages_scored"].append("entities")

            # SOAP facts
            facts_ref = ref_dir / "soap_facts.json"
            if facts_ref.is_file():
                facts_data = json.loads(facts_ref.read_text(encoding="utf-8"))
                facts = facts_data.get("facts", [])
                soap_text = _soap_text_from_run(hyp_dir)
                if soap_text:
                    sm = score_soap_facts(facts, soap_text)
                    row["soap_fact_recall"] = sm.get("fact_recall")
                    row["soap_facts_matched"] = sm.get("facts_matched")
                    row["soap_facts_total"] = sm.get("facts_total")
                    row["stages_scored"].append("soap")

        if row["stages_scored"]:
            rows.append(row)
        elif not has_full_gold and slug not in [r["audio_slug"] for r in rows]:
            if "step1" not in row.get("stages_scored", []):
                missing_references.append(slug)

    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None

    summary = {
        "files_scored": len(rows),
        "mean_step1_wer": _mean("step1_wer"),
        "mean_step1_cer": _mean("step1_cer"),
        "mean_step2_wer": _mean("step2_wer"),
        "mean_step2_cer": _mean("step2_cer"),
        "mean_entity_f1": _mean("entity_f1"),
        "mean_soap_fact_recall": _mean("soap_fact_recall"),
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis_root": str(hypothesis_root),
        "reference_root": str(reference_root),
        "gemini_fallback_dir": str(gemini_fallback_dir) if gemini_fallback_dir else None,
        "summary": summary,
        "rows": rows,
        "missing_references": missing_references,
        "missing_hypotheses": missing_hypotheses,
        "skipped_no_gold_pack": skipped_no_gold,
    }


def write_csv(report: Dict[str, Any], path: Path) -> None:
    fieldnames = [
        "audio_slug",
        "stages_scored",
        "step1_wer",
        "step1_cer",
        "step2_wer",
        "step2_cer",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "soap_fact_recall",
        "soap_facts_matched",
        "soap_facts_total",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in report.get("rows", []):
            out = dict(row)
            out["stages_scored"] = ",".join(row.get("stages_scored", []))
            writer.writerow(out)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Full pipeline stage benchmark scorer.")
    parser.add_argument("--hypothesis-root", required=True)
    parser.add_argument("--reference-root", default=str(BENCHMARK_ROOT / "references" / "full"))
    parser.add_argument(
        "--gemini-fallback-dir",
        default=str(BENCHMARK_ROOT / "references" / "gemini"),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if jiwer is None:
        print("Error: jiwer not installed. Run: pip install jiwer", file=sys.stderr)
        return 1

    hypothesis_root = Path(args.hypothesis_root).resolve()
    reference_root = Path(args.reference_root).resolve()
    gemini_dir = Path(args.gemini_fallback_dir).resolve() if args.gemini_fallback_dir else None

    report = run_pipeline_benchmark(
        hypothesis_root,
        reference_root,
        gemini_fallback_dir=gemini_dir,
    )

    out_prefix = Path(
        args.output
        or (BENCHMARK_ROOT / "reports" / f"pipeline_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json") if out_prefix.suffix else Path(str(out_prefix) + ".json")
    csv_path = json_path.with_suffix(".csv")

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(report, csv_path)

    s = report["summary"]
    print(f"Scored: {s['files_scored']} file(s)")
    if s["mean_step1_wer"] is not None:
        print(f"Mean Step1 WER: {s['mean_step1_wer']:.4f}")
    if s["mean_entity_f1"] is not None:
        print(f"Mean Entity F1: {s['mean_entity_f1']:.4f}")
    if s["mean_soap_fact_recall"] is not None:
        print(f"Mean SOAP fact recall: {s['mean_soap_fact_recall']:.4f}")
    print(f"Report: {json_path}")
    print(f"CSV:    {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

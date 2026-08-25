#!/usr/bin/env python3
"""Import clinical_notes.xlsx into benchmark_runs/clinical_gold/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import openpyxl
except ImportError:
    print("Install openpyxl: pip install openpyxl", file=sys.stderr)
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark_utils import CLINICAL_GOLD_DIR  # noqa: E402

COLUMNS = [
    "audio_url",
    "transcript",
    "tag",
    "subjective",
    "objective",
    "assessment",
    "plan",
    "conclusion",
    "key_issues",
    "abnormal_findings",
    "customer_instructions",
    "reminders",
]

HEADER_MAP = {
    "Audio Files (URL)": "audio_url",
    "Transcript": "transcript",
    "Tag": "tag",
    "Subjective": "subjective",
    "Objective": "objective",
    "Assessment": "assessment",
    "Plan": "plan",
    "Conclusion": "conclusion",
    "Key Issues": "key_issues",
    "Abnormal Findings": "abnormal_findings",
    "Customer Instructions": "customer_instructions",
    "Reminders": "reminders",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Import clinical_notes.xlsx to clinical_gold/")
    parser.add_argument(
        "--xlsx",
        default=str(Path.home() / "Downloads" / "clinical_notes.xlsx"),
        help="Path to clinical_notes.xlsx",
    )
    args = parser.parse_args()
    xlsx = Path(args.xlsx)
    if not xlsx.is_file():
        print(f"File not found: {xlsx}", file=sys.stderr)
        return 1

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    header = next(rows_iter)
    col_idx = {}
    for i, h in enumerate(header):
        if h and str(h) in HEADER_MAP:
            col_idx[HEADER_MAP[str(h)]] = i

    out_rows = CLINICAL_GOLD_DIR / "rows"
    out_rows.mkdir(parents=True, exist_ok=True)
    index = []

    for n, row in enumerate(rows_iter, start=1):
        if not row or not any(row):
            continue
        record = {}
        for key, idx in col_idx.items():
            val = row[idx] if idx < len(row) else None
            record[key] = val if val is not None else ""
        row_id = f"row_{n:03d}"
        path = out_rows / f"{row_id}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        index.append({"id": row_id, "tag": record.get("tag", ""), "file": str(path.name)})

    index_path = CLINICAL_GOLD_DIR / "index.json"
    index_path.write_text(
        json.dumps({"count": len(index), "rows": index}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Imported {len(index)} rows to {CLINICAL_GOLD_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

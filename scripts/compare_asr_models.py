#!/usr/bin/env python3
"""
Benchmark ASR providers/models on gold audio. Only recommend a switch when WER/CER
is equal or better than baseline (quality-preserving).

Example:
  python scripts/compare_asr_models.py --input-dir input_audio_examples --reference-dir benchmark_runs/references/gemini
  python scripts/compare_asr_models.py --audio test.wav --configs deepgram:nova-3,deepgram:nova-2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip("\"'")
        if key not in os.environ:
            os.environ[key] = val


_load_dotenv()

from asr_benchmark import score_pair  # noqa: E402
from asr_providers import transcribe  # noqa: E402
from benchmark_utils import audio_slug  # noqa: E402

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".webm"}


def _parse_config(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Config must be provider:model, got {spec!r}")
    provider, model = spec.split(":", 1)
    return provider.strip().lower(), model.strip()


def _discover_audio(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def _reference_for(audio_path: Path, reference_dir: Optional[Path]) -> Optional[Path]:
    if not reference_dir:
        return None
    slug = audio_slug(audio_path)
    ref = reference_dir / f"{slug}.txt"
    return ref if ref.is_file() else None


def compare_configs(
    audio_files: List[Path],
    configs: List[Tuple[str, str]],
    *,
    reference_dir: Optional[Path],
    language: Optional[str],
) -> Dict[str, Any]:
    baseline_key = f"{configs[0][0]}:{configs[0][1]}" if configs else ""
    results: Dict[str, Any] = {"configs": [], "files": [], "recommendations": []}

    per_config: Dict[str, Dict[str, List[float]]] = {}
    for provider, model in configs:
        key = f"{provider}:{model}"
        per_config[key] = {"wer": [], "cer": [], "latency_ms": []}
        config_row = {"key": key, "provider": provider, "model": model, "files": []}

        for audio in audio_files:
            ref_path = _reference_for(audio, reference_dir)
            try:
                tr = transcribe(str(audio), provider=provider, model=model, language=language)
            except Exception as exc:
                file_row = {
                    "audio": str(audio),
                    "error": str(exc),
                    "latency_ms": None,
                    "char_count": None,
                    "wer": None,
                    "cer": None,
                }
                config_row["files"].append(file_row)
                continue
            file_row: Dict[str, Any] = {
                "audio": str(audio),
                "latency_ms": tr.latency_ms,
                "char_count": tr.char_count,
                "wer": None,
                "cer": None,
                "quality_ok_vs_baseline": None,
            }
            if ref_path:
                scores = score_pair(ref_path.read_text(encoding="utf-8"), tr.text)
                file_row["wer"] = scores.get("wer")
                file_row["cer"] = scores.get("cer")
                if file_row["wer"] is not None:
                    per_config[key]["wer"].append(file_row["wer"])
                if file_row["cer"] is not None:
                    per_config[key]["cer"].append(file_row["cer"])
            per_config[key]["latency_ms"].append(float(tr.latency_ms))
            config_row["files"].append(file_row)

        if per_config[key]["latency_ms"]:
            config_row["mean_latency_ms"] = sum(per_config[key]["latency_ms"]) / len(per_config[key]["latency_ms"])
        if per_config[key]["wer"]:
            config_row["mean_wer"] = sum(per_config[key]["wer"]) / len(per_config[key]["wer"])
        if per_config[key]["cer"]:
            config_row["mean_cer"] = sum(per_config[key]["cer"]) / len(per_config[key]["cer"])
        results["configs"].append(config_row)

    if len(configs) > 1 and baseline_key in per_config:
        base = next(c for c in results["configs"] if c["key"] == baseline_key)
        base_wer = base.get("mean_wer")
        base_cer = base.get("mean_cer")
        for cfg in results["configs"]:
            if cfg["key"] == baseline_key:
                continue
            wer_ok = base_wer is None or (
                cfg.get("mean_wer") is not None and cfg["mean_wer"] <= base_wer
            )
            cer_ok = base_cer is None or (
                cfg.get("mean_cer") is not None and cfg["mean_cer"] <= base_cer
            )
            faster = cfg.get("mean_latency_ms", 1e9) < base.get("mean_latency_ms", 1e9)
            if wer_ok and cer_ok and faster:
                results["recommendations"].append(
                    f"Consider {cfg['key']}: faster ({cfg.get('mean_latency_ms'):.0f}ms vs "
                    f"{base.get('mean_latency_ms'):.0f}ms) with WER/CER <= baseline"
                )
            elif faster and not (wer_ok and cer_ok):
                results["recommendations"].append(
                    f"Do NOT switch to {cfg['key']}: faster but WER/CER worse than {baseline_key}"
                )

    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ASR models with quality gate.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--audio", action="append", default=[])
    parser.add_argument("--reference-dir", default=str(ROOT / "benchmark_runs" / "references" / "gemini"))
    parser.add_argument("--configs", default="deepgram:nova-3,deepgram:nova-2")
    parser.add_argument("--asr-language", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    files = [Path(a) for a in args.audio]
    if args.input_dir:
        files.extend(_discover_audio(Path(args.input_dir)))
    if not files:
        print("No audio files.", file=sys.stderr)
        return 1

    configs = [_parse_config(c.strip()) for c in args.configs.split(",") if c.strip()]
    report = compare_configs(
        files,
        configs,
        reference_dir=Path(args.reference_dir) if args.reference_dir else None,
        language=args.asr_language,
    )

    print(f"Compared {len(files)} file(s) across {len(configs)} config(s)\n")
    for cfg in report["configs"]:
        print(
            f"  {cfg['key']}: mean_latency={cfg.get('mean_latency_ms', 'n/a'):.0f}ms "
            f"mean_wer={cfg.get('mean_wer', 'n/a')} mean_cer={cfg.get('mean_cer', 'n/a')}"
            if cfg.get("mean_latency_ms") is not None
            else f"  {cfg['key']}"
        )
    for rec in report.get("recommendations") or []:
        print(f"\n→ {rec}")

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

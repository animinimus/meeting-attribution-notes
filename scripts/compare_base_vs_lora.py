#!/usr/bin/env python3
"""
Compare base Qwen2.5-7B-Instruct vs LoRA-tuned meeting attribution on held-out cases.

Variants:
  base        — Hugging Face Inference API, model Qwen/Qwen2.5-7B-Instruct
  lora_hf     — HF API, default endpoint (fine-tuned deployment if available)
  lora_local  — Local 4-bit base + LoRA adapter (requires GPU + adapter weights)

Example:
  python3 scripts/compare_base_vs_lora.py --variants base,lora_local
  python3 scripts/compare_base_vs_lora.py --write-held-out
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.speaker_eval import (  # noqa: E402
    DEFAULT_HELD_OUT_PATH,
    build_held_out_from_training,
    load_eval_cases,
    run_speaker_eval,
)


def _hardware_note() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return f"GPU: {out.stdout.strip()}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "GPU: none detected (CPU-only or no nvidia-smi)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base vs LoRA speaker attribution on held-out JSONL cases.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=DEFAULT_HELD_OUT_PATH,
        help="JSONL eval cases (input + expected_speakers).",
    )
    parser.add_argument(
        "--write-held-out",
        action="store_true",
        help="Regenerate held-out file from data/train/meeting_examples.jsonl indices 6,7,8.",
    )
    parser.add_argument(
        "--variants",
        default="base,lora_local",
        help="Comma-separated: base, lora_hf, lora_local",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="LoRA adapter directory (lora_local only).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write full report JSON.",
    )
    args = parser.parse_args()

    train_path = ROOT / "data" / "train" / "meeting_examples.jsonl"
    if args.write_held_out or not args.eval_path.exists():
        build_held_out_from_training(
            train_path,
            holdout_indices=(6, 7, 8),
            out_path=args.eval_path,
        )
        print(f"Wrote held-out cases to {args.eval_path}")

    cases = load_eval_cases(args.eval_path)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "hardware": _hardware_note(),
        "dataset": str(args.eval_path.relative_to(ROOT)),
        "dataset_note": (
            "Three examples (indices 6–8) from meeting_examples.jsonl: "
            "incident hotfix, unlabeled VTT design review, Acme SSO check-in. "
            "Gold labels from training file; not used for SFT holdout in training "
            "script (none in repo)—treat as a small sanity suite only."
        ),
        "case_ids": [c.get("id") for c in cases],
    }

    report: dict = {"meta": meta, "variants": {}}
    errors: list[str] = []

    for variant in variants:
        try:
            result = run_speaker_eval(
                cases,
                variant,  # type: ignore[arg-type]
                adapter_path=args.adapter_path,
                temperature=args.temperature,
            )
            report["variants"][variant] = result
            s = result["summary"]
            print(f"\n=== {variant} ===")
            print(f"Cases scored: {s['cases_scored']}/{s['cases_total']}")
            if s["length_mismatches"]:
                print(f"Length mismatches (skipped): {s['length_mismatches']}")
            print(f"Mean line accuracy: {s['mean_line_accuracy']}")
            print(f"Mean macro F1:      {s['mean_macro_f1']}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{variant}: {exc}")
            print(f"\n=== {variant} === FAILED: {exc}", file=sys.stderr)

    if errors:
        report["errors"] = errors

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote report to {args.output_json}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
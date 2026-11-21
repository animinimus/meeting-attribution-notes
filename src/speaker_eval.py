"""Load speaker-attribution eval cases and score model runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from src.inference import BASE_MODEL, DEFAULT_MODEL, run_meeting_analysis
from src.schemas import MeetingInput, Participant, TranscriptLine
from src.speaker_metrics import aggregate_metrics, line_accuracy, per_speaker_f1
from src.training_data import load_training_jsonl

Variant = Literal["base", "lora_hf", "lora_local"]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HELD_OUT_PATH = ROOT / "data" / "eval" / "speaker_attribution_held_out.jsonl"


def _meeting_from_row(row: dict[str, Any]) -> MeetingInput:
    inp = row["input"]
    return MeetingInput(
        title=inp["title"],
        participants=[Participant(**p) for p in inp.get("participants", [])],
        lines=[TranscriptLine(**line) for line in inp["lines"]],
    )


def expected_speakers_from_row(row: dict[str, Any]) -> list[str]:
    return [line["speaker_name"] for line in row["output"]["attributed_lines"]]


def load_eval_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            cases.append(json.loads(stripped))
    return cases


def build_held_out_from_training(
    train_path: Path,
    holdout_indices: tuple[int, ...],
    out_path: Path,
) -> list[dict[str, Any]]:
    """
    Materialize held-out JSONL rows from gold training examples (by 0-based index).
    Does not modify the training file.
    """
    examples = load_training_jsonl(train_path)
    rows: list[dict[str, Any]] = []
    for idx in holdout_indices:
        ex = examples[idx]
        case_id = ex.get("id") or f"train_idx_{idx}"
        rows.append(
            {
                "id": case_id,
                "source": f"{train_path.name}#{idx}",
                "input": ex["input"],
                "expected_speakers": expected_speakers_from_row(ex),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def run_variant(
    meeting: MeetingInput,
    variant: Variant,
    *,
    adapter_path: str | Path | None = None,
    temperature: float = 0.2,
) -> list[str]:
    if variant == "base":
        result = run_meeting_analysis(
            meeting,
            model=BASE_MODEL,
            backend="hf",
            temperature=temperature,
        )
    elif variant == "lora_hf":
        result = run_meeting_analysis(
            meeting,
            model=DEFAULT_MODEL,
            backend="hf",
            temperature=temperature,
        )
    elif variant == "lora_local":
        result = run_meeting_analysis(
            meeting,
            backend="local",
            adapter_path=adapter_path,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")
    return [line.speaker_name for line in result.attributed_lines]


def score_case_row(
    row: dict[str, Any],
    variant: Variant,
    *,
    adapter_path: str | Path | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    meeting = _meeting_from_row(row)
    expected = row["expected_speakers"]
    predicted = run_variant(
        meeting, variant, adapter_path=adapter_path, temperature=temperature
    )
    acc = line_accuracy(expected, predicted)
    macro_f1, per_label = per_speaker_f1(expected, predicted)
    return {
        "id": row.get("id"),
        "expected_speakers": expected,
        "predicted_speakers": predicted,
        "line_accuracy": acc,
        "macro_f1": macro_f1,
        "per_speaker_f1": per_label,
    }


def run_speaker_eval(
    cases: list[dict[str, Any]],
    variant: Variant,
    *,
    adapter_path: str | Path | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    case_results: list[dict] = []
    for row in cases:
        case_results.append(
            score_case_row(
                row, variant, adapter_path=adapter_path, temperature=temperature
            )
        )
    summary = aggregate_metrics(case_results)
    return {
        "variant": variant,
        "case_results": case_results,
        "summary": summary,
    }
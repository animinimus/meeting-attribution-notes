"""Speaker-attribution metrics for eval reports."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


def _normalize_name(name: str) -> str:
    return name.strip().casefold()


def line_accuracy(expected: Sequence[str], predicted: Sequence[str]) -> float | None:
    """
    Fraction of lines where predicted speaker matches expected (case-insensitive).
    Returns None if lengths differ.
    """
    if len(expected) != len(predicted):
        return None
    if not expected:
        return 1.0
    hits = sum(
        1
        for exp, pred in zip(expected, predicted, strict=True)
        if _normalize_name(exp) == _normalize_name(pred)
    )
    return hits / len(expected)


def per_speaker_f1(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> tuple[float | None, dict[str, dict[str, float]]]:
    """
    Macro-averaged F1 over speaker labels appearing in expected labels.

    Each transcript line is one classification into a speaker name.
    """
    if len(expected) != len(predicted):
        return None, {}

    labels = sorted({_normalize_name(s) for s in expected})
    if not labels:
        return 1.0, {}

    pred_norm = [_normalize_name(p) for p in predicted]
    exp_norm = [_normalize_name(e) for e in expected]
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for label in labels:
        for pred_l, exp_l in zip(pred_norm, exp_norm, strict=True):
            if pred_l == label and exp_l == label:
                tp[label] += 1
            elif pred_l == label and exp_l != label:
                fp[label] += 1
            elif pred_l != label and exp_l == label:
                fn[label] += 1

    per_label: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values)
    return macro_f1, per_label


def aggregate_metrics(case_results: list[dict]) -> dict:
    """Combine per-case line accuracy and macro F1 (length mismatches skipped)."""
    accs: list[float] = []
    f1s: list[float] = []
    length_mismatches = 0
    for row in case_results:
        acc = row.get("line_accuracy")
        if acc is None:
            length_mismatches += 1
            continue
        accs.append(acc)
        macro = row.get("macro_f1")
        if macro is not None:
            f1s.append(macro)

    return {
        "cases_total": len(case_results),
        "cases_scored": len(accs),
        "length_mismatches": length_mismatches,
        "mean_line_accuracy": sum(accs) / len(accs) if accs else None,
        "mean_macro_f1": sum(f1s) / len(f1s) if f1s else None,
    }
from src.speaker_metrics import aggregate_metrics, line_accuracy, per_speaker_f1


def test_line_accuracy_perfect():
    expected = ["Alice", "Bob", "Alice"]
    predicted = ["Alice", "Bob", "Alice"]
    assert line_accuracy(expected, predicted) == 1.0


def test_line_accuracy_mismatch_length():
    assert line_accuracy(["A"], ["A", "B"]) is None


def test_macro_f1_two_speakers():
    expected = ["Alice", "Bob", "Alice", "Bob"]
    predicted = ["Alice", "Bob", "Bob", "Bob"]
    macro, per = per_speaker_f1(expected, predicted)
    assert macro is not None
    assert "alice" in per and "bob" in per
    assert per["alice"]["recall"] == 0.5


def test_aggregate_metrics():
    rows = [
        {"line_accuracy": 1.0, "macro_f1": 1.0},
        {"line_accuracy": 0.5, "macro_f1": 0.5},
        {"line_accuracy": None, "macro_f1": None},
    ]
    agg = aggregate_metrics(rows)
    assert agg["cases_total"] == 3
    assert agg["cases_scored"] == 2
    assert agg["mean_line_accuracy"] == 0.75
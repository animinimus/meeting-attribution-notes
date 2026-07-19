import json
from pathlib import Path

from src.training_data import load_training_jsonl, row_to_messages, row_to_meeting_input


def test_load_training_jsonl_has_examples():
    path = Path("data/train/meeting_examples.jsonl")
    examples = load_training_jsonl(path)
    assert len(examples) >= 1


def test_row_to_messages_matches_inference_shape():
    examples = load_training_jsonl("data/train/meeting_examples.jsonl")
    row = examples[0]
    messages = row_to_messages(row)

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"

    meeting = row_to_meeting_input(row)
    assert meeting.title == row["input"]["title"]
    assert len(meeting.lines) == len(row["input"]["lines"])

    assistant = json.loads(messages[2]["content"])
    assert len(assistant["attributed_lines"]) == len(row["input"]["lines"])

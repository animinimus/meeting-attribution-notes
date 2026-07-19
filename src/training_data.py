import json
from pathlib import Path
from typing import Any

from src.prompt import build_system_prompt, build_user_prompt
from src.schemas import MeetingInput, Participant, TranscriptLine


def row_to_meeting_input(row: dict[str, Any]) -> MeetingInput:
    """
    Convert one JSONL training row's input block into a MeetingInput.
    """
    inp = row["input"]
    return MeetingInput(
        title=inp["title"],
        participants=[Participant(**p) for p in inp.get("participants", [])],
        lines=[TranscriptLine(**line) for line in inp["lines"]],
    )


def row_to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """
    Build chat messages for SFT: system + user prompts match inference.py,
    assistant message is the gold JSON output.
    """
    meeting = row_to_meeting_input(row)
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(meeting)},
        {"role": "assistant", "content": json.dumps(row["output"], indent=2)},
    ]


def load_training_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Load all examples from a JSONL file
    """
    examples: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            examples.append(json.loads(stripped))
    return examples

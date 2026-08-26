from pathlib import Path

import pytest

from src.inference import _resolve_adapter_path
from src.schemas import MeetingInput, TranscriptLine
from src.inference import run_meeting_analysis


def test_resolve_adapter_path_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="LoRA adapter not found"):
        _resolve_adapter_path(tmp_path / "missing-adapter")


def test_local_backend_raises_when_adapter_missing():
    meeting = MeetingInput(
        title="Test",
        participants=[],
        lines=[TranscriptLine(raw_label="Speaker 1", text="Hello.")],
    )
    with pytest.raises(FileNotFoundError):
        run_meeting_analysis(
            meeting,
            backend="local",
            adapter_path=Path("/nonexistent/adapter"),
        )

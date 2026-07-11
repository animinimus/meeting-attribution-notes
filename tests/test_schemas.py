from src.schemas import (
    Participant,
    MeetingInput,
    TranscriptLine,
    AttributedLine,
)


def test_meeting_input_basic():
    inp = MeetingInput(
        title="Sync with Alice and Bob",
        participants=[
            Participant(name="Alice", role="design"),
            Participant(name="Bob", role="finance"),
        ],
        lines=[
            TranscriptLine(raw_label="Speaker 1", text="Hey Bob, how's the budget?"),
            TranscriptLine(raw_label="Speaker 2", text="Tight, but we can shuffle Q3."),
        ],
    )
    assert inp.participants[0].name == "Alice"
    assert len(inp.lines) == 2


def test_attributed_line_confidence():
    line = AttributedLine(
        text="We need new mockups.",
        speaker_name="Alice",
        confidence=0.82,
        reasoning="Mentioned design team needs",
    )
    assert line.speaker_name == "Alice"
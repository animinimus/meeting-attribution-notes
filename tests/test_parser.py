from pathlib import Path

from src.parser import (
    parse_meeting_text, 
    parse_transcript_lines, 
    split_title_from_transcript,
    is_vtt_transcript,
)
from src.schemas import Participant

SAMPLE_PATH = Path(__file__).resolve().parents[1]/"data"/"sample_transcript.txt"
ZOOM_PATH = Path(__file__).resolve().parents[1]/"data"/"sample_transcript_zoom.txt"
TEAMS_PATH = Path(__file__).resolve().parents[1]/"data"/"sample_transcript_teams.txt"

def test_is_vtt_transcript_zoom():
    raw = ZOOM_PATH.read_text(encoding="utf-8")
    assert is_vtt_transcript(raw) is True

def test_is_vtt_transcript_teams():
    raw = TEAMS_PATH.read_text(encoding="utf-8")
    assert is_vtt_transcript(raw) is True

def test_parse_zoom_vtt_lines():
    raw = ZOOM_PATH.read_text(encoding="utf-8")
    lines = parse_transcript_lines(raw)
    assert len(lines) == 6
    assert lines[0].raw_label == "Alice McAliceFace"
    assert "budget" in lines[0].text.lower()
    assert lines[1].raw_label == "Bob McBobFace"

def test_parse_teams_vtt_lines():
    raw = TEAMS_PATH.read_text(encoding="utf-8")
    lines = parse_transcript_lines(raw)
    assert len(lines) == 6
    assert lines[0].raw_label == "Alice McAliceFace"
    assert lines[0].text.startswith("Hey Bob")
    assert lines[-1].raw_label == "Bob McBobFace"
    assert "headcount" in lines[-1].text.lower()

def test_parse_unlabeled_vtt_lines():
    """VTT with timestamp then dialogue only (no speaker name rows)."""
    raw = """00:00:00.000 --> 00:00:05.200
Hey Bob, how's the budget?
00:00:05.800 --> 00:00:11.400
Tight. Alice, can design pause hires?
"""
    lines = parse_transcript_lines(raw)
    assert len(lines) == 2
    assert lines[0].raw_label is None
    assert "budget" in lines[0].text.lower()
    assert lines[1].raw_label is None
    assert "alice" in lines[1].text.lower()

def test_parse_transcript_lines_basic():
    raw = """
    Speaker 1: Hey bob, Happy Monday! How are you doing today?
    Speaker 2: I'm doing great, thanks! How about you?
    """

    lines = parse_transcript_lines(raw)

    assert len(lines) == 2
    assert lines[0].raw_label == "Speaker 1"
    assert lines[0].text == "Hey bob, Happy Monday! How are you doing today?"
    assert lines[1].raw_label == "Speaker 2"

def test_split_title_from_transcript():
    raw = SAMPLE_PATH.read_text()
    title, body = split_title_from_transcript(raw)

    assert title == "Sync with Alice and Bob"
    assert "Speaker 1:" in body
    assert "Speaker 2:" in body

def test_parse_meeting_text_from_file():
    raw = SAMPLE_PATH.read_text()
    participants = [
        Participant(name = "Alice", role = "design"),
        Participant(name = "Bob", role = "finance"),
    ]

    meeting = parse_meeting_text(raw, participants=participants)

    assert meeting.title == "Sync with Alice and Bob"
    assert len(meeting.participants) == 2
    assert len(meeting.lines) == 4
    assert meeting.lines[0].raw_label == "Speaker 1"

def test_parse_named_speaker_lines():
    raw = """
    Alice McAliceface: Hey Bob, can you design pause hires?
    Bob McBobface: Only if finance signs off on the runway numbers.
    """
    lines = parse_transcript_lines(raw)

    assert len(lines) == 2
    assert lines[0].raw_label == "Alice McAliceface"
    assert lines[1].raw_label == "Bob McBobface"
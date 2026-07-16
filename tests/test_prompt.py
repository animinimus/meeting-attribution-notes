from src.parser import parse_meeting_text
from src.prompt import build_messages, build_user_prompt
from src.schemas import Participant 
from pathlib import Path 

SAMPLE_PATH = Path(__file__).resolve().parents[1]/"data"/"sample_transcript.txt"

def test_user_prompt_containers_participants_and_lines():
    """
    test that the user prompt contains the participants and lines.
    """
    raw = SAMPLE_PATH.read_text(encoding="utf-8")

    participants = [
        Participant(name="Alice McAliceFace", role="design"),
        Participant(name="Bob McBobFace", role="finance"),
    ]

    meeting = parse_meeting_text(raw, participants=participants)

    prompt = build_user_prompt(meeting)

    assert "Alice McAliceFace" in prompt
    assert "Bob McBobFace" in prompt
    assert "Speaker 1" in prompt
    assert "attributed_lines" in prompt

def test_build_messages_has_system_and_user_roles():
    """
    test that the build_messages function returns a list with system and user roles.
    """
    raw = SAMPLE_PATH.read_text(encoding="utf-8")
    meeting = parse_meeting_text(raw, participants=[
        Participant(name="Alice McAliceFace"),
        Participant(name="Bob McBobFace"),
    ])

    messages = build_messages(meeting)
    
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    
    
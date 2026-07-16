import json
from src.schemas import MeetingInput

def build_system_prompt() -> str:
    """
    Ensure the model replies with JSON only via system prompting.
    """
    return (
        "You are a helpful meeting assistant analyst. Given a meeting title, participant list, and transcript lines, attribute each line to the most likely speaker by name. Use participant roles and departments as hints when attributing lines. Use dialogue cues (mentioned names, who is being addressed, topic expertise, etc.) to make your best guess. Confidence should be a float between 0 and 1. Respond with JSON ONLY. Do NOT reply with anything else (markdown, extra text, code, etc.)."
    )

def build_user_prompt(meeting: MeetingInput) -> str:
    """
    Build the user prompt for the meeting.
    """
    payload = {
        "title": meeting.title,
        "participants": [p.model_dump() for p in meeting.participants],
        "lines": [l.model_dump() for l in meeting.lines],
    }

    schema_hint = {
        "attributed_lines": [
            {
                "text": "original line text",
                "speaker_name": "must match a participant name",
                "confidence": 0.85,
                "reasoning": "brief explanation for why you made this attribution"
            }
        ],
        "summary_bullets": ["concise meeting summary points"],
        "action_items": ["who should do what"],
    }

    return (
        "Analyze this meeting and return JSON matching the output schema.\n\n"
        f"INPUT:\n{json.dumps(payload, indent=2)}\n\n"
        f"OUTPUT_SCHEMA:\n{json.dumps(schema_hint, indent=2)}"
    )

def build_messages(meeting: MeetingInput) -> list[dict]:
    """
    Build the messages for the meeting.
    """
    return [
        { "role": "system", "content": build_system_prompt() },
        { "role": "user", "content": build_user_prompt(meeting) }
    ]
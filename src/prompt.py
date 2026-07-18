import json
from src.schemas import MeetingInput


def build_system_prompt() -> str:
    """
    Ensure the model replies with JSON only via system prompting.
    """
    return (
        "You are a helpful meeting assistant analyst. "
        "Given a meeting title, an optional participant list, and transcript lines, "
        "attribute each line to the most likely speaker by their real name when possible. "
        "If participants are provided: use roles and departments as hints. "
        "If participants are empty: infer speaker names from dialogue across the whole transcript. "
        "When someone is addressed by name (e.g. 'Hey Bob'), the speaker is usually NOT Bob — "
        "they are the person talking to Bob. Use raw_label (Speaker 1, Speaker 2) only to track "
        "who said which line, then map speakers to real names from context. "
        "Do NOT output Speaker 1 or Speaker 2 as speaker_name when the transcript contains "
        "name clues that let you infer Alice, Bob, etc. "
        "If truly unknown, use Unknown with low confidence. "
        "For action_items: extract commitments and requests separately. Preserve direction — "
        "who does what and for whom. 'I'll send the spreadsheet' means the speaker sends it to the other party. "
        "'Send me the headcount sheet' means the speaker is asking the other party to send it to them. "
        "Do not merge two lines into one action item if they have different directions or owners. "
        "Confidence is a float between 0.0 and 1.0. "
        "Respond with JSON ONLY. Do NOT reply with markdown, extra text, or code fences."
    )


def _inference_instructions(meeting: MeetingInput) -> str:
    """Extra guidance when we have no participant hints."""
    if meeting.participants:
        return ""

    return (
        "INFERENCE RULES (participants list is empty):\n"
        "- Read the full transcript before attributing.\n"
        "- If Speaker 1 says 'Hey Bob', Speaker 1 is likely NOT Bob.\n"
        "- If Speaker 2 says 'Alice, can design pause hires?', Speaker 2 is likely NOT Alice.\n"
        "- If someone is addressed by name, they are likely not the speaker.\n"
        "- Do not use Speaker 1/Speaker 2, etc. as speaker_name when dialogue has name clues.\n"
        "- Assign real names (Alice, Bob, etc.) when dialogue makes them clear.\n"
        "- speaker_name must be a person's name, not 'Speaker 1' or 'Speaker 2', unless no names appear anywhere, "
        "in which case you should use clues from the transcript to infer names and ultimately use them if there are none.\n\n"
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
                "speaker_name": "real name inferred from dialogue (e.g. Alice), not Speaker N",
                "confidence": 0.85,
                "reasoning": "brief explanation for why you made this attribution",
            }
        ],
        "summary_bullets": ["concise meeting summary points"],
        "action_items": [
            "Bob McBobFace: Send spreadsheet to Alice after the call.",
            "Alice McAliceFace: Send headcount sheet to Bob by end of day.",
        ],
    }

    return (
        f"{_inference_instructions(meeting)}"
        "Analyze this meeting and return JSON matching the output schema.\n\n"
        f"INPUT:\n{json.dumps(payload, indent=2)}\n\n"
        f"OUTPUT_SCHEMA:\n{json.dumps(schema_hint, indent=2)}"
    )


def build_messages(meeting: MeetingInput) -> list[dict]:
    """
    Build the messages for the meeting.
    """
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(meeting)},
    ]

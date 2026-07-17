import json
from pathlib import Path

from src.inference import run_meeting_analysis
from src.parser import parse_meeting_text
from src.schemas import Participant

def load_eval_case(path: Path) -> dict:
    """
    Load an evaluation case from a JSON file.
    """
    return json.loads(path.read_text(encoding="utf-8"))

def build_meeting_from_eval(case: dict, root: Path):
    """
    Turn eval case input into MeetingInput via parser
    """
    inp = case["input"]
    transcript_path = root/inp["transcript_file"]
    raw = transcript_path.read_text(encoding="utf-8")

    participants = [Participant(**p) for p in inp["participants"]]

    meeting = parse_meeting_text(
        raw,
        participants=participants,
        title = inp.get("title")
    )
    return meeting

def score_case(case: dict, root: Path) -> dict:
    """
    Run model on eval case and compare it to expected output.
    """
    meeting = build_meeting_from_eval(case, root)
    result = run_meeting_analysis(meeting)

    expected = case["expected"]
    expected_speakers = expected["attributed_speakers"]

    actual_speakers = [l.speaker_name for l in result.attributed_lines]
    speaker_matches = [
        actual == expected for actual, expected in zip(actual_speakers, expected_speakers)
    ]

    line_count_match = len(actual_speakers) == len(expected_speakers)
    summary_ok = len(result.summary_bullets) >= expected.get("min_smmary_bullets", 1)

    action_text = " ".join(result.action_items).lower()
    required_keywords = expected.get("required_action_keywords", [])
    keywords_ok = all(kw.lower() in action_text for kw in required_keywords)

    passed = (
        line_count_match and
        all(speaker_matches) and
        summary_ok and
        keywords_ok
    )

    return {
        "id": case["id"],
        "passed": passed,
        "line_count_ok": line_count_match,
        "speaker_matches": speaker_matches,
        "actual_speakers": actual_speakers,
        "expected_speakers": expected_speakers,
        "summary_bullets": result.summary_bullets,
        "action_items": result.action_items,
    }
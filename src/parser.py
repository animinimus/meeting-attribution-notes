import re
from typing import List, Optional, Tuple

from src.schemas import MeetingInput, Participant, TranscriptLine

SPEAKER_LINE_PATTERN = re.compile(
    r"^\s*speaker\s*(\d+)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)

NAMED_LINE_PATTERN = re.compile(
    r"^\s*(.+?)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)

def parse_transcript_lines(raw_text: str) -> List[TranscriptLine]:
    """Parse raw text into transcript lines."""
    lines: List[TranscriptLine] = []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        match = SPEAKER_LINE_PATTERN.match(line)
        if match:
            speaker_num, text = match.group(1), match.group(2)
            lines.append(
                TranscriptLine(
                    raw_label = f"Speaker { speaker_num }",
                    text = text
                )
            )
            continue

        name_match = NAMED_LINE_PATTERN.match(line)
        if name_match:
            name = name_match.group(1).strip()
            text = name_match.group(2).strip()
            lines.append(
                TranscriptLine(
                    raw_label = name,
                    text = text,
                )
            )
        continue

        if lines:
            # if we have a line that does not have an explicit speaker, use the last speaker
            prev = lines[-1]
            lines[-1] = TranscriptLine(
                raw_label = prev.raw_label,
                text = f"{prev.text} {line}"
            )

    return lines

def split_title_from_transcript(raw_text: str) -> Tuple[str, str]:
    """Extract possible title text from the beginning of the transcript."""
    title_parts: List[str] = []
    body_parts: List[str] = []
    seen_speaker_line = False

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        
        if SPEAKER_LINE_PATTERN.match(line):
            seen_speaker_line = True
            body_parts.append(line)
        elif not seen_speaker_line:
            title_parts.append(line)
        else:
            body_parts.append(line)
        
    title = " ".join(title_parts).strip() or "Untitled Meeting"
    body = "\n".join(body_parts)

    return title, body

def parse_meeting_text( raw_text: str, participants: List[Participant], title: Optional[str] = None ) -> MeetingInput:
        """
        Parse raw text into a MeetingInput.
        
        In the case of no title, we try to extract it from the beginning of the transcript.
        """
        if title is None:
            detected_title, body = split_title_from_transcript(raw_text)
            title = detected_title # Untitle Meeting already handled
        else:
            body = raw_text

        transcript_lines = parse_transcript_lines(body)

        return MeetingInput(
            title = title,
            participants = participants,
            lines = transcript_lines
        )

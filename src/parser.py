import re
from typing import List, Optional, Tuple

from src.schemas import MeetingInput, Participant, TranscriptLine

# Matches speaker lines
# E.g., Speaker 1: Hello, how are you?
#       Speaker 2: I'm doing well, thank you!
SPEAKER_LINE_PATTERN = re.compile(
    r"^\s*speaker\s*(\d+)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)


# Matches named lines
# E.g., Alice: Hello, how are you?
#       Bob: I'm doing well, thank you!
NAMED_LINE_PATTERN = re.compile(
    r"^\s*(.+?)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)

# Matches timestamps
# E.g., 00:00:00.000 --> 00:00:04.500
TIMESTAMP_PATTERN = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}$"
)

# Matches cue numbers
# E.g., 1, 2, 3, ...
CUE_NUMBER_PATTERN = re.compile(
    r"^\d+$"
)

def is_vtt_transcript(raw_text: str) -> bool:
    """
    Detect WebVTT-style transcripts (Zoom, Teams)

    Common examples:
    - file starts with WEBVTT or there's a timestamp line, usually within the
    first 30 lines
    """
    text = raw_text.strip()
    if text.startswith("WEBVTT"):
        return True
    
    for raw_line in raw_text.splitlines()[:30]:
        if bool(TIMESTAMP_PATTERN.match(raw_line)):
            return True

    return False

def _should_skip_vtt_metadata(line: str) -> bool:
    """
    Skip header and cue numbers but not dialogue
    """
    if not line or line == "WEBVTT" or CUE_NUMBER_PATTERN.match(line) or bool(TIMESTAMP_PATTERN.match(line)):
        return True
    else:
        return False

def _looks_like_speaker_name_line(line: str) -> bool:
    """
    Guess whether a line is a Teams speaker-name row vs spoken dialogue.

    Teams puts the display name alone on its own line, e.g. 'Alice McAliceFace'.
    Unlabeled VTT has dialogue immediately after the timestamp with no name row.

    Heuristics (not perfect):
    - Dialogue is usually a full sentence; names are short (we use 60 chars as a cutoff).
    - Dialogue often has . ? ! or , ; names typically do not.
    - Names are usually 1-5 words; dialogue is often longer.
    """
    if len(line) > 60:
        return False
    if any(ch in line for ch in ".?!,"):
        return False
    word_count = len(line.split())
    return 1 <= word_count <= 5

def parse_vtt_transcript_lines(raw_text: str) -> List[TranscriptLine]:
    """
    Parse Zoom or Teams VTT-style transcript into transcript lines.
    """
    lines: List[TranscriptLine] = []

    # for Teams format where the speaker name is the next line
    expect_speaker_name = False
    pending_speaker: Optional[str] = None

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()

        if _should_skip_vtt_metadata(line):
            # no vtt metadata, check for timestamp
            if bool(TIMESTAMP_PATTERN.match(line)):
                expect_speaker_name = True
                pending_speaker = None
            continue

        # check for speaker line, e.g., Speaker 1: Hello, how are you? 
        speaker_match = SPEAKER_LINE_PATTERN.match(line)
        if speaker_match:
            speaker_num, text = speaker_match.group(1), speaker_match.group(2)
            lines.append(
                TranscriptLine(
                    raw_label = f"Speaker { speaker_num }",
                    text = text,
                )
            )
            expect_speaker_name = False
            pending_speaker = None
            continue

        # check for named line, e.g., Alice: Hello, how are you?
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
            expect_speaker_name = False
            pending_speaker = None
            continue

        # after a timestamp: next line may be a Teams name OR unlabeled dialogue
        if expect_speaker_name:
            expect_speaker_name = False
            if _looks_like_speaker_name_line(line):
                # Teams, e.g. 'Alice McAliceFace' then dialogue on the following line
                pending_speaker = line
            else:
                # unlabeled VTT, e.g. timestamp then 'Hey Bob, how's the budget?'
                lines.append(
                    TranscriptLine(
                        raw_label = None,
                        text = line,
                    )
                )
            continue
        
        # handle dialogue after a Teams speaker name line
        if pending_speaker:
            lines.append(
                TranscriptLine(
                    raw_label = pending_speaker,
                    text = line,
                )
            )
            pending_speaker = None
            continue

        # if we have a line that does not have an explicit speaker, use the last speaker
        if lines:
            prev = lines[-1]
            lines[-1] = TranscriptLine(
                raw_label = prev.raw_label,
                text = f"{prev.text} {line}"
            )

    return lines


def parse_plain_transcript_lines(raw_text: str) -> List[TranscriptLine]:
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

def parse_transcript_lines(raw_text: str) -> List[TranscriptLine]:
    """
    Decide whether to parse as VTT or plain text and call the appropriate function.
    """
    if is_vtt_transcript(raw_text):
        return parse_vtt_transcript_lines(raw_text)
    else:
        return parse_plain_transcript_lines(raw_text)
    

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
            if is_vtt_transcript(raw_text):
                title = "Untitled Meeting"
                body = raw_text
            else:
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
